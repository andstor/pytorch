import os
import torch
from torchgen.gen import parse_native_yaml, get_grouped_native_functions, get_grouped_by_view_native_functions
from torchgen.model import Argument, FunctionSchema, NativeFunction, NativeFunctionsGroup, NativeFunctionsViewGroup

from typing import Set, List, Dict

try:
    from tabulate import tabulate
except ImportError:
    print("`print_tabular` relies on the library `tabulate`, "
            "which could not be found on this machine. Run `pip "
            "install tabulate` to install the library.")

class Dialect:
    def __init__(self, name):
        self.name = name
        self.ops: Dict[str, NativeFunction] = {}

        parsed_yaml = parse_native_yaml("aten/src/ATen/native/native_functions.yaml", "aten/src/ATen/native/tags.yaml")

        native_functions, backend_indices = (
            parsed_yaml.native_functions,
            parsed_yaml.backend_indices,
        )

        for function in native_functions:
            if "canonical" in function.tags:
                self.ops[function.func.name] = function

        grouped_native_functions = get_grouped_native_functions(native_functions)
        self.structured_native_functions: Dict[str, NativeFunctionsGroup] = {
            g.functional.func.name: g for g in grouped_native_functions
                if isinstance(g, NativeFunctionsGroup) and g.functional.func.name in self.ops
        }

        native_functions_with_view_groups = get_grouped_by_view_native_functions(
            native_functions
        )
        self.view_groups: Dict[str, NativeFunctionsViewGroup] = {
            g.view.func.name: g for g in native_functions_with_view_groups
                if isinstance(g, NativeFunctionsViewGroup) and g.view.func.name in self.ops
        }

        # assert len(self.ops) == len(self.structured_native_functions) + len(self.view_groups), \
        #     f"len(self.ops) = {len(self.ops)}, \
        #       len(self.structured_native_functions) = {len(self.structured_native_functions)}, \
        #       len(self.view_groups) = {len(self.view_groups)}"

        self._is_functional = True
        for op in self.ops.values():
            if not op.func.is_functional_fn():
                self._is_functional = False

    def print_tabular(self):
        header = ['functional', 'out', 'inplace']
        table = [[g.functional.func.name,
                  g.out.func.name if g.out else 'N/A',
                  g.inplace.func.name if g.inplace else 'N/A']
                    for g in self.structured_native_functions.values()]
        print(tabulate(table, headers=header))
        print()

        header = ['view', 'view_copy', 'view_inplace']
        table = [[g.view.func.name,
                  g.view_copy.func.name if g.view_copy else 'N/A',
                  g.view_inplace.func.name if g.view_inplace else 'N/A']
                    for g in self.view_groups.values()]
        print(tabulate(table, headers=header))

    def print_schema(self):
        header = ['op', 'schema', 'aliased_return_names']

        table = []
        for op in self.ops.values():
            aliased_return_names = op.func.aliased_return_names()
            if all([alised_name is None for alised_name in aliased_return_names]):
                aliased_return_names = 'N/A'
            table.append([op.func.name, op.func.signature(), aliased_return_names])

        print(tabulate(table, headers=header))

    @property
    def is_functional(self) -> bool:
        return self._is_functional

    def infer_shape(self, op, input_shapes):
        pass

    def infer_dtype(self, op, input_dtypes):
        pass



d = Dialect("aten")

d.print_tabular()
d.print_schema()
print(d.is_functional)

# print(backend_indices)