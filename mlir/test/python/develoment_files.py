# RUN: %PYTHON %s 2>&1

import os

from mlir._mlir_libs import get_include_dirs, get_lib_dirs


header_file = os.path.join(get_include_dirs()[0], "mlir-c", "IR.h")
assert os.path.isfile(header_file), f"Header does not exist: {header_file}"

# Since actual library names are platform specific, just scan the directory
# for a filename that contains the library name.
expected_lib_name = "MLIRPythonCAPI"
all_libs = os.listdir(get_lib_dirs()[0])
found_lib = False
for file_name in all_libs:
    if expected_lib_name in file_name:
        found_lib = True
assert found_lib, f"Did not find '{expected_lib_name}' lib in {all_libs}"
