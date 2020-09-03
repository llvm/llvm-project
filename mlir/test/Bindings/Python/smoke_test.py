# RUN: %PYTHON %s | FileCheck %s

import mlir

# CHECK: From the native module
print(mlir.get_test_value())
