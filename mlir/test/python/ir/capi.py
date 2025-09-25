# RUN: %PYTHON %s | FileCheck %s

from mlir._mlir_libs._capi import _capi

print("success")
# CHECK: success