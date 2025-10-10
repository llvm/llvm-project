# RUN: %PYTHON %s | FileCheck %s

import ctypes

from mlir._mlir_libs._capi import (
    _capi,
    PyCapsule_New,
    MLIR_PYTHON_CAPSULE_DIALECT_HANDLE,
    register_dialect,
)
from mlir.ir import DialectHandle, DialectRegistry

print("success")
# CHECK: success


if not hasattr(_capi, "mlirGetDialectHandle__arith__"):
    raise Exception("missing API")
_capi.mlirGetDialectHandle__arith__.argtypes = []
_capi.mlirGetDialectHandle__arith__.restype = ctypes.c_void_p

if not hasattr(_capi, "mlirGetDialectHandle__quant__"):
    raise Exception("missing API")
_capi.mlirGetDialectHandle__quant__.argtypes = []
_capi.mlirGetDialectHandle__quant__.restype = ctypes.c_void_p

dialect_registry = DialectRegistry()
# CHECK: ['builtin']
print(dialect_registry.dialect_names)

arith_handle = _capi.mlirGetDialectHandle__arith__()
dialect_registry.insert_dialect(arith_handle)
# CHECK: ['arith', 'builtin']
print(sorted(dialect_registry.dialect_names))

quant_handle = _capi.mlirGetDialectHandle__quant__()
capsule = PyCapsule_New(quant_handle, MLIR_PYTHON_CAPSULE_DIALECT_HANDLE, None)
dialect_handle = DialectHandle._CAPICreate(capsule)
dialect_registry.insert_dialect(dialect_handle)
# CHECK: ['arith', 'builtin', 'quant']
print(sorted(dialect_registry.dialect_names))

register_dialect("math", dialect_registry)
# CHECK: ['arith', 'builtin', 'math', 'quant']
print(sorted(dialect_registry.dialect_names))
