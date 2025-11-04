#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import platform
from pathlib import Path

from . import _mlir, get_dialect_registry as _get_dialect_registry

_get_dialect_registry()

MLIR_PYTHON_CAPSULE_DIALECT_HANDLE = _mlir.ir.DialectHandle._capsule_name.encode()

MLIR_PYTHON_CAPSULE_DIALECT_REGISTRY = _mlir.ir.DialectRegistry._capsule_name.encode()

if platform.system() == "Windows":
    _ext_suffix = "dll"
elif platform.system() == "Darwin":
    _ext_suffix = "dylib"
else:
    _ext_suffix = "so"

for fp in Path(__file__).parent.glob(f"*.{_ext_suffix}"):
    if "CAPI" in fp.name:
        _capi_dylib = fp
        break
else:
    raise ValueError("Couldn't find CAPI dylib")


_capi = ctypes.CDLL(str(Path(__file__).parent / _capi_dylib))

PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.restype = ctypes.py_object
PyCapsule_New.argtypes = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p

PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
PyCapsule_GetPointer.restype = ctypes.c_void_p


def register_dialect(dialect_name, dialect_registry):
    dialect_handle_capi = f"mlirGetDialectHandle__{dialect_name}__"
    if not hasattr(_capi, dialect_handle_capi):
        raise RuntimeError(f"missing {dialect_handle_capi} API")
    dialect_handle_capi = getattr(_capi, dialect_handle_capi)
    dialect_handle_capi.argtypes = []
    dialect_handle_capi.restype = ctypes.c_void_p
    handle = dialect_handle_capi()
    dialect_registry.insert_dialect(handle)
