#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._nvgpu_transform_ops_gen import *

from ..._mlir_libs import get_dialect_registry as _get_dialect_registry
from ..._mlir_libs._capi import (
    register_transform_dialect_extension as _register_transform_dialect_extension,
)

_register_transform_dialect_extension(
    "mlirNVGPURegisterTransformDialectExtension", _get_dialect_registry()
)
