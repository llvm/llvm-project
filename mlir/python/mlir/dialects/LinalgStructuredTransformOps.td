//===-- LinalgStructuredTransformOps.td --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry point of the Python bindings generator for the structured transform ops
// provided by Linalg (and other dialects).
//
//===----------------------------------------------------------------------===//


#ifndef PYTHON_BINDINGS_LINALG_STRUCTURED_TRANSFORM_OPS
#define PYTHON_BINDINGS_LINALG_STRUCTURED_TRANSFORM_OPS

include "mlir/Bindings/Python/Attributes.td"
include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td"

#endif // PYTHON_BINDINGS_LINALG_STRUCTURED_TRANSFORM_OPS
