//===-- SCFLoopTransformOps.td -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry point of the Python bindings generator for the loop transform ops
// provided by the SCF (and other) dialects.
//
//===----------------------------------------------------------------------===//


#ifndef PYTHON_BINDINGS_SCF_LOOP_TRANSFORM_OPS
#define PYTHON_BINDINGS_SCF_LOOP_TRANSFORM_OPS

include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.td"
include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.td"

#endif // PYTHON_BINDINGS_SCF_LOOP_TRANSFORM_OPS
