//===- TuneExtensionOps.h - Tune ext. for Transform dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS_H
#define MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/TuneExtension/TuneExtensionOps.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_TUNEEXTENSION_TUNEEXTENSIONOPS_H
