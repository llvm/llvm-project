//===- VectorUnrollInterface.h - Vector unrolling interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for vector ops that can be
// unrolled.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VECTORUNROLLINTERFACE_H
#define MLIR_INTERFACES_VECTORUNROLLINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

/// Include the generated interface declarations.
#include "mlir/Interfaces/VectorUnrollInterface.h.inc"

#endif // MLIR_INTERFACES_VECTORUNROLLINTERFACE_H
