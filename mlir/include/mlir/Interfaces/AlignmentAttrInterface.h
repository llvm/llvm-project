//===- AlignmentAttrInterface.h - Alignment attribute interface -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_ALIGNMENTATTRINTERFACE_H
#define MLIR_INTERFACES_ALIGNMENTATTRINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/Alignment.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

#include "mlir/Interfaces/AlignmentAttrInterface.h.inc"

#endif // MLIR_INTERFACES_ALIGNMENTATTRINTERFACE_H
