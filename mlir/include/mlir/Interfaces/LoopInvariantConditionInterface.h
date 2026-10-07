//===- LoopInvariantConditionInterface.h - Cond. hoisting ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interface for if-like operations that can prove a
// region is unconditionally entered relative to an enclosing loop.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_LOOPINVARIANTCONDITIONINTERFACE_H_
#define MLIR_INTERFACES_LOOPINVARIANTCONDITIONINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

/// Include the generated interface declarations.
#include "mlir/Interfaces/LoopInvariantConditionInterface.h.inc"

#endif // MLIR_INTERFACES_LOOPINVARIANTCONDITIONINTERFACE_H_
