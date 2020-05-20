//===- AffineMemoryOpInterfaces.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for affine memory ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_AFFINEMEMORYOPINTERFACES_H_
#define MLIR_INTERFACES_AFFINEMEMORYOPINTERFACES_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h.inc"
} // namespace mlir

#endif // MLIR_INTERFACES_AFFINEMEMORYOPINTERFACES_H_
