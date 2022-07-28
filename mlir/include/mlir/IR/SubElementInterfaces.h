//===- SubElementInterfaces.h - Attr and Type SubElements -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains interfaces and utilities for querying the sub elements of
// an attribute or type.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SUBELEMENTINTERFACES_H
#define MLIR_IR_SUBELEMENTINTERFACES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
template <typename T>
using SubElementReplFn = function_ref<T(T)>;
template <typename T>
using SubElementResultReplFn = function_ref<std::pair<T, WalkResult>(T)>;
} // namespace mlir

/// Include the definitions of the sub elemnt interfaces.
#include "mlir/IR/SubElementAttrInterfaces.h.inc"
#include "mlir/IR/SubElementTypeInterfaces.h.inc"

#endif // MLIR_IR_SUBELEMENTINTERFACES_H
