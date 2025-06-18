//===- BufferizationTypeInterfaces.cpp - Type Interfaces --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h"

//===----------------------------------------------------------------------===//
// Bufferization Type Interfaces
//===----------------------------------------------------------------------===//

namespace mlir {
namespace bufferization {

#include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.cpp.inc"

} // namespace bufferization
} // namespace mlir
