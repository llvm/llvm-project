//===- BufferViewFlowOpInterface.cpp - Buffer View Flow Analysis ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

namespace mlir {
namespace bufferization {

#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.cpp.inc"

} // namespace bufferization
} // namespace mlir
