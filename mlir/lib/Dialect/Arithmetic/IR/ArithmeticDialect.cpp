//===- ArithmeticDialect.cpp - MLIR Arithmetic dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::arith;

#include "mlir/Dialect/Arithmetic/IR/ArithmeticOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining for arithmetic
/// dialect operations.
struct ArithmeticInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All arithmetic dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

void mlir::arith::ArithmeticDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.cpp.inc"
      >();
  addInterfaces<ArithmeticInlinerInterface>();
}
