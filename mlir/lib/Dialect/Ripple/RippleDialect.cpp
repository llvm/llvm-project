//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// The Ripple dialect implementation.
//
//==============================================================================

#include "mlir/Dialect/Ripple/Ripple.h"

using namespace mlir;
using namespace mlir::ripple;

#include "mlir/Dialect/Ripple/RippleOpsDialect.cpp.inc"

void mlir::ripple::RippleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Ripple/RippleOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Ripple/RippleOps.cpp.inc"
