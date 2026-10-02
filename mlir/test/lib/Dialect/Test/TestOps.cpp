//===- TestOps.cpp - MLIR Test Dialect Operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"

// The fold tests keep a legacy fold trait to cover the deprecated trait form.
// Its warning fires inside OpDefinition.h, so the suppression must start
// before the includes.
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

#include "TestOps.h"
#include "TestDialect.h"
#include "TestFormatUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace test;

#include "TestOps.cpp.inc"

LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
