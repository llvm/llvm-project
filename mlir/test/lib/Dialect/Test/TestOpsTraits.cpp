//===- TestOpsTraits.cpp - Test dialect ops of TestOpsTraits.td -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOpsTraits.h"

using namespace mlir;
using namespace test;

#define GET_OP_CLASSES
#include "TestOpsTraits.cpp.inc"

void TestDialect::registerOpsTraits() {
  addOperations<
#define GET_OP_LIST
#include "TestOpsTraits.cpp.inc"
      >();
}
