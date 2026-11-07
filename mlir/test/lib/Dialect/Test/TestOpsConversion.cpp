//===- TestOpsConversion.cpp - Test dialect ops of TestOpsConversion.td ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOpsConversion.h"

using namespace mlir;
using namespace test;

#define GET_OP_CLASSES
#include "TestOpsConversion.cpp.inc"

void TestDialect::registerOpsConversion() {
  addOperations<
#define GET_OP_LIST
#include "TestOpsConversion.cpp.inc"
      >();
}
