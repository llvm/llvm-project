//===- TestOpsCore.cpp - Test dialect ops of TestOpsCore.td ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOpsCore.h"
#include "TestFormatUtils.h"
// Some implicit terminators of `Core` ops are in other TestOps<Category>.td
// files.
#include "TestOpsInterfaces.h"
#include "TestOpsTraits.h"

using namespace mlir;
using namespace test;

#define GET_OP_CLASSES
#include "TestOpsCore.cpp.inc"

void TestDialect::registerOpsCore() {
  addOperations<
#define GET_OP_LIST
#include "TestOpsCore.cpp.inc"
      >();
}
