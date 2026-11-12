//===- TestOpsPatterns.cpp - Test dialect ops in partition `Patterns` -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOpsPatterns.h"
// The definitions of `Patterns` ops use the types in `Core`.
#include "TestTypesCore.h"

using namespace mlir;
using namespace test;

#define GET_OP_CLASSES
#include "TestOpsPatterns.cpp.inc"
