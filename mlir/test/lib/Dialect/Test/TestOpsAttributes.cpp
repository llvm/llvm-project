//===- TestOpsAttributes.cpp - Test dialect ops in partition `Attributes` -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOpsAttributes.h"
#include "TestFormatUtils.h"
// The definitions of `Attributes` ops use the type interfaces of `Core`.
#include "TestTypesCore.h"
// The generated op definitions parse prop-dicts with `FieldParser`.
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace test;

#define GET_OP_CLASSES
#include "TestOpsAttributes.cpp.inc"
