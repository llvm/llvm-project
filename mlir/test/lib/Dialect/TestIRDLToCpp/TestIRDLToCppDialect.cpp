//===- TestDialect.cpp - MLIR Test Dialect Types ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file includes TestIRDLToCpp dialect.
//
//===----------------------------------------------------------------------===//

// #include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TestIRDLToCppDialect.h"

// #define GEN_DIALECT_DEF
// #include "test_irdl_to_cpp.irdl.mlir.cpp.inc"
// #undef GEN_DIALECT_DEF