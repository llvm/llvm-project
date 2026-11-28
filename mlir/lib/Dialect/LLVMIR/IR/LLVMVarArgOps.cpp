//===- LLVMVarArgOps.cpp - LLVM dialect variadic argument ops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `VarArg` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMVarArgOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::LLVM;

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMVarArgOps.cpp.inc"
