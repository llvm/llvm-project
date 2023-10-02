//===- MergeFunctionsIgnoringConst.h - Merge Functions ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers used in the MergeFunctionsIgnoringConst.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_MERGEFUNCTIONSIGNORINGCONST_H
#define LLVM_TRANSFORMS_UTILS_MERGEFUNCTIONSIGNORINGCONST_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

bool isEligibleInstrunctionForConstantSharing(const Instruction *I);

bool isEligibleOperandForConstantSharing(const Instruction *I, unsigned OpIdx);

bool isEligibleFunction(Function *F);

Value *createCast(IRBuilder<> &Builder, Value *V, Type *DestTy);
#endif // LLVM_TRANSFORMS_UTILS_MERGEFUNCTIONSIGNORINGCONST_H
