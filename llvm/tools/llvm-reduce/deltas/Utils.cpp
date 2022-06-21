//===- Utils.cpp - llvm-reduce utility functions --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some utility functions supporting llvm-reduce.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

Value *llvm::getDefaultValue(Type *T) {
  return T->isVoidTy() ? PoisonValue::get(T) : Constant::getNullValue(T);
}
