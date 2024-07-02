//===- DynamicAPInt.cpp - DynamicAPInt Implementation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

hash_code llvm::hash_value(const DynamicAPInt &X) {
  if (X.isSmall())
    return llvm::hash_value(X.getSmall());
  return detail::hash_value(X.getLarge());
}

raw_ostream &DynamicAPInt::print(raw_ostream &OS) const {
  if (isSmall())
    return OS << ValSmall;
  return OS << ValLarge;
}

void DynamicAPInt::dump() const { print(dbgs()); }
