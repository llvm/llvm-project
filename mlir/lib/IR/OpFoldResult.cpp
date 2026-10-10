//===- OpFoldResult.cpp - Results of an operation fold --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpFoldResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// OpFoldResult
//===----------------------------------------------------------------------===//

void OpFoldResult::dump() const { llvm::errs() << *this << "\n"; }

raw_ostream &mlir::operator<<(raw_ostream &os, OpFoldResult ofr) {
  if (Value value = llvm::dyn_cast_if_present<Value>(ofr))
    value.print(os);
  else
    llvm::dyn_cast_if_present<Attribute>(ofr).print(os);
  return os;
}
