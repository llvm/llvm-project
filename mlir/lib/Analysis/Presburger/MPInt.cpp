//===- MPInt.cpp - MLIR MPInt Class ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/MPInt.h"
#include "mlir/Analysis/Presburger/SlowMPInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace presburger;

llvm::hash_code mlir::presburger::hash_value(const MPInt &x) {
  if (x.isSmall())
    return llvm::hash_value(x.getSmall());
  return detail::hash_value(x.getLarge());
}

/// ---------------------------------------------------------------------------
/// Printing.
/// ---------------------------------------------------------------------------
llvm::raw_ostream &MPInt::print(llvm::raw_ostream &os) const {
  if (isSmall())
    return os << valSmall;
  return os << valLarge;
}

void MPInt::dump() const { print(llvm::errs()); }

llvm::raw_ostream &mlir::presburger::operator<<(llvm::raw_ostream &os,
                                                const MPInt &x) {
  x.print(os);
  return os;
}
