//===--- Assumptions.h - Assumption handling and organization ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// String assumptions that are known to optimization passes should be placed in
// the KnownAssumptionStrings set. This can be done in various ways, i.a.,
// via a static KnownAssumptionString object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ASSUMPTIONS_H
#define LLVM_IR_ASSUMPTIONS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace llvm {

class Function;

/// The key we use for assumption attributes.
constexpr StringRef AssumptionAttrKey = "llvm.assume";

/// A set of known assumption strings that are accepted without warning and
/// which can be recommended as typo correction.
extern StringSet<> KnownAssumptionStrings;

/// Helper that allows to insert a new assumption string in the known assumption
/// set by creating a (static) object.
struct KnownAssumptionString {
  KnownAssumptionString(StringRef AssumptionStr)
      : AssumptionStr(AssumptionStr) {
    KnownAssumptionStrings.insert(AssumptionStr);
  }
  operator StringRef() const { return AssumptionStr; }

private:
  StringRef AssumptionStr;
};

/// Return true if \p F has the assumption \p AssumptionStr attached.
bool hasAssumption(Function &F, const KnownAssumptionString &AssumptionStr);

} // namespace llvm

#endif
