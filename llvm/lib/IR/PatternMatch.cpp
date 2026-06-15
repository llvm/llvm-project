//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Out-of-line implementations for PatternMatch.h.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Value.h"

using namespace llvm;

bool llvm::PatternMatch::undef_match::checkAggregate(
    const ConstantAggregate *CA) {
  SmallPtrSet<const ConstantAggregate *, 8> Seen;
  SmallVector<const ConstantAggregate *, 8> Worklist;

  // Either UndefValue, PoisonValue, or an aggregate that only contains
  // these is accepted by matcher.
  // CheckValue returns false if CA cannot satisfy this constraint.
  auto CheckValue = [&](const ConstantAggregate *CA) {
    for (const Value *Op : CA->operand_values()) {
      if (isa<UndefValue>(Op))
        continue;

      const auto *CA = dyn_cast<ConstantAggregate>(Op);
      if (!CA)
        return false;
      if (Seen.insert(CA).second)
        Worklist.emplace_back(CA);
    }

    return true;
  };

  if (!CheckValue(CA))
    return false;

  while (!Worklist.empty()) {
    if (!CheckValue(Worklist.pop_back_val()))
      return false;
  }
  return true;
}
