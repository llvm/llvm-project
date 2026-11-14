//===- GuardedIntrinsic.h - Guarded intrinsic TableGen records --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_GUARDEDINTRINSIC_H
#define LLVM_TABLEGEN_GUARDEDINTRINSIC_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/TableGen/Record.h"

namespace llvm {

struct GuardedIntrinsicVariant {
  // The GuardedVariant record that owns the feature guards.
  const Record *Guard;
  // Null when the guard applies to every valid type of the intrinsic.
  const ListInit *OverloadTypes;
};

inline SmallVector<const Record *>
getGuardedOverloadTypes(const ListInit *OverloadTypes) {
  SmallVector<const Record *> Types;
  for (const Init *Type : *OverloadTypes)
    Types.push_back(cast<DefInit>(Type)->getDef());
  return Types;
}

inline SmallVector<GuardedIntrinsicVariant>
collectGuardedIntrinsicVariants(const Record *Intrinsic) {
  SmallVector<GuardedIntrinsicVariant> Variants;
  for (const Record *Guard : Intrinsic->getValueAsListOfDefs("TargetGuards")) {
    const ListInit *Overloads = Guard->getValueAsListInit("Overloads");
    if (Overloads->empty())
      Variants.push_back({Guard, nullptr});
    else
      for (const Init *OverloadTypes : *Overloads)
        Variants.push_back({Guard, cast<ListInit>(OverloadTypes)});
  }
  return Variants;
}

} // namespace llvm

#endif // LLVM_TABLEGEN_GUARDEDINTRINSIC_H
