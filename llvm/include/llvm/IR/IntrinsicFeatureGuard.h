//===- IntrinsicFeatureGuard.h - Intrinsic feature guard tables -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INTRINSICFEATUREGUARD_H
#define LLVM_IR_INTRINSICFEATUREGUARD_H

namespace llvm {

struct IntrinsicFeatureGuard {
  unsigned FirstClause;
  unsigned NumClauses;
  const char *Expression;
};

struct IntrinsicFeatureVariant {
  unsigned FirstType;
  unsigned NumTypes;
  unsigned FirstGuard;
  unsigned NumGuards;
  bool AppliesToAllTypes;
};

} // namespace llvm

#endif // LLVM_IR_INTRINSICFEATUREGUARD_H
