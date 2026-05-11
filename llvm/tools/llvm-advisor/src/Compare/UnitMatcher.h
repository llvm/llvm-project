//===------------------- UnitMatcher.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of UnitMatcher in Compare
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"

namespace llvm::advisor {

class UnitMatcher {
public:
  bool exact(const UnitRecord &LHS, const UnitRecord &RHS) const {
    return LHS.ID == RHS.ID;
  }
};

} // namespace llvm::advisor
