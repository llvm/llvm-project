//===------------------- UnitIdentity.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes UnitID and CapabilityRunKey for build units.
// Uniquely identifies a compilation unit across builds.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"

namespace llvm::advisor {

class UnitIdentity {
public:
  std::string compute(const UnitRecord &Unit) const;
};

} // namespace llvm::advisor
