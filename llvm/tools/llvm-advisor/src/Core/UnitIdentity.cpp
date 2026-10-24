//===------------------- UnitIdentity.cpp - LLVM Advisor
//-------------------===//
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

#include "Core/UnitIdentity.h"
#include "Utils/Hashing.h"

using namespace llvm;
using namespace llvm::advisor;

std::string UnitIdentity::compute(const UnitRecord &Unit) const {
  return computeUnitID(Unit);
}
