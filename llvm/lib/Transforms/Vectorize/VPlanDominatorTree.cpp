//===-- VPlanDominatorTree.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VPlanDominatorTree.h"
#include "llvm/Analysis/DominanceFrontierImpl.h"

using namespace llvm;

VPPostDominanceFrontier::VPPostDominanceFrontier(const DomTreeT &VPDT) {
  analyze(VPDT);
}
