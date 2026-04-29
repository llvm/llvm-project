//===------------------- CapabilityScheduler.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of CapabilityScheduler in Capability
//
//===----------------------------------------------------------------------===//

#include "Capability/CapabilityScheduler.h"

using namespace llvm;
using namespace llvm::advisor;

static unsigned costRank(StringRef Cost) {
  if (Cost == "cheap")
    return 0;
  if (Cost == "moderate")
    return 1;
  if (Cost == "expensive")
    return 2;
  // Unknown cost classes default to expensive so the scheduler stays
  // conservative even if capability specs introduce new values.
  return 2;
}

SmallVector<CapabilityNode, 16>
CapabilityScheduler::schedule(ArrayRef<CapabilityNode> Plan) const {
  SmallVector<CapabilityNode, 16> Out;
  BitVector Used(Plan.size());
  StringSet<> Done;

  while (Out.size() != Plan.size()) {
    int Best = -1;
    for (unsigned I = 0; I != Plan.size(); ++I) {
      if (Used[I])
        continue;
      bool Ready = true;
      for (const std::string &Input : Plan[I].Inputs) {
        if (!Done.contains(Input)) {
          Ready = false;
          break;
        }
      }
      if (!Ready)
        continue;
      if (Best < 0 || costRank(Plan[I].Spec.CostClass) <
                          costRank(Plan[Best].Spec.CostClass))
        Best = I;
    }

    assert(Best >= 0 && "capability scheduling failed: unsatisfied dependencies");
    Used[Best] = true;
    Done.insert(Plan[Best].Spec.ID);
    Out.push_back(Plan[Best]);
  }

  return Out;
}
