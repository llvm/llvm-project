//===------------------- CapabilityPlanner.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of CapabilityPlanner in Capability
//
//===----------------------------------------------------------------------===//

#include "Capability/CapabilityPlanner.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<SmallVector<CapabilityNode, 16>>
CapabilityPlanner::plan(ArrayRef<std::string> CapabilityIDs) const {
  SmallVector<CapabilityNode, 16> Plan;
  StringSet<> Visiting;
  StringSet<> Visited;
  for (const std::string &ID : CapabilityIDs) {
    if (Error Err = visit(ID, Visiting, Visited, Plan))
      return std::move(Err);
  }
  return Plan;
}

Error CapabilityPlanner::visit(StringRef ID, StringSet<> &Visiting,
                               StringSet<> &Visited,
                               SmallVectorImpl<CapabilityNode> &Plan) const {
  if (Visited.contains(ID))
    return Error::success();
  if (Visiting.contains(ID))
    return createStringError(inconvertibleErrorCode(), "capability cycle at %s",
                             ID.str().c_str());

  Expected<CapabilitySpec> Spec = Registry.getSpec(ID);
  if (!Spec)
    return Spec.takeError();

  Visiting.insert(ID);
  for (const std::string &Dep : Spec->Dependencies) {
    if (Error Err = visit(Dep, Visiting, Visited, Plan))
      return Err;
  }
  Visiting.erase(ID);
  Visited.insert(ID);

  CapabilityNode Node;
  Node.Inputs = Spec->Dependencies;
  Node.Spec = std::move(*Spec);
  Plan.push_back(std::move(Node));
  return Error::success();
}
