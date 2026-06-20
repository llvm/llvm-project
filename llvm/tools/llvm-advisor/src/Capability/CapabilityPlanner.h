//===------------------- CapabilityPlanner.h - LLVM Advisor -------------------===//
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

#pragma once

#include "AdvisorCommon.h"
#include "Capability/CapabilityRegistry.h"

namespace llvm::advisor {

class CapabilityPlanner {
public:
  explicit CapabilityPlanner(const CapabilityRegistry &Registry)
      : Registry(Registry) {}

  Expected<SmallVector<CapabilityNode, 16>>
  plan(ArrayRef<std::string> CapabilityIDs) const;

private:
  Error visit(StringRef ID, StringSet<> &Visiting, StringSet<> &Visited,
              SmallVectorImpl<CapabilityNode> &Plan) const;

  const CapabilityRegistry &Registry;
};

} // namespace llvm::advisor
