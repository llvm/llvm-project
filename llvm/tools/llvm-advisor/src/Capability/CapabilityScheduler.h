//===------------------- CapabilityScheduler.h - LLVM Advisor -------------------===//
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

#pragma once

#include "AdvisorCommon.h"
#include "Capability/CapabilitySpec.h"
#include "llvm/ADT/BitVector.h"

namespace llvm::advisor {

class CapabilityScheduler {
public:
  SmallVector<CapabilityNode, 16> schedule(ArrayRef<CapabilityNode> Plan) const;
};

} // namespace llvm::advisor
