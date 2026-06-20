//===------------------- PassListAnalyzer.cpp - LLVM Advisor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/PassListAnalyzer.h"

#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {
class CollectPasses final : public PassRegistrationListener {
public:
  void passEnumerate(const PassInfo *PI) override {
    if (!PI)
      return;
    json::Object Entry;
    Entry["name"] = PI->getPassName();
    Entry["arg"] = PI->getPassArgument();
    Passes.push_back(std::move(Entry));
  }

  json::Array Passes;
};
} // namespace

Expected<std::unique_ptr<CapabilityResult>>
PassListAnalyzer::run(const CapabilityContext &Context) {
  CollectPasses Listener;
  Listener.enumeratePasses();
  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"pass_count", static_cast<int64_t>(Listener.Passes.size())},
      {"passes", std::move(Listener.Passes)},
  });
}
