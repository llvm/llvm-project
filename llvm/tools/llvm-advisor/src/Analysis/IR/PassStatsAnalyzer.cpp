//===--- PassStatsAnalyzer.cpp - LLVM Advisor ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/IR/PassStatsAnalyzer.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

using namespace llvm;
using namespace llvm::advisor;

namespace {
class CollectPassesListener final : public PassRegistrationListener {
public:
  void passEnumerate(const PassInfo *PI) override {
    if (!PI)
      return;
    ++Count;
  }
  int64_t Count = 0;
};
} // namespace

Expected<std::unique_ptr<CapabilityResult>>
PassStatsAnalyzer::run(const CapabilityContext &Context) {
  CollectPassesListener Listener;
  Listener.enumeratePasses();

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"registered_pass_count", Listener.Count}});
}
