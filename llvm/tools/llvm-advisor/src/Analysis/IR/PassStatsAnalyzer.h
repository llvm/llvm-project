#pragma once
#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {
class PassStatsAnalyzer final : public CapabilityRunner {
public:
  StringRef getCapabilityID() const override { return "llvm.pass.stats"; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;
};
} // namespace llvm::advisor
