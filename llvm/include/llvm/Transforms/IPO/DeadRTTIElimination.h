#ifndef LLVM_TRANSFORMS_IPO_DEADRTTIELIMINATION_H
#define LLVM_TRANSFORMS_IPO_DEADRTTIELIMINATION_H

#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/LibCXXABI.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
class DeadRTTIElimIndex {
  ModuleSummaryIndex &ExportSummary;
  std::unique_ptr<CXXABI> ABI;

public:
  DeadRTTIElimIndex(ModuleSummaryIndex &ExportSummary, Triple &TT)
      : ExportSummary(ExportSummary), ABI(CXXABI::Create(TT)) {}

  void run();
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_DEADRTTIELIMINATION_H
