#ifndef LLVM_MC_CFI_ANALYSIS_FUNCTION_UNIT_UNWIND_INFO_ANALYZER_H
#define LLVM_MC_CFI_ANALYSIS_FUNCTION_UNIT_UNWIND_INFO_ANALYZER_H

#include "FunctionUnitAnalyzer.h"
#include "UnwindInfoAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdio>

namespace llvm {

class FunctionUnitUnwindInfoAnalyzer : public FunctionUnitAnalyzer {
private:
  std::vector<UnwindInfoAnalysis> UIAs;
  MCInstrInfo const &MCII;

public:
  FunctionUnitUnwindInfoAnalyzer(MCContext &Context, const MCInstrInfo &MCII)
      : FunctionUnitAnalyzer(Context), MCII(MCII) {}

  void startFunctionUnit(bool IsEH,
                         ArrayRef<MCCFIInstruction> Prologue) override;
  void
  emitInstructionAndDirectives(const MCInst &Inst,
                               ArrayRef<MCCFIInstruction> Directives) override;
  void finishFunctionUnit() override;
};

} // namespace llvm

#endif