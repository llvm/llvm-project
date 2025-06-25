//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares FunctionUnitUnwindInfoAnalyzer class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNWINDINFOCHECKER_FUNCTIONUNITUNWINDINFOANALYZER_H
#define LLVM_UNWINDINFOCHECKER_FUNCTIONUNITUNWINDINFOANALYZER_H

#include "FunctionUnitAnalyzer.h"
#include "UnwindInfoAnalysis.h"
#include "llvm/ADT/ArrayRef.h"

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
