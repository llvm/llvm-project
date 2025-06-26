//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares CFIFunctionFrameAnalyzer class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFCFICHECKER_DWARFCFIFUNCTIONFRAMEANALYZER_H
#define LLVM_DWARFCFICHECKER_DWARFCFIFUNCTIONFRAMEANALYZER_H

#include "DWARFCFIAnalysis.h"
#include "DWARFCFIFunctionFrameReceiver.h"
#include "llvm/ADT/ArrayRef.h"

namespace llvm {

class CFIFunctionFrameAnalyzer : public CFIFunctionFrameReceiver {
private:
  std::vector<DWARFCFIAnalysis> UIAs;
  MCInstrInfo const &MCII;

public:
  CFIFunctionFrameAnalyzer(MCContext &Context, const MCInstrInfo &MCII)
      : CFIFunctionFrameReceiver(Context), MCII(MCII) {}

  void startFunctionUnit(bool IsEH,
                         ArrayRef<MCCFIInstruction> Prologue) override;
  void
  emitInstructionAndDirectives(const MCInst &Inst,
                               ArrayRef<MCCFIInstruction> Directives) override;
  void finishFunctionUnit() override;
};

} // namespace llvm

#endif
