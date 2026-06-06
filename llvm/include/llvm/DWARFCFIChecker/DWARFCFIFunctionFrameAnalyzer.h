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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// This class implements the `CFIFunctionFrameReceiver` interface to validate
/// Call Frame Information in a stream of function frames. For validation, it
/// instantiates a `DWARFCFIAnalysis` for each frame. The errors/warnings are
/// emitted through the `MCContext` instance to the constructor. If a frame
/// finishes without being started or if all the frames are not finished before
/// this classes is destructed, the program fails through an assertion.
class LLVM_ABI CFIFunctionFrameAnalyzer : public CFIFunctionFrameReceiver {
public:
  CFIFunctionFrameAnalyzer(MCContext &Context, const MCInstrInfo &MCII)
      : CFIFunctionFrameReceiver(Context), MCII(MCII) {}
  ~CFIFunctionFrameAnalyzer() override;

  void startFunctionFrame(bool IsEH,
                          ArrayRef<MCCFIInstruction> Prologue) override;
  void
  emitInstructionAndDirectives(const MCInst &Inst,
                               ArrayRef<MCCFIInstruction> Directives) override;
  void finishFunctionFrame() override;

private:
  MCInstrInfo const &MCII;
  SmallVector<DWARFCFIAnalysis> UIAs;
};

} // namespace llvm

#endif
