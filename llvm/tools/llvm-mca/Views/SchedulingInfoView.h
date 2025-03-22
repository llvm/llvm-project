//===--------------------- SchedulingInfoView.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the instruction scheduling info view.
///
/// The goal fo the instruction scheduling info view is to print the latency,
/// latency with bypass,
//  throughput, pipeline ressources and uOps information for every instruction
//  in the input sequence.
///
/// Example:
///
/// Instruction Info:
/// [1]: #uOps
/// [2]: Latency
/// [3]: Bypass latency
/// [3]: Throughput
/// [4]: Resources
///
/// [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
///  1      4     1                 	        vmulps	%xmm0, %xmm1, %xmm2
///  1      3     1.00                    	vhaddps	%xmm2, %xmm2, %xmm3
///  1      3     1.00                    	vhaddps	%xmm3, %xmm3, %xmm4
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_SCHEDULINGINFOVIEW_H
#define LLVM_TOOLS_LLVM_MCA_SCHEDULINGINFOVIEW_H

#include "Views/InstructionView.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/CodeEmitter.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "llvm-mca"

namespace llvm {
namespace mca {

/// A view that prints out generic instruction information.
class SchedulingInfoView : public InstructionView {
  const llvm::MCInstrInfo &MCII;
  CodeEmitter &CE;
  using UniqueInst = std::unique_ptr<Instruction>;
  ArrayRef<UniqueInst> LoweredInsts;
  const InstrumentManager &IM;
  using InstToInstrumentsT =
      DenseMap<const MCInst *, SmallVector<mca::Instrument *>>;
  const InstToInstrumentsT &InstToInstruments;

  struct SchedulingInfoViewData {
    unsigned NumMicroOpcodes = 0;
    unsigned Latency = 0;
    unsigned Bypass = 0; // ReadAvance Bypasses cycles
    std::optional<double> Throughput = 0.0;
    std::string OpcodeName = "";
    std::string Resources = "";
  };
  using IIVDVec = SmallVector<SchedulingInfoViewData, 16>;

  /// Place the data into the array of SchedulingInfoViewData IIVD.
  void collectData(MutableArrayRef<SchedulingInfoViewData> IIVD) const;

public:
  SchedulingInfoView(const llvm::MCSubtargetInfo &ST,
                     const llvm::MCInstrInfo &II, CodeEmitter &C,
                     llvm::ArrayRef<llvm::MCInst> S, llvm::MCInstPrinter &IP,
                     ArrayRef<UniqueInst> LoweredInsts,
                     const InstrumentManager &IM,
                     const InstToInstrumentsT &InstToInstruments)
      : InstructionView(ST, IP, S), MCII(II), CE(C), LoweredInsts(LoweredInsts),
        IM(IM), InstToInstruments(InstToInstruments) {}

  /// Extract comment (//, /* */) from the source assembly placed just after
  /// instruction.
  void getComment(const llvm::MCInst &Inst, std::string &CommentString) const;
  void printView(llvm::raw_ostream &OS) const override;
  StringRef getNameAsString() const override { return "SchedulingInfoView"; }
  json::Value toJSON() const override;
  json::Object toJSON(const SchedulingInfoViewData &IIVD) const;
};
} // namespace mca
} // namespace llvm

#endif
