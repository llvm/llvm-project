//===---------------------- CustomBehaviour.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the base class CustomBehaviour which can be inherited from
/// by specific targets (ex. llvm/tools/llvm-mca/lib/X86CustomBehaviour.h).
/// CustomBehaviour is designed to enforce custom behaviour and dependencies
/// within the llvm-mca pipeline simulation that llvm-mca isn't already capable
/// of extracting from the Scheduling Models.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCA_CUSTOMBEHAVIOUR_H
#define LLVM_MCA_CUSTOMBEHAVIOUR_H

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/SourceMgr.h"

namespace llvm {
namespace mca {

/// Class which can be overriden by targets to modify the
/// mca::Instruction objects before the pipeline starts.
/// A common usage of this class is to add immediate operands to certain
/// instructions or to remove Defs/Uses from an instruction where the
/// schedulinng model is incorrect.
class InstrPostProcess {
protected:
  const MCSubtargetInfo &STI;
  const MCInstrInfo &MCII;

public:
  InstrPostProcess(const MCSubtargetInfo &STI, const MCInstrInfo &MCII)
      : STI(STI), MCII(MCII) {}

  virtual ~InstrPostProcess() {}

  virtual void postProcessInstruction(std::unique_ptr<Instruction> &Inst,
                                      const MCInst &MCI) {}
};

/// Class which can be overriden by targets to enforce instruction
/// dependencies and behaviours that aren't expressed well enough
/// within the scheduling model for mca to automatically simulate
/// them properly.
/// If you implement this class for your target, make sure to also implement
/// a target specific InstrPostProcess class as well.
class CustomBehaviour {
protected:
  const MCSubtargetInfo &STI;
  const mca::SourceMgr &SrcMgr;
  const MCInstrInfo &MCII;

public:
  CustomBehaviour(const MCSubtargetInfo &STI, const mca::SourceMgr &SrcMgr,
                  const MCInstrInfo &MCII)
      : STI(STI), SrcMgr(SrcMgr), MCII(MCII) {}

  virtual ~CustomBehaviour();

  // Before the llvm-mca pipeline dispatches an instruction, it first checks
  // for any register or resource dependencies / hazards. If it doesn't find
  // any, this method will be invoked to determine if there are any custom
  // hazards that the instruction needs to wait for.
  // The return value of this method is the number of cycles that the
  // instruction needs to wait for.
  // It's safe to underestimate the number of cycles to wait for since these
  // checks will be invoked again before the intruction gets dispatched.
  // However, it's not safe (accurate) to overestimate the number of cycles
  // to wait for since the instruction will wait for AT LEAST that number of
  // cycles before attempting to be dispatched again.
  virtual unsigned checkCustomHazard(ArrayRef<InstRef> IssuedInst,
                                     const InstRef &IR);
};

} // namespace mca
} // namespace llvm

#endif /* LLVM_MCA_CUSTOMBEHAVIOUR_H */
