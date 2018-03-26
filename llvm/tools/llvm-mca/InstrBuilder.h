//===--------------------- InstrBuilder.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// A builder class for instructions that are statically analyzed by llvm-mca.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_INSTRBUILDER_H
#define LLVM_TOOLS_LLVM_MCA_INSTRBUILDER_H

#include "Instruction.h"
#include "Support.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace mca {

class DispatchUnit;

/// \brief A builder class that knows how to construct Instruction objects.
///
/// Every llvm-mca Instruction is described by an object of class InstrDesc.
/// An InstrDesc describes which registers are read/written by the instruction,
/// as well as the instruction latency and hardware resources consumed.
///
/// This class is used by the tool to construct Instructions and instruction
/// descriptors (i.e. InstrDesc objects).
/// Information from the machine scheduling model is used to identify processor
/// resources that are consumed by an instruction.
class InstrBuilder {
  const llvm::MCSubtargetInfo &STI;
  const llvm::MCInstrInfo &MCII;
  llvm::SmallVector<uint64_t, 8> ProcResourceMasks;

  llvm::DenseMap<unsigned short, std::unique_ptr<const InstrDesc>> Descriptors;

  void createInstrDescImpl(const llvm::MCInst &MCI);

public:
  InstrBuilder(const llvm::MCSubtargetInfo &sti, const llvm::MCInstrInfo &mcii)
      : STI(sti), MCII(mcii),
        ProcResourceMasks(STI.getSchedModel().getNumProcResourceKinds()) {
    computeProcResourceMasks(STI.getSchedModel(), ProcResourceMasks);
  }

  const InstrDesc &getOrCreateInstrDesc(const llvm::MCInst &MCI);
  // Returns an array of processor resource masks.
  // Masks are computed by function mca::computeProcResourceMasks. see
  // Support.h for a description of how masks are computed and how masks can be
  // used to solve set membership problems.
  llvm::ArrayRef<uint64_t> getProcResourceMasks() const {
    return ProcResourceMasks;
  }

  std::unique_ptr<Instruction> createInstruction(unsigned Idx,
                                                 const llvm::MCInst &MCI);
};
} // namespace mca

#endif
