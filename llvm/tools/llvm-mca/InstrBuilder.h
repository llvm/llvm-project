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
  const llvm::MCInstrInfo &MCII;
  const llvm::ArrayRef<uint64_t> ProcResourceMasks;

  llvm::DenseMap<unsigned short, std::unique_ptr<const InstrDesc>> Descriptors;
  llvm::DenseMap<unsigned, std::unique_ptr<Instruction>> Instructions;

  void createInstrDescImpl(const llvm::MCSubtargetInfo &STI,
                           const llvm::MCInst &MCI);

public:
  InstrBuilder(const llvm::MCInstrInfo &mcii,
               const llvm::ArrayRef<uint64_t> Masks)
      : MCII(mcii), ProcResourceMasks(Masks) {}

  const InstrDesc &getOrCreateInstrDesc(const llvm::MCSubtargetInfo &STI,
                                        const llvm::MCInst &MCI);

  Instruction *createInstruction(const llvm::MCSubtargetInfo &STI,
                                 unsigned Idx,
                                 const llvm::MCInst &MCI);
};

} // namespace mca

#endif
