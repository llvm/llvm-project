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
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"

namespace mca {

class DispatchUnit;

/// A builder class that knows how to construct Instruction objects.
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
  const llvm::MCRegisterInfo &MRI;
  const llvm::MCInstrAnalysis &MCIA;
  llvm::MCInstPrinter &MCIP;
  llvm::SmallVector<uint64_t, 8> ProcResourceMasks;

  llvm::DenseMap<unsigned short, std::unique_ptr<const InstrDesc>> Descriptors;
  llvm::DenseMap<const llvm::MCInst *, std::unique_ptr<const InstrDesc>>
      VariantDescriptors;

  llvm::Expected<const InstrDesc &>
  createInstrDescImpl(const llvm::MCInst &MCI);
  llvm::Expected<const InstrDesc &>
  getOrCreateInstrDesc(const llvm::MCInst &MCI);

  InstrBuilder(const InstrBuilder &) = delete;
  InstrBuilder &operator=(const InstrBuilder &) = delete;

  llvm::Error populateWrites(InstrDesc &ID, const llvm::MCInst &MCI,
                             unsigned SchedClassID);
  llvm::Error populateReads(InstrDesc &ID, const llvm::MCInst &MCI,
                            unsigned SchedClassID);

public:
  InstrBuilder(const llvm::MCSubtargetInfo &sti, const llvm::MCInstrInfo &mcii,
               const llvm::MCRegisterInfo &mri,
               const llvm::MCInstrAnalysis &mcia, llvm::MCInstPrinter &mcip)
      : STI(sti), MCII(mcii), MRI(mri), MCIA(mcia), MCIP(mcip),
        ProcResourceMasks(STI.getSchedModel().getNumProcResourceKinds()) {
    computeProcResourceMasks(STI.getSchedModel(), ProcResourceMasks);
  }

  // Returns an array of processor resource masks.
  // Masks are computed by function mca::computeProcResourceMasks. see
  // Support.h for a description of how masks are computed and how masks can be
  // used to solve set membership problems.
  llvm::ArrayRef<uint64_t> getProcResourceMasks() const {
    return ProcResourceMasks;
  }

  void clear() { VariantDescriptors.shrink_and_clear(); }

  llvm::Expected<std::unique_ptr<Instruction>>
  createInstruction(const llvm::MCInst &MCI);
};
} // namespace mca

#endif
