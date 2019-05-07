//===- DPUDisassembler.cpp - Disassembler for DPU ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the DPU Disassembler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUDISASSEMBLER_H
#define LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUDISASSEMBLER_H

#define GET_SUBTARGETINFO_ENUM
#include "DPUGenSubtargetInfo.inc"

#include "DPUInstructionDecoder.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {

class DPUDisassembler : public MCDisassembler {
private:
  bool useSugar;

public:
  DPUDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx),
        useSugar(!(STI.hasFeature(DPU::FeatureDPUNoSugar))) {}

  ~DPUDisassembler() override = default;

  // getInstruction - See MCDisassembler.
  MCDisassembler::DecodeStatus
  getInstruction(MCInst &Instr, uint64_t &Size, ArrayRef<uint8_t> Bytes,
                 uint64_t Address, raw_ostream &VStream,
                 raw_ostream &CStream) const override;
};

//===----------------------------------------------------------------------===//
// DPUSymbolizer
//===----------------------------------------------------------------------===//

class DPUSymbolizer : public MCSymbolizer {
private:
  void *DisInfo;

public:
  DPUSymbolizer(MCContext &Ctx, std::unique_ptr<MCRelocationInfo> &&RelInfo,
                void *disInfo)
      : MCSymbolizer(Ctx, std::move(RelInfo)), DisInfo(disInfo) {}

  bool tryAddingSymbolicOperand(MCInst &Inst, raw_ostream &cStream,
                                int64_t Value, uint64_t Address, bool IsBranch,
                                uint64_t Offset, uint64_t InstSize) override;

  void tryAddingPcLoadReferenceComment(raw_ostream &cStream, int64_t Value,
                                       uint64_t Address) override;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_DPU_DISASSEMBLER_DPUDISASSEMBLER_H
