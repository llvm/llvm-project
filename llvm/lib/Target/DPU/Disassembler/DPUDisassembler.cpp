//===-- DPUDisassembler.cpp - Disassembler for DPU ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DPUDisassembler class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "DPU-disassembler"

#include "DPUDisassembler.h"
#include "MCTargetDesc/DPUMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

using namespace llvm;
namespace llvm {
extern Target TheDPUTarget;
}

typedef MCDisassembler::DecodeStatus DecodeStatus;

static DecodeStatus readInstruction48(ArrayRef<uint8_t> Bytes, uint64_t &Size,
                                      uint64_t &Insn) {
  // We want to read exactly 8 bytes of data.
  if (Bytes.size() < 8) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  // Receive a stream of little-endian words of 64 bytes.
  // Two MSBytes are not significant.
  Insn = (((uint64_t)Bytes[0]) << 0) | (((uint64_t)Bytes[1]) << 8) |
         (((uint64_t)Bytes[2]) << 16) | (((uint64_t)Bytes[3]) << 24) |
         (((uint64_t)Bytes[4]) << 32) | (((uint64_t)Bytes[5]) << 40);

  return MCDisassembler::Success;
}

DecodeStatus DPUDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                             ArrayRef<uint8_t> Bytes,
                                             uint64_t Address, raw_ostream &OS,
                                             raw_ostream &CS) const {
  uint64_t Insn;
  DPUInstructionDecoder instructionDecoder;
  DPUOperandDecoder operandDecoder(Address, this);

  DecodeStatus Result = readInstruction48(Bytes, Size, Insn);

  if (Result == MCDisassembler::Fail)
    return MCDisassembler::Fail;

  // Call auto-generated decoder function
  DecodeStatus status =
      instructionDecoder.getInstruction(MI, Insn, operandDecoder, useSugar);
  Size = 8;
  return status;
}

//===----------------------------------------------------------------------===//
// DPUSymbolizer
//===----------------------------------------------------------------------===//

// Try to find symbol name for specified label
bool DPUSymbolizer::tryAddingSymbolicOperand(MCInst &Inst,
                                             raw_ostream & /*cStream*/,
                                             int64_t Value,
                                             uint64_t /*Address*/,
                                             bool IsBranch, uint64_t /*Offset*/,
                                             uint64_t /*InstSize*/) {
  using SymbolInfoTy = std::tuple<uint64_t, StringRef, uint8_t>;
  using SectionSymbolsTy = std::vector<SymbolInfoTy>;

  if (!IsBranch) {
    return false;
  }

  auto *Symbols = static_cast<SectionSymbolsTy *>(DisInfo);
  if (!Symbols)
    return false;

  auto Result = std::find_if(
      Symbols->begin(), Symbols->end(), [Value](const SymbolInfoTy &Val) {
        return std::get<0>(Val) == static_cast<uint64_t>(Value);
      });
  if (Result != Symbols->end()) {
    auto *Sym = Ctx.getOrCreateSymbol(std::get<1>(*Result));
    const auto *Add = MCSymbolRefExpr::create(Sym, Ctx);
    Inst.addOperand(MCOperand::createExpr(Add));
    return true;
  }
  return false;
}

void DPUSymbolizer::tryAddingPcLoadReferenceComment(raw_ostream &cStream,
                                                    int64_t Value,
                                                    uint64_t Address) {
  llvm_unreachable("unimplemented");
}

static MCDisassembler *createDPUDisassembler(const Target &T,
                                             const MCSubtargetInfo &STI,
                                             MCContext &Ctx) {
  return new DPUDisassembler(STI, Ctx);
}

static MCSymbolizer *
createDPUSymbolizer(const Triple & /*TT*/, LLVMOpInfoCallback /*GetOpInfo*/,
                    LLVMSymbolLookupCallback /*SymbolLookUp*/, void *DisInfo,
                    MCContext *Ctx,
                    std::unique_ptr<MCRelocationInfo> &&RelInfo) {
  return new DPUSymbolizer(*Ctx, std::move(RelInfo), DisInfo);
}

extern "C" void LLVMInitializeDPUDisassembler() {
  TargetRegistry::RegisterMCDisassembler(TheDPUTarget, createDPUDisassembler);
  TargetRegistry::RegisterMCSymbolizer(TheDPUTarget, createDPUSymbolizer);
}
