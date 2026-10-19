//===-- AGXMCCodeEmitter.cpp - Convert AGX code to machine code -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The TableGen'erated encoder produces the long form of every instruction.
// Instructions with a short and a long form are emitted in the short form
// when all bits beyond it are zero; otherwise the length bit is set.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AGXFixupKinds.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

namespace {

class AGXMCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;
  MCContext &Ctx;

public:
  AGXMCCodeEmitter(const MCInstrInfo &MCII, MCContext &Ctx)
      : MCII(MCII), Ctx(Ctx) {}

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  // TableGen'erated.
  void getBinaryCodeForInstr(const MCInst &MI, SmallVectorImpl<MCFixup> &Fixups,
                             APInt &Inst, APInt &Scratch,
                             const MCSubtargetInfo &STI) const;

  void getMachineOpValue(const MCInst &MI, const MCOperand &MO, APInt &Op,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;

#define AGX_ENCODER(Kind)                                                      \
  void get##Kind##Encoding(const MCInst &MI, unsigned OpNo, APInt &Op,         \
                           SmallVectorImpl<MCFixup> &Fixups,                   \
                           const MCSubtargetInfo &STI) const;
  AGX_ENCODER(Src)
  AGX_ENCODER(Src64)
  AGX_ENCODER(Src16)
  AGX_ENCODER(SrcSX)
  AGX_ENCODER(SrcSX64)
  AGX_ENCODER(SrcF)
  AGX_ENCODER(SrcF16)
  AGX_ENCODER(CmpselSrc)
  AGX_ENCODER(Dst)
  AGX_ENCODER(MemReg)
  AGX_ENCODER(AtomicDst)
  AGX_ENCODER(MemBase)
  AGX_ENCODER(MemIndex)
  AGX_ENCODER(LocalBase)
  AGX_ENCODER(LocalIndex)
  AGX_ENCODER(AtomicSrc)
  AGX_ENCODER(BrTarget)
#undef AGX_ENCODER

private:
  AGX::RegInfo regInfo(const MCOperand &MO) const {
    return AGX::getRegInfo(MO.getReg(), *Ctx.getRegisterInfo());
  }
  /// The source field in the common layout: [5:0] value, [9:6] type and
  /// [11:10] extension.
  uint64_t encodeALUSrc(const MCInst &MI, unsigned OpNo) const;
  void report(const MCInst &MI, const Twine &Msg) const;
};

} // namespace

void AGXMCCodeEmitter::report(const MCInst &MI, const Twine &Msg) const {
  Ctx.reportError(MI.getLoc(), "cannot encode " +
                                   Twine(MCII.getName(MI.getOpcode())) + ": " +
                                   Msg);
}

uint64_t AGXMCCodeEmitter::encodeALUSrc(const MCInst &MI,
                                        unsigned OpNo) const {
  unsigned Mods = MI.getOperand(OpNo).getImm();
  const MCOperand &Val = MI.getOperand(OpNo + 1);
  if (Val.isImm()) {
    uint64_t V = uint64_t(Val.getImm());
    if (V > 0xff)
      report(MI, "ALU immediate out of range");
    V &= 0xff;
    return (V & 63) | ((V >> 6) << 10);
  }

  AGX::RegInfo Info = regInfo(Val);
  unsigned Idx = Info.HalfIndex;
  if (Info.isUniform()) {
    if (Info.Kind == AGX::RegKind::UGPR64)
      report(MI, "64-bit uniform registers cannot be ALU sources");
    bool Is32 = Info.Kind == AGX::RegKind::UGPR32;
    return (Idx & 63) | (((Idx >> 8) & 1) << 6) | (uint64_t(Is32) << 7) |
           (1u << 8) | (((Idx >> 6) & 3) << 10);
  }
  if (!Info.isGPR()) {
    report(MI, "invalid ALU source register");
    return 0;
  }
  unsigned Hint = (Mods & AGX::SRC_MOD_DISCARD) ? 3
                  : (Mods & AGX::SRC_MOD_CACHE) ? 2
                                                : 1;
  unsigned Size = Info.getSizeInBits();
  unsigned SizeFlag = Size == 64 ? 3 : Size == 32 ? 2 : 0;
  return (Idx & 63) | (Hint << 6) | (SizeFlag << 8) | (((Idx >> 6) & 3) << 10);
}

static void setOp(APInt &Op, uint64_t V) { Op = APInt(Op.getBitWidth(), V); }

void AGXMCCodeEmitter::getSrcEncoding(const MCInst &MI, unsigned OpNo,
                                      APInt &Op, SmallVectorImpl<MCFixup> &,
                                      const MCSubtargetInfo &) const {
  setOp(Op, encodeALUSrc(MI, OpNo));
}

void AGXMCCodeEmitter::getSrc64Encoding(const MCInst &MI, unsigned OpNo,
                                        APInt &Op, SmallVectorImpl<MCFixup> &,
                                        const MCSubtargetInfo &) const {
  setOp(Op, encodeALUSrc(MI, OpNo));
}

void AGXMCCodeEmitter::getSrc16Encoding(const MCInst &MI, unsigned OpNo,
                                        APInt &Op, SmallVectorImpl<MCFixup> &,
                                        const MCSubtargetInfo &) const {
  setOp(Op, encodeALUSrc(MI, OpNo));
}

void AGXMCCodeEmitter::getSrcSXEncoding(const MCInst &MI, unsigned OpNo,
                                        APInt &Op, SmallVectorImpl<MCFixup> &,
                                        const MCSubtargetInfo &) const {
  uint64_t M = encodeALUSrc(MI, OpNo);
  unsigned SX = (MI.getOperand(OpNo).getImm() & AGX::SRC_MOD_SX) ? 1 : 0;
  setOp(Op, (M & 0x3ff) | (SX << 10) | ((M >> 10) << 11));
}

void AGXMCCodeEmitter::getSrcSX64Encoding(const MCInst &MI, unsigned OpNo,
                                          APInt &Op,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
  getSrcSXEncoding(MI, OpNo, Op, Fixups, STI);
}

static unsigned getFPMods(unsigned Mods) {
  return ((Mods & AGX::SRC_MOD_ABS) ? 1 : 0) |
         ((Mods & AGX::SRC_MOD_NEG) ? 2 : 0);
}

void AGXMCCodeEmitter::getSrcFEncoding(const MCInst &MI, unsigned OpNo,
                                       APInt &Op, SmallVectorImpl<MCFixup> &,
                                       const MCSubtargetInfo &) const {
  uint64_t M = encodeALUSrc(MI, OpNo);
  unsigned FM = getFPMods(MI.getOperand(OpNo).getImm());
  setOp(Op, (M & 0x3ff) | (FM << 10) | ((M >> 10) << 12));
}

void AGXMCCodeEmitter::getSrcF16Encoding(const MCInst &MI, unsigned OpNo,
                                         APInt &Op, SmallVectorImpl<MCFixup> &,
                                         const MCSubtargetInfo &) const {
  uint64_t M = encodeALUSrc(MI, OpNo);
  if (M & (1u << 9))
    report(MI, "half precision sources must be 16-bit");
  unsigned FM = getFPMods(MI.getOperand(OpNo).getImm());
  setOp(Op, (M & 0x1ff) | (FM << 9) | ((M >> 10) << 11));
}

void AGXMCCodeEmitter::getCmpselSrcEncoding(const MCInst &MI, unsigned OpNo,
                                            APInt &Op,
                                            SmallVectorImpl<MCFixup> &,
                                            const MCSubtargetInfo &) const {
  unsigned Mods = MI.getOperand(OpNo).getImm();
  const MCOperand &Val = MI.getOperand(OpNo + 1);
  uint64_t V;
  if (Val.isImm()) {
    uint64_t I = uint64_t(Val.getImm()) & 0xff;
    V = (I & 63) | (0b100 << 6) | ((I >> 6) << 9);
  } else {
    AGX::RegInfo Info = regInfo(Val);
    unsigned Idx = Info.HalfIndex;
    if (Info.isUniform()) {
      V = (Idx & 63) | (((Idx >> 8) & 1) << 6) | (0b11 << 7) |
          (((Idx >> 6) & 3) << 9);
    } else {
      unsigned Hint = (Mods & AGX::SRC_MOD_DISCARD) ? 3
                      : (Mods & AGX::SRC_MOD_CACHE) ? 2
                                                    : 1;
      V = (Idx & 63) | (Hint << 6) | (((Idx >> 6) & 3) << 9);
    }
  }
  setOp(Op, V);
}

void AGXMCCodeEmitter::getDstEncoding(const MCInst &MI, unsigned OpNo,
                                      APInt &Op, SmallVectorImpl<MCFixup> &,
                                      const MCSubtargetInfo &) const {
  AGX::RegInfo Info = regInfo(MI.getOperand(OpNo));
  if (!Info.isGPR())
    report(MI, "invalid destination register");
  bool Cache = MI.getOperand(OpNo + 1).getImm();
  unsigned Size = Info.getSizeInBits();
  uint64_t D = uint64_t(Cache) | (uint64_t(Size >= 32) << 1) |
               (uint64_t(Info.HalfIndex | (Size == 64 ? 1 : 0)) << 2);
  setOp(Op, D);
}

void AGXMCCodeEmitter::getMemRegEncoding(const MCInst &MI, unsigned OpNo,
                                         APInt &Op, SmallVectorImpl<MCFixup> &,
                                         const MCSubtargetInfo &) const {
  AGX::RegInfo Info = regInfo(MI.getOperand(OpNo));
  if (!Info.isGPR())
    report(MI, "invalid memory register");
  bool Is32 = Info.Kind != AGX::RegKind::GPR16;
  setOp(Op, Info.HalfIndex | (uint64_t(Is32) << 8));
}

void AGXMCCodeEmitter::getAtomicDstEncoding(const MCInst &MI, unsigned OpNo,
                                            APInt &Op,
                                            SmallVectorImpl<MCFixup> &,
                                            const MCSubtargetInfo &) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (!MO.getReg()) {
    setOp(Op, 0);
    return;
  }
  AGX::RegInfo Info = regInfo(MO);
  if (Info.Kind != AGX::RegKind::GPR32)
    report(MI, "atomic results are 32-bit registers");
  setOp(Op, Info.HalfIndex | (1u << 8));
}

void AGXMCCodeEmitter::getMemBaseEncoding(const MCInst &MI, unsigned OpNo,
                                          APInt &Op, SmallVectorImpl<MCFixup> &,
                                          const MCSubtargetInfo &) const {
  AGX::RegInfo Info = regInfo(MI.getOperand(OpNo));
  if (Info.Kind != AGX::RegKind::GPR64 && Info.Kind != AGX::RegKind::UGPR64)
    report(MI, "memory base must be a 64-bit register pair");
  if (Info.HalfIndex >= 0x100)
    report(MI, "memory base register out of range");
  setOp(Op, (Info.HalfIndex & 0xff) | (uint64_t(Info.isUniform()) << 8));
}

void AGXMCCodeEmitter::getMemIndexEncoding(const MCInst &MI, unsigned OpNo,
                                           APInt &Op,
                                           SmallVectorImpl<MCFixup> &,
                                           const MCSubtargetInfo &) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isImm()) {
    int64_t V = MO.getImm();
    if (!isInt<16>(V) && !isUInt<16>(V))
      report(MI, "memory index immediate out of range");
    setOp(Op, (uint64_t(V) & 0xffff) | (1u << 16));
    return;
  }
  AGX::RegInfo Info = regInfo(MO);
  if (Info.Kind != AGX::RegKind::GPR32)
    report(MI, "memory index must be a 32-bit register");
  setOp(Op, Info.HalfIndex);
}

void AGXMCCodeEmitter::getLocalBaseEncoding(const MCInst &MI, unsigned OpNo,
                                            APInt &Op,
                                            SmallVectorImpl<MCFixup> &,
                                            const MCSubtargetInfo &) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isImm()) {
    if (MO.getImm() != 0)
      report(MI, "threadgroup base immediate must be zero");
    setOp(Op, 2u << 8);
    return;
  }
  AGX::RegInfo Info = regInfo(MO);
  if (Info.Kind == AGX::RegKind::UGPR16) {
    unsigned Idx = Info.HalfIndex;
    setOp(Op, (Idx & 0xff) | ((1u | ((Idx >> 8) << 1)) << 8));
    return;
  }
  if (Info.Kind != AGX::RegKind::GPR16)
    report(MI, "threadgroup base must be a 16-bit register");
  setOp(Op, Info.HalfIndex);
}

void AGXMCCodeEmitter::getLocalIndexEncoding(const MCInst &MI, unsigned OpNo,
                                             APInt &Op,
                                             SmallVectorImpl<MCFixup> &,
                                             const MCSubtargetInfo &) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isImm()) {
    setOp(Op, (uint64_t(MO.getImm()) & 0xffff) | (1u << 16));
    return;
  }
  AGX::RegInfo Info = regInfo(MO);
  if (Info.Kind != AGX::RegKind::GPR16)
    report(MI, "threadgroup index must be a 16-bit register");
  setOp(Op, Info.HalfIndex);
}

void AGXMCCodeEmitter::getAtomicSrcEncoding(const MCInst &MI, unsigned OpNo,
                                            APInt &Op,
                                            SmallVectorImpl<MCFixup> &,
                                            const MCSubtargetInfo &) const {
  AGX::RegInfo Info = regInfo(MI.getOperand(OpNo));
  if (Info.Kind != AGX::RegKind::GPR32 && Info.Kind != AGX::RegKind::GPR64)
    report(MI, "atomic data must be a 32-bit register or a pair");
  setOp(Op, Info.HalfIndex);
}

void AGXMCCodeEmitter::getBrTargetEncoding(const MCInst &MI, unsigned OpNo,
                                           APInt &Op,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isImm()) {
    setOp(Op, uint32_t(MO.getImm()));
    return;
  }
  // The offset field is 2 bytes into the instruction.
  Fixups.push_back(MCFixup::create(2, MO.getExpr(),
                                   MCFixupKind(AGX::fixup_agx_branch32),
                                   /*PCRel=*/true));
  setOp(Op, 0);
}

void AGXMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                         APInt &Op,
                                         SmallVectorImpl<MCFixup> &,
                                         const MCSubtargetInfo &) const {
  if (MO.isImm()) {
    setOp(Op, uint64_t(MO.getImm()));
    return;
  }
  if (MO.isReg()) {
    setOp(Op, regInfo(MO).HalfIndex);
    return;
  }
  report(MI, "unexpected operand");
  setOp(Op, 0);
}

void AGXMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                         SmallVectorImpl<char> &CB,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  APInt Inst(96, 0), Scratch(96, 0);
  getBinaryCodeForInstr(MI, Fixups, Inst, Scratch, STI);

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  unsigned Short = AGX::TSFlags::getShortSize(Desc.TSFlags);
  unsigned Long = AGX::TSFlags::getLongSize(Desc.TSFlags);
  unsigned Size = Long;
  if (Short != Long) {
    if (Inst.lshr(Short * 8).isZero())
      Size = Short;
    else
      Inst.setBit(AGX::TSFlags::getLengthBit(Desc.TSFlags));
  }

  for (unsigned I = 0; I < Size; ++I)
    CB.push_back(char(Inst.extractBitsAsZExtValue(8, I * 8)));
}

#include "AGXGenMCCodeEmitter.inc"

MCCodeEmitter *llvm::createAGXMCCodeEmitter(const MCInstrInfo &MCII,
                                            MCContext &Ctx) {
  return new AGXMCCodeEmitter(MCII, Ctx);
}
