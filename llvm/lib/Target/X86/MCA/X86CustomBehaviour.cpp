//===------------------- X86CustomBehaviour.cpp -----------------*-C++ -* -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods from the X86CustomBehaviour class.
///
//===----------------------------------------------------------------------===//

#include "X86CustomBehaviour.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "TargetInfo/X86TargetInfo.h"
#include "llvm-c/Visibility.h"
#include "llvm/MC/TargetRegistry.h"

namespace llvm {
namespace mca {

namespace {

bool isStackPointerReg(MCRegister Reg) {
  return Reg == X86::RSP || Reg == X86::ESP || Reg == X86::SP;
}

bool isFramePointerReg(MCRegister Reg) {
  return Reg == X86::RBP || Reg == X86::EBP || Reg == X86::BP;
}

unsigned getRegClassMemSize(int16_t RegClass) {
  switch (RegClass) {
  case X86::GR8RegClassID:
  case X86::GRH8RegClassID:
  case X86::GR8_ABCD_LRegClassID:
  case X86::GR8_ABCD_HRegClassID:
  case X86::GR8_NOREXRegClassID:
  case X86::GR8_NOREX2RegClassID:
    return 1;
  case X86::GR16RegClassID:
  case X86::GRH16RegClassID:
  case X86::GR16_ABCDRegClassID:
  case X86::GR16_NOREXRegClassID:
  case X86::GR16_NOREX2RegClassID:
  case X86::FR16RegClassID:
  case X86::FR16XRegClassID:
    return 2;
  case X86::GR32RegClassID:
  case X86::GR32_ABCDRegClassID:
  case X86::GR32_NOREXRegClassID:
  case X86::GR32_NOREX2RegClassID:
  case X86::GR32_NOSPRegClassID:
  case X86::GR32_NOREX_NOSPRegClassID:
  case X86::GR32_NOREX2_NOSPRegClassID:
  case X86::FR32RegClassID:
  case X86::FR32XRegClassID:
    return 4;
  case X86::GR64RegClassID:
  case X86::GR64_ABCDRegClassID:
  case X86::GR64_NOREXRegClassID:
  case X86::GR64_NOREX2RegClassID:
  case X86::GR64_NOSPRegClassID:
  case X86::GR64_NOREX_NOSPRegClassID:
  case X86::GR64_NOREX2_NOSPRegClassID:
  case X86::FR64RegClassID:
  case X86::FR64XRegClassID:
  case X86::VR64RegClassID:
    return 8;
  case X86::VR128RegClassID:
  case X86::VR128XRegClassID:
    return 16;
  case X86::VR256RegClassID:
  case X86::VR256XRegClassID:
    return 32;
  case X86::VR512RegClassID:
  case X86::VR512_0_15RegClassID:
    return 64;
  default:
    return 0;
  }
}

/// Best-effort determination of the memory access size in bytes for an X86
/// instruction with an explicit memory operand starting at MemOpIdx.
unsigned getX86MemOpSize(const MCInstrDesc &Desc, const MCInst &MCI,
                         int MemOpIdx) {
  unsigned Opcode = MCI.getOpcode();

  // Explicit zero/sign-extending loads where the memory operand is narrower
  // than the destination register.
  switch (Opcode) {
  case X86::MOVZX16rm8:
  case X86::MOVZX32rm8:
  case X86::MOVZX64rm8:
  case X86::MOVSX16rm8:
  case X86::MOVSX32rm8:
  case X86::MOVSX64rm8:
    return 1;
  case X86::MOVZX32rm16:
  case X86::MOVZX64rm16:
  case X86::MOVSX32rm16:
  case X86::MOVSX64rm16:
    return 2;
  case X86::MOVSX64rm32:
    return 4;
  default:
    break;
  }

  uint64_t TSFlags = Desc.TSFlags;

  // EVEX instructions encode the exact memory element/vector size in CD8_Scale.
  if ((TSFlags & X86II::EncodingMask) == X86II::EVEX) {
    unsigned CD8Scale =
        (TSFlags & X86II::CD8_Scale_Mask) >> X86II::CD8_Scale_Shift;
    if (CD8Scale)
      return 1U << (CD8Scale - 1);
  }

  // Check the paired explicit register operand (outside the 5-operand memory
  // addressing mode).
  for (unsigned I = 0, E = Desc.getNumOperands(); I < E; ++I) {
    if (static_cast<int>(I) >= MemOpIdx &&
        static_cast<int>(I) < MemOpIdx + X86::AddrNumOperands)
      continue;
    int16_t RC = Desc.operands()[I].RegClass;
    if (RC != -1) {
      if (unsigned Size = getRegClassMemSize(RC))
        return Size;
    }
  }

  // Fallback for memory-immediate or unary memory operations (e.g., MOV64mi32,
  // ADD32mi8, INC64m, MOV8mi).
  if (TSFlags & X86II::REX_W)
    return 8;
  uint64_t OpSize = TSFlags & X86II::OpSizeMask;
  if (OpSize == X86II::OpSize16)
    return 2;
  if (OpSize == X86II::OpSize32)
    return 4;
  if ((TSFlags & X86II::ImmMask) == X86II::Imm8)
    return 1;

  return 0;
}

unsigned getPushPopWidth(const MCInstrDesc &Desc, unsigned PtrSize) {
  uint64_t OpSize = Desc.TSFlags & X86II::OpSizeMask;
  if (OpSize == X86II::OpSize16)
    return 2;
  if (OpSize == X86II::OpSize32)
    return 4;
  return PtrSize;
}

} // namespace

void X86InstrPostProcess::setMemBarriers(Instruction &Inst, const MCInst &MCI) {
  switch (MCI.getOpcode()) {
  case X86::MFENCE:
    Inst.setLoadBarrier(true);
    Inst.setStoreBarrier(true);
    break;
  case X86::LFENCE:
    Inst.setLoadBarrier(true);
    break;
  case X86::SFENCE:
    Inst.setStoreBarrier(true);
    break;
  }
}

void X86InstrPostProcess::useStackEngine(Instruction &Inst, const MCInst &MCI) {
  // TODO(boomanaiden154): We currently do not handle PUSHF/POPF because we
  // have not done the necessary benchmarking to see if they are also
  // optimized by the stack engine.
  // TODO: We currently just remove all RSP writes from stack operations. This
  // is not fully correct because we do not model sync uops which will
  // delay subsequent rsp using non-stack instructions.
  if (X86::isPOP(MCI.getOpcode()) || X86::isPUSH(MCI.getOpcode())) {
    auto *StackRegisterDef =
        llvm::find_if(Inst.getDefs(), [](const WriteState &State) {
          return State.getRegisterID() == X86::RSP;
        });
    assert(
        StackRegisterDef != Inst.getDefs().end() &&
        "Expected push instruction to implicitly use stack pointer register.");
    Inst.getDefs().erase(StackRegisterDef);
  }
}

void X86InstrPostProcess::setStackAccessInfo(Instruction &Inst,
                                             const MCInst &MCI) {
  unsigned Opcode = MCI.getOpcode();
  const MCInstrDesc &Desc = MCII.get(Opcode);
  unsigned PtrSize = STI.hasFeature(X86::Is64Bit) ? 8 : 4;

  // 1. Handle implicit stack instructions (PUSH, POP, LEAVE).
  if (X86::isPUSH(Opcode)) {
    unsigned Width = getPushPopWidth(Desc, PtrSize);
    Inst.setSPAdjust({StackPointerAdjustment::Kind::Delta,
                      -static_cast<int64_t>(Width)});
    if (X86II::getMemoryOperandIdx(Desc) < 0)
      Inst.setStackAccess({StackAccessInfo::BaseKind::SP,
                           -static_cast<int64_t>(Width), Width});
    return;
  }

  if (X86::isPOP(Opcode)) {
    unsigned Width = getPushPopWidth(Desc, PtrSize);
    if (X86II::getMemoryOperandIdx(Desc) < 0)
      Inst.setStackAccess({StackAccessInfo::BaseKind::SP, 0, Width});

    bool PopsSP = false;
    bool PopsFP = false;
    if (Desc.getNumDefs() > 0 && MCI.getOperand(0).isReg()) {
      MCRegister Dst = MCI.getOperand(0).getReg();
      PopsSP = isStackPointerReg(Dst);
      PopsFP = isFramePointerReg(Dst);
    }
    if (PopsSP)
      Inst.setSPAdjust({StackPointerAdjustment::Kind::Unknown, 0});
    else
      Inst.setSPAdjust({StackPointerAdjustment::Kind::Delta,
                        static_cast<int64_t>(Width)});
    if (PopsFP)
      Inst.setFPAdjust({StackPointerAdjustment::Kind::Unknown, 0});
    return;
  }

  if (Opcode == X86::LEAVE || Opcode == X86::LEAVE64) {
    Inst.setStackAccess({StackAccessInfo::BaseKind::FP, 0, PtrSize});
    Inst.setSPAdjust({StackPointerAdjustment::Kind::FromOther,
                      static_cast<int64_t>(PtrSize)});
    Inst.setFPAdjust({StackPointerAdjustment::Kind::Unknown, 0});
    return;
  }

  // 2. Handle explicit memory operands referencing SP/FP + imm.
  int MemOpIdx = X86II::getMemoryOperandIdx(Desc);
  if (Inst.isMemOp() && MemOpIdx >= 0 &&
      static_cast<unsigned>(MemOpIdx + X86::AddrNumOperands) <=
          MCI.getNumOperands()) {
    const MCOperand &BaseOp = MCI.getOperand(MemOpIdx + X86::AddrBaseReg);
    const MCOperand &IndexOp = MCI.getOperand(MemOpIdx + X86::AddrIndexReg);
    const MCOperand &DispOp = MCI.getOperand(MemOpIdx + X86::AddrDisp);
    const MCOperand &SegOp = MCI.getOperand(MemOpIdx + X86::AddrSegmentReg);

    if (BaseOp.isReg() && IndexOp.isReg() && !IndexOp.getReg() &&
        SegOp.isReg() && !SegOp.getReg() && DispOp.isImm()) {
      MCRegister BaseReg = BaseOp.getReg();
      StackAccessInfo::BaseKind Base = StackAccessInfo::BaseKind::None;
      if (BaseReg == X86::RSP || BaseReg == X86::ESP)
        Base = StackAccessInfo::BaseKind::SP;
      else if (BaseReg == X86::RBP || BaseReg == X86::EBP)
        Base = StackAccessInfo::BaseKind::FP;

      if (Base != StackAccessInfo::BaseKind::None) {
        if (unsigned Size = getX86MemOpSize(Desc, MCI, MemOpIdx))
          Inst.setStackAccess({Base, DispOp.getImm(), Size});
      }
    }
  }

  // 3. Track explicit SP / FP modifications.
  switch (Opcode) {
  case X86::SUB64ri8:
  case X86::SUB64ri32:
  case X86::SUB32ri8:
  case X86::SUB32ri:
    if (MCI.getNumOperands() >= 3 && MCI.getOperand(0).isReg() &&
        MCI.getOperand(2).isImm()) {
      MCRegister Dst = MCI.getOperand(0).getReg();
      int64_t Imm = MCI.getOperand(2).getImm();
      if (Dst == X86::RSP || Dst == X86::ESP)
        Inst.setSPAdjust({StackPointerAdjustment::Kind::Delta, -Imm});
      else if (Dst == X86::RBP || Dst == X86::EBP)
        Inst.setFPAdjust({StackPointerAdjustment::Kind::Delta, -Imm});
    }
    break;
  case X86::ADD64ri8:
  case X86::ADD64ri32:
  case X86::ADD32ri8:
  case X86::ADD32ri:
    if (MCI.getNumOperands() >= 3 && MCI.getOperand(0).isReg() &&
        MCI.getOperand(2).isImm()) {
      MCRegister Dst = MCI.getOperand(0).getReg();
      int64_t Imm = MCI.getOperand(2).getImm();
      if (Dst == X86::RSP || Dst == X86::ESP)
        Inst.setSPAdjust({StackPointerAdjustment::Kind::Delta, Imm});
      else if (Dst == X86::RBP || Dst == X86::EBP)
        Inst.setFPAdjust({StackPointerAdjustment::Kind::Delta, Imm});
    }
    break;
  case X86::MOV64rr:
  case X86::MOV32rr:
  case X86::MOV64rr_REV:
  case X86::MOV32rr_REV:
    if (MCI.getNumOperands() >= 2 && MCI.getOperand(0).isReg() &&
        MCI.getOperand(1).isReg()) {
      MCRegister Dst = MCI.getOperand(0).getReg();
      MCRegister Src = MCI.getOperand(1).getReg();
      if ((Dst == X86::RBP || Dst == X86::EBP) &&
          (Src == X86::RSP || Src == X86::ESP))
        Inst.setFPAdjust({StackPointerAdjustment::Kind::FromOther, 0});
      else if ((Dst == X86::RSP || Dst == X86::ESP) &&
               (Src == X86::RBP || Src == X86::EBP))
        Inst.setSPAdjust({StackPointerAdjustment::Kind::FromOther, 0});
    }
    break;
  case X86::LEA64r:
  case X86::LEA32r:
    if (MCI.getNumOperands() >= 1 + X86::AddrNumOperands &&
        MCI.getOperand(0).isReg()) {
      MCRegister Dst = MCI.getOperand(0).getReg();
      const MCOperand &BaseOp = MCI.getOperand(1 + X86::AddrBaseReg);
      const MCOperand &IndexOp = MCI.getOperand(1 + X86::AddrIndexReg);
      const MCOperand &DispOp = MCI.getOperand(1 + X86::AddrDisp);
      const MCOperand &SegOp = MCI.getOperand(1 + X86::AddrSegmentReg);
      if (BaseOp.isReg() && IndexOp.isReg() && !IndexOp.getReg() &&
          SegOp.isReg() && !SegOp.getReg() && DispOp.isImm()) {
        MCRegister Base = BaseOp.getReg();
        int64_t Disp = DispOp.getImm();
        if (Dst == X86::RSP || Dst == X86::ESP) {
          if (Base == X86::RSP || Base == X86::ESP)
            Inst.setSPAdjust({StackPointerAdjustment::Kind::Delta, Disp});
          else if (Base == X86::RBP || Base == X86::EBP)
            Inst.setSPAdjust({StackPointerAdjustment::Kind::FromOther, Disp});
        } else if (Dst == X86::RBP || Dst == X86::EBP) {
          if (Base == X86::RBP || Base == X86::EBP)
            Inst.setFPAdjust({StackPointerAdjustment::Kind::Delta, Disp});
          else if (Base == X86::RSP || Base == X86::ESP)
            Inst.setFPAdjust({StackPointerAdjustment::Kind::FromOther, Disp});
        }
      }
    }
    break;
  default:
    break;
  }

  // Catch-all: if SP or FP is modified by an untracked instruction (excluding
  // calls, which are assumed to preserve SP across the call boundary), mark
  // the adjustment as unknown so subsequent accesses start a new coordinate
  // space.
  if (!Desc.isCall()) {
    auto CheckDef = [&](MCRegister Reg) {
      if (Inst.getSPAdjust().isNone() && isStackPointerReg(Reg))
        Inst.setSPAdjust({StackPointerAdjustment::Kind::Unknown, 0});
      if (Inst.getFPAdjust().isNone() && isFramePointerReg(Reg))
        Inst.setFPAdjust({StackPointerAdjustment::Kind::Unknown, 0});
    };
    for (unsigned I = 0, E = Desc.getNumDefs(); I < E; ++I) {
      if (MCI.getOperand(I).isReg())
        CheckDef(MCI.getOperand(I).getReg());
    }
    for (MCPhysReg ImpDef : Desc.implicit_defs())
      CheckDef(ImpDef);
  }
}

void X86InstrPostProcess::postProcessInstruction(Instruction &Inst,
                                                 const MCInst &MCI) {
  // Set IsALoadBarrier and IsAStoreBarrier flags.
  setMemBarriers(Inst, MCI);
  useStackEngine(Inst, MCI);
  setStackAccessInfo(Inst, MCI);
}

} // namespace mca
} // namespace llvm

using namespace llvm;
using namespace mca;

static InstrPostProcess *createX86InstrPostProcess(const MCSubtargetInfo &STI,
                                                   const MCInstrInfo &MCII) {
  return new X86InstrPostProcess(STI, MCII);
}

/// Extern function to initialize the targets for the X86 backend

extern "C" LLVM_C_ABI void LLVMInitializeX86TargetMCA() {
  TargetRegistry::RegisterInstrPostProcess(getTheX86_32Target(),
                                           createX86InstrPostProcess);
  TargetRegistry::RegisterInstrPostProcess(getTheX86_64Target(),
                                           createX86InstrPostProcess);
}
