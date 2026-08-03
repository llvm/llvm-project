//===-- PISAInstrInfo.cpp - PISA Instruction Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAInstrInfo.h"
#include "PISA.h"
#include "PISASubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "PISAGenInstrInfo.inc"

using namespace llvm;
using namespace PISA;

PISAInstrInfo::PISAInstrInfo(const PISASubtarget &STI)
    : PISAGenInstrInfo(STI, RI) {}

bool PISAInstrInfo::isNoEmissionInstr(const MachineInstr &MI) const {
  // Functions are emitted to C-style function signature.
  // These instructions are not required to be in the output.
  return isFunctionParamInstr(MI);
}

bool PISAInstrInfo::isFunctionParamInstr(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case PISA::functionParameter_i8:
  case PISA::functionParameter_i16:
  case PISA::functionParameter_i32:
  case PISA::functionParameter_i64:
  case PISA::functionParameter_v2i8:
  case PISA::functionParameter_v3i8:
  case PISA::functionParameter_v4i8:
  case PISA::functionParameter_v2i16:
  case PISA::functionParameter_v3i16:
  case PISA::functionParameter_v4i16:
  case PISA::functionParameter_v2i32:
  case PISA::functionParameter_v3i32:
  case PISA::functionParameter_v4i32:
  case PISA::functionParameter_v5i32:
  case PISA::functionParameter_v6i32:
  case PISA::functionParameter_v7i32:
  case PISA::functionParameter_v8i32:
  case PISA::functionParameter_v16i32:
  case PISA::functionParameter_v32i32:
  case PISA::functionParameter_v64i32:
  case PISA::functionParameter_v2i64:
  case PISA::functionParameter_v3i64:
  case PISA::functionParameter_v4i64:
    return true;
  default:
    return false;
  }
}

namespace llvm {
namespace PISA {
static const MachineOperand &getMO(const MachineInstr &MI, PISA::OpName Name) {
  int16_t Idx = getNamedOperandIdx(MI.getOpcode(), Name);
  assert(Idx >= 0 && "name not present!");

  return MI.getOperand(Idx);
}
} // namespace PISA
} // namespace llvm

// See description in TargetInstrInfo.h
bool PISAInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond,
                                  bool /*AllowModify*/) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end() || !isUnpredicatedTerminator(*I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = &*I;

  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(*--I)) {
    if (LastInst->isUnconditionalBranch()) {
      if (LastInst->getOpcode() != PISA::gotolabel)
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (LastInst->isConditionalBranch()) {
      if (LastInst->getOpcode() != PISA::predgoto)
        return true;
      // Block ends with fall-through condbranch.
      TBB = getMO(*LastInst, OpName::label).getMBB();
      // mod, pred
      Cond.push_back(LastInst->getOperand(0));
      Cond.push_back(LastInst->getOperand(1));
      return false;
    }
    return true; // Can't handle indirect branch.
  }

  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = &*I;

  if (!SecondLastInst->isConditionalBranch() ||
      !LastInst->isUnconditionalBranch() ||
      // triple terminator?
      (I != MBB.begin() && isUnpredicatedTerminator(*--I)))
    return true;

  if (SecondLastInst->getOpcode() != PISA::predgoto ||
      LastInst->getOpcode() != PISA::gotolabel)
    return true;

  TBB = getMO(*SecondLastInst, OpName::label).getMBB();
  FBB = LastInst->getOperand(0).getMBB();

  // mod, pred
  Cond.push_back(SecondLastInst->getOperand(0));
  Cond.push_back(SecondLastInst->getOperand(1));

  return false;
}

// See description in TargetInstrInfo.h
unsigned PISAInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                     int *BytesRemoved) const {
  assert(!BytesRemoved && "not supported!");

  unsigned Count = 0;
  for (auto &MI : llvm::make_early_inc_range(MBB.terminators())) {
    assert(MI.isBranch() && "not a branch?");
    MI.eraseFromParent();
    Count++;
  }

  return Count;
}

// See description in TargetInstrInfo.h
unsigned PISAInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {

  assert(!BytesAdded && "not supported!");
  assert(TBB && "Should have at least one branch!");

  if (!FBB) {
    if (Cond.empty()) { // Unconditional branch
      BuildMI(&MBB, DL, get(PISA::gotolabel)).addMBB(TBB);
    } else { // Conditional branch
      assert(Cond.size() == 2 && "wrong number of args?");
      BuildMI(&MBB, DL, get(PISA::predgoto))
          .add(Cond[0])
          .add(Cond[1])
          .addMBB(TBB);
    }
    return 1;
  }

  // Two-way Conditional Branch.
  assert(Cond.size() == 2 && "wrong number of args?");
  BuildMI(&MBB, DL, get(PISA::predgoto)).add(Cond[0]).add(Cond[1]).addMBB(TBB);
  BuildMI(&MBB, DL, get(PISA::gotolabel)).addMBB(FBB);
  return 2;
}

bool PISAInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  if (Cond.empty())
    return true;

  uint64_t DoNegate = Cond[0].getImm();
  Cond[0].setImm(!DoNegate);

  return false;
}

// We have to override this because LowerCopy() in ExpandPostRAPseudos
// gets rid of identity copies without checking subregs. It doesn't check
// subregs because register allocation has already happened (but not for PISA)
// so there aren't any subregs.
bool PISAInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  if (!MI.isCopy())
    return false;

  MachineOperand &DstMO = MI.getOperand(0);
  MachineOperand &SrcMO = MI.getOperand(1);

  // only generate instructions using defined registers
  if (!SrcMO.isUndef())
    copyPhysReg(*MI.getParent(), MI, MI.getDebugLoc(), DstMO.getReg(),
                SrcMO.getReg(), SrcMO.isKill());

  MI.eraseFromParent();

  return true;
}

bool PISAInstrInfo::isSafeToMove(const MachineInstr &MI,
                                 const MachineBasicBlock *MBB,
                                 const MachineFunction &MF) const {
  // Convergent instructions must not be moved across basic blocks because
  // doing so can change the set of threads executing the instruction,
  // potentially producing incorrect results.
  if (MI.isConvergent())
    return false;
  return true;
}

void PISAInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I,
                                const DebugLoc &DL, Register DestReg,
                                Register SrcReg, bool KillSrc,
                                bool RenamableDest, bool RenamableSrc) const {

  assert(I->isCopy() && "Copy instruction is expected");
  auto &MRI = I->getMF()->getRegInfo();

  auto GetSubregRc = [&](MachineOperand &MO) -> const TargetRegisterClass * {
    unsigned Subreg = MO.getSubReg();
    Register Reg = MO.getReg();
    auto *SuperRC = Reg.isPhysical() ? RI.getMinimalPhysRegClass(Reg)
                                     : MRI.getRegClass(Reg);
    if (Subreg == 0)
      return SuperRC;

    return RI.getSubRegisterClass(SuperRC, Subreg);
  };

  auto &DstOp = I->getOperand(0);
  auto &SrcOp = I->getOperand(1);
  auto *DstSubRC = GetSubregRc(DstOp);
  auto *SrcSubRC = GetSubregRc(SrcOp);

  unsigned Op = 0;

  const unsigned DstSubNumElts = RI.getNumEltsFromRegClass(DstSubRC);
  const unsigned DstSubEltSize = RI.getBitSizeFromRegClass(DstSubRC);
  const unsigned SrcSubNumElts = RI.getNumEltsFromRegClass(SrcSubRC);
  const unsigned SrcSubEltSize = RI.getBitSizeFromRegClass(SrcSubRC);
  const bool DstSubIsVector = DstSubNumElts > 1;
  const bool SrcSubIsVector = SrcSubNumElts > 1;
  const bool DstSubIsScalar = !DstSubIsVector;
  const bool SrcSubIsScalar = !SrcSubIsVector;

  if (DstSubIsVector && SrcSubIsVector) {
    if (DstSubNumElts == 4 && DstSubEltSize == 8 && SrcSubNumElts == 2 &&
        SrcSubEltSize == 16)
      Op = PISA::mov_v4i8_v2i16_r;
    else if (DstSubNumElts == 2 && DstSubEltSize == 16 && SrcSubNumElts == 4 &&
             SrcSubEltSize == 8)
      Op = PISA::mov_v2i16_v4i8_r;
    else if (DstSubNumElts == 4 && DstSubEltSize == 16 && SrcSubNumElts == 2 &&
             SrcSubEltSize == 32)
      Op = PISA::mov_v4i16_v2i32_r;
    else if (DstSubNumElts == 2 && DstSubEltSize == 32 && SrcSubNumElts == 4 &&
             SrcSubEltSize == 16)
      Op = PISA::mov_v2i32_v4i16_r;
    else if (DstSubNumElts == 4 && DstSubEltSize == 32 && SrcSubNumElts == 2 &&
             SrcSubEltSize == 64)
      Op = PISA::mov_v4i32_v2i64_r;
    else if (DstSubNumElts == 2 && DstSubEltSize == 64 && SrcSubNumElts == 4 &&
             SrcSubEltSize == 32)
      Op = PISA::mov_v2i64_v4i32_r;
    else {
      assert(DstSubNumElts == SrcSubNumElts && "num elts mismatch!");
      if (DstSubNumElts == 2) {
        switch (DstSubEltSize) {
        case 8:
          Op = PISA::mov_v2i8_r;
          break;
        case 16:
          Op = PISA::mov_v2i16_r;
          break;
        case 32:
          Op = PISA::mov_v2i32_r;
          break;
        case 64:
          Op = PISA::mov_v2i64_r;
          break;
        default:
          llvm_unreachable("unknown elt size!");
        }
      } else if (DstSubNumElts == 3) {
        switch (DstSubEltSize) {
        case 8:
          Op = PISA::mov_v3i8_r;
          break;
        case 16:
          Op = PISA::mov_v3i16_r;
          break;
        default:
          llvm_unreachable("unknown elt size!");
        }
      } else if (DstSubNumElts == 4) {
        switch (DstSubEltSize) {
        case 8:
          Op = PISA::mov_v4i8_r;
          break;
        case 16:
          Op = PISA::mov_v4i16_r;
          break;
        case 32:
          Op = PISA::mov_v4i32_r;
          break;
        default:
          llvm_unreachable("unknown elt size!");
        }
      } else {
        llvm_unreachable("unknown vec size!");
      }
    }
  } else if (DstSubIsVector && SrcSubIsScalar) {
    if (DstSubNumElts == 2 && DstSubEltSize == 8 && SrcSubEltSize == 16)
      Op = PISA::mov_v2i8_i16_r;
    else if (DstSubNumElts == 4 && DstSubEltSize == 8 && SrcSubEltSize == 32)
      Op = PISA::mov_v4i8_i32_r;
    else if (DstSubNumElts == 2 && DstSubEltSize == 16 && SrcSubEltSize == 32)
      Op = PISA::mov_v2i16_i32_r;
    else if (DstSubNumElts == 4 && DstSubEltSize == 16 && SrcSubEltSize == 64)
      Op = PISA::mov_v4i16_i64_r;
    else if (DstSubNumElts == 2 && DstSubEltSize == 32 && SrcSubEltSize == 64)
      Op = PISA::mov_v2i32_i64_r;
    else if (DstSubNumElts == 4 && DstSubEltSize == 32 && SrcSubEltSize == 128)
      Op = PISA::mov_v4i32_i128_r;
    else if (DstSubNumElts == 2 && DstSubEltSize == 64 && SrcSubEltSize == 128)
      Op = PISA::mov_v2i64_i128_r;
    else
      llvm_unreachable("wrong copy instruction");
  } else if (DstSubIsScalar && SrcSubIsVector) {
    if (DstSubEltSize == 16 && SrcSubNumElts == 2 && SrcSubEltSize == 8)
      Op = PISA::mov_i16_v2i8_r;
    else if (DstSubEltSize == 32 && SrcSubNumElts == 2 && SrcSubEltSize == 16)
      Op = PISA::mov_i32_v2i16_r;
    else if (DstSubEltSize == 32 && SrcSubNumElts == 4 && SrcSubEltSize == 8)
      Op = PISA::mov_i32_v4i8_r;
    else if (DstSubEltSize == 64 && SrcSubNumElts == 2 && SrcSubEltSize == 32)
      Op = PISA::mov_i64_v2i32_r;
    else if (DstSubEltSize == 64 && SrcSubNumElts == 4 && SrcSubEltSize == 16)
      Op = PISA::mov_i64_v4i16_r;
    else if (DstSubEltSize == 128 && SrcSubNumElts == 2 && SrcSubEltSize == 64)
      Op = PISA::mov_i128_v2i64_r;
    else if (DstSubEltSize == 128 && SrcSubNumElts == 4 && SrcSubEltSize == 32)
      Op = PISA::mov_i128_v4i32_r;
    else
      llvm_unreachable("wrong copy operation");
  } else if (DstSubIsScalar && SrcSubIsScalar) {
    const auto *TRI = static_cast<const PISARegisterInfo *>(
        I->getMF()->getSubtarget().getRegisterInfo());
    auto I16 = LLT::integer(16);

    if ((DstSubEltSize == 1) && (SrcSubEltSize == 1)) {
      // sel.16b %tmp, 1, 0, %p_in
      // ucmp.ne.16b %p_out, %tmp, 0
      auto TmpReg = MRI.createGenericVirtualRegister(I16);
      MRI.setRegClass(TmpReg, TRI->getRegClassFromLLT(I16));
      Op = PISA::sel_16_iip;
      BuildMI(MBB, I, DL, get(Op))
          .addDef(TmpReg)
          .addReg(SrcOp.getReg())
          .addImm(1)
          .addImm(0);
      Op = PISA::ucmp_ne_16b_pri;
      BuildMI(MBB, I, DL, get(Op))
          .addDef(DstOp.getReg())
          .addReg(TmpReg)
          .addImm(0);
      return;
    }
    if (DstSubEltSize == 1) {
      // ucmp.ne.??b %p, %src, 0
      auto TmpReg = SrcOp.getReg();
      if (SrcSubEltSize == 8) {
        TmpReg = MRI.createGenericVirtualRegister(I16);
        MRI.setRegClass(TmpReg, TRI->getRegClassFromLLT(I16));
        Op = PISA::zext_16b_8b_r;
        BuildMI(MBB, I, DL, get(Op)).addDef(TmpReg).addReg(SrcOp.getReg());
      }
      switch (SrcSubEltSize) {
      default:
        llvm_unreachable("unsupported source size!");
      case 8:
      case 16:
        Op = PISA::ucmp_ne_16b_pri;
        break;
      case 32:
        Op = PISA::ucmp_ne_32b_pri;
        break;
      case 64:
        Op = PISA::ucmp_ne_64b_pri;
        break;
      }
      BuildMI(MBB, I, DL, get(Op))
          .addDef(DstOp.getReg())
          .addReg(TmpReg)
          .addImm(0);
      return;
    }
    if (SrcSubEltSize == 1) {
      // sel.??b %dst, 1, 0, %p
      auto TmpReg = DstOp.getReg();
      if (DstSubEltSize == 8) {
        TmpReg = MRI.createGenericVirtualRegister(I16);
        MRI.setRegClass(TmpReg, TRI->getRegClassFromLLT(I16));
      }
      switch (DstSubEltSize) {
      default:
        llvm_unreachable("unsupported destination size!");
      case 8:
      case 16:
        Op = PISA::sel_16_iip;
        break;
      case 32:
        Op = PISA::sel_32_iip;
        break;
      case 64:
        Op = PISA::sel_64_iip;
        break;
      }
      BuildMI(MBB, I, DL, get(Op))
          .addDef(TmpReg)
          .addReg(SrcOp.getReg())
          .addImm(1)
          .addImm(0);
      if (DstSubEltSize == 8) {
        Op = PISA::trunc_8b_16b_r;
        BuildMI(MBB, I, DL, get(Op)).addDef(DstOp.getReg()).addReg(TmpReg);
      }
      return;
    }
    switch (DstSubEltSize) {
    case 8:
      Op = PISA::mov_i8_r;
      break;
    case 16:
      Op = PISA::mov_i16_r;
      break;
    case 32:
      Op = PISA::mov_i32_r;
      break;
    case 64:
      Op = PISA::mov_i64_r;
      break;
    default:
      llvm_unreachable("unknown elt size!");
    }
  }

  unsigned DstSubreg = DstOp.getSubReg();
  unsigned SrcSubreg = SrcOp.getSubReg();

  RegState Flags = DstOp.isUndef() ? RegState::Undef : RegState::NoFlags;
  BuildMI(MBB, I, DL, get(Op))
      .addDef(DstOp.getReg(), Flags, DstSubreg)
      .addReg(SrcOp.getReg(), getKillRegState(KillSrc), SrcSubreg);
}
