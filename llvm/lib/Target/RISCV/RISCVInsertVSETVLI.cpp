//===- RISCVInsertVSETVLI.cpp - Insert VSETVLI instructions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts VSETVLI instructions where
// needed.
//
// The pass consists of a single pass over each basic block looking for changes
// in VL/VTYPE usage that requires a vsetvli to be inserted. We assume the
// VL/VTYPE values are unknown from predecessors so the first vector instruction
// will always require a new VSETVLI.
//
// TODO: Future enhancements to this pass will take into account VL/VTYPE from
// predecessors.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-vsetvli"
#define RISCV_INSERT_VSETVLI_NAME "RISCV Insert VSETVLI pass"

namespace {

class VSETVLIInfo {
  union {
    Register AVLReg;
    unsigned AVLImm;
  };

  enum : uint8_t {
    Uninitialized,
    AVLIsReg,
    AVLIsImm,
    Unknown,
  } State = Uninitialized;

  // Fields from VTYPE.
  RISCVII::VLMUL VLMul = RISCVII::LMUL_1;
  uint8_t SEW = 0;
  bool TailAgnostic = false;
  bool MaskAgnostic = false;

public:
  VSETVLIInfo() : AVLImm(0) {}

  bool isValid() const { return State != Uninitialized; }
  void setUnknown() { State = Unknown; }
  bool isUnknown() const { return State == Unknown; }

  void setAVLReg(Register Reg) {
    AVLReg = Reg;
    State = AVLIsReg;
  }

  void setAVLImm(unsigned Imm) {
    AVLImm = Imm;
    State = AVLIsImm;
  }

  bool hasAVLImm() const { return State == AVLIsImm; }
  bool hasAVLReg() const { return State == AVLIsReg; }
  Register getAVLReg() const {
    assert(hasAVLReg());
    return AVLReg;
  }
  unsigned getAVLImm() const {
    assert(hasAVLImm());
    return AVLImm;
  }

  bool hasSameAVL(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare AVL in unknown state");
    if (hasAVLReg() && Other.hasAVLReg())
      return getAVLReg() == Other.getAVLReg();

    if (hasAVLImm() && Other.hasAVLImm())
      return getAVLImm() == Other.getAVLImm();

    return false;
  }

  void setVTYPE(unsigned VType) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = RISCVVType::getVLMUL(VType);
    SEW = RISCVVType::getSEW(VType);
    TailAgnostic = RISCVVType::isTailAgnostic(VType);
    MaskAgnostic = RISCVVType::isMaskAgnostic(VType);
  }
  void setVTYPE(RISCVII::VLMUL L, unsigned S, bool TA, bool MA) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = L;
    SEW = S;
    TailAgnostic = TA;
    MaskAgnostic = MA;
  }

  unsigned encodeVTYPE() const {
    return RISCVVType::encodeVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);
  }

  bool hasSameVTYPE(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    return std::tie(VLMul, SEW, TailAgnostic, MaskAgnostic) ==
           std::tie(Other.VLMul, Other.SEW, Other.TailAgnostic,
                    Other.MaskAgnostic);
  }

  bool isCompatible(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    // Nothing is compatible with Unknown.
    if (isUnknown() || Other.isUnknown())
      return false;

    // If other doesn't need an AVLReg and the SEW matches, consider it
    // compatible.
    if (Other.hasAVLReg() && Other.AVLReg == RISCV::NoRegister) {
      if (SEW == Other.SEW)
        return true;
    }

    // VTypes must match.
    if (!hasSameVTYPE(Other))
      return false;

    if (hasAVLImm() != Other.hasAVLImm())
      return false;

    if (hasAVLImm())
      return getAVLImm() == Other.getAVLImm();

    return getAVLReg() == Other.getAVLReg();
  }
};

class RISCVInsertVSETVLI : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;

public:
  static char ID;

  RISCVInsertVSETVLI() : MachineFunctionPass(ID) {
    initializeRISCVInsertVSETVLIPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_VSETVLI_NAME; }

private:
  void insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                     const VSETVLIInfo &Info);

  bool emitVSETVLIs(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVInsertVSETVLI::ID = 0;

INITIALIZE_PASS(RISCVInsertVSETVLI, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

static MachineInstr *elideCopies(MachineInstr *MI,
                                 const MachineRegisterInfo *MRI) {
  while (true) {
    if (!MI->isFullCopy())
      return MI;
    if (!Register::isVirtualRegister(MI->getOperand(1).getReg()))
      return nullptr;
    MI = MRI->getVRegDef(MI->getOperand(1).getReg());
    if (!MI)
      return nullptr;
  }
}

static VSETVLIInfo computeInfoForInstr(const MachineInstr &MI, uint64_t TSFlags,
                                       const MachineRegisterInfo *MRI) {
  VSETVLIInfo InstrInfo;
  unsigned NumOperands = MI.getNumExplicitOperands();

  RISCVII::VLMUL VLMul = RISCVII::getLMul(TSFlags);

  unsigned Log2SEW = MI.getOperand(NumOperands - 1).getImm();
  unsigned SEW = 1 << Log2SEW;
  assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");

  // Default to tail agnostic unless the destination is tied to a source.
  // Unless the source is undef. In that case the user would have some control
  // over the tail values. The tail policy is also ignored on instructions
  // that only update element 0 like vmv.s.x or reductions so use agnostic
  // there to match the common case.
  // FIXME: This is conservatively correct, but we might want to detect that
  // the input is undefined.
  bool ForceTailAgnostic = RISCVII::doesForceTailAgnostic(TSFlags);
  bool TailAgnostic = true;
  unsigned UseOpIdx;
  if (!ForceTailAgnostic && MI.isRegTiedToUseOperand(0, &UseOpIdx)) {
    TailAgnostic = false;
    // If the tied operand is an IMPLICIT_DEF we can keep TailAgnostic.
    const MachineOperand &UseMO = MI.getOperand(UseOpIdx);
    MachineInstr *UseMI = MRI->getVRegDef(UseMO.getReg());
    if (UseMI) {
      UseMI = elideCopies(UseMI, MRI);
      if (UseMI && UseMI->isImplicitDef())
        TailAgnostic = true;
    }
  }

  if (RISCVII::hasVLOp(TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(MI.getNumExplicitOperands() - 2);
    if (VLOp.isImm())
      InstrInfo.setAVLImm(VLOp.getImm());
    else
      InstrInfo.setAVLReg(VLOp.getReg());
  } else
    InstrInfo.setAVLReg(RISCV::NoRegister);
  InstrInfo.setVTYPE(VLMul, SEW, /*TailAgnostic*/ TailAgnostic,
                     /*MaskAgnostic*/ false);

  return InstrInfo;
}

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                                       const VSETVLIInfo &Info) {
  DebugLoc DL = MI.getDebugLoc();

  if (Info.hasAVLImm()) {
    // TODO: Use X0 as the destination.
    Register DestReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, MI, DL, TII->get(RISCV::PseudoVSETIVLI))
        .addReg(DestReg, RegState::Define | RegState::Dead)
        .addImm(Info.getAVLImm())
        .addImm(Info.encodeVTYPE());
    return;
  }

  Register AVLReg = Info.getAVLReg();
  if (AVLReg == RISCV::NoRegister) {
    BuildMI(MBB, MI, DL, TII->get(RISCV::PseudoVSETVLI))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .addReg(RISCV::X0, RegState::Kill)
        .addImm(Info.encodeVTYPE())
        .addReg(RISCV::VL, RegState::Implicit);
    return;
  }

  Register DestReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
  BuildMI(MBB, MI, DL, TII->get(RISCV::PseudoVSETVLI))
      .addReg(DestReg, RegState::Define | RegState::Dead)
      .addReg(Info.getAVLReg())
      .addImm(Info.encodeVTYPE());
}

// Return a VSETVLIInfo representing the changes made by this VSETVLI or
// VSETIVLI instruction.
VSETVLIInfo getInfoForVSETVLI(const MachineInstr &MI) {
  VSETVLIInfo NewInfo;
  if (MI.getOpcode() == RISCV::PseudoVSETVLI) {
    Register AVLReg = MI.getOperand(1).getReg();
    assert((AVLReg != RISCV::X0 || MI.getOperand(0).getReg() != RISCV::X0) &&
           "Can't handle X0, X0 vsetvli yet");
    NewInfo.setAVLReg(AVLReg);
  } else {
    assert(MI.getOpcode() == RISCV::PseudoVSETIVLI);
    NewInfo.setAVLImm(MI.getOperand(1).getImm());
  }
  NewInfo.setVTYPE(MI.getOperand(2).getImm());

  return NewInfo;
}

bool RISCVInsertVSETVLI::emitVSETVLIs(MachineBasicBlock &MBB) {
  bool MadeChange = false;

  // Assume predecessor state is unknown.
  VSETVLIInfo CurInfo;
  CurInfo.setUnknown();

  for (MachineInstr &MI : MBB) {
    // If this is an explicit VSETVLI or VSETIVLI, update our state.
    if (MI.getOpcode() == RISCV::PseudoVSETVLI ||
        MI.getOpcode() == RISCV::PseudoVSETIVLI) {
      // Conservatively, mark the VL and VTYPE as live.
      assert(MI.getOperand(3).getReg() == RISCV::VL &&
             MI.getOperand(4).getReg() == RISCV::VTYPE &&
             "Unexpected operands where VL and VTYPE should be");
      MI.getOperand(3).setIsDead(false);
      MI.getOperand(4).setIsDead(false);
      MadeChange = true;
      CurInfo = getInfoForVSETVLI(MI);
      continue;
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      VSETVLIInfo NewInfo = computeInfoForInstr(MI, TSFlags, MRI);
      if (RISCVII::hasVLOp(TSFlags)) {
        MachineOperand &VLOp = MI.getOperand(MI.getNumExplicitOperands() - 2);
        if (VLOp.isReg()) {
          // Erase the AVL operand from the instruction.
          VLOp.setReg(RISCV::NoRegister);
          VLOp.setIsKill(false);
        }
        MI.addOperand(MachineOperand::CreateReg(RISCV::VL, /*isDef*/ false,
                                                /*isImp*/ true));
      }
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ false,
                                              /*isImp*/ true));

      bool NeedVSETVLI = true;
      if (CurInfo.isValid() && CurInfo.isCompatible(NewInfo))
        NeedVSETVLI = false;

      // We didn't find a compatible value. If our AVL is a virtual register,
      // it might be defined by a VSET(I)VLI. If it has the same VTYPE we need
      // and the last VL/VTYPE we observed is the same, we don't need a
      // VSETVLI here.
      if (NeedVSETVLI && !CurInfo.isUnknown() && NewInfo.hasAVLReg() &&
          NewInfo.getAVLReg().isVirtual() && NewInfo.hasSameVTYPE(CurInfo)) {
        if (MachineInstr *DefMI = MRI->getVRegDef(NewInfo.getAVLReg())) {
          if (DefMI->getOpcode() == RISCV::PseudoVSETVLI ||
              DefMI->getOpcode() == RISCV::PseudoVSETIVLI) {
            VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
            if (DefInfo.hasSameAVL(CurInfo) && DefInfo.hasSameVTYPE(CurInfo))
              NeedVSETVLI = false;
          }
        }
      }

      // If this instruction isn't compatible with the previous VL/VTYPE
      // we need to insert a VSETVLI.
      if (NeedVSETVLI) {
        insertVSETVLI(MBB, MI, NewInfo);
        CurInfo = NewInfo;
      }

      // If we find an instruction we at least changed the operands.
      MadeChange = true;
    }
    // If this is something updates VL/VTYPE that we don't know about, set
    // the state to unknown.
    if (MI.isCall() || MI.modifiesRegister(RISCV::VL) ||
        MI.modifiesRegister(RISCV::VTYPE)) {
      VSETVLIInfo NewInfo;
      NewInfo.setUnknown();
      CurInfo = NewInfo;
    }
  }

  return MadeChange;
}

bool RISCVInsertVSETVLI::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasStdExtV())
    return false;

  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF)
    Changed |= emitVSETVLIs(MBB);

  return Changed;
}

/// Returns an instance of the Insert VSETVLI pass.
FunctionPass *llvm::createRISCVInsertVSETVLIPass() {
  return new RISCVInsertVSETVLI();
}
