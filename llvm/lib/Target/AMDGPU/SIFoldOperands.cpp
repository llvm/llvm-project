//===-- SIFoldOperands.cpp - Fold operands --- ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//
//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "si-fold-operands"
using namespace llvm;

namespace {

class SIFoldOperands : public MachineFunctionPass {
public:
  static char ID;

public:
  SIFoldOperands() : MachineFunctionPass(ID) {
    initializeSIFoldOperandsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Fold Operands";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

struct FoldCandidate {
  MachineInstr *UseMI;
  unsigned UseOpNo;
  MachineOperand *OpToFold;
  uint64_t ImmToFold;

  FoldCandidate(MachineInstr *MI, unsigned OpNo, MachineOperand *FoldOp) :
                UseMI(MI), UseOpNo(OpNo) {

    if (FoldOp->isImm()) {
      OpToFold = nullptr;
      ImmToFold = FoldOp->getImm();
    } else {
      assert(FoldOp->isReg());
      OpToFold = FoldOp;
    }
  }

  bool isImm() const {
    return !OpToFold;
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(SIFoldOperands, DEBUG_TYPE,
                "SI Fold Operands", false, false)

char SIFoldOperands::ID = 0;

char &llvm::SIFoldOperandsID = SIFoldOperands::ID;

FunctionPass *llvm::createSIFoldOperandsPass() {
  return new SIFoldOperands();
}

static bool isSafeToFold(unsigned Opcode) {
  switch(Opcode) {
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
  case AMDGPU::V_MOV_B64_PSEUDO:
  case AMDGPU::S_MOV_B32:
  case AMDGPU::S_MOV_B64:
  case AMDGPU::COPY:
    return true;
  default:
    return false;
  }
}

static bool updateOperand(FoldCandidate &Fold,
                          const TargetRegisterInfo &TRI) {
  MachineInstr *MI = Fold.UseMI;
  MachineOperand &Old = MI->getOperand(Fold.UseOpNo);
  assert(Old.isReg());

  if (Fold.isImm()) {
    Old.ChangeToImmediate(Fold.ImmToFold);
    return true;
  }

  MachineOperand *New = Fold.OpToFold;
  if (TargetRegisterInfo::isVirtualRegister(Old.getReg()) &&
      TargetRegisterInfo::isVirtualRegister(New->getReg())) {
    Old.substVirtReg(New->getReg(), New->getSubReg(), TRI);
    return true;
  }

  // FIXME: Handle physical registers.

  return false;
}

static bool isUseMIInFoldList(const std::vector<FoldCandidate> &FoldList,
                              const MachineInstr *MI) {
  for (auto Candidate : FoldList) {
    if (Candidate.UseMI == MI)
      return true;
  }
  return false;
}

static bool tryAddToFoldList(std::vector<FoldCandidate> &FoldList,
                             MachineInstr *MI, unsigned OpNo,
                             MachineOperand *OpToFold,
                             const SIInstrInfo *TII) {
  if (!TII->isOperandLegal(MI, OpNo, OpToFold)) {

    // Special case for v_mac_f32_e64 if we are trying to fold into src2
    unsigned Opc = MI->getOpcode();
    if (Opc == AMDGPU::V_MAC_F32_e64 &&
        (int)OpNo == AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2)) {
      // Check if changing this to a v_mad_f32 instruction will allow us to
      // fold the operand.
      MI->setDesc(TII->get(AMDGPU::V_MAD_F32));
      bool FoldAsMAD = tryAddToFoldList(FoldList, MI, OpNo, OpToFold, TII);
      if (FoldAsMAD) {
        MI->untieRegOperand(OpNo);
        return true;
      }
      MI->setDesc(TII->get(Opc));
    }

    // If we are already folding into another operand of MI, then
    // we can't commute the instruction, otherwise we risk making the
    // other fold illegal.
    if (isUseMIInFoldList(FoldList, MI))
      return false;

    // Operand is not legal, so try to commute the instruction to
    // see if this makes it possible to fold.
    unsigned CommuteIdx0 = TargetInstrInfo::CommuteAnyOperandIndex;
    unsigned CommuteIdx1 = TargetInstrInfo::CommuteAnyOperandIndex;
    bool CanCommute = TII->findCommutedOpIndices(MI, CommuteIdx0, CommuteIdx1);

    if (CanCommute) {
      if (CommuteIdx0 == OpNo)
        OpNo = CommuteIdx1;
      else if (CommuteIdx1 == OpNo)
        OpNo = CommuteIdx0;
    }

    // One of operands might be an Imm operand, and OpNo may refer to it after
    // the call of commuteInstruction() below. Such situations are avoided
    // here explicitly as OpNo must be a register operand to be a candidate
    // for memory folding.
    if (CanCommute && (!MI->getOperand(CommuteIdx0).isReg() ||
                       !MI->getOperand(CommuteIdx1).isReg()))
      return false;

    if (!CanCommute ||
        !TII->commuteInstruction(MI, false, CommuteIdx0, CommuteIdx1))
      return false;

    if (!TII->isOperandLegal(MI, OpNo, OpToFold))
      return false;
  }

  FoldList.push_back(FoldCandidate(MI, OpNo, OpToFold));
  return true;
}

static void foldOperand(MachineOperand &OpToFold, MachineInstr *UseMI,
                        unsigned UseOpIdx,
                        std::vector<FoldCandidate> &FoldList,
                        SmallVectorImpl<MachineInstr *> &CopiesToReplace,
                        const SIInstrInfo *TII, const SIRegisterInfo &TRI,
                        MachineRegisterInfo &MRI) {
  const MachineOperand &UseOp = UseMI->getOperand(UseOpIdx);

  // FIXME: Fold operands with subregs.
  if (UseOp.isReg() && ((UseOp.getSubReg() && OpToFold.isReg()) ||
      UseOp.isImplicit())) {
    return;
  }

  bool FoldingImm = OpToFold.isImm();
  APInt Imm;

  if (FoldingImm) {
    unsigned UseReg = UseOp.getReg();
    const TargetRegisterClass *UseRC
      = TargetRegisterInfo::isVirtualRegister(UseReg) ?
      MRI.getRegClass(UseReg) :
      TRI.getPhysRegClass(UseReg);

    Imm = APInt(64, OpToFold.getImm());

    const MCInstrDesc &FoldDesc = TII->get(OpToFold.getParent()->getOpcode());
    const TargetRegisterClass *FoldRC =
        TRI.getRegClass(FoldDesc.OpInfo[0].RegClass);

    // Split 64-bit constants into 32-bits for folding.
    if (FoldRC->getSize() == 8 && UseOp.getSubReg()) {
      if (UseRC->getSize() != 8)
        return;

      if (UseOp.getSubReg() == AMDGPU::sub0) {
        Imm = Imm.getLoBits(32);
      } else {
        assert(UseOp.getSubReg() == AMDGPU::sub1);
        Imm = Imm.getHiBits(32);
      }
    }

    // In order to fold immediates into copies, we need to change the
    // copy to a MOV.
    if (UseMI->getOpcode() == AMDGPU::COPY) {
      unsigned DestReg = UseMI->getOperand(0).getReg();
      const TargetRegisterClass *DestRC
        = TargetRegisterInfo::isVirtualRegister(DestReg) ?
        MRI.getRegClass(DestReg) :
        TRI.getPhysRegClass(DestReg);

      unsigned MovOp = TII->getMovOpcode(DestRC);
      if (MovOp == AMDGPU::COPY)
        return;

      UseMI->setDesc(TII->get(MovOp));
      CopiesToReplace.push_back(UseMI);
    }
  }

  // Special case for REG_SEQUENCE: We can't fold literals into
  // REG_SEQUENCE instructions, so we have to fold them into the
  // uses of REG_SEQUENCE.
  if (UseMI->getOpcode() == AMDGPU::REG_SEQUENCE) {
    unsigned RegSeqDstReg = UseMI->getOperand(0).getReg();
    unsigned RegSeqDstSubReg = UseMI->getOperand(UseOpIdx + 1).getImm();

    for (MachineRegisterInfo::use_iterator
         RSUse = MRI.use_begin(RegSeqDstReg),
         RSE = MRI.use_end(); RSUse != RSE; ++RSUse) {

      MachineInstr *RSUseMI = RSUse->getParent();
      if (RSUse->getSubReg() != RegSeqDstSubReg)
        continue;

      foldOperand(OpToFold, RSUseMI, RSUse.getOperandNo(), FoldList,
                  CopiesToReplace, TII, TRI, MRI);
    }
    return;
  }

  const MCInstrDesc &UseDesc = UseMI->getDesc();

  // Don't fold into target independent nodes.  Target independent opcodes
  // don't have defined register classes.
  if (UseDesc.isVariadic() ||
      UseDesc.OpInfo[UseOpIdx].RegClass == -1)
    return;

  if (FoldingImm) {
    MachineOperand ImmOp = MachineOperand::CreateImm(Imm.getSExtValue());
    tryAddToFoldList(FoldList, UseMI, UseOpIdx, &ImmOp, TII);
    return;
  }

  tryAddToFoldList(FoldList, UseMI, UseOpIdx, &OpToFold, TII);

  // FIXME: We could try to change the instruction from 64-bit to 32-bit
  // to enable more folding opportunites.  The shrink operands pass
  // already does this.
  return;
}

bool SIFoldOperands::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      if (!isSafeToFold(MI.getOpcode()))
        continue;

      unsigned OpSize = TII->getOpSize(MI, 1);
      MachineOperand &OpToFold = MI.getOperand(1);
      bool FoldingImm = OpToFold.isImm();

      // FIXME: We could also be folding things like FrameIndexes and
      // TargetIndexes.
      if (!FoldingImm && !OpToFold.isReg())
        continue;

      // Folding immediates with more than one use will increase program size.
      // FIXME: This will also reduce register usage, which may be better
      // in some cases.  A better heuristic is needed.
      if (FoldingImm && !TII->isInlineConstant(OpToFold, OpSize) &&
          !MRI.hasOneUse(MI.getOperand(0).getReg()))
        continue;

      if (OpToFold.isReg() &&
          !TargetRegisterInfo::isVirtualRegister(OpToFold.getReg()))
        continue;

      // Prevent folding operands backwards in the function. For example,
      // the COPY opcode must not be replaced by 1 in this example:
      //
      //    %vreg3<def> = COPY %VGPR0; VGPR_32:%vreg3
      //    ...
      //    %VGPR0<def> = V_MOV_B32_e32 1, %EXEC<imp-use>
      MachineOperand &Dst = MI.getOperand(0);
      if (Dst.isReg() &&
          !TargetRegisterInfo::isVirtualRegister(Dst.getReg()))
        continue;

      // We need mutate the operands of new mov instructions to add implicit
      // uses of EXEC, but adding them invalidates the use_iterator, so defer
      // this.
      SmallVector<MachineInstr *, 4> CopiesToReplace;

      std::vector<FoldCandidate> FoldList;
      for (MachineRegisterInfo::use_iterator
           Use = MRI.use_begin(MI.getOperand(0).getReg()), E = MRI.use_end();
           Use != E; ++Use) {

        MachineInstr *UseMI = Use->getParent();

        foldOperand(OpToFold, UseMI, Use.getOperandNo(), FoldList,
                    CopiesToReplace, TII, TRI, MRI);
      }

      // Make sure we add EXEC uses to any new v_mov instructions created.
      for (MachineInstr *Copy : CopiesToReplace)
        Copy->addImplicitDefUseOperands(MF);

      for (FoldCandidate &Fold : FoldList) {
        if (updateOperand(Fold, TRI)) {
          // Clear kill flags.
          if (!Fold.isImm()) {
            assert(Fold.OpToFold && Fold.OpToFold->isReg());
            // FIXME: Probably shouldn't bother trying to fold if not an
            // SGPR. PeepholeOptimizer can eliminate redundant VGPR->VGPR
            // copies.
            MRI.clearKillFlags(Fold.OpToFold->getReg());
          }
          DEBUG(dbgs() << "Folded source from " << MI << " into OpNo " <<
                Fold.UseOpNo << " of " << *Fold.UseMI << '\n');
        }
      }
    }
  }
  return false;
}
