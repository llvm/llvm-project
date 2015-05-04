//===-- SILowerControlFlow.cpp - Use predicates for control flow ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass lowers the pseudo control flow instructions to real
/// machine instructions.
///
/// All control flow is handled using predicated instructions and
/// a predicate stack.  Each Scalar ALU controls the operations of 64 Vector
/// ALUs.  The Scalar ALU can update the predicate for any of the Vector ALUs
/// by writting to the 64-bit EXEC register (each bit corresponds to a
/// single vector ALU).  Typically, for predicates, a vector ALU will write
/// to its bit of the VCC register (like EXEC VCC is 64-bits, one for each
/// Vector ALU) and then the ScalarALU will AND the VCC register with the
/// EXEC to update the predicates.
///
/// For example:
/// %VCC = V_CMP_GT_F32 %VGPR1, %VGPR2
/// %SGPR0 = SI_IF %VCC
///   %VGPR0 = V_ADD_F32 %VGPR0, %VGPR0
/// %SGPR0 = SI_ELSE %SGPR0
///   %VGPR0 = V_SUB_F32 %VGPR0, %VGPR0
/// SI_END_CF %SGPR0
///
/// becomes:
///
/// %SGPR0 = S_AND_SAVEEXEC_B64 %VCC  // Save and update the exec mask
/// %SGPR0 = S_XOR_B64 %SGPR0, %EXEC  // Clear live bits from saved exec mask
/// S_CBRANCH_EXECZ label0            // This instruction is an optional
///                                   // optimization which allows us to
///                                   // branch if all the bits of
///                                   // EXEC are zero.
/// %VGPR0 = V_ADD_F32 %VGPR0, %VGPR0 // Do the IF block of the branch
///
/// label0:
/// %SGPR0 = S_OR_SAVEEXEC_B64 %EXEC   // Restore the exec mask for the Then block
/// %EXEC = S_XOR_B64 %SGPR0, %EXEC    // Clear live bits from saved exec mask
/// S_BRANCH_EXECZ label1              // Use our branch optimization
///                                    // instruction again.
/// %VGPR0 = V_SUB_F32 %VGPR0, %VGPR   // Do the THEN block
/// label1:
/// %EXEC = S_OR_B64 %EXEC, %SGPR0     // Re-enable saved exec mask bits
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

namespace {

class SILowerControlFlowPass : public MachineFunctionPass {

private:
  static const unsigned SkipThreshold = 12;

  static char ID;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;

  bool shouldSkip(MachineBasicBlock *From, MachineBasicBlock *To);

  void Skip(MachineInstr &From, MachineOperand &To);
  void SkipIfDead(MachineInstr &MI);

  void If(MachineInstr &MI);
  void Else(MachineInstr &MI);
  void Break(MachineInstr &MI);
  void IfBreak(MachineInstr &MI);
  void ElseBreak(MachineInstr &MI);
  void Loop(MachineInstr &MI);
  void EndCf(MachineInstr &MI);

  void Kill(MachineInstr &MI);
  void Branch(MachineInstr &MI);

  void LoadM0(MachineInstr &MI, MachineInstr *MovRel, int Offset = 0);
  void computeIndirectRegAndOffset(unsigned VecReg, unsigned &Reg, int &Offset);
  void IndirectSrc(MachineInstr &MI);
  void IndirectDst(MachineInstr &MI);

public:
  SILowerControlFlowPass(TargetMachine &tm) :
    MachineFunctionPass(ID), TRI(nullptr), TII(nullptr) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Lower control flow instructions";
  }

};

} // End anonymous namespace

char SILowerControlFlowPass::ID = 0;

FunctionPass *llvm::createSILowerControlFlowPass(TargetMachine &tm) {
  return new SILowerControlFlowPass(tm);
}

bool SILowerControlFlowPass::shouldSkip(MachineBasicBlock *From,
                                        MachineBasicBlock *To) {

  unsigned NumInstr = 0;

  for (MachineBasicBlock *MBB = From; MBB != To && !MBB->succ_empty();
       MBB = *MBB->succ_begin()) {

    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         NumInstr < SkipThreshold && I != E; ++I) {

      if (I->isBundle() || !I->isBundled())
        if (++NumInstr >= SkipThreshold)
          return true;
    }
  }

  return false;
}

void SILowerControlFlowPass::Skip(MachineInstr &From, MachineOperand &To) {

  if (!shouldSkip(*From.getParent()->succ_begin(), To.getMBB()))
    return;

  DebugLoc DL = From.getDebugLoc();
  BuildMI(*From.getParent(), &From, DL, TII->get(AMDGPU::S_CBRANCH_EXECZ))
          .addOperand(To)
          .addReg(AMDGPU::EXEC);
}

void SILowerControlFlowPass::SkipIfDead(MachineInstr &MI) {

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  if (MBB.getParent()->getInfo<SIMachineFunctionInfo>()->getShaderType() !=
      ShaderType::PIXEL ||
      !shouldSkip(&MBB, &MBB.getParent()->back()))
    return;

  MachineBasicBlock::iterator Insert = &MI;
  ++Insert;

  // If the exec mask is non-zero, skip the next two instructions
  BuildMI(MBB, Insert, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
          .addImm(3)
          .addReg(AMDGPU::EXEC);

  // Exec mask is zero: Export to NULL target...
  BuildMI(MBB, Insert, DL, TII->get(AMDGPU::EXP))
          .addImm(0)
          .addImm(0x09) // V_008DFC_SQ_EXP_NULL
          .addImm(0)
          .addImm(1)
          .addImm(1)
          .addReg(AMDGPU::VGPR0)
          .addReg(AMDGPU::VGPR0)
          .addReg(AMDGPU::VGPR0)
          .addReg(AMDGPU::VGPR0);

  // ... and terminate wavefront
  BuildMI(MBB, Insert, DL, TII->get(AMDGPU::S_ENDPGM));
}

void SILowerControlFlowPass::If(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Reg = MI.getOperand(0).getReg();
  unsigned Vcc = MI.getOperand(1).getReg();

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_AND_SAVEEXEC_B64), Reg)
          .addReg(Vcc);

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_XOR_B64), Reg)
          .addReg(AMDGPU::EXEC)
          .addReg(Reg);

  Skip(MI, MI.getOperand(2));

  MI.eraseFromParent();
}

void SILowerControlFlowPass::Else(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();

  BuildMI(MBB, MBB.getFirstNonPHI(), DL,
          TII->get(AMDGPU::S_OR_SAVEEXEC_B64), Dst)
          .addReg(Src); // Saved EXEC

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_XOR_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Dst);

  Skip(MI, MI.getOperand(2));

  MI.eraseFromParent();
}

void SILowerControlFlowPass::Break(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();
 
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_OR_B64), Dst)
          .addReg(AMDGPU::EXEC)
          .addReg(Src);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::IfBreak(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Vcc = MI.getOperand(1).getReg();
  unsigned Src = MI.getOperand(2).getReg();
 
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_OR_B64), Dst)
          .addReg(Vcc)
          .addReg(Src);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::ElseBreak(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Saved = MI.getOperand(1).getReg();
  unsigned Src = MI.getOperand(2).getReg();
 
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_OR_B64), Dst)
          .addReg(Saved)
          .addReg(Src);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::Loop(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Src = MI.getOperand(0).getReg();

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_ANDN2_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Src);

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
          .addOperand(MI.getOperand(1))
          .addReg(AMDGPU::EXEC);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::EndCf(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Reg = MI.getOperand(0).getReg();

  BuildMI(MBB, MBB.getFirstNonPHI(), DL,
          TII->get(AMDGPU::S_OR_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Reg);

  MI.eraseFromParent();
}

void SILowerControlFlowPass::Branch(MachineInstr &MI) {
  if (MI.getOperand(0).getMBB() == MI.getParent()->getNextNode())
    MI.eraseFromParent();

  // If these aren't equal, this is probably an infinite loop.
}

void SILowerControlFlowPass::Kill(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  const MachineOperand &Op = MI.getOperand(0);

#ifndef NDEBUG
  const SIMachineFunctionInfo *MFI
    = MBB.getParent()->getInfo<SIMachineFunctionInfo>();
  // Kill is only allowed in pixel / geometry shaders.
  assert(MFI->getShaderType() == ShaderType::PIXEL ||
         MFI->getShaderType() == ShaderType::GEOMETRY);
#endif

  // Clear this thread from the exec mask if the operand is negative
  if ((Op.isImm())) {
    // Constant operand: Set exec mask to 0 or do nothing
    if (Op.getImm() & 0x80000000) {
      BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_MOV_B64), AMDGPU::EXEC)
              .addImm(0);
    }
  } else {
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::V_CMPX_LE_F32_e32), AMDGPU::VCC)
           .addImm(0)
           .addOperand(Op);
  }

  MI.eraseFromParent();
}

void SILowerControlFlowPass::LoadM0(MachineInstr &MI, MachineInstr *MovRel, int Offset) {

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I = MI;

  unsigned Save = MI.getOperand(1).getReg();
  unsigned Idx = MI.getOperand(3).getReg();

  if (AMDGPU::SReg_32RegClass.contains(Idx)) {
    if (Offset) {
      BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
              .addReg(Idx)
              .addImm(Offset);
    } else {
      BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
              .addReg(Idx);
    }
    MBB.insert(I, MovRel);
  } else {

    assert(AMDGPU::SReg_64RegClass.contains(Save));
    assert(AMDGPU::VGPR_32RegClass.contains(Idx));

    // Save the EXEC mask
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_MOV_B64), Save)
            .addReg(AMDGPU::EXEC);

    // Read the next variant into VCC (lower 32 bits) <- also loop target
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32),
            AMDGPU::VCC_LO)
            .addReg(Idx);

    // Move index from VCC into M0
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
            .addReg(AMDGPU::VCC_LO);

    // Compare the just read M0 value to all possible Idx values
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::V_CMP_EQ_U32_e32), AMDGPU::VCC)
            .addReg(AMDGPU::M0)
            .addReg(Idx);

    // Update EXEC, save the original EXEC value to VCC
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_AND_SAVEEXEC_B64), AMDGPU::VCC)
            .addReg(AMDGPU::VCC);

    if (Offset) {
      BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
              .addReg(AMDGPU::M0)
              .addImm(Offset);
    }
    // Do the actual move
    MBB.insert(I, MovRel);

    // Update EXEC, switch all done bits to 0 and all todo bits to 1
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_XOR_B64), AMDGPU::EXEC)
            .addReg(AMDGPU::EXEC)
            .addReg(AMDGPU::VCC);

    // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
            .addImm(-7)
            .addReg(AMDGPU::EXEC);

    // Restore EXEC
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_MOV_B64), AMDGPU::EXEC)
            .addReg(Save);

  }
  MI.eraseFromParent();
}

/// \param @VecReg The register which holds element zero of the vector
///                 being addressed into.
/// \param[out] @Reg The base register to use in the indirect addressing instruction.
/// \param[in,out] @Offset As an input, this is the constant offset part of the
//                         indirect Index. e.g. v0 = v[VecReg + Offset]
//                         As an output, this is a constant value that needs
//                         to be added to the value stored in M0.
void SILowerControlFlowPass::computeIndirectRegAndOffset(unsigned VecReg,
                                                         unsigned &Reg,
                                                         int &Offset) {
  unsigned SubReg = TRI->getSubReg(VecReg, AMDGPU::sub0);
  if (!SubReg)
    SubReg = VecReg;

  const TargetRegisterClass *RC = TRI->getPhysRegClass(SubReg);
  int RegIdx = TRI->getHWRegIndex(SubReg) + Offset;

  if (RegIdx < 0) {
    Offset = RegIdx;
    RegIdx = 0;
  } else {
    Offset = 0;
  }

  Reg = RC->getRegister(RegIdx);
}

void SILowerControlFlowPass::IndirectSrc(MachineInstr &MI) {

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Vec = MI.getOperand(2).getReg();
  int Off = MI.getOperand(4).getImm();
  unsigned Reg;

  computeIndirectRegAndOffset(Vec, Reg, Off);

  MachineInstr *MovRel =
    BuildMI(*MBB.getParent(), DL, TII->get(AMDGPU::V_MOVRELS_B32_e32), Dst)
            .addReg(Reg)
            .addReg(AMDGPU::M0, RegState::Implicit)
            .addReg(Vec, RegState::Implicit);

  LoadM0(MI, MovRel, Off);
}

void SILowerControlFlowPass::IndirectDst(MachineInstr &MI) {

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  int Off = MI.getOperand(4).getImm();
  unsigned Val = MI.getOperand(5).getReg();
  unsigned Reg;

  computeIndirectRegAndOffset(Dst, Reg, Off);

  MachineInstr *MovRel = 
    BuildMI(*MBB.getParent(), DL, TII->get(AMDGPU::V_MOVRELD_B32_e32))
            .addReg(Reg, RegState::Define)
            .addReg(Val)
            .addReg(AMDGPU::M0, RegState::Implicit)
            .addReg(Dst, RegState::Implicit);

  LoadM0(MI, MovRel, Off);
}

bool SILowerControlFlowPass::runOnMachineFunction(MachineFunction &MF) {
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  bool HaveKill = false;
  bool NeedWQM = false;
  bool NeedFlat = false;
  unsigned Depth = 0;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    MachineBasicBlock::iterator I, Next;
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);

      MachineInstr &MI = *I;
      if (TII->isWQM(MI.getOpcode()) || TII->isDS(MI.getOpcode()))
        NeedWQM = true;

      // Flat uses m0 in case it needs to access LDS.
      if (TII->isFLAT(MI.getOpcode()))
        NeedFlat = true;

      switch (MI.getOpcode()) {
        default: break;
        case AMDGPU::SI_IF:
          ++Depth;
          If(MI);
          break;

        case AMDGPU::SI_ELSE:
          Else(MI);
          break;

        case AMDGPU::SI_BREAK:
          Break(MI);
          break;

        case AMDGPU::SI_IF_BREAK:
          IfBreak(MI);
          break;

        case AMDGPU::SI_ELSE_BREAK:
          ElseBreak(MI);
          break;

        case AMDGPU::SI_LOOP:
          ++Depth;
          Loop(MI);
          break;

        case AMDGPU::SI_END_CF:
          if (--Depth == 0 && HaveKill) {
            SkipIfDead(MI);
            HaveKill = false;
          }
          EndCf(MI);
          break;

        case AMDGPU::SI_KILL:
          if (Depth == 0)
            SkipIfDead(MI);
          else
            HaveKill = true;
          Kill(MI);
          break;

        case AMDGPU::S_BRANCH:
          Branch(MI);
          break;

        case AMDGPU::SI_INDIRECT_SRC:
          IndirectSrc(MI);
          break;

        case AMDGPU::SI_INDIRECT_DST_V1:
        case AMDGPU::SI_INDIRECT_DST_V2:
        case AMDGPU::SI_INDIRECT_DST_V4:
        case AMDGPU::SI_INDIRECT_DST_V8:
        case AMDGPU::SI_INDIRECT_DST_V16:
          IndirectDst(MI);
          break;
      }
    }
  }

  if (NeedWQM && MFI->getShaderType() == ShaderType::PIXEL) {
    MachineBasicBlock &MBB = MF.front();
    BuildMI(MBB, MBB.getFirstNonPHI(), DebugLoc(), TII->get(AMDGPU::S_WQM_B64),
            AMDGPU::EXEC).addReg(AMDGPU::EXEC);
  }

  // FIXME: This seems inappropriate to do here.
  if (NeedFlat && MFI->IsKernel) {
    // Insert the prologue initializing the SGPRs pointing to the scratch space
    // for flat accesses.
    const MachineFrameInfo *FrameInfo = MF.getFrameInfo();

    // TODO: What to use with function calls?

    // FIXME: This is reporting stack size that is used in a scratch buffer
    // rather than registers as well.
    uint64_t StackSizeBytes = FrameInfo->getStackSize();

    int IndirectBegin
      = static_cast<const AMDGPUInstrInfo*>(TII)->getIndirectIndexBegin(MF);
    // Convert register index to 256-byte unit.
    uint64_t StackOffset = IndirectBegin < 0 ? 0 : (4 * IndirectBegin / 256);

    assert((StackSizeBytes < 0xffff) && StackOffset < 0xffff &&
           "Stack limits should be smaller than 16-bits");

    // Initialize the flat scratch register pair.
    // TODO: Can we use one s_mov_b64 here?

    // Offset is in units of 256-bytes.
    MachineBasicBlock &MBB = MF.front();
    DebugLoc NoDL;
    MachineBasicBlock::iterator Start = MBB.getFirstNonPHI();
    const MCInstrDesc &SMovK = TII->get(AMDGPU::S_MOVK_I32);

    assert(isInt<16>(StackOffset) && isInt<16>(StackSizeBytes));

    BuildMI(MBB, Start, NoDL, SMovK, AMDGPU::FLAT_SCR_LO)
      .addImm(StackOffset);

    // Documentation says size is "per-thread scratch size in bytes"
    BuildMI(MBB, Start, NoDL, SMovK, AMDGPU::FLAT_SCR_HI)
      .addImm(StackSizeBytes);
  }

  return true;
}
