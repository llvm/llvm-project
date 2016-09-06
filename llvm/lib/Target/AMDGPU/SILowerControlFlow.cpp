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
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/MCAsmInfo.h"

using namespace llvm;

#define DEBUG_TYPE "si-lower-control-flow"

namespace {

static cl::opt<unsigned> SkipThresholdFlag(
  "amdgpu-skip-threshold",
  cl::desc("Number of instructions before jumping over divergent control flow"),
  cl::init(12), cl::Hidden);

class SILowerControlFlow : public MachineFunctionPass {
private:
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  unsigned SkipThreshold;

  bool shouldSkip(MachineBasicBlock *From, MachineBasicBlock *To);

  MachineInstr *Skip(MachineInstr &From, MachineOperand &To);
  bool skipIfDead(MachineInstr &MI, MachineBasicBlock &NextBB);

  void If(MachineInstr &MI);
  void Else(MachineInstr &MI);
  void Break(MachineInstr &MI);
  void IfBreak(MachineInstr &MI);
  void ElseBreak(MachineInstr &MI);
  void Loop(MachineInstr &MI);
  void EndCf(MachineInstr &MI);

  void Kill(MachineInstr &MI);
  void Branch(MachineInstr &MI);

  MachineBasicBlock *insertSkipBlock(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;
public:
  static char ID;

  SILowerControlFlow() :
    MachineFunctionPass(ID), TRI(nullptr), TII(nullptr), SkipThreshold(0) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Lower control flow pseudo instructions";
  }
};

} // End anonymous namespace

char SILowerControlFlow::ID = 0;

INITIALIZE_PASS(SILowerControlFlow, DEBUG_TYPE,
                "SI lower control flow", false, false)

char &llvm::SILowerControlFlowPassID = SILowerControlFlow::ID;


FunctionPass *llvm::createSILowerControlFlowPass() {
  return new SILowerControlFlow();
}

static bool opcodeEmitsNoInsts(unsigned Opc) {
  switch (Opc) {
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::BUNDLE:
  case TargetOpcode::CFI_INSTRUCTION:
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::GC_LABEL:
  case TargetOpcode::DBG_VALUE:
    return true;
  default:
    return false;
  }
}

bool SILowerControlFlow::shouldSkip(MachineBasicBlock *From,
                                    MachineBasicBlock *To) {
  if (From->succ_empty())
    return false;

  unsigned NumInstr = 0;
  MachineFunction *MF = From->getParent();

  for (MachineFunction::iterator MBBI(From), ToI(To), End = MF->end();
       MBBI != End && MBBI != ToI; ++MBBI) {
    MachineBasicBlock &MBB = *MBBI;

    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         NumInstr < SkipThreshold && I != E; ++I) {
      if (opcodeEmitsNoInsts(I->getOpcode()))
        continue;

      // When a uniform loop is inside non-uniform control flow, the branch
      // leaving the loop might be an S_CBRANCH_VCCNZ, which is never taken
      // when EXEC = 0. We should skip the loop lest it becomes infinite.
      if (I->getOpcode() == AMDGPU::S_CBRANCH_VCCNZ ||
          I->getOpcode() == AMDGPU::S_CBRANCH_VCCZ)
        return true;

      if (I->isInlineAsm()) {
        const MCAsmInfo *MAI = MF->getTarget().getMCAsmInfo();
        const char *AsmStr = I->getOperand(0).getSymbolName();

        // inlineasm length estimate is number of bytes assuming the longest
        // instruction.
        uint64_t MaxAsmSize = TII->getInlineAsmLength(AsmStr, *MAI);
        NumInstr += MaxAsmSize / MAI->getMaxInstLength();
      } else {
        ++NumInstr;
      }

      if (NumInstr >= SkipThreshold)
        return true;
    }
  }

  return false;
}

MachineInstr *SILowerControlFlow::Skip(MachineInstr &From, MachineOperand &To) {
  if (!shouldSkip(*From.getParent()->succ_begin(), To.getMBB()))
    return nullptr;

  const DebugLoc &DL = From.getDebugLoc();
  MachineInstr *Skip =
    BuildMI(*From.getParent(), &From, DL, TII->get(AMDGPU::S_CBRANCH_EXECZ))
    .addOperand(To);
  return Skip;
}

bool SILowerControlFlow::skipIfDead(MachineInstr &MI, MachineBasicBlock &NextBB) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction *MF = MBB.getParent();

  if (MF->getFunction()->getCallingConv() != CallingConv::AMDGPU_PS ||
      !shouldSkip(&MBB, &MBB.getParent()->back()))
    return false;

  MachineBasicBlock *SkipBB = insertSkipBlock(MBB, MI.getIterator());
  MBB.addSuccessor(SkipBB);

  const DebugLoc &DL = MI.getDebugLoc();

  // If the exec mask is non-zero, skip the next two instructions
  BuildMI(&MBB, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
    .addMBB(&NextBB);

  MachineBasicBlock::iterator Insert = SkipBB->begin();

  // Exec mask is zero: Export to NULL target...
  BuildMI(*SkipBB, Insert, DL, TII->get(AMDGPU::EXP))
    .addImm(0)
    .addImm(0x09) // V_008DFC_SQ_EXP_NULL
    .addImm(0)
    .addImm(1)
    .addImm(1)
    .addReg(AMDGPU::VGPR0, RegState::Undef)
    .addReg(AMDGPU::VGPR0, RegState::Undef)
    .addReg(AMDGPU::VGPR0, RegState::Undef)
    .addReg(AMDGPU::VGPR0, RegState::Undef);

  // ... and terminate wavefront.
  BuildMI(*SkipBB, Insert, DL, TII->get(AMDGPU::S_ENDPGM));

  return true;
}

void SILowerControlFlow::If(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Reg = MI.getOperand(0).getReg();
  unsigned Vcc = MI.getOperand(1).getReg();

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_AND_SAVEEXEC_B64), Reg)
          .addReg(Vcc);

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_XOR_B64), Reg)
          .addReg(AMDGPU::EXEC)
          .addReg(Reg);

  MachineInstr *SkipInst = Skip(MI, MI.getOperand(2));

  // Insert before the new branch instruction.
  MachineInstr *InsPt = SkipInst ? SkipInst : &MI;

  // Insert a pseudo terminator to help keep the verifier happy.
  BuildMI(MBB, InsPt, DL, TII->get(AMDGPU::SI_MASK_BRANCH))
    .addOperand(MI.getOperand(2))
    .addReg(Reg);

  MI.eraseFromParent();
}

void SILowerControlFlow::Else(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();

  BuildMI(MBB, MBB.getFirstNonPHI(), DL,
          TII->get(AMDGPU::S_OR_SAVEEXEC_B64), Dst)
          .addReg(Src); // Saved EXEC

  if (MI.getOperand(3).getImm() != 0) {
    // Adjust the saved exec to account for the modifications during the flow
    // block that contains the ELSE. This can happen when WQM mode is switched
    // off.
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_AND_B64), Dst)
            .addReg(AMDGPU::EXEC)
            .addReg(Dst);
  }

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_XOR_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Dst);

  MachineInstr *SkipInst = Skip(MI, MI.getOperand(2));

  // Insert before the new branch instruction.
  MachineInstr *InsPt = SkipInst ? SkipInst : &MI;

  // Insert a pseudo terminator to help keep the verifier happy.
  BuildMI(MBB, InsPt, DL, TII->get(AMDGPU::SI_MASK_BRANCH))
    .addOperand(MI.getOperand(2))
    .addReg(Dst);

  MI.eraseFromParent();
}

void SILowerControlFlow::Break(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_OR_B64), Dst)
          .addReg(AMDGPU::EXEC)
          .addReg(Src);

  MI.eraseFromParent();
}

void SILowerControlFlow::IfBreak(MachineInstr &MI) {
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

void SILowerControlFlow::ElseBreak(MachineInstr &MI) {
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

void SILowerControlFlow::Loop(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Src = MI.getOperand(0).getReg();

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_ANDN2_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Src);

  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
    .addOperand(MI.getOperand(1));

  MI.eraseFromParent();
}

void SILowerControlFlow::EndCf(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Reg = MI.getOperand(0).getReg();

  BuildMI(MBB, MBB.getFirstNonPHI(), DL,
          TII->get(AMDGPU::S_OR_B64), AMDGPU::EXEC)
          .addReg(AMDGPU::EXEC)
          .addReg(Reg);

  MI.eraseFromParent();
}

void SILowerControlFlow::Branch(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getOperand(0).getMBB();
  if (MBB == MI.getParent()->getNextNode())
    MI.eraseFromParent();

  // If these aren't equal, this is probably an infinite loop.
}

void SILowerControlFlow::Kill(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  const MachineOperand &Op = MI.getOperand(0);

#ifndef NDEBUG
  CallingConv::ID CallConv = MBB.getParent()->getFunction()->getCallingConv();
  // Kill is only allowed in pixel / geometry shaders.
  assert(CallConv == CallingConv::AMDGPU_PS ||
         CallConv == CallingConv::AMDGPU_GS);
#endif

  // Clear this thread from the exec mask if the operand is negative
  if ((Op.isImm())) {
    // Constant operand: Set exec mask to 0 or do nothing
    if (Op.getImm() & 0x80000000) {
      BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_MOV_B64), AMDGPU::EXEC)
              .addImm(0);
    }
  } else {
    BuildMI(MBB, &MI, DL, TII->get(AMDGPU::V_CMPX_LE_F32_e32))
           .addImm(0)
           .addOperand(Op);
  }

  MI.eraseFromParent();
}

MachineBasicBlock *SILowerControlFlow::insertSkipBlock(
  MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const {
  MachineFunction *MF = MBB.getParent();

  MachineBasicBlock *SkipBB = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;

  MF->insert(MBBI, SkipBB);

  return SkipBB;
}

bool SILowerControlFlow::runOnMachineFunction(MachineFunction &MF) {
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();
  SkipThreshold = SkipThresholdFlag;

  bool HaveKill = false;
  unsigned Depth = 0;

  MachineFunction::iterator NextBB;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; BI = NextBB) {
    NextBB = std::next(BI);
    MachineBasicBlock &MBB = *BI;

    MachineBasicBlock *EmptyMBBAtEnd = nullptr;
    MachineBasicBlock::iterator I, Next;

    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);

      MachineInstr &MI = *I;

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
            HaveKill = false;
            // TODO: Insert skip if exec is 0?
          }

          EndCf(MI);
          break;

        case AMDGPU::SI_KILL_TERMINATOR:
          if (Depth == 0) {
            if (skipIfDead(MI, *NextBB)) {
              NextBB = std::next(BI);
              BE = MF.end();
            }
          } else
            HaveKill = true;
          Kill(MI);
          break;

        case AMDGPU::S_BRANCH:
          Branch(MI);
          break;

        case AMDGPU::SI_RETURN: {
          assert(!MF.getInfo<SIMachineFunctionInfo>()->returnsVoid());

          // Graphics shaders returning non-void shouldn't contain S_ENDPGM,
          // because external bytecode will be appended at the end.
          if (BI != --MF.end() || I != MBB.getFirstTerminator()) {
            // SI_RETURN is not the last instruction. Add an empty block at
            // the end and jump there.
            if (!EmptyMBBAtEnd) {
              EmptyMBBAtEnd = MF.CreateMachineBasicBlock();
              MF.insert(MF.end(), EmptyMBBAtEnd);
            }

            MBB.addSuccessor(EmptyMBBAtEnd);
            BuildMI(*BI, I, MI.getDebugLoc(), TII->get(AMDGPU::S_BRANCH))
                    .addMBB(EmptyMBBAtEnd);
            I->eraseFromParent();
          }
          break;
        }
      }
    }
  }
  return true;
}
