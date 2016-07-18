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

using namespace llvm;

#define DEBUG_TYPE "si-lower-control-flow"

namespace {

class SILowerControlFlow : public MachineFunctionPass {
private:
  static const unsigned SkipThreshold = 12;

  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;

  bool shouldSkip(MachineBasicBlock *From, MachineBasicBlock *To);

  void Skip(MachineInstr &From, MachineOperand &To);
  bool skipIfDead(MachineInstr &MI, MachineBasicBlock &NextBB);

  void If(MachineInstr &MI);
  void Else(MachineInstr &MI, bool ExecModified);
  void Break(MachineInstr &MI);
  void IfBreak(MachineInstr &MI);
  void ElseBreak(MachineInstr &MI);
  void Loop(MachineInstr &MI);
  void EndCf(MachineInstr &MI);

  void Kill(MachineInstr &MI);
  void Branch(MachineInstr &MI);

  MachineBasicBlock *insertSkipBlock(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  std::pair<MachineBasicBlock *, MachineBasicBlock *>
  splitBlock(MachineBasicBlock &MBB, MachineBasicBlock::iterator I);

  void splitLoadM0BlockLiveIns(LivePhysRegs &RemainderLiveRegs,
                               const MachineRegisterInfo &MRI,
                               const MachineInstr &MI,
                               MachineBasicBlock &LoopBB,
                               MachineBasicBlock &RemainderBB,
                               unsigned SaveReg,
                               const MachineOperand &IdxReg);

  void emitLoadM0FromVGPRLoop(MachineBasicBlock &LoopBB, DebugLoc DL,
                              MachineInstr *MovRel,
                              const MachineOperand &IdxReg,
                              int Offset);

  bool loadM0(MachineInstr &MI, MachineInstr *MovRel, int Offset = 0);
  std::pair<unsigned, int> computeIndirectRegAndOffset(unsigned VecReg,
                                                       int Offset) const;
  bool indirectSrc(MachineInstr &MI);
  bool indirectDst(MachineInstr &MI);

public:
  static char ID;

  SILowerControlFlow() :
    MachineFunctionPass(ID), TRI(nullptr), TII(nullptr) { }

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

void SILowerControlFlow::Skip(MachineInstr &From, MachineOperand &To) {

  if (!shouldSkip(*From.getParent()->succ_begin(), To.getMBB()))
    return;

  DebugLoc DL = From.getDebugLoc();
  BuildMI(*From.getParent(), &From, DL, TII->get(AMDGPU::S_CBRANCH_EXECZ))
    .addOperand(To);
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

  Skip(MI, MI.getOperand(2));

  // Insert a pseudo terminator to help keep the verifier happy.
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::SI_MASK_BRANCH))
    .addOperand(MI.getOperand(2))
    .addReg(Reg);

  MI.eraseFromParent();
}

void SILowerControlFlow::Else(MachineInstr &MI, bool ExecModified) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  unsigned Dst = MI.getOperand(0).getReg();
  unsigned Src = MI.getOperand(1).getReg();

  BuildMI(MBB, MBB.getFirstNonPHI(), DL,
          TII->get(AMDGPU::S_OR_SAVEEXEC_B64), Dst)
          .addReg(Src); // Saved EXEC

  if (ExecModified) {
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

  Skip(MI, MI.getOperand(2));

  // Insert a pseudo terminator to help keep the verifier happy.
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::SI_MASK_BRANCH))
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

// All currently live registers must remain so in the remainder block.
void SILowerControlFlow::splitLoadM0BlockLiveIns(LivePhysRegs &RemainderLiveRegs,
                                                 const MachineRegisterInfo &MRI,
                                                 const MachineInstr &MI,
                                                 MachineBasicBlock &LoopBB,
                                                 MachineBasicBlock &RemainderBB,
                                                 unsigned SaveReg,
                                                 const MachineOperand &IdxReg) {
  // Add reg defined in loop body.
  RemainderLiveRegs.addReg(SaveReg);

  if (const MachineOperand *Val = TII->getNamedOperand(MI, AMDGPU::OpName::val)) {
    if (!Val->isUndef()) {
      RemainderLiveRegs.addReg(Val->getReg());
      LoopBB.addLiveIn(Val->getReg());
    }
  }

  for (unsigned Reg : RemainderLiveRegs) {
    if (MRI.isAllocatable(Reg))
      RemainderBB.addLiveIn(Reg);
  }

  const MachineOperand *Src = TII->getNamedOperand(MI, AMDGPU::OpName::src);
  if (!Src->isUndef())
    LoopBB.addLiveIn(Src->getReg());

  if (!IdxReg.isUndef())
    LoopBB.addLiveIn(IdxReg.getReg());
  LoopBB.sortUniqueLiveIns();
}

void SILowerControlFlow::emitLoadM0FromVGPRLoop(MachineBasicBlock &LoopBB,
                                                DebugLoc DL,
                                                MachineInstr *MovRel,
                                                const MachineOperand &IdxReg,
                                                int Offset) {
  MachineBasicBlock::iterator I = LoopBB.begin();

  // Read the next variant into VCC (lower 32 bits) <- also loop target
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), AMDGPU::VCC_LO)
    .addReg(IdxReg.getReg(), getUndefRegState(IdxReg.isUndef()));

  // Move index from VCC into M0
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
    .addReg(AMDGPU::VCC_LO);

  // Compare the just read M0 value to all possible Idx values
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::V_CMP_EQ_U32_e32))
    .addReg(AMDGPU::M0)
    .addReg(IdxReg.getReg(), getUndefRegState(IdxReg.isUndef()));

  // Update EXEC, save the original EXEC value to VCC
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_AND_SAVEEXEC_B64), AMDGPU::VCC)
    .addReg(AMDGPU::VCC);

  if (Offset != 0) {
    BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
      .addReg(AMDGPU::M0)
      .addImm(Offset);
  }

  // Do the actual move
  LoopBB.insert(I, MovRel);

  // Update EXEC, switch all done bits to 0 and all todo bits to 1
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_XOR_B64), AMDGPU::EXEC)
    .addReg(AMDGPU::EXEC)
    .addReg(AMDGPU::VCC);

  // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover
  BuildMI(LoopBB, I, DL, TII->get(AMDGPU::S_CBRANCH_EXECNZ))
    .addMBB(&LoopBB);
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

std::pair<MachineBasicBlock *, MachineBasicBlock *>
SILowerControlFlow::splitBlock(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I) {
  MachineFunction *MF = MBB.getParent();

  // To insert the loop we need to split the block. Move everything after this
  // point to a new block, and insert a new empty block between the two.
  MachineBasicBlock *LoopBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *RemainderBB = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;

  MF->insert(MBBI, LoopBB);
  MF->insert(MBBI, RemainderBB);

  // Move the rest of the block into a new block.
  RemainderBB->transferSuccessors(&MBB);
  RemainderBB->splice(RemainderBB->begin(), &MBB, I, MBB.end());

  MBB.addSuccessor(LoopBB);

  return std::make_pair(LoopBB, RemainderBB);
}

// Returns true if a new block was inserted.
bool SILowerControlFlow::loadM0(MachineInstr &MI, MachineInstr *MovRel, int Offset) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);

  if (AMDGPU::SReg_32RegClass.contains(Idx->getReg())) {
    if (Offset != 0) {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADD_I32), AMDGPU::M0)
        .addReg(Idx->getReg(), getUndefRegState(Idx->isUndef()))
        .addImm(Offset);
    } else {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), AMDGPU::M0)
        .addReg(Idx->getReg(), getUndefRegState(Idx->isUndef()));
    }

    MBB.insert(I, MovRel);
    MI.eraseFromParent();
    return false;
  }

  MachineOperand *SaveOp = TII->getNamedOperand(MI, AMDGPU::OpName::sdst);
  SaveOp->setIsDead(false);
  unsigned Save = SaveOp->getReg();

  // Reading from a VGPR requires looping over all workitems in the wavefront.
  assert(AMDGPU::SReg_64RegClass.contains(Save) &&
         AMDGPU::VGPR_32RegClass.contains(Idx->getReg()));

  // Save the EXEC mask
  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B64), Save)
    .addReg(AMDGPU::EXEC);

  LivePhysRegs RemainderLiveRegs(TRI);

  RemainderLiveRegs.addLiveOuts(MBB);

  MachineBasicBlock *LoopBB;
  MachineBasicBlock *RemainderBB;

  std::tie(LoopBB, RemainderBB) = splitBlock(MBB, I);

  for (const MachineInstr &Inst : reverse(*RemainderBB))
    RemainderLiveRegs.stepBackward(Inst);

  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  LoopBB->addSuccessor(RemainderBB);
  LoopBB->addSuccessor(LoopBB);

  splitLoadM0BlockLiveIns(RemainderLiveRegs, MRI, MI, *LoopBB,
                          *RemainderBB, Save, *Idx);

  emitLoadM0FromVGPRLoop(*LoopBB, DL, MovRel, *Idx, Offset);

  MachineBasicBlock::iterator First = RemainderBB->begin();
  BuildMI(*RemainderBB, First, DL, TII->get(AMDGPU::S_MOV_B64), AMDGPU::EXEC)
    .addReg(Save);

  MI.eraseFromParent();
  return true;
}

/// \param @VecReg The register which holds element zero of the vector being
///                 addressed into.
//
/// \param[in] @Idx The index operand from the movrel instruction. This must be
// a register, but may be NoRegister.
///
/// \param[in] @Offset As an input, this is the constant offset part of the
// indirect Index. e.g. v0 = v[VecReg + Offset] As an output, this is a constant
// value that needs to be added to the value stored in M0.
std::pair<unsigned, int>
SILowerControlFlow::computeIndirectRegAndOffset(unsigned VecReg, int Offset) const {
  unsigned SubReg = TRI->getSubReg(VecReg, AMDGPU::sub0);
  if (!SubReg)
    SubReg = VecReg;

  const TargetRegisterClass *SuperRC = TRI->getPhysRegClass(VecReg);
  const TargetRegisterClass *RC = TRI->getPhysRegClass(SubReg);
  int NumElts = SuperRC->getSize() / RC->getSize();

  int BaseRegIdx = TRI->getHWRegIndex(SubReg);

  // Skip out of bounds offsets, or else we would end up using an undefined
  // register.
  if (Offset >= NumElts)
    return std::make_pair(RC->getRegister(BaseRegIdx), Offset);

  int RegIdx = BaseRegIdx + Offset;
  if (RegIdx < 0) {
    Offset = RegIdx;
    RegIdx = 0;
  } else {
    Offset = 0;
  }

  unsigned Reg = RC->getRegister(RegIdx);
  return std::make_pair(Reg, Offset);
}

// Return true if a new block was inserted.
bool SILowerControlFlow::indirectSrc(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  const MachineOperand *SrcVec = TII->getNamedOperand(MI, AMDGPU::OpName::src);
  int Offset = TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm();
  unsigned Reg;

  std::tie(Reg, Offset) = computeIndirectRegAndOffset(SrcVec->getReg(), Offset);

  const MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
  if (Idx->getReg() == AMDGPU::NoRegister) {
    // Only had a constant offset, copy the register directly.
    BuildMI(MBB, MI.getIterator(), DL, TII->get(AMDGPU::V_MOV_B32_e32), Dst)
      .addReg(Reg, getUndefRegState(SrcVec->isUndef()));
    MI.eraseFromParent();
    return false;
  }

  MachineInstr *MovRel =
    BuildMI(*MBB.getParent(), DL, TII->get(AMDGPU::V_MOVRELS_B32_e32), Dst)
    .addReg(Reg, getUndefRegState(SrcVec->isUndef()))
    .addReg(SrcVec->getReg(), RegState::Implicit);

  return loadM0(MI, MovRel, Offset);
}

// Return true if a new block was inserted.
bool SILowerControlFlow::indirectDst(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  unsigned Dst = MI.getOperand(0).getReg();
  int Offset = TII->getNamedOperand(MI, AMDGPU::OpName::offset)->getImm();
  unsigned Reg;

  const MachineOperand *Val = TII->getNamedOperand(MI, AMDGPU::OpName::val);
  std::tie(Reg, Offset) = computeIndirectRegAndOffset(Dst, Offset);

  MachineOperand *Idx = TII->getNamedOperand(MI, AMDGPU::OpName::idx);
  if (Idx->getReg() == AMDGPU::NoRegister) {
    // Only had a constant offset, copy the register directly.
    BuildMI(MBB, MI.getIterator(), DL, TII->get(AMDGPU::V_MOV_B32_e32), Reg)
      .addOperand(*Val);
    MI.eraseFromParent();
    return false;
  }

  MachineInstr *MovRel =
    BuildMI(*MBB.getParent(), DL, TII->get(AMDGPU::V_MOVRELD_B32_e32), Reg)
    .addReg(Val->getReg(), getUndefRegState(Val->isUndef()))
    .addReg(Dst, RegState::Implicit);

  return loadM0(MI, MovRel, Offset);
}

bool SILowerControlFlow::runOnMachineFunction(MachineFunction &MF) {
  const SISubtarget &ST = MF.getSubtarget<SISubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  bool HaveKill = false;
  bool NeedFlat = false;
  unsigned Depth = 0;

  MachineFunction::iterator NextBB;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; BI = NextBB) {
    NextBB = std::next(BI);
    MachineBasicBlock &MBB = *BI;

    MachineBasicBlock *EmptyMBBAtEnd = nullptr;
    MachineBasicBlock::iterator I, Next;
    bool ExecModified = false;

    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);

      MachineInstr &MI = *I;

      // Flat uses m0 in case it needs to access LDS.
      if (TII->isFLAT(MI))
        NeedFlat = true;

      if (I->modifiesRegister(AMDGPU::EXEC, TRI))
        ExecModified = true;

      switch (MI.getOpcode()) {
        default: break;
        case AMDGPU::SI_IF:
          ++Depth;
          If(MI);
          break;

        case AMDGPU::SI_ELSE:
          Else(MI, ExecModified);
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

        case AMDGPU::SI_INDIRECT_SRC_V1:
        case AMDGPU::SI_INDIRECT_SRC_V2:
        case AMDGPU::SI_INDIRECT_SRC_V4:
        case AMDGPU::SI_INDIRECT_SRC_V8:
        case AMDGPU::SI_INDIRECT_SRC_V16:
          if (indirectSrc(MI)) {
            // The block was split at this point. We can safely skip the middle
            // inserted block to the following which contains the rest of this
            // block's instructions.
            NextBB = std::next(BI);
            BE = MF.end();
            Next = MBB.end();
          }

          break;

        case AMDGPU::SI_INDIRECT_DST_V1:
        case AMDGPU::SI_INDIRECT_DST_V2:
        case AMDGPU::SI_INDIRECT_DST_V4:
        case AMDGPU::SI_INDIRECT_DST_V8:
        case AMDGPU::SI_INDIRECT_DST_V16:
          if (indirectDst(MI)) {
            // The block was split at this point. We can safely skip the middle
            // inserted block to the following which contains the rest of this
            // block's instructions.
            NextBB = std::next(BI);
            BE = MF.end();
            Next = MBB.end();
          }

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

  if (NeedFlat && MFI->IsKernel) {
    // TODO: What to use with function calls?
    // We will need to Initialize the flat scratch register pair.
    if (NeedFlat)
      MFI->setHasFlatInstructions(true);
  }

  return true;
}
