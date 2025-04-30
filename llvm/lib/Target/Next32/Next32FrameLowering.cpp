//===-- Next32FrameLowering.cpp - Next32 Frame Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Next32 implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "Next32FrameLowering.h"
#include "Next32InstrInfo.h"
#include "Next32MachineFunctionInfo.h"
#include "Next32Subtarget.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"

#define DEBUG_TYPE "next32-framelowering"

using namespace llvm;

Next32FrameLowering::Next32FrameLowering(const Next32Subtarget &sti)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsUp,
                          /* StackAl= */ Align(16),
                          /* LAO= */ 0,
                          /* TransAl= */ Align(16)) {}

bool Next32FrameLowering::hasFP(const MachineFunction &MF) const {
  return true;
}

StackOffset
Next32FrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                            Register &FrameReg) const {
  // If stack direction changed to StackGrowsDown, delete the override the
  // default implementation should be good enough (the issue with it is exactly
  // that assumption.)
  assert(getStackGrowthDirection() == TargetFrameLowering::StackGrowsUp);

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();

  // By default, assume all frame indices are referenced via whatever
  // getFrameRegister() says. The target can override this if it's doing
  // something different.
  FrameReg = RI->getFrameRegister(MF);

  return StackOffset::getFixed(MFI.getObjectOffset(FI) +
                               getOffsetOfLocalArea() +
                               MFI.getOffsetAdjustment());
}

MachineBasicBlock::iterator
Next32FrameLowering::addArgumentFeeders(MachineFunction &MF,
                                        MachineBasicBlock &MBB) const {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineBasicBlock::iterator FeederPos = MBB.begin();
  MachineInstr *FeederArgs = nullptr;
  DebugLoc dl;

  // Skipping implicit FEEDERs such as TID and RET_FID.
  while (FeederPos->getOpcode() == Next32::FEEDER)
    ++FeederPos;

  for (auto &MI : MBB) {
    if (MI.getOpcode() == Next32::FEEDER_ARGS) {
      FeederArgs = &MI;
      break;
    }
  }

  if (!FeederArgs)
    report_fatal_error("FEEDER_ARGS pseudo instrucion doesn't exist!");

  for (unsigned i = 0; i < FeederArgs->getNumOperands(); i += 2) {
    unsigned int Reg = FeederArgs->getOperand(i).getImm();
    unsigned int FeederSize = FeederArgs->getOperand(i + 1).getImm();
    BuildMI(MBB, FeederPos, dl, TII.get(Next32::FEEDER), Reg)
        .addReg(Reg)
        .addImm(FeederSize);
  }

  // Increment iterator in case when we are pointing to FEEDER_ARGS
  // since FEEDER_ARGS will be removed.
  if (FeederPos->getOpcode() == Next32::FEEDER_ARGS)
    FeederPos++;

  FeederArgs->removeFromParent();
  return FeederPos;
}

void Next32FrameLowering::debugMarkParameterFeeders(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MBBI) const {
  const Next32InstrInfo &TII =
      *MF.getSubtarget<Next32Subtarget>().getInstrInfo();

  if (MBBI == MBB.end())
    return;

  std::vector<MachineInstr *> RegDbgVals(TII.getRegisterInfo().getNumRegs());
  auto UpdateRegDbgVal = [](MachineInstr *&DbgVal, MachineInstr &I) {
    if (!DbgVal)
      DbgVal = &I;
  };

  for (MachineBasicBlock::iterator I = MBBI, E = MBB.end(); I != E; ++I) {
    if (!I->isDebugValue() || !I->getDebugVariable()->isParameter() ||
        !I->getOperand(0).isReg())
      continue;

    Register DbgReg = I->getOperand(0).getReg();
    UpdateRegDbgVal(RegDbgVals[DbgReg], *I);
  }
  for (MachineBasicBlock::iterator S = MBB.begin(), E = std::next(MBBI); S != E;
       ++S) {
    if (S->getOpcode() != Next32::FEEDER)
      continue;

    Register FeederOutputReg = S->getOperand(0).getReg();
    MachineInstr *I = RegDbgVals[FeederOutputReg];
    if (!I)
      continue;

    BuildMI(MBB, S, I->getDebugLoc(), TII.get(TargetOpcode::DBG_VALUE),
            I->isIndirectDebugValue(), I->getOperand(0), I->getDebugVariable(),
            I->getDebugExpression());
    LLVM_DEBUG(dbgs() << "\tProlouge creating debug value for reg: "
                      << FeederOutputReg << "\n");
  }
}

void Next32FrameLowering::emitPrologue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc dl;
  BuildMI(MBB, MBBI, dl, TII.get(Next32::FEEDER), Next32::TID)
      .addReg(Next32::TID)
      .addImm(llvm::Next32Constants::InstructionSize::InstructionSize32);

  BuildMI(MBB, MBBI, dl, TII.get(Next32::FEEDER), Next32::RET_FID)
      .addReg(Next32::RET_FID)
      .addImm(llvm::Next32Constants::InstructionSize::InstructionSize32);

  MBBI = addArgumentFeeders(MF, MBB);

  if (MF.getFunction().isVarArg()) {
    unsigned int FirstReg = Next32::VA_LOW;
    unsigned int SecondReg = Next32::VA_HIGH;

    if (MF.getDataLayout().isBigEndian())
      std::swap(FirstReg, SecondReg);

    BuildMI(MBB, MBBI, dl, TII.get(Next32::FEEDER), FirstReg)
        .addReg(FirstReg)
        .addImm(llvm::Next32Constants::InstructionSize::InstructionSize64);
    BuildMI(MBB, MBBI, dl, TII.get(Next32::FEEDER), SecondReg)
        .addReg(SecondReg)
        .addImm(llvm::Next32Constants::InstructionSize::InstructionSize64);
  }
  debugMarkParameterFeeders(MF, MBB, MBBI);

  MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t StackSize = MFI.getStackSize();

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const bool RealignStack = TRI->shouldRealignStack(MF);
  const unsigned Alignment = MFI.getMaxAlign().value();

  if (StackSize == 0 && !MFI.hasVarSizedObjects() && !RealignStack)
    return;

  auto *NFI = MF.getInfo<Next32MachineFunctionInfo>();
  NFI->setHasTopLevelStackFrame();

  if (RealignStack) {
    StackSize += Alignment;
    MFI.setStackSize(StackSize);
  }

  // Going to use StackSize as a imm for 32 bit register, validate
  // it will not truncate the value.
  if (StackSize > std::numeric_limits<uint32_t>::max()) {
    report_fatal_error("Next32 - function stack size is over 4GB");
  }

  BuildMI(MBB, MBBI, dl, TII.get(Next32::MOVL), Next32::STACK_SIZE)
      .addImm(StackSize);
  BuildMI(MBB, MBBI, dl, TII.get(Next32::SETFRAME))
      .addReg(Next32::SP_HIGH, RegState::Define)
      .addReg(Next32::SP_LOW, RegState::Define)
      .addReg(Next32::STACK_SIZE, RegState::Define)
      .addReg(Next32::STACK_SIZE)
      .addReg(Next32::TID);
  BuildMI(MBB, MBBI, dl, TII.get(Next32::BARRIER), Next32::TID)
      .addReg(Next32::TID)
      .addReg(Next32::STACK_SIZE);

  if (RealignStack) {
    assert(isPowerOf2_32(Alignment));
    assert(getStackGrowthDirection() == TargetFrameLowering::StackGrowsUp);

    // STACK_BIAS = SP_LOW
    TII.copyPhysReg(MBB, MBBI, dl, Next32::STACK_BIAS, Next32::SP_LOW, false);

    // SP_LOW += ALIGNMENT - 1 and remember CARRY
    BuildMI(MBB, MBBI, dl, TII.get(Next32::MOVL), Next32::SCRATCH1)
        .addImm(Alignment - 1);
    BuildMI(MBB, MBBI, dl, TII.get(Next32::ADD), Next32::SP_LOW)
        .addReg(Next32::SP_LOW)
        .addReg(Next32::SCRATCH1);
    TII.copyPhysReg(MBB, MBBI, dl, Next32::SCRATCH2, Next32::SP_LOW, false);
    BuildMI(MBB, MBBI, dl, TII.get(Next32::FLAGS), Next32::SCRATCH2)
        .addReg(Next32::SCRATCH2);

    // SP_LOW &= ~(ALIGNMENT - 1)
    BuildMI(MBB, MBBI, dl, TII.get(Next32::MOVL), Next32::SCRATCH1)
        .addImm(~(Alignment - 1));
    BuildMI(MBB, MBBI, dl, TII.get(Next32::AND), Next32::SP_LOW)
        .addReg(Next32::SP_LOW)
        .addReg(Next32::SCRATCH1);

    // STACK_BIAS -= SP_LOW
    BuildMI(MBB, MBBI, dl, TII.get(Next32::SUB), Next32::STACK_BIAS)
        .addReg(Next32::STACK_BIAS)
        .addReg(Next32::SP_LOW);

    // SP_HIGH += 0 + CARRY
    // Zero is used as an immediate for the high-part because the maximum added
    // value is (Alignment-1) which is bound to 2^32
    BuildMI(MBB, MBBI, dl, TII.get(Next32::MOVL), Next32::SCRATCH1).addImm(0);
    BuildMI(MBB, MBBI, dl, TII.get(Next32::ADC), Next32::SP_HIGH)
        .addReg(Next32::SP_HIGH)
        .addReg(Next32::SCRATCH1, RegState::Kill)
        .addReg(Next32::SCRATCH2, RegState::Kill);
  }
}

void Next32FrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.end();
  --MBBI;

  DebugLoc dl = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();

  auto *NFI = MF.getInfo<Next32MachineFunctionInfo>();
  if (!NFI->hasTopLevelStackFrame())
    return;

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const bool RealignStack = TRI->shouldRealignStack(MF);

  for (auto &MBBI : MF) {
    MBBI.addLiveIn(Next32::SP_HIGH);
    MBBI.addLiveIn(Next32::SP_LOW);
    if (RealignStack)
      MBBI.addLiveIn(Next32::STACK_BIAS);
  }

  if (RealignStack) {
    assert(getStackGrowthDirection() == TargetFrameLowering::StackGrowsUp);

    // Remember high part of sign extended STACK_BIAS
    TII.copyPhysReg(MBB, MBBI, dl, Next32::SCRATCH1, Next32::STACK_BIAS, false);
    BuildMI(MBB, MBBI, dl, TII.get(Next32::MOVL), Next32::SCRATCH2).addImm(31);
    BuildMI(MBB, MBBI, dl, TII.get(Next32::SHRI), Next32::SCRATCH1)
        .addReg(Next32::SCRATCH1)
        .addReg(Next32::SCRATCH2, RegState::Kill);

    // SP_LOW += STACK_BIAS and remember CARRY
    BuildMI(MBB, MBBI, dl, TII.get(Next32::ADD), Next32::SP_LOW)
        .addReg(Next32::SP_LOW)
        .addReg(Next32::STACK_BIAS, RegState::Kill);
    TII.copyPhysReg(MBB, MBBI, dl, Next32::SCRATCH2, Next32::SP_LOW, false);
    BuildMI(MBB, MBBI, dl, TII.get(Next32::FLAGS), Next32::SCRATCH2)
        .addReg(Next32::SCRATCH2);

    // SP_HIGH += (High part of sign extended STACK_BIAS) + CARRY
    BuildMI(MBB, MBBI, dl, TII.get(Next32::ADC), Next32::SP_HIGH)
        .addReg(Next32::SP_HIGH)
        .addReg(Next32::SCRATCH1, RegState::Kill)
        .addReg(Next32::SCRATCH2, RegState::Kill);
  }

  BuildMI(MBB, MBBI, dl, TII.get(Next32::RESETFRAME), Next32::SP_HIGH)
      .addReg(Next32::SP_HIGH)
      .addReg(Next32::SP_LOW)
      .addReg(Next32::TID);
  BuildMI(MBB, MBBI, dl, TII.get(Next32::BARRIER), Next32::TID)
      .addReg(Next32::TID)
      .addReg(Next32::SP_HIGH);
}

void Next32FrameLowering::determineCalleeSaves(MachineFunction &MF,
                                               BitVector &SavedRegs,
                                               RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  SavedRegs.reset(Next32::TID);
}
