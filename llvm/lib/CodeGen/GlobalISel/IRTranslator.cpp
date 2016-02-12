//===-- llvm/CodeGen/GlobalISel/IRTranslator.cpp - IRTranslator --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the IRTranslator class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/IRTranslator.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Target/TargetLowering.h"

#define DEBUG_TYPE "irtranslator"

using namespace llvm;

char IRTranslator::ID = 0;

IRTranslator::IRTranslator() : MachineFunctionPass(ID), MRI(nullptr) {
}

unsigned IRTranslator::getOrCreateVReg(const Value *Val) {
  unsigned &ValReg = ValToVReg[Val];
  // Check if this is the first time we see Val.
  if (!ValReg) {
    // Fill ValRegsSequence with the sequence of registers
    // we need to concat together to produce the value.
    assert(Val->getType()->isSized() &&
           "Don't know how to create an empty vreg");
    assert(!Val->getType()->isAggregateType() && "Not yet implemented");
    unsigned Size = Val->getType()->getPrimitiveSizeInBits();
    unsigned VReg = MRI->createGenericVirtualRegister(Size);
    ValReg = VReg;
    assert(!isa<Constant>(Val) && "Not yet implemented");
  }
  return ValReg;
}

MachineBasicBlock &IRTranslator::getOrCreateBB(const BasicBlock *BB) {
  MachineBasicBlock *&MBB = BBToMBB[BB];
  if (!MBB) {
    MachineFunction &MF = MIRBuilder.getMF();
    MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);
  }
  return *MBB;
}

bool IRTranslator::translateADD(const Instruction &Inst) {
  // Get or create a virtual register for each value.
  // Unless the value is a Constant => loadimm cst?
  // or inline constant each time?
  // Creation of a virtual register needs to have a size.
  unsigned Op0 = getOrCreateVReg(Inst.getOperand(0));
  unsigned Op1 = getOrCreateVReg(Inst.getOperand(1));
  unsigned Res = getOrCreateVReg(&Inst);
  MIRBuilder.buildInstr(TargetOpcode::G_ADD, Inst.getType(), Res, Op0, Op1);
  return true;
}

bool IRTranslator::translateReturn(const Instruction &Inst) {
  assert(isa<ReturnInst>(Inst) && "Return expected");
  const Value *Ret = cast<ReturnInst>(Inst).getReturnValue();
  // The target may mess up with the insertion point, but
  // this is not important as a return is the last instruction
  // of the block anyway.
  return TLI->LowerReturn(MIRBuilder, Ret,
                          !Ret ? 0 : getOrCreateVReg(Ret));
}

bool IRTranslator::translate(const Instruction &Inst) {
  MIRBuilder.setDebugLoc(Inst.getDebugLoc());
  switch(Inst.getOpcode()) {
  case Instruction::Add:
    return translateADD(Inst);
  case Instruction::Ret:
    return translateReturn(Inst);

  default:
    llvm_unreachable("Opcode not supported");
  }
}


void IRTranslator::finalize() {
  // Release the memory used by the different maps we
  // needed during the translation.
  ValToVReg.clear();
  Constants.clear();
}

bool IRTranslator::runOnMachineFunction(MachineFunction &MF) {
  const Function &F = *MF.getFunction();
  if (F.empty())
    return false;
  TLI = MF.getSubtarget().getTargetLowering();
  MIRBuilder.setFunction(MF);
  MRI = &MF.getRegInfo();
  // Setup the arguments.
  MachineBasicBlock &MBB = getOrCreateBB(&F.front());
  MIRBuilder.setBasicBlock(MBB);
  SmallVector<unsigned, 8> VRegArgs;
  for (const Argument &Arg: F.args())
    VRegArgs.push_back(getOrCreateVReg(&Arg));
  bool Succeeded = TLI->LowerFormalArguments(MIRBuilder, F.getArgumentList(),
                                             VRegArgs);
  if (!Succeeded)
    report_fatal_error("Unable to lower arguments");

  for (const BasicBlock &BB: F) {
    MachineBasicBlock &MBB = getOrCreateBB(&BB);
    MIRBuilder.setBasicBlock(MBB);
    for (const Instruction &Inst: BB) {
      bool Succeeded = translate(Inst);
      if (!Succeeded) {
        DEBUG(dbgs() << "Cannot translate: " << Inst << '\n');
        report_fatal_error("Unable to translate instruction");
      }
    }
  }
  return false;
}
