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
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetLowering.h"

#define DEBUG_TYPE "irtranslator"

using namespace llvm;

char IRTranslator::ID = 0;
INITIALIZE_PASS(IRTranslator, "irtranslator", "IRTranslator LLVM IR -> MI",
                false, false)

IRTranslator::IRTranslator() : MachineFunctionPass(ID), MRI(nullptr) {
  initializeIRTranslatorPass(*PassRegistry::getPassRegistry());
}

unsigned IRTranslator::getOrCreateVReg(const Value &Val) {
  unsigned &ValReg = ValToVReg[&Val];
  // Check if this is the first time we see Val.
  if (!ValReg) {
    // Fill ValRegsSequence with the sequence of registers
    // we need to concat together to produce the value.
    assert(Val.getType()->isSized() &&
           "Don't know how to create an empty vreg");
    assert(!Val.getType()->isAggregateType() && "Not yet implemented");
    unsigned Size = DL->getTypeSizeInBits(Val.getType());
    unsigned VReg = MRI->createGenericVirtualRegister(Size);
    ValReg = VReg;
    assert(!isa<Constant>(Val) && "Not yet implemented");
  }
  return ValReg;
}

unsigned IRTranslator::getMemOpAlignment(const Instruction &I) {
  unsigned Alignment = 0;
  Type *ValTy = nullptr;
  if (const StoreInst *SI = dyn_cast<StoreInst>(&I)) {
    Alignment = SI->getAlignment();
    ValTy = SI->getValueOperand()->getType();
  } else if (const LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    Alignment = LI->getAlignment();
    ValTy = LI->getType();
  } else
    llvm_unreachable("unhandled memory instruction");

  return Alignment ? Alignment : DL->getABITypeAlignment(ValTy);
}

MachineBasicBlock &IRTranslator::getOrCreateBB(const BasicBlock &BB) {
  MachineBasicBlock *&MBB = BBToMBB[&BB];
  if (!MBB) {
    MachineFunction &MF = MIRBuilder.getMF();
    MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);
  }
  return *MBB;
}

bool IRTranslator::translateBinaryOp(unsigned Opcode,
                                     const BinaryOperator &Inst) {
  // FIXME: handle signed/unsigned wrapping flags.

  // Get or create a virtual register for each value.
  // Unless the value is a Constant => loadimm cst?
  // or inline constant each time?
  // Creation of a virtual register needs to have a size.
  unsigned Op0 = getOrCreateVReg(*Inst.getOperand(0));
  unsigned Op1 = getOrCreateVReg(*Inst.getOperand(1));
  unsigned Res = getOrCreateVReg(Inst);
  MIRBuilder.buildInstr(Opcode, LLT{*Inst.getType()})
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
  return true;
}

bool IRTranslator::translateReturn(const ReturnInst &RI) {
  const Value *Ret = RI.getReturnValue();
  // The target may mess up with the insertion point, but
  // this is not important as a return is the last instruction
  // of the block anyway.
  return CLI->lowerReturn(MIRBuilder, Ret, !Ret ? 0 : getOrCreateVReg(*Ret));
}

bool IRTranslator::translateBr(const BranchInst &BrInst) {
  unsigned Succ = 0;
  if (!BrInst.isUnconditional()) {
    // We want a G_BRCOND to the true BB followed by an unconditional branch.
    unsigned Tst = getOrCreateVReg(*BrInst.getCondition());
    const BasicBlock &TrueTgt = *cast<BasicBlock>(BrInst.getSuccessor(Succ++));
    MachineBasicBlock &TrueBB = getOrCreateBB(TrueTgt);
    MIRBuilder.buildBrCond(LLT{*BrInst.getCondition()->getType()}, Tst, TrueBB);
  }

  const BasicBlock &BrTgt = *cast<BasicBlock>(BrInst.getSuccessor(Succ));
  MachineBasicBlock &TgtBB = getOrCreateBB(BrTgt);
  MIRBuilder.buildBr(TgtBB);

  // Link successors.
  MachineBasicBlock &CurBB = MIRBuilder.getMBB();
  for (const BasicBlock *Succ : BrInst.successors())
    CurBB.addSuccessor(&getOrCreateBB(*Succ));
  return true;
}

bool IRTranslator::translateLoad(const LoadInst &LI) {
  assert(LI.isSimple() && "only simple loads are supported at the moment");

  MachineFunction &MF = MIRBuilder.getMF();
  unsigned Res = getOrCreateVReg(LI);
  unsigned Addr = getOrCreateVReg(*LI.getPointerOperand());
  LLT VTy{*LI.getType()}, PTy{*LI.getPointerOperand()->getType()};

  MIRBuilder.buildLoad(
      VTy, PTy, Res, Addr,
      *MF.getMachineMemOperand(MachinePointerInfo(LI.getPointerOperand()),
                               MachineMemOperand::MOLoad,
                               VTy.getSizeInBits() / 8, getMemOpAlignment(LI)));
  return true;
}

bool IRTranslator::translateStore(const StoreInst &SI) {
  assert(SI.isSimple() && "only simple loads are supported at the moment");

  MachineFunction &MF = MIRBuilder.getMF();
  unsigned Val = getOrCreateVReg(*SI.getValueOperand());
  unsigned Addr = getOrCreateVReg(*SI.getPointerOperand());
  LLT VTy{*SI.getValueOperand()->getType()},
      PTy{*SI.getPointerOperand()->getType()};

  MIRBuilder.buildStore(
      VTy, PTy, Val, Addr,
      *MF.getMachineMemOperand(MachinePointerInfo(SI.getPointerOperand()),
                               MachineMemOperand::MOStore,
                               VTy.getSizeInBits() / 8, getMemOpAlignment(SI)));
  return true;
}

bool IRTranslator::translateBitCast(const CastInst &CI) {
  if (LLT{*CI.getDestTy()} == LLT{*CI.getSrcTy()}) {
    MIRBuilder.buildCopy(getOrCreateVReg(CI),
                         getOrCreateVReg(*CI.getOperand(0)));
    return true;
  }
  return translateCast(TargetOpcode::G_BITCAST, CI);
}

bool IRTranslator::translateCast(unsigned Opcode, const CastInst &CI) {
  unsigned Op = getOrCreateVReg(*CI.getOperand(0));
  unsigned Res = getOrCreateVReg(CI);
  MIRBuilder.buildInstr(Opcode, {LLT{*CI.getDestTy()}, LLT{*CI.getSrcTy()}})
      .addDef(Res)
      .addUse(Op);
  return true;
}

bool IRTranslator::translateCall(const CallInst &CI) {
  auto TII = MIRBuilder.getMF().getTarget().getIntrinsicInfo();
  const Function &F = *CI.getCalledFunction();
  Intrinsic::ID ID = F.getIntrinsicID();
  if (TII && ID == Intrinsic::not_intrinsic)
    ID = static_cast<Intrinsic::ID>(TII->getIntrinsicID(&F));

  assert(ID != Intrinsic::not_intrinsic && "FIXME: support real calls");

  // Need types (starting with return) & args.
  SmallVector<LLT, 4> Tys;
  Tys.emplace_back(*CI.getType());
  for (auto &Arg : CI.arg_operands())
    Tys.emplace_back(*Arg->getType());

  unsigned Res = CI.getType()->isVoidTy() ? 0 : getOrCreateVReg(CI);
  MachineInstrBuilder MIB =
      MIRBuilder.buildIntrinsic(Tys, ID, Res, !CI.doesNotAccessMemory());

  for (auto &Arg : CI.arg_operands()) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Arg))
      MIB.addImm(CI->getSExtValue());
    else
      MIB.addUse(getOrCreateVReg(*Arg));
  }
  return true;
}

bool IRTranslator::translateStaticAlloca(const AllocaInst &AI) {
  assert(AI.isStaticAlloca() && "only handle static allocas now");
  MachineFunction &MF = MIRBuilder.getMF();
  unsigned ElementSize = DL->getTypeStoreSize(AI.getAllocatedType());
  unsigned Size =
      ElementSize * cast<ConstantInt>(AI.getArraySize())->getZExtValue();

  // Always allocate at least one byte.
  Size = std::max(Size, 1u);

  unsigned Alignment = AI.getAlignment();
  if (!Alignment)
    Alignment = DL->getABITypeAlignment(AI.getAllocatedType());

  unsigned Res = getOrCreateVReg(AI);
  int FI = MF.getFrameInfo().CreateStackObject(Size, Alignment, false, &AI);
  MIRBuilder.buildFrameIndex(LLT::pointer(0), Res, FI);
  return true;
}

bool IRTranslator::translate(const Instruction &Inst) {
  MIRBuilder.setDebugLoc(Inst.getDebugLoc());
  switch(Inst.getOpcode()) {
  // Arithmetic operations.
  case Instruction::Add:
    return translateBinaryOp(TargetOpcode::G_ADD, cast<BinaryOperator>(Inst));
  case Instruction::Sub:
    return translateBinaryOp(TargetOpcode::G_SUB, cast<BinaryOperator>(Inst));

  // Bitwise operations.
  case Instruction::And:
    return translateBinaryOp(TargetOpcode::G_AND, cast<BinaryOperator>(Inst));
  case Instruction::Or:
    return translateBinaryOp(TargetOpcode::G_OR, cast<BinaryOperator>(Inst));
  case Instruction::Xor:
    return translateBinaryOp(TargetOpcode::G_XOR, cast<BinaryOperator>(Inst));

  // Branch operations.
  case Instruction::Br:
    return translateBr(cast<BranchInst>(Inst));
  case Instruction::Ret:
    return translateReturn(cast<ReturnInst>(Inst));

  // Calls
  case Instruction::Call:
    return translateCall(cast<CallInst>(Inst));

  // Casts
  case Instruction::BitCast:
    return translateBitCast(cast<CastInst>(Inst));
  case Instruction::IntToPtr:
    return translateCast(TargetOpcode::G_INTTOPTR, cast<CastInst>(Inst));
  case Instruction::PtrToInt:
    return translateCast(TargetOpcode::G_PTRTOINT, cast<CastInst>(Inst));

  // Memory ops.
  case Instruction::Load:
    return translateLoad(cast<LoadInst>(Inst));
  case Instruction::Store:
    return translateStore(cast<StoreInst>(Inst));

  case Instruction::Alloca:
    return translateStaticAlloca(cast<AllocaInst>(Inst));

  case Instruction::Unreachable:
    return true;

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
  CLI = MF.getSubtarget().getCallLowering();
  MIRBuilder.setMF(MF);
  MRI = &MF.getRegInfo();
  DL = &F.getParent()->getDataLayout();

  // Setup the arguments.
  MachineBasicBlock &MBB = getOrCreateBB(F.front());
  MIRBuilder.setMBB(MBB);
  SmallVector<unsigned, 8> VRegArgs;
  for (const Argument &Arg: F.args())
    VRegArgs.push_back(getOrCreateVReg(Arg));
  bool Succeeded =
      CLI->lowerFormalArguments(MIRBuilder, F.getArgumentList(), VRegArgs);
  if (!Succeeded)
    report_fatal_error("Unable to lower arguments");

  for (const BasicBlock &BB: F) {
    MachineBasicBlock &MBB = getOrCreateBB(BB);
    // Set the insertion point of all the following translations to
    // the end of this basic block.
    MIRBuilder.setMBB(MBB);
    for (const Instruction &Inst: BB) {
      bool Succeeded = translate(Inst);
      if (!Succeeded) {
        DEBUG(dbgs() << "Cannot translate: " << Inst << '\n');
        report_fatal_error("Unable to translate instruction");
      }
    }
  }

  // Now that the MachineFrameInfo has been configured, no further changes to
  // the reserved registers are possible.
  MRI->freezeReservedRegs(MF);

  return false;
}
