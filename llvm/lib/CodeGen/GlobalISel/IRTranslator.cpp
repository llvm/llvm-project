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
    unsigned Size = DL->getTypeSizeInBits(Val.getType());
    unsigned VReg = MRI->createGenericVirtualRegister(Size);
    ValReg = VReg;

    if (auto CV = dyn_cast<Constant>(&Val)) {
      bool Success = translate(*CV, VReg);
      if (!Success)
        report_fatal_error("unable to translate constant");
    }
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

bool IRTranslator::translateBinaryOp(unsigned Opcode, const User &U) {
  // FIXME: handle signed/unsigned wrapping flags.

  // Get or create a virtual register for each value.
  // Unless the value is a Constant => loadimm cst?
  // or inline constant each time?
  // Creation of a virtual register needs to have a size.
  unsigned Op0 = getOrCreateVReg(*U.getOperand(0));
  unsigned Op1 = getOrCreateVReg(*U.getOperand(1));
  unsigned Res = getOrCreateVReg(U);
  MIRBuilder.buildInstr(Opcode, LLT{*U.getType()})
      .addDef(Res)
      .addUse(Op0)
      .addUse(Op1);
  return true;
}

bool IRTranslator::translateICmp(const User &U) {
  const CmpInst &CI = cast<CmpInst>(U);
  unsigned Op0 = getOrCreateVReg(*CI.getOperand(0));
  unsigned Op1 = getOrCreateVReg(*CI.getOperand(1));
  unsigned Res = getOrCreateVReg(CI);
  CmpInst::Predicate Pred = CI.getPredicate();

  assert(isa<ICmpInst>(CI) && "only integer comparisons supported now");
  assert(CmpInst::isIntPredicate(Pred) && "only int comparisons supported now");
  MIRBuilder.buildICmp({LLT{*CI.getType()}, LLT{*CI.getOperand(0)->getType()}},
                       Pred, Res, Op0, Op1);
  return true;
}

bool IRTranslator::translateRet(const User &U) {
  const ReturnInst &RI = cast<ReturnInst>(U);
  const Value *Ret = RI.getReturnValue();
  // The target may mess up with the insertion point, but
  // this is not important as a return is the last instruction
  // of the block anyway.
  return CLI->lowerReturn(MIRBuilder, Ret, !Ret ? 0 : getOrCreateVReg(*Ret));
}

bool IRTranslator::translateBr(const User &U) {
  const BranchInst &BrInst = cast<BranchInst>(U);
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

bool IRTranslator::translateLoad(const User &U) {
  const LoadInst &LI = cast<LoadInst>(U);
  assert(LI.isSimple() && "only simple loads are supported at the moment");

  MachineFunction &MF = MIRBuilder.getMF();
  unsigned Res = getOrCreateVReg(LI);
  unsigned Addr = getOrCreateVReg(*LI.getPointerOperand());
  LLT VTy{*LI.getType(), DL}, PTy{*LI.getPointerOperand()->getType()};

  MIRBuilder.buildLoad(
      VTy, PTy, Res, Addr,
      *MF.getMachineMemOperand(
          MachinePointerInfo(LI.getPointerOperand()), MachineMemOperand::MOLoad,
          DL->getTypeStoreSize(LI.getType()), getMemOpAlignment(LI)));
  return true;
}

bool IRTranslator::translateStore(const User &U) {
  const StoreInst &SI = cast<StoreInst>(U);
  assert(SI.isSimple() && "only simple loads are supported at the moment");

  MachineFunction &MF = MIRBuilder.getMF();
  unsigned Val = getOrCreateVReg(*SI.getValueOperand());
  unsigned Addr = getOrCreateVReg(*SI.getPointerOperand());
  LLT VTy{*SI.getValueOperand()->getType(), DL},
      PTy{*SI.getPointerOperand()->getType()};

  MIRBuilder.buildStore(
      VTy, PTy, Val, Addr,
      *MF.getMachineMemOperand(
          MachinePointerInfo(SI.getPointerOperand()),
          MachineMemOperand::MOStore,
          DL->getTypeStoreSize(SI.getValueOperand()->getType()),
          getMemOpAlignment(SI)));
  return true;
}

bool IRTranslator::translateBitCast(const User &U) {
  if (LLT{*U.getOperand(0)->getType()} == LLT{*U.getType()}) {
    unsigned &Reg = ValToVReg[&U];
    if (Reg)
      MIRBuilder.buildCopy(Reg, getOrCreateVReg(*U.getOperand(0)));
    else
      Reg = getOrCreateVReg(*U.getOperand(0));
    return true;
  }
  return translateCast(TargetOpcode::G_BITCAST, U);
}

bool IRTranslator::translateCast(unsigned Opcode, const User &U) {
  unsigned Op = getOrCreateVReg(*U.getOperand(0));
  unsigned Res = getOrCreateVReg(U);
  MIRBuilder
      .buildInstr(Opcode, {LLT{*U.getType()}, LLT{*U.getOperand(0)->getType()}})
      .addDef(Res)
      .addUse(Op);
  return true;
}

bool IRTranslator::translateCall(const User &U) {
  const CallInst &CI = cast<CallInst>(U);
  auto TII = MIRBuilder.getMF().getTarget().getIntrinsicInfo();
  const Function *F = CI.getCalledFunction();

  if (!F || !F->isIntrinsic()) {
    // FIXME: handle multiple return values.
    unsigned Res = CI.getType()->isVoidTy() ? 0 : getOrCreateVReg(CI);
    SmallVector<unsigned, 8> Args;
    for (auto &Arg: CI.arg_operands())
      Args.push_back(getOrCreateVReg(*Arg));

    return CLI->lowerCall(MIRBuilder, CI,
                          F ? 0 : getOrCreateVReg(*CI.getCalledValue()), Res,
                          Args);
  }

  Intrinsic::ID ID = F->getIntrinsicID();
  if (TII && ID == Intrinsic::not_intrinsic)
    ID = static_cast<Intrinsic::ID>(TII->getIntrinsicID(F));

  assert(ID != Intrinsic::not_intrinsic && "unknown intrinsic");

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

bool IRTranslator::translatePHI(const User &U) {
  const PHINode &PI = cast<PHINode>(U);
  MachineInstrBuilder MIB = MIRBuilder.buildInstr(TargetOpcode::PHI);
  MIB.addDef(getOrCreateVReg(PI));

  PendingPHIs.emplace_back(&PI, MIB.getInstr());
  return true;
}

void IRTranslator::finishPendingPhis() {
  for (std::pair<const PHINode *, MachineInstr *> &Phi : PendingPHIs) {
    const PHINode *PI = Phi.first;
    MachineInstrBuilder MIB(MIRBuilder.getMF(), Phi.second);

    // All MachineBasicBlocks exist, add them to the PHI. We assume IRTranslator
    // won't create extra control flow here, otherwise we need to find the
    // dominating predecessor here (or perhaps force the weirder IRTranslators
    // to provide a simple boundary).
    for (unsigned i = 0; i < PI->getNumIncomingValues(); ++i) {
      assert(BBToMBB[PI->getIncomingBlock(i)]->isSuccessor(MIB->getParent()) &&
             "I appear to have misunderstood Machine PHIs");
      MIB.addUse(getOrCreateVReg(*PI->getIncomingValue(i)));
      MIB.addMBB(BBToMBB[PI->getIncomingBlock(i)]);
    }
  }

  PendingPHIs.clear();
}

bool IRTranslator::translate(const Instruction &Inst) {
  MIRBuilder.setDebugLoc(Inst.getDebugLoc());
  switch(Inst.getOpcode()) {
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE: return translate##OPCODE(Inst);
#include "llvm/IR/Instruction.def"
  default:
    llvm_unreachable("unknown opcode");
  }
}

bool IRTranslator::translate(const Constant &C, unsigned Reg) {
  if (auto CI = dyn_cast<ConstantInt>(&C))
    EntryBuilder.buildConstant(LLT{*CI->getType()}, Reg, CI->getZExtValue());
  else if (isa<UndefValue>(C))
    EntryBuilder.buildInstr(TargetOpcode::IMPLICIT_DEF).addDef(Reg);
  else if (isa<ConstantPointerNull>(C))
    EntryBuilder.buildInstr(TargetOpcode::G_CONSTANT, LLT{*C.getType()})
        .addDef(Reg)
        .addImm(0);
  else if (auto CE = dyn_cast<ConstantExpr>(&C)) {
    switch(CE->getOpcode()) {
#define HANDLE_INST(NUM, OPCODE, CLASS)                         \
      case Instruction::OPCODE: return translate##OPCODE(*CE);
#include "llvm/IR/Instruction.def"
    default:
      llvm_unreachable("unknown opcode");
    }
  } else
    llvm_unreachable("unhandled constant kind");

  return true;
}


void IRTranslator::finalizeFunction() {
  finishPendingPhis();

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
  EntryBuilder.setMF(MF);
  MRI = &MF.getRegInfo();
  DL = &F.getParent()->getDataLayout();

  assert(PendingPHIs.empty() && "stale PHIs");

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

  // Now that we've got the ABI handling code, it's safe to set a location for
  // any Constants we find in the IR.
  if (MBB.empty())
    EntryBuilder.setMBB(MBB);
  else
    EntryBuilder.setInstr(MBB.back(), /* Before */ false);

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

  finalizeFunction();

  // Now that the MachineFrameInfo has been configured, no further changes to
  // the reserved registers are possible.
  MRI->freezeReservedRegs(MF);

  return false;
}
