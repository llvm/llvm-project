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
#include "llvm/CodeGen/TargetPassConfig.h"
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
INITIALIZE_PASS_BEGIN(IRTranslator, DEBUG_TYPE, "IRTranslator LLVM IR -> MI",
                false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(IRTranslator, DEBUG_TYPE, "IRTranslator LLVM IR -> MI",
                false, false)

IRTranslator::IRTranslator() : MachineFunctionPass(ID), MRI(nullptr) {
  initializeIRTranslatorPass(*PassRegistry::getPassRegistry());
}

void IRTranslator::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
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
      if (!Success) {
        if (!TPC->isGlobalISelAbortEnabled()) {
          MIRBuilder.getMF().getProperties().set(
              MachineFunctionProperties::Property::FailedISel);
          return 0;
        }
        report_fatal_error("unable to translate constant");
      }
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
  } else if (!TPC->isGlobalISelAbortEnabled()) {
    MIRBuilder.getMF().getProperties().set(
        MachineFunctionProperties::Property::FailedISel);
    return 1;
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

bool IRTranslator::translateCompare(const User &U) {
  const CmpInst *CI = dyn_cast<CmpInst>(&U);
  unsigned Op0 = getOrCreateVReg(*U.getOperand(0));
  unsigned Op1 = getOrCreateVReg(*U.getOperand(1));
  unsigned Res = getOrCreateVReg(U);
  CmpInst::Predicate Pred =
      CI ? CI->getPredicate() : static_cast<CmpInst::Predicate>(
                                    cast<ConstantExpr>(U).getPredicate());

  if (CmpInst::isIntPredicate(Pred))
    MIRBuilder.buildICmp(
        {LLT{*U.getType()}, LLT{*U.getOperand(0)->getType()}}, Pred, Res, Op0,
        Op1);
  else
    MIRBuilder.buildFCmp(
        {LLT{*U.getType()}, LLT{*U.getOperand(0)->getType()}}, Pred, Res, Op0,
        Op1);

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

  if (!TPC->isGlobalISelAbortEnabled() && !LI.isSimple())
    return false;

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

  if (!TPC->isGlobalISelAbortEnabled() && !SI.isSimple())
    return false;

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

bool IRTranslator::translateExtractValue(const User &U) {
  const Value *Src = U.getOperand(0);
  Type *Int32Ty = Type::getInt32Ty(U.getContext());
  SmallVector<Value *, 1> Indices;

  // getIndexedOffsetInType is designed for GEPs, so the first index is the
  // usual array element rather than looking into the actual aggregate.
  Indices.push_back(ConstantInt::get(Int32Ty, 0));

  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(&U)) {
    for (auto Idx : EVI->indices())
      Indices.push_back(ConstantInt::get(Int32Ty, Idx));
  } else {
    for (unsigned i = 1; i < U.getNumOperands(); ++i)
      Indices.push_back(U.getOperand(i));
  }

  uint64_t Offset = 8 * DL->getIndexedOffsetInType(Src->getType(), Indices);

  unsigned Res = getOrCreateVReg(U);
  MIRBuilder.buildExtract(LLT{*U.getType(), DL}, Res, Offset,
                          LLT{*Src->getType(), DL}, getOrCreateVReg(*Src));

  return true;
}

bool IRTranslator::translateInsertValue(const User &U) {
  const Value *Src = U.getOperand(0);
  Type *Int32Ty = Type::getInt32Ty(U.getContext());
  SmallVector<Value *, 1> Indices;

  // getIndexedOffsetInType is designed for GEPs, so the first index is the
  // usual array element rather than looking into the actual aggregate.
  Indices.push_back(ConstantInt::get(Int32Ty, 0));

  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(&U)) {
    for (auto Idx : IVI->indices())
      Indices.push_back(ConstantInt::get(Int32Ty, Idx));
  } else {
    for (unsigned i = 2; i < U.getNumOperands(); ++i)
      Indices.push_back(U.getOperand(i));
  }

  uint64_t Offset = 8 * DL->getIndexedOffsetInType(Src->getType(), Indices);

  unsigned Res = getOrCreateVReg(U);
  const Value &Inserted = *U.getOperand(1);
  MIRBuilder.buildInsert(LLT{*U.getType(), DL}, Res, getOrCreateVReg(*Src),
                         LLT{*Inserted.getType(), DL},
                         getOrCreateVReg(Inserted), Offset);

  return true;
}

bool IRTranslator::translateSelect(const User &U) {
  MIRBuilder.buildSelect(
      LLT{*U.getType()}, getOrCreateVReg(U), getOrCreateVReg(*U.getOperand(0)),
      getOrCreateVReg(*U.getOperand(1)), getOrCreateVReg(*U.getOperand(2)));
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

bool IRTranslator::translateKnownIntrinsic(const CallInst &CI,
                                           Intrinsic::ID ID) {
  unsigned Op = 0;
  switch (ID) {
  default: return false;
  case Intrinsic::uadd_with_overflow: Op = TargetOpcode::G_UADDE; break;
  case Intrinsic::sadd_with_overflow: Op = TargetOpcode::G_SADDO; break;
  case Intrinsic::usub_with_overflow: Op = TargetOpcode::G_USUBE; break;
  case Intrinsic::ssub_with_overflow: Op = TargetOpcode::G_SSUBO; break;
  case Intrinsic::umul_with_overflow: Op = TargetOpcode::G_UMULO; break;
  case Intrinsic::smul_with_overflow: Op = TargetOpcode::G_SMULO; break;
  }

  LLT Ty{*CI.getOperand(0)->getType()};
  LLT s1 = LLT::scalar(1);
  unsigned Width = Ty.getSizeInBits();
  unsigned Res = MRI->createGenericVirtualRegister(Width);
  unsigned Overflow = MRI->createGenericVirtualRegister(1);
  auto MIB = MIRBuilder.buildInstr(Op, {Ty, s1})
                 .addDef(Res)
                 .addDef(Overflow)
                 .addUse(getOrCreateVReg(*CI.getOperand(0)))
                 .addUse(getOrCreateVReg(*CI.getOperand(1)));

  if (Op == TargetOpcode::G_UADDE || Op == TargetOpcode::G_USUBE) {
    unsigned Zero = MRI->createGenericVirtualRegister(1);
    EntryBuilder.buildConstant(s1, Zero, 0);
    MIB.addUse(Zero);
  }

  MIRBuilder.buildSequence(LLT{*CI.getType(), DL}, getOrCreateVReg(CI), Ty, Res,
                           0, s1, Overflow, Width);
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

    return CLI->lowerCall(MIRBuilder, CI, Res, Args, [&]() {
      return getOrCreateVReg(*CI.getCalledValue());
    });
  }

  Intrinsic::ID ID = F->getIntrinsicID();
  if (TII && ID == Intrinsic::not_intrinsic)
    ID = static_cast<Intrinsic::ID>(TII->getIntrinsicID(F));

  assert(ID != Intrinsic::not_intrinsic && "unknown intrinsic");

  if (translateKnownIntrinsic(CI, ID))
    return true;

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
  if (!TPC->isGlobalISelAbortEnabled() && !AI.isStaticAlloca())
    return false;

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
  auto MIB = MIRBuilder.buildInstr(TargetOpcode::G_PHI, LLT{*U.getType()});
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
    if (!TPC->isGlobalISelAbortEnabled())
      return false;
    llvm_unreachable("unknown opcode");
  }
}

bool IRTranslator::translate(const Constant &C, unsigned Reg) {
  if (auto CI = dyn_cast<ConstantInt>(&C))
    EntryBuilder.buildConstant(LLT{*CI->getType()}, Reg, CI->getZExtValue());
  else if (auto CF = dyn_cast<ConstantFP>(&C))
    EntryBuilder.buildFConstant(LLT{*CF->getType()}, Reg, *CF);
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
      if (!TPC->isGlobalISelAbortEnabled())
        return false;
      llvm_unreachable("unknown opcode");
    }
  } else if (!TPC->isGlobalISelAbortEnabled())
    return false;
  else
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
  TPC = &getAnalysis<TargetPassConfig>();

  assert(PendingPHIs.empty() && "stale PHIs");

  // Setup the arguments.
  MachineBasicBlock &MBB = getOrCreateBB(F.front());
  MIRBuilder.setMBB(MBB);
  SmallVector<unsigned, 8> VRegArgs;
  for (const Argument &Arg: F.args())
    VRegArgs.push_back(getOrCreateVReg(Arg));
  bool Succeeded =
      CLI->lowerFormalArguments(MIRBuilder, F.getArgumentList(), VRegArgs);
  if (!Succeeded) {
    if (!TPC->isGlobalISelAbortEnabled()) {
      MIRBuilder.getMF().getProperties().set(
          MachineFunctionProperties::Property::FailedISel);
      return false;
    }
    report_fatal_error("Unable to lower arguments");
  }

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
        if (TPC->isGlobalISelAbortEnabled())
          report_fatal_error("Unable to translate instruction");
        MF.getProperties().set(MachineFunctionProperties::Property::FailedISel);
        break;
      }
    }
  }

  finalizeFunction();

  // Now that the MachineFrameInfo has been configured, no further changes to
  // the reserved registers are possible.
  MRI->freezeReservedRegs(MF);

  return false;
}
