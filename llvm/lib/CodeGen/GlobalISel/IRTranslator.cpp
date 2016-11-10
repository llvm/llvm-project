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
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
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

static void reportTranslationError(const Value &V, const Twine &Message) {
  std::string ErrStorage;
  raw_string_ostream Err(ErrStorage);
  Err << Message << ": " << V << '\n';
  report_fatal_error(Err.str());
}

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
    unsigned VReg = MRI->createGenericVirtualRegister(LLT{*Val.getType(), *DL});
    ValReg = VReg;

    if (auto CV = dyn_cast<Constant>(&Val)) {
      bool Success = translate(*CV, VReg);
      if (!Success) {
        if (!TPC->isGlobalISelAbortEnabled()) {
          MIRBuilder.getMF().getProperties().set(
              MachineFunctionProperties::Property::FailedISel);
          return 0;
        }
        reportTranslationError(Val, "unable to translate constant");
      }
    }
  }
  return ValReg;
}

int IRTranslator::getOrCreateFrameIndex(const AllocaInst &AI) {
  if (FrameIndices.find(&AI) != FrameIndices.end())
    return FrameIndices[&AI];

  MachineFunction &MF = MIRBuilder.getMF();
  unsigned ElementSize = DL->getTypeStoreSize(AI.getAllocatedType());
  unsigned Size =
      ElementSize * cast<ConstantInt>(AI.getArraySize())->getZExtValue();

  // Always allocate at least one byte.
  Size = std::max(Size, 1u);

  unsigned Alignment = AI.getAlignment();
  if (!Alignment)
    Alignment = DL->getABITypeAlignment(AI.getAllocatedType());

  int &FI = FrameIndices[&AI];
  FI = MF.getFrameInfo().CreateStackObject(Size, Alignment, false, &AI);
  return FI;
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
  MIRBuilder.buildInstr(Opcode).addDef(Res).addUse(Op0).addUse(Op1);
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
    MIRBuilder.buildICmp(Pred, Res, Op0, Op1);
  else
    MIRBuilder.buildFCmp(Pred, Res, Op0, Op1);

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
    MIRBuilder.buildBrCond(Tst, TrueBB);
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

  if (!TPC->isGlobalISelAbortEnabled() && LI.isAtomic())
    return false;

  assert(!LI.isAtomic() && "only non-atomic loads are supported at the moment");
  auto Flags = LI.isVolatile() ? MachineMemOperand::MOVolatile
                               : MachineMemOperand::MONone;
  Flags |= MachineMemOperand::MOLoad;

  MachineFunction &MF = MIRBuilder.getMF();
  unsigned Res = getOrCreateVReg(LI);
  unsigned Addr = getOrCreateVReg(*LI.getPointerOperand());
  LLT VTy{*LI.getType(), *DL}, PTy{*LI.getPointerOperand()->getType(), *DL};
  MIRBuilder.buildLoad(
      Res, Addr,
      *MF.getMachineMemOperand(MachinePointerInfo(LI.getPointerOperand()),
                               Flags, DL->getTypeStoreSize(LI.getType()),
                               getMemOpAlignment(LI)));
  return true;
}

bool IRTranslator::translateStore(const User &U) {
  const StoreInst &SI = cast<StoreInst>(U);

  if (!TPC->isGlobalISelAbortEnabled() && SI.isAtomic())
    return false;

  assert(!SI.isAtomic() && "only non-atomic stores supported at the moment");
  auto Flags = SI.isVolatile() ? MachineMemOperand::MOVolatile
                               : MachineMemOperand::MONone;
  Flags |= MachineMemOperand::MOStore;

  MachineFunction &MF = MIRBuilder.getMF();
  unsigned Val = getOrCreateVReg(*SI.getValueOperand());
  unsigned Addr = getOrCreateVReg(*SI.getPointerOperand());
  LLT VTy{*SI.getValueOperand()->getType(), *DL},
      PTy{*SI.getPointerOperand()->getType(), *DL};

  MIRBuilder.buildStore(
      Val, Addr, *MF.getMachineMemOperand(
                     MachinePointerInfo(SI.getPointerOperand()), Flags,
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
  MIRBuilder.buildExtract(Res, Offset, getOrCreateVReg(*Src));

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
  MIRBuilder.buildInsert(Res, getOrCreateVReg(*Src), getOrCreateVReg(Inserted),
                         Offset);

  return true;
}

bool IRTranslator::translateSelect(const User &U) {
  MIRBuilder.buildSelect(getOrCreateVReg(U), getOrCreateVReg(*U.getOperand(0)),
                         getOrCreateVReg(*U.getOperand(1)),
                         getOrCreateVReg(*U.getOperand(2)));
  return true;
}

bool IRTranslator::translateBitCast(const User &U) {
  if (LLT{*U.getOperand(0)->getType(), *DL} == LLT{*U.getType(), *DL}) {
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
  MIRBuilder.buildInstr(Opcode).addDef(Res).addUse(Op);
  return true;
}

bool IRTranslator::translateGetElementPtr(const User &U) {
  // FIXME: support vector GEPs.
  if (U.getType()->isVectorTy())
    return false;

  Value &Op0 = *U.getOperand(0);
  unsigned BaseReg = getOrCreateVReg(Op0);
  LLT PtrTy{*Op0.getType(), *DL};
  unsigned PtrSize = DL->getPointerSizeInBits(PtrTy.getAddressSpace());
  LLT OffsetTy = LLT::scalar(PtrSize);

  int64_t Offset = 0;
  for (gep_type_iterator GTI = gep_type_begin(&U), E = gep_type_end(&U);
       GTI != E; ++GTI) {
    const Value *Idx = GTI.getOperand();
    if (StructType *StTy = dyn_cast<StructType>(*GTI)) {
      unsigned Field = cast<Constant>(Idx)->getUniqueInteger().getZExtValue();
      Offset += DL->getStructLayout(StTy)->getElementOffset(Field);
      continue;
    } else {
      uint64_t ElementSize = DL->getTypeAllocSize(GTI.getIndexedType());

      // If this is a scalar constant or a splat vector of constants,
      // handle it quickly.
      if (const auto *CI = dyn_cast<ConstantInt>(Idx)) {
        Offset += ElementSize * CI->getSExtValue();
        continue;
      }

      if (Offset != 0) {
        unsigned NewBaseReg = MRI->createGenericVirtualRegister(PtrTy);
        unsigned OffsetReg = MRI->createGenericVirtualRegister(OffsetTy);
        MIRBuilder.buildConstant(OffsetReg, Offset);
        MIRBuilder.buildGEP(NewBaseReg, BaseReg, OffsetReg);

        BaseReg = NewBaseReg;
        Offset = 0;
      }

      // N = N + Idx * ElementSize;
      unsigned ElementSizeReg = MRI->createGenericVirtualRegister(OffsetTy);
      MIRBuilder.buildConstant(ElementSizeReg, ElementSize);

      unsigned IdxReg = getOrCreateVReg(*Idx);
      if (MRI->getType(IdxReg) != OffsetTy) {
        unsigned NewIdxReg = MRI->createGenericVirtualRegister(OffsetTy);
        MIRBuilder.buildSExtOrTrunc(NewIdxReg, IdxReg);
        IdxReg = NewIdxReg;
      }

      unsigned OffsetReg = MRI->createGenericVirtualRegister(OffsetTy);
      MIRBuilder.buildMul(OffsetReg, ElementSizeReg, IdxReg);

      unsigned NewBaseReg = MRI->createGenericVirtualRegister(PtrTy);
      MIRBuilder.buildGEP(NewBaseReg, BaseReg, OffsetReg);
      BaseReg = NewBaseReg;
    }
  }

  if (Offset != 0) {
    unsigned OffsetReg = MRI->createGenericVirtualRegister(OffsetTy);
    MIRBuilder.buildConstant(OffsetReg, Offset);
    MIRBuilder.buildGEP(getOrCreateVReg(U), BaseReg, OffsetReg);
    return true;
  }

  MIRBuilder.buildCopy(getOrCreateVReg(U), BaseReg);
  return true;
}

bool IRTranslator::translateMemcpy(const CallInst &CI) {
  LLT SizeTy{*CI.getArgOperand(2)->getType(), *DL};
  if (cast<PointerType>(CI.getArgOperand(0)->getType())->getAddressSpace() !=
          0 ||
      cast<PointerType>(CI.getArgOperand(1)->getType())->getAddressSpace() !=
          0 ||
      SizeTy.getSizeInBits() != DL->getPointerSizeInBits(0))
    return false;

  SmallVector<CallLowering::ArgInfo, 8> Args;
  for (int i = 0; i < 3; ++i) {
    const auto &Arg = CI.getArgOperand(i);
    Args.emplace_back(getOrCreateVReg(*Arg), Arg->getType());
  }

  MachineOperand Callee = MachineOperand::CreateES("memcpy");

  return CLI->lowerCall(MIRBuilder, Callee,
                        CallLowering::ArgInfo(0, CI.getType()), Args);
}

void IRTranslator::getStackGuard(unsigned DstReg) {
  auto MIB = MIRBuilder.buildInstr(TargetOpcode::LOAD_STACK_GUARD);
  MIB.addDef(DstReg);

  auto &MF = MIRBuilder.getMF();
  auto &TLI = *MF.getSubtarget().getTargetLowering();
  Value *Global = TLI.getSDagStackGuard(*MF.getFunction()->getParent());
  if (!Global)
    return;

  MachinePointerInfo MPInfo(Global);
  MachineInstr::mmo_iterator MemRefs = MF.allocateMemRefsArray(1);
  auto Flags = MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant |
               MachineMemOperand::MODereferenceable;
  *MemRefs =
      MF.getMachineMemOperand(MPInfo, Flags, DL->getPointerSizeInBits() / 8,
                              DL->getPointerABIAlignment());
  MIB.setMemRefs(MemRefs, MemRefs + 1);
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
  case Intrinsic::memcpy:
    return translateMemcpy(CI);
  case Intrinsic::eh_typeid_for: {
    GlobalValue *GV = ExtractTypeInfo(CI.getArgOperand(0));
    unsigned Reg = getOrCreateVReg(CI);
    unsigned TypeID = MIRBuilder.getMF().getMMI().getTypeIDFor(GV);
    MIRBuilder.buildConstant(Reg, TypeID);
    return true;
  }
  case Intrinsic::objectsize: {
    // If we don't know by now, we're never going to know.
    const ConstantInt *Min = cast<ConstantInt>(CI.getArgOperand(1));

    MIRBuilder.buildConstant(getOrCreateVReg(CI), Min->isZero() ? -1ULL : 0);
    return true;
  }
  case Intrinsic::stackguard:
    getStackGuard(getOrCreateVReg(CI));
    return true;
  case Intrinsic::stackprotector: {
    MachineFunction &MF = MIRBuilder.getMF();
    LLT PtrTy{*CI.getArgOperand(0)->getType(), *DL};
    unsigned GuardVal = MRI->createGenericVirtualRegister(PtrTy);
    getStackGuard(GuardVal);

    AllocaInst *Slot = cast<AllocaInst>(CI.getArgOperand(1));
    MIRBuilder.buildStore(
        GuardVal, getOrCreateVReg(*Slot),
        *MF.getMachineMemOperand(
            MachinePointerInfo::getFixedStack(MF, getOrCreateFrameIndex(*Slot)),
            MachineMemOperand::MOStore | MachineMemOperand::MOVolatile,
            PtrTy.getSizeInBits() / 8, 8));
    return true;
  }
  }

  LLT Ty{*CI.getOperand(0)->getType(), *DL};
  LLT s1 = LLT::scalar(1);
  unsigned Width = Ty.getSizeInBits();
  unsigned Res = MRI->createGenericVirtualRegister(Ty);
  unsigned Overflow = MRI->createGenericVirtualRegister(s1);
  auto MIB = MIRBuilder.buildInstr(Op)
                 .addDef(Res)
                 .addDef(Overflow)
                 .addUse(getOrCreateVReg(*CI.getOperand(0)))
                 .addUse(getOrCreateVReg(*CI.getOperand(1)));

  if (Op == TargetOpcode::G_UADDE || Op == TargetOpcode::G_USUBE) {
    unsigned Zero = MRI->createGenericVirtualRegister(s1);
    EntryBuilder.buildConstant(Zero, 0);
    MIB.addUse(Zero);
  }

  MIRBuilder.buildSequence(getOrCreateVReg(CI), Res, 0, Overflow, Width);
  return true;
}

bool IRTranslator::translateCall(const User &U) {
  const CallInst &CI = cast<CallInst>(U);
  auto TII = MIRBuilder.getMF().getTarget().getIntrinsicInfo();
  const Function *F = CI.getCalledFunction();

  if (!F || !F->isIntrinsic()) {
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

  unsigned Res = CI.getType()->isVoidTy() ? 0 : getOrCreateVReg(CI);
  MachineInstrBuilder MIB =
      MIRBuilder.buildIntrinsic(ID, Res, !CI.doesNotAccessMemory());

  for (auto &Arg : CI.arg_operands()) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Arg))
      MIB.addImm(CI->getSExtValue());
    else
      MIB.addUse(getOrCreateVReg(*Arg));
  }
  return true;
}

bool IRTranslator::translateInvoke(const User &U) {
  const InvokeInst &I = cast<InvokeInst>(U);
  MachineFunction &MF = MIRBuilder.getMF();
  MachineModuleInfo &MMI = MF.getMMI();

  const BasicBlock *ReturnBB = I.getSuccessor(0);
  const BasicBlock *EHPadBB = I.getSuccessor(1);

  const Value *Callee(I.getCalledValue());
  const Function *Fn = dyn_cast<Function>(Callee);
  if (isa<InlineAsm>(Callee))
    return false;

  // FIXME: support invoking patchpoint and statepoint intrinsics.
  if (Fn && Fn->isIntrinsic())
    return false;

  // FIXME: support whatever these are.
  if (I.countOperandBundlesOfType(LLVMContext::OB_deopt))
    return false;

  // FIXME: support Windows exception handling.
  if (!isa<LandingPadInst>(EHPadBB->front()))
    return false;


  // Emit the actual call, bracketed by EH_LABELs so that the MMI knows about
  // the region covered by the try.
  MCSymbol *BeginSymbol = MMI.getContext().createTempSymbol();
  MIRBuilder.buildInstr(TargetOpcode::EH_LABEL).addSym(BeginSymbol);

  unsigned Res = I.getType()->isVoidTy() ? 0 : getOrCreateVReg(I);
  SmallVector<CallLowering::ArgInfo, 8> Args;
  for (auto &Arg: I.arg_operands())
    Args.emplace_back(getOrCreateVReg(*Arg), Arg->getType());

  if (!CLI->lowerCall(MIRBuilder, MachineOperand::CreateGA(Fn, 0),
                      CallLowering::ArgInfo(Res, I.getType()), Args))
    return false;

  MCSymbol *EndSymbol = MMI.getContext().createTempSymbol();
  MIRBuilder.buildInstr(TargetOpcode::EH_LABEL).addSym(EndSymbol);

  // FIXME: track probabilities.
  MachineBasicBlock &EHPadMBB = getOrCreateBB(*EHPadBB),
                    &ReturnMBB = getOrCreateBB(*ReturnBB);
  MMI.addInvoke(&EHPadMBB, BeginSymbol, EndSymbol);
  MIRBuilder.getMBB().addSuccessor(&ReturnMBB);
  MIRBuilder.getMBB().addSuccessor(&EHPadMBB);

  return true;
}

bool IRTranslator::translateLandingPad(const User &U) {
  const LandingPadInst &LP = cast<LandingPadInst>(U);

  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  MachineFunction &MF = MIRBuilder.getMF();
  MachineModuleInfo &MMI = MF.getMMI();
  AddLandingPadInfo(LP, MMI, &MBB);

  MBB.setIsEHPad();

  // If there aren't registers to copy the values into (e.g., during SjLj
  // exceptions), then don't bother.
  auto &TLI = *MF.getSubtarget().getTargetLowering();
  const Constant *PersonalityFn = MF.getFunction()->getPersonalityFn();
  if (TLI.getExceptionPointerRegister(PersonalityFn) == 0 &&
      TLI.getExceptionSelectorRegister(PersonalityFn) == 0)
    return true;

  // If landingpad's return type is token type, we don't create DAG nodes
  // for its exception pointer and selector value. The extraction of exception
  // pointer or selector value from token type landingpads is not currently
  // supported.
  if (LP.getType()->isTokenTy())
    return true;

  // Add a label to mark the beginning of the landing pad.  Deletion of the
  // landing pad can thus be detected via the MachineModuleInfo.
  MIRBuilder.buildInstr(TargetOpcode::EH_LABEL)
    .addSym(MMI.addLandingPad(&MBB));

  // Mark exception register as live in.
  SmallVector<unsigned, 2> Regs;
  SmallVector<uint64_t, 2> Offsets;
  LLT p0 = LLT::pointer(0, DL->getPointerSizeInBits());
  if (unsigned Reg = TLI.getExceptionPointerRegister(PersonalityFn)) {
    unsigned VReg = MRI->createGenericVirtualRegister(p0);
    MIRBuilder.buildCopy(VReg, Reg);
    Regs.push_back(VReg);
    Offsets.push_back(0);
  }

  if (unsigned Reg = TLI.getExceptionSelectorRegister(PersonalityFn)) {
    unsigned VReg = MRI->createGenericVirtualRegister(p0);
    MIRBuilder.buildCopy(VReg, Reg);
    Regs.push_back(VReg);
    Offsets.push_back(p0.getSizeInBits());
  }

  MIRBuilder.buildSequence(getOrCreateVReg(LP), Regs, Offsets);
  return true;
}

bool IRTranslator::translateStaticAlloca(const AllocaInst &AI) {
  if (!TPC->isGlobalISelAbortEnabled() && !AI.isStaticAlloca())
    return false;

  assert(AI.isStaticAlloca() && "only handle static allocas now");
  unsigned Res = getOrCreateVReg(AI);
  int FI = getOrCreateFrameIndex(AI);
  MIRBuilder.buildFrameIndex(Res, FI);
  return true;
}

bool IRTranslator::translatePHI(const User &U) {
  const PHINode &PI = cast<PHINode>(U);
  auto MIB = MIRBuilder.buildInstr(TargetOpcode::PHI);
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
    EntryBuilder.buildConstant(Reg, CI->getZExtValue());
  else if (auto CF = dyn_cast<ConstantFP>(&C))
    EntryBuilder.buildFConstant(Reg, *CF);
  else if (isa<UndefValue>(C))
    EntryBuilder.buildInstr(TargetOpcode::IMPLICIT_DEF).addDef(Reg);
  else if (isa<ConstantPointerNull>(C))
    EntryBuilder.buildInstr(TargetOpcode::G_CONSTANT)
        .addDef(Reg)
        .addImm(0);
  else if (auto GV = dyn_cast<GlobalValue>(&C))
    EntryBuilder.buildGlobalValue(Reg, GV);
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
  FrameIndices.clear();
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
  bool Succeeded = CLI->lowerFormalArguments(MIRBuilder, F, VRegArgs);
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
        if (TPC->isGlobalISelAbortEnabled())
          reportTranslationError(Inst, "unable to translate instruction");
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
