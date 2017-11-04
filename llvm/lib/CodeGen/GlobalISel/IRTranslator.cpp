//===- llvm/CodeGen/GlobalISel/IRTranslator.cpp - IRTranslator ---*- C++ -*-==//
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LowLevelTypeImpl.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#define DEBUG_TYPE "irtranslator"

using namespace llvm;

char IRTranslator::ID = 0;

INITIALIZE_PASS_BEGIN(IRTranslator, DEBUG_TYPE, "IRTranslator LLVM IR -> MI",
                false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(IRTranslator, DEBUG_TYPE, "IRTranslator LLVM IR -> MI",
                false, false)

static void reportTranslationError(MachineFunction &MF,
                                   const TargetPassConfig &TPC,
                                   OptimizationRemarkEmitter &ORE,
                                   OptimizationRemarkMissed &R) {
  MF.getProperties().set(MachineFunctionProperties::Property::FailedISel);

  // Print the function name explicitly if we don't have a debug location (which
  // makes the diagnostic less useful) or if we're going to emit a raw error.
  if (!R.getLocation().isValid() || TPC.isGlobalISelAbortEnabled())
    R << (" (in function: " + MF.getName() + ")").str();

  if (TPC.isGlobalISelAbortEnabled())
    report_fatal_error(R.getMsg());
  else
    ORE.emit(R);
}

IRTranslator::IRTranslator() : MachineFunctionPass(ID) {
  initializeIRTranslatorPass(*PassRegistry::getPassRegistry());
}

void IRTranslator::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

unsigned IRTranslator::getOrCreateVReg(const Value &Val) {
  unsigned &ValReg = ValToVReg[&Val];

  if (ValReg)
    return ValReg;

  // Fill ValRegsSequence with the sequence of registers
  // we need to concat together to produce the value.
  assert(Val.getType()->isSized() &&
         "Don't know how to create an empty vreg");
  unsigned VReg =
      MRI->createGenericVirtualRegister(getLLTForType(*Val.getType(), *DL));
  ValReg = VReg;

  if (auto CV = dyn_cast<Constant>(&Val)) {
    bool Success = translate(*CV, VReg);
    if (!Success) {
      OptimizationRemarkMissed R("gisel-irtranslator", "GISelFailure",
                                 MF->getFunction()->getSubprogram(),
                                 &MF->getFunction()->getEntryBlock());
      R << "unable to translate constant: " << ore::NV("Type", Val.getType());
      reportTranslationError(*MF, *TPC, *ORE, R);
      return VReg;
    }
  }

  return VReg;
}

int IRTranslator::getOrCreateFrameIndex(const AllocaInst &AI) {
  if (FrameIndices.find(&AI) != FrameIndices.end())
    return FrameIndices[&AI];

  unsigned ElementSize = DL->getTypeStoreSize(AI.getAllocatedType());
  unsigned Size =
      ElementSize * cast<ConstantInt>(AI.getArraySize())->getZExtValue();

  // Always allocate at least one byte.
  Size = std::max(Size, 1u);

  unsigned Alignment = AI.getAlignment();
  if (!Alignment)
    Alignment = DL->getABITypeAlignment(AI.getAllocatedType());

  int &FI = FrameIndices[&AI];
  FI = MF->getFrameInfo().CreateStackObject(Size, Alignment, false, &AI);
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
  } else {
    OptimizationRemarkMissed R("gisel-irtranslator", "", &I);
    R << "unable to translate memop: " << ore::NV("Opcode", &I);
    reportTranslationError(*MF, *TPC, *ORE, R);
    return 1;
  }

  return Alignment ? Alignment : DL->getABITypeAlignment(ValTy);
}

MachineBasicBlock &IRTranslator::getMBB(const BasicBlock &BB) {
  MachineBasicBlock *&MBB = BBToMBB[&BB];
  assert(MBB && "BasicBlock was not encountered before");
  return *MBB;
}

void IRTranslator::addMachineCFGPred(CFGEdge Edge, MachineBasicBlock *NewPred) {
  assert(NewPred && "new predecessor must be a real MachineBasicBlock");
  MachinePreds[Edge].push_back(NewPred);
}

bool IRTranslator::translateBinaryOp(unsigned Opcode, const User &U,
                                     MachineIRBuilder &MIRBuilder) {
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

bool IRTranslator::translateFSub(const User &U, MachineIRBuilder &MIRBuilder) {
  // -0.0 - X --> G_FNEG
  if (isa<Constant>(U.getOperand(0)) &&
      U.getOperand(0) == ConstantFP::getZeroValueForNegation(U.getType())) {
    MIRBuilder.buildInstr(TargetOpcode::G_FNEG)
        .addDef(getOrCreateVReg(U))
        .addUse(getOrCreateVReg(*U.getOperand(1)));
    return true;
  }
  return translateBinaryOp(TargetOpcode::G_FSUB, U, MIRBuilder);
}

bool IRTranslator::translateCompare(const User &U,
                                    MachineIRBuilder &MIRBuilder) {
  const CmpInst *CI = dyn_cast<CmpInst>(&U);
  unsigned Op0 = getOrCreateVReg(*U.getOperand(0));
  unsigned Op1 = getOrCreateVReg(*U.getOperand(1));
  unsigned Res = getOrCreateVReg(U);
  CmpInst::Predicate Pred =
      CI ? CI->getPredicate() : static_cast<CmpInst::Predicate>(
                                    cast<ConstantExpr>(U).getPredicate());
  if (CmpInst::isIntPredicate(Pred))
    MIRBuilder.buildICmp(Pred, Res, Op0, Op1);
  else if (Pred == CmpInst::FCMP_FALSE)
    MIRBuilder.buildCopy(
        Res, getOrCreateVReg(*Constant::getNullValue(CI->getType())));
  else if (Pred == CmpInst::FCMP_TRUE)
    MIRBuilder.buildCopy(
        Res, getOrCreateVReg(*Constant::getAllOnesValue(CI->getType())));
  else
    MIRBuilder.buildFCmp(Pred, Res, Op0, Op1);

  return true;
}

bool IRTranslator::translateRet(const User &U, MachineIRBuilder &MIRBuilder) {
  const ReturnInst &RI = cast<ReturnInst>(U);
  const Value *Ret = RI.getReturnValue();
  // The target may mess up with the insertion point, but
  // this is not important as a return is the last instruction
  // of the block anyway.
  return CLI->lowerReturn(MIRBuilder, Ret, !Ret ? 0 : getOrCreateVReg(*Ret));
}

bool IRTranslator::translateBr(const User &U, MachineIRBuilder &MIRBuilder) {
  const BranchInst &BrInst = cast<BranchInst>(U);
  unsigned Succ = 0;
  if (!BrInst.isUnconditional()) {
    // We want a G_BRCOND to the true BB followed by an unconditional branch.
    unsigned Tst = getOrCreateVReg(*BrInst.getCondition());
    const BasicBlock &TrueTgt = *cast<BasicBlock>(BrInst.getSuccessor(Succ++));
    MachineBasicBlock &TrueBB = getMBB(TrueTgt);
    MIRBuilder.buildBrCond(Tst, TrueBB);
  }

  const BasicBlock &BrTgt = *cast<BasicBlock>(BrInst.getSuccessor(Succ));
  MachineBasicBlock &TgtBB = getMBB(BrTgt);
  MachineBasicBlock &CurBB = MIRBuilder.getMBB();

  // If the unconditional target is the layout successor, fallthrough.
  if (!CurBB.isLayoutSuccessor(&TgtBB))
    MIRBuilder.buildBr(TgtBB);

  // Link successors.
  for (const BasicBlock *Succ : BrInst.successors())
    CurBB.addSuccessor(&getMBB(*Succ));
  return true;
}

bool IRTranslator::translateSwitch(const User &U,
                                   MachineIRBuilder &MIRBuilder) {
  // For now, just translate as a chain of conditional branches.
  // FIXME: could we share most of the logic/code in
  // SelectionDAGBuilder::visitSwitch between SelectionDAG and GlobalISel?
  // At first sight, it seems most of the logic in there is independent of
  // SelectionDAG-specifics and a lot of work went in to optimize switch
  // lowering in there.

  const SwitchInst &SwInst = cast<SwitchInst>(U);
  const unsigned SwCondValue = getOrCreateVReg(*SwInst.getCondition());
  const BasicBlock *OrigBB = SwInst.getParent();

  LLT LLTi1 = getLLTForType(*Type::getInt1Ty(U.getContext()), *DL);
  for (auto &CaseIt : SwInst.cases()) {
    const unsigned CaseValueReg = getOrCreateVReg(*CaseIt.getCaseValue());
    const unsigned Tst = MRI->createGenericVirtualRegister(LLTi1);
    MIRBuilder.buildICmp(CmpInst::ICMP_EQ, Tst, CaseValueReg, SwCondValue);
    MachineBasicBlock &CurMBB = MIRBuilder.getMBB();
    const BasicBlock *TrueBB = CaseIt.getCaseSuccessor();
    MachineBasicBlock &TrueMBB = getMBB(*TrueBB);

    MIRBuilder.buildBrCond(Tst, TrueMBB);
    CurMBB.addSuccessor(&TrueMBB);
    addMachineCFGPred({OrigBB, TrueBB}, &CurMBB);

    MachineBasicBlock *FalseMBB =
        MF->CreateMachineBasicBlock(SwInst.getParent());
    // Insert the comparison blocks one after the other.
    MF->insert(std::next(CurMBB.getIterator()), FalseMBB);
    MIRBuilder.buildBr(*FalseMBB);
    CurMBB.addSuccessor(FalseMBB);

    MIRBuilder.setMBB(*FalseMBB);
  }
  // handle default case
  const BasicBlock *DefaultBB = SwInst.getDefaultDest();
  MachineBasicBlock &DefaultMBB = getMBB(*DefaultBB);
  MIRBuilder.buildBr(DefaultMBB);
  MachineBasicBlock &CurMBB = MIRBuilder.getMBB();
  CurMBB.addSuccessor(&DefaultMBB);
  addMachineCFGPred({OrigBB, DefaultBB}, &CurMBB);

  return true;
}

bool IRTranslator::translateIndirectBr(const User &U,
                                       MachineIRBuilder &MIRBuilder) {
  const IndirectBrInst &BrInst = cast<IndirectBrInst>(U);

  const unsigned Tgt = getOrCreateVReg(*BrInst.getAddress());
  MIRBuilder.buildBrIndirect(Tgt);

  // Link successors.
  MachineBasicBlock &CurBB = MIRBuilder.getMBB();
  for (const BasicBlock *Succ : BrInst.successors())
    CurBB.addSuccessor(&getMBB(*Succ));

  return true;
}

bool IRTranslator::translateLoad(const User &U, MachineIRBuilder &MIRBuilder) {
  const LoadInst &LI = cast<LoadInst>(U);

  auto Flags = LI.isVolatile() ? MachineMemOperand::MOVolatile
                               : MachineMemOperand::MONone;
  Flags |= MachineMemOperand::MOLoad;

  unsigned Res = getOrCreateVReg(LI);
  unsigned Addr = getOrCreateVReg(*LI.getPointerOperand());

  MIRBuilder.buildLoad(
      Res, Addr,
      *MF->getMachineMemOperand(MachinePointerInfo(LI.getPointerOperand()),
                                Flags, DL->getTypeStoreSize(LI.getType()),
                                getMemOpAlignment(LI), AAMDNodes(), nullptr,
                                LI.getSyncScopeID(), LI.getOrdering()));
  return true;
}

bool IRTranslator::translateStore(const User &U, MachineIRBuilder &MIRBuilder) {
  const StoreInst &SI = cast<StoreInst>(U);
  auto Flags = SI.isVolatile() ? MachineMemOperand::MOVolatile
                               : MachineMemOperand::MONone;
  Flags |= MachineMemOperand::MOStore;

  unsigned Val = getOrCreateVReg(*SI.getValueOperand());
  unsigned Addr = getOrCreateVReg(*SI.getPointerOperand());

  MIRBuilder.buildStore(
      Val, Addr,
      *MF->getMachineMemOperand(
          MachinePointerInfo(SI.getPointerOperand()), Flags,
          DL->getTypeStoreSize(SI.getValueOperand()->getType()),
          getMemOpAlignment(SI), AAMDNodes(), nullptr, SI.getSyncScopeID(),
          SI.getOrdering()));
  return true;
}

bool IRTranslator::translateExtractValue(const User &U,
                                         MachineIRBuilder &MIRBuilder) {
  const Value *Src = U.getOperand(0);
  Type *Int32Ty = Type::getInt32Ty(U.getContext());
  SmallVector<Value *, 1> Indices;

  // If Src is a single element ConstantStruct, translate extractvalue
  // to that element to avoid inserting a cast instruction.
  if (auto CS = dyn_cast<ConstantStruct>(Src))
    if (CS->getNumOperands() == 1) {
      unsigned Res = getOrCreateVReg(*CS->getOperand(0));
      ValToVReg[&U] = Res;
      return true;
    }

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
  MIRBuilder.buildExtract(Res, getOrCreateVReg(*Src), Offset);

  return true;
}

bool IRTranslator::translateInsertValue(const User &U,
                                        MachineIRBuilder &MIRBuilder) {
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
  unsigned Inserted = getOrCreateVReg(*U.getOperand(1));
  MIRBuilder.buildInsert(Res, getOrCreateVReg(*Src), Inserted, Offset);

  return true;
}

bool IRTranslator::translateSelect(const User &U,
                                   MachineIRBuilder &MIRBuilder) {
  unsigned Res = getOrCreateVReg(U);
  unsigned Tst = getOrCreateVReg(*U.getOperand(0));
  unsigned Op0 = getOrCreateVReg(*U.getOperand(1));
  unsigned Op1 = getOrCreateVReg(*U.getOperand(2));
  MIRBuilder.buildSelect(Res, Tst, Op0, Op1);
  return true;
}

bool IRTranslator::translateBitCast(const User &U,
                                    MachineIRBuilder &MIRBuilder) {
  // If we're bitcasting to the source type, we can reuse the source vreg.
  if (getLLTForType(*U.getOperand(0)->getType(), *DL) ==
      getLLTForType(*U.getType(), *DL)) {
    // Get the source vreg now, to avoid invalidating ValToVReg.
    unsigned SrcReg = getOrCreateVReg(*U.getOperand(0));
    unsigned &Reg = ValToVReg[&U];
    // If we already assigned a vreg for this bitcast, we can't change that.
    // Emit a copy to satisfy the users we already emitted.
    if (Reg)
      MIRBuilder.buildCopy(Reg, SrcReg);
    else
      Reg = SrcReg;
    return true;
  }
  return translateCast(TargetOpcode::G_BITCAST, U, MIRBuilder);
}

bool IRTranslator::translateCast(unsigned Opcode, const User &U,
                                 MachineIRBuilder &MIRBuilder) {
  unsigned Op = getOrCreateVReg(*U.getOperand(0));
  unsigned Res = getOrCreateVReg(U);
  MIRBuilder.buildInstr(Opcode).addDef(Res).addUse(Op);
  return true;
}

bool IRTranslator::translateGetElementPtr(const User &U,
                                          MachineIRBuilder &MIRBuilder) {
  // FIXME: support vector GEPs.
  if (U.getType()->isVectorTy())
    return false;

  Value &Op0 = *U.getOperand(0);
  unsigned BaseReg = getOrCreateVReg(Op0);
  Type *PtrIRTy = Op0.getType();
  LLT PtrTy = getLLTForType(*PtrIRTy, *DL);
  Type *OffsetIRTy = DL->getIntPtrType(PtrIRTy);
  LLT OffsetTy = getLLTForType(*OffsetIRTy, *DL);

  int64_t Offset = 0;
  for (gep_type_iterator GTI = gep_type_begin(&U), E = gep_type_end(&U);
       GTI != E; ++GTI) {
    const Value *Idx = GTI.getOperand();
    if (StructType *StTy = GTI.getStructTypeOrNull()) {
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
        unsigned OffsetReg =
            getOrCreateVReg(*ConstantInt::get(OffsetIRTy, Offset));
        MIRBuilder.buildGEP(NewBaseReg, BaseReg, OffsetReg);

        BaseReg = NewBaseReg;
        Offset = 0;
      }

      // N = N + Idx * ElementSize;
      unsigned ElementSizeReg =
          getOrCreateVReg(*ConstantInt::get(OffsetIRTy, ElementSize));

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
    unsigned OffsetReg = getOrCreateVReg(*ConstantInt::get(OffsetIRTy, Offset));
    MIRBuilder.buildGEP(getOrCreateVReg(U), BaseReg, OffsetReg);
    return true;
  }

  MIRBuilder.buildCopy(getOrCreateVReg(U), BaseReg);
  return true;
}

bool IRTranslator::translateMemfunc(const CallInst &CI,
                                    MachineIRBuilder &MIRBuilder,
                                    unsigned ID) {
  LLT SizeTy = getLLTForType(*CI.getArgOperand(2)->getType(), *DL);
  Type *DstTy = CI.getArgOperand(0)->getType();
  if (cast<PointerType>(DstTy)->getAddressSpace() != 0 ||
      SizeTy.getSizeInBits() != DL->getPointerSizeInBits(0))
    return false;

  SmallVector<CallLowering::ArgInfo, 8> Args;
  for (int i = 0; i < 3; ++i) {
    const auto &Arg = CI.getArgOperand(i);
    Args.emplace_back(getOrCreateVReg(*Arg), Arg->getType());
  }

  const char *Callee;
  switch (ID) {
  case Intrinsic::memmove:
  case Intrinsic::memcpy: {
    Type *SrcTy = CI.getArgOperand(1)->getType();
    if(cast<PointerType>(SrcTy)->getAddressSpace() != 0)
      return false;
    Callee = ID == Intrinsic::memcpy ? "memcpy" : "memmove";
    break;
  }
  case Intrinsic::memset:
    Callee = "memset";
    break;
  default:
    return false;
  }

  return CLI->lowerCall(MIRBuilder, CI.getCallingConv(),
                        MachineOperand::CreateES(Callee),
                        CallLowering::ArgInfo(0, CI.getType()), Args);
}

void IRTranslator::getStackGuard(unsigned DstReg,
                                 MachineIRBuilder &MIRBuilder) {
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  MRI->setRegClass(DstReg, TRI->getPointerRegClass(*MF));
  auto MIB = MIRBuilder.buildInstr(TargetOpcode::LOAD_STACK_GUARD);
  MIB.addDef(DstReg);

  auto &TLI = *MF->getSubtarget().getTargetLowering();
  Value *Global = TLI.getSDagStackGuard(*MF->getFunction()->getParent());
  if (!Global)
    return;

  MachinePointerInfo MPInfo(Global);
  MachineInstr::mmo_iterator MemRefs = MF->allocateMemRefsArray(1);
  auto Flags = MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant |
               MachineMemOperand::MODereferenceable;
  *MemRefs =
      MF->getMachineMemOperand(MPInfo, Flags, DL->getPointerSizeInBits() / 8,
                               DL->getPointerABIAlignment());
  MIB.setMemRefs(MemRefs, MemRefs + 1);
}

bool IRTranslator::translateOverflowIntrinsic(const CallInst &CI, unsigned Op,
                                              MachineIRBuilder &MIRBuilder) {
  LLT Ty = getLLTForType(*CI.getOperand(0)->getType(), *DL);
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
    unsigned Zero = getOrCreateVReg(
        *Constant::getNullValue(Type::getInt1Ty(CI.getContext())));
    MIB.addUse(Zero);
  }

  MIRBuilder.buildSequence(getOrCreateVReg(CI), {Res, Overflow}, {0, Width});
  return true;
}

bool IRTranslator::translateKnownIntrinsic(const CallInst &CI, Intrinsic::ID ID,
                                           MachineIRBuilder &MIRBuilder) {
  switch (ID) {
  default:
    break;
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
    // Stack coloring is not enabled in O0 (which we care about now) so we can
    // drop these. Make sure someone notices when we start compiling at higher
    // opts though.
    if (MF->getTarget().getOptLevel() != CodeGenOpt::None)
      return false;
    return true;
  case Intrinsic::dbg_declare: {
    const DbgDeclareInst &DI = cast<DbgDeclareInst>(CI);
    assert(DI.getVariable() && "Missing variable");

    const Value *Address = DI.getAddress();
    if (!Address || isa<UndefValue>(Address)) {
      DEBUG(dbgs() << "Dropping debug info for " << DI << "\n");
      return true;
    }

    assert(DI.getVariable()->isValidLocationForIntrinsic(
               MIRBuilder.getDebugLoc()) &&
           "Expected inlined-at fields to agree");
    auto AI = dyn_cast<AllocaInst>(Address);
    if (AI && AI->isStaticAlloca()) {
      // Static allocas are tracked at the MF level, no need for DBG_VALUE
      // instructions (in fact, they get ignored if they *do* exist).
      MF->setVariableDbgInfo(DI.getVariable(), DI.getExpression(),
                             getOrCreateFrameIndex(*AI), DI.getDebugLoc());
    } else
      MIRBuilder.buildDirectDbgValue(getOrCreateVReg(*Address),
                                     DI.getVariable(), DI.getExpression());
    return true;
  }
  case Intrinsic::vaend:
    // No target I know of cares about va_end. Certainly no in-tree target
    // does. Simplest intrinsic ever!
    return true;
  case Intrinsic::vastart: {
    auto &TLI = *MF->getSubtarget().getTargetLowering();
    Value *Ptr = CI.getArgOperand(0);
    unsigned ListSize = TLI.getVaListSizeInBits(*DL) / 8;

    MIRBuilder.buildInstr(TargetOpcode::G_VASTART)
        .addUse(getOrCreateVReg(*Ptr))
        .addMemOperand(MF->getMachineMemOperand(
            MachinePointerInfo(Ptr), MachineMemOperand::MOStore, ListSize, 0));
    return true;
  }
  case Intrinsic::dbg_value: {
    // This form of DBG_VALUE is target-independent.
    const DbgValueInst &DI = cast<DbgValueInst>(CI);
    const Value *V = DI.getValue();
    assert(DI.getVariable()->isValidLocationForIntrinsic(
               MIRBuilder.getDebugLoc()) &&
           "Expected inlined-at fields to agree");
    if (!V) {
      // Currently the optimizer can produce this; insert an undef to
      // help debugging.  Probably the optimizer should not do this.
      MIRBuilder.buildIndirectDbgValue(0, DI.getVariable(), DI.getExpression());
    } else if (const auto *CI = dyn_cast<Constant>(V)) {
      MIRBuilder.buildConstDbgValue(*CI, DI.getVariable(), DI.getExpression());
    } else {
      unsigned Reg = getOrCreateVReg(*V);
      // FIXME: This does not handle register-indirect values at offset 0. The
      // direct/indirect thing shouldn't really be handled by something as
      // implicit as reg+noreg vs reg+imm in the first palce, but it seems
      // pretty baked in right now.
      MIRBuilder.buildDirectDbgValue(Reg, DI.getVariable(), DI.getExpression());
    }
    return true;
  }
  case Intrinsic::uadd_with_overflow:
    return translateOverflowIntrinsic(CI, TargetOpcode::G_UADDE, MIRBuilder);
  case Intrinsic::sadd_with_overflow:
    return translateOverflowIntrinsic(CI, TargetOpcode::G_SADDO, MIRBuilder);
  case Intrinsic::usub_with_overflow:
    return translateOverflowIntrinsic(CI, TargetOpcode::G_USUBE, MIRBuilder);
  case Intrinsic::ssub_with_overflow:
    return translateOverflowIntrinsic(CI, TargetOpcode::G_SSUBO, MIRBuilder);
  case Intrinsic::umul_with_overflow:
    return translateOverflowIntrinsic(CI, TargetOpcode::G_UMULO, MIRBuilder);
  case Intrinsic::smul_with_overflow:
    return translateOverflowIntrinsic(CI, TargetOpcode::G_SMULO, MIRBuilder);
  case Intrinsic::pow:
    MIRBuilder.buildInstr(TargetOpcode::G_FPOW)
        .addDef(getOrCreateVReg(CI))
        .addUse(getOrCreateVReg(*CI.getArgOperand(0)))
        .addUse(getOrCreateVReg(*CI.getArgOperand(1)));
    return true;
  case Intrinsic::exp:
    MIRBuilder.buildInstr(TargetOpcode::G_FEXP)
        .addDef(getOrCreateVReg(CI))
        .addUse(getOrCreateVReg(*CI.getArgOperand(0)));
    return true;
  case Intrinsic::exp2:
    MIRBuilder.buildInstr(TargetOpcode::G_FEXP2)
        .addDef(getOrCreateVReg(CI))
        .addUse(getOrCreateVReg(*CI.getArgOperand(0)));
    return true;
  case Intrinsic::log:
    MIRBuilder.buildInstr(TargetOpcode::G_FLOG)
        .addDef(getOrCreateVReg(CI))
        .addUse(getOrCreateVReg(*CI.getArgOperand(0)));
    return true;
  case Intrinsic::log2:
    MIRBuilder.buildInstr(TargetOpcode::G_FLOG2)
        .addDef(getOrCreateVReg(CI))
        .addUse(getOrCreateVReg(*CI.getArgOperand(0)));
    return true;
  case Intrinsic::fma:
    MIRBuilder.buildInstr(TargetOpcode::G_FMA)
        .addDef(getOrCreateVReg(CI))
        .addUse(getOrCreateVReg(*CI.getArgOperand(0)))
        .addUse(getOrCreateVReg(*CI.getArgOperand(1)))
        .addUse(getOrCreateVReg(*CI.getArgOperand(2)));
    return true;
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset:
    return translateMemfunc(CI, MIRBuilder, ID);
  case Intrinsic::eh_typeid_for: {
    GlobalValue *GV = ExtractTypeInfo(CI.getArgOperand(0));
    unsigned Reg = getOrCreateVReg(CI);
    unsigned TypeID = MF->getTypeIDFor(GV);
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
    getStackGuard(getOrCreateVReg(CI), MIRBuilder);
    return true;
  case Intrinsic::stackprotector: {
    LLT PtrTy = getLLTForType(*CI.getArgOperand(0)->getType(), *DL);
    unsigned GuardVal = MRI->createGenericVirtualRegister(PtrTy);
    getStackGuard(GuardVal, MIRBuilder);

    AllocaInst *Slot = cast<AllocaInst>(CI.getArgOperand(1));
    MIRBuilder.buildStore(
        GuardVal, getOrCreateVReg(*Slot),
        *MF->getMachineMemOperand(
            MachinePointerInfo::getFixedStack(*MF,
                                              getOrCreateFrameIndex(*Slot)),
            MachineMemOperand::MOStore | MachineMemOperand::MOVolatile,
            PtrTy.getSizeInBits() / 8, 8));
    return true;
  }
  }
  return false;
}

bool IRTranslator::translateInlineAsm(const CallInst &CI,
                                      MachineIRBuilder &MIRBuilder) {
  const InlineAsm &IA = cast<InlineAsm>(*CI.getCalledValue());
  if (!IA.getConstraintString().empty())
    return false;

  unsigned ExtraInfo = 0;
  if (IA.hasSideEffects())
    ExtraInfo |= InlineAsm::Extra_HasSideEffects;
  if (IA.getDialect() == InlineAsm::AD_Intel)
    ExtraInfo |= InlineAsm::Extra_AsmDialect;

  MIRBuilder.buildInstr(TargetOpcode::INLINEASM)
    .addExternalSymbol(IA.getAsmString().c_str())
    .addImm(ExtraInfo);

  return true;
}

bool IRTranslator::translateCall(const User &U, MachineIRBuilder &MIRBuilder) {
  const CallInst &CI = cast<CallInst>(U);
  auto TII = MF->getTarget().getIntrinsicInfo();
  const Function *F = CI.getCalledFunction();

  if (CI.isInlineAsm())
    return translateInlineAsm(CI, MIRBuilder);

  if (!F || !F->isIntrinsic()) {
    unsigned Res = CI.getType()->isVoidTy() ? 0 : getOrCreateVReg(CI);
    SmallVector<unsigned, 8> Args;
    for (auto &Arg: CI.arg_operands())
      Args.push_back(getOrCreateVReg(*Arg));

    MF->getFrameInfo().setHasCalls(true);
    return CLI->lowerCall(MIRBuilder, &CI, Res, Args, [&]() {
      return getOrCreateVReg(*CI.getCalledValue());
    });
  }

  Intrinsic::ID ID = F->getIntrinsicID();
  if (TII && ID == Intrinsic::not_intrinsic)
    ID = static_cast<Intrinsic::ID>(TII->getIntrinsicID(F));

  assert(ID != Intrinsic::not_intrinsic && "unknown intrinsic");

  if (translateKnownIntrinsic(CI, ID, MIRBuilder))
    return true;

  unsigned Res = CI.getType()->isVoidTy() ? 0 : getOrCreateVReg(CI);
  MachineInstrBuilder MIB =
      MIRBuilder.buildIntrinsic(ID, Res, !CI.doesNotAccessMemory());

  for (auto &Arg : CI.arg_operands()) {
    // Some intrinsics take metadata parameters. Reject them.
    if (isa<MetadataAsValue>(Arg))
      return false;
    MIB.addUse(getOrCreateVReg(*Arg));
  }

  // Add a MachineMemOperand if it is a target mem intrinsic.
  const TargetLowering &TLI = *MF->getSubtarget().getTargetLowering();
  TargetLowering::IntrinsicInfo Info;
  // TODO: Add a GlobalISel version of getTgtMemIntrinsic.
  if (TLI.getTgtMemIntrinsic(Info, CI, ID)) {
    MachineMemOperand::Flags Flags =
        Info.vol ? MachineMemOperand::MOVolatile : MachineMemOperand::MONone;
    Flags |=
        Info.readMem ? MachineMemOperand::MOLoad : MachineMemOperand::MOStore;
    uint64_t Size = Info.memVT.getSizeInBits() >> 3;
    MIB.addMemOperand(MF->getMachineMemOperand(MachinePointerInfo(Info.ptrVal),
                                               Flags, Size, Info.align));
  }

  return true;
}

bool IRTranslator::translateInvoke(const User &U,
                                   MachineIRBuilder &MIRBuilder) {
  const InvokeInst &I = cast<InvokeInst>(U);
  MCContext &Context = MF->getContext();

  const BasicBlock *ReturnBB = I.getSuccessor(0);
  const BasicBlock *EHPadBB = I.getSuccessor(1);

  const Value *Callee = I.getCalledValue();
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

  // Emit the actual call, bracketed by EH_LABELs so that the MF knows about
  // the region covered by the try.
  MCSymbol *BeginSymbol = Context.createTempSymbol();
  MIRBuilder.buildInstr(TargetOpcode::EH_LABEL).addSym(BeginSymbol);

  unsigned Res = I.getType()->isVoidTy() ? 0 : getOrCreateVReg(I);
  SmallVector<unsigned, 8> Args;
  for (auto &Arg: I.arg_operands())
    Args.push_back(getOrCreateVReg(*Arg));

  if (!CLI->lowerCall(MIRBuilder, &I, Res, Args,
                      [&]() { return getOrCreateVReg(*I.getCalledValue()); }))
    return false;

  MCSymbol *EndSymbol = Context.createTempSymbol();
  MIRBuilder.buildInstr(TargetOpcode::EH_LABEL).addSym(EndSymbol);

  // FIXME: track probabilities.
  MachineBasicBlock &EHPadMBB = getMBB(*EHPadBB),
                    &ReturnMBB = getMBB(*ReturnBB);
  MF->addInvoke(&EHPadMBB, BeginSymbol, EndSymbol);
  MIRBuilder.getMBB().addSuccessor(&ReturnMBB);
  MIRBuilder.getMBB().addSuccessor(&EHPadMBB);
  MIRBuilder.buildBr(ReturnMBB);

  return true;
}

bool IRTranslator::translateLandingPad(const User &U,
                                       MachineIRBuilder &MIRBuilder) {
  const LandingPadInst &LP = cast<LandingPadInst>(U);

  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  addLandingPadInfo(LP, MBB);

  MBB.setIsEHPad();

  // If there aren't registers to copy the values into (e.g., during SjLj
  // exceptions), then don't bother.
  auto &TLI = *MF->getSubtarget().getTargetLowering();
  const Constant *PersonalityFn = MF->getFunction()->getPersonalityFn();
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
    .addSym(MF->addLandingPad(&MBB));

  LLT Ty = getLLTForType(*LP.getType(), *DL);
  unsigned Undef = MRI->createGenericVirtualRegister(Ty);
  MIRBuilder.buildUndef(Undef);

  SmallVector<LLT, 2> Tys;
  for (Type *Ty : cast<StructType>(LP.getType())->elements())
    Tys.push_back(getLLTForType(*Ty, *DL));
  assert(Tys.size() == 2 && "Only two-valued landingpads are supported");

  // Mark exception register as live in.
  unsigned ExceptionReg = TLI.getExceptionPointerRegister(PersonalityFn);
  if (!ExceptionReg)
    return false;

  MBB.addLiveIn(ExceptionReg);
  unsigned VReg = MRI->createGenericVirtualRegister(Tys[0]),
           Tmp = MRI->createGenericVirtualRegister(Ty);
  MIRBuilder.buildCopy(VReg, ExceptionReg);
  MIRBuilder.buildInsert(Tmp, Undef, VReg, 0);

  unsigned SelectorReg = TLI.getExceptionSelectorRegister(PersonalityFn);
  if (!SelectorReg)
    return false;

  MBB.addLiveIn(SelectorReg);

  // N.b. the exception selector register always has pointer type and may not
  // match the actual IR-level type in the landingpad so an extra cast is
  // needed.
  unsigned PtrVReg = MRI->createGenericVirtualRegister(Tys[0]);
  MIRBuilder.buildCopy(PtrVReg, SelectorReg);

  VReg = MRI->createGenericVirtualRegister(Tys[1]);
  MIRBuilder.buildInstr(TargetOpcode::G_PTRTOINT).addDef(VReg).addUse(PtrVReg);
  MIRBuilder.buildInsert(getOrCreateVReg(LP), Tmp, VReg,
                         Tys[0].getSizeInBits());
  return true;
}

bool IRTranslator::translateAlloca(const User &U,
                                   MachineIRBuilder &MIRBuilder) {
  auto &AI = cast<AllocaInst>(U);

  if (AI.isStaticAlloca()) {
    unsigned Res = getOrCreateVReg(AI);
    int FI = getOrCreateFrameIndex(AI);
    MIRBuilder.buildFrameIndex(Res, FI);
    return true;
  }

  // Now we're in the harder dynamic case.
  Type *Ty = AI.getAllocatedType();
  unsigned Align =
      std::max((unsigned)DL->getPrefTypeAlignment(Ty), AI.getAlignment());

  unsigned NumElts = getOrCreateVReg(*AI.getArraySize());

  Type *IntPtrIRTy = DL->getIntPtrType(AI.getType());
  LLT IntPtrTy = getLLTForType(*IntPtrIRTy, *DL);
  if (MRI->getType(NumElts) != IntPtrTy) {
    unsigned ExtElts = MRI->createGenericVirtualRegister(IntPtrTy);
    MIRBuilder.buildZExtOrTrunc(ExtElts, NumElts);
    NumElts = ExtElts;
  }

  unsigned AllocSize = MRI->createGenericVirtualRegister(IntPtrTy);
  unsigned TySize =
      getOrCreateVReg(*ConstantInt::get(IntPtrIRTy, -DL->getTypeAllocSize(Ty)));
  MIRBuilder.buildMul(AllocSize, NumElts, TySize);

  LLT PtrTy = getLLTForType(*AI.getType(), *DL);
  auto &TLI = *MF->getSubtarget().getTargetLowering();
  unsigned SPReg = TLI.getStackPointerRegisterToSaveRestore();

  unsigned SPTmp = MRI->createGenericVirtualRegister(PtrTy);
  MIRBuilder.buildCopy(SPTmp, SPReg);

  unsigned AllocTmp = MRI->createGenericVirtualRegister(PtrTy);
  MIRBuilder.buildGEP(AllocTmp, SPTmp, AllocSize);

  // Handle alignment. We have to realign if the allocation granule was smaller
  // than stack alignment, or the specific alloca requires more than stack
  // alignment.
  unsigned StackAlign =
      MF->getSubtarget().getFrameLowering()->getStackAlignment();
  Align = std::max(Align, StackAlign);
  if (Align > StackAlign || DL->getTypeAllocSize(Ty) % StackAlign != 0) {
    // Round the size of the allocation up to the stack alignment size
    // by add SA-1 to the size. This doesn't overflow because we're computing
    // an address inside an alloca.
    unsigned AlignedAlloc = MRI->createGenericVirtualRegister(PtrTy);
    MIRBuilder.buildPtrMask(AlignedAlloc, AllocTmp, Log2_32(Align));
    AllocTmp = AlignedAlloc;
  }

  MIRBuilder.buildCopy(SPReg, AllocTmp);
  MIRBuilder.buildCopy(getOrCreateVReg(AI), AllocTmp);

  MF->getFrameInfo().CreateVariableSizedObject(Align ? Align : 1, &AI);
  assert(MF->getFrameInfo().hasVarSizedObjects());
  return true;
}

bool IRTranslator::translateVAArg(const User &U, MachineIRBuilder &MIRBuilder) {
  // FIXME: We may need more info about the type. Because of how LLT works,
  // we're completely discarding the i64/double distinction here (amongst
  // others). Fortunately the ABIs I know of where that matters don't use va_arg
  // anyway but that's not guaranteed.
  MIRBuilder.buildInstr(TargetOpcode::G_VAARG)
    .addDef(getOrCreateVReg(U))
    .addUse(getOrCreateVReg(*U.getOperand(0)))
    .addImm(DL->getABITypeAlignment(U.getType()));
  return true;
}

bool IRTranslator::translateInsertElement(const User &U,
                                          MachineIRBuilder &MIRBuilder) {
  // If it is a <1 x Ty> vector, use the scalar as it is
  // not a legal vector type in LLT.
  if (U.getType()->getVectorNumElements() == 1) {
    unsigned Elt = getOrCreateVReg(*U.getOperand(1));
    ValToVReg[&U] = Elt;
    return true;
  }
  unsigned Res = getOrCreateVReg(U);
  unsigned Val = getOrCreateVReg(*U.getOperand(0));
  unsigned Elt = getOrCreateVReg(*U.getOperand(1));
  unsigned Idx = getOrCreateVReg(*U.getOperand(2));
  MIRBuilder.buildInsertVectorElement(Res, Val, Elt, Idx);
  return true;
}

bool IRTranslator::translateExtractElement(const User &U,
                                           MachineIRBuilder &MIRBuilder) {
  // If it is a <1 x Ty> vector, use the scalar as it is
  // not a legal vector type in LLT.
  if (U.getOperand(0)->getType()->getVectorNumElements() == 1) {
    unsigned Elt = getOrCreateVReg(*U.getOperand(0));
    ValToVReg[&U] = Elt;
    return true;
  }
  unsigned Res = getOrCreateVReg(U);
  unsigned Val = getOrCreateVReg(*U.getOperand(0));
  unsigned Idx = getOrCreateVReg(*U.getOperand(1));
  MIRBuilder.buildExtractVectorElement(Res, Val, Idx);
  return true;
}

bool IRTranslator::translateShuffleVector(const User &U,
                                          MachineIRBuilder &MIRBuilder) {
  MIRBuilder.buildInstr(TargetOpcode::G_SHUFFLE_VECTOR)
      .addDef(getOrCreateVReg(U))
      .addUse(getOrCreateVReg(*U.getOperand(0)))
      .addUse(getOrCreateVReg(*U.getOperand(1)))
      .addUse(getOrCreateVReg(*U.getOperand(2)));
  return true;
}

bool IRTranslator::translatePHI(const User &U, MachineIRBuilder &MIRBuilder) {
  const PHINode &PI = cast<PHINode>(U);
  auto MIB = MIRBuilder.buildInstr(TargetOpcode::G_PHI);
  MIB.addDef(getOrCreateVReg(PI));

  PendingPHIs.emplace_back(&PI, MIB.getInstr());
  return true;
}

void IRTranslator::finishPendingPhis() {
  for (std::pair<const PHINode *, MachineInstr *> &Phi : PendingPHIs) {
    const PHINode *PI = Phi.first;
    MachineInstrBuilder MIB(*MF, Phi.second);

    // All MachineBasicBlocks exist, add them to the PHI. We assume IRTranslator
    // won't create extra control flow here, otherwise we need to find the
    // dominating predecessor here (or perhaps force the weirder IRTranslators
    // to provide a simple boundary).
    SmallSet<const BasicBlock *, 4> HandledPreds;

    for (unsigned i = 0; i < PI->getNumIncomingValues(); ++i) {
      auto IRPred = PI->getIncomingBlock(i);
      if (HandledPreds.count(IRPred))
        continue;

      HandledPreds.insert(IRPred);
      unsigned ValReg = getOrCreateVReg(*PI->getIncomingValue(i));
      for (auto Pred : getMachinePredBBs({IRPred, PI->getParent()})) {
        assert(Pred->isSuccessor(MIB->getParent()) &&
               "incorrect CFG at MachineBasicBlock level");
        MIB.addUse(ValReg);
        MIB.addMBB(Pred);
      }
    }
  }
}

bool IRTranslator::translate(const Instruction &Inst) {
  CurBuilder.setDebugLoc(Inst.getDebugLoc());
  switch(Inst.getOpcode()) {
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE: return translate##OPCODE(Inst, CurBuilder);
#include "llvm/IR/Instruction.def"
  default:
    return false;
  }
}

bool IRTranslator::translate(const Constant &C, unsigned Reg) {
  if (auto CI = dyn_cast<ConstantInt>(&C))
    EntryBuilder.buildConstant(Reg, *CI);
  else if (auto CF = dyn_cast<ConstantFP>(&C))
    EntryBuilder.buildFConstant(Reg, *CF);
  else if (isa<UndefValue>(C))
    EntryBuilder.buildUndef(Reg);
  else if (isa<ConstantPointerNull>(C))
    EntryBuilder.buildConstant(Reg, 0);
  else if (auto GV = dyn_cast<GlobalValue>(&C))
    EntryBuilder.buildGlobalValue(Reg, GV);
  else if (auto CAZ = dyn_cast<ConstantAggregateZero>(&C)) {
    if (!CAZ->getType()->isVectorTy())
      return false;
    // Return the scalar if it is a <1 x Ty> vector.
    if (CAZ->getNumElements() == 1)
      return translate(*CAZ->getElementValue(0u), Reg);
    std::vector<unsigned> Ops;
    for (unsigned i = 0; i < CAZ->getNumElements(); ++i) {
      Constant &Elt = *CAZ->getElementValue(i);
      Ops.push_back(getOrCreateVReg(Elt));
    }
    EntryBuilder.buildMerge(Reg, Ops);
  } else if (auto CV = dyn_cast<ConstantDataVector>(&C)) {
    // Return the scalar if it is a <1 x Ty> vector.
    if (CV->getNumElements() == 1)
      return translate(*CV->getElementAsConstant(0), Reg);
    std::vector<unsigned> Ops;
    for (unsigned i = 0; i < CV->getNumElements(); ++i) {
      Constant &Elt = *CV->getElementAsConstant(i);
      Ops.push_back(getOrCreateVReg(Elt));
    }
    EntryBuilder.buildMerge(Reg, Ops);
  } else if (auto CE = dyn_cast<ConstantExpr>(&C)) {
    switch(CE->getOpcode()) {
#define HANDLE_INST(NUM, OPCODE, CLASS)                         \
      case Instruction::OPCODE: return translate##OPCODE(*CE, EntryBuilder);
#include "llvm/IR/Instruction.def"
    default:
      return false;
    }
  } else if (auto CS = dyn_cast<ConstantStruct>(&C)) {
    // Return the element if it is a single element ConstantStruct.
    if (CS->getNumOperands() == 1) {
      unsigned EltReg = getOrCreateVReg(*CS->getOperand(0));
      EntryBuilder.buildCast(Reg, EltReg);
      return true;
    }
    SmallVector<unsigned, 4> Ops;
    SmallVector<uint64_t, 4> Indices;
    uint64_t Offset = 0;
    for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
      unsigned OpReg = getOrCreateVReg(*CS->getOperand(i));
      Ops.push_back(OpReg);
      Indices.push_back(Offset);
      Offset += MRI->getType(OpReg).getSizeInBits();
    }
    EntryBuilder.buildSequence(Reg, Ops, Indices);
  } else if (auto CV = dyn_cast<ConstantVector>(&C)) {
    if (CV->getNumOperands() == 1)
      return translate(*CV->getOperand(0), Reg);
    SmallVector<unsigned, 4> Ops;
    for (unsigned i = 0; i < CV->getNumOperands(); ++i) {
      Ops.push_back(getOrCreateVReg(*CV->getOperand(i)));
    }
    EntryBuilder.buildMerge(Reg, Ops);
  } else
    return false;

  return true;
}

void IRTranslator::finalizeFunction() {
  // Release the memory used by the different maps we
  // needed during the translation.
  PendingPHIs.clear();
  ValToVReg.clear();
  FrameIndices.clear();
  MachinePreds.clear();
  // MachineIRBuilder::DebugLoc can outlive the DILocation it holds. Clear it
  // to avoid accessing free’d memory (in runOnMachineFunction) and to avoid
  // destroying it twice (in ~IRTranslator() and ~LLVMContext())
  EntryBuilder = MachineIRBuilder();
  CurBuilder = MachineIRBuilder();
}

bool IRTranslator::runOnMachineFunction(MachineFunction &CurMF) {
  MF = &CurMF;
  const Function &F = *MF->getFunction();
  if (F.empty())
    return false;
  CLI = MF->getSubtarget().getCallLowering();
  CurBuilder.setMF(*MF);
  EntryBuilder.setMF(*MF);
  MRI = &MF->getRegInfo();
  DL = &F.getParent()->getDataLayout();
  TPC = &getAnalysis<TargetPassConfig>();
  ORE = llvm::make_unique<OptimizationRemarkEmitter>(&F);

  assert(PendingPHIs.empty() && "stale PHIs");

  // Release the per-function state when we return, whether we succeeded or not.
  auto FinalizeOnReturn = make_scope_exit([this]() { finalizeFunction(); });

  // Setup a separate basic-block for the arguments and constants
  MachineBasicBlock *EntryBB = MF->CreateMachineBasicBlock();
  MF->push_back(EntryBB);
  EntryBuilder.setMBB(*EntryBB);

  // Create all blocks, in IR order, to preserve the layout.
  for (const BasicBlock &BB: F) {
    auto *&MBB = BBToMBB[&BB];

    MBB = MF->CreateMachineBasicBlock(&BB);
    MF->push_back(MBB);

    if (BB.hasAddressTaken())
      MBB->setHasAddressTaken();
  }

  // Make our arguments/constants entry block fallthrough to the IR entry block.
  EntryBB->addSuccessor(&getMBB(F.front()));

  // Lower the actual args into this basic block.
  SmallVector<unsigned, 8> VRegArgs;
  for (const Argument &Arg: F.args())
    VRegArgs.push_back(getOrCreateVReg(Arg));
  if (!CLI->lowerFormalArguments(EntryBuilder, F, VRegArgs)) {
    OptimizationRemarkMissed R("gisel-irtranslator", "GISelFailure",
                               MF->getFunction()->getSubprogram(),
                               &MF->getFunction()->getEntryBlock());
    R << "unable to lower arguments: " << ore::NV("Prototype", F.getType());
    reportTranslationError(*MF, *TPC, *ORE, R);
    return false;
  }

  // And translate the function!
  for (const BasicBlock &BB: F) {
    MachineBasicBlock &MBB = getMBB(BB);
    // Set the insertion point of all the following translations to
    // the end of this basic block.
    CurBuilder.setMBB(MBB);

    for (const Instruction &Inst: BB) {
      if (translate(Inst))
        continue;

      OptimizationRemarkMissed R("gisel-irtranslator", "GISelFailure",
                                 Inst.getDebugLoc(), &BB);
      R << "unable to translate instruction: " << ore::NV("Opcode", &Inst);

      if (ORE->allowExtraAnalysis("gisel-irtranslator")) {
        std::string InstStrStorage;
        raw_string_ostream InstStr(InstStrStorage);
        InstStr << Inst;

        R << ": '" << InstStr.str() << "'";
      }

      reportTranslationError(*MF, *TPC, *ORE, R);
      return false;
    }
  }

  finishPendingPhis();

  // Merge the argument lowering and constants block with its single
  // successor, the LLVM-IR entry block.  We want the basic block to
  // be maximal.
  assert(EntryBB->succ_size() == 1 &&
         "Custom BB used for lowering should have only one successor");
  // Get the successor of the current entry block.
  MachineBasicBlock &NewEntryBB = **EntryBB->succ_begin();
  assert(NewEntryBB.pred_size() == 1 &&
         "LLVM-IR entry block has a predecessor!?");
  // Move all the instruction from the current entry block to the
  // new entry block.
  NewEntryBB.splice(NewEntryBB.begin(), EntryBB, EntryBB->begin(),
                    EntryBB->end());

  // Update the live-in information for the new entry block.
  for (const MachineBasicBlock::RegisterMaskPair &LiveIn : EntryBB->liveins())
    NewEntryBB.addLiveIn(LiveIn);
  NewEntryBB.sortUniqueLiveIns();

  // Get rid of the now empty basic block.
  EntryBB->removeSuccessor(&NewEntryBB);
  MF->remove(EntryBB);
  MF->DeleteMachineBasicBlock(EntryBB);

  assert(&MF->front() == &NewEntryBB &&
         "New entry wasn't next in the list of basic block!");

  return false;
}
