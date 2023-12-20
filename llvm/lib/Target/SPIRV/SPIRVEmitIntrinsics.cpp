//===-- SPIRVEmitIntrinsics.cpp - emit SPIRV intrinsics ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pass emits SPIRV intrinsics keeping essential high-level information for
// the translation of LLVM IR to SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

#include <queue>

// This pass performs the following transformation on LLVM IR level required
// for the following translation to SPIR-V:
// - replaces direct usages of aggregate constants with target-specific
//   intrinsics;
// - replaces aggregates-related instructions (extract/insert, ld/st, etc)
//   with a target-specific intrinsics;
// - emits intrinsics for the global variable initializers since IRTranslator
//   doesn't handle them and it's not very convenient to translate them
//   ourselves;
// - emits intrinsics to keep track of the string names assigned to the values;
// - emits intrinsics to keep track of constants (this is necessary to have an
//   LLVM IR constant after the IRTranslation is completed) for their further
//   deduplication;
// - emits intrinsics to keep track of original LLVM types of the values
//   to be able to emit proper SPIR-V types eventually.
//
// TODO: consider removing spv.track.constant in favor of spv.assign.type.

using namespace llvm;

namespace llvm {
void initializeSPIRVEmitIntrinsicsPass(PassRegistry &);
} // namespace llvm

namespace {
class SPIRVEmitIntrinsics
    : public FunctionPass,
      public InstVisitor<SPIRVEmitIntrinsics, Instruction *> {
  SPIRVTargetMachine *TM = nullptr;
  IRBuilder<> *IRB = nullptr;
  Function *F = nullptr;
  bool TrackConstants = true;
  DenseMap<Instruction *, Constant *> AggrConsts;
  DenseSet<Instruction *> AggrStores;
  void preprocessCompositeConstants();
  void preprocessUndefs();
  CallInst *buildIntrWithMD(Intrinsic::ID IntrID, ArrayRef<Type *> Types,
                            Value *Arg, Value *Arg2,
                            ArrayRef<Constant *> Imms) {
    ConstantAsMetadata *CM = ValueAsMetadata::getConstant(Arg);
    MDTuple *TyMD = MDNode::get(F->getContext(), CM);
    MetadataAsValue *VMD = MetadataAsValue::get(F->getContext(), TyMD);
    SmallVector<Value *, 4> Args;
    Args.push_back(Arg2);
    Args.push_back(VMD);
    for (auto *Imm : Imms)
      Args.push_back(Imm);
    return IRB->CreateIntrinsic(IntrID, {Types}, Args);
  }
  void replaceMemInstrUses(Instruction *Old, Instruction *New);
  void processInstrAfterVisit(Instruction *I);
  void insertAssignPtrTypeIntrs(Instruction *I);
  void insertAssignTypeIntrs(Instruction *I);
  void processGlobalValue(GlobalVariable &GV);

public:
  static char ID;
  SPIRVEmitIntrinsics() : FunctionPass(ID) {
    initializeSPIRVEmitIntrinsicsPass(*PassRegistry::getPassRegistry());
  }
  SPIRVEmitIntrinsics(SPIRVTargetMachine *_TM) : FunctionPass(ID), TM(_TM) {
    initializeSPIRVEmitIntrinsicsPass(*PassRegistry::getPassRegistry());
  }
  Instruction *visitInstruction(Instruction &I) { return &I; }
  Instruction *visitSwitchInst(SwitchInst &I);
  Instruction *visitGetElementPtrInst(GetElementPtrInst &I);
  Instruction *visitBitCastInst(BitCastInst &I);
  Instruction *visitInsertElementInst(InsertElementInst &I);
  Instruction *visitExtractElementInst(ExtractElementInst &I);
  Instruction *visitInsertValueInst(InsertValueInst &I);
  Instruction *visitExtractValueInst(ExtractValueInst &I);
  Instruction *visitLoadInst(LoadInst &I);
  Instruction *visitStoreInst(StoreInst &I);
  Instruction *visitAllocaInst(AllocaInst &I);
  Instruction *visitAtomicCmpXchgInst(AtomicCmpXchgInst &I);
  Instruction *visitUnreachableInst(UnreachableInst &I);
  bool runOnFunction(Function &F) override;
};
} // namespace

char SPIRVEmitIntrinsics::ID = 0;

INITIALIZE_PASS(SPIRVEmitIntrinsics, "emit-intrinsics", "SPIRV emit intrinsics",
                false, false)

static inline bool isAssignTypeInstr(const Instruction *I) {
  return isa<IntrinsicInst>(I) &&
         cast<IntrinsicInst>(I)->getIntrinsicID() == Intrinsic::spv_assign_type;
}

static bool isMemInstrToReplace(Instruction *I) {
  return isa<StoreInst>(I) || isa<LoadInst>(I) || isa<InsertValueInst>(I) ||
         isa<ExtractValueInst>(I) || isa<AtomicCmpXchgInst>(I);
}

static bool isAggrToReplace(const Value *V) {
  return isa<ConstantAggregate>(V) || isa<ConstantDataArray>(V) ||
         (isa<ConstantAggregateZero>(V) && !V->getType()->isVectorTy());
}

static void setInsertPointSkippingPhis(IRBuilder<> &B, Instruction *I) {
  if (isa<PHINode>(I))
    B.SetInsertPoint(I->getParent(), I->getParent()->getFirstInsertionPt());
  else
    B.SetInsertPoint(I);
}

static bool requireAssignPtrType(Instruction *I) {
  if (isa<AllocaInst>(I) || isa<GetElementPtrInst>(I))
    return true;

  return false;
}

static bool requireAssignType(Instruction *I) {
  IntrinsicInst *Intr = dyn_cast<IntrinsicInst>(I);
  if (Intr) {
    switch (Intr->getIntrinsicID()) {
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
      return false;
    }
  }
  return true;
}

void SPIRVEmitIntrinsics::replaceMemInstrUses(Instruction *Old,
                                              Instruction *New) {
  while (!Old->user_empty()) {
    auto *U = Old->user_back();
    if (isAssignTypeInstr(U)) {
      IRB->SetInsertPoint(U);
      SmallVector<Value *, 2> Args = {New, U->getOperand(1)};
      IRB->CreateIntrinsic(Intrinsic::spv_assign_type, {New->getType()}, Args);
      U->eraseFromParent();
    } else if (isMemInstrToReplace(U) || isa<ReturnInst>(U) ||
               isa<CallInst>(U)) {
      U->replaceUsesOfWith(Old, New);
    } else {
      llvm_unreachable("illegal aggregate intrinsic user");
    }
  }
  Old->eraseFromParent();
}

void SPIRVEmitIntrinsics::preprocessUndefs() {
  std::queue<Instruction *> Worklist;
  for (auto &I : instructions(F))
    Worklist.push(&I);

  while (!Worklist.empty()) {
    Instruction *I = Worklist.front();
    Worklist.pop();

    for (auto &Op : I->operands()) {
      auto *AggrUndef = dyn_cast<UndefValue>(Op);
      if (!AggrUndef || !Op->getType()->isAggregateType())
        continue;

      IRB->SetInsertPoint(I);
      auto *IntrUndef = IRB->CreateIntrinsic(Intrinsic::spv_undef, {}, {});
      Worklist.push(IntrUndef);
      I->replaceUsesOfWith(Op, IntrUndef);
      AggrConsts[IntrUndef] = AggrUndef;
    }
  }
}

void SPIRVEmitIntrinsics::preprocessCompositeConstants() {
  std::queue<Instruction *> Worklist;
  for (auto &I : instructions(F))
    Worklist.push(&I);

  while (!Worklist.empty()) {
    auto *I = Worklist.front();
    assert(I);
    bool KeepInst = false;
    for (const auto &Op : I->operands()) {
      auto BuildCompositeIntrinsic = [&KeepInst, &Worklist, &I, &Op,
                                      this](Constant *AggrC,
                                            ArrayRef<Value *> Args) {
        IRB->SetInsertPoint(I);
        auto *CCI =
            IRB->CreateIntrinsic(Intrinsic::spv_const_composite, {}, {Args});
        Worklist.push(CCI);
        I->replaceUsesOfWith(Op, CCI);
        KeepInst = true;
        AggrConsts[CCI] = AggrC;
      };

      if (auto *AggrC = dyn_cast<ConstantAggregate>(Op)) {
        SmallVector<Value *> Args(AggrC->op_begin(), AggrC->op_end());
        BuildCompositeIntrinsic(AggrC, Args);
      } else if (auto *AggrC = dyn_cast<ConstantDataArray>(Op)) {
        SmallVector<Value *> Args;
        for (unsigned i = 0; i < AggrC->getNumElements(); ++i)
          Args.push_back(AggrC->getElementAsConstant(i));
        BuildCompositeIntrinsic(AggrC, Args);
      } else if (isa<ConstantAggregateZero>(Op) &&
                 !Op->getType()->isVectorTy()) {
        auto *AggrC = cast<ConstantAggregateZero>(Op);
        SmallVector<Value *> Args(AggrC->op_begin(), AggrC->op_end());
        BuildCompositeIntrinsic(AggrC, Args);
      }
    }
    if (!KeepInst)
      Worklist.pop();
  }
}

Instruction *SPIRVEmitIntrinsics::visitSwitchInst(SwitchInst &I) {
  SmallVector<Value *, 4> Args;
  for (auto &Op : I.operands())
    if (Op.get()->getType()->isSized())
      Args.push_back(Op);
  IRB->SetInsertPoint(&I);
  IRB->CreateIntrinsic(Intrinsic::spv_switch, {I.getOperand(0)->getType()},
                       {Args});
  return &I;
}

Instruction *SPIRVEmitIntrinsics::visitGetElementPtrInst(GetElementPtrInst &I) {
  SmallVector<Type *, 2> Types = {I.getType(), I.getOperand(0)->getType()};
  SmallVector<Value *, 4> Args;
  Args.push_back(IRB->getInt1(I.isInBounds()));
  for (auto &Op : I.operands())
    Args.push_back(Op);
  auto *NewI = IRB->CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitBitCastInst(BitCastInst &I) {
  SmallVector<Type *, 2> Types = {I.getType(), I.getOperand(0)->getType()};
  SmallVector<Value *> Args(I.op_begin(), I.op_end());
  auto *NewI = IRB->CreateIntrinsic(Intrinsic::spv_bitcast, {Types}, {Args});
  std::string InstName = I.hasName() ? I.getName().str() : "";
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  NewI->setName(InstName);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitInsertElementInst(InsertElementInst &I) {
  SmallVector<Type *, 4> Types = {I.getType(), I.getOperand(0)->getType(),
                                  I.getOperand(1)->getType(),
                                  I.getOperand(2)->getType()};
  SmallVector<Value *> Args(I.op_begin(), I.op_end());
  auto *NewI = IRB->CreateIntrinsic(Intrinsic::spv_insertelt, {Types}, {Args});
  std::string InstName = I.hasName() ? I.getName().str() : "";
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  NewI->setName(InstName);
  return NewI;
}

Instruction *
SPIRVEmitIntrinsics::visitExtractElementInst(ExtractElementInst &I) {
  SmallVector<Type *, 3> Types = {I.getType(), I.getVectorOperandType(),
                                  I.getIndexOperand()->getType()};
  SmallVector<Value *, 2> Args = {I.getVectorOperand(), I.getIndexOperand()};
  auto *NewI = IRB->CreateIntrinsic(Intrinsic::spv_extractelt, {Types}, {Args});
  std::string InstName = I.hasName() ? I.getName().str() : "";
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  NewI->setName(InstName);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitInsertValueInst(InsertValueInst &I) {
  SmallVector<Type *, 1> Types = {I.getInsertedValueOperand()->getType()};
  SmallVector<Value *> Args;
  for (auto &Op : I.operands())
    if (isa<UndefValue>(Op))
      Args.push_back(UndefValue::get(IRB->getInt32Ty()));
    else
      Args.push_back(Op);
  for (auto &Op : I.indices())
    Args.push_back(IRB->getInt32(Op));
  Instruction *NewI =
      IRB->CreateIntrinsic(Intrinsic::spv_insertv, {Types}, {Args});
  replaceMemInstrUses(&I, NewI);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitExtractValueInst(ExtractValueInst &I) {
  SmallVector<Value *> Args;
  for (auto &Op : I.operands())
    Args.push_back(Op);
  for (auto &Op : I.indices())
    Args.push_back(IRB->getInt32(Op));
  auto *NewI =
      IRB->CreateIntrinsic(Intrinsic::spv_extractv, {I.getType()}, {Args});
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitLoadInst(LoadInst &I) {
  if (!I.getType()->isAggregateType())
    return &I;
  TrackConstants = false;
  const auto *TLI = TM->getSubtargetImpl()->getTargetLowering();
  MachineMemOperand::Flags Flags =
      TLI->getLoadMemOperandFlags(I, F->getParent()->getDataLayout());
  auto *NewI =
      IRB->CreateIntrinsic(Intrinsic::spv_load, {I.getOperand(0)->getType()},
                           {I.getPointerOperand(), IRB->getInt16(Flags),
                            IRB->getInt8(I.getAlign().value())});
  replaceMemInstrUses(&I, NewI);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitStoreInst(StoreInst &I) {
  if (!AggrStores.contains(&I))
    return &I;
  TrackConstants = false;
  const auto *TLI = TM->getSubtargetImpl()->getTargetLowering();
  MachineMemOperand::Flags Flags =
      TLI->getStoreMemOperandFlags(I, F->getParent()->getDataLayout());
  auto *PtrOp = I.getPointerOperand();
  auto *NewI = IRB->CreateIntrinsic(
      Intrinsic::spv_store, {I.getValueOperand()->getType(), PtrOp->getType()},
      {I.getValueOperand(), PtrOp, IRB->getInt16(Flags),
       IRB->getInt8(I.getAlign().value())});
  I.eraseFromParent();
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitAllocaInst(AllocaInst &I) {
  TrackConstants = false;
  Type *PtrTy = I.getType();
  auto *NewI = IRB->CreateIntrinsic(Intrinsic::spv_alloca, {PtrTy}, {});
  std::string InstName = I.hasName() ? I.getName().str() : "";
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  NewI->setName(InstName);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
  assert(I.getType()->isAggregateType() && "Aggregate result is expected");
  SmallVector<Value *> Args;
  for (auto &Op : I.operands())
    Args.push_back(Op);
  Args.push_back(IRB->getInt32(I.getSyncScopeID()));
  Args.push_back(IRB->getInt32(
      static_cast<uint32_t>(getMemSemantics(I.getSuccessOrdering()))));
  Args.push_back(IRB->getInt32(
      static_cast<uint32_t>(getMemSemantics(I.getFailureOrdering()))));
  auto *NewI = IRB->CreateIntrinsic(Intrinsic::spv_cmpxchg,
                                    {I.getPointerOperand()->getType()}, {Args});
  replaceMemInstrUses(&I, NewI);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitUnreachableInst(UnreachableInst &I) {
  IRB->SetInsertPoint(&I);
  IRB->CreateIntrinsic(Intrinsic::spv_unreachable, {}, {});
  return &I;
}

void SPIRVEmitIntrinsics::processGlobalValue(GlobalVariable &GV) {
  // Skip special artifical variable llvm.global.annotations.
  if (GV.getName() == "llvm.global.annotations")
    return;
  if (GV.hasInitializer() && !isa<UndefValue>(GV.getInitializer())) {
    Constant *Init = GV.getInitializer();
    Type *Ty = isAggrToReplace(Init) ? IRB->getInt32Ty() : Init->getType();
    Constant *Const = isAggrToReplace(Init) ? IRB->getInt32(1) : Init;
    auto *InitInst = IRB->CreateIntrinsic(Intrinsic::spv_init_global,
                                          {GV.getType(), Ty}, {&GV, Const});
    InitInst->setArgOperand(1, Init);
  }
  if ((!GV.hasInitializer() || isa<UndefValue>(GV.getInitializer())) &&
      GV.getNumUses() == 0)
    IRB->CreateIntrinsic(Intrinsic::spv_unref_global, GV.getType(), &GV);
}

void SPIRVEmitIntrinsics::insertAssignPtrTypeIntrs(Instruction *I) {
  if (I->getType()->isVoidTy() || !requireAssignPtrType(I))
    return;

  setInsertPointSkippingPhis(*IRB, I->getNextNode());

  Constant *EltTyConst;
  unsigned AddressSpace = 0;
  if (auto *AI = dyn_cast<AllocaInst>(I)) {
    EltTyConst = UndefValue::get(AI->getAllocatedType());
    AddressSpace = AI->getAddressSpace();
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    EltTyConst = UndefValue::get(GEP->getResultElementType());
    AddressSpace = GEP->getPointerAddressSpace();
  } else {
    llvm_unreachable("Unexpected instruction!");
  }

  buildIntrWithMD(Intrinsic::spv_assign_ptr_type, {I->getType()}, EltTyConst, I,
                  {IRB->getInt32(AddressSpace)});
}

void SPIRVEmitIntrinsics::insertAssignTypeIntrs(Instruction *I) {
  Type *Ty = I->getType();
  if (!Ty->isVoidTy() && requireAssignType(I) && !requireAssignPtrType(I)) {
    setInsertPointSkippingPhis(*IRB, I->getNextNode());
    Type *TypeToAssign = Ty;
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      if (II->getIntrinsicID() == Intrinsic::spv_const_composite ||
          II->getIntrinsicID() == Intrinsic::spv_undef) {
        auto t = AggrConsts.find(II);
        assert(t != AggrConsts.end());
        TypeToAssign = t->second->getType();
      }
    }
    Constant *Const = UndefValue::get(TypeToAssign);
    buildIntrWithMD(Intrinsic::spv_assign_type, {Ty}, Const, I, {});
  }
  for (const auto &Op : I->operands()) {
    if (isa<ConstantPointerNull>(Op) || isa<UndefValue>(Op) ||
        // Check GetElementPtrConstantExpr case.
        (isa<ConstantExpr>(Op) && isa<GEPOperator>(Op))) {
      setInsertPointSkippingPhis(*IRB, I);
      if (isa<UndefValue>(Op) && Op->getType()->isAggregateType())
        buildIntrWithMD(Intrinsic::spv_assign_type, {IRB->getInt32Ty()}, Op,
                        UndefValue::get(IRB->getInt32Ty()), {});
      else
        buildIntrWithMD(Intrinsic::spv_assign_type, {Op->getType()}, Op, Op,
                        {});
    }
  }
}

void SPIRVEmitIntrinsics::processInstrAfterVisit(Instruction *I) {
  auto *II = dyn_cast<IntrinsicInst>(I);
  if (II && II->getIntrinsicID() == Intrinsic::spv_const_composite &&
      TrackConstants) {
    IRB->SetInsertPoint(I->getNextNode());
    Type *Ty = IRB->getInt32Ty();
    auto t = AggrConsts.find(I);
    assert(t != AggrConsts.end());
    auto *NewOp = buildIntrWithMD(Intrinsic::spv_track_constant, {Ty, Ty},
                                  t->second, I, {});
    I->replaceAllUsesWith(NewOp);
    NewOp->setArgOperand(0, I);
  }
  for (const auto &Op : I->operands()) {
    if ((isa<ConstantAggregateZero>(Op) && Op->getType()->isVectorTy()) ||
        isa<PHINode>(I) || isa<SwitchInst>(I))
      TrackConstants = false;
    if ((isa<ConstantData>(Op) || isa<ConstantExpr>(Op)) && TrackConstants) {
      unsigned OpNo = Op.getOperandNo();
      if (II && ((II->getIntrinsicID() == Intrinsic::spv_gep && OpNo == 0) ||
                 (II->paramHasAttr(OpNo, Attribute::ImmArg))))
        continue;
      IRB->SetInsertPoint(I);
      auto *NewOp = buildIntrWithMD(Intrinsic::spv_track_constant,
                                    {Op->getType(), Op->getType()}, Op, Op, {});
      I->setOperand(OpNo, NewOp);
    }
  }
  if (I->hasName()) {
    setInsertPointSkippingPhis(*IRB, I->getNextNode());
    std::vector<Value *> Args = {I};
    addStringImm(I->getName(), *IRB, Args);
    IRB->CreateIntrinsic(Intrinsic::spv_assign_name, {I->getType()}, Args);
  }
}

bool SPIRVEmitIntrinsics::runOnFunction(Function &Func) {
  if (Func.isDeclaration())
    return false;
  F = &Func;
  IRB = new IRBuilder<>(Func.getContext());
  AggrConsts.clear();
  AggrStores.clear();

  // StoreInst's operand type can be changed during the next transformations,
  // so we need to store it in the set. Also store already transformed types.
  for (auto &I : instructions(Func)) {
    StoreInst *SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;
    Type *ElTy = SI->getValueOperand()->getType();
    if (ElTy->isAggregateType() || ElTy->isVectorTy())
      AggrStores.insert(&I);
  }

  IRB->SetInsertPoint(&Func.getEntryBlock(), Func.getEntryBlock().begin());
  for (auto &GV : Func.getParent()->globals())
    processGlobalValue(GV);

  preprocessUndefs();
  preprocessCompositeConstants();
  SmallVector<Instruction *> Worklist;
  for (auto &I : instructions(Func))
    Worklist.push_back(&I);

  for (auto &I : Worklist) {
    insertAssignPtrTypeIntrs(I);
    insertAssignTypeIntrs(I);
  }

  for (auto *I : Worklist) {
    TrackConstants = true;
    if (!I->getType()->isVoidTy() || isa<StoreInst>(I))
      IRB->SetInsertPoint(I->getNextNode());
    I = visit(*I);
    processInstrAfterVisit(I);
  }
  return true;
}

FunctionPass *llvm::createSPIRVEmitIntrinsicsPass(SPIRVTargetMachine *TM) {
  return new SPIRVEmitIntrinsics(TM);
}
