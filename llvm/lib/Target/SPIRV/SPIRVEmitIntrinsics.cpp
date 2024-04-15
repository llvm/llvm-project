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
#include "SPIRVBuiltins.h"
#include "SPIRVMetadata.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "SPIRVUtils.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/TypedPointerType.h"

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
    : public ModulePass,
      public InstVisitor<SPIRVEmitIntrinsics, Instruction *> {
  SPIRVTargetMachine *TM = nullptr;
  SPIRVGlobalRegistry *GR = nullptr;
  Function *F = nullptr;
  bool TrackConstants = true;
  DenseMap<Instruction *, Constant *> AggrConsts;
  DenseMap<Instruction *, Type *> AggrConstTypes;
  DenseSet<Instruction *> AggrStores;

  // a registry of created Intrinsic::spv_assign_ptr_type instructions
  DenseMap<Value *, CallInst *> AssignPtrTypeInstr;

  // deduce element type of untyped pointers
  Type *deduceElementType(Value *I);
  Type *deduceElementTypeHelper(Value *I);
  Type *deduceElementTypeHelper(Value *I, std::unordered_set<Value *> &Visited);
  Type *deduceElementTypeByValueDeep(Type *ValueTy, Value *Operand,
                                     std::unordered_set<Value *> &Visited);
  Type *deduceElementTypeByUsersDeep(Value *Op,
                                     std::unordered_set<Value *> &Visited);

  // deduce nested types of composites
  Type *deduceNestedTypeHelper(User *U);
  Type *deduceNestedTypeHelper(User *U, Type *Ty,
                               std::unordered_set<Value *> &Visited);

  // deduce Types of operands of the Instruction if possible
  void deduceOperandElementType(Instruction *I);

  void preprocessCompositeConstants(IRBuilder<> &B);
  void preprocessUndefs(IRBuilder<> &B);

  CallInst *buildIntrWithMD(Intrinsic::ID IntrID, ArrayRef<Type *> Types,
                            Value *Arg, Value *Arg2, ArrayRef<Constant *> Imms,
                            IRBuilder<> &B) {
    ConstantAsMetadata *CM = ValueAsMetadata::getConstant(Arg);
    MDTuple *TyMD = MDNode::get(F->getContext(), CM);
    MetadataAsValue *VMD = MetadataAsValue::get(F->getContext(), TyMD);
    SmallVector<Value *, 4> Args;
    Args.push_back(Arg2);
    Args.push_back(VMD);
    for (auto *Imm : Imms)
      Args.push_back(Imm);
    return B.CreateIntrinsic(IntrID, {Types}, Args);
  }

  void replaceMemInstrUses(Instruction *Old, Instruction *New, IRBuilder<> &B);
  void processInstrAfterVisit(Instruction *I, IRBuilder<> &B);
  void insertAssignPtrTypeIntrs(Instruction *I, IRBuilder<> &B);
  void insertAssignTypeIntrs(Instruction *I, IRBuilder<> &B);
  void insertAssignTypeInstrForTargetExtTypes(TargetExtType *AssignedType,
                                              Value *V, IRBuilder<> &B);
  void replacePointerOperandWithPtrCast(Instruction *I, Value *Pointer,
                                        Type *ExpectedElementType,
                                        unsigned OperandToReplace,
                                        IRBuilder<> &B);
  void insertPtrCastOrAssignTypeInstr(Instruction *I, IRBuilder<> &B);
  void processGlobalValue(GlobalVariable &GV, IRBuilder<> &B);
  void processParamTypes(Function *F, IRBuilder<> &B);
  Type *deduceFunParamElementType(Function *F, unsigned OpIdx);
  Type *deduceFunParamElementType(Function *F, unsigned OpIdx,
                                  std::unordered_set<Function *> &FVisited);

public:
  static char ID;
  SPIRVEmitIntrinsics() : ModulePass(ID) {
    initializeSPIRVEmitIntrinsicsPass(*PassRegistry::getPassRegistry());
  }
  SPIRVEmitIntrinsics(SPIRVTargetMachine *_TM) : ModulePass(ID), TM(_TM) {
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

  StringRef getPassName() const override { return "SPIRV emit intrinsics"; }

  bool runOnModule(Module &M) override;
  bool runOnFunction(Function &F);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
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

static inline void reportFatalOnTokenType(const Instruction *I) {
  if (I->getType()->isTokenTy())
    report_fatal_error("A token is encountered but SPIR-V without extensions "
                       "does not support token type",
                       false);
}

// Set element pointer type to the given value of ValueTy and tries to
// specify this type further (recursively) by Operand value, if needed.
Type *SPIRVEmitIntrinsics::deduceElementTypeByValueDeep(
    Type *ValueTy, Value *Operand, std::unordered_set<Value *> &Visited) {
  Type *Ty = ValueTy;
  if (Operand) {
    if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
      if (Type *NestedTy = deduceElementTypeHelper(Operand, Visited))
        Ty = TypedPointerType::get(NestedTy, PtrTy->getAddressSpace());
    } else {
      Ty = deduceNestedTypeHelper(dyn_cast<User>(Operand), Ty, Visited);
    }
  }
  return Ty;
}

// Traverse User instructions to deduce an element pointer type of the operand.
Type *SPIRVEmitIntrinsics::deduceElementTypeByUsersDeep(
    Value *Op, std::unordered_set<Value *> &Visited) {
  if (!Op || !isPointerTy(Op->getType()))
    return nullptr;

  if (auto PType = dyn_cast<TypedPointerType>(Op->getType()))
    return PType->getElementType();

  // maybe we already know operand's element type
  if (Type *KnownTy = GR->findDeducedElementType(Op))
    return KnownTy;

  for (User *OpU : Op->users()) {
    if (Instruction *Inst = dyn_cast<Instruction>(OpU)) {
      if (Type *Ty = deduceElementTypeHelper(Inst, Visited))
        return Ty;
    }
  }
  return nullptr;
}

// Deduce and return a successfully deduced Type of the Instruction,
// or nullptr otherwise.
Type *SPIRVEmitIntrinsics::deduceElementTypeHelper(Value *I) {
  std::unordered_set<Value *> Visited;
  return deduceElementTypeHelper(I, Visited);
}

Type *SPIRVEmitIntrinsics::deduceElementTypeHelper(
    Value *I, std::unordered_set<Value *> &Visited) {
  // allow to pass nullptr as an argument
  if (!I)
    return nullptr;

  // maybe already known
  if (Type *KnownTy = GR->findDeducedElementType(I))
    return KnownTy;

  // maybe a cycle
  if (Visited.find(I) != Visited.end())
    return nullptr;
  Visited.insert(I);

  // fallback value in case when we fail to deduce a type
  Type *Ty = nullptr;
  // look for known basic patterns of type inference
  if (auto *Ref = dyn_cast<AllocaInst>(I)) {
    Ty = Ref->getAllocatedType();
  } else if (auto *Ref = dyn_cast<GetElementPtrInst>(I)) {
    Ty = Ref->getResultElementType();
  } else if (auto *Ref = dyn_cast<GlobalValue>(I)) {
    Ty = deduceElementTypeByValueDeep(
        Ref->getValueType(),
        Ref->getNumOperands() > 0 ? Ref->getOperand(0) : nullptr, Visited);
  } else if (auto *Ref = dyn_cast<AddrSpaceCastInst>(I)) {
    Ty = deduceElementTypeHelper(Ref->getPointerOperand(), Visited);
  } else if (auto *Ref = dyn_cast<BitCastInst>(I)) {
    if (Type *Src = Ref->getSrcTy(), *Dest = Ref->getDestTy();
        isPointerTy(Src) && isPointerTy(Dest))
      Ty = deduceElementTypeHelper(Ref->getOperand(0), Visited);
  } else if (auto *Ref = dyn_cast<AtomicCmpXchgInst>(I)) {
    Value *Op = Ref->getNewValOperand();
    Ty = deduceElementTypeByValueDeep(Op->getType(), Op, Visited);
  } else if (auto *Ref = dyn_cast<AtomicRMWInst>(I)) {
    Value *Op = Ref->getValOperand();
    Ty = deduceElementTypeByValueDeep(Op->getType(), Op, Visited);
  } else if (auto *Ref = dyn_cast<PHINode>(I)) {
    for (unsigned i = 0; i < Ref->getNumIncomingValues(); i++) {
      Ty = deduceElementTypeByUsersDeep(Ref->getIncomingValue(i), Visited);
      if (Ty)
        break;
    }
  } else if (auto *Ref = dyn_cast<SelectInst>(I)) {
    for (Value *Op : {Ref->getTrueValue(), Ref->getFalseValue()}) {
      Ty = deduceElementTypeByUsersDeep(Op, Visited);
      if (Ty)
        break;
    }
  }

  // remember the found relationship
  if (Ty) {
    // specify nested types if needed, otherwise return unchanged
    GR->addDeducedElementType(I, Ty);
  }

  return Ty;
}

// Re-create a type of the value if it has untyped pointer fields, also nested.
// Return the original value type if no corrections of untyped pointer
// information is found or needed.
Type *SPIRVEmitIntrinsics::deduceNestedTypeHelper(User *U) {
  std::unordered_set<Value *> Visited;
  return deduceNestedTypeHelper(U, U->getType(), Visited);
}

Type *SPIRVEmitIntrinsics::deduceNestedTypeHelper(
    User *U, Type *OrigTy, std::unordered_set<Value *> &Visited) {
  if (!U)
    return OrigTy;

  // maybe already known
  if (Type *KnownTy = GR->findDeducedCompositeType(U))
    return KnownTy;

  // maybe a cycle
  if (Visited.find(U) != Visited.end())
    return OrigTy;
  Visited.insert(U);

  if (dyn_cast<StructType>(OrigTy)) {
    SmallVector<Type *> Tys;
    bool Change = false;
    for (unsigned i = 0; i < U->getNumOperands(); ++i) {
      Value *Op = U->getOperand(i);
      Type *OpTy = Op->getType();
      Type *Ty = OpTy;
      if (Op) {
        if (auto *PtrTy = dyn_cast<PointerType>(OpTy)) {
          if (Type *NestedTy = deduceElementTypeHelper(Op, Visited))
            Ty = TypedPointerType::get(NestedTy, PtrTy->getAddressSpace());
        } else {
          Ty = deduceNestedTypeHelper(dyn_cast<User>(Op), OpTy, Visited);
        }
      }
      Tys.push_back(Ty);
      Change |= Ty != OpTy;
    }
    if (Change) {
      Type *NewTy = StructType::create(Tys);
      GR->addDeducedCompositeType(U, NewTy);
      return NewTy;
    }
  } else if (auto *ArrTy = dyn_cast<ArrayType>(OrigTy)) {
    if (Value *Op = U->getNumOperands() > 0 ? U->getOperand(0) : nullptr) {
      Type *OpTy = ArrTy->getElementType();
      Type *Ty = OpTy;
      if (auto *PtrTy = dyn_cast<PointerType>(OpTy)) {
        if (Type *NestedTy = deduceElementTypeHelper(Op, Visited))
          Ty = TypedPointerType::get(NestedTy, PtrTy->getAddressSpace());
      } else {
        Ty = deduceNestedTypeHelper(dyn_cast<User>(Op), OpTy, Visited);
      }
      if (Ty != OpTy) {
        Type *NewTy = ArrayType::get(Ty, ArrTy->getNumElements());
        GR->addDeducedCompositeType(U, NewTy);
        return NewTy;
      }
    }
  } else if (auto *VecTy = dyn_cast<VectorType>(OrigTy)) {
    if (Value *Op = U->getNumOperands() > 0 ? U->getOperand(0) : nullptr) {
      Type *OpTy = VecTy->getElementType();
      Type *Ty = OpTy;
      if (auto *PtrTy = dyn_cast<PointerType>(OpTy)) {
        if (Type *NestedTy = deduceElementTypeHelper(Op, Visited))
          Ty = TypedPointerType::get(NestedTy, PtrTy->getAddressSpace());
      } else {
        Ty = deduceNestedTypeHelper(dyn_cast<User>(Op), OpTy, Visited);
      }
      if (Ty != OpTy) {
        Type *NewTy = VectorType::get(Ty, VecTy->getElementCount());
        GR->addDeducedCompositeType(U, NewTy);
        return NewTy;
      }
    }
  }

  return OrigTy;
}

Type *SPIRVEmitIntrinsics::deduceElementType(Value *I) {
  if (Type *Ty = deduceElementTypeHelper(I))
    return Ty;
  return IntegerType::getInt8Ty(I->getContext());
}

// If the Instruction has Pointer operands with unresolved types, this function
// tries to deduce them. If the Instruction has Pointer operands with known
// types which differ from expected, this function tries to insert a bitcast to
// resolve the issue.
void SPIRVEmitIntrinsics::deduceOperandElementType(Instruction *I) {
  SmallVector<std::pair<Value *, unsigned>> Ops;
  Type *KnownElemTy = nullptr;
  // look for known basic patterns of type inference
  if (auto *Ref = dyn_cast<PHINode>(I)) {
    if (!isPointerTy(I->getType()) ||
        !(KnownElemTy = GR->findDeducedElementType(I)))
      return;
    for (unsigned i = 0; i < Ref->getNumIncomingValues(); i++) {
      Value *Op = Ref->getIncomingValue(i);
      if (isPointerTy(Op->getType()))
        Ops.push_back(std::make_pair(Op, i));
    }
  } else if (auto *Ref = dyn_cast<SelectInst>(I)) {
    if (!isPointerTy(I->getType()) ||
        !(KnownElemTy = GR->findDeducedElementType(I)))
      return;
    for (unsigned i = 0; i < Ref->getNumOperands(); i++) {
      Value *Op = Ref->getOperand(i);
      if (isPointerTy(Op->getType()))
        Ops.push_back(std::make_pair(Op, i));
    }
  } else if (auto *Ref = dyn_cast<ReturnInst>(I)) {
    Type *RetTy = F->getReturnType();
    if (!isPointerTy(RetTy))
      return;
    Value *Op = Ref->getReturnValue();
    if (!Op)
      return;
    if (!(KnownElemTy = GR->findDeducedElementType(F))) {
      if (Type *OpElemTy = GR->findDeducedElementType(Op)) {
        GR->addDeducedElementType(F, OpElemTy);
        TypedPointerType *DerivedTy =
            TypedPointerType::get(OpElemTy, getPointerAddressSpace(RetTy));
        GR->addReturnType(F, DerivedTy);
      }
      return;
    }
    Ops.push_back(std::make_pair(Op, 0));
  } else if (auto *Ref = dyn_cast<ICmpInst>(I)) {
    if (!isPointerTy(Ref->getOperand(0)->getType()))
      return;
    Value *Op0 = Ref->getOperand(0);
    Value *Op1 = Ref->getOperand(1);
    Type *ElemTy0 = GR->findDeducedElementType(Op0);
    Type *ElemTy1 = GR->findDeducedElementType(Op1);
    if (ElemTy0) {
      KnownElemTy = ElemTy0;
      Ops.push_back(std::make_pair(Op1, 1));
    } else if (ElemTy1) {
      KnownElemTy = ElemTy1;
      Ops.push_back(std::make_pair(Op0, 0));
    }
  }

  // There is no enough info to deduce types or all is valid.
  if (!KnownElemTy || Ops.size() == 0)
    return;

  LLVMContext &Ctx = F->getContext();
  IRBuilder<> B(Ctx);
  for (auto &OpIt : Ops) {
    Value *Op = OpIt.first;
    if (Op->use_empty())
      continue;
    Type *Ty = GR->findDeducedElementType(Op);
    if (Ty == KnownElemTy)
      continue;
    if (Instruction *User = dyn_cast<Instruction>(Op->use_begin()->get()))
      setInsertPointSkippingPhis(B, User->getNextNode());
    else
      B.SetInsertPoint(I);
    Value *OpTyVal = Constant::getNullValue(KnownElemTy);
    Type *OpTy = Op->getType();
    if (!Ty) {
      GR->addDeducedElementType(Op, KnownElemTy);
      // check if there is existing Intrinsic::spv_assign_ptr_type instruction
      auto It = AssignPtrTypeInstr.find(Op);
      if (It == AssignPtrTypeInstr.end()) {
        CallInst *CI =
            buildIntrWithMD(Intrinsic::spv_assign_ptr_type, {OpTy}, OpTyVal, Op,
                            {B.getInt32(getPointerAddressSpace(OpTy))}, B);
        AssignPtrTypeInstr[Op] = CI;
      } else {
        It->second->setArgOperand(
            1,
            MetadataAsValue::get(
                Ctx, MDNode::get(Ctx, ValueAsMetadata::getConstant(OpTyVal))));
      }
    } else {
      SmallVector<Type *, 2> Types = {OpTy, OpTy};
      MetadataAsValue *VMD = MetadataAsValue::get(
          Ctx, MDNode::get(Ctx, ValueAsMetadata::getConstant(OpTyVal)));
      SmallVector<Value *, 2> Args = {Op, VMD,
                                      B.getInt32(getPointerAddressSpace(OpTy))};
      CallInst *PtrCastI =
          B.CreateIntrinsic(Intrinsic::spv_ptrcast, {Types}, Args);
      I->setOperand(OpIt.second, PtrCastI);
    }
  }
}

void SPIRVEmitIntrinsics::replaceMemInstrUses(Instruction *Old,
                                              Instruction *New,
                                              IRBuilder<> &B) {
  while (!Old->user_empty()) {
    auto *U = Old->user_back();
    if (isAssignTypeInstr(U)) {
      B.SetInsertPoint(U);
      SmallVector<Value *, 2> Args = {New, U->getOperand(1)};
      B.CreateIntrinsic(Intrinsic::spv_assign_type, {New->getType()}, Args);
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

void SPIRVEmitIntrinsics::preprocessUndefs(IRBuilder<> &B) {
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

      B.SetInsertPoint(I);
      auto *IntrUndef = B.CreateIntrinsic(Intrinsic::spv_undef, {}, {});
      Worklist.push(IntrUndef);
      I->replaceUsesOfWith(Op, IntrUndef);
      AggrConsts[IntrUndef] = AggrUndef;
      AggrConstTypes[IntrUndef] = AggrUndef->getType();
    }
  }
}

void SPIRVEmitIntrinsics::preprocessCompositeConstants(IRBuilder<> &B) {
  std::queue<Instruction *> Worklist;
  for (auto &I : instructions(F))
    Worklist.push(&I);

  while (!Worklist.empty()) {
    auto *I = Worklist.front();
    assert(I);
    bool KeepInst = false;
    for (const auto &Op : I->operands()) {
      auto BuildCompositeIntrinsic =
          [](Constant *AggrC, ArrayRef<Value *> Args, Value *Op, Instruction *I,
             IRBuilder<> &B, std::queue<Instruction *> &Worklist,
             bool &KeepInst, SPIRVEmitIntrinsics &SEI) {
            B.SetInsertPoint(I);
            auto *CCI =
                B.CreateIntrinsic(Intrinsic::spv_const_composite, {}, {Args});
            Worklist.push(CCI);
            I->replaceUsesOfWith(Op, CCI);
            KeepInst = true;
            SEI.AggrConsts[CCI] = AggrC;
            SEI.AggrConstTypes[CCI] = SEI.deduceNestedTypeHelper(AggrC);
          };

      if (auto *AggrC = dyn_cast<ConstantAggregate>(Op)) {
        SmallVector<Value *> Args(AggrC->op_begin(), AggrC->op_end());
        BuildCompositeIntrinsic(AggrC, Args, Op, I, B, Worklist, KeepInst,
                                *this);
      } else if (auto *AggrC = dyn_cast<ConstantDataArray>(Op)) {
        SmallVector<Value *> Args;
        for (unsigned i = 0; i < AggrC->getNumElements(); ++i)
          Args.push_back(AggrC->getElementAsConstant(i));
        BuildCompositeIntrinsic(AggrC, Args, Op, I, B, Worklist, KeepInst,
                                *this);
      } else if (isa<ConstantAggregateZero>(Op) &&
                 !Op->getType()->isVectorTy()) {
        auto *AggrC = cast<ConstantAggregateZero>(Op);
        SmallVector<Value *> Args(AggrC->op_begin(), AggrC->op_end());
        BuildCompositeIntrinsic(AggrC, Args, Op, I, B, Worklist, KeepInst,
                                *this);
      }
    }
    if (!KeepInst)
      Worklist.pop();
  }
}

Instruction *SPIRVEmitIntrinsics::visitSwitchInst(SwitchInst &I) {
  BasicBlock *ParentBB = I.getParent();
  IRBuilder<> B(ParentBB);
  B.SetInsertPoint(&I);
  SmallVector<Value *, 4> Args;
  SmallVector<BasicBlock *> BBCases;
  for (auto &Op : I.operands()) {
    if (Op.get()->getType()->isSized()) {
      Args.push_back(Op);
    } else if (BasicBlock *BB = dyn_cast<BasicBlock>(Op.get())) {
      BBCases.push_back(BB);
      Args.push_back(BlockAddress::get(BB->getParent(), BB));
    } else {
      report_fatal_error("Unexpected switch operand");
    }
  }
  CallInst *NewI = B.CreateIntrinsic(Intrinsic::spv_switch,
                                     {I.getOperand(0)->getType()}, {Args});
  // remove switch to avoid its unneeded and undesirable unwrap into branches
  // and conditions
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  // insert artificial and temporary instruction to preserve valid CFG,
  // it will be removed after IR translation pass
  B.SetInsertPoint(ParentBB);
  IndirectBrInst *BrI = B.CreateIndirectBr(
      Constant::getNullValue(PointerType::getUnqual(ParentBB->getContext())),
      BBCases.size());
  for (BasicBlock *BBCase : BBCases)
    BrI->addDestination(BBCase);
  return BrI;
}

Instruction *SPIRVEmitIntrinsics::visitGetElementPtrInst(GetElementPtrInst &I) {
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  SmallVector<Type *, 2> Types = {I.getType(), I.getOperand(0)->getType()};
  SmallVector<Value *, 4> Args;
  Args.push_back(B.getInt1(I.isInBounds()));
  for (auto &Op : I.operands())
    Args.push_back(Op);
  auto *NewI = B.CreateIntrinsic(Intrinsic::spv_gep, {Types}, {Args});
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitBitCastInst(BitCastInst &I) {
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  Value *Source = I.getOperand(0);

  // SPIR-V, contrary to LLVM 17+ IR, supports bitcasts between pointers of
  // varying element types. In case of IR coming from older versions of LLVM
  // such bitcasts do not provide sufficient information, should be just skipped
  // here, and handled in insertPtrCastOrAssignTypeInstr.
  if (isPointerTy(I.getType())) {
    I.replaceAllUsesWith(Source);
    I.eraseFromParent();
    return nullptr;
  }

  SmallVector<Type *, 2> Types = {I.getType(), Source->getType()};
  SmallVector<Value *> Args(I.op_begin(), I.op_end());
  auto *NewI = B.CreateIntrinsic(Intrinsic::spv_bitcast, {Types}, {Args});
  std::string InstName = I.hasName() ? I.getName().str() : "";
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  NewI->setName(InstName);
  return NewI;
}

void SPIRVEmitIntrinsics::insertAssignTypeInstrForTargetExtTypes(
    TargetExtType *AssignedType, Value *V, IRBuilder<> &B) {
  // Do not emit spv_assign_type if the V is of the AssignedType already.
  if (V->getType() == AssignedType)
    return;

  // Do not emit spv_assign_type if there is one already targetting V. If the
  // found spv_assign_type assigns a type different than AssignedType, report an
  // error. Builtin types cannot be redeclared or casted.
  for (auto User : V->users()) {
    auto *II = dyn_cast<IntrinsicInst>(User);
    if (!II || II->getIntrinsicID() != Intrinsic::spv_assign_type)
      continue;

    MetadataAsValue *VMD = cast<MetadataAsValue>(II->getOperand(1));
    Type *BuiltinType =
        dyn_cast<ConstantAsMetadata>(VMD->getMetadata())->getType();
    if (BuiltinType != AssignedType)
      report_fatal_error("Type mismatch " + BuiltinType->getTargetExtName() +
                             "/" + AssignedType->getTargetExtName() +
                             " for value " + V->getName(),
                         false);
    return;
  }

  Constant *Const = UndefValue::get(AssignedType);
  buildIntrWithMD(Intrinsic::spv_assign_type, {V->getType()}, Const, V, {}, B);
}

void SPIRVEmitIntrinsics::replacePointerOperandWithPtrCast(
    Instruction *I, Value *Pointer, Type *ExpectedElementType,
    unsigned OperandToReplace, IRBuilder<> &B) {
  // If Pointer is the result of nop BitCastInst (ptr -> ptr), use the source
  // pointer instead. The BitCastInst should be later removed when visited.
  while (BitCastInst *BC = dyn_cast<BitCastInst>(Pointer))
    Pointer = BC->getOperand(0);

  // Do not emit spv_ptrcast if Pointer's element type is ExpectedElementType
  Type *PointerElemTy = deduceElementTypeHelper(Pointer);
  if (PointerElemTy == ExpectedElementType)
    return;

  setInsertPointSkippingPhis(B, I);
  Constant *ExpectedElementTypeConst =
      Constant::getNullValue(ExpectedElementType);
  ConstantAsMetadata *CM =
      ValueAsMetadata::getConstant(ExpectedElementTypeConst);
  MDTuple *TyMD = MDNode::get(F->getContext(), CM);
  MetadataAsValue *VMD = MetadataAsValue::get(F->getContext(), TyMD);
  unsigned AddressSpace = getPointerAddressSpace(Pointer->getType());
  bool FirstPtrCastOrAssignPtrType = true;

  // Do not emit new spv_ptrcast if equivalent one already exists or when
  // spv_assign_ptr_type already targets this pointer with the same element
  // type.
  for (auto User : Pointer->users()) {
    auto *II = dyn_cast<IntrinsicInst>(User);
    if (!II ||
        (II->getIntrinsicID() != Intrinsic::spv_assign_ptr_type &&
         II->getIntrinsicID() != Intrinsic::spv_ptrcast) ||
        II->getOperand(0) != Pointer)
      continue;

    // There is some spv_ptrcast/spv_assign_ptr_type already targeting this
    // pointer.
    FirstPtrCastOrAssignPtrType = false;
    if (II->getOperand(1) != VMD ||
        dyn_cast<ConstantInt>(II->getOperand(2))->getSExtValue() !=
            AddressSpace)
      continue;

    // The spv_ptrcast/spv_assign_ptr_type targeting this pointer is of the same
    // element type and address space.
    if (II->getIntrinsicID() != Intrinsic::spv_ptrcast)
      return;

    // This must be a spv_ptrcast, do not emit new if this one has the same BB
    // as I. Otherwise, search for other spv_ptrcast/spv_assign_ptr_type.
    if (II->getParent() != I->getParent())
      continue;

    I->setOperand(OperandToReplace, II);
    return;
  }

  // // Do not emit spv_ptrcast if it would cast to the default pointer element
  // // type (i8) of the same address space.
  // if (ExpectedElementType->isIntegerTy(8))
  //   return;

  // If this would be the first spv_ptrcast, do not emit spv_ptrcast and emit
  // spv_assign_ptr_type instead.
  if (FirstPtrCastOrAssignPtrType &&
      (isa<Instruction>(Pointer) || isa<Argument>(Pointer))) {
    CallInst *CI = buildIntrWithMD(
        Intrinsic::spv_assign_ptr_type, {Pointer->getType()},
        ExpectedElementTypeConst, Pointer, {B.getInt32(AddressSpace)}, B);
    GR->addDeducedElementType(CI, ExpectedElementType);
    GR->addDeducedElementType(Pointer, ExpectedElementType);
    AssignPtrTypeInstr[Pointer] = CI;
    return;
  }

  // Emit spv_ptrcast
  SmallVector<Type *, 2> Types = {Pointer->getType(), Pointer->getType()};
  SmallVector<Value *, 2> Args = {Pointer, VMD, B.getInt32(AddressSpace)};
  auto *PtrCastI = B.CreateIntrinsic(Intrinsic::spv_ptrcast, {Types}, Args);
  I->setOperand(OperandToReplace, PtrCastI);
}

void SPIRVEmitIntrinsics::insertPtrCastOrAssignTypeInstr(Instruction *I,
                                                         IRBuilder<> &B) {
  // Handle basic instructions:
  StoreInst *SI = dyn_cast<StoreInst>(I);
  if (SI && F->getCallingConv() == CallingConv::SPIR_KERNEL &&
      isPointerTy(SI->getValueOperand()->getType()) &&
      isa<Argument>(SI->getValueOperand())) {
    return replacePointerOperandWithPtrCast(
        I, SI->getValueOperand(), IntegerType::getInt8Ty(F->getContext()), 0,
        B);
  } else if (SI) {
    return replacePointerOperandWithPtrCast(
        I, SI->getPointerOperand(), SI->getValueOperand()->getType(), 1, B);
  } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    return replacePointerOperandWithPtrCast(I, LI->getPointerOperand(),
                                            LI->getType(), 0, B);
  } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
    return replacePointerOperandWithPtrCast(I, GEPI->getPointerOperand(),
                                            GEPI->getSourceElementType(), 0, B);
  }

  // Handle calls to builtins (non-intrinsics):
  CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI || CI->isIndirectCall() || CI->isInlineAsm() ||
      !CI->getCalledFunction() || CI->getCalledFunction()->isIntrinsic())
    return;

  // collect information about formal parameter types
  Function *CalledF = CI->getCalledFunction();
  SmallVector<Type *, 4> CalledArgTys;
  bool HaveTypes = false;
  for (unsigned OpIdx = 0; OpIdx < CalledF->arg_size(); ++OpIdx) {
    Argument *CalledArg = CalledF->getArg(OpIdx);
    Type *ArgType = CalledArg->getType();
    if (!isPointerTy(ArgType)) {
      CalledArgTys.push_back(nullptr);
    } else if (isTypedPointerTy(ArgType)) {
      CalledArgTys.push_back(cast<TypedPointerType>(ArgType)->getElementType());
      HaveTypes = true;
    } else {
      Type *ElemTy = GR->findDeducedElementType(CalledArg);
      if (!ElemTy && hasPointeeTypeAttr(CalledArg))
        ElemTy = getPointeeTypeByAttr(CalledArg);
      if (!ElemTy) {
        for (User *U : CalledArg->users()) {
          if (Instruction *Inst = dyn_cast<Instruction>(U)) {
            if ((ElemTy = deduceElementTypeHelper(Inst)) != nullptr)
              break;
          }
        }
      }
      HaveTypes |= ElemTy != nullptr;
      CalledArgTys.push_back(ElemTy);
    }
  }

  std::string DemangledName =
      getOclOrSpirvBuiltinDemangledName(CI->getCalledFunction()->getName());
  if (DemangledName.empty() && !HaveTypes)
    return;

  for (unsigned OpIdx = 0; OpIdx < CI->arg_size(); OpIdx++) {
    Value *ArgOperand = CI->getArgOperand(OpIdx);
    if (!isa<PointerType>(ArgOperand->getType()) &&
        !isa<TypedPointerType>(ArgOperand->getType()))
      continue;

    // Constants (nulls/undefs) are handled in insertAssignPtrTypeIntrs()
    if (!isa<Instruction>(ArgOperand) && !isa<Argument>(ArgOperand))
      continue;

    Type *ExpectedType =
        OpIdx < CalledArgTys.size() ? CalledArgTys[OpIdx] : nullptr;
    if (!ExpectedType && !DemangledName.empty())
      ExpectedType = SPIRV::parseBuiltinCallArgumentBaseType(
          DemangledName, OpIdx, I->getContext());
    if (!ExpectedType)
      continue;

    if (ExpectedType->isTargetExtTy())
      insertAssignTypeInstrForTargetExtTypes(cast<TargetExtType>(ExpectedType),
                                             ArgOperand, B);
    else
      replacePointerOperandWithPtrCast(CI, ArgOperand, ExpectedType, OpIdx, B);
  }
}

Instruction *SPIRVEmitIntrinsics::visitInsertElementInst(InsertElementInst &I) {
  SmallVector<Type *, 4> Types = {I.getType(), I.getOperand(0)->getType(),
                                  I.getOperand(1)->getType(),
                                  I.getOperand(2)->getType()};
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  SmallVector<Value *> Args(I.op_begin(), I.op_end());
  auto *NewI = B.CreateIntrinsic(Intrinsic::spv_insertelt, {Types}, {Args});
  std::string InstName = I.hasName() ? I.getName().str() : "";
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  NewI->setName(InstName);
  return NewI;
}

Instruction *
SPIRVEmitIntrinsics::visitExtractElementInst(ExtractElementInst &I) {
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  SmallVector<Type *, 3> Types = {I.getType(), I.getVectorOperandType(),
                                  I.getIndexOperand()->getType()};
  SmallVector<Value *, 2> Args = {I.getVectorOperand(), I.getIndexOperand()};
  auto *NewI = B.CreateIntrinsic(Intrinsic::spv_extractelt, {Types}, {Args});
  std::string InstName = I.hasName() ? I.getName().str() : "";
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  NewI->setName(InstName);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitInsertValueInst(InsertValueInst &I) {
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  SmallVector<Type *, 1> Types = {I.getInsertedValueOperand()->getType()};
  SmallVector<Value *> Args;
  for (auto &Op : I.operands())
    if (isa<UndefValue>(Op))
      Args.push_back(UndefValue::get(B.getInt32Ty()));
    else
      Args.push_back(Op);
  for (auto &Op : I.indices())
    Args.push_back(B.getInt32(Op));
  Instruction *NewI =
      B.CreateIntrinsic(Intrinsic::spv_insertv, {Types}, {Args});
  replaceMemInstrUses(&I, NewI, B);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitExtractValueInst(ExtractValueInst &I) {
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  SmallVector<Value *> Args;
  for (auto &Op : I.operands())
    Args.push_back(Op);
  for (auto &Op : I.indices())
    Args.push_back(B.getInt32(Op));
  auto *NewI =
      B.CreateIntrinsic(Intrinsic::spv_extractv, {I.getType()}, {Args});
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitLoadInst(LoadInst &I) {
  if (!I.getType()->isAggregateType())
    return &I;
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  TrackConstants = false;
  const auto *TLI = TM->getSubtargetImpl()->getTargetLowering();
  MachineMemOperand::Flags Flags =
      TLI->getLoadMemOperandFlags(I, F->getParent()->getDataLayout());
  auto *NewI =
      B.CreateIntrinsic(Intrinsic::spv_load, {I.getOperand(0)->getType()},
                        {I.getPointerOperand(), B.getInt16(Flags),
                         B.getInt8(I.getAlign().value())});
  replaceMemInstrUses(&I, NewI, B);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitStoreInst(StoreInst &I) {
  if (!AggrStores.contains(&I))
    return &I;
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  TrackConstants = false;
  const auto *TLI = TM->getSubtargetImpl()->getTargetLowering();
  MachineMemOperand::Flags Flags =
      TLI->getStoreMemOperandFlags(I, F->getParent()->getDataLayout());
  auto *PtrOp = I.getPointerOperand();
  auto *NewI = B.CreateIntrinsic(
      Intrinsic::spv_store, {I.getValueOperand()->getType(), PtrOp->getType()},
      {I.getValueOperand(), PtrOp, B.getInt16(Flags),
       B.getInt8(I.getAlign().value())});
  I.eraseFromParent();
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitAllocaInst(AllocaInst &I) {
  Value *ArraySize = nullptr;
  if (I.isArrayAllocation()) {
    const SPIRVSubtarget *STI = TM->getSubtargetImpl(*I.getFunction());
    if (!STI->canUseExtension(
            SPIRV::Extension::SPV_INTEL_variable_length_array))
      report_fatal_error(
          "array allocation: this instruction requires the following "
          "SPIR-V extension: SPV_INTEL_variable_length_array",
          false);
    ArraySize = I.getArraySize();
  }
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  TrackConstants = false;
  Type *PtrTy = I.getType();
  auto *NewI =
      ArraySize ? B.CreateIntrinsic(Intrinsic::spv_alloca_array,
                                    {PtrTy, ArraySize->getType()}, {ArraySize})
                : B.CreateIntrinsic(Intrinsic::spv_alloca, {PtrTy}, {});
  std::string InstName = I.hasName() ? I.getName().str() : "";
  I.replaceAllUsesWith(NewI);
  I.eraseFromParent();
  NewI->setName(InstName);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
  assert(I.getType()->isAggregateType() && "Aggregate result is expected");
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  SmallVector<Value *> Args;
  for (auto &Op : I.operands())
    Args.push_back(Op);
  Args.push_back(B.getInt32(I.getSyncScopeID()));
  Args.push_back(B.getInt32(
      static_cast<uint32_t>(getMemSemantics(I.getSuccessOrdering()))));
  Args.push_back(B.getInt32(
      static_cast<uint32_t>(getMemSemantics(I.getFailureOrdering()))));
  auto *NewI = B.CreateIntrinsic(Intrinsic::spv_cmpxchg,
                                 {I.getPointerOperand()->getType()}, {Args});
  replaceMemInstrUses(&I, NewI, B);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitUnreachableInst(UnreachableInst &I) {
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  B.CreateIntrinsic(Intrinsic::spv_unreachable, {}, {});
  return &I;
}

void SPIRVEmitIntrinsics::processGlobalValue(GlobalVariable &GV,
                                             IRBuilder<> &B) {
  // Skip special artifical variable llvm.global.annotations.
  if (GV.getName() == "llvm.global.annotations")
    return;
  if (GV.hasInitializer() && !isa<UndefValue>(GV.getInitializer())) {
    // Deduce element type and store results in Global Registry.
    // Result is ignored, because TypedPointerType is not supported
    // by llvm IR general logic.
    deduceElementTypeHelper(&GV);
    Constant *Init = GV.getInitializer();
    Type *Ty = isAggrToReplace(Init) ? B.getInt32Ty() : Init->getType();
    Constant *Const = isAggrToReplace(Init) ? B.getInt32(1) : Init;
    auto *InitInst = B.CreateIntrinsic(Intrinsic::spv_init_global,
                                       {GV.getType(), Ty}, {&GV, Const});
    InitInst->setArgOperand(1, Init);
  }
  if ((!GV.hasInitializer() || isa<UndefValue>(GV.getInitializer())) &&
      GV.getNumUses() == 0)
    B.CreateIntrinsic(Intrinsic::spv_unref_global, GV.getType(), &GV);
}

void SPIRVEmitIntrinsics::insertAssignPtrTypeIntrs(Instruction *I,
                                                   IRBuilder<> &B) {
  reportFatalOnTokenType(I);
  if (!isPointerTy(I->getType()) || !requireAssignType(I) ||
      isa<BitCastInst>(I))
    return;

  setInsertPointSkippingPhis(B, I->getNextNode());

  Type *ElemTy = deduceElementType(I);
  Constant *EltTyConst = UndefValue::get(ElemTy);
  unsigned AddressSpace = getPointerAddressSpace(I->getType());
  CallInst *CI = buildIntrWithMD(Intrinsic::spv_assign_ptr_type, {I->getType()},
                                 EltTyConst, I, {B.getInt32(AddressSpace)}, B);
  GR->addDeducedElementType(CI, ElemTy);
  AssignPtrTypeInstr[I] = CI;
}

void SPIRVEmitIntrinsics::insertAssignTypeIntrs(Instruction *I,
                                                IRBuilder<> &B) {
  reportFatalOnTokenType(I);
  Type *Ty = I->getType();
  if (!Ty->isVoidTy() && !isPointerTy(Ty) && requireAssignType(I)) {
    setInsertPointSkippingPhis(B, I->getNextNode());
    Type *TypeToAssign = Ty;
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      if (II->getIntrinsicID() == Intrinsic::spv_const_composite ||
          II->getIntrinsicID() == Intrinsic::spv_undef) {
        auto It = AggrConstTypes.find(II);
        if (It == AggrConstTypes.end())
          report_fatal_error("Unknown composite intrinsic type");
        TypeToAssign = It->second;
      }
    }
    Constant *Const = UndefValue::get(TypeToAssign);
    buildIntrWithMD(Intrinsic::spv_assign_type, {Ty}, Const, I, {}, B);
  }
  for (const auto &Op : I->operands()) {
    if (isa<ConstantPointerNull>(Op) || isa<UndefValue>(Op) ||
        // Check GetElementPtrConstantExpr case.
        (isa<ConstantExpr>(Op) && isa<GEPOperator>(Op))) {
      setInsertPointSkippingPhis(B, I);
      if (isa<UndefValue>(Op) && Op->getType()->isAggregateType())
        buildIntrWithMD(Intrinsic::spv_assign_type, {B.getInt32Ty()}, Op,
                        UndefValue::get(B.getInt32Ty()), {}, B);
      else if (!isa<Instruction>(Op)) // TODO: This case could be removed
        buildIntrWithMD(Intrinsic::spv_assign_type, {Op->getType()}, Op, Op, {},
                        B);
    }
  }
}

void SPIRVEmitIntrinsics::processInstrAfterVisit(Instruction *I,
                                                 IRBuilder<> &B) {
  auto *II = dyn_cast<IntrinsicInst>(I);
  if (II && II->getIntrinsicID() == Intrinsic::spv_const_composite &&
      TrackConstants) {
    B.SetInsertPoint(I->getNextNode());
    Type *Ty = B.getInt32Ty();
    auto t = AggrConsts.find(I);
    assert(t != AggrConsts.end());
    auto *NewOp = buildIntrWithMD(Intrinsic::spv_track_constant, {Ty, Ty},
                                  t->second, I, {}, B);
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
      B.SetInsertPoint(I);
      auto *NewOp =
          buildIntrWithMD(Intrinsic::spv_track_constant,
                          {Op->getType(), Op->getType()}, Op, Op, {}, B);
      I->setOperand(OpNo, NewOp);
    }
  }
  if (I->hasName()) {
    reportFatalOnTokenType(I);
    setInsertPointSkippingPhis(B, I->getNextNode());
    std::vector<Value *> Args = {I};
    addStringImm(I->getName(), B, Args);
    B.CreateIntrinsic(Intrinsic::spv_assign_name, {I->getType()}, Args);
  }
}

Type *SPIRVEmitIntrinsics::deduceFunParamElementType(Function *F,
                                                     unsigned OpIdx) {
  std::unordered_set<Function *> FVisited;
  return deduceFunParamElementType(F, OpIdx, FVisited);
}

Type *SPIRVEmitIntrinsics::deduceFunParamElementType(
    Function *F, unsigned OpIdx, std::unordered_set<Function *> &FVisited) {
  // maybe a cycle
  if (FVisited.find(F) != FVisited.end())
    return nullptr;
  FVisited.insert(F);

  std::unordered_set<Value *> Visited;
  SmallVector<std::pair<Function *, unsigned>> Lookup;
  // search in function's call sites
  for (User *U : F->users()) {
    CallInst *CI = dyn_cast<CallInst>(U);
    if (!CI || OpIdx >= CI->arg_size())
      continue;
    Value *OpArg = CI->getArgOperand(OpIdx);
    if (!isPointerTy(OpArg->getType()))
      continue;
    // maybe we already know operand's element type
    if (Type *KnownTy = GR->findDeducedElementType(OpArg))
      return KnownTy;
    // try to deduce from the operand itself
    Visited.clear();
    if (Type *Ty = deduceElementTypeHelper(OpArg, Visited))
      return Ty;
    // search in actual parameter's users
    for (User *OpU : OpArg->users()) {
      Instruction *Inst = dyn_cast<Instruction>(OpU);
      if (!Inst || Inst == CI)
        continue;
      Visited.clear();
      if (Type *Ty = deduceElementTypeHelper(Inst, Visited))
        return Ty;
    }
    // check if it's a formal parameter of the outer function
    if (!CI->getParent() || !CI->getParent()->getParent())
      continue;
    Function *OuterF = CI->getParent()->getParent();
    if (FVisited.find(OuterF) != FVisited.end())
      continue;
    for (unsigned i = 0; i < OuterF->arg_size(); ++i) {
      if (OuterF->getArg(i) == OpArg) {
        Lookup.push_back(std::make_pair(OuterF, i));
        break;
      }
    }
  }

  // search in function parameters
  for (auto &Pair : Lookup) {
    if (Type *Ty = deduceFunParamElementType(Pair.first, Pair.second, FVisited))
      return Ty;
  }

  return nullptr;
}

void SPIRVEmitIntrinsics::processParamTypes(Function *F, IRBuilder<> &B) {
  B.SetInsertPointPastAllocas(F);
  for (unsigned OpIdx = 0; OpIdx < F->arg_size(); ++OpIdx) {
    Argument *Arg = F->getArg(OpIdx);
    if (!isUntypedPointerTy(Arg->getType()))
      continue;

    Type *ElemTy = GR->findDeducedElementType(Arg);
    if (!ElemTy) {
      if (hasPointeeTypeAttr(Arg) &&
          (ElemTy = getPointeeTypeByAttr(Arg)) != nullptr) {
        GR->addDeducedElementType(Arg, ElemTy);
      } else if ((ElemTy = deduceFunParamElementType(F, OpIdx)) != nullptr) {
        CallInst *AssignPtrTyCI = buildIntrWithMD(
            Intrinsic::spv_assign_ptr_type, {Arg->getType()},
            Constant::getNullValue(ElemTy), Arg,
            {B.getInt32(getPointerAddressSpace(Arg->getType()))}, B);
        GR->addDeducedElementType(AssignPtrTyCI, ElemTy);
        GR->addDeducedElementType(Arg, ElemTy);
        AssignPtrTypeInstr[Arg] = AssignPtrTyCI;
      }
    }
  }
}

bool SPIRVEmitIntrinsics::runOnFunction(Function &Func) {
  if (Func.isDeclaration())
    return false;

  const SPIRVSubtarget &ST = TM->getSubtarget<SPIRVSubtarget>(Func);
  GR = ST.getSPIRVGlobalRegistry();

  F = &Func;
  IRBuilder<> B(Func.getContext());
  AggrConsts.clear();
  AggrConstTypes.clear();
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

  B.SetInsertPoint(&Func.getEntryBlock(), Func.getEntryBlock().begin());
  for (auto &GV : Func.getParent()->globals())
    processGlobalValue(GV, B);

  preprocessUndefs(B);
  preprocessCompositeConstants(B);
  SmallVector<Instruction *> Worklist;
  for (auto &I : instructions(Func))
    Worklist.push_back(&I);

  for (auto &I : Worklist) {
    insertAssignPtrTypeIntrs(I, B);
    insertAssignTypeIntrs(I, B);
    insertPtrCastOrAssignTypeInstr(I, B);
  }

  for (auto &I : instructions(Func))
    deduceOperandElementType(&I);

  for (auto *I : Worklist) {
    TrackConstants = true;
    if (!I->getType()->isVoidTy() || isa<StoreInst>(I))
      B.SetInsertPoint(I->getNextNode());
    // Visitors return either the original/newly created instruction for further
    // processing, nullptr otherwise.
    I = visit(*I);
    if (!I)
      continue;
    processInstrAfterVisit(I, B);
  }

  return true;
}

bool SPIRVEmitIntrinsics::runOnModule(Module &M) {
  bool Changed = false;

  for (auto &F : M) {
    Changed |= runOnFunction(F);
  }

  for (auto &F : M) {
    // check if function parameter types are set
    if (!F.isDeclaration() && !F.isIntrinsic()) {
      const SPIRVSubtarget &ST = TM->getSubtarget<SPIRVSubtarget>(F);
      GR = ST.getSPIRVGlobalRegistry();
      IRBuilder<> B(F.getContext());
      processParamTypes(&F, B);
    }
  }

  return Changed;
}

ModulePass *llvm::createSPIRVEmitIntrinsicsPass(SPIRVTargetMachine *TM) {
  return new SPIRVEmitIntrinsics(TM);
}
