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
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/TypedPointerType.h"

#include <queue>
#include <unordered_set>

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
namespace SPIRV {
#define GET_BuiltinGroup_DECL
#include "SPIRVGenTables.inc"
} // namespace SPIRV
void initializeSPIRVEmitIntrinsicsPass(PassRegistry &);
} // namespace llvm

namespace {

inline MetadataAsValue *buildMD(Value *Arg) {
  LLVMContext &Ctx = Arg->getContext();
  return MetadataAsValue::get(
      Ctx, MDNode::get(Ctx, ValueAsMetadata::getConstant(Arg)));
}

class SPIRVEmitIntrinsics
    : public ModulePass,
      public InstVisitor<SPIRVEmitIntrinsics, Instruction *> {
  SPIRVTargetMachine *TM = nullptr;
  SPIRVGlobalRegistry *GR = nullptr;
  Function *CurrF = nullptr;
  bool TrackConstants = true;
  bool HaveFunPtrs = false;
  DenseMap<Instruction *, Constant *> AggrConsts;
  DenseMap<Instruction *, Type *> AggrConstTypes;
  DenseSet<Instruction *> AggrStores;
  SPIRV::InstructionSet::InstructionSet InstrSet;

  // map of function declarations to <pointer arg index => element type>
  DenseMap<Function *, SmallVector<std::pair<unsigned, Type *>>> FDeclPtrTys;

  // a register of Instructions that don't have a complete type definition
  bool CanTodoType = true;
  unsigned TodoTypeSz = 0;
  DenseMap<Value *, bool> TodoType;
  void insertTodoType(Value *Op) {
    // TODO: add isa<CallInst>(Op) to no-insert
    if (CanTodoType && !isa<GetElementPtrInst>(Op)) {
      auto It = TodoType.try_emplace(Op, true);
      if (It.second)
        ++TodoTypeSz;
    }
  }
  void eraseTodoType(Value *Op) {
    auto It = TodoType.find(Op);
    if (It != TodoType.end() && It->second) {
      TodoType[Op] = false;
      --TodoTypeSz;
    }
  }
  bool isTodoType(Value *Op) {
    if (isa<GetElementPtrInst>(Op))
      return false;
    auto It = TodoType.find(Op);
    return It != TodoType.end() && It->second;
  }
  // a register of Instructions that were visited by deduceOperandElementType()
  // to validate operand types with an instruction
  std::unordered_set<Instruction *> TypeValidated;

  // well known result types of builtins
  enum WellKnownTypes { Event };

  // deduce element type of untyped pointers
  Type *deduceElementType(Value *I, bool UnknownElemTypeI8);
  Type *deduceElementTypeHelper(Value *I, bool UnknownElemTypeI8);
  Type *deduceElementTypeHelper(Value *I, std::unordered_set<Value *> &Visited,
                                bool UnknownElemTypeI8,
                                bool IgnoreKnownType = false);
  Type *deduceElementTypeByValueDeep(Type *ValueTy, Value *Operand,
                                     bool UnknownElemTypeI8);
  Type *deduceElementTypeByValueDeep(Type *ValueTy, Value *Operand,
                                     std::unordered_set<Value *> &Visited,
                                     bool UnknownElemTypeI8);
  Type *deduceElementTypeByUsersDeep(Value *Op,
                                     std::unordered_set<Value *> &Visited,
                                     bool UnknownElemTypeI8);
  void maybeAssignPtrType(Type *&Ty, Value *I, Type *RefTy,
                          bool UnknownElemTypeI8);

  // deduce nested types of composites
  Type *deduceNestedTypeHelper(User *U, bool UnknownElemTypeI8);
  Type *deduceNestedTypeHelper(User *U, Type *Ty,
                               std::unordered_set<Value *> &Visited,
                               bool UnknownElemTypeI8);

  // deduce Types of operands of the Instruction if possible
  void deduceOperandElementType(Instruction *I,
                                SmallPtrSet<Instruction *, 4> *UncompleteRets,
                                const SmallPtrSet<Value *, 4> *AskOps = nullptr,
                                bool IsPostprocessing = false);

  void preprocessCompositeConstants(IRBuilder<> &B);
  void preprocessUndefs(IRBuilder<> &B);

  CallInst *buildIntrWithMD(Intrinsic::ID IntrID, ArrayRef<Type *> Types,
                            Value *Arg, Value *Arg2, ArrayRef<Constant *> Imms,
                            IRBuilder<> &B) {
    SmallVector<Value *, 4> Args;
    Args.push_back(Arg2);
    Args.push_back(buildMD(Arg));
    for (auto *Imm : Imms)
      Args.push_back(Imm);
    return B.CreateIntrinsic(IntrID, {Types}, Args);
  }

  Type *reconstructType(Value *Op, bool UnknownElemTypeI8,
                        bool IsPostprocessing);

  void buildAssignType(IRBuilder<> &B, Type *ElemTy, Value *Arg);
  void buildAssignPtr(IRBuilder<> &B, Type *ElemTy, Value *Arg);
  void updateAssignType(CallInst *AssignCI, Value *Arg, Value *OfType);

  void replaceMemInstrUses(Instruction *Old, Instruction *New, IRBuilder<> &B);
  void processInstrAfterVisit(Instruction *I, IRBuilder<> &B);
  bool insertAssignPtrTypeIntrs(Instruction *I, IRBuilder<> &B,
                                bool UnknownElemTypeI8);
  void insertAssignTypeIntrs(Instruction *I, IRBuilder<> &B);
  void insertAssignPtrTypeTargetExt(TargetExtType *AssignedType, Value *V,
                                    IRBuilder<> &B);
  void replacePointerOperandWithPtrCast(Instruction *I, Value *Pointer,
                                        Type *ExpectedElementType,
                                        unsigned OperandToReplace,
                                        IRBuilder<> &B);
  void insertPtrCastOrAssignTypeInstr(Instruction *I, IRBuilder<> &B);
  void insertSpirvDecorations(Instruction *I, IRBuilder<> &B);
  void processGlobalValue(GlobalVariable &GV, IRBuilder<> &B);
  void processParamTypes(Function *F, IRBuilder<> &B);
  void processParamTypesByFunHeader(Function *F, IRBuilder<> &B);
  Type *deduceFunParamElementType(Function *F, unsigned OpIdx);
  Type *deduceFunParamElementType(Function *F, unsigned OpIdx,
                                  std::unordered_set<Function *> &FVisited);

  bool deduceOperandElementTypeCalledFunction(
      CallInst *CI, SmallVector<std::pair<Value *, unsigned>> &Ops,
      Type *&KnownElemTy);
  void deduceOperandElementTypeFunctionPointer(
      CallInst *CI, SmallVector<std::pair<Value *, unsigned>> &Ops,
      Type *&KnownElemTy, bool IsPostprocessing);
  bool deduceOperandElementTypeFunctionRet(
      Instruction *I, SmallPtrSet<Instruction *, 4> *UncompleteRets,
      const SmallPtrSet<Value *, 4> *AskOps, bool IsPostprocessing,
      Type *&KnownElemTy, Value *Op, Function *F);

  CallInst *buildSpvPtrcast(Function *F, Value *Op, Type *ElemTy);
  void replaceUsesOfWithSpvPtrcast(Value *Op, Type *ElemTy, Instruction *I,
                                   DenseMap<Function *, CallInst *> Ptrcasts);
  void propagateElemType(Value *Op, Type *ElemTy,
                         DenseSet<std::pair<Value *, Value *>> &VisitedSubst);
  void
  propagateElemTypeRec(Value *Op, Type *PtrElemTy, Type *CastElemTy,
                       DenseSet<std::pair<Value *, Value *>> &VisitedSubst);
  void propagateElemTypeRec(Value *Op, Type *PtrElemTy, Type *CastElemTy,
                            DenseSet<std::pair<Value *, Value *>> &VisitedSubst,
                            std::unordered_set<Value *> &Visited,
                            DenseMap<Function *, CallInst *> Ptrcasts);

  void replaceAllUsesWith(Value *Src, Value *Dest, bool DeleteOld = true);
  void replaceAllUsesWithAndErase(IRBuilder<> &B, Instruction *Src,
                                  Instruction *Dest, bool DeleteOld = true);

  void applyDemangledPtrArgTypes(IRBuilder<> &B);

  bool runOnFunction(Function &F);
  bool postprocessTypes(Module &M);
  bool processFunctionPointers(Module &M);
  void parseFunDeclarations(Module &M);

  void useRoundingMode(ConstrainedFPIntrinsic *FPI, IRBuilder<> &B);

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
  Instruction *visitCallInst(CallInst &I);

  StringRef getPassName() const override { return "SPIRV emit intrinsics"; }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
};

bool isConvergenceIntrinsic(const Instruction *I) {
  const auto *II = dyn_cast<IntrinsicInst>(I);
  if (!II)
    return false;

  return II->getIntrinsicID() == Intrinsic::experimental_convergence_entry ||
         II->getIntrinsicID() == Intrinsic::experimental_convergence_loop ||
         II->getIntrinsicID() == Intrinsic::experimental_convergence_anchor;
}

bool expectIgnoredInIRTranslation(const Instruction *I) {
  const auto *II = dyn_cast<IntrinsicInst>(I);
  if (!II)
    return false;
  return II->getIntrinsicID() == Intrinsic::invariant_start;
}

bool allowEmitFakeUse(const Value *Arg) {
  if (isSpvIntrinsic(Arg))
    return false;
  if (dyn_cast<AtomicCmpXchgInst>(Arg) || dyn_cast<InsertValueInst>(Arg) ||
      dyn_cast<UndefValue>(Arg))
    return false;
  if (const auto *LI = dyn_cast<LoadInst>(Arg))
    if (LI->getType()->isAggregateType())
      return false;
  return true;
}

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

static bool isAggrConstForceInt32(const Value *V) {
  return isa<ConstantArray>(V) || isa<ConstantStruct>(V) ||
         isa<ConstantDataArray>(V) ||
         (isa<ConstantAggregateZero>(V) && !V->getType()->isVectorTy());
}

static void setInsertPointSkippingPhis(IRBuilder<> &B, Instruction *I) {
  if (isa<PHINode>(I))
    B.SetInsertPoint(I->getParent()->getFirstNonPHIOrDbgOrAlloca());
  else
    B.SetInsertPoint(I);
}

static void setInsertPointAfterDef(IRBuilder<> &B, Instruction *I) {
  B.SetCurrentDebugLocation(I->getDebugLoc());
  if (I->getType()->isVoidTy())
    B.SetInsertPoint(I->getNextNode());
  else
    B.SetInsertPoint(*I->getInsertionPointAfterDef());
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

static void emitAssignName(Instruction *I, IRBuilder<> &B) {
  if (!I->hasName() || I->getType()->isAggregateType() ||
      expectIgnoredInIRTranslation(I))
    return;
  reportFatalOnTokenType(I);
  setInsertPointAfterDef(B, I);
  std::vector<Value *> Args = {I};
  addStringImm(I->getName(), B, Args);
  B.CreateIntrinsic(Intrinsic::spv_assign_name, {I->getType()}, Args);
}

void SPIRVEmitIntrinsics::replaceAllUsesWith(Value *Src, Value *Dest,
                                             bool DeleteOld) {
  Src->replaceAllUsesWith(Dest);
  // Update deduced type records
  GR->updateIfExistDeducedElementType(Src, Dest, DeleteOld);
  GR->updateIfExistAssignPtrTypeInstr(Src, Dest, DeleteOld);
  // Update uncomplete type records if any
  if (isTodoType(Src)) {
    if (DeleteOld)
      eraseTodoType(Src);
    insertTodoType(Dest);
  }
}

void SPIRVEmitIntrinsics::replaceAllUsesWithAndErase(IRBuilder<> &B,
                                                     Instruction *Src,
                                                     Instruction *Dest,
                                                     bool DeleteOld) {
  replaceAllUsesWith(Src, Dest, DeleteOld);
  std::string Name = Src->hasName() ? Src->getName().str() : "";
  Src->eraseFromParent();
  if (!Name.empty()) {
    Dest->setName(Name);
    emitAssignName(Dest, B);
  }
}

static bool IsKernelArgInt8(Function *F, StoreInst *SI) {
  return SI && F->getCallingConv() == CallingConv::SPIR_KERNEL &&
         isPointerTy(SI->getValueOperand()->getType()) &&
         isa<Argument>(SI->getValueOperand());
}

// Maybe restore original function return type.
static inline Type *restoreMutatedType(SPIRVGlobalRegistry *GR, Instruction *I,
                                       Type *Ty) {
  CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI || CI->isIndirectCall() || CI->isInlineAsm() ||
      !CI->getCalledFunction() || CI->getCalledFunction()->isIntrinsic())
    return Ty;
  if (Type *OriginalTy = GR->findMutated(CI->getCalledFunction()))
    return OriginalTy;
  return Ty;
}

// Reconstruct type with nested element types according to deduced type info.
// Return nullptr if no detailed type info is available.
Type *SPIRVEmitIntrinsics::reconstructType(Value *Op, bool UnknownElemTypeI8,
                                           bool IsPostprocessing) {
  Type *Ty = Op->getType();
  if (auto *OpI = dyn_cast<Instruction>(Op))
    Ty = restoreMutatedType(GR, OpI, Ty);
  if (!isUntypedPointerTy(Ty))
    return Ty;
  // try to find the pointee type
  if (Type *NestedTy = GR->findDeducedElementType(Op))
    return getTypedPointerWrapper(NestedTy, getPointerAddressSpace(Ty));
  // not a pointer according to the type info (e.g., Event object)
  CallInst *CI = GR->findAssignPtrTypeInstr(Op);
  if (CI) {
    MetadataAsValue *MD = cast<MetadataAsValue>(CI->getArgOperand(1));
    return cast<ConstantAsMetadata>(MD->getMetadata())->getType();
  }
  if (UnknownElemTypeI8) {
    if (!IsPostprocessing)
      insertTodoType(Op);
    return getTypedPointerWrapper(IntegerType::getInt8Ty(Op->getContext()),
                                  getPointerAddressSpace(Ty));
  }
  return nullptr;
}

void SPIRVEmitIntrinsics::buildAssignType(IRBuilder<> &B, Type *Ty,
                                          Value *Arg) {
  Value *OfType = PoisonValue::get(Ty);
  CallInst *AssignCI = nullptr;
  if (Arg->getType()->isAggregateType() && Ty->isAggregateType() &&
      allowEmitFakeUse(Arg)) {
    LLVMContext &Ctx = Arg->getContext();
    SmallVector<Metadata *, 2> ArgMDs{
        MDNode::get(Ctx, ValueAsMetadata::getConstant(OfType)),
        MDString::get(Ctx, Arg->getName())};
    B.CreateIntrinsic(Intrinsic::spv_value_md, {},
                      {MetadataAsValue::get(Ctx, MDTuple::get(Ctx, ArgMDs))});
    AssignCI = B.CreateIntrinsic(Intrinsic::fake_use, {}, {Arg});
  } else {
    AssignCI = buildIntrWithMD(Intrinsic::spv_assign_type, {Arg->getType()},
                               OfType, Arg, {}, B);
  }
  GR->addAssignPtrTypeInstr(Arg, AssignCI);
}

void SPIRVEmitIntrinsics::buildAssignPtr(IRBuilder<> &B, Type *ElemTy,
                                         Value *Arg) {
  Value *OfType = PoisonValue::get(ElemTy);
  CallInst *AssignPtrTyCI = GR->findAssignPtrTypeInstr(Arg);
  if (AssignPtrTyCI == nullptr ||
      AssignPtrTyCI->getParent()->getParent() != CurrF) {
    AssignPtrTyCI = buildIntrWithMD(
        Intrinsic::spv_assign_ptr_type, {Arg->getType()}, OfType, Arg,
        {B.getInt32(getPointerAddressSpace(Arg->getType()))}, B);
    GR->addDeducedElementType(AssignPtrTyCI, ElemTy);
    GR->addDeducedElementType(Arg, ElemTy);
    GR->addAssignPtrTypeInstr(Arg, AssignPtrTyCI);
  } else {
    updateAssignType(AssignPtrTyCI, Arg, OfType);
  }
}

void SPIRVEmitIntrinsics::updateAssignType(CallInst *AssignCI, Value *Arg,
                                           Value *OfType) {
  AssignCI->setArgOperand(1, buildMD(OfType));
  if (cast<IntrinsicInst>(AssignCI)->getIntrinsicID() !=
      Intrinsic::spv_assign_ptr_type)
    return;

  // update association with the pointee type
  Type *ElemTy = OfType->getType();
  GR->addDeducedElementType(AssignCI, ElemTy);
  GR->addDeducedElementType(Arg, ElemTy);
}

CallInst *SPIRVEmitIntrinsics::buildSpvPtrcast(Function *F, Value *Op,
                                               Type *ElemTy) {
  IRBuilder<> B(Op->getContext());
  if (auto *OpI = dyn_cast<Instruction>(Op)) {
    // spv_ptrcast's argument Op denotes an instruction that generates
    // a value, and we may use getInsertionPointAfterDef()
    setInsertPointAfterDef(B, OpI);
  } else if (auto *OpA = dyn_cast<Argument>(Op)) {
    B.SetInsertPointPastAllocas(OpA->getParent());
    B.SetCurrentDebugLocation(DebugLoc());
  } else {
    B.SetInsertPoint(F->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
  }
  Type *OpTy = Op->getType();
  SmallVector<Type *, 2> Types = {OpTy, OpTy};
  SmallVector<Value *, 2> Args = {Op, buildMD(PoisonValue::get(ElemTy)),
                                  B.getInt32(getPointerAddressSpace(OpTy))};
  CallInst *PtrCasted =
      B.CreateIntrinsic(Intrinsic::spv_ptrcast, {Types}, Args);
  buildAssignPtr(B, ElemTy, PtrCasted);
  return PtrCasted;
}

void SPIRVEmitIntrinsics::replaceUsesOfWithSpvPtrcast(
    Value *Op, Type *ElemTy, Instruction *I,
    DenseMap<Function *, CallInst *> Ptrcasts) {
  Function *F = I->getParent()->getParent();
  CallInst *PtrCastedI = nullptr;
  auto It = Ptrcasts.find(F);
  if (It == Ptrcasts.end()) {
    PtrCastedI = buildSpvPtrcast(F, Op, ElemTy);
    Ptrcasts[F] = PtrCastedI;
  } else {
    PtrCastedI = It->second;
  }
  I->replaceUsesOfWith(Op, PtrCastedI);
}

void SPIRVEmitIntrinsics::propagateElemType(
    Value *Op, Type *ElemTy,
    DenseSet<std::pair<Value *, Value *>> &VisitedSubst) {
  DenseMap<Function *, CallInst *> Ptrcasts;
  SmallVector<User *> Users(Op->users());
  for (auto *U : Users) {
    if (!isa<Instruction>(U) || isSpvIntrinsic(U))
      continue;
    if (!VisitedSubst.insert(std::make_pair(U, Op)).second)
      continue;
    Instruction *UI = dyn_cast<Instruction>(U);
    // If the instruction was validated already, we need to keep it valid by
    // keeping current Op type.
    if (isa<GetElementPtrInst>(UI) ||
        TypeValidated.find(UI) != TypeValidated.end())
      replaceUsesOfWithSpvPtrcast(Op, ElemTy, UI, Ptrcasts);
  }
}

void SPIRVEmitIntrinsics::propagateElemTypeRec(
    Value *Op, Type *PtrElemTy, Type *CastElemTy,
    DenseSet<std::pair<Value *, Value *>> &VisitedSubst) {
  std::unordered_set<Value *> Visited;
  DenseMap<Function *, CallInst *> Ptrcasts;
  propagateElemTypeRec(Op, PtrElemTy, CastElemTy, VisitedSubst, Visited,
                       Ptrcasts);
}

void SPIRVEmitIntrinsics::propagateElemTypeRec(
    Value *Op, Type *PtrElemTy, Type *CastElemTy,
    DenseSet<std::pair<Value *, Value *>> &VisitedSubst,
    std::unordered_set<Value *> &Visited,
    DenseMap<Function *, CallInst *> Ptrcasts) {
  if (!Visited.insert(Op).second)
    return;
  SmallVector<User *> Users(Op->users());
  for (auto *U : Users) {
    if (!isa<Instruction>(U) || isSpvIntrinsic(U))
      continue;
    if (!VisitedSubst.insert(std::make_pair(U, Op)).second)
      continue;
    Instruction *UI = dyn_cast<Instruction>(U);
    // If the instruction was validated already, we need to keep it valid by
    // keeping current Op type.
    if (isa<GetElementPtrInst>(UI) ||
        TypeValidated.find(UI) != TypeValidated.end())
      replaceUsesOfWithSpvPtrcast(Op, CastElemTy, UI, Ptrcasts);
  }
}

// Set element pointer type to the given value of ValueTy and tries to
// specify this type further (recursively) by Operand value, if needed.

Type *
SPIRVEmitIntrinsics::deduceElementTypeByValueDeep(Type *ValueTy, Value *Operand,
                                                  bool UnknownElemTypeI8) {
  std::unordered_set<Value *> Visited;
  return deduceElementTypeByValueDeep(ValueTy, Operand, Visited,
                                      UnknownElemTypeI8);
}

Type *SPIRVEmitIntrinsics::deduceElementTypeByValueDeep(
    Type *ValueTy, Value *Operand, std::unordered_set<Value *> &Visited,
    bool UnknownElemTypeI8) {
  Type *Ty = ValueTy;
  if (Operand) {
    if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
      if (Type *NestedTy =
              deduceElementTypeHelper(Operand, Visited, UnknownElemTypeI8))
        Ty = getTypedPointerWrapper(NestedTy, PtrTy->getAddressSpace());
    } else {
      Ty = deduceNestedTypeHelper(dyn_cast<User>(Operand), Ty, Visited,
                                  UnknownElemTypeI8);
    }
  }
  return Ty;
}

// Traverse User instructions to deduce an element pointer type of the operand.
Type *SPIRVEmitIntrinsics::deduceElementTypeByUsersDeep(
    Value *Op, std::unordered_set<Value *> &Visited, bool UnknownElemTypeI8) {
  if (!Op || !isPointerTy(Op->getType()) || isa<ConstantPointerNull>(Op) ||
      isa<UndefValue>(Op))
    return nullptr;

  if (auto ElemTy = getPointeeType(Op->getType()))
    return ElemTy;

  // maybe we already know operand's element type
  if (Type *KnownTy = GR->findDeducedElementType(Op))
    return KnownTy;

  for (User *OpU : Op->users()) {
    if (Instruction *Inst = dyn_cast<Instruction>(OpU)) {
      if (Type *Ty = deduceElementTypeHelper(Inst, Visited, UnknownElemTypeI8))
        return Ty;
    }
  }
  return nullptr;
}

// Implements what we know in advance about intrinsics and builtin calls
// TODO: consider feasibility of this particular case to be generalized by
// encoding knowledge about intrinsics and builtin calls by corresponding
// specification rules
static Type *getPointeeTypeByCallInst(StringRef DemangledName,
                                      Function *CalledF, unsigned OpIdx) {
  if ((DemangledName.starts_with("__spirv_ocl_printf(") ||
       DemangledName.starts_with("printf(")) &&
      OpIdx == 0)
    return IntegerType::getInt8Ty(CalledF->getContext());
  return nullptr;
}

// Deduce and return a successfully deduced Type of the Instruction,
// or nullptr otherwise.
Type *SPIRVEmitIntrinsics::deduceElementTypeHelper(Value *I,
                                                   bool UnknownElemTypeI8) {
  std::unordered_set<Value *> Visited;
  return deduceElementTypeHelper(I, Visited, UnknownElemTypeI8);
}

void SPIRVEmitIntrinsics::maybeAssignPtrType(Type *&Ty, Value *Op, Type *RefTy,
                                             bool UnknownElemTypeI8) {
  if (isUntypedPointerTy(RefTy)) {
    if (!UnknownElemTypeI8)
      return;
    insertTodoType(Op);
  }
  Ty = RefTy;
}

Type *SPIRVEmitIntrinsics::deduceElementTypeHelper(
    Value *I, std::unordered_set<Value *> &Visited, bool UnknownElemTypeI8,
    bool IgnoreKnownType) {
  // allow to pass nullptr as an argument
  if (!I)
    return nullptr;

  // maybe already known
  if (!IgnoreKnownType)
    if (Type *KnownTy = GR->findDeducedElementType(I))
      return KnownTy;

  // maybe a cycle
  if (!Visited.insert(I).second)
    return nullptr;

  // fallback value in case when we fail to deduce a type
  Type *Ty = nullptr;
  // look for known basic patterns of type inference
  if (auto *Ref = dyn_cast<AllocaInst>(I)) {
    maybeAssignPtrType(Ty, I, Ref->getAllocatedType(), UnknownElemTypeI8);
  } else if (auto *Ref = dyn_cast<GetElementPtrInst>(I)) {
    // TODO: not sure if GetElementPtrInst::getTypeAtIndex() does anything
    // useful here
    if (isNestedPointer(Ref->getSourceElementType())) {
      Ty = Ref->getSourceElementType();
      for (Use &U : drop_begin(Ref->indices()))
        Ty = GetElementPtrInst::getTypeAtIndex(Ty, U.get());
    } else {
      Ty = Ref->getResultElementType();
    }
  } else if (auto *Ref = dyn_cast<LoadInst>(I)) {
    Value *Op = Ref->getPointerOperand();
    Type *KnownTy = GR->findDeducedElementType(Op);
    if (!KnownTy)
      KnownTy = Op->getType();
    if (Type *ElemTy = getPointeeType(KnownTy))
      maybeAssignPtrType(Ty, I, ElemTy, UnknownElemTypeI8);
  } else if (auto *Ref = dyn_cast<GlobalValue>(I)) {
    Ty = deduceElementTypeByValueDeep(
        Ref->getValueType(),
        Ref->getNumOperands() > 0 ? Ref->getOperand(0) : nullptr, Visited,
        UnknownElemTypeI8);
  } else if (auto *Ref = dyn_cast<AddrSpaceCastInst>(I)) {
    Type *RefTy = deduceElementTypeHelper(Ref->getPointerOperand(), Visited,
                                          UnknownElemTypeI8);
    maybeAssignPtrType(Ty, I, RefTy, UnknownElemTypeI8);
  } else if (auto *Ref = dyn_cast<BitCastInst>(I)) {
    if (Type *Src = Ref->getSrcTy(), *Dest = Ref->getDestTy();
        isPointerTy(Src) && isPointerTy(Dest))
      Ty = deduceElementTypeHelper(Ref->getOperand(0), Visited,
                                   UnknownElemTypeI8);
  } else if (auto *Ref = dyn_cast<AtomicCmpXchgInst>(I)) {
    Value *Op = Ref->getNewValOperand();
    if (isPointerTy(Op->getType()))
      Ty = deduceElementTypeHelper(Op, Visited, UnknownElemTypeI8);
  } else if (auto *Ref = dyn_cast<AtomicRMWInst>(I)) {
    Value *Op = Ref->getValOperand();
    if (isPointerTy(Op->getType()))
      Ty = deduceElementTypeHelper(Op, Visited, UnknownElemTypeI8);
  } else if (auto *Ref = dyn_cast<PHINode>(I)) {
    Type *BestTy = nullptr;
    unsigned MaxN = 1;
    DenseMap<Type *, unsigned> PhiTys;
    for (int i = Ref->getNumIncomingValues() - 1; i >= 0; --i) {
      Ty = deduceElementTypeByUsersDeep(Ref->getIncomingValue(i), Visited,
                                        UnknownElemTypeI8);
      if (!Ty)
        continue;
      auto It = PhiTys.try_emplace(Ty, 1);
      if (!It.second) {
        ++It.first->second;
        if (It.first->second > MaxN) {
          MaxN = It.first->second;
          BestTy = Ty;
        }
      }
    }
    if (BestTy)
      Ty = BestTy;
  } else if (auto *Ref = dyn_cast<SelectInst>(I)) {
    for (Value *Op : {Ref->getTrueValue(), Ref->getFalseValue()}) {
      Ty = deduceElementTypeByUsersDeep(Op, Visited, UnknownElemTypeI8);
      if (Ty)
        break;
    }
  } else if (auto *CI = dyn_cast<CallInst>(I)) {
    static StringMap<unsigned> ResTypeByArg = {
        {"to_global", 0},
        {"to_local", 0},
        {"to_private", 0},
        {"__spirv_GenericCastToPtr_ToGlobal", 0},
        {"__spirv_GenericCastToPtr_ToLocal", 0},
        {"__spirv_GenericCastToPtr_ToPrivate", 0},
        {"__spirv_GenericCastToPtrExplicit_ToGlobal", 0},
        {"__spirv_GenericCastToPtrExplicit_ToLocal", 0},
        {"__spirv_GenericCastToPtrExplicit_ToPrivate", 0}};
    // TODO: maybe improve performance by caching demangled names
    if (Function *CalledF = CI->getCalledFunction()) {
      std::string DemangledName =
          getOclOrSpirvBuiltinDemangledName(CalledF->getName());
      if (DemangledName.length() > 0)
        DemangledName = SPIRV::lookupBuiltinNameHelper(DemangledName);
      auto AsArgIt = ResTypeByArg.find(DemangledName);
      if (AsArgIt != ResTypeByArg.end())
        Ty = deduceElementTypeHelper(CI->getArgOperand(AsArgIt->second),
                                     Visited, UnknownElemTypeI8);
      else if (Type *KnownRetTy = GR->findDeducedElementType(CalledF))
        Ty = KnownRetTy;
    }
  }

  // remember the found relationship
  if (Ty && !IgnoreKnownType) {
    // specify nested types if needed, otherwise return unchanged
    GR->addDeducedElementType(I, Ty);
  }

  return Ty;
}

// Re-create a type of the value if it has untyped pointer fields, also nested.
// Return the original value type if no corrections of untyped pointer
// information is found or needed.
Type *SPIRVEmitIntrinsics::deduceNestedTypeHelper(User *U,
                                                  bool UnknownElemTypeI8) {
  std::unordered_set<Value *> Visited;
  return deduceNestedTypeHelper(U, U->getType(), Visited, UnknownElemTypeI8);
}

Type *SPIRVEmitIntrinsics::deduceNestedTypeHelper(
    User *U, Type *OrigTy, std::unordered_set<Value *> &Visited,
    bool UnknownElemTypeI8) {
  if (!U)
    return OrigTy;

  // maybe already known
  if (Type *KnownTy = GR->findDeducedCompositeType(U))
    return KnownTy;

  // maybe a cycle
  if (!Visited.insert(U).second)
    return OrigTy;

  if (dyn_cast<StructType>(OrigTy)) {
    SmallVector<Type *> Tys;
    bool Change = false;
    for (unsigned i = 0; i < U->getNumOperands(); ++i) {
      Value *Op = U->getOperand(i);
      Type *OpTy = Op->getType();
      Type *Ty = OpTy;
      if (Op) {
        if (auto *PtrTy = dyn_cast<PointerType>(OpTy)) {
          if (Type *NestedTy =
                  deduceElementTypeHelper(Op, Visited, UnknownElemTypeI8))
            Ty = getTypedPointerWrapper(NestedTy, PtrTy->getAddressSpace());
        } else {
          Ty = deduceNestedTypeHelper(dyn_cast<User>(Op), OpTy, Visited,
                                      UnknownElemTypeI8);
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
        if (Type *NestedTy =
                deduceElementTypeHelper(Op, Visited, UnknownElemTypeI8))
          Ty = getTypedPointerWrapper(NestedTy, PtrTy->getAddressSpace());
      } else {
        Ty = deduceNestedTypeHelper(dyn_cast<User>(Op), OpTy, Visited,
                                    UnknownElemTypeI8);
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
        if (Type *NestedTy =
                deduceElementTypeHelper(Op, Visited, UnknownElemTypeI8))
          Ty = getTypedPointerWrapper(NestedTy, PtrTy->getAddressSpace());
      } else {
        Ty = deduceNestedTypeHelper(dyn_cast<User>(Op), OpTy, Visited,
                                    UnknownElemTypeI8);
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

Type *SPIRVEmitIntrinsics::deduceElementType(Value *I, bool UnknownElemTypeI8) {
  if (Type *Ty = deduceElementTypeHelper(I, UnknownElemTypeI8))
    return Ty;
  if (!UnknownElemTypeI8)
    return nullptr;
  insertTodoType(I);
  return IntegerType::getInt8Ty(I->getContext());
}

static inline Type *getAtomicElemTy(SPIRVGlobalRegistry *GR, Instruction *I,
                                    Value *PointerOperand) {
  Type *PointeeTy = GR->findDeducedElementType(PointerOperand);
  if (PointeeTy && !isUntypedPointerTy(PointeeTy))
    return nullptr;
  auto *PtrTy = dyn_cast<PointerType>(I->getType());
  if (!PtrTy)
    return I->getType();
  if (Type *NestedTy = GR->findDeducedElementType(I))
    return getTypedPointerWrapper(NestedTy, PtrTy->getAddressSpace());
  return nullptr;
}

// Try to deduce element type for a call base. Returns false if this is an
// indirect function invocation, and true otherwise.
bool SPIRVEmitIntrinsics::deduceOperandElementTypeCalledFunction(
    CallInst *CI, SmallVector<std::pair<Value *, unsigned>> &Ops,
    Type *&KnownElemTy) {
  Function *CalledF = CI->getCalledFunction();
  if (!CalledF)
    return false;
  std::string DemangledName =
      getOclOrSpirvBuiltinDemangledName(CalledF->getName());
  if (DemangledName.length() > 0 &&
      !StringRef(DemangledName).starts_with("llvm.")) {
    auto [Grp, Opcode, ExtNo] =
        SPIRV::mapBuiltinToOpcode(DemangledName, InstrSet);
    if (Opcode == SPIRV::OpGroupAsyncCopy) {
      for (unsigned i = 0, PtrCnt = 0; i < CI->arg_size() && PtrCnt < 2; ++i) {
        Value *Op = CI->getArgOperand(i);
        if (!isPointerTy(Op->getType()))
          continue;
        ++PtrCnt;
        if (Type *ElemTy = GR->findDeducedElementType(Op))
          KnownElemTy = ElemTy; // src will rewrite dest if both are defined
        Ops.push_back(std::make_pair(Op, i));
      }
    } else if (Grp == SPIRV::Atomic || Grp == SPIRV::AtomicFloating) {
      if (CI->arg_size() < 2)
        return true;
      Value *Op = CI->getArgOperand(0);
      if (!isPointerTy(Op->getType()))
        return true;
      switch (Opcode) {
      case SPIRV::OpAtomicLoad:
      case SPIRV::OpAtomicCompareExchangeWeak:
      case SPIRV::OpAtomicCompareExchange:
      case SPIRV::OpAtomicExchange:
      case SPIRV::OpAtomicIAdd:
      case SPIRV::OpAtomicISub:
      case SPIRV::OpAtomicOr:
      case SPIRV::OpAtomicXor:
      case SPIRV::OpAtomicAnd:
      case SPIRV::OpAtomicUMin:
      case SPIRV::OpAtomicUMax:
      case SPIRV::OpAtomicSMin:
      case SPIRV::OpAtomicSMax: {
        KnownElemTy = getAtomicElemTy(GR, CI, Op);
        if (!KnownElemTy)
          return true;
        Ops.push_back(std::make_pair(Op, 0));
      } break;
      }
    }
  }
  return true;
}

// Try to deduce element type for a function pointer.
void SPIRVEmitIntrinsics::deduceOperandElementTypeFunctionPointer(
    CallInst *CI, SmallVector<std::pair<Value *, unsigned>> &Ops,
    Type *&KnownElemTy, bool IsPostprocessing) {
  Value *Op = CI->getCalledOperand();
  if (!Op || !isPointerTy(Op->getType()))
    return;
  Ops.push_back(std::make_pair(Op, std::numeric_limits<unsigned>::max()));
  FunctionType *FTy = CI->getFunctionType();
  bool IsNewFTy = false, IsUncomplete = false;
  SmallVector<Type *, 4> ArgTys;
  for (Value *Arg : CI->args()) {
    Type *ArgTy = Arg->getType();
    if (ArgTy->isPointerTy()) {
      if (Type *ElemTy = GR->findDeducedElementType(Arg)) {
        IsNewFTy = true;
        ArgTy = getTypedPointerWrapper(ElemTy, getPointerAddressSpace(ArgTy));
        if (isTodoType(Arg))
          IsUncomplete = true;
      } else {
        IsUncomplete = true;
      }
    }
    ArgTys.push_back(ArgTy);
  }
  Type *RetTy = FTy->getReturnType();
  if (CI->getType()->isPointerTy()) {
    if (Type *ElemTy = GR->findDeducedElementType(CI)) {
      IsNewFTy = true;
      RetTy =
          getTypedPointerWrapper(ElemTy, getPointerAddressSpace(CI->getType()));
      if (isTodoType(CI))
        IsUncomplete = true;
    } else {
      IsUncomplete = true;
    }
  }
  if (!IsPostprocessing && IsUncomplete)
    insertTodoType(Op);
  KnownElemTy =
      IsNewFTy ? FunctionType::get(RetTy, ArgTys, FTy->isVarArg()) : FTy;
}

bool SPIRVEmitIntrinsics::deduceOperandElementTypeFunctionRet(
    Instruction *I, SmallPtrSet<Instruction *, 4> *UncompleteRets,
    const SmallPtrSet<Value *, 4> *AskOps, bool IsPostprocessing,
    Type *&KnownElemTy, Value *Op, Function *F) {
  KnownElemTy = GR->findDeducedElementType(F);
  if (KnownElemTy)
    return false;
  if (Type *OpElemTy = GR->findDeducedElementType(Op)) {
    GR->addDeducedElementType(F, OpElemTy);
    GR->addReturnType(
        F, TypedPointerType::get(OpElemTy,
                                 getPointerAddressSpace(F->getReturnType())));
    // non-recursive update of types in function uses
    DenseSet<std::pair<Value *, Value *>> VisitedSubst{std::make_pair(I, Op)};
    for (User *U : F->users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != F)
        continue;
      if (CallInst *AssignCI = GR->findAssignPtrTypeInstr(CI)) {
        if (Type *PrevElemTy = GR->findDeducedElementType(CI)) {
          updateAssignType(AssignCI, CI, PoisonValue::get(OpElemTy));
          propagateElemType(CI, PrevElemTy, VisitedSubst);
        }
      }
    }
    // Non-recursive update of types in the function uncomplete returns.
    // This may happen just once per a function, the latch is a pair of
    // findDeducedElementType(F) / addDeducedElementType(F, ...).
    // With or without the latch it is a non-recursive call due to
    // UncompleteRets set to nullptr in this call.
    if (UncompleteRets)
      for (Instruction *UncompleteRetI : *UncompleteRets)
        deduceOperandElementType(UncompleteRetI, nullptr, AskOps,
                                 IsPostprocessing);
  } else if (UncompleteRets) {
    UncompleteRets->insert(I);
  }
  TypeValidated.insert(I);
  return true;
}

// If the Instruction has Pointer operands with unresolved types, this function
// tries to deduce them. If the Instruction has Pointer operands with known
// types which differ from expected, this function tries to insert a bitcast to
// resolve the issue.
void SPIRVEmitIntrinsics::deduceOperandElementType(
    Instruction *I, SmallPtrSet<Instruction *, 4> *UncompleteRets,
    const SmallPtrSet<Value *, 4> *AskOps, bool IsPostprocessing) {
  SmallVector<std::pair<Value *, unsigned>> Ops;
  Type *KnownElemTy = nullptr;
  bool Uncomplete = false;
  // look for known basic patterns of type inference
  if (auto *Ref = dyn_cast<PHINode>(I)) {
    if (!isPointerTy(I->getType()) ||
        !(KnownElemTy = GR->findDeducedElementType(I)))
      return;
    Uncomplete = isTodoType(I);
    for (unsigned i = 0; i < Ref->getNumIncomingValues(); i++) {
      Value *Op = Ref->getIncomingValue(i);
      if (isPointerTy(Op->getType()))
        Ops.push_back(std::make_pair(Op, i));
    }
  } else if (auto *Ref = dyn_cast<AddrSpaceCastInst>(I)) {
    KnownElemTy = GR->findDeducedElementType(I);
    if (!KnownElemTy)
      return;
    Uncomplete = isTodoType(I);
    Ops.push_back(std::make_pair(Ref->getPointerOperand(), 0));
  } else if (auto *Ref = dyn_cast<BitCastInst>(I)) {
    if (!isPointerTy(I->getType()))
      return;
    KnownElemTy = GR->findDeducedElementType(I);
    if (!KnownElemTy)
      return;
    Uncomplete = isTodoType(I);
    Ops.push_back(std::make_pair(Ref->getOperand(0), 0));
  } else if (auto *Ref = dyn_cast<GetElementPtrInst>(I)) {
    if (GR->findDeducedElementType(Ref->getPointerOperand()))
      return;
    KnownElemTy = Ref->getSourceElementType();
    Ops.push_back(std::make_pair(Ref->getPointerOperand(),
                                 GetElementPtrInst::getPointerOperandIndex()));
  } else if (auto *Ref = dyn_cast<LoadInst>(I)) {
    KnownElemTy = I->getType();
    if (isUntypedPointerTy(KnownElemTy))
      return;
    Type *PointeeTy = GR->findDeducedElementType(Ref->getPointerOperand());
    if (PointeeTy && !isUntypedPointerTy(PointeeTy))
      return;
    Ops.push_back(std::make_pair(Ref->getPointerOperand(),
                                 LoadInst::getPointerOperandIndex()));
  } else if (auto *Ref = dyn_cast<StoreInst>(I)) {
    if (!(KnownElemTy =
              reconstructType(Ref->getValueOperand(), false, IsPostprocessing)))
      return;
    Type *PointeeTy = GR->findDeducedElementType(Ref->getPointerOperand());
    if (PointeeTy && !isUntypedPointerTy(PointeeTy))
      return;
    Ops.push_back(std::make_pair(Ref->getPointerOperand(),
                                 StoreInst::getPointerOperandIndex()));
  } else if (auto *Ref = dyn_cast<AtomicCmpXchgInst>(I)) {
    KnownElemTy = getAtomicElemTy(GR, I, Ref->getPointerOperand());
    if (!KnownElemTy)
      return;
    Ops.push_back(std::make_pair(Ref->getPointerOperand(),
                                 AtomicCmpXchgInst::getPointerOperandIndex()));
  } else if (auto *Ref = dyn_cast<AtomicRMWInst>(I)) {
    KnownElemTy = getAtomicElemTy(GR, I, Ref->getPointerOperand());
    if (!KnownElemTy)
      return;
    Ops.push_back(std::make_pair(Ref->getPointerOperand(),
                                 AtomicRMWInst::getPointerOperandIndex()));
  } else if (auto *Ref = dyn_cast<SelectInst>(I)) {
    if (!isPointerTy(I->getType()) ||
        !(KnownElemTy = GR->findDeducedElementType(I)))
      return;
    Uncomplete = isTodoType(I);
    for (unsigned i = 0; i < Ref->getNumOperands(); i++) {
      Value *Op = Ref->getOperand(i);
      if (isPointerTy(Op->getType()))
        Ops.push_back(std::make_pair(Op, i));
    }
  } else if (auto *Ref = dyn_cast<ReturnInst>(I)) {
    if (!isPointerTy(CurrF->getReturnType()))
      return;
    Value *Op = Ref->getReturnValue();
    if (!Op)
      return;
    if (deduceOperandElementTypeFunctionRet(I, UncompleteRets, AskOps,
                                            IsPostprocessing, KnownElemTy, Op,
                                            CurrF))
      return;
    Uncomplete = isTodoType(CurrF);
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
      Uncomplete = isTodoType(Op0);
      Ops.push_back(std::make_pair(Op1, 1));
    } else if (ElemTy1) {
      KnownElemTy = ElemTy1;
      Uncomplete = isTodoType(Op1);
      Ops.push_back(std::make_pair(Op0, 0));
    }
  } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (!CI->isIndirectCall())
      deduceOperandElementTypeCalledFunction(CI, Ops, KnownElemTy);
    else if (HaveFunPtrs)
      deduceOperandElementTypeFunctionPointer(CI, Ops, KnownElemTy,
                                              IsPostprocessing);
  }

  // There is no enough info to deduce types or all is valid.
  if (!KnownElemTy || Ops.size() == 0)
    return;

  LLVMContext &Ctx = CurrF->getContext();
  IRBuilder<> B(Ctx);
  for (auto &OpIt : Ops) {
    Value *Op = OpIt.first;
    if (Op->use_empty())
      continue;
    if (AskOps && !AskOps->contains(Op))
      continue;
    Type *AskTy = nullptr;
    CallInst *AskCI = nullptr;
    if (IsPostprocessing && AskOps) {
      AskTy = GR->findDeducedElementType(Op);
      AskCI = GR->findAssignPtrTypeInstr(Op);
      assert(AskTy && AskCI);
    }
    Type *Ty = AskTy ? AskTy : GR->findDeducedElementType(Op);
    if (Ty == KnownElemTy)
      continue;
    Value *OpTyVal = PoisonValue::get(KnownElemTy);
    Type *OpTy = Op->getType();
    if (!Ty || AskTy || isUntypedPointerTy(Ty) || isTodoType(Op)) {
      Type *PrevElemTy = GR->findDeducedElementType(Op);
      GR->addDeducedElementType(Op, KnownElemTy);
      // check if KnownElemTy is complete
      if (!Uncomplete)
        eraseTodoType(Op);
      else if (!IsPostprocessing)
        insertTodoType(Op);
      // check if there is existing Intrinsic::spv_assign_ptr_type instruction
      CallInst *AssignCI = AskCI ? AskCI : GR->findAssignPtrTypeInstr(Op);
      if (AssignCI == nullptr) {
        Instruction *User = dyn_cast<Instruction>(Op->use_begin()->get());
        setInsertPointSkippingPhis(B, User ? User->getNextNode() : I);
        CallInst *CI =
            buildIntrWithMD(Intrinsic::spv_assign_ptr_type, {OpTy}, OpTyVal, Op,
                            {B.getInt32(getPointerAddressSpace(OpTy))}, B);
        GR->addAssignPtrTypeInstr(Op, CI);
      } else {
        updateAssignType(AssignCI, Op, OpTyVal);
        DenseSet<std::pair<Value *, Value *>> VisitedSubst{
            std::make_pair(I, Op)};
        propagateElemTypeRec(Op, KnownElemTy, PrevElemTy, VisitedSubst);
      }
    } else {
      eraseTodoType(Op);
      CallInst *PtrCastI =
          buildSpvPtrcast(I->getParent()->getParent(), Op, KnownElemTy);
      if (OpIt.second == std::numeric_limits<unsigned>::max())
        dyn_cast<CallInst>(I)->setCalledOperand(PtrCastI);
      else
        I->setOperand(OpIt.second, PtrCastI);
    }
  }
  TypeValidated.insert(I);
}

void SPIRVEmitIntrinsics::replaceMemInstrUses(Instruction *Old,
                                              Instruction *New,
                                              IRBuilder<> &B) {
  while (!Old->user_empty()) {
    auto *U = Old->user_back();
    if (isAssignTypeInstr(U)) {
      B.SetInsertPoint(U);
      SmallVector<Value *, 2> Args = {New, U->getOperand(1)};
      CallInst *AssignCI =
          B.CreateIntrinsic(Intrinsic::spv_assign_type, {New->getType()}, Args);
      GR->addAssignPtrTypeInstr(New, AssignCI);
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
  for (auto &I : instructions(CurrF))
    Worklist.push(&I);

  while (!Worklist.empty()) {
    Instruction *I = Worklist.front();
    bool BPrepared = false;
    Worklist.pop();

    for (auto &Op : I->operands()) {
      auto *AggrUndef = dyn_cast<UndefValue>(Op);
      if (!AggrUndef || !Op->getType()->isAggregateType())
        continue;

      if (!BPrepared) {
        setInsertPointSkippingPhis(B, I);
        BPrepared = true;
      }
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
  for (auto &I : instructions(CurrF))
    Worklist.push(&I);

  while (!Worklist.empty()) {
    auto *I = Worklist.front();
    bool IsPhi = isa<PHINode>(I), BPrepared = false;
    assert(I);
    bool KeepInst = false;
    for (const auto &Op : I->operands()) {
      Constant *AggrConst = nullptr;
      Type *ResTy = nullptr;
      if (auto *COp = dyn_cast<ConstantVector>(Op)) {
        AggrConst = cast<Constant>(COp);
        ResTy = COp->getType();
      } else if (auto *COp = dyn_cast<ConstantArray>(Op)) {
        AggrConst = cast<Constant>(COp);
        ResTy = B.getInt32Ty();
      } else if (auto *COp = dyn_cast<ConstantStruct>(Op)) {
        AggrConst = cast<Constant>(COp);
        ResTy = B.getInt32Ty();
      } else if (auto *COp = dyn_cast<ConstantDataArray>(Op)) {
        AggrConst = cast<Constant>(COp);
        ResTy = B.getInt32Ty();
      } else if (auto *COp = dyn_cast<ConstantAggregateZero>(Op)) {
        AggrConst = cast<Constant>(COp);
        ResTy = Op->getType()->isVectorTy() ? COp->getType() : B.getInt32Ty();
      }
      if (AggrConst) {
        SmallVector<Value *> Args;
        if (auto *COp = dyn_cast<ConstantDataSequential>(Op))
          for (unsigned i = 0; i < COp->getNumElements(); ++i)
            Args.push_back(COp->getElementAsConstant(i));
        else
          for (auto &COp : AggrConst->operands())
            Args.push_back(COp);
        if (!BPrepared) {
          IsPhi ? B.SetInsertPointPastAllocas(I->getParent()->getParent())
                : B.SetInsertPoint(I);
          BPrepared = true;
        }
        auto *CI =
            B.CreateIntrinsic(Intrinsic::spv_const_composite, {ResTy}, {Args});
        Worklist.push(CI);
        I->replaceUsesOfWith(Op, CI);
        KeepInst = true;
        AggrConsts[CI] = AggrConst;
        AggrConstTypes[CI] = deduceNestedTypeHelper(AggrConst, false);
      }
    }
    if (!KeepInst)
      Worklist.pop();
  }
}

static void createDecorationIntrinsic(Instruction *I, MDNode *Node,
                                      IRBuilder<> &B) {
  LLVMContext &Ctx = I->getContext();
  setInsertPointAfterDef(B, I);
  B.CreateIntrinsic(Intrinsic::spv_assign_decoration, {I->getType()},
                    {I, MetadataAsValue::get(Ctx, MDNode::get(Ctx, {Node}))});
}

static void createRoundingModeDecoration(Instruction *I,
                                         unsigned RoundingModeDeco,
                                         IRBuilder<> &B) {
  LLVMContext &Ctx = I->getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  MDNode *RoundingModeNode = MDNode::get(
      Ctx,
      {ConstantAsMetadata::get(
           ConstantInt::get(Int32Ty, SPIRV::Decoration::FPRoundingMode)),
       ConstantAsMetadata::get(ConstantInt::get(Int32Ty, RoundingModeDeco))});
  createDecorationIntrinsic(I, RoundingModeNode, B);
}

static void createSaturatedConversionDecoration(Instruction *I,
                                                IRBuilder<> &B) {
  LLVMContext &Ctx = I->getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  MDNode *SaturatedConversionNode =
      MDNode::get(Ctx, {ConstantAsMetadata::get(ConstantInt::get(
                           Int32Ty, SPIRV::Decoration::SaturatedConversion))});
  createDecorationIntrinsic(I, SaturatedConversionNode, B);
}

Instruction *SPIRVEmitIntrinsics::visitCallInst(CallInst &Call) {
  if (!Call.isInlineAsm())
    return &Call;

  const InlineAsm *IA = cast<InlineAsm>(Call.getCalledOperand());
  LLVMContext &Ctx = CurrF->getContext();

  Constant *TyC = UndefValue::get(IA->getFunctionType());
  MDString *ConstraintString = MDString::get(Ctx, IA->getConstraintString());
  SmallVector<Value *> Args = {
      buildMD(TyC),
      MetadataAsValue::get(Ctx, MDNode::get(Ctx, ConstraintString))};
  for (unsigned OpIdx = 0; OpIdx < Call.arg_size(); OpIdx++)
    Args.push_back(Call.getArgOperand(OpIdx));

  IRBuilder<> B(Call.getParent());
  B.SetInsertPoint(&Call);
  B.CreateIntrinsic(Intrinsic::spv_inline_asm, {}, {Args});
  return &Call;
}

// Use a tip about rounding mode to create a decoration.
void SPIRVEmitIntrinsics::useRoundingMode(ConstrainedFPIntrinsic *FPI,
                                          IRBuilder<> &B) {
  std::optional<RoundingMode> RM = FPI->getRoundingMode();
  if (!RM.has_value())
    return;
  unsigned RoundingModeDeco = std::numeric_limits<unsigned>::max();
  switch (RM.value()) {
  default:
    // ignore unknown rounding modes
    break;
  case RoundingMode::NearestTiesToEven:
    RoundingModeDeco = SPIRV::FPRoundingMode::FPRoundingMode::RTE;
    break;
  case RoundingMode::TowardNegative:
    RoundingModeDeco = SPIRV::FPRoundingMode::FPRoundingMode::RTN;
    break;
  case RoundingMode::TowardPositive:
    RoundingModeDeco = SPIRV::FPRoundingMode::FPRoundingMode::RTP;
    break;
  case RoundingMode::TowardZero:
    RoundingModeDeco = SPIRV::FPRoundingMode::FPRoundingMode::RTZ;
    break;
  case RoundingMode::Dynamic:
  case RoundingMode::NearestTiesToAway:
    // TODO: check if supported
    break;
  }
  if (RoundingModeDeco == std::numeric_limits<unsigned>::max())
    return;
  // Convert the tip about rounding mode into a decoration record.
  createRoundingModeDecoration(FPI, RoundingModeDeco, B);
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
  replaceAllUsesWith(&I, NewI);
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
  replaceAllUsesWithAndErase(B, &I, NewI);
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
    replaceAllUsesWith(&I, Source);
    I.eraseFromParent();
    return nullptr;
  }

  SmallVector<Type *, 2> Types = {I.getType(), Source->getType()};
  SmallVector<Value *> Args(I.op_begin(), I.op_end());
  auto *NewI = B.CreateIntrinsic(Intrinsic::spv_bitcast, {Types}, {Args});
  replaceAllUsesWithAndErase(B, &I, NewI);
  return NewI;
}

void SPIRVEmitIntrinsics::insertAssignPtrTypeTargetExt(
    TargetExtType *AssignedType, Value *V, IRBuilder<> &B) {
  Type *VTy = V->getType();

  // A couple of sanity checks.
  assert((isPointerTy(VTy)) && "Expect a pointer type!");
  if (Type *ElemTy = getPointeeType(VTy))
    if (ElemTy != AssignedType)
      report_fatal_error("Unexpected pointer element type!");

  CallInst *AssignCI = GR->findAssignPtrTypeInstr(V);
  if (!AssignCI) {
    buildAssignType(B, AssignedType, V);
    return;
  }

  Type *CurrentType =
      dyn_cast<ConstantAsMetadata>(
          cast<MetadataAsValue>(AssignCI->getOperand(1))->getMetadata())
          ->getType();
  if (CurrentType == AssignedType)
    return;

  // Builtin types cannot be redeclared or casted.
  if (CurrentType->isTargetExtTy())
    report_fatal_error("Type mismatch " + CurrentType->getTargetExtName() +
                           "/" + AssignedType->getTargetExtName() +
                           " for value " + V->getName(),
                       false);

  // Our previous guess about the type seems to be wrong, let's update
  // inferred type according to a new, more precise type information.
  updateAssignType(AssignCI, V, PoisonValue::get(AssignedType));
}

void SPIRVEmitIntrinsics::replacePointerOperandWithPtrCast(
    Instruction *I, Value *Pointer, Type *ExpectedElementType,
    unsigned OperandToReplace, IRBuilder<> &B) {
  TypeValidated.insert(I);

  // Do not emit spv_ptrcast if Pointer's element type is ExpectedElementType
  Type *PointerElemTy = deduceElementTypeHelper(Pointer, false);
  if (PointerElemTy == ExpectedElementType ||
      isEquivalentTypes(PointerElemTy, ExpectedElementType))
    return;

  setInsertPointSkippingPhis(B, I);
  Value *ExpectedElementVal = PoisonValue::get(ExpectedElementType);
  MetadataAsValue *VMD = buildMD(ExpectedElementVal);
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

  if (isa<Instruction>(Pointer) || isa<Argument>(Pointer)) {
    if (FirstPtrCastOrAssignPtrType) {
      // If this would be the first spv_ptrcast, do not emit spv_ptrcast and
      // emit spv_assign_ptr_type instead.
      buildAssignPtr(B, ExpectedElementType, Pointer);
      return;
    } else if (isTodoType(Pointer)) {
      eraseTodoType(Pointer);
      if (!isa<CallInst>(Pointer) && !isa<GetElementPtrInst>(Pointer)) {
        //  If this wouldn't be the first spv_ptrcast but existing type info is
        //  uncomplete, update spv_assign_ptr_type arguments.
        if (CallInst *AssignCI = GR->findAssignPtrTypeInstr(Pointer)) {
          Type *PrevElemTy = GR->findDeducedElementType(Pointer);
          assert(PrevElemTy);
          DenseSet<std::pair<Value *, Value *>> VisitedSubst{
              std::make_pair(I, Pointer)};
          updateAssignType(AssignCI, Pointer, ExpectedElementVal);
          propagateElemType(Pointer, PrevElemTy, VisitedSubst);
        } else {
          buildAssignPtr(B, ExpectedElementType, Pointer);
        }
        return;
      }
    }
  }

  // Emit spv_ptrcast
  SmallVector<Type *, 2> Types = {Pointer->getType(), Pointer->getType()};
  SmallVector<Value *, 2> Args = {Pointer, VMD, B.getInt32(AddressSpace)};
  auto *PtrCastI = B.CreateIntrinsic(Intrinsic::spv_ptrcast, {Types}, Args);
  I->setOperand(OperandToReplace, PtrCastI);
  // We need to set up a pointee type for the newly created spv_ptrcast.
  buildAssignPtr(B, ExpectedElementType, PtrCastI);
}

void SPIRVEmitIntrinsics::insertPtrCastOrAssignTypeInstr(Instruction *I,
                                                         IRBuilder<> &B) {
  // Handle basic instructions:
  StoreInst *SI = dyn_cast<StoreInst>(I);
  if (IsKernelArgInt8(CurrF, SI)) {
    replacePointerOperandWithPtrCast(
        I, SI->getValueOperand(), IntegerType::getInt8Ty(CurrF->getContext()),
        0, B);
  }
  if (SI) {
    Value *Op = SI->getValueOperand();
    Value *Pointer = SI->getPointerOperand();
    Type *OpTy = Op->getType();
    if (auto *OpI = dyn_cast<Instruction>(Op))
      OpTy = restoreMutatedType(GR, OpI, OpTy);
    if (OpTy == Op->getType())
      OpTy = deduceElementTypeByValueDeep(OpTy, Op, false);
    replacePointerOperandWithPtrCast(I, Pointer, OpTy, 1, B);
    return;
  }
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Value *Pointer = LI->getPointerOperand();
    Type *OpTy = LI->getType();
    if (auto *PtrTy = dyn_cast<PointerType>(OpTy)) {
      if (Type *ElemTy = GR->findDeducedElementType(LI)) {
        OpTy = getTypedPointerWrapper(ElemTy, PtrTy->getAddressSpace());
      } else {
        Type *NewOpTy = OpTy;
        OpTy = deduceElementTypeByValueDeep(OpTy, LI, false);
        if (OpTy == NewOpTy)
          insertTodoType(Pointer);
      }
    }
    replacePointerOperandWithPtrCast(I, Pointer, OpTy, 0, B);
    return;
  }
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
    Value *Pointer = GEPI->getPointerOperand();
    Type *OpTy = GEPI->getSourceElementType();
    replacePointerOperandWithPtrCast(I, Pointer, OpTy, 0, B);
    if (isNestedPointer(OpTy))
      insertTodoType(Pointer);
    return;
  }

  // TODO: review and merge with existing logics:
  // Handle calls to builtins (non-intrinsics):
  CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI || CI->isIndirectCall() || CI->isInlineAsm() ||
      !CI->getCalledFunction() || CI->getCalledFunction()->isIntrinsic())
    return;

  // collect information about formal parameter types
  std::string DemangledName =
      getOclOrSpirvBuiltinDemangledName(CI->getCalledFunction()->getName());
  Function *CalledF = CI->getCalledFunction();
  SmallVector<Type *, 4> CalledArgTys;
  bool HaveTypes = false;
  for (unsigned OpIdx = 0; OpIdx < CalledF->arg_size(); ++OpIdx) {
    Argument *CalledArg = CalledF->getArg(OpIdx);
    Type *ArgType = CalledArg->getType();
    if (!isPointerTy(ArgType)) {
      CalledArgTys.push_back(nullptr);
    } else if (Type *ArgTypeElem = getPointeeType(ArgType)) {
      CalledArgTys.push_back(ArgTypeElem);
      HaveTypes = true;
    } else {
      Type *ElemTy = GR->findDeducedElementType(CalledArg);
      if (!ElemTy && hasPointeeTypeAttr(CalledArg))
        ElemTy = getPointeeTypeByAttr(CalledArg);
      if (!ElemTy) {
        ElemTy = getPointeeTypeByCallInst(DemangledName, CalledF, OpIdx);
        if (ElemTy) {
          GR->addDeducedElementType(CalledArg, ElemTy);
        } else {
          for (User *U : CalledArg->users()) {
            if (Instruction *Inst = dyn_cast<Instruction>(U)) {
              if ((ElemTy = deduceElementTypeHelper(Inst, false)) != nullptr)
                break;
            }
          }
        }
      }
      HaveTypes |= ElemTy != nullptr;
      CalledArgTys.push_back(ElemTy);
    }
  }

  if (DemangledName.empty() && !HaveTypes)
    return;

  for (unsigned OpIdx = 0; OpIdx < CI->arg_size(); OpIdx++) {
    Value *ArgOperand = CI->getArgOperand(OpIdx);
    if (!isPointerTy(ArgOperand->getType()))
      continue;

    // Constants (nulls/undefs) are handled in insertAssignPtrTypeIntrs()
    if (!isa<Instruction>(ArgOperand) && !isa<Argument>(ArgOperand)) {
      // However, we may have assumptions about the formal argument's type and
      // may have a need to insert a ptr cast for the actual parameter of this
      // call.
      Argument *CalledArg = CalledF->getArg(OpIdx);
      if (!GR->findDeducedElementType(CalledArg))
        continue;
    }

    Type *ExpectedType =
        OpIdx < CalledArgTys.size() ? CalledArgTys[OpIdx] : nullptr;
    if (!ExpectedType && !DemangledName.empty())
      ExpectedType = SPIRV::parseBuiltinCallArgumentBaseType(
          DemangledName, OpIdx, I->getContext());
    if (!ExpectedType || ExpectedType->isVoidTy())
      continue;

    if (ExpectedType->isTargetExtTy() &&
        !isTypedPointerWrapper(cast<TargetExtType>(ExpectedType)))
      insertAssignPtrTypeTargetExt(cast<TargetExtType>(ExpectedType),
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
  replaceAllUsesWithAndErase(B, &I, NewI);
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
  replaceAllUsesWithAndErase(B, &I, NewI);
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
  if (I.getAggregateOperand()->getType()->isAggregateType())
    return &I;
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  SmallVector<Value *> Args;
  for (auto &Op : I.operands())
    Args.push_back(Op);
  for (auto &Op : I.indices())
    Args.push_back(B.getInt32(Op));
  auto *NewI =
      B.CreateIntrinsic(Intrinsic::spv_extractv, {I.getType()}, {Args});
  replaceAllUsesWithAndErase(B, &I, NewI);
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
      TLI->getLoadMemOperandFlags(I, CurrF->getDataLayout());
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
      TLI->getStoreMemOperandFlags(I, CurrF->getDataLayout());
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
      ArraySize
          ? B.CreateIntrinsic(Intrinsic::spv_alloca_array,
                              {PtrTy, ArraySize->getType()},
                              {ArraySize, B.getInt8(I.getAlign().value())})
          : B.CreateIntrinsic(Intrinsic::spv_alloca, {PtrTy},
                              {B.getInt8(I.getAlign().value())});
  replaceAllUsesWithAndErase(B, &I, NewI);
  return NewI;
}

Instruction *SPIRVEmitIntrinsics::visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
  assert(I.getType()->isAggregateType() && "Aggregate result is expected");
  IRBuilder<> B(I.getParent());
  B.SetInsertPoint(&I);
  SmallVector<Value *> Args;
  for (auto &Op : I.operands())
    Args.push_back(Op);
  Args.push_back(B.getInt32(
      static_cast<uint32_t>(getMemScope(I.getContext(), I.getSyncScopeID()))));
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
    deduceElementTypeHelper(&GV, false);
    Constant *Init = GV.getInitializer();
    Type *Ty = isAggrConstForceInt32(Init) ? B.getInt32Ty() : Init->getType();
    Constant *Const = isAggrConstForceInt32(Init) ? B.getInt32(1) : Init;
    auto *InitInst = B.CreateIntrinsic(Intrinsic::spv_init_global,
                                       {GV.getType(), Ty}, {&GV, Const});
    InitInst->setArgOperand(1, Init);
  }
  if ((!GV.hasInitializer() || isa<UndefValue>(GV.getInitializer())) &&
      GV.getNumUses() == 0)
    B.CreateIntrinsic(Intrinsic::spv_unref_global, GV.getType(), &GV);
}

// Return true, if we can't decide what is the pointee type now and will get
// back to the question later. Return false is spv_assign_ptr_type is not needed
// or can be inserted immediately.
bool SPIRVEmitIntrinsics::insertAssignPtrTypeIntrs(Instruction *I,
                                                   IRBuilder<> &B,
                                                   bool UnknownElemTypeI8) {
  reportFatalOnTokenType(I);
  if (!isPointerTy(I->getType()) || !requireAssignType(I))
    return false;

  setInsertPointAfterDef(B, I);
  if (Type *ElemTy = deduceElementType(I, UnknownElemTypeI8)) {
    buildAssignPtr(B, ElemTy, I);
    return false;
  }
  return true;
}

void SPIRVEmitIntrinsics::insertAssignTypeIntrs(Instruction *I,
                                                IRBuilder<> &B) {
  // TODO: extend the list of functions with known result types
  static StringMap<unsigned> ResTypeWellKnown = {
      {"async_work_group_copy", WellKnownTypes::Event},
      {"async_work_group_strided_copy", WellKnownTypes::Event},
      {"__spirv_GroupAsyncCopy", WellKnownTypes::Event}};

  reportFatalOnTokenType(I);

  bool IsKnown = false;
  if (auto *CI = dyn_cast<CallInst>(I)) {
    if (!CI->isIndirectCall() && !CI->isInlineAsm() &&
        CI->getCalledFunction() && !CI->getCalledFunction()->isIntrinsic()) {
      Function *CalledF = CI->getCalledFunction();
      std::string DemangledName =
          getOclOrSpirvBuiltinDemangledName(CalledF->getName());
      FPDecorationId DecorationId = FPDecorationId::NONE;
      if (DemangledName.length() > 0)
        DemangledName =
            SPIRV::lookupBuiltinNameHelper(DemangledName, &DecorationId);
      auto ResIt = ResTypeWellKnown.find(DemangledName);
      if (ResIt != ResTypeWellKnown.end()) {
        IsKnown = true;
        setInsertPointAfterDef(B, I);
        switch (ResIt->second) {
        case WellKnownTypes::Event:
          buildAssignType(B, TargetExtType::get(I->getContext(), "spirv.Event"),
                          I);
          break;
        }
      }
      // check if a floating rounding mode or saturation info is present
      switch (DecorationId) {
      default:
        break;
      case FPDecorationId::SAT:
        createSaturatedConversionDecoration(CI, B);
        break;
      case FPDecorationId::RTE:
        createRoundingModeDecoration(
            CI, SPIRV::FPRoundingMode::FPRoundingMode::RTE, B);
        break;
      case FPDecorationId::RTZ:
        createRoundingModeDecoration(
            CI, SPIRV::FPRoundingMode::FPRoundingMode::RTZ, B);
        break;
      case FPDecorationId::RTP:
        createRoundingModeDecoration(
            CI, SPIRV::FPRoundingMode::FPRoundingMode::RTP, B);
        break;
      case FPDecorationId::RTN:
        createRoundingModeDecoration(
            CI, SPIRV::FPRoundingMode::FPRoundingMode::RTN, B);
        break;
      }
    }
  }

  Type *Ty = I->getType();
  if (!IsKnown && !Ty->isVoidTy() && !isPointerTy(Ty) && requireAssignType(I)) {
    setInsertPointAfterDef(B, I);
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
    TypeToAssign = restoreMutatedType(GR, I, TypeToAssign);
    buildAssignType(B, TypeToAssign, I);
  }
  for (const auto &Op : I->operands()) {
    if (isa<ConstantPointerNull>(Op) || isa<UndefValue>(Op) ||
        // Check GetElementPtrConstantExpr case.
        (isa<ConstantExpr>(Op) && isa<GEPOperator>(Op))) {
      setInsertPointSkippingPhis(B, I);
      Type *OpTy = Op->getType();
      if (isa<UndefValue>(Op) && OpTy->isAggregateType()) {
        CallInst *AssignCI =
            buildIntrWithMD(Intrinsic::spv_assign_type, {B.getInt32Ty()}, Op,
                            UndefValue::get(B.getInt32Ty()), {}, B);
        GR->addAssignPtrTypeInstr(Op, AssignCI);
      } else if (!isa<Instruction>(Op)) {
        Type *OpTy = Op->getType();
        Type *OpTyElem = getPointeeType(OpTy);
        if (OpTyElem) {
          buildAssignPtr(B, OpTyElem, Op);
        } else if (isPointerTy(OpTy)) {
          Type *ElemTy = GR->findDeducedElementType(Op);
          buildAssignPtr(B, ElemTy ? ElemTy : deduceElementType(Op, true), Op);
        } else {
          CallInst *AssignCI = buildIntrWithMD(Intrinsic::spv_assign_type,
                                               {OpTy}, Op, Op, {}, B);
          GR->addAssignPtrTypeInstr(Op, AssignCI);
        }
      }
    }
  }
}

void SPIRVEmitIntrinsics::insertSpirvDecorations(Instruction *I,
                                                 IRBuilder<> &B) {
  if (MDNode *MD = I->getMetadata("spirv.Decorations")) {
    setInsertPointAfterDef(B, I);
    B.CreateIntrinsic(Intrinsic::spv_assign_decoration, {I->getType()},
                      {I, MetadataAsValue::get(I->getContext(), MD)});
  }
}

void SPIRVEmitIntrinsics::processInstrAfterVisit(Instruction *I,
                                                 IRBuilder<> &B) {
  auto *II = dyn_cast<IntrinsicInst>(I);
  bool IsConstComposite =
      II && II->getIntrinsicID() == Intrinsic::spv_const_composite;
  if (IsConstComposite && TrackConstants) {
    setInsertPointAfterDef(B, I);
    auto t = AggrConsts.find(I);
    assert(t != AggrConsts.end());
    auto *NewOp =
        buildIntrWithMD(Intrinsic::spv_track_constant,
                        {II->getType(), II->getType()}, t->second, I, {}, B);
    replaceAllUsesWith(I, NewOp, false);
    NewOp->setArgOperand(0, I);
  }
  bool IsPhi = isa<PHINode>(I), BPrepared = false;
  for (const auto &Op : I->operands()) {
    if (isa<PHINode>(I) || isa<SwitchInst>(I))
      TrackConstants = false;
    if ((isa<ConstantData>(Op) || isa<ConstantExpr>(Op)) && TrackConstants) {
      unsigned OpNo = Op.getOperandNo();
      if (II && ((II->getIntrinsicID() == Intrinsic::spv_gep && OpNo == 0) ||
                 (II->paramHasAttr(OpNo, Attribute::ImmArg))))
        continue;
      if (!BPrepared) {
        IsPhi ? B.SetInsertPointPastAllocas(I->getParent()->getParent())
              : B.SetInsertPoint(I);
        BPrepared = true;
      }
      Type *OpTy = Op->getType();
      Value *OpTyVal = Op;
      if (OpTy->isTargetExtTy())
        OpTyVal = PoisonValue::get(OpTy);
      CallInst *NewOp =
          buildIntrWithMD(Intrinsic::spv_track_constant,
                          {OpTy, OpTyVal->getType()}, Op, OpTyVal, {}, B);
      Type *OpElemTy = nullptr;
      if (!IsConstComposite && isPointerTy(OpTy) &&
          (OpElemTy = GR->findDeducedElementType(Op)) != nullptr &&
          OpElemTy != IntegerType::getInt8Ty(I->getContext())) {
        buildAssignPtr(B, IntegerType::getInt8Ty(I->getContext()), NewOp);
        SmallVector<Type *, 2> Types = {OpTy, OpTy};
        SmallVector<Value *, 2> Args = {
            NewOp, buildMD(PoisonValue::get(OpElemTy)),
            B.getInt32(getPointerAddressSpace(OpTy))};
        CallInst *PtrCasted =
            B.CreateIntrinsic(Intrinsic::spv_ptrcast, {Types}, Args);
        buildAssignPtr(B, OpElemTy, PtrCasted);
        NewOp = PtrCasted;
      }
      I->setOperand(OpNo, NewOp);
    }
  }
  emitAssignName(I, B);
}

Type *SPIRVEmitIntrinsics::deduceFunParamElementType(Function *F,
                                                     unsigned OpIdx) {
  std::unordered_set<Function *> FVisited;
  return deduceFunParamElementType(F, OpIdx, FVisited);
}

Type *SPIRVEmitIntrinsics::deduceFunParamElementType(
    Function *F, unsigned OpIdx, std::unordered_set<Function *> &FVisited) {
  // maybe a cycle
  if (!FVisited.insert(F).second)
    return nullptr;

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
    if (Type *Ty = deduceElementTypeHelper(OpArg, Visited, false))
      return Ty;
    // search in actual parameter's users
    for (User *OpU : OpArg->users()) {
      Instruction *Inst = dyn_cast<Instruction>(OpU);
      if (!Inst || Inst == CI)
        continue;
      Visited.clear();
      if (Type *Ty = deduceElementTypeHelper(Inst, Visited, false))
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

void SPIRVEmitIntrinsics::processParamTypesByFunHeader(Function *F,
                                                       IRBuilder<> &B) {
  B.SetInsertPointPastAllocas(F);
  for (unsigned OpIdx = 0; OpIdx < F->arg_size(); ++OpIdx) {
    Argument *Arg = F->getArg(OpIdx);
    if (!isUntypedPointerTy(Arg->getType()))
      continue;
    Type *ElemTy = GR->findDeducedElementType(Arg);
    if (ElemTy)
      continue;
    if (hasPointeeTypeAttr(Arg) &&
        (ElemTy = getPointeeTypeByAttr(Arg)) != nullptr) {
      buildAssignPtr(B, ElemTy, Arg);
      continue;
    }
    // search in function's call sites
    for (User *U : F->users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI || OpIdx >= CI->arg_size())
        continue;
      Value *OpArg = CI->getArgOperand(OpIdx);
      if (!isPointerTy(OpArg->getType()))
        continue;
      // maybe we already know operand's element type
      if ((ElemTy = GR->findDeducedElementType(OpArg)) != nullptr)
        break;
    }
    if (ElemTy) {
      buildAssignPtr(B, ElemTy, Arg);
      continue;
    }
    if (HaveFunPtrs) {
      for (User *U : Arg->users()) {
        CallInst *CI = dyn_cast<CallInst>(U);
        if (CI && !isa<IntrinsicInst>(CI) && CI->isIndirectCall() &&
            CI->getCalledOperand() == Arg &&
            CI->getParent()->getParent() == CurrF) {
          SmallVector<std::pair<Value *, unsigned>> Ops;
          deduceOperandElementTypeFunctionPointer(CI, Ops, ElemTy, false);
          if (ElemTy) {
            buildAssignPtr(B, ElemTy, Arg);
            break;
          }
        }
      }
    }
  }
}

void SPIRVEmitIntrinsics::processParamTypes(Function *F, IRBuilder<> &B) {
  B.SetInsertPointPastAllocas(F);
  for (unsigned OpIdx = 0; OpIdx < F->arg_size(); ++OpIdx) {
    Argument *Arg = F->getArg(OpIdx);
    if (!isUntypedPointerTy(Arg->getType()))
      continue;
    Type *ElemTy = GR->findDeducedElementType(Arg);
    if (!ElemTy && (ElemTy = deduceFunParamElementType(F, OpIdx)) != nullptr) {
      if (CallInst *AssignCI = GR->findAssignPtrTypeInstr(Arg)) {
        DenseSet<std::pair<Value *, Value *>> VisitedSubst;
        updateAssignType(AssignCI, Arg, PoisonValue::get(ElemTy));
        propagateElemType(Arg, IntegerType::getInt8Ty(F->getContext()),
                          VisitedSubst);
      } else {
        buildAssignPtr(B, ElemTy, Arg);
      }
    }
  }
}

static FunctionType *getFunctionPointerElemType(Function *F,
                                                SPIRVGlobalRegistry *GR) {
  FunctionType *FTy = F->getFunctionType();
  bool IsNewFTy = false;
  SmallVector<Type *, 4> ArgTys;
  for (Argument &Arg : F->args()) {
    Type *ArgTy = Arg.getType();
    if (ArgTy->isPointerTy())
      if (Type *ElemTy = GR->findDeducedElementType(&Arg)) {
        IsNewFTy = true;
        ArgTy = getTypedPointerWrapper(ElemTy, getPointerAddressSpace(ArgTy));
      }
    ArgTys.push_back(ArgTy);
  }
  return IsNewFTy
             ? FunctionType::get(FTy->getReturnType(), ArgTys, FTy->isVarArg())
             : FTy;
}

bool SPIRVEmitIntrinsics::processFunctionPointers(Module &M) {
  SmallVector<Function *> Worklist;
  for (auto &F : M) {
    if (F.isIntrinsic())
      continue;
    if (F.isDeclaration()) {
      for (User *U : F.users()) {
        CallInst *CI = dyn_cast<CallInst>(U);
        if (!CI || CI->getCalledFunction() != &F) {
          Worklist.push_back(&F);
          break;
        }
      }
    } else {
      if (F.user_empty())
        continue;
      Type *FPElemTy = GR->findDeducedElementType(&F);
      if (!FPElemTy)
        FPElemTy = getFunctionPointerElemType(&F, GR);
      for (User *U : F.users()) {
        IntrinsicInst *II = dyn_cast<IntrinsicInst>(U);
        if (!II || II->arg_size() != 3 || II->getOperand(0) != &F)
          continue;
        if (II->getIntrinsicID() == Intrinsic::spv_assign_ptr_type ||
            II->getIntrinsicID() == Intrinsic::spv_ptrcast) {
          updateAssignType(II, &F, PoisonValue::get(FPElemTy));
          break;
        }
      }
    }
  }
  if (Worklist.empty())
    return false;

  std::string ServiceFunName = SPIRV_BACKEND_SERVICE_FUN_NAME;
  if (!getVacantFunctionName(M, ServiceFunName))
    report_fatal_error(
        "cannot allocate a name for the internal service function");
  LLVMContext &Ctx = M.getContext();
  Function *SF =
      Function::Create(FunctionType::get(Type::getVoidTy(Ctx), {}, false),
                       GlobalValue::PrivateLinkage, ServiceFunName, M);
  SF->addFnAttr(SPIRV_BACKEND_SERVICE_FUN_NAME, "");
  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", SF);
  IRBuilder<> IRB(BB);

  for (Function *F : Worklist) {
    SmallVector<Value *> Args;
    for (const auto &Arg : F->args())
      Args.push_back(PoisonValue::get(Arg.getType()));
    IRB.CreateCall(F, Args);
  }
  IRB.CreateRetVoid();

  return true;
}

// Apply types parsed from demangled function declarations.
void SPIRVEmitIntrinsics::applyDemangledPtrArgTypes(IRBuilder<> &B) {
  for (auto It : FDeclPtrTys) {
    Function *F = It.first;
    for (auto *U : F->users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != F)
        continue;
      unsigned Sz = CI->arg_size();
      for (auto [Idx, ElemTy] : It.second) {
        if (Idx >= Sz)
          continue;
        Value *Param = CI->getArgOperand(Idx);
        if (GR->findDeducedElementType(Param) || isa<GlobalValue>(Param))
          continue;
        if (Argument *Arg = dyn_cast<Argument>(Param)) {
          if (!hasPointeeTypeAttr(Arg)) {
            B.SetInsertPointPastAllocas(Arg->getParent());
            B.SetCurrentDebugLocation(DebugLoc());
            buildAssignPtr(B, ElemTy, Arg);
          }
        } else if (isa<Instruction>(Param)) {
          GR->addDeducedElementType(Param, ElemTy);
          // insertAssignTypeIntrs() will complete buildAssignPtr()
        } else {
          B.SetInsertPoint(CI->getParent()
                               ->getParent()
                               ->getEntryBlock()
                               .getFirstNonPHIOrDbgOrAlloca());
          buildAssignPtr(B, ElemTy, Param);
        }
        CallInst *Ref = dyn_cast<CallInst>(Param);
        if (!Ref)
          continue;
        Function *RefF = Ref->getCalledFunction();
        if (!RefF || !isPointerTy(RefF->getReturnType()) ||
            GR->findDeducedElementType(RefF))
          continue;
        GR->addDeducedElementType(RefF, ElemTy);
        GR->addReturnType(
            RefF, TypedPointerType::get(
                      ElemTy, getPointerAddressSpace(RefF->getReturnType())));
      }
    }
  }
}

bool SPIRVEmitIntrinsics::runOnFunction(Function &Func) {
  if (Func.isDeclaration())
    return false;

  const SPIRVSubtarget &ST = TM->getSubtarget<SPIRVSubtarget>(Func);
  GR = ST.getSPIRVGlobalRegistry();
  InstrSet = ST.isOpenCLEnv() ? SPIRV::InstructionSet::OpenCL_std
                              : SPIRV::InstructionSet::GLSL_std_450;

  if (!CurrF)
    HaveFunPtrs =
        ST.canUseExtension(SPIRV::Extension::SPV_INTEL_function_pointers);

  CurrF = &Func;
  IRBuilder<> B(Func.getContext());
  AggrConsts.clear();
  AggrConstTypes.clear();
  AggrStores.clear();

  processParamTypesByFunHeader(CurrF, B);

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

  applyDemangledPtrArgTypes(B);

  // Pass forward: use operand to deduce instructions result.
  for (auto &I : Worklist) {
    // Don't emit intrinsincs for convergence intrinsics.
    if (isConvergenceIntrinsic(I))
      continue;

    bool Postpone = insertAssignPtrTypeIntrs(I, B, false);
    // if Postpone is true, we can't decide on pointee type yet
    insertAssignTypeIntrs(I, B);
    insertPtrCastOrAssignTypeInstr(I, B);
    insertSpirvDecorations(I, B);
    // if instruction requires a pointee type set, let's check if we know it
    // already, and force it to be i8 if not
    if (Postpone && !GR->findAssignPtrTypeInstr(I))
      insertAssignPtrTypeIntrs(I, B, true);

    if (auto *FPI = dyn_cast<ConstrainedFPIntrinsic>(I))
      useRoundingMode(FPI, B);
  }

  // Pass backward: use instructions results to specify/update/cast operands
  // where needed.
  SmallPtrSet<Instruction *, 4> UncompleteRets;
  for (auto &I : llvm::reverse(instructions(Func)))
    deduceOperandElementType(&I, &UncompleteRets);

  // Pass forward for PHIs only, their operands are not preceed the instruction
  // in meaning of `instructions(Func)`.
  for (BasicBlock &BB : Func)
    for (PHINode &Phi : BB.phis())
      if (isPointerTy(Phi.getType()))
        deduceOperandElementType(&Phi, nullptr);

  for (auto *I : Worklist) {
    TrackConstants = true;
    if (!I->getType()->isVoidTy() || isa<StoreInst>(I))
      setInsertPointAfterDef(B, I);
    // Visitors return either the original/newly created instruction for further
    // processing, nullptr otherwise.
    I = visit(*I);
    if (!I)
      continue;

    // Don't emit intrinsics for convergence operations.
    if (isConvergenceIntrinsic(I))
      continue;

    processInstrAfterVisit(I, B);
  }

  return true;
}

// Try to deduce a better type for pointers to untyped ptr.
bool SPIRVEmitIntrinsics::postprocessTypes(Module &M) {
  if (!GR || TodoTypeSz == 0)
    return false;

  unsigned SzTodo = TodoTypeSz;
  DenseMap<Value *, SmallPtrSet<Value *, 4>> ToProcess;
  for (auto [Op, Enabled] : TodoType) {
    // TODO: add isa<CallInst>(Op) to continue
    if (!Enabled || isa<GetElementPtrInst>(Op))
      continue;
    CallInst *AssignCI = GR->findAssignPtrTypeInstr(Op);
    Type *KnownTy = GR->findDeducedElementType(Op);
    if (!KnownTy || !AssignCI)
      continue;
    assert(Op == AssignCI->getArgOperand(0));
    // Try to improve the type deduced after all Functions are processed.
    if (auto *CI = dyn_cast<Instruction>(Op)) {
      CurrF = CI->getParent()->getParent();
      std::unordered_set<Value *> Visited;
      if (Type *ElemTy = deduceElementTypeHelper(Op, Visited, false, true)) {
        if (ElemTy != KnownTy) {
          DenseSet<std::pair<Value *, Value *>> VisitedSubst;
          propagateElemType(CI, ElemTy, VisitedSubst);
          eraseTodoType(Op);
          continue;
        }
      }
    }
    for (User *U : Op->users()) {
      Instruction *Inst = dyn_cast<Instruction>(U);
      if (Inst && !isa<IntrinsicInst>(Inst))
        ToProcess[Inst].insert(Op);
    }
  }
  if (TodoTypeSz == 0)
    return true;

  for (auto &F : M) {
    CurrF = &F;
    SmallPtrSet<Instruction *, 4> UncompleteRets;
    for (auto &I : llvm::reverse(instructions(F))) {
      auto It = ToProcess.find(&I);
      if (It == ToProcess.end())
        continue;
      It->second.remove_if([this](Value *V) { return !isTodoType(V); });
      if (It->second.size() == 0)
        continue;
      deduceOperandElementType(&I, &UncompleteRets, &It->second, true);
      if (TodoTypeSz == 0)
        return true;
    }
  }

  return SzTodo > TodoTypeSz;
}

// Parse and store argument types of function declarations where needed.
void SPIRVEmitIntrinsics::parseFunDeclarations(Module &M) {
  for (auto &F : M) {
    if (!F.isDeclaration() || F.isIntrinsic())
      continue;
    // get the demangled name
    std::string DemangledName = getOclOrSpirvBuiltinDemangledName(F.getName());
    if (DemangledName.empty())
      continue;
    // allow only OpGroupAsyncCopy use case at the moment
    auto [Grp, Opcode, ExtNo] =
        SPIRV::mapBuiltinToOpcode(DemangledName, InstrSet);
    if (Opcode != SPIRV::OpGroupAsyncCopy)
      continue;
    // find pointer arguments
    SmallVector<unsigned> Idxs;
    for (unsigned OpIdx = 0; OpIdx < F.arg_size(); ++OpIdx) {
      Argument *Arg = F.getArg(OpIdx);
      if (isPointerTy(Arg->getType()) && !hasPointeeTypeAttr(Arg))
        Idxs.push_back(OpIdx);
    }
    if (!Idxs.size())
      continue;
    // parse function arguments
    LLVMContext &Ctx = F.getContext();
    SmallVector<StringRef, 10> TypeStrs;
    SPIRV::parseBuiltinTypeStr(TypeStrs, DemangledName, Ctx);
    if (!TypeStrs.size())
      continue;
    // find type info for pointer arguments
    for (unsigned Idx : Idxs) {
      if (Idx >= TypeStrs.size())
        continue;
      if (Type *ElemTy =
              SPIRV::parseBuiltinCallArgumentType(TypeStrs[Idx].trim(), Ctx))
        if (TypedPointerType::isValidElementType(ElemTy) &&
            !ElemTy->isTargetExtTy())
          FDeclPtrTys[&F].push_back(std::make_pair(Idx, ElemTy));
    }
  }
}

bool SPIRVEmitIntrinsics::runOnModule(Module &M) {
  bool Changed = false;

  parseFunDeclarations(M);

  TodoType.clear();
  for (auto &F : M)
    Changed |= runOnFunction(F);

  // Specify function parameters after all functions were processed.
  for (auto &F : M) {
    // check if function parameter types are set
    CurrF = &F;
    if (!F.isDeclaration() && !F.isIntrinsic()) {
      IRBuilder<> B(F.getContext());
      processParamTypes(&F, B);
    }
  }

  CanTodoType = false;
  Changed |= postprocessTypes(M);

  if (HaveFunPtrs)
    Changed |= processFunctionPointers(M);

  return Changed;
}

ModulePass *llvm::createSPIRVEmitIntrinsicsPass(SPIRVTargetMachine *TM) {
  return new SPIRVEmitIntrinsics(TM);
}
