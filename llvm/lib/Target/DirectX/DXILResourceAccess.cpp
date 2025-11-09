//===- DXILResourceAccess.cpp - Resource access via load/store ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILResourceAccess.h"
#include "DirectX.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/User.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#define DEBUG_TYPE "dxil-resource-access"

using namespace llvm;

static Value *calculateGEPOffset(GetElementPtrInst *GEP, Value *PrevOffset,
                                 dxil::ResourceTypeInfo &RTI) {
  assert(!PrevOffset && "Non-constant GEP chains not handled yet");

  const DataLayout &DL = GEP->getDataLayout();

  uint64_t ScalarSize = 1;
  if (RTI.isTyped()) {
    Type *ContainedType = RTI.getHandleTy()->getTypeParameter(0);
    // We need the size of an element in bytes so that we can calculate the
    // offset in elements given a total offset in bytes.
    Type *ScalarType = ContainedType->getScalarType();
    ScalarSize = DL.getTypeSizeInBits(ScalarType) / 8;
  }

  APInt ConstantOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
  if (GEP->accumulateConstantOffset(DL, ConstantOffset)) {
    APInt Scaled = ConstantOffset.udiv(ScalarSize);
    return ConstantInt::get(Type::getInt32Ty(GEP->getContext()), Scaled);
  }

  auto IndexIt = GEP->idx_begin();
  assert(cast<ConstantInt>(IndexIt)->getZExtValue() == 0 &&
         "GEP is not indexing through pointer");
  ++IndexIt;
  Value *Offset = *IndexIt;
  assert(++IndexIt == GEP->idx_end() && "Too many indices in GEP");
  return Offset;
}

static void createTypedBufferStore(IntrinsicInst *II, StoreInst *SI,
                                   Value *Offset, dxil::ResourceTypeInfo &RTI) {
  IRBuilder<> Builder(SI);
  Type *ContainedType = RTI.getHandleTy()->getTypeParameter(0);
  Type *LoadType = StructType::get(ContainedType, Builder.getInt1Ty());

  Value *V = SI->getValueOperand();
  if (V->getType() == ContainedType) {
    // V is already the right type.
    assert(!Offset && "store of whole element has offset?");
  } else if (V->getType() == ContainedType->getScalarType()) {
    // We're storing a scalar, so we need to load the current value and only
    // replace the relevant part.
    auto *Load = Builder.CreateIntrinsic(
        LoadType, Intrinsic::dx_resource_load_typedbuffer,
        {II->getOperand(0), II->getOperand(1)});
    auto *Struct = Builder.CreateExtractValue(Load, {0});

    // If we have an offset from seeing a GEP earlier, use that. Otherwise, 0.
    if (!Offset)
      Offset = ConstantInt::get(Builder.getInt32Ty(), 0);
    V = Builder.CreateInsertElement(Struct, V, Offset);
  } else {
    llvm_unreachable("Store to typed resource has invalid type");
  }

  auto *Inst = Builder.CreateIntrinsic(
      Builder.getVoidTy(), Intrinsic::dx_resource_store_typedbuffer,
      {II->getOperand(0), II->getOperand(1), V});
  SI->replaceAllUsesWith(Inst);
}

static void createRawStore(IntrinsicInst *II, StoreInst *SI, Value *Offset) {
  IRBuilder<> Builder(SI);

  if (!Offset)
    Offset = ConstantInt::get(Builder.getInt32Ty(), 0);
  Value *V = SI->getValueOperand();
  // TODO: break up larger types
  auto *Inst = Builder.CreateIntrinsic(
      Builder.getVoidTy(), Intrinsic::dx_resource_store_rawbuffer,
      {II->getOperand(0), II->getOperand(1), Offset, V});
  SI->replaceAllUsesWith(Inst);
}

static void createStoreIntrinsic(IntrinsicInst *II, StoreInst *SI,
                                 Value *Offset, dxil::ResourceTypeInfo &RTI) {
  switch (RTI.getResourceKind()) {
  case dxil::ResourceKind::TypedBuffer:
    return createTypedBufferStore(II, SI, Offset, RTI);
  case dxil::ResourceKind::RawBuffer:
  case dxil::ResourceKind::StructuredBuffer:
    return createRawStore(II, SI, Offset);
  case dxil::ResourceKind::Texture1D:
  case dxil::ResourceKind::Texture2D:
  case dxil::ResourceKind::Texture2DMS:
  case dxil::ResourceKind::Texture3D:
  case dxil::ResourceKind::TextureCube:
  case dxil::ResourceKind::Texture1DArray:
  case dxil::ResourceKind::Texture2DArray:
  case dxil::ResourceKind::Texture2DMSArray:
  case dxil::ResourceKind::TextureCubeArray:
  case dxil::ResourceKind::FeedbackTexture2D:
  case dxil::ResourceKind::FeedbackTexture2DArray:
    reportFatalUsageError("DXIL Load not implemented yet");
    return;
  case dxil::ResourceKind::CBuffer:
  case dxil::ResourceKind::Sampler:
  case dxil::ResourceKind::TBuffer:
  case dxil::ResourceKind::RTAccelerationStructure:
  case dxil::ResourceKind::Invalid:
  case dxil::ResourceKind::NumEntries:
    llvm_unreachable("Invalid resource kind for store");
  }
  llvm_unreachable("Unhandled case in switch");
}

static void createTypedBufferLoad(IntrinsicInst *II, LoadInst *LI,
                                  Value *Offset, dxil::ResourceTypeInfo &RTI) {
  IRBuilder<> Builder(LI);
  Type *ContainedType = RTI.getHandleTy()->getTypeParameter(0);
  Type *LoadType = StructType::get(ContainedType, Builder.getInt1Ty());

  Value *V =
      Builder.CreateIntrinsic(LoadType, Intrinsic::dx_resource_load_typedbuffer,
                              {II->getOperand(0), II->getOperand(1)});
  V = Builder.CreateExtractValue(V, {0});

  if (Offset)
    V = Builder.CreateExtractElement(V, Offset);

  // If we loaded a <1 x ...> instead of a scalar (presumably to feed a
  // shufflevector), then make sure we're maintaining the resulting type.
  if (auto *VT = dyn_cast<FixedVectorType>(LI->getType()))
    if (VT->getNumElements() == 1 && !isa<FixedVectorType>(V->getType()))
      V = Builder.CreateInsertElement(PoisonValue::get(VT), V,
                                      Builder.getInt32(0));

  LI->replaceAllUsesWith(V);
}

static void createRawLoad(IntrinsicInst *II, LoadInst *LI, Value *Offset) {
  IRBuilder<> Builder(LI);
  // TODO: break up larger types
  Type *LoadType = StructType::get(LI->getType(), Builder.getInt1Ty());
  if (!Offset)
    Offset = ConstantInt::get(Builder.getInt32Ty(), 0);
  Value *V =
      Builder.CreateIntrinsic(LoadType, Intrinsic::dx_resource_load_rawbuffer,
                              {II->getOperand(0), II->getOperand(1), Offset});
  V = Builder.CreateExtractValue(V, {0});

  LI->replaceAllUsesWith(V);
}

static void createLoadIntrinsic(IntrinsicInst *II, LoadInst *LI, Value *Offset,
                                dxil::ResourceTypeInfo &RTI) {
  switch (RTI.getResourceKind()) {
  case dxil::ResourceKind::TypedBuffer:
    return createTypedBufferLoad(II, LI, Offset, RTI);
  case dxil::ResourceKind::RawBuffer:
  case dxil::ResourceKind::StructuredBuffer:
    return createRawLoad(II, LI, Offset);
  case dxil::ResourceKind::Texture1D:
  case dxil::ResourceKind::Texture2D:
  case dxil::ResourceKind::Texture2DMS:
  case dxil::ResourceKind::Texture3D:
  case dxil::ResourceKind::TextureCube:
  case dxil::ResourceKind::Texture1DArray:
  case dxil::ResourceKind::Texture2DArray:
  case dxil::ResourceKind::Texture2DMSArray:
  case dxil::ResourceKind::TextureCubeArray:
  case dxil::ResourceKind::FeedbackTexture2D:
  case dxil::ResourceKind::FeedbackTexture2DArray:
  case dxil::ResourceKind::CBuffer:
  case dxil::ResourceKind::TBuffer:
    // TODO: handle these
    return;
  case dxil::ResourceKind::Sampler:
  case dxil::ResourceKind::RTAccelerationStructure:
  case dxil::ResourceKind::Invalid:
  case dxil::ResourceKind::NumEntries:
    llvm_unreachable("Invalid resource kind for load");
  }
  llvm_unreachable("Unhandled case in switch");
}

static SmallVector<Instruction *> collectBlockUseDef(Instruction *Start) {
  SmallPtrSet<Instruction *, 32> Visited;
  SmallVector<Instruction *, 32> Worklist;
  SmallVector<Instruction *> Out;
  auto *BB = Start->getParent();

  // Seed with direct users in this block.
  for (User *U : Start->users()) {
    if (auto *I = dyn_cast<Instruction>(U)) {
      if (I->getParent() == BB)
        Worklist.push_back(I);
    }
  }

  // BFS over transitive users, constrained to the same block.
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!Visited.insert(I).second)
      continue;
    Out.push_back(I);

    for (User *U : I->users()) {
      if (auto *J = dyn_cast<Instruction>(U)) {
        if (J->getParent() == BB)
          Worklist.push_back(J);
      }
    }
    for (Use &V : I->operands()) {
      if (auto *J = dyn_cast<Instruction>(V)) {
        if (J->getParent() == BB && V != Start)
          Worklist.push_back(J);
      }
    }
  }

  // Order results in program order.
  DenseMap<const Instruction *, unsigned> Ord;
  unsigned Idx = 0;
  for (Instruction &I : *BB)
    Ord[&I] = Idx++;

  llvm::sort(Out, [&](Instruction *A, Instruction *B) {
    return Ord.lookup(A) < Ord.lookup(B);
  });

  return Out;
}

static void phiNodeRemapHelper(PHINode *Phi, BasicBlock *BB,
                               IRBuilder<> &Builder,
                               SmallVector<Instruction *> &UsesInBlock) {

  ValueToValueMapTy VMap;
  Value *Val = Phi->getIncomingValueForBlock(BB);
  VMap[Phi] = Val;
  Builder.SetInsertPoint(&BB->back());
  for (Instruction *I : UsesInBlock) {
    // don't clone over the Phi just remap them
    if (auto *PhiNested = dyn_cast<PHINode>(I)) {
      VMap[PhiNested] = PhiNested->getIncomingValueForBlock(BB);
      continue;
    }
    Instruction *Clone = I->clone();
    RemapInstruction(Clone, VMap,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
    Builder.Insert(Clone);
    VMap[I] = Clone;
  }
}

static void phiNodeReplacement(IntrinsicInst *II,
                               SmallVectorImpl<Instruction *> &PrevBBDeadInsts,
                               SetVector<BasicBlock *> &DeadBB) {
  SmallVector<Instruction *> CurrBBDeadInsts;
  for (User *U : II->users()) {
    auto *Phi = dyn_cast<PHINode>(U);
    if (!Phi)
      continue;

    IRBuilder<> Builder(Phi);
    SmallVector<Instruction *> UsesInBlock = collectBlockUseDef(Phi);
    bool HasReturnUse = isa<ReturnInst>(UsesInBlock.back());

    for (unsigned I = 0, E = Phi->getNumIncomingValues(); I < E; I++) {
      auto *CurrIncomingBB = Phi->getIncomingBlock(I);
      phiNodeRemapHelper(Phi, CurrIncomingBB, Builder, UsesInBlock);
      if (HasReturnUse)
        PrevBBDeadInsts.push_back(&CurrIncomingBB->back());
    }

    CurrBBDeadInsts.push_back(Phi);

    for (Instruction *I : UsesInBlock) {
      CurrBBDeadInsts.push_back(I);
    }
    if (HasReturnUse) {
      BasicBlock *PhiBB = Phi->getParent();
      DeadBB.insert(PhiBB);
    }
  }
  // Traverse the now-dead instructions in RPO and remove them.
  for (Instruction *Dead : llvm::reverse(CurrBBDeadInsts))
    Dead->eraseFromParent();
  CurrBBDeadInsts.clear();
}

static void replaceAccess(IntrinsicInst *II, dxil::ResourceTypeInfo &RTI) {
  // Process users keeping track of indexing accumulated from GEPs.
  struct AccessAndOffset {
    User *Access;
    Value *Offset;
  };
  SmallVector<AccessAndOffset> Worklist;
  for (User *U : II->users())
    Worklist.push_back({U, nullptr});

  SmallVector<Instruction *> DeadInsts;
  while (!Worklist.empty()) {
    AccessAndOffset Current = Worklist.back();
    Worklist.pop_back();

    if (auto *GEP = dyn_cast<GetElementPtrInst>(Current.Access)) {
      IRBuilder<> Builder(GEP);

      Value *Offset = calculateGEPOffset(GEP, Current.Offset, RTI);
      for (User *U : GEP->users())
        Worklist.push_back({U, Offset});
      DeadInsts.push_back(GEP);

    } else if (auto *SI = dyn_cast<StoreInst>(Current.Access)) {
      assert(SI->getValueOperand() != II && "Pointer escaped!");
      createStoreIntrinsic(II, SI, Current.Offset, RTI);
      DeadInsts.push_back(SI);

    } else if (auto *LI = dyn_cast<LoadInst>(Current.Access)) {
      createLoadIntrinsic(II, LI, Current.Offset, RTI);
      DeadInsts.push_back(LI);
    } else
      llvm_unreachable("Unhandled instruction - pointer escaped?");
  }

  // Traverse the now-dead instructions in RPO and remove them.
  for (Instruction *Dead : llvm::reverse(DeadInsts))
    Dead->eraseFromParent();
  II->eraseFromParent();
}

static bool transformResourcePointers(Function &F, DXILResourceTypeMap &DRTM) {
  SmallVector<std::pair<IntrinsicInst *, dxil::ResourceTypeInfo>> Resources;
  SetVector<BasicBlock *> DeadBB;
  SmallVector<Instruction *> PrevBBDeadInsts;
  for (BasicBlock &BB : make_early_inc_range(F)) {
    for (Instruction &I : make_early_inc_range(BB))
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        if (II->getIntrinsicID() == Intrinsic::dx_resource_getpointer)
          phiNodeReplacement(II, PrevBBDeadInsts, DeadBB);

    for (Instruction &I : BB)
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        if (II->getIntrinsicID() == Intrinsic::dx_resource_getpointer) {
          auto *HandleTy = cast<TargetExtType>(II->getArgOperand(0)->getType());
          Resources.emplace_back(II, DRTM[HandleTy]);
        }
  }
  for (auto *Dead : PrevBBDeadInsts)
    Dead->eraseFromParent();
  PrevBBDeadInsts.clear();
  for (auto *Dead : DeadBB)
    Dead->eraseFromParent();
  DeadBB.clear();

  for (auto &[II, RI] : Resources)
    replaceAccess(II, RI);

  return !Resources.empty();
}

PreservedAnalyses DXILResourceAccess::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  auto &MAMProxy = FAM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  DXILResourceTypeMap *DRTM =
      MAMProxy.getCachedResult<DXILResourceTypeAnalysis>(*F.getParent());
  assert(DRTM && "DXILResourceTypeAnalysis must be available");

  bool MadeChanges = transformResourcePointers(F, *DRTM);
  if (!MadeChanges)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<DXILResourceTypeAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

namespace {
class DXILResourceAccessLegacy : public FunctionPass {
public:
  bool runOnFunction(Function &F) override {
    DXILResourceTypeMap &DRTM =
        getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();
    return transformResourcePointers(F, DRTM);
  }
  StringRef getPassName() const override { return "DXIL Resource Access"; }
  DXILResourceAccessLegacy() : FunctionPass(ID) {}

  static char ID; // Pass identification.
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<DXILResourceTypeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }
};
char DXILResourceAccessLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DXILResourceAccessLegacy, DEBUG_TYPE,
                      "DXIL Resource Access", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_END(DXILResourceAccessLegacy, DEBUG_TYPE,
                    "DXIL Resource Access", false, false)

FunctionPass *llvm::createDXILResourceAccessLegacyPass() {
  return new DXILResourceAccessLegacy();
}
