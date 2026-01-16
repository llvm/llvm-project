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
#include "llvm/Frontend/HLSL/HLSLResource.h"
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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#define DEBUG_TYPE "dxil-resource-access"

using namespace llvm;

static Value *traverseGEPOffsets(const DataLayout &DL, IRBuilder<> &Builder,
                                 Value *Ptr, uint64_t AccessSize) {
  Value *Offset = nullptr;

  while (Ptr) {
    if (auto *II = dyn_cast<IntrinsicInst>(Ptr)) {
      assert(II->getIntrinsicID() == Intrinsic::dx_resource_getpointer &&
             "Resource access through unexpected intrinsic");
      return Offset ? Offset : ConstantInt::get(Builder.getInt32Ty(), 0);
    }

    auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
    assert(GEP && "Resource access through unexpected instruction");

    unsigned NumIndices = GEP->getNumIndices();
    uint64_t IndexScale = DL.getTypeAllocSize(GEP->getSourceElementType());
    APInt ConstantOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
    Value *GEPOffset;
    if (GEP->accumulateConstantOffset(DL, ConstantOffset)) {
      // We have a constant offset (in bytes).
      GEPOffset =
          ConstantInt::get(DL.getIndexType(GEP->getType()), ConstantOffset);
      IndexScale = 1;
    } else if (NumIndices == 1) {
      // If we have a single index we're indexing into a top level array. This
      // generally only happens with cbuffers.
      GEPOffset = *GEP->idx_begin();
    } else if (NumIndices == 2) {
      // If we have two indices, this should be an access through a pointer.
      auto IndexIt = GEP->idx_begin();
      assert(cast<ConstantInt>(IndexIt)->getZExtValue() == 0 &&
             "GEP is not indexing through pointer");
      GEPOffset = *(++IndexIt);
    } else
      llvm_unreachable("Unhandled GEP structure for resource access");

    uint64_t ElemSize = AccessSize;
    if (!(IndexScale % ElemSize)) {
      // If our scale is an exact multiple of the access size, adjust the
      // scaling to avoid an unnecessary division.
      IndexScale /= ElemSize;
      ElemSize = 1;
    }
    if (IndexScale != 1)
      GEPOffset = Builder.CreateMul(
          GEPOffset, ConstantInt::get(Builder.getInt32Ty(), IndexScale));
    if (ElemSize != 1)
      GEPOffset = Builder.CreateUDiv(
          GEPOffset, ConstantInt::get(Builder.getInt32Ty(), ElemSize));

    Offset = Offset ? Builder.CreateAdd(Offset, GEPOffset) : GEPOffset;
    Ptr = GEP->getPointerOperand();
  }

  llvm_unreachable("GEP of null pointer?");
}

static void createTypedBufferStore(IntrinsicInst *II, StoreInst *SI,
                                   dxil::ResourceTypeInfo &RTI) {
  const DataLayout &DL = SI->getDataLayout();
  IRBuilder<> Builder(SI);
  Type *ContainedType = RTI.getHandleTy()->getTypeParameter(0);
  Type *ScalarType = ContainedType->getScalarType();
  Type *LoadType = StructType::get(ContainedType, Builder.getInt1Ty());

  Value *V = SI->getValueOperand();
  if (V->getType() == ContainedType) {
    // V is already the right type.
    assert(SI->getPointerOperand() == II &&
           "Store of whole element has mismatched address to store to");
  } else if (V->getType() == ScalarType) {
    // We're storing a scalar, so we need to load the current value and only
    // replace the relevant part.
    auto *Load = Builder.CreateIntrinsic(
        LoadType, Intrinsic::dx_resource_load_typedbuffer,
        {II->getOperand(0), II->getOperand(1)});
    auto *Struct = Builder.CreateExtractValue(Load, {0});

    uint64_t AccessSize = DL.getTypeSizeInBits(ScalarType) / 8;
    Value *Offset =
        traverseGEPOffsets(DL, Builder, SI->getPointerOperand(), AccessSize);
    V = Builder.CreateInsertElement(Struct, V, Offset);
  } else {
    llvm_unreachable("Store to typed resource has invalid type");
  }

  auto *Inst = Builder.CreateIntrinsic(
      Builder.getVoidTy(), Intrinsic::dx_resource_store_typedbuffer,
      {II->getOperand(0), II->getOperand(1), V});
  SI->replaceAllUsesWith(Inst);
}

static void createRawStore(IntrinsicInst *II, StoreInst *SI,
                           dxil::ResourceTypeInfo &RTI) {
  const DataLayout &DL = SI->getDataLayout();
  IRBuilder<> Builder(SI);

  Value *V = SI->getValueOperand();
  assert(!V->getType()->isAggregateType() &&
         "Resource store should be scalar or vector type");

  Value *Index = II->getOperand(1);
  // The offset for the rawbuffer load and store ops is always in bytes.
  uint64_t AccessSize = 1;
  Value *Offset =
      traverseGEPOffsets(DL, Builder, SI->getPointerOperand(), AccessSize);

  // For raw buffer (ie, HLSL's ByteAddressBuffer), we need to fold the access
  // entirely into the index.
  if (!RTI.isStruct()) {
    auto *ConstantOffset = dyn_cast<ConstantInt>(Offset);
    if (!ConstantOffset || !ConstantOffset->isZero())
      Index = Builder.CreateAdd(Index, Offset);
    Offset = llvm::PoisonValue::get(Builder.getInt32Ty());
  }

  auto *Inst = Builder.CreateIntrinsic(Builder.getVoidTy(),
                                       Intrinsic::dx_resource_store_rawbuffer,
                                       {II->getOperand(0), Index, Offset, V});
  SI->replaceAllUsesWith(Inst);
}

static void createStoreIntrinsic(IntrinsicInst *II, StoreInst *SI,
                                 dxil::ResourceTypeInfo &RTI) {
  switch (RTI.getResourceKind()) {
  case dxil::ResourceKind::TypedBuffer:
    return createTypedBufferStore(II, SI, RTI);
  case dxil::ResourceKind::RawBuffer:
  case dxil::ResourceKind::StructuredBuffer:
    return createRawStore(II, SI, RTI);
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
                                  dxil::ResourceTypeInfo &RTI) {
  const DataLayout &DL = LI->getDataLayout();
  IRBuilder<> Builder(LI);
  Type *ContainedType = RTI.getHandleTy()->getTypeParameter(0);
  Type *LoadType = StructType::get(ContainedType, Builder.getInt1Ty());

  Value *V =
      Builder.CreateIntrinsic(LoadType, Intrinsic::dx_resource_load_typedbuffer,
                              {II->getOperand(0), II->getOperand(1)});
  V = Builder.CreateExtractValue(V, {0});

  Type *ScalarType = ContainedType->getScalarType();
  uint64_t AccessSize = DL.getTypeSizeInBits(ScalarType) / 8;
  Value *Offset =
      traverseGEPOffsets(DL, Builder, LI->getPointerOperand(), AccessSize);
  auto *ConstantOffset = dyn_cast<ConstantInt>(Offset);
  if (!ConstantOffset || !ConstantOffset->isZero())
    V = Builder.CreateExtractElement(V, Offset);

  // If we loaded a <1 x ...> instead of a scalar (presumably to feed a
  // shufflevector), then make sure we're maintaining the resulting type.
  if (auto *VT = dyn_cast<FixedVectorType>(LI->getType()))
    if (VT->getNumElements() == 1 && !isa<FixedVectorType>(V->getType()))
      V = Builder.CreateInsertElement(PoisonValue::get(VT), V,
                                      Builder.getInt32(0));

  LI->replaceAllUsesWith(V);
}

static void createRawLoad(IntrinsicInst *II, LoadInst *LI,
                          dxil::ResourceTypeInfo &RTI) {
  const DataLayout &DL = LI->getDataLayout();
  IRBuilder<> Builder(LI);

  Type *LoadType = StructType::get(LI->getType(), Builder.getInt1Ty());
  assert(!LI->getType()->isAggregateType() &&
         "Resource load should be scalar or vector type");

  Value *Index = II->getOperand(1);
  // The offset for the rawbuffer load and store ops is always in bytes.
  uint64_t AccessSize = 1;
  Value *Offset =
      traverseGEPOffsets(DL, Builder, LI->getPointerOperand(), AccessSize);

  // For raw buffer (ie, HLSL's ByteAddressBuffer), we need to fold the access
  // entirely into the index.
  if (!RTI.isStruct()) {
    auto *ConstantOffset = dyn_cast<ConstantInt>(Offset);
    if (!ConstantOffset || !ConstantOffset->isZero())
      Index = Builder.CreateAdd(Index, Offset);
    Offset = llvm::PoisonValue::get(Builder.getInt32Ty());
  }

  Value *V =
      Builder.CreateIntrinsic(LoadType, Intrinsic::dx_resource_load_rawbuffer,
                              {II->getOperand(0), Index, Offset});
  V = Builder.CreateExtractValue(V, {0});

  LI->replaceAllUsesWith(V);
}

namespace {
/// Helper for building a `load.cbufferrow` intrinsic given a simple type.
struct CBufferRowIntrin {
  Intrinsic::ID IID;
  Type *RetTy;
  unsigned int EltSize;
  unsigned int NumElts;

  CBufferRowIntrin(const DataLayout &DL, Type *Ty) {
    assert(Ty == Ty->getScalarType() && "Expected scalar type");

    switch (DL.getTypeSizeInBits(Ty)) {
    case 16:
      IID = Intrinsic::dx_resource_load_cbufferrow_8;
      RetTy = StructType::get(Ty, Ty, Ty, Ty, Ty, Ty, Ty, Ty);
      EltSize = 2;
      NumElts = 8;
      break;
    case 32:
      IID = Intrinsic::dx_resource_load_cbufferrow_4;
      RetTy = StructType::get(Ty, Ty, Ty, Ty);
      EltSize = 4;
      NumElts = 4;
      break;
    case 64:
      IID = Intrinsic::dx_resource_load_cbufferrow_2;
      RetTy = StructType::get(Ty, Ty);
      EltSize = 8;
      NumElts = 2;
      break;
    default:
      llvm_unreachable("Only 16, 32, and 64 bit types supported");
    }
  }
};
} // namespace

static void createCBufferLoad(IntrinsicInst *II, LoadInst *LI,
                              dxil::ResourceTypeInfo &RTI) {
  const DataLayout &DL = LI->getDataLayout();

  Type *Ty = LI->getType();
  assert(!isa<StructType>(Ty) && "Structs not handled yet");
  CBufferRowIntrin Intrin(DL, Ty->getScalarType());

  StringRef Name = LI->getName();
  Value *Handle = II->getOperand(0);

  IRBuilder<> Builder(LI);

  ConstantInt *GlobalOffset = dyn_cast<ConstantInt>(II->getOperand(1));
  assert(GlobalOffset && "CBuffer getpointer index must be constant");

  uint64_t GlobalOffsetVal = GlobalOffset->getZExtValue();
  Value *CurrentRow = ConstantInt::get(
      Builder.getInt32Ty(), GlobalOffsetVal / hlsl::CBufferRowSizeInBytes);
  unsigned int CurrentIndex =
      (GlobalOffsetVal % hlsl::CBufferRowSizeInBytes) / Intrin.EltSize;

  // Every object in a cbuffer either fits in a row or is aligned to a row. This
  // means that only the very last pointer access can point into a row.
  auto *LastGEP = dyn_cast<GEPOperator>(LI->getPointerOperand());
  if (!LastGEP) {
    // If we don't have a GEP at all we're just accessing the resource through
    // the result of getpointer directly.
    assert(LI->getPointerOperand() == II &&
           "Unexpected indirect access to resource without GEP");
  } else {
    Value *GEPOffset = traverseGEPOffsets(
        DL, Builder, LastGEP->getPointerOperand(), hlsl::CBufferRowSizeInBytes);
    CurrentRow = Builder.CreateAdd(GEPOffset, CurrentRow);

    APInt ConstantOffset(DL.getIndexTypeSizeInBits(LastGEP->getType()), 0);
    if (LastGEP->accumulateConstantOffset(DL, ConstantOffset)) {
      APInt Remainder(DL.getIndexTypeSizeInBits(LastGEP->getType()),
                      hlsl::CBufferRowSizeInBytes);
      APInt::udivrem(ConstantOffset, Remainder, ConstantOffset, Remainder);
      CurrentRow = Builder.CreateAdd(
          CurrentRow, ConstantInt::get(Builder.getInt32Ty(), ConstantOffset));
      CurrentIndex += Remainder.udiv(Intrin.EltSize).getZExtValue();
    } else {
      assert(LastGEP->getNumIndices() == 1 &&
             "Last GEP of cbuffer access is not array or struct access");
      // We assume a non-constant access will be row-aligned. This is safe
      // because arrays and structs are always row aligned, and accesses to
      // vector elements will show up as a load of the vector followed by an
      // extractelement.
      CurrentRow = cast<ConstantInt>(CurrentRow)->isZero()
                       ? *LastGEP->idx_begin()
                       : Builder.CreateAdd(CurrentRow, *LastGEP->idx_begin());
      CurrentIndex = 0;
    }
  }

  auto *CBufLoad = Builder.CreateIntrinsic(
      Intrin.RetTy, Intrin.IID, {Handle, CurrentRow}, nullptr, Name + ".load");
  auto *Elt =
      Builder.CreateExtractValue(CBufLoad, {CurrentIndex++}, Name + ".extract");

  // At this point we've loaded the first scalar of our result, but our original
  // type may have been a vector.
  unsigned int Remaining =
      ((DL.getTypeSizeInBits(Ty) / 8) / Intrin.EltSize) - 1;
  if (Remaining == 0) {
    // We only have a single element, so we're done.
    Value *Result = Elt;

    // However, if we loaded a <1 x T>, then we need to adjust the type.
    if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
      assert(VT->getNumElements() == 1 && "Can't have multiple elements here");
      Result = Builder.CreateInsertElement(PoisonValue::get(VT), Result,
                                           Builder.getInt32(0), Name);
    }
    LI->replaceAllUsesWith(Result);
    return;
  }

  // Walk each element and extract it, wrapping to new rows as needed.
  SmallVector<Value *> Extracts{Elt};
  while (Remaining--) {
    CurrentIndex %= Intrin.NumElts;

    if (CurrentIndex == 0) {
      CurrentRow = Builder.CreateAdd(CurrentRow,
                                     ConstantInt::get(Builder.getInt32Ty(), 1));
      CBufLoad = Builder.CreateIntrinsic(Intrin.RetTy, Intrin.IID,
                                         {Handle, CurrentRow}, nullptr,
                                         Name + ".load");
    }

    Extracts.push_back(Builder.CreateExtractValue(CBufLoad, {CurrentIndex++},
                                                  Name + ".extract"));
  }

  // Finally, we build up the original loaded value.
  Value *Result = PoisonValue::get(Ty);
  for (int I = 0, E = Extracts.size(); I < E; ++I)
    Result = Builder.CreateInsertElement(
        Result, Extracts[I], Builder.getInt32(I), Name + formatv(".upto{}", I));
  LI->replaceAllUsesWith(Result);
}

static void createLoadIntrinsic(IntrinsicInst *II, LoadInst *LI,
                                dxil::ResourceTypeInfo &RTI) {
  switch (RTI.getResourceKind()) {
  case dxil::ResourceKind::TypedBuffer:
    return createTypedBufferLoad(II, LI, RTI);
  case dxil::ResourceKind::RawBuffer:
  case dxil::ResourceKind::StructuredBuffer:
    return createRawLoad(II, LI, RTI);
  case dxil::ResourceKind::CBuffer:
    return createCBufferLoad(II, LI, RTI);
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
  case dxil::ResourceKind::TBuffer:
    reportFatalUsageError("Load not yet implemented for resource type");
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
  SmallVector<User *> Worklist;
  for (User *U : II->users())
    Worklist.push_back(U);

  SmallVector<Instruction *> DeadInsts;
  while (!Worklist.empty()) {
    User *U = Worklist.back();
    Worklist.pop_back();

    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      for (User *U : GEP->users())
        Worklist.push_back(U);
      DeadInsts.push_back(GEP);

    } else if (auto *SI = dyn_cast<StoreInst>(U)) {
      assert(SI->getValueOperand() != II && "Pointer escaped!");
      createStoreIntrinsic(II, SI, RTI);
      DeadInsts.push_back(SI);

    } else if (auto *LI = dyn_cast<LoadInst>(U)) {
      createLoadIntrinsic(II, LI, RTI);
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
