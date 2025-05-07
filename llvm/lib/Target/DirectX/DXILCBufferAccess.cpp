//===- DXILCBufferAccess.cpp - Translate CBuffer Loads --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILCBufferAccess.h"
#include "DirectX.h"
#include "llvm/Frontend/HLSL/CBuffer.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "dxil-cbuffer-access"
using namespace llvm;

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

static size_t getOffsetForCBufferGEP(GEPOperator *GEP, GlobalVariable *Global,
                                     const DataLayout &DL) {
  // Since we should always have a constant offset, we should only ever have a
  // single GEP of indirection from the Global.
  assert(GEP->getPointerOperand() == Global &&
         "Indirect access to resource handle");

  APInt ConstantOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
  bool Success = GEP->accumulateConstantOffset(DL, ConstantOffset);
  (void)Success;
  assert(Success && "Offsets into cbuffer globals must be constant");

  if (auto *ATy = dyn_cast<ArrayType>(Global->getValueType()))
    ConstantOffset = hlsl::translateCBufArrayOffset(DL, ConstantOffset, ATy);

  return ConstantOffset.getZExtValue();
}

/// Replace access via cbuffer global with a load from the cbuffer handle
/// itself.
static void replaceAccess(LoadInst *LI, GlobalVariable *Global,
                          GlobalVariable *HandleGV, size_t BaseOffset,
                          SmallVectorImpl<WeakTrackingVH> &DeadInsts) {
  const DataLayout &DL = HandleGV->getDataLayout();

  size_t Offset = BaseOffset;
  if (auto *GEP = dyn_cast<GEPOperator>(LI->getPointerOperand()))
    Offset += getOffsetForCBufferGEP(GEP, Global, DL);
  else if (LI->getPointerOperand() != Global)
    llvm_unreachable("Load instruction doesn't reference cbuffer global");

  IRBuilder<> Builder(LI);
  auto *Handle = Builder.CreateLoad(HandleGV->getValueType(), HandleGV,
                                    HandleGV->getName());

  Type *Ty = LI->getType();
  CBufferRowIntrin Intrin(DL, Ty->getScalarType());
  // The cbuffer consists of some number of 16-byte rows.
  unsigned int CurrentRow = Offset / hlsl::CBufferRowSizeInBytes;
  unsigned int CurrentIndex =
      (Offset % hlsl::CBufferRowSizeInBytes) / Intrin.EltSize;

  auto *CBufLoad = Builder.CreateIntrinsic(
      Intrin.RetTy, Intrin.IID,
      {Handle, ConstantInt::get(Builder.getInt32Ty(), CurrentRow)}, nullptr,
      LI->getName());
  auto *Elt =
      Builder.CreateExtractValue(CBufLoad, {CurrentIndex++}, LI->getName());

  Value *Result = nullptr;
  unsigned int Remaining =
      ((DL.getTypeSizeInBits(Ty) / 8) / Intrin.EltSize) - 1;
  if (Remaining == 0) {
    // We only have a single element, so we're done.
    Result = Elt;

    // However, if we loaded a <1 x T>, then we need to adjust the type here.
    if (auto *VT = dyn_cast<FixedVectorType>(LI->getType())) {
      assert(VT->getNumElements() == 1 && "Can't have multiple elements here");
      Result = Builder.CreateInsertElement(PoisonValue::get(VT), Result,
                                           Builder.getInt32(0));
    }
  } else {
    // Walk each element and extract it, wrapping to new rows as needed.
    SmallVector<Value *> Extracts{Elt};
    while (Remaining--) {
      CurrentIndex %= Intrin.NumElts;

      if (CurrentIndex == 0)
        CBufLoad = Builder.CreateIntrinsic(
            Intrin.RetTy, Intrin.IID,
            {Handle, ConstantInt::get(Builder.getInt32Ty(), ++CurrentRow)},
            nullptr, LI->getName());

      Extracts.push_back(Builder.CreateExtractValue(CBufLoad, {CurrentIndex++},
                                                    LI->getName()));
    }

    // Finally, we build up the original loaded value.
    Result = PoisonValue::get(Ty);
    for (int I = 0, E = Extracts.size(); I < E; ++I)
      Result =
          Builder.CreateInsertElement(Result, Extracts[I], Builder.getInt32(I));
  }

  LI->replaceAllUsesWith(Result);
  DeadInsts.push_back(LI);
}

static void replaceAccessesWithHandle(GlobalVariable *Global,
                                      GlobalVariable *HandleGV,
                                      size_t BaseOffset) {
  SmallVector<WeakTrackingVH> DeadInsts;

  SmallVector<User *> ToProcess{Global->users()};
  while (!ToProcess.empty()) {
    User *Cur = ToProcess.pop_back_val();

    // If we have a load instruction, replace the access.
    if (auto *LI = dyn_cast<LoadInst>(Cur)) {
      replaceAccess(LI, Global, HandleGV, BaseOffset, DeadInsts);
      continue;
    }

    // Otherwise, walk users looking for a load...
    ToProcess.append(Cur->user_begin(), Cur->user_end());
  }
  RecursivelyDeleteTriviallyDeadInstructions(DeadInsts);
}

static bool replaceCBufferAccesses(Module &M) {
  std::optional<hlsl::CBufferMetadata> CBufMD = hlsl::CBufferMetadata::get(M);
  if (!CBufMD)
    return false;

  for (const hlsl::CBufferMapping &Mapping : *CBufMD)
    for (const hlsl::CBufferMember &Member : Mapping.Members) {
      replaceAccessesWithHandle(Member.GV, Mapping.Handle, Member.Offset);
      Member.GV->removeFromParent();
    }

  CBufMD->eraseFromModule();
  return true;
}

PreservedAnalyses DXILCBufferAccess::run(Module &M, ModuleAnalysisManager &AM) {
  PreservedAnalyses PA;
  bool Changed = replaceCBufferAccesses(M);

  if (!Changed)
    return PreservedAnalyses::all();
  return PA;
}

namespace {
class DXILCBufferAccessLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override { return replaceCBufferAccesses(M); }
  StringRef getPassName() const override { return "DXIL CBuffer Access"; }
  DXILCBufferAccessLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
};
char DXILCBufferAccessLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(DXILCBufferAccessLegacy, DEBUG_TYPE, "DXIL CBuffer Access",
                false, false)

ModulePass *llvm::createDXILCBufferAccessLegacyPass() {
  return new DXILCBufferAccessLegacy();
}
