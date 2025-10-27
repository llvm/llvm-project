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
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormatVariadic.h"
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

// Helper for creating CBuffer handles and loading data from them
struct CBufferResource {
  GlobalVariable *GVHandle;
  GlobalVariable *Member;
  size_t MemberOffset;

  LoadInst *Handle;

  CBufferResource(GlobalVariable *GVHandle, GlobalVariable *Member,
                  size_t MemberOffset)
      : GVHandle(GVHandle), Member(Member), MemberOffset(MemberOffset) {}

  const DataLayout &getDataLayout() { return GVHandle->getDataLayout(); }
  Type *getValueType() { return Member->getValueType(); }
  iterator_range<ConstantDataSequential::user_iterator> users() {
    return Member->users();
  }

  /// Get the byte offset of a Pointer-typed Value * `Val` relative to Member.
  /// `Val` can either be Member itself, or a GEP of a constant offset from
  /// Member
  size_t getOffsetForCBufferGEP(Value *Val) {
    assert(isa<PointerType>(Val->getType()) &&
           "Expected a pointer-typed value");

    if (Val == Member)
      return 0;

    if (auto *GEP = dyn_cast<GEPOperator>(Val)) {
      // Since we should always have a constant offset, we should only ever have
      // a single GEP of indirection from the Global.
      assert(GEP->getPointerOperand() == Member &&
             "Indirect access to resource handle");

      const DataLayout &DL = getDataLayout();
      APInt ConstantOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      bool Success = GEP->accumulateConstantOffset(DL, ConstantOffset);
      (void)Success;
      assert(Success && "Offsets into cbuffer globals must be constant");

      if (auto *ATy = dyn_cast<ArrayType>(Member->getValueType()))
        ConstantOffset =
            hlsl::translateCBufArrayOffset(DL, ConstantOffset, ATy);

      return ConstantOffset.getZExtValue();
    }

    llvm_unreachable("Expected Val to be a GlobalVariable or GEP");
  }

  /// Create a handle for this cbuffer resource using the IRBuilder `Builder`
  /// and sets the handle as the current one to use for subsequent calls to
  /// `loadValue`
  void createAndSetCurrentHandle(IRBuilder<> &Builder) {
    Handle = Builder.CreateLoad(GVHandle->getValueType(), GVHandle,
                                GVHandle->getName());
  }

  /// Load a value of type `Ty` at offset `Offset` using the handle from the
  /// last call to `createAndSetCurrentHandle`
  Value *loadValue(IRBuilder<> &Builder, Type *Ty, size_t Offset,
                   const Twine &Name = "") {
    assert(Handle &&
           "Expected a handle for this cbuffer global resource to be created "
           "before loading a value from it");
    const DataLayout &DL = getDataLayout();

    size_t TargetOffset = MemberOffset + Offset;
    CBufferRowIntrin Intrin(DL, Ty->getScalarType());
    // The cbuffer consists of some number of 16-byte rows.
    unsigned int CurrentRow = TargetOffset / hlsl::CBufferRowSizeInBytes;
    unsigned int CurrentIndex =
        (TargetOffset % hlsl::CBufferRowSizeInBytes) / Intrin.EltSize;

    auto *CBufLoad = Builder.CreateIntrinsic(
        Intrin.RetTy, Intrin.IID,
        {Handle, ConstantInt::get(Builder.getInt32Ty(), CurrentRow)}, nullptr,
        Name + ".load");
    auto *Elt = Builder.CreateExtractValue(CBufLoad, {CurrentIndex++},
                                           Name + ".extract");

    Value *Result = nullptr;
    unsigned int Remaining =
        ((DL.getTypeSizeInBits(Ty) / 8) / Intrin.EltSize) - 1;

    if (Remaining == 0) {
      // We only have a single element, so we're done.
      Result = Elt;

      // However, if we loaded a <1 x T>, then we need to adjust the type here.
      if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
        assert(VT->getNumElements() == 1 &&
               "Can't have multiple elements here");
        Result = Builder.CreateInsertElement(PoisonValue::get(VT), Result,
                                             Builder.getInt32(0), Name);
      }
      return Result;
    }

    // Walk each element and extract it, wrapping to new rows as needed.
    SmallVector<Value *> Extracts{Elt};
    while (Remaining--) {
      CurrentIndex %= Intrin.NumElts;

      if (CurrentIndex == 0)
        CBufLoad = Builder.CreateIntrinsic(
            Intrin.RetTy, Intrin.IID,
            {Handle, ConstantInt::get(Builder.getInt32Ty(), ++CurrentRow)},
            nullptr, Name + ".load");

      Extracts.push_back(Builder.CreateExtractValue(CBufLoad, {CurrentIndex++},
                                                    Name + ".extract"));
    }

    // Finally, we build up the original loaded value.
    Result = PoisonValue::get(Ty);
    for (int I = 0, E = Extracts.size(); I < E; ++I)
      Result =
          Builder.CreateInsertElement(Result, Extracts[I], Builder.getInt32(I),
                                      Name + formatv(".upto{}", I));
    return Result;
  }
};

} // namespace

/// Replace load via cbuffer global with a load from the cbuffer handle itself.
static void replaceLoad(LoadInst *LI, CBufferResource &CBR,
                        SmallVectorImpl<WeakTrackingVH> &DeadInsts) {
  size_t Offset = CBR.getOffsetForCBufferGEP(LI->getPointerOperand());
  IRBuilder<> Builder(LI);
  CBR.createAndSetCurrentHandle(Builder);
  Value *Result = CBR.loadValue(Builder, LI->getType(), Offset, LI->getName());
  LI->replaceAllUsesWith(Result);
  DeadInsts.push_back(LI);
}

/// This function recursively copies N array elements from the cbuffer resource
/// CBR to the MemCpy Destination. Recursion is used to unravel multidimensional
/// arrays into a sequence of scalar/vector extracts and stores.
static void copyArrayElemsForMemCpy(IRBuilder<> &Builder, MemCpyInst *MCI,
                                    CBufferResource &CBR, ArrayType *ArrTy,
                                    size_t ArrOffset, size_t N,
                                    const Twine &Name = "") {
  const DataLayout &DL = MCI->getDataLayout();
  Type *ElemTy = ArrTy->getElementType();
  size_t ElemTySize = DL.getTypeAllocSize(ElemTy);
  for (unsigned I = 0; I < N; ++I) {
    size_t Offset = ArrOffset + I * ElemTySize;

    // Recursively copy nested arrays
    if (ArrayType *ElemArrTy = dyn_cast<ArrayType>(ElemTy)) {
      copyArrayElemsForMemCpy(Builder, MCI, CBR, ElemArrTy, Offset,
                              ElemArrTy->getNumElements(), Name);
      continue;
    }

    // Load CBuffer value and store it in Dest
    APInt CBufArrayOffset(
        DL.getIndexTypeSizeInBits(MCI->getSource()->getType()), Offset);
    CBufArrayOffset =
        hlsl::translateCBufArrayOffset(DL, CBufArrayOffset, ArrTy);
    Value *CBufferVal =
        CBR.loadValue(Builder, ElemTy, CBufArrayOffset.getZExtValue(), Name);
    Value *GEP =
        Builder.CreateInBoundsGEP(Builder.getInt8Ty(), MCI->getDest(),
                                  {Builder.getInt32(Offset)}, Name + ".dest");
    Builder.CreateStore(CBufferVal, GEP, MCI->isVolatile());
  }
}

/// Replace memcpy from a cbuffer global with a memcpy from the cbuffer handle
/// itself. Assumes the cbuffer global is an array, and the length of bytes to
/// copy is divisible by array element allocation size.
/// The memcpy source must also be a direct cbuffer global reference, not a GEP.
static void replaceMemCpy(MemCpyInst *MCI, CBufferResource &CBR) {

  ArrayType *ArrTy = dyn_cast<ArrayType>(CBR.getValueType());
  assert(ArrTy && "MemCpy lowering is only supported for array types");

  // This assumption vastly simplifies the implementation
  if (MCI->getSource() != CBR.Member)
    reportFatalUsageError(
        "Expected MemCpy source to be a cbuffer global variable");

  ConstantInt *Length = dyn_cast<ConstantInt>(MCI->getLength());
  uint64_t ByteLength = Length->getZExtValue();

  // If length to copy is zero, no memcpy is needed
  if (ByteLength == 0) {
    MCI->eraseFromParent();
    return;
  }

  const DataLayout &DL = CBR.getDataLayout();

  Type *ElemTy = ArrTy->getElementType();
  size_t ElemSize = DL.getTypeAllocSize(ElemTy);
  assert(ByteLength % ElemSize == 0 &&
         "Length of bytes to MemCpy must be divisible by allocation size of "
         "source/destination array elements");
  size_t ElemsToCpy = ByteLength / ElemSize;

  IRBuilder<> Builder(MCI);
  CBR.createAndSetCurrentHandle(Builder);

  copyArrayElemsForMemCpy(Builder, MCI, CBR, ArrTy, 0, ElemsToCpy,
                          "memcpy." + MCI->getDest()->getName() + "." +
                              MCI->getSource()->getName());

  MCI->eraseFromParent();
}

static void replaceAccessesWithHandle(CBufferResource &CBR) {
  SmallVector<WeakTrackingVH> DeadInsts;

  SmallVector<User *> ToProcess{CBR.users()};
  while (!ToProcess.empty()) {
    User *Cur = ToProcess.pop_back_val();

    // If we have a load instruction, replace the access.
    if (auto *LI = dyn_cast<LoadInst>(Cur)) {
      replaceLoad(LI, CBR, DeadInsts);
      continue;
    }

    // If we have a memcpy instruction, replace it with multiple accesses and
    // subsequent stores to the destination
    if (auto *MCI = dyn_cast<MemCpyInst>(Cur)) {
      replaceMemCpy(MCI, CBR);
      continue;
    }

    // Otherwise, walk users looking for a load...
    if (isa<GetElementPtrInst>(Cur) || isa<GEPOperator>(Cur)) {
      ToProcess.append(Cur->user_begin(), Cur->user_end());
      continue;
    }

    llvm_unreachable("Unexpected user of Global");
  }
  RecursivelyDeleteTriviallyDeadInstructions(DeadInsts);
}

static bool replaceCBufferAccesses(Module &M) {
  std::optional<hlsl::CBufferMetadata> CBufMD = hlsl::CBufferMetadata::get(M);
  if (!CBufMD)
    return false;

  for (const hlsl::CBufferMapping &Mapping : *CBufMD)
    for (const hlsl::CBufferMember &Member : Mapping.Members) {
      CBufferResource CBR(Mapping.Handle, Member.GV, Member.Offset);
      replaceAccessesWithHandle(CBR);
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
