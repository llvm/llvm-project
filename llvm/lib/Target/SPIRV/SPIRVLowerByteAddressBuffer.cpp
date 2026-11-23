//===- SPIRVLowerByteAddressBuffer.cpp - Lower raw buffer accesses
//-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower typed raw-buffer accesses to load/stores. If SPIRV untyped pointers are
// supported, emit them directly at the specified byte offset with appropriate
// alignment.
// When untyped pointers aren't supported, for some load/store,
// create an alias of the resource treating it as a runtime-sized array of
// the given type. For now, scalarize vector types, although if we do analysis
// of the byte offset and it's divisible by the vector size, we could use
// an array of vectors.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRV.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-lower-byte-address-buffer"

namespace {
class LowerByteAddressBuffer {
  Module &M;
  const DataLayout &DL;
  bool UseUntypedPointers;
  DenseMap<std::pair<IntrinsicInst *, Type *>, Value *> Aliases;

  Value *getPointer(IRBuilder<> &B, IntrinsicInst &Access, Type *Ty,
                    Value *Offset, Align Alignment) {
    Value *Handle = Access.getArgOperand(0);
    auto *HandleFromBinding = dyn_cast<IntrinsicInst>(Handle);
    uint64_t Size = DL.getTypeAllocSize(Ty).getFixedValue();
    if (!UseUntypedPointers && HandleFromBinding &&
        HandleFromBinding->getIntrinsicID() ==
            Intrinsic::spv_resource_handlefrombinding &&
        isPowerOf2_64(Size) && Alignment >= DL.getABITypeAlign(Ty)) {
      Value *&Alias = Aliases[{HandleFromBinding, Ty}];
      if (!Alias) {
        StringRef Name;
        if (getConstantStringInfo(HandleFromBinding->getArgOperand(4), Name)) {
          auto *HandleTy = cast<TargetExtType>(Handle->getType());
          auto *AliasTy = TargetExtType::get(
              M.getContext(), "spirv.VulkanBuffer", {ArrayType::get(Ty, 0)},
              {SPIRV::StorageClass::StorageBuffer,
               HandleTy->getIntParameter(1)});
          std::string TypeName;
          raw_string_ostream OS(TypeName);
          Ty->print(OS);
          std::string AliasName =
              (Twine(".resource.alias.") +
               Twine(cast<ConstantInt>(HandleFromBinding->getArgOperand(0))
                         ->getZExtValue()) +
               "." +
               Twine(cast<ConstantInt>(HandleFromBinding->getArgOperand(1))
                         ->getZExtValue()) +
               "." + TypeName + "." +
               (HandleTy->getIntParameter(1) ? "rw." : "ro.") + Name)
                  .str();
          // TODO: If multiple views of this binding remain live, decorate all
          // corresponding resource variables with Aliased.
          IRBuilder<> AliasBuilder(HandleFromBinding);
          Constant *NamePtr = AliasBuilder.CreateGlobalString(
              AliasName, ".resource.alias.name");
          SmallVector<Value *> Args(HandleFromBinding->args());
          Args[4] = NamePtr;
          Alias = AliasBuilder.CreateIntrinsic(
              Intrinsic::spv_resource_handlefrombinding, {AliasTy}, Args);
        }
      }
      if (Alias) {
        Handle = Alias;
        Offset = B.CreateLShr(Offset, Log2_64(Size));
      }
    }
    Type *PtrTy = PointerType::get(M.getContext(), 11);
    return B.CreateIntrinsic(Intrinsic::spv_resource_getpointer,
                             {PtrTy, Handle->getType(), Offset->getType()},
                             {Handle, Offset});
  }

  Value *lowerAccess(IRBuilder<> &B, IntrinsicInst &Access, Type *Ty,
                     Value *Offset, Align Alignment, Value *Stored = nullptr) {
    assert(!Ty->isAggregateType() &&
           "Raw buffer accesses must be scalar or vector");
    auto *VT = dyn_cast<FixedVectorType>(Ty);
    if (VT && !UseUntypedPointers) {
      Value *Result = PoisonValue::get(Ty);
      Type *ElementTy = VT->getElementType();
      uint64_t Stride = DL.getTypeAllocSize(ElementTy).getFixedValue();
      for (unsigned I = 0; I != VT->getNumElements(); ++I) {
        uint64_t ByteOffset = I * Stride;
        Value *ElementOffset = B.CreateAdd(Offset, B.getInt32(ByteOffset));
        Value *Element = Stored ? B.CreateExtractElement(Stored, I) : nullptr;
        Element = lowerAccess(B, Access, ElementTy, ElementOffset,
                              commonAlignment(Alignment, ByteOffset), Element);
        if (!Stored)
          Result = B.CreateInsertElement(Result, Element, I);
      }
      return Result;
    }
    Value *Pointer = getPointer(B, Access, Ty, Offset, Alignment);
    if (Stored)
      return B.CreateAlignedStore(Stored, Pointer, Alignment);
    return B.CreateAlignedLoad(Ty, Pointer, Alignment);
  }

public:
  LowerByteAddressBuffer(Module &M, bool UseUntypedPointers)
      : M(M), DL(M.getDataLayout()), UseUntypedPointers(UseUntypedPointers) {}

  bool run() {
    bool Changed = false;
    for (Function &F : M) {
      Aliases.clear();
      // TODO: Handle BAB atomics separately, preserving their full scalar width.
      SmallVector<IntrinsicInst *> Accesses;
      for (Instruction &I : instructions(F))
        if (auto *II = dyn_cast<IntrinsicInst>(&I))
          if (II->getIntrinsicID() == Intrinsic::spv_resource_load_rawbuffer ||
              II->getIntrinsicID() == Intrinsic::spv_resource_store_rawbuffer)
            Accesses.push_back(II);
      Changed |= !Accesses.empty();
      for (IntrinsicInst *Access : Accesses) {
        bool IsLoad =
            Access->getIntrinsicID() == Intrinsic::spv_resource_load_rawbuffer;
        Value *Stored = IsLoad ? nullptr : Access->getArgOperand(2);
        Type *Ty = IsLoad ? Access->getType() : Stored->getType();
        Align Alignment(cast<ConstantInt>(Access->getArgOperand(IsLoad ? 2 : 3))
                            ->getZExtValue());
        IRBuilder<> B(Access);
        Value *Handle = Access->getArgOperand(0);
        Value *Result = lowerAccess(B, *Access, Ty, Access->getArgOperand(1),
                                    Alignment, Stored);
        if (IsLoad)
          Access->replaceAllUsesWith(Result);
        Access->eraseFromParent();
        RecursivelyDeleteTriviallyDeadInstructions(Handle);
      }
    }
    return Changed;
  }
};
} // namespace

PreservedAnalyses
SPIRVLowerByteAddressBufferPass::run(Module &M, ModuleAnalysisManager &) {
  return LowerByteAddressBuffer(M, UseUntypedPointers).run()
             ? PreservedAnalyses::none()
             : PreservedAnalyses::all();
}

namespace {
class SPIRVLowerByteAddressBufferLegacy : public ModulePass {
  bool UseUntypedPointers;

public:
  static char ID;
  SPIRVLowerByteAddressBufferLegacy(bool UseUntypedPointers = false)
      : ModulePass(ID), UseUntypedPointers(UseUntypedPointers) {}
  bool runOnModule(Module &M) override {
    return LowerByteAddressBuffer(M, UseUntypedPointers).run();
  }
};
char SPIRVLowerByteAddressBufferLegacy::ID = 0;
} // namespace

INITIALIZE_PASS(SPIRVLowerByteAddressBufferLegacy, DEBUG_TYPE,
                "SPIRV Lower ByteAddressBuffer", false, false)

ModulePass *
llvm::createSPIRVLowerByteAddressBufferLegacyPass(bool UseUntypedPointers) {
  return new SPIRVLowerByteAddressBufferLegacy(UseUntypedPointers);
}
