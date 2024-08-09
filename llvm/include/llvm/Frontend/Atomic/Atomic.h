//===--- Atomic.h - Shared codegen of atomic operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_ATOMIC_ATOMIC_H
#define LLVM_FRONTEND_ATOMIC_ATOMIC_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"

namespace llvm {

template <typename IRBuilderTy> struct AtomicInfo {
  IRBuilderTy *Builder;

  Type *Ty;
  uint64_t AtomicSizeInBits;
  uint64_t ValueSizeInBits;
  llvm::Align AtomicAlign;
  llvm::Align ValueAlign;
  bool UseLibcall;

public:
  AtomicInfo(IRBuilderTy *Builder, Type *Ty, uint64_t AtomicSizeInBits,
             uint64_t ValueSizeInBits, llvm::Align AtomicAlign,
             llvm::Align ValueAlign, bool UseLibcall)
      : Builder(Builder), Ty(Ty), AtomicSizeInBits(AtomicSizeInBits),
        ValueSizeInBits(ValueSizeInBits), AtomicAlign(AtomicAlign),
        ValueAlign(ValueAlign), UseLibcall(UseLibcall) {}

  virtual ~AtomicInfo() = default;

  llvm::Align getAtomicAlignment() const { return AtomicAlign; }
  uint64_t getAtomicSizeInBits() const { return AtomicSizeInBits; }
  uint64_t getValueSizeInBits() const { return ValueSizeInBits; }
  bool shouldUseLibcall() const { return UseLibcall; }
  llvm::Type *getAtomicTy() const { return Ty; }

  virtual llvm::Value *getAtomicPointer() const = 0;
  virtual void decorateWithTBAA(Instruction *I) {}
  virtual llvm::AllocaInst *CreateAlloca(llvm::Type *Ty,
                                         const llvm::Twine &Name) = 0;

  /// Is the atomic size larger than the underlying value type?
  ///
  /// Note that the absence of padding does not mean that atomic
  /// objects are completely interchangeable with non-atomic
  /// objects: we might have promoted the alignment of a type
  /// without making it bigger.
  bool hasPadding() const { return (ValueSizeInBits != AtomicSizeInBits); }

  LLVMContext &getLLVMContext() const { return Builder->getContext(); }

  static bool shouldCastToInt(llvm::Type *ValTy, bool CmpXchg) {
    if (ValTy->isFloatingPointTy())
      return ValTy->isX86_FP80Ty() || CmpXchg;
    return !ValTy->isIntegerTy() && !ValTy->isPointerTy();
  }

  llvm::Value *EmitAtomicLoadOp(llvm::AtomicOrdering AO, bool IsVolatile,
                                bool CmpXchg = false) {
    Value *Ptr = getAtomicPointer();
    Type *AtomicTy = Ty;
    if (shouldCastToInt(Ty, CmpXchg))
      AtomicTy = llvm::IntegerType::get(getLLVMContext(), AtomicSizeInBits);
    LoadInst *Load =
        Builder->CreateAlignedLoad(AtomicTy, Ptr, AtomicAlign, "atomic-load");
    Load->setAtomic(AO);

    // Other decoration.
    if (IsVolatile)
      Load->setVolatile(true);
    decorateWithTBAA(Load);
    return Load;
  }

  static CallInst *emitAtomicLibcalls2(IRBuilderTy *Builder, StringRef fnName,
                                       Type *ResultType,
                                       ArrayRef<Value *> Args) {
    auto &C = Builder->getContext();

    SmallVector<Type *, 6> ArgTys;
    for (Value *Arg : Args)
      ArgTys.push_back(Arg->getType());
    FunctionType *FnType = FunctionType::get(ResultType, ArgTys, false);
    Module *M = Builder->GetInsertBlock()->getModule();

    // TODO: Use llvm::TargetLowering for Libcall ABI
    llvm::AttrBuilder fnAttrB(C);
    fnAttrB.addAttribute(llvm::Attribute::NoUnwind);
    fnAttrB.addAttribute(llvm::Attribute::WillReturn);
    llvm::AttributeList fnAttrs = llvm::AttributeList::get(
        C, llvm::AttributeList::FunctionIndex, fnAttrB);
    FunctionCallee LibcallFn = M->getOrInsertFunction(fnName, FnType, fnAttrs);
    CallInst *Call = Builder->CreateCall(LibcallFn, Args);

    return Call;
  }

  llvm::Value *getAtomicSizeValue() const {
    auto &C = getLLVMContext();

    // TODO: Get from llvm::TargetMachine / clang::TargetInfo
    uint16_t SizeTBits = 64;
    uint16_t BitsPerByte = 8;
    return llvm::ConstantInt::get(llvm::IntegerType::get(C, SizeTBits),
                                  AtomicSizeInBits / BitsPerByte);
  }

  std::pair<llvm::Value *, llvm::Value *> EmitAtomicCompareExchangeLibcall(
      llvm::Value *ExpectedVal, llvm::Value *DesiredVal,
      llvm::AtomicOrdering Success, llvm::AtomicOrdering Failure) {
    auto &C = getLLVMContext();

    // __atomic_compare_exchange's expected and desired are passed by pointers
    // FIXME: types

    // TODO: Get from llvm::TargetMachine / clang::TargetInfo
    uint64_t IntBits = 32;

    // bool __atomic_compare_exchange(size_t size, void *obj, void *expected,
    // void *desired, int success, int failure);
    llvm::Value *Args[6] = {
        getAtomicSizeValue(),
        getAtomicPointer(),
        ExpectedVal,
        DesiredVal,
        llvm::Constant::getIntegerValue(
            llvm::IntegerType::get(C, IntBits),
            llvm::APInt(IntBits, (uint64_t)Success, /*signed=*/true)),
        llvm::Constant::getIntegerValue(
            llvm::IntegerType::get(C, IntBits),
            llvm::APInt(IntBits, (uint64_t)Failure, /*signed=*/true)),
    };
    auto Res = emitAtomicLibcalls2(Builder, "__atomic_compare_exchange",
                                   llvm::IntegerType::getInt1Ty(C), Args);
    return std::make_pair(ExpectedVal, Res);
  }

  Value *castToAtomicIntPointer(Value *addr) const {
    return addr; // opaque pointer
  }

  Value *getAtomicAddressAsAtomicIntPointer() const {
    return castToAtomicIntPointer(getAtomicPointer());
  }

  std::pair<llvm::Value *, llvm::Value *>
  EmitAtomicCompareExchangeOp(llvm::Value *ExpectedVal, llvm::Value *DesiredVal,
                              llvm::AtomicOrdering Success,
                              llvm::AtomicOrdering Failure,
                              bool IsVolatile = false, bool IsWeak = false) {
    // Do the atomic store.
    Value *Addr = getAtomicAddressAsAtomicIntPointer();
    auto *Inst = Builder->CreateAtomicCmpXchg(Addr, ExpectedVal, DesiredVal,
                                              getAtomicAlignment(), Success,
                                              Failure, llvm::SyncScope::System);
    // Other decoration.
    Inst->setVolatile(IsVolatile);
    Inst->setWeak(IsWeak);

    // Okay, turn that back into the original value type.
    auto *PreviousVal = Builder->CreateExtractValue(Inst, /*Idxs=*/0);
    auto *SuccessFailureVal = Builder->CreateExtractValue(Inst, /*Idxs=*/1);
    return std::make_pair(PreviousVal, SuccessFailureVal);
  }

  std::pair<llvm::Value *, llvm::Value *>
  EmitAtomicCompareExchange(llvm::Value *ExpectedVal, llvm::Value *DesiredVal,
                            llvm::AtomicOrdering Success,
                            llvm::AtomicOrdering Failure, bool IsVolatile,
                            bool IsWeak) {

    // Check whether we should use a library call.
    if (shouldUseLibcall()) {
      return EmitAtomicCompareExchangeLibcall(ExpectedVal, DesiredVal, Success,
                                              Failure);
    }

    // If we've got a scalar value of the right size, try to avoid going
    // through memory.
    auto Res = EmitAtomicCompareExchangeOp(ExpectedVal, DesiredVal, Success,
                                           Failure, IsVolatile, IsWeak);
    return Res;
  }

  // void __atomic_load(size_t size, void *mem, void *return, int order);
  std::pair<llvm::LoadInst *, llvm::AllocaInst *>
  EmitAtomicLoadLibcall(llvm::AtomicOrdering AO) {
    LLVMContext &Ctx = getLLVMContext();
    Type *SizedIntTy = Type::getIntNTy(Ctx, getAtomicSizeInBits() * 8);
    Type *ResultTy;
    SmallVector<Value *, 6> Args;
    AttributeList Attr;
    Module *M = Builder->GetInsertBlock()->getModule();
    const DataLayout &DL = M->getDataLayout();
    Args.push_back(ConstantInt::get(DL.getIntPtrType(Ctx),
                                    this->getAtomicSizeInBits() / 8));
    Value *PtrVal = getAtomicPointer();
    PtrVal = Builder->CreateAddrSpaceCast(PtrVal, PointerType::getUnqual(Ctx));
    Args.push_back(PtrVal);
    AllocaInst *AllocaResult =
        CreateAlloca(Ty, getAtomicPointer()->getName() + "atomic.temp.load");
    const Align AllocaAlignment = DL.getPrefTypeAlign(SizedIntTy);
    AllocaResult->setAlignment(AllocaAlignment);
    Args.push_back(AllocaResult);
    Constant *OrderingVal =
        ConstantInt::get(Type::getInt32Ty(Ctx), (int)toCABI(AO));
    Args.push_back(OrderingVal);
    ResultTy = Type::getVoidTy(Ctx);
    SmallVector<Type *, 6> ArgTys;
    for (Value *Arg : Args)
      ArgTys.push_back(Arg->getType());
    FunctionType *FnType = FunctionType::get(ResultTy, ArgTys, false);
    FunctionCallee LibcallFn =
        M->getOrInsertFunction("__atomic_load", FnType, Attr);
    CallInst *Call = Builder->CreateCall(LibcallFn, Args);
    Call->setAttributes(Attr);
    return std::make_pair(
        Builder->CreateAlignedLoad(Ty, AllocaResult, AllocaAlignment),
        AllocaResult);
  }
};
} // end namespace llvm

#endif /* LLVM_FRONTEND_ATOMIC_ATOMIC_H */
