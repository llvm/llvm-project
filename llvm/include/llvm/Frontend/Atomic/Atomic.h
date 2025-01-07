//===--- Atomic.h - Codegen of atomic operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_ATOMIC_ATOMIC_H
#define LLVM_FRONTEND_ATOMIC_ATOMIC_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace llvm {
class AtomicInfo {
protected:
  IRBuilderBase *Builder;
  Type *Ty;
  uint64_t AtomicSizeInBits;
  uint64_t ValueSizeInBits;
  Align AtomicAlign;
  Align ValueAlign;
  bool UseLibcall;

public:
  AtomicInfo(IRBuilderBase *Builder, Type *Ty, uint64_t AtomicSizeInBits,
             uint64_t ValueSizeInBits, Align AtomicAlign, Align ValueAlign,
             bool UseLibcall)
      : Builder(Builder), Ty(Ty), AtomicSizeInBits(AtomicSizeInBits),
        ValueSizeInBits(ValueSizeInBits), AtomicAlign(AtomicAlign),
        ValueAlign(ValueAlign), UseLibcall(UseLibcall) {}

  virtual ~AtomicInfo() = default;

  Align getAtomicAlignment() const { return AtomicAlign; }
  uint64_t getAtomicSizeInBits() const { return AtomicSizeInBits; }
  uint64_t getValueSizeInBits() const { return ValueSizeInBits; }
  bool shouldUseLibcall() const { return UseLibcall; }
  Type *getAtomicTy() const { return Ty; }

  virtual Value *getAtomicPointer() const = 0;
  virtual void decorateWithTBAA(Instruction *I) = 0;
  virtual AllocaInst *CreateAlloca(Type *Ty, const Twine &Name) const = 0;

  /*
   * Is the atomic size larger than the underlying value type?
   * Note that the absence of padding does not mean that atomic
   * objects are completely interchangeable with non-atomic
   * objects: we might have promoted the alignment of a type
   * without making it bigger.
   */
  bool hasPadding() const { return (ValueSizeInBits != AtomicSizeInBits); }

  LLVMContext &getLLVMContext() const { return Builder->getContext(); }

  bool shouldCastToInt(Type *ValTy, bool CmpXchg);

  Value *EmitAtomicLoadOp(AtomicOrdering AO, bool IsVolatile,
                          bool CmpXchg = false);

  CallInst *EmitAtomicLibcall(StringRef fnName, Type *ResultType,
                              ArrayRef<Value *> Args);

  Value *getAtomicSizeValue() const {
    LLVMContext &ctx = getLLVMContext();
    // TODO: Get from llvm::TargetMachine / clang::TargetInfo
    // if clang shares this codegen in future
    constexpr uint16_t SizeTBits = 64;
    constexpr uint16_t BitsPerByte = 8;
    return ConstantInt::get(IntegerType::get(ctx, SizeTBits),
                            AtomicSizeInBits / BitsPerByte);
  }

  std::pair<Value *, Value *>
  EmitAtomicCompareExchangeLibcall(Value *ExpectedVal, Value *DesiredVal,
                                   AtomicOrdering Success,
                                   AtomicOrdering Failure);

  Value *castToAtomicIntPointer(Value *addr) const {
    return addr; // opaque pointer
  }

  Value *getAtomicAddressAsAtomicIntPointer() const {
    return castToAtomicIntPointer(getAtomicPointer());
  }

  std::pair<Value *, Value *>
  EmitAtomicCompareExchangeOp(Value *ExpectedVal, Value *DesiredVal,
                              AtomicOrdering Success, AtomicOrdering Failure,
                              bool IsVolatile = false, bool IsWeak = false);

  std::pair<Value *, Value *>
  EmitAtomicCompareExchange(Value *ExpectedVal, Value *DesiredVal,
                            AtomicOrdering Success, AtomicOrdering Failure,
                            bool IsVolatile, bool IsWeak);

  std::pair<LoadInst *, AllocaInst *> EmitAtomicLoadLibcall(AtomicOrdering AO);
};
} // end namespace llvm

#endif /* LLVM_FRONTEND_ATOMIC_ATOMIC_H */
