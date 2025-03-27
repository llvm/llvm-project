//===--- Atomic.cpp - Codegen of atomic operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Atomic/Atomic.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include <utility>

using namespace llvm;

bool AtomicInfo::shouldCastToInt(Type *ValTy, bool CmpXchg) {
  if (ValTy->isFloatingPointTy())
    return ValTy->isX86_FP80Ty() || CmpXchg;
  return !ValTy->isIntegerTy() && !ValTy->isPointerTy();
}

Value *AtomicInfo::EmitAtomicLoadOp(AtomicOrdering AO, bool IsVolatile,
                                    bool CmpXchg) {
  Value *Ptr = getAtomicPointer();
  Type *AtomicTy = Ty;
  if (shouldCastToInt(Ty, CmpXchg))
    AtomicTy = IntegerType::get(getLLVMContext(), AtomicSizeInBits);
  LoadInst *Load =
      Builder->CreateAlignedLoad(AtomicTy, Ptr, AtomicAlign, "atomic-load");
  Load->setAtomic(AO);
  if (IsVolatile)
    Load->setVolatile(true);
  decorateWithTBAA(Load);
  return Load;
}

CallInst *AtomicInfo::EmitAtomicLibcall(StringRef fnName, Type *ResultType,
                                        ArrayRef<Value *> Args) {
  LLVMContext &ctx = Builder->getContext();
  SmallVector<Type *, 6> ArgTys;
  for (Value *Arg : Args)
    ArgTys.push_back(Arg->getType());
  FunctionType *FnType = FunctionType::get(ResultType, ArgTys, false);
  Module *M = Builder->GetInsertBlock()->getModule();

  // TODO: Use llvm::TargetLowering for Libcall ABI
  AttrBuilder fnAttrBuilder(ctx);
  fnAttrBuilder.addAttribute(Attribute::NoUnwind);
  fnAttrBuilder.addAttribute(Attribute::WillReturn);
  AttributeList fnAttrs =
      AttributeList::get(ctx, AttributeList::FunctionIndex, fnAttrBuilder);
  FunctionCallee LibcallFn = M->getOrInsertFunction(fnName, FnType, fnAttrs);
  CallInst *Call = Builder->CreateCall(LibcallFn, Args);
  return Call;
}

std::pair<Value *, Value *> AtomicInfo::EmitAtomicCompareExchangeLibcall(
    Value *ExpectedVal, Value *DesiredVal, AtomicOrdering Success,
    AtomicOrdering Failure) {
  LLVMContext &ctx = getLLVMContext();

  // __atomic_compare_exchange's expected and desired are passed by pointers
  // FIXME: types

  // TODO: Get from llvm::TargetMachine / clang::TargetInfo
  // if clang shares this codegen in future
  constexpr uint64_t IntBits = 32;

  // bool __atomic_compare_exchange(size_t size, void *obj, void *expected,
  //  void *desired, int success, int failure);

  Value *Args[6] = {
      getAtomicSizeValue(),
      getAtomicPointer(),
      ExpectedVal,
      DesiredVal,
      Constant::getIntegerValue(IntegerType::get(ctx, IntBits),
                                APInt(IntBits, static_cast<uint64_t>(Success),
                                      /*signed=*/true)),
      Constant::getIntegerValue(IntegerType::get(ctx, IntBits),
                                APInt(IntBits, static_cast<uint64_t>(Failure),
                                      /*signed=*/true)),
  };
  auto Result = EmitAtomicLibcall("__atomic_compare_exchange",
                                  IntegerType::getInt1Ty(ctx), Args);
  return std::make_pair(ExpectedVal, Result);
}

std::pair<Value *, Value *> AtomicInfo::EmitAtomicCompareExchangeOp(
    Value *ExpectedVal, Value *DesiredVal, AtomicOrdering Success,
    AtomicOrdering Failure, bool IsVolatile, bool IsWeak) {
  // Do the atomic store.
  Value *Addr = getAtomicAddressAsAtomicIntPointer();
  auto *Inst = Builder->CreateAtomicCmpXchg(Addr, ExpectedVal, DesiredVal,
                                            getAtomicAlignment(), Success,
                                            Failure, SyncScope::System);

  // Other decoration.
  Inst->setVolatile(IsVolatile);
  Inst->setWeak(IsWeak);
  auto *PreviousVal = Builder->CreateExtractValue(Inst, /*Idxs=*/0);
  auto *SuccessFailureVal = Builder->CreateExtractValue(Inst, /*Idxs=*/1);
  return std::make_pair(PreviousVal, SuccessFailureVal);
}

std::pair<LoadInst *, AllocaInst *>
AtomicInfo::EmitAtomicLoadLibcall(AtomicOrdering AO) {
  LLVMContext &Ctx = getLLVMContext();
  Type *SizedIntTy = Type::getIntNTy(Ctx, getAtomicSizeInBits());
  Type *ResultTy;
  SmallVector<Value *, 6> Args;
  AttributeList Attr;
  Module *M = Builder->GetInsertBlock()->getModule();
  const DataLayout &DL = M->getDataLayout();
  Args.push_back(
      ConstantInt::get(DL.getIntPtrType(Ctx), this->getAtomicSizeInBits() / 8));

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

std::pair<Value *, Value *> AtomicInfo::EmitAtomicCompareExchange(
    Value *ExpectedVal, Value *DesiredVal, AtomicOrdering Success,
    AtomicOrdering Failure, bool IsVolatile, bool IsWeak) {
  if (shouldUseLibcall())
    return EmitAtomicCompareExchangeLibcall(ExpectedVal, DesiredVal, Success,
                                            Failure);

  auto Res = EmitAtomicCompareExchangeOp(ExpectedVal, DesiredVal, Success,
                                         Failure, IsVolatile, IsWeak);
  return Res;
}
