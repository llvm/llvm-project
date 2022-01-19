//===- MemoryBuiltins.cpp - Identify calls to memory builtins -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This family of functions identifies calls to builtin functions that allocate
// or free memory.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/Utils/Local.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "memory-builtins"

enum AllocType : uint8_t {
  OpNewLike          = 1<<0, // allocates; never returns null
  MallocLike         = 1<<1, // allocates; may return null
  AlignedAllocLike   = 1<<2, // allocates with alignment; may return null
  CallocLike         = 1<<3, // allocates + bzero
  ReallocLike        = 1<<4, // reallocates
  StrDupLike         = 1<<5,
  MallocOrOpNewLike  = MallocLike | OpNewLike,
  MallocOrCallocLike = MallocLike | OpNewLike | CallocLike | AlignedAllocLike,
  AllocLike          = MallocOrCallocLike | StrDupLike,
  AnyAlloc           = AllocLike | ReallocLike
};

struct AllocFnsTy {
  AllocType AllocTy;
  unsigned NumParams;
  // First and Second size parameters (or -1 if unused)
  int FstParam, SndParam;
  // Alignment parameter for aligned_alloc and aligned new
  int AlignParam;
};

// FIXME: certain users need more information. E.g., SimplifyLibCalls needs to
// know which functions are nounwind, noalias, nocapture parameters, etc.
static const std::pair<LibFunc, AllocFnsTy> AllocationFnData[] = {
    {LibFunc_malloc,                            {MallocLike,       1,  0, -1, -1}},
    {LibFunc_vec_malloc,                        {MallocLike,       1,  0, -1, -1}},
    {LibFunc_valloc,                            {MallocLike,       1,  0, -1, -1}},
    {LibFunc_Znwj,                              {OpNewLike,        1,  0, -1, -1}}, // new(unsigned int)
    {LibFunc_ZnwjRKSt9nothrow_t,                {MallocLike,       2,  0, -1, -1}}, // new(unsigned int, nothrow)
    {LibFunc_ZnwjSt11align_val_t,               {OpNewLike,        2,  0, -1,  1}}, // new(unsigned int, align_val_t)
    {LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t, {MallocLike,       3,  0, -1,  1}}, // new(unsigned int, align_val_t, nothrow)
    {LibFunc_Znwm,                              {OpNewLike,        1,  0, -1, -1}}, // new(unsigned long)
    {LibFunc_ZnwmRKSt9nothrow_t,                {MallocLike,       2,  0, -1, -1}}, // new(unsigned long, nothrow)
    {LibFunc_ZnwmSt11align_val_t,               {OpNewLike,        2,  0, -1,  1}}, // new(unsigned long, align_val_t)
    {LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t, {MallocLike,       3,  0, -1,  1}}, // new(unsigned long, align_val_t, nothrow)
    {LibFunc_Znaj,                              {OpNewLike,        1,  0, -1, -1}}, // new[](unsigned int)
    {LibFunc_ZnajRKSt9nothrow_t,                {MallocLike,       2,  0, -1, -1}}, // new[](unsigned int, nothrow)
    {LibFunc_ZnajSt11align_val_t,               {OpNewLike,        2,  0, -1,  1}}, // new[](unsigned int, align_val_t)
    {LibFunc_ZnajSt11align_val_tRKSt9nothrow_t, {MallocLike,       3,  0, -1,  1}}, // new[](unsigned int, align_val_t, nothrow)
    {LibFunc_Znam,                              {OpNewLike,        1,  0, -1, -1}}, // new[](unsigned long)
    {LibFunc_ZnamRKSt9nothrow_t,                {MallocLike,       2,  0, -1, -1}}, // new[](unsigned long, nothrow)
    {LibFunc_ZnamSt11align_val_t,               {OpNewLike,        2,  0, -1,  1}}, // new[](unsigned long, align_val_t)
    {LibFunc_ZnamSt11align_val_tRKSt9nothrow_t, {MallocLike,       3,  0, -1,  1}}, // new[](unsigned long, align_val_t, nothrow)
    {LibFunc_msvc_new_int,                      {OpNewLike,        1,  0, -1, -1}}, // new(unsigned int)
    {LibFunc_msvc_new_int_nothrow,              {MallocLike,       2,  0, -1, -1}}, // new(unsigned int, nothrow)
    {LibFunc_msvc_new_longlong,                 {OpNewLike,        1,  0, -1, -1}}, // new(unsigned long long)
    {LibFunc_msvc_new_longlong_nothrow,         {MallocLike,       2,  0, -1, -1}}, // new(unsigned long long, nothrow)
    {LibFunc_msvc_new_array_int,                {OpNewLike,        1,  0, -1, -1}}, // new[](unsigned int)
    {LibFunc_msvc_new_array_int_nothrow,        {MallocLike,       2,  0, -1, -1}}, // new[](unsigned int, nothrow)
    {LibFunc_msvc_new_array_longlong,           {OpNewLike,        1,  0, -1, -1}}, // new[](unsigned long long)
    {LibFunc_msvc_new_array_longlong_nothrow,   {MallocLike,       2,  0, -1, -1}}, // new[](unsigned long long, nothrow)
    {LibFunc_aligned_alloc,                     {AlignedAllocLike, 2,  1, -1,  0}},
    {LibFunc_memalign,                          {AlignedAllocLike, 2,  1, -1,  0}},
    {LibFunc_calloc,                            {CallocLike,       2,  0,  1, -1}},
    {LibFunc_vec_calloc,                        {CallocLike,       2,  0,  1, -1}},
    {LibFunc_realloc,                           {ReallocLike,      2,  1, -1, -1}},
    {LibFunc_vec_realloc,                       {ReallocLike,      2,  1, -1, -1}},
    {LibFunc_reallocf,                          {ReallocLike,      2,  1, -1, -1}},
    {LibFunc_strdup,                            {StrDupLike,       1, -1, -1, -1}},
    {LibFunc_strndup,                           {StrDupLike,       2,  1, -1, -1}},
    {LibFunc___kmpc_alloc_shared,               {MallocLike,       1,  0, -1, -1}},
    // TODO: Handle "int posix_memalign(void **, size_t, size_t)"
};

static const Function *getCalledFunction(const Value *V,
                                         bool &IsNoBuiltin) {
  // Don't care about intrinsics in this case.
  if (isa<IntrinsicInst>(V))
    return nullptr;

  const auto *CB = dyn_cast<CallBase>(V);
  if (!CB)
    return nullptr;

  IsNoBuiltin = CB->isNoBuiltin();

  if (const Function *Callee = CB->getCalledFunction())
    return Callee;
  return nullptr;
}

/// Returns the allocation data for the given value if it's a call to a known
/// allocation function.
static Optional<AllocFnsTy>
getAllocationDataForFunction(const Function *Callee, AllocType AllocTy,
                             const TargetLibraryInfo *TLI) {
  // Make sure that the function is available.
  LibFunc TLIFn;
  if (!TLI || !TLI->getLibFunc(*Callee, TLIFn) || !TLI->has(TLIFn))
    return None;

  const auto *Iter = find_if(
      AllocationFnData, [TLIFn](const std::pair<LibFunc, AllocFnsTy> &P) {
        return P.first == TLIFn;
      });

  if (Iter == std::end(AllocationFnData))
    return None;

  const AllocFnsTy *FnData = &Iter->second;
  if ((FnData->AllocTy & AllocTy) != FnData->AllocTy)
    return None;

  // Check function prototype.
  int FstParam = FnData->FstParam;
  int SndParam = FnData->SndParam;
  FunctionType *FTy = Callee->getFunctionType();

  if (FTy->getReturnType() == Type::getInt8PtrTy(FTy->getContext()) &&
      FTy->getNumParams() == FnData->NumParams &&
      (FstParam < 0 ||
       (FTy->getParamType(FstParam)->isIntegerTy(32) ||
        FTy->getParamType(FstParam)->isIntegerTy(64))) &&
      (SndParam < 0 ||
       FTy->getParamType(SndParam)->isIntegerTy(32) ||
       FTy->getParamType(SndParam)->isIntegerTy(64)))
    return *FnData;
  return None;
}

static Optional<AllocFnsTy> getAllocationData(const Value *V, AllocType AllocTy,
                                              const TargetLibraryInfo *TLI) {
  bool IsNoBuiltinCall;
  if (const Function *Callee = getCalledFunction(V, IsNoBuiltinCall))
    if (!IsNoBuiltinCall)
      return getAllocationDataForFunction(Callee, AllocTy, TLI);
  return None;
}

static Optional<AllocFnsTy>
getAllocationData(const Value *V, AllocType AllocTy,
                  function_ref<const TargetLibraryInfo &(Function &)> GetTLI) {
  bool IsNoBuiltinCall;
  if (const Function *Callee = getCalledFunction(V, IsNoBuiltinCall))
    if (!IsNoBuiltinCall)
      return getAllocationDataForFunction(
          Callee, AllocTy, &GetTLI(const_cast<Function &>(*Callee)));
  return None;
}

static Optional<AllocFnsTy> getAllocationSize(const Value *V,
                                              const TargetLibraryInfo *TLI) {
  bool IsNoBuiltinCall;
  const Function *Callee =
      getCalledFunction(V, IsNoBuiltinCall);
  if (!Callee)
    return None;

  // Prefer to use existing information over allocsize. This will give us an
  // accurate AllocTy.
  if (!IsNoBuiltinCall)
    if (Optional<AllocFnsTy> Data =
            getAllocationDataForFunction(Callee, AnyAlloc, TLI))
      return Data;

  Attribute Attr = Callee->getFnAttribute(Attribute::AllocSize);
  if (Attr == Attribute())
    return None;

  std::pair<unsigned, Optional<unsigned>> Args = Attr.getAllocSizeArgs();

  AllocFnsTy Result;
  // Because allocsize only tells us how many bytes are allocated, we're not
  // really allowed to assume anything, so we use MallocLike.
  Result.AllocTy = MallocLike;
  Result.NumParams = Callee->getNumOperands();
  Result.FstParam = Args.first;
  Result.SndParam = Args.second.getValueOr(-1);
  // Allocsize has no way to specify an alignment argument
  Result.AlignParam = -1;
  return Result;
}

/// Tests if a value is a call or invoke to a library function that
/// allocates or reallocates memory (either malloc, calloc, realloc, or strdup
/// like).
bool llvm::isAllocationFn(const Value *V, const TargetLibraryInfo *TLI) {
  return getAllocationData(V, AnyAlloc, TLI).hasValue();
}
bool llvm::isAllocationFn(
    const Value *V, function_ref<const TargetLibraryInfo &(Function &)> GetTLI) {
  return getAllocationData(V, AnyAlloc, GetTLI).hasValue();
}

/// Tests if a value is a call or invoke to a library function that
/// allocates uninitialized memory (such as malloc).
static bool isMallocLikeFn(const Value *V, const TargetLibraryInfo *TLI) {
  return getAllocationData(V, MallocOrOpNewLike, TLI).hasValue();
}

/// Tests if a value is a call or invoke to a library function that
/// allocates uninitialized memory with alignment (such as aligned_alloc).
static bool isAlignedAllocLikeFn(const Value *V, const TargetLibraryInfo *TLI) {
  return getAllocationData(V, AlignedAllocLike, TLI)
      .hasValue();
}

/// Tests if a value is a call or invoke to a library function that
/// allocates zero-filled memory (such as calloc).
static bool isCallocLikeFn(const Value *V, const TargetLibraryInfo *TLI) {
  return getAllocationData(V, CallocLike, TLI).hasValue();
}

/// Tests if a value is a call or invoke to a library function that
/// allocates memory similar to malloc or calloc.
bool llvm::isMallocOrCallocLikeFn(const Value *V, const TargetLibraryInfo *TLI) {
  return getAllocationData(V, MallocOrCallocLike, TLI).hasValue();
}

/// Tests if a value is a call or invoke to a library function that
/// allocates memory (either malloc, calloc, or strdup like).
bool llvm::isAllocLikeFn(const Value *V, const TargetLibraryInfo *TLI) {
  return getAllocationData(V, AllocLike, TLI).hasValue();
}

/// Tests if a value is a call or invoke to a library function that
/// reallocates memory (e.g., realloc).
bool llvm::isReallocLikeFn(const Value *V, const TargetLibraryInfo *TLI) {
  return getAllocationData(V, ReallocLike, TLI).hasValue();
}

/// Tests if a functions is a call or invoke to a library function that
/// reallocates memory (e.g., realloc).
bool llvm::isReallocLikeFn(const Function *F, const TargetLibraryInfo *TLI) {
  return getAllocationDataForFunction(F, ReallocLike, TLI).hasValue();
}

bool llvm::isAllocRemovable(const CallBase *CB, const TargetLibraryInfo *TLI) {
  assert(isAllocationFn(CB, TLI));

  // Note: Removability is highly dependent on the source language.  For
  // example, recent C++ requires direct calls to the global allocation
  // [basic.stc.dynamic.allocation] to be observable unless part of a new
  // expression [expr.new paragraph 13].

  // Historically we've treated the C family allocation routines as removable
  return isAllocLikeFn(CB, TLI);
}

Value *llvm::getAllocAlignment(const CallBase *V,
                               const TargetLibraryInfo *TLI) {
  assert(isAllocationFn(V, TLI));

  const Optional<AllocFnsTy> FnData = getAllocationData(V, AnyAlloc, TLI);
  if (!FnData.hasValue() || FnData->AlignParam < 0) {
    return nullptr;
  }
  return V->getOperand(FnData->AlignParam);
}

/// When we're compiling N-bit code, and the user uses parameters that are
/// greater than N bits (e.g. uint64_t on a 32-bit build), we can run into
/// trouble with APInt size issues. This function handles resizing + overflow
/// checks for us. Check and zext or trunc \p I depending on IntTyBits and
/// I's value.
static bool CheckedZextOrTrunc(APInt &I, unsigned IntTyBits) {
  // More bits than we can handle. Checking the bit width isn't necessary, but
  // it's faster than checking active bits, and should give `false` in the
  // vast majority of cases.
  if (I.getBitWidth() > IntTyBits && I.getActiveBits() > IntTyBits)
    return false;
  if (I.getBitWidth() != IntTyBits)
    I = I.zextOrTrunc(IntTyBits);
  return true;
}

Optional<APInt>
llvm::getAllocSize(const CallBase *CB,
                   const TargetLibraryInfo *TLI,
                   std::function<const Value*(const Value*)> Mapper) {
  // Note: This handles both explicitly listed allocation functions and
  // allocsize.  The code structure could stand to be cleaned up a bit.
  Optional<AllocFnsTy> FnData = getAllocationSize(CB, TLI);
  if (!FnData)
    return None;

  // Get the index type for this address space, results and intermediate
  // computations are performed at that width.
  auto &DL = CB->getModule()->getDataLayout();
  const unsigned IntTyBits = DL.getIndexTypeSizeInBits(CB->getType());

  // Handle strdup-like functions separately.
  if (FnData->AllocTy == StrDupLike) {
    APInt Size(IntTyBits, GetStringLength(Mapper(CB->getArgOperand(0))));
    if (!Size)
      return None;

    // Strndup limits strlen.
    if (FnData->FstParam > 0) {
      const ConstantInt *Arg =
        dyn_cast<ConstantInt>(Mapper(CB->getArgOperand(FnData->FstParam)));
      if (!Arg)
        return None;

      APInt MaxSize = Arg->getValue().zextOrSelf(IntTyBits);
      if (Size.ugt(MaxSize))
        Size = MaxSize + 1;
    }
    return Size;
  }

  const ConstantInt *Arg =
    dyn_cast<ConstantInt>(Mapper(CB->getArgOperand(FnData->FstParam)));
  if (!Arg)
    return None;

  APInt Size = Arg->getValue();
  if (!CheckedZextOrTrunc(Size, IntTyBits))
    return None;

  // Size is determined by just 1 parameter.
  if (FnData->SndParam < 0)
    return Size;

  Arg = dyn_cast<ConstantInt>(Mapper(CB->getArgOperand(FnData->SndParam)));
  if (!Arg)
    return None;

  APInt NumElems = Arg->getValue();
  if (!CheckedZextOrTrunc(NumElems, IntTyBits))
    return None;

  bool Overflow;
  Size = Size.umul_ov(NumElems, Overflow);
  if (Overflow)
    return None;
  return Size;
}

Constant *llvm::getInitialValueOfAllocation(const CallBase *Alloc,
                                            const TargetLibraryInfo *TLI,
                                            Type *Ty) {
  assert(isAllocationFn(Alloc, TLI));

  // malloc and aligned_alloc are uninitialized (undef)
  if (isMallocLikeFn(Alloc, TLI) || isAlignedAllocLikeFn(Alloc, TLI))
    return UndefValue::get(Ty);

  // calloc zero initializes
  if (isCallocLikeFn(Alloc, TLI))
    return Constant::getNullValue(Ty);

  return nullptr;
}

/// isLibFreeFunction - Returns true if the function is a builtin free()
bool llvm::isLibFreeFunction(const Function *F, const LibFunc TLIFn) {
  unsigned ExpectedNumParams;
  if (TLIFn == LibFunc_free ||
      TLIFn == LibFunc_ZdlPv || // operator delete(void*)
      TLIFn == LibFunc_ZdaPv || // operator delete[](void*)
      TLIFn == LibFunc_msvc_delete_ptr32 || // operator delete(void*)
      TLIFn == LibFunc_msvc_delete_ptr64 || // operator delete(void*)
      TLIFn == LibFunc_msvc_delete_array_ptr32 || // operator delete[](void*)
      TLIFn == LibFunc_msvc_delete_array_ptr64)   // operator delete[](void*)
    ExpectedNumParams = 1;
  else if (TLIFn == LibFunc_ZdlPvj ||              // delete(void*, uint)
           TLIFn == LibFunc_ZdlPvm ||              // delete(void*, ulong)
           TLIFn == LibFunc_ZdlPvRKSt9nothrow_t || // delete(void*, nothrow)
           TLIFn == LibFunc_ZdlPvSt11align_val_t || // delete(void*, align_val_t)
           TLIFn == LibFunc_ZdaPvj ||              // delete[](void*, uint)
           TLIFn == LibFunc_ZdaPvm ||              // delete[](void*, ulong)
           TLIFn == LibFunc_ZdaPvRKSt9nothrow_t || // delete[](void*, nothrow)
           TLIFn == LibFunc_ZdaPvSt11align_val_t || // delete[](void*, align_val_t)
           TLIFn == LibFunc_msvc_delete_ptr32_int ||      // delete(void*, uint)
           TLIFn == LibFunc_msvc_delete_ptr64_longlong || // delete(void*, ulonglong)
           TLIFn == LibFunc_msvc_delete_ptr32_nothrow || // delete(void*, nothrow)
           TLIFn == LibFunc_msvc_delete_ptr64_nothrow || // delete(void*, nothrow)
           TLIFn == LibFunc_msvc_delete_array_ptr32_int ||      // delete[](void*, uint)
           TLIFn == LibFunc_msvc_delete_array_ptr64_longlong || // delete[](void*, ulonglong)
           TLIFn == LibFunc_msvc_delete_array_ptr32_nothrow || // delete[](void*, nothrow)
           TLIFn == LibFunc_msvc_delete_array_ptr64_nothrow || // delete[](void*, nothrow)
           TLIFn == LibFunc___kmpc_free_shared) // OpenMP Offloading RTL free
    ExpectedNumParams = 2;
  else if (TLIFn == LibFunc_ZdaPvSt11align_val_tRKSt9nothrow_t || // delete(void*, align_val_t, nothrow)
           TLIFn == LibFunc_ZdlPvSt11align_val_tRKSt9nothrow_t || // delete[](void*, align_val_t, nothrow)
           TLIFn == LibFunc_ZdlPvjSt11align_val_t || // delete(void*, unsigned long, align_val_t)
           TLIFn == LibFunc_ZdlPvmSt11align_val_t || // delete(void*, unsigned long, align_val_t)
           TLIFn == LibFunc_ZdaPvjSt11align_val_t || // delete[](void*, unsigned int, align_val_t)
           TLIFn == LibFunc_ZdaPvmSt11align_val_t) // delete[](void*, unsigned long, align_val_t)
    ExpectedNumParams = 3;
  else
    return false;

  // Check free prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin
  // attribute will exist.
  FunctionType *FTy = F->getFunctionType();
  if (!FTy->getReturnType()->isVoidTy())
    return false;
  if (FTy->getNumParams() != ExpectedNumParams)
    return false;
  if (FTy->getParamType(0) != Type::getInt8PtrTy(F->getContext()))
    return false;

  return true;
}

/// isFreeCall - Returns non-null if the value is a call to the builtin free()
const CallInst *llvm::isFreeCall(const Value *I, const TargetLibraryInfo *TLI) {
  bool IsNoBuiltinCall;
  const Function *Callee = getCalledFunction(I, IsNoBuiltinCall);
  if (Callee == nullptr || IsNoBuiltinCall)
    return nullptr;

  LibFunc TLIFn;
  if (!TLI || !TLI->getLibFunc(*Callee, TLIFn) || !TLI->has(TLIFn))
    return nullptr;

  return isLibFreeFunction(Callee, TLIFn) ? dyn_cast<CallInst>(I) : nullptr;
}


//===----------------------------------------------------------------------===//
//  Utility functions to compute size of objects.
//
static APInt getSizeWithOverflow(const SizeOffsetType &Data) {
  if (Data.second.isNegative() || Data.first.ult(Data.second))
    return APInt(Data.first.getBitWidth(), 0);
  return Data.first - Data.second;
}

/// Compute the size of the object pointed by Ptr. Returns true and the
/// object size in Size if successful, and false otherwise.
/// If RoundToAlign is true, then Size is rounded up to the alignment of
/// allocas, byval arguments, and global variables.
bool llvm::getObjectSize(const Value *Ptr, uint64_t &Size, const DataLayout &DL,
                         const TargetLibraryInfo *TLI, ObjectSizeOpts Opts) {
  ObjectSizeOffsetVisitor Visitor(DL, TLI, Ptr->getContext(), Opts);
  SizeOffsetType Data = Visitor.compute(const_cast<Value*>(Ptr));
  if (!Visitor.bothKnown(Data))
    return false;

  Size = getSizeWithOverflow(Data).getZExtValue();
  return true;
}

Value *llvm::lowerObjectSizeCall(IntrinsicInst *ObjectSize,
                                 const DataLayout &DL,
                                 const TargetLibraryInfo *TLI,
                                 bool MustSucceed) {
  assert(ObjectSize->getIntrinsicID() == Intrinsic::objectsize &&
         "ObjectSize must be a call to llvm.objectsize!");

  bool MaxVal = cast<ConstantInt>(ObjectSize->getArgOperand(1))->isZero();
  ObjectSizeOpts EvalOptions;
  // Unless we have to fold this to something, try to be as accurate as
  // possible.
  if (MustSucceed)
    EvalOptions.EvalMode =
        MaxVal ? ObjectSizeOpts::Mode::Max : ObjectSizeOpts::Mode::Min;
  else
    EvalOptions.EvalMode = ObjectSizeOpts::Mode::Exact;

  EvalOptions.NullIsUnknownSize =
      cast<ConstantInt>(ObjectSize->getArgOperand(2))->isOne();

  auto *ResultType = cast<IntegerType>(ObjectSize->getType());
  bool StaticOnly = cast<ConstantInt>(ObjectSize->getArgOperand(3))->isZero();
  if (StaticOnly) {
    // FIXME: Does it make sense to just return a failure value if the size won't
    // fit in the output and `!MustSucceed`?
    uint64_t Size;
    if (getObjectSize(ObjectSize->getArgOperand(0), Size, DL, TLI, EvalOptions) &&
        isUIntN(ResultType->getBitWidth(), Size))
      return ConstantInt::get(ResultType, Size);
  } else {
    LLVMContext &Ctx = ObjectSize->getFunction()->getContext();
    ObjectSizeOffsetEvaluator Eval(DL, TLI, Ctx, EvalOptions);
    SizeOffsetEvalType SizeOffsetPair =
        Eval.compute(ObjectSize->getArgOperand(0));

    if (SizeOffsetPair != ObjectSizeOffsetEvaluator::unknown()) {
      IRBuilder<TargetFolder> Builder(Ctx, TargetFolder(DL));
      Builder.SetInsertPoint(ObjectSize);

      // If we've outside the end of the object, then we can always access
      // exactly 0 bytes.
      Value *ResultSize =
          Builder.CreateSub(SizeOffsetPair.first, SizeOffsetPair.second);
      Value *UseZero =
          Builder.CreateICmpULT(SizeOffsetPair.first, SizeOffsetPair.second);
      ResultSize = Builder.CreateZExtOrTrunc(ResultSize, ResultType);
      Value *Ret = Builder.CreateSelect(
          UseZero, ConstantInt::get(ResultType, 0), ResultSize);

      // The non-constant size expression cannot evaluate to -1.
      if (!isa<Constant>(SizeOffsetPair.first) ||
          !isa<Constant>(SizeOffsetPair.second))
        Builder.CreateAssumption(
            Builder.CreateICmpNE(Ret, ConstantInt::get(ResultType, -1)));

      return Ret;
    }
  }

  if (!MustSucceed)
    return nullptr;

  return ConstantInt::get(ResultType, MaxVal ? -1ULL : 0);
}

STATISTIC(ObjectVisitorArgument,
          "Number of arguments with unsolved size and offset");
STATISTIC(ObjectVisitorLoad,
          "Number of load instructions with unsolved size and offset");

APInt ObjectSizeOffsetVisitor::align(APInt Size, MaybeAlign Alignment) {
  if (Options.RoundToAlign && Alignment)
    return APInt(IntTyBits, alignTo(Size.getZExtValue(), Alignment));
  return Size;
}

ObjectSizeOffsetVisitor::ObjectSizeOffsetVisitor(const DataLayout &DL,
                                                 const TargetLibraryInfo *TLI,
                                                 LLVMContext &Context,
                                                 ObjectSizeOpts Options)
    : DL(DL), TLI(TLI), Options(Options) {
  // Pointer size must be rechecked for each object visited since it could have
  // a different address space.
}

SizeOffsetType ObjectSizeOffsetVisitor::compute(Value *V) {
  IntTyBits = DL.getIndexTypeSizeInBits(V->getType());
  Zero = APInt::getZero(IntTyBits);

  V = V->stripPointerCasts();
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If we have already seen this instruction, bail out. Cycles can happen in
    // unreachable code after constant propagation.
    if (!SeenInsts.insert(I).second)
      return unknown();

    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V))
      return visitGEPOperator(*GEP);
    return visit(*I);
  }
  if (Argument *A = dyn_cast<Argument>(V))
    return visitArgument(*A);
  if (ConstantPointerNull *P = dyn_cast<ConstantPointerNull>(V))
    return visitConstantPointerNull(*P);
  if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return visitGlobalAlias(*GA);
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return visitGlobalVariable(*GV);
  if (UndefValue *UV = dyn_cast<UndefValue>(V))
    return visitUndefValue(*UV);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::IntToPtr)
      return unknown(); // clueless
    if (CE->getOpcode() == Instruction::GetElementPtr)
      return visitGEPOperator(cast<GEPOperator>(*CE));
  }

  LLVM_DEBUG(dbgs() << "ObjectSizeOffsetVisitor::compute() unhandled value: "
                    << *V << '\n');
  return unknown();
}

bool ObjectSizeOffsetVisitor::CheckedZextOrTrunc(APInt &I) {
  return ::CheckedZextOrTrunc(I, IntTyBits);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitAllocaInst(AllocaInst &I) {
  if (!I.getAllocatedType()->isSized())
    return unknown();

  if (isa<ScalableVectorType>(I.getAllocatedType()))
    return unknown();

  APInt Size(IntTyBits, DL.getTypeAllocSize(I.getAllocatedType()));
  if (!I.isArrayAllocation())
    return std::make_pair(align(Size, I.getAlign()), Zero);

  Value *ArraySize = I.getArraySize();
  if (const ConstantInt *C = dyn_cast<ConstantInt>(ArraySize)) {
    APInt NumElems = C->getValue();
    if (!CheckedZextOrTrunc(NumElems))
      return unknown();

    bool Overflow;
    Size = Size.umul_ov(NumElems, Overflow);
    return Overflow ? unknown()
                    : std::make_pair(align(Size, I.getAlign()), Zero);
  }
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitArgument(Argument &A) {
  Type *MemoryTy = A.getPointeeInMemoryValueType();
  // No interprocedural analysis is done at the moment.
  if (!MemoryTy|| !MemoryTy->isSized()) {
    ++ObjectVisitorArgument;
    return unknown();
  }

  APInt Size(IntTyBits, DL.getTypeAllocSize(MemoryTy));
  return std::make_pair(align(Size, A.getParamAlign()), Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitCallBase(CallBase &CB) {
  auto Mapper = [](const Value *V) { return V; };
  if (Optional<APInt> Size = getAllocSize(&CB, TLI, Mapper))
    return std::make_pair(*Size, Zero);
  return unknown();
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitConstantPointerNull(ConstantPointerNull& CPN) {
  // If null is unknown, there's nothing we can do. Additionally, non-zero
  // address spaces can make use of null, so we don't presume to know anything
  // about that.
  //
  // TODO: How should this work with address space casts? We currently just drop
  // them on the floor, but it's unclear what we should do when a NULL from
  // addrspace(1) gets casted to addrspace(0) (or vice-versa).
  if (Options.NullIsUnknownSize || CPN.getType()->getAddressSpace())
    return unknown();
  return std::make_pair(Zero, Zero);
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitExtractElementInst(ExtractElementInst&) {
  return unknown();
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitExtractValueInst(ExtractValueInst&) {
  // Easy cases were already folded by previous passes.
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitGEPOperator(GEPOperator &GEP) {
  SizeOffsetType PtrData = compute(GEP.getPointerOperand());
  APInt Offset(DL.getIndexTypeSizeInBits(GEP.getPointerOperand()->getType()), 0);
  if (!bothKnown(PtrData) || !GEP.accumulateConstantOffset(DL, Offset))
    return unknown();

  return std::make_pair(PtrData.first, PtrData.second + Offset);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitGlobalAlias(GlobalAlias &GA) {
  if (GA.isInterposable())
    return unknown();
  return compute(GA.getAliasee());
}

SizeOffsetType ObjectSizeOffsetVisitor::visitGlobalVariable(GlobalVariable &GV){
  if (!GV.hasDefinitiveInitializer())
    return unknown();

  APInt Size(IntTyBits, DL.getTypeAllocSize(GV.getValueType()));
  return std::make_pair(align(Size, GV.getAlign()), Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitIntToPtrInst(IntToPtrInst&) {
  // clueless
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitLoadInst(LoadInst&) {
  ++ObjectVisitorLoad;
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitPHINode(PHINode&) {
  // too complex to analyze statically.
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitSelectInst(SelectInst &I) {
  SizeOffsetType TrueSide  = compute(I.getTrueValue());
  SizeOffsetType FalseSide = compute(I.getFalseValue());
  if (bothKnown(TrueSide) && bothKnown(FalseSide)) {
    if (TrueSide == FalseSide) {
        return TrueSide;
    }

    APInt TrueResult = getSizeWithOverflow(TrueSide);
    APInt FalseResult = getSizeWithOverflow(FalseSide);

    if (TrueResult == FalseResult) {
      return TrueSide;
    }
    if (Options.EvalMode == ObjectSizeOpts::Mode::Min) {
      if (TrueResult.slt(FalseResult))
        return TrueSide;
      return FalseSide;
    }
    if (Options.EvalMode == ObjectSizeOpts::Mode::Max) {
      if (TrueResult.sgt(FalseResult))
        return TrueSide;
      return FalseSide;
    }
  }
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitUndefValue(UndefValue&) {
  return std::make_pair(Zero, Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitInstruction(Instruction &I) {
  LLVM_DEBUG(dbgs() << "ObjectSizeOffsetVisitor unknown instruction:" << I
                    << '\n');
  return unknown();
}

ObjectSizeOffsetEvaluator::ObjectSizeOffsetEvaluator(
    const DataLayout &DL, const TargetLibraryInfo *TLI, LLVMContext &Context,
    ObjectSizeOpts EvalOpts)
    : DL(DL), TLI(TLI), Context(Context),
      Builder(Context, TargetFolder(DL),
              IRBuilderCallbackInserter(
                  [&](Instruction *I) { InsertedInstructions.insert(I); })),
      EvalOpts(EvalOpts) {
  // IntTy and Zero must be set for each compute() since the address space may
  // be different for later objects.
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::compute(Value *V) {
  // XXX - Are vectors of pointers possible here?
  IntTy = cast<IntegerType>(DL.getIndexType(V->getType()));
  Zero = ConstantInt::get(IntTy, 0);

  SizeOffsetEvalType Result = compute_(V);

  if (!bothKnown(Result)) {
    // Erase everything that was computed in this iteration from the cache, so
    // that no dangling references are left behind. We could be a bit smarter if
    // we kept a dependency graph. It's probably not worth the complexity.
    for (const Value *SeenVal : SeenVals) {
      CacheMapTy::iterator CacheIt = CacheMap.find(SeenVal);
      // non-computable results can be safely cached
      if (CacheIt != CacheMap.end() && anyKnown(CacheIt->second))
        CacheMap.erase(CacheIt);
    }

    // Erase any instructions we inserted as part of the traversal.
    for (Instruction *I : InsertedInstructions) {
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }
  }

  SeenVals.clear();
  InsertedInstructions.clear();
  return Result;
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::compute_(Value *V) {
  ObjectSizeOffsetVisitor Visitor(DL, TLI, Context, EvalOpts);
  SizeOffsetType Const = Visitor.compute(V);
  if (Visitor.bothKnown(Const))
    return std::make_pair(ConstantInt::get(Context, Const.first),
                          ConstantInt::get(Context, Const.second));

  V = V->stripPointerCasts();

  // Check cache.
  CacheMapTy::iterator CacheIt = CacheMap.find(V);
  if (CacheIt != CacheMap.end())
    return CacheIt->second;

  // Always generate code immediately before the instruction being
  // processed, so that the generated code dominates the same BBs.
  BuilderTy::InsertPointGuard Guard(Builder);
  if (Instruction *I = dyn_cast<Instruction>(V))
    Builder.SetInsertPoint(I);

  // Now compute the size and offset.
  SizeOffsetEvalType Result;

  // Record the pointers that were handled in this run, so that they can be
  // cleaned later if something fails. We also use this set to break cycles that
  // can occur in dead code.
  if (!SeenVals.insert(V).second) {
    Result = unknown();
  } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    Result = visitGEPOperator(*GEP);
  } else if (Instruction *I = dyn_cast<Instruction>(V)) {
    Result = visit(*I);
  } else if (isa<Argument>(V) ||
             (isa<ConstantExpr>(V) &&
              cast<ConstantExpr>(V)->getOpcode() == Instruction::IntToPtr) ||
             isa<GlobalAlias>(V) ||
             isa<GlobalVariable>(V)) {
    // Ignore values where we cannot do more than ObjectSizeVisitor.
    Result = unknown();
  } else {
    LLVM_DEBUG(
        dbgs() << "ObjectSizeOffsetEvaluator::compute() unhandled value: " << *V
               << '\n');
    Result = unknown();
  }

  // Don't reuse CacheIt since it may be invalid at this point.
  CacheMap[V] = Result;
  return Result;
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitAllocaInst(AllocaInst &I) {
  if (!I.getAllocatedType()->isSized())
    return unknown();

  // must be a VLA
  assert(I.isArrayAllocation());

  // If needed, adjust the alloca's operand size to match the pointer size.
  // Subsequent math operations expect the types to match.
  Value *ArraySize = Builder.CreateZExtOrTrunc(
      I.getArraySize(), DL.getIntPtrType(I.getContext()));
  assert(ArraySize->getType() == Zero->getType() &&
         "Expected zero constant to have pointer type");

  Value *Size = ConstantInt::get(ArraySize->getType(),
                                 DL.getTypeAllocSize(I.getAllocatedType()));
  Size = Builder.CreateMul(Size, ArraySize);
  return std::make_pair(Size, Zero);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitCallBase(CallBase &CB) {
  Optional<AllocFnsTy> FnData = getAllocationSize(&CB, TLI);
  if (!FnData)
    return unknown();

  // Handle strdup-like functions separately.
  if (FnData->AllocTy == StrDupLike) {
    // TODO: implement evaluation of strdup/strndup
    return unknown();
  }

  Value *FirstArg = CB.getArgOperand(FnData->FstParam);
  FirstArg = Builder.CreateZExtOrTrunc(FirstArg, IntTy);
  if (FnData->SndParam < 0)
    return std::make_pair(FirstArg, Zero);

  Value *SecondArg = CB.getArgOperand(FnData->SndParam);
  SecondArg = Builder.CreateZExtOrTrunc(SecondArg, IntTy);
  Value *Size = Builder.CreateMul(FirstArg, SecondArg);
  return std::make_pair(Size, Zero);
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitExtractElementInst(ExtractElementInst&) {
  return unknown();
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitExtractValueInst(ExtractValueInst&) {
  return unknown();
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitGEPOperator(GEPOperator &GEP) {
  SizeOffsetEvalType PtrData = compute_(GEP.getPointerOperand());
  if (!bothKnown(PtrData))
    return unknown();

  Value *Offset = EmitGEPOffset(&Builder, DL, &GEP, /*NoAssumptions=*/true);
  Offset = Builder.CreateAdd(PtrData.second, Offset);
  return std::make_pair(PtrData.first, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitIntToPtrInst(IntToPtrInst&) {
  // clueless
  return unknown();
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitLoadInst(LoadInst&) {
  return unknown();
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitPHINode(PHINode &PHI) {
  // Create 2 PHIs: one for size and another for offset.
  PHINode *SizePHI   = Builder.CreatePHI(IntTy, PHI.getNumIncomingValues());
  PHINode *OffsetPHI = Builder.CreatePHI(IntTy, PHI.getNumIncomingValues());

  // Insert right away in the cache to handle recursive PHIs.
  CacheMap[&PHI] = std::make_pair(SizePHI, OffsetPHI);

  // Compute offset/size for each PHI incoming pointer.
  for (unsigned i = 0, e = PHI.getNumIncomingValues(); i != e; ++i) {
    Builder.SetInsertPoint(&*PHI.getIncomingBlock(i)->getFirstInsertionPt());
    SizeOffsetEvalType EdgeData = compute_(PHI.getIncomingValue(i));

    if (!bothKnown(EdgeData)) {
      OffsetPHI->replaceAllUsesWith(UndefValue::get(IntTy));
      OffsetPHI->eraseFromParent();
      InsertedInstructions.erase(OffsetPHI);
      SizePHI->replaceAllUsesWith(UndefValue::get(IntTy));
      SizePHI->eraseFromParent();
      InsertedInstructions.erase(SizePHI);
      return unknown();
    }
    SizePHI->addIncoming(EdgeData.first, PHI.getIncomingBlock(i));
    OffsetPHI->addIncoming(EdgeData.second, PHI.getIncomingBlock(i));
  }

  Value *Size = SizePHI, *Offset = OffsetPHI;
  if (Value *Tmp = SizePHI->hasConstantValue()) {
    Size = Tmp;
    SizePHI->replaceAllUsesWith(Size);
    SizePHI->eraseFromParent();
    InsertedInstructions.erase(SizePHI);
  }
  if (Value *Tmp = OffsetPHI->hasConstantValue()) {
    Offset = Tmp;
    OffsetPHI->replaceAllUsesWith(Offset);
    OffsetPHI->eraseFromParent();
    InsertedInstructions.erase(OffsetPHI);
  }
  return std::make_pair(Size, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitSelectInst(SelectInst &I) {
  SizeOffsetEvalType TrueSide  = compute_(I.getTrueValue());
  SizeOffsetEvalType FalseSide = compute_(I.getFalseValue());

  if (!bothKnown(TrueSide) || !bothKnown(FalseSide))
    return unknown();
  if (TrueSide == FalseSide)
    return TrueSide;

  Value *Size = Builder.CreateSelect(I.getCondition(), TrueSide.first,
                                     FalseSide.first);
  Value *Offset = Builder.CreateSelect(I.getCondition(), TrueSide.second,
                                       FalseSide.second);
  return std::make_pair(Size, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitInstruction(Instruction &I) {
  LLVM_DEBUG(dbgs() << "ObjectSizeOffsetEvaluator unknown instruction:" << I
                    << '\n');
  return unknown();
}
