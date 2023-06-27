//==- llvm/Analysis/MemoryBuiltins.h - Calls to memory builtins --*- C++ -*-==//
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

#ifndef LLVM_ANALYSIS_MEMORYBUILTINS_H
#define LLVM_ANALYSIS_MEMORYBUILTINS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/ValueHandle.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace llvm {

class AllocaInst;
class AAResults;
class Argument;
class ConstantPointerNull;
class DataLayout;
class ExtractElementInst;
class ExtractValueInst;
class GEPOperator;
class GlobalAlias;
class GlobalVariable;
class Instruction;
class IntegerType;
class IntrinsicInst;
class IntToPtrInst;
class LLVMContext;
class LoadInst;
class PHINode;
class SelectInst;
class Type;
class UndefValue;
class Value;

/// Tests if a value is a call or invoke to a library function that
/// allocates or reallocates memory (either malloc, calloc, realloc, or strdup
/// like).
bool isAllocationFn(const Value *V, const TargetLibraryInfo *TLI);
bool isAllocationFn(const Value *V,
                    function_ref<const TargetLibraryInfo &(Function &)> GetTLI);

/// Tests if a value is a call or invoke to a library function that
/// allocates memory via new.
bool isNewLikeFn(const Value *V, const TargetLibraryInfo *TLI);

/// Tests if a value is a call or invoke to a library function that
/// allocates memory similar to malloc or calloc.
bool isMallocOrCallocLikeFn(const Value *V, const TargetLibraryInfo *TLI);

/// Tests if a value is a call or invoke to a library function that
/// allocates memory (either malloc, calloc, or strdup like).
bool isAllocLikeFn(const Value *V, const TargetLibraryInfo *TLI);

/// Tests if a function is a call or invoke to a library function that
/// reallocates memory (e.g., realloc).
bool isReallocLikeFn(const Function *F);

/// If this is a call to a realloc function, return the reallocated operand.
Value *getReallocatedOperand(const CallBase *CB);

//===----------------------------------------------------------------------===//
//  free Call Utility Functions.
//

/// isLibFreeFunction - Returns true if the function is a builtin free()
bool isLibFreeFunction(const Function *F, const LibFunc TLIFn);

/// If this if a call to a free function, return the freed operand.
Value *getFreedOperand(const CallBase *CB, const TargetLibraryInfo *TLI);

//===----------------------------------------------------------------------===//
//  Properties of allocation functions
//

/// Return true if this is a call to an allocation function that does not have
/// side effects that we are required to preserve beyond the effect of
/// allocating a new object.
/// Ex: If our allocation routine has a counter for the number of objects
/// allocated, and the program prints it on exit, can the value change due
/// to optimization? Answer is highly language dependent.
/// Note: *Removable* really does mean removable; it does not mean observable.
/// A language (e.g. C++) can allow removing allocations without allowing
/// insertion or speculative execution of allocation routines.
bool isRemovableAlloc(const CallBase *V, const TargetLibraryInfo *TLI);

/// Gets the alignment argument for an aligned_alloc-like function, using either
/// built-in knowledge based on fuction names/signatures or allocalign
/// attributes. Note: the Value returned may not indicate a valid alignment, per
/// the definition of the allocalign attribute.
Value *getAllocAlignment(const CallBase *V, const TargetLibraryInfo *TLI);

/// Return the size of the requested allocation. With a trivial mapper, this is
/// similar to calling getObjectSize(..., Exact), but without looking through
/// calls that return their argument. A mapper function can be used to replace
/// one Value* (operand to the allocation) with another. This is useful when
/// doing abstract interpretation.
std::optional<APInt> getAllocSize(
    const CallBase *CB, const TargetLibraryInfo *TLI,
    function_ref<const Value *(const Value *)> Mapper = [](const Value *V) {
      return V;
    });

/// If this is a call to an allocation function that initializes memory to a
/// fixed value, return said value in the requested type.  Otherwise, return
/// nullptr.
Constant *getInitialValueOfAllocation(const Value *V,
                                      const TargetLibraryInfo *TLI,
                                      Type *Ty);

/// If a function is part of an allocation family (e.g.
/// malloc/realloc/calloc/free), return the identifier for its family
/// of functions.
std::optional<StringRef> getAllocationFamily(const Value *I,
                                             const TargetLibraryInfo *TLI);

//===----------------------------------------------------------------------===//
//  Utility functions to compute size of objects.
//

/// Various options to control the behavior of getObjectSize.
struct ObjectSizeOpts {
  /// Controls how we handle conditional statements with unknown conditions.
  enum class Mode : uint8_t {
    /// All branches must be known and have the same size, starting from the
    /// offset, to be merged.
    ExactSizeFromOffset,
    /// All branches must be known and have the same underlying size and offset
    /// to be merged.
    ExactUnderlyingSizeAndOffset,
    /// Evaluate all branches of an unknown condition. If all evaluations
    /// succeed, pick the minimum size.
    Min,
    /// Same as Min, except we pick the maximum size of all of the branches.
    Max,
  };

  /// How we want to evaluate this object's size.
  Mode EvalMode = Mode::ExactSizeFromOffset;
  /// Whether to round the result up to the alignment of allocas, byval
  /// arguments, and global variables.
  bool RoundToAlign = false;
  /// If this is true, null pointers in address space 0 will be treated as
  /// though they can't be evaluated. Otherwise, null is always considered to
  /// point to a 0 byte region of memory.
  bool NullIsUnknownSize = false;
  /// If set, used for more accurate evaluation
  AAResults *AA = nullptr;
};

/// Compute the size of the object pointed by Ptr. Returns true and the
/// object size in Size if successful, and false otherwise. In this context, by
/// object we mean the region of memory starting at Ptr to the end of the
/// underlying object pointed to by Ptr.
///
/// WARNING: The object size returned is the allocation size.  This does not
/// imply dereferenceability at site of use since the object may be freeed in
/// between.
bool getObjectSize(const Value *Ptr, uint64_t &Size, const DataLayout &DL,
                   const TargetLibraryInfo *TLI, ObjectSizeOpts Opts = {});

/// Try to turn a call to \@llvm.objectsize into an integer value of the given
/// Type. Returns null on failure. If MustSucceed is true, this function will
/// not return null, and may return conservative values governed by the second
/// argument of the call to objectsize.
Value *lowerObjectSizeCall(IntrinsicInst *ObjectSize, const DataLayout &DL,
                           const TargetLibraryInfo *TLI, bool MustSucceed);
Value *lowerObjectSizeCall(
    IntrinsicInst *ObjectSize, const DataLayout &DL,
    const TargetLibraryInfo *TLI, AAResults *AA, bool MustSucceed,
    SmallVectorImpl<Instruction *> *InsertedInstructions = nullptr);

using SizeOffsetType = std::pair<APInt, APInt>;

/// Evaluate the size and offset of an object pointed to by a Value*
/// statically. Fails if size or offset are not known at compile time.
class ObjectSizeOffsetVisitor
  : public InstVisitor<ObjectSizeOffsetVisitor, SizeOffsetType> {
  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
  ObjectSizeOpts Options;
  unsigned IntTyBits;
  APInt Zero;
  SmallPtrSet<Instruction *, 8> SeenInsts;

  APInt align(APInt Size, MaybeAlign Align);

  SizeOffsetType unknown() {
    return std::make_pair(APInt(), APInt());
  }

public:
  ObjectSizeOffsetVisitor(const DataLayout &DL, const TargetLibraryInfo *TLI,
                          LLVMContext &Context, ObjectSizeOpts Options = {});

  SizeOffsetType compute(Value *V);

  static bool knownSize(const SizeOffsetType &SizeOffset) {
    return SizeOffset.first.getBitWidth() > 1;
  }

  static bool knownOffset(const SizeOffsetType &SizeOffset) {
    return SizeOffset.second.getBitWidth() > 1;
  }

  static bool bothKnown(const SizeOffsetType &SizeOffset) {
    return knownSize(SizeOffset) && knownOffset(SizeOffset);
  }

  // These are "private", except they can't actually be made private. Only
  // compute() should be used by external users.
  SizeOffsetType visitAllocaInst(AllocaInst &I);
  SizeOffsetType visitArgument(Argument &A);
  SizeOffsetType visitCallBase(CallBase &CB);
  SizeOffsetType visitConstantPointerNull(ConstantPointerNull&);
  SizeOffsetType visitExtractElementInst(ExtractElementInst &I);
  SizeOffsetType visitExtractValueInst(ExtractValueInst &I);
  SizeOffsetType visitGlobalAlias(GlobalAlias &GA);
  SizeOffsetType visitGlobalVariable(GlobalVariable &GV);
  SizeOffsetType visitIntToPtrInst(IntToPtrInst&);
  SizeOffsetType visitLoadInst(LoadInst &I);
  SizeOffsetType visitPHINode(PHINode&);
  SizeOffsetType visitSelectInst(SelectInst &I);
  SizeOffsetType visitUndefValue(UndefValue&);
  SizeOffsetType visitInstruction(Instruction &I);

private:
  SizeOffsetType findLoadSizeOffset(
      LoadInst &LoadFrom, BasicBlock &BB, BasicBlock::iterator From,
      SmallDenseMap<BasicBlock *, SizeOffsetType, 8> &VisitedBlocks,
      unsigned &ScannedInstCount);
  SizeOffsetType combineSizeOffset(SizeOffsetType LHS, SizeOffsetType RHS);
  SizeOffsetType computeImpl(Value *V);
  bool CheckedZextOrTrunc(APInt &I);
};

using SizeOffsetEvalType = std::pair<Value *, Value *>;

/// Evaluate the size and offset of an object pointed to by a Value*.
/// May create code to compute the result at run-time.
class ObjectSizeOffsetEvaluator
  : public InstVisitor<ObjectSizeOffsetEvaluator, SizeOffsetEvalType> {
  using BuilderTy = IRBuilder<TargetFolder, IRBuilderCallbackInserter>;
  using WeakEvalType = std::pair<WeakTrackingVH, WeakTrackingVH>;
  using CacheMapTy = DenseMap<const Value *, WeakEvalType>;
  using PtrSetTy = SmallPtrSet<const Value *, 8>;

  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
  LLVMContext &Context;
  BuilderTy Builder;
  IntegerType *IntTy;
  Value *Zero;
  CacheMapTy CacheMap;
  PtrSetTy SeenVals;
  ObjectSizeOpts EvalOpts;
  SmallPtrSet<Instruction *, 8> InsertedInstructions;

  SizeOffsetEvalType compute_(Value *V);

public:
  static SizeOffsetEvalType unknown() {
    return std::make_pair(nullptr, nullptr);
  }

  ObjectSizeOffsetEvaluator(const DataLayout &DL, const TargetLibraryInfo *TLI,
                            LLVMContext &Context, ObjectSizeOpts EvalOpts = {});

  SizeOffsetEvalType compute(Value *V);

  bool knownSize(SizeOffsetEvalType SizeOffset) {
    return SizeOffset.first;
  }

  bool knownOffset(SizeOffsetEvalType SizeOffset) {
    return SizeOffset.second;
  }

  bool anyKnown(SizeOffsetEvalType SizeOffset) {
    return knownSize(SizeOffset) || knownOffset(SizeOffset);
  }

  bool bothKnown(SizeOffsetEvalType SizeOffset) {
    return knownSize(SizeOffset) && knownOffset(SizeOffset);
  }

  // The individual instruction visitors should be treated as private.
  SizeOffsetEvalType visitAllocaInst(AllocaInst &I);
  SizeOffsetEvalType visitCallBase(CallBase &CB);
  SizeOffsetEvalType visitExtractElementInst(ExtractElementInst &I);
  SizeOffsetEvalType visitExtractValueInst(ExtractValueInst &I);
  SizeOffsetEvalType visitGEPOperator(GEPOperator &GEP);
  SizeOffsetEvalType visitIntToPtrInst(IntToPtrInst&);
  SizeOffsetEvalType visitLoadInst(LoadInst &I);
  SizeOffsetEvalType visitPHINode(PHINode &PHI);
  SizeOffsetEvalType visitSelectInst(SelectInst &I);
  SizeOffsetEvalType visitInstruction(Instruction &I);
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_MEMORYBUILTINS_H
