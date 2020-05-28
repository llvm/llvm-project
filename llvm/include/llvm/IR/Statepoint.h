//===- llvm/IR/Statepoint.h - gc.statepoint utilities -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions and a wrapper class analogous to
// CallBase for accessing the fields of gc.statepoint, gc.relocate,
// gc.result intrinsics; and some general utilities helpful when dealing with
// gc.statepoint.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_STATEPOINT_H
#define LLVM_IR_STATEPOINT_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llvm {

/// The statepoint intrinsic accepts a set of flags as its third argument.
/// Valid values come out of this set.
enum class StatepointFlags {
  None = 0,
  GCTransition = 1, ///< Indicates that this statepoint is a transition from
                    ///< GC-aware code to code that is not GC-aware.
  /// Mark the deopt arguments associated with the statepoint as only being
  /// "live-in". By default, deopt arguments are "live-through".  "live-through"
  /// requires that they the value be live on entry, on exit, and at any point
  /// during the call.  "live-in" only requires the value be available at the
  /// start of the call.  In particular, "live-in" values can be placed in
  /// unused argument registers or other non-callee saved registers.
  DeoptLiveIn = 2,

  MaskAll = 3 ///< A bitmask that includes all valid flags.
};

class GCRelocateInst;
class GCResultInst;

bool isStatepoint(const CallBase *Call);
bool isStatepoint(const Value *V);
bool isStatepoint(const Value &V);

bool isGCRelocate(const CallBase *Call);
bool isGCRelocate(const Value *V);

bool isGCResult(const CallBase *Call);
bool isGCResult(const Value *V);

/// Represents a gc.statepoint intrinsic call.  This extends directly from
/// CallBase as the IntrinsicInst only supports calls and gc.statepoint is
/// invokable.
class GCStatepointInst : public CallBase {
public:
  GCStatepointInst() = delete;
  GCStatepointInst(const GCStatepointInst &) = delete;
  GCStatepointInst &operator=(const GCStatepointInst &) = delete;

  static bool classof(const CallBase *I) {
    if (const Function *CF = I->getCalledFunction())
      return CF->getIntrinsicID() == Intrinsic::experimental_gc_statepoint;
    return false;
  }

  static bool classof(const Value *V) {
    return isa<CallBase>(V) && classof(cast<CallBase>(V));
  }

  enum {
    IDPos = 0,
    NumPatchBytesPos = 1,
    CalledFunctionPos = 2,
    NumCallArgsPos = 3,
    FlagsPos = 4,
    CallArgsBeginPos = 5,
  };

  /// Return the ID associated with this statepoint.
  uint64_t getID() const {
    return cast<ConstantInt>(getArgOperand(IDPos))->getZExtValue();
  }

  /// Return the number of patchable bytes associated with this statepoint.
  uint32_t getNumPatchBytes() const {
    const Value *NumPatchBytesVal = getArgOperand(NumPatchBytesPos);
    uint64_t NumPatchBytes =
      cast<ConstantInt>(NumPatchBytesVal)->getZExtValue();
    assert(isInt<32>(NumPatchBytes) && "should fit in 32 bits!");
    return NumPatchBytes;
  }

  /// Number of arguments to be passed to the actual callee.
  int getNumCallArgs() const {
    return cast<ConstantInt>(getArgOperand(NumCallArgsPos))->getZExtValue();
  }

  uint64_t getFlags() const {
    return cast<ConstantInt>(getArgOperand(FlagsPos))->getZExtValue();
  }

  /// Return the value actually being called or invoked.
  Value *getActualCalledOperand() const {
    return getArgOperand(CalledFunctionPos);
  }

  /// Returns the function called if this is a wrapping a direct call, and null
  /// otherwise.
  Function *getActualCalledFunction() const {
    return dyn_cast_or_null<Function>(getActualCalledOperand());
  }

  /// Return the type of the value returned by the call underlying the
  /// statepoint.
  Type *getActualReturnType() const {
    auto *CalleeTy =
      cast<PointerType>(getActualCalledOperand()->getType())->getElementType();
    return cast<FunctionType>(CalleeTy)->getReturnType();
  }
};

/// A wrapper around a GC intrinsic call, this provides most of the actual
/// functionality for Statepoint and ImmutableStatepoint.  It is
/// templatized to allow easily specializing of const and non-const
/// concrete subtypes.
template <typename FunTy, typename InstructionTy, typename ValueTy,
          typename CallTy>
class StatepointBase {
  CallTy *StatepointCall;

protected:
  explicit StatepointBase(InstructionTy *I) {
    StatepointCall = dyn_cast<GCStatepointInst>(I);
  }

  explicit StatepointBase(CallTy *Call) {
    StatepointCall = dyn_cast<GCStatepointInst>(Call);
  }

public:
  using arg_iterator = typename CallTy::const_op_iterator;

  enum {
    CallArgsBeginPos = GCStatepointInst::CallArgsBeginPos,
  };

  void *operator new(size_t, unsigned) = delete;
  void *operator new(size_t s) = delete;

  explicit operator bool() const {
    // We do not assign non-statepoint call instructions to StatepointCall.
    return (bool)StatepointCall;
  }

  /// Return the underlying call instruction.
  CallTy *getCall() const {
    assert(*this && "check validity first!");
    return StatepointCall;
  }

  // Deprecated shims (update all callers to remove)
  uint64_t getFlags() const { return getCall()->getFlags(); }
  uint64_t getID() const { return getCall()->getID(); }
  uint32_t getNumPatchBytes() const { return getCall()->getNumPatchBytes(); }
  int getNumCallArgs() const { return getCall()->getNumCallArgs(); }
  ValueTy *getCalledValue() const {
    return getCall()->getActualCalledOperand();
  }
  Type *getActualReturnType() const { return getCall()->getActualReturnType(); }
  FunTy *getCalledFunction() const {
    return getCall()->getActualCalledFunction();
  }

  
  // FIXME: Migrate users of this to `getCall` and remove it.
  InstructionTy *getInstruction() const { return getCall(); }

  /// Return the caller function for this statepoint.
  FunTy *getCaller() const { return getCall()->getCaller(); }

  /// Determine if the statepoint cannot unwind.
  bool doesNotThrow() const {
    Function *F = getCalledFunction();
    return getCall()->doesNotThrow() || (F ? F->doesNotThrow() : false);
  }


  size_t arg_size() const { return getNumCallArgs(); }
  arg_iterator arg_begin() const {
    assert(CallArgsBeginPos <= (int)getCall()->arg_size());
    return getCall()->arg_begin() + CallArgsBeginPos;
  }
  arg_iterator arg_end() const {
    auto I = arg_begin() + arg_size();
    assert((getCall()->arg_end() - I) >= 0);
    return I;
  }

  ValueTy *getArgument(unsigned Index) {
    assert(Index < arg_size() && "out of bounds!");
    return *(arg_begin() + Index);
  }

  /// range adapter for call arguments
  iterator_range<arg_iterator> call_args() const {
    return make_range(arg_begin(), arg_end());
  }

  /// Return true if the call or the callee has the given attribute.
  bool paramHasAttr(unsigned i, Attribute::AttrKind A) const {
    Function *F = getCalledFunction();
    return getCall()->paramHasAttr(i + CallArgsBeginPos, A) ||
           (F ? F->getAttributes().hasAttribute(i, A) : false);
  }

  /// Number of GC transition args.
  int getNumTotalGCTransitionArgs() const {
    const Value *NumGCTransitionArgs = *arg_end();
    return cast<ConstantInt>(NumGCTransitionArgs)->getZExtValue();
  }
  arg_iterator gc_transition_args_begin() const {
    auto I = arg_end() + 1;
    assert((getCall()->arg_end() - I) >= 0);
    return I;
  }
  arg_iterator gc_transition_args_end() const {
    auto I = gc_transition_args_begin() + getNumTotalGCTransitionArgs();
    assert((getCall()->arg_end() - I) >= 0);
    return I;
  }

  /// range adapter for GC transition arguments
  iterator_range<arg_iterator> gc_transition_args() const {
    return make_range(gc_transition_args_begin(), gc_transition_args_end());
  }

  /// Number of additional arguments excluding those intended
  /// for garbage collection.
  int getNumTotalVMSArgs() const {
    const Value *NumVMSArgs = *gc_transition_args_end();
    return cast<ConstantInt>(NumVMSArgs)->getZExtValue();
  }

  arg_iterator deopt_begin() const {
    auto I = gc_transition_args_end() + 1;
    assert((getCall()->arg_end() - I) >= 0);
    return I;
  }
  arg_iterator deopt_end() const {
    auto I = deopt_begin() + getNumTotalVMSArgs();
    assert((getCall()->arg_end() - I) >= 0);
    return I;
  }

  /// range adapter for vm state arguments
  iterator_range<arg_iterator> deopt_operands() const {
    return make_range(deopt_begin(), deopt_end());
  }

  arg_iterator gc_args_begin() const { return deopt_end(); }
  arg_iterator gc_args_end() const { return getCall()->arg_end(); }

  unsigned gcArgsStartIdx() const {
    return gc_args_begin() - getCall()->op_begin();
  }

  /// range adapter for gc arguments
  iterator_range<arg_iterator> gc_args() const {
    return make_range(gc_args_begin(), gc_args_end());
  }

  /// Get list of all gc reloactes linked to this statepoint
  /// May contain several relocations for the same base/derived pair.
  /// For example this could happen due to relocations on unwinding
  /// path of invoke.
  std::vector<const GCRelocateInst *> getRelocates() const;

  /// Get the experimental_gc_result call tied to this statepoint.  Can be
  /// nullptr if there isn't a gc_result tied to this statepoint.  Guaranteed to
  /// be a CallInst if non-null.
  const GCResultInst *getGCResult() const {
    for (auto *U : getInstruction()->users())
      if (auto *GRI = dyn_cast<GCResultInst>(U))
        return GRI;
    return nullptr;
  }

#ifndef NDEBUG
  /// Asserts if this statepoint is malformed.  Common cases for failure
  /// include incorrect length prefixes for variable length sections or
  /// illegal values for parameters.
  void verify() {
    assert(getNumCallArgs() >= 0 &&
           "number of arguments to actually callee can't be negative");

    // The internal asserts in the iterator accessors do the rest.
    (void)arg_begin();
    (void)arg_end();
    (void)gc_transition_args_begin();
    (void)gc_transition_args_end();
    (void)deopt_begin();
    (void)deopt_end();
    (void)gc_args_begin();
    (void)gc_args_end();
  }
#endif
};

/// A specialization of it's base class for read only access
/// to a gc.statepoint.
class ImmutableStatepoint
    : public StatepointBase<const Function, const Instruction, const Value,
                            const GCStatepointInst> {
  using Base = StatepointBase<const Function, const Instruction, const Value,
                              const GCStatepointInst>;

public:
  explicit ImmutableStatepoint(const Instruction *I) : Base(I) {}
  explicit ImmutableStatepoint(const CallBase *Call) : Base(Call) {}
};

/// A specialization of it's base class for read-write access
/// to a gc.statepoint.
class Statepoint
    : public StatepointBase<Function, Instruction, Value, GCStatepointInst> {
  using Base = StatepointBase<Function, Instruction, Value, GCStatepointInst>;

public:
  explicit Statepoint(Instruction *I) : Base(I) {}
  explicit Statepoint(CallBase *Call) : Base(Call) {}
};

/// Common base class for representing values projected from a statepoint.
/// Currently, the only projections available are gc.result and gc.relocate.
class GCProjectionInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::experimental_gc_relocate ||
      I->getIntrinsicID() == Intrinsic::experimental_gc_result;
  }

  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  /// Return true if this relocate is tied to the invoke statepoint.
  /// This includes relocates which are on the unwinding path.
  bool isTiedToInvoke() const {
    const Value *Token = getArgOperand(0);

    return isa<LandingPadInst>(Token) || isa<InvokeInst>(Token);
  }

  /// The statepoint with which this gc.relocate is associated.
  const GCStatepointInst *getStatepoint() const {
    const Value *Token = getArgOperand(0);

    // This takes care both of relocates for call statepoints and relocates
    // on normal path of invoke statepoint.
    if (!isa<LandingPadInst>(Token))
      return cast<GCStatepointInst>(Token);

    // This relocate is on exceptional path of an invoke statepoint
    const BasicBlock *InvokeBB =
        cast<Instruction>(Token)->getParent()->getUniquePredecessor();

    assert(InvokeBB && "safepoints should have unique landingpads");
    assert(InvokeBB->getTerminator() &&
           "safepoint block should be well formed");

    return cast<GCStatepointInst>(InvokeBB->getTerminator());
  }
};

/// Represents calls to the gc.relocate intrinsic.
class GCRelocateInst : public GCProjectionInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::experimental_gc_relocate;
  }

  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  /// The index into the associate statepoint's argument list
  /// which contains the base pointer of the pointer whose
  /// relocation this gc.relocate describes.
  unsigned getBasePtrIndex() const {
    return cast<ConstantInt>(getArgOperand(1))->getZExtValue();
  }

  /// The index into the associate statepoint's argument list which
  /// contains the pointer whose relocation this gc.relocate describes.
  unsigned getDerivedPtrIndex() const {
    return cast<ConstantInt>(getArgOperand(2))->getZExtValue();
  }

  Value *getBasePtr() const {
    return *(getStatepoint()->arg_begin() + getBasePtrIndex());
  }

  Value *getDerivedPtr() const {
    return *(getStatepoint()->arg_begin() + getDerivedPtrIndex());
  }
};

/// Represents calls to the gc.result intrinsic.
class GCResultInst : public GCProjectionInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::experimental_gc_result;
  }

  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

template <typename FunTy, typename InstructionTy, typename ValueTy,
          typename CallTy>
std::vector<const GCRelocateInst *>
StatepointBase<FunTy, InstructionTy, ValueTy, CallTy>::getRelocates()
    const {
  std::vector<const GCRelocateInst *> Result;

  // Search for relocated pointers.  Note that working backwards from the
  // gc_relocates ensures that we only get pairs which are actually relocated
  // and used after the statepoint.
  for (const User *U : StatepointCall->users())
    if (auto *Relocate = dyn_cast<GCRelocateInst>(U))
      Result.push_back(Relocate);

  auto *StatepointInvoke = dyn_cast<InvokeInst>(StatepointCall);
  if (!StatepointInvoke)
    return Result;

  // We need to scan thorough exceptional relocations if it is invoke statepoint
  LandingPadInst *LandingPad = StatepointInvoke->getLandingPadInst();

  // Search for gc relocates that are attached to this landingpad.
  for (const User *LandingPadUser : LandingPad->users()) {
    if (auto *Relocate = dyn_cast<GCRelocateInst>(LandingPadUser))
      Result.push_back(Relocate);
  }
  return Result;
}

/// Call sites that get wrapped by a gc.statepoint (currently only in
/// RewriteStatepointsForGC and potentially in other passes in the future) can
/// have attributes that describe properties of gc.statepoint call they will be
/// eventually be wrapped in.  This struct is used represent such directives.
struct StatepointDirectives {
  Optional<uint32_t> NumPatchBytes;
  Optional<uint64_t> StatepointID;

  static const uint64_t DefaultStatepointID = 0xABCDEF00;
  static const uint64_t DeoptBundleStatepointID = 0xABCDEF0F;
};

/// Parse out statepoint directives from the function attributes present in \p
/// AS.
StatepointDirectives parseStatepointDirectivesFromAttrs(AttributeList AS);

/// Return \c true if the \p Attr is an attribute that is a statepoint
/// directive.
bool isStatepointDirectiveAttr(Attribute Attr);

} // end namespace llvm

#endif // LLVM_IR_STATEPOINT_H
