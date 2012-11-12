//===-- llvm/Support/CallSite.h - Abstract Call & Invoke instrs -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the CallSite class, which is a handy wrapper for code that
// wants to treat Call and Invoke instructions in a generic way. When in non-
// mutation context (e.g. an analysis) ImmutableCallSite should be used.
// Finally, when some degree of customization is necessary between these two
// extremes, CallSiteBase<> can be supplied with fine-tuned parameters.
//
// NOTE: These classes are supposed to have "value semantics". So they should be
// passed by value, not by reference; they should not be "new"ed or "delete"d.
// They are efficiently copyable, assignable and constructable, with cost
// equivalent to copying a pointer (notice that they have only a single data
// member). The internal representation carries a flag which indicates which of
// the two variants is enclosed. This allows for cheaper checks when various
// accessors of CallSite are employed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CALLSITE_H
#define LLVM_SUPPORT_CALLSITE_H

#include "llvm/Attributes.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/BasicBlock.h"
#include "llvm/CallingConv.h"
#include "llvm/Instructions.h"

namespace llvm {

class CallInst;
class InvokeInst;

template <typename FunTy = const Function,
          typename ValTy = const Value,
          typename UserTy = const User,
          typename InstrTy = const Instruction,
          typename CallTy = const CallInst,
          typename InvokeTy = const InvokeInst,
          typename IterTy = User::const_op_iterator>
class CallSiteBase {
protected:
  PointerIntPair<InstrTy*, 1, bool> I;
public:
  CallSiteBase() : I(0, false) {}
  CallSiteBase(CallTy *CI) : I(CI, true) { assert(CI); }
  CallSiteBase(InvokeTy *II) : I(II, false) { assert(II); }
  CallSiteBase(ValTy *II) { *this = get(II); }
protected:
  /// CallSiteBase::get - This static method is sort of like a constructor.  It
  /// will create an appropriate call site for a Call or Invoke instruction, but
  /// it can also create a null initialized CallSiteBase object for something
  /// which is NOT a call site.
  ///
  static CallSiteBase get(ValTy *V) {
    if (InstrTy *II = dyn_cast<InstrTy>(V)) {
      if (II->getOpcode() == Instruction::Call)
        return CallSiteBase(static_cast<CallTy*>(II));
      else if (II->getOpcode() == Instruction::Invoke)
        return CallSiteBase(static_cast<InvokeTy*>(II));
    }
    return CallSiteBase();
  }
public:
  /// isCall - true if a CallInst is enclosed.
  /// Note that !isCall() does not mean it is an InvokeInst enclosed,
  /// it also could signify a NULL Instruction pointer.
  bool isCall() const { return I.getInt(); }

  /// isInvoke - true if a InvokeInst is enclosed.
  ///
  bool isInvoke() const { return getInstruction() && !I.getInt(); }

  InstrTy *getInstruction() const { return I.getPointer(); }
  InstrTy *operator->() const { return I.getPointer(); }
  operator bool() const { return I.getPointer(); }

  /// getCalledValue - Return the pointer to function that is being called.
  ///
  ValTy *getCalledValue() const {
    assert(getInstruction() && "Not a call or invoke instruction!");
    return *getCallee();
  }

  /// getCalledFunction - Return the function being called if this is a direct
  /// call, otherwise return null (if it's an indirect call).
  ///
  FunTy *getCalledFunction() const {
    return dyn_cast<FunTy>(getCalledValue());
  }

  /// setCalledFunction - Set the callee to the specified value.
  ///
  void setCalledFunction(Value *V) {
    assert(getInstruction() && "Not a call or invoke instruction!");
    *getCallee() = V;
  }

  /// isCallee - Determine whether the passed iterator points to the
  /// callee operand's Use.
  ///
  bool isCallee(value_use_iterator<UserTy> UI) const {
    return getCallee() == &UI.getUse();
  }

  ValTy *getArgument(unsigned ArgNo) const {
    assert(arg_begin() + ArgNo < arg_end() && "Argument # out of range!");
    return *(arg_begin() + ArgNo);
  }

  void setArgument(unsigned ArgNo, Value* newVal) {
    assert(getInstruction() && "Not a call or invoke instruction!");
    assert(arg_begin() + ArgNo < arg_end() && "Argument # out of range!");
    getInstruction()->setOperand(ArgNo, newVal);
  }

  /// Given a value use iterator, returns the argument that corresponds to it.
  /// Iterator must actually correspond to an argument.
  unsigned getArgumentNo(value_use_iterator<UserTy> I) const {
    assert(getInstruction() && "Not a call or invoke instruction!");
    assert(arg_begin() <= &I.getUse() && &I.getUse() < arg_end()
           && "Argument # out of range!");
    return &I.getUse() - arg_begin();
  }

  /// arg_iterator - The type of iterator to use when looping over actual
  /// arguments at this call site.
  typedef IterTy arg_iterator;

  /// arg_begin/arg_end - Return iterators corresponding to the actual argument
  /// list for a call site.
  IterTy arg_begin() const {
    assert(getInstruction() && "Not a call or invoke instruction!");
    // Skip non-arguments
    return (*this)->op_begin();
  }

  IterTy arg_end() const { return (*this)->op_end() - getArgumentEndOffset(); }
  bool arg_empty() const { return arg_end() == arg_begin(); }
  unsigned arg_size() const { return unsigned(arg_end() - arg_begin()); }
  
  /// getType - Return the type of the instruction that generated this call site
  ///
  Type *getType() const { return (*this)->getType(); }

  /// getCaller - Return the caller function for this call site
  ///
  FunTy *getCaller() const { return (*this)->getParent()->getParent(); }

#define CALLSITE_DELEGATE_GETTER(METHOD) \
  InstrTy *II = getInstruction();    \
  return isCall()                        \
    ? cast<CallInst>(II)->METHOD         \
    : cast<InvokeInst>(II)->METHOD

#define CALLSITE_DELEGATE_SETTER(METHOD) \
  InstrTy *II = getInstruction();    \
  if (isCall())                          \
    cast<CallInst>(II)->METHOD;          \
  else                                   \
    cast<InvokeInst>(II)->METHOD

  /// getCallingConv/setCallingConv - get or set the calling convention of the
  /// call.
  CallingConv::ID getCallingConv() const {
    CALLSITE_DELEGATE_GETTER(getCallingConv());
  }
  void setCallingConv(CallingConv::ID CC) {
    CALLSITE_DELEGATE_SETTER(setCallingConv(CC));
  }

  /// getAttributes/setAttributes - get or set the parameter attributes of
  /// the call.
  const AttrListPtr &getAttributes() const {
    CALLSITE_DELEGATE_GETTER(getAttributes());
  }
  void setAttributes(const AttrListPtr &PAL) {
    CALLSITE_DELEGATE_SETTER(setAttributes(PAL));
  }

  /// \brief Return true if this function has the given attribute.
  bool hasFnAttr(Attributes::AttrVal A) const {
    CALLSITE_DELEGATE_GETTER(hasFnAttr(A));
  }

  /// \brief Return true if the call or the callee has the given attribute.
  bool paramHasAttr(unsigned i, Attributes::AttrVal A) const {
    CALLSITE_DELEGATE_GETTER(paramHasAttr(i, A));
  }

  /// @brief Extract the alignment for a call or parameter (0=unknown).
  uint16_t getParamAlignment(uint16_t i) const {
    CALLSITE_DELEGATE_GETTER(getParamAlignment(i));
  }

  /// @brief Return true if the call should not be inlined.
  bool isNoInline() const {
    CALLSITE_DELEGATE_GETTER(isNoInline());
  }
  void setIsNoInline(bool Value = true) {
    CALLSITE_DELEGATE_SETTER(setIsNoInline(Value));
  }

  /// @brief Determine if the call does not access memory.
  bool doesNotAccessMemory() const {
    CALLSITE_DELEGATE_GETTER(doesNotAccessMemory());
  }
  void setDoesNotAccessMemory() {
    CALLSITE_DELEGATE_SETTER(setDoesNotAccessMemory());
  }

  /// @brief Determine if the call does not access or only reads memory.
  bool onlyReadsMemory() const {
    CALLSITE_DELEGATE_GETTER(onlyReadsMemory());
  }
  void setOnlyReadsMemory() {
    CALLSITE_DELEGATE_SETTER(setOnlyReadsMemory());
  }

  /// @brief Determine if the call cannot return.
  bool doesNotReturn() const {
    CALLSITE_DELEGATE_GETTER(doesNotReturn());
  }
  void setDoesNotReturn() {
    CALLSITE_DELEGATE_SETTER(setDoesNotReturn());
  }

  /// @brief Determine if the call cannot unwind.
  bool doesNotThrow() const {
    CALLSITE_DELEGATE_GETTER(doesNotThrow());
  }
  void setDoesNotThrow() {
    CALLSITE_DELEGATE_SETTER(setDoesNotThrow());
  }

#undef CALLSITE_DELEGATE_GETTER
#undef CALLSITE_DELEGATE_SETTER

  /// @brief Determine whether this argument is not captured.
  bool doesNotCapture(unsigned ArgNo) const {
    return paramHasAttr(ArgNo + 1, Attributes::NoCapture);
  }

  /// @brief Determine whether this argument is passed by value.
  bool isByValArgument(unsigned ArgNo) const {
    return paramHasAttr(ArgNo + 1, Attributes::ByVal);
  }

  /// hasArgument - Returns true if this CallSite passes the given Value* as an
  /// argument to the called function.
  bool hasArgument(const Value *Arg) const {
    for (arg_iterator AI = this->arg_begin(), E = this->arg_end(); AI != E;
         ++AI)
      if (AI->get() == Arg)
        return true;
    return false;
  }

private:
  unsigned getArgumentEndOffset() const {
    if (isCall())
      return 1; // Skip Callee
    else
      return 3; // Skip BB, BB, Callee
  }

  IterTy getCallee() const {
    if (isCall()) // Skip Callee
      return cast<CallInst>(getInstruction())->op_end() - 1;
    else // Skip BB, BB, Callee
      return cast<InvokeInst>(getInstruction())->op_end() - 3;
  }
};

class CallSite : public CallSiteBase<Function, Value, User, Instruction,
                                     CallInst, InvokeInst, User::op_iterator> {
  typedef CallSiteBase<Function, Value, User, Instruction,
                       CallInst, InvokeInst, User::op_iterator> Base;
public:
  CallSite() {}
  CallSite(Base B) : Base(B) {}
  CallSite(Value* V) : Base(V) {}
  CallSite(CallInst *CI) : Base(CI) {}
  CallSite(InvokeInst *II) : Base(II) {}
  CallSite(Instruction *II) : Base(II) {}

  bool operator==(const CallSite &CS) const { return I == CS.I; }
  bool operator!=(const CallSite &CS) const { return I != CS.I; }
  bool operator<(const CallSite &CS) const {
    return getInstruction() < CS.getInstruction();
  }

private:
  User::op_iterator getCallee() const;
};

/// ImmutableCallSite - establish a view to a call site for examination
class ImmutableCallSite : public CallSiteBase<> {
  typedef CallSiteBase<> Base;
public:
  ImmutableCallSite(const Value* V) : Base(V) {}
  ImmutableCallSite(const CallInst *CI) : Base(CI) {}
  ImmutableCallSite(const InvokeInst *II) : Base(II) {}
  ImmutableCallSite(const Instruction *II) : Base(II) {}
  ImmutableCallSite(CallSite CS) : Base(CS.getInstruction()) {}
};

} // End llvm namespace

#endif
