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
// wants to treat Call and Invoke instructions in a generic way.
//
// NOTE: This class is supposed to have "value semantics". So it should be
// passed by value, not by reference; it should not be "new"ed or "delete"d. It
// is efficiently copyable, assignable and constructable, with cost equivalent
// to copying a pointer (notice that it has only a single data member).
// The internal representation carries a flag which indicates which of the two
// variants is enclosed. This allows for cheaper checks when various accessors
// of CallSite are employed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CALLSITE_H
#define LLVM_SUPPORT_CALLSITE_H

#include "llvm/Attributes.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"

namespace llvm {

class CallInst;
class InvokeInst;

class CallSite {
  PointerIntPair<Instruction*, 1, bool> I;
public:
  CallSite() : I(0, false) {}
  CallSite(CallInst *CI) : I(reinterpret_cast<Instruction*>(CI), true) {}
  CallSite(InvokeInst *II) : I(reinterpret_cast<Instruction*>(II), false) {}
  CallSite(Instruction *C);
  CallSite(const CallSite &CS) : I(CS.I) {}
  CallSite &operator=(const CallSite &CS) { I = CS.I; return *this; }

  bool operator==(const CallSite &CS) const { return I == CS.I; }
  bool operator!=(const CallSite &CS) const { return I != CS.I; }

  /// CallSite::get - This static method is sort of like a constructor.  It will
  /// create an appropriate call site for a Call or Invoke instruction, but it
  /// can also create a null initialized CallSite object for something which is
  /// NOT a call site.
  ///
  static CallSite get(Value *V) {
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      if (I->getOpcode() == Instruction::Call)
        return CallSite(reinterpret_cast<CallInst*>(I));
      else if (I->getOpcode() == Instruction::Invoke)
        return CallSite(reinterpret_cast<InvokeInst*>(I));
    }
    return CallSite();
  }

  /// getCallingConv/setCallingConv - get or set the calling convention of the
  /// call.
  unsigned getCallingConv() const;
  void setCallingConv(unsigned CC);

  /// getAttributes/setAttributes - get or set the parameter attributes of
  /// the call.
  const AttrListPtr &getAttributes() const;
  void setAttributes(const AttrListPtr &PAL);

  /// paramHasAttr - whether the call or the callee has the given attribute.
  bool paramHasAttr(uint16_t i, Attributes attr) const;

  /// @brief Extract the alignment for a call or parameter (0=unknown).
  uint16_t getParamAlignment(uint16_t i) const;

  /// @brief Determine if the call does not access memory.
  bool doesNotAccessMemory() const;
  void setDoesNotAccessMemory(bool doesNotAccessMemory = true);

  /// @brief Determine if the call does not access or only reads memory.
  bool onlyReadsMemory() const;
  void setOnlyReadsMemory(bool onlyReadsMemory = true);

  /// @brief Determine if the call cannot return.
  bool doesNotReturn() const;
  void setDoesNotReturn(bool doesNotReturn = true);

  /// @brief Determine if the call cannot unwind.
  bool doesNotThrow() const;
  void setDoesNotThrow(bool doesNotThrow = true);

  /// getType - Return the type of the instruction that generated this call site
  ///
  const Type *getType() const { return getInstruction()->getType(); }

  /// isCall - true if a CallInst is enclosed.
  /// Note that !isCall() does not mean it is an InvokeInst enclosed,
  /// it also could signify a NULL Instruction pointer.
  bool isCall() const { return I.getInt(); }

  /// isInvoke - true if a InvokeInst is enclosed.
  ///
  bool isInvoke() const { return getInstruction() && !I.getInt(); }

  /// getInstruction - Return the instruction this call site corresponds to
  ///
  Instruction *getInstruction() const { return I.getPointer(); }

  /// getCaller - Return the caller function for this call site
  ///
  Function *getCaller() const { return getInstruction()
                                  ->getParent()->getParent(); }

  /// getCalledValue - Return the pointer to function that is being called...
  ///
  Value *getCalledValue() const {
    assert(getInstruction() && "Not a call or invoke instruction!");
    return getInstruction()->getOperand(0);
  }

  /// getCalledFunction - Return the function being called if this is a direct
  /// call, otherwise return null (if it's an indirect call).
  ///
  Function *getCalledFunction() const {
    return dyn_cast<Function>(getCalledValue());
  }

  /// setCalledFunction - Set the callee to the specified value...
  ///
  void setCalledFunction(Value *V) {
    assert(getInstruction() && "Not a call or invoke instruction!");
    getInstruction()->setOperand(0, V);
  }

  Value *getArgument(unsigned ArgNo) const {
    assert(arg_begin() + ArgNo < arg_end() && "Argument # out of range!");
    return *(arg_begin()+ArgNo);
  }

  void setArgument(unsigned ArgNo, Value* newVal) {
    assert(getInstruction() && "Not a call or invoke instruction!");
    assert(arg_begin() + ArgNo < arg_end() && "Argument # out of range!");
    getInstruction()->setOperand(getArgumentOffset() + ArgNo, newVal);
  }

  /// Given an operand number, returns the argument that corresponds to it.
  /// OperandNo must be a valid operand number that actually corresponds to an
  /// argument.
  unsigned getArgumentNo(unsigned OperandNo) const {
    assert(OperandNo >= getArgumentOffset() && "Operand number passed was not "
                                               "a valid argument");
    return OperandNo - getArgumentOffset();
  }

  /// hasArgument - Returns true if this CallSite passes the given Value* as an
  /// argument to the called function.
  bool hasArgument(const Value *Arg) const;

  /// arg_iterator - The type of iterator to use when looping over actual
  /// arguments at this call site...
  typedef User::op_iterator arg_iterator;

  /// arg_begin/arg_end - Return iterators corresponding to the actual argument
  /// list for a call site.
  arg_iterator arg_begin() const {
    assert(getInstruction() && "Not a call or invoke instruction!");
    // Skip non-arguments
    return getInstruction()->op_begin() + getArgumentOffset();
  }

  arg_iterator arg_end() const { return getInstruction()->op_end(); }
  bool arg_empty() const { return arg_end() == arg_begin(); }
  unsigned arg_size() const { return unsigned(arg_end() - arg_begin()); }

  bool operator<(const CallSite &CS) const {
    return getInstruction() < CS.getInstruction();
  }

  bool isCallee(Value::use_iterator UI) const {
    return getInstruction()->op_begin() == &UI.getUse();
  }

private:
  /// Returns the operand number of the first argument
  unsigned getArgumentOffset() const {
    if (isCall())
      return 1; // Skip Function
    else
      return 3; // Skip Function, BB, BB
  }
};

} // End llvm namespace

#endif
