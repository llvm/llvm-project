//===-- llvm/IntrinsicInst.h - Intrinsic Instruction Wrappers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes that make it really easy to deal with intrinsic
// functions with the isa/dyncast family of functions.  In particular, this
// allows you to do things like:
//
//     if (MemCpyInst *MCI = dyn_cast<MemCpyInst>(Inst))
//        ... MCI->getDest() ... MCI->getSource() ...
//
// All intrinsic function calls are instances of the call instruction, so these
// are all subclasses of the CallInst class.  Note that none of these classes
// has state or virtual methods, which is an important part of this gross/neat
// hack working.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INTRINSICINST_H
#define LLVM_IR_INTRINSICINST_H

#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstdint>

namespace llvm {

/// A wrapper class for inspecting calls to intrinsic functions.
/// This allows the standard isa/dyncast/cast functionality to work with calls
/// to intrinsic functions.
class IntrinsicInst : public CallInst {
public:
  IntrinsicInst() = delete;
  IntrinsicInst(const IntrinsicInst &) = delete;
  IntrinsicInst &operator=(const IntrinsicInst &) = delete;

  /// Return the intrinsic ID of this intrinsic.
  Intrinsic::ID getIntrinsicID() const {
    return getCalledFunction()->getIntrinsicID();
  }

  /// Return true if swapping the first two arguments to the intrinsic produces
  /// the same result.
  bool isCommutative() const {
    switch (getIntrinsicID()) {
    case Intrinsic::maxnum:
    case Intrinsic::minnum:
    case Intrinsic::maximum:
    case Intrinsic::minimum:
    case Intrinsic::smax:
    case Intrinsic::smin:
    case Intrinsic::umax:
    case Intrinsic::umin:
    case Intrinsic::sadd_sat:
    case Intrinsic::uadd_sat:
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::uadd_with_overflow:
    case Intrinsic::smul_with_overflow:
    case Intrinsic::umul_with_overflow:
    case Intrinsic::smul_fix:
    case Intrinsic::umul_fix:
    case Intrinsic::smul_fix_sat:
    case Intrinsic::umul_fix_sat:
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
      return true;
    default:
      return false;
    }
  }

  // Checks if the intrinsic is an annotation.
  bool isAssumeLikeIntrinsic() const {
    switch (getIntrinsicID()) {
    default: break;
    case Intrinsic::assume:
    case Intrinsic::sideeffect:
    case Intrinsic::pseudoprobe:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::dbg_label:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::experimental_noalias_scope_decl:
    case Intrinsic::objectsize:
    case Intrinsic::ptr_annotation:
    case Intrinsic::var_annotation:
      return true;
    }
    return false;
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const CallInst *I) {
    if (const Function *CF = I->getCalledFunction())
      return CF->isIntrinsic();
    return false;
  }
  static bool classof(const Value *V) {
    return isa<CallInst>(V) && classof(cast<CallInst>(V));
  }
};

/// Check if \p ID corresponds to a debug info intrinsic.
static inline bool isDbgInfoIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_value:
  case Intrinsic::dbg_addr:
  case Intrinsic::dbg_label:
    return true;
  default:
    return false;
  }
}

/// This is the common base class for debug info intrinsics.
class DbgInfoIntrinsic : public IntrinsicInst {
public:
  /// \name Casting methods
  /// @{
  static bool classof(const IntrinsicInst *I) {
    return isDbgInfoIntrinsic(I->getIntrinsicID());
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
  /// @}
};

/// This is the common base class for debug info intrinsics for variables.
class DbgVariableIntrinsic : public DbgInfoIntrinsic {
public:
  // Iterator for ValueAsMetadata that internally uses direct pointer iteration
  // over either a ValueAsMetadata* or a ValueAsMetadata**, dereferencing to the
  // ValueAsMetadata .
  class location_op_iterator
      : public iterator_facade_base<location_op_iterator,
                                    std::bidirectional_iterator_tag, Value *> {
    PointerUnion<ValueAsMetadata *, ValueAsMetadata **> I;

  public:
    location_op_iterator(ValueAsMetadata *SingleIter) : I(SingleIter) {}
    location_op_iterator(ValueAsMetadata **MultiIter) : I(MultiIter) {}

    location_op_iterator(const location_op_iterator &R) : I(R.I) {}
    location_op_iterator &operator=(const location_op_iterator &R) {
      I = R.I;
      return *this;
    }
    bool operator==(const location_op_iterator &RHS) const {
      return I == RHS.I;
    }
    const Value *operator*() const {
      ValueAsMetadata *VAM = I.is<ValueAsMetadata *>()
                                 ? I.get<ValueAsMetadata *>()
                                 : *I.get<ValueAsMetadata **>();
      return VAM->getValue();
    };
    Value *operator*() {
      ValueAsMetadata *VAM = I.is<ValueAsMetadata *>()
                                 ? I.get<ValueAsMetadata *>()
                                 : *I.get<ValueAsMetadata **>();
      return VAM->getValue();
    }
    location_op_iterator &operator++() {
      if (I.is<ValueAsMetadata *>())
        I = I.get<ValueAsMetadata *>() + 1;
      else
        I = I.get<ValueAsMetadata **>() + 1;
      return *this;
    }
    location_op_iterator &operator--() {
      if (I.is<ValueAsMetadata *>())
        I = I.get<ValueAsMetadata *>() - 1;
      else
        I = I.get<ValueAsMetadata **>() - 1;
      return *this;
    }
  };

  /// Get the locations corresponding to the variable referenced by the debug
  /// info intrinsic.  Depending on the intrinsic, this could be the
  /// variable's value or its address.
  iterator_range<location_op_iterator> location_ops() const;

  Value *getVariableLocationOp(unsigned OpIdx) const;

  void replaceVariableLocationOp(Value *OldValue, Value *NewValue);
  void replaceVariableLocationOp(unsigned OpIdx, Value *NewValue);

  void setVariable(DILocalVariable *NewVar) {
    setArgOperand(1, MetadataAsValue::get(NewVar->getContext(), NewVar));
  }

  void setExpression(DIExpression *NewExpr) {
    setArgOperand(2, MetadataAsValue::get(NewExpr->getContext(), NewExpr));
  }

  unsigned getNumVariableLocationOps() const {
    if (hasArgList())
      return cast<DIArgList>(getRawLocation())->getArgs().size();
    return 1;
  }

  bool hasArgList() const { return isa<DIArgList>(getRawLocation()); }

  /// Does this describe the address of a local variable. True for dbg.addr
  /// and dbg.declare, but not dbg.value, which describes its value.
  bool isAddressOfVariable() const {
    return getIntrinsicID() != Intrinsic::dbg_value;
  }

  void setUndef() {
    // TODO: When/if we remove duplicate values from DIArgLists, we don't need
    // this set anymore.
    SmallPtrSet<Value *, 4> RemovedValues;
    for (Value *OldValue : location_ops()) {
      if (!RemovedValues.insert(OldValue).second)
        continue;
      Value *Undef = UndefValue::get(OldValue->getType());
      replaceVariableLocationOp(OldValue, Undef);
    }
  }

  bool isUndef() const {
    return (getNumVariableLocationOps() == 0 &&
            !getExpression()->isComplex()) ||
           any_of(location_ops(), [](Value *V) { return isa<UndefValue>(V); });
  }

  DILocalVariable *getVariable() const {
    return cast<DILocalVariable>(getRawVariable());
  }

  DIExpression *getExpression() const {
    return cast<DIExpression>(getRawExpression());
  }

  Metadata *getRawLocation() const {
    return cast<MetadataAsValue>(getArgOperand(0))->getMetadata();
  }

  Metadata *getRawVariable() const {
    return cast<MetadataAsValue>(getArgOperand(1))->getMetadata();
  }

  Metadata *getRawExpression() const {
    return cast<MetadataAsValue>(getArgOperand(2))->getMetadata();
  }

  /// Get the size (in bits) of the variable, or fragment of the variable that
  /// is described.
  Optional<uint64_t> getFragmentSizeInBits() const;

  /// \name Casting methods
  /// @{
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::dbg_addr:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
  /// @}
private:
  void setArgOperand(unsigned i, Value *v) {
    DbgInfoIntrinsic::setArgOperand(i, v);
  }
  void setOperand(unsigned i, Value *v) { DbgInfoIntrinsic::setOperand(i, v); }
};

/// This represents the llvm.dbg.declare instruction.
class DbgDeclareInst : public DbgVariableIntrinsic {
public:
  Value *getAddress() const {
    assert(getNumVariableLocationOps() == 1 &&
           "dbg.declare must have exactly 1 location operand.");
    return getVariableLocationOp(0);
  }

  /// \name Casting methods
  /// @{
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::dbg_declare;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
  /// @}
};

/// This represents the llvm.dbg.addr instruction.
class DbgAddrIntrinsic : public DbgVariableIntrinsic {
public:
  Value *getAddress() const {
    assert(getNumVariableLocationOps() == 1 &&
           "dbg.addr must have exactly 1 location operand.");
    return getVariableLocationOp(0);
  }

  /// \name Casting methods
  /// @{
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::dbg_addr;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.dbg.value instruction.
class DbgValueInst : public DbgVariableIntrinsic {
public:
  // The default argument should only be used in ISel, and the default option
  // should be removed once ISel support for multiple location ops is complete.
  Value *getValue(unsigned OpIdx = 0) const {
    return getVariableLocationOp(OpIdx);
  }
  iterator_range<location_op_iterator> getValues() const {
    return location_ops();
  }

  /// \name Casting methods
  /// @{
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::dbg_value;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
  /// @}
};

/// This represents the llvm.dbg.label instruction.
class DbgLabelInst : public DbgInfoIntrinsic {
public:
  DILabel *getLabel() const { return cast<DILabel>(getRawLabel()); }

  Metadata *getRawLabel() const {
    return cast<MetadataAsValue>(getArgOperand(0))->getMetadata();
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  /// @{
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::dbg_label;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
  /// @}
};

/// This is the common base class for vector predication intrinsics.
class VPIntrinsic : public IntrinsicInst {
public:
  static Optional<int> GetMaskParamPos(Intrinsic::ID IntrinsicID);
  static Optional<int> GetVectorLengthParamPos(Intrinsic::ID IntrinsicID);

  /// The llvm.vp.* intrinsics for this instruction Opcode
  static Intrinsic::ID GetForOpcode(unsigned OC);

  // Whether \p ID is a VP intrinsic ID.
  static bool IsVPIntrinsic(Intrinsic::ID);

  /// \return the mask parameter or nullptr.
  Value *getMaskParam() const;

  /// \return the vector length parameter or nullptr.
  Value *getVectorLengthParam() const;

  /// \return whether the vector length param can be ignored.
  bool canIgnoreVectorLengthParam() const;

  /// \return the static element count (vector number of elements) the vector
  /// length parameter applies to.
  ElementCount getStaticVectorLength() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return IsVPIntrinsic(I->getIntrinsicID());
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  // Equivalent non-predicated opcode
  unsigned getFunctionalOpcode() const {
    return GetFunctionalOpcodeForVP(getIntrinsicID());
  }

  // Equivalent non-predicated opcode
  static unsigned GetFunctionalOpcodeForVP(Intrinsic::ID ID);
};

/// This is the common base class for constrained floating point intrinsics.
class ConstrainedFPIntrinsic : public IntrinsicInst {
public:
  bool isUnaryOp() const;
  bool isTernaryOp() const;
  Optional<RoundingMode> getRoundingMode() const;
  Optional<fp::ExceptionBehavior> getExceptionBehavior() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I);
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// Constrained floating point compare intrinsics.
class ConstrainedFPCmpIntrinsic : public ConstrainedFPIntrinsic {
public:
  FCmpInst::Predicate getPredicate() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::experimental_constrained_fcmp:
    case Intrinsic::experimental_constrained_fcmps:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents an intrinsic that is based on a binary operation.
/// This includes op.with.overflow and saturating add/sub intrinsics.
class BinaryOpIntrinsic : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::uadd_with_overflow:
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::usub_with_overflow:
    case Intrinsic::ssub_with_overflow:
    case Intrinsic::umul_with_overflow:
    case Intrinsic::smul_with_overflow:
    case Intrinsic::uadd_sat:
    case Intrinsic::sadd_sat:
    case Intrinsic::usub_sat:
    case Intrinsic::ssub_sat:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  Value *getLHS() const { return const_cast<Value *>(getArgOperand(0)); }
  Value *getRHS() const { return const_cast<Value *>(getArgOperand(1)); }

  /// Returns the binary operation underlying the intrinsic.
  Instruction::BinaryOps getBinaryOp() const;

  /// Whether the intrinsic is signed or unsigned.
  bool isSigned() const;

  /// Returns one of OBO::NoSignedWrap or OBO::NoUnsignedWrap.
  unsigned getNoWrapKind() const;
};

/// Represents an op.with.overflow intrinsic.
class WithOverflowInst : public BinaryOpIntrinsic {
public:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::uadd_with_overflow:
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::usub_with_overflow:
    case Intrinsic::ssub_with_overflow:
    case Intrinsic::umul_with_overflow:
    case Intrinsic::smul_with_overflow:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// Represents a saturating add/sub intrinsic.
class SaturatingInst : public BinaryOpIntrinsic {
public:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::uadd_sat:
    case Intrinsic::sadd_sat:
    case Intrinsic::usub_sat:
    case Intrinsic::ssub_sat:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// Common base class for all memory intrinsics. Simply provides
/// common methods.
/// Written as CRTP to avoid a common base class amongst the
/// three atomicity hierarchies.
template <typename Derived> class MemIntrinsicBase : public IntrinsicInst {
private:
  enum { ARG_DEST = 0, ARG_LENGTH = 2 };

public:
  Value *getRawDest() const {
    return const_cast<Value *>(getArgOperand(ARG_DEST));
  }
  const Use &getRawDestUse() const { return getArgOperandUse(ARG_DEST); }
  Use &getRawDestUse() { return getArgOperandUse(ARG_DEST); }

  Value *getLength() const {
    return const_cast<Value *>(getArgOperand(ARG_LENGTH));
  }
  const Use &getLengthUse() const { return getArgOperandUse(ARG_LENGTH); }
  Use &getLengthUse() { return getArgOperandUse(ARG_LENGTH); }

  /// This is just like getRawDest, but it strips off any cast
  /// instructions (including addrspacecast) that feed it, giving the
  /// original input.  The returned value is guaranteed to be a pointer.
  Value *getDest() const { return getRawDest()->stripPointerCasts(); }

  unsigned getDestAddressSpace() const {
    return cast<PointerType>(getRawDest()->getType())->getAddressSpace();
  }

  /// FIXME: Remove this function once transition to Align is over.
  /// Use getDestAlign() instead.
  unsigned getDestAlignment() const {
    if (auto MA = getParamAlign(ARG_DEST))
      return MA->value();
    return 0;
  }
  MaybeAlign getDestAlign() const { return getParamAlign(ARG_DEST); }

  /// Set the specified arguments of the instruction.
  void setDest(Value *Ptr) {
    assert(getRawDest()->getType() == Ptr->getType() &&
           "setDest called with pointer of wrong type!");
    setArgOperand(ARG_DEST, Ptr);
  }

  /// FIXME: Remove this function once transition to Align is over.
  /// Use the version that takes MaybeAlign instead of this one.
  void setDestAlignment(unsigned Alignment) {
    setDestAlignment(MaybeAlign(Alignment));
  }
  void setDestAlignment(MaybeAlign Alignment) {
    removeParamAttr(ARG_DEST, Attribute::Alignment);
    if (Alignment)
      addParamAttr(ARG_DEST,
                   Attribute::getWithAlignment(getContext(), *Alignment));
  }
  void setDestAlignment(Align Alignment) {
    removeParamAttr(ARG_DEST, Attribute::Alignment);
    addParamAttr(ARG_DEST,
                 Attribute::getWithAlignment(getContext(), Alignment));
  }

  void setLength(Value *L) {
    assert(getLength()->getType() == L->getType() &&
           "setLength called with value of wrong type!");
    setArgOperand(ARG_LENGTH, L);
  }
};

/// Common base class for all memory transfer intrinsics. Simply provides
/// common methods.
template <class BaseCL> class MemTransferBase : public BaseCL {
private:
  enum { ARG_SOURCE = 1 };

public:
  /// Return the arguments to the instruction.
  Value *getRawSource() const {
    return const_cast<Value *>(BaseCL::getArgOperand(ARG_SOURCE));
  }
  const Use &getRawSourceUse() const {
    return BaseCL::getArgOperandUse(ARG_SOURCE);
  }
  Use &getRawSourceUse() { return BaseCL::getArgOperandUse(ARG_SOURCE); }

  /// This is just like getRawSource, but it strips off any cast
  /// instructions that feed it, giving the original input.  The returned
  /// value is guaranteed to be a pointer.
  Value *getSource() const { return getRawSource()->stripPointerCasts(); }

  unsigned getSourceAddressSpace() const {
    return cast<PointerType>(getRawSource()->getType())->getAddressSpace();
  }

  /// FIXME: Remove this function once transition to Align is over.
  /// Use getSourceAlign() instead.
  unsigned getSourceAlignment() const {
    if (auto MA = BaseCL::getParamAlign(ARG_SOURCE))
      return MA->value();
    return 0;
  }

  MaybeAlign getSourceAlign() const {
    return BaseCL::getParamAlign(ARG_SOURCE);
  }

  void setSource(Value *Ptr) {
    assert(getRawSource()->getType() == Ptr->getType() &&
           "setSource called with pointer of wrong type!");
    BaseCL::setArgOperand(ARG_SOURCE, Ptr);
  }

  /// FIXME: Remove this function once transition to Align is over.
  /// Use the version that takes MaybeAlign instead of this one.
  void setSourceAlignment(unsigned Alignment) {
    setSourceAlignment(MaybeAlign(Alignment));
  }
  void setSourceAlignment(MaybeAlign Alignment) {
    BaseCL::removeParamAttr(ARG_SOURCE, Attribute::Alignment);
    if (Alignment)
      BaseCL::addParamAttr(ARG_SOURCE, Attribute::getWithAlignment(
                                           BaseCL::getContext(), *Alignment));
  }
  void setSourceAlignment(Align Alignment) {
    BaseCL::removeParamAttr(ARG_SOURCE, Attribute::Alignment);
    BaseCL::addParamAttr(ARG_SOURCE, Attribute::getWithAlignment(
                                         BaseCL::getContext(), Alignment));
  }
};

/// Common base class for all memset intrinsics. Simply provides
/// common methods.
template <class BaseCL> class MemSetBase : public BaseCL {
private:
  enum { ARG_VALUE = 1 };

public:
  Value *getValue() const {
    return const_cast<Value *>(BaseCL::getArgOperand(ARG_VALUE));
  }
  const Use &getValueUse() const { return BaseCL::getArgOperandUse(ARG_VALUE); }
  Use &getValueUse() { return BaseCL::getArgOperandUse(ARG_VALUE); }

  void setValue(Value *Val) {
    assert(getValue()->getType() == Val->getType() &&
           "setValue called with value of wrong type!");
    BaseCL::setArgOperand(ARG_VALUE, Val);
  }
};

// The common base class for the atomic memset/memmove/memcpy intrinsics
// i.e. llvm.element.unordered.atomic.memset/memcpy/memmove
class AtomicMemIntrinsic : public MemIntrinsicBase<AtomicMemIntrinsic> {
private:
  enum { ARG_ELEMENTSIZE = 3 };

public:
  Value *getRawElementSizeInBytes() const {
    return const_cast<Value *>(getArgOperand(ARG_ELEMENTSIZE));
  }

  ConstantInt *getElementSizeInBytesCst() const {
    return cast<ConstantInt>(getRawElementSizeInBytes());
  }

  uint32_t getElementSizeInBytes() const {
    return getElementSizeInBytesCst()->getZExtValue();
  }

  void setElementSizeInBytes(Constant *V) {
    assert(V->getType() == Type::getInt8Ty(getContext()) &&
           "setElementSizeInBytes called with value of wrong type!");
    setArgOperand(ARG_ELEMENTSIZE, V);
  }

  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memcpy_element_unordered_atomic:
    case Intrinsic::memmove_element_unordered_atomic:
    case Intrinsic::memset_element_unordered_atomic:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents atomic memset intrinsic
// i.e. llvm.element.unordered.atomic.memset
class AtomicMemSetInst : public MemSetBase<AtomicMemIntrinsic> {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::memset_element_unordered_atomic;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

// This class wraps the atomic memcpy/memmove intrinsics
// i.e. llvm.element.unordered.atomic.memcpy/memmove
class AtomicMemTransferInst : public MemTransferBase<AtomicMemIntrinsic> {
public:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memcpy_element_unordered_atomic:
    case Intrinsic::memmove_element_unordered_atomic:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents the atomic memcpy intrinsic
/// i.e. llvm.element.unordered.atomic.memcpy
class AtomicMemCpyInst : public AtomicMemTransferInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::memcpy_element_unordered_atomic;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents the atomic memmove intrinsic
/// i.e. llvm.element.unordered.atomic.memmove
class AtomicMemMoveInst : public AtomicMemTransferInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::memmove_element_unordered_atomic;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This is the common base class for memset/memcpy/memmove.
class MemIntrinsic : public MemIntrinsicBase<MemIntrinsic> {
private:
  enum { ARG_VOLATILE = 3 };

public:
  ConstantInt *getVolatileCst() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(ARG_VOLATILE)));
  }

  bool isVolatile() const { return !getVolatileCst()->isZero(); }

  void setVolatile(Constant *V) { setArgOperand(ARG_VOLATILE, V); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memset:
    case Intrinsic::memcpy_inline:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class wraps the llvm.memset intrinsic.
class MemSetInst : public MemSetBase<MemIntrinsic> {
public:
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::memset;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class wraps the llvm.memcpy/memmove intrinsics.
class MemTransferInst : public MemTransferBase<MemIntrinsic> {
public:
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memcpy_inline:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class wraps the llvm.memcpy intrinsic.
class MemCpyInst : public MemTransferInst {
public:
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::memcpy;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class wraps the llvm.memmove intrinsic.
class MemMoveInst : public MemTransferInst {
public:
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::memmove;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class wraps the llvm.memcpy.inline intrinsic.
class MemCpyInlineInst : public MemTransferInst {
public:
  ConstantInt *getLength() const {
    return cast<ConstantInt>(MemTransferInst::getLength());
  }
  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::memcpy_inline;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

// The common base class for any memset/memmove/memcpy intrinsics;
// whether they be atomic or non-atomic.
// i.e. llvm.element.unordered.atomic.memset/memcpy/memmove
//  and llvm.memset/memcpy/memmove
class AnyMemIntrinsic : public MemIntrinsicBase<AnyMemIntrinsic> {
public:
  bool isVolatile() const {
    // Only the non-atomic intrinsics can be volatile
    if (auto *MI = dyn_cast<MemIntrinsic>(this))
      return MI->isVolatile();
    return false;
  }

  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_inline:
    case Intrinsic::memmove:
    case Intrinsic::memset:
    case Intrinsic::memcpy_element_unordered_atomic:
    case Intrinsic::memmove_element_unordered_atomic:
    case Intrinsic::memset_element_unordered_atomic:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents any memset intrinsic
// i.e. llvm.element.unordered.atomic.memset
// and  llvm.memset
class AnyMemSetInst : public MemSetBase<AnyMemIntrinsic> {
public:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memset:
    case Intrinsic::memset_element_unordered_atomic:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

// This class wraps any memcpy/memmove intrinsics
// i.e. llvm.element.unordered.atomic.memcpy/memmove
// and  llvm.memcpy/memmove
class AnyMemTransferInst : public MemTransferBase<AnyMemIntrinsic> {
public:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_inline:
    case Intrinsic::memmove:
    case Intrinsic::memcpy_element_unordered_atomic:
    case Intrinsic::memmove_element_unordered_atomic:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents any memcpy intrinsic
/// i.e. llvm.element.unordered.atomic.memcpy
///  and llvm.memcpy
class AnyMemCpyInst : public AnyMemTransferInst {
public:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memcpy_inline:
    case Intrinsic::memcpy_element_unordered_atomic:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This class represents any memmove intrinsic
/// i.e. llvm.element.unordered.atomic.memmove
///  and llvm.memmove
class AnyMemMoveInst : public AnyMemTransferInst {
public:
  static bool classof(const IntrinsicInst *I) {
    switch (I->getIntrinsicID()) {
    case Intrinsic::memmove:
    case Intrinsic::memmove_element_unordered_atomic:
      return true;
    default:
      return false;
    }
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.va_start intrinsic.
class VAStartInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::vastart;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  Value *getArgList() const { return const_cast<Value *>(getArgOperand(0)); }
};

/// This represents the llvm.va_end intrinsic.
class VAEndInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::vaend;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  Value *getArgList() const { return const_cast<Value *>(getArgOperand(0)); }
};

/// This represents the llvm.va_copy intrinsic.
class VACopyInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::vacopy;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  Value *getDest() const { return const_cast<Value *>(getArgOperand(0)); }
  Value *getSrc() const { return const_cast<Value *>(getArgOperand(1)); }
};

/// This represents the llvm.instrprof_increment intrinsic.
class InstrProfIncrementInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::instrprof_increment;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  GlobalVariable *getName() const {
    return cast<GlobalVariable>(
        const_cast<Value *>(getArgOperand(0))->stripPointerCasts());
  }

  ConstantInt *getHash() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(1)));
  }

  ConstantInt *getNumCounters() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(2)));
  }

  ConstantInt *getIndex() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(3)));
  }

  Value *getStep() const;
};

class InstrProfIncrementInstStep : public InstrProfIncrementInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::instrprof_increment_step;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

/// This represents the llvm.instrprof_value_profile intrinsic.
class InstrProfValueProfileInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::instrprof_value_profile;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  GlobalVariable *getName() const {
    return cast<GlobalVariable>(
        const_cast<Value *>(getArgOperand(0))->stripPointerCasts());
  }

  ConstantInt *getHash() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(1)));
  }

  Value *getTargetValue() const {
    return cast<Value>(const_cast<Value *>(getArgOperand(2)));
  }

  ConstantInt *getValueKind() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(3)));
  }

  // Returns the value site index.
  ConstantInt *getIndex() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(4)));
  }
};

class PseudoProbeInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::pseudoprobe;
  }

  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  ConstantInt *getFuncGuid() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(0)));
  }

  ConstantInt *getIndex() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(1)));
  }

  ConstantInt *getAttributes() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(2)));
  }

  ConstantInt *getFactor() const {
    return cast<ConstantInt>(const_cast<Value *>(getArgOperand(3)));
  }
};

class NoAliasScopeDeclInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::experimental_noalias_scope_decl;
  }

  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }

  MDNode *getScopeList() const {
    auto *MV =
        cast<MetadataAsValue>(getOperand(Intrinsic::NoAliasScopeDeclScopeArg));
    return cast<MDNode>(MV->getMetadata());
  }

  void setScopeList(MDNode *ScopeList) {
    setOperand(Intrinsic::NoAliasScopeDeclScopeArg,
               MetadataAsValue::get(getContext(), ScopeList));
  }
};

// Defined in Statepoint.h -- NOT a subclass of IntrinsicInst
class GCStatepointInst;

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
  const GCStatepointInst *getStatepoint() const;
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

  Value *getBasePtr() const;
  Value *getDerivedPtr() const;
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


/// This represents the llvm.assume intrinsic.
class AssumeInst : public IntrinsicInst {
public:
  static bool classof(const IntrinsicInst *I) {
    return I->getIntrinsicID() == Intrinsic::assume;
  }
  static bool classof(const Value *V) {
    return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
  }
};

} // end namespace llvm

#endif // LLVM_IR_INTRINSICINST_H
