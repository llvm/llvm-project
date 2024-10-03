//===- Constant.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_CONSTANT_H
#define LLVM_SANDBOXIR_CONSTANT_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/SandboxIR/Argument.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Type.h"
#include "llvm/SandboxIR/User.h"

namespace llvm::sandboxir {

class BasicBlock;
class Function;

class Constant : public sandboxir::User {
protected:
  Constant(llvm::Constant *C, sandboxir::Context &SBCtx)
      : sandboxir::User(ClassID::Constant, C, SBCtx) {}
  Constant(ClassID ID, llvm::Constant *C, sandboxir::Context &SBCtx)
      : sandboxir::User(ID, C, SBCtx) {}
  friend class ConstantInt; // For constructor.
  friend class Function;    // For constructor
  friend class Context;     // For constructor.
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const override {
    return getOperandUseDefault(OpIdx, Verify);
  }

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    switch (From->getSubclassID()) {
#define DEF_CONST(ID, CLASS) case ClassID::ID:
#include "llvm/SandboxIR/SandboxIRValues.def"
      return true;
    default:
      return false;
    }
  }
  sandboxir::Context &getParent() const { return getContext(); }
  unsigned getUseOperandNo(const Use &Use) const override {
    return getUseOperandNoDefault(Use);
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::Constant>(Val) && "Expected Constant!");
  }
  void dumpOS(raw_ostream &OS) const override;
#endif
};

// TODO: This should inherit from ConstantData.
class ConstantInt : public Constant {
  ConstantInt(llvm::ConstantInt *C, Context &Ctx)
      : Constant(ClassID::ConstantInt, C, Ctx) {}
  friend class Context; // For constructor.

  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    llvm_unreachable("ConstantInt has no operands!");
  }

public:
  static ConstantInt *getTrue(Context &Ctx);
  static ConstantInt *getFalse(Context &Ctx);
  static ConstantInt *getBool(Context &Ctx, bool V);
  static Constant *getTrue(Type *Ty);
  static Constant *getFalse(Type *Ty);
  static Constant *getBool(Type *Ty, bool V);

  /// If Ty is a vector type, return a Constant with a splat of the given
  /// value. Otherwise return a ConstantInt for the given value.
  static ConstantInt *get(Type *Ty, uint64_t V, bool IsSigned = false);

  /// Return a ConstantInt with the specified integer value for the specified
  /// type. If the type is wider than 64 bits, the value will be zero-extended
  /// to fit the type, unless IsSigned is true, in which case the value will
  /// be interpreted as a 64-bit signed integer and sign-extended to fit
  /// the type.
  /// Get a ConstantInt for a specific value.
  static ConstantInt *get(IntegerType *Ty, uint64_t V, bool IsSigned = false);

  /// Return a ConstantInt with the specified value for the specified type. The
  /// value V will be canonicalized to a an unsigned APInt. Accessing it with
  /// either getSExtValue() or getZExtValue() will yield a correctly sized and
  /// signed value for the type Ty.
  /// Get a ConstantInt for a specific signed value.
  static ConstantInt *getSigned(IntegerType *Ty, int64_t V);
  static Constant *getSigned(Type *Ty, int64_t V);

  /// Return a ConstantInt with the specified value and an implied Type. The
  /// type is the integer type that corresponds to the bit width of the value.
  static ConstantInt *get(Context &Ctx, const APInt &V);

  /// Return a ConstantInt constructed from the string strStart with the given
  /// radix.
  static ConstantInt *get(IntegerType *Ty, StringRef Str, uint8_t Radix);

  /// If Ty is a vector type, return a Constant with a splat of the given
  /// value. Otherwise return a ConstantInt for the given value.
  static Constant *get(Type *Ty, const APInt &V);

  /// Return the constant as an APInt value reference. This allows clients to
  /// obtain a full-precision copy of the value.
  /// Return the constant's value.
  inline const APInt &getValue() const {
    return cast<llvm::ConstantInt>(Val)->getValue();
  }

  /// getBitWidth - Return the scalar bitwidth of this constant.
  unsigned getBitWidth() const {
    return cast<llvm::ConstantInt>(Val)->getBitWidth();
  }
  /// Return the constant as a 64-bit unsigned integer value after it
  /// has been zero extended as appropriate for the type of this constant. Note
  /// that this method can assert if the value does not fit in 64 bits.
  /// Return the zero extended value.
  inline uint64_t getZExtValue() const {
    return cast<llvm::ConstantInt>(Val)->getZExtValue();
  }

  /// Return the constant as a 64-bit integer value after it has been sign
  /// extended as appropriate for the type of this constant. Note that
  /// this method can assert if the value does not fit in 64 bits.
  /// Return the sign extended value.
  inline int64_t getSExtValue() const {
    return cast<llvm::ConstantInt>(Val)->getSExtValue();
  }

  /// Return the constant as an llvm::MaybeAlign.
  /// Note that this method can assert if the value does not fit in 64 bits or
  /// is not a power of two.
  inline MaybeAlign getMaybeAlignValue() const {
    return cast<llvm::ConstantInt>(Val)->getMaybeAlignValue();
  }

  /// Return the constant as an llvm::Align, interpreting `0` as `Align(1)`.
  /// Note that this method can assert if the value does not fit in 64 bits or
  /// is not a power of two.
  inline Align getAlignValue() const {
    return cast<llvm::ConstantInt>(Val)->getAlignValue();
  }

  /// A helper method that can be used to determine if the constant contained
  /// within is equal to a constant.  This only works for very small values,
  /// because this is all that can be represented with all types.
  /// Determine if this constant's value is same as an unsigned char.
  bool equalsInt(uint64_t V) const {
    return cast<llvm::ConstantInt>(Val)->equalsInt(V);
  }

  /// Variant of the getType() method to always return an IntegerType, which
  /// reduces the amount of casting needed in parts of the compiler.
  IntegerType *getIntegerType() const;

  /// This static method returns true if the type Ty is big enough to
  /// represent the value V. This can be used to avoid having the get method
  /// assert when V is larger than Ty can represent. Note that there are two
  /// versions of this method, one for unsigned and one for signed integers.
  /// Although ConstantInt canonicalizes everything to an unsigned integer,
  /// the signed version avoids callers having to convert a signed quantity
  /// to the appropriate unsigned type before calling the method.
  /// @returns true if V is a valid value for type Ty
  /// Determine if the value is in range for the given type.
  static bool isValueValidForType(Type *Ty, uint64_t V);
  static bool isValueValidForType(Type *Ty, int64_t V);

  bool isNegative() const { return cast<llvm::ConstantInt>(Val)->isNegative(); }

  /// This is just a convenience method to make client code smaller for a
  /// common code. It also correctly performs the comparison without the
  /// potential for an assertion from getZExtValue().
  bool isZero() const { return cast<llvm::ConstantInt>(Val)->isZero(); }

  /// This is just a convenience method to make client code smaller for a
  /// common case. It also correctly performs the comparison without the
  /// potential for an assertion from getZExtValue().
  /// Determine if the value is one.
  bool isOne() const { return cast<llvm::ConstantInt>(Val)->isOne(); }

  /// This function will return true iff every bit in this constant is set
  /// to true.
  /// @returns true iff this constant's bits are all set to true.
  /// Determine if the value is all ones.
  bool isMinusOne() const { return cast<llvm::ConstantInt>(Val)->isMinusOne(); }

  /// This function will return true iff this constant represents the largest
  /// value that may be represented by the constant's type.
  /// @returns true iff this is the largest value that may be represented
  /// by this type.
  /// Determine if the value is maximal.
  bool isMaxValue(bool IsSigned) const {
    return cast<llvm::ConstantInt>(Val)->isMaxValue(IsSigned);
  }

  /// This function will return true iff this constant represents the smallest
  /// value that may be represented by this constant's type.
  /// @returns true if this is the smallest value that may be represented by
  /// this type.
  /// Determine if the value is minimal.
  bool isMinValue(bool IsSigned) const {
    return cast<llvm::ConstantInt>(Val)->isMinValue(IsSigned);
  }

  /// This function will return true iff this constant represents a value with
  /// active bits bigger than 64 bits or a value greater than the given uint64_t
  /// value.
  /// @returns true iff this constant is greater or equal to the given number.
  /// Determine if the value is greater or equal to the given number.
  bool uge(uint64_t Num) const {
    return cast<llvm::ConstantInt>(Val)->uge(Num);
  }

  /// getLimitedValue - If the value is smaller than the specified limit,
  /// return it, otherwise return the limit value.  This causes the value
  /// to saturate to the limit.
  /// @returns the min of the value of the constant and the specified value
  /// Get the constant's value with a saturation limit
  uint64_t getLimitedValue(uint64_t Limit = ~0ULL) const {
    return cast<llvm::ConstantInt>(Val)->getLimitedValue(Limit);
  }

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantInt;
  }
  unsigned getUseOperandNo(const Use &Use) const override {
    llvm_unreachable("ConstantInt has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantInt>(Val) && "Expected a ConstantInst!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

// TODO: This should inherit from ConstantData.
class ConstantFP final : public Constant {
  ConstantFP(llvm::ConstantFP *C, Context &Ctx)
      : Constant(ClassID::ConstantFP, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// This returns a ConstantFP, or a vector containing a splat of a ConstantFP,
  /// for the specified value in the specified type. This should only be used
  /// for simple constant values like 2.0/1.0 etc, that are known-valid both as
  /// host double and as the target format.
  static Constant *get(Type *Ty, double V);

  /// If Ty is a vector type, return a Constant with a splat of the given
  /// value. Otherwise return a ConstantFP for the given value.
  static Constant *get(Type *Ty, const APFloat &V);

  static Constant *get(Type *Ty, StringRef Str);

  static ConstantFP *get(const APFloat &V, Context &Ctx);

  static Constant *getNaN(Type *Ty, bool Negative = false,
                          uint64_t Payload = 0);
  static Constant *getQNaN(Type *Ty, bool Negative = false,
                           APInt *Payload = nullptr);
  static Constant *getSNaN(Type *Ty, bool Negative = false,
                           APInt *Payload = nullptr);
  static Constant *getZero(Type *Ty, bool Negative = false);

  static Constant *getNegativeZero(Type *Ty);
  static Constant *getInfinity(Type *Ty, bool Negative = false);

  /// Return true if Ty is big enough to represent V.
  static bool isValueValidForType(Type *Ty, const APFloat &V);

  inline const APFloat &getValueAPF() const {
    return cast<llvm::ConstantFP>(Val)->getValueAPF();
  }
  inline const APFloat &getValue() const {
    return cast<llvm::ConstantFP>(Val)->getValue();
  }

  /// Return true if the value is positive or negative zero.
  bool isZero() const { return cast<llvm::ConstantFP>(Val)->isZero(); }

  /// Return true if the sign bit is set.
  bool isNegative() const { return cast<llvm::ConstantFP>(Val)->isNegative(); }

  /// Return true if the value is infinity
  bool isInfinity() const { return cast<llvm::ConstantFP>(Val)->isInfinity(); }

  /// Return true if the value is a NaN.
  bool isNaN() const { return cast<llvm::ConstantFP>(Val)->isNaN(); }

  /// We don't rely on operator== working on double values, as it returns true
  /// for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.  The version with a double operand is retained
  /// because it's so convenient to write isExactlyValue(2.0), but please use
  /// it only for simple constants.
  bool isExactlyValue(const APFloat &V) const {
    return cast<llvm::ConstantFP>(Val)->isExactlyValue(V);
  }

  bool isExactlyValue(double V) const {
    return cast<llvm::ConstantFP>(Val)->isExactlyValue(V);
  }

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantFP;
  }

  // TODO: Better name: getOperandNo(const Use&). Should be private.
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("ConstantFP has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantFP>(Val) && "Expected a ConstantFP!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

/// Base class for aggregate constants (with operands).
class ConstantAggregate : public Constant {
protected:
  ConstantAggregate(ClassID ID, llvm::Constant *C, Context &Ctx)
      : Constant(ID, C, Ctx) {}

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    auto ID = From->getSubclassID();
    return ID == ClassID::ConstantVector || ID == ClassID::ConstantStruct ||
           ID == ClassID::ConstantArray;
  }
};

class ConstantArray final : public ConstantAggregate {
  ConstantArray(llvm::ConstantArray *C, Context &Ctx)
      : ConstantAggregate(ClassID::ConstantArray, C, Ctx) {}
  friend class Context; // For constructor.

public:
  static Constant *get(ArrayType *T, ArrayRef<Constant *> V);
  ArrayType *getType() const;

  // TODO: Missing functions: getType(), getTypeForElements(), getAnon(), get().

  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ConstantArray;
  }
};

class ConstantStruct final : public ConstantAggregate {
  ConstantStruct(llvm::ConstantStruct *C, Context &Ctx)
      : ConstantAggregate(ClassID::ConstantStruct, C, Ctx) {}
  friend class Context; // For constructor.

public:
  static Constant *get(StructType *T, ArrayRef<Constant *> V);

  template <typename... Csts>
  static std::enable_if_t<are_base_of<Constant, Csts...>::value, Constant *>
  get(StructType *T, Csts *...Vs) {
    return get(T, ArrayRef<Constant *>({Vs...}));
  }
  /// Return an anonymous struct that has the specified elements.
  /// If the struct is possibly empty, then you must specify a context.
  static Constant *getAnon(ArrayRef<Constant *> V, bool Packed = false) {
    return get(getTypeForElements(V, Packed), V);
  }
  static Constant *getAnon(Context &Ctx, ArrayRef<Constant *> V,
                           bool Packed = false) {
    return get(getTypeForElements(Ctx, V, Packed), V);
  }
  /// This version of the method allows an empty list.
  static StructType *getTypeForElements(Context &Ctx, ArrayRef<Constant *> V,
                                        bool Packed = false);
  /// Return an anonymous struct type to use for a constant with the specified
  /// set of elements. The list must not be empty.
  static StructType *getTypeForElements(ArrayRef<Constant *> V,
                                        bool Packed = false) {
    assert(!V.empty() &&
           "ConstantStruct::getTypeForElements cannot be called on empty list");
    return getTypeForElements(V[0]->getContext(), V, Packed);
  }

  /// Specialization - reduce amount of casting.
  inline StructType *getType() const {
    return cast<StructType>(Value::getType());
  }

  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ConstantStruct;
  }
};

class ConstantVector final : public ConstantAggregate {
  ConstantVector(llvm::ConstantVector *C, Context &Ctx)
      : ConstantAggregate(ClassID::ConstantVector, C, Ctx) {}
  friend class Context; // For constructor.

public:
  // TODO: Missing functions: getSplat(), getType(), getSplatValue(), get().

  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ConstantVector;
  }
};

// TODO: Inherit from ConstantData.
class ConstantAggregateZero final : public Constant {
  ConstantAggregateZero(llvm::ConstantAggregateZero *C, Context &Ctx)
      : Constant(ClassID::ConstantAggregateZero, C, Ctx) {}
  friend class Context; // For constructor.

public:
  static ConstantAggregateZero *get(Type *Ty);
  /// If this CAZ has array or vector type, return a zero with the right element
  /// type.
  Constant *getSequentialElement() const;
  /// If this CAZ has struct type, return a zero with the right element type for
  /// the specified element.
  Constant *getStructElement(unsigned Elt) const;
  /// Return a zero of the right value for the specified GEP index if we can,
  /// otherwise return null (e.g. if C is a ConstantExpr).
  Constant *getElementValue(Constant *C) const;
  /// Return a zero of the right value for the specified GEP index.
  Constant *getElementValue(unsigned Idx) const;
  /// Return the number of elements in the array, vector, or struct.
  ElementCount getElementCount() const {
    return cast<llvm::ConstantAggregateZero>(Val)->getElementCount();
  }

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantAggregateZero;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("ConstantAggregateZero has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantAggregateZero>(Val) && "Expected a CAZ!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

// TODO: Inherit from ConstantData.
class ConstantPointerNull final : public Constant {
  ConstantPointerNull(llvm::ConstantPointerNull *C, Context &Ctx)
      : Constant(ClassID::ConstantPointerNull, C, Ctx) {}
  friend class Context; // For constructor.

public:
  static ConstantPointerNull *get(PointerType *Ty);

  PointerType *getType() const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantPointerNull;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("ConstantPointerNull has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantPointerNull>(Val) && "Expected a CPNull!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

// TODO: Inherit from ConstantData.
class UndefValue : public Constant {
protected:
  UndefValue(llvm::UndefValue *C, Context &Ctx)
      : Constant(ClassID::UndefValue, C, Ctx) {}
  UndefValue(ClassID ID, llvm::Constant *C, Context &Ctx)
      : Constant(ID, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Static factory methods - Return an 'undef' object of the specified type.
  static UndefValue *get(Type *T);

  /// If this Undef has array or vector type, return a undef with the right
  /// element type.
  UndefValue *getSequentialElement() const;

  /// If this undef has struct type, return a undef with the right element type
  /// for the specified element.
  UndefValue *getStructElement(unsigned Elt) const;

  /// Return an undef of the right value for the specified GEP index if we can,
  /// otherwise return null (e.g. if C is a ConstantExpr).
  UndefValue *getElementValue(Constant *C) const;

  /// Return an undef of the right value for the specified GEP index.
  UndefValue *getElementValue(unsigned Idx) const;

  /// Return the number of elements in the array, vector, or struct.
  unsigned getNumElements() const {
    return cast<llvm::UndefValue>(Val)->getNumElements();
  }

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::UndefValue ||
           From->getSubclassID() == ClassID::PoisonValue;
  }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("UndefValue has no operands!");
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::UndefValue>(Val) && "Expected an UndefValue!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class PoisonValue final : public UndefValue {
  PoisonValue(llvm::PoisonValue *C, Context &Ctx)
      : UndefValue(ClassID::PoisonValue, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Static factory methods - Return an 'poison' object of the specified type.
  static PoisonValue *get(Type *T);

  /// If this poison has array or vector type, return a poison with the right
  /// element type.
  PoisonValue *getSequentialElement() const;

  /// If this poison has struct type, return a poison with the right element
  /// type for the specified element.
  PoisonValue *getStructElement(unsigned Elt) const;

  /// Return an poison of the right value for the specified GEP index if we can,
  /// otherwise return null (e.g. if C is a ConstantExpr).
  PoisonValue *getElementValue(Constant *C) const;

  /// Return an poison of the right value for the specified GEP index.
  PoisonValue *getElementValue(unsigned Idx) const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::PoisonValue;
  }
#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::PoisonValue>(Val) && "Expected a PoisonValue!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class GlobalValue : public Constant {
protected:
  GlobalValue(ClassID ID, llvm::GlobalValue *C, Context &Ctx)
      : Constant(ID, C, Ctx) {}
  friend class Context; // For constructor.

public:
  using LinkageTypes = llvm::GlobalValue::LinkageTypes;
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    switch (From->getSubclassID()) {
    case ClassID::Function:
    case ClassID::GlobalVariable:
    case ClassID::GlobalAlias:
    case ClassID::GlobalIFunc:
      return true;
    default:
      return false;
    }
  }

  unsigned getAddressSpace() const {
    return cast<llvm::GlobalValue>(Val)->getAddressSpace();
  }
  bool hasGlobalUnnamedAddr() const {
    return cast<llvm::GlobalValue>(Val)->hasGlobalUnnamedAddr();
  }

  /// Returns true if this value's address is not significant in this module.
  /// This attribute is intended to be used only by the code generator and LTO
  /// to allow the linker to decide whether the global needs to be in the symbol
  /// table. It should probably not be used in optimizations, as the value may
  /// have uses outside the module; use hasGlobalUnnamedAddr() instead.
  bool hasAtLeastLocalUnnamedAddr() const {
    return cast<llvm::GlobalValue>(Val)->hasAtLeastLocalUnnamedAddr();
  }

  using UnnamedAddr = llvm::GlobalValue::UnnamedAddr;

  UnnamedAddr getUnnamedAddr() const {
    return cast<llvm::GlobalValue>(Val)->getUnnamedAddr();
  }
  void setUnnamedAddr(UnnamedAddr V);

  static UnnamedAddr getMinUnnamedAddr(UnnamedAddr A, UnnamedAddr B) {
    return llvm::GlobalValue::getMinUnnamedAddr(A, B);
  }

  bool hasComdat() const { return cast<llvm::GlobalValue>(Val)->hasComdat(); }

  // TODO: We need a SandboxIR Comdat if we want to implement getComdat().
  using VisibilityTypes = llvm::GlobalValue::VisibilityTypes;
  VisibilityTypes getVisibility() const {
    return cast<llvm::GlobalValue>(Val)->getVisibility();
  }
  bool hasDefaultVisibility() const {
    return cast<llvm::GlobalValue>(Val)->hasDefaultVisibility();
  }
  bool hasHiddenVisibility() const {
    return cast<llvm::GlobalValue>(Val)->hasHiddenVisibility();
  }
  bool hasProtectedVisibility() const {
    return cast<llvm::GlobalValue>(Val)->hasProtectedVisibility();
  }
  void setVisibility(VisibilityTypes V);

  // TODO: Add missing functions.
};

class GlobalObject : public GlobalValue {
protected:
  GlobalObject(ClassID ID, llvm::GlobalObject *C, Context &Ctx)
      : GlobalValue(ID, C, Ctx) {}
  friend class Context; // For constructor.
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    switch (From->getSubclassID()) {
    case ClassID::Function:
    case ClassID::GlobalVariable:
    case ClassID::GlobalIFunc:
      return true;
    default:
      return false;
    }
  }

  /// FIXME: Remove this function once transition to Align is over.
  uint64_t getAlignment() const {
    return cast<llvm::GlobalObject>(Val)->getAlignment();
  }

  /// Returns the alignment of the given variable or function.
  ///
  /// Note that for functions this is the alignment of the code, not the
  /// alignment of a function pointer.
  MaybeAlign getAlign() const {
    return cast<llvm::GlobalObject>(Val)->getAlign();
  }

  // TODO: Add missing: setAlignment(Align)

  /// Sets the alignment attribute of the GlobalObject.
  /// This method will be deprecated as the alignment property should always be
  /// defined.
  void setAlignment(MaybeAlign Align);

  unsigned getGlobalObjectSubClassData() const {
    return cast<llvm::GlobalObject>(Val)->getGlobalObjectSubClassData();
  }

  void setGlobalObjectSubClassData(unsigned V);

  /// Check if this global has a custom object file section.
  ///
  /// This is more efficient than calling getSection() and checking for an empty
  /// string.
  bool hasSection() const {
    return cast<llvm::GlobalObject>(Val)->hasSection();
  }

  /// Get the custom section of this global if it has one.
  ///
  /// If this global does not have a custom section, this will be empty and the
  /// default object file section (.text, .data, etc) will be used.
  StringRef getSection() const {
    return cast<llvm::GlobalObject>(Val)->getSection();
  }

  /// Change the section for this global.
  ///
  /// Setting the section to the empty string tells LLVM to choose an
  /// appropriate default object file section.
  void setSection(StringRef S);

  bool hasComdat() const { return cast<llvm::GlobalObject>(Val)->hasComdat(); }

  // TODO: implement get/setComdat(), etc. once we have a sandboxir::Comdat.

  // TODO: We currently don't support Metadata in sandboxir so all
  // Metadata-related functions are missing.

  using VCallVisibility = llvm::GlobalObject::VCallVisibility;

  VCallVisibility getVCallVisibility() const {
    return cast<llvm::GlobalObject>(Val)->getVCallVisibility();
  }

  /// Returns true if the alignment of the value can be unilaterally
  /// increased.
  ///
  /// Note that for functions this is the alignment of the code, not the
  /// alignment of a function pointer.
  bool canIncreaseAlignment() const {
    return cast<llvm::GlobalObject>(Val)->canIncreaseAlignment();
  }
};

/// Provides API functions, like getIterator() and getReverseIterator() to
/// GlobalIFunc, Function, GlobalVariable and GlobalAlias. In LLVM IR these are
/// provided by ilist_node.
template <typename GlobalT, typename LLVMGlobalT, typename ParentT,
          typename LLVMParentT>
class GlobalWithNodeAPI : public ParentT {
  /// Helper for mapped_iterator.
  struct LLVMGVToGV {
    Context &Ctx;
    LLVMGVToGV(Context &Ctx) : Ctx(Ctx) {}
    GlobalT &operator()(LLVMGlobalT &LLVMGV) const;
  };

public:
  GlobalWithNodeAPI(Value::ClassID ID, LLVMParentT *C, Context &Ctx)
      : ParentT(ID, C, Ctx) {}

  Module *getParent() const {
    llvm::Module *LLVMM = cast<LLVMGlobalT>(this->Val)->getParent();
    return this->Ctx.getModule(LLVMM);
  }

  using iterator = mapped_iterator<
      decltype(static_cast<LLVMGlobalT *>(nullptr)->getIterator()), LLVMGVToGV>;
  using reverse_iterator = mapped_iterator<
      decltype(static_cast<LLVMGlobalT *>(nullptr)->getReverseIterator()),
      LLVMGVToGV>;
  iterator getIterator() const {
    auto *LLVMGV = cast<LLVMGlobalT>(this->Val);
    LLVMGVToGV ToGV(this->Ctx);
    return map_iterator(LLVMGV->getIterator(), ToGV);
  }
  reverse_iterator getReverseIterator() const {
    auto *LLVMGV = cast<LLVMGlobalT>(this->Val);
    LLVMGVToGV ToGV(this->Ctx);
    return map_iterator(LLVMGV->getReverseIterator(), ToGV);
  }
};

class GlobalIFunc final
    : public GlobalWithNodeAPI<GlobalIFunc, llvm::GlobalIFunc, GlobalObject,
                               llvm::GlobalObject> {
  GlobalIFunc(llvm::GlobalObject *C, Context &Ctx)
      : GlobalWithNodeAPI(ClassID::GlobalIFunc, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::GlobalIFunc;
  }

  // TODO: Missing create() because we don't have a sandboxir::Module yet.

  // TODO: Missing functions: copyAttributesFrom(), removeFromParent(),
  // eraseFromParent()

  void setResolver(Constant *Resolver);

  Constant *getResolver() const;

  // Return the resolver function after peeling off potential ConstantExpr
  // indirection.
  Function *getResolverFunction();
  const Function *getResolverFunction() const {
    return const_cast<GlobalIFunc *>(this)->getResolverFunction();
  }

  static bool isValidLinkage(LinkageTypes L) {
    return llvm::GlobalIFunc::isValidLinkage(L);
  }

  // TODO: Missing applyAlongResolverPath().

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::GlobalIFunc>(Val) && "Expected a GlobalIFunc!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class GlobalVariable final
    : public GlobalWithNodeAPI<GlobalVariable, llvm::GlobalVariable,
                               GlobalObject, llvm::GlobalObject> {
  GlobalVariable(llvm::GlobalObject *C, Context &Ctx)
      : GlobalWithNodeAPI(ClassID::GlobalVariable, C, Ctx) {}
  friend class Context; // For constructor.

  /// Helper for mapped_iterator.
  struct LLVMGVToGV {
    Context &Ctx;
    LLVMGVToGV(Context &Ctx) : Ctx(Ctx) {}
    GlobalVariable &operator()(llvm::GlobalVariable &LLVMGV) const;
  };

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::GlobalVariable;
  }

  /// Definitions have initializers, declarations don't.
  ///
  inline bool hasInitializer() const {
    return cast<llvm::GlobalVariable>(Val)->hasInitializer();
  }

  /// hasDefinitiveInitializer - Whether the global variable has an initializer,
  /// and any other instances of the global (this can happen due to weak
  /// linkage) are guaranteed to have the same initializer.
  ///
  /// Note that if you want to transform a global, you must use
  /// hasUniqueInitializer() instead, because of the *_odr linkage type.
  ///
  /// Example:
  ///
  /// @a = global SomeType* null - Initializer is both definitive and unique.
  ///
  /// @b = global weak SomeType* null - Initializer is neither definitive nor
  /// unique.
  ///
  /// @c = global weak_odr SomeType* null - Initializer is definitive, but not
  /// unique.
  inline bool hasDefinitiveInitializer() const {
    return cast<llvm::GlobalVariable>(Val)->hasDefinitiveInitializer();
  }

  /// hasUniqueInitializer - Whether the global variable has an initializer, and
  /// any changes made to the initializer will turn up in the final executable.
  inline bool hasUniqueInitializer() const {
    return cast<llvm::GlobalVariable>(Val)->hasUniqueInitializer();
  }

  /// getInitializer - Return the initializer for this global variable.  It is
  /// illegal to call this method if the global is external, because we cannot
  /// tell what the value is initialized to!
  ///
  Constant *getInitializer() const;
  /// setInitializer - Sets the initializer for this global variable, removing
  /// any existing initializer if InitVal==NULL. The initializer must have the
  /// type getValueType().
  void setInitializer(Constant *InitVal);

  // TODO: Add missing replaceInitializer(). Requires special tracker

  /// If the value is a global constant, its value is immutable throughout the
  /// runtime execution of the program.  Assigning a value into the constant
  /// leads to undefined behavior.
  ///
  bool isConstant() const {
    return cast<llvm::GlobalVariable>(Val)->isConstant();
  }
  void setConstant(bool V);

  bool isExternallyInitialized() const {
    return cast<llvm::GlobalVariable>(Val)->isExternallyInitialized();
  }
  void setExternallyInitialized(bool Val);

  // TODO: Missing copyAttributesFrom()

  // TODO: Missing removeFromParent(), eraseFromParent(), dropAllReferences()

  // TODO: Missing addDebugInfo(), getDebugInfo()

  // TODO: Missing attribute setter functions: addAttribute(), setAttributes().
  //       There seems to be no removeAttribute() so we can't undo them.

  /// Return true if the attribute exists.
  bool hasAttribute(Attribute::AttrKind Kind) const {
    return cast<llvm::GlobalVariable>(Val)->hasAttribute(Kind);
  }

  /// Return true if the attribute exists.
  bool hasAttribute(StringRef Kind) const {
    return cast<llvm::GlobalVariable>(Val)->hasAttribute(Kind);
  }

  /// Return true if any attributes exist.
  bool hasAttributes() const {
    return cast<llvm::GlobalVariable>(Val)->hasAttributes();
  }

  /// Return the attribute object.
  Attribute getAttribute(Attribute::AttrKind Kind) const {
    return cast<llvm::GlobalVariable>(Val)->getAttribute(Kind);
  }

  /// Return the attribute object.
  Attribute getAttribute(StringRef Kind) const {
    return cast<llvm::GlobalVariable>(Val)->getAttribute(Kind);
  }

  /// Return the attribute set for this global
  AttributeSet getAttributes() const {
    return cast<llvm::GlobalVariable>(Val)->getAttributes();
  }

  /// Return attribute set as list with index.
  /// FIXME: This may not be required once ValueEnumerators
  /// in bitcode-writer can enumerate attribute-set.
  AttributeList getAttributesAsList(unsigned Index) const {
    return cast<llvm::GlobalVariable>(Val)->getAttributesAsList(Index);
  }

  /// Check if section name is present
  bool hasImplicitSection() const {
    return cast<llvm::GlobalVariable>(Val)->hasImplicitSection();
  }

  /// Get the custom code model raw value of this global.
  ///
  unsigned getCodeModelRaw() const {
    return cast<llvm::GlobalVariable>(Val)->getCodeModelRaw();
  }

  /// Get the custom code model of this global if it has one.
  ///
  /// If this global does not have a custom code model, the empty instance
  /// will be returned.
  std::optional<CodeModel::Model> getCodeModel() const {
    return cast<llvm::GlobalVariable>(Val)->getCodeModel();
  }

  // TODO: Missing setCodeModel(). Requires custom tracker.

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::GlobalVariable>(Val) && "Expected a GlobalVariable!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class GlobalAlias final
    : public GlobalWithNodeAPI<GlobalAlias, llvm::GlobalAlias, GlobalValue,
                               llvm::GlobalValue> {
  GlobalAlias(llvm::GlobalAlias *C, Context &Ctx)
      : GlobalWithNodeAPI(ClassID::GlobalAlias, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::GlobalAlias;
  }

  // TODO: Missing create() due to unimplemented sandboxir::Module.

  // TODO: Missing copyAttributresFrom().
  // TODO: Missing removeFromParent(), eraseFromParent().

  void setAliasee(Constant *Aliasee);
  Constant *getAliasee() const;

  const GlobalObject *getAliaseeObject() const;
  GlobalObject *getAliaseeObject() {
    return const_cast<GlobalObject *>(
        static_cast<const GlobalAlias *>(this)->getAliaseeObject());
  }

  static bool isValidLinkage(LinkageTypes L) {
    return llvm::GlobalAlias::isValidLinkage(L);
  }
};

class NoCFIValue final : public Constant {
  NoCFIValue(llvm::NoCFIValue *C, Context &Ctx)
      : Constant(ClassID::NoCFIValue, C, Ctx) {}
  friend class Context; // For constructor.

  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }

public:
  /// Return a NoCFIValue for the specified function.
  static NoCFIValue *get(GlobalValue *GV);

  GlobalValue *getGlobalValue() const;

  /// NoCFIValue is always a pointer.
  PointerType *getType() const;
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::NoCFIValue;
  }

  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::NoCFIValue>(Val) && "Expected a NoCFIValue!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class ConstantPtrAuth final : public Constant {
  ConstantPtrAuth(llvm::ConstantPtrAuth *C, Context &Ctx)
      : Constant(ClassID::ConstantPtrAuth, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return a pointer signed with the specified parameters.
  static ConstantPtrAuth *get(Constant *Ptr, ConstantInt *Key,
                              ConstantInt *Disc, Constant *AddrDisc);
  /// The pointer that is signed in this ptrauth signed pointer.
  Constant *getPointer() const;

  /// The Key ID, an i32 constant.
  ConstantInt *getKey() const;

  /// The integer discriminator, an i64 constant, or 0.
  ConstantInt *getDiscriminator() const;

  /// The address discriminator if any, or the null constant.
  /// If present, this must be a value equivalent to the storage location of
  /// the only global-initializer user of the ptrauth signed pointer.
  Constant *getAddrDiscriminator() const;

  /// Whether there is any non-null address discriminator.
  bool hasAddressDiscriminator() const {
    return cast<llvm::ConstantPtrAuth>(Val)->hasAddressDiscriminator();
  }

  /// Whether the address uses a special address discriminator.
  /// These discriminators can't be used in real pointer-auth values; they
  /// can only be used in "prototype" values that indicate how some real
  /// schema is supposed to be produced.
  bool hasSpecialAddressDiscriminator(uint64_t Value) const {
    return cast<llvm::ConstantPtrAuth>(Val)->hasSpecialAddressDiscriminator(
        Value);
  }

  /// Check whether an authentication operation with key \p Key and (possibly
  /// blended) discriminator \p Discriminator is known to be compatible with
  /// this ptrauth signed pointer.
  bool isKnownCompatibleWith(const Value *Key, const Value *Discriminator,
                             const DataLayout &DL) const {
    return cast<llvm::ConstantPtrAuth>(Val)->isKnownCompatibleWith(
        Key->Val, Discriminator->Val, DL);
  }

  /// Produce a new ptrauth expression signing the given value using
  /// the same schema as is stored in one.
  ConstantPtrAuth *getWithSameSchema(Constant *Pointer) const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantPtrAuth;
  }
};

class ConstantExpr : public Constant {
  ConstantExpr(llvm::ConstantExpr *C, Context &Ctx)
      : Constant(ClassID::ConstantExpr, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantExpr;
  }
  // TODO: Missing functions.
};

class BlockAddress final : public Constant {
  BlockAddress(llvm::BlockAddress *C, Context &Ctx)
      : Constant(ClassID::BlockAddress, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return a BlockAddress for the specified function and basic block.
  static BlockAddress *get(Function *F, BasicBlock *BB);

  /// Return a BlockAddress for the specified basic block.  The basic
  /// block must be embedded into a function.
  static BlockAddress *get(BasicBlock *BB);

  /// Lookup an existing \c BlockAddress constant for the given BasicBlock.
  ///
  /// \returns 0 if \c !BB->hasAddressTaken(), otherwise the \c BlockAddress.
  static BlockAddress *lookup(const BasicBlock *BB);

  Function *getFunction() const;
  BasicBlock *getBasicBlock() const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::BlockAddress;
  }
};

class DSOLocalEquivalent final : public Constant {
  DSOLocalEquivalent(llvm::DSOLocalEquivalent *C, Context &Ctx)
      : Constant(ClassID::DSOLocalEquivalent, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return a DSOLocalEquivalent for the specified global value.
  static DSOLocalEquivalent *get(GlobalValue *GV);

  GlobalValue *getGlobalValue() const;

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::DSOLocalEquivalent;
  }

  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("DSOLocalEquivalent has no operands!");
  }

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::DSOLocalEquivalent>(Val) &&
           "Expected a DSOLocalEquivalent!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

// TODO: This should inherit from ConstantData.
class ConstantTokenNone final : public Constant {
  ConstantTokenNone(llvm::ConstantTokenNone *C, Context &Ctx)
      : Constant(ClassID::ConstantTokenNone, C, Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return the ConstantTokenNone.
  static ConstantTokenNone *get(Context &Ctx);

  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::ConstantTokenNone;
  }

  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("ConstantTokenNone has no operands!");
  }

#ifndef NDEBUG
  void verify() const override {
    assert(isa<llvm::ConstantTokenNone>(Val) &&
           "Expected a ConstantTokenNone!");
  }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class Function : public GlobalWithNodeAPI<Function, llvm::Function,
                                          GlobalObject, llvm::GlobalObject> {
  /// Helper for mapped_iterator.
  struct LLVMBBToBB {
    Context &Ctx;
    LLVMBBToBB(Context &Ctx) : Ctx(Ctx) {}
    BasicBlock &operator()(llvm::BasicBlock &LLVMBB) const {
      return *cast<BasicBlock>(Ctx.getValue(&LLVMBB));
    }
  };
  /// Use Context::createFunction() instead.
  Function(llvm::Function *F, sandboxir::Context &Ctx)
      : GlobalWithNodeAPI(ClassID::Function, F, Ctx) {}
  friend class Context; // For constructor.

public:
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Function;
  }

  Module *getParent() {
    return Ctx.getModule(cast<llvm::Function>(Val)->getParent());
  }

  Argument *getArg(unsigned Idx) const {
    llvm::Argument *Arg = cast<llvm::Function>(Val)->getArg(Idx);
    return cast<Argument>(Ctx.getValue(Arg));
  }

  size_t arg_size() const { return cast<llvm::Function>(Val)->arg_size(); }
  bool arg_empty() const { return cast<llvm::Function>(Val)->arg_empty(); }

  using iterator = mapped_iterator<llvm::Function::iterator, LLVMBBToBB>;
  iterator begin() const {
    LLVMBBToBB BBGetter(Ctx);
    return iterator(cast<llvm::Function>(Val)->begin(), BBGetter);
  }
  iterator end() const {
    LLVMBBToBB BBGetter(Ctx);
    return iterator(cast<llvm::Function>(Val)->end(), BBGetter);
  }
  FunctionType *getFunctionType() const;

#ifndef NDEBUG
  void verify() const final {
    assert(isa<llvm::Function>(Val) && "Expected Function!");
  }
  void dumpNameAndArgs(raw_ostream &OS) const;
  void dumpOS(raw_ostream &OS) const final;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_CONSTANT_H
