//===-- llvm/Constants.h - Constant class subclass definitions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file This file contains the declarations for the subclasses of Constant, 
/// which represent the different flavors of constant values that live in LLVM.
/// Note that Constants are immutable (once created they never change) and are 
/// fully shared by structural equivalence.  This means that two structurally
/// equivalent constants will always have the same address.  Constant's are
/// created on demand as needed and never deleted: thus clients don't have to
/// worry about the lifetime of the objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANTS_H
#define LLVM_CONSTANTS_H

#include "llvm/Constant.h"
#include "llvm/Type.h"
#include "llvm/OperandTraits.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class ArrayType;
class StructType;
class PointerType;
class VectorType;

template<class ConstantClass, class TypeClass, class ValType>
struct ConstantCreator;
template<class ConstantClass, class TypeClass>
struct ConvertConstantType;

//===----------------------------------------------------------------------===//
/// This is the shared class of boolean and integer constants. This class 
/// represents both boolean and integral constants.
/// @brief Class for constant integers.
class ConstantInt : public Constant {
  static ConstantInt *TheTrueVal, *TheFalseVal;
  void *operator new(size_t, unsigned);  // DO NOT IMPLEMENT
  ConstantInt(const ConstantInt &);      // DO NOT IMPLEMENT
  ConstantInt(const IntegerType *Ty, const APInt& V);
  APInt Val;
protected:
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
public:
  /// Return the constant as an APInt value reference. This allows clients to
  /// obtain a copy of the value, with all its precision in tact.
  /// @brief Return the constant's value.
  inline const APInt& getValue() const {
    return Val;
  }
  
  /// getBitWidth - Return the bitwidth of this constant.
  unsigned getBitWidth() const { return Val.getBitWidth(); }

  /// Return the constant as a 64-bit unsigned integer value after it
  /// has been zero extended as appropriate for the type of this constant. Note
  /// that this method can assert if the value does not fit in 64 bits.
  /// @deprecated
  /// @brief Return the zero extended value.
  inline uint64_t getZExtValue() const {
    return Val.getZExtValue();
  }

  /// Return the constant as a 64-bit integer value after it has been sign
  /// sign extended as appropriate for the type of this constant. Note that
  /// this method can assert if the value does not fit in 64 bits.
  /// @deprecated
  /// @brief Return the sign extended value.
  inline int64_t getSExtValue() const {
    return Val.getSExtValue();
  }

  /// A helper method that can be used to determine if the constant contained 
  /// within is equal to a constant.  This only works for very small values, 
  /// because this is all that can be represented with all types.
  /// @brief Determine if this constant's value is same as an unsigned char.
  bool equalsInt(uint64_t V) const {
    return Val == V;
  }

  /// getTrue/getFalse - Return the singleton true/false values.
  static inline ConstantInt *getTrue() {
    if (TheTrueVal) return TheTrueVal;
    return CreateTrueFalseVals(true);
  }
  static inline ConstantInt *getFalse() {
    if (TheFalseVal) return TheFalseVal;
    return CreateTrueFalseVals(false);
  }

  /// Return a ConstantInt with the specified value for the specified type. The
  /// value V will be canonicalized to a an unsigned APInt. Accessing it with
  /// either getSExtValue() or getZExtValue() will yield a correctly sized and
  /// signed value for the type Ty.
  /// @brief Get a ConstantInt for a specific value.
  static ConstantInt *get(const Type *Ty, uint64_t V, bool isSigned = false);

  /// Return a ConstantInt with the specified value and an implied Type. The
  /// type is the integer type that corresponds to the bit width of the value.
  static ConstantInt *get(const APInt &V);

  /// getType - Specialize the getType() method to always return an IntegerType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const IntegerType *getType() const {
    return reinterpret_cast<const IntegerType*>(Value::getType());
  }

  /// This static method returns true if the type Ty is big enough to 
  /// represent the value V. This can be used to avoid having the get method 
  /// assert when V is larger than Ty can represent. Note that there are two
  /// versions of this method, one for unsigned and one for signed integers.
  /// Although ConstantInt canonicalizes everything to an unsigned integer, 
  /// the signed version avoids callers having to convert a signed quantity
  /// to the appropriate unsigned type before calling the method.
  /// @returns true if V is a valid value for type Ty
  /// @brief Determine if the value is in range for the given type.
  static bool isValueValidForType(const Type *Ty, uint64_t V);
  static bool isValueValidForType(const Type *Ty, int64_t V);

  /// This function will return true iff this constant represents the "null"
  /// value that would be returned by the getNullValue method.
  /// @returns true if this is the null integer value.
  /// @brief Determine if the value is null.
  virtual bool isNullValue() const { 
    return Val == 0; 
  }

  /// This is just a convenience method to make client code smaller for a
  /// common code. It also correctly performs the comparison without the
  /// potential for an assertion from getZExtValue().
  bool isZero() const {
    return Val == 0;
  }

  /// This is just a convenience method to make client code smaller for a 
  /// common case. It also correctly performs the comparison without the
  /// potential for an assertion from getZExtValue().
  /// @brief Determine if the value is one.
  bool isOne() const {
    return Val == 1;
  }

  /// This function will return true iff every bit in this constant is set
  /// to true.
  /// @returns true iff this constant's bits are all set to true.
  /// @brief Determine if the value is all ones.
  bool isAllOnesValue() const { 
    return Val.isAllOnesValue();
  }

  /// This function will return true iff this constant represents the largest
  /// value that may be represented by the constant's type.
  /// @returns true iff this is the largest value that may be represented 
  /// by this type.
  /// @brief Determine if the value is maximal.
  bool isMaxValue(bool isSigned) const {
    if (isSigned) 
      return Val.isMaxSignedValue();
    else
      return Val.isMaxValue();
  }

  /// This function will return true iff this constant represents the smallest
  /// value that may be represented by this constant's type.
  /// @returns true if this is the smallest value that may be represented by 
  /// this type.
  /// @brief Determine if the value is minimal.
  bool isMinValue(bool isSigned) const {
    if (isSigned) 
      return Val.isMinSignedValue();
    else
      return Val.isMinValue();
  }

  /// This function will return true iff this constant represents a value with
  /// active bits bigger than 64 bits or a value greater than the given uint64_t
  /// value.
  /// @returns true iff this constant is greater or equal to the given number.
  /// @brief Determine if the value is greater or equal to the given number.
  bool uge(uint64_t Num) {
    return Val.getActiveBits() > 64 || Val.getZExtValue() >= Num;
  }

  /// getLimitedValue - If the value is smaller than the specified limit,
  /// return it, otherwise return the limit value.  This causes the value
  /// to saturate to the limit.
  /// @returns the min of the value of the constant and the specified value
  /// @brief Get the constant's value with a saturation limit
  uint64_t getLimitedValue(uint64_t Limit = ~0ULL) const {
    return Val.getLimitedValue(Limit);
  }

  /// @returns the value for an integer constant of the given type that has all
  /// its bits set to true.
  /// @brief Get the all ones value
  static ConstantInt *getAllOnesValue(const Type *Ty);

  /// @brief Methods to support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const ConstantInt *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == ConstantIntVal;
  }
  static void ResetTrueFalse() { TheTrueVal = TheFalseVal = 0; }
private:
  static ConstantInt *CreateTrueFalseVals(bool WhichOne);
};


//===----------------------------------------------------------------------===//
/// ConstantFP - Floating Point Values [float, double]
///
class ConstantFP : public Constant {
  APFloat Val;
  void *operator new(size_t, unsigned);// DO NOT IMPLEMENT
  ConstantFP(const ConstantFP &);      // DO NOT IMPLEMENT
protected:
  ConstantFP(const Type *Ty, const APFloat& V);
protected:
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantFP *get(const APFloat &V);

  /// get() - This returns a constant fp for the specified value in the
  /// specified type.  This should only be used for simple constant values like
  /// 2.0/1.0 etc, that are known-valid both as double and as the target format.
  static ConstantFP *get(const Type *Ty, double V);

  /// isValueValidForType - return true if Ty is big enough to represent V.
  static bool isValueValidForType(const Type *Ty, const APFloat& V);
  inline const APFloat& getValueAPF() const { return Val; }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  Don't depend on == for doubles to tell us it's zero, it
  /// considers -0.0 to be null as well as 0.0.  :(
  virtual bool isNullValue() const;

  // Get a negative zero.
  static ConstantFP *getNegativeZero(const Type* Ty);

  /// isExactlyValue - We don't rely on operator== working on double values, as
  /// it returns true for things that are clearly not equal, like -0.0 and 0.0.
  /// As such, this method can be used to do an exact bit-for-bit comparison of
  /// two floating point values.  The version with a double operand is retained
  /// because it's so convenient to write isExactlyValue(2.0), but please use
  /// it only for simple constants.
  bool isExactlyValue(const APFloat& V) const;

  bool isExactlyValue(double V) const {
    bool ignored;
    // convert is not supported on this type
    if (&Val.getSemantics() == &APFloat::PPCDoubleDouble)
      return false;
    APFloat FV(V);
    FV.convert(Val.getSemantics(), APFloat::rmNearestTiesToEven, &ignored);
    return isExactlyValue(FV);
  }
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantFP *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == ConstantFPVal;
  }
};

//===----------------------------------------------------------------------===//
/// ConstantAggregateZero - All zero aggregate value
///
class ConstantAggregateZero : public Constant {
  friend struct ConstantCreator<ConstantAggregateZero, Type, char>;
  void *operator new(size_t, unsigned);                      // DO NOT IMPLEMENT
  ConstantAggregateZero(const ConstantAggregateZero &);      // DO NOT IMPLEMENT
protected:
  explicit ConstantAggregateZero(const Type *ty)
    : Constant(ty, ConstantAggregateZeroVal, 0, 0) {}
protected:
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
public:
  /// get() - static factory method for creating a null aggregate.  It is
  /// illegal to call this method with a non-aggregate type.
  static ConstantAggregateZero *get(const Type *Ty);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return true; }

  virtual void destroyConstant();

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  ///
  static bool classof(const ConstantAggregateZero *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == ConstantAggregateZeroVal;
  }
};


//===----------------------------------------------------------------------===//
/// ConstantArray - Constant Array Declarations
///
class ConstantArray : public Constant {
  friend struct ConstantCreator<ConstantArray, ArrayType,
                                    std::vector<Constant*> >;
  ConstantArray(const ConstantArray &);      // DO NOT IMPLEMENT
protected:
  ConstantArray(const ArrayType *T, const std::vector<Constant*> &Val);
public:
  /// get() - Static factory methods - Return objects of the specified value
  static Constant *get(const ArrayType *T, const std::vector<Constant*> &);
  static Constant *get(const ArrayType *T,
                       Constant*const*Vals, unsigned NumVals) {
    // FIXME: make this the primary ctor method.
    return get(T, std::vector<Constant*>(Vals, Vals+NumVals));
  }

  /// This method constructs a ConstantArray and initializes it with a text
  /// string. The default behavior (AddNull==true) causes a null terminator to
  /// be placed at the end of the array. This effectively increases the length
  /// of the array by one (you've been warned).  However, in some situations 
  /// this is not desired so if AddNull==false then the string is copied without
  /// null termination. 
  static Constant *get(const std::string &Initializer, bool AddNull = true);

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Constant);

  /// getType - Specialize the getType() method to always return an ArrayType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const ArrayType *getType() const {
    return reinterpret_cast<const ArrayType*>(Value::getType());
  }

  /// isString - This method returns true if the array is an array of i8 and
  /// the elements of the array are all ConstantInt's.
  bool isString() const;

  /// isCString - This method returns true if the array is a string (see
  /// @verbatim
  /// isString) and it ends in a null byte \0 and does not contains any other
  /// @endverbatim
  /// null bytes except its terminator.
  bool isCString() const;

  /// getAsString - If this array is isString(), then this method converts the
  /// array to an std::string and returns it.  Otherwise, it asserts out.
  ///
  std::string getAsString() const;

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because zero arrays are always
  /// created as ConstantAggregateZero objects.
  virtual bool isNullValue() const { return false; }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantArray *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == ConstantArrayVal;
  }
};

template <>
struct OperandTraits<ConstantArray> : VariadicOperandTraits<> {
};

DEFINE_TRANSPARENT_CASTED_OPERAND_ACCESSORS(ConstantArray, Constant)

//===----------------------------------------------------------------------===//
// ConstantStruct - Constant Struct Declarations
//
class ConstantStruct : public Constant {
  friend struct ConstantCreator<ConstantStruct, StructType,
                                    std::vector<Constant*> >;
  ConstantStruct(const ConstantStruct &);      // DO NOT IMPLEMENT
protected:
  ConstantStruct(const StructType *T, const std::vector<Constant*> &Val);
public:
  /// get() - Static factory methods - Return objects of the specified value
  ///
  static Constant *get(const StructType *T, const std::vector<Constant*> &V);
  static Constant *get(const std::vector<Constant*> &V, bool Packed = false);
  static Constant *get(Constant*const* Vals, unsigned NumVals,
                       bool Packed = false) {
    // FIXME: make this the primary ctor method.
    return get(std::vector<Constant*>(Vals, Vals+NumVals), Packed);
  }
  
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Constant);

  /// getType() specialization - Reduce amount of casting...
  ///
  inline const StructType *getType() const {
    return reinterpret_cast<const StructType*>(Value::getType());
  }

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because zero structs are always
  /// created as ConstantAggregateZero objects.
  virtual bool isNullValue() const {
    return false;
  }

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantStruct *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == ConstantStructVal;
  }
};

template <>
struct OperandTraits<ConstantStruct> : VariadicOperandTraits<> {
};

DEFINE_TRANSPARENT_CASTED_OPERAND_ACCESSORS(ConstantStruct, Constant)

//===----------------------------------------------------------------------===//
/// ConstantVector - Constant Vector Declarations
///
class ConstantVector : public Constant {
  friend struct ConstantCreator<ConstantVector, VectorType,
                                    std::vector<Constant*> >;
  ConstantVector(const ConstantVector &);      // DO NOT IMPLEMENT
protected:
  ConstantVector(const VectorType *T, const std::vector<Constant*> &Val);
public:
  /// get() - Static factory methods - Return objects of the specified value
  static Constant *get(const VectorType *T, const std::vector<Constant*> &);
  static Constant *get(const std::vector<Constant*> &V);
  static Constant *get(Constant*const* Vals, unsigned NumVals) {
    // FIXME: make this the primary ctor method.
    return get(std::vector<Constant*>(Vals, Vals+NumVals));
  }
  
  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Constant);

  /// getType - Specialize the getType() method to always return a VectorType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const VectorType *getType() const {
    return reinterpret_cast<const VectorType*>(Value::getType());
  }

  /// @returns the value for a vector integer constant of the given type that
  /// has all its bits set to true.
  /// @brief Get the all ones value
  static ConstantVector *getAllOnesValue(const VectorType *Ty);
  
  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.  This always returns false because zero vectors are always
  /// created as ConstantAggregateZero objects.
  virtual bool isNullValue() const { return false; }

  /// This function will return true iff every element in this vector constant
  /// is set to all ones.
  /// @returns true iff this constant's emements are all set to all ones.
  /// @brief Determine if the value is all ones.
  bool isAllOnesValue() const;

  /// getSplatValue - If this is a splat constant, meaning that all of the
  /// elements have the same value, return that value. Otherwise return NULL.
  Constant *getSplatValue();

  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantVector *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == ConstantVectorVal;
  }
};

template <>
struct OperandTraits<ConstantVector> : VariadicOperandTraits<> {
};

DEFINE_TRANSPARENT_CASTED_OPERAND_ACCESSORS(ConstantVector, Constant)

//===----------------------------------------------------------------------===//
/// ConstantPointerNull - a constant pointer value that points to null
///
class ConstantPointerNull : public Constant {
  friend struct ConstantCreator<ConstantPointerNull, PointerType, char>;
  void *operator new(size_t, unsigned);                  // DO NOT IMPLEMENT
  ConstantPointerNull(const ConstantPointerNull &);      // DO NOT IMPLEMENT
protected:
  explicit ConstantPointerNull(const PointerType *T)
    : Constant(reinterpret_cast<const Type*>(T),
               Value::ConstantPointerNullVal, 0, 0) {}

protected:
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
public:
  /// get() - Static factory methods - Return objects of the specified value
  static ConstantPointerNull *get(const PointerType *T);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return true; }

  virtual void destroyConstant();

  /// getType - Specialize the getType() method to always return an PointerType,
  /// which reduces the amount of casting needed in parts of the compiler.
  ///
  inline const PointerType *getType() const {
    return reinterpret_cast<const PointerType*>(Value::getType());
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantPointerNull *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == ConstantPointerNullVal;
  }
};


/// ConstantExpr - a constant value that is initialized with an expression using
/// other constant values.
///
/// This class uses the standard Instruction opcodes to define the various
/// constant expressions.  The Opcode field for the ConstantExpr class is
/// maintained in the Value::SubclassData field.
class ConstantExpr : public Constant {
  friend struct ConstantCreator<ConstantExpr,Type,
                            std::pair<unsigned, std::vector<Constant*> > >;
  friend struct ConvertConstantType<ConstantExpr, Type>;

protected:
  ConstantExpr(const Type *ty, unsigned Opcode, Use *Ops, unsigned NumOps)
    : Constant(ty, ConstantExprVal, Ops, NumOps) {
    // Operation type (an Instruction opcode) is stored as the SubclassData.
    SubclassData = Opcode;
  }

  // These private methods are used by the type resolution code to create
  // ConstantExprs in intermediate forms.
  static Constant *getTy(const Type *Ty, unsigned Opcode,
                         Constant *C1, Constant *C2);
  static Constant *getCompareTy(unsigned short pred, Constant *C1,
                                Constant *C2);
  static Constant *getSelectTy(const Type *Ty,
                               Constant *C1, Constant *C2, Constant *C3);
  static Constant *getGetElementPtrTy(const Type *Ty, Constant *C,
                                      Value* const *Idxs, unsigned NumIdxs);
  static Constant *getExtractElementTy(const Type *Ty, Constant *Val,
                                       Constant *Idx);
  static Constant *getInsertElementTy(const Type *Ty, Constant *Val,
                                      Constant *Elt, Constant *Idx);
  static Constant *getShuffleVectorTy(const Type *Ty, Constant *V1,
                                      Constant *V2, Constant *Mask);
  static Constant *getExtractValueTy(const Type *Ty, Constant *Agg,
                                     const unsigned *Idxs, unsigned NumIdxs);
  static Constant *getInsertValueTy(const Type *Ty, Constant *Agg,
                                    Constant *Val,
                                    const unsigned *Idxs, unsigned NumIdxs);

public:
  // Static methods to construct a ConstantExpr of different kinds.  Note that
  // these methods may return a object that is not an instance of the
  // ConstantExpr class, because they will attempt to fold the constant
  // expression into something simpler if possible.

  /// Cast constant expr
  ///
  static Constant *getTrunc   (Constant *C, const Type *Ty);
  static Constant *getSExt    (Constant *C, const Type *Ty);
  static Constant *getZExt    (Constant *C, const Type *Ty);
  static Constant *getFPTrunc (Constant *C, const Type *Ty);
  static Constant *getFPExtend(Constant *C, const Type *Ty);
  static Constant *getUIToFP  (Constant *C, const Type *Ty);
  static Constant *getSIToFP  (Constant *C, const Type *Ty);
  static Constant *getFPToUI  (Constant *C, const Type *Ty);
  static Constant *getFPToSI  (Constant *C, const Type *Ty);
  static Constant *getPtrToInt(Constant *C, const Type *Ty);
  static Constant *getIntToPtr(Constant *C, const Type *Ty);
  static Constant *getBitCast (Constant *C, const Type *Ty);

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Constant);

  // @brief Convenience function for getting one of the casting operations
  // using a CastOps opcode.
  static Constant *getCast(
    unsigned ops,  ///< The opcode for the conversion
    Constant *C,   ///< The constant to be converted
    const Type *Ty ///< The type to which the constant is converted
  );

  // @brief Create a ZExt or BitCast cast constant expression
  static Constant *getZExtOrBitCast(
    Constant *C,   ///< The constant to zext or bitcast
    const Type *Ty ///< The type to zext or bitcast C to
  );

  // @brief Create a SExt or BitCast cast constant expression 
  static Constant *getSExtOrBitCast(
    Constant *C,   ///< The constant to sext or bitcast
    const Type *Ty ///< The type to sext or bitcast C to
  );

  // @brief Create a Trunc or BitCast cast constant expression
  static Constant *getTruncOrBitCast(
    Constant *C,   ///< The constant to trunc or bitcast
    const Type *Ty ///< The type to trunc or bitcast C to
  );

  /// @brief Create a BitCast or a PtrToInt cast constant expression
  static Constant *getPointerCast(
    Constant *C,   ///< The pointer value to be casted (operand 0)
    const Type *Ty ///< The type to which cast should be made
  );

  /// @brief Create a ZExt, Bitcast or Trunc for integer -> integer casts
  static Constant *getIntegerCast(
    Constant *C,    ///< The integer constant to be casted 
    const Type *Ty, ///< The integer type to cast to
    bool isSigned   ///< Whether C should be treated as signed or not
  );

  /// @brief Create a FPExt, Bitcast or FPTrunc for fp -> fp casts
  static Constant *getFPCast(
    Constant *C,    ///< The integer constant to be casted 
    const Type *Ty ///< The integer type to cast to
  );

  /// @brief Return true if this is a convert constant expression
  bool isCast() const;

  /// @brief Return true if this is a compare constant expression
  bool isCompare() const;

  /// @brief Return true if this is an insertvalue or extractvalue expression,
  /// and the getIndices() method may be used.
  bool hasIndices() const;

  /// Select constant expr
  ///
  static Constant *getSelect(Constant *C, Constant *V1, Constant *V2) {
    return getSelectTy(V1->getType(), C, V1, V2);
  }

  /// getSizeOf constant expr - computes the size of a type in a target
  /// independent way (Note: the return type is an i64).
  ///
  static Constant *getSizeOf(const Type *Ty);

  /// ConstantExpr::get - Return a binary or shift operator constant expression,
  /// folding if possible.
  ///
  static Constant *get(unsigned Opcode, Constant *C1, Constant *C2);

  /// @brief Return an ICmp, FCmp, VICmp, or VFCmp comparison operator constant
  /// expression.
  static Constant *getCompare(unsigned short pred, Constant *C1, Constant *C2);

  /// ConstantExpr::get* - Return some common constants without having to
  /// specify the full Instruction::OPCODE identifier.
  ///
  static Constant *getNeg(Constant *C);
  static Constant *getNot(Constant *C);
  static Constant *getAdd(Constant *C1, Constant *C2);
  static Constant *getSub(Constant *C1, Constant *C2);
  static Constant *getMul(Constant *C1, Constant *C2);
  static Constant *getUDiv(Constant *C1, Constant *C2);
  static Constant *getSDiv(Constant *C1, Constant *C2);
  static Constant *getFDiv(Constant *C1, Constant *C2);
  static Constant *getURem(Constant *C1, Constant *C2); // unsigned rem
  static Constant *getSRem(Constant *C1, Constant *C2); // signed rem
  static Constant *getFRem(Constant *C1, Constant *C2);
  static Constant *getAnd(Constant *C1, Constant *C2);
  static Constant *getOr(Constant *C1, Constant *C2);
  static Constant *getXor(Constant *C1, Constant *C2);
  static Constant *getICmp(unsigned short pred, Constant *LHS, Constant *RHS);
  static Constant *getFCmp(unsigned short pred, Constant *LHS, Constant *RHS);
  static Constant *getVICmp(unsigned short pred, Constant *LHS, Constant *RHS);
  static Constant *getVFCmp(unsigned short pred, Constant *LHS, Constant *RHS);
  static Constant *getShl(Constant *C1, Constant *C2);
  static Constant *getLShr(Constant *C1, Constant *C2);
  static Constant *getAShr(Constant *C1, Constant *C2);

  /// Getelementptr form.  std::vector<Value*> is only accepted for convenience:
  /// all elements must be Constant's.
  ///
  static Constant *getGetElementPtr(Constant *C,
                                    Constant* const *IdxList, unsigned NumIdx);
  static Constant *getGetElementPtr(Constant *C,
                                    Value* const *IdxList, unsigned NumIdx);
  
  static Constant *getExtractElement(Constant *Vec, Constant *Idx);
  static Constant *getInsertElement(Constant *Vec, Constant *Elt,Constant *Idx);
  static Constant *getShuffleVector(Constant *V1, Constant *V2, Constant *Mask);
  static Constant *getExtractValue(Constant *Agg,
                                   const unsigned *IdxList, unsigned NumIdx);
  static Constant *getInsertValue(Constant *Agg, Constant *Val,
                                  const unsigned *IdxList, unsigned NumIdx);

  /// Floating point negation must be implemented with f(x) = -0.0 - x. This
  /// method returns the negative zero constant for floating point or vector
  /// floating point types; for all other types, it returns the null value.
  static Constant *getZeroValueForNegationExpr(const Type *Ty);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return false; }

  /// getOpcode - Return the opcode at the root of this constant expression
  unsigned getOpcode() const { return SubclassData; }

  /// getPredicate - Return the ICMP or FCMP predicate value. Assert if this is
  /// not an ICMP or FCMP constant expression.
  unsigned getPredicate() const;

  /// getIndices - Assert that this is an insertvalue or exactvalue
  /// expression and return the list of indices.
  const SmallVector<unsigned, 4> &getIndices() const;

  /// getOpcodeName - Return a string representation for an opcode.
  const char *getOpcodeName() const;

  /// getWithOperandReplaced - Return a constant expression identical to this
  /// one, but with the specified operand set to the specified value.
  Constant *getWithOperandReplaced(unsigned OpNo, Constant *Op) const;
  
  /// getWithOperands - This returns the current constant expression with the
  /// operands replaced with the specified values.  The specified operands must
  /// match count and type with the existing ones.
  Constant *getWithOperands(const std::vector<Constant*> &Ops) const {
    return getWithOperands(&Ops[0], (unsigned)Ops.size());
  }
  Constant *getWithOperands(Constant* const *Ops, unsigned NumOps) const;
  
  virtual void destroyConstant();
  virtual void replaceUsesOfWithOnConstant(Value *From, Value *To, Use *U);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantExpr *) { return true; }
  static inline bool classof(const Value *V) {
    return V->getValueID() == ConstantExprVal;
  }
};

template <>
struct OperandTraits<ConstantExpr> : VariadicOperandTraits<1> {
};

DEFINE_TRANSPARENT_CASTED_OPERAND_ACCESSORS(ConstantExpr, Constant)

//===----------------------------------------------------------------------===//
/// UndefValue - 'undef' values are things that do not have specified contents.
/// These are used for a variety of purposes, including global variable
/// initializers and operands to instructions.  'undef' values can occur with
/// any type.
///
class UndefValue : public Constant {
  friend struct ConstantCreator<UndefValue, Type, char>;
  void *operator new(size_t, unsigned); // DO NOT IMPLEMENT
  UndefValue(const UndefValue &);      // DO NOT IMPLEMENT
protected:
  explicit UndefValue(const Type *T) : Constant(T, UndefValueVal, 0, 0) {}
protected:
  // allocate space for exactly zero operands
  void *operator new(size_t s) {
    return User::operator new(s, 0);
  }
public:
  /// get() - Static factory methods - Return an 'undef' object of the specified
  /// type.
  ///
  static UndefValue *get(const Type *T);

  /// isNullValue - Return true if this is the value that would be returned by
  /// getNullValue.
  virtual bool isNullValue() const { return false; }

  virtual void destroyConstant();

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const UndefValue *) { return true; }
  static bool classof(const Value *V) {
    return V->getValueID() == UndefValueVal;
  }
};

} // End llvm namespace

#endif
