//===-- llvm/DerivedTypes.h - Classes for handling data types ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of classes that represent "derived
// types".  These are things like "arrays of x" or "structure of x, y, z" or
// "method returning x taking (y,z) as parameters", etc...
//
// The implementations of these classes live in the Type.cpp file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DERIVED_TYPES_H
#define LLVM_DERIVED_TYPES_H

#include "llvm/Type.h"

namespace llvm {

class Value;
template<class ValType, class TypeClass> class TypeMap;
class FunctionValType;
class ArrayValType;
class StructValType;
class PointerValType;
class VectorValType;
class IntegerValType;
class APInt;

class DerivedType : public Type {
  friend class Type;

protected:
  explicit DerivedType(TypeID id) : Type(id) {}

  /// notifyUsesThatTypeBecameConcrete - Notify AbstractTypeUsers of this type
  /// that the current type has transitioned from being abstract to being
  /// concrete.
  ///
  void notifyUsesThatTypeBecameConcrete();

  /// dropAllTypeUses - When this (abstract) type is resolved to be equal to
  /// another (more concrete) type, we must eliminate all references to other
  /// types, to avoid some circular reference problems.
  ///
  void dropAllTypeUses();

public:

  //===--------------------------------------------------------------------===//
  // Abstract Type handling methods - These types have special lifetimes, which
  // are managed by (add|remove)AbstractTypeUser. See comments in
  // AbstractTypeUser.h for more information.

  /// refineAbstractTypeTo - This function is used to when it is discovered that
  /// the 'this' abstract type is actually equivalent to the NewType specified.
  /// This causes all users of 'this' to switch to reference the more concrete
  /// type NewType and for 'this' to be deleted.
  ///
  void refineAbstractTypeTo(const Type *NewType);

  void dump() const { Type::dump(); }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const DerivedType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->isDerivedType();
  }
};

/// Class to represent integer types. Note that this class is also used to
/// represent the built-in integer types: Int1Ty, Int8Ty, Int16Ty, Int32Ty and
/// Int64Ty. 
/// @brief Integer representation type
class IntegerType : public DerivedType {
protected:
  explicit IntegerType(unsigned NumBits) : DerivedType(IntegerTyID) {
    setSubclassData(NumBits);
  }
  friend class TypeMap<IntegerValType, IntegerType>;
public:
  /// This enum is just used to hold constants we need for IntegerType.
  enum {
    MIN_INT_BITS = 1,        ///< Minimum number of bits that can be specified
    MAX_INT_BITS = (1<<23)-1 ///< Maximum number of bits that can be specified
      ///< Note that bit width is stored in the Type classes SubclassData field
      ///< which has 23 bits. This yields a maximum bit width of 8,388,607 bits.
  };

  /// This static method is the primary way of constructing an IntegerType. 
  /// If an IntegerType with the same NumBits value was previously instantiated,
  /// that instance will be returned. Otherwise a new one will be created. Only
  /// one instance with a given NumBits value is ever created.
  /// @brief Get or create an IntegerType instance.
  static const IntegerType* get(unsigned NumBits);

  /// @brief Get the number of bits in this IntegerType
  unsigned getBitWidth() const { return getSubclassData(); }

  /// getBitMask - Return a bitmask with ones set for all of the bits
  /// that can be set by an unsigned version of this type.  This is 0xFF for
  /// i8, 0xFFFF for i16, etc.
  uint64_t getBitMask() const {
    return ~uint64_t(0UL) >> (64-getBitWidth());
  }

  /// getSignBit - Return a uint64_t with just the most significant bit set (the
  /// sign bit, if the value is treated as a signed number).
  uint64_t getSignBit() const {
    return 1ULL << (getBitWidth()-1);
  }
  
  /// For example, this is 0xFF for an 8 bit integer, 0xFFFF for i16, etc.
  /// @returns a bit mask with ones set for all the bits of this type.
  /// @brief Get a bit mask for this type.
  APInt getMask() const;

  /// This method determines if the width of this IntegerType is a power-of-2
  /// in terms of 8 bit bytes. 
  /// @returns true if this is a power-of-2 byte width.
  /// @brief Is this a power-of-2 byte-width IntegerType ?
  bool isPowerOf2ByteWidth() const;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const IntegerType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == IntegerTyID;
  }
};


/// FunctionType - Class to represent function types
///
class FunctionType : public DerivedType {
  friend class TypeMap<FunctionValType, FunctionType>;
  bool isVarArgs;

  FunctionType(const FunctionType &);                   // Do not implement
  const FunctionType &operator=(const FunctionType &);  // Do not implement
  FunctionType(const Type *Result, const std::vector<const Type*> &Params,
               bool IsVarArgs);

public:
  /// FunctionType::get - This static method is the primary way of constructing
  /// a FunctionType. 
  ///
  static FunctionType *get(
    const Type *Result, ///< The result type
    const std::vector<const Type*> &Params, ///< The types of the parameters
    bool isVarArg  ///< Whether this is a variable argument length function
  );
  
  /// isValidReturnType - Return true if the specified type is valid as a return
  /// type.
  static bool isValidReturnType(const Type *RetTy);

  inline bool isVarArg() const { return isVarArgs; }
  inline const Type *getReturnType() const { return ContainedTys[0]; }

  typedef Type::subtype_iterator param_iterator;
  param_iterator param_begin() const { return ContainedTys + 1; }
  param_iterator param_end() const { return &ContainedTys[NumContainedTys]; }

  // Parameter type accessors...
  const Type *getParamType(unsigned i) const { return ContainedTys[i+1]; }

  /// getNumParams - Return the number of fixed parameters this function type
  /// requires.  This does not consider varargs.
  ///
  unsigned getNumParams() const { return NumContainedTys - 1; }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const FunctionType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == FunctionTyID;
  }
};


/// CompositeType - Common super class of ArrayType, StructType, PointerType
/// and VectorType
class CompositeType : public DerivedType {
protected:
  inline explicit CompositeType(TypeID id) : DerivedType(id) { }
public:

  /// getTypeAtIndex - Given an index value into the type, return the type of
  /// the element.
  ///
  virtual const Type *getTypeAtIndex(const Value *V) const = 0;
  virtual const Type *getTypeAtIndex(unsigned Idx) const = 0;
  virtual bool indexValid(const Value *V) const = 0;
  virtual bool indexValid(unsigned Idx) const = 0;

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CompositeType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID ||
           T->getTypeID() == StructTyID ||
           T->getTypeID() == PointerTyID ||
           T->getTypeID() == VectorTyID;
  }
};


/// StructType - Class to represent struct types
///
class StructType : public CompositeType {
  friend class TypeMap<StructValType, StructType>;
  StructType(const StructType &);                   // Do not implement
  const StructType &operator=(const StructType &);  // Do not implement
  StructType(const std::vector<const Type*> &Types, bool isPacked);
public:
  /// StructType::get - This static method is the primary way to create a
  /// StructType.
  ///
  static StructType *get(const std::vector<const Type*> &Params, 
                         bool isPacked=false);

  /// StructType::get - This static method is a convenience method for
  /// creating structure types by specifying the elements as arguments.
  /// Note that this method always returns a non-packed struct.  To get
  /// an empty struct, pass NULL, NULL.
  static StructType *get(const Type *type, ...) END_WITH_NULL;

  // Iterator access to the elements
  typedef Type::subtype_iterator element_iterator;
  element_iterator element_begin() const { return ContainedTys; }
  element_iterator element_end() const { return &ContainedTys[NumContainedTys];}

  // Random access to the elements
  unsigned getNumElements() const { return NumContainedTys; }
  const Type *getElementType(unsigned N) const {
    assert(N < NumContainedTys && "Element number out of range!");
    return ContainedTys[N];
  }

  /// getTypeAtIndex - Given an index value into the type, return the type of
  /// the element.  For a structure type, this must be a constant value...
  ///
  virtual const Type *getTypeAtIndex(const Value *V) const;
  virtual const Type *getTypeAtIndex(unsigned Idx) const;
  virtual bool indexValid(const Value *V) const;
  virtual bool indexValid(unsigned Idx) const;

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StructType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == StructTyID;
  }

  bool isPacked() const { return (0 != getSubclassData()) ? true : false; }
};


/// SequentialType - This is the superclass of the array, pointer and vector
/// type classes.  All of these represent "arrays" in memory.  The array type
/// represents a specifically sized array, pointer types are unsized/unknown
/// size arrays, vector types represent specifically sized arrays that
/// allow for use of SIMD instructions.  SequentialType holds the common
/// features of all, which stem from the fact that all three lay their
/// components out in memory identically.
///
class SequentialType : public CompositeType {
  PATypeHandle ContainedType; ///< Storage for the single contained type
  SequentialType(const SequentialType &);                  // Do not implement!
  const SequentialType &operator=(const SequentialType &); // Do not implement!

  // avoiding warning: 'this' : used in base member initializer list
  SequentialType* this_() { return this; }
protected:
  SequentialType(TypeID TID, const Type *ElType) 
    : CompositeType(TID), ContainedType(ElType, this_()) {
    ContainedTys = &ContainedType; 
    NumContainedTys = 1;
  }

public:
  inline const Type *getElementType() const { return ContainedTys[0]; }

  virtual bool indexValid(const Value *V) const;
  virtual bool indexValid(unsigned) const {
    return true;
  }

  /// getTypeAtIndex - Given an index value into the type, return the type of
  /// the element.  For sequential types, there is only one subtype...
  ///
  virtual const Type *getTypeAtIndex(const Value *) const {
    return ContainedTys[0];
  }
  virtual const Type *getTypeAtIndex(unsigned) const {
    return ContainedTys[0];
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const SequentialType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID ||
           T->getTypeID() == PointerTyID ||
           T->getTypeID() == VectorTyID;
  }
};


/// ArrayType - Class to represent array types
///
class ArrayType : public SequentialType {
  friend class TypeMap<ArrayValType, ArrayType>;
  uint64_t NumElements;

  ArrayType(const ArrayType &);                   // Do not implement
  const ArrayType &operator=(const ArrayType &);  // Do not implement
  ArrayType(const Type *ElType, uint64_t NumEl);
public:
  /// ArrayType::get - This static method is the primary way to construct an
  /// ArrayType
  ///
  static ArrayType *get(const Type *ElementType, uint64_t NumElements);

  inline uint64_t getNumElements() const { return NumElements; }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ArrayType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == ArrayTyID;
  }
};

/// VectorType - Class to represent vector types
///
class VectorType : public SequentialType {
  friend class TypeMap<VectorValType, VectorType>;
  unsigned NumElements;

  VectorType(const VectorType &);                   // Do not implement
  const VectorType &operator=(const VectorType &);  // Do not implement
  VectorType(const Type *ElType, unsigned NumEl);
public:
  /// VectorType::get - This static method is the primary way to construct an
  /// VectorType
  ///
  static VectorType *get(const Type *ElementType, unsigned NumElements);

  /// VectorType::getInteger - This static method gets a VectorType with the
  /// same number of elements as the input type, and the element type is an
  /// integer type of the same width as the input element type.
  ///
  static VectorType *getInteger(const VectorType *VTy) {
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    const Type *EltTy = IntegerType::get(EltBits);
    return VectorType::get(EltTy, VTy->getNumElements());
  }

  /// VectorType::getExtendedElementVectorType - This static method is like
  /// getInteger except that the element types are twice as wide as the
  /// elements in the input type.
  ///
  static VectorType *getExtendedElementVectorType(const VectorType *VTy) {
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    const Type *EltTy = IntegerType::get(EltBits * 2);
    return VectorType::get(EltTy, VTy->getNumElements());
  }

  /// VectorType::getTruncatedElementVectorType - This static method is like
  /// getInteger except that the element types are half as wide as the
  /// elements in the input type.
  ///
  static VectorType *getTruncatedElementVectorType(const VectorType *VTy) {
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    assert((EltBits & 1) == 0 &&
           "Cannot truncate vector element with odd bit-width");
    const Type *EltTy = IntegerType::get(EltBits / 2);
    return VectorType::get(EltTy, VTy->getNumElements());
  }

  /// @brief Return the number of elements in the Vector type.
  inline unsigned getNumElements() const { return NumElements; }

  /// @brief Return the number of bits in the Vector type.
  inline unsigned getBitWidth() const { 
    return NumElements *getElementType()->getPrimitiveSizeInBits();
  }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VectorType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == VectorTyID;
  }
};


/// PointerType - Class to represent pointers
///
class PointerType : public SequentialType {
  friend class TypeMap<PointerValType, PointerType>;
  unsigned AddressSpace;
  
  PointerType(const PointerType &);                   // Do not implement
  const PointerType &operator=(const PointerType &);  // Do not implement
  explicit PointerType(const Type *ElType, unsigned AddrSpace);
public:
  /// PointerType::get - This constructs a pointer to an object of the specified 
  /// type in a numbered address space.
  static PointerType *get(const Type *ElementType, unsigned AddressSpace);
  
  /// PointerType::getUnqual - This constructs a pointer to an object of the  
  /// specified type in the generic address space (address space zero).
  static PointerType *getUnqual(const Type *ElementType) { 
    return PointerType::get(ElementType, 0);
  }
  
  /// @brief Return the address space of the Pointer type.
  inline unsigned getAddressSpace() const { return AddressSpace; }

  // Implement the AbstractTypeUser interface.
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
  virtual void typeBecameConcrete(const DerivedType *AbsTy);

  // Implement support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PointerType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == PointerTyID;
  }
};


/// OpaqueType - Class to represent abstract types
///
class OpaqueType : public DerivedType {
  OpaqueType(const OpaqueType &);                   // DO NOT IMPLEMENT
  const OpaqueType &operator=(const OpaqueType &);  // DO NOT IMPLEMENT
  OpaqueType();
public:
  /// OpaqueType::get - Static factory method for the OpaqueType class...
  ///
  static OpaqueType *get() {
    return new OpaqueType();           // All opaque types are distinct
  }

  // Implement support for type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const OpaqueType *) { return true; }
  static inline bool classof(const Type *T) {
    return T->getTypeID() == OpaqueTyID;
  }
};

} // End llvm namespace

#endif
