//===- llvm/SandboxIR/Type.h - Classes for handling data types --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a thin wrapper over llvm::Type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_TYPE_H
#define LLVM_SANDBOXIR_TYPE_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::sandboxir {

class Context;
// Forward declare friend classes for MSVC.
class ArrayType;
class CallBase;
class CmpInst;
class ConstantDataSequential;
class FixedVectorType;
class FPMathOperator;
class FunctionType;
class IntegerType;
class Module;
class PointerType;
class ScalableVectorType;
class StructType;
class TargetExtType;
class VectorType;
#define DEF_INSTR(ID, OPCODE, CLASS) class CLASS;
#define DEF_CONST(ID, CLASS) class CLASS;
#include "llvm/SandboxIR/Values.def"

/// Just like llvm::Type these are immutable, unique, never get freed and
/// can only be created via static factory methods.
class Type {
protected:
  llvm::Type *LLVMTy;
  friend class ArrayType;          // For LLVMTy.
  friend class StructType;         // For LLVMTy.
  friend class VectorType;         // For LLVMTy.
  friend class FixedVectorType;    // For LLVMTy.
  friend class ScalableVectorType; // For LLVMTy.
  friend class PointerType;        // For LLVMTy.
  friend class FunctionType;       // For LLVMTy.
  friend class IntegerType;        // For LLVMTy.
  friend class Function;           // For LLVMTy.
  friend class CallBase;           // For LLVMTy.
  friend class ConstantInt;        // For LLVMTy.
  friend class ConstantArray;      // For LLVMTy.
  friend class ConstantStruct;     // For LLVMTy.
  friend class ConstantVector;     // For LLVMTy.
  friend class CmpInst;            // For LLVMTy. TODO: Cleanup after
                                   // sandboxir::VectorType is more complete.
  friend class Utils;              // for LLVMTy
  friend class TargetExtType;      // For LLVMTy.
  friend class Module;             // For LLVMTy.
  friend class FPMathOperator;     // For LLVMTy.
  friend class ConstantDataSequential; // For LLVMTy.

  // Friend all instruction classes because `create()` functions use LLVMTy.
#define DEF_INSTR(ID, OPCODE, CLASS) friend class CLASS;
#define DEF_CONST(ID, CLASS) friend class CLASS;
#include "llvm/SandboxIR/Values.def"
  Context &Ctx;

  Type(llvm::Type *LLVMTy, Context &Ctx) : LLVMTy(LLVMTy), Ctx(Ctx) {}
  friend class Context; // For constructor and ~Type().
  ~Type() = default;

public:
  /// Print the current type.
  /// Omit the type details if \p NoDetails == true.
  /// E.g., let %st = type { i32, i16 }
  /// When \p NoDetails is true, we only print %st.
  /// Put differently, \p NoDetails prints the type as if
  /// inlined with the operands when printing an instruction.
  void print(raw_ostream &OS, bool IsForDebug = false,
             bool NoDetails = false) const {
    LLVMTy->print(OS, IsForDebug, NoDetails);
  }

  Context &getContext() const { return Ctx; }

  /// Return true if this is 'void'.
  bool isVoidTy() const { return LLVMTy->isVoidTy(); }

  /// Return true if this is 'half', a 16-bit IEEE fp type.
  bool isHalfTy() const { return LLVMTy->isHalfTy(); }

  /// Return true if this is 'bfloat', a 16-bit bfloat type.
  bool isBFloatTy() const { return LLVMTy->isBFloatTy(); }

  /// Return true if this is a 16-bit float type.
  bool is16bitFPTy() const { return LLVMTy->is16bitFPTy(); }

  /// Return true if this is 'float', a 32-bit IEEE fp type.
  bool isFloatTy() const { return LLVMTy->isFloatTy(); }

  /// Return true if this is 'double', a 64-bit IEEE fp type.
  bool isDoubleTy() const { return LLVMTy->isDoubleTy(); }

  /// Return true if this is x86 long double.
  bool isX86_FP80Ty() const { return LLVMTy->isX86_FP80Ty(); }

  /// Return true if this is 'fp128'.
  bool isFP128Ty() const { return LLVMTy->isFP128Ty(); }

  /// Return true if this is powerpc long double.
  bool isPPC_FP128Ty() const { return LLVMTy->isPPC_FP128Ty(); }

  /// Return true if this is a well-behaved IEEE-like type, which has a IEEE
  /// compatible layout, and does not have non-IEEE values, such as x86_fp80's
  /// unnormal values.
  bool isIEEELikeFPTy() const { return LLVMTy->isIEEELikeFPTy(); }

  /// Return true if this is one of the floating-point types
  bool isFloatingPointTy() const { return LLVMTy->isFloatingPointTy(); }

  /// Returns true if this is a floating-point type that is an unevaluated sum
  /// of multiple floating-point units.
  /// An example of such a type is ppc_fp128, also known as double-double, which
  /// consists of two IEEE 754 doubles.
  bool isMultiUnitFPType() const { return LLVMTy->isMultiUnitFPType(); }

  const fltSemantics &getFltSemantics() const {
    return LLVMTy->getFltSemantics();
  }

  /// Return true if this is X86 AMX.
  bool isX86_AMXTy() const { return LLVMTy->isX86_AMXTy(); }

  /// Return true if this is a target extension type.
  bool isTargetExtTy() const { return LLVMTy->isTargetExtTy(); }

  /// Return true if this is a target extension type with a scalable layout.
  bool isScalableTargetExtTy() const { return LLVMTy->isScalableTargetExtTy(); }

  /// Return true if this is a type whose size is a known multiple of vscale.
  bool isScalableTy() const { return LLVMTy->isScalableTy(); }

  /// Return true if this is a FP type or a vector of FP.
  bool isFPOrFPVectorTy() const { return LLVMTy->isFPOrFPVectorTy(); }

  /// Return true if this is 'label'.
  bool isLabelTy() const { return LLVMTy->isLabelTy(); }

  /// Return true if this is 'metadata'.
  bool isMetadataTy() const { return LLVMTy->isMetadataTy(); }

  /// Return true if this is 'token'.
  bool isTokenTy() const { return LLVMTy->isTokenTy(); }

  /// True if this is an instance of IntegerType.
  bool isIntegerTy() const { return LLVMTy->isIntegerTy(); }

  /// Return true if this is an IntegerType of the given width.
  bool isIntegerTy(unsigned Bitwidth) const {
    return LLVMTy->isIntegerTy(Bitwidth);
  }

  /// Return true if this is an integer type or a vector of integer types.
  bool isIntOrIntVectorTy() const { return LLVMTy->isIntOrIntVectorTy(); }

  /// Return true if this is an integer type or a vector of integer types of
  /// the given width.
  bool isIntOrIntVectorTy(unsigned BitWidth) const {
    return LLVMTy->isIntOrIntVectorTy(BitWidth);
  }

  /// Return true if this is an integer type or a pointer type.
  bool isIntOrPtrTy() const { return LLVMTy->isIntOrPtrTy(); }

  /// True if this is an instance of FunctionType.
  bool isFunctionTy() const { return LLVMTy->isFunctionTy(); }

  /// True if this is an instance of StructType.
  bool isStructTy() const { return LLVMTy->isStructTy(); }

  /// True if this is an instance of ArrayType.
  bool isArrayTy() const { return LLVMTy->isArrayTy(); }

  /// True if this is an instance of PointerType.
  bool isPointerTy() const { return LLVMTy->isPointerTy(); }

  /// Return true if this is a pointer type or a vector of pointer types.
  bool isPtrOrPtrVectorTy() const { return LLVMTy->isPtrOrPtrVectorTy(); }

  /// True if this is an instance of VectorType.
  inline bool isVectorTy() const { return LLVMTy->isVectorTy(); }

  /// Return true if this type could be converted with a lossless BitCast to
  /// type 'Ty'. For example, i8* to i32*. BitCasts are valid for types of the
  /// same size only where no re-interpretation of the bits is done.
  /// Determine if this type could be losslessly bitcast to Ty
  bool canLosslesslyBitCastTo(Type *Ty) const {
    return LLVMTy->canLosslesslyBitCastTo(Ty->LLVMTy);
  }

  /// Return true if this type is empty, that is, it has no elements or all of
  /// its elements are empty.
  bool isEmptyTy() const { return LLVMTy->isEmptyTy(); }

  /// Return true if the type is "first class", meaning it is a valid type for a
  /// Value.
  bool isFirstClassType() const { return LLVMTy->isFirstClassType(); }

  /// Return true if the type is a valid type for a register in codegen. This
  /// includes all first-class types except struct and array types.
  bool isSingleValueType() const { return LLVMTy->isSingleValueType(); }

  /// Return true if the type is an aggregate type. This means it is valid as
  /// the first operand of an insertvalue or extractvalue instruction. This
  /// includes struct and array types, but does not include vector types.
  bool isAggregateType() const { return LLVMTy->isAggregateType(); }

  /// Return true if it makes sense to take the size of this type. To get the
  /// actual size for a particular target, it is reasonable to use the
  /// DataLayout subsystem to do this.
  bool isSized(SmallPtrSetImpl<Type *> *Visited = nullptr) const {
    SmallPtrSet<llvm::Type *, 8> LLVMVisited;
    LLVMVisited.reserve(Visited->size());
    for (Type *Ty : *Visited)
      LLVMVisited.insert(Ty->LLVMTy);
    return LLVMTy->isSized(&LLVMVisited);
  }

  /// Return the basic size of this type if it is a primitive type. These are
  /// fixed by LLVM and are not target-dependent.
  /// This will return zero if the type does not have a size or is not a
  /// primitive type.
  ///
  /// If this is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// Note that this may not reflect the size of memory allocated for an
  /// instance of the type or the number of bytes that are written when an
  /// instance of the type is stored to memory. The DataLayout class provides
  /// additional query functions to provide this information.
  ///
  TypeSize getPrimitiveSizeInBits() const {
    return LLVMTy->getPrimitiveSizeInBits();
  }

  /// If this is a vector type, return the getPrimitiveSizeInBits value for the
  /// element type. Otherwise return the getPrimitiveSizeInBits value for this
  /// type.
  unsigned getScalarSizeInBits() const { return LLVMTy->getScalarSizeInBits(); }

  /// Return the width of the mantissa of this type. This is only valid on
  /// floating-point types. If the FP type does not have a stable mantissa (e.g.
  /// ppc long double), this method returns -1.
  int getFPMantissaWidth() const { return LLVMTy->getFPMantissaWidth(); }

  /// If this is a vector type, return the element type, otherwise return
  /// 'this'.
  LLVM_ABI Type *getScalarType() const;

  // TODO: ADD MISSING

  LLVM_ABI static Type *getInt64Ty(Context &Ctx);
  LLVM_ABI static Type *getInt32Ty(Context &Ctx);
  LLVM_ABI static Type *getInt16Ty(Context &Ctx);
  LLVM_ABI static Type *getInt8Ty(Context &Ctx);
  LLVM_ABI static Type *getInt1Ty(Context &Ctx);
  LLVM_ABI static Type *getDoubleTy(Context &Ctx);
  LLVM_ABI static Type *getFloatTy(Context &Ctx);
  LLVM_ABI static Type *getHalfTy(Context &Ctx);
  // TODO: missing get*

  /// Get the address space of this pointer or pointer vector type.
  inline unsigned getPointerAddressSpace() const {
    return LLVMTy->getPointerAddressSpace();
  }

#ifndef NDEBUG
  void dumpOS(raw_ostream &OS);
  LLVM_DUMP_METHOD void dump();
#endif // NDEBUG
};

class PointerType : public Type {
public:
  // TODO: add missing functions

  LLVM_ABI static PointerType *get(Context &Ctx, unsigned AddressSpace);

  static bool classof(const Type *From) {
    return isa<llvm::PointerType>(From->LLVMTy);
  }
};

class ArrayType : public Type {
public:
  LLVM_ABI static ArrayType *get(Type *ElementType, uint64_t NumElements);
  // TODO: add missing functions
  static bool classof(const Type *From) {
    return isa<llvm::ArrayType>(From->LLVMTy);
  }
};

class StructType : public Type {
public:
  /// This static method is the primary way to create a literal StructType.
  LLVM_ABI static StructType *get(Context &Ctx, ArrayRef<Type *> Elements,
                                  bool IsPacked = false);

  bool isPacked() const { return cast<llvm::StructType>(LLVMTy)->isPacked(); }

  // TODO: add missing functions
  static bool classof(const Type *From) {
    return isa<llvm::StructType>(From->LLVMTy);
  }
};

class VectorType : public Type {
public:
  LLVM_ABI static VectorType *get(Type *ElementType, ElementCount EC);
  static VectorType *get(Type *ElementType, unsigned NumElements,
                         bool Scalable) {
    return VectorType::get(ElementType,
                           ElementCount::get(NumElements, Scalable));
  }
  LLVM_ABI Type *getElementType() const;

  static VectorType *get(Type *ElementType, const VectorType *Other) {
    return VectorType::get(ElementType, Other->getElementCount());
  }

  inline ElementCount getElementCount() const {
    return cast<llvm::VectorType>(LLVMTy)->getElementCount();
  }
  LLVM_ABI static VectorType *getInteger(VectorType *VTy);
  LLVM_ABI static VectorType *getExtendedElementVectorType(VectorType *VTy);
  LLVM_ABI static VectorType *getTruncatedElementVectorType(VectorType *VTy);
  LLVM_ABI static VectorType *getSubdividedVectorType(VectorType *VTy,
                                                      int NumSubdivs);
  LLVM_ABI static VectorType *getHalfElementsVectorType(VectorType *VTy);
  LLVM_ABI static VectorType *getDoubleElementsVectorType(VectorType *VTy);
  LLVM_ABI static bool isValidElementType(Type *ElemTy);

  static bool classof(const Type *From) {
    return isa<llvm::VectorType>(From->LLVMTy);
  }
};

class FixedVectorType : public VectorType {
public:
  LLVM_ABI static FixedVectorType *get(Type *ElementType, unsigned NumElts);

  static FixedVectorType *get(Type *ElementType, const FixedVectorType *FVTy) {
    return get(ElementType, FVTy->getNumElements());
  }

  static FixedVectorType *getInteger(FixedVectorType *VTy) {
    return cast<FixedVectorType>(VectorType::getInteger(VTy));
  }

  static FixedVectorType *getExtendedElementVectorType(FixedVectorType *VTy) {
    return cast<FixedVectorType>(VectorType::getExtendedElementVectorType(VTy));
  }

  static FixedVectorType *getTruncatedElementVectorType(FixedVectorType *VTy) {
    return cast<FixedVectorType>(
        VectorType::getTruncatedElementVectorType(VTy));
  }

  static FixedVectorType *getSubdividedVectorType(FixedVectorType *VTy,
                                                  int NumSubdivs) {
    return cast<FixedVectorType>(
        VectorType::getSubdividedVectorType(VTy, NumSubdivs));
  }

  static FixedVectorType *getHalfElementsVectorType(FixedVectorType *VTy) {
    return cast<FixedVectorType>(VectorType::getHalfElementsVectorType(VTy));
  }

  static FixedVectorType *getDoubleElementsVectorType(FixedVectorType *VTy) {
    return cast<FixedVectorType>(VectorType::getDoubleElementsVectorType(VTy));
  }

  static bool classof(const Type *T) {
    return isa<llvm::FixedVectorType>(T->LLVMTy);
  }

  unsigned getNumElements() const {
    return cast<llvm::FixedVectorType>(LLVMTy)->getNumElements();
  }
};

class ScalableVectorType : public VectorType {
public:
  LLVM_ABI static ScalableVectorType *get(Type *ElementType,
                                          unsigned MinNumElts);

  static ScalableVectorType *get(Type *ElementType,
                                 const ScalableVectorType *SVTy) {
    return get(ElementType, SVTy->getMinNumElements());
  }

  static ScalableVectorType *getInteger(ScalableVectorType *VTy) {
    return cast<ScalableVectorType>(VectorType::getInteger(VTy));
  }

  static ScalableVectorType *
  getExtendedElementVectorType(ScalableVectorType *VTy) {
    return cast<ScalableVectorType>(
        VectorType::getExtendedElementVectorType(VTy));
  }

  static ScalableVectorType *
  getTruncatedElementVectorType(ScalableVectorType *VTy) {
    return cast<ScalableVectorType>(
        VectorType::getTruncatedElementVectorType(VTy));
  }

  static ScalableVectorType *getSubdividedVectorType(ScalableVectorType *VTy,
                                                     int NumSubdivs) {
    return cast<ScalableVectorType>(
        VectorType::getSubdividedVectorType(VTy, NumSubdivs));
  }

  static ScalableVectorType *
  getHalfElementsVectorType(ScalableVectorType *VTy) {
    return cast<ScalableVectorType>(VectorType::getHalfElementsVectorType(VTy));
  }

  static ScalableVectorType *
  getDoubleElementsVectorType(ScalableVectorType *VTy) {
    return cast<ScalableVectorType>(
        VectorType::getDoubleElementsVectorType(VTy));
  }

  unsigned getMinNumElements() const {
    return cast<llvm::ScalableVectorType>(LLVMTy)->getMinNumElements();
  }

  static bool classof(const Type *T) {
    return isa<llvm::ScalableVectorType>(T->LLVMTy);
  }
};

class FunctionType : public Type {
public:
  // TODO: add missing functions
  static bool classof(const Type *From) {
    return isa<llvm::FunctionType>(From->LLVMTy);
  }
};

/// Class to represent integer types. Note that this class is also used to
/// represent the built-in integer types: Int1Ty, Int8Ty, Int16Ty, Int32Ty and
/// Int64Ty.
/// Integer representation type
class IntegerType : public Type {
public:
  LLVM_ABI static IntegerType *get(Context &C, unsigned NumBits);
  // TODO: add missing functions
  static bool classof(const Type *From) {
    return isa<llvm::IntegerType>(From->LLVMTy);
  }
  operator llvm::IntegerType &() const {
    return *cast<llvm::IntegerType>(LLVMTy);
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_TYPE_H
