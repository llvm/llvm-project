#ifndef LLVM_ABI_TYPES_H
#define LLVM_ABI_TYPES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include <cstdint>

namespace llvm {
namespace abi {

enum class TypeKind {
  Void,
  Integer,
  Float,
  Pointer,
  Array,
  Vector,
  Struct,
  Union,
  Function
};

class Type {
protected:
  TypeKind Kind;
  uint64_t SizeInBits;
  uint64_t AlignInBits;
  bool IsExplicitlyAligned;

  Type(TypeKind K, uint64_t Size, uint64_t Align, bool ExplicitAlign = false)
      : Kind(K), SizeInBits(Size), AlignInBits(Align),
        IsExplicitlyAligned(ExplicitAlign) {}

public:
  TypeKind getKind() const { return Kind; }
  uint64_t getSizeInBits() const { return SizeInBits; }
  uint64_t getAlignInBits() const { return AlignInBits; }
  bool hasExplicitAlignment() const { return IsExplicitlyAligned; }

  void setExplicitAlignment(uint64_t Align) {
    AlignInBits = Align;
    IsExplicitlyAligned = true;
  }

  bool isVoid() const { return Kind == TypeKind::Void; }
  bool isInteger() const { return Kind == TypeKind::Integer; }
  bool isFloat() const { return Kind == TypeKind::Float; }
  bool isPointer() const { return Kind == TypeKind::Pointer; }
  bool isArray() const { return Kind == TypeKind::Array; }
  bool isVector() const { return Kind == TypeKind::Vector; }
  bool isStruct() const { return Kind == TypeKind::Struct; }
  bool isUnion() const { return Kind == TypeKind::Union; }
  bool isFunction() const { return Kind == TypeKind::Function; }
};

class VoidType : public Type {
public:
  VoidType() : Type(TypeKind::Void, 0, 0) {}

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Void; }
};

class IntegerType : public Type {
private:
  bool IsSigned;

public:
  IntegerType(uint64_t BitWidth, uint64_t Align, bool Signed)
      : Type(TypeKind::Integer, BitWidth, Align), IsSigned(Signed) {}

  bool isSigned() const { return IsSigned; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Integer;
  }
};

class FloatType : public Type {
public:
  FloatType(uint64_t BitWidth, uint64_t Align)
      : Type(TypeKind::Float, BitWidth, Align) {}

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Float; }
};

class PointerType : public Type {
public:
  PointerType(uint64_t Size, uint64_t Align)
      : Type(TypeKind::Pointer, Size, Align) {}

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Pointer;
  }
};

class ArrayType : public Type {
private:
  const Type *ElementType;
  uint64_t NumElements;

public:
  ArrayType(const Type *ElemType, uint64_t NumElems)
      : Type(TypeKind::Array, ElemType->getSizeInBits() * NumElems,
             ElemType->getAlignInBits()),
        ElementType(ElemType), NumElements(NumElems) {}

  const Type *getElementType() const { return ElementType; }
  uint64_t getNumElements() const { return NumElements; }

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Array; }
};

class VectorType : public Type {
private:
  const Type *ElementType;
  uint64_t NumElements;

public:
  VectorType(const Type *ElemType, uint64_t NumElems, uint64_t Align)
      : Type(TypeKind::Vector, ElemType->getSizeInBits() * NumElems, Align),
        ElementType(ElemType), NumElements(NumElems) {}

  const Type *getElementType() const { return ElementType; }
  uint64_t getNumElements() const { return NumElements; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Vector;
  }
};

struct FieldInfo {
  const Type *FieldType;
  uint64_t OffsetInBits;
  bool IsBitField;
  uint64_t BitFieldWidth;

  FieldInfo(const Type *Type, uint64_t Offset = 0, bool BitField = false,
            uint64_t BFWidth = 0)
      : FieldType(Type), OffsetInBits(Offset), IsBitField(BitField),
        BitFieldWidth(BFWidth) {}
};

enum class StructPacking { Default, Packed, ExplicitPacking };

class StructType : public Type {
private:
  const FieldInfo *Fields;
  uint32_t NumFields;
  StructPacking Packing;

public:
  StructType(const FieldInfo *StructFields, uint32_t FieldCount, uint64_t Size,
             uint64_t Align, StructPacking Pack = StructPacking::Default)
      : Type(TypeKind::Struct, Size, Align), Fields(StructFields),
        NumFields(FieldCount), Packing(Pack) {}

  const FieldInfo *getFields() const { return Fields; }
  uint32_t getNumFields() const { return NumFields; }
  StructPacking getPacking() const { return Packing; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Struct;
  }
};

class UnionType : public Type {
private:
  const FieldInfo *Fields;
  uint32_t NumFields;
  StructPacking Packing;

public:
  UnionType(const FieldInfo *UnionFields, uint32_t FieldCount, uint64_t Size,
            uint64_t Align, StructPacking Pack = StructPacking::Default)
      : Type(TypeKind::Union, Size, Align), Fields(UnionFields),
        NumFields(FieldCount), Packing(Pack) {}

  const FieldInfo *getFields() const { return Fields; }
  uint32_t getNumFields() const { return NumFields; }
  StructPacking getPacking() const { return Packing; }

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Union; }
};

enum class CallConv {
  C,
  // TODO: extend for more CallConvs
};

class FunctionType : public Type {
private:
  const Type *ReturnType;
  const Type *const *ParameterTypes;
  uint32_t NumParams;
  bool IsVarArg;
  CallConv CC;

public:
  FunctionType(const Type *RetType, const Type *const *ParamTypes,
               uint32_t ParamCount, bool VarArgs, CallConv CallConv)
      : Type(TypeKind::Function, 0, 0), ReturnType(RetType),
        ParameterTypes(ParamTypes), NumParams(ParamCount), IsVarArg(VarArgs),
        CC(CallConv) {}

  const Type *getReturnType() const { return ReturnType; }
  const Type *const *getParameterTypes() const { return ParameterTypes; }
  uint32_t getNumParameters() const { return NumParams; }
  const Type *getParameterType(uint32_t Index) const {
    assert(Index < NumParams && "Parameter index out of bounds");
    return ParameterTypes[Index];
  }
  bool isVarArg() const { return IsVarArg; }
  CallConv getCallingConv() const { return CC; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Function;
  }
};

// API for creating ABI Types
class TypeBuilder {
private:
  BumpPtrAllocator &Allocator;

public:
  explicit TypeBuilder(BumpPtrAllocator &Alloc) : Allocator(Alloc) {}

  const VoidType *getVoidType() {
    return new (Allocator.Allocate<VoidType>()) VoidType();
  }

  const IntegerType *getIntegerType(uint64_t BitWidth, uint64_t Align,
                                    bool Signed) {
    return new (Allocator.Allocate<IntegerType>())
        IntegerType(BitWidth, Align, Signed);
  }

  const FloatType *getFloatType(uint64_t BitWidth, uint64_t Align) {
    return new (Allocator.Allocate<FloatType>()) FloatType(BitWidth, Align);
  }

  const PointerType *getPointerType(uint64_t Size, uint64_t Align) {
    return new (Allocator.Allocate<PointerType>()) PointerType(Size, Align);
  }

  const ArrayType *getArrayType(const Type *ElementType, uint64_t NumElements) {
    return new (Allocator.Allocate<ArrayType>())
        ArrayType(ElementType, NumElements);
  }

  const VectorType *getVectorType(const Type *ElementType, uint64_t NumElements,
                                  uint64_t Align) {
    return new (Allocator.Allocate<VectorType>())
        VectorType(ElementType, NumElements, Align);
  }

  const StructType *getStructType(ArrayRef<FieldInfo> Fields, uint64_t Size,
                                  uint64_t Align,
                                  StructPacking Pack = StructPacking::Default) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());

    for (size_t I = 0; I < Fields.size(); ++I) {
      new (&FieldArray[I]) FieldInfo(Fields[I]);
    }

    return new (Allocator.Allocate<StructType>()) StructType(
        FieldArray, static_cast<uint32_t>(Fields.size()), Size, Align, Pack);
  }

  const UnionType *getUnionType(ArrayRef<FieldInfo> Fields, uint64_t Size,
                                uint64_t Align,
                                StructPacking Pack = StructPacking::Default) {
    FieldInfo *FieldArray = Allocator.Allocate<FieldInfo>(Fields.size());

    for (size_t I = 0; I < Fields.size(); ++I) {
      new (&FieldArray[I]) FieldInfo(Fields[I]);
    }

    return new (Allocator.Allocate<UnionType>()) UnionType(
        FieldArray, static_cast<uint32_t>(Fields.size()), Size, Align, Pack);
  }

  const FunctionType *getFunctionType(const Type *ReturnType,
                                      ArrayRef<const Type *> ParamTypes,
                                      bool IsVarArg,
                                      CallConv CC = CallConv::C) {
    const Type **ParamArray =
        Allocator.Allocate<const Type *>(ParamTypes.size());

    for (size_t I = 0; I < ParamTypes.size(); ++I) {
      ParamArray[I] = ParamTypes[I];
    }

    return new (Allocator.Allocate<FunctionType>())
        FunctionType(ReturnType, ParamArray,
                     static_cast<uint32_t>(ParamTypes.size()), IsVarArg, CC);
  }
};

} // namespace abi
} // namespace llvm

#endif
