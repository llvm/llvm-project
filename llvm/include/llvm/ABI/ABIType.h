#ifndef LLVM_ABI_ABITYPE_H
#define LLVM_ABI_ABITYPE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace abi {

enum class ABITypeKind {
  Void,
  Bool,
  Char,
  SChar,
  UChar,
  Short,
  UShort,
  Int,
  UInt,
  Long,
  ULong,
  LongLong,
  ULongLong,
  Float,
  Double,
  LongDouble,
  Pointer,
  Array,
  Record,
  Union
};

/// Base class for all types in the ABI type system
class ABIType {
  const ABITypeKind Kind;
  unsigned Alignment;

protected:
  ABIType(ABITypeKind K, unsigned Align) : Kind(K), Alignment(Align) {}

public:
  ABITypeKind getKind() const { return Kind; }
  unsigned getAlignment() const { return Alignment; }

  // Support for LLVM RTTI
  static bool classof(const ABIType *) { return true; }

  virtual ~ABIType() = default;
};

class VoidType : public ABIType {
public:
  VoidType() : ABIType(ABITypeKind::Void, 0) {}

  static bool classof(const ABIType *T) {
    return T->getKind() == ABITypeKind::Void;
  }
};

class ScalarType : public ABIType {
  uint64_t Size;

public:
  ScalarType(ABITypeKind Kind, unsigned Align, uint64_t Size)
      : ABIType(Kind, Align), Size(Size) {}

  uint64_t getSize() const { return Size; }

  static bool classof(const ABIType *T) {
    return T->getKind() >= ABITypeKind::Bool &&
           T->getKind() <= ABITypeKind::LongDouble;
  }
};

class PointerType : public ABIType {
  const ABIType *PointeeType;
  uint64_t Size;

public:
  PointerType(const ABIType *Pointee, unsigned Align, uint64_t Size)
      : ABIType(ABITypeKind::Pointer, Align), PointeeType(Pointee), Size(Size) {
  }

  const ABIType *getPointeeType() const { return PointeeType; }
  uint64_t getSize() const { return Size; }

  static bool classof(const ABIType *T) {
    return T->getKind() == ABITypeKind::Pointer;
  }
};

class ArrayType : public ABIType {
  const ABIType *ElementType;
  uint64_t NumElements;
  uint64_t Size;

public:
  ArrayType(const ABIType *ElemType, uint64_t Count, unsigned Align,
            uint64_t Size)
      : ABIType(ABITypeKind::Array, Align), ElementType(ElemType),
        NumElements(Count), Size(Size) {}

  const ABIType *getElementType() const { return ElementType; }
  uint64_t getNumElements() const { return NumElements; }
  uint64_t getSize() const { return Size; }

  static bool classof(const ABIType *T) {
    return T->getKind() == ABITypeKind::Array;
  }
};

struct RecordField {
  StringRef Name;
  const ABIType *Type;
  uint64_t Offset;

  RecordField(StringRef Name, const ABIType *Type, uint64_t Offset)
      : Name(Name), Type(Type), Offset(Offset) {}
};

class RecordType : public ABIType {
  std::vector<RecordField> Fields;
  uint64_t Size;
  bool Packed;

public:
  RecordType(ArrayRef<RecordField> Fields, unsigned Align, uint64_t Size,
             bool Packed)
      : ABIType(ABITypeKind::Record, Align),
        Fields(Fields.begin(), Fields.end()), Size(Size), Packed(Packed) {}

  ArrayRef<RecordField> getFields() const { return Fields; }
  uint64_t getSize() const { return Size; }
  bool isPacked() const { return Packed; }

  static bool classof(const ABIType *T) {
    return T->getKind() == ABITypeKind::Record;
  }
};

class UnionType : public ABIType {
  std::vector<RecordField> Fields;
  uint64_t Size;

public:
  UnionType(ArrayRef<RecordField> Fields, unsigned Align, uint64_t Size)
      : ABIType(ABITypeKind::Union, Align),
        Fields(Fields.begin(), Fields.end()), Size(Size) {}

  ArrayRef<RecordField> getFields() const { return Fields; }
  uint64_t getSize() const { return Size; }

  static bool classof(const ABIType *T) {
    return T->getKind() == ABITypeKind::Union;
  }
};

class ABITypeContext {
  BumpPtrAllocator Allocator;
  VoidType VoidTy;

  ScalarType BoolTy;
  ScalarType CharTy;
  ScalarType SCharTy;
  ScalarType UCharTy;
  ScalarType ShortTy;
  ScalarType UShortTy;
  ScalarType IntTy;
  ScalarType UIntTy;
  ScalarType LongTy;
  ScalarType ULongTy;
  ScalarType LongLongTy;
  ScalarType ULongLongTy;
  ScalarType FloatTy;
  ScalarType DoubleTy;
  ScalarType LongDoubleTy;

public:
  ABITypeContext()
      : VoidTy(),
        // Initialize scalar types with target-independent defaults
        // Real implementations would initialize these based on target info
        BoolTy(ABITypeKind::Bool, 1, 1), CharTy(ABITypeKind::Char, 1, 1),
        SCharTy(ABITypeKind::SChar, 1, 1), UCharTy(ABITypeKind::UChar, 1, 1),
        ShortTy(ABITypeKind::Short, 2, 2), UShortTy(ABITypeKind::UShort, 2, 2),
        IntTy(ABITypeKind::Int, 4, 4), UIntTy(ABITypeKind::UInt, 4, 4),
        LongTy(ABITypeKind::Long, 8, 8), ULongTy(ABITypeKind::ULong, 8, 8),
        LongLongTy(ABITypeKind::LongLong, 8, 8),
        ULongLongTy(ABITypeKind::ULongLong, 8, 8),
        FloatTy(ABITypeKind::Float, 4, 4), DoubleTy(ABITypeKind::Double, 8, 8),
        LongDoubleTy(ABITypeKind::LongDouble, 16, 16) {}

  const VoidType *getVoidType() { return &VoidTy; }
  const ScalarType *getBoolType() { return &BoolTy; }
  const ScalarType *getCharType() { return &CharTy; }
  const ScalarType *getSCharType() { return &SCharTy; }
  const ScalarType *getUCharType() { return &UCharTy; }
  const ScalarType *getShortType() { return &ShortTy; }
  const ScalarType *getUShortType() { return &UShortTy; }
  const ScalarType *getIntType() { return &IntTy; }
  const ScalarType *getUIntType() { return &UIntTy; }
  const ScalarType *getLongType() { return &LongTy; }
  const ScalarType *getULongType() { return &ULongTy; }
  const ScalarType *getLongLongType() { return &LongLongTy; }
  const ScalarType *getULongLongType() { return &ULongLongTy; }
  const ScalarType *getFloatType() { return &FloatTy; }
  const ScalarType *getDoubleType() { return &DoubleTy; }
  const ScalarType *getLongDoubleType() { return &LongDoubleTy; }

  const PointerType *getPointerType(const ABIType *Pointee, unsigned Align,
                                    uint64_t Size) {
    return new (Allocator) PointerType(Pointee, Align, Size);
  }

  const ArrayType *getArrayType(const ABIType *ElementType,
                                uint64_t NumElements, unsigned Align,
                                uint64_t Size) {
    return new (Allocator) ArrayType(ElementType, NumElements, Align, Size);
  }
  const RecordType *getRecordType(ArrayRef<RecordField> Fields, unsigned Align,
                                  uint64_t Size, bool Packed = false) {
    // Take the allocator and make copy
    SmallVector<RecordField, 8> FieldsCopy;
    for (const auto &Field : Fields) {
      StringRef Name;
      if (!Field.Name.empty()) {
        char *NameBuf = Allocator.Allocate<char>(Field.Name.size() + 1);
        std::copy(Field.Name.begin(), Field.Name.end(), NameBuf);
        NameBuf[Field.Name.size()] = '\0';
        Name = StringRef(NameBuf, Field.Name.size());
      }
      FieldsCopy.emplace_back(Name, Field.Type, Field.Offset);
    }
    return new (Allocator) RecordType(FieldsCopy, Align, Size, Packed);
  }
  const UnionType *getUnionType(ArrayRef<RecordField> Fields, unsigned Align,
                                uint64_t Size) {
    // Make a copy with the allocator
    SmallVector<RecordField, 8> FieldsCopy;
    for (const auto &Field : Fields) {
      StringRef Name;
      if (!Field.Name.empty()) {
        char *NameBuf = Allocator.Allocate<char>(Field.Name.size() + 1);
        std::copy(Field.Name.begin(), Field.Name.end(), NameBuf);
        NameBuf[Field.Name.size()] = '\0';
        Name = StringRef(NameBuf, Field.Name.size());
      }
      FieldsCopy.emplace_back(Name, Field.Type, Field.Offset);
    }
    return new (Allocator) UnionType(FieldsCopy, Align, Size);
  }
};
} // namespace abi
} // namespace llvm

#endif // LLVM_ABI_ABITYPE_H
