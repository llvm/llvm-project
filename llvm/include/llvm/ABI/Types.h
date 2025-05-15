#ifndef LLVM_ABI_TYPES_H
#define LLVM_ABI_TYPES_H

#include <cstdint>
#include <memory>
#include <string>

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
  virtual ~Type() = default;

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

  static bool classof(const Type *) { return true; }
};
class VoidType : public Type {
public:
  VoidType() : Type(TypeKind::Void, 0, 0) {}

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Void; }
};

class IntegerType : public Type {
private:
  bool IsSigned;
  bool IsAltRepresentation;
  std::string TypeName;

public:
  IntegerType(uint64_t BitWidth, uint64_t Align, bool Signed,
              bool AltRep = false, const std::string &Name = "")
      : Type(TypeKind::Integer, BitWidth, Align), IsSigned(Signed),
        IsAltRepresentation(AltRep), TypeName(Name) {}

  bool isSigned() const { return IsSigned; }
  bool isAltRepresentation() const { return IsAltRepresentation; }
  const std::string &getTypeName() const { return TypeName; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Integer;
  }
};
class FloatType : public Type {
private:
  std::string TypeName;

public:
  FloatType(uint64_t BitWidth, uint64_t Align, const std::string &Name)
      : Type(TypeKind::Float, BitWidth, Align), TypeName(Name) {}

  const std::string &getTypeName() const { return TypeName; }

  static bool classof(const Type *T) { return T->getKind() == TypeKind::Float; }
};
class PointerType : public Type {
private:
  std::unique_ptr<Type> PointeeType;
  bool IsConst;
  bool IsVolatile;

public:
  PointerType(std::unique_ptr<Type> Pointee, uint64_t Size, uint64_t Align,
              bool Const = false, bool Volatile = false)
      : Type(TypeKind::Pointer, Size, Align), PointeeType(std::move(Pointee)),
        IsConst(Const), IsVolatile(Volatile) {}

  const Type *getPointeeType() const { return PointeeType.get(); }
  bool isConst() const { return IsConst; }
  bool isVolatile() const { return IsVolatile; }

  static bool classof(const Type *T) {
    return T->getKind() == TypeKind::Pointer;
  }
};

} // namespace abi
} // namespace llvm

#endif
