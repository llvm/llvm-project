//===- ValueTypes.cpp - Tablegen extended ValueType implementation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The EVT type is used by tablegen as well as in LLVM. In order to handle
// extended types, the EVT type uses support functions that call into
// LLVM's type system code. These aren't accessible in tablegen, so this
// file provides simple replacements.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Casting.h"
#include <map>
using namespace llvm;

namespace llvm {

class Type {
protected:
  enum TypeKind {
    TK_ExtendedIntegerType,
    TK_ExtendedVectorType
  };
private:
  TypeKind Kind;
public:
  TypeKind getKind() const {
    return Kind;
  }
  Type(TypeKind K) : Kind(K) {}
  virtual unsigned getSizeInBits() const = 0;
  virtual ~Type();
};

// Provide out-of-line definition to prevent weak vtable.
Type::~Type() {}

}

namespace {
class ExtendedIntegerType : public Type {
  unsigned BitWidth;
public:
  explicit ExtendedIntegerType(unsigned bits)
    : Type(TK_ExtendedIntegerType), BitWidth(bits) {}
  static bool classof(const Type *T) {
    return T->getKind() == TK_ExtendedIntegerType;
  }
  virtual unsigned getSizeInBits() const {
    return getBitWidth();
  }
  unsigned getBitWidth() const {
    return BitWidth;
  }
};

class ExtendedVectorType : public Type {
  EVT ElementType;
  unsigned NumElements;
public:
  ExtendedVectorType(EVT elty, unsigned num)
    : Type(TK_ExtendedVectorType), ElementType(elty), NumElements(num) {}
  static bool classof(const Type *T) {
    return T->getKind() == TK_ExtendedVectorType;
  }
  virtual unsigned getSizeInBits() const {
    return getNumElements() * getElementType().getSizeInBits();
  }
  EVT getElementType() const {
    return ElementType;
  }
  unsigned getNumElements() const {
    return NumElements;
  }
};
} // end anonymous namespace

static std::map<unsigned, const Type *>
  ExtendedIntegerTypeMap;
static std::map<std::pair<uintptr_t, uintptr_t>, const Type *>
  ExtendedVectorTypeMap;

bool EVT::isExtendedFloatingPoint() const {
  assert(isExtended() && "Type is not extended!");
  // Extended floating-point types are not supported yet.
  return false;
}

bool EVT::isExtendedInteger() const {
  assert(isExtended() && "Type is not extended!");
  return isa<ExtendedIntegerType>(LLVMTy);
}

bool EVT::isExtendedVector() const {
  assert(isExtended() && "Type is not extended!");
  return isa<ExtendedVectorType>(LLVMTy);
}

bool EVT::isExtended64BitVector() const {
  assert(isExtended() && "Type is not extended!");
  return isExtendedVector() && getSizeInBits() == 64;
}

bool EVT::isExtended128BitVector() const {
  assert(isExtended() && "Type is not extended!");
  return isExtendedVector() && getSizeInBits() == 128;
}

EVT EVT::getExtendedVectorElementType() const {
  assert(isExtendedVector() && "Type is not an extended vector!");
  return static_cast<const ExtendedVectorType *>(LLVMTy)->getElementType();
}

unsigned EVT::getExtendedVectorNumElements() const {
  assert(isExtendedVector() && "Type is not an extended vector!");
  return static_cast<const ExtendedVectorType *>(LLVMTy)->getNumElements();
}

unsigned EVT::getExtendedSizeInBits() const {
  assert(isExtended() && "Type is not extended!");
  return LLVMTy->getSizeInBits();
}
