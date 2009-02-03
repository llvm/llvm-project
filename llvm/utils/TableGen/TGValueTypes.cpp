//===- ValueTypes.cpp - Tablegen extended ValueType implementation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The MVT type is used by tablegen as well as in LLVM. In order to handle
// extended types, the MVT type uses support functions that call into
// LLVM's type system code. These aren't accessible in tablegen, so this
// file provides simple replacements.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Streams.h"
#include <map>
#include <vector>
using namespace llvm;

namespace llvm {

class Type {
public:
  virtual unsigned getSizeInBits() const = 0;
  virtual ~Type() {}
};

}

class ExtendedIntegerType : public Type {
  unsigned BitWidth;
public:
  explicit ExtendedIntegerType(unsigned bits)
    : BitWidth(bits) {}
  unsigned getSizeInBits() const {
    return getBitWidth();
  }
  unsigned getBitWidth() const {
    return BitWidth;
  }
};

class ExtendedVectorType : public Type {
  MVT ElementType;
  unsigned NumElements;
public:
  ExtendedVectorType(MVT elty, unsigned num)
    : ElementType(elty), NumElements(num) {}
  unsigned getSizeInBits() const {
    return getNumElements() * getElementType().getSizeInBits();
  }
  MVT getElementType() const {
    return ElementType;
  }
  unsigned getNumElements() const {
    return NumElements;
  }
};

static std::map<unsigned, const Type *>
  ExtendedIntegerTypeMap;
static std::map<std::pair<uintptr_t, uintptr_t>, const Type *>
  ExtendedVectorTypeMap;

MVT MVT::getExtendedIntegerVT(unsigned BitWidth) {
  const Type *&ET = ExtendedIntegerTypeMap[BitWidth];
  if (!ET) ET = new ExtendedIntegerType(BitWidth);
  MVT VT;
  VT.LLVMTy = ET;
  assert(VT.isExtended() && "Type is not extended!");
  return VT;
}

MVT MVT::getExtendedVectorVT(MVT VT, unsigned NumElements) {
  const Type *&ET = ExtendedVectorTypeMap[std::make_pair(VT.getRawBits(),
                                                         NumElements)];
  if (!ET) ET = new ExtendedVectorType(VT, NumElements);
  MVT ResultVT;
  ResultVT.LLVMTy = ET;
  assert(ResultVT.isExtended() && "Type is not extended!");
  return ResultVT;
}

bool MVT::isExtendedFloatingPoint() const {
  assert(isExtended() && "Type is not extended!");
  // Extended floating-point types are not supported yet.
  return false;
}

bool MVT::isExtendedInteger() const {
  assert(isExtended() && "Type is not extended!");
  return dynamic_cast<const ExtendedIntegerType *>(LLVMTy) != 0;
}

bool MVT::isExtendedVector() const {
  assert(isExtended() && "Type is not extended!");
  return dynamic_cast<const ExtendedVectorType *>(LLVMTy) != 0;
}

bool MVT::isExtended64BitVector() const {
  assert(isExtended() && "Type is not extended!");
  return isExtendedVector() && getSizeInBits() == 64;
}

bool MVT::isExtended128BitVector() const {
  assert(isExtended() && "Type is not extended!");
  return isExtendedVector() && getSizeInBits() == 128;
}

MVT MVT::getExtendedVectorElementType() const {
  assert(isExtendedVector() && "Type is not an extended vector!");
  return static_cast<const ExtendedVectorType *>(LLVMTy)->getElementType();
}

unsigned MVT::getExtendedVectorNumElements() const {
  assert(isExtendedVector() && "Type is not an extended vector!");
  return static_cast<const ExtendedVectorType *>(LLVMTy)->getNumElements();
}

unsigned MVT::getExtendedSizeInBits() const {
  assert(isExtended() && "Type is not extended!");
  return LLVMTy->getSizeInBits();
}
