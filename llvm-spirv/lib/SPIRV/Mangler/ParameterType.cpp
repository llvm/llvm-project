//===------------------------ ParameterType.cpp --------------------------===//
//
//                              SPIR Tools
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
/*
 * Contributed by: Intel Corporation.
 */
#include "ParameterType.h"
#include "ManglingUtils.h"
#include <assert.h>
#include <cctype>
#include <sstream>

namespace SPIR {
//
// Primitive Type
//

PrimitiveType::PrimitiveType(TypePrimitiveEnum Primitive)
    : ParamType(TYPE_ID_PRIMITIVE), Primitive(Primitive) {}

MangleError PrimitiveType::accept(TypeVisitor *Visitor) const {
  if (getSupportedVersion(this->getPrimitive()) >= SPIR20 &&
      Visitor->SpirVer < SPIR20) {
    return MANGLE_TYPE_NOT_SUPPORTED;
  }
  return Visitor->visit(this);
}

std::string PrimitiveType::toString() const {
  assert((Primitive >= PRIMITIVE_FIRST && Primitive <= PRIMITIVE_LAST) &&
         "illegal primitive");
  std::stringstream MyName;
  MyName << readablePrimitiveString(Primitive);
  return MyName.str();
}

bool PrimitiveType::equals(const ParamType *Type) const {
  const PrimitiveType *P = SPIR::dynCast<PrimitiveType>(Type);
  return P && (Primitive == P->Primitive);
}

//
// Pointer Type
//

PointerType::PointerType(const RefParamType Type)
    : ParamType(TYPE_ID_POINTER), PType(Type) {
  for (unsigned int I = ATTR_QUALIFIER_FIRST; I <= ATTR_QUALIFIER_LAST; I++) {
    setQualifier((TypeAttributeEnum)I, false);
  }
  AddressSpace = ATTR_PRIVATE;
}

MangleError PointerType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

void PointerType::setAddressSpace(TypeAttributeEnum Attr) {
  if (Attr < ATTR_ADDR_SPACE_FIRST || Attr > ATTR_ADDR_SPACE_LAST) {
    return;
  }
  AddressSpace = Attr;
}

TypeAttributeEnum PointerType::getAddressSpace() const { return AddressSpace; }

void PointerType::setQualifier(TypeAttributeEnum Qual, bool Enabled) {
  if (Qual < ATTR_QUALIFIER_FIRST || Qual > ATTR_QUALIFIER_LAST) {
    return;
  }
  Qualifiers[Qual - ATTR_QUALIFIER_FIRST] = Enabled;
}

bool PointerType::hasQualifier(TypeAttributeEnum Qual) const {
  if (Qual < ATTR_QUALIFIER_FIRST || Qual > ATTR_QUALIFIER_LAST) {
    return false;
  }
  return Qualifiers[Qual - ATTR_QUALIFIER_FIRST];
}

std::string PointerType::toString() const {
  std::stringstream MyName;
  for (unsigned int I = ATTR_QUALIFIER_FIRST; I <= ATTR_QUALIFIER_LAST; I++) {
    TypeAttributeEnum Qual = (TypeAttributeEnum)I;
    if (hasQualifier(Qual)) {
      MyName << getReadableAttribute(Qual) << " ";
    }
  }
  MyName << getReadableAttribute(TypeAttributeEnum(AddressSpace)) << " ";
  MyName << getPointee()->toString() << " *";
  return MyName.str();
}

bool PointerType::equals(const ParamType *Type) const {
  const PointerType *P = SPIR::dynCast<PointerType>(Type);
  if (!P) {
    return false;
  }
  if (getAddressSpace() != P->getAddressSpace()) {
    return false;
  }
  for (unsigned int I = ATTR_QUALIFIER_FIRST; I <= ATTR_QUALIFIER_LAST; I++) {
    TypeAttributeEnum Qual = (TypeAttributeEnum)I;
    if (hasQualifier(Qual) != P->hasQualifier(Qual)) {
      return false;
    }
  }
  return (*getPointee()).equals(&*(P->getPointee()));
}

//
// Vector Type
//

VectorType::VectorType(const RefParamType Type, int Len)
    : ParamType(TYPE_ID_VECTOR), PType(Type), Len(Len) {}

MangleError VectorType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

std::string VectorType::toString() const {
  std::stringstream MyName;
  MyName << getScalarType()->toString();
  MyName << Len;
  return MyName.str();
}

bool VectorType::equals(const ParamType *Type) const {
  const VectorType *PVec = SPIR::dynCast<VectorType>(Type);
  return PVec && (Len == PVec->Len) &&
         (*getScalarType()).equals(&*(PVec->getScalarType()));
}

//
// Atomic Type
//

AtomicType::AtomicType(const RefParamType Type)
    : ParamType(TYPE_ID_ATOMIC), PType(Type) {}

MangleError AtomicType::accept(TypeVisitor *Visitor) const {
  if (Visitor->SpirVer < SPIR20) {
    return MANGLE_TYPE_NOT_SUPPORTED;
  }
  return Visitor->visit(this);
}

std::string AtomicType::toString() const {
  std::stringstream MyName;
  MyName << "atomic_" << getBaseType()->toString();
  return MyName.str();
}

bool AtomicType::equals(const ParamType *Type) const {
  const AtomicType *A = dynCast<AtomicType>(Type);
  return (A && (*getBaseType()).equals(&*(A->getBaseType())));
}

//
// Block Type
//

BlockType::BlockType() : ParamType(TYPE_ID_BLOCK) {}

MangleError BlockType::accept(TypeVisitor *Visitor) const {
  if (Visitor->SpirVer < SPIR20) {
    return MANGLE_TYPE_NOT_SUPPORTED;
  }
  return Visitor->visit(this);
}

std::string BlockType::toString() const {
  std::stringstream MyName;
  MyName << "void (";
  for (unsigned int I = 0; I < getNumOfParams(); ++I) {
    if (I > 0)
      MyName << ", ";
    MyName << Params[I]->toString();
  }
  MyName << ")*";
  return MyName.str();
}

bool BlockType::equals(const ParamType *Type) const {
  const BlockType *PBlock = dynCast<BlockType>(Type);
  if (!PBlock || getNumOfParams() != PBlock->getNumOfParams()) {
    return false;
  }
  for (unsigned int I = 0; I < getNumOfParams(); ++I) {
    if (!getParam(I)->equals(&*PBlock->getParam(I))) {
      return false;
    }
  }
  return true;
}

//
// User Defined Type
//
UserDefinedType::UserDefinedType(const std::string &Name)
    : ParamType(TYPE_ID_STRUCTURE), Name(Name) {}

MangleError UserDefinedType::accept(TypeVisitor *Visitor) const {
  return Visitor->visit(this);
}

std::string UserDefinedType::toString() const {
  std::stringstream MyName;
  MyName << Name;
  return MyName.str();
}

bool UserDefinedType::equals(const ParamType *PType) const {
  const UserDefinedType *PTy = SPIR::dynCast<UserDefinedType>(PType);
  return PTy && (Name == PTy->Name);
}

//
// Static enums
//
const TypeEnum PrimitiveType::EnumTy = TYPE_ID_PRIMITIVE;
const TypeEnum PointerType::EnumTy = TYPE_ID_POINTER;
const TypeEnum VectorType::EnumTy = TYPE_ID_VECTOR;
const TypeEnum AtomicType::EnumTy = TYPE_ID_ATOMIC;
const TypeEnum BlockType::EnumTy = TYPE_ID_BLOCK;
const TypeEnum UserDefinedType::EnumTy = TYPE_ID_STRUCTURE;

} // namespace SPIR
