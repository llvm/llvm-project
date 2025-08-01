#include "llvm/ABI/ABIInfo.h"

using namespace llvm::abi;
bool ABIInfo::isAggregateTypeForABI(const Type *Ty) const {
  // Member pointers are always aggregates
  if (Ty->isMemberPointer())
    return true;

  // Check for fundamental scalar types
  if (Ty->isInteger() || Ty->isFloat() || Ty->isPointer() || Ty->isVector())
    return false;

  // Everything else is treated as aggregate
  return true;
}

// Check if an integer type should be promoted
bool ABIInfo::isPromotableIntegerType(const IntegerType *Ty) const {
  unsigned BitWidth = Ty->getSizeInBits().getFixedValue();
  return BitWidth < 32;
}

// Create indirect return with natural alignment
ABIArgInfo ABIInfo::getNaturalAlignIndirect(const Type *Ty) const {
  return ABIArgInfo::getIndirect(Ty->getAlignment().value(), /*ByVal=*/true);
}
RecordArgABI ABIInfo::getRecordArgABI(const StructType *ST) const {
  if (!ST->canPassInRegisters())
    return RAA_Indirect;
  return RAA_Default;
}

bool ABIInfo::isZeroSizedType(const Type *Ty) const {
  return Ty->getSizeInBits().getFixedValue() == 0;
}

bool ABIInfo::isEmptyRecord(const StructType *ST) const {
  if (ST->hasFlexibleArrayMember()          ||
      ST->isPolymorphic()                   ||
      ST->getNumVirtualBaseClasses() != 0)    
    return false;

  for (unsigned I = 0; I < ST->getNumBaseClasses(); ++I) {
    const Type *BaseTy = ST->getBaseClasses()[I].FieldType;
    auto *BaseST = dyn_cast<StructType>(BaseTy);
    if (!BaseST || !isEmptyRecord(BaseST))
      return false;
  }

  for (unsigned I = 0; I < ST->getNumFields(); ++I) {
    const FieldInfo &FI = ST->getFields()[I];

    if (FI.IsBitField && FI.BitFieldWidth == 0)
      continue;
    if (FI.IsUnnamedBitfield)
      continue;

    if (!isZeroSizedType(FI.FieldType))
      return false;
  }
  return true;
}


bool ABIInfo::isEmptyField(const FieldInfo &FI) const  {
  if (FI.IsUnnamedBitfield)
    return true;
  if (FI.IsBitField && FI.BitFieldWidth == 0)
    return true;

  const Type *Ty = FI.FieldType;
  while (auto *AT = dyn_cast<ArrayType>(Ty)) {
    if (AT->getNumElements() != 1)
      break;
    Ty = AT->getElementType();
  }

  if (auto *ST = dyn_cast<StructType>(Ty))
    return isEmptyRecord(ST);

  return isZeroSizedType(Ty);
}
