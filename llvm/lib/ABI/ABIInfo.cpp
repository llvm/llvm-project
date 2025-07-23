#include "llvm/ABI/ABIInfo.h"

using namespace llvm::abi;
bool ABIInfo::isAggregateTypeForABI(const Type *Ty) const {
  // Member pointers are always aggregates
  if (Ty->isMemberPointer())
    return true;

  // Check for fundamental scalar types
  if (Ty->isInteger() || Ty->isFloat() || Ty->isPointer())
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
