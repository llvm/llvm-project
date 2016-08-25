//===-- Attributes.cpp - Implement AttributesList -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// \brief This file implements the Attribute, AttributeImpl, AttrBuilder,
// AttributeSetImpl, and AttributeSet classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "AttributeImpl.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Attribute Construction Methods
//===----------------------------------------------------------------------===//

// allocsize has two integer arguments, but because they're both 32 bits, we can
// pack them into one 64-bit value, at the cost of making said value
// nonsensical.
//
// In order to do this, we need to reserve one value of the second (optional)
// allocsize argument to signify "not present."
static const unsigned AllocSizeNumElemsNotPresent = -1;

static uint64_t packAllocSizeArgs(unsigned ElemSizeArg,
                                  const Optional<unsigned> &NumElemsArg) {
  assert((!NumElemsArg.hasValue() ||
          *NumElemsArg != AllocSizeNumElemsNotPresent) &&
         "Attempting to pack a reserved value");

  return uint64_t(ElemSizeArg) << 32 |
         NumElemsArg.getValueOr(AllocSizeNumElemsNotPresent);
}

static std::pair<unsigned, Optional<unsigned>>
unpackAllocSizeArgs(uint64_t Num) {
  unsigned NumElems = Num & std::numeric_limits<unsigned>::max();
  unsigned ElemSizeArg = Num >> 32;

  Optional<unsigned> NumElemsArg;
  if (NumElems != AllocSizeNumElemsNotPresent)
    NumElemsArg = NumElems;
  return std::make_pair(ElemSizeArg, NumElemsArg);
}

Attribute Attribute::get(LLVMContext &Context, Attribute::AttrKind Kind,
                         uint64_t Val) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddInteger(Kind);
  if (Val) ID.AddInteger(Val);

  void *InsertPoint;
  AttributeImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    if (!Val)
      PA = new EnumAttributeImpl(Kind);
    else
      PA = new IntAttributeImpl(Kind, Val);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the Attribute that we found or created.
  return Attribute(PA);
}

Attribute Attribute::get(LLVMContext &Context, StringRef Kind, StringRef Val) {
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddString(Kind);
  if (!Val.empty()) ID.AddString(Val);

  void *InsertPoint;
  AttributeImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = new StringAttributeImpl(Kind, Val);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the Attribute that we found or created.
  return Attribute(PA);
}

Attribute Attribute::getWithAlignment(LLVMContext &Context, uint64_t Align) {
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");
  return get(Context, Alignment, Align);
}

Attribute Attribute::getWithStackAlignment(LLVMContext &Context,
                                           uint64_t Align) {
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");
  return get(Context, StackAlignment, Align);
}

Attribute Attribute::getWithDereferenceableBytes(LLVMContext &Context,
                                                uint64_t Bytes) {
  assert(Bytes && "Bytes must be non-zero.");
  return get(Context, Dereferenceable, Bytes);
}

Attribute Attribute::getWithDereferenceableOrNullBytes(LLVMContext &Context,
                                                       uint64_t Bytes) {
  assert(Bytes && "Bytes must be non-zero.");
  return get(Context, DereferenceableOrNull, Bytes);
}

Attribute
Attribute::getWithAllocSizeArgs(LLVMContext &Context, unsigned ElemSizeArg,
                                const Optional<unsigned> &NumElemsArg) {
  assert(!(ElemSizeArg == 0 && NumElemsArg && *NumElemsArg == 0) &&
         "Invalid allocsize arguments -- given allocsize(0, 0)");
  return get(Context, AllocSize, packAllocSizeArgs(ElemSizeArg, NumElemsArg));
}

//===----------------------------------------------------------------------===//
// Attribute Accessor Methods
//===----------------------------------------------------------------------===//

bool Attribute::isEnumAttribute() const {
  return pImpl && pImpl->isEnumAttribute();
}

bool Attribute::isIntAttribute() const {
  return pImpl && pImpl->isIntAttribute();
}

bool Attribute::isStringAttribute() const {
  return pImpl && pImpl->isStringAttribute();
}

Attribute::AttrKind Attribute::getKindAsEnum() const {
  if (!pImpl) return None;
  assert((isEnumAttribute() || isIntAttribute()) &&
         "Invalid attribute type to get the kind as an enum!");
  return pImpl->getKindAsEnum();
}

uint64_t Attribute::getValueAsInt() const {
  if (!pImpl) return 0;
  assert(isIntAttribute() &&
         "Expected the attribute to be an integer attribute!");
  return pImpl->getValueAsInt();
}

StringRef Attribute::getKindAsString() const {
  if (!pImpl) return StringRef();
  assert(isStringAttribute() &&
         "Invalid attribute type to get the kind as a string!");
  return pImpl->getKindAsString();
}

StringRef Attribute::getValueAsString() const {
  if (!pImpl) return StringRef();
  assert(isStringAttribute() &&
         "Invalid attribute type to get the value as a string!");
  return pImpl->getValueAsString();
}

bool Attribute::hasAttribute(AttrKind Kind) const {
  return (pImpl && pImpl->hasAttribute(Kind)) || (!pImpl && Kind == None);
}

bool Attribute::hasAttribute(StringRef Kind) const {
  if (!isStringAttribute()) return false;
  return pImpl && pImpl->hasAttribute(Kind);
}

unsigned Attribute::getAlignment() const {
  assert(hasAttribute(Attribute::Alignment) &&
         "Trying to get alignment from non-alignment attribute!");
  return pImpl->getValueAsInt();
}

unsigned Attribute::getStackAlignment() const {
  assert(hasAttribute(Attribute::StackAlignment) &&
         "Trying to get alignment from non-alignment attribute!");
  return pImpl->getValueAsInt();
}

uint64_t Attribute::getDereferenceableBytes() const {
  assert(hasAttribute(Attribute::Dereferenceable) &&
         "Trying to get dereferenceable bytes from "
         "non-dereferenceable attribute!");
  return pImpl->getValueAsInt();
}

uint64_t Attribute::getDereferenceableOrNullBytes() const {
  assert(hasAttribute(Attribute::DereferenceableOrNull) &&
         "Trying to get dereferenceable bytes from "
         "non-dereferenceable attribute!");
  return pImpl->getValueAsInt();
}

std::pair<unsigned, Optional<unsigned>> Attribute::getAllocSizeArgs() const {
  assert(hasAttribute(Attribute::AllocSize) &&
         "Trying to get allocsize args from non-allocsize attribute");
  return unpackAllocSizeArgs(pImpl->getValueAsInt());
}

std::string Attribute::getAsString(bool InAttrGrp) const {
  if (!pImpl) return "";

  if (hasAttribute(Attribute::SanitizeAddress))
    return "sanitize_address";
  if (hasAttribute(Attribute::AlwaysInline))
    return "alwaysinline";
  if (hasAttribute(Attribute::ArgMemOnly))
    return "argmemonly";
  if (hasAttribute(Attribute::Builtin))
    return "builtin";
  if (hasAttribute(Attribute::ByVal))
    return "byval";
  if (hasAttribute(Attribute::Convergent))
    return "convergent";
  if (hasAttribute(Attribute::SwiftError))
    return "swifterror";
  if (hasAttribute(Attribute::SwiftSelf))
    return "swiftself";
  if (hasAttribute(Attribute::InaccessibleMemOnly))
    return "inaccessiblememonly";
  if (hasAttribute(Attribute::InaccessibleMemOrArgMemOnly))
    return "inaccessiblemem_or_argmemonly";
  if (hasAttribute(Attribute::InAlloca))
    return "inalloca";
  if (hasAttribute(Attribute::InlineHint))
    return "inlinehint";
  if (hasAttribute(Attribute::InReg))
    return "inreg";
  if (hasAttribute(Attribute::JumpTable))
    return "jumptable";
  if (hasAttribute(Attribute::MinSize))
    return "minsize";
  if (hasAttribute(Attribute::Naked))
    return "naked";
  if (hasAttribute(Attribute::Nest))
    return "nest";
  if (hasAttribute(Attribute::NoAlias))
    return "noalias";
  if (hasAttribute(Attribute::NoBuiltin))
    return "nobuiltin";
  if (hasAttribute(Attribute::NoCapture))
    return "nocapture";
  if (hasAttribute(Attribute::NoDuplicate))
    return "noduplicate";
  if (hasAttribute(Attribute::NoImplicitFloat))
    return "noimplicitfloat";
  if (hasAttribute(Attribute::NoInline))
    return "noinline";
  if (hasAttribute(Attribute::NonLazyBind))
    return "nonlazybind";
  if (hasAttribute(Attribute::NonNull))
    return "nonnull";
  if (hasAttribute(Attribute::NoRedZone))
    return "noredzone";
  if (hasAttribute(Attribute::NoReturn))
    return "noreturn";
  if (hasAttribute(Attribute::NoRecurse))
    return "norecurse";
  if (hasAttribute(Attribute::NoUnwind))
    return "nounwind";
  if (hasAttribute(Attribute::OptimizeNone))
    return "optnone";
  if (hasAttribute(Attribute::OptimizeForSize))
    return "optsize";
  if (hasAttribute(Attribute::ReadNone))
    return "readnone";
  if (hasAttribute(Attribute::ReadOnly))
    return "readonly";
  if (hasAttribute(Attribute::WriteOnly))
    return "writeonly";
  if (hasAttribute(Attribute::Returned))
    return "returned";
  if (hasAttribute(Attribute::ReturnsTwice))
    return "returns_twice";
  if (hasAttribute(Attribute::SExt))
    return "signext";
  if (hasAttribute(Attribute::StackProtect))
    return "ssp";
  if (hasAttribute(Attribute::StackProtectReq))
    return "sspreq";
  if (hasAttribute(Attribute::StackProtectStrong))
    return "sspstrong";
  if (hasAttribute(Attribute::SafeStack))
    return "safestack";
  if (hasAttribute(Attribute::StructRet))
    return "sret";
  if (hasAttribute(Attribute::SanitizeThread))
    return "sanitize_thread";
  if (hasAttribute(Attribute::SanitizeMemory))
    return "sanitize_memory";
  if (hasAttribute(Attribute::UWTable))
    return "uwtable";
  if (hasAttribute(Attribute::ZExt))
    return "zeroext";
  if (hasAttribute(Attribute::Cold))
    return "cold";

  // FIXME: These should be output like this:
  //
  //   align=4
  //   alignstack=8
  //
  if (hasAttribute(Attribute::Alignment)) {
    std::string Result;
    Result += "align";
    Result += (InAttrGrp) ? "=" : " ";
    Result += utostr(getValueAsInt());
    return Result;
  }

  auto AttrWithBytesToString = [&](const char *Name) {
    std::string Result;
    Result += Name;
    if (InAttrGrp) {
      Result += "=";
      Result += utostr(getValueAsInt());
    } else {
      Result += "(";
      Result += utostr(getValueAsInt());
      Result += ")";
    }
    return Result;
  };

  if (hasAttribute(Attribute::StackAlignment))
    return AttrWithBytesToString("alignstack");

  if (hasAttribute(Attribute::Dereferenceable))
    return AttrWithBytesToString("dereferenceable");

  if (hasAttribute(Attribute::DereferenceableOrNull))
    return AttrWithBytesToString("dereferenceable_or_null");

  if (hasAttribute(Attribute::AllocSize)) {
    unsigned ElemSize;
    Optional<unsigned> NumElems;
    std::tie(ElemSize, NumElems) = getAllocSizeArgs();

    std::string Result = "allocsize(";
    Result += utostr(ElemSize);
    if (NumElems.hasValue()) {
      Result += ',';
      Result += utostr(*NumElems);
    }
    Result += ')';
    return Result;
  }

  // Convert target-dependent attributes to strings of the form:
  //
  //   "kind"
  //   "kind" = "value"
  //
  if (isStringAttribute()) {
    std::string Result;
    Result += (Twine('"') + getKindAsString() + Twine('"')).str();

    StringRef Val = pImpl->getValueAsString();
    if (Val.empty()) return Result;

    Result += ("=\"" + Val + Twine('"')).str();
    return Result;
  }

  llvm_unreachable("Unknown attribute");
}

bool Attribute::operator<(Attribute A) const {
  if (!pImpl && !A.pImpl) return false;
  if (!pImpl) return true;
  if (!A.pImpl) return false;
  return *pImpl < *A.pImpl;
}

//===----------------------------------------------------------------------===//
// AttributeImpl Definition
//===----------------------------------------------------------------------===//

// Pin the vtables to this file.
AttributeImpl::~AttributeImpl() {}
void EnumAttributeImpl::anchor() {}
void IntAttributeImpl::anchor() {}
void StringAttributeImpl::anchor() {}

bool AttributeImpl::hasAttribute(Attribute::AttrKind A) const {
  if (isStringAttribute()) return false;
  return getKindAsEnum() == A;
}

bool AttributeImpl::hasAttribute(StringRef Kind) const {
  if (!isStringAttribute()) return false;
  return getKindAsString() == Kind;
}

Attribute::AttrKind AttributeImpl::getKindAsEnum() const {
  assert(isEnumAttribute() || isIntAttribute());
  return static_cast<const EnumAttributeImpl *>(this)->getEnumKind();
}

uint64_t AttributeImpl::getValueAsInt() const {
  assert(isIntAttribute());
  return static_cast<const IntAttributeImpl *>(this)->getValue();
}

StringRef AttributeImpl::getKindAsString() const {
  assert(isStringAttribute());
  return static_cast<const StringAttributeImpl *>(this)->getStringKind();
}

StringRef AttributeImpl::getValueAsString() const {
  assert(isStringAttribute());
  return static_cast<const StringAttributeImpl *>(this)->getStringValue();
}

bool AttributeImpl::operator<(const AttributeImpl &AI) const {
  // This sorts the attributes with Attribute::AttrKinds coming first (sorted
  // relative to their enum value) and then strings.
  if (isEnumAttribute()) {
    if (AI.isEnumAttribute()) return getKindAsEnum() < AI.getKindAsEnum();
    if (AI.isIntAttribute()) return true;
    if (AI.isStringAttribute()) return true;
  }

  if (isIntAttribute()) {
    if (AI.isEnumAttribute()) return false;
    if (AI.isIntAttribute()) {
      if (getKindAsEnum() == AI.getKindAsEnum())
        return getValueAsInt() < AI.getValueAsInt();
      return getKindAsEnum() < AI.getKindAsEnum();
    }
    if (AI.isStringAttribute()) return true;
  }

  if (AI.isEnumAttribute()) return false;
  if (AI.isIntAttribute()) return false;
  if (getKindAsString() == AI.getKindAsString())
    return getValueAsString() < AI.getValueAsString();
  return getKindAsString() < AI.getKindAsString();
}

uint64_t AttributeImpl::getAttrMask(Attribute::AttrKind Val) {
  // FIXME: Remove this.
  switch (Val) {
  case Attribute::EndAttrKinds:
    llvm_unreachable("Synthetic enumerators which should never get here");

  case Attribute::None:            return 0;
  case Attribute::ZExt:            return 1 << 0;
  case Attribute::SExt:            return 1 << 1;
  case Attribute::NoReturn:        return 1 << 2;
  case Attribute::InReg:           return 1 << 3;
  case Attribute::StructRet:       return 1 << 4;
  case Attribute::NoUnwind:        return 1 << 5;
  case Attribute::NoAlias:         return 1 << 6;
  case Attribute::ByVal:           return 1 << 7;
  case Attribute::Nest:            return 1 << 8;
  case Attribute::ReadNone:        return 1 << 9;
  case Attribute::ReadOnly:        return 1 << 10;
  case Attribute::NoInline:        return 1 << 11;
  case Attribute::AlwaysInline:    return 1 << 12;
  case Attribute::OptimizeForSize: return 1 << 13;
  case Attribute::StackProtect:    return 1 << 14;
  case Attribute::StackProtectReq: return 1 << 15;
  case Attribute::Alignment:       return 31 << 16;
  case Attribute::NoCapture:       return 1 << 21;
  case Attribute::NoRedZone:       return 1 << 22;
  case Attribute::NoImplicitFloat: return 1 << 23;
  case Attribute::Naked:           return 1 << 24;
  case Attribute::InlineHint:      return 1 << 25;
  case Attribute::StackAlignment:  return 7 << 26;
  case Attribute::ReturnsTwice:    return 1 << 29;
  case Attribute::UWTable:         return 1 << 30;
  case Attribute::NonLazyBind:     return 1U << 31;
  case Attribute::SanitizeAddress: return 1ULL << 32;
  case Attribute::MinSize:         return 1ULL << 33;
  case Attribute::NoDuplicate:     return 1ULL << 34;
  case Attribute::StackProtectStrong: return 1ULL << 35;
  case Attribute::SanitizeThread:  return 1ULL << 36;
  case Attribute::SanitizeMemory:  return 1ULL << 37;
  case Attribute::NoBuiltin:       return 1ULL << 38;
  case Attribute::Returned:        return 1ULL << 39;
  case Attribute::Cold:            return 1ULL << 40;
  case Attribute::Builtin:         return 1ULL << 41;
  case Attribute::OptimizeNone:    return 1ULL << 42;
  case Attribute::InAlloca:        return 1ULL << 43;
  case Attribute::NonNull:         return 1ULL << 44;
  case Attribute::JumpTable:       return 1ULL << 45;
  case Attribute::Convergent:      return 1ULL << 46;
  case Attribute::SafeStack:       return 1ULL << 47;
  case Attribute::NoRecurse:       return 1ULL << 48;
  case Attribute::InaccessibleMemOnly:         return 1ULL << 49;
  case Attribute::InaccessibleMemOrArgMemOnly: return 1ULL << 50;
  case Attribute::SwiftSelf:       return 1ULL << 51;
  case Attribute::SwiftError:      return 1ULL << 52;
  case Attribute::WriteOnly:       return 1ULL << 53;
  case Attribute::Dereferenceable:
    llvm_unreachable("dereferenceable attribute not supported in raw format");
    break;
  case Attribute::DereferenceableOrNull:
    llvm_unreachable("dereferenceable_or_null attribute not supported in raw "
                     "format");
    break;
  case Attribute::ArgMemOnly:
    llvm_unreachable("argmemonly attribute not supported in raw format");
    break;
  case Attribute::AllocSize:
    llvm_unreachable("allocsize not supported in raw format");
    break;
  }
  llvm_unreachable("Unsupported attribute type");
}

//===----------------------------------------------------------------------===//
// AttributeSetNode Definition
//===----------------------------------------------------------------------===//

AttributeSetNode *AttributeSetNode::get(LLVMContext &C,
                                        ArrayRef<Attribute> Attrs) {
  if (Attrs.empty())
    return nullptr;

  // Otherwise, build a key to look up the existing attributes.
  LLVMContextImpl *pImpl = C.pImpl;
  FoldingSetNodeID ID;

  SmallVector<Attribute, 8> SortedAttrs(Attrs.begin(), Attrs.end());
  std::sort(SortedAttrs.begin(), SortedAttrs.end());

  for (Attribute Attr : SortedAttrs)
    Attr.Profile(ID);

  void *InsertPoint;
  AttributeSetNode *PA =
    pImpl->AttrsSetNodes.FindNodeOrInsertPos(ID, InsertPoint);

  // If we didn't find any existing attributes of the same shape then create a
  // new one and insert it.
  if (!PA) {
    // Coallocate entries after the AttributeSetNode itself.
    void *Mem = ::operator new(totalSizeToAlloc<Attribute>(SortedAttrs.size()));
    PA = new (Mem) AttributeSetNode(SortedAttrs);
    pImpl->AttrsSetNodes.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesListNode that we found or created.
  return PA;
}

bool AttributeSetNode::hasAttribute(StringRef Kind) const {
  for (Attribute I : *this)
    if (I.hasAttribute(Kind))
      return true;
  return false;
}

Attribute AttributeSetNode::getAttribute(Attribute::AttrKind Kind) const {
  if (hasAttribute(Kind)) {
    for (Attribute I : *this)
      if (I.hasAttribute(Kind))
        return I;
  }
  return Attribute();
}

Attribute AttributeSetNode::getAttribute(StringRef Kind) const {
  for (Attribute I : *this)
    if (I.hasAttribute(Kind))
      return I;
  return Attribute();
}

unsigned AttributeSetNode::getAlignment() const {
  for (Attribute I : *this)
    if (I.hasAttribute(Attribute::Alignment))
      return I.getAlignment();
  return 0;
}

unsigned AttributeSetNode::getStackAlignment() const {
  for (Attribute I : *this)
    if (I.hasAttribute(Attribute::StackAlignment))
      return I.getStackAlignment();
  return 0;
}

uint64_t AttributeSetNode::getDereferenceableBytes() const {
  for (Attribute I : *this)
    if (I.hasAttribute(Attribute::Dereferenceable))
      return I.getDereferenceableBytes();
  return 0;
}

uint64_t AttributeSetNode::getDereferenceableOrNullBytes() const {
  for (Attribute I : *this)
    if (I.hasAttribute(Attribute::DereferenceableOrNull))
      return I.getDereferenceableOrNullBytes();
  return 0;
}

std::pair<unsigned, Optional<unsigned>>
AttributeSetNode::getAllocSizeArgs() const {
  for (Attribute I : *this)
    if (I.hasAttribute(Attribute::AllocSize))
      return I.getAllocSizeArgs();
  return std::make_pair(0, 0);
}

std::string AttributeSetNode::getAsString(bool InAttrGrp) const {
  std::string Str;
  for (iterator I = begin(), E = end(); I != E; ++I) {
    if (I != begin())
      Str += ' ';
    Str += I->getAsString(InAttrGrp);
  }
  return Str;
}

//===----------------------------------------------------------------------===//
// AttributeSetImpl Definition
//===----------------------------------------------------------------------===//

uint64_t AttributeSetImpl::Raw(unsigned Index) const {
  for (unsigned I = 0, E = getNumSlots(); I != E; ++I) {
    if (getSlotIndex(I) != Index) continue;
    const AttributeSetNode *ASN = getSlotNode(I);
    uint64_t Mask = 0;

    for (AttributeSetNode::iterator II = ASN->begin(),
           IE = ASN->end(); II != IE; ++II) {
      Attribute Attr = *II;

      // This cannot handle string attributes.
      if (Attr.isStringAttribute()) continue;

      Attribute::AttrKind Kind = Attr.getKindAsEnum();

      if (Kind == Attribute::Alignment)
        Mask |= (Log2_32(ASN->getAlignment()) + 1) << 16;
      else if (Kind == Attribute::StackAlignment)
        Mask |= (Log2_32(ASN->getStackAlignment()) + 1) << 26;
      else if (Kind == Attribute::Dereferenceable)
        llvm_unreachable("dereferenceable not supported in bit mask");
      else if (Kind == Attribute::AllocSize)
        llvm_unreachable("allocsize not supported in bit mask");
      else
        Mask |= AttributeImpl::getAttrMask(Kind);
    }

    return Mask;
  }

  return 0;
}

LLVM_DUMP_METHOD void AttributeSetImpl::dump() const {
  AttributeSet(const_cast<AttributeSetImpl *>(this)).dump();
}

//===----------------------------------------------------------------------===//
// AttributeSet Construction and Mutation Methods
//===----------------------------------------------------------------------===//

AttributeSet
AttributeSet::getImpl(LLVMContext &C,
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> > Attrs) {
  LLVMContextImpl *pImpl = C.pImpl;
  FoldingSetNodeID ID;
  AttributeSetImpl::Profile(ID, Attrs);

  void *InsertPoint;
  AttributeSetImpl *PA = pImpl->AttrsLists.FindNodeOrInsertPos(ID, InsertPoint);

  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PA) {
    // Coallocate entries after the AttributeSetImpl itself.
    void *Mem = ::operator new(
        AttributeSetImpl::totalSizeToAlloc<IndexAttrPair>(Attrs.size()));
    PA = new (Mem) AttributeSetImpl(C, Attrs);
    pImpl->AttrsLists.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return AttributeSet(PA);
}

AttributeSet AttributeSet::get(LLVMContext &C,
                               ArrayRef<std::pair<unsigned, Attribute> > Attrs){
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return AttributeSet();

  assert(std::is_sorted(Attrs.begin(), Attrs.end(),
                        [](const std::pair<unsigned, Attribute> &LHS,
                           const std::pair<unsigned, Attribute> &RHS) {
                          return LHS.first < RHS.first;
                        }) && "Misordered Attributes list!");
  assert(none_of(Attrs,
                 [](const std::pair<unsigned, Attribute> &Pair) {
                   return Pair.second.hasAttribute(Attribute::None);
                 }) &&
         "Pointless attribute!");

  // Create a vector if (unsigned, AttributeSetNode*) pairs from the attributes
  // list.
  SmallVector<std::pair<unsigned, AttributeSetNode*>, 8> AttrPairVec;
  for (ArrayRef<std::pair<unsigned, Attribute> >::iterator I = Attrs.begin(),
         E = Attrs.end(); I != E; ) {
    unsigned Index = I->first;
    SmallVector<Attribute, 4> AttrVec;
    while (I != E && I->first == Index) {
      AttrVec.push_back(I->second);
      ++I;
    }

    AttrPairVec.emplace_back(Index, AttributeSetNode::get(C, AttrVec));
  }

  return getImpl(C, AttrPairVec);
}

AttributeSet AttributeSet::get(LLVMContext &C,
                               ArrayRef<std::pair<unsigned,
                                                  AttributeSetNode*> > Attrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return AttributeSet();

  return getImpl(C, Attrs);
}

AttributeSet AttributeSet::get(LLVMContext &C, unsigned Index,
                               const AttrBuilder &B) {
  if (!B.hasAttributes())
    return AttributeSet();

  // Add target-independent attributes.
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (Attribute::AttrKind Kind = Attribute::None;
       Kind != Attribute::EndAttrKinds; Kind = Attribute::AttrKind(Kind + 1)) {
    if (!B.contains(Kind))
      continue;

    Attribute Attr;
    switch (Kind) {
    case Attribute::Alignment:
      Attr = Attribute::getWithAlignment(C, B.getAlignment());
      break;
    case Attribute::StackAlignment:
      Attr = Attribute::getWithStackAlignment(C, B.getStackAlignment());
      break;
    case Attribute::Dereferenceable:
      Attr = Attribute::getWithDereferenceableBytes(
          C, B.getDereferenceableBytes());
      break;
    case Attribute::DereferenceableOrNull:
      Attr = Attribute::getWithDereferenceableOrNullBytes(
          C, B.getDereferenceableOrNullBytes());
      break;
    case Attribute::AllocSize: {
      auto A = B.getAllocSizeArgs();
      Attr = Attribute::getWithAllocSizeArgs(C, A.first, A.second);
      break;
    }
    default:
      Attr = Attribute::get(C, Kind);
    }
    Attrs.emplace_back(Index, Attr);
  }

  // Add target-dependent (string) attributes.
  for (const auto &TDA : B.td_attrs())
    Attrs.emplace_back(Index, Attribute::get(C, TDA.first, TDA.second));

  return get(C, Attrs);
}

AttributeSet AttributeSet::get(LLVMContext &C, unsigned Index,
                               ArrayRef<Attribute::AttrKind> Kinds) {
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (Attribute::AttrKind K : Kinds)
    Attrs.emplace_back(Index, Attribute::get(C, K));
  return get(C, Attrs);
}

AttributeSet AttributeSet::get(LLVMContext &C, unsigned Index,
                               ArrayRef<StringRef> Kinds) {
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (StringRef K : Kinds)
    Attrs.emplace_back(Index, Attribute::get(C, K));
  return get(C, Attrs);
}

AttributeSet AttributeSet::get(LLVMContext &C, ArrayRef<AttributeSet> Attrs) {
  if (Attrs.empty()) return AttributeSet();
  if (Attrs.size() == 1) return Attrs[0];

  SmallVector<std::pair<unsigned, AttributeSetNode*>, 8> AttrNodeVec;
  AttributeSetImpl *A0 = Attrs[0].pImpl;
  if (A0)
    AttrNodeVec.append(A0->getNode(0), A0->getNode(A0->getNumSlots()));
  // Copy all attributes from Attrs into AttrNodeVec while keeping AttrNodeVec
  // ordered by index.  Because we know that each list in Attrs is ordered by
  // index we only need to merge each successive list in rather than doing a
  // full sort.
  for (unsigned I = 1, E = Attrs.size(); I != E; ++I) {
    AttributeSetImpl *AS = Attrs[I].pImpl;
    if (!AS) continue;
    SmallVector<std::pair<unsigned, AttributeSetNode *>, 8>::iterator
      ANVI = AttrNodeVec.begin(), ANVE;
    for (const IndexAttrPair *AI = AS->getNode(0),
                             *AE = AS->getNode(AS->getNumSlots());
         AI != AE; ++AI) {
      ANVE = AttrNodeVec.end();
      while (ANVI != ANVE && ANVI->first <= AI->first)
        ++ANVI;
      ANVI = AttrNodeVec.insert(ANVI, *AI) + 1;
    }
  }

  return getImpl(C, AttrNodeVec);
}

AttributeSet AttributeSet::addAttribute(LLVMContext &C, unsigned Index,
                                        Attribute::AttrKind Kind) const {
  if (hasAttribute(Index, Kind)) return *this;
  return addAttributes(C, Index, AttributeSet::get(C, Index, Kind));
}

AttributeSet AttributeSet::addAttribute(LLVMContext &C, unsigned Index,
                                        StringRef Kind, StringRef Value) const {
  llvm::AttrBuilder B;
  B.addAttribute(Kind, Value);
  return addAttributes(C, Index, AttributeSet::get(C, Index, B));
}

AttributeSet AttributeSet::addAttribute(LLVMContext &C,
                                        ArrayRef<unsigned> Indices,
                                        Attribute A) const {
  unsigned I = 0, E = pImpl ? pImpl->getNumSlots() : 0;
  auto IdxI = Indices.begin(), IdxE = Indices.end();
  SmallVector<AttributeSet, 4> AttrSet;

  while (I != E && IdxI != IdxE) {
    if (getSlotIndex(I) < *IdxI)
      AttrSet.emplace_back(getSlotAttributes(I++));
    else if (getSlotIndex(I) > *IdxI)
      AttrSet.emplace_back(AttributeSet::get(C, std::make_pair(*IdxI++, A)));
    else {
      AttrBuilder B(getSlotAttributes(I), *IdxI);
      B.addAttribute(A);
      AttrSet.emplace_back(AttributeSet::get(C, *IdxI, B));
      ++I;
      ++IdxI;
    }
  }

  while (I != E)
    AttrSet.emplace_back(getSlotAttributes(I++));

  while (IdxI != IdxE)
    AttrSet.emplace_back(AttributeSet::get(C, std::make_pair(*IdxI++, A)));

  return get(C, AttrSet);
}

AttributeSet AttributeSet::addAttributes(LLVMContext &C, unsigned Index,
                                         AttributeSet Attrs) const {
  if (!pImpl) return Attrs;
  if (!Attrs.pImpl) return *this;

#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment. For now, say
  // we can't change a known alignment.
  unsigned OldAlign = getParamAlignment(Index);
  unsigned NewAlign = Attrs.getParamAlignment(Index);
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeSet, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumSlots();
  AttributeSet AS;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Index) {
      if (getSlotIndex(I) == Index) AS = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now add the attribute into the correct slot. There may already be an
  // AttributeSet there.
  AttrBuilder B(AS, Index);

  for (unsigned I = 0, E = Attrs.pImpl->getNumSlots(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Index) {
      for (AttributeSetImpl::iterator II = Attrs.pImpl->begin(I),
             IE = Attrs.pImpl->end(I); II != IE; ++II)
        B.addAttribute(*II);
      break;
    }

  AttrSet.push_back(AttributeSet::get(C, Index, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

AttributeSet AttributeSet::removeAttribute(LLVMContext &C, unsigned Index,
                                           Attribute::AttrKind Kind) const {
  if (!hasAttribute(Index, Kind)) return *this;
  return removeAttributes(C, Index, AttributeSet::get(C, Index, Kind));
}

AttributeSet AttributeSet::removeAttribute(LLVMContext &C, unsigned Index,
                                           StringRef Kind) const {
  if (!hasAttribute(Index, Kind)) return *this;
  return removeAttributes(C, Index, AttributeSet::get(C, Index, Kind));
}

AttributeSet AttributeSet::removeAttributes(LLVMContext &C, unsigned Index,
                                            AttributeSet Attrs) const {
  if (!pImpl) return AttributeSet();
  if (!Attrs.pImpl) return *this;

  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAttribute(Index, Attribute::Alignment) &&
         "Attempt to change alignment!");

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeSet, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumSlots();
  AttributeSet AS;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Index) {
      if (getSlotIndex(I) == Index) AS = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now remove the attribute from the correct slot. There may already be an
  // AttributeSet there.
  AttrBuilder B(AS, Index);

  for (unsigned I = 0, E = Attrs.pImpl->getNumSlots(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Index) {
      B.removeAttributes(Attrs.pImpl->getSlotAttributes(I), Index);
      break;
    }

  AttrSet.push_back(AttributeSet::get(C, Index, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

AttributeSet AttributeSet::removeAttributes(LLVMContext &C, unsigned Index,
                                            const AttrBuilder &Attrs) const {
  if (!pImpl) return AttributeSet();

  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAlignmentAttr() && "Attempt to change alignment!");

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeSet, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumSlots();
  AttributeSet AS;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Index) {
      if (getSlotIndex(I) == Index) AS = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now remove the attribute from the correct slot. There may already be an
  // AttributeSet there.
  AttrBuilder B(AS, Index);
  B.remove(Attrs);

  AttrSet.push_back(AttributeSet::get(C, Index, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

AttributeSet AttributeSet::addDereferenceableAttr(LLVMContext &C, unsigned Index,
                                                  uint64_t Bytes) const {
  llvm::AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
  return addAttributes(C, Index, AttributeSet::get(C, Index, B));
}

AttributeSet AttributeSet::addDereferenceableOrNullAttr(LLVMContext &C,
                                                        unsigned Index,
                                                        uint64_t Bytes) const {
  llvm::AttrBuilder B;
  B.addDereferenceableOrNullAttr(Bytes);
  return addAttributes(C, Index, AttributeSet::get(C, Index, B));
}

AttributeSet
AttributeSet::addAllocSizeAttr(LLVMContext &C, unsigned Index,
                               unsigned ElemSizeArg,
                               const Optional<unsigned> &NumElemsArg) {
  llvm::AttrBuilder B;
  B.addAllocSizeAttr(ElemSizeArg, NumElemsArg);
  return addAttributes(C, Index, AttributeSet::get(C, Index, B));
}

//===----------------------------------------------------------------------===//
// AttributeSet Accessor Methods
//===----------------------------------------------------------------------===//

LLVMContext &AttributeSet::getContext() const {
  return pImpl->getContext();
}

AttributeSet AttributeSet::getParamAttributes(unsigned Index) const {
  return pImpl && hasAttributes(Index) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> >(
                        std::make_pair(Index, getAttributes(Index)))) :
    AttributeSet();
}

AttributeSet AttributeSet::getRetAttributes() const {
  return pImpl && hasAttributes(ReturnIndex) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> >(
                        std::make_pair(ReturnIndex,
                                       getAttributes(ReturnIndex)))) :
    AttributeSet();
}

AttributeSet AttributeSet::getFnAttributes() const {
  return pImpl && hasAttributes(FunctionIndex) ?
    AttributeSet::get(pImpl->getContext(),
                      ArrayRef<std::pair<unsigned, AttributeSetNode*> >(
                        std::make_pair(FunctionIndex,
                                       getAttributes(FunctionIndex)))) :
    AttributeSet();
}

bool AttributeSet::hasAttribute(unsigned Index, Attribute::AttrKind Kind) const{
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN && ASN->hasAttribute(Kind);
}

bool AttributeSet::hasAttribute(unsigned Index, StringRef Kind) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN && ASN->hasAttribute(Kind);
}

bool AttributeSet::hasAttributes(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN && ASN->hasAttributes();
}

bool AttributeSet::hasFnAttribute(Attribute::AttrKind Kind) const {
  return pImpl && pImpl->hasFnAttribute(Kind);
}

bool AttributeSet::hasAttrSomewhere(Attribute::AttrKind Attr,
                                    unsigned *Index) const {
  if (!pImpl) return false;

  for (unsigned I = 0, E = pImpl->getNumSlots(); I != E; ++I)
    for (AttributeSetImpl::iterator II = pImpl->begin(I),
           IE = pImpl->end(I); II != IE; ++II)
      if (II->hasAttribute(Attr)) {
        if (Index) *Index = pImpl->getSlotIndex(I);
        return true;
      }

  return false;
}

Attribute AttributeSet::getAttribute(unsigned Index,
                                     Attribute::AttrKind Kind) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAttribute(Kind) : Attribute();
}

Attribute AttributeSet::getAttribute(unsigned Index,
                                     StringRef Kind) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAttribute(Kind) : Attribute();
}

unsigned AttributeSet::getParamAlignment(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAlignment() : 0;
}

unsigned AttributeSet::getStackAlignment(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getStackAlignment() : 0;
}

uint64_t AttributeSet::getDereferenceableBytes(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getDereferenceableBytes() : 0;
}

uint64_t AttributeSet::getDereferenceableOrNullBytes(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getDereferenceableOrNullBytes() : 0;
}

std::pair<unsigned, Optional<unsigned>>
AttributeSet::getAllocSizeArgs(unsigned Index) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAllocSizeArgs() : std::make_pair(0u, Optional<unsigned>(0u));
}

std::string AttributeSet::getAsString(unsigned Index, bool InAttrGrp) const {
  AttributeSetNode *ASN = getAttributes(Index);
  return ASN ? ASN->getAsString(InAttrGrp) : std::string("");
}

AttributeSetNode *AttributeSet::getAttributes(unsigned Index) const {
  if (!pImpl) return nullptr;

  // Loop through to find the attribute node we want.
  for (unsigned I = 0, E = pImpl->getNumSlots(); I != E; ++I)
    if (pImpl->getSlotIndex(I) == Index)
      return pImpl->getSlotNode(I);

  return nullptr;
}

AttributeSet::iterator AttributeSet::begin(unsigned Slot) const {
  if (!pImpl)
    return ArrayRef<Attribute>().begin();
  return pImpl->begin(Slot);
}

AttributeSet::iterator AttributeSet::end(unsigned Slot) const {
  if (!pImpl)
    return ArrayRef<Attribute>().end();
  return pImpl->end(Slot);
}

//===----------------------------------------------------------------------===//
// AttributeSet Introspection Methods
//===----------------------------------------------------------------------===//

unsigned AttributeSet::getNumSlots() const {
  return pImpl ? pImpl->getNumSlots() : 0;
}

unsigned AttributeSet::getSlotIndex(unsigned Slot) const {
  assert(pImpl && Slot < pImpl->getNumSlots() &&
         "Slot # out of range!");
  return pImpl->getSlotIndex(Slot);
}

AttributeSet AttributeSet::getSlotAttributes(unsigned Slot) const {
  assert(pImpl && Slot < pImpl->getNumSlots() &&
         "Slot # out of range!");
  return pImpl->getSlotAttributes(Slot);
}

uint64_t AttributeSet::Raw(unsigned Index) const {
  // FIXME: Remove this.
  return pImpl ? pImpl->Raw(Index) : 0;
}

LLVM_DUMP_METHOD void AttributeSet::dump() const {
  dbgs() << "PAL[\n";

  for (unsigned i = 0, e = getNumSlots(); i < e; ++i) {
    uint64_t Index = getSlotIndex(i);
    dbgs() << "  { ";
    if (Index == ~0U)
      dbgs() << "~0U";
    else
      dbgs() << Index;
    dbgs() << " => " << getAsString(Index) << " }\n";
  }

  dbgs() << "]\n";
}

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

AttrBuilder::AttrBuilder(AttributeSet AS, unsigned Index)
    : Attrs(0), Alignment(0), StackAlignment(0), DerefBytes(0),
      DerefOrNullBytes(0), AllocSizeArgs(0) {
  AttributeSetImpl *pImpl = AS.pImpl;
  if (!pImpl) return;

  for (unsigned I = 0, E = pImpl->getNumSlots(); I != E; ++I) {
    if (pImpl->getSlotIndex(I) != Index) continue;

    for (AttributeSetImpl::iterator II = pImpl->begin(I),
           IE = pImpl->end(I); II != IE; ++II)
      addAttribute(*II);

    break;
  }
}

void AttrBuilder::clear() {
  Attrs.reset();
  TargetDepAttrs.clear();
  Alignment = StackAlignment = DerefBytes = DerefOrNullBytes = 0;
  AllocSizeArgs = 0;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute::AttrKind Val) {
  assert((unsigned)Val < Attribute::EndAttrKinds && "Attribute out of range!");
  assert(Val != Attribute::Alignment && Val != Attribute::StackAlignment &&
         Val != Attribute::Dereferenceable && Val != Attribute::AllocSize &&
         "Adding integer attribute without adding a value!");
  Attrs[Val] = true;
  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute Attr) {
  if (Attr.isStringAttribute()) {
    addAttribute(Attr.getKindAsString(), Attr.getValueAsString());
    return *this;
  }

  Attribute::AttrKind Kind = Attr.getKindAsEnum();
  Attrs[Kind] = true;

  if (Kind == Attribute::Alignment)
    Alignment = Attr.getAlignment();
  else if (Kind == Attribute::StackAlignment)
    StackAlignment = Attr.getStackAlignment();
  else if (Kind == Attribute::Dereferenceable)
    DerefBytes = Attr.getDereferenceableBytes();
  else if (Kind == Attribute::DereferenceableOrNull)
    DerefOrNullBytes = Attr.getDereferenceableOrNullBytes();
  else if (Kind == Attribute::AllocSize)
    AllocSizeArgs = Attr.getValueAsInt();
  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(StringRef A, StringRef V) {
  TargetDepAttrs[A] = V;
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(Attribute::AttrKind Val) {
  assert((unsigned)Val < Attribute::EndAttrKinds && "Attribute out of range!");
  Attrs[Val] = false;

  if (Val == Attribute::Alignment)
    Alignment = 0;
  else if (Val == Attribute::StackAlignment)
    StackAlignment = 0;
  else if (Val == Attribute::Dereferenceable)
    DerefBytes = 0;
  else if (Val == Attribute::DereferenceableOrNull)
    DerefOrNullBytes = 0;
  else if (Val == Attribute::AllocSize)
    AllocSizeArgs = 0;

  return *this;
}

AttrBuilder &AttrBuilder::removeAttributes(AttributeSet A, uint64_t Index) {
  unsigned Slot = ~0U;
  for (unsigned I = 0, E = A.getNumSlots(); I != E; ++I)
    if (A.getSlotIndex(I) == Index) {
      Slot = I;
      break;
    }

  assert(Slot != ~0U && "Couldn't find index in AttributeSet!");

  for (AttributeSet::iterator I = A.begin(Slot), E = A.end(Slot); I != E; ++I) {
    Attribute Attr = *I;
    if (Attr.isEnumAttribute() || Attr.isIntAttribute()) {
      removeAttribute(Attr.getKindAsEnum());
    } else {
      assert(Attr.isStringAttribute() && "Invalid attribute type!");
      removeAttribute(Attr.getKindAsString());
    }
  }

  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(StringRef A) {
  std::map<std::string, std::string>::iterator I = TargetDepAttrs.find(A);
  if (I != TargetDepAttrs.end())
    TargetDepAttrs.erase(I);
  return *this;
}

std::pair<unsigned, Optional<unsigned>> AttrBuilder::getAllocSizeArgs() const {
  return unpackAllocSizeArgs(AllocSizeArgs);
}

AttrBuilder &AttrBuilder::addAlignmentAttr(unsigned Align) {
  if (Align == 0) return *this;

  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");

  Attrs[Attribute::Alignment] = true;
  Alignment = Align;
  return *this;
}

AttrBuilder &AttrBuilder::addStackAlignmentAttr(unsigned Align) {
  // Default alignment, allow the target to define how to align it.
  if (Align == 0) return *this;

  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");

  Attrs[Attribute::StackAlignment] = true;
  StackAlignment = Align;
  return *this;
}

AttrBuilder &AttrBuilder::addDereferenceableAttr(uint64_t Bytes) {
  if (Bytes == 0) return *this;

  Attrs[Attribute::Dereferenceable] = true;
  DerefBytes = Bytes;
  return *this;
}

AttrBuilder &AttrBuilder::addDereferenceableOrNullAttr(uint64_t Bytes) {
  if (Bytes == 0)
    return *this;

  Attrs[Attribute::DereferenceableOrNull] = true;
  DerefOrNullBytes = Bytes;
  return *this;
}

AttrBuilder &AttrBuilder::addAllocSizeAttr(unsigned ElemSize,
                                           const Optional<unsigned> &NumElems) {
  return addAllocSizeAttrFromRawRepr(packAllocSizeArgs(ElemSize, NumElems));
}

AttrBuilder &AttrBuilder::addAllocSizeAttrFromRawRepr(uint64_t RawArgs) {
  // (0, 0) is our "not present" value, so we need to check for it here.
  assert(RawArgs && "Invalid allocsize arguments -- given allocsize(0, 0)");

  Attrs[Attribute::AllocSize] = true;
  // Reuse existing machinery to store this as a single 64-bit integer so we can
  // save a few bytes over using a pair<unsigned, Optional<unsigned>>.
  AllocSizeArgs = RawArgs;
  return *this;
}

AttrBuilder &AttrBuilder::merge(const AttrBuilder &B) {
  // FIXME: What if both have alignments, but they don't match?!
  if (!Alignment)
    Alignment = B.Alignment;

  if (!StackAlignment)
    StackAlignment = B.StackAlignment;

  if (!DerefBytes)
    DerefBytes = B.DerefBytes;

  if (!DerefOrNullBytes)
    DerefOrNullBytes = B.DerefOrNullBytes;

  if (!AllocSizeArgs)
    AllocSizeArgs = B.AllocSizeArgs;

  Attrs |= B.Attrs;

  for (auto I : B.td_attrs())
    TargetDepAttrs[I.first] = I.second;

  return *this;
}

AttrBuilder &AttrBuilder::remove(const AttrBuilder &B) {
  // FIXME: What if both have alignments, but they don't match?!
  if (B.Alignment)
    Alignment = 0;

  if (B.StackAlignment)
    StackAlignment = 0;

  if (B.DerefBytes)
    DerefBytes = 0;

  if (B.DerefOrNullBytes)
    DerefOrNullBytes = 0;

  if (B.AllocSizeArgs)
    AllocSizeArgs = 0;

  Attrs &= ~B.Attrs;

  for (auto I : B.td_attrs())
    TargetDepAttrs.erase(I.first);

  return *this;
}

bool AttrBuilder::overlaps(const AttrBuilder &B) const {
  // First check if any of the target independent attributes overlap.
  if ((Attrs & B.Attrs).any())
    return true;

  // Then check if any target dependent ones do.
  for (auto I : td_attrs())
    if (B.contains(I.first))
      return true;

  return false;
}

bool AttrBuilder::contains(StringRef A) const {
  return TargetDepAttrs.find(A) != TargetDepAttrs.end();
}

bool AttrBuilder::hasAttributes() const {
  return !Attrs.none() || !TargetDepAttrs.empty();
}

bool AttrBuilder::hasAttributes(AttributeSet A, uint64_t Index) const {
  unsigned Slot = ~0U;
  for (unsigned I = 0, E = A.getNumSlots(); I != E; ++I)
    if (A.getSlotIndex(I) == Index) {
      Slot = I;
      break;
    }

  assert(Slot != ~0U && "Couldn't find the index!");

  for (AttributeSet::iterator I = A.begin(Slot), E = A.end(Slot); I != E; ++I) {
    Attribute Attr = *I;
    if (Attr.isEnumAttribute() || Attr.isIntAttribute()) {
      if (Attrs[I->getKindAsEnum()])
        return true;
    } else {
      assert(Attr.isStringAttribute() && "Invalid attribute kind!");
      return TargetDepAttrs.find(Attr.getKindAsString())!=TargetDepAttrs.end();
    }
  }

  return false;
}

bool AttrBuilder::hasAlignmentAttr() const {
  return Alignment != 0;
}

bool AttrBuilder::operator==(const AttrBuilder &B) {
  if (Attrs != B.Attrs)
    return false;

  for (td_const_iterator I = TargetDepAttrs.begin(),
         E = TargetDepAttrs.end(); I != E; ++I)
    if (B.TargetDepAttrs.find(I->first) == B.TargetDepAttrs.end())
      return false;

  return Alignment == B.Alignment && StackAlignment == B.StackAlignment &&
         DerefBytes == B.DerefBytes;
}

AttrBuilder &AttrBuilder::addRawValue(uint64_t Val) {
  // FIXME: Remove this in 4.0.
  if (!Val) return *this;

  for (Attribute::AttrKind I = Attribute::None; I != Attribute::EndAttrKinds;
       I = Attribute::AttrKind(I + 1)) {
    if (I == Attribute::Dereferenceable ||
        I == Attribute::DereferenceableOrNull ||
        I == Attribute::ArgMemOnly ||
        I == Attribute::AllocSize)
      continue;
    if (uint64_t A = (Val & AttributeImpl::getAttrMask(I))) {
      Attrs[I] = true;
 
      if (I == Attribute::Alignment)
        Alignment = 1ULL << ((A >> 16) - 1);
      else if (I == Attribute::StackAlignment)
        StackAlignment = 1ULL << ((A >> 26)-1);
    }
  }
 
  return *this;
}

//===----------------------------------------------------------------------===//
// AttributeFuncs Function Defintions
//===----------------------------------------------------------------------===//

/// \brief Which attributes cannot be applied to a type.
AttrBuilder AttributeFuncs::typeIncompatible(Type *Ty) {
  AttrBuilder Incompatible;

  if (!Ty->isIntegerTy())
    // Attribute that only apply to integers.
    Incompatible.addAttribute(Attribute::SExt)
      .addAttribute(Attribute::ZExt);

  if (!Ty->isPointerTy())
    // Attribute that only apply to pointers.
    Incompatible.addAttribute(Attribute::ByVal)
      .addAttribute(Attribute::Nest)
      .addAttribute(Attribute::NoAlias)
      .addAttribute(Attribute::NoCapture)
      .addAttribute(Attribute::NonNull)
      .addDereferenceableAttr(1) // the int here is ignored
      .addDereferenceableOrNullAttr(1) // the int here is ignored
      .addAttribute(Attribute::ReadNone)
      .addAttribute(Attribute::ReadOnly)
      .addAttribute(Attribute::StructRet)
      .addAttribute(Attribute::InAlloca);

  return Incompatible;
}

template<typename AttrClass>
static bool isEqual(const Function &Caller, const Function &Callee) {
  return Caller.getFnAttribute(AttrClass::getKind()) ==
         Callee.getFnAttribute(AttrClass::getKind());
}

/// \brief Compute the logical AND of the attributes of the caller and the
/// callee.
///
/// This function sets the caller's attribute to false if the callee's attribute
/// is false.
template<typename AttrClass>
static void setAND(Function &Caller, const Function &Callee) {
  if (AttrClass::isSet(Caller, AttrClass::getKind()) &&
      !AttrClass::isSet(Callee, AttrClass::getKind()))
    AttrClass::set(Caller, AttrClass::getKind(), false);
}

/// \brief Compute the logical OR of the attributes of the caller and the
/// callee.
///
/// This function sets the caller's attribute to true if the callee's attribute
/// is true.
template<typename AttrClass>
static void setOR(Function &Caller, const Function &Callee) {
  if (!AttrClass::isSet(Caller, AttrClass::getKind()) &&
      AttrClass::isSet(Callee, AttrClass::getKind()))
    AttrClass::set(Caller, AttrClass::getKind(), true);
}

/// \brief If the inlined function had a higher stack protection level than the
/// calling function, then bump up the caller's stack protection level.
static void adjustCallerSSPLevel(Function &Caller, const Function &Callee) {
  // If upgrading the SSP attribute, clear out the old SSP Attributes first.
  // Having multiple SSP attributes doesn't actually hurt, but it adds useless
  // clutter to the IR.
  AttrBuilder B;
  B.addAttribute(Attribute::StackProtect)
    .addAttribute(Attribute::StackProtectStrong)
    .addAttribute(Attribute::StackProtectReq);
  AttributeSet OldSSPAttr = AttributeSet::get(Caller.getContext(),
                                              AttributeSet::FunctionIndex,
                                              B);

  if (Callee.hasFnAttribute(Attribute::StackProtectReq)) {
    Caller.removeAttributes(AttributeSet::FunctionIndex, OldSSPAttr);
    Caller.addFnAttr(Attribute::StackProtectReq);
  } else if (Callee.hasFnAttribute(Attribute::StackProtectStrong) &&
             !Caller.hasFnAttribute(Attribute::StackProtectReq)) {
    Caller.removeAttributes(AttributeSet::FunctionIndex, OldSSPAttr);
    Caller.addFnAttr(Attribute::StackProtectStrong);
  } else if (Callee.hasFnAttribute(Attribute::StackProtect) &&
             !Caller.hasFnAttribute(Attribute::StackProtectReq) &&
             !Caller.hasFnAttribute(Attribute::StackProtectStrong))
    Caller.addFnAttr(Attribute::StackProtect);
}

#define GET_ATTR_COMPAT_FUNC
#include "AttributesCompatFunc.inc"

bool AttributeFuncs::areInlineCompatible(const Function &Caller,
                                         const Function &Callee) {
  return hasCompatibleFnAttrs(Caller, Callee);
}


void AttributeFuncs::mergeAttributesForInlining(Function &Caller,
                                                const Function &Callee) {
  mergeFnAttrs(Caller, Callee);
}
