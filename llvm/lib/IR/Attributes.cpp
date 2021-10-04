//===- Attributes.cpp - Implement AttributesList --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements the Attribute, AttributeImpl, AttrBuilder,
// AttributeListImpl, and AttributeList classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Attributes.h"
#include "AttributeImpl.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

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

static uint64_t packVScaleRangeArgs(unsigned MinValue, unsigned MaxValue) {
  return uint64_t(MinValue) << 32 | MaxValue;
}

static std::pair<unsigned, unsigned> unpackVScaleRangeArgs(uint64_t Value) {
  unsigned MaxValue = Value & std::numeric_limits<unsigned>::max();
  unsigned MinValue = Value >> 32;

  return std::make_pair(MinValue, MaxValue);
}

Attribute Attribute::get(LLVMContext &Context, Attribute::AttrKind Kind,
                         uint64_t Val) {
  if (Val)
    assert(Attribute::isIntAttrKind(Kind) && "Not an int attribute");
  else
    assert(Attribute::isEnumAttrKind(Kind) && "Not an enum attribute");

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
      PA = new (pImpl->Alloc) EnumAttributeImpl(Kind);
    else
      PA = new (pImpl->Alloc) IntAttributeImpl(Kind, Val);
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
    void *Mem =
        pImpl->Alloc.Allocate(StringAttributeImpl::totalSizeToAlloc(Kind, Val),
                              alignof(StringAttributeImpl));
    PA = new (Mem) StringAttributeImpl(Kind, Val);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the Attribute that we found or created.
  return Attribute(PA);
}

Attribute Attribute::get(LLVMContext &Context, Attribute::AttrKind Kind,
                         Type *Ty) {
  assert(Attribute::isTypeAttrKind(Kind) && "Not a type attribute");
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddInteger(Kind);
  ID.AddPointer(Ty);

  void *InsertPoint;
  AttributeImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = new (pImpl->Alloc) TypeAttributeImpl(Kind, Ty);
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the Attribute that we found or created.
  return Attribute(PA);
}

Attribute Attribute::getWithAlignment(LLVMContext &Context, Align A) {
  assert(A <= llvm::Value::MaximumAlignment && "Alignment too large.");
  return get(Context, Alignment, A.value());
}

Attribute Attribute::getWithStackAlignment(LLVMContext &Context, Align A) {
  assert(A <= 0x100 && "Alignment too large.");
  return get(Context, StackAlignment, A.value());
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

Attribute Attribute::getWithByValType(LLVMContext &Context, Type *Ty) {
  return get(Context, ByVal, Ty);
}

Attribute Attribute::getWithStructRetType(LLVMContext &Context, Type *Ty) {
  return get(Context, StructRet, Ty);
}

Attribute Attribute::getWithByRefType(LLVMContext &Context, Type *Ty) {
  return get(Context, ByRef, Ty);
}

Attribute Attribute::getWithPreallocatedType(LLVMContext &Context, Type *Ty) {
  return get(Context, Preallocated, Ty);
}

Attribute Attribute::getWithInAllocaType(LLVMContext &Context, Type *Ty) {
  return get(Context, InAlloca, Ty);
}

Attribute
Attribute::getWithAllocSizeArgs(LLVMContext &Context, unsigned ElemSizeArg,
                                const Optional<unsigned> &NumElemsArg) {
  assert(!(ElemSizeArg == 0 && NumElemsArg && *NumElemsArg == 0) &&
         "Invalid allocsize arguments -- given allocsize(0, 0)");
  return get(Context, AllocSize, packAllocSizeArgs(ElemSizeArg, NumElemsArg));
}

Attribute Attribute::getWithVScaleRangeArgs(LLVMContext &Context,
                                            unsigned MinValue,
                                            unsigned MaxValue) {
  return get(Context, VScaleRange, packVScaleRangeArgs(MinValue, MaxValue));
}

Attribute::AttrKind Attribute::getAttrKindFromName(StringRef AttrName) {
  return StringSwitch<Attribute::AttrKind>(AttrName)
#define GET_ATTR_NAMES
#define ATTRIBUTE_ENUM(ENUM_NAME, DISPLAY_NAME)                                \
  .Case(#DISPLAY_NAME, Attribute::ENUM_NAME)
#include "llvm/IR/Attributes.inc"
      .Default(Attribute::None);
}

StringRef Attribute::getNameFromAttrKind(Attribute::AttrKind AttrKind) {
  switch (AttrKind) {
#define GET_ATTR_NAMES
#define ATTRIBUTE_ENUM(ENUM_NAME, DISPLAY_NAME)                                \
  case Attribute::ENUM_NAME:                                                   \
    return #DISPLAY_NAME;
#include "llvm/IR/Attributes.inc"
  case Attribute::None:
    return "none";
  default:
    llvm_unreachable("invalid Kind");
  }
}

bool Attribute::isExistingAttribute(StringRef Name) {
  return StringSwitch<bool>(Name)
#define GET_ATTR_NAMES
#define ATTRIBUTE_ALL(ENUM_NAME, DISPLAY_NAME) .Case(#DISPLAY_NAME, true)
#include "llvm/IR/Attributes.inc"
      .Default(false);
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

bool Attribute::isTypeAttribute() const {
  return pImpl && pImpl->isTypeAttribute();
}

Attribute::AttrKind Attribute::getKindAsEnum() const {
  if (!pImpl) return None;
  assert((isEnumAttribute() || isIntAttribute() || isTypeAttribute()) &&
         "Invalid attribute type to get the kind as an enum!");
  return pImpl->getKindAsEnum();
}

uint64_t Attribute::getValueAsInt() const {
  if (!pImpl) return 0;
  assert(isIntAttribute() &&
         "Expected the attribute to be an integer attribute!");
  return pImpl->getValueAsInt();
}

bool Attribute::getValueAsBool() const {
  if (!pImpl) return false;
  assert(isStringAttribute() &&
         "Expected the attribute to be a string attribute!");
  return pImpl->getValueAsBool();
}

StringRef Attribute::getKindAsString() const {
  if (!pImpl) return {};
  assert(isStringAttribute() &&
         "Invalid attribute type to get the kind as a string!");
  return pImpl->getKindAsString();
}

StringRef Attribute::getValueAsString() const {
  if (!pImpl) return {};
  assert(isStringAttribute() &&
         "Invalid attribute type to get the value as a string!");
  return pImpl->getValueAsString();
}

Type *Attribute::getValueAsType() const {
  if (!pImpl) return {};
  assert(isTypeAttribute() &&
         "Invalid attribute type to get the value as a type!");
  return pImpl->getValueAsType();
}


bool Attribute::hasAttribute(AttrKind Kind) const {
  return (pImpl && pImpl->hasAttribute(Kind)) || (!pImpl && Kind == None);
}

bool Attribute::hasAttribute(StringRef Kind) const {
  if (!isStringAttribute()) return false;
  return pImpl && pImpl->hasAttribute(Kind);
}

MaybeAlign Attribute::getAlignment() const {
  assert(hasAttribute(Attribute::Alignment) &&
         "Trying to get alignment from non-alignment attribute!");
  return MaybeAlign(pImpl->getValueAsInt());
}

MaybeAlign Attribute::getStackAlignment() const {
  assert(hasAttribute(Attribute::StackAlignment) &&
         "Trying to get alignment from non-alignment attribute!");
  return MaybeAlign(pImpl->getValueAsInt());
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

std::pair<unsigned, unsigned> Attribute::getVScaleRangeArgs() const {
  assert(hasAttribute(Attribute::VScaleRange) &&
         "Trying to get vscale args from non-vscale attribute");
  return unpackVScaleRangeArgs(pImpl->getValueAsInt());
}

std::string Attribute::getAsString(bool InAttrGrp) const {
  if (!pImpl) return {};

  if (isEnumAttribute())
    return getNameFromAttrKind(getKindAsEnum()).str();

  if (isTypeAttribute()) {
    std::string Result = getNameFromAttrKind(getKindAsEnum()).str();
    Result += '(';
    raw_string_ostream OS(Result);
    getValueAsType()->print(OS, false, true);
    OS.flush();
    Result += ')';
    return Result;
  }

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

  if (hasAttribute(Attribute::VScaleRange)) {
    unsigned MinValue, MaxValue;
    std::tie(MinValue, MaxValue) = getVScaleRangeArgs();

    std::string Result = "vscale_range(";
    Result += utostr(MinValue);
    Result += ',';
    Result += utostr(MaxValue);
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
    {
      raw_string_ostream OS(Result);
      OS << '"' << getKindAsString() << '"';

      // Since some attribute strings contain special characters that cannot be
      // printable, those have to be escaped to make the attribute value
      // printable as is.  e.g. "\01__gnu_mcount_nc"
      const auto &AttrVal = pImpl->getValueAsString();
      if (!AttrVal.empty()) {
        OS << "=\"";
        printEscapedString(AttrVal, OS);
        OS << "\"";
      }
    }
    return Result;
  }

  llvm_unreachable("Unknown attribute");
}

bool Attribute::hasParentContext(LLVMContext &C) const {
  assert(isValid() && "invalid Attribute doesn't refer to any context");
  FoldingSetNodeID ID;
  pImpl->Profile(ID);
  void *Unused;
  return C.pImpl->AttrsSet.FindNodeOrInsertPos(ID, Unused) == pImpl;
}

bool Attribute::operator<(Attribute A) const {
  if (!pImpl && !A.pImpl) return false;
  if (!pImpl) return true;
  if (!A.pImpl) return false;
  return *pImpl < *A.pImpl;
}

void Attribute::Profile(FoldingSetNodeID &ID) const {
  ID.AddPointer(pImpl);
}

enum AttributeProperty {
  FnAttr = (1 << 0),
  ParamAttr = (1 << 1),
  RetAttr = (1 << 2),
};

#define GET_ATTR_PROP_TABLE
#include "llvm/IR/Attributes.inc"

static bool hasAttributeProperty(Attribute::AttrKind Kind,
                                 AttributeProperty Prop) {
  unsigned Index = Kind - 1;
  assert(Index < sizeof(AttrPropTable) / sizeof(AttrPropTable[0]) &&
         "Invalid attribute kind");
  return AttrPropTable[Index] & Prop;
}

bool Attribute::canUseAsFnAttr(AttrKind Kind) {
  return hasAttributeProperty(Kind, AttributeProperty::FnAttr);
}

bool Attribute::canUseAsParamAttr(AttrKind Kind) {
  return hasAttributeProperty(Kind, AttributeProperty::ParamAttr);
}

bool Attribute::canUseAsRetAttr(AttrKind Kind) {
  return hasAttributeProperty(Kind, AttributeProperty::RetAttr);
}

//===----------------------------------------------------------------------===//
// AttributeImpl Definition
//===----------------------------------------------------------------------===//

bool AttributeImpl::hasAttribute(Attribute::AttrKind A) const {
  if (isStringAttribute()) return false;
  return getKindAsEnum() == A;
}

bool AttributeImpl::hasAttribute(StringRef Kind) const {
  if (!isStringAttribute()) return false;
  return getKindAsString() == Kind;
}

Attribute::AttrKind AttributeImpl::getKindAsEnum() const {
  assert(isEnumAttribute() || isIntAttribute() || isTypeAttribute());
  return static_cast<const EnumAttributeImpl *>(this)->getEnumKind();
}

uint64_t AttributeImpl::getValueAsInt() const {
  assert(isIntAttribute());
  return static_cast<const IntAttributeImpl *>(this)->getValue();
}

bool AttributeImpl::getValueAsBool() const {
  assert(getValueAsString().empty() || getValueAsString() == "false" || getValueAsString() == "true");
  return getValueAsString() == "true";
}

StringRef AttributeImpl::getKindAsString() const {
  assert(isStringAttribute());
  return static_cast<const StringAttributeImpl *>(this)->getStringKind();
}

StringRef AttributeImpl::getValueAsString() const {
  assert(isStringAttribute());
  return static_cast<const StringAttributeImpl *>(this)->getStringValue();
}

Type *AttributeImpl::getValueAsType() const {
  assert(isTypeAttribute());
  return static_cast<const TypeAttributeImpl *>(this)->getTypeValue();
}

bool AttributeImpl::operator<(const AttributeImpl &AI) const {
  if (this == &AI)
    return false;

  // This sorts the attributes with Attribute::AttrKinds coming first (sorted
  // relative to their enum value) and then strings.
  if (!isStringAttribute()) {
    if (AI.isStringAttribute())
      return true;
    if (getKindAsEnum() != AI.getKindAsEnum())
      return getKindAsEnum() < AI.getKindAsEnum();
    assert(!AI.isEnumAttribute() && "Non-unique attribute");
    assert(!AI.isTypeAttribute() && "Comparison of types would be unstable");
    // TODO: Is this actually needed?
    assert(AI.isIntAttribute() && "Only possibility left");
    return getValueAsInt() < AI.getValueAsInt();
  }

  if (!AI.isStringAttribute())
    return false;
  if (getKindAsString() == AI.getKindAsString())
    return getValueAsString() < AI.getValueAsString();
  return getKindAsString() < AI.getKindAsString();
}

//===----------------------------------------------------------------------===//
// AttributeSet Definition
//===----------------------------------------------------------------------===//

AttributeSet AttributeSet::get(LLVMContext &C, const AttrBuilder &B) {
  return AttributeSet(AttributeSetNode::get(C, B));
}

AttributeSet AttributeSet::get(LLVMContext &C, ArrayRef<Attribute> Attrs) {
  return AttributeSet(AttributeSetNode::get(C, Attrs));
}

AttributeSet AttributeSet::addAttribute(LLVMContext &C,
                                        Attribute::AttrKind Kind) const {
  if (hasAttribute(Kind)) return *this;
  AttrBuilder B;
  B.addAttribute(Kind);
  return addAttributes(C, AttributeSet::get(C, B));
}

AttributeSet AttributeSet::addAttribute(LLVMContext &C, StringRef Kind,
                                        StringRef Value) const {
  AttrBuilder B;
  B.addAttribute(Kind, Value);
  return addAttributes(C, AttributeSet::get(C, B));
}

AttributeSet AttributeSet::addAttributes(LLVMContext &C,
                                         const AttributeSet AS) const {
  if (!hasAttributes())
    return AS;

  if (!AS.hasAttributes())
    return *this;

  AttrBuilder B(AS);
  for (const auto &I : *this)
    B.addAttribute(I);

 return get(C, B);
}

AttributeSet AttributeSet::removeAttribute(LLVMContext &C,
                                             Attribute::AttrKind Kind) const {
  if (!hasAttribute(Kind)) return *this;
  AttrBuilder B(*this);
  B.removeAttribute(Kind);
  return get(C, B);
}

AttributeSet AttributeSet::removeAttribute(LLVMContext &C,
                                             StringRef Kind) const {
  if (!hasAttribute(Kind)) return *this;
  AttrBuilder B(*this);
  B.removeAttribute(Kind);
  return get(C, B);
}

AttributeSet AttributeSet::removeAttributes(LLVMContext &C,
                                            const AttrBuilder &Attrs) const {
  AttrBuilder B(*this);
  // If there is nothing to remove, directly return the original set.
  if (!B.overlaps(Attrs))
    return *this;

  B.remove(Attrs);
  return get(C, B);
}

unsigned AttributeSet::getNumAttributes() const {
  return SetNode ? SetNode->getNumAttributes() : 0;
}

bool AttributeSet::hasAttribute(Attribute::AttrKind Kind) const {
  return SetNode ? SetNode->hasAttribute(Kind) : false;
}

bool AttributeSet::hasAttribute(StringRef Kind) const {
  return SetNode ? SetNode->hasAttribute(Kind) : false;
}

Attribute AttributeSet::getAttribute(Attribute::AttrKind Kind) const {
  return SetNode ? SetNode->getAttribute(Kind) : Attribute();
}

Attribute AttributeSet::getAttribute(StringRef Kind) const {
  return SetNode ? SetNode->getAttribute(Kind) : Attribute();
}

MaybeAlign AttributeSet::getAlignment() const {
  return SetNode ? SetNode->getAlignment() : None;
}

MaybeAlign AttributeSet::getStackAlignment() const {
  return SetNode ? SetNode->getStackAlignment() : None;
}

uint64_t AttributeSet::getDereferenceableBytes() const {
  return SetNode ? SetNode->getDereferenceableBytes() : 0;
}

uint64_t AttributeSet::getDereferenceableOrNullBytes() const {
  return SetNode ? SetNode->getDereferenceableOrNullBytes() : 0;
}

Type *AttributeSet::getByRefType() const {
  return SetNode ? SetNode->getAttributeType(Attribute::ByRef) : nullptr;
}

Type *AttributeSet::getByValType() const {
  return SetNode ? SetNode->getAttributeType(Attribute::ByVal) : nullptr;
}

Type *AttributeSet::getStructRetType() const {
  return SetNode ? SetNode->getAttributeType(Attribute::StructRet) : nullptr;
}

Type *AttributeSet::getPreallocatedType() const {
  return SetNode ? SetNode->getAttributeType(Attribute::Preallocated) : nullptr;
}

Type *AttributeSet::getInAllocaType() const {
  return SetNode ? SetNode->getAttributeType(Attribute::InAlloca) : nullptr;
}

Type *AttributeSet::getElementType() const {
  return SetNode ? SetNode->getAttributeType(Attribute::ElementType) : nullptr;
}

std::pair<unsigned, Optional<unsigned>> AttributeSet::getAllocSizeArgs() const {
  return SetNode ? SetNode->getAllocSizeArgs()
                 : std::pair<unsigned, Optional<unsigned>>(0, 0);
}

std::pair<unsigned, unsigned> AttributeSet::getVScaleRangeArgs() const {
  return SetNode ? SetNode->getVScaleRangeArgs()
                 : std::pair<unsigned, unsigned>(0, 0);
}

std::string AttributeSet::getAsString(bool InAttrGrp) const {
  return SetNode ? SetNode->getAsString(InAttrGrp) : "";
}

bool AttributeSet::hasParentContext(LLVMContext &C) const {
  assert(hasAttributes() && "empty AttributeSet doesn't refer to any context");
  FoldingSetNodeID ID;
  SetNode->Profile(ID);
  void *Unused;
  return C.pImpl->AttrsSetNodes.FindNodeOrInsertPos(ID, Unused) == SetNode;
}

AttributeSet::iterator AttributeSet::begin() const {
  return SetNode ? SetNode->begin() : nullptr;
}

AttributeSet::iterator AttributeSet::end() const {
  return SetNode ? SetNode->end() : nullptr;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void AttributeSet::dump() const {
  dbgs() << "AS =\n";
    dbgs() << "  { ";
    dbgs() << getAsString(true) << " }\n";
}
#endif

//===----------------------------------------------------------------------===//
// AttributeSetNode Definition
//===----------------------------------------------------------------------===//

AttributeSetNode::AttributeSetNode(ArrayRef<Attribute> Attrs)
    : NumAttrs(Attrs.size()) {
  // There's memory after the node where we can store the entries in.
  llvm::copy(Attrs, getTrailingObjects<Attribute>());

  for (const auto &I : *this) {
    if (I.isStringAttribute())
      StringAttrs.insert({ I.getKindAsString(), I });
    else
      AvailableAttrs.addAttribute(I.getKindAsEnum());
  }
}

AttributeSetNode *AttributeSetNode::get(LLVMContext &C,
                                        ArrayRef<Attribute> Attrs) {
  SmallVector<Attribute, 8> SortedAttrs(Attrs.begin(), Attrs.end());
  llvm::sort(SortedAttrs);
  return getSorted(C, SortedAttrs);
}

AttributeSetNode *AttributeSetNode::getSorted(LLVMContext &C,
                                              ArrayRef<Attribute> SortedAttrs) {
  if (SortedAttrs.empty())
    return nullptr;

  // Build a key to look up the existing attributes.
  LLVMContextImpl *pImpl = C.pImpl;
  FoldingSetNodeID ID;

  assert(llvm::is_sorted(SortedAttrs) && "Expected sorted attributes!");
  for (const auto &Attr : SortedAttrs)
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

  // Return the AttributeSetNode that we found or created.
  return PA;
}

AttributeSetNode *AttributeSetNode::get(LLVMContext &C, const AttrBuilder &B) {
  // Add target-independent attributes.
  SmallVector<Attribute, 8> Attrs;
  for (Attribute::AttrKind Kind = Attribute::None;
       Kind != Attribute::EndAttrKinds; Kind = Attribute::AttrKind(Kind + 1)) {
    if (!B.contains(Kind))
      continue;

    Attribute Attr;
    if (Attribute::isTypeAttrKind(Kind))
      Attr = Attribute::get(C, Kind, B.getTypeAttr(Kind));
    else if (Attribute::isIntAttrKind(Kind))
      Attr = Attribute::get(C, Kind, B.getRawIntAttr(Kind));
    else
      Attr = Attribute::get(C, Kind);
    Attrs.push_back(Attr);
  }

  // Add target-dependent (string) attributes.
  for (const auto &TDA : B.td_attrs())
    Attrs.emplace_back(Attribute::get(C, TDA.first, TDA.second));

  return getSorted(C, Attrs);
}

bool AttributeSetNode::hasAttribute(StringRef Kind) const {
  return StringAttrs.count(Kind);
}

Optional<Attribute>
AttributeSetNode::findEnumAttribute(Attribute::AttrKind Kind) const {
  // Do a quick presence check.
  if (!hasAttribute(Kind))
    return None;

  // Attributes in a set are sorted by enum value, followed by string
  // attributes. Binary search the one we want.
  const Attribute *I =
      std::lower_bound(begin(), end() - StringAttrs.size(), Kind,
                       [](Attribute A, Attribute::AttrKind Kind) {
                         return A.getKindAsEnum() < Kind;
                       });
  assert(I != end() && I->hasAttribute(Kind) && "Presence check failed?");
  return *I;
}

Attribute AttributeSetNode::getAttribute(Attribute::AttrKind Kind) const {
  if (auto A = findEnumAttribute(Kind))
    return *A;
  return {};
}

Attribute AttributeSetNode::getAttribute(StringRef Kind) const {
  return StringAttrs.lookup(Kind);
}

MaybeAlign AttributeSetNode::getAlignment() const {
  if (auto A = findEnumAttribute(Attribute::Alignment))
    return A->getAlignment();
  return None;
}

MaybeAlign AttributeSetNode::getStackAlignment() const {
  if (auto A = findEnumAttribute(Attribute::StackAlignment))
    return A->getStackAlignment();
  return None;
}

Type *AttributeSetNode::getAttributeType(Attribute::AttrKind Kind) const {
  if (auto A = findEnumAttribute(Kind))
    return A->getValueAsType();
  return nullptr;
}

uint64_t AttributeSetNode::getDereferenceableBytes() const {
  if (auto A = findEnumAttribute(Attribute::Dereferenceable))
    return A->getDereferenceableBytes();
  return 0;
}

uint64_t AttributeSetNode::getDereferenceableOrNullBytes() const {
  if (auto A = findEnumAttribute(Attribute::DereferenceableOrNull))
    return A->getDereferenceableOrNullBytes();
  return 0;
}

std::pair<unsigned, Optional<unsigned>>
AttributeSetNode::getAllocSizeArgs() const {
  if (auto A = findEnumAttribute(Attribute::AllocSize))
    return A->getAllocSizeArgs();
  return std::make_pair(0, 0);
}

std::pair<unsigned, unsigned> AttributeSetNode::getVScaleRangeArgs() const {
  if (auto A = findEnumAttribute(Attribute::VScaleRange))
    return A->getVScaleRangeArgs();
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
// AttributeListImpl Definition
//===----------------------------------------------------------------------===//

/// Map from AttributeList index to the internal array index. Adding one happens
/// to work, because -1 wraps around to 0.
static unsigned attrIdxToArrayIdx(unsigned Index) {
  return Index + 1;
}

AttributeListImpl::AttributeListImpl(ArrayRef<AttributeSet> Sets)
    : NumAttrSets(Sets.size()) {
  assert(!Sets.empty() && "pointless AttributeListImpl");

  // There's memory after the node where we can store the entries in.
  llvm::copy(Sets, getTrailingObjects<AttributeSet>());

  // Initialize AvailableFunctionAttrs and AvailableSomewhereAttrs
  // summary bitsets.
  for (const auto &I : Sets[attrIdxToArrayIdx(AttributeList::FunctionIndex)])
    if (!I.isStringAttribute())
      AvailableFunctionAttrs.addAttribute(I.getKindAsEnum());

  for (const auto &Set : Sets)
    for (const auto &I : Set)
      if (!I.isStringAttribute())
        AvailableSomewhereAttrs.addAttribute(I.getKindAsEnum());
}

void AttributeListImpl::Profile(FoldingSetNodeID &ID) const {
  Profile(ID, makeArrayRef(begin(), end()));
}

void AttributeListImpl::Profile(FoldingSetNodeID &ID,
                                ArrayRef<AttributeSet> Sets) {
  for (const auto &Set : Sets)
    ID.AddPointer(Set.SetNode);
}

bool AttributeListImpl::hasAttrSomewhere(Attribute::AttrKind Kind,
                                        unsigned *Index) const {
  if (!AvailableSomewhereAttrs.hasAttribute(Kind))
    return false;

  if (Index) {
    for (unsigned I = 0, E = NumAttrSets; I != E; ++I) {
      if (begin()[I].hasAttribute(Kind)) {
        *Index = I - 1;
        break;
      }
    }
  }

  return true;
}


#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void AttributeListImpl::dump() const {
  AttributeList(const_cast<AttributeListImpl *>(this)).dump();
}
#endif

//===----------------------------------------------------------------------===//
// AttributeList Construction and Mutation Methods
//===----------------------------------------------------------------------===//

AttributeList AttributeList::getImpl(LLVMContext &C,
                                     ArrayRef<AttributeSet> AttrSets) {
  assert(!AttrSets.empty() && "pointless AttributeListImpl");

  LLVMContextImpl *pImpl = C.pImpl;
  FoldingSetNodeID ID;
  AttributeListImpl::Profile(ID, AttrSets);

  void *InsertPoint;
  AttributeListImpl *PA =
      pImpl->AttrsLists.FindNodeOrInsertPos(ID, InsertPoint);

  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PA) {
    // Coallocate entries after the AttributeListImpl itself.
    void *Mem = pImpl->Alloc.Allocate(
        AttributeListImpl::totalSizeToAlloc<AttributeSet>(AttrSets.size()),
        alignof(AttributeListImpl));
    PA = new (Mem) AttributeListImpl(AttrSets);
    pImpl->AttrsLists.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return AttributeList(PA);
}

AttributeList
AttributeList::get(LLVMContext &C,
                   ArrayRef<std::pair<unsigned, Attribute>> Attrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return {};

  assert(llvm::is_sorted(Attrs,
                         [](const std::pair<unsigned, Attribute> &LHS,
                            const std::pair<unsigned, Attribute> &RHS) {
                           return LHS.first < RHS.first;
                         }) &&
         "Misordered Attributes list!");
  assert(llvm::all_of(Attrs,
                      [](const std::pair<unsigned, Attribute> &Pair) {
                        return Pair.second.isValid();
                      }) &&
         "Pointless attribute!");

  // Create a vector if (unsigned, AttributeSetNode*) pairs from the attributes
  // list.
  SmallVector<std::pair<unsigned, AttributeSet>, 8> AttrPairVec;
  for (ArrayRef<std::pair<unsigned, Attribute>>::iterator I = Attrs.begin(),
         E = Attrs.end(); I != E; ) {
    unsigned Index = I->first;
    SmallVector<Attribute, 4> AttrVec;
    while (I != E && I->first == Index) {
      AttrVec.push_back(I->second);
      ++I;
    }

    AttrPairVec.emplace_back(Index, AttributeSet::get(C, AttrVec));
  }

  return get(C, AttrPairVec);
}

AttributeList
AttributeList::get(LLVMContext &C,
                   ArrayRef<std::pair<unsigned, AttributeSet>> Attrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return {};

  assert(llvm::is_sorted(Attrs,
                         [](const std::pair<unsigned, AttributeSet> &LHS,
                            const std::pair<unsigned, AttributeSet> &RHS) {
                           return LHS.first < RHS.first;
                         }) &&
         "Misordered Attributes list!");
  assert(llvm::none_of(Attrs,
                       [](const std::pair<unsigned, AttributeSet> &Pair) {
                         return !Pair.second.hasAttributes();
                       }) &&
         "Pointless attribute!");

  unsigned MaxIndex = Attrs.back().first;
  // If the MaxIndex is FunctionIndex and there are other indices in front
  // of it, we need to use the largest of those to get the right size.
  if (MaxIndex == FunctionIndex && Attrs.size() > 1)
    MaxIndex = Attrs[Attrs.size() - 2].first;

  SmallVector<AttributeSet, 4> AttrVec(attrIdxToArrayIdx(MaxIndex) + 1);
  for (const auto &Pair : Attrs)
    AttrVec[attrIdxToArrayIdx(Pair.first)] = Pair.second;

  return getImpl(C, AttrVec);
}

AttributeList AttributeList::get(LLVMContext &C, AttributeSet FnAttrs,
                                 AttributeSet RetAttrs,
                                 ArrayRef<AttributeSet> ArgAttrs) {
  // Scan from the end to find the last argument with attributes.  Most
  // arguments don't have attributes, so it's nice if we can have fewer unique
  // AttributeListImpls by dropping empty attribute sets at the end of the list.
  unsigned NumSets = 0;
  for (size_t I = ArgAttrs.size(); I != 0; --I) {
    if (ArgAttrs[I - 1].hasAttributes()) {
      NumSets = I + 2;
      break;
    }
  }
  if (NumSets == 0) {
    // Check function and return attributes if we didn't have argument
    // attributes.
    if (RetAttrs.hasAttributes())
      NumSets = 2;
    else if (FnAttrs.hasAttributes())
      NumSets = 1;
  }

  // If all attribute sets were empty, we can use the empty attribute list.
  if (NumSets == 0)
    return {};

  SmallVector<AttributeSet, 8> AttrSets;
  AttrSets.reserve(NumSets);
  // If we have any attributes, we always have function attributes.
  AttrSets.push_back(FnAttrs);
  if (NumSets > 1)
    AttrSets.push_back(RetAttrs);
  if (NumSets > 2) {
    // Drop the empty argument attribute sets at the end.
    ArgAttrs = ArgAttrs.take_front(NumSets - 2);
    llvm::append_range(AttrSets, ArgAttrs);
  }

  return getImpl(C, AttrSets);
}

AttributeList AttributeList::get(LLVMContext &C, unsigned Index,
                                 const AttrBuilder &B) {
  if (!B.hasAttributes())
    return {};
  Index = attrIdxToArrayIdx(Index);
  SmallVector<AttributeSet, 8> AttrSets(Index + 1);
  AttrSets[Index] = AttributeSet::get(C, B);
  return getImpl(C, AttrSets);
}

AttributeList AttributeList::get(LLVMContext &C, unsigned Index,
                                 ArrayRef<Attribute::AttrKind> Kinds) {
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (const auto K : Kinds)
    Attrs.emplace_back(Index, Attribute::get(C, K));
  return get(C, Attrs);
}

AttributeList AttributeList::get(LLVMContext &C, unsigned Index,
                                 ArrayRef<Attribute::AttrKind> Kinds,
                                 ArrayRef<uint64_t> Values) {
  assert(Kinds.size() == Values.size() && "Mismatched attribute values.");
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  auto VI = Values.begin();
  for (const auto K : Kinds)
    Attrs.emplace_back(Index, Attribute::get(C, K, *VI++));
  return get(C, Attrs);
}

AttributeList AttributeList::get(LLVMContext &C, unsigned Index,
                                 ArrayRef<StringRef> Kinds) {
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (const auto &K : Kinds)
    Attrs.emplace_back(Index, Attribute::get(C, K));
  return get(C, Attrs);
}

AttributeList AttributeList::get(LLVMContext &C,
                                 ArrayRef<AttributeList> Attrs) {
  if (Attrs.empty())
    return {};
  if (Attrs.size() == 1)
    return Attrs[0];

  unsigned MaxSize = 0;
  for (const auto &List : Attrs)
    MaxSize = std::max(MaxSize, List.getNumAttrSets());

  // If every list was empty, there is no point in merging the lists.
  if (MaxSize == 0)
    return {};

  SmallVector<AttributeSet, 8> NewAttrSets(MaxSize);
  for (unsigned I = 0; I < MaxSize; ++I) {
    AttrBuilder CurBuilder;
    for (const auto &List : Attrs)
      CurBuilder.merge(List.getAttributes(I - 1));
    NewAttrSets[I] = AttributeSet::get(C, CurBuilder);
  }

  return getImpl(C, NewAttrSets);
}

AttributeList
AttributeList::addAttributeAtIndex(LLVMContext &C, unsigned Index,
                                   Attribute::AttrKind Kind) const {
  if (hasAttributeAtIndex(Index, Kind))
    return *this;
  AttributeSet Attrs = getAttributes(Index);
  // TODO: Insert at correct position and avoid sort.
  SmallVector<Attribute, 8> NewAttrs(Attrs.begin(), Attrs.end());
  NewAttrs.push_back(Attribute::get(C, Kind));
  return setAttributesAtIndex(C, Index, AttributeSet::get(C, NewAttrs));
}

AttributeList AttributeList::addAttributeAtIndex(LLVMContext &C, unsigned Index,
                                                 StringRef Kind,
                                                 StringRef Value) const {
  AttrBuilder B;
  B.addAttribute(Kind, Value);
  return addAttributesAtIndex(C, Index, B);
}

AttributeList AttributeList::addAttributeAtIndex(LLVMContext &C, unsigned Index,
                                                 Attribute A) const {
  AttrBuilder B;
  B.addAttribute(A);
  return addAttributesAtIndex(C, Index, B);
}

AttributeList AttributeList::setAttributesAtIndex(LLVMContext &C,
                                                  unsigned Index,
                                                  AttributeSet Attrs) const {
  Index = attrIdxToArrayIdx(Index);
  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  if (Index >= AttrSets.size())
    AttrSets.resize(Index + 1);
  AttrSets[Index] = Attrs;
  return AttributeList::getImpl(C, AttrSets);
}

AttributeList AttributeList::addAttributesAtIndex(LLVMContext &C,
                                                  unsigned Index,
                                                  const AttrBuilder &B) const {
  if (!B.hasAttributes())
    return *this;

  if (!pImpl)
    return AttributeList::get(C, {{Index, AttributeSet::get(C, B)}});

#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment. For now, say
  // we can't change a known alignment.
  const MaybeAlign OldAlign = getAttributes(Index).getAlignment();
  const MaybeAlign NewAlign = B.getAlignment();
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  AttrBuilder Merged(getAttributes(Index));
  Merged.merge(B);
  return setAttributesAtIndex(C, Index, AttributeSet::get(C, Merged));
}

AttributeList AttributeList::addParamAttribute(LLVMContext &C,
                                               ArrayRef<unsigned> ArgNos,
                                               Attribute A) const {
  assert(llvm::is_sorted(ArgNos));

  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  unsigned MaxIndex = attrIdxToArrayIdx(ArgNos.back() + FirstArgIndex);
  if (MaxIndex >= AttrSets.size())
    AttrSets.resize(MaxIndex + 1);

  for (unsigned ArgNo : ArgNos) {
    unsigned Index = attrIdxToArrayIdx(ArgNo + FirstArgIndex);
    AttrBuilder B(AttrSets[Index]);
    B.addAttribute(A);
    AttrSets[Index] = AttributeSet::get(C, B);
  }

  return getImpl(C, AttrSets);
}

AttributeList
AttributeList::removeAttributeAtIndex(LLVMContext &C, unsigned Index,
                                      Attribute::AttrKind Kind) const {
  if (!hasAttributeAtIndex(Index, Kind))
    return *this;

  Index = attrIdxToArrayIdx(Index);
  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  assert(Index < AttrSets.size());

  AttrSets[Index] = AttrSets[Index].removeAttribute(C, Kind);

  return getImpl(C, AttrSets);
}

AttributeList AttributeList::removeAttributeAtIndex(LLVMContext &C,
                                                    unsigned Index,
                                                    StringRef Kind) const {
  if (!hasAttributeAtIndex(Index, Kind))
    return *this;

  Index = attrIdxToArrayIdx(Index);
  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  assert(Index < AttrSets.size());

  AttrSets[Index] = AttrSets[Index].removeAttribute(C, Kind);

  return getImpl(C, AttrSets);
}

AttributeList
AttributeList::removeAttributesAtIndex(LLVMContext &C, unsigned Index,
                                       const AttrBuilder &AttrsToRemove) const {
  AttributeSet Attrs = getAttributes(Index);
  AttributeSet NewAttrs = Attrs.removeAttributes(C, AttrsToRemove);
  // If nothing was removed, return the original list.
  if (Attrs == NewAttrs)
    return *this;
  return setAttributesAtIndex(C, Index, NewAttrs);
}

AttributeList
AttributeList::removeAttributesAtIndex(LLVMContext &C,
                                       unsigned WithoutIndex) const {
  if (!pImpl)
    return {};
  WithoutIndex = attrIdxToArrayIdx(WithoutIndex);
  if (WithoutIndex >= getNumAttrSets())
    return *this;
  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  AttrSets[WithoutIndex] = AttributeSet();
  return getImpl(C, AttrSets);
}

AttributeList AttributeList::addDereferenceableRetAttr(LLVMContext &C,
                                                       uint64_t Bytes) const {
  AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
  return addRetAttributes(C, B);
}

AttributeList AttributeList::addDereferenceableParamAttr(LLVMContext &C,
                                                         unsigned Index,
                                                         uint64_t Bytes) const {
  AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
  return addParamAttributes(C, Index, B);
}

AttributeList
AttributeList::addDereferenceableOrNullParamAttr(LLVMContext &C, unsigned Index,
                                                 uint64_t Bytes) const {
  AttrBuilder B;
  B.addDereferenceableOrNullAttr(Bytes);
  return addParamAttributes(C, Index, B);
}

AttributeList
AttributeList::addAllocSizeParamAttr(LLVMContext &C, unsigned Index,
                                     unsigned ElemSizeArg,
                                     const Optional<unsigned> &NumElemsArg) {
  AttrBuilder B;
  B.addAllocSizeAttr(ElemSizeArg, NumElemsArg);
  return addParamAttributes(C, Index, B);
}

//===----------------------------------------------------------------------===//
// AttributeList Accessor Methods
//===----------------------------------------------------------------------===//

AttributeSet AttributeList::getParamAttrs(unsigned ArgNo) const {
  return getAttributes(ArgNo + FirstArgIndex);
}

AttributeSet AttributeList::getRetAttrs() const {
  return getAttributes(ReturnIndex);
}

AttributeSet AttributeList::getFnAttrs() const {
  return getAttributes(FunctionIndex);
}

bool AttributeList::hasAttributeAtIndex(unsigned Index,
                                        Attribute::AttrKind Kind) const {
  return getAttributes(Index).hasAttribute(Kind);
}

bool AttributeList::hasAttributeAtIndex(unsigned Index, StringRef Kind) const {
  return getAttributes(Index).hasAttribute(Kind);
}

bool AttributeList::hasAttributesAtIndex(unsigned Index) const {
  return getAttributes(Index).hasAttributes();
}

bool AttributeList::hasFnAttr(Attribute::AttrKind Kind) const {
  return pImpl && pImpl->hasFnAttribute(Kind);
}

bool AttributeList::hasFnAttr(StringRef Kind) const {
  return hasAttributeAtIndex(AttributeList::FunctionIndex, Kind);
}

bool AttributeList::hasAttrSomewhere(Attribute::AttrKind Attr,
                                     unsigned *Index) const {
  return pImpl && pImpl->hasAttrSomewhere(Attr, Index);
}

Attribute AttributeList::getAttributeAtIndex(unsigned Index,
                                             Attribute::AttrKind Kind) const {
  return getAttributes(Index).getAttribute(Kind);
}

Attribute AttributeList::getAttributeAtIndex(unsigned Index,
                                             StringRef Kind) const {
  return getAttributes(Index).getAttribute(Kind);
}

MaybeAlign AttributeList::getRetAlignment() const {
  return getAttributes(ReturnIndex).getAlignment();
}

MaybeAlign AttributeList::getParamAlignment(unsigned ArgNo) const {
  return getAttributes(ArgNo + FirstArgIndex).getAlignment();
}

MaybeAlign AttributeList::getParamStackAlignment(unsigned ArgNo) const {
  return getAttributes(ArgNo + FirstArgIndex).getStackAlignment();
}

Type *AttributeList::getParamByValType(unsigned Index) const {
  return getAttributes(Index+FirstArgIndex).getByValType();
}

Type *AttributeList::getParamStructRetType(unsigned Index) const {
  return getAttributes(Index + FirstArgIndex).getStructRetType();
}

Type *AttributeList::getParamByRefType(unsigned Index) const {
  return getAttributes(Index + FirstArgIndex).getByRefType();
}

Type *AttributeList::getParamPreallocatedType(unsigned Index) const {
  return getAttributes(Index + FirstArgIndex).getPreallocatedType();
}

Type *AttributeList::getParamInAllocaType(unsigned Index) const {
  return getAttributes(Index + FirstArgIndex).getInAllocaType();
}

Type *AttributeList::getParamElementType(unsigned Index) const {
  return getAttributes(Index + FirstArgIndex).getElementType();
}

MaybeAlign AttributeList::getFnStackAlignment() const {
  return getFnAttrs().getStackAlignment();
}

MaybeAlign AttributeList::getRetStackAlignment() const {
  return getRetAttrs().getStackAlignment();
}

uint64_t AttributeList::getRetDereferenceableBytes() const {
  return getRetAttrs().getDereferenceableBytes();
}

uint64_t AttributeList::getParamDereferenceableBytes(unsigned Index) const {
  return getParamAttrs(Index).getDereferenceableBytes();
}

uint64_t AttributeList::getRetDereferenceableOrNullBytes() const {
  return getRetAttrs().getDereferenceableOrNullBytes();
}

uint64_t
AttributeList::getParamDereferenceableOrNullBytes(unsigned Index) const {
  return getParamAttrs(Index).getDereferenceableOrNullBytes();
}

std::string AttributeList::getAsString(unsigned Index, bool InAttrGrp) const {
  return getAttributes(Index).getAsString(InAttrGrp);
}

AttributeSet AttributeList::getAttributes(unsigned Index) const {
  Index = attrIdxToArrayIdx(Index);
  if (!pImpl || Index >= getNumAttrSets())
    return {};
  return pImpl->begin()[Index];
}

bool AttributeList::hasParentContext(LLVMContext &C) const {
  assert(!isEmpty() && "an empty attribute list has no parent context");
  FoldingSetNodeID ID;
  pImpl->Profile(ID);
  void *Unused;
  return C.pImpl->AttrsLists.FindNodeOrInsertPos(ID, Unused) == pImpl;
}

AttributeList::iterator AttributeList::begin() const {
  return pImpl ? pImpl->begin() : nullptr;
}

AttributeList::iterator AttributeList::end() const {
  return pImpl ? pImpl->end() : nullptr;
}

//===----------------------------------------------------------------------===//
// AttributeList Introspection Methods
//===----------------------------------------------------------------------===//

unsigned AttributeList::getNumAttrSets() const {
  return pImpl ? pImpl->NumAttrSets : 0;
}

void AttributeList::print(raw_ostream &O) const {
  O << "AttributeList[\n";

  for (unsigned i : indexes()) {
    if (!getAttributes(i).hasAttributes())
      continue;
    O << "  { ";
    switch (i) {
    case AttrIndex::ReturnIndex:
      O << "return";
      break;
    case AttrIndex::FunctionIndex:
      O << "function";
      break;
    default:
      O << "arg(" << i - AttrIndex::FirstArgIndex << ")";
    }
    O << " => " << getAsString(i) << " }\n";
  }

  O << "]\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void AttributeList::dump() const { print(dbgs()); }
#endif

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

// FIXME: Remove this ctor, use AttributeSet.
AttrBuilder::AttrBuilder(AttributeList AL, unsigned Index) {
  AttributeSet AS = AL.getAttributes(Index);
  for (const auto &A : AS)
    addAttribute(A);
}

AttrBuilder::AttrBuilder(AttributeSet AS) {
  for (const auto &A : AS)
    addAttribute(A);
}

void AttrBuilder::clear() {
  Attrs.reset();
  TargetDepAttrs.clear();
  IntAttrs = {};
  TypeAttrs = {};
}

Optional<unsigned>
AttrBuilder::kindToIntIndex(Attribute::AttrKind Kind) const {
  if (Attribute::isIntAttrKind(Kind))
    return Kind - Attribute::FirstIntAttr;
  return None;
}

Optional<unsigned>
AttrBuilder::kindToTypeIndex(Attribute::AttrKind Kind) const {
  if (Attribute::isTypeAttrKind(Kind))
    return Kind - Attribute::FirstTypeAttr;
  return None;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute Attr) {
  if (Attr.isStringAttribute()) {
    addAttribute(Attr.getKindAsString(), Attr.getValueAsString());
    return *this;
  }

  Attribute::AttrKind Kind = Attr.getKindAsEnum();
  Attrs[Kind] = true;

  if (Optional<unsigned> TypeIndex = kindToTypeIndex(Kind))
    TypeAttrs[*TypeIndex] = Attr.getValueAsType();
  else if (Optional<unsigned> IntIndex = kindToIntIndex(Kind))
    IntAttrs[*IntIndex] = Attr.getValueAsInt();

  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(StringRef A, StringRef V) {
  TargetDepAttrs[A] = V;
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(Attribute::AttrKind Val) {
  assert((unsigned)Val < Attribute::EndAttrKinds && "Attribute out of range!");
  Attrs[Val] = false;

  if (Optional<unsigned> TypeIndex = kindToTypeIndex(Val))
    TypeAttrs[*TypeIndex] = nullptr;
  else if (Optional<unsigned> IntIndex = kindToIntIndex(Val))
    IntAttrs[*IntIndex] = 0;

  return *this;
}

AttrBuilder &AttrBuilder::removeAttributes(AttributeList A, uint64_t Index) {
  remove(A.getAttributes(Index));
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(StringRef A) {
  auto I = TargetDepAttrs.find(A);
  if (I != TargetDepAttrs.end())
    TargetDepAttrs.erase(I);
  return *this;
}

uint64_t AttrBuilder::getRawIntAttr(Attribute::AttrKind Kind) const {
  Optional<unsigned> IntIndex = kindToIntIndex(Kind);
  assert(IntIndex && "Not an int attribute");
  return IntAttrs[*IntIndex];
}

AttrBuilder &AttrBuilder::addRawIntAttr(Attribute::AttrKind Kind,
                                        uint64_t Value) {
  Optional<unsigned> IntIndex = kindToIntIndex(Kind);
  assert(IntIndex && "Not an int attribute");
  assert(Value && "Value cannot be zero");
  Attrs[Kind] = true;
  IntAttrs[*IntIndex] = Value;
  return *this;
}

std::pair<unsigned, Optional<unsigned>> AttrBuilder::getAllocSizeArgs() const {
  return unpackAllocSizeArgs(getRawIntAttr(Attribute::AllocSize));
}

std::pair<unsigned, unsigned> AttrBuilder::getVScaleRangeArgs() const {
  return unpackVScaleRangeArgs(getRawIntAttr(Attribute::VScaleRange));
}

AttrBuilder &AttrBuilder::addAlignmentAttr(MaybeAlign Align) {
  if (!Align)
    return *this;

  assert(*Align <= llvm::Value::MaximumAlignment && "Alignment too large.");
  return addRawIntAttr(Attribute::Alignment, Align->value());
}

AttrBuilder &AttrBuilder::addStackAlignmentAttr(MaybeAlign Align) {
  // Default alignment, allow the target to define how to align it.
  if (!Align)
    return *this;

  assert(*Align <= 0x100 && "Alignment too large.");
  return addRawIntAttr(Attribute::StackAlignment, Align->value());
}

AttrBuilder &AttrBuilder::addDereferenceableAttr(uint64_t Bytes) {
  if (Bytes == 0) return *this;

  return addRawIntAttr(Attribute::Dereferenceable, Bytes);
}

AttrBuilder &AttrBuilder::addDereferenceableOrNullAttr(uint64_t Bytes) {
  if (Bytes == 0)
    return *this;

  return addRawIntAttr(Attribute::DereferenceableOrNull, Bytes);
}

AttrBuilder &AttrBuilder::addAllocSizeAttr(unsigned ElemSize,
                                           const Optional<unsigned> &NumElems) {
  return addAllocSizeAttrFromRawRepr(packAllocSizeArgs(ElemSize, NumElems));
}

AttrBuilder &AttrBuilder::addAllocSizeAttrFromRawRepr(uint64_t RawArgs) {
  // (0, 0) is our "not present" value, so we need to check for it here.
  assert(RawArgs && "Invalid allocsize arguments -- given allocsize(0, 0)");
  return addRawIntAttr(Attribute::AllocSize, RawArgs);
}

AttrBuilder &AttrBuilder::addVScaleRangeAttr(unsigned MinValue,
                                             unsigned MaxValue) {
  return addVScaleRangeAttrFromRawRepr(packVScaleRangeArgs(MinValue, MaxValue));
}

AttrBuilder &AttrBuilder::addVScaleRangeAttrFromRawRepr(uint64_t RawArgs) {
  // (0, 0) is not present hence ignore this case
  if (RawArgs == 0)
    return *this;

  return addRawIntAttr(Attribute::VScaleRange, RawArgs);
}

Type *AttrBuilder::getTypeAttr(Attribute::AttrKind Kind) const {
  Optional<unsigned> TypeIndex = kindToTypeIndex(Kind);
  assert(TypeIndex && "Not a type attribute");
  return TypeAttrs[*TypeIndex];
}

AttrBuilder &AttrBuilder::addTypeAttr(Attribute::AttrKind Kind, Type *Ty) {
  Optional<unsigned> TypeIndex = kindToTypeIndex(Kind);
  assert(TypeIndex && "Not a type attribute");
  Attrs[Kind] = true;
  TypeAttrs[*TypeIndex] = Ty;
  return *this;
}

AttrBuilder &AttrBuilder::addByValAttr(Type *Ty) {
  return addTypeAttr(Attribute::ByVal, Ty);
}

AttrBuilder &AttrBuilder::addStructRetAttr(Type *Ty) {
  return addTypeAttr(Attribute::StructRet, Ty);
}

AttrBuilder &AttrBuilder::addByRefAttr(Type *Ty) {
  return addTypeAttr(Attribute::ByRef, Ty);
}

AttrBuilder &AttrBuilder::addPreallocatedAttr(Type *Ty) {
  return addTypeAttr(Attribute::Preallocated, Ty);
}

AttrBuilder &AttrBuilder::addInAllocaAttr(Type *Ty) {
  return addTypeAttr(Attribute::InAlloca, Ty);
}

AttrBuilder &AttrBuilder::merge(const AttrBuilder &B) {
  // FIXME: What if both have an int/type attribute, but they don't match?!
  for (unsigned Index = 0; Index < Attribute::NumIntAttrKinds; ++Index)
    if (!IntAttrs[Index])
      IntAttrs[Index] = B.IntAttrs[Index];

  for (unsigned Index = 0; Index < Attribute::NumTypeAttrKinds; ++Index)
    if (!TypeAttrs[Index])
      TypeAttrs[Index] = B.TypeAttrs[Index];

  Attrs |= B.Attrs;

  for (const auto &I : B.td_attrs())
    TargetDepAttrs[I.first] = I.second;

  return *this;
}

AttrBuilder &AttrBuilder::remove(const AttrBuilder &B) {
  // FIXME: What if both have an int/type attribute, but they don't match?!
  for (unsigned Index = 0; Index < Attribute::NumIntAttrKinds; ++Index)
    if (B.IntAttrs[Index])
      IntAttrs[Index] = 0;

  for (unsigned Index = 0; Index < Attribute::NumTypeAttrKinds; ++Index)
    if (B.TypeAttrs[Index])
      TypeAttrs[Index] = nullptr;

  Attrs &= ~B.Attrs;

  for (const auto &I : B.td_attrs())
    TargetDepAttrs.erase(I.first);

  return *this;
}

bool AttrBuilder::overlaps(const AttrBuilder &B) const {
  // First check if any of the target independent attributes overlap.
  if ((Attrs & B.Attrs).any())
    return true;

  // Then check if any target dependent ones do.
  for (const auto &I : td_attrs())
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

bool AttrBuilder::hasAttributes(AttributeList AL, uint64_t Index) const {
  AttributeSet AS = AL.getAttributes(Index);

  for (const auto &Attr : AS) {
    if (Attr.isEnumAttribute() || Attr.isIntAttribute()) {
      if (contains(Attr.getKindAsEnum()))
        return true;
    } else {
      assert(Attr.isStringAttribute() && "Invalid attribute kind!");
      return contains(Attr.getKindAsString());
    }
  }

  return false;
}

bool AttrBuilder::hasAlignmentAttr() const {
  return getRawIntAttr(Attribute::Alignment) != 0;
}

bool AttrBuilder::operator==(const AttrBuilder &B) const {
  if (Attrs != B.Attrs)
    return false;

  for (const auto &TDA : TargetDepAttrs)
    if (B.TargetDepAttrs.find(TDA.first) == B.TargetDepAttrs.end())
      return false;

  return IntAttrs == B.IntAttrs && TypeAttrs == B.TypeAttrs;
}

//===----------------------------------------------------------------------===//
// AttributeFuncs Function Defintions
//===----------------------------------------------------------------------===//

/// Which attributes cannot be applied to a type.
AttrBuilder AttributeFuncs::typeIncompatible(Type *Ty) {
  AttrBuilder Incompatible;

  if (!Ty->isIntegerTy())
    // Attribute that only apply to integers.
    Incompatible.addAttribute(Attribute::SExt)
      .addAttribute(Attribute::ZExt);

  if (!Ty->isPointerTy())
    // Attribute that only apply to pointers.
    Incompatible.addAttribute(Attribute::Nest)
        .addAttribute(Attribute::NoAlias)
        .addAttribute(Attribute::NoCapture)
        .addAttribute(Attribute::NonNull)
        .addAttribute(Attribute::ReadNone)
        .addAttribute(Attribute::ReadOnly)
        .addAttribute(Attribute::SwiftError)
        .addAlignmentAttr(1)             // the int here is ignored
        .addDereferenceableAttr(1)       // the int here is ignored
        .addDereferenceableOrNullAttr(1) // the int here is ignored
        .addPreallocatedAttr(Ty)
        .addInAllocaAttr(Ty)
        .addByValAttr(Ty)
        .addStructRetAttr(Ty)
        .addByRefAttr(Ty)
        .addTypeAttr(Attribute::ElementType, Ty);

  // Some attributes can apply to all "values" but there are no `void` values.
  if (Ty->isVoidTy())
    Incompatible.addAttribute(Attribute::NoUndef);

  return Incompatible;
}

AttrBuilder AttributeFuncs::getUBImplyingAttributes() {
  AttrBuilder B;
  B.addAttribute(Attribute::NoUndef);
  B.addDereferenceableAttr(1);
  B.addDereferenceableOrNullAttr(1);
  return B;
}

template<typename AttrClass>
static bool isEqual(const Function &Caller, const Function &Callee) {
  return Caller.getFnAttribute(AttrClass::getKind()) ==
         Callee.getFnAttribute(AttrClass::getKind());
}

/// Compute the logical AND of the attributes of the caller and the
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

/// Compute the logical OR of the attributes of the caller and the
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

/// If the inlined function had a higher stack protection level than the
/// calling function, then bump up the caller's stack protection level.
static void adjustCallerSSPLevel(Function &Caller, const Function &Callee) {
  // If upgrading the SSP attribute, clear out the old SSP Attributes first.
  // Having multiple SSP attributes doesn't actually hurt, but it adds useless
  // clutter to the IR.
  AttrBuilder OldSSPAttr;
  OldSSPAttr.addAttribute(Attribute::StackProtect)
      .addAttribute(Attribute::StackProtectStrong)
      .addAttribute(Attribute::StackProtectReq);

  if (Callee.hasFnAttribute(Attribute::StackProtectReq)) {
    Caller.removeFnAttrs(OldSSPAttr);
    Caller.addFnAttr(Attribute::StackProtectReq);
  } else if (Callee.hasFnAttribute(Attribute::StackProtectStrong) &&
             !Caller.hasFnAttribute(Attribute::StackProtectReq)) {
    Caller.removeFnAttrs(OldSSPAttr);
    Caller.addFnAttr(Attribute::StackProtectStrong);
  } else if (Callee.hasFnAttribute(Attribute::StackProtect) &&
             !Caller.hasFnAttribute(Attribute::StackProtectReq) &&
             !Caller.hasFnAttribute(Attribute::StackProtectStrong))
    Caller.addFnAttr(Attribute::StackProtect);
}

/// If the inlined function required stack probes, then ensure that
/// the calling function has those too.
static void adjustCallerStackProbes(Function &Caller, const Function &Callee) {
  if (!Caller.hasFnAttribute("probe-stack") &&
      Callee.hasFnAttribute("probe-stack")) {
    Caller.addFnAttr(Callee.getFnAttribute("probe-stack"));
  }
}

/// If the inlined function defines the size of guard region
/// on the stack, then ensure that the calling function defines a guard region
/// that is no larger.
static void
adjustCallerStackProbeSize(Function &Caller, const Function &Callee) {
  Attribute CalleeAttr = Callee.getFnAttribute("stack-probe-size");
  if (CalleeAttr.isValid()) {
    Attribute CallerAttr = Caller.getFnAttribute("stack-probe-size");
    if (CallerAttr.isValid()) {
      uint64_t CallerStackProbeSize, CalleeStackProbeSize;
      CallerAttr.getValueAsString().getAsInteger(0, CallerStackProbeSize);
      CalleeAttr.getValueAsString().getAsInteger(0, CalleeStackProbeSize);

      if (CallerStackProbeSize > CalleeStackProbeSize) {
        Caller.addFnAttr(CalleeAttr);
      }
    } else {
      Caller.addFnAttr(CalleeAttr);
    }
  }
}

/// If the inlined function defines a min legal vector width, then ensure
/// the calling function has the same or larger min legal vector width. If the
/// caller has the attribute, but the callee doesn't, we need to remove the
/// attribute from the caller since we can't make any guarantees about the
/// caller's requirements.
/// This function is called after the inlining decision has been made so we have
/// to merge the attribute this way. Heuristics that would use
/// min-legal-vector-width to determine inline compatibility would need to be
/// handled as part of inline cost analysis.
static void
adjustMinLegalVectorWidth(Function &Caller, const Function &Callee) {
  Attribute CallerAttr = Caller.getFnAttribute("min-legal-vector-width");
  if (CallerAttr.isValid()) {
    Attribute CalleeAttr = Callee.getFnAttribute("min-legal-vector-width");
    if (CalleeAttr.isValid()) {
      uint64_t CallerVectorWidth, CalleeVectorWidth;
      CallerAttr.getValueAsString().getAsInteger(0, CallerVectorWidth);
      CalleeAttr.getValueAsString().getAsInteger(0, CalleeVectorWidth);
      if (CallerVectorWidth < CalleeVectorWidth)
        Caller.addFnAttr(CalleeAttr);
    } else {
      // If the callee doesn't have the attribute then we don't know anything
      // and must drop the attribute from the caller.
      Caller.removeFnAttr("min-legal-vector-width");
    }
  }
}

/// If the inlined function has null_pointer_is_valid attribute,
/// set this attribute in the caller post inlining.
static void
adjustNullPointerValidAttr(Function &Caller, const Function &Callee) {
  if (Callee.nullPointerIsDefined() && !Caller.nullPointerIsDefined()) {
    Caller.addFnAttr(Attribute::NullPointerIsValid);
  }
}

struct EnumAttr {
  static bool isSet(const Function &Fn,
                    Attribute::AttrKind Kind) {
    return Fn.hasFnAttribute(Kind);
  }

  static void set(Function &Fn,
                  Attribute::AttrKind Kind, bool Val) {
    if (Val)
      Fn.addFnAttr(Kind);
    else
      Fn.removeFnAttr(Kind);
  }
};

struct StrBoolAttr {
  static bool isSet(const Function &Fn,
                    StringRef Kind) {
    auto A = Fn.getFnAttribute(Kind);
    return A.getValueAsString().equals("true");
  }

  static void set(Function &Fn,
                  StringRef Kind, bool Val) {
    Fn.addFnAttr(Kind, Val ? "true" : "false");
  }
};

#define GET_ATTR_NAMES
#define ATTRIBUTE_ENUM(ENUM_NAME, DISPLAY_NAME)                                \
  struct ENUM_NAME##Attr : EnumAttr {                                          \
    static enum Attribute::AttrKind getKind() {                                \
      return llvm::Attribute::ENUM_NAME;                                       \
    }                                                                          \
  };
#define ATTRIBUTE_STRBOOL(ENUM_NAME, DISPLAY_NAME)                             \
  struct ENUM_NAME##Attr : StrBoolAttr {                                       \
    static StringRef getKind() { return #DISPLAY_NAME; }                       \
  };
#include "llvm/IR/Attributes.inc"

#define GET_ATTR_COMPAT_FUNC
#include "llvm/IR/Attributes.inc"

bool AttributeFuncs::areInlineCompatible(const Function &Caller,
                                         const Function &Callee) {
  return hasCompatibleFnAttrs(Caller, Callee);
}

bool AttributeFuncs::areOutlineCompatible(const Function &A,
                                          const Function &B) {
  return hasCompatibleFnAttrs(A, B);
}

void AttributeFuncs::mergeAttributesForInlining(Function &Caller,
                                                const Function &Callee) {
  mergeFnAttrs(Caller, Callee);
}

void AttributeFuncs::mergeAttributesForOutlining(Function &Base,
                                                const Function &ToMerge) {

  // We merge functions so that they meet the most general case.
  // For example, if the NoNansFPMathAttr is set in one function, but not in
  // the other, in the merged function we can say that the NoNansFPMathAttr
  // is not set.
  // However if we have the SpeculativeLoadHardeningAttr set true in one
  // function, but not the other, we make sure that the function retains
  // that aspect in the merged function.
  mergeFnAttrs(Base, ToMerge);
}
