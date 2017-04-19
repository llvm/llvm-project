//===- Attributes.cpp - Implement AttributesList --------------------------===//
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
// AttributeListImpl, and AttributeList classes.
//
//===----------------------------------------------------------------------===//

#include "AttributeImpl.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
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
#include <cstdint>
#include <limits>
#include <map>
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

    std::string AttrVal = pImpl->getValueAsString();
    if (AttrVal.empty()) return Result;

    // Since some attribute strings contain special characters that cannot be
    // printable, those have to be escaped to make the attribute value printable
    // as is.  e.g. "\01__gnu_mcount_nc"
    {
      raw_string_ostream OS(Result);
      OS << "=\"";
      PrintEscapedString(AttrVal, OS);
      OS << "\"";
    }
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
AttributeImpl::~AttributeImpl() = default;

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

//===----------------------------------------------------------------------===//
// AttributeSet Definition
//===----------------------------------------------------------------------===//

AttributeSet AttributeSet::get(LLVMContext &C, const AttrBuilder &B) {
  return AttributeSet(AttributeSetNode::get(C, B));
}

AttributeSet AttributeSet::get(LLVMContext &C, ArrayRef<Attribute> Attrs) {
  return AttributeSet(AttributeSetNode::get(C, Attrs));
}

unsigned AttributeSet::getNumAttributes() const {
  return SetNode ? SetNode->getNumAttributes() : 0;
}

bool AttributeSet::hasAttribute(Attribute::AttrKind Kind) const {
  return SetNode ? SetNode->hasAttribute(Kind) : 0;
}

bool AttributeSet::hasAttribute(StringRef Kind) const {
  return SetNode ? SetNode->hasAttribute(Kind) : 0;
}

Attribute AttributeSet::getAttribute(Attribute::AttrKind Kind) const {
  return SetNode ? SetNode->getAttribute(Kind) : Attribute();
}

Attribute AttributeSet::getAttribute(StringRef Kind) const {
  return SetNode ? SetNode->getAttribute(Kind) : Attribute();
}

unsigned AttributeSet::getAlignment() const {
  return SetNode ? SetNode->getAlignment() : 0;
}

unsigned AttributeSet::getStackAlignment() const {
  return SetNode ? SetNode->getStackAlignment() : 0;
}

uint64_t AttributeSet::getDereferenceableBytes() const {
  return SetNode ? SetNode->getDereferenceableBytes() : 0;
}

uint64_t AttributeSet::getDereferenceableOrNullBytes() const {
  return SetNode ? SetNode->getDereferenceableOrNullBytes() : 0;
}

std::pair<unsigned, Optional<unsigned>> AttributeSet::getAllocSizeArgs() const {
  return SetNode ? SetNode->getAllocSizeArgs()
                 : std::pair<unsigned, Optional<unsigned>>(0, 0);
}

std::string AttributeSet::getAsString(bool InAttrGrp) const {
  return SetNode ? SetNode->getAsString(InAttrGrp) : "";
}

AttributeSet::iterator AttributeSet::begin() const {
  return SetNode ? SetNode->begin() : nullptr;
}

AttributeSet::iterator AttributeSet::end() const {
  return SetNode ? SetNode->end() : nullptr;
}

//===----------------------------------------------------------------------===//
// AttributeSetNode Definition
//===----------------------------------------------------------------------===//

AttributeSetNode::AttributeSetNode(ArrayRef<Attribute> Attrs)
    : AvailableAttrs(0), NumAttrs(Attrs.size()) {
  // There's memory after the node where we can store the entries in.
  std::copy(Attrs.begin(), Attrs.end(), getTrailingObjects<Attribute>());

  for (Attribute I : *this) {
    if (!I.isStringAttribute()) {
      AvailableAttrs |= ((uint64_t)1) << I.getKindAsEnum();
    }
  }
}

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
    Attrs.push_back(Attr);
  }

  // Add target-dependent (string) attributes.
  for (const auto &TDA : B.td_attrs())
    Attrs.emplace_back(Attribute::get(C, TDA.first, TDA.second));

  return get(C, Attrs);
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
// AttributeListImpl Definition
//===----------------------------------------------------------------------===//

AttributeListImpl::AttributeListImpl(
    LLVMContext &C, ArrayRef<std::pair<unsigned, AttributeSet>> Slots)
    : Context(C), NumSlots(Slots.size()), AvailableFunctionAttrs(0) {
#ifndef NDEBUG
  assert(!Slots.empty() && "pointless AttributeListImpl");
  if (Slots.size() >= 2) {
    auto &PrevPair = Slots.front();
    for (auto &CurPair : Slots.drop_front()) {
      assert(PrevPair.first <= CurPair.first && "Attribute set not ordered!");
    }
  }
#endif

  // There's memory after the node where we can store the entries in.
  std::copy(Slots.begin(), Slots.end(), getTrailingObjects<IndexAttrPair>());

  // Initialize AvailableFunctionAttrs summary bitset.
  static_assert(Attribute::EndAttrKinds <=
                    sizeof(AvailableFunctionAttrs) * CHAR_BIT,
                "Too many attributes");
  static_assert(AttributeList::FunctionIndex == ~0u,
                "FunctionIndex should be biggest possible index");
  const auto &Last = Slots.back();
  if (Last.first == AttributeList::FunctionIndex) {
    AttributeSet Node = Last.second;
    for (Attribute I : Node) {
      if (!I.isStringAttribute())
        AvailableFunctionAttrs |= ((uint64_t)1) << I.getKindAsEnum();
    }
  }
}

void AttributeListImpl::Profile(FoldingSetNodeID &ID) const {
  Profile(ID, makeArrayRef(getSlotPair(0), getNumSlots()));
}

void AttributeListImpl::Profile(
    FoldingSetNodeID &ID, ArrayRef<std::pair<unsigned, AttributeSet>> Nodes) {
  for (const auto &Node : Nodes) {
    ID.AddInteger(Node.first);
    ID.AddPointer(Node.second.SetNode);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void AttributeListImpl::dump() const {
  AttributeList(const_cast<AttributeListImpl *>(this)).dump();
}
#endif

//===----------------------------------------------------------------------===//
// AttributeList Construction and Mutation Methods
//===----------------------------------------------------------------------===//

AttributeList AttributeList::getImpl(
    LLVMContext &C, ArrayRef<std::pair<unsigned, AttributeSet>> Attrs) {
  assert(!Attrs.empty() && "creating pointless AttributeList");
#ifndef NDEBUG
  unsigned LastIndex = 0;
  bool IsFirst = true;
  for (auto &&AttrPair : Attrs) {
    assert((IsFirst || LastIndex < AttrPair.first) &&
           "unsorted or duplicate AttributeList indices");
    assert(AttrPair.second.hasAttributes() && "pointless AttributeList slot");
    LastIndex = AttrPair.first;
    IsFirst = false;
  }
#endif

  LLVMContextImpl *pImpl = C.pImpl;
  FoldingSetNodeID ID;
  AttributeListImpl::Profile(ID, Attrs);

  void *InsertPoint;
  AttributeListImpl *PA =
      pImpl->AttrsLists.FindNodeOrInsertPos(ID, InsertPoint);

  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PA) {
    // Coallocate entries after the AttributeListImpl itself.
    void *Mem = ::operator new(
        AttributeListImpl::totalSizeToAlloc<IndexAttrPair>(Attrs.size()));
    PA = new (Mem) AttributeListImpl(C, Attrs);
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
    return AttributeList();

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

  return getImpl(C, AttrPairVec);
}

AttributeList
AttributeList::get(LLVMContext &C,
                   ArrayRef<std::pair<unsigned, AttributeSet>> Attrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return AttributeList();

  return getImpl(C, Attrs);
}

AttributeList AttributeList::get(LLVMContext &C, AttributeSet FnAttrs,
                                 AttributeSet RetAttrs,
                                 ArrayRef<AttributeSet> ArgAttrs) {
  SmallVector<std::pair<unsigned, AttributeSet>, 8> AttrPairs;
  if (RetAttrs.hasAttributes())
    AttrPairs.emplace_back(ReturnIndex, RetAttrs);
  size_t Index = 1;
  for (AttributeSet AS : ArgAttrs) {
    if (AS.hasAttributes())
      AttrPairs.emplace_back(Index, AS);
    ++Index;
  }
  if (FnAttrs.hasAttributes())
    AttrPairs.emplace_back(FunctionIndex, FnAttrs);
  if (AttrPairs.empty())
    return AttributeList();
  return getImpl(C, AttrPairs);
}

AttributeList AttributeList::get(LLVMContext &C, unsigned Index,
                                 const AttrBuilder &B) {
  if (!B.hasAttributes())
    return AttributeList();
  AttributeSet AS = AttributeSet::get(C, B);
  std::pair<unsigned, AttributeSet> Arr[1] = {{Index, AS}};
  return getImpl(C, Arr);
}

AttributeList AttributeList::get(LLVMContext &C, unsigned Index,
                                 ArrayRef<Attribute::AttrKind> Kinds) {
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (Attribute::AttrKind K : Kinds)
    Attrs.emplace_back(Index, Attribute::get(C, K));
  return get(C, Attrs);
}

AttributeList AttributeList::get(LLVMContext &C, unsigned Index,
                                 ArrayRef<StringRef> Kinds) {
  SmallVector<std::pair<unsigned, Attribute>, 8> Attrs;
  for (StringRef K : Kinds)
    Attrs.emplace_back(Index, Attribute::get(C, K));
  return get(C, Attrs);
}

AttributeList AttributeList::get(LLVMContext &C,
                                 ArrayRef<AttributeList> Attrs) {
  if (Attrs.empty())
    return AttributeList();
  if (Attrs.size() == 1) return Attrs[0];

  SmallVector<std::pair<unsigned, AttributeSet>, 8> AttrNodeVec;
  AttributeListImpl *A0 = Attrs[0].pImpl;
  if (A0)
    AttrNodeVec.append(A0->getSlotPair(0), A0->getSlotPair(A0->getNumSlots()));
  // Copy all attributes from Attrs into AttrNodeVec while keeping AttrNodeVec
  // ordered by index.  Because we know that each list in Attrs is ordered by
  // index we only need to merge each successive list in rather than doing a
  // full sort.
  for (unsigned I = 1, E = Attrs.size(); I != E; ++I) {
    AttributeListImpl *ALI = Attrs[I].pImpl;
    if (!ALI) continue;
    SmallVector<std::pair<unsigned, AttributeSet>, 8>::iterator
      ANVI = AttrNodeVec.begin(), ANVE;
    for (const IndexAttrPair *AI = ALI->getSlotPair(0),
                             *AE = ALI->getSlotPair(ALI->getNumSlots());
         AI != AE; ++AI) {
      ANVE = AttrNodeVec.end();
      while (ANVI != ANVE && ANVI->first <= AI->first)
        ++ANVI;
      ANVI = AttrNodeVec.insert(ANVI, *AI) + 1;
    }
  }

  return getImpl(C, AttrNodeVec);
}

AttributeList AttributeList::addAttribute(LLVMContext &C, unsigned Index,
                                          Attribute::AttrKind Kind) const {
  if (hasAttribute(Index, Kind)) return *this;
  return addAttributes(C, Index, AttributeList::get(C, Index, Kind));
}

AttributeList AttributeList::addAttribute(LLVMContext &C, unsigned Index,
                                          StringRef Kind,
                                          StringRef Value) const {
  AttrBuilder B;
  B.addAttribute(Kind, Value);
  return addAttributes(C, Index, AttributeList::get(C, Index, B));
}

AttributeList AttributeList::addAttribute(LLVMContext &C,
                                          ArrayRef<unsigned> Indices,
                                          Attribute A) const {
  assert(std::is_sorted(Indices.begin(), Indices.end()));

  unsigned I = 0, E = pImpl ? pImpl->getNumSlots() : 0;
  SmallVector<IndexAttrPair, 4> AttrVec;
  for (unsigned Index : Indices) {
    // Add all attribute slots before the current index.
    for (; I < E && getSlotIndex(I) < Index; ++I)
      AttrVec.emplace_back(getSlotIndex(I), pImpl->getSlotNode(I));

    // Add the attribute at this index. If we already have attributes at this
    // index, merge them into a new set.
    AttrBuilder B;
    if (I < E && getSlotIndex(I) == Index) {
      B.merge(AttrBuilder(pImpl->getSlotNode(I)));
      ++I;
    }
    B.addAttribute(A);
    AttrVec.emplace_back(Index, AttributeSet::get(C, B));
  }

  // Add remaining attributes.
  for (; I < E; ++I)
    AttrVec.emplace_back(getSlotIndex(I), pImpl->getSlotNode(I));

  return get(C, AttrVec);
}

AttributeList AttributeList::addAttributes(LLVMContext &C, unsigned Index,
                                           AttributeList Attrs) const {
  if (!pImpl) return Attrs;
  if (!Attrs.pImpl) return *this;

  return addAttributes(C, Index, Attrs.getAttributes(Index));
}

AttributeList AttributeList::addAttributes(LLVMContext &C, unsigned Index,
                                           const AttrBuilder &B) const {
  if (!B.hasAttributes())
    return *this;

  if (!pImpl)
    return AttributeList::get(C, {{Index, AttributeSet::get(C, B)}});

#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment. For now, say
  // we can't change a known alignment.
  unsigned OldAlign = getParamAlignment(Index);
  unsigned NewAlign = B.getAlignment();
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  SmallVector<IndexAttrPair, 4> AttrVec;
  uint64_t NumAttrs = pImpl->getNumSlots();
  unsigned I;

  // Add all the attribute slots before the one we need to merge.
  for (I = 0; I < NumAttrs; ++I) {
    if (getSlotIndex(I) >= Index)
      break;
    AttrVec.emplace_back(getSlotIndex(I), pImpl->getSlotNode(I));
  }

  AttrBuilder NewAttrs;
  if (I < NumAttrs && getSlotIndex(I) == Index) {
    // We need to merge the attribute sets.
    NewAttrs.merge(pImpl->getSlotNode(I));
    ++I;
  }
  NewAttrs.merge(B);

  // Add the new or merged attribute set at this index.
  AttrVec.emplace_back(Index, AttributeSet::get(C, NewAttrs));

  // Add the remaining entries.
  for (; I < NumAttrs; ++I)
    AttrVec.emplace_back(getSlotIndex(I), pImpl->getSlotNode(I));

  return get(C, AttrVec);
}

AttributeList AttributeList::removeAttribute(LLVMContext &C, unsigned Index,
                                             Attribute::AttrKind Kind) const {
  if (!hasAttribute(Index, Kind)) return *this;
  return removeAttributes(C, Index, AttributeList::get(C, Index, Kind));
}

AttributeList AttributeList::removeAttribute(LLVMContext &C, unsigned Index,
                                             StringRef Kind) const {
  if (!hasAttribute(Index, Kind)) return *this;
  return removeAttributes(C, Index, AttributeList::get(C, Index, Kind));
}

AttributeList AttributeList::removeAttributes(LLVMContext &C, unsigned Index,
                                              AttributeList Attrs) const {
  if (!pImpl)
    return AttributeList();
  if (!Attrs.pImpl) return *this;

  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAttribute(Index, Attribute::Alignment) &&
         "Attempt to change alignment!");

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeList, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumSlots();
  AttributeList AL;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Index) {
      if (getSlotIndex(I) == Index) AL = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now remove the attribute from the correct slot. There may already be an
  // AttributeList there.
  AttrBuilder B(AL, Index);

  for (unsigned I = 0, E = Attrs.pImpl->getNumSlots(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Index) {
      B.removeAttributes(Attrs.pImpl->getSlotAttributes(I), Index);
      break;
    }

  AttrSet.push_back(AttributeList::get(C, Index, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

AttributeList AttributeList::removeAttributes(LLVMContext &C, unsigned Index,
                                              const AttrBuilder &Attrs) const {
  if (!pImpl)
    return AttributeList();

  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAlignmentAttr() && "Attempt to change alignment!");

  // Add the attribute slots before the one we're trying to add.
  SmallVector<AttributeList, 4> AttrSet;
  uint64_t NumAttrs = pImpl->getNumSlots();
  AttributeList AL;
  uint64_t LastIndex = 0;
  for (unsigned I = 0, E = NumAttrs; I != E; ++I) {
    if (getSlotIndex(I) >= Index) {
      if (getSlotIndex(I) == Index) AL = getSlotAttributes(LastIndex++);
      break;
    }
    LastIndex = I + 1;
    AttrSet.push_back(getSlotAttributes(I));
  }

  // Now remove the attribute from the correct slot. There may already be an
  // AttributeList there.
  AttrBuilder B(AL, Index);
  B.remove(Attrs);

  AttrSet.push_back(AttributeList::get(C, Index, B));

  // Add the remaining attribute slots.
  for (unsigned I = LastIndex, E = NumAttrs; I < E; ++I)
    AttrSet.push_back(getSlotAttributes(I));

  return get(C, AttrSet);
}

AttributeList AttributeList::removeAttributes(LLVMContext &C,
                                              unsigned WithoutIndex) const {
  if (!pImpl)
    return AttributeList();

  SmallVector<std::pair<unsigned, AttributeSet>, 4> AttrSet;
  for (unsigned I = 0, E = pImpl->getNumSlots(); I != E; ++I) {
    unsigned Index = getSlotIndex(I);
    if (Index != WithoutIndex)
      AttrSet.push_back({Index, pImpl->getSlotNode(I)});
  }
  return get(C, AttrSet);
}

AttributeList AttributeList::addDereferenceableAttr(LLVMContext &C,
                                                    unsigned Index,
                                                    uint64_t Bytes) const {
  AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
  return addAttributes(C, Index, AttributeList::get(C, Index, B));
}

AttributeList
AttributeList::addDereferenceableOrNullAttr(LLVMContext &C, unsigned Index,
                                            uint64_t Bytes) const {
  AttrBuilder B;
  B.addDereferenceableOrNullAttr(Bytes);
  return addAttributes(C, Index, AttributeList::get(C, Index, B));
}

AttributeList
AttributeList::addAllocSizeAttr(LLVMContext &C, unsigned Index,
                                unsigned ElemSizeArg,
                                const Optional<unsigned> &NumElemsArg) {
  AttrBuilder B;
  B.addAllocSizeAttr(ElemSizeArg, NumElemsArg);
  return addAttributes(C, Index, AttributeList::get(C, Index, B));
}

//===----------------------------------------------------------------------===//
// AttributeList Accessor Methods
//===----------------------------------------------------------------------===//

LLVMContext &AttributeList::getContext() const { return pImpl->getContext(); }

AttributeSet AttributeList::getParamAttributes(unsigned ArgNo) const {
  return getAttributes(ArgNo + 1);
}

AttributeSet AttributeList::getRetAttributes() const {
  return getAttributes(ReturnIndex);
}

AttributeSet AttributeList::getFnAttributes() const {
  return getAttributes(FunctionIndex);
}

bool AttributeList::hasAttribute(unsigned Index,
                                 Attribute::AttrKind Kind) const {
  return getAttributes(Index).hasAttribute(Kind);
}

bool AttributeList::hasAttribute(unsigned Index, StringRef Kind) const {
  return getAttributes(Index).hasAttribute(Kind);
}

bool AttributeList::hasAttributes(unsigned Index) const {
  return getAttributes(Index).hasAttributes();
}

bool AttributeList::hasFnAttribute(Attribute::AttrKind Kind) const {
  return pImpl && pImpl->hasFnAttribute(Kind);
}

bool AttributeList::hasFnAttribute(StringRef Kind) const {
  return hasAttribute(AttributeList::FunctionIndex, Kind);
}

bool AttributeList::hasParamAttribute(unsigned ArgNo,
                                      Attribute::AttrKind Kind) const {
  return hasAttribute(ArgNo + 1, Kind);
}

bool AttributeList::hasAttrSomewhere(Attribute::AttrKind Attr,
                                     unsigned *Index) const {
  if (!pImpl) return false;

  for (unsigned I = 0, E = pImpl->getNumSlots(); I != E; ++I)
    for (AttributeListImpl::iterator II = pImpl->begin(I), IE = pImpl->end(I);
         II != IE; ++II)
      if (II->hasAttribute(Attr)) {
        if (Index) *Index = pImpl->getSlotIndex(I);
        return true;
      }

  return false;
}

Attribute AttributeList::getAttribute(unsigned Index,
                                      Attribute::AttrKind Kind) const {
  return getAttributes(Index).getAttribute(Kind);
}

Attribute AttributeList::getAttribute(unsigned Index, StringRef Kind) const {
  return getAttributes(Index).getAttribute(Kind);
}

unsigned AttributeList::getParamAlignment(unsigned Index) const {
  return getAttributes(Index).getAlignment();
}

unsigned AttributeList::getStackAlignment(unsigned Index) const {
  return getAttributes(Index).getStackAlignment();
}

uint64_t AttributeList::getDereferenceableBytes(unsigned Index) const {
  return getAttributes(Index).getDereferenceableBytes();
}

uint64_t AttributeList::getDereferenceableOrNullBytes(unsigned Index) const {
  return getAttributes(Index).getDereferenceableOrNullBytes();
}

std::pair<unsigned, Optional<unsigned>>
AttributeList::getAllocSizeArgs(unsigned Index) const {
  return getAttributes(Index).getAllocSizeArgs();
}

std::string AttributeList::getAsString(unsigned Index, bool InAttrGrp) const {
  return getAttributes(Index).getAsString(InAttrGrp);
}

AttributeSet AttributeList::getAttributes(unsigned Index) const {
  if (!pImpl) return AttributeSet();

  // Loop through to find the attribute node we want.
  for (unsigned I = 0, E = pImpl->getNumSlots(); I != E; ++I)
    if (pImpl->getSlotIndex(I) == Index)
      return pImpl->getSlotNode(I);

  return AttributeSet();
}

AttributeList::iterator AttributeList::begin(unsigned Slot) const {
  if (!pImpl)
    return ArrayRef<Attribute>().begin();
  return pImpl->begin(Slot);
}

AttributeList::iterator AttributeList::end(unsigned Slot) const {
  if (!pImpl)
    return ArrayRef<Attribute>().end();
  return pImpl->end(Slot);
}

//===----------------------------------------------------------------------===//
// AttributeList Introspection Methods
//===----------------------------------------------------------------------===//

unsigned AttributeList::getNumSlots() const {
  return pImpl ? pImpl->getNumSlots() : 0;
}

unsigned AttributeList::getSlotIndex(unsigned Slot) const {
  assert(pImpl && Slot < pImpl->getNumSlots() &&
         "Slot # out of range!");
  return pImpl->getSlotIndex(Slot);
}

AttributeList AttributeList::getSlotAttributes(unsigned Slot) const {
  assert(pImpl && Slot < pImpl->getNumSlots() &&
         "Slot # out of range!");
  return pImpl->getSlotAttributes(Slot);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void AttributeList::dump() const {
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
#endif

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

AttrBuilder::AttrBuilder(AttributeList AL, unsigned Index) {
  AttributeListImpl *pImpl = AL.pImpl;
  if (!pImpl) return;

  for (unsigned I = 0, E = pImpl->getNumSlots(); I != E; ++I) {
    if (pImpl->getSlotIndex(I) != Index) continue;

    for (AttributeListImpl::iterator II = pImpl->begin(I), IE = pImpl->end(I);
         II != IE; ++II)
      addAttribute(*II);

    break;
  }
}

AttrBuilder::AttrBuilder(AttributeSet AS) {
  if (AS.hasAttributes()) {
    for (const Attribute &A : AS)
      addAttribute(A);
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

AttrBuilder &AttrBuilder::removeAttributes(AttributeList A, uint64_t Index) {
  unsigned Slot = ~0U;
  for (unsigned I = 0, E = A.getNumSlots(); I != E; ++I)
    if (A.getSlotIndex(I) == Index) {
      Slot = I;
      break;
    }

  assert(Slot != ~0U && "Couldn't find index in AttributeList!");

  for (AttributeList::iterator I = A.begin(Slot), E = A.end(Slot); I != E;
       ++I) {
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

bool AttrBuilder::hasAttributes(AttributeList A, uint64_t Index) const {
  unsigned Slot = ~0U;
  for (unsigned I = 0, E = A.getNumSlots(); I != E; ++I)
    if (A.getSlotIndex(I) == Index) {
      Slot = I;
      break;
    }

  assert(Slot != ~0U && "Couldn't find the index!");

  for (AttributeList::iterator I = A.begin(Slot), E = A.end(Slot); I != E;
       ++I) {
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
  AttributeList OldSSPAttr =
      AttributeList::get(Caller.getContext(), AttributeList::FunctionIndex, B);

  if (Callee.hasFnAttribute(Attribute::StackProtectReq)) {
    Caller.removeAttributes(AttributeList::FunctionIndex, OldSSPAttr);
    Caller.addFnAttr(Attribute::StackProtectReq);
  } else if (Callee.hasFnAttribute(Attribute::StackProtectStrong) &&
             !Caller.hasFnAttribute(Attribute::StackProtectReq)) {
    Caller.removeAttributes(AttributeList::FunctionIndex, OldSSPAttr);
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
