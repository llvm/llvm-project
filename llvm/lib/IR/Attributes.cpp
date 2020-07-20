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

Attribute Attribute::getWithByRefType(LLVMContext &Context, Type *Ty) {
  return get(Context, ByRef, Ty);
}

Attribute Attribute::getWithPreallocatedType(LLVMContext &Context, Type *Ty) {
  return get(Context, Preallocated, Ty);
}

Attribute
Attribute::getWithAllocSizeArgs(LLVMContext &Context, unsigned ElemSizeArg,
                                const Optional<unsigned> &NumElemsArg) {
  assert(!(ElemSizeArg == 0 && NumElemsArg && *NumElemsArg == 0) &&
         "Invalid allocsize arguments -- given allocsize(0, 0)");
  return get(Context, AllocSize, packAllocSizeArgs(ElemSizeArg, NumElemsArg));
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

bool Attribute::doesAttrKindHaveArgument(Attribute::AttrKind AttrKind) {
  return AttrKind == Attribute::Alignment ||
         AttrKind == Attribute::StackAlignment ||
         AttrKind == Attribute::Dereferenceable ||
         AttrKind == Attribute::AllocSize ||
         AttrKind == Attribute::DereferenceableOrNull;
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

std::string Attribute::getAsString(bool InAttrGrp) const {
  if (!pImpl) return {};

  if (hasAttribute(Attribute::SanitizeAddress))
    return "sanitize_address";
  if (hasAttribute(Attribute::SanitizeHWAddress))
    return "sanitize_hwaddress";
  if (hasAttribute(Attribute::SanitizeMemTag))
    return "sanitize_memtag";
  if (hasAttribute(Attribute::AlwaysInline))
    return "alwaysinline";
  if (hasAttribute(Attribute::ArgMemOnly))
    return "argmemonly";
  if (hasAttribute(Attribute::Builtin))
    return "builtin";
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
  if (hasAttribute(Attribute::NoFree))
    return "nofree";
  if (hasAttribute(Attribute::NoImplicitFloat))
    return "noimplicitfloat";
  if (hasAttribute(Attribute::NoInline))
    return "noinline";
  if (hasAttribute(Attribute::NonLazyBind))
    return "nonlazybind";
  if (hasAttribute(Attribute::NoMerge))
    return "nomerge";
  if (hasAttribute(Attribute::NonNull))
    return "nonnull";
  if (hasAttribute(Attribute::NoRedZone))
    return "noredzone";
  if (hasAttribute(Attribute::NoReturn))
    return "noreturn";
  if (hasAttribute(Attribute::NoSync))
    return "nosync";
  if (hasAttribute(Attribute::NullPointerIsValid))
    return "null_pointer_is_valid";
  if (hasAttribute(Attribute::WillReturn))
    return "willreturn";
  if (hasAttribute(Attribute::NoCfCheck))
    return "nocf_check";
  if (hasAttribute(Attribute::NoRecurse))
    return "norecurse";
  if (hasAttribute(Attribute::NoUnwind))
    return "nounwind";
  if (hasAttribute(Attribute::OptForFuzzing))
    return "optforfuzzing";
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
  if (hasAttribute(Attribute::SpeculativeLoadHardening))
    return "speculative_load_hardening";
  if (hasAttribute(Attribute::Speculatable))
    return "speculatable";
  if (hasAttribute(Attribute::StackProtect))
    return "ssp";
  if (hasAttribute(Attribute::StackProtectReq))
    return "sspreq";
  if (hasAttribute(Attribute::StackProtectStrong))
    return "sspstrong";
  if (hasAttribute(Attribute::SafeStack))
    return "safestack";
  if (hasAttribute(Attribute::ShadowCallStack))
    return "shadowcallstack";
  if (hasAttribute(Attribute::StrictFP))
    return "strictfp";
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
  if (hasAttribute(Attribute::ImmArg))
    return "immarg";
  if (hasAttribute(Attribute::NoUndef))
    return "noundef";

  if (hasAttribute(Attribute::ByVal)) {
    std::string Result;
    Result += "byval";
    if (Type *Ty = getValueAsType()) {
      raw_string_ostream OS(Result);
      Result += '(';
      Ty->print(OS, false, true);
      OS.flush();
      Result += ')';
    }
    return Result;
  }

  const bool IsByRef = hasAttribute(Attribute::ByRef);
  if (IsByRef || hasAttribute(Attribute::Preallocated)) {
    std::string Result = IsByRef ? "byref" : "preallocated";
    raw_string_ostream OS(Result);
    Result += '(';
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

bool Attribute::operator<(Attribute A) const {
  if (!pImpl && !A.pImpl) return false;
  if (!pImpl) return true;
  if (!A.pImpl) return false;
  return *pImpl < *A.pImpl;
}

void Attribute::Profile(FoldingSetNodeID &ID) const {
  ID.AddPointer(pImpl);
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
  if (isEnumAttribute()) {
    if (AI.isEnumAttribute()) return getKindAsEnum() < AI.getKindAsEnum();
    if (AI.isIntAttribute()) return true;
    if (AI.isStringAttribute()) return true;
    if (AI.isTypeAttribute()) return true;
  }

  if (isTypeAttribute()) {
    if (AI.isEnumAttribute()) return false;
    if (AI.isTypeAttribute()) {
      assert(getKindAsEnum() != AI.getKindAsEnum() &&
             "Comparison of types would be unstable");
      return getKindAsEnum() < AI.getKindAsEnum();
    }
    if (AI.isIntAttribute()) return true;
    if (AI.isStringAttribute()) return true;
  }

  if (isIntAttribute()) {
    if (AI.isEnumAttribute()) return false;
    if (AI.isTypeAttribute()) return false;
    if (AI.isIntAttribute()) {
      if (getKindAsEnum() == AI.getKindAsEnum())
        return getValueAsInt() < AI.getValueAsInt();
      return getKindAsEnum() < AI.getKindAsEnum();
    }
    if (AI.isStringAttribute()) return true;
  }

  assert(isStringAttribute());
  if (AI.isEnumAttribute()) return false;
  if (AI.isTypeAttribute()) return false;
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
  return SetNode ? SetNode->getByRefType() : nullptr;
}

Type *AttributeSet::getByValType() const {
  return SetNode ? SetNode->getByValType() : nullptr;
}

Type *AttributeSet::getPreallocatedType() const {
  return SetNode ? SetNode->getPreallocatedType() : nullptr;
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
    switch (Kind) {
    case Attribute::ByVal:
      Attr = Attribute::getWithByValType(C, B.getByValType());
      break;
    case Attribute::ByRef:
      Attr = Attribute::getWithByRefType(C, B.getByRefType());
      break;
    case Attribute::Preallocated:
      Attr = Attribute::getWithPreallocatedType(C, B.getPreallocatedType());
      break;
    case Attribute::Alignment:
      assert(B.getAlignment() && "Alignment must be set");
      Attr = Attribute::getWithAlignment(C, *B.getAlignment());
      break;
    case Attribute::StackAlignment:
      assert(B.getStackAlignment() && "StackAlignment must be set");
      Attr = Attribute::getWithStackAlignment(C, *B.getStackAlignment());
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

Type *AttributeSetNode::getByValType() const {
  if (auto A = findEnumAttribute(Attribute::ByVal))
    return A->getValueAsType();
  return nullptr;
}

Type *AttributeSetNode::getByRefType() const {
  if (auto A = findEnumAttribute(Attribute::ByRef))
    return A->getValueAsType();
  return nullptr;
}

Type *AttributeSetNode::getPreallocatedType() const {
  if (auto A = findEnumAttribute(Attribute::Preallocated))
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
  assert(llvm::none_of(Attrs,
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
    AttrSets.insert(AttrSets.end(), ArgAttrs.begin(), ArgAttrs.end());
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

AttributeList AttributeList::addAttribute(LLVMContext &C, unsigned Index,
                                          Attribute::AttrKind Kind) const {
  if (hasAttribute(Index, Kind)) return *this;
  AttrBuilder B;
  B.addAttribute(Kind);
  return addAttributes(C, Index, B);
}

AttributeList AttributeList::addAttribute(LLVMContext &C, unsigned Index,
                                          StringRef Kind,
                                          StringRef Value) const {
  AttrBuilder B;
  B.addAttribute(Kind, Value);
  return addAttributes(C, Index, B);
}

AttributeList AttributeList::addAttribute(LLVMContext &C, unsigned Index,
                                          Attribute A) const {
  AttrBuilder B;
  B.addAttribute(A);
  return addAttributes(C, Index, B);
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
  const MaybeAlign OldAlign = getAttributes(Index).getAlignment();
  const MaybeAlign NewAlign = B.getAlignment();
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  Index = attrIdxToArrayIdx(Index);
  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  if (Index >= AttrSets.size())
    AttrSets.resize(Index + 1);

  AttrBuilder Merged(AttrSets[Index]);
  Merged.merge(B);
  AttrSets[Index] = AttributeSet::get(C, Merged);

  return getImpl(C, AttrSets);
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

AttributeList AttributeList::removeAttribute(LLVMContext &C, unsigned Index,
                                             Attribute::AttrKind Kind) const {
  if (!hasAttribute(Index, Kind)) return *this;

  Index = attrIdxToArrayIdx(Index);
  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  assert(Index < AttrSets.size());

  AttrSets[Index] = AttrSets[Index].removeAttribute(C, Kind);

  return getImpl(C, AttrSets);
}

AttributeList AttributeList::removeAttribute(LLVMContext &C, unsigned Index,
                                             StringRef Kind) const {
  if (!hasAttribute(Index, Kind)) return *this;

  Index = attrIdxToArrayIdx(Index);
  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  assert(Index < AttrSets.size());

  AttrSets[Index] = AttrSets[Index].removeAttribute(C, Kind);

  return getImpl(C, AttrSets);
}

AttributeList
AttributeList::removeAttributes(LLVMContext &C, unsigned Index,
                                const AttrBuilder &AttrsToRemove) const {
  if (!pImpl)
    return {};

  Index = attrIdxToArrayIdx(Index);
  SmallVector<AttributeSet, 4> AttrSets(this->begin(), this->end());
  if (Index >= AttrSets.size())
    AttrSets.resize(Index + 1);

  AttrSets[Index] = AttrSets[Index].removeAttributes(C, AttrsToRemove);

  return getImpl(C, AttrSets);
}

AttributeList AttributeList::removeAttributes(LLVMContext &C,
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

AttributeList AttributeList::addDereferenceableAttr(LLVMContext &C,
                                                    unsigned Index,
                                                    uint64_t Bytes) const {
  AttrBuilder B;
  B.addDereferenceableAttr(Bytes);
  return addAttributes(C, Index, B);
}

AttributeList
AttributeList::addDereferenceableOrNullAttr(LLVMContext &C, unsigned Index,
                                            uint64_t Bytes) const {
  AttrBuilder B;
  B.addDereferenceableOrNullAttr(Bytes);
  return addAttributes(C, Index, B);
}

AttributeList
AttributeList::addAllocSizeAttr(LLVMContext &C, unsigned Index,
                                unsigned ElemSizeArg,
                                const Optional<unsigned> &NumElemsArg) {
  AttrBuilder B;
  B.addAllocSizeAttr(ElemSizeArg, NumElemsArg);
  return addAttributes(C, Index, B);
}

//===----------------------------------------------------------------------===//
// AttributeList Accessor Methods
//===----------------------------------------------------------------------===//

AttributeSet AttributeList::getParamAttributes(unsigned ArgNo) const {
  return getAttributes(ArgNo + FirstArgIndex);
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
  return hasAttribute(ArgNo + FirstArgIndex, Kind);
}

bool AttributeList::hasAttrSomewhere(Attribute::AttrKind Attr,
                                     unsigned *Index) const {
  return pImpl && pImpl->hasAttrSomewhere(Attr, Index);
}

Attribute AttributeList::getAttribute(unsigned Index,
                                      Attribute::AttrKind Kind) const {
  return getAttributes(Index).getAttribute(Kind);
}

Attribute AttributeList::getAttribute(unsigned Index, StringRef Kind) const {
  return getAttributes(Index).getAttribute(Kind);
}

MaybeAlign AttributeList::getRetAlignment() const {
  return getAttributes(ReturnIndex).getAlignment();
}

MaybeAlign AttributeList::getParamAlignment(unsigned ArgNo) const {
  return getAttributes(ArgNo + FirstArgIndex).getAlignment();
}

Type *AttributeList::getParamByValType(unsigned Index) const {
  return getAttributes(Index+FirstArgIndex).getByValType();
}

Type *AttributeList::getParamByRefType(unsigned Index) const {
  return getAttributes(Index + FirstArgIndex).getByRefType();
}

Type *AttributeList::getParamPreallocatedType(unsigned Index) const {
  return getAttributes(Index + FirstArgIndex).getPreallocatedType();
}

MaybeAlign AttributeList::getStackAlignment(unsigned Index) const {
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
  Index = attrIdxToArrayIdx(Index);
  if (!pImpl || Index >= getNumAttrSets())
    return {};
  return pImpl->begin()[Index];
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

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void AttributeList::dump() const {
  dbgs() << "PAL[\n";

  for (unsigned i = index_begin(), e = index_end(); i != e; ++i) {
    if (getAttributes(i).hasAttributes())
      dbgs() << "  { " << i << " => " << getAsString(i) << " }\n";
  }

  dbgs() << "]\n";
}
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
  Alignment.reset();
  StackAlignment.reset();
  DerefBytes = DerefOrNullBytes = 0;
  AllocSizeArgs = 0;
  ByValType = nullptr;
  ByRefType = nullptr;
  PreallocatedType = nullptr;
}

AttrBuilder &AttrBuilder::addAttribute(Attribute::AttrKind Val) {
  assert((unsigned)Val < Attribute::EndAttrKinds && "Attribute out of range!");
  assert(!Attribute::doesAttrKindHaveArgument(Val) &&
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
  else if (Kind == Attribute::ByVal)
    ByValType = Attr.getValueAsType();
  else if (Kind == Attribute::ByRef)
    ByRefType = Attr.getValueAsType();
  else if (Kind == Attribute::Preallocated)
    PreallocatedType = Attr.getValueAsType();
  else if (Kind == Attribute::Dereferenceable)
    DerefBytes = Attr.getDereferenceableBytes();
  else if (Kind == Attribute::DereferenceableOrNull)
    DerefOrNullBytes = Attr.getDereferenceableOrNullBytes();
  else if (Kind == Attribute::AllocSize)
    AllocSizeArgs = Attr.getValueAsInt();
  return *this;
}

AttrBuilder &AttrBuilder::addAttribute(StringRef A, StringRef V) {
  TargetDepAttrs[std::string(A)] = std::string(V);
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(Attribute::AttrKind Val) {
  assert((unsigned)Val < Attribute::EndAttrKinds && "Attribute out of range!");
  Attrs[Val] = false;

  if (Val == Attribute::Alignment)
    Alignment.reset();
  else if (Val == Attribute::StackAlignment)
    StackAlignment.reset();
  else if (Val == Attribute::ByVal)
    ByValType = nullptr;
  else if (Val == Attribute::ByRef)
    ByRefType = nullptr;
  else if (Val == Attribute::Preallocated)
    PreallocatedType = nullptr;
  else if (Val == Attribute::Dereferenceable)
    DerefBytes = 0;
  else if (Val == Attribute::DereferenceableOrNull)
    DerefOrNullBytes = 0;
  else if (Val == Attribute::AllocSize)
    AllocSizeArgs = 0;

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

std::pair<unsigned, Optional<unsigned>> AttrBuilder::getAllocSizeArgs() const {
  return unpackAllocSizeArgs(AllocSizeArgs);
}

AttrBuilder &AttrBuilder::addAlignmentAttr(MaybeAlign Align) {
  if (!Align)
    return *this;

  assert(*Align <= llvm::Value::MaximumAlignment && "Alignment too large.");

  Attrs[Attribute::Alignment] = true;
  Alignment = Align;
  return *this;
}

AttrBuilder &AttrBuilder::addStackAlignmentAttr(MaybeAlign Align) {
  // Default alignment, allow the target to define how to align it.
  if (!Align)
    return *this;

  assert(*Align <= 0x100 && "Alignment too large.");

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

AttrBuilder &AttrBuilder::addByValAttr(Type *Ty) {
  Attrs[Attribute::ByVal] = true;
  ByValType = Ty;
  return *this;
}

AttrBuilder &AttrBuilder::addByRefAttr(Type *Ty) {
  Attrs[Attribute::ByRef] = true;
  ByRefType = Ty;
  return *this;
}

AttrBuilder &AttrBuilder::addPreallocatedAttr(Type *Ty) {
  Attrs[Attribute::Preallocated] = true;
  PreallocatedType = Ty;
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

  if (!ByValType)
    ByValType = B.ByValType;

  if (!ByRefType)
    ByRefType = B.ByRefType;

  if (!PreallocatedType)
    PreallocatedType = B.PreallocatedType;

  Attrs |= B.Attrs;

  for (const auto &I : B.td_attrs())
    TargetDepAttrs[I.first] = I.second;

  return *this;
}

AttrBuilder &AttrBuilder::remove(const AttrBuilder &B) {
  // FIXME: What if both have alignments, but they don't match?!
  if (B.Alignment)
    Alignment.reset();

  if (B.StackAlignment)
    StackAlignment.reset();

  if (B.DerefBytes)
    DerefBytes = 0;

  if (B.DerefOrNullBytes)
    DerefOrNullBytes = 0;

  if (B.AllocSizeArgs)
    AllocSizeArgs = 0;

  if (B.ByValType)
    ByValType = nullptr;

  if (B.ByRefType)
    ByRefType = nullptr;

  if (B.PreallocatedType)
    PreallocatedType = nullptr;

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
         DerefBytes == B.DerefBytes && ByValType == B.ByValType &&
         ByRefType == B.ByRefType && PreallocatedType == B.PreallocatedType;
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
        .addDereferenceableAttr(1)       // the int here is ignored
        .addDereferenceableOrNullAttr(1) // the int here is ignored
        .addAttribute(Attribute::ReadNone)
        .addAttribute(Attribute::ReadOnly)
        .addAttribute(Attribute::StructRet)
        .addAttribute(Attribute::InAlloca)
        .addPreallocatedAttr(Ty)
        .addByValAttr(Ty)
        .addByRefAttr(Ty);

  return Incompatible;
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
  if (Callee.hasFnAttribute("stack-probe-size")) {
    uint64_t CalleeStackProbeSize;
    Callee.getFnAttribute("stack-probe-size")
          .getValueAsString()
          .getAsInteger(0, CalleeStackProbeSize);
    if (Caller.hasFnAttribute("stack-probe-size")) {
      uint64_t CallerStackProbeSize;
      Caller.getFnAttribute("stack-probe-size")
            .getValueAsString()
            .getAsInteger(0, CallerStackProbeSize);
      if (CallerStackProbeSize > CalleeStackProbeSize) {
        Caller.addFnAttr(Callee.getFnAttribute("stack-probe-size"));
      }
    } else {
      Caller.addFnAttr(Callee.getFnAttribute("stack-probe-size"));
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
  if (Caller.hasFnAttribute("min-legal-vector-width")) {
    if (Callee.hasFnAttribute("min-legal-vector-width")) {
      uint64_t CallerVectorWidth;
      Caller.getFnAttribute("min-legal-vector-width")
            .getValueAsString()
            .getAsInteger(0, CallerVectorWidth);
      uint64_t CalleeVectorWidth;
      Callee.getFnAttribute("min-legal-vector-width")
            .getValueAsString()
            .getAsInteger(0, CalleeVectorWidth);
      if (CallerVectorWidth < CalleeVectorWidth)
        Caller.addFnAttr(Callee.getFnAttribute("min-legal-vector-width"));
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

void AttributeFuncs::mergeAttributesForInlining(Function &Caller,
                                                const Function &Callee) {
  mergeFnAttrs(Caller, Callee);
}
