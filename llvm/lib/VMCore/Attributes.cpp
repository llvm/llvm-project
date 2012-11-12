//===-- Attributes.cpp - Implement AttributesList -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Attributes, AttributeImpl, AttrBuilder,
// AttributeListImpl, and AttrListPtr classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Attributes.h"
#include "AttributesImpl.h"
#include "LLVMContextImpl.h"
#include "llvm/Type.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Attributes Implementation
//===----------------------------------------------------------------------===//

Attributes Attributes::get(LLVMContext &Context, ArrayRef<AttrVal> Vals) {
  AttrBuilder B;
  for (ArrayRef<AttrVal>::iterator I = Vals.begin(), E = Vals.end();
       I != E; ++I)
    B.addAttribute(*I);
  return Attributes::get(Context, B);
}

Attributes Attributes::get(LLVMContext &Context, AttrBuilder &B) {
  // If there are no attributes, return an empty Attributes class.
  if (!B.hasAttributes())
    return Attributes();

  // Otherwise, build a key to look up the existing attributes.
  LLVMContextImpl *pImpl = Context.pImpl;
  FoldingSetNodeID ID;
  ID.AddInteger(B.Raw());

  void *InsertPoint;
  AttributesImpl *PA = pImpl->AttrsSet.FindNodeOrInsertPos(ID, InsertPoint);

  if (!PA) {
    // If we didn't find any existing attributes of the same shape then create a
    // new one and insert it.
    PA = new AttributesImpl(B.Raw());
    pImpl->AttrsSet.InsertNode(PA, InsertPoint);
  }

  // Return the AttributesList that we found or created.
  return Attributes(PA);
}

bool Attributes::hasAttribute(AttrVal Val) const {
  return Attrs && Attrs->hasAttribute(Val);
}

bool Attributes::hasAttributes() const {
  return Attrs && Attrs->hasAttributes();
}

bool Attributes::hasAttributes(const Attributes &A) const {
  return Attrs && Attrs->hasAttributes(A);
}

/// This returns the alignment field of an attribute as a byte alignment value.
unsigned Attributes::getAlignment() const {
  if (!hasAttribute(Attributes::Alignment))
    return 0;
  return 1U << ((Attrs->getAlignment() >> 16) - 1);
}

/// This returns the stack alignment field of an attribute as a byte alignment
/// value.
unsigned Attributes::getStackAlignment() const {
  if (!hasAttribute(Attributes::StackAlignment))
    return 0;
  return 1U << ((Attrs->getStackAlignment() >> 26) - 1);
}

uint64_t Attributes::Raw() const {
  return Attrs ? Attrs->Raw() : 0;
}

Attributes Attributes::typeIncompatible(Type *Ty) {
  AttrBuilder Incompatible;

  if (!Ty->isIntegerTy())
    // Attributes that only apply to integers.
    Incompatible.addAttribute(Attributes::SExt)
      .addAttribute(Attributes::ZExt);

  if (!Ty->isPointerTy())
    // Attributes that only apply to pointers.
    Incompatible.addAttribute(Attributes::ByVal)
      .addAttribute(Attributes::Nest)
      .addAttribute(Attributes::NoAlias)
      .addAttribute(Attributes::NoCapture)
      .addAttribute(Attributes::StructRet);

  return Attributes::get(Ty->getContext(), Incompatible);
}

/// encodeLLVMAttributesForBitcode - This returns an integer containing an
/// encoding of all the LLVM attributes found in the given attribute bitset.
/// Any change to this encoding is a breaking change to bitcode compatibility.
uint64_t Attributes::encodeLLVMAttributesForBitcode(Attributes Attrs) {
  // FIXME: It doesn't make sense to store the alignment information as an
  // expanded out value, we should store it as a log2 value.  However, we can't
  // just change that here without breaking bitcode compatibility.  If this ever
  // becomes a problem in practice, we should introduce new tag numbers in the
  // bitcode file and have those tags use a more efficiently encoded alignment
  // field.

  // Store the alignment in the bitcode as a 16-bit raw value instead of a 5-bit
  // log2 encoded value. Shift the bits above the alignment up by 11 bits.
  uint64_t EncodedAttrs = Attrs.Raw() & 0xffff;
  if (Attrs.hasAttribute(Attributes::Alignment))
    EncodedAttrs |= Attrs.getAlignment() << 16;
  EncodedAttrs |= (Attrs.Raw() & (0xffffULL << 21)) << 11;
  return EncodedAttrs;
}

/// decodeLLVMAttributesForBitcode - This returns an attribute bitset containing
/// the LLVM attributes that have been decoded from the given integer.  This
/// function must stay in sync with 'encodeLLVMAttributesForBitcode'.
Attributes Attributes::decodeLLVMAttributesForBitcode(LLVMContext &C,
                                                      uint64_t EncodedAttrs) {
  // The alignment is stored as a 16-bit raw value from bits 31--16.  We shift
  // the bits above 31 down by 11 bits.
  unsigned Alignment = (EncodedAttrs & (0xffffULL << 16)) >> 16;
  assert((!Alignment || isPowerOf2_32(Alignment)) &&
         "Alignment must be a power of two.");

  AttrBuilder B(EncodedAttrs & 0xffff);
  if (Alignment)
    B.addAlignmentAttr(Alignment);
  B.addRawValue((EncodedAttrs & (0xffffULL << 32)) >> 11);
  return Attributes::get(C, B);
}

std::string Attributes::getAsString() const {
  std::string Result;
  if (hasAttribute(Attributes::ZExt))
    Result += "zeroext ";
  if (hasAttribute(Attributes::SExt))
    Result += "signext ";
  if (hasAttribute(Attributes::NoReturn))
    Result += "noreturn ";
  if (hasAttribute(Attributes::NoUnwind))
    Result += "nounwind ";
  if (hasAttribute(Attributes::UWTable))
    Result += "uwtable ";
  if (hasAttribute(Attributes::ReturnsTwice))
    Result += "returns_twice ";
  if (hasAttribute(Attributes::InReg))
    Result += "inreg ";
  if (hasAttribute(Attributes::NoAlias))
    Result += "noalias ";
  if (hasAttribute(Attributes::NoCapture))
    Result += "nocapture ";
  if (hasAttribute(Attributes::StructRet))
    Result += "sret ";
  if (hasAttribute(Attributes::ByVal))
    Result += "byval ";
  if (hasAttribute(Attributes::Nest))
    Result += "nest ";
  if (hasAttribute(Attributes::ReadNone))
    Result += "readnone ";
  if (hasAttribute(Attributes::ReadOnly))
    Result += "readonly ";
  if (hasAttribute(Attributes::OptimizeForSize))
    Result += "optsize ";
  if (hasAttribute(Attributes::NoInline))
    Result += "noinline ";
  if (hasAttribute(Attributes::InlineHint))
    Result += "inlinehint ";
  if (hasAttribute(Attributes::AlwaysInline))
    Result += "alwaysinline ";
  if (hasAttribute(Attributes::StackProtect))
    Result += "ssp ";
  if (hasAttribute(Attributes::StackProtectReq))
    Result += "sspreq ";
  if (hasAttribute(Attributes::NoRedZone))
    Result += "noredzone ";
  if (hasAttribute(Attributes::NoImplicitFloat))
    Result += "noimplicitfloat ";
  if (hasAttribute(Attributes::Naked))
    Result += "naked ";
  if (hasAttribute(Attributes::NonLazyBind))
    Result += "nonlazybind ";
  if (hasAttribute(Attributes::AddressSafety))
    Result += "address_safety ";
  if (hasAttribute(Attributes::MinSize))
    Result += "minsize ";
  if (hasAttribute(Attributes::StackAlignment)) {
    Result += "alignstack(";
    Result += utostr(getStackAlignment());
    Result += ") ";
  }
  if (hasAttribute(Attributes::Alignment)) {
    Result += "align ";
    Result += utostr(getAlignment());
    Result += " ";
  }
  // Trim the trailing space.
  assert(!Result.empty() && "Unknown attribute!");
  Result.erase(Result.end()-1);
  return Result;
}

//===----------------------------------------------------------------------===//
// AttrBuilder Implementation
//===----------------------------------------------------------------------===//

AttrBuilder &AttrBuilder::addAttribute(Attributes::AttrVal Val){
  Bits |= AttributesImpl::getAttrMask(Val);
  return *this;
}

AttrBuilder &AttrBuilder::addRawValue(uint64_t Val) {
  Bits |= Val;
  return *this;
}

AttrBuilder &AttrBuilder::addAlignmentAttr(unsigned Align) {
  if (Align == 0) return *this;
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x40000000 && "Alignment too large.");
  Bits |= (Log2_32(Align) + 1) << 16;
  return *this;
}
AttrBuilder &AttrBuilder::addStackAlignmentAttr(unsigned Align){
  // Default alignment, allow the target to define how to align it.
  if (Align == 0) return *this;
  assert(isPowerOf2_32(Align) && "Alignment must be a power of two.");
  assert(Align <= 0x100 && "Alignment too large.");
  Bits |= (Log2_32(Align) + 1) << 26;
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(Attributes::AttrVal Val) {
  Bits &= ~AttributesImpl::getAttrMask(Val);
  return *this;
}

AttrBuilder &AttrBuilder::addAttributes(const Attributes &A) {
  Bits |= A.Raw();
  return *this;
}

AttrBuilder &AttrBuilder::removeAttributes(const Attributes &A){
  Bits &= ~A.Raw();
  return *this;
}

bool AttrBuilder::hasAttribute(Attributes::AttrVal A) const {
  return Bits & AttributesImpl::getAttrMask(A);
}

bool AttrBuilder::hasAttributes() const {
  return Bits != 0;
}
bool AttrBuilder::hasAttributes(const Attributes &A) const {
  return Bits & A.Raw();
}
bool AttrBuilder::hasAlignmentAttr() const {
  return Bits & AttributesImpl::getAttrMask(Attributes::Alignment);
}

uint64_t AttrBuilder::getAlignment() const {
  if (!hasAlignmentAttr())
    return 0;
  return 1U <<
    (((Bits & AttributesImpl::getAttrMask(Attributes::Alignment)) >> 16) - 1);
}

uint64_t AttrBuilder::getStackAlignment() const {
  if (!hasAlignmentAttr())
    return 0;
  return 1U <<
    (((Bits & AttributesImpl::getAttrMask(Attributes::StackAlignment))>>26)-1);
}

//===----------------------------------------------------------------------===//
// AttributeImpl Definition
//===----------------------------------------------------------------------===//

uint64_t AttributesImpl::getAttrMask(uint64_t Val) {
  switch (Val) {
  case Attributes::None:            return 0;
  case Attributes::ZExt:            return 1 << 0;
  case Attributes::SExt:            return 1 << 1;
  case Attributes::NoReturn:        return 1 << 2;
  case Attributes::InReg:           return 1 << 3;
  case Attributes::StructRet:       return 1 << 4;
  case Attributes::NoUnwind:        return 1 << 5;
  case Attributes::NoAlias:         return 1 << 6;
  case Attributes::ByVal:           return 1 << 7;
  case Attributes::Nest:            return 1 << 8;
  case Attributes::ReadNone:        return 1 << 9;
  case Attributes::ReadOnly:        return 1 << 10;
  case Attributes::NoInline:        return 1 << 11;
  case Attributes::AlwaysInline:    return 1 << 12;
  case Attributes::OptimizeForSize: return 1 << 13;
  case Attributes::StackProtect:    return 1 << 14;
  case Attributes::StackProtectReq: return 1 << 15;
  case Attributes::Alignment:       return 31 << 16;
  case Attributes::NoCapture:       return 1 << 21;
  case Attributes::NoRedZone:       return 1 << 22;
  case Attributes::NoImplicitFloat: return 1 << 23;
  case Attributes::Naked:           return 1 << 24;
  case Attributes::InlineHint:      return 1 << 25;
  case Attributes::StackAlignment:  return 7 << 26;
  case Attributes::ReturnsTwice:    return 1 << 29;
  case Attributes::UWTable:         return 1 << 30;
  case Attributes::NonLazyBind:     return 1U << 31;
  case Attributes::AddressSafety:   return 1ULL << 32;
  case Attributes::MinSize:         return 1ULL << 33;
  }
  llvm_unreachable("Unsupported attribute type");
}

bool AttributesImpl::hasAttribute(uint64_t A) const {
  return (Bits & getAttrMask(A)) != 0;
}

bool AttributesImpl::hasAttributes() const {
  return Bits != 0;
}

bool AttributesImpl::hasAttributes(const Attributes &A) const {
  return Bits & A.Raw();        // FIXME: Raw() won't work here in the future.
}

uint64_t AttributesImpl::getAlignment() const {
  return Bits & getAttrMask(Attributes::Alignment);
}

uint64_t AttributesImpl::getStackAlignment() const {
  return Bits & getAttrMask(Attributes::StackAlignment);
}

//===----------------------------------------------------------------------===//
// AttributeListImpl Definition
//===----------------------------------------------------------------------===//

namespace llvm {
  class AttributeListImpl;
}

static ManagedStatic<FoldingSet<AttributeListImpl> > AttributesLists;

namespace llvm {
static ManagedStatic<sys::SmartMutex<true> > ALMutex;

class AttributeListImpl : public FoldingSetNode {
  sys::cas_flag RefCount;

  // AttributesList is uniqued, these should not be publicly available.
  void operator=(const AttributeListImpl &) LLVM_DELETED_FUNCTION;
  AttributeListImpl(const AttributeListImpl &) LLVM_DELETED_FUNCTION;
  ~AttributeListImpl();                        // Private implementation
public:
  SmallVector<AttributeWithIndex, 4> Attrs;

  AttributeListImpl(ArrayRef<AttributeWithIndex> attrs)
    : Attrs(attrs.begin(), attrs.end()) {
    RefCount = 0;
  }

  void AddRef() {
    sys::SmartScopedLock<true> Lock(*ALMutex);
    ++RefCount;
  }
  void DropRef() {
    sys::SmartScopedLock<true> Lock(*ALMutex);
    if (!AttributesLists.isConstructed())
      return;
    sys::cas_flag new_val = --RefCount;
    if (new_val == 0)
      delete this;
  }

  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, Attrs);
  }
  static void Profile(FoldingSetNodeID &ID, ArrayRef<AttributeWithIndex> Attrs){
    for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
      ID.AddInteger(Attrs[i].Attrs.Raw());
      ID.AddInteger(Attrs[i].Index);
    }
  }
};

} // end llvm namespace

AttributeListImpl::~AttributeListImpl() {
  // NOTE: Lock must be acquired by caller.
  AttributesLists->RemoveNode(this);
}

AttrListPtr AttrListPtr::get(ArrayRef<AttributeWithIndex> Attrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (Attrs.empty())
    return AttrListPtr();

#ifndef NDEBUG
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
    assert(Attrs[i].Attrs.hasAttributes() &&
           "Pointless attribute!");
    assert((!i || Attrs[i-1].Index < Attrs[i].Index) &&
           "Misordered AttributesList!");
  }
#endif

  // Otherwise, build a key to look up the existing attributes.
  FoldingSetNodeID ID;
  AttributeListImpl::Profile(ID, Attrs);
  void *InsertPos;

  sys::SmartScopedLock<true> Lock(*ALMutex);

  AttributeListImpl *PAL =
    AttributesLists->FindNodeOrInsertPos(ID, InsertPos);

  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PAL) {
    PAL = new AttributeListImpl(Attrs);
    AttributesLists->InsertNode(PAL, InsertPos);
  }

  // Return the AttributesList that we found or created.
  return AttrListPtr(PAL);
}

//===----------------------------------------------------------------------===//
// AttrListPtr Method Implementations
//===----------------------------------------------------------------------===//

AttrListPtr::AttrListPtr(AttributeListImpl *LI) : AttrList(LI) {
  if (LI) LI->AddRef();
}

AttrListPtr::AttrListPtr(const AttrListPtr &P) : AttrList(P.AttrList) {
  if (AttrList) AttrList->AddRef();
}

const AttrListPtr &AttrListPtr::operator=(const AttrListPtr &RHS) {
  sys::SmartScopedLock<true> Lock(*ALMutex);
  if (AttrList == RHS.AttrList) return *this;
  if (AttrList) AttrList->DropRef();
  AttrList = RHS.AttrList;
  if (AttrList) AttrList->AddRef();
  return *this;
}

AttrListPtr::~AttrListPtr() {
  if (AttrList) AttrList->DropRef();
}

/// getNumSlots - Return the number of slots used in this attribute list.
/// This is the number of arguments that have an attribute set on them
/// (including the function itself).
unsigned AttrListPtr::getNumSlots() const {
  return AttrList ? AttrList->Attrs.size() : 0;
}

/// getSlot - Return the AttributeWithIndex at the specified slot.  This
/// holds a number plus a set of attributes.
const AttributeWithIndex &AttrListPtr::getSlot(unsigned Slot) const {
  assert(AttrList && Slot < AttrList->Attrs.size() && "Slot # out of range!");
  return AttrList->Attrs[Slot];
}

/// getAttributes - The attributes for the specified index are returned.
/// Attributes for the result are denoted with Idx = 0.  Function notes are
/// denoted with idx = ~0.
Attributes AttrListPtr::getAttributes(unsigned Idx) const {
  if (AttrList == 0) return Attributes();

  const SmallVector<AttributeWithIndex, 4> &Attrs = AttrList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e && Attrs[i].Index <= Idx; ++i)
    if (Attrs[i].Index == Idx)
      return Attrs[i].Attrs;

  return Attributes();
}

/// hasAttrSomewhere - Return true if the specified attribute is set for at
/// least one parameter or for the return value.
bool AttrListPtr::hasAttrSomewhere(Attributes::AttrVal Attr) const {
  if (AttrList == 0) return false;

  const SmallVector<AttributeWithIndex, 4> &Attrs = AttrList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i)
    if (Attrs[i].Attrs.hasAttribute(Attr))
      return true;
  return false;
}

unsigned AttrListPtr::getNumAttrs() const {
  return AttrList ? AttrList->Attrs.size() : 0;
}

Attributes &AttrListPtr::getAttributesAtIndex(unsigned i) const {
  assert(AttrList && "Trying to get an attribute from an empty list!");
  assert(i < AttrList->Attrs.size() && "Index out of range!");
  return AttrList->Attrs[i].Attrs;
}

AttrListPtr AttrListPtr::addAttr(LLVMContext &C, unsigned Idx,
                                 Attributes Attrs) const {
  Attributes OldAttrs = getAttributes(Idx);
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't change a known alignment.
  unsigned OldAlign = OldAttrs.getAlignment();
  unsigned NewAlign = Attrs.getAlignment();
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif

  AttrBuilder NewAttrs =
    AttrBuilder(OldAttrs).addAttributes(Attrs);
  if (NewAttrs == AttrBuilder(OldAttrs))
    return *this;

  SmallVector<AttributeWithIndex, 8> NewAttrList;
  if (AttrList == 0)
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));
  else {
    const SmallVector<AttributeWithIndex, 4> &OldAttrList = AttrList->Attrs;
    unsigned i = 0, e = OldAttrList.size();
    // Copy attributes for arguments before this one.
    for (; i != e && OldAttrList[i].Index < Idx; ++i)
      NewAttrList.push_back(OldAttrList[i]);

    // If there are attributes already at this index, merge them in.
    if (i != e && OldAttrList[i].Index == Idx) {
      Attrs =
        Attributes::get(C, AttrBuilder(Attrs).
                        addAttributes(OldAttrList[i].Attrs));
      ++i;
    }

    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));

    // Copy attributes for arguments after this one.
    NewAttrList.insert(NewAttrList.end(),
                       OldAttrList.begin()+i, OldAttrList.end());
  }

  return get(NewAttrList);
}

AttrListPtr AttrListPtr::removeAttr(LLVMContext &C, unsigned Idx,
                                    Attributes Attrs) const {
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!Attrs.hasAttribute(Attributes::Alignment) &&
         "Attempt to exclude alignment!");
#endif
  if (AttrList == 0) return AttrListPtr();

  Attributes OldAttrs = getAttributes(Idx);
  AttrBuilder NewAttrs =
    AttrBuilder(OldAttrs).removeAttributes(Attrs);
  if (NewAttrs == AttrBuilder(OldAttrs))
    return *this;

  SmallVector<AttributeWithIndex, 8> NewAttrList;
  const SmallVector<AttributeWithIndex, 4> &OldAttrList = AttrList->Attrs;
  unsigned i = 0, e = OldAttrList.size();

  // Copy attributes for arguments before this one.
  for (; i != e && OldAttrList[i].Index < Idx; ++i)
    NewAttrList.push_back(OldAttrList[i]);

  // If there are attributes already at this index, merge them in.
  assert(OldAttrList[i].Index == Idx && "Attribute isn't set?");
  Attrs = Attributes::get(C, AttrBuilder(OldAttrList[i].Attrs).
                          removeAttributes(Attrs));
  ++i;
  if (Attrs.hasAttributes()) // If any attributes left for this param, add them.
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));

  // Copy attributes for arguments after this one.
  NewAttrList.insert(NewAttrList.end(),
                     OldAttrList.begin()+i, OldAttrList.end());

  return get(NewAttrList);
}

void AttrListPtr::dump() const {
  dbgs() << "PAL[ ";
  for (unsigned i = 0; i < getNumSlots(); ++i) {
    const AttributeWithIndex &PAWI = getSlot(i);
    dbgs() << "{" << PAWI.Index << "," << PAWI.Attrs.getAsString() << "} ";
  }

  dbgs() << "]\n";
}
