//===-- Attributes.cpp - Implement AttributesList -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AttributesList class and Attribute utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Attributes.h"
#include "llvm/Type.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/ManagedStatic.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Attribute Function Definitions
//===----------------------------------------------------------------------===//

std::string Attribute::getAsString(Attributes Attrs) {
  std::string Result;
  if (Attrs & Attribute::ZExt)
    Result += "zeroext ";
  if (Attrs & Attribute::SExt)
    Result += "signext ";
  if (Attrs & Attribute::NoReturn)
    Result += "noreturn ";
  if (Attrs & Attribute::NoUnwind)
    Result += "nounwind ";
  if (Attrs & Attribute::InReg)
    Result += "inreg ";
  if (Attrs & Attribute::NoAlias)
    Result += "noalias ";
  if (Attrs & Attribute::NoCapture)
    Result += "nocapture ";
  if (Attrs & Attribute::StructRet)
    Result += "sret ";  
  if (Attrs & Attribute::ByVal)
    Result += "byval ";
  if (Attrs & Attribute::Nest)
    Result += "nest ";
  if (Attrs & Attribute::ReadNone)
    Result += "readnone ";
  if (Attrs & Attribute::ReadOnly)
    Result += "readonly ";
  if (Attrs & Attribute::OptimizeForSize)
    Result += "optsize ";
  if (Attrs & Attribute::NoInline)
    Result += "noinline ";
  if (Attrs & Attribute::AlwaysInline)
    Result += "alwaysinline ";
  if (Attrs & Attribute::StackProtect)
    Result += "ssp ";
  if (Attrs & Attribute::StackProtectReq)
    Result += "sspreq ";
  if (Attrs & Attribute::Alignment) {
    Result += "align ";
    Result += utostr(Attribute::getAlignmentFromAttrs(Attrs));
    Result += " ";
  }
  // Trim the trailing space.
  assert(!Result.empty() && "Unknown attribute!");
  Result.erase(Result.end()-1);
  return Result;
}

Attributes Attribute::typeIncompatible(const Type *Ty) {
  Attributes Incompatible = None;
  
  if (!Ty->isInteger())
    // Attributes that only apply to integers.
    Incompatible |= SExt | ZExt;
  
  if (!isa<PointerType>(Ty))
    // Attributes that only apply to pointers.
    Incompatible |= ByVal | Nest | NoAlias | StructRet | NoCapture;
  
  return Incompatible;
}

//===----------------------------------------------------------------------===//
// AttributeListImpl Definition
//===----------------------------------------------------------------------===//

namespace llvm {
class AttributeListImpl : public FoldingSetNode {
  unsigned RefCount;
  
  // AttributesList is uniqued, these should not be publicly available.
  void operator=(const AttributeListImpl &); // Do not implement
  AttributeListImpl(const AttributeListImpl &); // Do not implement
  ~AttributeListImpl();                        // Private implementation
public:
  SmallVector<AttributeWithIndex, 4> Attrs;
  
  AttributeListImpl(const AttributeWithIndex *Attr, unsigned NumAttrs)
    : Attrs(Attr, Attr+NumAttrs) {
    RefCount = 0;
  }
  
  void AddRef() { ++RefCount; }
  void DropRef() { if (--RefCount == 0) delete this; }
  
  void Profile(FoldingSetNodeID &ID) const {
    Profile(ID, &Attrs[0], Attrs.size());
  }
  static void Profile(FoldingSetNodeID &ID, const AttributeWithIndex *Attr,
                      unsigned NumAttrs) {
    for (unsigned i = 0; i != NumAttrs; ++i)
      ID.AddInteger(uint64_t(Attr[i].Attrs) << 32 | unsigned(Attr[i].Index));
  }
};
}

static ManagedStatic<FoldingSet<AttributeListImpl> > AttributesLists;

AttributeListImpl::~AttributeListImpl() {
  AttributesLists->RemoveNode(this);
}


AttrListPtr AttrListPtr::get(const AttributeWithIndex *Attrs, unsigned NumAttrs) {
  // If there are no attributes then return a null AttributesList pointer.
  if (NumAttrs == 0)
    return AttrListPtr();
  
#ifndef NDEBUG
  for (unsigned i = 0; i != NumAttrs; ++i) {
    assert(Attrs[i].Attrs != Attribute::None && 
           "Pointless attribute!");
    assert((!i || Attrs[i-1].Index < Attrs[i].Index) &&
           "Misordered AttributesList!");
  }
#endif
  
  // Otherwise, build a key to look up the existing attributes.
  FoldingSetNodeID ID;
  AttributeListImpl::Profile(ID, Attrs, NumAttrs);
  void *InsertPos;
  AttributeListImpl *PAL =
    AttributesLists->FindNodeOrInsertPos(ID, InsertPos);
  
  // If we didn't find any existing attributes of the same shape then
  // create a new one and insert it.
  if (!PAL) {
    PAL = new AttributeListImpl(Attrs, NumAttrs);
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


/// getAttributes - The attributes for the specified index are
/// returned.  Attributes for the result are denoted with Idx = 0.
/// Function notes are denoted with idx = ~0.
Attributes AttrListPtr::getAttributes(unsigned Idx) const {
  if (AttrList == 0) return Attribute::None;
  
  const SmallVector<AttributeWithIndex, 4> &Attrs = AttrList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e && Attrs[i].Index <= Idx; ++i)
    if (Attrs[i].Index == Idx)
      return Attrs[i].Attrs;
  return Attribute::None;
}

/// hasAttrSomewhere - Return true if the specified attribute is set for at
/// least one parameter or for the return value.
bool AttrListPtr::hasAttrSomewhere(Attributes Attr) const {
  if (AttrList == 0) return false;
  
  const SmallVector<AttributeWithIndex, 4> &Attrs = AttrList->Attrs;
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i)
    if (Attrs[i].Attrs & Attr)
      return true;
  return false;
}


AttrListPtr AttrListPtr::addAttr(unsigned Idx, Attributes Attrs) const {
  Attributes OldAttrs = getAttributes(Idx);
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't change a known alignment.
  Attributes OldAlign = OldAttrs & Attribute::Alignment;
  Attributes NewAlign = Attrs & Attribute::Alignment;
  assert((!OldAlign || !NewAlign || OldAlign == NewAlign) &&
         "Attempt to change alignment!");
#endif
  
  Attributes NewAttrs = OldAttrs | Attrs;
  if (NewAttrs == OldAttrs)
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
      Attrs |= OldAttrList[i].Attrs;
      ++i;
    }
    
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));
    
    // Copy attributes for arguments after this one.
    NewAttrList.insert(NewAttrList.end(), 
                       OldAttrList.begin()+i, OldAttrList.end());
  }
  
  return get(&NewAttrList[0], NewAttrList.size());
}

AttrListPtr AttrListPtr::removeAttr(unsigned Idx, Attributes Attrs) const {
#ifndef NDEBUG
  // FIXME it is not obvious how this should work for alignment.
  // For now, say we can't pass in alignment, which no current use does.
  assert(!(Attrs & Attribute::Alignment) && "Attempt to exclude alignment!");
#endif
  if (AttrList == 0) return AttrListPtr();
  
  Attributes OldAttrs = getAttributes(Idx);
  Attributes NewAttrs = OldAttrs & ~Attrs;
  if (NewAttrs == OldAttrs)
    return *this;

  SmallVector<AttributeWithIndex, 8> NewAttrList;
  const SmallVector<AttributeWithIndex, 4> &OldAttrList = AttrList->Attrs;
  unsigned i = 0, e = OldAttrList.size();
  
  // Copy attributes for arguments before this one.
  for (; i != e && OldAttrList[i].Index < Idx; ++i)
    NewAttrList.push_back(OldAttrList[i]);
  
  // If there are attributes already at this index, merge them in.
  assert(OldAttrList[i].Index == Idx && "Attribute isn't set?");
  Attrs = OldAttrList[i].Attrs & ~Attrs;
  ++i;
  if (Attrs)  // If any attributes left for this parameter, add them.
    NewAttrList.push_back(AttributeWithIndex::get(Idx, Attrs));
  
  // Copy attributes for arguments after this one.
  NewAttrList.insert(NewAttrList.end(), 
                     OldAttrList.begin()+i, OldAttrList.end());
  
  return get(&NewAttrList[0], NewAttrList.size());
}

void AttrListPtr::dump() const {
  cerr << "PAL[ ";
  for (unsigned i = 0; i < getNumSlots(); ++i) {
    const AttributeWithIndex &PAWI = getSlot(i);
    cerr << "{" << PAWI.Index << "," << PAWI.Attrs << "} ";
  }
  
  cerr << "]\n";
}
