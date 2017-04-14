//===- UDTLayout.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/UDTLayout.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBaseClass.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"

#include <utility>

using namespace llvm;
using namespace llvm::pdb;

static std::unique_ptr<PDBSymbol> getSymbolType(const PDBSymbol &Symbol) {
  const IPDBSession &Session = Symbol.getSession();
  const IPDBRawSymbol &RawSymbol = Symbol.getRawSymbol();
  uint32_t TypeId = RawSymbol.getTypeId();
  return Session.getSymbolById(TypeId);
}

static uint32_t getTypeLength(const PDBSymbol &Symbol) {
  auto SymbolType = getSymbolType(Symbol);
  const IPDBRawSymbol &RawType = SymbolType->getRawSymbol();

  return RawType.getLength();
}

StorageItemBase::StorageItemBase(const UDTLayoutBase &Parent,
                                 const PDBSymbol &Symbol,
                                 const std::string &Name,
                                 uint32_t OffsetInParent, uint32_t Size)
    : Parent(Parent), Symbol(Symbol), Name(Name),
      OffsetInParent(OffsetInParent), SizeOf(Size) {
  UsedBytes.resize(SizeOf, true);
}

uint32_t StorageItemBase::deepPaddingSize() const {
  // sizeof(Field) - sizeof(typeof(Field)) is trailing padding.
  return SizeOf - getTypeLength(Symbol);
}

DataMemberLayoutItem::DataMemberLayoutItem(
    const UDTLayoutBase &Parent, std::unique_ptr<PDBSymbolData> DataMember)
    : StorageItemBase(Parent, *DataMember, DataMember->getName(),
                      DataMember->getOffset(), getTypeLength(*DataMember)),
      DataMember(std::move(DataMember)) {
  auto Type = this->DataMember->getType();
  if (auto UDT = unique_dyn_cast<PDBSymbolTypeUDT>(Type)) {
    // UDT data members might have padding in between fields, but otherwise
    // a member should occupy its entire storage.
    UsedBytes.resize(SizeOf, false);
    UdtLayout = llvm::make_unique<ClassLayout>(std::move(UDT));
  }
}

const PDBSymbolData &DataMemberLayoutItem::getDataMember() {
  return *dyn_cast<PDBSymbolData>(&Symbol);
}

bool DataMemberLayoutItem::hasUDTLayout() const { return UdtLayout != nullptr; }

const ClassLayout &DataMemberLayoutItem::getUDTLayout() const {
  return *UdtLayout;
}

uint32_t DataMemberLayoutItem::deepPaddingSize() const {
  uint32_t Result = StorageItemBase::deepPaddingSize();
  if (UdtLayout)
    Result += UdtLayout->deepPaddingSize();
  return Result;
}

VTableLayoutItem::VTableLayoutItem(const UDTLayoutBase &Parent,
                                   std::unique_ptr<PDBSymbolTypeVTable> VTable)
    : StorageItemBase(Parent, *VTable, "<vtbl>", 0, getTypeLength(*VTable)),
      VTable(std::move(VTable)) {
  auto VTableType = cast<PDBSymbolTypePointer>(this->VTable->getType());
  ElementSize = VTableType->getLength();

  Shape =
      unique_dyn_cast<PDBSymbolTypeVTableShape>(VTableType->getPointeeType());
  if (Shape)
    VTableFuncs.resize(Shape->getCount());
}

UDTLayoutBase::UDTLayoutBase(const PDBSymbol &Symbol, const std::string &Name,
                             uint32_t Size)
    : SymbolBase(Symbol), Name(Name), SizeOf(Size) {
  UsedBytes.resize(Size);
  ChildrenPerByte.resize(Size);
  initializeChildren(Symbol);
}

ClassLayout::ClassLayout(const PDBSymbolTypeUDT &UDT)
    : UDTLayoutBase(UDT, UDT.getName(), UDT.getLength()), UDT(UDT) {}

ClassLayout::ClassLayout(std::unique_ptr<PDBSymbolTypeUDT> UDT)
    : ClassLayout(*UDT) {
  OwnedStorage = std::move(UDT);
}

BaseClassLayout::BaseClassLayout(const UDTLayoutBase &Parent,
                                 std::unique_ptr<PDBSymbolTypeBaseClass> Base)
    : UDTLayoutBase(*Base, Base->getName(), Base->getLength()),
      StorageItemBase(Parent, *Base, Base->getName(), Base->getOffset(),
                      Base->getLength()),
      Base(std::move(Base)) {
  IsVirtualBase = this->Base->isVirtualBaseClass();
}

uint32_t UDTLayoutBase::shallowPaddingSize() const {
  return UsedBytes.size() - UsedBytes.count();
}

uint32_t UDTLayoutBase::deepPaddingSize() const {
  uint32_t Result = shallowPaddingSize();
  for (auto &Child : ChildStorage)
    Result += Child->deepPaddingSize();
  return Result;
}

void UDTLayoutBase::initializeChildren(const PDBSymbol &Sym) {
  // Handled bases first, followed by VTables, followed by data members,
  // followed by functions, followed by other.  This ordering is necessary
  // so that bases and vtables get initialized before any functions which
  // may override them.

  UniquePtrVector<PDBSymbolTypeBaseClass> Bases;
  UniquePtrVector<PDBSymbolTypeVTable> VTables;
  UniquePtrVector<PDBSymbolData> Members;
  auto Children = Sym.findAllChildren();
  while (auto Child = Children->getNext()) {
    if (auto Base = unique_dyn_cast<PDBSymbolTypeBaseClass>(Child)) {
      if (Base->isVirtualBaseClass())
        VirtualBases.push_back(std::move(Base));
      else
        Bases.push_back(std::move(Base));
    }

    else if (auto Data = unique_dyn_cast<PDBSymbolData>(Child)) {
      if (Data->getDataKind() == PDB_DataKind::Member)
        Members.push_back(std::move(Data));
      else
        Other.push_back(std::move(Child));
    } else if (auto VT = unique_dyn_cast<PDBSymbolTypeVTable>(Child))
      VTables.push_back(std::move(VT));
    else if (auto Func = unique_dyn_cast<PDBSymbolFunc>(Child))
      Funcs.push_back(std::move(Func));
    else
      Other.push_back(std::move(Child));
  }

  for (auto &Base : Bases) {
    auto BL = llvm::make_unique<BaseClassLayout>(*this, std::move(Base));
    BaseClasses.push_back(BL.get());

    addChildToLayout(std::move(BL));
  }

  for (auto &VT : VTables) {
    auto VTLayout = llvm::make_unique<VTableLayoutItem>(*this, std::move(VT));

    VTable = VTLayout.get();

    addChildToLayout(std::move(VTLayout));
    continue;
  }

  for (auto &Data : Members) {
    auto DM = llvm::make_unique<DataMemberLayoutItem>(*this, std::move(Data));

    addChildToLayout(std::move(DM));
  }

  for (auto &Func : Funcs) {
    if (!Func->isVirtual())
      continue;

    if (Func->isIntroVirtualFunction())
      addVirtualIntro(*Func);
    else
      addVirtualOverride(*Func);
  }
}

void UDTLayoutBase::addVirtualIntro(PDBSymbolFunc &Func) {
  // Kind of a hack, but we prefer the more common destructor name that people
  // are familiar with, e.g. ~ClassName.  It seems there are always both and
  // the vector deleting destructor overwrites the nice destructor, so just
  // ignore the vector deleting destructor.
  if (Func.getName() == "__vecDelDtor")
    return;

  if (!VTable) {
    // FIXME: Handle this.  What's most likely happening is we have an intro
    // virtual in a derived class where the base also has an intro virtual.
    // In this case the vtable lives in the base.  What we really need is
    // for each UDTLayoutBase to contain a list of all its vtables, and
    // then propagate this list up the hierarchy so that derived classes have
    // direct access to their bases' vtables.
    return;
  }

  uint32_t Stride = VTable->getElementSize();

  uint32_t Index = Func.getVirtualBaseOffset();
  assert(Index % Stride == 0);
  Index /= Stride;

  VTable->setFunction(Index, Func);
}

VTableLayoutItem *UDTLayoutBase::findVTableAtOffset(uint32_t RelativeOffset) {
  if (VTable && VTable->getOffsetInParent() == RelativeOffset)
    return VTable;
  for (auto Base : BaseClasses) {
    uint32_t Begin = Base->getOffsetInParent();
    uint32_t End = Begin + Base->getSize();
    if (RelativeOffset < Begin || RelativeOffset >= End)
      continue;

    return Base->findVTableAtOffset(RelativeOffset - Begin);
  }

  return nullptr;
}

void UDTLayoutBase::addVirtualOverride(PDBSymbolFunc &Func) {
  auto Signature = Func.getSignature();
  auto ThisAdjust = Signature->getThisAdjust();
  // ThisAdjust tells us which VTable we're looking for.  Specifically, it's
  // the offset into the current class of the VTable we're looking for.  So
  // look through the base hierarchy until we find one such that
  // AbsoluteOffset(VT) == ThisAdjust
  VTableLayoutItem *VT = findVTableAtOffset(ThisAdjust);
  if (!VT) {
    // FIXME: There really should be a vtable here.  If there's not it probably
    // means that the vtable is in a virtual base, which we don't yet support.
    assert(!VirtualBases.empty());
    return;
  }
  int32_t OverrideIndex = -1;
  // Now we've found the VTable.  Func will not have a virtual base offset set,
  // so instead we need to compare names and signatures.  We iterate each item
  // in the VTable.  All items should already have non null entries because they
  // were initialized by the intro virtual, which was guaranteed to come before.
  for (auto ItemAndIndex : enumerate(VT->funcs())) {
    auto Item = ItemAndIndex.value();
    assert(Item);
    // If the name doesn't match, this isn't an override.  Note that it's ok
    // for the return type to not match (e.g. co-variant return).
    if (Item->getName() != Func.getName()) {
      if (Item->isDestructor() && Func.isDestructor()) {
        OverrideIndex = ItemAndIndex.index();
        break;
      }
      continue;
    }
    // Now make sure it's the right overload.  Get the signature of the existing
    // vtable method and make sure it has the same arglist and the same cv-ness.
    auto ExistingSig = Item->getSignature();
    if (ExistingSig->isConstType() != Signature->isConstType())
      continue;
    if (ExistingSig->isVolatileType() != Signature->isVolatileType())
      continue;

    // Now compare arguments.  Using the raw bytes of the PDB this would be
    // trivial
    // because there is an ArgListId and they should be identical.  But DIA
    // doesn't
    // expose this, so the best we can do is iterate each argument and confirm
    // that
    // each one is identical.
    if (ExistingSig->getCount() != Signature->getCount())
      continue;
    bool IsMatch = true;
    auto ExistingEnumerator = ExistingSig->getArguments();
    auto NewEnumerator = Signature->getArguments();
    for (uint32_t I = 0; I < ExistingEnumerator->getChildCount(); ++I) {
      auto ExistingArg = ExistingEnumerator->getNext();
      auto NewArg = NewEnumerator->getNext();
      if (ExistingArg->getSymIndexId() != NewArg->getSymIndexId()) {
        IsMatch = false;
        break;
      }
    }
    if (!IsMatch)
      continue;

    // It's a match!  Stick the new function into the VTable.
    OverrideIndex = ItemAndIndex.index();
    break;
  }
  if (OverrideIndex == -1) {
    // FIXME: This is probably due to one of the other FIXMEs in this file.
    return;
  }
  VT->setFunction(OverrideIndex, Func);
}

void UDTLayoutBase::addChildToLayout(std::unique_ptr<StorageItemBase> Child) {
  uint32_t Begin = Child->getOffsetInParent();
  uint32_t End = Begin + Child->getSize();
  // Due to the empty base optimization, End might point outside the bounds of
  // the parent class.  If that happens, just clamp the value.
  End = std::min(End, getClassSize());

  UsedBytes.set(Begin, End);
  while (Begin != End) {
    ChildrenPerByte[Begin].push_back(Child.get());
    ++Begin;
  }

  auto Loc = std::upper_bound(
      ChildStorage.begin(), ChildStorage.end(), Begin,
      [](uint32_t Off, const std::unique_ptr<StorageItemBase> &Item) {
        return Off < Item->getOffsetInParent();
      });

  ChildStorage.insert(Loc, std::move(Child));
}