//===- UDTLayout.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/UDTLayout.h"

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
  // initialize vtbl methods.
  auto VTableType = cast<PDBSymbolTypePointer>(this->VTable->getType());
  uint32_t PointerSize = VTableType->getLength();

  if (auto Shape = unique_dyn_cast<PDBSymbolTypeVTableShape>(
          VTableType->getPointeeType())) {
    VTableFuncs.resize(Shape->getCount());

    auto ParentFunctions =
        Parent.getSymbolBase().findAllChildren<PDBSymbolFunc>();
    while (auto Func = ParentFunctions->getNext()) {
      if (Func->isVirtual()) {
        uint32_t Index = Func->getVirtualBaseOffset();
        assert(Index % PointerSize == 0);
        Index /= PointerSize;

        // Don't allow a compiler generated function to overwrite a user
        // function in the VTable.  Not sure why this happens, but a function
        // named __vecDelDtor sometimes shows up on top of the destructor.
        if (Func->isCompilerGenerated() && VTableFuncs[Index])
          continue;
        VTableFuncs[Index] = std::move(Func);
      }
    }
  }
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
  auto Children = Sym.findAllChildren();
  while (auto Child = Children->getNext()) {
    if (auto Data = unique_dyn_cast<PDBSymbolData>(Child)) {
      if (Data->getDataKind() == PDB_DataKind::Member) {
        auto DM =
            llvm::make_unique<DataMemberLayoutItem>(*this, std::move(Data));

        addChildToLayout(std::move(DM));
      } else {
        NonStorageItems.push_back(std::move(Data));
      }
      continue;
    }

    if (auto Base = unique_dyn_cast<PDBSymbolTypeBaseClass>(Child)) {
      auto BL = llvm::make_unique<BaseClassLayout>(*this, std::move(Base));
      BaseClasses.push_back(BL.get());

      addChildToLayout(std::move(BL));
      continue;
    }

    if (auto VT = unique_dyn_cast<PDBSymbolTypeVTable>(Child)) {
      auto VTLayout = llvm::make_unique<VTableLayoutItem>(*this, std::move(VT));

      VTable = VTLayout.get();

      addChildToLayout(std::move(VTLayout));
      continue;
    }

    NonStorageItems.push_back(std::move(Child));
  }
}

void UDTLayoutBase::addChildToLayout(std::unique_ptr<StorageItemBase> Child) {
  uint32_t Begin = Child->getOffsetInParent();
  uint32_t End = Begin + Child->getSize();
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