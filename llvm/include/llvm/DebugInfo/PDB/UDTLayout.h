//===- UDTLayout.h - UDT layout info ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_UDTLAYOUT_H
#define LLVM_DEBUGINFO_PDB_UDTLAYOUT_H

#include "PDBSymbol.h"
#include "PDBTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"

#include <list>
#include <memory>

namespace llvm {

class raw_ostream;

namespace pdb {

class PDBSymTypeBaseClass;
class PDBSymbolData;
class PDBSymbolTypeUDT;
class PDBSymbolTypeVTable;

class ClassLayout;
class BaseClassLayout;
class StorageItemBase;
class UDTLayoutBase;

class StorageItemBase {
public:
  StorageItemBase(const UDTLayoutBase &Parent, const PDBSymbol &Symbol,
                  const std::string &Name, uint32_t OffsetInParent,
                  uint32_t Size);
  virtual ~StorageItemBase() {}

  virtual uint32_t deepPaddingSize() const;

  const UDTLayoutBase &getParent() const { return Parent; }
  StringRef getName() const { return Name; }
  uint32_t getOffsetInParent() const { return OffsetInParent; }
  uint32_t getSize() const { return SizeOf; }
  const PDBSymbol &getSymbol() const { return Symbol; }

protected:
  const UDTLayoutBase &Parent;
  const PDBSymbol &Symbol;
  BitVector UsedBytes;
  std::string Name;
  uint32_t OffsetInParent = 0;
  uint32_t SizeOf = 0;
};

class DataMemberLayoutItem : public StorageItemBase {
public:
  DataMemberLayoutItem(const UDTLayoutBase &Parent,
                       std::unique_ptr<PDBSymbolData> DataMember);

  virtual uint32_t deepPaddingSize() const;

  const PDBSymbolData &getDataMember();
  bool hasUDTLayout() const;
  const ClassLayout &getUDTLayout() const;

private:
  std::unique_ptr<PDBSymbolData> DataMember;
  std::unique_ptr<ClassLayout> UdtLayout;
};

class VTableLayoutItem : public StorageItemBase {
public:
  VTableLayoutItem(const UDTLayoutBase &Parent,
                   std::unique_ptr<PDBSymbolTypeVTable> VTable);
  ArrayRef<PDBSymbolFunc *> funcs() const { return VTableFuncs; }

  uint32_t getElementSize() const { return ElementSize; }

  void setFunction(uint32_t Index, PDBSymbolFunc &Func) {
    VTableFuncs[Index] = &Func;
  }

private:
  uint32_t ElementSize = 0;
  std::unique_ptr<PDBSymbolTypeVTableShape> Shape;
  std::unique_ptr<PDBSymbolTypeVTable> VTable;
  std::vector<PDBSymbolFunc *> VTableFuncs;
};

class UDTLayoutBase {
  template <typename T> using UniquePtrVector = std::vector<std::unique_ptr<T>>;

public:
  UDTLayoutBase(const PDBSymbol &Symbol, const std::string &Name,
                uint32_t Size);

  uint32_t shallowPaddingSize() const;
  uint32_t deepPaddingSize() const;

  const BitVector &usedBytes() const { return UsedBytes; }

  uint32_t getClassSize() const { return SizeOf; }

  ArrayRef<std::unique_ptr<StorageItemBase>> layout_items() const {
    return ChildStorage;
  }

  VTableLayoutItem *findVTableAtOffset(uint32_t RelativeOffset);

  StringRef getUDTName() const { return Name; }

  ArrayRef<BaseClassLayout *> bases() const { return BaseClasses; }
  ArrayRef<std::unique_ptr<PDBSymbolTypeBaseClass>> vbases() const {
    return VirtualBases;
  }

  ArrayRef<std::unique_ptr<PDBSymbolFunc>> funcs() const { return Funcs; }

  ArrayRef<std::unique_ptr<PDBSymbol>> other_items() const { return Other; }

  const PDBSymbol &getSymbolBase() const { return SymbolBase; }

protected:
  void initializeChildren(const PDBSymbol &Sym);

  void addChildToLayout(std::unique_ptr<StorageItemBase> Child);
  void addVirtualOverride(PDBSymbolFunc &Func);
  void addVirtualIntro(PDBSymbolFunc &Func);

  const PDBSymbol &SymbolBase;
  std::string Name;
  uint32_t SizeOf = 0;

  BitVector UsedBytes;
  UniquePtrVector<PDBSymbol> Other;
  UniquePtrVector<PDBSymbolFunc> Funcs;
  UniquePtrVector<PDBSymbolTypeBaseClass> VirtualBases;
  UniquePtrVector<StorageItemBase> ChildStorage;
  std::vector<std::list<StorageItemBase *>> ChildrenPerByte;
  std::vector<BaseClassLayout *> BaseClasses;
  VTableLayoutItem *VTable = nullptr;
};

class ClassLayout : public UDTLayoutBase {
public:
  explicit ClassLayout(const PDBSymbolTypeUDT &UDT);
  explicit ClassLayout(std::unique_ptr<PDBSymbolTypeUDT> UDT);

  ClassLayout(ClassLayout &&Other) = default;

  const PDBSymbolTypeUDT &getClass() const { return UDT; }

private:
  std::unique_ptr<PDBSymbolTypeUDT> OwnedStorage;
  const PDBSymbolTypeUDT &UDT;
};

class BaseClassLayout : public UDTLayoutBase, public StorageItemBase {
public:
  BaseClassLayout(const UDTLayoutBase &Parent,
                  std::unique_ptr<PDBSymbolTypeBaseClass> Base);

  const PDBSymbolTypeBaseClass &getBase() const { return *Base; }
  bool isVirtualBase() const { return IsVirtualBase; }

private:
  std::unique_ptr<PDBSymbolTypeBaseClass> Base;
  bool IsVirtualBase;
};
}
} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_UDTLAYOUT_H
