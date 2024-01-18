//===- RecordsSlice.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Records Slice APIs.
//
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/RecordsSlice.h"
#include "llvm/TextAPI/Record.h"
#include "llvm/TextAPI/Symbol.h"
#include <utility>

using namespace llvm;
using namespace llvm::MachO;

Record *RecordsSlice::addRecord(StringRef Name, SymbolFlags Flags,
                                GlobalRecord::Kind GV, RecordLinkage Linkage) {
  // Find a specific Record type to capture.
  auto [APIName, SymKind] = parseSymbol(Name, Flags);
  Name = APIName;
  switch (SymKind) {
  case SymbolKind::GlobalSymbol:
    return addGlobal(Name, Linkage, GV, Flags);
  case SymbolKind::ObjectiveCClass:
    return addObjCInterface(Name, Linkage);
  case SymbolKind::ObjectiveCClassEHType:
    return addObjCInterface(Name, Linkage, /*HasEHType=*/true);
  case SymbolKind::ObjectiveCInstanceVariable: {
    auto [Super, IVar] = Name.split('.');
    // Attempt to find super class.
    ObjCContainerRecord *Container = findContainer(/*isIVar=*/false, Super);
    // If not found, create extension since there is no mapped class symbol.
    if (Container == nullptr)
      Container = addObjCCategory(Super, {});
    return addObjCIVar(Container, IVar, Linkage);
  }
  }

  llvm_unreachable("unexpected symbol kind when adding to Record Slice");
}

ObjCContainerRecord *RecordsSlice::findContainer(bool IsIVar,
                                                 StringRef Name) const {
  StringRef Super = IsIVar ? Name.split('.').first : Name;
  ObjCContainerRecord *Container = findObjCInterface(Super);
  // Ivars can only exist with extensions, if they did not come from
  // class.
  if (Container == nullptr)
    Container = findObjCCategory(Super, "");
  return Container;
}

template <typename R, typename C = RecordMap<R>, typename K = StringRef>
R *findRecord(K Key, const C &Container) {
  const auto *Record = Container.find(Key);
  if (Record == Container.end())
    return nullptr;
  return Record->second.get();
}

GlobalRecord *RecordsSlice::findGlobal(StringRef Name,
                                       GlobalRecord::Kind GV) const {
  auto *Record = findRecord<GlobalRecord>(Name, Globals);
  if (!Record)
    return nullptr;

  switch (GV) {
  case GlobalRecord::Kind::Variable: {
    if (!Record->isVariable())
      return nullptr;
    break;
  }
  case GlobalRecord::Kind::Function: {
    if (!Record->isFunction())
      return nullptr;
    break;
  }
  case GlobalRecord::Kind::Unknown:
    return Record;
  }

  return Record;
}

ObjCInterfaceRecord *RecordsSlice::findObjCInterface(StringRef Name) const {
  return findRecord<ObjCInterfaceRecord>(Name, Classes);
}

ObjCCategoryRecord *RecordsSlice::findObjCCategory(StringRef ClassToExtend,
                                                   StringRef Category) const {
  return findRecord<ObjCCategoryRecord>(std::make_pair(ClassToExtend, Category),
                                        Categories);
}

ObjCIVarRecord *ObjCContainerRecord::findObjCIVar(StringRef IVar) const {
  return findRecord<ObjCIVarRecord>(IVar, IVars);
}

ObjCIVarRecord *RecordsSlice::findObjCIVar(bool IsScopedName,
                                           StringRef Name) const {
  // If scoped name, the name of the container is known.
  if (IsScopedName) {
    // IVar does not exist if there is not a container assigned to it.
    auto *Container = findContainer(/*IsIVar=*/true, Name);
    if (!Container)
      return nullptr;

    StringRef IVar = Name.substr(Name.find_first_of('.') + 1);
    return Container->findObjCIVar(IVar);
  }

  // Otherwise traverse through containers and attempt to find IVar.
  auto getIVar = [Name](auto &Records) -> ObjCIVarRecord * {
    for (const auto &[_, Container] : Records) {
      if (auto *IVarR = Container->findObjCIVar(Name))
        return IVarR;
    }
    return nullptr;
  };

  if (auto *IVarRecord = getIVar(Classes))
    return IVarRecord;

  return getIVar(Categories);
}

GlobalRecord *RecordsSlice::addGlobal(StringRef Name, RecordLinkage Linkage,
                                      GlobalRecord::Kind GV,
                                      SymbolFlags Flags) {
  if (GV == GlobalRecord::Kind::Function)
    Flags |= SymbolFlags::Text;
  else if (GV == GlobalRecord::Kind::Variable)
    Flags |= SymbolFlags::Data;

  Name = copyString(Name);
  auto Result = Globals.insert({Name, nullptr});
  if (Result.second)
    Result.first->second =
        std::make_unique<GlobalRecord>(Name, Linkage, Flags, GV);
  else
    updateLinkage(Result.first->second.get(), Linkage);
  return Result.first->second.get();
}

ObjCInterfaceRecord *RecordsSlice::addObjCInterface(StringRef Name,
                                                    RecordLinkage Linkage,
                                                    bool HasEHType) {
  Name = copyString(Name);
  auto Result = Classes.insert({Name, nullptr});
  if (Result.second) {
    Result.first->second =
        std::make_unique<ObjCInterfaceRecord>(Name, Linkage, HasEHType);
  } else {
    // ObjC classes represent multiple symbols that could have competing
    // linkages, in those cases assign the largest one.
    if (Linkage >= RecordLinkage::Rexported)
      updateLinkage(Result.first->second.get(), Linkage);
  }

  return Result.first->second.get();
}

bool ObjCInterfaceRecord::addObjCCategory(ObjCCategoryRecord *Record) {
  auto Result = Categories.insert({Name, Record});
  return Result.second;
}

ObjCCategoryRecord *RecordsSlice::addObjCCategory(StringRef ClassToExtend,
                                                  StringRef Category) {
  Category = copyString(Category);

  // Add owning record first into record slice.
  auto Result =
      Categories.insert({std::make_pair(ClassToExtend, Category), nullptr});
  if (Result.second)
    Result.first->second =
        std::make_unique<ObjCCategoryRecord>(ClassToExtend, Category);

  // Then add reference to it in in the class.
  if (auto *ObjCClass = findObjCInterface(ClassToExtend))
    ObjCClass->addObjCCategory(Result.first->second.get());

  return Result.first->second.get();
}

ObjCIVarRecord *ObjCContainerRecord::addObjCIVar(StringRef IVar,
                                                 RecordLinkage Linkage) {
  auto Result = IVars.insert({IVar, nullptr});
  if (Result.second)
    Result.first->second = std::make_unique<ObjCIVarRecord>(Name, Linkage);
  return Result.first->second.get();
}

ObjCIVarRecord *RecordsSlice::addObjCIVar(ObjCContainerRecord *Container,
                                          StringRef Name,
                                          RecordLinkage Linkage) {
  Name = copyString(Name);
  ObjCIVarRecord *Record = Container->addObjCIVar(Name, Linkage);
  updateLinkage(Record, Linkage);
  return Record;
}

StringRef RecordsSlice::copyString(StringRef String) {
  if (String.empty())
    return {};

  if (StringAllocator.identifyObject(String.data()))
    return String;

  void *Ptr = StringAllocator.Allocate(String.size(), 1);
  memcpy(Ptr, String.data(), String.size());
  return StringRef(reinterpret_cast<const char *>(Ptr), String.size());
}

RecordsSlice::BinaryAttrs &RecordsSlice::getBinaryAttrs() {
  if (!hasBinaryAttrs())
    BA = std::make_unique<BinaryAttrs>();
  return *BA;
}
