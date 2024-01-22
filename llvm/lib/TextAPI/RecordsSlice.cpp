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
#include "llvm/ADT/SetVector.h"
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
  else {
    updateLinkage(Result.first->second.get(), Linkage);
    updateFlags(Result.first->second.get(), Flags);
  }
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
SymbolFlags Record::mergeFlags(SymbolFlags Flags, RecordLinkage Linkage) {
  // Add Linkage properties into Flags.
  switch (Linkage) {
  case RecordLinkage::Rexported:
    Flags |= SymbolFlags::Rexported;
    return Flags;
  case RecordLinkage::Undefined:
    Flags |= SymbolFlags::Undefined;
    return Flags;
  default:
    return Flags;
  }
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

std::vector<ObjCIVarRecord *> ObjCContainerRecord::getObjCIVars() const {
  std::vector<ObjCIVarRecord *> Records;
  llvm::for_each(IVars,
                 [&](auto &Record) { Records.push_back(Record.second.get()); });
  return Records;
}

std::vector<ObjCCategoryRecord *>
ObjCInterfaceRecord::getObjCCategories() const {
  std::vector<ObjCCategoryRecord *> Records;
  llvm::for_each(Categories,
                 [&](auto &Record) { Records.push_back(Record.second); });
  return Records;
}

ObjCIVarRecord *ObjCContainerRecord::addObjCIVar(StringRef IVar,
                                                 RecordLinkage Linkage) {
  auto Result = IVars.insert({IVar, nullptr});
  if (Result.second)
    Result.first->second = std::make_unique<ObjCIVarRecord>(IVar, Linkage);
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

void RecordsSlice::visit(RecordVisitor &V) const {
  for (auto &G : Globals)
    V.visitGlobal(*G.second);
  for (auto &C : Classes)
    V.visitObjCInterface(*C.second);
  for (auto &Cat : Categories)
    V.visitObjCCategory(*Cat.second);
}

static std::unique_ptr<InterfaceFile>
createInterfaceFile(const Records &Slices, StringRef InstallName) {
  // Pickup symbols first.
  auto Symbols = std::make_unique<SymbolSet>();
  for (auto &S : Slices) {
    if (S->empty())
      continue;
    auto &BA = S->getBinaryAttrs();
    if (BA.InstallName != InstallName)
      continue;

    SymbolConverter Converter(Symbols.get(), S->getTarget(),
                              !BA.TwoLevelNamespace);
    S->visit(Converter);
  }

  auto File = std::make_unique<InterfaceFile>(std::move(Symbols));
  File->setInstallName(InstallName);
  // Assign other attributes.
  for (auto &S : Slices) {
    if (S->empty())
      continue;
    auto &BA = S->getBinaryAttrs();
    if (BA.InstallName != InstallName)
      continue;
    const Target &Targ = S->getTarget();
    File->addTarget(Targ);
    if (File->getFileType() == FileType::Invalid)
      File->setFileType(BA.File);
    if (BA.AppExtensionSafe && !File->isApplicationExtensionSafe())
      File->setApplicationExtensionSafe();
    if (BA.TwoLevelNamespace && !File->isTwoLevelNamespace())
      File->setTwoLevelNamespace();
    if (BA.OSLibNotForSharedCache && !File->isOSLibNotForSharedCache())
      File->setOSLibNotForSharedCache();
    if (File->getCurrentVersion().empty())
      File->setCurrentVersion(BA.CurrentVersion);
    if (File->getCompatibilityVersion().empty())
      File->setCompatibilityVersion(BA.CompatVersion);
    if (File->getSwiftABIVersion() == 0)
      File->setSwiftABIVersion(BA.SwiftABI);
    if (File->getPath().empty())
      File->setPath(BA.Path);
    if (!BA.ParentUmbrella.empty())
      File->addParentUmbrella(Targ, BA.ParentUmbrella);
    for (const auto &Client : BA.AllowableClients)
      File->addAllowableClient(Client, Targ);
    for (const auto &Lib : BA.RexportedLibraries)
      File->addReexportedLibrary(Lib, Targ);
  }

  return File;
}

std::unique_ptr<InterfaceFile>
llvm::MachO::convertToInterfaceFile(const Records &Slices) {
  std::unique_ptr<InterfaceFile> File;
  if (Slices.empty())
    return File;

  SetVector<StringRef> InstallNames;
  for (auto &S : Slices) {
    auto Name = S->getBinaryAttrs().InstallName;
    if (Name.empty())
      continue;
    InstallNames.insert(Name);
  }

  File = createInterfaceFile(Slices, *InstallNames.begin());
  for (StringRef IN : llvm::drop_begin(InstallNames))
    File->addDocument(createInterfaceFile(Slices, IN));

  return File;
}
