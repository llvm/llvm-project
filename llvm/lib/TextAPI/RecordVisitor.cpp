//===- RecordVisitor.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implements the TAPI Record Visitor.
///
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/RecordVisitor.h"

using namespace llvm;
using namespace llvm::MachO;

RecordVisitor::~RecordVisitor() {}
void RecordVisitor::visitObjCInterface(const ObjCInterfaceRecord &) {}
void RecordVisitor::visitObjCCategory(const ObjCCategoryRecord &) {}

static bool shouldSkipRecord(const Record &R, const bool RecordUndefs) {
  if (R.isExported())
    return false;

  // Skip non exported symbols unless for flat namespace libraries.
  return !(RecordUndefs && R.isUndefined());
}

void SymbolConverter::visitGlobal(const GlobalRecord &GR) {
  auto [SymName, SymKind] = parseSymbol(GR.getName(), GR.getFlags());
  if (shouldSkipRecord(GR, RecordUndefs))
    return;
  Symbols->addGlobal(SymKind, SymName, GR.getFlags(), Targ);
}

void SymbolConverter::addIVars(const ArrayRef<ObjCIVarRecord *> IVars,
                               StringRef ContainerName) {
  for (auto *IV : IVars) {
    if (shouldSkipRecord(*IV, RecordUndefs))
      continue;
    std::string Name =
        ObjCIVarRecord::createScopedName(ContainerName, IV->getName());
    Symbols->addGlobal(SymbolKind::ObjectiveCInstanceVariable, Name,
                       IV->getFlags(), Targ);
  }
}

void SymbolConverter::visitObjCInterface(const ObjCInterfaceRecord &ObjCR) {
  if (!shouldSkipRecord(ObjCR, RecordUndefs)) {
    Symbols->addGlobal(SymbolKind::ObjectiveCClass, ObjCR.getName(),
                       ObjCR.getFlags(), Targ);
    if (ObjCR.hasExceptionAttribute())
      Symbols->addGlobal(SymbolKind::ObjectiveCClassEHType, ObjCR.getName(),
                         ObjCR.getFlags(), Targ);
  }

  addIVars(ObjCR.getObjCIVars(), ObjCR.getName());
  for (const auto *Cat : ObjCR.getObjCCategories())
    addIVars(Cat->getObjCIVars(), ObjCR.getName());
}

void SymbolConverter::visitObjCCategory(const ObjCCategoryRecord &Cat) {
  addIVars(Cat.getObjCIVars(), Cat.getName());
}
