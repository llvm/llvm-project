//===-- LVSymbol.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the LVSymbol class.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/LogicalView/Core/LVSymbol.h"
#include "llvm/DebugInfo/LogicalView/Core/LVReader.h"
#include "llvm/DebugInfo/LogicalView/Core/LVScope.h"

using namespace llvm;
using namespace llvm::logicalview;

#define DEBUG_TYPE "Symbol"

namespace {
const char *const KindCallSiteParameter = "CallSiteParameter";
const char *const KindConstant = "Constant";
const char *const KindInherits = "Inherits";
const char *const KindMember = "Member";
const char *const KindParameter = "Parameter";
const char *const KindUndefined = "Undefined";
const char *const KindUnspecified = "Unspecified";
const char *const KindVariable = "Variable";
} // end anonymous namespace

// Return a string representation for the symbol kind.
const char *LVSymbol::kind() const {
  const char *Kind = KindUndefined;
  if (getIsCallSiteParameter())
    Kind = KindCallSiteParameter;
  else if (getIsConstant())
    Kind = KindConstant;
  else if (getIsInheritance())
    Kind = KindInherits;
  else if (getIsMember())
    Kind = KindMember;
  else if (getIsParameter())
    Kind = KindParameter;
  else if (getIsUnspecified())
    Kind = KindUnspecified;
  else if (getIsVariable())
    Kind = KindVariable;
  return Kind;
}

void LVSymbol::resolveName() {
  if (getIsResolvedName())
    return;
  setIsResolvedName();

  LVElement::resolveName();
}

void LVSymbol::resolveReferences() {
  // The symbols can have the following references to other elements:
  //   A Type:
  //     DW_AT_type             ->  Type or Scope
  //     DW_AT_import           ->  Type
  //   A Reference:
  //     DW_AT_specification    ->  Symbol
  //     DW_AT_abstract_origin  ->  Symbol
  //     DW_AT_extension        ->  Symbol

  // Resolve any referenced symbol.
  LVSymbol *Reference = getReference();
  if (Reference) {
    Reference->resolve();
    // Recursively resolve the symbol names.
    resolveReferencesChain();
  }

  // Set the file/line information using the Debug Information entry.
  setFile(Reference);

  // Resolve symbol type.
  if (LVElement *Element = getType()) {
    Element->resolve();

    // In the case of demoted typedefs, use the underlying type.
    if (Element->getIsTypedefReduced()) {
      Element = Element->getType();
      Element->resolve();
    }

    // If the type is a template parameter, get its type, which can
    // point to a type or scope, depending on the argument instance.
    setGenericType(Element);
  }

  // Resolve the variable associated type.
  if (!getType() && Reference)
    setType(Reference->getType());
}

StringRef LVSymbol::resolveReferencesChain() {
  // If the symbol have a DW_AT_specification or DW_AT_abstract_origin,
  // follow the chain to resolve the name from those references.
  if (getHasReference() && !isNamed())
    setName(getReference()->resolveReferencesChain());

  return getName();
}

void LVSymbol::print(raw_ostream &OS, bool Full) const {
  if (getIncludeInPrint() && getReader().doPrintSymbol(this)) {
    getReaderCompileUnit()->incrementPrintedSymbols();
    LVElement::print(OS, Full);
    printExtra(OS, Full);
  }
}

void LVSymbol::printExtra(raw_ostream &OS, bool Full) const {
  // Accessibility depends on the parent (class, structure).
  uint32_t AccessCode = 0;
  if (getIsMember() || getIsInheritance())
    AccessCode = getParentScope()->getIsClass() ? dwarf::DW_ACCESS_private
                                                : dwarf::DW_ACCESS_public;

  const LVSymbol *Symbol = getIsInlined() ? Reference : this;
  std::string Attributes =
      Symbol->getIsCallSiteParameter()
          ? ""
          : formatAttributes(Symbol->externalString(),
                             Symbol->accessibilityString(AccessCode),
                             virtualityString());

  OS << formattedKind(Symbol->kind()) << " " << Attributes;
  if (Symbol->getIsUnspecified())
    OS << formattedName(Symbol->getName());
  else {
    if (Symbol->getIsInheritance())
      OS << Symbol->typeOffsetAsString()
         << formattedNames(Symbol->getTypeQualifiedName(),
                           Symbol->typeAsString());
    else {
      OS << formattedName(Symbol->getName());
      // Print any bitfield information.
      if (uint32_t Size = getBitSize())
        OS << ":" << Size;
      OS << " -> " << Symbol->typeOffsetAsString()
         << formattedNames(Symbol->getTypeQualifiedName(),
                           Symbol->typeAsString());
    }
  }

  // Print any initial value if any.
  if (ValueIndex)
    OS << " = " << formattedName(getValue());
  OS << "\n";

  if (Full && options().getPrintFormatting()) {
    if (getLinkageNameIndex())
      printLinkageName(OS, Full, const_cast<LVSymbol *>(this));
    if (LVSymbol *Reference = getReference())
      Reference->printReference(OS, Full, const_cast<LVSymbol *>(this));
  }
}
