//===- TypeDumper.cpp - PDBSymDumper implementation for types *----- C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TypeDumper.h"

#include "BuiltinDumper.h"
#include "ClassDefinitionDumper.h"
#include "EnumDumper.h"
#include "LinePrinter.h"
#include "llvm-pdbdump.h"
#include "TypedefDumper.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

using namespace llvm;
using namespace llvm::pdb;

TypeDumper::TypeDumper(LinePrinter &P) : PDBSymDumper(true), Printer(P) {}

void TypeDumper::start(const PDBSymbolExe &Exe) {
  auto Enums = Exe.findAllChildren<PDBSymbolTypeEnum>();
  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Enums";
  Printer << ": (" << Enums->getChildCount() << " items)";
  Printer.Indent();
  while (auto Enum = Enums->getNext())
    Enum->dump(*this);
  Printer.Unindent();

  auto Typedefs = Exe.findAllChildren<PDBSymbolTypeTypedef>();
  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Typedefs";
  Printer << ": (" << Typedefs->getChildCount() << " items)";
  Printer.Indent();
  while (auto Typedef = Typedefs->getNext())
    Typedef->dump(*this);
  Printer.Unindent();

  auto Classes = Exe.findAllChildren<PDBSymbolTypeUDT>();
  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Identifier).get() << "Classes";
  Printer << ": (" << Classes->getChildCount() << " items)";
  Printer.Indent();
  while (auto Class = Classes->getNext())
    Class->dump(*this);
  Printer.Unindent();
}

void TypeDumper::dump(const PDBSymbolTypeEnum &Symbol) {
  if (Symbol.getUnmodifiedTypeId() != 0)
    return;
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;
  // Dump member enums when dumping their class definition.
  if (nullptr != Symbol.getClassParent())
    return;

  Printer.NewLine();
  EnumDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void TypeDumper::dump(const PDBSymbolTypeTypedef &Symbol) {
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  Printer.NewLine();
  TypedefDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void TypeDumper::dump(const PDBSymbolTypeUDT &Symbol) {
  if (Symbol.getUnmodifiedTypeId() != 0)
    return;
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  Printer.NewLine();

  if (opts::pretty::NoClassDefs) {
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "class ";
    WithColor(Printer, PDB_ColorItem::Identifier).get() << Symbol.getName();
  } else {
    ClassDefinitionDumper Dumper(Printer);
    Dumper.start(Symbol);
  }
}
