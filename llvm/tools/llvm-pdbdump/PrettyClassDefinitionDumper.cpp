//===- PrettyClassDefinitionDumper.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PrettyClassDefinitionDumper.h"

#include "LinePrinter.h"
#include "PrettyEnumDumper.h"
#include "PrettyFunctionDumper.h"
#include "PrettyTypedefDumper.h"
#include "PrettyVariableDumper.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBaseClass.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::pdb;

ClassDefinitionDumper::ClassDefinitionDumper(LinePrinter &P)
    : PDBSymDumper(true), Printer(P) {}

void ClassDefinitionDumper::start(const PDBSymbolTypeUDT &Class) {
  assert(opts::pretty::ClassFormat !=
         opts::pretty::ClassDefinitionFormat::None);

  std::string Name = Class.getName();
  WithColor(Printer, PDB_ColorItem::Keyword).get() << Class.getUdtKind() << " ";
  WithColor(Printer, PDB_ColorItem::Type).get() << Class.getName();

  auto Bases = Class.findAllChildren<PDBSymbolTypeBaseClass>();
  if (Bases->getChildCount() > 0) {
    Printer.Indent();
    Printer.NewLine();
    Printer << ":";
    uint32_t BaseIndex = 0;
    while (auto Base = Bases->getNext()) {
      Printer << " ";
      WithColor(Printer, PDB_ColorItem::Keyword).get() << Base->getAccess();
      if (Base->isVirtualBaseClass())
        WithColor(Printer, PDB_ColorItem::Keyword).get() << " virtual";
      WithColor(Printer, PDB_ColorItem::Type).get() << " " << Base->getName();
      if (++BaseIndex < Bases->getChildCount()) {
        Printer.NewLine();
        Printer << ",";
      }
    }
    Printer.Unindent();
  }

  Printer << " {";
  auto Children = Class.findAllChildren();
  Printer.Indent();
  int DumpedCount = 0;
  while (auto Child = Children->getNext()) {

    if (opts::pretty::ClassFormat ==
        opts::pretty::ClassDefinitionFormat::LayoutOnly) {
      if (auto Data = dyn_cast<PDBSymbolData>(Child.get())) {
        switch (Data->getLocationType()) {
        case PDB_LocType::ThisRel:
        case PDB_LocType::BitField:
          break;
        default:
          // All other types of data field do not occupy any storage (e.g. are
          // const),
          // so in layout mode we skip them.
          continue;
        }
      } else {
        // Only data symbols affect record layout, so skip any non-data symbols
        // if
        // we're in record layout mode.
        continue;
      }
    }

    if (auto Func = dyn_cast<PDBSymbolFunc>(Child.get())) {
      if (Func->isCompilerGenerated() && opts::pretty::ExcludeCompilerGenerated)
        continue;

      if (Func->getLength() == 0 && !Func->isPureVirtual() &&
          !Func->isIntroVirtualFunction())
        continue;
    }

    ++DumpedCount;
    Child->dump(*this);
  }

  Printer.Unindent();
  if (DumpedCount > 0)
    Printer.NewLine();
  Printer << "}";
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeBaseClass &Symbol) {}

void ClassDefinitionDumper::dump(const PDBSymbolData &Symbol) {
  VariableDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void ClassDefinitionDumper::dump(const PDBSymbolFunc &Symbol) {
  if (Printer.IsSymbolExcluded(Symbol.getName()))
    return;

  Printer.NewLine();
  FunctionDumper Dumper(Printer);
  Dumper.start(Symbol, FunctionDumper::PointerType::None);
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeVTable &Symbol) {}

void ClassDefinitionDumper::dump(const PDBSymbolTypeEnum &Symbol) {
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  Printer.NewLine();
  EnumDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeTypedef &Symbol) {
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  Printer.NewLine();
  TypedefDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void ClassDefinitionDumper::dump(const PDBSymbolTypeUDT &Symbol) {}
