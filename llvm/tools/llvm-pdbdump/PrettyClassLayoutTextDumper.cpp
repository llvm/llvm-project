//===- PrettyClassLayoutTextDumper.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PrettyClassLayoutTextDumper.h"

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
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeVTable.h"
#include "llvm/DebugInfo/PDB/UDTLayout.h"

#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::pdb;

PrettyClassLayoutTextDumper::PrettyClassLayoutTextDumper(LinePrinter &P)
    : PDBSymDumper(true), Printer(P) {}

bool PrettyClassLayoutTextDumper::start(const ClassLayout &Layout) {
  if (opts::pretty::ClassFormat ==
      opts::pretty::ClassDefinitionFormat::Standard) {
    for (auto &Other : Layout.other_items())
      Other->dump(*this);
  }

  const BitVector &UseMap = Layout.usedBytes();
  int NextUnusedByte = Layout.usedBytes().find_first_unset();
  // Next dump items which affect class layout.
  for (auto &LayoutItem : Layout.layout_items()) {
    if (NextUnusedByte >= 0) {
      // If there are padding bytes remaining, see if this field is the first to
      // cross a padding boundary, and print a padding field indicator if so.
      int Off = LayoutItem->getOffsetInParent();
      if (Off > NextUnusedByte) {
        uint32_t Amount = Off - NextUnusedByte;
        Printer.NewLine();
        WithColor(Printer, PDB_ColorItem::Padding).get() << "<padding> ("
                                                         << Amount << " bytes)";
        assert(UseMap.find_next(NextUnusedByte) == Off);
        NextUnusedByte = UseMap.find_next_unset(Off);
      }
    }
    LayoutItem->getSymbol().dump(*this);
  }

  if (NextUnusedByte >= 0 && Layout.getClassSize() > 1) {
    uint32_t Amount = Layout.getClassSize() - NextUnusedByte;
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::Padding).get() << "<padding> (" << Amount
                                                     << " bytes)";
    DumpedAnything = true;
  }

  return DumpedAnything;
}

void PrettyClassLayoutTextDumper::dump(const PDBSymbolTypeBaseClass &Symbol) {}

void PrettyClassLayoutTextDumper::dump(const PDBSymbolData &Symbol) {
  VariableDumper Dumper(Printer);
  Dumper.start(Symbol);
  DumpedAnything = true;
}

void PrettyClassLayoutTextDumper::dump(const PDBSymbolFunc &Symbol) {
  if (Printer.IsSymbolExcluded(Symbol.getName()))
    return;
  if (Symbol.isCompilerGenerated() && opts::pretty::ExcludeCompilerGenerated)
    return;
  if (Symbol.getLength() == 0 && !Symbol.isPureVirtual() &&
      !Symbol.isIntroVirtualFunction())
    return;

  DumpedAnything = true;
  Printer.NewLine();
  FunctionDumper Dumper(Printer);
  Dumper.start(Symbol, FunctionDumper::PointerType::None);
}

void PrettyClassLayoutTextDumper::dump(const PDBSymbolTypeVTable &Symbol) {
  VariableDumper Dumper(Printer);
  Dumper.start(Symbol);
  DumpedAnything = true;
}

void PrettyClassLayoutTextDumper::dump(const PDBSymbolTypeEnum &Symbol) {
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  DumpedAnything = true;
  Printer.NewLine();
  EnumDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void PrettyClassLayoutTextDumper::dump(const PDBSymbolTypeTypedef &Symbol) {
  if (Printer.IsTypeExcluded(Symbol.getName()))
    return;

  DumpedAnything = true;
  Printer.NewLine();
  TypedefDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void PrettyClassLayoutTextDumper::dump(const PDBSymbolTypeUDT &Symbol) {}
