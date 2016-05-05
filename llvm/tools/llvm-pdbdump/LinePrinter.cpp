//===- LinePrinter.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LinePrinter.h"

#include "llvm-pdbdump.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Regex.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::pdb;

namespace {
bool IsItemExcluded(llvm::StringRef Item,
                    std::list<llvm::Regex> &IncludeFilters,
                    std::list<llvm::Regex> &ExcludeFilters) {
  if (Item.empty())
    return false;

  auto match_pred = [Item](llvm::Regex &R) { return R.match(Item); };

  // Include takes priority over exclude.  If the user specified include
  // filters, and none of them include this item, them item is gone.
  if (!IncludeFilters.empty() && !any_of(IncludeFilters, match_pred))
    return true;

  if (any_of(ExcludeFilters, match_pred))
    return true;

  return false;
}
}

using namespace llvm;

LinePrinter::LinePrinter(int Indent, llvm::raw_ostream &Stream)
    : OS(Stream), IndentSpaces(Indent), CurrentIndent(0) {
  SetFilters(ExcludeTypeFilters, opts::ExcludeTypes.begin(),
             opts::ExcludeTypes.end());
  SetFilters(ExcludeSymbolFilters, opts::ExcludeSymbols.begin(),
             opts::ExcludeSymbols.end());
  SetFilters(ExcludeCompilandFilters, opts::ExcludeCompilands.begin(),
             opts::ExcludeCompilands.end());

  SetFilters(IncludeTypeFilters, opts::IncludeTypes.begin(),
             opts::IncludeTypes.end());
  SetFilters(IncludeSymbolFilters, opts::IncludeSymbols.begin(),
             opts::IncludeSymbols.end());
  SetFilters(IncludeCompilandFilters, opts::IncludeCompilands.begin(),
             opts::IncludeCompilands.end());
}

void LinePrinter::Indent() { CurrentIndent += IndentSpaces; }

void LinePrinter::Unindent() {
  CurrentIndent = std::max(0, CurrentIndent - IndentSpaces);
}

void LinePrinter::NewLine() {
  OS << "\n";
  OS.indent(CurrentIndent);
}

bool LinePrinter::IsTypeExcluded(llvm::StringRef TypeName) {
  return IsItemExcluded(TypeName, IncludeTypeFilters, ExcludeTypeFilters);
}

bool LinePrinter::IsSymbolExcluded(llvm::StringRef SymbolName) {
  return IsItemExcluded(SymbolName, IncludeSymbolFilters, ExcludeSymbolFilters);
}

bool LinePrinter::IsCompilandExcluded(llvm::StringRef CompilandName) {
  return IsItemExcluded(CompilandName, IncludeCompilandFilters,
                        ExcludeCompilandFilters);
}

WithColor::WithColor(LinePrinter &P, PDB_ColorItem C) : OS(P.OS) {
  applyColor(C);
}

WithColor::~WithColor() { OS.resetColor(); }

void WithColor::applyColor(PDB_ColorItem C) {
  switch (C) {
  case PDB_ColorItem::None:
    OS.resetColor();
    return;
  case PDB_ColorItem::Address:
    OS.changeColor(raw_ostream::YELLOW, /*bold=*/true);
    return;
  case PDB_ColorItem::Keyword:
    OS.changeColor(raw_ostream::MAGENTA, true);
    return;
  case PDB_ColorItem::Register:
  case PDB_ColorItem::Offset:
    OS.changeColor(raw_ostream::YELLOW, false);
    return;
  case PDB_ColorItem::Type:
    OS.changeColor(raw_ostream::CYAN, true);
    return;
  case PDB_ColorItem::Identifier:
    OS.changeColor(raw_ostream::CYAN, false);
    return;
  case PDB_ColorItem::Path:
    OS.changeColor(raw_ostream::CYAN, false);
    return;
  case PDB_ColorItem::SectionHeader:
    OS.changeColor(raw_ostream::RED, true);
    return;
  case PDB_ColorItem::LiteralValue:
    OS.changeColor(raw_ostream::GREEN, true);
    return;
  }
}
