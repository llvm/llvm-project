//===- PrettyTypeDumper.cpp - PDBSymDumper type dumper *------------ C++ *-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PrettyTypeDumper.h"

#include "LinePrinter.h"
#include "PrettyBuiltinDumper.h"
#include "PrettyClassDefinitionDumper.h"
#include "PrettyEnumDumper.h"
#include "PrettyTypedefDumper.h"
#include "llvm-pdbdump.h"

#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeBuiltin.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"
#include "llvm/DebugInfo/PDB/UDTLayout.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::pdb;

using LayoutPtr = std::unique_ptr<ClassLayout>;

typedef bool (*CompareFunc)(const LayoutPtr &S1, const LayoutPtr &S2);

static bool CompareNames(const LayoutPtr &S1, const LayoutPtr &S2) {
  return S1->getName() < S2->getName();
}

static bool CompareSizes(const LayoutPtr &S1, const LayoutPtr &S2) {
  return S1->getSize() < S2->getSize();
}

static bool ComparePadding(const LayoutPtr &S1, const LayoutPtr &S2) {
  return S1->deepPaddingSize() < S2->deepPaddingSize();
}

static bool ComparePaddingPct(const LayoutPtr &S1, const LayoutPtr &S2) {
  double Pct1 = (double)S1->deepPaddingSize() / (double)S1->getSize();
  double Pct2 = (double)S2->deepPaddingSize() / (double)S2->getSize();
  return Pct1 < Pct2;
}

static CompareFunc getComparisonFunc(opts::pretty::ClassSortMode Mode) {
  switch (Mode) {
  case opts::pretty::ClassSortMode::Name:
    return CompareNames;
  case opts::pretty::ClassSortMode::Size:
    return CompareSizes;
  case opts::pretty::ClassSortMode::Padding:
    return ComparePadding;
  case opts::pretty::ClassSortMode::PaddingPct:
    return ComparePaddingPct;
  default:
    return nullptr;
  }
}

template <typename Enumerator>
static std::vector<std::unique_ptr<ClassLayout>>
filterAndSortClassDefs(LinePrinter &Printer, Enumerator &E,
                       uint32_t UnfilteredCount) {
  std::vector<std::unique_ptr<ClassLayout>> Filtered;

  Filtered.reserve(UnfilteredCount);
  CompareFunc Comp = getComparisonFunc(opts::pretty::ClassOrder);

  if (UnfilteredCount > 10000) {
    errs() << formatv("Filtering and sorting {0} types", UnfilteredCount);
    errs().flush();
  }
  uint32_t Examined = 0;
  uint32_t Discarded = 0;
  while (auto Class = E.getNext()) {
    ++Examined;
    if (Examined % 10000 == 0) {
      errs() << formatv("Examined {0}/{1} items.  {2} items discarded\n",
                        Examined, UnfilteredCount, Discarded);
      errs().flush();
    }

    if (Class->getUnmodifiedTypeId() != 0) {
      ++Discarded;
      continue;
    }

    if (Printer.IsTypeExcluded(Class->getName(), Class->getLength())) {
      ++Discarded;
      continue;
    }

    auto Layout = llvm::make_unique<ClassLayout>(std::move(Class));
    if (Layout->deepPaddingSize() < opts::pretty::PaddingThreshold) {
      ++Discarded;
      continue;
    }

    Filtered.push_back(std::move(Layout));
  }

  if (Comp)
    std::sort(Filtered.begin(), Filtered.end(), Comp);
  return Filtered;
}

TypeDumper::TypeDumper(LinePrinter &P) : PDBSymDumper(true), Printer(P) {}

void TypeDumper::start(const PDBSymbolExe &Exe) {
  if (opts::pretty::Enums) {
    auto Enums = Exe.findAllChildren<PDBSymbolTypeEnum>();
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::Identifier).get() << "Enums";
    Printer << ": (" << Enums->getChildCount() << " items)";
    Printer.Indent();
    while (auto Enum = Enums->getNext())
      Enum->dump(*this);
    Printer.Unindent();
  }

  if (opts::pretty::Typedefs) {
    auto Typedefs = Exe.findAllChildren<PDBSymbolTypeTypedef>();
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::Identifier).get() << "Typedefs";
    Printer << ": (" << Typedefs->getChildCount() << " items)";
    Printer.Indent();
    while (auto Typedef = Typedefs->getNext())
      Typedef->dump(*this);
    Printer.Unindent();
  }

  if (opts::pretty::Classes) {
    auto Classes = Exe.findAllChildren<PDBSymbolTypeUDT>();
    uint32_t All = Classes->getChildCount();

    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::Identifier).get() << "Classes";

    bool Precompute = false;
    Precompute =
        (opts::pretty::ClassOrder != opts::pretty::ClassSortMode::None);

    // If we're using no sort mode, then we can start getting immediate output
    // from the tool by just filtering as we go, rather than processing
    // everything up front so that we can sort it.  This makes the tool more
    // responsive.  So only precompute the filtered/sorted set of classes if
    // necessary due to the specified options.
    std::vector<LayoutPtr> Filtered;
    uint32_t Shown = All;
    if (Precompute) {
      Filtered = filterAndSortClassDefs(Printer, *Classes, All);

      Shown = Filtered.size();
    }

    Printer << ": (Showing " << Shown << " items";
    if (Shown < All)
      Printer << ", " << (All - Shown) << " filtered";
    Printer << ")";
    Printer.Indent();

    // If we pre-computed, iterate the filtered/sorted list, otherwise iterate
    // the DIA enumerator and filter on the fly.
    if (Precompute) {
      for (auto &Class : Filtered)
        dumpClassLayout(*Class);
    } else {
      while (auto Class = Classes->getNext()) {
        if (Class->getUnmodifiedTypeId() != 0)
          continue;

        if (Printer.IsTypeExcluded(Class->getName(), Class->getLength()))
          continue;

        auto Layout = llvm::make_unique<ClassLayout>(std::move(Class));
        if (Layout->deepPaddingSize() < opts::pretty::PaddingThreshold)
          continue;

        dumpClassLayout(*Layout);
      }
    }

    Printer.Unindent();
  }
}

void TypeDumper::dump(const PDBSymbolTypeEnum &Symbol) {
  assert(opts::pretty::Enums);

  if (Printer.IsTypeExcluded(Symbol.getName(), Symbol.getLength()))
    return;
  // Dump member enums when dumping their class definition.
  if (nullptr != Symbol.getClassParent())
    return;

  Printer.NewLine();
  EnumDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void TypeDumper::dump(const PDBSymbolTypeTypedef &Symbol) {
  assert(opts::pretty::Typedefs);

  if (Printer.IsTypeExcluded(Symbol.getName(), Symbol.getLength()))
    return;

  Printer.NewLine();
  TypedefDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void TypeDumper::dumpClassLayout(const ClassLayout &Class) {
  assert(opts::pretty::Classes);

  if (opts::pretty::ClassFormat == opts::pretty::ClassDefinitionFormat::None) {
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "class ";
    WithColor(Printer, PDB_ColorItem::Identifier).get() << Class.getName();
  } else {
    ClassDefinitionDumper Dumper(Printer);
    Dumper.start(Class);
  }
}
