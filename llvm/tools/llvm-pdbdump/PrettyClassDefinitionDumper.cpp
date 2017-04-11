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

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::pdb;

ClassDefinitionDumper::ClassDefinitionDumper(LinePrinter &P)
    : PDBSymDumper(true), Printer(P) {}

static void analyzePadding(const PDBSymbolTypeUDT &Class, BitVector &Padding,
                           uint32_t &FirstFieldOffset) {
  Padding.resize(Class.getLength(), true);
  auto Children = Class.findAllChildren<PDBSymbolData>();
  bool IsFirst = true;
  FirstFieldOffset = Class.getLength();

  while (auto Data = Children->getNext()) {
    // Ignore data members which are not relative to this.  Usually these are
    // static data members or constexpr and occupy no space.  We also need to
    // handle BitFields since the PDB doesn't consider them ThisRel, but they
    // still occupy space in the record layout.
    auto LocType = Data->getLocationType();
    if (LocType != PDB_LocType::ThisRel && LocType != PDB_LocType::BitField)
      continue;

    uint64_t Start = Data->getOffset();
    if (IsFirst) {
      FirstFieldOffset = Start;
      IsFirst = false;
    }

    auto VarType = Data->getType();
    uint64_t Size = VarType->getRawSymbol().getLength();
    Padding.reset(Start, Start + Size);
  }

  // Unmark anything that comes before the first field so it doesn't get
  // counted as padding.  In reality this is going to be vptrs or base class
  // members, but we don't correctly handle that yet.
  // FIXME: Handle it.
  Padding.reset(0, FirstFieldOffset);
}

void ClassDefinitionDumper::start(const PDBSymbolTypeUDT &Class) {
  assert(opts::pretty::ClassFormat !=
         opts::pretty::ClassDefinitionFormat::None);

  uint32_t Size = Class.getLength();
  uint32_t FirstFieldOffset = 0;
  BitVector Padding;
  analyzePadding(Class, Padding, FirstFieldOffset);

  if (opts::pretty::OnlyPaddingClasses && (Padding.count() == 0))
    return;

  Printer.NewLine();
  WithColor(Printer, PDB_ColorItem::Comment).get() << "// sizeof = " << Size;
  Printer.NewLine();

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

  int NextPaddingByte = Padding.find_first();
  while (auto Child = Children->getNext()) {
    if (auto Data = llvm::dyn_cast<PDBSymbolData>(Child.get())) {
      if (Data->getDataKind() == PDB_DataKind::Member && NextPaddingByte >= 0) {
        // If there are padding bytes remaining, see if this field is the first
        // to cross a padding boundary, and print a padding field indicator if
        // so.
        int Off = Data->getOffset();
        if (Off > NextPaddingByte) {
          uint32_t Amount = Off - NextPaddingByte;
          Printer.NewLine();
          WithColor(Printer, PDB_ColorItem::Padding).get()
              << "<padding> (" << Amount << " bytes)";
          assert(Padding.find_next_unset(NextPaddingByte) == Off);
          NextPaddingByte = Padding.find_next(Off);
        }
      }
    }

    if (auto Func = Child->cast<PDBSymbolFunc>()) {
      if (Func->isCompilerGenerated() && opts::pretty::ExcludeCompilerGenerated)
        continue;

      if (Func->getLength() == 0 && !Func->isPureVirtual() &&
          !Func->isIntroVirtualFunction())
        continue;
    }

    ++DumpedCount;
    Child->dump(*this);
  }

  if (NextPaddingByte >= 0) {
    uint32_t Amount = Size - NextPaddingByte;
    Printer.NewLine();
    WithColor(Printer, PDB_ColorItem::Padding).get() << "<padding> (" << Amount
                                                     << " bytes)";
  }
  Printer.Unindent();
  if (DumpedCount > 0)
    Printer.NewLine();
  Printer << "}";
  Printer.NewLine();
  if (Padding.count() > 0) {
    APFloat Pct(100.0 * (double)Padding.count() /
                (double)(Size - FirstFieldOffset));
    SmallString<8> PctStr;
    Pct.toString(PctStr, 4);
    WithColor(Printer, PDB_ColorItem::Padding).get()
        << "Total padding " << Padding.count() << " bytes (" << PctStr
        << "% of class size)";
    Printer.NewLine();
  }
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
