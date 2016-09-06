//===- VariableDumper.cpp - -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "VariableDumper.h"

#include "BuiltinDumper.h"
#include "LinePrinter.h"
#include "llvm-pdbdump.h"
#include "FunctionDumper.h"

#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeArray.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeFunctionSig.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypePointer.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::pdb;

VariableDumper::VariableDumper(LinePrinter &P)
    : PDBSymDumper(true), Printer(P) {}

void VariableDumper::start(const PDBSymbolData &Var) {
  if (Var.isCompilerGenerated() && opts::pretty::ExcludeCompilerGenerated)
    return;
  if (Printer.IsSymbolExcluded(Var.getName()))
    return;

  auto VarType = Var.getType();

  switch (auto LocType = Var.getLocationType()) {
  case PDB_LocType::Static:
    Printer.NewLine();
    Printer << "data [";
    WithColor(Printer, PDB_ColorItem::Address).get()
        << format_hex(Var.getVirtualAddress(), 10);
    Printer << "] ";
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "static ";
    dumpSymbolTypeAndName(*VarType, Var.getName());
    break;
  case PDB_LocType::Constant:
    if (isa<PDBSymbolTypeEnum>(*VarType))
      break;
    Printer.NewLine();
    Printer << "data ";
    WithColor(Printer, PDB_ColorItem::Keyword).get() << "const ";
    dumpSymbolTypeAndName(*VarType, Var.getName());
    Printer << " = ";
    WithColor(Printer, PDB_ColorItem::LiteralValue).get() << Var.getValue();
    break;
  case PDB_LocType::ThisRel:
    Printer.NewLine();
    Printer << "data ";
    WithColor(Printer, PDB_ColorItem::Offset).get()
        << "+" << format_hex(Var.getOffset(), 4) << " ";
    dumpSymbolTypeAndName(*VarType, Var.getName());
    break;
  case PDB_LocType::BitField:
    Printer.NewLine();
    Printer << "data ";
    WithColor(Printer, PDB_ColorItem::Offset).get()
        << "+" << format_hex(Var.getOffset(), 4) << " ";
    dumpSymbolTypeAndName(*VarType, Var.getName());
    Printer << " : ";
    WithColor(Printer, PDB_ColorItem::LiteralValue).get() << Var.getLength();
    break;
  default:
    Printer.NewLine();
    Printer << "data ";
    Printer << "unknown(" << LocType << ") ";
    WithColor(Printer, PDB_ColorItem::Identifier).get() << Var.getName();
    break;
  }
}

void VariableDumper::dump(const PDBSymbolTypeBuiltin &Symbol) {
  BuiltinDumper Dumper(Printer);
  Dumper.start(Symbol);
}

void VariableDumper::dump(const PDBSymbolTypeEnum &Symbol) {
  WithColor(Printer, PDB_ColorItem::Type).get() << Symbol.getName();
}

void VariableDumper::dump(const PDBSymbolTypeFunctionSig &Symbol) {}

void VariableDumper::dump(const PDBSymbolTypePointer &Symbol) {
  auto PointeeType = Symbol.getPointeeType();
  if (!PointeeType)
    return;

  if (auto Func = dyn_cast<PDBSymbolFunc>(PointeeType.get())) {
    FunctionDumper NestedDumper(Printer);
    FunctionDumper::PointerType Pointer =
        Symbol.isReference() ? FunctionDumper::PointerType::Reference
                             : FunctionDumper::PointerType::Pointer;
    NestedDumper.start(*Func, Pointer);
  } else {
    if (Symbol.isConstType())
      WithColor(Printer, PDB_ColorItem::Keyword).get() << "const ";
    if (Symbol.isVolatileType())
      WithColor(Printer, PDB_ColorItem::Keyword).get() << "volatile ";
    PointeeType->dump(*this);
    Printer << (Symbol.isReference() ? "&" : "*");
  }
}

void VariableDumper::dump(const PDBSymbolTypeTypedef &Symbol) {
  WithColor(Printer, PDB_ColorItem::Keyword).get() << "typedef ";
  WithColor(Printer, PDB_ColorItem::Type).get() << Symbol.getName();
}

void VariableDumper::dump(const PDBSymbolTypeUDT &Symbol) {
  WithColor(Printer, PDB_ColorItem::Type).get() << Symbol.getName();
}

void VariableDumper::dumpSymbolTypeAndName(const PDBSymbol &Type,
                                           StringRef Name) {
  if (auto *ArrayType = dyn_cast<PDBSymbolTypeArray>(&Type)) {
    std::string IndexSpec;
    raw_string_ostream IndexStream(IndexSpec);
    std::unique_ptr<PDBSymbol> ElementType = ArrayType->getElementType();
    while (auto NestedArray = dyn_cast<PDBSymbolTypeArray>(ElementType.get())) {
      IndexStream << "[";
      IndexStream << NestedArray->getCount();
      IndexStream << "]";
      ElementType = NestedArray->getElementType();
    }
    IndexStream << "[" << ArrayType->getCount() << "]";
    ElementType->dump(*this);
    WithColor(Printer, PDB_ColorItem::Identifier).get() << " " << Name;
    Printer << IndexStream.str();
  } else {
    if (!tryDumpFunctionPointer(Type, Name)) {
      Type.dump(*this);
      WithColor(Printer, PDB_ColorItem::Identifier).get() << " " << Name;
    }
  }
}

bool VariableDumper::tryDumpFunctionPointer(const PDBSymbol &Type,
                                            StringRef Name) {
  // Function pointers come across as pointers to function signatures.  But the
  // signature carries no name, so we have to handle this case separately.
  if (auto *PointerType = dyn_cast<PDBSymbolTypePointer>(&Type)) {
    auto PointeeType = PointerType->getPointeeType();
    if (auto *FunctionSig =
            dyn_cast<PDBSymbolTypeFunctionSig>(PointeeType.get())) {
      FunctionDumper Dumper(Printer);
      FunctionDumper::PointerType PT = FunctionDumper::PointerType::Pointer;
      if (PointerType->isReference())
        PT = FunctionDumper::PointerType::Reference;
      std::string NameStr(Name.begin(), Name.end());
      Dumper.start(*FunctionSig, NameStr.c_str(), PT);
      return true;
    }
  }
  return false;
}
