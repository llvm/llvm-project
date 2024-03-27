//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCCASPrinter.h"
#include "CASDWARFObject.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/MCCAS/MCCASDebugV1.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::cas;
using namespace llvm::mccasformats::v1;

namespace {
struct IndentGuard {
  constexpr static int IndentWidth = 2;
  IndentGuard(int &Indent) : Indent{Indent} { Indent += IndentWidth; }
  ~IndentGuard() { Indent -= IndentWidth; }
  int &Indent;
};

bool isDwarfSection(MCObjectProxy MCObj) {
  // Currently, the only way to detect debug sections is through the kind of its
  // children objects. TODO: find a better way to check this.
  // Dwarf Sections have >= 1 references.
  if (MCObj.getNumReferences() == 0)
    return false;

  ObjectRef FirstRef = MCObj.getReference(0);
  const MCSchema &Schema = MCObj.getSchema();
  Expected<MCObjectProxy> FirstMCRef = Schema.get(FirstRef);
  if (!FirstMCRef)
    return false;

  return FirstMCRef->getKindString().contains("debug");
}
} // namespace

MCCASPrinter::MCCASPrinter(PrinterOptions Options, ObjectStore &CAS,
                           raw_ostream &OS)
    : Options(Options), MCSchema(CAS), Indent{0}, OS(OS) {}

MCCASPrinter::~MCCASPrinter() { OS << "\n"; }

Expected<CASDWARFObject>
MCCASPrinter::discoverDwarfSections(cas::ObjectRef CASObj) {
  Expected<MCObjectProxy> MCObj = MCSchema.get(CASObj);
  if (!MCObj)
    return MCObj.takeError();
  CASDWARFObject DWARFObj(MCObj->getSchema());
  if (Options.DwarfDump) {
    if (Error E = DWARFObj.discoverDwarfSections(*MCObj))
      return std::move(E);
  }
  return DWARFObj;
}

Error MCCASPrinter::printMCObject(ObjectRef CASObj, CASDWARFObject &Obj,
                                  DWARFContext *DWARFCtx) {
  // The object identifying the schema is not considered an MCObject, as such we
  // don't attempt to cast or print it.
  if (CASObj == MCSchema.getRootNodeTypeID())
    return Error::success();

  Expected<MCObjectProxy> MCObj = MCSchema.get(CASObj);
  if (!MCObj)
    return MCObj.takeError();
  return printMCObject(*MCObj, Obj, DWARFCtx);
}

Error MCCASPrinter::printMCObject(MCObjectProxy MCObj, CASDWARFObject &Obj,
                                  DWARFContext *DWARFCtx) {
  // Initialize DWARFObj.
  std::unique_ptr<DWARFContext> DWARFContextHolder;
  if (Options.DwarfDump && !DWARFCtx) {
    auto DWARFObj = std::make_unique<CASDWARFObject>(Obj);
    DWARFContextHolder = std::make_unique<DWARFContext>(std::move(DWARFObj));
    DWARFCtx = DWARFContextHolder.get();
  }

  // If only debug sections were requested, skip non-debug sections.
  if (Options.DwarfSectionsOnly && SectionRef::Cast(MCObj) &&
      !isDwarfSection(MCObj))
    return Error::success();

  // Print CAS Id.
  OS.indent(Indent);
  OS << formatv("{0, -15} {1} \n", MCObj.getKindString(), MCObj.getID());
  if (Options.HexDump) {
    auto data = MCObj.getData();
    if (Options.HexDumpOneLine) {
      OS.indent(Indent);
      llvm::interleave(
          data.take_front(data.size()), OS,
          [this](unsigned char c) { OS << llvm::format_hex(c, 4); }, " ");
      OS << "\n";
    } else {
      while (!data.empty()) {
        OS.indent(Indent);
        llvm::interleave(
            data.take_front(8), OS,
            [this](unsigned char c) { OS << llvm::format_hex(c, 4); }, " ");
        OS << "\n";
        data = data.drop_front(data.size() < 8 ? data.size() : 8);
      }
    }
  }

  // Dwarfdump.
  if (DWARFCtx) {
    IndentGuard Guard(Indent);
    if (Error Err = Obj.dump(OS, Indent, *DWARFCtx, MCObj, Options.Verbose))
      return Err;
  }
  if (DIETopLevelRef::Cast(MCObj) && Options.DIERefs)
    return Error::success();
  return printSimpleNested(MCObj, Obj, DWARFCtx);
}

Error printDIE(DIETopLevelRef TopRef, raw_ostream &OS, int Indent,
               SmallVector<StringRef, 0> &TotAbbrevEntries, bool IsLittleEndian,
               uint8_t AddressSize) {
  auto HeaderCallback = [&](StringRef HeaderData) {
    OS.indent(Indent) << "Header = " << '[';
    llvm::interleave(
        HeaderData, OS, [&](uint8_t Char) { OS << utohexstr(Char); }, " ");
    OS << "]\n";
  };
  auto AttrCallback = [&](dwarf::Attribute Attr, dwarf::Form Form,
                          StringRef Data, bool InSeparateBlock) {
    OS.indent(Indent) << formatv("{0, -30} {1, -25}  {2, -10} [",
                                 dwarf::AttributeString(Attr),
                                 dwarf::FormEncodingString(Form),
                                 InSeparateBlock ? "[distinct]" : "[dedups]");
    llvm::interleave(
        Data, OS, [&](uint8_t Char) { OS << utohexstr(Char); }, " ");
    OS << "]\n";
  };
  auto StartTagCallback = [&](dwarf::Tag Tag, uint64_t AbbrevIdx) {
    OS.indent(Indent) << formatv("{0, -25} AbbrevIdx = {1}\n",
                                 dwarf::TagString(Tag), AbbrevIdx);
    Indent += 2;
  };
  auto EndTagCallback = [&](bool) { Indent -= 2; };
  auto NewBlockCallback = [&](StringRef BlockId) {
    OS.indent(Indent) << "CAS Block: " << BlockId << "\n";
  };
  return visitDebugInfo(TotAbbrevEntries, TopRef, HeaderCallback,
                        StartTagCallback, AttrCallback, EndTagCallback,
                        IsLittleEndian, AddressSize, NewBlockCallback);
}

Error MCCASPrinter::printSimpleNested(MCObjectProxy Ref, CASDWARFObject &Obj,
                                      DWARFContext *DWARFCtx) {
  IndentGuard Guard(Indent);

  auto Data = Ref.getData();
  if (DebugAbbrevSectionRef::Cast(Ref) || GroupRef::Cast(Ref) ||
      SymbolTableRef::Cast(Ref) || SectionRef::Cast(Ref) ||
      DebugLineSectionRef::Cast(Ref) || AtomRef::Cast(Ref)) {
    auto Refs = MCObjectProxy::decodeReferences(Ref, Data);
    if (!Refs)
      return Refs.takeError();
    for (auto Ref : *Refs) {
      if (Error E = printMCObject(Ref, Obj, DWARFCtx))
        return E;
    }
    return Error::success();
  }

  if (DebugInfoSectionRef::Cast(Ref) && Options.DIERefs) {
    SmallVector<StringRef, 0> TotAbbrevEntries;
    if (auto E = Ref.forEachReference([&](cas::ObjectRef ID) -> Error {
          Expected<MCObjectProxy> MCObj = MCSchema.get(ID);
          if (!MCObj)
            return MCObj.takeError();
          if (Error E = printMCObject(*MCObj, Obj, DWARFCtx))
            return E;
          if (auto TopRef = DIETopLevelRef::Cast(*MCObj))
            return printDIE(*TopRef, OS, Indent, TotAbbrevEntries,
                            Obj.isLittleEndian(), Obj.getAddressSize());
          return Error::success();
        }))
      return E;
    return Error::success();
  }

  return Ref.forEachReference(
      [&](ObjectRef CASObj) { return printMCObject(CASObj, Obj, DWARFCtx); });
}
