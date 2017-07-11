//===-- WasmDumper.cpp - Wasm-specific object file dumper -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Wasm-specific dumper for llvm-readobj.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "ObjDumper.h"
#include "llvm-readobj.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace object;

namespace {

static const EnumEntry<unsigned> WasmSymbolTypes[] = {
#define ENUM_ENTRY(X) { #X, static_cast<unsigned>(WasmSymbol::SymbolType::X) }
  ENUM_ENTRY(FUNCTION_IMPORT),
  ENUM_ENTRY(FUNCTION_EXPORT),
  ENUM_ENTRY(GLOBAL_IMPORT),
  ENUM_ENTRY(GLOBAL_EXPORT),
  ENUM_ENTRY(DEBUG_FUNCTION_NAME),
#undef ENUM_ENTRY
};

static const EnumEntry<uint32_t> WasmSectionTypes[] = {
#define ENUM_ENTRY(X) { #X, wasm::WASM_SEC_##X }
  ENUM_ENTRY(CUSTOM),
  ENUM_ENTRY(TYPE),
  ENUM_ENTRY(IMPORT),
  ENUM_ENTRY(FUNCTION),
  ENUM_ENTRY(TABLE),
  ENUM_ENTRY(MEMORY),
  ENUM_ENTRY(GLOBAL),
  ENUM_ENTRY(EXPORT),
  ENUM_ENTRY(START),
  ENUM_ENTRY(ELEM),
  ENUM_ENTRY(CODE),
  ENUM_ENTRY(DATA),
#undef ENUM_ENTRY
};

class WasmDumper : public ObjDumper {
public:
  WasmDumper(const WasmObjectFile *Obj, ScopedPrinter &Writer)
      : ObjDumper(Writer), Obj(Obj) {}

  void printFileHeaders() override;
  void printSections() override;
  void printRelocations() override;
  void printSymbols() override;
  void printDynamicSymbols() override { llvm_unreachable("unimplemented"); }
  void printUnwindInfo() override { llvm_unreachable("unimplemented"); }
  void printStackMap() const override { llvm_unreachable("unimplemented"); }

protected:
  void printSymbol(const SymbolRef &Sym);
  void printRelocation(const SectionRef &Section, const RelocationRef &Reloc);

private:
  const WasmObjectFile *Obj;
};

void WasmDumper::printFileHeaders() {
  W.printHex("Version", Obj->getHeader().Version);
}

void WasmDumper::printRelocation(const SectionRef &Section,
                                 const RelocationRef &Reloc) {
  SmallString<64> RelocTypeName;
  uint64_t RelocType = Reloc.getType();
  Reloc.getTypeName(RelocTypeName);
  const wasm::WasmRelocation &WasmReloc = Obj->getWasmRelocation(Reloc);

  bool HasAddend = false;
  switch (RelocType) {
  case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_LEB:
  case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_SLEB:
  case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_I32:
    HasAddend = true;
    break;
  default:
    break;
  }
  if (opts::ExpandRelocs) {
    DictScope Group(W, "Relocation");
    W.printNumber("Type", RelocTypeName, RelocType);
    W.printHex("Offset", Reloc.getOffset());
    W.printHex("Index", WasmReloc.Index);
    if (HasAddend)
      W.printNumber("Addend", WasmReloc.Addend);
  } else {
    raw_ostream& OS = W.startLine();
    OS << W.hex(Reloc.getOffset())
       << " " << RelocTypeName << "[" << WasmReloc.Index << "]";
    if (HasAddend)
      OS << " " << WasmReloc.Addend;
    OS << "\n";
  }
}

void WasmDumper::printRelocations() {
  ListScope D(W, "Relocations");

  int SectionNumber = 0;
  for (const SectionRef &Section : Obj->sections()) {
    bool PrintedGroup = false;
    StringRef Name;
    error(Section.getName(Name));
    ++SectionNumber;

    for (const RelocationRef &Reloc : Section.relocations()) {
      if (!PrintedGroup) {
        W.startLine() << "Section (" << SectionNumber << ") " << Name << " {\n";
        W.indent();
        PrintedGroup = true;
      }

      printRelocation(Section, Reloc);
    }

    if (PrintedGroup) {
      W.unindent();
      W.startLine() << "}\n";
    }
  }
}

void WasmDumper::printSymbols() {
  ListScope Group(W, "Symbols");

  for (const SymbolRef &Symbol : Obj->symbols())
    printSymbol(Symbol);
}

void WasmDumper::printSections() {
  ListScope Group(W, "Sections");
  for (const SectionRef &Section : Obj->sections()) {
    const WasmSection &WasmSec = Obj->getWasmSection(Section);
    DictScope SectionD(W, "Section");
    W.printEnum("Type", WasmSec.Type, makeArrayRef(WasmSectionTypes));
    W.printNumber("Size", (uint64_t)WasmSec.Content.size());
    W.printNumber("Offset", WasmSec.Offset);
    switch (WasmSec.Type) {
    case wasm::WASM_SEC_CUSTOM:
      W.printString("Name", WasmSec.Name);
      if (WasmSec.Name == "linking") {
        const wasm::WasmLinkingData &LinkingData = Obj->linkingData();
        W.printNumber("DataSize", LinkingData.DataSize);
        if (LinkingData.DataAlignment)
          W.printNumber("DataAlignment", LinkingData.DataAlignment);
      }
      break;
    case wasm::WASM_SEC_MEMORY:
      ListScope Group(W, "Memories");
      for (const wasm::WasmLimits &Memory : Obj->memories()) {
        DictScope Group(W, "Memory");
        W.printNumber("InitialPages", Memory.Initial);
        if (Memory.Flags & wasm::WASM_LIMITS_FLAG_HAS_MAX) {
          W.printNumber("MaxPages", WasmSec.Offset);
        }
      }
      break;
    }

    if (opts::SectionRelocations) {
      ListScope D(W, "Relocations");
      for (const RelocationRef &Reloc : Section.relocations())
        printRelocation(Section, Reloc);
    }

    if (opts::SectionData) {
      W.printBinaryBlock("SectionData", WasmSec.Content);
    }
  }
}

void WasmDumper::printSymbol(const SymbolRef &Sym) {
  DictScope D(W, "Symbol");
  WasmSymbol Symbol = Obj->getWasmSymbol(Sym.getRawDataRefImpl());
  W.printString("Name", Symbol.Name);
  W.printEnum("Type", static_cast<unsigned>(Symbol.Type), makeArrayRef(WasmSymbolTypes));
  W.printHex("Flags", Symbol.Flags);
}

}

namespace llvm {

std::error_code createWasmDumper(const object::ObjectFile *Obj,
                                 ScopedPrinter &Writer,
                                 std::unique_ptr<ObjDumper> &Result) {
  const WasmObjectFile *WasmObj = dyn_cast<WasmObjectFile>(Obj);
  assert(WasmObj && "createWasmDumper called with non-wasm object");

  Result.reset(new WasmDumper(WasmObj, Writer));
  return readobj_error::success;
}

} // namespace llvm
