//==-- WebAssemblyTargetStreamer.cpp - WebAssembly Target Streamer Methods --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines WebAssembly-specific target streamer classes.
/// These are for implementing support for target-specific assembly directives.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyTargetStreamer.h"
#include "MCTargetDesc/WebAssemblyMCAsmInfo.h"
#include "MCTargetDesc/WebAssemblyMCTypeUtilities.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

static bool shouldQuoteName(StringRef Name) {
  // Wasm export/import names and import module names can contain characters
  // that are not allowed in identifiers, so we need to quote them.
  auto isValidStartChar = [](char C) { return isalpha(C) || C == '_' || C == '.'; };
  auto isValidChar = [&](char C) { return isValidStartChar(C) || isdigit(C); };
  return !(Name.size() > 0 && isValidStartChar(Name.front())) || 
        llvm::any_of(Name, [&](char C) { return !isValidChar(C); });
}

WebAssemblyTargetStreamer::WebAssemblyTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) {}

void WebAssemblyTargetStreamer::emitValueType(wasm::ValType Type) {
  Streamer.emitIntValue(uint8_t(Type), 1);
}

WebAssemblyTargetAsmStreamer::WebAssemblyTargetAsmStreamer(
    MCStreamer &S, formatted_raw_ostream &OS)
    : WebAssemblyTargetStreamer(S), OS(OS) {}

WebAssemblyTargetWasmStreamer::WebAssemblyTargetWasmStreamer(MCStreamer &S)
    : WebAssemblyTargetStreamer(S) {}

static void printTypes(formatted_raw_ostream &OS,
                       ArrayRef<wasm::ValType> Types) {
  bool First = true;
  for (auto Type : Types) {
    if (First)
      First = false;
    else
      OS << ", ";
    OS << WebAssembly::typeToString(Type);
  }
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitLocal(ArrayRef<wasm::ValType> Types) {
  if (!Types.empty()) {
    OS << "\t.local  \t";
    printTypes(OS, Types);
  }
}

void WebAssemblyTargetAsmStreamer::emitFunctionType(const MCSymbolWasm *Sym) {
  assert(Sym->isFunction());
  OS << "\t.functype\t" << Sym->getName() << " ";
  OS << WebAssembly::signatureToString(Sym->getSignature());
  OS << "\n";
}

void WebAssemblyTargetAsmStreamer::emitGlobalType(const MCSymbolWasm *Sym) {
  assert(Sym->isGlobal());
  OS << "\t.globaltype\t" << Sym->getName() << ", "
     << WebAssembly::typeToString(
            static_cast<wasm::ValType>(Sym->getGlobalType().Type));
  if (!Sym->getGlobalType().Mutable)
    OS << ", immutable";
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitTableType(const MCSymbolWasm *Sym) {
  assert(Sym->isTable());
  const wasm::WasmTableType &Type = Sym->getTableType();
  OS << "\t.tabletype\t" << Sym->getName() << ", "
     << WebAssembly::typeToString(static_cast<wasm::ValType>(Type.ElemType));
  bool HasMaximum = Type.Limits.Flags & wasm::WASM_LIMITS_FLAG_HAS_MAX;
  if (Type.Limits.Minimum != 0 || HasMaximum) {
    OS << ", " << Type.Limits.Minimum;
    if (HasMaximum)
      OS << ", " << Type.Limits.Maximum;
  }
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitTagType(const MCSymbolWasm *Sym) {
  assert(Sym->isTag());
  OS << "\t.tagtype\t" << Sym->getName() << " ";
  OS << WebAssembly::typeListToString(Sym->getSignature()->Params);
  OS << "\n";
}

static void emitNameWithQuoting(formatted_raw_ostream &OS, StringRef Name) {
  if (shouldQuoteName(Name))
    OS << '"' << Name << '"';
  else
    OS << Name;
}

void WebAssemblyTargetAsmStreamer::emitImportModule(const MCSymbolWasm *Sym,
                                                    StringRef ImportModule) {
  OS << "\t.import_module\t" << Sym->getName() << ", ";
  emitNameWithQuoting(OS, ImportModule);
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitImportName(const MCSymbolWasm *Sym,
                                                  StringRef ImportName) {
  OS << "\t.import_name\t" << Sym->getName() << ", ";
  emitNameWithQuoting(OS, ImportName);
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitExportName(const MCSymbolWasm *Sym,
                                                  StringRef ExportName) {
  OS << "\t.export_name\t" << Sym->getName() << ", ";
  emitNameWithQuoting(OS, ExportName);
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitIndIdx(const MCExpr *Value) {
  OS << "\t.indidx\t";
  getContext().getAsmInfo()->printExpr(OS, *Value);
  OS << '\n';
}

void WebAssemblyTargetWasmStreamer::emitLocal(ArrayRef<wasm::ValType> Types) {
  SmallVector<std::pair<wasm::ValType, uint32_t>, 4> Grouped;
  for (auto Type : Types) {
    if (Grouped.empty() || Grouped.back().first != Type)
      Grouped.push_back(std::make_pair(Type, 1));
    else
      ++Grouped.back().second;
  }

  Streamer.emitULEB128IntValue(Grouped.size());
  for (auto Pair : Grouped) {
    Streamer.emitULEB128IntValue(Pair.second);
    emitValueType(Pair.first);
  }
}

void WebAssemblyTargetWasmStreamer::emitIndIdx(const MCExpr *Value) {
  llvm_unreachable(".indidx encoding not yet implemented");
}
