//==-- WebAssemblyTargetStreamer.cpp - WebAssembly Target Streamer Methods --=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines WebAssembly-specific target streamer classes.
/// These are for implementing support for target-specific assembly directives.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetStreamer.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "WebAssemblyMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

WebAssemblyTargetStreamer::WebAssemblyTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) {}

void WebAssemblyTargetStreamer::emitValueType(wasm::ValType Type) {
  Streamer.EmitIntValue(uint8_t(Type), 1);
}

WebAssemblyTargetAsmStreamer::WebAssemblyTargetAsmStreamer(
    MCStreamer &S, formatted_raw_ostream &OS)
    : WebAssemblyTargetStreamer(S), OS(OS) {}

WebAssemblyTargetWasmStreamer::WebAssemblyTargetWasmStreamer(MCStreamer &S)
    : WebAssemblyTargetStreamer(S) {}

static void PrintTypes(formatted_raw_ostream &OS, ArrayRef<wasm::ValType> Types) {
  bool First = true;
  for (auto Type : Types) {
    if (First)
      First = false;
    else
      OS << ", ";
    OS << WebAssembly::TypeToString(Type);
  }
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitLocal(ArrayRef<wasm::ValType> Types) {
  if (!Types.empty()) {
    OS << "\t.local  \t";
    PrintTypes(OS, Types);
  }
}

void WebAssemblyTargetAsmStreamer::emitEndFunc() { OS << "\t.endfunc\n"; }

void WebAssemblyTargetAsmStreamer::emitFunctionType(MCSymbolWasm *Symbol) {
  OS << "\t.functype\t" << Symbol->getName() << " (";
  auto &Params = Symbol->getSignature()->Params;
  for (auto &Ty : Params) {
    if (&Ty != &Params[0]) OS << ", ";
    OS << WebAssembly::TypeToString(Ty);
  }
  OS << ") -> (";
  auto &Returns = Symbol->getSignature()->Returns;
  for (auto &Ty : Returns) {
    if (&Ty != &Returns[0]) OS << ", ";
    OS << WebAssembly::TypeToString(Ty);
  }
  OS << ")\n";
}

void WebAssemblyTargetAsmStreamer::emitGlobalType(MCSymbolWasm *Sym) {
  assert(Sym->isGlobal());
  OS << "\t.globaltype\t" << Sym->getName() << ", " <<
        WebAssembly::TypeToString(
          static_cast<wasm::ValType>(Sym->getGlobalType().Type)) <<
        '\n';
}

void WebAssemblyTargetAsmStreamer::emitEventType(MCSymbolWasm *Sym) {
  assert(Sym->isEvent());
  OS << "\t.eventtype\t" << Sym->getName();
  if (Sym->getSignature()->Returns.empty())
    OS << ", void";
  else {
    assert(Sym->getSignature()->Returns.size() == 1);
    OS << ", "
       << WebAssembly::TypeToString(Sym->getSignature()->Returns.front());
  }
  for (auto Ty : Sym->getSignature()->Params)
    OS << ", " << WebAssembly::TypeToString(Ty);
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitImportModule(MCSymbolWasm *Sym,
                                                    StringRef ModuleName) {
  OS << "\t.import_module\t" << Sym->getName() << ", " << ModuleName << '\n';
}

void WebAssemblyTargetAsmStreamer::emitIndIdx(const MCExpr *Value) {
  OS << "\t.indidx  \t" << *Value << '\n';
}

void WebAssemblyTargetWasmStreamer::emitLocal(ArrayRef<wasm::ValType> Types) {
  SmallVector<std::pair<wasm::ValType, uint32_t>, 4> Grouped;
  for (auto Type : Types) {
    if (Grouped.empty() || Grouped.back().first != Type)
      Grouped.push_back(std::make_pair(Type, 1));
    else
      ++Grouped.back().second;
  }

  Streamer.EmitULEB128IntValue(Grouped.size());
  for (auto Pair : Grouped) {
    Streamer.EmitULEB128IntValue(Pair.second);
    emitValueType(Pair.first);
  }
}

void WebAssemblyTargetWasmStreamer::emitEndFunc() {
  llvm_unreachable(".end_func is not needed for direct wasm output");
}

void WebAssemblyTargetWasmStreamer::emitIndIdx(const MCExpr *Value) {
  llvm_unreachable(".indidx encoding not yet implemented");
}

void WebAssemblyTargetWasmStreamer::emitFunctionType(MCSymbolWasm *Symbol) {
  // Symbol already has its arguments and result set.
  Symbol->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);
}

void WebAssemblyTargetWasmStreamer::emitGlobalType(MCSymbolWasm *Sym) {
  // Not needed.
}

void WebAssemblyTargetWasmStreamer::emitEventType(MCSymbolWasm *Sym) {
  // Not needed.
}
void WebAssemblyTargetWasmStreamer::emitImportModule(MCSymbolWasm *Sym,
                                                     StringRef ModuleName) {
  Sym->setModuleName(ModuleName);
}
