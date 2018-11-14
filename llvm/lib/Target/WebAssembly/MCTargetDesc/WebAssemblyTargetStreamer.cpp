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

static void PrintTypes(formatted_raw_ostream &OS, ArrayRef<MVT> Types) {
  bool First = true;
  for (MVT Type : Types) {
    if (First)
      First = false;
    else
      OS << ", ";
    OS << WebAssembly::TypeToString(WebAssembly::toValType(Type));
  }
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitParam(MCSymbol *Symbol,
                                             ArrayRef<MVT> Types) {
  if (!Types.empty()) {
    OS << "\t.param  \t";

    // FIXME: Currently this applies to the "current" function; it may
    // be cleaner to specify an explicit symbol as part of the directive.

    PrintTypes(OS, Types);
  }
}

void WebAssemblyTargetAsmStreamer::emitResult(MCSymbol *Symbol,
                                              ArrayRef<MVT> Types) {
  if (!Types.empty()) {
    OS << "\t.result \t";

    // FIXME: Currently this applies to the "current" function; it may
    // be cleaner to specify an explicit symbol as part of the directive.

    PrintTypes(OS, Types);
  }
}

void WebAssemblyTargetAsmStreamer::emitLocal(ArrayRef<MVT> Types) {
  if (!Types.empty()) {
    OS << "\t.local  \t";
    PrintTypes(OS, Types);
  }
}

void WebAssemblyTargetAsmStreamer::emitEndFunc() { OS << "\t.endfunc\n"; }

void WebAssemblyTargetAsmStreamer::emitIndirectFunctionType(
    MCSymbolWasm *Symbol) {
  OS << "\t.functype\t" << Symbol->getName();
  if (Symbol->getSignature()->Returns.empty())
    OS << ", void";
  else {
    assert(Symbol->getSignature()->Returns.size() == 1);
    OS << ", "
       << WebAssembly::TypeToString(Symbol->getSignature()->Returns.front());
  }
  for (auto Ty : Symbol->getSignature()->Params)
    OS << ", " << WebAssembly::TypeToString(Ty);
  OS << '\n';
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

void WebAssemblyTargetWasmStreamer::emitParam(MCSymbol *Symbol,
                                              ArrayRef<MVT> Types) {
  // The Symbol already has its signature
}

void WebAssemblyTargetWasmStreamer::emitResult(MCSymbol *Symbol,
                                               ArrayRef<MVT> Types) {
  // The Symbol already has its signature
}

void WebAssemblyTargetWasmStreamer::emitLocal(ArrayRef<MVT> Types) {
  SmallVector<std::pair<MVT, uint32_t>, 4> Grouped;
  for (MVT Type : Types) {
    if (Grouped.empty() || Grouped.back().first != Type)
      Grouped.push_back(std::make_pair(Type, 1));
    else
      ++Grouped.back().second;
  }

  Streamer.EmitULEB128IntValue(Grouped.size());
  for (auto Pair : Grouped) {
    Streamer.EmitULEB128IntValue(Pair.second);
    emitValueType(WebAssembly::toValType(Pair.first));
  }
}

void WebAssemblyTargetWasmStreamer::emitEndFunc() {
  llvm_unreachable(".end_func is not needed for direct wasm output");
}

void WebAssemblyTargetWasmStreamer::emitIndIdx(const MCExpr *Value) {
  llvm_unreachable(".indidx encoding not yet implemented");
}

void WebAssemblyTargetWasmStreamer::emitIndirectFunctionType(
    MCSymbolWasm *Symbol) {
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
