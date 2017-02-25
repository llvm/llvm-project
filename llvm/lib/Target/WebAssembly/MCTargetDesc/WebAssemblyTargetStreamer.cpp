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
/// \brief This file defines WebAssembly-specific target streamer classes.
/// These are for implementing support for target-specific assembly directives.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetStreamer.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "WebAssemblyMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Wasm.h"
using namespace llvm;

WebAssemblyTargetStreamer::WebAssemblyTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) {}

WebAssemblyTargetAsmStreamer::WebAssemblyTargetAsmStreamer(
    MCStreamer &S, formatted_raw_ostream &OS)
    : WebAssemblyTargetStreamer(S), OS(OS) {}

WebAssemblyTargetELFStreamer::WebAssemblyTargetELFStreamer(MCStreamer &S)
    : WebAssemblyTargetStreamer(S) {}

WebAssemblyTargetWasmStreamer::WebAssemblyTargetWasmStreamer(MCStreamer &S)
    : WebAssemblyTargetStreamer(S) {}

static void PrintTypes(formatted_raw_ostream &OS, ArrayRef<MVT> Types) {
  bool First = true;
  for (MVT Type : Types) {
    if (First)
      First = false;
    else
      OS << ", ";
    OS << WebAssembly::TypeToString(Type);
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

void WebAssemblyTargetAsmStreamer::emitGlobal(ArrayRef<MVT> Types) {
  if (!Types.empty()) {
    OS << "\t.globalvar  \t";
    PrintTypes(OS, Types);
  }
}

void WebAssemblyTargetAsmStreamer::emitEndFunc() { OS << "\t.endfunc\n"; }

void WebAssemblyTargetAsmStreamer::emitIndirectFunctionType(
    StringRef name, SmallVectorImpl<MVT> &Params, SmallVectorImpl<MVT> &Results) {
  OS << "\t.functype\t" << name;
  if (Results.empty())
    OS << ", void";
  else {
    assert(Results.size() == 1);
    OS << ", " << WebAssembly::TypeToString(Results.front());
  }
  for (auto Ty : Params)
    OS << ", " << WebAssembly::TypeToString(Ty);
  OS << '\n';
}

void WebAssemblyTargetAsmStreamer::emitGlobalImport(StringRef name) {
  OS << "\t.import_global\t" << name << '\n';
}

void WebAssemblyTargetAsmStreamer::emitIndIdx(const MCExpr *Value) {
  OS << "\t.indidx  \t" << *Value << '\n';
}

void WebAssemblyTargetELFStreamer::emitParam(MCSymbol *Symbol,
                                             ArrayRef<MVT> Types) {
  // Nothing to emit; params are declared as part of the function signature.
}

void WebAssemblyTargetELFStreamer::emitResult(MCSymbol *Symbol,
                                              ArrayRef<MVT> Types) {
  // Nothing to emit; results are declared as part of the function signature.
}

void WebAssemblyTargetELFStreamer::emitLocal(ArrayRef<MVT> Types) {
  Streamer.EmitULEB128IntValue(Types.size());
  for (MVT Type : Types)
    Streamer.EmitIntValue(int64_t(WebAssembly::toValType(Type)), 1);
}

void WebAssemblyTargetELFStreamer::emitGlobal(ArrayRef<MVT> Types) {
  llvm_unreachable(".globalvar encoding not yet implemented");
}

void WebAssemblyTargetELFStreamer::emitEndFunc() {
  Streamer.EmitIntValue(WebAssembly::End, 1);
}

void WebAssemblyTargetELFStreamer::emitIndIdx(const MCExpr *Value) {
  llvm_unreachable(".indidx encoding not yet implemented");
}

void WebAssemblyTargetELFStreamer::emitIndirectFunctionType(
    StringRef name, SmallVectorImpl<MVT> &Params, SmallVectorImpl<MVT> &Results) {
  // Nothing to emit here. TODO: Re-design how linking works and re-evaluate
  // whether it's necessary for .o files to declare indirect function types.
}

void WebAssemblyTargetELFStreamer::emitGlobalImport(StringRef name) {
}

static unsigned MVT2WasmType(MVT Ty) {
  switch (Ty.SimpleTy) {
  case MVT::i32: return wasm::WASM_TYPE_I32;
  case MVT::i64: return wasm::WASM_TYPE_I64;
  case MVT::f32: return wasm::WASM_TYPE_F32;
  case MVT::f64: return wasm::WASM_TYPE_F64;
  default: llvm_unreachable("unsupported type");
  }
}

void WebAssemblyTargetWasmStreamer::emitParam(MCSymbol *Symbol,
                                              ArrayRef<MVT> Types) {
  SmallVector<unsigned, 4> Params;
  for (MVT Ty : Types)
    Params.push_back(MVT2WasmType(Ty));

  cast<MCSymbolWasm>(Symbol)->setParams(std::move(Params));
}

void WebAssemblyTargetWasmStreamer::emitResult(MCSymbol *Symbol,
                                               ArrayRef<MVT> Types) {
  SmallVector<unsigned, 4> Returns;
  for (MVT Ty : Types)
    Returns.push_back(MVT2WasmType(Ty));

  cast<MCSymbolWasm>(Symbol)->setReturns(std::move(Returns));
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
    Streamer.EmitULEB128IntValue(uint64_t(WebAssembly::toValType(Pair.first)));
  }
}

void WebAssemblyTargetWasmStreamer::emitGlobal(ArrayRef<MVT> Types) {
  // Encode the globals use by the funciton into the special .global_variables
  // section. This will later be decoded and turned into contents for the
  // Globals Section.
  Streamer.PushSection();
  Streamer.SwitchSection(Streamer.getContext()
                                 .getWasmSection(".global_variables", 0, 0));
  for (MVT Ty : Types)
    Streamer.EmitIntValue(uint64_t(WebAssembly::toValType(Ty)), 1);
  Streamer.PopSection();
}

void WebAssemblyTargetWasmStreamer::emitEndFunc() {
  llvm_unreachable(".end_func is not needed for direct wasm output");
}

void WebAssemblyTargetWasmStreamer::emitIndIdx(const MCExpr *Value) {
  llvm_unreachable(".indidx encoding not yet implemented");
}

void WebAssemblyTargetWasmStreamer::emitIndirectFunctionType(
    StringRef name, SmallVectorImpl<MVT> &Params, SmallVectorImpl<MVT> &Results) {
  // Nothing to emit here. TODO: Re-design how linking works and re-evaluate
  // whether it's necessary for .o files to declare indirect function types.
}

void WebAssemblyTargetWasmStreamer::emitGlobalImport(StringRef name) {
}
