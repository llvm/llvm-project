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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
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

void WebAssemblyTargetAsmStreamer::emitParam(ArrayRef<MVT> Types) {
  OS << "\t.param  \t";
  PrintTypes(OS, Types);
}

void WebAssemblyTargetAsmStreamer::emitResult(ArrayRef<MVT> Types) {
  OS << "\t.result \t";
  PrintTypes(OS, Types);
}

void WebAssemblyTargetAsmStreamer::emitLocal(ArrayRef<MVT> Types) {
  if (!Types.empty()) {
    OS << "\t.local  \t";
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

void WebAssemblyTargetELFStreamer::emitParam(ArrayRef<MVT> Types) {
  // Nothing to emit; params are declared as part of the function signature.
}

void WebAssemblyTargetELFStreamer::emitResult(ArrayRef<MVT> Types) {
  // Nothing to emit; results are declared as part of the function signature.
}

void WebAssemblyTargetELFStreamer::emitLocal(ArrayRef<MVT> Types) {
  Streamer.EmitULEB128IntValue(Types.size());
  for (MVT Type : Types)
    Streamer.EmitIntValue(int64_t(WebAssembly::toValType(Type)), 1);
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

void WebAssemblyTargetWasmStreamer::emitParam(ArrayRef<MVT> Types) {
  // Nothing to emit; params are declared as part of the function signature.
}

void WebAssemblyTargetWasmStreamer::emitResult(ArrayRef<MVT> Types) {
  // Nothing to emit; results are declared as part of the function signature.
}

void WebAssemblyTargetWasmStreamer::emitLocal(ArrayRef<MVT> Types) {
  Streamer.EmitULEB128IntValue(Types.size());
  for (MVT Type : Types)
    Streamer.EmitIntValue(int64_t(WebAssembly::toValType(Type)), 1);
}

void WebAssemblyTargetWasmStreamer::emitEndFunc() {
  Streamer.EmitIntValue(WebAssembly::End, 1);
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
