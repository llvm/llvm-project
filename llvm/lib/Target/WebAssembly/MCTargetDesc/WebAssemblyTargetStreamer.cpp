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
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/ELF.h"
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
  OS << "\t.local  \t";
  PrintTypes(OS, Types);
}

void WebAssemblyTargetAsmStreamer::emitEndFunc() { OS << "\t.endfunc\n"; }

void WebAssemblyTargetAsmStreamer::emitIndirectFunctionType(
    StringRef name, SmallVectorImpl<MVT> &SignatureVTs, size_t NumResults) {
  OS << "\t.functype\t" << name;
  if (NumResults == 0)
    OS << ", void";
  for (auto Ty : SignatureVTs) {
    OS << ", " << WebAssembly::TypeToString(Ty);
  }
  OS << "\n";
}

void WebAssemblyTargetAsmStreamer::emitIndIdx(const MCExpr *Value) {
  OS << "\t.indidx  \t" << *Value << '\n';
}

// FIXME: What follows is not the real binary encoding.

static void EncodeTypes(MCStreamer &Streamer, ArrayRef<MVT> Types) {
  Streamer.EmitIntValue(Types.size(), sizeof(uint64_t));
  for (MVT Type : Types)
    Streamer.EmitIntValue(Type.SimpleTy, sizeof(uint64_t));
}

void WebAssemblyTargetELFStreamer::emitParam(ArrayRef<MVT> Types) {
  Streamer.EmitIntValue(WebAssembly::DotParam, sizeof(uint64_t));
  EncodeTypes(Streamer, Types);
}

void WebAssemblyTargetELFStreamer::emitResult(ArrayRef<MVT> Types) {
  Streamer.EmitIntValue(WebAssembly::DotResult, sizeof(uint64_t));
  EncodeTypes(Streamer, Types);
}

void WebAssemblyTargetELFStreamer::emitLocal(ArrayRef<MVT> Types) {
  Streamer.EmitIntValue(WebAssembly::DotLocal, sizeof(uint64_t));
  EncodeTypes(Streamer, Types);
}

void WebAssemblyTargetELFStreamer::emitEndFunc() {
  Streamer.EmitIntValue(WebAssembly::DotEndFunc, sizeof(uint64_t));
}

void WebAssemblyTargetELFStreamer::emitIndIdx(const MCExpr *Value) {
  Streamer.EmitIntValue(WebAssembly::DotIndIdx, sizeof(uint64_t));
  Streamer.EmitValue(Value, sizeof(uint64_t));
}
