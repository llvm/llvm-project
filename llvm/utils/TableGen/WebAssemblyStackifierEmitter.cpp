//===- WebAssemblyStackifierEmitter.cpp - Stackifier cases ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file emits the switch statement cases to translate WebAssembly
// instructions to their stack forms.
//
//===----------------------------------------------------------------------===//

#include "WebAssemblyDisassemblerEmitter.h"
#include "llvm/TableGen/Record.h"

namespace llvm {

void EmitWebAssemblyStackifier(RecordKeeper &RK, raw_ostream &OS) {
  Record *InstrClass = RK.getClass("WebAssemblyInst");
  for (auto &RecordPair : RK.getDefs()) {
    if (!RecordPair.second->isSubClassOf(InstrClass)) continue;
    bool IsStackBased = RecordPair.second->getValueAsBit("StackBased");
    if (IsStackBased) continue;
    OS << "  case WebAssembly::" << RecordPair.first << ": return "
       << "WebAssembly::" << RecordPair.first << "_S; break;\n";
  }
}

} // namespace llvm
