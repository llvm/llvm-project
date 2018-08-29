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

// Find all register WebAssembly instructions and their corresponding stack
// instructions. For each pair, emit a switch case of the form
//
//   case WebAssembly::RegisterInstr: return WebAssembly::StackInstr;
//
// For example,
//
//   case WebAssembly::ADD_I32: return WebAssembly::ADD_I32_S;
//
// This is useful for converting instructions from their register form to their
// equivalent stack form.
void EmitWebAssemblyStackifier(RecordKeeper &RK, raw_ostream &OS) {
  Record *InstrClass = RK.getClass("WebAssemblyInst");
  for (auto &RecordPair : RK.getDefs()) {
    if (!RecordPair.second->isSubClassOf(InstrClass))
      continue;
    bool IsStackBased = RecordPair.second->getValueAsBit("StackBased");
    if (IsStackBased)
      continue;
    OS << "  case WebAssembly::" << RecordPair.first << ": return "
       << "WebAssembly::" << RecordPair.first << "_S;\n";
  }
}

} // namespace llvm
