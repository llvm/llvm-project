//==- X86MnemonicTables.cpp - Generate mnemonic extraction tables. -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting tables that group
// instructions by their mnemonic name wrt AsmWriter Variant (e.g. isADD, etc).
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenTarget.h"
#include "X86RecognizableInstr.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {

class X86MnemonicTablesEmitter {
  const CodeGenTarget Target;

public:
  X86MnemonicTablesEmitter(const RecordKeeper &R) : Target(R) {}

  // Output X86 mnemonic tables.
  void run(raw_ostream &OS);
};
} // namespace

void X86MnemonicTablesEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("X86 Mnemonic tables", OS);
  OS << "namespace llvm {\nnamespace X86 {\n\n";
  const Record *AsmWriter = Target.getAsmWriter();
  unsigned Variant = AsmWriter->getValueAsInt("Variant");

  // Hold all instructions grouped by mnemonic
  StringMap<SmallVector<const CodeGenInstruction *, 0>> MnemonicToCGInstrMap;

  for (const CodeGenInstruction *I : Target.getInstructions()) {
    const Record *Def = I->TheDef;
    // Filter non-X86 instructions.
    if (!Def->isSubClassOf("X86Inst"))
      continue;
    X86Disassembler::RecognizableInstrBase RI(*I);
    if (!RI.shouldBeEmitted())
      continue;
    if ( // Non-parsable instruction defs contain prefix as part of AsmString
        Def->getValueAsString("AsmVariantName") == "NonParsable" ||
        // Skip prefix byte
        RI.Form == X86Local::PrefixByte)
      continue;
    std::string Mnemonic = X86Disassembler::getMnemonic(I, Variant);
    MnemonicToCGInstrMap[Mnemonic].push_back(I);
  }

  OS << "#ifdef GET_X86_MNEMONIC_TABLES_H\n";
  OS << "#undef GET_X86_MNEMONIC_TABLES_H\n\n";
  for (StringRef Mnemonic : MnemonicToCGInstrMap.keys())
    OS << "bool is" << Mnemonic << "(unsigned Opcode);\n";
  OS << "#endif // GET_X86_MNEMONIC_TABLES_H\n\n";

  OS << "#ifdef GET_X86_MNEMONIC_TABLES_CPP\n";
  OS << "#undef GET_X86_MNEMONIC_TABLES_CPP\n\n";
  for (StringRef Mnemonic : MnemonicToCGInstrMap.keys()) {
    OS << "bool is" << Mnemonic << "(unsigned Opcode) {\n";
    auto Mnemonics = MnemonicToCGInstrMap[Mnemonic];
    if (Mnemonics.size() == 1) {
      const CodeGenInstruction *CGI = *Mnemonics.begin();
      OS << "\treturn Opcode == " << CGI->getName() << ";\n}\n\n";
    } else {
      OS << "\tswitch (Opcode) {\n";
      for (const CodeGenInstruction *CGI : Mnemonics) {
        OS << "\tcase " << CGI->getName() << ":\n";
      }
      OS << "\t\treturn true;\n\t}\n\treturn false;\n}\n\n";
    }
  }
  OS << "#endif // GET_X86_MNEMONIC_TABLES_CPP\n\n";
  OS << "} // end namespace X86\n} // end namespace llvm";
}

static TableGen::Emitter::OptClass<X86MnemonicTablesEmitter>
    X("gen-x86-mnemonic-tables", "Generate X86 mnemonic tables");
