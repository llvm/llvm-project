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

#include "CodeGenInstruction.h"
#include "CodeGenTarget.h"
#include "X86DisassemblerTables.h"
#include "X86RecognizableInstr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

namespace {

class X86MnemonicTablesEmitter {
  CodeGenTarget Target;

public:
  X86MnemonicTablesEmitter(RecordKeeper &R) : Target(R) {}

  // Output X86 mnemonic tables.
  void run(raw_ostream &OS);
};

void X86MnemonicTablesEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("X86 Mnemonic tables", OS);
  OS << "namespace llvm {\nnamespace X86 {\n\n";
  Record *AsmWriter = Target.getAsmWriter();
  unsigned Variant = AsmWriter->getValueAsInt("Variant");

  // Hold all instructions grouped by mnemonic
  StringMap<SmallVector<const CodeGenInstruction *, 0>> MnemonicToCGInstrMap;

  // Unused
  X86Disassembler::DisassemblerTables Tables;
  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();
  for (unsigned II = 0, IE = NumberedInstructions.size(); II != IE; ++II) {
    const CodeGenInstruction *I = NumberedInstructions[II];
    X86Disassembler::RecognizableInstr RI(Tables, *I, II);
    Record *Def = I->TheDef;
    if ( // Filter non-X86 instructions
        !Def->isSubClassOf("X86Inst") ||
        // Skip pseudo instructions as they may contain non-alnum characters in
        // mnemonic
        (RI.IsCodeGenOnly && !RI.ForceDisassemble) ||
        // Non-parsable instruction defs contain prefix as part of AsmString
        Def->getValueAsString("AsmVariantName") == "NonParsable" ||
        // Skip CodeGenInstructions that are not real standalone instructions
        RI.Form == X86Local::PrefixByte || RI.Form == X86Local::Pseudo)
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
      OS << "\treturn Opcode == " << CGI->TheDef->getName() << ";\n}\n\n";
    } else {
      OS << "\tswitch (Opcode) {\n";
      for (const CodeGenInstruction *CGI : Mnemonics) {
        OS << "\tcase " << CGI->TheDef->getName() << ":\n";
      }
      OS << "\t\treturn true;\n\t}\n\treturn false;\n}\n\n";
    }
  }
  OS << "#endif // GET_X86_MNEMONIC_TABLES_CPP\n\n";
  OS << "} // end namespace X86\n} // end namespace llvm";
}

} // namespace

namespace llvm {
void EmitX86MnemonicTables(RecordKeeper &RK, raw_ostream &OS) {
  X86MnemonicTablesEmitter(RK).run(OS);
}
} // namespace llvm
