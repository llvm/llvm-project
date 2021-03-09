//===---------- CodeBeadsGen.cpp - Code Beads Generator -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// CodeBeads are data fields carrying auxiliary information for instructions.
//
// Under the hood it's simply implemented by a `bits` field (with arbitrary
// length) in each TG instruction description, where this TG backend will
// generate a helper function to access it.
//
// This is especially useful for expressing variable length encoding
// instructions and complex addressing modes. Since in those cases each
// instruction is usually associated with large amount of information like
// addressing mode details used on a specific operand. Instead of retreating to
// ad-hoc methods to figure out these information when encoding an instruction,
// CodeBeads provide a clean table for the instruction encoder to lookup.
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <string>
#include <vector>
using namespace llvm;

namespace {

class CodeBeadsGen {
  RecordKeeper &Records;

public:
  CodeBeadsGen(RecordKeeper &R) : Records(R) {}
  void run(raw_ostream &OS);
};

void CodeBeadsGen::run(raw_ostream &OS) {
  CodeGenTarget Target(Records);
  std::vector<Record *> Insts = Records.getAllDerivedDefinitions("Instruction");

  // For little-endian instruction bit encodings, reverse the bit order
  Target.reverseBitsForLittleEndianEncoding();

  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructionsByEnumValue();

  // Emit function declaration
  OS << "const uint8_t *llvm::" << Target.getInstNamespace();
  OS << "::getMCInstrBeads(unsigned Opcode) {\n";

  // First, get the maximum bit length among all beads. And do some
  // simple validation
  unsigned MaxBitLength = 0;

  for (const CodeGenInstruction *CGI : NumberedInstructions) {
    Record *R = CGI->TheDef;
    if (!R->getValue("Beads"))
      continue;

    BitsInit *BI = R->getValueAsBitsInit("Beads");
    if (!BI->isComplete()) {
      PrintFatalError(R->getLoc(), "Record `" + R->getName() +
                                       "', bit field 'Beads' is not complete");
    }

    MaxBitLength = std::max(MaxBitLength, BI->getNumBits());
  }

  // Number of bytes
  unsigned Parts = MaxBitLength / 8;

  // Emit instruction base values
  OS << "  static const uint8_t InstBits[][" << Parts << "] = {\n";
  for (const CodeGenInstruction *CGI : NumberedInstructions) {
    Record *R = CGI->TheDef;

    if (R->getValueAsString("Namespace") == "TargetOpcode" ||
        !R->getValue("Beads")) {
      OS << "\t{ 0x0 },\t// ";
      if (R->getValueAsBit("isPseudo"))
        OS << "(Pseudo) ";
      OS << R->getName() << "\n";
      continue;
    }

    BitsInit *BI = R->getValueAsBitsInit("Beads");

    // Convert to byte array:
    // [dcba] -> [a][b][c][d]
    OS << "\t{";
    for (unsigned p = 0; p < Parts; ++p) {
      unsigned Right = 8 * p;
      unsigned Left = Right + 8;

      uint8_t Value = 0;
      for (unsigned i = Right; i != Left; ++i) {
        unsigned Shift = i % 8;
        if (auto *B = dyn_cast<BitInit>(BI->getBit(i))) {
          Value |= (static_cast<uint8_t>(B->getValue()) << Shift);
        } else {
          PrintFatalError(R->getLoc(), "Record `" + R->getName() +
                                           "', bit 'Beads[" + Twine(i) +
                                           "]' is not defined");
        }
      }

      if (p)
        OS << ',';
      OS << " 0x";
      OS.write_hex(Value);
      OS << "";
    }
    OS << " }," << '\t' << "// " << R->getName() << "\n";
  }
  OS << "\t{ 0x0 }\n  };\n";

  // Emit initial function code
  OS << "  return InstBits[Opcode];\n"
     << "}\n\n";
}

} // End anonymous namespace

namespace llvm {

void EmitCodeBeads(RecordKeeper &RK, raw_ostream &OS) {
  emitSourceFileHeader("Machine Code Beads", OS);
  CodeBeadsGen(RK).run(OS);
}

} // namespace llvm
