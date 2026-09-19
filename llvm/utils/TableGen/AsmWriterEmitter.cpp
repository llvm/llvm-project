//===- AsmWriterEmitter.cpp - Generate an assembly writer -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits an assembly printer for the current target.
// Note that this is currently fairly skeletal, but will grow over time.
//
//===----------------------------------------------------------------------===//

#include "Basic/SequenceToOffsetTable.h"
#include "Common/AsmWriterInst.h"
#include "Common/CodeGenInstAlias.h"
#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenRegisters.h"
#include "Common/CodeGenTarget.h"
#include "Common/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "asm-writer-emitter"

namespace {

class AsmWriterEmitter {
  const RecordKeeper &Records;
  CodeGenTarget Target;
  ArrayRef<const CodeGenInstruction *> NumberedInstructions;
  std::vector<AsmWriterInst> Instructions;

public:
  AsmWriterEmitter(const RecordKeeper &R);

  void run(raw_ostream &o);

private:
  void EmitGetMnemonic(
      raw_ostream &o,
      std::vector<std::vector<std::string>> &TableDrivenOperandPrinters,
      unsigned &BitsLeft, unsigned &AsmStrBits);
  void EmitPrintInstruction(
      raw_ostream &o,
      std::vector<std::vector<std::string>> &TableDrivenOperandPrinters,
      unsigned &BitsLeft, unsigned &AsmStrBits);
  void EmitGetRegisterName(raw_ostream &o);
  void EmitPrintAliasInstruction(raw_ostream &O);

  void FindUniqueOperandCommands(std::vector<std::string> &UOC,
                                 std::vector<std::vector<unsigned>> &InstIdxs,
                                 std::vector<unsigned> &InstOpsUsed,
                                 bool PassSubtarget) const;

  // Emit a flat function-pointer table covering all opcodes.  Each unique
  // operand-printing sequence in the overflow set becomes one printPattern_N
  // static helper; instructions sharing a sequence share the same helper.
  // The table is indexed directly by getOpcode(), so dispatch is a single
  // unconditional indirect call with no switch or branch.
  void EmitOpcodePatternTable(raw_ostream &O, StringRef TargetName,
                              StringRef ClassName, bool PassSubtarget);
  void EmitOpcodePatternDispatch(raw_ostream &O, StringRef TargetName,
                                 bool PassSubtarget);
};

} // end anonymous namespace

void AsmWriterEmitter::FindUniqueOperandCommands(
    std::vector<std::string> &UniqueOperandCommands,
    std::vector<std::vector<unsigned>> &InstIdxs,
    std::vector<unsigned> &InstOpsUsed, bool PassSubtarget) const {
  // This vector parallels UniqueOperandCommands, keeping track of which
  // instructions each case are used for.  It is a comma separated string of
  // enums.
  std::vector<std::string> InstrsForCase;
  InstrsForCase.resize(UniqueOperandCommands.size());
  InstOpsUsed.assign(UniqueOperandCommands.size(), 0);

  for (size_t i = 0, e = Instructions.size(); i != e; ++i) {
    const AsmWriterInst &Inst = Instructions[i];
    if (Inst.Operands.empty())
      continue; // Instruction already done.

    std::string Command =
        "    " + Inst.Operands[0].getCode(PassSubtarget) + "\n";

    // Check to see if we already have 'Command' in UniqueOperandCommands.
    // If not, add it.
    auto I = llvm::find(UniqueOperandCommands, Command);
    if (I != UniqueOperandCommands.end()) {
      size_t idx = I - UniqueOperandCommands.begin();
      InstrsForCase[idx] += ", ";
      InstrsForCase[idx] += Inst.CGI->getName();
      InstIdxs[idx].push_back(i);
    } else {
      UniqueOperandCommands.push_back(std::move(Command));
      InstrsForCase.push_back(Inst.CGI->getName().str());
      InstIdxs.emplace_back();
      InstIdxs.back().push_back(i);

      // This command matches one operand so far.
      InstOpsUsed.push_back(1);
    }
  }

  // For each entry of UniqueOperandCommands, there is a set of instructions
  // that uses it.  If the next command of all instructions in the set are
  // identical, fold it into the command.
  for (size_t CommandIdx = 0, e = UniqueOperandCommands.size(); CommandIdx != e;
       ++CommandIdx) {

    const auto &Idxs = InstIdxs[CommandIdx];

    for (unsigned Op = 1;; ++Op) {
      // Find the first instruction in the set.
      const AsmWriterInst &FirstInst = Instructions[Idxs.front()];
      // If this instruction has no more operands, we isn't anything to merge
      // into this command.
      if (FirstInst.Operands.size() == Op)
        break;

      // Otherwise, scan to see if all of the other instructions in this command
      // set share the operand.
      if (any_of(drop_begin(Idxs), [&](unsigned Idx) {
            const AsmWriterInst &OtherInst = Instructions[Idx];
            return OtherInst.Operands.size() == Op ||
                   OtherInst.Operands[Op] != FirstInst.Operands[Op];
          }))
        break;

      // Okay, everything in this command set has the same next operand.  Add it
      // to UniqueOperandCommands and remember that it was consumed.
      std::string Command =
          "    " + FirstInst.Operands[Op].getCode(PassSubtarget) + "\n";

      UniqueOperandCommands[CommandIdx] += Command;
      InstOpsUsed[CommandIdx]++;
    }
  }

  // Prepend some of the instructions each case is used for onto the case val.
  for (unsigned i = 0, e = InstrsForCase.size(); i != e; ++i) {
    std::string Instrs = InstrsForCase[i];
    if (Instrs.size() > 70) {
      Instrs.erase(Instrs.begin() + 70, Instrs.end());
      Instrs += "...";
    }

    if (!Instrs.empty())
      UniqueOperandCommands[i] =
          "    // " + Instrs + "\n" + UniqueOperandCommands[i];
  }
}

static void UnescapeString(std::string &Str) {
  for (unsigned i = 0; i != Str.size(); ++i) {
    if (Str[i] == '\\' && i != Str.size() - 1) {
      switch (Str[i + 1]) {
      default:
        continue; // Don't execute the code after the switch.
      case 'a':
        Str[i] = '\a';
        break;
      case 'b':
        Str[i] = '\b';
        break;
      case 'e':
        Str[i] = 27;
        break;
      case 'f':
        Str[i] = '\f';
        break;
      case 'n':
        Str[i] = '\n';
        break;
      case 'r':
        Str[i] = '\r';
        break;
      case 't':
        Str[i] = '\t';
        break;
      case 'v':
        Str[i] = '\v';
        break;
      case '"':
        Str[i] = '\"';
        break;
      case '\'':
        Str[i] = '\'';
        break;
      case '\\':
        Str[i] = '\\';
        break;
      }
      // Nuke the second character.
      Str.erase(Str.begin() + i + 1);
    }
  }
}

/// UnescapeAliasString - Supports literal braces in InstAlias asm string which
/// are escaped with '\\' to avoid being interpreted as variants. Braces must
/// be unescaped before c++ code is generated as (e.g.):
///
///   AsmString = "foo \{$\x01\}";
///
/// causes non-standard escape character warnings.
static void UnescapeAliasString(std::string &Str) {
  for (unsigned i = 0; i != Str.size(); ++i) {
    if (Str[i] == '\\' && i != Str.size() - 1) {
      switch (Str[i + 1]) {
      default:
        continue; // Don't execute the code after the switch.
      case '{':
        Str[i] = '{';
        break;
      case '}':
        Str[i] = '}';
        break;
      }
      // Nuke the second character.
      Str.erase(Str.begin() + i + 1);
    }
  }
}

void AsmWriterEmitter::EmitGetMnemonic(
    raw_ostream &O,
    std::vector<std::vector<std::string>> &TableDrivenOperandPrinters,
    unsigned &BitsLeft, unsigned &AsmStrBits) {
  const Record *AsmWriter = Target.getAsmWriter();
  StringRef ClassName = AsmWriter->getValueAsString("AsmWriterClassName");
  bool PassSubtarget = AsmWriter->getValueAsInt("PassSubtarget");

  O << "/// getMnemonic - This method is automatically generated by "
       "tablegen\n"
       "/// from the instruction set description.\n"
       "std::pair<const char *, uint64_t>\n"
    << Target.getName() << ClassName
    << "::getMnemonic(const MCInst &MI) const {\n";

  // Build an aggregate string, and build a table of offsets into it.
  SequenceToOffsetTable<std::string> StringTable;

  /// OpcodeInfo - This encodes the index of the string to use for the first
  /// chunk of the output as well as indices used for operand printing.
  std::vector<uint64_t> OpcodeInfo(NumberedInstructions.size());
  const unsigned OpcodeInfoBits = 64;

  // Add all strings to the string table upfront so it can generate an optimized
  // representation.
  for (AsmWriterInst &AWI : Instructions) {
    if (AWI.Operands[0].OperandType == AsmWriterOperand::isLiteralTextOperand &&
        !AWI.Operands[0].Str.empty()) {
      std::string Str = AWI.Operands[0].Str;
      UnescapeString(Str);
      StringTable.add(Str);
    }
  }

  StringTable.layout();

  unsigned MaxStringIdx = 0;
  for (AsmWriterInst &AWI : Instructions) {
    unsigned Idx;
    if (AWI.Operands[0].OperandType != AsmWriterOperand::isLiteralTextOperand ||
        AWI.Operands[0].Str.empty()) {
      // Something handled by the asmwriter printer, but with no leading string.
      Idx = StringTable.get("");
    } else {
      std::string Str = AWI.Operands[0].Str;
      UnescapeString(Str);
      Idx = StringTable.get(Str);
      MaxStringIdx = std::max(MaxStringIdx, Idx);

      // Nuke the string from the operand list.  It is now handled!
      AWI.Operands.erase(AWI.Operands.begin());
    }

    // Bias offset by one since we want 0 as a sentinel.
    OpcodeInfo[AWI.CGIIndex] = Idx + 1;
  }

  // Figure out how many bits we used for the string index.
  AsmStrBits = Log2_32_Ceil(MaxStringIdx + 2);

  // To reduce code size, we compactify common instructions into a few bits
  // in the opcode-indexed table.
  BitsLeft = OpcodeInfoBits - AsmStrBits;

  while (true) {
    std::vector<std::string> UniqueOperandCommands;
    std::vector<std::vector<unsigned>> InstIdxs;
    std::vector<unsigned> NumInstOpsHandled;
    FindUniqueOperandCommands(UniqueOperandCommands, InstIdxs,
                              NumInstOpsHandled, PassSubtarget);

    // If we ran out of operands to print, we're done.
    if (UniqueOperandCommands.empty())
      break;

    // Compute the number of bits we need to represent these cases, this is
    // ceil(log2(numentries)).
    unsigned NumBits = Log2_32_Ceil(UniqueOperandCommands.size());

    // If we don't have enough bits for this operand, don't include it.
    if (NumBits > BitsLeft) {
      LLVM_DEBUG(dbgs() << "Not enough bits to densely encode " << NumBits
                        << " more bits\n");
      break;
    }

    // Otherwise, we can include this in the initial lookup table.  Add it in.
    for (size_t i = 0, e = InstIdxs.size(); i != e; ++i) {
      unsigned NumOps = NumInstOpsHandled[i];
      for (unsigned Idx : InstIdxs[i]) {
        OpcodeInfo[Instructions[Idx].CGIIndex] |=
            (uint64_t)i << (OpcodeInfoBits - BitsLeft);
        // Remove the info about this operand from the instruction.
        AsmWriterInst &Inst = Instructions[Idx];
        if (!Inst.Operands.empty()) {
          assert(NumOps <= Inst.Operands.size() &&
                 "Can't remove this many ops!");
          Inst.Operands.erase(Inst.Operands.begin(),
                              Inst.Operands.begin() + NumOps);
        }
      }
    }
    BitsLeft -= NumBits;

    // Remember the handlers for this set of operands.
    TableDrivenOperandPrinters.push_back(std::move(UniqueOperandCommands));
  }

  // Emit the string table itself.
  StringTable.emitStringLiteralDef(O, "  static const char AsmStrs[]");

  // Emit the lookup tables in pieces to minimize wasted bytes.
  unsigned BytesNeeded = ((OpcodeInfoBits - BitsLeft) + 7) / 8;
  unsigned Table = 0, Shift = 0;
  SmallString<128> BitsString;
  raw_svector_ostream BitsOS(BitsString);
  // If the total bits is more than 32-bits we need to use a 64-bit type.
  BitsOS << "  uint" << ((BitsLeft < (OpcodeInfoBits - 32)) ? 64 : 32)
         << "_t Bits = 0;\n";
  while (BytesNeeded != 0) {
    // Figure out how big this table section needs to be, but no bigger than 4.
    unsigned TableSize = std::min(llvm::bit_floor(BytesNeeded), 4u);
    BytesNeeded -= TableSize;
    TableSize *= 8; // Convert to bits;
    uint64_t Mask = (1ULL << TableSize) - 1;
    O << "  static const uint" << TableSize << "_t OpInfo" << Table
      << "[] = {\n";
    for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
      O << "    " << ((OpcodeInfo[i] >> Shift) & Mask) << "U,\t// "
        << NumberedInstructions[i]->getName() << '\n';
    }
    O << "  };\n\n";
    // Emit string to combine the individual table lookups.
    BitsOS << "  Bits |= ";
    // If the total bits is more than 32-bits we need to use a 64-bit type.
    if (BitsLeft < (OpcodeInfoBits - 32))
      BitsOS << "(uint64_t)";
    BitsOS << "OpInfo" << Table << "[MI.getOpcode()] << " << Shift << ";\n";
    // Prepare the shift for the next iteration and increment the table count.
    Shift += TableSize;
    ++Table;
  }

  O << "  // Emit the opcode for the instruction.\n";
  O << BitsString;

  // Make sure we don't return an invalid pointer if bits is 0
  O << "  if (Bits == 0)\n"
       "    return {nullptr, Bits};\n";

  // Return mnemonic string and bits.
  O << "  return {AsmStrs+(Bits & " << (1 << AsmStrBits) - 1
    << ")-1, Bits};\n\n";

  O << "}\n";
}

/// EmitPrintInstruction - Generate the code for the "printInstruction" method
/// implementation. Destroys all instances of AsmWriterInst information, by
/// clearing the Instructions vector.
void AsmWriterEmitter::EmitPrintInstruction(
    raw_ostream &O,
    std::vector<std::vector<std::string>> &TableDrivenOperandPrinters,
    unsigned &BitsLeft, unsigned &AsmStrBits) {
  const unsigned OpcodeInfoBits = 64;
  const Record *AsmWriter = Target.getAsmWriter();
  StringRef ClassName = AsmWriter->getValueAsString("AsmWriterClassName");
  bool PassSubtarget = AsmWriter->getValueAsInt("PassSubtarget");

  // Emit static printPattern_N helpers and the function-pointer table before
  // printInstruction() opens.  The table covers every opcode; overflow
  // instructions dispatch to a deduplicated pattern helper, non-overflow
  // instructions dispatch to a no-op stub.  Indirect calls through a const
  // function-pointer table cannot be inlined, so printInstruction() stays small
  // regardless of instruction-set size.
  bool HasOverflow = llvm::any_of(Instructions, [](const AsmWriterInst &AWI) {
    return !AWI.Operands.empty();
  });
  if (HasOverflow)
    EmitOpcodePatternTable(O, Target.getName(), ClassName, PassSubtarget);

  // This function has some huge switch statements that causing excessive
  // compile time in LLVM profile instrumenation build. This print function
  // usually is not frequently called in compilation. Here we disable the
  // profile instrumenation for this function.
  O << "/// printInstruction - This method is automatically generated by "
       "tablegen\n"
       "/// from the instruction set description.\n"
       "LLVM_NO_PROFILE_INSTRUMENT_FUNCTION\n"
       "void "
    << Target.getName() << ClassName
    << "::printInstruction(const MCInst *MI, uint64_t Address, "
    << (PassSubtarget ? "const MCSubtargetInfo &STI, " : "")
    << "raw_ostream &O) {\n";

  // Emit the initial tab character.
  O << "  O << \"\\t\";\n\n";

  // Emit the starting string.
  O << "  auto MnemonicInfo = getMnemonic(*MI);\n\n";
  O << "  O << MnemonicInfo.first;\n\n";

  O << "  uint" << ((BitsLeft < (OpcodeInfoBits - 32)) ? 64 : 32)
    << "_t Bits = MnemonicInfo.second;\n"
    << "  assert(Bits != 0 && \"Cannot print this instruction.\");\n";

  // Output the table driven operand information.
  BitsLeft = OpcodeInfoBits - AsmStrBits;
  for (unsigned i = 0, e = TableDrivenOperandPrinters.size(); i != e; ++i) {
    std::vector<std::string> &Commands = TableDrivenOperandPrinters[i];

    // Compute the number of bits we need to represent these cases, this is
    // ceil(log2(numentries)).
    unsigned NumBits = Log2_32_Ceil(Commands.size());
    assert(NumBits <= BitsLeft && "consistency error");

    // Emit code to extract this field from Bits.
    O << "\n  // Fragment " << i << " encoded into " << NumBits << " bits for "
      << Commands.size() << " unique commands.\n";

    if (Commands.size() == 2) {
      // Emit two possibilitys with if/else.
      O << "  if ((Bits >> " << (OpcodeInfoBits - BitsLeft) << ") & "
        << ((1 << NumBits) - 1) << ") {\n"
        << Commands[1] << "  } else {\n"
        << Commands[0] << "  }\n\n";
    } else if (Commands.size() == 1) {
      // Emit a single possibility.
      O << Commands[0] << "\n\n";
    } else {
      O << "  switch ((Bits >> " << (OpcodeInfoBits - BitsLeft) << ") & "
        << ((1 << NumBits) - 1) << ") {\n"
        << "  default: llvm_unreachable(\"Invalid command number.\");\n";

      // Print out all the cases.
      for (unsigned j = 0, e = Commands.size(); j != e; ++j) {
        O << "  case " << j << ":\n";
        O << Commands[j];
        O << "    break;\n";
      }
      O << "  }\n\n";
    }
    BitsLeft -= NumBits;
  }

  // Emit the overflow-operand dispatch: a single unconditional indirect call
  // through the function-pointer table built above.
  if (HasOverflow)
    EmitOpcodePatternDispatch(O, Target.getName(), PassSubtarget);

  O << "}\n";
}

void AsmWriterEmitter::EmitOpcodePatternTable(raw_ostream &O,
                                              StringRef TargetName,
                                              StringRef ClassName,
                                              bool PassSubtarget) {
  // Collect instructions with overflow operands (not yet handled by the
  // table-driven bit-encoding path).
  std::vector<AsmWriterInst> OpcodeInsts;
  for (const AsmWriterInst &AWI : Instructions)
    if (!AWI.Operands.empty())
      OpcodeInsts.push_back(AWI);

  assert(!OpcodeInsts.empty() && "caller should have checked HasOverflow");

  // Serialize each instruction's complete operand sequence into a string key.
  // Instructions with identical keys share one printPattern_N function.
  // getCode() with Receiver="P->" turns member calls like printOperand(...)
  // into P->printOperand(...), which is correct for a static free function.
  using PatternBody = std::vector<std::string>;
  std::map<PatternBody, unsigned> PatternMap;
  SmallVector<PatternBody, 64> Patterns;

  // Table: index by CGIIndex, value = pattern index (~0U for non-overflow).
  std::vector<unsigned> OpcodeToPattern(NumberedInstructions.size(), ~0U);

  for (const AsmWriterInst &AWI : OpcodeInsts) {
    PatternBody Body;
    for (const AsmWriterOperand &Op : AWI.Operands)
      Body.push_back(Op.getCode(PassSubtarget, "P->"));
    auto [It, Inserted] = PatternMap.emplace(Body, Patterns.size());
    if (Inserted)
      Patterns.push_back(Body);
    OpcodeToPattern[AWI.CGIIndex] = It->second;
  }

  std::string FullClassName = (TargetName + ClassName).str();
  std::string PtrTableName = (TargetName + "InstPrinters").str();
  std::string ParamList = "    " + FullClassName +
                          " *, const MCInst *,\n"
                          "    uint64_t," +
                          (PassSubtarget ? " const MCSubtargetInfo &," : "") +
                          " raw_ostream &";

  // printPattern_None: no-op for instructions fully handled by table-driven
  // path.
  O << "static void printPattern_None(\n" << ParamList << ") {}\n\n";

  // One printPattern_N per unique operand sequence.
  for (unsigned PIdx = 0, E = Patterns.size(); PIdx < E; ++PIdx) {
    O << "static void printPattern_" << PIdx << "(\n"
      << "    " << FullClassName << " *P, const MCInst *MI,\n"
      << "    uint64_t Address,";
    if (PassSubtarget)
      O << " const MCSubtargetInfo &STI,";
    O << " raw_ostream &O) {\n";
    for (const std::string &Line : Patterns[PIdx])
      O << "  " << Line << "\n";
    O << "}\n\n";
  }

  // Function-pointer table indexed by opcode, covering the full opcode range.
  // Non-overflow opcodes → printPattern_None.
  // Overflow opcodes     → their deduplicated printPattern_N.
  O << "static void (*const " << PtrTableName << "[])(\n"
    << ParamList << ") = {\n";
  for (unsigned i = 0, E = NumberedInstructions.size(); i < E; ++i) {
    unsigned PIdx = OpcodeToPattern[i];
    if (PIdx == ~0U)
      O << "  &printPattern_None";
    else
      O << "  &printPattern_" << PIdx;
    if (i + 1 < E)
      O << ",";
    O << "\t// " << NumberedInstructions[i]->getName() << "\n";
  }
  O << "};\n\n";

  LLVM_DEBUG(dbgs() << "[AsmWriter] " << TargetName << ": "
                    << OpcodeInsts.size() << " overflow instructions -> "
                    << Patterns.size() << " unique patterns\n");
}

void AsmWriterEmitter::EmitOpcodePatternDispatch(raw_ostream &O,
                                                 StringRef TargetName,
                                                 bool PassSubtarget) {
  // Single unconditional indirect call; no switch, no branch.
  // The compiler cannot inline through a const function pointer, so
  // printInstruction() itself stays small regardless of instruction-set size.
  std::string PtrTableName = (TargetName + "InstPrinters").str();
  O << "  " << PtrTableName << "[MI->getOpcode()](this, MI, Address, ";
  if (PassSubtarget)
    O << "STI, ";
  O << "O);\n";
}

static void
emitRegisterNameString(raw_ostream &O, StringRef AltName,
                       const std::deque<CodeGenRegister> &Registers) {
  SequenceToOffsetTable<std::string> StringTable;
  SmallVector<std::string, 4> AsmNames(Registers.size());
  unsigned i = 0;
  for (const auto &Reg : Registers) {
    std::string &AsmName = AsmNames[i++];

    // "NoRegAltName" is special. We don't need to do a lookup for that,
    // as it's just a reference to the default register name.
    if (AltName == "" || AltName == "NoRegAltName") {
      AsmName = Reg.TheDef->getValueAsString("AsmName").str();
      if (AsmName.empty())
        AsmName = Reg.getName().str();
    } else {
      // Make sure the register has an alternate name for this index.
      std::vector<const Record *> AltNameList =
          Reg.TheDef->getValueAsListOfDefs("RegAltNameIndices");
      unsigned Idx = 0, e;
      for (e = AltNameList.size();
           Idx < e && (AltNameList[Idx]->getName() != AltName); ++Idx)
        ;
      // If the register has an alternate name for this index, use it.
      // Otherwise, leave it empty as an error flag.
      if (Idx < e) {
        std::vector<StringRef> AltNames =
            Reg.TheDef->getValueAsListOfStrings("AltNames");
        if (AltNames.size() <= Idx)
          PrintFatalError(Reg.TheDef->getLoc(),
                          "Register definition missing alt name for '" +
                              AltName + "'.");
        AsmName = AltNames[Idx].str();
      }
    }
    StringTable.add(AsmName);
  }

  StringTable.layout();
  StringTable.emitStringLiteralDef(O, Twine("  static const char AsmStrs") +
                                          AltName + "[]");

  O << "  static const " << getMinimalTypeForRange(StringTable.size() - 1, 32)
    << " RegAsmOffset" << AltName << "[] = {";
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    if ((i % 14) == 0)
      O << "\n    ";
    O << StringTable.get(AsmNames[i]) << ", ";
  }
  O << "\n  };\n"
    << "\n";
}

void AsmWriterEmitter::EmitGetRegisterName(raw_ostream &O) {
  const Record *AsmWriter = Target.getAsmWriter();
  StringRef ClassName = AsmWriter->getValueAsString("AsmWriterClassName");
  const auto &Registers = Target.getRegBank().getRegisters();
  ArrayRef<const Record *> AltNameIndices = Target.getRegAltNameIndices();
  bool hasAltNames = AltNameIndices.size() > 1;
  StringRef Namespace = Registers.front().TheDef->getValueAsString("Namespace");

  O << "\n\n/// getRegisterName - This method is automatically generated by "
       "tblgen\n"
       "/// from the register set description.  This returns the assembler "
       "name\n"
       "/// for the specified register.\n"
       "const char *"
    << Target.getName() << ClassName << "::";
  if (hasAltNames)
    O << "\ngetRegisterName(MCRegister Reg, unsigned AltIdx) {\n";
  else
    O << "getRegisterName(MCRegister Reg) {\n";
  O << "  unsigned RegNo = Reg.id();\n"
    << "  assert(RegNo && RegNo < " << (Registers.size() + 1)
    << " && \"Invalid register number!\");\n"
    << "\n";

  if (hasAltNames) {
    for (const Record *R : AltNameIndices)
      emitRegisterNameString(O, R->getName(), Registers);
  } else {
    emitRegisterNameString(O, "", Registers);
  }

  if (hasAltNames) {
    O << "  switch(AltIdx) {\n"
      << "  default: llvm_unreachable(\"Invalid register alt name index!\");\n";
    for (const Record *R : AltNameIndices) {
      StringRef AltName = R->getName();
      O << "  case ";
      if (!Namespace.empty())
        O << Namespace << "::";
      O << AltName << ":\n";
      if (R->isValueUnset("FallbackRegAltNameIndex"))
        O << "    assert(*(AsmStrs" << AltName << "+RegAsmOffset" << AltName
          << "[RegNo-1]) &&\n"
          << "           \"Invalid alt name index for register!\");\n";
      else {
        O << "    if (!*(AsmStrs" << AltName << "+RegAsmOffset" << AltName
          << "[RegNo-1]))\n"
          << "      return getRegisterName(RegNo, ";
        if (!Namespace.empty())
          O << Namespace << "::";
        O << R->getValueAsDef("FallbackRegAltNameIndex")->getName() << ");\n";
      }
      O << "    return AsmStrs" << AltName << "+RegAsmOffset" << AltName
        << "[RegNo-1];\n";
    }
    O << "  }\n";
  } else {
    O << "  assert (*(AsmStrs+RegAsmOffset[RegNo-1]) &&\n"
      << "          \"Invalid alt name index for register!\");\n"
      << "  return AsmStrs+RegAsmOffset[RegNo-1];\n";
  }
  O << "}\n";
}

namespace {

// IAPrinter - Holds information about an InstAlias. Two InstAliases match if
// they both have the same conditionals. In which case, we cannot print out the
// alias for that pattern.
class IAPrinter {
  std::map<StringRef, std::pair<int, int>> OpMap;

  std::vector<std::string> Conds;

  std::string Result;
  std::string AsmString;

  unsigned NumMIOps;

public:
  IAPrinter(std::string R, std::string AS, unsigned NumMIOps)
      : Result(std::move(R)), AsmString(std::move(AS)), NumMIOps(NumMIOps) {}

  void addCond(std::string C) { Conds.push_back(std::move(C)); }
  ArrayRef<std::string> getConds() const { return Conds; }
  size_t getCondCount() const { return Conds.size(); }

  void addOperand(StringRef Op, int OpIdx, int PrintMethodIdx = -1) {
    assert(OpIdx >= 0 && OpIdx < 0xFE && "Idx out of range");
    assert(PrintMethodIdx >= -1 && PrintMethodIdx < 0xFF && "Idx out of range");
    OpMap[Op] = {OpIdx, PrintMethodIdx};
  }

  unsigned getNumMIOps() { return NumMIOps; }

  StringRef getResult() { return Result; }

  bool isOpMapped(StringRef Op) { return OpMap.find(Op) != OpMap.end(); }
  int getOpIndex(StringRef Op) { return OpMap[Op].first; }
  std::pair<int, int> &getOpData(StringRef Op) { return OpMap[Op]; }

  std::pair<StringRef, StringRef::iterator> parseName(StringRef::iterator Start,
                                                      StringRef::iterator End) {
    StringRef::iterator I = Start;
    StringRef::iterator Next;
    if (*I == '{') {
      // ${some_name}
      Start = ++I;
      while (I != End && *I != '}')
        ++I;
      Next = I;
      // eat the final '}'
      if (Next != End)
        ++Next;
    } else {
      // $name, just eat the usual suspects.
      while (I != End && (isAlnum(*I) || *I == '_'))
        ++I;
      Next = I;
    }

    return {StringRef(Start, I - Start), Next};
  }

  std::string formatAliasString(uint32_t &UnescapedSize) {
    // Directly mangle mapped operands into the string. Each operand is
    // identified by a '$' sign followed by a byte identifying the number of the
    // operand. We add one to the index to avoid zero bytes.
    StringRef ASM(AsmString);
    std::string OutString;
    raw_string_ostream OS(OutString);
    for (StringRef::iterator I = ASM.begin(), E = ASM.end(); I != E;) {
      OS << *I;
      ++UnescapedSize;
      if (*I == '$') {
        StringRef Name;
        std::tie(Name, I) = parseName(++I, E);
        assert(isOpMapped(Name) && "Unmapped operand!");

        int OpIndex, PrintIndex;
        std::tie(OpIndex, PrintIndex) = getOpData(Name);
        if (PrintIndex == -1) {
          // Can use the default printOperand route.
          OS << format("\\x%02X", (unsigned char)OpIndex + 1);
          ++UnescapedSize;
        } else {
          // 3 bytes if a PrintMethod is needed: 0xFF, the MCInst operand
          // number, and which of our pre-detected Methods to call.
          OS << format("\\xFF\\x%02X\\x%02X", OpIndex + 1, PrintIndex + 1);
          UnescapedSize += 3;
        }
      } else {
        ++I;
      }
    }
    return OutString;
  }

  bool operator==(const IAPrinter &RHS) const {
    if (NumMIOps != RHS.NumMIOps)
      return false;
    if (Conds.size() != RHS.Conds.size())
      return false;

    unsigned Idx = 0;
    for (const auto &str : Conds)
      if (str != RHS.Conds[Idx++])
        return false;

    return true;
  }
};

} // end anonymous namespace

static unsigned CountNumOperands(StringRef AsmString, unsigned Variant) {
  return AsmString.count(' ') + AsmString.count('\t');
}

namespace {

struct AliasPriorityComparator {
  using ValueType = std::pair<CodeGenInstAlias, int>;
  bool operator()(const ValueType &LHS, const ValueType &RHS) const {
    if (LHS.second == RHS.second) {
      // We don't actually care about the order, but for consistency it
      // shouldn't depend on pointer comparisons.
      return LessRecordByID()(LHS.first.TheDef, RHS.first.TheDef);
    }

    // Aliases with larger priorities should be considered first.
    return LHS.second > RHS.second;
  }
};

} // end anonymous namespace

void AsmWriterEmitter::EmitPrintAliasInstruction(raw_ostream &O) {
  const Record *AsmWriter = Target.getAsmWriter();

  O << "\n#ifdef PRINT_ALIAS_INSTR\n";
  O << "#undef PRINT_ALIAS_INSTR\n\n";

  //////////////////////////////
  // Gather information about aliases we need to print
  //////////////////////////////

  // Emit the method that prints the alias instruction.
  StringRef ClassName = AsmWriter->getValueAsString("AsmWriterClassName");
  unsigned Variant = AsmWriter->getValueAsInt("Variant");
  bool PassSubtarget = AsmWriter->getValueAsInt("PassSubtarget");

  // Create a map from the qualified name to a list of potential matches.
  using AliasWithPriority =
      std::set<std::pair<CodeGenInstAlias, int>, AliasPriorityComparator>;
  std::map<std::string, AliasWithPriority> AliasMap;
  for (const Record *R : Records.getAllDerivedDefinitions("InstAlias")) {
    int Priority = R->getValueAsInt("EmitPriority");
    if (Priority < 1)
      continue; // Aliases with priority 0 are never emitted.

    const DagInit *DI = R->getValueAsDag("ResultInst");
    AliasMap[getQualifiedName(DI->getOperatorAsDef(R->getLoc()))].emplace(
        CodeGenInstAlias(R, Target), Priority);
  }

  // A map of which conditions need to be met for each instruction operand
  // before it can be matched to the mnemonic.
  std::map<std::string, std::vector<IAPrinter>> IAPrinterMap;

  std::vector<std::pair<std::string, bool>> PrintMethods;

  // A list of MCOperandPredicates for all operands in use, and the reverse map
  std::vector<const Record *> MCOpPredicates;
  DenseMap<const Record *, unsigned> MCOpPredicateMap;

  for (auto &Aliases : AliasMap) {
    for (auto &Alias : Aliases.second) {
      const CodeGenInstAlias &CGA = Alias.first;
      unsigned LastOpNo = CGA.ResultInstOperandIndex.size();
      std::string FlatInstAsmString =
          CodeGenInstruction::FlattenAsmStringVariants(
              CGA.ResultInst->AsmString, Variant);
      unsigned NumResultOps = CountNumOperands(FlatInstAsmString, Variant);

      std::string FlatAliasAsmString =
          CodeGenInstruction::FlattenAsmStringVariants(CGA.AsmString, Variant);
      UnescapeAliasString(FlatAliasAsmString);

      // Don't emit the alias if it has more operands than what it's aliasing.
      if (NumResultOps < CountNumOperands(FlatAliasAsmString, Variant))
        continue;

      StringRef Namespace = Target.getName();
      unsigned NumMIOps = 0;
      for (auto &ResultInstOpnd : CGA.ResultInst->Operands)
        NumMIOps += ResultInstOpnd.MINumOperands;

      IAPrinter IAP(CGA.Result->getAsString(), FlatAliasAsmString, NumMIOps);

      unsigned MIOpNum = 0;
      for (unsigned i = 0, e = LastOpNo; i != e; ++i) {
        // Skip over tied operands as they're not part of an alias declaration.
        auto &Operands = CGA.ResultInst->Operands;
        while (true) {
          unsigned OpNum = Operands.getSubOperandNumber(MIOpNum).first;
          if (Operands[OpNum].MINumOperands == 1 &&
              Operands[OpNum].getTiedRegister() != -1) {
            // Tied operands of different RegisterClass should be explicit
            // within an instruction's syntax and so cannot be skipped.
            int TiedOpNum = Operands[OpNum].getTiedRegister();
            if (Operands[OpNum].Rec->getName() ==
                Operands[TiedOpNum].Rec->getName()) {
              ++MIOpNum;
              continue;
            }
          }
          break;
        }

        // Ignore unchecked result operands.
        while (IAP.getCondCount() < MIOpNum)
          IAP.addCond("AliasPatternCond::K_Ignore, 0");

        const CodeGenInstAlias::ResultOperand &RO = CGA.ResultOperands[i];

        switch (RO.Kind) {
        case CodeGenInstAlias::ResultOperand::K_Record: {
          const Record *Rec = RO.getRecord();
          StringRef ROName = RO.getName();
          int PrintMethodIdx = -1;

          // These two may have a PrintMethod, which we want to record (if it's
          // the first time we've seen it) and provide an index for the aliasing
          // code to use.
          if (Rec->isSubClassOf("RegisterOperand") ||
              Rec->isSubClassOf("Operand")) {
            StringRef PrintMethod = Rec->getValueAsString("PrintMethod");
            bool IsPCRel =
                Rec->getValueAsString("OperandType") == "OPERAND_PCREL";
            if (PrintMethod != "" && PrintMethod != "printOperand") {
              PrintMethodIdx = llvm::find_if(PrintMethods,
                                             [&](auto &X) {
                                               return X.first == PrintMethod;
                                             }) -
                               PrintMethods.begin();
              if (static_cast<unsigned>(PrintMethodIdx) == PrintMethods.size())
                PrintMethods.emplace_back(PrintMethod.str(), IsPCRel);
            }
          }

          if (Target.getAsRegClassLike(Rec)) {
            if (!IAP.isOpMapped(ROName)) {
              IAP.addOperand(ROName, MIOpNum, PrintMethodIdx);
              const Record *R =
                  Target.getAsRegClassLike(CGA.ResultOperands[i].getRecord());
              assert(R && "Not a valid register class?");
              if (R->isSubClassOf("RegClassByHwMode")) {
                IAP.addCond(std::string(
                    formatv("AliasPatternCond::K_RegClassByHwMode, {}::{}",
                            Namespace, R->getName())));
              } else {
                IAP.addCond(std::string(
                    formatv("AliasPatternCond::K_RegClass, {}::{}RegClassID",
                            Namespace, R->getName())));
              }
            } else {
              IAP.addCond(std::string(formatv("AliasPatternCond::K_TiedReg, {}",
                                              IAP.getOpIndex(ROName))));
            }
          } else {
            // Assume all printable operands are desired for now. This can be
            // overridden in the InstAlias instantiation if necessary.
            IAP.addOperand(ROName, MIOpNum, PrintMethodIdx);

            // There might be an additional predicate on the MCOperand
            unsigned &Entry = MCOpPredicateMap[Rec];
            if (!Entry) {
              if (!Rec->isValueUnset("MCOperandPredicate")) {
                MCOpPredicates.push_back(Rec);
                Entry = MCOpPredicates.size();
              } else {
                break; // No conditions on this operand at all
              }
            }
            IAP.addCond(
                std::string(formatv("AliasPatternCond::K_Custom, {}", Entry)));
          }
          break;
        }
        case CodeGenInstAlias::ResultOperand::K_Imm: {
          // Just because the alias has an immediate result, doesn't mean the
          // MCInst will. An MCExpr could be present, for example.
          auto Imm = CGA.ResultOperands[i].getImm();
          int32_t Imm32 = int32_t(Imm);
          if (Imm != Imm32)
            PrintFatalError("Matching an alias with an immediate out of the "
                            "range of int32_t is not supported");
          IAP.addCond(std::string(
              formatv("AliasPatternCond::K_Imm, uint32_t({})", Imm32)));
          break;
        }
        case CodeGenInstAlias::ResultOperand::K_Reg:
          if (!CGA.ResultOperands[i].getRegister()) {
            IAP.addCond(std::string(
                formatv("AliasPatternCond::K_Reg, {}::NoRegister", Namespace)));
            break;
          }

          const Record *Rec = CGA.ResultOperands[i].getRegister();
          StringRef Reg = Rec->getName();
          if (Rec->isSubClassOf("RegisterByHwMode")) {
            // Use a custom predicate to handle RegisterByHwMode since there
            // is no way to handle this in the generic code.
            unsigned &Entry = MCOpPredicateMap[Rec];
            if (!Entry) {
              MCOpPredicates.push_back(Rec);
              Entry = MCOpPredicates.size();
            }
            IAP.addCond(std::string(
                formatv("AliasPatternCond::K_Custom, {}/*{}*/", Entry, Reg)));
          } else {
            IAP.addCond(std::string(
                formatv("AliasPatternCond::K_Reg, {}::{}", Namespace, Reg)));
          }
          break;
        }

        MIOpNum += RO.getMINumOperands();
      }

      std::vector<const Record *> ReqFeatures;
      if (PassSubtarget) {
        // We only consider ReqFeatures predicates if PassSubtarget
        std::vector<const Record *> RF =
            CGA.TheDef->getValueAsListOfDefs("Predicates");
        copy_if(RF, std::back_inserter(ReqFeatures), [](const Record *R) {
          return R->getValueAsBit("AssemblerMatcherPredicate");
        });
      }

      for (const Record *R : ReqFeatures) {
        const DagInit *D = R->getValueAsDag("AssemblerCondDag");
        auto *Op = dyn_cast<DefInit>(D->getOperator());
        if (!Op)
          PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
        StringRef CombineType = Op->getDef()->getName();
        if (CombineType != "any_of" && CombineType != "all_of")
          PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
        if (D->getNumArgs() == 0)
          PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
        bool IsOr = CombineType == "any_of";
        // Change (any_of FeatureAll, (any_of ...)) to (any_of FeatureAll, ...).
        if (IsOr && D->getNumArgs() == 2 && isa<DagInit>(D->getArg(1))) {
          const DagInit *RHS = cast<DagInit>(D->getArg(1));
          SmallVector<std::pair<const Init *, const StringInit *>> Args{
              *D->getArgAndNames().begin()};
          llvm::append_range(Args, RHS->getArgAndNames());
          D = DagInit::get(D->getOperator(), Args);
        }

        for (auto *Arg : D->getArgs()) {
          bool IsNeg = false;
          if (auto *NotArg = dyn_cast<DagInit>(Arg)) {
            if (NotArg->getOperator()->getAsString() != "not" ||
                NotArg->getNumArgs() != 1)
              PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");
            Arg = NotArg->getArg(0);
            IsNeg = true;
          }
          if (!isa<DefInit>(Arg) ||
              !cast<DefInit>(Arg)->getDef()->isSubClassOf("SubtargetFeature"))
            PrintFatalError(R->getLoc(), "Invalid AssemblerCondDag!");

          IAP.addCond(std::string(formatv(
              "AliasPatternCond::K_{}{}Feature, {}::{}", IsOr ? "Or" : "",
              IsNeg ? "Neg" : "", Namespace, Arg->getAsString())));
        }
        // If an AssemblerPredicate with ors is used, note end of list should
        // these be combined.
        if (IsOr)
          IAP.addCond("AliasPatternCond::K_EndOrFeatures, 0");
      }

      IAPrinterMap[Aliases.first].push_back(std::move(IAP));
    }
  }

  //////////////////////////////
  // Write out the printAliasInstr function
  //////////////////////////////

  std::string Header;
  raw_string_ostream HeaderO(Header);

  HeaderO << "bool " << Target.getName() << ClassName
          << "::printAliasInstr(const MCInst"
          << " *MI, uint64_t Address, "
          << (PassSubtarget ? "const MCSubtargetInfo &STI, " : "")
          << "raw_ostream &OS) {\n";

  std::string PatternsForOpcode;
  raw_string_ostream OpcodeO(PatternsForOpcode);

  unsigned PatternCount = 0;
  std::string Patterns;
  raw_string_ostream PatternO(Patterns);

  unsigned CondCount = 0;
  std::string Conds;
  raw_string_ostream CondO(Conds);

  // All flattened alias strings.
  std::map<std::string, uint32_t> AsmStringOffsets;
  std::vector<std::pair<uint32_t, std::string>> AsmStrings;
  size_t AsmStringsSize = 0;

  // Iterate over the opcodes in enum order so they are sorted by opcode for
  // binary search.
  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    auto It = IAPrinterMap.find(getQualifiedName(Inst->TheDef));
    if (It == IAPrinterMap.end())
      continue;
    std::vector<IAPrinter> &IAPs = It->second;
    std::vector<IAPrinter *> UniqueIAPs;

    // Remove any ambiguous alias rules.
    for (auto &LHS : IAPs) {
      bool IsDup = false;
      for (const auto &RHS : IAPs) {
        if (&LHS != &RHS && LHS == RHS) {
          IsDup = true;
          break;
        }
      }

      if (!IsDup)
        UniqueIAPs.push_back(&LHS);
    }

    if (UniqueIAPs.empty())
      continue;

    unsigned PatternStart = PatternCount;

    // Insert the pattern start and opcode in the pattern list for debugging.
    PatternO << formatv("    // {} - {}\n", It->first, PatternStart);

    for (IAPrinter *IAP : UniqueIAPs) {
      // Start each condition list with a comment of the resulting pattern that
      // we're trying to match.
      unsigned CondStart = CondCount;
      CondO << formatv("    // {} - {}\n", IAP->getResult(), CondStart);
      for (const auto &Cond : IAP->getConds())
        CondO << "    {" << Cond << "},\n";
      CondCount += IAP->getCondCount();

      // After operands have been examined, re-encode the alias string with
      // escapes indicating how operands should be printed.
      uint32_t UnescapedSize = 0;
      std::string EncodedAsmString = IAP->formatAliasString(UnescapedSize);
      auto Insertion =
          AsmStringOffsets.try_emplace(EncodedAsmString, AsmStringsSize);
      if (Insertion.second) {
        // If the string is new, add it to the vector.
        AsmStrings.emplace_back(AsmStringsSize, EncodedAsmString);
        AsmStringsSize += UnescapedSize + 1;
      }
      unsigned AsmStrOffset = Insertion.first->second;

      PatternO << formatv("    {{{}, {}, {}, {} },\n", AsmStrOffset, CondStart,
                          IAP->getNumMIOps(), IAP->getCondCount());
      ++PatternCount;
    }

    OpcodeO << formatv("    {{{}, {}, {} },\n", It->first, PatternStart,
                       PatternCount - PatternStart);
  }

  if (PatternsForOpcode.empty()) {
    O << Header;
    O << "  return false;\n";
    O << "}\n\n";
    O << "#endif // PRINT_ALIAS_INSTR\n";
    return;
  }

  // Forward declare the validation method if needed.
  if (!MCOpPredicates.empty())
    O << "static bool " << Target.getName() << ClassName
      << "ValidateMCOperand(const MCOperand &MCOp,\n"
      << "                  const MCSubtargetInfo &STI,\n"
      << "                  unsigned PredicateIndex);\n";

  O << Header;
  O.indent(2) << "static const PatternsForOpcode OpToPatterns[] = {\n";
  O << PatternsForOpcode;
  O.indent(2) << "};\n\n";
  O.indent(2) << "static const AliasPattern Patterns[] = {\n";
  O << Patterns;
  O.indent(2) << "};\n\n";
  O.indent(2) << "static const AliasPatternCond Conds[] = {\n";
  O << Conds;
  O.indent(2) << "};\n\n";
  O.indent(2) << "static const char AsmStrings[] =\n";
  for (const auto &P : AsmStrings) {
    O.indent(4) << "/* " << P.first << " */ \"" << P.second << "\\0\"\n";
  }

  O.indent(2) << ";\n\n";

  // Assert that the opcode table is sorted. Use a static local constructor to
  // ensure that the check only happens once on first run.
  O << "#ifndef NDEBUG\n";
  O.indent(2) << "static struct SortCheck {\n";
  O.indent(2) << "  SortCheck(ArrayRef<PatternsForOpcode> OpToPatterns) {\n";
  O.indent(2) << "    assert(std::is_sorted(\n";
  O.indent(2) << "               OpToPatterns.begin(), OpToPatterns.end(),\n";
  O.indent(2) << "               [](const PatternsForOpcode &L, const "
                 "PatternsForOpcode &R) {\n";
  O.indent(2) << "                 return L.Opcode < R.Opcode;\n";
  O.indent(2) << "               }) &&\n";
  O.indent(2) << "           \"tablegen failed to sort opcode patterns\");\n";
  O.indent(2) << "  }\n";
  O.indent(2) << "} sortCheckVar(OpToPatterns);\n";
  O << "#endif\n\n";

  O.indent(2) << "AliasMatchingData M {\n";
  O.indent(2) << "  ArrayRef(OpToPatterns),\n";
  O.indent(2) << "  ArrayRef(Patterns),\n";
  O.indent(2) << "  ArrayRef(Conds),\n";
  O.indent(2) << "  StringRef(AsmStrings, std::size(AsmStrings)),\n";
  if (MCOpPredicates.empty())
    O.indent(2) << "  nullptr,\n";
  else
    O.indent(2) << "  &" << Target.getName() << ClassName
                << "ValidateMCOperand,\n";
  O.indent(2) << "};\n";

  O.indent(2) << "const char *AsmString = matchAliasPatterns(MI, "
              << (PassSubtarget ? "&STI" : "nullptr") << ", M);\n";
  O.indent(2) << "if (!AsmString) return false;\n\n";

  // Code that prints the alias, replacing the operands with the ones from the
  // MCInst.
  O << "  unsigned I = 0;\n";
  O << "  while (AsmString[I] != ' ' && AsmString[I] != '\\t' &&\n";
  O << "         AsmString[I] != '$' && AsmString[I] != '\\0')\n";
  O << "    ++I;\n";
  O << "  OS << '\\t' << StringRef(AsmString, I);\n";

  O << "  if (AsmString[I] != '\\0') {\n";
  O << "    if (AsmString[I] == ' ' || AsmString[I] == '\\t') {\n";
  O << "      OS << '\\t';\n";
  O << "      ++I;\n";
  O << "    }\n";
  O << "    do {\n";
  O << "      if (AsmString[I] == '$') {\n";
  O << "        ++I;\n";
  O << "        if (AsmString[I] == (char)0xff) {\n";
  O << "          ++I;\n";
  O << "          int OpIdx = AsmString[I++] - 1;\n";
  O << "          int PrintMethodIdx = AsmString[I++] - 1;\n";
  O << "          printCustomAliasOperand(MI, Address, OpIdx, PrintMethodIdx, ";
  O << (PassSubtarget ? "STI, " : "");
  O << "OS);\n";
  O << "        } else\n";
  O << "          printOperand(MI, unsigned(AsmString[I++]) - 1, ";
  O << (PassSubtarget ? "STI, " : "");
  O << "OS);\n";
  O << "      } else {\n";
  O << "        OS << AsmString[I++];\n";
  O << "      }\n";
  O << "    } while (AsmString[I] != '\\0');\n";
  O << "  }\n\n";

  O << "  return true;\n";
  O << "}\n\n";

  //////////////////////////////
  // Write out the printCustomAliasOperand function
  //////////////////////////////

  O << "void " << Target.getName() << ClassName << "::"
    << "printCustomAliasOperand(\n"
    << "         const MCInst *MI, uint64_t Address, unsigned OpIdx,\n"
    << "         unsigned PrintMethodIdx,\n"
    << (PassSubtarget ? "         const MCSubtargetInfo &STI,\n" : "")
    << "         raw_ostream &OS) {\n";
  if (PrintMethods.empty())
    O << "  llvm_unreachable(\"Unknown PrintMethod kind\");\n";
  else {
    O << "  switch (PrintMethodIdx) {\n"
      << "  default:\n"
      << "    llvm_unreachable(\"Unknown PrintMethod kind\");\n"
      << "    break;\n";

    for (unsigned i = 0; i < PrintMethods.size(); ++i) {
      O << "  case " << i << ":\n"
        << "    " << PrintMethods[i].first << "(MI, "
        << (PrintMethods[i].second ? "Address, " : "") << "OpIdx, "
        << (PassSubtarget ? "STI, " : "") << "OS);\n"
        << "    break;\n";
    }
    O << "  }\n";
  }
  O << "}\n\n";

  if (!MCOpPredicates.empty()) {
    O << "static bool " << Target.getName() << ClassName
      << "ValidateMCOperand(const MCOperand &MCOp,\n"
      << "                  const MCSubtargetInfo &STI,\n"
      << "                  unsigned PredicateIndex) {\n"
      << "  switch (PredicateIndex) {\n"
      << "  default:\n"
      << "    llvm_unreachable(\"Unknown MCOperandPredicate kind\");\n"
      << "    break;\n";

    for (auto [I, Rec] : enumerate(MCOpPredicates)) {
      O << "  case " << I + 1 << ": {\n";
      // We have to handle RegClassByHwMode predicates here since there is no
      // special case opcode for them.
      if (Rec->isSubClassOf("RegisterByHwMode")) {
        if (!PassSubtarget)
          PrintFatalError(Target.getAsmWriter()->getLoc(),
                          "PassSubtarget must be set in "
                          "AsmWriter to handle RegisterByHwMode");
        O << "    return MCOp.isReg() && MCOp.getReg() == ";
        RegisterByHwMode(Rec, Target.getRegBank())
            .emitResolverCall(O,
                              "STI.getHwMode(MCSubtargetInfo::HwMode_RegInfo)");
        O << ";\n";
      } else {
        // Normal MCOperandPredicate code snippet, emit verbatim.
        O << Rec->getValueAsString("MCOperandPredicate") << "\n";
      }
      O << "  }\n";
    }
    O << "  }\n"
      << "}\n\n";
  }

  O << "#endif // PRINT_ALIAS_INSTR\n";
}

AsmWriterEmitter::AsmWriterEmitter(const RecordKeeper &R)
    : Records(R), Target(R) {
  const Record *AsmWriter = Target.getAsmWriter();
  unsigned Variant = AsmWriter->getValueAsInt("Variant");

  // Get the instruction numbering.
  NumberedInstructions = Target.getInstructions();

  for (const auto &[Idx, I] : enumerate(NumberedInstructions)) {
    if (!I->AsmString.empty() && I->getName() != "PHI")
      Instructions.emplace_back(*I, Idx, Variant);
  }
}

void AsmWriterEmitter::run(raw_ostream &O) {
  std::vector<std::vector<std::string>> TableDrivenOperandPrinters;
  unsigned BitsLeft = 0;
  unsigned AsmStrBits = 0;
  emitSourceFileHeader("Assembly Writer Source Fragment", O, Records);
  EmitGetMnemonic(O, TableDrivenOperandPrinters, BitsLeft, AsmStrBits);
  EmitPrintInstruction(O, TableDrivenOperandPrinters, BitsLeft, AsmStrBits);
  EmitGetRegisterName(O);
  EmitPrintAliasInstruction(O);
}

static TableGen::Emitter::OptClass<AsmWriterEmitter>
    X("gen-asm-writer", "Generate assembly writer");
