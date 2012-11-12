//===- AsmWriterEmitter.cpp - Generate an assembly writer -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is emits an assembly printer for the current target.
// Note that this is currently fairly skeletal, but will grow over time.
//
//===----------------------------------------------------------------------===//

#include "AsmWriterInst.h"
#include "CodeGenTarget.h"
#include "SequenceToOffsetTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <vector>
using namespace llvm;

namespace {
class AsmWriterEmitter {
  RecordKeeper &Records;
  std::map<const CodeGenInstruction*, AsmWriterInst*> CGIAWIMap;
  std::vector<const CodeGenInstruction*> NumberedInstructions;
public:
  AsmWriterEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &o);

private:
  void EmitPrintInstruction(raw_ostream &o);
  void EmitGetRegisterName(raw_ostream &o);
  void EmitPrintAliasInstruction(raw_ostream &O);

  AsmWriterInst *getAsmWriterInstByID(unsigned ID) const {
    assert(ID < NumberedInstructions.size());
    std::map<const CodeGenInstruction*, AsmWriterInst*>::const_iterator I =
      CGIAWIMap.find(NumberedInstructions[ID]);
    assert(I != CGIAWIMap.end() && "Didn't find inst!");
    return I->second;
  }
  void FindUniqueOperandCommands(std::vector<std::string> &UOC,
                                 std::vector<unsigned> &InstIdxs,
                                 std::vector<unsigned> &InstOpsUsed) const;
};
} // end anonymous namespace

static void PrintCases(std::vector<std::pair<std::string,
                       AsmWriterOperand> > &OpsToPrint, raw_ostream &O) {
  O << "    case " << OpsToPrint.back().first << ": ";
  AsmWriterOperand TheOp = OpsToPrint.back().second;
  OpsToPrint.pop_back();

  // Check to see if any other operands are identical in this list, and if so,
  // emit a case label for them.
  for (unsigned i = OpsToPrint.size(); i != 0; --i)
    if (OpsToPrint[i-1].second == TheOp) {
      O << "\n    case " << OpsToPrint[i-1].first << ": ";
      OpsToPrint.erase(OpsToPrint.begin()+i-1);
    }

  // Finally, emit the code.
  O << TheOp.getCode();
  O << "break;\n";
}


/// EmitInstructions - Emit the last instruction in the vector and any other
/// instructions that are suitably similar to it.
static void EmitInstructions(std::vector<AsmWriterInst> &Insts,
                             raw_ostream &O) {
  AsmWriterInst FirstInst = Insts.back();
  Insts.pop_back();

  std::vector<AsmWriterInst> SimilarInsts;
  unsigned DifferingOperand = ~0;
  for (unsigned i = Insts.size(); i != 0; --i) {
    unsigned DiffOp = Insts[i-1].MatchesAllButOneOp(FirstInst);
    if (DiffOp != ~1U) {
      if (DifferingOperand == ~0U)  // First match!
        DifferingOperand = DiffOp;

      // If this differs in the same operand as the rest of the instructions in
      // this class, move it to the SimilarInsts list.
      if (DifferingOperand == DiffOp || DiffOp == ~0U) {
        SimilarInsts.push_back(Insts[i-1]);
        Insts.erase(Insts.begin()+i-1);
      }
    }
  }

  O << "  case " << FirstInst.CGI->Namespace << "::"
    << FirstInst.CGI->TheDef->getName() << ":\n";
  for (unsigned i = 0, e = SimilarInsts.size(); i != e; ++i)
    O << "  case " << SimilarInsts[i].CGI->Namespace << "::"
      << SimilarInsts[i].CGI->TheDef->getName() << ":\n";
  for (unsigned i = 0, e = FirstInst.Operands.size(); i != e; ++i) {
    if (i != DifferingOperand) {
      // If the operand is the same for all instructions, just print it.
      O << "    " << FirstInst.Operands[i].getCode();
    } else {
      // If this is the operand that varies between all of the instructions,
      // emit a switch for just this operand now.
      O << "    switch (MI->getOpcode()) {\n";
      std::vector<std::pair<std::string, AsmWriterOperand> > OpsToPrint;
      OpsToPrint.push_back(std::make_pair(FirstInst.CGI->Namespace + "::" +
                                          FirstInst.CGI->TheDef->getName(),
                                          FirstInst.Operands[i]));

      for (unsigned si = 0, e = SimilarInsts.size(); si != e; ++si) {
        AsmWriterInst &AWI = SimilarInsts[si];
        OpsToPrint.push_back(std::make_pair(AWI.CGI->Namespace+"::"+
                                            AWI.CGI->TheDef->getName(),
                                            AWI.Operands[i]));
      }
      std::reverse(OpsToPrint.begin(), OpsToPrint.end());
      while (!OpsToPrint.empty())
        PrintCases(OpsToPrint, O);
      O << "    }";
    }
    O << "\n";
  }
  O << "    break;\n";
}

void AsmWriterEmitter::
FindUniqueOperandCommands(std::vector<std::string> &UniqueOperandCommands,
                          std::vector<unsigned> &InstIdxs,
                          std::vector<unsigned> &InstOpsUsed) const {
  InstIdxs.assign(NumberedInstructions.size(), ~0U);

  // This vector parallels UniqueOperandCommands, keeping track of which
  // instructions each case are used for.  It is a comma separated string of
  // enums.
  std::vector<std::string> InstrsForCase;
  InstrsForCase.resize(UniqueOperandCommands.size());
  InstOpsUsed.assign(UniqueOperandCommands.size(), 0);

  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    const AsmWriterInst *Inst = getAsmWriterInstByID(i);
    if (Inst == 0) continue;  // PHI, INLINEASM, PROLOG_LABEL, etc.

    std::string Command;
    if (Inst->Operands.empty())
      continue;   // Instruction already done.

    Command = "    " + Inst->Operands[0].getCode() + "\n";

    // Check to see if we already have 'Command' in UniqueOperandCommands.
    // If not, add it.
    bool FoundIt = false;
    for (unsigned idx = 0, e = UniqueOperandCommands.size(); idx != e; ++idx)
      if (UniqueOperandCommands[idx] == Command) {
        InstIdxs[i] = idx;
        InstrsForCase[idx] += ", ";
        InstrsForCase[idx] += Inst->CGI->TheDef->getName();
        FoundIt = true;
        break;
      }
    if (!FoundIt) {
      InstIdxs[i] = UniqueOperandCommands.size();
      UniqueOperandCommands.push_back(Command);
      InstrsForCase.push_back(Inst->CGI->TheDef->getName());

      // This command matches one operand so far.
      InstOpsUsed.push_back(1);
    }
  }

  // For each entry of UniqueOperandCommands, there is a set of instructions
  // that uses it.  If the next command of all instructions in the set are
  // identical, fold it into the command.
  for (unsigned CommandIdx = 0, e = UniqueOperandCommands.size();
       CommandIdx != e; ++CommandIdx) {

    for (unsigned Op = 1; ; ++Op) {
      // Scan for the first instruction in the set.
      std::vector<unsigned>::iterator NIT =
        std::find(InstIdxs.begin(), InstIdxs.end(), CommandIdx);
      if (NIT == InstIdxs.end()) break;  // No commonality.

      // If this instruction has no more operands, we isn't anything to merge
      // into this command.
      const AsmWriterInst *FirstInst =
        getAsmWriterInstByID(NIT-InstIdxs.begin());
      if (!FirstInst || FirstInst->Operands.size() == Op)
        break;

      // Otherwise, scan to see if all of the other instructions in this command
      // set share the operand.
      bool AllSame = true;
      // Keep track of the maximum, number of operands or any
      // instruction we see in the group.
      size_t MaxSize = FirstInst->Operands.size();

      for (NIT = std::find(NIT+1, InstIdxs.end(), CommandIdx);
           NIT != InstIdxs.end();
           NIT = std::find(NIT+1, InstIdxs.end(), CommandIdx)) {
        // Okay, found another instruction in this command set.  If the operand
        // matches, we're ok, otherwise bail out.
        const AsmWriterInst *OtherInst =
          getAsmWriterInstByID(NIT-InstIdxs.begin());

        if (OtherInst &&
            OtherInst->Operands.size() > FirstInst->Operands.size())
          MaxSize = std::max(MaxSize, OtherInst->Operands.size());

        if (!OtherInst || OtherInst->Operands.size() == Op ||
            OtherInst->Operands[Op] != FirstInst->Operands[Op]) {
          AllSame = false;
          break;
        }
      }
      if (!AllSame) break;

      // Okay, everything in this command set has the same next operand.  Add it
      // to UniqueOperandCommands and remember that it was consumed.
      std::string Command = "    " + FirstInst->Operands[Op].getCode() + "\n";

      UniqueOperandCommands[CommandIdx] += Command;
      InstOpsUsed[CommandIdx]++;
    }
  }

  // Prepend some of the instructions each case is used for onto the case val.
  for (unsigned i = 0, e = InstrsForCase.size(); i != e; ++i) {
    std::string Instrs = InstrsForCase[i];
    if (Instrs.size() > 70) {
      Instrs.erase(Instrs.begin()+70, Instrs.end());
      Instrs += "...";
    }

    if (!Instrs.empty())
      UniqueOperandCommands[i] = "    // " + Instrs + "\n" +
        UniqueOperandCommands[i];
  }
}


static void UnescapeString(std::string &Str) {
  for (unsigned i = 0; i != Str.size(); ++i) {
    if (Str[i] == '\\' && i != Str.size()-1) {
      switch (Str[i+1]) {
      default: continue;  // Don't execute the code after the switch.
      case 'a': Str[i] = '\a'; break;
      case 'b': Str[i] = '\b'; break;
      case 'e': Str[i] = 27; break;
      case 'f': Str[i] = '\f'; break;
      case 'n': Str[i] = '\n'; break;
      case 'r': Str[i] = '\r'; break;
      case 't': Str[i] = '\t'; break;
      case 'v': Str[i] = '\v'; break;
      case '"': Str[i] = '\"'; break;
      case '\'': Str[i] = '\''; break;
      case '\\': Str[i] = '\\'; break;
      }
      // Nuke the second character.
      Str.erase(Str.begin()+i+1);
    }
  }
}

/// EmitPrintInstruction - Generate the code for the "printInstruction" method
/// implementation.
void AsmWriterEmitter::EmitPrintInstruction(raw_ostream &O) {
  CodeGenTarget Target(Records);
  Record *AsmWriter = Target.getAsmWriter();
  std::string ClassName = AsmWriter->getValueAsString("AsmWriterClassName");
  bool isMC = AsmWriter->getValueAsBit("isMCAsmWriter");
  const char *MachineInstrClassName = isMC ? "MCInst" : "MachineInstr";

  O <<
  "/// printInstruction - This method is automatically generated by tablegen\n"
  "/// from the instruction set description.\n"
    "void " << Target.getName() << ClassName
            << "::printInstruction(const " << MachineInstrClassName
            << " *MI, raw_ostream &O) {\n";

  std::vector<AsmWriterInst> Instructions;

  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I)
    if (!(*I)->AsmString.empty() &&
        (*I)->TheDef->getName() != "PHI")
      Instructions.push_back(
        AsmWriterInst(**I,
                      AsmWriter->getValueAsInt("Variant"),
                      AsmWriter->getValueAsInt("FirstOperandColumn"),
                      AsmWriter->getValueAsInt("OperandSpacing")));

  // Get the instruction numbering.
  NumberedInstructions = Target.getInstructionsByEnumValue();

  // Compute the CodeGenInstruction -> AsmWriterInst mapping.  Note that not
  // all machine instructions are necessarily being printed, so there may be
  // target instructions not in this map.
  for (unsigned i = 0, e = Instructions.size(); i != e; ++i)
    CGIAWIMap.insert(std::make_pair(Instructions[i].CGI, &Instructions[i]));

  // Build an aggregate string, and build a table of offsets into it.
  SequenceToOffsetTable<std::string> StringTable;

  /// OpcodeInfo - This encodes the index of the string to use for the first
  /// chunk of the output as well as indices used for operand printing.
  /// To reduce the number of unhandled cases, we expand the size from 32-bit
  /// to 32+16 = 48-bit.
  std::vector<uint64_t> OpcodeInfo;

  // Add all strings to the string table upfront so it can generate an optimized
  // representation.
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    AsmWriterInst *AWI = CGIAWIMap[NumberedInstructions[i]];
    if (AWI != 0 &&
        AWI->Operands[0].OperandType ==
                 AsmWriterOperand::isLiteralTextOperand &&
        !AWI->Operands[0].Str.empty()) {
      std::string Str = AWI->Operands[0].Str;
      UnescapeString(Str);
      StringTable.add(Str);
    }
  }

  StringTable.layout();

  unsigned MaxStringIdx = 0;
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    AsmWriterInst *AWI = CGIAWIMap[NumberedInstructions[i]];
    unsigned Idx;
    if (AWI == 0) {
      // Something not handled by the asmwriter printer.
      Idx = ~0U;
    } else if (AWI->Operands[0].OperandType !=
                        AsmWriterOperand::isLiteralTextOperand ||
               AWI->Operands[0].Str.empty()) {
      // Something handled by the asmwriter printer, but with no leading string.
      Idx = StringTable.get("");
    } else {
      std::string Str = AWI->Operands[0].Str;
      UnescapeString(Str);
      Idx = StringTable.get(Str);
      MaxStringIdx = std::max(MaxStringIdx, Idx);

      // Nuke the string from the operand list.  It is now handled!
      AWI->Operands.erase(AWI->Operands.begin());
    }

    // Bias offset by one since we want 0 as a sentinel.
    OpcodeInfo.push_back(Idx+1);
  }

  // Figure out how many bits we used for the string index.
  unsigned AsmStrBits = Log2_32_Ceil(MaxStringIdx+2);

  // To reduce code size, we compactify common instructions into a few bits
  // in the opcode-indexed table.
  unsigned BitsLeft = 64-AsmStrBits;

  std::vector<std::vector<std::string> > TableDrivenOperandPrinters;

  while (1) {
    std::vector<std::string> UniqueOperandCommands;
    std::vector<unsigned> InstIdxs;
    std::vector<unsigned> NumInstOpsHandled;
    FindUniqueOperandCommands(UniqueOperandCommands, InstIdxs,
                              NumInstOpsHandled);

    // If we ran out of operands to print, we're done.
    if (UniqueOperandCommands.empty()) break;

    // Compute the number of bits we need to represent these cases, this is
    // ceil(log2(numentries)).
    unsigned NumBits = Log2_32_Ceil(UniqueOperandCommands.size());

    // If we don't have enough bits for this operand, don't include it.
    if (NumBits > BitsLeft) {
      DEBUG(errs() << "Not enough bits to densely encode " << NumBits
                   << " more bits\n");
      break;
    }

    // Otherwise, we can include this in the initial lookup table.  Add it in.
    for (unsigned i = 0, e = InstIdxs.size(); i != e; ++i)
      if (InstIdxs[i] != ~0U) {
        OpcodeInfo[i] |= (uint64_t)InstIdxs[i] << (64-BitsLeft);
      }
    BitsLeft -= NumBits;

    // Remove the info about this operand.
    for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
      if (AsmWriterInst *Inst = getAsmWriterInstByID(i))
        if (!Inst->Operands.empty()) {
          unsigned NumOps = NumInstOpsHandled[InstIdxs[i]];
          assert(NumOps <= Inst->Operands.size() &&
                 "Can't remove this many ops!");
          Inst->Operands.erase(Inst->Operands.begin(),
                               Inst->Operands.begin()+NumOps);
        }
    }

    // Remember the handlers for this set of operands.
    TableDrivenOperandPrinters.push_back(UniqueOperandCommands);
  }


  // We always emit at least one 32-bit table. A second table is emitted if
  // more bits are needed.
  O<<"  static const uint32_t OpInfo[] = {\n";
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    O << "    " << (OpcodeInfo[i] & 0xffffffff) << "U,\t// "
      << NumberedInstructions[i]->TheDef->getName() << "\n";
  }
  // Add a dummy entry so the array init doesn't end with a comma.
  O << "    0U\n";
  O << "  };\n\n";

  if (BitsLeft < 32) {
    // Add a second OpInfo table only when it is necessary.
    // Adjust the type of the second table based on the number of bits needed.
    O << "  static const uint"
      << ((BitsLeft < 16) ? "32" : (BitsLeft < 24) ? "16" : "8")
      << "_t OpInfo2[] = {\n";
    for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
      O << "    " << (OpcodeInfo[i] >> 32) << "U,\t// "
        << NumberedInstructions[i]->TheDef->getName() << "\n";
    }
    // Add a dummy entry so the array init doesn't end with a comma.
    O << "    0U\n";
    O << "  };\n\n";
  }

  // Emit the string itself.
  O << "  const char AsmStrs[] = {\n";
  StringTable.emit(O, printChar);
  O << "  };\n\n";

  O << "  O << \"\\t\";\n\n";

  O << "  // Emit the opcode for the instruction.\n";
  if (BitsLeft < 32) {
    // If we have two tables then we need to perform two lookups and combine
    // the results into a single 64-bit value.
    O << "  uint64_t Bits1 = OpInfo[MI->getOpcode()];\n"
      << "  uint64_t Bits2 = OpInfo2[MI->getOpcode()];\n"
      << "  uint64_t Bits = (Bits2 << 32) | Bits1;\n";
  } else {
    // If only one table is used we just need to perform a single lookup.
    O << "  uint32_t Bits = OpInfo[MI->getOpcode()];\n";
  }
  O << "  assert(Bits != 0 && \"Cannot print this instruction.\");\n"
    << "  O << AsmStrs+(Bits & " << (1 << AsmStrBits)-1 << ")-1;\n\n";

  // Output the table driven operand information.
  BitsLeft = 64-AsmStrBits;
  for (unsigned i = 0, e = TableDrivenOperandPrinters.size(); i != e; ++i) {
    std::vector<std::string> &Commands = TableDrivenOperandPrinters[i];

    // Compute the number of bits we need to represent these cases, this is
    // ceil(log2(numentries)).
    unsigned NumBits = Log2_32_Ceil(Commands.size());
    assert(NumBits <= BitsLeft && "consistency error");

    // Emit code to extract this field from Bits.
    O << "\n  // Fragment " << i << " encoded into " << NumBits
      << " bits for " << Commands.size() << " unique commands.\n";

    if (Commands.size() == 2) {
      // Emit two possibilitys with if/else.
      O << "  if ((Bits >> "
        << (64-BitsLeft) << ") & "
        << ((1 << NumBits)-1) << ") {\n"
        << Commands[1]
        << "  } else {\n"
        << Commands[0]
        << "  }\n\n";
    } else if (Commands.size() == 1) {
      // Emit a single possibility.
      O << Commands[0] << "\n\n";
    } else {
      O << "  switch ((Bits >> "
        << (64-BitsLeft) << ") & "
        << ((1 << NumBits)-1) << ") {\n"
        << "  default:   // unreachable.\n";

      // Print out all the cases.
      for (unsigned i = 0, e = Commands.size(); i != e; ++i) {
        O << "  case " << i << ":\n";
        O << Commands[i];
        O << "    break;\n";
      }
      O << "  }\n\n";
    }
    BitsLeft -= NumBits;
  }

  // Okay, delete instructions with no operand info left.
  for (unsigned i = 0, e = Instructions.size(); i != e; ++i) {
    // Entire instruction has been emitted?
    AsmWriterInst &Inst = Instructions[i];
    if (Inst.Operands.empty()) {
      Instructions.erase(Instructions.begin()+i);
      --i; --e;
    }
  }


  // Because this is a vector, we want to emit from the end.  Reverse all of the
  // elements in the vector.
  std::reverse(Instructions.begin(), Instructions.end());


  // Now that we've emitted all of the operand info that fit into 32 bits, emit
  // information for those instructions that are left.  This is a less dense
  // encoding, but we expect the main 32-bit table to handle the majority of
  // instructions.
  if (!Instructions.empty()) {
    // Find the opcode # of inline asm.
    O << "  switch (MI->getOpcode()) {\n";
    while (!Instructions.empty())
      EmitInstructions(Instructions, O);

    O << "  }\n";
    O << "  return;\n";
  }

  O << "}\n";
}

static void
emitRegisterNameString(raw_ostream &O, StringRef AltName,
                       const std::vector<CodeGenRegister*> &Registers) {
  SequenceToOffsetTable<std::string> StringTable;
  SmallVector<std::string, 4> AsmNames(Registers.size());
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    const CodeGenRegister &Reg = *Registers[i];
    std::string &AsmName = AsmNames[i];

    // "NoRegAltName" is special. We don't need to do a lookup for that,
    // as it's just a reference to the default register name.
    if (AltName == "" || AltName == "NoRegAltName") {
      AsmName = Reg.TheDef->getValueAsString("AsmName");
      if (AsmName.empty())
        AsmName = Reg.getName();
    } else {
      // Make sure the register has an alternate name for this index.
      std::vector<Record*> AltNameList =
        Reg.TheDef->getValueAsListOfDefs("RegAltNameIndices");
      unsigned Idx = 0, e;
      for (e = AltNameList.size();
           Idx < e && (AltNameList[Idx]->getName() != AltName);
           ++Idx)
        ;
      // If the register has an alternate name for this index, use it.
      // Otherwise, leave it empty as an error flag.
      if (Idx < e) {
        std::vector<std::string> AltNames =
          Reg.TheDef->getValueAsListOfStrings("AltNames");
        if (AltNames.size() <= Idx)
          PrintFatalError(Reg.TheDef->getLoc(),
            (Twine("Register definition missing alt name for '") +
             AltName + "'.").str());
        AsmName = AltNames[Idx];
      }
    }
    StringTable.add(AsmName);
  }

  StringTable.layout();
  O << "  static const char AsmStrs" << AltName << "[] = {\n";
  StringTable.emit(O, printChar);
  O << "  };\n\n";

  O << "  static const uint32_t RegAsmOffset" << AltName << "[] = {";
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    if ((i % 14) == 0)
      O << "\n    ";
    O << StringTable.get(AsmNames[i]) << ", ";
  }
  O << "\n  };\n"
    << "\n";
}

void AsmWriterEmitter::EmitGetRegisterName(raw_ostream &O) {
  CodeGenTarget Target(Records);
  Record *AsmWriter = Target.getAsmWriter();
  std::string ClassName = AsmWriter->getValueAsString("AsmWriterClassName");
  const std::vector<CodeGenRegister*> &Registers =
    Target.getRegBank().getRegisters();
  std::vector<Record*> AltNameIndices = Target.getRegAltNameIndices();
  bool hasAltNames = AltNameIndices.size() > 1;

  O <<
  "\n\n/// getRegisterName - This method is automatically generated by tblgen\n"
  "/// from the register set description.  This returns the assembler name\n"
  "/// for the specified register.\n"
  "const char *" << Target.getName() << ClassName << "::";
  if (hasAltNames)
    O << "\ngetRegisterName(unsigned RegNo, unsigned AltIdx) {\n";
  else
    O << "getRegisterName(unsigned RegNo) {\n";
  O << "  assert(RegNo && RegNo < " << (Registers.size()+1)
    << " && \"Invalid register number!\");\n"
    << "\n";

  if (hasAltNames) {
    for (unsigned i = 0, e = AltNameIndices.size(); i < e; ++i)
      emitRegisterNameString(O, AltNameIndices[i]->getName(), Registers);
  } else
    emitRegisterNameString(O, "", Registers);

  if (hasAltNames) {
    O << "  const uint32_t *RegAsmOffset;\n"
      << "  const char *AsmStrs;\n"
      << "  switch(AltIdx) {\n"
      << "  default: llvm_unreachable(\"Invalid register alt name index!\");\n";
    for (unsigned i = 0, e = AltNameIndices.size(); i < e; ++i) {
      StringRef Namespace = AltNameIndices[1]->getValueAsString("Namespace");
      StringRef AltName(AltNameIndices[i]->getName());
      O << "  case " << Namespace << "::" << AltName
        << ":\n"
        << "    AsmStrs = AsmStrs" << AltName  << ";\n"
        << "    RegAsmOffset = RegAsmOffset" << AltName << ";\n"
        << "    break;\n";
    }
    O << "}\n";
  }

  O << "  assert (*(AsmStrs+RegAsmOffset[RegNo-1]) &&\n"
    << "          \"Invalid alt name index for register!\");\n"
    << "  return AsmStrs+RegAsmOffset[RegNo-1];\n"
    << "}\n";
}

namespace {
// IAPrinter - Holds information about an InstAlias. Two InstAliases match if
// they both have the same conditionals. In which case, we cannot print out the
// alias for that pattern.
class IAPrinter {
  std::vector<std::string> Conds;
  std::map<StringRef, unsigned> OpMap;
  std::string Result;
  std::string AsmString;
  SmallVector<Record*, 4> ReqFeatures;
public:
  IAPrinter(std::string R, std::string AS)
    : Result(R), AsmString(AS) {}

  void addCond(const std::string &C) { Conds.push_back(C); }

  void addOperand(StringRef Op, unsigned Idx) { OpMap[Op] = Idx; }
  unsigned getOpIndex(StringRef Op) { return OpMap[Op]; }
  bool isOpMapped(StringRef Op) { return OpMap.find(Op) != OpMap.end(); }

  void print(raw_ostream &O) {
    if (Conds.empty() && ReqFeatures.empty()) {
      O.indent(6) << "return true;\n";
      return;
    }

    O << "if (";

    for (std::vector<std::string>::iterator
           I = Conds.begin(), E = Conds.end(); I != E; ++I) {
      if (I != Conds.begin()) {
        O << " &&\n";
        O.indent(8);
      }

      O << *I;
    }

    O << ") {\n";
    O.indent(6) << "// " << Result << "\n";
    O.indent(6) << "AsmString = \"" << AsmString << "\";\n";

    for (std::map<StringRef, unsigned>::iterator
           I = OpMap.begin(), E = OpMap.end(); I != E; ++I)
      O.indent(6) << "OpMap.push_back(std::make_pair(\"" << I->first << "\", "
                  << I->second << "));\n";

    O.indent(6) << "break;\n";
    O.indent(4) << '}';
  }

  bool operator==(const IAPrinter &RHS) {
    if (Conds.size() != RHS.Conds.size())
      return false;

    unsigned Idx = 0;
    for (std::vector<std::string>::iterator
           I = Conds.begin(), E = Conds.end(); I != E; ++I)
      if (*I != RHS.Conds[Idx++])
        return false;

    return true;
  }

  bool operator()(const IAPrinter &RHS) {
    if (Conds.size() < RHS.Conds.size())
      return true;

    unsigned Idx = 0;
    for (std::vector<std::string>::iterator
           I = Conds.begin(), E = Conds.end(); I != E; ++I)
      if (*I != RHS.Conds[Idx++])
        return *I < RHS.Conds[Idx++];

    return false;
  }
};

} // end anonymous namespace

static void EmitGetMapOperandNumber(raw_ostream &O) {
  O << "static unsigned getMapOperandNumber("
    << "const SmallVectorImpl<std::pair<StringRef, unsigned> > &OpMap,\n";
  O << "                                    StringRef Name) {\n";
  O << "  for (SmallVectorImpl<std::pair<StringRef, unsigned> >::"
    << "const_iterator\n";
  O << "         I = OpMap.begin(), E = OpMap.end(); I != E; ++I)\n";
  O << "    if (I->first == Name)\n";
  O << "      return I->second;\n";
  O << "  llvm_unreachable(\"Operand not in map!\");\n";
  O << "}\n\n";
}

static unsigned CountNumOperands(StringRef AsmString) {
  unsigned NumOps = 0;
  std::pair<StringRef, StringRef> ASM = AsmString.split(' ');

  while (!ASM.second.empty()) {
    ++NumOps;
    ASM = ASM.second.split(' ');
  }

  return NumOps;
}

static unsigned CountResultNumOperands(StringRef AsmString) {
  unsigned NumOps = 0;
  std::pair<StringRef, StringRef> ASM = AsmString.split('\t');

  if (!ASM.second.empty()) {
    size_t I = ASM.second.find('{');
    StringRef Str = ASM.second;
    if (I != StringRef::npos)
      Str = ASM.second.substr(I, ASM.second.find('|', I));

    ASM = Str.split(' ');

    do {
      ++NumOps;
      ASM = ASM.second.split(' ');
    } while (!ASM.second.empty());
  }

  return NumOps;
}

void AsmWriterEmitter::EmitPrintAliasInstruction(raw_ostream &O) {
  CodeGenTarget Target(Records);
  Record *AsmWriter = Target.getAsmWriter();

  if (!AsmWriter->getValueAsBit("isMCAsmWriter"))
    return;

  O << "\n#ifdef PRINT_ALIAS_INSTR\n";
  O << "#undef PRINT_ALIAS_INSTR\n\n";

  // Emit the method that prints the alias instruction.
  std::string ClassName = AsmWriter->getValueAsString("AsmWriterClassName");

  std::vector<Record*> AllInstAliases =
    Records.getAllDerivedDefinitions("InstAlias");

  // Create a map from the qualified name to a list of potential matches.
  std::map<std::string, std::vector<CodeGenInstAlias*> > AliasMap;
  for (std::vector<Record*>::iterator
         I = AllInstAliases.begin(), E = AllInstAliases.end(); I != E; ++I) {
    CodeGenInstAlias *Alias = new CodeGenInstAlias(*I, Target);
    const Record *R = *I;
    if (!R->getValueAsBit("EmitAlias"))
      continue; // We were told not to emit the alias, but to emit the aliasee.
    const DagInit *DI = R->getValueAsDag("ResultInst");
    const DefInit *Op = cast<DefInit>(DI->getOperator());
    AliasMap[getQualifiedName(Op->getDef())].push_back(Alias);
  }

  // A map of which conditions need to be met for each instruction operand
  // before it can be matched to the mnemonic.
  std::map<std::string, std::vector<IAPrinter*> > IAPrinterMap;

  for (std::map<std::string, std::vector<CodeGenInstAlias*> >::iterator
         I = AliasMap.begin(), E = AliasMap.end(); I != E; ++I) {
    std::vector<CodeGenInstAlias*> &Aliases = I->second;

    for (std::vector<CodeGenInstAlias*>::iterator
           II = Aliases.begin(), IE = Aliases.end(); II != IE; ++II) {
      const CodeGenInstAlias *CGA = *II;
      unsigned LastOpNo = CGA->ResultInstOperandIndex.size();
      unsigned NumResultOps =
        CountResultNumOperands(CGA->ResultInst->AsmString);

      // Don't emit the alias if it has more operands than what it's aliasing.
      if (NumResultOps < CountNumOperands(CGA->AsmString))
        continue;

      IAPrinter *IAP = new IAPrinter(CGA->Result->getAsString(),
                                     CGA->AsmString);

      std::string Cond;
      Cond = std::string("MI->getNumOperands() == ") + llvm::utostr(LastOpNo);
      IAP->addCond(Cond);

      std::map<StringRef, unsigned> OpMap;
      bool CantHandle = false;

      for (unsigned i = 0, e = LastOpNo; i != e; ++i) {
        const CodeGenInstAlias::ResultOperand &RO = CGA->ResultOperands[i];

        switch (RO.Kind) {
        case CodeGenInstAlias::ResultOperand::K_Record: {
          const Record *Rec = RO.getRecord();
          StringRef ROName = RO.getName();


          if (Rec->isSubClassOf("RegisterOperand"))
            Rec = Rec->getValueAsDef("RegClass");
          if (Rec->isSubClassOf("RegisterClass")) {
            Cond = std::string("MI->getOperand(")+llvm::utostr(i)+").isReg()";
            IAP->addCond(Cond);

            if (!IAP->isOpMapped(ROName)) {
              IAP->addOperand(ROName, i);
              Cond = std::string("MRI.getRegClass(") + Target.getName() + "::" +
                CGA->ResultOperands[i].getRecord()->getName() + "RegClassID)"
                ".contains(MI->getOperand(" + llvm::utostr(i) + ").getReg())";
              IAP->addCond(Cond);
            } else {
              Cond = std::string("MI->getOperand(") +
                llvm::utostr(i) + ").getReg() == MI->getOperand(" +
                llvm::utostr(IAP->getOpIndex(ROName)) + ").getReg()";
              IAP->addCond(Cond);
            }
          } else {
            assert(Rec->isSubClassOf("Operand") && "Unexpected operand!");
            // FIXME: We may need to handle these situations.
            delete IAP;
            IAP = 0;
            CantHandle = true;
            break;
          }

          break;
        }
        case CodeGenInstAlias::ResultOperand::K_Imm:
          Cond = std::string("MI->getOperand(") +
            llvm::utostr(i) + ").getImm() == " +
            llvm::utostr(CGA->ResultOperands[i].getImm());
          IAP->addCond(Cond);
          break;
        case CodeGenInstAlias::ResultOperand::K_Reg:
          // If this is zero_reg, something's playing tricks we're not
          // equipped to handle.
          if (!CGA->ResultOperands[i].getRegister()) {
            CantHandle = true;
            break;
          }

          Cond = std::string("MI->getOperand(") +
            llvm::utostr(i) + ").getReg() == " + Target.getName() +
            "::" + CGA->ResultOperands[i].getRegister()->getName();
          IAP->addCond(Cond);
          break;
        }

        if (!IAP) break;
      }

      if (CantHandle) continue;
      IAPrinterMap[I->first].push_back(IAP);
    }
  }

  std::string Header;
  raw_string_ostream HeaderO(Header);

  HeaderO << "bool " << Target.getName() << ClassName
          << "::printAliasInstr(const MCInst"
          << " *MI, raw_ostream &OS) {\n";

  std::string Cases;
  raw_string_ostream CasesO(Cases);

  for (std::map<std::string, std::vector<IAPrinter*> >::iterator
         I = IAPrinterMap.begin(), E = IAPrinterMap.end(); I != E; ++I) {
    std::vector<IAPrinter*> &IAPs = I->second;
    std::vector<IAPrinter*> UniqueIAPs;

    for (std::vector<IAPrinter*>::iterator
           II = IAPs.begin(), IE = IAPs.end(); II != IE; ++II) {
      IAPrinter *LHS = *II;
      bool IsDup = false;
      for (std::vector<IAPrinter*>::iterator
             III = IAPs.begin(), IIE = IAPs.end(); III != IIE; ++III) {
        IAPrinter *RHS = *III;
        if (LHS != RHS && *LHS == *RHS) {
          IsDup = true;
          break;
        }
      }

      if (!IsDup) UniqueIAPs.push_back(LHS);
    }

    if (UniqueIAPs.empty()) continue;

    CasesO.indent(2) << "case " << I->first << ":\n";

    for (std::vector<IAPrinter*>::iterator
           II = UniqueIAPs.begin(), IE = UniqueIAPs.end(); II != IE; ++II) {
      IAPrinter *IAP = *II;
      CasesO.indent(4);
      IAP->print(CasesO);
      CasesO << '\n';
    }

    CasesO.indent(4) << "return false;\n";
  }

  if (CasesO.str().empty()) {
    O << HeaderO.str();
    O << "  return false;\n";
    O << "}\n\n";
    O << "#endif // PRINT_ALIAS_INSTR\n";
    return;
  }

  EmitGetMapOperandNumber(O);

  O << HeaderO.str();
  O.indent(2) << "StringRef AsmString;\n";
  O.indent(2) << "SmallVector<std::pair<StringRef, unsigned>, 4> OpMap;\n";
  O.indent(2) << "switch (MI->getOpcode()) {\n";
  O.indent(2) << "default: return false;\n";
  O << CasesO.str();
  O.indent(2) << "}\n\n";

  // Code that prints the alias, replacing the operands with the ones from the
  // MCInst.
  O << "  std::pair<StringRef, StringRef> ASM = AsmString.split(' ');\n";
  O << "  OS << '\\t' << ASM.first;\n";

  O << "  if (!ASM.second.empty()) {\n";
  O << "    OS << '\\t';\n";
  O << "    for (StringRef::iterator\n";
  O << "         I = ASM.second.begin(), E = ASM.second.end(); I != E; ) {\n";
  O << "      if (*I == '$') {\n";
  O << "        StringRef::iterator Start = ++I;\n";
  O << "        while (I != E &&\n";
  O << "               ((*I >= 'a' && *I <= 'z') ||\n";
  O << "                (*I >= 'A' && *I <= 'Z') ||\n";
  O << "                (*I >= '0' && *I <= '9') ||\n";
  O << "                *I == '_'))\n";
  O << "          ++I;\n";
  O << "        StringRef Name(Start, I - Start);\n";
  O << "        printOperand(MI, getMapOperandNumber(OpMap, Name), OS);\n";
  O << "      } else {\n";
  O << "        OS << *I++;\n";
  O << "      }\n";
  O << "    }\n";
  O << "  }\n\n";

  O << "  return true;\n";
  O << "}\n\n";

  O << "#endif // PRINT_ALIAS_INSTR\n";
}

void AsmWriterEmitter::run(raw_ostream &O) {
  EmitPrintInstruction(O);
  EmitGetRegisterName(O);
  EmitPrintAliasInstruction(O);
}


namespace llvm {

void EmitAsmWriter(RecordKeeper &RK, raw_ostream &OS) {
  emitSourceFileHeader("Assembly Writer Source Fragment", OS);
  AsmWriterEmitter(RK).run(OS);
}

} // End llvm namespace
