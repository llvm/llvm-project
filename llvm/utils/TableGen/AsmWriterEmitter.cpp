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

#include "AsmWriterEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
using namespace llvm;

static bool isIdentChar(char C) {
  return (C >= 'a' && C <= 'z') ||
         (C >= 'A' && C <= 'Z') ||
         (C >= '0' && C <= '9') ||
         C == '_';
}

// This should be an anon namespace, this works around a GCC warning.
namespace llvm {  
  struct AsmWriterOperand {
    enum { isLiteralTextOperand, isMachineInstrOperand } OperandType;

    /// Str - For isLiteralTextOperand, this IS the literal text.  For
    /// isMachineInstrOperand, this is the PrinterMethodName for the operand.
    std::string Str;

    /// MiOpNo - For isMachineInstrOperand, this is the operand number of the
    /// machine instruction.
    unsigned MIOpNo;
    
    /// MiModifier - For isMachineInstrOperand, this is the modifier string for
    /// an operand, specified with syntax like ${opname:modifier}.
    std::string MiModifier;

    // To make VS STL happy
    AsmWriterOperand():OperandType(isLiteralTextOperand) {}

    AsmWriterOperand(const std::string &LitStr)
      : OperandType(isLiteralTextOperand), Str(LitStr) {}

    AsmWriterOperand(const std::string &Printer, unsigned OpNo, 
                     const std::string &Modifier) 
      : OperandType(isMachineInstrOperand), Str(Printer), MIOpNo(OpNo),
      MiModifier(Modifier) {}

    bool operator!=(const AsmWriterOperand &Other) const {
      if (OperandType != Other.OperandType || Str != Other.Str) return true;
      if (OperandType == isMachineInstrOperand)
        return MIOpNo != Other.MIOpNo || MiModifier != Other.MiModifier;
      return false;
    }
    bool operator==(const AsmWriterOperand &Other) const {
      return !operator!=(Other);
    }
    
    /// getCode - Return the code that prints this operand.
    std::string getCode() const;
  };
}

namespace llvm {
  class AsmWriterInst {
  public:
    std::vector<AsmWriterOperand> Operands;
    const CodeGenInstruction *CGI;

    AsmWriterInst(const CodeGenInstruction &CGI, unsigned Variant);

    /// MatchesAllButOneOp - If this instruction is exactly identical to the
    /// specified instruction except for one differing operand, return the
    /// differing operand number.  Otherwise return ~0.
    unsigned MatchesAllButOneOp(const AsmWriterInst &Other) const;

  private:
    void AddLiteralString(const std::string &Str) {
      // If the last operand was already a literal text string, append this to
      // it, otherwise add a new operand.
      if (!Operands.empty() &&
          Operands.back().OperandType == AsmWriterOperand::isLiteralTextOperand)
        Operands.back().Str.append(Str);
      else
        Operands.push_back(AsmWriterOperand(Str));
    }
  };
}


std::string AsmWriterOperand::getCode() const {
  if (OperandType == isLiteralTextOperand)
    return "O << \"" + Str + "\"; ";

  std::string Result = Str + "(MI";
  if (MIOpNo != ~0U)
    Result += ", " + utostr(MIOpNo);
  if (!MiModifier.empty())
    Result += ", \"" + MiModifier + '"';
  return Result + "); ";
}


/// ParseAsmString - Parse the specified Instruction's AsmString into this
/// AsmWriterInst.
///
AsmWriterInst::AsmWriterInst(const CodeGenInstruction &CGI, unsigned Variant) {
  this->CGI = &CGI;
  unsigned CurVariant = ~0U;  // ~0 if we are outside a {.|.|.} region, other #.

  // NOTE: Any extensions to this code need to be mirrored in the 
  // AsmPrinter::printInlineAsm code that executes as compile time (assuming
  // that inline asm strings should also get the new feature)!
  const std::string &AsmString = CGI.AsmString;
  std::string::size_type LastEmitted = 0;
  while (LastEmitted != AsmString.size()) {
    std::string::size_type DollarPos =
      AsmString.find_first_of("${|}\\", LastEmitted);
    if (DollarPos == std::string::npos) DollarPos = AsmString.size();

    // Emit a constant string fragment.
    if (DollarPos != LastEmitted) {
      // TODO: this should eventually handle escaping.
      if (CurVariant == Variant || CurVariant == ~0U)
        AddLiteralString(std::string(AsmString.begin()+LastEmitted,
                                     AsmString.begin()+DollarPos));
      LastEmitted = DollarPos;
    } else if (AsmString[DollarPos] == '\\') {
      if (DollarPos+1 != AsmString.size() &&
          (CurVariant == Variant || CurVariant == ~0U)) {
        if (AsmString[DollarPos+1] == 'n') {
          AddLiteralString("\\n");
        } else if (AsmString[DollarPos+1] == 't') {
          AddLiteralString("\\t");
        } else if (std::string("${|}\\").find(AsmString[DollarPos+1]) 
                   != std::string::npos) {
          AddLiteralString(std::string(1, AsmString[DollarPos+1]));
        } else {
          throw "Non-supported escaped character found in instruction '" +
            CGI.TheDef->getName() + "'!";
        }
        LastEmitted = DollarPos+2;
        continue;
      }
    } else if (AsmString[DollarPos] == '{') {
      if (CurVariant != ~0U)
        throw "Nested variants found for instruction '" +
              CGI.TheDef->getName() + "'!";
      LastEmitted = DollarPos+1;
      CurVariant = 0;   // We are now inside of the variant!
    } else if (AsmString[DollarPos] == '|') {
      if (CurVariant == ~0U)
        throw "'|' character found outside of a variant in instruction '"
          + CGI.TheDef->getName() + "'!";
      ++CurVariant;
      ++LastEmitted;
    } else if (AsmString[DollarPos] == '}') {
      if (CurVariant == ~0U)
        throw "'}' character found outside of a variant in instruction '"
          + CGI.TheDef->getName() + "'!";
      ++LastEmitted;
      CurVariant = ~0U;
    } else if (DollarPos+1 != AsmString.size() &&
               AsmString[DollarPos+1] == '$') {
      if (CurVariant == Variant || CurVariant == ~0U) 
        AddLiteralString("$");  // "$$" -> $
      LastEmitted = DollarPos+2;
    } else {
      // Get the name of the variable.
      std::string::size_type VarEnd = DollarPos+1;

      // handle ${foo}bar as $foo by detecting whether the character following
      // the dollar sign is a curly brace.  If so, advance VarEnd and DollarPos
      // so the variable name does not contain the leading curly brace.
      bool hasCurlyBraces = false;
      if (VarEnd < AsmString.size() && '{' == AsmString[VarEnd]) {
        hasCurlyBraces = true;
        ++DollarPos;
        ++VarEnd;
      }

      while (VarEnd < AsmString.size() && isIdentChar(AsmString[VarEnd]))
        ++VarEnd;
      std::string VarName(AsmString.begin()+DollarPos+1,
                          AsmString.begin()+VarEnd);

      // Modifier - Support ${foo:modifier} syntax, where "modifier" is passed
      // into printOperand.  Also support ${:feature}, which is passed into
      // PrintSpecial.
      std::string Modifier;
      
      // In order to avoid starting the next string at the terminating curly
      // brace, advance the end position past it if we found an opening curly
      // brace.
      if (hasCurlyBraces) {
        if (VarEnd >= AsmString.size())
          throw "Reached end of string before terminating curly brace in '"
                + CGI.TheDef->getName() + "'";
        
        // Look for a modifier string.
        if (AsmString[VarEnd] == ':') {
          ++VarEnd;
          if (VarEnd >= AsmString.size())
            throw "Reached end of string before terminating curly brace in '"
              + CGI.TheDef->getName() + "'";
          
          unsigned ModifierStart = VarEnd;
          while (VarEnd < AsmString.size() && isIdentChar(AsmString[VarEnd]))
            ++VarEnd;
          Modifier = std::string(AsmString.begin()+ModifierStart,
                                 AsmString.begin()+VarEnd);
          if (Modifier.empty())
            throw "Bad operand modifier name in '"+ CGI.TheDef->getName() + "'";
        }
        
        if (AsmString[VarEnd] != '}')
          throw "Variable name beginning with '{' did not end with '}' in '"
                + CGI.TheDef->getName() + "'";
        ++VarEnd;
      }
      if (VarName.empty() && Modifier.empty())
        throw "Stray '$' in '" + CGI.TheDef->getName() +
              "' asm string, maybe you want $$?";

      if (VarName.empty()) {
        // Just a modifier, pass this into PrintSpecial.
        Operands.push_back(AsmWriterOperand("PrintSpecial", ~0U, Modifier));
      } else {
        // Otherwise, normal operand.
        unsigned OpNo = CGI.getOperandNamed(VarName);
        CodeGenInstruction::OperandInfo OpInfo = CGI.OperandList[OpNo];

        if (CurVariant == Variant || CurVariant == ~0U) {
          unsigned MIOp = OpInfo.MIOperandNo;
          Operands.push_back(AsmWriterOperand(OpInfo.PrinterMethodName, MIOp,
                                              Modifier));
        }
      }
      LastEmitted = VarEnd;
    }
  }

  AddLiteralString("\\n");
}

/// MatchesAllButOneOp - If this instruction is exactly identical to the
/// specified instruction except for one differing operand, return the differing
/// operand number.  If more than one operand mismatches, return ~1, otherwise
/// if the instructions are identical return ~0.
unsigned AsmWriterInst::MatchesAllButOneOp(const AsmWriterInst &Other)const{
  if (Operands.size() != Other.Operands.size()) return ~1;

  unsigned MismatchOperand = ~0U;
  for (unsigned i = 0, e = Operands.size(); i != e; ++i) {
    if (Operands[i] != Other.Operands[i]) {
      if (MismatchOperand != ~0U)  // Already have one mismatch?
        return ~1U;
      else
        MismatchOperand = i;
    }
  }
  return MismatchOperand;
}

static void PrintCases(std::vector<std::pair<std::string,
                       AsmWriterOperand> > &OpsToPrint, std::ostream &O) {
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
                             std::ostream &O) {
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
    if (Inst == 0) continue;  // PHI, INLINEASM, DBG_LABEL, etc.
    
    std::string Command;
    if (Inst->Operands.empty())
      continue;   // Instruction already done.

    Command = "    " + Inst->Operands[0].getCode() + "\n";

    // If this is the last operand, emit a return.
    if (Inst->Operands.size() == 1)
      Command += "    return true;\n";
    
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
      
      for (NIT = std::find(NIT+1, InstIdxs.end(), CommandIdx);
           NIT != InstIdxs.end();
           NIT = std::find(NIT+1, InstIdxs.end(), CommandIdx)) {
        // Okay, found another instruction in this command set.  If the operand
        // matches, we're ok, otherwise bail out.
        const AsmWriterInst *OtherInst = 
          getAsmWriterInstByID(NIT-InstIdxs.begin());
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
      
      // If this is the last operand, emit a return after the code.
      if (FirstInst->Operands.size() == Op+1)
        Command += "    return true;\n";
      
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



void AsmWriterEmitter::run(std::ostream &O) {
  EmitSourceFileHeader("Assembly Writer Source Fragment", O);

  CodeGenTarget Target;
  Record *AsmWriter = Target.getAsmWriter();
  std::string ClassName = AsmWriter->getValueAsString("AsmWriterClassName");
  unsigned Variant = AsmWriter->getValueAsInt("Variant");

  O <<
  "/// printInstruction - This method is automatically generated by tablegen\n"
  "/// from the instruction set description.  This method returns true if the\n"
  "/// machine instruction was sufficiently described to print it, otherwise\n"
  "/// it returns false.\n"
    "bool " << Target.getName() << ClassName
            << "::printInstruction(const MachineInstr *MI) {\n";

  std::vector<AsmWriterInst> Instructions;

  for (CodeGenTarget::inst_iterator I = Target.inst_begin(),
         E = Target.inst_end(); I != E; ++I)
    if (!I->second.AsmString.empty())
      Instructions.push_back(AsmWriterInst(I->second, Variant));

  // Get the instruction numbering.
  Target.getInstructionsByEnumValue(NumberedInstructions);
  
  // Compute the CodeGenInstruction -> AsmWriterInst mapping.  Note that not
  // all machine instructions are necessarily being printed, so there may be
  // target instructions not in this map.
  for (unsigned i = 0, e = Instructions.size(); i != e; ++i)
    CGIAWIMap.insert(std::make_pair(Instructions[i].CGI, &Instructions[i]));

  // Build an aggregate string, and build a table of offsets into it.
  std::map<std::string, unsigned> StringOffset;
  std::string AggregateString;
  AggregateString.push_back(0);  // "\0"
  AggregateString.push_back(0);  // "\0"
  
  /// OpcodeInfo - This encodes the index of the string to use for the first
  /// chunk of the output as well as indices used for operand printing.
  std::vector<unsigned> OpcodeInfo;
  
  unsigned MaxStringIdx = 0;
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    AsmWriterInst *AWI = CGIAWIMap[NumberedInstructions[i]];
    unsigned Idx;
    if (AWI == 0) {
      // Something not handled by the asmwriter printer.
      Idx = 0;
    } else if (AWI->Operands[0].OperandType != 
                        AsmWriterOperand::isLiteralTextOperand ||
               AWI->Operands[0].Str.empty()) {
      // Something handled by the asmwriter printer, but with no leading string.
      Idx = 1;
    } else {
      unsigned &Entry = StringOffset[AWI->Operands[0].Str];
      if (Entry == 0) {
        // Add the string to the aggregate if this is the first time found.
        MaxStringIdx = Entry = AggregateString.size();
        std::string Str = AWI->Operands[0].Str;
        UnescapeString(Str);
        AggregateString += Str;
        AggregateString += '\0';
      }
      Idx = Entry;

      // Nuke the string from the operand list.  It is now handled!
      AWI->Operands.erase(AWI->Operands.begin());
    }
    OpcodeInfo.push_back(Idx);
  }
  
  // Figure out how many bits we used for the string index.
  unsigned AsmStrBits = Log2_32_Ceil(MaxStringIdx+1);
  
  // To reduce code size, we compactify common instructions into a few bits
  // in the opcode-indexed table.
  unsigned BitsLeft = 32-AsmStrBits;

  std::vector<std::vector<std::string> > TableDrivenOperandPrinters;
  
  bool isFirst = true;
  while (1) {
    std::vector<std::string> UniqueOperandCommands;

    // For the first operand check, add a default value for instructions with
    // just opcode strings to use.
    if (isFirst) {
      UniqueOperandCommands.push_back("    return true;\n");
      isFirst = false;
    }
    
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
      DOUT << "Not enough bits to densely encode " << NumBits
           << " more bits\n";
      break;
    }
    
    // Otherwise, we can include this in the initial lookup table.  Add it in.
    BitsLeft -= NumBits;
    for (unsigned i = 0, e = InstIdxs.size(); i != e; ++i)
      if (InstIdxs[i] != ~0U)
        OpcodeInfo[i] |= InstIdxs[i] << (BitsLeft+AsmStrBits);
    
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
  
  
  
  O<<"  static const unsigned OpInfo[] = {\n";
  for (unsigned i = 0, e = NumberedInstructions.size(); i != e; ++i) {
    O << "    " << OpcodeInfo[i] << "U,\t// "
      << NumberedInstructions[i]->TheDef->getName() << "\n";
  }
  // Add a dummy entry so the array init doesn't end with a comma.
  O << "    0U\n";
  O << "  };\n\n";
  
  // Emit the string itself.
  O << "  const char *AsmStrs = \n    \"";
  unsigned CharsPrinted = 0;
  EscapeString(AggregateString);
  for (unsigned i = 0, e = AggregateString.size(); i != e; ++i) {
    if (CharsPrinted > 70) {
      O << "\"\n    \"";
      CharsPrinted = 0;
    }
    O << AggregateString[i];
    ++CharsPrinted;
    
    // Print escape sequences all together.
    if (AggregateString[i] == '\\') {
      assert(i+1 < AggregateString.size() && "Incomplete escape sequence!");
      if (isdigit(AggregateString[i+1])) {
        assert(isdigit(AggregateString[i+2]) && isdigit(AggregateString[i+3]) &&
               "Expected 3 digit octal escape!");
        O << AggregateString[++i];
        O << AggregateString[++i];
        O << AggregateString[++i];
        CharsPrinted += 3;
      } else {
        O << AggregateString[++i];
        ++CharsPrinted;
      }
    }
  }
  O << "\";\n\n";

  O << "  if (MI->getOpcode() == TargetInstrInfo::INLINEASM) {\n"
    << "    O << \"\\t\";\n"
    << "    printInlineAsm(MI);\n"
    << "    return true;\n"
    << "  } else if (MI->isLabel()) {\n"
    << "    printLabel(MI);\n"
    << "    return true;\n"
    << "  } else if (MI->getOpcode() == TargetInstrInfo::DECLARE) {\n"
    << "    printDeclare(MI);\n"
    << "    return true;\n"
    << "  } else if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF) {\n"
    << "    printImplicitDef(MI);\n"
    << "    return true;\n"
    << "  }\n\n";
  
  O << "  O << \"\\t\";\n\n";

  O << "  // Emit the opcode for the instruction.\n"
    << "  unsigned Bits = OpInfo[MI->getOpcode()];\n"
    << "  if (Bits == 0) return false;\n"
    << "  O << AsmStrs+(Bits & " << (1 << AsmStrBits)-1 << ");\n\n";

  // Output the table driven operand information.
  BitsLeft = 32-AsmStrBits;
  for (unsigned i = 0, e = TableDrivenOperandPrinters.size(); i != e; ++i) {
    std::vector<std::string> &Commands = TableDrivenOperandPrinters[i];

    // Compute the number of bits we need to represent these cases, this is
    // ceil(log2(numentries)).
    unsigned NumBits = Log2_32_Ceil(Commands.size());
    assert(NumBits <= BitsLeft && "consistency error");
    
    // Emit code to extract this field from Bits.
    BitsLeft -= NumBits;
    
    O << "\n  // Fragment " << i << " encoded into " << NumBits
      << " bits for " << Commands.size() << " unique commands.\n";
    
    if (Commands.size() == 2) {
      // Emit two possibilitys with if/else.
      O << "  if ((Bits >> " << (BitsLeft+AsmStrBits) << ") & "
        << ((1 << NumBits)-1) << ") {\n"
        << Commands[1]
        << "  } else {\n"
        << Commands[0]
        << "  }\n\n";
    } else {
      O << "  switch ((Bits >> " << (BitsLeft+AsmStrBits) << ") & "
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
  
  if (!Instructions.empty()) {
    // Find the opcode # of inline asm.
    O << "  switch (MI->getOpcode()) {\n";
    while (!Instructions.empty())
      EmitInstructions(Instructions, O);

    O << "  }\n";
    O << "  return true;\n";
  }
  
  O << "}\n";
}
