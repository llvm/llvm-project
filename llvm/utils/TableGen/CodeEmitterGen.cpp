//===- CodeEmitterGen.cpp - Code Emitter Generator ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CodeEmitterGen uses the descriptions of instructions and their fields to
// construct an automated code emitter: a function that, given a MachineInstr,
// returns the (currently, 32-bit unsigned) value of the instruction.
//
//===----------------------------------------------------------------------===//

#include "CodeGenHwModes.h"
#include "CodeGenInstruction.h"
#include "CodeGenTarget.h"
#include "InfoByHwMode.h"
#include "TableGenBackends.h"
#include "VarLenCodeEmitterGen.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

class CodeEmitterGen {
  RecordKeeper &Records;

public:
  CodeEmitterGen(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &o);

private:
  int getVariableBit(const std::string &VarName, BitsInit *BI, int bit);
  std::string getInstructionCase(Record *R, CodeGenTarget &Target);
  std::string getInstructionCaseForEncoding(Record *R, Record *EncodingDef,
                                            CodeGenTarget &Target);
  bool addCodeToMergeInOperand(Record *R, BitsInit *BI,
                               const std::string &VarName,
                               std::string &Case, CodeGenTarget &Target);

  void emitInstructionBaseValues(
      raw_ostream &o, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
      CodeGenTarget &Target, int HwMode = -1);
  unsigned BitWidth;
  bool UseAPInt;
};

// If the VarBitInit at position 'bit' matches the specified variable then
// return the variable bit position.  Otherwise return -1.
int CodeEmitterGen::getVariableBit(const std::string &VarName,
                                   BitsInit *BI, int bit) {
  if (VarBitInit *VBI = dyn_cast<VarBitInit>(BI->getBit(bit))) {
    if (VarInit *VI = dyn_cast<VarInit>(VBI->getBitVar()))
      if (VI->getName() == VarName)
        return VBI->getBitNum();
  } else if (VarInit *VI = dyn_cast<VarInit>(BI->getBit(bit))) {
    if (VI->getName() == VarName)
      return 0;
  }

  return -1;
}

// Returns true if it succeeds, false if an error.
bool CodeEmitterGen::addCodeToMergeInOperand(Record *R, BitsInit *BI,
                                             const std::string &VarName,
                                             std::string &Case,
                                             CodeGenTarget &Target) {
  CodeGenInstruction &CGI = Target.getInstruction(R);

  // Determine if VarName actually contributes to the Inst encoding.
  int bit = BI->getNumBits()-1;

  // Scan for a bit that this contributed to.
  for (; bit >= 0; ) {
    if (getVariableBit(VarName, BI, bit) != -1)
      break;
    
    --bit;
  }
  
  // If we found no bits, ignore this value, otherwise emit the call to get the
  // operand encoding.
  if (bit < 0)
    return true;

  // If the operand matches by name, reference according to that
  // operand number. Non-matching operands are assumed to be in
  // order.
  unsigned OpIdx;
  std::pair<unsigned, unsigned> SubOp;
  if (CGI.Operands.hasSubOperandAlias(VarName, SubOp)) {
    OpIdx = CGI.Operands[SubOp.first].MIOperandNo + SubOp.second;
  } else if (CGI.Operands.hasOperandNamed(VarName, OpIdx)) {
    // Get the machine operand number for the indicated operand.
    OpIdx = CGI.Operands[OpIdx].MIOperandNo;
  } else {
    PrintError(R, Twine("No operand named ") + VarName + " in record " + R->getName());
    return false;
  }

  if (CGI.Operands.isFlatOperandNotEmitted(OpIdx)) {
    PrintError(R, "Operand " + VarName + " used but also marked as not emitted!");
    return false;
  }

  std::pair<unsigned, unsigned> SO = CGI.Operands.getSubOperandNumber(OpIdx);
  std::string &EncoderMethodName =
      CGI.Operands[SO.first].EncoderMethodNames[SO.second];

  if (UseAPInt)
    Case += "      op.clearAllBits();\n";

  Case += "      // op: " + VarName + "\n";

  // If the source operand has a custom encoder, use it.
  if (!EncoderMethodName.empty()) {
    if (UseAPInt) {
      Case += "      " + EncoderMethodName + "(MI, " + utostr(OpIdx);
      Case += ", op";
    } else {
      Case += "      op = " + EncoderMethodName + "(MI, " + utostr(OpIdx);
    }
    Case += ", Fixups, STI);\n";
  } else {
    if (UseAPInt) {
      Case += "      getMachineOpValue(MI, MI.getOperand(" + utostr(OpIdx) + ")";
      Case += ", op, Fixups, STI";
    } else {
      Case += "      op = getMachineOpValue(MI, MI.getOperand(" + utostr(OpIdx) + ")";
      Case += ", Fixups, STI";
    }
    Case += ");\n";
  }

  // Precalculate the number of lits this variable contributes to in the
  // operand. If there is a single lit (consecutive range of bits) we can use a
  // destructive sequence on APInt that reduces memory allocations.
  int numOperandLits = 0;
  for (int tmpBit = bit; tmpBit >= 0;) {
    int varBit = getVariableBit(VarName, BI, tmpBit);

    // If this bit isn't from a variable, skip it.
    if (varBit == -1) {
      --tmpBit;
      continue;
    }

    // Figure out the consecutive range of bits covered by this operand, in
    // order to generate better encoding code.
    int beginVarBit = varBit;
    int N = 1;
    for (--tmpBit; tmpBit >= 0;) {
      varBit = getVariableBit(VarName, BI, tmpBit);
      if (varBit == -1 || varBit != (beginVarBit - N))
        break;
      ++N;
      --tmpBit;
    }
    ++numOperandLits;
  }

  for (; bit >= 0; ) {
    int varBit = getVariableBit(VarName, BI, bit);
    
    // If this bit isn't from a variable, skip it.
    if (varBit == -1) {
      --bit;
      continue;
    }
    
    // Figure out the consecutive range of bits covered by this operand, in
    // order to generate better encoding code.
    int beginInstBit = bit;
    int beginVarBit = varBit;
    int N = 1;
    for (--bit; bit >= 0;) {
      varBit = getVariableBit(VarName, BI, bit);
      if (varBit == -1 || varBit != (beginVarBit - N)) break;
      ++N;
      --bit;
    }

    std::string maskStr;
    int opShift;

    unsigned loBit = beginVarBit - N + 1;
    unsigned hiBit = loBit + N;
    unsigned loInstBit = beginInstBit - N + 1;
    if (UseAPInt) {
      std::string extractStr;
      if (N >= 64) {
        extractStr = "op.extractBits(" + itostr(hiBit - loBit) + ", " +
                     itostr(loBit) + ")";
        Case += "      Value.insertBits(" + extractStr + ", " +
                itostr(loInstBit) + ");\n";
      } else {
        extractStr = "op.extractBitsAsZExtValue(" + itostr(hiBit - loBit) +
                     ", " + itostr(loBit) + ")";
        Case += "      Value.insertBits(" + extractStr + ", " +
                itostr(loInstBit) + ", " + itostr(hiBit - loBit) + ");\n";
      }
    } else {
      uint64_t opMask = ~(uint64_t)0 >> (64 - N);
      opShift = beginVarBit - N + 1;
      opMask <<= opShift;
      maskStr = "UINT64_C(" + utostr(opMask) + ")";
      opShift = beginInstBit - beginVarBit;

      if (numOperandLits == 1) {
        Case += "      op &= " + maskStr + ";\n";
        if (opShift > 0) {
          Case += "      op <<= " + itostr(opShift) + ";\n";
        } else if (opShift < 0) {
          Case += "      op >>= " + itostr(-opShift) + ";\n";
        }
        Case += "      Value |= op;\n";
      } else {
        if (opShift > 0) {
          Case += "      Value |= (op & " + maskStr + ") << " +
                  itostr(opShift) + ";\n";
        } else if (opShift < 0) {
          Case += "      Value |= (op & " + maskStr + ") >> " +
                  itostr(-opShift) + ";\n";
        } else {
          Case += "      Value |= (op & " + maskStr + ");\n";
        }
      }
    }
  }
  return true;
}

std::string CodeEmitterGen::getInstructionCase(Record *R,
                                               CodeGenTarget &Target) {
  std::string Case;
  if (const RecordVal *RV = R->getValue("EncodingInfos")) {
    if (auto *DI = dyn_cast_or_null<DefInit>(RV->getValue())) {
      const CodeGenHwModes &HWM = Target.getHwModes();
      EncodingInfoByHwMode EBM(DI->getDef(), HWM);
      Case += "      switch (HwMode) {\n";
      Case += "      default: llvm_unreachable(\"Unhandled HwMode\");\n";
      for (auto &KV : EBM) {
        Case += "      case " + itostr(KV.first) + ": {\n";
        Case += getInstructionCaseForEncoding(R, KV.second, Target);
        Case += "      break;\n";
        Case += "      }\n";
      }
      Case += "      }\n";
      return Case;
    }
  }
  return getInstructionCaseForEncoding(R, R, Target);
}

std::string CodeEmitterGen::getInstructionCaseForEncoding(Record *R, Record *EncodingDef,
                                                          CodeGenTarget &Target) {
  std::string Case;
  BitsInit *BI = EncodingDef->getValueAsBitsInit("Inst");

  // Loop over all of the fields in the instruction, determining which are the
  // operands to the instruction.
  bool Success = true;
  for (const RecordVal &RV : EncodingDef->getValues()) {
    // Ignore fixed fields in the record, we're looking for values like:
    //    bits<5> RST = { ?, ?, ?, ?, ? };
    if (RV.isNonconcreteOK() || RV.getValue()->isComplete())
      continue;

    Success &=
        addCodeToMergeInOperand(R, BI, std::string(RV.getName()),
                                Case, Target);
  }

  if (!Success) {
    // Dump the record, so we can see what's going on...
    std::string E;
    raw_string_ostream S(E);
    S << "Dumping record for previous error:\n";
    S << *R;
    PrintNote(E);
  }

  StringRef PostEmitter = R->getValueAsString("PostEncoderMethod");
  if (!PostEmitter.empty()) {
    Case += "      Value = ";
    Case += PostEmitter;
    Case += "(MI, Value";
    Case += ", STI";
    Case += ");\n";
  }
  
  return Case;
}

static void emitInstBits(raw_ostream &OS, const APInt &Bits) {
  for (unsigned I = 0; I < Bits.getNumWords(); ++I)
    OS << ((I > 0) ? ", " : "") << "UINT64_C(" << utostr(Bits.getRawData()[I])
       << ")";
}

void CodeEmitterGen::emitInstructionBaseValues(
    raw_ostream &o, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
    CodeGenTarget &Target, int HwMode) {
  const CodeGenHwModes &HWM = Target.getHwModes();
  if (HwMode == -1)
    o << "  static const uint64_t InstBits[] = {\n";
  else
    o << "  static const uint64_t InstBits_" << HWM.getMode(HwMode).Name
      << "[] = {\n";

  for (const CodeGenInstruction *CGI : NumberedInstructions) {
    Record *R = CGI->TheDef;

    if (R->getValueAsString("Namespace") == "TargetOpcode" ||
        R->getValueAsBit("isPseudo")) {
      o << "    "; emitInstBits(o, APInt(BitWidth, 0)); o << ",\n";
      continue;
    }

    Record *EncodingDef = R;
    if (const RecordVal *RV = R->getValue("EncodingInfos")) {
      if (auto *DI = dyn_cast_or_null<DefInit>(RV->getValue())) {
        EncodingInfoByHwMode EBM(DI->getDef(), HWM);
        if (EBM.hasMode(HwMode))
          EncodingDef = EBM.get(HwMode);
      }
    }
    BitsInit *BI = EncodingDef->getValueAsBitsInit("Inst");

    // Start by filling in fixed values.
    APInt Value(BitWidth, 0);
    for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i) {
      if (auto *B = dyn_cast<BitInit>(BI->getBit(i)); B && B->getValue())
        Value.setBit(i);
    }
    o << "    ";
    emitInstBits(o, Value);
    o << "," << '\t' << "// " << R->getName() << "\n";
  }
  o << "    UINT64_C(0)\n  };\n";
}

void CodeEmitterGen::run(raw_ostream &o) {
  CodeGenTarget Target(Records);
  std::vector<Record*> Insts = Records.getAllDerivedDefinitions("Instruction");

  // For little-endian instruction bit encodings, reverse the bit order
  Target.reverseBitsForLittleEndianEncoding();

  ArrayRef<const CodeGenInstruction*> NumberedInstructions =
    Target.getInstructionsByEnumValue();

  if (any_of(NumberedInstructions, [](const CodeGenInstruction *CGI) {
        Record *R = CGI->TheDef;
        return R->getValue("Inst") && isa<DagInit>(R->getValueInit("Inst"));
      })) {
    emitVarLenCodeEmitter(Records, o);
  } else {
    const CodeGenHwModes &HWM = Target.getHwModes();
    // The set of HwModes used by instruction encodings.
    std::set<unsigned> HwModes;
    BitWidth = 0;
    for (const CodeGenInstruction *CGI : NumberedInstructions) {
      Record *R = CGI->TheDef;
      if (R->getValueAsString("Namespace") == "TargetOpcode" ||
          R->getValueAsBit("isPseudo"))
        continue;

      if (const RecordVal *RV = R->getValue("EncodingInfos")) {
        if (DefInit *DI = dyn_cast_or_null<DefInit>(RV->getValue())) {
          EncodingInfoByHwMode EBM(DI->getDef(), HWM);
          for (auto &KV : EBM) {
            BitsInit *BI = KV.second->getValueAsBitsInit("Inst");
            BitWidth = std::max(BitWidth, BI->getNumBits());
            HwModes.insert(KV.first);
          }
          continue;
        }
      }
      BitsInit *BI = R->getValueAsBitsInit("Inst");
      BitWidth = std::max(BitWidth, BI->getNumBits());
    }
    UseAPInt = BitWidth > 64;

    // Emit function declaration
    if (UseAPInt) {
      o << "void " << Target.getName()
        << "MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,\n"
        << "    SmallVectorImpl<MCFixup> &Fixups,\n"
        << "    APInt &Inst,\n"
        << "    APInt &Scratch,\n"
        << "    const MCSubtargetInfo &STI) const {\n";
    } else {
      o << "uint64_t " << Target.getName();
      o << "MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,\n"
        << "    SmallVectorImpl<MCFixup> &Fixups,\n"
        << "    const MCSubtargetInfo &STI) const {\n";
    }

    // Emit instruction base values
    if (HwModes.empty()) {
      emitInstructionBaseValues(o, NumberedInstructions, Target, -1);
    } else {
      for (unsigned HwMode : HwModes)
        emitInstructionBaseValues(o, NumberedInstructions, Target, (int)HwMode);
    }

    if (!HwModes.empty()) {
      o << "  const uint64_t *InstBits;\n";
      o << "  unsigned HwMode = STI.getHwMode();\n";
      o << "  switch (HwMode) {\n";
      o << "  default: llvm_unreachable(\"Unknown hardware mode!\"); break;\n";
      for (unsigned I : HwModes) {
        o << "  case " << I << ": InstBits = InstBits_" << HWM.getMode(I).Name
          << "; break;\n";
      }
      o << "  };\n";
    }

    // Map to accumulate all the cases.
    std::map<std::string, std::vector<std::string>> CaseMap;

    // Construct all cases statement for each opcode
    for (Record *R : Insts) {
      if (R->getValueAsString("Namespace") == "TargetOpcode" ||
          R->getValueAsBit("isPseudo"))
        continue;
      std::string InstName =
          (R->getValueAsString("Namespace") + "::" + R->getName()).str();
      std::string Case = getInstructionCase(R, Target);

      CaseMap[Case].push_back(std::move(InstName));
    }

    // Emit initial function code
    if (UseAPInt) {
      int NumWords = APInt::getNumWords(BitWidth);
      o << "  const unsigned opcode = MI.getOpcode();\n"
        << "  if (Scratch.getBitWidth() != " << BitWidth << ")\n"
        << "    Scratch = Scratch.zext(" << BitWidth << ");\n"
        << "  Inst = APInt(" << BitWidth << ", ArrayRef(InstBits + opcode * "
        << NumWords << ", " << NumWords << "));\n"
        << "  APInt &Value = Inst;\n"
        << "  APInt &op = Scratch;\n"
        << "  switch (opcode) {\n";
    } else {
      o << "  const unsigned opcode = MI.getOpcode();\n"
        << "  uint64_t Value = InstBits[opcode];\n"
        << "  uint64_t op = 0;\n"
        << "  (void)op;  // suppress warning\n"
        << "  switch (opcode) {\n";
    }

    // Emit each case statement
    std::map<std::string, std::vector<std::string>>::iterator IE, EE;
    for (IE = CaseMap.begin(), EE = CaseMap.end(); IE != EE; ++IE) {
      const std::string &Case = IE->first;
      std::vector<std::string> &InstList = IE->second;

      for (int i = 0, N = InstList.size(); i < N; i++) {
        if (i)
          o << "\n";
        o << "    case " << InstList[i] << ":";
      }
      o << " {\n";
      o << Case;
      o << "      break;\n"
        << "    }\n";
    }

    // Default case: unhandled opcode
    o << "  default:\n"
      << "    std::string msg;\n"
      << "    raw_string_ostream Msg(msg);\n"
      << "    Msg << \"Not supported instr: \" << MI;\n"
      << "    report_fatal_error(Msg.str().c_str());\n"
      << "  }\n";
    if (UseAPInt)
      o << "  Inst = Value;\n";
    else
      o << "  return Value;\n";
    o << "}\n\n";
  }
}

} // end anonymous namespace

namespace llvm {

void EmitCodeEmitter(RecordKeeper &RK, raw_ostream &OS) {
  emitSourceFileHeader("Machine Code Emitter", OS);
  CodeEmitterGen(RK).run(OS);
}

} // end namespace llvm
