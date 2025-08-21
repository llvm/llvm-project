//===- CodeEmitterGen.cpp - Code Emitter Generator ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CodeEmitterGen uses the descriptions of instructions and their fields to
// construct an automated code emitter: a function called
// getBinaryCodeForInstr() that, given a MCInst, returns the value of the
// instruction - either as an uint64_t or as an APInt, depending on the
// maximum bit width of all Inst definitions.
//
// In addition, it generates another function called getOperandBitOffset()
// that, given a MCInst and an operand index, returns the minimum of indices of
// all bits that carry some portion of the respective operand. When the target's
// encodeInstruction() stores the instruction in a little-endian byte order, the
// returned value is the offset of the start of the operand in the encoded
// instruction. Other targets might need to adjust the returned value according
// to their encodeInstruction() implementation.
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenHwModes.h"
#include "Common/CodeGenInstruction.h"
#include "Common/CodeGenTarget.h"
#include "Common/InfoByHwMode.h"
#include "Common/VarLenCodeEmitterGen.h"
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
  const RecordKeeper &Records;

public:
  CodeEmitterGen(const RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &O);

private:
  int getVariableBit(const std::string &VarName, const BitsInit *BI, int Bit);
  std::pair<std::string, std::string>
  getInstructionCases(const Record *R, const CodeGenTarget &Target);
  void addInstructionCasesForEncoding(const Record *R,
                                      const Record *EncodingDef,
                                      const CodeGenTarget &Target,
                                      std::string &Case,
                                      std::string &BitOffsetCase);
  bool addCodeToMergeInOperand(const Record *R, const BitsInit *BI,
                               const std::string &VarName, std::string &Case,
                               std::string &BitOffsetCase,
                               const CodeGenTarget &Target);

  void emitInstructionBaseValues(
      raw_ostream &O, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
      const CodeGenTarget &Target, unsigned HwMode = DefaultMode);
  void
  emitCaseMap(raw_ostream &O,
              const std::map<std::string, std::vector<std::string>> &CaseMap);
  unsigned BitWidth = 0u;
  bool UseAPInt = false;
};

// If the VarBitInit at position 'bit' matches the specified variable then
// return the variable bit position.  Otherwise return -1.
int CodeEmitterGen::getVariableBit(const std::string &VarName,
                                   const BitsInit *BI, int Bit) {
  if (const VarBitInit *VBI = dyn_cast<VarBitInit>(BI->getBit(Bit))) {
    if (const VarInit *VI = dyn_cast<VarInit>(VBI->getBitVar()))
      if (VI->getName() == VarName)
        return VBI->getBitNum();
  } else if (const VarInit *VI = dyn_cast<VarInit>(BI->getBit(Bit))) {
    if (VI->getName() == VarName)
      return 0;
  }

  return -1;
}

// Returns true if it succeeds, false if an error.
bool CodeEmitterGen::addCodeToMergeInOperand(const Record *R,
                                             const BitsInit *BI,
                                             const std::string &VarName,
                                             std::string &Case,
                                             std::string &BitOffsetCase,
                                             const CodeGenTarget &Target) {
  CodeGenInstruction &CGI = Target.getInstruction(R);

  // Determine if VarName actually contributes to the Inst encoding.
  int Bit = BI->getNumBits() - 1;

  // Scan for a bit that this contributed to.
  for (; Bit >= 0;) {
    if (getVariableBit(VarName, BI, Bit) != -1)
      break;

    --Bit;
  }

  // If we found no bits, ignore this value, otherwise emit the call to get the
  // operand encoding.
  if (Bit < 0)
    return true;

  // If the operand matches by name, reference according to that
  // operand number. Non-matching operands are assumed to be in
  // order.
  unsigned OpIdx;
  if (auto SubOp = CGI.Operands.findSubOperandAlias(VarName)) {
    OpIdx = CGI.Operands[SubOp->first].MIOperandNo + SubOp->second;
  } else if (auto MayBeOpIdx = CGI.Operands.findOperandNamed(VarName)) {
    // Get the machine operand number for the indicated operand.
    OpIdx = CGI.Operands[*MayBeOpIdx].MIOperandNo;
  } else {
    PrintError(R, Twine("No operand named ") + VarName + " in record " +
                      R->getName());
    return false;
  }

  if (CGI.Operands.isFlatOperandNotEmitted(OpIdx)) {
    PrintError(R,
               "Operand " + VarName + " used but also marked as not emitted!");
    return false;
  }

  std::pair<unsigned, unsigned> SO = CGI.Operands.getSubOperandNumber(OpIdx);
  StringRef EncoderMethodName =
      CGI.Operands[SO.first].EncoderMethodNames[SO.second];

  if (UseAPInt)
    Case += "      op.clearAllBits();\n";

  Case += "      // op: " + VarName + "\n";

  // If the source operand has a custom encoder, use it.
  if (!EncoderMethodName.empty()) {
    raw_string_ostream CaseOS(Case);
    CaseOS << indent(6);
    if (UseAPInt) {
      CaseOS << EncoderMethodName << "(MI, " + utostr(OpIdx) << ", op";
    } else {
      CaseOS << "op = " << EncoderMethodName << "(MI, " << utostr(OpIdx);
    }
    CaseOS << ", Fixups, STI);\n";
  } else {
    if (UseAPInt) {
      Case +=
          "      getMachineOpValue(MI, MI.getOperand(" + utostr(OpIdx) + ")";
      Case += ", op, Fixups, STI";
    } else {
      Case += "      op = getMachineOpValue(MI, MI.getOperand(" +
              utostr(OpIdx) + ")";
      Case += ", Fixups, STI";
    }
    Case += ");\n";
  }

  // Precalculate the number of lits this variable contributes to in the
  // operand. If there is a single lit (consecutive range of bits) we can use a
  // destructive sequence on APInt that reduces memory allocations.
  int NumOperandLits = 0;
  for (int TmpBit = Bit; TmpBit >= 0;) {
    int VarBit = getVariableBit(VarName, BI, TmpBit);

    // If this bit isn't from a variable, skip it.
    if (VarBit == -1) {
      --TmpBit;
      continue;
    }

    // Figure out the consecutive range of bits covered by this operand, in
    // order to generate better encoding code.
    int BeginVarBit = VarBit;
    int N = 1;
    for (--TmpBit; TmpBit >= 0;) {
      VarBit = getVariableBit(VarName, BI, TmpBit);
      if (VarBit == -1 || VarBit != (BeginVarBit - N))
        break;
      ++N;
      --TmpBit;
    }
    ++NumOperandLits;
  }

  unsigned BitOffset = -1;
  for (; Bit >= 0;) {
    int VarBit = getVariableBit(VarName, BI, Bit);

    // If this bit isn't from a variable, skip it.
    if (VarBit == -1) {
      --Bit;
      continue;
    }

    // Figure out the consecutive range of bits covered by this operand, in
    // order to generate better encoding code.
    int BeginInstBit = Bit;
    int BeginVarBit = VarBit;
    int N = 1;
    for (--Bit; Bit >= 0;) {
      VarBit = getVariableBit(VarName, BI, Bit);
      if (VarBit == -1 || VarBit != (BeginVarBit - N))
        break;
      ++N;
      --Bit;
    }

    std::string MaskStr;
    int OpShift;

    unsigned LoBit = BeginVarBit - N + 1;
    unsigned HiBit = LoBit + N;
    unsigned LoInstBit = BeginInstBit - N + 1;
    BitOffset = LoInstBit;
    if (UseAPInt) {
      std::string ExtractStr;
      if (N >= 64) {
        ExtractStr = "op.extractBits(" + itostr(HiBit - LoBit) + ", " +
                     itostr(LoBit) + ")";
        Case += "      Value.insertBits(" + ExtractStr + ", " +
                itostr(LoInstBit) + ");\n";
      } else {
        ExtractStr = "op.extractBitsAsZExtValue(" + itostr(HiBit - LoBit) +
                     ", " + itostr(LoBit) + ")";
        Case += "      Value.insertBits(" + ExtractStr + ", " +
                itostr(LoInstBit) + ", " + itostr(HiBit - LoBit) + ");\n";
      }
    } else {
      uint64_t OpMask = ~(uint64_t)0 >> (64 - N);
      OpShift = BeginVarBit - N + 1;
      OpMask <<= OpShift;
      MaskStr = "UINT64_C(" + utostr(OpMask) + ")";
      OpShift = BeginInstBit - BeginVarBit;

      if (NumOperandLits == 1) {
        Case += "      op &= " + MaskStr + ";\n";
        if (OpShift > 0) {
          Case += "      op <<= " + itostr(OpShift) + ";\n";
        } else if (OpShift < 0) {
          Case += "      op >>= " + itostr(-OpShift) + ";\n";
        }
        Case += "      Value |= op;\n";
      } else {
        if (OpShift > 0) {
          Case += "      Value |= (op & " + MaskStr + ") << " +
                  itostr(OpShift) + ";\n";
        } else if (OpShift < 0) {
          Case += "      Value |= (op & " + MaskStr + ") >> " +
                  itostr(-OpShift) + ";\n";
        } else {
          Case += "      Value |= (op & " + MaskStr + ");\n";
        }
      }
    }
  }

  if (BitOffset != (unsigned)-1) {
    BitOffsetCase += "      case " + utostr(OpIdx) + ":\n";
    BitOffsetCase += "        // op: " + VarName + "\n";
    BitOffsetCase += "        return " + utostr(BitOffset) + ";\n";
  }

  return true;
}

std::pair<std::string, std::string>
CodeEmitterGen::getInstructionCases(const Record *R,
                                    const CodeGenTarget &Target) {
  std::string Case, BitOffsetCase;

  auto Append = [&](const std::string &S) {
    Case += S;
    BitOffsetCase += S;
  };

  if (const Record *RV = R->getValueAsOptionalDef("EncodingInfos")) {
    const CodeGenHwModes &HWM = Target.getHwModes();
    EncodingInfoByHwMode EBM(RV, HWM);

    // Invoke the interface to obtain the HwMode ID controlling the
    // EncodingInfo for the current subtarget. This interface will
    // mask off irrelevant HwMode IDs.
    Append("      unsigned HwMode = "
           "STI.getHwMode(MCSubtargetInfo::HwMode_EncodingInfo);\n");
    Case += "      switch (HwMode) {\n";
    Case += "      default: llvm_unreachable(\"Unknown hardware mode!\"); "
            "break;\n";
    for (auto &[ModeId, Encoding] : EBM) {
      if (ModeId == DefaultMode) {
        Case +=
            "      case " + itostr(DefaultMode) + ": InstBitsByHw = InstBits";
      } else {
        Case += "      case " + itostr(ModeId) + ": InstBitsByHw = InstBits_" +
                HWM.getMode(ModeId).Name.str();
      }
      Case += "; break;\n";
    }
    Case += "      };\n";

    // We need to remodify the 'Inst' value from the table we found above.
    if (UseAPInt) {
      int NumWords = APInt::getNumWords(BitWidth);
      Case += "      Inst = APInt(" + itostr(BitWidth);
      Case += ", ArrayRef(InstBitsByHw + opcode * " + itostr(NumWords) + ", " +
              itostr(NumWords);
      Case += "));\n";
      Case += "      Value = Inst;\n";
    } else {
      Case += "      Value = InstBitsByHw[opcode];\n";
    }

    Append("      switch (HwMode) {\n");
    Append("      default: llvm_unreachable(\"Unhandled HwMode\");\n");
    for (auto &[ModeId, Encoding] : EBM) {
      Append("      case " + itostr(ModeId) + ": {\n");
      addInstructionCasesForEncoding(R, Encoding, Target, Case, BitOffsetCase);
      Append("      break;\n");
      Append("      }\n");
    }
    Append("      }\n");
    return {std::move(Case), std::move(BitOffsetCase)};
  }
  addInstructionCasesForEncoding(R, R, Target, Case, BitOffsetCase);
  return {std::move(Case), std::move(BitOffsetCase)};
}

void CodeEmitterGen::addInstructionCasesForEncoding(
    const Record *R, const Record *EncodingDef, const CodeGenTarget &Target,
    std::string &Case, std::string &BitOffsetCase) {
  const BitsInit *BI = EncodingDef->getValueAsBitsInit("Inst");

  // Loop over all of the fields in the instruction, determining which are the
  // operands to the instruction.
  bool Success = true;
  size_t OrigBitOffsetCaseSize = BitOffsetCase.size();
  BitOffsetCase += "      switch (OpNum) {\n";
  size_t BitOffsetCaseSizeBeforeLoop = BitOffsetCase.size();
  for (const RecordVal &RV : EncodingDef->getValues()) {
    // Ignore fixed fields in the record, we're looking for values like:
    //    bits<5> RST = { ?, ?, ?, ?, ? };
    if (RV.isNonconcreteOK() || RV.getValue()->isComplete())
      continue;

    Success &= addCodeToMergeInOperand(R, BI, RV.getName().str(), Case,
                                       BitOffsetCase, Target);
  }
  // Avoid empty switches.
  if (BitOffsetCase.size() == BitOffsetCaseSizeBeforeLoop)
    BitOffsetCase.resize(OrigBitOffsetCaseSize);
  else
    BitOffsetCase += "      }\n";

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
}

static void emitInstBits(raw_ostream &OS, const APInt &Bits) {
  for (unsigned I = 0; I < Bits.getNumWords(); ++I)
    OS << ((I > 0) ? ", " : "") << "UINT64_C(" << utostr(Bits.getRawData()[I])
       << ")";
}

void CodeEmitterGen::emitInstructionBaseValues(
    raw_ostream &O, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
    const CodeGenTarget &Target, unsigned HwMode) {
  const CodeGenHwModes &HWM = Target.getHwModes();
  if (HwMode == DefaultMode)
    O << "  static const uint64_t InstBits[] = {\n";
  else
    O << "  static const uint64_t InstBits_"
      << HWM.getModeName(HwMode, /*IncludeDefault=*/true) << "[] = {\n";

  for (const CodeGenInstruction *CGI : NumberedInstructions) {
    const Record *R = CGI->TheDef;

    if (R->getValueAsString("Namespace") == "TargetOpcode" ||
        R->getValueAsBit("isPseudo")) {
      O << "    ";
      emitInstBits(O, APInt(BitWidth, 0));
      O << ",\n";
      continue;
    }

    const Record *EncodingDef = R;
    if (const Record *RV = R->getValueAsOptionalDef("EncodingInfos")) {
      EncodingInfoByHwMode EBM(RV, HWM);
      if (EBM.hasMode(HwMode)) {
        EncodingDef = EBM.get(HwMode);
      } else {
        // If the HwMode does not match, then Encoding '0'
        // should be generated.
        APInt Value(BitWidth, 0);
        O << "    ";
        emitInstBits(O, Value);
        O << "," << '\t' << "// " << R->getName() << "\n";
        continue;
      }
    }
    const BitsInit *BI = EncodingDef->getValueAsBitsInit("Inst");

    // Start by filling in fixed values.
    APInt Value(BitWidth, 0);
    for (unsigned I = 0, E = BI->getNumBits(); I != E; ++I) {
      if (const auto *B = dyn_cast<BitInit>(BI->getBit(I)); B && B->getValue())
        Value.setBit(I);
    }
    O << "    ";
    emitInstBits(O, Value);
    O << "," << '\t' << "// " << R->getName() << "\n";
  }
  O << "  };\n";
}

void CodeEmitterGen::emitCaseMap(
    raw_ostream &O,
    const std::map<std::string, std::vector<std::string>> &CaseMap) {
  for (const auto &[Case, InstList] : CaseMap) {
    bool First = true;
    for (const auto &Inst : InstList) {
      if (!First)
        O << "\n";
      O << "    case " << Inst << ":";
      First = false;
    }
    O << " {\n";
    O << Case;
    O << "      break;\n"
      << "    }\n";
  }
}

void CodeEmitterGen::run(raw_ostream &O) {
  emitSourceFileHeader("Machine Code Emitter", O);

  CodeGenTarget Target(Records);

  // For little-endian instruction bit encodings, reverse the bit order
  Target.reverseBitsForLittleEndianEncoding();

  ArrayRef<const CodeGenInstruction *> NumberedInstructions =
      Target.getInstructions();

  if (Target.hasVariableLengthEncodings()) {
    emitVarLenCodeEmitter(Records, O);
  } else {
    const CodeGenHwModes &HWM = Target.getHwModes();
    // The set of HwModes used by instruction encodings.
    std::set<unsigned> HwModes;
    BitWidth = 0;
    for (const CodeGenInstruction *CGI : NumberedInstructions) {
      const Record *R = CGI->TheDef;
      if (R->getValueAsString("Namespace") == "TargetOpcode" ||
          R->getValueAsBit("isPseudo"))
        continue;

      if (const Record *RV = R->getValueAsOptionalDef("EncodingInfos")) {
        EncodingInfoByHwMode EBM(RV, HWM);
        for (const auto &[Key, Value] : EBM) {
          const BitsInit *BI = Value->getValueAsBitsInit("Inst");
          BitWidth = std::max(BitWidth, BI->getNumBits());
          HwModes.insert(Key);
        }
        continue;
      }
      const BitsInit *BI = R->getValueAsBitsInit("Inst");
      BitWidth = std::max(BitWidth, BI->getNumBits());
    }
    UseAPInt = BitWidth > 64;

    // Emit function declaration
    if (UseAPInt) {
      O << "void " << Target.getName()
        << "MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,\n"
        << "    SmallVectorImpl<MCFixup> &Fixups,\n"
        << "    APInt &Inst,\n"
        << "    APInt &Scratch,\n"
        << "    const MCSubtargetInfo &STI) const {\n";
    } else {
      O << "uint64_t " << Target.getName();
      O << "MCCodeEmitter::getBinaryCodeForInstr(const MCInst &MI,\n"
        << "    SmallVectorImpl<MCFixup> &Fixups,\n"
        << "    const MCSubtargetInfo &STI) const {\n";
    }

    // Emit instruction base values
    emitInstructionBaseValues(O, NumberedInstructions, Target, DefaultMode);
    if (!HwModes.empty()) {
      // Emit table for instrs whose encodings are controlled by HwModes.
      for (unsigned HwMode : HwModes) {
        if (HwMode == DefaultMode)
          continue;
        emitInstructionBaseValues(O, NumberedInstructions, Target, HwMode);
      }

      // This pointer will be assigned to the HwMode table later.
      O << "  const uint64_t *InstBitsByHw;\n";
    }

    // Map to accumulate all the cases.
    std::map<std::string, std::vector<std::string>> CaseMap;
    std::map<std::string, std::vector<std::string>> BitOffsetCaseMap;

    // Construct all cases statement for each opcode
    for (const Record *R : Records.getAllDerivedDefinitions("Instruction")) {
      if (R->getValueAsString("Namespace") == "TargetOpcode" ||
          R->getValueAsBit("isPseudo"))
        continue;
      std::string InstName =
          (R->getValueAsString("Namespace") + "::" + R->getName()).str();
      std::string Case, BitOffsetCase;
      std::tie(Case, BitOffsetCase) = getInstructionCases(R, Target);

      CaseMap[Case].push_back(InstName);
      BitOffsetCaseMap[BitOffsetCase].push_back(std::move(InstName));
    }

    // Emit initial function code
    if (UseAPInt) {
      int NumWords = APInt::getNumWords(BitWidth);
      O << "  const unsigned opcode = MI.getOpcode();\n"
        << "  if (Scratch.getBitWidth() != " << BitWidth << ")\n"
        << "    Scratch = Scratch.zext(" << BitWidth << ");\n"
        << "  Inst = APInt(" << BitWidth << ", ArrayRef(InstBits + opcode * "
        << NumWords << ", " << NumWords << "));\n"
        << "  APInt &Value = Inst;\n"
        << "  APInt &op = Scratch;\n"
        << "  switch (opcode) {\n";
    } else {
      O << "  const unsigned opcode = MI.getOpcode();\n"
        << "  uint64_t Value = InstBits[opcode];\n"
        << "  uint64_t op = 0;\n"
        << "  (void)op;  // suppress warning\n"
        << "  switch (opcode) {\n";
    }

    // Emit each case statement
    emitCaseMap(O, CaseMap);

    // Default case: unhandled opcode
    O << "  default:\n"
      << "    std::string msg;\n"
      << "    raw_string_ostream Msg(msg);\n"
      << "    Msg << \"Not supported instr: \" << MI;\n"
      << "    report_fatal_error(Msg.str().c_str());\n"
      << "  }\n";
    if (UseAPInt)
      O << "  Inst = Value;\n";
    else
      O << "  return Value;\n";
    O << "}\n\n";

    O << "#ifdef GET_OPERAND_BIT_OFFSET\n"
      << "#undef GET_OPERAND_BIT_OFFSET\n\n"
      << "uint32_t " << Target.getName()
      << "MCCodeEmitter::getOperandBitOffset(const MCInst &MI,\n"
      << "    unsigned OpNum,\n"
      << "    const MCSubtargetInfo &STI) const {\n"
      << "  switch (MI.getOpcode()) {\n";
    emitCaseMap(O, BitOffsetCaseMap);
    O << "  }\n"
      << "  std::string msg;\n"
      << "  raw_string_ostream Msg(msg);\n"
      << "  Msg << \"Not supported instr[opcode]: \" << MI << \"[\" << OpNum "
         "<< \"]\";\n"
      << "  report_fatal_error(Msg.str().c_str());\n"
      << "}\n\n"
      << "#endif // GET_OPERAND_BIT_OFFSET\n\n";
  }
}

} // end anonymous namespace

static TableGen::Emitter::OptClass<CodeEmitterGen>
    X("gen-emitter", "Generate machine code emitter");
