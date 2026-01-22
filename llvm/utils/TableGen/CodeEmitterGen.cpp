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
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
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

// A map of uniqued case statements. The key is the body of the case statement
// and the value is a list of cases which share the same body.
using CaseMapT = std::map<std::string, std::vector<unsigned>>;

class CodeEmitterGen {
  const RecordKeeper &RK;
  CodeGenTarget Target;
  const CodeGenHwModes &CGH;

public:
  explicit CodeEmitterGen(const RecordKeeper &RK);

  void run(raw_ostream &O);

private:
  int getVariableBit(const std::string &VarName, const BitsInit *BI, int Bit);
  std::pair<std::string, std::string> getInstructionCases(const Record *R);
  void addInstructionCasesForEncoding(const Record *R,
                                      const Record *EncodingDef,
                                      std::string &Case,
                                      std::string &BitOffsetCase);
  bool addCodeToMergeInOperand(const Record *R, const BitsInit *BI,
                               const std::string &VarName, std::string &Case,
                               std::string &BitOffsetCase);

  void emitInstructionBaseValues(
      raw_ostream &O, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
      unsigned HwMode = DefaultMode);
  unsigned BitWidth = 0u;
  bool UseAPInt = false;
};

} // end anonymous namespace

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
                                             std::string &BitOffsetCase) {
  const CodeGenInstruction &CGI = Target.getInstruction(R);

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

  std::pair<unsigned, unsigned> SO = CGI.Operands.getSubOperandNumber(OpIdx);
  StringRef EncoderMethodName =
      CGI.Operands[SO.first].EncoderMethodNames[SO.second];

  raw_string_ostream OS(Case);
  indent Indent(6);

  OS << Indent << "// op: " << VarName << '\n';

  if (UseAPInt)
    OS << Indent << "op.clearAllBits();\n";

  if (!EncoderMethodName.empty()) {
    if (UseAPInt)
      OS << Indent << EncoderMethodName << "(MI, " << OpIdx
         << ", op, Fixups, STI);\n";
    else
      OS << Indent << "op = " << EncoderMethodName << "(MI, " << OpIdx
         << ", Fixups, STI);\n";
  } else {
    if (UseAPInt)
      OS << Indent << "getMachineOpValue(MI, MI.getOperand(" << OpIdx
         << "), op, Fixups, STI);\n";
    else
      OS << Indent << "op = getMachineOpValue(MI, MI.getOperand(" << OpIdx
         << "), Fixups, STI);\n";
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

    unsigned LoBit = BeginVarBit - N + 1;
    unsigned LoInstBit = BeginInstBit - N + 1;
    BitOffset = LoInstBit;
    if (UseAPInt) {
      if (N > 64)
        OS << Indent << "Value.insertBits(op.extractBits(" << N << ", " << LoBit
           << "), " << LoInstBit << ");\n";
      else
        OS << Indent << "Value.insertBits(op.extractBitsAsZExtValue(" << N
           << ", " << LoBit << "), " << LoInstBit << ", " << N << ");\n";
    } else {
      uint64_t OpMask = maskTrailingOnes<uint64_t>(N) << LoBit;
      OS << Indent << "Value |= (op & " << format_hex(OpMask, 0) << ')';
      int OpShift = BeginInstBit - BeginVarBit;
      if (OpShift > 0)
        OS << " << " << OpShift;
      else if (OpShift < 0)
        OS << " >> " << -OpShift;
      OS << ";\n";
    }
  }

  if (BitOffset != (unsigned)-1) {
    BitOffsetCase += "      case " + utostr(OpIdx) + ":\n";
    BitOffsetCase += "        // op: " + VarName + "\n";
    BitOffsetCase += "        return " + utostr(BitOffset) + ";\n";
  }

  return true;
}

static void emitCaseMap(raw_ostream &O, const CaseMapT &CaseMap,
                        function_ref<void(raw_ostream &, unsigned)> PrintCase) {
  for (const auto &[CaseBody, Cases] : CaseMap) {
    ListSeparator LS("\n");
    for (unsigned Case : Cases) {
      O << LS << "    case ";
      PrintCase(O, Case);
      O << ":";
    }
    O << " {\n";
    O << CaseBody;
    O << "      break;\n"
      << "    }\n";
  }
}

std::pair<std::string, std::string>
CodeEmitterGen::getInstructionCases(const Record *R) {
  std::string Case, BitOffsetCase;

  auto Append = [&](const std::string &S) {
    Case += S;
    BitOffsetCase += S;
  };

  if (const Record *RV = R->getValueAsOptionalDef("EncodingInfos")) {
    EncodingInfoByHwMode EBM(RV, CGH);

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
                CGH.getMode(ModeId).Name.str();
      }
      Case += "; break;\n";
    }
    Case += "      };\n";

    // We need to remodify the 'Inst' value from the table we found above.
    if (UseAPInt) {
      int NumWords = APInt::getNumWords(BitWidth);
      Case += "      Inst = APInt(" + itostr(BitWidth);
      Case += ", ArrayRef(InstBitsByHw + TableIndex * " + itostr(NumWords) +
              ", " + itostr(NumWords);
      Case += "));\n";
      Case += "      Value = Inst;\n";
    } else {
      Case += "      Value = InstBitsByHw[TableIndex];\n";
    }

    Append("      switch (HwMode) {\n");
    Append("      default: llvm_unreachable(\"Unhandled HwMode\");\n");

    // Attempt to unique the per-hw-mode encoding case statements. This helps
    // reduce the code size if 2 or more hw-modes share the same encoding for
    // the fields of the instruction.
    CaseMapT CaseMap, BitOffsetCaseMap;
    std::string ModeCase, ModeBitOffsetCase;

    auto PrintHWMode = [](raw_ostream &O, unsigned Mode) { O << Mode; };

    for (auto &[ModeId, Encoding] : EBM) {
      ModeCase.clear();
      ModeBitOffsetCase.clear();
      addInstructionCasesForEncoding(R, Encoding, ModeCase, ModeBitOffsetCase);
      CaseMap[ModeCase].push_back(ModeId);
      BitOffsetCaseMap[ModeBitOffsetCase].push_back(ModeId);
    }

    raw_string_ostream CaseOS(Case);
    raw_string_ostream BitOffsetCaseOS(BitOffsetCase);
    emitCaseMap(CaseOS, CaseMap, PrintHWMode);
    emitCaseMap(BitOffsetCaseOS, BitOffsetCaseMap, PrintHWMode);

    Append("      }\n");
    return {std::move(Case), std::move(BitOffsetCase)};
  }
  addInstructionCasesForEncoding(R, R, Case, BitOffsetCase);
  return {std::move(Case), std::move(BitOffsetCase)};
}

void CodeEmitterGen::addInstructionCasesForEncoding(
    const Record *R, const Record *EncodingDef, std::string &Case,
    std::string &BitOffsetCase) {
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

    Success &=
        addCodeToMergeInOperand(R, BI, RV.getName().str(), Case, BitOffsetCase);
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
    OS << ((I > 0) ? ", " : "") << "UINT64_C(" << Bits.getRawData()[I] << ")";
}

void CodeEmitterGen::emitInstructionBaseValues(
    raw_ostream &O, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
    unsigned HwMode) {
  if (HwMode == DefaultMode)
    O << "  static const uint64_t InstBits[] = {\n";
  else
    O << "  static const uint64_t InstBits_" << CGH.getModeName(HwMode)
      << "[] = {\n";

  for (const CodeGenInstruction *CGI : NumberedInstructions) {
    const Record *R = CGI->TheDef;
    const Record *EncodingDef = R;
    if (const Record *RV = R->getValueAsOptionalDef("EncodingInfos")) {
      EncodingInfoByHwMode EBM(RV, CGH);
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

CodeEmitterGen::CodeEmitterGen(const RecordKeeper &RK)
    : RK(RK), Target(RK), CGH(Target.getHwModes()) {
  // For little-endian instruction bit encodings, reverse the bit order.
  Target.reverseBitsForLittleEndianEncoding();
}

void CodeEmitterGen::run(raw_ostream &O) {
  emitSourceFileHeader("Machine Code Emitter", O);

  ArrayRef<const CodeGenInstruction *> EncodedInstructions =
      Target.getTargetNonPseudoInstructions();

  if (Target.hasVariableLengthEncodings()) {
    emitVarLenCodeEmitter(RK, O);
    return;
  }
  // The set of HwModes used by instruction encodings.
  std::set<unsigned> HwModes;
  BitWidth = 0;
  for (const CodeGenInstruction *CGI : EncodedInstructions) {
    const Record *R = CGI->TheDef;
    if (const Record *RV = R->getValueAsOptionalDef("EncodingInfos")) {
      EncodingInfoByHwMode EBM(RV, CGH);
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
  emitInstructionBaseValues(O, EncodedInstructions, DefaultMode);
  if (!HwModes.empty()) {
    // Emit table for instrs whose encodings are controlled by HwModes.
    for (unsigned HwMode : HwModes) {
      if (HwMode == DefaultMode)
        continue;
      emitInstructionBaseValues(O, EncodedInstructions, HwMode);
    }

    // This pointer will be assigned to the HwMode table later.
    O << "  const uint64_t *InstBitsByHw;\n";
  }

  // Map to accumulate all the cases.
  CaseMapT CaseMap, BitOffsetCaseMap;

  // Construct all cases statement for each opcode
  for (auto [Index, CGI] : enumerate(EncodedInstructions)) {
    const Record *R = CGI->TheDef;
    auto [Case, BitOffsetCase] = getInstructionCases(R);

    CaseMap[Case].push_back(Index);
    BitOffsetCaseMap[BitOffsetCase].push_back(Index);
  }

  auto PrintInstName = [&](raw_ostream &OS, unsigned Index) {
    const CodeGenInstruction *CGI = EncodedInstructions[Index];
    const Record *R = CGI->TheDef;
    OS << R->getValueAsString("Namespace") << "::" << R->getName();
  };

  unsigned FirstSupportedOpcode = EncodedInstructions.front()->EnumVal;
  O << "  constexpr unsigned FirstSupportedOpcode = " << FirstSupportedOpcode
    << ";\n";
  O << R"(
  const unsigned opcode = MI.getOpcode();
  if (opcode < FirstSupportedOpcode)
    reportUnsupportedInst(MI);
  unsigned TableIndex = opcode - FirstSupportedOpcode;
)";

  // Emit initial function code
  if (UseAPInt) {
    int NumWords = APInt::getNumWords(BitWidth);
    O << "  if (Scratch.getBitWidth() != " << BitWidth << ")\n"
      << "    Scratch = Scratch.zext(" << BitWidth << ");\n"
      << "  Inst = APInt(" << BitWidth << ", ArrayRef(InstBits + TableIndex * "
      << NumWords << ", " << NumWords << "));\n"
      << "  APInt &Value = Inst;\n"
      << "  APInt &op = Scratch;\n"
      << "  switch (opcode) {\n";
  } else {
    O << "  uint64_t Value = InstBits[TableIndex];\n"
      << "  uint64_t op = 0;\n"
      << "  (void)op;  // suppress warning\n"
      << "  switch (opcode) {\n";
  }

  // Emit each case statement
  emitCaseMap(O, CaseMap, PrintInstName);

  // Default case: unhandled opcode.
  O << "  default:\n"
    << "    reportUnsupportedInst(MI);\n"
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
  emitCaseMap(O, BitOffsetCaseMap, PrintInstName);
  O << "  default:\n"
    << "    reportUnsupportedInst(MI);\n"
    << "  }\n"
    << "  reportUnsupportedOperand(MI, OpNum);\n"
    << "}\n\n"
    << "#endif // GET_OPERAND_BIT_OFFSET\n\n";
}

static TableGen::Emitter::OptClass<CodeEmitterGen>
    X("gen-emitter", "Generate machine code emitter");
