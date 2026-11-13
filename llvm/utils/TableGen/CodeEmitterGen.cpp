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
#include "Common/InstructionEncoding.h"
#include "Common/VarLenCodeEmitterGen.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/CodeGenHelpers.h"
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

struct BaseEncodingPool {
  std::vector<APInt> Values;
  std::map<unsigned, std::vector<unsigned>> IndicesByHwMode;
  unsigned IndexBitWidth = 0;

  bool empty() const { return Values.empty(); }
};

class CodeEmitterGen {
  const RecordKeeper &RK;
  CodeGenTarget Target;
  const CodeGenHwModes &CGH;

public:
  explicit CodeEmitterGen(const RecordKeeper &RK);

  void run(raw_ostream &O);

private:
  const InstructionEncoding &addEncoding(const Record *EncodingDef,
                                         const CodeGenInstruction *CGI);
  const InstructionEncoding &getEncoding(const Record *EncodingDef,
                                         const CodeGenInstruction *CGI);
  std::pair<std::string, std::string> getInstructionCases(const Record *R);
  void addInstructionCasesForEncoding(const Record *R,
                                      const Record *EncodingDef,
                                      std::string &Case,
                                      std::string &BitOffsetCase);
  bool addCodeToMergeInOperand(const InstructionEncoding &Encoding,
                               const std::string &OperandName,
                               std::string &Case, std::string &BitOffsetCase);

  APInt getInstructionBaseValue(const CodeGenInstruction *CGI, unsigned HwMode);
  void buildBaseEncodingPool(
      ArrayRef<const CodeGenInstruction *> NumberedInstructions,
      const std::set<unsigned> &HwModes);
  void emitBaseEncodingPool(raw_ostream &O);
  void emitInstructionBaseValues(
      raw_ostream &O, ArrayRef<const CodeGenInstruction *> NumberedInstructions,
      unsigned HwMode = DefaultMode);
  /// Parsed encodings, keyed on the encoding record and the instruction it
  /// belongs to. Both are needed: an InstructionEncoding records its
  /// CodeGenInstruction and parses the operands from it, so instructions
  /// sharing an encoding record must not share a parsed encoding.
  DenseMap<std::pair<const Record *, const CodeGenInstruction *>,
           std::unique_ptr<InstructionEncoding>>
      EncodingCache;
  BaseEncodingPool BaseEncodings;
  unsigned BitWidth = 0u;
  bool UseAPInt = false;
};

} // end anonymous namespace

const InstructionEncoding &
CodeEmitterGen::addEncoding(const Record *EncodingDef,
                            const CodeGenInstruction *CGI) {
  auto [It, Inserted] =
      EncodingCache.try_emplace(std::make_pair(EncodingDef, CGI));
  // Two HwModes may legitimately select the same encoding record, in which
  // case both share the one parsed encoding.
  if (Inserted)
    It->second = std::make_unique<InstructionEncoding>(Target, EncodingDef, CGI,
                                                       EncodingUse::Encoder);
  return *It->second;
}

const InstructionEncoding &
CodeEmitterGen::getEncoding(const Record *EncodingDef,
                            const CodeGenInstruction *CGI) {
  auto It = EncodingCache.find(std::make_pair(EncodingDef, CGI));
  assert(It != EncodingCache.end() && "encoding not parsed by run()");
  return *It->second;
}

// Returns true if it succeeds, false if an error.
bool CodeEmitterGen::addCodeToMergeInOperand(
    const InstructionEncoding &Encoding, const std::string &OperandName,
    std::string &Case, std::string &BitOffsetCase) {
  const CodeGenInstruction &CGI = *Encoding.getInstruction();
  const Record *R = CGI.TheDef;

  // If the operand contributes no bits to the encoding, ignore it.
  std::optional<OperandInfo> OpInfo =
      Encoding.getOperandInfoByName(OperandName);
  if (!OpInfo || OpInfo->Fields.empty())
    return true;

  // If the operand matches by name, reference according to that
  // operand number. Non-matching operands are assumed to be in
  // order.
  unsigned OpIdx;
  if (auto SubOp = CGI.Operands.findSubOperandAlias(OperandName)) {
    OpIdx = CGI.Operands[SubOp->first].MIOperandNo + SubOp->second;
  } else if (auto MayBeOpIdx = CGI.Operands.findOperandNamed(OperandName)) {
    // Get the machine operand number for the indicated operand.
    OpIdx = CGI.Operands[*MayBeOpIdx].MIOperandNo;
  } else {
    PrintError(R, Twine("No operand named ") + OperandName + " in record " +
                      R->getName());
    return false;
  }

  std::pair<unsigned, unsigned> SO = CGI.Operands.getSubOperandNumber(OpIdx);
  StringRef EncoderMethodName =
      CGI.Operands[SO.first].EncoderMethodNames[SO.second];

  raw_string_ostream OS(Case);
  indent Indent(6);

  OS << Indent << "// op: " << OperandName << '\n';

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

  // Segments are ordered by ascending Base. Emit them in reverse so that the
  // resulting statements match the historical highest-run-first order.
  for (const EncodingField &F : llvm::reverse(OpInfo->Fields)) {
    if (UseAPInt) {
      if (F.Width > 64)
        OS << Indent << "Value.insertBits(op.extractBits(" << F.Width << ", "
           << F.Offset << "), " << F.Base << ");\n";
      else
        OS << Indent << "Value.insertBits(op.extractBitsAsZExtValue(" << F.Width
           << ", " << F.Offset << "), " << F.Base << ", " << F.Width << ");\n";
    } else {
      uint64_t OpMask = maskTrailingOnes<uint64_t>(F.Width) << F.Offset;
      OS << Indent << "Value |= (op & " << format_hex(OpMask, 0) << ')';
      if (F.Base > F.Offset)
        OS << " << " << (F.Base - F.Offset);
      else if (F.Base < F.Offset)
        OS << " >> " << (F.Offset - F.Base);
      OS << ";\n";
    }
  }

  // The lowest bit of the operand is the one with the smallest Base, which is
  // the first segment in ascending order.
  BitOffsetCase += "      case " + utostr(OpIdx) + ":\n";
  BitOffsetCase += "        // op: " + OperandName + "\n";
  BitOffsetCase +=
      "        return " + utostr(OpInfo->Fields.front().Base) + ";\n";

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
    std::string InstBitsTableName =
        BaseEncodings.empty() ? "InstBits" : "InstBitsIndices";
    for (auto &[ModeId, Encoding] : EBM) {
      if (ModeId == DefaultMode) {
        Case += "      case " + itostr(DefaultMode) + ": " + InstBitsTableName +
                "ByHw = " + InstBitsTableName;
      } else {
        Case += "      case " + itostr(ModeId) + ": " + InstBitsTableName +
                "ByHw = " + InstBitsTableName + "_" +
                CGH.getModeName(ModeId).str();
      }
      Case += "; break;\n";
    }
    Case += "      };\n";

    // Reload Inst from the selected HwMode table.
    if (UseAPInt) {
      if (BaseEncodings.empty()) {
        int NumWords = APInt::getNumWords(BitWidth);
        Case += "      Inst = APInt(" + itostr(BitWidth);
        Case += ", ArrayRef(InstBitsByHw + TableIndex * " + itostr(NumWords) +
                ", " + itostr(NumWords);
        Case += "));\n";
      } else {
        Case += "      InstBitsIndex = InstBitsIndicesByHw[TableIndex];\n";
        Case += "      Inst = APInt(" + itostr(BitWidth) +
                ", ArrayRef(InstBits[InstBitsIndex]));\n";
      }
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
  // TODO: Iterating the encoding's operands (Encoding.getOperands()) instead
  // of the encoding record's field declarations would remove the need for
  // getOperandInfoByName and for the operand name lookup in
  // addCodeToMergeInOperand, if OperandInfo also carried the machine operand
  // number and the encoder method name. Two things come first:
  //  - Tied operands are not de-duplicated in parseFixedLenOperands, which
  //    populates both halves of a tie with the same segments, so iterating
  //    operands would emit the operand fetch (and any fixup) twice.
  //  - The "no operand named X" diagnostic in addCodeToMergeInOperand is
  //    reported once per encoding record declaration; an operand-order loop
  //    would no longer see a declaration that names no operand.
  const InstructionEncoding &Encoding =
      getEncoding(EncodingDef, &Target.getInstruction(R));

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

    Success &= addCodeToMergeInOperand(Encoding, RV.getName().str(), Case,
                                       BitOffsetCase);
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

APInt CodeEmitterGen::getInstructionBaseValue(const CodeGenInstruction *CGI,
                                              unsigned HwMode) {
  const Record *R = CGI->TheDef;
  const Record *EncodingDef = R;
  if (const Record *RV = R->getValueAsOptionalDef("EncodingInfos")) {
    EncodingInfoByHwMode EBM(RV, CGH);
    if (!EBM.hasMode(HwMode))
      return APInt(BitWidth, 0);
    EncodingDef = EBM.get(HwMode);
  }

  // The encoder's width is the width of `Inst`, so the known bits of the
  // encoding are already the right width.
  const InstructionEncoding &Encoding = getEncoding(EncodingDef, CGI);
  assert(Encoding.getBitWidth() <= BitWidth);
  return Encoding.getInstBits().One.zextOrTrunc(BitWidth);
}

void CodeEmitterGen::buildBaseEncodingPool(
    ArrayRef<const CodeGenInstruction *> NumberedInstructions,
    const std::set<unsigned> &HwModes) {
  assert(UseAPInt && "pooling is only used for multiword encodings");

  const unsigned NumWords = APInt::getNumWords(BitWidth);
  DenseMap<APInt, unsigned> ValueToIndex;
  BaseEncodings.IndicesByHwMode.try_emplace(DefaultMode);
  for (unsigned HwMode : HwModes)
    BaseEncodings.IndicesByHwMode.try_emplace(HwMode);

  for (auto &[HwMode, Indices] : BaseEncodings.IndicesByHwMode) {
    for (const CodeGenInstruction *CGI : NumberedInstructions) {
      APInt Value = getInstructionBaseValue(CGI, HwMode);
      auto [It, Inserted] =
          ValueToIndex.try_emplace(Value, BaseEncodings.Values.size());
      if (Inserted)
        BaseEncodings.Values.push_back(std::move(Value));
      Indices.push_back(It->second);
    }
  }

  const uint64_t NumValues = BaseEncodings.Values.size();
  BaseEncodings.IndexBitWidth =
      std::max<uint64_t>(8, PowerOf2Ceil(Log2_64_Ceil(NumValues)));
  if (BaseEncodings.IndexBitWidth > 32) {
    BaseEncodings = {};
    return;
  }

  const uint64_t NumTables = BaseEncodings.IndicesByHwMode.size();
  const uint64_t RawSize =
      NumTables * NumberedInstructions.size() * NumWords * sizeof(uint64_t);
  const uint64_t PooledSize = NumValues * NumWords * sizeof(uint64_t) +
                              NumTables * NumberedInstructions.size() *
                                  (BaseEncodings.IndexBitWidth / 8);
  if (PooledSize >= RawSize)
    BaseEncodings = {};
}

void CodeEmitterGen::emitBaseEncodingPool(raw_ostream &O) {
  const unsigned NumWords = APInt::getNumWords(BitWidth);
  O << "  static const uint64_t InstBits[][" << NumWords << "] = {\n";
  for (const APInt &Value : BaseEncodings.Values) {
    O << "    {";
    emitInstBits(O, Value);
    O << "},\n";
  }
  O << "  };\n";

  for (const auto &[HwMode, Indices] : BaseEncodings.IndicesByHwMode) {
    O << formatv("  static const uint{}_t InstBitsIndices",
                 BaseEncodings.IndexBitWidth);
    if (HwMode != DefaultMode)
      O << "_" << CGH.getModeName(HwMode);
    O << "[] = {\n";
    for (unsigned Index : Indices)
      O << "    " << Index << ",\n";
    O << "  };\n";
  }
  O << "  static_assert(sizeof(InstBits) / sizeof(InstBits[0]) <=\n"
       "                (UINT64_C(1) << (sizeof(InstBitsIndices[0]) * 8)));\n";
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
    APInt Value = getInstructionBaseValue(CGI, HwMode);
    O << "    ";
    emitInstBits(O, Value);
    O << "," << '\t' << "// " << R->getName() << "\n";
  }
  O << "  };\n";
}

CodeEmitterGen::CodeEmitterGen(const RecordKeeper &RK)
    : RK(RK), Target(RK), CGH(Target.getHwModes()) {}

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
        BitWidth = std::max(BitWidth, addEncoding(Value, CGI).getBitWidth());
        HwModes.insert(Key);
      }
      continue;
    }
    BitWidth = std::max(BitWidth, addEncoding(R, CGI).getBitWidth());
  }
  UseAPInt = BitWidth > 64;
  if (UseAPInt)
    buildBaseEncodingPool(EncodedInstructions, HwModes);

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
  if (BaseEncodings.empty()) {
    emitInstructionBaseValues(O, EncodedInstructions, DefaultMode);
    for (unsigned HwMode : HwModes) {
      if (HwMode == DefaultMode)
        continue;
      emitInstructionBaseValues(O, EncodedInstructions, HwMode);
    }
  } else {
    emitBaseEncodingPool(O);
  }

  if (!HwModes.empty()) {
    // This pointer will be assigned to the HwMode table later.
    if (BaseEncodings.empty())
      O << "  const uint64_t *InstBitsByHw;\n";
    else
      O << formatv("  const uint{0}_t *InstBitsIndicesByHw;\n",
                   BaseEncodings.IndexBitWidth);
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
      << "    Scratch = Scratch.zext(" << BitWidth << ");\n";
    if (BaseEncodings.empty()) {
      O << "  Inst = APInt(" << BitWidth
        << ", ArrayRef(InstBits + TableIndex * " << NumWords << ", " << NumWords
        << "));\n";
    } else {
      O << "  unsigned InstBitsIndex = InstBitsIndices[TableIndex];\n"
        << "  Inst = APInt(" << BitWidth
        << ", ArrayRef(InstBits[InstBitsIndex]));\n";
    }
    O << "  APInt &Value = Inst;\n"
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

  IfDefEmitter IfDef(O, "GET_OPERAND_BIT_OFFSET");
  O << "uint32_t " << Target.getName()
    << "MCCodeEmitter::getOperandBitOffset(const MCInst &MI,\n"
    << "    unsigned OpNum,\n"
    << "    const MCSubtargetInfo &STI) const {\n"
    << "  switch (MI.getOpcode()) {\n";
  emitCaseMap(O, BitOffsetCaseMap, PrintInstName);
  O << "  default:\n"
    << "    reportUnsupportedInst(MI);\n"
    << "  }\n"
    << "  reportUnsupportedOperand(MI, OpNum);\n"
    << "}\n";
}

static TableGen::Emitter::OptClass<CodeEmitterGen>
    X("gen-emitter", "Generate machine code emitter");
