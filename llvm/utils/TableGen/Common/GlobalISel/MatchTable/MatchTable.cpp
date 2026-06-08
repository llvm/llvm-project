//===- MatchTable.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MatchTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

#define DEBUG_TYPE "gi-match-table"

namespace llvm {
namespace gi {
// GIMT_Encode2/4/8
constexpr StringLiteral EncodeMacroName = "GIMT_Encode";

// Avoid adding specialized bytecode to small tables where the linked size
// reduction is negligible.
constexpr unsigned MinMatchTableSizeForCompaction = 64 * 1024;

//===- Helpers ------------------------------------------------------------===//

void emitEncodingMacrosDef(raw_ostream &OS) {
  OS << "#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__\n"
     << "#define " << EncodeMacroName << "2(Val)"
     << " uint8_t(Val), uint8_t((Val) >> 8)\n"
     << "#define " << EncodeMacroName << "4(Val)"
     << " uint8_t(Val), uint8_t((Val) >> 8), "
        "uint8_t((Val) >> 16), uint8_t((Val) >> 24)\n"
     << "#define " << EncodeMacroName << "8(Val)"
     << " uint8_t(Val), uint8_t((Val) >> 8), "
        "uint8_t((Val) >> 16), uint8_t((Val) >> 24),  "
        "uint8_t(uint64_t(Val) >> 32), uint8_t(uint64_t(Val) >> 40), "
        "uint8_t(uint64_t(Val) >> 48), uint8_t(uint64_t(Val) >> 56)\n"
     << "#else\n"
     << "#define " << EncodeMacroName << "2(Val)"
     << " uint8_t((Val) >> 8), uint8_t(Val)\n"
     << "#define " << EncodeMacroName << "4(Val)"
     << " uint8_t((Val) >> 24), uint8_t((Val) >> 16), "
        "uint8_t((Val) >> 8), uint8_t(Val)\n"
     << "#define " << EncodeMacroName << "8(Val)"
     << " uint8_t(uint64_t(Val) >> 56), uint8_t(uint64_t(Val) >> 48), "
        "uint8_t(uint64_t(Val) >> 40), uint8_t(uint64_t(Val) >> 32),  "
        "uint8_t((Val) >> 24), uint8_t((Val) >> 16), "
        "uint8_t((Val) >> 8), uint8_t(Val)\n"
     << "#endif\n";
}

void emitEncodingMacrosUndef(raw_ostream &OS) {
  OS << "#undef " << EncodeMacroName << "2\n"
     << "#undef " << EncodeMacroName << "4\n"
     << "#undef " << EncodeMacroName << "8\n";
}

std::string getNameForFeatureBitset(ArrayRef<const Record *> FeatureBitset,
                                    int HwModeIdx) {
  std::string Name = "GIFBS";
  for (const Record *Feature : FeatureBitset)
    Name += ("_" + Feature->getName()).str();
  if (HwModeIdx >= 0)
    Name += ("_HwMode" + std::to_string(HwModeIdx));
  return Name;
}

static std::string getEncodedEmitStr(StringRef NamedValue, unsigned NumBytes) {
  if (NumBytes == 2 || NumBytes == 4 || NumBytes == 8)
    return (EncodeMacroName + Twine(NumBytes) + "(" + NamedValue + ")").str();
  llvm_unreachable("Unsupported number of bytes!");
}

//===- MatchTableRecord ---------------------------------------------------===//

void MatchTableRecord::emit(raw_ostream &OS, bool LineBreakIsNextAfterThis,
                            const MatchTable &Table,
                            unsigned CurrentIndex) const {
  bool UseLineComment =
      LineBreakIsNextAfterThis || (Flags & MTRF_LineBreakFollows);
  if (Flags & (MTRF_JumpTarget | MTRF_CommaFollows))
    UseLineComment = false;

  if (Flags & MTRF_Comment)
    OS << (UseLineComment ? "// " : "/*");

  if (NumElements > 1 && !(Flags & (MTRF_PreEncoded | MTRF_Comment)))
    OS << getEncodedEmitStr(EmitStr, NumElements);
  else
    OS << EmitStr;

  if (Flags & MTRF_Label)
    OS << ": @" << Table.getLabelIndex(LabelID);

  if ((Flags & MTRF_Comment) && !UseLineComment)
    OS << "*/";

  if (Flags & MTRF_JumpTarget) {
    if (Flags & MTRF_Comment)
      OS << " ";
    unsigned Target = Table.getLabelIndex(LabelID);
    if (Flags & MTRF_RelativeJumpTarget) {
      assert(Target >= CurrentIndex + NumElements &&
             "Relative jumps must be forward");
      Target -= CurrentIndex + NumElements;
    }
    // TODO: Could encode this AOT to speed up build of generated file.
    if (NumElements == 1)
      OS << Target;
    else
      OS << getEncodedEmitStr(llvm::to_string(Target), NumElements);
  }

  if (Flags & MTRF_CommaFollows) {
    OS << ",";
    if (!LineBreakIsNextAfterThis && !(Flags & MTRF_LineBreakFollows))
      OS << " ";
  }

  if (Flags & MTRF_LineBreakFollows)
    OS << "\n";
}

//===- MatchTable ---------------------------------------------------------===//

MatchTableRecord MatchTable::LineBreak = {
    std::nullopt, "" /* Emit String */, 0 /* Elements */,
    MatchTableRecord::MTRF_LineBreakFollows};

MatchTableRecord MatchTable::Comment(StringRef Comment) {
  return MatchTableRecord(std::nullopt, Comment, 0,
                          MatchTableRecord::MTRF_Comment);
}

MatchTableRecord MatchTable::Opcode(StringRef Opcode, int IndentAdjust) {
  unsigned ExtraFlags = 0;
  if (IndentAdjust > 0)
    ExtraFlags |= MatchTableRecord::MTRF_Indent;
  if (IndentAdjust < 0)
    ExtraFlags |= MatchTableRecord::MTRF_Outdent;

  return MatchTableRecord(std::nullopt, Opcode, 1,
                          MatchTableRecord::MTRF_CommaFollows | ExtraFlags);
}

MatchTableRecord MatchTable::NamedValue(unsigned NumBytes,
                                        StringRef NamedValue) {
  return MatchTableRecord(std::nullopt, NamedValue, NumBytes,
                          MatchTableRecord::MTRF_CommaFollows);
}

MatchTableRecord MatchTable::NamedValue(unsigned NumBytes, StringRef Namespace,
                                        StringRef NamedValue) {
  return MatchTableRecord(std::nullopt, (Namespace + "::" + NamedValue).str(),
                          NumBytes, MatchTableRecord::MTRF_CommaFollows);
}

MatchTableRecord MatchTable::IntValue(unsigned NumBytes, int64_t IntValue) {
  assert(isUIntN(NumBytes * 8, IntValue) || isIntN(NumBytes * 8, IntValue));
  uint64_t UIntValue = IntValue;
  if (NumBytes < 8)
    UIntValue &= (UINT64_C(1) << NumBytes * 8) - 1;
  std::string Str = llvm::to_string(UIntValue);
  if (UIntValue > INT64_MAX)
    Str += 'u';
  // TODO: Could optimize this directly to save the compiler some work when
  // building the file
  return MatchTableRecord(std::nullopt, Str, NumBytes,
                          MatchTableRecord::MTRF_CommaFollows);
}

MatchTableRecord MatchTable::ULEB128Value(uint64_t IntValue) {
  uint8_t Buffer[10];
  unsigned Len = encodeULEB128(IntValue, Buffer);

  // Simple case (most common)
  if (Len == 1) {
    return MatchTableRecord(std::nullopt, llvm::to_string((unsigned)Buffer[0]),
                            1, MatchTableRecord::MTRF_CommaFollows);
  }

  // Print it as, e.g. /* -123456 (*/, 0xC0, 0xBB, 0x78 /*)*/
  std::string Str;
  raw_string_ostream OS(Str);
  OS << "/* " << llvm::to_string(IntValue) << "(*/";
  for (unsigned K = 0; K < Len; ++K) {
    if (K)
      OS << ", ";
    OS << "0x" << llvm::toHex({Buffer[K]});
  }
  OS << "/*)*/";
  return MatchTableRecord(std::nullopt, Str, Len,
                          MatchTableRecord::MTRF_CommaFollows |
                              MatchTableRecord::MTRF_PreEncoded);
}

MatchTableRecord MatchTable::Label(unsigned LabelID) {
  return MatchTableRecord(LabelID, "Label " + llvm::to_string(LabelID), 0,
                          MatchTableRecord::MTRF_Label |
                              MatchTableRecord::MTRF_Comment |
                              MatchTableRecord::MTRF_LineBreakFollows);
}

MatchTableRecord MatchTable::JumpTarget(unsigned LabelID) {
  return MatchTableRecord(LabelID, "Label " + llvm::to_string(LabelID), 4,
                          MatchTableRecord::MTRF_JumpTarget |
                              MatchTableRecord::MTRF_Comment |
                              MatchTableRecord::MTRF_CommaFollows);
}

void MatchTable::emitUse(raw_ostream &OS) const { OS << "MatchTable" << ID; }

void MatchTable::emitDeclaration(raw_ostream &OS) const {
  static constexpr unsigned BaseIndent = 4;
  unsigned Indentation = 0;
  OS << "  constexpr static uint8_t MatchTable" << ID << "[] = {";
  LineBreak.emit(OS, true, *this, 0);

  // We want to display the table index of each line in a consistent
  // manner. It has to appear as a column on the left side of the table.
  // To determine how wide the column needs to be, check how many characters
  // we need to fit the largest possible index in the current table.
  const unsigned NumColsForIdx = llvm::to_string(CurrentSize).size();

  unsigned CurIndex = 0;
  const auto BeginLine = [&]() {
    OS.indent(BaseIndent);
    std::string IdxStr = llvm::to_string(CurIndex);
    // Pad the string with spaces to keep the size of the prefix consistent.
    OS << " /* ";
    OS.indent(NumColsForIdx - IdxStr.size()) << IdxStr << " */ ";
    OS.indent(Indentation);
  };

  BeginLine();
  for (auto I = Contents.begin(), E = Contents.end(); I != E; ++I) {
    bool LineBreakIsNext = false;
    const auto &NextI = std::next(I);

    if (NextI != E) {
      if (NextI->EmitStr == "" &&
          NextI->Flags == MatchTableRecord::MTRF_LineBreakFollows)
        LineBreakIsNext = true;
    }

    if (I->Flags & MatchTableRecord::MTRF_Indent)
      Indentation += 2;

    I->emit(OS, LineBreakIsNext, *this, CurIndex);
    if (I->Flags & MatchTableRecord::MTRF_LineBreakFollows)
      BeginLine();

    if (I->Flags & MatchTableRecord::MTRF_Outdent)
      Indentation -= 2;

    CurIndex += I->size();
  }
  assert(CurIndex == CurrentSize);
  OS << "}; // Size: " << CurrentSize << " bytes\n";
}

void MatchTable::rebuildLabelMap() {
  LabelMap.clear();
  CurrentSize = 0;
  for (const MatchTableRecord &Record : Contents) {
    if (Record.Flags & MatchTableRecord::MTRF_Label)
      defineLabel(Record.LabelID);
    CurrentSize += Record.size();
  }
}

void MatchTable::compactFailureTargets() {
  if (CurrentSize < MinMatchTableSizeForCompaction)
    return;

  std::vector<unsigned> RecordOffsets;
  RecordOffsets.reserve(Contents.size());
  unsigned Offset = 0;
  for (const MatchTableRecord &Record : Contents) {
    RecordOffsets.push_back(Offset);
    Offset += Record.size();
  }

  for (unsigned I = 0, E = Contents.size(); I != E; ++I) {
    MatchTableRecord &Opcode = Contents[I];
    StringRef OpcodeName = Opcode.EmitStr;
    if (OpcodeName != "GIM_Try" && OpcodeName != "GIM_Try_CheckFeatures")
      continue;

    unsigned JumpRecord = I + 1;
    while (JumpRecord != E && Contents[JumpRecord].size() == 0)
      ++JumpRecord;
    assert(JumpRecord != E &&
           (Contents[JumpRecord].Flags & MatchTableRecord::MTRF_JumpTarget));

    MatchTableRecord &Jump = Contents[JumpRecord];
    unsigned Target = getLabelIndex(Jump.LabelID);
    unsigned JumpOffset = RecordOffsets[JumpRecord];
    assert(Target > JumpOffset && "GIM_Try targets must be forward");

    unsigned NumBytes;
    StringRef Suffix;
    unsigned Distance = Target - JumpOffset;
    if (Distance <= 256) {
      NumBytes = 1;
      Suffix = "8";
    } else if (Distance <= 65537) {
      NumBytes = 2;
      Suffix = "16";
    } else {
      continue;
    }

    Opcode.EmitStr += Suffix;
    Jump.makeRelativeJumpTarget(NumBytes);
  }

  rebuildLabelMap();
}

void MatchTable::compactRootOperandIndices() {
  if (CurrentSize < MinMatchTableSizeForCompaction)
    return;

  static constexpr StringLiteral RootOperandOpcodes[] = {
      "GIM_RootCheckType", "GIM_RootCheckRegBankForClass",
      "GIR_RootToRootCopy"};

  for (unsigned I = 0, E = Contents.size(); I != E; ++I) {
    MatchTableRecord &Opcode = Contents[I];
    if (!is_contained(RootOperandOpcodes, Opcode.EmitStr))
      continue;

    unsigned OperandRecord = I + 1;
    while (OperandRecord != E && Contents[OperandRecord].size() == 0)
      ++OperandRecord;
    assert(OperandRecord != E && "Missing root operand index");

    MatchTableRecord &Operand = Contents[OperandRecord];
    unsigned OperandIndex;
    if (Operand.size() != 1 ||
        StringRef(Operand.EmitStr).getAsInteger(10, OperandIndex) ||
        OperandIndex > 8)
      continue;

    Opcode.EmitStr += llvm::to_string(OperandIndex);
    Operand.clear();
  }

  rebuildLabelMap();
}

void MatchTable::compact() {
  compactFailureTargets();
  compactRootOperandIndices();
}

} // namespace gi
} // namespace llvm
