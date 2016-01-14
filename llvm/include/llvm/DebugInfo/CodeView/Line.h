//===- Line.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_LINE_H
#define LLVM_DEBUGINFO_CODEVIEW_LINE_H

#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/Endian.h"
#include <cinttypes>

namespace llvm {
namespace codeview {

using llvm::support::ulittle32_t;

class LineInfo {
public:
  static const uint32_t AlwaysStepIntoLineNumber = 0xfeefee;
  static const uint32_t NeverStepIntoLineNumber = 0xf00f00;

private:
  static const uint32_t StartLineMask = 0x00ffffff;
  static const uint32_t EndLineDeltaMask = 0x7f000000;
  static const int EndLineDeltaShift = 24;
  static const uint32_t StatementFlag = 0x80000000u;

public:
  LineInfo(uint32_t StartLine, uint32_t EndLine, bool IsStatement);

  uint32_t getStartLine() const { return LineData & StartLineMask; }

  uint32_t getLineDelta() const {
    return (LineData & EndLineDeltaMask) >> EndLineDeltaShift;
  }

  uint32_t getEndLine() const { return getStartLine() + getLineDelta(); }

  bool isStatement() const { return (LineData & StatementFlag) != 0; }

  uint32_t getRawData() const { return LineData; }

  bool isAlwaysStepInto() const {
    return getStartLine() == AlwaysStepIntoLineNumber;
  }

  bool isNeverStepInto() const {
    return getStartLine() == NeverStepIntoLineNumber;
  }

private:
  uint32_t LineData;
};

class ColumnInfo {
private:
  static const uint32_t StartColumnMask = 0x0000ffffu;
  static const uint32_t EndColumnMask = 0xffff0000u;
  static const int EndColumnShift = 16;

public:
  ColumnInfo(uint16_t StartColumn, uint16_t EndColumn) {
    ColumnData =
        (static_cast<uint32_t>(StartColumn) & StartColumnMask) |
        ((static_cast<uint32_t>(EndColumn) << EndColumnShift) & EndColumnMask);
  }

  uint16_t getStartColumn() const {
    return static_cast<uint16_t>(ColumnData & StartColumnMask);
  }

  uint16_t getEndColumn() const {
    return static_cast<uint16_t>((ColumnData & EndColumnMask) >>
                                 EndColumnShift);
  }

  uint32_t getRawData() const { return ColumnData; }

private:
  uint32_t ColumnData;
};

class Line {
private:
  int32_t CodeOffset;
  LineInfo LineInf;
  ColumnInfo ColumnInf;

public:
  Line(int32_t CodeOffset, uint32_t StartLine, uint32_t EndLine,
       uint16_t StartColumn, uint16_t EndColumn, bool IsStatement)
      : CodeOffset(CodeOffset), LineInf(StartLine, EndLine, IsStatement),
        ColumnInf(StartColumn, EndColumn) {}

  Line(int32_t CodeOffset, LineInfo LineInf, ColumnInfo ColumnInf)
      : CodeOffset(CodeOffset), LineInf(LineInf), ColumnInf(ColumnInf) {}

  LineInfo getLineInfo() const { return LineInf; }

  ColumnInfo getColumnInfo() const { return ColumnInf; }

  int32_t getCodeOffset() const { return CodeOffset; }

  uint32_t getStartLine() const { return LineInf.getStartLine(); }

  uint32_t getLineDelta() const { return LineInf.getLineDelta(); }

  uint32_t getEndLine() const { return LineInf.getEndLine(); }

  uint16_t getStartColumn() const { return ColumnInf.getStartColumn(); }

  uint16_t getEndColumn() const { return ColumnInf.getEndColumn(); }

  bool isStatement() const { return LineInf.isStatement(); }

  bool isAlwaysStepInto() const { return LineInf.isAlwaysStepInto(); }

  bool isNeverStepInto() const { return LineInf.isNeverStepInto(); }
};

enum class InlineeLinesSignature : uint32_t {
  Normal,    // CV_INLINEE_SOURCE_LINE_SIGNATURE
  ExtraFiles // CV_INLINEE_SOURCE_LINE_SIGNATURE_EX
};

struct InlineeSourceLine {
  TypeIndex Inlinee;         // ID of the function that was inlined.
  ulittle32_t FileID;        // Offset into FileChecksums subsection.
  ulittle32_t SourceLineNum; // First line of inlined code.
  // If extra files present:
  //   ulittle32_t ExtraFileCount;
  //   ulittle32_t Files[];
};

} // namespace codeview
} // namespace llvm

#endif
