//===- MatchTable.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the generic emitter for the GlobalISel Match Table
/// system. This file only contains the code used to emit the table itself.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_GLOBALISEL_MATCHTABLE_MATCHTABLE_H
#define LLVM_UTILS_TABLEGEN_COMMON_GLOBALISEL_MATCHTABLE_MATCHTABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace llvm {
class raw_ostream;
class Record;
namespace gi {
class Matcher;
class MatchTable;

void emitEncodingMacrosDef(raw_ostream &OS);
void emitEncodingMacrosUndef(raw_ostream &OS);

std::string getNameForFeatureBitset(ArrayRef<const Record *> FeatureBitset,
                                    int HwModeIdx);

/// A record to be stored in a MatchTable.
///
/// This class represents any and all output that may be required to emit the
/// MatchTable. Instances  are most often configured to represent an opcode or
/// value that will be emitted to the table with some formatting but it can also
/// represent commas, comments, and other formatting instructions.
struct MatchTableRecord {
  enum RecordFlagsBits {
    MTRF_None = 0x0,
    /// Causes EmitStr to be formatted as comment when emitted.
    MTRF_Comment = 0x1,
    /// Causes the record value to be followed by a comma when emitted.
    MTRF_CommaFollows = 0x2,
    /// Causes the record value to be followed by a line break when emitted.
    MTRF_LineBreakFollows = 0x4,
    /// Indicates that the record defines a label and causes an additional
    /// comment to be emitted containing the index of the label.
    MTRF_Label = 0x8,
    /// Causes the record to be emitted as the index of the label specified by
    /// LabelID along with a comment indicating where that label is.
    MTRF_JumpTarget = 0x10,
    /// Causes the formatter to add a level of indentation before emitting the
    /// record.
    MTRF_Indent = 0x20,
    /// Causes the formatter to remove a level of indentation after emitting the
    /// record.
    MTRF_Outdent = 0x40,
    /// Causes the formatter to not use encoding macros to emit this multi-byte
    /// value.
    MTRF_PreEncoded = 0x80,
    /// Causes a jump target to be emitted relative to the end of this record.
    MTRF_RelativeJumpTarget = 0x100,
  };

  /// When MTRF_Label or MTRF_JumpTarget is used, indicates a label id to
  /// reference or define.
  unsigned LabelID;
  /// The string to emit. Depending on the MTRF_* flags it may be a comment, a
  /// value, a label name.
  std::string EmitStr;

private:
  /// The number of MatchTable elements described by this record. Comments are 0
  /// while values are typically 1. Values >1 may occur when we need to emit
  /// values that exceed the size of a MatchTable element.
  unsigned NumElements;

public:
  /// A bitfield of RecordFlagsBits flags.
  unsigned Flags;

  MatchTableRecord(std::optional<unsigned> LabelID_, StringRef EmitStr,
                   unsigned NumElements, unsigned Flags)
      : LabelID(LabelID_.value_or(~0u)), EmitStr(EmitStr),
        NumElements(NumElements), Flags(Flags) {
    assert((!LabelID_ || LabelID != ~0u) &&
           "This value is reserved for non-labels");
  }
  MatchTableRecord(const MatchTableRecord &Other) = default;
  MatchTableRecord(MatchTableRecord &&Other) = default;

  /// Useful if a Match Table Record gets optimized out
  void turnIntoComment() {
    Flags |= MTRF_Comment;
    Flags &= ~MTRF_CommaFollows;
    NumElements = 0;
  }

  void makeRelativeJumpTarget(unsigned NumBytes) {
    assert(Flags & MTRF_JumpTarget);
    assert((NumBytes == 1 || NumBytes == 2) && "Unsupported jump size");
    Flags |= MTRF_RelativeJumpTarget;
    NumElements = NumBytes;
  }

  void clear() {
    EmitStr.clear();
    NumElements = 0;
    Flags = MTRF_None;
  }

  void emit(raw_ostream &OS, bool LineBreakNextAfterThis,
            const MatchTable &Table, unsigned CurrentIndex) const;
  unsigned size() const { return NumElements; }
};

/// Holds the contents of a generated MatchTable to enable formatting and the
/// necessary index tracking needed to support GIM_Try.
class MatchTable {
  /// An unique identifier for the table. The generated table will be named
  /// MatchTable${ID}.
  unsigned ID;
  /// The records that make up the table. Also includes comments describing the
  /// values being emitted and line breaks to format it.
  std::vector<MatchTableRecord> Contents;
  /// The currently defined labels.
  DenseMap<unsigned, unsigned> LabelMap;
  /// Tracks the sum of MatchTableRecord::NumElements as the table is built.
  unsigned CurrentSize = 0;
  /// A unique identifier for a MatchTable label.
  unsigned CurrentLabelID = 0;
  /// Determines if the table should be instrumented for rule coverage tracking.
  bool IsWithCoverage;
  /// Whether this table is for the GISel combiner.
  bool IsCombinerTable;

  void compactFailureTargets();
  void compactRootOperandIndices();
  void rebuildLabelMap();

public:
  static MatchTableRecord LineBreak;
  static MatchTableRecord Comment(StringRef Comment);
  static MatchTableRecord Opcode(StringRef Opcode, int IndentAdjust = 0);
  static MatchTableRecord NamedValue(unsigned NumBytes, StringRef NamedValue);
  static MatchTableRecord NamedValue(unsigned NumBytes, StringRef Namespace,
                                     StringRef NamedValue);
  static MatchTableRecord IntValue(unsigned NumBytes, int64_t IntValue);
  static MatchTableRecord ULEB128Value(uint64_t IntValue);
  static MatchTableRecord Label(unsigned LabelID);
  static MatchTableRecord JumpTarget(unsigned LabelID);

  MatchTable(bool WithCoverage, bool IsCombinerTable, unsigned ID = 0)
      : ID(ID), IsWithCoverage(WithCoverage), IsCombinerTable(IsCombinerTable) {
  }

  bool isWithCoverage() const { return IsWithCoverage; }
  bool isCombiner() const { return IsCombinerTable; }
  void compact();

  void push_back(const MatchTableRecord &Value) {
    if (Value.Flags & MatchTableRecord::MTRF_Label)
      defineLabel(Value.LabelID);
    Contents.push_back(Value);
    CurrentSize += Value.size();
  }

  unsigned allocateLabelID() { return CurrentLabelID++; }

  void defineLabel(unsigned LabelID) {
    LabelMap.try_emplace(LabelID, CurrentSize);
  }

  unsigned getLabelIndex(unsigned LabelID) const {
    const auto I = LabelMap.find(LabelID);
    assert(I != LabelMap.end() && "Use of undeclared label");
    return I->second;
  }

  void emitUse(raw_ostream &OS) const;
  void emitDeclaration(raw_ostream &OS) const;
};

inline MatchTable &operator<<(MatchTable &Table,
                              const MatchTableRecord &Value) {
  Table.push_back(Value);
  return Table;
}

} // namespace gi
} // namespace llvm

#endif
