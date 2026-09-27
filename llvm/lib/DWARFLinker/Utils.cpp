//===- Utils.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinker/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <map>

namespace llvm {
namespace dwarf_linker {

void buildStmtSeqOffsetToFirstRowIndex(
    const DWARFDebugLine::LineTable &LT,
    ArrayRef<uint64_t> SortedStmtSeqOffsets,
    DenseMap<uint64_t, uint64_t> &SeqOffToFirstRow) {
  // Use std::map for ordered iteration by input stmt-sequence offset.
  std::map<uint64_t, uint64_t> LineTableMapping;
  for (const DWARFDebugLine::Sequence &Seq : LT.Sequences)
    LineTableMapping[Seq.StmtSeqOffset] = Seq.FirstRowIndex;

  if (LT.Rows.empty()) {
    for (const auto &[Off, Row] : LineTableMapping)
      SeqOffToFirstRow[Off] = Row;
    return;
  }

  // Row indices that look like sequence starts: row 0, plus every row
  // immediately following an end_sequence marker.
  SmallVector<uint64_t> SeqStartRows;
  SeqStartRows.push_back(0);
  for (auto [I, Row] : llvm::enumerate(ArrayRef(LT.Rows).drop_back()))
    if (Row.EndSequence)
      SeqStartRows.push_back(I + 1);

  ArrayRef<uint64_t> StmtAttrsRef(SortedStmtSeqOffsets);
  ArrayRef<uint64_t> SeqStartRowsRef(SeqStartRows);

  // While SeqOffToFirstRow parsed from LT could be the ground truth, e.g.
  //
  // SeqOff     Row
  // 0x08        9
  // 0x14       15
  //
  // The StmtAttrs and SeqStartRows may not match perfectly, e.g.
  //
  // StmtAttrs  SeqStartRows
  // 0x04        3
  // 0x08        5
  // 0x10        9
  // 0x12       11
  // 0x14       15
  //
  // In this case, we don't want to assign 5 to 0x08, since we know 0x08
  // maps to 9. If we do a dummy 1:1 mapping 0x10 will be mapped to 9
  // which is incorrect. The expected behavior is ignore 5, realign the
  // table based on the result from the line table:
  //
  // StmtAttrs  SeqStartRows
  // 0x04        3
  //   --        5
  // 0x08        9 <- LineTableMapping ground truth
  // 0x10       11
  // 0x12       --
  // 0x14       15 <- LineTableMapping ground truth

  // Dummy trailing anchor so both refs always drain before we run out
  // of map entries to walk.
  constexpr uint64_t DummyKey = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t DummyVal = std::numeric_limits<uint64_t>::max();
  LineTableMapping[DummyKey] = DummyVal;

  for (auto [NextSeqOff, NextRow] : LineTableMapping) {
    auto StmtAttrSmallerThanNext = [N = NextSeqOff](uint64_t SA) {
      return SA < N;
    };
    auto SeqStartSmallerThanNext = [N = NextRow](uint64_t Row) {
      return Row < N;
    };
    // While both lists still point strictly before the next anchor,
    // pair them up 1:1 — this captures sequences the parser missed.
    while (!StmtAttrsRef.empty() && !SeqStartRowsRef.empty() &&
           StmtAttrSmallerThanNext(StmtAttrsRef.front()) &&
           SeqStartSmallerThanNext(SeqStartRowsRef.front())) {
      SeqOffToFirstRow[StmtAttrsRef.consume_front()] =
          SeqStartRowsRef.consume_front();
    }
    // Either list may now be ahead of or at the anchor: drop entries we
    // can't safely pair, then use the parser's (NextSeqOff,NextRow)
    // mapping as ground truth.
    StmtAttrsRef = StmtAttrsRef.drop_while(StmtAttrSmallerThanNext);
    SeqStartRowsRef = SeqStartRowsRef.drop_while(SeqStartSmallerThanNext);
    if (NextSeqOff != DummyKey)
      SeqOffToFirstRow[NextSeqOff] = NextRow;
    // Advance each list past the anchor only if it was pointing exactly
    // at it.
    if (!StmtAttrsRef.empty() && StmtAttrsRef.front() == NextSeqOff)
      StmtAttrsRef = StmtAttrsRef.drop_front();
    if (!SeqStartRowsRef.empty() && SeqStartRowsRef.front() == NextRow)
      SeqStartRowsRef = SeqStartRowsRef.drop_front();
  }
}

} // namespace dwarf_linker
} // namespace llvm
