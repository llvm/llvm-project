//===- AttributeTableEmitter.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for emitting compact tables of IR attributes. AttributeSetTable
// uniques attribute sets and emits the cases of a switch constructing each one.
// AttributeSequenceTable uniques per-entity sequences of argument attribute
// entries and emits them as one table, sharing storage between sequences.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_BASIC_ATTRIBUTETABLEEMITTER_H
#define LLVM_UTILS_TABLEGEN_BASIC_ATTRIBUTETABLEEMITTER_H

#include "SequenceToOffsetTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// Uniques attribute sets, keyed by \p KeyT, and assigns them dense IDs in
/// insertion order.
template <typename KeyT> class AttributeSetTable {
  UniqueVector<KeyT> Sets;

public:
  /// Returns the ID of the set \p Attrs.
  unsigned add(const KeyT &Attrs) { return Sets.insert(Attrs) - 1; }

  unsigned size() const { return static_cast<unsigned>(Sets.size()); }

  /// Emits a switch case returning each set, in ID order. \p Print emits the
  /// attributes of a non-empty set. If provided, \p WriteComment writes a
  /// comment for the case of each ID.
  void emitCases(
      raw_ostream &OS, function_ref<void(raw_ostream &, const KeyT &)> Print,
      function_ref<void(raw_ostream &, unsigned)> WriteComment = {}) const {
    for (const auto &[ID, Attrs] : enumerate(Sets)) {
      OS << "\n  case " << ID << ':';
      if (WriteComment) {
        OS << " // ";
        WriteComment(OS, ID);
      }

      if (Attrs.empty()) {
        OS << "\n    return AttributeSet();";
        continue;
      }

      OS << "\n    return AttributeSet::get(C, {\n";
      Print(OS, Attrs);
      OS << "    });";
    }
  }
};

/// Uniques sequences of argument attribute entries, assigning them dense IDs in
/// insertion order, and lays them out in a single table where a sequence that
/// is a suffix of another shares its storage.
template <typename EntryT> class AttributeSequenceTable {
public:
  using Sequence = SmallVector<EntryT, 0>;

private:
  SequenceToOffsetTable<Sequence> Table{std::nullopt};
  UniqueVector<Sequence> Seqs;

public:
  /// Returns the ID of \p Seq. Must be called before the table is laid out.
  unsigned add(const Sequence &Seq) {
    size_t NumSeqs = Seqs.size();
    unsigned ID = Seqs.insert(Seq) - 1;
    if (Seqs.size() != NumSeqs && !Seq.empty())
      Table.add(Seq);
    return ID;
  }

  void layout() { Table.layout(); }

  /// Number of sequences.
  unsigned numSequences() const { return static_cast<unsigned>(Seqs.size()); }

  /// Number of entries in the laid out table.
  unsigned size() const { return Table.size(); }

  /// Emits the entries of the laid out table.
  void emitEntries(raw_ostream &OS,
                   function_ref<void(raw_ostream &, EntryT)> Print) const {
    Table.emit(OS, Print);
  }

  /// Emits the start index and length of each sequence, in ID order.
  void emitInfo(raw_ostream &OS) const {
    for (const Sequence &Seq : Seqs) {
      unsigned Start = Seq.empty() ? 0 : Table.get(Seq);
      OS << formatv("  {{{}, {}},\n", Start, Seq.size());
    }
  }
};

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_BASIC_ATTRIBUTETABLEEMITTER_H
