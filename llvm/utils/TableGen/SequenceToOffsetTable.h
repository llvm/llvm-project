//===-- SequenceToOffsetTable.h - Compress similar sequences ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SequenceToOffsetTable can be used to emit a number of null-terminated
// sequences as one big array.  Use the same memory when a sequence is a suffix
// of another.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_SEQUENCETOOFFSETTABLE_H
#define LLVM_UTILS_TABLEGEN_SEQUENCETOOFFSETTABLE_H

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <functional>
#include <map>

namespace llvm {
extern llvm::cl::opt<bool> EmitLongStrLiterals;

// Helper function for SequenceToOffsetTable<string>.
static inline void printStrLitEscChar(raw_ostream &OS, char C) {
  const char *Escapes[] = {
      "\\000", "\\001", "\\002", "\\003", "\\004", "\\005", "\\006", "\\007",
      "\\010", "\\t",   "\\n",   "\\013", "\\014", "\\r",   "\\016", "\\017",
      "\\020", "\\021", "\\022", "\\023", "\\024", "\\025", "\\026", "\\027",
      "\\030", "\\031", "\\032", "\\033", "\\034", "\\035", "\\036", "\\037",
      " ",     "!",     "\\\"",  "#",     "$",     "%",     "&",     "'",
      "(",     ")",     "*",     "+",     ",",     "-",     ".",     "/",
      "0",     "1",     "2",     "3",     "4",     "5",     "6",     "7",
      "8",     "9",     ":",     ";",     "<",     "=",     ">",     "?",
      "@",     "A",     "B",     "C",     "D",     "E",     "F",     "G",
      "H",     "I",     "J",     "K",     "L",     "M",     "N",     "O",
      "P",     "Q",     "R",     "S",     "T",     "U",     "V",     "W",
      "X",     "Y",     "Z",     "[",     "\\\\",  "]",     "^",     "_",
      "`",     "a",     "b",     "c",     "d",     "e",     "f",     "g",
      "h",     "i",     "j",     "k",     "l",     "m",     "n",     "o",
      "p",     "q",     "r",     "s",     "t",     "u",     "v",     "w",
      "x",     "y",     "z",     "{",     "|",     "}",     "~",     "\\177",
      "\\200", "\\201", "\\202", "\\203", "\\204", "\\205", "\\206", "\\207",
      "\\210", "\\211", "\\212", "\\213", "\\214", "\\215", "\\216", "\\217",
      "\\220", "\\221", "\\222", "\\223", "\\224", "\\225", "\\226", "\\227",
      "\\230", "\\231", "\\232", "\\233", "\\234", "\\235", "\\236", "\\237",
      "\\240", "\\241", "\\242", "\\243", "\\244", "\\245", "\\246", "\\247",
      "\\250", "\\251", "\\252", "\\253", "\\254", "\\255", "\\256", "\\257",
      "\\260", "\\261", "\\262", "\\263", "\\264", "\\265", "\\266", "\\267",
      "\\270", "\\271", "\\272", "\\273", "\\274", "\\275", "\\276", "\\277",
      "\\300", "\\301", "\\302", "\\303", "\\304", "\\305", "\\306", "\\307",
      "\\310", "\\311", "\\312", "\\313", "\\314", "\\315", "\\316", "\\317",
      "\\320", "\\321", "\\322", "\\323", "\\324", "\\325", "\\326", "\\327",
      "\\330", "\\331", "\\332", "\\333", "\\334", "\\335", "\\336", "\\337",
      "\\340", "\\341", "\\342", "\\343", "\\344", "\\345", "\\346", "\\347",
      "\\350", "\\351", "\\352", "\\353", "\\354", "\\355", "\\356", "\\357",
      "\\360", "\\361", "\\362", "\\363", "\\364", "\\365", "\\366", "\\367",
      "\\370", "\\371", "\\372", "\\373", "\\374", "\\375", "\\376", "\\377"};

  static_assert(sizeof Escapes / sizeof Escapes[0] ==
                    std::numeric_limits<unsigned char>::max() + 1,
                "unsupported character type");
  OS << Escapes[static_cast<unsigned char>(C)];
}

static inline void printChar(raw_ostream &OS, char C) {
  unsigned char UC(C);
  if (isalnum(UC) || ispunct(UC)) {
    OS << '\'';
    if (C == '\\' || C == '\'')
      OS << '\\';
    OS << C << '\'';
  } else {
    OS << unsigned(UC);
  }
}

/// SequenceToOffsetTable - Collect a number of terminated sequences of T.
/// Compute the layout of a table that contains all the sequences, possibly by
/// reusing entries.
///
/// @tparam SeqT The sequence container. (vector or string).
/// @tparam Less A stable comparator for SeqT elements.
template<typename SeqT, typename Less = std::less<typename SeqT::value_type> >
class SequenceToOffsetTable {
  typedef typename SeqT::value_type ElemT;

  // Define a comparator for SeqT that sorts a suffix immediately before a
  // sequence with that suffix.
  struct SeqLess {
    Less L;
    bool operator()(const SeqT &A, const SeqT &B) const {
      return std::lexicographical_compare(A.rbegin(), A.rend(),
                                          B.rbegin(), B.rend(), L);
    }
  };

  // Keep sequences ordered according to SeqLess so suffixes are easy to find.
  // Map each sequence to its offset in the table.
  typedef std::map<SeqT, unsigned, SeqLess> SeqMap;

  // Sequences added so far, with suffixes removed.
  SeqMap Seqs;

  // Entries in the final table, or 0 before layout was called.
  unsigned Entries;

  // isSuffix - Returns true if A is a suffix of B.
  static bool isSuffix(const SeqT &A, const SeqT &B) {
    return A.size() <= B.size() && std::equal(A.rbegin(), A.rend(), B.rbegin());
  }

public:
  SequenceToOffsetTable() : Entries(0) {}

  /// add - Add a sequence to the table.
  /// This must be called before layout().
  void add(const SeqT &Seq) {
    assert(Entries == 0 && "Cannot call add() after layout()");
    typename SeqMap::iterator I = Seqs.lower_bound(Seq);

    // If SeqMap contains a sequence that has Seq as a suffix, I will be
    // pointing to it.
    if (I != Seqs.end() && isSuffix(Seq, I->first))
      return;

    I = Seqs.insert(I, std::make_pair(Seq, 0u));

    // The entry before I may be a suffix of Seq that can now be erased.
    if (I != Seqs.begin() && isSuffix((--I)->first, Seq))
      Seqs.erase(I);
  }

  bool empty() const { return Seqs.empty(); }

  unsigned size() const {
    assert((empty() || Entries) && "Call layout() before size()");
    return Entries;
  }

  /// layout - Computes the final table layout.
  void layout() {
    assert(Entries == 0 && "Can only call layout() once");
    // Lay out the table in Seqs iteration order.
    for (typename SeqMap::iterator I = Seqs.begin(), E = Seqs.end(); I != E;
         ++I) {
      I->second = Entries;
      // Include space for a terminator.
      Entries += I->first.size() + 1;
    }
  }

  /// get - Returns the offset of Seq in the final table.
  unsigned get(const SeqT &Seq) const {
    assert(Entries && "Call layout() before get()");
    typename SeqMap::const_iterator I = Seqs.lower_bound(Seq);
    assert(I != Seqs.end() && isSuffix(Seq, I->first) &&
           "get() called with sequence that wasn't added first");
    return I->second + (I->first.size() - Seq.size());
  }

  /// `emitStringLiteralDef` - Print out the table as the body of an array
  /// initializer, where each element is a C string literal terminated by
  /// `\0`. Falls back to emitting a comma-separated integer list if
  /// `EmitLongStrLiterals` is false
  void emitStringLiteralDef(raw_ostream &OS, const llvm::Twine &Decl) const {
    assert(Entries && "Call layout() before emitStringLiteralDef()");
    if (EmitLongStrLiterals) {
      OS << "\n#ifdef __GNUC__\n"
         << "#pragma GCC diagnostic push\n"
         << "#pragma GCC diagnostic ignored \"-Woverlength-strings\"\n"
         << "#endif\n"
         << Decl << " = {\n";
    } else {
      OS << Decl << " = {\n";
      emit(OS, printChar, "0");
      OS << "\n};\n\n";
      return;
    }
    for (auto I : Seqs) {
      OS << "  /* " << I.second << " */ \"";
      for (auto C : I.first) {
        printStrLitEscChar(OS, C);
      }
      OS << "\\0\"\n";
    }
    OS << "};\n"
       << "#ifdef __GNUC__\n"
       << "#pragma GCC diagnostic pop\n"
       << "#endif\n\n";
  }

  /// emit - Print out the table as the body of an array initializer.
  /// Use the Print function to print elements.
  void emit(raw_ostream &OS,
            void (*Print)(raw_ostream&, ElemT),
            const char *Term = "0") const {
    assert((empty() || Entries) && "Call layout() before emit()");
    for (typename SeqMap::const_iterator I = Seqs.begin(), E = Seqs.end();
         I != E; ++I) {
      OS << "  /* " << I->second << " */ ";
      for (typename SeqT::const_iterator SI = I->first.begin(),
             SE = I->first.end(); SI != SE; ++SI) {
        Print(OS, *SI);
        OS << ", ";
      }
      OS << Term << ",\n";
    }
  }
};

} // end namespace llvm

#endif
