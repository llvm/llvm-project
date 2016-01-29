//===-- StringTableBuilder.cpp - String table building utility ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"

#include <vector>

using namespace llvm;

StringTableBuilder::StringTableBuilder(Kind K) : K(K) {
  // Account for leading bytes in table so that offsets returned from add are
  // correct.
  switch (K) {
  case RAW:
    Size = 0;
    break;
  case MachO:
  case ELF:
    Size = 1;
    break;
  case WinCOFF:
    Size = 4;
    break;
  }
}

typedef std::pair<StringRef, size_t> StringPair;

// Returns the character at Pos from end of a string.
static int charTailAt(StringPair *P, size_t Pos) {
  StringRef S = P->first;
  if (Pos >= S.size())
    return -1;
  return (unsigned char)S[S.size() - Pos - 1];
}

// Three-way radix quicksort. This is much faster than std::sort with strcmp
// because it does not compare characters that we already know the same.
static void multikey_qsort(StringPair **Begin, StringPair **End, int Pos) {
tailcall:
  if (End - Begin <= 1)
    return;

  // Partition items. Items in [Begin, P) are greater than the pivot,
  // [P, Q) are the same as the pivot, and [Q, End) are less than the pivot.
  int Pivot = charTailAt(*Begin, Pos);
  StringPair **P = Begin;
  StringPair **Q = End;
  for (StringPair **R = Begin + 1; R < Q;) {
    int C = charTailAt(*R, Pos);
    if (C > Pivot)
      std::swap(*P++, *R++);
    else if (C < Pivot)
      std::swap(*--Q, *R);
    else
      R++;
  }

  multikey_qsort(Begin, P, Pos);
  multikey_qsort(Q, End, Pos);
  if (Pivot != -1) {
    // qsort(P, Q, Pos + 1), but with tail call optimization.
    Begin = P;
    End = Q;
    ++Pos;
    goto tailcall;
  }
}

void StringTableBuilder::finalize() {
  finalizeStringTable(/*Optimize=*/true);
}

void StringTableBuilder::finalizeInOrder() {
  finalizeStringTable(/*Optimize=*/false);
}

void StringTableBuilder::finalizeStringTable(bool Optimize) {
  typedef std::pair<StringRef, size_t> StringOffsetPair;
  std::vector<StringOffsetPair *> Strings;
  Strings.reserve(StringIndexMap.size());
  for (StringOffsetPair &P : StringIndexMap)
    Strings.push_back(&P);

  if (!Strings.empty()) {
    // If we're optimizing, sort by name. If not, sort by previously assigned
    // offset.
    if (Optimize) {
      multikey_qsort(&Strings[0], &Strings[0] + Strings.size(), 0);
    } else {
      std::sort(Strings.begin(), Strings.end(),
                [](const StringOffsetPair *LHS, const StringOffsetPair *RHS) {
                  return LHS->second < RHS->second;
                });
    }
  }

  switch (K) {
  case RAW:
    break;
  case ELF:
  case MachO:
    // Start the table with a NUL byte.
    StringTable += '\x00';
    break;
  case WinCOFF:
    // Make room to write the table size later.
    StringTable.append(4, '\x00');
    break;
  }

  StringRef Previous;
  for (StringOffsetPair *P : Strings) {
    StringRef S = P->first;
    if (K == WinCOFF)
      assert(S.size() > COFF::NameSize && "Short string in COFF string table!");

    if (Optimize && Previous.endswith(S)) {
      P->second = StringTable.size() - S.size() - (K != RAW);
      continue;
    }

    if (Optimize)
      P->second = StringTable.size();
    else
      assert(P->second == StringTable.size() &&
             "different strtab offset after finalization");

    StringTable += S;
    if (K != RAW)
      StringTable += '\x00';
    Previous = S;
  }

  switch (K) {
  case RAW:
  case ELF:
    break;
  case MachO:
    // Pad to multiple of 4.
    while (StringTable.size() % 4)
      StringTable += '\x00';
    break;
  case WinCOFF:
    // Write the table size in the first word.
    assert(StringTable.size() <= std::numeric_limits<uint32_t>::max());
    uint32_t Size = static_cast<uint32_t>(StringTable.size());
    support::endian::write<uint32_t, support::little, support::unaligned>(
        StringTable.data(), Size);
    break;
  }

  Size = StringTable.size();
}

void StringTableBuilder::clear() {
  StringTable.clear();
  StringIndexMap.clear();
}

size_t StringTableBuilder::getOffset(StringRef S) const {
  assert(isFinalized());
  auto I = StringIndexMap.find(S);
  assert(I != StringIndexMap.end() && "String is not in table!");
  return I->second;
}

size_t StringTableBuilder::add(StringRef S) {
  assert(!isFinalized());
  auto P = StringIndexMap.insert(std::make_pair(S, Size));
  if (P.second)
    Size += S.size() + (K != RAW);
  return P.first->second;
}
