//===- StringTableBuilder.cpp - String table building utility -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

using namespace llvm;

StringTableBuilder::~StringTableBuilder() = default;

void StringTableBuilder::initSize() {
  // Account for leading bytes in table so that offsets returned from add are
  // correct.
  switch (K) {
  case RAW:
    Size = 0;
    break;
  case MachO:
  case ELF:
    // Start the table with a NUL byte.
    Size = 1;
    break;
  case WinCOFF:
    // Make room to write the table size later.
    Size = 4;
    break;
  }
}

StringTableBuilder::StringTableBuilder(Kind K, unsigned Alignment)
    : K(K), Alignment(Alignment) {
  initSize();
}

void StringTableBuilder::write(raw_ostream &OS) const {
  assert(isFinalized());
  SmallString<0> Data;
  Data.resize(getSize());
  write((uint8_t *)Data.data());
  OS << Data;
}

typedef std::pair<CachedHashStringRef, size_t> StringPair;

void StringTableBuilder::write(uint8_t *Buf) const {
  assert(isFinalized());
  for (const StringPair &P : StringIndexMap) {
    StringRef Data = P.first.val();
    if (!Data.empty())
      memcpy(Buf + P.second, Data.data(), Data.size());
  }
  if (K != WinCOFF)
    return;
  support::endian::write32le(Buf, Size);
}

// Returns the character at Pos from end of a string.
static int charTailAt(StringPair *P, size_t Pos) {
  StringRef S = P->first.val();
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
  Finalized = true;

  if (Optimize) {
    std::vector<StringPair *> Strings;
    Strings.reserve(StringIndexMap.size());
    for (StringPair &P : StringIndexMap)
      Strings.push_back(&P);

    if (!Strings.empty()) {
      // If we're optimizing, sort by name. If not, sort by previously assigned
      // offset.
      multikey_qsort(&Strings[0], &Strings[0] + Strings.size(), 0);
    }

    initSize();

    StringRef Previous;
    for (StringPair *P : Strings) {
      StringRef S = P->first.val();
      if (Previous.endswith(S)) {
        size_t Pos = Size - S.size() - (K != RAW);
        if (!(Pos & (Alignment - 1))) {
          P->second = Pos;
          continue;
        }
      }

      Size = alignTo(Size, Alignment);
      P->second = Size;

      Size += S.size();
      if (K != RAW)
        ++Size;
      Previous = S;
    }
  }

  if (K == MachO)
    Size = alignTo(Size, 4); // Pad to multiple of 4.
}

void StringTableBuilder::clear() {
  Finalized = false;
  StringIndexMap.clear();
}

size_t StringTableBuilder::getOffset(CachedHashStringRef S) const {
  assert(isFinalized());
  auto I = StringIndexMap.find(S);
  assert(I != StringIndexMap.end() && "String is not in table!");
  return I->second;
}

size_t StringTableBuilder::add(CachedHashStringRef S) {
  if (K == WinCOFF)
    assert(S.size() > COFF::NameSize && "Short string in COFF string table!");

  assert(!isFinalized());
  auto P = StringIndexMap.insert(std::make_pair(S, 0));
  if (P.second) {
    size_t Start = alignTo(Size, Alignment);
    P.first->second = Start;
    Size = Start + S.size() + (K != RAW);
  }
  return P.first->second;
}
