//===- StringTableBuilder.cpp - String table building utility -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
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
  case DWARF:
    Size = 0;
    break;
  case MachOLinked:
  case MachO64Linked:
    Size = 2;
    break;
  case MachO:
  case MachO64:
  case ELF:
  case DXContainer:
    // Start the table with a NUL byte.
    Size = 1;
    break;
  case XCOFF:
  case WinCOFF:
    // Make room to write the table size later.
    Size = 4;
    break;
  }
}

StringTableBuilder::StringTableBuilder(Kind K, Align Alignment)
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

using StringPair = DenseMap<CachedHashStringRef, size_t>::value_type;

void StringTableBuilder::write(uint8_t *Buf) const {
  assert(isFinalized());
  for (const StringPair &P : StringIndexMap) {
    StringRef Data = P.first.val();
    if (!Data.empty())
      memcpy(Buf + P.second, Data.data(), Data.size());
  }
  // The COFF formats store the size of the string table in the first 4 bytes.
  // For Windows, the format is little-endian; for AIX, it is big-endian.
  if (K == WinCOFF)
    support::endian::write32le(Buf, Size);
  else if (K == XCOFF)
    support::endian::write32be(Buf, Size);
}

namespace {
// A string being sorted. Key caches tailKey at the current sort depth.
struct TailEntry {
  uint64_t Key;
  const char *Data;
  uint32_t Len;
  uint32_t Idx;
};
} // namespace

// Return the 8 characters ending Pos characters from the end of a string, the
// last one in the most significant byte. Missing characters are 0.
static uint64_t tailKey(const char *Data, size_t Len, size_t Pos) {
  if (Pos >= Len)
    return 0;
  size_t R = Len - Pos;
  if (R >= 8)
    return support::endian::read64le(Data + R - 8);
  uint64_t Key = 0;
  for (size_t I = 0; I != R; ++I)
    Key |= uint64_t((unsigned char)Data[I]) << (8 * (8 - R + I));
  return Key;
}

// Return true if A reversed is greater than B reversed, given that their last
// Pos characters are equal.
static bool tailGreater(const TailEntry &A, const TailEntry &B, size_t Pos) {
  const char *EA = A.Data + A.Len, *EB = B.Data + B.Len;
  size_t N = std::min(A.Len, B.Len), I = Pos;
  for (; I + 8 <= N; I += 8)
    if (memcmp(EA - I - 8, EB - I - 8, 8))
      break;
  for (; I < N; ++I) {
    unsigned char CA = *(EA - I - 1), CB = *(EB - I - 1);
    if (CA != CB)
      return CA > CB;
  }
  return A.Len > B.Len;
}

// Sort strings in descending order of their reversed strings, so that a string
// that is a suffix of others immediately follows one of them. Their last `Pos`
// characters are equal. Sort by the keys at `Pos`, then recurse into each run
// of equal keys at Pos + 8.
static void multikeySort(MutableArrayRef<TailEntry> Vec, size_t Pos) {
tailcall:
  if (Vec.size() <= 1)
    return;
  for (TailEntry &E : Vec)
    E.Key = tailKey(E.Data, E.Len, Pos);
  if (Vec.size() > 8) {
    llvm::sort(Vec, [](const TailEntry &A, const TailEntry &B) {
      return A.Key > B.Key;
    });
    // Keys represent characters past the end as 0, like NUL characters, so
    // keys of strings that differ only in leading NULs stay 0.
    if (Vec[0].Key) {
      size_t I = 0;
      for (size_t J = 1; J != Vec.size(); ++J) {
        if (Vec[J].Key != Vec[I].Key) {
          multikeySort(Vec.slice(I, J - I), Pos + 8);
          I = J;
        }
      }
      Vec = Vec.slice(I);
      Pos += 8;
      goto tailcall;
    }
  }
  llvm::sort(Vec, [Pos](const TailEntry &A, const TailEntry &B) {
    if (A.Key != B.Key)
      return A.Key > B.Key;
    return tailGreater(A, B, Pos);
  });
}

void StringTableBuilder::finalize() {
  assert(K != DWARF);
  finalizeStringTable(/*Optimize=*/true);
}

void StringTableBuilder::finalizeInOrder() {
  finalizeStringTable(/*Optimize=*/false);
}

void StringTableBuilder::finalizeStringTable(bool Optimize) {
  Finalized = true;

  if (Optimize && StringIndexMap.size()) {
    std::vector<StringPair *> Strings;
    Strings.reserve(StringIndexMap.size());
    for (StringPair &P : StringIndexMap)
      Strings.push_back(&P);

    if (StringPriorityMap.size()) {
      llvm::sort(Strings,
                 [&](const StringPair *LHS, const StringPair *RHS) -> bool {
                   return StringPriorityMap.lookup(LHS->first) >
                          StringPriorityMap.lookup(RHS->first);
                 });
    }
    std::vector<TailEntry> Entries(Strings.size());
    for (auto [I, P] : enumerate(Strings)) {
      StringRef S = P->first.val();
      assert(S.size() <= UINT32_MAX);
      Entries[I] = {0, S.data(), uint32_t(S.size()), uint32_t(I)};
    }

    size_t RangeBegin = 0;
    MutableArrayRef<TailEntry> EntriesRef(Entries);
    if (StringPriorityMap.size()) {
      uint8_t RangePriority = StringPriorityMap.lookup(Strings[0]->first);
      for (size_t I = 1, E = Strings.size(); I != E && RangePriority; ++I) {
        uint8_t Priority = StringPriorityMap.lookup(Strings[I]->first);
        if (Priority != RangePriority) {
          multikeySort(EntriesRef.slice(RangeBegin, I - RangeBegin), 0);
          RangePriority = Priority;
          RangeBegin = I;
        }
      }
    }
    multikeySort(EntriesRef.slice(RangeBegin), 0);
    initSize();

    StringRef Previous;
    for (const TailEntry &E : Entries) {
      StringPair *P = Strings[E.Idx];
      StringRef S = P->first.val();
      if (Previous.ends_with(S)) {
        size_t Pos = Size - S.size() - (K != RAW);
        if (isAligned(Alignment, Pos)) {
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

  if (K == MachO || K == MachOLinked || K == DXContainer)
    Size = alignTo(Size, 4); // Pad to multiple of 4.
  if (K == MachO64 || K == MachO64Linked)
    Size = alignTo(Size, 8); // Pad to multiple of 8.

  // According to ld64 the string table of a final linked Mach-O binary starts
  // with " ", i.e. the first byte is ' ' and the second byte is zero. In
  // 'initSize()' we reserved the first two bytes for holding this string.
  if (K == MachOLinked || K == MachO64Linked)
    StringIndexMap[CachedHashStringRef(" ")] = 0;

  // The first byte in an ELF string table must be null, according to the ELF
  // specification. In 'initSize()' we reserved the first byte to hold null for
  // this purpose and here we actually add the string to allow 'getOffset()' to
  // be called on an empty string.
  if (K == ELF || K == DXContainer)
    StringIndexMap[CachedHashStringRef("")] = 0;
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

