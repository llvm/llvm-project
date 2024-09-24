//===- SeedCollector.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains the mechanism for collecting the seed instructions that
// are used as starting points for forming the vectorization graph.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SEEDCOLLECTOR_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SEEDCOLLECTOR_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/SandboxIR/Value.h"
#include <iterator>
#include <memory>

namespace llvm::sandboxir {
class Instruction;
class StoreInst;
class BasicBlock;

/// An ordered set of Instructions that can be vectorized.
class SeedBundle {
public:
  using SeedList = SmallVector<sandboxir::Instruction *>;
  /// Initialize a bundle with \p I.
  explicit SeedBundle(sandboxir::Instruction *I, const DataLayout &DL) {
    insertAt(begin(), I, DL);
  }
  explicit SeedBundle(SeedList &&L, const DataLayout &DL)
      : Seeds(std::move(L)) {
    for (auto &S : Seeds) {
      NumUnusedBits += sandboxir::Utils::getNumBits(S, DL);
    }
  }
  /// No need to allow copies.
  SeedBundle(const SeedBundle &) = delete;
  SeedBundle &operator=(const SeedBundle &) = delete;
  virtual ~SeedBundle() {}

  using iterator = SeedList::iterator;
  using const_iterator = SeedList::const_iterator;
  iterator begin() { return Seeds.begin(); }
  iterator end() { return Seeds.end(); }
  const_iterator begin() const { return Seeds.begin(); }
  const_iterator end() const { return Seeds.end(); }

  sandboxir::Instruction *operator[](unsigned Idx) const { return Seeds[Idx]; }

  /// Insert \p I into position \p P. Clients should choose Pos
  /// by symbol, symbol-offset, and program order (which depends if scheduling
  /// bottom-up or top-down).
  void insertAt(iterator Pos, sandboxir::Instruction *I, const DataLayout &DL) {
#ifdef EXPENSIVE_CHECKS
    for (auto Itr : Seeds) {
      assert(*Itr != I && "Attempt to insert an instruction twice.");
    }
#endif
    Seeds.insert(Pos, I);
    NumUnusedBits += sandboxir::Utils::getNumBits(I, DL);
  }

  unsigned getFirstUnusedElementIdx() const {
    for (unsigned ElmIdx : seq<unsigned>(0, Seeds.size()))
      if (!isUsed(ElmIdx))
        return ElmIdx;
    return Seeds.size();
  }
  /// Marks elements as 'used' so that we skip them in `getSlice()`.
  void setUsed(unsigned ElementIdx, const DataLayout &DL, unsigned Sz = 1,
               bool VerifyUnused = true) {
    if (ElementIdx + Sz >= UsedLanes.size())
      UsedLanes.resize(ElementIdx + Sz);
    for (unsigned Idx : seq<unsigned>(ElementIdx, ElementIdx + Sz)) {
      assert((!VerifyUnused || !UsedLanes.test(Idx)) &&
             "Already marked as used!");
      UsedLanes.set(Idx);
      UsedLaneCount++;
    }
    NumUnusedBits -= sandboxir::Utils::getNumBits(Seeds[ElementIdx], DL);
  }

  void setUsed(sandboxir::Instruction *V, const DataLayout &DL) {
    auto It = std::find(begin(), end(), V);
    assert(It != end() && "V not in the bundle!");
    auto Idx = It - begin();
    setUsed(Idx, DL, 1, /*VerifyUnused=*/false);
  }
  bool isUsed(unsigned Element) const {
    return Element >= UsedLanes.size() ? false : UsedLanes.test(Element);
  }
  bool allUsed() const { return UsedLaneCount == Seeds.size(); }
  unsigned getNumUnusedBits() const { return NumUnusedBits; }

  /// \Returns a slice of seed elements, starting at the element \p StartIdx,
  /// with a total size <= \p MaxVecRegBits. If \p ForcePowOf2 is true, then the
  /// returned slice should have a total number of bits that is a power of 2.
  MutableArrayRef<SeedList> getSlice(unsigned StartIdx, unsigned MaxVecRegBits,
                                     bool ForcePowOf2, const DataLayout &DL);

protected:
  SeedList Seeds;
  /// The lanes that we have already vectorized.
  BitVector UsedLanes;
  /// Tracks used lanes for constant-time accessor.
  unsigned UsedLaneCount = 0;
  /// Tracks the remaining bits available to vectorize
  unsigned NumUnusedBits = 0;

public:
#ifndef NDEBUG
  void dump(raw_ostream &OS) const {
    for (auto [ElmIdx, I] : enumerate(*this)) {
      OS.indent(2) << ElmIdx << ". ";
      if (isUsed(ElmIdx))
        OS << "[USED]";
      else
        OS << *I;
      OS << "\n";
    }
  }
  LLVM_DUMP_METHOD void dump() const {
    dump(dbgs());
    dbgs() << "\n";
  }
#endif // NDEBUG
};
} // namespace llvm::sandboxir
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SEEDCOLLECTOR_H
