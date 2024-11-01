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

/// A set of candidate Instructions for vectorizing together.
class SeedBundle {
public:
  /// Initialize a bundle with \p I.
  explicit SeedBundle(Instruction *I) { insertAt(begin(), I); }
  explicit SeedBundle(SmallVector<Instruction *> &&L) : Seeds(std::move(L)) {
    for (auto &S : Seeds)
      NumUnusedBits += Utils::getNumBits(S);
  }
  /// No need to allow copies.
  SeedBundle(const SeedBundle &) = delete;
  SeedBundle &operator=(const SeedBundle &) = delete;
  virtual ~SeedBundle() {}

  using iterator = SmallVector<Instruction *>::iterator;
  using const_iterator = SmallVector<Instruction *>::const_iterator;
  iterator begin() { return Seeds.begin(); }
  iterator end() { return Seeds.end(); }
  const_iterator begin() const { return Seeds.begin(); }
  const_iterator end() const { return Seeds.end(); }

  Instruction *operator[](unsigned Idx) const { return Seeds[Idx]; }

  /// Insert \p I into position \p P. Clients should choose Pos
  /// by symbol, symbol-offset, and program order (which depends if scheduling
  /// bottom-up or top-down).
  void insertAt(iterator Pos, Instruction *I) {
    Seeds.insert(Pos, I);
    NumUnusedBits += Utils::getNumBits(I);
  }

  unsigned getFirstUnusedElementIdx() const {
    for (unsigned ElmIdx : seq<unsigned>(0, Seeds.size()))
      if (!isUsed(ElmIdx))
        return ElmIdx;
    return Seeds.size();
  }
  /// Marks instruction \p I "used" within the bundle. Clients
  /// use this property when assembling a vectorized instruction from
  /// the seeds in a bundle. This allows constant time evaluation
  /// and "removal" from the list.
  void setUsed(Instruction *I) {
    auto It = std::find(begin(), end(), I);
    assert(It != end() && "Instruction not in the bundle!");
    auto Idx = It - begin();
    setUsed(Idx, 1, /*VerifyUnused=*/false);
  }

  void setUsed(unsigned ElementIdx, unsigned Sz = 1, bool VerifyUnused = true) {
    if (ElementIdx + Sz >= UsedLanes.size())
      UsedLanes.resize(ElementIdx + Sz);
    for (unsigned Idx : seq<unsigned>(ElementIdx, ElementIdx + Sz)) {
      assert((!VerifyUnused || !UsedLanes.test(Idx)) &&
             "Already marked as used!");
      UsedLanes.set(Idx);
      UsedLaneCount++;
    }
    NumUnusedBits -= Utils::getNumBits(Seeds[ElementIdx]);
  }
  /// \Returns whether or not \p Element has been used.
  bool isUsed(unsigned Element) const {
    return Element < UsedLanes.size() && UsedLanes.test(Element);
  }
  bool allUsed() const { return UsedLaneCount == Seeds.size(); }
  unsigned getNumUnusedBits() const { return NumUnusedBits; }

  /// \Returns a slice of seed elements, starting at the element \p StartIdx,
  /// with a total size <= \p MaxVecRegBits, or an empty slice if the
  /// requirements cannot be met . If \p ForcePowOf2 is true, then the returned
  /// slice will have a total number of bits that is a power of 2.
  MutableArrayRef<Instruction *>
  getSlice(unsigned StartIdx, unsigned MaxVecRegBits, bool ForcePowOf2);

protected:
  SmallVector<Instruction *> Seeds;
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

/// Specialization of SeedBundle for memory access instructions. Keeps
/// seeds sorted in ascending memory order, which is convenient for slicing
/// these bundles into vectorizable groups.
template <typename LoadOrStoreT> class MemSeedBundle : public SeedBundle {
public:
  explicit MemSeedBundle(SmallVector<Instruction *> &&SV, ScalarEvolution &SE)
      : SeedBundle(std::move(SV)) {
    static_assert(std::is_same<LoadOrStoreT, LoadInst>::value ||
                      std::is_same<LoadOrStoreT, StoreInst>::value,
                  "Expected LoadInst or StoreInst!");
    assert(all_of(Seeds, [](auto *S) { return isa<LoadOrStoreT>(S); }) &&
           "Expected Load or Store instructions!");
    auto Cmp = [&SE](Instruction *I0, Instruction *I1) {
      return Utils::atLowerAddress(cast<LoadOrStoreT>(I0),
                                   cast<LoadOrStoreT>(I1), SE);
    };
    std::sort(Seeds.begin(), Seeds.end(), Cmp);
  }
  explicit MemSeedBundle(LoadOrStoreT *MemI) : SeedBundle(MemI) {
    static_assert(std::is_same<LoadOrStoreT, LoadInst>::value ||
                      std::is_same<LoadOrStoreT, StoreInst>::value,
                  "Expected LoadInst or StoreInst!");
    assert(isa<LoadOrStoreT>(MemI) && "Expected Load or Store!");
  }
  void insert(sandboxir::Instruction *I, ScalarEvolution &SE) {
    assert(isa<LoadOrStoreT>(I) && "Expected a Store or a Load!");
    auto Cmp = [&SE](Instruction *I0, Instruction *I1) {
      return Utils::atLowerAddress(cast<LoadOrStoreT>(I0),
                                   cast<LoadOrStoreT>(I1), SE);
    };
    // Find the first element after I in mem. Then insert I before it.
    insertAt(std::upper_bound(begin(), end(), I, Cmp), I);
  }
};

using StoreSeedBundle = MemSeedBundle<sandboxir::StoreInst>;
using LoadSeedBundle = MemSeedBundle<sandboxir::LoadInst>;

} // namespace llvm::sandboxir
#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SEEDCOLLECTOR_H
