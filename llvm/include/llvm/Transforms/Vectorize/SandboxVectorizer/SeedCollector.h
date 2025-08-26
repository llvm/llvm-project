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
#include "llvm/Support/Compiler.h"
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

  virtual void insert(Instruction *I, ScalarEvolution &SE) = 0;

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
    auto It = llvm::find(*this, I);
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
  LLVM_ABI ArrayRef<Instruction *>
  getSlice(unsigned StartIdx, unsigned MaxVecRegBits, bool ForcePowOf2);

  /// \Returns the number of seed elements in the bundle.
  std::size_t size() const { return Seeds.size(); }

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
  void insert(sandboxir::Instruction *I, ScalarEvolution &SE) override {
    assert(isa<LoadOrStoreT>(I) && "Expected a Store or a Load!");
    auto Cmp = [&SE](Instruction *I0, Instruction *I1) {
      return Utils::atLowerAddress(cast<LoadOrStoreT>(I0),
                                   cast<LoadOrStoreT>(I1), SE);
    };
    // Find the first element after I in mem. Then insert I before it.
    insertAt(llvm::upper_bound(*this, I, Cmp), I);
  }
};

using StoreSeedBundle = MemSeedBundle<sandboxir::StoreInst>;
using LoadSeedBundle = MemSeedBundle<sandboxir::LoadInst>;

/// Class to conveniently track Seeds within SeedBundles. Saves newly collected
/// seeds in the proper bundle. Supports constant-time removal, as seeds and
/// entire bundles are vectorized and marked used to signify removal. Iterators
/// skip bundles that are completely used.
class SeedContainer {
  // Use the same key for different seeds if they are the same type and
  // reference the same pointer, even if at different offsets. This directs
  // potentially vectorizable seeds into the same bundle.
  using KeyT = std::tuple<Value *, Type *, Instruction::Opcode>;
  // Trying to vectorize too many seeds at once is expensive in
  // compilation-time. Use a vector of bundles (all with the same key) to
  // partition the candidate set into more manageable units. Each bundle is
  // size-limited by sbvec-seed-bundle-size-limit.  TODO: There might be a
  // better way to divide these than by simple insertion order.
  using ValT = SmallVector<std::unique_ptr<SeedBundle>>;
  using BundleMapT = MapVector<KeyT, ValT>;
  // Map from {pointer, Type, Opcode} to a vector of bundles.
  BundleMapT Bundles;
  // Allows finding a particular Instruction's bundle.
  DenseMap<Instruction *, SeedBundle *> SeedLookupMap;

  ScalarEvolution &SE;

  template <typename LoadOrStoreT>
  KeyT getKey(LoadOrStoreT *LSI, bool AllowDiffTypes) const;

public:
  SeedContainer(ScalarEvolution &SE) : SE(SE) {}

  class iterator {
    BundleMapT *Map = nullptr;
    BundleMapT::iterator MapIt;
    ValT *Vec = nullptr;
    size_t VecIdx;

  public:
    using difference_type = std::ptrdiff_t;
    using value_type = SeedBundle;
    using pointer = value_type *;
    using reference = value_type &;
    using iterator_category = std::input_iterator_tag;

    /// Iterates over the \p Map of SeedBundle Vectors, starting at \p MapIt,
    /// and \p Vec at \p VecIdx, skipping vectors that are completely
    /// used. Iteration order over the keys {Pointer, Type, Opcode} follows
    /// DenseMap iteration order. For a given key, the vectors of
    /// SeedBundles will be returned in insertion order. As in the
    /// pseudo code below:
    ///
    /// for Key,Value in Bundles
    ///   for SeedBundleVector in Value
    ///     for SeedBundle in SeedBundleVector
    ///        if !SeedBundle.allUsed() ...
    ///
    /// Note that the bundles themselves may have additional ordering, created
    /// by the subclasses by insertAt. The bundles themselves may also have used
    /// instructions.

    // TODO: Range_size counts fully used-bundles. Further, iterating over
    // anything other than the Bundles in a SeedContainer includes used
    // seeds. Rework the iterator logic to clean this up.
    iterator(BundleMapT &Map, BundleMapT::iterator MapIt, ValT *Vec, int VecIdx)
        : Map(&Map), MapIt(MapIt), Vec(Vec), VecIdx(VecIdx) {}
    value_type &operator*() {
      assert(Vec != nullptr && "Already at end!");
      return *(*Vec)[VecIdx];
    }
    // Skip completely used bundles by repeatedly calling operator++().
    void skipUsed() {
      while (Vec && VecIdx < Vec->size() && this->operator*().allUsed())
        ++(*this);
    }
    // Iterators iterate over the bundles
    iterator &operator++() {
      ++VecIdx;
      if (VecIdx >= Vec->size()) {
        assert(MapIt != Map->end() && "Already at end!");
        VecIdx = 0;
        ++MapIt;
        if (MapIt != Map->end())
          Vec = &MapIt->second;
        else {
          Vec = nullptr;
        }
      }
      skipUsed();
      return *this;
    }
    iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }
    bool operator==(const iterator &Other) const {
      assert(Map == Other.Map && "Iterator of different objects!");
      return MapIt == Other.MapIt && VecIdx == Other.VecIdx;
    }
    bool operator!=(const iterator &Other) const { return !(*this == Other); }
  };
  using const_iterator = BundleMapT::const_iterator;
  template <typename LoadOrStoreT>
  void insert(LoadOrStoreT *LSI, bool AllowDiffTypes);
  // To support constant-time erase, these just mark the element used, rather
  // than actually removing them from the bundle.
  LLVM_ABI bool erase(Instruction *I);
  bool erase(const KeyT &Key) { return Bundles.erase(Key); }
  iterator begin() {
    if (Bundles.empty())
      return end();
    auto BeginIt =
        iterator(Bundles, Bundles.begin(), &Bundles.begin()->second, 0);
    BeginIt.skipUsed();
    return BeginIt;
  }
  iterator end() { return iterator(Bundles, Bundles.end(), nullptr, 0); }
  unsigned size() const { return Bundles.size(); }

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

// Explicit instantiations
extern template LLVM_TEMPLATE_ABI void
SeedContainer::insert<LoadInst>(LoadInst *, bool);
extern template LLVM_TEMPLATE_ABI void
SeedContainer::insert<StoreInst>(StoreInst *, bool);

class SeedCollector {
  SeedContainer StoreSeeds;
  SeedContainer LoadSeeds;
  Context &Ctx;
  Context::CallbackID EraseCallbackID;
  /// \Returns the number of SeedBundle groups for all seed types.
  /// This is to be used for limiting compilation time.
  unsigned totalNumSeedGroups() const {
    return StoreSeeds.size() + LoadSeeds.size();
  }

public:
  LLVM_ABI SeedCollector(BasicBlock *BB, ScalarEvolution &SE,
                         bool CollectStores, bool CollectLoads,
                         bool AllowDiffTypes = false);
  LLVM_ABI ~SeedCollector();

  iterator_range<SeedContainer::iterator> getStoreSeeds() {
    return {StoreSeeds.begin(), StoreSeeds.end()};
  }
  iterator_range<SeedContainer::iterator> getLoadSeeds() {
    return {LoadSeeds.begin(), LoadSeeds.end()};
  }
#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_SEEDCOLLECTOR_H
