//===------ WaitingOnGraph.h - ORC symbol dependence graph ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines WaitingOnGraph and related utilities.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_EXECUTIONENGINE_ORC_WAITINGONGRAPH_H
#define LLVM_EXECUTIONENGINE_ORC_WAITINGONGRAPH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <vector>

namespace llvm::orc::detail {

class WaitingOnGraphTest;

/// WaitingOnGraph class template.
///
/// This type is intended to provide efficient dependence tracking for Symbols
/// in an ORC program.
///
/// WaitingOnGraph models a directed graph with four partitions:
///   1. Not-yet-emitted nodes: Nodes identified as waited-on in an emit
///      operation.
///   2. Emitted nodes: Nodes emitted and waiting on some non-empty set of
///      other nodes.
///   3. Ready nodes: Nodes emitted and not waiting on any other nodes
///      (either because they weren't waiting on any nodes when they were
///      emitted, or because all transitively waited-on nodes have since
///      been emitted).
///   4. Failed nodes: Nodes that have been marked as failed-to-emit, and
///      nodes that were found to transitively wait-on some failed node.
///
/// Nodes are added to the graph by *emit* and *fail* operations.
///
/// The *emit* operation takes a bipartite *local dependence graph* as an
/// argument and returns...
///   a. the set of nodes (both existing and newly added from the local
///      dependence graph) whose waiting-on set is the empty set, and...
///   b. the set of newly added nodes that are found to depend on failed
///      nodes.
///
/// The *fail* operation takes a set of failed nodes and returns the set of
/// Emitted nodes that were waiting on the failed nodes.
///
/// The concrete representation adopts several approaches for efficiency:
///
/// 1. Only *Emitted* and *Not-yet-emitted* nodes are represented explicitly.
///    *Ready* and *Failed* nodes are represented by the values returned by the
///    GetExternalStateFn argument to *emit*.
///
/// 2. Labels are (*Container*, *Element*) pairs that are intended to represent
///    ORC symbols (ORC uses types Container = JITDylib,
///    Element = NonOwningSymbolStringPtr). The internal representation of the
///    graph is optimized on the assumption that there are many more Elements
///    (symbol names) than Containers (JITDylibs) used to construct the labels.
///    (Consider for example the common case where most JIT'd code is placed in
///    a single "main" JITDylib).
///
/// 3. The data structure stores *SuperNodes* which have multiple labels. This
///    reduces the number of nodes and edges in the graph in the common case
///    where many JIT symbols have the same set of dependencies. SuperNodes are
///    coalesced when their dependence sets become equal.
///
/// 4. The *simplify* method can be applied to an initial *local dependence
///    graph* (as a list of SuperNodes) to eliminate any internal dependence
///    relationships that would have to be propagated internally by *emit*.
///    Access to the WaitingOnGraph is assumed to be guarded by a mutex (ORC
///    will access it from multiple threads) so this allows some pre-processing
///    to be performed outside the mutex.
template <typename ContainerIdT, typename ElementIdT> class WaitingOnGraph {
  friend class WaitingOnGraphTest;

public:
  using ContainerId = ContainerIdT;
  using ElementId = ElementIdT;

  class ElementSet : public DenseSet<ElementId> {
    friend class ElementSetTest;

  public:
    using DenseSet<ElementId>::DenseSet;

    /// Merge the elements of Other into this set. Returns true if any new
    /// elements are added.
    bool merge(const ElementSet &Other, bool AssertNoOverlap = false) {
      size_t OrigSize = this->size();
      this->insert(Other.begin(), Other.end());
      assert((!AssertNoOverlap || this->size() == (OrigSize + Other.size())) &&
             "merge of overlapping elements");
      return this->size() != OrigSize;
    }

    /// Remove all elements in Other from this set. Returns true if any
    /// elements were removed.
    bool remove(const ElementSet &Other) {
      size_t OrigSize = this->size();

      // Early out for empty sets.
      if (OrigSize == 0 || Other.empty())
        return false;

      // TODO: Tweak condition to account for SmallVector cost. We may want to
      //       prefer iterating over elements if the size difference is small.
      if (OrigSize > Other.size()) {
        for (auto &Elem : Other)
          this->erase(Elem);
      } else {
        SmallVector<ElementId> ToRemove;
        for (auto &Elem : *this)
          if (Other.count(Elem))
            ToRemove.push_back(Elem);
        for (auto &Elem : ToRemove)
          this->erase(Elem);
      }
      return this->size() < OrigSize;
    }

    /// Remove all elements for which Pred returns true.
    /// Returns true if any elements were removed.
    template <typename Pred> bool remove_if(Pred &&P) {
      if (this->empty())
        return false;

      SmallVector<ElementId> ToRemove;
      for (auto &Elem : *this)
        if (P(Elem))
          ToRemove.push_back(Elem);

      for (auto &Elem : ToRemove)
        this->erase(Elem);

      return !ToRemove.empty();
    }
  };

  class ContainerElementsMap : public DenseMap<ContainerId, ElementSet> {
    friend class ContainerElementsMapTest;

  public:
    using DenseMap<ContainerId, ElementSet>::DenseMap;

    /// Merge the elements of Other into this map. Returns true if any new
    /// elements are added.
    bool merge(const ContainerElementsMap &Other,
               bool AssertNoElementsOverlap = false) {
      bool Changed = false;
      for (auto &[Container, Elements] : Other)
        Changed |= (*this)[Container].merge(Elements, AssertNoElementsOverlap);
      return Changed;
    }

    /// Remove all elements in Other from this map. Returns true if any
    /// elements were removed.
    bool remove(const ContainerElementsMap &Other) {
      bool Changed = false;
      for (auto &[Container, Elements] : Other) {
        assert(!Elements.empty() && "Stale row for Container in Other");
        auto I = this->find(Container);
        if (I == this->end())
          continue;
        Changed |= I->second.remove(Elements);
        if (I->second.empty())
          this->erase(Container);
      }
      return Changed;
    }

    /// Call V on each (Container, Elements) pair in this map.
    ///
    /// V should return true if it modifies any elements.
    ///
    /// Returns true if V returns true for any pair.
    template <typename Visitor> bool visit(Visitor &&V) {
      if (this->empty())
        return false;

      bool Changed = false;
      SmallVector<ContainerId> ToRemove;
      for (auto &[Container, Elements] : *this) {
        assert(!Elements.empty() && "empty row for container");
        if (V(Container, Elements)) {
          Changed = true;
          if (Elements.empty())
            ToRemove.push_back(Container);
        }
      }

      for (auto &Container : ToRemove)
        this->erase(Container);

      return Changed;
    }
  };

  class SuperNode;

private:
  using ElemToSuperNodeMap =
      DenseMap<ContainerId, DenseMap<ElementId, SuperNode *>>;

  using SuperNodeDepsMap = DenseMap<SuperNode *, DenseSet<SuperNode *>>;

public:
  class SuperNode {
    friend class WaitingOnGraph;
    friend class WaitingOnGraphTest;

  public:
    SuperNode(ContainerElementsMap Defs, ContainerElementsMap Deps)
        : Defs(std::move(Defs)), Deps(std::move(Deps)) {}
    ContainerElementsMap &defs() { return Defs; }
    const ContainerElementsMap &defs() const { return Defs; }
    ContainerElementsMap &deps() { return Deps; }
    const ContainerElementsMap &deps() const { return Deps; }

  private:
    ContainerElementsMap Defs;
    ContainerElementsMap Deps;

    ElemToSuperNodeMap *RegisteredElemToSN = nullptr;

    /// Add a mapping from the Defs in this SuperNode to SN (which may or may
    /// not be the same as this).
    void mapDefsTo(ElemToSuperNodeMap &ElemToSN, SuperNode *SN,
                   bool AbandonOldMapping = false) {
      assert(!Defs.empty() && "Empty defs!?");
      for (auto &[Container, Elements] : Defs) {
        assert(!Elements.empty() && "Empty elements for container?");
        auto &ContainerElemToSN = ElemToSN[Container];
        for (auto &Elem : Elements)
          ContainerElemToSN[Elem] = SN;
      }
      assert((AbandonOldMapping || !SN->RegisteredElemToSN ||
              SN->RegisteredElemToSN == &ElemToSN) &&
             "SN defs split across maps");
      SN->RegisteredElemToSN = &ElemToSN;
    }

    /// Add a mapping from the Defs in this SuperNode to this.
    /// (Equivalent to `SN.mapDefsTo(ElemToSN, &SN);`)
    void mapDefsToThis(ElemToSuperNodeMap &ElemToSN,
                       bool AbandonOldMapping = false) {
      mapDefsTo(ElemToSN, this, AbandonOldMapping);
    }

    /// Remove a mapping from the Defs in this SuperNode from the registered
    /// ElemToSuperNodeMap. The mapping must already exist.
    void unmapDefsFromThis() {
      assert(RegisteredElemToSN && "No registered ElemToSuperNodeMap");
      for (auto &[Container, Elements] : Defs) {
        auto I = RegisteredElemToSN->find(Container);
        assert(I != RegisteredElemToSN->end() && "Container not in map");
        auto &ContainerElemToSN = I->second;
        for (auto &Elem : Elements) {
          assert(ContainerElemToSN[Elem] == this && "Mapping not present");
          ContainerElemToSN.erase(Elem);
        }
        if (ContainerElemToSN.empty())
          RegisteredElemToSN->erase(I);
      }
      RegisteredElemToSN = nullptr;
    }

    /// For all Defs of this node that are defined by some node in ElemToSN,
    /// remove the Def from this map and add this SuperNode to the list of
    /// dependants of the defining node.
    ///
    /// Returns true if any elements were removed.
    bool hoistDeps(SuperNodeDepsMap &SuperNodeDeps,
                   ElemToSuperNodeMap &ElemToSN) {
      return Deps.visit([&](ContainerId &Container, ElementSet &Elements) {
        auto I = ElemToSN.find(Container);
        if (I == ElemToSN.end())
          return false;

        auto &ContainerElemToSN = I->second;
        return Elements.remove_if([&](const ElementId &Elem) {
          auto J = ContainerElemToSN.find(Elem);
          if (J == ContainerElemToSN.end())
            return false;

          auto *DefSN = J->second;
          if (DefSN != this)
            SuperNodeDeps[DefSN].insert(this);
          return true;
        });
      });
    }
  };

private:
  /// Fast visit with removal.
  ///
  /// Visits the elements of Vec, removing each element for which V returns
  /// true.
  ///
  /// This is O(1) in the number of elements removed, but does not preserve
  /// element order.
  template <typename Vector, typename Visitor>
  static void visitWithRemoval(Vector &Vec, Visitor &&V) {
    for (size_t I = 0; I != Vec.size();) {
      if (V(Vec[I])) {
        if (I != Vec.size() - 1)
          std::swap(Vec[I], Vec.back());
        Vec.pop_back();
      } else
        ++I;
    }
  }

  class Coalescer {
  public:
    std::unique_ptr<SuperNode> addOrCreateSuperNode(ContainerElementsMap Defs,
                                                    ContainerElementsMap Deps) {
      auto H = getHash(Deps);
      if (auto *ExistingSN = findCanonicalSuperNode(H, Deps)) {
        ExistingSN->Defs.merge(Defs, /* AssertNoElementsOverlap */ true);
        return nullptr;
      }

      auto NewSN =
          std::make_unique<SuperNode>(std::move(Defs), std::move(Deps));
      CanonicalSNs[H].push_back(NewSN.get());
      assert(!SNHashes.count(NewSN.get()));
      SNHashes[NewSN.get()] = H;
      return NewSN;
    }

    void coalesce(std::vector<std::unique_ptr<SuperNode>> &SNs,
                  ElemToSuperNodeMap &ElemToSN,
                  bool AbandonOldMapping = false) {
      visitWithRemoval(SNs, [&](std::unique_ptr<SuperNode> &SN) {
        assert(!SNHashes.count(SN.get()) &&
               "Elements of SNs should be new to the coalescer");
        auto H = getHash(SN->Deps);
        if (auto *CanonicalSN = findCanonicalSuperNode(H, SN->Deps)) {
          SN->mapDefsTo(ElemToSN, CanonicalSN, AbandonOldMapping);
          CanonicalSN->Defs.merge(SN->Defs, /* AssertNoElementsOverlap */ true);
          return true;
        }
        CanonicalSNs[H].push_back(SN.get());
        SNHashes[SN.get()] = H;
        return false;
      });
    }

    /// Remove all coalescing information.
    ///
    /// This resets the Coalescer to the same functional state that it was
    /// constructed in.
    void clear() {
      CanonicalSNs.clear();
      SNHashes.clear();
    }

    /// Remove the given node from the Coalescer.
    void erase(SuperNode *SN) {
      hash_code H;

      {
        // Look up hash. We expect to find it in SNHashes.
        auto I = SNHashes.find(SN);
        assert(I != SNHashes.end() && "SN not tracked by coalescer");
        H = I->second;
        SNHashes.erase(I);
      }

      // Now remove from CanonicalSNs.
      auto I = CanonicalSNs.find(H);
      assert(I != CanonicalSNs.end() && "Hash not in CanonicalSNs");
      auto &SNs = I->second;

      size_t J = 0;
      for (; J != SNs.size(); ++J)
        if (SNs[J] == SN)
          break;

      assert(J < SNs.size() && "SN not in CanonicalSNs map");
      std::swap(SNs[J], SNs.back());
      SNs.pop_back();

      if (SNs.empty())
        CanonicalSNs.erase(I);
    }

  private:
    hash_code getHash(const ContainerElementsMap &M) {
      SmallVector<ContainerId> SortedContainers;
      SortedContainers.reserve(M.size());
      for (auto &[Container, Elems] : M)
        SortedContainers.push_back(Container);
      llvm::sort(SortedContainers);
      hash_code Hash(0);
      for (auto &Container : SortedContainers) {
        auto &ContainerElems = M.at(Container);
        SmallVector<ElementId> SortedElems(ContainerElems.begin(),
                                           ContainerElems.end());
        llvm::sort(SortedElems);
        Hash = hash_combine(Hash, Container, hash_combine_range(SortedElems));
      }
      return Hash;
    }

    SuperNode *findCanonicalSuperNode(hash_code H,
                                      const ContainerElementsMap &M) {
      for (auto *SN : CanonicalSNs[H])
        if (SN->Deps == M)
          return SN;
      return nullptr;
    }

    DenseMap<hash_code, SmallVector<SuperNode *>> CanonicalSNs;
    DenseMap<SuperNode *, hash_code> SNHashes;
  };

public:
  /// Build SuperNodes from (definition-set, dependence-set) pairs.
  ///
  /// Coalesces definition-sets with identical dependence-sets.
  class SuperNodeBuilder {
  public:
    void add(ContainerElementsMap Defs, ContainerElementsMap Deps) {
      if (Defs.empty())
        return;
      Deps.remove(Defs); // Remove any self-reference.
      if (auto SN = C.addOrCreateSuperNode(std::move(Defs), std::move(Deps)))
        SNs.push_back(std::move(SN));
    }
    std::vector<std::unique_ptr<SuperNode>> takeSuperNodes() {
      C.clear();
      return std::move(SNs);
    }

  private:
    Coalescer C;
    std::vector<std::unique_ptr<SuperNode>> SNs;
  };

  class SimplifyResult {
    friend class WaitingOnGraph;
    friend class WaitingOnGraphTest;

  public:
    const std::vector<std::unique_ptr<SuperNode>> &superNodes() const {
      return SNs;
    }

  private:
    SimplifyResult(std::vector<std::unique_ptr<SuperNode>> SNs,
                   ElemToSuperNodeMap ElemToSN)
        : SNs(std::move(SNs)), ElemToSN(std::move(ElemToSN)) {}
    std::vector<std::unique_ptr<SuperNode>> SNs;
    ElemToSuperNodeMap ElemToSN;
  };

  class OpRecorder {
  public:
    virtual ~OpRecorder() = default;
    virtual void
    recordSimplify(const std::vector<std::unique_ptr<SuperNode>> &SNs) = 0;
    virtual void recordFail(const ContainerElementsMap &Failed) = 0;
    virtual void recordEnd() = 0;
  };

  /// Preprocess a list of SuperNodes to remove all intra-SN dependencies.
  static SimplifyResult simplify(std::vector<std::unique_ptr<SuperNode>> SNs,
                                 OpRecorder *Rec = nullptr) {
    if (Rec)
      Rec->recordSimplify(SNs);

    // Build ElemToSN map.
    ElemToSuperNodeMap ElemToSN;
    for (auto &SN : SNs)
      SN->mapDefsToThis(ElemToSN);

    SuperNodeDepsMap SuperNodeDeps;
    hoistDeps(SNs, SuperNodeDeps, ElemToSN);
    propagateDeps(SuperNodeDeps);

    // Pre-coalesce nodes.
    Coalescer().coalesce(SNs, ElemToSN);

    return {std::move(SNs), std::move(ElemToSN)};
  }

  struct EmitResult {
    std::vector<std::unique_ptr<SuperNode>> Ready;
    std::vector<std::unique_ptr<SuperNode>> Failed;
  };

  enum class ExternalState { None, Ready, Failed };

  /// Add the given SuperNodes to the graph, returning any SuperNodes that
  /// move to the Ready or Failed states as a result.
  /// The GetExternalState function is used to represent SuperNodes that have
  /// already become Ready or Failed (since such nodes are not explicitly
  /// represented in the graph).
  template <typename GetExternalStateFn>
  EmitResult emit(SimplifyResult SR, GetExternalStateFn &&GetExternalState) {
    auto NewSNs = std::move(SR.SNs);
    auto ElemToNewSN = std::move(SR.ElemToSN);

    // First process any dependencies on nodes with external state.
    auto FailedSNs = processExternalDeps(NewSNs, GetExternalState);

    SuperNodeDepsMap SuperNodeDeps;

    // Collect the PendingSNs whose dep sets are about to be modified.
    std::vector<std::unique_ptr<SuperNode>> ModifiedPendingSNs;
    visitWithRemoval(PendingSNs, [&](std::unique_ptr<SuperNode> &SN) {
      if (SN->hoistDeps(SuperNodeDeps, ElemToNewSN)) {
        ModifiedPendingSNs.push_back(std::move(SN));
        return true;
      }
      return false;
    });

    // Remove SNs whose deps have been modified from the coalescer.
    for (auto &SN : ModifiedPendingSNs)
      CoalesceToPendingSNs.erase(SN.get());

    hoistDeps(NewSNs, SuperNodeDeps, ElemToPendingSN);
    propagateDeps(SuperNodeDeps);

    propagateFailures(FailedSNs, SuperNodeDeps);

    // Process supernodes. Pending first, since we'll update PendingSNs when we
    // incorporate NewSNs.
    std::vector<std::unique_ptr<SuperNode>> ReadyNodes, FailedNodes;
    processReadyOrFailed(ModifiedPendingSNs, ReadyNodes, FailedNodes,
                         SuperNodeDeps, FailedSNs, true);
    processReadyOrFailed(NewSNs, ReadyNodes, FailedNodes, SuperNodeDeps,
                         FailedSNs, false);

    CoalesceToPendingSNs.coalesce(ModifiedPendingSNs, ElemToPendingSN);
    CoalesceToPendingSNs.coalesce(NewSNs, ElemToPendingSN,
                                  /* AbandonOldMapping = */ true);

    // Integrate remaining ModifiedPendingSNs and NewSNs into PendingSNs.
    for (auto &SN : ModifiedPendingSNs)
      PendingSNs.push_back(std::move(SN));

    // Update ElemToPendingSN for the remaining elements.
    for (auto &SN : NewSNs) {
      SN->mapDefsToThis(ElemToPendingSN, /* AbandonOldMapping = */ true);
      PendingSNs.push_back(std::move(SN));
    }

    return {std::move(ReadyNodes), std::move(FailedNodes)};
  }

  /// Identify the given symbols as Failed.
  /// The elements of the Failed map will not be included in the returned
  /// result, so clients should take whatever actions are needed to mark
  /// this as failed in their external representation.
  std::vector<std::unique_ptr<SuperNode>>
  fail(const ContainerElementsMap &Failed, OpRecorder *Rec = nullptr) {
    if (Rec)
      Rec->recordFail(Failed);

    std::vector<std::unique_ptr<SuperNode>> FailedSNs;

    visitWithRemoval(PendingSNs, [&](std::unique_ptr<SuperNode> &SN) {
      for (auto &[Container, Elements] : SN->Deps) {
        auto I = Failed.find(Container);
        if (I == Failed.end())
          continue;

        auto &FailedElems = I->second;
        for (auto &Elem : Elements) {
          if (FailedElems.count(Elem)) {
            CoalesceToPendingSNs.erase(SN.get());
            SN->unmapDefsFromThis();
            FailedSNs.push_back(std::move(SN));
            return true;
          }
        }
      }
      return false;
    });

    return FailedSNs;
  }

  bool validate(raw_ostream &Log) {
    bool AllGood = true;
    auto ErrLog = [&]() -> raw_ostream & {
      AllGood = false;
      return Log;
    };

    size_t DefCount = 0;
    for (auto &PendingSN : PendingSNs) {
      if (PendingSN->Deps.empty())
        ErrLog() << "Pending SN " << PendingSN.get() << " has empty dep set.\n";
      else {
        bool BadElem = false;
        for (auto &[Container, Elems] : PendingSN->Deps) {
          auto I = ElemToPendingSN.find(Container);
          if (I == ElemToPendingSN.end())
            continue;
          if (Elems.empty())
            ErrLog() << "Pending SN " << PendingSN.get()
                     << " has dependence map entry for " << Container
                     << " with empty element set.\n";
          for (auto &Elem : Elems) {
            if (I->second.count(Elem)) {
              ErrLog() << "Pending SN " << PendingSN.get()
                       << " has dependence on emitted element ( " << Container
                       << ", " << Elem << ")\n";
              BadElem = true;
              break;
            }
          }
          if (BadElem)
            break;
        }
      }

      for (auto &[Container, Elems] : PendingSN->Defs) {
        if (Elems.empty())
          ErrLog() << "Pending SN " << PendingSN.get()
                   << " has def map entry for " << Container
                   << " with empty element set.\n";
        DefCount += Elems.size();
        auto I = ElemToPendingSN.find(Container);
        if (I == ElemToPendingSN.end())
          ErrLog() << "Pending SN " << PendingSN.get() << " has "
                   << Elems.size() << " defs in container " << Container
                   << " not covered by ElemsToPendingSN.\n";
        else {
          for (auto &Elem : Elems) {
            auto J = I->second.find(Elem);
            if (J == I->second.end())
              ErrLog() << "Pending SN " << PendingSN.get() << " has element ("
                       << Container << ", " << Elem
                       << ") not covered by ElemsToPendingSN.\n";
            else if (J->second != PendingSN.get())
              ErrLog() << "ElemToPendingSN value invalid for (" << Container
                       << ", " << Elem << ")\n";
          }
        }
      }
    }

    size_t DefCount2 = 0;
    for (auto &[Container, Elems] : ElemToPendingSN)
      DefCount2 += Elems.size();

    assert(DefCount2 >= DefCount);
    if (DefCount2 != DefCount)
      ErrLog() << "ElemToPendingSN contains extra elements.\n";

    return AllGood;
  }

private:
  // Replace individual dependencies with supernode dependencies.
  static void hoistDeps(std::vector<std::unique_ptr<SuperNode>> &SNs,
                        SuperNodeDepsMap &SuperNodeDeps,
                        ElemToSuperNodeMap &ElemToSN) {
    // For all SNs...
    for (auto &SN : SNs)
      SN->hoistDeps(SuperNodeDeps, ElemToSN);
  }

  // Compute transitive closure of deps for each node.
  static void propagateDeps(SuperNodeDepsMap &SuperNodeDeps) {

    // Early exit for self-contained emits.
    if (SuperNodeDeps.empty())
      return;

    SmallVector<SuperNode *> Worklist;
    Worklist.reserve(SuperNodeDeps.size());
    for (auto &[SN, SNDependants] : SuperNodeDeps)
      Worklist.push_back(SN);

    while (true) {
      DenseSet<SuperNode *> ToVisitNext;

      // TODO: See if topo-sorting worklist improves convergence.

      while (!Worklist.empty()) {
        auto *SN = Worklist.pop_back_val();
        auto I = SuperNodeDeps.find(SN);
        if (I == SuperNodeDeps.end())
          continue;

        for (auto *DependantSN : I->second)
          if (DependantSN->Deps.merge(SN->Deps))
            ToVisitNext.insert(DependantSN);
      }

      if (ToVisitNext.empty())
        break;

      Worklist.append(ToVisitNext.begin(), ToVisitNext.end());
    }
  }

  static void propagateFailures(DenseSet<SuperNode *> &FailedNodes,
                                SuperNodeDepsMap &SuperNodeDeps) {
    if (FailedNodes.empty())
      return;

    SmallVector<SuperNode *> Worklist(FailedNodes.begin(), FailedNodes.end());

    while (!Worklist.empty()) {
      auto *SN = Worklist.pop_back_val();
      auto I = SuperNodeDeps.find(SN);
      if (I == SuperNodeDeps.end())
        continue;

      for (auto *DependantSN : I->second)
        if (FailedNodes.insert(DependantSN).second)
          Worklist.push_back(DependantSN);
    }
  }

  template <typename GetExternalStateFn>
  static DenseSet<SuperNode *>
  processExternalDeps(std::vector<std::unique_ptr<SuperNode>> &SNs,
                      GetExternalStateFn &GetExternalState) {
    DenseSet<SuperNode *> FailedSNs;
    for (auto &SN : SNs)
      SN->Deps.visit([&](ContainerId &Container, ElementSet &Elements) {
        return Elements.remove_if([&](ElementId &Elem) {
          switch (GetExternalState(Container, Elem)) {
          case ExternalState::None:
            return false;
          case ExternalState::Ready:
            return true;
          case ExternalState::Failed:
            FailedSNs.insert(SN.get());
            return true;
          };
          llvm_unreachable("Unknown ExternalState enum");
        });
      });

    return FailedSNs;
  }

  void processReadyOrFailed(std::vector<std::unique_ptr<SuperNode>> &SNs,
                            std::vector<std::unique_ptr<SuperNode>> &Ready,
                            std::vector<std::unique_ptr<SuperNode>> &Failed,
                            SuperNodeDepsMap &SuperNodeDeps,
                            const DenseSet<SuperNode *> &FailedSNs,
                            bool UnmapFromElemToSN) {

    visitWithRemoval(SNs, [&](std::unique_ptr<SuperNode> &SN) {
      bool SNFailed = FailedSNs.count(SN.get());
      bool SNReady = SN->Deps.empty();

      if (SNReady || SNFailed) {
        if (UnmapFromElemToSN)
          SN->unmapDefsFromThis();
        auto &ToList = SNFailed ? Failed : Ready;
        ToList.push_back(std::move(SN));
        return true;
      }
      return false;
    });
  }

  std::vector<std::unique_ptr<SuperNode>> PendingSNs;
  ElemToSuperNodeMap ElemToPendingSN;
  Coalescer CoalesceToPendingSNs;
};

} // namespace llvm::orc::detail

#endif // LLVM_EXECUTIONENGINE_ORC_WAITINGONGRAPH_H
