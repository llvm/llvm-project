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
  using ElementSet = DenseSet<ElementId>;
  using ContainerElementsMap = DenseMap<ContainerId, ElementSet>;

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
  };

private:
  using ElemToSuperNodeMap =
      DenseMap<ContainerId, DenseMap<ElementId, SuperNode *>>;

  using SuperNodeDepsMap = DenseMap<SuperNode *, DenseSet<SuperNode *>>;

  class Coalescer {
  public:
    std::unique_ptr<SuperNode> addOrCreateSuperNode(ContainerElementsMap Defs,
                                                    ContainerElementsMap Deps) {
      auto H = getHash(Deps);
      if (auto *ExistingSN = findCanonicalSuperNode(H, Deps)) {
        for (auto &[Container, Elems] : Defs) {
          auto &DstCElems = ExistingSN->Defs[Container];
          [[maybe_unused]] size_t ExpectedSize =
              DstCElems.size() + Elems.size();
          DstCElems.insert(Elems.begin(), Elems.end());
          assert(DstCElems.size() == ExpectedSize);
        }
        return nullptr;
      }

      auto NewSN =
          std::make_unique<SuperNode>(std::move(Defs), std::move(Deps));
      CanonicalSNs[H].push_back(NewSN.get());
      return NewSN;
    }

    void coalesce(std::vector<std::unique_ptr<SuperNode>> &SNs,
                  ElemToSuperNodeMap &ElemToSN) {
      for (size_t I = 0; I != SNs.size();) {
        auto &SN = SNs[I];
        auto H = getHash(SN->Deps);
        if (auto *CanonicalSN = findCanonicalSuperNode(H, SN->Deps)) {
          for (auto &[Container, Elems] : SN->Defs) {
            CanonicalSN->Defs[Container].insert(Elems.begin(), Elems.end());
            auto &ContainerElemToSN = ElemToSN[Container];
            for (auto &Elem : Elems)
              ContainerElemToSN[Elem] = CanonicalSN;
          }
          std::swap(SN, SNs.back());
          SNs.pop_back();
        } else {
          CanonicalSNs[H].push_back(SN.get());
          ++I;
        }
      }
    }

    template <typename Pred> void remove(Pred &&Remove) {
      for (auto &[Hash, SNs] : CanonicalSNs) {
        bool Found = false;
        for (size_t I = 0; I != SNs.size(); ++I) {
          if (Remove(SNs[I])) {
            std::swap(SNs[I], SNs.back());
            SNs.pop_back();
            Found = true;
            break;
          }
        }
        if (Found) {
          if (SNs.empty())
            CanonicalSNs.erase(Hash);
          break;
        }
      }
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
      // Remove any self-reference.
      SmallVector<ContainerId> ToRemove;
      for (auto &[Container, Elems] : Defs) {
        assert(!Elems.empty() && "Defs for container must not be empty");
        auto I = Deps.find(Container);
        if (I == Deps.end())
          continue;
        auto &DepsForContainer = I->second;
        for (auto &Elem : Elems)
          DepsForContainer.erase(Elem);
        if (DepsForContainer.empty())
          ToRemove.push_back(Container);
      }
      for (auto &Container : ToRemove)
        Deps.erase(Container);
      if (auto SN = C.addOrCreateSuperNode(std::move(Defs), std::move(Deps)))
        SNs.push_back(std::move(SN));
    }
    std::vector<std::unique_ptr<SuperNode>> takeSuperNodes() {
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

  /// Preprocess a list of SuperNodes to remove all intra-SN dependencies.
  static SimplifyResult simplify(std::vector<std::unique_ptr<SuperNode>> SNs) {
    // Build ElemToSN map.
    ElemToSuperNodeMap ElemToSN;
    for (auto &SN : SNs) {
      for (auto &[Container, Elements] : SN->Defs) {
        auto &ContainerElemToSN = ElemToSN[Container];
        for (auto &E : Elements)
          ContainerElemToSN[E] = SN.get();
      }
    }

    SuperNodeDepsMap SuperNodeDeps;
    hoistDeps(SuperNodeDeps, SNs, ElemToSN);
    propagateSuperNodeDeps(SuperNodeDeps);
    sinkDeps(SNs, SuperNodeDeps);

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

    // Collect the PendingSNs whose dep sets are about to be modified.
    std::vector<std::unique_ptr<SuperNode>> ModifiedPendingSNs;
    for (size_t I = 0; I != PendingSNs.size();) {
      auto &SN = PendingSNs[I];
      bool Remove = false;
      for (auto &[Container, Elems] : SN->Deps) {
        auto I = ElemToNewSN.find(Container);
        if (I == ElemToNewSN.end())
          continue;
        for (auto Elem : Elems) {
          if (I->second.contains(Elem)) {
            Remove = true;
            break;
          }
        }
        if (Remove)
          break;
      }
      if (Remove) {
        ModifiedPendingSNs.push_back(std::move(SN));
        std::swap(SN, PendingSNs.back());
        PendingSNs.pop_back();
      } else
        ++I;
    }

    // Remove cycles from the graphs.
    SuperNodeDepsMap SuperNodeDeps;
    hoistDeps(SuperNodeDeps, ModifiedPendingSNs, ElemToNewSN);

    CoalesceToPendingSNs.remove(
        [&](SuperNode *SN) { return SuperNodeDeps.count(SN); });

    hoistDeps(SuperNodeDeps, NewSNs, ElemToPendingSN);
    propagateSuperNodeDeps(SuperNodeDeps);
    sinkDeps(NewSNs, SuperNodeDeps);
    sinkDeps(ModifiedPendingSNs, SuperNodeDeps);

    // Process supernodes. Pending first, since we'll update PendingSNs when we
    // incorporate NewSNs.
    std::vector<std::unique_ptr<SuperNode>> ReadyNodes, FailedNodes;
    processReadyOrFailed(ModifiedPendingSNs, ReadyNodes, FailedNodes,
                         SuperNodeDeps, ElemToPendingSN, FailedSNs);
    processReadyOrFailed(NewSNs, ReadyNodes, FailedNodes, SuperNodeDeps,
                         ElemToNewSN, FailedSNs);

    CoalesceToPendingSNs.coalesce(ModifiedPendingSNs, ElemToPendingSN);
    CoalesceToPendingSNs.coalesce(NewSNs, ElemToPendingSN);

    // Integrate remaining ModifiedPendingSNs and NewSNs into PendingSNs.
    for (auto &SN : ModifiedPendingSNs)
      PendingSNs.push_back(std::move(SN));

    // Update ElemToPendingSN for the remaining elements.
    for (auto &SN : NewSNs) {
      for (auto &[Container, Elems] : SN->Defs) {
        auto &Row = ElemToPendingSN[Container];
        for (auto &Elem : Elems)
          Row[Elem] = SN.get();
      }
      PendingSNs.push_back(std::move(SN));
    }

    return {std::move(ReadyNodes), std::move(FailedNodes)};
  }

  /// Identify the given symbols as Failed.
  /// The elements of the Failed map will not be included in the returned
  /// result, so clients should take whatever actions are needed to mark
  /// this as failed in their external representation.
  std::vector<std::unique_ptr<SuperNode>>
  fail(const ContainerElementsMap &Failed) {
    std::vector<std::unique_ptr<SuperNode>> FailedSNs;

    for (size_t I = 0; I != PendingSNs.size();) {
      auto &PendingSN = PendingSNs[I];
      bool FailPendingSN = false;
      for (auto &[Container, Elems] : PendingSN->Deps) {
        if (FailPendingSN)
          break;
        auto I = Failed.find(Container);
        if (I == Failed.end())
          continue;
        for (auto &Elem : Elems) {
          if (I->second.count(Elem)) {
            FailPendingSN = true;
            break;
          }
        }
      }
      if (FailPendingSN) {
        FailedSNs.push_back(std::move(PendingSN));
        PendingSN = std::move(PendingSNs.back());
        PendingSNs.pop_back();
      } else
        ++I;
    }

    for (auto &SN : FailedSNs) {
      CoalesceToPendingSNs.remove(
          [&](SuperNode *SNC) { return SNC == SN.get(); });
      for (auto &[Container, Elems] : SN->Defs) {
        assert(ElemToPendingSN.count(Container));
        auto &CElems = ElemToPendingSN[Container];
        for (auto &Elem : Elems)
          CElems.erase(Elem);
        if (CElems.empty())
          ElemToPendingSN.erase(Container);
      }
    }

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
  //
  // For all dependencies in SNs, if the corresponding node is defined in
  // ElemToSN then remove the individual dependency and add the record the
  // dependency on the corresponding supernode in SuperNodeDeps.
  static void hoistDeps(SuperNodeDepsMap &SuperNodeDeps,
                        std::vector<std::unique_ptr<SuperNode>> &SNs,
                        ElemToSuperNodeMap &ElemToSN) {
    for (auto &SN : SNs) {
      auto &SNDeps = SuperNodeDeps[SN.get()];
      for (auto &[DefContainer, DefElems] : ElemToSN) {
        auto I = SN->Deps.find(DefContainer);
        if (I == SN->Deps.end())
          continue;
        for (auto &[DefElem, DefSN] : DefElems)
          if (I->second.erase(DefElem))
            SNDeps.insert(DefSN);
        if (I->second.empty())
          SN->Deps.erase(I);
      }
    }
  }

  // Compute transitive closure of deps for each node.
  static void propagateSuperNodeDeps(SuperNodeDepsMap &SuperNodeDeps) {
    for (auto &[SN, Deps] : SuperNodeDeps) {
      DenseSet<SuperNode *> Reachable({SN});
      SmallVector<SuperNode *> Worklist(Deps.begin(), Deps.end());

      while (!Worklist.empty()) {
        auto *DepSN = Worklist.pop_back_val();
        if (!Reachable.insert(DepSN).second)
          continue;
        auto I = SuperNodeDeps.find(DepSN);
        if (I == SuperNodeDeps.end())
          continue;
        for (auto *DepSNDep : I->second)
          Worklist.push_back(DepSNDep);
      }

      Deps = std::move(Reachable);
    }
  }

  // Sink SuperNode dependencies back to dependencies on individual nodes.
  static void sinkDeps(std::vector<std::unique_ptr<SuperNode>> &SNs,
                       SuperNodeDepsMap &SuperNodeDeps) {
    for (auto &SN : SNs) {
      auto I = SuperNodeDeps.find(SN.get());
      if (I == SuperNodeDeps.end())
        continue;

      for (auto *DepSN : I->second)
        for (auto &[Container, Elems] : DepSN->Deps)
          SN->Deps[Container].insert(Elems.begin(), Elems.end());
    }
  }

  template <typename GetExternalStateFn>
  static std::vector<SuperNode *>
  processExternalDeps(std::vector<std::unique_ptr<SuperNode>> &SNs,
                      GetExternalStateFn &GetExternalState) {
    std::vector<SuperNode *> FailedSNs;
    for (auto &SN : SNs) {
      bool SNHasError = false;
      SmallVector<ContainerId> ContainersToRemove;
      for (auto &[Container, Elems] : SN->Deps) {
        SmallVector<ElementId> ElemToRemove;
        for (auto &Elem : Elems) {
          switch (GetExternalState(Container, Elem)) {
          case ExternalState::None:
            break;
          case ExternalState::Ready:
            ElemToRemove.push_back(Elem);
            break;
          case ExternalState::Failed:
            ElemToRemove.push_back(Elem);
            SNHasError = true;
            break;
          }
        }
        for (auto &Elem : ElemToRemove)
          Elems.erase(Elem);
        if (Elems.empty())
          ContainersToRemove.push_back(Container);
      }
      for (auto &Container : ContainersToRemove)
        SN->Deps.erase(Container);
      if (SNHasError)
        FailedSNs.push_back(SN.get());
    }

    return FailedSNs;
  }

  void processReadyOrFailed(std::vector<std::unique_ptr<SuperNode>> &SNs,
                            std::vector<std::unique_ptr<SuperNode>> &Ready,
                            std::vector<std::unique_ptr<SuperNode>> &Failed,
                            SuperNodeDepsMap &SuperNodeDeps,
                            ElemToSuperNodeMap &ElemToSNs,
                            std::vector<SuperNode *> FailedSNs) {
    for (size_t I = 0; I != SNs.size();) {
      auto &SN = SNs[I];

      bool SNFailed = false;
      assert(SuperNodeDeps.count(SN.get()));
      auto &SNSuperNodeDeps = SuperNodeDeps[SN.get()];
      for (auto *FailedSN : FailedSNs) {
        if (FailedSN == SN.get() || SNSuperNodeDeps.count(FailedSN)) {
          SNFailed = true;
          break;
        }
      }

      bool SNReady = SN->Deps.empty();

      if (SNReady || SNFailed) {
        auto &NodeList = SNFailed ? Failed : Ready;
        NodeList.push_back(std::move(SN));
        std::swap(SN, SNs.back());
        SNs.pop_back();
      } else
        ++I;
    }
  }

  std::vector<std::unique_ptr<SuperNode>> PendingSNs;
  ElemToSuperNodeMap ElemToPendingSN;
  Coalescer CoalesceToPendingSNs;
};

} // namespace llvm::orc::detail

#endif // LLVM_EXECUTIONENGINE_ORC_WAITINGONGRAPH_H
