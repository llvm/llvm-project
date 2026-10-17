//===- SplitModuleCG.h - Split a module by its call graph -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the llvm::SplitModuleCG class, which splits a module
// into partitions based on its call graph, so that the partitions can be
// optimized and codegen'd in parallel.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
#define LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstructionCost.h"

#include <map>
#include <memory>

namespace llvm {

class LLVMContext;
class SimplifiedCallGraph;
class SimplifiedCallGraphNode;

/// A simplified view of the LLVM CallGraph used by SplitModuleCG to drive
/// callgraph-based module partitioning.
///
/// SimplifiedCallGraph drops the function-instruction-level details that the
/// full CallGraph carries and keeps only the information needed for
/// partitioning decisions:
///   - The set of functions in the module (one SimplifiedCallGraphNode each).
///   - The static call edges between them.
///   - A reference count (NumReferences) recording how many other functions
///     call a given function. Functions with a reference count of zero are
///     treated as call-graph roots during partitioning.
///
/// The simplified graph is built once (in the constructor) and is
/// consumed by SplitModuleCG::createWorkList to discover roots and their
/// transitive dependencies.
class SimplifiedCallGraph {
  using FunctionMapTy =
      std::map<const Function *, std::unique_ptr<SimplifiedCallGraphNode>>;

  /// A map from \c Function* to \c SimplifiedCallGraphNode*.
  FunctionMapTy FunctionMap;

public:
  LLVM_ABI explicit SimplifiedCallGraph(CallGraph &CG);
  ~SimplifiedCallGraph() = default;

  using iterator = FunctionMapTy::iterator;
  using const_iterator = FunctionMapTy::const_iterator;

  /// Iterates over all (Function*, SimplifiedCallGraphNode) pairs in the
  /// call graph.
  iterator begin() { return FunctionMap.begin(); }
  iterator end() { return FunctionMap.end(); }
  const_iterator begin() const { return FunctionMap.begin(); }
  const_iterator end() const { return FunctionMap.end(); }

  /// Iterates over all SimplifiedCallGraphNode (unique_ptr) values.
  auto values() { return llvm::make_second_range(FunctionMap); }
  auto values() const { return llvm::make_second_range(FunctionMap); }

  /// Returns the call graph node for the provided function.
  const SimplifiedCallGraphNode *at(const Function *F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  SimplifiedCallGraphNode *at(const Function *F) {
    return const_cast<SimplifiedCallGraphNode *>(
        static_cast<const SimplifiedCallGraph &>(*this).at(F));
  }

  LLVM_ABI void print();
  LLVM_ABI SimplifiedCallGraphNode *getOrInsertFunction(const Function *F);
};

/// A node in SimplifiedCallGraph representing a single function, plus the set
/// of functions it calls. Provides reference counting so the caller
/// can identify roots (in-degree 0) during partitioning.
class SimplifiedCallGraphNode {
public:
  SimplifiedCallGraphNode(Function *F) : F(F) {}

  SimplifiedCallGraphNode(const SimplifiedCallGraphNode &) = delete;
  SimplifiedCallGraphNode &operator=(const SimplifiedCallGraphNode &) = delete;

  ~SimplifiedCallGraphNode() = default;

  Function *getFunction() const { return F; }

  unsigned getNumReferences() const { return NumReferences; }

  using iterator = DenseSet<SimplifiedCallGraphNode *>::iterator;
  using const_iterator = DenseSet<SimplifiedCallGraphNode *>::const_iterator;

  iterator begin() { return CalledFunctions.begin(); }
  iterator end() { return CalledFunctions.end(); }
  const_iterator begin() const { return CalledFunctions.begin(); }
  const_iterator end() const { return CalledFunctions.end(); }
  bool empty() const { return CalledFunctions.empty(); }
  unsigned size() const {
    return static_cast<unsigned>(CalledFunctions.size());
  }

  void addCalledFunction(SimplifiedCallGraphNode *Called) {
    auto [It, Inserted] = CalledFunctions.insert(Called);
    if (Inserted)
      Called->addRef();
  }

private:
  friend class SimplifiedCallGraph;

  Function *F;

  DenseSet<SimplifiedCallGraphNode *> CalledFunctions;
  unsigned NumReferences = 0;

  void addRef() { ++NumReferences; }
};

/// The root function of the call graph, along with its transitive dependency
/// closure and cumulative cost. Used by createWorkList to build the
/// partitioning worklist and by doPartitioning for load-balanced
/// bin-packing; it is the smallest unit allocated by doPartitioning.
struct FunctionWithDependencies {
  using CostType = InstructionCost::CostType;

  /// Collects \p F and all non-declaration functions transitively called by
  /// \p F into Dependencies, and computes TotalCost as the sum of the costs
  /// of all collected functions per \p FnCosts.
  LLVM_ABI
  FunctionWithDependencies(SimplifiedCallGraph &SCG,
                           const DenseMap<const Function *, CostType> &FnCosts,
                           const Function *F);

  // The root function of the call graph.
  const Function *F = nullptr;
  // Transitive closure of non-declaration functions called by F (includes F).
  DenseSet<const Function *> Dependencies;
  // Sum of IR-instruction counts over F and all its dependencies.
  CostType TotalCost = 0;
};

/// Splits a module into linkable partitions by traversing its call graph,
/// so that each partition carries a self-consistent subset of functions
/// (a root + its callees) and is balanced by IR-instruction cost. The
/// partitions can be optimized and codegen'd in parallel.
///
/// calculateFunctionCosts() and createWorkList() run in the constructor;
/// splitModule then:
///   1. externalize(): promotes locals to external+hidden; unnamed entities
///      get a stable name.
///   2. sortWorkList(): sorts the worklist by accumulated cost.
///   3. doPartitioning(): greedily assigns each root + dependencies to the
///      least-loaded partition.
///   4. Per partition: CloneModule, then dealWithMpart downgrades duplicate
///      definitions to available_externally, erases unused declarations,
///      and renames promoted locals with a common per-module suffix.
///   5. Serializes each partition and re-parses it into a fresh
///      caller-created LLVMContext on a worker thread.
class SplitModuleCG {
public:
  /// Invoked once per partition on its worker thread, passing the partition
  /// module parsed into a fresh context created by ContextCreationCallback.
  using ModuleCreationCallback =
      function_ref<void(std::unique_ptr<Module> MPart, unsigned PartitionId)>;

  /// Factory creating the fresh LLVMContext each partition is parsed into
  /// (LLVMContext cannot be shared across threads).
  using ContextCreationCallback = function_ref<std::unique_ptr<LLVMContext>()>;

  /// Construct a SplitModuleCG over module \p M.
  ///
  /// \param M The module to partition. Must outlive the SplitModuleCG
  ///          instance and any partitions emitted via splitModule().
  /// \param LimitPartition Upper bound on the number of partitions to
  ///          produce. Pass 0 (the default) to derive the partition count
  ///          from the number of call-graph roots discovered in
  ///          createWorkList, capped to the available hardware parallelism.
  ///          The actual partition count is finalized in the constructor.
  LLVM_ABI SplitModuleCG(Module &M, unsigned LimitPartition = 0);

  /// Splits the module and invokes \p ModuleCallback once per partition from
  /// the worker threads, using a fresh context per partition (see
  /// ContextCreationCallback).
  LLVM_ABI void splitModule(ModuleCreationCallback ModuleCallback,
                            ContextCreationCallback MakeCtx);

private:
  using CostType = InstructionCost::CostType;

  /// Number of partitions to produce; finalized in the constructor.
  unsigned NumPartitions;
  Module &M;
  CallGraph CG;
  std::unique_ptr<SimplifiedCallGraph> SCG;
  /// Total IR-instruction cost of all non-declaration functions in M.
  CostType ModuleCost;
  /// Call-graph roots discovered by createWorkList.
  DenseSet<const Function *> EntryFuncs;
  /// Names of GVs already external before externalize; excluded from renaming.
  StringSet<> OriginalExternals;
  /// Tracks functions that may be defined in multiple partitions so that
  /// dealWithMpart can downgrade duplicates to available_externally:
  ///   - Membership: the definition is safely downgradable (see
  ///     canDowngradeToAvailableExternally); absent functions are untouched.
  ///   - The mapped bool: true until the first partition holding the
  ///     definition claims it (sets false) and keeps it; later partitions
  ///     downgrade their copies to available_externally.
  DenseMap<const Function *, bool> ExternalFunction;
  /// IR-instruction cost of each non-declaration function in M.
  DenseMap<const Function *, CostType> FuncsCosts;
  /// Partitioning worklist built by createWorkList, sorted by TotalCost.
  SmallVector<FunctionWithDependencies> FWDWorkList;

  /// Compute the IR-instruction cost of every non-declaration function in M
  /// and populate FuncsCosts / ModuleCost.
  void calculateFunctionCosts();

  /// Walk FWDWorkList in cost-sorted order and greedily assign each root and
  /// its dependencies to the partition with the lowest accumulated cost
  /// (load-balanced bin-packing). Returns one set per partition.
  std::vector<DenseSet<const Function *>> doPartitioning();

  /// Post-process a cloned partition \p MPart (partition index \p I):
  ///   - Downgrade duplicate definitions of originally-external functions to
  ///     available_externally.
  ///   - Erase declarations that are no longer used.
  ///   - Rename promoted local symbols (now external, not in OriginalExternals)
  ///     to "name.llvm.<suffix>" to avoid duplicate symbols across partitions.
  void dealWithMpart(Module &MPart, unsigned I);

  /// Discover call-graph roots (functions with in-degree 0 in SCG) and
  /// build FWDWorkList, where each entry is a root + its transitive
  /// dependency closure + the total cost. Functions in cycles that no
  /// root reaches are treated as standalone roots themselves. The list
  /// is sorted by (TotalCost desc, Name asc) so the most expensive roots
  /// are assigned first during partitioning.
  void createWorkList();

  /// Sorts FWDWorkList by total cost (descending) and then by function name.
  void sortWorkList();
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
