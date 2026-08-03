#ifndef LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
#define LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H

#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/LTO/Config.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {

class SimplifyCallGraph;
class SimplifyCallGraphNode;

using CostType = InstructionCost::CostType;

/// A simplified view of the LLVM CallGraph used by SplitModuleCG to drive
/// callgraph-based module partitioning.
///
/// SimplifyCallGraph drops the function-instruction-level details that the
/// full CallGraph carries and keeps only the information needed for
/// partitioning decisions:
///   - The set of functions in the module (one SimplifyCallGraphNode each).
///   - The static call edges between them.
///   - A reference count (NumReferences) recording how many other functions
///     call a given function. Functions with a reference count of zero are
///     treated as call-graph roots during partitioning.
///
/// The simplified graph is built once (in createSimplifyCallGraph) and is
/// consumed by SplitModuleCG::createWorkList to discover roots and their
/// transitive dependencies.
class SimplifyCallGraph {
  using FunctionMapTy =
      DenseMap<const Function *, std::unique_ptr<SimplifyCallGraphNode>>;

  /// A map from \c Function* to \c SimplifyCallGraphNode*.
  FunctionMapTy FunctionMap;

public:
  explicit SimplifyCallGraph(CallGraph &CG, Module &M) : CG(CG) {
    createSimplifyCallGraph();
  }
  ~SimplifyCallGraph() {};

  using iterator = FunctionMapTy::iterator;
  using const_iterator = FunctionMapTy::const_iterator;

  /// Returns the module the call graph corresponds to.
  inline iterator begin() { return FunctionMap.begin(); }
  inline iterator end() { return FunctionMap.end(); }
  inline const_iterator begin() const { return FunctionMap.begin(); }
  inline const_iterator end() const { return FunctionMap.end(); }

  /// Returns the call graph node for the provided function.
  inline const SimplifyCallGraphNode *operator[](const Function *F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  /// Returns the call graph node for the provided function.
  inline SimplifyCallGraphNode *operator[](const Function *F) {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get(); 
  }

  /// Returns the call graph node for the provided function.
  inline const SimplifyCallGraphNode *at(const Function *F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  /// Returns the call graph node for the provided function.
  inline SimplifyCallGraphNode *at(const Function *F) {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  void createSimplifyCallGraph();
  void print();
  SimplifyCallGraphNode *getOrInsertFunction(const Function *F);

private:
  CallGraph &CG;
};

/// A node in SimplifyCallGraph representing a single function, plus the set
/// of functions it calls. Provides reference counting so the caller
/// can identify roots (in-degree 0) during partitioning.
class SimplifyCallGraphNode {
public:
  using CalledFunctionsSet = DenseSet<SimplifyCallGraphNode *>;
  inline SimplifyCallGraphNode(SimplifyCallGraph *SCG, Function *F)
      : F(F) {}

  SimplifyCallGraphNode(const SimplifyCallGraphNode &) = delete;
  SimplifyCallGraphNode &operator=(const SimplifyCallGraphNode &) = delete;

  ~SimplifyCallGraphNode() {}

  Function *getFunction() const { return F; }

  unsigned getNumReferences() const { return NumReferences; }

  using iterator = DenseSet<SimplifyCallGraphNode *>::iterator;
  using const_iterator = DenseSet<SimplifyCallGraphNode *>::const_iterator;

  inline iterator begin() { return CalledFunctions.begin(); }
  inline iterator end() { return CalledFunctions.end(); }
  inline const_iterator begin() const { return CalledFunctions.begin(); }
  inline const_iterator end() const { return CalledFunctions.end(); }
  inline size_t count(SimplifyCallGraphNode * SCGNode) { return CalledFunctions.count(SCGNode); }
  inline bool empty() const { return CalledFunctions.empty(); }
  inline unsigned size() const { return (unsigned)CalledFunctions.size(); }

  void addCalledFunction(SimplifyCallGraphNode *Called) {
    auto [It, Inserted] = CalledFunctions.insert(Called);
    if (Inserted)
      Called->AddRef();
  }

  void removeCalledFunction(SimplifyCallGraphNode *Called) {
    auto NumRemoved = CalledFunctions.erase(Called);
    if (NumRemoved > 0)
      Called->DropRef();
  }

private:
  friend class SimplifyCallGraph;

  Function *F;

  DenseSet<SimplifyCallGraphNode *> CalledFunctions;
  unsigned NumReferences = 0;

  void DropRef() { --NumReferences; }
  void AddRef() { ++NumReferences; }
};

static void addAllDependencies(SimplifyCallGraph &SCG, const Function &F,
                               DenseSet<const Function *> &Fns) {
  assert(!F.isDeclaration());
  SmallVector<const Function *> WorkList({&F});

  while (!WorkList.empty()) {
    const auto &CurFn = *WorkList.pop_back_val();
    assert(!CurFn.isDeclaration());

    // Walk the callees of CurFn recorded in SimplifyCallGraph and
    // add them to Fns, recursing transitively via the WorkList.
    for (auto &SCGNode : *SCG.at(&CurFn)) {
      auto *Callee = SCGNode->getFunction();
      if (!Callee || Callee->isDeclaration())
        continue;
      // Don't recurse into the starting function itself (would re-add F).
      if (Callee == &F)
        continue;
      if (Fns.insert(Callee).second)
        WorkList.push_back(Callee);
    }
  }
}

struct FunctionWithDependencies {
  FunctionWithDependencies(SimplifyCallGraph &SCG,
                           const DenseMap<const Function *, CostType> &FnCosts,
                           const Function *F)
      : F(F) {
    addAllDependencies(SCG, *F, Dependencies);

    TotalCost = FnCosts.at(F);
    for (const auto *Dep : Dependencies) {
      TotalCost += FnCosts.lookup(Dep);
    }
  }

  const Function *F = nullptr;
  DenseSet<const Function *> Dependencies;
  CostType TotalCost = 0;
};

/// Splits a module into N linkable partitions by traversing its call graph,
/// so that each partition carries a self-consistent subset of functions
/// (a root + its callees) and is balanced by IR-instruction cost. The
/// resulting partitions can be optimized and codegen'd in parallel by the
/// LTO backend and merged back into a single object.
///
/// Workflow (driven by SplitModule):
///   1. externalize(): promote local symbols to external+hidden so they are
///      visible across partitions. Unnamed entities get a stable name.
///   2. calculateFunctionCosts(): compute per-function IR instruction counts.
///   3. createWorkList(): walk SimplifyCallGraph to discover call-graph roots
///      and their transitive dependencies.
///   4. doPartitioning(): greedily assign each root + dependencies to the
///      least-loaded partition, balancing by accumulated cost.
///   5. For each partition: CloneModule the original module filtered by
///      ShouldCloneDefinition, then dealWithMpart cleans up unused locals,
///      marks available-externally-defined functions, and records (in
///      PromotedRenames) the renaming for promoted locals so the caller can
///      apply it later.
///   6. Each partition bitcode is serialized to its own LLVMContext (via
///      write+read) so partitions can be processed on concurrent threads
///      without sharing LLVMContext state.
class SplitModuleCG {
public:
  using ModuleCreationCallback =
      function_ref<void(std::unique_ptr<Module> MPart, unsigned PartitionId)>;

  /// Construct a SplitModuleCG over module \p M.
  ///
  /// \param M The module to partition. Must outlive the SplitModuleCG
  ///          instance and any partitions emitted via SplitModule().
  /// \param LimitPartition Upper bound on the number of partitions to
  ///          produce. Pass 0 (the default) to derive the partition count
  ///          from the number of call-graph roots discovered in
  ///          createWorkList (one root per partition at most). The actual
  ///          partition count is finalized in the constructor and can be
  ///          queried via getPartitionNum().
  SplitModuleCG(Module &M, unsigned LimitPartition = 0);
  void SplitModule(ModuleCreationCallback ModuleCallback,
                   const llvm::lto::Config &C);

  unsigned getPartitionNum() { return N; }
  StringSet<> &getOriginalExternals() { return OriginalExternals; }
  StringMap<std::string> &getPromotedRenames() { return PromotedRenames; }

private:
  unsigned N;
  Module &M;
  CallGraph CG;
  std::unique_ptr<SimplifyCallGraph> SCG;
  CostType ModuleCost;
  DenseSet<const Function *> EntryFuncs;
  StringSet<> OriginalExternals;
  StringMap<std::string> PromotedRenames;
  DenseMap<const Function *, bool> externalFunction;
  DenseMap<const Function *, CostType> FuncsCosts;
  SmallVector<FunctionWithDependencies> FWDWorkList;

  /// Compute the IR-instruction cost of every non-declaration function in M
  /// and populate FuncsCosts / ModuleCost.
  void calculateFunctionCosts();

  /// Walk FWDWorkList in cost-sorted order and greedily assign each root and
  /// its dependencies to the partition with the lowest accumulated cost
  /// (load-balanced bin-packing). Returns N partition sets, one per partition.
  std::vector<DenseSet<const Function *>> doPartitioning();

  /// Post-process a cloned partition \p MPart (partition index \p I):
  ///   - Record promoted names for symbols that were local but
  ///     are now external (not in OriginalExternals) into PromotedRenames.
  ///   - Erase conservatively-cloned local globals that ended up with no users.
  ///   - For functions that were already external in the source module and
  ///     are being defined in this partition, downgrade their duplicate
  ///     definitions in other partitions to available_externally via the
  ///     externalFunction map.
  /// \p NeedsConservativeImport is the predicate (captured by SplitModule)
  /// that identifies local globals that must be cloned into every partition.
  void dealWithMpart(
      Module &MPart, unsigned I,
      function_ref<bool(const GlobalValue *)> NeedsConservativeImport);

  /// Discover call-graph roots (functions with in-degree 0 in SCG) and
  /// build FWDWorkList, where each entry is a root + its transitive
  /// dependency closure + the total cost. Functions in cycles that no
  /// root reaches are treated as standalone roots themselves. The list
  /// is sorted by (TotalCost desc, Name asc) so the most expensive roots
  /// are assigned first during partitioning.
  void createWorkList();
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
