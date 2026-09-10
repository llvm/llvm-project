#ifndef LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
#define LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/LTO/Config.h"
#include <map>

namespace llvm {

class SimplifiedCallGraph;
class SimplifiedCallGraphNode;

using CostType = InstructionCost::CostType;

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
  explicit SimplifiedCallGraph(CallGraph &CG);
  ~SimplifiedCallGraph() = default;

  using iterator = FunctionMapTy::iterator;
  using const_iterator = FunctionMapTy::const_iterator;

  /// Iterates over all (Function*, SimplifiedCallGraphNode) pairs in the
  /// call graph.
  inline iterator begin() { return FunctionMap.begin(); }
  inline iterator end() { return FunctionMap.end(); }
  inline const_iterator begin() const { return FunctionMap.begin(); }
  inline const_iterator end() const { return FunctionMap.end(); }

  /// Iterates over all SimplifiedCallGraphNode (unique_ptr) values.
  auto values() { return llvm::make_second_range(FunctionMap); }
  auto values() const { return llvm::make_second_range(FunctionMap); }

  /// Returns the call graph node for the provided function.
  inline const SimplifiedCallGraphNode *at(const Function *F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  inline SimplifiedCallGraphNode *at(const Function *F) {
    return const_cast<SimplifiedCallGraphNode *>(
        static_cast<const SimplifiedCallGraph &>(*this).at(F));
  }

  void print();
  SimplifiedCallGraphNode *getOrInsertFunction(const Function *F);
};

/// A node in SimplifiedCallGraph representing a single function, plus the set
/// of functions it calls. Provides reference counting so the caller
/// can identify roots (in-degree 0) during partitioning.
class SimplifiedCallGraphNode {
public:
  inline SimplifiedCallGraphNode(Function *F) : F(F) {}

  SimplifiedCallGraphNode(const SimplifiedCallGraphNode &) = delete;
  SimplifiedCallGraphNode &operator=(const SimplifiedCallGraphNode &) = delete;

  ~SimplifiedCallGraphNode() = default;

  Function *getFunction() const { return F; }

  unsigned getNumReferences() const { return NumReferences; }

  using iterator = DenseSet<SimplifiedCallGraphNode *>::iterator;
  using const_iterator = DenseSet<SimplifiedCallGraphNode *>::const_iterator;

  inline iterator begin() { return CalledFunctions.begin(); }
  inline iterator end() { return CalledFunctions.end(); }
  inline const_iterator begin() const { return CalledFunctions.begin(); }
  inline const_iterator end() const { return CalledFunctions.end(); }
  inline bool empty() const { return CalledFunctions.empty(); }
  inline unsigned size() const { return (unsigned)CalledFunctions.size(); }

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

/// Collect \p F and all non-declaration functions transitively called by \p F,
/// using the SimplifiedCallGraph \p SCG, and insert them into \p Fns.
static void addAllDependencies(SimplifiedCallGraph &SCG, const Function &F,
                               DenseSet<const Function *> &Fns) {
  assert(!F.isDeclaration());
  SmallVector<const Function *> WorkList({&F});
  Fns.insert(&F);

  while (!WorkList.empty()) {
    const auto &CurFn = *WorkList.pop_back_val();
    assert(!CurFn.isDeclaration());

    // Walk the callees of CurFn recorded in SimplifiedCallGraph and
    // add them to Fns, recursing transitively via the WorkList.
    for (auto &SCGNode : *SCG.at(&CurFn)) {
      auto *Callee = SCGNode->getFunction();
      if (!Callee || Callee->isDeclaration())
        continue;
      if (Fns.insert(Callee).second)
        WorkList.push_back(Callee);
    }
  }
}

/// The root function of the call graph, along with its transitive dependency
/// closure and cumulative cost. Used by createWorkList to build the
/// partitioning worklist and by doPartitioning for load-balanced
/// bin-packing; it is the smallest unit allocated by doPartitioning.
struct FunctionWithDependencies {
  FunctionWithDependencies(SimplifiedCallGraph &SCG,
                           const DenseMap<const Function *, CostType> &FnCosts,
                           const Function *F)
      : F(F) {
    addAllDependencies(SCG, *F, Dependencies);

    for (const auto *Dep : Dependencies)
      TotalCost += FnCosts.lookup(Dep);
  }

  // The root function of the call graph.
  const Function *F = nullptr;
  // Transitive closure of non-declaration functions called by F (includes F).
  DenseSet<const Function *> Dependencies;
  // Sum of IR-instruction counts over F and all its dependencies.
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
///   3. createWorkList(): walk SimplifiedCallGraph to discover call-graph roots
///      and their transitive dependencies.
///   4. doPartitioning(): greedily assign each root + dependencies to the
///      least-loaded partition, balancing by accumulated cost.
///   5. For each partition: CloneModule the original module filtered by
///      ShouldCloneDefinition, then dealWithMpart downgrades duplicate
///      external function definitions to available_externally and renames
///      promoted locals to avoid duplicate symbols across partitions.
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
  ///          partition count is finalized in the constructor.
  SplitModuleCG(Module &M, unsigned LimitPartition = 0);
  void SplitModule(ModuleCreationCallback ModuleCallback,
                   const llvm::lto::Config &C);

private:
  unsigned N;
  Module &M;
  CallGraph CG;
  std::unique_ptr<SimplifiedCallGraph> SCG;
  CostType ModuleCost;
  DenseSet<const Function *> EntryFuncs;
  StringSet<> OriginalExternals;
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
  ///   - Downgrade duplicate definitions of originally-external functions to
  ///     available_externally.
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
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
