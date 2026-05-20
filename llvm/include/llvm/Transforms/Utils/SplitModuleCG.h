#ifndef LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
#define LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H

#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/LTO/Config.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {

class SimplifyCallGraph;
class SimplifyCallGraphNode;

using CostType = InstructionCost::CostType;

class SimplifyCallGraph {
  using FunctionMapTy =
      std::map<const Function *, std::unique_ptr<SimplifyCallGraphNode>>;

  /// A map from \c Function* to \c SimplifyCallGraphNode*.
  FunctionMapTy FunctionMap;

public:
  explicit SimplifyCallGraph(CallGraph &CG,
                             const ModuleSummaryIndex &CombinedIndex,
                             Module &M)
      : CG(CG), M(M) {
    createSimplifyCallGraph(CombinedIndex);
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

  void createSimplifyCallGraph(const ModuleSummaryIndex &CombinedIndex);
  void print();
  SimplifyCallGraphNode *getOrInsertFunction(const Function *F);

private:
  CallGraph &CG;
  Module &M;
};

class SimplifyCallGraphNode {
public:
  using CalledFunctionsSet = DenseSet<SimplifyCallGraphNode *>;
  inline SimplifyCallGraphNode(SimplifyCallGraph *SCG, Function *F)
      : SCG(SCG), F(F) {}

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

  SimplifyCallGraph *SCG;
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

    // Scan for an indirect call. If such a call is found, we have to
    // conservatively assume this can call all non-entrypoint functions in 
    // the module.
    for (auto &SCGNode : *SCG.at(&CurFn)) {
      auto *Callee = SCGNode->getFunction();
      if (!Callee || Callee->isDeclaration())
        continue;
      if (Callee != &F)
      {
        auto [It, Inserted] = Fns.insert(Callee);
        if (Inserted)
          WorkList.push_back(Callee);
      }
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

/// Splits the module M into N linkable partitions. The function ModuleCallback
/// is called N times passing each individual partition as the MPart argument.
class SplitModuleCG {
public:
  using ModuleCreationCallback =
      function_ref<void(std::unique_ptr<Module> MPart, unsigned PartitionId)>;
  SplitModuleCG(Module &M,
                const ModuleSummaryIndex &CombinedIndex,
                unsigned LimitPartition = 0);
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

  void calculateFunctionCosts();
  std::vector<DenseSet<const Function *>> doPartitioning();
  void dealWithMpart(
      Module &MPart, unsigned I,
      function_ref<bool(const GlobalValue *)> NeedsConservativeImport);
  void createWorkList();
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
