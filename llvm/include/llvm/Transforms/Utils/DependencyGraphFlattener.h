#ifndef LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H
#define LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Pass.h"

// Procedure to build this Pass was copied from the BranchProbabilityInfo Pass
// You can search for this pass by looking for "branch-prob" in PassRegistry.def

namespace llvm {

// Class holding the Flattened Dependency Graph
class FlattenedDependencyGraph {
public:
  ValueMap<Value*, std::vector<Value*>> dependency_map;

  /// Track the last function we run over for printing.
  const Function *LastF = nullptr;

  // Default Constructor
  FlattenedDependencyGraph() = default;

  // Copy Constructor
  FlattenedDependencyGraph(FlattenedDependencyGraph const &graph) {
    for (std::pair<Value*, std::vector<Value*>> dependency_pair: graph.dependency_map) {
      this->dependency_map.insert(std::make_pair(dependency_pair.first, dependency_pair.second));
    }
    this->LastF = graph.LastF;
  }

  void print(raw_ostream &OS) const;

};

// Analysis pass giving out a Flattened version of the Dependency Graph of a
// function.
class DependencyGraphFlattenerAnalysis : public AnalysisInfoMixin<DependencyGraphFlattenerAnalysis> {
  // Making AnalysisInfoMixin friend, so it can access members of this class
  friend AnalysisInfoMixin<DependencyGraphFlattenerAnalysis>;

  // Analysis passes require a Key identifying the Analysis pass which is used
  // by the PassManager.
  static AnalysisKey Key;
public:

  // Type aliasing FlattenedDependencyGraph as Result, as Result is used by the
  // PassManager.
  using Result = FlattenedDependencyGraph;

  FlattenedDependencyGraph run(Function &F, FunctionAnalysisManager &AM);
};

// Printer Pass for the DependencyGraphFlattenerAnalysis results
class DependencyGraphFlattenerPrinterPass
    : public PassInfoMixin<DependencyGraphFlattenerPrinterPass> {
  raw_ostream &OS;

public:
  explicit DependencyGraphFlattenerPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

// Legacy analysis pass giving a Flattened version of the Dependency Graph of a
// function.
class FlattenedDependencyGraphWrapperPass : public FunctionPass {
  FlattenedDependencyGraph FDG;

public:
  static char ID;

  FlattenedDependencyGraphWrapperPass();

  FlattenedDependencyGraph &getFDG() { return FDG; }
  const FlattenedDependencyGraph &getFDG() const { return FDG; }

//  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
//  void releaseMemory() override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H