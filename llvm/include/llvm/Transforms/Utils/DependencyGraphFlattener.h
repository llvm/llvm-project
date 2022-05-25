#ifndef LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H
#define LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {

// Class holding the Flattened Dependency Graph
class FlattenedDependencyGraph {
public:
  FlattenedDependencyGraph() = default;
  FlattenedDependencyGraph(FlattenedDependencyGraph const &graph) {}
  ValueMap<Value*, std::vector<Value*>> dependency_map;
};

// Analysis pass giving out a Flattened version of the Dependency Graph of a
// function.
class DependencyGraphFlattenerAnalysis : public AnalysisInfoMixin<DependencyGraphFlattenerAnalysis> {
  // Making AnalysisInfoMixin friend so it can access members of this class
  friend AnalysisInfoMixin<DependencyGraphFlattenerAnalysis>;

  // Analysis passes require a Key identifying the Analysis pass which is used
  // by the PassManager.
  static AnalysisKey Key;
public:

  // Type aliasing FlattenedDependencyGraph as Result as Result is used by the
  // PassManager.
  using Result = FlattenedDependencyGraph;

  FlattenedDependencyGraph run(Function &F, FunctionAnalysisManager &AM);
};



} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DEPENDENCYGRAPHFLATTENER_H