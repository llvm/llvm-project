//
// Created by tanmay on 5/25/22.
//

#ifndef LLVM_INFLUENCEGRAPHFLATTENER_H
#define LLVM_INFLUENCEGRAPHFLATTENER_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Pass.h"

namespace llvm {

class FlattenedInfluenceGraph {
public:
  ValueMap<Value*, std::vector<Value*>> influence_map;
  /// Track the last function we run over for printing.
  const Function *LastF = nullptr;

  // Default Constructor
  FlattenedInfluenceGraph() = default;

  // Copy Constructor
  FlattenedInfluenceGraph(FlattenedInfluenceGraph const &graph) {
    for (std::pair<Value*, std::vector<Value*>> influence_pair: graph.influence_map) {
      this->influence_map.insert(std::make_pair(influence_pair.first, influence_pair.second));
    }
    this->LastF = graph.LastF;
  }

  void print(raw_ostream &OS) const;

};

// Analysis pass giving out a Flattened version of the Influence Graph of a
// function.
class InfluenceGraphFlattenerAnalysis : public AnalysisInfoMixin<InfluenceGraphFlattenerAnalysis> {
  // Making AnalysisInfoMixin friend, so it can access members of this class
  friend AnalysisInfoMixin<InfluenceGraphFlattenerAnalysis>;

  // Analysis passes require a Key identifying the Analysis pass which is used
  // by the PassManager.
  static AnalysisKey Key;
public:

  // Type aliasing FlattenedInfluenceGraph as Result, as Result is used by the
  // PassManager.
  using Result = FlattenedInfluenceGraph;

  FlattenedInfluenceGraph run(Function &F, FunctionAnalysisManager &AM);
};

// Printer Pass for the InfluenceGraphFlattenerAnalysis results
class InfluenceGraphFlattenerPrinterPass
    : public PassInfoMixin<InfluenceGraphFlattenerPrinterPass> {
  raw_ostream &OS;

public:
  explicit InfluenceGraphFlattenerPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

// Legacy analysis pass giving a Flattened version of the Influence Graph of a
// function.
class FlattenedInfluenceGraphWrapperPass : public FunctionPass {
  FlattenedInfluenceGraph FIG;

public:
  static char ID;

  FlattenedInfluenceGraphWrapperPass();

  FlattenedInfluenceGraph &getFIG() { return FIG; }
  const FlattenedInfluenceGraph &getFIG() const { return FIG; }

  //  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
  //  void releaseMemory() override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;
};

} // namespace llvm

#endif // LLVM_INFLUENCEGRAPHFLATTENER_H
