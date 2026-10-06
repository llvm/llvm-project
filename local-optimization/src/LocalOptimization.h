//===- LocalOptimization.h - Five local optimizations as one LLVM pass ----===//
//
// Compiler certification assignment.
//
// Each optimization is a separate member function; run() executes them.
//
//   opt -load-pass-plugin=LocalOptimization.so -passes=local-opt in.ll
//
//===----------------------------------------------------------------------===//

#ifndef LOCAL_OPTIMIZATION_H
#define LOCAL_OPTIMIZATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Instruction;
class Value;
} // namespace llvm

/// Selects what run() does. Set from the pass parameters, e.g.
/// "local-opt<constprop;instcombine;verbose>". By default all five run.
struct LocalOptimizationOptions {
  bool ConstantPropagation = true;
  bool InstCombine = true;
  bool DeadCodeElimination = true;
  bool StrengthReduction = true;
  bool CommonSubexpressionElimination = true;
  /// Print every transformation to stderr.
  bool Verbose = false;
};

class LocalOptimizationPass
    : public llvm::OptionalPassInfoMixin<LocalOptimizationPass> {
public:
  explicit LocalOptimizationPass(LocalOptimizationOptions Opts = {})
      : Opts(Opts) {}

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

private:
  LocalOptimizationOptions Opts;

  // The five optimizations. Each returns true if it changed F.
  bool constantPropagation(llvm::Function &F);
  bool instCombine(llvm::Function &F);
  bool deadCodeElimination(llvm::Function &F);
  bool strengthReduction(llvm::Function &F);
  bool commonSubexpressionElimination(llvm::Function &F);

  /// Replace all uses of I with With and delete I. If With is a new
  /// instruction that has not been inserted yet, it is inserted in I's place.
  void replace(llvm::StringRef Opt, llvm::Instruction &I, llvm::Value *With);
  /// Delete I, which must have no uses.
  void erase(llvm::StringRef Opt, llvm::Instruction &I, llvm::StringRef Why);
};

#endif // LOCAL_OPTIMIZATION_H
