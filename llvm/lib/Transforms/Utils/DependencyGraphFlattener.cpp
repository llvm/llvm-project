#include "llvm/Transforms/Utils/DependencyGraphFlattener.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

PreservedAnalyses DependencyGraphFlattenerPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {

  for (inst_iterator instruction_iterator = inst_begin(F), last_instruction = inst_end(F);
       instruction_iterator != last_instruction; ++instruction_iterator) {
    errs() << *instruction_iterator << "\n";

  }

  return PreservedAnalyses::all();
}