
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "llvm/IR/Function.h"

#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <deque>

// TODO note this doesn't go through [loop, unreachable], and we could get more
// performance by doing this can consider doing some domtree magic potentially
static inline llvm::SmallPtrSet<llvm::BasicBlock *, 4>
getGuaranteedUnreachable(llvm::Function *F) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> knownUnreachables;
  if (F->empty())
    return knownUnreachables;
  std::deque<llvm::BasicBlock *> todo;
  for (auto &BB : *F) {
    todo.push_back(&BB);
  }

  while (!todo.empty()) {
    llvm::BasicBlock *next = todo.front();
    todo.pop_front();

    if (knownUnreachables.find(next) != knownUnreachables.end())
      continue;

    if (llvm::isa<llvm::ReturnInst>(next->getTerminator()))
      continue;

    if (llvm::isa<llvm::UnreachableInst>(next->getTerminator())) {
      knownUnreachables.insert(next);
      for (llvm::BasicBlock *Pred : predecessors(next)) {
        todo.push_back(Pred);
      }
      continue;
    }

    // Assume resumes don't happen
    // TODO consider EH
    if (llvm::isa<llvm::ResumeInst>(next->getTerminator())) {
      knownUnreachables.insert(next);
      for (llvm::BasicBlock *Pred : predecessors(next)) {
        todo.push_back(Pred);
      }
      continue;
    }

    bool unreachable = true;
    for (llvm::BasicBlock *Succ : llvm::successors(next)) {
      if (knownUnreachables.find(Succ) == knownUnreachables.end()) {
        unreachable = false;
        break;
      }
    }

    if (!unreachable)
      continue;
    knownUnreachables.insert(next);
    for (llvm::BasicBlock *Pred : llvm::predecessors(next)) {
      todo.push_back(Pred);
    }
    continue;
  }

  return knownUnreachables;
}