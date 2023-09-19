#include "llvm/Transforms/Scalar/LVN.h"

#include <map>
#include <tuple>
#include <unordered_map>

#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

PreservedAnalyses LVNPass::run(Function &F, FunctionAnalysisManager &AM) {
  for (auto &BB : F) {
    runImpl(BB);
  }
  return PreservedAnalyses::all();
}

void LVNPass::runImpl(BasicBlock &BB) {
  std::map<std::tuple<Instruction::BinaryOps, Value *, Value *>, Value *>
      InstrToValue;
  for (auto &I : BB) {
    BinaryOperator *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO) {
      continue;
    }
    Value *RHS = BO->getOperand(0);
    Value *LHS = BO->getOperand(1);
    auto InstrTuple = std::make_tuple(BO->getOpcode(), RHS, LHS);
    Value *BOValue = static_cast<Value *>(BO);
    auto FoundInstrIt = InstrToValue.find(InstrTuple);
    if (FoundInstrIt == InstrToValue.end()) {
      InstrToValue.insert({InstrTuple, BOValue});
    } else {
      BO->replaceAllUsesWith(FoundInstrIt->second);
    }
  }
}
