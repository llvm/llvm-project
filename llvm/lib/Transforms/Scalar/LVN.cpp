#include "llvm/Transforms/Scalar/LVN.h"

#include <cstdint>
#include <tuple>
#include <unordered_map>

#include "llvm/IR/InstrTypes.h"

using namespace llvm;

namespace {

using InstrTupleTy = std::tuple<Instruction::BinaryOps, Value *, Value *>;

struct InstrTupleHash {
  size_t operator()(const InstrTupleTy &InstrTuple) const {
    auto LHSPtrVal = reinterpret_cast<uintptr_t>(std::get<1>(InstrTuple));
    auto RHSPtrVal = reinterpret_cast<uintptr_t>(std::get<2>(InstrTuple));
    return std::get<0>(InstrTuple) ^ LHSPtrVal * RHSPtrVal;
  }
};

} // end namespace

PreservedAnalyses LVNPass::run(Function &F, FunctionAnalysisManager &FAM) {
  for (auto &BB : F) {
    runOnBasicBlock(BB);
  }
  return PreservedAnalyses::all();
}

void LVNPass::runOnBasicBlock(BasicBlock &BB) {
  std::unordered_map<InstrTupleTy, Value *, InstrTupleHash> InstrToValue;
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
