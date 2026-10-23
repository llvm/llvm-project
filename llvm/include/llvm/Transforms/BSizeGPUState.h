#ifndef BSIZE_GPU_STATE_H
#define BSIZE_GPU_STATE_H

#include "llvm/Transforms/AbstractState.h"
#include "llvm/Transforms/BSizeDependenceValue.h"

using namespace llvm;

// Defines an abstract state used by the increment value analysis.
class BSizeGPUState : public AbstractState<BSizeDependenceValue, BSizeGPUState> {
 public:
  BSizeGPUState() : numThreads_(
      BSizeDependenceValue(BSizeDependenceValueType::B_CONST)) {}

  void clear() {
    AbstractState::clear();
    numThreads_ = BSizeDependenceValue(BSizeDependenceValueType::B_CONST);
  } 

  bool operator==(const BSizeGPUState& st) const {
    return AbstractState::operator==(st) && numThreads_ == st.numThreads_;
  }

  void operator=(BSizeGPUState st) {
    AbstractState::operator=(st);
    numThreads_ = st.numThreads_;
  }

  // Getters and Setters
  BSizeDependenceValue getNumThreads() const { return numThreads_; } 
  void setNumThreads(BSizeDependenceValue numThreads) { numThreads_ = numThreads; } 

  BSizeDependenceValue getValue(const Value* in) const override;

  BSizeGPUState mergeState(const BSizeGPUState& st) const override;

  // Pretty printing 
  std::string getString() const;
  std::string printInstructionState(const Instruction* I) const override;

  // Test to check correctness.
  static bool testBSizeGPUState();

  private:
  // Predicate representing set of active threads. It takes following values:
  // - (b_const): The predicate is independent of block-size.
  // - (b_bsize): The predicate is dependent of block-size.
  // - (-b_bsize): The predicate is negation of some (b_bsize) value.
  BSizeDependenceValue numThreads_;
};

#endif /* BSizeGPUState.h */
