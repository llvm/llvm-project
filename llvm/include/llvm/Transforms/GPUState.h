#ifndef GPU_STATE_H
#define GPU_STATE_H

#include "llvm/Transforms/AbstractState.h"
#include "llvm/Transforms/MultiplierValue.h"

using namespace llvm;

// Defines an abstract state used by the increment value analysis.
class GPUState : public AbstractState<MultiplierValue, GPUState> {
 public:
  GPUState() : numThreads_(MultiplierValue(MultiplierValueType::TOP)) {}

  void clear() {
    AbstractState::clear();
    numThreads_ = MultiplierValue(MultiplierValueType::TOP);
  } 

  bool operator==(const GPUState& st) const {
    return AbstractState::operator==(st) && numThreads_ == st.numThreads_;
  }

  void operator=(GPUState st) {
    AbstractState::operator=(st);
    numThreads_ = st.numThreads_;
  }

  // Getters and Setters
  MultiplierValue getNumThreads() const { return numThreads_; } 
  void setNumThreads(MultiplierValue numThreads) { numThreads_ = numThreads; } 

  MultiplierValue getValue(const Value* in) const override;

  GPUState mergeState(const GPUState& st) const override;

  // Pretty printing 
  std::string getString() const;
  std::string printInstructionState(const Instruction* I) const override;

  // Test to check correctness.
  static bool testGPUState();

  private:
  // Number of active threads. It takes two values:
  // - (one): A single thread is active.
  // - (unknown): Arbitrary number of threads are active.
  MultiplierValue numThreads_;
};

#endif /* GPUState.h */
