#ifndef BLOCK_SIZE_INVARIANCE_ANALYSIS_H
#define BLOCK_SIZE_INVARIANCE_ANALYSIS_H

#include "llvm/Transforms/AbstractExecutionEngine.h"
#include "llvm/Transforms/BSizeDependenceValue.h"
#include "llvm/Transforms/BSizeGPUState.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"
#include <map> 
#include <set> 
#include <list> 
#include <utility> 

using namespace llvm;

// Class to compute dependences of variables on thread ID and hence,
// the uncoalesced accesses.
class BlockSizeInvarianceAnalysis
  : public AbstractExecutionEngine<BSizeDependenceValue, BSizeGPUState> {
 public: 
  BlockSizeInvarianceAnalysis(
       const Function* F, const DominatorTree* DomTree, int ThreadDim)
    : F_(F), DT_(DomTree), ThreadDim_(ThreadDim),
      FunctionReturnValueMap_(nullptr),
      FunctionBSIMap_(nullptr) {}

  BlockSizeInvarianceAnalysis(
       const Function* F, const DominatorTree* DomTree, int ThreadDim,
       std::map<const Function *, BSizeDependenceValue>* FunctionReturnValueMap,
       const std::map<const Function *, bool>* FunctionBSIMap)
    : F_(F), DT_(DomTree), ThreadDim_(ThreadDim),
      FunctionReturnValueMap_(FunctionReturnValueMap),
      FunctionBSIMap_(FunctionBSIMap) {}


  // Getters 
  const Function* getFunction() const { return F_; }
  const std::set<const Instruction*>& getBlockSizeDependentAccesses() const {
    return BlockSizeDependentAccesses_;
  } 
  const std::set<const Instruction*>& getSyncThreads() const {
    return SyncThreads_;
  } 

  // Builds initial GPU state for the function. Assumes all arguments are
  // block-size independent.
  BSizeGPUState BuildInitialState() const;

  // Builds analysis information for the function given the initial state.
  void BuildAnalysisInfo(BSizeGPUState st);

  // Implements execution of different instructions on the abstract state.
  BSizeGPUState ExecuteInstruction(const Instruction* I, BSizeGPUState st);

 private:
  // Prints access pattern for an access.
  std::string printAccessPattern(const Value* root,
      const std::vector<BSizeDependenceValue>& pattern);

  // Returns values for special calls to get tid, bid, and bdim.
  BSizeDependenceValue getCalledFunctionValue(const StringRef& name,
      const CallInst* I, const BSizeGPUState& st); 

  // Handles shared memory loads/stores and returns appropriate
  // value for accessed pointer.
  BSizeDependenceValue HandleSharedMemoryAccess(const Value* p,
    const Instruction* I);

  // Updates current access pattern with new access index.
  void UpdateCurrentAccessPattern(const Value* initp, const Value* finalp,
    BSizeDependenceValue vIdx);

  // Handles special cases where pointer is a constant expr. 
  BSizeDependenceValue HandleConstantExprPointer(const Value* p,
    const BSizeGPUState& st);

  // Checks if for a given function call all arguments are BSI.
  bool AreFunctionCallArgsBSI(const CallInst* CI, const BSizeGPUState& st);

  // Handles special BSI library calls.
  bool isBSILibraryCall(const StringRef& name);

  // Function being analyzed for uncoalesced accesses.
  const Function *F_;

  // Dominator Tree Information.
  const DominatorTree* DT_;

  // Thread dimension being analyzed.
  int ThreadDim_;

  // Set of (shared/global) accesses that depend on block-size.
  std::set<const Instruction*> BlockSizeDependentAccesses_;

  // Set of syncthread statements. 
  std::set<const Instruction*> SyncThreads_;

  // Map to track access patterns for shared memory variables.
  // Maps each root shared memory variable to a unique access pattern.
  // If the access pattern is not unique and consists of values other than
  // (const) and (tid), block-size invariance fails.
  std::map<const Value*, std::vector<BSizeDependenceValue>>
      SharedMemoryAccessPatternMap_;

  // Map from values to their current access pattern (should be finite
  // and non-recursive).
  std::map<const Value*,
           std::pair<const Value*, std::vector<BSizeDependenceValue>>>
      CurrentAccessPatternMap_;

  // Function to return value map. 
  // Assuming the function takes in block-size independent values, does it
  // return a block-independent value?
  std::map<const Function *, BSizeDependenceValue>* FunctionReturnValueMap_;

  // Is function block-size independent?
  const std::map<const Function *, bool>* FunctionBSIMap_;
  
};

#endif /* BlockSizeInvarianceAnalysis.h */
