#ifndef UNCOALESCED_ACCESS_ANALYSIS_H
#define UNCOALESCED_ACCESS_ANALYSIS_H

#include "llvm/Transforms/AbstractExecutionEngine.h"
#include "llvm/Transforms/MultiplierValue.h"
#include "llvm/Transforms/GPUState.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <map> 
#include <set> 
#include <list> 
#include <utility> 

using namespace llvm;

// Class to compute dependences of variables on thread ID and hence,
// the uncoalesced accesses.
class UncoalescedAnalysis
  : public AbstractExecutionEngine<MultiplierValue, GPUState> {
 public: 
  UncoalescedAnalysis(const Function* F, const DominatorTree* DomTree)
    : F_(F), DT_(DomTree), FunctionArgumentValues_(nullptr) {}

  UncoalescedAnalysis(
      const Function* F, const DominatorTree* DomTree,
      std::map<const Function *, 
               std::map<const Value *, MultiplierValue>>* FunctionArgumentValues)
    : F_(F), DT_(DomTree), FunctionArgumentValues_(FunctionArgumentValues) {}

  // Getters 
  const Function* getFunction() const { return F_; }
  const std::set<const Instruction*>& getUncoalescedAccesses() const {
    return UncoalescedAccesses_;
  } 

  // Builds initial GPU state for the function.
  GPUState BuildInitialState() const;

  // Builds analysis information for the function given the initial state.
  void BuildAnalysisInfo(GPUState st);

  // Implements execution of different instructions on the abstract state.
  GPUState ExecuteInstruction(const Instruction* I, GPUState st);

 private:
  // Returns base type size for a Type.
  size_t getBaseTypeSize(const Value *v, const Type *ty,
                           const DataLayout &DL) const;
 
  // Sets base type size.
  void setBaseTypeSize(const Value *v, size_t size);

  // Map from values to their base type size. 
  std::map<const Value*, size_t> baseSizeMap_; 

  // Handles special cases where pointer is a constant expr. 
  MultiplierValue getConstantExprValue(const Value* p);

  std::set<const Instruction*> UncoalescedAccesses_;

  // Function being analyzed for uncoalesced accesses.
  const Function *F_;

  // Dominator Tree Information.
  const DominatorTree* DT_;

  // Function to argument values map (initialized by merging different call
  // contexts of the function.
  // For e.g. if there is a call to F, say x = F(x1, x2) and abstract values of
  // x1 and x2 are v1 and v2, then the following mapping 
  //     F -> (arg0 -> v1, arg0 -> v2)
  // is added to the map.
  std::map<const Function *, std::map<const Value *, MultiplierValue>>* 
      FunctionArgumentValues_;
};

#endif /* UncoalescedAnalysis.h */
