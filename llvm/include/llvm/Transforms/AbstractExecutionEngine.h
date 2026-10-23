#ifndef ABSTRACT_EXECUTION_ENGINE_H
#define ABSTRACT_EXECUTION_ENGINE_H

#include "llvm/Transforms/AbstractState.h"
#include "llvm/Transforms/AbstractValue.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <list> 
#include <map> 
#include <string> 
#include <utility> 

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "abstract-execution"
#endif

#define NUM_RECENT_BLOCKS 16

using namespace llvm;

// This class defines an abstract execution engine. An abstract execution engine
// takes in a program and executes the program abstractly using semantics
// defined for an abstract value.
// T is the type of abstract value used for abstract execution. It must implement
// AbstractValue<T>.
// U is the type of abstract value used for abstract execution. It must implement
// AbstractState<T>.
template<typename T, typename U>
class AbstractExecutionEngine {
 static_assert(
   std::is_base_of<AbstractValue<T>, T>::value, 
   "T must be a descendant of AbstractValue<T>"
 );
 static_assert(
   std::is_base_of<AbstractState<T, U>, U>::value, 
   "U must be a descendant of AbstractState<T, U>"
 ); 
 public:
  AbstractExecutionEngine()
    : entryBlock_(nullptr) {}
  AbstractExecutionEngine(const BasicBlock* entryBlock, U initialState)
    : entryBlock_(entryBlock), initialState_(initialState) {}

  virtual ~AbstractExecutionEngine() = 0;

  // Queries the state before an instruction.
  const U& getStateBeforeInstruction(const Instruction* inst){
    return StateBeforeInstructionMap_[inst];
  }

  // Adds a block to execute next and the state in which the block must be 
  // executed.
  void AddBlockToExecute(const BasicBlock* b, U st);
 
  // Executes program (can be overriden).
  virtual void Execute();

  // Executes the instruction on a state and returns the state after execution.
  virtual U ExecuteInstruction(const Instruction* inst,
                               U st) = 0;

  // Utils
  static bool IsTerminator(const Instruction* I);

 protected:
  // Entry block where the abstract execution begins.
  const BasicBlock* entryBlock_;

  // Initial state before execution of the program.
  U initialState_;

 private:
  // Returns the next unit to execute.
  std::pair<const BasicBlock*, U> getNextExecutionUnit(
     std::list<std::pair<const BasicBlock*, U>>& worklist);

  // Add block to recently executed blocks.
  void AddRecentBlock(const BasicBlock* block);

  // Stores some recent blocks executed by the engine.
  SmallVector<const BasicBlock*, NUM_RECENT_BLOCKS> recentBlocks_;

  // Records abstract state before an instruction is executed.
  std::map<const Instruction*, U> StateBeforeInstructionMap_;

  // Buffer to store the set of blocks that must be executed after this block
  // completes execution. 
  std::list<std::pair<const BasicBlock*, U>> BlocksToExecuteBuffer_;
};

template<typename T, typename U>
AbstractExecutionEngine<T, U>::~AbstractExecutionEngine() {}

template<typename T, typename U>
void AbstractExecutionEngine<T, U>::AddBlockToExecute(const BasicBlock* b, U st) {
  BlocksToExecuteBuffer_.push_back(std::pair<const BasicBlock*, U>(b, st));
}

// Returns a block in recentBlocks_ if found. Otherwise returns the
// first block in worklist. This optimization is useful for execution
// of loops. All blocks within the loop are given priority over blocks
// after the loop. This ensures that the blocks after the loop are 
// executed only after the loop reaches a fixed point.
template<typename T, typename U>
std::pair<const BasicBlock*, U> 
    AbstractExecutionEngine<T, U>::getNextExecutionUnit(
        std::list<std::pair<const BasicBlock*, U>>& worklist) {
  for (const BasicBlock* block : recentBlocks_) {
    auto listIt = find_if(worklist.begin(), worklist.end(), 
             [block](const std::pair<const BasicBlock*, U>& item){
               if (item.first == block) return true;
               else return false;
             });
    if (listIt != worklist.end()) {
      // Block found.
      auto unit = *listIt;
      worklist.erase(listIt);
      AddRecentBlock(unit.first);
      return unit;
    }
  }
  auto unit = worklist.front();
  worklist.pop_front();
  AddRecentBlock(unit.first);
  return unit;
} 

// Adds block to the set of recent blocks.
template<typename T, typename U>
void AbstractExecutionEngine<T, U>::AddRecentBlock(const BasicBlock* block) {
  auto pos = recentBlocks_.begin();
  while (*pos != block && pos != recentBlocks_.end()) ++pos;
  if (pos != recentBlocks_.end()) { recentBlocks_.erase(pos); }
  if (recentBlocks_.size() >= NUM_RECENT_BLOCKS) {
    recentBlocks_.pop_back();
  }
  recentBlocks_.insert(recentBlocks_.begin(), block);
}

template<typename T, typename U>
void AbstractExecutionEngine<T, U>::Execute() {
  // Worklist to execute basic blocks.
  // Each worklist item consists of a basicblock and an abstract state to be 
  // propagated through the block.
  std::list<std::pair<const BasicBlock*, U>> worklist;
  worklist.push_back(std::pair<const BasicBlock*, U>(entryBlock_, initialState_));

  // Execute work items in worklist.
  StateBeforeInstructionMap_.clear();
  while (!worklist.empty()) {
    auto unit = getNextExecutionUnit(worklist);
    const BasicBlock *b = unit.first; // next block to be executed.
    U st = unit.second; // state before next block.
    LLVM_DEBUG(errs() << "BasicBlock: " << b->getName() << "\n");

    // Clear buffer.
    BlocksToExecuteBuffer_.clear();
    // Execute instructions within the block.
    for (BasicBlock::const_iterator it = b->begin(), ite = b->end(); 
                                                    it != ite; ++it) {
      const Instruction* I = &*it;
      // If I is the first statement in the block, merge I's pre-state
      // with incoming state.
      if (it == b->begin()) {
        if(StateBeforeInstructionMap_.find(I) != 
               StateBeforeInstructionMap_.end()) {
          U oldState = StateBeforeInstructionMap_[I];
          U newState = oldState.mergeState(st);
          // State before block unchanged; no need to execute block.
          if (oldState == newState) break;

          StateBeforeInstructionMap_[I] = newState;
        } else {
          StateBeforeInstructionMap_[I] = st;
        }
      } else {
        StateBeforeInstructionMap_[I] = st;
      }
      
      LLVM_DEBUG(errs() << "   " << *I << ", " << st.printInstructionState(I) << "\n");
      st = ExecuteInstruction(I, st);
    }
    // Add subsequent blocks to be executed. Note that these were added to
    // the buffer during the execution of instructions in the current block.
    for (auto bufferIt = BlocksToExecuteBuffer_.begin(), 
         bufferIte = BlocksToExecuteBuffer_.end(); bufferIt != bufferIte; 
                                                              ++bufferIt) {
      // Check if the key already exists in worklist, if so, merge the two
      // work items. This is an optimization that helps scale the execution,
      // at the cost of being slightly imprecise.
      const BasicBlock* block = bufferIt->first;
      auto listIt = find_if(worklist.begin(), worklist.end(), 
               [block](const std::pair<const BasicBlock*, U>& item){
                 if (item.first == block) return true;
                 else return false;
               });
      if (listIt != worklist.end()) {
        listIt->second = listIt->second.mergeState(bufferIt->second);
      } else {
        worklist.push_back(std::pair<const BasicBlock*, U>(bufferIt->first, 
                                                           bufferIt->second));
      }
    }
  }
}

// Utils
template<typename T, typename U>
bool AbstractExecutionEngine<T, U>::IsTerminator(const Instruction* I) {
    return isa<UncondBrInst>(I) ||
	   isa<CondBrInst>(I) ||
	   isa<IndirectBrInst>(I) ||
	   isa<CatchSwitchInst>(I) ||
	   isa<CatchReturnInst>(I) ||
	   isa<CleanupReturnInst>(I);
}
 
#endif /* AbstractExecutionEngine.h */
