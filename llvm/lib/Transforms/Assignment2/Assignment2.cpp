#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <cxxabi.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
using namespace llvm;
using namespace std;

// Strings for output
std::string output_str;
raw_string_ostream output(output_str);

// Demangles the function name.
std::string demangle(const char *name) {
  int status = -1;
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : std::string(name);
}

// Returns the source code line number cooresponding to the LLVM instruction.
// Returns -1 if the instruction has no associated Metadata.
int getSourceCodeLine(Instruction *I) {
  llvm::DebugLoc debugInfo = I->getDebugLoc();
  int line = -1;
  if (debugInfo) line = debugInfo.getLine();
  return line;
}

// Topologically sort all the basic blocks in a function.
// Handle cycles in the directed graph using Tarjan's algorithm
// of Strongly Connected Components (SCCs).
vector<BasicBlock *> topoSortBBs(Function &F) {
  vector<BasicBlock *> tempBB;
  for (scc_iterator<Function *> I = scc_begin(&F), IE = scc_end(&F); I != IE; ++I) {
    const std::vector<BasicBlock *> &SCCBBs = *I;
    for (std::vector<BasicBlock *>::const_iterator BBI = SCCBBs.begin(), BBIE = SCCBBs.end(); BBI != BBIE; ++BBI) {
      BasicBlock *b = const_cast<llvm::BasicBlock *>(*BBI);
      tempBB.push_back(b);
    }
  }

  reverse(tempBB.begin(), tempBB.end());
  return tempBB;
}

namespace {
  struct Assignment2 : public FunctionPass {
    static char ID;

    // Vector to store the line numbers at which undefined variable(s) is(are) used.
    vector<int> bugs;

    // Keep track of all the functions we have encountered so far.
    unordered_map<string, bool> funcNames;

    // All Basic Block exit sets
    unordered_map<BasicBlock *, unordered_set<Value *>> exitSets;

    // Reset all global variables when a new function is called.
    void cleanGlobalVariables() {
      bugs.clear();
      exitSets.clear();
      output_str = "";
    }

    Assignment2() : FunctionPass(ID) {}

    // The function should insert the buggy line numbers in the "bugs" vector.
    void checkUseBeforeDef(Instruction *I, unordered_set<Value *> *entrySet) {
      bool isBug = false;
      Value *V = dyn_cast<Value>(I);

      if (CallInst* call = dyn_cast<CallInst>(I)) {
        Function* function = call->getCalledFunction();
        output << function->getName() << '\n';
      }

      // If store instruction, investigate further
      if (isa<StoreInst>(I)) {
        // If 0th operand in entry set, add 1st operand to entry set (BUG)
        if (entrySet->find(I->getOperand(0)) != entrySet->end()) {
          entrySet->insert(I->getOperand(1));
          isBug = true;
        }
        // If 1st operand in entry set, remove 1st operand from entry set
        else if (entrySet->find(I->getOperand(1)) != entrySet->end()) {
          entrySet->erase(I->getOperand(1));
        }
      } 
      // If load instruction, investigate further
      else if (isa<LoadInst>(I)) {
        // If 0th operand in entry set, add value to entry set (BUG)
        if (entrySet->find(I->getOperand(0)) != entrySet->end()) {
          entrySet->insert(V);
          isBug = true;
        }
      }

      // Add bug line to bugs vector
      if (isBug) {
        int line = getSourceCodeLine(I);
        if (line > 0 && std::find(bugs.begin(), bugs.end(), line) == bugs.end()) bugs.push_back(line);
      }

      return;
    }

    // Function to return the line numbers that uses an undefined variable.
    bool runOnFunction(Function &F) override {
      std::string funcName = demangle(F.getName().str().c_str());

      // Remove all non user-defined functions and functions that start with '_' or have 'std'.
      if (F.isDeclaration() || funcName[0] == '_' || funcName.find("std") != std::string::npos) return false;

      // Remove all functions that we have previously encountered.
      if (funcNames.find(funcName) != funcNames.end()) return false;

      funcNames.insert(make_pair(funcName, true));

      // Iterate over basic blocks within function
      for (BasicBlock *BB : topoSortBBs(F)) {
        // Entry set formed from union of predecessors exit sets
        unordered_set<Value *> entrySet;
        
        // Iterate over predecessors of basic block
        for (BasicBlock *Pred : predecessors(BB)) {
          if (exitSets.find(Pred) != exitSets.end()) {
            // Get exit set for predecessor and add to entry set
            unordered_set<Value *> exitSet = exitSets.at(Pred); 
            entrySet.insert(exitSet.begin(), exitSet.end());
          }
        }

        // Iterate over instructions within basic block
        for (Instruction &I : *BB) checkUseBeforeDef(&I, &entrySet);

        // Exit set for basic block set to entry set
        exitSets.insert(make_pair(BB, entrySet));
      }

      // Export data from Set to Vector
      vector<int> temp;
      for (auto line : bugs) temp.push_back(line);

      // Sort vector
      std::sort(temp.begin(), temp.end());

      // Print the source code line number(s).
      for (auto line : temp) output << "Line " << line << ": " << "\n";

      // Print output
      errs() << output.str();
      output.flush();

      cleanGlobalVariables();
      return false;
    }
  };
}

char Assignment2::ID = 0;
static RegisterPass<Assignment2> X("taintanalysis", "Pass to find tainted variables");