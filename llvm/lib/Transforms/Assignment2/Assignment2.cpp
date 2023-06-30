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
#include <map>
#include <set>
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

    // Keep track of all the functions we have encountered so far.
    unordered_map<string, bool> funcNames;

    map<Value *, int> tainted;

    set<Instruction *> worklist;

    // Reset all global variables when a new function is called.
    void cleanGlobalVariables() {
      output_str = "";
      tainted.clear();
      worklist.clear();
    }

    Assignment2() : FunctionPass(ID) {}

    void updateWorklist(Value *V, Instruction *I) {
      output << "Tainted: " << V << " at " << getSourceCodeLine(I) << "\n";
      tainted.insert(make_pair(V, getSourceCodeLine(I)));
      for (User *U : V->users()) {
        if (Instruction *Inst = dyn_cast<Instruction>(U)) {
          worklist.insert(Inst);
        }
      }
    }

    void checkTainted(Instruction *I) {
      Value *V = dyn_cast<Value>(I);
      if (isa<StoreInst>(I)) {
        output << "Store Inst: " << getSourceCodeLine(I) << "\n";
        if (tainted.find(I->getOperand(0)) != tainted.end()) {
          updateWorklist(I->getOperand(1), I);
        } else if (tainted.find(I->getOperand(1)) != tainted.end()) {
          output << "Untainted: " << I->getOperand(1) << " at " << getSourceCodeLine(I) << "\n";
          tainted.erase(I->getOperand(1));
        }
      } else if (isa<LoadInst>(I)) {
        output << "Load Inst: " << getSourceCodeLine(I) << "\n";
        if (tainted.find(I->getOperand(0)) != tainted.end()) {
          updateWorklist(V, I);
        }
      } else if (isa<CmpInst>(I)) {
        output << "Cmp Inst: " << getSourceCodeLine(I) << "\n";
        if (tainted.find(I->getOperand(0)) != tainted.end()) {
          output << "Untainted: " << I->getOperand(0) << " at " << getSourceCodeLine(I) << "\n";
          tainted.erase(I->getOperand(0));
        } else if (tainted.find(I->getOperand(1)) != tainted.end()) {
          output << "Untainted: " << I->getOperand(1) << " at " << getSourceCodeLine(I) << "\n";
          tainted.erase(I->getOperand(1));
        }
      }
    }

    void findSource(Function &F) {   
      // Iterate over basic blocks within function
      for (BasicBlock *BB : topoSortBBs(F)) {
        // Iterate over instructions within basic block
        for (Instruction &I : *BB) {
          if (isa<CallInst>(I) && demangle(I.getOperand(0)->getName().str().c_str()) == "std::__1::cin") {
            updateWorklist(I.getOperand(1), &I);
          }
        }
      }  
    }

    // Function to return the line numbers that uses an undefined variable.
    bool runOnFunction(Function &F) override {
      std::string funcName = demangle(F.getName().str().c_str());

      // Remove all non user-defined functions and functions that start with '_' or have 'std'.
      if (F.isDeclaration() || funcName[0] == '_' || funcName.find("std") != std::string::npos) return false;

      // Remove all functions that we have previously encountered.
      if (funcNames.find(funcName) != funcNames.end()) return false;

      funcNames.insert(make_pair(funcName, true));

      findSource(F);

      while (worklist.size() > 0) {
        set<Instruction *>::iterator curr = worklist.begin();
        checkTainted(*curr);
        worklist.erase(curr);
      }

      for (auto &taint : tainted) {
        output << "Line " << taint.second << ": " << taint.first << "\n";
      }

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