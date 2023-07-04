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

const string func = "main";
const string source = "std::__1::cin";

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

    map<Value *, int> variables;

    set<Value *> tainted;

    // Reset all global variables when a new function is called.
    void cleanGlobalVariables() {
      output_str = "";
      variables.clear();
      tainted.clear();
    }

    Assignment2() : FunctionPass(ID) {}

    void updateVariable(Value *V, int line) {
      map<Value *, int>::iterator it = variables.find(V);
      if (it != variables.end()) {
        it->second = line;
        if (line != -1) {
          output << "Line " << line << ": " << V << " is tainted" << "\n";
        } else {
          output << "Line " << line << ": " << V << " is now untainted" << "\n";
        }
      }
    }

    void checkTainted(Instruction *I) {
      Value *V = dyn_cast<Value>(I);
      int line = getSourceCodeLine(I);

      if (isa<AllocaInst>(I)) {
        output << "Alloca Inst: " << line << "\n";
        variables.insert(make_pair(V, -1));
      } 
      else if (isa<CallInst>(I)) {
        output << "Call Inst: " << line << "\n";
        
        if (demangle(I->getOperand(0)->getName().str().c_str()) == source) {
          output << "Tainted: " << I->getOperand(1) << "\n";
          tainted.insert(I->getOperand(1));
          updateVariable(I->getOperand(1), line);
        } else {
          for (Use &U : I->operands()) {
            Value *v = U.get();
            if (tainted.find(v) != tainted.end()) {
              output << "Tainted: " << V << "\n";
              tainted.insert(V);
              break;
            } 
          }
        }
      } 
      else if (isa<StoreInst>(I)) {
        output << "Store Inst: " << line << "\n";

        if (tainted.find(I->getOperand(0)) != tainted.end()) {
          output << "Tainted: " << I->getOperand(1) << "\n";
          tainted.insert(I->getOperand(1));
          updateVariable(I->getOperand(1), line);
        } 
        else if (tainted.find(I->getOperand(1)) != tainted.end()) {
          output << "Untainted: " << I->getOperand(1) << "\n";
          tainted.erase(I->getOperand(1));
          updateVariable(I->getOperand(1), -1);
        }
      } 
      else if (isa<LoadInst>(I)) {
        output << "Load Inst: " << line << "\n";

        if (tainted.find(I->getOperand(0)) != tainted.end()) {
          output << "Tainted: " << V << "\n";
          tainted.insert(V);
        }
      }
    }

    bool runOnFunction(Function &F) override {
      // We only want to examine the main method
      if (demangle(F.getName().str().c_str()) != func) return false;

      // Iterate over basic blocks within function
      for (BasicBlock *BB : topoSortBBs(F)) {
        // Iterate over instructions within basic block
        for (Instruction &I : *BB) checkTainted(&I);
      }  

      output << "Tainted: {";
      for (auto &var : variables) {
        if (var.second != -1) output << var.first << ",";
      }
      output << "}" << "\n";

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