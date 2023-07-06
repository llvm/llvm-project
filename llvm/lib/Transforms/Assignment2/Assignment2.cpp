#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
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

struct Variable {
  string name;
  bool tainted;
};

namespace {
  struct Assignment2 : public FunctionPass {
    static char ID;

    map<Value *, Variable> variables;

    map<Value *, bool> vars;

    set<Value *> tainted;

    // Reset all global variables when a new function is called.
    void cleanGlobalVariables() {
      output_str = "";
      variables.clear();
      vars.clear();
      tainted.clear();
    }

    Assignment2() : FunctionPass(ID) {}

    void checkTainted(Instruction *I, bool isInline) {
      Value *V = dyn_cast<Value>(I);
      int line = getSourceCodeLine(I);

      if (isa<CallInst>(I)) {
        output << "Call Inst: " << line << "\n";
        
        for (Use &U : I->operands()) {
          if (tainted.find(U.get()) != tainted.end()) {
            output << "Tainted: " << V << "\n";
            tainted.insert(V);
            break;
          } 
        }
      } 
      else if (isa<StoreInst>(I)) {
        output << "Store Inst: " << line << "\n";

        if (tainted.find(I->getOperand(0)) != tainted.end()) {
          output << "Tainted: " << I->getOperand(1) << "\n";
          tainted.insert(I->getOperand(1));
        } 
        else if (tainted.find(I->getOperand(1)) != tainted.end() && isInline) {
          output << "Untainted: " << I->getOperand(1) << "\n";
          tainted.erase(I->getOperand(1));
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

      output << "MAIN" << "\n";

      // Iterate over basic blocks within function
      for (BasicBlock *BB : topoSortBBs(F)) {        
        // Iterate over instructions within basic block
        for (Instruction &I : *BB) {

          if (isa<AllocaInst>(I)) {
            Value *V = dyn_cast<Value>(&I);
            vars.insert(make_pair(V, false));
            
            for (User *U : V->users()) {
              if (CallInst *CI = dyn_cast<CallInst>(U)) {
                if (demangle(CI->getOperand(0)->getName().str().c_str()) == source) {
                  map<Value *, bool>::iterator it = vars.find(V);
                  if (it != vars.end()) {
                    it->second = true;
                    output << "Line " << getSourceCodeLine(CI) << ": " << V << " is tainted" << "\n";
                  }
                }
              }
            }
          }
        }
      }  

      for (auto &var : vars) {
        output << "Variable: " << var.first << "\n";
        output << "Tainted: " << var.second << "\n";
      }

      // string solution = "";
      // for (auto &var : variables) {
      //   if (var.second.tainted) {
      //     if (solution.size() == 0) solution += var.second.name;
      //     else solution += "," + var.second.name;
      //   }
      // }

      // output << "Tainted: {" << solution << "}" << "\n";

      // Print output
      errs() << output.str();
      output.flush();

      cleanGlobalVariables();
      return false;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<PostDominatorTreeWrapperPass>();
      AU.setPreservesAll();
    }
  };
}

char Assignment2::ID = 0;
static RegisterPass<Assignment2> X("taintanalysis", "Pass to find tainted variables");