//===- BasicInliner.cpp - Basic function level inliner --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple function based inliner that does not use
// call graph information. 
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "basicinliner"

#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/Transforms/Utils/BasicInliner.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <vector>

using namespace llvm;

static cl::opt<unsigned>     
BasicInlineThreshold("inline-threshold", cl::Hidden, cl::init(200),
   cl::desc("Control the amount of basic inlining to perform (default = 200)"));

namespace llvm {

  /// BasicInlinerImpl - BasicInliner implemantation class. This hides
  /// container info, used by basic inliner, from public interface.
  struct VISIBILITY_HIDDEN BasicInlinerImpl {
    
    BasicInlinerImpl(const BasicInlinerImpl&); // DO NOT IMPLEMENT
    void operator=(const BasicInlinerImpl&); // DO NO IMPLEMENT
  public:
    BasicInlinerImpl(TargetData *T) : TD(T) {}

    /// addFunction - Add function into the list of functions to process.
    /// All functions must be inserted using this interface before invoking
    /// inlineFunctions().
    void addFunction(Function *F) {
      Functions.push_back(F);
    }

    /// neverInlineFunction - Sometimes a function is never to be inlined 
    /// because of one or other reason. 
    void neverInlineFunction(Function *F) {
      NeverInline.insert(F);
    }

    /// inlineFuctions - Walk all call sites in all functions supplied by
    /// client. Inline as many call sites as possible. Delete completely
    /// inlined functions.
    void inlineFunctions();
    
  private:
    TargetData *TD;
    std::vector<Function *> Functions;
    SmallPtrSet<const Function *, 16> NeverInline;
    SmallPtrSet<Function *, 8> DeadFunctions;
    InlineCostAnalyzer CA;
  };

/// inlineFuctions - Walk all call sites in all functions supplied by
/// client. Inline as many call sites as possible. Delete completely
/// inlined functions.
void BasicInlinerImpl::inlineFunctions() {
      
  // Scan through and identify all call sites ahead of time so that we only
  // inline call sites in the original functions, not call sites that result
  // from inlining other functions.
  std::vector<CallSite> CallSites;
  
  for (std::vector<Function *>::iterator FI = Functions.begin(),
         FE = Functions.end(); FI != FE; ++FI) {
    Function *F = *FI;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::iterator I = BB->begin(); I != BB->end(); ++I) {
        CallSite CS = CallSite::get(I);
        if (CS.getInstruction() && CS.getCalledFunction()
            && !CS.getCalledFunction()->isDeclaration())
          CallSites.push_back(CS);
      }
  }
  
  DOUT << ": " << CallSites.size() << " call sites.\n";
  
  // Inline call sites.
  bool Changed = false;
  do {
    Changed = false;
    for (unsigned index = 0; index != CallSites.size() && !CallSites.empty(); 
         ++index) {
      CallSite CS = CallSites[index];
      if (Function *Callee = CS.getCalledFunction()) {
        
        // Eliminate calls that are never inlinable.
        if (Callee->isDeclaration() ||
            CS.getInstruction()->getParent()->getParent() == Callee) {
          CallSites.erase(CallSites.begin() + index);
          --index;
          continue;
        }
        InlineCost IC = CA.getInlineCost(CS, NeverInline);
        if (IC.isAlways()) {        
          DOUT << "  Inlining: cost=always"
               <<", call: " << *CS.getInstruction();
        } else if (IC.isNever()) {
          DOUT << "  NOT Inlining: cost=never"
               <<", call: " << *CS.getInstruction();
          continue;
        } else {
          int Cost = IC.getValue();
          
          if (Cost >= (int) BasicInlineThreshold) {
            DOUT << "  NOT Inlining: cost = " << Cost
                 << ", call: " <<  *CS.getInstruction();
            continue;
          } else {
            DOUT << "  Inlining: cost = " << Cost
                 << ", call: " <<  *CS.getInstruction();
          }
        }
        
        // Inline
        if (InlineFunction(CS, NULL, TD)) {
          if (Callee->use_empty() && Callee->hasLocalLinkage())
            DeadFunctions.insert(Callee);
          Changed = true;
          CallSites.erase(CallSites.begin() + index);
          --index;
        }
      }
    }
  } while (Changed);
  
  // Remove completely inlined functions from module.
  for(SmallPtrSet<Function *, 8>::iterator I = DeadFunctions.begin(),
        E = DeadFunctions.end(); I != E; ++I) {
    Function *D = *I;
    Module *M = D->getParent();
    M->getFunctionList().remove(D);
  }
}

BasicInliner::BasicInliner(TargetData *TD) {
  Impl = new BasicInlinerImpl(TD);
}

BasicInliner::~BasicInliner() {
  delete Impl;
}

/// addFunction - Add function into the list of functions to process.
/// All functions must be inserted using this interface before invoking
/// inlineFunctions().
void BasicInliner::addFunction(Function *F) {
  Impl->addFunction(F);
}

/// neverInlineFunction - Sometimes a function is never to be inlined because
/// of one or other reason. 
void BasicInliner::neverInlineFunction(Function *F) {
  Impl->neverInlineFunction(F);
}

/// inlineFuctions - Walk all call sites in all functions supplied by
/// client. Inline as many call sites as possible. Delete completely
/// inlined functions.
void BasicInliner::inlineFunctions() {
  Impl->inlineFunctions();
}

}
