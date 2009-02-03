//===-- PartialSpecialization.cpp - Specialize for common constants--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass finds function arguments that are often a common constant and 
// specializes a version of the called function for that constant.
//
// This pass simply does the cloning for functions it specializes.  It depends
// on IPSCCP and DAE to clean up the results.
//
// The initial heuristic favors constant arguments that are used in control 
// flow.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "partialspecialization"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
using namespace llvm;

STATISTIC(numSpecialized, "Number of specialized functions created");

// Call must be used at least occasionally
static const int CallsMin = 5;

// Must have 10% of calls having the same constant to specialize on
static const double ConstValPercent = .1;

namespace {
  class VISIBILITY_HIDDEN PartSpec : public ModulePass {
    void scanForInterest(Function&, SmallVector<int, 6>&);
    int scanDistribution(Function&, int, std::map<Constant*, int>&);
  public :
    static char ID; // Pass identification, replacement for typeid
    PartSpec() : ModulePass(&ID) {}
    bool runOnModule(Module &M);
  };
}

char PartSpec::ID = 0;
static RegisterPass<PartSpec>
X("partialspecialization", "Partial Specialization");

// Specialize F by replacing the arguments (keys) in replacements with the 
// constants (values).  Replace all calls to F with those constants with
// a call to the specialized function.  Returns the specialized function
static Function* 
SpecializeFunction(Function* F, 
                   DenseMap<const Value*, Value*>& replacements) {
  // arg numbers of deleted arguments
  DenseSet<unsigned> deleted;
  for (DenseMap<const Value*, Value*>::iterator 
         repb = replacements.begin(), repe = replacements.end();
       repb != repe; ++ repb)
    deleted.insert(cast<Argument>(repb->first)->getArgNo());

  Function* NF = CloneFunction(F, replacements);
  NF->setLinkage(GlobalValue::InternalLinkage);
  F->getParent()->getFunctionList().push_back(NF);

  for (Value::use_iterator ii = F->use_begin(), ee = F->use_end(); 
       ii != ee; ) {
    Value::use_iterator i = ii;;
    ++ii;
    if (isa<CallInst>(i) || isa<InvokeInst>(i)) {
      CallSite CS(cast<Instruction>(i));
      if (CS.getCalledFunction() == F) {
        
        SmallVector<Value*, 6> args;
        for (unsigned x = 0; x < CS.arg_size(); ++x)
          if (!deleted.count(x))
            args.push_back(CS.getArgument(x));
        Value* NCall;
        if (isa<CallInst>(i))
          NCall = CallInst::Create(NF, args.begin(), args.end(), 
                                   CS.getInstruction()->getName(), 
                                   CS.getInstruction());
        else
          NCall = InvokeInst::Create(NF, cast<InvokeInst>(i)->getNormalDest(),
                                     cast<InvokeInst>(i)->getUnwindDest(),
                                     args.begin(), args.end(), 
                                     CS.getInstruction()->getName(), 
                                     CS.getInstruction());
        CS.getInstruction()->replaceAllUsesWith(NCall);
        CS.getInstruction()->eraseFromParent();
      }
    }
  }
  return NF;
}


bool PartSpec::runOnModule(Module &M) {
  bool Changed = false;
  for (Module::iterator I = M.begin(); I != M.end(); ++I) {
    Function &F = *I;
    if (F.isDeclaration() || F.mayBeOverridden()) continue;
    SmallVector<int, 6> interestingArgs;
    scanForInterest(F, interestingArgs);

    // Find the first interesting Argument that we can specialize on
    // If there are multiple interesting Arguments, then those will be found
    // when processing the cloned function.
    bool breakOuter = false;
    for (unsigned int x = 0; !breakOuter && x < interestingArgs.size(); ++x) {
      std::map<Constant*, int> distribution;
      int total = scanDistribution(F, interestingArgs[x], distribution);
      if (total > CallsMin) 
        for (std::map<Constant*, int>::iterator ii = distribution.begin(),
               ee = distribution.end(); ii != ee; ++ii)
          if (total > ii->second && ii->first &&
               ii->second > total * ConstValPercent) {
            DenseMap<const Value*, Value*> m;
            Function::arg_iterator arg = F.arg_begin();
            for (int y = 0; y < interestingArgs[x]; ++y)
              ++arg;
            m[&*arg] = ii->first;
            SpecializeFunction(&F, m);
            ++numSpecialized;
            breakOuter = true;
            Changed = true;
          }
    }
  }
  return Changed;
}

/// scanForInterest - This function decides which arguments would be worth
/// specializing on.
void PartSpec::scanForInterest(Function& F, SmallVector<int, 6>& args) {
  for(Function::arg_iterator ii = F.arg_begin(), ee = F.arg_end();
      ii != ee; ++ii) {
    for(Value::use_iterator ui = ii->use_begin(), ue = ii->use_end();
        ui != ue; ++ui) {

      bool interesting = false;

      if (isa<CmpInst>(ui)) interesting = true;
      else if (isa<CallInst>(ui))
        interesting = ui->getOperand(0) == ii;
      else if (isa<InvokeInst>(ui))
        interesting = ui->getOperand(0) == ii;
      else if (isa<SwitchInst>(ui)) interesting = true;
      else if (isa<BranchInst>(ui)) interesting = true;

      if (interesting) {
        args.push_back(std::distance(F.arg_begin(), ii));
        break;
      }
    }
  }
}

/// scanDistribution - Construct a histogram of constants for arg of F at arg.
int PartSpec::scanDistribution(Function& F, int arg, 
                               std::map<Constant*, int>& dist) {
  bool hasIndirect = false;
  int total = 0;
  for(Value::use_iterator ii = F.use_begin(), ee = F.use_end();
      ii != ee; ++ii)
    if ((isa<CallInst>(ii) || isa<InvokeInst>(ii))
        && ii->getOperand(0) == &F) {
      ++dist[dyn_cast<Constant>(ii->getOperand(arg + 1))];
      ++total;
    } else
      hasIndirect = true;

  // Preserve the original address taken function even if all other uses
  // will be specialized.
  if (hasIndirect) ++total;
  return total;
}

ModulePass* llvm::createPartialSpecializationPass() { return new PartSpec(); }
