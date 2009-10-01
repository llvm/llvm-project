//===-- IPConstantPropagation.cpp - Propagate constants through calls -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements an _extremely_ simple interprocedural constant
// propagation pass.  It could certainly be improved in many different ways,
// like using a worklist.  This pass makes arguments dead, but does not remove
// them.  The existing dead argument elimination pass should be run after this
// to clean up the mess.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ipconstprop"
#include "llvm/Transforms/IPO.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

STATISTIC(NumArgumentsProped, "Number of args turned into constants");
STATISTIC(NumReturnValProped, "Number of return values turned into constants");

namespace {
  /// IPCP - The interprocedural constant propagation pass
  ///
  struct VISIBILITY_HIDDEN IPCP : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    IPCP() : ModulePass(&ID) {}

    bool runOnModule(Module &M);
  private:
    bool PropagateConstantsIntoArguments(Function &F);
    bool PropagateConstantReturn(Function &F);
  };
}

char IPCP::ID = 0;
static RegisterPass<IPCP>
X("ipconstprop", "Interprocedural constant propagation");

ModulePass *llvm::createIPConstantPropagationPass() { return new IPCP(); }

bool IPCP::runOnModule(Module &M) {
  bool Changed = false;
  bool LocalChange = true;

  // FIXME: instead of using smart algorithms, we just iterate until we stop
  // making changes.
  while (LocalChange) {
    LocalChange = false;
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
      if (!I->isDeclaration()) {
        // Delete any klingons.
        I->removeDeadConstantUsers();
        if (I->hasLocalLinkage())
          LocalChange |= PropagateConstantsIntoArguments(*I);
        Changed |= PropagateConstantReturn(*I);
      }
    Changed |= LocalChange;
  }
  return Changed;
}

/// PropagateConstantsIntoArguments - Look at all uses of the specified
/// function.  If all uses are direct call sites, and all pass a particular
/// constant in for an argument, propagate that constant in as the argument.
///
bool IPCP::PropagateConstantsIntoArguments(Function &F) {
  if (F.arg_empty() || F.use_empty()) return false; // No arguments? Early exit.

  // For each argument, keep track of its constant value and whether it is a
  // constant or not.  The bool is driven to true when found to be non-constant.
  SmallVector<std::pair<Constant*, bool>, 16> ArgumentConstants;
  ArgumentConstants.resize(F.arg_size());

  unsigned NumNonconstant = 0;
  for (Value::use_iterator UI = F.use_begin(), E = F.use_end(); UI != E; ++UI) {
    // Used by a non-instruction, or not the callee of a function, do not
    // transform.
    if (!isa<CallInst>(*UI) && !isa<InvokeInst>(*UI))
      return false;
    
    CallSite CS = CallSite::get(cast<Instruction>(*UI));
    if (!CS.isCallee(UI))
      return false;

    // Check out all of the potentially constant arguments.  Note that we don't
    // inspect varargs here.
    CallSite::arg_iterator AI = CS.arg_begin();
    Function::arg_iterator Arg = F.arg_begin();
    for (unsigned i = 0, e = ArgumentConstants.size(); i != e;
         ++i, ++AI, ++Arg) {
      
      // If this argument is known non-constant, ignore it.
      if (ArgumentConstants[i].second)
        continue;
      
      Constant *C = dyn_cast<Constant>(*AI);
      if (C && ArgumentConstants[i].first == 0) {
        ArgumentConstants[i].first = C;   // First constant seen.
      } else if (C && ArgumentConstants[i].first == C) {
        // Still the constant value we think it is.
      } else if (*AI == &*Arg) {
        // Ignore recursive calls passing argument down.
      } else {
        // Argument became non-constant.  If all arguments are non-constant now,
        // give up on this function.
        if (++NumNonconstant == ArgumentConstants.size())
          return false;
        ArgumentConstants[i].second = true;
      }
    }
  }

  // If we got to this point, there is a constant argument!
  assert(NumNonconstant != ArgumentConstants.size());
  bool MadeChange = false;
  Function::arg_iterator AI = F.arg_begin();
  for (unsigned i = 0, e = ArgumentConstants.size(); i != e; ++i, ++AI) {
    // Do we have a constant argument?
    if (ArgumentConstants[i].second || AI->use_empty() || 
        (AI->hasByValAttr() && isa<PointerType>(AI->getType())))
      continue;
  
    Value *V = ArgumentConstants[i].first;
    if (V == 0) V = UndefValue::get(AI->getType());
    AI->replaceAllUsesWith(V);
    ++NumArgumentsProped;
    MadeChange = true;
  }
  return MadeChange;
}


// Check to see if this function returns one or more constants. If so, replace
// all callers that use those return values with the constant value. This will
// leave in the actual return values and instructions, but deadargelim will
// clean that up.
//
// Additionally if a function always returns one of its arguments directly,
// callers will be updated to use the value they pass in directly instead of
// using the return value.
bool IPCP::PropagateConstantReturn(Function &F) {
  if (F.getReturnType() == Type::getVoidTy(F.getContext()))
    return false; // No return value.

  // If this function could be overridden later in the link stage, we can't
  // propagate information about its results into callers.
  if (F.mayBeOverridden())
    return false;
    
  LLVMContext &Context = F.getContext();
  
  // Check to see if this function returns a constant.
  SmallVector<Value *,4> RetVals;
  const StructType *STy = dyn_cast<StructType>(F.getReturnType());
  if (STy)
    for (unsigned i = 0, e = STy->getNumElements(); i < e; ++i) 
      RetVals.push_back(UndefValue::get(STy->getElementType(i)));
  else
    RetVals.push_back(UndefValue::get(F.getReturnType()));

  unsigned NumNonConstant = 0;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
      for (unsigned i = 0, e = RetVals.size(); i != e; ++i) {
        // Already found conflicting return values?
        Value *RV = RetVals[i];
        if (!RV)
          continue;

        // Find the returned value
        Value *V;
        if (!STy)
          V = RI->getOperand(i);
        else
          V = FindInsertedValue(RI->getOperand(0), i, Context);

        if (V) {
          // Ignore undefs, we can change them into anything
          if (isa<UndefValue>(V))
            continue;
          
          // Try to see if all the rets return the same constant or argument.
          if (isa<Constant>(V) || isa<Argument>(V)) {
            if (isa<UndefValue>(RV)) {
              // No value found yet? Try the current one.
              RetVals[i] = V;
              continue;
            }
            // Returning the same value? Good.
            if (RV == V)
              continue;
          }
        }
        // Different or no known return value? Don't propagate this return
        // value.
        RetVals[i] = 0;
        // All values non constant? Stop looking.
        if (++NumNonConstant == RetVals.size())
          return false;
      }
    }

  // If we got here, the function returns at least one constant value.  Loop
  // over all users, replacing any uses of the return value with the returned
  // constant.
  bool MadeChange = false;
  for (Value::use_iterator UI = F.use_begin(), E = F.use_end(); UI != E; ++UI) {
    CallSite CS = CallSite::get(*UI);
    Instruction* Call = CS.getInstruction();

    // Not a call instruction or a call instruction that's not calling F
    // directly?
    if (!Call || !CS.isCallee(UI))
      continue;
    
    // Call result not used?
    if (Call->use_empty())
      continue;

    MadeChange = true;

    if (STy == 0) {
      Value* New = RetVals[0];
      if (Argument *A = dyn_cast<Argument>(New))
        // Was an argument returned? Then find the corresponding argument in
        // the call instruction and use that.
        New = CS.getArgument(A->getArgNo());
      Call->replaceAllUsesWith(New);
      continue;
    }
   
    for (Value::use_iterator I = Call->use_begin(), E = Call->use_end();
         I != E;) {
      Instruction *Ins = cast<Instruction>(*I);

      // Increment now, so we can remove the use
      ++I;

      // Find the index of the retval to replace with
      int index = -1;
      if (ExtractValueInst *EV = dyn_cast<ExtractValueInst>(Ins))
        if (EV->hasIndices())
          index = *EV->idx_begin();

      // If this use uses a specific return value, and we have a replacement,
      // replace it.
      if (index != -1) {
        Value *New = RetVals[index];
        if (New) {
          if (Argument *A = dyn_cast<Argument>(New))
            // Was an argument returned? Then find the corresponding argument in
            // the call instruction and use that.
            New = CS.getArgument(A->getArgNo());
          Ins->replaceAllUsesWith(New);
          Ins->eraseFromParent();
        }
      }
    }
  }

  if (MadeChange) ++NumReturnValProped;
  return MadeChange;
}
