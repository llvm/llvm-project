//===- InlineSimple.cpp - Code to perform simple function inlining --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#include "Inliner.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Type.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Transforms/IPO.h"
using namespace llvm;

namespace {
  struct ArgInfo {
    unsigned ConstantWeight;
    unsigned AllocaWeight;

    ArgInfo(unsigned CWeight, unsigned AWeight)
      : ConstantWeight(CWeight), AllocaWeight(AWeight) {}
  };

  // FunctionInfo - For each function, calculate the size of it in blocks and
  // instructions.
  struct FunctionInfo {
    // HasAllocas - Keep track of whether or not a function contains an alloca
    // instruction that is not in the entry block of the function.  Inlining
    // this call could cause us to blow out the stack, because the stack memory
    // would never be released.
    //
    // FIXME: LLVM needs a way of dealloca'ing memory, which would make this
    // irrelevant!
    //
    bool HasAllocas;

    // NumInsts, NumBlocks - Keep track of how large each function is, which is
    // used to estimate the code size cost of inlining it.
    unsigned NumInsts, NumBlocks;

    // ArgumentWeights - Each formal argument of the function is inspected to
    // see if it is used in any contexts where making it a constant or alloca
    // would reduce the code size.  If so, we add some value to the argument
    // entry here.
    std::vector<ArgInfo> ArgumentWeights;

    FunctionInfo() : HasAllocas(false), NumInsts(0), NumBlocks(0) {}

    /// analyzeFunction - Fill in the current structure with information gleaned
    /// from the specified function.
    void analyzeFunction(Function *F);
  };

  class SimpleInliner : public Inliner {
    std::map<const Function*, FunctionInfo> CachedFunctionInfo;
  public:
    int getInlineCost(CallSite CS);
  };
  RegisterOpt<SimpleInliner> X("inline", "Function Integration/Inlining");
}

Pass *llvm::createFunctionInliningPass() { return new SimpleInliner(); }

// CountCodeReductionForConstant - Figure out an approximation for how many
// instructions will be constant folded if the specified value is constant.
//
static unsigned CountCodeReductionForConstant(Value *V) {
  unsigned Reduction = 0;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (isa<BranchInst>(*UI))
      Reduction += 40;          // Eliminating a conditional branch is a big win
    else if (SwitchInst *SI = dyn_cast<SwitchInst>(*UI))
      // Eliminating a switch is a big win, proportional to the number of edges
      // deleted.
      Reduction += (SI->getNumSuccessors()-1) * 40;
    else if (CallInst *CI = dyn_cast<CallInst>(*UI)) {
      // Turning an indirect call into a direct call is a BIG win
      Reduction += CI->getCalledValue() == V ? 500 : 0;
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(*UI)) {
      // Turning an indirect call into a direct call is a BIG win
      Reduction += II->getCalledValue() == V ? 500 : 0;
    } else {
      // Figure out if this instruction will be removed due to simple constant
      // propagation.
      Instruction &Inst = cast<Instruction>(**UI);
      bool AllOperandsConstant = true;
      for (unsigned i = 0, e = Inst.getNumOperands(); i != e; ++i)
        if (!isa<Constant>(Inst.getOperand(i)) && Inst.getOperand(i) != V) {
          AllOperandsConstant = false;
          break;
        }

      if (AllOperandsConstant) {
        // We will get to remove this instruction...
        Reduction += 7;
        
        // And any other instructions that use it which become constants
        // themselves.
        Reduction += CountCodeReductionForConstant(&Inst);
      }
    }

  return Reduction;
}

// CountCodeReductionForAlloca - Figure out an approximation of how much smaller
// the function will be if it is inlined into a context where an argument
// becomes an alloca.
//
static unsigned CountCodeReductionForAlloca(Value *V) {
  if (!isa<PointerType>(V->getType())) return 0;  // Not a pointer
  unsigned Reduction = 0;
  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E;++UI){
    Instruction *I = cast<Instruction>(*UI);
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      Reduction += 10;
    else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
      // If the GEP has variable indices, we won't be able to do much with it.
      for (Instruction::op_iterator I = GEP->op_begin()+1, E = GEP->op_end();
           I != E; ++I)
        if (!isa<Constant>(*I)) return 0;
      Reduction += CountCodeReductionForAlloca(GEP)+15;
    } else {
      // If there is some other strange instruction, we're not going to be able
      // to do much if we inline this.
      return 0;
    }
  }

  return Reduction;
}

/// analyzeFunction - Fill in the current structure with information gleaned
/// from the specified function.
void FunctionInfo::analyzeFunction(Function *F) {
  unsigned NumInsts = 0, NumBlocks = 0;

  // Look at the size of the callee.  Each basic block counts as 20 units, and
  // each instruction counts as 10.
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator II = BB->begin(), E = BB->end();
         II != E; ++II) {
      ++NumInsts;

      // If there is an alloca in the body of the function, we cannot currently
      // inline the function without the risk of exploding the stack.
      if (isa<AllocaInst>(II) && BB != F->begin()) {
        HasAllocas = true;
        this->NumBlocks = this->NumInsts = 1;
        return;
      }
    }

    ++NumBlocks;
  }

  this->NumBlocks = NumBlocks;
  this->NumInsts  = NumInsts;

  // Check out all of the arguments to the function, figuring out how much
  // code can be eliminated if one of the arguments is a constant.
  for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    ArgumentWeights.push_back(ArgInfo(CountCodeReductionForConstant(I),
                                      CountCodeReductionForAlloca(I)));
}


// getInlineCost - The heuristic used to determine if we should inline the
// function call or not.
//
int SimpleInliner::getInlineCost(CallSite CS) {
  Instruction *TheCall = CS.getInstruction();
  Function *Callee = CS.getCalledFunction();
  const Function *Caller = TheCall->getParent()->getParent();

  // Don't inline a directly recursive call.
  if (Caller == Callee) return 2000000000;

  // InlineCost - This value measures how good of an inline candidate this call
  // site is to inline.  A lower inline cost make is more likely for the call to
  // be inlined.  This value may go negative.
  //
  int InlineCost = 0;

  // If there is only one call of the function, and it has internal linkage,
  // make it almost guaranteed to be inlined.
  //
  if (Callee->hasInternalLinkage() && Callee->hasOneUse())
    InlineCost -= 30000;

  // Get information about the callee...
  FunctionInfo &CalleeFI = CachedFunctionInfo[Callee];

  // If we haven't calculated this information yet, do so now.
  if (CalleeFI.NumBlocks == 0)
    CalleeFI.analyzeFunction(Callee);

  // Don't inline calls to functions with allocas that are not in the entry
  // block of the function.
  if (CalleeFI.HasAllocas)
    return 2000000000;

  // Add to the inline quality for properties that make the call valuable to
  // inline.  This includes factors that indicate that the result of inlining
  // the function will be optimizable.  Currently this just looks at arguments
  // passed into the function.
  //
  unsigned ArgNo = 0;
  for (CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
       I != E; ++I, ++ArgNo) {
    // Each argument passed in has a cost at both the caller and the callee
    // sides.  This favors functions that take many arguments over functions
    // that take few arguments.
    InlineCost -= 20;

    // If this is a function being passed in, it is very likely that we will be
    // able to turn an indirect function call into a direct function call.
    if (isa<Function>(I))
      InlineCost -= 100;

    // If an alloca is passed in, inlining this function is likely to allow
    // significant future optimization possibilities (like scalar promotion, and
    // scalarization), so encourage the inlining of the function.
    //
    else if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) {
      if (ArgNo < CalleeFI.ArgumentWeights.size())
        InlineCost -= CalleeFI.ArgumentWeights[ArgNo].AllocaWeight;

    // If this is a constant being passed into the function, use the argument
    // weights calculated for the callee to determine how much will be folded
    // away with this information.
    } else if (isa<Constant>(I)) {
      if (ArgNo < CalleeFI.ArgumentWeights.size())
        InlineCost -= CalleeFI.ArgumentWeights[ArgNo].ConstantWeight;
    }
  }

  // Now that we have considered all of the factors that make the call site more
  // likely to be inlined, look at factors that make us not want to inline it.

  // Don't inline into something too big, which would make it bigger.  Here, we
  // count each basic block as a single unit.
  //
  InlineCost += Caller->size()/20;


  // Look at the size of the callee.  Each basic block counts as 20 units, and
  // each instruction counts as 5.
  InlineCost += CalleeFI.NumInsts*5 + CalleeFI.NumBlocks*20;
  return InlineCost;
}

