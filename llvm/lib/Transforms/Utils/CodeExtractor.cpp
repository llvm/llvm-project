//===- CodeExtractor.cpp - Pull code region into a new function -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the interface to tear out a code region, such as an
// individual loop or a parallel section, into a new function, replacing it with
// a call to the new function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/StringExtras.h"
#include <algorithm>
#include <set>
using namespace llvm;

// Provide a command-line option to aggregate function arguments into a struct
// for functions produced by the code extrator. This is useful when converting
// extracted functions to pthread-based code, as only one argument (void*) can
// be passed in to pthread_create().
static cl::opt<bool>
AggregateArgsOpt("aggregate-extracted-args", cl::Hidden,
                 cl::desc("Aggregate arguments to code-extracted functions"));

namespace {
  class CodeExtractor {
    typedef std::vector<Value*> Values;
    std::set<BasicBlock*> BlocksToExtract;
    DominatorSet *DS;
    bool AggregateArgs;
    unsigned NumExitBlocks;
    const Type *RetTy;
  public:
    CodeExtractor(DominatorSet *ds = 0, bool AggArgs = false)
      : DS(ds), AggregateArgs(AggregateArgsOpt), NumExitBlocks(~0U) {}

    Function *ExtractCodeRegion(const std::vector<BasicBlock*> &code);

    bool isEligible(const std::vector<BasicBlock*> &code);

  private:
    /// definedInRegion - Return true if the specified value is defined in the
    /// extracted region.
    bool definedInRegion(Value *V) const {
      if (Instruction *I = dyn_cast<Instruction>(V))
        if (BlocksToExtract.count(I->getParent()))
          return true;
      return false;
    }
    
    /// definedInCaller - Return true if the specified value is defined in the
    /// function being code extracted, but not in the region being extracted.
    /// These values must be passed in as live-ins to the function.
    bool definedInCaller(Value *V) const {
      if (isa<Argument>(V)) return true;
      if (Instruction *I = dyn_cast<Instruction>(V))
        if (!BlocksToExtract.count(I->getParent()))
          return true;
      return false;
    }

    void severSplitPHINodes(BasicBlock *&Header);
    void splitReturnBlocks();
    void findInputsOutputs(Values &inputs, Values &outputs);

    Function *constructFunction(const Values &inputs,
                                const Values &outputs,
                                BasicBlock *header,
                                BasicBlock *newRootNode, BasicBlock *newHeader,
                                Function *oldFunction, Module *M);

    void moveCodeToFunction(Function *newFunction);

    void emitCallAndSwitchStatement(Function *newFunction,
                                    BasicBlock *newHeader,
                                    Values &inputs,
                                    Values &outputs);

  };
}

/// severSplitPHINodes - If a PHI node has multiple inputs from outside of the
/// region, we need to split the entry block of the region so that the PHI node
/// is easier to deal with.
void CodeExtractor::severSplitPHINodes(BasicBlock *&Header) {
  bool HasPredsFromRegion = false;
  unsigned NumPredsOutsideRegion = 0;

  if (Header != &Header->getParent()->front()) {
    PHINode *PN = dyn_cast<PHINode>(Header->begin());
    if (!PN) return;  // No PHI nodes.

    // If the header node contains any PHI nodes, check to see if there is more
    // than one entry from outside the region.  If so, we need to sever the
    // header block into two.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (BlocksToExtract.count(PN->getIncomingBlock(i)))
        HasPredsFromRegion = true;
      else
        ++NumPredsOutsideRegion;

    // If there is one (or fewer) predecessor from outside the region, we don't
    // need to do anything special.
    if (NumPredsOutsideRegion <= 1) return;
  }

  // Otherwise, we need to split the header block into two pieces: one
  // containing PHI nodes merging values from outside of the region, and a
  // second that contains all of the code for the block and merges back any
  // incoming values from inside of the region.
  BasicBlock::iterator AfterPHIs = Header->begin();
  while (isa<PHINode>(AfterPHIs)) ++AfterPHIs;
  BasicBlock *NewBB = Header->splitBasicBlock(AfterPHIs,
                                              Header->getName()+".ce");

  // We only want to code extract the second block now, and it becomes the new
  // header of the region.
  BasicBlock *OldPred = Header;
  BlocksToExtract.erase(OldPred);
  BlocksToExtract.insert(NewBB);
  Header = NewBB;

  // Okay, update dominator sets. The blocks that dominate the new one are the
  // blocks that dominate TIBB plus the new block itself.
  if (DS) {
    DominatorSet::DomSetType DomSet = DS->getDominators(OldPred);
    DomSet.insert(NewBB);  // A block always dominates itself.
    DS->addBasicBlock(NewBB, DomSet);

    // Additionally, NewBB dominates all blocks in the function that are
    // dominated by OldPred.
    Function *F = Header->getParent();
    for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
      if (DS->properlyDominates(OldPred, I))
        DS->addDominator(I, NewBB);
  }

  // Okay, now we need to adjust the PHI nodes and any branches from within the
  // region to go to the new header block instead of the old header block.
  if (HasPredsFromRegion) {
    PHINode *PN = cast<PHINode>(OldPred->begin());
    // Loop over all of the predecessors of OldPred that are in the region,
    // changing them to branch to NewBB instead.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (BlocksToExtract.count(PN->getIncomingBlock(i))) {
        TerminatorInst *TI = PN->getIncomingBlock(i)->getTerminator();
        TI->replaceUsesOfWith(OldPred, NewBB);
      }

    // Okay, everthing within the region is now branching to the right block, we
    // just have to update the PHI nodes now, inserting PHI nodes into NewBB.
    for (AfterPHIs = OldPred->begin();
         PHINode *PN = dyn_cast<PHINode>(AfterPHIs); ++AfterPHIs) {
      // Create a new PHI node in the new region, which has an incoming value
      // from OldPred of PN.
      PHINode *NewPN = new PHINode(PN->getType(), PN->getName()+".ce",
                                   NewBB->begin());
      NewPN->addIncoming(PN, OldPred);

      // Loop over all of the incoming value in PN, moving them to NewPN if they
      // are from the extracted region.
      for (unsigned i = 0; i != PN->getNumIncomingValues(); ++i) {
        if (BlocksToExtract.count(PN->getIncomingBlock(i))) {
          NewPN->addIncoming(PN->getIncomingValue(i), PN->getIncomingBlock(i));
          PN->removeIncomingValue(i);
          --i;
        }
      }
    }
  }
}

void CodeExtractor::splitReturnBlocks() {
  for (std::set<BasicBlock*>::iterator I = BlocksToExtract.begin(),
         E = BlocksToExtract.end(); I != E; ++I)
    if (ReturnInst *RI = dyn_cast<ReturnInst>((*I)->getTerminator()))
      (*I)->splitBasicBlock(RI, (*I)->getName()+".ret");
}

// findInputsOutputs - Find inputs to, outputs from the code region.
//
void CodeExtractor::findInputsOutputs(Values &inputs, Values &outputs) {
  std::set<BasicBlock*> ExitBlocks;
  for (std::set<BasicBlock*>::const_iterator ci = BlocksToExtract.begin(), 
       ce = BlocksToExtract.end(); ci != ce; ++ci) {
    BasicBlock *BB = *ci;

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // If a used value is defined outside the region, it's an input.  If an
      // instruction is used outside the region, it's an output.
      for (User::op_iterator O = I->op_begin(), E = I->op_end(); O != E; ++O)
        if (definedInCaller(*O))
          inputs.push_back(*O);
      
      // Consider uses of this instruction (outputs).
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
           UI != E; ++UI)
        if (!definedInRegion(*UI)) {
          outputs.push_back(I);
          break;
        }
    } // for: insts

    // Keep track of the exit blocks from the region.
    TerminatorInst *TI = BB->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      if (!BlocksToExtract.count(TI->getSuccessor(i)))
        ExitBlocks.insert(TI->getSuccessor(i));
  } // for: basic blocks

  NumExitBlocks = ExitBlocks.size();

  // Eliminate duplicates.
  std::sort(inputs.begin(), inputs.end());
  inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
  std::sort(outputs.begin(), outputs.end());
  outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
}

/// constructFunction - make a function based on inputs and outputs, as follows:
/// f(in0, ..., inN, out0, ..., outN)
///
Function *CodeExtractor::constructFunction(const Values &inputs,
                                           const Values &outputs,
                                           BasicBlock *header,
                                           BasicBlock *newRootNode,
                                           BasicBlock *newHeader,
                                           Function *oldFunction,
                                           Module *M) {
  DEBUG(std::cerr << "inputs: " << inputs.size() << "\n");
  DEBUG(std::cerr << "outputs: " << outputs.size() << "\n");

  // This function returns unsigned, outputs will go back by reference.
  switch (NumExitBlocks) {
  case 0:
  case 1: RetTy = Type::VoidTy; break;
  case 2: RetTy = Type::BoolTy; break;
  default: RetTy = Type::UShortTy; break;
  }

  std::vector<const Type*> paramTy;

  // Add the types of the input values to the function's argument list
  for (Values::const_iterator i = inputs.begin(),
         e = inputs.end(); i != e; ++i) {
    const Value *value = *i;
    DEBUG(std::cerr << "value used in func: " << *value << "\n");
    paramTy.push_back(value->getType());
  }

  // Add the types of the output values to the function's argument list.
  for (Values::const_iterator I = outputs.begin(), E = outputs.end();
       I != E; ++I) {
    DEBUG(std::cerr << "instr used in func: " << **I << "\n");
    if (AggregateArgs)
      paramTy.push_back((*I)->getType());
    else
      paramTy.push_back(PointerType::get((*I)->getType()));
  }

  DEBUG(std::cerr << "Function type: " << *RetTy << " f(");
  DEBUG(for (std::vector<const Type*>::iterator i = paramTy.begin(),
               e = paramTy.end(); i != e; ++i) std::cerr << **i << ", ");
  DEBUG(std::cerr << ")\n");

  if (AggregateArgs && (inputs.size() + outputs.size() > 0)) {
    PointerType *StructPtr = PointerType::get(StructType::get(paramTy));
    paramTy.clear();
    paramTy.push_back(StructPtr);
  }
  const FunctionType *funcType = FunctionType::get(RetTy, paramTy, false);

  // Create the new function
  Function *newFunction = new Function(funcType,
                                       GlobalValue::InternalLinkage,
                                       oldFunction->getName() + "_" +
                                       header->getName(), M);
  newFunction->getBasicBlockList().push_back(newRootNode);

  // Create an iterator to name all of the arguments we inserted.
  Function::aiterator AI = newFunction->abegin();

  // Rewrite all users of the inputs in the extracted region to use the
  // arguments (or appropriate addressing into struct) instead.
  for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
    Value *RewriteVal;
    if (AggregateArgs) {
      std::vector<Value*> Indices;
      Indices.push_back(Constant::getNullValue(Type::UIntTy));
      Indices.push_back(ConstantUInt::get(Type::UIntTy, i));
      std::string GEPname = "gep_" + inputs[i]->getName();
      TerminatorInst *TI = newFunction->begin()->getTerminator();
      GetElementPtrInst *GEP = new GetElementPtrInst(AI, Indices, GEPname, TI);
      RewriteVal = new LoadInst(GEP, "load" + GEPname, TI);
    } else
      RewriteVal = AI++;

    std::vector<User*> Users(inputs[i]->use_begin(), inputs[i]->use_end());
    for (std::vector<User*>::iterator use = Users.begin(), useE = Users.end();
         use != useE; ++use)
      if (Instruction* inst = dyn_cast<Instruction>(*use))
        if (BlocksToExtract.count(inst->getParent()))
          inst->replaceUsesOfWith(inputs[i], RewriteVal);
  }

  // Set names for input and output arguments.
  if (!AggregateArgs) {
    AI = newFunction->abegin();
    for (unsigned i = 0, e = inputs.size(); i != e; ++i, ++AI)
      AI->setName(inputs[i]->getName());
    for (unsigned i = 0, e = outputs.size(); i != e; ++i, ++AI)
      AI->setName(outputs[i]->getName()+".out");  
  }

  // Rewrite branches to basic blocks outside of the loop to new dummy blocks
  // within the new function. This must be done before we lose track of which
  // blocks were originally in the code region.
  std::vector<User*> Users(header->use_begin(), header->use_end());
  for (unsigned i = 0, e = Users.size(); i != e; ++i)
    // The BasicBlock which contains the branch is not in the region
    // modify the branch target to a new block
    if (TerminatorInst *TI = dyn_cast<TerminatorInst>(Users[i]))
      if (!BlocksToExtract.count(TI->getParent()) &&
          TI->getParent()->getParent() == oldFunction)
        TI->replaceUsesOfWith(header, newHeader);

  return newFunction;
}

/// emitCallAndSwitchStatement - This method sets up the caller side by adding
/// the call instruction, splitting any PHI nodes in the header block as
/// necessary.
void CodeExtractor::
emitCallAndSwitchStatement(Function *newFunction, BasicBlock *codeReplacer,
                           Values &inputs, Values &outputs) {
  // Emit a call to the new function, passing in: *pointer to struct (if
  // aggregating parameters), or plan inputs and allocated memory for outputs
  std::vector<Value*> params, StructValues, ReloadOutputs;

  // Add inputs as params, or to be filled into the struct
  for (Values::iterator i = inputs.begin(), e = inputs.end(); i != e; ++i)
    if (AggregateArgs)
      StructValues.push_back(*i);
    else
      params.push_back(*i);

  // Create allocas for the outputs
  for (Values::iterator i = outputs.begin(), e = outputs.end(); i != e; ++i) {
    if (AggregateArgs) {
      StructValues.push_back(*i);
    } else {
      AllocaInst *alloca =
        new AllocaInst((*i)->getType(), 0, (*i)->getName()+".loc",
                       codeReplacer->getParent()->begin()->begin());
      ReloadOutputs.push_back(alloca);
      params.push_back(alloca);
    }
  }

  AllocaInst *Struct = 0;
  if (AggregateArgs && (inputs.size() + outputs.size() > 0)) {
    std::vector<const Type*> ArgTypes;
    for (Values::iterator v = StructValues.begin(),
           ve = StructValues.end(); v != ve; ++v)
      ArgTypes.push_back((*v)->getType());

    // Allocate a struct at the beginning of this function
    Type *StructArgTy = StructType::get(ArgTypes);
    Struct = 
      new AllocaInst(StructArgTy, 0, "structArg", 
                     codeReplacer->getParent()->begin()->begin());
    params.push_back(Struct);

    for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
      std::vector<Value*> Indices;
      Indices.push_back(Constant::getNullValue(Type::UIntTy));
      Indices.push_back(ConstantUInt::get(Type::UIntTy, i));
      GetElementPtrInst *GEP =
        new GetElementPtrInst(Struct, Indices,
                              "gep_" + StructValues[i]->getName());
      codeReplacer->getInstList().push_back(GEP);
      StoreInst *SI = new StoreInst(StructValues[i], GEP);
      codeReplacer->getInstList().push_back(SI);
    }
  } 

  // Emit the call to the function
  CallInst *call = new CallInst(newFunction, params,
                                NumExitBlocks > 1 ? "targetBlock": "");
  codeReplacer->getInstList().push_back(call);

  Function::aiterator OutputArgBegin = newFunction->abegin();
  unsigned FirstOut = inputs.size();
  if (!AggregateArgs)
    std::advance(OutputArgBegin, inputs.size());

  // Reload the outputs passed in by reference
  for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
    Value *Output = 0;
    if (AggregateArgs) {
      std::vector<Value*> Indices;
      Indices.push_back(Constant::getNullValue(Type::UIntTy));
      Indices.push_back(ConstantUInt::get(Type::UIntTy, FirstOut + i));
      GetElementPtrInst *GEP 
        = new GetElementPtrInst(Struct, Indices,
                                "gep_reload_" + outputs[i]->getName());
      codeReplacer->getInstList().push_back(GEP);
      Output = GEP;
    } else {
      Output = ReloadOutputs[i];
    }
    LoadInst *load = new LoadInst(Output, outputs[i]->getName()+".reload");
    codeReplacer->getInstList().push_back(load);
    std::vector<User*> Users(outputs[i]->use_begin(), outputs[i]->use_end());
    for (unsigned u = 0, e = Users.size(); u != e; ++u) {
      Instruction *inst = cast<Instruction>(Users[u]);
      if (!BlocksToExtract.count(inst->getParent()))
        inst->replaceUsesOfWith(outputs[i], load);
    }
  }

  // Now we can emit a switch statement using the call as a value.
  SwitchInst *TheSwitch =
    new SwitchInst(ConstantUInt::getNullValue(Type::UShortTy),
                   codeReplacer, codeReplacer);

  // Since there may be multiple exits from the original region, make the new
  // function return an unsigned, switch on that number.  This loop iterates
  // over all of the blocks in the extracted region, updating any terminator
  // instructions in the to-be-extracted region that branch to blocks that are
  // not in the region to be extracted.
  std::map<BasicBlock*, BasicBlock*> ExitBlockMap;

  unsigned switchVal = 0;
  for (std::set<BasicBlock*>::const_iterator i = BlocksToExtract.begin(),
         e = BlocksToExtract.end(); i != e; ++i) {
    TerminatorInst *TI = (*i)->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      if (!BlocksToExtract.count(TI->getSuccessor(i))) {
        BasicBlock *OldTarget = TI->getSuccessor(i);
        // add a new basic block which returns the appropriate value
        BasicBlock *&NewTarget = ExitBlockMap[OldTarget];
        if (!NewTarget) {
          // If we don't already have an exit stub for this non-extracted
          // destination, create one now!
          NewTarget = new BasicBlock(OldTarget->getName() + ".exitStub",
                                     newFunction);
          unsigned SuccNum = switchVal++;

          Value *brVal = 0;
          switch (NumExitBlocks) {
          case 0:
          case 1: break;  // No value needed.
          case 2:         // Conditional branch, return a bool
            brVal = SuccNum ? ConstantBool::False : ConstantBool::True;
            break;
          default:
            brVal = ConstantUInt::get(Type::UShortTy, SuccNum);
            break;
          }

          ReturnInst *NTRet = new ReturnInst(brVal, NewTarget);

          // Update the switch instruction.
          TheSwitch->addCase(ConstantUInt::get(Type::UShortTy, SuccNum),
                             OldTarget);

          // Restore values just before we exit
          Function::aiterator OAI = OutputArgBegin;
          for (unsigned out = 0, e = outputs.size(); out != e; ++out) {
            // For an invoke, the normal destination is the only one that is
            // dominated by the result of the invocation
            BasicBlock *DefBlock = cast<Instruction>(outputs[out])->getParent();
            if (InvokeInst *Invoke = dyn_cast<InvokeInst>(outputs[out]))
              DefBlock = Invoke->getNormalDest();
            if (!DS || DS->dominates(DefBlock, TI->getParent()))
              if (AggregateArgs) {
                std::vector<Value*> Indices;
                Indices.push_back(Constant::getNullValue(Type::UIntTy));
                Indices.push_back(ConstantUInt::get(Type::UIntTy,FirstOut+out));
                GetElementPtrInst *GEP =
                  new GetElementPtrInst(OAI, Indices,
                                        "gep_" + outputs[out]->getName(), 
                                        NTRet);
                new StoreInst(outputs[out], GEP, NTRet);
              } else
                new StoreInst(outputs[out], OAI, NTRet);
            // Advance output iterator even if we don't emit a store
            if (!AggregateArgs) ++OAI;
          }
        }

        // rewrite the original branch instruction with this new target
        TI->setSuccessor(i, NewTarget);
      }
  }

  // Now that we've done the deed, simplify the switch instruction.
  const Type *OldFnRetTy = TheSwitch->getParent()->getParent()->getReturnType();
  switch (NumExitBlocks) {
  case 0:
    // There are no successors (the block containing the switch itself), which
    // means that previously this was the last part of the function, and hence
    // this should be rewritten as a `ret'
    
    // Check if the function should return a value
    if (OldFnRetTy == Type::VoidTy) {
      new ReturnInst(0, TheSwitch);  // Return void
    } else if (OldFnRetTy == TheSwitch->getCondition()->getType()) {
      // return what we have
      new ReturnInst(TheSwitch->getCondition(), TheSwitch);
    } else {
      // Otherwise we must have code extracted an unwind or something, just
      // return whatever we want.
      new ReturnInst(Constant::getNullValue(OldFnRetTy), TheSwitch);
    }

    TheSwitch->getParent()->getInstList().erase(TheSwitch);
    break;
  case 1:
    // Only a single destination, change the switch into an unconditional
    // branch.
    new BranchInst(TheSwitch->getSuccessor(1), TheSwitch);
    TheSwitch->getParent()->getInstList().erase(TheSwitch);
    break;
  case 2:
    new BranchInst(TheSwitch->getSuccessor(1), TheSwitch->getSuccessor(2),
                   call, TheSwitch);
    TheSwitch->getParent()->getInstList().erase(TheSwitch);
    break;
  default:
    // Otherwise, make the default destination of the switch instruction be one
    // of the other successors.
    TheSwitch->setOperand(0, call);
    TheSwitch->setSuccessor(0, TheSwitch->getSuccessor(NumExitBlocks));
    TheSwitch->removeCase(NumExitBlocks);  // Remove redundant case
    break;
  }
}

void CodeExtractor::moveCodeToFunction(Function *newFunction) {
  Function *oldFunc = (*BlocksToExtract.begin())->getParent();
  Function::BasicBlockListType &oldBlocks = oldFunc->getBasicBlockList();
  Function::BasicBlockListType &newBlocks = newFunction->getBasicBlockList();

  for (std::set<BasicBlock*>::const_iterator i = BlocksToExtract.begin(),
         e = BlocksToExtract.end(); i != e; ++i) {
    // Delete the basic block from the old function, and the list of blocks
    oldBlocks.remove(*i);

    // Insert this basic block into the new function
    newBlocks.push_back(*i);
  }
}

/// ExtractRegion - Removes a loop from a function, replaces it with a call to
/// new function. Returns pointer to the new function.
///
/// algorithm:
///
/// find inputs and outputs for the region
///
/// for inputs: add to function as args, map input instr* to arg# 
/// for outputs: add allocas for scalars, 
///             add to func as args, map output instr* to arg#
///
/// rewrite func to use argument #s instead of instr*
///
/// for each scalar output in the function: at every exit, store intermediate 
/// computed result back into memory.
///
Function *CodeExtractor::
ExtractCodeRegion(const std::vector<BasicBlock*> &code) {
  if (!isEligible(code))
    return 0;

  // 1) Find inputs, outputs
  // 2) Construct new function
  //  * Add allocas for defs, pass as args by reference
  //  * Pass in uses as args
  // 3) Move code region, add call instr to func
  //
  BlocksToExtract.insert(code.begin(), code.end());

  Values inputs, outputs;

  // Assumption: this is a single-entry code region, and the header is the first
  // block in the region.
  BasicBlock *header = code[0];

  for (unsigned i = 1, e = code.size(); i != e; ++i)
    for (pred_iterator PI = pred_begin(code[i]), E = pred_end(code[i]);
         PI != E; ++PI)
      assert(BlocksToExtract.count(*PI) &&
             "No blocks in this region may have entries from outside the region"
             " except for the first block!");
  
  // If we have to split PHI nodes or the entry block, do so now.
  severSplitPHINodes(header);

  // If we have any return instructions in the region, split those blocks so
  // that the return is not in the region.
  splitReturnBlocks();

  Function *oldFunction = header->getParent();

  // This takes place of the original loop
  BasicBlock *codeReplacer = new BasicBlock("codeRepl", oldFunction, header);

  // The new function needs a root node because other nodes can branch to the
  // head of the region, but the entry node of a function cannot have preds.
  BasicBlock *newFuncRoot = new BasicBlock("newFuncRoot");
  newFuncRoot->getInstList().push_back(new BranchInst(header));

  // Find inputs to, outputs from the code region.
  findInputsOutputs(inputs, outputs);

  // Construct new function based on inputs/outputs & add allocas for all defs.
  Function *newFunction = constructFunction(inputs, outputs, header,
                                            newFuncRoot, 
                                            codeReplacer, oldFunction,
                                            oldFunction->getParent());

  emitCallAndSwitchStatement(newFunction, codeReplacer, inputs, outputs);

  moveCodeToFunction(newFunction);

  // Loop over all of the PHI nodes in the header block, and change any
  // references to the old incoming edge to be the new incoming edge.
  for (BasicBlock::iterator I = header->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (!BlocksToExtract.count(PN->getIncomingBlock(i)))
        PN->setIncomingBlock(i, newFuncRoot);

  // Look at all successors of the codeReplacer block.  If any of these blocks
  // had PHI nodes in them, we need to update the "from" block to be the code
  // replacer, not the original block in the extracted region.
  std::vector<BasicBlock*> Succs(succ_begin(codeReplacer),
                                 succ_end(codeReplacer));
  for (unsigned i = 0, e = Succs.size(); i != e; ++i)
    for (BasicBlock::iterator I = Succs[i]->begin();
         PHINode *PN = dyn_cast<PHINode>(I); ++I) {
      std::set<BasicBlock*> ProcessedPreds;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (BlocksToExtract.count(PN->getIncomingBlock(i)))
          if (ProcessedPreds.insert(PN->getIncomingBlock(i)).second)
            PN->setIncomingBlock(i, codeReplacer);
          else {
            // There were multiple entries in the PHI for this block, now there
            // is only one, so remove the duplicated entries.
            PN->removeIncomingValue(i, false);
            --i; --e;
          }
    }
  
  //std::cerr << "NEW FUNCTION: " << *newFunction;
  //  verifyFunction(*newFunction);

  //  std::cerr << "OLD FUNCTION: " << *oldFunction;
  //  verifyFunction(*oldFunction);

  DEBUG(if (verifyFunction(*newFunction)) abort());
  return newFunction;
}

bool CodeExtractor::isEligible(const std::vector<BasicBlock*> &code) {
  // Deny code region if it contains allocas or vastarts.
  for (std::vector<BasicBlock*>::const_iterator BB = code.begin(), e=code.end();
       BB != e; ++BB)
    for (BasicBlock::const_iterator I = (*BB)->begin(), Ie = (*BB)->end();
         I != Ie; ++I)
      if (isa<AllocaInst>(*I))
        return false;
      else if (const CallInst *CI = dyn_cast<CallInst>(I))
        if (const Function *F = CI->getCalledFunction())
          if (F->getIntrinsicID() == Intrinsic::vastart)
            return false;
  return true;
}


/// ExtractCodeRegion - slurp a sequence of basic blocks into a brand new
/// function
///
Function* llvm::ExtractCodeRegion(DominatorSet &DS,
                                  const std::vector<BasicBlock*> &code,
                                  bool AggregateArgs) {
  return CodeExtractor(&DS, AggregateArgs).ExtractCodeRegion(code);
}

/// ExtractBasicBlock - slurp a natural loop into a brand new function
///
Function* llvm::ExtractLoop(DominatorSet &DS, Loop *L, bool AggregateArgs) {
  return CodeExtractor(&DS, AggregateArgs).ExtractCodeRegion(L->getBlocks());
}

/// ExtractBasicBlock - slurp a basic block into a brand new function
///
Function* llvm::ExtractBasicBlock(BasicBlock *BB, bool AggregateArgs) {
  std::vector<BasicBlock*> Blocks;
  Blocks.push_back(BB);
  return CodeExtractor(0, AggregateArgs).ExtractCodeRegion(Blocks);  
}
