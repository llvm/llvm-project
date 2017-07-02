//===- CodeExtractor.cpp - Pull code region into a new function -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface to tear out a code region, such as an
// individual loop or a parallel section, into a new function, replacing it with
// a call to the new function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <algorithm>
#include <set>
using namespace llvm;

#define DEBUG_TYPE "code-extractor"

// Provide a command-line option to aggregate function arguments into a struct
// for functions produced by the code extractor. This is useful when converting
// extracted functions to pthread-based code, as only one argument (void*) can
// be passed in to pthread_create().
static cl::opt<bool>
AggregateArgsOpt("aggregate-extracted-args", cl::Hidden,
                 cl::desc("Aggregate arguments to code-extracted functions"));

/// \brief Test whether a block is valid for extraction.
bool CodeExtractor::isBlockValidForExtraction(const BasicBlock &BB) {
  // Landing pads must be in the function where they were inserted for cleanup.
  if (BB.isEHPad())
    return false;
  // taking the address of a basic block moved to another function is illegal
  if (BB.hasAddressTaken())
    return false;

  // don't hoist code that uses another basicblock address, as it's likely to
  // lead to unexpected behavior, like cross-function jumps
  SmallPtrSet<User const *, 16> Visited;
  SmallVector<User const *, 16> ToVisit;

  for (Instruction const &Inst : BB)
    ToVisit.push_back(&Inst);

  while (!ToVisit.empty()) {
    User const *Curr = ToVisit.pop_back_val();
    if (!Visited.insert(Curr).second)
      continue;
    if (isa<BlockAddress const>(Curr))
      return false; // even a reference to self is likely to be not compatible

    if (isa<Instruction>(Curr) && cast<Instruction>(Curr)->getParent() != &BB)
      continue;

    for (auto const &U : Curr->operands()) {
      if (auto *UU = dyn_cast<User>(U))
        ToVisit.push_back(UU);
    }
  }

  // Don't hoist code containing allocas, invokes, or vastarts.
  for (BasicBlock::const_iterator I = BB.begin(), E = BB.end(); I != E; ++I) {
    if (isa<AllocaInst>(I) || isa<InvokeInst>(I))
      return false;
    if (const CallInst *CI = dyn_cast<CallInst>(I))
      if (const Function *F = CI->getCalledFunction())
        if (F->getIntrinsicID() == Intrinsic::vastart)
          return false;
  }

  return true;
}

/// \brief Build a set of blocks to extract if the input blocks are viable.
static SetVector<BasicBlock *>
buildExtractionBlockSet(ArrayRef<BasicBlock *> BBs, DominatorTree *DT) {
  assert(!BBs.empty() && "The set of blocks to extract must be non-empty");
  SetVector<BasicBlock *> Result;

  // Loop over the blocks, adding them to our set-vector, and aborting with an
  // empty set if we encounter invalid blocks.
  for (BasicBlock *BB : BBs) {

    // If this block is dead, don't process it.
    if (DT && !DT->isReachableFromEntry(BB))
      continue;

    if (!Result.insert(BB))
      llvm_unreachable("Repeated basic blocks in extraction input");
    if (!CodeExtractor::isBlockValidForExtraction(*BB)) {
      Result.clear();
      return Result;
    }
  }

#ifndef NDEBUG
  for (SetVector<BasicBlock *>::iterator I = std::next(Result.begin()),
                                         E = Result.end();
       I != E; ++I)
    for (pred_iterator PI = pred_begin(*I), PE = pred_end(*I);
         PI != PE; ++PI)
      assert(Result.count(*PI) &&
             "No blocks in this region may have entries from outside the region"
             " except for the first block!");
#endif

  return Result;
}

CodeExtractor::CodeExtractor(ArrayRef<BasicBlock *> BBs, DominatorTree *DT,
                             bool AggregateArgs, BlockFrequencyInfo *BFI,
                             BranchProbabilityInfo *BPI)
    : DT(DT), AggregateArgs(AggregateArgs || AggregateArgsOpt), BFI(BFI),
      BPI(BPI), Blocks(buildExtractionBlockSet(BBs, DT)), NumExitBlocks(~0U) {}

CodeExtractor::CodeExtractor(DominatorTree &DT, Loop &L, bool AggregateArgs,
                             BlockFrequencyInfo *BFI,
                             BranchProbabilityInfo *BPI)
    : DT(&DT), AggregateArgs(AggregateArgs || AggregateArgsOpt), BFI(BFI),
      BPI(BPI), Blocks(buildExtractionBlockSet(L.getBlocks(), &DT)),
      NumExitBlocks(~0U) {}

/// definedInRegion - Return true if the specified value is defined in the
/// extracted region.
static bool definedInRegion(const SetVector<BasicBlock *> &Blocks, Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (Blocks.count(I->getParent()))
      return true;
  return false;
}

/// definedInCaller - Return true if the specified value is defined in the
/// function being code extracted, but not in the region being extracted.
/// These values must be passed in as live-ins to the function.
static bool definedInCaller(const SetVector<BasicBlock *> &Blocks, Value *V) {
  if (isa<Argument>(V)) return true;
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (!Blocks.count(I->getParent()))
      return true;
  return false;
}

static BasicBlock *getCommonExitBlock(const SetVector<BasicBlock *> &Blocks) {
  BasicBlock *CommonExitBlock = nullptr;
  auto hasNonCommonExitSucc = [&](BasicBlock *Block) {
    for (auto *Succ : successors(Block)) {
      // Internal edges, ok.
      if (Blocks.count(Succ))
        continue;
      if (!CommonExitBlock) {
        CommonExitBlock = Succ;
        continue;
      }
      if (CommonExitBlock == Succ)
        continue;

      return true;
    }
    return false;
  };

  if (any_of(Blocks, hasNonCommonExitSucc))
    return nullptr;

  return CommonExitBlock;
}

bool CodeExtractor::isLegalToShrinkwrapLifetimeMarkers(
    Instruction *Addr) const {
  AllocaInst *AI = cast<AllocaInst>(Addr->stripInBoundsConstantOffsets());
  Function *Func = (*Blocks.begin())->getParent();
  for (BasicBlock &BB : *Func) {
    if (Blocks.count(&BB))
      continue;
    for (Instruction &II : BB) {

      if (isa<DbgInfoIntrinsic>(II))
        continue;

      unsigned Opcode = II.getOpcode();
      Value *MemAddr = nullptr;
      switch (Opcode) {
      case Instruction::Store:
      case Instruction::Load: {
        if (Opcode == Instruction::Store) {
          StoreInst *SI = cast<StoreInst>(&II);
          MemAddr = SI->getPointerOperand();
        } else {
          LoadInst *LI = cast<LoadInst>(&II);
          MemAddr = LI->getPointerOperand();
        }
        // Global variable can not be aliased with locals.
        if (dyn_cast<Constant>(MemAddr))
          break;
        Value *Base = MemAddr->stripInBoundsConstantOffsets();
        if (!dyn_cast<AllocaInst>(Base) || Base == AI)
          return false;
        break;
      }
      default: {
        IntrinsicInst *IntrInst = dyn_cast<IntrinsicInst>(&II);
        if (IntrInst) {
          if (IntrInst->getIntrinsicID() == Intrinsic::lifetime_start ||
              IntrInst->getIntrinsicID() == Intrinsic::lifetime_end)
            break;
          return false;
        }
        // Treat all the other cases conservatively if it has side effects.
        if (II.mayHaveSideEffects())
          return false;
      }
      }
    }
  }

  return true;
}

BasicBlock *
CodeExtractor::findOrCreateBlockForHoisting(BasicBlock *CommonExitBlock) {
  BasicBlock *SinglePredFromOutlineRegion = nullptr;
  assert(!Blocks.count(CommonExitBlock) &&
         "Expect a block outside the region!");
  for (auto *Pred : predecessors(CommonExitBlock)) {
    if (!Blocks.count(Pred))
      continue;
    if (!SinglePredFromOutlineRegion) {
      SinglePredFromOutlineRegion = Pred;
    } else if (SinglePredFromOutlineRegion != Pred) {
      SinglePredFromOutlineRegion = nullptr;
      break;
    }
  }

  if (SinglePredFromOutlineRegion)
    return SinglePredFromOutlineRegion;

#ifndef NDEBUG
  auto getFirstPHI = [](BasicBlock *BB) {
    BasicBlock::iterator I = BB->begin();
    PHINode *FirstPhi = nullptr;
    while (I != BB->end()) {
      PHINode *Phi = dyn_cast<PHINode>(I);
      if (!Phi)
        break;
      if (!FirstPhi) {
        FirstPhi = Phi;
        break;
      }
    }
    return FirstPhi;
  };
  // If there are any phi nodes, the single pred either exists or has already
  // be created before code extraction.
  assert(!getFirstPHI(CommonExitBlock) && "Phi not expected");
#endif

  BasicBlock *NewExitBlock = CommonExitBlock->splitBasicBlock(
      CommonExitBlock->getFirstNonPHI()->getIterator());

  for (auto *Pred : predecessors(CommonExitBlock)) {
    if (Blocks.count(Pred))
      continue;
    Pred->getTerminator()->replaceUsesOfWith(CommonExitBlock, NewExitBlock);
  }
  // Now add the old exit block to the outline region.
  Blocks.insert(CommonExitBlock);
  return CommonExitBlock;
}

void CodeExtractor::findAllocas(ValueSet &SinkCands, ValueSet &HoistCands,
                                BasicBlock *&ExitBlock) const {
  Function *Func = (*Blocks.begin())->getParent();
  ExitBlock = getCommonExitBlock(Blocks);

  for (BasicBlock &BB : *Func) {
    if (Blocks.count(&BB))
      continue;
    for (Instruction &II : BB) {
      auto *AI = dyn_cast<AllocaInst>(&II);
      if (!AI)
        continue;

      // Find the pair of life time markers for address 'Addr' that are either
      // defined inside the outline region or can legally be shrinkwrapped into
      // the outline region. If there are not other untracked uses of the
      // address, return the pair of markers if found; otherwise return a pair
      // of nullptr.
      auto GetLifeTimeMarkers =
          [&](Instruction *Addr, bool &SinkLifeStart,
              bool &HoistLifeEnd) -> std::pair<Instruction *, Instruction *> {
        Instruction *LifeStart = nullptr, *LifeEnd = nullptr;

        for (User *U : Addr->users()) {
          IntrinsicInst *IntrInst = dyn_cast<IntrinsicInst>(U);
          if (IntrInst) {
            if (IntrInst->getIntrinsicID() == Intrinsic::lifetime_start) {
              // Do not handle the case where AI has multiple start markers.
              if (LifeStart)
                return std::make_pair<Instruction *>(nullptr, nullptr);
              LifeStart = IntrInst;
            }
            if (IntrInst->getIntrinsicID() == Intrinsic::lifetime_end) {
              if (LifeEnd)
                return std::make_pair<Instruction *>(nullptr, nullptr);
              LifeEnd = IntrInst;
            }
            continue;
          }
          // Find untracked uses of the address, bail.
          if (!definedInRegion(Blocks, U))
            return std::make_pair<Instruction *>(nullptr, nullptr);
        }

        if (!LifeStart || !LifeEnd)
          return std::make_pair<Instruction *>(nullptr, nullptr);

        SinkLifeStart = !definedInRegion(Blocks, LifeStart);
        HoistLifeEnd = !definedInRegion(Blocks, LifeEnd);
        // Do legality Check.
        if ((SinkLifeStart || HoistLifeEnd) &&
            !isLegalToShrinkwrapLifetimeMarkers(Addr))
          return std::make_pair<Instruction *>(nullptr, nullptr);

        // Check to see if we have a place to do hoisting, if not, bail.
        if (HoistLifeEnd && !ExitBlock)
          return std::make_pair<Instruction *>(nullptr, nullptr);

        return std::make_pair(LifeStart, LifeEnd);
      };

      bool SinkLifeStart = false, HoistLifeEnd = false;
      auto Markers = GetLifeTimeMarkers(AI, SinkLifeStart, HoistLifeEnd);

      if (Markers.first) {
        if (SinkLifeStart)
          SinkCands.insert(Markers.first);
        SinkCands.insert(AI);
        if (HoistLifeEnd)
          HoistCands.insert(Markers.second);
        continue;
      }

      // Follow the bitcast.
      Instruction *MarkerAddr = nullptr;
      for (User *U : AI->users()) {

        if (U->stripInBoundsConstantOffsets() == AI) {
          SinkLifeStart = false;
          HoistLifeEnd = false;
          Instruction *Bitcast = cast<Instruction>(U);
          Markers = GetLifeTimeMarkers(Bitcast, SinkLifeStart, HoistLifeEnd);
          if (Markers.first) {
            MarkerAddr = Bitcast;
            continue;
          }
        }

        // Found unknown use of AI.
        if (!definedInRegion(Blocks, U)) {
          MarkerAddr = nullptr;
          break;
        }
      }

      if (MarkerAddr) {
        if (SinkLifeStart)
          SinkCands.insert(Markers.first);
        if (!definedInRegion(Blocks, MarkerAddr))
          SinkCands.insert(MarkerAddr);
        SinkCands.insert(AI);
        if (HoistLifeEnd)
          HoistCands.insert(Markers.second);
      }
    }
  }
}

void CodeExtractor::findInputsOutputs(ValueSet &Inputs, ValueSet &Outputs,
                                      const ValueSet &SinkCands) const {

  for (BasicBlock *BB : Blocks) {
    // If a used value is defined outside the region, it's an input.  If an
    // instruction is used outside the region, it's an output.
    for (Instruction &II : *BB) {
      for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
           ++OI) {
        Value *V = *OI;
        if (!SinkCands.count(V) && definedInCaller(Blocks, V))
          Inputs.insert(V);
      }

      for (User *U : II.users())
        if (!definedInRegion(Blocks, U)) {
          Outputs.insert(&II);
          break;
        }
    }
  }
}

/// severSplitPHINodes - If a PHI node has multiple inputs from outside of the
/// region, we need to split the entry block of the region so that the PHI node
/// is easier to deal with.
void CodeExtractor::severSplitPHINodes(BasicBlock *&Header) {
  unsigned NumPredsFromRegion = 0;
  unsigned NumPredsOutsideRegion = 0;

  if (Header != &Header->getParent()->getEntryBlock()) {
    PHINode *PN = dyn_cast<PHINode>(Header->begin());
    if (!PN) return;  // No PHI nodes.

    // If the header node contains any PHI nodes, check to see if there is more
    // than one entry from outside the region.  If so, we need to sever the
    // header block into two.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (Blocks.count(PN->getIncomingBlock(i)))
        ++NumPredsFromRegion;
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
  BasicBlock *NewBB = llvm::SplitBlock(Header, Header->getFirstNonPHI(), DT);

  // We only want to code extract the second block now, and it becomes the new
  // header of the region.
  BasicBlock *OldPred = Header;
  Blocks.remove(OldPred);
  Blocks.insert(NewBB);
  Header = NewBB;

  // Okay, now we need to adjust the PHI nodes and any branches from within the
  // region to go to the new header block instead of the old header block.
  if (NumPredsFromRegion) {
    PHINode *PN = cast<PHINode>(OldPred->begin());
    // Loop over all of the predecessors of OldPred that are in the region,
    // changing them to branch to NewBB instead.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (Blocks.count(PN->getIncomingBlock(i))) {
        TerminatorInst *TI = PN->getIncomingBlock(i)->getTerminator();
        TI->replaceUsesOfWith(OldPred, NewBB);
      }

    // Okay, everything within the region is now branching to the right block, we
    // just have to update the PHI nodes now, inserting PHI nodes into NewBB.
    BasicBlock::iterator AfterPHIs;
    for (AfterPHIs = OldPred->begin(); isa<PHINode>(AfterPHIs); ++AfterPHIs) {
      PHINode *PN = cast<PHINode>(AfterPHIs);
      // Create a new PHI node in the new region, which has an incoming value
      // from OldPred of PN.
      PHINode *NewPN = PHINode::Create(PN->getType(), 1 + NumPredsFromRegion,
                                       PN->getName() + ".ce", &NewBB->front());
      PN->replaceAllUsesWith(NewPN);
      NewPN->addIncoming(PN, OldPred);

      // Loop over all of the incoming value in PN, moving them to NewPN if they
      // are from the extracted region.
      for (unsigned i = 0; i != PN->getNumIncomingValues(); ++i) {
        if (Blocks.count(PN->getIncomingBlock(i))) {
          NewPN->addIncoming(PN->getIncomingValue(i), PN->getIncomingBlock(i));
          PN->removeIncomingValue(i);
          --i;
        }
      }
    }
  }
}

void CodeExtractor::splitReturnBlocks() {
  for (BasicBlock *Block : Blocks)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(Block->getTerminator())) {
      BasicBlock *New =
          Block->splitBasicBlock(RI->getIterator(), Block->getName() + ".ret");
      if (DT) {
        // Old dominates New. New node dominates all other nodes dominated
        // by Old.
        DomTreeNode *OldNode = DT->getNode(Block);
        SmallVector<DomTreeNode *, 8> Children(OldNode->begin(),
                                               OldNode->end());

        DomTreeNode *NewNode = DT->addNewBlock(New, Block);

        for (DomTreeNode *I : Children)
          DT->changeImmediateDominator(I, NewNode);
      }
    }
}

/// constructFunction - make a function based on inputs and outputs, as follows:
/// f(in0, ..., inN, out0, ..., outN)
///
Function *CodeExtractor::constructFunction(const ValueSet &inputs,
                                           const ValueSet &outputs,
                                           BasicBlock *header,
                                           BasicBlock *newRootNode,
                                           BasicBlock *newHeader,
                                           Function *oldFunction,
                                           Module *M) {
  DEBUG(dbgs() << "inputs: " << inputs.size() << "\n");
  DEBUG(dbgs() << "outputs: " << outputs.size() << "\n");

  // This function returns unsigned, outputs will go back by reference.
  switch (NumExitBlocks) {
  case 0:
  case 1: RetTy = Type::getVoidTy(header->getContext()); break;
  case 2: RetTy = Type::getInt1Ty(header->getContext()); break;
  default: RetTy = Type::getInt16Ty(header->getContext()); break;
  }

  std::vector<Type*> paramTy;

  // Add the types of the input values to the function's argument list
  for (Value *value : inputs) {
    DEBUG(dbgs() << "value used in func: " << *value << "\n");
    paramTy.push_back(value->getType());
  }

  // Add the types of the output values to the function's argument list.
  for (Value *output : outputs) {
    DEBUG(dbgs() << "instr used in func: " << *output << "\n");
    if (AggregateArgs)
      paramTy.push_back(output->getType());
    else
      paramTy.push_back(PointerType::getUnqual(output->getType()));
  }

  DEBUG({
    dbgs() << "Function type: " << *RetTy << " f(";
    for (Type *i : paramTy)
      dbgs() << *i << ", ";
    dbgs() << ")\n";
  });

  StructType *StructTy;
  if (AggregateArgs && (inputs.size() + outputs.size() > 0)) {
    StructTy = StructType::get(M->getContext(), paramTy);
    paramTy.clear();
    paramTy.push_back(PointerType::getUnqual(StructTy));
  }
  FunctionType *funcType =
                  FunctionType::get(RetTy, paramTy, false);

  // Create the new function
  Function *newFunction = Function::Create(funcType,
                                           GlobalValue::InternalLinkage,
                                           oldFunction->getName() + "_" +
                                           header->getName(), M);
  // If the old function is no-throw, so is the new one.
  if (oldFunction->doesNotThrow())
    newFunction->setDoesNotThrow();

  // Inherit the uwtable attribute if we need to.
  if (oldFunction->hasUWTable())
    newFunction->setHasUWTable();

  // Inherit all of the target dependent attributes.
  //  (e.g. If the extracted region contains a call to an x86.sse
  //  instruction we need to make sure that the extracted region has the
  //  "target-features" attribute allowing it to be lowered.
  // FIXME: This should be changed to check to see if a specific
  //           attribute can not be inherited.
  AttrBuilder AB(oldFunction->getAttributes().getFnAttributes());
  for (const auto &Attr : AB.td_attrs())
    newFunction->addFnAttr(Attr.first, Attr.second);

  newFunction->getBasicBlockList().push_back(newRootNode);

  // Create an iterator to name all of the arguments we inserted.
  Function::arg_iterator AI = newFunction->arg_begin();

  // Rewrite all users of the inputs in the extracted region to use the
  // arguments (or appropriate addressing into struct) instead.
  for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
    Value *RewriteVal;
    if (AggregateArgs) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(header->getContext()));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(header->getContext()), i);
      TerminatorInst *TI = newFunction->begin()->getTerminator();
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructTy, &*AI, Idx, "gep_" + inputs[i]->getName(), TI);
      RewriteVal = new LoadInst(GEP, "loadgep_" + inputs[i]->getName(), TI);
    } else
      RewriteVal = &*AI++;

    std::vector<User*> Users(inputs[i]->user_begin(), inputs[i]->user_end());
    for (User *use : Users)
      if (Instruction *inst = dyn_cast<Instruction>(use))
        if (Blocks.count(inst->getParent()))
          inst->replaceUsesOfWith(inputs[i], RewriteVal);
  }

  // Set names for input and output arguments.
  if (!AggregateArgs) {
    AI = newFunction->arg_begin();
    for (unsigned i = 0, e = inputs.size(); i != e; ++i, ++AI)
      AI->setName(inputs[i]->getName());
    for (unsigned i = 0, e = outputs.size(); i != e; ++i, ++AI)
      AI->setName(outputs[i]->getName()+".out");
  }

  // Rewrite branches to basic blocks outside of the loop to new dummy blocks
  // within the new function. This must be done before we lose track of which
  // blocks were originally in the code region.
  std::vector<User*> Users(header->user_begin(), header->user_end());
  for (unsigned i = 0, e = Users.size(); i != e; ++i)
    // The BasicBlock which contains the branch is not in the region
    // modify the branch target to a new block
    if (TerminatorInst *TI = dyn_cast<TerminatorInst>(Users[i]))
      if (!Blocks.count(TI->getParent()) &&
          TI->getParent()->getParent() == oldFunction)
        TI->replaceUsesOfWith(header, newHeader);

  return newFunction;
}

/// FindPhiPredForUseInBlock - Given a value and a basic block, find a PHI
/// that uses the value within the basic block, and return the predecessor
/// block associated with that use, or return 0 if none is found.
static BasicBlock* FindPhiPredForUseInBlock(Value* Used, BasicBlock* BB) {
  for (Use &U : Used->uses()) {
     PHINode *P = dyn_cast<PHINode>(U.getUser());
     if (P && P->getParent() == BB)
       return P->getIncomingBlock(U);
  }

  return nullptr;
}

/// emitCallAndSwitchStatement - This method sets up the caller side by adding
/// the call instruction, splitting any PHI nodes in the header block as
/// necessary.
void CodeExtractor::
emitCallAndSwitchStatement(Function *newFunction, BasicBlock *codeReplacer,
                           ValueSet &inputs, ValueSet &outputs) {
  // Emit a call to the new function, passing in: *pointer to struct (if
  // aggregating parameters), or plan inputs and allocated memory for outputs
  std::vector<Value*> params, StructValues, ReloadOutputs, Reloads;

  Module *M = newFunction->getParent();
  LLVMContext &Context = M->getContext();
  const DataLayout &DL = M->getDataLayout();

  // Add inputs as params, or to be filled into the struct
  for (Value *input : inputs)
    if (AggregateArgs)
      StructValues.push_back(input);
    else
      params.push_back(input);

  // Create allocas for the outputs
  for (Value *output : outputs) {
    if (AggregateArgs) {
      StructValues.push_back(output);
    } else {
      AllocaInst *alloca =
        new AllocaInst(output->getType(), DL.getAllocaAddrSpace(),
                       nullptr, output->getName() + ".loc",
                       &codeReplacer->getParent()->front().front());
      ReloadOutputs.push_back(alloca);
      params.push_back(alloca);
    }
  }

  StructType *StructArgTy = nullptr;
  AllocaInst *Struct = nullptr;
  if (AggregateArgs && (inputs.size() + outputs.size() > 0)) {
    std::vector<Type*> ArgTypes;
    for (ValueSet::iterator v = StructValues.begin(),
           ve = StructValues.end(); v != ve; ++v)
      ArgTypes.push_back((*v)->getType());

    // Allocate a struct at the beginning of this function
    StructArgTy = StructType::get(newFunction->getContext(), ArgTypes);
    Struct = new AllocaInst(StructArgTy, DL.getAllocaAddrSpace(), nullptr,
                            "structArg",
                            &codeReplacer->getParent()->front().front());
    params.push_back(Struct);

    for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), i);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, Struct, Idx, "gep_" + StructValues[i]->getName());
      codeReplacer->getInstList().push_back(GEP);
      StoreInst *SI = new StoreInst(StructValues[i], GEP);
      codeReplacer->getInstList().push_back(SI);
    }
  }

  // Emit the call to the function
  CallInst *call = CallInst::Create(newFunction, params,
                                    NumExitBlocks > 1 ? "targetBlock" : "");
  codeReplacer->getInstList().push_back(call);

  Function::arg_iterator OutputArgBegin = newFunction->arg_begin();
  unsigned FirstOut = inputs.size();
  if (!AggregateArgs)
    std::advance(OutputArgBegin, inputs.size());

  // Reload the outputs passed in by reference
  for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
    Value *Output = nullptr;
    if (AggregateArgs) {
      Value *Idx[2];
      Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
      Idx[1] = ConstantInt::get(Type::getInt32Ty(Context), FirstOut + i);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          StructArgTy, Struct, Idx, "gep_reload_" + outputs[i]->getName());
      codeReplacer->getInstList().push_back(GEP);
      Output = GEP;
    } else {
      Output = ReloadOutputs[i];
    }
    LoadInst *load = new LoadInst(Output, outputs[i]->getName()+".reload");
    Reloads.push_back(load);
    codeReplacer->getInstList().push_back(load);
    std::vector<User*> Users(outputs[i]->user_begin(), outputs[i]->user_end());
    for (unsigned u = 0, e = Users.size(); u != e; ++u) {
      Instruction *inst = cast<Instruction>(Users[u]);
      if (!Blocks.count(inst->getParent()))
        inst->replaceUsesOfWith(outputs[i], load);
    }
  }

  // Now we can emit a switch statement using the call as a value.
  SwitchInst *TheSwitch =
      SwitchInst::Create(Constant::getNullValue(Type::getInt16Ty(Context)),
                         codeReplacer, 0, codeReplacer);

  // Since there may be multiple exits from the original region, make the new
  // function return an unsigned, switch on that number.  This loop iterates
  // over all of the blocks in the extracted region, updating any terminator
  // instructions in the to-be-extracted region that branch to blocks that are
  // not in the region to be extracted.
  std::map<BasicBlock*, BasicBlock*> ExitBlockMap;

  unsigned switchVal = 0;
  for (BasicBlock *Block : Blocks) {
    TerminatorInst *TI = Block->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      if (!Blocks.count(TI->getSuccessor(i))) {
        BasicBlock *OldTarget = TI->getSuccessor(i);
        // add a new basic block which returns the appropriate value
        BasicBlock *&NewTarget = ExitBlockMap[OldTarget];
        if (!NewTarget) {
          // If we don't already have an exit stub for this non-extracted
          // destination, create one now!
          NewTarget = BasicBlock::Create(Context,
                                         OldTarget->getName() + ".exitStub",
                                         newFunction);
          unsigned SuccNum = switchVal++;

          Value *brVal = nullptr;
          switch (NumExitBlocks) {
          case 0:
          case 1: break;  // No value needed.
          case 2:         // Conditional branch, return a bool
            brVal = ConstantInt::get(Type::getInt1Ty(Context), !SuccNum);
            break;
          default:
            brVal = ConstantInt::get(Type::getInt16Ty(Context), SuccNum);
            break;
          }

          ReturnInst *NTRet = ReturnInst::Create(Context, brVal, NewTarget);

          // Update the switch instruction.
          TheSwitch->addCase(ConstantInt::get(Type::getInt16Ty(Context),
                                              SuccNum),
                             OldTarget);

          // Restore values just before we exit
          Function::arg_iterator OAI = OutputArgBegin;
          for (unsigned out = 0, e = outputs.size(); out != e; ++out) {
            // For an invoke, the normal destination is the only one that is
            // dominated by the result of the invocation
            BasicBlock *DefBlock = cast<Instruction>(outputs[out])->getParent();

            bool DominatesDef = true;

            BasicBlock *NormalDest = nullptr;
            if (auto *Invoke = dyn_cast<InvokeInst>(outputs[out]))
              NormalDest = Invoke->getNormalDest();

            if (NormalDest) {
              DefBlock = NormalDest;

              // Make sure we are looking at the original successor block, not
              // at a newly inserted exit block, which won't be in the dominator
              // info.
              for (const auto &I : ExitBlockMap)
                if (DefBlock == I.second) {
                  DefBlock = I.first;
                  break;
                }

              // In the extract block case, if the block we are extracting ends
              // with an invoke instruction, make sure that we don't emit a
              // store of the invoke value for the unwind block.
              if (!DT && DefBlock != OldTarget)
                DominatesDef = false;
            }

            if (DT) {
              DominatesDef = DT->dominates(DefBlock, OldTarget);
              
              // If the output value is used by a phi in the target block,
              // then we need to test for dominance of the phi's predecessor
              // instead.  Unfortunately, this a little complicated since we
              // have already rewritten uses of the value to uses of the reload.
              BasicBlock* pred = FindPhiPredForUseInBlock(Reloads[out], 
                                                          OldTarget);
              if (pred && DT && DT->dominates(DefBlock, pred))
                DominatesDef = true;
            }

            if (DominatesDef) {
              if (AggregateArgs) {
                Value *Idx[2];
                Idx[0] = Constant::getNullValue(Type::getInt32Ty(Context));
                Idx[1] = ConstantInt::get(Type::getInt32Ty(Context),
                                          FirstOut+out);
                GetElementPtrInst *GEP = GetElementPtrInst::Create(
                    StructArgTy, &*OAI, Idx, "gep_" + outputs[out]->getName(),
                    NTRet);
                new StoreInst(outputs[out], GEP, NTRet);
              } else {
                new StoreInst(outputs[out], &*OAI, NTRet);
              }
            }
            // Advance output iterator even if we don't emit a store
            if (!AggregateArgs) ++OAI;
          }
        }

        // rewrite the original branch instruction with this new target
        TI->setSuccessor(i, NewTarget);
      }
  }

  // Now that we've done the deed, simplify the switch instruction.
  Type *OldFnRetTy = TheSwitch->getParent()->getParent()->getReturnType();
  switch (NumExitBlocks) {
  case 0:
    // There are no successors (the block containing the switch itself), which
    // means that previously this was the last part of the function, and hence
    // this should be rewritten as a `ret'

    // Check if the function should return a value
    if (OldFnRetTy->isVoidTy()) {
      ReturnInst::Create(Context, nullptr, TheSwitch);  // Return void
    } else if (OldFnRetTy == TheSwitch->getCondition()->getType()) {
      // return what we have
      ReturnInst::Create(Context, TheSwitch->getCondition(), TheSwitch);
    } else {
      // Otherwise we must have code extracted an unwind or something, just
      // return whatever we want.
      ReturnInst::Create(Context, 
                         Constant::getNullValue(OldFnRetTy), TheSwitch);
    }

    TheSwitch->eraseFromParent();
    break;
  case 1:
    // Only a single destination, change the switch into an unconditional
    // branch.
    BranchInst::Create(TheSwitch->getSuccessor(1), TheSwitch);
    TheSwitch->eraseFromParent();
    break;
  case 2:
    BranchInst::Create(TheSwitch->getSuccessor(1), TheSwitch->getSuccessor(2),
                       call, TheSwitch);
    TheSwitch->eraseFromParent();
    break;
  default:
    // Otherwise, make the default destination of the switch instruction be one
    // of the other successors.
    TheSwitch->setCondition(call);
    TheSwitch->setDefaultDest(TheSwitch->getSuccessor(NumExitBlocks));
    // Remove redundant case
    TheSwitch->removeCase(SwitchInst::CaseIt(TheSwitch, NumExitBlocks-1));
    break;
  }
}

void CodeExtractor::moveCodeToFunction(Function *newFunction) {
  Function *oldFunc = (*Blocks.begin())->getParent();
  Function::BasicBlockListType &oldBlocks = oldFunc->getBasicBlockList();
  Function::BasicBlockListType &newBlocks = newFunction->getBasicBlockList();

  for (BasicBlock *Block : Blocks) {
    // Delete the basic block from the old function, and the list of blocks
    oldBlocks.remove(Block);

    // Insert this basic block into the new function
    newBlocks.push_back(Block);
  }
}

void CodeExtractor::calculateNewCallTerminatorWeights(
    BasicBlock *CodeReplacer,
    DenseMap<BasicBlock *, BlockFrequency> &ExitWeights,
    BranchProbabilityInfo *BPI) {
  typedef BlockFrequencyInfoImplBase::Distribution Distribution;
  typedef BlockFrequencyInfoImplBase::BlockNode BlockNode;

  // Update the branch weights for the exit block.
  TerminatorInst *TI = CodeReplacer->getTerminator();
  SmallVector<unsigned, 8> BranchWeights(TI->getNumSuccessors(), 0);

  // Block Frequency distribution with dummy node.
  Distribution BranchDist;

  // Add each of the frequencies of the successors.
  for (unsigned i = 0, e = TI->getNumSuccessors(); i < e; ++i) {
    BlockNode ExitNode(i);
    uint64_t ExitFreq = ExitWeights[TI->getSuccessor(i)].getFrequency();
    if (ExitFreq != 0)
      BranchDist.addExit(ExitNode, ExitFreq);
    else
      BPI->setEdgeProbability(CodeReplacer, i, BranchProbability::getZero());
  }

  // Check for no total weight.
  if (BranchDist.Total == 0)
    return;

  // Normalize the distribution so that they can fit in unsigned.
  BranchDist.normalize();

  // Create normalized branch weights and set the metadata.
  for (unsigned I = 0, E = BranchDist.Weights.size(); I < E; ++I) {
    const auto &Weight = BranchDist.Weights[I];

    // Get the weight and update the current BFI.
    BranchWeights[Weight.TargetNode.Index] = Weight.Amount;
    BranchProbability BP(Weight.Amount, BranchDist.Total);
    BPI->setEdgeProbability(CodeReplacer, Weight.TargetNode.Index, BP);
  }
  TI->setMetadata(
      LLVMContext::MD_prof,
      MDBuilder(TI->getContext()).createBranchWeights(BranchWeights));
}

Function *CodeExtractor::extractCodeRegion() {
  if (!isEligible())
    return nullptr;

  ValueSet inputs, outputs, SinkingCands, HoistingCands;
  BasicBlock *CommonExit = nullptr;

  // Assumption: this is a single-entry code region, and the header is the first
  // block in the region.
  BasicBlock *header = *Blocks.begin();

  // Calculate the entry frequency of the new function before we change the root
  //   block.
  BlockFrequency EntryFreq;
  if (BFI) {
    assert(BPI && "Both BPI and BFI are required to preserve profile info");
    for (BasicBlock *Pred : predecessors(header)) {
      if (Blocks.count(Pred))
        continue;
      EntryFreq +=
          BFI->getBlockFreq(Pred) * BPI->getEdgeProbability(Pred, header);
    }
  }

  // If we have to split PHI nodes or the entry block, do so now.
  severSplitPHINodes(header);

  // If we have any return instructions in the region, split those blocks so
  // that the return is not in the region.
  splitReturnBlocks();

  Function *oldFunction = header->getParent();

  // This takes place of the original loop
  BasicBlock *codeReplacer = BasicBlock::Create(header->getContext(), 
                                                "codeRepl", oldFunction,
                                                header);

  // The new function needs a root node because other nodes can branch to the
  // head of the region, but the entry node of a function cannot have preds.
  BasicBlock *newFuncRoot = BasicBlock::Create(header->getContext(), 
                                               "newFuncRoot");
  newFuncRoot->getInstList().push_back(BranchInst::Create(header));

  findAllocas(SinkingCands, HoistingCands, CommonExit);
  assert(HoistingCands.empty() || CommonExit);

  // Find inputs to, outputs from the code region.
  findInputsOutputs(inputs, outputs, SinkingCands);

  // Now sink all instructions which only have non-phi uses inside the region
  for (auto *II : SinkingCands)
    cast<Instruction>(II)->moveBefore(*newFuncRoot,
                                      newFuncRoot->getFirstInsertionPt());

  if (!HoistingCands.empty()) {
    auto *HoistToBlock = findOrCreateBlockForHoisting(CommonExit);
    Instruction *TI = HoistToBlock->getTerminator();
    for (auto *II : HoistingCands)
      cast<Instruction>(II)->moveBefore(TI);
  }

  // Calculate the exit blocks for the extracted region and the total exit
  //  weights for each of those blocks.
  DenseMap<BasicBlock *, BlockFrequency> ExitWeights;
  SmallPtrSet<BasicBlock *, 1> ExitBlocks;
  for (BasicBlock *Block : Blocks) {
    for (succ_iterator SI = succ_begin(Block), SE = succ_end(Block); SI != SE;
         ++SI) {
      if (!Blocks.count(*SI)) {
        // Update the branch weight for this successor.
        if (BFI) {
          BlockFrequency &BF = ExitWeights[*SI];
          BF += BFI->getBlockFreq(Block) * BPI->getEdgeProbability(Block, *SI);
        }
        ExitBlocks.insert(*SI);
      }
    }
  }
  NumExitBlocks = ExitBlocks.size();

  // Construct new function based on inputs/outputs & add allocas for all defs.
  Function *newFunction = constructFunction(inputs, outputs, header,
                                            newFuncRoot,
                                            codeReplacer, oldFunction,
                                            oldFunction->getParent());

  // Update the entry count of the function.
  if (BFI) {
    Optional<uint64_t> EntryCount =
        BFI->getProfileCountFromFreq(EntryFreq.getFrequency());
    if (EntryCount.hasValue())
      newFunction->setEntryCount(EntryCount.getValue());
    BFI->setBlockFreq(codeReplacer, EntryFreq.getFrequency());
  }

  emitCallAndSwitchStatement(newFunction, codeReplacer, inputs, outputs);

  moveCodeToFunction(newFunction);

  // Update the branch weights for the exit block.
  if (BFI && NumExitBlocks > 1)
    calculateNewCallTerminatorWeights(codeReplacer, ExitWeights, BPI);

  // Loop over all of the PHI nodes in the header block, and change any
  // references to the old incoming edge to be the new incoming edge.
  for (BasicBlock::iterator I = header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (!Blocks.count(PN->getIncomingBlock(i)))
        PN->setIncomingBlock(i, newFuncRoot);
  }

  // Look at all successors of the codeReplacer block.  If any of these blocks
  // had PHI nodes in them, we need to update the "from" block to be the code
  // replacer, not the original block in the extracted region.
  std::vector<BasicBlock*> Succs(succ_begin(codeReplacer),
                                 succ_end(codeReplacer));
  for (unsigned i = 0, e = Succs.size(); i != e; ++i)
    for (BasicBlock::iterator I = Succs[i]->begin(); isa<PHINode>(I); ++I) {
      PHINode *PN = cast<PHINode>(I);
      std::set<BasicBlock*> ProcessedPreds;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (Blocks.count(PN->getIncomingBlock(i))) {
          if (ProcessedPreds.insert(PN->getIncomingBlock(i)).second)
            PN->setIncomingBlock(i, codeReplacer);
          else {
            // There were multiple entries in the PHI for this block, now there
            // is only one, so remove the duplicated entries.
            PN->removeIncomingValue(i, false);
            --i; --e;
          }
        }
    }

  DEBUG(if (verifyFunction(*newFunction)) 
        report_fatal_error("verifyFunction failed!"));
  return newFunction;
}
