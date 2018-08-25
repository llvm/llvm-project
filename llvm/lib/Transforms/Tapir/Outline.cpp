//===- TapirOutline.cpp - Outlining for Tapir -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements helper functions for outlining portions of code
// containing Tapir instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "outlining"

/// definedInRegion - Return true if the specified value is defined in the
/// extracted region.
static bool definedInRegion(const SmallPtrSetImpl<BasicBlock *> &Blocks,
                            Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (Blocks.count(I->getParent()))
      return true;
  return false;
}

/// definedInCaller - Return true if the specified value is defined in the
/// function being code extracted, but not in the region being extracted.
/// These values must be passed in as live-ins to the function.
static bool definedInCaller(const SmallPtrSetImpl<BasicBlock *> &Blocks,
                            Value *V) {
  if (isa<Argument>(V)) return true;
  if (Instruction *I = dyn_cast<Instruction>(V))
    if (!Blocks.count(I->getParent()))
      return true;
  return false;
}

// findInputsOutputs - Find inputs and outputs for Blocks.  Any blocks in
// ExitBlocks are handled in a special manner: PHI nodes in Exit Blocks are
// ignored when determining inputs.
void llvm::findInputsOutputs(const SmallPtrSetImpl<BasicBlock *> &Blocks,
                             ValueSet &Inputs, ValueSet &Outputs,
                             const SmallPtrSetImpl<BasicBlock *> *ExitBlocks,
                             DominatorTree *DT) {
  for (BasicBlock *BB : Blocks) {
    // If a used value is defined outside the region, it's an input.  If an
    // instruction is used outside the region, it's an output.
    for (Instruction &II : *BB) {
      for (User::op_iterator OI = II.op_begin(), OE = II.op_end(); OI != OE;
           ++OI) {
        // The PHI nodes in each exit block will be updated after the exit block
        // is cloned.  Hence, we don't want to count their uses of values
        // defined outside the region.
        if (ExitBlocks && ExitBlocks->count(BB))
          if (PHINode *PN = dyn_cast<PHINode>(&II))
            if (!Blocks.count(PN->getIncomingBlock(*OI)))
              continue;
        if (definedInCaller(Blocks, *OI))
          Inputs.insert(*OI);
      }

      // Ignore outputs from exit blocks.
      if (!ExitBlocks || !ExitBlocks->count(BB)) {
        for (User *U : II.users()) {
          if (!definedInRegion(Blocks, U)) {
            // It looks like we have a use outside of the given blocks, but it's
            // possible for the use to appear in a basic block that is no longer
            // alive.  We use the DT to check that this use is still alive.
            if (Instruction *I = dyn_cast<Instruction>(U)) {
              if (DT && DT->isReachableFromEntry(I->getParent())) {
                Outputs.insert(&II);
                break;
              }
            }
          }
        }
      }
    }
  }
}

// Clone Blocks into NewFunc, transforming the old arguments into references to
// VMap values.
//
/// TODO: Fix the std::vector part of the type of this function.
void llvm::CloneIntoFunction(
    Function *NewFunc, const Function *OldFunc,
    std::vector<BasicBlock *> Blocks, ValueToValueMapTy &VMap,
    bool ModuleLevelChanges, SmallVectorImpl<ReturnInst *> &Returns,
    const StringRef NameSuffix, SmallPtrSetImpl<BasicBlock *> *ReattachBlocks,
    SmallPtrSetImpl<BasicBlock *> *DetachedRethrowBlocks,
    SmallPtrSetImpl<BasicBlock *> *SharedEHEntries,
    DISubprogram *SP, ClonedCodeInfo *CodeInfo,
    ValueMapTypeRemapper *TypeMapper, ValueMaterializer *Materializer) {
  // Get the predecessors of the exit blocks
  SmallPtrSet<const BasicBlock *, 4> EHEntryPreds, ClonedEHEntryPreds;
  if (SharedEHEntries)
    for (BasicBlock *EHEntry : *SharedEHEntries)
      for (BasicBlock *Pred : predecessors(EHEntry))
        EHEntryPreds.insert(Pred);

  // When we remap instructions, we want to avoid duplicating inlined
  // DISubprograms, so record all subprograms we find as we duplicate
  // instructions and then freeze them in the MD map.
  DebugInfoFinder DIFinder;

  // Loop over all of the basic blocks in the function, cloning them as
  // appropriate.
  for (const BasicBlock *BB : Blocks) {
    // Record all exit block predecessors that are cloned.
    if (EHEntryPreds.count(BB))
      ClonedEHEntryPreds.insert(BB);

    // Create a new basic block and copy instructions into it!
    BasicBlock *CBB = CloneBasicBlock(BB, VMap, NameSuffix, NewFunc, CodeInfo,
                                      SP ? &DIFinder : nullptr);

    // Add basic block mapping.
    VMap[BB] = CBB;

    // It is only legal to clone a function if a block address within that
    // function is never referenced outside of the function.  Given that, we
    // want to map block addresses from the old function to block addresses in
    // the clone. (This is different from the generic ValueMapper
    // implementation, which generates an invalid blockaddress when cloning a
    // function.)
    if (BB->hasAddressTaken()) {
      Constant *OldBBAddr = BlockAddress::get(const_cast<Function *>(OldFunc),
                                              const_cast<BasicBlock *>(BB));
      VMap[OldBBAddr] = BlockAddress::get(NewFunc, CBB);
    }

    // Note return instructions for the caller.
    if (ReturnInst *RI = dyn_cast<ReturnInst>(CBB->getTerminator()))
      Returns.push_back(RI);
  }

  // For each exit block, clean up its phi nodes to exclude predecessors that
  // were not cloned.  Also replace detached_rethrow invokes with resumes.
  if (SharedEHEntries) {
    for (BasicBlock *EHEntry : *SharedEHEntries) {
      // Get the predecessors of this exit block that were not cloned.
      SmallVector<BasicBlock *, 4> PredNotCloned;
      for (BasicBlock *Pred : predecessors(EHEntry))
        if (!ClonedEHEntryPreds.count(Pred))
          PredNotCloned.push_back(Pred);

      // Iterate over the phi nodes in the cloned exit block and remove incoming
      // values from predecessors that were not cloned.
      BasicBlock *ClonedEHEntry = cast<BasicBlock>(VMap[EHEntry]);
      BasicBlock::iterator BI = ClonedEHEntry->begin();
      while (PHINode *PN = dyn_cast<PHINode>(BI)) {
        for (BasicBlock *DeadPred : PredNotCloned)
          if (PN->getBasicBlockIndex(DeadPred) > -1)
            PN->removeIncomingValue(DeadPred);
        ++BI;
      }
    }
  }
  if (ReattachBlocks) {
    for (BasicBlock *ReattachBlk : *ReattachBlocks) {
      BasicBlock *ClonedRB = cast<BasicBlock>(VMap[ReattachBlk]);
      // Don't get the remapped name of this successor yet.  Subsequent
      // remapping will take correct the name.
      BasicBlock *Succ = ClonedRB->getSingleSuccessor();
      ReplaceInstWithInst(ClonedRB->getTerminator(),
                          BranchInst::Create(Succ));
    }
  }
  if (DetachedRethrowBlocks) {
    for (BasicBlock *DetRethrowBlk : *DetachedRethrowBlocks) {
      // Skip blocks that are not terminated by a detached-rethrow.
      if (!isDetachedRethrow(DetRethrowBlk->getTerminator()))
        continue;

      BasicBlock *ClonedDRB = cast<BasicBlock>(VMap[DetRethrowBlk]);
      // If this exit block terminates in a detached_rethrow, replace the
      // terminator with a resume.
      InvokeInst *II = cast<InvokeInst>(ClonedDRB->getTerminator());
      Value *RethrowArg = II->getArgOperand(1);
      ReplaceInstWithInst(ClonedDRB->getTerminator(),
                          ResumeInst::Create(RethrowArg));
      // if (isDetachedRethrow(EB->getTerminator())) {
      //   InvokeInst *II = cast<InvokeInst>(ClonedEB->getTerminator());
      //   Value *RethrowArg = II->getArgOperand(1);
      //   ReplaceInstWithInst(ClonedEB->getTerminator(),
      //                       ResumeInst::Create(RethrowArg));
      // }
    }
  }

  // for (DISubprogram *ISP : DIFinder.subprograms()) {
  //   if (ISP != SP) {
  //     VMap.MD()[ISP].reset(ISP);
  //   }
  // }

  // Loop over all of the instructions in the function, fixing up operand
  // references as we go.  This uses VMap to do all the hard work.
  for (const BasicBlock *BB : Blocks) {
    BasicBlock *CBB = cast<BasicBlock>(VMap[BB]);
    // Loop over all instructions, fixing each one as we find it...
    for (Instruction &II : *CBB)
      RemapInstruction(&II, VMap,
                       ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                       TypeMapper, Materializer);
  }
}

/// Create a helper function whose signature is based on Inputs and
/// Outputs as follows: f(in0, ..., inN, out0, ..., outN)
///
/// TODO: Fix the std::vector part of the type of this function.
Function *llvm::CreateHelper(
    const ValueSet &Inputs, const ValueSet &Outputs,
    std::vector<BasicBlock *> Blocks, BasicBlock *Header,
    const BasicBlock *OldEntry, const BasicBlock *OldExit,
    ValueToValueMapTy &VMap, Module *DestM, bool ModuleLevelChanges,
    SmallVectorImpl<ReturnInst *> &Returns, const StringRef NameSuffix,
    SmallPtrSetImpl<BasicBlock *> *ReattachBlocks,
    SmallPtrSetImpl<BasicBlock *> *DetachRethrowBlocks,
    SmallPtrSetImpl<BasicBlock *> *SharedEHEntries,
    const BasicBlock *OldUnwind,
    const Instruction *InputSyncRegion, ClonedCodeInfo *CodeInfo,
    ValueMapTypeRemapper *TypeMapper, ValueMaterializer *Materializer) {
  DEBUG(dbgs() << "inputs: " << Inputs.size() << "\n");
  DEBUG(dbgs() << "outputs: " << Outputs.size() << "\n");

  Function *OldFunc = Header->getParent();
  Type *RetTy = Type::getVoidTy(Header->getContext());

  std::vector<Type *> paramTy;

  // Add the types of the input values to the function's argument list
  for (Value *value : Inputs) {
    DEBUG(dbgs() << "value used in func: " << *value << "\n");
    paramTy.push_back(value->getType());
  }

  // Add the types of the output values to the function's argument list.
  for (Value *output : Outputs) {
    DEBUG(dbgs() << "instr used in func: " << *output << "\n");
    paramTy.push_back(PointerType::getUnqual(output->getType()));
  }

  DEBUG({
      dbgs() << "Function type: " << *RetTy << " f(";
      for (Type *i : paramTy)
	dbgs() << *i << ", ";
      dbgs() << ")\n";
    });

  FunctionType *FTy = FunctionType::get(RetTy, paramTy, false);

  // Create the new function
  Function *NewFunc = Function::Create(
      FTy, GlobalValue::InternalLinkage,
      OldFunc->getName() + "_" + Header->getName() + NameSuffix, DestM);

  // Set names for input and output arguments.
  Function::arg_iterator DestI = NewFunc->arg_begin();
  for (Value *I : Inputs)
    if (VMap.count(I) == 0) {       // Is this argument preserved?
      DestI->setName(I->getName()+NameSuffix); // Copy the name over...
      VMap[I] = &*DestI++;          // Add mapping to VMap
    }
  for (Value *I : Outputs)
    if (VMap.count(I) == 0) {              // Is this argument preserved?
      DestI->setName(I->getName()+NameSuffix); // Copy the name over...
      VMap[I] = &*DestI++;                 // Add mapping to VMap
    }

  // Copy all attributes other than those stored in the AttributeSet.  We need
  // to remap the parameter indices of the AttributeSet.
  AttributeList NewAttrs = NewFunc->getAttributes();
  NewFunc->copyAttributesFrom(OldFunc);
  NewFunc->setAttributes(NewAttrs);

  // Fix up the personality function that got copied over.
  if (OldFunc->hasPersonalityFn())
    NewFunc->setPersonalityFn(
        MapValue(OldFunc->getPersonalityFn(), VMap,
                 ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                 TypeMapper, Materializer));

  SmallVector<AttributeSet, 4> NewArgAttrs(NewFunc->arg_size());
  AttributeList OldAttrs = OldFunc->getAttributes();

  // Clone any argument attributes
  for (Argument &OldArg : OldFunc->args()) {
    // Check if we're passing this argument to the helper.  We check Inputs here
    // instead of the VMap to avoid potentially populating the VMap with a null
    // entry for the old argument.
    if (Inputs.count(&OldArg) || Outputs.count(&OldArg)) {
      Argument *NewArg = dyn_cast<Argument>(VMap[&OldArg]);
      NewArgAttrs[NewArg->getArgNo()] =
          OldAttrs.getParamAttributes(OldArg.getArgNo())
          .removeAttribute(NewFunc->getContext(), Attribute::Returned);
    }
  }

  NewFunc->setAttributes(
      AttributeList::get(NewFunc->getContext(), OldAttrs.getFnAttributes(),
                         OldAttrs.getRetAttributes(), NewArgAttrs));

  // Remove old return attributes.
  NewFunc->removeAttributes(
      AttributeList::ReturnIndex,
      AttributeFuncs::typeIncompatible(NewFunc->getReturnType()));

  // Clone the metadata from the old function into the new.
  bool MustCloneSP =
      OldFunc->getParent() && OldFunc->getParent() == NewFunc->getParent();
  DISubprogram *SP = OldFunc->getSubprogram();
  if (SP) {
    assert(!MustCloneSP || ModuleLevelChanges);
    // Add mappings for some DebugInfo nodes that we don't want duplicated
    // even if they're distinct.
    auto &MD = VMap.MD();
    MD[SP->getUnit()].reset(SP->getUnit());
    MD[SP->getType()].reset(SP->getType());
    MD[SP->getFile()].reset(SP->getFile());
    // If we're not cloning into the same module, no need to clone the
    // subprogram
    if (!MustCloneSP)
      MD[SP].reset(SP);
  }

  SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
  OldFunc->getAllMetadata(MDs);
  for (auto MD : MDs) {
    NewFunc->addMetadata(
        MD.first,
        *MapMetadata(MD.second, VMap,
                     ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                     TypeMapper, Materializer));
  }

  // We assume that the Helper reads and writes its arguments.  If the parent
  // function had stronger attributes on memory access -- specifically, if the
  // parent is marked as only reading memory -- we must replace this attribute
  // with an appropriate weaker form.
  if (OldFunc->onlyReadsMemory()) {
    NewFunc->removeFnAttr(Attribute::ReadNone);
    NewFunc->removeFnAttr(Attribute::ReadOnly);
    NewFunc->setOnlyAccessesArgMemory();
  }

  // Inherit the calling convention from the parent.
  NewFunc->setCallingConv(OldFunc->getCallingConv());

  // The new function needs a root node because other nodes can branch to the
  // head of the region, but the entry node of a function cannot have preds.
  BasicBlock *NewEntry = BasicBlock::Create(
      Header->getContext(), OldEntry->getName()+NameSuffix, NewFunc);
  // The new function also needs an exit node.
  BasicBlock *NewExit = BasicBlock::Create(
      Header->getContext(), OldExit->getName()+NameSuffix, NewFunc);

  // Add mappings to the NewEntry and NewExit.
  VMap[OldEntry] = NewEntry;
  VMap[OldExit] = NewExit;

  BasicBlock *NewUnwind = nullptr;
  // Create a new unwind destination for the cloned blocks if it's needed.
  if (OldUnwind) {
    NewUnwind = BasicBlock::Create(
        NewFunc->getContext(), OldUnwind->getName()+NameSuffix, NewFunc);
    VMap[OldUnwind] = NewUnwind;
  }

  // Create new sync region to replace the old one containing any cloned Tapir
  // instructions, and add the appropriate mappings.
  if (InputSyncRegion) {
    Instruction *NewSR = InputSyncRegion->clone();
    if (InputSyncRegion->hasName())
      NewSR->setName(InputSyncRegion->getName()+NameSuffix);
    NewEntry->getInstList().push_back(NewSR);
    VMap[InputSyncRegion] = NewSR;
  }

  // Clone Blocks into the new function.
  CloneIntoFunction(NewFunc, OldFunc, Blocks, VMap, ModuleLevelChanges,
                    Returns, NameSuffix, ReattachBlocks, DetachRethrowBlocks,
                    SharedEHEntries, SP, CodeInfo, TypeMapper, Materializer);

  // Add a branch in the new function to the cloned Header.
  BranchInst::Create(cast<BasicBlock>(VMap[Header]), NewEntry);
  // Add a return in the new function.
  ReturnInst::Create(Header->getContext(), NewExit);

  // If needed, create a landing pad and resume for the unwind destination in
  // the new function.
  if (OldUnwind) {
    LandingPadInst *LPad =
      LandingPadInst::Create(OldUnwind->getLandingPadInst()->getType(), 1,
                             "lpadval", NewUnwind);
    LPad->addClause(ConstantPointerNull::get(
                        PointerType::getInt8PtrTy(Header->getContext())));
    ResumeInst::Create(LPad, NewUnwind);
  }

  return NewFunc;
}

// Add alignment assumptions to parameters of outlined function, based on known
// alignment data in the caller.
void llvm::AddAlignmentAssumptions(
    const Function *Caller, const ValueSet &Args, ValueToValueMapTy &VMap,
    const Instruction *CallSite, AssumptionCache *AC, DominatorTree *DT) {
  auto &DL = Caller->getParent()->getDataLayout();
  for (Value *ArgVal : Args) {
    // Ignore arguments to non-pointer types
    if (!ArgVal->getType()->isPointerTy()) continue;
    Argument *Arg = cast<Argument>(VMap[ArgVal]);
    // Ignore arguments to non-pointer types
    if (!Arg->getType()->isPointerTy()) continue;
    // If the argument already has an alignment attribute, skip it.
    if (Arg->getParamAlignment()) continue;
    // Get any known alignment information for this argument's value.
    unsigned Align = getKnownAlignment(ArgVal, DL, CallSite, AC, DT);
    // If we have alignment data, add it as an attribute to the outlined
    // function's parameter.
    if (Align)
      Arg->addAttr(Attribute::getWithAlignment(Arg->getContext(), Align));
  }
}
