//===- IROutliner.cpp -- Outline Similar Regions ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
// Implementation for the IROutliner which is used by the IROutliner Pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/IROutliner.h"
#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include <map>
#include <set>
#include <vector>

#define DEBUG_TYPE "iroutliner"

using namespace llvm;
using namespace IRSimilarity;

/// The OutlinableGroup holds all the overarching information for outlining
/// a set of regions that are structurally similar to one another, such as the
/// types of the overall function, the output blocks, the sets of stores needed
/// and a list of the different regions. This information is used in the
/// deduplication of extracted regions with the same structure.
struct OutlinableGroup {
  /// The sections that could be outlined
  std::vector<OutlinableRegion *> Regions;

  /// The argument types for the function created as the overall function to
  /// replace the extracted function for each region.
  std::vector<Type *> ArgumentTypes;
  /// The FunctionType for the overall function.
  FunctionType *OutlinedFunctionType = nullptr;
  /// The Function for the collective overall function.
  Function *OutlinedFunction = nullptr;

  /// Flag for whether we should not consider this group of OutlinableRegions
  /// for extraction.
  bool IgnoreGroup = false;

  /// Flag for whether the \ref ArgumentTypes have been defined after the
  /// extraction of the first region.
  bool InputTypesSet = false;

  /// The number of input values in \ref ArgumentTypes.  Anything after this
  /// index in ArgumentTypes is an output argument.
  unsigned NumAggregateInputs = 0;

  /// For the \ref Regions, we look at every Value.  If it is a constant,
  /// we check whether it is the same in Region.
  ///
  /// \param [in,out] NotSame contains the global value numbers where the
  /// constant is not always the same, and must be passed in as an argument.
  void findSameConstants(DenseSet<unsigned> &NotSame);
};

/// Move the contents of \p SourceBB to before the last instruction of \p
/// TargetBB.
/// \param SourceBB - the BasicBlock to pull Instructions from.
/// \param TargetBB - the BasicBlock to put Instruction into.
static void moveBBContents(BasicBlock &SourceBB, BasicBlock &TargetBB) {
  BasicBlock::iterator BBCurr, BBEnd, BBNext;
  for (BBCurr = SourceBB.begin(), BBEnd = SourceBB.end(); BBCurr != BBEnd;
       BBCurr = BBNext) {
    BBNext = std::next(BBCurr);
    BBCurr->moveBefore(TargetBB, TargetBB.end());
  }
}

void OutlinableRegion::splitCandidate() {
  assert(!CandidateSplit && "Candidate already split!");

  Instruction *StartInst = (*Candidate->begin()).Inst;
  Instruction *EndInst = (*Candidate->end()).Inst;
  assert(StartInst && EndInst && "Expected a start and end instruction?");
  StartBB = StartInst->getParent();
  PrevBB = StartBB;

  // The basic block gets split like so:
  // block:                 block:
  //   inst1                  inst1
  //   inst2                  inst2
  //   region1               br block_to_outline
  //   region2              block_to_outline:
  //   region3          ->    region1
  //   region4                region2
  //   inst3                  region3
  //   inst4                  region4
  //                          br block_after_outline
  //                        block_after_outline:
  //                          inst3
  //                          inst4

  std::string OriginalName = PrevBB->getName().str();

  StartBB = PrevBB->splitBasicBlock(StartInst, OriginalName + "_to_outline");

  // This is the case for the inner block since we do not have to include
  // multiple blocks.
  EndBB = StartBB;
  FollowBB = EndBB->splitBasicBlock(EndInst, OriginalName + "_after_outline");

  CandidateSplit = true;
}

void OutlinableRegion::reattachCandidate() {
  assert(CandidateSplit && "Candidate is not split!");

  // The basic block gets reattached like so:
  // block:                        block:
  //   inst1                         inst1
  //   inst2                         inst2
  //   br block_to_outline           region1
  // block_to_outline:        ->     region2
  //   region1                       region3
  //   region2                       region4
  //   region3                       inst3
  //   region4                       inst4
  //   br block_after_outline
  // block_after_outline:
  //   inst3
  //   inst4
  assert(StartBB != nullptr && "StartBB for Candidate is not defined!");
  assert(FollowBB != nullptr && "StartBB for Candidate is not defined!");

  // StartBB should only have one predecessor since we put an unconditional
  // branch at the end of PrevBB when we split the BasicBlock.
  PrevBB = StartBB->getSinglePredecessor();
  assert(PrevBB != nullptr &&
         "No Predecessor for the region start basic block!");

  assert(PrevBB->getTerminator() && "Terminator removed from PrevBB!");
  assert(EndBB->getTerminator() && "Terminator removed from EndBB!");
  PrevBB->getTerminator()->eraseFromParent();
  EndBB->getTerminator()->eraseFromParent();

  moveBBContents(*StartBB, *PrevBB);

  BasicBlock *PlacementBB = PrevBB;
  if (StartBB != EndBB)
    PlacementBB = EndBB;
  moveBBContents(*FollowBB, *PlacementBB);

  PrevBB->replaceSuccessorsPhiUsesWith(StartBB, PrevBB);
  PrevBB->replaceSuccessorsPhiUsesWith(FollowBB, PlacementBB);
  StartBB->eraseFromParent();
  FollowBB->eraseFromParent();

  // Make sure to save changes back to the StartBB.
  StartBB = PrevBB;
  EndBB = nullptr;
  PrevBB = nullptr;
  FollowBB = nullptr;

  CandidateSplit = false;
}

/// Find whether \p V matches the Constants previously found for the \p GVN.
///
/// \param V - The value to check for consistency.
/// \param GVN - The global value number assigned to \p V.
/// \param GVNToConstant - The mapping of global value number to Constants.
/// \returns true if the Value matches the Constant mapped to by V and false if
/// it \p V is a Constant but does not match.
/// \returns None if \p V is not a Constant.
static Optional<bool>
constantMatches(Value *V, unsigned GVN,
                DenseMap<unsigned, Constant *> &GVNToConstant) {
  // See if we have a constants
  Constant *CST = dyn_cast<Constant>(V);
  if (!CST)
    return None;

  // Holds a mapping from a global value number to a Constant.
  DenseMap<unsigned, Constant *>::iterator GVNToConstantIt;
  bool Inserted;

  // If we have a constant, try to make a new entry in the GVNToConstant.
  std::tie(GVNToConstantIt, Inserted) =
      GVNToConstant.insert(std::make_pair(GVN, CST));
  // If it was found and is not equal, it is not the same. We do not
  // handle this case yet, and exit early.
  if (Inserted || (GVNToConstantIt->second == CST))
    return true;

  return false;
}

/// Find whether \p Region matches the global value numbering to Constant
/// mapping found so far.
///
/// \param Region - The OutlinableRegion we are checking for constants
/// \param GVNToConstant - The mapping of global value number to Constants.
/// \param NotSame - The set of global value numbers that do not have the same
/// constant in each region.
/// \returns true if all Constants are the same in every use of a Constant in \p
/// Region and false if not
static bool
collectRegionsConstants(OutlinableRegion &Region,
                        DenseMap<unsigned, Constant *> &GVNToConstant,
                        DenseSet<unsigned> &NotSame) {
  IRSimilarityCandidate &C = *Region.Candidate;
  for (IRInstructionData &ID : C) {

    // Iterate over the operands in an instruction. If the global value number,
    // assigned by the IRSimilarityCandidate, has been seen before, we check if
    // the the number has been found to be not the same value in each instance.
    for (Value *V : ID.OperVals) {
      Optional<unsigned> GVNOpt = C.getGVN(V);
      assert(GVNOpt.hasValue() && "Expected a GVN for operand?");
      unsigned GVN = GVNOpt.getValue();

      // If this global value has been found to not be the same, it could have
      // just been a register, check that it is not a constant value.
      if (NotSame.find(GVN) != NotSame.end()) {
        if (isa<Constant>(V))
          return false;
        continue;
      }

      // If it has been the same so far, we check the value for if the
      // associated Constant value match the previous instances of the same
      // global value number.  If the global value does not map to a Constant,
      // it is considered to not be the same value.
      Optional<bool> ConstantMatches = constantMatches(V, GVN, GVNToConstant);
      if (ConstantMatches.hasValue()) {
        if (ConstantMatches.getValue())
          continue;
        else
          return false;
      }

      // While this value is a register, it might not have been previously,
      // make sure we don't already have a constant mapped to this global value
      // number.
      if (GVNToConstant.find(GVN) != GVNToConstant.end())
        return false;

      NotSame.insert(GVN);
    }
  }

  return true;
}

void OutlinableGroup::findSameConstants(DenseSet<unsigned> &NotSame) {
  DenseMap<unsigned, Constant *> GVNToConstant;

  for (OutlinableRegion *Region : Regions)
    if (!collectRegionsConstants(*Region, GVNToConstant, NotSame)) {
      IgnoreGroup = true;
      return;
    }
}

Function *IROutliner::createFunction(Module &M, OutlinableGroup &Group,
                                     unsigned FunctionNameSuffix) {
  assert(!Group.OutlinedFunction && "Function is already defined!");

  Group.OutlinedFunctionType = FunctionType::get(
      Type::getVoidTy(M.getContext()), Group.ArgumentTypes, false);

  // These functions will only be called from within the same module, so
  // we can set an internal linkage.
  Group.OutlinedFunction = Function::Create(
      Group.OutlinedFunctionType, GlobalValue::InternalLinkage,
      "outlined_ir_func_" + std::to_string(FunctionNameSuffix), M);

  Group.OutlinedFunction->addFnAttr(Attribute::OptimizeForSize);
  Group.OutlinedFunction->addFnAttr(Attribute::MinSize);

  return Group.OutlinedFunction;
}

/// Move each BasicBlock in \p Old to \p New.
///
/// \param [in] Old - the function to move the basic blocks from.
/// \param [in] New - The function to move the basic blocks to.
/// \returns the first return block for the function in New.
static BasicBlock *moveFunctionData(Function &Old, Function &New) {
  Function::iterator CurrBB, NextBB, FinalBB;
  BasicBlock *NewEnd = nullptr;
  std::vector<Instruction *> DebugInsts;
  for (CurrBB = Old.begin(), FinalBB = Old.end(); CurrBB != FinalBB;
       CurrBB = NextBB) {
    NextBB = std::next(CurrBB);
    CurrBB->removeFromParent();
    CurrBB->insertInto(&New);
    Instruction *I = CurrBB->getTerminator();
    if (ReturnInst *RI = dyn_cast<ReturnInst>(I))
      NewEnd = &(*CurrBB);
  }

  assert(NewEnd && "No return instruction for new function?");
  return NewEnd;
}

/// Find the GVN for the inputs that have been found by the CodeExtractor,
/// excluding the ones that will be removed by llvm.assumes as these will be
/// removed by the CodeExtractor.
///
/// \param [in] C - The IRSimilarityCandidate containing the region we are
/// analyzing.
/// \param [in] CurrentInputs - The set of inputs found by the
/// CodeExtractor.
/// \param [out] CurrentInputNumbers - The global value numbers for the
/// extracted arguments.
static void mapInputsToGVNs(IRSimilarityCandidate &C,
                            SetVector<Value *> &CurrentInputs,
                            std::vector<unsigned> &EndInputNumbers) {
  // Get the global value number for each input.
  for (Value *Input : CurrentInputs) {
    assert(Input && "Have a nullptr as an input");
    assert(C.getGVN(Input).hasValue() &&
           "Could not find a numbering for the given input");
    EndInputNumbers.push_back(C.getGVN(Input).getValue());
  }
}

/// Find the input GVNs and the output values for a region of Instructions.
/// Using the code extractor, we collect the inputs to the extracted function.
///
/// The \p Region can be identifed as needing to be ignored in this function.
/// It should be checked whether it should be ignored after a call to this
/// function.
///
/// \param [in,out] Region - The region of code to be analyzed.
/// \param [out] InputGVNs - The global value numbers for the extracted
/// arguments.
/// \param [out] ArgInputs - The values of the inputs to the extracted function.
static void getCodeExtractorArguments(OutlinableRegion &Region,
                                      std::vector<unsigned> &InputGVNs,
                                      SetVector<Value *> &ArgInputs) {
  IRSimilarityCandidate &C = *Region.Candidate;

  // OverallInputs are the inputs to the region found by the CodeExtractor,
  // SinkCands and HoistCands are used by the CodeExtractor to find sunken
  // allocas of values whose lifetimes are contained completely within the
  // outlined region. Outputs are values used outside of the outlined region
  // found by the CodeExtractor.
  SetVector<Value *> OverallInputs, SinkCands, HoistCands, Outputs;

  // Use the code extractor to get the inputs and outputs, without sunken
  // allocas or removing llvm.assumes.
  CodeExtractor *CE = Region.CE;
  CE->findInputsOutputs(OverallInputs, Outputs, SinkCands);
  assert(Region.StartBB && "Region must have a start BasicBlock!");
  Function *OrigF = Region.StartBB->getParent();
  CodeExtractorAnalysisCache CEAC(*OrigF);
  BasicBlock *Dummy = nullptr;

  // The region may be ineligible due to VarArgs in the parent function. In this
  // case we ignore the region.
  if (!CE->isEligible()) {
    Region.IgnoreRegion = true;
    return;
  }

  // Find if any values are going to be sunk into the function when extracted
  CE->findAllocas(CEAC, SinkCands, HoistCands, Dummy);
  CE->findInputsOutputs(ArgInputs, Outputs, SinkCands);

  // TODO: Support regions with output values.  Outputs add an extra layer of
  // resolution that adds too much complexity at this stage.
  if (Outputs.size() > 0) {
    Region.IgnoreRegion = true;
    return;
  }

  // TODO: Support regions with sunken allocas: values whose lifetimes are
  // contained completely within the outlined region.  These are not guaranteed
  // to be the same in every region, so we must elevate them all to arguments
  // when they appear.  If these values are not equal, it means there is some
  // Input in OverallInputs that was removed for ArgInputs.
  if (ArgInputs.size() != OverallInputs.size()) {
    Region.IgnoreRegion = true;
    return;
  }

  mapInputsToGVNs(C, OverallInputs, InputGVNs);
}

/// Look over the inputs and map each input argument to an argument in the
/// overall function for the regions.  This creates a way to replace the
/// arguments of the extracted function, with the arguments of the new overall
/// function.
///
/// \param [in,out] Region - The region of code to be analyzed.
/// \param [in] InputsGVNs - The global value numbering of the input values
/// collected.
/// \param [in] ArgInputs - The values of the arguments to the extracted
/// function.
static void
findExtractedInputToOverallInputMapping(OutlinableRegion &Region,
                                        std::vector<unsigned> InputGVNs,
                                        SetVector<Value *> &ArgInputs) {

  IRSimilarityCandidate &C = *Region.Candidate;
  OutlinableGroup &Group = *Region.Parent;

  // This counts the argument number in the overall function.
  unsigned TypeIndex = 0;

  // This counts the argument number in the extracted function.
  unsigned OriginalIndex = 0;

  // Find the mapping of the extracted arguments to the arguments for the
  // overall function.
  for (unsigned InputVal : InputGVNs) {
    Optional<Value *> InputOpt = C.fromGVN(InputVal);
    assert(InputOpt.hasValue() && "Global value number not found?");
    Value *Input = InputOpt.getValue();

    if (!Group.InputTypesSet)
      Group.ArgumentTypes.push_back(Input->getType());

    // It is not a constant, check if it is a sunken alloca.  If it is not,
    // create the mapping from extracted to overall.  If it is, create the
    // mapping of the index to the value.
    assert(ArgInputs.count(Input) && "Input cannot be found!");

    Region.ExtractedArgToAgg.insert(std::make_pair(OriginalIndex, TypeIndex));
    Region.AggArgToExtracted.insert(std::make_pair(TypeIndex, OriginalIndex));
    OriginalIndex++;
    TypeIndex++;
  }

  // If we do not have definitions for the OutlinableGroup holding the region,
  // set the length of the inputs here.  We should have the same inputs for
  // all of the different regions contained in the OutlinableGroup since they
  // are all structurally similar to one another
  if (!Group.InputTypesSet) {
    Group.NumAggregateInputs = TypeIndex;
    Group.InputTypesSet = true;
  }

  Region.NumExtractedInputs = OriginalIndex;
}

void IROutliner::findAddInputsOutputs(Module &M, OutlinableRegion &Region) {
  std::vector<unsigned> Inputs;
  SetVector<Value *> ArgInputs;

  getCodeExtractorArguments(Region, Inputs, ArgInputs);

  if (Region.IgnoreRegion)
    return;

  // Map the inputs found by the CodeExtractor to the arguments found for
  // the overall function.
  findExtractedInputToOverallInputMapping(Region, Inputs, ArgInputs);
}

/// Replace the extracted function in the Region with a call to the overall
/// function constructed from the deduplicated similar regions, replacing and
/// remapping the values passed to the extracted function as arguments to the
/// new arguments of the overall function.
///
/// \param [in] M - The module to outline from.
/// \param [in] Region - The regions of extracted code to be replaced with a new
/// function.
/// \returns a call instruction with the replaced function.
CallInst *replaceCalledFunction(Module &M, OutlinableRegion &Region) {
  std::vector<Value *> NewCallArgs;

  OutlinableGroup &Group = *Region.Parent;
  CallInst *Call = Region.Call;
  assert(Call && "Call to replace is nullptr?");
  Function *AggFunc = Group.OutlinedFunction;
  assert(AggFunc && "Function to replace with is nullptr?");

  // If the arguments are the same size, there are not values that need to be
  // made argument, or different output registers to handle.  We can simply
  // replace the called function in this case.
  assert(AggFunc->arg_size() == Call->arg_size() &&
         "Can only replace calls with the same number of arguments!");

  LLVM_DEBUG(dbgs() << "Replace call to " << *Call << " with call to "
                    << *AggFunc << " with same number of arguments\n");
  Call->setCalledFunction(AggFunc);
  return Call;
}

// Within an extracted function, replace the argument uses of the extracted
// region with the arguments of the function for an OutlinableGroup.
//
// \param OS [in] - The region of extracted code to be changed.
static void replaceArgumentUses(OutlinableRegion &Region) {
  OutlinableGroup &Group = *Region.Parent;
  assert(Region.ExtractedFunction && "Region has no extracted function?");

  for (unsigned ArgIdx = 0; ArgIdx < Region.ExtractedFunction->arg_size();
       ArgIdx++) {
    assert(Region.ExtractedArgToAgg.find(ArgIdx) !=
               Region.ExtractedArgToAgg.end() &&
           "No mapping from extracted to outlined?");
    unsigned AggArgIdx = Region.ExtractedArgToAgg.find(ArgIdx)->second;
    Argument *AggArg = Group.OutlinedFunction->getArg(AggArgIdx);
    Argument *Arg = Region.ExtractedFunction->getArg(ArgIdx);
    // The argument is an input, so we can simply replace it with the overall
    // argument value
    LLVM_DEBUG(dbgs() << "Replacing uses of input " << *Arg << " in function "
                      << *Region.ExtractedFunction << " with " << *AggArg
                      << " in function " << *Group.OutlinedFunction << "\n");
    Arg->replaceAllUsesWith(AggArg);
  }
}

/// Fill the new function that will serve as the replacement function for all of
/// the extracted regions of a certain structure from the first region in the
/// list of regions.  Replace this first region's extracted function with the
/// new overall function.
///
/// \param M [in] - The module we are outlining from.
/// \param CurrentGroup [in] - The group of regions to be outlined.
/// \param FuncsToRemove [in,out] - Extracted functions to erase from module
/// once outlining is complete.
static void fillOverallFunction(Module &M, OutlinableGroup &CurrentGroup,
                                std::vector<Function *> &FuncsToRemove) {
  OutlinableRegion *CurrentOS = CurrentGroup.Regions[0];

  // Move first extracted function's instructions into new function
  LLVM_DEBUG(dbgs() << "Move instructions from "
                    << *CurrentOS->ExtractedFunction << " to instruction "
                    << *CurrentGroup.OutlinedFunction << "\n");
  moveFunctionData(*CurrentOS->ExtractedFunction,
                   *CurrentGroup.OutlinedFunction);

  // Transfer the attributes
  for (Attribute A :
       CurrentOS->ExtractedFunction->getAttributes().getFnAttributes())
    CurrentGroup.OutlinedFunction->addFnAttr(A);

  replaceArgumentUses(*CurrentOS);

  // Replace the call to the extracted function with the outlined function.
  CurrentOS->Call = replaceCalledFunction(M, *CurrentOS);

  // We only delete the extracted funcitons at the end since we may need to
  // reference instructions contained in them for mapping purposes.
  FuncsToRemove.push_back(CurrentOS->ExtractedFunction);
}

void IROutliner::deduplicateExtractedSections(
    Module &M, OutlinableGroup &CurrentGroup,
    std::vector<Function *> &FuncsToRemove, unsigned &OutlinedFunctionNum) {
  createFunction(M, CurrentGroup, OutlinedFunctionNum);

  std::vector<BasicBlock *> OutputStoreBBs;

  OutlinableRegion *CurrentOS;

  fillOverallFunction(M, CurrentGroup, FuncsToRemove);

  // Do the same for the other extracted functions
  for (unsigned Idx = 1; Idx < CurrentGroup.Regions.size(); Idx++) {
    CurrentOS = CurrentGroup.Regions[Idx];

    replaceArgumentUses(*CurrentOS);
    CurrentOS->Call = replaceCalledFunction(M, *CurrentOS);
    FuncsToRemove.push_back(CurrentOS->ExtractedFunction);
  }

  OutlinedFunctionNum++;
}

void IROutliner::pruneIncompatibleRegions(
    std::vector<IRSimilarityCandidate> &CandidateVec,
    OutlinableGroup &CurrentGroup) {
  bool PreviouslyOutlined;

  // Sort from beginning to end, so the IRSimilarityCandidates are in order.
  stable_sort(CandidateVec, [](const IRSimilarityCandidate &LHS,
                               const IRSimilarityCandidate &RHS) {
    return LHS.getStartIdx() < RHS.getStartIdx();
  });

  unsigned CurrentEndIdx = 0;
  for (IRSimilarityCandidate &IRSC : CandidateVec) {
    PreviouslyOutlined = false;
    unsigned StartIdx = IRSC.getStartIdx();
    unsigned EndIdx = IRSC.getEndIdx();

    for (unsigned Idx = StartIdx; Idx <= EndIdx; Idx++)
      if (Outlined.contains(Idx)) {
        PreviouslyOutlined = true;
        break;
      }

    if (PreviouslyOutlined)
      continue;

    // TODO: If in the future we can outline across BasicBlocks, we will need to
    // check all BasicBlocks contained in the region.
    if (IRSC.getStartBB()->hasAddressTaken())
      continue;

    // Greedily prune out any regions that will overlap with already chosen
    // regions.
    if (CurrentEndIdx != 0 && StartIdx <= CurrentEndIdx)
      continue;

    bool BadInst = any_of(IRSC, [this](IRInstructionData &ID) {
      return !this->InstructionClassifier.visit(ID.Inst);
    });

    if (BadInst)
      continue;

    OutlinableRegion *OS = new (RegionAllocator.Allocate())
        OutlinableRegion(IRSC, CurrentGroup);
    CurrentGroup.Regions.push_back(OS);

    CurrentEndIdx = EndIdx;
  }
}

bool IROutliner::extractSection(OutlinableRegion &Region) {
  assert(Region.StartBB != nullptr &&
         "StartBB for the OutlinableRegion is nullptr!");
  assert(Region.FollowBB != nullptr &&
         "StartBB for the OutlinableRegion is nullptr!");
  Function *OrigF = Region.StartBB->getParent();
  CodeExtractorAnalysisCache CEAC(*OrigF);
  Region.ExtractedFunction = Region.CE->extractCodeRegion(CEAC);

  // If the extraction was successful, find the BasicBlock, and reassign the
  // OutlinableRegion blocks
  if (!Region.ExtractedFunction) {
    LLVM_DEBUG(dbgs() << "CodeExtractor failed to outline " << Region.StartBB
                      << "\n");
    Region.reattachCandidate();
    return false;
  }

  BasicBlock *RewrittenBB = Region.FollowBB->getSinglePredecessor();
  Region.StartBB = RewrittenBB;
  Region.EndBB = RewrittenBB;

  // The sequences of outlinable regions has now changed.  We must fix the
  // IRInstructionDataList for consistency.  Although they may not be illegal
  // instructions, they should not be compared with anything else as they
  // should not be outlined in this round.  So marking these as illegal is
  // allowed.
  IRInstructionDataList *IDL = Region.Candidate->front()->IDL;
  Instruction *BeginRewritten = &*RewrittenBB->begin();
  Instruction *EndRewritten = &*RewrittenBB->begin();
  Region.NewFront = new (InstDataAllocator.Allocate()) IRInstructionData(
      *BeginRewritten, InstructionClassifier.visit(*BeginRewritten), *IDL);
  Region.NewBack = new (InstDataAllocator.Allocate()) IRInstructionData(
      *EndRewritten, InstructionClassifier.visit(*EndRewritten), *IDL);

  // Insert the first IRInstructionData of the new region in front of the
  // first IRInstructionData of the IRSimilarityCandidate.
  IDL->insert(Region.Candidate->begin(), *Region.NewFront);
  // Insert the first IRInstructionData of the new region after the
  // last IRInstructionData of the IRSimilarityCandidate.
  IDL->insert(Region.Candidate->end(), *Region.NewBack);
  // Remove the IRInstructionData from the IRSimilarityCandidate.
  IDL->erase(Region.Candidate->begin(), std::prev(Region.Candidate->end()));

  assert(RewrittenBB != nullptr &&
         "Could not find a predecessor after extraction!");

  // Iterate over the new set of instructions to find the new call
  // instruction.
  for (Instruction &I : *RewrittenBB)
    if (CallInst *CI = dyn_cast<CallInst>(&I))
      if (Region.ExtractedFunction == CI->getCalledFunction()) {
        Region.Call = CI;
        break;
      }
  Region.reattachCandidate();
  return true;
}

unsigned IROutliner::doOutline(Module &M) {
  // Find the possibile similarity sections.
  IRSimilarityIdentifier &Identifier = getIRSI(M);
  SimilarityGroupList &SimilarityCandidates = *Identifier.getSimilarity();

  // Sort them by size of extracted sections
  unsigned OutlinedFunctionNum = 0;
  // If we only have one SimilarityGroup in SimilarityCandidates, we do not have
  // to sort them by the potential number of instructions to be outlined
  if (SimilarityCandidates.size() > 1)
    llvm::stable_sort(SimilarityCandidates,
                      [](const std::vector<IRSimilarityCandidate> &LHS,
                         const std::vector<IRSimilarityCandidate> &RHS) {
                        return LHS[0].getLength() * LHS.size() >
                               RHS[0].getLength() * RHS.size();
                      });

  DenseSet<unsigned> NotSame;
  std::vector<Function *> FuncsToRemove;
  // Iterate over the possible sets of similarity.
  for (SimilarityGroup &CandidateVec : SimilarityCandidates) {
    OutlinableGroup CurrentGroup;

    // Remove entries that were previously outlined
    pruneIncompatibleRegions(CandidateVec, CurrentGroup);

    // We pruned the number of regions to 0 to 1, meaning that it's not worth
    // trying to outlined since there is no compatible similar instance of this
    // code.
    if (CurrentGroup.Regions.size() < 2)
      continue;

    // Determine if there are any values that are the same constant throughout
    // each section in the set.
    NotSame.clear();
    CurrentGroup.findSameConstants(NotSame);

    if (CurrentGroup.IgnoreGroup)
      continue;

    // Create a CodeExtractor for each outlinable region. Identify inputs and
    // outputs for each section using the code extractor and create the argument
    // types for the Aggregate Outlining Function.
    std::vector<OutlinableRegion *> OutlinedRegions;
    for (OutlinableRegion *OS : CurrentGroup.Regions) {
      // Break the outlinable region out of its parent BasicBlock into its own
      // BasicBlocks (see function implementation).
      OS->splitCandidate();
      std::vector<BasicBlock *> BE = {OS->StartBB};
      OS->CE = new (ExtractorAllocator.Allocate())
          CodeExtractor(BE, nullptr, false, nullptr, nullptr, nullptr, false,
                        false, "outlined");
      findAddInputsOutputs(M, *OS);
      if (!OS->IgnoreRegion)
        OutlinedRegions.push_back(OS);
      else
        OS->reattachCandidate();
    }

    CurrentGroup.Regions = std::move(OutlinedRegions);

    // Create functions out of all the sections, and mark them as outlined.
    OutlinedRegions.clear();
    for (OutlinableRegion *OS : CurrentGroup.Regions) {
      bool FunctionOutlined = extractSection(*OS);
      if (FunctionOutlined) {
        unsigned StartIdx = OS->Candidate->getStartIdx();
        unsigned EndIdx = OS->Candidate->getEndIdx();
        for (unsigned Idx = StartIdx; Idx <= EndIdx; Idx++)
          Outlined.insert(Idx);

        OutlinedRegions.push_back(OS);
      }
    }

    CurrentGroup.Regions = std::move(OutlinedRegions);

    if (CurrentGroup.Regions.empty())
      continue;

    deduplicateExtractedSections(M, CurrentGroup, FuncsToRemove,
                                 OutlinedFunctionNum);
  }

  for (Function *F : FuncsToRemove)
    F->eraseFromParent();

  return OutlinedFunctionNum;
}

bool IROutliner::run(Module &M) { return doOutline(M) > 0; }

// Pass Manager Boilerplate
class IROutlinerLegacyPass : public ModulePass {
public:
  static char ID;
  IROutlinerLegacyPass() : ModulePass(ID) {
    initializeIROutlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<IRSimilarityIdentifierWrapperPass>();
  }

  bool runOnModule(Module &M) override;
};

bool IROutlinerLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  auto GTTI = [this](Function &F) -> TargetTransformInfo & {
    return this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  };

  auto GIRSI = [this](Module &) -> IRSimilarityIdentifier & {
    return this->getAnalysis<IRSimilarityIdentifierWrapperPass>().getIRSI();
  };

  return IROutliner(GTTI, GIRSI).run(M);
}

PreservedAnalyses IROutlinerPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  std::function<TargetTransformInfo &(Function &)> GTTI =
      [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  std::function<IRSimilarityIdentifier &(Module &)> GIRSI =
      [&AM](Module &M) -> IRSimilarityIdentifier & {
    return AM.getResult<IRSimilarityAnalysis>(M);
  };

  if (IROutliner(GTTI, GIRSI).run(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

char IROutlinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(IROutlinerLegacyPass, "iroutliner", "IR Outliner", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(IRSimilarityIdentifierWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(IROutlinerLegacyPass, "iroutliner", "IR Outliner", false,
                    false)

ModulePass *llvm::createIROutlinerPass() { return new IROutlinerLegacyPass(); }
