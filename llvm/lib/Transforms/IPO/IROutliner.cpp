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
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Mangler.h"
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

// Set to true if the user wants the ir outliner to run on linkonceodr linkage
// functions. This is false by default because the linker can dedupe linkonceodr
// functions. Since the outliner is confined to a single module (modulo LTO),
// this is off by default. It should, however, be the default behavior in
// LTO.
static cl::opt<bool> EnableLinkOnceODRIROutlining(
    "enable-linkonceodr-ir-outlining", cl::Hidden,
    cl::desc("Enable the IR outliner on linkonceodr functions"),
    cl::init(false));

// This is a debug option to test small pieces of code to ensure that outlining
// works correctly.
static cl::opt<bool> NoCostModel(
    "ir-outlining-no-cost", cl::init(false), cl::ReallyHidden,
    cl::desc("Debug option to outline greedily, without restriction that "
             "calculated benefit outweighs cost"));

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

  /// The return block for the overall function.
  BasicBlock *EndBB = nullptr;

  /// A set containing the different GVN store sets needed. Each array contains
  /// a sorted list of the different values that need to be stored into output
  /// registers.
  DenseSet<ArrayRef<unsigned>> OutputGVNCombinations;

  /// Flag for whether the \ref ArgumentTypes have been defined after the
  /// extraction of the first region.
  bool InputTypesSet = false;

  /// The number of input values in \ref ArgumentTypes.  Anything after this
  /// index in ArgumentTypes is an output argument.
  unsigned NumAggregateInputs = 0;

  /// The number of instructions that will be outlined by extracting \ref
  /// Regions.
  InstructionCost Benefit = 0;
  /// The number of added instructions needed for the outlining of the \ref
  /// Regions.
  InstructionCost Cost = 0;

  /// The argument that needs to be marked with the swifterr attribute.  If not
  /// needed, there is no value.
  Optional<unsigned> SwiftErrorArgument;

  /// For the \ref Regions, we look at every Value.  If it is a constant,
  /// we check whether it is the same in Region.
  ///
  /// \param [in,out] NotSame contains the global value numbers where the
  /// constant is not always the same, and must be passed in as an argument.
  void findSameConstants(DenseSet<unsigned> &NotSame);

  /// For the regions, look at each set of GVN stores needed and account for
  /// each combination.  Add an argument to the argument types if there is
  /// more than one combination.
  ///
  /// \param [in] M - The module we are outlining from.
  void collectGVNStoreSets(Module &M);
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

InstructionCost OutlinableRegion::getBenefit(TargetTransformInfo &TTI) {
  InstructionCost Benefit = 0;

  // Estimate the benefit of outlining a specific sections of the program.  We
  // delegate mostly this task to the TargetTransformInfo so that if the target
  // has specific changes, we can have a more accurate estimate.

  // However, getInstructionCost delegates the code size calculation for
  // arithmetic instructions to getArithmeticInstrCost in
  // include/Analysis/TargetTransformImpl.h, where it always estimates that the
  // code size for a division and remainder instruction to be equal to 4, and
  // everything else to 1.  This is not an accurate representation of the
  // division instruction for targets that have a native division instruction.
  // To be overly conservative, we only add 1 to the number of instructions for
  // each division instruction.
  for (Instruction &I : *StartBB) {
    switch (I.getOpcode()) {
    case Instruction::FDiv:
    case Instruction::FRem:
    case Instruction::SDiv:
    case Instruction::SRem:
    case Instruction::UDiv:
    case Instruction::URem:
      Benefit += 1;
      break;
    default:
      Benefit += TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);
      break;
    }
  }

  return Benefit;
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
  bool ConstantsTheSame = true;

  IRSimilarityCandidate &C = *Region.Candidate;
  for (IRInstructionData &ID : C) {

    // Iterate over the operands in an instruction. If the global value number,
    // assigned by the IRSimilarityCandidate, has been seen before, we check if
    // the the number has been found to be not the same value in each instance.
    for (Value *V : ID.OperVals) {
      Optional<unsigned> GVNOpt = C.getGVN(V);
      assert(GVNOpt.hasValue() && "Expected a GVN for operand?");
      unsigned GVN = GVNOpt.getValue();

      // Check if this global value has been found to not be the same already.
      if (NotSame.contains(GVN)) {
        if (isa<Constant>(V))
          ConstantsTheSame = false;
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
          ConstantsTheSame = false;
      }

      // While this value is a register, it might not have been previously,
      // make sure we don't already have a constant mapped to this global value
      // number.
      if (GVNToConstant.find(GVN) != GVNToConstant.end())
        ConstantsTheSame = false;

      NotSame.insert(GVN);
    }
  }

  return ConstantsTheSame;
}

void OutlinableGroup::findSameConstants(DenseSet<unsigned> &NotSame) {
  DenseMap<unsigned, Constant *> GVNToConstant;

  for (OutlinableRegion *Region : Regions)
    collectRegionsConstants(*Region, GVNToConstant, NotSame);
}

void OutlinableGroup::collectGVNStoreSets(Module &M) {
  for (OutlinableRegion *OS : Regions)
    OutputGVNCombinations.insert(OS->GVNStores);

  // We are adding an extracted argument to decide between which output path
  // to use in the basic block.  It is used in a switch statement and only
  // needs to be an integer.
  if (OutputGVNCombinations.size() > 1)
    ArgumentTypes.push_back(Type::getInt32Ty(M.getContext()));
}

/// Get the subprogram if it exists for one of the outlined regions.
///
/// \param [in] Group - The set of regions to find a subprogram for.
/// \returns the subprogram if it exists, or nullptr.
static DISubprogram *getSubprogramOrNull(OutlinableGroup &Group) {
  for (OutlinableRegion *OS : Group.Regions)
    if (Function *F = OS->Call->getFunction())
      if (DISubprogram *SP = F->getSubprogram())
        return SP;

  return nullptr;
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

  // Transfer the swifterr attribute to the correct function parameter.
  if (Group.SwiftErrorArgument.hasValue())
    Group.OutlinedFunction->addParamAttr(Group.SwiftErrorArgument.getValue(),
                                         Attribute::SwiftError);

  Group.OutlinedFunction->addFnAttr(Attribute::OptimizeForSize);
  Group.OutlinedFunction->addFnAttr(Attribute::MinSize);

  // If there's a DISubprogram associated with this outlined function, then
  // emit debug info for the outlined function.
  if (DISubprogram *SP = getSubprogramOrNull(Group)) {
    Function *F = Group.OutlinedFunction;
    // We have a DISubprogram. Get its DICompileUnit.
    DICompileUnit *CU = SP->getUnit();
    DIBuilder DB(M, true, CU);
    DIFile *Unit = SP->getFile();
    Mangler Mg;
    // Get the mangled name of the function for the linkage name.
    std::string Dummy;
    llvm::raw_string_ostream MangledNameStream(Dummy);
    Mg.getNameWithPrefix(MangledNameStream, F, false);

    DISubprogram *OutlinedSP = DB.createFunction(
        Unit /* Context */, F->getName(), MangledNameStream.str(),
        Unit /* File */,
        0 /* Line 0 is reserved for compiler-generated code. */,
        DB.createSubroutineType(DB.getOrCreateTypeArray(None)), /* void type */
        0, /* Line 0 is reserved for compiler-generated code. */
        DINode::DIFlags::FlagArtificial /* Compiler-generated code. */,
        /* Outlined code is optimized code by definition. */
        DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);

    // Don't add any new variables to the subprogram.
    DB.finalizeSubprogram(OutlinedSP);

    // Attach subprogram to the function.
    F->setSubprogram(OutlinedSP);
    // We're done with the DIBuilder.
    DB.finalize();
  }

  return Group.OutlinedFunction;
}

/// Move each BasicBlock in \p Old to \p New.
///
/// \param [in] Old - The function to move the basic blocks from.
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
    if (isa<ReturnInst>(I))
      NewEnd = &(*CurrBB);

    for (Instruction &Val : *CurrBB) {
      // We must handle the scoping of called functions differently than
      // other outlined instructions.
      if (!isa<CallInst>(&Val)) {
        // Remove the debug information for outlined functions.
        Val.setDebugLoc(DebugLoc());
        continue;
      }

      // From this point we are only handling call instructions.
      CallInst *CI = cast<CallInst>(&Val);

      // We add any debug statements here, to be removed after.  Since the
      // instructions originate from many different locations in the program,
      // it will cause incorrect reporting from a debugger if we keep the
      // same debug instructions.
      if (isa<DbgInfoIntrinsic>(CI)) {
        DebugInsts.push_back(&Val);
        continue;
      }

      // Edit the scope of called functions inside of outlined functions.
      if (DISubprogram *SP = New.getSubprogram()) {
        DILocation *DI = DILocation::get(New.getContext(), 0, 0, SP);
        Val.setDebugLoc(DI);
      }
    }

    for (Instruction *I : DebugInsts)
      I->eraseFromParent();
  }

  assert(NewEnd && "No return instruction for new function?");
  return NewEnd;
}

/// Find the the constants that will need to be lifted into arguments
/// as they are not the same in each instance of the region.
///
/// \param [in] C - The IRSimilarityCandidate containing the region we are
/// analyzing.
/// \param [in] NotSame - The set of global value numbers that do not have a
/// single Constant across all OutlinableRegions similar to \p C.
/// \param [out] Inputs - The list containing the global value numbers of the
/// arguments needed for the region of code.
static void findConstants(IRSimilarityCandidate &C, DenseSet<unsigned> &NotSame,
                          std::vector<unsigned> &Inputs) {
  DenseSet<unsigned> Seen;
  // Iterate over the instructions, and find what constants will need to be
  // extracted into arguments.
  for (IRInstructionDataList::iterator IDIt = C.begin(), EndIDIt = C.end();
       IDIt != EndIDIt; IDIt++) {
    for (Value *V : (*IDIt).OperVals) {
      // Since these are stored before any outlining, they will be in the
      // global value numbering.
      unsigned GVN = C.getGVN(V).getValue();
      if (isa<Constant>(V))
        if (NotSame.contains(GVN) && !Seen.contains(GVN)) {
          Inputs.push_back(GVN);
          Seen.insert(GVN);
        }
    }
  }
}

/// Find the GVN for the inputs that have been found by the CodeExtractor.
///
/// \param [in] C - The IRSimilarityCandidate containing the region we are
/// analyzing.
/// \param [in] CurrentInputs - The set of inputs found by the
/// CodeExtractor.
/// \param [in] OutputMappings - The mapping of values that have been replaced
/// by a new output value.
/// \param [out] EndInputNumbers - The global value numbers for the extracted
/// arguments.
static void mapInputsToGVNs(IRSimilarityCandidate &C,
                            SetVector<Value *> &CurrentInputs,
                            const DenseMap<Value *, Value *> &OutputMappings,
                            std::vector<unsigned> &EndInputNumbers) {
  // Get the Global Value Number for each input.  We check if the Value has been
  // replaced by a different value at output, and use the original value before
  // replacement.
  for (Value *Input : CurrentInputs) {
    assert(Input && "Have a nullptr as an input");
    if (OutputMappings.find(Input) != OutputMappings.end())
      Input = OutputMappings.find(Input)->second;
    assert(C.getGVN(Input).hasValue() &&
           "Could not find a numbering for the given input");
    EndInputNumbers.push_back(C.getGVN(Input).getValue());
  }
}

/// Find the original value for the \p ArgInput values if any one of them was
/// replaced during a previous extraction.
///
/// \param [in] ArgInputs - The inputs to be extracted by the code extractor.
/// \param [in] OutputMappings - The mapping of values that have been replaced
/// by a new output value.
/// \param [out] RemappedArgInputs - The remapped values according to
/// \p OutputMappings that will be extracted.
static void
remapExtractedInputs(const ArrayRef<Value *> ArgInputs,
                     const DenseMap<Value *, Value *> &OutputMappings,
                     SetVector<Value *> &RemappedArgInputs) {
  // Get the global value number for each input that will be extracted as an
  // argument by the code extractor, remapping if needed for reloaded values.
  for (Value *Input : ArgInputs) {
    if (OutputMappings.find(Input) != OutputMappings.end())
      Input = OutputMappings.find(Input)->second;
    RemappedArgInputs.insert(Input);
  }
}

/// Find the input GVNs and the output values for a region of Instructions.
/// Using the code extractor, we collect the inputs to the extracted function.
///
/// The \p Region can be identified as needing to be ignored in this function.
/// It should be checked whether it should be ignored after a call to this
/// function.
///
/// \param [in,out] Region - The region of code to be analyzed.
/// \param [out] InputGVNs - The global value numbers for the extracted
/// arguments.
/// \param [in] NotSame - The global value numbers in the region that do not
/// have the same constant value in the regions structurally similar to
/// \p Region.
/// \param [in] OutputMappings - The mapping of values that have been replaced
/// by a new output value after extraction.
/// \param [out] ArgInputs - The values of the inputs to the extracted function.
/// \param [out] Outputs - The set of values extracted by the CodeExtractor
/// as outputs.
static void getCodeExtractorArguments(
    OutlinableRegion &Region, std::vector<unsigned> &InputGVNs,
    DenseSet<unsigned> &NotSame, DenseMap<Value *, Value *> &OutputMappings,
    SetVector<Value *> &ArgInputs, SetVector<Value *> &Outputs) {
  IRSimilarityCandidate &C = *Region.Candidate;

  // OverallInputs are the inputs to the region found by the CodeExtractor,
  // SinkCands and HoistCands are used by the CodeExtractor to find sunken
  // allocas of values whose lifetimes are contained completely within the
  // outlined region. PremappedInputs are the arguments found by the
  // CodeExtractor, removing conditions such as sunken allocas, but that
  // may need to be remapped due to the extracted output values replacing
  // the original values. We use DummyOutputs for this first run of finding
  // inputs and outputs since the outputs could change during findAllocas,
  // the correct set of extracted outputs will be in the final Outputs ValueSet.
  SetVector<Value *> OverallInputs, PremappedInputs, SinkCands, HoistCands,
      DummyOutputs;

  // Use the code extractor to get the inputs and outputs, without sunken
  // allocas or removing llvm.assumes.
  CodeExtractor *CE = Region.CE;
  CE->findInputsOutputs(OverallInputs, DummyOutputs, SinkCands);
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
  CE->findInputsOutputs(PremappedInputs, Outputs, SinkCands);

  // TODO: Support regions with sunken allocas: values whose lifetimes are
  // contained completely within the outlined region.  These are not guaranteed
  // to be the same in every region, so we must elevate them all to arguments
  // when they appear.  If these values are not equal, it means there is some
  // Input in OverallInputs that was removed for ArgInputs.
  if (OverallInputs.size() != PremappedInputs.size()) {
    Region.IgnoreRegion = true;
    return;
  }

  findConstants(C, NotSame, InputGVNs);

  mapInputsToGVNs(C, OverallInputs, OutputMappings, InputGVNs);

  remapExtractedInputs(PremappedInputs.getArrayRef(), OutputMappings,
                       ArgInputs);

  // Sort the GVNs, since we now have constants included in the \ref InputGVNs
  // we need to make sure they are in a deterministic order.
  stable_sort(InputGVNs);
}

/// Look over the inputs and map each input argument to an argument in the
/// overall function for the OutlinableRegions.  This creates a way to replace
/// the arguments of the extracted function with the arguments of the new
/// overall function.
///
/// \param [in,out] Region - The region of code to be analyzed.
/// \param [in] InputGVNs - The global value numbering of the input values
/// collected.
/// \param [in] ArgInputs - The values of the arguments to the extracted
/// function.
static void
findExtractedInputToOverallInputMapping(OutlinableRegion &Region,
                                        std::vector<unsigned> &InputGVNs,
                                        SetVector<Value *> &ArgInputs) {

  IRSimilarityCandidate &C = *Region.Candidate;
  OutlinableGroup &Group = *Region.Parent;

  // This counts the argument number in the overall function.
  unsigned TypeIndex = 0;

  // This counts the argument number in the extracted function.
  unsigned OriginalIndex = 0;

  // Find the mapping of the extracted arguments to the arguments for the
  // overall function. Since there may be extra arguments in the overall
  // function to account for the extracted constants, we have two different
  // counters as we find extracted arguments, and as we come across overall
  // arguments.
  for (unsigned InputVal : InputGVNs) {
    Optional<Value *> InputOpt = C.fromGVN(InputVal);
    assert(InputOpt.hasValue() && "Global value number not found?");
    Value *Input = InputOpt.getValue();

    if (!Group.InputTypesSet) {
      Group.ArgumentTypes.push_back(Input->getType());
      // If the input value has a swifterr attribute, make sure to mark the
      // argument in the overall function.
      if (Input->isSwiftError()) {
        assert(
            !Group.SwiftErrorArgument.hasValue() &&
            "Argument already marked with swifterr for this OutlinableGroup!");
        Group.SwiftErrorArgument = TypeIndex;
      }
    }

    // Check if we have a constant. If we do add it to the overall argument
    // number to Constant map for the region, and continue to the next input.
    if (Constant *CST = dyn_cast<Constant>(Input)) {
      Region.AggArgToConstant.insert(std::make_pair(TypeIndex, CST));
      TypeIndex++;
      continue;
    }

    // It is not a constant, we create the mapping from extracted argument list
    // to the overall argument list.
    assert(ArgInputs.count(Input) && "Input cannot be found!");

    Region.ExtractedArgToAgg.insert(std::make_pair(OriginalIndex, TypeIndex));
    Region.AggArgToExtracted.insert(std::make_pair(TypeIndex, OriginalIndex));
    OriginalIndex++;
    TypeIndex++;
  }

  // If the function type definitions for the OutlinableGroup holding the region
  // have not been set, set the length of the inputs here.  We should have the
  // same inputs for all of the different regions contained in the
  // OutlinableGroup since they are all structurally similar to one another.
  if (!Group.InputTypesSet) {
    Group.NumAggregateInputs = TypeIndex;
    Group.InputTypesSet = true;
  }

  Region.NumExtractedInputs = OriginalIndex;
}

/// Create a mapping of the output arguments for the \p Region to the output
/// arguments of the overall outlined function.
///
/// \param [in,out] Region - The region of code to be analyzed.
/// \param [in] Outputs - The values found by the code extractor.
static void
findExtractedOutputToOverallOutputMapping(OutlinableRegion &Region,
                                          ArrayRef<Value *> Outputs) {
  OutlinableGroup &Group = *Region.Parent;
  IRSimilarityCandidate &C = *Region.Candidate;

  // This counts the argument number in the extracted function.
  unsigned OriginalIndex = Region.NumExtractedInputs;

  // This counts the argument number in the overall function.
  unsigned TypeIndex = Group.NumAggregateInputs;
  bool TypeFound;
  DenseSet<unsigned> AggArgsUsed;

  // Iterate over the output types and identify if there is an aggregate pointer
  // type whose base type matches the current output type. If there is, we mark
  // that we will use this output register for this value. If not we add another
  // type to the overall argument type list. We also store the GVNs used for
  // stores to identify which values will need to be moved into an special
  // block that holds the stores to the output registers.
  for (Value *Output : Outputs) {
    TypeFound = false;
    // We can do this since it is a result value, and will have a number
    // that is necessarily the same. BUT if in the future, the instructions
    // do not have to be in same order, but are functionally the same, we will
    // have to use a different scheme, as one-to-one correspondence is not
    // guaranteed.
    unsigned GlobalValue = C.getGVN(Output).getValue();
    unsigned ArgumentSize = Group.ArgumentTypes.size();

    for (unsigned Jdx = TypeIndex; Jdx < ArgumentSize; Jdx++) {
      if (Group.ArgumentTypes[Jdx] != PointerType::getUnqual(Output->getType()))
        continue;

      if (AggArgsUsed.contains(Jdx))
        continue;

      TypeFound = true;
      AggArgsUsed.insert(Jdx);
      Region.ExtractedArgToAgg.insert(std::make_pair(OriginalIndex, Jdx));
      Region.AggArgToExtracted.insert(std::make_pair(Jdx, OriginalIndex));
      Region.GVNStores.push_back(GlobalValue);
      break;
    }

    // We were unable to find an unused type in the output type set that matches
    // the output, so we add a pointer type to the argument types of the overall
    // function to handle this output and create a mapping to it.
    if (!TypeFound) {
      Group.ArgumentTypes.push_back(PointerType::getUnqual(Output->getType()));
      AggArgsUsed.insert(Group.ArgumentTypes.size() - 1);
      Region.ExtractedArgToAgg.insert(
          std::make_pair(OriginalIndex, Group.ArgumentTypes.size() - 1));
      Region.AggArgToExtracted.insert(
          std::make_pair(Group.ArgumentTypes.size() - 1, OriginalIndex));
      Region.GVNStores.push_back(GlobalValue);
    }

    stable_sort(Region.GVNStores);
    OriginalIndex++;
    TypeIndex++;
  }
}

void IROutliner::findAddInputsOutputs(Module &M, OutlinableRegion &Region,
                                      DenseSet<unsigned> &NotSame) {
  std::vector<unsigned> Inputs;
  SetVector<Value *> ArgInputs, Outputs;

  getCodeExtractorArguments(Region, Inputs, NotSame, OutputMappings, ArgInputs,
                            Outputs);

  if (Region.IgnoreRegion)
    return;

  // Map the inputs found by the CodeExtractor to the arguments found for
  // the overall function.
  findExtractedInputToOverallInputMapping(Region, Inputs, ArgInputs);

  // Map the outputs found by the CodeExtractor to the arguments found for
  // the overall function.
  findExtractedOutputToOverallOutputMapping(Region, Outputs.getArrayRef());
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
  DenseMap<unsigned, unsigned>::iterator ArgPair;

  OutlinableGroup &Group = *Region.Parent;
  CallInst *Call = Region.Call;
  assert(Call && "Call to replace is nullptr?");
  Function *AggFunc = Group.OutlinedFunction;
  assert(AggFunc && "Function to replace with is nullptr?");

  // If the arguments are the same size, there are not values that need to be
  // made argument, or different output registers to handle.  We can simply
  // replace the called function in this case.
  if (AggFunc->arg_size() == Call->arg_size()) {
    LLVM_DEBUG(dbgs() << "Replace call to " << *Call << " with call to "
                      << *AggFunc << " with same number of arguments\n");
    Call->setCalledFunction(AggFunc);
    return Call;
  }

  // We have a different number of arguments than the new function, so
  // we need to use our previously mappings off extracted argument to overall
  // function argument, and constants to overall function argument to create the
  // new argument list.
  for (unsigned AggArgIdx = 0; AggArgIdx < AggFunc->arg_size(); AggArgIdx++) {

    if (AggArgIdx == AggFunc->arg_size() - 1 &&
        Group.OutputGVNCombinations.size() > 1) {
      // If we are on the last argument, and we need to differentiate between
      // output blocks, add an integer to the argument list to determine
      // what block to take
      LLVM_DEBUG(dbgs() << "Set switch block argument to "
                        << Region.OutputBlockNum << "\n");
      NewCallArgs.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                                             Region.OutputBlockNum));
      continue;
    }

    ArgPair = Region.AggArgToExtracted.find(AggArgIdx);
    if (ArgPair != Region.AggArgToExtracted.end()) {
      Value *ArgumentValue = Call->getArgOperand(ArgPair->second);
      // If we found the mapping from the extracted function to the overall
      // function, we simply add it to the argument list.  We use the same
      // value, it just needs to honor the new order of arguments.
      LLVM_DEBUG(dbgs() << "Setting argument " << AggArgIdx << " to value "
                        << *ArgumentValue << "\n");
      NewCallArgs.push_back(ArgumentValue);
      continue;
    }

    // If it is a constant, we simply add it to the argument list as a value.
    if (Region.AggArgToConstant.find(AggArgIdx) !=
        Region.AggArgToConstant.end()) {
      Constant *CST = Region.AggArgToConstant.find(AggArgIdx)->second;
      LLVM_DEBUG(dbgs() << "Setting argument " << AggArgIdx << " to value "
                        << *CST << "\n");
      NewCallArgs.push_back(CST);
      continue;
    }

    // Add a nullptr value if the argument is not found in the extracted
    // function.  If we cannot find a value, it means it is not in use
    // for the region, so we should not pass anything to it.
    LLVM_DEBUG(dbgs() << "Setting argument " << AggArgIdx << " to nullptr\n");
    NewCallArgs.push_back(ConstantPointerNull::get(
        static_cast<PointerType *>(AggFunc->getArg(AggArgIdx)->getType())));
  }

  LLVM_DEBUG(dbgs() << "Replace call to " << *Call << " with call to "
                    << *AggFunc << " with new set of arguments\n");
  // Create the new call instruction and erase the old one.
  Call = CallInst::Create(AggFunc->getFunctionType(), AggFunc, NewCallArgs, "",
                          Call);

  // It is possible that the call to the outlined function is either the first
  // instruction is in the new block, the last instruction, or both.  If either
  // of these is the case, we need to make sure that we replace the instruction
  // in the IRInstructionData struct with the new call.
  CallInst *OldCall = Region.Call;
  if (Region.NewFront->Inst == OldCall)
    Region.NewFront->Inst = Call;
  if (Region.NewBack->Inst == OldCall)
    Region.NewBack->Inst = Call;

  // Transfer any debug information.
  Call->setDebugLoc(Region.Call->getDebugLoc());

  // Remove the old instruction.
  OldCall->eraseFromParent();
  Region.Call = Call;

  // Make sure that the argument in the new function has the SwiftError
  // argument.
  if (Group.SwiftErrorArgument.hasValue())
    Call->addParamAttr(Group.SwiftErrorArgument.getValue(),
                       Attribute::SwiftError);

  return Call;
}

// Within an extracted function, replace the argument uses of the extracted
// region with the arguments of the function for an OutlinableGroup.
//
/// \param [in] Region - The region of extracted code to be changed.
/// \param [in,out] OutputBB - The BasicBlock for the output stores for this
/// region.
static void replaceArgumentUses(OutlinableRegion &Region,
                                BasicBlock *OutputBB) {
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
    if (ArgIdx < Region.NumExtractedInputs) {
      LLVM_DEBUG(dbgs() << "Replacing uses of input " << *Arg << " in function "
                        << *Region.ExtractedFunction << " with " << *AggArg
                        << " in function " << *Group.OutlinedFunction << "\n");
      Arg->replaceAllUsesWith(AggArg);
      continue;
    }

    // If we are replacing an output, we place the store value in its own
    // block inside the overall function before replacing the use of the output
    // in the function.
    assert(Arg->hasOneUse() && "Output argument can only have one use");
    User *InstAsUser = Arg->user_back();
    assert(InstAsUser && "User is nullptr!");

    Instruction *I = cast<Instruction>(InstAsUser);
    I->setDebugLoc(DebugLoc());
    LLVM_DEBUG(dbgs() << "Move store for instruction " << *I << " to "
                      << *OutputBB << "\n");

    I->moveBefore(*OutputBB, OutputBB->end());

    LLVM_DEBUG(dbgs() << "Replacing uses of output " << *Arg << " in function "
                      << *Region.ExtractedFunction << " with " << *AggArg
                      << " in function " << *Group.OutlinedFunction << "\n");
    Arg->replaceAllUsesWith(AggArg);
  }
}

/// Within an extracted function, replace the constants that need to be lifted
/// into arguments with the actual argument.
///
/// \param Region [in] - The region of extracted code to be changed.
void replaceConstants(OutlinableRegion &Region) {
  OutlinableGroup &Group = *Region.Parent;
  // Iterate over the constants that need to be elevated into arguments
  for (std::pair<unsigned, Constant *> &Const : Region.AggArgToConstant) {
    unsigned AggArgIdx = Const.first;
    Function *OutlinedFunction = Group.OutlinedFunction;
    assert(OutlinedFunction && "Overall Function is not defined?");
    Constant *CST = Const.second;
    Argument *Arg = Group.OutlinedFunction->getArg(AggArgIdx);
    // Identify the argument it will be elevated to, and replace instances of
    // that constant in the function.

    // TODO: If in the future constants do not have one global value number,
    // i.e. a constant 1 could be mapped to several values, this check will
    // have to be more strict.  It cannot be using only replaceUsesWithIf.

    LLVM_DEBUG(dbgs() << "Replacing uses of constant " << *CST
                      << " in function " << *OutlinedFunction << " with "
                      << *Arg << "\n");
    CST->replaceUsesWithIf(Arg, [OutlinedFunction](Use &U) {
      if (Instruction *I = dyn_cast<Instruction>(U.getUser()))
        return I->getFunction() == OutlinedFunction;
      return false;
    });
  }
}

/// For the given function, find all the nondebug or lifetime instructions,
/// and return them as a vector. Exclude any blocks in \p ExludeBlocks.
///
/// \param [in] F - The function we collect the instructions from.
/// \param [in] ExcludeBlocks - BasicBlocks to ignore.
/// \returns the list of instructions extracted.
static std::vector<Instruction *>
collectRelevantInstructions(Function &F,
                            DenseSet<BasicBlock *> &ExcludeBlocks) {
  std::vector<Instruction *> RelevantInstructions;

  for (BasicBlock &BB : F) {
    if (ExcludeBlocks.contains(&BB))
      continue;

    for (Instruction &Inst : BB) {
      if (Inst.isLifetimeStartOrEnd())
        continue;
      if (isa<DbgInfoIntrinsic>(Inst))
        continue;

      RelevantInstructions.push_back(&Inst);
    }
  }

  return RelevantInstructions;
}

/// It is possible that there is a basic block that already performs the same
/// stores. This returns a duplicate block, if it exists
///
/// \param OutputBB [in] the block we are looking for a duplicate of.
/// \param OutputStoreBBs [in] The existing output blocks.
/// \returns an optional value with the number output block if there is a match.
Optional<unsigned>
findDuplicateOutputBlock(BasicBlock *OutputBB,
                         ArrayRef<BasicBlock *> OutputStoreBBs) {

  bool WrongInst = false;
  bool WrongSize = false;
  unsigned MatchingNum = 0;
  for (BasicBlock *CompBB : OutputStoreBBs) {
    WrongInst = false;
    if (CompBB->size() - 1 != OutputBB->size()) {
      WrongSize = true;
      MatchingNum++;
      continue;
    }

    WrongSize = false;
    BasicBlock::iterator NIt = OutputBB->begin();
    for (Instruction &I : *CompBB) {
      if (isa<BranchInst>(&I))
        continue;

      if (!I.isIdenticalTo(&(*NIt))) {
        WrongInst = true;
        break;
      }

      NIt++;
    }
    if (!WrongInst && !WrongSize)
      return MatchingNum;

    MatchingNum++;
  }

  return None;
}

/// For the outlined section, move needed the StoreInsts for the output
/// registers into their own block. Then, determine if there is a duplicate
/// output block already created.
///
/// \param [in] OG - The OutlinableGroup of regions to be outlined.
/// \param [in] Region - The OutlinableRegion that is being analyzed.
/// \param [in,out] OutputBB - the block that stores for this region will be
/// placed in.
/// \param [in] EndBB - the final block of the extracted function.
/// \param [in] OutputMappings - OutputMappings the mapping of values that have
/// been replaced by a new output value.
/// \param [in,out] OutputStoreBBs - The existing output blocks.
static void
alignOutputBlockWithAggFunc(OutlinableGroup &OG, OutlinableRegion &Region,
                            BasicBlock *OutputBB, BasicBlock *EndBB,
                            const DenseMap<Value *, Value *> &OutputMappings,
                            std::vector<BasicBlock *> &OutputStoreBBs) {
  DenseSet<unsigned> ValuesToFind(Region.GVNStores.begin(),
                                  Region.GVNStores.end());

  // We iterate over the instructions in the extracted function, and find the
  // global value number of the instructions.  If we find a value that should
  // be contained in a store, we replace the uses of the value with the value
  // from the overall function, so that the store is storing the correct
  // value from the overall function.
  DenseSet<BasicBlock *> ExcludeBBs(OutputStoreBBs.begin(),
                                    OutputStoreBBs.end());
  ExcludeBBs.insert(OutputBB);
  std::vector<Instruction *> ExtractedFunctionInsts =
      collectRelevantInstructions(*(Region.ExtractedFunction), ExcludeBBs);
  std::vector<Instruction *> OverallFunctionInsts =
      collectRelevantInstructions(*OG.OutlinedFunction, ExcludeBBs);

  assert(ExtractedFunctionInsts.size() == OverallFunctionInsts.size() &&
         "Number of relevant instructions not equal!");

  unsigned NumInstructions = ExtractedFunctionInsts.size();
  for (unsigned Idx = 0; Idx < NumInstructions; Idx++) {
    Value *V = ExtractedFunctionInsts[Idx];

    if (OutputMappings.find(V) != OutputMappings.end())
      V = OutputMappings.find(V)->second;
    Optional<unsigned> GVN = Region.Candidate->getGVN(V);

    // If we have found one of the stored values for output, replace the value
    // with the corresponding one from the overall function.
    if (GVN.hasValue() && ValuesToFind.erase(GVN.getValue())) {
      V->replaceAllUsesWith(OverallFunctionInsts[Idx]);
      if (ValuesToFind.size() == 0)
        break;
    }

    if (ValuesToFind.size() == 0)
      break;
  }

  assert(ValuesToFind.size() == 0 && "Not all store values were handled!");

  // If the size of the block is 0, then there are no stores, and we do not
  // need to save this block.
  if (OutputBB->size() == 0) {
    Region.OutputBlockNum = -1;
    OutputBB->eraseFromParent();
    return;
  }

  // Determine is there is a duplicate block.
  Optional<unsigned> MatchingBB =
      findDuplicateOutputBlock(OutputBB, OutputStoreBBs);

  // If there is, we remove the new output block.  If it does not,
  // we add it to our list of output blocks.
  if (MatchingBB.hasValue()) {
    LLVM_DEBUG(dbgs() << "Set output block for region in function"
                      << Region.ExtractedFunction << " to "
                      << MatchingBB.getValue());

    Region.OutputBlockNum = MatchingBB.getValue();
    OutputBB->eraseFromParent();
    return;
  }

  Region.OutputBlockNum = OutputStoreBBs.size();

  LLVM_DEBUG(dbgs() << "Create output block for region in"
                    << Region.ExtractedFunction << " to "
                    << *OutputBB);
  OutputStoreBBs.push_back(OutputBB);
  BranchInst::Create(EndBB, OutputBB);
}

/// Create the switch statement for outlined function to differentiate between
/// all the output blocks.
///
/// For the outlined section, determine if an outlined block already exists that
/// matches the needed stores for the extracted section.
/// \param [in] M - The module we are outlining from.
/// \param [in] OG - The group of regions to be outlined.
/// \param [in] EndBB - The final block of the extracted function.
/// \param [in,out] OutputStoreBBs - The existing output blocks.
void createSwitchStatement(Module &M, OutlinableGroup &OG, BasicBlock *EndBB,
                           ArrayRef<BasicBlock *> OutputStoreBBs) {
  // We only need the switch statement if there is more than one store
  // combination.
  if (OG.OutputGVNCombinations.size() > 1) {
    Function *AggFunc = OG.OutlinedFunction;
    // Create a final block
    BasicBlock *ReturnBlock =
        BasicBlock::Create(M.getContext(), "final_block", AggFunc);
    Instruction *Term = EndBB->getTerminator();
    Term->moveBefore(*ReturnBlock, ReturnBlock->end());
    // Put the switch statement in the old end basic block for the function with
    // a fall through to the new return block
    LLVM_DEBUG(dbgs() << "Create switch statement in " << *AggFunc << " for "
                      << OutputStoreBBs.size() << "\n");
    SwitchInst *SwitchI =
        SwitchInst::Create(AggFunc->getArg(AggFunc->arg_size() - 1),
                           ReturnBlock, OutputStoreBBs.size(), EndBB);

    unsigned Idx = 0;
    for (BasicBlock *BB : OutputStoreBBs) {
      SwitchI->addCase(ConstantInt::get(Type::getInt32Ty(M.getContext()), Idx),
                       BB);
      Term = BB->getTerminator();
      Term->setSuccessor(0, ReturnBlock);
      Idx++;
    }
    return;
  }

  // If there needs to be stores, move them from the output block to the end
  // block to save on branching instructions.
  if (OutputStoreBBs.size() == 1) {
    LLVM_DEBUG(dbgs() << "Move store instructions to the end block in "
                      << *OG.OutlinedFunction << "\n");
    BasicBlock *OutputBlock = OutputStoreBBs[0];
    Instruction *Term = OutputBlock->getTerminator();
    Term->eraseFromParent();
    Term = EndBB->getTerminator();
    moveBBContents(*OutputBlock, *EndBB);
    Term->moveBefore(*EndBB, EndBB->end());
    OutputBlock->eraseFromParent();
  }
}

/// Fill the new function that will serve as the replacement function for all of
/// the extracted regions of a certain structure from the first region in the
/// list of regions.  Replace this first region's extracted function with the
/// new overall function.
///
/// \param [in] M - The module we are outlining from.
/// \param [in] CurrentGroup - The group of regions to be outlined.
/// \param [in,out] OutputStoreBBs - The output blocks for each different
/// set of stores needed for the different functions.
/// \param [in,out] FuncsToRemove - Extracted functions to erase from module
/// once outlining is complete.
static void fillOverallFunction(Module &M, OutlinableGroup &CurrentGroup,
                                std::vector<BasicBlock *> &OutputStoreBBs,
                                std::vector<Function *> &FuncsToRemove) {
  OutlinableRegion *CurrentOS = CurrentGroup.Regions[0];

  // Move first extracted function's instructions into new function.
  LLVM_DEBUG(dbgs() << "Move instructions from "
                    << *CurrentOS->ExtractedFunction << " to instruction "
                    << *CurrentGroup.OutlinedFunction << "\n");

  CurrentGroup.EndBB = moveFunctionData(*CurrentOS->ExtractedFunction,
                                        *CurrentGroup.OutlinedFunction);

  // Transfer the attributes from the function to the new function.
  for (Attribute A :
       CurrentOS->ExtractedFunction->getAttributes().getFnAttributes())
    CurrentGroup.OutlinedFunction->addFnAttr(A);

  // Create an output block for the first extracted function.
  BasicBlock *NewBB = BasicBlock::Create(
      M.getContext(), Twine("output_block_") + Twine(static_cast<unsigned>(0)),
      CurrentGroup.OutlinedFunction);
  CurrentOS->OutputBlockNum = 0;

  replaceArgumentUses(*CurrentOS, NewBB);
  replaceConstants(*CurrentOS);

  // If the new basic block has no new stores, we can erase it from the module.
  // It it does, we create a branch instruction to the last basic block from the
  // new one.
  if (NewBB->size() == 0) {
    CurrentOS->OutputBlockNum = -1;
    NewBB->eraseFromParent();
  } else {
    BranchInst::Create(CurrentGroup.EndBB, NewBB);
    OutputStoreBBs.push_back(NewBB);
  }

  // Replace the call to the extracted function with the outlined function.
  CurrentOS->Call = replaceCalledFunction(M, *CurrentOS);

  // We only delete the extracted functions at the end since we may need to
  // reference instructions contained in them for mapping purposes.
  FuncsToRemove.push_back(CurrentOS->ExtractedFunction);
}

void IROutliner::deduplicateExtractedSections(
    Module &M, OutlinableGroup &CurrentGroup,
    std::vector<Function *> &FuncsToRemove, unsigned &OutlinedFunctionNum) {
  createFunction(M, CurrentGroup, OutlinedFunctionNum);

  std::vector<BasicBlock *> OutputStoreBBs;

  OutlinableRegion *CurrentOS;

  fillOverallFunction(M, CurrentGroup, OutputStoreBBs, FuncsToRemove);

  for (unsigned Idx = 1; Idx < CurrentGroup.Regions.size(); Idx++) {
    CurrentOS = CurrentGroup.Regions[Idx];
    AttributeFuncs::mergeAttributesForOutlining(*CurrentGroup.OutlinedFunction,
                                               *CurrentOS->ExtractedFunction);

    // Create a new BasicBlock to hold the needed store instructions.
    BasicBlock *NewBB = BasicBlock::Create(
        M.getContext(), "output_block_" + std::to_string(Idx),
        CurrentGroup.OutlinedFunction);
    replaceArgumentUses(*CurrentOS, NewBB);

    alignOutputBlockWithAggFunc(CurrentGroup, *CurrentOS, NewBB,
                                CurrentGroup.EndBB, OutputMappings,
                                OutputStoreBBs);

    CurrentOS->Call = replaceCalledFunction(M, *CurrentOS);
    FuncsToRemove.push_back(CurrentOS->ExtractedFunction);
  }

  // Create a switch statement to handle the different output schemes.
  createSwitchStatement(M, CurrentGroup, CurrentGroup.EndBB, OutputStoreBBs);

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

    if (IRSC.front()->Inst->getFunction()->hasLinkOnceODRLinkage() &&
        !OutlineFromLinkODRs)
      continue;

    // Greedily prune out any regions that will overlap with already chosen
    // regions.
    if (CurrentEndIdx != 0 && StartIdx <= CurrentEndIdx)
      continue;

    bool BadInst = any_of(IRSC, [this](IRInstructionData &ID) {
      // We check if there is a discrepancy between the InstructionDataList
      // and the actual next instruction in the module.  If there is, it means
      // that an extra instruction was added, likely by the CodeExtractor.

      // Since we do not have any similarity data about this particular
      // instruction, we cannot confidently outline it, and must discard this
      // candidate.
      if (std::next(ID.getIterator())->Inst !=
          ID.Inst->getNextNonDebugInstruction())
        return true;
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

InstructionCost
IROutliner::findBenefitFromAllRegions(OutlinableGroup &CurrentGroup) {
  InstructionCost RegionBenefit = 0;
  for (OutlinableRegion *Region : CurrentGroup.Regions) {
    TargetTransformInfo &TTI = getTTI(*Region->StartBB->getParent());
    // We add the number of instructions in the region to the benefit as an
    // estimate as to how much will be removed.
    RegionBenefit += Region->getBenefit(TTI);
    LLVM_DEBUG(dbgs() << "Adding: " << RegionBenefit
                      << " saved instructions to overfall benefit.\n");
  }

  return RegionBenefit;
}

InstructionCost
IROutliner::findCostOutputReloads(OutlinableGroup &CurrentGroup) {
  InstructionCost OverallCost = 0;
  for (OutlinableRegion *Region : CurrentGroup.Regions) {
    TargetTransformInfo &TTI = getTTI(*Region->StartBB->getParent());

    // Each output incurs a load after the call, so we add that to the cost.
    for (unsigned OutputGVN : Region->GVNStores) {
      Optional<Value *> OV = Region->Candidate->fromGVN(OutputGVN);
      assert(OV.hasValue() && "Could not find value for GVN?");
      Value *V = OV.getValue();
      InstructionCost LoadCost =
          TTI.getMemoryOpCost(Instruction::Load, V->getType(), Align(1), 0,
                              TargetTransformInfo::TCK_CodeSize);

      LLVM_DEBUG(dbgs() << "Adding: " << LoadCost
                        << " instructions to cost for output of type "
                        << *V->getType() << "\n");
      OverallCost += LoadCost;
    }
  }

  return OverallCost;
}

/// Find the extra instructions needed to handle any output values for the
/// region.
///
/// \param [in] M - The Module to outline from.
/// \param [in] CurrentGroup - The collection of OutlinableRegions to analyze.
/// \param [in] TTI - The TargetTransformInfo used to collect information for
/// new instruction costs.
/// \returns the additional cost to handle the outputs.
static InstructionCost findCostForOutputBlocks(Module &M,
                                               OutlinableGroup &CurrentGroup,
                                               TargetTransformInfo &TTI) {
  InstructionCost OutputCost = 0;

  for (const ArrayRef<unsigned> &OutputUse :
       CurrentGroup.OutputGVNCombinations) {
    IRSimilarityCandidate &Candidate = *CurrentGroup.Regions[0]->Candidate;
    for (unsigned GVN : OutputUse) {
      Optional<Value *> OV = Candidate.fromGVN(GVN);
      assert(OV.hasValue() && "Could not find value for GVN?");
      Value *V = OV.getValue();
      InstructionCost StoreCost =
          TTI.getMemoryOpCost(Instruction::Load, V->getType(), Align(1), 0,
                              TargetTransformInfo::TCK_CodeSize);

      // An instruction cost is added for each store set that needs to occur for
      // various output combinations inside the function, plus a branch to
      // return to the exit block.
      LLVM_DEBUG(dbgs() << "Adding: " << StoreCost
                        << " instructions to cost for output of type "
                        << *V->getType() << "\n");
      OutputCost += StoreCost;
    }

    InstructionCost BranchCost =
        TTI.getCFInstrCost(Instruction::Br, TargetTransformInfo::TCK_CodeSize);
    LLVM_DEBUG(dbgs() << "Adding " << BranchCost << " to the current cost for"
                      << " a branch instruction\n");
    OutputCost += BranchCost;
  }

  // If there is more than one output scheme, we must have a comparison and
  // branch for each different item in the switch statement.
  if (CurrentGroup.OutputGVNCombinations.size() > 1) {
    InstructionCost ComparisonCost = TTI.getCmpSelInstrCost(
        Instruction::ICmp, Type::getInt32Ty(M.getContext()),
        Type::getInt32Ty(M.getContext()), CmpInst::BAD_ICMP_PREDICATE,
        TargetTransformInfo::TCK_CodeSize);
    InstructionCost BranchCost =
        TTI.getCFInstrCost(Instruction::Br, TargetTransformInfo::TCK_CodeSize);

    unsigned DifferentBlocks = CurrentGroup.OutputGVNCombinations.size();
    InstructionCost TotalCost = ComparisonCost * BranchCost * DifferentBlocks;

    LLVM_DEBUG(dbgs() << "Adding: " << TotalCost
                      << " instructions for each switch case for each different"
                      << " output path in a function\n");
    OutputCost += TotalCost;
  }

  return OutputCost;
}

void IROutliner::findCostBenefit(Module &M, OutlinableGroup &CurrentGroup) {
  InstructionCost RegionBenefit = findBenefitFromAllRegions(CurrentGroup);
  CurrentGroup.Benefit += RegionBenefit;
  LLVM_DEBUG(dbgs() << "Current Benefit: " << CurrentGroup.Benefit << "\n");

  InstructionCost OutputReloadCost = findCostOutputReloads(CurrentGroup);
  CurrentGroup.Cost += OutputReloadCost;
  LLVM_DEBUG(dbgs() << "Current Cost: " << CurrentGroup.Cost << "\n");

  InstructionCost AverageRegionBenefit =
      RegionBenefit / CurrentGroup.Regions.size();
  unsigned OverallArgumentNum = CurrentGroup.ArgumentTypes.size();
  unsigned NumRegions = CurrentGroup.Regions.size();
  TargetTransformInfo &TTI =
      getTTI(*CurrentGroup.Regions[0]->Candidate->getFunction());

  // We add one region to the cost once, to account for the instructions added
  // inside of the newly created function.
  LLVM_DEBUG(dbgs() << "Adding: " << AverageRegionBenefit
                    << " instructions to cost for body of new function.\n");
  CurrentGroup.Cost += AverageRegionBenefit;
  LLVM_DEBUG(dbgs() << "Current Cost: " << CurrentGroup.Cost << "\n");

  // For each argument, we must add an instruction for loading the argument
  // out of the register and into a value inside of the newly outlined function.
  LLVM_DEBUG(dbgs() << "Adding: " << OverallArgumentNum
                    << " instructions to cost for each argument in the new"
                    << " function.\n");
  CurrentGroup.Cost +=
      OverallArgumentNum * TargetTransformInfo::TCC_Basic;
  LLVM_DEBUG(dbgs() << "Current Cost: " << CurrentGroup.Cost << "\n");

  // Each argument needs to either be loaded into a register or onto the stack.
  // Some arguments will only be loaded into the stack once the argument
  // registers are filled.
  LLVM_DEBUG(dbgs() << "Adding: " << OverallArgumentNum
                    << " instructions to cost for each argument in the new"
                    << " function " << NumRegions << " times for the "
                    << "needed argument handling at the call site.\n");
  CurrentGroup.Cost +=
      2 * OverallArgumentNum * TargetTransformInfo::TCC_Basic * NumRegions;
  LLVM_DEBUG(dbgs() << "Current Cost: " << CurrentGroup.Cost << "\n");

  CurrentGroup.Cost += findCostForOutputBlocks(M, CurrentGroup, TTI);
  LLVM_DEBUG(dbgs() << "Current Cost: " << CurrentGroup.Cost << "\n");
}

void IROutliner::updateOutputMapping(OutlinableRegion &Region,
                                     ArrayRef<Value *> Outputs,
                                     LoadInst *LI) {
  // For and load instructions following the call
  Value *Operand = LI->getPointerOperand();
  Optional<unsigned> OutputIdx = None;
  // Find if the operand it is an output register.
  for (unsigned ArgIdx = Region.NumExtractedInputs;
       ArgIdx < Region.Call->arg_size(); ArgIdx++) {
    if (Operand == Region.Call->getArgOperand(ArgIdx)) {
      OutputIdx = ArgIdx - Region.NumExtractedInputs;
      break;
    }
  }

  // If we found an output register, place a mapping of the new value
  // to the original in the mapping.
  if (!OutputIdx.hasValue())
    return;

  if (OutputMappings.find(Outputs[OutputIdx.getValue()]) ==
      OutputMappings.end()) {
    LLVM_DEBUG(dbgs() << "Mapping extracted output " << *LI << " to "
                      << *Outputs[OutputIdx.getValue()] << "\n");
    OutputMappings.insert(std::make_pair(LI, Outputs[OutputIdx.getValue()]));
  } else {
    Value *Orig = OutputMappings.find(Outputs[OutputIdx.getValue()])->second;
    LLVM_DEBUG(dbgs() << "Mapping extracted output " << *Orig << " to "
                      << *Outputs[OutputIdx.getValue()] << "\n");
    OutputMappings.insert(std::make_pair(LI, Orig));
  }
}

bool IROutliner::extractSection(OutlinableRegion &Region) {
  SetVector<Value *> ArgInputs, Outputs, SinkCands;
  Region.CE->findInputsOutputs(ArgInputs, Outputs, SinkCands);

  assert(Region.StartBB && "StartBB for the OutlinableRegion is nullptr!");
  assert(Region.FollowBB && "FollowBB for the OutlinableRegion is nullptr!");
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
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      if (Region.ExtractedFunction == CI->getCalledFunction())
        Region.Call = CI;
    } else if (LoadInst *LI = dyn_cast<LoadInst>(&I))
      updateOutputMapping(Region, Outputs.getArrayRef(), LI);
  Region.reattachCandidate();
  return true;
}

unsigned IROutliner::doOutline(Module &M) {
  // Find the possible similarity sections.
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
      findAddInputsOutputs(M, *OS, NotSame);
      if (!OS->IgnoreRegion)
        OutlinedRegions.push_back(OS);
      else
        OS->reattachCandidate();
    }

    CurrentGroup.Regions = std::move(OutlinedRegions);

    if (CurrentGroup.Regions.empty())
      continue;

    CurrentGroup.collectGVNStoreSets(M);

    if (CostModel)
      findCostBenefit(M, CurrentGroup);

    // If we are adhering to the cost model, reattach all the candidates
    if (CurrentGroup.Cost >= CurrentGroup.Benefit && CostModel) {
      for (OutlinableRegion *OS : CurrentGroup.Regions)
        OS->reattachCandidate();
      OptimizationRemarkEmitter &ORE = getORE(
          *CurrentGroup.Regions[0]->Candidate->getFunction());
      ORE.emit([&]() {
        IRSimilarityCandidate *C = CurrentGroup.Regions[0]->Candidate;
        OptimizationRemarkMissed R(DEBUG_TYPE, "WouldNotDecreaseSize",
                                   C->frontInstruction());
        R << "did not outline "
          << ore::NV(std::to_string(CurrentGroup.Regions.size()))
          << " regions due to estimated increase of "
          << ore::NV("InstructionIncrease",
                     CurrentGroup.Cost - CurrentGroup.Benefit)
          << " instructions at locations ";
        interleave(
            CurrentGroup.Regions.begin(), CurrentGroup.Regions.end(),
            [&R](OutlinableRegion *Region) {
              R << ore::NV(
                  "DebugLoc",
                  Region->Candidate->frontInstruction()->getDebugLoc());
            },
            [&R]() { R << " "; });
        return R;
      });
      continue;
    }

    LLVM_DEBUG(dbgs() << "Outlining regions with cost " << CurrentGroup.Cost
                      << " and benefit " << CurrentGroup.Benefit << "\n");

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

    LLVM_DEBUG(dbgs() << "Outlined " << OutlinedRegions.size()
                      << " with benefit " << CurrentGroup.Benefit
                      << " and cost " << CurrentGroup.Cost << "\n");

    CurrentGroup.Regions = std::move(OutlinedRegions);

    if (CurrentGroup.Regions.empty())
      continue;

    OptimizationRemarkEmitter &ORE =
        getORE(*CurrentGroup.Regions[0]->Call->getFunction());
    ORE.emit([&]() {
      IRSimilarityCandidate *C = CurrentGroup.Regions[0]->Candidate;
      OptimizationRemark R(DEBUG_TYPE, "Outlined", C->front()->Inst);
      R << "outlined " << ore::NV(std::to_string(CurrentGroup.Regions.size()))
        << " regions with decrease of "
        << ore::NV("Benefit", CurrentGroup.Benefit - CurrentGroup.Cost)
        << " instructions at locations ";
      interleave(
          CurrentGroup.Regions.begin(), CurrentGroup.Regions.end(),
          [&R](OutlinableRegion *Region) {
            R << ore::NV("DebugLoc",
                         Region->Candidate->frontInstruction()->getDebugLoc());
          },
          [&R]() { R << " "; });
      return R;
    });

    deduplicateExtractedSections(M, CurrentGroup, FuncsToRemove,
                                 OutlinedFunctionNum);
  }

  for (Function *F : FuncsToRemove)
    F->eraseFromParent();

  return OutlinedFunctionNum;
}

bool IROutliner::run(Module &M) {
  CostModel = !NoCostModel;
  OutlineFromLinkODRs = EnableLinkOnceODRIROutlining;

  return doOutline(M) > 0;
}

// Pass Manager Boilerplate
class IROutlinerLegacyPass : public ModulePass {
public:
  static char ID;
  IROutlinerLegacyPass() : ModulePass(ID) {
    initializeIROutlinerLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<IRSimilarityIdentifierWrapperPass>();
  }

  bool runOnModule(Module &M) override;
};

bool IROutlinerLegacyPass::runOnModule(Module &M) {
  if (skipModule(M))
    return false;

  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  auto GORE = [&ORE](Function &F) -> OptimizationRemarkEmitter & {
    ORE.reset(new OptimizationRemarkEmitter(&F));
    return *ORE.get();
  };

  auto GTTI = [this](Function &F) -> TargetTransformInfo & {
    return this->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  };

  auto GIRSI = [this](Module &) -> IRSimilarityIdentifier & {
    return this->getAnalysis<IRSimilarityIdentifierWrapperPass>().getIRSI();
  };

  return IROutliner(GTTI, GIRSI, GORE).run(M);
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

  std::unique_ptr<OptimizationRemarkEmitter> ORE;
  std::function<OptimizationRemarkEmitter &(Function &)> GORE =
      [&ORE](Function &F) -> OptimizationRemarkEmitter & {
    ORE.reset(new OptimizationRemarkEmitter(&F));
    return *ORE.get();
  };

  if (IROutliner(GTTI, GIRSI, GORE).run(M))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

char IROutlinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(IROutlinerLegacyPass, "iroutliner", "IR Outliner", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(IRSimilarityIdentifierWrapperPass)
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(IROutlinerLegacyPass, "iroutliner", "IR Outliner", false,
                    false)

ModulePass *llvm::createIROutlinerPass() { return new IROutlinerLegacyPass(); }
