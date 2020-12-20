//===- IROutliner.h - Extract similar IR regions into functions ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// The interface file for the IROutliner which is used by the IROutliner Pass.
//
// The outliner uses the IRSimilarityIdentifier to identify the similar regions
// of code.  It evaluates each set of IRSimilarityCandidates with an estimate of
// whether it will provide code size reduction.  Each region is extracted using
// the code extractor.  These extracted functions are consolidated into a single
// function and called from the extracted call site.
//
// For example:
// \code
//   %1 = add i32 %a, %b
//   %2 = add i32 %b, %a
//   %3 = add i32 %b, %a
//   %4 = add i32 %a, %b
// \endcode
// would become function
// \code
// define internal void outlined_ir_function(i32 %0, i32 %1) {
//   %1 = add i32 %0, %1
//   %2 = add i32 %1, %0
//   ret void
// }
// \endcode
// with calls:
// \code
//   call void outlined_ir_function(i32 %a, i32 %b)
//   call void outlined_ir_function(i32 %b, i32 %a)
// \endcode
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_IROUTLINER_H
#define LLVM_TRANSFORMS_IPO_IROUTLINER_H

#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include <set>

struct OutlinableGroup;

namespace llvm {
using namespace IRSimilarity;

class Module;
class TargetTransformInfo;
class OptimizationRemarkEmitter;

/// The OutlinableRegion holds all the information for a specific region, or
/// sequence of instructions. This includes what values need to be hoisted to
/// arguments from the extracted function, inputs and outputs to the region, and
/// mapping from the extracted function arguments to overall function arguments.
struct OutlinableRegion {
  /// Describes the region of code.
  IRSimilarityCandidate *Candidate;

  /// If this region is outlined, the front and back IRInstructionData could
  /// potentially become invalidated if the only new instruction is a call.
  /// This ensures that we replace in the instruction in the IRInstructionData.
  IRInstructionData *NewFront = nullptr;
  IRInstructionData *NewBack = nullptr;

  /// The number of extracted inputs from the CodeExtractor.
  unsigned NumExtractedInputs;

  /// Mapping the extracted argument number to the argument number in the
  /// overall function.  Since there will be inputs, such as elevated constants
  /// that are not the same in each region in a SimilarityGroup, or values that
  /// cannot be sunk into the extracted section in every region, we must keep
  /// track of which extracted argument maps to which overall argument.
  DenseMap<unsigned, unsigned> ExtractedArgToAgg;
  DenseMap<unsigned, unsigned> AggArgToExtracted;

  /// Used to create an outlined function.
  CodeExtractor *CE = nullptr;

  /// The call site of the extracted region.
  CallInst *Call = nullptr;

  /// The function for the extracted region.
  Function *ExtractedFunction = nullptr;

  /// Flag for whether we have split out the IRSimilarityCanidate. That is,
  /// make the region contained the IRSimilarityCandidate its own BasicBlock.
  bool CandidateSplit = false;

  /// Flag for whether we should not consider this region for extraction.
  bool IgnoreRegion = false;

  /// The BasicBlock that is before the start of the region BasicBlock,
  /// only defined when the region has been split.
  BasicBlock *PrevBB = nullptr;

  /// The BasicBlock that contains the starting instruction of the region.
  BasicBlock *StartBB = nullptr;

  /// The BasicBlock that contains the ending instruction of the region.
  BasicBlock *EndBB = nullptr;

  /// The BasicBlock that is after the start of the region BasicBlock,
  /// only defined when the region has been split.
  BasicBlock *FollowBB = nullptr;

  /// The Outlinable Group that contains this region and structurally similar
  /// regions to this region.
  OutlinableGroup *Parent = nullptr;

  OutlinableRegion(IRSimilarityCandidate &C, OutlinableGroup &Group)
      : Candidate(&C), Parent(&Group) {
    StartBB = C.getStartBB();
    EndBB = C.getEndBB();
  }

  /// For the contained region, split the parent BasicBlock at the starting and
  /// ending instructions of the contained IRSimilarityCandidate.
  void splitCandidate();

  /// For the contained region, reattach the BasicBlock at the starting and
  /// ending instructions of the contained IRSimilarityCandidate, or if the
  /// function has been extracted, the start and end of the BasicBlock
  /// containing the called function.
  void reattachCandidate();
};

/// This class is a pass that identifies similarity in a Module, extracts
/// instances of the similarity, and then consolidating the similar regions
/// in an effort to reduce code size.  It uses the IRSimilarityIdentifier pass
/// to identify the similar regions of code, and then extracts the similar
/// sections into a single function.  See the above for an example as to
/// how code is extracted and consolidated into a single function.
class IROutliner {
public:
  IROutliner(function_ref<TargetTransformInfo &(Function &)> GTTI,
             function_ref<IRSimilarityIdentifier &(Module &)> GIRSI)
      : getTTI(GTTI), getIRSI(GIRSI) {}
  bool run(Module &M);

private:
  /// Find repeated similar code sequences in \p M and outline them into new
  /// Functions.
  ///
  /// \param [in] M - The module to outline from.
  /// \returns The number of Functions created.
  unsigned doOutline(Module &M);

  /// Remove all the IRSimilarityCandidates from \p CandidateVec that have
  /// instructions contained in a previously outlined region and put the
  /// remaining regions in \p CurrentGroup.
  ///
  /// \param [in] CandidateVec - List of similarity candidates for regions with
  /// the same similarity structure.
  /// \param [in,out] CurrentGroup - Contains the potential sections to
  /// be outlined.
  void
  pruneIncompatibleRegions(std::vector<IRSimilarityCandidate> &CandidateVec,
                           OutlinableGroup &CurrentGroup);

  /// Create the function based on the overall types found in the current
  /// regions being outlined.
  ///
  /// \param M - The module to outline from.
  /// \param [in,out] CG - The OutlinableGroup for the regions to be outlined.
  /// \param [in] FunctionNameSuffix - How many functions have we previously
  /// created.
  /// \returns the newly created function.
  Function *createFunction(Module &M, OutlinableGroup &CG,
                           unsigned FunctionNameSuffix);

  /// Identify the needed extracted inputs in a section, and add to the overall
  /// function if needed.
  ///
  /// \param [in] M - The module to outline from.
  /// \param [in,out] Region - The region to be extracted
  void findAddInputsOutputs(Module &M, OutlinableRegion &Region);

  /// Extract \p Region into its own function.
  ///
  /// \param [in] Region - The region to be extracted into its own function.
  /// \returns True if it was successfully outlined.
  bool extractSection(OutlinableRegion &Region);

  /// For the similarities found, and the extracted sections, create a single
  /// outlined function with appropriate output blocks as necessary.
  ///
  /// \param [in] M - The module to outline from
  /// \param [in] CurrentGroup - The set of extracted sections to consolidate.
  /// \param [in,out] FuncsToRemove - List of functions to remove from the
  /// module after outlining is completed.
  /// \param [in,out] OutlinedFunctionNum - the number of new outlined
  /// functions.
  void deduplicateExtractedSections(Module &M, OutlinableGroup &CurrentGroup,
                                    std::vector<Function *> &FuncsToRemove,
                                    unsigned &OutlinedFunctionNum);

  /// The set of outlined Instructions, identified by their location in the
  /// sequential ordering of instructions in a Module.
  DenseSet<unsigned> Outlined;

  /// TargetTransformInfo lambda for target specific information.
  function_ref<TargetTransformInfo &(Function &)> getTTI;

  /// IRSimilarityIdentifier lambda to retrieve IRSimilarityIdentifier.
  function_ref<IRSimilarityIdentifier &(Module &)> getIRSI;

  /// The memory allocator used to allocate the CodeExtractors.
  SpecificBumpPtrAllocator<CodeExtractor> ExtractorAllocator;

  /// The memory allocator used to allocate the OutlinableRegions.
  SpecificBumpPtrAllocator<OutlinableRegion> RegionAllocator;

  /// The memory allocator used to allocate new IRInstructionData.
  SpecificBumpPtrAllocator<IRInstructionData> InstDataAllocator;

  /// Custom InstVisitor to classify different instructions for whether it can
  /// be analyzed for similarity.  This is needed as there may be instruction we
  /// can identify as having similarity, but are more complicated to outline.
  struct InstructionAllowed : public InstVisitor<InstructionAllowed, bool> {
    InstructionAllowed() {}

    // TODO: Determine a scheme to resolve when the label is similar enough.
    bool visitBranchInst(BranchInst &BI) { return false; }
    // TODO: Determine a scheme to resolve when the labels are similar enough.
    bool visitPHINode(PHINode &PN) { return false; }
    // TODO: Handle allocas.
    bool visitAllocaInst(AllocaInst &AI) { return false; }
    // VAArg instructions are not allowed since this could cause difficulty when
    // differentiating between different sets of variable instructions in
    // the deduplicated outlined regions.
    bool visitVAArgInst(VAArgInst &VI) { return false; }
    // We exclude all exception handling cases since they are so context
    // dependent.
    bool visitLandingPadInst(LandingPadInst &LPI) { return false; }
    bool visitFuncletPadInst(FuncletPadInst &FPI) { return false; }
    // DebugInfo should be included in the regions, but should not be
    // analyzed for similarity as it has no bearing on the outcome of the
    // program.
    bool visitDbgInfoIntrinsic(DbgInfoIntrinsic &DII) { return true; }
    // TODO: Handle GetElementPtrInsts
    bool visitGetElementPtrInst(GetElementPtrInst &GEPI) { return false; }
    // TODO: Handle specific intrinsics individually from those that can be
    // handled.
    bool IntrinsicInst(IntrinsicInst &II) { return false; }
    // TODO: Handle CallInsts, there will need to be handling for special kinds
    // of calls, as well as calls to intrinsics.
    bool visitCallInst(CallInst &CI) { return false; }
    // TODO: Handle FreezeInsts.  Since a frozen value could be frozen inside
    // the outlined region, and then returned as an output, this will have to be
    // handled differently.
    bool visitFreezeInst(FreezeInst &CI) { return false; }
    // TODO: We do not current handle similarity that changes the control flow.
    bool visitInvokeInst(InvokeInst &II) { return false; }
    // TODO: We do not current handle similarity that changes the control flow.
    bool visitCallBrInst(CallBrInst &CBI) { return false; }
    // TODO: Handle interblock similarity.
    bool visitTerminator(Instruction &I) { return false; }
    bool visitInstruction(Instruction &I) { return true; }
  };

  /// A InstVisitor used to exclude certain instructions from being outlined.
  InstructionAllowed InstructionClassifier;
};

/// Pass to outline similar regions.
class IROutlinerPass : public PassInfoMixin<IROutlinerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_IROUTLINER_H
