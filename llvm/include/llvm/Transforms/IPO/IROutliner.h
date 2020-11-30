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

  /// Extract \p Region into its own function.
  ///
  /// \param [in] Region - The region to be extracted into its own function.
  /// \returns True if it was successfully outlined.
  bool extractSection(OutlinableRegion &Region);

  /// The set of outlined Instructions, identified by their location in the
  /// sequential ordering of instructions in a Module.
  DenseSet<unsigned> Outlined;

  /// TargetTransformInfo lambda for target specific information.
  function_ref<TargetTransformInfo &(Function &)> getTTI;

  /// IRSimilarityIdentifier lambda to retrieve IRSimilarityIdentifier.
  function_ref<IRSimilarityIdentifier &(Module &)> getIRSI;

  /// The memory allocator used to allocate the OutlinableRegions and
  /// CodeExtractors.
  BumpPtrAllocator OutlinerAllocator;
};

/// Pass to outline similar regions.
class IROutlinerPass : public PassInfoMixin<IROutlinerPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_IROUTLINER_H
