//===- LoopVectorizationUtils.cpp - Utilities for LoopVectorize -----------===//
///
/// \file
/// This file implements stateless functions that are used by the LoopVectorize
/// and its related files.
//===----------------------------------------------------------------------===//

#include "LoopVectorizationUtils.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionPatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loop-vectorize"

using namespace llvm;
using namespace SCEVPatternMatch;

/// Write a \p DebugMsg about vectorization to the debug output stream. If \p I
/// is passed, the message relates to that particular instruction.
#ifndef NDEBUG
static void debugVectorizationMessage(const StringRef Prefix,
                                      const StringRef DebugMsg,
                                      Instruction *I) {
  dbgs() << "LV: " << Prefix << DebugMsg;
  if (I != nullptr)
    dbgs() << " " << *I;
  else
    dbgs() << '.';
  dbgs() << '\n';
}
#endif

/// Create an analysis remark that explains why vectorization failed
///
/// \p PassName is the name of the pass (e.g. can be AlwaysPrint).  \p
/// RemarkName is the identifier for the remark.  If \p I is passed it is an
/// instruction that prevents vectorization.  Otherwise \p TheLoop is used for
/// the location of the remark. If \p DL is passed, use it as debug location for
/// the remark. \return the remark object that can be streamed to.
static OptimizationRemarkAnalysis
createLVAnalysis(const char *PassName, StringRef RemarkName,
                 const Loop *TheLoop, Instruction *I, DebugLoc DL = {}) {
  BasicBlock *CodeRegion = I ? I->getParent() : TheLoop->getHeader();
  // If debug location is attached to the instruction, use it. Otherwise if DL
  // was not provided, use the loop's.
  if (I && I->getDebugLoc())
    DL = I->getDebugLoc();
  else if (!DL)
    DL = TheLoop->getStartLoc();

  return OptimizationRemarkAnalysis(PassName, RemarkName, DL, CodeRegion);
}

void LoopVectorizationUtils::reportVectorizationFailure(
    const char *PassName, const StringRef DebugMsg, const StringRef OREMsg,
    const StringRef ORETag, OptimizationRemarkEmitter *ORE, const Loop *TheLoop,
    Instruction *I) {
  LLVM_DEBUG(debugVectorizationMessage("Not vectorizing: ", DebugMsg, I));
  ORE->emit(createLVAnalysis(PassName, ORETag, TheLoop, I)
            << "loop not vectorized: " << OREMsg);
}

void LoopVectorizationUtils::reportVectorizationInfo(
    const char *PassName, const StringRef Msg, const StringRef ORETag,
    OptimizationRemarkEmitter *ORE, const Loop *TheLoop, Instruction *I,
    DebugLoc DL) {
  LLVM_DEBUG(debugVectorizationMessage("", Msg, I));
  ORE->emit(createLVAnalysis(PassName, ORETag, TheLoop, I, DL) << Msg);
}

void LoopVectorizationUtils::reportVectorization(const char *PassName,
                                                 OptimizationRemarkEmitter *ORE,
                                                 Loop *TheLoop,
                                                 ElementCount VFWidth,
                                                 unsigned IC) {
  LLVM_DEBUG(debugVectorizationMessage(
      "Vectorizing: ", TheLoop->isInnermost() ? "innermost loop" : "outer loop",
      nullptr));
  StringRef LoopType = TheLoop->isInnermost() ? "" : "outer ";
  ORE->emit([&]() {
    return OptimizationRemark(PassName, "Vectorized", TheLoop->getStartLoc(),
                              TheLoop->getHeader())
           << "vectorized " << LoopType << "loop (vectorization width: "
           << ore::NV("VectorizationFactor", VFWidth)
           << ", interleaved count: " << ore::NV("InterleaveCount", IC) << ")";
  });
}
