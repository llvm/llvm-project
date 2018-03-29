//===- LoopRotationUtils.h - Utilities to perform loop rotation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities to convert a loop into a loop with bottom test.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPROTATIONUTILS_H
#define LLVM_TRANSFORMS_UTILS_LOOPROTATIONUTILS_H

namespace llvm {

class AssumptionCache;
class DominatorTree;
class Loop;
class LoopInfo;
class ScalarEvolution;
struct SimplifyQuery;
class TargetTransformInfo;

/// \brief Convert a loop into a loop with bottom test.
bool LoopRotation(Loop *L, unsigned MaxHeaderSize, LoopInfo *LI,
                  const TargetTransformInfo *TTI, AssumptionCache *AC,
                  DominatorTree *DT, ScalarEvolution *SE,
                  const SimplifyQuery &SQ);

} // namespace llvm

#endif
