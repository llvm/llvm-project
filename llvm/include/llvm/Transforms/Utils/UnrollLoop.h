//===- llvm/Transforms/Utils/UnrollLoop.h - Unrolling utilities -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some loop unrolling utilities. It does not define any
// actual pass or policy, but provides a single function to perform loop
// unrolling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_UNROLLLOOP_H
#define LLVM_TRANSFORMS_UTILS_UNROLLLOOP_H


namespace llvm {

class StringRef;
class AssumptionCache;
class DominatorTree;
class Loop;
class LoopInfo;
class LPPassManager;
class MDNode;
class Pass;
class ScalarEvolution;

bool UnrollLoop(Loop *L, unsigned Count, unsigned TripCount, bool Force,
                bool AllowRuntime, bool AllowExpensiveTripCount,
                unsigned TripMultiple, LoopInfo *LI, ScalarEvolution *SE,
                DominatorTree *DT, AssumptionCache *AC, bool PreserveLCSSA);

bool UnrollRuntimeLoopRemainder(Loop *L, unsigned Count,
                                bool AllowExpensiveTripCount,
                                bool UseEpilogRemainder, LoopInfo *LI,
                                ScalarEvolution *SE, DominatorTree *DT,
                                bool PreserveLCSSA);

MDNode *GetUnrollMetadata(MDNode *LoopID, StringRef Name);
}

#endif
