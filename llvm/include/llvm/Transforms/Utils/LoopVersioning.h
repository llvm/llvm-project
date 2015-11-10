//===- LoopVersioning.h - Utility to version a loop -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a utility class to perform loop versioning.  The versioned
// loop speculates that otherwise may-aliasing memory accesses don't overlap and
// emits checks to prove this.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOOPVERSIONING_H
#define LLVM_TRANSFORMS_UTILS_LOOPVERSIONING_H

#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

namespace llvm {

class Loop;
class LoopAccessInfo;
class LoopInfo;
class ScalarEvolution;

/// \brief This class emits a version of the loop where run-time checks ensure
/// that may-alias pointers can't overlap.
///
/// It currently only supports single-exit loops and assumes that the loop
/// already has a preheader.
class LoopVersioning {
public:
  /// \brief Expects LoopAccessInfo, Loop, LoopInfo, DominatorTree as input.
  /// It uses runtime check provided by the user. If \p UseLAIChecks is true,
  /// we will retain the default checks made by LAI. Otherwise, construct an
  /// object having no checks and we expect the user to add them.
  LoopVersioning(const LoopAccessInfo &LAI, Loop *L, LoopInfo *LI,
                 DominatorTree *DT, ScalarEvolution *SE,
                 bool UseLAIChecks = true);

  /// \brief Performs the CFG manipulation part of versioning the loop including
  /// the DominatorTree and LoopInfo updates.
  ///
  /// The loop that was used to construct the class will be the "versioned" loop
  /// i.e. the loop that will receive control if all the memchecks pass.
  ///
  /// This allows the loop transform pass to operate on the same loop regardless
  /// of whether versioning was necessary or not:
  ///
  ///    for each loop L:
  ///        analyze L
  ///        if versioning is necessary version L
  ///        transform L
  void versionLoop() { versionLoop(findDefsUsedOutsideOfLoop(VersionedLoop)); }

  /// \brief Same but if the client has already precomputed the set of values
  /// used outside the loop, this API will allows passing that.
  void versionLoop(const SmallVectorImpl<Instruction *> &DefsUsedOutside);

  /// \brief Returns the versioned loop.  Control flows here if pointers in the
  /// loop don't alias (i.e. all memchecks passed).  (This loop is actually the
  /// same as the original loop that we got constructed with.)
  Loop *getVersionedLoop() { return VersionedLoop; }

  /// \brief Returns the fall-back loop.  Control flows here if pointers in the
  /// loop may alias (i.e. one of the memchecks failed).
  Loop *getNonVersionedLoop() { return NonVersionedLoop; }

  /// \brief Sets the runtime alias checks for versioning the loop.
  void setAliasChecks(
      const SmallVector<RuntimePointerChecking::PointerCheck, 4> Checks);

  /// \brief Sets the runtime SCEV checks for versioning the loop.
  void setSCEVChecks(SCEVUnionPredicate Check);

private:
  /// \brief Adds the necessary PHI nodes for the versioned loops based on the
  /// loop-defined values used outside of the loop.
  ///
  /// This needs to be called after versionLoop if there are defs in the loop
  /// that are used outside the loop.
  void addPHINodes(const SmallVectorImpl<Instruction *> &DefsUsedOutside);

  /// \brief The original loop.  This becomes the "versioned" one.  I.e.,
  /// control flows here if pointers in the loop don't alias.
  Loop *VersionedLoop;
  /// \brief The fall-back loop.  I.e. control flows here if pointers in the
  /// loop may alias (memchecks failed).
  Loop *NonVersionedLoop;

  /// \brief This maps the instructions from VersionedLoop to their counterpart
  /// in NonVersionedLoop.
  ValueToValueMapTy VMap;

  /// \brief The set of alias checks that we are versioning for.
  SmallVector<RuntimePointerChecking::PointerCheck, 4> AliasChecks;

  /// \brief The set of SCEV checks that we are versioning for.
  SCEVUnionPredicate Preds;

  /// \brief Analyses used.
  const LoopAccessInfo &LAI;
  LoopInfo *LI;
  DominatorTree *DT;
  ScalarEvolution *SE;
};
}

#endif
