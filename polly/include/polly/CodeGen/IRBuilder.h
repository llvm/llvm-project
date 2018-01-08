//===- Codegen/IRBuilder.h - The IR builder used by Polly -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The Polly IRBuilder file contains Polly specific extensions for the IRBuilder
// that are used e.g. to emit the llvm.loop.parallel metadata.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CODEGEN_IRBUILDER_H
#define POLLY_CODEGEN_IRBUILDER_H

#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {
class ScalarEvolution;
} // namespace llvm

namespace polly {
class Scop;

/// Helper class to annotate newly generated SCoPs with metadata.
///
/// The annotations are twofold:
///   1) Loops are stored in a stack-like structure in the order they are
///      constructed and the LoopID metadata node is added to the backedge.
///      Contained memory instructions and loop headers are annotated according
///      to all parallel surrounding loops.
///   2) The new SCoP is assumed alias free (either due to the result of
///      AliasAnalysis queries or runtime alias checks). We annotate therefore
///      all memory instruction with alias scopes to indicate that fact to
///      later optimizations.
///      These alias scopes live in a new alias domain only used in this SCoP.
///      Each base pointer has its own alias scope and is annotated to not
///      alias with any access to different base pointers.
class ScopAnnotator {
public:
  ScopAnnotator();

  /// Build all alias scopes for the given SCoP.
  void buildAliasScopes(Scop &S);

  /// Add a new loop @p L which is parallel if @p IsParallel is true.
  void pushLoop(llvm::Loop *L, bool IsParallel);

  /// Remove the last added loop.
  void popLoop(bool isParallel);

  /// Annotate the new instruction @p I for all parallel loops.
  void annotate(llvm::Instruction *I);

  /// Annotate the loop latch @p B wrt. @p L.
  void annotateLoopLatch(llvm::BranchInst *B, llvm::Loop *L, bool IsParallel,
                         bool IsLoopVectorizerDisabled) const;

  /// Add alternative alias based pointers
  ///
  /// When annotating instructions with alias scope metadata, the right metadata
  /// is identified through the base pointer of the memory access. In some cases
  /// (e.g. OpenMP code generation), the base pointer of the memory accesses is
  /// not the original base pointer, but was changed when passing the original
  /// base pointer over a function boundary. This function allows to provide a
  /// map that maps from these new base pointers to the original base pointers
  /// to allow the ScopAnnotator to still find the right alias scop annotations.
  ///
  /// @param NewMap A map from new base pointers to original base pointers.
  void addAlternativeAliasBases(
      llvm::DenseMap<llvm::AssertingVH<llvm::Value>,
                     llvm::AssertingVH<llvm::Value>> &NewMap) {
    AlternativeAliasBases.insert(NewMap.begin(), NewMap.end());
  }

  /// Delete the set of alternative alias bases
  void resetAlternativeAliasBases() { AlternativeAliasBases.clear(); }

  /// Add inter iteration alias-free base pointer @p BasePtr.
  void addInterIterationAliasFreeBasePtr(llvm::Value *BasePtr);

private:
  /// Annotate with the second level alias metadata
  ///
  /// Annotate the instruction @p I with the second level alias metadata
  /// to distinguish the individual non-aliasing accesses that have inter
  /// iteration alias-free base pointers.
  ///
  /// @param I The instruction to be annotated.
  /// @param BasePtr The base pointer of @p I.
  void annotateSecondLevel(llvm::Instruction *I, llvm::Value *BasePtr);

  /// The ScalarEvolution analysis we use to find base pointers.
  llvm::ScalarEvolution *SE;

  /// All loops currently under construction.
  llvm::SmallVector<llvm::Loop *, 8> ActiveLoops;

  /// Metadata pointing to parallel loops currently under construction.
  llvm::SmallVector<llvm::MDNode *, 8> ParallelLoops;

  /// The alias scope domain for the current SCoP.
  llvm::MDNode *AliasScopeDomain;

  /// A map from base pointers to its alias scope.
  llvm::MapVector<llvm::AssertingVH<llvm::Value>, llvm::MDNode *> AliasScopeMap;

  /// A map from base pointers to an alias scope list of other pointers.
  llvm::DenseMap<llvm::AssertingVH<llvm::Value>, llvm::MDNode *>
      OtherAliasScopeListMap;

  /// A map from pointers to second level alias scopes.
  llvm::DenseMap<const llvm::SCEV *, llvm::MDNode *> SecondLevelAliasScopeMap;

  /// A map from pointers to second level alias scope list of other pointers.
  llvm::DenseMap<const llvm::SCEV *, llvm::MDNode *>
      SecondLevelOtherAliasScopeListMap;

  /// Inter iteration alias-free base pointers.
  llvm::SmallPtrSet<llvm::Value *, 4> InterIterationAliasFreeBasePtrs;

  llvm::DenseMap<llvm::AssertingVH<llvm::Value>, llvm::AssertingVH<llvm::Value>>
      AlternativeAliasBases;
};

/// Add Polly specifics when running IRBuilder.
///
/// This is used to add additional items such as e.g. the llvm.loop.parallel
/// metadata.
class IRInserter : protected llvm::IRBuilderDefaultInserter {
public:
  IRInserter() = default;
  IRInserter(class ScopAnnotator &A) : Annotator(&A) {}

protected:
  void InsertHelper(llvm::Instruction *I, const llvm::Twine &Name,
                    llvm::BasicBlock *BB,
                    llvm::BasicBlock::iterator InsertPt) const {
    llvm::IRBuilderDefaultInserter::InsertHelper(I, Name, BB, InsertPt);
    if (Annotator)
      Annotator->annotate(I);
  }

private:
  class ScopAnnotator *Annotator = nullptr;
};

// TODO: We should not name instructions in NDEBUG builds.
//
// We currently always name instructions, as the polly test suite currently
// matches for certain names.
typedef llvm::IRBuilder<llvm::ConstantFolder, IRInserter> PollyIRBuilder;

/// Return an IR builder pointed before the @p BB terminator.
static inline PollyIRBuilder createPollyIRBuilder(llvm::BasicBlock *BB,
                                                  ScopAnnotator &LA) {
  PollyIRBuilder Builder(BB->getContext(), llvm::ConstantFolder(),
                         polly::IRInserter(LA));
  Builder.SetInsertPoint(BB->getTerminator());
  return Builder;
}
} // namespace polly
#endif
