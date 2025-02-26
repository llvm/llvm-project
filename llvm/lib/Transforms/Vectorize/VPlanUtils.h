//===- VPlanUtils.h - VPlan-related utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLANUTILS_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLANUTILS_H

#include "VPlan.h"

namespace llvm {
class ScalarEvolution;
class SCEV;
} // namespace llvm

namespace llvm {

namespace vputils {
/// Returns true if only the first lane of \p Def is used.
bool onlyFirstLaneUsed(const VPValue *Def);

/// Returns true if only the first part of \p Def is used.
bool onlyFirstPartUsed(const VPValue *Def);

/// Get or create a VPValue that corresponds to the expansion of \p Expr. If \p
/// Expr is a SCEVConstant or SCEVUnknown, return a VPValue wrapping the live-in
/// value. Otherwise return a VPExpandSCEVRecipe to expand \p Expr. If \p Plan's
/// pre-header already contains a recipe expanding \p Expr, return it. If not,
/// create a new one.
VPValue *getOrCreateVPValueForSCEVExpr(VPlan &Plan, const SCEV *Expr,
                                       ScalarEvolution &SE);

/// Return the SCEV expression for \p V. Returns SCEVCouldNotCompute if no
/// SCEV expression could be constructed.
const SCEV *getSCEVExprForVPValue(VPValue *V, ScalarEvolution &SE);

/// Returns true if \p VPV is uniform after vectorization.
inline bool isUniformAfterVectorization(const VPValue *VPV) {
  // A value defined outside the vector region must be uniform after
  // vectorization inside a vector region.
  if (VPV->isDefinedOutsideLoopRegions())
    return true;
  if (auto *Rep = dyn_cast<VPReplicateRecipe>(VPV))
    return Rep->isUniform();
  if (isa<VPWidenGEPRecipe, VPDerivedIVRecipe>(VPV))
    return all_of(VPV->getDefiningRecipe()->operands(),
                  isUniformAfterVectorization);
  if (auto *VPI = dyn_cast<VPInstruction>(VPV))
    return VPI->isSingleScalar() || VPI->isVectorToScalar() ||
           ((Instruction::isBinaryOp(VPI->getOpcode()) ||
             VPI->getOpcode() == VPInstruction::PtrAdd) &&
            all_of(VPI->operands(), isUniformAfterVectorization));
  if (auto *IV = dyn_cast<VPDerivedIVRecipe>(VPV))
    return all_of(IV->operands(), isUniformAfterVectorization);

  // VPExpandSCEVRecipes must be placed in the entry and are alway uniform.
  return isa<VPExpandSCEVRecipe>(VPV);
}

/// Return true if \p V is a header mask in \p Plan.
bool isHeaderMask(const VPValue *V, VPlan &Plan);

/// Checks if \p V is uniform across all VF lanes and UF parts. It is considered
/// as such if it is either loop invariant (defined outside the vector region)
/// or its operand is known to be uniform across all VFs and UFs (e.g.
/// VPDerivedIV or VPCanonicalIVPHI).
bool isUniformAcrossVFsAndUFs(VPValue *V);

} // namespace vputils

//===----------------------------------------------------------------------===//
// Utilities for modifying predecessors and successors of VPlan blocks.
//===----------------------------------------------------------------------===//

/// Class that provides utilities for VPBlockBases in VPlan.
class VPBlockUtils {
public:
  VPBlockUtils() = delete;

  /// Insert disconnected VPBlockBase \p NewBlock after \p BlockPtr. Add \p
  /// NewBlock as successor of \p BlockPtr and \p BlockPtr as predecessor of \p
  /// NewBlock, and propagate \p BlockPtr parent to \p NewBlock. \p BlockPtr's
  /// successors are moved from \p BlockPtr to \p NewBlock. \p NewBlock must
  /// have neither successors nor predecessors.
  static void insertBlockAfter(VPBlockBase *NewBlock, VPBlockBase *BlockPtr) {
    assert(NewBlock->getSuccessors().empty() &&
           NewBlock->getPredecessors().empty() &&
           "Can't insert new block with predecessors or successors.");
    NewBlock->setParent(BlockPtr->getParent());
    SmallVector<VPBlockBase *> Succs(BlockPtr->successors());
    for (VPBlockBase *Succ : Succs) {
      disconnectBlocks(BlockPtr, Succ);
      connectBlocks(NewBlock, Succ);
    }
    connectBlocks(BlockPtr, NewBlock);
  }

  /// Insert disconnected block \p NewBlock before \p Blockptr. First
  /// disconnects all predecessors of \p BlockPtr and connects them to \p
  /// NewBlock. Add \p NewBlock as predecessor of \p BlockPtr and \p BlockPtr as
  /// successor of \p NewBlock.
  static void insertBlockBefore(VPBlockBase *NewBlock, VPBlockBase *BlockPtr) {
    assert(NewBlock->getSuccessors().empty() &&
           NewBlock->getPredecessors().empty() &&
           "Can't insert new block with predecessors or successors.");
    NewBlock->setParent(BlockPtr->getParent());
    for (VPBlockBase *Pred : to_vector(BlockPtr->predecessors())) {
      disconnectBlocks(Pred, BlockPtr);
      connectBlocks(Pred, NewBlock);
    }
    connectBlocks(NewBlock, BlockPtr);
  }

  /// Insert disconnected VPBlockBases \p IfTrue and \p IfFalse after \p
  /// BlockPtr. Add \p IfTrue and \p IfFalse as succesors of \p BlockPtr and \p
  /// BlockPtr as predecessor of \p IfTrue and \p IfFalse. Propagate \p BlockPtr
  /// parent to \p IfTrue and \p IfFalse. \p BlockPtr must have no successors
  /// and \p IfTrue and \p IfFalse must have neither successors nor
  /// predecessors.
  static void insertTwoBlocksAfter(VPBlockBase *IfTrue, VPBlockBase *IfFalse,
                                   VPBlockBase *BlockPtr) {
    assert(IfTrue->getSuccessors().empty() &&
           "Can't insert IfTrue with successors.");
    assert(IfFalse->getSuccessors().empty() &&
           "Can't insert IfFalse with successors.");
    BlockPtr->setTwoSuccessors(IfTrue, IfFalse);
    IfTrue->setPredecessors({BlockPtr});
    IfFalse->setPredecessors({BlockPtr});
    IfTrue->setParent(BlockPtr->getParent());
    IfFalse->setParent(BlockPtr->getParent());
  }

  /// Connect VPBlockBases \p From and \p To bi-directionally. If \p PredIdx is
  /// -1, append \p From to the predecessors of \p To, otherwise set \p To's
  /// predecessor at \p PredIdx to \p From. If \p SuccIdx is -1, append \p To to
  /// the successors of \p From, otherwise set \p From's successor at \p SuccIdx
  /// to \p To. Both VPBlockBases must have the same parent, which can be null.
  /// Both VPBlockBases can be already connected to other VPBlockBases.
  static void connectBlocks(VPBlockBase *From, VPBlockBase *To,
                            unsigned PredIdx = -1u, unsigned SuccIdx = -1u) {
    assert((From->getParent() == To->getParent()) &&
           "Can't connect two block with different parents");
    assert((SuccIdx != -1u || From->getNumSuccessors() < 2) &&
           "Blocks can't have more than two successors.");
    if (SuccIdx == -1u)
      From->appendSuccessor(To);
    else
      From->getSuccessors()[SuccIdx] = To;

    if (PredIdx == -1u)
      To->appendPredecessor(From);
    else
      To->getPredecessors()[PredIdx] = From;
  }

  /// Disconnect VPBlockBases \p From and \p To bi-directionally. Remove \p To
  /// from the successors of \p From and \p From from the predecessors of \p To.
  static void disconnectBlocks(VPBlockBase *From, VPBlockBase *To) {
    assert(To && "Successor to disconnect is null.");
    From->removeSuccessor(To);
    To->removePredecessor(From);
  }

  /// Reassociate all the blocks connected to \p Old so that they now point to
  /// \p New.
  static void reassociateBlocks(VPBlockBase *Old, VPBlockBase *New) {
    for (auto *Pred : to_vector(Old->getPredecessors()))
      Pred->replaceSuccessor(Old, New);
    for (auto *Succ : to_vector(Old->getSuccessors()))
      Succ->replacePredecessor(Old, New);
    New->setPredecessors(Old->getPredecessors());
    New->setSuccessors(Old->getSuccessors());
    Old->clearPredecessors();
    Old->clearSuccessors();
  }

  /// Return an iterator range over \p Range which only includes \p BlockTy
  /// blocks. The accesses are casted to \p BlockTy.
  template <typename BlockTy, typename T>
  static auto blocksOnly(const T &Range) {
    // Create BaseTy with correct const-ness based on BlockTy.
    using BaseTy = std::conditional_t<std::is_const<BlockTy>::value,
                                      const VPBlockBase, VPBlockBase>;

    // We need to first create an iterator range over (const) BlocktTy & instead
    // of (const) BlockTy * for filter_range to work properly.
    auto Mapped =
        map_range(Range, [](BaseTy *Block) -> BaseTy & { return *Block; });
    auto Filter = make_filter_range(
        Mapped, [](BaseTy &Block) { return isa<BlockTy>(&Block); });
    return map_range(Filter, [](BaseTy &Block) -> BlockTy * {
      return cast<BlockTy>(&Block);
    });
  }

  /// Inserts \p BlockPtr on the edge between \p From and \p To. That is, update
  /// \p From's successor to \p To to point to \p BlockPtr and \p To's
  /// predecessor from \p From to \p BlockPtr. \p From and \p To are added to \p
  /// BlockPtr's predecessors and successors respectively. There must be a
  /// single edge between \p From and \p To.
  static void insertOnEdge(VPBlockBase *From, VPBlockBase *To,
                           VPBlockBase *BlockPtr) {
    auto &Successors = From->getSuccessors();
    auto &Predecessors = To->getPredecessors();
    assert(count(Successors, To) == 1 && count(Predecessors, From) == 1 &&
           "must have single between From and To");
    unsigned SuccIdx = std::distance(Successors.begin(), find(Successors, To));
    unsigned PredIx =
        std::distance(Predecessors.begin(), find(Predecessors, From));
    VPBlockUtils::connectBlocks(From, BlockPtr, -1, SuccIdx);
    VPBlockUtils::connectBlocks(BlockPtr, To, PredIx, -1);
  }
};

} // namespace llvm

#endif
