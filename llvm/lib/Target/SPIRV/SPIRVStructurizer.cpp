//===-- SPIRVStructurizer.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Analysis/SPIRVConvergenceRegionAnalysis.h"
#include "SPIRV.h"
#include "SPIRVStructurizerWrapper.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include <stack>
#include <unordered_set>

using namespace llvm;
using namespace SPIRV;

using BlockSet = std::unordered_set<BasicBlock *>;
using Edge = std::pair<BasicBlock *, BasicBlock *>;

// Helper function to do a partial order visit from the block |Start|, calling
// |Op| on each visited node.
static void partialOrderVisit(BasicBlock &Start,
                              std::function<bool(BasicBlock *)> Op) {
  PartialOrderingVisitor V(*Start.getParent());
  V.partialOrderVisit(Start, std::move(Op));
}

// Returns the exact convergence region in the tree defined by `Node` for which
// `BB` is the header, nullptr otherwise.
static const ConvergenceRegion *
getRegionForHeader(const ConvergenceRegion *Node, BasicBlock *BB) {
  if (Node->Entry == BB)
    return Node;

  for (auto *Child : Node->Children) {
    const auto *CR = getRegionForHeader(Child, BB);
    if (CR != nullptr)
      return CR;
  }
  return nullptr;
}

// Returns the single BasicBlock exiting the convergence region `CR`,
// nullptr if no such exit exists.
static BasicBlock *getExitFor(const ConvergenceRegion *CR) {
  std::unordered_set<BasicBlock *> ExitTargets;
  for (BasicBlock *Exit : CR->Exits) {
    for (BasicBlock *Successor : successors(Exit)) {
      if (CR->Blocks.count(Successor) == 0)
        ExitTargets.insert(Successor);
    }
  }

  assert(ExitTargets.size() <= 1);
  if (ExitTargets.size() == 0)
    return nullptr;

  return *ExitTargets.begin();
}

// Returns the merge block designated by I if I is a merge instruction, nullptr
// otherwise.
static BasicBlock *getDesignatedMergeBlock(Instruction *I) {
  IntrinsicInst *II = dyn_cast_or_null<IntrinsicInst>(I);
  if (II == nullptr)
    return nullptr;

  if (II->getIntrinsicID() != Intrinsic::spv_loop_merge &&
      II->getIntrinsicID() != Intrinsic::spv_selection_merge)
    return nullptr;

  BlockAddress *BA = cast<BlockAddress>(II->getOperand(0));
  return BA->getBasicBlock();
}

// Returns the continue block designated by I if I is an OpLoopMerge, nullptr
// otherwise.
static BasicBlock *getDesignatedContinueBlock(Instruction *I) {
  IntrinsicInst *II = dyn_cast_or_null<IntrinsicInst>(I);
  if (II == nullptr)
    return nullptr;

  if (II->getIntrinsicID() != Intrinsic::spv_loop_merge)
    return nullptr;

  BlockAddress *BA = cast<BlockAddress>(II->getOperand(1));
  return BA->getBasicBlock();
}

// Returns true if Header has one merge instruction which designated Merge as
// merge block.
static bool isDefinedAsSelectionMergeBy(BasicBlock &Header, BasicBlock &Merge) {
  for (auto &I : Header) {
    BasicBlock *MB = getDesignatedMergeBlock(&I);
    if (MB == &Merge)
      return true;
  }
  return false;
}

// Returns true if the BB has one OpLoopMerge instruction.
static bool hasLoopMergeInstruction(BasicBlock &BB) {
  for (auto &I : BB)
    if (getDesignatedContinueBlock(&I))
      return true;
  return false;
}

// Returns true is I is an OpSelectionMerge or OpLoopMerge instruction, false
// otherwise.
static bool isMergeInstruction(Instruction *I) {
  return getDesignatedMergeBlock(I) != nullptr;
}

// Returns all blocks in F having at least one OpLoopMerge or OpSelectionMerge
// instruction.
static SmallPtrSet<BasicBlock *, 2> getHeaderBlocks(Function &F) {
  SmallPtrSet<BasicBlock *, 2> Output;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (getDesignatedMergeBlock(&I) != nullptr)
        Output.insert(&BB);
    }
  }
  return Output;
}

// Returns all basic blocks in |F| referenced by at least 1
// OpSelectionMerge/OpLoopMerge instruction.
static SmallPtrSet<BasicBlock *, 2> getMergeBlocks(Function &F) {
  SmallPtrSet<BasicBlock *, 2> Output;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      BasicBlock *MB = getDesignatedMergeBlock(&I);
      if (MB != nullptr)
        Output.insert(MB);
    }
  }
  return Output;
}

// Return all the merge instructions contained in BB.
// Note: the SPIR-V spec doesn't allow a single BB to contain more than 1 merge
// instruction, but this can happen while we structurize the CFG.
static std::vector<Instruction *> getMergeInstructions(BasicBlock &BB) {
  std::vector<Instruction *> Output;
  for (Instruction &I : BB)
    if (isMergeInstruction(&I))
      Output.push_back(&I);
  return Output;
}

// Returns all basic blocks in |F| referenced as continue target by at least 1
// OpLoopMerge instruction.
static SmallPtrSet<BasicBlock *, 2> getContinueBlocks(Function &F) {
  SmallPtrSet<BasicBlock *, 2> Output;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      BasicBlock *MB = getDesignatedContinueBlock(&I);
      if (MB != nullptr)
        Output.insert(MB);
    }
  }
  return Output;
}

// Do a preorder traversal of the CFG starting from the BB |Start|.
// point. Calls |op| on each basic block encountered during the traversal.
static void visit(BasicBlock &Start, std::function<bool(BasicBlock *)> op) {
  std::stack<BasicBlock *> ToVisit;
  SmallPtrSet<BasicBlock *, 8> Seen;

  ToVisit.push(&Start);
  Seen.insert(ToVisit.top());
  while (ToVisit.size() != 0) {
    BasicBlock *BB = ToVisit.top();
    ToVisit.pop();

    if (!op(BB))
      continue;

    for (auto Succ : successors(BB)) {
      if (Seen.contains(Succ))
        continue;
      ToVisit.push(Succ);
      Seen.insert(Succ);
    }
  }
}

// Replaces the conditional and unconditional branch targets of |BB| by
// |NewTarget| if the target was |OldTarget|. This function also makes sure the
// associated merge instruction gets updated accordingly.
static void replaceIfBranchTargets(BasicBlock *BB, BasicBlock *OldTarget,
                                   BasicBlock *NewTarget) {
  auto *BI = cast<BranchInst>(BB->getTerminator());

  // 1. Replace all matching successors.
  for (size_t i = 0; i < BI->getNumSuccessors(); i++) {
    if (BI->getSuccessor(i) == OldTarget)
      BI->setSuccessor(i, NewTarget);
  }

  // Branch was unconditional, no fixup required.
  if (BI->isUnconditional())
    return;

  // Branch had 2 successors, maybe now both are the same?
  if (BI->getSuccessor(0) != BI->getSuccessor(1))
    return;

  // Note: we may end up here because the original IR had such branches.
  // This means Target is not necessarily equal to NewTarget.
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BI);
  Builder.CreateBr(BI->getSuccessor(0));
  BI->eraseFromParent();

  // The branch was the only instruction, nothing else to do.
  if (BB->size() == 1)
    return;

  // Otherwise, we need to check: was there an OpSelectionMerge before this
  // branch? If we removed the OpBranchConditional, we must also remove the
  // OpSelectionMerge. This is not valid for OpLoopMerge:
  IntrinsicInst *II =
      dyn_cast<IntrinsicInst>(BB->getTerminator()->getPrevNode());
  if (!II || II->getIntrinsicID() != Intrinsic::spv_selection_merge)
    return;

  Constant *C = cast<Constant>(II->getOperand(0));
  II->eraseFromParent();
  if (!C->isConstantUsed())
    C->destroyConstant();
}

// Replaces the target of branch instruction in |BB| with |NewTarget| if it
// was |OldTarget|. This function also fixes the associated merge instruction.
// Note: this function does not simplify branching instructions, it only updates
// targets. See also: simplifyBranches.
static void replaceBranchTargets(BasicBlock *BB, BasicBlock *OldTarget,
                                 BasicBlock *NewTarget) {
  auto *T = BB->getTerminator();
  if (isa<ReturnInst>(T))
    return;

  if (isa<BranchInst>(T))
    return replaceIfBranchTargets(BB, OldTarget, NewTarget);

  if (auto *SI = dyn_cast<SwitchInst>(T)) {
    for (size_t i = 0; i < SI->getNumSuccessors(); i++) {
      if (SI->getSuccessor(i) == OldTarget)
        SI->setSuccessor(i, NewTarget);
    }
    return;
  }

  assert(false && "Unhandled terminator type.");
}

namespace {
// Given a reducible CFG, produces a structurized CFG in the SPIR-V sense,
// adding merge instructions when required.
class SPIRVStructurizer : public FunctionPass {
  struct DivergentConstruct;
  // Represents a list of condition/loops/switch constructs.
  // See SPIR-V 2.11.2. Structured Control-flow Constructs for the list of
  // constructs.
  using ConstructList = std::vector<std::unique_ptr<DivergentConstruct>>;

  // Represents a divergent construct in the SPIR-V sense.
  // Such constructs are represented by a header (entry), a merge block (exit),
  // and possibly a continue block (back-edge). A construct can contain other
  // constructs, but their boundaries do not cross.
  struct DivergentConstruct {
    BasicBlock *Header = nullptr;
    BasicBlock *Merge = nullptr;
    BasicBlock *Continue = nullptr;

    DivergentConstruct *Parent = nullptr;
    ConstructList Children;
  };

  // An helper class to clean the construct boundaries.
  // It is used to gather the list of blocks that should belong to each
  // divergent construct, and possibly modify CFG edges when exits would cross
  // the boundary of multiple constructs.
  struct Splitter {
    Function &F;
    LoopInfo &LI;
    DomTreeBuilder::BBDomTree DT;
    DomTreeBuilder::BBPostDomTree PDT;

    Splitter(Function &F, LoopInfo &LI) : F(F), LI(LI) { invalidate(); }

    void invalidate() {
      PDT.recalculate(F);
      DT.recalculate(F);
    }

    // Returns the list of blocks that belong to a SPIR-V loop construct,
    // including the continue construct.
    std::vector<BasicBlock *> getLoopConstructBlocks(BasicBlock *Header,
                                                     BasicBlock *Merge) {
      assert(DT.dominates(Header, Merge));
      std::vector<BasicBlock *> Output;
      partialOrderVisit(*Header, [&](BasicBlock *BB) {
        if (BB == Merge)
          return false;
        if (DT.dominates(Merge, BB) || !DT.dominates(Header, BB))
          return false;
        Output.push_back(BB);
        return true;
      });
      return Output;
    }

    // Returns the list of blocks that belong to a SPIR-V selection construct.
    std::vector<BasicBlock *>
    getSelectionConstructBlocks(DivergentConstruct *Node) {
      assert(DT.dominates(Node->Header, Node->Merge));
      BlockSet OutsideBlocks;
      OutsideBlocks.insert(Node->Merge);

      for (DivergentConstruct *It = Node->Parent; It != nullptr;
           It = It->Parent) {
        OutsideBlocks.insert(It->Merge);
        if (It->Continue)
          OutsideBlocks.insert(It->Continue);
      }

      std::vector<BasicBlock *> Output;
      partialOrderVisit(*Node->Header, [&](BasicBlock *BB) {
        if (OutsideBlocks.count(BB) != 0)
          return false;
        if (DT.dominates(Node->Merge, BB) || !DT.dominates(Node->Header, BB))
          return false;
        Output.push_back(BB);
        return true;
      });
      return Output;
    }

    // Returns the list of blocks that belong to a SPIR-V switch construct.
    std::vector<BasicBlock *> getSwitchConstructBlocks(BasicBlock *Header,
                                                       BasicBlock *Merge) {
      assert(DT.dominates(Header, Merge));

      std::vector<BasicBlock *> Output;
      partialOrderVisit(*Header, [&](BasicBlock *BB) {
        // the blocks structurally dominated by a switch header,
        if (!DT.dominates(Header, BB))
          return false;
        // excluding blocks structurally dominated by the switch header’s merge
        // block.
        if (DT.dominates(Merge, BB) || BB == Merge)
          return false;
        Output.push_back(BB);
        return true;
      });
      return Output;
    }

    // Returns the list of blocks that belong to a SPIR-V case construct.
    std::vector<BasicBlock *> getCaseConstructBlocks(BasicBlock *Target,
                                                     BasicBlock *Merge) {
      assert(DT.dominates(Target, Merge));

      std::vector<BasicBlock *> Output;
      partialOrderVisit(*Target, [&](BasicBlock *BB) {
        // the blocks structurally dominated by an OpSwitch Target or Default
        // block
        if (!DT.dominates(Target, BB))
          return false;
        // excluding the blocks structurally dominated by the OpSwitch
        // construct’s corresponding merge block.
        if (DT.dominates(Merge, BB) || BB == Merge)
          return false;
        Output.push_back(BB);
        return true;
      });
      return Output;
    }

    // Splits the given edges by recreating proxy nodes so that the destination
    // has unique incoming edges from this region.
    //
    // clang-format off
    //
    // In SPIR-V, constructs must have a single exit/merge.
    // Given nodes A and B in the construct, a node C outside, and the following edges.
    //  A -> C
    //  B -> C
    //
    // In such cases, we must create a new exit node D, that belong to the construct to make is viable:
    // A -> D -> C
    // B -> D -> C
    //
    // This is fine (assuming C has no PHI nodes), but requires handling the merge instruction here.
    // By adding a proxy node, we create a regular divergent shape which can easily be regularized later on.
    // A -> D -> D1 -> C
    // B -> D -> D2 -> C
    //
    // A, B, D belongs to the construct. D is the exit. D1 and D2 are empty.
    //
    // clang-format on
    std::vector<Edge>
    createAliasBlocksForComplexEdges(std::vector<Edge> Edges) {
      std::unordered_set<BasicBlock *> Seen;
      std::vector<Edge> Output;
      Output.reserve(Edges.size());

      for (auto &[Src, Dst] : Edges) {
        auto [Iterator, Inserted] = Seen.insert(Src);
        if (!Inserted) {
          // Src already a source node. Cannot have 2 edges from A to B.
          // Creating alias source block.
          BasicBlock *NewSrc = BasicBlock::Create(
              F.getContext(), Src->getName() + ".new.src", &F);
          replaceBranchTargets(Src, Dst, NewSrc);
          IRBuilder<> Builder(NewSrc);
          Builder.CreateBr(Dst);
          Src = NewSrc;
        }

        Output.emplace_back(Src, Dst);
      }

      return Output;
    }

    AllocaInst *CreateVariable(Function &F, Type *Type,
                               BasicBlock::iterator Position) {
      const DataLayout &DL = F.getDataLayout();
      return new AllocaInst(Type, DL.getAllocaAddrSpace(), nullptr, "reg",
                            Position);
    }

    // Given a construct defined by |Header|, and a list of exiting edges
    // |Edges|, creates a new single exit node, fixing up those edges.
    BasicBlock *createSingleExitNode(BasicBlock *Header,
                                     std::vector<Edge> &Edges) {

      std::vector<Edge> FixedEdges = createAliasBlocksForComplexEdges(Edges);

      std::vector<BasicBlock *> Dsts;
      std::unordered_map<BasicBlock *, ConstantInt *> DstToIndex;
      auto NewExit = BasicBlock::Create(F.getContext(),
                                        Header->getName() + ".new.exit", &F);
      IRBuilder<> ExitBuilder(NewExit);
      for (auto &[Src, Dst] : FixedEdges) {
        if (DstToIndex.count(Dst) != 0)
          continue;
        DstToIndex.emplace(Dst, ExitBuilder.getInt32(DstToIndex.size()));
        Dsts.push_back(Dst);
      }

      if (Dsts.size() == 1) {
        for (auto &[Src, Dst] : FixedEdges) {
          replaceBranchTargets(Src, Dst, NewExit);
        }
        ExitBuilder.CreateBr(Dsts[0]);
        return NewExit;
      }

      AllocaInst *Variable = CreateVariable(F, ExitBuilder.getInt32Ty(),
                                            F.begin()->getFirstInsertionPt());
      for (auto &[Src, Dst] : FixedEdges) {
        IRBuilder<> B2(Src);
        B2.SetInsertPoint(Src->getFirstInsertionPt());
        B2.CreateStore(DstToIndex[Dst], Variable);
        replaceBranchTargets(Src, Dst, NewExit);
      }

      Value *Load = ExitBuilder.CreateLoad(ExitBuilder.getInt32Ty(), Variable);

      // If we can avoid an OpSwitch, generate an OpBranch. Reason is some
      // OpBranch are allowed to exist without a new OpSelectionMerge if one of
      // the branch is the parent's merge node, while OpSwitches are not.
      if (Dsts.size() == 2) {
        Value *Condition =
            ExitBuilder.CreateCmp(CmpInst::ICMP_EQ, DstToIndex[Dsts[0]], Load);
        ExitBuilder.CreateCondBr(Condition, Dsts[0], Dsts[1]);
        return NewExit;
      }

      SwitchInst *Sw = ExitBuilder.CreateSwitch(Load, Dsts[0], Dsts.size() - 1);
      for (BasicBlock *BB : drop_begin(Dsts))
        Sw->addCase(DstToIndex[BB], BB);
      return NewExit;
    }
  };

  /// Create a value in BB set to the value associated with the branch the block
  /// terminator will take.
  Value *createExitVariable(
      BasicBlock *BB,
      const DenseMap<BasicBlock *, ConstantInt *> &TargetToValue) {
    auto *T = BB->getTerminator();
    if (isa<ReturnInst>(T))
      return nullptr;

    IRBuilder<> Builder(BB);
    Builder.SetInsertPoint(T);

    if (auto *BI = dyn_cast<BranchInst>(T)) {

      BasicBlock *LHSTarget = BI->getSuccessor(0);
      BasicBlock *RHSTarget =
          BI->isConditional() ? BI->getSuccessor(1) : nullptr;

      Value *LHS = TargetToValue.lookup(LHSTarget);
      Value *RHS = TargetToValue.lookup(RHSTarget);

      if (LHS == nullptr || RHS == nullptr)
        return LHS == nullptr ? RHS : LHS;
      return Builder.CreateSelect(BI->getCondition(), LHS, RHS);
    }

    // TODO: add support for switch cases.
    llvm_unreachable("Unhandled terminator type.");
  }

  // Creates a new basic block in F with a single OpUnreachable instruction.
  BasicBlock *CreateUnreachable(Function &F) {
    BasicBlock *BB = BasicBlock::Create(F.getContext(), "unreachable", &F);
    IRBuilder<> Builder(BB);
    Builder.CreateUnreachable();
    return BB;
  }

  // Add OpLoopMerge instruction on cycles.
  bool addMergeForLoops(Function &F) {
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto *TopLevelRegion =
        getAnalysis<SPIRVConvergenceRegionAnalysisWrapperPass>()
            .getRegionInfo()
            .getTopLevelRegion();

    bool Modified = false;
    for (auto &BB : F) {
      // Not a loop header. Ignoring for now.
      if (!LI.isLoopHeader(&BB))
        continue;
      auto *L = LI.getLoopFor(&BB);

      // This loop header is not the entrance of a convergence region. Ignoring
      // this block.
      auto *CR = getRegionForHeader(TopLevelRegion, &BB);
      if (CR == nullptr)
        continue;

      IRBuilder<> Builder(&BB);

      auto *Merge = getExitFor(CR);
      // We are indeed in a loop, but there are no exits (infinite loop).
      // This could be caused by a bad shader, but also could be an artifact
      // from an earlier optimization. It is not always clear if structurally
      // reachable means runtime reachable, so we cannot error-out. What we must
      // do however is to make is legal on the SPIR-V point of view, hence
      // adding an unreachable merge block.
      if (Merge == nullptr) {
        BranchInst *Br = cast<BranchInst>(BB.getTerminator());
        assert(Br->isUnconditional());

        Merge = CreateUnreachable(F);
        Builder.SetInsertPoint(Br);
        Builder.CreateCondBr(Builder.getFalse(), Merge, Br->getSuccessor(0));
        Br->eraseFromParent();
      }

      auto *Continue = L->getLoopLatch();

      Builder.SetInsertPoint(BB.getTerminator());
      auto MergeAddress = BlockAddress::get(Merge->getParent(), Merge);
      auto ContinueAddress = BlockAddress::get(Continue->getParent(), Continue);
      SmallVector<Value *, 2> Args = {MergeAddress, ContinueAddress};
      SmallVector<unsigned, 1> LoopControlImms =
          getSpirvLoopControlOperandsFromLoopMetadata(L);
      for (unsigned Imm : LoopControlImms)
        Args.emplace_back(ConstantInt::get(Builder.getInt32Ty(), Imm));
      Builder.CreateIntrinsic(Intrinsic::spv_loop_merge, {Args});
      Modified = true;
    }

    return Modified;
  }

  // Adds an OpSelectionMerge to the immediate dominator or each node with an
  // in-degree of 2 or more which is not already the merge target of an
  // OpLoopMerge/OpSelectionMerge.
  bool addMergeForNodesWithMultiplePredecessors(Function &F) {
    DomTreeBuilder::BBDomTree DT;
    DT.recalculate(F);

    bool Modified = false;
    for (auto &BB : F) {
      if (pred_size(&BB) <= 1)
        continue;

      if (hasLoopMergeInstruction(BB) && pred_size(&BB) <= 2)
        continue;

      assert(DT.getNode(&BB)->getIDom());
      BasicBlock *Header = DT.getNode(&BB)->getIDom()->getBlock();

      if (isDefinedAsSelectionMergeBy(*Header, BB))
        continue;

      IRBuilder<> Builder(Header);
      Builder.SetInsertPoint(Header->getTerminator());

      auto MergeAddress = BlockAddress::get(BB.getParent(), &BB);
      createOpSelectMerge(&Builder, MergeAddress);

      Modified = true;
    }

    return Modified;
  }

  // When a block has multiple OpSelectionMerge/OpLoopMerge instructions, sorts
  // them to put the "largest" first. A merge instruction is defined as larger
  // than another when its target merge block post-dominates the other target's
  // merge block. (This ordering should match the nesting ordering of the source
  // HLSL).
  bool sortSelectionMerge(Function &F, BasicBlock &Block) {
    std::vector<Instruction *> MergeInstructions;
    for (Instruction &I : Block)
      if (isMergeInstruction(&I))
        MergeInstructions.push_back(&I);

    if (MergeInstructions.size() <= 1)
      return false;

    Instruction *InsertionPoint = *MergeInstructions.begin();

    PartialOrderingVisitor Visitor(F);
    std::sort(MergeInstructions.begin(), MergeInstructions.end(),
              [&Visitor](Instruction *Left, Instruction *Right) {
                if (Left == Right)
                  return false;
                BasicBlock *RightMerge = getDesignatedMergeBlock(Right);
                BasicBlock *LeftMerge = getDesignatedMergeBlock(Left);
                return !Visitor.compare(RightMerge, LeftMerge);
              });

    for (Instruction *I : MergeInstructions) {
      I->moveBefore(InsertionPoint->getIterator());
      InsertionPoint = I;
    }

    return true;
  }

  // Sorts selection merge headers in |F|.
  // A is sorted before B if the merge block designated by B is an ancestor of
  // the one designated by A.
  bool sortSelectionMergeHeaders(Function &F) {
    bool Modified = false;
    for (BasicBlock &BB : F) {
      Modified |= sortSelectionMerge(F, BB);
    }
    return Modified;
  }

  // Split basic blocks containing multiple OpLoopMerge/OpSelectionMerge
  // instructions so each basic block contains only a single merge instruction.
  bool splitBlocksWithMultipleHeaders(Function &F) {
    std::stack<BasicBlock *> Work;
    for (auto &BB : F) {
      std::vector<Instruction *> MergeInstructions = getMergeInstructions(BB);
      if (MergeInstructions.size() <= 1)
        continue;
      Work.push(&BB);
    }

    const bool Modified = Work.size() > 0;
    while (Work.size() > 0) {
      BasicBlock *Header = Work.top();
      Work.pop();

      std::vector<Instruction *> MergeInstructions =
          getMergeInstructions(*Header);
      for (unsigned i = 1; i < MergeInstructions.size(); i++) {
        BasicBlock *NewBlock =
            Header->splitBasicBlock(MergeInstructions[i], "new.header");

        if (getDesignatedContinueBlock(MergeInstructions[0]) == nullptr) {
          BasicBlock *Unreachable = CreateUnreachable(F);

          BranchInst *BI = cast<BranchInst>(Header->getTerminator());
          IRBuilder<> Builder(Header);
          Builder.SetInsertPoint(BI);
          Builder.CreateCondBr(Builder.getTrue(), NewBlock, Unreachable);
          BI->eraseFromParent();
        }

        Header = NewBlock;
      }
    }

    return Modified;
  }

  // Adds an OpSelectionMerge to each block with an out-degree >= 2 which
  // doesn't already have an OpSelectionMerge.
  bool addMergeForDivergentBlocks(Function &F) {
    DomTreeBuilder::BBPostDomTree PDT;
    PDT.recalculate(F);
    bool Modified = false;

    auto MergeBlocks = getMergeBlocks(F);
    auto ContinueBlocks = getContinueBlocks(F);

    for (auto &BB : F) {
      if (getMergeInstructions(BB).size() != 0)
        continue;

      std::vector<BasicBlock *> Candidates;
      for (BasicBlock *Successor : successors(&BB)) {
        if (MergeBlocks.contains(Successor))
          continue;
        if (ContinueBlocks.contains(Successor))
          continue;
        Candidates.push_back(Successor);
      }

      if (Candidates.size() <= 1)
        continue;

      Modified = true;
      BasicBlock *Merge = Candidates[0];

      auto MergeAddress = BlockAddress::get(Merge->getParent(), Merge);
      IRBuilder<> Builder(&BB);
      Builder.SetInsertPoint(BB.getTerminator());
      createOpSelectMerge(&Builder, MergeAddress);
    }

    return Modified;
  }

  // Gather all the exit nodes for the construct header by |Header| and
  // containing the blocks |Construct|.
  std::vector<Edge> getExitsFrom(const BlockSet &Construct,
                                 BasicBlock &Header) {
    std::vector<Edge> Output;
    visit(Header, [&](BasicBlock *Item) {
      if (Construct.count(Item) == 0)
        return false;

      for (BasicBlock *Successor : successors(Item)) {
        if (Construct.count(Successor) == 0)
          Output.emplace_back(Item, Successor);
      }
      return true;
    });

    return Output;
  }

  // Build a divergent construct tree searching from |BB|.
  // If |Parent| is not null, this tree is attached to the parent's tree.
  void constructDivergentConstruct(BlockSet &Visited, Splitter &S,
                                   BasicBlock *BB, DivergentConstruct *Parent) {
    if (Visited.count(BB) != 0)
      return;
    Visited.insert(BB);

    auto MIS = getMergeInstructions(*BB);
    if (MIS.size() == 0) {
      for (BasicBlock *Successor : successors(BB))
        constructDivergentConstruct(Visited, S, Successor, Parent);
      return;
    }

    assert(MIS.size() == 1);
    Instruction *MI = MIS[0];

    BasicBlock *Merge = getDesignatedMergeBlock(MI);
    BasicBlock *Continue = getDesignatedContinueBlock(MI);

    auto Output = std::make_unique<DivergentConstruct>();
    Output->Header = BB;
    Output->Merge = Merge;
    Output->Continue = Continue;
    Output->Parent = Parent;

    constructDivergentConstruct(Visited, S, Merge, Parent);
    if (Continue)
      constructDivergentConstruct(Visited, S, Continue, Output.get());

    for (BasicBlock *Successor : successors(BB))
      constructDivergentConstruct(Visited, S, Successor, Output.get());

    if (Parent)
      Parent->Children.emplace_back(std::move(Output));
  }

  // Returns the blocks belonging to the divergent construct |Node|.
  BlockSet getConstructBlocks(Splitter &S, DivergentConstruct *Node) {
    assert(Node->Header && Node->Merge);

    if (Node->Continue) {
      auto LoopBlocks = S.getLoopConstructBlocks(Node->Header, Node->Merge);
      return BlockSet(LoopBlocks.begin(), LoopBlocks.end());
    }

    auto SelectionBlocks = S.getSelectionConstructBlocks(Node);
    return BlockSet(SelectionBlocks.begin(), SelectionBlocks.end());
  }

  // Fixup the construct |Node| to respect a set of rules defined by the SPIR-V
  // spec.
  bool fixupConstruct(Splitter &S, DivergentConstruct *Node) {
    bool Modified = false;
    for (auto &Child : Node->Children)
      Modified |= fixupConstruct(S, Child.get());

    // This construct is the root construct. Does not represent any real
    // construct, just a way to access the first level of the forest.
    if (Node->Parent == nullptr)
      return Modified;

    // This node's parent is the root. Meaning this is a top-level construct.
    // There can be multiple exists, but all are guaranteed to exit at most 1
    // construct since we are at first level.
    if (Node->Parent->Header == nullptr)
      return Modified;

    // Health check for the structure.
    assert(Node->Header && Node->Merge);
    assert(Node->Parent->Header && Node->Parent->Merge);

    BlockSet ConstructBlocks = getConstructBlocks(S, Node);
    auto Edges = getExitsFrom(ConstructBlocks, *Node->Header);

    //  No edges exiting the construct.
    if (Edges.size() < 1)
      return Modified;

    bool HasBadEdge = Node->Merge == Node->Parent->Merge ||
                      Node->Merge == Node->Parent->Continue;
    // BasicBlock *Target = Edges[0].second;
    for (auto &[Src, Dst] : Edges) {
      // - Breaking from a selection construct: S is a selection construct, S is
      // the innermost structured
      //   control-flow construct containing A, and B is the merge block for S
      // - Breaking from the innermost loop: S is the innermost loop construct
      // containing A,
      //   and B is the merge block for S
      if (Node->Merge == Dst)
        continue;

      // Entering the innermost loop’s continue construct: S is the innermost
      // loop construct containing A, and B is the continue target for S
      if (Node->Continue == Dst)
        continue;

      // TODO: what about cases branching to another case in the switch? Seems
      // to work, but need to double check.
      HasBadEdge = true;
    }

    if (!HasBadEdge)
      return Modified;

    // Create a single exit node gathering all exit edges.
    BasicBlock *NewExit = S.createSingleExitNode(Node->Header, Edges);

    // Fixup this construct's merge node to point to the new exit.
    // Note: this algorithm fixes inner-most divergence construct first. So
    // recursive structures sharing a single merge node are fixed from the
    // inside toward the outside.
    auto MergeInstructions = getMergeInstructions(*Node->Header);
    assert(MergeInstructions.size() == 1);
    Instruction *I = MergeInstructions[0];
    BlockAddress *BA = cast<BlockAddress>(I->getOperand(0));
    if (BA->getBasicBlock() == Node->Merge) {
      auto MergeAddress = BlockAddress::get(NewExit->getParent(), NewExit);
      I->setOperand(0, MergeAddress);
    }

    // Clean up of the possible dangling BockAddr operands to prevent MIR
    // comments about "address of removed block taken".
    if (!BA->isConstantUsed())
      BA->destroyConstant();

    Node->Merge = NewExit;
    // Regenerate the dom trees.
    S.invalidate();
    return true;
  }

  bool splitCriticalEdges(Function &F) {
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    Splitter S(F, LI);

    DivergentConstruct Root;
    BlockSet Visited;
    constructDivergentConstruct(Visited, S, &*F.begin(), &Root);
    return fixupConstruct(S, &Root);
  }

  // Simplify branches when possible:
  //  - if the 2 sides of a conditional branch are the same, transforms it to an
  //  unconditional branch.
  //  - if a switch has only 2 distinct successors, converts it to a conditional
  //  branch.
  bool simplifyBranches(Function &F) {
    bool Modified = false;

    for (BasicBlock &BB : F) {
      SwitchInst *SI = dyn_cast<SwitchInst>(BB.getTerminator());
      if (!SI)
        continue;
      if (SI->getNumCases() > 1)
        continue;

      Modified = true;
      IRBuilder<> Builder(&BB);
      Builder.SetInsertPoint(SI);

      if (SI->getNumCases() == 0) {
        Builder.CreateBr(SI->getDefaultDest());
      } else {
        Value *Condition =
            Builder.CreateCmp(CmpInst::ICMP_EQ, SI->getCondition(),
                              SI->case_begin()->getCaseValue());
        Builder.CreateCondBr(Condition, SI->case_begin()->getCaseSuccessor(),
                             SI->getDefaultDest());
      }
      SI->eraseFromParent();
    }

    return Modified;
  }

  // Makes sure every case target in |F| is unique. If 2 cases branch to the
  // same basic block, one of the targets is updated so it jumps to a new basic
  // block ending with a single unconditional branch to the original target.
  bool splitSwitchCases(Function &F) {
    bool Modified = false;

    for (BasicBlock &BB : F) {
      SwitchInst *SI = dyn_cast<SwitchInst>(BB.getTerminator());
      if (!SI)
        continue;

      BlockSet Seen;
      Seen.insert(SI->getDefaultDest());

      auto It = SI->case_begin();
      while (It != SI->case_end()) {
        BasicBlock *Target = It->getCaseSuccessor();
        if (Seen.count(Target) == 0) {
          Seen.insert(Target);
          ++It;
          continue;
        }

        Modified = true;
        BasicBlock *NewTarget =
            BasicBlock::Create(F.getContext(), "new.sw.case", &F);
        IRBuilder<> Builder(NewTarget);
        Builder.CreateBr(Target);
        SI->addCase(It->getCaseValue(), NewTarget);
        It = SI->removeCase(It);
      }
    }

    return Modified;
  }

  // Removes blocks not contributing to any structured CFG. This assumes there
  // is no PHI nodes.
  bool removeUselessBlocks(Function &F) {
    std::vector<BasicBlock *> ToRemove;

    auto MergeBlocks = getMergeBlocks(F);
    auto ContinueBlocks = getContinueBlocks(F);

    for (BasicBlock &BB : F) {
      if (BB.size() != 1)
        continue;

      if (isa<ReturnInst>(BB.getTerminator()))
        continue;

      if (MergeBlocks.count(&BB) != 0 || ContinueBlocks.count(&BB) != 0)
        continue;

      if (BB.getUniqueSuccessor() == nullptr)
        continue;

      BasicBlock *Successor = BB.getUniqueSuccessor();
      std::vector<BasicBlock *> Predecessors(predecessors(&BB).begin(),
                                             predecessors(&BB).end());
      for (BasicBlock *Predecessor : Predecessors)
        replaceBranchTargets(Predecessor, &BB, Successor);
      ToRemove.push_back(&BB);
    }

    for (BasicBlock *BB : ToRemove)
      BB->eraseFromParent();

    return ToRemove.size() != 0;
  }

  bool addHeaderToRemainingDivergentDAG(Function &F) {
    bool Modified = false;

    auto MergeBlocks = getMergeBlocks(F);
    auto ContinueBlocks = getContinueBlocks(F);
    auto HeaderBlocks = getHeaderBlocks(F);

    DomTreeBuilder::BBDomTree DT;
    DomTreeBuilder::BBPostDomTree PDT;
    PDT.recalculate(F);
    DT.recalculate(F);

    for (BasicBlock &BB : F) {
      if (HeaderBlocks.count(&BB) != 0)
        continue;
      if (succ_size(&BB) < 2)
        continue;

      size_t CandidateEdges = 0;
      for (BasicBlock *Successor : successors(&BB)) {
        if (MergeBlocks.count(Successor) != 0 ||
            ContinueBlocks.count(Successor) != 0)
          continue;
        if (HeaderBlocks.count(Successor) != 0)
          continue;
        CandidateEdges += 1;
      }

      if (CandidateEdges <= 1)
        continue;

      BasicBlock *Header = &BB;
      BasicBlock *Merge = PDT.getNode(&BB)->getIDom()->getBlock();

      bool HasBadBlock = false;
      visit(*Header, [&](const BasicBlock *Node) {
        if (DT.dominates(Header, Node))
          return false;
        if (PDT.dominates(Merge, Node))
          return false;
        if (Node == Header || Node == Merge)
          return true;

        HasBadBlock |= MergeBlocks.count(Node) != 0 ||
                       ContinueBlocks.count(Node) != 0 ||
                       HeaderBlocks.count(Node) != 0;
        return !HasBadBlock;
      });

      if (HasBadBlock)
        continue;

      Modified = true;

      if (Merge == nullptr) {
        Merge = *successors(Header).begin();
        IRBuilder<> Builder(Header);
        Builder.SetInsertPoint(Header->getTerminator());

        auto MergeAddress = BlockAddress::get(Merge->getParent(), Merge);
        createOpSelectMerge(&Builder, MergeAddress);
        continue;
      }

      Instruction *SplitInstruction = Merge->getTerminator();
      if (isMergeInstruction(SplitInstruction->getPrevNode()))
        SplitInstruction = SplitInstruction->getPrevNode();
      BasicBlock *NewMerge =
          Merge->splitBasicBlockBefore(SplitInstruction, "new.merge");

      IRBuilder<> Builder(Header);
      Builder.SetInsertPoint(Header->getTerminator());

      auto MergeAddress = BlockAddress::get(NewMerge->getParent(), NewMerge);
      createOpSelectMerge(&Builder, MergeAddress);
    }

    return Modified;
  }

public:
  static char ID;

  SPIRVStructurizer() : FunctionPass(ID) {}

  virtual bool runOnFunction(Function &F) override {
    bool Modified = false;

    // In LLVM, Switches are allowed to have several cases branching to the same
    // basic block. This is allowed in SPIR-V, but can make structurizing SPIR-V
    // harder, so first remove edge cases.
    Modified |= splitSwitchCases(F);

    // LLVM allows conditional branches to have both side jumping to the same
    // block. It also allows switched to have a single default, or just one
    // case. Cleaning this up now.
    Modified |= simplifyBranches(F);

    // At this state, we should have a reducible CFG with cycles.
    // STEP 1: Adding OpLoopMerge instructions to loop headers.
    Modified |= addMergeForLoops(F);

    // STEP 2: adding OpSelectionMerge to each node with an in-degree >= 2.
    Modified |= addMergeForNodesWithMultiplePredecessors(F);

    // STEP 3:
    // Sort selection merge, the largest construct goes first.
    // This simplifies the next step.
    Modified |= sortSelectionMergeHeaders(F);

    // STEP 4: As this stage, we can have a single basic block with multiple
    // OpLoopMerge/OpSelectionMerge instructions. Splitting this block so each
    // BB has a single merge instruction.
    Modified |= splitBlocksWithMultipleHeaders(F);

    // STEP 5: In the previous steps, we added merge blocks the loops and
    // natural merge blocks (in-degree >= 2). What remains are conditions with
    // an exiting branch (return, unreachable). In such case, we must start from
    // the header, and add headers to divergent construct with no headers.
    Modified |= addMergeForDivergentBlocks(F);

    // STEP 6: At this stage, we have several divergent construct defines by a
    // header and a merge block. But their boundaries have no constraints: a
    // construct exit could be outside of the parents' construct exit. Such
    // edges are called critical edges. What we need is to split those edges
    // into several parts. Each part exiting the parent's construct by its merge
    // block.
    Modified |= splitCriticalEdges(F);

    // STEP 7: The previous steps possibly created a lot of "proxy" blocks.
    // Blocks with a single unconditional branch, used to create a valid
    // divergent construct tree. Some nodes are still requires (e.g: nodes
    // allowing a valid exit through the parent's merge block). But some are
    // left-overs of past transformations, and could cause actual validation
    // issues. E.g: the SPIR-V spec allows a construct to break to the parents
    // loop construct without an OpSelectionMerge, but this requires a straight
    // jump. If a proxy block lies between the conditional branch and the
    // parent's merge, the CFG is not valid.
    Modified |= removeUselessBlocks(F);

    // STEP 8: Final fix-up steps: our tree boundaries are correct, but some
    // blocks are branching with no header. Those are often simple conditional
    // branches with 1 or 2 returning edges. Adding a header for those.
    Modified |= addHeaderToRemainingDivergentDAG(F);

    // STEP 9: sort basic blocks to match both the LLVM & SPIR-V requirements.
    Modified |= sortBlocks(F);

    return Modified;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<SPIRVConvergenceRegionAnalysisWrapperPass>();

    AU.addPreserved<SPIRVConvergenceRegionAnalysisWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

  void createOpSelectMerge(IRBuilder<> *Builder, BlockAddress *MergeAddress) {
    Instruction *BBTerminatorInst = Builder->GetInsertBlock()->getTerminator();

    MDNode *MDNode = BBTerminatorInst->getMetadata("hlsl.controlflow.hint");

    ConstantInt *BranchHint = ConstantInt::get(Builder->getInt32Ty(), 0);

    if (MDNode) {
      assert(MDNode->getNumOperands() == 2 &&
             "invalid metadata hlsl.controlflow.hint");
      BranchHint = mdconst::extract<ConstantInt>(MDNode->getOperand(1));
    }

    SmallVector<Value *, 2> Args = {MergeAddress, BranchHint};

    Builder->CreateIntrinsic(Intrinsic::spv_selection_merge,
                             {MergeAddress->getType()}, Args);
  }
};
} // anonymous namespace

char SPIRVStructurizer::ID = 0;

INITIALIZE_PASS_BEGIN(SPIRVStructurizer, "spirv-structurizer",
                      "structurize SPIRV", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SPIRVConvergenceRegionAnalysisWrapperPass)

INITIALIZE_PASS_END(SPIRVStructurizer, "spirv-structurizer",
                    "structurize SPIRV", false, false)

FunctionPass *llvm::createSPIRVStructurizerPass() {
  return new SPIRVStructurizer();
}

PreservedAnalyses SPIRVStructurizerWrapper::run(Function &F,
                                                FunctionAnalysisManager &AF) {

  auto FPM = legacy::FunctionPassManager(F.getParent());
  FPM.add(createSPIRVStructurizerPass());

  if (!FPM.run(F))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
