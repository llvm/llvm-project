//===-- SPIRVStructurizer.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass adds the required OpSelection/OpLoop merge instructions to
// generate valid SPIR-V.
// This pass trims convergence intrinsics as those were only useful when
// modifying the CFG during IR passes.
//
//===----------------------------------------------------------------------===//

#include "Analysis/SPIRVConvergenceRegionAnalysis.h"
#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/InitializePasses.h"
#include <stack>

using namespace llvm;
using namespace SPIRV;

namespace llvm {
void initializeSPIRVStructurizerPass(PassRegistry &);
}

namespace {

// Returns the exact convergence region in the tree defined by `Node` for which
// `MBB` is the header, nullptr otherwise.
const ConvergenceRegion *getRegionForHeader(const ConvergenceRegion *Node,
                                            MachineBasicBlock *MBB) {
  if (Node->Entry == MBB->getBasicBlock())
    return Node;

  for (auto *Child : Node->Children) {
    const auto *CR = getRegionForHeader(Child, MBB);
    if (CR != nullptr)
      return CR;
  }
  return nullptr;
}

// Returns the MachineBasicBlock in `MF` matching `BB`, nullptr otherwise.
MachineBasicBlock *getMachineBlockFor(MachineFunction &MF, BasicBlock *BB) {
  for (auto &MBB : MF)
    if (MBB.getBasicBlock() == BB)
      return &MBB;
  return nullptr;
}

// Gather all the successors of |BB|.
// This function asserts if the terminator neither a branch, switch or return.
std::unordered_set<BasicBlock *> gatherSuccessors(BasicBlock *BB) {
  std::unordered_set<BasicBlock *> Output;
  auto *T = BB->getTerminator();

  if (auto *BI = dyn_cast<BranchInst>(T)) {
    Output.insert(BI->getSuccessor(0));
    if (BI->isConditional())
      Output.insert(BI->getSuccessor(1));
    return Output;
  }

  if (auto *SI = dyn_cast<SwitchInst>(T)) {
    Output.insert(SI->getDefaultDest());
    for (auto &Case : SI->cases())
      Output.insert(Case.getCaseSuccessor());
    return Output;
  }

  if (auto *RI = dyn_cast<ReturnInst>(T))
    return Output;

  assert(false && "Unhandled terminator type.");
  return Output;
}

// Returns the single MachineBasicBlock exiting the convergence region `CR`,
// nullptr if no such exit exists. MF must be the function CR belongs to.
MachineBasicBlock *getExitFor(MachineFunction &MF,
                              const ConvergenceRegion *CR) {
  std::unordered_set<BasicBlock *> ExitTargets;
  for (BasicBlock *Exit : CR->Exits) {
    for (BasicBlock *Target : gatherSuccessors(Exit)) {
      if (CR->Blocks.count(Target) == 0)
        ExitTargets.insert(Target);
    }
  }

  assert(ExitTargets.size() <= 1);
  if (ExitTargets.size() == 0)
    return nullptr;

  auto *Exit = *ExitTargets.begin();
  return getMachineBlockFor(MF, Exit);
}

// Returns true is |I| is a OpLoopMerge or OpSelectionMerge instruction.
bool isMergeInstruction(const MachineInstr &I) {
  return I.getOpcode() == SPIRV::OpLoopMerge ||
         I.getOpcode() == SPIRV::OpSelectionMerge;
}

// Returns the first OpLoopMerge/OpSelectionMerge instruction found in |MBB|,
// nullptr otherwise.
MachineInstr *getMergeInstruction(MachineBasicBlock &MBB) {
  for (auto &I : MBB) {
    if (isMergeInstruction(I))
      return &I;
  }
  return nullptr;
}

// Returns the first OpLoopMerge instruction found in |MBB|, nullptr otherwise.
MachineInstr *getLoopMergeInstruction(MachineBasicBlock &MBB) {
  for (auto &I : MBB) {
    if (I.getOpcode() == SPIRV::OpLoopMerge)
      return &I;
  }
  return nullptr;
}

// Returns the first OpSelectionMerge instruction found in |MBB|, nullptr
// otherwise.
MachineInstr *getSelectionMergeInstruction(MachineBasicBlock &MBB) {
  for (auto &I : MBB) {
    if (I.getOpcode() == SPIRV::OpSelectionMerge)
      return &I;
  }
  return nullptr;
}

// Do a preorder traversal of the CFG starting from the given function's entry
// point. Calls |op| on each basic block encountered during the traversal.
void visit(MachineFunction &MF, std::function<void(MachineBasicBlock *)> op) {
  std::stack<MachineBasicBlock *> ToVisit;
  SmallPtrSet<MachineBasicBlock *, 8> Seen;

  ToVisit.push(&*MF.begin());
  Seen.insert(ToVisit.top());
  while (ToVisit.size() != 0) {
    MachineBasicBlock *MBB = ToVisit.top();
    ToVisit.pop();

    op(MBB);

    for (auto Succ : MBB->successors()) {
      if (Seen.contains(Succ))
        continue;
      ToVisit.push(Succ);
      Seen.insert(Succ);
    }
  }
}

// Returns all basic blocks in |MF| with at least one SelectionMerge/LoopMerge
// instruction.
SmallPtrSet<MachineBasicBlock *, 8> getHeaderBlocks(MachineFunction &MF) {
  SmallPtrSet<MachineBasicBlock *, 8> Output;
  for (MachineBasicBlock &MBB : MF) {
    auto *MI = getMergeInstruction(MBB);
    if (MI != nullptr)
      Output.insert(&MBB);
  }
  return Output;
}

// Returns all basic blocks in |MF| referenced by at least 1
// OpSelectionMerge/OpLoopMerge instruction.
SmallPtrSet<MachineBasicBlock *, 8> getMergeBlocks(MachineFunction &MF) {
  SmallPtrSet<MachineBasicBlock *, 8> Output;
  for (MachineBasicBlock &MBB : MF) {
    auto *MI = getMergeInstruction(MBB);
    if (MI != nullptr)
      Output.insert(MI->getOperand(0).getMBB());
  }
  return Output;
}

// Returns all basic blocks in |MF| referenced as continue target by at least 1
// OpLoopMerge.
SmallPtrSet<MachineBasicBlock *, 8> getContinueBlocks(MachineFunction &MF) {
  SmallPtrSet<MachineBasicBlock *, 8> Output;
  for (MachineBasicBlock &MBB : MF) {
    auto *MI = getMergeInstruction(MBB);
    if (MI != nullptr && MI->getOpcode() == SPIRV::OpLoopMerge)
      Output.insert(MI->getOperand(1).getMBB());
  }
  return Output;
}

// Returns the block immediatly post-dominating every block in |Range| if any,
// nullptr otherwise.
MachineBasicBlock *findNearestCommonPostDominator(
    const iterator_range<std::vector<MachineBasicBlock *>::iterator> &Range,
    MachinePostDominatorTree &MPDT) {
  assert(!Range.empty());
  MachineBasicBlock *Dom = *Range.begin();
  for (MachineBasicBlock *Item : Range)
    Dom = MPDT.findNearestCommonDominator(Dom, Item);
  return Dom;
}

// Finds the first merge instruction in |MBB| and store it in |MI|.
// If it defines a merge target, sets |Merge| to the merge target.
// If it defines a continue target, sets |Continue| to the continue target.
// Returns true if such merge instruction was found, false otherwise.
bool getMergeInstructionTargets(MachineBasicBlock *MBB, MachineInstr **MI,
                                MachineBasicBlock **Merge,
                                MachineBasicBlock **Continue) {
  *Merge = nullptr;
  *Continue = nullptr;

  *MI = getMergeInstruction(*MBB);
  if (*MI == nullptr)
    return false;

  *Merge = (*MI)->getOperand(0).getMBB();
  *Continue = (*MI)->getOpcode() == SPIRV::OpLoopMerge
                  ? (*MI)->getOperand(1).getMBB()
                  : nullptr;
  return true;
}

} // anonymous namespace

class SPIRVStructurizer : public MachineFunctionPass {
public:
  static char ID;

  SPIRVStructurizer() : MachineFunctionPass(ID) {
    initializeSPIRVStructurizerPass(*PassRegistry::getPassRegistry());
  };

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<SPIRVConvergenceRegionAnalysisWrapperPass>();
  }

  // Creates a MachineBasicBlock ending with an OpReturn just after
  // |Predecessor|. This function does not add any branch to |Predecessor|, but
  // adds the new block to its successors.
  MachineBasicBlock *createReturnBlock(MachineFunction &MF,
                                       MachineBasicBlock &Predecessor) {
    MachineBasicBlock *MBB =
        MF.CreateMachineBasicBlock(Predecessor.getBasicBlock());
    MF.push_back(MBB);
    MBB->moveAfter(&Predecessor);
    // This code doesn't add a branch instruction to this new return block.
    // The caller will have to handle that.
    Predecessor.addSuccessorWithoutProb(MBB);

    MachineIRBuilder MIRBuilder(MF);
    MIRBuilder.setInsertPt(*MBB, MBB->end());
    MIRBuilder.buildInstr(SPIRV::OpReturn);

    return MBB;
  }

  // Replace switches with a single target with an unconditional branch.
  bool replaceEmptySwitchWithBranch(MachineFunction &MF) {
    bool Modified = false;
    for (MachineBasicBlock &MBB : MF) {
      MachineInstr *I = &*MBB.rbegin();
      GIntrinsic *II = dyn_cast<GIntrinsic>(I);
      if (!II || II->getIntrinsicID() != Intrinsic::spv_switch ||
          II->getNumOperands() > 3)
        continue;

      Modified = true;
      assert(II->getOperand(2).isMBB());
      MachineBasicBlock *Target = II->getOperand(2).getMBB();

      MachineIRBuilder MIRBuilder(MF);
      MIRBuilder.setInsertPt(MBB, MBB.end());
      MIRBuilder.buildBr(*Target);
      MBB.erase(I);
    }

    return Modified;
  }

  // Traverse each loop, and adds an OpLoopMerge instruction to its header
  // that respect the convergence region node it belongs to.
  // The Continue target is the only back-edge in that loop.
  // The merge target is the only exiting node of the convergence region.
  bool addMergeForLoops(MachineFunction &MF) {
    auto &TII = *MF.getSubtarget<SPIRVSubtarget>().getInstrInfo();
    auto &TRI = *MF.getSubtarget<SPIRVSubtarget>().getRegisterInfo();
    auto &RBI = *MF.getSubtarget<SPIRVSubtarget>().getRegBankInfo();

    const auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    const auto *TopLevelRegion =
        getAnalysis<SPIRVConvergenceRegionAnalysisWrapperPass>()
            .getRegionInfo()
            .getTopLevelRegion();

    bool Modified = false;
    for (auto &MBB : MF) {
      // Not a loop header. Ignoring for now.
      if (!MLI.isLoopHeader(&MBB))
        continue;
      auto *L = MLI.getLoopFor(&MBB);

      // This loop header is not the entrance of a convergence region. Ignoring
      // this block.
      auto *CR = getRegionForHeader(TopLevelRegion, &MBB);
      if (CR == nullptr)
        continue;

      auto *Merge = getExitFor(MF, CR);
      // This is a special case:
      // We are indeed in a loop, but there are no exits (infinite loop).
      // This means the actual branch is unconditional, hence won't require any
      // OpLoopMerge.
      if (Merge == nullptr) {
        Merge = createReturnBlock(MF, MBB);
      }

      auto *Continue = L->getLoopLatch();

      // Conditional branch are built using a fallthrough if false + BR.
      // So the last instruction is not always the first branch.
      auto *I = &*MBB.getFirstTerminator();
      BuildMI(MBB, I, I->getDebugLoc(), TII.get(SPIRV::OpLoopMerge))
          .addMBB(Merge)
          .addMBB(Continue)
          .addImm(SPIRV::SelectionControl::None)
          .constrainAllUses(TII, TRI, RBI);
      Modified = true;
    }

    return Modified;
  }

  // Add an OpSelectionMerge to each node with an out-degree of 2 or more.
  bool addMergeForConditionalBranches(MachineFunction &MF) {
    MachinePostDominatorTree MPDT(MF);

    auto &TII = *MF.getSubtarget<SPIRVSubtarget>().getInstrInfo();
    auto &TRI = *MF.getSubtarget<SPIRVSubtarget>().getRegisterInfo();
    auto &RBI = *MF.getSubtarget<SPIRVSubtarget>().getRegBankInfo();

    auto MergeBlocks = getMergeBlocks(MF);
    auto ContinueBlocks = getContinueBlocks(MF);

    for (auto &MBB : MF) {
      if (MBB.succ_size() <= 1)
        continue;

      // Block already has an OpSelectionMerge instruction. Ignoring.
      if (getSelectionMergeInstruction(MBB)) {
        continue;
      }

      assert(MBB.succ_size() >= 2);
      size_t NonStructurizedTargets = 0;
      for (MachineBasicBlock *Successor : MBB.successors()) {
        if (!MergeBlocks.contains(Successor) &&
            !ContinueBlocks.contains(Successor))
          NonStructurizedTargets += 1;
      }

      if (NonStructurizedTargets <= 1)
        continue;

      MachineBasicBlock *Merge =
          findNearestCommonPostDominator(MBB.successors(), MPDT);
      if (!Merge) {
        // TODO: we should check which block is not another construct merge
        // block, and select this one. For now, tests passes with this strategy,
        // but once we find a test case, we should fix that.
        Merge = *MBB.succ_begin();
      }

      assert(Merge);
      auto *II = MBB.getFirstTerminator() == MBB.end()
                     ? &*MBB.rbegin()
                     : &*MBB.getFirstTerminator();
      BuildMI(MBB, II, II->getDebugLoc(), TII.get(SPIRV::OpSelectionMerge))
          .addMBB(Merge)
          .addImm(SPIRV::SelectionControl::None)
          .constrainAllUses(TII, TRI, RBI);
    }

    return false;
  }

  // Cut |Block| just after the first OpLoopMerge/OpSelectionMerge instruction.
  // The newly created block lies just after |Block|, and |Block| branches
  // unconditionally to this new block. Returns the newly created block.
  MachineBasicBlock *splitHeaderBlock(MachineFunction &MF,
                                      MachineBasicBlock &Block) {
    auto FirstMerge = Block.begin();
    while (!isMergeInstruction(*FirstMerge)) {
      FirstMerge++;
    }

    MachineBasicBlock *NewBlock = Block.splitAt(*FirstMerge);

    MachineIRBuilder MIRBuilder(MF);
    MIRBuilder.setInsertPt(Block, Block.end());
    MIRBuilder.buildBr(*NewBlock);

    return NewBlock;
  }

  // Split basic blocks containing multiple OpLoopMerge/OpSelectionMerge
  // instructions so each basic block contains only a single merge instruction.
  bool splitBlocksWithMultipleHeaders(MachineFunction &MF) {
    bool Modified = false;
    for (auto &MBB : MF) {
      MachineInstr *SelectionMerge = getSelectionMergeInstruction(MBB);
      MachineInstr *LoopMerge = getLoopMergeInstruction(MBB);
      if (!SelectionMerge || !LoopMerge) {
        continue;
      }

      splitHeaderBlock(MF, MBB);
      Modified = true;
    }
    return Modified;
  }

  // Splits the basic block |OldMerge| in two.
  // The newly created block will become the predecessor of |OldMerge|.
  // |HeaderBlock| becomes the only block using |OldMerge| as merge target.
  // Each other Merge instruction having |OldMerge| as target will have the
  // newly created block as target.
  MachineBasicBlock *splitMergeBlock(MachineDominatorTree &MDT,
                                     MachineFunction &MF,
                                     MachineBasicBlock &OldMerge,
                                     MachineBasicBlock &HeaderBlock) {

    std::vector<MachineBasicBlock *> toUpdate;
    for (MachineBasicBlock *Predecessor : OldMerge.predecessors())
      toUpdate.push_back(Predecessor);

    MachineBasicBlock *NewMerge =
        MF.CreateMachineBasicBlock(OldMerge.getBasicBlock());
    MF.push_back(NewMerge);
    NewMerge->moveBefore(&OldMerge);
    NewMerge->addSuccessorWithoutProb(&OldMerge);
    MachineIRBuilder MIRBuilder(MF);
    MIRBuilder.setInsertPt(*NewMerge, NewMerge->end());
    MIRBuilder.buildBr(OldMerge);

    for (MachineBasicBlock *Predecessor : toUpdate) {
      if (!MDT.dominates(&HeaderBlock, Predecessor))
        continue;

      OldMerge.replacePhiUsesWith(Predecessor, NewMerge);
      Predecessor->removeSuccessor(&OldMerge);
      Predecessor->addSuccessorWithoutProb(NewMerge);
      for (auto &I : *Predecessor) {
        for (auto &O : I.operands()) {
          if (O.isMBB() && O.getMBB() == &OldMerge)
            O.setMBB(NewMerge);
        }
      }
    }

    auto *MI = getMergeInstruction(HeaderBlock);
    assert(MI);
    MI->getOperand(0).setMBB(NewMerge);

    return NewMerge;
  }

  // Modifies the CFG to make sure each merge block is the target of a single
  // header.
  bool splitMergeBlocks(MachineFunction &MF) {
    MachineDominatorTree MDT(MF);

    // Determine all the blocks we need to analyse.
    auto HeaderBlocks = getHeaderBlocks(MF);
    // Visit the CFG DFS-style to process header blocks.
    std::vector<MachineBasicBlock *> ToProcess;
    visit(MF, [&ToProcess, &HeaderBlocks](MachineBasicBlock *MBB) {
      if (HeaderBlocks.count(MBB) != 0)
        ToProcess.push_back(MBB);
    });

    // Maps each merge-block to its associated header block.
    std::unordered_map<MachineBasicBlock *, MachineBasicBlock *> MergeToHeader;
    bool Modified = false;
    for (auto *MBB : ToProcess) {
      auto *MI = getMergeInstruction(*MBB);
      assert(MI != nullptr);

      auto *Merge = MI->getOperand(0).getMBB();
      // If the merge block hasn't been seen yet, no conflict.
      if (MergeToHeader.count(Merge) == 0) {
        MergeToHeader.emplace(Merge, MBB);
        continue;
      }

      // Otherwise, we need to split the merge block, and update the references.
      Modified = true;
      MachineBasicBlock *ConflictingHeader = MergeToHeader[Merge];
      MachineBasicBlock *NewMerge = splitMergeBlock(MDT, MF, *Merge, *MBB);
      // Each selection/loop construct that is not already processed (hence
      // deeper in the CFG) is updated to use the new merge. Conflicts are
      // resolved layer by layer.
      for (auto *Header : HeaderBlocks) {
        if (Header == ConflictingHeader)
          continue;
        auto *Instr = getMergeInstruction(*Header);
        if (Instr->getOperand(0).getMBB() == Merge)
          Instr->getOperand(0).setMBB(NewMerge);
      }
      MergeToHeader.emplace(NewMerge, MBB);
    }
    return Modified;
  }

  // Modifies the CFG to make sure the same block is not both a continue target,
  // and a merge target.
  bool splitMergeAndContinueBlocks(MachineFunction &MF) {
    MachineDominatorTree MDT(MF);
    std::vector<MachineBasicBlock *> toProcess;
    visit(MF,
          [&toProcess](MachineBasicBlock *MBB) { toProcess.push_back(MBB); });

    auto ContinueBlocks = getContinueBlocks(MF);
    bool Modified = false;
    for (auto *MBB : toProcess) {
      MachineBasicBlock *Merge = nullptr;
      MachineBasicBlock *Continue = nullptr;
      MachineInstr *MI = nullptr;
      if (!getMergeInstructionTargets(MBB, &MI, &Merge, &Continue))
        continue;

      if (ContinueBlocks.count(Merge) == 0)
        continue;

      // This blocks' merge is another block's continue.
      Modified = true;
      MachineBasicBlock *NewMerge = splitMergeBlock(MDT, MF, *Merge, *MBB);
      MI->getOperand(0).setMBB(NewMerge);
    }
    return Modified;
  }

  // Sorts basic blocks by dominance to respect the SPIR-V spec.
  bool sortBlocks(MachineFunction &MF) {
    MachineDominatorTree MDT(MF);

    std::unordered_map<MachineBasicBlock *, size_t> Order;
    size_t Index = 0;
    visit(MF,
          [&Order, &Index](MachineBasicBlock *MBB) { Order[MBB] = Index++; });

    auto Comparator = [&Order](MachineBasicBlock &LHS, MachineBasicBlock &RHS) {
      return Order[&LHS] < Order[&RHS];
    };

    MF.sort(Comparator);
    // FIXME: need to check if the order changed. Maybe if the comparator
    // returns false once, it did?
    return true;
  }

  // In some cases, divergence is allowed without any OpSelectionMerge
  // instruction because paths only one path doesn't end-up to the parent's
  // selection/loop construct merge. Example:
  //          A
  //         / \
  //        B<--C
  //        \    \
  //         \    D
  //          \  /
  //           E
  // In this case, E is A's merge node.
  // Previous steps marked C as a selection construct header because it has an
  // out-degree of 2. But the thread divergence state cannot merge earlier: due
  // to this triangle configuration, there is no earlier merge node than E. This
  // means we are still in the same selection construct, hence don't require a
  // new OpSelectionMerge.
  bool removeSuperfluousSelectionHeaders(MachineFunction &MF) {
    MachineDominatorTree MDT(MF);

    bool Modified = false;
    for (MachineBasicBlock &MBB : MF) {
      MachineInstr *MI = getMergeInstruction(MBB);
      if (MI == nullptr)
        continue;
      // This doesn't apply to block targeted by a back-edge.
      if (MI->getOpcode() == SPIRV::OpLoopMerge)
        continue;

      size_t Dominated_count = 0;
      for (auto *Successor : MBB.successors()) {
        if (MDT.dominates(&MBB, Successor))
          Dominated_count += 1;
      }

      if (Dominated_count > 1)
        continue;

      MBB.erase(MI);
      Modified = true;
    }

    return Modified;
  }

  virtual bool runOnMachineFunction(MachineFunction &MF) override {
    bool Modified = false;

    Modified |= replaceEmptySwitchWithBranch(MF);
    Modified |= addMergeForLoops(MF);
    Modified |= addMergeForConditionalBranches(MF);
    Modified |= splitBlocksWithMultipleHeaders(MF);
    Modified |= splitMergeBlocks(MF);
    Modified |= splitMergeAndContinueBlocks(MF);
    Modified |= removeSuperfluousSelectionHeaders(MF);
    Modified |= sortBlocks(MF);

    return Modified;
  }
};

char SPIRVStructurizer::ID = 0;

INITIALIZE_PASS_BEGIN(SPIRVStructurizer, "structurizer", "SPIRV structurizer",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SPIRVStructurizer, "structurizer", "SPIRV structurizer",
                    false, false)

FunctionPass *llvm::createSPIRVStructurizerPass() {
  return new SPIRVStructurizer();
}
