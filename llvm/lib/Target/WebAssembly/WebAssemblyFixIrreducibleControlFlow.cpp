//=- WebAssemblyFixIrreducibleControlFlow.cpp - Fix irreducible control flow -//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a pass that removes irreducible control flow.
/// Irreducible control flow means multiple-entry loops, which this pass
/// transforms to have a single entry.
///
/// Note that LLVM has a generic pass that lowers irreducible control flow, but
/// it linearizes control flow, turning diamonds into two triangles, which is
/// both unnecessary and undesirable for WebAssembly.
///
/// The big picture: We recursively process each "region", defined as a group
/// of blocks with a single entry and no branches back to that entry. A region
/// may be the entire function body, or the inner part of a loop, i.e., the
/// loop's body without branches back to the loop entry. In each region we
/// identify all the strongly-connected components (SCCs). We fix up multi-entry
/// loops (SCCs) by adding a new block that can dispatch to each of the loop
/// entries, based on the value of a label "helper" variable, and we replace
/// direct branches to the entries with assignments to the label variable and a
/// branch to the dispatch block. Then the dispatch block is the single entry in
/// the loop containing the previous multiple entries. Each time we fix some
/// irreducibility, we recalculate the SCCs. After ensuring all the SCCs in a
/// region are reducible, we recurse into them. The total time complexity of
/// this pass is:
///
///   O(NumBlocks * NumNestedLoops * NumIrreducibleLoops +
///     NumLoops * NumLoops)
///
/// This pass is similar to what the Relooper [1] does. Both identify looping
/// code that requires multiple entries, and resolve it in a similar way (in
/// Relooper terminology, we implement a Multiple shape in a Loop shape). Note
/// also that like the Relooper, we implement a "minimal" intervention: we only
/// use the "label" helper for the blocks we absolutely must and no others. We
/// also prioritize code size and do not duplicate code in order to resolve
/// irreducibility. The graph algorithms for finding loops and entries and so
/// forth are also similar to the Relooper. The main differences between this
/// pass and the Relooper are:
///
///  * We just care about irreducibility, so we just look at loops.
///  * The Relooper emits structured control flow (with ifs etc.), while we
///    emit a CFG.
///
/// [1] Alon Zakai. 2011. Emscripten: an LLVM-to-JavaScript compiler. In
/// Proceedings of the ACM international conference companion on Object oriented
/// programming systems languages and applications companion (SPLASH '11). ACM,
/// New York, NY, USA, 301-312. DOI=10.1145/2048147.2048224
/// http://doi.acm.org/10.1145/2048147.2048224
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
#include <limits>
using namespace llvm;

#define DEBUG_TYPE "wasm-fix-irreducible-control-flow"

namespace {

using BlockVector = SmallVector<MachineBasicBlock *, 4>;
using BlockSet = SmallPtrSet<MachineBasicBlock *, 4>;

static BlockVector getSortedEntries(const BlockSet &Entries) {
  BlockVector SortedEntries(Entries.begin(), Entries.end());
  llvm::sort(SortedEntries,
             [](const MachineBasicBlock *A, const MachineBasicBlock *B) {
               auto ANum = A->getNumber();
               auto BNum = B->getNumber();
               return ANum < BNum;
             });
  return SortedEntries;
}

struct ReachabilityNode {
  MachineBasicBlock *MBB;
  SmallVector<ReachabilityNode *, 4> Succs;
  unsigned SCCId = std::numeric_limits<unsigned>::max();
};

// Analyzes the SCC (strongly-connected component) structure in a region.
// Ignores branches to blocks outside of the region, and ignores branches to the
// region entry (for the case where the region is the inner part of a loop).
class ReachabilityGraph {
public:
  ReachabilityGraph(MachineBasicBlock *Entry, const BlockSet &Blocks)
      : Entry(Entry), Blocks(Blocks) {
#ifndef NDEBUG
    // The region must have a single entry.
    for (auto *MBB : Blocks) {
      if (MBB != Entry) {
        for (auto *Pred : MBB->predecessors()) {
          assert(inRegion(Pred));
        }
      }
    }
#endif
    calculate();
  }

  // Get all blocks that are loop entries.
  const BlockSet &getLoopEntries() const { return LoopEntries; }
  const BlockSet &getLoopEntriesForSCC(unsigned SCCId) const {
    return LoopEntriesBySCC[SCCId];
  }

  unsigned getSCCId(MachineBasicBlock *MBB) const {
    return getNode(MBB)->SCCId;
  }

  friend struct GraphTraits<ReachabilityGraph *>;

private:
  MachineBasicBlock *Entry;
  const BlockSet &Blocks;

  BlockSet LoopEntries;
  SmallVector<BlockSet, 0> LoopEntriesBySCC;

  bool inRegion(MachineBasicBlock *MBB) const { return Blocks.count(MBB); }

  SmallVector<ReachabilityNode, 0> Nodes;
  DenseMap<MachineBasicBlock *, ReachabilityNode *> MBBToNodeMap;

  ReachabilityNode *getNode(MachineBasicBlock *MBB) const {
    return MBBToNodeMap.at(MBB);
  }

  void calculate();
};
} // end anonymous namespace

namespace llvm {
template <> struct GraphTraits<ReachabilityGraph *> {
  using NodeRef = ReachabilityNode *;
  using ChildIteratorType = SmallVectorImpl<NodeRef>::iterator;

  static NodeRef getEntryNode(ReachabilityGraph *G) {
    return G->getNode(G->Entry);
  }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->Succs.begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->Succs.end();
  }
};
} // end namespace llvm

namespace {

void ReachabilityGraph::calculate() {
  auto NumBlocks = Blocks.size();
  Nodes.assign(NumBlocks, {});

  MBBToNodeMap.clear();
  MBBToNodeMap.reserve(NumBlocks);

  // Initialize mappings.
  unsigned MBBIdx = 0;
  for (auto *MBB : Blocks) {
    auto &Node = Nodes[MBBIdx++];

    Node.MBB = MBB;
    MBBToNodeMap[MBB] = &Node;
  }

  // Add all relevant direct branches.
  MBBIdx = 0;
  for (auto *MBB : Blocks) {
    auto &Node = Nodes[MBBIdx++];

    for (auto *Succ : MBB->successors()) {
      if (Succ != Entry && inRegion(Succ)) {
        Node.Succs.push_back(getNode(Succ));
      }
    }
  }

  unsigned CurrSCCIdx = 0;
  for (auto &SCC : make_range(scc_begin(this), scc_end(this))) {
    LoopEntriesBySCC.push_back({});
    auto &SCCLoopEntries = LoopEntriesBySCC[CurrSCCIdx];

    for (auto *Node : SCC) {
      // Make sure nodes are only ever assigned one SCC
      assert(Node->SCCId == std::numeric_limits<unsigned>::max());

      Node->SCCId = CurrSCCIdx;
    }

    bool SelfLoop = false;
    if (SCC.size() == 1) {
      auto &Node = SCC[0];

      for (auto *Succ : Node->Succs) {
        if (Succ == Node) {
          SelfLoop = true;
          break;
        }
      }
    }

    // Blocks outside any (multi-block) loop will be isolated in their own
    // single-element SCC. Thus blocks that are in a loop are those in
    // multi-element SCCs or are self-looping.
    if (SCC.size() > 1 || SelfLoop) {
      // Find the loop entries - loop body blocks with predecessors outside
      // their SCC
      for (auto *Node : SCC) {
        if (Node->MBB == Entry)
          continue;

        for (auto *Pred : Node->MBB->predecessors()) {
          // This test is accurate despite not having assigned all nodes an SCC
          // yet. We only care if a node has been assigned into this SCC or not.
          if (getSCCId(Pred) != CurrSCCIdx) {
            LoopEntries.insert(Node->MBB);
            SCCLoopEntries.insert(Node->MBB);
          }
        }
      }
    }
    ++CurrSCCIdx;
  }

  // Make sure all nodes have been processed
  for (auto &Node : Nodes) {
    assert(Node.SCCId != std::numeric_limits<unsigned>::max());
  }
}

class WebAssemblyFixIrreducibleControlFlow final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Fix Irreducible Control Flow";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool processRegion(MachineBasicBlock *Entry, BlockSet &Blocks,
                     MachineFunction &MF);

  void makeSingleEntryLoop(const BlockSet &Entries, BlockSet &Blocks,
                           MachineFunction &MF, const ReachabilityGraph &Graph);

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyFixIrreducibleControlFlow() : MachineFunctionPass(ID) {}
};

bool WebAssemblyFixIrreducibleControlFlow::processRegion(
    MachineBasicBlock *Entry, BlockSet &Blocks, MachineFunction &MF) {
  bool Changed = false;
  // Remove irreducibility before processing child loops, which may take
  // multiple iterations.
  while (true) {
    ReachabilityGraph Graph(Entry, Blocks);

    bool FoundIrreducibility = false;

    for (auto *LoopEntry : getSortedEntries(Graph.getLoopEntries())) {
      // Find mutual entries - all entries which can reach this one, and
      // are reached by it (that always includes LoopEntry itself). All mutual
      // entries must be in the same SCC, so if we have more than one, then we
      // have irreducible control flow.
      //
      // (Note that we need to sort the entries here, as otherwise the order can
      // matter: being mutual is a symmetric relationship, and each set of
      // mutuals will be handled properly no matter which we see first. However,
      // there can be multiple disjoint sets of mutuals, and which we process
      // first changes the output.)
      //
      // Note that irreducibility may involve inner loops, e.g. imagine A
      // starts one loop, and it has B inside it which starts an inner loop.
      // If we add a branch from all the way on the outside to B, then in a
      // sense B is no longer an "inner" loop, semantically speaking. We will
      // fix that irreducibility by adding a block that dispatches to either
      // either A or B, so B will no longer be an inner loop in our output.
      // (A fancier approach might try to keep it as such.)
      //
      // Note that we still need to recurse into inner loops later, to handle
      // the case where the irreducibility is entirely nested - we would not
      // be able to identify that at this point, since the enclosing loop is
      // a group of blocks all of whom can reach each other. (We'll see the
      // irreducibility after removing branches to the top of that enclosing
      // loop.)
      auto &MutualLoopEntries =
          Graph.getLoopEntriesForSCC(Graph.getSCCId(LoopEntry));

      if (MutualLoopEntries.size() > 1) {
        makeSingleEntryLoop(MutualLoopEntries, Blocks, MF, Graph);
        FoundIrreducibility = true;
        Changed = true;
        break;
      }
    }

    // Only go on to actually process the inner loops when we are done
    // removing irreducible control flow and changing the graph. Modifying
    // the graph as we go is possible, and that might let us avoid looking at
    // the already-fixed loops again if we are careful, but all that is
    // complex and bug-prone. Since irreducible loops are rare, just starting
    // another iteration is best.
    if (FoundIrreducibility) {
      continue;
    }

    for (auto *LoopEntry : Graph.getLoopEntries()) {
      BlockSet InnerBlocks;

      auto EntrySCCId = Graph.getSCCId(LoopEntry);
      for (auto *Block : Blocks) {
        if (EntrySCCId == Graph.getSCCId(Block)) {
          InnerBlocks.insert(Block);
        }
      }

      // Each of these calls to processRegion may change the graph, but are
      // guaranteed not to interfere with each other. The only changes we make
      // to the graph are to add blocks on the way to a loop entry. As the
      // loops are disjoint, that means we may only alter branches that exit
      // another loop, which are ignored when recursing into that other loop
      // anyhow.
      if (processRegion(LoopEntry, InnerBlocks, MF)) {
        Changed = true;
      }
    }

    return Changed;
  }
}

// Given a set of entries to a single loop, create a single entry for that
// loop by creating a dispatch block for them, routing control flow using
// a helper variable. Also updates Blocks with any new blocks created, so
// that we properly track all the blocks in the region. But this does not update
// ReachabilityGraph; this will be updated in the caller of this function as
// needed.
void WebAssemblyFixIrreducibleControlFlow::makeSingleEntryLoop(
    const BlockSet &Entries, BlockSet &Blocks, MachineFunction &MF,
    const ReachabilityGraph &Graph) {
  assert(Entries.size() >= 2);

  // Sort the entries to ensure a deterministic build.
  BlockVector SortedEntries = getSortedEntries(Entries);

#ifndef NDEBUG
  for (auto *Block : SortedEntries)
    assert(Block->getNumber() != -1);
  if (SortedEntries.size() > 1) {
    for (auto I = SortedEntries.begin(), E = SortedEntries.end() - 1; I != E;
         ++I) {
      auto ANum = (*I)->getNumber();
      auto BNum = (*(std::next(I)))->getNumber();
      assert(ANum != BNum);
    }
  }
#endif

  // Create a dispatch block which will contain a jump table to the entries.
  MachineBasicBlock *Dispatch = MF.CreateMachineBasicBlock();
  MF.insert(MF.end(), Dispatch);
  Blocks.insert(Dispatch);

  // Add the jump table.
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  MachineInstrBuilder MIB =
      BuildMI(Dispatch, DebugLoc(), TII.get(WebAssembly::BR_TABLE_I32));

  // Add the register which will be used to tell the jump table which block to
  // jump to.
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register Reg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
  MIB.addReg(Reg);

  // Compute the indices in the superheader, one for each bad block, and
  // add them as successors.
  DenseMap<MachineBasicBlock *, unsigned> Indices;
  for (auto *Entry : SortedEntries) {
    auto Pair = Indices.try_emplace(Entry);
    assert(Pair.second);

    unsigned Index = MIB.getInstr()->getNumExplicitOperands() - 1;
    Pair.first->second = Index;

    MIB.addMBB(Entry);
    Dispatch->addSuccessor(Entry);
  }

  // Rewrite the problematic successors for every block that wants to reach
  // the bad blocks. For simplicity, we just introduce a new block for every
  // edge we need to rewrite. (Fancier things are possible.)

  BlockVector AllPreds;
  for (auto *Entry : SortedEntries) {
    for (auto *Pred : Entry->predecessors()) {
      if (Pred != Dispatch) {
        AllPreds.push_back(Pred);
      }
    }
  }

  // This set stores predecessors within this loop.
  DenseSet<MachineBasicBlock *> InLoop;
  for (auto *Pred : AllPreds) {
    auto PredSCCId = Graph.getSCCId(Pred);

    for (auto *Entry : Pred->successors()) {
      if (!Entries.count(Entry))
        continue;
      if (Graph.getSCCId(Entry) == PredSCCId) {
        InLoop.insert(Pred);
        break;
      }
    }
  }

  // Record if each entry has a layout predecessor. This map stores
  // <<loop entry, Predecessor is within the loop?>, layout predecessor>
  DenseMap<PointerIntPair<MachineBasicBlock *, 1, bool>, MachineBasicBlock *>
      EntryToLayoutPred;
  for (auto *Pred : AllPreds) {
    bool PredInLoop = InLoop.count(Pred);
    for (auto *Entry : Pred->successors())
      if (Entries.count(Entry) && Pred->isLayoutSuccessor(Entry))
        EntryToLayoutPred[{Entry, PredInLoop}] = Pred;
  }

  // We need to create at most two routing blocks per entry: one for
  // predecessors outside the loop and one for predecessors inside the loop.
  // This map stores
  // <<loop entry, Predecessor is within the loop?>, routing block>
  DenseMap<PointerIntPair<MachineBasicBlock *, 1, bool>, MachineBasicBlock *>
      Map;
  for (auto *Pred : AllPreds) {
    bool PredInLoop = InLoop.count(Pred);
    for (auto *Entry : Pred->successors()) {
      if (!Entries.count(Entry) || Map.count({Entry, PredInLoop}))
        continue;
      // If there exists a layout predecessor of this entry and this predecessor
      // is not that, we rather create a routing block after that layout
      // predecessor to save a branch.
      if (auto *OtherPred = EntryToLayoutPred.lookup({Entry, PredInLoop}))
        if (OtherPred != Pred)
          continue;

      // This is a successor we need to rewrite.
      MachineBasicBlock *Routing = MF.CreateMachineBasicBlock();
      MF.insert(Pred->isLayoutSuccessor(Entry)
                    ? MachineFunction::iterator(Entry)
                    : MF.end(),
                Routing);
      Blocks.insert(Routing);

      // Set the jump table's register of the index of the block we wish to
      // jump to, and jump to the jump table.
      BuildMI(Routing, DebugLoc(), TII.get(WebAssembly::CONST_I32), Reg)
          .addImm(Indices[Entry]);
      BuildMI(Routing, DebugLoc(), TII.get(WebAssembly::BR)).addMBB(Dispatch);
      Routing->addSuccessor(Dispatch);
      Map[{Entry, PredInLoop}] = Routing;
    }
  }

  for (auto *Pred : AllPreds) {
    bool PredInLoop = InLoop.count(Pred);
    // Remap the terminator operands and the successor list.
    for (MachineInstr &Term : Pred->terminators())
      for (auto &Op : Term.explicit_uses())
        if (Op.isMBB() && Indices.count(Op.getMBB()))
          Op.setMBB(Map[{Op.getMBB(), PredInLoop}]);

    for (auto *Succ : Pred->successors()) {
      if (!Entries.count(Succ))
        continue;
      auto *Routing = Map[{Succ, PredInLoop}];
      Pred->replaceSuccessor(Succ, Routing);
    }
  }

  // Create a fake default label, because br_table requires one.
  MIB.addMBB(MIB.getInstr()
                 ->getOperand(MIB.getInstr()->getNumExplicitOperands() - 1)
                 .getMBB());
}

} // end anonymous namespace

char WebAssemblyFixIrreducibleControlFlow::ID = 0;
INITIALIZE_PASS(WebAssemblyFixIrreducibleControlFlow, DEBUG_TYPE,
                "Removes irreducible control flow", false, false)

FunctionPass *llvm::createWebAssemblyFixIrreducibleControlFlow() {
  return new WebAssemblyFixIrreducibleControlFlow();
}

// Test whether the given register has an ARGUMENT def.
static bool hasArgumentDef(unsigned Reg, const MachineRegisterInfo &MRI) {
  for (const auto &Def : MRI.def_instructions(Reg))
    if (WebAssembly::isArgument(Def.getOpcode()))
      return true;
  return false;
}

// Add a register definition with IMPLICIT_DEFs for every register to cover for
// register uses that don't have defs in every possible path.
// TODO: This is fairly heavy-handed; find a better approach.
static void addImplicitDefs(MachineFunction &MF) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  MachineBasicBlock &Entry = *MF.begin();
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    Register Reg = Register::index2VirtReg(I);

    // Skip unused registers.
    if (MRI.use_nodbg_empty(Reg))
      continue;

    // Skip registers that have an ARGUMENT definition.
    if (hasArgumentDef(Reg, MRI))
      continue;

    BuildMI(Entry, Entry.begin(), DebugLoc(),
            TII.get(WebAssembly::IMPLICIT_DEF), Reg);
  }

  // Move ARGUMENT_* instructions to the top of the entry block, so that their
  // liveness reflects the fact that these really are live-in values.
  for (MachineInstr &MI : llvm::make_early_inc_range(Entry)) {
    if (WebAssembly::isArgument(MI.getOpcode())) {
      MI.removeFromParent();
      Entry.insert(Entry.begin(), &MI);
    }
  }
}

bool WebAssemblyFixIrreducibleControlFlow::runOnMachineFunction(
    MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Fixing Irreducible Control Flow **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');

  // Start the recursive process on the entire function body.
  BlockSet AllBlocks;
  for (auto &MBB : MF) {
    AllBlocks.insert(&MBB);
  }

  if (LLVM_UNLIKELY(processRegion(&*MF.begin(), AllBlocks, MF))) {
    // We rewrote part of the function; recompute relevant things.
    MF.RenumberBlocks();
    // Now we've inserted dispatch blocks, some register uses can have incoming
    // paths without a def. For example, before this pass register %a was
    // defined in BB1 and used in BB2, and there was only one path from BB1 and
    // BB2. But if this pass inserts a dispatch block having multiple
    // predecessors between the two BBs, now there are paths to BB2 without
    // visiting BB1, and %a's use in BB2 is not dominated by its def. Adding
    // IMPLICIT_DEFs to all regs is one simple way to fix it.
    addImplicitDefs(MF);
    return true;
  }

  return false;
}
