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
/// loop's body without branches back to the loop entry. In each region we fix
/// up multi-entry loops by adding a new block that can dispatch to each of the
/// loop entries, based on the value of a label "helper" variable, and we
/// replace direct branches to the entries with assignments to the label
/// variable and a branch to the dispatch block. Then the dispatch block is the
/// single entry in the loop containing the previous multiple entries. After
/// ensuring all the loops in a region are reducible, we recurse into them. The
/// total time complexity of this pass is:
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
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-fix-irreducible-control-flow"

namespace {

using BlockVector = SmallVector<MachineBasicBlock *, 4>;
using BlockSet = SmallPtrSet<MachineBasicBlock *, 4>;

class WebAssemblyFixIrreducibleControlFlow final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Fix Irreducible Control Flow";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyFixIrreducibleControlFlow() : MachineFunctionPass(ID) {}
};

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

static bool fixIrreducible(MachineCycle &C, MachineCycleInfo &MCI,
                           MachineFunction &MF) {
  if (C.isReducible())
    return false;

  auto &Entries = C.getEntries();

  assert(Entries.size() >= 2);

  // Sort the entries to ensure a deterministic build.
  BlockVector SortedEntries(Entries.begin(), Entries.end());
  llvm::sort(SortedEntries,
             [](const MachineBasicBlock *A, const MachineBasicBlock *B) {
               auto ANum = A->getNumber();
               auto BNum = B->getNumber();
               return ANum < BNum;
             });

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
  MCI.addBlockToCycle(Dispatch, &C);

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
    if (C.contains(Pred)) {
      InLoop.insert(Pred);
    }
  }

  // Record if each entry has a layout predecessor. This map stores
  // <<loop entry, Predecessor is within the loop?>, layout predecessor>
  DenseMap<PointerIntPair<MachineBasicBlock *, 1, bool>, MachineBasicBlock *>
      EntryToLayoutPred;
  for (auto *Pred : AllPreds) {
    bool PredInLoop = InLoop.count(Pred);
    for (auto *Entry : Pred->successors())
      if (C.isEntry(Entry) && Pred->isLayoutSuccessor(Entry))
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
      if (!C.isEntry(Entry) || Map.count({Entry, PredInLoop}))
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

      if (PredInLoop) {
        MCI.addBlockToCycle(Routing, &C);
      } else {
        if (auto *Parent = C.getParentCycle())
          MCI.addBlockToCycle(Routing, Parent);
      }

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
      if (!C.isEntry(Succ))
        continue;
      auto *Routing = Map[{Succ, PredInLoop}];
      Pred->replaceSuccessor(Succ, Routing);
    }
  }

  // Create a fake default label, because br_table requires one.
  MIB.addMBB(MIB.getInstr()
                 ->getOperand(MIB.getInstr()->getNumExplicitOperands() - 1)
                 .getMBB());

  C.setSingleEntry(Dispatch);

  return true;
}

bool WebAssemblyFixIrreducibleControlFlow::runOnMachineFunction(
    MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Fixing Irreducible Control Flow **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');

  auto &MCI = getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();

  bool Changed = false;
  for (MachineCycle *TopCycle : MCI.toplevel_cycles()) {
    for (MachineCycle *C : depth_first(TopCycle)) {
      Changed |= fixIrreducible(*C, MCI, MF);
    }
  }

  if (LLVM_UNLIKELY(Changed)) {
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

void WebAssemblyFixIrreducibleControlFlow::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<MachineCycleInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}
