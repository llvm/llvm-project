//=- WebAssemblyFixIrreducibleControlFlow.cpp - Fix irreducible control flow -//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a pass that transforms irreducible control flow
/// into reducible control flow. Irreducible control flow means multiple-entry
/// loops; they appear as CFG cycles that are not recorded in MachineLoopInfo
/// due to being unnatural.
///
/// Note that LLVM has a generic pass that lowers irreducible control flow, but
/// it linearizes control flow, turning diamonds into two triangles, which is
/// both unnecessary and undesirable for WebAssembly.
///
/// TODO: The transformation implemented here handles all irreducible control
/// flow, without exponential code-size expansion, though it does so by creating
/// inefficient code in many cases. Ideally, we should add other
/// transformations, including code-duplicating cases, which can be more
/// efficient in common cases, and they can fall back to this conservative
/// implementation as needed.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-fix-irreducible-control-flow"

namespace {
class WebAssemblyFixIrreducibleControlFlow final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Fix Irreducible Control Flow";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.addPreserved<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool VisitLoop(MachineFunction &MF, MachineLoopInfo &MLI, MachineLoop *Loop);

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyFixIrreducibleControlFlow() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyFixIrreducibleControlFlow::ID = 0;
FunctionPass *llvm::createWebAssemblyFixIrreducibleControlFlow() {
  return new WebAssemblyFixIrreducibleControlFlow();
}

namespace {

/// A utility for walking the blocks of a loop, handling a nested inner
/// loop as a monolithic conceptual block.
class MetaBlock {
  MachineBasicBlock *Block;
  SmallVector<MachineBasicBlock *, 2> Preds;
  SmallVector<MachineBasicBlock *, 2> Succs;

public:
  explicit MetaBlock(MachineBasicBlock *MBB)
      : Block(MBB), Preds(MBB->pred_begin(), MBB->pred_end()),
        Succs(MBB->succ_begin(), MBB->succ_end()) {}

  explicit MetaBlock(MachineLoop *Loop) : Block(Loop->getHeader()) {
    Loop->getExitBlocks(Succs);
    for (MachineBasicBlock *Pred : Block->predecessors())
      if (!Loop->contains(Pred))
        Preds.push_back(Pred);
  }

  MachineBasicBlock *getBlock() const { return Block; }

  const SmallVectorImpl<MachineBasicBlock *> &predecessors() const {
    return Preds;
  }
  const SmallVectorImpl<MachineBasicBlock *> &successors() const {
    return Succs;
  }

  bool operator==(const MetaBlock &MBB) { return Block == MBB.Block; }
  bool operator!=(const MetaBlock &MBB) { return Block != MBB.Block; }
};

class SuccessorList final : public MetaBlock {
  size_t Index;
  size_t Num;

public:
  explicit SuccessorList(MachineBasicBlock *MBB)
      : MetaBlock(MBB), Index(0), Num(successors().size()) {}

  explicit SuccessorList(MachineLoop *Loop)
      : MetaBlock(Loop), Index(0), Num(successors().size()) {}

  bool HasNext() const { return Index != Num; }

  MachineBasicBlock *Next() {
    assert(HasNext());
    return successors()[Index++];
  }
};

} // end anonymous namespace

bool WebAssemblyFixIrreducibleControlFlow::VisitLoop(MachineFunction &MF,
                                                     MachineLoopInfo &MLI,
                                                     MachineLoop *Loop) {
  MachineBasicBlock *Header = Loop ? Loop->getHeader() : &*MF.begin();
  SetVector<MachineBasicBlock *> RewriteSuccs;

  // DFS through Loop's body, looking for irreducible control flow. Loop is
  // natural, and we stay in its body, and we treat any nested loops
  // monolithically, so any cycles we encounter indicate irreducibility.
  SmallPtrSet<MachineBasicBlock *, 8> OnStack;
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<SuccessorList, 4> LoopWorklist;
  LoopWorklist.push_back(SuccessorList(Header));
  OnStack.insert(Header);
  Visited.insert(Header);
  while (!LoopWorklist.empty()) {
    SuccessorList &Top = LoopWorklist.back();
    if (Top.HasNext()) {
      MachineBasicBlock *Next = Top.Next();
      if (Next == Header || (Loop && !Loop->contains(Next)))
        continue;
      if (LLVM_LIKELY(OnStack.insert(Next).second)) {
        if (!Visited.insert(Next).second) {
          OnStack.erase(Next);
          continue;
        }
        MachineLoop *InnerLoop = MLI.getLoopFor(Next);
        if (InnerLoop != Loop)
          LoopWorklist.push_back(SuccessorList(InnerLoop));
        else
          LoopWorklist.push_back(SuccessorList(Next));
      } else {
        RewriteSuccs.insert(Top.getBlock());
      }
      continue;
    }
    OnStack.erase(Top.getBlock());
    LoopWorklist.pop_back();
  }

  // Most likely, we didn't find any irreducible control flow.
  if (LLVM_LIKELY(RewriteSuccs.empty()))
    return false;

  DEBUG(dbgs() << "Irreducible control flow detected!\n");

  // Ok. We have irreducible control flow! Create a dispatch block which will
  // contains a jump table to any block in the problematic set of blocks.
  MachineBasicBlock *Dispatch = MF.CreateMachineBasicBlock();
  MF.insert(MF.end(), Dispatch);
  MLI.changeLoopFor(Dispatch, Loop);

  // Add the jump table.
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  MachineInstrBuilder MIB = BuildMI(*Dispatch, Dispatch->end(), DebugLoc(),
                                    TII.get(WebAssembly::BR_TABLE_I32));

  // Add the register which will be used to tell the jump table which block to
  // jump to.
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned Reg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
  MIB.addReg(Reg);

  // Collect all the blocks which need to have their successors rewritten,
  // add the successors to the jump table, and remember their index.
  DenseMap<MachineBasicBlock *, unsigned> Indices;
  SmallVector<MachineBasicBlock *, 4> SuccWorklist(RewriteSuccs.begin(),
                                                   RewriteSuccs.end());
  while (!SuccWorklist.empty()) {
    MachineBasicBlock *MBB = SuccWorklist.pop_back_val();
    auto Pair = Indices.insert(std::make_pair(MBB, 0));
    if (!Pair.second)
      continue;

    unsigned Index = MIB.getInstr()->getNumExplicitOperands() - 1;
    DEBUG(dbgs() << printMBBReference(*MBB) << " has index " << Index << "\n");

    Pair.first->second = Index;
    for (auto Pred : MBB->predecessors())
      RewriteSuccs.insert(Pred);

    MIB.addMBB(MBB);
    Dispatch->addSuccessor(MBB);

    MetaBlock Meta(MBB);
    for (auto *Succ : Meta.successors())
      if (Succ != Header && (!Loop || Loop->contains(Succ)))
        SuccWorklist.push_back(Succ);
  }

  // Rewrite the problematic successors for every block in RewriteSuccs.
  // For simplicity, we just introduce a new block for every edge we need to
  // rewrite. Fancier things are possible.
  for (MachineBasicBlock *MBB : RewriteSuccs) {
    DenseMap<MachineBasicBlock *, MachineBasicBlock *> Map;
    for (auto *Succ : MBB->successors()) {
      if (!Indices.count(Succ))
        continue;

      MachineBasicBlock *Split = MF.CreateMachineBasicBlock();
      MF.insert(MBB->isLayoutSuccessor(Succ) ? MachineFunction::iterator(Succ)
                                             : MF.end(),
                Split);
      MLI.changeLoopFor(Split, Loop);

      // Set the jump table's register of the index of the block we wish to
      // jump to, and jump to the jump table.
      BuildMI(*Split, Split->end(), DebugLoc(), TII.get(WebAssembly::CONST_I32),
              Reg)
          .addImm(Indices[Succ]);
      BuildMI(*Split, Split->end(), DebugLoc(), TII.get(WebAssembly::BR))
          .addMBB(Dispatch);
      Split->addSuccessor(Dispatch);
      Map[Succ] = Split;
    }
    // Remap the terminator operands and the successor list.
    for (MachineInstr &Term : MBB->terminators())
      for (auto &Op : Term.explicit_uses())
        if (Op.isMBB() && Indices.count(Op.getMBB()))
          Op.setMBB(Map[Op.getMBB()]);
    for (auto Rewrite : Map)
      MBB->replaceSuccessor(Rewrite.first, Rewrite.second);
  }

  // Create a fake default label, because br_table requires one.
  MIB.addMBB(MIB.getInstr()
                 ->getOperand(MIB.getInstr()->getNumExplicitOperands() - 1)
                 .getMBB());

  return true;
}

bool WebAssemblyFixIrreducibleControlFlow::runOnMachineFunction(
    MachineFunction &MF) {
  DEBUG(dbgs() << "********** Fixing Irreducible Control Flow **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  bool Changed = false;
  auto &MLI = getAnalysis<MachineLoopInfo>();

  // Visit the function body, which is identified as a null loop.
  Changed |= VisitLoop(MF, MLI, nullptr);

  // Visit all the loops.
  SmallVector<MachineLoop *, 8> Worklist(MLI.begin(), MLI.end());
  while (!Worklist.empty()) {
    MachineLoop *CurLoop = Worklist.pop_back_val();
    Worklist.append(CurLoop->begin(), CurLoop->end());
    Changed |= VisitLoop(MF, MLI, CurLoop);
  }

  // If we made any changes, completely recompute everything.
  if (LLVM_UNLIKELY(Changed)) {
    DEBUG(dbgs() << "Recomputing dominators and loops.\n");
    MF.getRegInfo().invalidateLiveness();
    MF.RenumberBlocks();
    getAnalysis<MachineDominatorTree>().runOnMachineFunction(MF);
    MLI.runOnMachineFunction(MF);
  }

  return Changed;
}
