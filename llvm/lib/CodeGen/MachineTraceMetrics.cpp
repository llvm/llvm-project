//===- lib/CodeGen/MachineTraceMetrics.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-trace-metrics"
#include "MachineTraceMetrics.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SparseSet.h"

using namespace llvm;

char MachineTraceMetrics::ID = 0;
char &llvm::MachineTraceMetricsID = MachineTraceMetrics::ID;

INITIALIZE_PASS_BEGIN(MachineTraceMetrics,
                  "machine-trace-metrics", "Machine Trace Metrics", false, true)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(MachineTraceMetrics,
                  "machine-trace-metrics", "Machine Trace Metrics", false, true)

MachineTraceMetrics::MachineTraceMetrics()
  : MachineFunctionPass(ID), MF(0), TII(0), TRI(0), MRI(0), Loops(0) {
  std::fill(Ensembles, array_endof(Ensembles), (Ensemble*)0);
}

void MachineTraceMetrics::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.addRequired<MachineLoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineTraceMetrics::runOnMachineFunction(MachineFunction &Func) {
  MF = &Func;
  TII = MF->getTarget().getInstrInfo();
  TRI = MF->getTarget().getRegisterInfo();
  MRI = &MF->getRegInfo();
  Loops = &getAnalysis<MachineLoopInfo>();
  const TargetSubtargetInfo &ST =
    MF->getTarget().getSubtarget<TargetSubtargetInfo>();
  SchedModel.init(*ST.getSchedModel(), &ST, TII);
  BlockInfo.resize(MF->getNumBlockIDs());
  return false;
}

void MachineTraceMetrics::releaseMemory() {
  MF = 0;
  BlockInfo.clear();
  for (unsigned i = 0; i != TS_NumStrategies; ++i) {
    delete Ensembles[i];
    Ensembles[i] = 0;
  }
}

//===----------------------------------------------------------------------===//
//                          Fixed block information
//===----------------------------------------------------------------------===//
//
// The number of instructions in a basic block and the CPU resources used by
// those instructions don't depend on any given trace strategy.

/// Compute the resource usage in basic block MBB.
const MachineTraceMetrics::FixedBlockInfo*
MachineTraceMetrics::getResources(const MachineBasicBlock *MBB) {
  assert(MBB && "No basic block");
  FixedBlockInfo *FBI = &BlockInfo[MBB->getNumber()];
  if (FBI->hasResources())
    return FBI;

  // Compute resource usage in the block.
  // FIXME: Compute per-functional unit counts.
  FBI->HasCalls = false;
  unsigned InstrCount = 0;
  for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
       I != E; ++I) {
    const MachineInstr *MI = I;
    if (MI->isTransient())
      continue;
    ++InstrCount;
    if (MI->isCall())
      FBI->HasCalls = true;
  }
  FBI->InstrCount = InstrCount;
  return FBI;
}

//===----------------------------------------------------------------------===//
//                         Ensemble utility functions
//===----------------------------------------------------------------------===//

MachineTraceMetrics::Ensemble::Ensemble(MachineTraceMetrics *ct)
  : MTM(*ct) {
  BlockInfo.resize(MTM.BlockInfo.size());
}

// Virtual destructor serves as an anchor.
MachineTraceMetrics::Ensemble::~Ensemble() {}

const MachineLoop*
MachineTraceMetrics::Ensemble::getLoopFor(const MachineBasicBlock *MBB) const {
  return MTM.Loops->getLoopFor(MBB);
}

// Update resource-related information in the TraceBlockInfo for MBB.
// Only update resources related to the trace above MBB.
void MachineTraceMetrics::Ensemble::
computeDepthResources(const MachineBasicBlock *MBB) {
  TraceBlockInfo *TBI = &BlockInfo[MBB->getNumber()];

  // Compute resources from trace above. The top block is simple.
  if (!TBI->Pred) {
    TBI->InstrDepth = 0;
    TBI->Head = MBB->getNumber();
    return;
  }

  // Compute from the block above. A post-order traversal ensures the
  // predecessor is always computed first.
  TraceBlockInfo *PredTBI = &BlockInfo[TBI->Pred->getNumber()];
  assert(PredTBI->hasValidDepth() && "Trace above has not been computed yet");
  const FixedBlockInfo *PredFBI = MTM.getResources(TBI->Pred);
  TBI->InstrDepth = PredTBI->InstrDepth + PredFBI->InstrCount;
  TBI->Head = PredTBI->Head;
}

// Update resource-related information in the TraceBlockInfo for MBB.
// Only update resources related to the trace below MBB.
void MachineTraceMetrics::Ensemble::
computeHeightResources(const MachineBasicBlock *MBB) {
  TraceBlockInfo *TBI = &BlockInfo[MBB->getNumber()];

  // Compute resources for the current block.
  TBI->InstrHeight = MTM.getResources(MBB)->InstrCount;

  // The trace tail is done.
  if (!TBI->Succ) {
    TBI->Tail = MBB->getNumber();
    return;
  }

  // Compute from the block below. A post-order traversal ensures the
  // predecessor is always computed first.
  TraceBlockInfo *SuccTBI = &BlockInfo[TBI->Succ->getNumber()];
  assert(SuccTBI->hasValidHeight() && "Trace below has not been computed yet");
  TBI->InstrHeight += SuccTBI->InstrHeight;
  TBI->Tail = SuccTBI->Tail;
}

// Check if depth resources for MBB are valid and return the TBI.
// Return NULL if the resources have been invalidated.
const MachineTraceMetrics::TraceBlockInfo*
MachineTraceMetrics::Ensemble::
getDepthResources(const MachineBasicBlock *MBB) const {
  const TraceBlockInfo *TBI = &BlockInfo[MBB->getNumber()];
  return TBI->hasValidDepth() ? TBI : 0;
}

// Check if height resources for MBB are valid and return the TBI.
// Return NULL if the resources have been invalidated.
const MachineTraceMetrics::TraceBlockInfo*
MachineTraceMetrics::Ensemble::
getHeightResources(const MachineBasicBlock *MBB) const {
  const TraceBlockInfo *TBI = &BlockInfo[MBB->getNumber()];
  return TBI->hasValidHeight() ? TBI : 0;
}

//===----------------------------------------------------------------------===//
//                         Trace Selection Strategies
//===----------------------------------------------------------------------===//
//
// A trace selection strategy is implemented as a sub-class of Ensemble. The
// trace through a block B is computed by two DFS traversals of the CFG
// starting from B. One upwards, and one downwards. During the upwards DFS,
// pickTracePred() is called on the post-ordered blocks. During the downwards
// DFS, pickTraceSucc() is called in a post-order.
//

// We never allow traces that leave loops, but we do allow traces to enter
// nested loops. We also never allow traces to contain back-edges.
//
// This means that a loop header can never appear above the center block of a
// trace, except as the trace head. Below the center block, loop exiting edges
// are banned.
//
// Return true if an edge from the From loop to the To loop is leaving a loop.
// Either of To and From can be null.
static bool isExitingLoop(const MachineLoop *From, const MachineLoop *To) {
  return From && !From->contains(To);
}

// MinInstrCountEnsemble - Pick the trace that executes the least number of
// instructions.
namespace {
class MinInstrCountEnsemble : public MachineTraceMetrics::Ensemble {
  const char *getName() const { return "MinInstr"; }
  const MachineBasicBlock *pickTracePred(const MachineBasicBlock*);
  const MachineBasicBlock *pickTraceSucc(const MachineBasicBlock*);

public:
  MinInstrCountEnsemble(MachineTraceMetrics *mtm)
    : MachineTraceMetrics::Ensemble(mtm) {}
};
}

// Select the preferred predecessor for MBB.
const MachineBasicBlock*
MinInstrCountEnsemble::pickTracePred(const MachineBasicBlock *MBB) {
  if (MBB->pred_empty())
    return 0;
  const MachineLoop *CurLoop = getLoopFor(MBB);
  // Don't leave loops, and never follow back-edges.
  if (CurLoop && MBB == CurLoop->getHeader())
    return 0;
  unsigned CurCount = MTM.getResources(MBB)->InstrCount;
  const MachineBasicBlock *Best = 0;
  unsigned BestDepth = 0;
  for (MachineBasicBlock::const_pred_iterator
       I = MBB->pred_begin(), E = MBB->pred_end(); I != E; ++I) {
    const MachineBasicBlock *Pred = *I;
    const MachineTraceMetrics::TraceBlockInfo *PredTBI =
      getDepthResources(Pred);
    // Ignore cycles that aren't natural loops.
    if (!PredTBI)
      continue;
    // Pick the predecessor that would give this block the smallest InstrDepth.
    unsigned Depth = PredTBI->InstrDepth + CurCount;
    if (!Best || Depth < BestDepth)
      Best = Pred, BestDepth = Depth;
  }
  return Best;
}

// Select the preferred successor for MBB.
const MachineBasicBlock*
MinInstrCountEnsemble::pickTraceSucc(const MachineBasicBlock *MBB) {
  if (MBB->pred_empty())
    return 0;
  const MachineLoop *CurLoop = getLoopFor(MBB);
  const MachineBasicBlock *Best = 0;
  unsigned BestHeight = 0;
  for (MachineBasicBlock::const_succ_iterator
       I = MBB->succ_begin(), E = MBB->succ_end(); I != E; ++I) {
    const MachineBasicBlock *Succ = *I;
    // Don't consider back-edges.
    if (CurLoop && Succ == CurLoop->getHeader())
      continue;
    // Don't consider successors exiting CurLoop.
    if (isExitingLoop(CurLoop, getLoopFor(Succ)))
      continue;
    const MachineTraceMetrics::TraceBlockInfo *SuccTBI =
      getHeightResources(Succ);
    // Ignore cycles that aren't natural loops.
    if (!SuccTBI)
      continue;
    // Pick the successor that would give this block the smallest InstrHeight.
    unsigned Height = SuccTBI->InstrHeight;
    if (!Best || Height < BestHeight)
      Best = Succ, BestHeight = Height;
  }
  return Best;
}

// Get an Ensemble sub-class for the requested trace strategy.
MachineTraceMetrics::Ensemble *
MachineTraceMetrics::getEnsemble(MachineTraceMetrics::Strategy strategy) {
  assert(strategy < TS_NumStrategies && "Invalid trace strategy enum");
  Ensemble *&E = Ensembles[strategy];
  if (E)
    return E;

  // Allocate new Ensemble on demand.
  switch (strategy) {
  case TS_MinInstrCount: return (E = new MinInstrCountEnsemble(this));
  default: llvm_unreachable("Invalid trace strategy enum");
  }
}

void MachineTraceMetrics::invalidate(const MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Invalidate traces through BB#" << MBB->getNumber() << '\n');
  BlockInfo[MBB->getNumber()].invalidate();
  for (unsigned i = 0; i != TS_NumStrategies; ++i)
    if (Ensembles[i])
      Ensembles[i]->invalidate(MBB);
}

void MachineTraceMetrics::verifyAnalysis() const {
  if (!MF)
    return;
#ifndef NDEBUG
  assert(BlockInfo.size() == MF->getNumBlockIDs() && "Outdated BlockInfo size");
  for (unsigned i = 0; i != TS_NumStrategies; ++i)
    if (Ensembles[i])
      Ensembles[i]->verify();
#endif
}

//===----------------------------------------------------------------------===//
//                               Trace building
//===----------------------------------------------------------------------===//
//
// Traces are built by two CFG traversals. To avoid recomputing too much, use a
// set abstraction that confines the search to the current loop, and doesn't
// revisit blocks.

namespace {
struct LoopBounds {
  MutableArrayRef<MachineTraceMetrics::TraceBlockInfo> Blocks;
  SmallPtrSet<const MachineBasicBlock*, 8> Visited;
  const MachineLoopInfo *Loops;
  bool Downward;
  LoopBounds(MutableArrayRef<MachineTraceMetrics::TraceBlockInfo> blocks,
             const MachineLoopInfo *loops)
    : Blocks(blocks), Loops(loops), Downward(false) {}
};
}

// Specialize po_iterator_storage in order to prune the post-order traversal so
// it is limited to the current loop and doesn't traverse the loop back edges.
namespace llvm {
template<>
class po_iterator_storage<LoopBounds, true> {
  LoopBounds &LB;
public:
  po_iterator_storage(LoopBounds &lb) : LB(lb) {}
  void finishPostorder(const MachineBasicBlock*) {}

  bool insertEdge(const MachineBasicBlock *From, const MachineBasicBlock *To) {
    // Skip already visited To blocks.
    MachineTraceMetrics::TraceBlockInfo &TBI = LB.Blocks[To->getNumber()];
    if (LB.Downward ? TBI.hasValidHeight() : TBI.hasValidDepth())
      return false;
    // From is null once when To is the trace center block.
    if (From) {
      if (const MachineLoop *FromLoop = LB.Loops->getLoopFor(From)) {
        // Don't follow backedges, don't leave FromLoop when going upwards.
        if ((LB.Downward ? To : From) == FromLoop->getHeader())
          return false;
        // Don't leave FromLoop.
        if (isExitingLoop(FromLoop, LB.Loops->getLoopFor(To)))
          return false;
      }
    }
    // To is a new block. Mark the block as visited in case the CFG has cycles
    // that MachineLoopInfo didn't recognize as a natural loop.
    return LB.Visited.insert(To);
  }
};
}

/// Compute the trace through MBB.
void MachineTraceMetrics::Ensemble::computeTrace(const MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Computing " << getName() << " trace through BB#"
               << MBB->getNumber() << '\n');
  // Set up loop bounds for the backwards post-order traversal.
  LoopBounds Bounds(BlockInfo, MTM.Loops);

  // Run an upwards post-order search for the trace start.
  Bounds.Downward = false;
  Bounds.Visited.clear();
  typedef ipo_ext_iterator<const MachineBasicBlock*, LoopBounds> UpwardPO;
  for (UpwardPO I = ipo_ext_begin(MBB, Bounds), E = ipo_ext_end(MBB, Bounds);
       I != E; ++I) {
    DEBUG(dbgs() << "  pred for BB#" << I->getNumber() << ": ");
    TraceBlockInfo &TBI = BlockInfo[I->getNumber()];
    // All the predecessors have been visited, pick the preferred one.
    TBI.Pred = pickTracePred(*I);
    DEBUG({
      if (TBI.Pred)
        dbgs() << "BB#" << TBI.Pred->getNumber() << '\n';
      else
        dbgs() << "null\n";
    });
    // The trace leading to I is now known, compute the depth resources.
    computeDepthResources(*I);
  }

  // Run a downwards post-order search for the trace end.
  Bounds.Downward = true;
  Bounds.Visited.clear();
  typedef po_ext_iterator<const MachineBasicBlock*, LoopBounds> DownwardPO;
  for (DownwardPO I = po_ext_begin(MBB, Bounds), E = po_ext_end(MBB, Bounds);
       I != E; ++I) {
    DEBUG(dbgs() << "  succ for BB#" << I->getNumber() << ": ");
    TraceBlockInfo &TBI = BlockInfo[I->getNumber()];
    // All the successors have been visited, pick the preferred one.
    TBI.Succ = pickTraceSucc(*I);
    DEBUG({
      if (TBI.Succ)
        dbgs() << "BB#" << TBI.Succ->getNumber() << '\n';
      else
        dbgs() << "null\n";
    });
    // The trace leaving I is now known, compute the height resources.
    computeHeightResources(*I);
  }
}

/// Invalidate traces through BadMBB.
void
MachineTraceMetrics::Ensemble::invalidate(const MachineBasicBlock *BadMBB) {
  SmallVector<const MachineBasicBlock*, 16> WorkList;
  TraceBlockInfo &BadTBI = BlockInfo[BadMBB->getNumber()];

  // Invalidate height resources of blocks above MBB.
  if (BadTBI.hasValidHeight()) {
    BadTBI.invalidateHeight();
    WorkList.push_back(BadMBB);
    do {
      const MachineBasicBlock *MBB = WorkList.pop_back_val();
      DEBUG(dbgs() << "Invalidate BB#" << MBB->getNumber() << ' ' << getName()
            << " height.\n");
      // Find any MBB predecessors that have MBB as their preferred successor.
      // They are the only ones that need to be invalidated.
      for (MachineBasicBlock::const_pred_iterator
           I = MBB->pred_begin(), E = MBB->pred_end(); I != E; ++I) {
        TraceBlockInfo &TBI = BlockInfo[(*I)->getNumber()];
        if (!TBI.hasValidHeight())
          continue;
        if (TBI.Succ == MBB) {
          TBI.invalidateHeight();
          WorkList.push_back(*I);
          continue;
        }
        // Verify that TBI.Succ is actually a *I successor.
        assert((!TBI.Succ || (*I)->isSuccessor(TBI.Succ)) && "CFG changed");
      }
    } while (!WorkList.empty());
  }

  // Invalidate depth resources of blocks below MBB.
  if (BadTBI.hasValidDepth()) {
    BadTBI.invalidateDepth();
    WorkList.push_back(BadMBB);
    do {
      const MachineBasicBlock *MBB = WorkList.pop_back_val();
      DEBUG(dbgs() << "Invalidate BB#" << MBB->getNumber() << ' ' << getName()
            << " depth.\n");
      // Find any MBB successors that have MBB as their preferred predecessor.
      // They are the only ones that need to be invalidated.
      for (MachineBasicBlock::const_succ_iterator
           I = MBB->succ_begin(), E = MBB->succ_end(); I != E; ++I) {
        TraceBlockInfo &TBI = BlockInfo[(*I)->getNumber()];
        if (!TBI.hasValidDepth())
          continue;
        if (TBI.Pred == MBB) {
          TBI.invalidateDepth();
          WorkList.push_back(*I);
          continue;
        }
        // Verify that TBI.Pred is actually a *I predecessor.
        assert((!TBI.Pred || (*I)->isPredecessor(TBI.Pred)) && "CFG changed");
      }
    } while (!WorkList.empty());
  }

  // Clear any per-instruction data. We only have to do this for BadMBB itself
  // because the instructions in that block may change. Other blocks may be
  // invalidated, but their instructions will stay the same, so there is no
  // need to erase the Cycle entries. They will be overwritten when we
  // recompute.
  for (MachineBasicBlock::const_iterator I = BadMBB->begin(), E = BadMBB->end();
       I != E; ++I)
    Cycles.erase(I);
}

void MachineTraceMetrics::Ensemble::verify() const {
#ifndef NDEBUG
  assert(BlockInfo.size() == MTM.MF->getNumBlockIDs() &&
         "Outdated BlockInfo size");
  for (unsigned Num = 0, e = BlockInfo.size(); Num != e; ++Num) {
    const TraceBlockInfo &TBI = BlockInfo[Num];
    if (TBI.hasValidDepth() && TBI.Pred) {
      const MachineBasicBlock *MBB = MTM.MF->getBlockNumbered(Num);
      assert(MBB->isPredecessor(TBI.Pred) && "CFG doesn't match trace");
      assert(BlockInfo[TBI.Pred->getNumber()].hasValidDepth() &&
             "Trace is broken, depth should have been invalidated.");
      const MachineLoop *Loop = getLoopFor(MBB);
      assert(!(Loop && MBB == Loop->getHeader()) && "Trace contains backedge");
    }
    if (TBI.hasValidHeight() && TBI.Succ) {
      const MachineBasicBlock *MBB = MTM.MF->getBlockNumbered(Num);
      assert(MBB->isSuccessor(TBI.Succ) && "CFG doesn't match trace");
      assert(BlockInfo[TBI.Succ->getNumber()].hasValidHeight() &&
             "Trace is broken, height should have been invalidated.");
      const MachineLoop *Loop = getLoopFor(MBB);
      const MachineLoop *SuccLoop = getLoopFor(TBI.Succ);
      assert(!(Loop && Loop == SuccLoop && TBI.Succ == Loop->getHeader()) &&
             "Trace contains backedge");
    }
  }
#endif
}

//===----------------------------------------------------------------------===//
//                             Data Dependencies
//===----------------------------------------------------------------------===//
//
// Compute the depth and height of each instruction based on data dependencies
// and instruction latencies. These cycle numbers assume that the CPU can issue
// an infinite number of instructions per cycle as long as their dependencies
// are ready.

// A data dependency is represented as a defining MI and operand numbers on the
// defining and using MI.
namespace {
struct DataDep {
  const MachineInstr *DefMI;
  unsigned DefOp;
  unsigned UseOp;

  DataDep(const MachineInstr *DefMI, unsigned DefOp, unsigned UseOp)
    : DefMI(DefMI), DefOp(DefOp), UseOp(UseOp) {}

  /// Create a DataDep from an SSA form virtual register.
  DataDep(const MachineRegisterInfo *MRI, unsigned VirtReg, unsigned UseOp)
    : UseOp(UseOp) {
    assert(TargetRegisterInfo::isVirtualRegister(VirtReg));
    MachineRegisterInfo::def_iterator DefI = MRI->def_begin(VirtReg);
    assert(!DefI.atEnd() && "Register has no defs");
    DefMI = &*DefI;
    DefOp = DefI.getOperandNo();
    assert((++DefI).atEnd() && "Register has multiple defs");
  }
};
}

// Get the input data dependencies that must be ready before UseMI can issue.
// Return true if UseMI has any physreg operands.
static bool getDataDeps(const MachineInstr *UseMI,
                        SmallVectorImpl<DataDep> &Deps,
                        const MachineRegisterInfo *MRI) {
  bool HasPhysRegs = false;
  for (ConstMIOperands MO(UseMI); MO.isValid(); ++MO) {
    if (!MO->isReg())
      continue;
    unsigned Reg = MO->getReg();
    if (!Reg)
      continue;
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      HasPhysRegs = true;
      continue;
    }
    // Collect virtual register reads.
    if (MO->readsReg())
      Deps.push_back(DataDep(MRI, Reg, MO.getOperandNo()));
  }
  return HasPhysRegs;
}

// Get the input data dependencies of a PHI instruction, using Pred as the
// preferred predecessor.
// This will add at most one dependency to Deps.
static void getPHIDeps(const MachineInstr *UseMI,
                       SmallVectorImpl<DataDep> &Deps,
                       const MachineBasicBlock *Pred,
                       const MachineRegisterInfo *MRI) {
  // No predecessor at the beginning of a trace. Ignore dependencies.
  if (!Pred)
    return;
  assert(UseMI->isPHI() && UseMI->getNumOperands() % 2 && "Bad PHI");
  for (unsigned i = 1; i != UseMI->getNumOperands(); i += 2) {
    if (UseMI->getOperand(i + 1).getMBB() == Pred) {
      unsigned Reg = UseMI->getOperand(i).getReg();
      Deps.push_back(DataDep(MRI, Reg, i));
      return;
    }
  }
}

// Keep track of physreg data dependencies by recording each live register unit.
// Associate each regunit with an instruction operand. Depending on the
// direction instructions are scanned, it could be the operand that defined the
// regunit, or the highest operand to read the regunit.
namespace {
struct LiveRegUnit {
  unsigned RegUnit;
  unsigned Cycle;
  const MachineInstr *MI;
  unsigned Op;

  unsigned getSparseSetIndex() const { return RegUnit; }

  LiveRegUnit(unsigned RU) : RegUnit(RU), Cycle(0), MI(0), Op(0) {}
};
}

// Identify physreg dependencies for UseMI, and update the live regunit
// tracking set when scanning instructions downwards.
static void updatePhysDepsDownwards(const MachineInstr *UseMI,
                                    SmallVectorImpl<DataDep> &Deps,
                                    SparseSet<LiveRegUnit> &RegUnits,
                                    const TargetRegisterInfo *TRI) {
  SmallVector<unsigned, 8> Kills;
  SmallVector<unsigned, 8> LiveDefOps;

  for (ConstMIOperands MO(UseMI); MO.isValid(); ++MO) {
    if (!MO->isReg())
      continue;
    unsigned Reg = MO->getReg();
    if (!TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    // Track live defs and kills for updating RegUnits.
    if (MO->isDef()) {
      if (MO->isDead())
        Kills.push_back(Reg);
      else
        LiveDefOps.push_back(MO.getOperandNo());
    } else if (MO->isKill())
      Kills.push_back(Reg);
    // Identify dependencies.
    if (!MO->readsReg())
      continue;
    for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units) {
      SparseSet<LiveRegUnit>::iterator I = RegUnits.find(*Units);
      if (I == RegUnits.end())
        continue;
      Deps.push_back(DataDep(I->MI, I->Op, MO.getOperandNo()));
      break;
    }
  }

  // Update RegUnits to reflect live registers after UseMI.
  // First kills.
  for (unsigned i = 0, e = Kills.size(); i != e; ++i)
    for (MCRegUnitIterator Units(Kills[i], TRI); Units.isValid(); ++Units)
      RegUnits.erase(*Units);

  // Second, live defs.
  for (unsigned i = 0, e = LiveDefOps.size(); i != e; ++i) {
    unsigned DefOp = LiveDefOps[i];
    for (MCRegUnitIterator Units(UseMI->getOperand(DefOp).getReg(), TRI);
         Units.isValid(); ++Units) {
      LiveRegUnit &LRU = RegUnits[*Units];
      LRU.MI = UseMI;
      LRU.Op = DefOp;
    }
  }
}

/// The length of the critical path through a trace is the maximum of two path
/// lengths:
///
/// 1. The maximum height+depth over all instructions in the trace center block.
///
/// 2. The longest cross-block dependency chain. For small blocks, it is
///    possible that the critical path through the trace doesn't include any
///    instructions in the block.
///
/// This function computes the second number from the live-in list of the
/// center block.
unsigned MachineTraceMetrics::Ensemble::
computeCrossBlockCriticalPath(const TraceBlockInfo &TBI) {
  assert(TBI.HasValidInstrDepths && "Missing depth info");
  assert(TBI.HasValidInstrHeights && "Missing height info");
  unsigned MaxLen = 0;
  for (unsigned i = 0, e = TBI.LiveIns.size(); i != e; ++i) {
    const LiveInReg &LIR = TBI.LiveIns[i];
    if (!TargetRegisterInfo::isVirtualRegister(LIR.Reg))
      continue;
    const MachineInstr *DefMI = MTM.MRI->getVRegDef(LIR.Reg);
    // Ignore dependencies outside the current trace.
    const TraceBlockInfo &DefTBI = BlockInfo[DefMI->getParent()->getNumber()];
    if (!DefTBI.isEarlierInSameTrace(TBI))
      continue;
    unsigned Len = LIR.Height + Cycles[DefMI].Depth;
    MaxLen = std::max(MaxLen, Len);
  }
  return MaxLen;
}

/// Compute instruction depths for all instructions above or in MBB in its
/// trace. This assumes that the trace through MBB has already been computed.
void MachineTraceMetrics::Ensemble::
computeInstrDepths(const MachineBasicBlock *MBB) {
  // The top of the trace may already be computed, and HasValidInstrDepths
  // implies Head->HasValidInstrDepths, so we only need to start from the first
  // block in the trace that needs to be recomputed.
  SmallVector<const MachineBasicBlock*, 8> Stack;
  do {
    TraceBlockInfo &TBI = BlockInfo[MBB->getNumber()];
    assert(TBI.hasValidDepth() && "Incomplete trace");
    if (TBI.HasValidInstrDepths)
      break;
    Stack.push_back(MBB);
    MBB = TBI.Pred;
  } while (MBB);

  // FIXME: If MBB is non-null at this point, it is the last pre-computed block
  // in the trace. We should track any live-out physregs that were defined in
  // the trace. This is quite rare in SSA form, typically created by CSE
  // hoisting a compare.
  SparseSet<LiveRegUnit> RegUnits;
  RegUnits.setUniverse(MTM.TRI->getNumRegUnits());

  // Go through trace blocks in top-down order, stopping after the center block.
  SmallVector<DataDep, 8> Deps;
  while (!Stack.empty()) {
    MBB = Stack.pop_back_val();
    DEBUG(dbgs() << "Depths for BB#" << MBB->getNumber() << ":\n");
    TraceBlockInfo &TBI = BlockInfo[MBB->getNumber()];
    TBI.HasValidInstrDepths = true;
    TBI.CriticalPath = 0;

    // Also compute the critical path length through MBB when possible.
    if (TBI.HasValidInstrHeights)
      TBI.CriticalPath = computeCrossBlockCriticalPath(TBI);

    for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      const MachineInstr *UseMI = I;

      // Collect all data dependencies.
      Deps.clear();
      if (UseMI->isPHI())
        getPHIDeps(UseMI, Deps, TBI.Pred, MTM.MRI);
      else if (getDataDeps(UseMI, Deps, MTM.MRI))
        updatePhysDepsDownwards(UseMI, Deps, RegUnits, MTM.TRI);

      // Filter and process dependencies, computing the earliest issue cycle.
      unsigned Cycle = 0;
      for (unsigned i = 0, e = Deps.size(); i != e; ++i) {
        const DataDep &Dep = Deps[i];
        const TraceBlockInfo&DepTBI =
          BlockInfo[Dep.DefMI->getParent()->getNumber()];
        // Ignore dependencies from outside the current trace.
        if (!DepTBI.isEarlierInSameTrace(TBI))
          continue;
        assert(DepTBI.HasValidInstrDepths && "Inconsistent dependency");
        unsigned DepCycle = Cycles.lookup(Dep.DefMI).Depth;
        // Add latency if DefMI is a real instruction. Transients get latency 0.
        if (!Dep.DefMI->isTransient())
          DepCycle += MTM.SchedModel
            .computeOperandLatency(Dep.DefMI, Dep.DefOp, UseMI, Dep.UseOp,
                                   /* FindMin = */ false);
        Cycle = std::max(Cycle, DepCycle);
      }
      // Remember the instruction depth.
      InstrCycles &MICycles = Cycles[UseMI];
      MICycles.Depth = Cycle;

      if (!TBI.HasValidInstrHeights) {
        DEBUG(dbgs() << Cycle << '\t' << *UseMI);
        continue;
      }
      // Update critical path length.
      TBI.CriticalPath = std::max(TBI.CriticalPath, Cycle + MICycles.Height);
      DEBUG(dbgs() << TBI.CriticalPath << '\t' << Cycle << '\t' << *UseMI);
    }
  }
}

// Identify physreg dependencies for MI when scanning instructions upwards.
// Return the issue height of MI after considering any live regunits.
// Height is the issue height computed from virtual register dependencies alone.
static unsigned updatePhysDepsUpwards(const MachineInstr *MI, unsigned Height,
                                      SparseSet<LiveRegUnit> &RegUnits,
                                      const TargetSchedModel &SchedModel,
                                      const TargetInstrInfo *TII,
                                      const TargetRegisterInfo *TRI) {
  SmallVector<unsigned, 8> ReadOps;
  for (ConstMIOperands MO(MI); MO.isValid(); ++MO) {
    if (!MO->isReg())
      continue;
    unsigned Reg = MO->getReg();
    if (!TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (MO->readsReg())
      ReadOps.push_back(MO.getOperandNo());
    if (!MO->isDef())
      continue;
    // This is a def of Reg. Remove corresponding entries from RegUnits, and
    // update MI Height to consider the physreg dependencies.
    for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units) {
      SparseSet<LiveRegUnit>::iterator I = RegUnits.find(*Units);
      if (I == RegUnits.end())
        continue;
      unsigned DepHeight = I->Cycle;
      if (!MI->isTransient()) {
        // We may not know the UseMI of this dependency, if it came from the
        // live-in list. SchedModel can handle a NULL UseMI.
        DepHeight += SchedModel
          .computeOperandLatency(MI, MO.getOperandNo(), I->MI, I->Op,
                                 /* FindMin = */ false);
      }
      Height = std::max(Height, DepHeight);
      // This regunit is dead above MI.
      RegUnits.erase(I);
    }
  }

  // Now we know the height of MI. Update any regunits read.
  for (unsigned i = 0, e = ReadOps.size(); i != e; ++i) {
    unsigned Reg = MI->getOperand(ReadOps[i]).getReg();
    for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units) {
      LiveRegUnit &LRU = RegUnits[*Units];
      // Set the height to the highest reader of the unit.
      if (LRU.Cycle <= Height && LRU.MI != MI) {
        LRU.Cycle = Height;
        LRU.MI = MI;
        LRU.Op = ReadOps[i];
      }
    }
  }

  return Height;
}


typedef DenseMap<const MachineInstr *, unsigned> MIHeightMap;

// Push the height of DefMI upwards if required to match UseMI.
// Return true if this is the first time DefMI was seen.
static bool pushDepHeight(const DataDep &Dep,
                          const MachineInstr *UseMI, unsigned UseHeight,
                          MIHeightMap &Heights,
                          const TargetSchedModel &SchedModel,
                          const TargetInstrInfo *TII) {
  // Adjust height by Dep.DefMI latency.
  if (!Dep.DefMI->isTransient())
    UseHeight += SchedModel.computeOperandLatency(Dep.DefMI, Dep.DefOp,
                                                  UseMI, Dep.UseOp, false);

  // Update Heights[DefMI] to be the maximum height seen.
  MIHeightMap::iterator I;
  bool New;
  tie(I, New) = Heights.insert(std::make_pair(Dep.DefMI, UseHeight));
  if (New)
    return true;

  // DefMI has been pushed before. Give it the max height.
  if (I->second < UseHeight)
    I->second = UseHeight;
  return false;
}

/// Assuming that the virtual register defined by DefMI:DefOp was used by
/// Trace.back(), add it to the live-in lists of all the blocks in Trace. Stop
/// when reaching the block that contains DefMI.
void MachineTraceMetrics::Ensemble::
addLiveIns(const MachineInstr *DefMI, unsigned DefOp,
           ArrayRef<const MachineBasicBlock*> Trace) {
  assert(!Trace.empty() && "Trace should contain at least one block");
  unsigned Reg = DefMI->getOperand(DefOp).getReg();
  assert(TargetRegisterInfo::isVirtualRegister(Reg));
  const MachineBasicBlock *DefMBB = DefMI->getParent();

  // Reg is live-in to all blocks in Trace that follow DefMBB.
  for (unsigned i = Trace.size(); i; --i) {
    const MachineBasicBlock *MBB = Trace[i-1];
    if (MBB == DefMBB)
      return;
    TraceBlockInfo &TBI = BlockInfo[MBB->getNumber()];
    // Just add the register. The height will be updated later.
    TBI.LiveIns.push_back(Reg);
  }
}

/// Compute instruction heights in the trace through MBB. This updates MBB and
/// the blocks below it in the trace. It is assumed that the trace has already
/// been computed.
void MachineTraceMetrics::Ensemble::
computeInstrHeights(const MachineBasicBlock *MBB) {
  // The bottom of the trace may already be computed.
  // Find the blocks that need updating.
  SmallVector<const MachineBasicBlock*, 8> Stack;
  do {
    TraceBlockInfo &TBI = BlockInfo[MBB->getNumber()];
    assert(TBI.hasValidHeight() && "Incomplete trace");
    if (TBI.HasValidInstrHeights)
      break;
    Stack.push_back(MBB);
    TBI.LiveIns.clear();
    MBB = TBI.Succ;
  } while (MBB);

  // As we move upwards in the trace, keep track of instructions that are
  // required by deeper trace instructions. Map MI -> height required so far.
  MIHeightMap Heights;

  // For physregs, the def isn't known when we see the use.
  // Instead, keep track of the highest use of each regunit.
  SparseSet<LiveRegUnit> RegUnits;
  RegUnits.setUniverse(MTM.TRI->getNumRegUnits());

  // If the bottom of the trace was already precomputed, initialize heights
  // from its live-in list.
  // MBB is the highest precomputed block in the trace.
  if (MBB) {
    TraceBlockInfo &TBI = BlockInfo[MBB->getNumber()];
    for (unsigned i = 0, e = TBI.LiveIns.size(); i != e; ++i) {
      LiveInReg LI = TBI.LiveIns[i];
      if (TargetRegisterInfo::isVirtualRegister(LI.Reg)) {
        // For virtual registers, the def latency is included.
        unsigned &Height = Heights[MTM.MRI->getVRegDef(LI.Reg)];
        if (Height < LI.Height)
          Height = LI.Height;
      } else {
        // For register units, the def latency is not included because we don't
        // know the def yet.
        RegUnits[LI.Reg].Cycle = LI.Height;
      }
    }
  }

  // Go through the trace blocks in bottom-up order.
  SmallVector<DataDep, 8> Deps;
  for (;!Stack.empty(); Stack.pop_back()) {
    MBB = Stack.back();
    DEBUG(dbgs() << "Heights for BB#" << MBB->getNumber() << ":\n");
    TraceBlockInfo &TBI = BlockInfo[MBB->getNumber()];
    TBI.HasValidInstrHeights = true;
    TBI.CriticalPath = 0;

    // Get dependencies from PHIs in the trace successor.
    const MachineBasicBlock *Succ = TBI.Succ;
    // If MBB is the last block in the trace, and it has a back-edge to the
    // loop header, get loop-carried dependencies from PHIs in the header. For
    // that purpose, pretend that all the loop header PHIs have height 0.
    if (!Succ)
      if (const MachineLoop *Loop = getLoopFor(MBB))
        if (MBB->isSuccessor(Loop->getHeader()))
          Succ = Loop->getHeader();

    if (Succ) {
      for (MachineBasicBlock::const_iterator I = Succ->begin(), E = Succ->end();
           I != E && I->isPHI(); ++I) {
        const MachineInstr *PHI = I;
        Deps.clear();
        getPHIDeps(PHI, Deps, MBB, MTM.MRI);
        if (!Deps.empty()) {
          // Loop header PHI heights are all 0.
          unsigned Height = TBI.Succ ? Cycles.lookup(PHI).Height : 0;
          DEBUG(dbgs() << "pred\t" << Height << '\t' << *PHI);
          if (pushDepHeight(Deps.front(), PHI, Height,
                            Heights, MTM.SchedModel, MTM.TII))
            addLiveIns(Deps.front().DefMI, Deps.front().DefOp, Stack);
        }
      }
    }

    // Go through the block backwards.
    for (MachineBasicBlock::const_iterator BI = MBB->end(), BB = MBB->begin();
         BI != BB;) {
      const MachineInstr *MI = --BI;

      // Find the MI height as determined by virtual register uses in the
      // trace below.
      unsigned Cycle = 0;
      MIHeightMap::iterator HeightI = Heights.find(MI);
      if (HeightI != Heights.end()) {
        Cycle = HeightI->second;
        // We won't be seeing any more MI uses.
        Heights.erase(HeightI);
      }

      // Don't process PHI deps. They depend on the specific predecessor, and
      // we'll get them when visiting the predecessor.
      Deps.clear();
      bool HasPhysRegs = !MI->isPHI() && getDataDeps(MI, Deps, MTM.MRI);

      // There may also be regunit dependencies to include in the height.
      if (HasPhysRegs)
        Cycle = updatePhysDepsUpwards(MI, Cycle, RegUnits,
                                      MTM.SchedModel, MTM.TII, MTM.TRI);

      // Update the required height of any virtual registers read by MI.
      for (unsigned i = 0, e = Deps.size(); i != e; ++i)
        if (pushDepHeight(Deps[i], MI, Cycle, Heights, MTM.SchedModel, MTM.TII))
          addLiveIns(Deps[i].DefMI, Deps[i].DefOp, Stack);

      InstrCycles &MICycles = Cycles[MI];
      MICycles.Height = Cycle;
      if (!TBI.HasValidInstrDepths) {
        DEBUG(dbgs() << Cycle << '\t' << *MI);
        continue;
      }
      // Update critical path length.
      TBI.CriticalPath = std::max(TBI.CriticalPath, Cycle + MICycles.Depth);
      DEBUG(dbgs() << TBI.CriticalPath << '\t' << Cycle << '\t' << *MI);
    }

    // Update virtual live-in heights. They were added by addLiveIns() with a 0
    // height because the final height isn't known until now.
    DEBUG(dbgs() << "BB#" << MBB->getNumber() <<  " Live-ins:");
    for (unsigned i = 0, e = TBI.LiveIns.size(); i != e; ++i) {
      LiveInReg &LIR = TBI.LiveIns[i];
      const MachineInstr *DefMI = MTM.MRI->getVRegDef(LIR.Reg);
      LIR.Height = Heights.lookup(DefMI);
      DEBUG(dbgs() << ' ' << PrintReg(LIR.Reg) << '@' << LIR.Height);
    }

    // Transfer the live regunits to the live-in list.
    for (SparseSet<LiveRegUnit>::const_iterator
         RI = RegUnits.begin(), RE = RegUnits.end(); RI != RE; ++RI) {
      TBI.LiveIns.push_back(LiveInReg(RI->RegUnit, RI->Cycle));
      DEBUG(dbgs() << ' ' << PrintRegUnit(RI->RegUnit, MTM.TRI)
                   << '@' << RI->Cycle);
    }
    DEBUG(dbgs() << '\n');

    if (!TBI.HasValidInstrDepths)
      continue;
    // Add live-ins to the critical path length.
    TBI.CriticalPath = std::max(TBI.CriticalPath,
                                computeCrossBlockCriticalPath(TBI));
    DEBUG(dbgs() << "Critical path: " << TBI.CriticalPath << '\n');
  }
}

MachineTraceMetrics::Trace
MachineTraceMetrics::Ensemble::getTrace(const MachineBasicBlock *MBB) {
  // FIXME: Check cache tags, recompute as needed.
  computeTrace(MBB);
  computeInstrDepths(MBB);
  computeInstrHeights(MBB);
  return Trace(*this, BlockInfo[MBB->getNumber()]);
}

unsigned
MachineTraceMetrics::Trace::getInstrSlack(const MachineInstr *MI) const {
  assert(MI && "Not an instruction.");
  assert(getBlockNum() == unsigned(MI->getParent()->getNumber()) &&
         "MI must be in the trace center block");
  InstrCycles Cyc = getInstrCycles(MI);
  return getCriticalPath() - (Cyc.Depth + Cyc.Height);
}

unsigned
MachineTraceMetrics::Trace::getPHIDepth(const MachineInstr *PHI) const {
  const MachineBasicBlock *MBB = TE.MTM.MF->getBlockNumbered(getBlockNum());
  SmallVector<DataDep, 1> Deps;
  getPHIDeps(PHI, Deps, MBB, TE.MTM.MRI);
  assert(Deps.size() == 1 && "PHI doesn't have MBB as a predecessor");
  DataDep &Dep = Deps.front();
  unsigned DepCycle = getInstrCycles(Dep.DefMI).Depth;
  // Add latency if DefMI is a real instruction. Transients get latency 0.
  if (!Dep.DefMI->isTransient())
    DepCycle += TE.MTM.SchedModel
      .computeOperandLatency(Dep.DefMI, Dep.DefOp, PHI, Dep.UseOp, false);
  return DepCycle;
}

unsigned MachineTraceMetrics::Trace::getResourceDepth(bool Bottom) const {
  // For now, we compute the resource depth from instruction count / issue
  // width. Eventually, we should compute resource depth per functional unit
  // and return the max.
  unsigned Instrs = TBI.InstrDepth;
  if (Bottom)
    Instrs += TE.MTM.BlockInfo[getBlockNum()].InstrCount;
  if (unsigned IW = TE.MTM.SchedModel.getIssueWidth())
    Instrs /= IW;
  // Assume issue width 1 without a schedule model.
  return Instrs;
}

unsigned MachineTraceMetrics::Trace::
getResourceLength(ArrayRef<const MachineBasicBlock*> Extrablocks) const {
  unsigned Instrs = TBI.InstrDepth + TBI.InstrHeight;
  for (unsigned i = 0, e = Extrablocks.size(); i != e; ++i)
    Instrs += TE.MTM.getResources(Extrablocks[i])->InstrCount;
  if (unsigned IW = TE.MTM.SchedModel.getIssueWidth())
    Instrs /= IW;
  // Assume issue width 1 without a schedule model.
  return Instrs;
}

void MachineTraceMetrics::Ensemble::print(raw_ostream &OS) const {
  OS << getName() << " ensemble:\n";
  for (unsigned i = 0, e = BlockInfo.size(); i != e; ++i) {
    OS << "  BB#" << i << '\t';
    BlockInfo[i].print(OS);
    OS << '\n';
  }
}

void MachineTraceMetrics::TraceBlockInfo::print(raw_ostream &OS) const {
  if (hasValidDepth()) {
    OS << "depth=" << InstrDepth;
    if (Pred)
      OS << " pred=BB#" << Pred->getNumber();
    else
      OS << " pred=null";
    OS << " head=BB#" << Head;
    if (HasValidInstrDepths)
      OS << " +instrs";
  } else
    OS << "depth invalid";
  OS << ", ";
  if (hasValidHeight()) {
    OS << "height=" << InstrHeight;
    if (Succ)
      OS << " succ=BB#" << Succ->getNumber();
    else
      OS << " succ=null";
    OS << " tail=BB#" << Tail;
    if (HasValidInstrHeights)
      OS << " +instrs";
  } else
    OS << "height invalid";
  if (HasValidInstrDepths && HasValidInstrHeights)
    OS << ", crit=" << CriticalPath;
}

void MachineTraceMetrics::Trace::print(raw_ostream &OS) const {
  unsigned MBBNum = &TBI - &TE.BlockInfo[0];

  OS << TE.getName() << " trace BB#" << TBI.Head << " --> BB#" << MBBNum
     << " --> BB#" << TBI.Tail << ':';
  if (TBI.hasValidHeight() && TBI.hasValidDepth())
    OS << ' ' << getInstrCount() << " instrs.";
  if (TBI.HasValidInstrDepths && TBI.HasValidInstrHeights)
    OS << ' ' << TBI.CriticalPath << " cycles.";

  const MachineTraceMetrics::TraceBlockInfo *Block = &TBI;
  OS << "\nBB#" << MBBNum;
  while (Block->hasValidDepth() && Block->Pred) {
    unsigned Num = Block->Pred->getNumber();
    OS << " <- BB#" << Num;
    Block = &TE.BlockInfo[Num];
  }

  Block = &TBI;
  OS << "\n    ";
  while (Block->hasValidHeight() && Block->Succ) {
    unsigned Num = Block->Succ->getNumber();
    OS << " -> BB#" << Num;
    Block = &TE.BlockInfo[Num];
  }
  OS << '\n';
}
