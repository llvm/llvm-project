
//===----------------- HexagonLiveVariables.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Hexagon Live Variable Analysis
// This file implements the Hexagon specific LiveVariables analysis pass.
// This pass recomputes physical register liveness and updates live-ins for
// non-entry blocks based on use/def information.
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "hexagon_live_vars"

#include "HexagonLiveVariables.h"
#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

char HexagonLiveVariables::ID = 0;
char &llvm::HexagonLiveVariablesID = HexagonLiveVariables::ID;

INITIALIZE_PASS(HexagonLiveVariables, "hexagon-live-vars",
                "Hexagon Live Variable Analysis", false, false)

// TODO: Establish a protocol to handle liveness of predicated instructions.
// Liveness for predicated instruction is a little convoluted.
// TODO: In PhysRegDef and PhysRegUse, use a bit vector instead of 126 elems.
class HexagonLiveVariablesImpl {
  // Intermediate data structures
  friend class llvm::HexagonLiveVariables;
  typedef MachineBasicBlock::const_instr_iterator MICInstIterType;

  MachineFunction *MF;

  MachineRegisterInfo *MRI;

  const TargetRegisterInfo *TRI;

  const HexagonInstrInfo *QII;

  unsigned NumRegs;

  /// PhysRegInfo - Keep track of which instruction was the last def of a
  /// physical register (possibly after a use). This is purely local to a BB.
  SmallVector<MachineInstr *, 0> PhysRegDef;

  /// PhysRegInfo - Keep track of which instruction was the last use of a
  /// physical register (before any def). This is purely local property to a BB.
  SmallVector<MachineInstr *, 0> PhysRegUse;

  /// MBB -> (Uses, Defs)
  /// Uses - use before any def in that MBB.
  /// Defs - def before any uses  in that MBB.
  MBBUseDef_t MBBUseDefs;

  /// MI -> (Uses, Defs)
  MIUseDef_t MIUseDefs;

  /// Live-out data for each MBB => U LiveIns (For all Successors of a MBB).
  DenseMap<const MachineBasicBlock *, BitVector> MBBLiveOuts;

  /// Each MachineBasicBlock is assigned a Distance which is
  /// an approximation of MBB->size()*INSTR_SIZE+Some offsets.
  /// This is helpful in quickly finding distance between
  /// a branch and its target.
  /// @note A pass which moves instructions should update this.
  /// @note The data in distance map should be used carefully because
  /// difference in the distances of two MI might not give relative distances
  /// between them. The DistanceMap is mainly useful during pullup.
  DenseMap<const MachineBasicBlock *, unsigned> DistanceMap;

  // Blocks in depth first order
  SmallVector<MachineBasicBlock *, 16> BlocksDepthFirst;

  /// @brief Constructs use-defs of \p MBB by analyzing each MachineOperand.
  /// Collects relevant information so that global liveness can be updated.
  void constructUseDef(MachineBasicBlock *MBB);

  /// Collects used-before-define set of registers.
  /// A register is considered to be completely defined if
  /// 1. The register
  /// 2. Any of its super-reg
  /// 3. All of its subregs
  /// are defined. In these cases the register is not considered as
  /// used-before-defined. In case of partial definition of a register
  /// before its use, only the remaining subregs are included in the use-set.
  /// @note: Assumes that a register can be completely defined, by defining
  /// all of its sub-regs (if any).
  void handlePhysRegUse(MachineOperand *MO, MachineInstr *MI, BitVector &Uses);

  /// Collects defined-before-use set of registers. If there is any
  /// use of register or its aliases then the register is not counted
  /// as defined-before-use
  /// @note: Assumes that a register can be completely defined, by defining
  /// all of its sub-regs (if any).
  void handlePhysRegDef(MachineOperand *MO, MachineInstr *MI, BitVector &Defs);

  /// updateGlobalLiveness - wrapper around another overload
  inline bool updateGlobalLiveness(MachineFunction &Fn);
  bool updateGlobalLiveness(MachineBasicBlock *X, MachineBasicBlock *Y);

  /// updateGlobalLiveness - updates liveness based on
  /// livein and liveout entries.
  bool updateGlobalLiveness(MachineBasicBlock *MBB, BitVector &Defs,
                            BitVector &LiveIns);

  /// update live-ins when live-out has been calculated
  bool updateLiveIns(MachineBasicBlock *MBB, BitVector &LiveIns,
                     const BitVector &LiveOuts);

  bool updateLiveOuts(MachineBasicBlock *MBB, BitVector &LiveOuts);

  /// updateLocalLiveness - update only kill flags of operands.
  inline bool updateLocalLiveness(MachineFunction &Fn);

  /// updateLocalLiveness - update only kill flags of operands.
  bool updateLocalLiveness(MachineBasicBlock *MBB, bool UpdateBundle);

  /// incrementalUpdate - update the liveness when \p MIDelta is moved from
  /// \p From to \p To.
  /// @note: This is extremely fragile now. It 'assumes' that the other
  /// successor(s) of \p To do not use Defs of MIDelta.
  /// It deletes the live-in of the \p From MBB.
  bool incrementalUpdate(MICInstIterType MIDelta, MachineBasicBlock *From,
                         MachineBasicBlock *To);

  /// addNewMBB - inform the LiveVariable Analysis that new MBB has been added.
  /// update the liveness of this new MBB.
  /// @note MBB should be empty. If we want to add an MI, add it after calling
  /// this function.
  void addNewMBB(MachineBasicBlock *MBB);

  void addNewMI(MachineInstr *MI, MachineBasicBlock *MBB);
  unsigned getNumRegs() const { return NumRegs; }

  // Useful for clearing out after passes which move instructions around.
  // e.g. GlobalScheduler.
  void clearDistanceMap() { DistanceMap.clear(); }

  /// Computes \p DistanceMap.
  void generateDistanceMap(const MachineFunction &Fn);

public:
  bool runOnMachineFunction(MachineFunction &Fn, MachineDominatorTree &MDT,
                            MachinePostDominatorTree &MPDT);
};

//===----------------------------------------------------------------------===//
//                    HexagonLiveVariables Functions
//===----------------------------------------------------------------------===//
HexagonLiveVariables::HexagonLiveVariables()
    : MachineFunctionPass(ID), HLVComplete(false),
      HLV(std::make_unique<HexagonLiveVariablesImpl>()) {
  initializeHexagonLiveVariablesPass(*PassRegistry::getPassRegistry());
}

void HexagonLiveVariables::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachinePostDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachinePostDominatorTreeWrapperPass>();
  AU.addPreserved("packets");
  MachineFunctionPass::getAnalysisUsage(AU);
}

void HexagonLiveVariables::recalculate(MachineFunction &MF) {
  if (HLVComplete)
    return;
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &MPDT =
      getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  HLV->runOnMachineFunction(MF, MDT, MPDT);
}

bool HexagonLiveVariables::updateLocalLiveness(MachineFunction &Fn) {
  return HLV->updateLocalLiveness(Fn);
}

bool HexagonLiveVariables::updateLocalLiveness(MachineBasicBlock *MBB,
                                               bool updateBundle) {
  HLV->constructUseDef(MBB); // XXX: This destroys MBBLiveOuts!
  return HLV->updateLocalLiveness(MBB, updateBundle);
}

bool HexagonLiveVariables::incrementalUpdate(MICInstIterType MIDelta,
                                             MachineBasicBlock *From,
                                             MachineBasicBlock *To) {
  assert(MIDelta->getParent() == To);
  assert(From != To);
  return HLV->incrementalUpdate(MIDelta, From, To);
}

void HexagonLiveVariables::addNewMBB(MachineBasicBlock *MBB) {
  assert(MBB->empty());
  HLV->addNewMBB(MBB);
}

void HexagonLiveVariables::addNewMI(MachineInstr *MI, MachineBasicBlock *MBB) {
  HLV->addNewMI(MI, MBB);
}

void HexagonLiveVariables::constructUseDef(MachineBasicBlock *MBB) {
  HLV->constructUseDef(MBB);
}

bool HexagonLiveVariables::runOnMachineFunction(MachineFunction &Fn) {
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &MPDT =
      getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  HLVComplete = !HLV->runOnMachineFunction(Fn, MDT, MPDT);
  return HLVComplete;
}

bool HexagonLiveVariables::isLiveOut(const MachineBasicBlock *MBB,
                                     unsigned Reg) const {
  assert(HLVComplete && "Liveness Analysis not available");
  auto It = HLV->MBBLiveOuts.find(MBB);
  if (It == HLV->MBBLiveOuts.end())
    llvm_unreachable("MBB not found in liveness map");
  if (Reg >= It->second.size())
    llvm_unreachable("Register index out of bounds");
  return It->second[Reg];
}

const BitVector &
HexagonLiveVariables::getLiveOuts(const MachineBasicBlock *MBB) const {
  assert(HLVComplete && "Liveness Analysis not available");
  auto It = HLV->MBBLiveOuts.find(MBB);
  if (It == HLV->MBBLiveOuts.end())
    llvm_unreachable("MBB not found in liveness map");
  return It->second;
}

// Returns true when \p Reg is used within [MIBegin, MIEnd)
// @note: MIBegin and MIEnd should be from same MBB
// @note: It returns just the first use found in the range.
// The Use is closest to MIEnd.
// Takes care of aliases and predicated defs as well.
bool HexagonLiveVariables::isUsedWithin(
    MICInstIterType MIBegin, MICInstIterType MIEnd, unsigned Reg,
    MICInstIterType &Use,
    SmallPtrSet<MachineInstr *, 2> *ExceptionsList) const {
  assert(HLVComplete && "Liveness Analysis not available");
  Use = MIEnd;
  if (MIBegin == MIEnd) // NULL Range.
    return false;
  MICInstIterType MII = MIEnd;
  do {
    --MII;
    if (MII->isBundle() || MII->isDebugInstr())
      continue;
    if (ExceptionsList && ExceptionsList->contains(&*MII))
      continue;
    auto It = HLV->MIUseDefs.find(&*MII);
    assert(It != HLV->MIUseDefs.end());
    for (MCRegAliasIterator AI(Reg, HLV->TRI, true); AI.isValid(); ++AI)
      if (It->second.first[*AI]) {
        Use = MII;
        return true;
      }
  } while (MII != MIBegin);
  return false;
}

// Returns true when \p Reg id defined within [MIBegin, MIEnd)
// @note: MIBegin and MIEnd should be from same MBB
// The Def is closest to MIEnd.
// Takes care of aliases and predicated defs as well.
bool HexagonLiveVariables::isDefinedWithin(MICInstIterType MIBegin,
                                           MICInstIterType MIEnd, unsigned Reg,
                                           MICInstIterType &Def) const {
  assert(HLVComplete && "Liveness Analysis not available");
  Def = MIEnd;
  if (MIBegin == MIEnd) // NULL Range.
    return false;
  MICInstIterType MII = MIEnd;
  do {
    --MII;
    if (MII->isBundle() || MII->isDebugInstr())
      continue;
    auto It = HLV->MIUseDefs.find(&*MII);
    assert(It != HLV->MIUseDefs.end());
    for (MCRegAliasIterator AI(Reg, HLV->TRI, true); AI.isValid(); ++AI)
      if (It->second.second[*AI]) {
        Def = MII;
        return true;
      }
  } while (MII != MIBegin);
  return false;
}

// Returns true if any of the defs of MII is live-in in the MBB.
bool HexagonLiveVariables::isDefLiveIn(const MachineInstr *MI,
                                       const MachineBasicBlock *MBB) const {
  assert(HLVComplete && "Liveness Analysis not available");
  assert(MI && "Invalid machine instruction");
  assert(MBB && "Invalid machine basic block");
  auto It = HLV->MIUseDefs.find(MI);
  assert(It != HLV->MIUseDefs.end() && "Missing MI use/def information");
  BitVector MBBLiveIns(HLV->NumRegs);
  for (MachineBasicBlock::livein_iterator lit = MBB->livein_begin();
       lit != MBB->livein_end(); ++lit) {
    // Include all the aliases of reg *lit.
    for (MCRegAliasIterator AI((*lit).PhysReg, HLV->TRI, true); AI.isValid();
         ++AI)
      MBBLiveIns.set(*AI);
  }
  // Intersect.
  return MBBLiveIns.anyCommon(It->second.second);
}

MBBUseDef_t &HexagonLiveVariables::getMBBUseDefs() { return HLV->MBBUseDefs; }

MIUseDef_t &HexagonLiveVariables::getMIUseDefs() { return HLV->MIUseDefs; }

unsigned HexagonLiveVariables::getDistanceBetween(const MachineBasicBlock *From,
                                                  const MachineBasicBlock *To,
                                                  unsigned BufferPerMBB) const {
  assert(HLV->DistanceMap.find(From) != HLV->DistanceMap.end());
  assert(HLV->DistanceMap.find(To) != HLV->DistanceMap.end());
  unsigned FromSize = HLV->DistanceMap[From];
  if (From == To)
    return FromSize;
  const MachineFunction *MF = From->getParent();
  MachineFunction::const_iterator MBBI = MF->begin();
  unsigned S = BufferPerMBB;
  bool ToFirst = false;
  while (MBBI != MF->end()) {
    const MachineBasicBlock *MBB = &*MBBI;
    if (MBB == From)
      break;
    else if (MBB == To) {
      ToFirst = true;
      break;
    }
    ++MBBI;
  }
  const MachineBasicBlock *ToFind = To;
  if (ToFirst)
    ToFind = From;
  while (MBBI != MF->end()) {
    const MachineBasicBlock *MBB = &*MBBI;
    if (MBB == ToFind)
      break;
    S += HLV->DistanceMap[MBB] + BufferPerMBB;
    ++MBBI;
  }
  if (ToFirst) // Jump in the opposite direction.
    S += FromSize + HLV->DistanceMap[To] + 2 * BufferPerMBB;
  return S;
}

void HexagonLiveVariables::regenerateDistanceMap(const MachineFunction &Fn) {
  HLV->clearDistanceMap();
  HLV->generateDistanceMap(Fn);
}

//===----------------------------------------------------------------------===//
//                    HexagonLiveVariablesImpl Functions
//===----------------------------------------------------------------------===//
bool HexagonLiveVariablesImpl::runOnMachineFunction(
    MachineFunction &Fn, MachineDominatorTree &MDT,
    MachinePostDominatorTree &MPDT) {
  LLVM_DEBUG(dbgs() << "\nHexagon Live Variables";);
  Fn.RenumberBlocks();
  // Update the block numbers in the dominator tree since we preserve it.
  MDT.updateBlockNumbers();
  MPDT.updateBlockNumbers();

  MF = &Fn;
  MRI = &Fn.getRegInfo();
  auto &ST = Fn.getSubtarget<HexagonSubtarget>();
  TRI = ST.getRegisterInfo();
  QII = ST.getInstrInfo();

  NumRegs = TRI->getNumRegs();

  MBBUseDefs.clear();
  MIUseDefs.clear();
  MBBLiveOuts.clear();

  LLVM_DEBUG(dbgs() << "\nNumber of registers in Hexagon is:" << NumRegs);

  PhysRegDef.resize(NumRegs);
  PhysRegUse.resize(NumRegs);

  for (MachineFunction::iterator MBBI = Fn.begin(), E = Fn.end(); MBBI != E;
       ++MBBI) {
    constructUseDef(&*MBBI);
  }
  updateGlobalLiveness(Fn);
  return false;
}

void HexagonLiveVariablesImpl::constructUseDef(MachineBasicBlock *MBB) {
  std::fill(PhysRegDef.begin(), PhysRegDef.end(), (MachineInstr *)0);
  std::fill(PhysRegUse.begin(), PhysRegUse.end(), (MachineInstr *)0);

  // Loop over all of the instructions, processing them.
  std::pair<BitVector, BitVector> &UseDef = MBBUseDefs[MBB];
  // Use before any def in a BB.
  BitVector &Uses = UseDef.first;
  // Defs before any use in a BB.
  BitVector &Defs = UseDef.second;
  // Initializing the LiveOut bit vector.
  BitVector &LiveOuts = MBBLiveOuts[MBB];
  Uses.resize(NumRegs, false);
  Defs.resize(NumRegs, false);
  LiveOuts.resize(NumRegs, false);
  // BitVector might contain set bits out of previous liveness updates.
  Uses.reset();
  Defs.reset();
  LiveOuts.reset();
  LLVM_DEBUG(dbgs() << "\nBB#" << MBB->getNumber(););
  // MBB Number in the MSB 32 bits.
  unsigned MBBInsSize = 0;
  for (MachineBasicBlock::instr_iterator MII = MBB->instr_begin(),
                                         E = MBB->instr_end();
       MII != E; ++MII) {
    MachineInstr *MI = &*MII;
    MBBInsSize += QII->getSize(*MI);
    // TODO: Handle isDebugInstr
    if (MI->isBundle() || MI->isDebugInstr())
      continue;
    LLVM_DEBUG(dbgs() << "\n\n" << *MI;);
    // Clear kill and dead markers. LV will recompute them.
    UseDef_t &MIUseDef = MIUseDefs[MI];
    MIUseDef.first.resize(NumRegs);  // Uses
    MIUseDef.second.resize(NumRegs); // Defs
    MIUseDef.first.reset();          // Uses
    MIUseDef.second.reset();         // Defs

    SmallVector<MachineOperand *, 4> UseRegs;
    SmallVector<MachineOperand *, 4> DefRegs;
    SmallVector<unsigned, 1> RegMasks;
    // Process all of the operands of the instruction...
    unsigned NumOperandsToProcess = MI->getNumOperands();
    for (unsigned i = 0; i != NumOperandsToProcess; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isRegMask()) {
        // Assuming that predicated defs are not defs, for now.
        if (!QII->isPredicated(*MI))
          DefRegs.push_back(&MO);
        continue;
      }
      if (!MO.isReg() || MO.getReg() == 0)
        continue;
      unsigned Reg = MO.getReg();
      if (MO.isUse()) {
        // Assuming that the kill-flags on call-instructions are correct.
        MO.setIsKill(false);
        UseRegs.push_back(&MO);
        MIUseDef.first.set(Reg);
      } else /*MO.isDef()*/ {
        assert(MO.isDef());
        if (!QII->isPredicated(*MI) && !MI->isKill()) {
          // Assuming that predicated defs are not defs, for now.
          // KILL instructions are no-ops
          MO.setIsDead(false);
          DefRegs.push_back(&MO);
        }
        MIUseDef.second.set(Reg); // Set all defs (including predicated).
      }
    }
    // Process all uses.
    for (unsigned i = 0, e = UseRegs.size(); i != e; ++i)
      handlePhysRegUse(UseRegs[i], MI, Uses);
    // Process all defs.
    for (unsigned i = 0, e = DefRegs.size(); i != e; ++i)
      handlePhysRegDef(DefRegs[i], MI, Defs);
  }
  DistanceMap[MBB] = MBBInsSize;
}

void HexagonLiveVariablesImpl::handlePhysRegUse(MachineOperand *MO,
                                                MachineInstr *MI,
                                                BitVector &Uses) {
  unsigned Reg = MO->getReg();
  LLVM_DEBUG(dbgs() << "\nLooking at:";);
  // If the reg/super-reg is already defined in this MBB => return.
  for (MCSuperRegIterator SupI(Reg, TRI, true); SupI.isValid(); ++SupI) {
    LLVM_DEBUG(dbgs() << printReg(*SupI, TRI););
    if (PhysRegDef[*SupI])
      return;
  }
  // Handle if sub-regs are defined.
  SmallVector<unsigned, 2> undefSubRegs;
  bool subRegDefined = false;
  for (MCSubRegIterator SubI(Reg, TRI); SubI.isValid(); ++SubI) {
    LLVM_DEBUG(dbgs() << printReg(*SubI, TRI););
    if (PhysRegDef[*SubI])
      subRegDefined = true;
    else
      undefSubRegs.push_back(*SubI);
  }

  LLVM_DEBUG(dbgs() << "\nUses:");
  if (undefSubRegs.empty()) {
    if (!subRegDefined) { // None of the subregs are defined.
      // Include all subregs (including self) to the uses.
      for (MCSubRegIterator SubI(Reg, TRI, true); SubI.isValid(); ++SubI) {
        LLVM_DEBUG(dbgs() << printReg(*SubI, TRI));
        PhysRegUse[*SubI] = MI;
        Uses.set(*SubI);
      }
    } // All subregs defined.
    return;
  }
  // Some subregs are defined.
  for (unsigned i = 0; i < undefSubRegs.size(); ++i) {
    LLVM_DEBUG(dbgs() << printReg(undefSubRegs[i], TRI));
    PhysRegUse[undefSubRegs[i]] = MI;
    Uses.set(undefSubRegs[i]);
  }
}

// Assumes that an MI cannot have a reg and its super/sub reg as uses.
void HexagonLiveVariablesImpl::handlePhysRegDef(MachineOperand *MO,
                                                MachineInstr *MI,
                                                BitVector &Defs) {
  auto SetRegDef = [&](unsigned Reg) -> void {
    PhysRegDef[Reg] = MI;
    for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
      if (PhysRegUse[*AI]) {
        LLVM_DEBUG(dbgs() << "\nUsed in current BB:" << printReg(*AI, TRI));
        return;
      }
    }
    LLVM_DEBUG(dbgs() << "\nDefs:" << printReg(Reg, TRI));
    Defs.set(Reg);
  };

  if (MO->isReg()) {
    SetRegDef(MO->getReg());
  } else if (MO->isRegMask()) {
    for (unsigned R = 1, NR = TRI->getNumRegs(); R != NR; ++R)
      if (MO->clobbersPhysReg(R))
        SetRegDef(R);
  }
}

namespace {
struct BlockState {
  bool SuccQueued : 1;
  bool Done : 1;
  BlockState() : SuccQueued(false), Done(false) {}
};
} // namespace

// Populates 'Blocks' with basic blocks of 'Fn' in depth-first order
static void gatherBlocksDF(MachineFunction &Fn,
                           SmallVectorImpl<MachineBasicBlock *> *Blocks) {
  Blocks->clear();
  Blocks->reserve(Fn.size());

  SmallVector<BlockState, 16> State(Fn.size());
  SmallVector<MachineBasicBlock *, 16> WorkStack;
  WorkStack.push_back(&Fn.front());
  while (!WorkStack.empty()) {
    MachineBasicBlock *W = WorkStack.back();
    BlockState &WState = State[W->getNumber()];
    if (WState.Done) {
      WorkStack.pop_back();
      continue;
    }
    if (W->succ_empty() || WState.SuccQueued) {
      WorkStack.pop_back();
      Blocks->push_back(W);
      WState.SuccQueued = true;
      WState.Done = true;
      continue;
    }
    WState.SuccQueued = true;
    for (MachineBasicBlock::succ_iterator I = W->succ_begin(),
                                          E = W->succ_end();
         I != E; ++I) {
      MachineBasicBlock *S = *I;
      if (State[S->getNumber()].SuccQueued)
        continue;
      WorkStack.push_back(S);
    }
  }

  LLVM_DEBUG(
      dbgs() << "gatherBlocksDF: {";
      for (SmallVectorImpl<MachineBasicBlock *>::iterator B = Blocks->begin(),
           BE = Blocks->end();
           B != BE; ++B) { dbgs() << " BB#" << (*B)->getNumber(); } dbgs()
      << " }\n";);
}

bool HexagonLiveVariablesImpl::updateGlobalLiveness(MachineFunction &Fn) {
  bool Changed = false;
  // Removing live-ins and recomputing.
  MachineFunction::iterator I = Fn.begin(), E = Fn.end();
  // Not touching the live-ins of entry basic block.
  for (++I; I != E; ++I) {
    std::vector<MachineBasicBlock::RegisterMaskPair> OldLiveIn(
        I->livein_begin(), I->livein_end());
    for (unsigned i = 0; i < OldLiveIn.size(); ++i)
      I->removeLiveIn(OldLiveIn[i].PhysReg);
  }

  gatherBlocksDF(Fn, &BlocksDepthFirst);

  BitVector Defs;
  BitVector LiveIns;
  bool Repeat;
  do {
    Repeat = false;
    for (SmallVectorImpl<MachineBasicBlock *>::iterator
             B = BlocksDepthFirst.begin(),
             BE = BlocksDepthFirst.end();
         B != BE; ++B) {
      Repeat |= updateGlobalLiveness(*B, Defs, LiveIns);
    }
    Changed |= Repeat;
  } while (Repeat);

  Changed |= updateLocalLiveness(Fn);
  return Changed;
}

bool HexagonLiveVariablesImpl::updateGlobalLiveness(MachineBasicBlock *X,
                                                    MachineBasicBlock *Y) {
  assert(X && "Invalid start block");
  assert(Y && "Invalid end block");

  bool Changed = false;
  BitVector Defs;
  BitVector LiveIns;

  const SmallVectorImpl<MachineBasicBlock *>::iterator BE =
      BlocksDepthFirst.end();
  SmallVectorImpl<MachineBasicBlock *>::iterator B;
  for (B = BlocksDepthFirst.begin(); (B != BE); ++B) {
    if (*B == X)
      break;
    if (*B == Y)
      break;
  }

  bool Repeat;
  do {
    Repeat = false;
    for (; B != BE; ++B)
      Repeat |= updateGlobalLiveness(*B, Defs, LiveIns);
    Changed |= Repeat;
    B = BlocksDepthFirst.begin();
  } while (Repeat);

  return Changed;
}

// Defs and LiveIns could be local variables within updateGlobalLiveness, but
// have been pulled out to (hopefully) improve performance.
bool HexagonLiveVariablesImpl::updateGlobalLiveness(MachineBasicBlock *MBB,
                                                    BitVector &Defs,
                                                    BitVector &LiveIns) {
  LLVM_DEBUG(dbgs() << "\nTrying to Update Liveness MBB#" << MBB->getNumber());
  bool Changed = false;
  LLVM_DEBUG(dbgs() << "\nUpdating Liveness MBB#" << MBB->getNumber());
  // Update live-outs
  auto LiveOutIt = MBBLiveOuts.find(MBB);
  if (LiveOutIt == MBBLiveOuts.end())
    LiveOutIt = MBBLiveOuts.insert({MBB, BitVector(NumRegs)}).first;
  BitVector &LiveOuts = LiveOutIt->second;
  for (MachineBasicBlock::succ_iterator MBBSucc = MBB->succ_begin();
       MBBSucc != MBB->succ_end(); ++MBBSucc) {
    MachineBasicBlock *Succ = *MBBSucc;
    LLVM_DEBUG(dbgs() << "\n\t\tAdding LiveOut:";);
    for (MachineBasicBlock::livein_iterator LI = Succ->livein_begin(),
                                            LE = Succ->livein_end();
         LI != LE; ++LI) {
      if (!LiveOuts[(*LI).PhysReg]) {
        LLVM_DEBUG(dbgs() << " " << printReg((*LI).PhysReg, TRI););
        LiveOuts.set((*LI).PhysReg);
        Changed = true;
      }
    }
  }
  LLVM_DEBUG(dbgs() << "\nUpdated Successors of MBB#" << MBB->getNumber());
  // Update live-ins
  Changed |= updateLiveIns(MBB, LiveIns, LiveOuts);

  return Changed;
}

// update live-ins when live-out has been calculated
bool HexagonLiveVariablesImpl::updateLiveIns(MachineBasicBlock *MBB,
                                             BitVector &LiveIns,
                                             const BitVector &LiveOuts) {
  LLVM_DEBUG(dbgs() << "\n[updateLiveIns] MBB#" << MBB->getNumber());
  bool Changed = false;
  const std::pair<BitVector, BitVector> &UseDefs = MBBUseDefs[MBB];
  LiveIns = LiveOuts;
  // LiveIns = (LiveOuts - Defs) | Uses
  // Equivalent to: LiveIns = (LiveOuts & ~Defs) | Uses
  LiveIns.reset(UseDefs.second);
  LiveIns |= UseDefs.first;
  LLVM_DEBUG(dbgs() << "\n\t\tAdded LiveIn:";);
  for (int i = LiveIns.find_first(); i >= 0; i = LiveIns.find_next(i)) {
    // TODO: remove costly check of MBB->isLiveIn when fully functional.
    if (!MBB->isLiveIn(i) && MRI->isAllocatable(i)) {
      LLVM_DEBUG(dbgs() << " " << printReg(i, TRI));
      MBB->addLiveIn(i);
      Changed = true;
    }
  }
  return Changed;
}

bool HexagonLiveVariablesImpl::updateLiveOuts(MachineBasicBlock *MBB,
                                              BitVector &LiveOuts) {
  bool Changed = false;
  for (auto SI = MBB->succ_begin(), SE = MBB->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock *SB = *SI;
    for (auto I = SB->livein_begin(), E = SB->livein_end(); I != E; ++I) {
      unsigned R = (*I).PhysReg;
      if (LiveOuts[R])
        continue;
      LiveOuts.set(R);
      Changed = true;
    }
  }
  return Changed;
}

bool HexagonLiveVariablesImpl::updateLocalLiveness(MachineFunction &Fn) {
  LLVM_DEBUG(dbgs() << "\n[updateLocalLiveness]");
  for (MachineFunction::iterator B = Fn.begin(), E = Fn.end(); B != E; ++B)
    updateLocalLiveness(&*B, false);
  return true;
}

bool HexagonLiveVariablesImpl::updateLocalLiveness(MachineBasicBlock *MBB,
                                                   bool UpdateBundle) {
  assert(MBB && "Invalid basic block");
  LLVM_DEBUG(dbgs() << "\n[updateLocalLiveness] MBB#" << MBB->getNumber());

  BitVector &LiveOut = MBBLiveOuts[MBB];
  updateLiveOuts(MBB, LiveOut);

  BitVector Used = LiveOut;
  SmallVector<MachineInstr *, 2> BundleHeads;
  // Bottom up traversal of MBB.
  for (MachineBasicBlock::reverse_instr_iterator MII = MBB->instr_rbegin(),
                                                 MIREnd = MBB->instr_rend();
       MII != MIREnd; ++MII) {
    MachineInstr *MI = &*MII;
    // The bundle liveness is updated differently.
    if (MI->isBundle()) {
      if (UpdateBundle)
        BundleHeads.push_back(MI);
      continue;
    }
    if (MI->isDebugInstr()) // DBG_VALUE may have invalid reg.
      continue;
    SmallVector<MachineOperand *, 4> UseRegs;
    SmallVector<MachineOperand *, 2> DefRegs;
    for (unsigned i = 0; i < MI->getNumOperands(); ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg()) { // DBG_VALUE may have invalid reg.
        if (MO.isUse())
          UseRegs.push_back(&MO);
        else { // Def
          if (!QII->isPredicated(*MI) && !MI->isKill()) {
            // Assuming that predicated defs are not defs, for now.
            // KILL instructions are no-ops
            DefRegs.push_back(&MO);
          }
        }
      } else if (MO.isRegMask()) {
        if (!QII->isPredicated(*MI))
          DefRegs.push_back(&MO);
      }
    }
    // In case of a def. remove Reg and its sub-regs from Used list
    // such that uses in the same MI can be marked as kill.
    auto RemoveDef = [&](unsigned Reg, bool Implicit) -> void {
      for (MCSubRegIterator SI(Reg, TRI, true); SI.isValid(); ++SI) {
        Used.reset(*SI);
        if (Implicit) {
          // For implicit defs, check if there is an implicit use of an
          // aliased register. If so, mark the aliased reg as used.
          for (auto *UseOp : UseRegs)
            if (UseOp->isImplicit() && TRI->regsOverlap(*SI, UseOp->getReg()))
              Used.set(UseOp->getReg());
        }
      }
    };
    for (unsigned i = 0; i < DefRegs.size(); ++i) {
      MachineOperand &MO = *DefRegs[i];
      if (MO.isReg()) {
        RemoveDef(MO.getReg(), MO.isImplicit());
      } else if (MO.isRegMask()) {
        for (unsigned R = 1, NR = TRI->getNumRegs(); R != NR; ++R)
          if (MO.clobbersPhysReg(R))
            RemoveDef(R, true);
      }
    }
    // The order is important as we are looking from right to left.
    for (unsigned i = UseRegs.size(); i > 0;) {
      --i;
      unsigned UseReg = UseRegs[i]->getReg();
      bool Killed = true;
      for (MCRegAliasIterator AI(UseReg, TRI, true); AI.isValid(); ++AI) {
        if (Used[*AI])
          Killed = false;
      }
      Used.set(UseReg);
      if (Killed && !UseRegs[i]->isDebug())
        UseRegs[i]->setIsKill(true);
    }
  }
  // Recreates bundle for updating liveness.
  for (SmallVectorImpl<MachineInstr *>::iterator MII = BundleHeads.begin();
       MII != BundleHeads.end(); ++MII) {
    MachineInstr *MI = *MII;
    assert(MI && "Invalid bundle head");
    assert(MI->isBundle() && "Expected a bundle head instruction");
    assert(MI->getParent() == MBB && "Bundle head not in expected block");
    MachineBasicBlock::instr_iterator BS = MI->getIterator();
    MachineBasicBlock::instr_iterator BE = getBundleEnd(BS);
    for (++BS; BS != BE; ++BS)
      // Remove from bundle so that BUNDLE head can be erased.
      BS->unbundleFromPred();

    BS = MI->getIterator();
    ++BS;
    bool memShufDisabled = QII->getBundleNoShuf(*MI);
    MI->eraseFromParent();
    finalizeBundle(*MBB, BS, BE);
    MachineBasicBlock::instr_iterator BundleMII = std::prev(BS);
    if (memShufDisabled)
      QII->setBundleNoShuf(BundleMII);
  }
  return true;
}

// It deletes the live-in of the \p From MBB.
bool HexagonLiveVariablesImpl::incrementalUpdate(MICInstIterType MIDelta,
                                                 MachineBasicBlock *From,
                                                 MachineBasicBlock *To) {
  while (!From->livein_empty())
    From->removeLiveIn((*From->livein_begin()).PhysReg);
  // Handle MI use-def of From.
  constructUseDef(From);
  // Handle MI use-def of To.
  constructUseDef(To);
  // Calculate live-in of From and To
  // Reuse this by setting all MBBs except From and To as visited.
  updateGlobalLiveness(From, To);
  // Update local liveness of To.
  updateLocalLiveness(From, true);
  updateLocalLiveness(To, true);

  // Do this after the liveness update because MIDelta might not be in the
  // MIUseDefs before liveness update (since MIDelta might be newly inserted).
  MIUseDef_t::const_iterator MIUseDef = MIUseDefs.find(&*MIDelta);
  if (MIUseDef == MIUseDefs.end())
    llvm_unreachable("MIDelta not found in MIUseDefs after liveness update");
  const BitVector &Defs = MIUseDef->second.second;
  int Reg = Defs.find_first();
  // Adding all the defs as live-ins. This is conservative approach but we
  // need to add them so as to avoid dealing with callee saved registers and
  // any unwanted errors in liveness that might arise.
  while (Reg >= 0) {
    From->addLiveIn(Reg);
    Reg = Defs.find_next(Reg);
  }
  return true;
}

void HexagonLiveVariablesImpl::addNewMBB(MachineBasicBlock *MBB) {
  // Resize and init.
  constructUseDef(MBB); // This is to set up some containers for MBB.
  gatherBlocksDF(*MBB->getParent(), &BlocksDepthFirst);
  updateGlobalLiveness(MBB, MBB);
}

// TODO: This is a slow implementation because constructUseDef destroys
// the MBBLiveOuts which is generated again by updateGlobalLiveness.
void HexagonLiveVariablesImpl::addNewMI(MachineInstr *MI,
                                        MachineBasicBlock *MBB) {
  constructUseDef(MBB); // This is to set up some containers for MBB.
  updateGlobalLiveness(MBB, MBB);
}

void HexagonLiveVariablesImpl::generateDistanceMap(const MachineFunction &Fn) {
  assert(DistanceMap.empty() && "DistanceMap not empty, first clear!");
  for (MachineFunction::const_iterator MBBI = Fn.begin(), E = Fn.end();
       MBBI != E; ++MBBI) {
    const MachineBasicBlock *MBB = &*MBBI;
    unsigned MBBInsSize = 0;
    for (MachineBasicBlock::const_instr_iterator MII = MBB->instr_begin(),
                                                 E = MBB->instr_end();
         MII != E; ++MII) {
      const MachineInstr *MI = &*MII;
      MBBInsSize += QII->getSize(*MI);
    }
    DistanceMap[MBB] = MBBInsSize;
  }
}
