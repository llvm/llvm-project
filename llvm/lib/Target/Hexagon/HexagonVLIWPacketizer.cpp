//===----- HexagonPacketizer.cpp - vliw packetizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a simple VLIW packetizer using DFA. The packetizer works on
// machine basic blocks. For each instruction I in BB, the packetizer consults
// the DFA to see if machine resources are available to execute I. If so, the
// packetizer checks if I depends on any instruction J in the current packet.
// If no dependency is found, I is added to current packet and machine resource
// is marked as taken. If any dependency is found, a target API call is made to
// prune the dependence.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "packets"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/LatencyPriorityQueue.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "Hexagon.h"
#include "HexagonTargetMachine.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonMachineFunctionInfo.h"

#include <map>

using namespace llvm;

namespace {
  class HexagonPacketizer : public MachineFunctionPass {

  public:
    static char ID;
    HexagonPacketizer() : MachineFunctionPass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
      AU.addRequired<MachineLoopInfo>();
      AU.addPreserved<MachineLoopInfo>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    const char *getPassName() const {
      return "Hexagon Packetizer";
    }

    bool runOnMachineFunction(MachineFunction &Fn);
  };
  char HexagonPacketizer::ID = 0;

  class HexagonPacketizerList : public VLIWPacketizerList {

  private:

    // Has the instruction been promoted to a dot-new instruction.
    bool PromotedToDotNew;

    // Has the instruction been glued to allocframe.
    bool GlueAllocframeStore;

    // Has the feeder instruction been glued to new value jump.
    bool GlueToNewValueJump;

    // Check if there is a dependence between some instruction already in this
    // packet and this instruction.
    bool Dependence;

    // Only check for dependence if there are resources available to
    // schedule this instruction.
    bool FoundSequentialDependence;

  public:
    // Ctor.
    HexagonPacketizerList(MachineFunction &MF, MachineLoopInfo &MLI,
                          MachineDominatorTree &MDT);

    // initPacketizerState - initialize some internal flags.
    void initPacketizerState();

    // ignorePseudoInstruction - Ignore bundling of pseudo instructions.
    bool ignorePseudoInstruction(MachineInstr *MI, MachineBasicBlock *MBB);

    // isSoloInstruction - return true if instruction MI can not be packetized
    // with any other instruction, which means that MI itself is a packet.
    bool isSoloInstruction(MachineInstr *MI);

    // isLegalToPacketizeTogether - Is it legal to packetize SUI and SUJ
    // together.
    bool isLegalToPacketizeTogether(SUnit *SUI, SUnit *SUJ);

    // isLegalToPruneDependencies - Is it legal to prune dependece between SUI
    // and SUJ.
    bool isLegalToPruneDependencies(SUnit *SUI, SUnit *SUJ);

    MachineBasicBlock::iterator addToPacket(MachineInstr *MI);
  private:
    bool IsCallDependent(MachineInstr* MI, SDep::Kind DepType, unsigned DepReg);
    bool PromoteToDotNew(MachineInstr* MI, SDep::Kind DepType,
                    MachineBasicBlock::iterator &MII,
                    const TargetRegisterClass* RC);
    bool CanPromoteToDotNew(MachineInstr* MI, SUnit* PacketSU,
                    unsigned DepReg,
                    std::map <MachineInstr*, SUnit*> MIToSUnit,
                    MachineBasicBlock::iterator &MII,
                    const TargetRegisterClass* RC);
    bool CanPromoteToNewValue(MachineInstr* MI, SUnit* PacketSU,
                    unsigned DepReg,
                    std::map <MachineInstr*, SUnit*> MIToSUnit,
                    MachineBasicBlock::iterator &MII);
    bool CanPromoteToNewValueStore(MachineInstr* MI, MachineInstr* PacketMI,
                    unsigned DepReg,
                    std::map <MachineInstr*, SUnit*> MIToSUnit);
    bool DemoteToDotOld(MachineInstr* MI);
    bool ArePredicatesComplements(MachineInstr* MI1, MachineInstr* MI2,
                    std::map <MachineInstr*, SUnit*> MIToSUnit);
    bool RestrictingDepExistInPacket(MachineInstr*,
                    unsigned, std::map <MachineInstr*, SUnit*>);
    bool isNewifiable(MachineInstr* MI);
    bool isCondInst(MachineInstr* MI);
    bool IsNewifyStore (MachineInstr* MI);
    bool tryAllocateResourcesForConstExt(MachineInstr* MI);
    bool canReserveResourcesForConstExt(MachineInstr *MI);
    void reserveResourcesForConstExt(MachineInstr* MI);
    bool isNewValueInst(MachineInstr* MI);
    bool isDotNewInst(MachineInstr* MI);
  };
}

// HexagonPacketizerList Ctor.
HexagonPacketizerList::HexagonPacketizerList(
  MachineFunction &MF, MachineLoopInfo &MLI,MachineDominatorTree &MDT)
  : VLIWPacketizerList(MF, MLI, MDT, true){
}

bool HexagonPacketizer::runOnMachineFunction(MachineFunction &Fn) {
  const TargetInstrInfo *TII = Fn.getTarget().getInstrInfo();
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();

  // Instantiate the packetizer.
  HexagonPacketizerList Packetizer(Fn, MLI, MDT);

  // DFA state table should not be empty.
  assert(Packetizer.getResourceTracker() && "Empty DFA table!");

  //
  // Loop over all basic blocks and remove KILL pseudo-instructions
  // These instructions confuse the dependence analysis. Consider:
  // D0 = ...   (Insn 0)
  // R0 = KILL R0, D0 (Insn 1)
  // R0 = ... (Insn 2)
  // Here, Insn 1 will result in the dependence graph not emitting an output
  // dependence between Insn 0 and Insn 2. This can lead to incorrect
  // packetization
  //
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    MachineBasicBlock::iterator End = MBB->end();
    MachineBasicBlock::iterator MI = MBB->begin();
    while (MI != End) {
      if (MI->isKill()) {
        MachineBasicBlock::iterator DeleteMI = MI;
        ++MI;
        MBB->erase(DeleteMI);
        End = MBB->end();
        continue;
      }
      ++MI;
    }
  }

  // Loop over all of the basic blocks.
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB) {
    // Find scheduling regions and schedule / packetize each region.
    unsigned RemainingCount = MBB->size();
    for(MachineBasicBlock::iterator RegionEnd = MBB->end();
        RegionEnd != MBB->begin();) {
      // The next region starts above the previous region. Look backward in the
      // instruction stream until we find the nearest boundary.
      MachineBasicBlock::iterator I = RegionEnd;
      for(;I != MBB->begin(); --I, --RemainingCount) {
        if (TII->isSchedulingBoundary(llvm::prior(I), MBB, Fn))
          break;
      }
      I = MBB->begin();

      // Skip empty scheduling regions.
      if (I == RegionEnd) {
        RegionEnd = llvm::prior(RegionEnd);
        --RemainingCount;
        continue;
      }
      // Skip regions with one instruction.
      if (I == llvm::prior(RegionEnd)) {
        RegionEnd = llvm::prior(RegionEnd);
        continue;
      }

      Packetizer.PacketizeMIs(MBB, I, RegionEnd);
      RegionEnd = I;
    }
  }

  return true;
}


static bool IsIndirectCall(MachineInstr* MI) {
  return ((MI->getOpcode() == Hexagon::CALLR) ||
          (MI->getOpcode() == Hexagon::CALLRv3));
}

// Reserve resources for constant extender. Trigure an assertion if
// reservation fail.
void HexagonPacketizerList::reserveResourcesForConstExt(MachineInstr* MI) {
  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  MachineInstr *PseudoMI = MI->getParent()->getParent()->CreateMachineInstr(
                                  QII->get(Hexagon::IMMEXT), MI->getDebugLoc());

  if (ResourceTracker->canReserveResources(PseudoMI)) {
    ResourceTracker->reserveResources(PseudoMI);
    MI->getParent()->getParent()->DeleteMachineInstr(PseudoMI);
  } else {
    MI->getParent()->getParent()->DeleteMachineInstr(PseudoMI);
    llvm_unreachable("can not reserve resources for constant extender.");
  }
  return;
}

bool HexagonPacketizerList::canReserveResourcesForConstExt(MachineInstr *MI) {
  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  assert(QII->isExtended(MI) &&
         "Should only be called for constant extended instructions");
  MachineFunction *MF = MI->getParent()->getParent();
  MachineInstr *PseudoMI = MF->CreateMachineInstr(QII->get(Hexagon::IMMEXT),
                                                  MI->getDebugLoc());
  bool CanReserve = ResourceTracker->canReserveResources(PseudoMI);
  MF->DeleteMachineInstr(PseudoMI);
  return CanReserve;
}

// Allocate resources (i.e. 4 bytes) for constant extender. If succeed, return
// true, otherwise, return false.
bool HexagonPacketizerList::tryAllocateResourcesForConstExt(MachineInstr* MI) {
  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  MachineInstr *PseudoMI = MI->getParent()->getParent()->CreateMachineInstr(
                                  QII->get(Hexagon::IMMEXT), MI->getDebugLoc());

  if (ResourceTracker->canReserveResources(PseudoMI)) {
    ResourceTracker->reserveResources(PseudoMI);
    MI->getParent()->getParent()->DeleteMachineInstr(PseudoMI);
    return true;
  } else {
    MI->getParent()->getParent()->DeleteMachineInstr(PseudoMI);
    return false;
  }
}


bool HexagonPacketizerList::IsCallDependent(MachineInstr* MI,
                                          SDep::Kind DepType,
                                          unsigned DepReg) {

  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  const HexagonRegisterInfo* QRI =
              (const HexagonRegisterInfo *) TM.getRegisterInfo();

  // Check for lr dependence
  if (DepReg == QRI->getRARegister()) {
    return true;
  }

  if (QII->isDeallocRet(MI)) {
    if (DepReg == QRI->getFrameRegister() ||
        DepReg == QRI->getStackRegister())
      return true;
  }

  // Check if this is a predicate dependence
  const TargetRegisterClass* RC = QRI->getMinimalPhysRegClass(DepReg);
  if (RC == &Hexagon::PredRegsRegClass) {
    return true;
  }

  //
  // Lastly check for an operand used in an indirect call
  // If we had an attribute for checking if an instruction is an indirect call,
  // then we could have avoided this relatively brittle implementation of
  // IsIndirectCall()
  //
  // Assumes that the first operand of the CALLr is the function address
  //
  if (IsIndirectCall(MI) && (DepType == SDep::Data)) {
    MachineOperand MO = MI->getOperand(0);
    if (MO.isReg() && MO.isUse() && (MO.getReg() == DepReg)) {
      return true;
    }
  }

  return false;
}

static bool IsRegDependence(const SDep::Kind DepType) {
  return (DepType == SDep::Data || DepType == SDep::Anti ||
          DepType == SDep::Output);
}

static bool IsDirectJump(MachineInstr* MI) {
  return (MI->getOpcode() == Hexagon::JMP);
}

static bool IsSchedBarrier(MachineInstr* MI) {
  switch (MI->getOpcode()) {
  case Hexagon::BARRIER:
    return true;
  }
  return false;
}

static bool IsControlFlow(MachineInstr* MI) {
  return (MI->getDesc().isTerminator() || MI->getDesc().isCall());
}

bool HexagonPacketizerList::isNewValueInst(MachineInstr* MI) {
  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  if (QII->isNewValueJump(MI))
    return true;

  if (QII->isNewValueStore(MI))
    return true;

  return false;
}

// Function returns true if an instruction can be promoted to the new-value
// store. It will always return false for v2 and v3.
// It lists all the conditional and unconditional stores that can be promoted
// to the new-value stores.

bool HexagonPacketizerList::IsNewifyStore (MachineInstr* MI) {
  const HexagonRegisterInfo* QRI =
                          (const HexagonRegisterInfo *) TM.getRegisterInfo();
  switch (MI->getOpcode())
  {
    // store byte
    case Hexagon::STrib:
    case Hexagon::STrib_indexed:
    case Hexagon::STrib_indexed_shl_V4:
    case Hexagon::STrib_shl_V4:
    case Hexagon::STrib_GP_V4:
    case Hexagon::STb_GP_V4:
    case Hexagon::POST_STbri:
    case Hexagon::STrib_cPt:
    case Hexagon::STrib_cdnPt_V4:
    case Hexagon::STrib_cNotPt:
    case Hexagon::STrib_cdnNotPt_V4:
    case Hexagon::STrib_indexed_cPt:
    case Hexagon::STrib_indexed_cdnPt_V4:
    case Hexagon::STrib_indexed_cNotPt:
    case Hexagon::STrib_indexed_cdnNotPt_V4:
    case Hexagon::STrib_indexed_shl_cPt_V4:
    case Hexagon::STrib_indexed_shl_cdnPt_V4:
    case Hexagon::STrib_indexed_shl_cNotPt_V4:
    case Hexagon::STrib_indexed_shl_cdnNotPt_V4:
    case Hexagon::POST_STbri_cPt:
    case Hexagon::POST_STbri_cdnPt_V4:
    case Hexagon::POST_STbri_cNotPt:
    case Hexagon::POST_STbri_cdnNotPt_V4:
    case Hexagon::STb_GP_cPt_V4:
    case Hexagon::STb_GP_cNotPt_V4:
    case Hexagon::STb_GP_cdnPt_V4:
    case Hexagon::STb_GP_cdnNotPt_V4:
    case Hexagon::STrib_GP_cPt_V4:
    case Hexagon::STrib_GP_cNotPt_V4:
    case Hexagon::STrib_GP_cdnPt_V4:
    case Hexagon::STrib_GP_cdnNotPt_V4:

    // store halfword
    case Hexagon::STrih:
    case Hexagon::STrih_indexed:
    case Hexagon::STrih_indexed_shl_V4:
    case Hexagon::STrih_shl_V4:
    case Hexagon::STrih_GP_V4:
    case Hexagon::STh_GP_V4:
    case Hexagon::POST_SThri:
    case Hexagon::STrih_cPt:
    case Hexagon::STrih_cdnPt_V4:
    case Hexagon::STrih_cNotPt:
    case Hexagon::STrih_cdnNotPt_V4:
    case Hexagon::STrih_indexed_cPt:
    case Hexagon::STrih_indexed_cdnPt_V4:
    case Hexagon::STrih_indexed_cNotPt:
    case Hexagon::STrih_indexed_cdnNotPt_V4:
    case Hexagon::STrih_indexed_shl_cPt_V4:
    case Hexagon::STrih_indexed_shl_cdnPt_V4:
    case Hexagon::STrih_indexed_shl_cNotPt_V4:
    case Hexagon::STrih_indexed_shl_cdnNotPt_V4:
    case Hexagon::POST_SThri_cPt:
    case Hexagon::POST_SThri_cdnPt_V4:
    case Hexagon::POST_SThri_cNotPt:
    case Hexagon::POST_SThri_cdnNotPt_V4:
    case Hexagon::STh_GP_cPt_V4:
    case Hexagon::STh_GP_cNotPt_V4:
    case Hexagon::STh_GP_cdnPt_V4:
    case Hexagon::STh_GP_cdnNotPt_V4:
    case Hexagon::STrih_GP_cPt_V4:
    case Hexagon::STrih_GP_cNotPt_V4:
    case Hexagon::STrih_GP_cdnPt_V4:
    case Hexagon::STrih_GP_cdnNotPt_V4:

    // store word
    case Hexagon::STriw:
    case Hexagon::STriw_indexed:
    case Hexagon::STriw_indexed_shl_V4:
    case Hexagon::STriw_shl_V4:
    case Hexagon::STriw_GP_V4:
    case Hexagon::STw_GP_V4:
    case Hexagon::POST_STwri:
    case Hexagon::STriw_cPt:
    case Hexagon::STriw_cdnPt_V4:
    case Hexagon::STriw_cNotPt:
    case Hexagon::STriw_cdnNotPt_V4:
    case Hexagon::STriw_indexed_cPt:
    case Hexagon::STriw_indexed_cdnPt_V4:
    case Hexagon::STriw_indexed_cNotPt:
    case Hexagon::STriw_indexed_cdnNotPt_V4:
    case Hexagon::STriw_indexed_shl_cPt_V4:
    case Hexagon::STriw_indexed_shl_cdnPt_V4:
    case Hexagon::STriw_indexed_shl_cNotPt_V4:
    case Hexagon::STriw_indexed_shl_cdnNotPt_V4:
    case Hexagon::POST_STwri_cPt:
    case Hexagon::POST_STwri_cdnPt_V4:
    case Hexagon::POST_STwri_cNotPt:
    case Hexagon::POST_STwri_cdnNotPt_V4:
    case Hexagon::STw_GP_cPt_V4:
    case Hexagon::STw_GP_cNotPt_V4:
    case Hexagon::STw_GP_cdnPt_V4:
    case Hexagon::STw_GP_cdnNotPt_V4:
    case Hexagon::STriw_GP_cPt_V4:
    case Hexagon::STriw_GP_cNotPt_V4:
    case Hexagon::STriw_GP_cdnPt_V4:
    case Hexagon::STriw_GP_cdnNotPt_V4:
        return QRI->Subtarget.hasV4TOps();
  }
  return false;
}

static bool IsLoopN(MachineInstr *MI) {
  return (MI->getOpcode() == Hexagon::LOOP0_i ||
          MI->getOpcode() == Hexagon::LOOP0_r);
}

/// DoesModifyCalleeSavedReg - Returns true if the instruction modifies a
/// callee-saved register.
static bool DoesModifyCalleeSavedReg(MachineInstr *MI,
                                     const TargetRegisterInfo *TRI) {
  for (const uint16_t *CSR = TRI->getCalleeSavedRegs(); *CSR; ++CSR) {
    unsigned CalleeSavedReg = *CSR;
    if (MI->modifiesRegister(CalleeSavedReg, TRI))
      return true;
  }
  return false;
}

// Return the new value instruction for a given store.
static int GetDotNewOp(const int opc) {
  switch (opc) {
  default: llvm_unreachable("Unknown .new type");
  // store new value byte
  case Hexagon::STrib:
    return Hexagon::STrib_nv_V4;

  case Hexagon::STrib_indexed:
    return Hexagon::STrib_indexed_nv_V4;

  case Hexagon::STrib_indexed_shl_V4:
    return Hexagon::STrib_indexed_shl_nv_V4;

  case Hexagon::STrib_shl_V4:
    return Hexagon::STrib_shl_nv_V4;

  case Hexagon::STrib_GP_V4:
    return Hexagon::STrib_GP_nv_V4;

  case Hexagon::STb_GP_V4:
    return Hexagon::STb_GP_nv_V4;

  case Hexagon::POST_STbri:
    return Hexagon::POST_STbri_nv_V4;

  case Hexagon::STrib_cPt:
    return Hexagon::STrib_cPt_nv_V4;

  case Hexagon::STrib_cdnPt_V4:
    return Hexagon::STrib_cdnPt_nv_V4;

  case Hexagon::STrib_cNotPt:
    return Hexagon::STrib_cNotPt_nv_V4;

  case Hexagon::STrib_cdnNotPt_V4:
    return Hexagon::STrib_cdnNotPt_nv_V4;

  case Hexagon::STrib_indexed_cPt:
    return Hexagon::STrib_indexed_cPt_nv_V4;

  case Hexagon::STrib_indexed_cdnPt_V4:
    return Hexagon::STrib_indexed_cdnPt_nv_V4;

  case Hexagon::STrib_indexed_cNotPt:
    return Hexagon::STrib_indexed_cNotPt_nv_V4;

  case Hexagon::STrib_indexed_cdnNotPt_V4:
    return Hexagon::STrib_indexed_cdnNotPt_nv_V4;

  case Hexagon::STrib_indexed_shl_cPt_V4:
    return Hexagon::STrib_indexed_shl_cPt_nv_V4;

  case Hexagon::STrib_indexed_shl_cdnPt_V4:
    return Hexagon::STrib_indexed_shl_cdnPt_nv_V4;

  case Hexagon::STrib_indexed_shl_cNotPt_V4:
    return Hexagon::STrib_indexed_shl_cNotPt_nv_V4;

  case Hexagon::STrib_indexed_shl_cdnNotPt_V4:
    return Hexagon::STrib_indexed_shl_cdnNotPt_nv_V4;

  case Hexagon::POST_STbri_cPt:
    return Hexagon::POST_STbri_cPt_nv_V4;

  case Hexagon::POST_STbri_cdnPt_V4:
    return Hexagon::POST_STbri_cdnPt_nv_V4;

  case Hexagon::POST_STbri_cNotPt:
    return Hexagon::POST_STbri_cNotPt_nv_V4;

  case Hexagon::POST_STbri_cdnNotPt_V4:
    return Hexagon::POST_STbri_cdnNotPt_nv_V4;

  case Hexagon::STb_GP_cPt_V4:
    return Hexagon::STb_GP_cPt_nv_V4;

  case Hexagon::STb_GP_cNotPt_V4:
    return Hexagon::STb_GP_cNotPt_nv_V4;

  case Hexagon::STb_GP_cdnPt_V4:
    return Hexagon::STb_GP_cdnPt_nv_V4;

  case Hexagon::STb_GP_cdnNotPt_V4:
    return Hexagon::STb_GP_cdnNotPt_nv_V4;

  case Hexagon::STrib_GP_cPt_V4:
    return Hexagon::STrib_GP_cPt_nv_V4;

  case Hexagon::STrib_GP_cNotPt_V4:
    return Hexagon::STrib_GP_cNotPt_nv_V4;

  case Hexagon::STrib_GP_cdnPt_V4:
    return Hexagon::STrib_GP_cdnPt_nv_V4;

  case Hexagon::STrib_GP_cdnNotPt_V4:
    return Hexagon::STrib_GP_cdnNotPt_nv_V4;

  // store new value halfword
  case Hexagon::STrih:
    return Hexagon::STrih_nv_V4;

  case Hexagon::STrih_indexed:
    return Hexagon::STrih_indexed_nv_V4;

  case Hexagon::STrih_indexed_shl_V4:
    return Hexagon::STrih_indexed_shl_nv_V4;

  case Hexagon::STrih_shl_V4:
    return Hexagon::STrih_shl_nv_V4;

  case Hexagon::STrih_GP_V4:
    return Hexagon::STrih_GP_nv_V4;

  case Hexagon::STh_GP_V4:
    return Hexagon::STh_GP_nv_V4;

  case Hexagon::POST_SThri:
    return Hexagon::POST_SThri_nv_V4;

  case Hexagon::STrih_cPt:
    return Hexagon::STrih_cPt_nv_V4;

  case Hexagon::STrih_cdnPt_V4:
    return Hexagon::STrih_cdnPt_nv_V4;

  case Hexagon::STrih_cNotPt:
    return Hexagon::STrih_cNotPt_nv_V4;

  case Hexagon::STrih_cdnNotPt_V4:
    return Hexagon::STrih_cdnNotPt_nv_V4;

  case Hexagon::STrih_indexed_cPt:
    return Hexagon::STrih_indexed_cPt_nv_V4;

  case Hexagon::STrih_indexed_cdnPt_V4:
    return Hexagon::STrih_indexed_cdnPt_nv_V4;

  case Hexagon::STrih_indexed_cNotPt:
    return Hexagon::STrih_indexed_cNotPt_nv_V4;

  case Hexagon::STrih_indexed_cdnNotPt_V4:
    return Hexagon::STrih_indexed_cdnNotPt_nv_V4;

  case Hexagon::STrih_indexed_shl_cPt_V4:
    return Hexagon::STrih_indexed_shl_cPt_nv_V4;

  case Hexagon::STrih_indexed_shl_cdnPt_V4:
    return Hexagon::STrih_indexed_shl_cdnPt_nv_V4;

  case Hexagon::STrih_indexed_shl_cNotPt_V4:
    return Hexagon::STrih_indexed_shl_cNotPt_nv_V4;

  case Hexagon::STrih_indexed_shl_cdnNotPt_V4:
    return Hexagon::STrih_indexed_shl_cdnNotPt_nv_V4;

  case Hexagon::POST_SThri_cPt:
    return Hexagon::POST_SThri_cPt_nv_V4;

  case Hexagon::POST_SThri_cdnPt_V4:
    return Hexagon::POST_SThri_cdnPt_nv_V4;

  case Hexagon::POST_SThri_cNotPt:
    return Hexagon::POST_SThri_cNotPt_nv_V4;

  case Hexagon::POST_SThri_cdnNotPt_V4:
    return Hexagon::POST_SThri_cdnNotPt_nv_V4;

  case Hexagon::STh_GP_cPt_V4:
    return Hexagon::STh_GP_cPt_nv_V4;

  case Hexagon::STh_GP_cNotPt_V4:
    return Hexagon::STh_GP_cNotPt_nv_V4;

  case Hexagon::STh_GP_cdnPt_V4:
    return Hexagon::STh_GP_cdnPt_nv_V4;

  case Hexagon::STh_GP_cdnNotPt_V4:
    return Hexagon::STh_GP_cdnNotPt_nv_V4;

  case Hexagon::STrih_GP_cPt_V4:
    return Hexagon::STrih_GP_cPt_nv_V4;

  case Hexagon::STrih_GP_cNotPt_V4:
    return Hexagon::STrih_GP_cNotPt_nv_V4;

  case Hexagon::STrih_GP_cdnPt_V4:
    return Hexagon::STrih_GP_cdnPt_nv_V4;

  case Hexagon::STrih_GP_cdnNotPt_V4:
    return Hexagon::STrih_GP_cdnNotPt_nv_V4;

  // store new value word
  case Hexagon::STriw:
    return Hexagon::STriw_nv_V4;

  case Hexagon::STriw_indexed:
    return Hexagon::STriw_indexed_nv_V4;

  case Hexagon::STriw_indexed_shl_V4:
    return Hexagon::STriw_indexed_shl_nv_V4;

  case Hexagon::STriw_shl_V4:
    return Hexagon::STriw_shl_nv_V4;

  case Hexagon::STriw_GP_V4:
    return Hexagon::STriw_GP_nv_V4;

  case Hexagon::STw_GP_V4:
    return Hexagon::STw_GP_nv_V4;

  case Hexagon::POST_STwri:
    return Hexagon::POST_STwri_nv_V4;

  case Hexagon::STriw_cPt:
    return Hexagon::STriw_cPt_nv_V4;

  case Hexagon::STriw_cdnPt_V4:
    return Hexagon::STriw_cdnPt_nv_V4;

  case Hexagon::STriw_cNotPt:
    return Hexagon::STriw_cNotPt_nv_V4;

  case Hexagon::STriw_cdnNotPt_V4:
    return Hexagon::STriw_cdnNotPt_nv_V4;

  case Hexagon::STriw_indexed_cPt:
    return Hexagon::STriw_indexed_cPt_nv_V4;

  case Hexagon::STriw_indexed_cdnPt_V4:
    return Hexagon::STriw_indexed_cdnPt_nv_V4;

  case Hexagon::STriw_indexed_cNotPt:
    return Hexagon::STriw_indexed_cNotPt_nv_V4;

  case Hexagon::STriw_indexed_cdnNotPt_V4:
    return Hexagon::STriw_indexed_cdnNotPt_nv_V4;

  case Hexagon::STriw_indexed_shl_cPt_V4:
    return Hexagon::STriw_indexed_shl_cPt_nv_V4;

  case Hexagon::STriw_indexed_shl_cdnPt_V4:
    return Hexagon::STriw_indexed_shl_cdnPt_nv_V4;

  case Hexagon::STriw_indexed_shl_cNotPt_V4:
    return Hexagon::STriw_indexed_shl_cNotPt_nv_V4;

  case Hexagon::STriw_indexed_shl_cdnNotPt_V4:
    return Hexagon::STriw_indexed_shl_cdnNotPt_nv_V4;

  case Hexagon::POST_STwri_cPt:
    return Hexagon::POST_STwri_cPt_nv_V4;

  case Hexagon::POST_STwri_cdnPt_V4:
    return Hexagon::POST_STwri_cdnPt_nv_V4;

  case Hexagon::POST_STwri_cNotPt:
    return Hexagon::POST_STwri_cNotPt_nv_V4;

  case Hexagon::POST_STwri_cdnNotPt_V4:
    return Hexagon::POST_STwri_cdnNotPt_nv_V4;

  case Hexagon::STw_GP_cPt_V4:
    return Hexagon::STw_GP_cPt_nv_V4;

  case Hexagon::STw_GP_cNotPt_V4:
    return Hexagon::STw_GP_cNotPt_nv_V4;

  case Hexagon::STw_GP_cdnPt_V4:
    return Hexagon::STw_GP_cdnPt_nv_V4;

  case Hexagon::STw_GP_cdnNotPt_V4:
    return Hexagon::STw_GP_cdnNotPt_nv_V4;

  case Hexagon::STriw_GP_cPt_V4:
    return Hexagon::STriw_GP_cPt_nv_V4;

  case Hexagon::STriw_GP_cNotPt_V4:
    return Hexagon::STriw_GP_cNotPt_nv_V4;

  case Hexagon::STriw_GP_cdnPt_V4:
    return Hexagon::STriw_GP_cdnPt_nv_V4;

  case Hexagon::STriw_GP_cdnNotPt_V4:
    return Hexagon::STriw_GP_cdnNotPt_nv_V4;
  }
}

// Return .new predicate version for an instruction
static int GetDotNewPredOp(const int opc) {
  switch (opc) {
  default: llvm_unreachable("Unknown .new type");
  // Conditional stores
  // Store byte conditionally
  case Hexagon::STrib_cPt :
    return Hexagon::STrib_cdnPt_V4;

  case Hexagon::STrib_cNotPt :
    return Hexagon::STrib_cdnNotPt_V4;

  case Hexagon::STrib_indexed_cPt :
    return Hexagon::STrib_indexed_cdnPt_V4;

  case Hexagon::STrib_indexed_cNotPt :
    return Hexagon::STrib_indexed_cdnNotPt_V4;

  case Hexagon::STrib_imm_cPt_V4 :
    return Hexagon::STrib_imm_cdnPt_V4;

  case Hexagon::STrib_imm_cNotPt_V4 :
    return Hexagon::STrib_imm_cdnNotPt_V4;

  case Hexagon::POST_STbri_cPt :
    return Hexagon::POST_STbri_cdnPt_V4;

  case Hexagon::POST_STbri_cNotPt :
    return Hexagon::POST_STbri_cdnNotPt_V4;

  case Hexagon::STrib_indexed_shl_cPt_V4 :
    return Hexagon::STrib_indexed_shl_cdnPt_V4;

  case Hexagon::STrib_indexed_shl_cNotPt_V4 :
    return Hexagon::STrib_indexed_shl_cdnNotPt_V4;

  case Hexagon::STb_GP_cPt_V4 :
    return Hexagon::STb_GP_cdnPt_V4;

  case Hexagon::STb_GP_cNotPt_V4 :
    return Hexagon::STb_GP_cdnNotPt_V4;

  case Hexagon::STrib_GP_cPt_V4 :
    return Hexagon::STrib_GP_cdnPt_V4;

  case Hexagon::STrib_GP_cNotPt_V4 :
    return Hexagon::STrib_GP_cdnNotPt_V4;

  // Store doubleword conditionally
  case Hexagon::STrid_cPt :
    return Hexagon::STrid_cdnPt_V4;

  case Hexagon::STrid_cNotPt :
    return Hexagon::STrid_cdnNotPt_V4;

  case Hexagon::STrid_indexed_cPt :
    return Hexagon::STrid_indexed_cdnPt_V4;

  case Hexagon::STrid_indexed_cNotPt :
    return Hexagon::STrid_indexed_cdnNotPt_V4;

  case Hexagon::STrid_indexed_shl_cPt_V4 :
    return Hexagon::STrid_indexed_shl_cdnPt_V4;

  case Hexagon::STrid_indexed_shl_cNotPt_V4 :
    return Hexagon::STrid_indexed_shl_cdnNotPt_V4;

  case Hexagon::POST_STdri_cPt :
    return Hexagon::POST_STdri_cdnPt_V4;

  case Hexagon::POST_STdri_cNotPt :
    return Hexagon::POST_STdri_cdnNotPt_V4;

  case Hexagon::STd_GP_cPt_V4 :
    return Hexagon::STd_GP_cdnPt_V4;

  case Hexagon::STd_GP_cNotPt_V4 :
    return Hexagon::STd_GP_cdnNotPt_V4;

  case Hexagon::STrid_GP_cPt_V4 :
    return Hexagon::STrid_GP_cdnPt_V4;

  case Hexagon::STrid_GP_cNotPt_V4 :
    return Hexagon::STrid_GP_cdnNotPt_V4;

  // Store halfword conditionally
  case Hexagon::STrih_cPt :
    return Hexagon::STrih_cdnPt_V4;

  case Hexagon::STrih_cNotPt :
    return Hexagon::STrih_cdnNotPt_V4;

  case Hexagon::STrih_indexed_cPt :
    return Hexagon::STrih_indexed_cdnPt_V4;

  case Hexagon::STrih_indexed_cNotPt :
    return Hexagon::STrih_indexed_cdnNotPt_V4;

  case Hexagon::STrih_imm_cPt_V4 :
    return Hexagon::STrih_imm_cdnPt_V4;

  case Hexagon::STrih_imm_cNotPt_V4 :
    return Hexagon::STrih_imm_cdnNotPt_V4;

  case Hexagon::STrih_indexed_shl_cPt_V4 :
    return Hexagon::STrih_indexed_shl_cdnPt_V4;

  case Hexagon::STrih_indexed_shl_cNotPt_V4 :
    return Hexagon::STrih_indexed_shl_cdnNotPt_V4;

  case Hexagon::POST_SThri_cPt :
    return Hexagon::POST_SThri_cdnPt_V4;

  case Hexagon::POST_SThri_cNotPt :
    return Hexagon::POST_SThri_cdnNotPt_V4;

  case Hexagon::STh_GP_cPt_V4 :
    return Hexagon::STh_GP_cdnPt_V4;

  case Hexagon::STh_GP_cNotPt_V4 :
    return Hexagon::STh_GP_cdnNotPt_V4;

  case Hexagon::STrih_GP_cPt_V4 :
    return Hexagon::STrih_GP_cdnPt_V4;

  case Hexagon::STrih_GP_cNotPt_V4 :
    return Hexagon::STrih_GP_cdnNotPt_V4;

  // Store word conditionally
  case Hexagon::STriw_cPt :
    return Hexagon::STriw_cdnPt_V4;

  case Hexagon::STriw_cNotPt :
    return Hexagon::STriw_cdnNotPt_V4;

  case Hexagon::STriw_indexed_cPt :
    return Hexagon::STriw_indexed_cdnPt_V4;

  case Hexagon::STriw_indexed_cNotPt :
    return Hexagon::STriw_indexed_cdnNotPt_V4;

  case Hexagon::STriw_imm_cPt_V4 :
    return Hexagon::STriw_imm_cdnPt_V4;

  case Hexagon::STriw_imm_cNotPt_V4 :
    return Hexagon::STriw_imm_cdnNotPt_V4;

  case Hexagon::STriw_indexed_shl_cPt_V4 :
    return Hexagon::STriw_indexed_shl_cdnPt_V4;

  case Hexagon::STriw_indexed_shl_cNotPt_V4 :
    return Hexagon::STriw_indexed_shl_cdnNotPt_V4;

  case Hexagon::POST_STwri_cPt :
    return Hexagon::POST_STwri_cdnPt_V4;

  case Hexagon::POST_STwri_cNotPt :
    return Hexagon::POST_STwri_cdnNotPt_V4;

  case Hexagon::STw_GP_cPt_V4 :
    return Hexagon::STw_GP_cdnPt_V4;

  case Hexagon::STw_GP_cNotPt_V4 :
    return Hexagon::STw_GP_cdnNotPt_V4;

  case Hexagon::STriw_GP_cPt_V4 :
    return Hexagon::STriw_GP_cdnPt_V4;

  case Hexagon::STriw_GP_cNotPt_V4 :
    return Hexagon::STriw_GP_cdnNotPt_V4;

  // Condtional Jumps
  case Hexagon::JMP_c:
    return Hexagon::JMP_cdnPt;

  case Hexagon::JMP_cNot:
    return Hexagon::JMP_cdnNotPt;

  case Hexagon::JMPR_cPt:
    return Hexagon::JMPR_cdnPt_V3;

  case Hexagon::JMPR_cNotPt:
    return Hexagon::JMPR_cdnNotPt_V3;

  // Conditional Transfers
  case Hexagon::TFR_cPt:
    return Hexagon::TFR_cdnPt;

  case Hexagon::TFR_cNotPt:
    return Hexagon::TFR_cdnNotPt;

  case Hexagon::TFRI_cPt:
    return Hexagon::TFRI_cdnPt;

  case Hexagon::TFRI_cNotPt:
    return Hexagon::TFRI_cdnNotPt;

  // Load double word
  case Hexagon::LDrid_cPt :
    return Hexagon::LDrid_cdnPt;

  case Hexagon::LDrid_cNotPt :
    return Hexagon::LDrid_cdnNotPt;

  case Hexagon::LDrid_indexed_cPt :
    return Hexagon::LDrid_indexed_cdnPt;

  case Hexagon::LDrid_indexed_cNotPt :
    return Hexagon::LDrid_indexed_cdnNotPt;

  case Hexagon::POST_LDrid_cPt :
    return Hexagon::POST_LDrid_cdnPt_V4;

  case Hexagon::POST_LDrid_cNotPt :
    return Hexagon::POST_LDrid_cdnNotPt_V4;

  // Load word
  case Hexagon::LDriw_cPt :
    return Hexagon::LDriw_cdnPt;

  case Hexagon::LDriw_cNotPt :
    return Hexagon::LDriw_cdnNotPt;

  case Hexagon::LDriw_indexed_cPt :
    return Hexagon::LDriw_indexed_cdnPt;

  case Hexagon::LDriw_indexed_cNotPt :
    return Hexagon::LDriw_indexed_cdnNotPt;

  case Hexagon::POST_LDriw_cPt :
    return Hexagon::POST_LDriw_cdnPt_V4;

  case Hexagon::POST_LDriw_cNotPt :
    return Hexagon::POST_LDriw_cdnNotPt_V4;

  // Load halfword
  case Hexagon::LDrih_cPt :
    return Hexagon::LDrih_cdnPt;

  case Hexagon::LDrih_cNotPt :
    return Hexagon::LDrih_cdnNotPt;

  case Hexagon::LDrih_indexed_cPt :
    return Hexagon::LDrih_indexed_cdnPt;

  case Hexagon::LDrih_indexed_cNotPt :
    return Hexagon::LDrih_indexed_cdnNotPt;

  case Hexagon::POST_LDrih_cPt :
    return Hexagon::POST_LDrih_cdnPt_V4;

  case Hexagon::POST_LDrih_cNotPt :
    return Hexagon::POST_LDrih_cdnNotPt_V4;

  // Load byte
  case Hexagon::LDrib_cPt :
    return Hexagon::LDrib_cdnPt;

  case Hexagon::LDrib_cNotPt :
    return Hexagon::LDrib_cdnNotPt;

  case Hexagon::LDrib_indexed_cPt :
    return Hexagon::LDrib_indexed_cdnPt;

  case Hexagon::LDrib_indexed_cNotPt :
    return Hexagon::LDrib_indexed_cdnNotPt;

  case Hexagon::POST_LDrib_cPt :
    return Hexagon::POST_LDrib_cdnPt_V4;

  case Hexagon::POST_LDrib_cNotPt :
    return Hexagon::POST_LDrib_cdnNotPt_V4;

  // Load unsigned halfword
  case Hexagon::LDriuh_cPt :
    return Hexagon::LDriuh_cdnPt;

  case Hexagon::LDriuh_cNotPt :
    return Hexagon::LDriuh_cdnNotPt;

  case Hexagon::LDriuh_indexed_cPt :
    return Hexagon::LDriuh_indexed_cdnPt;

  case Hexagon::LDriuh_indexed_cNotPt :
    return Hexagon::LDriuh_indexed_cdnNotPt;

  case Hexagon::POST_LDriuh_cPt :
    return Hexagon::POST_LDriuh_cdnPt_V4;

  case Hexagon::POST_LDriuh_cNotPt :
    return Hexagon::POST_LDriuh_cdnNotPt_V4;

  // Load unsigned byte
  case Hexagon::LDriub_cPt :
    return Hexagon::LDriub_cdnPt;

  case Hexagon::LDriub_cNotPt :
    return Hexagon::LDriub_cdnNotPt;

  case Hexagon::LDriub_indexed_cPt :
    return Hexagon::LDriub_indexed_cdnPt;

  case Hexagon::LDriub_indexed_cNotPt :
    return Hexagon::LDriub_indexed_cdnNotPt;

  case Hexagon::POST_LDriub_cPt :
    return Hexagon::POST_LDriub_cdnPt_V4;

  case Hexagon::POST_LDriub_cNotPt :
    return Hexagon::POST_LDriub_cdnNotPt_V4;

  // V4 indexed+scaled load

  case Hexagon::LDrid_indexed_cPt_V4 :
    return Hexagon::LDrid_indexed_cdnPt_V4;

  case Hexagon::LDrid_indexed_cNotPt_V4 :
    return Hexagon::LDrid_indexed_cdnNotPt_V4;

  case Hexagon::LDrid_indexed_shl_cPt_V4 :
    return Hexagon::LDrid_indexed_shl_cdnPt_V4;

  case Hexagon::LDrid_indexed_shl_cNotPt_V4 :
    return Hexagon::LDrid_indexed_shl_cdnNotPt_V4;

  case Hexagon::LDrib_indexed_cPt_V4 :
    return Hexagon::LDrib_indexed_cdnPt_V4;

  case Hexagon::LDrib_indexed_cNotPt_V4 :
    return Hexagon::LDrib_indexed_cdnNotPt_V4;

  case Hexagon::LDrib_indexed_shl_cPt_V4 :
    return Hexagon::LDrib_indexed_shl_cdnPt_V4;

  case Hexagon::LDrib_indexed_shl_cNotPt_V4 :
    return Hexagon::LDrib_indexed_shl_cdnNotPt_V4;

  case Hexagon::LDriub_indexed_cPt_V4 :
    return Hexagon::LDriub_indexed_cdnPt_V4;

  case Hexagon::LDriub_indexed_cNotPt_V4 :
    return Hexagon::LDriub_indexed_cdnNotPt_V4;

  case Hexagon::LDriub_indexed_shl_cPt_V4 :
    return Hexagon::LDriub_indexed_shl_cdnPt_V4;

  case Hexagon::LDriub_indexed_shl_cNotPt_V4 :
    return Hexagon::LDriub_indexed_shl_cdnNotPt_V4;

  case Hexagon::LDrih_indexed_cPt_V4 :
    return Hexagon::LDrih_indexed_cdnPt_V4;

  case Hexagon::LDrih_indexed_cNotPt_V4 :
    return Hexagon::LDrih_indexed_cdnNotPt_V4;

  case Hexagon::LDrih_indexed_shl_cPt_V4 :
    return Hexagon::LDrih_indexed_shl_cdnPt_V4;

  case Hexagon::LDrih_indexed_shl_cNotPt_V4 :
    return Hexagon::LDrih_indexed_shl_cdnNotPt_V4;

  case Hexagon::LDriuh_indexed_cPt_V4 :
    return Hexagon::LDriuh_indexed_cdnPt_V4;

  case Hexagon::LDriuh_indexed_cNotPt_V4 :
    return Hexagon::LDriuh_indexed_cdnNotPt_V4;

  case Hexagon::LDriuh_indexed_shl_cPt_V4 :
    return Hexagon::LDriuh_indexed_shl_cdnPt_V4;

  case Hexagon::LDriuh_indexed_shl_cNotPt_V4 :
    return Hexagon::LDriuh_indexed_shl_cdnNotPt_V4;

  case Hexagon::LDriw_indexed_cPt_V4 :
    return Hexagon::LDriw_indexed_cdnPt_V4;

  case Hexagon::LDriw_indexed_cNotPt_V4 :
    return Hexagon::LDriw_indexed_cdnNotPt_V4;

  case Hexagon::LDriw_indexed_shl_cPt_V4 :
    return Hexagon::LDriw_indexed_shl_cdnPt_V4;

  case Hexagon::LDriw_indexed_shl_cNotPt_V4 :
    return Hexagon::LDriw_indexed_shl_cdnNotPt_V4;

  // V4 global address load

  case Hexagon::LDd_GP_cPt_V4:
    return Hexagon::LDd_GP_cdnPt_V4;

  case Hexagon::LDd_GP_cNotPt_V4:
    return Hexagon::LDd_GP_cdnNotPt_V4;

  case Hexagon::LDb_GP_cPt_V4:
    return Hexagon::LDb_GP_cdnPt_V4;

  case Hexagon::LDb_GP_cNotPt_V4:
    return Hexagon::LDb_GP_cdnNotPt_V4;

  case Hexagon::LDub_GP_cPt_V4:
    return Hexagon::LDub_GP_cdnPt_V4;

  case Hexagon::LDub_GP_cNotPt_V4:
    return Hexagon::LDub_GP_cdnNotPt_V4;

  case Hexagon::LDh_GP_cPt_V4:
    return Hexagon::LDh_GP_cdnPt_V4;

  case Hexagon::LDh_GP_cNotPt_V4:
    return Hexagon::LDh_GP_cdnNotPt_V4;

  case Hexagon::LDuh_GP_cPt_V4:
    return Hexagon::LDuh_GP_cdnPt_V4;

  case Hexagon::LDuh_GP_cNotPt_V4:
    return Hexagon::LDuh_GP_cdnNotPt_V4;

  case Hexagon::LDw_GP_cPt_V4:
    return Hexagon::LDw_GP_cdnPt_V4;

  case Hexagon::LDw_GP_cNotPt_V4:
    return Hexagon::LDw_GP_cdnNotPt_V4;

  case Hexagon::LDrid_GP_cPt_V4:
    return Hexagon::LDrid_GP_cdnPt_V4;

  case Hexagon::LDrid_GP_cNotPt_V4:
    return Hexagon::LDrid_GP_cdnNotPt_V4;

  case Hexagon::LDrib_GP_cPt_V4:
    return Hexagon::LDrib_GP_cdnPt_V4;

  case Hexagon::LDrib_GP_cNotPt_V4:
    return Hexagon::LDrib_GP_cdnNotPt_V4;

  case Hexagon::LDriub_GP_cPt_V4:
    return Hexagon::LDriub_GP_cdnPt_V4;

  case Hexagon::LDriub_GP_cNotPt_V4:
    return Hexagon::LDriub_GP_cdnNotPt_V4;

  case Hexagon::LDrih_GP_cPt_V4:
    return Hexagon::LDrih_GP_cdnPt_V4;

  case Hexagon::LDrih_GP_cNotPt_V4:
    return Hexagon::LDrih_GP_cdnNotPt_V4;

  case Hexagon::LDriuh_GP_cPt_V4:
    return Hexagon::LDriuh_GP_cdnPt_V4;

  case Hexagon::LDriuh_GP_cNotPt_V4:
    return Hexagon::LDriuh_GP_cdnNotPt_V4;

  case Hexagon::LDriw_GP_cPt_V4:
    return Hexagon::LDriw_GP_cdnPt_V4;

  case Hexagon::LDriw_GP_cNotPt_V4:
    return Hexagon::LDriw_GP_cdnNotPt_V4;

  // Conditional store new-value byte
  case Hexagon::STrib_cPt_nv_V4 :
    return Hexagon::STrib_cdnPt_nv_V4;
  case Hexagon::STrib_cNotPt_nv_V4 :
    return Hexagon::STrib_cdnNotPt_nv_V4;

  case Hexagon::STrib_indexed_cPt_nv_V4 :
    return Hexagon::STrib_indexed_cdnPt_nv_V4;
  case Hexagon::STrib_indexed_cNotPt_nv_V4 :
    return Hexagon::STrib_indexed_cdnNotPt_nv_V4;

  case Hexagon::STrib_indexed_shl_cPt_nv_V4 :
    return Hexagon::STrib_indexed_shl_cdnPt_nv_V4;
  case Hexagon::STrib_indexed_shl_cNotPt_nv_V4 :
    return Hexagon::STrib_indexed_shl_cdnNotPt_nv_V4;

  case Hexagon::POST_STbri_cPt_nv_V4 :
    return Hexagon::POST_STbri_cdnPt_nv_V4;
  case Hexagon::POST_STbri_cNotPt_nv_V4 :
    return Hexagon::POST_STbri_cdnNotPt_nv_V4;

  case Hexagon::STb_GP_cPt_nv_V4 :
    return Hexagon::STb_GP_cdnPt_nv_V4;

  case Hexagon::STb_GP_cNotPt_nv_V4 :
    return Hexagon::STb_GP_cdnNotPt_nv_V4;

  case Hexagon::STrib_GP_cPt_nv_V4 :
    return Hexagon::STrib_GP_cdnPt_nv_V4;

  case Hexagon::STrib_GP_cNotPt_nv_V4 :
    return Hexagon::STrib_GP_cdnNotPt_nv_V4;

  // Conditional store new-value halfword
  case Hexagon::STrih_cPt_nv_V4 :
    return Hexagon::STrih_cdnPt_nv_V4;
  case Hexagon::STrih_cNotPt_nv_V4 :
    return Hexagon::STrih_cdnNotPt_nv_V4;

  case Hexagon::STrih_indexed_cPt_nv_V4 :
    return Hexagon::STrih_indexed_cdnPt_nv_V4;
  case Hexagon::STrih_indexed_cNotPt_nv_V4 :
    return Hexagon::STrih_indexed_cdnNotPt_nv_V4;

  case Hexagon::STrih_indexed_shl_cPt_nv_V4 :
    return Hexagon::STrih_indexed_shl_cdnPt_nv_V4;
  case Hexagon::STrih_indexed_shl_cNotPt_nv_V4 :
    return Hexagon::STrih_indexed_shl_cdnNotPt_nv_V4;

  case Hexagon::POST_SThri_cPt_nv_V4 :
    return Hexagon::POST_SThri_cdnPt_nv_V4;
  case Hexagon::POST_SThri_cNotPt_nv_V4 :
    return Hexagon::POST_SThri_cdnNotPt_nv_V4;

  case Hexagon::STh_GP_cPt_nv_V4 :
    return Hexagon::STh_GP_cdnPt_nv_V4;

  case Hexagon::STh_GP_cNotPt_nv_V4 :
    return Hexagon::STh_GP_cdnNotPt_nv_V4;

  case Hexagon::STrih_GP_cPt_nv_V4 :
    return Hexagon::STrih_GP_cdnPt_nv_V4;

  case Hexagon::STrih_GP_cNotPt_nv_V4 :
    return Hexagon::STrih_GP_cdnNotPt_nv_V4;

  // Conditional store new-value word
  case Hexagon::STriw_cPt_nv_V4 :
    return  Hexagon::STriw_cdnPt_nv_V4;
  case Hexagon::STriw_cNotPt_nv_V4 :
    return Hexagon::STriw_cdnNotPt_nv_V4;

  case Hexagon::STriw_indexed_cPt_nv_V4 :
    return Hexagon::STriw_indexed_cdnPt_nv_V4;
  case Hexagon::STriw_indexed_cNotPt_nv_V4 :
    return Hexagon::STriw_indexed_cdnNotPt_nv_V4;

  case Hexagon::STriw_indexed_shl_cPt_nv_V4 :
    return Hexagon::STriw_indexed_shl_cdnPt_nv_V4;
  case Hexagon::STriw_indexed_shl_cNotPt_nv_V4 :
    return Hexagon::STriw_indexed_shl_cdnNotPt_nv_V4;

  case Hexagon::POST_STwri_cPt_nv_V4 :
    return Hexagon::POST_STwri_cdnPt_nv_V4;
  case Hexagon::POST_STwri_cNotPt_nv_V4:
    return Hexagon::POST_STwri_cdnNotPt_nv_V4;

  case Hexagon::STw_GP_cPt_nv_V4 :
    return Hexagon::STw_GP_cdnPt_nv_V4;

  case Hexagon::STw_GP_cNotPt_nv_V4 :
    return Hexagon::STw_GP_cdnNotPt_nv_V4;

  case Hexagon::STriw_GP_cPt_nv_V4 :
    return Hexagon::STriw_GP_cdnPt_nv_V4;

  case Hexagon::STriw_GP_cNotPt_nv_V4 :
    return Hexagon::STriw_GP_cdnNotPt_nv_V4;

  // Conditional add
  case Hexagon::ADD_ri_cPt :
    return Hexagon::ADD_ri_cdnPt;
  case Hexagon::ADD_ri_cNotPt :
    return Hexagon::ADD_ri_cdnNotPt;

  case Hexagon::ADD_rr_cPt :
    return Hexagon::ADD_rr_cdnPt;
  case Hexagon::ADD_rr_cNotPt :
    return Hexagon::ADD_rr_cdnNotPt;

  // Conditional logical Operations
  case Hexagon::XOR_rr_cPt :
    return Hexagon::XOR_rr_cdnPt;
  case Hexagon::XOR_rr_cNotPt :
    return Hexagon::XOR_rr_cdnNotPt;

  case Hexagon::AND_rr_cPt :
    return Hexagon::AND_rr_cdnPt;
  case Hexagon::AND_rr_cNotPt :
    return Hexagon::AND_rr_cdnNotPt;

  case Hexagon::OR_rr_cPt :
    return Hexagon::OR_rr_cdnPt;
  case Hexagon::OR_rr_cNotPt :
    return Hexagon::OR_rr_cdnNotPt;

  // Conditional Subtract
  case Hexagon::SUB_rr_cPt :
    return Hexagon::SUB_rr_cdnPt;
  case Hexagon::SUB_rr_cNotPt :
    return Hexagon::SUB_rr_cdnNotPt;

  // Conditional combine
  case Hexagon::COMBINE_rr_cPt :
    return Hexagon::COMBINE_rr_cdnPt;
  case Hexagon::COMBINE_rr_cNotPt :
    return Hexagon::COMBINE_rr_cdnNotPt;

  case Hexagon::ASLH_cPt_V4 :
    return Hexagon::ASLH_cdnPt_V4;
  case Hexagon::ASLH_cNotPt_V4 :
    return Hexagon::ASLH_cdnNotPt_V4;

  case Hexagon::ASRH_cPt_V4 :
    return Hexagon::ASRH_cdnPt_V4;
  case Hexagon::ASRH_cNotPt_V4 :
    return Hexagon::ASRH_cdnNotPt_V4;

  case Hexagon::SXTB_cPt_V4 :
    return Hexagon::SXTB_cdnPt_V4;
  case Hexagon::SXTB_cNotPt_V4 :
    return Hexagon::SXTB_cdnNotPt_V4;

  case Hexagon::SXTH_cPt_V4 :
    return Hexagon::SXTH_cdnPt_V4;
  case Hexagon::SXTH_cNotPt_V4 :
    return Hexagon::SXTH_cdnNotPt_V4;

  case Hexagon::ZXTB_cPt_V4 :
    return Hexagon::ZXTB_cdnPt_V4;
  case Hexagon::ZXTB_cNotPt_V4 :
    return Hexagon::ZXTB_cdnNotPt_V4;

  case Hexagon::ZXTH_cPt_V4 :
    return Hexagon::ZXTH_cdnPt_V4;
  case Hexagon::ZXTH_cNotPt_V4 :
    return Hexagon::ZXTH_cdnNotPt_V4;
  }
}

// Returns true if an instruction can be promoted to .new predicate
// or new-value store.
bool HexagonPacketizerList::isNewifiable(MachineInstr* MI) {
  if ( isCondInst(MI) || IsNewifyStore(MI))
    return true;
  else
    return false;
}

bool HexagonPacketizerList::isCondInst (MachineInstr* MI) {
  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  const MCInstrDesc& TID = MI->getDesc();
                                    // bug 5670: until that is fixed,
                                    // this portion is disabled.
  if (   TID.isConditionalBranch()  // && !IsRegisterJump(MI)) ||
      || QII->isConditionalTransfer(MI)
      || QII->isConditionalALU32(MI)
      || QII->isConditionalLoad(MI)
      || QII->isConditionalStore(MI)) {
    return true;
  }
  return false;
}


// Promote an instructiont to its .new form.
// At this time, we have already made a call to CanPromoteToDotNew
// and made sure that it can *indeed* be promoted.
bool HexagonPacketizerList::PromoteToDotNew(MachineInstr* MI,
                        SDep::Kind DepType, MachineBasicBlock::iterator &MII,
                        const TargetRegisterClass* RC) {

  assert (DepType == SDep::Data);
  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;

  int NewOpcode;
  if (RC == &Hexagon::PredRegsRegClass)
    NewOpcode = GetDotNewPredOp(MI->getOpcode());
  else
    NewOpcode = GetDotNewOp(MI->getOpcode());
  MI->setDesc(QII->get(NewOpcode));

  return true;
}

// Returns the most basic instruction for the .new predicated instructions and
// new-value stores.
// For example, all of the following instructions will be converted back to the
// same instruction:
// 1) if (p0.new) memw(R0+#0) = R1.new  --->
// 2) if (p0) memw(R0+#0)= R1.new      -------> if (p0) memw(R0+#0) = R1
// 3) if (p0.new) memw(R0+#0) = R1      --->
//
// To understand the translation of instruction 1 to its original form, consider
// a packet with 3 instructions.
// { p0 = cmp.eq(R0,R1)
//   if (p0.new) R2 = add(R3, R4)
//   R5 = add (R3, R1)
//   }
// if (p0) memw(R5+#0) = R2 <--- trying to include it in the previous packet
//
// This instruction can be part of the previous packet only if both p0 and R2
// are promoted to .new values. This promotion happens in steps, first
// predicate register is promoted to .new and in the next iteration R2 is
// promoted. Therefore, in case of dependence check failure (due to R5) during
// next iteration, it should be converted back to its most basic form.

static int GetDotOldOp(const int opc) {
  switch (opc) {
  default: llvm_unreachable("Unknown .old type");
  case Hexagon::TFR_cdnPt:
    return Hexagon::TFR_cPt;

  case Hexagon::TFR_cdnNotPt:
    return Hexagon::TFR_cNotPt;

  case Hexagon::TFRI_cdnPt:
    return Hexagon::TFRI_cPt;

  case Hexagon::TFRI_cdnNotPt:
    return Hexagon::TFRI_cNotPt;

  case Hexagon::JMP_cdnPt:
    return Hexagon::JMP_c;

  case Hexagon::JMP_cdnNotPt:
    return Hexagon::JMP_cNot;

  case Hexagon::JMPR_cdnPt_V3:
    return Hexagon::JMPR_cPt;

  case Hexagon::JMPR_cdnNotPt_V3:
    return Hexagon::JMPR_cNotPt;

  // Load double word

  case Hexagon::LDrid_cdnPt :
    return Hexagon::LDrid_cPt;

  case Hexagon::LDrid_cdnNotPt :
    return Hexagon::LDrid_cNotPt;

  case Hexagon::LDrid_indexed_cdnPt :
    return Hexagon::LDrid_indexed_cPt;

  case Hexagon::LDrid_indexed_cdnNotPt :
    return Hexagon::LDrid_indexed_cNotPt;

  case Hexagon::POST_LDrid_cdnPt_V4 :
    return Hexagon::POST_LDrid_cPt;

  case Hexagon::POST_LDrid_cdnNotPt_V4 :
    return Hexagon::POST_LDrid_cNotPt;

  // Load word

  case Hexagon::LDriw_cdnPt :
    return Hexagon::LDriw_cPt;

  case Hexagon::LDriw_cdnNotPt :
    return Hexagon::LDriw_cNotPt;

  case Hexagon::LDriw_indexed_cdnPt :
    return Hexagon::LDriw_indexed_cPt;

  case Hexagon::LDriw_indexed_cdnNotPt :
    return Hexagon::LDriw_indexed_cNotPt;

  case Hexagon::POST_LDriw_cdnPt_V4 :
    return Hexagon::POST_LDriw_cPt;

  case Hexagon::POST_LDriw_cdnNotPt_V4 :
    return Hexagon::POST_LDriw_cNotPt;

  // Load half

  case Hexagon::LDrih_cdnPt :
    return Hexagon::LDrih_cPt;

  case Hexagon::LDrih_cdnNotPt :
    return Hexagon::LDrih_cNotPt;

  case Hexagon::LDrih_indexed_cdnPt :
    return Hexagon::LDrih_indexed_cPt;

  case Hexagon::LDrih_indexed_cdnNotPt :
    return Hexagon::LDrih_indexed_cNotPt;

  case Hexagon::POST_LDrih_cdnPt_V4 :
    return Hexagon::POST_LDrih_cPt;

  case Hexagon::POST_LDrih_cdnNotPt_V4 :
    return Hexagon::POST_LDrih_cNotPt;

  // Load byte

  case Hexagon::LDrib_cdnPt :
    return Hexagon::LDrib_cPt;

  case Hexagon::LDrib_cdnNotPt :
    return Hexagon::LDrib_cNotPt;

  case Hexagon::LDrib_indexed_cdnPt :
    return Hexagon::LDrib_indexed_cPt;

  case Hexagon::LDrib_indexed_cdnNotPt :
    return Hexagon::LDrib_indexed_cNotPt;

  case Hexagon::POST_LDrib_cdnPt_V4 :
    return Hexagon::POST_LDrib_cPt;

  case Hexagon::POST_LDrib_cdnNotPt_V4 :
    return Hexagon::POST_LDrib_cNotPt;

  // Load unsigned half

  case Hexagon::LDriuh_cdnPt :
    return Hexagon::LDriuh_cPt;

  case Hexagon::LDriuh_cdnNotPt :
    return Hexagon::LDriuh_cNotPt;

  case Hexagon::LDriuh_indexed_cdnPt :
    return Hexagon::LDriuh_indexed_cPt;

  case Hexagon::LDriuh_indexed_cdnNotPt :
    return Hexagon::LDriuh_indexed_cNotPt;

  case Hexagon::POST_LDriuh_cdnPt_V4 :
    return Hexagon::POST_LDriuh_cPt;

  case Hexagon::POST_LDriuh_cdnNotPt_V4 :
    return Hexagon::POST_LDriuh_cNotPt;

  // Load unsigned byte
  case Hexagon::LDriub_cdnPt :
    return Hexagon::LDriub_cPt;

  case Hexagon::LDriub_cdnNotPt :
    return Hexagon::LDriub_cNotPt;

  case Hexagon::LDriub_indexed_cdnPt :
    return Hexagon::LDriub_indexed_cPt;

  case Hexagon::LDriub_indexed_cdnNotPt :
    return Hexagon::LDriub_indexed_cNotPt;

  case Hexagon::POST_LDriub_cdnPt_V4 :
    return Hexagon::POST_LDriub_cPt;

  case Hexagon::POST_LDriub_cdnNotPt_V4 :
    return Hexagon::POST_LDriub_cNotPt;

  // V4 indexed+scaled Load

  case Hexagon::LDrid_indexed_cdnPt_V4 :
    return Hexagon::LDrid_indexed_cPt_V4;

  case Hexagon::LDrid_indexed_cdnNotPt_V4 :
    return Hexagon::LDrid_indexed_cNotPt_V4;

  case Hexagon::LDrid_indexed_shl_cdnPt_V4 :
    return Hexagon::LDrid_indexed_shl_cPt_V4;

  case Hexagon::LDrid_indexed_shl_cdnNotPt_V4 :
    return Hexagon::LDrid_indexed_shl_cNotPt_V4;

  case Hexagon::LDrib_indexed_cdnPt_V4 :
    return Hexagon::LDrib_indexed_cPt_V4;

  case Hexagon::LDrib_indexed_cdnNotPt_V4 :
    return Hexagon::LDrib_indexed_cNotPt_V4;

  case Hexagon::LDrib_indexed_shl_cdnPt_V4 :
    return Hexagon::LDrib_indexed_shl_cPt_V4;

  case Hexagon::LDrib_indexed_shl_cdnNotPt_V4 :
    return Hexagon::LDrib_indexed_shl_cNotPt_V4;

  case Hexagon::LDriub_indexed_cdnPt_V4 :
    return Hexagon::LDriub_indexed_cPt_V4;

  case Hexagon::LDriub_indexed_cdnNotPt_V4 :
    return Hexagon::LDriub_indexed_cNotPt_V4;

  case Hexagon::LDriub_indexed_shl_cdnPt_V4 :
    return Hexagon::LDriub_indexed_shl_cPt_V4;

  case Hexagon::LDriub_indexed_shl_cdnNotPt_V4 :
    return Hexagon::LDriub_indexed_shl_cNotPt_V4;

  case Hexagon::LDrih_indexed_cdnPt_V4 :
    return Hexagon::LDrih_indexed_cPt_V4;

  case Hexagon::LDrih_indexed_cdnNotPt_V4 :
    return Hexagon::LDrih_indexed_cNotPt_V4;

  case Hexagon::LDrih_indexed_shl_cdnPt_V4 :
    return Hexagon::LDrih_indexed_shl_cPt_V4;

  case Hexagon::LDrih_indexed_shl_cdnNotPt_V4 :
    return Hexagon::LDrih_indexed_shl_cNotPt_V4;

  case Hexagon::LDriuh_indexed_cdnPt_V4 :
    return Hexagon::LDriuh_indexed_cPt_V4;

  case Hexagon::LDriuh_indexed_cdnNotPt_V4 :
    return Hexagon::LDriuh_indexed_cNotPt_V4;

  case Hexagon::LDriuh_indexed_shl_cdnPt_V4 :
    return Hexagon::LDriuh_indexed_shl_cPt_V4;

  case Hexagon::LDriuh_indexed_shl_cdnNotPt_V4 :
    return Hexagon::LDriuh_indexed_shl_cNotPt_V4;

  case Hexagon::LDriw_indexed_cdnPt_V4 :
    return Hexagon::LDriw_indexed_cPt_V4;

  case Hexagon::LDriw_indexed_cdnNotPt_V4 :
    return Hexagon::LDriw_indexed_cNotPt_V4;

  case Hexagon::LDriw_indexed_shl_cdnPt_V4 :
    return Hexagon::LDriw_indexed_shl_cPt_V4;

  case Hexagon::LDriw_indexed_shl_cdnNotPt_V4 :
    return Hexagon::LDriw_indexed_shl_cNotPt_V4;

  // V4 global address load

  case Hexagon::LDd_GP_cdnPt_V4:
    return Hexagon::LDd_GP_cPt_V4;

  case Hexagon::LDd_GP_cdnNotPt_V4:
    return Hexagon::LDd_GP_cNotPt_V4;

  case Hexagon::LDb_GP_cdnPt_V4:
    return Hexagon::LDb_GP_cPt_V4;

  case Hexagon::LDb_GP_cdnNotPt_V4:
    return Hexagon::LDb_GP_cNotPt_V4;

  case Hexagon::LDub_GP_cdnPt_V4:
    return Hexagon::LDub_GP_cPt_V4;

  case Hexagon::LDub_GP_cdnNotPt_V4:
    return Hexagon::LDub_GP_cNotPt_V4;

  case Hexagon::LDh_GP_cdnPt_V4:
    return Hexagon::LDh_GP_cPt_V4;

  case Hexagon::LDh_GP_cdnNotPt_V4:
    return Hexagon::LDh_GP_cNotPt_V4;

  case Hexagon::LDuh_GP_cdnPt_V4:
    return Hexagon::LDuh_GP_cPt_V4;

  case Hexagon::LDuh_GP_cdnNotPt_V4:
    return Hexagon::LDuh_GP_cNotPt_V4;

  case Hexagon::LDw_GP_cdnPt_V4:
    return Hexagon::LDw_GP_cPt_V4;

  case Hexagon::LDw_GP_cdnNotPt_V4:
    return Hexagon::LDw_GP_cNotPt_V4;

  case Hexagon::LDrid_GP_cdnPt_V4:
    return Hexagon::LDrid_GP_cPt_V4;

  case Hexagon::LDrid_GP_cdnNotPt_V4:
    return Hexagon::LDrid_GP_cNotPt_V4;

  case Hexagon::LDrib_GP_cdnPt_V4:
    return Hexagon::LDrib_GP_cPt_V4;

  case Hexagon::LDrib_GP_cdnNotPt_V4:
    return Hexagon::LDrib_GP_cNotPt_V4;

  case Hexagon::LDriub_GP_cdnPt_V4:
    return Hexagon::LDriub_GP_cPt_V4;

  case Hexagon::LDriub_GP_cdnNotPt_V4:
    return Hexagon::LDriub_GP_cNotPt_V4;

  case Hexagon::LDrih_GP_cdnPt_V4:
    return Hexagon::LDrih_GP_cPt_V4;

  case Hexagon::LDrih_GP_cdnNotPt_V4:
    return Hexagon::LDrih_GP_cNotPt_V4;

  case Hexagon::LDriuh_GP_cdnPt_V4:
    return Hexagon::LDriuh_GP_cPt_V4;

  case Hexagon::LDriuh_GP_cdnNotPt_V4:
    return Hexagon::LDriuh_GP_cNotPt_V4;

  case Hexagon::LDriw_GP_cdnPt_V4:
    return Hexagon::LDriw_GP_cPt_V4;

  case Hexagon::LDriw_GP_cdnNotPt_V4:
    return Hexagon::LDriw_GP_cNotPt_V4;

  // Conditional add

  case Hexagon::ADD_ri_cdnPt :
    return Hexagon::ADD_ri_cPt;
  case Hexagon::ADD_ri_cdnNotPt :
    return Hexagon::ADD_ri_cNotPt;

  case Hexagon::ADD_rr_cdnPt :
    return Hexagon::ADD_rr_cPt;
  case Hexagon::ADD_rr_cdnNotPt:
    return Hexagon::ADD_rr_cNotPt;

  // Conditional logical Operations

  case Hexagon::XOR_rr_cdnPt :
    return Hexagon::XOR_rr_cPt;
  case Hexagon::XOR_rr_cdnNotPt :
    return Hexagon::XOR_rr_cNotPt;

  case Hexagon::AND_rr_cdnPt :
    return Hexagon::AND_rr_cPt;
  case Hexagon::AND_rr_cdnNotPt :
    return Hexagon::AND_rr_cNotPt;

  case Hexagon::OR_rr_cdnPt :
    return Hexagon::OR_rr_cPt;
  case Hexagon::OR_rr_cdnNotPt :
    return Hexagon::OR_rr_cNotPt;

  // Conditional Subtract

  case Hexagon::SUB_rr_cdnPt :
    return Hexagon::SUB_rr_cPt;
  case Hexagon::SUB_rr_cdnNotPt :
    return Hexagon::SUB_rr_cNotPt;

  // Conditional combine

  case Hexagon::COMBINE_rr_cdnPt :
    return Hexagon::COMBINE_rr_cPt;
  case Hexagon::COMBINE_rr_cdnNotPt :
    return Hexagon::COMBINE_rr_cNotPt;

// Conditional shift operations

  case Hexagon::ASLH_cdnPt_V4 :
    return Hexagon::ASLH_cPt_V4;
  case Hexagon::ASLH_cdnNotPt_V4 :
    return Hexagon::ASLH_cNotPt_V4;

  case Hexagon::ASRH_cdnPt_V4 :
    return Hexagon::ASRH_cPt_V4;
  case Hexagon::ASRH_cdnNotPt_V4 :
    return Hexagon::ASRH_cNotPt_V4;

  case Hexagon::SXTB_cdnPt_V4 :
    return Hexagon::SXTB_cPt_V4;
  case Hexagon::SXTB_cdnNotPt_V4 :
    return Hexagon::SXTB_cNotPt_V4;

  case Hexagon::SXTH_cdnPt_V4 :
    return Hexagon::SXTH_cPt_V4;
  case Hexagon::SXTH_cdnNotPt_V4 :
    return Hexagon::SXTH_cNotPt_V4;

  case Hexagon::ZXTB_cdnPt_V4 :
    return Hexagon::ZXTB_cPt_V4;
  case Hexagon::ZXTB_cdnNotPt_V4 :
    return Hexagon::ZXTB_cNotPt_V4;

  case Hexagon::ZXTH_cdnPt_V4 :
    return Hexagon::ZXTH_cPt_V4;
  case Hexagon::ZXTH_cdnNotPt_V4 :
    return Hexagon::ZXTH_cNotPt_V4;

  // Store byte

  case Hexagon::STrib_imm_cdnPt_V4 :
    return Hexagon::STrib_imm_cPt_V4;

  case Hexagon::STrib_imm_cdnNotPt_V4 :
    return Hexagon::STrib_imm_cNotPt_V4;

  case Hexagon::STrib_cdnPt_nv_V4 :
  case Hexagon::STrib_cPt_nv_V4 :
  case Hexagon::STrib_cdnPt_V4 :
    return Hexagon::STrib_cPt;

  case Hexagon::STrib_cdnNotPt_nv_V4 :
  case Hexagon::STrib_cNotPt_nv_V4 :
  case Hexagon::STrib_cdnNotPt_V4 :
    return Hexagon::STrib_cNotPt;

  case Hexagon::STrib_indexed_cdnPt_V4 :
  case Hexagon::STrib_indexed_cPt_nv_V4 :
  case Hexagon::STrib_indexed_cdnPt_nv_V4 :
    return Hexagon::STrib_indexed_cPt;

  case Hexagon::STrib_indexed_cdnNotPt_V4 :
  case Hexagon::STrib_indexed_cNotPt_nv_V4 :
  case Hexagon::STrib_indexed_cdnNotPt_nv_V4 :
    return Hexagon::STrib_indexed_cNotPt;

  case Hexagon::STrib_indexed_shl_cdnPt_nv_V4:
  case Hexagon::STrib_indexed_shl_cPt_nv_V4 :
  case Hexagon::STrib_indexed_shl_cdnPt_V4 :
    return Hexagon::STrib_indexed_shl_cPt_V4;

  case Hexagon::STrib_indexed_shl_cdnNotPt_nv_V4:
  case Hexagon::STrib_indexed_shl_cNotPt_nv_V4 :
  case Hexagon::STrib_indexed_shl_cdnNotPt_V4 :
    return Hexagon::STrib_indexed_shl_cNotPt_V4;

  case Hexagon::POST_STbri_cdnPt_nv_V4 :
  case Hexagon::POST_STbri_cPt_nv_V4 :
  case Hexagon::POST_STbri_cdnPt_V4 :
    return Hexagon::POST_STbri_cPt;

  case Hexagon::POST_STbri_cdnNotPt_nv_V4 :
  case Hexagon::POST_STbri_cNotPt_nv_V4:
  case Hexagon::POST_STbri_cdnNotPt_V4 :
    return Hexagon::POST_STbri_cNotPt;

  case Hexagon::STb_GP_cdnPt_nv_V4:
  case Hexagon::STb_GP_cdnPt_V4:
  case Hexagon::STb_GP_cPt_nv_V4:
    return Hexagon::STb_GP_cPt_V4;

  case Hexagon::STb_GP_cdnNotPt_nv_V4:
  case Hexagon::STb_GP_cdnNotPt_V4:
  case Hexagon::STb_GP_cNotPt_nv_V4:
    return Hexagon::STb_GP_cNotPt_V4;

  case Hexagon::STrib_GP_cdnPt_nv_V4:
  case Hexagon::STrib_GP_cdnPt_V4:
  case Hexagon::STrib_GP_cPt_nv_V4:
    return Hexagon::STrib_GP_cPt_V4;

  case Hexagon::STrib_GP_cdnNotPt_nv_V4:
  case Hexagon::STrib_GP_cdnNotPt_V4:
  case Hexagon::STrib_GP_cNotPt_nv_V4:
    return Hexagon::STrib_GP_cNotPt_V4;

  // Store new-value byte - unconditional
  case Hexagon::STrib_nv_V4:
    return Hexagon::STrib;

  case Hexagon::STrib_indexed_nv_V4:
    return Hexagon::STrib_indexed;

  case Hexagon::STrib_indexed_shl_nv_V4:
    return Hexagon::STrib_indexed_shl_V4;

  case Hexagon::STrib_shl_nv_V4:
    return Hexagon::STrib_shl_V4;

  case Hexagon::STrib_GP_nv_V4:
    return Hexagon::STrib_GP_V4;

  case Hexagon::STb_GP_nv_V4:
    return Hexagon::STb_GP_V4;

  case Hexagon::POST_STbri_nv_V4:
    return Hexagon::POST_STbri;

  // Store halfword
  case Hexagon::STrih_imm_cdnPt_V4 :
    return Hexagon::STrih_imm_cPt_V4;

  case Hexagon::STrih_imm_cdnNotPt_V4 :
    return Hexagon::STrih_imm_cNotPt_V4;

  case Hexagon::STrih_cdnPt_nv_V4 :
  case Hexagon::STrih_cPt_nv_V4 :
  case Hexagon::STrih_cdnPt_V4 :
    return Hexagon::STrih_cPt;

  case Hexagon::STrih_cdnNotPt_nv_V4 :
  case Hexagon::STrih_cNotPt_nv_V4 :
  case Hexagon::STrih_cdnNotPt_V4 :
    return Hexagon::STrih_cNotPt;

  case Hexagon::STrih_indexed_cdnPt_nv_V4:
  case Hexagon::STrih_indexed_cPt_nv_V4 :
  case Hexagon::STrih_indexed_cdnPt_V4 :
    return Hexagon::STrih_indexed_cPt;

  case Hexagon::STrih_indexed_cdnNotPt_nv_V4:
  case Hexagon::STrih_indexed_cNotPt_nv_V4 :
  case Hexagon::STrih_indexed_cdnNotPt_V4 :
    return Hexagon::STrih_indexed_cNotPt;

  case Hexagon::STrih_indexed_shl_cdnPt_nv_V4 :
  case Hexagon::STrih_indexed_shl_cPt_nv_V4 :
  case Hexagon::STrih_indexed_shl_cdnPt_V4 :
    return Hexagon::STrih_indexed_shl_cPt_V4;

  case Hexagon::STrih_indexed_shl_cdnNotPt_nv_V4 :
  case Hexagon::STrih_indexed_shl_cNotPt_nv_V4 :
  case Hexagon::STrih_indexed_shl_cdnNotPt_V4 :
    return Hexagon::STrih_indexed_shl_cNotPt_V4;

  case Hexagon::POST_SThri_cdnPt_nv_V4 :
  case Hexagon::POST_SThri_cPt_nv_V4 :
  case Hexagon::POST_SThri_cdnPt_V4 :
    return Hexagon::POST_SThri_cPt;

  case Hexagon::POST_SThri_cdnNotPt_nv_V4 :
  case Hexagon::POST_SThri_cNotPt_nv_V4 :
  case Hexagon::POST_SThri_cdnNotPt_V4 :
    return Hexagon::POST_SThri_cNotPt;

  case Hexagon::STh_GP_cdnPt_nv_V4:
  case Hexagon::STh_GP_cdnPt_V4:
  case Hexagon::STh_GP_cPt_nv_V4:
    return Hexagon::STh_GP_cPt_V4;

  case Hexagon::STh_GP_cdnNotPt_nv_V4:
  case Hexagon::STh_GP_cdnNotPt_V4:
  case Hexagon::STh_GP_cNotPt_nv_V4:
    return Hexagon::STh_GP_cNotPt_V4;

  case Hexagon::STrih_GP_cdnPt_nv_V4:
  case Hexagon::STrih_GP_cdnPt_V4:
  case Hexagon::STrih_GP_cPt_nv_V4:
    return Hexagon::STrih_GP_cPt_V4;

  case Hexagon::STrih_GP_cdnNotPt_nv_V4:
  case Hexagon::STrih_GP_cdnNotPt_V4:
  case Hexagon::STrih_GP_cNotPt_nv_V4:
    return Hexagon::STrih_GP_cNotPt_V4;

  // Store new-value halfword - unconditional

  case Hexagon::STrih_nv_V4:
    return Hexagon::STrih;

  case Hexagon::STrih_indexed_nv_V4:
    return Hexagon::STrih_indexed;

  case Hexagon::STrih_indexed_shl_nv_V4:
    return Hexagon::STrih_indexed_shl_V4;

  case Hexagon::STrih_shl_nv_V4:
    return Hexagon::STrih_shl_V4;

  case Hexagon::STrih_GP_nv_V4:
    return Hexagon::STrih_GP_V4;

  case Hexagon::STh_GP_nv_V4:
    return Hexagon::STh_GP_V4;

  case Hexagon::POST_SThri_nv_V4:
    return Hexagon::POST_SThri;

   // Store word

   case Hexagon::STriw_imm_cdnPt_V4 :
    return Hexagon::STriw_imm_cPt_V4;

  case Hexagon::STriw_imm_cdnNotPt_V4 :
    return Hexagon::STriw_imm_cNotPt_V4;

  case Hexagon::STriw_cdnPt_nv_V4 :
  case Hexagon::STriw_cPt_nv_V4 :
  case Hexagon::STriw_cdnPt_V4 :
    return Hexagon::STriw_cPt;

  case Hexagon::STriw_cdnNotPt_nv_V4 :
  case Hexagon::STriw_cNotPt_nv_V4 :
  case Hexagon::STriw_cdnNotPt_V4 :
    return Hexagon::STriw_cNotPt;

  case Hexagon::STriw_indexed_cdnPt_nv_V4 :
  case Hexagon::STriw_indexed_cPt_nv_V4 :
  case Hexagon::STriw_indexed_cdnPt_V4 :
    return Hexagon::STriw_indexed_cPt;

  case Hexagon::STriw_indexed_cdnNotPt_nv_V4 :
  case Hexagon::STriw_indexed_cNotPt_nv_V4 :
  case Hexagon::STriw_indexed_cdnNotPt_V4 :
    return Hexagon::STriw_indexed_cNotPt;

  case Hexagon::STriw_indexed_shl_cdnPt_nv_V4 :
  case Hexagon::STriw_indexed_shl_cPt_nv_V4 :
  case Hexagon::STriw_indexed_shl_cdnPt_V4 :
    return Hexagon::STriw_indexed_shl_cPt_V4;

  case Hexagon::STriw_indexed_shl_cdnNotPt_nv_V4 :
  case Hexagon::STriw_indexed_shl_cNotPt_nv_V4 :
  case Hexagon::STriw_indexed_shl_cdnNotPt_V4 :
    return Hexagon::STriw_indexed_shl_cNotPt_V4;

  case Hexagon::POST_STwri_cdnPt_nv_V4 :
  case Hexagon::POST_STwri_cPt_nv_V4 :
  case Hexagon::POST_STwri_cdnPt_V4 :
    return Hexagon::POST_STwri_cPt;

  case Hexagon::POST_STwri_cdnNotPt_nv_V4 :
  case Hexagon::POST_STwri_cNotPt_nv_V4 :
  case Hexagon::POST_STwri_cdnNotPt_V4 :
    return Hexagon::POST_STwri_cNotPt;

  case Hexagon::STw_GP_cdnPt_nv_V4:
  case Hexagon::STw_GP_cdnPt_V4:
  case Hexagon::STw_GP_cPt_nv_V4:
    return Hexagon::STw_GP_cPt_V4;

  case Hexagon::STw_GP_cdnNotPt_nv_V4:
  case Hexagon::STw_GP_cdnNotPt_V4:
  case Hexagon::STw_GP_cNotPt_nv_V4:
    return Hexagon::STw_GP_cNotPt_V4;

  case Hexagon::STriw_GP_cdnPt_nv_V4:
  case Hexagon::STriw_GP_cdnPt_V4:
  case Hexagon::STriw_GP_cPt_nv_V4:
    return Hexagon::STriw_GP_cPt_V4;

  case Hexagon::STriw_GP_cdnNotPt_nv_V4:
  case Hexagon::STriw_GP_cdnNotPt_V4:
  case Hexagon::STriw_GP_cNotPt_nv_V4:
    return Hexagon::STriw_GP_cNotPt_V4;

  // Store new-value word - unconditional

  case Hexagon::STriw_nv_V4:
    return Hexagon::STriw;

  case Hexagon::STriw_indexed_nv_V4:
    return Hexagon::STriw_indexed;

  case Hexagon::STriw_indexed_shl_nv_V4:
    return Hexagon::STriw_indexed_shl_V4;

  case Hexagon::STriw_shl_nv_V4:
    return Hexagon::STriw_shl_V4;

  case Hexagon::STriw_GP_nv_V4:
    return Hexagon::STriw_GP_V4;

  case Hexagon::STw_GP_nv_V4:
    return Hexagon::STw_GP_V4;

  case Hexagon::POST_STwri_nv_V4:
    return Hexagon::POST_STwri;

 // Store doubleword

  case Hexagon::STrid_cdnPt_V4 :
    return Hexagon::STrid_cPt;

  case Hexagon::STrid_cdnNotPt_V4 :
    return Hexagon::STrid_cNotPt;

  case Hexagon::STrid_indexed_cdnPt_V4 :
    return Hexagon::STrid_indexed_cPt;

  case Hexagon::STrid_indexed_cdnNotPt_V4 :
    return Hexagon::STrid_indexed_cNotPt;

  case Hexagon::STrid_indexed_shl_cdnPt_V4 :
    return Hexagon::STrid_indexed_shl_cPt_V4;

  case Hexagon::STrid_indexed_shl_cdnNotPt_V4 :
    return Hexagon::STrid_indexed_shl_cNotPt_V4;

  case Hexagon::POST_STdri_cdnPt_V4 :
    return Hexagon::POST_STdri_cPt;

  case Hexagon::POST_STdri_cdnNotPt_V4 :
    return Hexagon::POST_STdri_cNotPt;

  case Hexagon::STd_GP_cdnPt_V4 :
    return Hexagon::STd_GP_cPt_V4;

  case Hexagon::STd_GP_cdnNotPt_V4 :
    return Hexagon::STd_GP_cNotPt_V4;

  case Hexagon::STrid_GP_cdnPt_V4 :
    return Hexagon::STrid_GP_cPt_V4;

  case Hexagon::STrid_GP_cdnNotPt_V4 :
    return Hexagon::STrid_GP_cNotPt_V4;
  }
}

bool HexagonPacketizerList::DemoteToDotOld(MachineInstr* MI) {
  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  int NewOpcode = GetDotOldOp(MI->getOpcode());
  MI->setDesc(QII->get(NewOpcode));
  return true;
}

// Returns true if an instruction is predicated on p0 and false if it's
// predicated on !p0.

static bool GetPredicateSense(MachineInstr* MI,
                              const HexagonInstrInfo *QII) {

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unknown predicate sense of the instruction");
  case Hexagon::TFR_cPt:
  case Hexagon::TFR_cdnPt:
  case Hexagon::TFRI_cPt:
  case Hexagon::TFRI_cdnPt:
  case Hexagon::STrib_cPt :
  case Hexagon::STrib_cdnPt_V4 :
  case Hexagon::STrib_indexed_cPt :
  case Hexagon::STrib_indexed_cdnPt_V4 :
  case Hexagon::STrib_indexed_shl_cPt_V4 :
  case Hexagon::STrib_indexed_shl_cdnPt_V4 :
  case Hexagon::POST_STbri_cPt :
  case Hexagon::POST_STbri_cdnPt_V4 :
  case Hexagon::STrih_cPt :
  case Hexagon::STrih_cdnPt_V4 :
  case Hexagon::STrih_indexed_cPt :
  case Hexagon::STrih_indexed_cdnPt_V4 :
  case Hexagon::STrih_indexed_shl_cPt_V4 :
  case Hexagon::STrih_indexed_shl_cdnPt_V4 :
  case Hexagon::POST_SThri_cPt :
  case Hexagon::POST_SThri_cdnPt_V4 :
  case Hexagon::STriw_cPt :
  case Hexagon::STriw_cdnPt_V4 :
  case Hexagon::STriw_indexed_cPt :
  case Hexagon::STriw_indexed_cdnPt_V4 :
  case Hexagon::STriw_indexed_shl_cPt_V4 :
  case Hexagon::STriw_indexed_shl_cdnPt_V4 :
  case Hexagon::POST_STwri_cPt :
  case Hexagon::POST_STwri_cdnPt_V4 :
  case Hexagon::STrib_imm_cPt_V4 :
  case Hexagon::STrib_imm_cdnPt_V4 :
  case Hexagon::STrid_cPt :
  case Hexagon::STrid_cdnPt_V4 :
  case Hexagon::STrid_indexed_cPt :
  case Hexagon::STrid_indexed_cdnPt_V4 :
  case Hexagon::STrid_indexed_shl_cPt_V4 :
  case Hexagon::STrid_indexed_shl_cdnPt_V4 :
  case Hexagon::POST_STdri_cPt :
  case Hexagon::POST_STdri_cdnPt_V4 :
  case Hexagon::STrih_imm_cPt_V4 :
  case Hexagon::STrih_imm_cdnPt_V4 :
  case Hexagon::STriw_imm_cPt_V4 :
  case Hexagon::STriw_imm_cdnPt_V4 :
  case Hexagon::JMP_cdnPt :
  case Hexagon::LDrid_cPt :
  case Hexagon::LDrid_cdnPt :
  case Hexagon::LDrid_indexed_cPt :
  case Hexagon::LDrid_indexed_cdnPt :
  case Hexagon::POST_LDrid_cPt :
  case Hexagon::POST_LDrid_cdnPt_V4 :
  case Hexagon::LDriw_cPt :
  case Hexagon::LDriw_cdnPt :
  case Hexagon::LDriw_indexed_cPt :
  case Hexagon::LDriw_indexed_cdnPt :
  case Hexagon::POST_LDriw_cPt :
  case Hexagon::POST_LDriw_cdnPt_V4 :
  case Hexagon::LDrih_cPt :
  case Hexagon::LDrih_cdnPt :
  case Hexagon::LDrih_indexed_cPt :
  case Hexagon::LDrih_indexed_cdnPt :
  case Hexagon::POST_LDrih_cPt :
  case Hexagon::POST_LDrih_cdnPt_V4 :
  case Hexagon::LDrib_cPt :
  case Hexagon::LDrib_cdnPt :
  case Hexagon::LDrib_indexed_cPt :
  case Hexagon::LDrib_indexed_cdnPt :
  case Hexagon::POST_LDrib_cPt :
  case Hexagon::POST_LDrib_cdnPt_V4 :
  case Hexagon::LDriuh_cPt :
  case Hexagon::LDriuh_cdnPt :
  case Hexagon::LDriuh_indexed_cPt :
  case Hexagon::LDriuh_indexed_cdnPt :
  case Hexagon::POST_LDriuh_cPt :
  case Hexagon::POST_LDriuh_cdnPt_V4 :
  case Hexagon::LDriub_cPt :
  case Hexagon::LDriub_cdnPt :
  case Hexagon::LDriub_indexed_cPt :
  case Hexagon::LDriub_indexed_cdnPt :
  case Hexagon::POST_LDriub_cPt :
  case Hexagon::POST_LDriub_cdnPt_V4 :
  case Hexagon::LDrid_indexed_cPt_V4 :
  case Hexagon::LDrid_indexed_cdnPt_V4 :
  case Hexagon::LDrid_indexed_shl_cPt_V4 :
  case Hexagon::LDrid_indexed_shl_cdnPt_V4 :
  case Hexagon::LDrib_indexed_cPt_V4 :
  case Hexagon::LDrib_indexed_cdnPt_V4 :
  case Hexagon::LDrib_indexed_shl_cPt_V4 :
  case Hexagon::LDrib_indexed_shl_cdnPt_V4 :
  case Hexagon::LDriub_indexed_cPt_V4 :
  case Hexagon::LDriub_indexed_cdnPt_V4 :
  case Hexagon::LDriub_indexed_shl_cPt_V4 :
  case Hexagon::LDriub_indexed_shl_cdnPt_V4 :
  case Hexagon::LDrih_indexed_cPt_V4 :
  case Hexagon::LDrih_indexed_cdnPt_V4 :
  case Hexagon::LDrih_indexed_shl_cPt_V4 :
  case Hexagon::LDrih_indexed_shl_cdnPt_V4 :
  case Hexagon::LDriuh_indexed_cPt_V4 :
  case Hexagon::LDriuh_indexed_cdnPt_V4 :
  case Hexagon::LDriuh_indexed_shl_cPt_V4 :
  case Hexagon::LDriuh_indexed_shl_cdnPt_V4 :
  case Hexagon::LDriw_indexed_cPt_V4 :
  case Hexagon::LDriw_indexed_cdnPt_V4 :
  case Hexagon::LDriw_indexed_shl_cPt_V4 :
  case Hexagon::LDriw_indexed_shl_cdnPt_V4 :
  case Hexagon::ADD_ri_cPt :
  case Hexagon::ADD_ri_cdnPt :
  case Hexagon::ADD_rr_cPt :
  case Hexagon::ADD_rr_cdnPt :
  case Hexagon::XOR_rr_cPt :
  case Hexagon::XOR_rr_cdnPt :
  case Hexagon::AND_rr_cPt :
  case Hexagon::AND_rr_cdnPt :
  case Hexagon::OR_rr_cPt :
  case Hexagon::OR_rr_cdnPt :
  case Hexagon::SUB_rr_cPt :
  case Hexagon::SUB_rr_cdnPt :
  case Hexagon::COMBINE_rr_cPt :
  case Hexagon::COMBINE_rr_cdnPt :
  case Hexagon::ASLH_cPt_V4 :
  case Hexagon::ASLH_cdnPt_V4 :
  case Hexagon::ASRH_cPt_V4 :
  case Hexagon::ASRH_cdnPt_V4 :
  case Hexagon::SXTB_cPt_V4 :
  case Hexagon::SXTB_cdnPt_V4 :
  case Hexagon::SXTH_cPt_V4 :
  case Hexagon::SXTH_cdnPt_V4 :
  case Hexagon::ZXTB_cPt_V4 :
  case Hexagon::ZXTB_cdnPt_V4 :
  case Hexagon::ZXTH_cPt_V4 :
  case Hexagon::ZXTH_cdnPt_V4 :
  case Hexagon::LDrid_GP_cPt_V4 :
  case Hexagon::LDrib_GP_cPt_V4 :
  case Hexagon::LDriub_GP_cPt_V4 :
  case Hexagon::LDrih_GP_cPt_V4 :
  case Hexagon::LDriuh_GP_cPt_V4 :
  case Hexagon::LDriw_GP_cPt_V4 :
  case Hexagon::LDd_GP_cPt_V4 :
  case Hexagon::LDb_GP_cPt_V4 :
  case Hexagon::LDub_GP_cPt_V4 :
  case Hexagon::LDh_GP_cPt_V4 :
  case Hexagon::LDuh_GP_cPt_V4 :
  case Hexagon::LDw_GP_cPt_V4 :
  case Hexagon::STrid_GP_cPt_V4 :
  case Hexagon::STrib_GP_cPt_V4 :
  case Hexagon::STrih_GP_cPt_V4 :
  case Hexagon::STriw_GP_cPt_V4 :
  case Hexagon::STd_GP_cPt_V4 :
  case Hexagon::STb_GP_cPt_V4 :
  case Hexagon::STh_GP_cPt_V4 :
  case Hexagon::STw_GP_cPt_V4 :
  case Hexagon::LDrid_GP_cdnPt_V4 :
  case Hexagon::LDrib_GP_cdnPt_V4 :
  case Hexagon::LDriub_GP_cdnPt_V4 :
  case Hexagon::LDrih_GP_cdnPt_V4 :
  case Hexagon::LDriuh_GP_cdnPt_V4 :
  case Hexagon::LDriw_GP_cdnPt_V4 :
  case Hexagon::LDd_GP_cdnPt_V4 :
  case Hexagon::LDb_GP_cdnPt_V4 :
  case Hexagon::LDub_GP_cdnPt_V4 :
  case Hexagon::LDh_GP_cdnPt_V4 :
  case Hexagon::LDuh_GP_cdnPt_V4 :
  case Hexagon::LDw_GP_cdnPt_V4 :
  case Hexagon::STrid_GP_cdnPt_V4 :
  case Hexagon::STrib_GP_cdnPt_V4 :
  case Hexagon::STrih_GP_cdnPt_V4 :
  case Hexagon::STriw_GP_cdnPt_V4 :
  case Hexagon::STd_GP_cdnPt_V4 :
  case Hexagon::STb_GP_cdnPt_V4 :
  case Hexagon::STh_GP_cdnPt_V4 :
  case Hexagon::STw_GP_cdnPt_V4 :
    return true;

  case Hexagon::TFR_cNotPt:
  case Hexagon::TFR_cdnNotPt:
  case Hexagon::TFRI_cNotPt:
  case Hexagon::TFRI_cdnNotPt:
  case Hexagon::STrib_cNotPt :
  case Hexagon::STrib_cdnNotPt_V4 :
  case Hexagon::STrib_indexed_cNotPt :
  case Hexagon::STrib_indexed_cdnNotPt_V4 :
  case Hexagon::STrib_indexed_shl_cNotPt_V4 :
  case Hexagon::STrib_indexed_shl_cdnNotPt_V4 :
  case Hexagon::POST_STbri_cNotPt :
  case Hexagon::POST_STbri_cdnNotPt_V4 :
  case Hexagon::STrih_cNotPt :
  case Hexagon::STrih_cdnNotPt_V4 :
  case Hexagon::STrih_indexed_cNotPt :
  case Hexagon::STrih_indexed_cdnNotPt_V4 :
  case Hexagon::STrih_indexed_shl_cNotPt_V4 :
  case Hexagon::STrih_indexed_shl_cdnNotPt_V4 :
  case Hexagon::POST_SThri_cNotPt :
  case Hexagon::POST_SThri_cdnNotPt_V4 :
  case Hexagon::STriw_cNotPt :
  case Hexagon::STriw_cdnNotPt_V4 :
  case Hexagon::STriw_indexed_cNotPt :
  case Hexagon::STriw_indexed_cdnNotPt_V4 :
  case Hexagon::STriw_indexed_shl_cNotPt_V4 :
  case Hexagon::STriw_indexed_shl_cdnNotPt_V4 :
  case Hexagon::POST_STwri_cNotPt :
  case Hexagon::POST_STwri_cdnNotPt_V4 :
  case Hexagon::STrib_imm_cNotPt_V4 :
  case Hexagon::STrib_imm_cdnNotPt_V4 :
  case Hexagon::STrid_cNotPt :
  case Hexagon::STrid_cdnNotPt_V4 :
  case Hexagon::STrid_indexed_cdnNotPt_V4 :
  case Hexagon::STrid_indexed_cNotPt :
  case Hexagon::STrid_indexed_shl_cNotPt_V4 :
  case Hexagon::STrid_indexed_shl_cdnNotPt_V4 :
  case Hexagon::POST_STdri_cNotPt :
  case Hexagon::POST_STdri_cdnNotPt_V4 :
  case Hexagon::STrih_imm_cNotPt_V4 :
  case Hexagon::STrih_imm_cdnNotPt_V4 :
  case Hexagon::STriw_imm_cNotPt_V4 :
  case Hexagon::STriw_imm_cdnNotPt_V4 :
  case Hexagon::JMP_cdnNotPt :
  case Hexagon::LDrid_cNotPt :
  case Hexagon::LDrid_cdnNotPt :
  case Hexagon::LDrid_indexed_cNotPt :
  case Hexagon::LDrid_indexed_cdnNotPt :
  case Hexagon::POST_LDrid_cNotPt :
  case Hexagon::POST_LDrid_cdnNotPt_V4 :
  case Hexagon::LDriw_cNotPt :
  case Hexagon::LDriw_cdnNotPt :
  case Hexagon::LDriw_indexed_cNotPt :
  case Hexagon::LDriw_indexed_cdnNotPt :
  case Hexagon::POST_LDriw_cNotPt :
  case Hexagon::POST_LDriw_cdnNotPt_V4 :
  case Hexagon::LDrih_cNotPt :
  case Hexagon::LDrih_cdnNotPt :
  case Hexagon::LDrih_indexed_cNotPt :
  case Hexagon::LDrih_indexed_cdnNotPt :
  case Hexagon::POST_LDrih_cNotPt :
  case Hexagon::POST_LDrih_cdnNotPt_V4 :
  case Hexagon::LDrib_cNotPt :
  case Hexagon::LDrib_cdnNotPt :
  case Hexagon::LDrib_indexed_cNotPt :
  case Hexagon::LDrib_indexed_cdnNotPt :
  case Hexagon::POST_LDrib_cNotPt :
  case Hexagon::POST_LDrib_cdnNotPt_V4 :
  case Hexagon::LDriuh_cNotPt :
  case Hexagon::LDriuh_cdnNotPt :
  case Hexagon::LDriuh_indexed_cNotPt :
  case Hexagon::LDriuh_indexed_cdnNotPt :
  case Hexagon::POST_LDriuh_cNotPt :
  case Hexagon::POST_LDriuh_cdnNotPt_V4 :
  case Hexagon::LDriub_cNotPt :
  case Hexagon::LDriub_cdnNotPt :
  case Hexagon::LDriub_indexed_cNotPt :
  case Hexagon::LDriub_indexed_cdnNotPt :
  case Hexagon::POST_LDriub_cNotPt :
  case Hexagon::POST_LDriub_cdnNotPt_V4 :
  case Hexagon::LDrid_indexed_cNotPt_V4 :
  case Hexagon::LDrid_indexed_cdnNotPt_V4 :
  case Hexagon::LDrid_indexed_shl_cNotPt_V4 :
  case Hexagon::LDrid_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDrib_indexed_cNotPt_V4 :
  case Hexagon::LDrib_indexed_cdnNotPt_V4 :
  case Hexagon::LDrib_indexed_shl_cNotPt_V4 :
  case Hexagon::LDrib_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDriub_indexed_cNotPt_V4 :
  case Hexagon::LDriub_indexed_cdnNotPt_V4 :
  case Hexagon::LDriub_indexed_shl_cNotPt_V4 :
  case Hexagon::LDriub_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDrih_indexed_cNotPt_V4 :
  case Hexagon::LDrih_indexed_cdnNotPt_V4 :
  case Hexagon::LDrih_indexed_shl_cNotPt_V4 :
  case Hexagon::LDrih_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDriuh_indexed_cNotPt_V4 :
  case Hexagon::LDriuh_indexed_cdnNotPt_V4 :
  case Hexagon::LDriuh_indexed_shl_cNotPt_V4 :
  case Hexagon::LDriuh_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDriw_indexed_cNotPt_V4 :
  case Hexagon::LDriw_indexed_cdnNotPt_V4 :
  case Hexagon::LDriw_indexed_shl_cNotPt_V4 :
  case Hexagon::LDriw_indexed_shl_cdnNotPt_V4 :
  case Hexagon::ADD_ri_cNotPt :
  case Hexagon::ADD_ri_cdnNotPt :
  case Hexagon::ADD_rr_cNotPt :
  case Hexagon::ADD_rr_cdnNotPt :
  case Hexagon::XOR_rr_cNotPt :
  case Hexagon::XOR_rr_cdnNotPt :
  case Hexagon::AND_rr_cNotPt :
  case Hexagon::AND_rr_cdnNotPt :
  case Hexagon::OR_rr_cNotPt :
  case Hexagon::OR_rr_cdnNotPt :
  case Hexagon::SUB_rr_cNotPt :
  case Hexagon::SUB_rr_cdnNotPt :
  case Hexagon::COMBINE_rr_cNotPt :
  case Hexagon::COMBINE_rr_cdnNotPt :
  case Hexagon::ASLH_cNotPt_V4 :
  case Hexagon::ASLH_cdnNotPt_V4 :
  case Hexagon::ASRH_cNotPt_V4 :
  case Hexagon::ASRH_cdnNotPt_V4 :
  case Hexagon::SXTB_cNotPt_V4 :
  case Hexagon::SXTB_cdnNotPt_V4 :
  case Hexagon::SXTH_cNotPt_V4 :
  case Hexagon::SXTH_cdnNotPt_V4 :
  case Hexagon::ZXTB_cNotPt_V4 :
  case Hexagon::ZXTB_cdnNotPt_V4 :
  case Hexagon::ZXTH_cNotPt_V4 :
  case Hexagon::ZXTH_cdnNotPt_V4 :

  case Hexagon::LDrid_GP_cNotPt_V4 :
  case Hexagon::LDrib_GP_cNotPt_V4 :
  case Hexagon::LDriub_GP_cNotPt_V4 :
  case Hexagon::LDrih_GP_cNotPt_V4 :
  case Hexagon::LDriuh_GP_cNotPt_V4 :
  case Hexagon::LDriw_GP_cNotPt_V4 :
  case Hexagon::LDd_GP_cNotPt_V4 :
  case Hexagon::LDb_GP_cNotPt_V4 :
  case Hexagon::LDub_GP_cNotPt_V4 :
  case Hexagon::LDh_GP_cNotPt_V4 :
  case Hexagon::LDuh_GP_cNotPt_V4 :
  case Hexagon::LDw_GP_cNotPt_V4 :
  case Hexagon::STrid_GP_cNotPt_V4 :
  case Hexagon::STrib_GP_cNotPt_V4 :
  case Hexagon::STrih_GP_cNotPt_V4 :
  case Hexagon::STriw_GP_cNotPt_V4 :
  case Hexagon::STd_GP_cNotPt_V4 :
  case Hexagon::STb_GP_cNotPt_V4 :
  case Hexagon::STh_GP_cNotPt_V4 :
  case Hexagon::STw_GP_cNotPt_V4 :
  case Hexagon::LDrid_GP_cdnNotPt_V4 :
  case Hexagon::LDrib_GP_cdnNotPt_V4 :
  case Hexagon::LDriub_GP_cdnNotPt_V4 :
  case Hexagon::LDrih_GP_cdnNotPt_V4 :
  case Hexagon::LDriuh_GP_cdnNotPt_V4 :
  case Hexagon::LDriw_GP_cdnNotPt_V4 :
  case Hexagon::LDd_GP_cdnNotPt_V4 :
  case Hexagon::LDb_GP_cdnNotPt_V4 :
  case Hexagon::LDub_GP_cdnNotPt_V4 :
  case Hexagon::LDh_GP_cdnNotPt_V4 :
  case Hexagon::LDuh_GP_cdnNotPt_V4 :
  case Hexagon::LDw_GP_cdnNotPt_V4 :
  case Hexagon::STrid_GP_cdnNotPt_V4 :
  case Hexagon::STrib_GP_cdnNotPt_V4 :
  case Hexagon::STrih_GP_cdnNotPt_V4 :
  case Hexagon::STriw_GP_cdnNotPt_V4 :
  case Hexagon::STd_GP_cdnNotPt_V4 :
  case Hexagon::STb_GP_cdnNotPt_V4 :
  case Hexagon::STh_GP_cdnNotPt_V4 :
  case Hexagon::STw_GP_cdnNotPt_V4 :
    return false;
  }
  // return *some value* to avoid compiler warning
  return false;
}

bool HexagonPacketizerList::isDotNewInst(MachineInstr* MI) {
  if (isNewValueInst(MI))
    return true;

  switch (MI->getOpcode()) {
  case Hexagon::TFR_cdnNotPt:
  case Hexagon::TFR_cdnPt:
  case Hexagon::TFRI_cdnNotPt:
  case Hexagon::TFRI_cdnPt:
  case Hexagon::LDrid_cdnPt :
  case Hexagon::LDrid_cdnNotPt :
  case Hexagon::LDrid_indexed_cdnPt :
  case Hexagon::LDrid_indexed_cdnNotPt :
  case Hexagon::POST_LDrid_cdnPt_V4 :
  case Hexagon::POST_LDrid_cdnNotPt_V4 :
  case Hexagon::LDriw_cdnPt :
  case Hexagon::LDriw_cdnNotPt :
  case Hexagon::LDriw_indexed_cdnPt :
  case Hexagon::LDriw_indexed_cdnNotPt :
  case Hexagon::POST_LDriw_cdnPt_V4 :
  case Hexagon::POST_LDriw_cdnNotPt_V4 :
  case Hexagon::LDrih_cdnPt :
  case Hexagon::LDrih_cdnNotPt :
  case Hexagon::LDrih_indexed_cdnPt :
  case Hexagon::LDrih_indexed_cdnNotPt :
  case Hexagon::POST_LDrih_cdnPt_V4 :
  case Hexagon::POST_LDrih_cdnNotPt_V4 :
  case Hexagon::LDrib_cdnPt :
  case Hexagon::LDrib_cdnNotPt :
  case Hexagon::LDrib_indexed_cdnPt :
  case Hexagon::LDrib_indexed_cdnNotPt :
  case Hexagon::POST_LDrib_cdnPt_V4 :
  case Hexagon::POST_LDrib_cdnNotPt_V4 :
  case Hexagon::LDriuh_cdnPt :
  case Hexagon::LDriuh_cdnNotPt :
  case Hexagon::LDriuh_indexed_cdnPt :
  case Hexagon::LDriuh_indexed_cdnNotPt :
  case Hexagon::POST_LDriuh_cdnPt_V4 :
  case Hexagon::POST_LDriuh_cdnNotPt_V4 :
  case Hexagon::LDriub_cdnPt :
  case Hexagon::LDriub_cdnNotPt :
  case Hexagon::LDriub_indexed_cdnPt :
  case Hexagon::LDriub_indexed_cdnNotPt :
  case Hexagon::POST_LDriub_cdnPt_V4 :
  case Hexagon::POST_LDriub_cdnNotPt_V4 :

  case Hexagon::LDrid_indexed_cdnPt_V4 :
  case Hexagon::LDrid_indexed_cdnNotPt_V4 :
  case Hexagon::LDrid_indexed_shl_cdnPt_V4 :
  case Hexagon::LDrid_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDrib_indexed_cdnPt_V4 :
  case Hexagon::LDrib_indexed_cdnNotPt_V4 :
  case Hexagon::LDrib_indexed_shl_cdnPt_V4 :
  case Hexagon::LDrib_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDriub_indexed_cdnPt_V4 :
  case Hexagon::LDriub_indexed_cdnNotPt_V4 :
  case Hexagon::LDriub_indexed_shl_cdnPt_V4 :
  case Hexagon::LDriub_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDrih_indexed_cdnPt_V4 :
  case Hexagon::LDrih_indexed_cdnNotPt_V4 :
  case Hexagon::LDrih_indexed_shl_cdnPt_V4 :
  case Hexagon::LDrih_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDriuh_indexed_cdnPt_V4 :
  case Hexagon::LDriuh_indexed_cdnNotPt_V4 :
  case Hexagon::LDriuh_indexed_shl_cdnPt_V4 :
  case Hexagon::LDriuh_indexed_shl_cdnNotPt_V4 :
  case Hexagon::LDriw_indexed_cdnPt_V4 :
  case Hexagon::LDriw_indexed_cdnNotPt_V4 :
  case Hexagon::LDriw_indexed_shl_cdnPt_V4 :
  case Hexagon::LDriw_indexed_shl_cdnNotPt_V4 :

// Coditional add
  case Hexagon::ADD_ri_cdnPt:
  case Hexagon::ADD_ri_cdnNotPt:
  case Hexagon::ADD_rr_cdnPt:
  case Hexagon::ADD_rr_cdnNotPt:

  // Conditional logical operations
  case Hexagon::XOR_rr_cdnPt :
  case Hexagon::XOR_rr_cdnNotPt :
  case Hexagon::AND_rr_cdnPt :
  case Hexagon::AND_rr_cdnNotPt :
  case Hexagon::OR_rr_cdnPt :
  case Hexagon::OR_rr_cdnNotPt :

  // Conditonal subtract
  case Hexagon::SUB_rr_cdnPt :
  case Hexagon::SUB_rr_cdnNotPt :

  // Conditional combine
  case Hexagon::COMBINE_rr_cdnPt :
  case Hexagon::COMBINE_rr_cdnNotPt :

  // Conditional shift operations
  case Hexagon::ASLH_cdnPt_V4:
  case Hexagon::ASLH_cdnNotPt_V4:
  case Hexagon::ASRH_cdnPt_V4:
  case Hexagon::ASRH_cdnNotPt_V4:
  case Hexagon::SXTB_cdnPt_V4:
  case Hexagon::SXTB_cdnNotPt_V4:
  case Hexagon::SXTH_cdnPt_V4:
  case Hexagon::SXTH_cdnNotPt_V4:
  case Hexagon::ZXTB_cdnPt_V4:
  case Hexagon::ZXTB_cdnNotPt_V4:
  case Hexagon::ZXTH_cdnPt_V4:
  case Hexagon::ZXTH_cdnNotPt_V4:

  // Conditional stores
  case Hexagon::STrib_imm_cdnPt_V4 :
  case Hexagon::STrib_imm_cdnNotPt_V4 :
  case Hexagon::STrib_cdnPt_V4 :
  case Hexagon::STrib_cdnNotPt_V4 :
  case Hexagon::STrib_indexed_cdnPt_V4 :
  case Hexagon::STrib_indexed_cdnNotPt_V4 :
  case Hexagon::POST_STbri_cdnPt_V4 :
  case Hexagon::POST_STbri_cdnNotPt_V4 :
  case Hexagon::STrib_indexed_shl_cdnPt_V4 :
  case Hexagon::STrib_indexed_shl_cdnNotPt_V4 :

  // Store doubleword conditionally
  case Hexagon::STrid_indexed_cdnPt_V4 :
  case Hexagon::STrid_indexed_cdnNotPt_V4 :
  case Hexagon::STrid_indexed_shl_cdnPt_V4 :
  case Hexagon::STrid_indexed_shl_cdnNotPt_V4 :
  case Hexagon::POST_STdri_cdnPt_V4 :
  case Hexagon::POST_STdri_cdnNotPt_V4 :

  // Store halfword conditionally
  case Hexagon::STrih_cdnPt_V4 :
  case Hexagon::STrih_cdnNotPt_V4 :
  case Hexagon::STrih_indexed_cdnPt_V4 :
  case Hexagon::STrih_indexed_cdnNotPt_V4 :
  case Hexagon::STrih_imm_cdnPt_V4 :
  case Hexagon::STrih_imm_cdnNotPt_V4 :
  case Hexagon::STrih_indexed_shl_cdnPt_V4 :
  case Hexagon::STrih_indexed_shl_cdnNotPt_V4 :
  case Hexagon::POST_SThri_cdnPt_V4 :
  case Hexagon::POST_SThri_cdnNotPt_V4 :

  // Store word conditionally
  case Hexagon::STriw_cdnPt_V4 :
  case Hexagon::STriw_cdnNotPt_V4 :
  case Hexagon::STriw_indexed_cdnPt_V4 :
  case Hexagon::STriw_indexed_cdnNotPt_V4 :
  case Hexagon::STriw_imm_cdnPt_V4 :
  case Hexagon::STriw_imm_cdnNotPt_V4 :
  case Hexagon::STriw_indexed_shl_cdnPt_V4 :
  case Hexagon::STriw_indexed_shl_cdnNotPt_V4 :
  case Hexagon::POST_STwri_cdnPt_V4 :
  case Hexagon::POST_STwri_cdnNotPt_V4 :

  case Hexagon::LDd_GP_cdnPt_V4:
  case Hexagon::LDd_GP_cdnNotPt_V4:
  case Hexagon::LDb_GP_cdnPt_V4:
  case Hexagon::LDb_GP_cdnNotPt_V4:
  case Hexagon::LDub_GP_cdnPt_V4:
  case Hexagon::LDub_GP_cdnNotPt_V4:
  case Hexagon::LDh_GP_cdnPt_V4:
  case Hexagon::LDh_GP_cdnNotPt_V4:
  case Hexagon::LDuh_GP_cdnPt_V4:
  case Hexagon::LDuh_GP_cdnNotPt_V4:
  case Hexagon::LDw_GP_cdnPt_V4:
  case Hexagon::LDw_GP_cdnNotPt_V4:
  case Hexagon::LDrid_GP_cdnPt_V4:
  case Hexagon::LDrid_GP_cdnNotPt_V4:
  case Hexagon::LDrib_GP_cdnPt_V4:
  case Hexagon::LDrib_GP_cdnNotPt_V4:
  case Hexagon::LDriub_GP_cdnPt_V4:
  case Hexagon::LDriub_GP_cdnNotPt_V4:
  case Hexagon::LDrih_GP_cdnPt_V4:
  case Hexagon::LDrih_GP_cdnNotPt_V4:
  case Hexagon::LDriuh_GP_cdnPt_V4:
  case Hexagon::LDriuh_GP_cdnNotPt_V4:
  case Hexagon::LDriw_GP_cdnPt_V4:
  case Hexagon::LDriw_GP_cdnNotPt_V4:

  case Hexagon::STrid_GP_cdnPt_V4:
  case Hexagon::STrid_GP_cdnNotPt_V4:
  case Hexagon::STrib_GP_cdnPt_V4:
  case Hexagon::STrib_GP_cdnNotPt_V4:
  case Hexagon::STrih_GP_cdnPt_V4:
  case Hexagon::STrih_GP_cdnNotPt_V4:
  case Hexagon::STriw_GP_cdnPt_V4:
  case Hexagon::STriw_GP_cdnNotPt_V4:
  case Hexagon::STd_GP_cdnPt_V4:
  case Hexagon::STd_GP_cdnNotPt_V4:
  case Hexagon::STb_GP_cdnPt_V4:
  case Hexagon::STb_GP_cdnNotPt_V4:
  case Hexagon::STh_GP_cdnPt_V4:
  case Hexagon::STh_GP_cdnNotPt_V4:
  case Hexagon::STw_GP_cdnPt_V4:
  case Hexagon::STw_GP_cdnNotPt_V4:
    return true;
  }
  return false;
}

static MachineOperand& GetPostIncrementOperand(MachineInstr *MI,
                                               const HexagonInstrInfo *QII) {
  assert(QII->isPostIncrement(MI) && "Not a post increment operation.");
#ifndef NDEBUG
  // Post Increment means duplicates. Use dense map to find duplicates in the
  // list. Caution: Densemap initializes with the minimum of 64 buckets,
  // whereas there are at most 5 operands in the post increment.
  DenseMap<unsigned,  unsigned> DefRegsSet;
  for(unsigned opNum = 0; opNum < MI->getNumOperands(); opNum++)
    if (MI->getOperand(opNum).isReg() &&
        MI->getOperand(opNum).isDef()) {
      DefRegsSet[MI->getOperand(opNum).getReg()] = 1;
    }

  for(unsigned opNum = 0; opNum < MI->getNumOperands(); opNum++)
    if (MI->getOperand(opNum).isReg() &&
        MI->getOperand(opNum).isUse()) {
      if (DefRegsSet[MI->getOperand(opNum).getReg()]) {
        return MI->getOperand(opNum);
      }
    }
#else
  if (MI->getDesc().mayLoad()) {
    // The 2nd operand is always the post increment operand in load.
    assert(MI->getOperand(1).isReg() &&
                "Post increment operand has be to a register.");
    return (MI->getOperand(1));
  }
  if (MI->getDesc().mayStore()) {
    // The 1st operand is always the post increment operand in store.
    assert(MI->getOperand(0).isReg() &&
                "Post increment operand has be to a register.");
    return (MI->getOperand(0));
  }
#endif
  // we should never come here.
  llvm_unreachable("mayLoad or mayStore not set for Post Increment operation");
}

// get the value being stored
static MachineOperand& GetStoreValueOperand(MachineInstr *MI) {
  // value being stored is always the last operand.
  return (MI->getOperand(MI->getNumOperands()-1));
}

// can be new value store?
// Following restrictions are to be respected in convert a store into
// a new value store.
// 1. If an instruction uses auto-increment, its address register cannot
//    be a new-value register. Arch Spec 5.4.2.1
// 2. If an instruction uses absolute-set addressing mode,
//    its address register cannot be a new-value register.
//    Arch Spec 5.4.2.1.TODO: This is not enabled as
//    as absolute-set address mode patters are not implemented.
// 3. If an instruction produces a 64-bit result, its registers cannot be used
//    as new-value registers. Arch Spec 5.4.2.2.
// 4. If the instruction that sets a new-value register is conditional, then
//    the instruction that uses the new-value register must also be conditional,
//    and both must always have their predicates evaluate identically.
//    Arch Spec 5.4.2.3.
// 5. There is an implied restriction of a packet can not have another store,
//    if there is a  new value store in the packet. Corollary, if there is
//    already a store in a packet, there can not be a new value store.
//    Arch Spec: 3.4.4.2
bool HexagonPacketizerList::CanPromoteToNewValueStore( MachineInstr *MI,
                MachineInstr *PacketMI, unsigned DepReg,
                std::map <MachineInstr*, SUnit*> MIToSUnit)
{
  // Make sure we are looking at the store
  if (!IsNewifyStore(MI))
    return false;

  // Make sure there is dependency and can be new value'ed
  if (GetStoreValueOperand(MI).isReg() &&
      GetStoreValueOperand(MI).getReg() != DepReg)
    return false;

  const HexagonRegisterInfo* QRI = 
                            (const HexagonRegisterInfo *) TM.getRegisterInfo();
  const MCInstrDesc& MCID = PacketMI->getDesc();
  // first operand is always the result

  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  const TargetRegisterClass* PacketRC = QII->getRegClass(MCID, 0, QRI, MF);

  // if there is already an store in the packet, no can do new value store
  // Arch Spec 3.4.4.2.
  for (std::vector<MachineInstr*>::iterator VI = CurrentPacketMIs.begin(),
         VE = CurrentPacketMIs.end();
       (VI != VE); ++VI) {
    SUnit* PacketSU = MIToSUnit[*VI];
    if (PacketSU->getInstr()->getDesc().mayStore() ||
        // if we have mayStore = 1 set on ALLOCFRAME and DEALLOCFRAME,
        // then we don't need this
        PacketSU->getInstr()->getOpcode() == Hexagon::ALLOCFRAME ||
        PacketSU->getInstr()->getOpcode() == Hexagon::DEALLOCFRAME)
      return false;
  }

  if (PacketRC == &Hexagon::DoubleRegsRegClass) {
    // new value store constraint: double regs can not feed into new value store
    // arch spec section: 5.4.2.2
    return false;
  }

  // Make sure it's NOT the post increment register that we are going to
  // new value.
  if (QII->isPostIncrement(MI) &&
      MI->getDesc().mayStore() &&
      GetPostIncrementOperand(MI, QII).getReg() == DepReg) {
    return false;
  }

  if (QII->isPostIncrement(PacketMI) &&
      PacketMI->getDesc().mayLoad() &&
      GetPostIncrementOperand(PacketMI, QII).getReg() == DepReg) {
    // if source is post_inc, or absolute-set addressing,
    // it can not feed into new value store
    //  r3 = memw(r2++#4)
    //  memw(r30 + #-1404) = r2.new -> can not be new value store
    // arch spec section: 5.4.2.1
    return false;
  }

  // If the source that feeds the store is predicated, new value store must
  // also be also predicated.
  if (QII->isPredicated(PacketMI)) {
    if (!QII->isPredicated(MI))
      return false;

    // Check to make sure that they both will have their predicates
    // evaluate identically
    unsigned predRegNumSrc = 0;
    unsigned predRegNumDst = 0;
    const TargetRegisterClass* predRegClass = NULL;

    // Get predicate register used in the source instruction
    for(unsigned opNum = 0; opNum < PacketMI->getNumOperands(); opNum++) {
      if ( PacketMI->getOperand(opNum).isReg())
      predRegNumSrc = PacketMI->getOperand(opNum).getReg();
      predRegClass = QRI->getMinimalPhysRegClass(predRegNumSrc);
      if (predRegClass == &Hexagon::PredRegsRegClass) {
        break;
      }
    }
    assert ((predRegClass == &Hexagon::PredRegsRegClass ) &&
        ("predicate register not found in a predicated PacketMI instruction"));

    // Get predicate register used in new-value store instruction
    for(unsigned opNum = 0; opNum < MI->getNumOperands(); opNum++) {
      if ( MI->getOperand(opNum).isReg())
      predRegNumDst = MI->getOperand(opNum).getReg();
      predRegClass = QRI->getMinimalPhysRegClass(predRegNumDst);
      if (predRegClass == &Hexagon::PredRegsRegClass) {
        break;
      }
    }
    assert ((predRegClass == &Hexagon::PredRegsRegClass ) &&
            ("predicate register not found in a predicated MI instruction"));

    // New-value register producer and user (store) need to satisfy these
    // constraints:
    // 1) Both instructions should be predicated on the same register.
    // 2) If producer of the new-value register is .new predicated then store
    // should also be .new predicated and if producer is not .new predicated
    // then store should not be .new predicated.
    // 3) Both new-value register producer and user should have same predicate
    // sense, i.e, either both should be negated or both should be none negated.

    if (( predRegNumDst != predRegNumSrc) ||
          isDotNewInst(PacketMI) != isDotNewInst(MI)  ||
          GetPredicateSense(MI, QII) != GetPredicateSense(PacketMI, QII)) {
      return false;
    }
  }

  // Make sure that other than the new-value register no other store instruction
  // register has been modified in the same packet. Predicate registers can be
  // modified by they should not be modified between the producer and the store
  // instruction as it will make them both conditional on different values.
  // We already know this to be true for all the instructions before and
  // including PacketMI. Howerver, we need to perform the check for the
  // remaining instructions in the packet.

  std::vector<MachineInstr*>::iterator VI;
  std::vector<MachineInstr*>::iterator VE;
  unsigned StartCheck = 0;

  for (VI=CurrentPacketMIs.begin(), VE = CurrentPacketMIs.end();
      (VI != VE); ++VI) {
    SUnit* TempSU = MIToSUnit[*VI];
    MachineInstr* TempMI = TempSU->getInstr();

    // Following condition is true for all the instructions until PacketMI is
    // reached (StartCheck is set to 0 before the for loop).
    // StartCheck flag is 1 for all the instructions after PacketMI.
    if (TempMI != PacketMI && !StartCheck) // start processing only after
      continue;                            // encountering PacketMI

    StartCheck = 1;
    if (TempMI == PacketMI) // We don't want to check PacketMI for dependence
      continue;

    for(unsigned opNum = 0; opNum < MI->getNumOperands(); opNum++) {
      if (MI->getOperand(opNum).isReg() &&
          TempSU->getInstr()->modifiesRegister(MI->getOperand(opNum).getReg(),
                                               QRI))
        return false;
    }
  }

  // Make sure that for non POST_INC stores:
  // 1. The only use of reg is DepReg and no other registers.
  //    This handles V4 base+index registers.
  //    The following store can not be dot new.
  //    Eg.   r0 = add(r0, #3)a
  //          memw(r1+r0<<#2) = r0
  if (!QII->isPostIncrement(MI) &&
      GetStoreValueOperand(MI).isReg() &&
      GetStoreValueOperand(MI).getReg() == DepReg) {
    for(unsigned opNum = 0; opNum < MI->getNumOperands()-1; opNum++) {
      if (MI->getOperand(opNum).isReg() &&
          MI->getOperand(opNum).getReg() == DepReg) {
        return false;
      }
    }
    // 2. If data definition is because of implicit definition of the register,
    //    do not newify the store. Eg.
    //    %R9<def> = ZXTH %R12, %D6<imp-use>, %R12<imp-def>
    //    STrih_indexed %R8, 2, %R12<kill>; mem:ST2[%scevgep343]
    for(unsigned opNum = 0; opNum < PacketMI->getNumOperands(); opNum++) {
      if (PacketMI->getOperand(opNum).isReg() &&
          PacketMI->getOperand(opNum).getReg() == DepReg &&
          PacketMI->getOperand(opNum).isDef() &&
          PacketMI->getOperand(opNum).isImplicit()) {
        return false;
      }
    }
  }

  // Can be dot new store.
  return true;
}

// can this MI to promoted to either
// new value store or new value jump
bool HexagonPacketizerList::CanPromoteToNewValue( MachineInstr *MI,
                SUnit *PacketSU, unsigned DepReg,
                std::map <MachineInstr*, SUnit*> MIToSUnit,
                MachineBasicBlock::iterator &MII)
{

  const HexagonRegisterInfo* QRI =
                            (const HexagonRegisterInfo *) TM.getRegisterInfo();
  if (!QRI->Subtarget.hasV4TOps() ||
      !IsNewifyStore(MI))
    return false;

  MachineInstr *PacketMI = PacketSU->getInstr();

  // Check to see the store can be new value'ed.
  if (CanPromoteToNewValueStore(MI, PacketMI, DepReg, MIToSUnit))
    return true;

  // Check to see the compare/jump can be new value'ed.
  // This is done as a pass on its own. Don't need to check it here.
  return false;
}

// Check to see if an instruction can be dot new
// There are three kinds.
// 1. dot new on predicate - V2/V3/V4
// 2. dot new on stores NV/ST - V4
// 3. dot new on jump NV/J - V4 -- This is generated in a pass.
bool HexagonPacketizerList::CanPromoteToDotNew( MachineInstr *MI,
                              SUnit *PacketSU, unsigned DepReg,
                              std::map <MachineInstr*, SUnit*> MIToSUnit,
                              MachineBasicBlock::iterator &MII,
                              const TargetRegisterClass* RC )
{
  // already a dot new instruction
  if (isDotNewInst(MI) && !IsNewifyStore(MI))
    return false;

  if (!isNewifiable(MI))
    return false;

  // predicate .new
  if (RC == &Hexagon::PredRegsRegClass && isCondInst(MI))
      return true;
  else if (RC != &Hexagon::PredRegsRegClass &&
      !IsNewifyStore(MI)) // MI is not a new-value store
    return false;
  else {
    // Create a dot new machine instruction to see if resources can be
    // allocated. If not, bail out now.
    const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
    int NewOpcode = GetDotNewOp(MI->getOpcode());
    const MCInstrDesc &desc = QII->get(NewOpcode);
    DebugLoc dl;
    MachineInstr *NewMI =
                    MI->getParent()->getParent()->CreateMachineInstr(desc, dl);
    bool ResourcesAvailable = ResourceTracker->canReserveResources(NewMI);
    MI->getParent()->getParent()->DeleteMachineInstr(NewMI);

    if (!ResourcesAvailable)
      return false;

    // new value store only
    // new new value jump generated as a passes
    if (!CanPromoteToNewValue(MI, PacketSU, DepReg, MIToSUnit, MII)) {
      return false;
    }
  }
  return true;
}

// Go through the packet instructions and search for anti dependency
// between them and DepReg from MI
// Consider this case:
// Trying to add
// a) %R1<def> = TFRI_cdNotPt %P3, 2
// to this packet:
// {
//   b) %P0<def> = OR_pp %P3<kill>, %P0<kill>
//   c) %P3<def> = TFR_PdRs %R23
//   d) %R1<def> = TFRI_cdnPt %P3, 4
//  }
// The P3 from a) and d) will be complements after
// a)'s P3 is converted to .new form
// Anti Dep between c) and b) is irrelevant for this case
bool HexagonPacketizerList::RestrictingDepExistInPacket (MachineInstr* MI,
      unsigned DepReg,
      std::map <MachineInstr*, SUnit*> MIToSUnit) {

  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  SUnit* PacketSUDep = MIToSUnit[MI];

  for (std::vector<MachineInstr*>::iterator VIN = CurrentPacketMIs.begin(),
       VEN = CurrentPacketMIs.end(); (VIN != VEN); ++VIN) {

    // We only care for dependencies to predicated instructions
    if(!QII->isPredicated(*VIN)) continue;

    // Scheduling Unit for current insn in the packet
    SUnit* PacketSU = MIToSUnit[*VIN];

    // Look at dependencies between current members of the packet
    // and predicate defining instruction MI.
    // Make sure that dependency is on the exact register
    // we care about.
    if (PacketSU->isSucc(PacketSUDep)) {
      for (unsigned i = 0; i < PacketSU->Succs.size(); ++i) {
        if ((PacketSU->Succs[i].getSUnit() == PacketSUDep) &&
            (PacketSU->Succs[i].getKind() == SDep::Anti) &&
            (PacketSU->Succs[i].getReg() == DepReg)) {
          return true;
        }
      }
    }
  }

  return false;
}


// Given two predicated instructions, this function detects whether
// the predicates are complements
bool HexagonPacketizerList::ArePredicatesComplements (MachineInstr* MI1,
     MachineInstr* MI2, std::map <MachineInstr*, SUnit*> MIToSUnit) {

  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;
  // Currently can only reason about conditional transfers
  if (!QII->isConditionalTransfer(MI1) || !QII->isConditionalTransfer(MI2)) {
    return false;
  }

  // Scheduling unit for candidate
  SUnit* SU = MIToSUnit[MI1];

  // One corner case deals with the following scenario:
  // Trying to add
  // a) %R24<def> = TFR_cPt %P0, %R25
  // to this packet:
  //
  // {
  //   b) %R25<def> = TFR_cNotPt %P0, %R24
  //   c) %P0<def> = CMPEQri %R26, 1
  // }
  //
  // On general check a) and b) are complements, but
  // presence of c) will convert a) to .new form, and
  // then it is not a complement
  // We attempt to detect it by analyzing  existing
  // dependencies in the packet

  // Analyze relationships between all existing members of the packet.
  // Look for Anti dependecy on the same predicate reg
  // as used in the candidate
  for (std::vector<MachineInstr*>::iterator VIN = CurrentPacketMIs.begin(),
       VEN = CurrentPacketMIs.end(); (VIN != VEN); ++VIN) {

    // Scheduling Unit for current insn in the packet
    SUnit* PacketSU = MIToSUnit[*VIN];

    // If this instruction in the packet is succeeded by the candidate...
    if (PacketSU->isSucc(SU)) {
      for (unsigned i = 0; i < PacketSU->Succs.size(); ++i) {
        // The corner case exist when there is true data
        // dependency between candidate and one of current
        // packet members, this dep is on predicate reg, and
        // there already exist anti dep on the same pred in
        // the packet.
        if (PacketSU->Succs[i].getSUnit() == SU &&
            Hexagon::PredRegsRegClass.contains(
              PacketSU->Succs[i].getReg()) &&
            PacketSU->Succs[i].getKind() == SDep::Data &&
            // Here I know that *VIN is predicate setting instruction
            // with true data dep to candidate on the register
            // we care about - c) in the above example.
            // Now I need to see if there is an anti dependency
            // from c) to any other instruction in the
            // same packet on the pred reg of interest
            RestrictingDepExistInPacket(*VIN,PacketSU->Succs[i].getReg(),
                                        MIToSUnit)) {
           return false;
        }
      }
    }
  }

  // If the above case does not apply, check regular
  // complement condition.
  // Check that the predicate register is the same and
  // that the predicate sense is different
  // We also need to differentiate .old vs. .new:
  // !p0 is not complimentary to p0.new
  return ((MI1->getOperand(1).getReg() == MI2->getOperand(1).getReg()) &&
          (GetPredicateSense(MI1, QII) != GetPredicateSense(MI2, QII)) &&
          (isDotNewInst(MI1) == isDotNewInst(MI2)));
}

// initPacketizerState - Initialize packetizer flags
void HexagonPacketizerList::initPacketizerState() {

  Dependence = false;
  PromotedToDotNew = false;
  GlueToNewValueJump = false;
  GlueAllocframeStore = false;
  FoundSequentialDependence = false;

  return;
}

// ignorePseudoInstruction - Ignore bundling of pseudo instructions.
bool HexagonPacketizerList::ignorePseudoInstruction(MachineInstr *MI,
                                                    MachineBasicBlock *MBB) {
  if (MI->isDebugValue())
    return true;

  // We must print out inline assembly
  if (MI->isInlineAsm())
    return false;

  // We check if MI has any functional units mapped to it.
  // If it doesn't, we ignore the instruction.
  const MCInstrDesc& TID = MI->getDesc();
  unsigned SchedClass = TID.getSchedClass();
  const InstrStage* IS =
                    ResourceTracker->getInstrItins()->beginStage(SchedClass);
  unsigned FuncUnits = IS->getUnits();
  return !FuncUnits;
}

// isSoloInstruction: - Returns true for instructions that must be
// scheduled in their own packet.
bool HexagonPacketizerList::isSoloInstruction(MachineInstr *MI) {

  if (MI->isInlineAsm())
    return true;

  if (MI->isEHLabel())
    return true;

  // From Hexagon V4 Programmer's Reference Manual 3.4.4 Grouping constraints:
  // trap, pause, barrier, icinva, isync, and syncht are solo instructions.
  // They must not be grouped with other instructions in a packet.
  if (IsSchedBarrier(MI))
    return true;

  return false;
}

// isLegalToPacketizeTogether:
// SUI is the current instruction that is out side of the current packet.
// SUJ is the current instruction inside the current packet against which that
// SUI will be packetized.
bool HexagonPacketizerList::isLegalToPacketizeTogether(SUnit *SUI, SUnit *SUJ) {
  MachineInstr *I = SUI->getInstr();
  MachineInstr *J = SUJ->getInstr();
  assert(I && J && "Unable to packetize null instruction!");

  const MCInstrDesc &MCIDI = I->getDesc();
  const MCInstrDesc &MCIDJ = J->getDesc();

  MachineBasicBlock::iterator II = I;

  const unsigned FrameSize = MF.getFrameInfo()->getStackSize();
  const HexagonRegisterInfo* QRI =
                      (const HexagonRegisterInfo *) TM.getRegisterInfo();
  const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;

  // Inline asm cannot go in the packet.
  if (I->getOpcode() == Hexagon::INLINEASM)
    llvm_unreachable("Should not meet inline asm here!");

  if (isSoloInstruction(I))
    llvm_unreachable("Should not meet solo instr here!");

  // A save callee-save register function call can only be in a packet
  // with instructions that don't write to the callee-save registers.
  if ((QII->isSaveCalleeSavedRegsCall(I) &&
       DoesModifyCalleeSavedReg(J, QRI)) ||
      (QII->isSaveCalleeSavedRegsCall(J) &&
       DoesModifyCalleeSavedReg(I, QRI))) {
    Dependence = true;
    return false;
  }

  // Two control flow instructions cannot go in the same packet.
  if (IsControlFlow(I) && IsControlFlow(J)) {
    Dependence = true;
    return false;
  }

  // A LoopN instruction cannot appear in the same packet as a jump or call.
  if (IsLoopN(I) && (   IsDirectJump(J)
                     || MCIDJ.isCall()
                     || QII->isDeallocRet(J))) {
    Dependence = true;
    return false;
  }
  if (IsLoopN(J) && (   IsDirectJump(I)
                     || MCIDI.isCall()
                     || QII->isDeallocRet(I))) {
    Dependence = true;
    return false;
  }

  // dealloc_return cannot appear in the same packet as a conditional or
  // unconditional jump.
  if (QII->isDeallocRet(I) && (   MCIDJ.isBranch()
                               || MCIDJ.isCall()
                               || MCIDJ.isBarrier())) {
    Dependence = true;
    return false;
  }


  // V4 allows dual store. But does not allow second store, if the
  // first store is not in SLOT0. New value store, new value jump,
  // dealloc_return and memop always take SLOT0.
  // Arch spec 3.4.4.2
  if (QRI->Subtarget.hasV4TOps()) {

    if (MCIDI.mayStore() && MCIDJ.mayStore() && isNewValueInst(J)) {
      Dependence = true;
      return false;
    }

    if (   (QII->isMemOp(J) && MCIDI.mayStore())
        || (MCIDJ.mayStore() && QII->isMemOp(I))
        || (QII->isMemOp(J) && QII->isMemOp(I))) {
      Dependence = true;
      return false;
    }

    //if dealloc_return
    if (MCIDJ.mayStore() && QII->isDeallocRet(I)){
      Dependence = true;
      return false;
    }

    // If an instruction feeds new value jump, glue it.
    MachineBasicBlock::iterator NextMII = I;
    ++NextMII;
    MachineInstr *NextMI = NextMII;

    if (QII->isNewValueJump(NextMI)) {

      bool secondRegMatch = false;
      bool maintainNewValueJump = false;

      if (NextMI->getOperand(1).isReg() &&
          I->getOperand(0).getReg() == NextMI->getOperand(1).getReg()) {
        secondRegMatch = true;
        maintainNewValueJump = true;
      }

      if (!secondRegMatch &&
           I->getOperand(0).getReg() == NextMI->getOperand(0).getReg()) {
        maintainNewValueJump = true;
      }

      for (std::vector<MachineInstr*>::iterator
            VI = CurrentPacketMIs.begin(),
             VE = CurrentPacketMIs.end();
           (VI != VE && maintainNewValueJump); ++VI) {
        SUnit* PacketSU = MIToSUnit[*VI];

        // NVJ can not be part of the dual jump - Arch Spec: section 7.8
        if (PacketSU->getInstr()->getDesc().isCall()) {
          Dependence = true;
          break;
        }
        // Validate
        // 1. Packet does not have a store in it.
        // 2. If the first operand of the nvj is newified, and the second
        //    operand is also a reg, it (second reg) is not defined in
        //    the same packet.
        // 3. If the second operand of the nvj is newified, (which means
        //    first operand is also a reg), first reg is not defined in
        //    the same packet.
        if (PacketSU->getInstr()->getDesc().mayStore()               ||
            PacketSU->getInstr()->getOpcode() == Hexagon::ALLOCFRAME ||
            // Check #2.
            (!secondRegMatch && NextMI->getOperand(1).isReg() &&
             PacketSU->getInstr()->modifiesRegister(
                               NextMI->getOperand(1).getReg(), QRI)) ||
            // Check #3.
            (secondRegMatch &&
             PacketSU->getInstr()->modifiesRegister(
                               NextMI->getOperand(0).getReg(), QRI))) {
          Dependence = true;
          break;
        }
      }
      if (!Dependence)
        GlueToNewValueJump = true;
      else
        return false;
    }
  }

  if (SUJ->isSucc(SUI)) {
    for (unsigned i = 0;
         (i < SUJ->Succs.size()) && !FoundSequentialDependence;
         ++i) {

      if (SUJ->Succs[i].getSUnit() != SUI) {
        continue;
      }

      SDep::Kind DepType = SUJ->Succs[i].getKind();

      // For direct calls:
      // Ignore register dependences for call instructions for
      // packetization purposes except for those due to r31 and
      // predicate registers.
      //
      // For indirect calls:
      // Same as direct calls + check for true dependences to the register
      // used in the indirect call.
      //
      // We completely ignore Order dependences for call instructions
      //
      // For returns:
      // Ignore register dependences for return instructions like jumpr,
      // dealloc return unless we have dependencies on the explicit uses
      // of the registers used by jumpr (like r31) or dealloc return
      // (like r29 or r30).
      //
      // TODO: Currently, jumpr is handling only return of r31. So, the
      // following logic (specificaly IsCallDependent) is working fine.
      // We need to enable jumpr for register other than r31 and then,
      // we need to rework the last part, where it handles indirect call
      // of that (IsCallDependent) function. Bug 6216 is opened for this.
      //
      unsigned DepReg = 0;
      const TargetRegisterClass* RC = NULL;
      if (DepType == SDep::Data) {
        DepReg = SUJ->Succs[i].getReg();
        RC = QRI->getMinimalPhysRegClass(DepReg);
      }
      if ((MCIDI.isCall() || MCIDI.isReturn()) &&
          (!IsRegDependence(DepType) ||
            !IsCallDependent(I, DepType, SUJ->Succs[i].getReg()))) {
        /* do nothing */
      }

      // For instructions that can be promoted to dot-new, try to promote.
      else if ((DepType == SDep::Data) &&
               CanPromoteToDotNew(I, SUJ, DepReg, MIToSUnit, II, RC) &&
               PromoteToDotNew(I, DepType, II, RC)) {
        PromotedToDotNew = true;
        /* do nothing */
      }

      else if ((DepType == SDep::Data) &&
               (QII->isNewValueJump(I))) {
        /* do nothing */
      }

      // For predicated instructions, if the predicates are complements
      // then there can be no dependence.
      else if (QII->isPredicated(I) &&
               QII->isPredicated(J) &&
          ArePredicatesComplements(I, J, MIToSUnit)) {
        /* do nothing */

      }
      else if (IsDirectJump(I) &&
               !MCIDJ.isBranch() &&
               !MCIDJ.isCall() &&
               (DepType == SDep::Order)) {
        // Ignore Order dependences between unconditional direct branches
        // and non-control-flow instructions
        /* do nothing */
      }
      else if (MCIDI.isConditionalBranch() && (DepType != SDep::Data) &&
               (DepType != SDep::Output)) {
        // Ignore all dependences for jumps except for true and output
        // dependences
        /* do nothing */
      }

      // Ignore output dependences due to superregs. We can
      // write to two different subregisters of R1:0 for instance
      // in the same cycle
      //

      //
      // Let the
      // If neither I nor J defines DepReg, then this is a
      // superfluous output dependence. The dependence must be of the
      // form:
      //  R0 = ...
      //  R1 = ...
      // and there is an output dependence between the two instructions
      // with
      // DepReg = D0
      // We want to ignore these dependences.
      // Ideally, the dependence constructor should annotate such
      // dependences. We can then avoid this relatively expensive check.
      //
      else if (DepType == SDep::Output) {
        // DepReg is the register that's responsible for the dependence.
        unsigned DepReg = SUJ->Succs[i].getReg();

        // Check if I and J really defines DepReg.
        if (I->definesRegister(DepReg) ||
            J->definesRegister(DepReg)) {
          FoundSequentialDependence = true;
          break;
        }
      }

      // We ignore Order dependences for
      // 1. Two loads unless they are volatile.
      // 2. Two stores in V4 unless they are volatile.
      else if ((DepType == SDep::Order) &&
               !I->hasOrderedMemoryRef() &&
               !J->hasOrderedMemoryRef()) {
        if (QRI->Subtarget.hasV4TOps() &&
            // hexagonv4 allows dual store.
            MCIDI.mayStore() && MCIDJ.mayStore()) {
          /* do nothing */
        }
        // store followed by store-- not OK on V2
        // store followed by load -- not OK on all (OK if addresses
        // are not aliased)
        // load followed by store -- OK on all
        // load followed by load  -- OK on all
        else if ( !MCIDJ.mayStore()) {
          /* do nothing */
        }
        else {
          FoundSequentialDependence = true;
          break;
        }
      }

      // For V4, special case ALLOCFRAME. Even though there is dependency
      // between ALLOCAFRAME and subsequent store, allow it to be
      // packetized in a same packet. This implies that the store is using
      // caller's SP. Hense, offset needs to be updated accordingly.
      else if (DepType == SDep::Data
               && QRI->Subtarget.hasV4TOps()
               && J->getOpcode() == Hexagon::ALLOCFRAME
               && (I->getOpcode() == Hexagon::STrid
                   || I->getOpcode() == Hexagon::STriw
                   || I->getOpcode() == Hexagon::STrib)
               && I->getOperand(0).getReg() == QRI->getStackRegister()
               && QII->isValidOffset(I->getOpcode(),
                                     I->getOperand(1).getImm() -
                                     (FrameSize + HEXAGON_LRFP_SIZE)))
      {
        GlueAllocframeStore = true;
        // Since this store is to be glued with allocframe in the same
        // packet, it will use SP of the previous stack frame, i.e
        // caller's SP. Therefore, we need to recalculate offset according
        // to this change.
        I->getOperand(1).setImm(I->getOperand(1).getImm() -
                                        (FrameSize + HEXAGON_LRFP_SIZE));
      }

      //
      // Skip over anti-dependences. Two instructions that are
      // anti-dependent can share a packet
      //
      else if (DepType != SDep::Anti) {
        FoundSequentialDependence = true;
        break;
      }
    }

    if (FoundSequentialDependence) {
      Dependence = true;
      return false;
    }
  }

  return true;
}

// isLegalToPruneDependencies
bool HexagonPacketizerList::isLegalToPruneDependencies(SUnit *SUI, SUnit *SUJ) {
  MachineInstr *I = SUI->getInstr();
  assert(I && SUJ->getInstr() && "Unable to packetize null instruction!");

  const unsigned FrameSize = MF.getFrameInfo()->getStackSize();

  if (Dependence) {

    // Check if the instruction was promoted to a dot-new. If so, demote it
    // back into a dot-old.
    if (PromotedToDotNew) {
      DemoteToDotOld(I);
    }

    // Check if the instruction (must be a store) was glued with an Allocframe
    // instruction. If so, restore its offset to its original value, i.e. use
    // curent SP instead of caller's SP.
    if (GlueAllocframeStore) {
      I->getOperand(1).setImm(I->getOperand(1).getImm() +
                                             FrameSize + HEXAGON_LRFP_SIZE);
    }

    return false;
  }
  return true;
}

MachineBasicBlock::iterator
HexagonPacketizerList::addToPacket(MachineInstr *MI) {

    MachineBasicBlock::iterator MII = MI;
    MachineBasicBlock *MBB = MI->getParent();

    const HexagonInstrInfo *QII = (const HexagonInstrInfo *) TII;

    if (GlueToNewValueJump) {

      ++MII;
      MachineInstr *nvjMI = MII;
      assert(ResourceTracker->canReserveResources(MI));
      ResourceTracker->reserveResources(MI);
      if (QII->isExtended(MI) &&
          !tryAllocateResourcesForConstExt(MI)) {
        endPacket(MBB, MI);
        ResourceTracker->reserveResources(MI);
        assert(canReserveResourcesForConstExt(MI) &&
               "Ensure that there is a slot");
        reserveResourcesForConstExt(MI);
        // Reserve resources for new value jump constant extender.
        assert(canReserveResourcesForConstExt(MI) &&
               "Ensure that there is a slot");
        reserveResourcesForConstExt(nvjMI);
        assert(ResourceTracker->canReserveResources(nvjMI) &&
               "Ensure that there is a slot");

      } else if (   // Extended instruction takes two slots in the packet.
        // Try reserve and allocate 4-byte in the current packet first.
        (QII->isExtended(nvjMI)
            && (!tryAllocateResourcesForConstExt(nvjMI)
                || !ResourceTracker->canReserveResources(nvjMI)))
        || // For non-extended instruction, no need to allocate extra 4 bytes.
        (!QII->isExtended(nvjMI) && 
              !ResourceTracker->canReserveResources(nvjMI)))
      {
        endPacket(MBB, MI);
        // A new and empty packet starts.
        // We are sure that the resources requirements can be satisfied.
        // Therefore, do not need to call "canReserveResources" anymore.
        ResourceTracker->reserveResources(MI);
        if (QII->isExtended(nvjMI))
          reserveResourcesForConstExt(nvjMI);
      }
      // Here, we are sure that "reserveResources" would succeed.
      ResourceTracker->reserveResources(nvjMI);
      CurrentPacketMIs.push_back(MI);
      CurrentPacketMIs.push_back(nvjMI);
    } else {
      if (   QII->isExtended(MI)
          && (   !tryAllocateResourcesForConstExt(MI)
              || !ResourceTracker->canReserveResources(MI)))
      {
        endPacket(MBB, MI);
        // Check if the instruction was promoted to a dot-new. If so, demote it
        // back into a dot-old
        if (PromotedToDotNew) {
          DemoteToDotOld(MI);
        }
        reserveResourcesForConstExt(MI);
      }
      // In case that "MI" is not an extended insn,
      // the resource availability has already been checked.
      ResourceTracker->reserveResources(MI);
      CurrentPacketMIs.push_back(MI);
    }
    return MII;
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createHexagonPacketizer() {
  return new HexagonPacketizer();
}

