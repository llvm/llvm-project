//=- llvm/CodeGen/DFAPacketizer.cpp - DFA Packetizer for VLIW -*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This class implements a deterministic finite automaton (DFA) based
// packetizing mechanism for VLIW architectures. It provides APIs to
// determine whether there exists a legal mapping of instructions to
// functional unit assignments in a packet. The DFA is auto-generated from
// the target's Schedule.td file.
//
// A DFA consists of 3 major elements: states, inputs, and transitions. For
// the packetizing mechanism, the input is the set of instruction classes for
// a target. The state models all possible combinations of functional unit
// consumption for a given set of instructions in a packet. A transition
// models the addition of an instruction to a packet. In the DFA constructed
// by this class, if an instruction can be added to a packet, then a valid
// transition exists from the corresponding state. Invalid transitions
// indicate that the instruction cannot be added to the current packet.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Target/TargetInstrInfo.h"
using namespace llvm;

// --------------------------------------------------------------------
// Definitions shared between DFAPacketizer.cpp and DFAPacketizerEmitter.cpp

namespace {
  DFAInput addDFAFuncUnits(DFAInput Inp, unsigned FuncUnits) {
    return (Inp << DFA_MAX_RESOURCES) | FuncUnits;
  }

  /// Return the DFAInput for an instruction class input vector.
  /// This function is used in both DFAPacketizer.cpp and in
  /// DFAPacketizerEmitter.cpp.
  DFAInput getDFAInsnInput(const std::vector<unsigned> &InsnClass) {
    DFAInput InsnInput = 0;
    assert ((InsnClass.size() <= DFA_MAX_RESTERMS) &&
            "Exceeded maximum number of DFA terms");
    for (auto U : InsnClass)
      InsnInput = addDFAFuncUnits(InsnInput, U);
    return InsnInput;
  }
}
// --------------------------------------------------------------------

DFAPacketizer::DFAPacketizer(const InstrItineraryData *I,
                             const DFAStateInput (*SIT)[2],
                             const unsigned *SET):
  InstrItins(I), CurrentState(0), DFAStateInputTable(SIT),
  DFAStateEntryTable(SET) {
  // Make sure DFA types are large enough for the number of terms & resources.
  assert((DFA_MAX_RESTERMS * DFA_MAX_RESOURCES) <= (8 * sizeof(DFAInput))
        && "(DFA_MAX_RESTERMS * DFA_MAX_RESOURCES) too big for DFAInput");
  assert((DFA_MAX_RESTERMS * DFA_MAX_RESOURCES) <= (8 * sizeof(DFAStateInput))
        && "(DFA_MAX_RESTERMS * DFA_MAX_RESOURCES) too big for DFAStateInput");
}


//
// ReadTable - Read the DFA transition table and update CachedTable.
//
// Format of the transition tables:
// DFAStateInputTable[][2] = pairs of <Input, Transition> for all valid
//                           transitions
// DFAStateEntryTable[i] = Index of the first entry in DFAStateInputTable
//                         for the ith state
//
void DFAPacketizer::ReadTable(unsigned int state) {
  unsigned ThisState = DFAStateEntryTable[state];
  unsigned NextStateInTable = DFAStateEntryTable[state+1];
  // Early exit in case CachedTable has already contains this
  // state's transitions.
  if (CachedTable.count(UnsignPair(state,
                                   DFAStateInputTable[ThisState][0])))
    return;

  for (unsigned i = ThisState; i < NextStateInTable; i++)
    CachedTable[UnsignPair(state, DFAStateInputTable[i][0])] =
      DFAStateInputTable[i][1];
}

//
// getInsnInput - Return the DFAInput for an instruction class.
//
DFAInput DFAPacketizer::getInsnInput(unsigned InsnClass) {
  // Note: this logic must match that in DFAPacketizerDefs.h for input vectors.
  DFAInput InsnInput = 0;
  unsigned i = 0;
  for (const InstrStage *IS = InstrItins->beginStage(InsnClass),
        *IE = InstrItins->endStage(InsnClass); IS != IE; ++IS, ++i) {
    InsnInput = addDFAFuncUnits(InsnInput, IS->getUnits());
    assert ((i < DFA_MAX_RESTERMS) && "Exceeded maximum number of DFA inputs");
  }
  return InsnInput;
}

// getInsnInput - Return the DFAInput for an instruction class input vector.
DFAInput DFAPacketizer::getInsnInput(const std::vector<unsigned> &InsnClass) {
  return getDFAInsnInput(InsnClass);
}

// canReserveResources - Check if the resources occupied by a MCInstrDesc
// are available in the current state.
bool DFAPacketizer::canReserveResources(const llvm::MCInstrDesc *MID) {
  unsigned InsnClass = MID->getSchedClass();
  DFAInput InsnInput = getInsnInput(InsnClass);
  UnsignPair StateTrans = UnsignPair(CurrentState, InsnInput);
  ReadTable(CurrentState);
  return (CachedTable.count(StateTrans) != 0);
}

// reserveResources - Reserve the resources occupied by a MCInstrDesc and
// change the current state to reflect that change.
void DFAPacketizer::reserveResources(const llvm::MCInstrDesc *MID) {
  unsigned InsnClass = MID->getSchedClass();
  DFAInput InsnInput = getInsnInput(InsnClass);
  UnsignPair StateTrans = UnsignPair(CurrentState, InsnInput);
  ReadTable(CurrentState);
  assert(CachedTable.count(StateTrans) != 0);
  CurrentState = CachedTable[StateTrans];
}


// canReserveResources - Check if the resources occupied by a machine
// instruction are available in the current state.
bool DFAPacketizer::canReserveResources(llvm::MachineInstr *MI) {
  const llvm::MCInstrDesc &MID = MI->getDesc();
  return canReserveResources(&MID);
}

// reserveResources - Reserve the resources occupied by a machine
// instruction and change the current state to reflect that change.
void DFAPacketizer::reserveResources(llvm::MachineInstr *MI) {
  const llvm::MCInstrDesc &MID = MI->getDesc();
  reserveResources(&MID);
}

namespace llvm {
// DefaultVLIWScheduler - This class extends ScheduleDAGInstrs and overrides
// Schedule method to build the dependence graph.
class DefaultVLIWScheduler : public ScheduleDAGInstrs {
private:
  AliasAnalysis *AA;
public:
  DefaultVLIWScheduler(MachineFunction &MF, MachineLoopInfo &MLI,
                       AliasAnalysis *AA);
  // Schedule - Actual scheduling work.
  void schedule() override;
};
}

DefaultVLIWScheduler::DefaultVLIWScheduler(MachineFunction &MF,
                                           MachineLoopInfo &MLI,
                                           AliasAnalysis *AA)
    : ScheduleDAGInstrs(MF, &MLI), AA(AA) {
  CanHandleTerminators = true;
}

void DefaultVLIWScheduler::schedule() {
  // Build the scheduling graph.
  buildSchedGraph(AA);
}

// VLIWPacketizerList Ctor
VLIWPacketizerList::VLIWPacketizerList(MachineFunction &MF,
                                       MachineLoopInfo &MLI, AliasAnalysis *AA)
    : MF(MF), AA(AA) {
  TII = MF.getSubtarget().getInstrInfo();
  ResourceTracker = TII->CreateTargetScheduleState(MF.getSubtarget());
  VLIWScheduler = new DefaultVLIWScheduler(MF, MLI, AA);
}

// VLIWPacketizerList Dtor
VLIWPacketizerList::~VLIWPacketizerList() {
  if (VLIWScheduler)
    delete VLIWScheduler;

  if (ResourceTracker)
    delete ResourceTracker;
}

// endPacket - End the current packet, bundle packet instructions and reset
// DFA state.
void VLIWPacketizerList::endPacket(MachineBasicBlock *MBB,
                                         MachineInstr *MI) {
  if (CurrentPacketMIs.size() > 1) {
    MachineInstr *MIFirst = CurrentPacketMIs.front();
    finalizeBundle(*MBB, MIFirst->getIterator(), MI->getIterator());
  }
  CurrentPacketMIs.clear();
  ResourceTracker->clearResources();
}

// PacketizeMIs - Bundle machine instructions into packets.
void VLIWPacketizerList::PacketizeMIs(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator BeginItr,
                                      MachineBasicBlock::iterator EndItr) {
  assert(VLIWScheduler && "VLIW Scheduler is not initialized!");
  VLIWScheduler->startBlock(MBB);
  VLIWScheduler->enterRegion(MBB, BeginItr, EndItr,
                             std::distance(BeginItr, EndItr));
  VLIWScheduler->schedule();

  // Generate MI -> SU map.
  MIToSUnit.clear();
  for (unsigned i = 0, e = VLIWScheduler->SUnits.size(); i != e; ++i) {
    SUnit *SU = &VLIWScheduler->SUnits[i];
    MIToSUnit[SU->getInstr()] = SU;
  }

  // The main packetizer loop.
  for (; BeginItr != EndItr; ++BeginItr) {
    MachineInstr *MI = BeginItr;

    this->initPacketizerState();

    // End the current packet if needed.
    if (this->isSoloInstruction(MI)) {
      endPacket(MBB, MI);
      continue;
    }

    // Ignore pseudo instructions.
    if (this->ignorePseudoInstruction(MI, MBB))
      continue;

    SUnit *SUI = MIToSUnit[MI];
    assert(SUI && "Missing SUnit Info!");

    // Ask DFA if machine resource is available for MI.
    bool ResourceAvail = ResourceTracker->canReserveResources(MI);
    if (ResourceAvail) {
      // Dependency check for MI with instructions in CurrentPacketMIs.
      for (std::vector<MachineInstr*>::iterator VI = CurrentPacketMIs.begin(),
           VE = CurrentPacketMIs.end(); VI != VE; ++VI) {
        MachineInstr *MJ = *VI;
        SUnit *SUJ = MIToSUnit[MJ];
        assert(SUJ && "Missing SUnit Info!");

        // Is it legal to packetize SUI and SUJ together.
        if (!this->isLegalToPacketizeTogether(SUI, SUJ)) {
          // Allow packetization if dependency can be pruned.
          if (!this->isLegalToPruneDependencies(SUI, SUJ)) {
            // End the packet if dependency cannot be pruned.
            endPacket(MBB, MI);
            break;
          } // !isLegalToPruneDependencies.
        } // !isLegalToPacketizeTogether.
      } // For all instructions in CurrentPacketMIs.
    } else {
      // End the packet if resource is not available.
      endPacket(MBB, MI);
    }

    // Add MI to the current packet.
    BeginItr = this->addToPacket(MI);
  } // For all instructions in BB.

  // End any packet left behind.
  endPacket(MBB, EndItr);
  VLIWScheduler->exitRegion();
  VLIWScheduler->finishBlock();
}
