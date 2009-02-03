//===-- PPCHazardRecognizers.cpp - PowerPC Hazard Recognizer Impls --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements hazard recognizers for scheduling on PowerPC processors.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "PPCHazardRecognizers.h"
#include "PPC.h"
#include "PPCInstrInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// PowerPC 970 Hazard Recognizer
//
// This models the dispatch group formation of the PPC970 processor.  Dispatch
// groups are bundles of up to five instructions that can contain various mixes
// of instructions.  The PPC970 can dispatch a peak of 4 non-branch and one 
// branch instruction per-cycle.
//
// There are a number of restrictions to dispatch group formation: some
// instructions can only be issued in the first slot of a dispatch group, & some
// instructions fill an entire dispatch group.  Additionally, only branches can
// issue in the 5th (last) slot.
//
// Finally, there are a number of "structural" hazards on the PPC970.  These
// conditions cause large performance penalties due to misprediction, recovery,
// and replay logic that has to happen.  These cases include setting a CTR and
// branching through it in the same dispatch group, and storing to an address,
// then loading from the same address within a dispatch group.  To avoid these
// conditions, we insert no-op instructions when appropriate.
//
// FIXME: This is missing some significant cases:
//   1. Modeling of microcoded instructions.
//   2. Handling of serialized operations.
//   3. Handling of the esoteric cases in "Resource-based Instruction Grouping".
//

PPCHazardRecognizer970::PPCHazardRecognizer970(const TargetInstrInfo &tii)
  : TII(tii) {
  EndDispatchGroup();
}

void PPCHazardRecognizer970::EndDispatchGroup() {
  DOUT << "=== Start of dispatch group\n";
  NumIssued = 0;
  
  // Structural hazard info.
  HasCTRSet = false;
  NumStores = 0;
}


PPCII::PPC970_Unit 
PPCHazardRecognizer970::GetInstrType(unsigned Opcode,
                                     bool &isFirst, bool &isSingle,
                                     bool &isCracked,
                                     bool &isLoad, bool &isStore) {
  if ((int)Opcode >= 0) {
    isFirst = isSingle = isCracked = isLoad = isStore = false;
    return PPCII::PPC970_Pseudo;
  }
  Opcode = ~Opcode;
  
  const TargetInstrDesc &TID = TII.get(Opcode);
  
  isLoad  = TID.mayLoad();
  isStore = TID.mayStore();
  
  unsigned TSFlags = TID.TSFlags;
  
  isFirst   = TSFlags & PPCII::PPC970_First;
  isSingle  = TSFlags & PPCII::PPC970_Single;
  isCracked = TSFlags & PPCII::PPC970_Cracked;
  return (PPCII::PPC970_Unit)(TSFlags & PPCII::PPC970_Mask);
}

/// isLoadOfStoredAddress - If we have a load from the previously stored pointer
/// as indicated by StorePtr1/StorePtr2/StoreSize, return true.
bool PPCHazardRecognizer970::
isLoadOfStoredAddress(unsigned LoadSize, SDValue Ptr1, SDValue Ptr2) const {
  for (unsigned i = 0, e = NumStores; i != e; ++i) {
    // Handle exact and commuted addresses.
    if (Ptr1 == StorePtr1[i] && Ptr2 == StorePtr2[i])
      return true;
    if (Ptr2 == StorePtr1[i] && Ptr1 == StorePtr2[i])
      return true;
    
    // Okay, we don't have an exact match, if this is an indexed offset, see if
    // we have overlap (which happens during fp->int conversion for example).
    if (StorePtr2[i] == Ptr2) {
      if (ConstantSDNode *StoreOffset = dyn_cast<ConstantSDNode>(StorePtr1[i]))
        if (ConstantSDNode *LoadOffset = dyn_cast<ConstantSDNode>(Ptr1)) {
          // Okay the base pointers match, so we have [c1+r] vs [c2+r].  Check
          // to see if the load and store actually overlap.
          int StoreOffs = StoreOffset->getZExtValue();
          int LoadOffs  = LoadOffset->getZExtValue();
          if (StoreOffs < LoadOffs) {
            if (int(StoreOffs+StoreSize[i]) > LoadOffs) return true;
          } else {
            if (int(LoadOffs+LoadSize) > StoreOffs) return true;
          }
        }
    }
  }
  return false;
}

/// getHazardType - We return hazard for any non-branch instruction that would
/// terminate terminate the dispatch group.  We turn NoopHazard for any
/// instructions that wouldn't terminate the dispatch group that would cause a
/// pipeline flush.
ScheduleHazardRecognizer::HazardType PPCHazardRecognizer970::
getHazardType(SUnit *SU) {
  const SDNode *Node = SU->getNode()->getFlaggedMachineNode();
  bool isFirst, isSingle, isCracked, isLoad, isStore;
  PPCII::PPC970_Unit InstrType = 
    GetInstrType(Node->getOpcode(), isFirst, isSingle, isCracked,
                 isLoad, isStore);
  if (InstrType == PPCII::PPC970_Pseudo) return NoHazard;  
  unsigned Opcode = Node->getMachineOpcode();

  // We can only issue a PPC970_First/PPC970_Single instruction (such as
  // crand/mtspr/etc) if this is the first cycle of the dispatch group.
  if (NumIssued != 0 && (isFirst || isSingle))
    return Hazard;
  
  // If this instruction is cracked into two ops by the decoder, we know that
  // it is not a branch and that it cannot issue if 3 other instructions are
  // already in the dispatch group.
  if (isCracked && NumIssued > 2)
    return Hazard;
      
  switch (InstrType) {
  default: assert(0 && "Unknown instruction type!");
  case PPCII::PPC970_FXU:
  case PPCII::PPC970_LSU:
  case PPCII::PPC970_FPU:
  case PPCII::PPC970_VALU:
  case PPCII::PPC970_VPERM:
    // We can only issue a branch as the last instruction in a group.
    if (NumIssued == 4) return Hazard;
    break;
  case PPCII::PPC970_CRU:
    // We can only issue a CR instruction in the first two slots.
    if (NumIssued >= 2) return Hazard;
    break;
  case PPCII::PPC970_BRU:
    break;
  }
  
  // Do not allow MTCTR and BCTRL to be in the same dispatch group.
  if (HasCTRSet && (Opcode == PPC::BCTRL_Macho || Opcode == PPC::BCTRL_ELF))
    return NoopHazard;
  
  // If this is a load following a store, make sure it's not to the same or
  // overlapping address.
  if (isLoad && NumStores) {
    unsigned LoadSize;
    switch (Opcode) {
    default: assert(0 && "Unknown load!");
    case PPC::LBZ:   case PPC::LBZU:
    case PPC::LBZX:
    case PPC::LBZ8:  case PPC::LBZU8:
    case PPC::LBZX8:
    case PPC::LVEBX:
      LoadSize = 1;
      break;
    case PPC::LHA:   case PPC::LHAU:
    case PPC::LHAX:
    case PPC::LHZ:   case PPC::LHZU:
    case PPC::LHZX:
    case PPC::LVEHX:
    case PPC::LHBRX:
    case PPC::LHA8:   case PPC::LHAU8:
    case PPC::LHAX8:
    case PPC::LHZ8:   case PPC::LHZU8:
    case PPC::LHZX8:
      LoadSize = 2;
      break;
    case PPC::LFS:    case PPC::LFSU:
    case PPC::LFSX:
    case PPC::LWZ:    case PPC::LWZU:
    case PPC::LWZX:
    case PPC::LWA:
    case PPC::LWAX:
    case PPC::LVEWX:
    case PPC::LWBRX:
    case PPC::LWZ8:
    case PPC::LWZX8:
      LoadSize = 4;
      break;
    case PPC::LFD:    case PPC::LFDU:
    case PPC::LFDX:
    case PPC::LD:     case PPC::LDU:
    case PPC::LDX:
      LoadSize = 8;
      break;
    case PPC::LVX:
    case PPC::LVXL:
      LoadSize = 16;
      break;
    }
    
    if (isLoadOfStoredAddress(LoadSize, 
                              Node->getOperand(0), Node->getOperand(1)))
      return NoopHazard;
  }
  
  return NoHazard;
}

void PPCHazardRecognizer970::EmitInstruction(SUnit *SU) {
  const SDNode *Node = SU->getNode()->getFlaggedMachineNode();
  bool isFirst, isSingle, isCracked, isLoad, isStore;
  PPCII::PPC970_Unit InstrType = 
    GetInstrType(Node->getOpcode(), isFirst, isSingle, isCracked,
                 isLoad, isStore);
  if (InstrType == PPCII::PPC970_Pseudo) return;  
  unsigned Opcode = Node->getMachineOpcode();

  // Update structural hazard information.
  if (Opcode == PPC::MTCTR) HasCTRSet = true;
  
  // Track the address stored to.
  if (isStore) {
    unsigned ThisStoreSize;
    switch (Opcode) {
    default: assert(0 && "Unknown store instruction!");
    case PPC::STB:    case PPC::STB8:
    case PPC::STBU:   case PPC::STBU8:
    case PPC::STBX:   case PPC::STBX8:
    case PPC::STVEBX:
      ThisStoreSize = 1;
      break;
    case PPC::STH:    case PPC::STH8:
    case PPC::STHU:   case PPC::STHU8:
    case PPC::STHX:   case PPC::STHX8:
    case PPC::STVEHX:
    case PPC::STHBRX:
      ThisStoreSize = 2;
      break;
    case PPC::STFS:
    case PPC::STFSU:
    case PPC::STFSX:
    case PPC::STWX:   case PPC::STWX8:
    case PPC::STWUX:
    case PPC::STW:    case PPC::STW8:
    case PPC::STWU:   case PPC::STWU8:
    case PPC::STVEWX:
    case PPC::STFIWX:
    case PPC::STWBRX:
      ThisStoreSize = 4;
      break;
    case PPC::STD_32:
    case PPC::STDX_32:
    case PPC::STD:
    case PPC::STDU:
    case PPC::STFD:
    case PPC::STFDX:
    case PPC::STDX:
    case PPC::STDUX:
      ThisStoreSize = 8;
      break;
    case PPC::STVX:
    case PPC::STVXL:
      ThisStoreSize = 16;
      break;
    }
    
    StoreSize[NumStores] = ThisStoreSize;
    StorePtr1[NumStores] = Node->getOperand(1);
    StorePtr2[NumStores] = Node->getOperand(2);
    ++NumStores;
  }
  
  if (InstrType == PPCII::PPC970_BRU || isSingle)
    NumIssued = 4;  // Terminate a d-group.
  ++NumIssued;
  
  // If this instruction is cracked into two ops by the decoder, remember that
  // we issued two pieces.
  if (isCracked)
    ++NumIssued;
  
  if (NumIssued == 5)
    EndDispatchGroup();
}

void PPCHazardRecognizer970::AdvanceCycle() {
  assert(NumIssued < 5 && "Illegal dispatch group!");
  ++NumIssued;
  if (NumIssued == 5)
    EndDispatchGroup();
}
