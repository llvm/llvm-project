//===-- AMDGPUInstrInfo.cpp - Base class for AMD GPU InstrInfo ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Implementation of the TargetInstrInfo class that is common to all
/// AMD GPUs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUInstrInfo.h"
#include "AMDGPURegisterInfo.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRINFO_NAMED_OPS
#define GET_INSTRMAP_INFO
#include "AMDGPUGenInstrInfo.inc"

// Pin the vtable to this file.
void AMDGPUInstrInfo::anchor() {}

AMDGPUInstrInfo::AMDGPUInstrInfo(const AMDGPUSubtarget &st)
  : AMDGPUGenInstrInfo(-1,-1), RI(st), ST(st) { }

const AMDGPURegisterInfo &AMDGPUInstrInfo::getRegisterInfo() const {
  return RI;
}

bool AMDGPUInstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                           unsigned &SrcReg, unsigned &DstReg,
                                           unsigned &SubIdx) const {
// TODO: Implement this function
  return false;
}

unsigned AMDGPUInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {
// TODO: Implement this function
  return 0;
}

unsigned AMDGPUInstrInfo::isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                                   int &FrameIndex) const {
// TODO: Implement this function
  return 0;
}

bool AMDGPUInstrInfo::hasLoadFromStackSlot(const MachineInstr *MI,
                                          const MachineMemOperand *&MMO,
                                          int &FrameIndex) const {
// TODO: Implement this function
  return false;
}
unsigned AMDGPUInstrInfo::isStoreFromStackSlot(const MachineInstr *MI,
                                              int &FrameIndex) const {
// TODO: Implement this function
  return 0;
}
unsigned AMDGPUInstrInfo::isStoreFromStackSlotPostFE(const MachineInstr *MI,
                                                    int &FrameIndex) const {
// TODO: Implement this function
  return 0;
}
bool AMDGPUInstrInfo::hasStoreFromStackSlot(const MachineInstr *MI,
                                           const MachineMemOperand *&MMO,
                                           int &FrameIndex) const {
// TODO: Implement this function
  return false;
}

MachineInstr *
AMDGPUInstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                      MachineBasicBlock::iterator &MBBI,
                                      LiveVariables *LV) const {
// TODO: Implement this function
  return nullptr;
}

void
AMDGPUInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned SrcReg, bool isKill,
                                    int FrameIndex,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const {
  llvm_unreachable("Not Implemented");
}

void
AMDGPUInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI,
                                     unsigned DestReg, int FrameIndex,
                                     const TargetRegisterClass *RC,
                                     const TargetRegisterInfo *TRI) const {
  llvm_unreachable("Not Implemented");
}

bool AMDGPUInstrInfo::expandPostRAPseudo (MachineBasicBlock::iterator MI) const {
  MachineBasicBlock *MBB = MI->getParent();
  int OffsetOpIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                               AMDGPU::OpName::addr);
   // addr is a custom operand with multiple MI operands, and only the
   // first MI operand is given a name.
  int RegOpIdx = OffsetOpIdx + 1;
  int ChanOpIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                             AMDGPU::OpName::chan);
  if (isRegisterLoad(*MI)) {
    int DstOpIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                              AMDGPU::OpName::dst);
    unsigned RegIndex = MI->getOperand(RegOpIdx).getImm();
    unsigned Channel = MI->getOperand(ChanOpIdx).getImm();
    unsigned Address = calculateIndirectAddress(RegIndex, Channel);
    unsigned OffsetReg = MI->getOperand(OffsetOpIdx).getReg();
    if (OffsetReg == AMDGPU::INDIRECT_BASE_ADDR) {
      buildMovInstr(MBB, MI, MI->getOperand(DstOpIdx).getReg(),
                    getIndirectAddrRegClass()->getRegister(Address));
    } else {
      buildIndirectRead(MBB, MI, MI->getOperand(DstOpIdx).getReg(),
                        Address, OffsetReg);
    }
  } else if (isRegisterStore(*MI)) {
    int ValOpIdx = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                              AMDGPU::OpName::val);
    unsigned RegIndex = MI->getOperand(RegOpIdx).getImm();
    unsigned Channel = MI->getOperand(ChanOpIdx).getImm();
    unsigned Address = calculateIndirectAddress(RegIndex, Channel);
    unsigned OffsetReg = MI->getOperand(OffsetOpIdx).getReg();
    if (OffsetReg == AMDGPU::INDIRECT_BASE_ADDR) {
      buildMovInstr(MBB, MI, getIndirectAddrRegClass()->getRegister(Address),
                    MI->getOperand(ValOpIdx).getReg());
    } else {
      buildIndirectWrite(MBB, MI, MI->getOperand(ValOpIdx).getReg(),
                         calculateIndirectAddress(RegIndex, Channel),
                         OffsetReg);
    }
  } else {
    return false;
  }

  MBB->erase(MI);
  return true;
}


MachineInstr *
AMDGPUInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr *MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      int FrameIndex) const {
// TODO: Implement this function
  return nullptr;
}
MachineInstr*
AMDGPUInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr *MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      MachineInstr *LoadMI) const {
  // TODO: Implement this function
  return nullptr;
}
bool
AMDGPUInstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                     const SmallVectorImpl<unsigned> &Ops) const {
  // TODO: Implement this function
  return false;
}
bool
AMDGPUInstrInfo::unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                                 unsigned Reg, bool UnfoldLoad,
                                 bool UnfoldStore,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  // TODO: Implement this function
  return false;
}

bool
AMDGPUInstrInfo::unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                    SmallVectorImpl<SDNode*> &NewNodes) const {
  // TODO: Implement this function
  return false;
}

unsigned
AMDGPUInstrInfo::getOpcodeAfterMemoryUnfold(unsigned Opc,
                                           bool UnfoldLoad, bool UnfoldStore,
                                           unsigned *LoadRegIndex) const {
  // TODO: Implement this function
  return 0;
}

bool AMDGPUInstrInfo::enableClusterLoads() const {
  return true;
}

// FIXME: This behaves strangely. If, for example, you have 32 load + stores,
// the first 16 loads will be interleaved with the stores, and the next 16 will
// be clustered as expected. It should really split into 2 16 store batches.
//
// Loads are clustered until this returns false, rather than trying to schedule
// groups of stores. This also means we have to deal with saying different
// address space loads should be clustered, and ones which might cause bank
// conflicts.
//
// This might be deprecated so it might not be worth that much effort to fix.
bool AMDGPUInstrInfo::shouldScheduleLoadsNear(SDNode *Load0, SDNode *Load1,
                                              int64_t Offset0, int64_t Offset1,
                                              unsigned NumLoads) const {
  assert(Offset1 > Offset0 &&
         "Second offset should be larger than first offset!");
  // If we have less than 16 loads in a row, and the offsets are within 64
  // bytes, then schedule together.

  // A cacheline is 64 bytes (for global memory).
  return (NumLoads <= 16 && (Offset1 - Offset0) < 64);
}

bool
AMDGPUInstrInfo::ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond)
  const {
  // TODO: Implement this function
  return true;
}
void AMDGPUInstrInfo::insertNoop(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const {
  // TODO: Implement this function
}

bool AMDGPUInstrInfo::isPredicated(const MachineInstr *MI) const {
  // TODO: Implement this function
  return false;
}
bool
AMDGPUInstrInfo::SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                                  const SmallVectorImpl<MachineOperand> &Pred2)
  const {
  // TODO: Implement this function
  return false;
}

bool AMDGPUInstrInfo::DefinesPredicate(MachineInstr *MI,
                                      std::vector<MachineOperand> &Pred) const {
  // TODO: Implement this function
  return false;
}

bool AMDGPUInstrInfo::isPredicable(MachineInstr *MI) const {
  // TODO: Implement this function
  return MI->getDesc().isPredicable();
}

bool
AMDGPUInstrInfo::isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const {
  // TODO: Implement this function
  return true;
}

bool AMDGPUInstrInfo::isRegisterStore(const MachineInstr &MI) const {
  return get(MI.getOpcode()).TSFlags & AMDGPU_FLAG_REGISTER_STORE;
}

bool AMDGPUInstrInfo::isRegisterLoad(const MachineInstr &MI) const {
  return get(MI.getOpcode()).TSFlags & AMDGPU_FLAG_REGISTER_LOAD;
}

int AMDGPUInstrInfo::getIndirectIndexBegin(const MachineFunction &MF) const {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  int Offset = -1;

  if (MFI->getNumObjects() == 0) {
    return -1;
  }

  if (MRI.livein_empty()) {
    return 0;
  }

  const TargetRegisterClass *IndirectRC = getIndirectAddrRegClass();
  for (MachineRegisterInfo::livein_iterator LI = MRI.livein_begin(),
                                            LE = MRI.livein_end();
                                            LI != LE; ++LI) {
    unsigned Reg = LI->first;
    if (TargetRegisterInfo::isVirtualRegister(Reg) ||
        !IndirectRC->contains(Reg))
      continue;

    unsigned RegIndex;
    unsigned RegEnd;
    for (RegIndex = 0, RegEnd = IndirectRC->getNumRegs(); RegIndex != RegEnd;
                                                          ++RegIndex) {
      if (IndirectRC->getRegister(RegIndex) == Reg)
        break;
    }
    Offset = std::max(Offset, (int)RegIndex);
  }

  return Offset + 1;
}

int AMDGPUInstrInfo::getIndirectIndexEnd(const MachineFunction &MF) const {
  int Offset = 0;
  const MachineFrameInfo *MFI = MF.getFrameInfo();

  // Variable sized objects are not supported
  assert(!MFI->hasVarSizedObjects());

  if (MFI->getNumObjects() == 0) {
    return -1;
  }

  Offset = MF.getTarget()
               .getSubtargetImpl()
               ->getFrameLowering()
               ->getFrameIndexOffset(MF, -1);

  return getIndirectIndexBegin(MF) + Offset;
}

int AMDGPUInstrInfo::getMaskedMIMGOp(uint16_t Opcode, unsigned Channels) const {
  switch (Channels) {
  default: return Opcode;
  case 1: return AMDGPU::getMaskedMIMGOp(Opcode, AMDGPU::Channels_1);
  case 2: return AMDGPU::getMaskedMIMGOp(Opcode, AMDGPU::Channels_2);
  case 3: return AMDGPU::getMaskedMIMGOp(Opcode, AMDGPU::Channels_3);
  }
}

// Wrapper for Tablegen'd function.  enum Subtarget is not defined in any
// header files, so we need to wrap it in a function that takes unsigned
// instead.
namespace llvm {
namespace AMDGPU {
static int getMCOpcode(uint16_t Opcode, unsigned Gen) {
  return getMCOpcodeGen(Opcode, (enum Subtarget)Gen);
}
}
}

// This must be kept in sync with the SISubtarget class in SIInstrInfo.td
enum SISubtarget {
  SI = 0,
  VI = 1
};

enum SISubtarget AMDGPUSubtargetToSISubtarget(unsigned Gen) {
  switch (Gen) {
  default:
    return SI;
  case AMDGPUSubtarget::VOLCANIC_ISLANDS:
    return VI;
  }
}

int AMDGPUInstrInfo::pseudoToMCOpcode(int Opcode) const {
  int MCOp = AMDGPU::getMCOpcode(Opcode,
                        AMDGPUSubtargetToSISubtarget(RI.ST.getGeneration()));

  // -1 means that Opcode is already a native instruction.
  if (MCOp == -1)
    return Opcode;

  // (uint16_t)-1 means that Opcode is a pseudo instruction that has
  // no encoding in the given subtarget generation.
  if (MCOp == (uint16_t)-1)
    return -1;

  return MCOp;
}
