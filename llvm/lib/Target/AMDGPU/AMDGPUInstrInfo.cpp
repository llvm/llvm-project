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
#include "AMDGPUGenInstrInfo.inc"

// Pin the vtable to this file.
void AMDGPUInstrInfo::anchor() {}

AMDGPUInstrInfo::AMDGPUInstrInfo(const AMDGPUSubtarget &ST)
  : AMDGPUGenInstrInfo(AMDGPU::ADJCALLSTACKUP, AMDGPU::ADJCALLSTACKDOWN),
    ST(ST),
    AMDGPUASI(ST.getAMDGPUAS()) {}

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

// This must be kept in sync with the SIEncodingFamily class in SIInstrInfo.td
enum SIEncodingFamily {
  SI = 0,
  VI = 1,
  SDWA = 2,
  SDWA9 = 3,
  GFX80 = 4,
  GFX9 = 5
};

static SIEncodingFamily subtargetEncodingFamily(const AMDGPUSubtarget &ST) {
  switch (ST.getGeneration()) {
  case AMDGPUSubtarget::SOUTHERN_ISLANDS:
  case AMDGPUSubtarget::SEA_ISLANDS:
    return SIEncodingFamily::SI;
  case AMDGPUSubtarget::VOLCANIC_ISLANDS:
  case AMDGPUSubtarget::GFX9:
    return SIEncodingFamily::VI;

  // FIXME: This should never be called for r600 GPUs.
  case AMDGPUSubtarget::R600:
  case AMDGPUSubtarget::R700:
  case AMDGPUSubtarget::EVERGREEN:
  case AMDGPUSubtarget::NORTHERN_ISLANDS:
    return SIEncodingFamily::SI;
  }

  llvm_unreachable("Unknown subtarget generation!");
}

int AMDGPUInstrInfo::pseudoToMCOpcode(int Opcode) const {
  SIEncodingFamily Gen = subtargetEncodingFamily(ST);

  if ((get(Opcode).TSFlags & SIInstrFlags::renamedInGFX9) != 0 &&
    ST.getGeneration() >= AMDGPUSubtarget::GFX9)
    Gen = SIEncodingFamily::GFX9;

  if (get(Opcode).TSFlags & SIInstrFlags::SDWA)
    Gen = ST.getGeneration() == AMDGPUSubtarget::GFX9 ? SIEncodingFamily::SDWA9
                                                      : SIEncodingFamily::SDWA;
  // Adjust the encoding family to GFX80 for D16 buffer instructions when the
  // subtarget has UnpackedD16VMem feature.
  // TODO: remove this when we discard GFX80 encoding.
  if (ST.hasUnpackedD16VMem() && (get(Opcode).TSFlags & SIInstrFlags::D16)
                              && !(get(Opcode).TSFlags & SIInstrFlags::MIMG))
    Gen = SIEncodingFamily::GFX80;

  int MCOp = AMDGPU::getMCOpcode(Opcode, Gen);

  // -1 means that Opcode is already a native instruction.
  if (MCOp == -1)
    return Opcode;

  // (uint16_t)-1 means that Opcode is a pseudo instruction that has
  // no encoding in the given subtarget generation.
  if (MCOp == (uint16_t)-1)
    return -1;

  return MCOp;
}

// TODO: Should largely merge with AMDGPUTTIImpl::isSourceOfDivergence.
bool AMDGPUInstrInfo::isUniformMMO(const MachineMemOperand *MMO) {
  const Value *Ptr = MMO->getValue();
  // UndefValue means this is a load of a kernel input.  These are uniform.
  // Sometimes LDS instructions have constant pointers.
  // If Ptr is null, then that means this mem operand contains a
  // PseudoSourceValue like GOT.
  if (!Ptr || isa<UndefValue>(Ptr) ||
      isa<Constant>(Ptr) || isa<GlobalValue>(Ptr))
    return true;

  if (MMO->getAddrSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT)
    return true;

  if (const Argument *Arg = dyn_cast<Argument>(Ptr))
    return AMDGPU::isArgPassedInSGPR(Arg);

  const Instruction *I = dyn_cast<Instruction>(Ptr);
  return I && I->getMetadata("amdgpu.uniform");
}
