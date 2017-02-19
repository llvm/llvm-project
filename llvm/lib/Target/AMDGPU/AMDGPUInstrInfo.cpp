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
#define GET_INSTRMAP_INFO
#include "AMDGPUGenInstrInfo.inc"

// Pin the vtable to this file.
void AMDGPUInstrInfo::anchor() {}

AMDGPUInstrInfo::AMDGPUInstrInfo(const AMDGPUSubtarget &ST)
  : AMDGPUGenInstrInfo(-1, -1), ST(ST) {}

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

int AMDGPUInstrInfo::getMaskedMIMGOp(uint16_t Opcode, unsigned Channels) const {
  switch (Channels) {
  default: return Opcode;
  case 1: return AMDGPU::getMaskedMIMGOp(Opcode, AMDGPU::Channels_1);
  case 2: return AMDGPU::getMaskedMIMGOp(Opcode, AMDGPU::Channels_2);
  case 3: return AMDGPU::getMaskedMIMGOp(Opcode, AMDGPU::Channels_3);
  }
}

// This must be kept in sync with the SIEncodingFamily class in SIInstrInfo.td
enum SIEncodingFamily {
  SI = 0,
  VI = 1
};

// Wrapper for Tablegen'd function.  enum Subtarget is not defined in any
// header files, so we need to wrap it in a function that takes unsigned
// instead.
namespace llvm {
namespace AMDGPU {
static int getMCOpcode(uint16_t Opcode, unsigned Gen) {
  return getMCOpcodeGen(Opcode, static_cast<Subtarget>(Gen));
}
}
}

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
  int MCOp = AMDGPU::getMCOpcode(Opcode, subtargetEncodingFamily(ST));

  // -1 means that Opcode is already a native instruction.
  if (MCOp == -1)
    return Opcode;

  // (uint16_t)-1 means that Opcode is a pseudo instruction that has
  // no encoding in the given subtarget generation.
  if (MCOp == (uint16_t)-1)
    return -1;

  return MCOp;
}
