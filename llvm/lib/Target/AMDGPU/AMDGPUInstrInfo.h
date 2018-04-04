//===-- AMDGPUInstrInfo.h - AMDGPU Instruction Information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Contains the definition of a TargetInstrInfo class that is common
/// to all AMD GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUINSTRINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUINSTRINFO_H

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "AMDGPUGenInstrInfo.inc"
#undef GET_INSTRINFO_HEADER

namespace llvm {

class AMDGPUSubtarget;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;

class AMDGPUInstrInfo : public AMDGPUGenInstrInfo {
private:
  const AMDGPUSubtarget &ST;

  virtual void anchor();
protected:
  AMDGPUAS AMDGPUASI;

public:
  explicit AMDGPUInstrInfo(const AMDGPUSubtarget &st);

  bool shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                               int64_t Offset1, int64_t Offset2,
                               unsigned NumLoads) const override;

  /// \brief Return a target-specific opcode if Opcode is a pseudo instruction.
  /// Return -1 if the target-specific opcode for the pseudo instruction does
  /// not exist. If Opcode is not a pseudo instruction, this is identity.
  int pseudoToMCOpcode(int Opcode) const;

  static bool isUniformMMO(const MachineMemOperand *MMO);
};

namespace AMDGPU {

struct RsrcIntrinsic {
  unsigned Intr;
  uint8_t RsrcArg;
  bool IsImage;
};
const RsrcIntrinsic *lookupRsrcIntrinsicByIntr(unsigned Intr);

struct D16ImageDimIntrinsic {
  unsigned Intr;
  unsigned D16HelperIntr;
};
const D16ImageDimIntrinsic *lookupD16ImageDimIntrinsicByIntr(unsigned Intr);

} // end AMDGPU namespace
} // End llvm namespace

#endif
