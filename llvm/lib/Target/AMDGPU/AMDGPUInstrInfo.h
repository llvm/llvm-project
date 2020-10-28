//===-- AMDGPUInstrInfo.h - AMDGPU Instruction Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Contains the definition of a TargetInstrInfo class that is common
/// to all AMD GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUINSTRINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUINSTRINFO_H

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

namespace llvm {

class GCNSubtarget;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;

class AMDGPUInstrInfo {
public:
  explicit AMDGPUInstrInfo(const GCNSubtarget &st);

  static bool isUniformMMO(const MachineMemOperand *MMO);
};

namespace AMDGPU {

struct RsrcIntrinsic {
  unsigned Intr;
  uint8_t RsrcArg;
  bool IsImage;
};
const RsrcIntrinsic *lookupRsrcIntrinsic(unsigned Intr);

struct D16ImageDimIntrinsic {
  unsigned Intr;
  unsigned D16HelperIntr;
};
const D16ImageDimIntrinsic *lookupD16ImageDimIntrinsic(unsigned Intr);

struct ImageDimIntrinsicInfo {
  unsigned Intr;
  unsigned BaseOpcode;
  MIMGDim Dim;
  unsigned GradientStart;
  unsigned CoordStart;
  unsigned VAddrEnd;
  unsigned GradientTyArg;
  unsigned CoordTyArg;
};
const ImageDimIntrinsicInfo *getImageDimIntrinsicInfo(unsigned Intr);

const ImageDimIntrinsicInfo *getImageDimInstrinsicByBaseOpcode(unsigned BaseOpcode,
                                                               unsigned Dim);

} // end AMDGPU namespace
} // End llvm namespace

#endif
