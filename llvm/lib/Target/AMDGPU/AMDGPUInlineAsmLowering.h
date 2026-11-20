//===-- AMDGPUInlineAsmLowering.h - AMDGPU inline asm ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the declaration of the AMDGPUInlineAsmLowering class,
/// which adds the AMDGPU specific handling of inline asm operands.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUINLINEASMLOWERING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUINLINEASMLOWERING_H

#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"

namespace llvm {

class AMDGPUInlineAsmLowering : public InlineAsmLowering {
public:
  AMDGPUInlineAsmLowering(const TargetLowering *TLI) : InlineAsmLowering(TLI) {}

  Register lowerAsmInputValue(Register Dst, Register Src,
                              MachineIRBuilder &MIRBuilder) const override;
};

} // end namespace llvm

#endif
