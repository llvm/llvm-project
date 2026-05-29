//===-- MetalTargetLowering.h - Define Metal TargetLowering -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Metal specific subclass of TargetLowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALTARGETLOWERING_H
#define LLVM_LIB_TARGET_METAL_METALTARGETLOWERING_H

#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class MetalSubtarget;
class MetalTargetMachine;

class MetalTargetLowering : public TargetLowering {
public:
  explicit MetalTargetLowering(const MetalTargetMachine &TM,
                               const MetalSubtarget &STI);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALTARGETLOWERING_H
