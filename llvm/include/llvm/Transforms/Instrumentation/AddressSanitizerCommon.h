//===--------- Definition of the AddressSanitizer class ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common infrastructure for AddressSanitizer and
// HWAddressSanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZERCOMMON_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_ADDRESSSANITIZERCOMMON_H

#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/MemoryRefInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

namespace llvm {

// Get AddressSanitizer parameters.
void getAddressSanitizerParams(const Triple &TargetTriple, int LongSize,
                               bool IsKasan, uint64_t *ShadowBase,
                               int *MappingScale, bool *OrShadowOffset);

} // namespace llvm

#endif
