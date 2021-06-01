//===-- M68kLegalizerInfo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the targeting of the Machinelegalizer class for M68k.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M68K_M68KLEGALIZERINFO_H
#define LLVM_LIB_TARGET_M68K_M68KLEGALIZERINFO_H

#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"

namespace llvm {

class M68kSubtarget;

/// This class provides the information for the target register banks.
class M68kLegalizerInfo : public LegalizerInfo {
public:
  M68kLegalizerInfo(const M68kSubtarget &ST);
};
} // end namespace llvm
#endif
