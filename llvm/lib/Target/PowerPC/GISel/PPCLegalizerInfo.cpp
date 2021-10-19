//===- PPCLegalizerInfo.h ----------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the Machinelegalizer class for PowerPC
//===----------------------------------------------------------------------===//

#include "PPCLegalizerInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ppc-legalinfo"

using namespace llvm;
using namespace LegalizeActions;

PPCLegalizerInfo::PPCLegalizerInfo(const PPCSubtarget &ST) {
  getLegacyLegalizerInfo().computeTables();
}
