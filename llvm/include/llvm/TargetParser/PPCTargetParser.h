//===---- PPCTargetParser - Parser for target features ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features
// for PPC CPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_PPCTARGETPARSER_H
#define LLVM_TARGETPARSER_PPCTARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace PPC {
bool isValidCPU(StringRef CPU);
void fillValidCPUList(SmallVectorImpl<StringRef> &Values);
void fillValidTuneCPUList(SmallVectorImpl<StringRef> &Values);

// Get target CPU name.
// If CPUName is empty or generic, return the default CPU name.
// If CPUName is not empty or generic, return the normalized CPU name.
StringRef getNormalizedPPCTargetCPU(const Triple &T, StringRef CPUName = "");

// Get the tune CPU name.
StringRef getNormalizedPPCTuneCPU(const Triple &T, StringRef CPUName = "");

// For PPC, there are some cpu names for same CPU, like pwr10 and power10,
// normalize them.
StringRef normalizeCPUName(StringRef CPUName);
} // namespace PPC
} // namespace llvm

#endif
