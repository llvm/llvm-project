//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a target parser to recognise Xtensa hardware features.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_XTENSATARGETPARSER_H
#define LLVM_TARGETPARSER_XTENSATARGETPARSER_H

#include "llvm/TargetParser/Triple.h"
#include <vector>

namespace llvm {
class StringRef;

namespace Xtensa {

enum CPUKind : unsigned {
#define XTENSA_CPU(ENUM, NAME, FEATURES) CK_##ENUM,
#include "XtensaTargetParser.def"
};

enum XtensaFeatureKind : uint64_t {
  XF_INVALID = 0,
  XF_NONE = 1,
  XF_FP = 1 << 1,
  XF_WINDOWED = 1 << 2,
  XF_BOOLEAN = 1 << 3,
  XF_DENSITY = 1 << 4,
  XF_LOOP = 1 << 5,
  XF_SEXT = 1 << 6,
  XF_NSA = 1 << 7,
  XF_CLAMPS = 1 << 8,
  XF_MINMAX = 1 << 9,
  XF_MAC16 = 1 << 10,
  XF_MUL32 = 1 << 11,
  XF_MUL32HIGH = 1 << 12,
  XF_DIV32 = 1 << 13,
  XF_MUL16 = 1 << 14,
  XF_DFPACCEL = 1 << 15,
  XF_S32C1I = 1 << 16,
  XF_THREADPTR = 1 << 17,
  XF_EXTENDEDL32R = 1 << 18,
  XF_DATACACHE = 1 << 19,
  XF_DEBUG = 1 << 20,
  XF_EXCEPTION = 1 << 21,
  XF_HIGHPRIINTERRUPTS = 1 << 22,
  XF_HIGHPRIINTERRUPTSLEVEL3 = 1 << 23,
  XF_HIGHPRIINTERRUPTSLEVEL4 = 1 << 24,
  XF_HIGHPRIINTERRUPTSLEVEL5 = 1 << 25,
  XF_HIGHPRIINTERRUPTSLEVEL6 = 1 << 26,
  XF_HIGHPRIINTERRUPTSLEVEL7 = 1 << 27,
  XF_COPROCESSOR = 1 << 28,
  XF_INTERRUPT = 1 << 29,
  XF_RVECTOR = 1 << 30,
  XF_TIMERS1 = 1ULL << 31,
  XF_TIMERS2 = 1ULL << 32,
  XF_TIMERS3 = 1ULL << 33,
  XF_PRID = 1ULL << 34,
  XF_REGPROTECT = 1ULL << 35,
  XF_MISCSR = 1ULL << 36
};

CPUKind parseCPUKind(StringRef CPU);
StringRef getBaseName(StringRef CPU);
void getCPUFeatures(StringRef CPU, SmallVectorImpl<StringRef> &Features);
void fillValidCPUList(SmallVectorImpl<StringRef> &Values);

} // namespace Xtensa
} // namespace llvm

#endif // LLVM_SUPPORT_XTENSATARGETPARSER_H
