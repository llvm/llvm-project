//==-- XtensaTargetParser - Parser for Xtensa features --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise Xtensa hardware features
//
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

enum FeatureKind : uint64_t {
  FK_INVALID = 0,
  FK_NONE = 1,
  FK_FP = 1 << 1,
  FK_WINDOWED = 1 << 2,
  FK_BOOLEAN = 1 << 3,
  FK_DENSITY = 1 << 4,
  FK_LOOP = 1 << 5,
  FK_SEXT = 1 << 6,
  FK_NSA = 1 << 7,
  FK_CLAMPS = 1 << 8,
  FK_MINMAX = 1 << 9,
  FK_MAC16 = 1 << 10,
  FK_MUL32 = 1 << 11,
  FK_MUL32HIGH = 1 << 12,
  FK_DIV32 = 1 << 13,
  FK_MUL16 = 1 << 14,
  FK_DFPACCEL = 1 << 15,
  FK_S32C1I = 1 << 16,
  FK_THREADPTR = 1 << 17,
  FK_EXTENDEDL32R = 1 << 18,
  FK_DATACACHE = 1 << 19,
  FK_DEBUG = 1 << 20,
  FK_EXCEPTION = 1 << 21,
  FK_HIGHPRIINTERRUPTS = 1 << 22,
  FK_HIGHPRIINTERRUPTSLEVEL3 = 1 << 23,
  FK_HIGHPRIINTERRUPTSLEVEL4 = 1 << 24,
  FK_HIGHPRIINTERRUPTSLEVEL5 = 1 << 25,
  FK_HIGHPRIINTERRUPTSLEVEL6 = 1 << 26,
  FK_HIGHPRIINTERRUPTSLEVEL7 = 1 << 27,
  FK_COPROCESSOR = 1 << 28,
  FK_INTERRUPT = 1 << 29,
  FK_RVECTOR = 1 << 30,
  FK_TIMERS1 = 1ULL << 31,
  FK_TIMERS2 = 1ULL << 32,
  FK_TIMERS3 = 1ULL << 33,
  FK_PRID = 1ULL << 34,
  FK_REGPROTECT = 1ULL << 35,
  FK_MISCSR = 1ULL << 36
};

CPUKind parseCPUKind(StringRef CPU);
StringRef getBaseName(StringRef CPU);
void getCPUFeatures(StringRef CPU, SmallVectorImpl<StringRef> &Features);
void fillValidCPUList(SmallVectorImpl<StringRef> &Values);

} // namespace Xtensa
} // namespace llvm

#endif // LLVM_SUPPORT_XTENSATARGETPARSER_H
