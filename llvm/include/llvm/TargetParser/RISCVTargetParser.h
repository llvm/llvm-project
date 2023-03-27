//===-- RISCVTargetParser - Parser for target features ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features
// for RISC-V CPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_RISCVTARGETPARSER_H
#define LLVM_TARGETPARSER_RISCVTARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include <vector>

namespace llvm {

class Triple;

namespace RISCV {

// We use 64 bits as the known part in the scalable vector types.
static constexpr unsigned RVVBitsPerBlock = 64;

enum CPUKind : unsigned {
#define PROC(ENUM, NAME, DEFAULT_MARCH) CK_##ENUM,
#define TUNE_PROC(ENUM, NAME) CK_##ENUM,
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
};

bool checkCPUKind(CPUKind Kind, bool IsRV64);
bool checkTuneCPUKind(CPUKind Kind, bool IsRV64);
CPUKind parseCPUKind(StringRef CPU);
CPUKind parseTuneCPUKind(StringRef CPU, bool IsRV64);
StringRef getMArchFromMcpu(StringRef CPU);
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64);
void fillValidTuneCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64);
bool getCPUFeaturesExceptStdExt(CPUKind Kind, std::vector<StringRef> &Features);

bool isX18ReservedByDefault(const Triple &TT);

} // namespace RISCV
} // namespace llvm

#endif
