//===-- RISCVTargetParser.cpp - Parser for target features ------*- C++ -*-===//
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

#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace RISCV {

struct CPUInfo {
  StringLiteral Name;
  CPUKind Kind;
  StringLiteral DefaultMarch;
  bool isInvalid() const { return DefaultMarch.empty(); }
  bool is64Bit() const { return DefaultMarch.starts_with("rv64"); }
};

constexpr CPUInfo RISCVCPUInfo[] = {
#define PROC(ENUM, NAME, DEFAULT_MARCH)                              \
  {NAME, CK_##ENUM, DEFAULT_MARCH},
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
};

bool checkCPUKind(CPUKind Kind, bool IsRV64) {
  if (Kind == CK_INVALID)
    return false;
  return RISCVCPUInfo[static_cast<unsigned>(Kind)].is64Bit() == IsRV64;
}

bool checkTuneCPUKind(CPUKind Kind, bool IsRV64) {
  if (Kind == CK_INVALID)
    return false;
#define TUNE_PROC(ENUM, NAME)                                                  \
  if (Kind == CK_##ENUM)                                                       \
    return true;
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
  return RISCVCPUInfo[static_cast<unsigned>(Kind)].is64Bit() == IsRV64;
}

CPUKind parseCPUKind(StringRef CPU) {
  return llvm::StringSwitch<CPUKind>(CPU)
#define PROC(ENUM, NAME, DEFAULT_MARCH) .Case(NAME, CK_##ENUM)
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
      .Default(CK_INVALID);
}

CPUKind parseTuneCPUKind(StringRef TuneCPU, bool IsRV64) {
  return llvm::StringSwitch<CPUKind>(TuneCPU)
#define PROC(ENUM, NAME, DEFAULT_MARCH) .Case(NAME, CK_##ENUM)
#define TUNE_PROC(ENUM, NAME) .Case(NAME, CK_##ENUM)
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
      .Default(CK_INVALID);
}

StringRef getMArchFromMcpu(StringRef CPU) {
  CPUKind Kind = parseCPUKind(CPU);
  return RISCVCPUInfo[static_cast<unsigned>(Kind)].DefaultMarch;
}

void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (C.Kind != CK_INVALID && IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
}

void fillValidTuneCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (C.Kind != CK_INVALID && IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
#define TUNE_PROC(ENUM, NAME) Values.emplace_back(StringRef(NAME));
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
}

// Get all features except standard extension feature
bool getCPUFeaturesExceptStdExt(CPUKind Kind,
                                std::vector<StringRef> &Features) {
  const CPUInfo &Info = RISCVCPUInfo[static_cast<unsigned>(Kind)];

  if (Info.isInvalid())
    return false;

  if (Info.is64Bit())
    Features.push_back("+64bit");
  else
    Features.push_back("-64bit");

  return true;
}

bool isX18ReservedByDefault(const Triple &TT) {
  // X18 is reserved for the ShadowCallStack ABI (even when not enabled).
  return TT.isOSFuchsia() || TT.isAndroid();
}

} // namespace RISCV
} // namespace llvm
