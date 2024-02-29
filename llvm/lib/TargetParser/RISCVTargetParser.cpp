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
#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace RISCV {

enum CPUKind : unsigned {
#define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_UNALIGN) CK_##ENUM,
#define TUNE_PROC(ENUM, NAME) CK_##ENUM,
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
};

struct CPUInfo {
  StringLiteral Name;
  StringLiteral DefaultMarch;
  bool FastUnalignedAccess;
  bool is64Bit() const { return DefaultMarch.starts_with("rv64"); }
};

constexpr CPUInfo RISCVCPUInfo[] = {
#define PROC(ENUM, NAME, DEFAULT_MARCH, FAST_UNALIGN)                          \
  {NAME, DEFAULT_MARCH, FAST_UNALIGN},
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
};

static const CPUInfo *getCPUInfoByName(StringRef CPU) {
  for (auto &C : RISCVCPUInfo)
    if (C.Name == CPU)
      return &C;
  return nullptr;
}

bool hasFastUnalignedAccess(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  return Info && Info->FastUnalignedAccess;
}

bool parseCPU(StringRef CPU, bool IsRV64) {
  const CPUInfo *Info = getCPUInfoByName(CPU);

  if (!Info)
    return false;
  return Info->is64Bit() == IsRV64;
}

bool parseTuneCPU(StringRef TuneCPU, bool IsRV64) {
  std::optional<CPUKind> Kind =
      llvm::StringSwitch<std::optional<CPUKind>>(TuneCPU)
#define TUNE_PROC(ENUM, NAME) .Case(NAME, CK_##ENUM)
  #include "llvm/TargetParser/RISCVTargetParserDef.inc"
      .Default(std::nullopt);

  if (Kind.has_value())
    return true;

  // Fallback to parsing as a CPU.
  return parseCPU(TuneCPU, IsRV64);
}

StringRef getMArchFromMcpu(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  if (!Info)
    return "";
  return Info->DefaultMarch;
}

void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
}

void fillValidTuneCPUArchList(SmallVectorImpl<StringRef> &Values, bool IsRV64) {
  for (const auto &C : RISCVCPUInfo) {
    if (IsRV64 == C.is64Bit())
      Values.emplace_back(C.Name);
  }
#define TUNE_PROC(ENUM, NAME) Values.emplace_back(StringRef(NAME));
#include "llvm/TargetParser/RISCVTargetParserDef.inc"
}

// This function is currently used by IREE, so it's not dead code.
void getFeaturesForCPU(StringRef CPU,
                       SmallVectorImpl<std::string> &EnabledFeatures,
                       bool NeedPlus) {
  StringRef MarchFromCPU = llvm::RISCV::getMArchFromMcpu(CPU);
  if (MarchFromCPU == "")
    return;

  EnabledFeatures.clear();
  auto RII = RISCVISAInfo::parseArchString(
      MarchFromCPU, /* EnableExperimentalExtension */ true);

  if (llvm::errorToBool(RII.takeError()))
    return;

  std::vector<std::string> FeatStrings =
      (*RII)->toFeatures(/* AddAllExtensions */ false);
  for (const auto &F : FeatStrings)
    if (NeedPlus)
      EnabledFeatures.push_back(F);
    else
      EnabledFeatures.push_back(F.substr(1));
}
} // namespace RISCV
} // namespace llvm
