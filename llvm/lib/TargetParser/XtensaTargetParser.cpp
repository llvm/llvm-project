//==-- XtensaTargetParser - Parser for Xtensa features ------------*- C++ -*-=//
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

#include "llvm/TargetParser/XtensaTargetParser.h"
#include "llvm/ADT/Enum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <vector>

namespace llvm {

namespace Xtensa {
struct CPUInfo {
  CPUKind Kind;
  uint64_t Features;
};

constexpr EnumStringDef<uint64_t> FeatureNameDefs[] = {
#define XTENSA_FEATURE(ID, NAME) {{NAME}, ID},
#include "llvm/TargetParser/XtensaTargetParser.def"
};
constexpr auto FeatureNames = BUILD_ENUM_STRINGS(FeatureNameDefs);

constexpr EnumStringDef<CPUInfo> CPUInfoDefs[] = {
#define XTENSA_CPU(ENUM, NAME, FEATURES) {{NAME}, {CK_##ENUM, FEATURES}},
#include "llvm/TargetParser/XtensaTargetParser.def"
};
constexpr auto CPUInfos = BUILD_ENUM_STRINGS(CPUInfoDefs);

StringRef getBaseName(StringRef CPU) {
  return llvm::StringSwitch<StringRef>(CPU)
#define XTENSA_CPU_ALIAS(NAME, ANAME) .Case(ANAME, NAME)
#include "llvm/TargetParser/XtensaTargetParser.def"
      .Default(CPU);
}

StringRef getAliasName(StringRef CPU) {
  return llvm::StringSwitch<StringRef>(CPU)
#define XTENSA_CPU_ALIAS(NAME, ANAME) .Case(NAME, ANAME)
#include "llvm/TargetParser/XtensaTargetParser.def"
      .Default(CPU);
}

CPUKind parseCPUKind(StringRef CPU) {
  CPU = getBaseName(CPU);
  for (const auto &C : CPUInfos)
    if (C.name() == CPU)
      return C.value().Kind;
  return CK_INVALID;
}

// Get all features for the CPU
void getCPUFeatures(StringRef CPU, std::vector<StringRef> &Features) {
  CPU = getBaseName(CPU);
  auto I =
      llvm::find_if(CPUInfos, [&](const auto &CI) { return CI.name() == CPU; });
  assert(I != std::end(CPUInfos) && "CPU not found!");
  uint64_t Bits = I->value().Features;

  for (const auto &F : FeatureNames) {
    if ((Bits & F.value()) == F.value())
      Features.push_back(F.name());
  }
}

// Find all valid CPUs
void fillValidCPUList(std::vector<StringRef> &Values) {
  for (const auto &C : CPUInfos) {
    if (C.value().Kind != CK_INVALID) {
      Values.emplace_back(C.name());
      StringRef Name = getAliasName(C.name());
      if (Name != C.name())
        Values.emplace_back(Name);
    }
  }
}

} // namespace Xtensa
} // namespace llvm
