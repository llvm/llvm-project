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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

namespace llvm {

namespace Xtensa {
struct CPUInfo {
  StringLiteral Name;
  CPUKind Kind;
  uint64_t Features;
};

struct FeatureName {
  uint64_t ID;
  const char *NameCStr;
  size_t NameLength;

  StringRef getName() const { return StringRef(NameCStr, NameLength); }
};

const FeatureName XtensaFeatureNames[] = {
#define XTENSA_FEATURE(ID, NAME) {ID, "+" NAME, sizeof(NAME)},
#include "llvm/TargetParser/XtensaTargetParser.def"
};

constexpr CPUInfo XtensaCPUInfo[] = {
#define XTENSA_CPU(ENUM, NAME, FEATURES) {NAME, CK_##ENUM, FEATURES},
#include "llvm/TargetParser/XtensaTargetParser.def"
};

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
  return llvm::StringSwitch<CPUKind>(CPU)
#define XTENSA_CPU(ENUM, NAME, FEATURES) .Case(NAME, CK_##ENUM)
#include "llvm/TargetParser/XtensaTargetParser.def"
      .Default(CK_INVALID);
}

// Get all features for the CPU
void getCPUFeatures(StringRef CPU, std::vector<StringRef> &Features) {
  CPU = getBaseName(CPU);
  auto I = llvm::find_if(XtensaCPUInfo,
                         [&](const CPUInfo &CI) { return CI.Name == CPU; });
  assert(I != std::end(XtensaCPUInfo) && "CPU not found!");
  uint64_t Bits = I->Features;

  for (const auto &F : XtensaFeatureNames) {
    if ((Bits & F.ID) == F.ID)
      Features.push_back(F.getName());
  }
}

// Find all valid CPUs
void fillValidCPUList(std::vector<StringRef> &Values) {
  for (const auto &C : XtensaCPUInfo) {
    if (C.Kind != CK_INVALID) {
      Values.emplace_back(C.Name);
      StringRef Name = getAliasName(C.Name);
      if (Name != C.Name)
        Values.emplace_back(Name);
    }
  }
}

} // namespace Xtensa
} // namespace llvm
