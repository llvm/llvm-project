//===---- PPCTargetParser.cpp - Parser for target features ------*- C++ -*-===//
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

#include "llvm/TargetParser/PPCTargetParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Host.h"

namespace llvm {
namespace PPC {

struct CPUInfo {
  StringLiteral Name;
  // FIXME: add the features field for this CPU.
};

constexpr CPUInfo PPCCPUInfo[] = {
#define PPC_CPU(Name, Linux_SUPPORT_METHOD, LinuxID, AIX_SUPPORT_METHOD,       \
                AIXID)                                                         \
  {Name},
#include "llvm/TargetParser/PPCTargetParser.def"
};

static const CPUInfo *getCPUInfoByName(StringRef CPU) {
  for (auto &C : PPCCPUInfo)
    if (C.Name == CPU)
      return &C;
  return nullptr;
}

StringRef normalizeCPUName(StringRef CPUName) {
  // Clang/LLVM does not actually support code generation
  // for the 405 CPU. However, there are uses of this CPU ID
  // in projects that previously used GCC and rely on Clang
  // accepting it. Clang has always ignored it and passed the
  // generic CPU ID to the back end.
  return StringSwitch<StringRef>(CPUName)
      .Cases("common", "405", "generic")
      .Cases("ppc440", "440fp", "440")
      .Cases("630", "power3", "pwr3")
      .Case("G3", "g3")
      .Case("G4", "g4")
      .Case("G4+", "g4+")
      .Case("8548", "e500")
      .Case("ppc970", "970")
      .Case("G5", "g5")
      .Case("ppca2", "a2")
      .Case("power4", "pwr4")
      .Case("power5", "pwr5")
      .Case("power5x", "pwr5x")
      .Case("power5+", "pwr5+")
      .Case("power6", "pwr6")
      .Case("power6x", "pwr6x")
      .Case("power7", "pwr7")
      .Case("power8", "pwr8")
      .Case("power9", "pwr9")
      .Case("power10", "pwr10")
      .Case("power11", "pwr11")
      .Cases("powerpc", "powerpc32", "ppc")
      .Case("powerpc64", "ppc64")
      .Case("powerpc64le", "ppc64le")
      .Default(CPUName);
}

void fillValidCPUList(SmallVectorImpl<StringRef> &Values) {
  for (const auto &C : PPCCPUInfo)
    Values.emplace_back(C.Name);
}

void fillValidTuneCPUList(SmallVectorImpl<StringRef> &Values) {
  for (const auto &C : PPCCPUInfo)
    Values.emplace_back(C.Name);
}

bool isValidCPU(StringRef CPU) {
  const CPUInfo *Info = getCPUInfoByName(CPU);
  if (!Info)
    return false;
  return true;
}

StringRef getNormalizedPPCTargetCPU(const Triple &T, StringRef CPUName) {
  if (!CPUName.empty()) {
    if (CPUName == "native") {
      StringRef CPU = sys::getHostCPUName();
      if (!CPU.empty() && CPU != "generic")
        return CPU;
    }

    StringRef CPU = normalizeCPUName(CPUName);
    if (CPU != "generic" && CPU != "native")
      return CPU;
  }

  // LLVM may default to generating code for the native CPU, but, like gcc, we
  // default to a more generic option for each architecture. (except on AIX)
  if (T.isOSAIX())
    return "pwr7";
  else if (T.getArch() == Triple::ppc64le)
    return "ppc64le";
  else if (T.getArch() == Triple::ppc64)
    return "ppc64";

  return "ppc";
}

StringRef getNormalizedPPCTuneCPU(const Triple &T, StringRef CPUName) {
  return getNormalizedPPCTargetCPU(T, CPUName);
}

} // namespace PPC
} // namespace llvm
