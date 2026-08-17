//===- lib/MC/MCTargetOptions.cpp - MC Target Options ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCTargetOptions.h"
#include "llvm/ADT/StringRef.h"
#include <climits>

using namespace llvm;

MCTargetOptions::MCTargetOptions() {
#define MC_TARGET_OPTION_INIT_BITFIELD(Type, Name, Bits, Default)              \
  Name = Default;
#define MC_TARGET_OPTION_INIT_BOOL(Type, Name, Bits, Default)
#define MC_TARGET_OPTION_INIT_ENUM(Type, Name, Bits, Default)
#define MC_TARGET_OPTION_INIT_OPTIONAL_UINT(Type, Name, Bits, Default)
#define MC_TARGET_OPTION_INIT_INT(Type, Name, Bits, Default)
#define MC_TARGET_OPTION_INIT_PAIR(Type, Name, Bits, Default)
#define MC_TARGET_OPTION_INIT_STRING(Type, Name, Bits, Default)
#define MC_TARGET_OPTION_INIT_STRING_LIST(Type, Name, Bits, Default)
#define MC_TARGET_OPTION(Type, Name, Bits, Default, Kind)                      \
  MC_TARGET_OPTION_INIT_##Kind(Type, Name, Bits, Default)
#include "llvm/MC/MCTargetOptions.def"
#undef MC_TARGET_OPTION_INIT_BITFIELD
#undef MC_TARGET_OPTION_INIT_BOOL
#undef MC_TARGET_OPTION_INIT_ENUM
#undef MC_TARGET_OPTION_INIT_OPTIONAL_UINT
#undef MC_TARGET_OPTION_INIT_INT
#undef MC_TARGET_OPTION_INIT_PAIR
#undef MC_TARGET_OPTION_INIT_STRING
#undef MC_TARGET_OPTION_INIT_STRING_LIST
}

std::pair<int, int> MCTargetOptions::parseBinutilsVersion(StringRef Version) {
  if (Version == "none")
    return {INT_MAX, INT_MAX}; // Make binutilsIsAtLeast() return true.
  std::pair<int, int> Ret;
  if (!Version.consumeInteger(10, Ret.first) && Version.consume_front("."))
    Version.consumeInteger(10, Ret.second);
  return Ret;
}

StringRef MCTargetOptions::getABIName() const { return ABIName; }

StringRef MCTargetOptions::getAssemblyLanguage() const {
  return AssemblyLanguage;
}
