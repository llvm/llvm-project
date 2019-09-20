//===- Architecture.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the architecture helper functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/MachO/Architecture.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/MachO.h"

namespace llvm {
namespace MachO {

Architecture getArchitectureFromCpuType(uint32_t CPUType, uint32_t CPUSubType) {
#define ARCHINFO(Arch, Type, Subtype)                                          \
  if (CPUType == (Type) &&                                                     \
      (CPUSubType & ~MachO::CPU_SUBTYPE_MASK) == (Subtype))                    \
    return AK_##Arch;
#include "llvm/TextAPI/MachO/Architecture.def"
#undef ARCHINFO

  return AK_unknown;
}

Architecture getArchitectureFromName(StringRef Name) {
  return StringSwitch<Architecture>(Name)
#define ARCHINFO(Arch, Type, Subtype) .Case(#Arch, AK_##Arch)
#include "llvm/TextAPI/MachO/Architecture.def"
#undef ARCHINFO
      .Default(AK_unknown);
}

StringRef getArchitectureName(Architecture Arch) {
  switch (Arch) {
#define ARCHINFO(Arch, Type, Subtype)                                          \
  case AK_##Arch:                                                              \
    return #Arch;
#include "llvm/TextAPI/MachO/Architecture.def"
#undef ARCHINFO
  case AK_unknown:
    return "unknown";
  }

  // Appease some compilers that cannot figure out that this is a fully covered
  // switch statement.
  return "unknown";
}

std::pair<uint32_t, uint32_t> getCPUTypeFromArchitecture(Architecture Arch) {
  switch (Arch) {
#define ARCHINFO(Arch, Type, Subtype)                                          \
  case AK_##Arch:                                                              \
    return std::make_pair(Type, Subtype);
#include "llvm/TextAPI/MachO/Architecture.def"
#undef ARCHINFO
  case AK_unknown:
    return std::make_pair(0, 0);
  }

  // Appease some compilers that cannot figure out that this is a fully covered
  // switch statement.
  return std::make_pair(0, 0);
}

Architecture mapToArchitecture(const Triple &Target) {
  return getArchitectureFromName(Target.getArchName());
}

raw_ostream &operator<<(raw_ostream &OS, Architecture Arch) {
  OS << getArchitectureName(Arch);
  return OS;
}

} // end namespace MachO.
} // end namespace llvm.
