//===- llvm/TextAPI/Target.h - TAPI Target ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_MACHO_TARGET_H
#define LLVM_TEXTAPI_MACHO_TARGET_H

#include "llvm/ADT/Triple.h"
#include "llvm/Support/Error.h"
#include "llvm/TextAPI/Architecture.h"
#include "llvm/TextAPI/ArchitectureSet.h"
#include "llvm/TextAPI/Platform.h"

namespace llvm {
namespace MachO {

// This is similar to a llvm Triple, but the triple doesn't have all the
// information we need. For example there is no enum value for x86_64h. The
// only way to get that information is to parse the triple string.
class Target {
public:
  Target() = default;
  Target(Architecture Arch, PlatformKind Platform)
      : Arch(Arch), Platform(Platform) {}
  explicit Target(const llvm::Triple &Triple)
      : Arch(mapToArchitecture(Triple)), Platform(mapToPlatformKind(Triple)) {}

  static llvm::Expected<Target> create(StringRef Target);

  operator std::string() const;

  Architecture Arch;
  PlatformKind Platform;
};

inline bool operator==(const Target &LHS, const Target &RHS) {
  return std::tie(LHS.Arch, LHS.Platform) == std::tie(RHS.Arch, RHS.Platform);
}

inline bool operator!=(const Target &LHS, const Target &RHS) {
  return std::tie(LHS.Arch, LHS.Platform) != std::tie(RHS.Arch, RHS.Platform);
}

inline bool operator<(const Target &LHS, const Target &RHS) {
  return std::tie(LHS.Arch, LHS.Platform) < std::tie(RHS.Arch, RHS.Platform);
}

inline bool operator==(const Target &LHS, const Architecture &RHS) {
  return LHS.Arch == RHS;
}

inline bool operator!=(const Target &LHS, const Architecture &RHS) {
  return LHS.Arch != RHS;
}

PlatformSet mapToPlatformSet(ArrayRef<Target> Targets);
ArchitectureSet mapToArchitectureSet(ArrayRef<Target> Targets);

std::string getTargetTripleName(const Target &Targ);

raw_ostream &operator<<(raw_ostream &OS, const Target &Target);

} // namespace MachO
} // namespace llvm

#endif // LLVM_TEXTAPI_MACHO_TARGET_H
