//===- Target.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/Target.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace MachO {

Expected<Target> Target::create(StringRef TargetValue) {
  auto Result = TargetValue.split('-');
  auto ArchitectureStr = Result.first;
  auto Architecture = getArchitectureFromName(ArchitectureStr);
  auto PlatformStr = Result.second;
  PlatformKind Platform;
  Platform = StringSwitch<PlatformKind>(PlatformStr)
                 .Case("macos", PlatformKind::macOS)
                 .Case("ios", PlatformKind::iOS)
                 .Case("tvos", PlatformKind::tvOS)
                 .Case("watchos", PlatformKind::watchOS)
                 .Case("bridgeos", PlatformKind::bridgeOS)
                 .Case("maccatalyst", PlatformKind::macCatalyst)
                 .Case("ios-simulator", PlatformKind::iOSSimulator)
                 .Case("tvos-simulator", PlatformKind::tvOSSimulator)
                 .Case("watchos-simulator", PlatformKind::watchOSSimulator)
                 .Case("driverkit", PlatformKind::driverKit)
                 .Default(PlatformKind::unknown);

  if (Platform == PlatformKind::unknown) {
    if (PlatformStr.startswith("<") && PlatformStr.endswith(">")) {
      PlatformStr = PlatformStr.drop_front().drop_back();
      unsigned long long RawValue;
      if (!PlatformStr.getAsInteger(10, RawValue))
        Platform = (PlatformKind)RawValue;
    }
  }

  return Target{Architecture, Platform};
}

Target::operator std::string() const {
  return (getArchitectureName(Arch) + " (" + getPlatformName(Platform) + ")")
      .str();
}

raw_ostream &operator<<(raw_ostream &OS, const Target &Target) {
  OS << std::string(Target);
  return OS;
}

PlatformSet mapToPlatformSet(ArrayRef<Target> Targets) {
  PlatformSet Result;
  for (const auto &Target : Targets)
    Result.insert(Target.Platform);
  return Result;
}

ArchitectureSet mapToArchitectureSet(ArrayRef<Target> Targets) {
  ArchitectureSet Result;
  for (const auto &Target : Targets)
    Result.set(Target.Arch);
  return Result;
}

std::string getTargetTripleName(const Target &Targ) {
  return (getArchitectureName(Targ.Arch) + "-apple-" +
          getOSAndEnvironmentName(Targ.Platform))
      .str();
}

} // end namespace MachO.
} // end namespace llvm.
