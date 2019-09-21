//===- llvm/TextAPI/MachO/Platform.cpp - Platform ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementations of Platform Helper functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/TextAPI/MachO/Platform.h"

namespace llvm {
namespace MachO {

PlatformKind mapToPlatformKind(const Triple &Target) {
  switch (Target.getOS()) {
  default:
    return PlatformKind::unknown;
  case Triple::MacOSX:
    return PlatformKind::macOS;
  case Triple::IOS:
    return PlatformKind::iOS;
  case Triple::TvOS:
    return PlatformKind::tvOS;
  case Triple::WatchOS:
    return PlatformKind::watchOS;
    // TODO: add bridgeOS once in llvm::Triple
  }
}

PlatformSet mapToPlatformSet(ArrayRef<Triple> Targets) {
  PlatformSet Result;
  for (const auto &Target : Targets)
    Result.insert(mapToPlatformKind(Target));
  return Result;
}

StringRef getPlatformName(PlatformKind Platform) {
  switch (Platform) {
  case PlatformKind::unknown:
    return "unknown";
  case PlatformKind::macOS:
    return "macOS";
  case PlatformKind::iOS:
    return "iOS";
  case PlatformKind::tvOS:
    return "tvOS";
  case PlatformKind::watchOS:
    return "watchOS";
  case PlatformKind::bridgeOS:
    return "bridgeOS";
  }
  llvm_unreachable("Unknown llvm.MachO.PlatformKind enum");
}

} // end namespace MachO.
} // end namespace llvm.
