//===- TextStubCommon.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implememts common Text Stub YAML mappings.
//
//===----------------------------------------------------------------------===//

#include "TextStubCommon.h"
#include "TextAPIContext.h"

using namespace llvm::MachO;

namespace llvm {
namespace yaml {

void ScalarTraits<FlowStringRef>::output(const FlowStringRef &Value, void *Ctx,
                                         raw_ostream &OS) {
  ScalarTraits<StringRef>::output(Value, Ctx, OS);
}
StringRef ScalarTraits<FlowStringRef>::input(StringRef Value, void *Ctx,
                                             FlowStringRef &Out) {
  return ScalarTraits<StringRef>::input(Value, Ctx, Out.value);
}
QuotingType ScalarTraits<FlowStringRef>::mustQuote(StringRef Name) {
  return ScalarTraits<StringRef>::mustQuote(Name);
}

void ScalarEnumerationTraits<ObjCConstraintType>::enumeration(
    IO &IO, ObjCConstraintType &Constraint) {
  IO.enumCase(Constraint, "none", ObjCConstraintType::None);
  IO.enumCase(Constraint, "retain_release", ObjCConstraintType::Retain_Release);
  IO.enumCase(Constraint, "retain_release_for_simulator",
              ObjCConstraintType::Retain_Release_For_Simulator);
  IO.enumCase(Constraint, "retain_release_or_gc",
              ObjCConstraintType::Retain_Release_Or_GC);
  IO.enumCase(Constraint, "gc", ObjCConstraintType::GC);
}

void ScalarTraits<PlatformKind>::output(const PlatformKind &Value, void *,
                                        raw_ostream &OS) {
  switch (Value) {
  default:
    llvm_unreachable("unexpected platform");
    break;
  case PlatformKind::macOS:
    OS << "macosx";
    break;
  case PlatformKind::iOS:
    OS << "ios";
    break;
  case PlatformKind::watchOS:
    OS << "watchos";
    break;
  case PlatformKind::tvOS:
    OS << "tvos";
    break;
  case PlatformKind::bridgeOS:
    OS << "bridgeos";
    break;
  }
}
StringRef ScalarTraits<PlatformKind>::input(StringRef Scalar, void *,
                                            PlatformKind &Value) {
  Value = StringSwitch<PlatformKind>(Scalar)
              .Case("macosx", PlatformKind::macOS)
              .Case("ios", PlatformKind::iOS)
              .Case("watchos", PlatformKind::watchOS)
              .Case("tvos", PlatformKind::tvOS)
              .Case("bridgeos", PlatformKind::bridgeOS)
              .Default(PlatformKind::unknown);

  if (Value == PlatformKind::unknown)
    return "unknown platform";
  return {};
}
QuotingType ScalarTraits<PlatformKind>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarBitSetTraits<ArchitectureSet>::bitset(IO &IO,
                                                 ArchitectureSet &Archs) {
#define ARCHINFO(arch, type, subtype)                                          \
  IO.bitSetCase(Archs, #arch, 1U << static_cast<int>(AK_##arch));
#include "llvm/TextAPI/MachO/Architecture.def"
#undef ARCHINFO
}

void ScalarTraits<Architecture>::output(const Architecture &Value, void *,
                                        raw_ostream &OS) {
  OS << Value;
}
StringRef ScalarTraits<Architecture>::input(StringRef Scalar, void *,
                                            Architecture &Value) {
  Value = getArchitectureFromName(Scalar);
  return {};
}
QuotingType ScalarTraits<Architecture>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarTraits<PackedVersion>::output(const PackedVersion &Value, void *,
                                         raw_ostream &OS) {
  OS << Value;
}
StringRef ScalarTraits<PackedVersion>::input(StringRef Scalar, void *,
                                             PackedVersion &Value) {
  if (!Value.parse32(Scalar))
    return "invalid packed version string.";
  return {};
}
QuotingType ScalarTraits<PackedVersion>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarTraits<SwiftVersion>::output(const SwiftVersion &Value, void *,
                                        raw_ostream &OS) {
  switch (Value) {
  case 1:
    OS << "1.0";
    break;
  case 2:
    OS << "1.1";
    break;
  case 3:
    OS << "2.0";
    break;
  case 4:
    OS << "3.0";
    break;
  default:
    OS << (unsigned)Value;
    break;
  }
}
StringRef ScalarTraits<SwiftVersion>::input(StringRef Scalar, void *,
                                            SwiftVersion &Value) {
  Value = StringSwitch<SwiftVersion>(Scalar)
              .Case("1.0", 1)
              .Case("1.1", 2)
              .Case("2.0", 3)
              .Case("3.0", 4)
              .Default(0);
  if (Value != SwiftVersion(0))
    return {};

  if (Scalar.getAsInteger(10, Value))
    return "invalid Swift ABI version.";

  return StringRef();
}
QuotingType ScalarTraits<SwiftVersion>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarTraits<UUID>::output(const UUID &Value, void *, raw_ostream &OS) {
  OS << Value.first << ": " << Value.second;
}
StringRef ScalarTraits<UUID>::input(StringRef Scalar, void *, UUID &Value) {
  auto Split = Scalar.split(':');
  auto Arch = Split.first.trim();
  auto UUID = Split.second.trim();
  if (UUID.empty())
    return "invalid uuid string pair";
  Value.first = getArchitectureFromName(Arch);
  Value.second = UUID;
  return {};
}
QuotingType ScalarTraits<UUID>::mustQuote(StringRef) {
  return QuotingType::Single;
}

} // end namespace yaml.
} // end namespace llvm.
