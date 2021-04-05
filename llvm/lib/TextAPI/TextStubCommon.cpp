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
#include "llvm/ADT/StringSwitch.h"

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

void ScalarTraits<PlatformSet>::output(const PlatformSet &Values, void *IO,
                                       raw_ostream &OS) {

  const auto *Ctx = reinterpret_cast<TextAPIContext *>(IO);
  assert((!Ctx || Ctx->FileKind != FileType::Invalid) &&
         "File type is not set in context");

  if (Ctx && Ctx->FileKind == TBD_V3 && Values.count(PlatformKind::macOS) &&
      Values.count(PlatformKind::macCatalyst)) {
    OS << "zippered";
    return;
  }

  assert(Values.size() == 1U);
  switch (*Values.begin()) {
  default:
    llvm_unreachable("unexpected platform");
    break;
  case PlatformKind::macOS:
    OS << "macosx";
    break;
  case PlatformKind::iOSSimulator:
    LLVM_FALLTHROUGH;
  case PlatformKind::iOS:
    OS << "ios";
    break;
  case PlatformKind::watchOSSimulator:
    LLVM_FALLTHROUGH;
  case PlatformKind::watchOS:
    OS << "watchos";
    break;
  case PlatformKind::tvOSSimulator:
    LLVM_FALLTHROUGH;
  case PlatformKind::tvOS:
    OS << "tvos";
    break;
  case PlatformKind::bridgeOS:
    OS << "bridgeos";
    break;
  case PlatformKind::macCatalyst:
    OS << "iosmac";
    break;
  case PlatformKind::driverKit:
    OS << "driverkit";
    break;
  }
}

StringRef ScalarTraits<PlatformSet>::input(StringRef Scalar, void *IO,
                                           PlatformSet &Values) {
  const auto *Ctx = reinterpret_cast<TextAPIContext *>(IO);
  assert((!Ctx || Ctx->FileKind != FileType::Invalid) &&
         "File type is not set in context");

  if (Scalar == "zippered") {
    if (Ctx && Ctx->FileKind == FileType::TBD_V3) {
      Values.insert(PlatformKind::macOS);
      Values.insert(PlatformKind::macCatalyst);
      return {};
    }
    return "invalid platform";
  }

  auto Platform = StringSwitch<PlatformKind>(Scalar)
                      .Case("unknown", PlatformKind::unknown)
                      .Case("macosx", PlatformKind::macOS)
                      .Case("ios", PlatformKind::iOS)
                      .Case("watchos", PlatformKind::watchOS)
                      .Case("tvos", PlatformKind::tvOS)
                      .Case("bridgeos", PlatformKind::bridgeOS)
                      .Case("iosmac", PlatformKind::macCatalyst)
                      .Default(PlatformKind::unknown);

  if (Platform == PlatformKind::macCatalyst)
    if (Ctx && Ctx->FileKind != FileType::TBD_V3)
      return "invalid platform";

  if (Platform == PlatformKind::unknown)
    return "unknown platform";

  Values.insert(Platform);
  return {};
}

QuotingType ScalarTraits<PlatformSet>::mustQuote(StringRef) {
  return QuotingType::None;
}

void ScalarBitSetTraits<ArchitectureSet>::bitset(IO &IO,
                                                 ArchitectureSet &Archs) {
#define ARCHINFO(arch, type, subtype, numbits)                                 \
  IO.bitSetCase(Archs, #arch, 1U << static_cast<int>(AK_##arch));
#include "llvm/TextAPI/Architecture.def"
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
StringRef ScalarTraits<SwiftVersion>::input(StringRef Scalar, void *IO,
                                            SwiftVersion &Value) {
  const auto *Ctx = reinterpret_cast<TextAPIContext *>(IO);
  assert((!Ctx || Ctx->FileKind != FileType::Invalid) &&
         "File type is not set in context");

  if (Ctx->FileKind == FileType::TBD_V4) {
    if (Scalar.getAsInteger(10, Value))
      return "invalid Swift ABI version.";
    return {};
  } else {
    Value = StringSwitch<SwiftVersion>(Scalar)
                .Case("1.0", 1)
                .Case("1.1", 2)
                .Case("2.0", 3)
                .Case("3.0", 4)
                .Default(0);
  }

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
  Value.second = std::string(UUID);
  Value.first = Target{getArchitectureFromName(Arch), PlatformKind::unknown};
  return {};
}

QuotingType ScalarTraits<UUID>::mustQuote(StringRef) {
  return QuotingType::Single;
}

} // end namespace yaml.
} // end namespace llvm.
