//===-- PISATargetParser.h - PISA target parsing defines ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_PISATARGETPARSER_H
#define LLVM_TARGETPARSER_PISATARGETPARSER_H

#include "TargetParser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace PISA {

// last known PISA version
constexpr unsigned LatestPISAVersion = 100; // PISA 1.0

enum class PISATargetVariant : unsigned {
  VariantNone = 0,
};

struct PISATargetInfo {
  StringRef Name;
  unsigned Gen;
  PISATargetVariant Variant;
  bool FwdCompat;
};

inline PISATargetInfo getPISATargetInfo(StringRef Name) {
  static constexpr PISATargetInfo DefaultInfo = {
      "", 0, PISATargetVariant::VariantNone, false};
  static constexpr PISATargetInfo Info[] = {
#define PISA_TARGET(NAME, GEN, VARIANT, FWDCOMPAT)                             \
  {NAME, GEN, PISATargetVariant::VARIANT, FWDCOMPAT},
#include "PISATargetParser.def"
#undef PISA_TARGET
  };
  auto *It = llvm::find_if(Info, [&Name](const PISATargetInfo &Entry) {
    return Entry.Name == Name;
  });
  return It == std::end(Info) ? DefaultInfo : *It;
}

// Defined as a macro (rather than only a StringRef) so the bare target names
// from PISATargetParser.def can be concatenated into full CPU names as
// compile-time string literals in fillValidCPUList(); PISACPUPrefix is derived
// from it so the "igca_" literal lives in exactly one place.
#define PISA_CPU_PREFIX "igca_"
inline constexpr StringRef PISACPUPrefix = PISA_CPU_PREFIX;

inline StringRef stripCPUPrefix(StringRef Name) {
  Name.consume_front(PISACPUPrefix);
  return Name;
}

// Full CPU name (with the "igca_" prefix) used as the default device when no
// -mcpu/-march is specified on the command-line. This is the single source of
// truth for the default PISA target; the backend subtarget uses the bare name
// via stripCPUPrefix(), while the Clang driver passes the prefixed name to
// cc1's -target-cpu.
inline StringRef getDefaultCPUName() { return PISA_CPU_PREFIX "100"; }

inline bool isValidCPU(StringRef Name) {
  if (!Name.consume_front(PISACPUPrefix))
    return false;
  return !getPISATargetInfo(Name).Name.empty();
}

inline void fillValidCPUList(SmallVectorImpl<StringRef> &Values) {
  Values.append({
#define PISA_TARGET(NAME, GEN, VARIANT, FWDCOMPAT) PISA_CPU_PREFIX NAME,
#include "PISATargetParser.def"
#undef PISA_TARGET
  });
}
#undef PISA_CPU_PREFIX

// Fills Features with the full, transitively-implied default feature set for
// CPU, as declared by the Proc<>/SubtargetFeature Implies lists in
// PISAFeatures.td/PISA.td (see PISATargetParser.cpp) -- this keeps the
// CPU->feature expansion clang sees in sync with the real subtarget's
// TableGen-generated expansion, with the .td file as the single source of
// truth for both.
LLVM_ABI void fillFeatureMap(StringRef CPU, StringMap<bool> &Features);

// Check for compatible PISATargetInfo
// - TInfo - PISA target specified on command-line via -mcpu=
// - IInfo - instruction PISA target encoded in .td files
inline bool isCompatiblePISATargetInfo(const PISATargetInfo &TInfo,
                                       const PISATargetInfo &IInfo) {
  if (TInfo.Gen == 0)
    return false; // no -mcpu specified
  if (IInfo.Gen == 0)
    return false; // no instruction target
  if (IInfo.Gen > TInfo.Gen)
    return false; // future instruction
  if ((IInfo.Variant != TInfo.Variant) &&
      (IInfo.Variant != PISATargetVariant::VariantNone))
    return false;       // variant mismatch
  if (!IInfo.FwdCompat) // exact match required
    return (IInfo.Gen == TInfo.Gen) && (IInfo.Variant == TInfo.Variant) &&
           !TInfo.FwdCompat;
  return true;
}

} // namespace PISA
} // namespace llvm

#endif // LLVM_TARGETPARSER_PISATARGETPARSER_H
