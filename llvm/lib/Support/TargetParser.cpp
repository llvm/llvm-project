//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include <cctype>

using namespace llvm;
using namespace ARM;
using namespace AArch64;

namespace {

// List of canonical FPU names (use getFPUSynonym) and which architectural
// features they correspond to (use getFPUFeatures).
// FIXME: TableGen this.
// The entries must appear in the order listed in ARM::FPUKind for correct indexing
static const struct {
  const char *NameCStr;
  size_t NameLength;
  ARM::FPUKind ID;
  ARM::FPUVersion FPUVersion;
  ARM::NeonSupportLevel NeonSupport;
  ARM::FPURestriction Restriction;

  StringRef getName() const { return StringRef(NameCStr, NameLength); }
} FPUNames[] = {
#define ARM_FPU(NAME, KIND, VERSION, NEON_SUPPORT, RESTRICTION) \
  { NAME, sizeof(NAME) - 1, KIND, VERSION, NEON_SUPPORT, RESTRICTION },
#include "llvm/Support/ARMTargetParser.def"
};

// List of canonical arch names (use getArchSynonym).
// This table also provides the build attribute fields for CPU arch
// and Arch ID, according to the Addenda to the ARM ABI, chapters
// 2.4 and 2.3.5.2 respectively.
// FIXME: SubArch values were simplified to fit into the expectations
// of the triples and are not conforming with their official names.
// Check to see if the expectation should be changed.
// FIXME: TableGen this.
template <typename T> struct ArchNames {
  const char *NameCStr;
  size_t NameLength;
  const char *CPUAttrCStr;
  size_t CPUAttrLength;
  const char *SubArchCStr;
  size_t SubArchLength;
  unsigned DefaultFPU;
  unsigned ArchBaseExtensions;
  T ID;
  ARMBuildAttrs::CPUArch ArchAttr; // Arch ID in build attributes.

  StringRef getName() const { return StringRef(NameCStr, NameLength); }

  // CPU class in build attributes.
  StringRef getCPUAttr() const { return StringRef(CPUAttrCStr, CPUAttrLength); }

  // Sub-Arch name.
  StringRef getSubArch() const { return StringRef(SubArchCStr, SubArchLength); }
};
ArchNames<ARM::ArchKind> ARCHNames[] = {
#define ARM_ARCH(NAME, ID, CPU_ATTR, SUB_ARCH, ARCH_ATTR, ARCH_FPU, ARCH_BASE_EXT)       \
  {NAME, sizeof(NAME) - 1, CPU_ATTR, sizeof(CPU_ATTR) - 1, SUB_ARCH,       \
   sizeof(SUB_ARCH) - 1, ARCH_FPU, ARCH_BASE_EXT, ID, ARCH_ATTR},
#include "llvm/Support/ARMTargetParser.def"
};

ArchNames<AArch64::ArchKind> AArch64ARCHNames[] = {
 #define AARCH64_ARCH(NAME, ID, CPU_ATTR, SUB_ARCH, ARCH_ATTR, ARCH_FPU, ARCH_BASE_EXT)       \
   {NAME, sizeof(NAME) - 1, CPU_ATTR, sizeof(CPU_ATTR) - 1, SUB_ARCH,       \
    sizeof(SUB_ARCH) - 1, ARCH_FPU, ARCH_BASE_EXT, AArch64::ArchKind::ID, ARCH_ATTR},
 #include "llvm/Support/AArch64TargetParser.def"
 };


// List of Arch Extension names.
// FIXME: TableGen this.
static const struct {
  const char *NameCStr;
  size_t NameLength;
  unsigned ID;
  const char *Feature;
  const char *NegFeature;

  StringRef getName() const { return StringRef(NameCStr, NameLength); }
} ARCHExtNames[] = {
#define ARM_ARCH_EXT_NAME(NAME, ID, FEATURE, NEGFEATURE) \
  { NAME, sizeof(NAME) - 1, ID, FEATURE, NEGFEATURE },
#include "llvm/Support/ARMTargetParser.def"
},AArch64ARCHExtNames[] = {
#define AARCH64_ARCH_EXT_NAME(NAME, ID, FEATURE, NEGFEATURE) \
  { NAME, sizeof(NAME) - 1, ID, FEATURE, NEGFEATURE },
#include "llvm/Support/AArch64TargetParser.def"
};

// List of HWDiv names (use getHWDivSynonym) and which architectural
// features they correspond to (use getHWDivFeatures).
// FIXME: TableGen this.
static const struct {
  const char *NameCStr;
  size_t NameLength;
  unsigned ID;

  StringRef getName() const { return StringRef(NameCStr, NameLength); }
} HWDivNames[] = {
#define ARM_HW_DIV_NAME(NAME, ID) { NAME, sizeof(NAME) - 1, ID },
#include "llvm/Support/ARMTargetParser.def"
};

// List of CPU names and their arches.
// The same CPU can have multiple arches and can be default on multiple arches.
// When finding the Arch for a CPU, first-found prevails. Sort them accordingly.
// When this becomes table-generated, we'd probably need two tables.
// FIXME: TableGen this.
template <typename T> struct CpuNames {
  const char *NameCStr;
  size_t NameLength;
  T ArchID;
  bool Default; // is $Name the default CPU for $ArchID ?
  unsigned DefaultExtensions;

  StringRef getName() const { return StringRef(NameCStr, NameLength); }
};
CpuNames<ARM::ArchKind> CPUNames[] = {
#define ARM_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT) \
  { NAME, sizeof(NAME) - 1, ID, IS_DEFAULT, DEFAULT_EXT },
#include "llvm/Support/ARMTargetParser.def"
};

CpuNames<AArch64::ArchKind> AArch64CPUNames[] = {
 #define AARCH64_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT) \
   { NAME, sizeof(NAME) - 1, AArch64::ArchKind::ID, IS_DEFAULT, DEFAULT_EXT },
 #include "llvm/Support/AArch64TargetParser.def"
 };

} // namespace

// ======================================================= //
// Information by ID
// ======================================================= //

StringRef llvm::ARM::getFPUName(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return StringRef();
  return FPUNames[FPUKind].getName();
}

unsigned llvm::ARM::getFPUVersion(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return 0;
  return FPUNames[FPUKind].FPUVersion;
}

unsigned llvm::ARM::getFPUNeonSupportLevel(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return 0;
  return FPUNames[FPUKind].NeonSupport;
}

unsigned llvm::ARM::getFPURestriction(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return 0;
  return FPUNames[FPUKind].Restriction;
}

unsigned llvm::ARM::getDefaultFPU(StringRef CPU, unsigned ArchKind) {
  if (CPU == "generic")
    return ARCHNames[ArchKind].DefaultFPU;

  return StringSwitch<unsigned>(CPU)
#define ARM_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT) \
    .Case(NAME, DEFAULT_FPU)
#include "llvm/Support/ARMTargetParser.def"
    .Default(ARM::FK_INVALID);
}

unsigned llvm::ARM::getDefaultExtensions(StringRef CPU, unsigned ArchKind) {
  if (CPU == "generic")
    return ARCHNames[ArchKind].ArchBaseExtensions;

  return StringSwitch<unsigned>(CPU)
#define ARM_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT) \
    .Case(NAME, ARCHNames[ID].ArchBaseExtensions | DEFAULT_EXT)
#include "llvm/Support/ARMTargetParser.def"
    .Default(ARM::AEK_INVALID);
}

bool llvm::ARM::getHWDivFeatures(unsigned HWDivKind,
                                 std::vector<StringRef> &Features) {

  if (HWDivKind == ARM::AEK_INVALID)
    return false;

  if (HWDivKind & ARM::AEK_HWDIVARM)
    Features.push_back("+hwdiv-arm");
  else
    Features.push_back("-hwdiv-arm");

  if (HWDivKind & ARM::AEK_HWDIVTHUMB)
    Features.push_back("+hwdiv");
  else
    Features.push_back("-hwdiv");

  return true;
}

bool llvm::ARM::getExtensionFeatures(unsigned Extensions,
                                     std::vector<StringRef> &Features) {

  if (Extensions == ARM::AEK_INVALID)
    return false;

  if (Extensions & ARM::AEK_CRC)
    Features.push_back("+crc");
  else
    Features.push_back("-crc");

  if (Extensions & ARM::AEK_DSP)
    Features.push_back("+dsp");
  else
    Features.push_back("-dsp");

  return getHWDivFeatures(Extensions, Features);
}

bool llvm::ARM::getFPUFeatures(unsigned FPUKind,
                               std::vector<StringRef> &Features) {

  if (FPUKind >= ARM::FK_LAST || FPUKind == ARM::FK_INVALID)
    return false;

  // fp-only-sp and d16 subtarget features are independent of each other, so we
  // must enable/disable both.
  switch (FPUNames[FPUKind].Restriction) {
  case ARM::FR_SP_D16:
    Features.push_back("+fp-only-sp");
    Features.push_back("+d16");
    break;
  case ARM::FR_D16:
    Features.push_back("-fp-only-sp");
    Features.push_back("+d16");
    break;
  case ARM::FR_None:
    Features.push_back("-fp-only-sp");
    Features.push_back("-d16");
    break;
  }

  // FPU version subtarget features are inclusive of lower-numbered ones, so
  // enable the one corresponding to this version and disable all that are
  // higher. We also have to make sure to disable fp16 when vfp4 is disabled,
  // as +vfp4 implies +fp16 but -vfp4 does not imply -fp16.
  switch (FPUNames[FPUKind].FPUVersion) {
  case ARM::FV_VFPV5:
    Features.push_back("+fp-armv8");
    break;
  case ARM::FV_VFPV4:
    Features.push_back("+vfp4");
    Features.push_back("-fp-armv8");
    break;
  case ARM::FV_VFPV3_FP16:
    Features.push_back("+vfp3");
    Features.push_back("+fp16");
    Features.push_back("-vfp4");
    Features.push_back("-fp-armv8");
    break;
  case ARM::FV_VFPV3:
    Features.push_back("+vfp3");
    Features.push_back("-fp16");
    Features.push_back("-vfp4");
    Features.push_back("-fp-armv8");
    break;
  case ARM::FV_VFPV2:
    Features.push_back("+vfp2");
    Features.push_back("-vfp3");
    Features.push_back("-fp16");
    Features.push_back("-vfp4");
    Features.push_back("-fp-armv8");
    break;
  case ARM::FV_NONE:
    Features.push_back("-vfp2");
    Features.push_back("-vfp3");
    Features.push_back("-fp16");
    Features.push_back("-vfp4");
    Features.push_back("-fp-armv8");
    break;
  }

  // crypto includes neon, so we handle this similarly to FPU version.
  switch (FPUNames[FPUKind].NeonSupport) {
  case ARM::NS_Crypto:
    Features.push_back("+neon");
    Features.push_back("+crypto");
    break;
  case ARM::NS_Neon:
    Features.push_back("+neon");
    Features.push_back("-crypto");
    break;
  case ARM::NS_None:
    Features.push_back("-neon");
    Features.push_back("-crypto");
    break;
  }

  return true;
}

StringRef llvm::ARM::getArchName(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return StringRef();
  return ARCHNames[ArchKind].getName();
}

StringRef llvm::ARM::getCPUAttr(unsigned ArchKind) {
  if (ArchKind == ARM::AK_INVALID || ArchKind >= ARM::AK_LAST)
    return StringRef();
  return ARCHNames[ArchKind].getCPUAttr();
}

StringRef llvm::ARM::getSubArch(unsigned ArchKind) {
  if (ArchKind == ARM::AK_INVALID || ArchKind >= ARM::AK_LAST)
    return StringRef();
  return ARCHNames[ArchKind].getSubArch();
}

unsigned llvm::ARM::getArchAttr(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return ARMBuildAttrs::CPUArch::Pre_v4;
  return ARCHNames[ArchKind].ArchAttr;
}

StringRef llvm::ARM::getArchExtName(unsigned ArchExtKind) {
  for (const auto AE : ARCHExtNames) {
    if (ArchExtKind == AE.ID)
      return AE.getName();
  }
  return StringRef();
}

StringRef llvm::ARM::getArchExtFeature(StringRef ArchExt) {
  if (ArchExt.startswith("no")) {
    StringRef ArchExtBase(ArchExt.substr(2));
    for (const auto AE : ARCHExtNames) {
      if (AE.NegFeature && ArchExtBase == AE.getName())
        return StringRef(AE.NegFeature);
    }
  }
  for (const auto AE : ARCHExtNames) {
    if (AE.Feature && ArchExt == AE.getName())
      return StringRef(AE.Feature);
  }

  return StringRef();
}

StringRef llvm::ARM::getHWDivName(unsigned HWDivKind) {
  for (const auto D : HWDivNames) {
    if (HWDivKind == D.ID)
      return D.getName();
  }
  return StringRef();
}

StringRef llvm::ARM::getDefaultCPU(StringRef Arch) {
  unsigned AK = parseArch(Arch);
  if (AK == ARM::AK_INVALID)
    return StringRef();

  // Look for multiple AKs to find the default for pair AK+Name.
  for (const auto CPU : CPUNames) {
    if (CPU.ArchID == AK && CPU.Default)
      return CPU.getName();
  }

  // If we can't find a default then target the architecture instead
  return "generic";
}

StringRef llvm::AArch64::getFPUName(unsigned FPUKind) {
  return ARM::getFPUName(FPUKind);
}

unsigned llvm::AArch64::getFPUVersion(unsigned FPUKind) {
  return ARM::getFPUVersion(FPUKind);
}

unsigned llvm::AArch64::getFPUNeonSupportLevel(unsigned FPUKind) {
  return ARM::getFPUNeonSupportLevel( FPUKind);
}

unsigned llvm::AArch64::getFPURestriction(unsigned FPUKind) {
  return ARM::getFPURestriction(FPUKind);
}

unsigned llvm::AArch64::getDefaultFPU(StringRef CPU, unsigned ArchKind) {
  if (CPU == "generic")
    return AArch64ARCHNames[ArchKind].DefaultFPU;

  return StringSwitch<unsigned>(CPU)
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT) \
    .Case(NAME, DEFAULT_FPU)
#include "llvm/Support/AArch64TargetParser.def"
    .Default(ARM::FK_INVALID);
}

unsigned llvm::AArch64::getDefaultExtensions(StringRef CPU, unsigned ArchKind) {
  if (CPU == "generic")
    return AArch64ARCHNames[ArchKind].ArchBaseExtensions;

  return StringSwitch<unsigned>(CPU)
#define AARCH64_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT)       \
  .Case(NAME,                                                                  \
        AArch64ARCHNames[(unsigned)AArch64::ArchKind::ID].ArchBaseExtensions | \
            DEFAULT_EXT)
#include "llvm/Support/AArch64TargetParser.def"
    .Default(AArch64::AEK_INVALID);
}

bool llvm::AArch64::getExtensionFeatures(unsigned Extensions,
                                     std::vector<StringRef> &Features) {

  if (Extensions == AArch64::AEK_INVALID)
    return false;

  if (Extensions & AArch64::AEK_FP)
    Features.push_back("+fp-armv8");
  if (Extensions & AArch64::AEK_SIMD)
    Features.push_back("+neon");
  if (Extensions & AArch64::AEK_CRC)
    Features.push_back("+crc");
  if (Extensions & AArch64::AEK_CRYPTO)
    Features.push_back("+crypto");
  if (Extensions & AArch64::AEK_FP16)
    Features.push_back("+fullfp16");
  if (Extensions & AArch64::AEK_PROFILE)
    Features.push_back("+spe");
  if (Extensions & AArch64::AEK_RAS)
    Features.push_back("+ras");
  if (Extensions & AArch64::AEK_LSE)
    Features.push_back("+lse");

  return true;
}

bool llvm::AArch64::getFPUFeatures(unsigned FPUKind,
                               std::vector<StringRef> &Features) {
  return ARM::getFPUFeatures(FPUKind, Features);
}

bool llvm::AArch64::getArchFeatures(unsigned ArchKind,
                                     std::vector<StringRef> &Features) {
  if (ArchKind == static_cast<unsigned>(AArch64::ArchKind::AK_ARMV8_1A))
    Features.push_back("+v8.1a");
  if (ArchKind == static_cast<unsigned>(AArch64::ArchKind::AK_ARMV8_2A))
    Features.push_back("+v8.2a");

  return ArchKind > static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) &&
         ArchKind < static_cast<unsigned>(AArch64::ArchKind::AK_LAST);
}

StringRef llvm::AArch64::getArchName(unsigned ArchKind) {
  if (ArchKind >= static_cast<unsigned>(AArch64::ArchKind::AK_LAST))
    return StringRef();
  return AArch64ARCHNames[ArchKind].getName();
}

StringRef llvm::AArch64::getCPUAttr(unsigned ArchKind) {
  if (ArchKind == static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) ||
      ArchKind >= static_cast<unsigned>(AArch64::ArchKind::AK_LAST))
    return StringRef();
  return AArch64ARCHNames[ArchKind].getCPUAttr();
}

StringRef llvm::AArch64::getSubArch(unsigned ArchKind) {
  if (ArchKind == static_cast<unsigned>(AArch64::ArchKind::AK_INVALID) ||
      ArchKind >= static_cast<unsigned>(AArch64::ArchKind::AK_LAST))
    return StringRef();
  return AArch64ARCHNames[ArchKind].getSubArch();
}

unsigned llvm::AArch64::getArchAttr(unsigned ArchKind) {
  if (ArchKind >= static_cast<unsigned>(AArch64::ArchKind::AK_LAST))
    return ARMBuildAttrs::CPUArch::v8_A;
  return AArch64ARCHNames[ArchKind].ArchAttr;
}

StringRef llvm::AArch64::getArchExtName(unsigned ArchExtKind) {
  for (const auto &AE : AArch64ARCHExtNames)
    if (ArchExtKind == AE.ID)
      return AE.getName();
  return StringRef();
}

StringRef llvm::AArch64::getArchExtFeature(StringRef ArchExt) {
  if (ArchExt.startswith("no")) {
    StringRef ArchExtBase(ArchExt.substr(2));
    for (const auto &AE : AArch64ARCHExtNames) {
      if (AE.NegFeature && ArchExtBase == AE.getName())
        return StringRef(AE.NegFeature);
    }
  }

  for (const auto &AE : AArch64ARCHExtNames)
    if (AE.Feature && ArchExt == AE.getName())
      return StringRef(AE.Feature);
  return StringRef();
}

StringRef llvm::AArch64::getDefaultCPU(StringRef Arch) {
  unsigned AK = parseArch(Arch);
  if (AK == static_cast<unsigned>(AArch64::ArchKind::AK_INVALID))
    return StringRef();

  // Look for multiple AKs to find the default for pair AK+Name.
  for (const auto &CPU : AArch64CPUNames)
    if (static_cast<unsigned>(CPU.ArchID) == AK && CPU.Default)
      return CPU.getName();

  // If we can't find a default then target the architecture instead
  return "generic";
}

unsigned llvm::AArch64::checkArchVersion(StringRef Arch) {
  if (Arch[0] == 'v' && std::isdigit(Arch[1]))
    return (Arch[1] - 48);
  return 0;
}

// ======================================================= //
// Parsers
// ======================================================= //

static StringRef getHWDivSynonym(StringRef HWDiv) {
  return StringSwitch<StringRef>(HWDiv)
      .Case("thumb,arm", "arm,thumb")
      .Default(HWDiv);
}

static StringRef getFPUSynonym(StringRef FPU) {
  return StringSwitch<StringRef>(FPU)
      .Cases("fpa", "fpe2", "fpe3", "maverick", "invalid") // Unsupported
      .Case("vfp2", "vfpv2")
      .Case("vfp3", "vfpv3")
      .Case("vfp4", "vfpv4")
      .Case("vfp3-d16", "vfpv3-d16")
      .Case("vfp4-d16", "vfpv4-d16")
      .Cases("fp4-sp-d16", "vfpv4-sp-d16", "fpv4-sp-d16")
      .Cases("fp4-dp-d16", "fpv4-dp-d16", "vfpv4-d16")
      .Case("fp5-sp-d16", "fpv5-sp-d16")
      .Cases("fp5-dp-d16", "fpv5-dp-d16", "fpv5-d16")
      // FIXME: Clang uses it, but it's bogus, since neon defaults to vfpv3.
      .Case("neon-vfpv3", "neon")
      .Default(FPU);
}

static StringRef getArchSynonym(StringRef Arch) {
  return StringSwitch<StringRef>(Arch)
      .Case("v5", "v5t")
      .Case("v5e", "v5te")
      .Case("v6j", "v6")
      .Case("v6hl", "v6k")
      .Cases("v6m", "v6sm", "v6s-m", "v6-m")
      .Cases("v6z", "v6zk", "v6kz")
      .Cases("v7", "v7a", "v7hl", "v7l", "v7-a")
      .Case("v7r", "v7-r")
      .Case("v7m", "v7-m")
      .Case("v7em", "v7e-m")
      .Cases("v8", "v8a", "aarch64", "arm64", "v8-a")
      .Case("v8.1a", "v8.1-a")
      .Case("v8.2a", "v8.2-a")
      .Case("v8r", "v8-r")
      .Case("v8m.base", "v8-m.base")
      .Case("v8m.main", "v8-m.main")
      .Default(Arch);
}

// MArch is expected to be of the form (arm|thumb)?(eb)?(v.+)?(eb)?, but
// (iwmmxt|xscale)(eb)? is also permitted. If the former, return
// "v.+", if the latter, return unmodified string, minus 'eb'.
// If invalid, return empty string.
StringRef llvm::ARM::getCanonicalArchName(StringRef Arch) {
  size_t offset = StringRef::npos;
  StringRef A = Arch;
  StringRef Error = "";

  // Begins with "arm" / "thumb", move past it.
  if (A.startswith("arm64"))
    offset = 5;
  else if (A.startswith("arm"))
    offset = 3;
  else if (A.startswith("thumb"))
    offset = 5;
  else if (A.startswith("aarch64")) {
    offset = 7;
    // AArch64 uses "_be", not "eb" suffix.
    if (A.find("eb") != StringRef::npos)
      return Error;
    if (A.substr(offset, 3) == "_be")
      offset += 3;
  }

  // Ex. "armebv7", move past the "eb".
  if (offset != StringRef::npos && A.substr(offset, 2) == "eb")
    offset += 2;
  // Or, if it ends with eb ("armv7eb"), chop it off.
  else if (A.endswith("eb"))
    A = A.substr(0, A.size() - 2);
  // Trim the head
  if (offset != StringRef::npos)
    A = A.substr(offset);

  // Empty string means offset reached the end, which means it's valid.
  if (A.empty())
    return Arch;

  // Only match non-marketing names
  if (offset != StringRef::npos) {
    // Must start with 'vN'.
    if (A[0] != 'v' || !std::isdigit(A[1]))
      return Error;
    // Can't have an extra 'eb'.
    if (A.find("eb") != StringRef::npos)
      return Error;
  }

  // Arch will either be a 'v' name (v7a) or a marketing name (xscale).
  return A;
}

unsigned llvm::ARM::parseHWDiv(StringRef HWDiv) {
  StringRef Syn = getHWDivSynonym(HWDiv);
  for (const auto D : HWDivNames) {
    if (Syn == D.getName())
      return D.ID;
  }
  return ARM::AEK_INVALID;
}

unsigned llvm::ARM::parseFPU(StringRef FPU) {
  StringRef Syn = getFPUSynonym(FPU);
  for (const auto F : FPUNames) {
    if (Syn == F.getName())
      return F.ID;
  }
  return ARM::FK_INVALID;
}

// Allows partial match, ex. "v7a" matches "armv7a".
unsigned llvm::ARM::parseArch(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  StringRef Syn = getArchSynonym(Arch);
  for (const auto A : ARCHNames) {
    if (A.getName().endswith(Syn))
      return A.ID;
  }
  return ARM::AK_INVALID;
}

unsigned llvm::ARM::parseArchExt(StringRef ArchExt) {
  for (const auto A : ARCHExtNames) {
    if (ArchExt == A.getName())
      return A.ID;
  }
  return ARM::AEK_INVALID;
}

unsigned llvm::ARM::parseCPUArch(StringRef CPU) {
  for (const auto C : CPUNames) {
    if (CPU == C.getName())
      return C.ArchID;
  }
  return ARM::AK_INVALID;
}

// ARM, Thumb, AArch64
unsigned llvm::ARM::parseArchISA(StringRef Arch) {
  return StringSwitch<unsigned>(Arch)
      .StartsWith("aarch64", ARM::IK_AARCH64)
      .StartsWith("arm64", ARM::IK_AARCH64)
      .StartsWith("thumb", ARM::IK_THUMB)
      .StartsWith("arm", ARM::IK_ARM)
      .Default(ARM::IK_INVALID);
}

// Little/Big endian
unsigned llvm::ARM::parseArchEndian(StringRef Arch) {
  if (Arch.startswith("armeb") || Arch.startswith("thumbeb") ||
      Arch.startswith("aarch64_be"))
    return ARM::EK_BIG;

  if (Arch.startswith("arm") || Arch.startswith("thumb")) {
    if (Arch.endswith("eb"))
      return ARM::EK_BIG;
    else
      return ARM::EK_LITTLE;
  }

  if (Arch.startswith("aarch64"))
    return ARM::EK_LITTLE;

  return ARM::EK_INVALID;
}

// Profile A/R/M
unsigned llvm::ARM::parseArchProfile(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  switch (parseArch(Arch)) {
  case ARM::AK_ARMV6M:
  case ARM::AK_ARMV7M:
  case ARM::AK_ARMV7EM:
  case ARM::AK_ARMV8MMainline:
  case ARM::AK_ARMV8MBaseline:
    return ARM::PK_M;
  case ARM::AK_ARMV7R:
  case ARM::AK_ARMV8R:
    return ARM::PK_R;
  case ARM::AK_ARMV7A:
  case ARM::AK_ARMV7VE:
  case ARM::AK_ARMV7K:
  case ARM::AK_ARMV8A:
  case ARM::AK_ARMV8_1A:
  case ARM::AK_ARMV8_2A:
    return ARM::PK_A;
  }
  return ARM::PK_INVALID;
}

// Version number (ex. v7 = 7).
unsigned llvm::ARM::parseArchVersion(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  switch (parseArch(Arch)) {
  case ARM::AK_ARMV2:
  case ARM::AK_ARMV2A:
    return 2;
  case ARM::AK_ARMV3:
  case ARM::AK_ARMV3M:
    return 3;
  case ARM::AK_ARMV4:
  case ARM::AK_ARMV4T:
    return 4;
  case ARM::AK_ARMV5T:
  case ARM::AK_ARMV5TE:
  case ARM::AK_IWMMXT:
  case ARM::AK_IWMMXT2:
  case ARM::AK_XSCALE:
  case ARM::AK_ARMV5TEJ:
    return 5;
  case ARM::AK_ARMV6:
  case ARM::AK_ARMV6K:
  case ARM::AK_ARMV6T2:
  case ARM::AK_ARMV6KZ:
  case ARM::AK_ARMV6M:
    return 6;
  case ARM::AK_ARMV7A:
  case ARM::AK_ARMV7VE:
  case ARM::AK_ARMV7R:
  case ARM::AK_ARMV7M:
  case ARM::AK_ARMV7S:
  case ARM::AK_ARMV7EM:
  case ARM::AK_ARMV7K:
    return 7;
  case ARM::AK_ARMV8A:
  case ARM::AK_ARMV8_1A:
  case ARM::AK_ARMV8_2A:
  case ARM::AK_ARMV8R:
  case ARM::AK_ARMV8MBaseline:
  case ARM::AK_ARMV8MMainline:
    return 8;
  }
  return 0;
}

StringRef llvm::ARM::computeDefaultTargetABI(const Triple &TT, StringRef CPU) {
  StringRef ArchName =
      CPU.empty() ? TT.getArchName() : ARM::getArchName(ARM::parseCPUArch(CPU));

  if (TT.isOSBinFormatMachO()) {
    if (TT.getEnvironment() == Triple::EABI ||
        TT.getOS() == Triple::UnknownOS ||
        llvm::ARM::parseArchProfile(ArchName) == ARM::PK_M)
      return "aapcs";
    if (TT.isWatchABI())
      return "aapcs16";
    return "apcs-gnu";
  } else if (TT.isOSWindows())
    // FIXME: this is invalid for WindowsCE.
    return "aapcs";

  // Select the default based on the platform.
  switch (TT.getEnvironment()) {
  case Triple::Android:
  case Triple::GNUEABI:
  case Triple::GNUEABIHF:
  case Triple::MuslEABI:
  case Triple::MuslEABIHF:
    return "aapcs-linux";
  case Triple::EABIHF:
  case Triple::EABI:
    return "aapcs";
  default:
    if (TT.isOSNetBSD())
      return "apcs-gnu";
    if (TT.isOSOpenBSD())
      return "aapcs-linux";
    return "aapcs";
  }
}

StringRef llvm::AArch64::getCanonicalArchName(StringRef Arch) {
  return ARM::getCanonicalArchName(Arch);
}

unsigned llvm::AArch64::parseFPU(StringRef FPU) {
  return ARM::parseFPU(FPU);
}

// Allows partial match, ex. "v8a" matches "armv8a".
unsigned llvm::AArch64::parseArch(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  if (checkArchVersion(Arch) < 8)
    return static_cast<unsigned>(AArch64::ArchKind::AK_INVALID);

  StringRef Syn = getArchSynonym(Arch);
  for (const auto A : AArch64ARCHNames) {
    if (A.getName().endswith(Syn))
      return static_cast<unsigned>(A.ID);
  }
  return static_cast<unsigned>(AArch64::ArchKind::AK_INVALID);
}

unsigned llvm::AArch64::parseArchExt(StringRef ArchExt) {
  for (const auto A : AArch64ARCHExtNames) {
    if (ArchExt == A.getName())
      return A.ID;
  }
  return AArch64::AEK_INVALID;
}

unsigned llvm::AArch64::parseCPUArch(StringRef CPU) {
  for (const auto C : AArch64CPUNames) {
    if (CPU == C.getName())
      return static_cast<unsigned>(C.ArchID);
  }
  return static_cast<unsigned>(AArch64::ArchKind::AK_INVALID);
}

// ARM, Thumb, AArch64
unsigned llvm::AArch64::parseArchISA(StringRef Arch) {
  return ARM::parseArchISA(Arch);
}

// Little/Big endian
unsigned llvm::AArch64::parseArchEndian(StringRef Arch) {
  return ARM::parseArchEndian(Arch);
}

// Profile A/R/M
unsigned llvm::AArch64::parseArchProfile(StringRef Arch) {
  return ARM::parseArchProfile(Arch);
}

// Version number (ex. v8 = 8).
unsigned llvm::AArch64::parseArchVersion(StringRef Arch) {
  return ARM::parseArchVersion(Arch);
}
