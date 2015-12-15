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
static const struct {
  const char *NameCStr;
  size_t NameLength;
  const char *CPUAttrCStr;
  size_t CPUAttrLength;
  const char *SubArchCStr;
  size_t SubArchLength;
  unsigned DefaultFPU;
  unsigned ArchBaseExtensions;
  ARM::ArchKind ID;
  ARMBuildAttrs::CPUArch ArchAttr; // Arch ID in build attributes.

  StringRef getName() const { return StringRef(NameCStr, NameLength); }

  // CPU class in build attributes.
  StringRef getCPUAttr() const { return StringRef(CPUAttrCStr, CPUAttrLength); }

  // Sub-Arch name.
  StringRef getSubArch() const { return StringRef(SubArchCStr, SubArchLength); }
} ARCHNames[] = {
#define ARM_ARCH(NAME, ID, CPU_ATTR, SUB_ARCH, ARCH_ATTR, ARCH_FPU, ARCH_BASE_EXT)       \
  {NAME, sizeof(NAME) - 1, CPU_ATTR, sizeof(CPU_ATTR) - 1, SUB_ARCH,       \
   sizeof(SUB_ARCH) - 1, ARCH_FPU, ARCH_BASE_EXT, ID, ARCH_ATTR},
#include "llvm/Support/ARMTargetParser.def"
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
static const struct {
  const char *NameCStr;
  size_t NameLength;
  ARM::ArchKind ArchID;
  bool Default; // is $Name the default CPU for $ArchID ?
  unsigned DefaultExtensions;

  StringRef getName() const { return StringRef(NameCStr, NameLength); }
} CPUNames[] = {
#define ARM_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT) \
  { NAME, sizeof(NAME) - 1, ID, IS_DEFAULT, DEFAULT_EXT },
#include "llvm/Support/ARMTargetParser.def"
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
                                 std::vector<const char *> &Features) {

  if (HWDivKind == ARM::AEK_INVALID)
    return false;

  if (HWDivKind & ARM::AEK_HWDIVARM)
    Features.push_back("+hwdiv-arm");
  else
    Features.push_back("-hwdiv-arm");

  if (HWDivKind & ARM::AEK_HWDIV)
    Features.push_back("+hwdiv");
  else
    Features.push_back("-hwdiv");

  return true;
}

bool llvm::ARM::getExtensionFeatures(unsigned Extensions,
                                     std::vector<const char *> &Features) {

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
                               std::vector<const char *> &Features) {

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
  if (ArchKind >= ARM::AK_LAST)
    return StringRef();
  return ARCHNames[ArchKind].getCPUAttr();
}

StringRef llvm::ARM::getSubArch(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
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

const char *llvm::ARM::getArchExtFeature(StringRef ArchExt) {
  if (ArchExt.startswith("no")) {
    StringRef ArchExtBase(ArchExt.substr(2));
    for (const auto AE : ARCHExtNames) {
      if (AE.NegFeature && ArchExtBase == AE.getName())
        return AE.NegFeature;
    }
  }
  for (const auto AE : ARCHExtNames) {
    if (AE.Feature && ArchExt == AE.getName())
      return AE.Feature;
  }

  return nullptr;
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
      .Default(ARM::EK_INVALID);
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
    return ARM::PK_M;
  case ARM::AK_ARMV7R:
    return ARM::PK_R;
  case ARM::AK_ARMV7A:
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
  case ARM::AK_ARMV7R:
  case ARM::AK_ARMV7M:
  case ARM::AK_ARMV7S:
  case ARM::AK_ARMV7EM:
  case ARM::AK_ARMV7K:
    return 7;
  case ARM::AK_ARMV8A:
  case ARM::AK_ARMV8_1A:
  case ARM::AK_ARMV8_2A:
    return 8;
  }
  return 0;
}
