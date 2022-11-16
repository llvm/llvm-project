//===-- ARMTargetParser - Parser for ARM target features --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise ARM hardware features
// such as FPU/CPU/ARCH/extensions and specific support such as HWDIV.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ARMTargetParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/ARMTargetParserCommon.h"
#include <cctype>

using namespace llvm;

static StringRef getHWDivSynonym(StringRef HWDiv) {
  return StringSwitch<StringRef>(HWDiv)
      .Case("thumb,arm", "arm,thumb")
      .Default(HWDiv);
}

// Allows partial match, ex. "v7a" matches "armv7a".
ARM::ArchKind ARM::parseArch(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  StringRef Syn = getArchSynonym(Arch);
  for (const auto &A : ARCHNames) {
    if (A.getName().endswith(Syn))
      return A.ID;
  }
  return ArchKind::INVALID;
}

// Version number (ex. v7 = 7).
unsigned ARM::parseArchVersion(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  switch (parseArch(Arch)) {
  case ArchKind::ARMV4:
  case ArchKind::ARMV4T:
    return 4;
  case ArchKind::ARMV5T:
  case ArchKind::ARMV5TE:
  case ArchKind::IWMMXT:
  case ArchKind::IWMMXT2:
  case ArchKind::XSCALE:
  case ArchKind::ARMV5TEJ:
    return 5;
  case ArchKind::ARMV6:
  case ArchKind::ARMV6K:
  case ArchKind::ARMV6T2:
  case ArchKind::ARMV6KZ:
  case ArchKind::ARMV6M:
    return 6;
  case ArchKind::ARMV7A:
  case ArchKind::ARMV7VE:
  case ArchKind::ARMV7R:
  case ArchKind::ARMV7M:
  case ArchKind::ARMV7S:
  case ArchKind::ARMV7EM:
  case ArchKind::ARMV7K:
    return 7;
  case ArchKind::ARMV8A:
  case ArchKind::ARMV8_1A:
  case ArchKind::ARMV8_2A:
  case ArchKind::ARMV8_3A:
  case ArchKind::ARMV8_4A:
  case ArchKind::ARMV8_5A:
  case ArchKind::ARMV8_6A:
  case ArchKind::ARMV8_7A:
  case ArchKind::ARMV8_8A:
  case ArchKind::ARMV8_9A:
  case ArchKind::ARMV8R:
  case ArchKind::ARMV8MBaseline:
  case ArchKind::ARMV8MMainline:
  case ArchKind::ARMV8_1MMainline:
    return 8;
  case ArchKind::ARMV9A:
  case ArchKind::ARMV9_1A:
  case ArchKind::ARMV9_2A:
  case ArchKind::ARMV9_3A:
  case ArchKind::ARMV9_4A:
    return 9;
  case ArchKind::INVALID:
    return 0;
  }
  llvm_unreachable("Unhandled architecture");
}

static ARM::ProfileKind getProfileKind(ARM::ArchKind AK) {
  switch (AK) {
  case ARM::ArchKind::ARMV6M:
  case ARM::ArchKind::ARMV7M:
  case ARM::ArchKind::ARMV7EM:
  case ARM::ArchKind::ARMV8MMainline:
  case ARM::ArchKind::ARMV8MBaseline:
  case ARM::ArchKind::ARMV8_1MMainline:
    return ARM::ProfileKind::M;
  case ARM::ArchKind::ARMV7R:
  case ARM::ArchKind::ARMV8R:
    return ARM::ProfileKind::R;
  case ARM::ArchKind::ARMV7A:
  case ARM::ArchKind::ARMV7VE:
  case ARM::ArchKind::ARMV7K:
  case ARM::ArchKind::ARMV8A:
  case ARM::ArchKind::ARMV8_1A:
  case ARM::ArchKind::ARMV8_2A:
  case ARM::ArchKind::ARMV8_3A:
  case ARM::ArchKind::ARMV8_4A:
  case ARM::ArchKind::ARMV8_5A:
  case ARM::ArchKind::ARMV8_6A:
  case ARM::ArchKind::ARMV8_7A:
  case ARM::ArchKind::ARMV8_8A:
  case ARM::ArchKind::ARMV8_9A:
  case ARM::ArchKind::ARMV9A:
  case ARM::ArchKind::ARMV9_1A:
  case ARM::ArchKind::ARMV9_2A:
  case ARM::ArchKind::ARMV9_3A:
  case ARM::ArchKind::ARMV9_4A:
    return ARM::ProfileKind::A;
  case ARM::ArchKind::ARMV4:
  case ARM::ArchKind::ARMV4T:
  case ARM::ArchKind::ARMV5T:
  case ARM::ArchKind::ARMV5TE:
  case ARM::ArchKind::ARMV5TEJ:
  case ARM::ArchKind::ARMV6:
  case ARM::ArchKind::ARMV6K:
  case ARM::ArchKind::ARMV6T2:
  case ARM::ArchKind::ARMV6KZ:
  case ARM::ArchKind::ARMV7S:
  case ARM::ArchKind::IWMMXT:
  case ARM::ArchKind::IWMMXT2:
  case ARM::ArchKind::XSCALE:
  case ARM::ArchKind::INVALID:
    return ARM::ProfileKind::INVALID;
  }
  llvm_unreachable("Unhandled architecture");
}

// Profile A/R/M
ARM::ProfileKind ARM::parseArchProfile(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  return getProfileKind(parseArch(Arch));
}

bool ARM::getFPUFeatures(unsigned FPUKind, std::vector<StringRef> &Features) {

  if (FPUKind >= FK_LAST || FPUKind == FK_INVALID)
    return false;

  static const struct FPUFeatureNameInfo {
    const char *PlusName, *MinusName;
    FPUVersion MinVersion;
    FPURestriction MaxRestriction;
  } FPUFeatureInfoList[] = {
    // We have to specify the + and - versions of the name in full so
    // that we can return them as static StringRefs.
    //
    // Also, the SubtargetFeatures ending in just "sp" are listed here
    // under FPURestriction::None, which is the only FPURestriction in
    // which they would be valid (since FPURestriction::SP doesn't
    // exist).
    {"+vfp2", "-vfp2", FPUVersion::VFPV2, FPURestriction::D16},
    {"+vfp2sp", "-vfp2sp", FPUVersion::VFPV2, FPURestriction::SP_D16},
    {"+vfp3", "-vfp3", FPUVersion::VFPV3, FPURestriction::None},
    {"+vfp3d16", "-vfp3d16", FPUVersion::VFPV3, FPURestriction::D16},
    {"+vfp3d16sp", "-vfp3d16sp", FPUVersion::VFPV3, FPURestriction::SP_D16},
    {"+vfp3sp", "-vfp3sp", FPUVersion::VFPV3, FPURestriction::None},
    {"+fp16", "-fp16", FPUVersion::VFPV3_FP16, FPURestriction::SP_D16},
    {"+vfp4", "-vfp4", FPUVersion::VFPV4, FPURestriction::None},
    {"+vfp4d16", "-vfp4d16", FPUVersion::VFPV4, FPURestriction::D16},
    {"+vfp4d16sp", "-vfp4d16sp", FPUVersion::VFPV4, FPURestriction::SP_D16},
    {"+vfp4sp", "-vfp4sp", FPUVersion::VFPV4, FPURestriction::None},
    {"+fp-armv8", "-fp-armv8", FPUVersion::VFPV5, FPURestriction::None},
    {"+fp-armv8d16", "-fp-armv8d16", FPUVersion::VFPV5, FPURestriction::D16},
    {"+fp-armv8d16sp", "-fp-armv8d16sp", FPUVersion::VFPV5, FPURestriction::SP_D16},
    {"+fp-armv8sp", "-fp-armv8sp", FPUVersion::VFPV5, FPURestriction::None},
    {"+fullfp16", "-fullfp16", FPUVersion::VFPV5_FULLFP16, FPURestriction::SP_D16},
    {"+fp64", "-fp64", FPUVersion::VFPV2, FPURestriction::D16},
    {"+d32", "-d32", FPUVersion::VFPV3, FPURestriction::None},
  };

  for (const auto &Info: FPUFeatureInfoList) {
    if (FPUNames[FPUKind].FPUVer >= Info.MinVersion &&
        FPUNames[FPUKind].Restriction <= Info.MaxRestriction)
      Features.push_back(Info.PlusName);
    else
      Features.push_back(Info.MinusName);
  }

  static const struct NeonFeatureNameInfo {
    const char *PlusName, *MinusName;
    NeonSupportLevel MinSupportLevel;
  } NeonFeatureInfoList[] = {
      {"+neon", "-neon", NeonSupportLevel::Neon},
      {"+sha2", "-sha2", NeonSupportLevel::Crypto},
      {"+aes", "-aes", NeonSupportLevel::Crypto},
  };

  for (const auto &Info: NeonFeatureInfoList) {
    if (FPUNames[FPUKind].NeonSupport >= Info.MinSupportLevel)
      Features.push_back(Info.PlusName);
    else
      Features.push_back(Info.MinusName);
  }

  return true;
}

// Little/Big endian
ARM::EndianKind ARM::parseArchEndian(StringRef Arch) {
  if (Arch.startswith("armeb") || Arch.startswith("thumbeb") ||
      Arch.startswith("aarch64_be"))
    return EndianKind::BIG;

  if (Arch.startswith("arm") || Arch.startswith("thumb")) {
    if (Arch.endswith("eb"))
      return EndianKind::BIG;
    else
      return EndianKind::LITTLE;
  }

  if (Arch.startswith("aarch64") || Arch.startswith("aarch64_32"))
    return EndianKind::LITTLE;

  return EndianKind::INVALID;
}

// ARM, Thumb, AArch64
ARM::ISAKind ARM::parseArchISA(StringRef Arch) {
  return StringSwitch<ISAKind>(Arch)
      .StartsWith("aarch64", ISAKind::AARCH64)
      .StartsWith("arm64", ISAKind::AARCH64)
      .StartsWith("thumb", ISAKind::THUMB)
      .StartsWith("arm", ISAKind::ARM)
      .Default(ISAKind::INVALID);
}

unsigned ARM::parseFPU(StringRef FPU) {
  StringRef Syn = getFPUSynonym(FPU);
  for (const auto &F : FPUNames) {
    if (Syn == F.getName())
      return F.ID;
  }
  return FK_INVALID;
}

ARM::NeonSupportLevel ARM::getFPUNeonSupportLevel(unsigned FPUKind) {
  if (FPUKind >= FK_LAST)
    return NeonSupportLevel::None;
  return FPUNames[FPUKind].NeonSupport;
}

StringRef ARM::getFPUSynonym(StringRef FPU) {
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

StringRef ARM::getFPUName(unsigned FPUKind) {
  if (FPUKind >= FK_LAST)
    return StringRef();
  return FPUNames[FPUKind].getName();
}

ARM::FPUVersion ARM::getFPUVersion(unsigned FPUKind) {
  if (FPUKind >= FK_LAST)
    return FPUVersion::NONE;
  return FPUNames[FPUKind].FPUVer;
}

ARM::FPURestriction ARM::getFPURestriction(unsigned FPUKind) {
  if (FPUKind >= FK_LAST)
    return FPURestriction::None;
  return FPUNames[FPUKind].Restriction;
}

unsigned ARM::getDefaultFPU(StringRef CPU, ARM::ArchKind AK) {
  if (CPU == "generic")
    return ARM::ARCHNames[static_cast<unsigned>(AK)].DefaultFPU;

  return StringSwitch<unsigned>(CPU)
#define ARM_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT)           \
  .Case(NAME, DEFAULT_FPU)
#include "llvm/Support/ARMTargetParser.def"
   .Default(ARM::FK_INVALID);
}

uint64_t ARM::getDefaultExtensions(StringRef CPU, ARM::ArchKind AK) {
  if (CPU == "generic")
    return ARM::ARCHNames[static_cast<unsigned>(AK)].ArchBaseExtensions;

  return StringSwitch<uint64_t>(CPU)
#define ARM_CPU_NAME(NAME, ID, DEFAULT_FPU, IS_DEFAULT, DEFAULT_EXT)           \
  .Case(NAME,                                                                  \
        ARCHNames[static_cast<unsigned>(ArchKind::ID)].ArchBaseExtensions |    \
            DEFAULT_EXT)
#include "llvm/Support/ARMTargetParser.def"
  .Default(ARM::AEK_INVALID);
}

bool ARM::getHWDivFeatures(uint64_t HWDivKind,
                           std::vector<StringRef> &Features) {

  if (HWDivKind == AEK_INVALID)
    return false;

  if (HWDivKind & AEK_HWDIVARM)
    Features.push_back("+hwdiv-arm");
  else
    Features.push_back("-hwdiv-arm");

  if (HWDivKind & AEK_HWDIVTHUMB)
    Features.push_back("+hwdiv");
  else
    Features.push_back("-hwdiv");

  return true;
}

bool ARM::getExtensionFeatures(uint64_t Extensions,
                               std::vector<StringRef> &Features) {

  if (Extensions == AEK_INVALID)
    return false;

  for (const auto &AE : ARCHExtNames) {
    if ((Extensions & AE.ID) == AE.ID && AE.Feature)
      Features.push_back(AE.Feature);
    else if (AE.NegFeature)
      Features.push_back(AE.NegFeature);
  }

  return getHWDivFeatures(Extensions, Features);
}

StringRef ARM::getArchName(ARM::ArchKind AK) {
  return ARCHNames[static_cast<unsigned>(AK)].getName();
}

StringRef ARM::getCPUAttr(ARM::ArchKind AK) {
  return ARCHNames[static_cast<unsigned>(AK)].getCPUAttr();
}

StringRef ARM::getSubArch(ARM::ArchKind AK) {
  return ARCHNames[static_cast<unsigned>(AK)].getSubArch();
}

unsigned ARM::getArchAttr(ARM::ArchKind AK) {
  return ARCHNames[static_cast<unsigned>(AK)].ArchAttr;
}

StringRef ARM::getArchExtName(uint64_t ArchExtKind) {
  for (const auto &AE : ARCHExtNames) {
    if (ArchExtKind == AE.ID)
      return AE.getName();
  }
  return StringRef();
}

static bool stripNegationPrefix(StringRef &Name) {
  if (Name.startswith("no")) {
    Name = Name.substr(2);
    return true;
  }
  return false;
}

StringRef ARM::getArchExtFeature(StringRef ArchExt) {
  bool Negated = stripNegationPrefix(ArchExt);
  for (const auto &AE : ARCHExtNames) {
    if (AE.Feature && ArchExt == AE.getName())
      return StringRef(Negated ? AE.NegFeature : AE.Feature);
  }

  return StringRef();
}

static unsigned findDoublePrecisionFPU(unsigned InputFPUKind) {
  const ARM::FPUName &InputFPU = ARM::FPUNames[InputFPUKind];

  // If the input FPU already supports double-precision, then there
  // isn't any different FPU we can return here.
  //
  // The current available FPURestriction values are None (no
  // restriction), D16 (only 16 d-regs) and SP_D16 (16 d-regs
  // and single precision only); there's no value representing
  // SP restriction without D16. So this test just means 'is it
  // SP only?'.
  if (InputFPU.Restriction != ARM::FPURestriction::SP_D16)
    return ARM::FK_INVALID;

  // Otherwise, look for an FPU entry with all the same fields, except
  // that SP_D16 has been replaced with just D16, representing adding
  // double precision and not changing anything else.
  for (const ARM::FPUName &CandidateFPU : ARM::FPUNames) {
    if (CandidateFPU.FPUVer == InputFPU.FPUVer &&
        CandidateFPU.NeonSupport == InputFPU.NeonSupport &&
        CandidateFPU.Restriction == ARM::FPURestriction::D16) {
      return CandidateFPU.ID;
    }
  }

  // nothing found
  return ARM::FK_INVALID;
}

bool ARM::appendArchExtFeatures(StringRef CPU, ARM::ArchKind AK,
                                StringRef ArchExt,
                                std::vector<StringRef> &Features,
                                unsigned &ArgFPUID) {

  size_t StartingNumFeatures = Features.size();
  const bool Negated = stripNegationPrefix(ArchExt);
  uint64_t ID = parseArchExt(ArchExt);

  if (ID == AEK_INVALID)
    return false;

  for (const auto &AE : ARCHExtNames) {
    if (Negated) {
      if ((AE.ID & ID) == ID && AE.NegFeature)
        Features.push_back(AE.NegFeature);
    } else {
      if ((AE.ID & ID) == AE.ID && AE.Feature)
        Features.push_back(AE.Feature);
    }
  }

  if (CPU == "")
    CPU = "generic";

  if (ArchExt == "fp" || ArchExt == "fp.dp") {
    unsigned FPUKind;
    if (ArchExt == "fp.dp") {
      if (Negated) {
        Features.push_back("-fp64");
        return true;
      }
      FPUKind = findDoublePrecisionFPU(getDefaultFPU(CPU, AK));
    } else if (Negated) {
      FPUKind = ARM::FK_NONE;
    } else {
      FPUKind = getDefaultFPU(CPU, AK);
    }
    ArgFPUID = FPUKind;
    return ARM::getFPUFeatures(FPUKind, Features);
  }
  return StartingNumFeatures != Features.size();
}

ARM::ArchKind ARM::convertV9toV8(ARM::ArchKind AK) {
  if (getProfileKind(AK) != ProfileKind::A)
    return ARM::ArchKind::INVALID;
  if (AK < ARM::ArchKind::ARMV9A || AK > ARM::ArchKind::ARMV9_3A)
    return ARM::ArchKind::INVALID;
  unsigned AK_v8 = static_cast<unsigned>(ARM::ArchKind::ARMV8_5A);
  AK_v8 += static_cast<unsigned>(AK) -
           static_cast<unsigned>(ARM::ArchKind::ARMV9A);
  return static_cast<ARM::ArchKind>(AK_v8);
}

StringRef ARM::getDefaultCPU(StringRef Arch) {
  ArchKind AK = parseArch(Arch);
  if (AK == ArchKind::INVALID)
    return StringRef();

  // Look for multiple AKs to find the default for pair AK+Name.
  for (const auto &CPU : CPUNames) {
    if (CPU.ArchID == AK && CPU.Default)
      return CPU.getName();
  }

  // If we can't find a default then target the architecture instead
  return "generic";
}

uint64_t ARM::parseHWDiv(StringRef HWDiv) {
  StringRef Syn = getHWDivSynonym(HWDiv);
  for (const auto &D : HWDivNames) {
    if (Syn == D.getName())
      return D.ID;
  }
  return AEK_INVALID;
}

uint64_t ARM::parseArchExt(StringRef ArchExt) {
  for (const auto &A : ARCHExtNames) {
    if (ArchExt == A.getName())
      return A.ID;
  }
  return AEK_INVALID;
}

ARM::ArchKind ARM::parseCPUArch(StringRef CPU) {
  for (const auto &C : CPUNames) {
    if (CPU == C.getName())
      return C.ArchID;
  }
  return ArchKind::INVALID;
}

void ARM::fillValidCPUArchList(SmallVectorImpl<StringRef> &Values) {
  for (const CpuNames<ArchKind> &Arch : CPUNames) {
    if (Arch.ArchID != ArchKind::INVALID)
      Values.push_back(Arch.getName());
  }
}

StringRef ARM::computeDefaultTargetABI(const Triple &TT, StringRef CPU) {
  StringRef ArchName =
      CPU.empty() ? TT.getArchName() : getArchName(parseCPUArch(CPU));

  if (TT.isOSBinFormatMachO()) {
    if (TT.getEnvironment() == Triple::EABI ||
        TT.getOS() == Triple::UnknownOS ||
        parseArchProfile(ArchName) == ProfileKind::M)
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

StringRef ARM::getARMCPUForArch(const llvm::Triple &Triple, StringRef MArch) {
  if (MArch.empty())
    MArch = Triple.getArchName();
  MArch = llvm::ARM::getCanonicalArchName(MArch);

  // Some defaults are forced.
  switch (Triple.getOS()) {
  case llvm::Triple::FreeBSD:
  case llvm::Triple::NetBSD:
  case llvm::Triple::OpenBSD:
    if (!MArch.empty() && MArch == "v6")
      return "arm1176jzf-s";
    if (!MArch.empty() && MArch == "v7")
      return "cortex-a8";
    break;
  case llvm::Triple::Win32:
    // FIXME: this is invalid for WindowsCE
    if (llvm::ARM::parseArchVersion(MArch) <= 7)
      return "cortex-a9";
    break;
  case llvm::Triple::IOS:
  case llvm::Triple::MacOSX:
  case llvm::Triple::TvOS:
  case llvm::Triple::WatchOS:
  case llvm::Triple::DriverKit:
    if (MArch == "v7k")
      return "cortex-a7";
    break;
  default:
    break;
  }

  if (MArch.empty())
    return StringRef();

  StringRef CPU = llvm::ARM::getDefaultCPU(MArch);
  if (!CPU.empty() && !CPU.equals("invalid"))
    return CPU;

  // If no specific architecture version is requested, return the minimum CPU
  // required by the OS and environment.
  switch (Triple.getOS()) {
  case llvm::Triple::NetBSD:
    switch (Triple.getEnvironment()) {
    case llvm::Triple::EABI:
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABI:
    case llvm::Triple::GNUEABIHF:
      return "arm926ej-s";
    default:
      return "strongarm";
    }
  case llvm::Triple::NaCl:
  case llvm::Triple::OpenBSD:
    return "cortex-a8";
  default:
    switch (Triple.getEnvironment()) {
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABIHF:
    case llvm::Triple::MuslEABIHF:
      return "arm1176jzf-s";
    default:
      return "arm7tdmi";
    }
  }

  llvm_unreachable("invalid arch name");
}
