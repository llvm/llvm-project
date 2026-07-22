//===-- AMDGPUTargetParser - Parser for AMDGPU features ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise AMDGPU hardware features.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace AMDGPU;

StringRef llvm::AMDGPU::getArchFamilyNameAMDGCN(GPUKind AK) {
  StringRef ArchName = getArchNameAMDGCN(AK);
  assert((AK >= GK_AMDGCN_GENERIC_FIRST && AK <= GK_AMDGCN_GENERIC_LAST) ==
             ArchName.ends_with("-generic") &&
         "Generic AMDGCN arch not classified correctly!");
  if (AK >= GK_AMDGCN_GENERIC_FIRST && AK <= GK_AMDGCN_GENERIC_LAST) {
    // Return the part before the first '-', e.g. "gfx9-4-generic" -> "gfx9".
    return ArchName.take_front(ArchName.find('-'));
  }
  return ArchName.empty() ? "" : ArchName.drop_back(2);
}

Triple::SubArchType llvm::AMDGPU::getSubArch(GPUKind AK) {
  switch (AK) {
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES)                  \
  case ENUM:                                                                   \
    return SUBARCH;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  default:
    return Triple::SubArchType::NoSubArch;
  }
}

AMDGPU::GPUKind
llvm::AMDGPU::getGPUKindFromSubArch(Triple::SubArchType SubArch) {
  switch (SubArch) {
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES)                  \
  case SUBARCH:                                                                \
    return ENUM;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  default:
    return GK_NONE;
  }
}

static const Triple::SubArchType
    AMDGPUMajorFamilies[Triple::LastAMDGPUSubArch - Triple::FirstAMDGPUSubArch +
                        1] = {
        Triple::AMDGPUSubArch6,    Triple::AMDGPUSubArch6,
        Triple::AMDGPUSubArch6,    Triple::AMDGPUSubArch6,

        Triple::AMDGPUSubArch7,    Triple::AMDGPUSubArch7,
        Triple::AMDGPUSubArch7,    Triple::AMDGPUSubArch7,
        Triple::AMDGPUSubArch7,    Triple::AMDGPUSubArch7,
        Triple::AMDGPUSubArch7,

        Triple::AMDGPUSubArch8,    Triple::AMDGPUSubArch8,
        Triple::AMDGPUSubArch8,    Triple::AMDGPUSubArch8,
        Triple::AMDGPUSubArch8,

        Triple::AMDGPUSubArch810,

        Triple::AMDGPUSubArch9,    Triple::AMDGPUSubArch9,
        Triple::AMDGPUSubArch9,    Triple::AMDGPUSubArch9,
        Triple::AMDGPUSubArch9,    Triple::AMDGPUSubArch9,
        Triple::AMDGPUSubArch9,

        Triple::AMDGPUSubArch908,  Triple::AMDGPUSubArch90A,

        Triple::AMDGPUSubArch9_4,  Triple::AMDGPUSubArch9_4,
        Triple::AMDGPUSubArch9_4,

        Triple::AMDGPUSubArch10_1, Triple::AMDGPUSubArch10_1,
        Triple::AMDGPUSubArch10_1, Triple::AMDGPUSubArch10_1,
        Triple::AMDGPUSubArch10_1,

        Triple::AMDGPUSubArch10_3, Triple::AMDGPUSubArch10_3,
        Triple::AMDGPUSubArch10_3, Triple::AMDGPUSubArch10_3,
        Triple::AMDGPUSubArch10_3, Triple::AMDGPUSubArch10_3,
        Triple::AMDGPUSubArch10_3, Triple::AMDGPUSubArch10_3,

        Triple::AMDGPUSubArch11,   Triple::AMDGPUSubArch11,
        Triple::AMDGPUSubArch11,   Triple::AMDGPUSubArch11,
        Triple::AMDGPUSubArch11,   Triple::AMDGPUSubArch11,
        Triple::AMDGPUSubArch11,   Triple::AMDGPUSubArch11,
        Triple::AMDGPUSubArch11,   Triple::AMDGPUSubArch11,

        Triple::AMDGPUSubArch11_7, Triple::AMDGPUSubArch11_7,
        Triple::AMDGPUSubArch11_7, Triple::AMDGPUSubArch11_7,

        Triple::AMDGPUSubArch12,   Triple::AMDGPUSubArch12,
        Triple::AMDGPUSubArch12,

        Triple::AMDGPUSubArch12_5, Triple::AMDGPUSubArch12_5,
        Triple::AMDGPUSubArch12_5,

        Triple::AMDGPUSubArch13,   Triple::AMDGPUSubArch13};

Triple::SubArchType AMDGPU::getMajorSubArch(Triple::SubArchType X) {
  if (X < Triple::FirstAMDGPUSubArch || X > Triple::LastAMDGPUSubArch)
    return Triple::NoSubArch;
  return AMDGPUMajorFamilies[X - Triple::FirstAMDGPUSubArch];
}

bool AMDGPU::isSubArchCompatible(Triple::SubArchType A, Triple::SubArchType B) {
  if (A == B || A == Triple::NoSubArch || B == Triple::NoSubArch)
    return true;

  Triple::SubArchType MajorA = AMDGPU::getMajorSubArch(A);
  Triple::SubArchType MajorB = AMDGPU::getMajorSubArch(B);

  // One side is the major-family subarch covering the other's family.
  if (A == MajorA)
    return MajorA == MajorB;
  if (B == MajorB)
    return MajorA == MajorB;

  return false;
}

bool AMDGPU::isCPUValidForSubArch(Triple::SubArchType SubArch, GPUKind AK) {
  // An unrecognized GPU is never valid.
  if (AK == GK_NONE)
    return false;
  // A legacy triple without a subarch accepts any known GPU.
  if (SubArch == Triple::NoSubArch)
    return true;
  return isSubArchCompatible(getSubArch(AK), SubArch);
}

bool AMDGPU::isCPUValidForSubArch(Triple::SubArchType SubArch, StringRef CPU) {
  return isCPUValidForSubArch(SubArch, parseArchAMDGCN(CPU));
}

bool AMDGPU::isSubArchCompatible(const Triple &A, const Triple &B) {
  // Tolerate subarch mismatch if one entry is none. This is a hack for bitcode
  // libraries.
  // There's a missing enum entry for an unknown subarch. Make sure the
  // subarch is really empty.
  if (A.getSubArch() == Triple::NoSubArch)
    return A.getArchName().size() == 6;

  if (B.getSubArch() == Triple::NoSubArch)
    return B.getArchName().size() == 6;

  return isSubArchCompatible(A.getSubArch(), B.getSubArch());
}

std::string AMDGPU::mergeSubArch(const Triple &A, const Triple &B) {
  if (A.getSubArch() == Triple::NoSubArch)
    return B.str();
  if (B.getSubArch() == Triple::NoSubArch)
    return A.str();

  Triple::SubArchType MajorA = AMDGPU::getMajorSubArch(A.getSubArch());
  Triple::SubArchType MajorB = AMDGPU::getMajorSubArch(B.getSubArch());

  // With a compatible major arch, return the specific subarch.
  if (A.getSubArch() == MajorA) {
    if (MajorA == MajorB)
      return B.str();
  }

  if (B.getSubArch() == MajorB) {
    if (MajorA == MajorB)
      return A.str();
  }

  // Invalid case.
  return B.str();
}

StringRef llvm::AMDGPU::getArchNameAMDGCN(GPUKind AK) {
  switch (AK) {
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES)                  \
  case ENUM:                                                                   \
    return NAME;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  default:
    return "";
  }
}

// Canonical GPU name for each AMDGPU subarch, indexed by SubArch -
// Triple::FirstAMDGPUSubArch.
static const StringLiteral AMDGPUSubArchNames[Triple::LastAMDGPUSubArch -
                                              Triple::FirstAMDGPUSubArch + 1] =
    {"gfx600", // AMDGPUSubArch6 (no generic target)
     "gfx600",          "gfx601",  "gfx602",

     "gfx700", // AMDGPUSubArch7 (no generic target)
     "gfx700",          "gfx701",  "gfx702",  "gfx703",          "gfx704",
     "gfx705",

     "gfx801", // AMDGPUSubArch8 (no generic target)
     "gfx801",          "gfx802",  "gfx803",  "gfx805",

     "gfx810",

     "gfx9-generic",    "gfx900",  "gfx902",  "gfx904",          "gfx906",
     "gfx909",          "gfx90c",

     "gfx908",          "gfx90a",

     "gfx9-4-generic",  "gfx942",  "gfx950",

     "gfx10-1-generic", "gfx1010", "gfx1011", "gfx1012",         "gfx1013",

     "gfx10-3-generic", "gfx1030", "gfx1031", "gfx1032",         "gfx1033",
     "gfx1034",         "gfx1035", "gfx1036",

     "gfx11-generic",   "gfx1100", "gfx1101", "gfx1102",         "gfx1103",
     "gfx1150",         "gfx1151", "gfx1152", "gfx1153",         "gfx1154",

     "gfx11-7-generic", "gfx1170", "gfx1171", "gfx1172",

     "gfx12-generic",   "gfx1200", "gfx1201", "gfx12-5-generic", "gfx1250",
     "gfx1251",

     "gfx13-generic",   "gfx1310"};

StringRef llvm::AMDGPU::getArchNameFromSubArch(Triple::SubArchType SubArch) {
  if (SubArch < Triple::FirstAMDGPUSubArch ||
      SubArch > Triple::LastAMDGPUSubArch)
    return "";
  return AMDGPUSubArchNames[SubArch - Triple::FirstAMDGPUSubArch];
}

StringRef llvm::AMDGPU::getArchNameR600(GPUKind AK) {
  switch (AK) {
#define R600_GPU(NAME, ENUM, FEATURES)                                         \
  case ENUM:                                                                   \
    return NAME;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  default:
    return "";
  }
}

AMDGPU::GPUKind llvm::AMDGPU::parseArchAMDGCN(StringRef CPU) {
  return StringSwitch<AMDGPU::GPUKind>(CPU)
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES) .Case(NAME, ENUM)
#define AMDGCN_GPU_ALIAS(NAME, ENUM) .Case(NAME, ENUM)
#include "llvm/TargetParser/AMDGPUTargetParser.def"
      .Case("generic", AMDGPU::GPUKind::GK_GFX600)
      .Case("generic-hsa", AMDGPU::GPUKind::GK_GFX700)
      .Default(AMDGPU::GPUKind::GK_NONE);
}

AMDGPU::GPUKind llvm::AMDGPU::parseArchR600(StringRef CPU) {
  return StringSwitch<AMDGPU::GPUKind>(CPU)
#define R600_GPU(NAME, ENUM, FEATURES) .Case(NAME, ENUM)
#define R600_GPU_ALIAS(NAME, ENUM) .Case(NAME, ENUM)
#include "llvm/TargetParser/AMDGPUTargetParser.def"
      .Default(AMDGPU::GPUKind::GK_NONE);
}

unsigned AMDGPU::getArchAttrAMDGCN(GPUKind AK) {
  switch (AK) {
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES)                  \
  case ENUM:                                                                   \
    return FEATURES;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  default:
    return FEATURE_NONE;
  }
}

unsigned AMDGPU::getArchAttrAMDGCN(Triple::SubArchType SubArch) {
  switch (SubArch) {
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES)                  \
  case SUBARCH:                                                                \
    return FEATURES;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  default:
    return FEATURE_NONE;
  }
}

unsigned AMDGPU::getArchAttrR600(GPUKind AK) {
  switch (AK) {
#define R600_GPU(NAME, ENUM, FEATURES)                                         \
  case ENUM:                                                                   \
    return FEATURES;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  default:
    return FEATURE_NONE;
  }
}

void AMDGPU::fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &Values,
                                     Triple::SubArchType SubArch) {
  // XXX: Should this only report unique canonical names?
  // An alias shares its GPU's GPUKind, so it is filtered alongside it.
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES)                  \
  if (isCPUValidForSubArch(SubArch, ENUM))                                     \
    Values.push_back(NAME);
#define AMDGCN_GPU_ALIAS(NAME, ENUM)                                           \
  if (isCPUValidForSubArch(SubArch, ENUM))                                     \
    Values.push_back(NAME);
#include "llvm/TargetParser/AMDGPUTargetParser.def"
}

void AMDGPU::fillValidArchListR600(SmallVectorImpl<StringRef> &Values) {
  Values.append({
#define R600_GPU(NAME, ENUM, FEATURES) NAME,
#define R600_GPU_ALIAS(NAME, ENUM) NAME,
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  });
}

AMDGPU::IsaVersion AMDGPU::getIsaVersion(StringRef GPU) {
  AMDGPU::GPUKind AK = parseArchAMDGCN(GPU);
  if (AK == AMDGPU::GPUKind::GK_NONE) {
    if (GPU == "generic-hsa")
      return {7, 0, 0};
    if (GPU == "generic")
      return {6, 0, 0};
    return {0, 0, 0};
  }

  switch (AK) {
#define MAKE_ISAVERSION(A, B, C) {A, B, C}
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES)                  \
  case ENUM:                                                                   \
    return MAKE_ISAVERSION ISAVERSION;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
#undef MAKE_ISAVERSION
  default:
    return {0, 0, 0};
  }
}

AMDGPU::IsaVersion AMDGPU::getIsaVersion(Triple::SubArchType SubArch) {
  switch (SubArch) {
#define MAKE_ISAVERSION(A, B, C) {A, B, C}
#define AMDGCN_GPU(NAME, ENUM, SUBARCH, ISAVERSION, FEATURES)                  \
  case SUBARCH:                                                                \
    return MAKE_ISAVERSION ISAVERSION;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
#undef MAKE_ISAVERSION
  default:
    return {0, 0, 0};
  }
}

unsigned AMDGPU::getTotalNumSGPRs(GPUKind AK) {
  IsaVersion Version = getIsaVersion(getSubArch(AK));
  if (Version.Major >= 8)
    return 800;
  return 512;
}

unsigned AMDGPU::getTotalNumSGPRs(Triple::SubArchType SubArch) {
  IsaVersion Version = getIsaVersion(SubArch);
  if (Version.Major >= 8)
    return 800;
  return 512;
}

unsigned AMDGPU::getAddressableNumSGPRs(GPUKind AK) {
  if (getArchAttrAMDGCN(AK) & FEATURE_SGPR_INIT_BUG)
    return FIXED_NUM_SGPRS_FOR_INIT_BUG;

  IsaVersion Version = getIsaVersion(getSubArch(AK));
  if (Version.Major >= 10)
    return 106;
  if (Version.Major >= 8)
    return 102;
  return 104;
}

unsigned AMDGPU::getAddressableNumSGPRs(Triple::SubArchType SubArch) {
  if (getArchAttrAMDGCN(SubArch) & FEATURE_SGPR_INIT_BUG)
    return FIXED_NUM_SGPRS_FOR_INIT_BUG;

  IsaVersion Version = getIsaVersion(SubArch);
  if (Version.Major >= 10)
    return 106;
  if (Version.Major >= 8)
    return 102;
  return 104;
}

unsigned AMDGPU::getSGPRAllocGranule(GPUKind AK) {
  IsaVersion Version = getIsaVersion(getSubArch(AK));
  if (Version.Major >= 10)
    return getAddressableNumSGPRs(AK);
  if (Version.Major >= 8)
    return 16;
  return 8;
}

unsigned AMDGPU::getSGPRAllocGranule(Triple::SubArchType SubArch) {
  IsaVersion Version = getIsaVersion(SubArch);
  if (Version.Major >= 10)
    return getAddressableNumSGPRs(SubArch);
  if (Version.Major >= 8)
    return 16;
  return 8;
}

StringRef AMDGPU::getCanonicalArchName(const Triple &T, StringRef Arch) {
  assert(T.isAMDGPU());
  auto ProcKind = T.isAMDGCN() ? parseArchAMDGCN(Arch) : parseArchR600(Arch);
  if (ProcKind == GK_NONE)
    return StringRef();

  return T.isAMDGCN() ? getArchNameAMDGCN(ProcKind) : getArchNameR600(ProcKind);
}

static std::pair<FeatureError, StringRef>
insertWaveSizeFeature(StringRef GPU, const Triple &T,
                      const StringMap<bool> &DefaultFeatures,
                      StringMap<bool> &Features) {
  const bool IsNullGPU = GPU.empty();
  const bool TargetHasWave32 = DefaultFeatures.count("wavefrontsize32");
  const bool TargetHasWave64 = DefaultFeatures.count("wavefrontsize64");

  auto Wave32Itr = Features.find("wavefrontsize32");
  auto Wave64Itr = Features.find("wavefrontsize64");
  const bool EnableWave32 =
      Wave32Itr != Features.end() && Wave32Itr->getValue();
  const bool EnableWave64 =
      Wave64Itr != Features.end() && Wave64Itr->getValue();
  const bool DisableWave32 =
      Wave32Itr != Features.end() && !Wave32Itr->getValue();
  const bool DisableWave64 =
      Wave64Itr != Features.end() && !Wave64Itr->getValue();

  if (EnableWave32 && EnableWave64)
    return {AMDGPU::INVALID_FEATURE_COMBINATION,
            "'+wavefrontsize32' and '+wavefrontsize64' are mutually exclusive"};
  if (DisableWave32 && DisableWave64)
    return {AMDGPU::INVALID_FEATURE_COMBINATION,
            "'-wavefrontsize32' and '-wavefrontsize64' are mutually exclusive"};

  if (!IsNullGPU) {
    if (TargetHasWave64) {
      if (EnableWave32)
        return {AMDGPU::UNSUPPORTED_TARGET_FEATURE, "+wavefrontsize32"};
      if (DisableWave64)
        return {AMDGPU::UNSUPPORTED_TARGET_FEATURE, "-wavefrontsize64"};
    }

    if (TargetHasWave32) {
      if (EnableWave64)
        return {AMDGPU::UNSUPPORTED_TARGET_FEATURE, "+wavefrontsize64"};
      if (DisableWave32)
        return {AMDGPU::UNSUPPORTED_TARGET_FEATURE, "-wavefrontsize32"};
    }
  }

  // Don't assume any wavesize with an unknown subtarget.
  // Default to wave32 if target supports both.
  if (!IsNullGPU && !EnableWave32 && !EnableWave64 && !TargetHasWave32 &&
      !TargetHasWave64)
    Features.insert(std::make_pair("wavefrontsize32", true));

  for (const auto &Entry : DefaultFeatures) {
    if (!Features.count(Entry.getKey()))
      Features[Entry.getKey()] = Entry.getValue();
  }

  return {NO_ERROR, StringRef()};
}

/// Fills Features map with default values for given target GPU.
/// \p Features contains overriding target features and this function returns
/// default target features with entries overridden by \p Features.
static void fillAMDGCNFeatureMap(StringRef GPU, const Triple &T,
                                 StringMap<bool> &Features) {
  AMDGPU::GPUKind Kind = parseArchAMDGCN(GPU);
  switch (Kind) {
  case GK_GFX1310:
  case GK_GFX13_GENERIC:
    Features["ci-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot8-insts"] = true;
    Features["dl-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["gfx12-insts"] = true;
    Features["gfx1250-insts"] = true;
    Features["gfx13-insts"] = true;
    Features["bitop3-insts"] = true;
    Features["prng-inst"] = true;
    Features["tanh-insts"] = true;
    Features["tensor-cvt-lut-insts"] = true;
    Features["bf16-trans-insts"] = true;
    Features["bf16-cvt-insts"] = true;
    Features["bf16-pk-insts"] = true;
    Features["fp8-conversion-insts"] = true;
    Features["permlane16-swap"] = true;
    Features["ashr-pk-insts"] = true;
    Features["atomic-buffer-pk-add-bf16-inst"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["atomic-buffer-global-pk-add-f16-insts"] = true;
    Features["atomic-flat-pk-add-16-insts"] = true;
    Features["atomic-global-pk-add-bf16-inst"] = true;
    Features["atomic-ds-pk-add-16-insts"] = true;
    Features["s-wakeup-barrier-inst"] = true;
    Features["f16bf16-to-fp6bf6-cvt-scale-insts"] = true;
    Features["f32-to-fp6bf6-cvt-scale-insts"] = true;
    Features["clusters"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["cvt-pknorm-vop3-insts"] = true;
    Features["image-insts"] = true;
    break;
  case GK_GFX1251:
    Features["gfx1251-gemm-insts"] = true;
    [[fallthrough]];
  case GK_GFX1250:
    Features["swmmac-gfx1200-insts"] = true;
    Features["swmmac-gfx1250-insts"] = true;
    Features["cube-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["lerp-inst"] = true;
    Features["qsad-insts"] = true;
    Features["sad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    [[fallthrough]];
  case GK_GFX12_5_GENERIC:
    Features["ci-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot8-insts"] = true;
    Features["dl-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["flat-global-insts"] = true;
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["gfx12-insts"] = true;
    Features["gfx1250-insts"] = true;
    Features["bitop3-insts"] = true;
    Features["prng-inst"] = true;
    Features["tanh-insts"] = true;
    Features["tensor-cvt-lut-insts"] = true;
    Features["transpose-load-f4f6-insts"] = true;
    Features["bf16-trans-insts"] = true;
    Features["bf16-cvt-insts"] = true;
    Features["bf16-pk-insts"] = true;
    Features["fp8-conversion-insts"] = true;
    Features["fp8e5m3-insts"] = true;
    Features["permlane16-swap"] = true;
    Features["ashr-pk-insts"] = true;
    Features["add-min-max-insts"] = true;
    Features["pk-add-min-max-insts"] = true;
    Features["atomic-buffer-pk-add-bf16-inst"] = true;
    Features["vmem-pref-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["atomic-buffer-global-pk-add-f16-insts"] = true;
    Features["atomic-flat-pk-add-16-insts"] = true;
    Features["atomic-global-pk-add-bf16-inst"] = true;
    Features["atomic-ds-pk-add-16-insts"] = true;
    Features["setprio-inc-wg-inst"] = true;
    Features["s-wakeup-barrier-inst"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    Features["atomic-fmin-fmax-global-f64"] = true;
    Features["wavefrontsize32"] = true;
    Features["clusters"] = true;
    Features["mcast-load-insts"] = true;
    Features["asynccnt"] = true;
    break;
  case GK_GFX1201:
  case GK_GFX1200:
  case GK_GFX12_GENERIC:
    Features["ci-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot8-insts"] = true;
    Features["dot9-insts"] = true;
    Features["dot10-insts"] = true;
    Features["dot11-insts"] = true;
    Features["dot12-insts"] = true;
    Features["dl-insts"] = true;
    Features["atomic-ds-pk-add-16-insts"] = true;
    Features["atomic-flat-pk-add-16-insts"] = true;
    Features["atomic-buffer-global-pk-add-f16-insts"] = true;
    Features["atomic-buffer-pk-add-bf16-inst"] = true;
    Features["atomic-global-pk-add-bf16-inst"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["flat-global-insts"] = true;
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["gfx12-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["image-insts"] = true;
    Features["bvh-ray-tracing-insts"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["fp8-conversion-insts"] = true;
    Features["wmma-128b-insts"] = true;
    Features["swmmac-gfx1200-insts"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    break;
  case GK_GFX1170:
  case GK_GFX1171:
  case GK_GFX1172:
  case GK_GFX11_7_GENERIC:
    Features["ci-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot8-insts"] = true;
    Features["dot9-insts"] = true;
    Features["dot10-insts"] = true;
    Features["dot12-insts"] = true;
    Features["dl-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["flat-global-insts"] = true;
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["image-insts"] = true;
    Features["bvh-ray-tracing-insts"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["gws"] = true;
    Features["dot11-insts"] = true;
    Features["fp8-conversion-insts"] = true;
    Features["wmma-128b-insts"] = true;
    Features["swmmac-gfx1200-insts"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    break;
  case GK_GFX1154:
  case GK_GFX1153:
  case GK_GFX1152:
  case GK_GFX1151:
  case GK_GFX1150:
  case GK_GFX1103:
  case GK_GFX1102:
  case GK_GFX1101:
  case GK_GFX1100:
  case GK_GFX11_GENERIC:
    Features["ci-insts"] = true;
    Features["dot5-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot8-insts"] = true;
    Features["dot9-insts"] = true;
    Features["dot10-insts"] = true;
    Features["dot12-insts"] = true;
    Features["dl-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["flat-global-insts"] = true;
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["image-insts"] = true;
    Features["bvh-ray-tracing-insts"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["gws"] = true;
    Features["wmma-256b-insts"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    break;
  case GK_GFX1036:
  case GK_GFX1035:
  case GK_GFX1034:
  case GK_GFX1033:
  case GK_GFX1032:
  case GK_GFX1031:
  case GK_GFX1030:
  case GK_GFX10_3_GENERIC:
    Features["ci-insts"] = true;
    Features["dot1-insts"] = true;
    Features["dot2-insts"] = true;
    Features["dot5-insts"] = true;
    Features["dot6-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot10-insts"] = true;
    Features["dl-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["flat-global-insts"] = true;
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["image-insts"] = true;
    Features["bvh-ray-tracing-insts"] = true;
    Features["s-memrealtime"] = true;
    Features["s-memtime-inst"] = true;
    Features["gws"] = true;
    Features["vmem-to-lds-load-insts"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    Features["atomic-fmin-fmax-global-f64"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    break;
  case GK_GFX1012:
  case GK_GFX1011:
    Features["dot1-insts"] = true;
    Features["dot2-insts"] = true;
    Features["dot5-insts"] = true;
    Features["dot6-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot10-insts"] = true;
    [[fallthrough]];
  case GK_GFX1013:
  case GK_GFX1010:
  case GK_GFX10_1_GENERIC:
    if (Kind == GK_GFX1013)
      Features["bvh-ray-tracing-insts"] = true;
    Features["dl-insts"] = true;
    Features["ci-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["flat-global-insts"] = true;
    Features["gfx10-insts"] = true;
    Features["image-insts"] = true;
    Features["s-memrealtime"] = true;
    Features["s-memtime-inst"] = true;
    Features["gws"] = true;
    Features["vmem-to-lds-load-insts"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    Features["atomic-fmin-fmax-global-f64"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    break;
  case GK_GFX950:
    Features["bitop3-insts"] = true;
    Features["fp6bf6-cvt-scale-insts"] = true;
    Features["fp4-cvt-scale-insts"] = true;
    Features["bf8-cvt-scale-insts"] = true;
    Features["fp8-cvt-scale-insts"] = true;
    Features["f16bf16-to-fp6bf6-cvt-scale-insts"] = true;
    Features["f32-to-f16bf16-cvt-sr-insts"] = true;
    Features["prng-inst"] = true;
    Features["permlane16-swap"] = true;
    Features["permlane32-swap"] = true;
    Features["ashr-pk-insts"] = true;
    Features["dot12-insts"] = true;
    Features["dot13-insts"] = true;
    Features["atomic-buffer-pk-add-bf16-inst"] = true;
    Features["gfx950-insts"] = true;
    [[fallthrough]];
  case GK_GFX942:
    Features["fp8-insts"] = true;
    Features["fp8-conversion-insts"] = true;
    if (Kind != GK_GFX950)
      Features["xf32-insts"] = true;
    [[fallthrough]];
  case GK_GFX9_4_GENERIC:
    Features["gfx940-insts"] = true;
    Features["atomic-ds-pk-add-16-insts"] = true;
    Features["atomic-flat-pk-add-16-insts"] = true;
    Features["atomic-global-pk-add-bf16-inst"] = true;
    Features["gfx90a-insts"] = true;
    Features["atomic-buffer-global-pk-add-f16-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["dot3-insts"] = true;
    Features["dot4-insts"] = true;
    Features["dot5-insts"] = true;
    Features["dot6-insts"] = true;
    Features["mai-insts"] = true;
    Features["dl-insts"] = true;
    Features["dot1-insts"] = true;
    Features["dot2-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot10-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["flat-global-insts"] = true;
    Features["gfx8-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["s-memrealtime"] = true;
    Features["ci-insts"] = true;
    Features["s-memtime-inst"] = true;
    Features["gws"] = true;
    Features["vmem-to-lds-load-insts"] = true;
    Features["atomic-fmin-fmax-global-f64"] = true;
    Features["wavefrontsize64"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    break;
  case GK_GFX90A:
    Features["gfx90a-insts"] = true;
    Features["atomic-buffer-global-pk-add-f16-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["atomic-fmin-fmax-global-f64"] = true;
    [[fallthrough]];
  case GK_GFX908:
    Features["dot3-insts"] = true;
    Features["dot4-insts"] = true;
    Features["dot5-insts"] = true;
    Features["dot6-insts"] = true;
    Features["mai-insts"] = true;
    [[fallthrough]];
  case GK_GFX906:
    Features["dl-insts"] = true;
    Features["dot1-insts"] = true;
    Features["dot2-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot10-insts"] = true;
    [[fallthrough]];
  case GK_GFX90C:
  case GK_GFX909:
  case GK_GFX904:
  case GK_GFX902:
  case GK_GFX900:
  case GK_GFX9_GENERIC:
    Features["gfx9-insts"] = true;
    Features["flat-global-insts"] = true;
    Features["vmem-to-lds-load-insts"] = true;
    [[fallthrough]];
  case GK_GFX810:
  case GK_GFX805:
  case GK_GFX803:
  case GK_GFX802:
  case GK_GFX801:
    Features["gfx8-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["s-memrealtime"] = true;
    Features["ci-insts"] = true;
    Features["image-insts"] = true;
    Features["s-memtime-inst"] = true;
    Features["gws"] = true;
    Features["wavefrontsize64"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    break;
  case GK_GFX705:
  case GK_GFX704:
  case GK_GFX703:
  case GK_GFX702:
  case GK_GFX701:
  case GK_GFX700:
    Features["ci-insts"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["mqsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["image-insts"] = true;
    Features["s-memtime-inst"] = true;
    Features["gws"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    Features["atomic-fmin-fmax-global-f64"] = true;
    Features["wavefrontsize64"] = true;
    break;
  case GK_GFX602:
  case GK_GFX601:
  case GK_GFX600:
    Features["image-insts"] = true;
    Features["s-memtime-inst"] = true;
    Features["gws"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    Features["atomic-fmin-fmax-global-f64"] = true;
    Features["wavefrontsize64"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["msad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    break;
  case GK_NONE:
    break;
  default:
    llvm_unreachable("Unhandled GPU!");
  }
}

/// Fills Features map with default values for given target GPU.
/// \p Features contains overriding target features and this function returns
/// default target features with entries overridden by \p Features.
std::pair<FeatureError, StringRef>
AMDGPU::fillAMDGPUFeatureMap(StringRef GPU, const Triple &T,
                             StringMap<bool> &Features) {
  // XXX - What does the member GPU mean if device name string passed here?
  if (T.isSPIRV() && T.getOS() == Triple::OSType::AMDHSA) {
    // AMDGCN SPIRV must support the union of all AMDGCN features.
    SmallVector<StringRef> GPUs;
    fillValidArchListAMDGCN(GPUs);

    static const Triple AMDGCN("amdgcn-amd-amdhsa");
    StringMap<bool> Tmp;
    for (auto &&GPU : GPUs) {
      fillAMDGCNFeatureMap(GPU, AMDGCN, Tmp);
      for (auto &&[F, B] : Tmp)
        Features[F] = B;
    }
    Features["wavefrontsize32"] = true;
    Features["wavefrontsize64"] = true;
  } else if (T.isAMDGCN()) {
    StringMap<bool> DefaultFeatures;
    fillAMDGCNFeatureMap(GPU, T, DefaultFeatures);
    return insertWaveSizeFeature(GPU, T, DefaultFeatures, Features);
  } else {
    if (GPU.empty())
      GPU = "r600";

    switch (llvm::AMDGPU::parseArchR600(GPU)) {
    case GK_CAYMAN:
    case GK_CYPRESS:
    case GK_RV770:
    case GK_RV670:
      // TODO: Add fp64 when implemented.
      break;
    case GK_TURKS:
    case GK_CAICOS:
    case GK_BARTS:
    case GK_SUMO:
    case GK_REDWOOD:
    case GK_JUNIPER:
    case GK_CEDAR:
    case GK_RV730:
    case GK_RV710:
    case GK_RS880:
    case GK_R630:
    case GK_R600:
      break;
    default:
      llvm_unreachable("Unhandled GPU!");
    }
  }
  return {NO_ERROR, StringRef()};
}

TargetID::TargetID(GPUKind Arch, const Triple &TT, TargetIDSetting XnackSetting,
                   TargetIDSetting SramEccSetting)
    : Arch(Arch),
      TargetTripleString(TT.normalize(Triple::CanonicalForm::FOUR_IDENT)),
      XnackSetting(XnackSetting), SramEccSetting(SramEccSetting),
      IsAMDHSA(TT.getOS() == Triple::AMDHSA) {}

// Parse a feature modifier sign ("+"/"-"). Returns "Unsupported" if \p Sign is
// neither (i.e. the modifier is malformed).
static TargetIDSetting getTargetIDSettingFromFeatureString(StringRef Sign) {
  if (Sign == "+")
    return TargetIDSetting::On;
  if (Sign == "-")
    return TargetIDSetting::Off;

  return TargetIDSetting::Unsupported;
}

// Derive the architecture from the processor name in \p TargetIDStr. "generic"
// and the empty processor name act as a wildcard.
static GPUKind getGPUKindFromTargetID(const Triple &TT, StringRef TargetIDStr) {
  StringRef CPUName = TargetIDStr.split(':').first;
  return (CPUName.empty() || CPUName == "generic")
             ? getGPUKindFromSubArch(TT.getSubArch())
             : parseArchAMDGCN(CPUName);
}

// Compute the xnack/sramecc settings for processor \p Arch from the
// processor+features string \p TargetIDStr
// (e.g. "gfx90a:xnack+:sramecc-"). Returns false if a modifier names an unknown
// or repeated feature, names one the processor does not support, or has a
// malformed sign.
static bool computeTargetIDFeatures(GPUKind Arch, StringRef TargetIDStr,
                                    TargetIDSetting &XnackSetting,
                                    TargetIDSetting &SramEccSetting) {
  unsigned ArchAttr = getArchAttrAMDGCN(Arch);
  XnackSetting = (ArchAttr & FEATURE_XNACK_ON_OFF_MODES)
                     ? TargetIDSetting::Any
                     : TargetIDSetting::Unsupported;
  SramEccSetting = (ArchAttr & FEATURE_SRAMECC) ? TargetIDSetting::Any
                                                : TargetIDSetting::Unsupported;

  // The first component is the processor; the rest are feature modifiers of the
  // form "<feature><+|->".
  SmallVector<StringRef, 3> Split;
  TargetIDStr.split(Split, ':');
  bool SeenXnack = false;
  bool SeenSramEcc = false;
  bool Valid = true;
  for (unsigned I = 1, E = Split.size(); I != E; ++I) {
    StringRef FeatureString = Split[I];
    if (FeatureString.consume_front("xnack")) {
      TargetIDSetting Sign = getTargetIDSettingFromFeatureString(FeatureString);
      if (SeenXnack || XnackSetting == TargetIDSetting::Unsupported ||
          Sign == TargetIDSetting::Unsupported)
        Valid = false;
      else
        XnackSetting = Sign;
      SeenXnack = true;
    } else if (FeatureString.consume_front("sramecc")) {
      TargetIDSetting Sign = getTargetIDSettingFromFeatureString(FeatureString);
      if (SeenSramEcc || SramEccSetting == TargetIDSetting::Unsupported ||
          Sign == TargetIDSetting::Unsupported)
        Valid = false;
      else
        SramEccSetting = Sign;
      SeenSramEcc = true;
    } else {
      // Unknown feature name.
      Valid = false;
    }
  }
  return Valid;
}

TargetID::TargetID(const Triple &TT, StringRef TargetIDStr)
    : TargetID(getGPUKindFromTargetID(TT, TargetIDStr), TT,
               TargetIDSetting::Unsupported, TargetIDSetting::Unsupported) {
  // Derive the feature settings from the string. Validity is not checked here;
  // parseTargetIDString validates untrusted input.
  computeTargetIDFeatures(Arch, TargetIDStr, XnackSetting, SramEccSetting);
}

std::optional<TargetID> TargetID::parse(const Triple &TT,
                                        StringRef ProcAndFeatures) {
  if (!TT.isAMDGCN())
    return std::nullopt;

  // A named processor (i.e. not the empty/generic wildcard, which is resolved
  // from the triple's subarch) must be a recognized GPU.
  StringRef CPUName = ProcAndFeatures.split(':').first;
  if (!CPUName.empty() && CPUName != "generic" &&
      parseArchAMDGCN(CPUName) == GK_NONE)
    return std::nullopt;

  // Parse the processor and its feature modifiers, then construct directly from
  // the resulting fields.
  GPUKind Arch = getGPUKindFromTargetID(TT, ProcAndFeatures);
  TargetIDSetting XnackSetting, SramEccSetting;
  if (!computeTargetIDFeatures(Arch, ProcAndFeatures, XnackSetting,
                               SramEccSetting))
    return std::nullopt;

  return TargetID(Arch, TT, XnackSetting, SramEccSetting);
}

std::optional<TargetID>
TargetID::parseTargetIDString(StringRef TargetIDDirective) {
  // Split on '-' to get arch-vendor-os-environment-processor:features. There is
  // a single dash separator after the 4-component triple, so the
  // processor+features field must be present (even if empty).
  SmallVector<StringRef, 5> Parts;
  TargetIDDirective.split(Parts, '-', /*MaxSplit=*/4);
  if (Parts.size() < 5)
    return std::nullopt;

  return parse(Triple(Parts[0], Parts[1], Parts[2], Parts[3]), Parts[4]);
}

// Append the explicit (On/Off) sramecc/xnack feature modifiers in canonical
// order, e.g. ":sramecc-:xnack+".
static void printFeatureModifiers(raw_ostream &OS, TargetIDSetting SramEcc,
                                  TargetIDSetting Xnack) {
  if (SramEcc == TargetIDSetting::Off)
    OS << ":sramecc-";
  else if (SramEcc == TargetIDSetting::On)
    OS << ":sramecc+";

  if (Xnack == TargetIDSetting::Off)
    OS << ":xnack-";
  else if (Xnack == TargetIDSetting::On)
    OS << ":xnack+";
}

void TargetID::print(raw_ostream &StreamRep) const {
  StreamRep << TargetTripleString << '-' << getArchNameAMDGCN(Arch);

  if (IsAMDHSA)
    printFeatureModifiers(StreamRep, getSramEccSetting(), getXnackSetting());
}

std::string TargetID::toString() const {
  std::string Str;
  raw_string_ostream OS(Str);
  OS << *this;
  return Str;
}

void TargetID::printCanonicalTargetIDString(raw_ostream &OS) const {
  OS << getArchNameAMDGCN(Arch);
  printFeatureModifiers(OS, getSramEccSetting(), getXnackSetting());
}

std::string TargetID::getCanonicalFeatureString() const {
  std::string Str;
  raw_string_ostream OS(Str);
  printCanonicalTargetIDString(OS);
  return Str;
}

bool TargetID::operator==(const TargetID &Other) const {
  return Arch == Other.Arch && XnackSetting == Other.XnackSetting &&
         SramEccSetting == Other.SramEccSetting && IsAMDHSA == Other.IsAMDHSA &&
         TargetTripleString == Other.TargetTripleString;
}

static bool featureProvidesFor(TargetIDSetting Provided,
                               TargetIDSetting Requested) {
  return Provided == TargetIDSetting::Any ||
         Provided == TargetIDSetting::Unsupported || Provided == Requested;
}

bool TargetID::isEquivalent(const TargetID &Other) const {
  // The processor and feature settings must match exactly
  if (Arch != Other.Arch || XnackSetting != Other.XnackSetting ||
      SramEccSetting != Other.SramEccSetting)
    return false;

  return Triple(getTargetTripleString())
      .isCompatibleWith(Triple(Other.getTargetTripleString()));
}

bool TargetID::providesFor(const TargetID &Other) const {
  // A major-family/generic processor (e.g. amdgpu9) provides for a specific
  // member of its family (e.g. gfx900), but not the reverse. Otherwise the
  // processors must match.
  if (Arch != Other.Arch && Arch != GK_NONE && Other.Arch != GK_NONE) {
    Triple::SubArchType ThisSubArch = getSubArch(Arch);
    if (ThisSubArch != getMajorSubArch(ThisSubArch) ||
        ThisSubArch != getMajorSubArch(getSubArch(Other.Arch)))
      return false;
  }

  if (!featureProvidesFor(XnackSetting, Other.XnackSetting) ||
      !featureProvidesFor(SramEccSetting, Other.SramEccSetting))
    return false;

  return Triple(getTargetTripleString())
      .isCompatibleWith(Triple(Other.getTargetTripleString()));
}
