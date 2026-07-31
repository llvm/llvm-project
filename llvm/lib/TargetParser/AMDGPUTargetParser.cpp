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
#include "llvm/ADT/StringTable.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <array>
#include <cassert>

using namespace llvm;
using namespace AMDGPU;

namespace {
constexpr unsigned NumAMDGPUSubArches =
    Triple::LastAMDGPUSubArch - Triple::FirstAMDGPUSubArch + 1;

// A legacy GPU name (e.g. "tahiti") mapped to the GPUKind it aliases.
struct GPUNameAlias {
  StringTable::Offset AltName;
  GPUKind Kind;
};

// Per-GPU data for the AMDGCN GPUKinds, from the generated table below.
struct GPUInfo {
  StringTable::Offset Name;
  Triple::SubArchType SubArch;
  unsigned ArchFeatures;
  IsaVersion Version;
  StringTable::Offset FamilyName;
};

// Per-GPU data for the R600 GPUKinds.
struct R600Info {
  StringTable::Offset Name;
  R600FeatureKind ArchFeatures;
};

#define GET_AMDGPU_NAME_TABLE
#define GET_AMDGPU_GPU_TABLE
#define GET_AMDGPU_GPU_ALIAS_TABLE
#define GET_AMDGPU_MAJOR_SUBARCH
#define GET_AMDGPU_SUBARCH_NAME
#include "llvm/TargetParser/AMDGPUTargetParserDef.inc"

#define GET_R600_NAME_TABLE
#define GET_R600_GPU_TABLE
#define GET_R600_GPU_ALIAS_TABLE
#include "llvm/TargetParser/R600TargetParserDef.inc"

// The string tables holding GPU-name-derived strings as offsets. R600 and
// AMDGPU come from separate generated headers, each with its own pool.
constexpr StringTable AMDGPUNameStrTab = AMDGPUNameTable;
constexpr StringTable R600NameStrTab = R600NameTable;

// Look up the GPUInfo row for an AMDGCN GPUKind, or nullptr for GK_NONE / a
// non-AMDGCN (R600) kind.
const GPUInfo *getAMDGPUInfo(GPUKind AK) {
  if (AK < AMDGPUFirstGPUKind)
    return nullptr;
  unsigned Idx = AK - AMDGPUFirstGPUKind;
  if (Idx >= std::size(AMDGPUGPUTable))
    return nullptr;
  return &AMDGPUGPUTable[Idx];
}

// Look up the R600Info row for an R600 GPUKind, or nullptr for a non-R600 kind.
const R600Info *getR600Info(GPUKind AK) {
  if (AK < R600FirstGPUKind)
    return nullptr;
  unsigned Idx = AK - R600FirstGPUKind;
  if (Idx >= std::size(R600GPUTable))
    return nullptr;
  return &R600GPUTable[Idx];
}

// Scan a name -> GPUKind table (canonical names, then aliases) for \p CPU.
template <typename InfoT, size_t N, size_t M>
GPUKind parseArchImpl(StringRef CPU, const InfoT (&Table)[N], GPUKind FirstKind,
                      const StringTable &StrTab,
                      const GPUNameAlias (&Aliases)[M]) {
  for (unsigned I = 0; I != N; ++I) {
    if (CPU == StrTab[Table[I].Name])
      return static_cast<GPUKind>(FirstKind + I);
  }

  for (const GPUNameAlias &A : Aliases) {
    if (CPU == StrTab[A.AltName])
      return A.Kind;
  }

  return GK_NONE;
}

// Reverse map: SubArch -> GPUKind, indexed by (SubArch - FirstAMDGPUSubArch).
// Subarches with no GPU (incl. the NoSubArch pseudo targets) map to GK_NONE.
constexpr std::array<GPUKind, NumAMDGPUSubArches> AMDGPUSubArchToGPUKind = [] {
  std::array<GPUKind, NumAMDGPUSubArches> Map{};

  for (unsigned I = 0; I < std::size(AMDGPUGPUTable); ++I) {
    Triple::SubArchType SubArch = AMDGPUGPUTable[I].SubArch;
    if (SubArch != Triple::NoSubArch) {
      Map[SubArch - Triple::FirstAMDGPUSubArch] =
          static_cast<GPUKind>(AMDGPUFirstGPUKind + I);
    }
  }
  return Map;
}();

/// SubArch -> major-family, indexed by (SubArch - FirstAMDGPUSubArch).
constexpr std::array<Triple::SubArchType, NumAMDGPUSubArches>
    AMDGPUMajorFamilies = [] {
      std::array<Triple::SubArchType, NumAMDGPUSubArches> Map{};

      for (unsigned I = 0; I < NumAMDGPUSubArches; ++I) {
        Map[I] =
            static_cast<Triple::SubArchType>(Triple::FirstAMDGPUSubArch + I);
      }

      for (const AMDGPUMajorSubArchEntry &Entry : AMDGPUMajorSubArch)
        Map[Entry.SubArch - Triple::FirstAMDGPUSubArch] = Entry.Major;
      return Map;
    }();

// SubArch -> name-offset, indexed by (SubArch - FirstAMDGPUSubArch). Unmapped
// subarches keep offset 0 (the empty string).
constexpr std::array<StringTable::Offset, NumAMDGPUSubArches>
    AMDGPUSubArchNameOffsets = [] {
      std::array<StringTable::Offset, NumAMDGPUSubArches> Map{};
      for (const AMDGPUSubArchNameEntry &Entry : AMDGPUSubArchNames)
        Map[Entry.SubArch - Triple::FirstAMDGPUSubArch] = Entry.NameOffset;
      return Map;
    }();

// SubArch -> triple-name-offset (e.g. "amdgpu9.00"), like
// AMDGPUSubArchNameOffsets.
constexpr std::array<StringTable::Offset, NumAMDGPUSubArches>
    AMDGPUSubArchTripleNameOffsets = [] {
      std::array<StringTable::Offset, NumAMDGPUSubArches> Map{};
      for (const AMDGPUSubArchNameEntry &Entry : AMDGPUSubArchNames)
        Map[Entry.SubArch - Triple::FirstAMDGPUSubArch] =
            Entry.TripleNameOffset;
      return Map;
    }();
} // namespace

StringRef llvm::AMDGPU::getArchFamilyNameAMDGCN(GPUKind AK) {
  const GPUInfo *Info = getAMDGPUInfo(AK);
  return Info ? AMDGPUNameStrTab[Info->FamilyName] : "";
}

Triple::SubArchType llvm::AMDGPU::getSubArch(GPUKind AK) {
  const GPUInfo *Info = getAMDGPUInfo(AK);
  return Info ? Info->SubArch : Triple::SubArchType::NoSubArch;
}

AMDGPU::GPUKind
llvm::AMDGPU::getGPUKindFromSubArch(Triple::SubArchType SubArch) {
  if (SubArch < Triple::FirstAMDGPUSubArch ||
      SubArch > Triple::LastAMDGPUSubArch)
    return GK_NONE;
  return AMDGPUSubArchToGPUKind[SubArch - Triple::FirstAMDGPUSubArch];
}

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

  // Reject the dummy "generic" targets
  Triple::SubArchType GPUSubArch = getSubArch(AK);
  if (GPUSubArch == Triple::NoSubArch)
    return false;

  return isSubArchCompatible(GPUSubArch, SubArch);
}

bool AMDGPU::isCPUValidForSubArch(Triple::SubArchType SubArch, StringRef CPU) {
  return isCPUValidForSubArch(SubArch, parseArchAMDGCN(CPU));
}

bool AMDGPU::isPseudoTarget(GPUKind AK) {
  const GPUInfo *Info = getAMDGPUInfo(AK);
  return Info && Info->SubArch == Triple::NoSubArch;
}

bool AMDGPU::isPseudoTarget(StringRef CPU) {
  return isPseudoTarget(parseArchAMDGCN(CPU));
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
  const GPUInfo *Info = getAMDGPUInfo(AK);
  return Info ? AMDGPUNameStrTab[Info->Name] : "";
}

StringRef llvm::AMDGPU::getArchNameFromSubArch(Triple::SubArchType SubArch) {
  if (SubArch < Triple::FirstAMDGPUSubArch ||
      SubArch > Triple::LastAMDGPUSubArch)
    return "";
  return AMDGPUNameStrTab[AMDGPUSubArchNameOffsets[SubArch -
                                                   Triple::FirstAMDGPUSubArch]];
}

StringRef llvm::AMDGPU::getSubArchName(Triple::SubArchType SubArch) {
  if (SubArch == Triple::NoSubArch)
    return AMDGPUNameStrTab[AMDGPUNoSubArchNameOffset];

  assert(SubArch >= Triple::FirstAMDGPUSubArch &&
         SubArch <= Triple::LastAMDGPUSubArch &&
         "expected an AMDGPU subarch or NoSubArch");
  return AMDGPUNameStrTab
      [AMDGPUSubArchTripleNameOffsets[SubArch - Triple::FirstAMDGPUSubArch]];
}

StringRef llvm::AMDGPU::getArchNameR600(GPUKind AK) {
  const R600Info *Info = getR600Info(AK);
  return Info ? R600NameStrTab[Info->Name] : "";
}

AMDGPU::GPUKind llvm::AMDGPU::parseArchAMDGCN(StringRef CPU) {
  return parseArchImpl(CPU, AMDGPUGPUTable, AMDGPUFirstGPUKind,
                       AMDGPUNameStrTab, AMDGPUGPUAliases);
}

AMDGPU::GPUKind llvm::AMDGPU::parseArchR600(StringRef CPU) {
  return parseArchImpl(CPU, R600GPUTable, R600FirstGPUKind, R600NameStrTab,
                       R600GPUAliases);
}

unsigned AMDGPU::getArchAttrAMDGCN(GPUKind AK) {
  const GPUInfo *Info = getAMDGPUInfo(AK);
  return Info ? Info->ArchFeatures : FEATURE_NONE;
}

unsigned AMDGPU::getArchAttrAMDGCN(Triple::SubArchType SubArch) {
  return getArchAttrAMDGCN(getGPUKindFromSubArch(SubArch));
}

R600FeatureKind AMDGPU::getArchAttrR600(GPUKind AK) {
  const R600Info *Info = getR600Info(AK);
  return Info ? Info->ArchFeatures : R600_FEATURE_NONE;
}

void AMDGPU::fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &Values,
                                     Triple::SubArchType SubArch) {
  // XXX: Should this only report unique canonical names?
  // An alias shares its GPU's GPUKind, so it is filtered alongside it.
  for (unsigned I = 0; I != std::size(AMDGPUGPUTable); ++I) {
    GPUKind Kind = static_cast<GPUKind>(AMDGPUFirstGPUKind + I);
    if (AMDGPUGPUTable[I].SubArch != Triple::NoSubArch &&
        isCPUValidForSubArch(SubArch, Kind))
      Values.push_back(AMDGPUNameStrTab[AMDGPUGPUTable[I].Name]);
  }

  for (const GPUNameAlias &A : AMDGPUGPUAliases) {
    if (isCPUValidForSubArch(SubArch, A.Kind))
      Values.push_back(AMDGPUNameStrTab[A.AltName]);
  }
}

void AMDGPU::fillValidArchListR600(SmallVectorImpl<StringRef> &Values) {
  for (const R600Info &Info : R600GPUTable)
    Values.push_back(R600NameStrTab[Info.Name]);
  for (const GPUNameAlias &A : R600GPUAliases)
    Values.push_back(R600NameStrTab[A.AltName]);
}

AMDGPU::IsaVersion AMDGPU::getIsaVersion(StringRef GPU) {
  const GPUInfo *Info = getAMDGPUInfo(parseArchAMDGCN(GPU));
  return Info ? Info->Version : IsaVersion{0, 0, 0};
}

AMDGPU::IsaVersion AMDGPU::getIsaVersion(Triple::SubArchType SubArch) {
  const GPUInfo *Info = getAMDGPUInfo(getGPUKindFromSubArch(SubArch));
  return Info ? Info->Version : IsaVersion{0, 0, 0};
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
  // A bare subarch triple (no -target-cpu) still pins down the target, so it is
  // not a null GPU: DefaultFeatures has already been populated from the
  // subarch.
  const bool IsNullGPU = T.getSubArch() == Triple::NoSubArch && GPU.empty();
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
  // With no explicit GPU, the triple's subarch identifies the target.
  AMDGPU::GPUKind Kind = GPU.empty() && T.getSubArch() != Triple::NoSubArch
                             ? getGPUKindFromSubArch(T.getSubArch())
                             : parseArchAMDGCN(GPU);
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
    Features["flat-global-insts"] = true;
    Features["bitop3-insts"] = true;
    Features["prng-inst"] = true;
    Features["tanh-insts"] = true;
    Features["tensor-cvt-lut-insts"] = true;
    Features["bf16-trans-insts"] = true;
    Features["bf16-cvt-insts"] = true;
    Features["cvt-sr-pk-bf16-f32-inst"] = true;
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
    Features["atomic-fmin-fmax-global-f32"] = true;
    Features["atomic-fmin-fmax-global-f64"] = true;
    Features["s-wakeup-barrier-inst"] = true;
    Features["f16bf16-to-fp6bf6-cvt-scale-insts"] = true;
    Features["f32-to-fp6bf6-cvt-scale-insts"] = true;
    Features["clusters"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["mqsad-pk-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["cvt-pknorm-vop3-insts"] = true;
    Features["image-insts"] = true;
    Features["extended-image-insts"] = true;
    Features["async-load-to-lds-insts"] = true;
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
    Features["cvt-sr-pk-bf16-f32-inst"] = true;
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
    Features["async-load-to-lds-insts"] = true;
    Features["async-store-from-lds-insts"] = true;
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
    Features["extended-image-insts"] = true;
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
    Features["smem-prefetch-insts"] = true;
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
    Features["extended-image-insts"] = true;
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
    Features["extended-image-insts"] = true;
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
    Features["extended-image-insts"] = true;
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
    Features["extended-image-insts"] = true;
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
    if (Kind != GK_GFX90A)
      Features["extended-image-insts"] = true;
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
    Features["extended-image-insts"] = true;
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
    Features["extended-image-insts"] = true;
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

  // Filter out unrecognized subarch suffixes.
  if (TT.getSubArch() == Triple::NoSubArch && TT.getArchName() != "amdgcn")
    return std::nullopt;

  // A named processor (i.e. not the empty/generic wildcard, which is resolved
  // from the triple's subarch) must be a recognized GPU that is consistent with
  // the triple's subarch.
  StringRef CPUName = ProcAndFeatures.split(':').first;
  if (!CPUName.empty() && CPUName != "generic" &&
      !isCPUValidForSubArch(TT.getSubArch(), CPUName))
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
