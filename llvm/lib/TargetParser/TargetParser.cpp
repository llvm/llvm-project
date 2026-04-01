//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/TargetParser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace AMDGPU;

/// Find KV in array using binary search.
static const BasicSubtargetSubTypeKV *
find(StringRef S, ArrayRef<BasicSubtargetSubTypeKV> A) {
  // Binary search the array
  auto F = llvm::lower_bound(A, S);
  // If not found then return NULL
  if (F == A.end() || StringRef(F->Key) != S)
    return nullptr;
  // Return the found array item
  return F;
}

/// For each feature that is (transitively) implied by this feature, set it.
static void setImpliedBits(FeatureBitset &Bits, const FeatureBitset &Implies,
                           ArrayRef<BasicSubtargetFeatureKV> FeatureTable) {
  // OR the Implies bits in outside the loop. This allows the Implies for CPUs
  // which might imply features not in FeatureTable to use this.
  Bits |= Implies;
  for (const auto &FE : FeatureTable)
    if (Implies.test(FE.Value))
      setImpliedBits(Bits, FE.Implies.getAsBitset(), FeatureTable);
}

std::optional<llvm::StringMap<bool>> llvm::getCPUDefaultTargetFeatures(
    StringRef CPU, ArrayRef<BasicSubtargetSubTypeKV> ProcDesc,
    ArrayRef<BasicSubtargetFeatureKV> ProcFeatures) {
  if (CPU.empty())
    return std::nullopt;

  const BasicSubtargetSubTypeKV *CPUEntry = ::find(CPU, ProcDesc);
  if (!CPUEntry)
    return std::nullopt;

  // Set the features implied by this CPU feature if there is a match.
  FeatureBitset Bits;
  llvm::StringMap<bool> DefaultFeatures;
  setImpliedBits(Bits, CPUEntry->Implies.getAsBitset(), ProcFeatures);

  [[maybe_unused]] unsigned BitSize = Bits.size();
  for (const BasicSubtargetFeatureKV &FE : ProcFeatures) {
    assert(FE.Value < BitSize && "Target Feature is out of range");
    if (Bits[FE.Value])
      DefaultFeatures[FE.Key] = true;
  }
  return DefaultFeatures;
}

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

StringRef llvm::AMDGPU::getArchNameAMDGCN(GPUKind AK) {
  switch (AK) {
#define AMDGCN_GPU(NAME, ENUM, ISAVERSION, FEATURES)                           \
  case ENUM:                                                                   \
    return NAME;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  default:
    return "";
  }
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
#define AMDGCN_GPU(NAME, ENUM, ISAVERSION, FEATURES) .Case(NAME, ENUM)
#define AMDGCN_GPU_ALIAS(NAME, ENUM) .Case(NAME, ENUM)
#include "llvm/TargetParser/AMDGPUTargetParser.def"
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
#define AMDGCN_GPU(NAME, ENUM, ISAVERSION, FEATURES)                           \
  case ENUM:                                                                   \
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

void AMDGPU::fillValidArchListAMDGCN(SmallVectorImpl<StringRef> &Values) {
  // XXX: Should this only report unique canonical names?
  Values.append({
#define AMDGCN_GPU(NAME, ENUM, ISAVERSION, FEATURES) NAME,
#define AMDGCN_GPU_ALIAS(NAME, ENUM) NAME,
#include "llvm/TargetParser/AMDGPUTargetParser.def"
  });
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
#define AMDGCN_GPU(NAME, ENUM, ISAVERSION, FEATURES)                           \
  case ENUM:                                                                   \
    return MAKE_ISAVERSION ISAVERSION;
#include "llvm/TargetParser/AMDGPUTargetParser.def"
#undef MAKE_ISAVERSION
  default:
    return {0, 0, 0};
  }
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
  case GK_GFX1251:
  case GK_GFX1250:
  case GK_GFX12_5_GENERIC:
    Features["swmmac-gfx1200-insts"] = true;
    Features["swmmac-gfx1250-insts"] = true;
    [[fallthrough]];
  case GK_GFX1310:
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
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
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
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["gfx12-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["image-insts"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["fp8-conversion-insts"] = true;
    Features["wmma-128b-insts"] = true;
    Features["swmmac-gfx1200-insts"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    break;
  case GK_GFX1170:
  case GK_GFX1171:
  case GK_GFX1172:
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
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["image-insts"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
    Features["cvt-pknorm-vop2-insts"] = true;
    Features["gws"] = true;
    Features["dot11-insts"] = true;
    Features["fp8-conversion-insts"] = true;
    Features["wmma-128b-insts"] = true;
    Features["swmmac-gfx1200-insts"] = true;
    Features["atomic-fmin-fmax-global-f32"] = true;
    break;
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
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["image-insts"] = true;
    Features["cube-insts"] = true;
    Features["lerp-inst"] = true;
    Features["sad-insts"] = true;
    Features["qsad-insts"] = true;
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
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
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
    Features["dl-insts"] = true;
    Features["ci-insts"] = true;
    Features["16-bit-insts"] = true;
    Features["dpp"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
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
    // AMDGCN SPIRV must support the union of all AMDGCN features. This list
    // should be kept in sorted order and updated whenever new features are
    // added.
    Features["16-bit-insts"] = true;
    Features["ashr-pk-insts"] = true;
    Features["atomic-buffer-pk-add-bf16-inst"] = true;
    Features["atomic-buffer-global-pk-add-f16-insts"] = true;
    Features["atomic-ds-pk-add-16-insts"] = true;
    Features["atomic-fadd-rtn-insts"] = true;
    Features["atomic-flat-pk-add-16-insts"] = true;
    Features["atomic-global-pk-add-bf16-inst"] = true;
    Features["bf16-trans-insts"] = true;
    Features["bf16-cvt-insts"] = true;
    Features["bf8-cvt-scale-insts"] = true;
    Features["bitop3-insts"] = true;
    Features["ci-insts"] = true;
    Features["dl-insts"] = true;
    Features["dot1-insts"] = true;
    Features["dot2-insts"] = true;
    Features["dot3-insts"] = true;
    Features["dot4-insts"] = true;
    Features["dot5-insts"] = true;
    Features["dot6-insts"] = true;
    Features["dot7-insts"] = true;
    Features["dot8-insts"] = true;
    Features["dot9-insts"] = true;
    Features["dot10-insts"] = true;
    Features["dot11-insts"] = true;
    Features["dot12-insts"] = true;
    Features["dot13-insts"] = true;
    Features["dpp"] = true;
    Features["f16bf16-to-fp6bf6-cvt-scale-insts"] = true;
    Features["f32-to-f16bf16-cvt-sr-insts"] = true;
    Features["fp4-cvt-scale-insts"] = true;
    Features["fp6bf6-cvt-scale-insts"] = true;
    Features["fp8e5m3-insts"] = true;
    Features["fp8-conversion-insts"] = true;
    Features["fp8-cvt-scale-insts"] = true;
    Features["fp8-insts"] = true;
    Features["gfx8-insts"] = true;
    Features["gfx9-insts"] = true;
    Features["gfx90a-insts"] = true;
    Features["gfx940-insts"] = true;
    Features["gfx950-insts"] = true;
    Features["gfx10-insts"] = true;
    Features["gfx10-3-insts"] = true;
    Features["gfx11-insts"] = true;
    Features["gfx12-insts"] = true;
    Features["gfx1250-insts"] = true;
    Features["gws"] = true;
    Features["image-insts"] = true;
    Features["mai-insts"] = true;
    Features["permlane16-swap"] = true;
    Features["permlane32-swap"] = true;
    Features["prng-inst"] = true;
    Features["setprio-inc-wg-inst"] = true;
    Features["s-memrealtime"] = true;
    Features["s-memtime-inst"] = true;
    Features["tanh-insts"] = true;
    Features["tensor-cvt-lut-insts"] = true;
    Features["transpose-load-f4f6-insts"] = true;
    Features["vmem-pref-insts"] = true;
    Features["vmem-to-lds-load-insts"] = true;
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
