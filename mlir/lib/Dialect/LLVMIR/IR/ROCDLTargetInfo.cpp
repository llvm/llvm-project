//===- ROCDLTargetInfo.cpp - AMDGPU target description --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::ROCDL;

namespace AMDGPU = ::llvm::AMDGPU;
using ::llvm::Triple;

namespace {
/// Reports \p message through \p emitError if it is non-null, and returns
/// failure.
LogicalResult fail(function_ref<InFlightDiagnostic()> emitError,
                   const Twine &message) {
  if (emitError)
    emitError() << message;
  return failure();
}
} // namespace

/// Resolves the wavefront size in \p bits, mirroring the policy LLVM applies in
/// fillAMDGCNFeatureMap: a target that only runs at one size rejects a request
/// for the other, and a target that supports both defaults to wave32.
static LogicalResult
resolveWavefrontSize(AMDGPU::AMDGPUFeatureBitset &bits, bool targetWave32,
                     bool targetWave64,
                     function_ref<InFlightDiagnostic()> emitError) {
  bool wave32 = bits.test(AMDGPU::FEAT_WAVEFRONTSIZE32);
  bool wave64 = bits.test(AMDGPU::FEAT_WAVEFRONTSIZE64);

  if (wave32 && wave64)
    return fail(emitError,
                "'+wavefrontsize32' and '+wavefrontsize64' are mutually "
                "exclusive");
  if (targetWave64 && !wave64)
    return fail(emitError, "target only supports wavefrontsize64");
  if (targetWave32 && !wave32)
    return fail(emitError, "target only supports wavefrontsize32");

  // A target that supports both sizes and was not asked for one runs wave32.
  if (!wave32 && !wave64)
    bits.set(AMDGPU::FEAT_WAVEFRONTSIZE32);
  return success();
}

FailureOr<TargetInfo>
TargetInfo::get(StringRef tripleOrChip, StringRef chip, StringRef features,
                function_ref<InFlightDiagnostic()> emitError) {
  if (tripleOrChip.empty())
    return fail(emitError, "target triple cannot be empty");

  TargetInfo info;

  // A bare GPU name is accepted in place of a triple, so that "gfx942" keeps
  // working where a chipset used to be given.
  if (AMDGPU::GPUKind named = AMDGPU::parseArchAMDGCN(tripleOrChip)) {
    if (!chip.empty() && chip != tripleOrChip)
      return fail(emitError,
                  "conflicting GPUs '" + tripleOrChip + "' and '" + chip + "'");
    info.kind = named;
    info.subArch = AMDGPU::getSubArch(named);
  } else {
    Triple triple(Triple::normalize(tripleOrChip));
    if (!triple.isAMDGCN())
      return fail(emitError,
                  "'" + tripleOrChip + "' is not an AMDGCN triple or GPU name");

    info.subArch = triple.getSubArch();
    // Triple parsing maps any unrecognized "amdgpu..." arch to NoSubArch
    // without complaining, so a typo would otherwise be silently accepted as a
    // target with no features. Only the bare "amdgcn"/"amdgpu" spellings
    // legitimately carry no subarch.
    if (info.subArch == Triple::NoSubArch && triple.getArchName().size() != 6)
      return fail(emitError, "unknown AMDGPU subarchitecture in triple '" +
                                 tripleOrChip + "'");

    if (!chip.empty()) {
      if (!AMDGPU::isCPUValidForSubArch(info.subArch, chip))
        return fail(emitError, "GPU '" + chip + "' is not valid for triple '" +
                                   tripleOrChip + "'");
      info.kind = AMDGPU::parseArchAMDGCN(chip);
      // The chip pins down the exact GPU, which may be more specific than the
      // triple's family subarch.
      info.subArch = AMDGPU::getSubArch(info.kind);
    } else {
      info.kind = AMDGPU::getGPUKindFromSubArch(info.subArch);
    }
  }

  info.featureBits = AMDGPU::getFeatureBitset(info.kind);

  bool targetWave32 = info.featureBits.test(AMDGPU::FEAT_WAVEFRONTSIZE32);
  bool targetWave64 = info.featureBits.test(AMDGPU::FEAT_WAVEFRONTSIZE64);
  // Recorded before the modifiers and the default below pin a size.
  info.dualWavefrontSize = !info.isUnknown() && !targetWave32 && !targetWave64;

  if (std::optional<StringRef> bad =
          AMDGPU::applyFeatureModifiers(features, info.featureBits))
    return fail(emitError, "invalid target feature '" + *bad + "'");

  if (!info.isUnknown() &&
      failed(resolveWavefrontSize(info.featureBits, targetWave32, targetWave64,
                                  emitError)))
    return failure();

  return info;
}

bool TargetInfo::isGeneration(unsigned major) const {
  // The generation features are cumulative: a gfx12 target has every
  // FEAT_GFX*_INSTS bit from gfx8 up to gfx12. So a target is *in* generation N
  // when it has N's bit but not N+1's. This holds for generic targets too,
  // unlike comparing ISA versions.
  auto hasGen = [&](unsigned gen) {
    switch (gen) {
    case 7:
      return has(AMDGPU::FEAT_CI_INSTS);
    case 8:
      return has(AMDGPU::FEAT_GFX8_INSTS);
    case 9:
      return has(AMDGPU::FEAT_GFX9_INSTS);
    case 10:
      return has(AMDGPU::FEAT_GFX10_INSTS);
    case 11:
      return has(AMDGPU::FEAT_GFX11_INSTS);
    case 12:
      return has(AMDGPU::FEAT_GFX12_INSTS);
    case 13:
      return has(AMDGPU::FEAT_GFX13_INSTS);
    default:
      return false;
    }
  };

  if (isUnknown())
    return false;
  // gfx6 is the base: it has none of the generation features.
  if (major == 6)
    return !hasGen(7);
  return hasGen(major) && !hasGen(major + 1);
}

std::optional<unsigned> TargetInfo::getBufferResourceNumRecordsWidth() const {
  return AMDGPU::getBufferResourceNumRecordsWidth(kind);
}

std::optional<unsigned> TargetInfo::getMaxAddressableLocalMemorySize() const {
  if (isUnknown())
    return std::nullopt;
  return AMDGPU::getMaxHWAddressableLocalMemorySize(kind);
}

std::optional<unsigned> TargetInfo::getWavefrontSize() const {
  if (has(AMDGPU::FEAT_WAVEFRONTSIZE64))
    return 64;
  if (has(AMDGPU::FEAT_WAVEFRONTSIZE32))
    return 32;
  return std::nullopt;
}

AMDGPU::IsaVersion TargetInfo::getIsaVersion() const {
  return AMDGPU::getIsaVersion(subArch);
}

StringRef TargetInfo::getArchName() const {
  return AMDGPU::getArchNameAMDGCN(kind);
}

bool TargetInfo::isGeneric() const {
  return !isUnknown() && AMDGPU::getMajorSubArch(subArch) == subArch;
}
