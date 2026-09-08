//===- ROCDLTargetInfo.cpp - AMDGPU target description --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::ROCDL;

namespace AMDGPU = ::llvm::AMDGPU;
using ::llvm::Triple;

/// Reports \p message through \p emitError if it is non-null, and returns
/// failure.
static LogicalResult fail(function_ref<InFlightDiagnostic()> emitError,
                          const Twine &message) {
  if (emitError)
    emitError() << message;
  return failure();
}

std::optional<AMDGPU::TargetID> TargetInfo::parseTargetID(StringRef arch) {
  // If we see a five-component triple, that's maximally authoritative.
  SmallVector<StringRef, 5> parts;
  arch.split(parts, '-', /*MaxSplit=*/4);
  if (parts.size() == 5)
    return AMDGPU::TargetID::parseTargetIDString(arch);

  // Otherwise, handle bare triples.
  Triple triple(Triple::normalize(arch));
  if (triple.isAMDGCN())
    return AMDGPU::TargetID::parse(triple, "");

  // Otherwise, take the implicit amdgcn-amd-amdhsa legacy triple and pair it
  // with an arch name.
  return AMDGPU::TargetID::parse(Triple("amdgcn-amd-amdhsa"), arch);
}

/// Pins the wavefront size in \p bits, mirroring the policy LLVM applies in
/// fillAMDGCNFeatureMap: a target that only runs at one size rejects a request
/// for the other, and a target that runs at either defaults to wave32.
static LogicalResult
resolveWavefrontSize(AMDGPU::AMDGPUFeatureBitset &bits, unsigned waveSize,
                     function_ref<InFlightDiagnostic()> emitError) {
  bool targetWave32 = bits.test(AMDGPU::FEAT_WAVEFRONTSIZE32);
  bool targetWave64 = bits.test(AMDGPU::FEAT_WAVEFRONTSIZE64);

  switch (waveSize) {
  case 0:
    // A target that runs at either size and was not asked for one runs wave32.
    if (!targetWave32 && !targetWave64)
      bits.set(AMDGPU::FEAT_WAVEFRONTSIZE32);
    return success();
  case 32:
    if (targetWave64)
      return fail(emitError, "target only supports a wavefront size of 64");
    bits.set(AMDGPU::FEAT_WAVEFRONTSIZE32);
    return success();
  case 64:
    if (targetWave32)
      return fail(emitError, "target only supports a wavefront size of 32");
    bits.set(AMDGPU::FEAT_WAVEFRONTSIZE64);
    return success();
  default:
    return fail(emitError,
                "wavefront size must be 32 or 64, got " + Twine(waveSize));
  }
}

FailureOr<TargetInfo>
TargetInfo::get(StringRef arch, unsigned waveSize,
                function_ref<InFlightDiagnostic()> emitError) {
  if (arch.empty())
    return fail(emitError, "target architecture cannot be empty");

  std::optional<AMDGPU::TargetID> id = parseTargetID(arch);
  if (!id)
    return fail(emitError, "'" + arch +
                               "' is not a valid AMDGPU architecture: expected "
                               "a GPU name, a triple, or a target ID");

  TargetInfo info;
  info.kind = id->getGPUKind();
  info.subArch = AMDGPU::getSubArch(info.kind);
  info.featureBits = AMDGPU::getFeatureBitset(info.kind);
  info.xnackSetting = id->getXnackSetting();
  info.sramEccSetting = id->getSramEccSetting();

  // Recorded before we record the user's choice since that lives in the same
  // bitmap.
  info.dualWavefrontSize =
      !info.isUnknown() &&
      !info.featureBits.test(AMDGPU::FEAT_WAVEFRONTSIZE32) &&
      !info.featureBits.test(AMDGPU::FEAT_WAVEFRONTSIZE64);

  if (info.isUnknown() && waveSize != 0 && waveSize != 32 && waveSize != 64)
    return fail(emitError,
                "wavefront size must be 32 or 64, got " + Twine(waveSize));
  if (!info.isUnknown() &&
      failed(resolveWavefrontSize(info.featureBits, waveSize, emitError)))
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

std::optional<unsigned> TargetInfo::getTotalNumSGPRs() const {
  if (isUnknown())
    return std::nullopt;
  return AMDGPU::getTotalNumSGPRs(kind);
}

std::optional<unsigned> TargetInfo::getAddressableNumSGPRs() const {
  if (isUnknown())
    return std::nullopt;
  return AMDGPU::getAddressableNumSGPRs(kind);
}

std::optional<unsigned> TargetInfo::getSGPRAllocGranule() const {
  if (isUnknown())
    return std::nullopt;
  return AMDGPU::getSGPRAllocGranule(kind);
}

std::optional<unsigned> TargetInfo::getVGPRAllocGranule() const {
  // The granule depends on the wavefront size, which get() has already pinned.
  std::optional<unsigned> waveSize = getWavefrontSize();
  if (isUnknown() || !waveSize)
    return std::nullopt;
  return AMDGPU::getVGPRAllocGranule(kind, /*IsWave32=*/*waveSize == 32);
}

std::optional<unsigned> TargetInfo::getLDSBankCount() const {
  if (isUnknown())
    return std::nullopt;
  return AMDGPU::getLDSBankCount(kind);
}

std::optional<unsigned> TargetInfo::getMaxWavesPerEU() const {
  if (isUnknown())
    return std::nullopt;
  return AMDGPU::getMaxWavesPerEU(kind);
}

std::optional<unsigned> TargetInfo::getWavefrontSize() const {
  if (has(AMDGPU::FEAT_WAVEFRONTSIZE64))
    return 64;
  if (has(AMDGPU::FEAT_WAVEFRONTSIZE32))
    return 32;
  return std::nullopt;
}

void TargetInfo::migrateArchFeaturesToModuleFlags(Operation *op) const {
  assert(LLVM::satisfiesLLVMModule(op) &&
         "xnack and sramecc describe a whole code object, so they can only be "
         "recorded on a module");
  ROCDLDialect *dialect =
      op->getContext()->getOrLoadDialect<ROCDL::ROCDLDialect>();
  Builder builder(op->getContext());
  // The helpers differ in type, hence the generic lambda.
  auto migrate = [&](AMDGPU::TargetIDSetting setting, auto helper) {
    if (setting != AMDGPU::TargetIDSetting::On &&
        setting != AMDGPU::TargetIDSetting::Off)
      return;
    helper.setAttr(op,
                   builder.getBoolAttr(setting == AMDGPU::TargetIDSetting::On));
  };
  migrate(xnackSetting, dialect->getXnackAttrHelper());
  migrate(sramEccSetting, dialect->getSrameccAttrHelper());
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
