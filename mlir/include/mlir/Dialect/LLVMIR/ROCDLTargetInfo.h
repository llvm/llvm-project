//===- ROCDLTargetInfo.h - AMDGPU target description ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_LLVMIR_ROCDLTARGETINFO_H_
#define MLIR_DIALECT_LLVMIR_ROCDLTARGETINFO_H_

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

namespace mlir::ROCDL {

/// Describes the AMDGPU target a lowering is producing code for: the triple's
/// subarch (which identifies the GPU) together with the resolved set of
/// frontend-visible target features.
///
/// Lowerings should gate on features (`has(FEAT_...)`) rather than on ISA
/// version arithmetic, and add features if necessary.
class TargetInfo {
public:
  using Feature = ::llvm::AMDGPU::AMDGPUFeature;

  /// Constructs an unknown target: no subarch, and every feature query answers
  /// false.
  TargetInfo() = default;

  /// Resolves a target description.
  ///
  /// \p arch names the architecture the way Clang does, and accepts any of:
  ///
  ///   - a full target ID, "<triple>-<processor>[:<feature><+|->]*", such as
  ///     "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-" (what `rocminfo` prints
  ///     for a device's ISA) or "amdgpu9.0a-amd-amdhsa--gfx90a";
  ///   - a triple on its own, such as "amdgpu9.42-amd-amdhsa" or the legacy
  ///     subarch-less "amdgcn-amd-amdhsa";
  ///   - a processor on its own, with optional target-ID modifiers: "gfx942",
  ///     "gfx942:xnack+", "gfx9-4-generic".
  ///
  /// Only xnack and sramecc may be given as modifiers, and only on a processor
  /// that supports them; this is the same grammar `clang::parseTargetID`
  /// accepts, and it is validated by `llvm::AMDGPU::TargetID`.
  ///
  /// \p waveSize pins the wavefront size for targets that run at either, and
  /// must be 0 (meaning the target's own default), 32, or 64.
  ///
  /// Diagnostics are emitted via `emitError`.
  static FailureOr<TargetInfo>
  get(StringRef arch, unsigned waveSize = 0,
      function_ref<InFlightDiagnostic()> emitError = nullptr);

  /// Parses \p arch into a target ID, accepting the spellings `get()`
  /// documents, or returns nullopt if it names no valid target.
  ///
  /// Use only if you need to get the individual components of the target ID.
  static std::optional<::llvm::AMDGPU::TargetID> parseTargetID(StringRef arch);

  /// Returns whether the target has \p feature.
  bool has(Feature feature) const { return featureBits.test(feature); }

  /// Returns whether the target's fp8 conversions exist and use the OCP formats
  /// (E4M3FN/E5M2) rather than the FNUZ ones.
  bool hasOcpFp8() const {
    return has(::llvm::AMDGPU::FEAT_OCP_FP8_CONVERSION_INSTS);
  }

  /// Returns whether the target has fp8 conversions that use the FNUZ formats
  /// (E4M3FNUZ/E5M2FNUZ).
  bool hasFnuzFp8() const {
    return has(::llvm::AMDGPU::FEAT_FP8_CONVERSION_INSTS) && !hasOcpFp8();
  }

  /// Returns whether the target belongs to gfx generation \p major (9 for any
  /// gfx9xx, 12 for any gfx12xx, ...).
  ///
  /// Prefer `has()` where a feature expresses the condition; this is used when
  /// no feature exists and the property being checked is a function of the
  /// major ISA generation (such as the details of buffer encoding).
  bool isGeneration(unsigned major) const;

  /// Returns the width in bits of the num_records field of the buffer resource
  /// (V#), or nullopt for an unknown target.
  std::optional<unsigned> getBufferResourceNumRecordsWidth() const;

  /// Returns the maximum LDS in bytes a single workgroup can address, or
  /// nullopt for an unknown target.
  std::optional<unsigned> getMaxAddressableLocalMemorySize() const;

  /// Returns the wavefront size, or nullopt for an unknown target. Targets that
  /// support both sizes report 32 unless "+wavefrontsize64" was requested.
  std::optional<unsigned> getWavefrontSize() const;

  /// Returns whether the GPU can be configured for 32-lane or 64-lane
  /// wavefronts.
  bool supportsBothWavefrontSizes() const { return dualWavefrontSize; }

  /// Returns the total number of SGPRs, or nullopt for an unknown target.
  std::optional<unsigned> getTotalNumSGPRs() const;

  /// Returns the number of SGPRs addressable by a kernel, or nullopt for an
  /// unknown target. This is below getTotalNumSGPRs() where some are reserved.
  std::optional<unsigned> getAddressableNumSGPRs() const;

  /// Returns the SGPR allocation granularity in registers, or nullopt for an
  /// unknown target.
  std::optional<unsigned> getSGPRAllocGranule() const;

  /// Returns the VGPR allocation granularity in registers, or nullopt for an
  /// unknown target. This property is wavesize-dependent.
  std::optional<unsigned> getVGPRAllocGranule() const;

  /// Returns the number of LDS banks per compute unit, or nullopt for an
  /// unknown target.
  std::optional<unsigned> getLDSBankCount() const;

  /// Returns the maximum number of waves per execution unit, ignoring any
  /// limits a particular kernel imposes, or nullopt for an unknown target.
  std::optional<unsigned> getMaxWavesPerEU() const;

  /// Returns whether xnack is on, off, either, or unsupported on this target.
  /// "Any" means the target supports both and no `:xnack+/-` modifier was used.
  ::llvm::AMDGPU::TargetIDSetting getXnackSetting() const {
    return xnackSetting;
  }

  /// Returns whether sramecc is on, off, either, or unsupported, as for
  /// getXnackSetting().
  ::llvm::AMDGPU::TargetIDSetting getSramEccSetting() const {
    return sramEccSetting;
  }

  /// Records the xnack and sramecc settings this target's ID pinned onto the
  /// module \p op, as the `rocdl.xnack` and `rocdl.sramecc` attributes that
  /// translate to the `amdgpu.xnack` and `amdgpu.sramecc` module flags.
  ///
  /// These flags are given as `:{xnack,sramecc}` target-ID "modifiers",
  /// since they used to be subtarget features, but now frontends (like us and
  /// Clang) need to migrate them into module flags. This representation keeps
  /// us compatible with Clang and the output of tools like `rocminfo`.
  ///
  /// If a particular modifier is not given, no attribute is set for it, putting
  /// that value into its "any" state if it is controllable.
  void migrateArchFeaturesToModuleFlags(Operation *op) const;

  /// Returns the ISA version. For a generic target this is the floor of the
  /// family it covers (gfx9-4-generic reports 9.4.0), so it must not be used to
  /// decide whether an instruction is available.
  ::llvm::AMDGPU::IsaVersion getIsaVersion() const;

  ::llvm::Triple::SubArchType getSubArch() const { return subArch; }
  ::llvm::AMDGPU::GPUKind getGPUKind() const { return kind; }

  /// Returns the canonical GPU name ("gfx942", "gfx9-4-generic"), or "" if the
  /// target is unknown.
  StringRef getArchName() const;

  /// Returns whether this is a "gfxN-generic" target, which carries only the
  /// features common to every GPU it covers.
  bool isGeneric() const;

  /// Returns whether no GPU was identified, in which case every feature query
  /// answers false.
  bool isUnknown() const { return kind == ::llvm::AMDGPU::GK_NONE; }

  const ::llvm::AMDGPU::AMDGPUFeatureBitset &getFeatures() const {
    return featureBits;
  }

private:
  ::llvm::Triple::SubArchType subArch = ::llvm::Triple::NoSubArch;
  ::llvm::AMDGPU::GPUKind kind = ::llvm::AMDGPU::GK_NONE;
  ::llvm::AMDGPU::AMDGPUFeatureBitset featureBits;
  ::llvm::AMDGPU::TargetIDSetting xnackSetting =
      ::llvm::AMDGPU::TargetIDSetting::Unsupported;
  ::llvm::AMDGPU::TargetIDSetting sramEccSetting =
      ::llvm::AMDGPU::TargetIDSetting::Unsupported;
  bool dualWavefrontSize = false;
};

/// Returns the target architecture that a pass should parse, given its `arch`
/// option and the value of the deprecated alias that `arch` replaced.
///
/// The alias is only consulted when `arch` is left at "invalid".
StringRef resolveArchOption(StringRef arch, StringRef deprecatedAlias);

} // namespace mlir::ROCDL

#endif // MLIR_DIALECT_LLVMIR_ROCDLTARGETINFO_H_
