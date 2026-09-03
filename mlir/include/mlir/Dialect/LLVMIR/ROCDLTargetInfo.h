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
  /// \p tripleOrChip is either a triple ("amdgpu9.42-amd-amdhsa", or the legacy
  /// subarch-less "amdgcn-amd-amdhsa") or a bare GPU name ("gfx942",
  /// "gfx9-4-generic"). \p chip plays the role of `-mcpu`: when given alongside
  /// a triple it names the exact GPU and must be compatible with the triple's
  /// subarch. \p features is an `-mattr`-style "+a,-b" list applied on top of
  /// the GPU's default features.
  ///
  /// Diagnostics are emitted via `emitError`.
  static FailureOr<TargetInfo>
  get(StringRef tripleOrChip, StringRef chip = "", StringRef features = "",
      function_ref<InFlightDiagnostic()> emitError = nullptr);

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
  /// (V#), or nullopt for an unknown target. This is a descriptor layout width
  /// rather than a capability, so it is a number: asking "does it have 45-bit
  /// num_records" only works while there are exactly two widths. There is no
  /// safe default, so a lowering that needs it must bail out when it is absent
  /// rather than guess.
  std::optional<unsigned> getBufferResourceNumRecordsWidth() const;

  /// Returns the maximum LDS in bytes a single workgroup can address, or
  /// nullopt for an unknown target. This is a fixed hardware cap and does not
  /// depend on how many SIMDs a workgroup runs on.
  std::optional<unsigned> getMaxAddressableLocalMemorySize() const;

  /// Returns the wavefront size, or nullopt for an unknown target. Targets that
  /// support both sizes report 32 unless "+wavefrontsize64" was requested.
  std::optional<unsigned> getWavefrontSize() const;

  /// Returns whether the GPU runs at either wavefront size, so that the choice
  /// comes from the features rather than from the GPU. This is the one thing a
  /// triple alone cannot express, and it is not recoverable from the resolved
  /// features, which always name a size.
  bool supportsBothWavefrontSizes() const { return dualWavefrontSize; }

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
  bool dualWavefrontSize = false;
};

} // namespace mlir::ROCDL

#endif // MLIR_DIALECT_LLVMIR_ROCDLTARGETINFO_H_
