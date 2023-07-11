//===-- SIModeRegisterDefaults.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIMODEREGISTERDEFAULTS_H
#define LLVM_LIB_TARGET_AMDGPU_SIMODEREGISTERDEFAULTS_H

#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/FloatingPointMode.h"

namespace llvm {

// Track defaults for fields in the MODE register.
struct SIModeRegisterDefaults {
  /// Floating point opcodes that support exception flag gathering quiet and
  /// propagate signaling NaN inputs per IEEE 754-2008. Min_dx10 and max_dx10
  /// become IEEE 754- 2008 compliant due to signaling NaN propagation and
  /// quieting.
  bool IEEE : 1;

  /// Used by the vector ALU to force DX10-style treatment of NaNs: when set,
  /// clamp NaN to zero; otherwise, pass NaN through.
  bool DX10Clamp : 1;

  /// If this is set, neither input or output denormals are flushed for most f32
  /// instructions.
  DenormalMode FP32Denormals;

  /// If this is set, neither input or output denormals are flushed for both f64
  /// and f16/v2f16 instructions.
  DenormalMode FP64FP16Denormals;

  SIModeRegisterDefaults() :
    IEEE(true),
    DX10Clamp(true),
    FP32Denormals(DenormalMode::getIEEE()),
    FP64FP16Denormals(DenormalMode::getIEEE()) {}

  SIModeRegisterDefaults(const Function &F);

  static SIModeRegisterDefaults getDefaultForCallingConv(CallingConv::ID CC) {
    SIModeRegisterDefaults Mode;
    Mode.IEEE = !AMDGPU::isShader(CC);
    return Mode;
  }

  bool operator==(const SIModeRegisterDefaults Other) const {
    return IEEE == Other.IEEE && DX10Clamp == Other.DX10Clamp &&
           FP32Denormals == Other.FP32Denormals &&
           FP64FP16Denormals == Other.FP64FP16Denormals;
  }

  /// Get the encoding value for the FP_DENORM bits of the mode register for the
  /// FP32 denormal mode.
  uint32_t fpDenormModeSPValue() const {
    if (FP32Denormals == DenormalMode::getPreserveSign())
      return FP_DENORM_FLUSH_IN_FLUSH_OUT;
    if (FP32Denormals.Output == DenormalMode::PreserveSign)
      return FP_DENORM_FLUSH_OUT;
    if (FP32Denormals.Input == DenormalMode::PreserveSign)
      return FP_DENORM_FLUSH_IN;
    return FP_DENORM_FLUSH_NONE;
  }

  /// Get the encoding value for the FP_DENORM bits of the mode register for the
  /// FP64/FP16 denormal mode.
  uint32_t fpDenormModeDPValue() const {
    if (FP64FP16Denormals == DenormalMode::getPreserveSign())
      return FP_DENORM_FLUSH_IN_FLUSH_OUT;
    if (FP64FP16Denormals.Output == DenormalMode::PreserveSign)
      return FP_DENORM_FLUSH_OUT;
    if (FP64FP16Denormals.Input == DenormalMode::PreserveSign)
      return FP_DENORM_FLUSH_IN;
    return FP_DENORM_FLUSH_NONE;
  }

  // FIXME: Inlining should be OK for dx10-clamp, since the caller's mode should
  // be able to override.
  bool isInlineCompatible(SIModeRegisterDefaults CalleeMode) const {
    return DX10Clamp == CalleeMode.DX10Clamp && IEEE == CalleeMode.IEEE;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SIMODEREGISTERDEFAULTS_H
