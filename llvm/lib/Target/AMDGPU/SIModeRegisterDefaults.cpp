//===-- SIModeRegisterDefaults.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SIModeRegisterDefaults.h"

using namespace llvm;

SIModeRegisterDefaults::SIModeRegisterDefaults(const Function &F) {
  *this = getDefaultForCallingConv(F.getCallingConv());

  StringRef IEEEAttr = F.getFnAttribute("amdgpu-ieee").getValueAsString();
  if (!IEEEAttr.empty())
    IEEE = IEEEAttr == "true";

  StringRef DX10ClampAttr =
      F.getFnAttribute("amdgpu-dx10-clamp").getValueAsString();
  if (!DX10ClampAttr.empty())
    DX10Clamp = DX10ClampAttr == "true";

  StringRef DenormF32Attr =
      F.getFnAttribute("denormal-fp-math-f32").getValueAsString();
  if (!DenormF32Attr.empty())
    FP32Denormals = parseDenormalFPAttribute(DenormF32Attr);

  StringRef DenormAttr =
      F.getFnAttribute("denormal-fp-math").getValueAsString();
  if (!DenormAttr.empty()) {
    DenormalMode DenormMode = parseDenormalFPAttribute(DenormAttr);
    if (DenormF32Attr.empty())
      FP32Denormals = DenormMode;
    FP64FP16Denormals = DenormMode;
  }
}

using namespace AMDGPU;

/// Combine f32 and f64 rounding modes into a combined rounding mode value.
static constexpr uint32_t getModeRegisterRoundMode(uint32_t HWFP32Val,
                                                   uint32_t HWFP64Val) {
  return HWFP32Val << F32FltRoundOffset | HWFP64Val << F64FltRoundOffset;
}

static constexpr uint64_t encodeFltRoundsTable(uint32_t FltRoundsVal,
                                               uint32_t HWF32Val,
                                               uint32_t HWF64Val) {
  uint32_t ModeVal = getModeRegisterRoundMode(HWF32Val, HWF64Val);
  if (FltRoundsVal > TowardNegative)
    FltRoundsVal -= ExtendedFltRoundOffset;

  uint32_t BitIndex = ModeVal << 2;
  return static_cast<uint64_t>(FltRoundsVal) << BitIndex;
}

// Encode FLT_ROUNDS value where the two rounding modes are the same and use a
// standard value
static constexpr uint64_t
encodeFltRoundsTableSame(AMDGPUFltRounds FltRoundsMode, uint32_t HWVal) {
  return encodeFltRoundsTable(FltRoundsMode, HWVal, HWVal);
}

// Convert mode register encoded rounding mode to AMDGPUFltRounds
static constexpr AMDGPUFltRounds
decodeIndexFltRoundConversionTable(uint32_t HWMode) {
  uint32_t TableRead = (FltRoundConversionTable >> (HWMode << 2)) & 0xf;
  if (TableRead > TowardNegative)
    TableRead += ExtendedFltRoundOffset;
  return static_cast<AMDGPUFltRounds>(TableRead);
}

static constexpr uint32_t HWTowardZero = FP_ROUND_ROUND_TO_ZERO;
static constexpr uint32_t HWNearestTiesToEven = FP_ROUND_ROUND_TO_NEAREST;
static constexpr uint32_t HWTowardPositive = FP_ROUND_ROUND_TO_INF;
static constexpr uint32_t HWTowardNegative = FP_ROUND_ROUND_TO_NEGINF;

constexpr uint64_t AMDGPU::FltRoundConversionTable =
    encodeFltRoundsTableSame(TowardZeroF32_TowardZeroF64, HWTowardZero) |
    encodeFltRoundsTableSame(NearestTiesToEvenF32_NearestTiesToEvenF64,
                             HWNearestTiesToEven) |
    encodeFltRoundsTableSame(TowardPositiveF32_TowardPositiveF64,
                             HWTowardPositive) |
    encodeFltRoundsTableSame(TowardNegativeF32_TowardNegativeF64,
                             HWTowardNegative) |

    encodeFltRoundsTable(TowardZeroF32_NearestTiesToEvenF64, HWTowardZero,
                         HWNearestTiesToEven) |
    encodeFltRoundsTable(TowardZeroF32_TowardPositiveF64, HWTowardZero,
                         HWTowardPositive) |
    encodeFltRoundsTable(TowardZeroF32_TowardNegativeF64, HWTowardZero,
                         HWTowardNegative) |

    encodeFltRoundsTable(NearestTiesToEvenF32_TowardZeroF64,
                         HWNearestTiesToEven, HWTowardZero) |
    encodeFltRoundsTable(NearestTiesToEvenF32_TowardPositiveF64,
                         HWNearestTiesToEven, HWTowardPositive) |
    encodeFltRoundsTable(NearestTiesToEvenF32_TowardNegativeF64,
                         HWNearestTiesToEven, HWTowardNegative) |

    encodeFltRoundsTable(TowardPositiveF32_TowardZeroF64, HWTowardPositive,
                         HWTowardZero) |
    encodeFltRoundsTable(TowardPositiveF32_NearestTiesToEvenF64,
                         HWTowardPositive, HWNearestTiesToEven) |
    encodeFltRoundsTable(TowardPositiveF32_TowardNegativeF64, HWTowardPositive,
                         HWTowardNegative) |

    encodeFltRoundsTable(TowardNegativeF32_TowardZeroF64, HWTowardNegative,
                         HWTowardZero) |
    encodeFltRoundsTable(TowardNegativeF32_NearestTiesToEvenF64,
                         HWTowardNegative, HWNearestTiesToEven) |
    encodeFltRoundsTable(TowardNegativeF32_TowardPositiveF64, HWTowardNegative,
                         HWTowardPositive);

// Verify evaluation of FltRoundConversionTable

// If both modes are the same, should return the standard values.
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWTowardZero, HWTowardZero)) == AMDGPUFltRounds::TowardZero);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWNearestTiesToEven, HWNearestTiesToEven)) ==
              AMDGPUFltRounds::NearestTiesToEven);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWTowardPositive, HWTowardPositive)) ==
              AMDGPUFltRounds::TowardPositive);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWTowardNegative, HWTowardNegative)) ==
              AMDGPUFltRounds::TowardNegative);

static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWTowardZero, HWNearestTiesToEven)) ==
              TowardZeroF32_NearestTiesToEvenF64);
static_assert(decodeIndexFltRoundConversionTable(
                  getModeRegisterRoundMode(HWTowardZero, HWTowardPositive)) ==
              TowardZeroF32_TowardPositiveF64);
static_assert(decodeIndexFltRoundConversionTable(
                  getModeRegisterRoundMode(HWTowardZero, HWTowardNegative)) ==
              TowardZeroF32_TowardNegativeF64);

static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWNearestTiesToEven, HWTowardZero)) ==
              NearestTiesToEvenF32_TowardZeroF64);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWNearestTiesToEven, HWTowardPositive)) ==
              NearestTiesToEvenF32_TowardPositiveF64);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWNearestTiesToEven, HWTowardNegative)) ==
              NearestTiesToEvenF32_TowardNegativeF64);

static_assert(decodeIndexFltRoundConversionTable(
                  getModeRegisterRoundMode(HWTowardPositive, HWTowardZero)) ==
              TowardPositiveF32_TowardZeroF64);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWTowardPositive, HWNearestTiesToEven)) ==
              TowardPositiveF32_NearestTiesToEvenF64);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWTowardPositive, HWTowardNegative)) ==
              TowardPositiveF32_TowardNegativeF64);

static_assert(decodeIndexFltRoundConversionTable(
                  getModeRegisterRoundMode(HWTowardNegative, HWTowardZero)) ==
              TowardNegativeF32_TowardZeroF64);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWTowardNegative, HWNearestTiesToEven)) ==
              TowardNegativeF32_NearestTiesToEvenF64);
static_assert(decodeIndexFltRoundConversionTable(getModeRegisterRoundMode(
                  HWTowardNegative, HWTowardPositive)) ==
              TowardNegativeF32_TowardPositiveF64);
