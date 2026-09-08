//===-- AMDGPUCoExecInfo.h - Co-execution info ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Shared types for co-execution modeling used by GCNHazardRecognizer and the
/// schedulers.
///
/// Multi-cycle instructions (WMMA, TRANS, etc.) have execution windows where
/// other instruction types can co-execute. For WMMA, slot patterns depend on
/// the variant:
///
///   E0 (Issue): Control instructions only (s_delay_alu, s_set_vgpr_msb)
///   E (External): Memory and SALU can co-execute, no VALU
///   I (Internal): VALU, TRANS, memory, and SALU can all co-execute
///   V (Vacant): Memory/SALU/next-WMMA ok, NO VALU/TRANS
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECINFO_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECINFO_H

#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <cstdint>
#include <optional>

namespace llvm {

namespace AMDGPU {

//===----------------------------------------------------------------------===//
// Co-execution Bitmasks
//===----------------------------------------------------------------------===//

/// Bitmask for instruction types allowed to co-execute at a stage.
enum class CoExecMask : uint16_t {
  None = 0,
  CTRL = 1 << 0,  // Control: s_delay_alu, s_set_vgpr_msb
  VALU = 1 << 1,  // Vector ALU
  TRANS = 1 << 2, // Transcendentals (V_EXP etc)
  SALU = 1 << 3,  // Scalar ALU
  DS = 1 << 4,    // LDS read/write
  VMEM = 1 << 5,  // Global memory
  SMEM = 1 << 6,  // Scalar memory
  WMMA = 1 << 7,  // Next WMMA (V stages only), or MFMA
  All = 0xFFFF,

  MEM = DS | VMEM | SMEM,
  StageE0 = CTRL,                            // Issue: control only
  StageE = CTRL | SALU | MEM,                // External: mem/salu
  StageI = CTRL | SALU | MEM | VALU | TRANS, // Internal: all ALU
  // Internal + scaled-WMMA absorb: same as StageI but the next scaled
  // WMMA may issue here - its LD_SCALE consumes the I cycle and the matrix
  // multiply lands in the V slot that follows. Used for the last I before
  // V of scaled patterns.
  StageIS = StageI | WMMA,
  StageV = CTRL | SALU | MEM | WMMA, // Vacant: no valu/trans
  StageTR = All & ~TRANS,            // TRANS co-exec: no TRANS

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/All)
};

using CoExecMaskT = CoExecMask;

//===----------------------------------------------------------------------===//
// Instruction Flavor Classification
//===----------------------------------------------------------------------===//

/// Classification of instructions by execution characteristics.
/// Used for scheduling decisions and co-execution slot preferences.
enum class InstructionFlavor : uint8_t {
  WMMA,            // WMMA/MFMA matrix operations
  SingleCycleVALU, // Single-cycle VALU (not TRANS, not multi-cycle CVT)
  TRANS,           // Transcendental ops (v_exp, v_log, etc.)
  MultiCycleVALU,  // VALU instructions with repeat rate > 1
  VMEM,            // FLAT/GLOBAL memory operations
  SMEM,            // Scalar memory operations
  DS,              // LDS/GDS operations
  SALU,            // Scalar ALU
  DMA,             // Tensor DMA operations
  Fence,           // Fences and waits
  Other,           // Everything else
  NUM_FLAVORS
};

constexpr StringRef getFlavorName(InstructionFlavor F) {
  switch (F) {
  case InstructionFlavor::WMMA:
    return "WMMA";
  case InstructionFlavor::SingleCycleVALU:
    return "VALU(1c)";
  case InstructionFlavor::TRANS:
    return "TRANS";
  case InstructionFlavor::MultiCycleVALU:
    return "VALU(Nc)";
  case InstructionFlavor::VMEM:
    return "VMEM";
  case InstructionFlavor::SMEM:
    return "SMEM";
  case InstructionFlavor::DS:
    return "DS";
  case InstructionFlavor::SALU:
    return "SALU";
  case InstructionFlavor::DMA:
    return "DMA";
  case InstructionFlavor::Fence:
    return "Fence";
  case InstructionFlavor::Other:
    return "Other";
  case InstructionFlavor::NUM_FLAVORS:
    return "???";
  }
  llvm_unreachable("Unknown InstructionFlavor");
}

/// Classify \p MI into the execution flavor that drives both the scheduler's
/// slot preferences and the hazard recognizer's co-execution masks.
InstructionFlavor classifyFlavor(const MachineInstr &MI,
                                 const SIInstrInfo &SII);

/// Map a flavor to the co-execution class it occupies in a window slot.
constexpr CoExecMaskT getCoExecMask(InstructionFlavor F) {
  switch (F) {
  case InstructionFlavor::WMMA:
    return CoExecMask::WMMA;
  case InstructionFlavor::TRANS:
    return CoExecMask::TRANS;
  case InstructionFlavor::SingleCycleVALU:
  case InstructionFlavor::MultiCycleVALU:
  // LDS DMA and tensor DMA issue on the VALU pipe.
  case InstructionFlavor::DMA:
    return CoExecMask::VALU;
  case InstructionFlavor::DS:
    return CoExecMask::DS;
  case InstructionFlavor::VMEM:
    return CoExecMask::VMEM;
  case InstructionFlavor::SMEM:
    return CoExecMask::SMEM;
  case InstructionFlavor::SALU:
  // Fences are s_barrier_*/s_wait_*, which issue on the scalar pipe.
  case InstructionFlavor::Fence:
    return CoExecMask::SALU;
  case InstructionFlavor::Other:
    return CoExecMask::CTRL;
  case InstructionFlavor::NUM_FLAVORS:
    break;
  }
  llvm_unreachable("Unknown InstructionFlavor");
}

//===----------------------------------------------------------------------===//
// Co-execution Stage Type
//===----------------------------------------------------------------------===//

/// Stage type for co-execution (for annotation/display).
enum class CoExecStageType : uint8_t {
  NONE = 0, // Not in co-exec window
  E0,       // Issue cycle - control only
  E,        // External - MEM/SALU allowed
  I,        // Internal - MEM/SALU/VALU allowed
  IS,       // Internal + scaled-WMMA absorb (I plus next-WMMA issue)
  V,        // Vacant - MEM/SALU/WMMA allowed, no VALU
  TR        // TRANS co-exec - everything except TRANS
};

inline const char *getStageTypeName(CoExecStageType T) {
  switch (T) {
  case CoExecStageType::NONE:
    return "--";
  case CoExecStageType::E0:
    return "E0";
  case CoExecStageType::E:
    return "E";
  case CoExecStageType::I:
    return "I";
  case CoExecStageType::IS:
    return "IS";
  case CoExecStageType::V:
    return "V";
  case CoExecStageType::TR:
    return "TR";
  }
  llvm_unreachable("Unknown CoExecStageType");
}

/// Return a human-readable name for a mask holding a single instruction class,
/// as produced by getCoExecMask().
inline const char *getCoExecMaskName(CoExecMaskT Mask) {
  switch (Mask) {
  case CoExecMask::CTRL:
    return "CTRL";
  case CoExecMask::VALU:
    return "VALU";
  case CoExecMask::TRANS:
    return "TRANS";
  case CoExecMask::SALU:
    return "SALU";
  case CoExecMask::DS:
    return "DS";
  case CoExecMask::VMEM:
    return "VMEM";
  case CoExecMask::SMEM:
    return "SMEM";
  case CoExecMask::WMMA:
    return "WMMA";
  default:
    llvm_unreachable("Not a single instruction class");
  }
}

/// Max stages: INT8 16x16x64 = 17 cycles, round up for safety.
constexpr unsigned MaxCoExecStages = 32;

//===----------------------------------------------------------------------===//
// Co-execution Slot Info
//===----------------------------------------------------------------------===//

/// Per-slot info: which instruction classes may co-execute here.
struct CoExecSlotInfo {
  CoExecMaskT Mask = CoExecMask::All; // What CAN execute (correctness)
};

//===----------------------------------------------------------------------===//
// Co-execution Info
//===----------------------------------------------------------------------===//

/// Co-execution characteristics for a multi-cycle instruction.
struct CoExecInfo {
  /// Number of cycles in the co-execution window, counting any trailing
  /// vacant stages.
  unsigned TotalWindow = 0;
  /// Per-stage slot info (capability mask).
  CoExecSlotInfo Slots[MaxCoExecStages];
  /// Pattern string for display (e.g., "0EIIEEIIV").
  StringRef Pattern;

  /// Default constructor - initialize to safe defaults.
  CoExecInfo() {
    for (unsigned I = 0; I < MaxCoExecStages; ++I)
      Slots[I].Mask = CoExecMask::All; // Default: permissive
  }

  /// Get capability mask for a stage.
  CoExecMaskT getMask(unsigned Stage) const {
    return Stage < TotalWindow ? Slots[Stage].Mask : CoExecMask::All;
  }

  /// Check if an instruction class mask can co-execute at a given stage.
  bool canCoExec(CoExecMaskT InstMask, unsigned Stage) const {
    if (Stage >= TotalWindow)
      return true;
    return any(Slots[Stage].Mask & InstMask);
  }

  /// Find next stage where the instruction class is allowed.
  std::optional<unsigned> findNextAllowedStage(CoExecMaskT InstMask,
                                               unsigned FromStage) const {
    for (unsigned I = FromStage; I < TotalWindow; ++I) {
      if (any(Slots[I].Mask & InstMask))
        return I;
    }
    return std::nullopt;
  }

  /// Get stage type from mask for display.
  static CoExecStageType getStageType(CoExecMaskT Mask) {
    if (Mask == CoExecMask::StageE0)
      return CoExecStageType::E0;
    if (Mask == CoExecMask::StageE)
      return CoExecStageType::E;
    if (Mask == CoExecMask::StageIS)
      return CoExecStageType::IS;
    if (Mask == CoExecMask::StageI)
      return CoExecStageType::I;
    if (Mask == CoExecMask::StageV)
      return CoExecStageType::V;
    if (Mask == CoExecMask::StageTR)
      return CoExecStageType::TR;
    // For 'All' or unknown, return based on what's allowed.
    if (any(Mask & CoExecMask::VALU))
      return CoExecStageType::I; // If VALU allowed, it's I-like
    if (any(Mask & CoExecMask::WMMA))
      return CoExecStageType::V; // If WMMA allowed (not VALU), V-like
    return CoExecStageType::E;   // Default to E
  }

  /// Get stage type for a specific stage.
  CoExecStageType getType(unsigned Stage) const {
    return getStageType(getMask(Stage));
  }

  /// Build a CoExecInfo from a pattern string.
  static CoExecInfo build(unsigned TotalWindow, const char *Pattern);
};

//===----------------------------------------------------------------------===//
// Co-execution Info Construction
//===----------------------------------------------------------------------===//

/// Build CoExecInfo from a pattern string.
/// Pattern chars: '0'=E0, 'E'=External, 'I'=Internal, 'V'=Vacant,
///                'S'=Internal+ScaleWMMAAbsorb (I plus next scaled WMMA),
///                'T'=TRANS co-exec (all except TRANS), 'A'=Any
inline CoExecInfo CoExecInfo::build(unsigned TotalWindow, const char *Pattern) {
  CoExecInfo Info;
  Info.TotalWindow = TotalWindow;
  Info.Pattern = Pattern;
  assert(Info.Pattern.size() == TotalWindow &&
         "Pattern must describe every cycle of the co-execution window");
  assert(TotalWindow <= MaxCoExecStages && "Co-execution window is too long");

  for (unsigned I = 0; I < Info.TotalWindow; ++I) {
    switch (Pattern[I]) {
    case '0':
      Info.Slots[I].Mask = CoExecMask::StageE0;
      break;
    case 'E':
      Info.Slots[I].Mask = CoExecMask::StageE;
      break;
    case 'I':
      Info.Slots[I].Mask = CoExecMask::StageI;
      break;
    case 'S':
      Info.Slots[I].Mask = CoExecMask::StageIS;
      break;
    case 'V':
      Info.Slots[I].Mask = CoExecMask::StageV;
      break;
    case 'T':
      Info.Slots[I].Mask = CoExecMask::StageTR;
      break;
    case 'A':
    default:
      Info.Slots[I].Mask = CoExecMask::All;
      break;
    }
  }
  return Info;
}

/// Get co-execution info for a gfx950 MFMA instruction.
CoExecInfo getMFMACoExecInfo(unsigned Opcode);

/// Get co-execution info for a WMMA instruction, selecting the per-cycle slot
/// pattern from the opcode (and operand formats for the F8F6F4 variants).
inline CoExecInfo getCoExecInfo(const MachineInstr &MI,
                                const SIInstrInfo &TII) {
  unsigned Opc = MI.getOpcode();

  if (TII.isMFMA(Opc))
    return getMFMACoExecInfo(Opc);

  // Scaled variants (LD_SCALE rule) absorb the next WMMA in the last I slot.
  bool HasScaling = AMDGPU::getHasMatrixScale(Opc);

  // The F8F6F4 family is the only WMMA carrying matrix format operands, and its
  // window depends on them: both inputs f4 issue in 4 cycles, anything wider in
  // 8. This matches the PredIsNotBothF4_WMMA_SCALE latency variant.
  if (const MachineOperand *FmtA =
          TII.getNamedOperand(MI, AMDGPU::OpName::matrix_a_fmt)) {
    const MachineOperand *FmtB =
        TII.getNamedOperand(MI, AMDGPU::OpName::matrix_b_fmt);
    bool BothF4 = FmtB && FmtA->getImm() == AMDGPU::WMMA::MATRIX_FMT_FP4 &&
                  FmtB->getImm() == AMDGPU::WMMA::MATRIX_FMT_FP4;
    if (BothF4)
      return CoExecInfo::build(6, HasScaling ? "0EESVV" : "0EEIVV");
    return CoExecInfo::build(10, HasScaling ? "0EEIEEISVV" : "0EEIEEIIVV");
  }

  switch (Opc) {
  // 16x16x64 IU8: 16-cycle occupancy, 17-cycle window.
  case AMDGPU::V_WMMA_I32_16X16X64_IU8_w32_threeaddr:
  case AMDGPU::V_WMMA_I32_16X16X64_IU8_w32_twoaddr:
    return CoExecInfo::build(17, "0EIIEEIIEEIIEEIIV");

  // 16x16x64 FP8/BF8: 4-cycle occupancy, 6-cycle window.
  case AMDGPU::V_WMMA_F16_16X16X64_BF8_BF8_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X64_BF8_BF8_w32_twoaddr:
  case AMDGPU::V_WMMA_F16_16X16X64_BF8_FP8_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X64_BF8_FP8_w32_twoaddr:
  case AMDGPU::V_WMMA_F16_16X16X64_FP8_BF8_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X64_FP8_BF8_w32_twoaddr:
  case AMDGPU::V_WMMA_F16_16X16X64_FP8_FP8_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X64_FP8_FP8_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X64_BF8_BF8_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X64_BF8_BF8_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X64_BF8_FP8_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X64_BF8_FP8_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X64_FP8_BF8_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X64_FP8_BF8_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X64_FP8_FP8_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr:
    return CoExecInfo::build(6, "0EEIVV");

  // 16x16x32 F16/BF16: 8-cycle occupancy, 9-cycle window.
  case AMDGPU::V_SWMMAC_BF16_16X16X32_BF16_w32_twoaddr:
  case AMDGPU::V_SWMMAC_BF16_16X16X32_BF16_w64_twoaddr:
  case AMDGPU::V_SWMMAC_F16_16X16X32_F16_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F16_16X16X32_F16_w64_twoaddr:
  case AMDGPU::V_SWMMAC_F32_16X16X32_BF16_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F32_16X16X32_BF16_w64_twoaddr:
  case AMDGPU::V_SWMMAC_F32_16X16X32_F16_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F32_16X16X32_F16_w64_twoaddr:
  case AMDGPU::V_WMMA_BF16F32_16X16X32_BF16_w32_threeaddr:
  case AMDGPU::V_WMMA_BF16F32_16X16X32_BF16_w32_twoaddr:
  case AMDGPU::V_WMMA_BF16_16X16X32_BF16_w32_threeaddr:
  case AMDGPU::V_WMMA_BF16_16X16X32_BF16_w32_twoaddr:
  case AMDGPU::V_WMMA_F16_16X16X32_F16_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X32_F16_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X32_BF16_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X32_BF16_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X32_F16_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X32_F16_w32_twoaddr:
    return CoExecInfo::build(9, "0EIIEEIIV");

  // 16x16x128 FP8/BF8: 8-cycle occupancy, 10-cycle window.
  case AMDGPU::V_SWMMAC_F16_16X16X128_BF8_BF8_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F16_16X16X128_BF8_FP8_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F16_16X16X128_FP8_BF8_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F16_16X16X128_FP8_FP8_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F32_16X16X128_BF8_BF8_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F32_16X16X128_BF8_FP8_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F32_16X16X128_FP8_BF8_w32_twoaddr:
  case AMDGPU::V_SWMMAC_F32_16X16X128_FP8_FP8_w32_twoaddr:
  case AMDGPU::V_WMMA_F16_16X16X128_BF8_BF8_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X128_BF8_BF8_w32_twoaddr:
  case AMDGPU::V_WMMA_F16_16X16X128_BF8_FP8_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X128_BF8_FP8_w32_twoaddr:
  case AMDGPU::V_WMMA_F16_16X16X128_FP8_BF8_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X128_FP8_BF8_w32_twoaddr:
  case AMDGPU::V_WMMA_F16_16X16X128_FP8_FP8_w32_threeaddr:
  case AMDGPU::V_WMMA_F16_16X16X128_FP8_FP8_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X128_BF8_BF8_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X128_BF8_BF8_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X128_BF8_FP8_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X128_BF8_FP8_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X128_FP8_BF8_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X128_FP8_BF8_w32_twoaddr:
  case AMDGPU::V_WMMA_F32_16X16X128_FP8_FP8_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_16X16X128_FP8_FP8_w32_twoaddr:
    return CoExecInfo::build(10, "0EEIEEIIVV");

  // 32x16x128 F4: 8-cycle occupancy, 10-cycle window.
  case AMDGPU::V_WMMA_F32_32X16X128_F4_w32_threeaddr:
  case AMDGPU::V_WMMA_F32_32X16X128_F4_w32_twoaddr:
  case AMDGPU::V_WMMA_SCALE16_F32_32X16X128_F4_w32_threeaddr:
  case AMDGPU::V_WMMA_SCALE16_F32_32X16X128_F4_w32_twoaddr:
  case AMDGPU::V_WMMA_SCALE_F32_32X16X128_F4_w32_threeaddr:
  case AMDGPU::V_WMMA_SCALE_F32_32X16X128_F4_w32_twoaddr:
    return CoExecInfo::build(10, HasScaling ? "0EEIEIESVV" : "0EEIEIEIVV");

  default:
    // Permissive window for variants without a modeled slot pattern.
    return CoExecInfo::build(9, "AAAAAAAAA");
  }
}

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECINFO_H
