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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace llvm {

namespace AMDGPU {

//===----------------------------------------------------------------------===//
// Co-execution Bitmasks
//===----------------------------------------------------------------------===//

using CoExecMaskT = uint16_t;

/// Bitmask for instruction types allowed to co-execute at a stage.
namespace CoExecMask {
constexpr CoExecMaskT None = 0;
constexpr CoExecMaskT CTRL = 1 << 0;  // Control: s_delay_alu, s_set_vgpr_msb
constexpr CoExecMaskT VALU = 1 << 1;  // Vector ALU
constexpr CoExecMaskT TRANS = 1 << 2; // Transcendentals (V_EXP etc)
constexpr CoExecMaskT SALU = 1 << 3;  // Scalar ALU
constexpr CoExecMaskT DS = 1 << 4;    // LDS read/write
constexpr CoExecMaskT VMEM = 1 << 5;  // Global memory
constexpr CoExecMaskT SMEM = 1 << 6;  // Scalar memory
constexpr CoExecMaskT WMMA = 1 << 7;  // Next WMMA (V stages only)
constexpr CoExecMaskT All = 0xFFFF;

constexpr CoExecMaskT MEM = DS | VMEM | SMEM;
constexpr CoExecMaskT StageE0 = CTRL;             // Issue: control only
constexpr CoExecMaskT StageE = CTRL | SALU | MEM; // External: mem/salu
constexpr CoExecMaskT StageI =
    CTRL | SALU | MEM | VALU | TRANS; // Internal: all ALU
// Internal + scaled-WMMA absorb: same as StageI but the next scaled
// WMMA may issue here - its LD_SCALE consumes the I cycle and the matrix
// multiply lands in the V slot that follows. Used for the last I before
// V of scaled patterns.
constexpr CoExecMaskT StageIS = StageI | WMMA;
constexpr CoExecMaskT StageV =
    CTRL | SALU | MEM | WMMA;                 // Vacant: no valu/trans
constexpr CoExecMaskT StageTR = All & ~TRANS; // TRANS co-exec: no TRANS
} // namespace CoExecMask

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
  DS,              // LDS/GDS operations
  SALU,            // Scalar ALU
  DMA,             // Tensor DMA operations
  Fence,           // Fences and waits
  Other,           // Everything else
  NUM_FLAVORS
};

inline StringRef getFlavorName(InstructionFlavor F) {
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

inline StringRef getFlavorShortName(InstructionFlavor F) {
  switch (F) {
  case InstructionFlavor::WMMA:
    return "W";
  case InstructionFlavor::SingleCycleVALU:
    return "V";
  case InstructionFlavor::TRANS:
    return "T";
  case InstructionFlavor::MultiCycleVALU:
    return "C";
  case InstructionFlavor::VMEM:
    return "M";
  case InstructionFlavor::DS:
    return "D";
  case InstructionFlavor::SALU:
    return "S";
  case InstructionFlavor::DMA:
    return "X";
  case InstructionFlavor::Fence:
    return "F";
  case InstructionFlavor::Other:
    return "O";
  case InstructionFlavor::NUM_FLAVORS:
    return "?";
  }
  llvm_unreachable("Unknown InstructionFlavor");
}

/// Vector-based flavor grouping for dynamic iteration.
using FlavorGroup = SmallVector<InstructionFlavor, 4>;

namespace FlavorGroups {
inline FlavorGroup allVALU() {
  return {InstructionFlavor::SingleCycleVALU, InstructionFlavor::TRANS,
          InstructionFlavor::MultiCycleVALU};
}
inline FlavorGroup allMem() {
  return {InstructionFlavor::VMEM, InstructionFlavor::DS,
          InstructionFlavor::DMA};
}
inline FlavorGroup individual(InstructionFlavor F) { return {F}; }
inline FlavorGroup all() {
  FlavorGroup G;
  for (unsigned I = 0;
       I < static_cast<unsigned>(InstructionFlavor::NUM_FLAVORS); ++I)
    G.push_back(static_cast<InstructionFlavor>(I));
  return G;
}
} // namespace FlavorGroups

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

/// Return a human-readable name for a CoExecMask bitmask value.
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
    return "???";
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
  /// Total co-execution window size including tail.
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
    return Stage < MaxCoExecStages ? Slots[Stage].Mask : CoExecMask::All;
  }

  /// Check if an instruction class mask can co-execute at a given stage.
  bool canCoExec(CoExecMaskT InstMask, unsigned Stage) const {
    if (Stage >= TotalWindow)
      return true;
    return (Slots[Stage].Mask & InstMask) != 0;
  }

  /// Find next stage where the instruction class is allowed.
  std::optional<unsigned> findNextAllowedStage(CoExecMaskT InstMask,
                                               unsigned FromStage) const {
    for (unsigned I = FromStage; I < TotalWindow; ++I) {
      if ((Slots[I].Mask & InstMask) != 0)
        return I;
    }
    return std::nullopt;
  }

  /// Get stage type from mask for display.
  static CoExecStageType getStageType(CoExecMaskT Mask) {
    using namespace CoExecMask;
    if (Mask == StageE0)
      return CoExecStageType::E0;
    if (Mask == StageE)
      return CoExecStageType::E;
    if (Mask == StageIS)
      return CoExecStageType::IS;
    if (Mask == StageI)
      return CoExecStageType::I;
    if (Mask == StageV)
      return CoExecStageType::V;
    if (Mask == StageTR)
      return CoExecStageType::TR;
    // For 'All' or unknown, return based on what's allowed.
    if (Mask & VALU)
      return CoExecStageType::I; // If VALU allowed, it's I-like
    if (Mask & CoExecMask::WMMA)
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

  for (unsigned I = 0; I < Info.TotalWindow && Pattern[I]; ++I) {
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

/// Get co-execution info for a WMMA instruction, selecting the per-cycle slot
/// pattern from the opcode (and operand formats for the F8F6F4 variants).
inline CoExecInfo getCoExecInfo(const MachineInstr &MI,
                                const SIInstrInfo &TII) {
  StringRef Name = TII.getName(MI.getOpcode());

  // Scaled variants (LD_SCALE rule) absorb the next WMMA in the last I slot.
  bool HasScaling = Name.contains_insensitive("scale");

  if (Name.contains_insensitive("16x16x64_iu8"))
    return CoExecInfo::build(17, "0EIIEEIIEEIIEEIIV");

  // F8F6F4 16x16x128: window size depends on operand formats.
  if (Name.contains_insensitive("16x16x128_f8f6f4")) {
    bool BothF4 = false;
    if (const MachineOperand *FmtA =
            TII.getNamedOperand(MI, AMDGPU::OpName::matrix_a_fmt)) {
      if (const MachineOperand *FmtB =
              TII.getNamedOperand(MI, AMDGPU::OpName::matrix_b_fmt)) {
        BothF4 = (FmtA->getImm() == AMDGPU::WMMA::MATRIX_FMT_FP4 &&
                  FmtB->getImm() == AMDGPU::WMMA::MATRIX_FMT_FP4);
      }
    }
    if (BothF4)
      return CoExecInfo::build(6, HasScaling ? "0EESVV" : "0EEIVV");
    return CoExecInfo::build(10, HasScaling ? "0EEIEEISVV" : "0EEIEEIIVV");
  }

  if (Name.contains_insensitive("16x16x64_fp8") ||
      Name.contains_insensitive("16x16x64_bf8"))
    return CoExecInfo::build(6, HasScaling ? "0EESVV" : "0EEIVV");

  if (Name.contains_insensitive("16x16x32_f16") ||
      Name.contains_insensitive("16x16x32_bf16"))
    return CoExecInfo::build(9, HasScaling ? "0EIIEEISV" : "0EIIEEIIV");

  if (Name.contains_insensitive("16x16x128_fp8") ||
      Name.contains_insensitive("16x16x128_bf8"))
    return CoExecInfo::build(10, HasScaling ? "0EEIEEISVV" : "0EEIEEIIVV");

  if (Name.contains_insensitive("32x16x128_f4"))
    return CoExecInfo::build(10, HasScaling ? "0EEIEIESVV" : "0EEIEIEIVV");

  // Default fallback: permissive window.
  return CoExecInfo::build(9, "AAAAAAAAA");
}

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCOEXECINFO_H
