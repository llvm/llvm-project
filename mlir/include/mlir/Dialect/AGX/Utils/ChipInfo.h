//===- ChipInfo.h - Capabilities of AGX GPUs --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The capabilities of each AGX GPU that the agx dialect's lowerings choose
// between. Lowerings test capabilities, never generations, so a new chip only
// needs a new row here (see mlir/docs/Dialects/AGX.md, "Capability table").
// A native capability is set only for chips where the hardware behaviour it
// relies on has been observed; elsewhere the portable lowering is used.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AGX_UTILS_CHIPINFO_H
#define MLIR_DIALECT_AGX_UTILS_CHIPINFO_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::agx {

/// How a chip multiplies 8x8 SIMD-group matrices.
enum class MatrixUnit {
  /// Shuffles and FMAs (the operands of the native form are not mapped).
  Emulated,
  /// A native 8x8x8 multiply-accumulate instruction.
  Native,
};

struct ChipInfo {
  /// GPU generation: 13 (M1), 14 (M2), 15 (M3), 16 (M4), 17 (A18 Pro, M5).
  unsigned generation = 0;
  /// Variant letter: 'g' (base), 's' (Pro), 'c' (Max), 'd' (Ultra), 'p'.
  char variant = 'g';

  unsigned simdWidth = 32;
  MatrixUnit matrixUnit = MatrixUnit::Emulated;
  /// The matrix unit computes A * B - C natively.
  bool matrixNegatedAccumulate = false;
  /// fmadd reads its first operand from a lower lane (G14 fmadd bit 53).
  bool laneShiftFMA = false;
  /// A barrier over an aligned block of 4 SIMD-groups (G14 0x78).
  bool blockBarrier = false;
  /// A preamble program runs once per core per dispatch.
  bool preamble = true;
  /// Threadgroup memory a workgroup can use, in bytes.
  unsigned threadgroupMemoryBytes = 32768;
  /// General purpose 32-bit registers before spilling.
  unsigned maxGPRs = 128;
  /// Registers and threadgroup memory are allocated dynamically (G15 and
  /// later), so occupancy hints are only advisory.
  bool dynamicCaching = false;
  /// strict mode (the "+agx-strict" target feature): code may use only the
  /// instructions Mesa's asahi compiler emits, so every native capability
  /// Mesa does not use is off and its agx operations take the portable form.
  bool mesaISA = false;

  /// The capabilities of the chip named `chip` (an LLVM AGX -mcpu such as
  /// "agx-g14c") with the LLVM target `features` (comma separated, as in
  /// #agxir.target), or failure for an unknown name or for "+agx-strict" on a
  /// chip Mesa does not compile for (only G13 and G14).
  static FailureOr<ChipInfo> get(StringRef chip, StringRef features = "");
};

} // namespace mlir::agx

#endif // MLIR_DIALECT_AGX_UTILS_CHIPINFO_H
