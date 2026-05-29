//===- MetalConstraints.h - Metal type constraints --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_METALCONSTRAINTS_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_METALCONSTRAINTS_H

/// Metal GPU constraints by address space and context.
///
/// Centralizes the rules currently sprinkled across 15+ passes:
/// - Which types are valid per address space
/// - When barriers/volatile/bitcasts are needed
/// - MMA-specific overrides
///
/// Passes query this instead of reimplementing the same checks.

#include "../MetalAddressSpaces.h"
#include "llvm/IR/Type.h"

namespace llvm {
namespace metal {

// ── Metal type constraints ────────────────────────────────────────────────

/// Metal has no double, no i1 in memory, no float8.
/// When MMA intrinsics are present, device pointers must all be float*.
struct MetalConstraints {
  /// Whether the module uses tile/matrix-multiply intrinsics.
  bool HasMMA = false;

  /// Can this type be stored/loaded in the given address space?
  bool isValidMemoryType(llvm::Type *T) const {
    if (T->isDoubleTy())
      return false; // no f64
    if (T->isIntegerTy(1))
      return false; // i1 → must use i8
    return true;
  }

  /// What pointee type should a pointer in this AS have?
  /// Returns nullptr if no override needed.
  llvm::Type *requiredPointeeType(unsigned AddrSpace,
                                  llvm::LLVMContext &Ctx) const {
    if (HasMMA && AddrSpace == AS::Device)
      return llvm::Type::getFloatTy(Ctx); // GPU JIT crashes on half*/i32*
    return nullptr;
  }

  /// Should i1 GEPs be rewritten to i8?
  static constexpr bool RemapI1ToI8 = true;

  /// Should bfloat GEPs be lowered to half? (same bit width)
  static constexpr bool RemapBFloatToHalf = true;

  /// Maximum threadgroup memory in bytes.
  /// Per Metal Shading Language Specification v3.2, threadgroup memory is
  /// capped at 32 KiB on Apple GPUs (M-series; verify per-target).
  static constexpr unsigned MaxTGBytes = 32 * 1024;

  /// Maximum pointer PHIs per basic block before converting to i64.
  static constexpr unsigned MaxPtrPhisPerBlock = 32;

  /// Does this address space require explicit barriers between
  /// store and subsequent load across threads?
  static bool needsBarriers(unsigned AddrSpace) {
    return AddrSpace == AS::Threadgroup;
  }

  /// Should device loads be marked volatile to prevent GPU JIT hoisting
  /// when a store to the same pointer exists in the loop?
  bool needsVolatileDeviceLoads() const {
    return true; // always - GPU JIT hoists non-volatile loads
  }

  /// When MMA present, should all device loads be widened to float?
  bool widenDeviceLoadsToFloat() const { return HasMMA; }
};

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_METALCONSTRAINTS_H
