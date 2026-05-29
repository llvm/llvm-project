//===- Metal.h - Top-level interface for Metal backend ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the Metal
// (Apple AIR) target library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METAL_H
#define LLVM_LIB_TARGET_METAL_METAL_H

namespace llvm {
class ModulePass;
class FunctionPass;
class PassRegistry;

/// Initializer for the Metal inline-non-kernel pass.
void initializeMetalInlineNonKernelLegacyPass(PassRegistry &);

/// Pass to inline all calls to defined functions (AIR has no call stack).
ModulePass *createMetalInlineNonKernelLegacyPass();

/// Initializer for the Metal lower-fneg pass.
void initializeMetalLowerFNegLegacyPass(PassRegistry &);

/// Pass to rewrite `fneg x` as `fsub -0.0, x`.
ModulePass *createMetalLowerFNegLegacyPass();

/// Initializer for the Metal NaN-safe min/max pass.
void initializeMetalNaNMinMaxLegacyPass(PassRegistry &);

/// Pass to lower llvm.minimum/maximum to air.fmin/fmax with NaN-propagation
/// selects.
ModulePass *createMetalNaNMinMaxLegacyPass();

/// Initializer for the Metal LLVM-to-AIR intrinsic renaming pass.
void initializeMetalLLVMToAIRIntrinsicsLegacyPass(PassRegistry &);

/// Pass to rename LLVM intrinsic declarations to their AIR equivalents.
ModulePass *createMetalLLVMToAIRIntrinsicsLegacyPass();

/// Initializer for the Metal barrier-rename pass.
void initializeMetalBarrierRenameLegacyPass(PassRegistry &);

/// Pass to rename air.threadgroup.barrier to air.wg.barrier and fix args.
ModulePass *createMetalBarrierRenameLegacyPass();

/// Initializer for the Metal lower-atomicrmw pass.
void initializeMetalLowerAtomicRMWLegacyPass(PassRegistry &);

/// Pass to lower `atomicrmw` instructions to `air.atomic.*` intrinsic calls.
ModulePass *createMetalLowerAtomicRMWLegacyPass();

/// Initializer for the Metal i64 simd shuffle split pass.
void initializeMetalSplitI64ShuffleLegacyPass(PassRegistry &);

/// Pass to split i64 `air.simd_shuffle` into two i32 shuffles.
ModulePass *createMetalSplitI64ShuffleLegacyPass();

/// Initializer for the Metal device-loads-volatile pass.
void initializeMetalDeviceLoadsVolatileLegacyPass(PassRegistry &);

/// Pass to mark loop device loads as volatile (and all device loads/stores
/// in CAS-atomic functions) to defeat Metal GPU JIT reordering.
ModulePass *createMetalDeviceLoadsVolatileLegacyPass();

/// Initializer for the Metal scalar-store-guard pass.
void initializeMetalScalarStoreGuardLegacyPass(PassRegistry &);

/// Pass to guard scalar device stores with a `tid.x == 0` check.
ModulePass *createMetalScalarStoreGuardLegacyPass();

/// Initializer for the Metal threadgroup-global coalesce pass.
void initializeMetalTGGlobalCoalesceLegacyPass(PassRegistry &);

/// Pass to merge `__tg_cvt_*` into `__tg_dot_ab_*` threadgroup globals when
/// MMA intrinsics are present.
ModulePass *createMetalTGGlobalCoalesceLegacyPass();

/// Initializer for the Metal threadgroup-barrier-insertion pass.
void initializeMetalTGBarrierInsertLegacyPass(PassRegistry &);

/// Pass to insert `air.wg.barrier` calls around threadgroup memory accesses.
ModulePass *createMetalTGBarrierInsertLegacyPass();

/// Initializer for the Metal async-event-to-alloca pass.
void initializeMetalAsyncEventToAllocaLegacyPass(PassRegistry &);

/// Pass to convert the `@__tg_async_events` threadgroup global to a stack
/// alloca and insert no-op bitcasts before async-copy / wait-event calls.
ModulePass *createMetalAsyncEventToAllocaLegacyPass();

/// Initializer for the Metal normalize-allocas pass.
void initializeMetalNormalizeAllocasLegacyPass(PassRegistry &);

/// Pass to hoist allocas to the entry block, normalize alloca sizes to i32,
/// strip `disjoint` flags, and insert typed-pointer bitcasts where needed.
ModulePass *createMetalNormalizeAllocasLegacyPass();

/// Initializer for the Metal bfloat16-cast-decompose pass.
void initializeMetalBFloat16CastDecomposeLegacyPass(PassRegistry &);

/// Pass to decompose sitofp/uitofp to bfloat (and sub-32-bit int-to-float)
/// into integer/bit operations for Metal v1 bitcode compatibility.
ModulePass *createMetalBFloat16CastDecomposeLegacyPass();

/// Initializer for the Metal AIR system-values pass.
void initializeMetalAIRSystemValuesLegacyPass(PassRegistry &);

/// Pass to convert AIR system-value intrinsic calls into kernel parameters
/// and emit `!air.kernel` / `!air.version` / `!air.language_version` metadata.
ModulePass *createMetalAIRSystemValuesLegacyPass();

/// Initializer for the Metal scalar-buffer-packing pass.
void initializeMetalScalarBufferPackingLegacyPass(PassRegistry &);

/// Pass to pack scalar / `addrspace(2)` parameters into a single trailing
/// `addrspace(1)` device-buffer parameter (matching the Python driver).
ModulePass *createMetalScalarBufferPackingLegacyPass();

/// Initializer for the Metal AIR alias-metadata annotate pass.
void initializeMetalAliasAnnotateLegacyPass(PassRegistry &);

/// Pass to emit Apple-style alias-scope metadata on memory ops and
/// `"air-buffer-no-alias"` parameter attributes on every kernel buffer arg.
ModulePass *createMetalAliasAnnotateLegacyPass();

/// Initializer for the Metal pre-serialization preparation pass.
void initializeMetalPrepareLegacyPass(PassRegistry &);

/// Pass to normalize i1 GEPs to i8, lower oversized / undef-bearing ptr phis
/// to i64, and insert typed-pointer transitions before atomic intrinsics.
ModulePass *createMetalPrepareLegacyPass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METAL_H
