//===- AIR.h - Top-level interface for AIR backend ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the AIR
// Apple Intermediate Representation target library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIR_H
#define LLVM_LIB_TARGET_AIR_AIR_H

namespace llvm {
class ModulePass;
class FunctionPass;
class PassRegistry;

/// Initializer for the AIR inline-non-kernel pass.
void initializeAIRInlineNonKernelLegacyPass(PassRegistry &);

/// Pass to inline all calls to defined functions (AIR has no call stack).
ModulePass *createAIRInlineNonKernelLegacyPass();

/// Initializer for the AIR f64-to-f32 demotion pass.
void initializeAIRDemoteF64LegacyPass(PassRegistry &);

/// Pass to rewrite all `double` operations to `float` (AIR has no f64).
ModulePass *createAIRDemoteF64LegacyPass();

/// Initializer for the AIR lower-fneg pass.
void initializeAIRLowerFNegLegacyPass(PassRegistry &);

/// Pass to rewrite `fneg x` as `fsub -0.0, x`.
ModulePass *createAIRLowerFNegLegacyPass();

/// Initializer for the AIR NaN-safe min/max pass.
void initializeAIRNaNMinMaxLegacyPass(PassRegistry &);

/// Pass to lower llvm.minimum/maximum to air.fmin/fmax with NaN-propagation
/// selects.
ModulePass *createAIRNaNMinMaxLegacyPass();

/// Initializer for the AIR LLVM-to-AIR intrinsic renaming pass.
void initializeLLVMToAIRIntrinsicsLegacyPass(PassRegistry &);

/// Pass to rename LLVM intrinsic declarations to their AIR equivalents.
ModulePass *createLLVMToAIRIntrinsicsLegacyPass();

/// Initializer for the AIR barrier-rename pass.
void initializeAIRBarrierRenameLegacyPass(PassRegistry &);

/// Pass to rename air.threadgroup.barrier to air.wg.barrier and fix args.
ModulePass *createAIRBarrierRenameLegacyPass();

/// Initializer for the AIR lower-atomicrmw pass.
void initializeAIRLowerAtomicRMWLegacyPass(PassRegistry &);

/// Pass to lower `atomicrmw` instructions to `air.atomic.*` intrinsic calls.
ModulePass *createAIRLowerAtomicRMWLegacyPass();

/// Initializer for the AIR i64 simd shuffle split pass.
void initializeAIRSplitI64ShuffleLegacyPass(PassRegistry &);

/// Pass to split i64 `air.simd_shuffle` into two i32 shuffles.
ModulePass *createAIRSplitI64ShuffleLegacyPass();

/// Initializer for the AIR device-loads-volatile pass.
void initializeAIRDeviceLoadsVolatileLegacyPass(PassRegistry &);

/// Pass to mark loop device loads as volatile (and all device loads/stores
/// in CAS-atomic functions) to defeat AIR GPU JIT reordering.
ModulePass *createAIRDeviceLoadsVolatileLegacyPass();

/// Initializer for the AIR scalar-store-guard pass.
void initializeAIRScalarStoreGuardLegacyPass(PassRegistry &);

/// Pass to guard scalar device stores with a `tid.x == 0` check.
ModulePass *createAIRScalarStoreGuardLegacyPass();

/// Initializer for the AIR threadgroup-global coalesce pass.
void initializeAIRTGGlobalCoalesceLegacyPass(PassRegistry &);

/// Pass to merge `__tg_cvt_*` into `__tg_dot_ab_*` threadgroup globals when
/// MMA intrinsics are present.
ModulePass *createAIRTGGlobalCoalesceLegacyPass();

/// Initializer for the AIR threadgroup-barrier-insertion pass.
void initializeAIRTGBarrierInsertLegacyPass(PassRegistry &);

/// Pass to insert `air.wg.barrier` calls around threadgroup memory accesses.
ModulePass *createAIRTGBarrierInsertLegacyPass();

/// Initializer for the AIR async-event-to-alloca pass.
void initializeAIRAsyncEventToAllocaLegacyPass(PassRegistry &);

/// Pass to convert the `@__tg_async_events` threadgroup global to a stack
/// alloca and insert no-op bitcasts before async-copy / wait-event calls.
ModulePass *createAIRAsyncEventToAllocaLegacyPass();

void initializeAIRAsyncCopyToCooperativeLegacyPass(PassRegistry &);

ModulePass *createAIRAsyncCopyToCooperativeLegacyPass();

/// Initializer for the AIR normalize-allocas pass.
void initializeAIRNormalizeAllocasLegacyPass(PassRegistry &);

/// Pass to hoist allocas to the entry block, normalize alloca sizes to i32,
/// strip `disjoint` flags, and insert typed-pointer bitcasts where needed.
ModulePass *createAIRNormalizeAllocasLegacyPass();

/// Initializer for the AIR bfloat16-cast-decompose pass.
void initializeAIRBFloat16CastDecomposeLegacyPass(PassRegistry &);

/// Pass to decompose sitofp/uitofp to bfloat (and sub-32-bit int-to-float)
/// into integer/bit operations for AIR v1 bitcode compatibility.
ModulePass *createAIRBFloat16CastDecomposeLegacyPass();

/// Initializer for the AIR system-values pass.
void initializeAIRSystemValuesLegacyPass(PassRegistry &);

/// Pass to convert AIR system-value intrinsic calls into kernel parameters
/// and emit `!air.kernel` / `!air.version` / `!air.language_version` metadata.
ModulePass *createAIRSystemValuesLegacyPass();

/// Initializer for the AIR scalar-buffer-packing pass.
void initializeAIRScalarBufferPackingLegacyPass(PassRegistry &);

/// Pass to pack scalar / `addrspace(2)` parameters into a single trailing
/// `addrspace(1)` device-buffer parameter (matching the Python driver).
ModulePass *createAIRScalarBufferPackingLegacyPass();

/// Initializer for the AIR alias-metadata annotate pass.
void initializeAIRAliasAnnotateLegacyPass(PassRegistry &);

/// Pass to emit Apple-style alias-scope metadata on memory ops and
/// `"air-buffer-no-alias"` parameter attributes on every kernel buffer arg.
ModulePass *createAIRAliasAnnotateLegacyPass();

/// Initializer for the AIR pre-serialization preparation pass.
void initializeAIRPrepareLegacyPass(PassRegistry &);

/// Pass to normalize i1 GEPs to i8, lower oversized / undef-bearing ptr phis
/// to i64, and insert typed-pointer transitions before atomic intrinsics.
ModulePass *createAIRPrepareLegacyPass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIR_H
