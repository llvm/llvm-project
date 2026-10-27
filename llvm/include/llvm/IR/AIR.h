//===- llvm/IR/AIR.h - Facts about AIR shared by its consumers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// AIR's address spaces and the names in its kernel description
/// (`!air.kernel`), for the targets that produce AIR (AIR) and consume it
/// (AGX), so that neither has to link the other.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_AIR_H
#define LLVM_IR_AIR_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace air {

/// AIR address spaces.
namespace AS {
constexpr unsigned Default = 0; // thread (private)
constexpr unsigned Device = 1;
constexpr unsigned Constant = 2;
constexpr unsigned Threadgroup = 3;
} // namespace AS

/// The named metadata listing the kernels: one node per kernel,
/// `!{ptr @kernel, !{outputs}, !{parameters}}`. Each parameter node starts with
/// the parameter's number and its role (a buffer or a system value), followed
/// by key-value pairs.
constexpr StringLiteral KernelMD = "air.kernel";

/// Parameter roles and keys.
constexpr StringLiteral Buffer = "air.buffer";
constexpr StringLiteral IndirectBuffer = "air.indirect_buffer";
constexpr StringLiteral LocationIndex = "air.location_index";
constexpr StringLiteral Read = "air.read";
constexpr StringLiteral ReadWrite = "air.read_write";
constexpr StringLiteral AddressSpace = "air.address_space";
constexpr StringLiteral ArgTypeSize = "air.arg_type_size";
constexpr StringLiteral ArgTypeAlignSize = "air.arg_type_align_size";
constexpr StringLiteral ArgTypeName = "air.arg_type_name";
constexpr StringLiteral ArgName = "air.arg_name";

/// Compute system values (the role of a system-value parameter).
constexpr StringLiteral ThreadPositionInGrid = "air.thread_position_in_grid";
constexpr StringLiteral ThreadgroupPositionInGrid =
    "air.threadgroup_position_in_grid";
constexpr StringLiteral ThreadPositionInThreadgroup =
    "air.thread_position_in_threadgroup";
constexpr StringLiteral ThreadsPerThreadgroup = "air.threads_per_threadgroup";
constexpr StringLiteral ThreadgroupsPerGrid = "air.threadgroups_per_grid";
constexpr StringLiteral ThreadsPerGrid = "air.threads_per_grid";
constexpr StringLiteral ThreadIndexInThreadgroup =
    "air.thread_index_in_threadgroup";
constexpr StringLiteral ThreadIndexInSimdgroup = "air.thread_index_in_simdgroup";
constexpr StringLiteral SimdgroupIndexInThreadgroup =
    "air.simdgroup_index_in_threadgroup";
constexpr StringLiteral ThreadsPerSimdgroup = "air.threads_per_simdgroup";
constexpr StringLiteral SimdgroupsPerThreadgroup =
    "air.simdgroups_per_threadgroup";

/// A hint on a relaxed AIR atomic that AIR's portable lowering placed between
/// two seq_cst `air.atomic.fence` calls (AIR 2.8 atomics are relaxed): the
/// MSL memory_order(s) of the LLVM atomic it came from, `!{i32 order}`, or
/// `!{i32 success, i32 failure}` for a compare-exchange. A consumer with
/// ordered atomics (AGX) may replace the fence, atomic, fence with one atomic
/// of that order. Metal-bound output does not carry it.
constexpr StringLiteral AtomicOrderingMD = "agx.atomic.ordering";

/// Hints: operations that AIR does not have but a consumer may (AGX), in a
/// form every AIR consumer can still compile (llvm/docs/AIR/AIRHints.rst).
///
/// A function hint is a call to an internal, noinline function whose body
/// computes the operation in AIR (its emulation) and that carries the
/// attribute HintAttr naming the operation. A consumer that has the
/// operation replaces the call with it; any other inlines the body.
constexpr StringLiteral HintAttr = "air-hint";
/// The prefix of the names of hint functions.
constexpr StringLiteral HintPrefix = "__air_hint.";
/// An address-space hint: pointers that may point to any memory (CUDA's and
/// AGX's generic address space). A consumer without generic pointers treats
/// them as device pointers.
constexpr unsigned GenericHintAS = 4;

} // namespace air
} // namespace llvm

#endif // LLVM_IR_AIR_H
