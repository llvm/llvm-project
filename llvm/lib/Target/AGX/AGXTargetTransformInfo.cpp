//===-- AGXTargetTransformInfo.cpp - AGX specific TTI ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXTargetTransformInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;

#define DEBUG_TYPE "agxtti"

static bool isThreadVaryingIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::agx_thread_position_in_grid_x:
  case Intrinsic::agx_thread_position_in_grid_y:
  case Intrinsic::agx_thread_position_in_grid_z:
  case Intrinsic::agx_thread_position_in_threadgroup_x:
  case Intrinsic::agx_thread_position_in_threadgroup_y:
  case Intrinsic::agx_thread_position_in_threadgroup_z:
  case Intrinsic::agx_thread_index_in_threadgroup:
  case Intrinsic::agx_thread_index_in_simdgroup:
  case Intrinsic::agx_simdgroup_index_in_threadgroup:
  case Intrinsic::agx_active_thread_index_in_simdgroup:
  case Intrinsic::agx_active_thread_index_in_quadgroup:
  case Intrinsic::agx_read_sr:
  case Intrinsic::agx_simd_shuffle:
  case Intrinsic::agx_simd_shuffle_xor:
  case Intrinsic::agx_simd_shuffle_up:
  case Intrinsic::agx_simd_shuffle_down:
  case Intrinsic::agx_quad_shuffle:
  case Intrinsic::agx_quad_shuffle_xor:
  case Intrinsic::agx_quad_shuffle_up:
  case Intrinsic::agx_quad_shuffle_down:
  case Intrinsic::agx_simd_prefix:
  case Intrinsic::agx_quad_prefix:
  case Intrinsic::agx_quad_reduce:
  case Intrinsic::agx_quad_ballot:
  case Intrinsic::agx_dfdx:
  case Intrinsic::agx_dfdy:
  case Intrinsic::agx_simdgroup_matrix_fma:
    return true;
  default:
    return false;
  }
}

ValueUniformity AGXTTIImpl::getValueUniformity(const Value *V) const {
  if (const auto *II = dyn_cast<IntrinsicInst>(V)) {
    if (isThreadVaryingIntrinsic(II->getIntrinsicID()))
      return ValueUniformity::NeverUniform;
    switch (II->getIntrinsicID()) {
    case Intrinsic::agx_simd_ballot:
    case Intrinsic::agx_simd_reduce:
    case Intrinsic::agx_total_active_threads_in_simdgroup:
    case Intrinsic::agx_threadgroup_position_in_grid_x:
    case Intrinsic::agx_threadgroup_position_in_grid_y:
    case Intrinsic::agx_threadgroup_position_in_grid_z:
    case Intrinsic::agx_threads_per_threadgroup_x:
    case Intrinsic::agx_threads_per_threadgroup_y:
    case Intrinsic::agx_threads_per_threadgroup_z:
    case Intrinsic::agx_threadgroups_per_grid_x:
    case Intrinsic::agx_threadgroups_per_grid_y:
    case Intrinsic::agx_threadgroups_per_grid_z:
      return ValueUniformity::AlwaysUniform;
    default:
      return ValueUniformity::Default;
    }
  }

  // Stack memory is per thread, and atomics return per thread values.
  if (const auto *LI = dyn_cast<LoadInst>(V))
    return LI->getPointerAddressSpace() == AGX::PRIVATE_ADDRESS
               ? ValueUniformity::NeverUniform
               : ValueUniformity::Default;
  if (isa<AtomicRMWInst>(V) || isa<AtomicCmpXchgInst>(V) || isa<CallInst>(V))
    return ValueUniformity::NeverUniform;
  return ValueUniformity::Default;
}

// A shuffle of a constant reads the same value in every lane. (Reading an
// inactive lane is undefined, so the constant is a valid result there too.)
std::optional<Instruction *>
AGXTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  switch (II.getIntrinsicID()) {
  case Intrinsic::agx_simd_shuffle:
  case Intrinsic::agx_simd_shuffle_xor:
  case Intrinsic::agx_simd_shuffle_up:
  case Intrinsic::agx_simd_shuffle_down:
  case Intrinsic::agx_quad_shuffle:
  case Intrinsic::agx_quad_shuffle_xor:
  case Intrinsic::agx_quad_shuffle_up:
  case Intrinsic::agx_quad_shuffle_down:
    if (isa<Constant>(II.getArgOperand(0)))
      return IC.replaceInstUsesWith(II, II.getArgOperand(0));
    break;
  default:
    break;
  }
  return std::nullopt;
}
