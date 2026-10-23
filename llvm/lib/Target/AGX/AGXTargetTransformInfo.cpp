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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;

#define DEBUG_TYPE "agxtti"

static cl::opt<unsigned> AGXThreadgroupUnroll(
    "agx-threadgroup-unroll", cl::init(1), cl::Hidden,
    cl::desc("Most iterations a loop that loads threadgroup memory is "
             "unrolled by, so that its loads can be merged (0 or 1: no "
             "unrolling)"));

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
  case Intrinsic::agx_simd_matrix_fmadd32:
  case Intrinsic::agx_simd_matrix_fmadd16:
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

// A threadgroup load costs about the same at any width (one to four 32-bit
// values: Geekbench 7's gemm_block_size_bias on the M2 Max takes 11-12 us per
// load of its inner loop whether it is x or xy, and the same with half the
// bytes). Unrolling loops that load threadgroup memory could let the
// load/store vectorizer merge the loads of neighbouring iterations into wider
// ones. Off by default: the shared addresses of PTX are integers turned into
// pointers, which the vectorizer does not see through, and the unrolled
// gemm_block_size_bias needs 121 instead of 98 registers and is 28% slower.
void AGXTTIImpl::getUnrollingPreferences(
    Loop *L, ScalarEvolution &SE, TargetTransformInfo::UnrollingPreferences &UP,
    OptimizationRemarkEmitter *ORE) const {
  BaseT::getUnrollingPreferences(L, SE, UP, ORE);
  bool ThreadgroupLoads = any_of(L->blocks(), [](const BasicBlock *BB) {
    return any_of(*BB, [](const Instruction &I) {
      auto *LI = dyn_cast<LoadInst>(&I);
      return LI && LI->getPointerAddressSpace() == AGX::THREADGROUP_ADDRESS;
    });
  });
  if (!ThreadgroupLoads || L->getNumBlocks() != 1 ||
      AGXThreadgroupUnroll < 2)
    return;
  UP.Partial = UP.Runtime = true;
  UP.MaxCount = AGXThreadgroupUnroll;
  // the unrolled body of such a loop (a tile of a matrix product) is a few
  // hundred instructions; the generic limit (150) would stop at its first copy
  UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
}
