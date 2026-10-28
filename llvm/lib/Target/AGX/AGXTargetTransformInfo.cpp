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
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;

#define DEBUG_TYPE "agxtti"

static cl::opt<unsigned> MaxStoreVectorBytes(
    "agx-max-store-vector-bytes", cl::init(16), cl::Hidden,
    cl::desc("The widest store the load/store vectorizer makes (bytes)"));

unsigned AGXTTIImpl::maxStoreVectorBytes() { return MaxStoreVectorBytes; }

static cl::opt<unsigned> AGXThreadgroupUnroll(
    "agx-threadgroup-unroll", cl::init(0), cl::Hidden,
    cl::desc("Most iterations a loop that loads threadgroup memory is "
             "unrolled by, so that its loads can be merged (0: as far as "
             "contiguous loads become 16-byte ones, 1: no unrolling)"));

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
/// The address of a threadgroup load as SCEV understands it: the pointer (a
/// getelementptr of a variable), else the integer PTX computed it in
/// ((zext) X -> inttoptr -> constant offsets, plus those offsets).
static const SCEV *threadgroupAddress(const LoadInst *LI, ScalarEvolution &SE) {
  Value *Ptr = const_cast<Value *>(LI->getPointerOperand());
  const SCEV *S = SE.getSCEV(Ptr);
  if (isa<SCEVAddRecExpr>(S))
    return S;
  const DataLayout &DL = LI->getDataLayout();
  APInt Offset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
  const Value *P =
      Ptr->stripAndAccumulateConstantOffsets(DL, Offset, /*AllowNonInbounds=*/true);
  auto *I2P = dyn_cast<IntToPtrInst>(P);
  if (!I2P)
    return nullptr;
  Value *X = I2P->getOperand(0);
  if (auto *Z = dyn_cast<ZExtInst>(X))
    X = Z->getOperand(0);
  if (!SE.isSCEVable(X->getType()))
    return nullptr;
  const SCEV *XS = SE.getSCEV(X);
  return SE.getAddExpr(
      XS, SE.getConstant(X->getType(), Offset.getSExtValue(), true));
}

/// The unroll factor after which the threadgroup loads of L that read
/// consecutive bytes every iteration (loads at a, a + 4, ... covering the
/// 4 or 8 bytes by which a advances) cover 16: xyzw loads, once the
/// load/store vectorizer joins the copies (AGXThreadgroupAddresses gives
/// their addresses one base). 1 if there are none.
static unsigned contiguousThreadgroupFactor(Loop *L, ScalarEvolution &SE) {
  SmallVector<std::pair<const SCEV *, uint64_t>, 32> Loads;
  for (const BasicBlock *BB : L->blocks())
    for (const Instruction &I : *BB)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (LI->getPointerAddressSpace() == AGX::THREADGROUP_ADDRESS)
          if (const SCEV *A = threadgroupAddress(LI, SE))
            Loads.push_back(
                {A, LI->getDataLayout()
                        .getTypeStoreSize(LI->getType())
                        .getFixedValue()});
  unsigned Factor = 1;
  for (auto [A, Size] : Loads) {
    LLVM_DEBUG(dbgs() << "  threadgroup load " << *A << " size " << Size
                      << "\n");
    auto *AR = dyn_cast<SCEVAddRecExpr>(A);
    if (!AR || AR->getLoop() != L || !AR->isAffine())
      continue;
    auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    if (!Step || (Step->getAPInt() != 4 && Step->getAPInt() != 8))
      continue;
    uint64_t S = Step->getAPInt().getZExtValue();
    // follow the loads of the next bytes until the step is covered
    uint64_t Covered = Size, Last = Size;
    const SCEV *Next = A;
    while (Covered < S) {
      Next = SE.getAddExpr(Next, SE.getConstant(A->getType(), Last));
      auto *It = find_if(Loads, [&](auto &E) { return E.first == Next; });
      if (It == Loads.end())
        break;
      Last = It->second;
      Covered += Last;
    }
    if (Covered == S)
      Factor = std::max<unsigned>(Factor, 16 / S);
  }
  return Factor;
}

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
  if (!ThreadgroupLoads || L->getNumBlocks() != 1)
    return;
  if (AGXThreadgroupUnroll == 1)
    return;
  if (AGXThreadgroupUnroll == 0) {
    // By default only as far as contiguous threadgroup loads become 16-byte
    // ones, in loops with a known trip count that the factor divides (a
    // matrix product's inner loop over a tile once launch-time
    // specialization made the tile size a constant).
    unsigned Factor = contiguousThreadgroupFactor(L, SE);
    unsigned Trips = SE.getSmallConstantTripCount(L);
    LLVM_DEBUG(dbgs() << "AGX threadgroup unroll: factor " << Factor
                      << ", trip count " << Trips << "\n");
    if (Factor < 2 || Trips < Factor || Trips % Factor != 0)
      return;
    UP.Partial = true;
    UP.MaxCount = Factor;
    UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
    return;
  }
  UP.Partial = UP.Runtime = true;
  UP.MaxCount = AGXThreadgroupUnroll;
  // the unrolled body of such a loop (a tile of a matrix product) is a few
  // hundred instructions; the generic limit (150) would stop at its first copy
  UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
}
