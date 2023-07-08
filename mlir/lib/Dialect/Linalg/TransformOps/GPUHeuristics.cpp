//===- GPUHeuristics.cpp - Heuristics Implementation for Transforms -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/GPUHeuristics.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <numeric>

using namespace mlir;

#define DEBUG_TYPE "linalg-transforms"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static Attribute linearIdX(MLIRContext *ctx) {
  return gpu::GPULinearIdMappingAttr::get(ctx, gpu::LinearId::DimX);
}
static Attribute linearIdY(MLIRContext *ctx) {
  return gpu::GPULinearIdMappingAttr::get(ctx, gpu::LinearId::DimY);
}
static Attribute linearIdZ(MLIRContext *ctx) {
  return gpu::GPULinearIdMappingAttr::get(ctx, gpu::LinearId::DimZ);
}

transform::gpu::CopyMappingInfo::CopyMappingInfo(MLIRContext *ctx,
                                                 int totalNumThreads,
                                                 int64_t desiredBitAlignment,
                                                 ArrayRef<int64_t> copySizes,
                                                 bool favorPredication,
                                                 int64_t elementalBitwidth) {
  assert(!copySizes.empty() && copySizes.size() <= 3 &&
         "only 1,2,3-D copies are supported for now");

  LDBG("START CopyMappingInfo, favorPredication: " << favorPredication);
  LLVM_DEBUG(llvm::interleaveComma(copySizes, DBGS() << "--copy shape: ");
             llvm::dbgs() << "\n";);

  // Greedily find the largest vector size that can be used to copy the most
  // minor dimension: we are in the business of filling kMaxVectorLoadBitWidth
  // contiguous memory transactions with as few threads as possible.
  int64_t desiredVectorSize = CopyMappingInfo::maxContiguousElementsToTransfer(
      desiredBitAlignment, copySizes.back(), elementalBitwidth);

  LDBG("--greedily determined vectorSize: "
       << desiredVectorSize << " elements of " << elementalBitwidth
       << "b each -> " << (desiredVectorSize * elementalBitwidth)
       << "b total out of a max of " << kMaxVectorLoadBitWidth << "b");

  status = inferNumThreads(totalNumThreads, copySizes, desiredVectorSize,
                           favorPredication);
  if (status == Status::Invalid)
    return;

  LLVM_DEBUG(llvm::interleaveComma(copySizes, DBGS() << "--copy: ");
             llvm::dbgs() << "\n"; llvm::interleaveComma(
                 this->numThreads, DBGS() << "--numThreads: ");
             llvm::dbgs() << "\n";);
  LDBG("--vectorSize: " << this->vectorSize);
  assert(this->numThreads.size() == copySizes.size() &&
         "compute copy mapping expected same number of threads and copy sizes");

  // Compute the smallest bounding box.
  this->smallestBoundingTileSizes = llvm::to_vector(
      llvm::map_range(llvm::zip(copySizes, this->numThreads), [](auto &&pair) {
        int64_t size, numThreads;
        std::tie(size, numThreads) = pair;
        return mlir::ceilDiv(size, numThreads);
      }));
  SmallVector<Attribute> allThreadMappings{linearIdZ(ctx), linearIdY(ctx),
                                           linearIdX(ctx)};

  // Set the thread mapping.
  this->threadMapping =
      llvm::to_vector(ArrayRef(allThreadMappings)
                          .take_back(this->smallestBoundingTileSizes.size()));
  LLVM_DEBUG(this->print(DBGS()); llvm::dbgs() << "\n");
}

int64_t transform::gpu::CopyMappingInfo::maxContiguousElementsToTransfer(
    int64_t desiredBitAlignment, int64_t numContiguousElements,
    int64_t elementalBitwidth) {
  assert(kMaxVectorLoadBitWidth % elementalBitwidth == 0 &&
         "elemental bitwidth does not divide kMaxVectorLoadBitWidth");
  assert(desiredBitAlignment % elementalBitwidth == 0 &&
         "elemental bitwidth does not divide desired bit alignment");
  return std::gcd(
      std::gcd(desiredBitAlignment / elementalBitwidth, numContiguousElements),
      kMaxVectorLoadBitWidth / elementalBitwidth);
}

/// Get the list of all factors that divide `val`, not just the prime factors.
static SmallVector<int64_t> getFactors(int64_t val) {
  SmallVector<int64_t> factors;
  factors.reserve(val);
  for (int64_t factor = 1; factor <= val; ++factor) {
    if (val % factor != 0)
      continue;
    factors.push_back(factor);
  }
  factors.push_back(val);
  return factors;
}

static int64_t product(ArrayRef<int64_t> vals) {
  int64_t res = 1;
  for (auto val : vals)
    res *= val;
  return res;
}

/// Extract `result` from `sizes` with the following constraints:
///   1. sizes[i] % result[i] for all i
///   2. product_of_threadsPerDim <= maxNumThreads
///   3. if `currentIndex` is sizes.size() - 1, then threadsPerDim[currentIndex]
///      must be sizes[currentIndex].
/// This is used to greedily extract the maximum number of threads usable for
/// mapping a copy of size `sizes`, while being bounded by `totalNumThreads` and
/// ensuring coalesced access along the most minor dimension.
/// Return the number of threads used in the range:
///   threadsPerDim[currentIndex .. sizes.end()]
// The implementation uses a dynamic programming approach to greedily extract
// the best combination under the constraints.
// TODO: Implementation details can be improved but putting effort there is a
// tradeoffs: `sizes` is expected to be of small rank and contain small values.
static SmallVector<int64_t> maximizeNumThreads(ArrayRef<int64_t> sizes,
                                               int64_t currentIndex,
                                               int64_t maxNumThreads) {
  assert(static_cast<size_t>(currentIndex) < sizes.size() &&
         "currentIndex out of bounds");
  std::string indent(2 * currentIndex, '-');
  if (static_cast<size_t>(currentIndex) == sizes.size() - 1) {
    LDBG(indent << "mandated globalBest: " << sizes[currentIndex]);
    return SmallVector<int64_t>{sizes[currentIndex]};
  }

  int64_t best = 0;
  int64_t s = sizes[currentIndex];
  SmallVector<int64_t> factors = getFactors(s);
  SmallVector<int64_t> localThreadsPerDim;
  localThreadsPerDim.reserve(sizes.size());
  LDBG(indent << "maximizeNumThreads in " << s
              << " with limit: " << maxNumThreads);
  for (auto factor : factors) {
    auto nestedThreadsPerDim =
        maximizeNumThreads(sizes, currentIndex + 1, maxNumThreads / factor);
    int64_t localBest = factor * product(nestedThreadsPerDim);
    if (localBest > best && localBest <= maxNumThreads) {
      LDBG(indent << "new localBest: " << localBest);
      LLVM_DEBUG(
          llvm::interleaveComma(nestedThreadsPerDim,
                                DBGS() << indent << "nestedThreadsPerDim: ");
          llvm::dbgs() << "\n";);
      localThreadsPerDim.clear();
      localThreadsPerDim.push_back(factor);
      llvm::append_range(localThreadsPerDim, nestedThreadsPerDim);
      best = localBest;
    }
  }

  LDBG(indent << "found globalBest: " << best);
  LLVM_DEBUG(llvm::interleaveComma(localThreadsPerDim,
                                   DBGS() << indent << "numThreads: ");
             llvm::dbgs() << "\n";);

  return localThreadsPerDim;
}

transform::gpu::CopyMappingInfo::Status
transform::gpu::CopyMappingInfo::inferNumThreads(int64_t totalNumThreads,
                                                 ArrayRef<int64_t> sizes,
                                                 int64_t desiredVectorSize,
                                                 bool favorPredication) {

  if (!favorPredication) {
    int64_t localVectorSize = desiredVectorSize;
    for (; localVectorSize >= 1; localVectorSize /= 2) {
      // Attempt to map the copy with predication and current fixed vector size:
      //   1. if the status is Success, we are done.
      //   2. if the status is Invalid, we fail immediately, no amount of
      //   vector size reduction can offset the bad tile size selection from the
      //   higher-level.
      //   3. if the status is RequiresPredication, we try again with a smaller
      //   vector size.
      Status status =
          inferNumThreadsImpl(totalNumThreads, sizes, localVectorSize);
      if (status == Status::Success || status == Status::Invalid)
        return status;

      LDBG("requires predication, try reducing vector size to "
           << (localVectorSize / 2));
    }
  }

  // If we have not yet returned, it means that we have tried all vector sizes
  // and we still require predication. Restart from the original vector size and
  // do not attempt to
  return inferNumThreadsImpl(totalNumThreads, sizes, desiredVectorSize);
}

transform::gpu::CopyMappingInfo::Status
transform::gpu::CopyMappingInfo::inferNumThreadsImpl(
    int64_t totalNumThreads, ArrayRef<int64_t> sizes,
    int64_t desiredVectorSize) {
  assert(sizes.back() % desiredVectorSize == 0 &&
         "most-minor size not divisible by actualVectorSize");

  LDBG("inferNumThreadsImpl with totalNumThreads: "
       << totalNumThreads << " and vectorSize: " << desiredVectorSize);

  // Scale the most minor size to account for the chosen vector size and
  // maximize the number of threads without exceeding the total number of
  // threads.
  SmallVector<int64_t> scaledSizes{sizes};
  scaledSizes.back() /= desiredVectorSize;
  if (scaledSizes.back() > totalNumThreads) {
    LDBG("--Too few threads given the required vector size -> FAIL");
    return Status::Invalid;
  }
  SmallVector<int64_t> inferredNumThreads =
      maximizeNumThreads(scaledSizes, 0, totalNumThreads);

  LLVM_DEBUG(llvm::interleaveComma(inferredNumThreads,
                                   DBGS() << "inferred numThreads: ");
             llvm::dbgs() << "\n";
             LDBG("computed actualVectorSize: " << desiredVectorSize););

  // Corner case: we cannot use more threads than available. If the dimension of
  // the copy is so bad it is because higher-level tiling did not do its job, we
  // do not try to recover from it here.
  int64_t totalNumThreadsUsed = product(inferredNumThreads);
  LDBG("--totalNumThreadsUsed: " << totalNumThreadsUsed);
  if (totalNumThreadsUsed == 0 || totalNumThreadsUsed > totalNumThreads) {
    LDBG("--Too few threads given the required vector size -> FAIL");
    return Status::Invalid;
  }

  this->vectorSize = desiredVectorSize;
  this->numThreads = inferredNumThreads;
  if (totalNumThreadsUsed == totalNumThreads)
    return Status::Success;

  return Status::RequiresPredication;
}

void transform::gpu::CopyMappingInfo::print(llvm::raw_ostream &os) const {
  os << "MappingInfo{";
  os << "CopyMappingInfo: ";
  os << "valid: " << (status != Status::Invalid) << ", ";
  os << "vectorSize: " << vectorSize << ", ";
  llvm::interleaveComma(numThreads, os << ", numThreads: {");
  llvm::interleaveComma(smallestBoundingTileSizes,
                        os << "}, smallestBoundingTileSizes: {");
  llvm::interleaveComma(threadMapping, os << "}, threadMapping: {");
  os << "}}";
}
