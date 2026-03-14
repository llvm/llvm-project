//=-- MemProfCommon.cpp - MemProf common utilities ---------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains MemProf common utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/MemProfCommon.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/HashBuilder.h"

using namespace llvm;
using namespace llvm::memprof;

namespace llvm {

// Upper bound on lifetime access density (accesses per byte per lifetime sec)
// for marking an allocation cold.
LLVM_ABI cl::opt<float> MemProfLifetimeAccessDensityColdThreshold(
    "memprof-lifetime-access-density-cold-threshold", cl::init(0.05),
    cl::Hidden,
    cl::desc("The threshold the lifetime access density (accesses per byte per "
             "lifetime sec) must be under to consider an allocation cold"));

// Lower bound on lifetime to mark an allocation cold (in addition to accesses
// per byte per sec above). This is to avoid pessimizing short lived objects.
LLVM_ABI cl::opt<unsigned> MemProfAveLifetimeColdThreshold(
    "memprof-ave-lifetime-cold-threshold", cl::init(200), cl::Hidden,
    cl::desc("The average lifetime (s) for an allocation to be considered "
             "cold"));

// Lower bound on average lifetime accesses density (total life time access
// density / alloc count) for marking an allocation hot.
LLVM_ABI cl::opt<unsigned> MemProfMinAveLifetimeAccessDensityHotThreshold(
    "memprof-min-ave-lifetime-access-density-hot-threshold", cl::init(1000),
    cl::Hidden,
    cl::desc("The minimum TotalLifetimeAccessDensity / AllocCount for an "
             "allocation to be considered hot"));

LLVM_ABI cl::opt<bool>
    MemProfUseHotHints("memprof-use-hot-hints", cl::init(false), cl::Hidden,
                       cl::desc("Enable use of hot hints (only supported for "
                                "unambigously hot allocations)"));

} // end namespace llvm

AllocationType llvm::memprof::getAllocType(uint64_t TotalLifetimeAccessDensity,
                                           uint64_t AllocCount,
                                           uint64_t TotalLifetime) {
  // The access densities are multiplied by 100 to hold 2 decimal places of
  // precision, so need to divide by 100.
  if (((float)TotalLifetimeAccessDensity) / AllocCount / 100 <
          MemProfLifetimeAccessDensityColdThreshold
      // Lifetime is expected to be in ms, so convert the threshold to ms.
      && ((float)TotalLifetime) / AllocCount >=
             MemProfAveLifetimeColdThreshold * 1000)
    return AllocationType::Cold;

  // The access densities are multiplied by 100 to hold 2 decimal places of
  // precision, so need to divide by 100.
  if (MemProfUseHotHints &&
      ((float)TotalLifetimeAccessDensity) / AllocCount / 100 >
          MemProfMinAveLifetimeAccessDensityHotThreshold)
    return AllocationType::Hot;

  return AllocationType::NotCold;
}

uint64_t llvm::memprof::computeFullStackId(ArrayRef<Frame> CallStack) {
  llvm::HashBuilder<llvm::TruncatedBLAKE3<8>, llvm::endianness::little>
      HashBuilder;
  for (auto &F : CallStack)
    HashBuilder.add(F.Function, F.LineOffset, F.Column);
  llvm::BLAKE3Result<8> Hash = HashBuilder.final();
  uint64_t Id;
  std::memcpy(&Id, Hash.data(), sizeof(Hash));
  return Id;
}
