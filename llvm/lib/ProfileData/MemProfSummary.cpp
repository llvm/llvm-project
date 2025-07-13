//=-- MemProfSummary.cpp - MemProf summary support ---------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains MemProf summary support and related interfaces.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/MemProfSummary.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/HashBuilder.h"

using namespace llvm;
using namespace llvm::memprof;

// Upper bound on lifetime access density (accesses per byte per lifetime sec)
// for marking an allocation cold.
cl::opt<float> MemProfLifetimeAccessDensityColdThreshold(
    "memprof-lifetime-access-density-cold-threshold", cl::init(0.05),
    cl::Hidden,
    cl::desc("The threshold the lifetime access density (accesses per byte per "
             "lifetime sec) must be under to consider an allocation cold"));

// Lower bound on lifetime to mark an allocation cold (in addition to accesses
// per byte per sec above). This is to avoid pessimizing short lived objects.
cl::opt<unsigned> MemProfAveLifetimeColdThreshold(
    "memprof-ave-lifetime-cold-threshold", cl::init(200), cl::Hidden,
    cl::desc("The average lifetime (s) for an allocation to be considered "
             "cold"));

// Lower bound on average lifetime accesses density (total life time access
// density / alloc count) for marking an allocation hot.
cl::opt<unsigned> MemProfMinAveLifetimeAccessDensityHotThreshold(
    "memprof-min-ave-lifetime-access-density-hot-threshold", cl::init(1000),
    cl::Hidden,
    cl::desc("The minimum TotalLifetimeAccessDensity / AllocCount for an "
             "allocation to be considered hot"));

cl::opt<bool>
    MemProfUseHotHints("memprof-use-hot-hints", cl::init(false), cl::Hidden,
                       cl::desc("Enable use of hot hints (only supported for "
                                "unambigously hot allocations)"));

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

void MemProfSummary::printSummaryYaml(raw_ostream &OS) const {
  // For now emit as YAML comments, since they aren't read on input.
  OS << "---\n";
  OS << "# MemProfSummary:\n";
  OS << "#   Total contexts: " << NumContexts << "\n";
  OS << "#   Total cold contexts: " << NumColdContexts << "\n";
  OS << "#   Total hot contexts: " << NumHotContexts << "\n";
  OS << "#   Maximum cold context total size: " << MaxColdTotalSize << "\n";
  OS << "#   Maximum warm context total size: " << MaxWarmTotalSize << "\n";
  OS << "#   Maximum hot context total size: " << MaxHotTotalSize << "\n";
}

void MemProfSummary::write(ProfOStream &OS) const {
  // Write the current number of fields first, which helps enable backwards and
  // forwards compatibility (see comment in header).
  OS.write32(memprof::MemProfSummary::getNumSummaryFields());
  auto StartPos = OS.tell();
  (void)StartPos;
  OS.write(NumContexts);
  OS.write(NumColdContexts);
  OS.write(NumHotContexts);
  OS.write(MaxColdTotalSize);
  OS.write(MaxWarmTotalSize);
  OS.write(MaxHotTotalSize);
  // Sanity check that the number of fields was kept in sync with actual fields.
  assert((OS.tell() - StartPos) / 8 == MemProfSummary::getNumSummaryFields());
}

std::unique_ptr<MemProfSummary>
MemProfSummary::deserialize(const unsigned char *&Ptr) {
  auto NumSummaryFields =
      support::endian::readNext<uint32_t, llvm::endianness::little>(Ptr);
  // The initial version of the summary contains 6 fields. To support backwards
  // compatibility with older profiles, if new summary fields are added (until a
  // version bump) this code will need to check NumSummaryFields against the
  // current value of MemProfSummary::getNumSummaryFields(). If NumSummaryFields
  // is lower then default values will need to be filled in for the newer fields
  // instead of trying to read them from the profile.
  //
  // For now, assert that the profile contains at least as many fields as
  // expected by the code.
  assert(NumSummaryFields >= MemProfSummary::getNumSummaryFields());

  auto MemProfSum = std::make_unique<MemProfSummary>(
      support::endian::read<uint64_t, llvm::endianness::little>(Ptr),
      support::endian::read<uint64_t, llvm::endianness::little>(Ptr + 8),
      support::endian::read<uint64_t, llvm::endianness::little>(Ptr + 16),
      support::endian::read<uint64_t, llvm::endianness::little>(Ptr + 24),
      support::endian::read<uint64_t, llvm::endianness::little>(Ptr + 32),
      support::endian::read<uint64_t, llvm::endianness::little>(Ptr + 40));

  // Enable forwards compatibility by skipping past any additional fields in the
  // profile's summary.
  Ptr += NumSummaryFields * sizeof(uint64_t);

  return MemProfSum;
}
