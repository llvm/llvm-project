//=-- MemProfSummary.cpp - MemProf summary support ---------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains MemProf summary support.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/MemProfSummary.h"

using namespace llvm;
using namespace llvm::memprof;

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
