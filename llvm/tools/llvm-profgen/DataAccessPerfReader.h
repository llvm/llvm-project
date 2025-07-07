//===-- DataAccessPerfReader.h - perfscript reader for data access profiles -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_DATAACCESSPERFREADER_H
#define LLVM_TOOLS_LLVM_PROFGEN_DATAACCESSPERFREADER_H

#include "PerfReader.h"
#include "ProfiledBinary.h"
#include "llvm/ADT/MapVector.h"

namespace llvm {

class DataAccessPerfReader : public PerfScriptReader {
public:
  class DataSegment {
  public:
    uint64_t FileOffset;
    uint64_t VirtualAddress;
  };
  DataAccessPerfReader(ProfiledBinary *Binary, StringRef PerfTrace,
                       std::optional<int32_t> PID)
      : PerfScriptReader(Binary, PerfTrace, PID, false),
        PerfTraceFilename(PerfTrace) {
    // Assume this is non-pie binary.
  }

  // Entry of the reader to parse multiple perf traces
  void parsePerfTraces() override;

  // A hack to demonstrate the symbolized output of vtable type profiling.
  void print() const {

    std::vector<ProfiledInfo> Entries;
    Entries.reserve(AddressMap.size());
    for (const auto &[IpAddr, DataCount] : AddressMap) {
      for (const auto [DataAddr, Count] : DataCount) {
        Entries.emplace_back(ProfiledInfo(IpAddr, DataAddr, Count));
      }
    }
    llvm::sort(Entries,
               [](const auto &A, const auto &B) { return A.Count > B.Count; });
    for (const auto &Entry : Entries) {
      if (Entry.Count == 0)
        continue; // Skip entries with zero count
      dbgs() << "Address: " << format("0x%llx", Entry.InstructionAddr)
             << " Data Address: " << Entry.DataSymbol
             << " Count: " << Entry.Count << "\n";
    }
  }

  const DenseMap<uint64_t, DenseMap<StringRef, uint64_t>> &
getAddressMap() const {
    return AddressMap;
  }

private:
  void parsePerfTrace(StringRef PerfTrace);

  DenseMap<uint64_t, DenseMap<StringRef, uint64_t>> AddressMap;

  StringRef PerfTraceFilename;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_PROFGEN_DATAACCESSPERFREADER_H
