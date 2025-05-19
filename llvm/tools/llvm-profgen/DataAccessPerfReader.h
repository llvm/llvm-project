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
  DataAccessPerfReader(ProfiledBinary *Binary, StringRef PerfTrace,
                       std::optional<int32_t> PID)
      : PerfScriptReader(Binary, PerfTrace, PID), PerfTraceFilename(PerfTrace) {
  }

  // Entry of the reader to parse multiple perf traces
  void parsePerfTraces() override;

  auto getAddressToCount() const {
    return AddressToCount.getArrayRef();
  }

  void print() const {
    auto addrCountArray = AddressToCount.getArrayRef();
    std::vector<std::pair<uint64_t, uint64_t>> SortedEntries(
        addrCountArray.begin(), addrCountArray.end());
    llvm::sort(SortedEntries, [](const auto &A, const auto &B) {
      return A.second > B.second;
    });
    for (const auto &Entry : SortedEntries) {
      if (Entry.second == 0)
        continue; // Skip entries with zero count
      dbgs() << "Address: " << format("0x%llx", Entry.first)
             << ", Count: " << Entry.second << "\n";
    }
  }

private:
  void parsePerfTrace(StringRef PerfTrace);

  MapVector<uint64_t, uint64_t> AddressToCount;

  StringRef PerfTraceFilename;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_PROFGEN_DATAACCESSPERFREADER_H
