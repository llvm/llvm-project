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
      : PerfScriptReader(Binary, PerfTrace, PID), PerfTraceFilename(PerfTrace) {
    hackMMapEventAndDataSegment(DataMMap, DataSegment, *Binary);
  }

  // The MMapEvent is hard-coded as a hack to illustrate the change.
  static void
  hackMMapEventAndDataSegment(PerfScriptReader::MMapEvent &MMap,
                              DataSegment &DataSegment,
                              const ProfiledBinary &ProfiledBinary) {
    // The PERF_RECORD_MMAP2 event is
    // 0 0x4e8 [0xa0]: PERF_RECORD_MMAP2 1849842/1849842:
    // [0x55d977426000(0x1000) @ 0x1000 fd:01 20869534 0]: r--p /path/to/binary
    MMap.PID = 256393; // Example PID
    MMap.BinaryPath = ProfiledBinary.getPath();
    MMap.Address = 0x00000000003b3e40;
    MMap.Size = 0x00b1c0;
    MMap.Offset = 0; // File Offset in the binary.

    // TODO: Set binary fields to do address canonicalization, and compute
    // static data address range.
    DataSegment.FileOffset =
        0xdb0; // The byte offset of the segment start in the binary.
    DataSegment.VirtualAddress =
        0x2db0; // The virtual address of the segment start in the binary.
  }

  uint64_t canonicalizeDataAddress(uint64_t Address,
                                   const ProfiledBinary &ProfiledBinary,
                                   const PerfScriptReader::MMapEvent &MMap,
                                   const DataSegment &DataSegment) {
    return Address;
    // virtual-addr = segment.virtual-addr (0x3180) + (runtime-addr -
    // map.adddress - segment.file-offset (0x1180) + map.file-offset (0x1000))
    return DataSegment.VirtualAddress +
           (Address - MMap.Address - (DataSegment.FileOffset - MMap.Offset));
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
             << " Data Address: " << format("0x%llx", Entry.DataAddr)
             << " Count: " << Entry.Count << "\n";
    }
  }

  const DenseMap<uint64_t, DenseMap<uint64_t, uint64_t>> &
getAddressMap() const {
    return AddressMap;
  }


private:
  void parsePerfTrace(StringRef PerfTrace);

  DenseMap<uint64_t, DenseMap<uint64_t, uint64_t>> AddressMap;

  StringRef PerfTraceFilename;

  PerfScriptReader::MMapEvent DataMMap;
  DataSegment DataSegment;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_PROFGEN_DATAACCESSPERFREADER_H
