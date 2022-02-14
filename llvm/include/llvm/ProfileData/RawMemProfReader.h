#ifndef LLVM_PROFILEDATA_RAWMEMPROFREADER_H_
#define LLVM_PROFILEDATA_RAWMEMPROFREADER_H_
//===- MemProfReader.h - Instrumented memory profiling reader ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading MemProf profiling data.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstddef>

namespace llvm {
namespace memprof {

// Map from id (recorded from sanitizer stack depot) to virtual addresses for
// each program counter address in the callstack.
using CallStackMap = llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 32>>;

class RawMemProfReader {
public:
  RawMemProfReader(std::unique_ptr<MemoryBuffer> DataBuffer)
      : DataBuffer(std::move(DataBuffer)) {}
  RawMemProfReader(const RawMemProfReader &) = delete;
  RawMemProfReader &operator=(const RawMemProfReader &) = delete;

  // Prints the contents of the profile in YAML format.
  void printYAML(raw_ostream &OS);

  // Return true if the \p DataBuffer starts with magic bytes indicating it is
  // a raw binary memprof profile.
  static bool hasFormat(const MemoryBuffer &DataBuffer);
  // Return true if the file at \p Path starts with magic bytes indicating it is
  // a raw binary memprof profile.
  static bool hasFormat(const StringRef Path);

  // Create a RawMemProfReader after sanity checking the contents of the file at
  // \p Path. The binary from which the profile has been collected is specified
  // via a path in \p ProfiledBinary.
  static Expected<std::unique_ptr<RawMemProfReader>>
  create(const Twine &Path, const StringRef ProfiledBinary);

  Error readNextRecord(MemProfRecord &Record);

  using Iterator = InstrProfIterator<MemProfRecord, RawMemProfReader>;
  Iterator end() { return Iterator(); }
  Iterator begin() {
    Iter = ProfileData.begin();
    return Iterator(this);
  }

  // Constructor for unittests only.
  RawMemProfReader(std::unique_ptr<llvm::symbolize::SymbolizableModule> Sym,
                   llvm::SmallVectorImpl<SegmentEntry> &Seg,
                   llvm::MapVector<uint64_t, MemInfoBlock> &Prof,
                   CallStackMap &SM)
      : Symbolizer(std::move(Sym)), SegmentInfo(Seg.begin(), Seg.end()),
        ProfileData(Prof), StackMap(SM) {}

private:
  RawMemProfReader(std::unique_ptr<MemoryBuffer> DataBuffer,
                   object::OwningBinary<object::Binary> &&Bin)
      : DataBuffer(std::move(DataBuffer)), Binary(std::move(Bin)) {}
  Error initialize();
  Error readRawProfile();

  object::SectionedAddress getModuleOffset(uint64_t VirtualAddress);
  Error fillRecord(const uint64_t Id, const MemInfoBlock &MIB,
                   MemProfRecord &Record);
  // Prints aggregate counts for each raw profile parsed from the DataBuffer in
  // YAML format.
  void printSummaries(raw_ostream &OS) const;

  std::unique_ptr<MemoryBuffer> DataBuffer;
  object::OwningBinary<object::Binary> Binary;
  std::unique_ptr<llvm::symbolize::SymbolizableModule> Symbolizer;

  // The contents of the raw profile.
  llvm::SmallVector<SegmentEntry, 16> SegmentInfo;
  // A map from callstack id (same as key in CallStackMap below) to the heap
  // information recorded for that allocation context.
  llvm::MapVector<uint64_t, MemInfoBlock> ProfileData;
  CallStackMap StackMap;

  // Iterator to read from the ProfileData MapVector.
  llvm::MapVector<uint64_t, MemInfoBlock>::iterator Iter = ProfileData.end();
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_RAWMEMPROFREADER_H_
