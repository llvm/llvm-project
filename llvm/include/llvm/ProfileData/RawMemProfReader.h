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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/IR/GlobalValue.h"
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
using CallStackMap = llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t>>;

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

  using GuidMemProfRecordPair = std::pair<GlobalValue::GUID, MemProfRecord>;
  using Iterator = InstrProfIterator<GuidMemProfRecordPair, RawMemProfReader>;
  Iterator end() { return Iterator(); }
  Iterator begin() {
    Iter = FunctionProfileData.begin();
    return Iterator(this);
  }

  Error readNextRecord(GuidMemProfRecordPair &GuidRecord);

  // The RawMemProfReader only holds memory profile information.
  InstrProfKind getProfileKind() const { return InstrProfKind::MemProf; }

  // Constructor for unittests only.
  RawMemProfReader(std::unique_ptr<llvm::symbolize::SymbolizableModule> Sym,
                   llvm::SmallVectorImpl<SegmentEntry> &Seg,
                   llvm::MapVector<uint64_t, MemInfoBlock> &Prof,
                   CallStackMap &SM)
      : Symbolizer(std::move(Sym)), SegmentInfo(Seg.begin(), Seg.end()),
        CallstackProfileData(Prof), StackMap(SM) {
    // We don't call initialize here since there is no raw profile to read. The
    // test should pass in the raw profile as structured data.

    // If there is an error here then the mock symbolizer has not been
    // initialized properly.
    if (Error E = symbolizeAndFilterStackFrames())
      report_fatal_error(std::move(E));
    if (Error E = mapRawProfileToRecords())
      report_fatal_error(std::move(E));
  }

  // Return a const reference to the internal Id to Frame mappings.
  const llvm::DenseMap<FrameId, Frame> &getFrameMapping() const {
    return IdToFrame;
  }

  // Return a const reference to the internal function profile data.
  const llvm::MapVector<GlobalValue::GUID, IndexedMemProfRecord> &
  getProfileData() const {
    return FunctionProfileData;
  }

private:
  RawMemProfReader(std::unique_ptr<MemoryBuffer> DataBuffer,
                   object::OwningBinary<object::Binary> &&Bin)
      : DataBuffer(std::move(DataBuffer)), Binary(std::move(Bin)) {}
  Error initialize();
  Error readRawProfile();
  // Symbolize and cache all the virtual addresses we encounter in the
  // callstacks from the raw profile. Also prune callstack frames which we can't
  // symbolize or those that belong to the runtime. For profile entries where
  // the entire callstack is pruned, we drop the entry from the profile.
  Error symbolizeAndFilterStackFrames();
  // Construct memprof records for each function and store it in the
  // `FunctionProfileData` map. A function may have allocation profile data or
  // callsite data or both.
  Error mapRawProfileToRecords();

  // A helper method to extract the frame from the IdToFrame map.
  const Frame &idToFrame(const FrameId Id) const {
    auto It = IdToFrame.find(Id);
    assert(It != IdToFrame.end() && "Id not found in map.");
    return It->getSecond();
  }

  object::SectionedAddress getModuleOffset(uint64_t VirtualAddress);
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
  llvm::MapVector<uint64_t, MemInfoBlock> CallstackProfileData;
  CallStackMap StackMap;

  // Cached symbolization from PC to Frame.
  llvm::DenseMap<uint64_t, llvm::SmallVector<FrameId>> SymbolizedFrame;
  llvm::DenseMap<FrameId, Frame> IdToFrame;

  llvm::MapVector<GlobalValue::GUID, IndexedMemProfRecord> FunctionProfileData;
  llvm::MapVector<GlobalValue::GUID, IndexedMemProfRecord>::iterator Iter;
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_RAWMEMPROFREADER_H_
