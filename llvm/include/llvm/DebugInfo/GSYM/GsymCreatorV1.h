//===- GsymCreatorV1.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMCREATORV1_H
#define LLVM_DEBUGINFO_GSYM_GSYMCREATORV1_H

#include "llvm/Support/Compiler.h"
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include "llvm/ADT/AddressRanges.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/GsymCreator.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace llvm {

namespace gsym {
class FileWriter;
class OutputAggregator;

/// GsymCreatorV1 is used to emit GSYM V1 data to a stand alone file or section
/// within a file.
///
/// See GsymCreator for the 3-stage usage pattern and file format documentation.
class GsymCreatorV1 : public GsymCreator {
  // Private member variables require Mutex protections
  mutable std::mutex Mutex;
  std::vector<FunctionInfo> Funcs;
  StringTableBuilder StrTab;
  StringSet<> StringStorage;
  DenseMap<llvm::gsym::FileEntry, uint32_t> FileEntryToIndex;
  // Needed for mapping string offsets back to the string stored in \a StrTab.
  DenseMap<uint64_t, CachedHashStringRef> StringOffsetMap;
  std::vector<llvm::gsym::FileEntry> Files;
  std::vector<uint8_t> UUID;
  std::optional<AddressRanges> ValidTextRanges;
  std::optional<uint64_t> BaseAddress;
  bool IsSegment = false;
  bool Finalized = false;
  bool Quiet;


  std::optional<uint64_t> getFirstFunctionAddress() const;
  std::optional<uint64_t> getLastFunctionAddress() const;
  std::optional<uint64_t> getBaseAddress() const;
  uint8_t getAddressOffsetSize() const;
  uint64_t getMaxAddressOffset() const;
  uint64_t calculateHeaderAndTableSize() const;
  uint64_t copyFunctionInfo(const GsymCreatorV1 &SrcGC, size_t FuncInfoIdx);
  uint32_t copyString(const GsymCreatorV1 &SrcGC, uint32_t StrOff);
  uint32_t copyFile(const GsymCreatorV1 &SrcGC, uint32_t FileIdx);
  uint32_t insertFileEntry(FileEntry FE);
  void fixupInlineInfo(const GsymCreatorV1 &SrcGC, InlineInfo &II);
  llvm::Error saveSegments(StringRef Path, llvm::endianness ByteOrder,
                           uint64_t SegmentSize) const;
  void setIsSegment() {
    IsSegment = true;
  }

public:
  LLVM_ABI GsymCreatorV1(bool Quiet = false);

  LLVM_ABI llvm::Error
  save(StringRef Path, llvm::endianness ByteOrder,
       std::optional<uint64_t> SegmentSize = std::nullopt) const override;

  LLVM_ABI llvm::Error encode(FileWriter &O) const override;

  LLVM_ABI uint32_t insertString(StringRef S, bool Copy = true) override;

  LLVM_ABI StringRef getString(uint32_t Offset) override;

  LLVM_ABI uint32_t
  insertFile(StringRef Path,
             sys::path::Style Style = sys::path::Style::native) override;

  LLVM_ABI void addFunctionInfo(FunctionInfo &&FI) override;

  LLVM_ABI llvm::Error loadCallSitesFromYAML(StringRef YAMLFile) override;

  LLVM_ABI void prepareMergedFunctions(OutputAggregator &Out) override;

  LLVM_ABI llvm::Error finalize(OutputAggregator &OS) override;

  void setUUID(llvm::ArrayRef<uint8_t> UUIDBytes) override {
    UUID.assign(UUIDBytes.begin(), UUIDBytes.end());
  }

  LLVM_ABI void
  forEachFunctionInfo(
      std::function<bool(FunctionInfo &)> const &Callback) override;

  LLVM_ABI void forEachFunctionInfo(
      std::function<bool(const FunctionInfo &)> const &Callback) const override;

  LLVM_ABI size_t getNumFunctionInfos() const override;

  void SetValidTextRanges(AddressRanges &TextRanges) override {
    ValidTextRanges = TextRanges;
  }

  const std::optional<AddressRanges> GetValidTextRanges() const override {
    return ValidTextRanges;
  }

  LLVM_ABI bool IsValidTextAddress(uint64_t Addr) const override;

  void setBaseAddress(uint64_t Addr) override {
    BaseAddress = Addr;
  }

  bool isQuiet() const override { return Quiet; }

  LLVM_ABI llvm::Expected<std::unique_ptr<GsymCreatorV1>>
  createSegment(uint64_t SegmentSize, size_t &FuncIdx) const;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMCREATORV1_H
