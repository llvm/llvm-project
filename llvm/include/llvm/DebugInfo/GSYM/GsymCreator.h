//===- GsymCreator.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMCREATOR_H
#define LLVM_DEBUGINFO_GSYM_GSYMCREATOR_H

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
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace llvm {

namespace gsym {
class FileWriter;
class OutputAggregator;

/// GsymCreator is the base class for creating GSYM data.
///
/// The GsymCreator is designed to be used in 3 stages:
/// - Create FunctionInfo objects and add them
/// - Finalize the GsymCreator object
/// - Save to file or section
///
/// This base class contains all shared state and logic. Subclasses
/// (GsymCreatorV1, GsymCreatorV2) implement version-specific encoding.
class GsymCreator {
protected:
  mutable std::mutex Mutex;
  std::vector<FunctionInfo> Funcs;
  StringTableBuilder StrTab;
  StringSet<> StringStorage;
  DenseMap<llvm::gsym::FileEntry, uint32_t> FileEntryToIndex;
  DenseMap<uint64_t, CachedHashStringRef> StringOffsetMap;
  std::vector<llvm::gsym::FileEntry> Files;
  std::vector<uint8_t> UUID;
  std::optional<AddressRanges> ValidTextRanges;
  std::optional<uint64_t> BaseAddress;
  bool IsSegment = false;
  bool Finalized = false;
  bool Quiet;

  LLVM_ABI std::optional<uint64_t> getFirstFunctionAddress() const;
  LLVM_ABI std::optional<uint64_t> getLastFunctionAddress() const;
  LLVM_ABI std::optional<uint64_t> getBaseAddress() const;
  LLVM_ABI uint8_t getAddressOffsetSize() const;
  LLVM_ABI uint64_t getMaxAddressOffset() const;

  LLVM_ABI uint32_t insertFileEntry(FileEntry FE);
  LLVM_ABI uint64_t copyFunctionInfo(const GsymCreator &SrcGC,
                                     size_t FuncInfoIdx);
  LLVM_ABI uint32_t copyString(const GsymCreator &SrcGC, uint32_t StrOff);
  LLVM_ABI uint32_t copyFile(const GsymCreator &SrcGC, uint32_t FileIdx);
  LLVM_ABI void fixupInlineInfo(const GsymCreator &SrcGC, InlineInfo &II);

  LLVM_ABI llvm::Error saveSegments(StringRef Path,
                                    llvm::endianness ByteOrder,
                                    uint64_t SegmentSize) const;

  void setIsSegment() { IsSegment = true; }

  /// Version-specific: calculate header and table sizes.
  virtual uint64_t calculateHeaderAndTableSize() const = 0;

  /// Version-specific: create a new empty creator of the same version.
  virtual std::unique_ptr<GsymCreator> createNew(bool Quiet) const = 0;

public:
  LLVM_ABI GsymCreator(bool Quiet = false);
  virtual ~GsymCreator() = default;

  /// Version-specific: encode to a FileWriter.
  virtual llvm::Error encode(FileWriter &O) const = 0;

  /// Version-specific: load call site info from YAML.
  virtual llvm::Error loadCallSitesFromYAML(StringRef YAMLFile) = 0;

  LLVM_ABI llvm::Error
  save(StringRef Path, llvm::endianness ByteOrder,
       std::optional<uint64_t> SegmentSize = std::nullopt) const;

  LLVM_ABI uint32_t insertString(StringRef S, bool Copy = true);
  LLVM_ABI StringRef getString(uint32_t Offset);

  LLVM_ABI uint32_t
  insertFile(StringRef Path,
             sys::path::Style Style = sys::path::Style::native);

  LLVM_ABI void addFunctionInfo(FunctionInfo &&FI);
  LLVM_ABI size_t getNumFunctionInfos() const;

  LLVM_ABI void
  forEachFunctionInfo(
      std::function<bool(FunctionInfo &)> const &Callback);
  LLVM_ABI void forEachFunctionInfo(
      std::function<bool(const FunctionInfo &)> const &Callback) const;

  LLVM_ABI llvm::Error finalize(OutputAggregator &OS);
  LLVM_ABI void prepareMergedFunctions(OutputAggregator &Out);

  void setUUID(llvm::ArrayRef<uint8_t> UUIDBytes) {
    UUID.assign(UUIDBytes.begin(), UUIDBytes.end());
  }

  void setBaseAddress(uint64_t Addr) { BaseAddress = Addr; }

  void SetValidTextRanges(AddressRanges &TextRanges) {
    ValidTextRanges = TextRanges;
  }

  const std::optional<AddressRanges> GetValidTextRanges() const {
    return ValidTextRanges;
  }

  LLVM_ABI bool IsValidTextAddress(uint64_t Addr) const;
  bool isQuiet() const { return Quiet; }

  LLVM_ABI llvm::Expected<std::unique_ptr<GsymCreator>>
  createSegment(uint64_t SegmentSize, size_t &FuncIdx) const;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMCREATOR_H
