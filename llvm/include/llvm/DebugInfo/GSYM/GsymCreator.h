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

/// GsymCreator is an abstract interface for creating GSYM data.
///
/// The GsymCreator is designed to be used in 3 stages:
/// - Create FunctionInfo objects and add them
/// - Finalize the GsymCreator object
/// - Save to file or section
///
/// The first stage involves creating FunctionInfo objects from another source
/// of information like compiler debug info metadata, DWARF or Breakpad files.
/// Any strings in the FunctionInfo or contained information, like InlineInfo
/// or LineTable objects, should get the string table offsets by calling
/// GsymCreator::insertString(...). Any file indexes that are needed should be
/// obtained by calling GsymCreator::insertFile(...). All of the function calls
/// in GsymCreator are thread safe. This allows multiple threads to create and
/// add FunctionInfo objects while parsing debug information.
///
/// Once all of the FunctionInfo objects have been added, the
/// GsymCreator::finalize(...) must be called prior to saving. This function
/// will sort the FunctionInfo objects, finalize the string table, and do any
/// other passes on the information needed to prepare the information to be
/// saved.
///
/// Once the object has been finalized, it can be saved to a file or section.
///
/// Both GsymCreatorV1 and GsymCreatorV2 implement this interface.
class GsymCreator {
public:
  virtual ~GsymCreator() = default;

  virtual uint32_t insertString(StringRef S, bool Copy = true) = 0;
  virtual StringRef getString(uint32_t Offset) = 0;
  virtual uint32_t
  insertFile(StringRef Path,
             sys::path::Style Style = sys::path::Style::native) = 0;
  virtual void addFunctionInfo(FunctionInfo &&FI) = 0;
  virtual size_t getNumFunctionInfos() const = 0;
  virtual void
  forEachFunctionInfo(
      std::function<bool(FunctionInfo &)> const &Callback) = 0;
  virtual void forEachFunctionInfo(
      std::function<bool(const FunctionInfo &)> const &Callback) const = 0;
  virtual llvm::Error finalize(OutputAggregator &OS) = 0;
  virtual llvm::Error
  save(StringRef Path, llvm::endianness ByteOrder,
       std::optional<uint64_t> SegmentSize = std::nullopt) const = 0;
  virtual llvm::Error encode(FileWriter &O) const = 0;
  virtual llvm::Error loadCallSitesFromYAML(StringRef YAMLFile) = 0;
  virtual void prepareMergedFunctions(OutputAggregator &Out) = 0;

  virtual void setUUID(llvm::ArrayRef<uint8_t> UUIDBytes) = 0;
  virtual void setBaseAddress(uint64_t Addr) = 0;
  virtual void SetValidTextRanges(AddressRanges &TextRanges) = 0;
  virtual const std::optional<AddressRanges> GetValidTextRanges() const = 0;
  virtual bool IsValidTextAddress(uint64_t Addr) const = 0;
  virtual bool isQuiet() const = 0;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMCREATOR_H
