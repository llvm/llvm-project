//===- GsymReader.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_GSYMREADER_H
#define LLVM_DEBUGINFO_GSYM_GSYMREADER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/GSYM/FileEntry.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/DebugInfo/GSYM/LineEntry.h"
#include "llvm/DebugInfo/GSYM/StringTable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorOr.h"
#include <inttypes.h>
#include <memory>
#include <stdint.h>
#include <vector>

namespace llvm {
class MemoryBuffer;
class raw_ostream;

namespace gsym {

/// GsymReader is an abstract interface for reading GSYM data.
///
/// This interface provides the methods needed by FunctionInfo::lookup and
/// InlineInfo::lookup to resolve strings and files during symbolication.
/// Both GsymReaderV1 and GsymReaderV2 implement this interface.
class GsymReader {
public:
  virtual ~GsymReader() = default;

  /// Open a GSYM file, auto-detecting the format version.
  ///
  /// \param Path The file path of the GSYM file to read.
  /// \returns An expected unique_ptr to a GsymReader or an error.
  LLVM_ABI static llvm::Expected<std::unique_ptr<GsymReader>>
  openFile(StringRef Path);

  /// Construct a GsymReader from a buffer, auto-detecting the format version.
  ///
  /// \param Bytes A set of bytes that will be copied and owned by the
  /// returned object on success.
  /// \returns An expected unique_ptr to a GsymReader or an error.
  LLVM_ABI static llvm::Expected<std::unique_ptr<GsymReader>>
  copyBuffer(StringRef Bytes);

  /// Get a string from the string table.
  virtual StringRef getString(uint32_t Offset) const = 0;

  /// Get a file entry for the supplied file index.
  virtual std::optional<FileEntry> getFile(uint32_t Index) const = 0;

  /// Get the full function info for an address.
  virtual llvm::Expected<FunctionInfo> getFunctionInfo(uint64_t Addr) const = 0;

  /// Get the full function info given an address index.
  virtual llvm::Expected<FunctionInfo>
  getFunctionInfoAtIndex(uint64_t AddrIdx) const = 0;

  /// Lookup an address in the GSYM.
  virtual llvm::Expected<LookupResult>
  lookup(uint64_t Addr,
         std::optional<DataExtractor> *MergedFuncsData = nullptr) const = 0;

  /// Lookup all merged functions for a given address.
  virtual llvm::Expected<std::vector<LookupResult>>
  lookupAll(uint64_t Addr) const = 0;

  /// Get the number of addresses in this GSYM file.
  virtual uint32_t getNumAddresses() const = 0;

  /// Gets an address from the address table.
  virtual std::optional<uint64_t> getAddress(size_t Index) const = 0;

  /// Dump the entire GSYM data contained in this object.
  virtual void dump(raw_ostream &OS) = 0;

  /// Dump a FunctionInfo object.
  virtual void dump(raw_ostream &OS, const FunctionInfo &FI,
                    uint32_t Indent = 0) = 0;

  /// Dump a MergedFunctionsInfo object.
  virtual void dump(raw_ostream &OS, const MergedFunctionsInfo &MFI) = 0;

  /// Dump a CallSiteInfo object.
  virtual void dump(raw_ostream &OS, const CallSiteInfo &CSI) = 0;

  /// Dump a CallSiteInfoCollection object.
  virtual void dump(raw_ostream &OS, const CallSiteInfoCollection &CSIC,
                    uint32_t Indent = 0) = 0;

  /// Dump a LineTable object.
  virtual void dump(raw_ostream &OS, const LineTable &LT,
                    uint32_t Indent = 0) = 0;

  /// Dump a InlineInfo object.
  virtual void dump(raw_ostream &OS, const InlineInfo &II,
                    uint32_t Indent = 0) = 0;

  /// Dump a FileEntry object.
  virtual void dump(raw_ostream &OS, std::optional<FileEntry> FE) = 0;
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_GSYMREADER_H
