//===- CodeGenDataReader.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading codegen data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CGDATA_CODEGENDATAREADER_H
#define LLVM_CGDATA_CODEGENDATAREADER_H

#include "llvm/CGData/CodeGenData.h"
#include "llvm/CGData/OutlinedHashTreeRecord.h"
#include "llvm/CGData/StableFunctionMapRecord.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace llvm {

class CodeGenDataReader {
  cgdata_error LastError = cgdata_error::success;
  std::string LastErrorMsg;

public:
  CodeGenDataReader() = default;
  virtual ~CodeGenDataReader() = default;

  /// Read the header.  Required before reading first record.
  virtual Error read() = 0;
  /// Return the codegen data version.
  virtual uint32_t getVersion() const = 0;
  /// Return the codegen data kind.
  virtual CGDataKind getDataKind() const = 0;
  /// Return true if the data has an outlined hash tree.
  virtual bool hasOutlinedHashTree() const = 0;
  /// Return true if the data has a stable function map.
  virtual bool hasStableFunctionMap() const = 0;
  /// Return the outlined hash tree that is released from the reader.
  std::unique_ptr<OutlinedHashTree> releaseOutlinedHashTree() {
    return std::move(HashTreeRecord.HashTree);
  }
  std::unique_ptr<StableFunctionMap> releaseStableFunctionMap() {
    return std::move(FunctionMapRecord.FunctionMap);
  }

  /// Factory method to create an appropriately typed reader for the given
  /// codegen data file path and file system.
  static Expected<std::unique_ptr<CodeGenDataReader>>
  create(const Twine &Path, vfs::FileSystem &FS);

  /// Factory method to create an appropriately typed reader for the given
  /// memory buffer.
  static Expected<std::unique_ptr<CodeGenDataReader>>
  create(std::unique_ptr<MemoryBuffer> Buffer);

  /// Extract the cgdata embedded in sections from the given object file and
  /// merge them into the GlobalOutlineRecord. This is a static helper that
  /// is used by `llvm-cgdata --merge` or ThinLTO's two-codegen rounds.
  /// Optionally, \p CombinedHash can be used to compuate the combined hash of
  /// the merged data.
  static Error
  mergeFromObjectFile(const object::ObjectFile *Obj,
                      OutlinedHashTreeRecord &GlobalOutlineRecord,
                      StableFunctionMapRecord &GlobalFunctionMapRecord,
                      stable_hash *CombinedHash = nullptr);

protected:
  /// The outlined hash tree that has been read. When it's released by
  /// releaseOutlinedHashTree(), it's no longer valid.
  OutlinedHashTreeRecord HashTreeRecord;

  /// The stable function map that has been read. When it's released by
  // releaseStableFunctionMap(), it's no longer valid.
  StableFunctionMapRecord FunctionMapRecord;

  /// Set the current error and return same.
  Error error(cgdata_error Err, const std::string &ErrMsg = "") {
    LastError = Err;
    LastErrorMsg = ErrMsg;
    if (Err == cgdata_error::success)
      return Error::success();
    return make_error<CGDataError>(Err, ErrMsg);
  }

  Error error(Error &&E) {
    handleAllErrors(std::move(E), [&](const CGDataError &IPE) {
      LastError = IPE.get();
      LastErrorMsg = IPE.getMessage();
    });
    return make_error<CGDataError>(LastError, LastErrorMsg);
  }

  /// Clear the current error and return a successful one.
  Error success() { return error(cgdata_error::success); }
};

class IndexedCodeGenDataReader : public CodeGenDataReader {
  /// The codegen data file contents.
  std::unique_ptr<MemoryBuffer> DataBuffer;
  /// The header
  IndexedCGData::Header Header;

public:
  IndexedCodeGenDataReader(std::unique_ptr<MemoryBuffer> DataBuffer)
      : DataBuffer(std::move(DataBuffer)) {}
  IndexedCodeGenDataReader(const IndexedCodeGenDataReader &) = delete;
  IndexedCodeGenDataReader &
  operator=(const IndexedCodeGenDataReader &) = delete;

  /// Return true if the given buffer is in binary codegen data format.
  static bool hasFormat(const MemoryBuffer &Buffer);
  /// Read the contents including the header.
  Error read() override;
  /// Return the codegen data version.
  uint32_t getVersion() const override { return Header.Version; }
  /// Return the codegen data kind.
  CGDataKind getDataKind() const override {
    return static_cast<CGDataKind>(Header.DataKind);
  }
  /// Return true if the header indicates the data has an outlined hash tree.
  /// This does not mean that the data is still available.
  bool hasOutlinedHashTree() const override {
    return Header.DataKind &
           static_cast<uint32_t>(CGDataKind::FunctionOutlinedHashTree);
  }
  /// Return true if the header indicates the data has a stable function map.
  bool hasStableFunctionMap() const override {
    return Header.DataKind &
           static_cast<uint32_t>(CGDataKind::StableFunctionMergingMap);
  }
};

/// This format is a simple text format that's suitable for test data.
/// The header is a custom format starting with `:` per line to indicate which
/// codegen data is recorded. `#` is used to indicate a comment.
/// The subsequent data is a YAML format per each codegen data in order.
/// Currently, it only has a function outlined hash tree.
class TextCodeGenDataReader : public CodeGenDataReader {
  /// The codegen data file contents.
  std::unique_ptr<MemoryBuffer> DataBuffer;
  /// Iterator over the profile data.
  line_iterator Line;
  /// Describe the kind of the codegen data.
  CGDataKind DataKind = CGDataKind::Unknown;

public:
  TextCodeGenDataReader(std::unique_ptr<MemoryBuffer> DataBuffer_)
      : DataBuffer(std::move(DataBuffer_)), Line(*DataBuffer, true, '#') {}
  TextCodeGenDataReader(const TextCodeGenDataReader &) = delete;
  TextCodeGenDataReader &operator=(const TextCodeGenDataReader &) = delete;

  /// Return true if the given buffer is in text codegen data format.
  static bool hasFormat(const MemoryBuffer &Buffer);
  /// Read the contents including the header.
  Error read() override;
  /// Text format does not have version, so return 0.
  uint32_t getVersion() const override { return 0; }
  /// Return the codegen data kind.
  CGDataKind getDataKind() const override { return DataKind; }
  /// Return true if the header indicates the data has an outlined hash tree.
  /// This does not mean that the data is still available.
  bool hasOutlinedHashTree() const override {
    return static_cast<uint32_t>(DataKind) &
           static_cast<uint32_t>(CGDataKind::FunctionOutlinedHashTree);
  }
  /// Return true if the header indicates the data has a stable function map.
  /// This does not mean that the data is still available.
  bool hasStableFunctionMap() const override {
    return static_cast<uint32_t>(DataKind) &
           static_cast<uint32_t>(CGDataKind::StableFunctionMergingMap);
  }
};

} // end namespace llvm

#endif // LLVM_CGDATA_CODEGENDATAREADER_H
