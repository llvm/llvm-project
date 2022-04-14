//===- Bitcode/Writer/DXILBitcodeWriter.cpp - DXIL Bitcode Writer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bitcode writer implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llvm {

class BitstreamWriter;
class Module;
class raw_ostream;

namespace dxil {

class BitcodeWriter {
  SmallVectorImpl<char> &Buffer;
  std::unique_ptr<BitstreamWriter> Stream;

  StringTableBuilder StrtabBuilder{StringTableBuilder::RAW};

  // Owns any strings created by the irsymtab writer until we create the
  // string table.
  BumpPtrAllocator Alloc;

  bool WroteStrtab = false, WroteSymtab = false;

  void writeBlob(unsigned Block, unsigned Record, StringRef Blob);

  std::vector<Module *> Mods;

public:
  /// Create a BitcodeWriter that writes to Buffer.
  BitcodeWriter(SmallVectorImpl<char> &Buffer, raw_fd_stream *FS = nullptr);

  ~BitcodeWriter();

  /// Attempt to write a symbol table to the bitcode file. This must be called
  /// at most once after all modules have been written.
  ///
  /// A reader does not require a symbol table to interpret a bitcode file;
  /// the symbol table is needed only to improve link-time performance. So
  /// this function may decide not to write a symbol table. It may so decide
  /// if, for example, the target is unregistered or the IR is malformed.
  void writeSymtab();

  /// Write the bitcode file's string table. This must be called exactly once
  /// after all modules and the optional symbol table have been written.
  void writeStrtab();

  /// Copy the string table for another module into this bitcode file. This
  /// should be called after copying the module itself into the bitcode file.
  void copyStrtab(StringRef Strtab);

  /// Write the specified module to the buffer specified at construction time.
  void writeModule(const Module &M);
};

/// Write the specified module to the specified raw output stream.
///
/// For streams where it matters, the given stream should be in "binary"
/// mode.
void WriteDXILToFile(const Module &M, raw_ostream &Out);

} // namespace dxil

} // namespace llvm
