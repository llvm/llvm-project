//===-- llvm/Bitcode/BitcodeWriter.h - Bitcode writers ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to write LLVM bitcode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_BITCODEWRITER_H
#define LLVM_BITCODE_BITCODEWRITER_H

#include "llvm/IR/ModuleSummaryIndex.h"
#include <string>

namespace llvm {
  class BitstreamWriter;
  class Module;
  class raw_ostream;

  class BitcodeWriter {
    SmallVectorImpl<char> &Buffer;
    std::unique_ptr<BitstreamWriter> Stream;

   public:
    /// Create a BitcodeWriter that writes to Buffer.
    BitcodeWriter(SmallVectorImpl<char> &Buffer);

    ~BitcodeWriter();

    /// Write the specified module to the buffer specified at construction time.
    ///
    /// If \c ShouldPreserveUseListOrder, encode the use-list order for each \a
    /// Value in \c M.  These will be reconstructed exactly when \a M is
    /// deserialized.
    ///
    /// If \c Index is supplied, the bitcode will contain the summary index
    /// (currently for use in ThinLTO optimization).
    ///
    /// \p GenerateHash enables hashing the Module and including the hash in the
    /// bitcode (currently for use in ThinLTO incremental build).
    ///
    /// If \p ModHash is non-null, when GenerateHash is true, the resulting
    /// hash is written into ModHash. When GenerateHash is false, that value
    /// is used as the hash instead of computing from the generated bitcode.
    /// Can be used to produce the same module hash for a minimized bitcode
    /// used just for the thin link as in the regular full bitcode that will
    /// be used in the backend.
    void writeModule(const Module *M, bool ShouldPreserveUseListOrder = false,
                     const ModuleSummaryIndex *Index = nullptr,
                     bool GenerateHash = false, ModuleHash *ModHash = nullptr);
  };

  /// \brief Write the specified module to the specified raw output stream.
  ///
  /// For streams where it matters, the given stream should be in "binary"
  /// mode.
  ///
  /// If \c ShouldPreserveUseListOrder, encode the use-list order for each \a
  /// Value in \c M.  These will be reconstructed exactly when \a M is
  /// deserialized.
  ///
  /// If \c Index is supplied, the bitcode will contain the summary index
  /// (currently for use in ThinLTO optimization).
  ///
  /// \p GenerateHash enables hashing the Module and including the hash in the
  /// bitcode (currently for use in ThinLTO incremental build).
  ///
  /// If \p ModHash is non-null, when GenerateHash is true, the resulting
  /// hash is written into ModHash. When GenerateHash is false, that value
  /// is used as the hash instead of computing from the generated bitcode.
  /// Can be used to produce the same module hash for a minimized bitcode
  /// used just for the thin link as in the regular full bitcode that will
  /// be used in the backend.
  void WriteBitcodeToFile(const Module *M, raw_ostream &Out,
                          bool ShouldPreserveUseListOrder = false,
                          const ModuleSummaryIndex *Index = nullptr,
                          bool GenerateHash = false,
                          ModuleHash *ModHash = nullptr);

  /// Write the specified module summary index to the given raw output stream,
  /// where it will be written in a new bitcode block. This is used when
  /// writing the combined index file for ThinLTO. When writing a subset of the
  /// index for a distributed backend, provide the \p ModuleToSummariesForIndex
  /// map.
  void WriteIndexToFile(const ModuleSummaryIndex &Index, raw_ostream &Out,
                        const std::map<std::string, GVSummaryMapTy>
                            *ModuleToSummariesForIndex = nullptr);
} // End llvm namespace

#endif
