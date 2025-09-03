//===-- BitstreamRemarkSerializer.h - Bitstream serializer ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides an implementation of the serializer using the LLVM
// Bitstream format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_REMARKS_BITSTREAMREMARKSERIALIZER_H
#define LLVM_REMARKS_BITSTREAMREMARKSERIALIZER_H

#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/Remarks/BitstreamRemarkContainer.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include <optional>

namespace llvm {
namespace remarks {

struct Remarks;

/// Serialize the remarks to LLVM bitstream.
/// This class provides ways to emit remarks in the LLVM bitstream format and
/// its associated metadata.
struct BitstreamRemarkSerializerHelper {
  /// Buffer used to construct records and pass to the bitstream writer.
  SmallVector<uint64_t, 64> R;
  /// The Bitstream writer.
  BitstreamWriter Bitstream;
  /// The type of the container we are serializing.
  BitstreamRemarkContainerType ContainerType;

  /// Abbrev IDs initialized in the block info block.
  /// Note: depending on the container type, some IDs might be uninitialized.
  /// Warning: When adding more abbrev IDs, make sure to update the
  /// BlockCodeSize (in the call to EnterSubblock).
  uint64_t RecordMetaContainerInfoAbbrevID = 0;
  uint64_t RecordMetaRemarkVersionAbbrevID = 0;
  uint64_t RecordMetaStrTabAbbrevID = 0;
  uint64_t RecordMetaExternalFileAbbrevID = 0;
  uint64_t RecordRemarkHeaderAbbrevID = 0;
  uint64_t RecordRemarkDebugLocAbbrevID = 0;
  uint64_t RecordRemarkHotnessAbbrevID = 0;
  uint64_t RecordRemarkArgWithDebugLocAbbrevID = 0;
  uint64_t RecordRemarkArgWithoutDebugLocAbbrevID = 0;

  BitstreamRemarkSerializerHelper(BitstreamRemarkContainerType ContainerType,
                                  raw_ostream &OS);

  // Disable copy and move: Bitstream points to Encoded, which needs special
  // handling during copy/move, but moving the vectors is probably useless
  // anyway.
  BitstreamRemarkSerializerHelper(const BitstreamRemarkSerializerHelper &) =
      delete;
  BitstreamRemarkSerializerHelper &
  operator=(const BitstreamRemarkSerializerHelper &) = delete;
  BitstreamRemarkSerializerHelper(BitstreamRemarkSerializerHelper &&) = delete;
  BitstreamRemarkSerializerHelper &
  operator=(BitstreamRemarkSerializerHelper &&) = delete;

  /// Set up the necessary block info entries according to the container type.
  void setupBlockInfo();

  /// Set up the block info for the metadata block.
  void setupMetaBlockInfo();
  /// The remark version in the metadata block.
  void setupMetaRemarkVersion();
  void emitMetaRemarkVersion(uint64_t RemarkVersion);
  /// The strtab in the metadata block.
  void setupMetaStrTab();
  void emitMetaStrTab(const StringTable &StrTab);
  /// The external file in the metadata block.
  void setupMetaExternalFile();
  void emitMetaExternalFile(StringRef Filename);

  /// The block info for the remarks block.
  void setupRemarkBlockInfo();

  /// Emit the main metadata at the beginning of the file
  void emitMetaBlock(std::optional<StringRef> Filename = std::nullopt);

  /// Emit the remaining metadata at the end of the file. Here we emit metadata
  /// that is only known once all remarks were emitted.
  void emitLateMetaBlock(const StringTable &StrTab);

  /// Emit a remark block. The string table is required.
  void emitRemark(const Remark &Remark, StringTable &StrTab);
};

/// Implementation of the remark serializer using LLVM bitstream.
struct BitstreamRemarkSerializer : public RemarkSerializer {
  /// The file should contain:
  /// 1) The block info block that describes how to read the blocks.
  /// 2) The metadata block that contains various information about the remarks
  ///    in the file.
  /// 3) A number of remark blocks.
  /// 4) Another metadata block for metadata that is only finalized once all
  ///    remarks were emitted (e.g. StrTab)

  /// The helper to emit bitstream. This is nullopt when the Serializer has not
  /// been setup yet.
  std::optional<BitstreamRemarkSerializerHelper> Helper;

  /// Construct a serializer that will create its own string table.
  BitstreamRemarkSerializer(raw_ostream &OS);
  /// Construct a serializer with a pre-filled string table.
  BitstreamRemarkSerializer(raw_ostream &OS, StringTable StrTab);

  ~BitstreamRemarkSerializer() override;

  /// Emit a remark to the stream. This also emits the metadata associated to
  /// the remarks. This writes the serialized output to the provided stream.
  void emit(const Remark &Remark) override;

  /// Finalize emission of remarks. This emits the late metadata block and
  /// flushes internal buffers. It is safe to call this function multiple times,
  /// and it is automatically executed on destruction of the Serializer.
  void finalize() override;

  /// The metadata serializer associated to this remark serializer. Based on the
  /// container type of the current serializer, the container type of the
  /// metadata serializer will change.
  std::unique_ptr<MetaSerializer>
  metaSerializer(raw_ostream &OS, StringRef ExternalFilename) override;

  static bool classof(const RemarkSerializer *S) {
    return S->SerializerFormat == Format::Bitstream;
  }

private:
  void setup();
};

/// Serializer of metadata for bitstream remarks.
struct BitstreamMetaSerializer : public MetaSerializer {
  std::optional<BitstreamRemarkSerializerHelper> Helper;

  StringRef ExternalFilename;

  /// Create a new meta serializer based on \p ContainerType.
  BitstreamMetaSerializer(raw_ostream &OS,
                          BitstreamRemarkContainerType ContainerType,
                          StringRef ExternalFilename)
      : MetaSerializer(OS), ExternalFilename(ExternalFilename) {
    Helper.emplace(ContainerType, OS);
  }

  void emit() override;
};

} // end namespace remarks
} // end namespace llvm

#endif // LLVM_REMARKS_BITSTREAMREMARKSERIALIZER_H
