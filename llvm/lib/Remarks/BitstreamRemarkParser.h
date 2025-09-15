//===-- BitstreamRemarkParser.h - Parser for Bitstream remarks --*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the impementation of the Bitstream remark parser.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_REMARKS_BITSTREAM_REMARK_PARSER_H
#define LLVM_LIB_REMARKS_BITSTREAM_REMARK_PARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/Remarks/BitstreamRemarkContainer.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace llvm {
namespace remarks {

struct Remark;

/// Helper to parse a META_BLOCK for a bitstream remark container.
struct BitstreamMetaParserHelper {
  /// The Bitstream reader.
  BitstreamCursor &Stream;
  /// Reference to the storage for the block info.
  BitstreamBlockInfo &BlockInfo;
  /// The parsed content: depending on the container type, some fields might be
  /// empty.
  std::optional<uint64_t> ContainerVersion;
  std::optional<uint8_t> ContainerType;
  std::optional<StringRef> StrTabBuf;
  std::optional<StringRef> ExternalFilePath;
  std::optional<uint64_t> RemarkVersion;

  /// Continue parsing with \p Stream. \p Stream is expected to contain a
  /// ENTER_SUBBLOCK to the META_BLOCK at the current position.
  /// \p Stream is expected to have a BLOCKINFO_BLOCK set.
  BitstreamMetaParserHelper(BitstreamCursor &Stream,
                            BitstreamBlockInfo &BlockInfo);

  /// Parse the META_BLOCK and fill the available entries.
  /// This helper does not check for the validity of the fields.
  Error parse();
};

/// Helper to parse a REMARK_BLOCK for a bitstream remark container.
struct BitstreamRemarkParserHelper {
  /// The Bitstream reader.
  BitstreamCursor &Stream;
  /// The parsed content: depending on the remark, some fields might be empty.
  std::optional<uint8_t> Type;
  std::optional<uint64_t> RemarkNameIdx;
  std::optional<uint64_t> PassNameIdx;
  std::optional<uint64_t> FunctionNameIdx;
  std::optional<uint64_t> SourceFileNameIdx;
  std::optional<uint32_t> SourceLine;
  std::optional<uint32_t> SourceColumn;
  std::optional<uint64_t> Hotness;
  struct Argument {
    std::optional<uint64_t> KeyIdx;
    std::optional<uint64_t> ValueIdx;
    std::optional<uint64_t> SourceFileNameIdx;
    std::optional<uint32_t> SourceLine;
    std::optional<uint32_t> SourceColumn;
  };
  std::optional<ArrayRef<Argument>> Args;
  /// Avoid re-allocating a vector every time.
  SmallVector<Argument, 8> TmpArgs;

  /// Continue parsing with \p Stream. \p Stream is expected to contain a
  /// ENTER_SUBBLOCK to the REMARK_BLOCK at the current position.
  /// \p Stream is expected to have a BLOCKINFO_BLOCK set and to have already
  /// parsed the META_BLOCK.
  BitstreamRemarkParserHelper(BitstreamCursor &Stream);

  /// Parse the REMARK_BLOCK and fill the available entries.
  /// This helper does not check for the validity of the fields.
  Error parse();
};

/// Helper to parse any bitstream remark container.
struct BitstreamParserHelper {
  /// The Bitstream reader.
  BitstreamCursor Stream;
  /// The block info block.
  BitstreamBlockInfo BlockInfo;
  /// Start parsing at \p Buffer.
  BitstreamParserHelper(StringRef Buffer);
  /// Parse the magic number.
  Expected<std::array<char, 4>> parseMagic();
  /// Parse the block info block containing all the abbrevs.
  /// This needs to be called before calling any other parsing function.
  Error parseBlockInfoBlock();
  /// Return true if the next block is a META_BLOCK. This function does not move
  /// the cursor.
  Expected<bool> isMetaBlock();
  /// Return true if the next block is a REMARK_BLOCK. This function does not
  /// move the cursor.
  Expected<bool> isRemarkBlock();
  /// Return true if the parser reached the end of the stream.
  bool atEndOfStream() { return Stream.AtEndOfStream(); }
  /// Jump to the end of the stream, skipping everything.
  void skipToEnd() { return Stream.skipToEnd(); }
};

/// Parses and holds the state of the latest parsed remark.
struct BitstreamRemarkParser : public RemarkParser {
  /// The buffer to parse.
  BitstreamParserHelper ParserHelper;
  /// The string table used for parsing strings.
  std::optional<ParsedStringTable> StrTab;
  /// Temporary remark buffer used when the remarks are stored separately.
  std::unique_ptr<MemoryBuffer> TmpRemarkBuffer;
  /// The common metadata used to decide how to parse the buffer.
  /// This is filled when parsing the metadata block.
  uint64_t ContainerVersion = 0;
  uint64_t RemarkVersion = 0;
  BitstreamRemarkContainerType ContainerType =
      BitstreamRemarkContainerType::Standalone;
  /// Wether the parser is ready to parse remarks.
  bool ReadyToParseRemarks = false;

  /// Create a parser that expects to find a string table embedded in the
  /// stream.
  explicit BitstreamRemarkParser(StringRef Buf);

  Expected<std::unique_ptr<Remark>> next() override;

  static bool classof(const RemarkParser *P) {
    return P->ParserFormat == Format::Bitstream;
  }

  /// Parse and process the metadata of the buffer.
  Error parseMeta();

  /// Parse a Bitstream remark.
  Expected<std::unique_ptr<Remark>> parseRemark();

private:
  /// Helper functions.
  Error processCommonMeta(BitstreamMetaParserHelper &Helper);
  Error processStandaloneMeta(BitstreamMetaParserHelper &Helper);
  Error processSeparateRemarksFileMeta(BitstreamMetaParserHelper &Helper);
  Error processSeparateRemarksMetaMeta(BitstreamMetaParserHelper &Helper);
  Expected<std::unique_ptr<Remark>>
  processRemark(BitstreamRemarkParserHelper &Helper);
  Error processExternalFilePath(std::optional<StringRef> ExternalFilePath);
};

Expected<std::unique_ptr<BitstreamRemarkParser>> createBitstreamParserFromMeta(
    StringRef Buf,
    std::optional<StringRef> ExternalFilePrependPath = std::nullopt);

} // end namespace remarks
} // end namespace llvm

#endif /* LLVM_LIB_REMARKS_BITSTREAM_REMARK_PARSER_H */
