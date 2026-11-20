//===- BitstreamReader.h - Low-level bitstream reader interface -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitstreamReader class.  This class can be used to
// read an arbitrary bitstream, regardless of its contents.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITSTREAM_BITSTREAMREADER_H
#define LLVM_BITSTREAM_BITSTREAMREADER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitstream/BitCodes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

/// This class maintains the abbreviations read from a block info block.
class BitstreamBlockInfo {
public:
  /// This contains information emitted to BLOCKINFO_BLOCK blocks. These
  /// describe abbreviations that all blocks of the specified ID inherit.
  struct BlockInfo {
    unsigned BlockID = 0;
    std::vector<std::shared_ptr<BitCodeAbbrev>> Abbrevs;
    std::string Name;
    std::vector<std::pair<unsigned, std::string>> RecordNames;
  };

private:
  std::vector<BlockInfo> BlockInfoRecords;

public:
  /// If there is block info for the specified ID, return it, otherwise return
  /// null.
  const BlockInfo *getBlockInfo(unsigned BlockID) const {
    // Common case, the most recent entry matches BlockID.
    if (!BlockInfoRecords.empty() && BlockInfoRecords.back().BlockID == BlockID)
      return &BlockInfoRecords.back();

    for (const BlockInfo &BI : BlockInfoRecords)
      if (BI.BlockID == BlockID)
        return &BI;
    return nullptr;
  }

  BlockInfo &getOrCreateBlockInfo(unsigned BlockID) {
    if (const BlockInfo *BI = getBlockInfo(BlockID))
      return *const_cast<BlockInfo*>(BI);

    // Otherwise, add a new record.
    BlockInfoRecords.emplace_back();
    BlockInfoRecords.back().BlockID = BlockID;
    return BlockInfoRecords.back();
  }
};

/// This represents a position within a bitstream. There may be multiple
/// independent cursors reading within one bitstream, each maintaining their
/// own local state.
class SimpleBitstreamCursor {
  ArrayRef<uint8_t> BitcodeBytes;

  /// The number of the next bit to read.
  uint64_t BitNo = 0;

  /// Reads at bits before this can load a whole word from their first byte.
  uint64_t WordLoadEnd = 0;

public:
  /// The type of the values returned by Read. This is specifically and
  /// intentionally defined to follow the word size of the host machine for
  /// efficiency. We use word_t in places that are aware of this to make it
  /// perfectly explicit what is going on.
  using word_t = size_t;

private:
  static constexpr unsigned BitsInWord = sizeof(word_t) * 8;

public:
  SimpleBitstreamCursor() = default;
  explicit SimpleBitstreamCursor(ArrayRef<uint8_t> BitcodeBytes)
      : BitcodeBytes(BitcodeBytes),
        WordLoadEnd(BitcodeBytes.size() < sizeof(word_t)
                        ? 0
                        : (BitcodeBytes.size() - sizeof(word_t) + 1) * 8) {}
  explicit SimpleBitstreamCursor(StringRef BitcodeBytes)
      : SimpleBitstreamCursor(arrayRefFromStringRef(BitcodeBytes)) {}
  explicit SimpleBitstreamCursor(MemoryBufferRef BitcodeBytes)
      : SimpleBitstreamCursor(BitcodeBytes.getBuffer()) {}

  bool canSkipToPos(size_t pos) const {
    // pos can be skipped to if it is a valid address or one byte past the end.
    return pos <= BitcodeBytes.size();
  }

  bool AtEndOfStream() { return BitNo >= sizeInBits(); }

  /// Return the bit # of the bit we are reading.
  uint64_t GetCurrentBitNo() const { return BitNo; }

  // Return the byte # of the current bit.
  uint64_t getCurrentByteNo() const { return GetCurrentBitNo() / 8; }

  ArrayRef<uint8_t> getBitcodeBytes() const { return BitcodeBytes; }

  /// Reset the stream to the specified bit number.
  Error JumpToBit(uint64_t NewBitNo) {
    assert(canSkipToPos(NewBitNo / 8) && "Invalid location");
    if (NewBitNo > sizeInBits())
      return createStringError(std::errc::io_error,
                               "Unexpected end of file jumping to bit %" PRIu64
                               " of %" PRIu64 " bits",
                               NewBitNo, sizeInBits());
    BitNo = NewBitNo;
    return Error::success();
  }

  /// Get a pointer into the bitstream at the specified byte offset.
  const uint8_t *getPointerToByte(uint64_t ByteNo, uint64_t NumBytes) {
    return BitcodeBytes.data() + ByteNo;
  }

  /// Get a pointer into the bitstream at the specified bit offset.
  ///
  /// The bit offset must be on a byte boundary.
  const uint8_t *getPointerToBit(uint64_t BitNo, uint64_t NumBytes) {
    assert(!(BitNo % 8) && "Expected bit on byte boundary");
    return getPointerToByte(BitNo / 8, NumBytes);
  }

private:
  /// Return the word starting at the current bit, which must be before
  /// WordLoadEnd. Only the low BitsInWord - 7 bits are guaranteed to be valid.
  word_t loadWordAtBit() const {
    assert(BitNo < WordLoadEnd && "Word load past the end of the stream");
    return support::endian::read<word_t, llvm::endianness::little>(
               BitcodeBytes.data() + BitNo / 8) >>
           (BitNo % 8);
  }

public:
  Expected<word_t> Read(unsigned NumBits) {
    assert(NumBits && NumBits <= BitsInWord &&
           "Cannot return zero or more than BitsInWord bits!");

    // If the field is contained in a word starting at the current byte, load
    // that word directly.
    if (NumBits <= BitsInWord - 7 && BitNo < WordLoadEnd) {
      word_t R = loadWordAtBit() & (~word_t(0) >> (BitsInWord - NumBits));
      BitNo += NumBits;
      return R;
    }

    return readSlow(NumBits);
  }

  /// Read \p NumBits bits that don't fit in a word load from the current byte,
  /// or are near the end of the stream.
  LLVM_ABI Expected<word_t> readSlow(unsigned NumBits);

  Expected<uint32_t> ReadVBR(const unsigned NumBits) {
    Expected<unsigned> MaybeRead = Read(NumBits);
    if (!MaybeRead)
      return MaybeRead;
    uint32_t Piece = MaybeRead.get();

    assert(NumBits <= 32 && NumBits >= 1 && "Invalid NumBits value");
    const uint32_t MaskBitOrder = (NumBits - 1);
    const uint32_t Mask = 1UL << MaskBitOrder;

    if ((Piece & Mask) == 0)
      return Piece;

    uint32_t Result = 0;
    unsigned NextBit = 0;
    while (true) {
      Result |= (Piece & (Mask - 1)) << NextBit;

      if ((Piece & Mask) == 0)
        return Result;

      NextBit += NumBits-1;
      if (NextBit >= 32)
        return createStringError(std::errc::illegal_byte_sequence,
                                 "Unterminated VBR");

      MaybeRead = Read(NumBits);
      if (!MaybeRead)
        return MaybeRead;
      Piece = MaybeRead.get();
    }
  }

  // Read a VBR that may have a value up to 64-bits in size. The chunk size of
  // the VBR must still be <= 32 bits though.
  Expected<uint64_t> ReadVBR64(const unsigned NumBits) {
    Expected<uint64_t> MaybeRead = Read(NumBits);
    if (!MaybeRead)
      return MaybeRead;
    uint32_t Piece = MaybeRead.get();
    assert(NumBits <= 32 && NumBits >= 1 && "Invalid NumBits value");
    const uint32_t MaskBitOrder = (NumBits - 1);
    const uint32_t Mask = 1UL << MaskBitOrder;

    if ((Piece & Mask) == 0)
      return uint64_t(Piece);

    uint64_t Result = 0;
    unsigned NextBit = 0;
    while (true) {
      Result |= uint64_t(Piece & (Mask - 1)) << NextBit;

      if ((Piece & Mask) == 0)
        return Result;

      NextBit += NumBits-1;
      if (NextBit >= 64)
        return createStringError(std::errc::illegal_byte_sequence,
                                 "Unterminated VBR");

      MaybeRead = Read(NumBits);
      if (!MaybeRead)
        return MaybeRead;
      Piece = MaybeRead.get();
    }
  }

  void SkipToFourByteBoundary() {
    BitNo = std::min(alignTo(BitNo, 32), sizeInBits());
  }

  /// Return the size of the stream in bytes.
  size_t SizeInBytes() const { return BitcodeBytes.size(); }

  /// Return the size of the stream in bits.
  uint64_t sizeInBits() const { return uint64_t(BitcodeBytes.size()) * 8; }

  /// Skip to the end of the file.
  void skipToEnd() { BitNo = sizeInBits(); }

  /// Check whether a reservation of Size elements is plausible.
  bool isSizePlausible(size_t Size) const {
    // Don't allow reserving more elements than the number of bits, assuming
    // at least one bit is needed to encode an element.
    return Size < BitcodeBytes.size() * 8;
  }
};

/// When advancing through a bitstream cursor, each advance can discover a few
/// different kinds of entries:
struct BitstreamEntry {
  enum {
    Error,    // Malformed bitcode was found.
    EndBlock, // We've reached the end of the current block, (or the end of the
              // file, which is treated like a series of EndBlock records.
    SubBlock, // This is the start of a new subblock of a specific ID.
    Record    // This is a record with a specific AbbrevID.
  } Kind;

  unsigned ID;

  static BitstreamEntry getError() {
    BitstreamEntry E; E.Kind = Error; return E;
  }

  static BitstreamEntry getEndBlock() {
    BitstreamEntry E; E.Kind = EndBlock; return E;
  }

  static BitstreamEntry getSubBlock(unsigned ID) {
    BitstreamEntry E; E.Kind = SubBlock; E.ID = ID; return E;
  }

  static BitstreamEntry getRecord(unsigned AbbrevID) {
    BitstreamEntry E; E.Kind = Record; E.ID = AbbrevID; return E;
  }
};

/// This represents a position within a bitcode file, implemented on top of a
/// SimpleBitstreamCursor.
///
/// Unlike iterators, BitstreamCursors are heavy-weight objects that should not
/// be passed by value.
class BitstreamCursor : SimpleBitstreamCursor {
  // This is the declared size of code values used for the current block, in
  // bits.
  unsigned CurCodeSize = 2;

  /// Abbrevs installed at in this block.
  std::vector<std::shared_ptr<BitCodeAbbrev>> CurAbbrevs;

  struct Block {
    unsigned PrevCodeSize;
    std::vector<std::shared_ptr<BitCodeAbbrev>> PrevAbbrevs;

    explicit Block(unsigned PCS) : PrevCodeSize(PCS) {}
  };

  /// This tracks the codesize of parent blocks. Clang's AST files nest blocks
  /// at most two levels deep, LLVM IR bitcode three. Some clients (e.g. clang's
  /// ModuleFile) embed many cursors, so keep the inline storage small.
  SmallVector<Block, 2> BlockScope;

  BitstreamBlockInfo *BlockInfo = nullptr;

public:
  static const size_t MaxChunkSize = 32;

  BitstreamCursor() = default;
  explicit BitstreamCursor(ArrayRef<uint8_t> BitcodeBytes)
      : SimpleBitstreamCursor(BitcodeBytes) {}
  explicit BitstreamCursor(StringRef BitcodeBytes)
      : SimpleBitstreamCursor(BitcodeBytes) {}
  explicit BitstreamCursor(MemoryBufferRef BitcodeBytes)
      : SimpleBitstreamCursor(BitcodeBytes) {}

  using SimpleBitstreamCursor::AtEndOfStream;
  using SimpleBitstreamCursor::canSkipToPos;
  using SimpleBitstreamCursor::getBitcodeBytes;
  using SimpleBitstreamCursor::GetCurrentBitNo;
  using SimpleBitstreamCursor::getCurrentByteNo;
  using SimpleBitstreamCursor::getPointerToByte;
  using SimpleBitstreamCursor::JumpToBit;
  using SimpleBitstreamCursor::Read;
  using SimpleBitstreamCursor::ReadVBR;
  using SimpleBitstreamCursor::ReadVBR64;
  using SimpleBitstreamCursor::SizeInBytes;
  using SimpleBitstreamCursor::skipToEnd;

  /// Return the number of bits used to encode an abbrev #.
  unsigned getAbbrevIDWidth() const { return CurCodeSize; }

  /// Flags that modify the behavior of advance().
  enum {
    /// If this flag is used, the advance() method does not automatically pop
    /// the block scope when the end of a block is reached.
    AF_DontPopBlockAtEnd = 1,

    /// If this flag is used, abbrev entries are returned just like normal
    /// records.
    AF_DontAutoprocessAbbrevs = 2
  };

  /// Advance the current bitstream, returning the next entry in the stream.
  Expected<BitstreamEntry> advance(unsigned Flags = 0) {
    while (true) {
      if (AtEndOfStream())
        return BitstreamEntry::getError();

      Expected<unsigned> MaybeCode = ReadCode();
      if (!MaybeCode)
        return MaybeCode.takeError();
      unsigned Code = MaybeCode.get();

      if (Code == bitc::END_BLOCK) {
        // Pop the end of the block unless Flags tells us not to.
        if (!(Flags & AF_DontPopBlockAtEnd) && ReadBlockEnd())
          return BitstreamEntry::getError();
        return BitstreamEntry::getEndBlock();
      }

      if (Code == bitc::ENTER_SUBBLOCK) {
        if (Expected<unsigned> MaybeSubBlock = ReadSubBlockID())
          return BitstreamEntry::getSubBlock(MaybeSubBlock.get());
        else
          return MaybeSubBlock.takeError();
      }

      if (Code == bitc::DEFINE_ABBREV &&
          !(Flags & AF_DontAutoprocessAbbrevs)) {
        // We read and accumulate abbrev's, the client can't do anything with
        // them anyway.
        if (Error Err = ReadAbbrevRecord())
          return std::move(Err);
        continue;
      }

      return BitstreamEntry::getRecord(Code);
    }
  }

  /// This is a convenience function for clients that don't expect any
  /// subblocks. This just skips over them automatically.
  Expected<BitstreamEntry> advanceSkippingSubblocks(unsigned Flags = 0) {
    while (true) {
      // If we found a normal entry, return it.
      Expected<BitstreamEntry> MaybeEntry = advance(Flags);
      if (!MaybeEntry)
        return MaybeEntry;
      BitstreamEntry Entry = MaybeEntry.get();

      if (Entry.Kind != BitstreamEntry::SubBlock)
        return Entry;

      // If we found a sub-block, just skip over it and check the next entry.
      if (Error Err = SkipBlock())
        return std::move(Err);
    }
  }

  Expected<unsigned> ReadCode() { return Read(CurCodeSize); }

  // Block header:
  //    [ENTER_SUBBLOCK, blockid, newcodelen, <align4bytes>, blocklen]

  /// Having read the ENTER_SUBBLOCK code, read the BlockID for the block.
  Expected<unsigned> ReadSubBlockID() { return ReadVBR(bitc::BlockIDWidth); }

  /// Having read the ENTER_SUBBLOCK abbrevid and a BlockID, skip over the body
  /// of this block.
  Error SkipBlock() {
    // Read and ignore the codelen value.
    if (Expected<uint32_t> Res = ReadVBR(bitc::CodeLenWidth))
      ; // Since we are skipping this block, we don't care what code widths are
        // used inside of it.
    else
      return Res.takeError();

    SkipToFourByteBoundary();
    Expected<unsigned> MaybeNum = Read(bitc::BlockSizeWidth);
    if (!MaybeNum)
      return MaybeNum.takeError();
    size_t NumFourBytes = MaybeNum.get();

    // Check that the block wasn't partially defined, and that the offset isn't
    // bogus.
    size_t SkipTo = GetCurrentBitNo() + NumFourBytes * 4 * 8;
    if (AtEndOfStream())
      return createStringError(std::errc::illegal_byte_sequence,
                               "can't skip block: already at end of stream");
    if (!canSkipToPos(SkipTo / 8))
      return createStringError(std::errc::illegal_byte_sequence,
                               "can't skip to bit %zu from %" PRIu64, SkipTo,
                               GetCurrentBitNo());

    if (Error Res = JumpToBit(SkipTo))
      return Res;

    return Error::success();
  }

  /// Having read the ENTER_SUBBLOCK abbrevid, and enter the block.
  LLVM_ABI Error EnterSubBlock(unsigned BlockID, unsigned *NumWordsP = nullptr);

  bool ReadBlockEnd() {
    if (BlockScope.empty()) return true;

    // Block tail:
    //    [END_BLOCK, <align4bytes>]
    SkipToFourByteBoundary();

    popBlockScope();
    return false;
  }

private:
  void popBlockScope() {
    CurCodeSize = BlockScope.back().PrevCodeSize;

    CurAbbrevs = std::move(BlockScope.back().PrevAbbrevs);
    BlockScope.pop_back();
  }

  //===--------------------------------------------------------------------===//
  // Record Processing
  //===--------------------------------------------------------------------===//

public:
  /// Return the abbreviation for the specified AbbrevId.
  Expected<const BitCodeAbbrev *> getAbbrev(unsigned AbbrevID) {
    unsigned AbbrevNo = AbbrevID - bitc::FIRST_APPLICATION_ABBREV;
    if (AbbrevNo >= CurAbbrevs.size())
      return createStringError(
          std::errc::illegal_byte_sequence, "Invalid abbrev number");
    return CurAbbrevs[AbbrevNo].get();
  }

  /// Read the current record and discard it, returning the code for the record.
  LLVM_ABI Expected<unsigned> skipRecord(unsigned AbbrevID);

  LLVM_ABI Expected<unsigned> readRecord(unsigned AbbrevID,
                                         SmallVectorImpl<uint64_t> &Vals,
                                         StringRef *Blob = nullptr);

  //===--------------------------------------------------------------------===//
  // Abbrev Processing
  //===--------------------------------------------------------------------===//
  LLVM_ABI Error ReadAbbrevRecord();

  /// Read and return a block info block from the bitstream. If an error was
  /// encountered, return std::nullopt.
  ///
  /// \param ReadBlockInfoNames Whether to read block/record name information in
  /// the BlockInfo block. Only llvm-bcanalyzer uses this.
  LLVM_ABI Expected<std::optional<BitstreamBlockInfo>>
  ReadBlockInfoBlock(bool ReadBlockInfoNames = false);

  /// Set the block info to be used by this BitstreamCursor to interpret
  /// abbreviated records.
  void setBlockInfo(BitstreamBlockInfo *BI) { BlockInfo = BI; }
};

} // end llvm namespace

#endif // LLVM_BITSTREAM_BITSTREAMREADER_H
