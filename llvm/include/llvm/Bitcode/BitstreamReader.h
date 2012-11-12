//===- BitstreamReader.h - Low-level bitstream reader interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitstreamReader class.  This class can be used to
// read an arbitrary bitstream, regardless of its contents.
//
//===----------------------------------------------------------------------===//

#ifndef BITSTREAM_READER_H
#define BITSTREAM_READER_H

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Bitcode/BitCodes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/StreamableMemoryObject.h"
#include <climits>
#include <string>
#include <vector>

namespace llvm {

  class Deserializer;

class BitstreamReader {
public:
  /// BlockInfo - This contains information emitted to BLOCKINFO_BLOCK blocks.
  /// These describe abbreviations that all blocks of the specified ID inherit.
  struct BlockInfo {
    unsigned BlockID;
    std::vector<BitCodeAbbrev*> Abbrevs;
    std::string Name;
    
    std::vector<std::pair<unsigned, std::string> > RecordNames;
  };
private:
  OwningPtr<StreamableMemoryObject> BitcodeBytes;
  
  std::vector<BlockInfo> BlockInfoRecords;

  /// IgnoreBlockInfoNames - This is set to true if we don't care about the
  /// block/record name information in the BlockInfo block. Only llvm-bcanalyzer
  /// uses this.
  bool IgnoreBlockInfoNames;

  BitstreamReader(const BitstreamReader&) LLVM_DELETED_FUNCTION;
  void operator=(const BitstreamReader&) LLVM_DELETED_FUNCTION;
public:
  BitstreamReader() : IgnoreBlockInfoNames(true) {
  }

  BitstreamReader(const unsigned char *Start, const unsigned char *End) {
    IgnoreBlockInfoNames = true;
    init(Start, End);
  }

  BitstreamReader(StreamableMemoryObject *bytes) {
    BitcodeBytes.reset(bytes);
  }

  void init(const unsigned char *Start, const unsigned char *End) {
    assert(((End-Start) & 3) == 0 &&"Bitcode stream not a multiple of 4 bytes");
    BitcodeBytes.reset(getNonStreamedMemoryObject(Start, End));
  }

  StreamableMemoryObject &getBitcodeBytes() { return *BitcodeBytes; }

  ~BitstreamReader() {
    // Free the BlockInfoRecords.
    while (!BlockInfoRecords.empty()) {
      BlockInfo &Info = BlockInfoRecords.back();
      // Free blockinfo abbrev info.
      for (unsigned i = 0, e = static_cast<unsigned>(Info.Abbrevs.size());
           i != e; ++i)
        Info.Abbrevs[i]->dropRef();
      BlockInfoRecords.pop_back();
    }
  }

  /// CollectBlockInfoNames - This is called by clients that want block/record
  /// name information.
  void CollectBlockInfoNames() { IgnoreBlockInfoNames = false; }
  bool isIgnoringBlockInfoNames() { return IgnoreBlockInfoNames; }
  
  //===--------------------------------------------------------------------===//
  // Block Manipulation
  //===--------------------------------------------------------------------===//

  /// hasBlockInfoRecords - Return true if we've already read and processed the
  /// block info block for this Bitstream.  We only process it for the first
  /// cursor that walks over it.
  bool hasBlockInfoRecords() const { return !BlockInfoRecords.empty(); }
  
  /// getBlockInfo - If there is block info for the specified ID, return it,
  /// otherwise return null.
  const BlockInfo *getBlockInfo(unsigned BlockID) const {
    // Common case, the most recent entry matches BlockID.
    if (!BlockInfoRecords.empty() && BlockInfoRecords.back().BlockID == BlockID)
      return &BlockInfoRecords.back();

    for (unsigned i = 0, e = static_cast<unsigned>(BlockInfoRecords.size());
         i != e; ++i)
      if (BlockInfoRecords[i].BlockID == BlockID)
        return &BlockInfoRecords[i];
    return 0;
  }

  BlockInfo &getOrCreateBlockInfo(unsigned BlockID) {
    if (const BlockInfo *BI = getBlockInfo(BlockID))
      return *const_cast<BlockInfo*>(BI);

    // Otherwise, add a new record.
    BlockInfoRecords.push_back(BlockInfo());
    BlockInfoRecords.back().BlockID = BlockID;
    return BlockInfoRecords.back();
  }

};

class BitstreamCursor {
  friend class Deserializer;
  BitstreamReader *BitStream;
  size_t NextChar;
  
  /// CurWord - This is the current data we have pulled from the stream but have
  /// not returned to the client.
  uint32_t CurWord;
  
  /// BitsInCurWord - This is the number of bits in CurWord that are valid. This
  /// is always from [0...31] inclusive.
  unsigned BitsInCurWord;
  
  // CurCodeSize - This is the declared size of code values used for the current
  // block, in bits.
  unsigned CurCodeSize;
  
  /// CurAbbrevs - Abbrevs installed at in this block.
  std::vector<BitCodeAbbrev*> CurAbbrevs;
  
  struct Block {
    unsigned PrevCodeSize;
    std::vector<BitCodeAbbrev*> PrevAbbrevs;
    explicit Block(unsigned PCS) : PrevCodeSize(PCS) {}
  };
  
  /// BlockScope - This tracks the codesize of parent blocks.
  SmallVector<Block, 8> BlockScope;
  
public:
  BitstreamCursor() : BitStream(0), NextChar(0) {
  }
  BitstreamCursor(const BitstreamCursor &RHS) : BitStream(0), NextChar(0) {
    operator=(RHS);
  }
  
  explicit BitstreamCursor(BitstreamReader &R) : BitStream(&R) {
    NextChar = 0;
    CurWord = 0;
    BitsInCurWord = 0;
    CurCodeSize = 2;
  }
  
  void init(BitstreamReader &R) {
    freeState();
    
    BitStream = &R;
    NextChar = 0;
    CurWord = 0;
    BitsInCurWord = 0;
    CurCodeSize = 2;
  }
  
  ~BitstreamCursor() {
    freeState();
  }
  
  void operator=(const BitstreamCursor &RHS) {
    freeState();
    
    BitStream = RHS.BitStream;
    NextChar = RHS.NextChar;
    CurWord = RHS.CurWord;
    BitsInCurWord = RHS.BitsInCurWord;
    CurCodeSize = RHS.CurCodeSize;
    
    // Copy abbreviations, and bump ref counts.
    CurAbbrevs = RHS.CurAbbrevs;
    for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
         i != e; ++i)
      CurAbbrevs[i]->addRef();
    
    // Copy block scope and bump ref counts.
    BlockScope = RHS.BlockScope;
    for (unsigned S = 0, e = static_cast<unsigned>(BlockScope.size());
         S != e; ++S) {
      std::vector<BitCodeAbbrev*> &Abbrevs = BlockScope[S].PrevAbbrevs;
      for (unsigned i = 0, e = static_cast<unsigned>(Abbrevs.size());
           i != e; ++i)
        Abbrevs[i]->addRef();
    }
  }
  
  void freeState() {
    // Free all the Abbrevs.
    for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
         i != e; ++i)
      CurAbbrevs[i]->dropRef();
    CurAbbrevs.clear();
    
    // Free all the Abbrevs in the block scope.
    for (unsigned S = 0, e = static_cast<unsigned>(BlockScope.size());
         S != e; ++S) {
      std::vector<BitCodeAbbrev*> &Abbrevs = BlockScope[S].PrevAbbrevs;
      for (unsigned i = 0, e = static_cast<unsigned>(Abbrevs.size());
           i != e; ++i)
        Abbrevs[i]->dropRef();
    }
    BlockScope.clear();
  }
  
  /// GetAbbrevIDWidth - Return the number of bits used to encode an abbrev #.
  unsigned GetAbbrevIDWidth() const { return CurCodeSize; }
  
  bool isEndPos(size_t pos) {
    return BitStream->getBitcodeBytes().isObjectEnd(static_cast<uint64_t>(pos));
  }

  bool canSkipToPos(size_t pos) const {
    // pos can be skipped to if it is a valid address or one byte past the end.
    return pos == 0 || BitStream->getBitcodeBytes().isValidAddress(
        static_cast<uint64_t>(pos - 1));
  }

  unsigned char getByte(size_t pos) {
    uint8_t byte = -1;
    BitStream->getBitcodeBytes().readByte(pos, &byte);
    return byte;
  }

  uint32_t getWord(size_t pos) {
    uint8_t buf[sizeof(uint32_t)];
    memset(buf, 0xFF, sizeof(buf));
    BitStream->getBitcodeBytes().readBytes(pos,
                                           sizeof(buf),
                                           buf,
                                           NULL);
    return *reinterpret_cast<support::ulittle32_t *>(buf);
  }

  bool AtEndOfStream() {
    return isEndPos(NextChar) && BitsInCurWord == 0;
  }
  
  /// GetCurrentBitNo - Return the bit # of the bit we are reading.
  uint64_t GetCurrentBitNo() const {
    return NextChar*CHAR_BIT - BitsInCurWord;
  }
  
  BitstreamReader *getBitStreamReader() {
    return BitStream;
  }
  const BitstreamReader *getBitStreamReader() const {
    return BitStream;
  }
  
  
  /// JumpToBit - Reset the stream to the specified bit number.
  void JumpToBit(uint64_t BitNo) {
    uintptr_t ByteNo = uintptr_t(BitNo/8) & ~3;
    uintptr_t WordBitNo = uintptr_t(BitNo) & 31;
    assert(canSkipToPos(ByteNo) && "Invalid location");
    
    // Move the cursor to the right word.
    NextChar = ByteNo;
    BitsInCurWord = 0;
    CurWord = 0;
    
    // Skip over any bits that are already consumed.
    if (WordBitNo)
      Read(static_cast<unsigned>(WordBitNo));
  }
  
  
  uint32_t Read(unsigned NumBits) {
    assert(NumBits <= 32 && "Cannot return more than 32 bits!");
    // If the field is fully contained by CurWord, return it quickly.
    if (BitsInCurWord >= NumBits) {
      uint32_t R = CurWord & ((1U << NumBits)-1);
      CurWord >>= NumBits;
      BitsInCurWord -= NumBits;
      return R;
    }

    // If we run out of data, stop at the end of the stream.
    if (isEndPos(NextChar)) {
      CurWord = 0;
      BitsInCurWord = 0;
      return 0;
    }

    unsigned R = CurWord;

    // Read the next word from the stream.
    CurWord = getWord(NextChar);
    NextChar += 4;

    // Extract NumBits-BitsInCurWord from what we just read.
    unsigned BitsLeft = NumBits-BitsInCurWord;

    // Be careful here, BitsLeft is in the range [1..32] inclusive.
    R |= (CurWord & (~0U >> (32-BitsLeft))) << BitsInCurWord;

    // BitsLeft bits have just been used up from CurWord.
    if (BitsLeft != 32)
      CurWord >>= BitsLeft;
    else
      CurWord = 0;
    BitsInCurWord = 32-BitsLeft;
    return R;
  }

  uint64_t Read64(unsigned NumBits) {
    if (NumBits <= 32) return Read(NumBits);

    uint64_t V = Read(32);
    return V | (uint64_t)Read(NumBits-32) << 32;
  }

  uint32_t ReadVBR(unsigned NumBits) {
    uint32_t Piece = Read(NumBits);
    if ((Piece & (1U << (NumBits-1))) == 0)
      return Piece;

    uint32_t Result = 0;
    unsigned NextBit = 0;
    while (1) {
      Result |= (Piece & ((1U << (NumBits-1))-1)) << NextBit;

      if ((Piece & (1U << (NumBits-1))) == 0)
        return Result;

      NextBit += NumBits-1;
      Piece = Read(NumBits);
    }
  }

  // ReadVBR64 - Read a VBR that may have a value up to 64-bits in size.  The
  // chunk size of the VBR must still be <= 32 bits though.
  uint64_t ReadVBR64(unsigned NumBits) {
    uint32_t Piece = Read(NumBits);
    if ((Piece & (1U << (NumBits-1))) == 0)
      return uint64_t(Piece);

    uint64_t Result = 0;
    unsigned NextBit = 0;
    while (1) {
      Result |= uint64_t(Piece & ((1U << (NumBits-1))-1)) << NextBit;

      if ((Piece & (1U << (NumBits-1))) == 0)
        return Result;

      NextBit += NumBits-1;
      Piece = Read(NumBits);
    }
  }

  void SkipToWord() {
    BitsInCurWord = 0;
    CurWord = 0;
  }

  unsigned ReadCode() {
    return Read(CurCodeSize);
  }


  // Block header:
  //    [ENTER_SUBBLOCK, blockid, newcodelen, <align4bytes>, blocklen]

  /// ReadSubBlockID - Having read the ENTER_SUBBLOCK code, read the BlockID for
  /// the block.
  unsigned ReadSubBlockID() {
    return ReadVBR(bitc::BlockIDWidth);
  }

  /// SkipBlock - Having read the ENTER_SUBBLOCK abbrevid and a BlockID, skip
  /// over the body of this block.  If the block record is malformed, return
  /// true.
  bool SkipBlock() {
    // Read and ignore the codelen value.  Since we are skipping this block, we
    // don't care what code widths are used inside of it.
    ReadVBR(bitc::CodeLenWidth);
    SkipToWord();
    unsigned NumWords = Read(bitc::BlockSizeWidth);

    // Check that the block wasn't partially defined, and that the offset isn't
    // bogus.
    size_t SkipTo = NextChar + NumWords*4;
    if (AtEndOfStream() || !canSkipToPos(SkipTo))
      return true;

    NextChar = SkipTo;
    return false;
  }

  /// EnterSubBlock - Having read the ENTER_SUBBLOCK abbrevid, enter
  /// the block, and return true if the block has an error.
  bool EnterSubBlock(unsigned BlockID, unsigned *NumWordsP = 0) {
    // Save the current block's state on BlockScope.
    BlockScope.push_back(Block(CurCodeSize));
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);

    // Add the abbrevs specific to this block to the CurAbbrevs list.
    if (const BitstreamReader::BlockInfo *Info =
          BitStream->getBlockInfo(BlockID)) {
      for (unsigned i = 0, e = static_cast<unsigned>(Info->Abbrevs.size());
           i != e; ++i) {
        CurAbbrevs.push_back(Info->Abbrevs[i]);
        CurAbbrevs.back()->addRef();
      }
    }

    // Get the codesize of this block.
    CurCodeSize = ReadVBR(bitc::CodeLenWidth);
    SkipToWord();
    unsigned NumWords = Read(bitc::BlockSizeWidth);
    if (NumWordsP) *NumWordsP = NumWords;

    // Validate that this block is sane.
    if (CurCodeSize == 0 || AtEndOfStream())
      return true;

    return false;
  }

  bool ReadBlockEnd() {
    if (BlockScope.empty()) return true;

    // Block tail:
    //    [END_BLOCK, <align4bytes>]
    SkipToWord();

    PopBlockScope();
    return false;
  }

private:
  void PopBlockScope() {
    CurCodeSize = BlockScope.back().PrevCodeSize;

    // Delete abbrevs from popped scope.
    for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
         i != e; ++i)
      CurAbbrevs[i]->dropRef();

    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
    BlockScope.pop_back();
  }

 //===--------------------------------------------------------------------===//
  // Record Processing
  //===--------------------------------------------------------------------===//

private:
  void ReadAbbreviatedLiteral(const BitCodeAbbrevOp &Op,
                              SmallVectorImpl<uint64_t> &Vals) {
    assert(Op.isLiteral() && "Not a literal");
    // If the abbrev specifies the literal value to use, use it.
    Vals.push_back(Op.getLiteralValue());
  }
  
  void ReadAbbreviatedField(const BitCodeAbbrevOp &Op,
                            SmallVectorImpl<uint64_t> &Vals) {
    assert(!Op.isLiteral() && "Use ReadAbbreviatedLiteral for literals!");

    // Decode the value as we are commanded.
    switch (Op.getEncoding()) {
    default: llvm_unreachable("Unknown encoding!");
    case BitCodeAbbrevOp::Fixed:
      Vals.push_back(Read((unsigned)Op.getEncodingData()));
      break;
    case BitCodeAbbrevOp::VBR:
      Vals.push_back(ReadVBR64((unsigned)Op.getEncodingData()));
      break;
    case BitCodeAbbrevOp::Char6:
      Vals.push_back(BitCodeAbbrevOp::DecodeChar6(Read(6)));
      break;
    }
  }
public:

  /// getAbbrev - Return the abbreviation for the specified AbbrevId. 
  const BitCodeAbbrev *getAbbrev(unsigned AbbrevID) {
    unsigned AbbrevNo = AbbrevID-bitc::FIRST_APPLICATION_ABBREV;
    assert(AbbrevNo < CurAbbrevs.size() && "Invalid abbrev #!");
    return CurAbbrevs[AbbrevNo];
  }
  
  unsigned ReadRecord(unsigned AbbrevID, SmallVectorImpl<uint64_t> &Vals,
                      const char **BlobStart = 0, unsigned *BlobLen = 0) {
    if (AbbrevID == bitc::UNABBREV_RECORD) {
      unsigned Code = ReadVBR(6);
      unsigned NumElts = ReadVBR(6);
      for (unsigned i = 0; i != NumElts; ++i)
        Vals.push_back(ReadVBR64(6));
      return Code;
    }

    const BitCodeAbbrev *Abbv = getAbbrev(AbbrevID);

    for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
      const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
      if (Op.isLiteral()) {
        ReadAbbreviatedLiteral(Op, Vals); 
      } else if (Op.getEncoding() == BitCodeAbbrevOp::Array) {
        // Array case.  Read the number of elements as a vbr6.
        unsigned NumElts = ReadVBR(6);

        // Get the element encoding.
        assert(i+2 == e && "array op not second to last?");
        const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);

        // Read all the elements.
        for (; NumElts; --NumElts)
          ReadAbbreviatedField(EltEnc, Vals);
      } else if (Op.getEncoding() == BitCodeAbbrevOp::Blob) {
        // Blob case.  Read the number of bytes as a vbr6.
        unsigned NumElts = ReadVBR(6);
        SkipToWord();  // 32-bit alignment

        // Figure out where the end of this blob will be including tail padding.
        size_t NewEnd = NextChar+((NumElts+3)&~3);
        
        // If this would read off the end of the bitcode file, just set the
        // record to empty and return.
        if (!canSkipToPos(NewEnd)) {
          Vals.append(NumElts, 0);
          NextChar = BitStream->getBitcodeBytes().getExtent();
          break;
        }
        
        // Otherwise, read the number of bytes.  If we can return a reference to
        // the data, do so to avoid copying it.
        if (BlobStart) {
          *BlobStart = (const char*)BitStream->getBitcodeBytes().getPointer(
              NextChar, NumElts);
          *BlobLen = NumElts;
        } else {
          for (; NumElts; ++NextChar, --NumElts)
            Vals.push_back(getByte(NextChar));
        }
        // Skip over tail padding.
        NextChar = NewEnd;
      } else {
        ReadAbbreviatedField(Op, Vals);
      }
    }

    unsigned Code = (unsigned)Vals[0];
    Vals.erase(Vals.begin());
    return Code;
  }

  unsigned ReadRecord(unsigned AbbrevID, SmallVectorImpl<uint64_t> &Vals,
                      const char *&BlobStart, unsigned &BlobLen) {
    return ReadRecord(AbbrevID, Vals, &BlobStart, &BlobLen);
  }

  
  //===--------------------------------------------------------------------===//
  // Abbrev Processing
  //===--------------------------------------------------------------------===//

  void ReadAbbrevRecord() {
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    unsigned NumOpInfo = ReadVBR(5);
    for (unsigned i = 0; i != NumOpInfo; ++i) {
      bool IsLiteral = Read(1) ? true : false;
      if (IsLiteral) {
        Abbv->Add(BitCodeAbbrevOp(ReadVBR64(8)));
        continue;
      }

      BitCodeAbbrevOp::Encoding E = (BitCodeAbbrevOp::Encoding)Read(3);
      if (BitCodeAbbrevOp::hasEncodingData(E))
        Abbv->Add(BitCodeAbbrevOp(E, ReadVBR64(5)));
      else
        Abbv->Add(BitCodeAbbrevOp(E));
    }
    CurAbbrevs.push_back(Abbv);
  }
  
public:

  bool ReadBlockInfoBlock() {
    // If this is the second stream to get to the block info block, skip it.
    if (BitStream->hasBlockInfoRecords())
      return SkipBlock();
    
    if (EnterSubBlock(bitc::BLOCKINFO_BLOCK_ID)) return true;

    SmallVector<uint64_t, 64> Record;
    BitstreamReader::BlockInfo *CurBlockInfo = 0;

    // Read all the records for this module.
    while (1) {
      unsigned Code = ReadCode();
      if (Code == bitc::END_BLOCK)
        return ReadBlockEnd();
      if (Code == bitc::ENTER_SUBBLOCK) {
        ReadSubBlockID();
        if (SkipBlock()) return true;
        continue;
      }

      // Read abbrev records, associate them with CurBID.
      if (Code == bitc::DEFINE_ABBREV) {
        if (!CurBlockInfo) return true;
        ReadAbbrevRecord();

        // ReadAbbrevRecord installs the abbrev in CurAbbrevs.  Move it to the
        // appropriate BlockInfo.
        BitCodeAbbrev *Abbv = CurAbbrevs.back();
        CurAbbrevs.pop_back();
        CurBlockInfo->Abbrevs.push_back(Abbv);
        continue;
      }

      // Read a record.
      Record.clear();
      switch (ReadRecord(Code, Record)) {
      default: break;  // Default behavior, ignore unknown content.
      case bitc::BLOCKINFO_CODE_SETBID:
        if (Record.size() < 1) return true;
        CurBlockInfo = &BitStream->getOrCreateBlockInfo((unsigned)Record[0]);
        break;
      case bitc::BLOCKINFO_CODE_BLOCKNAME: {
        if (!CurBlockInfo) return true;
        if (BitStream->isIgnoringBlockInfoNames()) break;  // Ignore name.
        std::string Name;
        for (unsigned i = 0, e = Record.size(); i != e; ++i)
          Name += (char)Record[i];
        CurBlockInfo->Name = Name;
        break;
      }
      case bitc::BLOCKINFO_CODE_SETRECORDNAME: {
        if (!CurBlockInfo) return true;
        if (BitStream->isIgnoringBlockInfoNames()) break;  // Ignore name.
        std::string Name;
        for (unsigned i = 1, e = Record.size(); i != e; ++i)
          Name += (char)Record[i];
        CurBlockInfo->RecordNames.push_back(std::make_pair((unsigned)Record[0],
                                                           Name));
        break;
      }
      }
    }
  }
};
  
} // End llvm namespace

#endif
