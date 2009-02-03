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

#include "llvm/Bitcode/BitCodes.h"
#include <vector>

namespace llvm {
  
  class Deserializer;
  
class BitstreamReader {
  const unsigned char *NextChar;
  const unsigned char *LastChar;
  friend class Deserializer;
  
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

  /// BlockInfo - This contains information emitted to BLOCKINFO_BLOCK blocks.
  /// These describe abbreviations that all blocks of the specified ID inherit.
  struct BlockInfo {
    unsigned BlockID;
    std::vector<BitCodeAbbrev*> Abbrevs;
  };
  std::vector<BlockInfo> BlockInfoRecords;
  
  /// FirstChar - This remembers the first byte of the stream.
  const unsigned char *FirstChar;
public:
  BitstreamReader() {
    NextChar = FirstChar = LastChar = 0;
    CurWord = 0;
    BitsInCurWord = 0;
    CurCodeSize = 0;
  }

  BitstreamReader(const unsigned char *Start, const unsigned char *End) {
    init(Start, End);
  }
  
  void init(const unsigned char *Start, const unsigned char *End) {
    NextChar = FirstChar = Start;
    LastChar = End;
    assert(((End-Start) & 3) == 0 &&"Bitcode stream not a multiple of 4 bytes");
    CurWord = 0;
    BitsInCurWord = 0;
    CurCodeSize = 2;
  }
  
  ~BitstreamReader() {
    // Abbrevs could still exist if the stream was broken.  If so, don't leak
    // them.
    for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
         i != e; ++i)
      CurAbbrevs[i]->dropRef();

    for (unsigned S = 0, e = static_cast<unsigned>(BlockScope.size());
         S != e; ++S) {
      std::vector<BitCodeAbbrev*> &Abbrevs = BlockScope[S].PrevAbbrevs;
      for (unsigned i = 0, e = static_cast<unsigned>(Abbrevs.size());
           i != e; ++i)
        Abbrevs[i]->dropRef();
    }
    
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
  
  bool AtEndOfStream() const {
    return NextChar == LastChar && BitsInCurWord == 0;
  }
  
  /// GetCurrentBitNo - Return the bit # of the bit we are reading.
  uint64_t GetCurrentBitNo() const {
    return (NextChar-FirstChar)*8 + ((32-BitsInCurWord) & 31);
  }
  
  /// JumpToBit - Reset the stream to the specified bit number.
  void JumpToBit(uint64_t BitNo) {
    uintptr_t ByteNo = uintptr_t(BitNo/8) & ~3;
    uintptr_t WordBitNo = uintptr_t(BitNo) & 31;
    assert(ByteNo < (uintptr_t)(LastChar-FirstChar) && "Invalid location");
    
    // Move the cursor to the right word.
    NextChar = FirstChar+ByteNo;
    BitsInCurWord = 0;
    CurWord = 0;
    
    // Skip over any bits that are already consumed.
    if (WordBitNo) {
      NextChar -= 4;
      Read(static_cast<unsigned>(WordBitNo));
    }
  }
  
  /// GetAbbrevIDWidth - Return the number of bits used to encode an abbrev #.
  unsigned GetAbbrevIDWidth() const { return CurCodeSize; }
  
  uint32_t Read(unsigned NumBits) {
    // If the field is fully contained by CurWord, return it quickly.
    if (BitsInCurWord >= NumBits) {
      uint32_t R = CurWord & ((1U << NumBits)-1);
      CurWord >>= NumBits;
      BitsInCurWord -= NumBits;
      return R;
    }

    // If we run out of data, stop at the end of the stream.
    if (LastChar == NextChar) {
      CurWord = 0;
      BitsInCurWord = 0;
      return 0;
    }
    
    unsigned R = CurWord;

    // Read the next word from the stream.
    CurWord = (NextChar[0] <<  0) | (NextChar[1] << 8) |
              (NextChar[2] << 16) | (NextChar[3] << 24);
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
  
  uint64_t ReadVBR64(unsigned NumBits) {
    uint64_t Piece = Read(NumBits);
    if ((Piece & (1U << (NumBits-1))) == 0)
      return Piece;
    
    uint64_t Result = 0;
    unsigned NextBit = 0;
    while (1) {
      Result |= (Piece & ((1U << (NumBits-1))-1)) << NextBit;
      
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

  //===--------------------------------------------------------------------===//
  // Block Manipulation
  //===--------------------------------------------------------------------===//
  
private:
  /// getBlockInfo - If there is block info for the specified ID, return it,
  /// otherwise return null.
  BlockInfo *getBlockInfo(unsigned BlockID) {
    // Common case, the most recent entry matches BlockID.
    if (!BlockInfoRecords.empty() && BlockInfoRecords.back().BlockID == BlockID)
      return &BlockInfoRecords.back();
    
    for (unsigned i = 0, e = static_cast<unsigned>(BlockInfoRecords.size());
         i != e; ++i)
      if (BlockInfoRecords[i].BlockID == BlockID)
        return &BlockInfoRecords[i];
    return 0;
  }
public:
  
  
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
    if (AtEndOfStream() || NextChar+NumWords*4 > LastChar)
      return true;
    
    NextChar += NumWords*4;
    return false;
  }
  
  /// EnterSubBlock - Having read the ENTER_SUBBLOCK abbrevid, enter
  /// the block, and return true if the block is valid.
  bool EnterSubBlock(unsigned BlockID, unsigned *NumWordsP = 0) {
    // Save the current block's state on BlockScope.
    BlockScope.push_back(Block(CurCodeSize));
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
    
    // Add the abbrevs specific to this block to the CurAbbrevs list.
    if (BlockInfo *Info = getBlockInfo(BlockID)) {
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
    if (CurCodeSize == 0 || AtEndOfStream() || NextChar+NumWords*4 > LastChar)
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
  void ReadAbbreviatedField(const BitCodeAbbrevOp &Op, 
                            SmallVectorImpl<uint64_t> &Vals) {
    if (Op.isLiteral()) {
      // If the abbrev specifies the literal value to use, use it.
      Vals.push_back(Op.getLiteralValue());
    } else {
      // Decode the value as we are commanded.
      switch (Op.getEncoding()) {
      default: assert(0 && "Unknown encoding!");
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
  }
public:
  unsigned ReadRecord(unsigned AbbrevID, SmallVectorImpl<uint64_t> &Vals) {
    if (AbbrevID == bitc::UNABBREV_RECORD) {
      unsigned Code = ReadVBR(6);
      unsigned NumElts = ReadVBR(6);
      for (unsigned i = 0; i != NumElts; ++i)
        Vals.push_back(ReadVBR64(6));
      return Code;
    }
    
    unsigned AbbrevNo = AbbrevID-bitc::FIRST_APPLICATION_ABBREV;
    assert(AbbrevNo < CurAbbrevs.size() && "Invalid abbrev #!");
    BitCodeAbbrev *Abbv = CurAbbrevs[AbbrevNo];

    for (unsigned i = 0, e = Abbv->getNumOperandInfos(); i != e; ++i) {
      const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
      if (Op.isLiteral() || Op.getEncoding() != BitCodeAbbrevOp::Array) {
        ReadAbbreviatedField(Op, Vals);
      } else {
        // Array case.  Read the number of elements as a vbr6.
        unsigned NumElts = ReadVBR(6);

        // Get the element encoding.
        assert(i+2 == e && "array op not second to last?");
        const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);

        // Read all the elements.
        for (; NumElts; --NumElts)
          ReadAbbreviatedField(EltEnc, Vals);
      }
    }
    
    unsigned Code = (unsigned)Vals[0];
    Vals.erase(Vals.begin());
    return Code;
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
  
  //===--------------------------------------------------------------------===//
  // BlockInfo Block Reading
  //===--------------------------------------------------------------------===//
  
private:  
  BlockInfo &getOrCreateBlockInfo(unsigned BlockID) {
    if (BlockInfo *BI = getBlockInfo(BlockID))
      return *BI;
    
    // Otherwise, add a new record.
    BlockInfoRecords.push_back(BlockInfo());
    BlockInfoRecords.back().BlockID = BlockID;
    return BlockInfoRecords.back();
  }
  
public:
    
  bool ReadBlockInfoBlock() {
    if (EnterSubBlock(bitc::BLOCKINFO_BLOCK_ID)) return true;

    SmallVector<uint64_t, 64> Record;
    BlockInfo *CurBlockInfo = 0;
    
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
        CurBlockInfo = &getOrCreateBlockInfo((unsigned)Record[0]);
        break;
      }
    }      
  }
};

} // End llvm namespace

#endif
