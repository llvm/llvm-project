//===- BitstreamWriter.h - Low-level bitstream writer interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitstreamWriter class.  This class can be used to
// write an arbitrary bitstream, regardless of its contents.
//
//===----------------------------------------------------------------------===//

#ifndef BITSTREAM_WRITER_H
#define BITSTREAM_WRITER_H

#include "llvm/Bitcode/BitCodes.h"
#include <vector>

namespace llvm {

class BitstreamWriter {
  std::vector<unsigned char> &Out;

  /// CurBit - Always between 0 and 31 inclusive, specifies the next bit to use.
  unsigned CurBit;
  
  /// CurValue - The current value.  Only bits < CurBit are valid.
  uint32_t CurValue;
  
  /// CurCodeSize - This is the declared size of code values used for the
  /// current block, in bits.
  unsigned CurCodeSize;

  /// BlockInfoCurBID - When emitting a BLOCKINFO_BLOCK, this is the currently
  /// selected BLOCK ID.
  unsigned BlockInfoCurBID;
  
  /// CurAbbrevs - Abbrevs installed at in this block.
  std::vector<BitCodeAbbrev*> CurAbbrevs;

  struct Block {
    unsigned PrevCodeSize;
    unsigned StartSizeWord;
    std::vector<BitCodeAbbrev*> PrevAbbrevs;
    Block(unsigned PCS, unsigned SSW) : PrevCodeSize(PCS), StartSizeWord(SSW) {}
  };
  
  /// BlockScope - This tracks the current blocks that we have entered.
  std::vector<Block> BlockScope;
  
  /// BlockInfo - This contains information emitted to BLOCKINFO_BLOCK blocks.
  /// These describe abbreviations that all blocks of the specified ID inherit.
  struct BlockInfo {
    unsigned BlockID;
    std::vector<BitCodeAbbrev*> Abbrevs;
  };
  std::vector<BlockInfo> BlockInfoRecords;
  
public:
  explicit BitstreamWriter(std::vector<unsigned char> &O) 
    : Out(O), CurBit(0), CurValue(0), CurCodeSize(2) {}

  ~BitstreamWriter() {
    assert(CurBit == 0 && "Unflused data remaining");
    assert(BlockScope.empty() && CurAbbrevs.empty() && "Block imbalance");
    
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

  std::vector<unsigned char> &getBuffer() { return Out; }

  //===--------------------------------------------------------------------===//
  // Basic Primitives for emitting bits to the stream.
  //===--------------------------------------------------------------------===//
  
  void Emit(uint32_t Val, unsigned NumBits) {
    assert(NumBits <= 32 && "Invalid value size!");
    assert((Val & ~(~0U >> (32-NumBits))) == 0 && "High bits set!");
    CurValue |= Val << CurBit;
    if (CurBit + NumBits < 32) {
      CurBit += NumBits;
      return;
    }
    
    // Add the current word.
    unsigned V = CurValue;
    Out.push_back((unsigned char)(V >>  0));
    Out.push_back((unsigned char)(V >>  8));
    Out.push_back((unsigned char)(V >> 16));
    Out.push_back((unsigned char)(V >> 24));
    
    if (CurBit)
      CurValue = Val >> (32-CurBit);
    else
      CurValue = 0;
    CurBit = (CurBit+NumBits) & 31;
  }
  
  void Emit64(uint64_t Val, unsigned NumBits) {
    if (NumBits <= 32)
      Emit((uint32_t)Val, NumBits);
    else {
      Emit((uint32_t)Val, 32);
      Emit((uint32_t)(Val >> 32), NumBits-32);
    }
  }
  
  void FlushToWord() {
    if (CurBit) {
      unsigned V = CurValue;
      Out.push_back((unsigned char)(V >>  0));
      Out.push_back((unsigned char)(V >>  8));
      Out.push_back((unsigned char)(V >> 16));
      Out.push_back((unsigned char)(V >> 24));
      CurBit = 0;
      CurValue = 0;
    }
  }
  
  void EmitVBR(uint32_t Val, unsigned NumBits) {
    uint32_t Threshold = 1U << (NumBits-1);
    
    // Emit the bits with VBR encoding, NumBits-1 bits at a time.
    while (Val >= Threshold) {
      Emit((Val & ((1 << (NumBits-1))-1)) | (1 << (NumBits-1)), NumBits);
      Val >>= NumBits-1;
    }
    
    Emit(Val, NumBits);
  }
  
  void EmitVBR64(uint64_t Val, unsigned NumBits) {
    if ((uint32_t)Val == Val)
      return EmitVBR((uint32_t)Val, NumBits);
    
    uint64_t Threshold = 1U << (NumBits-1);
    
    // Emit the bits with VBR encoding, NumBits-1 bits at a time.
    while (Val >= Threshold) {
      Emit(((uint32_t)Val & ((1 << (NumBits-1))-1)) |
           (1 << (NumBits-1)), NumBits);
      Val >>= NumBits-1;
    }
    
    Emit((uint32_t)Val, NumBits);
  }
  
  /// EmitCode - Emit the specified code.
  void EmitCode(unsigned Val) {
    Emit(Val, CurCodeSize);
  }
  
  // BackpatchWord - Backpatch a 32-bit word in the output with the specified
  // value.
  void BackpatchWord(unsigned ByteNo, unsigned NewWord) {
    Out[ByteNo++] = (unsigned char)(NewWord >>  0);
    Out[ByteNo++] = (unsigned char)(NewWord >>  8);
    Out[ByteNo++] = (unsigned char)(NewWord >> 16);
    Out[ByteNo  ] = (unsigned char)(NewWord >> 24);
  }
  
  //===--------------------------------------------------------------------===//
  // Block Manipulation
  //===--------------------------------------------------------------------===//
  
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
  
  void EnterSubblock(unsigned BlockID, unsigned CodeLen) {
    // Block header:
    //    [ENTER_SUBBLOCK, blockid, newcodelen, <align4bytes>, blocklen]
    EmitCode(bitc::ENTER_SUBBLOCK);
    EmitVBR(BlockID, bitc::BlockIDWidth);
    EmitVBR(CodeLen, bitc::CodeLenWidth);
    FlushToWord();
    
    unsigned BlockSizeWordLoc = static_cast<unsigned>(Out.size());
    unsigned OldCodeSize = CurCodeSize;
    
    // Emit a placeholder, which will be replaced when the block is popped.
    Emit(0, bitc::BlockSizeWidth);
    
    CurCodeSize = CodeLen;
    
    // Push the outer block's abbrev set onto the stack, start out with an
    // empty abbrev set.
    BlockScope.push_back(Block(OldCodeSize, BlockSizeWordLoc/4));
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);

    // If there is a blockinfo for this BlockID, add all the predefined abbrevs
    // to the abbrev list.
    if (BlockInfo *Info = getBlockInfo(BlockID)) {
      for (unsigned i = 0, e = static_cast<unsigned>(Info->Abbrevs.size());
           i != e; ++i) {
        CurAbbrevs.push_back(Info->Abbrevs[i]);
        Info->Abbrevs[i]->addRef();
      }
    }
  }
  
  void ExitBlock() {
    assert(!BlockScope.empty() && "Block scope imbalance!");
    
    // Delete all abbrevs.
    for (unsigned i = 0, e = static_cast<unsigned>(CurAbbrevs.size());
         i != e; ++i)
      CurAbbrevs[i]->dropRef();
    
    const Block &B = BlockScope.back();
    
    // Block tail:
    //    [END_BLOCK, <align4bytes>]
    EmitCode(bitc::END_BLOCK);
    FlushToWord();

    // Compute the size of the block, in words, not counting the size field.
    unsigned SizeInWords= static_cast<unsigned>(Out.size())/4-B.StartSizeWord-1;
    unsigned ByteNo = B.StartSizeWord*4;
    
    // Update the block size field in the header of this sub-block.
    BackpatchWord(ByteNo, SizeInWords);
    
    // Restore the inner block's code size and abbrev table.
    CurCodeSize = B.PrevCodeSize;
    BlockScope.back().PrevAbbrevs.swap(CurAbbrevs);
    BlockScope.pop_back();
  }
  
  //===--------------------------------------------------------------------===//
  // Record Emission
  //===--------------------------------------------------------------------===//
  
private:
  /// EmitAbbreviatedField - Emit a single scalar field value with the specified
  /// encoding.
  template<typename uintty>
  void EmitAbbreviatedField(const BitCodeAbbrevOp &Op, uintty V) {
    if (Op.isLiteral()) {
      // If the abbrev specifies the literal value to use, don't emit
      // anything.
      assert(V == Op.getLiteralValue() &&
             "Invalid abbrev for record!");
      return;
    }
    
    // Encode the value as we are commanded.
    switch (Op.getEncoding()) {
    default: assert(0 && "Unknown encoding!");
    case BitCodeAbbrevOp::Fixed:
      Emit((unsigned)V, (unsigned)Op.getEncodingData());
      break;
    case BitCodeAbbrevOp::VBR:
      EmitVBR64(V, (unsigned)Op.getEncodingData());
      break;
    case BitCodeAbbrevOp::Char6:
      Emit(BitCodeAbbrevOp::EncodeChar6((char)V), 6);
      break;
    }        
  }
public:
    
  /// EmitRecord - Emit the specified record to the stream, using an abbrev if
  /// we have one to compress the output.
  template<typename uintty>
  void EmitRecord(unsigned Code, SmallVectorImpl<uintty> &Vals,
                  unsigned Abbrev = 0) {
    if (Abbrev) {
      unsigned AbbrevNo = Abbrev-bitc::FIRST_APPLICATION_ABBREV;
      assert(AbbrevNo < CurAbbrevs.size() && "Invalid abbrev #!");
      BitCodeAbbrev *Abbv = CurAbbrevs[AbbrevNo];
      
      EmitCode(Abbrev);
      
      // Insert the code into Vals to treat it uniformly.
      Vals.insert(Vals.begin(), Code);
      
      unsigned RecordIdx = 0;
      for (unsigned i = 0, e = static_cast<unsigned>(Abbv->getNumOperandInfos());
           i != e; ++i) {
        const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
        if (Op.isLiteral() || Op.getEncoding() != BitCodeAbbrevOp::Array) {
          assert(RecordIdx < Vals.size() && "Invalid abbrev/record");
          EmitAbbreviatedField(Op, Vals[RecordIdx]);
          ++RecordIdx;
        } else {
          // Array case.
          assert(i+2 == e && "array op not second to last?");
          const BitCodeAbbrevOp &EltEnc = Abbv->getOperandInfo(++i);
          
          // Emit a vbr6 to indicate the number of elements present.
          EmitVBR(static_cast<uint32_t>(Vals.size()-RecordIdx), 6);
          
          // Emit each field.
          for (; RecordIdx != Vals.size(); ++RecordIdx)
            EmitAbbreviatedField(EltEnc, Vals[RecordIdx]);
        }
      }
      assert(RecordIdx == Vals.size() && "Not all record operands emitted!");
    } else {
      // If we don't have an abbrev to use, emit this in its fully unabbreviated
      // form.
      EmitCode(bitc::UNABBREV_RECORD);
      EmitVBR(Code, 6);
      EmitVBR(static_cast<uint32_t>(Vals.size()), 6);
      for (unsigned i = 0, e = static_cast<unsigned>(Vals.size()); i != e; ++i)
        EmitVBR64(Vals[i], 6);
    }
  }

  //===--------------------------------------------------------------------===//
  // Abbrev Emission
  //===--------------------------------------------------------------------===//
  
private:
  // Emit the abbreviation as a DEFINE_ABBREV record.
  void EncodeAbbrev(BitCodeAbbrev *Abbv) {
    EmitCode(bitc::DEFINE_ABBREV);
    EmitVBR(Abbv->getNumOperandInfos(), 5);
    for (unsigned i = 0, e = static_cast<unsigned>(Abbv->getNumOperandInfos());
         i != e; ++i) {
      const BitCodeAbbrevOp &Op = Abbv->getOperandInfo(i);
      Emit(Op.isLiteral(), 1);
      if (Op.isLiteral()) {
        EmitVBR64(Op.getLiteralValue(), 8);
      } else {
        Emit(Op.getEncoding(), 3);
        if (Op.hasEncodingData())
          EmitVBR64(Op.getEncodingData(), 5);
      }
    }
  }
public:
    
  /// EmitAbbrev - This emits an abbreviation to the stream.  Note that this
  /// method takes ownership of the specified abbrev.
  unsigned EmitAbbrev(BitCodeAbbrev *Abbv) {
    // Emit the abbreviation as a record.
    EncodeAbbrev(Abbv);
    CurAbbrevs.push_back(Abbv);
    return static_cast<unsigned>(CurAbbrevs.size())-1 +
      bitc::FIRST_APPLICATION_ABBREV;
  }
  
  //===--------------------------------------------------------------------===//
  // BlockInfo Block Emission
  //===--------------------------------------------------------------------===//
  
  /// EnterBlockInfoBlock - Start emitting the BLOCKINFO_BLOCK.
  void EnterBlockInfoBlock(unsigned CodeWidth) {
    EnterSubblock(bitc::BLOCKINFO_BLOCK_ID, CodeWidth);
    BlockInfoCurBID = -1U;
  }
private:  
  /// SwitchToBlockID - If we aren't already talking about the specified block
  /// ID, emit a BLOCKINFO_CODE_SETBID record.
  void SwitchToBlockID(unsigned BlockID) {
    if (BlockInfoCurBID == BlockID) return;
    SmallVector<unsigned, 2> V;
    V.push_back(BlockID);
    EmitRecord(bitc::BLOCKINFO_CODE_SETBID, V);
    BlockInfoCurBID = BlockID;
  }

  BlockInfo &getOrCreateBlockInfo(unsigned BlockID) {
    if (BlockInfo *BI = getBlockInfo(BlockID))
      return *BI;
    
    // Otherwise, add a new record.
    BlockInfoRecords.push_back(BlockInfo());
    BlockInfoRecords.back().BlockID = BlockID;
    return BlockInfoRecords.back();
  }
  
public:
  
  /// EmitBlockInfoAbbrev - Emit a DEFINE_ABBREV record for the specified
  /// BlockID.
  unsigned EmitBlockInfoAbbrev(unsigned BlockID, BitCodeAbbrev *Abbv) {
    SwitchToBlockID(BlockID);
    EncodeAbbrev(Abbv);
    
    // Add the abbrev to the specified block record.
    BlockInfo &Info = getOrCreateBlockInfo(BlockID);
    Info.Abbrevs.push_back(Abbv);
    
    return Info.Abbrevs.size()-1+bitc::FIRST_APPLICATION_ABBREV;
  }
};


} // End llvm namespace

#endif
