//===- ExplainOutputStyle.cpp --------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExplainOutputStyle.h"

#include "FormatUtil.h"
#include "StreamUtil.h"
#include "llvm-pdbutil.h"

#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

ExplainOutputStyle::ExplainOutputStyle(PDBFile &File, uint64_t FileOffset)
    : File(File), FileOffset(FileOffset),
      BlockIndex(FileOffset / File.getBlockSize()),
      OffsetInBlock(FileOffset - BlockIndex * File.getBlockSize()),
      P(2, false, outs()) {}

Error ExplainOutputStyle::dump() {
  P.formatLine("Explaining file offset {0} of file '{1}'.", FileOffset,
               File.getFilePath());

  bool IsAllocated = explainBlockStatus();
  if (!IsAllocated)
    return Error::success();

  AutoIndent Indent(P);
  if (isSuperBlock())
    explainSuperBlockOffset();
  else if (isFpmBlock())
    explainFpmBlockOffset();
  else if (isBlockMapBlock())
    explainBlockMapOffset();
  else if (isStreamDirectoryBlock())
    explainStreamDirectoryOffset();
  else if (auto Index = getBlockStreamIndex())
    explainStreamOffset(*Index);
  else
    explainUnknownBlock();
  return Error::success();
}

bool ExplainOutputStyle::isSuperBlock() const { return BlockIndex == 0; }

bool ExplainOutputStyle::isFpm1() const {
  return ((BlockIndex - 1) % File.getBlockSize() == 0);
}
bool ExplainOutputStyle::isFpm2() const {
  return ((BlockIndex - 2) % File.getBlockSize() == 0);
}

bool ExplainOutputStyle::isFpmBlock() const { return isFpm1() || isFpm2(); }

bool ExplainOutputStyle::isBlockMapBlock() const {
  return BlockIndex == File.getBlockMapIndex();
}

bool ExplainOutputStyle::isStreamDirectoryBlock() const {
  const auto &Layout = File.getMsfLayout();
  return llvm::is_contained(Layout.DirectoryBlocks, BlockIndex);
}

Optional<uint32_t> ExplainOutputStyle::getBlockStreamIndex() const {
  const auto &Layout = File.getMsfLayout();
  for (const auto &Entry : enumerate(Layout.StreamMap)) {
    if (!llvm::is_contained(Entry.value(), BlockIndex))
      continue;
    return Entry.index();
  }
  return None;
}

bool ExplainOutputStyle::explainBlockStatus() {
  if (FileOffset >= File.getFileSize()) {
    P.formatLine("Address {0} is not in the file (file size = {1}).",
                 FileOffset, File.getFileSize());
    return false;
  }
  P.formatLine("Block:Offset = {2:X-}:{1:X-4}.", FileOffset, OffsetInBlock,
               BlockIndex);

  bool IsFree = File.getMsfLayout().FreePageMap[BlockIndex];
  P.formatLine("Address is in block {0} ({1}allocated).", BlockIndex,
               IsFree ? "un" : "");
  return !IsFree;
}

#define endof(Class, Field) (offsetof(Class, Field) + sizeof(Class::Field))

void ExplainOutputStyle::explainSuperBlockOffset() {
  P.formatLine("This corresponds to offset {0} of the MSF super block, ",
               OffsetInBlock);
  if (OffsetInBlock < endof(SuperBlock, MagicBytes))
    P.printLine("which is part of the MSF file magic.");
  else if (OffsetInBlock < endof(SuperBlock, BlockSize)) {
    P.printLine("which contains the block size of the file.");
    P.formatLine("The current value is {0}.",
                 uint32_t(File.getMsfLayout().SB->BlockSize));
  } else if (OffsetInBlock < endof(SuperBlock, FreeBlockMapBlock)) {
    P.printLine("which contains the index of the FPM block (e.g. 1 or 2).");
    P.formatLine("The current value is {0}.",
                 uint32_t(File.getMsfLayout().SB->FreeBlockMapBlock));
  } else if (OffsetInBlock < endof(SuperBlock, NumBlocks)) {
    P.printLine("which contains the number of blocks in the file.");
    P.formatLine("The current value is {0}.",
                 uint32_t(File.getMsfLayout().SB->NumBlocks));
  } else if (OffsetInBlock < endof(SuperBlock, NumDirectoryBytes)) {
    P.printLine("which contains the number of bytes in the stream directory.");
    P.formatLine("The current value is {0}.",
                 uint32_t(File.getMsfLayout().SB->NumDirectoryBytes));
  } else if (OffsetInBlock < endof(SuperBlock, Unknown1)) {
    P.printLine("whose purpose is unknown.");
    P.formatLine("The current value is {0}.",
                 uint32_t(File.getMsfLayout().SB->Unknown1));
  } else if (OffsetInBlock < endof(SuperBlock, BlockMapAddr)) {
    P.printLine("which contains the file offset of the block map.");
    P.formatLine("The current value is {0}.",
                 uint32_t(File.getMsfLayout().SB->BlockMapAddr));
  } else {
    assert(OffsetInBlock > sizeof(SuperBlock));
    P.printLine(
        "which is outside the range of valid data for the super block.");
  }
}

static std::string toBinaryString(uint8_t Byte) {
  char Result[9] = {0};
  for (int I = 0; I < 8; ++I) {
    char C = (Byte & 1) ? '1' : '0';
    Result[I] = C;
    Byte >>= 1;
  }
  return std::string(Result);
}

void ExplainOutputStyle::explainFpmBlockOffset() {
  const MSFLayout &Layout = File.getMsfLayout();
  uint32_t MainFpm = Layout.mainFpmBlock();
  uint32_t AltFpm = Layout.alternateFpmBlock();

  assert(isFpmBlock());
  uint32_t Fpm = isFpm1() ? 1 : 2;
  uint32_t FpmChunk = BlockIndex / File.getBlockSize();
  assert((Fpm == MainFpm) || (Fpm == AltFpm));
  (void)AltFpm;
  bool IsMain = (Fpm == MainFpm);
  P.formatLine("Address is in FPM{0} ({1} FPM)", Fpm, IsMain ? "Main" : "Alt");
  uint32_t DescribedBlockStart =
      8 * (FpmChunk * File.getBlockSize() + OffsetInBlock);
  if (DescribedBlockStart > File.getBlockCount()) {
    P.printLine("Address is in extraneous FPM space.");
    return;
  }

  P.formatLine("Address describes the allocation status of blocks [{0},{1})",
               DescribedBlockStart, DescribedBlockStart + 8);
  ArrayRef<uint8_t> Bytes;
  cantFail(File.getMsfBuffer().readBytes(FileOffset, 1, Bytes));
  P.formatLine("Status = {0} (Note: 0 = allocated, 1 = free)",
               toBinaryString(Bytes[0]));
}

void ExplainOutputStyle::explainBlockMapOffset() {
  uint64_t BlockMapOffset = File.getBlockMapOffset();
  uint32_t OffsetInBlock = FileOffset - BlockMapOffset;
  P.formatLine("Address is at offset {0} of the directory block list",
               OffsetInBlock);
}

static uint32_t getOffsetInStream(ArrayRef<support::ulittle32_t> StreamBlocks,
                                  uint64_t FileOffset, uint32_t BlockSize) {
  uint32_t BlockIndex = FileOffset / BlockSize;
  uint32_t OffsetInBlock = FileOffset - BlockIndex * BlockSize;

  auto Iter = llvm::find(StreamBlocks, BlockIndex);
  assert(Iter != StreamBlocks.end());
  uint32_t StreamBlockIndex = std::distance(StreamBlocks.begin(), Iter);
  return StreamBlockIndex * BlockSize + OffsetInBlock;
}

void ExplainOutputStyle::explainStreamOffset(uint32_t Stream) {
  SmallVector<StreamInfo, 12> Streams;
  discoverStreamPurposes(File, Streams);

  assert(Stream <= Streams.size());
  const StreamInfo &S = Streams[Stream];
  const auto &Layout = File.getStreamLayout(Stream);
  uint32_t StreamOff =
      getOffsetInStream(Layout.Blocks, FileOffset, File.getBlockSize());
  P.formatLine("Address is at offset {0}/{1} of Stream {2} ({3}){4}.",
               StreamOff, Layout.Length, Stream, S.getLongName(),
               (StreamOff > Layout.Length) ? " in unused space" : "");
}

void ExplainOutputStyle::explainStreamDirectoryOffset() {
  auto DirectoryBlocks = File.getDirectoryBlockArray();
  const auto &Layout = File.getMsfLayout();
  uint32_t StreamOff =
      getOffsetInStream(DirectoryBlocks, FileOffset, File.getBlockSize());
  P.formatLine("Address is at offset {0}/{1} of Stream Directory{2}.",
               StreamOff, uint32_t(Layout.SB->NumDirectoryBytes),
               uint32_t(StreamOff > Layout.SB->NumDirectoryBytes)
                   ? " in unused space"
                   : "");
}

void ExplainOutputStyle::explainUnknownBlock() {
  P.formatLine("Address has unknown purpose.");
}
