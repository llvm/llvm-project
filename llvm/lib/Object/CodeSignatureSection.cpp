//===- CodeSignatureSection.cpp - CodeSignatureSection class definition ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CodeSignatureSection class
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/SHA256.h"
#include <cassert>

#if defined(__APPLE__)
#include <sys/mman.h>
#endif

using namespace llvm;
using namespace object;
using namespace support::endian;

static_assert((CodeSignatureSection::BlobHeadersSize % 8) == 0, "");
static_assert((CodeSignatureSection::FixedHeadersSize % 8) == 0, "");

CodeSignatureSection::CodeSignatureSection(uint64_t FileOff,
                                           StringRef OutputFilePath,
                                           MachO::HeaderFileType OutputFileType,
                                           uint64_t TextSegmentFileOff,
                                           uint64_t TextSegmentFileSize)
    : FileOff{FileOff}, OutputFileName{stripOutputFilePath(OutputFilePath)},
      OutputFileType{OutputFileType}, TextSegmentFileOff{TextSegmentFileOff},
      TextSegmentFileSize{TextSegmentFileSize} {}

StringRef
CodeSignatureSection::stripOutputFilePath(const StringRef OutputFilePath) {
  const size_t LastSlashIndex = OutputFilePath.rfind("/");
  if (LastSlashIndex == std::string::npos)
    return OutputFilePath;

  return OutputFilePath.drop_front(LastSlashIndex + 1);
}

uint32_t CodeSignatureSection::getAllHeadersSize() const {
  return alignTo<Align>(FixedHeadersSize + OutputFileName.size() + 1);
}

uint32_t CodeSignatureSection::getBlockCount() const {
  return (FileOff + BlockSize - 1) / BlockSize;
}

uint32_t CodeSignatureSection::getFileNamePad() const {
  return getAllHeadersSize() - FixedHeadersSize - OutputFileName.size();
}

uint32_t CodeSignatureSection::getRawSize() const {
  return getAllHeadersSize() + getBlockCount() * HashSize;
}

uint32_t CodeSignatureSection::getSize() const {
  return alignTo<Align>(getRawSize());
}

void CodeSignatureSection::write(uint8_t *Buf) const {
  const uint32_t AllHeadersSize = getAllHeadersSize();
  const uint32_t BlockCount = getBlockCount();
  const uint32_t FileNamePad = getFileNamePad();
  const uint32_t Size = getSize();

  uint8_t *Code = Buf;
  uint8_t *CodeEnd = Buf + FileOff;
  uint8_t *Hashes = CodeEnd + AllHeadersSize;

  // Write code section header.
  auto *SuperBlob = reinterpret_cast<MachO::CS_SuperBlob *>(CodeEnd);
  write32be(&SuperBlob->magic, MachO::CSMAGIC_EMBEDDED_SIGNATURE);
  write32be(&SuperBlob->length, Size);
  write32be(&SuperBlob->count, 1);
  auto *BlobIndex = reinterpret_cast<MachO::CS_BlobIndex *>(&SuperBlob[1]);
  write32be(&BlobIndex->type, MachO::CSSLOT_CODEDIRECTORY);
  write32be(&BlobIndex->offset, BlobHeadersSize);
  auto *CodeDirectory =
      reinterpret_cast<MachO::CS_CodeDirectory *>(CodeEnd + BlobHeadersSize);
  write32be(&CodeDirectory->magic, MachO::CSMAGIC_CODEDIRECTORY);
  write32be(&CodeDirectory->length, Size - BlobHeadersSize);
  write32be(&CodeDirectory->version, MachO::CS_SUPPORTSEXECSEG);
  write32be(&CodeDirectory->flags, MachO::CS_ADHOC | MachO::CS_LINKER_SIGNED);
  write32be(&CodeDirectory->hashOffset, sizeof(MachO::CS_CodeDirectory) +
                                            OutputFileName.size() +
                                            FileNamePad);
  write32be(&CodeDirectory->identOffset, sizeof(MachO::CS_CodeDirectory));
  CodeDirectory->nSpecialSlots = 0;
  write32be(&CodeDirectory->nCodeSlots, BlockCount);
  write32be(&CodeDirectory->codeLimit, FileOff);
  CodeDirectory->hashSize = static_cast<uint8_t>(HashSize);
  CodeDirectory->hashType = MachO::kSecCodeSignatureHashSHA256;
  CodeDirectory->platform = 0;
  CodeDirectory->pageSize = BlockSizeShift;
  CodeDirectory->spare2 = 0;
  CodeDirectory->scatterOffset = 0;
  CodeDirectory->teamOffset = 0;
  CodeDirectory->spare3 = 0;
  CodeDirectory->codeLimit64 = 0;
  write64be(&CodeDirectory->execSegBase, TextSegmentFileOff);
  write64be(&CodeDirectory->execSegLimit, TextSegmentFileSize);
  write64be(&CodeDirectory->execSegFlags, OutputFileType == MachO::MH_EXECUTE
                                              ? MachO::CS_EXECSEG_MAIN_BINARY
                                              : 0);
  auto *Id = reinterpret_cast<char *>(&CodeDirectory[1]);
  memcpy(Id, OutputFileName.begin(), OutputFileName.size());
  memset(Id + OutputFileName.size(), 0, FileNamePad);

  // Write code section signature.
  while (Code < CodeEnd) {
    StringRef Block(reinterpret_cast<char *>(Code),
                    std::min(CodeEnd - Code, static_cast<ssize_t>(BlockSize)));
    SHA256 Hasher;
    Hasher.update(Block);
    StringRef Hash = Hasher.final();
    assert(Hash.size() == HashSize);
    memcpy(Hashes, Hash.data(), HashSize);
    Code += BlockSize;
    Hashes += HashSize;
  }
#if defined(__APPLE__)
  // This is macOS-specific work-around and makes no sense for any
  // other host OS. See https://openradar.appspot.com/FB8914231
  //
  // The macOS kernel maintains a signature-verification cache to
  // quickly validate applications at time of execve(2).  The trouble
  // is that for the kernel creates the cache entry at the time of the
  // mmap(2) call, before we have a chance to write either the code to
  // sign or the signature header+hashes.  The fix is to invalidate
  // all cached data associated with the output file, thus discarding
  // the bogus prematurely-cached signature.
  msync(Buf, FileOff + Size, MS_INVALIDATE);
#endif
}
