//===- GlobalsStream.cpp - PDB Index of Symbols by Name ---- ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The on-disk structores used in this file are based on the reference
// implementation which is available at
// https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/gsi.h
//
// When you are reading the reference source code, you'd find the
// information below useful.
//
//  - ppdb1->m_fMinimalDbgInfo seems to be always true.
//  - SMALLBUCKETS macro is defined.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Error.h"
#include <algorithm>

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;

GlobalsStream::GlobalsStream(std::unique_ptr<MappedBlockStream> Stream)
    : Stream(std::move(Stream)) {}

GlobalsStream::~GlobalsStream() = default;

Error GlobalsStream::reload() {
  BinaryStreamReader Reader(*Stream);
  if (auto E = GlobalsTable.read(Reader))
    return E;
  return Error::success();
}

static Error checkHashHdrVersion(const GSIHashHeader *HashHdr) {
  if (HashHdr->VerHdr != GSIHashHeader::HdrVersion)
    return make_error<RawError>(
        raw_error_code::feature_unsupported,
        "Encountered unsupported globals stream version.");

  return Error::success();
}

static Error readGSIHashHeader(const GSIHashHeader *&HashHdr,
                               BinaryStreamReader &Reader) {
  if (Reader.readObject(HashHdr))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Stream does not contain a GSIHashHeader.");

  if (HashHdr->VerSignature != GSIHashHeader::HdrSignature)
    return make_error<RawError>(
        raw_error_code::feature_unsupported,
        "GSIHashHeader signature (0xffffffff) not found.");

  return Error::success();
}

static Error readGSIHashRecords(FixedStreamArray<PSHashRecord> &HashRecords,
                                const GSIHashHeader *HashHdr,
                                BinaryStreamReader &Reader) {
  if (auto EC = checkHashHdrVersion(HashHdr))
    return EC;

  // HashHdr->HrSize specifies the number of bytes of PSHashRecords we have.
  // Verify that we can read them all.
  if (HashHdr->HrSize % sizeof(PSHashRecord))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid HR array size.");
  uint32_t NumHashRecords = HashHdr->HrSize / sizeof(PSHashRecord);
  if (auto EC = Reader.readArray(HashRecords, NumHashRecords))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Error reading hash records."));

  return Error::success();
}

static Error
readGSIHashBuckets(FixedStreamArray<support::ulittle32_t> &HashBuckets,
                   ArrayRef<uint8_t> &HashBitmap, const GSIHashHeader *HashHdr,
                   BinaryStreamReader &Reader) {
  if (auto EC = checkHashHdrVersion(HashHdr))
    return EC;

  // Before the actual hash buckets, there is a bitmap of length determined by
  // IPHR_HASH.
  size_t BitmapSizeInBits = alignTo(IPHR_HASH + 1, 32);
  uint32_t NumBitmapEntries = BitmapSizeInBits / 8;
  if (auto EC = Reader.readBytes(HashBitmap, NumBitmapEntries))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Could not read a bitmap."));
  uint32_t NumBuckets = 0;
  for (uint8_t B : HashBitmap)
    NumBuckets += countPopulation(B);

  // Hash buckets follow.
  if (auto EC = Reader.readArray(HashBuckets, NumBuckets))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Hash buckets corrupted."));

  return Error::success();
}

Error GSIHashTable::read(BinaryStreamReader &Reader) {
  if (auto EC = readGSIHashHeader(HashHdr, Reader))
    return EC;
  if (auto EC = readGSIHashRecords(HashRecords, HashHdr, Reader))
    return EC;
  if (HashHdr->HrSize > 0)
    if (auto EC = readGSIHashBuckets(HashBuckets, HashBitmap, HashHdr, Reader))
      return EC;
  return Error::success();
}
