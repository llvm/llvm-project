//===- GSI.cpp - Common Functions for GlobalsStream and PublicsStream  ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GSI.h"

#include "llvm/DebugInfo/MSF/StreamArray.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"

#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {

static Error checkHashHdrVersion(const GSIHashHeader *HashHdr) {
  if (HashHdr->VerHdr != GSIHashHeader::HdrVersion)
    return make_error<RawError>(
        raw_error_code::feature_unsupported,
        "Encountered unsupported globals stream version.");

  return Error::success();
}

Error readGSIHashBuckets(
    msf::FixedStreamArray<support::ulittle32_t> &HashBuckets,
    const GSIHashHeader *HashHdr, msf::StreamReader &Reader) {
  if (auto EC = checkHashHdrVersion(HashHdr))
    return EC;

  // Before the actual hash buckets, there is a bitmap of length determined by
  // IPHR_HASH.
  ArrayRef<uint8_t> Bitmap;
  size_t BitmapSizeInBits = alignTo(IPHR_HASH + 1, 32);
  uint32_t NumBitmapEntries = BitmapSizeInBits / 8;
  if (auto EC = Reader.readBytes(Bitmap, NumBitmapEntries))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Could not read a bitmap."));
  uint32_t NumBuckets = 0;
  for (uint8_t B : Bitmap)
    NumBuckets += countPopulation(B);

  // Hash buckets follow.
  if (auto EC = Reader.readArray(HashBuckets, NumBuckets))
    return joinErrors(std::move(EC),
                      make_error<RawError>(raw_error_code::corrupt_file,
                                           "Hash buckets corrupted."));

  return Error::success();
}

Error readGSIHashHeader(const GSIHashHeader *&HashHdr,
                        msf::StreamReader &Reader) {
  if (Reader.readObject(HashHdr))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Stream does not contain a GSIHashHeader.");

  if (HashHdr->VerSignature != GSIHashHeader::HdrSignature)
    return make_error<RawError>(
        raw_error_code::feature_unsupported,
        "GSIHashHeader signature (0xffffffff) not found.");

  return Error::success();
}

Error readGSIHashRecords(msf::FixedStreamArray<PSHashRecord> &HashRecords,
                         const GSIHashHeader *HashHdr,
                         msf::StreamReader &Reader) {
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
}
}
