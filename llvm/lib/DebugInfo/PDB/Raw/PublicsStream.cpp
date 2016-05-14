//===- PublicsStream.cpp - PDB Public Symbol Stream -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The data structures defined in this file are based on the reference
// implementation which is available at
// https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/gsi.h
//
// When you are reading the reference source code, you'd find the
// information below useful.
//
//  - ppdb1->m_fMinimalDbgInfo seems to be always true.
//  - SMALLBUCKETS macro is defined.
//
// The reference doesn't compile, so I learned just by reading code.
// It's not guaranteed to be correct.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/PublicsStream.h"

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::pdb;


static const unsigned IPHR_HASH = 4096;

// This is PSGSIHDR struct defined in
// https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/gsi.h
struct PublicsStream::HeaderInfo {
  ulittle32_t SymHash;
  ulittle32_t AddrMap;
  ulittle32_t NumThunks;
  ulittle32_t SizeOfThunk;
  ulittle16_t ISectThunkTable;
  char Padding[2];
  ulittle32_t OffThunkTable;
  ulittle32_t NumSects;
};


// This is GSIHashHdr struct defined in
struct PublicsStream::GSIHashHeader {
  enum {
    HdrSignature = -1,
    HdrVersion = 0xeffe0000 + 19990810,
  };
  ulittle32_t VerSignature;
  ulittle32_t VerHdr;
  ulittle32_t HrSize;
  ulittle32_t NumBuckets;
};

struct PublicsStream::HRFile {
  ulittle32_t Off;
  ulittle32_t CRef;
};

PublicsStream::PublicsStream(PDBFile &File, uint32_t StreamNum)
    : StreamNum(StreamNum), Stream(StreamNum, File) {}

PublicsStream::~PublicsStream() {}

uint32_t PublicsStream::getSymHash() const { return Header->SymHash; }
uint32_t PublicsStream::getAddrMap() const { return Header->AddrMap; }

// Publics stream contains fixed-size headers and a serialized hash table.
// This implementation is not complete yet. It reads till the end of the
// stream so that we verify the stream is at least not corrupted. However,
// we skip over the hash table which we believe contains information about
// public symbols.
Error PublicsStream::reload() {
  StreamReader Reader(Stream);

  // Check stream size.
  if (Reader.bytesRemaining() < sizeof(HeaderInfo) + sizeof(GSIHashHeader))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Publics Stream does not contain a header.");

  // Read PSGSIHDR and GSIHashHdr structs.
  Header.reset(new HeaderInfo());
  if (Reader.readObject(Header.get()))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Publics Stream does not contain a header.");
  HashHdr.reset(new GSIHashHeader());
  if (Reader.readObject(HashHdr.get()))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Publics Stream does not contain a header.");

  // An array of HRFile follows. Read them.
  if (HashHdr->HrSize % sizeof(HRFile))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid HR array size.");
  std::vector<HRFile> HRs(HashHdr->HrSize / sizeof(HRFile));
  if (auto EC = Reader.readArray<HRFile>(HRs))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Could not read an HR array");

  // A bitmap of a fixed length follows.
  size_t BitmapSizeInBits = alignTo(IPHR_HASH + 1, 32);
  std::vector<uint8_t> Bitmap(BitmapSizeInBits / 8);
  if (auto EC = Reader.readArray<uint8_t>(Bitmap))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Could not read a bitmap.");
  for (uint8_t B : Bitmap)
    NumBuckets += countPopulation(B);

  // Buckets follow.
  if (Reader.bytesRemaining() < NumBuckets * sizeof(uint32_t))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Hash buckets corrupted.");

  return Error::success();
}
