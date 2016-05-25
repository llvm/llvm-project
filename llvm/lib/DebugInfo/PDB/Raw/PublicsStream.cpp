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
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"
#include "llvm/DebugInfo/PDB/Raw/SymbolStream.h"

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
  ulittle32_t NumSections;
};

// This is GSIHashHdr.
struct PublicsStream::GSIHashHeader {
  enum : unsigned {
    HdrSignature = ~0U,
    HdrVersion = 0xeffe0000 + 19990810,
  };
  ulittle32_t VerSignature;
  ulittle32_t VerHdr;
  ulittle32_t HrSize;
  ulittle32_t NumBuckets;
};

// This is HRFile.
struct PublicsStream::HashRecord {
  ulittle32_t Off; // Offset in the symbol record stream
  ulittle32_t CRef;
};

// This struct is defined as "SO" in langapi/include/pdb.h.
namespace {
struct SectionOffset {
  ulittle32_t Off;
  ulittle16_t Isect;
  char Padding[2];
};
}

PublicsStream::PublicsStream(PDBFile &File, uint32_t StreamNum)
    : Pdb(File), StreamNum(StreamNum), Stream(StreamNum, File) {}

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

  // An array of HashRecord follows. Read them.
  if (HashHdr->HrSize % sizeof(HashRecord))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid HR array size.");
  HashRecords.resize(HashHdr->HrSize / sizeof(HashRecord));
  if (auto EC = Reader.readArray<HashRecord>(HashRecords))
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

  // We don't yet understand the following data structures completely,
  // but we at least know the types and sizes. Here we are trying
  // to read the stream till end so that we at least can detect
  // corrupted streams.

  // Hash buckets follow.
  std::vector<ulittle32_t> TempHashBuckets(NumBuckets);
  if (auto EC = Reader.readArray<ulittle32_t>(TempHashBuckets))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Hash buckets corrupted.");
  HashBuckets.resize(NumBuckets);
  std::copy(TempHashBuckets.begin(), TempHashBuckets.end(),
            HashBuckets.begin());

  // Something called "address map" follows.
  std::vector<ulittle32_t> TempAddressMap(Header->AddrMap / sizeof(uint32_t));
  if (auto EC = Reader.readArray<ulittle32_t>(TempAddressMap))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Could not read an address map.");
  AddressMap.resize(Header->AddrMap / sizeof(uint32_t));
  std::copy(TempAddressMap.begin(), TempAddressMap.end(), AddressMap.begin());

  // Something called "thunk map" follows.
  std::vector<ulittle32_t> TempThunkMap(Header->NumThunks);
  ThunkMap.resize(Header->NumThunks);
  if (auto EC = Reader.readArray<ulittle32_t>(TempThunkMap))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Could not read a thunk map.");
  ThunkMap.resize(Header->NumThunks);
  std::copy(TempThunkMap.begin(), TempThunkMap.end(), ThunkMap.begin());

  // Something called "section map" follows.
  std::vector<SectionOffset> Offsets(Header->NumSections);
  if (auto EC = Reader.readArray<SectionOffset>(Offsets))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Could not read a section map.");
  for (auto &SO : Offsets) {
    SectionOffsets.push_back(SO.Off);
    SectionOffsets.push_back(SO.Isect);
  }

  if (Reader.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Corrupted publics stream.");
  return Error::success();
}

iterator_range<codeview::SymbolIterator> PublicsStream::getSymbols() const {
  using codeview::SymbolIterator;
  auto SymbolS = Pdb.getPDBSymbolStream();
  if (SymbolS.takeError()) {
    return llvm::make_range<SymbolIterator>(SymbolIterator(), SymbolIterator());
  }
  SymbolStream &SS = SymbolS.get();

  return SS.getSymbols();
}
