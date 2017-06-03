//===- DbiStream.cpp - PDB Dbi Stream (Stream 3) Access -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/ISectionContribVisitor.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;
using namespace llvm::support;

template <typename ContribType>
static Error loadSectionContribs(FixedStreamArray<ContribType> &Output,
                                 BinaryStreamReader &Reader) {
  if (Reader.bytesRemaining() % sizeof(ContribType) != 0)
    return make_error<RawError>(
        raw_error_code::corrupt_file,
        "Invalid number of bytes of section contributions");

  uint32_t Count = Reader.bytesRemaining() / sizeof(ContribType);
  if (auto EC = Reader.readArray(Output, Count))
    return EC;
  return Error::success();
}

DbiStream::DbiStream(PDBFile &File, std::unique_ptr<MappedBlockStream> Stream)
    : Pdb(File), Stream(std::move(Stream)), Header(nullptr) {}

DbiStream::~DbiStream() = default;

Error DbiStream::reload() {
  BinaryStreamReader Reader(*Stream);

  if (Stream->getLength() < sizeof(DbiStreamHeader))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "DBI Stream does not contain a header.");
  if (auto EC = Reader.readObject(Header))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "DBI Stream does not contain a header.");

  if (Header->VersionSignature != -1)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Invalid DBI version signature.");

  // Require at least version 7, which should be present in all PDBs
  // produced in the last decade and allows us to avoid having to
  // special case all kinds of complicated arcane formats.
  if (Header->VersionHeader < PdbDbiV70)
    return make_error<RawError>(raw_error_code::feature_unsupported,
                                "Unsupported DBI version.");

  if (Stream->getLength() !=
      sizeof(DbiStreamHeader) + Header->ModiSubstreamSize +
          Header->SecContrSubstreamSize + Header->SectionMapSize +
          Header->FileInfoSize + Header->TypeServerSize +
          Header->OptionalDbgHdrSize + Header->ECSubstreamSize)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "DBI Length does not equal sum of substreams.");

  // Only certain substreams are guaranteed to be aligned.  Validate
  // them here.
  if (Header->ModiSubstreamSize % sizeof(uint32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "DBI MODI substream not aligned.");
  if (Header->SecContrSubstreamSize % sizeof(uint32_t) != 0)
    return make_error<RawError>(
        raw_error_code::corrupt_file,
        "DBI section contribution substream not aligned.");
  if (Header->SectionMapSize % sizeof(uint32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "DBI section map substream not aligned.");
  if (Header->FileInfoSize % sizeof(uint32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "DBI file info substream not aligned.");
  if (Header->TypeServerSize % sizeof(uint32_t) != 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "DBI type server substream not aligned.");

  BinaryStreamRef ModInfoSubstream;
  BinaryStreamRef FileInfoSubstream;
  if (auto EC =
          Reader.readStreamRef(ModInfoSubstream, Header->ModiSubstreamSize))
    return EC;

  if (auto EC = Reader.readStreamRef(SecContrSubstream,
                                     Header->SecContrSubstreamSize))
    return EC;
  if (auto EC = Reader.readStreamRef(SecMapSubstream, Header->SectionMapSize))
    return EC;
  if (auto EC = Reader.readStreamRef(FileInfoSubstream, Header->FileInfoSize))
    return EC;
  if (auto EC =
          Reader.readStreamRef(TypeServerMapSubstream, Header->TypeServerSize))
    return EC;
  if (auto EC = Reader.readStreamRef(ECSubstream, Header->ECSubstreamSize))
    return EC;
  if (auto EC = Reader.readArray(
          DbgStreams, Header->OptionalDbgHdrSize / sizeof(ulittle16_t)))
    return EC;

  if (auto EC = Modules.initialize(ModInfoSubstream, FileInfoSubstream))
    return EC;

  if (auto EC = initializeSectionContributionData())
    return EC;
  if (auto EC = initializeSectionHeadersData())
    return EC;
  if (auto EC = initializeSectionMapData())
    return EC;
  if (auto EC = initializeFpoRecords())
    return EC;

  if (Reader.bytesRemaining() > 0)
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Found unexpected bytes in DBI Stream.");

  if (ECSubstream.getLength() > 0) {
    BinaryStreamReader ECReader(ECSubstream);
    if (auto EC = ECNames.reload(ECReader))
      return EC;
  }

  return Error::success();
}

PdbRaw_DbiVer DbiStream::getDbiVersion() const {
  uint32_t Value = Header->VersionHeader;
  return static_cast<PdbRaw_DbiVer>(Value);
}

uint32_t DbiStream::getAge() const { return Header->Age; }

uint16_t DbiStream::getPublicSymbolStreamIndex() const {
  return Header->PublicSymbolStreamIndex;
}

uint16_t DbiStream::getGlobalSymbolStreamIndex() const {
  return Header->GlobalSymbolStreamIndex;
}

uint16_t DbiStream::getFlags() const { return Header->Flags; }

bool DbiStream::isIncrementallyLinked() const {
  return (Header->Flags & DbiFlags::FlagIncrementalMask) != 0;
}

bool DbiStream::hasCTypes() const {
  return (Header->Flags & DbiFlags::FlagHasCTypesMask) != 0;
}

bool DbiStream::isStripped() const {
  return (Header->Flags & DbiFlags::FlagStrippedMask) != 0;
}

uint16_t DbiStream::getBuildNumber() const { return Header->BuildNumber; }

uint16_t DbiStream::getBuildMajorVersion() const {
  return (Header->BuildNumber & DbiBuildNo::BuildMajorMask) >>
         DbiBuildNo::BuildMajorShift;
}

uint16_t DbiStream::getBuildMinorVersion() const {
  return (Header->BuildNumber & DbiBuildNo::BuildMinorMask) >>
         DbiBuildNo::BuildMinorShift;
}

uint16_t DbiStream::getPdbDllRbld() const { return Header->PdbDllRbld; }

uint32_t DbiStream::getPdbDllVersion() const { return Header->PdbDllVersion; }

uint32_t DbiStream::getSymRecordStreamIndex() const {
  return Header->SymRecordStreamIndex;
}

PDB_Machine DbiStream::getMachineType() const {
  uint16_t Machine = Header->MachineType;
  return static_cast<PDB_Machine>(Machine);
}

FixedStreamArray<object::coff_section> DbiStream::getSectionHeaders() {
  return SectionHeaders;
}

FixedStreamArray<object::FpoData> DbiStream::getFpoRecords() {
  return FpoRecords;
}

const DbiModuleList &DbiStream::modules() const { return Modules; }

FixedStreamArray<SecMapEntry> DbiStream::getSectionMap() const {
  return SectionMap;
}

void DbiStream::visitSectionContributions(
    ISectionContribVisitor &Visitor) const {
  if (SectionContribVersion == DbiSecContribVer60) {
    for (auto &SC : SectionContribs)
      Visitor.visit(SC);
  } else if (SectionContribVersion == DbiSecContribV2) {
    for (auto &SC : SectionContribs2)
      Visitor.visit(SC);
  }
}

Error DbiStream::initializeSectionContributionData() {
  if (SecContrSubstream.getLength() == 0)
    return Error::success();

  BinaryStreamReader SCReader(SecContrSubstream);
  if (auto EC = SCReader.readEnum(SectionContribVersion))
    return EC;

  if (SectionContribVersion == DbiSecContribVer60)
    return loadSectionContribs<SectionContrib>(SectionContribs, SCReader);
  if (SectionContribVersion == DbiSecContribV2)
    return loadSectionContribs<SectionContrib2>(SectionContribs2, SCReader);

  return make_error<RawError>(raw_error_code::feature_unsupported,
                              "Unsupported DBI Section Contribution version");
}

// Initializes this->SectionHeaders.
Error DbiStream::initializeSectionHeadersData() {
  if (DbgStreams.size() == 0)
    return Error::success();

  uint32_t StreamNum = getDebugStreamIndex(DbgHeaderType::SectionHdr);
  if (StreamNum >= Pdb.getNumStreams())
    return make_error<RawError>(raw_error_code::no_stream);

  auto SHS = MappedBlockStream::createIndexedStream(
      Pdb.getMsfLayout(), Pdb.getMsfBuffer(), StreamNum, Pdb.getAllocator());

  size_t StreamLen = SHS->getLength();
  if (StreamLen % sizeof(object::coff_section))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Corrupted section header stream.");

  size_t NumSections = StreamLen / sizeof(object::coff_section);
  BinaryStreamReader Reader(*SHS);
  if (auto EC = Reader.readArray(SectionHeaders, NumSections))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Could not read a bitmap.");

  SectionHeaderStream = std::move(SHS);
  return Error::success();
}

// Initializes this->Fpos.
Error DbiStream::initializeFpoRecords() {
  if (DbgStreams.size() == 0)
    return Error::success();

  uint32_t StreamNum = getDebugStreamIndex(DbgHeaderType::NewFPO);

  // This means there is no FPO data.
  if (StreamNum == kInvalidStreamIndex)
    return Error::success();

  if (StreamNum >= Pdb.getNumStreams())
    return make_error<RawError>(raw_error_code::no_stream);

  auto FS = MappedBlockStream::createIndexedStream(
      Pdb.getMsfLayout(), Pdb.getMsfBuffer(), StreamNum, Pdb.getAllocator());

  size_t StreamLen = FS->getLength();
  if (StreamLen % sizeof(object::FpoData))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Corrupted New FPO stream.");

  size_t NumRecords = StreamLen / sizeof(object::FpoData);
  BinaryStreamReader Reader(*FS);
  if (auto EC = Reader.readArray(FpoRecords, NumRecords))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Corrupted New FPO stream.");
  FpoStream = std::move(FS);
  return Error::success();
}

Error DbiStream::initializeSectionMapData() {
  if (SecMapSubstream.getLength() == 0)
    return Error::success();

  BinaryStreamReader SMReader(SecMapSubstream);
  const SecMapHeader *Header;
  if (auto EC = SMReader.readObject(Header))
    return EC;
  if (auto EC = SMReader.readArray(SectionMap, Header->SecCount))
    return EC;
  return Error::success();
}

uint32_t DbiStream::getDebugStreamIndex(DbgHeaderType Type) const {
  uint16_t T = static_cast<uint16_t>(Type);
  if (T >= DbgStreams.size())
    return kInvalidStreamIndex;
  return DbgStreams[T];
}
