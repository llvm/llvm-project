//===- PDBDbiStream.cpp - PDB Dbi Stream (Stream 3) Access ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/PDBDbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModInfo.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/PDBInfoStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBRawConstants.h"

using namespace llvm;
using namespace llvm::support;

namespace {
// Some of the values are stored in bitfields.  Since this needs to be portable
// across compilers and architectures (big / little endian in particular) we
// can't use the actual structures below, but must instead do the shifting
// and masking ourselves.  The struct definitions are provided for reference.

// struct DbiFlags {
//  uint16_t IncrementalLinking : 1;  // True if linked incrementally
//  uint16_t IsStripped : 1;          // True if private symbols were stripped.
//  uint16_t HasCTypes : 1;           // True if linked with /debug:ctypes.
//  uint16_t Reserved : 13;
//};
const uint16_t FlagIncrementalMask = 0x0001;
const uint16_t FlagStrippedMask = 0x0002;
const uint16_t FlagHasCTypesMask = 0x0004;

// struct DbiBuildNo {
//  uint16_t MinorVersion : 8;
//  uint16_t MajorVersion : 7;
//  uint16_t NewVersionFormat : 1;
//};
const uint16_t BuildMinorMask = 0x00FF;
const uint16_t BuildMinorShift = 0;

const uint16_t BuildMajorMask = 0x7F00;
const uint16_t BuildMajorShift = 8;
}

struct PDBDbiStream::HeaderInfo {
  little32_t VersionSignature;
  ulittle32_t VersionHeader;
  ulittle32_t Age; // Should match PDBInfoStream.
  ulittle16_t GSSyms;
  ulittle16_t BuildNumber; // See DbiBuildNo structure.
  ulittle16_t PSSyms;
  ulittle16_t PdbDllVersion;        // version of mspdbNNN.dll
  ulittle16_t SymRecords;           // Number of symbols
  ulittle16_t PdbDllRbld;           // rbld number of mspdbNNN.dll
  little32_t ModiSubstreamSize;     // Size of module info stream
  little32_t SecContrSubstreamSize; // Size of sec. contribution stream
  little32_t SectionMapSize;
  little32_t FileInfoSize;
  little32_t TypeServerSize;      // Size of type server map
  ulittle32_t MFCTypeServerIndex; // Index of MFC Type Server
  little32_t OptionalDbgHdrSize;  // Size of DbgHeader info
  little32_t ECSubstreamSize;     // Size of EC stream (what is EC?)
  ulittle16_t Flags;              // See DbiFlags enum.
  ulittle16_t MachineType;        // See PDB_MachineType enum.

  ulittle32_t Reserved; // Pad to 64 bytes
};

PDBDbiStream::PDBDbiStream(PDBFile &File) : Pdb(File), Stream(3, File) {
  static_assert(sizeof(HeaderInfo) == 64, "Invalid HeaderInfo size!");
}

PDBDbiStream::~PDBDbiStream() {}

std::error_code PDBDbiStream::reload() {
  Stream.setOffset(0);
  Header.reset(new HeaderInfo());

  if (Stream.getLength() < sizeof(HeaderInfo))
    return std::make_error_code(std::errc::illegal_byte_sequence);
  Stream.readObject(Header.get());

  if (Header->VersionSignature != -1)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  // Require at least version 7, which should be present in all PDBs
  // produced in the last decade and allows us to avoid having to
  // special case all kinds of complicated arcane formats.
  if (Header->VersionHeader < PdbDbiV70)
    return std::make_error_code(std::errc::not_supported);

  if (Header->Age != Pdb.getPDBInfoStream().getAge())
    return std::make_error_code(std::errc::illegal_byte_sequence);

  if (Stream.getLength() !=
      sizeof(HeaderInfo) + Header->ModiSubstreamSize +
          Header->SecContrSubstreamSize + Header->SectionMapSize +
          Header->FileInfoSize + Header->TypeServerSize +
          Header->OptionalDbgHdrSize + Header->ECSubstreamSize)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  if (Header->ModiSubstreamSize % sizeof(uint32_t) != 0)
    return std::make_error_code(std::errc::illegal_byte_sequence);

  ModInfoSubstream.resize(Header->ModiSubstreamSize);
  if (auto EC =
          Stream.readBytes(&ModInfoSubstream[0], Header->ModiSubstreamSize))
    return EC;

  return std::error_code();
}

PdbRaw_DbiVer PDBDbiStream::getDbiVersion() const {
  uint32_t Value = Header->VersionHeader;
  return static_cast<PdbRaw_DbiVer>(Value);
}

uint32_t PDBDbiStream::getAge() const { return Header->Age; }

bool PDBDbiStream::isIncrementallyLinked() const {
  return (Header->Flags & FlagIncrementalMask) != 0;
}

bool PDBDbiStream::hasCTypes() const {
  return (Header->Flags & FlagHasCTypesMask) != 0;
}

bool PDBDbiStream::isStripped() const {
  return (Header->Flags & FlagStrippedMask) != 0;
}

uint16_t PDBDbiStream::getBuildMajorVersion() const {
  return (Header->BuildNumber & BuildMajorMask) >> BuildMajorShift;
}

uint16_t PDBDbiStream::getBuildMinorVersion() const {
  return (Header->BuildNumber & BuildMinorMask) >> BuildMinorShift;
}

uint32_t PDBDbiStream::getPdbDllVersion() const {
  return Header->PdbDllVersion;
}

uint32_t PDBDbiStream::getNumberOfSymbols() const { return Header->SymRecords; }

PDB_Machine PDBDbiStream::getMachineType() const {
  uint16_t Machine = Header->MachineType;
  return static_cast<PDB_Machine>(Machine);
}

llvm::iterator_range<ModInfoIterator> PDBDbiStream::modules() const {
  return llvm::make_range(ModInfoIterator(&ModInfoSubstream.front()),
                          ModInfoIterator(&ModInfoSubstream.back() + 1));
}
