//===- DbiStreamBuilder.cpp - PDB Dbi Stream Creation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/DbiStreamBuilder.h"

#include "llvm/DebugInfo/CodeView/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

DbiStreamBuilder::DbiStreamBuilder()
    : Age(1), BuildNumber(0), PdbDllVersion(0), PdbDllRbld(0), Flags(0),
      MachineType(PDB_Machine::x86) {}

void DbiStreamBuilder::setVersionHeader(PdbRaw_DbiVer V) { VerHeader = V; }

void DbiStreamBuilder::setAge(uint32_t A) { Age = A; }

void DbiStreamBuilder::setBuildNumber(uint16_t B) { BuildNumber = B; }

void DbiStreamBuilder::setPdbDllVersion(uint16_t V) { PdbDllVersion = V; }

void DbiStreamBuilder::setPdbDllRbld(uint16_t R) { PdbDllRbld = R; }

void DbiStreamBuilder::setFlags(uint16_t F) { Flags = F; }

void DbiStreamBuilder::setMachineType(PDB_Machine M) { MachineType = M; }

uint32_t DbiStreamBuilder::calculateSerializedLength() const {
  // For now we only support serializing the header.
  return sizeof(DbiStream::HeaderInfo);
}

Expected<std::unique_ptr<DbiStream>> DbiStreamBuilder::build(PDBFile &File) {
  if (!VerHeader.hasValue())
    return make_error<RawError>(raw_error_code::unspecified,
                                "Missing DBI Stream Version");

  auto DbiS = MappedBlockStream::createIndexedStream(StreamDBI, File);
  if (!DbiS)
    return DbiS.takeError();
  auto DS = std::move(*DbiS);
  DbiStream::HeaderInfo *H =
      static_cast<DbiStream::HeaderInfo *>(DS->getAllocator().Allocate(
          sizeof(DbiStream::HeaderInfo),
          llvm::AlignOf<DbiStream::HeaderInfo>::Alignment));
  H->VersionHeader = *VerHeader;
  H->VersionSignature = -1;
  H->Age = Age;
  H->BuildNumber = BuildNumber;
  H->Flags = Flags;
  H->PdbDllRbld = PdbDllRbld;
  H->PdbDllVersion = PdbDllVersion;
  H->MachineType = static_cast<uint16_t>(MachineType);

  H->ECSubstreamSize = 0;
  H->FileInfoSize = 0;
  H->ModiSubstreamSize = 0;
  H->OptionalDbgHdrSize = 0;
  H->SecContrSubstreamSize = 0;
  H->SectionMapSize = 0;
  H->TypeServerSize = 0;
  H->SymRecordStreamIndex = DbiStream::InvalidStreamIndex;
  H->PublicSymbolStreamIndex = DbiStream::InvalidStreamIndex;
  H->MFCTypeServerIndex = DbiStream::InvalidStreamIndex;
  H->GlobalSymbolStreamIndex = DbiStream::InvalidStreamIndex;

  auto Dbi = llvm::make_unique<DbiStream>(File, std::move(DS));
  Dbi->Header = H;
  return std::move(Dbi);
}
