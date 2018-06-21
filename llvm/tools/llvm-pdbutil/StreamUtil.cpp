//===- StreamUtil.cpp - PDB stream utilities --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StreamUtil.h"
#include "FormatUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleList.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"

using namespace llvm;
using namespace llvm::pdb;

std::string StreamInfo::getLongName() const {
  if (Purpose == StreamPurpose::NamedStream)
    return formatv("Named Stream \"{0}\"", Name).str();
  if (Purpose == StreamPurpose::ModuleStream)
    return formatv("Module \"{0}\"", Name).str();
  return Name;
}

StreamInfo StreamInfo::createStream(StreamPurpose Purpose, StringRef Name,
                                    uint32_t StreamIndex) {
  StreamInfo Result;
  Result.Name = Name;
  Result.StreamIndex = StreamIndex;
  Result.Purpose = Purpose;
  return Result;
}

StreamInfo StreamInfo::createModuleStream(StringRef Module,
                                          uint32_t StreamIndex, uint32_t Modi) {
  StreamInfo Result;
  Result.Name = Module;
  Result.StreamIndex = StreamIndex;
  Result.ModuleIndex = Modi;
  Result.Purpose = StreamPurpose::ModuleStream;
  return Result;
}

static inline StreamInfo otherStream(StringRef Label, uint32_t Idx) {
  return StreamInfo::createStream(StreamPurpose::Other, Label, Idx);
}

static inline StreamInfo namedStream(StringRef Label, uint32_t Idx) {
  return StreamInfo::createStream(StreamPurpose::NamedStream, Label, Idx);
}

static inline StreamInfo symbolStream(StringRef Label, uint32_t Idx) {
  return StreamInfo::createStream(StreamPurpose::Symbols, Label, Idx);
}

static inline StreamInfo moduleStream(StringRef Label, uint32_t StreamIdx,
                                      uint32_t Modi) {
  return StreamInfo::createModuleStream(Label, StreamIdx, Modi);
}

struct IndexedModuleDescriptor {
  uint32_t Modi;
  DbiModuleDescriptor Descriptor;
};

void llvm::pdb::discoverStreamPurposes(PDBFile &File,
                                       SmallVectorImpl<StreamInfo> &Streams) {
  // It's OK if we fail to load some of these streams, we still attempt to print
  // what we can.
  auto Dbi = File.getPDBDbiStream();
  auto Tpi = File.getPDBTpiStream();
  auto Ipi = File.getPDBIpiStream();
  auto Info = File.getPDBInfoStream();

  uint32_t StreamCount = File.getNumStreams();
  DenseMap<uint16_t, IndexedModuleDescriptor> ModStreams;
  DenseMap<uint16_t, std::string> NamedStreams;

  if (Dbi) {
    const DbiModuleList &Modules = Dbi->modules();
    for (uint32_t I = 0; I < Modules.getModuleCount(); ++I) {
      IndexedModuleDescriptor IMD;
      IMD.Modi = I;
      IMD.Descriptor = Modules.getModuleDescriptor(I);
      uint16_t SN = IMD.Descriptor.getModuleStreamIndex();
      if (SN != kInvalidStreamIndex)
        ModStreams[SN] = IMD;
    }
  }
  if (Info) {
    for (auto &NSE : Info->named_streams()) {
      if (NSE.second != kInvalidStreamIndex)
        NamedStreams[NSE.second] = NSE.first();
    }
  }

  Streams.resize(StreamCount);
  for (uint16_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
    if (StreamIdx == OldMSFDirectory)
      Streams[StreamIdx] = otherStream("Old MSF Directory", StreamIdx);
    else if (StreamIdx == StreamPDB)
      Streams[StreamIdx] = otherStream("PDB Stream", StreamIdx);
    else if (StreamIdx == StreamDBI)
      Streams[StreamIdx] = otherStream("DBI Stream", StreamIdx);
    else if (StreamIdx == StreamTPI)
      Streams[StreamIdx] = otherStream("TPI Stream", StreamIdx);
    else if (StreamIdx == StreamIPI)
      Streams[StreamIdx] = otherStream("IPI Stream", StreamIdx);
    else if (Dbi && StreamIdx == Dbi->getGlobalSymbolStreamIndex())
      Streams[StreamIdx] = otherStream("Global Symbol Hash", StreamIdx);
    else if (Dbi && StreamIdx == Dbi->getPublicSymbolStreamIndex())
      Streams[StreamIdx] = otherStream("Public Symbol Hash", StreamIdx);
    else if (Dbi && StreamIdx == Dbi->getSymRecordStreamIndex())
      Streams[StreamIdx] = symbolStream("Symbol Records", StreamIdx);
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamIndex())
      Streams[StreamIdx] = otherStream("TPI Hash", StreamIdx);
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamAuxIndex())
      Streams[StreamIdx] = otherStream("TPI Aux Hash", StreamIdx);
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamIndex())
      Streams[StreamIdx] = otherStream("IPI Hash", StreamIdx);
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamAuxIndex())
      Streams[StreamIdx] = otherStream("IPI Aux Hash", StreamIdx);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Exception))
      Streams[StreamIdx] = otherStream("Exception Data", StreamIdx);
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Fixup))
      Streams[StreamIdx] = otherStream("Fixup Data", StreamIdx);
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::FPO))
      Streams[StreamIdx] = otherStream("FPO Data", StreamIdx);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::NewFPO))
      Streams[StreamIdx] = otherStream("New FPO Data", StreamIdx);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapFromSrc))
      Streams[StreamIdx] = otherStream("Omap From Source Data", StreamIdx);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapToSrc))
      Streams[StreamIdx] = otherStream("Omap To Source Data", StreamIdx);
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Pdata))
      Streams[StreamIdx] = otherStream("Pdata", StreamIdx);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdr))
      Streams[StreamIdx] = otherStream("Section Header Data", StreamIdx);
    else if (Dbi &&
             StreamIdx ==
                 Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdrOrig))
      Streams[StreamIdx] =
          otherStream("Section Header Original Data", StreamIdx);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::TokenRidMap))
      Streams[StreamIdx] = otherStream("Token Rid Data", StreamIdx);
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Xdata))
      Streams[StreamIdx] = otherStream("Xdata", StreamIdx);
    else {
      auto ModIter = ModStreams.find(StreamIdx);
      auto NSIter = NamedStreams.find(StreamIdx);
      if (ModIter != ModStreams.end()) {
        Streams[StreamIdx] =
            moduleStream(ModIter->second.Descriptor.getModuleName(), StreamIdx,
                         ModIter->second.Modi);
      } else if (NSIter != NamedStreams.end()) {
        Streams[StreamIdx] = namedStream(NSIter->second, StreamIdx);
      } else {
        Streams[StreamIdx] = otherStream("???", StreamIdx);
      }
    }
  }

  // Consume errors from missing streams.
  if (!Dbi)
    consumeError(Dbi.takeError());
  if (!Tpi)
    consumeError(Tpi.takeError());
  if (!Ipi)
    consumeError(Ipi.takeError());
  if (!Info)
    consumeError(Info.takeError());
}
