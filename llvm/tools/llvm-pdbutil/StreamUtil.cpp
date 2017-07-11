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

void llvm::pdb::discoverStreamPurposes(
    PDBFile &File,
    SmallVectorImpl<std::pair<StreamPurpose, std::string>> &Purposes) {
  // It's OK if we fail to load some of these streams, we still attempt to print
  // what we can.
  auto Dbi = File.getPDBDbiStream();
  auto Tpi = File.getPDBTpiStream();
  auto Ipi = File.getPDBIpiStream();
  auto Info = File.getPDBInfoStream();

  uint32_t StreamCount = File.getNumStreams();
  DenseMap<uint16_t, DbiModuleDescriptor> ModStreams;
  DenseMap<uint16_t, std::string> NamedStreams;

  if (Dbi) {
    const DbiModuleList &Modules = Dbi->modules();
    for (uint32_t I = 0; I < Modules.getModuleCount(); ++I) {
      DbiModuleDescriptor Descriptor = Modules.getModuleDescriptor(I);
      uint16_t SN = Descriptor.getModuleStreamIndex();
      if (SN != kInvalidStreamIndex)
        ModStreams[SN] = Descriptor;
    }
  }
  if (Info) {
    for (auto &NSE : Info->named_streams()) {
      if (NSE.second != kInvalidStreamIndex)
        NamedStreams[NSE.second] = NSE.first();
    }
  }

  Purposes.resize(StreamCount);
  for (uint16_t StreamIdx = 0; StreamIdx < StreamCount; ++StreamIdx) {
    std::pair<StreamPurpose, std::string> Value;
    if (StreamIdx == OldMSFDirectory)
      Value = std::make_pair(StreamPurpose::Other, "Old MSF Directory");
    else if (StreamIdx == StreamPDB)
      Value = std::make_pair(StreamPurpose::Other, "PDB Stream");
    else if (StreamIdx == StreamDBI)
      Value = std::make_pair(StreamPurpose::Other, "DBI Stream");
    else if (StreamIdx == StreamTPI)
      Value = std::make_pair(StreamPurpose::Other, "TPI Stream");
    else if (StreamIdx == StreamIPI)
      Value = std::make_pair(StreamPurpose::Other, "IPI Stream");
    else if (Dbi && StreamIdx == Dbi->getGlobalSymbolStreamIndex())
      Value = std::make_pair(StreamPurpose::Other, "Global Symbol Hash");
    else if (Dbi && StreamIdx == Dbi->getPublicSymbolStreamIndex())
      Value = std::make_pair(StreamPurpose::Other, "Public Symbol Hash");
    else if (Dbi && StreamIdx == Dbi->getSymRecordStreamIndex())
      Value = std::make_pair(StreamPurpose::Other, "Public Symbol Records");
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamIndex())
      Value = std::make_pair(StreamPurpose::Other, "TPI Hash");
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamAuxIndex())
      Value = std::make_pair(StreamPurpose::Other, "TPI Aux Hash");
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamIndex())
      Value = std::make_pair(StreamPurpose::Other, "IPI Hash");
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamAuxIndex())
      Value = std::make_pair(StreamPurpose::Other, "IPI Aux Hash");
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Exception))
      Value = std::make_pair(StreamPurpose::Other, "Exception Data");
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Fixup))
      Value = std::make_pair(StreamPurpose::Other, "Fixup Data");
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::FPO))
      Value = std::make_pair(StreamPurpose::Other, "FPO Data");
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::NewFPO))
      Value = std::make_pair(StreamPurpose::Other, "New FPO Data");
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapFromSrc))
      Value = std::make_pair(StreamPurpose::Other, "Omap From Source Data");
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapToSrc))
      Value = std::make_pair(StreamPurpose::Other, "Omap To Source Data");
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Pdata))
      Value = std::make_pair(StreamPurpose::Other, "Pdata");
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdr))
      Value = std::make_pair(StreamPurpose::Other, "Section Header Data");
    else if (Dbi &&
             StreamIdx ==
                 Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdrOrig))
      Value =
          std::make_pair(StreamPurpose::Other, "Section Header Original Data");
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::TokenRidMap))
      Value = std::make_pair(StreamPurpose::Other, "Token Rid Data");
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Xdata))
      Value = std::make_pair(StreamPurpose::Other, "Xdata");
    else {
      auto ModIter = ModStreams.find(StreamIdx);
      auto NSIter = NamedStreams.find(StreamIdx);
      if (ModIter != ModStreams.end()) {
        Value = std::make_pair(StreamPurpose::ModuleStream,
                               ModIter->second.getModuleName());
      } else if (NSIter != NamedStreams.end()) {
        Value = std::make_pair(StreamPurpose::NamedStream, NSIter->second);
      } else {
        Value = std::make_pair(StreamPurpose::Other, "???");
      }
    }
    Purposes[StreamIdx] = Value;
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

void llvm::pdb::discoverStreamPurposes(PDBFile &File,
                                       SmallVectorImpl<std::string> &Purposes) {
  SmallVector<std::pair<StreamPurpose, std::string>, 24> SP;
  discoverStreamPurposes(File, SP);
  Purposes.reserve(SP.size());
  for (const auto &P : SP) {
    if (P.first == StreamPurpose::NamedStream)
      Purposes.push_back(formatv("Named Stream \"{0}\"", P.second));
    else if (P.first == StreamPurpose::ModuleStream)
      Purposes.push_back(formatv("Module \"{0}\"", P.second));
    else
      Purposes.push_back(P.second);
  }
}
