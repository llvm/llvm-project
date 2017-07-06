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

void llvm::pdb::discoverStreamPurposes(PDBFile &File,
                                       SmallVectorImpl<std::string> &Purposes,
                                       uint32_t MaxLen) {

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
    std::string Value;
    if (StreamIdx == OldMSFDirectory)
      Value = truncateStringBack("Old MSF Directory", MaxLen);
    else if (StreamIdx == StreamPDB)
      Value = truncateStringBack("PDB Stream", MaxLen);
    else if (StreamIdx == StreamDBI)
      Value = truncateStringBack("DBI Stream", MaxLen);
    else if (StreamIdx == StreamTPI)
      Value = truncateStringBack("TPI Stream", MaxLen);
    else if (StreamIdx == StreamIPI)
      Value = truncateStringBack("IPI Stream", MaxLen);
    else if (Dbi && StreamIdx == Dbi->getGlobalSymbolStreamIndex())
      Value = truncateStringBack("Global Symbol Hash", MaxLen);
    else if (Dbi && StreamIdx == Dbi->getPublicSymbolStreamIndex())
      Value = truncateStringBack("Public Symbol Hash", MaxLen);
    else if (Dbi && StreamIdx == Dbi->getSymRecordStreamIndex())
      Value = truncateStringBack("Public Symbol Records", MaxLen);
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamIndex())
      Value = truncateStringBack("TPI Hash", MaxLen);
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamAuxIndex())
      Value = truncateStringBack("TPI Aux Hash", MaxLen);
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamIndex())
      Value = truncateStringBack("IPI Hash", MaxLen);
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamAuxIndex())
      Value = truncateStringBack("IPI Aux Hash", MaxLen);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Exception))
      Value = truncateStringBack("Exception Data", MaxLen);
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Fixup))
      Value = truncateStringBack("Fixup Data", MaxLen);
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::FPO))
      Value = truncateStringBack("FPO Data", MaxLen);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::NewFPO))
      Value = truncateStringBack("New FPO Data", MaxLen);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapFromSrc))
      Value = truncateStringBack("Omap From Source Data", MaxLen);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapToSrc))
      Value = truncateStringBack("Omap To Source Data", MaxLen);
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Pdata))
      Value = truncateStringBack("Pdata", MaxLen);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdr))
      Value = truncateStringBack("Section Header Data", MaxLen);
    else if (Dbi &&
             StreamIdx ==
                 Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdrOrig))
      Value = truncateStringBack("Section Header Original Data", MaxLen);
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::TokenRidMap))
      Value = truncateStringBack("Token Rid Data", MaxLen);
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Xdata))
      Value = truncateStringBack("Xdata", MaxLen);
    else {
      auto ModIter = ModStreams.find(StreamIdx);
      auto NSIter = NamedStreams.find(StreamIdx);
      if (ModIter != ModStreams.end()) {
        Value = truncateQuotedNameFront(
            "Module", ModIter->second.getModuleName(), MaxLen);
      } else if (NSIter != NamedStreams.end()) {
        Value = truncateQuotedNameBack("Named Stream", NSIter->second, MaxLen);
      } else {
        Value = "???";
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
