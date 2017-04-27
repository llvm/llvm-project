//===- StreamUtil.cpp - PDB stream utilities --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "StreamUtil.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/DebugInfo/PDB/Native/DbiModuleDescriptor.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"

namespace llvm {
namespace pdb {
void discoverStreamPurposes(PDBFile &File,
                            SmallVectorImpl<std::string> &Purposes) {

  // It's OK if we fail to load some of these streams, we still attempt to print
  // what we can.
  auto Dbi = File.getPDBDbiStream();
  auto Tpi = File.getPDBTpiStream();
  auto Ipi = File.getPDBIpiStream();
  auto Info = File.getPDBInfoStream();

  uint32_t StreamCount = File.getNumStreams();
  DenseMap<uint16_t, const ModuleInfoEx *> ModStreams;
  DenseMap<uint16_t, std::string> NamedStreams;

  if (Dbi) {
    for (auto &ModI : Dbi->modules()) {
      uint16_t SN = ModI.Info.getModuleStreamIndex();
      if (SN != kInvalidStreamIndex)
        ModStreams[SN] = &ModI;
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
      Value = "Old MSF Directory";
    else if (StreamIdx == StreamPDB)
      Value = "PDB Stream";
    else if (StreamIdx == StreamDBI)
      Value = "DBI Stream";
    else if (StreamIdx == StreamTPI)
      Value = "TPI Stream";
    else if (StreamIdx == StreamIPI)
      Value = "IPI Stream";
    else if (Dbi && StreamIdx == Dbi->getGlobalSymbolStreamIndex())
      Value = "Global Symbol Hash";
    else if (Dbi && StreamIdx == Dbi->getPublicSymbolStreamIndex())
      Value = "Public Symbol Hash";
    else if (Dbi && StreamIdx == Dbi->getSymRecordStreamIndex())
      Value = "Public Symbol Records";
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamIndex())
      Value = "TPI Hash";
    else if (Tpi && StreamIdx == Tpi->getTypeHashStreamAuxIndex())
      Value = "TPI Aux Hash";
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamIndex())
      Value = "IPI Hash";
    else if (Ipi && StreamIdx == Ipi->getTypeHashStreamAuxIndex())
      Value = "IPI Aux Hash";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Exception))
      Value = "Exception Data";
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Fixup))
      Value = "Fixup Data";
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::FPO))
      Value = "FPO Data";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::NewFPO))
      Value = "New FPO Data";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapFromSrc))
      Value = "Omap From Source Data";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::OmapToSrc))
      Value = "Omap To Source Data";
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Pdata))
      Value = "Pdata";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdr))
      Value = "Section Header Data";
    else if (Dbi &&
             StreamIdx ==
                 Dbi->getDebugStreamIndex(DbgHeaderType::SectionHdrOrig))
      Value = "Section Header Original Data";
    else if (Dbi &&
             StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::TokenRidMap))
      Value = "Token Rid Data";
    else if (Dbi && StreamIdx == Dbi->getDebugStreamIndex(DbgHeaderType::Xdata))
      Value = "Xdata";
    else {
      auto ModIter = ModStreams.find(StreamIdx);
      auto NSIter = NamedStreams.find(StreamIdx);
      if (ModIter != ModStreams.end()) {
        Value = "Module \"";
        Value += ModIter->second->Info.getModuleName().str();
        Value += "\"";
      } else if (NSIter != NamedStreams.end()) {
        Value = "Named Stream \"";
        Value += NSIter->second;
        Value += "\"";
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
}
}
