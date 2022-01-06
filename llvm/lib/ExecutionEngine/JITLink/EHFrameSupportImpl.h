//===------- EHFrameSupportImpl.h - JITLink eh-frame utils ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EHFrame registration support for JITLink.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORTIMPL_H
#define LLVM_LIB_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORTIMPL_H

#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"

#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/BinaryStreamReader.h"

namespace llvm {
namespace jitlink {

/// A LinkGraph pass that splits blocks in an eh-frame section into sub-blocks
/// representing individual eh-frames.
/// EHFrameSplitter should not be run without EHFrameEdgeFixer, which is
/// responsible for adding FDE-to-CIE edges.
class EHFrameSplitter {
public:
  EHFrameSplitter(StringRef EHFrameSectionName);
  Error operator()(LinkGraph &G);

private:
  Error processBlock(LinkGraph &G, Block &B, LinkGraph::SplitBlockCache &Cache);

  StringRef EHFrameSectionName;
};

/// A LinkGraph pass that adds missing FDE-to-CIE, FDE-to-PC and FDE-to-LSDA
/// edges.
class EHFrameEdgeFixer {
public:
  EHFrameEdgeFixer(StringRef EHFrameSectionName, unsigned PointerSize,
                   Edge::Kind Delta64, Edge::Kind Delta32,
                   Edge::Kind NegDelta32);
  Error operator()(LinkGraph &G);

private:

  struct AugmentationInfo {
    bool AugmentationDataPresent = false;
    bool EHDataFieldPresent = false;
    uint8_t Fields[4] = {0x0, 0x0, 0x0, 0x0};
  };

  struct CIEInformation {
    CIEInformation() = default;
    CIEInformation(Symbol &CIESymbol) : CIESymbol(&CIESymbol) {}
    Symbol *CIESymbol = nullptr;
    bool FDEsHaveLSDAField = false;
    uint8_t FDEPointerEncoding = 0;
    uint8_t LSDAPointerEncoding = 0;
  };

  struct EdgeTarget {
    EdgeTarget() = default;
    EdgeTarget(const Edge &E) : Target(&E.getTarget()), Addend(E.getAddend()) {}

    Symbol *Target = nullptr;
    Edge::AddendT Addend = 0;
  };

  using BlockEdgeMap = DenseMap<Edge::OffsetT, EdgeTarget>;
  using CIEInfosMap = DenseMap<orc::ExecutorAddr, CIEInformation>;

  struct ParseContext {
    ParseContext(LinkGraph &G) : G(G) {}

    Expected<CIEInformation *> findCIEInfo(orc::ExecutorAddr Address) {
      auto I = CIEInfos.find(Address);
      if (I == CIEInfos.end())
        return make_error<JITLinkError>("No CIE found at address " +
                                        formatv("{0:x16}", Address));
      return &I->second;
    }

    LinkGraph &G;
    CIEInfosMap CIEInfos;
    BlockAddressMap AddrToBlock;
    SymbolAddressMap AddrToSyms;
  };

  Error processBlock(ParseContext &PC, Block &B);
  Error processCIE(ParseContext &PC, Block &B, size_t RecordOffset,
                   size_t RecordLength, size_t CIEDeltaFieldOffset);
  Error processFDE(ParseContext &PC, Block &B, size_t RecordOffset,
                   size_t RecordLength, size_t CIEDeltaFieldOffset,
                   uint32_t CIEDelta, BlockEdgeMap &BlockEdges);

  Expected<AugmentationInfo>
  parseAugmentationString(BinaryStreamReader &RecordReader);

  static bool isSupportedPointerEncoding(uint8_t PointerEncoding);
  unsigned getPointerEncodingDataSize(uint8_t PointerEncoding);
  Expected<std::pair<orc::ExecutorAddr, Edge::Kind>>
  readEncodedPointer(uint8_t PointerEncoding,
                     orc::ExecutorAddr PointerFieldAddress,
                     BinaryStreamReader &RecordReader);

  Expected<Symbol &> getOrCreateSymbol(ParseContext &PC,
                                       orc::ExecutorAddr Addr);

  StringRef EHFrameSectionName;
  unsigned PointerSize;
  Edge::Kind Delta64;
  Edge::Kind Delta32;
  Edge::Kind NegDelta32;
};

/// Add a 32-bit null-terminator to the end of the eh-frame section.
class EHFrameNullTerminator {
public:
  EHFrameNullTerminator(StringRef EHFrameSectionName);
  Error operator()(LinkGraph &G);

private:
  static char NullTerminatorBlockContent[];
  StringRef EHFrameSectionName;
};

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORTIMPL_H
