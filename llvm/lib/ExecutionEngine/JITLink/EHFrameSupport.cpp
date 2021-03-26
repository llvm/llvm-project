//===-------- JITLink_EHFrameSupport.cpp - JITLink eh-frame utils ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "EHFrameSupportImpl.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/Support/DynamicLibrary.h"

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

EHFrameSplitter::EHFrameSplitter(StringRef EHFrameSectionName)
    : EHFrameSectionName(EHFrameSectionName) {}

Error EHFrameSplitter::operator()(LinkGraph &G) {
  auto *EHFrame = G.findSectionByName(EHFrameSectionName);

  if (!EHFrame) {
    LLVM_DEBUG({
      dbgs() << "EHFrameSplitter: No " << EHFrameSectionName
             << " section. Nothing to do\n";
    });
    return Error::success();
  }

  LLVM_DEBUG({
    dbgs() << "EHFrameSplitter: Processing " << EHFrameSectionName << "...\n";
  });

  DenseMap<Block *, LinkGraph::SplitBlockCache> Caches;

  {
    // Pre-build the split caches.
    for (auto *B : EHFrame->blocks())
      Caches[B] = LinkGraph::SplitBlockCache::value_type();
    for (auto *Sym : EHFrame->symbols())
      Caches[&Sym->getBlock()]->push_back(Sym);
    for (auto *B : EHFrame->blocks())
      llvm::sort(*Caches[B], [](const Symbol *LHS, const Symbol *RHS) {
        return LHS->getOffset() > RHS->getOffset();
      });
  }

  // Iterate over blocks (we do this by iterating over Caches entries rather
  // than EHFrame->blocks() as we will be inserting new blocks along the way,
  // which would invalidate iterators in the latter sequence.
  for (auto &KV : Caches) {
    auto &B = *KV.first;
    auto &BCache = KV.second;
    if (auto Err = processBlock(G, B, BCache))
      return Err;
  }

  return Error::success();
}

Error EHFrameSplitter::processBlock(LinkGraph &G, Block &B,
                                    LinkGraph::SplitBlockCache &Cache) {
  LLVM_DEBUG({
    dbgs() << "  Processing block at " << formatv("{0:x16}", B.getAddress())
           << "\n";
  });

  // eh-frame should not contain zero-fill blocks.
  if (B.isZeroFill())
    return make_error<JITLinkError>("Unexpected zero-fill block in " +
                                    EHFrameSectionName + " section");

  if (B.getSize() == 0) {
    LLVM_DEBUG(dbgs() << "    Block is empty. Skipping.\n");
    return Error::success();
  }

  BinaryStreamReader BlockReader(B.getContent(), G.getEndianness());

  while (true) {
    uint64_t RecordStartOffset = BlockReader.getOffset();

    LLVM_DEBUG({
      dbgs() << "    Processing CFI record at "
             << formatv("{0:x16}", B.getAddress()) << "\n";
    });

    uint32_t Length;
    if (auto Err = BlockReader.readInteger(Length))
      return Err;
    if (Length != 0xffffffff) {
      if (auto Err = BlockReader.skip(Length))
        return Err;
    } else {
      uint64_t ExtendedLength;
      if (auto Err = BlockReader.readInteger(ExtendedLength))
        return Err;
      if (auto Err = BlockReader.skip(ExtendedLength))
        return Err;
    }

    // If this was the last block then there's nothing to split
    if (BlockReader.empty()) {
      LLVM_DEBUG(dbgs() << "      Extracted " << B << "\n");
      return Error::success();
    }

    uint64_t BlockSize = BlockReader.getOffset() - RecordStartOffset;
    auto &NewBlock = G.splitBlock(B, BlockSize);
    (void)NewBlock;
    LLVM_DEBUG(dbgs() << "      Extracted " << NewBlock << "\n");
  }
}

EHFrameEdgeFixer::EHFrameEdgeFixer(StringRef EHFrameSectionName,
                                   unsigned PointerSize, Edge::Kind Delta64,
                                   Edge::Kind Delta32, Edge::Kind NegDelta32)
    : EHFrameSectionName(EHFrameSectionName), PointerSize(PointerSize),
      Delta64(Delta64), Delta32(Delta32), NegDelta32(NegDelta32) {}

Error EHFrameEdgeFixer::operator()(LinkGraph &G) {
  auto *EHFrame = G.findSectionByName(EHFrameSectionName);

  if (!EHFrame) {
    LLVM_DEBUG({
      dbgs() << "EHFrameEdgeFixer: No " << EHFrameSectionName
             << " section. Nothing to do\n";
    });
    return Error::success();
  }

  // Check that we support the graph's pointer size.
  if (G.getPointerSize() != 4 && G.getPointerSize() != 8)
    return make_error<JITLinkError>(
        "EHFrameEdgeFixer only supports 32 and 64 bit targets");

  LLVM_DEBUG({
    dbgs() << "EHFrameEdgeFixer: Processing " << EHFrameSectionName << "...\n";
  });

  ParseContext PC(G);

  // Build a map of all blocks and symbols in the text sections. We will use
  // these for finding / building edge targets when processing FDEs.
  for (auto &Sec : G.sections()) {
    PC.AddrToSyms.addSymbols(Sec.symbols());
    if (auto Err = PC.AddrToBlock.addBlocks(Sec.blocks(),
                                            BlockAddressMap::includeNonNull))
      return Err;
  }

  // Sort eh-frame blocks into address order to ensure we visit CIEs before
  // their child FDEs.
  std::vector<Block *> EHFrameBlocks;
  for (auto *B : EHFrame->blocks())
    EHFrameBlocks.push_back(B);
  llvm::sort(EHFrameBlocks, [](const Block *LHS, const Block *RHS) {
    return LHS->getAddress() < RHS->getAddress();
  });

  // Loop over the blocks in address order.
  for (auto *B : EHFrameBlocks)
    if (auto Err = processBlock(PC, *B))
      return Err;

  return Error::success();
}

Error EHFrameEdgeFixer::processBlock(ParseContext &PC, Block &B) {

  LLVM_DEBUG({
    dbgs() << "  Processing block at " << formatv("{0:x16}", B.getAddress())
           << "\n";
  });

  // eh-frame should not contain zero-fill blocks.
  if (B.isZeroFill())
    return make_error<JITLinkError>("Unexpected zero-fill block in " +
                                    EHFrameSectionName + " section");

  if (B.getSize() == 0) {
    LLVM_DEBUG(dbgs() << "    Block is empty. Skipping.\n");
    return Error::success();
  }

  // Find the offsets of any existing edges from this block.
  BlockEdgeMap BlockEdges;
  for (auto &E : B.edges())
    if (E.isRelocation()) {
      if (BlockEdges.count(E.getOffset()))
        return make_error<JITLinkError>(
            "Multiple relocations at offset " +
            formatv("{0:x16}", E.getOffset()) + " in " + EHFrameSectionName +
            " block at address " + formatv("{0:x16}", B.getAddress()));

      BlockEdges[E.getOffset()] = EdgeTarget(E);
    }

  CIEInfosMap CIEInfos;
  BinaryStreamReader BlockReader(B.getContent(), PC.G.getEndianness());
  while (!BlockReader.empty()) {
    size_t RecordStartOffset = BlockReader.getOffset();

    LLVM_DEBUG({
      dbgs() << "    Processing CFI record at "
             << formatv("{0:x16}", B.getAddress() + RecordStartOffset) << "\n";
    });

    // Get the record length.
    size_t RecordRemaining;
    {
      uint32_t Length;
      if (auto Err = BlockReader.readInteger(Length))
        return Err;
      // If Length < 0xffffffff then use the regular length field, otherwise
      // read the extended length field.
      if (Length != 0xffffffff)
        RecordRemaining = Length;
      else {
        uint64_t ExtendedLength;
        if (auto Err = BlockReader.readInteger(ExtendedLength))
          return Err;
        RecordRemaining = ExtendedLength;
      }
    }

    if (BlockReader.bytesRemaining() < RecordRemaining)
      return make_error<JITLinkError>(
          "Incomplete CFI record at " +
          formatv("{0:x16}", B.getAddress() + RecordStartOffset));

    // Read the CIE delta for this record.
    uint64_t CIEDeltaFieldOffset = BlockReader.getOffset() - RecordStartOffset;
    uint32_t CIEDelta;
    if (auto Err = BlockReader.readInteger(CIEDelta))
      return Err;

    if (CIEDelta == 0) {
      if (auto Err = processCIE(PC, B, RecordStartOffset,
                                CIEDeltaFieldOffset + RecordRemaining,
                                CIEDeltaFieldOffset))
        return Err;
    } else {
      if (auto Err = processFDE(PC, B, RecordStartOffset,
                                CIEDeltaFieldOffset + RecordRemaining,
                                CIEDeltaFieldOffset, CIEDelta, BlockEdges))
        return Err;
    }

    // Move to the next record.
    BlockReader.setOffset(RecordStartOffset + CIEDeltaFieldOffset +
                          RecordRemaining);
  }

  return Error::success();
}

Error EHFrameEdgeFixer::processCIE(ParseContext &PC, Block &B,
                                   size_t RecordOffset, size_t RecordLength,
                                   size_t CIEDeltaFieldOffset) {

  LLVM_DEBUG(dbgs() << "      Record is CIE\n");

  auto RecordContent = B.getContent().substr(RecordOffset, RecordLength);
  BinaryStreamReader RecordReader(RecordContent, PC.G.getEndianness());

  // Skip past the CIE delta field: we've already processed this far.
  RecordReader.setOffset(CIEDeltaFieldOffset + 4);

  auto &CIESymbol =
      PC.G.addAnonymousSymbol(B, RecordOffset, RecordLength, false, false);
  CIEInformation CIEInfo(CIESymbol);

  uint8_t Version = 0;
  if (auto Err = RecordReader.readInteger(Version))
    return Err;

  if (Version != 0x01)
    return make_error<JITLinkError>("Bad CIE version " + Twine(Version) +
                                    " (should be 0x01) in eh-frame");

  auto AugInfo = parseAugmentationString(RecordReader);
  if (!AugInfo)
    return AugInfo.takeError();

  // Skip the EH Data field if present.
  if (AugInfo->EHDataFieldPresent)
    if (auto Err = RecordReader.skip(PC.G.getPointerSize()))
      return Err;

  // Read and sanity check the code alignment factor.
  {
    uint64_t CodeAlignmentFactor = 0;
    if (auto Err = RecordReader.readULEB128(CodeAlignmentFactor))
      return Err;
    if (CodeAlignmentFactor != 1)
      return make_error<JITLinkError>("Unsupported CIE code alignment factor " +
                                      Twine(CodeAlignmentFactor) +
                                      " (expected 1)");
  }

  // Read and sanity check the data alignment factor.
  {
    int64_t DataAlignmentFactor = 0;
    if (auto Err = RecordReader.readSLEB128(DataAlignmentFactor))
      return Err;
    if (DataAlignmentFactor != -8)
      return make_error<JITLinkError>("Unsupported CIE data alignment factor " +
                                      Twine(DataAlignmentFactor) +
                                      " (expected -8)");
  }

  // Skip the return address register field.
  if (auto Err = RecordReader.skip(1))
    return Err;

  uint64_t AugmentationDataLength = 0;
  if (auto Err = RecordReader.readULEB128(AugmentationDataLength))
    return Err;

  uint32_t AugmentationDataStartOffset = RecordReader.getOffset();

  uint8_t *NextField = &AugInfo->Fields[0];
  while (uint8_t Field = *NextField++) {
    switch (Field) {
    case 'L': {
      CIEInfo.FDEsHaveLSDAField = true;
      uint8_t LSDAPointerEncoding;
      if (auto Err = RecordReader.readInteger(LSDAPointerEncoding))
        return Err;
      if (!isSupportedPointerEncoding(LSDAPointerEncoding))
        return make_error<JITLinkError>(
            "Unsupported LSDA pointer encoding " +
            formatv("{0:x2}", LSDAPointerEncoding) + " in CIE at " +
            formatv("{0:x16}", CIESymbol.getAddress()));
      CIEInfo.LSDAPointerEncoding = LSDAPointerEncoding;
      break;
    }
    case 'P': {
      uint8_t PersonalityPointerEncoding = 0;
      if (auto Err = RecordReader.readInteger(PersonalityPointerEncoding))
        return Err;
      if (PersonalityPointerEncoding !=
          (dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |
           dwarf::DW_EH_PE_sdata4))
        return make_error<JITLinkError>(
            "Unspported personality pointer "
            "encoding " +
            formatv("{0:x2}", PersonalityPointerEncoding) + " in CIE at " +
            formatv("{0:x16}", CIESymbol.getAddress()));
      uint32_t PersonalityPointerAddress;
      if (auto Err = RecordReader.readInteger(PersonalityPointerAddress))
        return Err;
      break;
    }
    case 'R': {
      uint8_t FDEPointerEncoding;
      if (auto Err = RecordReader.readInteger(FDEPointerEncoding))
        return Err;
      if (!isSupportedPointerEncoding(FDEPointerEncoding))
        return make_error<JITLinkError>(
            "Unsupported FDE pointer encoding " +
            formatv("{0:x2}", FDEPointerEncoding) + " in CIE at " +
            formatv("{0:x16}", CIESymbol.getAddress()));
      CIEInfo.FDEPointerEncoding = FDEPointerEncoding;
      break;
    }
    default:
      llvm_unreachable("Invalid augmentation string field");
    }
  }

  if (RecordReader.getOffset() - AugmentationDataStartOffset >
      AugmentationDataLength)
    return make_error<JITLinkError>("Read past the end of the augmentation "
                                    "data while parsing fields");

  assert(!PC.CIEInfos.count(CIESymbol.getAddress()) &&
         "Multiple CIEs recorded at the same address?");
  PC.CIEInfos[CIESymbol.getAddress()] = std::move(CIEInfo);

  return Error::success();
}

Error EHFrameEdgeFixer::processFDE(ParseContext &PC, Block &B,
                                   size_t RecordOffset, size_t RecordLength,
                                   size_t CIEDeltaFieldOffset,
                                   uint32_t CIEDelta,
                                   BlockEdgeMap &BlockEdges) {
  LLVM_DEBUG(dbgs() << "      Record is FDE\n");

  JITTargetAddress RecordAddress = B.getAddress() + RecordOffset;

  auto RecordContent = B.getContent().substr(RecordOffset, RecordLength);
  BinaryStreamReader RecordReader(RecordContent, PC.G.getEndianness());

  // Skip past the CIE delta field: we've already read this far.
  RecordReader.setOffset(CIEDeltaFieldOffset + 4);

  auto &FDESymbol =
      PC.G.addAnonymousSymbol(B, RecordOffset, RecordLength, false, false);

  CIEInformation *CIEInfo = nullptr;

  {
    // Process the CIE pointer field.
    auto CIEEdgeItr = BlockEdges.find(RecordOffset + CIEDeltaFieldOffset);
    JITTargetAddress CIEAddress =
        RecordAddress + CIEDeltaFieldOffset - CIEDelta;
    if (CIEEdgeItr == BlockEdges.end()) {

      LLVM_DEBUG({
        dbgs() << "        Adding edge at "
               << formatv("{0:x16}", RecordAddress + CIEDeltaFieldOffset)
               << " to CIE at: " << formatv("{0:x16}", CIEAddress) << "\n";
      });
      if (auto CIEInfoOrErr = PC.findCIEInfo(CIEAddress))
        CIEInfo = *CIEInfoOrErr;
      else
        return CIEInfoOrErr.takeError();
      assert(CIEInfo->CIESymbol && "CIEInfo has no CIE symbol set");
      B.addEdge(NegDelta32, RecordOffset + CIEDeltaFieldOffset,
                *CIEInfo->CIESymbol, 0);
    } else {
      LLVM_DEBUG({
        dbgs() << "        Already has edge at "
               << formatv("{0:x16}", RecordAddress + CIEDeltaFieldOffset)
               << " to CIE at " << formatv("{0:x16}", CIEAddress) << "\n";
      });
      auto &EI = CIEEdgeItr->second;
      if (EI.Addend)
        return make_error<JITLinkError>(
            "CIE edge at " +
            formatv("{0:x16}", RecordAddress + CIEDeltaFieldOffset) +
            " has non-zero addend");
      if (auto CIEInfoOrErr = PC.findCIEInfo(EI.Target->getAddress()))
        CIEInfo = *CIEInfoOrErr;
      else
        return CIEInfoOrErr.takeError();
    }
  }

  {
    // Process the PC-Begin field.
    Block *PCBeginBlock = nullptr;
    JITTargetAddress PCBeginFieldOffset = RecordReader.getOffset();
    auto PCEdgeItr = BlockEdges.find(RecordOffset + PCBeginFieldOffset);
    if (PCEdgeItr == BlockEdges.end()) {
      auto PCBeginPtrInfo =
          readEncodedPointer(CIEInfo->FDEPointerEncoding,
                             RecordAddress + PCBeginFieldOffset, RecordReader);
      if (!PCBeginPtrInfo)
        return PCBeginPtrInfo.takeError();
      JITTargetAddress PCBegin = PCBeginPtrInfo->first;
      Edge::Kind PCBeginEdgeKind = PCBeginPtrInfo->second;
      LLVM_DEBUG({
        dbgs() << "        Adding edge at "
               << formatv("{0:x16}", RecordAddress + PCBeginFieldOffset)
               << " to PC at " << formatv("{0:x16}", PCBegin) << "\n";
      });
      auto PCBeginSym = getOrCreateSymbol(PC, PCBegin);
      if (!PCBeginSym)
        return PCBeginSym.takeError();
      B.addEdge(PCBeginEdgeKind, RecordOffset + PCBeginFieldOffset, *PCBeginSym,
                0);
      PCBeginBlock = &PCBeginSym->getBlock();
    } else {
      auto &EI = PCEdgeItr->second;
      LLVM_DEBUG({
        dbgs() << "        Already has edge at "
               << formatv("{0:x16}", RecordAddress + PCBeginFieldOffset)
               << " to PC at " << formatv("{0:x16}", EI.Target->getAddress());
        if (EI.Addend)
          dbgs() << " + " << formatv("{0:x16}", EI.Addend);
        dbgs() << "\n";
      });

      // Make sure the existing edge points at a defined block.
      if (!EI.Target->isDefined()) {
        auto EdgeAddr = RecordAddress + PCBeginFieldOffset;
        return make_error<JITLinkError>("FDE edge at " +
                                        formatv("{0:x16}", EdgeAddr) +
                                        " points at external block");
      }
      PCBeginBlock = &EI.Target->getBlock();
      if (auto Err = RecordReader.skip(
              getPointerEncodingDataSize(CIEInfo->FDEPointerEncoding)))
        return Err;
    }

    // Add a keep-alive edge from the FDE target to the FDE to ensure that the
    // FDE is kept alive if its target is.
    assert(PCBeginBlock && "PC-begin block not recorded");
    LLVM_DEBUG({
      dbgs() << "        Adding keep-alive edge from target at "
             << formatv("{0:x16}", PCBeginBlock->getAddress()) << " to FDE at "
             << formatv("{0:x16}", RecordAddress) << "\n";
    });
    PCBeginBlock->addEdge(Edge::KeepAlive, 0, FDESymbol, 0);
  }

  // Skip over the PC range size field.
  if (auto Err = RecordReader.skip(
          getPointerEncodingDataSize(CIEInfo->FDEPointerEncoding)))
    return Err;

  if (CIEInfo->FDEsHaveLSDAField) {
    uint64_t AugmentationDataSize;
    if (auto Err = RecordReader.readULEB128(AugmentationDataSize))
      return Err;

    JITTargetAddress LSDAFieldOffset = RecordReader.getOffset();
    auto LSDAEdgeItr = BlockEdges.find(RecordOffset + LSDAFieldOffset);
    if (LSDAEdgeItr == BlockEdges.end()) {
      auto LSDAPointerInfo =
          readEncodedPointer(CIEInfo->LSDAPointerEncoding,
                             RecordAddress + LSDAFieldOffset, RecordReader);
      if (!LSDAPointerInfo)
        return LSDAPointerInfo.takeError();
      JITTargetAddress LSDA = LSDAPointerInfo->first;
      Edge::Kind LSDAEdgeKind = LSDAPointerInfo->second;
      auto LSDASym = getOrCreateSymbol(PC, LSDA);
      if (!LSDASym)
        return LSDASym.takeError();
      LLVM_DEBUG({
        dbgs() << "        Adding edge at "
               << formatv("{0:x16}", RecordAddress + LSDAFieldOffset)
               << " to LSDA at " << formatv("{0:x16}", LSDA) << "\n";
      });
      B.addEdge(LSDAEdgeKind, RecordOffset + LSDAFieldOffset, *LSDASym, 0);
    } else {
      LLVM_DEBUG({
        auto &EI = LSDAEdgeItr->second;
        dbgs() << "        Already has edge at "
               << formatv("{0:x16}", RecordAddress + LSDAFieldOffset)
               << " to LSDA at " << formatv("{0:x16}", EI.Target->getAddress());
        if (EI.Addend)
          dbgs() << " + " << formatv("{0:x16}", EI.Addend);
        dbgs() << "\n";
      });
      if (auto Err = RecordReader.skip(AugmentationDataSize))
        return Err;
    }
  } else {
    LLVM_DEBUG(dbgs() << "        Record does not have LSDA field.\n");
  }

  return Error::success();
}

Expected<EHFrameEdgeFixer::AugmentationInfo>
EHFrameEdgeFixer::parseAugmentationString(BinaryStreamReader &RecordReader) {
  AugmentationInfo AugInfo;
  uint8_t NextChar;
  uint8_t *NextField = &AugInfo.Fields[0];

  if (auto Err = RecordReader.readInteger(NextChar))
    return std::move(Err);

  while (NextChar != 0) {
    switch (NextChar) {
    case 'z':
      AugInfo.AugmentationDataPresent = true;
      break;
    case 'e':
      if (auto Err = RecordReader.readInteger(NextChar))
        return std::move(Err);
      if (NextChar != 'h')
        return make_error<JITLinkError>("Unrecognized substring e" +
                                        Twine(NextChar) +
                                        " in augmentation string");
      AugInfo.EHDataFieldPresent = true;
      break;
    case 'L':
    case 'P':
    case 'R':
      *NextField++ = NextChar;
      break;
    default:
      return make_error<JITLinkError>("Unrecognized character " +
                                      Twine(NextChar) +
                                      " in augmentation string");
    }

    if (auto Err = RecordReader.readInteger(NextChar))
      return std::move(Err);
  }

  return std::move(AugInfo);
}

bool EHFrameEdgeFixer::isSupportedPointerEncoding(uint8_t PointerEncoding) {
  using namespace dwarf;

  // We only support PC-rel for now.
  if ((PointerEncoding & 0x70) != DW_EH_PE_pcrel)
    return false;

  // readEncodedPointer does not handle indirect.
  if (PointerEncoding & DW_EH_PE_indirect)
    return false;

  // Supported datatypes.
  switch (PointerEncoding & 0xf) {
  case DW_EH_PE_absptr:
  case DW_EH_PE_udata4:
  case DW_EH_PE_udata8:
  case DW_EH_PE_sdata4:
  case DW_EH_PE_sdata8:
    return true;
  }

  return false;
}

unsigned EHFrameEdgeFixer::getPointerEncodingDataSize(uint8_t PointerEncoding) {
  using namespace dwarf;

  assert(isSupportedPointerEncoding(PointerEncoding) &&
         "Unsupported pointer encoding");
  switch (PointerEncoding & 0xf) {
  case DW_EH_PE_absptr:
    return PointerSize;
  case DW_EH_PE_udata4:
  case DW_EH_PE_sdata4:
    return 4;
  case DW_EH_PE_udata8:
  case DW_EH_PE_sdata8:
    return 8;
  default:
    llvm_unreachable("Unsupported encoding");
  }
}

Expected<std::pair<JITTargetAddress, Edge::Kind>>
EHFrameEdgeFixer::readEncodedPointer(uint8_t PointerEncoding,
                                     JITTargetAddress PointerFieldAddress,
                                     BinaryStreamReader &RecordReader) {
  static_assert(sizeof(JITTargetAddress) == sizeof(uint64_t),
                "Result must be able to hold a uint64_t");
  assert(isSupportedPointerEncoding(PointerEncoding) &&
         "Unsupported pointer encoding");

  using namespace dwarf;

  // Isolate data type, remap absptr to udata4 or udata8. This relies on us
  // having verified that the graph uses 32-bit or 64-bit pointers only at the
  // start of this pass.
  uint8_t EffectiveType = PointerEncoding & 0xf;
  if (EffectiveType == DW_EH_PE_absptr)
    EffectiveType = (PointerSize == 8) ? DW_EH_PE_udata8 : DW_EH_PE_udata4;

  JITTargetAddress Addr;
  Edge::Kind PointerEdgeKind;
  switch (EffectiveType) {
  case DW_EH_PE_udata4: {
    uint32_t Val;
    if (auto Err = RecordReader.readInteger(Val))
      return std::move(Err);
    Addr = PointerFieldAddress + Val;
    PointerEdgeKind = Delta32;
    break;
  }
  case DW_EH_PE_udata8: {
    uint64_t Val;
    if (auto Err = RecordReader.readInteger(Val))
      return std::move(Err);
    Addr = PointerFieldAddress + Val;
    PointerEdgeKind = Delta64;
    break;
  }
  case DW_EH_PE_sdata4: {
    int32_t Val;
    if (auto Err = RecordReader.readInteger(Val))
      return std::move(Err);
    Addr = PointerFieldAddress + Val;
    PointerEdgeKind = Delta32;
    break;
  }
  case DW_EH_PE_sdata8: {
    int64_t Val;
    if (auto Err = RecordReader.readInteger(Val))
      return std::move(Err);
    Addr = PointerFieldAddress + Val;
    PointerEdgeKind = Delta64;
    break;
  }
  }

  if (PointerEdgeKind == Edge::Invalid)
    return make_error<JITLinkError>(
        "Unspported edge kind for encoded pointer at " +
        formatv("{0:x}", PointerFieldAddress));

  return std::make_pair(Addr, Delta64);
}

Expected<Symbol &> EHFrameEdgeFixer::getOrCreateSymbol(ParseContext &PC,
                                                       JITTargetAddress Addr) {
  Symbol *CanonicalSym = nullptr;

  auto UpdateCanonicalSym = [&](Symbol *Sym) {
    if (!CanonicalSym || Sym->getLinkage() < CanonicalSym->getLinkage() ||
        Sym->getScope() < CanonicalSym->getScope() ||
        (Sym->hasName() && !CanonicalSym->hasName()) ||
        Sym->getName() < CanonicalSym->getName())
      CanonicalSym = Sym;
  };

  if (auto *SymbolsAtAddr = PC.AddrToSyms.getSymbolsAt(Addr))
    for (auto *Sym : *SymbolsAtAddr)
      UpdateCanonicalSym(Sym);

  // If we found an existing symbol at the given address then use it.
  if (CanonicalSym)
    return *CanonicalSym;

  // Otherwise search for a block covering the address and create a new symbol.
  auto *B = PC.AddrToBlock.getBlockCovering(Addr);
  if (!B)
    return make_error<JITLinkError>("No symbol or block covering address " +
                                    formatv("{0:x16}", Addr));

  return PC.G.addAnonymousSymbol(*B, Addr - B->getAddress(), 0, false, false);
}

char EHFrameNullTerminator::NullTerminatorBlockContent[] = {0, 0, 0, 0};

EHFrameNullTerminator::EHFrameNullTerminator(StringRef EHFrameSectionName)
    : EHFrameSectionName(EHFrameSectionName) {}

Error EHFrameNullTerminator::operator()(LinkGraph &G) {
  auto *EHFrame = G.findSectionByName(EHFrameSectionName);

  if (!EHFrame)
    return Error::success();

  LLVM_DEBUG({
    dbgs() << "EHFrameNullTerminator adding null terminator to "
           << EHFrameSectionName << "\n";
  });

  auto &NullTerminatorBlock =
      G.createContentBlock(*EHFrame, StringRef(NullTerminatorBlockContent, 4),
                           0xfffffffffffffffc, 1, 0);
  G.addAnonymousSymbol(NullTerminatorBlock, 0, 4, false, true);
  return Error::success();
}

EHFrameRegistrar::~EHFrameRegistrar() {}

Error InProcessEHFrameRegistrar::registerEHFrames(
    JITTargetAddress EHFrameSectionAddr, size_t EHFrameSectionSize) {
  return orc::registerEHFrameSection(
      jitTargetAddressToPointer<void *>(EHFrameSectionAddr),
      EHFrameSectionSize);
}

Error InProcessEHFrameRegistrar::deregisterEHFrames(
    JITTargetAddress EHFrameSectionAddr, size_t EHFrameSectionSize) {
  return orc::deregisterEHFrameSection(
      jitTargetAddressToPointer<void *>(EHFrameSectionAddr),
      EHFrameSectionSize);
}

LinkGraphPassFunction
createEHFrameRecorderPass(const Triple &TT,
                          StoreFrameRangeFunction StoreRangeAddress) {
  const char *EHFrameSectionName = nullptr;
  if (TT.getObjectFormat() == Triple::MachO)
    EHFrameSectionName = "__TEXT,__eh_frame";
  else
    EHFrameSectionName = ".eh_frame";

  auto RecordEHFrame =
      [EHFrameSectionName,
       StoreFrameRange = std::move(StoreRangeAddress)](LinkGraph &G) -> Error {
    // Search for a non-empty eh-frame and record the address of the first
    // symbol in it.
    JITTargetAddress Addr = 0;
    size_t Size = 0;
    if (auto *S = G.findSectionByName(EHFrameSectionName)) {
      auto R = SectionRange(*S);
      Addr = R.getStart();
      Size = R.getSize();
    }
    if (Addr == 0 && Size != 0)
      return make_error<JITLinkError>(
          StringRef(EHFrameSectionName) +
          " section can not have zero address with non-zero size");
    StoreFrameRange(Addr, Size);
    return Error::success();
  };

  return RecordEHFrame;
}

} // end namespace jitlink
} // end namespace llvm
