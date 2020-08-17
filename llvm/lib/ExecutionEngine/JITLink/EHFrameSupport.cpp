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
                                   Edge::Kind FDEToCIE, Edge::Kind FDEToPCBegin,
                                   Edge::Kind FDEToLSDA)
    : EHFrameSectionName(EHFrameSectionName), FDEToCIE(FDEToCIE),
      FDEToPCBegin(FDEToPCBegin), FDEToLSDA(FDEToLSDA) {}

Error EHFrameEdgeFixer::operator()(LinkGraph &G) {
  auto *EHFrame = G.findSectionByName(EHFrameSectionName);

  if (!EHFrame) {
    LLVM_DEBUG({
      dbgs() << "EHFrameEdgeFixer: No " << EHFrameSectionName
             << " section. Nothing to do\n";
    });
    return Error::success();
  }

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
  using namespace dwarf;

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
      if (LSDAPointerEncoding != (DW_EH_PE_pcrel | DW_EH_PE_absptr))
        return make_error<JITLinkError>(
            "Unsupported LSDA pointer encoding " +
            formatv("{0:x2}", LSDAPointerEncoding) + " in CIE at " +
            formatv("{0:x16}", CIESymbol.getAddress()));
      break;
    }
    case 'P': {
      uint8_t PersonalityPointerEncoding = 0;
      if (auto Err = RecordReader.readInteger(PersonalityPointerEncoding))
        return Err;
      if (PersonalityPointerEncoding !=
          (DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4))
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
      if (FDEPointerEncoding != (DW_EH_PE_pcrel | DW_EH_PE_absptr))
        return make_error<JITLinkError>(
            "Unsupported FDE address pointer "
            "encoding " +
            formatv("{0:x2}", FDEPointerEncoding) + " in CIE at " +
            formatv("{0:x16}", CIESymbol.getAddress()));
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
      B.addEdge(FDEToCIE, RecordOffset + CIEDeltaFieldOffset,
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
      auto PCBeginDelta = readAbsolutePointer(PC.G, RecordReader);
      if (!PCBeginDelta)
        return PCBeginDelta.takeError();
      JITTargetAddress PCBegin =
          RecordAddress + PCBeginFieldOffset + *PCBeginDelta;
      LLVM_DEBUG({
        dbgs() << "        Adding edge at "
               << formatv("{0:x16}", RecordAddress + PCBeginFieldOffset)
               << " to PC at " << formatv("{0:x16}", PCBegin) << "\n";
      });
      auto PCBeginSym = getOrCreateSymbol(PC, PCBegin);
      if (!PCBeginSym)
        return PCBeginSym.takeError();
      B.addEdge(FDEToPCBegin, RecordOffset + PCBeginFieldOffset, *PCBeginSym,
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
      if (auto Err = RecordReader.skip(PC.G.getPointerSize()))
        return Err;
    }

    // Add a keep-alive edge from the FDE target to the FDE to ensure that the
    // FDE is kept alive if its target is.
    assert(PCBeginBlock && "PC-begin block not recorded");
    PCBeginBlock->addEdge(Edge::KeepAlive, 0, FDESymbol, 0);
  }

  // Skip over the PC range size field.
  if (auto Err = RecordReader.skip(PC.G.getPointerSize()))
    return Err;

  if (CIEInfo->FDEsHaveLSDAField) {
    uint64_t AugmentationDataSize;
    if (auto Err = RecordReader.readULEB128(AugmentationDataSize))
      return Err;
    if (AugmentationDataSize != PC.G.getPointerSize())
      return make_error<JITLinkError>(
          "Unexpected FDE augmentation data size (expected " +
          Twine(PC.G.getPointerSize()) + ", got " +
          Twine(AugmentationDataSize) + ") for FDE at " +
          formatv("{0:x16}", RecordAddress));

    JITTargetAddress LSDAFieldOffset = RecordReader.getOffset();
    auto LSDAEdgeItr = BlockEdges.find(RecordOffset + LSDAFieldOffset);
    if (LSDAEdgeItr == BlockEdges.end()) {
      auto LSDADelta = readAbsolutePointer(PC.G, RecordReader);
      if (!LSDADelta)
        return LSDADelta.takeError();
      JITTargetAddress LSDA = RecordAddress + LSDAFieldOffset + *LSDADelta;
      auto LSDASym = getOrCreateSymbol(PC, LSDA);
      if (!LSDASym)
        return LSDASym.takeError();
      LLVM_DEBUG({
        dbgs() << "        Adding edge at "
               << formatv("{0:x16}", RecordAddress + LSDAFieldOffset)
               << " to LSDA at " << formatv("{0:x16}", LSDA) << "\n";
      });
      B.addEdge(FDEToLSDA, RecordOffset + LSDAFieldOffset, *LSDASym, 0);
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
      if (auto Err = RecordReader.skip(PC.G.getPointerSize()))
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

Expected<JITTargetAddress>
EHFrameEdgeFixer::readAbsolutePointer(LinkGraph &G,
                                      BinaryStreamReader &RecordReader) {
  static_assert(sizeof(JITTargetAddress) == sizeof(uint64_t),
                "Result must be able to hold a uint64_t");
  JITTargetAddress Addr;
  if (G.getPointerSize() == 8) {
    if (auto Err = RecordReader.readInteger(Addr))
      return std::move(Err);
  } else if (G.getPointerSize() == 4) {
    uint32_t Addr32;
    if (auto Err = RecordReader.readInteger(Addr32))
      return std::move(Err);
    Addr = Addr32;
  } else
    llvm_unreachable("Pointer size is not 32-bit or 64-bit");
  return Addr;
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

// Determine whether we can register EH tables.
#if (defined(__GNUC__) && !defined(__ARM_EABI__) && !defined(__ia64__) &&      \
     !(defined(_AIX) && defined(__ibmxl__)) && !defined(__MVS__) &&            \
     !defined(__SEH__) && !defined(__USING_SJLJ_EXCEPTIONS__))
#define HAVE_EHTABLE_SUPPORT 1
#else
#define HAVE_EHTABLE_SUPPORT 0
#endif

#if HAVE_EHTABLE_SUPPORT
extern "C" void __register_frame(const void *);
extern "C" void __deregister_frame(const void *);

Error registerFrameWrapper(const void *P) {
  __register_frame(P);
  return Error::success();
}

Error deregisterFrameWrapper(const void *P) {
  __deregister_frame(P);
  return Error::success();
}

#else

// The building compiler does not have __(de)register_frame but
// it may be found at runtime in a dynamically-loaded library.
// For example, this happens when building LLVM with Visual C++
// but using the MingW runtime.
static Error registerFrameWrapper(const void *P) {
  static void((*RegisterFrame)(const void *)) = 0;

  if (!RegisterFrame)
    *(void **)&RegisterFrame =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("__register_frame");

  if (RegisterFrame) {
    RegisterFrame(P);
    return Error::success();
  }

  return make_error<JITLinkError>("could not register eh-frame: "
                                  "__register_frame function not found");
}

static Error deregisterFrameWrapper(const void *P) {
  static void((*DeregisterFrame)(const void *)) = 0;

  if (!DeregisterFrame)
    *(void **)&DeregisterFrame =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
            "__deregister_frame");

  if (DeregisterFrame) {
    DeregisterFrame(P);
    return Error::success();
  }

  return make_error<JITLinkError>("could not deregister eh-frame: "
                                  "__deregister_frame function not found");
}
#endif

#ifdef __APPLE__

template <typename HandleFDEFn>
Error walkAppleEHFrameSection(const char *const SectionStart,
                              size_t SectionSize,
                              HandleFDEFn HandleFDE) {
  const char *CurCFIRecord = SectionStart;
  const char *End = SectionStart + SectionSize;
  uint64_t Size = *reinterpret_cast<const uint32_t *>(CurCFIRecord);

  while (CurCFIRecord != End && Size != 0) {
    const char *OffsetField = CurCFIRecord + (Size == 0xffffffff ? 12 : 4);
    if (Size == 0xffffffff)
      Size = *reinterpret_cast<const uint64_t *>(CurCFIRecord + 4) + 12;
    else
      Size += 4;
    uint32_t Offset = *reinterpret_cast<const uint32_t *>(OffsetField);

    LLVM_DEBUG({
      dbgs() << "Registering eh-frame section:\n";
      dbgs() << "Processing " << (Offset ? "FDE" : "CIE") << " @"
             << (void *)CurCFIRecord << ": [";
      for (unsigned I = 0; I < Size; ++I)
        dbgs() << format(" 0x%02" PRIx8, *(CurCFIRecord + I));
      dbgs() << " ]\n";
    });

    if (Offset != 0)
      if (auto Err = HandleFDE(CurCFIRecord))
        return Err;

    CurCFIRecord += Size;

    Size = *reinterpret_cast<const uint32_t *>(CurCFIRecord);
  }

  return Error::success();
}

#endif // __APPLE__

Error registerEHFrameSection(const void *EHFrameSectionAddr,
                             size_t EHFrameSectionSize) {
#ifdef __APPLE__
  // On Darwin __register_frame has to be called for each FDE entry.
  return walkAppleEHFrameSection(static_cast<const char *>(EHFrameSectionAddr),
                                 EHFrameSectionSize,
                                 registerFrameWrapper);
#else
  // On Linux __register_frame takes a single argument:
  // a pointer to the start of the .eh_frame section.

  // How can it find the end? Because crtendS.o is linked
  // in and it has an .eh_frame section with four zero chars.
  return registerFrameWrapper(EHFrameSectionAddr);
#endif
}

Error deregisterEHFrameSection(const void *EHFrameSectionAddr,
                               size_t EHFrameSectionSize) {
#ifdef __APPLE__
  return walkAppleEHFrameSection(static_cast<const char *>(EHFrameSectionAddr),
                                 EHFrameSectionSize,
                                 deregisterFrameWrapper);
#else
  return deregisterFrameWrapper(EHFrameSectionAddr);
#endif
}

EHFrameRegistrar::~EHFrameRegistrar() {}

InProcessEHFrameRegistrar &InProcessEHFrameRegistrar::getInstance() {
  static InProcessEHFrameRegistrar Instance;
  return Instance;
}

InProcessEHFrameRegistrar::InProcessEHFrameRegistrar() {}

LinkGraphPassFunction
createEHFrameRecorderPass(const Triple &TT,
                          StoreFrameRangeFunction StoreRangeAddress) {
  const char *EHFrameSectionName = nullptr;
  if (TT.getObjectFormat() == Triple::MachO)
    EHFrameSectionName = "__eh_frame";
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
      return make_error<JITLinkError>("__eh_frame section can not have zero "
                                      "address with non-zero size");
    StoreFrameRange(Addr, Size);
    return Error::success();
  };

  return RecordEHFrame;
}

} // end namespace jitlink
} // end namespace llvm
