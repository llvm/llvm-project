//===- CompactUnwindSupportImpl.h - Compact Unwind format impl --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compact Unwind format support implementation details.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_EXECUTIONENGINE_JITLINK_COMPACTUNWINDSUPPORTIMPL_H
#define LIB_EXECUTIONENGINE_JITLINK_COMPACTUNWINDSUPPORTIMPL_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITLink/MachO.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"

#define DEBUG_TYPE "jitlink_cu"

namespace llvm {
namespace jitlink {

/// Split blocks in an __LD,__compact_unwind section on record boundaries.
/// When this function returns edges within each record are guaranteed to be
/// sorted by offset.
Error splitCompactUnwindBlocks(LinkGraph &G, Section &CompactUnwindSection,
                               size_t RecordSize);

/// CRTP base for compact unwind traits classes. Automatically provides derived
/// constants.
///
/// FIXME: Passing PtrSize as a template parameter is a hack to work around a
///        bug in older MSVC compilers (until at least MSVC 15) where constexpr
///        fields in the CRTP impl class were not visible to the base class.
///        Once we no longer need to support these compilers the PtrSize
///        template argument should be removed and PointerSize should be
///        defined as a member in the CRTP Impl classes.
template <typename CRTPImpl, size_t PtrSize> struct CompactUnwindTraits {
  static constexpr size_t PointerSize = PtrSize;
  static constexpr size_t Size = 3 * PointerSize + 2 * 4;
  static constexpr size_t FnFieldOffset = 0;
  static constexpr size_t SizeFieldOffset = FnFieldOffset + PointerSize;
  static constexpr size_t EncodingFieldOffset = SizeFieldOffset + 4;
  static constexpr size_t PersonalityFieldOffset = EncodingFieldOffset + 4;
  static constexpr size_t LSDAFieldOffset =
      PersonalityFieldOffset + PointerSize;

  static uint32_t readPCRangeSize(ArrayRef<char> RecordContent) {
    assert(SizeFieldOffset + 4 <= RecordContent.size() &&
           "Truncated CU record?");
    return support::endian::read32<CRTPImpl::Endianness>(RecordContent.data() +
                                                         SizeFieldOffset);
  }

  static uint32_t readEncoding(ArrayRef<char> RecordContent) {
    assert(EncodingFieldOffset + 4 <= RecordContent.size() &&
           "Truncated CU record?");
    return support::endian::read32<CRTPImpl::Endianness>(RecordContent.data() +
                                                         EncodingFieldOffset);
  }
};

/// Architecture specific implementation of CompactUnwindManager.
template <typename CURecTraits> class CompactUnwindManager {
public:
  CompactUnwindManager(StringRef CompactUnwindSectionName,
                       StringRef UnwindInfoSectionName,
                       StringRef EHFrameSectionName)
      : CompactUnwindSectionName(CompactUnwindSectionName),
        UnwindInfoSectionName(UnwindInfoSectionName),
        EHFrameSectionName(EHFrameSectionName) {}

  // Split compact unwind records, add keep-alive edges from functions to
  // compact unwind records, and from compact unwind records to FDEs where
  // needed.
  //
  // This method must be called *after* __eh_frame has been processed: it
  // assumes that eh-frame records have been split up and keep-alive edges have
  // been inserted.
  Error prepareForPrune(LinkGraph &G) {
    Section *CUSec = G.findSectionByName(CompactUnwindSectionName);
    if (!CUSec || CUSec->empty()) {
      LLVM_DEBUG({
        dbgs() << "Compact unwind: No compact unwind info for " << G.getName()
               << "\n";
      });
      return Error::success();
    }

    LLVM_DEBUG({
      dbgs() << "Compact unwind: preparing " << G.getName() << " for prune\n";
    });

    Section *EHFrameSec = G.findSectionByName(EHFrameSectionName);

    if (auto Err = splitCompactUnwindBlocks(G, *CUSec, CURecTraits::Size))
      return Err;

    LLVM_DEBUG({
      dbgs() << "  Preparing " << CUSec->blocks_size() << " blocks in "
             << CompactUnwindSectionName << "\n";
    });

    for (auto *B : CUSec->blocks()) {

      // Find target function edge.
      Edge *PCBeginEdge = nullptr;
      for (auto &E : B->edges_at(CURecTraits::FnFieldOffset)) {
        PCBeginEdge = &E;
        break;
      }

      if (!PCBeginEdge)
        return make_error<JITLinkError>(
            "In " + G.getName() + ", compact unwind record at " +
            formatv("{0:x}", B->getAddress()) + " has no pc-begin edge");

      if (!PCBeginEdge->getTarget().isDefined())
        return make_error<JITLinkError>(
            "In " + G.getName() + ", compact unwind record at " +
            formatv("{0:x}", B->getAddress()) + " points at external symbol " +
            *PCBeginEdge->getTarget().getName());

      auto &Fn = PCBeginEdge->getTarget();

      if (!Fn.isDefined()) {
        LLVM_DEBUG({
          dbgs() << "In " << CompactUnwindSectionName << " for " << G.getName()
                 << " encountered unexpected pc-edge to undefined symbol "
                 << Fn.getName() << "\n";
        });
        continue;
      } else {
        LLVM_DEBUG({
          dbgs() << "    Found record for function ";
          if (Fn.hasName())
            dbgs() << Fn.getName();
          else
            dbgs() << "<anon @ " << Fn.getAddress() << '>';
          dbgs() << '\n';
        });
      }

      bool NeedsDWARF = CURecTraits::encodingSpecifiesDWARF(
          CURecTraits::readEncoding(B->getContent()));

      auto &CURecSym =
          G.addAnonymousSymbol(*B, 0, CURecTraits::Size, false, false);

      bool KeepAliveAlreadyPresent = false;
      if (EHFrameSec) {
        Edge *KeepAliveEdge = nullptr;
        for (auto &E : Fn.getBlock().edges_at(0)) {
          if (E.getKind() == Edge::KeepAlive && E.getTarget().isDefined() &&
              &E.getTarget().getSection() == EHFrameSec) {
            KeepAliveEdge = &E;
            break;
          }
        }

        if (KeepAliveEdge) {
          // Found a keep-alive edge to an FDE in the eh-frame. Switch the keep
          // alive edge to point to the CU and if the CU needs DWARF then add
          // an extra keep-alive edge from the CU to the FDE.
          auto &FDE = KeepAliveEdge->getTarget();
          KeepAliveEdge->setTarget(CURecSym);
          KeepAliveAlreadyPresent = true;
          if (NeedsDWARF) {
            LLVM_DEBUG({
              dbgs() << "      Needs DWARF: adding keep-alive edge to FDE at "
                     << FDE.getAddress() << "\n";
            });
            B->addEdge(Edge::KeepAlive, 0, FDE, 0);
          }
        } else {
          if (NeedsDWARF)
            return make_error<JITLinkError>(
                "In " + G.getName() + ", compact unwind recard ot " +
                formatv("{0:x}", B->getAddress()) +
                " needs DWARF, but no FDE was found");
        }
      } else {
        if (NeedsDWARF)
          return make_error<JITLinkError>(
              "In " + G.getName() + ", compact unwind recard ot " +
              formatv("{0:x}", B->getAddress()) + " needs DWARF, but no " +
              EHFrameSectionName + " section exists");
      }

      if (!KeepAliveAlreadyPresent) {
        // No FDE edge. We'll need to add a new edge from the function back
        // to the CU record.
        Fn.getBlock().addEdge(Edge::KeepAlive, 0, CURecSym, 0);
      }
    }

    return Error::success();
  }

  /// Process all __compact_unwind records and reserve space for __unwind_info.
  Error processAndReserveUnwindInfo(LinkGraph &G) {
    // Bail out early if no unwind info.
    Section *CUSec = G.findSectionByName(CompactUnwindSectionName);
    if (!CUSec)
      return Error::success();

    // The __LD/__compact_unwind section is only used as input for the linker.
    // We'll create a new __TEXT,__unwind_info section for unwind info output.
    CUSec->setMemLifetime(orc::MemLifetime::NoAlloc);

    // Find / make a mach-header to act as the base for unwind-info offsets
    // (and to report the arch / subarch to libunwind).
    if (auto Err = getOrCreateCompactUnwindBase(G))
      return Err;

    // Error out if there's already unwind-info in the graph: We have no idea
    // how to merge unwind-info sections.
    if (G.findSectionByName(UnwindInfoSectionName))
      return make_error<JITLinkError>("In " + G.getName() + ", " +
                                      UnwindInfoSectionName +
                                      " already exists");

    // Process the __compact_unwind section to build the Records vector that
    // we'll use for writing the __unwind_info section.
    if (auto Err = processCompactUnwind(G, *CUSec))
      return Err;

    // Calculate the size of __unwind_info.
    size_t UnwindInfoSectionSize =
        UnwindInfoSectionHeaderSize +
        Personalities.size() * PersonalityEntrySize +
        (NumSecondLevelPages + 1) * IndexEntrySize + NumLSDAs * LSDAEntrySize +
        NumSecondLevelPages * SecondLevelPageHeaderSize +
        Records.size() * SecondLevelPageEntrySize;

    LLVM_DEBUG({
      dbgs() << "In " << G.getName() << ", reserving "
             << formatv("{0:x}", UnwindInfoSectionSize) << " bytes for "
             << UnwindInfoSectionName << "\n";
    });

    // Create the __unwind_info section and reserve space for it.
    Section &UnwindInfoSec =
        G.createSection(UnwindInfoSectionName, orc::MemProt::Read);

    auto UnwindInfoSectionContent = G.allocateBuffer(UnwindInfoSectionSize);
    memset(UnwindInfoSectionContent.data(), 0, UnwindInfoSectionContent.size());
    auto &B = G.createMutableContentBlock(
        UnwindInfoSec, UnwindInfoSectionContent, orc::ExecutorAddr(), 8, 0);

    // Add Keep-alive edges from the __unwind_info block to all of the target
    // functions.
    for (auto &R : Records)
      B.addEdge(Edge::KeepAlive, 0, *R.Fn, 0);

    return Error::success();
  }

  Error writeUnwindInfo(LinkGraph &G) {
    Section *CUSec = G.findSectionByName(CompactUnwindSectionName);
    if (!CUSec || CUSec->empty())
      return Error::success();

    Section *UnwindInfoSec = G.findSectionByName(UnwindInfoSectionName);
    if (!UnwindInfoSec)
      return make_error<JITLinkError>("In " + G.getName() + ", " +
                                      UnwindInfoSectionName +
                                      " missing after allocation");

    if (UnwindInfoSec->blocks_size() != 1)
      return make_error<JITLinkError>(
          "In " + G.getName() + ", " + UnwindInfoSectionName +
          " contains more than one block post-allocation");

    LLVM_DEBUG(
        { dbgs() << "Writing unwind info for " << G.getName() << "...\n"; });

    mergeRecords();

    auto &UnwindInfoBlock = **UnwindInfoSec->blocks().begin();
    auto Content = UnwindInfoBlock.getMutableContent(G);
    BinaryStreamWriter Writer(
        {reinterpret_cast<uint8_t *>(Content.data()), Content.size()},
        CURecTraits::Endianness);

    // __unwind_info format, from mach-o/compact_unwind_encoding.h on Darwin:
    //
    // #define UNWIND_SECTION_VERSION 1
    // struct unwind_info_section_header
    // {
    //     uint32_t    version;            // UNWIND_SECTION_VERSION
    //     uint32_t    commonEncodingsArraySectionOffset;
    //     uint32_t    commonEncodingsArrayCount;
    //     uint32_t    personalityArraySectionOffset;
    //     uint32_t    personalityArrayCount;
    //     uint32_t    indexSectionOffset;
    //     uint32_t    indexCount;
    //     // compact_unwind_encoding_t[]
    //     // uint32_t personalities[]
    //     // unwind_info_section_header_index_entry[]
    //     // unwind_info_section_header_lsda_index_entry[]
    // };

    if (auto Err = writeHeader(G, Writer))
      return Err;

    // Skip common encodings: JITLink doesn't use them.

    if (auto Err = writePersonalities(G, Writer))
      return Err;

    // Calculate the offset to the LSDAs.
    size_t SectionOffsetToLSDAs =
        Writer.getOffset() + (NumSecondLevelPages + 1) * IndexEntrySize;

    // Calculate offset to the 1st second-level page.
    size_t SectionOffsetToSecondLevelPages =
        SectionOffsetToLSDAs + NumLSDAs * LSDAEntrySize;

    if (auto Err = writeIndexes(G, Writer, SectionOffsetToLSDAs,
                                SectionOffsetToSecondLevelPages))
      return Err;

    if (auto Err = writeLSDAs(G, Writer))
      return Err;

    if (auto Err = writeSecondLevelPages(G, Writer))
      return Err;

    LLVM_DEBUG({
      dbgs() << "    Wrote " << formatv("{0:x}", Writer.getOffset())
             << " bytes of unwind info.\n";
    });

    return Error::success();
  }

private:
  // Calculate the size of unwind-info.
  static constexpr size_t MaxPersonalities = 4;
  static constexpr size_t PersonalityShift = 28;

  static constexpr size_t UnwindInfoSectionHeaderSize = 4 * 7;
  static constexpr size_t PersonalityEntrySize = 4;
  static constexpr size_t IndexEntrySize = 3 * 4;
  static constexpr size_t LSDAEntrySize = 2 * 4;
  static constexpr size_t SecondLevelPageSize = 4096;
  static constexpr size_t SecondLevelPageHeaderSize = 8;
  static constexpr size_t SecondLevelPageEntrySize = 8;
  static constexpr size_t NumRecordsPerSecondLevelPage =
      (SecondLevelPageSize - SecondLevelPageHeaderSize) /
      SecondLevelPageEntrySize;

  struct CompactUnwindRecord {
    Symbol *Fn = nullptr;
    uint32_t Size = 0;
    uint32_t Encoding = 0;
    Symbol *LSDA = nullptr;
    Symbol *FDE = nullptr;
  };

  Error processCompactUnwind(LinkGraph &G, Section &CUSec) {
    // TODO: Reset NumLSDAs, Personalities and CompactUnwindRecords if
    // processing more than once.
    assert(NumLSDAs == 0 && "NumLSDAs should be zero");
    assert(Records.empty() && "CompactUnwindRecords vector should be empty.");
    assert(Personalities.empty() && "Personalities vector should be empty.");

    SmallVector<CompactUnwindRecord> NonUniquedRecords;
    NonUniquedRecords.reserve(CUSec.blocks_size());

    // Process __compact_unwind blocks.
    for (auto *B : CUSec.blocks()) {
      CompactUnwindRecord R;
      R.Encoding = CURecTraits::readEncoding(B->getContent());
      for (auto &E : B->edges()) {
        switch (E.getOffset()) {
        case CURecTraits::FnFieldOffset:
          // This could be the function-pointer, or the FDE keep-alive. Check
          // the type to decide.
          if (E.getKind() == Edge::KeepAlive)
            R.FDE = &E.getTarget();
          else
            R.Fn = &E.getTarget();
          break;
        case CURecTraits::PersonalityFieldOffset: {
          // Add the Personality to the Personalities map and update the
          // encoding.
          size_t PersonalityIdx = 0;
          for (; PersonalityIdx != Personalities.size(); ++PersonalityIdx)
            if (Personalities[PersonalityIdx] == &E.getTarget())
              break;
          if (PersonalityIdx == MaxPersonalities)
            return make_error<JITLinkError>(
                "In " + G.getName() +
                ", __compact_unwind contains too many personalities (max " +
                formatv("{}", MaxPersonalities) + ")");
          if (PersonalityIdx == Personalities.size())
            Personalities.push_back(&E.getTarget());

          R.Encoding |= (PersonalityIdx + 1) << PersonalityShift;
          break;
        }
        case CURecTraits::LSDAFieldOffset:
          ++NumLSDAs;
          R.LSDA = &E.getTarget();
          break;
        default:
          return make_error<JITLinkError>("In " + G.getName() +
                                          ", compact unwind record at " +
                                          formatv("{0:x}", B->getAddress()) +
                                          " has unrecognized edge at offset " +
                                          formatv("{0:x}", E.getOffset()));
        }
      }
      Records.push_back(R);
    }

    // Sort the records into ascending order.
    llvm::sort(Records, [](const CompactUnwindRecord &LHS,
                           const CompactUnwindRecord &RHS) {
      return LHS.Fn->getAddress() < RHS.Fn->getAddress();
    });

    // Calculate the number of second-level pages required.
    NumSecondLevelPages = (Records.size() + NumRecordsPerSecondLevelPage - 1) /
                          NumRecordsPerSecondLevelPage;

    // Convert personality symbols to GOT entry pointers.
    typename CURecTraits::GOTManager GOT(G);
    for (auto &Personality : Personalities)
      Personality = &GOT.getEntryForTarget(G, *Personality);

    LLVM_DEBUG({
      dbgs() << "  In " << G.getName() << ", " << CompactUnwindSectionName
             << ": raw records = " << Records.size()
             << ", personalities = " << Personalities.size()
             << ", lsdas = " << NumLSDAs << "\n";
    });

    return Error::success();
  }

  void mergeRecords() {
    SmallVector<CompactUnwindRecord> NonUniqued = std::move(Records);
    Records.reserve(NonUniqued.size());

    Records.push_back(NonUniqued.front());
    for (size_t I = 1; I != NonUniqued.size(); ++I) {
      auto &Next = NonUniqued[I];
      auto &Last = Records.back();

      bool NextNeedsDWARF = CURecTraits::encodingSpecifiesDWARF(Next.Encoding);
      bool CannotBeMerged = CURecTraits::encodingCannotBeMerged(Next.Encoding);
      if (NextNeedsDWARF || (Next.Encoding != Last.Encoding) ||
          CannotBeMerged || Next.LSDA || Last.LSDA)
        Records.push_back(Next);
    }

    // Recalculate derived values that may have changed.
    NumSecondLevelPages = (Records.size() + NumRecordsPerSecondLevelPage - 1) /
                          NumRecordsPerSecondLevelPage;
  }

  Error writeHeader(LinkGraph &G, BinaryStreamWriter &W) {
    if (!isUInt<32>(NumSecondLevelPages + 1))
      return make_error<JITLinkError>("In " + G.getName() + ", too many " +
                                      UnwindInfoSectionName +
                                      "second-level pages required");

    // Write __unwind_info header.
    size_t IndexArrayOffset = UnwindInfoSectionHeaderSize +
                              Personalities.size() * PersonalityEntrySize;

    cantFail(W.writeInteger<uint32_t>(1));
    cantFail(W.writeInteger<uint32_t>(UnwindInfoSectionHeaderSize));
    cantFail(W.writeInteger<uint32_t>(0));
    cantFail(W.writeInteger<uint32_t>(UnwindInfoSectionHeaderSize));
    cantFail(W.writeInteger<uint32_t>(Personalities.size()));
    cantFail(W.writeInteger<uint32_t>(IndexArrayOffset));
    cantFail(W.writeInteger<uint32_t>(NumSecondLevelPages + 1));

    return Error::success();
  }

  Error writePersonalities(LinkGraph &G, BinaryStreamWriter &W) {
    // Write personalities.
    for (auto *PSym : Personalities) {
      auto Delta = PSym->getAddress() - CompactUnwindBase->getAddress();
      if (!isUInt<32>(Delta))
        return makePersonalityRangeError(G, *PSym);
      cantFail(W.writeInteger<uint32_t>(Delta));
    }
    return Error::success();
  }

  Error writeIndexes(LinkGraph &G, BinaryStreamWriter &W,
                     size_t SectionOffsetToLSDAs,
                     size_t SectionOffsetToSecondLevelPages) {
    // Assume that function deltas are ok in this method -- we'll error
    // check all of them when we write the second level pages.

    // Write the header index entries.
    size_t RecordIdx = 0;
    size_t NumPreviousLSDAs = 0;
    for (auto &R : Records) {
      // If this record marks the start of a new second level page.
      if (RecordIdx % NumRecordsPerSecondLevelPage == 0) {
        auto FnDelta = R.Fn->getAddress() - CompactUnwindBase->getAddress();
        auto SecondLevelPageOffset =
            SectionOffsetToSecondLevelPages +
            SecondLevelPageSize * (RecordIdx / NumRecordsPerSecondLevelPage);
        auto LSDAOffset =
            SectionOffsetToLSDAs + NumPreviousLSDAs * LSDAEntrySize;

        cantFail(W.writeInteger<uint32_t>(FnDelta));
        cantFail(W.writeInteger<uint32_t>(SecondLevelPageOffset));
        cantFail(W.writeInteger<uint32_t>(LSDAOffset));
      }
      if (R.LSDA)
        ++NumPreviousLSDAs;
      ++RecordIdx;
    }

    // Write the index array terminator.
    {
      auto FnEndDelta =
          Records.back().Fn->getRange().End - CompactUnwindBase->getAddress();

      if (LLVM_UNLIKELY(!isUInt<32>(FnEndDelta)))
        return make_error<JITLinkError>(
            "In " + G.getName() + " " + UnwindInfoSectionName +
            ", delta to end of functions  " +
            formatv("{0:x}", Records.back().Fn->getRange().End) +
            " exceeds 32 bits");

      cantFail(W.writeInteger<uint32_t>(FnEndDelta));
      cantFail(W.writeInteger<uint32_t>(0));
      cantFail(W.writeInteger<uint32_t>(SectionOffsetToSecondLevelPages));
    }

    return Error::success();
  }

  Error writeLSDAs(LinkGraph &G, BinaryStreamWriter &W) {
    // As with writeIndexes, assume that function deltas are ok for now.
    for (auto &R : Records) {
      if (R.LSDA) {
        auto FnDelta = R.Fn->getAddress() - CompactUnwindBase->getAddress();
        auto LSDADelta = R.LSDA->getAddress() - CompactUnwindBase->getAddress();

        if (LLVM_UNLIKELY(!isUInt<32>(LSDADelta)))
          return make_error<JITLinkError>(
              "In " + G.getName() + " " + UnwindInfoSectionName +
              ", delta to lsda at " + formatv("{0:x}", R.LSDA->getAddress()) +
              " exceeds 32 bits");

        cantFail(W.writeInteger<uint32_t>(FnDelta));
        cantFail(W.writeInteger<uint32_t>(LSDADelta));
      }
    }

    return Error::success();
  }

  Error writeSecondLevelPages(LinkGraph &G, BinaryStreamWriter &W) {
    size_t RecordIdx = 0;

    for (auto &R : Records) {
      // When starting a new second-level page, write the page header:
      //
      //   2     : uint32_t    -- UNWIND_SECOND_LEVEL_REGULAR
      //   8     : uint16_t    -- size of second level page table header
      //   count : uint16_t    -- num entries in this second-level page
      if (RecordIdx % NumRecordsPerSecondLevelPage == 0) {
        constexpr uint32_t SecondLevelPageHeaderKind = 2;
        constexpr uint16_t SecondLevelPageHeaderSize = 8;
        uint16_t SecondLevelPageNumEntries =
            std::min(Records.size() - RecordIdx, NumRecordsPerSecondLevelPage);

        cantFail(W.writeInteger<uint32_t>(SecondLevelPageHeaderKind));
        cantFail(W.writeInteger<uint16_t>(SecondLevelPageHeaderSize));
        cantFail(W.writeInteger<uint16_t>(SecondLevelPageNumEntries));
      }

      // Write entry.
      auto FnDelta = R.Fn->getAddress() - CompactUnwindBase->getAddress();

      if (LLVM_UNLIKELY(!isUInt<32>(FnDelta)))
        return make_error<JITLinkError>(
            "In " + G.getName() + " " + UnwindInfoSectionName +
            ", delta to function at " + formatv("{0:x}", R.Fn->getAddress()) +
            " exceeds 32 bits");

      cantFail(W.writeInteger<uint32_t>(FnDelta));
      cantFail(W.writeInteger<uint32_t>(R.Encoding));

      ++RecordIdx;
    }

    return Error::success();
  }

  Error getOrCreateCompactUnwindBase(LinkGraph &G) {
    auto Name = G.intern("__jitlink$libunwind_dso_base");
    CompactUnwindBase = G.findAbsoluteSymbolByName(Name);
    if (!CompactUnwindBase) {
      if (auto LocalCUBase = getOrCreateLocalMachOHeader(G)) {
        CompactUnwindBase = &*LocalCUBase;
        auto &B = LocalCUBase->getBlock();
        G.addDefinedSymbol(B, 0, *Name, B.getSize(), Linkage::Strong,
                           Scope::Local, false, true);
      } else
        return LocalCUBase.takeError();
    }
    CompactUnwindBase->setLive(true);
    return Error::success();
  }

  Error makePersonalityRangeError(LinkGraph &G, Symbol &PSym) {
    std::string ErrMsg;
    {
      raw_string_ostream ErrStream(ErrMsg);
      ErrStream << "In " << G.getName() << " " << UnwindInfoSectionName
                << ", personality ";
      if (PSym.hasName())
        ErrStream << PSym.getName() << " ";
      ErrStream << "at " << PSym.getAddress()
                << " is out of 32-bit delta range of compact-unwind base at "
                << CompactUnwindBase->getAddress();
    }
    return make_error<JITLinkError>(std::move(ErrMsg));
  }

  StringRef CompactUnwindSectionName;
  StringRef UnwindInfoSectionName;
  StringRef EHFrameSectionName;
  Symbol *CompactUnwindBase = nullptr;

  size_t NumLSDAs = 0;
  size_t NumSecondLevelPages = 0;
  SmallVector<Symbol *, MaxPersonalities> Personalities;
  SmallVector<CompactUnwindRecord> Records;
};

} // end namespace jitlink
} // end namespace llvm

#undef DEBUG_TYPE

#endif // LIB_EXECUTIONENGINE_JITLINK_COMPACTUNWINDSUPPORTIMPL_H
