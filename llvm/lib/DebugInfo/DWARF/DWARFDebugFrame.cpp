//===- DWARFDebugFrame.h - Parsing of .debug_frame ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFCFIPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFExpressionPrinter.h"
#include "llvm/DebugInfo/DWARF/DWARFUnwindTablePrinter.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFCFIProgram.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFExpression.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <optional>

using namespace llvm;
using namespace dwarf;

Expected<UnwindTable> llvm::dwarf::createUnwindTable(const FDE *Fde) {
  const CIE *Cie = Fde->getLinkedCIE();
  if (Cie == nullptr)
    return createStringError(errc::invalid_argument,
                             "unable to get CIE for FDE at offset 0x%" PRIx64,
                             Fde->getOffset());

  // Rows will be empty if there are no CFI instructions.
  if (Cie->cfis().empty() && Fde->cfis().empty())
    return UnwindTable({});

  UnwindTable::RowContainer CieRows;
  UnwindRow Row;
  Row.setAddress(Fde->getInitialLocation());
  if (Error CieError = parseRows(Cie->cfis(), Row, nullptr).moveInto(CieRows))
    return std::move(CieError);
  // We need to save the initial locations of registers from the CIE parsing
  // in case we run into DW_CFA_restore or DW_CFA_restore_extended opcodes.
  UnwindTable::RowContainer FdeRows;
  const RegisterLocations InitialLocs = Row.getRegisterLocations();
  if (Error FdeError =
          parseRows(Fde->cfis(), Row, &InitialLocs).moveInto(FdeRows))
    return std::move(FdeError);

  UnwindTable::RowContainer AllRows;
  AllRows.insert(AllRows.end(), CieRows.begin(), CieRows.end());
  AllRows.insert(AllRows.end(), FdeRows.begin(), FdeRows.end());

  // May be all the CFI instructions were DW_CFA_nop amd Row becomes empty.
  // Do not add that to the unwind table.
  if (Row.getRegisterLocations().hasLocations() ||
      Row.getCFAValue().getLocation() != UnwindLocation::Unspecified)
    AllRows.push_back(Row);
  return UnwindTable(std::move(AllRows));
}

Expected<UnwindTable> llvm::dwarf::createUnwindTable(const CIE *Cie) {
  // Rows will be empty if there are no CFI instructions.
  if (Cie->cfis().empty())
    return UnwindTable({});

  UnwindTable::RowContainer Rows;
  UnwindRow Row;
  if (Error CieError = parseRows(Cie->cfis(), Row, nullptr).moveInto(Rows))
    return std::move(CieError);
  // May be all the CFI instructions were DW_CFA_nop amd Row becomes empty.
  // Do not add that to the unwind table.
  if (Row.getRegisterLocations().hasLocations() ||
      Row.getCFAValue().getLocation() != UnwindLocation::Unspecified)
    Rows.push_back(Row);
  return UnwindTable(std::move(Rows));
}

// Returns the CIE identifier to be used by the requested format.
// CIE ids for .debug_frame sections are defined in Section 7.24 of DWARFv5.
// For CIE ID in .eh_frame sections see
// https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
constexpr uint64_t getCIEId(bool IsDWARF64, bool IsEH) {
  if (IsEH)
    return 0;
  if (IsDWARF64)
    return DW64_CIE_ID;
  return DW_CIE_ID;
}

void CIE::dump(raw_ostream &OS, DIDumpOptions DumpOpts) const {
  // A CIE with a zero length is a terminator entry in the .eh_frame section.
  if (DumpOpts.IsEH && Length == 0) {
    OS << format("%08" PRIx64, Offset) << " ZERO terminator\n";
    return;
  }

  OS << format("%08" PRIx64, Offset)
     << format(" %0*" PRIx64, IsDWARF64 ? 16 : 8, Length)
     << format(" %0*" PRIx64, IsDWARF64 && !DumpOpts.IsEH ? 16 : 8,
               getCIEId(IsDWARF64, DumpOpts.IsEH))
     << " CIE\n"
     << "  Format:                " << FormatString(IsDWARF64) << "\n";
  if (DumpOpts.IsEH && Version != 1)
    OS << "WARNING: unsupported CIE version\n";
  OS << format("  Version:               %d\n", Version)
     << "  Augmentation:          \"" << Augmentation << "\"\n";
  if (Version >= 4) {
    OS << format("  Address size:          %u\n", (uint32_t)AddressSize);
    OS << format("  Segment desc size:     %u\n",
                 (uint32_t)SegmentDescriptorSize);
  }
  OS << format("  Code alignment factor: %u\n", (uint32_t)CodeAlignmentFactor);
  OS << format("  Data alignment factor: %d\n", (int32_t)DataAlignmentFactor);
  OS << format("  Return address column: %d\n", (int32_t)ReturnAddressRegister);
  if (Personality)
    OS << format("  Personality Address: %016" PRIx64 "\n", *Personality);
  if (!AugmentationData.empty()) {
    OS << "  Augmentation data:    ";
    for (uint8_t Byte : AugmentationData)
      OS << ' ' << hexdigit(Byte >> 4) << hexdigit(Byte & 0xf);
    OS << "\n";
  }
  OS << "\n";
  printCFIProgram(CFIs, OS, DumpOpts, /*IndentLevel=*/1,
                  /*InitialLocation=*/{});
  OS << "\n";

  if (Expected<UnwindTable> RowsOrErr = createUnwindTable(this))
    printUnwindTable(*RowsOrErr, OS, DumpOpts, 1);
  else {
    DumpOpts.RecoverableErrorHandler(joinErrors(
        createStringError(errc::invalid_argument,
                          "decoding the CIE opcodes into rows failed"),
        RowsOrErr.takeError()));
  }
  OS << "\n";
}

void FDE::dump(raw_ostream &OS, DIDumpOptions DumpOpts) const {
  OS << format("%08" PRIx64, Offset)
     << format(" %0*" PRIx64, IsDWARF64 ? 16 : 8, Length)
     << format(" %0*" PRIx64, IsDWARF64 && !DumpOpts.IsEH ? 16 : 8, CIEPointer)
     << " FDE cie=";
  if (LinkedCIE)
    OS << format("%08" PRIx64, LinkedCIE->getOffset());
  else
    OS << "<invalid offset>";
  OS << format(" pc=%08" PRIx64 "...%08" PRIx64 "\n", InitialLocation,
               InitialLocation + AddressRange);
  OS << "  Format:       " << FormatString(IsDWARF64) << "\n";
  if (LSDAAddress)
    OS << format("  LSDA Address: %016" PRIx64 "\n", *LSDAAddress);
  printCFIProgram(CFIs, OS, DumpOpts, /*IndentLevel=*/1, InitialLocation);
  OS << "\n";

  if (Expected<UnwindTable> RowsOrErr = createUnwindTable(this))
    printUnwindTable(*RowsOrErr, OS, DumpOpts, 1);
  else {
    DumpOpts.RecoverableErrorHandler(joinErrors(
        createStringError(errc::invalid_argument,
                          "decoding the FDE opcodes into rows failed"),
        RowsOrErr.takeError()));
  }
  OS << "\n";
}

DWARFDebugFrame::DWARFDebugFrame(Triple::ArchType Arch,
    bool IsEH, uint64_t EHFrameAddress)
    : Arch(Arch), IsEH(IsEH), EHFrameAddress(EHFrameAddress) {}

DWARFDebugFrame::~DWARFDebugFrame() = default;

[[maybe_unused]] static void dumpDataAux(DataExtractor Data, uint64_t Offset,
                                         int Length) {
  errs() << "DUMP: ";
  for (int i = 0; i < Length; ++i) {
    uint8_t c = Data.getU8(&Offset);
    errs().write_hex(c); errs() << " ";
  }
  errs() << "\n";
}

Error DWARFDebugFrame::parse(DWARFDataExtractor Data) {
  uint64_t Offset = 0;
  DenseMap<uint64_t, CIE *> CIEs;

  while (Data.isValidOffset(Offset)) {
    uint64_t StartOffset = Offset;

    uint64_t Length;
    DwarfFormat Format;
    std::tie(Length, Format) = Data.getInitialLength(&Offset);
    bool IsDWARF64 = Format == DWARF64;

    // If the Length is 0, then this CIE is a terminator. We add it because some
    // dumper tools might need it to print something special for such entries
    // (e.g. llvm-objdump --dwarf=frames prints "ZERO terminator").
    if (Length == 0) {
      auto Cie = std::make_unique<CIE>(
          IsDWARF64, StartOffset, 0, 0, SmallString<8>(), 0, 0, 0, 0, 0,
          SmallString<8>(), 0, 0, std::nullopt, std::nullopt, Arch);
      CIEs[StartOffset] = Cie.get();
      Entries.push_back(std::move(Cie));
      break;
    }

    // At this point, Offset points to the next field after Length.
    // Length is the structure size excluding itself. Compute an offset one
    // past the end of the structure (needed to know how many instructions to
    // read).
    uint64_t StartStructureOffset = Offset;
    uint64_t EndStructureOffset = Offset + Length;

    // The Id field's size depends on the DWARF format
    Error Err = Error::success();
    uint64_t Id = Data.getRelocatedValue((IsDWARF64 && !IsEH) ? 8 : 4, &Offset,
                                         /*SectionIndex=*/nullptr, &Err);
    if (Err)
      return Err;

    if (Id == getCIEId(IsDWARF64, IsEH)) {
      uint8_t Version = Data.getU8(&Offset);
      const char *Augmentation = Data.getCStr(&Offset);
      StringRef AugmentationString(Augmentation ? Augmentation : "");
      uint8_t AddressSize = Version < 4 ? Data.getAddressSize() :
                                          Data.getU8(&Offset);
      Data.setAddressSize(AddressSize);
      uint8_t SegmentDescriptorSize = Version < 4 ? 0 : Data.getU8(&Offset);
      uint64_t CodeAlignmentFactor = Data.getULEB128(&Offset);
      int64_t DataAlignmentFactor = Data.getSLEB128(&Offset);
      uint64_t ReturnAddressRegister =
          Version == 1 ? Data.getU8(&Offset) : Data.getULEB128(&Offset);

      // Parse the augmentation data for EH CIEs
      StringRef AugmentationData("");
      uint32_t FDEPointerEncoding = DW_EH_PE_absptr;
      uint32_t LSDAPointerEncoding = DW_EH_PE_omit;
      std::optional<uint64_t> Personality;
      std::optional<uint32_t> PersonalityEncoding;
      if (IsEH) {
        std::optional<uint64_t> AugmentationLength;
        uint64_t StartAugmentationOffset;
        uint64_t EndAugmentationOffset;

        // Walk the augmentation string to get all the augmentation data.
        for (unsigned i = 0, e = AugmentationString.size(); i != e; ++i) {
          switch (AugmentationString[i]) {
          default:
            return createStringError(
                errc::invalid_argument,
                "unknown augmentation character %c in entry at 0x%" PRIx64,
                AugmentationString[i], StartOffset);
          case 'L':
            LSDAPointerEncoding = Data.getU8(&Offset);
            break;
          case 'P': {
            if (Personality)
              return createStringError(
                  errc::invalid_argument,
                  "duplicate personality in entry at 0x%" PRIx64, StartOffset);
            PersonalityEncoding = Data.getU8(&Offset);
            Personality = Data.getEncodedPointer(
                &Offset, *PersonalityEncoding,
                EHFrameAddress ? EHFrameAddress + Offset : 0);
            break;
          }
          case 'R':
            FDEPointerEncoding = Data.getU8(&Offset);
            break;
          case 'S':
            // Current frame is a signal trampoline.
            break;
          case 'z':
            if (i)
              return createStringError(
                  errc::invalid_argument,
                  "'z' must be the first character at 0x%" PRIx64, StartOffset);
            // Parse the augmentation length first.  We only parse it if
            // the string contains a 'z'.
            AugmentationLength = Data.getULEB128(&Offset);
            StartAugmentationOffset = Offset;
            EndAugmentationOffset = Offset + *AugmentationLength;
            break;
          case 'B':
            // B-Key is used for signing functions associated with this
            // augmentation string
            break;
            // This stack frame contains MTE tagged data, so needs to be
            // untagged on unwind.
          case 'G':
            break;
          }
        }

        if (AugmentationLength) {
          if (Offset != EndAugmentationOffset)
            return createStringError(errc::invalid_argument,
                                     "parsing augmentation data at 0x%" PRIx64
                                     " failed",
                                     StartOffset);
          AugmentationData = Data.getData().slice(StartAugmentationOffset,
                                                  EndAugmentationOffset);
        }
      }

      auto Cie = std::make_unique<CIE>(
          IsDWARF64, StartOffset, Length, Version, AugmentationString,
          AddressSize, SegmentDescriptorSize, CodeAlignmentFactor,
          DataAlignmentFactor, ReturnAddressRegister, AugmentationData,
          FDEPointerEncoding, LSDAPointerEncoding, Personality,
          PersonalityEncoding, Arch);
      CIEs[StartOffset] = Cie.get();
      Entries.emplace_back(std::move(Cie));
    } else {
      // FDE
      uint64_t CIEPointer = Id;
      uint64_t InitialLocation = 0;
      uint64_t AddressRange = 0;
      std::optional<uint64_t> LSDAAddress;
      CIE *Cie = CIEs[IsEH ? (StartStructureOffset - CIEPointer) : CIEPointer];

      if (IsEH) {
        // The address size is encoded in the CIE we reference.
        if (!Cie)
          return createStringError(errc::invalid_argument,
                                   "parsing FDE data at 0x%" PRIx64
                                   " failed due to missing CIE",
                                   StartOffset);
        if (auto Val =
                Data.getEncodedPointer(&Offset, Cie->getFDEPointerEncoding(),
                                       EHFrameAddress + Offset)) {
          InitialLocation = *Val;
        }
        if (auto Val = Data.getEncodedPointer(
                &Offset, Cie->getFDEPointerEncoding(), 0)) {
          AddressRange = *Val;
        }

        StringRef AugmentationString = Cie->getAugmentationString();
        if (!AugmentationString.empty()) {
          // Parse the augmentation length and data for this FDE.
          uint64_t AugmentationLength = Data.getULEB128(&Offset);

          uint64_t EndAugmentationOffset = Offset + AugmentationLength;

          // Decode the LSDA if the CIE augmentation string said we should.
          if (Cie->getLSDAPointerEncoding() != DW_EH_PE_omit) {
            LSDAAddress = Data.getEncodedPointer(
                &Offset, Cie->getLSDAPointerEncoding(),
                EHFrameAddress ? Offset + EHFrameAddress : 0);
          }

          if (Offset != EndAugmentationOffset)
            return createStringError(errc::invalid_argument,
                                     "parsing augmentation data at 0x%" PRIx64
                                     " failed",
                                     StartOffset);
        }
      } else {
        InitialLocation = Data.getRelocatedAddress(&Offset);
        AddressRange = Data.getRelocatedAddress(&Offset);
      }

      Entries.emplace_back(new FDE(IsDWARF64, StartOffset, Length, CIEPointer,
                                   InitialLocation, AddressRange, Cie,
                                   LSDAAddress, Arch));
    }

    if (Error E =
            Entries.back()->cfis().parse(Data, &Offset, EndStructureOffset))
      return E;

    if (Offset != EndStructureOffset)
      return createStringError(
          errc::invalid_argument,
          "parsing entry instructions at 0x%" PRIx64 " failed", StartOffset);
  }

  return Error::success();
}

FrameEntry *DWARFDebugFrame::getEntryAtOffset(uint64_t Offset) const {
  auto It = partition_point(Entries, [=](const std::unique_ptr<FrameEntry> &E) {
    return E->getOffset() < Offset;
  });
  if (It != Entries.end() && (*It)->getOffset() == Offset)
    return It->get();
  return nullptr;
}

void DWARFDebugFrame::dump(raw_ostream &OS, DIDumpOptions DumpOpts,
                           std::optional<uint64_t> Offset) const {
  DumpOpts.IsEH = IsEH;
  if (Offset) {
    if (auto *Entry = getEntryAtOffset(*Offset))
      Entry->dump(OS, DumpOpts);
    return;
  }

  OS << "\n";
  for (const auto &Entry : Entries)
    Entry->dump(OS, DumpOpts);
}
