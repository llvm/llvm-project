//=== DebugInfoLinker.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DebugInfoLinker.h"
#include "Error.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/DWARFLinker/DWARFLinker.h"
#include "llvm/DWARFLinker/DWARFStreamer.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
#include <memory>
#include <vector>

namespace llvm {
namespace dwarfutil {

// ObjFileAddressMap allows to check whether specified DIE referencing
// dead addresses. It uses tombstone values to determine dead addresses.
// The concrete values of tombstone constants were discussed in
// https://reviews.llvm.org/D81784 and https://reviews.llvm.org/D84825.
// So we use following values as indicators of dead addresses:
//
// bfd: (LowPC == 0) or (LowPC == 1 and HighPC == 1 and  DWARF v4 (or less))
//      or ([LowPC, HighPC] is not inside address ranges of .text sections).
//
// maxpc: (LowPC == -1) or (LowPC == -2 and  DWARF v4 (or less))
//        That value is assumed to be compatible with
//        http://www.dwarfstd.org/ShowIssue.php?issue=200609.1
//
// exec: [LowPC, HighPC] is not inside address ranges of .text sections
//
// universal: maxpc and bfd
class ObjFileAddressMap : public AddressesMap {
public:
  ObjFileAddressMap(DWARFContext &Context, const Options &Options,
                    object::ObjectFile &ObjFile)
      : Opts(Options), Context(Context) {
    // Remember addresses of existing text sections.
    for (const object::SectionRef &Sect : ObjFile.sections()) {
      if (!Sect.isText())
        continue;
      const uint64_t Size = Sect.getSize();
      if (Size == 0)
        continue;
      const uint64_t StartAddr = Sect.getAddress();
      TextAddressRanges.insert({StartAddr, StartAddr + Size});
    }

    // Check CU address ranges for tombstone value.
    for (std::unique_ptr<DWARFUnit> &CU : Context.compile_units()) {
      Expected<llvm::DWARFAddressRangesVector> ARanges =
          CU->getUnitDIE().getAddressRanges();
      if (ARanges) {
        for (auto &Range : *ARanges) {
          if (!isDeadAddressRange(Range.LowPC, Range.HighPC, CU->getVersion(),
                                  Options.Tombstone, CU->getAddressByteSize()))
            DWARFAddressRanges.insert({Range.LowPC, Range.HighPC}, 0);
        }
      }
    }
  }

  // should be renamed into has valid address ranges
  bool hasValidRelocs() override { return !DWARFAddressRanges.empty(); }

  bool isLiveSubprogram(const DWARFDie &DIE,
                        CompileUnit::DIEInfo &Info) override {
    assert((DIE.getTag() == dwarf::DW_TAG_subprogram ||
            DIE.getTag() == dwarf::DW_TAG_label) &&
           "Wrong type of input die");

    if (Optional<uint64_t> LowPC =
            dwarf::toAddress(DIE.find(dwarf::DW_AT_low_pc))) {
      if (!isDeadAddress(*LowPC, DIE.getDwarfUnit()->getVersion(),
                         Opts.Tombstone,
                         DIE.getDwarfUnit()->getAddressByteSize())) {
        Info.AddrAdjust = 0;
        Info.InDebugMap = true;
        return true;
      }
    }

    return false;
  }

  bool isLiveVariable(const DWARFDie &DIE,
                      CompileUnit::DIEInfo &Info) override {
    assert((DIE.getTag() == dwarf::DW_TAG_variable ||
            DIE.getTag() == dwarf::DW_TAG_constant) &&
           "Wrong type of input die");

    if (Expected<DWARFLocationExpressionsVector> Loc =
            DIE.getLocations(dwarf::DW_AT_location)) {
      DWARFUnit *U = DIE.getDwarfUnit();
      for (const auto &Entry : *Loc) {
        DataExtractor Data(toStringRef(Entry.Expr),
                           U->getContext().isLittleEndian(), 0);
        DWARFExpression Expression(Data, U->getAddressByteSize(),
                                   U->getFormParams().Format);
        bool HasLiveAddresses =
            any_of(Expression, [&](const DWARFExpression::Operation &Op) {
              // TODO: add handling of dwarf::DW_OP_addrx
              return !Op.isError() &&
                     (Op.getCode() == dwarf::DW_OP_addr &&
                      !isDeadAddress(Op.getRawOperand(0), U->getVersion(),
                                     Opts.Tombstone,
                                     DIE.getDwarfUnit()->getAddressByteSize()));
            });

        if (HasLiveAddresses) {
          Info.AddrAdjust = 0;
          Info.InDebugMap = true;
          return true;
        }
      }
    } else {
      // FIXME: missing DW_AT_location is OK here, but other errors should be
      // reported to the user.
      consumeError(Loc.takeError());
    }

    return false;
  }

  bool applyValidRelocs(MutableArrayRef<char>, uint64_t, bool) override {
    // no need to apply relocations to the linked binary.
    return false;
  }

  RangesTy &getValidAddressRanges() override { return DWARFAddressRanges; };

  void clear() override { DWARFAddressRanges.clear(); }

  llvm::Expected<uint64_t> relocateIndexedAddr(uint64_t StartOffset,
                                               uint64_t EndOffset) override {
    // No relocations in linked binary. Return just address value.

    const char *AddrPtr =
        Context.getDWARFObj().getAddrSection().Data.data() + StartOffset;
    support::endianness Endianess =
        Context.getDWARFObj().isLittleEndian() ? support::little : support::big;

    assert(EndOffset > StartOffset);
    switch (EndOffset - StartOffset) {
    case 1:
      return *AddrPtr;
    case 2:
      return support::endian::read16(AddrPtr, Endianess);
    case 4:
      return support::endian::read32(AddrPtr, Endianess);
    case 8:
      return support::endian::read64(AddrPtr, Endianess);
    }

    llvm_unreachable("relocateIndexedAddr unhandled case!");
  }

protected:
  // returns true if specified address range is inside address ranges
  // of executable sections.
  bool isInsideExecutableSectionsAddressRange(uint64_t LowPC,
                                              Optional<uint64_t> HighPC) {
    Optional<AddressRange> Range =
        TextAddressRanges.getRangeThatContains(LowPC);

    if (HighPC)
      return Range.has_value() && Range->end() >= *HighPC;

    return Range.has_value();
  }

  uint64_t isBFDDeadAddressRange(uint64_t LowPC, Optional<uint64_t> HighPC,
                                 uint16_t Version) {
    if (LowPC == 0)
      return true;

    if ((Version <= 4) && HighPC && (LowPC == 1 && *HighPC == 1))
      return true;

    return !isInsideExecutableSectionsAddressRange(LowPC, HighPC);
  }

  uint64_t isMAXPCDeadAddressRange(uint64_t LowPC, Optional<uint64_t> HighPC,
                                   uint16_t Version, uint8_t AddressByteSize) {
    if (Version <= 4 && HighPC) {
      if (LowPC == (dwarf::computeTombstoneAddress(AddressByteSize) - 1))
        return true;
    } else if (LowPC == dwarf::computeTombstoneAddress(AddressByteSize))
      return true;

    if (!isInsideExecutableSectionsAddressRange(LowPC, HighPC))
      warning("Address referencing invalid text section is not marked with "
              "tombstone value");

    return false;
  }

  bool isDeadAddressRange(uint64_t LowPC, Optional<uint64_t> HighPC,
                          uint16_t Version, TombstoneKind Tombstone,
                          uint8_t AddressByteSize) {
    switch (Tombstone) {
    case TombstoneKind::BFD:
      return isBFDDeadAddressRange(LowPC, HighPC, Version);
    case TombstoneKind::MaxPC:
      return isMAXPCDeadAddressRange(LowPC, HighPC, Version, AddressByteSize);
    case TombstoneKind::Universal:
      return isBFDDeadAddressRange(LowPC, HighPC, Version) ||
             isMAXPCDeadAddressRange(LowPC, HighPC, Version, AddressByteSize);
    case TombstoneKind::Exec:
      return !isInsideExecutableSectionsAddressRange(LowPC, HighPC);
    }

    llvm_unreachable("Unknown tombstone value");
  }

  bool isDeadAddress(uint64_t LowPC, uint16_t Version, TombstoneKind Tombstone,
                     uint8_t AddressByteSize) {
    return isDeadAddressRange(LowPC, std::nullopt, Version, Tombstone,
                              AddressByteSize);
  }

private:
  RangesTy DWARFAddressRanges;
  AddressRanges TextAddressRanges;
  const Options &Opts;
  DWARFContext &Context;
};

static bool knownByDWARFUtil(StringRef SecName) {
  return llvm::StringSwitch<bool>(SecName)
      .Case(".debug_info", true)
      .Case(".debug_types", true)
      .Case(".debug_abbrev", true)
      .Case(".debug_loc", true)
      .Case(".debug_loclists", true)
      .Case(".debug_frame", true)
      .Case(".debug_aranges", true)
      .Case(".debug_ranges", true)
      .Case(".debug_rnglists", true)
      .Case(".debug_line", true)
      .Case(".debug_line_str", true)
      .Case(".debug_addr", true)
      .Case(".debug_macro", true)
      .Case(".debug_macinfo", true)
      .Case(".debug_str", true)
      .Case(".debug_str_offsets", true)
      .Default(false);
}

Error linkDebugInfo(object::ObjectFile &File, const Options &Options,
                    raw_pwrite_stream &OutStream) {

  auto ReportWarn = [&](const Twine &Message, StringRef Context,
                        const DWARFDie *Die) {
    warning(Message, Context);

    if (!Options.Verbose || !Die)
      return;

    DIDumpOptions DumpOpts;
    DumpOpts.ChildRecurseDepth = 0;
    DumpOpts.Verbose = Options.Verbose;

    WithColor::note() << "    in DIE:\n";
    Die->dump(errs(), /*Indent=*/6, DumpOpts);
  };
  auto ReportErr = [&](const Twine &Message, StringRef Context,
                       const DWARFDie *) {
    WithColor::error(errs(), Context) << Message << '\n';
  };

  // Create output streamer.
  DwarfStreamer OutStreamer(OutputFileType::Object, OutStream, nullptr,
                            ReportWarn, ReportWarn);
  Triple TargetTriple = File.makeTriple();
  if (!OutStreamer.init(TargetTriple, formatv("cannot create a stream for {0}",
                                              TargetTriple.getTriple())
                                          .str()))
    return createStringError(std::errc::invalid_argument, "");

  std::unique_ptr<DWARFContext> Context = DWARFContext::create(File);

  uint16_t MaxDWARFVersion = 0;
  std::function<void(const DWARFUnit &Unit)> OnCUDieLoaded =
      [&MaxDWARFVersion](const DWARFUnit &Unit) {
        MaxDWARFVersion = std::max(Unit.getVersion(), MaxDWARFVersion);
      };

  // Create DWARF linker.
  DWARFLinker DebugInfoLinker(&OutStreamer, DwarfLinkerClient::LLD);

  DebugInfoLinker.setEstimatedObjfilesAmount(1);
  DebugInfoLinker.setAccelTableKind(DwarfLinkerAccelTableKind::None);
  DebugInfoLinker.setErrorHandler(ReportErr);
  DebugInfoLinker.setWarningHandler(ReportWarn);
  DebugInfoLinker.setNumThreads(Options.NumThreads);
  DebugInfoLinker.setNoODR(!Options.DoODRDeduplication);
  DebugInfoLinker.setVerbosity(Options.Verbose);
  DebugInfoLinker.setUpdate(!Options.DoGarbageCollection);

  std::vector<std::unique_ptr<DWARFFile>> ObjectsForLinking(1);
  std::vector<std::unique_ptr<AddressesMap>> AddresssMapForLinking(1);
  std::vector<std::string> EmptyWarnings;

  // Unknown debug sections would be removed. Display warning
  // for such sections.
  for (SectionName Sec : Context->getDWARFObj().getSectionNames()) {
    if (isDebugSection(Sec.Name) && !knownByDWARFUtil(Sec.Name))
      warning(
          formatv("'{0}' is not currently supported: section will be skipped",
                  Sec.Name),
          Options.InputFileName);
  }

  // Add object files to the DWARFLinker.
  AddresssMapForLinking[0] =
      std::make_unique<ObjFileAddressMap>(*Context, Options, File);

  ObjectsForLinking[0] = std::make_unique<DWARFFile>(
      File.getFileName(), &*Context, AddresssMapForLinking[0].get(),
      EmptyWarnings);

  for (size_t I = 0; I < ObjectsForLinking.size(); I++)
    DebugInfoLinker.addObjectFile(*ObjectsForLinking[I], nullptr,
                                  OnCUDieLoaded);

  // If we haven't seen any CUs, pick an arbitrary valid Dwarf version anyway.
  if (MaxDWARFVersion == 0)
    MaxDWARFVersion = 3;

  if (Error Err = DebugInfoLinker.setTargetDWARFVersion(MaxDWARFVersion))
    return Err;

  // Link debug info.
  if (Error Err = DebugInfoLinker.link())
    return Err;

  OutStreamer.finish();
  return Error::success();
}

} // end of namespace dwarfutil
} // end of namespace llvm
