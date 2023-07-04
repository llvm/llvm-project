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
#include "llvm/DWARFLinkerParallel/DWARFLinker.h"
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
template <typename AddressMapBase>
class ObjFileAddressMap : public AddressMapBase {
public:
  ObjFileAddressMap(DWARFContext &Context, const Options &Options,
                    object::ObjectFile &ObjFile)
      : Opts(Options) {
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
      if (!ARanges) {
        llvm::consumeError(ARanges.takeError());
        continue;
      }

      for (auto &Range : *ARanges) {
        if (!isDeadAddressRange(Range.LowPC, Range.HighPC, CU->getVersion(),
                                Options.Tombstone, CU->getAddressByteSize())) {
          HasValidAddressRanges = true;
          break;
        }
      }

      if (HasValidAddressRanges)
        break;
    }
  }

  // should be renamed into has valid address ranges
  bool hasValidRelocs() override { return HasValidAddressRanges; }

  std::optional<int64_t>
  getSubprogramRelocAdjustment(const DWARFDie &DIE) override {
    assert((DIE.getTag() == dwarf::DW_TAG_subprogram ||
            DIE.getTag() == dwarf::DW_TAG_label) &&
           "Wrong type of input die");

    if (std::optional<uint64_t> LowPC =
            dwarf::toAddress(DIE.find(dwarf::DW_AT_low_pc))) {
      if (!isDeadAddress(*LowPC, DIE.getDwarfUnit()->getVersion(),
                         Opts.Tombstone,
                         DIE.getDwarfUnit()->getAddressByteSize()))
        // Relocation value for the linked binary is 0.
        return 0;
    }

    return std::nullopt;
  }

  std::optional<int64_t> getExprOpAddressRelocAdjustment(
      DWARFUnit &U, const DWARFExpression::Operation &Op, uint64_t StartOffset,
      uint64_t EndOffset) override {
    switch (Op.getCode()) {
    default: {
      assert(false && "Specified operation does not have address operand");
    } break;
    case dwarf::DW_OP_const4u:
    case dwarf::DW_OP_const8u:
    case dwarf::DW_OP_const4s:
    case dwarf::DW_OP_const8s:
    case dwarf::DW_OP_addr: {
      if (!isDeadAddress(Op.getRawOperand(0), U.getVersion(), Opts.Tombstone,
                         U.getAddressByteSize()))
        // Relocation value for the linked binary is 0.
        return 0;
    } break;
    case dwarf::DW_OP_constx:
    case dwarf::DW_OP_addrx: {
      if (std::optional<object::SectionedAddress> Address =
              U.getAddrOffsetSectionItem(Op.getRawOperand(0))) {
        if (!isDeadAddress(Address->Address, U.getVersion(), Opts.Tombstone,
                           U.getAddressByteSize()))
          // Relocation value for the linked binary is 0.
          return 0;
      }
    } break;
    }

    return std::nullopt;
  }

  bool applyValidRelocs(MutableArrayRef<char>, uint64_t, bool) override {
    // no need to apply relocations to the linked binary.
    return false;
  }

  void clear() override {}

protected:
  // returns true if specified address range is inside address ranges
  // of executable sections.
  bool isInsideExecutableSectionsAddressRange(uint64_t LowPC,
                                              std::optional<uint64_t> HighPC) {
    std::optional<AddressRange> Range =
        TextAddressRanges.getRangeThatContains(LowPC);

    if (HighPC)
      return Range.has_value() && Range->end() >= *HighPC;

    return Range.has_value();
  }

  uint64_t isBFDDeadAddressRange(uint64_t LowPC, std::optional<uint64_t> HighPC,
                                 uint16_t Version) {
    if (LowPC == 0)
      return true;

    if ((Version <= 4) && HighPC && (LowPC == 1 && *HighPC == 1))
      return true;

    return !isInsideExecutableSectionsAddressRange(LowPC, HighPC);
  }

  uint64_t isMAXPCDeadAddressRange(uint64_t LowPC,
                                   std::optional<uint64_t> HighPC,
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

  bool isDeadAddressRange(uint64_t LowPC, std::optional<uint64_t> HighPC,
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
  AddressRanges TextAddressRanges;
  const Options &Opts;
  bool HasValidAddressRanges = false;
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
      .Case(".debug_pubnames", true)
      .Case(".debug_pubtypes", true)
      .Case(".debug_names", true)
      .Default(false);
}

template <typename AccelTableKind>
static std::optional<AccelTableKind>
getAcceleratorTableKind(StringRef SecName) {
  return llvm::StringSwitch<std::optional<AccelTableKind>>(SecName)
      .Case(".debug_pubnames", AccelTableKind::Pub)
      .Case(".debug_pubtypes", AccelTableKind::Pub)
      .Case(".debug_names", AccelTableKind::DebugNames)
      .Default(std::nullopt);
}

static std::string getMessageForReplacedAcceleratorTables(
    SmallVector<StringRef> &AccelTableNamesToReplace,
    DwarfUtilAccelKind TargetTable) {
  std::string Message;

  Message += "'";
  for (StringRef Name : AccelTableNamesToReplace) {
    if (Message.size() > 1)
      Message += ", ";
    Message += Name;
  }

  Message += "' will be replaced with requested ";

  switch (TargetTable) {
  case DwarfUtilAccelKind::DWARF:
    Message += ".debug_names table";
    break;

  default:
    assert(false);
  }

  return Message;
}

static std::string getMessageForDeletedAcceleratorTables(
    SmallVector<StringRef> &AccelTableNamesToReplace) {
  std::string Message;

  Message += "'";
  for (StringRef Name : AccelTableNamesToReplace) {
    if (Message.size() > 1)
      Message += ", ";
    Message += Name;
  }

  Message += "' will be deleted as no accelerator tables are requested";

  return Message;
}

template <typename Linker, typename OutDwarfFile, typename AddressMapBase>
Error linkDebugInfoImpl(object::ObjectFile &File, const Options &Options,
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

  // Create DWARF linker.
  std::unique_ptr<Linker> DebugInfoLinker =
      Linker::createLinker(ReportErr, ReportWarn);

  Triple TargetTriple = File.makeTriple();
  if (Error Err = DebugInfoLinker->createEmitter(
          TargetTriple, Linker::OutputFileType::Object, OutStream))
    return Err;

  DebugInfoLinker->setEstimatedObjfilesAmount(1);
  DebugInfoLinker->setNumThreads(Options.NumThreads);
  DebugInfoLinker->setNoODR(!Options.DoODRDeduplication);
  DebugInfoLinker->setVerbosity(Options.Verbose);
  DebugInfoLinker->setUpdateIndexTablesOnly(!Options.DoGarbageCollection);

  std::vector<std::unique_ptr<OutDwarfFile>> ObjectsForLinking(1);
  std::vector<std::string> EmptyWarnings;

  // Add object files to the DWARFLinker.
  std::unique_ptr<DWARFContext> Context = DWARFContext::create(File);
  std::unique_ptr<ObjFileAddressMap<AddressMapBase>> AddressesMap(
      std::make_unique<ObjFileAddressMap<AddressMapBase>>(*Context, Options,
                                                          File));

  ObjectsForLinking[0] =
      std::make_unique<OutDwarfFile>(File.getFileName(), std::move(Context),
                                     std::move(AddressesMap), EmptyWarnings);

  uint16_t MaxDWARFVersion = 0;
  std::function<void(const DWARFUnit &Unit)> OnCUDieLoaded =
      [&MaxDWARFVersion](const DWARFUnit &Unit) {
        MaxDWARFVersion = std::max(Unit.getVersion(), MaxDWARFVersion);
      };

  for (size_t I = 0; I < ObjectsForLinking.size(); I++)
    DebugInfoLinker->addObjectFile(*ObjectsForLinking[I], nullptr,
                                   OnCUDieLoaded);

  // If we haven't seen any CUs, pick an arbitrary valid Dwarf version anyway.
  if (MaxDWARFVersion == 0)
    MaxDWARFVersion = 3;

  if (Error Err = DebugInfoLinker->setTargetDWARFVersion(MaxDWARFVersion))
    return Err;

  SmallVector<typename Linker::AccelTableKind> AccelTables;

  switch (Options.AccelTableKind) {
  case DwarfUtilAccelKind::None:
    // Nothing to do.
    break;
  case DwarfUtilAccelKind::DWARF:
    // use .debug_names for all DWARF versions.
    AccelTables.push_back(Linker::AccelTableKind::DebugNames);
    break;
  }

  // Add accelerator tables to DWARFLinker.
  for (typename Linker::AccelTableKind Table : AccelTables)
    DebugInfoLinker->addAccelTableKind(Table);

  for (std::unique_ptr<OutDwarfFile> &CurFile : ObjectsForLinking) {
    SmallVector<StringRef> AccelTableNamesToReplace;
    SmallVector<StringRef> AccelTableNamesToDelete;

    // Unknown debug sections or non-requested accelerator sections would be
    // removed. Display warning for such sections.
    for (SectionName Sec : CurFile->Dwarf->getDWARFObj().getSectionNames()) {
      if (isDebugSection(Sec.Name)) {
        std::optional<typename Linker::AccelTableKind> SrcAccelTableKind =
            getAcceleratorTableKind<typename Linker::AccelTableKind>(Sec.Name);

        if (SrcAccelTableKind) {
          assert(knownByDWARFUtil(Sec.Name));

          if (Options.AccelTableKind == DwarfUtilAccelKind::None)
            AccelTableNamesToDelete.push_back(Sec.Name);
          else if (!llvm::is_contained(AccelTables, *SrcAccelTableKind))
            AccelTableNamesToReplace.push_back(Sec.Name);
        } else if (!knownByDWARFUtil(Sec.Name)) {
          assert(!SrcAccelTableKind);
          warning(
              formatv(
                  "'{0}' is not currently supported: section will be skipped",
                  Sec.Name),
              Options.InputFileName);
        }
      }
    }

    // Display message for the replaced accelerator tables.
    if (!AccelTableNamesToReplace.empty())
      warning(getMessageForReplacedAcceleratorTables(AccelTableNamesToReplace,
                                                     Options.AccelTableKind),
              Options.InputFileName);

    // Display message for the removed accelerator tables.
    if (!AccelTableNamesToDelete.empty())
      warning(getMessageForDeletedAcceleratorTables(AccelTableNamesToDelete),
              Options.InputFileName);
  }

  // Link debug info.
  if (Error Err = DebugInfoLinker->link())
    return Err;

  DebugInfoLinker->getEmitter()->finish();
  return Error::success();
}

Error linkDebugInfo(object::ObjectFile &File, const Options &Options,
                    raw_pwrite_stream &OutStream) {
  if (Options.UseLLVMDWARFLinker)
    return linkDebugInfoImpl<dwarflinker_parallel::DWARFLinker,
                             dwarflinker_parallel::DWARFFile,
                             dwarflinker_parallel::AddressesMap>(File, Options,
                                                                 OutStream);
  else
    return linkDebugInfoImpl<DWARFLinker, DWARFFile, AddressesMap>(
        File, Options, OutStream);
}

} // end of namespace dwarfutil
} // end of namespace llvm
