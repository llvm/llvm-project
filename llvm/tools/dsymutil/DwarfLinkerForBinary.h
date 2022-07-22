//===- tools/dsymutil/DwarfLinkerForBinary.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_DWARFLINKER_H
#define LLVM_TOOLS_DSYMUTIL_DWARFLINKER_H

#include "BinaryHolder.h"
#include "DebugMap.h"
#include "LinkUtils.h"
#include "MachOUtils.h"
#include "llvm/DWARFLinker/DWARFLinker.h"
#include "llvm/DWARFLinker/DWARFLinkerCompileUnit.h"
#include "llvm/DWARFLinker/DWARFLinkerDeclContext.h"
#include "llvm/DWARFLinker/DWARFStreamer.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Remarks/RemarkLinker.h"

namespace llvm {
namespace dsymutil {

/// The core of the Dsymutil Dwarf linking logic.
///
/// The link of the dwarf information from the object files will be
/// driven by DWARFLinker. DwarfLinkerForBinary reads DebugMap objects
/// and pass information to the DWARFLinker. DWARFLinker
/// optimizes DWARF taking into account valid relocations.
/// Finally, optimized DWARF is passed to DwarfLinkerForBinary through
/// DWARFEmitter interface.
class DwarfLinkerForBinary {
public:
  DwarfLinkerForBinary(raw_fd_ostream &OutFile, BinaryHolder &BinHolder,
                       LinkOptions Options)
      : OutFile(OutFile), BinHolder(BinHolder), Options(std::move(Options)) {}

  /// Link the contents of the DebugMap.
  bool link(const DebugMap &);

  void reportWarning(const Twine &Warning, StringRef Context,
                     const DWARFDie *DIE = nullptr) const;

  /// Flags passed to DwarfLinker::lookForDIEsToKeep
  enum TraversalFlags {
    TF_Keep = 1 << 0,            ///< Mark the traversed DIEs as kept.
    TF_InFunctionScope = 1 << 1, ///< Current scope is a function scope.
    TF_DependencyWalk = 1 << 2,  ///< Walking the dependencies of a kept DIE.
    TF_ParentWalk = 1 << 3,      ///< Walking up the parents of a kept DIE.
    TF_ODR = 1 << 4,             ///< Use the ODR while keeping dependents.
    TF_SkipPC = 1 << 5,          ///< Skip all location attributes.
  };

private:

  /// Keeps track of relocations.
  class AddressManager : public AddressesMap {
    struct ValidReloc {
      uint64_t Offset;
      uint32_t Size;
      uint64_t Addend;
      const DebugMapObject::DebugMapEntry *Mapping;

      ValidReloc(uint64_t Offset, uint32_t Size, uint64_t Addend,
                 const DebugMapObject::DebugMapEntry *Mapping)
          : Offset(Offset), Size(Size), Addend(Addend), Mapping(Mapping) {}

      bool operator<(const ValidReloc &RHS) const {
        return Offset < RHS.Offset;
      }
      bool operator<(uint64_t RHS) const { return Offset < RHS; }
    };

    const DwarfLinkerForBinary &Linker;

    /// The valid relocations for the current DebugMapObject.
    /// This vector is sorted by relocation offset.
    /// {
    std::vector<ValidReloc> ValidDebugInfoRelocs;
    std::vector<ValidReloc> ValidDebugAddrRelocs;
    /// }

    RangesTy AddressRanges;

    StringRef SrcFileName;

    /// Returns list of valid relocations from \p Relocs,
    /// between \p StartOffset and \p NextOffset.
    ///
    /// \returns true if any relocation is found.
    std::vector<ValidReloc>
    getRelocations(const std::vector<ValidReloc> &Relocs, uint64_t StartPos,
                   uint64_t EndPos);

    /// Resolve specified relocation \p Reloc.
    ///
    /// \returns resolved value.
    uint64_t relocate(const ValidReloc &Reloc) const;

    /// Fill \p Info with address information for the specified \p Reloc.
    void fillDieInfo(const ValidReloc &Reloc, CompileUnit::DIEInfo &Info);

    /// Print contents of debug map entry for the specified \p Reloc.
    void printReloc(const ValidReloc &Reloc);

  public:
    AddressManager(DwarfLinkerForBinary &Linker, const object::ObjectFile &Obj,
                   const DebugMapObject &DMO)
        : Linker(Linker), SrcFileName(DMO.getObjectFilename()) {
      findValidRelocsInDebugSections(Obj, DMO);

      // Iterate over the debug map entries and put all the ones that are
      // functions (because they have a size) into the Ranges map. This map is
      // very similar to the FunctionRanges that are stored in each unit, with 2
      // notable differences:
      //
      //  1. Obviously this one is global, while the other ones are per-unit.
      //
      //  2. This one contains not only the functions described in the DIE
      //     tree, but also the ones that are only in the debug map.
      //
      // The latter information is required to reproduce dsymutil's logic while
      // linking line tables. The cases where this information matters look like
      // bugs that need to be investigated, but for now we need to reproduce
      // dsymutil's behavior.
      // FIXME: Once we understood exactly if that information is needed,
      // maybe totally remove this (or try to use it to do a real
      // -gline-tables-only on Darwin.
      for (const auto &Entry : DMO.symbols()) {
        const auto &Mapping = Entry.getValue();
        if (Mapping.Size && Mapping.ObjectAddress)
          AddressRanges.insert(
              {*Mapping.ObjectAddress, *Mapping.ObjectAddress + Mapping.Size},
              int64_t(Mapping.BinaryAddress) - *Mapping.ObjectAddress);
      }
    }
    virtual ~AddressManager() override { clear(); }

    bool hasValidRelocs() override {
      return !ValidDebugInfoRelocs.empty() || !ValidDebugAddrRelocs.empty();
    }

    /// \defgroup FindValidRelocations Translate debug map into a list
    /// of relevant relocations
    ///
    /// @{
    bool findValidRelocsInDebugSections(const object::ObjectFile &Obj,
                                        const DebugMapObject &DMO);

    bool findValidRelocs(const object::SectionRef &Section,
                         const object::ObjectFile &Obj,
                         const DebugMapObject &DMO,
                         std::vector<ValidReloc> &ValidRelocs);

    void findValidRelocsMachO(const object::SectionRef &Section,
                              const object::MachOObjectFile &Obj,
                              const DebugMapObject &DMO,
                              std::vector<ValidReloc> &ValidRelocs);
    /// @}

    /// Checks that there is a relocation in the \p Relocs array against a
    /// debug map entry between \p StartOffset and \p NextOffset.
    ///
    /// \returns true and sets Info.InDebugMap if it is the case.
    bool hasValidRelocationAt(const std::vector<ValidReloc> &Relocs,
                              uint64_t StartOffset, uint64_t EndOffset,
                              CompileUnit::DIEInfo &Info);

    bool isLiveVariable(const DWARFDie &DIE,
                        CompileUnit::DIEInfo &Info) override;
    bool isLiveSubprogram(const DWARFDie &DIE,
                          CompileUnit::DIEInfo &Info) override;

    bool applyValidRelocs(MutableArrayRef<char> Data, uint64_t BaseOffset,
                          bool IsLittleEndian) override;

    llvm::Expected<uint64_t> relocateIndexedAddr(uint64_t StartOffset,
                                                 uint64_t EndOffset) override;

    RangesTy &getValidAddressRanges() override { return AddressRanges; }

    void clear() override {
      AddressRanges.clear();
      ValidDebugInfoRelocs.clear();
      ValidDebugAddrRelocs.clear();
    }
  };

private:
  /// \defgroup Helpers Various helper methods.
  ///
  /// @{
  bool createStreamer(const Triple &TheTriple, raw_fd_ostream &OutFile);

  /// Attempt to load a debug object from disk.
  ErrorOr<const object::ObjectFile &> loadObject(const DebugMapObject &Obj,
                                                 const Triple &triple);
  ErrorOr<DWARFFile &> loadObject(const DebugMapObject &Obj,
                                  const DebugMap &DebugMap,
                                  remarks::RemarkLinker &RL);

  void collectRelocationsToApplyToSwiftReflectionSections(
      const object::SectionRef &Section, StringRef &Contents,
      const llvm::object::MachOObjectFile *MO,
      const std::vector<uint64_t> &SectionToOffsetInDwarf,
      const llvm::dsymutil::DebugMapObject *Obj,
      std::vector<MachOUtils::DwarfRelocationApplicationInfo>
          &RelocationsToApply) const;

  void copySwiftReflectionMetadata(
      const llvm::dsymutil::DebugMapObject *Obj, DwarfStreamer *Streamer,
      std::vector<uint64_t> &SectionToOffsetInDwarf,
      std::vector<MachOUtils::DwarfRelocationApplicationInfo>
          &RelocationsToApply);

  raw_fd_ostream &OutFile;
  BinaryHolder &BinHolder;
  LinkOptions Options;
  std::unique_ptr<DwarfStreamer> Streamer;
  std::vector<std::unique_ptr<DWARFFile>> ObjectsForLinking;
  std::vector<std::unique_ptr<DWARFContext>> ContextForLinking;
  std::vector<std::unique_ptr<AddressManager>> AddressMapForLinking;
  std::vector<std::string> EmptyWarnings;

  /// A list of all .swiftinterface files referenced by the debug
  /// info, mapping Module name to path on disk. The entries need to
  /// be uniqued and sorted and there are only few entries expected
  /// per compile unit, which is why this is a std::map.
  std::map<std::string, std::string> ParseableSwiftInterfaces;

  bool ModuleCacheHintDisplayed = false;
  bool ArchiveHintDisplayed = false;
};

} // end namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_DWARFLINKER_H
