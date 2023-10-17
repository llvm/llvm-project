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
#include <mutex>

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
                       LinkOptions Options, std::mutex &ErrorHandlerMutex)
      : OutFile(OutFile), BinHolder(BinHolder), Options(std::move(Options)),
        ErrorHandlerMutex(ErrorHandlerMutex) {}

  /// Link the contents of the DebugMap.
  bool link(const DebugMap &);

  void reportWarning(Twine Warning, Twine Context = {},
                     const DWARFDie *DIE = nullptr) const;
  void reportError(Twine Error, Twine Context = {},
                   const DWARFDie *DIE = nullptr) const;

  /// Returns true if input verification is enabled and verification errors were
  /// found.
  bool InputVerificationFailed() const { return HasVerificationErrors; }

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
  template <typename AddressesMapBase>
  class AddressManager : public AddressesMapBase {
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

    /// \returns value for the specified \p Reloc.
    int64_t getRelocValue(const ValidReloc &Reloc);

    /// Print contents of debug map entry for the specified \p Reloc.
    void printReloc(const ValidReloc &Reloc);

  public:
    AddressManager(DwarfLinkerForBinary &Linker, const object::ObjectFile &Obj,
                   const DebugMapObject &DMO)
        : Linker(Linker), SrcFileName(DMO.getObjectFilename()) {
      findValidRelocsInDebugSections(Obj, DMO);
    }
    ~AddressManager() override { clear(); }

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
    /// \returns relocation value if relocation exist, otherwise std::nullopt.
    std::optional<int64_t>
    hasValidRelocationAt(const std::vector<ValidReloc> &Relocs,
                         uint64_t StartOffset, uint64_t EndOffset);

    std::optional<int64_t> getExprOpAddressRelocAdjustment(
        DWARFUnit &U, const DWARFExpression::Operation &Op,
        uint64_t StartOffset, uint64_t EndOffset) override;

    std::optional<int64_t>
    getSubprogramRelocAdjustment(const DWARFDie &DIE) override;

    bool applyValidRelocs(MutableArrayRef<char> Data, uint64_t BaseOffset,
                          bool IsLittleEndian) override;

    void clear() override {
      ValidDebugInfoRelocs.clear();
      ValidDebugAddrRelocs.clear();
    }
  };

private:
  /// \defgroup Helpers Various helper methods.
  ///
  /// @{
  template <typename OutStreamer>
  bool createStreamer(const Triple &TheTriple,
                      typename OutStreamer::OutputFileType FileType,
                      std::unique_ptr<OutStreamer> &Streamer,
                      raw_fd_ostream &OutFile);

  /// Attempt to load a debug object from disk.
  ErrorOr<const object::ObjectFile &> loadObject(const DebugMapObject &Obj,
                                                 const Triple &triple);

  template <typename OutDWARFFile, typename AddressesMap>
  ErrorOr<std::unique_ptr<OutDWARFFile>> loadObject(const DebugMapObject &Obj,
                                                    const DebugMap &DebugMap,
                                                    remarks::RemarkLinker &RL);

  void collectRelocationsToApplyToSwiftReflectionSections(
      const object::SectionRef &Section, StringRef &Contents,
      const llvm::object::MachOObjectFile *MO,
      const std::vector<uint64_t> &SectionToOffsetInDwarf,
      const llvm::dsymutil::DebugMapObject *Obj,
      std::vector<MachOUtils::DwarfRelocationApplicationInfo>
          &RelocationsToApply) const;

  Error copySwiftInterfaces(StringRef Architecture) const;

  template <typename OutStreamer>
  void copySwiftReflectionMetadata(
      const llvm::dsymutil::DebugMapObject *Obj, OutStreamer *Streamer,
      std::vector<uint64_t> &SectionToOffsetInDwarf,
      std::vector<MachOUtils::DwarfRelocationApplicationInfo>
          &RelocationsToApply);

  template <typename Linker, typename OutDwarfFile, typename AddressMapBase>
  bool linkImpl(const DebugMap &Map,
                typename Linker::OutputFileType ObjectType);

  raw_fd_ostream &OutFile;
  BinaryHolder &BinHolder;
  LinkOptions Options;
  std::mutex &ErrorHandlerMutex;

  std::vector<std::string> EmptyWarnings;

  /// A list of all .swiftinterface files referenced by the debug
  /// info, mapping Module name to path on disk. The entries need to
  /// be uniqued and sorted and there are only few entries expected
  /// per compile unit, which is why this is a std::map.
  std::map<std::string, std::string> ParseableSwiftInterfaces;

  bool ModuleCacheHintDisplayed = false;
  bool ArchiveHintDisplayed = false;
  bool HasVerificationErrors = false;
};

} // end namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_DWARFLINKER_H
