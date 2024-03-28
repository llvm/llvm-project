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
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/Remarks/RemarkLinker.h"
#include <mutex>
#include <optional>

namespace llvm {
using namespace dwarf_linker;

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
  class AddressManager : public dwarf_linker::AddressesMap {
    struct ValidReloc {
      uint64_t Offset = 0;
      uint32_t Size = 0;
      uint64_t Addend = 0;
      const DebugMapObject::DebugMapEntry *Mapping = nullptr;

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
    /// These vectors are sorted by relocation offset.
    /// {
    std::vector<ValidReloc> ValidDebugInfoRelocs;
    std::vector<ValidReloc> ValidDebugAddrRelocs;
    /// }

    StringRef SrcFileName;

    const DebugMapObject &DMO;

    std::optional<std::string> LibInstallName;

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

    /// Link specified \p Address through debug map.
    /// \returns Address adjustment value.
    std::optional<int64_t> linkAddrValue(uint64_t Address, bool Verbose);

    /// Print content of debug map entry for the specified \p Reloc.
    void printReloc(const ValidReloc &Reloc);

    /// Print content of debug map entry for the specifed \p Mapping.
    void printMapping(StringRef SymName,
                      const DebugMapObject::SymbolMapping &SymMapping);

  public:
    AddressManager(DwarfLinkerForBinary &Linker, const object::ObjectFile &Obj,
                   const DebugMapObject &DMO)
        : Linker(Linker), SrcFileName(DMO.getObjectFilename()), DMO(DMO) {
      findValidRelocsInDebugSections(Obj, DMO);
    }
    ~AddressManager() override { clear(); }

    bool hasLiveDebugInfo() override {
      // We can detect if object file can be linked not going long way of
      // linking. If it does not have any relocation then all debug info can be
      // quickly removed instead of doing linking. The MachO::N_LIB is a special
      // case(all relocations are already resolved), but we still want to link
      // it.
      return DMO.getType() == MachO::N_LIB || !ValidDebugInfoRelocs.empty() ||
             !ValidDebugAddrRelocs.empty();
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
    /// Print debug output if \p Verbose is set.
    ///
    /// \returns relocation value if relocation exist, otherwise std::nullopt.
    std::optional<int64_t>
    hasValidRelocationAt(const std::vector<ValidReloc> &Relocs,
                         uint64_t StartOffset, uint64_t EndOffset,
                         bool Verbose);

    std::optional<int64_t> getExprOpAddressRelocAdjustment(
        DWARFUnit &U, const DWARFExpression::Operation &Op,
        uint64_t StartOffset, uint64_t EndOffset, bool Verbose) override;

    std::optional<int64_t> getSubprogramRelocAdjustment(const DWARFDie &DIE,
                                                        bool Verbose) override;

    std::optional<StringRef> getLibraryInstallName() override;

    bool applyValidRelocs(MutableArrayRef<char> Data, uint64_t BaseOffset,
                          bool IsLittleEndian) override;

    void clear() {
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
  ErrorOr<std::unique_ptr<dwarf_linker::DWARFFile>>
  loadObject(const DebugMapObject &Obj, const DebugMap &DebugMap,
             remarks::RemarkLinker &RL);

  void collectRelocationsToApplyToSwiftReflectionSections(
      const object::SectionRef &Section, StringRef &Contents,
      const llvm::object::MachOObjectFile *MO,
      const std::vector<uint64_t> &SectionToOffsetInDwarf,
      const llvm::dsymutil::DebugMapObject *Obj,
      std::vector<MachOUtils::DwarfRelocationApplicationInfo>
          &RelocationsToApply) const;

  Error copySwiftInterfaces(StringRef Architecture) const;

  void copySwiftReflectionMetadata(
      const llvm::dsymutil::DebugMapObject *Obj,
      classic::DwarfStreamer *Streamer,
      std::vector<uint64_t> &SectionToOffsetInDwarf,
      std::vector<MachOUtils::DwarfRelocationApplicationInfo>
          &RelocationsToApply);

  template <typename Linker>
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
