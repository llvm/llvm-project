//===- tools/dsymutil/DwarfLinker.h - Dwarf debug info linker ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_DWARFLINKER_H
#define LLVM_TOOLS_DSYMUTIL_DWARFLINKER_H

#include "BinaryHolder.h"
#include "CompileUnit.h"
#include "DebugMap.h"
#include "DeclContext.h"
#include "DwarfStreamer.h"
#include "LinkUtils.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"

namespace llvm {
namespace dsymutil {

/// Partial address range for debug map objects. Besides an offset, only the
/// HighPC is stored. The structure is stored in a map where the LowPC is the
/// key.
struct DebugMapObjectRange {
  /// Function HighPC.
  uint64_t HighPC;
  /// Offset to apply to the linked address.
  int64_t Offset;

  DebugMapObjectRange(uint64_t EndPC, int64_t Offset)
      : HighPC(EndPC), Offset(Offset) {}

  DebugMapObjectRange() : HighPC(0), Offset(0) {}
};

/// Map LowPC to DebugMapObjectRange.
using RangesTy = std::map<uint64_t, DebugMapObjectRange>;
using UnitListTy = std::vector<std::unique_ptr<CompileUnit>>;

/// The core of the Dwarf linking logic.
///
/// The link of the dwarf information from the object files will be
/// driven by the selection of 'root DIEs', which are DIEs that
/// describe variables or functions that are present in the linked
/// binary (and thus have entries in the debug map). All the debug
/// information that will be linked (the DIEs, but also the line
/// tables, ranges, ...) is derived from that set of root DIEs.
///
/// The root DIEs are identified because they contain relocations that
/// correspond to a debug map entry at specific places (the low_pc for
/// a function, the location for a variable). These relocations are
/// called ValidRelocs in the DwarfLinker and are gathered as a very
/// first step when we start processing a DebugMapObject.
class DwarfLinker {
public:
  DwarfLinker(raw_fd_ostream &OutFile, BinaryHolder &BinHolder,
              const LinkOptions &Options)
      : OutFile(OutFile), BinHolder(BinHolder), Options(Options) {}

  /// Link the contents of the DebugMap.
  bool link(const DebugMap &);

  void reportWarning(const Twine &Warning, const DebugMapObject &DMO,
                     const DWARFDie *DIE = nullptr) const;

private:
  /// Remembers the newest DWARF version we've seen in a unit.
  void maybeUpdateMaxDwarfVersion(unsigned Version) {
    if (MaxDwarfVersion < Version)
      MaxDwarfVersion = Version;
  }

  /// Emit warnings as Dwarf compile units to leave a trail after linking.
  bool emitPaperTrailWarnings(const DebugMapObject &DMO, const DebugMap &Map,
                              OffsetsStringPool &StringPool);

  /// Keeps track of relocations.
  class RelocationManager {
    struct ValidReloc {
      uint32_t Offset;
      uint32_t Size;
      uint64_t Addend;
      const DebugMapObject::DebugMapEntry *Mapping;

      ValidReloc(uint32_t Offset, uint32_t Size, uint64_t Addend,
                 const DebugMapObject::DebugMapEntry *Mapping)
          : Offset(Offset), Size(Size), Addend(Addend), Mapping(Mapping) {}

      bool operator<(const ValidReloc &RHS) const {
        return Offset < RHS.Offset;
      }
    };

    const DwarfLinker &Linker;

    /// The valid relocations for the current DebugMapObject.
    /// This vector is sorted by relocation offset.
    std::vector<ValidReloc> ValidRelocs;

    /// Index into ValidRelocs of the next relocation to consider. As we walk
    /// the DIEs in acsending file offset and as ValidRelocs is sorted by file
    /// offset, keeping this index up to date is all we have to do to have a
    /// cheap lookup during the root DIE selection and during DIE cloning.
    unsigned NextValidReloc = 0;

  public:
    RelocationManager(DwarfLinker &Linker) : Linker(Linker) {}

    bool hasValidRelocs() const { return !ValidRelocs.empty(); }

    /// Reset the NextValidReloc counter.
    void resetValidRelocs() { NextValidReloc = 0; }

    /// \defgroup FindValidRelocations Translate debug map into a list
    /// of relevant relocations
    ///
    /// @{
    bool findValidRelocsInDebugInfo(const object::ObjectFile &Obj,
                                    const DebugMapObject &DMO);

    bool findValidRelocs(const object::SectionRef &Section,
                         const object::ObjectFile &Obj,
                         const DebugMapObject &DMO);

    void findValidRelocsMachO(const object::SectionRef &Section,
                              const object::MachOObjectFile &Obj,
                              const DebugMapObject &DMO);
    /// @}

    bool hasValidRelocation(uint32_t StartOffset, uint32_t EndOffset,
                            CompileUnit::DIEInfo &Info);

    bool applyValidRelocs(MutableArrayRef<char> Data, uint32_t BaseOffset,
                          bool isLittleEndian);
  };

  /// Keeps track of data associated with one object during linking.
  struct LinkContext {
    DebugMapObject &DMO;
    const object::ObjectFile *ObjectFile;
    RelocationManager RelocMgr;
    std::unique_ptr<DWARFContext> DwarfContext;
    RangesTy Ranges;
    UnitListTy CompileUnits;

    LinkContext(const DebugMap &Map, DwarfLinker &Linker, DebugMapObject &DMO)
        : DMO(DMO), RelocMgr(Linker) {
      // Swift ASTs are not object files.
      if (DMO.getType() == MachO::N_AST) {
        ObjectFile = nullptr;
        return;
      }
      auto ErrOrObj = Linker.loadObject(DMO, Map);
      ObjectFile = ErrOrObj ? &*ErrOrObj : nullptr;
      DwarfContext = ObjectFile ? DWARFContext::create(*ObjectFile) : nullptr;
    }

    /// Clear compile units and ranges.
    void Clear() {
      CompileUnits.clear();
      Ranges.clear();
    }
  };

  /// Called at the start of a debug object link.
  void startDebugObject(LinkContext &Context);

  /// Called at the end of a debug object link.
  void endDebugObject(LinkContext &Context);

  /// \defgroup FindRootDIEs Find DIEs corresponding to debug map entries.
  ///
  /// @{
  /// Recursively walk the \p DIE tree and look for DIEs to
  /// keep. Store that information in \p CU's DIEInfo.
  ///
  /// The return value indicates whether the DIE is incomplete.
  bool lookForDIEsToKeep(RelocationManager &RelocMgr, RangesTy &Ranges,
                         const UnitListTy &Units, const DWARFDie &DIE,
                         const DebugMapObject &DMO, CompileUnit &CU,
                         unsigned Flags);

  /// If this compile unit is really a skeleton CU that points to a
  /// clang module, register it in ClangModules and return true.
  ///
  /// A skeleton CU is a CU without children, a DW_AT_gnu_dwo_name
  /// pointing to the module, and a DW_AT_gnu_dwo_id with the module
  /// hash.
  bool registerModuleReference(const DWARFDie &CUDie, const DWARFUnit &Unit,
                               DebugMap &ModuleMap, const DebugMapObject &DMO,
                               RangesTy &Ranges,
                               OffsetsStringPool &OffsetsStringPool,
                               UniquingStringPool &UniquingStringPoolStringPool,
                               DeclContextTree &ODRContexts, unsigned &UnitID,
                               unsigned Indent = 0);

  /// Recursively add the debug info in this clang module .pcm
  /// file (and all the modules imported by it in a bottom-up fashion)
  /// to Units.
  Error loadClangModule(StringRef Filename, StringRef ModulePath,
                        StringRef ModuleName, uint64_t DwoId,
                        DebugMap &ModuleMap, const DebugMapObject &DMO,
                        RangesTy &Ranges, OffsetsStringPool &OffsetsStringPool,
                        UniquingStringPool &UniquingStringPool,
                        DeclContextTree &ODRContexts, unsigned &UnitID,
                        unsigned Indent = 0);

  /// Flags passed to DwarfLinker::lookForDIEsToKeep
  enum TraversalFlags {
    TF_Keep = 1 << 0,            ///< Mark the traversed DIEs as kept.
    TF_InFunctionScope = 1 << 1, ///< Current scope is a function scope.
    TF_DependencyWalk = 1 << 2,  ///< Walking the dependencies of a kept DIE.
    TF_ParentWalk = 1 << 3,      ///< Walking up the parents of a kept DIE.
    TF_ODR = 1 << 4,             ///< Use the ODR while keeping dependents.
    TF_SkipPC = 1 << 5,          ///< Skip all location attributes.
  };

  /// Mark the passed DIE as well as all the ones it depends on as kept.
  void keepDIEAndDependencies(RelocationManager &RelocMgr, RangesTy &Ranges,
                              const UnitListTy &Units, const DWARFDie &DIE,
                              CompileUnit::DIEInfo &MyInfo,
                              const DebugMapObject &DMO, CompileUnit &CU,
                              bool UseODR);

  unsigned shouldKeepDIE(RelocationManager &RelocMgr, RangesTy &Ranges,
                         const DWARFDie &DIE, const DebugMapObject &DMO,
                         CompileUnit &Unit, CompileUnit::DIEInfo &MyInfo,
                         unsigned Flags);

  unsigned shouldKeepVariableDIE(RelocationManager &RelocMgr,
                                 const DWARFDie &DIE, CompileUnit &Unit,
                                 CompileUnit::DIEInfo &MyInfo, unsigned Flags);

  unsigned shouldKeepSubprogramDIE(RelocationManager &RelocMgr,
                                   RangesTy &Ranges, const DWARFDie &DIE,
                                   const DebugMapObject &DMO, CompileUnit &Unit,
                                   CompileUnit::DIEInfo &MyInfo,
                                   unsigned Flags);

  bool hasValidRelocation(uint32_t StartOffset, uint32_t EndOffset,
                          CompileUnit::DIEInfo &Info);
  /// @}

  /// \defgroup Linking Methods used to link the debug information
  ///
  /// @{

  class DIECloner {
    DwarfLinker &Linker;
    RelocationManager &RelocMgr;

    /// Allocator used for all the DIEValue objects.
    BumpPtrAllocator &DIEAlloc;

    std::vector<std::unique_ptr<CompileUnit>> &CompileUnits;
    LinkOptions Options;

  public:
    DIECloner(DwarfLinker &Linker, RelocationManager &RelocMgr,
              BumpPtrAllocator &DIEAlloc,
              std::vector<std::unique_ptr<CompileUnit>> &CompileUnits,
              LinkOptions &Options)
        : Linker(Linker), RelocMgr(RelocMgr), DIEAlloc(DIEAlloc),
          CompileUnits(CompileUnits), Options(Options) {}

    /// Recursively clone \p InputDIE into an tree of DIE objects
    /// where useless (as decided by lookForDIEsToKeep()) bits have been
    /// stripped out and addresses have been rewritten according to the
    /// debug map.
    ///
    /// \param OutOffset is the offset the cloned DIE in the output
    /// compile unit.
    /// \param PCOffset (while cloning a function scope) is the offset
    /// applied to the entry point of the function to get the linked address.
    /// \param Die the output DIE to use, pass NULL to create one.
    /// \returns the root of the cloned tree or null if nothing was selected.
    DIE *cloneDIE(const DWARFDie &InputDIE, const DebugMapObject &DMO,
                  CompileUnit &U, OffsetsStringPool &StringPool,
                  int64_t PCOffset, uint32_t OutOffset, unsigned Flags,
                  DIE *Die = nullptr);

    /// Construct the output DIE tree by cloning the DIEs we
    /// chose to keep above. If there are no valid relocs, then there's
    /// nothing to clone/emit.
    void cloneAllCompileUnits(DWARFContext &DwarfContext,
                              const DebugMapObject &DMO, RangesTy &Ranges,
                              OffsetsStringPool &StringPool);

  private:
    using AttributeSpec = DWARFAbbreviationDeclaration::AttributeSpec;

    /// Information gathered and exchanged between the various
    /// clone*Attributes helpers about the attributes of a particular DIE.
    struct AttributesInfo {
      /// Names.
      DwarfStringPoolEntryRef Name, MangledName, NameWithoutTemplate;

      /// Offsets in the string pool.
      uint32_t NameOffset = 0;
      uint32_t MangledNameOffset = 0;

      /// Value of AT_low_pc in the input DIE
      uint64_t OrigLowPc = std::numeric_limits<uint64_t>::max();

      /// Value of AT_high_pc in the input DIE
      uint64_t OrigHighPc = 0;

      /// Offset to apply to PC addresses inside a function.
      int64_t PCOffset = 0;

      /// Does the DIE have a low_pc attribute?
      bool HasLowPc = false;

      /// Does the DIE have a ranges attribute?
      bool HasRanges = false;

      /// Is this DIE only a declaration?
      bool IsDeclaration = false;

      AttributesInfo() = default;
    };

    /// Helper for cloneDIE.
    unsigned cloneAttribute(DIE &Die, const DWARFDie &InputDIE,
                            const DebugMapObject &DMO, CompileUnit &U,
                            OffsetsStringPool &StringPool,
                            const DWARFFormValue &Val,
                            const AttributeSpec AttrSpec, unsigned AttrSize,
                            AttributesInfo &AttrInfo);

    /// Clone a string attribute described by \p AttrSpec and add
    /// it to \p Die.
    /// \returns the size of the new attribute.
    unsigned cloneStringAttribute(DIE &Die, AttributeSpec AttrSpec,
                                  const DWARFFormValue &Val, const DWARFUnit &U,
                                  OffsetsStringPool &StringPool,
                                  AttributesInfo &Info);

    /// Clone an attribute referencing another DIE and add
    /// it to \p Die.
    /// \returns the size of the new attribute.
    unsigned cloneDieReferenceAttribute(DIE &Die, const DWARFDie &InputDIE,
                                        AttributeSpec AttrSpec,
                                        unsigned AttrSize,
                                        const DWARFFormValue &Val,
                                        const DebugMapObject &DMO,
                                        CompileUnit &Unit);

    /// Clone an attribute referencing another DIE and add
    /// it to \p Die.
    /// \returns the size of the new attribute.
    unsigned cloneBlockAttribute(DIE &Die, AttributeSpec AttrSpec,
                                 const DWARFFormValue &Val, unsigned AttrSize);

    /// Clone an attribute referencing another DIE and add
    /// it to \p Die.
    /// \returns the size of the new attribute.
    unsigned cloneAddressAttribute(DIE &Die, AttributeSpec AttrSpec,
                                   const DWARFFormValue &Val,
                                   const CompileUnit &Unit,
                                   AttributesInfo &Info);

    /// Clone a scalar attribute  and add it to \p Die.
    /// \returns the size of the new attribute.
    unsigned cloneScalarAttribute(DIE &Die, const DWARFDie &InputDIE,
                                  const DebugMapObject &DMO, CompileUnit &U,
                                  AttributeSpec AttrSpec,
                                  const DWARFFormValue &Val, unsigned AttrSize,
                                  AttributesInfo &Info);

    /// Get the potential name and mangled name for the entity
    /// described by \p Die and store them in \Info if they are not
    /// already there.
    /// \returns is a name was found.
    bool getDIENames(const DWARFDie &Die, AttributesInfo &Info,
                     OffsetsStringPool &StringPool, bool StripTemplate = false);

    /// Create a copy of abbreviation Abbrev.
    void copyAbbrev(const DWARFAbbreviationDeclaration &Abbrev, bool hasODR);

    uint32_t hashFullyQualifiedName(DWARFDie DIE, CompileUnit &U,
                                    const DebugMapObject &DMO,
                                    int RecurseDepth = 0);

    /// Helper for cloneDIE.
    void addObjCAccelerator(CompileUnit &Unit, const DIE *Die,
                            DwarfStringPoolEntryRef Name,
                            OffsetsStringPool &StringPool, bool SkipPubSection);
  };

  /// Assign an abbreviation number to \p Abbrev
  void AssignAbbrev(DIEAbbrev &Abbrev);

  /// Compute and emit debug_ranges section for \p Unit, and
  /// patch the attributes referencing it.
  void patchRangesForUnit(const CompileUnit &Unit, DWARFContext &Dwarf,
                          const DebugMapObject &DMO) const;

  /// Generate and emit the DW_AT_ranges attribute for a compile_unit if it had
  /// one.
  void generateUnitRanges(CompileUnit &Unit) const;

  /// Extract the line tables from the original dwarf, extract the relevant
  /// parts according to the linked function ranges and emit the result in the
  /// debug_line section.
  void patchLineTableForUnit(CompileUnit &Unit, DWARFContext &OrigDwarf,
                             RangesTy &Ranges, const DebugMapObject &DMO);

  /// Emit the accelerator entries for \p Unit.
  void emitAcceleratorEntriesForUnit(CompileUnit &Unit);

  /// Patch the frame info for an object file and emit it.
  void patchFrameInfoForObject(const DebugMapObject &, RangesTy &Ranges,
                               DWARFContext &, unsigned AddressSize);

  /// FoldingSet that uniques the abbreviations.
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  /// Storage for the unique Abbreviations.
  /// This is passed to AsmPrinter::emitDwarfAbbrevs(), thus it cannot be
  /// changed to a vector of unique_ptrs.
  std::vector<std::unique_ptr<DIEAbbrev>> Abbreviations;

  /// DIELoc objects that need to be destructed (but not freed!).
  std::vector<DIELoc *> DIELocs;

  /// DIEBlock objects that need to be destructed (but not freed!).
  std::vector<DIEBlock *> DIEBlocks;

  /// Allocator used for all the DIEValue objects.
  BumpPtrAllocator DIEAlloc;
  /// @}

  /// \defgroup Helpers Various helper methods.
  ///
  /// @{
  bool createStreamer(const Triple &TheTriple, raw_fd_ostream &OutFile);

  /// Attempt to load a debug object from disk.
  ErrorOr<const object::ObjectFile &> loadObject(const DebugMapObject &Obj,
                                                 const DebugMap &Map);
  /// @}

  raw_fd_ostream &OutFile;
  BinaryHolder &BinHolder;
  LinkOptions Options;
  std::unique_ptr<DwarfStreamer> Streamer;
  uint64_t OutputDebugInfoSize;
  unsigned MaxDwarfVersion = 0;

  /// The CIEs that have been emitted in the output section. The actual CIE
  /// data serves a the key to this StringMap, this takes care of comparing the
  /// semantics of CIEs defined in different object files.
  StringMap<uint32_t> EmittedCIEs;

  /// Offset of the last CIE that has been emitted in the output
  /// debug_frame section.
  uint32_t LastCIEOffset = 0;

  /// Apple accelerator tables.
  AccelTable<AppleAccelTableStaticOffsetData> AppleNames;
  AccelTable<AppleAccelTableStaticOffsetData> AppleNamespaces;
  AccelTable<AppleAccelTableStaticOffsetData> AppleObjc;
  AccelTable<AppleAccelTableStaticTypeData> AppleTypes;

  /// Mapping the PCM filename to the DwoId.
  StringMap<uint64_t> ClangModules;

  bool ModuleCacheHintDisplayed = false;
  bool ArchiveHintDisplayed = false;
};

} // end namespace dsymutil
} // end namespace llvm

#endif // LLVM_TOOLS_DSYMUTIL_DWARFLINKER_H
