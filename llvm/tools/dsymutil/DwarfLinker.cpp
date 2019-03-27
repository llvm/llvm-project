//===- tools/dsymutil/DwarfLinker.cpp - Dwarf debug info linker -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DwarfLinker.h"
#include "BinaryHolder.h"
#include "DebugMap.h"
#include "DeclContext.h"
#include "DwarfStreamer.h"
#include "MachOUtils.h"
#include "NonRelocatableStringpool.h"
#include "dsymutil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/CodeGen/AccelTable.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DJB.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace llvm {
namespace dsymutil {

/// Similar to DWARFUnitSection::getUnitForOffset(), but returning our
/// CompileUnit object instead.
static CompileUnit *getUnitForOffset(const UnitListTy &Units, unsigned Offset) {
  auto CU = std::upper_bound(
      Units.begin(), Units.end(), Offset,
      [](uint32_t LHS, const std::unique_ptr<CompileUnit> &RHS) {
        return LHS < RHS->getOrigUnit().getNextUnitOffset();
      });
  return CU != Units.end() ? CU->get() : nullptr;
}

/// Resolve the DIE attribute reference that has been extracted in \p RefValue.
/// The resulting DIE might be in another CompileUnit which is stored into \p
/// ReferencedCU. \returns null if resolving fails for any reason.
static DWARFDie resolveDIEReference(const DwarfLinker &Linker,
                                    const DebugMapObject &DMO,
                                    const UnitListTy &Units,
                                    const DWARFFormValue &RefValue,
                                    const DWARFUnit &Unit, const DWARFDie &DIE,
                                    CompileUnit *&RefCU) {
  assert(RefValue.isFormClass(DWARFFormValue::FC_Reference));
  uint64_t RefOffset = *RefValue.getAsReference();
  if ((RefCU = getUnitForOffset(Units, RefOffset)))
    if (const auto RefDie = RefCU->getOrigUnit().getDIEForOffset(RefOffset)) {
      // In a file with broken references, an attribute might point to a NULL
      // DIE.
      if (!RefDie.isNULL())
        return RefDie;
    }

  Linker.reportWarning("could not find referenced DIE", DMO, &DIE);
  return DWARFDie();
}

/// \returns whether the passed \a Attr type might contain a DIE reference
/// suitable for ODR uniquing.
static bool isODRAttribute(uint16_t Attr) {
  switch (Attr) {
  default:
    return false;
  case dwarf::DW_AT_type:
  case dwarf::DW_AT_containing_type:
  case dwarf::DW_AT_specification:
  case dwarf::DW_AT_abstract_origin:
  case dwarf::DW_AT_import:
    return true;
  }
  llvm_unreachable("Improper attribute.");
}

static bool isTypeTag(uint16_t Tag) {
  switch (Tag) {
  case dwarf::DW_TAG_array_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_pointer_type:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_string_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_ptr_to_member_type:
  case dwarf::DW_TAG_set_type:
  case dwarf::DW_TAG_subrange_type:
  case dwarf::DW_TAG_base_type:
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_file_type:
  case dwarf::DW_TAG_namelist:
  case dwarf::DW_TAG_packed_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_atomic_type:
  case dwarf::DW_TAG_interface_type:
  case dwarf::DW_TAG_unspecified_type:
  case dwarf::DW_TAG_shared_type:
    return true;
  default:
    break;
  }
  return false;
}

bool DwarfLinker::DIECloner::getDIENames(const DWARFDie &Die,
                                         AttributesInfo &Info,
                                         OffsetsStringPool &StringPool,
                                         bool StripTemplate) {
  // This function will be called on DIEs having low_pcs and
  // ranges. As getting the name might be more expansive, filter out
  // blocks directly.
  if (Die.getTag() == dwarf::DW_TAG_lexical_block)
    return false;

  // FIXME: a bit wasteful as the first getName might return the
  // short name.
  if (!Info.MangledName)
    if (const char *MangledName = Die.getName(DINameKind::LinkageName))
      Info.MangledName = StringPool.getEntry(MangledName);

  if (!Info.Name)
    if (const char *Name = Die.getName(DINameKind::ShortName))
      Info.Name = StringPool.getEntry(Name);

  if (StripTemplate && Info.Name && Info.MangledName != Info.Name) {
    // FIXME: dsymutil compatibility. This is wrong for operator<
    auto Split = Info.Name.getString().split('<');
    if (!Split.second.empty())
      Info.NameWithoutTemplate = StringPool.getEntry(Split.first);
  }

  return Info.Name || Info.MangledName;
}

/// Report a warning to the user, optionally including information about a
/// specific \p DIE related to the warning.
void DwarfLinker::reportWarning(const Twine &Warning, const DebugMapObject &DMO,
                                const DWARFDie *DIE) const {
  StringRef Context = DMO.getObjectFilename();
  warn(Warning, Context);

  if (!Options.Verbose || !DIE)
    return;

  DIDumpOptions DumpOpts;
  DumpOpts.RecurseDepth = 0;
  DumpOpts.Verbose = Options.Verbose;

  WithColor::note() << "    in DIE:\n";
  DIE->dump(errs(), 6 /* Indent */, DumpOpts);
}

bool DwarfLinker::createStreamer(const Triple &TheTriple,
                                 raw_fd_ostream &OutFile) {
  if (Options.NoOutput)
    return true;

  Streamer = llvm::make_unique<DwarfStreamer>(OutFile, Options);
  return Streamer->init(TheTriple);
}

/// Recursive helper to build the global DeclContext information and
/// gather the child->parent relationships in the original compile unit.
///
/// \return true when this DIE and all of its children are only
/// forward declarations to types defined in external clang modules
/// (i.e., forward declarations that are children of a DW_TAG_module).
static bool analyzeContextInfo(const DWARFDie &DIE, unsigned ParentIdx,
                               CompileUnit &CU, DeclContext *CurrentDeclContext,
                               UniquingStringPool &StringPool,
                               DeclContextTree &Contexts,
                               uint64_t ModulesEndOffset,
                               bool InImportedModule = false) {
  unsigned MyIdx = CU.getOrigUnit().getDIEIndex(DIE);
  CompileUnit::DIEInfo &Info = CU.getInfo(MyIdx);

  // Clang imposes an ODR on modules(!) regardless of the language:
  //  "The module-id should consist of only a single identifier,
  //   which provides the name of the module being defined. Each
  //   module shall have a single definition."
  //
  // This does not extend to the types inside the modules:
  //  "[I]n C, this implies that if two structs are defined in
  //   different submodules with the same name, those two types are
  //   distinct types (but may be compatible types if their
  //   definitions match)."
  //
  // We treat non-C++ modules like namespaces for this reason.
  if (DIE.getTag() == dwarf::DW_TAG_module && ParentIdx == 0 &&
      dwarf::toString(DIE.find(dwarf::DW_AT_name), "") !=
          CU.getClangModuleName()) {
    InImportedModule = true;
  }

  Info.ParentIdx = ParentIdx;
  bool InClangModule = CU.isClangModule() || InImportedModule;
  if (CU.hasODR() || InClangModule) {
    if (CurrentDeclContext) {
      auto PtrInvalidPair = Contexts.getChildDeclContext(
          *CurrentDeclContext, DIE, CU, StringPool, InClangModule);
      CurrentDeclContext = PtrInvalidPair.getPointer();
      Info.Ctxt =
          PtrInvalidPair.getInt() ? nullptr : PtrInvalidPair.getPointer();
      if (Info.Ctxt)
        Info.Ctxt->setDefinedInClangModule(InClangModule);
    } else
      Info.Ctxt = CurrentDeclContext = nullptr;
  }

  Info.Prune = InImportedModule;
  if (DIE.hasChildren())
    for (auto Child : DIE.children())
      Info.Prune &=
          analyzeContextInfo(Child, MyIdx, CU, CurrentDeclContext, StringPool,
                             Contexts, ModulesEndOffset, InImportedModule);

  // Prune this DIE if it is either a forward declaration inside a
  // DW_TAG_module or a DW_TAG_module that contains nothing but
  // forward declarations.
  Info.Prune &= (DIE.getTag() == dwarf::DW_TAG_module) ||
                (isTypeTag(DIE.getTag()) &&
                 dwarf::toUnsigned(DIE.find(dwarf::DW_AT_declaration), 0));

  // Only prune forward declarations inside a DW_TAG_module for which a
  // definition exists elsewhere.
  if (ModulesEndOffset == 0)
    Info.Prune &= Info.Ctxt && Info.Ctxt->getCanonicalDIEOffset();
  else
    Info.Prune &= Info.Ctxt && Info.Ctxt->getCanonicalDIEOffset() > 0 &&
                  Info.Ctxt->getCanonicalDIEOffset() <= ModulesEndOffset;

  return Info.Prune;
} // namespace dsymutil

static bool dieNeedsChildrenToBeMeaningful(uint32_t Tag) {
  switch (Tag) {
  default:
    return false;
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_lexical_block:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_union_type:
    return true;
  }
  llvm_unreachable("Invalid Tag");
}

void DwarfLinker::startDebugObject(LinkContext &Context) {
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
  for (const auto &Entry : Context.DMO.symbols()) {
    const auto &Mapping = Entry.getValue();
    if (Mapping.Size && Mapping.ObjectAddress)
      Context.Ranges[*Mapping.ObjectAddress] = DebugMapObjectRange(
          *Mapping.ObjectAddress + Mapping.Size,
          int64_t(Mapping.BinaryAddress) - *Mapping.ObjectAddress);
  }
}

void DwarfLinker::endDebugObject(LinkContext &Context) {
  Context.Clear();

  for (auto I = DIEBlocks.begin(), E = DIEBlocks.end(); I != E; ++I)
    (*I)->~DIEBlock();
  for (auto I = DIELocs.begin(), E = DIELocs.end(); I != E; ++I)
    (*I)->~DIELoc();

  DIEBlocks.clear();
  DIELocs.clear();
  DIEAlloc.Reset();
}

static bool isMachOPairedReloc(uint64_t RelocType, uint64_t Arch) {
  switch (Arch) {
  case Triple::x86:
    return RelocType == MachO::GENERIC_RELOC_SECTDIFF ||
           RelocType == MachO::GENERIC_RELOC_LOCAL_SECTDIFF;
  case Triple::x86_64:
    return RelocType == MachO::X86_64_RELOC_SUBTRACTOR;
  case Triple::arm:
  case Triple::thumb:
    return RelocType == MachO::ARM_RELOC_SECTDIFF ||
           RelocType == MachO::ARM_RELOC_LOCAL_SECTDIFF ||
           RelocType == MachO::ARM_RELOC_HALF ||
           RelocType == MachO::ARM_RELOC_HALF_SECTDIFF;
  case Triple::aarch64:
    return RelocType == MachO::ARM64_RELOC_SUBTRACTOR;
  default:
    return false;
  }
}

/// Iterate over the relocations of the given \p Section and
/// store the ones that correspond to debug map entries into the
/// ValidRelocs array.
void DwarfLinker::RelocationManager::findValidRelocsMachO(
    const object::SectionRef &Section, const object::MachOObjectFile &Obj,
    const DebugMapObject &DMO) {
  StringRef Contents;
  Section.getContents(Contents);
  DataExtractor Data(Contents, Obj.isLittleEndian(), 0);
  bool SkipNext = false;

  for (const object::RelocationRef &Reloc : Section.relocations()) {
    if (SkipNext) {
      SkipNext = false;
      continue;
    }

    object::DataRefImpl RelocDataRef = Reloc.getRawDataRefImpl();
    MachO::any_relocation_info MachOReloc = Obj.getRelocation(RelocDataRef);

    if (isMachOPairedReloc(Obj.getAnyRelocationType(MachOReloc),
                           Obj.getArch())) {
      SkipNext = true;
      Linker.reportWarning("unsupported relocation in debug_info section.",
                           DMO);
      continue;
    }

    unsigned RelocSize = 1 << Obj.getAnyRelocationLength(MachOReloc);
    uint64_t Offset64 = Reloc.getOffset();
    if ((RelocSize != 4 && RelocSize != 8)) {
      Linker.reportWarning("unsupported relocation in debug_info section.",
                           DMO);
      continue;
    }
    uint32_t Offset = Offset64;
    // Mach-o uses REL relocations, the addend is at the relocation offset.
    uint64_t Addend = Data.getUnsigned(&Offset, RelocSize);
    uint64_t SymAddress;
    int64_t SymOffset;

    if (Obj.isRelocationScattered(MachOReloc)) {
      // The address of the base symbol for scattered relocations is
      // stored in the reloc itself. The actual addend will store the
      // base address plus the offset.
      SymAddress = Obj.getScatteredRelocationValue(MachOReloc);
      SymOffset = int64_t(Addend) - SymAddress;
    } else {
      SymAddress = Addend;
      SymOffset = 0;
    }

    auto Sym = Reloc.getSymbol();
    if (Sym != Obj.symbol_end()) {
      Expected<StringRef> SymbolName = Sym->getName();
      if (!SymbolName) {
        consumeError(SymbolName.takeError());
        Linker.reportWarning("error getting relocation symbol name.", DMO);
        continue;
      }
      if (const auto *Mapping = DMO.lookupSymbol(*SymbolName))
        ValidRelocs.emplace_back(Offset64, RelocSize, Addend, Mapping);
    } else if (const auto *Mapping = DMO.lookupObjectAddress(SymAddress)) {
      // Do not store the addend. The addend was the address of the symbol in
      // the object file, the address in the binary that is stored in the debug
      // map doesn't need to be offset.
      ValidRelocs.emplace_back(Offset64, RelocSize, SymOffset, Mapping);
    }
  }
}

/// Dispatch the valid relocation finding logic to the
/// appropriate handler depending on the object file format.
bool DwarfLinker::RelocationManager::findValidRelocs(
    const object::SectionRef &Section, const object::ObjectFile &Obj,
    const DebugMapObject &DMO) {
  // Dispatch to the right handler depending on the file type.
  if (auto *MachOObj = dyn_cast<object::MachOObjectFile>(&Obj))
    findValidRelocsMachO(Section, *MachOObj, DMO);
  else
    Linker.reportWarning(
        Twine("unsupported object file type: ") + Obj.getFileName(), DMO);

  if (ValidRelocs.empty())
    return false;

  // Sort the relocations by offset. We will walk the DIEs linearly in
  // the file, this allows us to just keep an index in the relocation
  // array that we advance during our walk, rather than resorting to
  // some associative container. See DwarfLinker::NextValidReloc.
  llvm::sort(ValidRelocs);
  return true;
}

/// Look for relocations in the debug_info section that match
/// entries in the debug map. These relocations will drive the Dwarf
/// link by indicating which DIEs refer to symbols present in the
/// linked binary.
/// \returns whether there are any valid relocations in the debug info.
bool DwarfLinker::RelocationManager::findValidRelocsInDebugInfo(
    const object::ObjectFile &Obj, const DebugMapObject &DMO) {
  // Find the debug_info section.
  for (const object::SectionRef &Section : Obj.sections()) {
    StringRef SectionName;
    Section.getName(SectionName);
    SectionName = SectionName.substr(SectionName.find_first_not_of("._"));
    if (SectionName != "debug_info")
      continue;
    return findValidRelocs(Section, Obj, DMO);
  }
  return false;
}

/// Checks that there is a relocation against an actual debug
/// map entry between \p StartOffset and \p NextOffset.
///
/// This function must be called with offsets in strictly ascending
/// order because it never looks back at relocations it already 'went past'.
/// \returns true and sets Info.InDebugMap if it is the case.
bool DwarfLinker::RelocationManager::hasValidRelocation(
    uint32_t StartOffset, uint32_t EndOffset, CompileUnit::DIEInfo &Info) {
  assert(NextValidReloc == 0 ||
         StartOffset > ValidRelocs[NextValidReloc - 1].Offset);
  if (NextValidReloc >= ValidRelocs.size())
    return false;

  uint64_t RelocOffset = ValidRelocs[NextValidReloc].Offset;

  // We might need to skip some relocs that we didn't consider. For
  // example the high_pc of a discarded DIE might contain a reloc that
  // is in the list because it actually corresponds to the start of a
  // function that is in the debug map.
  while (RelocOffset < StartOffset && NextValidReloc < ValidRelocs.size() - 1)
    RelocOffset = ValidRelocs[++NextValidReloc].Offset;

  if (RelocOffset < StartOffset || RelocOffset >= EndOffset)
    return false;

  const auto &ValidReloc = ValidRelocs[NextValidReloc++];
  const auto &Mapping = ValidReloc.Mapping->getValue();
  uint64_t ObjectAddress = Mapping.ObjectAddress
                               ? uint64_t(*Mapping.ObjectAddress)
                               : std::numeric_limits<uint64_t>::max();
  if (Linker.Options.Verbose)
    outs() << "Found valid debug map entry: " << ValidReloc.Mapping->getKey()
           << " "
           << format("\t%016" PRIx64 " => %016" PRIx64, ObjectAddress,
                     uint64_t(Mapping.BinaryAddress));

  Info.AddrAdjust = int64_t(Mapping.BinaryAddress) + ValidReloc.Addend;
  if (Mapping.ObjectAddress)
    Info.AddrAdjust -= ObjectAddress;
  Info.InDebugMap = true;
  return true;
}

/// Get the starting and ending (exclusive) offset for the
/// attribute with index \p Idx descibed by \p Abbrev. \p Offset is
/// supposed to point to the position of the first attribute described
/// by \p Abbrev.
/// \return [StartOffset, EndOffset) as a pair.
static std::pair<uint32_t, uint32_t>
getAttributeOffsets(const DWARFAbbreviationDeclaration *Abbrev, unsigned Idx,
                    unsigned Offset, const DWARFUnit &Unit) {
  DataExtractor Data = Unit.getDebugInfoExtractor();

  for (unsigned i = 0; i < Idx; ++i)
    DWARFFormValue::skipValue(Abbrev->getFormByIndex(i), Data, &Offset,
                              Unit.getFormParams());

  uint32_t End = Offset;
  DWARFFormValue::skipValue(Abbrev->getFormByIndex(Idx), Data, &End,
                            Unit.getFormParams());

  return std::make_pair(Offset, End);
}

/// Check if a variable describing DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DwarfLinker::shouldKeepVariableDIE(RelocationManager &RelocMgr,
                                            const DWARFDie &DIE,
                                            CompileUnit &Unit,
                                            CompileUnit::DIEInfo &MyInfo,
                                            unsigned Flags) {
  const auto *Abbrev = DIE.getAbbreviationDeclarationPtr();

  // Global variables with constant value can always be kept.
  if (!(Flags & TF_InFunctionScope) &&
      Abbrev->findAttributeIndex(dwarf::DW_AT_const_value)) {
    MyInfo.InDebugMap = true;
    return Flags | TF_Keep;
  }
  
  Optional<uint32_t> LocationIdx =
      Abbrev->findAttributeIndex(dwarf::DW_AT_location);
  if (!LocationIdx)
    return Flags;

  uint32_t Offset = DIE.getOffset() + getULEB128Size(Abbrev->getCode());
  const DWARFUnit &OrigUnit = Unit.getOrigUnit();
  uint32_t LocationOffset, LocationEndOffset;
  std::tie(LocationOffset, LocationEndOffset) =
      getAttributeOffsets(Abbrev, *LocationIdx, Offset, OrigUnit);

  // See if there is a relocation to a valid debug map entry inside
  // this variable's location. The order is important here. We want to
  // always check in the variable has a valid relocation, so that the
  // DIEInfo is filled. However, we don't want a static variable in a
  // function to force us to keep the enclosing function.
  if (!RelocMgr.hasValidRelocation(LocationOffset, LocationEndOffset, MyInfo) ||
      (Flags & TF_InFunctionScope))
    return Flags;

  if (Options.Verbose) {
    DIDumpOptions DumpOpts;
    DumpOpts.RecurseDepth = 0;
    DumpOpts.Verbose = Options.Verbose;
    DIE.dump(outs(), 8 /* Indent */, DumpOpts);
  }

  return Flags | TF_Keep;
}

/// Check if a function describing DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DwarfLinker::shouldKeepSubprogramDIE(
    RelocationManager &RelocMgr, RangesTy &Ranges, const DWARFDie &DIE,
    const DebugMapObject &DMO, CompileUnit &Unit, CompileUnit::DIEInfo &MyInfo,
    unsigned Flags) {
  const auto *Abbrev = DIE.getAbbreviationDeclarationPtr();

  Flags |= TF_InFunctionScope;

  Optional<uint32_t> LowPcIdx = Abbrev->findAttributeIndex(dwarf::DW_AT_low_pc);
  if (!LowPcIdx)
    return Flags;

  uint32_t Offset = DIE.getOffset() + getULEB128Size(Abbrev->getCode());
  DWARFUnit &OrigUnit = Unit.getOrigUnit();
  uint32_t LowPcOffset, LowPcEndOffset;
  std::tie(LowPcOffset, LowPcEndOffset) =
      getAttributeOffsets(Abbrev, *LowPcIdx, Offset, OrigUnit);

  auto LowPc = dwarf::toAddress(DIE.find(dwarf::DW_AT_low_pc));
  assert(LowPc.hasValue() && "low_pc attribute is not an address.");
  if (!LowPc ||
      !RelocMgr.hasValidRelocation(LowPcOffset, LowPcEndOffset, MyInfo))
    return Flags;

  if (Options.Verbose) {
    DIDumpOptions DumpOpts;
    DumpOpts.RecurseDepth = 0;
    DumpOpts.Verbose = Options.Verbose;
    DIE.dump(outs(), 8 /* Indent */, DumpOpts);
  }

  if (DIE.getTag() == dwarf::DW_TAG_label) {
    if (Unit.hasLabelAt(*LowPc))
      return Flags;
    // FIXME: dsymutil-classic compat. dsymutil-classic doesn't consider labels
    // that don't fall into the CU's aranges. This is wrong IMO. Debug info
    // generation bugs aside, this is really wrong in the case of labels, where
    // a label marking the end of a function will have a PC == CU's high_pc.
    if (dwarf::toAddress(OrigUnit.getUnitDIE().find(dwarf::DW_AT_high_pc))
            .getValueOr(UINT64_MAX) <= LowPc)
      return Flags;
    Unit.addLabelLowPc(*LowPc, MyInfo.AddrAdjust);
    return Flags | TF_Keep;
  }

  Flags |= TF_Keep;

  Optional<uint64_t> HighPc = DIE.getHighPC(*LowPc);
  if (!HighPc) {
    reportWarning("Function without high_pc. Range will be discarded.\n", DMO,
                  &DIE);
    return Flags;
  }

  // Replace the debug map range with a more accurate one.
  Ranges[*LowPc] = DebugMapObjectRange(*HighPc, MyInfo.AddrAdjust);
  Unit.addFunctionRange(*LowPc, *HighPc, MyInfo.AddrAdjust);
  return Flags;
}

/// Check if a DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DwarfLinker::shouldKeepDIE(RelocationManager &RelocMgr,
                                    RangesTy &Ranges, const DWARFDie &DIE,
                                    const DebugMapObject &DMO,
                                    CompileUnit &Unit,
                                    CompileUnit::DIEInfo &MyInfo,
                                    unsigned Flags) {
  switch (DIE.getTag()) {
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_variable:
    return shouldKeepVariableDIE(RelocMgr, DIE, Unit, MyInfo, Flags);
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_label:
    return shouldKeepSubprogramDIE(RelocMgr, Ranges, DIE, DMO, Unit, MyInfo,
                                   Flags);
  case dwarf::DW_TAG_base_type:
    // DWARF Expressions may reference basic types, but scanning them
    // is expensive. Basic types are tiny, so just keep all of them.
  case dwarf::DW_TAG_imported_module:
  case dwarf::DW_TAG_imported_declaration:
  case dwarf::DW_TAG_imported_unit:
    // We always want to keep these.
    return Flags | TF_Keep;
  default:
    break;
  }

  return Flags;
}

/// Mark the passed DIE as well as all the ones it depends on
/// as kept.
///
/// This function is called by lookForDIEsToKeep on DIEs that are
/// newly discovered to be needed in the link. It recursively calls
/// back to lookForDIEsToKeep while adding TF_DependencyWalk to the
/// TraversalFlags to inform it that it's not doing the primary DIE
/// tree walk.
void DwarfLinker::keepDIEAndDependencies(
    RelocationManager &RelocMgr, RangesTy &Ranges, const UnitListTy &Units,
    const DWARFDie &Die, CompileUnit::DIEInfo &MyInfo,
    const DebugMapObject &DMO, CompileUnit &CU, bool UseODR) {
  DWARFUnit &Unit = CU.getOrigUnit();
  MyInfo.Keep = true;

  // We're looking for incomplete types.
  MyInfo.Incomplete = Die.getTag() != dwarf::DW_TAG_subprogram &&
                      Die.getTag() != dwarf::DW_TAG_member &&
                      dwarf::toUnsigned(Die.find(dwarf::DW_AT_declaration), 0);

  // First mark all the parent chain as kept.
  unsigned AncestorIdx = MyInfo.ParentIdx;
  while (!CU.getInfo(AncestorIdx).Keep) {
    unsigned ODRFlag = UseODR ? TF_ODR : 0;
    lookForDIEsToKeep(RelocMgr, Ranges, Units, Unit.getDIEAtIndex(AncestorIdx),
                      DMO, CU,
                      TF_ParentWalk | TF_Keep | TF_DependencyWalk | ODRFlag);
    AncestorIdx = CU.getInfo(AncestorIdx).ParentIdx;
  }

  // Then we need to mark all the DIEs referenced by this DIE's
  // attributes as kept.
  DWARFDataExtractor Data = Unit.getDebugInfoExtractor();
  const auto *Abbrev = Die.getAbbreviationDeclarationPtr();
  uint32_t Offset = Die.getOffset() + getULEB128Size(Abbrev->getCode());

  // Mark all DIEs referenced through attributes as kept.
  for (const auto &AttrSpec : Abbrev->attributes()) {
    DWARFFormValue Val(AttrSpec.Form);
    if (!Val.isFormClass(DWARFFormValue::FC_Reference) ||
        AttrSpec.Attr == dwarf::DW_AT_sibling) {
      DWARFFormValue::skipValue(AttrSpec.Form, Data, &Offset,
                                Unit.getFormParams());
      continue;
    }

    Val.extractValue(Data, &Offset, Unit.getFormParams(), &Unit);
    CompileUnit *ReferencedCU;
    if (auto RefDie = resolveDIEReference(*this, DMO, Units, Val, Unit, Die,
                                          ReferencedCU)) {
      uint32_t RefIdx = ReferencedCU->getOrigUnit().getDIEIndex(RefDie);
      CompileUnit::DIEInfo &Info = ReferencedCU->getInfo(RefIdx);
      bool IsModuleRef = Info.Ctxt && Info.Ctxt->getCanonicalDIEOffset() &&
                         Info.Ctxt->isDefinedInClangModule();
      // If the referenced DIE has a DeclContext that has already been
      // emitted, then do not keep the one in this CU. We'll link to
      // the canonical DIE in cloneDieReferenceAttribute.
      // FIXME: compatibility with dsymutil-classic. UseODR shouldn't
      // be necessary and could be advantageously replaced by
      // ReferencedCU->hasODR() && CU.hasODR().
      // FIXME: compatibility with dsymutil-classic. There is no
      // reason not to unique ref_addr references.
      if (AttrSpec.Form != dwarf::DW_FORM_ref_addr && (UseODR || IsModuleRef) &&
          Info.Ctxt &&
          Info.Ctxt != ReferencedCU->getInfo(Info.ParentIdx).Ctxt &&
          Info.Ctxt->getCanonicalDIEOffset() && isODRAttribute(AttrSpec.Attr))
        continue;

      // Keep a module forward declaration if there is no definition.
      if (!(isODRAttribute(AttrSpec.Attr) && Info.Ctxt &&
            Info.Ctxt->getCanonicalDIEOffset()))
        Info.Prune = false;

      unsigned ODRFlag = UseODR ? TF_ODR : 0;
      lookForDIEsToKeep(RelocMgr, Ranges, Units, RefDie, DMO, *ReferencedCU,
                        TF_Keep | TF_DependencyWalk | ODRFlag);

      // The incomplete property is propagated if the current DIE is complete
      // but references an incomplete DIE.
      if (Info.Incomplete && !MyInfo.Incomplete &&
          (Die.getTag() == dwarf::DW_TAG_typedef ||
           Die.getTag() == dwarf::DW_TAG_member ||
           Die.getTag() == dwarf::DW_TAG_reference_type ||
           Die.getTag() == dwarf::DW_TAG_ptr_to_member_type ||
           Die.getTag() == dwarf::DW_TAG_pointer_type))
        MyInfo.Incomplete = true;
    }
  }
}

namespace {
/// This class represents an item in the work list. In addition to it's obvious
/// purpose of representing the state associated with a particular run of the
/// work loop, it also serves as a marker to indicate that we should run the
/// "continuation" code.
///
/// Originally, the latter was lambda which allowed arbitrary code to be run.
/// Because we always need to run the exact same code, it made more sense to
/// use a boolean and repurpose the already existing DIE field.
struct WorklistItem {
  DWARFDie Die;
  unsigned Flags;
  bool IsContinuation;
  CompileUnit::DIEInfo *ChildInfo = nullptr;

  /// Construct a classic worklist item.
  WorklistItem(DWARFDie Die, unsigned Flags)
      : Die(Die), Flags(Flags), IsContinuation(false){};

  /// Creates a continuation marker.
  WorklistItem(DWARFDie Die) : Die(Die), IsContinuation(true){};
};
} // namespace

// Helper that updates the completeness of the current DIE. It depends on the
// fact that the incompletness of its children is already computed.
static void updateIncompleteness(const DWARFDie &Die,
                                 CompileUnit::DIEInfo &ChildInfo,
                                 CompileUnit &CU) {
  // Only propagate incomplete members.
  if (Die.getTag() != dwarf::DW_TAG_structure_type &&
      Die.getTag() != dwarf::DW_TAG_class_type)
    return;

  unsigned Idx = CU.getOrigUnit().getDIEIndex(Die);
  CompileUnit::DIEInfo &MyInfo = CU.getInfo(Idx);

  if (MyInfo.Incomplete)
    return;

  if (ChildInfo.Incomplete || ChildInfo.Prune)
    MyInfo.Incomplete = true;
}

/// Recursively walk the \p DIE tree and look for DIEs to
/// keep. Store that information in \p CU's DIEInfo.
///
/// This function is the entry point of the DIE selection
/// algorithm. It is expected to walk the DIE tree in file order and
/// (though the mediation of its helper) call hasValidRelocation() on
/// each DIE that might be a 'root DIE' (See DwarfLinker class
/// comment).
/// While walking the dependencies of root DIEs, this function is
/// also called, but during these dependency walks the file order is
/// not respected. The TF_DependencyWalk flag tells us which kind of
/// traversal we are currently doing.
///
/// The return value indicates whether the DIE is incomplete.
void DwarfLinker::lookForDIEsToKeep(RelocationManager &RelocMgr,
                                    RangesTy &Ranges, const UnitListTy &Units,
                                    const DWARFDie &Die,
                                    const DebugMapObject &DMO, CompileUnit &CU,
                                    unsigned Flags) {
  // LIFO work list.
  SmallVector<WorklistItem, 4> Worklist;
  Worklist.emplace_back(Die, Flags);

  while (!Worklist.empty()) {
    WorklistItem Current = Worklist.back();
    Worklist.pop_back();

    if (Current.IsContinuation) {
      updateIncompleteness(Current.Die, *Current.ChildInfo, CU);
      continue;
    }

    unsigned Idx = CU.getOrigUnit().getDIEIndex(Current.Die);
    CompileUnit::DIEInfo &MyInfo = CU.getInfo(Idx);

    // At this point we are guaranteed to have a continuation marker before us
    // in the worklist, except for the last DIE.
    if (!Worklist.empty())
      Worklist.back().ChildInfo = &MyInfo;

    if (MyInfo.Prune)
      continue;

    // If the Keep flag is set, we are marking a required DIE's dependencies.
    // If our target is already marked as kept, we're all set.
    bool AlreadyKept = MyInfo.Keep;
    if ((Current.Flags & TF_DependencyWalk) && AlreadyKept)
      continue;

    // We must not call shouldKeepDIE while called from keepDIEAndDependencies,
    // because it would screw up the relocation finding logic.
    if (!(Current.Flags & TF_DependencyWalk))
      Current.Flags = shouldKeepDIE(RelocMgr, Ranges, Current.Die, DMO, CU,
                                    MyInfo, Current.Flags);

    // If it is a newly kept DIE mark it as well as all its dependencies as
    // kept.
    if (!AlreadyKept && (Current.Flags & TF_Keep)) {
      bool UseOdr = (Current.Flags & TF_DependencyWalk)
                        ? (Current.Flags & TF_ODR)
                        : CU.hasODR();
      keepDIEAndDependencies(RelocMgr, Ranges, Units, Current.Die, MyInfo, DMO,
                             CU, UseOdr);
    }

    // The TF_ParentWalk flag tells us that we are currently walking up
    // the parent chain of a required DIE, and we don't want to mark all
    // the children of the parents as kept (consider for example a
    // DW_TAG_namespace node in the parent chain). There are however a
    // set of DIE types for which we want to ignore that directive and still
    // walk their children.
    if (dieNeedsChildrenToBeMeaningful(Current.Die.getTag()))
      Current.Flags &= ~TF_ParentWalk;

    if (!Current.Die.hasChildren() || (Current.Flags & TF_ParentWalk))
      continue;

    // Add children in reverse order to the worklist to effectively process
    // them in order.
    for (auto Child : reverse(Current.Die.children())) {
      // Add continuation marker before every child to calculate incompleteness
      // after the last child is processed. We can't store this information in
      // the same item because we might have to process other continuations
      // first.
      Worklist.emplace_back(Current.Die);
      Worklist.emplace_back(Child, Current.Flags);
    }
  }
}

/// Assign an abbreviation number to \p Abbrev.
///
/// Our DIEs get freed after every DebugMapObject has been processed,
/// thus the FoldingSet we use to unique DIEAbbrevs cannot refer to
/// the instances hold by the DIEs. When we encounter an abbreviation
/// that we don't know, we create a permanent copy of it.
void DwarfLinker::AssignAbbrev(DIEAbbrev &Abbrev) {
  // Check the set for priors.
  FoldingSetNodeID ID;
  Abbrev.Profile(ID);
  void *InsertToken;
  DIEAbbrev *InSet = AbbreviationsSet.FindNodeOrInsertPos(ID, InsertToken);

  // If it's newly added.
  if (InSet) {
    // Assign existing abbreviation number.
    Abbrev.setNumber(InSet->getNumber());
  } else {
    // Add to abbreviation list.
    Abbreviations.push_back(
        llvm::make_unique<DIEAbbrev>(Abbrev.getTag(), Abbrev.hasChildren()));
    for (const auto &Attr : Abbrev.getData())
      Abbreviations.back()->AddAttribute(Attr.getAttribute(), Attr.getForm());
    AbbreviationsSet.InsertNode(Abbreviations.back().get(), InsertToken);
    // Assign the unique abbreviation number.
    Abbrev.setNumber(Abbreviations.size());
    Abbreviations.back()->setNumber(Abbreviations.size());
  }
}

unsigned DwarfLinker::DIECloner::cloneStringAttribute(
    DIE &Die, AttributeSpec AttrSpec, const DWARFFormValue &Val,
    const DWARFUnit &U, OffsetsStringPool &StringPool, AttributesInfo &Info) {
  // Switch everything to out of line strings.
  const char *String = *Val.getAsCString();
  auto StringEntry = StringPool.getEntry(String);

  // Update attributes info.
  if (AttrSpec.Attr == dwarf::DW_AT_name)
    Info.Name = StringEntry;
  else if (AttrSpec.Attr == dwarf::DW_AT_MIPS_linkage_name ||
           AttrSpec.Attr == dwarf::DW_AT_linkage_name)
    Info.MangledName = StringEntry;

  Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr), dwarf::DW_FORM_strp,
               DIEInteger(StringEntry.getOffset()));

  return 4;
}

unsigned DwarfLinker::DIECloner::cloneDieReferenceAttribute(
    DIE &Die, const DWARFDie &InputDIE, AttributeSpec AttrSpec,
    unsigned AttrSize, const DWARFFormValue &Val, const DebugMapObject &DMO,
    CompileUnit &Unit) {
  const DWARFUnit &U = Unit.getOrigUnit();
  uint32_t Ref = *Val.getAsReference();
  DIE *NewRefDie = nullptr;
  CompileUnit *RefUnit = nullptr;
  DeclContext *Ctxt = nullptr;

  DWARFDie RefDie =
      resolveDIEReference(Linker, DMO, CompileUnits, Val, U, InputDIE, RefUnit);

  // If the referenced DIE is not found,  drop the attribute.
  if (!RefDie || AttrSpec.Attr == dwarf::DW_AT_sibling)
    return 0;

  unsigned Idx = RefUnit->getOrigUnit().getDIEIndex(RefDie);
  CompileUnit::DIEInfo &RefInfo = RefUnit->getInfo(Idx);

  // If we already have emitted an equivalent DeclContext, just point
  // at it.
  if (isODRAttribute(AttrSpec.Attr)) {
    Ctxt = RefInfo.Ctxt;
    if (Ctxt && Ctxt->getCanonicalDIEOffset()) {
      DIEInteger Attr(Ctxt->getCanonicalDIEOffset());
      Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
                   dwarf::DW_FORM_ref_addr, Attr);
      return U.getRefAddrByteSize();
    }
  }

  if (!RefInfo.Clone) {
    assert(Ref > InputDIE.getOffset());
    // We haven't cloned this DIE yet. Just create an empty one and
    // store it. It'll get really cloned when we process it.
    RefInfo.Clone = DIE::get(DIEAlloc, dwarf::Tag(RefDie.getTag()));
  }
  NewRefDie = RefInfo.Clone;

  if (AttrSpec.Form == dwarf::DW_FORM_ref_addr ||
      (Unit.hasODR() && isODRAttribute(AttrSpec.Attr))) {
    // We cannot currently rely on a DIEEntry to emit ref_addr
    // references, because the implementation calls back to DwarfDebug
    // to find the unit offset. (We don't have a DwarfDebug)
    // FIXME: we should be able to design DIEEntry reliance on
    // DwarfDebug away.
    uint64_t Attr;
    if (Ref < InputDIE.getOffset()) {
      // We must have already cloned that DIE.
      uint32_t NewRefOffset =
          RefUnit->getStartOffset() + NewRefDie->getOffset();
      Attr = NewRefOffset;
      Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
                   dwarf::DW_FORM_ref_addr, DIEInteger(Attr));
    } else {
      // A forward reference. Note and fixup later.
      Attr = 0xBADDEF;
      Unit.noteForwardReference(
          NewRefDie, RefUnit, Ctxt,
          Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
                       dwarf::DW_FORM_ref_addr, DIEInteger(Attr)));
    }
    return U.getRefAddrByteSize();
  }

  Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
               dwarf::Form(AttrSpec.Form), DIEEntry(*NewRefDie));
  return AttrSize;
}

void DwarfLinker::DIECloner::cloneExpression(
    DataExtractor &Data, DWARFExpression Expression, const DebugMapObject &DMO,
    CompileUnit &Unit, SmallVectorImpl<uint8_t> &OutputBuffer) {
  using Encoding = DWARFExpression::Operation::Encoding;

  uint32_t OpOffset = 0;
  for (auto &Op : Expression) {
    auto Description = Op.getDescription();
    // DW_OP_const_type is variable-length and has 3
    // operands. DWARFExpression thus far only supports 2.
    auto Op0 = Description.Op[0];
    auto Op1 = Description.Op[1];
    if ((Op0 == Encoding::BaseTypeRef && Op1 != Encoding::SizeNA) ||
        (Op1 == Encoding::BaseTypeRef && Op0 != Encoding::Size1))
      Linker.reportWarning("Unsupported DW_OP encoding.", DMO);

    if ((Op0 == Encoding::BaseTypeRef && Op1 == Encoding::SizeNA) ||
        (Op1 == Encoding::BaseTypeRef && Op0 == Encoding::Size1)) {
      // This code assumes that the other non-typeref operand fits into 1 byte.
      assert(OpOffset < Op.getEndOffset());
      uint32_t ULEBsize = Op.getEndOffset() - OpOffset - 1;
      assert(ULEBsize <= 16);

      // Copy over the operation.
      OutputBuffer.push_back(Op.getCode());
      uint64_t RefOffset;
      if (Op1 == Encoding::SizeNA) {
        RefOffset = Op.getRawOperand(0);
      } else {
        OutputBuffer.push_back(Op.getRawOperand(0));
        RefOffset = Op.getRawOperand(1);
      }
      auto RefDie = Unit.getOrigUnit().getDIEForOffset(RefOffset);
      uint32_t RefIdx = Unit.getOrigUnit().getDIEIndex(RefDie);
      CompileUnit::DIEInfo &Info = Unit.getInfo(RefIdx);
      uint32_t Offset = 0;
      if (DIE *Clone = Info.Clone)
        Offset = Clone->getOffset();
      else
        Linker.reportWarning("base type ref doesn't point to DW_TAG_base_type.",
                             DMO);
      uint8_t ULEB[16];
      unsigned RealSize = encodeULEB128(Offset, ULEB, ULEBsize);
      if (RealSize > ULEBsize) {
        // Emit the generic type as a fallback.
        RealSize = encodeULEB128(0, ULEB, ULEBsize);
        Linker.reportWarning("base type ref doesn't fit.", DMO);
      }
      assert(RealSize == ULEBsize && "padding failed");
      ArrayRef<uint8_t> ULEBbytes(ULEB, ULEBsize);
      OutputBuffer.append(ULEBbytes.begin(), ULEBbytes.end());
    } else {
      // Copy over everything else unmodified.
      StringRef Bytes = Data.getData().slice(OpOffset, Op.getEndOffset());
      OutputBuffer.append(Bytes.begin(), Bytes.end());
    }
    OpOffset = Op.getEndOffset();
  }
}

unsigned DwarfLinker::DIECloner::cloneBlockAttribute(
    DIE &Die, const DebugMapObject &DMO, CompileUnit &Unit,
    AttributeSpec AttrSpec, const DWARFFormValue &Val, unsigned AttrSize,
    bool IsLittleEndian) {
  DIEValueList *Attr;
  DIEValue Value;
  DIELoc *Loc = nullptr;
  DIEBlock *Block = nullptr;
  if (AttrSpec.Form == dwarf::DW_FORM_exprloc) {
    Loc = new (DIEAlloc) DIELoc;
    Linker.DIELocs.push_back(Loc);
  } else {
    Block = new (DIEAlloc) DIEBlock;
    Linker.DIEBlocks.push_back(Block);
  }
  Attr = Loc ? static_cast<DIEValueList *>(Loc)
             : static_cast<DIEValueList *>(Block);

  if (Loc)
    Value = DIEValue(dwarf::Attribute(AttrSpec.Attr),
                     dwarf::Form(AttrSpec.Form), Loc);
  else
    Value = DIEValue(dwarf::Attribute(AttrSpec.Attr),
                     dwarf::Form(AttrSpec.Form), Block);

  // If the block is a DWARF Expression, clone it into the temporary
  // buffer using cloneExpression(), otherwise copy the data directly.
  SmallVector<uint8_t, 32> Buffer;
  ArrayRef<uint8_t> Bytes = *Val.getAsBlock();
  if (DWARFAttribute::mayHaveLocationDescription(AttrSpec.Attr) &&
      (Val.isFormClass(DWARFFormValue::FC_Block) ||
       Val.isFormClass(DWARFFormValue::FC_Exprloc))) {
    DWARFUnit &OrigUnit = Unit.getOrigUnit();
    DataExtractor Data(StringRef((const char *)Bytes.data(), Bytes.size()),
                       IsLittleEndian, OrigUnit.getAddressByteSize());
    DWARFExpression Expr(Data, OrigUnit.getVersion(),
                         OrigUnit.getAddressByteSize());
    cloneExpression(Data, Expr, DMO, Unit, Buffer);
    Bytes = Buffer;
  }
  for (auto Byte : Bytes)
    Attr->addValue(DIEAlloc, static_cast<dwarf::Attribute>(0),
                   dwarf::DW_FORM_data1, DIEInteger(Byte));

  // FIXME: If DIEBlock and DIELoc just reuses the Size field of
  // the DIE class, this if could be replaced by
  // Attr->setSize(Bytes.size()).
  if (Linker.Streamer) {
    auto *AsmPrinter = &Linker.Streamer->getAsmPrinter();
    if (Loc)
      Loc->ComputeSize(AsmPrinter);
    else
      Block->ComputeSize(AsmPrinter);
  }
  Die.addValue(DIEAlloc, Value);
  return AttrSize;
}

unsigned DwarfLinker::DIECloner::cloneAddressAttribute(
    DIE &Die, AttributeSpec AttrSpec, const DWARFFormValue &Val,
    const CompileUnit &Unit, AttributesInfo &Info) {
  uint64_t Addr = *Val.getAsAddress();

  if (LLVM_UNLIKELY(Linker.Options.Update)) {
    if (AttrSpec.Attr == dwarf::DW_AT_low_pc)
      Info.HasLowPc = true;
    Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
                 dwarf::Form(AttrSpec.Form), DIEInteger(Addr));
    return Unit.getOrigUnit().getAddressByteSize();
  }

  if (AttrSpec.Attr == dwarf::DW_AT_low_pc) {
    if (Die.getTag() == dwarf::DW_TAG_inlined_subroutine ||
        Die.getTag() == dwarf::DW_TAG_lexical_block)
      // The low_pc of a block or inline subroutine might get
      // relocated because it happens to match the low_pc of the
      // enclosing subprogram. To prevent issues with that, always use
      // the low_pc from the input DIE if relocations have been applied.
      Addr = (Info.OrigLowPc != std::numeric_limits<uint64_t>::max()
                  ? Info.OrigLowPc
                  : Addr) +
             Info.PCOffset;
    else if (Die.getTag() == dwarf::DW_TAG_compile_unit) {
      Addr = Unit.getLowPc();
      if (Addr == std::numeric_limits<uint64_t>::max())
        return 0;
    }
    Info.HasLowPc = true;
  } else if (AttrSpec.Attr == dwarf::DW_AT_high_pc) {
    if (Die.getTag() == dwarf::DW_TAG_compile_unit) {
      if (uint64_t HighPc = Unit.getHighPc())
        Addr = HighPc;
      else
        return 0;
    } else
      // If we have a high_pc recorded for the input DIE, use
      // it. Otherwise (when no relocations where applied) just use the
      // one we just decoded.
      Addr = (Info.OrigHighPc ? Info.OrigHighPc : Addr) + Info.PCOffset;
  }

  Die.addValue(DIEAlloc, static_cast<dwarf::Attribute>(AttrSpec.Attr),
               static_cast<dwarf::Form>(AttrSpec.Form), DIEInteger(Addr));
  return Unit.getOrigUnit().getAddressByteSize();
}

unsigned DwarfLinker::DIECloner::cloneScalarAttribute(
    DIE &Die, const DWARFDie &InputDIE, const DebugMapObject &DMO,
    CompileUnit &Unit, AttributeSpec AttrSpec, const DWARFFormValue &Val,
    unsigned AttrSize, AttributesInfo &Info) {
  uint64_t Value;

  if (LLVM_UNLIKELY(Linker.Options.Update)) {
    if (auto OptionalValue = Val.getAsUnsignedConstant())
      Value = *OptionalValue;
    else if (auto OptionalValue = Val.getAsSignedConstant())
      Value = *OptionalValue;
    else if (auto OptionalValue = Val.getAsSectionOffset())
      Value = *OptionalValue;
    else {
      Linker.reportWarning(
          "Unsupported scalar attribute form. Dropping attribute.", DMO,
          &InputDIE);
      return 0;
    }
    if (AttrSpec.Attr == dwarf::DW_AT_declaration && Value)
      Info.IsDeclaration = true;
    Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
                 dwarf::Form(AttrSpec.Form), DIEInteger(Value));
    return AttrSize;
  }

  if (AttrSpec.Attr == dwarf::DW_AT_high_pc &&
      Die.getTag() == dwarf::DW_TAG_compile_unit) {
    if (Unit.getLowPc() == -1ULL)
      return 0;
    // Dwarf >= 4 high_pc is an size, not an address.
    Value = Unit.getHighPc() - Unit.getLowPc();
  } else if (AttrSpec.Form == dwarf::DW_FORM_sec_offset)
    Value = *Val.getAsSectionOffset();
  else if (AttrSpec.Form == dwarf::DW_FORM_sdata)
    Value = *Val.getAsSignedConstant();
  else if (auto OptionalValue = Val.getAsUnsignedConstant())
    Value = *OptionalValue;
  else {
    Linker.reportWarning(
        "Unsupported scalar attribute form. Dropping attribute.", DMO,
        &InputDIE);
    return 0;
  }
  PatchLocation Patch =
      Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
                   dwarf::Form(AttrSpec.Form), DIEInteger(Value));
  if (AttrSpec.Attr == dwarf::DW_AT_ranges) {
    Unit.noteRangeAttribute(Die, Patch);
    Info.HasRanges = true;
  }

  // A more generic way to check for location attributes would be
  // nice, but it's very unlikely that any other attribute needs a
  // location list.
  // FIXME: use DWARFAttribute::mayHaveLocationDescription().
  else if (AttrSpec.Attr == dwarf::DW_AT_location ||
         AttrSpec.Attr == dwarf::DW_AT_frame_base)
    Unit.noteLocationAttribute(Patch, Info.PCOffset);
  else if (AttrSpec.Attr == dwarf::DW_AT_declaration && Value)
    Info.IsDeclaration = true;

  return AttrSize;
}

/// Clone \p InputDIE's attribute described by \p AttrSpec with
/// value \p Val, and add it to \p Die.
/// \returns the size of the cloned attribute.
unsigned DwarfLinker::DIECloner::cloneAttribute(
    DIE &Die, const DWARFDie &InputDIE, const DebugMapObject &DMO,
    CompileUnit &Unit, OffsetsStringPool &StringPool, const DWARFFormValue &Val,
    const AttributeSpec AttrSpec, unsigned AttrSize, AttributesInfo &Info,
    bool IsLittleEndian) {
  const DWARFUnit &U = Unit.getOrigUnit();

  switch (AttrSpec.Form) {
  case dwarf::DW_FORM_strp:
  case dwarf::DW_FORM_string:
    return cloneStringAttribute(Die, AttrSpec, Val, U, StringPool, Info);
  case dwarf::DW_FORM_ref_addr:
  case dwarf::DW_FORM_ref1:
  case dwarf::DW_FORM_ref2:
  case dwarf::DW_FORM_ref4:
  case dwarf::DW_FORM_ref8:
    return cloneDieReferenceAttribute(Die, InputDIE, AttrSpec, AttrSize, Val,
                                      DMO, Unit);
  case dwarf::DW_FORM_block:
  case dwarf::DW_FORM_block1:
  case dwarf::DW_FORM_block2:
  case dwarf::DW_FORM_block4:
  case dwarf::DW_FORM_exprloc:
    return cloneBlockAttribute(Die, DMO, Unit, AttrSpec, Val, AttrSize,
                               IsLittleEndian);
  case dwarf::DW_FORM_addr:
    return cloneAddressAttribute(Die, AttrSpec, Val, Unit, Info);
  case dwarf::DW_FORM_data1:
  case dwarf::DW_FORM_data2:
  case dwarf::DW_FORM_data4:
  case dwarf::DW_FORM_data8:
  case dwarf::DW_FORM_udata:
  case dwarf::DW_FORM_sdata:
  case dwarf::DW_FORM_sec_offset:
  case dwarf::DW_FORM_flag:
  case dwarf::DW_FORM_flag_present:
    return cloneScalarAttribute(Die, InputDIE, DMO, Unit, AttrSpec, Val,
                                AttrSize, Info);
  default:
    Linker.reportWarning(
        "Unsupported attribute form in cloneAttribute. Dropping.", DMO,
        &InputDIE);
  }

  return 0;
}

/// Apply the valid relocations found by findValidRelocs() to
/// the buffer \p Data, taking into account that Data is at \p BaseOffset
/// in the debug_info section.
///
/// Like for findValidRelocs(), this function must be called with
/// monotonic \p BaseOffset values.
///
/// \returns whether any reloc has been applied.
bool DwarfLinker::RelocationManager::applyValidRelocs(
    MutableArrayRef<char> Data, uint32_t BaseOffset, bool IsLittleEndian) {
  assert((NextValidReloc == 0 ||
          BaseOffset > ValidRelocs[NextValidReloc - 1].Offset) &&
         "BaseOffset should only be increasing.");
  if (NextValidReloc >= ValidRelocs.size())
    return false;

  // Skip relocs that haven't been applied.
  while (NextValidReloc < ValidRelocs.size() &&
         ValidRelocs[NextValidReloc].Offset < BaseOffset)
    ++NextValidReloc;

  bool Applied = false;
  uint64_t EndOffset = BaseOffset + Data.size();
  while (NextValidReloc < ValidRelocs.size() &&
         ValidRelocs[NextValidReloc].Offset >= BaseOffset &&
         ValidRelocs[NextValidReloc].Offset < EndOffset) {
    const auto &ValidReloc = ValidRelocs[NextValidReloc++];
    assert(ValidReloc.Offset - BaseOffset < Data.size());
    assert(ValidReloc.Offset - BaseOffset + ValidReloc.Size <= Data.size());
    char Buf[8];
    uint64_t Value = ValidReloc.Mapping->getValue().BinaryAddress;
    Value += ValidReloc.Addend;
    for (unsigned i = 0; i != ValidReloc.Size; ++i) {
      unsigned Index = IsLittleEndian ? i : (ValidReloc.Size - i - 1);
      Buf[i] = uint8_t(Value >> (Index * 8));
    }
    assert(ValidReloc.Size <= sizeof(Buf));
    memcpy(&Data[ValidReloc.Offset - BaseOffset], Buf, ValidReloc.Size);
    Applied = true;
  }

  return Applied;
}

static bool isObjCSelector(StringRef Name) {
  return Name.size() > 2 && (Name[0] == '-' || Name[0] == '+') &&
         (Name[1] == '[');
}

void DwarfLinker::DIECloner::addObjCAccelerator(CompileUnit &Unit,
                                                const DIE *Die,
                                                DwarfStringPoolEntryRef Name,
                                                OffsetsStringPool &StringPool,
                                                bool SkipPubSection) {
  assert(isObjCSelector(Name.getString()) && "not an objc selector");
  // Objective C method or class function.
  // "- [Class(Category) selector :withArg ...]"
  StringRef ClassNameStart(Name.getString().drop_front(2));
  size_t FirstSpace = ClassNameStart.find(' ');
  if (FirstSpace == StringRef::npos)
    return;

  StringRef SelectorStart(ClassNameStart.data() + FirstSpace + 1);
  if (!SelectorStart.size())
    return;

  StringRef Selector(SelectorStart.data(), SelectorStart.size() - 1);
  Unit.addNameAccelerator(Die, StringPool.getEntry(Selector), SkipPubSection);

  // Add an entry for the class name that points to this
  // method/class function.
  StringRef ClassName(ClassNameStart.data(), FirstSpace);
  Unit.addObjCAccelerator(Die, StringPool.getEntry(ClassName), SkipPubSection);

  if (ClassName[ClassName.size() - 1] == ')') {
    size_t OpenParens = ClassName.find('(');
    if (OpenParens != StringRef::npos) {
      StringRef ClassNameNoCategory(ClassName.data(), OpenParens);
      Unit.addObjCAccelerator(Die, StringPool.getEntry(ClassNameNoCategory),
                              SkipPubSection);

      std::string MethodNameNoCategory(Name.getString().data(), OpenParens + 2);
      // FIXME: The missing space here may be a bug, but
      //        dsymutil-classic also does it this way.
      MethodNameNoCategory.append(SelectorStart);
      Unit.addNameAccelerator(Die, StringPool.getEntry(MethodNameNoCategory),
                              SkipPubSection);
    }
  }
}

static bool
shouldSkipAttribute(DWARFAbbreviationDeclaration::AttributeSpec AttrSpec,
                    uint16_t Tag, bool InDebugMap, bool SkipPC,
                    bool InFunctionScope) {
  switch (AttrSpec.Attr) {
  default:
    return false;
  case dwarf::DW_AT_low_pc:
  case dwarf::DW_AT_high_pc:
  case dwarf::DW_AT_ranges:
    return SkipPC;
  case dwarf::DW_AT_location:
  case dwarf::DW_AT_frame_base:
    // FIXME: for some reason dsymutil-classic keeps the location attributes
    // when they are of block type (i.e. not location lists). This is totally
    // wrong for globals where we will keep a wrong address. It is mostly
    // harmless for locals, but there is no point in keeping these anyway when
    // the function wasn't linked.
    return (SkipPC || (!InFunctionScope && Tag == dwarf::DW_TAG_variable &&
                       !InDebugMap)) &&
           !DWARFFormValue(AttrSpec.Form).isFormClass(DWARFFormValue::FC_Block);
  }
}

DIE *DwarfLinker::DIECloner::cloneDIE(
    const DWARFDie &InputDIE, const DebugMapObject &DMO, CompileUnit &Unit,
    OffsetsStringPool &StringPool, int64_t PCOffset, uint32_t OutOffset,
    unsigned Flags, bool IsLittleEndian, DIE *Die) {
  DWARFUnit &U = Unit.getOrigUnit();
  unsigned Idx = U.getDIEIndex(InputDIE);
  CompileUnit::DIEInfo &Info = Unit.getInfo(Idx);

  // Should the DIE appear in the output?
  if (!Unit.getInfo(Idx).Keep)
    return nullptr;

  uint32_t Offset = InputDIE.getOffset();
  assert(!(Die && Info.Clone) && "Can't supply a DIE and a cloned DIE");
  if (!Die) {
    // The DIE might have been already created by a forward reference
    // (see cloneDieReferenceAttribute()).
    if (!Info.Clone)
      Info.Clone = DIE::get(DIEAlloc, dwarf::Tag(InputDIE.getTag()));
    Die = Info.Clone;
  }

  assert(Die->getTag() == InputDIE.getTag());
  Die->setOffset(OutOffset);
  if ((Unit.hasODR() || Unit.isClangModule()) && !Info.Incomplete &&
      Die->getTag() != dwarf::DW_TAG_namespace && Info.Ctxt &&
      Info.Ctxt != Unit.getInfo(Info.ParentIdx).Ctxt &&
      !Info.Ctxt->getCanonicalDIEOffset()) {
    // We are about to emit a DIE that is the root of its own valid
    // DeclContext tree. Make the current offset the canonical offset
    // for this context.
    Info.Ctxt->setCanonicalDIEOffset(OutOffset + Unit.getStartOffset());
  }

  // Extract and clone every attribute.
  DWARFDataExtractor Data = U.getDebugInfoExtractor();
  // Point to the next DIE (generally there is always at least a NULL
  // entry after the current one). If this is a lone
  // DW_TAG_compile_unit without any children, point to the next unit.
  uint32_t NextOffset = (Idx + 1 < U.getNumDIEs())
                            ? U.getDIEAtIndex(Idx + 1).getOffset()
                            : U.getNextUnitOffset();
  AttributesInfo AttrInfo;

  // We could copy the data only if we need to apply a relocation to it. After
  // testing, it seems there is no performance downside to doing the copy
  // unconditionally, and it makes the code simpler.
  SmallString<40> DIECopy(Data.getData().substr(Offset, NextOffset - Offset));
  Data =
      DWARFDataExtractor(DIECopy, Data.isLittleEndian(), Data.getAddressSize());
  // Modify the copy with relocated addresses.
  if (RelocMgr.applyValidRelocs(DIECopy, Offset, Data.isLittleEndian())) {
    // If we applied relocations, we store the value of high_pc that was
    // potentially stored in the input DIE. If high_pc is an address
    // (Dwarf version == 2), then it might have been relocated to a
    // totally unrelated value (because the end address in the object
    // file might be start address of another function which got moved
    // independently by the linker). The computation of the actual
    // high_pc value is done in cloneAddressAttribute().
    AttrInfo.OrigHighPc =
        dwarf::toAddress(InputDIE.find(dwarf::DW_AT_high_pc), 0);
    // Also store the low_pc. It might get relocated in an
    // inline_subprogram that happens at the beginning of its
    // inlining function.
    AttrInfo.OrigLowPc = dwarf::toAddress(InputDIE.find(dwarf::DW_AT_low_pc),
                                          std::numeric_limits<uint64_t>::max());
  }

  // Reset the Offset to 0 as we will be working on the local copy of
  // the data.
  Offset = 0;

  const auto *Abbrev = InputDIE.getAbbreviationDeclarationPtr();
  Offset += getULEB128Size(Abbrev->getCode());

  // We are entering a subprogram. Get and propagate the PCOffset.
  if (Die->getTag() == dwarf::DW_TAG_subprogram)
    PCOffset = Info.AddrAdjust;
  AttrInfo.PCOffset = PCOffset;

  if (Abbrev->getTag() == dwarf::DW_TAG_subprogram) {
    Flags |= TF_InFunctionScope;
    if (!Info.InDebugMap && LLVM_LIKELY(!Options.Update))
      Flags |= TF_SkipPC;
  }

  bool Copied = false;
  for (const auto &AttrSpec : Abbrev->attributes()) {
    if (LLVM_LIKELY(!Options.Update) &&
        shouldSkipAttribute(AttrSpec, Die->getTag(), Info.InDebugMap,
                            Flags & TF_SkipPC, Flags & TF_InFunctionScope)) {
      DWARFFormValue::skipValue(AttrSpec.Form, Data, &Offset,
                                U.getFormParams());
      // FIXME: dsymutil-classic keeps the old abbreviation around
      // even if it's not used. We can remove this (and the copyAbbrev
      // helper) as soon as bit-for-bit compatibility is not a goal anymore.
      if (!Copied) {
        copyAbbrev(*InputDIE.getAbbreviationDeclarationPtr(), Unit.hasODR());
        Copied = true;
      }
      continue;
    }

    DWARFFormValue Val(AttrSpec.Form);
    uint32_t AttrSize = Offset;
    Val.extractValue(Data, &Offset, U.getFormParams(), &U);
    AttrSize = Offset - AttrSize;

    OutOffset += cloneAttribute(*Die, InputDIE, DMO, Unit, StringPool, Val,
                                AttrSpec, AttrSize, AttrInfo, IsLittleEndian);
  }

  // Look for accelerator entries.
  uint16_t Tag = InputDIE.getTag();
  // FIXME: This is slightly wrong. An inline_subroutine without a
  // low_pc, but with AT_ranges might be interesting to get into the
  // accelerator tables too. For now stick with dsymutil's behavior.
  if ((Info.InDebugMap || AttrInfo.HasLowPc || AttrInfo.HasRanges) &&
      Tag != dwarf::DW_TAG_compile_unit &&
      getDIENames(InputDIE, AttrInfo, StringPool,
                  Tag != dwarf::DW_TAG_inlined_subroutine)) {
    if (AttrInfo.MangledName && AttrInfo.MangledName != AttrInfo.Name)
      Unit.addNameAccelerator(Die, AttrInfo.MangledName,
                              Tag == dwarf::DW_TAG_inlined_subroutine);
    if (AttrInfo.Name) {
      if (AttrInfo.NameWithoutTemplate)
        Unit.addNameAccelerator(Die, AttrInfo.NameWithoutTemplate,
                                /* SkipPubSection */ true);
      Unit.addNameAccelerator(Die, AttrInfo.Name,
                              Tag == dwarf::DW_TAG_inlined_subroutine);
    }
    if (AttrInfo.Name && isObjCSelector(AttrInfo.Name.getString()))
      addObjCAccelerator(Unit, Die, AttrInfo.Name, StringPool,
                         /* SkipPubSection =*/true);

  } else if (Tag == dwarf::DW_TAG_namespace) {
    if (!AttrInfo.Name)
      AttrInfo.Name = StringPool.getEntry("(anonymous namespace)");
    Unit.addNamespaceAccelerator(Die, AttrInfo.Name);
  } else if (isTypeTag(Tag) && !AttrInfo.IsDeclaration &&
             getDIENames(InputDIE, AttrInfo, StringPool) && AttrInfo.Name &&
             AttrInfo.Name.getString()[0]) {
    uint32_t Hash = hashFullyQualifiedName(InputDIE, Unit, DMO);
    uint64_t RuntimeLang =
        dwarf::toUnsigned(InputDIE.find(dwarf::DW_AT_APPLE_runtime_class))
            .getValueOr(0);
    bool ObjCClassIsImplementation =
        (RuntimeLang == dwarf::DW_LANG_ObjC ||
         RuntimeLang == dwarf::DW_LANG_ObjC_plus_plus) &&
        dwarf::toUnsigned(InputDIE.find(dwarf::DW_AT_APPLE_objc_complete_type))
            .getValueOr(0);
    Unit.addTypeAccelerator(Die, AttrInfo.Name, ObjCClassIsImplementation,
                            Hash);
  }

  // Determine whether there are any children that we want to keep.
  bool HasChildren = false;
  for (auto Child : InputDIE.children()) {
    unsigned Idx = U.getDIEIndex(Child);
    if (Unit.getInfo(Idx).Keep) {
      HasChildren = true;
      break;
    }
  }

  DIEAbbrev NewAbbrev = Die->generateAbbrev();
  if (HasChildren)
    NewAbbrev.setChildrenFlag(dwarf::DW_CHILDREN_yes);
  // Assign a permanent abbrev number
  Linker.AssignAbbrev(NewAbbrev);
  Die->setAbbrevNumber(NewAbbrev.getNumber());

  // Add the size of the abbreviation number to the output offset.
  OutOffset += getULEB128Size(Die->getAbbrevNumber());

  if (!HasChildren) {
    // Update our size.
    Die->setSize(OutOffset - Die->getOffset());
    return Die;
  }

  // Recursively clone children.
  for (auto Child : InputDIE.children()) {
    if (DIE *Clone = cloneDIE(Child, DMO, Unit, StringPool, PCOffset, OutOffset,
                              Flags, IsLittleEndian)) {
      Die->addChild(Clone);
      OutOffset = Clone->getOffset() + Clone->getSize();
    }
  }

  // Account for the end of children marker.
  OutOffset += sizeof(int8_t);
  // Update our size.
  Die->setSize(OutOffset - Die->getOffset());
  return Die;
}

/// Patch the input object file relevant debug_ranges entries
/// and emit them in the output file. Update the relevant attributes
/// to point at the new entries.
void DwarfLinker::patchRangesForUnit(const CompileUnit &Unit,
                                     DWARFContext &OrigDwarf,
                                     const DebugMapObject &DMO) const {
  DWARFDebugRangeList RangeList;
  const auto &FunctionRanges = Unit.getFunctionRanges();
  unsigned AddressSize = Unit.getOrigUnit().getAddressByteSize();
  DWARFDataExtractor RangeExtractor(OrigDwarf.getDWARFObj(),
                                    OrigDwarf.getDWARFObj().getRangeSection(),
                                    OrigDwarf.isLittleEndian(), AddressSize);
  auto InvalidRange = FunctionRanges.end(), CurrRange = InvalidRange;
  DWARFUnit &OrigUnit = Unit.getOrigUnit();
  auto OrigUnitDie = OrigUnit.getUnitDIE(false);
  uint64_t OrigLowPc =
      dwarf::toAddress(OrigUnitDie.find(dwarf::DW_AT_low_pc), -1ULL);
  // Ranges addresses are based on the unit's low_pc. Compute the
  // offset we need to apply to adapt to the new unit's low_pc.
  int64_t UnitPcOffset = 0;
  if (OrigLowPc != -1ULL)
    UnitPcOffset = int64_t(OrigLowPc) - Unit.getLowPc();

  for (const auto &RangeAttribute : Unit.getRangesAttributes()) {
    uint32_t Offset = RangeAttribute.get();
    RangeAttribute.set(Streamer->getRangesSectionSize());
    if (Error E = RangeList.extract(RangeExtractor, &Offset)) {
      llvm::consumeError(std::move(E));
      reportWarning("invalid range list ignored.", DMO);
      RangeList.clear();
    }
    const auto &Entries = RangeList.getEntries();
    if (!Entries.empty()) {
      const DWARFDebugRangeList::RangeListEntry &First = Entries.front();

      if (CurrRange == InvalidRange ||
          First.StartAddress + OrigLowPc < CurrRange.start() ||
          First.StartAddress + OrigLowPc >= CurrRange.stop()) {
        CurrRange = FunctionRanges.find(First.StartAddress + OrigLowPc);
        if (CurrRange == InvalidRange ||
            CurrRange.start() > First.StartAddress + OrigLowPc) {
          reportWarning("no mapping for range.", DMO);
          continue;
        }
      }
    }

    Streamer->emitRangesEntries(UnitPcOffset, OrigLowPc, CurrRange, Entries,
                                AddressSize);
  }
}

/// Generate the debug_aranges entries for \p Unit and if the
/// unit has a DW_AT_ranges attribute, also emit the debug_ranges
/// contribution for this attribute.
/// FIXME: this could actually be done right in patchRangesForUnit,
/// but for the sake of initial bit-for-bit compatibility with legacy
/// dsymutil, we have to do it in a delayed pass.
void DwarfLinker::generateUnitRanges(CompileUnit &Unit) const {
  auto Attr = Unit.getUnitRangesAttribute();
  if (Attr)
    Attr->set(Streamer->getRangesSectionSize());
  Streamer->emitUnitRangesEntries(Unit, static_cast<bool>(Attr));
}

/// Insert the new line info sequence \p Seq into the current
/// set of already linked line info \p Rows.
static void insertLineSequence(std::vector<DWARFDebugLine::Row> &Seq,
                               std::vector<DWARFDebugLine::Row> &Rows) {
  if (Seq.empty())
    return;

  if (!Rows.empty() && Rows.back().Address < Seq.front().Address) {
    Rows.insert(Rows.end(), Seq.begin(), Seq.end());
    Seq.clear();
    return;
  }

  auto InsertPoint = std::lower_bound(
      Rows.begin(), Rows.end(), Seq.front(),
      [](const DWARFDebugLine::Row &LHS, const DWARFDebugLine::Row &RHS) {
        return LHS.Address < RHS.Address;
      });

  // FIXME: this only removes the unneeded end_sequence if the
  // sequences have been inserted in order. Using a global sort like
  // described in patchLineTableForUnit() and delaying the end_sequene
  // elimination to emitLineTableForUnit() we can get rid of all of them.
  if (InsertPoint != Rows.end() &&
      InsertPoint->Address == Seq.front().Address && InsertPoint->EndSequence) {
    *InsertPoint = Seq.front();
    Rows.insert(InsertPoint + 1, Seq.begin() + 1, Seq.end());
  } else {
    Rows.insert(InsertPoint, Seq.begin(), Seq.end());
  }

  Seq.clear();
}

static void patchStmtList(DIE &Die, DIEInteger Offset) {
  for (auto &V : Die.values())
    if (V.getAttribute() == dwarf::DW_AT_stmt_list) {
      V = DIEValue(V.getAttribute(), V.getForm(), Offset);
      return;
    }

  llvm_unreachable("Didn't find DW_AT_stmt_list in cloned DIE!");
}

/// Extract the line table for \p Unit from \p OrigDwarf, and
/// recreate a relocated version of these for the address ranges that
/// are present in the binary.
void DwarfLinker::patchLineTableForUnit(CompileUnit &Unit,
                                        DWARFContext &OrigDwarf,
                                        RangesTy &Ranges,
                                        const DebugMapObject &DMO) {
  DWARFDie CUDie = Unit.getOrigUnit().getUnitDIE();
  auto StmtList = dwarf::toSectionOffset(CUDie.find(dwarf::DW_AT_stmt_list));
  if (!StmtList)
    return;

  // Update the cloned DW_AT_stmt_list with the correct debug_line offset.
  if (auto *OutputDIE = Unit.getOutputUnitDIE())
    patchStmtList(*OutputDIE, DIEInteger(Streamer->getLineSectionSize()));

  // Parse the original line info for the unit.
  DWARFDebugLine::LineTable LineTable;
  uint32_t StmtOffset = *StmtList;
  DWARFDataExtractor LineExtractor(
      OrigDwarf.getDWARFObj(), OrigDwarf.getDWARFObj().getLineSection(),
      OrigDwarf.isLittleEndian(), Unit.getOrigUnit().getAddressByteSize());
  if (Options.Translator)
    return Streamer->translateLineTable(LineExtractor, StmtOffset, Options);

  Error Err = LineTable.parse(LineExtractor, &StmtOffset, OrigDwarf,
                              &Unit.getOrigUnit(), DWARFContext::dumpWarning);
  DWARFContext::dumpWarning(std::move(Err));

  // This vector is the output line table.
  std::vector<DWARFDebugLine::Row> NewRows;
  NewRows.reserve(LineTable.Rows.size());

  // Current sequence of rows being extracted, before being inserted
  // in NewRows.
  std::vector<DWARFDebugLine::Row> Seq;
  const auto &FunctionRanges = Unit.getFunctionRanges();
  auto InvalidRange = FunctionRanges.end(), CurrRange = InvalidRange;

  // FIXME: This logic is meant to generate exactly the same output as
  // Darwin's classic dsymutil. There is a nicer way to implement this
  // by simply putting all the relocated line info in NewRows and simply
  // sorting NewRows before passing it to emitLineTableForUnit. This
  // should be correct as sequences for a function should stay
  // together in the sorted output. There are a few corner cases that
  // look suspicious though, and that required to implement the logic
  // this way. Revisit that once initial validation is finished.

  // Iterate over the object file line info and extract the sequences
  // that correspond to linked functions.
  for (auto &Row : LineTable.Rows) {
    // Check whether we stepped out of the range. The range is
    // half-open, but consider accept the end address of the range if
    // it is marked as end_sequence in the input (because in that
    // case, the relocation offset is accurate and that entry won't
    // serve as the start of another function).
    if (CurrRange == InvalidRange || Row.Address.Address < CurrRange.start() ||
        Row.Address.Address > CurrRange.stop() ||
        (Row.Address.Address == CurrRange.stop() && !Row.EndSequence)) {
      // We just stepped out of a known range. Insert a end_sequence
      // corresponding to the end of the range.
      uint64_t StopAddress = CurrRange != InvalidRange
                                 ? CurrRange.stop() + CurrRange.value()
                                 : -1ULL;
      CurrRange = FunctionRanges.find(Row.Address.Address);
      bool CurrRangeValid =
          CurrRange != InvalidRange && CurrRange.start() <= Row.Address.Address;
      if (!CurrRangeValid) {
        CurrRange = InvalidRange;
        if (StopAddress != -1ULL) {
          // Try harder by looking in the DebugMapObject function
          // ranges map. There are corner cases where this finds a
          // valid entry. It's unclear if this is right or wrong, but
          // for now do as dsymutil.
          // FIXME: Understand exactly what cases this addresses and
          // potentially remove it along with the Ranges map.
          auto Range = Ranges.lower_bound(Row.Address.Address);
          if (Range != Ranges.begin() && Range != Ranges.end())
            --Range;

          if (Range != Ranges.end() && Range->first <= Row.Address.Address &&
              Range->second.HighPC >= Row.Address.Address) {
            StopAddress = Row.Address.Address + Range->second.Offset;
          }
        }
      }
      if (StopAddress != -1ULL && !Seq.empty()) {
        // Insert end sequence row with the computed end address, but
        // the same line as the previous one.
        auto NextLine = Seq.back();
        NextLine.Address.Address = StopAddress;
        NextLine.EndSequence = 1;
        NextLine.PrologueEnd = 0;
        NextLine.BasicBlock = 0;
        NextLine.EpilogueBegin = 0;
        Seq.push_back(NextLine);
        insertLineSequence(Seq, NewRows);
      }

      if (!CurrRangeValid)
        continue;
    }

    // Ignore empty sequences.
    if (Row.EndSequence && Seq.empty())
      continue;

    // Relocate row address and add it to the current sequence.
    Row.Address.Address += CurrRange.value();
    Seq.emplace_back(Row);

    if (Row.EndSequence)
      insertLineSequence(Seq, NewRows);
  }

  // Finished extracting, now emit the line tables.
  // FIXME: LLVM hard-codes its prologue values. We just copy the
  // prologue over and that works because we act as both producer and
  // consumer. It would be nicer to have a real configurable line
  // table emitter.
  if (LineTable.Prologue.getVersion() < 2 ||
      LineTable.Prologue.getVersion() > 5 ||
      LineTable.Prologue.DefaultIsStmt != DWARF2_LINE_DEFAULT_IS_STMT ||
      LineTable.Prologue.OpcodeBase > 13)
    reportWarning("line table parameters mismatch. Cannot emit.", DMO);
  else {
    uint32_t PrologueEnd = *StmtList + 10 + LineTable.Prologue.PrologueLength;
    // DWARF v5 has an extra 2 bytes of information before the header_length
    // field.
    if (LineTable.Prologue.getVersion() == 5)
      PrologueEnd += 2;
    StringRef LineData = OrigDwarf.getDWARFObj().getLineSection().Data;
    MCDwarfLineTableParams Params;
    Params.DWARF2LineOpcodeBase = LineTable.Prologue.OpcodeBase;
    Params.DWARF2LineBase = LineTable.Prologue.LineBase;
    Params.DWARF2LineRange = LineTable.Prologue.LineRange;
    Streamer->emitLineTableForUnit(Params,
                                   LineData.slice(*StmtList + 4, PrologueEnd),
                                   LineTable.Prologue.MinInstLength, NewRows,
                                   Unit.getOrigUnit().getAddressByteSize());
  }
}

void DwarfLinker::emitAcceleratorEntriesForUnit(CompileUnit &Unit) {
  switch (Options.TheAccelTableKind) {
  case AccelTableKind::Apple:
    emitAppleAcceleratorEntriesForUnit(Unit);
    break;
  case AccelTableKind::Dwarf:
    emitDwarfAcceleratorEntriesForUnit(Unit);
    break;
  case AccelTableKind::Default:
    llvm_unreachable("The default must be updated to a concrete value.");
    break;
  }
}

void DwarfLinker::emitAppleAcceleratorEntriesForUnit(CompileUnit &Unit) {
  // Add namespaces.
  for (const auto &Namespace : Unit.getNamespaces())
    AppleNamespaces.addName(Namespace.Name,
                            Namespace.Die->getOffset() + Unit.getStartOffset());

  /// Add names.
  if (!Options.Minimize)
    Streamer->emitPubNamesForUnit(Unit);
  for (const auto &Pubname : Unit.getPubnames())
    AppleNames.addName(Pubname.Name,
                       Pubname.Die->getOffset() + Unit.getStartOffset());

  /// Add types.
  if (!Options.Minimize)
    Streamer->emitPubTypesForUnit(Unit);
  for (const auto &Pubtype : Unit.getPubtypes())
    AppleTypes.addName(
        Pubtype.Name, Pubtype.Die->getOffset() + Unit.getStartOffset(),
        Pubtype.Die->getTag(),
        Pubtype.ObjcClassImplementation ? dwarf::DW_FLAG_type_implementation
                                        : 0,
        Pubtype.QualifiedNameHash);

  /// Add ObjC names.
  for (const auto &ObjC : Unit.getObjC())
    AppleObjc.addName(ObjC.Name, ObjC.Die->getOffset() + Unit.getStartOffset());
}

void DwarfLinker::emitDwarfAcceleratorEntriesForUnit(CompileUnit &Unit) {
  for (const auto &Namespace : Unit.getNamespaces())
    DebugNames.addName(Namespace.Name, Namespace.Die->getOffset(),
                       Namespace.Die->getTag(), Unit.getUniqueID());
  for (const auto &Pubname : Unit.getPubnames())
    DebugNames.addName(Pubname.Name, Pubname.Die->getOffset(),
                       Pubname.Die->getTag(), Unit.getUniqueID());
  for (const auto &Pubtype : Unit.getPubtypes())
    DebugNames.addName(Pubtype.Name, Pubtype.Die->getOffset(),
                       Pubtype.Die->getTag(), Unit.getUniqueID());
}

/// Read the frame info stored in the object, and emit the
/// patched frame descriptions for the linked binary.
///
/// This is actually pretty easy as the data of the CIEs and FDEs can
/// be considered as black boxes and moved as is. The only thing to do
/// is to patch the addresses in the headers.
void DwarfLinker::patchFrameInfoForObject(const DebugMapObject &DMO,
                                          RangesTy &Ranges,
                                          DWARFContext &OrigDwarf,
                                          unsigned AddrSize) {
  StringRef FrameData = OrigDwarf.getDWARFObj().getDebugFrameSection();
  if (FrameData.empty())
    return;

  DataExtractor Data(FrameData, OrigDwarf.isLittleEndian(), 0);
  uint32_t InputOffset = 0;

  // Store the data of the CIEs defined in this object, keyed by their
  // offsets.
  DenseMap<uint32_t, StringRef> LocalCIES;

  while (Data.isValidOffset(InputOffset)) {
    uint32_t EntryOffset = InputOffset;
    uint32_t InitialLength = Data.getU32(&InputOffset);
    if (InitialLength == 0xFFFFFFFF)
      return reportWarning("Dwarf64 bits no supported", DMO);

    uint32_t CIEId = Data.getU32(&InputOffset);
    if (CIEId == 0xFFFFFFFF) {
      // This is a CIE, store it.
      StringRef CIEData = FrameData.substr(EntryOffset, InitialLength + 4);
      LocalCIES[EntryOffset] = CIEData;
      // The -4 is to account for the CIEId we just read.
      InputOffset += InitialLength - 4;
      continue;
    }

    uint32_t Loc = Data.getUnsigned(&InputOffset, AddrSize);

    // Some compilers seem to emit frame info that doesn't start at
    // the function entry point, thus we can't just lookup the address
    // in the debug map. Use the linker's range map to see if the FDE
    // describes something that we can relocate.
    auto Range = Ranges.upper_bound(Loc);
    if (Range != Ranges.begin())
      --Range;
    if (Range == Ranges.end() || Range->first > Loc ||
        Range->second.HighPC <= Loc) {
      // The +4 is to account for the size of the InitialLength field itself.
      InputOffset = EntryOffset + InitialLength + 4;
      continue;
    }

    // This is an FDE, and we have a mapping.
    // Have we already emitted a corresponding CIE?
    StringRef CIEData = LocalCIES[CIEId];
    if (CIEData.empty())
      return reportWarning("Inconsistent debug_frame content. Dropping.", DMO);

    // Look if we already emitted a CIE that corresponds to the
    // referenced one (the CIE data is the key of that lookup).
    auto IteratorInserted = EmittedCIEs.insert(
        std::make_pair(CIEData, Streamer->getFrameSectionSize()));
    // If there is no CIE yet for this ID, emit it.
    if (IteratorInserted.second ||
        // FIXME: dsymutil-classic only caches the last used CIE for
        // reuse. Mimic that behavior for now. Just removing that
        // second half of the condition and the LastCIEOffset variable
        // makes the code DTRT.
        LastCIEOffset != IteratorInserted.first->getValue()) {
      LastCIEOffset = Streamer->getFrameSectionSize();
      IteratorInserted.first->getValue() = LastCIEOffset;
      Streamer->emitCIE(CIEData);
    }

    // Emit the FDE with updated address and CIE pointer.
    // (4 + AddrSize) is the size of the CIEId + initial_location
    // fields that will get reconstructed by emitFDE().
    unsigned FDERemainingBytes = InitialLength - (4 + AddrSize);
    Streamer->emitFDE(IteratorInserted.first->getValue(), AddrSize,
                      Loc + Range->second.Offset,
                      FrameData.substr(InputOffset, FDERemainingBytes));
    InputOffset += FDERemainingBytes;
  }
}

void DwarfLinker::DIECloner::copyAbbrev(
    const DWARFAbbreviationDeclaration &Abbrev, bool hasODR) {
  DIEAbbrev Copy(dwarf::Tag(Abbrev.getTag()),
                 dwarf::Form(Abbrev.hasChildren()));

  for (const auto &Attr : Abbrev.attributes()) {
    uint16_t Form = Attr.Form;
    if (hasODR && isODRAttribute(Attr.Attr))
      Form = dwarf::DW_FORM_ref_addr;
    Copy.AddAttribute(dwarf::Attribute(Attr.Attr), dwarf::Form(Form));
  }

  Linker.AssignAbbrev(Copy);
}

uint32_t DwarfLinker::DIECloner::hashFullyQualifiedName(
    DWARFDie DIE, CompileUnit &U, const DebugMapObject &DMO, int RecurseDepth) {
  const char *Name = nullptr;
  DWARFUnit *OrigUnit = &U.getOrigUnit();
  CompileUnit *CU = &U;
  Optional<DWARFFormValue> Ref;

  while (1) {
    if (const char *CurrentName = DIE.getName(DINameKind::ShortName))
      Name = CurrentName;

    if (!(Ref = DIE.find(dwarf::DW_AT_specification)) &&
        !(Ref = DIE.find(dwarf::DW_AT_abstract_origin)))
      break;

    if (!Ref->isFormClass(DWARFFormValue::FC_Reference))
      break;

    CompileUnit *RefCU;
    if (auto RefDIE = resolveDIEReference(Linker, DMO, CompileUnits, *Ref,
                                          U.getOrigUnit(), DIE, RefCU)) {
      CU = RefCU;
      OrigUnit = &RefCU->getOrigUnit();
      DIE = RefDIE;
    }
  }

  unsigned Idx = OrigUnit->getDIEIndex(DIE);
  if (!Name && DIE.getTag() == dwarf::DW_TAG_namespace)
    Name = "(anonymous namespace)";

  if (CU->getInfo(Idx).ParentIdx == 0 ||
      // FIXME: dsymutil-classic compatibility. Ignore modules.
      CU->getOrigUnit().getDIEAtIndex(CU->getInfo(Idx).ParentIdx).getTag() ==
          dwarf::DW_TAG_module)
    return djbHash(Name ? Name : "", djbHash(RecurseDepth ? "" : "::"));

  DWARFDie Die = OrigUnit->getDIEAtIndex(CU->getInfo(Idx).ParentIdx);
  return djbHash(
      (Name ? Name : ""),
      djbHash((Name ? "::" : ""),
              hashFullyQualifiedName(Die, *CU, DMO, ++RecurseDepth)));
}

static uint64_t getDwoId(const DWARFDie &CUDie, const DWARFUnit &Unit) {
  auto DwoId = dwarf::toUnsigned(
      CUDie.find({dwarf::DW_AT_dwo_id, dwarf::DW_AT_GNU_dwo_id}));
  if (DwoId)
    return *DwoId;
  return 0;
}

bool DwarfLinker::registerModuleReference(
    const DWARFDie &CUDie, const DWARFUnit &Unit, DebugMap &ModuleMap,
    const DebugMapObject &DMO, RangesTy &Ranges, OffsetsStringPool &StringPool,
    UniquingStringPool &UniquingStringPool, DeclContextTree &ODRContexts,
    uint64_t ModulesEndOffset, unsigned &UnitID, bool IsLittleEndian,
    unsigned Indent, bool Quiet) {
  std::string PCMfile = dwarf::toString(
      CUDie.find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}), "");
  if (PCMfile.empty())
    return false;

  // Clang module DWARF skeleton CUs abuse this for the path to the module.
  std::string PCMpath = dwarf::toString(CUDie.find(dwarf::DW_AT_comp_dir), "");
  uint64_t DwoId = getDwoId(CUDie, Unit);

  std::string Name = dwarf::toString(CUDie.find(dwarf::DW_AT_name), "");
  if (Name.empty()) {
    if (!Quiet)
      reportWarning("Anonymous module skeleton CU for " + PCMfile, DMO);
    return true;
  }

  if (!Quiet && Options.Verbose) {
    outs().indent(Indent);
    outs() << "Found clang module reference " << PCMfile;
  }

  auto Cached = ClangModules.find(PCMfile);
  if (Cached != ClangModules.end()) {
    // FIXME: Until PR27449 (https://llvm.org/bugs/show_bug.cgi?id=27449) is
    // fixed in clang, only warn about DWO_id mismatches in verbose mode.
    // ASTFileSignatures will change randomly when a module is rebuilt.
    if (!Quiet && Options.Verbose && (Cached->second != DwoId))
      reportWarning(Twine("hash mismatch: this object file was built against a "
                          "different version of the module ") +
                        PCMfile,
                    DMO);
    if (!Quiet && Options.Verbose)
      outs() << " [cached].\n";
    return true;
  }
  if (!Quiet && Options.Verbose)
    outs() << " ...\n";

  // Cyclic dependencies are disallowed by Clang, but we still
  // shouldn't run into an infinite loop, so mark it as processed now.
  ClangModules.insert({PCMfile, DwoId});
  if (Error E = loadClangModule(PCMfile, PCMpath, Name, DwoId, ModuleMap, DMO,
                                Ranges, StringPool, UniquingStringPool,
                                ODRContexts, ModulesEndOffset, UnitID,
                                IsLittleEndian, Indent + 2, Quiet)) {
    consumeError(std::move(E));
    return false;
  }
  return true;
}

ErrorOr<const object::ObjectFile &>
DwarfLinker::loadObject(const DebugMapObject &Obj, const DebugMap &Map) {
  auto ObjectEntry =
      BinHolder.getObjectEntry(Obj.getObjectFilename(), Obj.getTimestamp());
  if (!ObjectEntry) {
    auto Err = ObjectEntry.takeError();
    reportWarning(
        Twine(Obj.getObjectFilename()) + ": " + toString(std::move(Err)), Obj);
    return errorToErrorCode(std::move(Err));
  }

  auto Object = ObjectEntry->getObject(Map.getTriple());
  if (!Object) {
    auto Err = Object.takeError();
    reportWarning(
        Twine(Obj.getObjectFilename()) + ": " + toString(std::move(Err)), Obj);
    return errorToErrorCode(std::move(Err));
  }

  return *Object;
}

Error DwarfLinker::loadClangModule(
    StringRef Filename, StringRef ModulePath, StringRef ModuleName,
    uint64_t DwoId, DebugMap &ModuleMap, const DebugMapObject &DMO,
    RangesTy &Ranges, OffsetsStringPool &StringPool,
    UniquingStringPool &UniquingStringPool, DeclContextTree &ODRContexts,
    uint64_t ModulesEndOffset, unsigned &UnitID, bool IsLittleEndian,
    unsigned Indent, bool Quiet) {
  SmallString<80> Path(Options.PrependPath);
  if (sys::path::is_relative(Filename))
    sys::path::append(Path, ModulePath, Filename);
  else
    sys::path::append(Path, Filename);
  // Don't use the cached binary holder because we have no thread-safety
  // guarantee and the lifetime is limited.
  auto &Obj = ModuleMap.addDebugMapObject(
      Path, sys::TimePoint<std::chrono::seconds>(), MachO::N_OSO);
  auto ErrOrObj = loadObject(Obj, ModuleMap);
  if (!ErrOrObj) {
    // Try and emit more helpful warnings by applying some heuristics.
    StringRef ObjFile = DMO.getObjectFilename();
    bool isClangModule = sys::path::extension(Filename).equals(".pcm");
    bool isArchive = ObjFile.endswith(")");
    if (isClangModule) {
      StringRef ModuleCacheDir = sys::path::parent_path(Path);
      if (sys::fs::exists(ModuleCacheDir)) {
        // If the module's parent directory exists, we assume that the module
        // cache has expired and was pruned by clang.  A more adventurous
        // dsymutil would invoke clang to rebuild the module now.
        if (!ModuleCacheHintDisplayed) {
          WithColor::note() << "The clang module cache may have expired since "
                               "this object file was built. Rebuilding the "
                               "object file will rebuild the module cache.\n";
          ModuleCacheHintDisplayed = true;
        }
      } else if (isArchive) {
        // If the module cache directory doesn't exist at all and the object
        // file is inside a static library, we assume that the static library
        // was built on a different machine. We don't want to discourage module
        // debugging for convenience libraries within a project though.
        if (!ArchiveHintDisplayed) {
          WithColor::note()
              << "Linking a static library that was built with "
                 "-gmodules, but the module cache was not found.  "
                 "Redistributable static libraries should never be "
                 "built with module debugging enabled.  The debug "
                 "experience will be degraded due to incomplete "
                 "debug information.\n";
          ArchiveHintDisplayed = true;
        }
      }
    }
    return Error::success();
  }

  std::unique_ptr<CompileUnit> Unit;

  // Setup access to the debug info.
  auto DwarfContext = DWARFContext::create(*ErrOrObj);
  RelocationManager RelocMgr(*this);

  for (const auto &CU : DwarfContext->compile_units()) {
    updateDwarfVersion(CU->getVersion());
    // Recursively get all modules imported by this one.
    auto CUDie = CU->getUnitDIE(false);
    if (!CUDie)
      continue;
    if (!registerModuleReference(CUDie, *CU, ModuleMap, DMO, Ranges, StringPool,
                                 UniquingStringPool, ODRContexts,
                                 ModulesEndOffset, UnitID, IsLittleEndian,
                                 Indent, Quiet)) {
      if (Unit) {
        std::string Err =
            (Filename +
             ": Clang modules are expected to have exactly 1 compile unit.\n")
                .str();
        error(Err);
        return make_error<StringError>(Err, inconvertibleErrorCode());
      }
      // FIXME: Until PR27449 (https://llvm.org/bugs/show_bug.cgi?id=27449) is
      // fixed in clang, only warn about DWO_id mismatches in verbose mode.
      // ASTFileSignatures will change randomly when a module is rebuilt.
      uint64_t PCMDwoId = getDwoId(CUDie, *CU);
      if (PCMDwoId != DwoId) {
        if (!Quiet && Options.Verbose)
          reportWarning(
              Twine("hash mismatch: this object file was built against a "
                    "different version of the module ") +
                  Filename,
              DMO);
        // Update the cache entry with the DwoId of the module loaded from disk.
        ClangModules[Filename] = PCMDwoId;
      }

      // Add this module.
      Unit = llvm::make_unique<CompileUnit>(*CU, UnitID++, !Options.NoODR,
                                            ModuleName);
      Unit->setHasInterestingContent();
      analyzeContextInfo(CUDie, 0, *Unit, &ODRContexts.getRoot(),
                         UniquingStringPool, ODRContexts, ModulesEndOffset);
      // Keep everything.
      Unit->markEverythingAsKept();
    }
  }
  if (!Unit->getOrigUnit().getUnitDIE().hasChildren())
    return Error::success();
  if (!Quiet && Options.Verbose) {
    outs().indent(Indent);
    outs() << "cloning .debug_info from " << Filename << "\n";
  }

  UnitListTy CompileUnits;
  CompileUnits.push_back(std::move(Unit));
  DIECloner(*this, RelocMgr, DIEAlloc, CompileUnits, Options)
      .cloneAllCompileUnits(*DwarfContext, DMO, Ranges, StringPool,
                            IsLittleEndian);
  return Error::success();
}

void DwarfLinker::DIECloner::cloneAllCompileUnits(
    DWARFContext &DwarfContext, const DebugMapObject &DMO, RangesTy &Ranges,
    OffsetsStringPool &StringPool, bool IsLittleEndian) {
  if (!Linker.Streamer)
    return;

  for (auto &CurrentUnit : CompileUnits) {
    auto InputDIE = CurrentUnit->getOrigUnit().getUnitDIE();
    CurrentUnit->setStartOffset(Linker.OutputDebugInfoSize);
    if (!InputDIE) {
      Linker.OutputDebugInfoSize = CurrentUnit->computeNextUnitOffset();
      continue;
    }
    if (CurrentUnit->getInfo(0).Keep) {
      // Clone the InputDIE into your Unit DIE in our compile unit since it
      // already has a DIE inside of it.
      CurrentUnit->createOutputDIE();
      cloneDIE(InputDIE, DMO, *CurrentUnit, StringPool, 0 /* PC offset */,
               11 /* Unit Header size */, 0, IsLittleEndian,
               CurrentUnit->getOutputUnitDIE());
    }

    Linker.OutputDebugInfoSize = CurrentUnit->computeNextUnitOffset();

    if (Linker.Options.NoOutput)
      continue;

    // FIXME: for compatibility with the classic dsymutil, we emit
    // an empty line table for the unit, even if the unit doesn't
    // actually exist in the DIE tree.
    if (LLVM_LIKELY(!Linker.Options.Update) || Linker.Options.Translator)
      Linker.patchLineTableForUnit(*CurrentUnit, DwarfContext, Ranges, DMO);

    Linker.emitAcceleratorEntriesForUnit(*CurrentUnit);

    if (LLVM_UNLIKELY(Linker.Options.Update))
      continue;

    Linker.patchRangesForUnit(*CurrentUnit, DwarfContext, DMO);
    auto ProcessExpr = [&](StringRef Bytes, SmallVectorImpl<uint8_t> &Buffer) {
      DWARFUnit &OrigUnit = CurrentUnit->getOrigUnit();
      DataExtractor Data(Bytes, IsLittleEndian, OrigUnit.getAddressByteSize());
      cloneExpression(Data,
                      DWARFExpression(Data, OrigUnit.getVersion(),
                                      OrigUnit.getAddressByteSize()),
                      DMO, *CurrentUnit, Buffer);
    };
    Linker.Streamer->emitLocationsForUnit(*CurrentUnit, DwarfContext,
                                          ProcessExpr);
  }

  if (Linker.Options.NoOutput)
    return;

  // Emit all the compile unit's debug information.
  for (auto &CurrentUnit : CompileUnits) {
    if (LLVM_LIKELY(!Linker.Options.Update))
      Linker.generateUnitRanges(*CurrentUnit);

    CurrentUnit->fixupForwardReferences();

    if (!CurrentUnit->getOutputUnitDIE())
      continue;

    Linker.Streamer->emitCompileUnitHeader(*CurrentUnit);
    Linker.Streamer->emitDIE(*CurrentUnit->getOutputUnitDIE());
  }
}

void DwarfLinker::updateAccelKind(DWARFContext &Dwarf) {
  if (Options.TheAccelTableKind != AccelTableKind::Default)
    return;

  auto &DwarfObj = Dwarf.getDWARFObj();

  if (!AtLeastOneDwarfAccelTable &&
      (!DwarfObj.getAppleNamesSection().Data.empty() ||
       !DwarfObj.getAppleTypesSection().Data.empty() ||
       !DwarfObj.getAppleNamespacesSection().Data.empty() ||
       !DwarfObj.getAppleObjCSection().Data.empty())) {
    AtLeastOneAppleAccelTable = true;
  }

  if (!AtLeastOneDwarfAccelTable &&
      !DwarfObj.getDebugNamesSection().Data.empty()) {
    AtLeastOneDwarfAccelTable = true;
  }
}

bool DwarfLinker::emitPaperTrailWarnings(const DebugMapObject &DMO,
                                         const DebugMap &Map,
                                         OffsetsStringPool &StringPool) {
  if (DMO.getWarnings().empty() || !DMO.empty())
    return false;

  Streamer->switchToDebugInfoSection(/* Version */ 2);
  DIE *CUDie = DIE::get(DIEAlloc, dwarf::DW_TAG_compile_unit);
  CUDie->setOffset(11);
  StringRef Producer = StringPool.internString("dsymutil");
  StringRef File = StringPool.internString(DMO.getObjectFilename());
  CUDie->addValue(DIEAlloc, dwarf::DW_AT_producer, dwarf::DW_FORM_strp,
                  DIEInteger(StringPool.getStringOffset(Producer)));
  DIEBlock *String = new (DIEAlloc) DIEBlock();
  DIEBlocks.push_back(String);
  for (auto &C : File)
    String->addValue(DIEAlloc, dwarf::Attribute(0), dwarf::DW_FORM_data1,
                     DIEInteger(C));
  String->addValue(DIEAlloc, dwarf::Attribute(0), dwarf::DW_FORM_data1,
                   DIEInteger(0));

  CUDie->addValue(DIEAlloc, dwarf::DW_AT_name, dwarf::DW_FORM_string, String);
  for (const auto &Warning : DMO.getWarnings()) {
    DIE &ConstDie = CUDie->addChild(DIE::get(DIEAlloc, dwarf::DW_TAG_constant));
    ConstDie.addValue(
        DIEAlloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp,
        DIEInteger(StringPool.getStringOffset("dsymutil_warning")));
    ConstDie.addValue(DIEAlloc, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag,
                      DIEInteger(1));
    ConstDie.addValue(DIEAlloc, dwarf::DW_AT_const_value, dwarf::DW_FORM_strp,
                      DIEInteger(StringPool.getStringOffset(Warning)));
  }
  unsigned Size = 4 /* FORM_strp */ + File.size() + 1 +
                  DMO.getWarnings().size() * (4 + 1 + 4) +
                  1 /* End of children */;
  DIEAbbrev Abbrev = CUDie->generateAbbrev();
  AssignAbbrev(Abbrev);
  CUDie->setAbbrevNumber(Abbrev.getNumber());
  Size += getULEB128Size(Abbrev.getNumber());
  // Abbreviation ordering needed for classic compatibility.
  for (auto &Child : CUDie->children()) {
    Abbrev = Child.generateAbbrev();
    AssignAbbrev(Abbrev);
    Child.setAbbrevNumber(Abbrev.getNumber());
    Size += getULEB128Size(Abbrev.getNumber());
  }
  CUDie->setSize(Size);
  auto &Asm = Streamer->getAsmPrinter();
  Asm.emitInt32(11 + CUDie->getSize() - 4);
  Asm.emitInt16(2);
  Asm.emitInt32(0);
  Asm.emitInt8(Map.getTriple().isArch64Bit() ? 8 : 4);
  Streamer->emitDIE(*CUDie);
  OutputDebugInfoSize += 11 /* Header */ + Size;

  return true;
}

bool DwarfLinker::link(const DebugMap &Map) {
  if (!createStreamer(Map.getTriple(), OutFile))
    return false;

  // Size of the DIEs (and headers) generated for the linked output.
  OutputDebugInfoSize = 0;
  // A unique ID that identifies each compile unit.
  unsigned UnitID = 0;
  DebugMap ModuleMap(Map.getTriple(), Map.getBinaryPath());

  // First populate the data structure we need for each iteration of the
  // parallel loop.
  unsigned NumObjects = Map.getNumberOfObjects();
  std::vector<LinkContext> ObjectContexts;
  ObjectContexts.reserve(NumObjects);
  for (const auto &Obj : Map.objects()) {
    ObjectContexts.emplace_back(Map, *this, *Obj.get());
    LinkContext &LC = ObjectContexts.back();
    if (LC.ObjectFile)
      updateAccelKind(*LC.DwarfContext);
  }

  // This Dwarf string pool which is only used for uniquing. This one should
  // never be used for offsets as its not thread-safe or predictable.
  UniquingStringPool UniquingStringPool;

  // This Dwarf string pool which is used for emission. It must be used
  // serially as the order of calling getStringOffset matters for
  // reproducibility.
  OffsetsStringPool OffsetsStringPool(Options.Translator);

  // ODR Contexts for the link.
  DeclContextTree ODRContexts;

  // If we haven't decided on an accelerator table kind yet, we base ourselves
  // on the DWARF we have seen so far. At this point we haven't pulled in debug
  // information from modules yet, so it is technically possible that they
  // would affect the decision. However, as they're built with the same
  // compiler and flags, it is safe to assume that they will follow the
  // decision made here.
  if (Options.TheAccelTableKind == AccelTableKind::Default) {
    if (AtLeastOneDwarfAccelTable && !AtLeastOneAppleAccelTable)
      Options.TheAccelTableKind = AccelTableKind::Dwarf;
    else
      Options.TheAccelTableKind = AccelTableKind::Apple;
  }

  for (LinkContext &LinkContext : ObjectContexts) {
    if (Options.Verbose)
      outs() << "DEBUG MAP OBJECT: " << LinkContext.DMO.getObjectFilename()
             << "\n";

    // N_AST objects (swiftmodule files) should get dumped directly into the
    // appropriate DWARF section.
    if (LinkContext.DMO.getType() == MachO::N_AST) {
      StringRef File = LinkContext.DMO.getObjectFilename();
      auto ErrorOrMem = MemoryBuffer::getFile(File);
      if (!ErrorOrMem) {
        warn("Could not open '" + File + "'\n");
        continue;
      }
      sys::fs::file_status Stat;
      if (auto Err = sys::fs::status(File, Stat)) {
        warn(Err.message());
        continue;
      }
      if (!Options.NoTimestamp) {
        // The modification can have sub-second precision so we need to cast
        // away the extra precision that's not present in the debug map.
        auto ModificationTime =
            std::chrono::time_point_cast<std::chrono::seconds>(
                Stat.getLastModificationTime());
        if (ModificationTime != LinkContext.DMO.getTimestamp()) {
          // Not using the helper here as we can easily stream TimePoint<>.
          WithColor::warning()
              << "Timestamp mismatch for " << File << ": "
              << Stat.getLastModificationTime() << " and "
              << sys::TimePoint<>(LinkContext.DMO.getTimestamp()) << "\n";
          continue;
        }
      }

      // Copy the module into the .swift_ast section.
      if (!Options.NoOutput)
        Streamer->emitSwiftAST((*ErrorOrMem)->getBuffer());
      continue;
    }

    if (emitPaperTrailWarnings(LinkContext.DMO, Map, OffsetsStringPool))
      continue;

    if (!LinkContext.ObjectFile)
      continue;

    // Look for relocations that correspond to debug map entries.

    if (LLVM_LIKELY(!Options.Update) &&
        !LinkContext.RelocMgr.findValidRelocsInDebugInfo(
            *LinkContext.ObjectFile, LinkContext.DMO)) {
      if (Options.Verbose)
        outs() << "No valid relocations found. Skipping.\n";

      // Clear this ObjFile entry as a signal to other loops that we should not
      // process this iteration.
      LinkContext.ObjectFile = nullptr;
      continue;
    }

    // Setup access to the debug info.
    if (!LinkContext.DwarfContext)
      continue;

    startDebugObject(LinkContext);

    // In a first phase, just read in the debug info and load all clang modules.
    LinkContext.CompileUnits.reserve(
        LinkContext.DwarfContext->getNumCompileUnits());

    for (const auto &CU : LinkContext.DwarfContext->compile_units()) {
      updateDwarfVersion(CU->getVersion());
      auto CUDie = CU->getUnitDIE(false);
      if (Options.Verbose) {
        outs() << "Input compilation unit:";
        DIDumpOptions DumpOpts;
        DumpOpts.RecurseDepth = 0;
        DumpOpts.Verbose = Options.Verbose;
        CUDie.dump(outs(), 0, DumpOpts);
      }
      if (CUDie && !LLVM_UNLIKELY(Options.Update))
        registerModuleReference(CUDie, *CU, ModuleMap, LinkContext.DMO,
                                LinkContext.Ranges, OffsetsStringPool,
                                UniquingStringPool, ODRContexts, 0, UnitID,
                                LinkContext.DwarfContext->isLittleEndian());
    }
  }

  // If we haven't seen any CUs, pick an arbitrary valid Dwarf version anyway.
  if (MaxDwarfVersion == 0)
    MaxDwarfVersion = 3;

  // At this point we know how much data we have emitted. We use this value to
  // compare canonical DIE offsets in analyzeContextInfo to see if a definition
  // is already emitted, without being affected by canonical die offsets set
  // later. This prevents undeterminism when analyze and clone execute
  // concurrently, as clone set the canonical DIE offset and analyze reads it.
  const uint64_t ModulesEndOffset = OutputDebugInfoSize;

  // These variables manage the list of processed object files.
  // The mutex and condition variable are to ensure that this is thread safe.
  std::mutex ProcessedFilesMutex;
  std::condition_variable ProcessedFilesConditionVariable;
  BitVector ProcessedFiles(NumObjects, false);

  //  Analyzing the context info is particularly expensive so it is executed in
  //  parallel with emitting the previous compile unit.
  auto AnalyzeLambda = [&](size_t i) {
    auto &LinkContext = ObjectContexts[i];

    if (!LinkContext.ObjectFile || !LinkContext.DwarfContext)
      return;

    for (const auto &CU : LinkContext.DwarfContext->compile_units()) {
      updateDwarfVersion(CU->getVersion());
      // The !registerModuleReference() condition effectively skips
      // over fully resolved skeleton units. This second pass of
      // registerModuleReferences doesn't do any new work, but it
      // will collect top-level errors, which are suppressed. Module
      // warnings were already displayed in the first iteration.
      bool Quiet = true;
      auto CUDie = CU->getUnitDIE(false);
      if (!CUDie || LLVM_UNLIKELY(Options.Update) ||
          !registerModuleReference(CUDie, *CU, ModuleMap, LinkContext.DMO,
                                   LinkContext.Ranges, OffsetsStringPool,
                                   UniquingStringPool, ODRContexts,
                                   ModulesEndOffset, UnitID, Quiet)) {
        LinkContext.CompileUnits.push_back(llvm::make_unique<CompileUnit>(
            *CU, UnitID++, !Options.NoODR && !Options.Update, ""));
      }
    }

    // Now build the DIE parent links that we will use during the next phase.
    for (auto &CurrentUnit : LinkContext.CompileUnits) {
      auto CUDie = CurrentUnit->getOrigUnit().getUnitDIE();
      if (!CUDie)
        continue;
      analyzeContextInfo(CurrentUnit->getOrigUnit().getUnitDIE(), 0,
                         *CurrentUnit, &ODRContexts.getRoot(),
                         UniquingStringPool, ODRContexts, ModulesEndOffset);
    }
  };

  // And then the remaining work in serial again.
  // Note, although this loop runs in serial, it can run in parallel with
  // the analyzeContextInfo loop so long as we process files with indices >=
  // than those processed by analyzeContextInfo.
  auto CloneLambda = [&](size_t i) {
    auto &LinkContext = ObjectContexts[i];
    if (!LinkContext.ObjectFile)
      return;

    // Then mark all the DIEs that need to be present in the linked output
    // and collect some information about them.
    // Note that this loop can not be merged with the previous one because
    // cross-cu references require the ParentIdx to be setup for every CU in
    // the object file before calling this.
    if (LLVM_UNLIKELY(Options.Update)) {
      for (auto &CurrentUnit : LinkContext.CompileUnits)
        CurrentUnit->markEverythingAsKept();
      Streamer->copyInvariantDebugSection(*LinkContext.ObjectFile);
    } else {
      for (auto &CurrentUnit : LinkContext.CompileUnits)
        lookForDIEsToKeep(LinkContext.RelocMgr, LinkContext.Ranges,
                          LinkContext.CompileUnits,
                          CurrentUnit->getOrigUnit().getUnitDIE(),
                          LinkContext.DMO, *CurrentUnit, 0);
    }

    // The calls to applyValidRelocs inside cloneDIE will walk the reloc
    // array again (in the same way findValidRelocsInDebugInfo() did). We
    // need to reset the NextValidReloc index to the beginning.
    LinkContext.RelocMgr.resetValidRelocs();
    if (LinkContext.RelocMgr.hasValidRelocs() || LLVM_UNLIKELY(Options.Update))
      DIECloner(*this, LinkContext.RelocMgr, DIEAlloc, LinkContext.CompileUnits,
                Options)
          .cloneAllCompileUnits(*LinkContext.DwarfContext, LinkContext.DMO,
                                LinkContext.Ranges, OffsetsStringPool,
                                LinkContext.DwarfContext->isLittleEndian());
    if (!Options.NoOutput && !LinkContext.CompileUnits.empty() &&
        LLVM_LIKELY(!Options.Update))
      patchFrameInfoForObject(
          LinkContext.DMO, LinkContext.Ranges, *LinkContext.DwarfContext,
          LinkContext.CompileUnits[0]->getOrigUnit().getAddressByteSize());

    // Clean-up before starting working on the next object.
    endDebugObject(LinkContext);
  };

  auto EmitLambda = [&]() {
    // Emit everything that's global.
    if (!Options.NoOutput) {
      Streamer->emitAbbrevs(Abbreviations, MaxDwarfVersion);
      Streamer->emitStrings(OffsetsStringPool);
      switch (Options.TheAccelTableKind) {
      case AccelTableKind::Apple:
        Streamer->emitAppleNames(AppleNames);
        Streamer->emitAppleNamespaces(AppleNamespaces);
        Streamer->emitAppleTypes(AppleTypes);
        Streamer->emitAppleObjc(AppleObjc);
        break;
      case AccelTableKind::Dwarf:
        Streamer->emitDebugNames(DebugNames);
        break;
      case AccelTableKind::Default:
        llvm_unreachable("Default should have already been resolved.");
        break;
      }
    }
  };

  auto AnalyzeAll = [&]() {
    for (unsigned i = 0, e = NumObjects; i != e; ++i) {
      AnalyzeLambda(i);

      std::unique_lock<std::mutex> LockGuard(ProcessedFilesMutex);
      ProcessedFiles.set(i);
      ProcessedFilesConditionVariable.notify_one();
    }
  };

  auto CloneAll = [&]() {
    for (unsigned i = 0, e = NumObjects; i != e; ++i) {
      {
        std::unique_lock<std::mutex> LockGuard(ProcessedFilesMutex);
        if (!ProcessedFiles[i]) {
          ProcessedFilesConditionVariable.wait(
              LockGuard, [&]() { return ProcessedFiles[i]; });
        }
      }

      CloneLambda(i);
    }
    EmitLambda();
  };

  // To limit memory usage in the single threaded case, analyze and clone are
  // run sequentially so the LinkContext is freed after processing each object
  // in endDebugObject.
  if (Options.Threads == 1) {
    for (unsigned i = 0, e = NumObjects; i != e; ++i) {
      AnalyzeLambda(i);
      CloneLambda(i);
    }
    EmitLambda();
  } else {
    ThreadPool pool(2);
    pool.async(AnalyzeAll);
    pool.async(CloneAll);
    pool.wait();
  }

  return Options.NoOutput ? true : Streamer->finish(Map, Options.Translator);
} // namespace dsymutil

bool linkDwarf(raw_fd_ostream &OutFile, BinaryHolder &BinHolder,
               const DebugMap &DM, const LinkOptions &Options) {
  DwarfLinker Linker(OutFile, BinHolder, Options);
  return Linker.link(DM);
}

} // namespace dsymutil
} // namespace llvm
