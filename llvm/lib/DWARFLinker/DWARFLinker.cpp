//=== DWARFLinker.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinker/DWARFLinker.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/NonRelocatableStringpool.h"
#include "llvm/DWARFLinker/DWARFLinkerDeclContext.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDataExtractor.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include <vector>

namespace llvm {

/// Hold the input and output of the debug info size in bytes.
struct DebugInfoSize {
  uint64_t Input;
  uint64_t Output;
};

/// Compute the total size of the debug info.
static uint64_t getDebugInfoSize(DWARFContext &Dwarf) {
  uint64_t Size = 0;
  for (auto &Unit : Dwarf.compile_units()) {
    Size += Unit->getLength();
  }
  return Size;
}

/// Similar to DWARFUnitSection::getUnitForOffset(), but returning our
/// CompileUnit object instead.
static CompileUnit *getUnitForOffset(const UnitListTy &Units, uint64_t Offset) {
  auto CU = llvm::upper_bound(
      Units, Offset, [](uint64_t LHS, const std::unique_ptr<CompileUnit> &RHS) {
        return LHS < RHS->getOrigUnit().getNextUnitOffset();
      });
  return CU != Units.end() ? CU->get() : nullptr;
}

/// Resolve the DIE attribute reference that has been extracted in \p RefValue.
/// The resulting DIE might be in another CompileUnit which is stored into \p
/// ReferencedCU. \returns null if resolving fails for any reason.
DWARFDie DWARFLinker::resolveDIEReference(const DWARFFile &File,
                                          const UnitListTy &Units,
                                          const DWARFFormValue &RefValue,
                                          const DWARFDie &DIE,
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

  reportWarning("could not find referenced DIE", File, &DIE);
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

AddressesMap::~AddressesMap() {}

DwarfEmitter::~DwarfEmitter() {}

static Optional<StringRef> StripTemplateParameters(StringRef Name) {
  // We are looking for template parameters to strip from Name. e.g.
  //
  //  operator<<B>
  //
  // We look for > at the end but if it does not contain any < then we
  // have something like operator>>. We check for the operator<=> case.
  if (!Name.endswith(">") || Name.count("<") == 0 || Name.endswith("<=>"))
    return {};

  // How many < until we have the start of the template parameters.
  size_t NumLeftAnglesToSkip = 1;

  // If we have operator<=> then we need to skip its < as well.
  NumLeftAnglesToSkip += Name.count("<=>");

  size_t RightAngleCount = Name.count('>');
  size_t LeftAngleCount = Name.count('<');

  // If we have more < than > we have operator< or operator<<
  // we to account for their < as well.
  if (LeftAngleCount > RightAngleCount)
    NumLeftAnglesToSkip += LeftAngleCount - RightAngleCount;

  size_t StartOfTemplate = 0;
  while (NumLeftAnglesToSkip--)
    StartOfTemplate = Name.find('<', StartOfTemplate) + 1;

  return Name.substr(0, StartOfTemplate - 1);
}

bool DWARFLinker::DIECloner::getDIENames(const DWARFDie &Die,
                                         AttributesInfo &Info,
                                         OffsetsStringPool &StringPool,
                                         bool StripTemplate) {
  // This function will be called on DIEs having low_pcs and
  // ranges. As getting the name might be more expansive, filter out
  // blocks directly.
  if (Die.getTag() == dwarf::DW_TAG_lexical_block)
    return false;

  if (!Info.MangledName)
    if (const char *MangledName = Die.getLinkageName())
      Info.MangledName = StringPool.getEntry(MangledName);

  if (!Info.Name)
    if (const char *Name = Die.getShortName())
      Info.Name = StringPool.getEntry(Name);

  if (!Info.MangledName)
    Info.MangledName = Info.Name;

  if (StripTemplate && Info.Name && Info.MangledName != Info.Name) {
    StringRef Name = Info.Name.getString();
    if (Optional<StringRef> StrippedName = StripTemplateParameters(Name))
      Info.NameWithoutTemplate = StringPool.getEntry(*StrippedName);
  }

  return Info.Name || Info.MangledName;
}

/// Resolve the relative path to a build artifact referenced by DWARF by
/// applying DW_AT_comp_dir.
static void resolveRelativeObjectPath(SmallVectorImpl<char> &Buf, DWARFDie CU) {
  sys::path::append(Buf, dwarf::toString(CU.find(dwarf::DW_AT_comp_dir), ""));
}

/// Collect references to parseable Swift interfaces in imported
/// DW_TAG_module blocks.
static void analyzeImportedModule(
    const DWARFDie &DIE, CompileUnit &CU,
    swiftInterfacesMap *ParseableSwiftInterfaces,
    std::function<void(const Twine &, const DWARFDie &)> ReportWarning) {
  if (CU.getLanguage() != dwarf::DW_LANG_Swift)
    return;

  if (!ParseableSwiftInterfaces)
    return;

  StringRef Path = dwarf::toStringRef(DIE.find(dwarf::DW_AT_LLVM_include_path));
  if (!Path.endswith(".swiftinterface"))
    return;
  // Don't track interfaces that are part of the SDK.
  StringRef SysRoot = dwarf::toStringRef(DIE.find(dwarf::DW_AT_LLVM_sysroot));
  if (SysRoot.empty())
    SysRoot = CU.getSysRoot();
  if (!SysRoot.empty() && Path.startswith(SysRoot))
    return;
  if (Optional<DWARFFormValue> Val = DIE.find(dwarf::DW_AT_name))
    if (Optional<const char *> Name = Val->getAsCString()) {
      auto &Entry = (*ParseableSwiftInterfaces)[*Name];
      // The prepend path is applied later when copying.
      DWARFDie CUDie = CU.getOrigUnit().getUnitDIE();
      SmallString<128> ResolvedPath;
      if (sys::path::is_relative(Path))
        resolveRelativeObjectPath(ResolvedPath, CUDie);
      sys::path::append(ResolvedPath, Path);
      if (!Entry.empty() && Entry != ResolvedPath)
        ReportWarning(
            Twine("Conflicting parseable interfaces for Swift Module ") +
                *Name + ": " + Entry + " and " + Path,
            DIE);
      Entry = std::string(ResolvedPath.str());
    }
}

/// The distinct types of work performed by the work loop in
/// analyzeContextInfo.
enum class ContextWorklistItemType : uint8_t {
  AnalyzeContextInfo,
  UpdateChildPruning,
  UpdatePruning,
};

/// This class represents an item in the work list. The type defines what kind
/// of work needs to be performed when processing the current item. Everything
/// but the Type and Die fields are optional based on the type.
struct ContextWorklistItem {
  DWARFDie Die;
  unsigned ParentIdx;
  union {
    CompileUnit::DIEInfo *OtherInfo;
    DeclContext *Context;
  };
  ContextWorklistItemType Type;
  bool InImportedModule;

  ContextWorklistItem(DWARFDie Die, ContextWorklistItemType T,
                      CompileUnit::DIEInfo *OtherInfo = nullptr)
      : Die(Die), ParentIdx(0), OtherInfo(OtherInfo), Type(T),
        InImportedModule(false) {}

  ContextWorklistItem(DWARFDie Die, DeclContext *Context, unsigned ParentIdx,
                      bool InImportedModule)
      : Die(Die), ParentIdx(ParentIdx), Context(Context),
        Type(ContextWorklistItemType::AnalyzeContextInfo),
        InImportedModule(InImportedModule) {}
};

static bool updatePruning(const DWARFDie &Die, CompileUnit &CU,
                          uint64_t ModulesEndOffset) {
  CompileUnit::DIEInfo &Info = CU.getInfo(Die);

  // Prune this DIE if it is either a forward declaration inside a
  // DW_TAG_module or a DW_TAG_module that contains nothing but
  // forward declarations.
  Info.Prune &= (Die.getTag() == dwarf::DW_TAG_module) ||
                (isTypeTag(Die.getTag()) &&
                 dwarf::toUnsigned(Die.find(dwarf::DW_AT_declaration), 0));

  // Only prune forward declarations inside a DW_TAG_module for which a
  // definition exists elsewhere.
  if (ModulesEndOffset == 0)
    Info.Prune &= Info.Ctxt && Info.Ctxt->getCanonicalDIEOffset();
  else
    Info.Prune &= Info.Ctxt && Info.Ctxt->getCanonicalDIEOffset() > 0 &&
                  Info.Ctxt->getCanonicalDIEOffset() <= ModulesEndOffset;

  return Info.Prune;
}

static void updateChildPruning(const DWARFDie &Die, CompileUnit &CU,
                               CompileUnit::DIEInfo &ChildInfo) {
  CompileUnit::DIEInfo &Info = CU.getInfo(Die);
  Info.Prune &= ChildInfo.Prune;
}

/// Recursive helper to build the global DeclContext information and
/// gather the child->parent relationships in the original compile unit.
///
/// This function uses the same work list approach as lookForDIEsToKeep.
///
/// \return true when this DIE and all of its children are only
/// forward declarations to types defined in external clang modules
/// (i.e., forward declarations that are children of a DW_TAG_module).
static bool analyzeContextInfo(
    const DWARFDie &DIE, unsigned ParentIdx, CompileUnit &CU,
    DeclContext *CurrentDeclContext, DeclContextTree &Contexts,
    uint64_t ModulesEndOffset, swiftInterfacesMap *ParseableSwiftInterfaces,
    std::function<void(const Twine &, const DWARFDie &)> ReportWarning,
    bool InImportedModule = false) {
  // LIFO work list.
  std::vector<ContextWorklistItem> Worklist;
  Worklist.emplace_back(DIE, CurrentDeclContext, ParentIdx, InImportedModule);

  while (!Worklist.empty()) {
    ContextWorklistItem Current = Worklist.back();
    Worklist.pop_back();

    switch (Current.Type) {
    case ContextWorklistItemType::UpdatePruning:
      updatePruning(Current.Die, CU, ModulesEndOffset);
      continue;
    case ContextWorklistItemType::UpdateChildPruning:
      updateChildPruning(Current.Die, CU, *Current.OtherInfo);
      continue;
    case ContextWorklistItemType::AnalyzeContextInfo:
      break;
    }

    unsigned Idx = CU.getOrigUnit().getDIEIndex(Current.Die);
    CompileUnit::DIEInfo &Info = CU.getInfo(Idx);

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
    if (Current.Die.getTag() == dwarf::DW_TAG_module &&
        Current.ParentIdx == 0 &&
        dwarf::toString(Current.Die.find(dwarf::DW_AT_name), "") !=
            CU.getClangModuleName()) {
      Current.InImportedModule = true;
      analyzeImportedModule(Current.Die, CU, ParseableSwiftInterfaces,
                            ReportWarning);
    }

    Info.ParentIdx = Current.ParentIdx;
    bool InClangModule = CU.isClangModule() || Current.InImportedModule;
    if (CU.hasODR() || InClangModule) {
      if (Current.Context) {
        auto PtrInvalidPair = Contexts.getChildDeclContext(
            *Current.Context, Current.Die, CU, InClangModule);
        Current.Context = PtrInvalidPair.getPointer();
        Info.Ctxt =
            PtrInvalidPair.getInt() ? nullptr : PtrInvalidPair.getPointer();
        if (Info.Ctxt)
          Info.Ctxt->setDefinedInClangModule(InClangModule);
      } else
        Info.Ctxt = Current.Context = nullptr;
    }

    Info.Prune = Current.InImportedModule;
    // Add children in reverse order to the worklist to effectively process
    // them in order.
    Worklist.emplace_back(Current.Die, ContextWorklistItemType::UpdatePruning);
    for (auto Child : reverse(Current.Die.children())) {
      CompileUnit::DIEInfo &ChildInfo = CU.getInfo(Child);
      Worklist.emplace_back(
          Current.Die, ContextWorklistItemType::UpdateChildPruning, &ChildInfo);
      Worklist.emplace_back(Child, Current.Context, Idx,
                            Current.InImportedModule);
    }
  }

  return CU.getInfo(DIE).Prune;
}

static bool dieNeedsChildrenToBeMeaningful(uint32_t Tag) {
  switch (Tag) {
  default:
    return false;
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_common_block:
  case dwarf::DW_TAG_lexical_block:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_union_type:
    return true;
  }
  llvm_unreachable("Invalid Tag");
}

void DWARFLinker::cleanupAuxiliarryData(LinkContext &Context) {
  Context.clear();

  for (auto I = DIEBlocks.begin(), E = DIEBlocks.end(); I != E; ++I)
    (*I)->~DIEBlock();
  for (auto I = DIELocs.begin(), E = DIELocs.end(); I != E; ++I)
    (*I)->~DIELoc();

  DIEBlocks.clear();
  DIELocs.clear();
  DIEAlloc.Reset();
}

/// Check if a variable describing DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DWARFLinker::shouldKeepVariableDIE(AddressesMap &RelocMgr,
                                            const DWARFDie &DIE,
                                            CompileUnit::DIEInfo &MyInfo,
                                            unsigned Flags) {
  const auto *Abbrev = DIE.getAbbreviationDeclarationPtr();

  // Global variables with constant value can always be kept.
  if (!(Flags & TF_InFunctionScope) &&
      Abbrev->findAttributeIndex(dwarf::DW_AT_const_value)) {
    MyInfo.InDebugMap = true;
    return Flags | TF_Keep;
  }

  // See if there is a relocation to a valid debug map entry inside
  // this variable's location. The order is important here. We want to
  // always check if the variable has a valid relocation, so that the
  // DIEInfo is filled. However, we don't want a static variable in a
  // function to force us to keep the enclosing function.
  if (!RelocMgr.hasLiveMemoryLocation(DIE, MyInfo) ||
      (Flags & TF_InFunctionScope))
    return Flags;

  if (Options.Verbose) {
    outs() << "Keeping variable DIE:";
    DIDumpOptions DumpOpts;
    DumpOpts.ChildRecurseDepth = 0;
    DumpOpts.Verbose = Options.Verbose;
    DIE.dump(outs(), 8 /* Indent */, DumpOpts);
  }

  return Flags | TF_Keep;
}

/// Check if a function describing DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DWARFLinker::shouldKeepSubprogramDIE(
    AddressesMap &RelocMgr, RangesTy &Ranges, const DWARFDie &DIE,
    const DWARFFile &File, CompileUnit &Unit, CompileUnit::DIEInfo &MyInfo,
    unsigned Flags) {
  Flags |= TF_InFunctionScope;

  auto LowPc = dwarf::toAddress(DIE.find(dwarf::DW_AT_low_pc));
  if (!LowPc)
    return Flags;

  assert(LowPc.hasValue() && "low_pc attribute is not an address.");
  if (!RelocMgr.hasLiveAddressRange(DIE, MyInfo))
    return Flags;

  if (Options.Verbose) {
    outs() << "Keeping subprogram DIE:";
    DIDumpOptions DumpOpts;
    DumpOpts.ChildRecurseDepth = 0;
    DumpOpts.Verbose = Options.Verbose;
    DIE.dump(outs(), 8 /* Indent */, DumpOpts);
  }

  if (DIE.getTag() == dwarf::DW_TAG_label) {
    if (Unit.hasLabelAt(*LowPc))
      return Flags;

    DWARFUnit &OrigUnit = Unit.getOrigUnit();
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
    reportWarning("Function without high_pc. Range will be discarded.\n", File,
                  &DIE);
    return Flags;
  }

  // Replace the debug map range with a more accurate one.
  Ranges[*LowPc] = ObjFileAddressRange(*HighPc, MyInfo.AddrAdjust);
  Unit.addFunctionRange(*LowPc, *HighPc, MyInfo.AddrAdjust);
  return Flags;
}

/// Check if a DIE should be kept.
/// \returns updated TraversalFlags.
unsigned DWARFLinker::shouldKeepDIE(AddressesMap &RelocMgr, RangesTy &Ranges,
                                    const DWARFDie &DIE, const DWARFFile &File,
                                    CompileUnit &Unit,
                                    CompileUnit::DIEInfo &MyInfo,
                                    unsigned Flags) {
  switch (DIE.getTag()) {
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_variable:
    return shouldKeepVariableDIE(RelocMgr, DIE, MyInfo, Flags);
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_label:
    return shouldKeepSubprogramDIE(RelocMgr, Ranges, DIE, File, Unit, MyInfo,
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

/// Helper that updates the completeness of the current DIE based on the
/// completeness of one of its children. It depends on the incompleteness of
/// the children already being computed.
static void updateChildIncompleteness(const DWARFDie &Die, CompileUnit &CU,
                                      CompileUnit::DIEInfo &ChildInfo) {
  switch (Die.getTag()) {
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_class_type:
    break;
  default:
    return;
  }

  CompileUnit::DIEInfo &MyInfo = CU.getInfo(Die);

  if (ChildInfo.Incomplete || ChildInfo.Prune)
    MyInfo.Incomplete = true;
}

/// Helper that updates the completeness of the current DIE based on the
/// completeness of the DIEs it references. It depends on the incompleteness of
/// the referenced DIE already being computed.
static void updateRefIncompleteness(const DWARFDie &Die, CompileUnit &CU,
                                    CompileUnit::DIEInfo &RefInfo) {
  switch (Die.getTag()) {
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_member:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_ptr_to_member_type:
  case dwarf::DW_TAG_pointer_type:
    break;
  default:
    return;
  }

  CompileUnit::DIEInfo &MyInfo = CU.getInfo(Die);

  if (MyInfo.Incomplete)
    return;

  if (RefInfo.Incomplete)
    MyInfo.Incomplete = true;
}

/// Look at the children of the given DIE and decide whether they should be
/// kept.
void DWARFLinker::lookForChildDIEsToKeep(
    const DWARFDie &Die, CompileUnit &CU, unsigned Flags,
    SmallVectorImpl<WorklistItem> &Worklist) {
  // The TF_ParentWalk flag tells us that we are currently walking up the
  // parent chain of a required DIE, and we don't want to mark all the children
  // of the parents as kept (consider for example a DW_TAG_namespace node in
  // the parent chain). There are however a set of DIE types for which we want
  // to ignore that directive and still walk their children.
  if (dieNeedsChildrenToBeMeaningful(Die.getTag()))
    Flags &= ~DWARFLinker::TF_ParentWalk;

  // We're finished if this DIE has no children or we're walking the parent
  // chain.
  if (!Die.hasChildren() || (Flags & DWARFLinker::TF_ParentWalk))
    return;

  // Add children in reverse order to the worklist to effectively process them
  // in order.
  for (auto Child : reverse(Die.children())) {
    // Add a worklist item before every child to calculate incompleteness right
    // after the current child is processed.
    CompileUnit::DIEInfo &ChildInfo = CU.getInfo(Child);
    Worklist.emplace_back(Die, CU, WorklistItemType::UpdateChildIncompleteness,
                          &ChildInfo);
    Worklist.emplace_back(Child, CU, Flags);
  }
}

/// Look at DIEs referenced by the given DIE and decide whether they should be
/// kept. All DIEs referenced though attributes should be kept.
void DWARFLinker::lookForRefDIEsToKeep(
    const DWARFDie &Die, CompileUnit &CU, unsigned Flags,
    const UnitListTy &Units, const DWARFFile &File,
    SmallVectorImpl<WorklistItem> &Worklist) {
  bool UseOdr = (Flags & DWARFLinker::TF_DependencyWalk)
                    ? (Flags & DWARFLinker::TF_ODR)
                    : CU.hasODR();
  DWARFUnit &Unit = CU.getOrigUnit();
  DWARFDataExtractor Data = Unit.getDebugInfoExtractor();
  const auto *Abbrev = Die.getAbbreviationDeclarationPtr();
  uint64_t Offset = Die.getOffset() + getULEB128Size(Abbrev->getCode());

  SmallVector<std::pair<DWARFDie, CompileUnit &>, 4> ReferencedDIEs;
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
    if (auto RefDie =
            resolveDIEReference(File, Units, Val, Die, ReferencedCU)) {
      CompileUnit::DIEInfo &Info = ReferencedCU->getInfo(RefDie);
      bool IsModuleRef = Info.Ctxt && Info.Ctxt->getCanonicalDIEOffset() &&
                         Info.Ctxt->isDefinedInClangModule();
      // If the referenced DIE has a DeclContext that has already been
      // emitted, then do not keep the one in this CU. We'll link to
      // the canonical DIE in cloneDieReferenceAttribute.
      //
      // FIXME: compatibility with dsymutil-classic. UseODR shouldn't
      // be necessary and could be advantageously replaced by
      // ReferencedCU->hasODR() && CU.hasODR().
      //
      // FIXME: compatibility with dsymutil-classic. There is no
      // reason not to unique ref_addr references.
      if (AttrSpec.Form != dwarf::DW_FORM_ref_addr && (UseOdr || IsModuleRef) &&
          Info.Ctxt &&
          Info.Ctxt != ReferencedCU->getInfo(Info.ParentIdx).Ctxt &&
          Info.Ctxt->getCanonicalDIEOffset() && isODRAttribute(AttrSpec.Attr))
        continue;

      // Keep a module forward declaration if there is no definition.
      if (!(isODRAttribute(AttrSpec.Attr) && Info.Ctxt &&
            Info.Ctxt->getCanonicalDIEOffset()))
        Info.Prune = false;
      ReferencedDIEs.emplace_back(RefDie, *ReferencedCU);
    }
  }

  unsigned ODRFlag = UseOdr ? DWARFLinker::TF_ODR : 0;

  // Add referenced DIEs in reverse order to the worklist to effectively
  // process them in order.
  for (auto &P : reverse(ReferencedDIEs)) {
    // Add a worklist item before every child to calculate incompleteness right
    // after the current child is processed.
    CompileUnit::DIEInfo &Info = P.second.getInfo(P.first);
    Worklist.emplace_back(Die, CU, WorklistItemType::UpdateRefIncompleteness,
                          &Info);
    Worklist.emplace_back(P.first, P.second,
                          DWARFLinker::TF_Keep |
                              DWARFLinker::TF_DependencyWalk | ODRFlag);
  }
}

/// Look at the parent of the given DIE and decide whether they should be kept.
void DWARFLinker::lookForParentDIEsToKeep(
    unsigned AncestorIdx, CompileUnit &CU, unsigned Flags,
    SmallVectorImpl<WorklistItem> &Worklist) {
  // Stop if we encounter an ancestor that's already marked as kept.
  if (CU.getInfo(AncestorIdx).Keep)
    return;

  DWARFUnit &Unit = CU.getOrigUnit();
  DWARFDie ParentDIE = Unit.getDIEAtIndex(AncestorIdx);
  Worklist.emplace_back(CU.getInfo(AncestorIdx).ParentIdx, CU, Flags);
  Worklist.emplace_back(ParentDIE, CU, Flags);
}

/// Recursively walk the \p DIE tree and look for DIEs to keep. Store that
/// information in \p CU's DIEInfo.
///
/// This function is the entry point of the DIE selection algorithm. It is
/// expected to walk the DIE tree in file order and (though the mediation of
/// its helper) call hasValidRelocation() on each DIE that might be a 'root
/// DIE' (See DwarfLinker class comment).
///
/// While walking the dependencies of root DIEs, this function is also called,
/// but during these dependency walks the file order is not respected. The
/// TF_DependencyWalk flag tells us which kind of traversal we are currently
/// doing.
///
/// The recursive algorithm is implemented iteratively as a work list because
/// very deep recursion could exhaust the stack for large projects. The work
/// list acts as a scheduler for different types of work that need to be
/// performed.
///
/// The recursive nature of the algorithm is simulated by running the "main"
/// algorithm (LookForDIEsToKeep) followed by either looking at more DIEs
/// (LookForChildDIEsToKeep, LookForRefDIEsToKeep, LookForParentDIEsToKeep) or
/// fixing up a computed property (UpdateChildIncompleteness,
/// UpdateRefIncompleteness).
///
/// The return value indicates whether the DIE is incomplete.
void DWARFLinker::lookForDIEsToKeep(AddressesMap &AddressesMap,
                                    RangesTy &Ranges, const UnitListTy &Units,
                                    const DWARFDie &Die, const DWARFFile &File,
                                    CompileUnit &Cu, unsigned Flags) {
  // LIFO work list.
  SmallVector<WorklistItem, 4> Worklist;
  Worklist.emplace_back(Die, Cu, Flags);

  while (!Worklist.empty()) {
    WorklistItem Current = Worklist.pop_back_val();

    // Look at the worklist type to decide what kind of work to perform.
    switch (Current.Type) {
    case WorklistItemType::UpdateChildIncompleteness:
      updateChildIncompleteness(Current.Die, Current.CU, *Current.OtherInfo);
      continue;
    case WorklistItemType::UpdateRefIncompleteness:
      updateRefIncompleteness(Current.Die, Current.CU, *Current.OtherInfo);
      continue;
    case WorklistItemType::LookForChildDIEsToKeep:
      lookForChildDIEsToKeep(Current.Die, Current.CU, Current.Flags, Worklist);
      continue;
    case WorklistItemType::LookForRefDIEsToKeep:
      lookForRefDIEsToKeep(Current.Die, Current.CU, Current.Flags, Units, File,
                           Worklist);
      continue;
    case WorklistItemType::LookForParentDIEsToKeep:
      lookForParentDIEsToKeep(Current.AncestorIdx, Current.CU, Current.Flags,
                              Worklist);
      continue;
    case WorklistItemType::LookForDIEsToKeep:
      break;
    }

    unsigned Idx = Current.CU.getOrigUnit().getDIEIndex(Current.Die);
    CompileUnit::DIEInfo &MyInfo = Current.CU.getInfo(Idx);

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
      Current.Flags = shouldKeepDIE(AddressesMap, Ranges, Current.Die, File,
                                    Current.CU, MyInfo, Current.Flags);

    // Finish by looking for child DIEs. Because of the LIFO worklist we need
    // to schedule that work before any subsequent items are added to the
    // worklist.
    Worklist.emplace_back(Current.Die, Current.CU, Current.Flags,
                          WorklistItemType::LookForChildDIEsToKeep);

    if (AlreadyKept || !(Current.Flags & TF_Keep))
      continue;

    // If it is a newly kept DIE mark it as well as all its dependencies as
    // kept.
    MyInfo.Keep = true;

    // We're looking for incomplete types.
    MyInfo.Incomplete =
        Current.Die.getTag() != dwarf::DW_TAG_subprogram &&
        Current.Die.getTag() != dwarf::DW_TAG_member &&
        dwarf::toUnsigned(Current.Die.find(dwarf::DW_AT_declaration), 0);

    // After looking at the parent chain, look for referenced DIEs. Because of
    // the LIFO worklist we need to schedule that work before any subsequent
    // items are added to the worklist.
    Worklist.emplace_back(Current.Die, Current.CU, Current.Flags,
                          WorklistItemType::LookForRefDIEsToKeep);

    bool UseOdr = (Current.Flags & TF_DependencyWalk) ? (Current.Flags & TF_ODR)
                                                      : Current.CU.hasODR();
    unsigned ODRFlag = UseOdr ? TF_ODR : 0;
    unsigned ParFlags = TF_ParentWalk | TF_Keep | TF_DependencyWalk | ODRFlag;

    // Now schedule the parent walk.
    Worklist.emplace_back(MyInfo.ParentIdx, Current.CU, ParFlags);
  }
}

/// Assign an abbreviation number to \p Abbrev.
///
/// Our DIEs get freed after every DebugMapObject has been processed,
/// thus the FoldingSet we use to unique DIEAbbrevs cannot refer to
/// the instances hold by the DIEs. When we encounter an abbreviation
/// that we don't know, we create a permanent copy of it.
void DWARFLinker::assignAbbrev(DIEAbbrev &Abbrev) {
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
        std::make_unique<DIEAbbrev>(Abbrev.getTag(), Abbrev.hasChildren()));
    for (const auto &Attr : Abbrev.getData())
      Abbreviations.back()->AddAttribute(Attr.getAttribute(), Attr.getForm());
    AbbreviationsSet.InsertNode(Abbreviations.back().get(), InsertToken);
    // Assign the unique abbreviation number.
    Abbrev.setNumber(Abbreviations.size());
    Abbreviations.back()->setNumber(Abbreviations.size());
  }
}

unsigned DWARFLinker::DIECloner::cloneStringAttribute(
    DIE &Die, AttributeSpec AttrSpec, const DWARFFormValue &Val,
    const DWARFUnit &U, OffsetsStringPool &StringPool, AttributesInfo &Info) {
  Optional<const char *> String = Val.getAsCString();
  if (!String)
    return 0;

  // Switch everything to out of line strings.
  auto StringEntry = StringPool.getEntry(*String);

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

unsigned DWARFLinker::DIECloner::cloneDieReferenceAttribute(
    DIE &Die, const DWARFDie &InputDIE, AttributeSpec AttrSpec,
    unsigned AttrSize, const DWARFFormValue &Val, const DWARFFile &File,
    CompileUnit &Unit) {
  const DWARFUnit &U = Unit.getOrigUnit();
  uint64_t Ref = *Val.getAsReference();

  DIE *NewRefDie = nullptr;
  CompileUnit *RefUnit = nullptr;
  DeclContext *Ctxt = nullptr;

  DWARFDie RefDie =
      Linker.resolveDIEReference(File, CompileUnits, Val, InputDIE, RefUnit);

  // If the referenced DIE is not found,  drop the attribute.
  if (!RefDie || AttrSpec.Attr == dwarf::DW_AT_sibling)
    return 0;

  CompileUnit::DIEInfo &RefInfo = RefUnit->getInfo(RefDie);

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

void DWARFLinker::DIECloner::cloneExpression(
    DataExtractor &Data, DWARFExpression Expression, const DWARFFile &File,
    CompileUnit &Unit, SmallVectorImpl<uint8_t> &OutputBuffer) {
  using Encoding = DWARFExpression::Operation::Encoding;

  uint64_t OpOffset = 0;
  for (auto &Op : Expression) {
    auto Description = Op.getDescription();
    // DW_OP_const_type is variable-length and has 3
    // operands. DWARFExpression thus far only supports 2.
    auto Op0 = Description.Op[0];
    auto Op1 = Description.Op[1];
    if ((Op0 == Encoding::BaseTypeRef && Op1 != Encoding::SizeNA) ||
        (Op1 == Encoding::BaseTypeRef && Op0 != Encoding::Size1))
      Linker.reportWarning("Unsupported DW_OP encoding.", File);

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
      uint32_t Offset = 0;
      // Look up the base type. For DW_OP_convert, the operand may be 0 to
      // instead indicate the generic type. The same holds for
      // DW_OP_reinterpret, which is currently not supported.
      if (RefOffset > 0 || Op.getCode() != dwarf::DW_OP_convert) {
        auto RefDie = Unit.getOrigUnit().getDIEForOffset(RefOffset);
        CompileUnit::DIEInfo &Info = Unit.getInfo(RefDie);
        if (DIE *Clone = Info.Clone)
          Offset = Clone->getOffset();
        else
          Linker.reportWarning(
              "base type ref doesn't point to DW_TAG_base_type.", File);
      }
      uint8_t ULEB[16];
      unsigned RealSize = encodeULEB128(Offset, ULEB, ULEBsize);
      if (RealSize > ULEBsize) {
        // Emit the generic type as a fallback.
        RealSize = encodeULEB128(0, ULEB, ULEBsize);
        Linker.reportWarning("base type ref doesn't fit.", File);
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

unsigned DWARFLinker::DIECloner::cloneBlockAttribute(
    DIE &Die, const DWARFFile &File, CompileUnit &Unit, AttributeSpec AttrSpec,
    const DWARFFormValue &Val, unsigned AttrSize, bool IsLittleEndian) {
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
    DWARFExpression Expr(Data, OrigUnit.getAddressByteSize(),
                         OrigUnit.getFormParams().Format);
    cloneExpression(Data, Expr, File, Unit, Buffer);
    Bytes = Buffer;
  }
  for (auto Byte : Bytes)
    Attr->addValue(DIEAlloc, static_cast<dwarf::Attribute>(0),
                   dwarf::DW_FORM_data1, DIEInteger(Byte));

  // FIXME: If DIEBlock and DIELoc just reuses the Size field of
  // the DIE class, this "if" could be replaced by
  // Attr->setSize(Bytes.size()).
  if (Loc)
    Loc->setSize(Bytes.size());
  else
    Block->setSize(Bytes.size());

  Die.addValue(DIEAlloc, Value);
  return AttrSize;
}

unsigned DWARFLinker::DIECloner::cloneAddressAttribute(
    DIE &Die, AttributeSpec AttrSpec, const DWARFFormValue &Val,
    const CompileUnit &Unit, AttributesInfo &Info) {
  if (LLVM_UNLIKELY(Linker.Options.Update)) {
    if (AttrSpec.Attr == dwarf::DW_AT_low_pc)
      Info.HasLowPc = true;
    Die.addValue(DIEAlloc, dwarf::Attribute(AttrSpec.Attr),
                 dwarf::Form(AttrSpec.Form), DIEInteger(Val.getRawUValue()));
    return Unit.getOrigUnit().getAddressByteSize();
  }

  dwarf::Form Form = AttrSpec.Form;
  uint64_t Addr = 0;
  if (Form == dwarf::DW_FORM_addrx) {
    if (Optional<uint64_t> AddrOffsetSectionBase =
            Unit.getOrigUnit().getAddrOffsetSectionBase()) {
      uint64_t StartOffset = *AddrOffsetSectionBase + Val.getRawUValue();
      uint64_t EndOffset =
          StartOffset + Unit.getOrigUnit().getAddressByteSize();
      if (llvm::Expected<uint64_t> RelocAddr =
              ObjFile.Addresses->relocateIndexedAddr(StartOffset, EndOffset))
        Addr = *RelocAddr;
      else
        Linker.reportWarning(toString(RelocAddr.takeError()), ObjFile);
    } else
      Linker.reportWarning("no base offset for address table", ObjFile);

    // If this is an indexed address emit the debug_info address.
    Form = dwarf::DW_FORM_addr;
  } else
    Addr = *Val.getAsAddress();

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
  } else if (AttrSpec.Attr == dwarf::DW_AT_call_return_pc) {
    // Relocate a return PC address within a call site entry.
    if (Die.getTag() == dwarf::DW_TAG_call_site)
      Addr = (Info.OrigCallReturnPc ? Info.OrigCallReturnPc : Addr) +
             Info.PCOffset;
  } else if (AttrSpec.Attr == dwarf::DW_AT_call_pc) {
    // Relocate the address of a branch instruction within a call site entry.
    if (Die.getTag() == dwarf::DW_TAG_call_site)
      Addr = (Info.OrigCallPc ? Info.OrigCallPc : Addr) + Info.PCOffset;
  }

  Die.addValue(DIEAlloc, static_cast<dwarf::Attribute>(AttrSpec.Attr),
               static_cast<dwarf::Form>(Form), DIEInteger(Addr));
  return Unit.getOrigUnit().getAddressByteSize();
}

unsigned DWARFLinker::DIECloner::cloneScalarAttribute(
    DIE &Die, const DWARFDie &InputDIE, const DWARFFile &File,
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
          "Unsupported scalar attribute form. Dropping attribute.", File,
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
        "Unsupported scalar attribute form. Dropping attribute.", File,
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
           AttrSpec.Attr == dwarf::DW_AT_frame_base) {
    Unit.noteLocationAttribute(Patch, Info.PCOffset);
  } else if (AttrSpec.Attr == dwarf::DW_AT_declaration && Value)
    Info.IsDeclaration = true;

  return AttrSize;
}

/// Clone \p InputDIE's attribute described by \p AttrSpec with
/// value \p Val, and add it to \p Die.
/// \returns the size of the cloned attribute.
unsigned DWARFLinker::DIECloner::cloneAttribute(
    DIE &Die, const DWARFDie &InputDIE, const DWARFFile &File,
    CompileUnit &Unit, OffsetsStringPool &StringPool, const DWARFFormValue &Val,
    const AttributeSpec AttrSpec, unsigned AttrSize, AttributesInfo &Info,
    bool IsLittleEndian) {
  const DWARFUnit &U = Unit.getOrigUnit();

  switch (AttrSpec.Form) {
  case dwarf::DW_FORM_strp:
  case dwarf::DW_FORM_string:
  case dwarf::DW_FORM_strx:
  case dwarf::DW_FORM_strx1:
  case dwarf::DW_FORM_strx2:
  case dwarf::DW_FORM_strx3:
  case dwarf::DW_FORM_strx4:
    return cloneStringAttribute(Die, AttrSpec, Val, U, StringPool, Info);
  case dwarf::DW_FORM_ref_addr:
  case dwarf::DW_FORM_ref1:
  case dwarf::DW_FORM_ref2:
  case dwarf::DW_FORM_ref4:
  case dwarf::DW_FORM_ref8:
    return cloneDieReferenceAttribute(Die, InputDIE, AttrSpec, AttrSize, Val,
                                      File, Unit);
  case dwarf::DW_FORM_block:
  case dwarf::DW_FORM_block1:
  case dwarf::DW_FORM_block2:
  case dwarf::DW_FORM_block4:
  case dwarf::DW_FORM_exprloc:
    return cloneBlockAttribute(Die, File, Unit, AttrSpec, Val, AttrSize,
                               IsLittleEndian);
  case dwarf::DW_FORM_addr:
  case dwarf::DW_FORM_addrx:
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
    return cloneScalarAttribute(Die, InputDIE, File, Unit, AttrSpec, Val,
                                AttrSize, Info);
  default:
    Linker.reportWarning("Unsupported attribute form " +
                             dwarf::FormEncodingString(AttrSpec.Form) +
                             " in cloneAttribute. Dropping.",
                         File, &InputDIE);
  }

  return 0;
}

static bool isObjCSelector(StringRef Name) {
  return Name.size() > 2 && (Name[0] == '-' || Name[0] == '+') &&
         (Name[1] == '[');
}

void DWARFLinker::DIECloner::addObjCAccelerator(CompileUnit &Unit,
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
      MethodNameNoCategory.append(std::string(SelectorStart));
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
  case dwarf::DW_AT_str_offsets_base:
    // FIXME: Use the string offset table with Dwarf 5.
    return true;
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

DIE *DWARFLinker::DIECloner::cloneDIE(const DWARFDie &InputDIE,
                                      const DWARFFile &File, CompileUnit &Unit,
                                      OffsetsStringPool &StringPool,
                                      int64_t PCOffset, uint32_t OutOffset,
                                      unsigned Flags, bool IsLittleEndian,
                                      DIE *Die) {
  DWARFUnit &U = Unit.getOrigUnit();
  unsigned Idx = U.getDIEIndex(InputDIE);
  CompileUnit::DIEInfo &Info = Unit.getInfo(Idx);

  // Should the DIE appear in the output?
  if (!Unit.getInfo(Idx).Keep)
    return nullptr;

  uint64_t Offset = InputDIE.getOffset();
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
  uint64_t NextOffset = (Idx + 1 < U.getNumDIEs())
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
  if (ObjFile.Addresses->areRelocationsResolved() &&
      ObjFile.Addresses->applyValidRelocs(DIECopy, Offset,
                                          Data.isLittleEndian())) {
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
    AttrInfo.OrigCallReturnPc =
        dwarf::toAddress(InputDIE.find(dwarf::DW_AT_call_return_pc), 0);
    AttrInfo.OrigCallPc =
        dwarf::toAddress(InputDIE.find(dwarf::DW_AT_call_pc), 0);
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
    if (!Info.InDebugMap && LLVM_LIKELY(!Update))
      Flags |= TF_SkipPC;
  }

  bool Copied = false;
  for (const auto &AttrSpec : Abbrev->attributes()) {
    if (LLVM_LIKELY(!Update) &&
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
    uint64_t AttrSize = Offset;
    Val.extractValue(Data, &Offset, U.getFormParams(), &U);
    AttrSize = Offset - AttrSize;

    OutOffset += cloneAttribute(*Die, InputDIE, File, Unit, StringPool, Val,
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
    uint32_t Hash = hashFullyQualifiedName(InputDIE, Unit, File);
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
  Linker.assignAbbrev(NewAbbrev);
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
    if (DIE *Clone = cloneDIE(Child, File, Unit, StringPool, PCOffset,
                              OutOffset, Flags, IsLittleEndian)) {
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
void DWARFLinker::patchRangesForUnit(const CompileUnit &Unit,
                                     DWARFContext &OrigDwarf,
                                     const DWARFFile &File) const {
  DWARFDebugRangeList RangeList;
  const auto &FunctionRanges = Unit.getFunctionRanges();
  unsigned AddressSize = Unit.getOrigUnit().getAddressByteSize();
  DWARFDataExtractor RangeExtractor(OrigDwarf.getDWARFObj(),
                                    OrigDwarf.getDWARFObj().getRangesSection(),
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
    uint64_t Offset = RangeAttribute.get();
    RangeAttribute.set(TheDwarfEmitter->getRangesSectionSize());
    if (Error E = RangeList.extract(RangeExtractor, &Offset)) {
      llvm::consumeError(std::move(E));
      reportWarning("invalid range list ignored.", File);
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
          reportWarning("no mapping for range.", File);
          continue;
        }
      }
    }

    TheDwarfEmitter->emitRangesEntries(UnitPcOffset, OrigLowPc, CurrRange,
                                       Entries, AddressSize);
  }
}

/// Generate the debug_aranges entries for \p Unit and if the
/// unit has a DW_AT_ranges attribute, also emit the debug_ranges
/// contribution for this attribute.
/// FIXME: this could actually be done right in patchRangesForUnit,
/// but for the sake of initial bit-for-bit compatibility with legacy
/// dsymutil, we have to do it in a delayed pass.
void DWARFLinker::generateUnitRanges(CompileUnit &Unit) const {
  auto Attr = Unit.getUnitRangesAttribute();
  if (Attr)
    Attr->set(TheDwarfEmitter->getRangesSectionSize());
  TheDwarfEmitter->emitUnitRangesEntries(Unit, static_cast<bool>(Attr));
}

/// Insert the new line info sequence \p Seq into the current
/// set of already linked line info \p Rows.
static void insertLineSequence(std::vector<DWARFDebugLine::Row> &Seq,
                               std::vector<DWARFDebugLine::Row> &Rows) {
  if (Seq.empty())
    return;

  if (!Rows.empty() && Rows.back().Address < Seq.front().Address) {
    llvm::append_range(Rows, Seq);
    Seq.clear();
    return;
  }

  object::SectionedAddress Front = Seq.front().Address;
  auto InsertPoint = partition_point(
      Rows, [=](const DWARFDebugLine::Row &O) { return O.Address < Front; });

  // FIXME: this only removes the unneeded end_sequence if the
  // sequences have been inserted in order. Using a global sort like
  // described in patchLineTableForUnit() and delaying the end_sequene
  // elimination to emitLineTableForUnit() we can get rid of all of them.
  if (InsertPoint != Rows.end() && InsertPoint->Address == Front &&
      InsertPoint->EndSequence) {
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
void DWARFLinker::patchLineTableForUnit(CompileUnit &Unit,
                                        DWARFContext &OrigDwarf,
                                        const DWARFFile &File) {
  DWARFDie CUDie = Unit.getOrigUnit().getUnitDIE();
  auto StmtList = dwarf::toSectionOffset(CUDie.find(dwarf::DW_AT_stmt_list));
  if (!StmtList)
    return;

  // Update the cloned DW_AT_stmt_list with the correct debug_line offset.
  if (auto *OutputDIE = Unit.getOutputUnitDIE())
    patchStmtList(*OutputDIE,
                  DIEInteger(TheDwarfEmitter->getLineSectionSize()));

  RangesTy &Ranges = File.Addresses->getValidAddressRanges();

  // Parse the original line info for the unit.
  DWARFDebugLine::LineTable LineTable;
  uint64_t StmtOffset = *StmtList;
  DWARFDataExtractor LineExtractor(
      OrigDwarf.getDWARFObj(), OrigDwarf.getDWARFObj().getLineSection(),
      OrigDwarf.isLittleEndian(), Unit.getOrigUnit().getAddressByteSize());
  if (needToTranslateStrings())
    return TheDwarfEmitter->translateLineTable(LineExtractor, StmtOffset);

  if (Error Err =
          LineTable.parse(LineExtractor, &StmtOffset, OrigDwarf,
                          &Unit.getOrigUnit(), OrigDwarf.getWarningHandler()))
    OrigDwarf.getWarningHandler()(std::move(Err));

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
          // Try harder by looking in the Address ranges map.
          // There are corner cases where this finds a
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
    reportWarning("line table parameters mismatch. Cannot emit.", File);
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
    TheDwarfEmitter->emitLineTableForUnit(
        Params, LineData.slice(*StmtList + 4, PrologueEnd),
        LineTable.Prologue.MinInstLength, NewRows,
        Unit.getOrigUnit().getAddressByteSize());
  }
}

void DWARFLinker::emitAcceleratorEntriesForUnit(CompileUnit &Unit) {
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

void DWARFLinker::emitAppleAcceleratorEntriesForUnit(CompileUnit &Unit) {
  // Add namespaces.
  for (const auto &Namespace : Unit.getNamespaces())
    AppleNamespaces.addName(Namespace.Name,
                            Namespace.Die->getOffset() + Unit.getStartOffset());

  /// Add names.
  TheDwarfEmitter->emitPubNamesForUnit(Unit);
  for (const auto &Pubname : Unit.getPubnames())
    AppleNames.addName(Pubname.Name,
                       Pubname.Die->getOffset() + Unit.getStartOffset());

  /// Add types.
  TheDwarfEmitter->emitPubTypesForUnit(Unit);
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

void DWARFLinker::emitDwarfAcceleratorEntriesForUnit(CompileUnit &Unit) {
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
/// patched frame descriptions for the resulting file.
///
/// This is actually pretty easy as the data of the CIEs and FDEs can
/// be considered as black boxes and moved as is. The only thing to do
/// is to patch the addresses in the headers.
void DWARFLinker::patchFrameInfoForObject(const DWARFFile &File,
                                          RangesTy &Ranges,
                                          DWARFContext &OrigDwarf,
                                          unsigned AddrSize) {
  StringRef FrameData = OrigDwarf.getDWARFObj().getFrameSection().Data;
  if (FrameData.empty())
    return;

  DataExtractor Data(FrameData, OrigDwarf.isLittleEndian(), 0);
  uint64_t InputOffset = 0;

  // Store the data of the CIEs defined in this object, keyed by their
  // offsets.
  DenseMap<uint64_t, StringRef> LocalCIES;

  while (Data.isValidOffset(InputOffset)) {
    uint64_t EntryOffset = InputOffset;
    uint32_t InitialLength = Data.getU32(&InputOffset);
    if (InitialLength == 0xFFFFFFFF)
      return reportWarning("Dwarf64 bits no supported", File);

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
    // in the debug map. Use the AddressInfo's range map to see if the FDE
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
      return reportWarning("Inconsistent debug_frame content. Dropping.", File);

    // Look if we already emitted a CIE that corresponds to the
    // referenced one (the CIE data is the key of that lookup).
    auto IteratorInserted = EmittedCIEs.insert(
        std::make_pair(CIEData, TheDwarfEmitter->getFrameSectionSize()));
    // If there is no CIE yet for this ID, emit it.
    if (IteratorInserted.second ||
        // FIXME: dsymutil-classic only caches the last used CIE for
        // reuse. Mimic that behavior for now. Just removing that
        // second half of the condition and the LastCIEOffset variable
        // makes the code DTRT.
        LastCIEOffset != IteratorInserted.first->getValue()) {
      LastCIEOffset = TheDwarfEmitter->getFrameSectionSize();
      IteratorInserted.first->getValue() = LastCIEOffset;
      TheDwarfEmitter->emitCIE(CIEData);
    }

    // Emit the FDE with updated address and CIE pointer.
    // (4 + AddrSize) is the size of the CIEId + initial_location
    // fields that will get reconstructed by emitFDE().
    unsigned FDERemainingBytes = InitialLength - (4 + AddrSize);
    TheDwarfEmitter->emitFDE(IteratorInserted.first->getValue(), AddrSize,
                             Loc + Range->second.Offset,
                             FrameData.substr(InputOffset, FDERemainingBytes));
    InputOffset += FDERemainingBytes;
  }
}

void DWARFLinker::DIECloner::copyAbbrev(
    const DWARFAbbreviationDeclaration &Abbrev, bool HasODR) {
  DIEAbbrev Copy(dwarf::Tag(Abbrev.getTag()),
                 dwarf::Form(Abbrev.hasChildren()));

  for (const auto &Attr : Abbrev.attributes()) {
    uint16_t Form = Attr.Form;
    if (HasODR && isODRAttribute(Attr.Attr))
      Form = dwarf::DW_FORM_ref_addr;
    Copy.AddAttribute(dwarf::Attribute(Attr.Attr), dwarf::Form(Form));
  }

  Linker.assignAbbrev(Copy);
}

uint32_t DWARFLinker::DIECloner::hashFullyQualifiedName(DWARFDie DIE,
                                                        CompileUnit &U,
                                                        const DWARFFile &File,
                                                        int ChildRecurseDepth) {
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
    if (auto RefDIE =
            Linker.resolveDIEReference(File, CompileUnits, *Ref, DIE, RefCU)) {
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
    return djbHash(Name ? Name : "", djbHash(ChildRecurseDepth ? "" : "::"));

  DWARFDie Die = OrigUnit->getDIEAtIndex(CU->getInfo(Idx).ParentIdx);
  return djbHash(
      (Name ? Name : ""),
      djbHash((Name ? "::" : ""),
              hashFullyQualifiedName(Die, *CU, File, ++ChildRecurseDepth)));
}

static uint64_t getDwoId(const DWARFDie &CUDie, const DWARFUnit &Unit) {
  auto DwoId = dwarf::toUnsigned(
      CUDie.find({dwarf::DW_AT_dwo_id, dwarf::DW_AT_GNU_dwo_id}));
  if (DwoId)
    return *DwoId;
  return 0;
}

static std::string remapPath(StringRef Path,
                             const objectPrefixMap &ObjectPrefixMap) {
  if (ObjectPrefixMap.empty())
    return Path.str();

  SmallString<256> p = Path;
  for (const auto &Entry : ObjectPrefixMap)
    if (llvm::sys::path::replace_path_prefix(p, Entry.first, Entry.second))
      break;
  return p.str().str();
}

bool DWARFLinker::registerModuleReference(DWARFDie CUDie, const DWARFUnit &Unit,
                                          const DWARFFile &File,
                                          OffsetsStringPool &StringPool,
                                          DeclContextTree &ODRContexts,
                                          uint64_t ModulesEndOffset,
                                          unsigned &UnitID, bool IsLittleEndian,
                                          unsigned Indent, bool Quiet) {
  std::string PCMfile = dwarf::toString(
      CUDie.find({dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}), "");
  if (PCMfile.empty())
    return false;
  if (Options.ObjectPrefixMap)
    PCMfile = remapPath(PCMfile, *Options.ObjectPrefixMap);

  // Clang module DWARF skeleton CUs abuse this for the path to the module.
  uint64_t DwoId = getDwoId(CUDie, Unit);

  std::string Name = dwarf::toString(CUDie.find(dwarf::DW_AT_name), "");
  if (Name.empty()) {
    if (!Quiet)
      reportWarning("Anonymous module skeleton CU for " + PCMfile, File);
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
                    File);
    if (!Quiet && Options.Verbose)
      outs() << " [cached].\n";
    return true;
  }
  if (!Quiet && Options.Verbose)
    outs() << " ...\n";

  // Cyclic dependencies are disallowed by Clang, but we still
  // shouldn't run into an infinite loop, so mark it as processed now.
  ClangModules.insert({PCMfile, DwoId});

  if (Error E = loadClangModule(CUDie, PCMfile, Name, DwoId, File, StringPool,
                                ODRContexts, ModulesEndOffset, UnitID,
                                IsLittleEndian, Indent + 2, Quiet)) {
    consumeError(std::move(E));
    return false;
  }
  return true;
}

Error DWARFLinker::loadClangModule(
    DWARFDie CUDie, StringRef Filename, StringRef ModuleName, uint64_t DwoId,
    const DWARFFile &File, OffsetsStringPool &StringPool,
    DeclContextTree &ODRContexts, uint64_t ModulesEndOffset, unsigned &UnitID,
    bool IsLittleEndian, unsigned Indent, bool Quiet) {
  /// Using a SmallString<0> because loadClangModule() is recursive.
  SmallString<0> Path(Options.PrependPath);
  if (sys::path::is_relative(Filename))
    resolveRelativeObjectPath(Path, CUDie);
  sys::path::append(Path, Filename);
  // Don't use the cached binary holder because we have no thread-safety
  // guarantee and the lifetime is limited.

  if (Options.ObjFileLoader == nullptr)
    return Error::success();

  auto ErrOrObj = Options.ObjFileLoader(File.FileName, Path);
  if (!ErrOrObj)
    return Error::success();

  std::unique_ptr<CompileUnit> Unit;

  for (const auto &CU : ErrOrObj->Dwarf->compile_units()) {
    updateDwarfVersion(CU->getVersion());
    // Recursively get all modules imported by this one.
    auto CUDie = CU->getUnitDIE(false);
    if (!CUDie)
      continue;
    if (!registerModuleReference(CUDie, *CU, File, StringPool, ODRContexts,
                                 ModulesEndOffset, UnitID, IsLittleEndian,
                                 Indent, Quiet)) {
      if (Unit) {
        std::string Err =
            (Filename +
             ": Clang modules are expected to have exactly 1 compile unit.\n")
                .str();
        reportError(Err, File);
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
              File);
        // Update the cache entry with the DwoId of the module loaded from disk.
        ClangModules[Filename] = PCMDwoId;
      }

      // Add this module.
      Unit = std::make_unique<CompileUnit>(*CU, UnitID++, !Options.NoODR,
                                           ModuleName);
      Unit->setHasInterestingContent();
      analyzeContextInfo(CUDie, 0, *Unit, &ODRContexts.getRoot(), ODRContexts,
                         ModulesEndOffset, Options.ParseableSwiftInterfaces,
                         [&](const Twine &Warning, const DWARFDie &DIE) {
                           reportWarning(Warning, File, &DIE);
                         });
      // Keep everything.
      Unit->markEverythingAsKept();
    }
  }
  assert(Unit && "CompileUnit is not set!");
  if (!Unit->getOrigUnit().getUnitDIE().hasChildren())
    return Error::success();
  if (!Quiet && Options.Verbose) {
    outs().indent(Indent);
    outs() << "cloning .debug_info from " << Filename << "\n";
  }

  UnitListTy CompileUnits;
  CompileUnits.push_back(std::move(Unit));
  assert(TheDwarfEmitter);
  DIECloner(*this, TheDwarfEmitter, *ErrOrObj, DIEAlloc, CompileUnits,
            Options.Update)
      .cloneAllCompileUnits(*(ErrOrObj->Dwarf), File, StringPool,
                            IsLittleEndian);
  return Error::success();
}

uint64_t DWARFLinker::DIECloner::cloneAllCompileUnits(
    DWARFContext &DwarfContext, const DWARFFile &File,
    OffsetsStringPool &StringPool, bool IsLittleEndian) {
  uint64_t OutputDebugInfoSize =
      Linker.Options.NoOutput ? 0 : Emitter->getDebugInfoSectionSize();
  const uint64_t StartOutputDebugInfoSize = OutputDebugInfoSize;

  for (auto &CurrentUnit : CompileUnits) {
    const uint16_t DwarfVersion = CurrentUnit->getOrigUnit().getVersion();
    const uint32_t UnitHeaderSize = DwarfVersion >= 5 ? 12 : 11;
    auto InputDIE = CurrentUnit->getOrigUnit().getUnitDIE();
    CurrentUnit->setStartOffset(OutputDebugInfoSize);
    if (!InputDIE) {
      OutputDebugInfoSize = CurrentUnit->computeNextUnitOffset(DwarfVersion);
      continue;
    }
    if (CurrentUnit->getInfo(0).Keep) {
      // Clone the InputDIE into your Unit DIE in our compile unit since it
      // already has a DIE inside of it.
      CurrentUnit->createOutputDIE();
      cloneDIE(InputDIE, File, *CurrentUnit, StringPool, 0 /* PC offset */,
               UnitHeaderSize, 0, IsLittleEndian,
               CurrentUnit->getOutputUnitDIE());
    }

    OutputDebugInfoSize = CurrentUnit->computeNextUnitOffset(DwarfVersion);

    if (!Linker.Options.NoOutput) {
      assert(Emitter);

      if (LLVM_LIKELY(!Linker.Options.Update) ||
          Linker.needToTranslateStrings())
        Linker.patchLineTableForUnit(*CurrentUnit, DwarfContext, File);

      Linker.emitAcceleratorEntriesForUnit(*CurrentUnit);

      if (LLVM_UNLIKELY(Linker.Options.Update))
        continue;

      Linker.patchRangesForUnit(*CurrentUnit, DwarfContext, File);
      auto ProcessExpr = [&](StringRef Bytes,
                             SmallVectorImpl<uint8_t> &Buffer) {
        DWARFUnit &OrigUnit = CurrentUnit->getOrigUnit();
        DataExtractor Data(Bytes, IsLittleEndian,
                           OrigUnit.getAddressByteSize());
        cloneExpression(Data,
                        DWARFExpression(Data, OrigUnit.getAddressByteSize(),
                                        OrigUnit.getFormParams().Format),
                        File, *CurrentUnit, Buffer);
      };
      Emitter->emitLocationsForUnit(*CurrentUnit, DwarfContext, ProcessExpr);
    }
  }

  if (!Linker.Options.NoOutput) {
    assert(Emitter);
    // Emit all the compile unit's debug information.
    for (auto &CurrentUnit : CompileUnits) {
      if (LLVM_LIKELY(!Linker.Options.Update))
        Linker.generateUnitRanges(*CurrentUnit);

      CurrentUnit->fixupForwardReferences();

      if (!CurrentUnit->getOutputUnitDIE())
        continue;

      unsigned DwarfVersion = CurrentUnit->getOrigUnit().getVersion();

      assert(Emitter->getDebugInfoSectionSize() ==
             CurrentUnit->getStartOffset());
      Emitter->emitCompileUnitHeader(*CurrentUnit, DwarfVersion);
      Emitter->emitDIE(*CurrentUnit->getOutputUnitDIE());
      assert(Emitter->getDebugInfoSectionSize() ==
             CurrentUnit->computeNextUnitOffset(DwarfVersion));
    }
  }

  return OutputDebugInfoSize - StartOutputDebugInfoSize;
}

void DWARFLinker::updateAccelKind(DWARFContext &Dwarf) {
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

  if (!AtLeastOneDwarfAccelTable && !DwarfObj.getNamesSection().Data.empty()) {
    AtLeastOneDwarfAccelTable = true;
  }
}

bool DWARFLinker::emitPaperTrailWarnings(const DWARFFile &File,
                                         OffsetsStringPool &StringPool) {

  if (File.Warnings.empty())
    return false;

  DIE *CUDie = DIE::get(DIEAlloc, dwarf::DW_TAG_compile_unit);
  CUDie->setOffset(11);
  StringRef Producer;
  StringRef WarningHeader;

  switch (DwarfLinkerClientID) {
  case DwarfLinkerClient::Dsymutil:
    Producer = StringPool.internString("dsymutil");
    WarningHeader = "dsymutil_warning";
    break;

  default:
    Producer = StringPool.internString("dwarfopt");
    WarningHeader = "dwarfopt_warning";
    break;
  }

  StringRef FileName = StringPool.internString(File.FileName);
  CUDie->addValue(DIEAlloc, dwarf::DW_AT_producer, dwarf::DW_FORM_strp,
                  DIEInteger(StringPool.getStringOffset(Producer)));
  DIEBlock *String = new (DIEAlloc) DIEBlock();
  DIEBlocks.push_back(String);
  for (auto &C : FileName)
    String->addValue(DIEAlloc, dwarf::Attribute(0), dwarf::DW_FORM_data1,
                     DIEInteger(C));
  String->addValue(DIEAlloc, dwarf::Attribute(0), dwarf::DW_FORM_data1,
                   DIEInteger(0));

  CUDie->addValue(DIEAlloc, dwarf::DW_AT_name, dwarf::DW_FORM_string, String);
  for (const auto &Warning : File.Warnings) {
    DIE &ConstDie = CUDie->addChild(DIE::get(DIEAlloc, dwarf::DW_TAG_constant));
    ConstDie.addValue(DIEAlloc, dwarf::DW_AT_name, dwarf::DW_FORM_strp,
                      DIEInteger(StringPool.getStringOffset(WarningHeader)));
    ConstDie.addValue(DIEAlloc, dwarf::DW_AT_artificial, dwarf::DW_FORM_flag,
                      DIEInteger(1));
    ConstDie.addValue(DIEAlloc, dwarf::DW_AT_const_value, dwarf::DW_FORM_strp,
                      DIEInteger(StringPool.getStringOffset(Warning)));
  }
  unsigned Size = 4 /* FORM_strp */ + FileName.size() + 1 +
                  File.Warnings.size() * (4 + 1 + 4) + 1 /* End of children */;
  DIEAbbrev Abbrev = CUDie->generateAbbrev();
  assignAbbrev(Abbrev);
  CUDie->setAbbrevNumber(Abbrev.getNumber());
  Size += getULEB128Size(Abbrev.getNumber());
  // Abbreviation ordering needed for classic compatibility.
  for (auto &Child : CUDie->children()) {
    Abbrev = Child.generateAbbrev();
    assignAbbrev(Abbrev);
    Child.setAbbrevNumber(Abbrev.getNumber());
    Size += getULEB128Size(Abbrev.getNumber());
  }
  CUDie->setSize(Size);
  TheDwarfEmitter->emitPaperTrailWarningsDie(*CUDie);

  return true;
}

void DWARFLinker::copyInvariantDebugSection(DWARFContext &Dwarf) {
  if (!needToTranslateStrings())
    TheDwarfEmitter->emitSectionContents(
        Dwarf.getDWARFObj().getLineSection().Data, "debug_line");
  TheDwarfEmitter->emitSectionContents(Dwarf.getDWARFObj().getLocSection().Data,
                                       "debug_loc");
  TheDwarfEmitter->emitSectionContents(
      Dwarf.getDWARFObj().getRangesSection().Data, "debug_ranges");
  TheDwarfEmitter->emitSectionContents(
      Dwarf.getDWARFObj().getFrameSection().Data, "debug_frame");
  TheDwarfEmitter->emitSectionContents(Dwarf.getDWARFObj().getArangesSection(),
                                       "debug_aranges");
}

void DWARFLinker::addObjectFile(DWARFFile &File) {
  ObjectContexts.emplace_back(LinkContext(File));

  if (ObjectContexts.back().File.Dwarf)
    updateAccelKind(*ObjectContexts.back().File.Dwarf);
}

bool DWARFLinker::link() {
  assert(Options.NoOutput || TheDwarfEmitter);

  // A unique ID that identifies each compile unit.
  unsigned UnitID = 0;

  // First populate the data structure we need for each iteration of the
  // parallel loop.
  unsigned NumObjects = ObjectContexts.size();

  // This Dwarf string pool which is used for emission. It must be used
  // serially as the order of calling getStringOffset matters for
  // reproducibility.
  OffsetsStringPool OffsetsStringPool(StringsTranslator, true);

  // ODR Contexts for the optimize.
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

  for (LinkContext &OptContext : ObjectContexts) {
    if (Options.Verbose) {
      if (DwarfLinkerClientID == DwarfLinkerClient::Dsymutil)
        outs() << "DEBUG MAP OBJECT: " << OptContext.File.FileName << "\n";
      else
        outs() << "OBJECT FILE: " << OptContext.File.FileName << "\n";
    }

    if (emitPaperTrailWarnings(OptContext.File, OffsetsStringPool))
      continue;

    if (!OptContext.File.Dwarf)
      continue;
    // Look for relocations that correspond to address map entries.

    // there was findvalidrelocations previously ... probably we need to gather
    // info here
    if (LLVM_LIKELY(!Options.Update) &&
        !OptContext.File.Addresses->hasValidRelocs()) {
      if (Options.Verbose)
        outs() << "No valid relocations found. Skipping.\n";

      // Set "Skip" flag as a signal to other loops that we should not
      // process this iteration.
      OptContext.Skip = true;
      continue;
    }

    // Setup access to the debug info.
    if (!OptContext.File.Dwarf)
      continue;

    // In a first phase, just read in the debug info and load all clang modules.
    OptContext.CompileUnits.reserve(
        OptContext.File.Dwarf->getNumCompileUnits());

    for (const auto &CU : OptContext.File.Dwarf->compile_units()) {
      updateDwarfVersion(CU->getVersion());
      auto CUDie = CU->getUnitDIE(false);
      if (Options.Verbose) {
        outs() << "Input compilation unit:";
        DIDumpOptions DumpOpts;
        DumpOpts.ChildRecurseDepth = 0;
        DumpOpts.Verbose = Options.Verbose;
        CUDie.dump(outs(), 0, DumpOpts);
      }
      if (CUDie && !LLVM_UNLIKELY(Options.Update))
        registerModuleReference(CUDie, *CU, OptContext.File, OffsetsStringPool,
                                ODRContexts, 0, UnitID,
                                OptContext.File.Dwarf->isLittleEndian());
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
  const uint64_t ModulesEndOffset =
      Options.NoOutput ? 0 : TheDwarfEmitter->getDebugInfoSectionSize();

  // These variables manage the list of processed object files.
  // The mutex and condition variable are to ensure that this is thread safe.
  std::mutex ProcessedFilesMutex;
  std::condition_variable ProcessedFilesConditionVariable;
  BitVector ProcessedFiles(NumObjects, false);

  //  Analyzing the context info is particularly expensive so it is executed in
  //  parallel with emitting the previous compile unit.
  auto AnalyzeLambda = [&](size_t I) {
    auto &Context = ObjectContexts[I];

    if (Context.Skip || !Context.File.Dwarf)
      return;

    for (const auto &CU : Context.File.Dwarf->compile_units()) {
      updateDwarfVersion(CU->getVersion());
      // The !registerModuleReference() condition effectively skips
      // over fully resolved skeleton units. This second pass of
      // registerModuleReferences doesn't do any new work, but it
      // will collect top-level errors, which are suppressed. Module
      // warnings were already displayed in the first iteration.
      bool Quiet = true;
      auto CUDie = CU->getUnitDIE(false);
      if (!CUDie || LLVM_UNLIKELY(Options.Update) ||
          !registerModuleReference(CUDie, *CU, Context.File, OffsetsStringPool,
                                   ODRContexts, ModulesEndOffset, UnitID,
                                   Quiet)) {
        Context.CompileUnits.push_back(std::make_unique<CompileUnit>(
            *CU, UnitID++, !Options.NoODR && !Options.Update, ""));
      }
    }

    // Now build the DIE parent links that we will use during the next phase.
    for (auto &CurrentUnit : Context.CompileUnits) {
      auto CUDie = CurrentUnit->getOrigUnit().getUnitDIE();
      if (!CUDie)
        continue;
      analyzeContextInfo(CurrentUnit->getOrigUnit().getUnitDIE(), 0,
                         *CurrentUnit, &ODRContexts.getRoot(), ODRContexts,
                         ModulesEndOffset, Options.ParseableSwiftInterfaces,
                         [&](const Twine &Warning, const DWARFDie &DIE) {
                           reportWarning(Warning, Context.File, &DIE);
                         });
    }
  };

  // For each object file map how many bytes were emitted.
  StringMap<DebugInfoSize> SizeByObject;

  // And then the remaining work in serial again.
  // Note, although this loop runs in serial, it can run in parallel with
  // the analyzeContextInfo loop so long as we process files with indices >=
  // than those processed by analyzeContextInfo.
  auto CloneLambda = [&](size_t I) {
    auto &OptContext = ObjectContexts[I];
    if (OptContext.Skip || !OptContext.File.Dwarf)
      return;

    // Then mark all the DIEs that need to be present in the generated output
    // and collect some information about them.
    // Note that this loop can not be merged with the previous one because
    // cross-cu references require the ParentIdx to be setup for every CU in
    // the object file before calling this.
    if (LLVM_UNLIKELY(Options.Update)) {
      for (auto &CurrentUnit : OptContext.CompileUnits)
        CurrentUnit->markEverythingAsKept();
      copyInvariantDebugSection(*OptContext.File.Dwarf);
    } else {
      for (auto &CurrentUnit : OptContext.CompileUnits)
        lookForDIEsToKeep(*OptContext.File.Addresses,
                          OptContext.File.Addresses->getValidAddressRanges(),
                          OptContext.CompileUnits,
                          CurrentUnit->getOrigUnit().getUnitDIE(),
                          OptContext.File, *CurrentUnit, 0);
    }

    // The calls to applyValidRelocs inside cloneDIE will walk the reloc
    // array again (in the same way findValidRelocsInDebugInfo() did). We
    // need to reset the NextValidReloc index to the beginning.
    if (OptContext.File.Addresses->hasValidRelocs() ||
        LLVM_UNLIKELY(Options.Update)) {
      SizeByObject[OptContext.File.FileName].Input =
          getDebugInfoSize(*OptContext.File.Dwarf);
      SizeByObject[OptContext.File.FileName].Output =
          DIECloner(*this, TheDwarfEmitter, OptContext.File, DIEAlloc,
                    OptContext.CompileUnits, Options.Update)
              .cloneAllCompileUnits(*OptContext.File.Dwarf, OptContext.File,
                                    OffsetsStringPool,
                                    OptContext.File.Dwarf->isLittleEndian());
    }
    if (!Options.NoOutput && !OptContext.CompileUnits.empty() &&
        LLVM_LIKELY(!Options.Update))
      patchFrameInfoForObject(
          OptContext.File, OptContext.File.Addresses->getValidAddressRanges(),
          *OptContext.File.Dwarf,
          OptContext.CompileUnits[0]->getOrigUnit().getAddressByteSize());

    // Clean-up before starting working on the next object.
    cleanupAuxiliarryData(OptContext);
  };

  auto EmitLambda = [&]() {
    // Emit everything that's global.
    if (!Options.NoOutput) {
      TheDwarfEmitter->emitAbbrevs(Abbreviations, MaxDwarfVersion);
      TheDwarfEmitter->emitStrings(OffsetsStringPool);
      switch (Options.TheAccelTableKind) {
      case AccelTableKind::Apple:
        TheDwarfEmitter->emitAppleNames(AppleNames);
        TheDwarfEmitter->emitAppleNamespaces(AppleNamespaces);
        TheDwarfEmitter->emitAppleTypes(AppleTypes);
        TheDwarfEmitter->emitAppleObjc(AppleObjc);
        break;
      case AccelTableKind::Dwarf:
        TheDwarfEmitter->emitDebugNames(DebugNames);
        break;
      case AccelTableKind::Default:
        llvm_unreachable("Default should have already been resolved.");
        break;
      }
    }
  };

  auto AnalyzeAll = [&]() {
    for (unsigned I = 0, E = NumObjects; I != E; ++I) {
      AnalyzeLambda(I);

      std::unique_lock<std::mutex> LockGuard(ProcessedFilesMutex);
      ProcessedFiles.set(I);
      ProcessedFilesConditionVariable.notify_one();
    }
  };

  auto CloneAll = [&]() {
    for (unsigned I = 0, E = NumObjects; I != E; ++I) {
      {
        std::unique_lock<std::mutex> LockGuard(ProcessedFilesMutex);
        if (!ProcessedFiles[I]) {
          ProcessedFilesConditionVariable.wait(
              LockGuard, [&]() { return ProcessedFiles[I]; });
        }
      }

      CloneLambda(I);
    }
    EmitLambda();
  };

  // To limit memory usage in the single threaded case, analyze and clone are
  // run sequentially so the OptContext is freed after processing each object
  // in endDebugObject.
  if (Options.Threads == 1) {
    for (unsigned I = 0, E = NumObjects; I != E; ++I) {
      AnalyzeLambda(I);
      CloneLambda(I);
    }
    EmitLambda();
  } else {
    ThreadPool Pool(hardware_concurrency(2));
    Pool.async(AnalyzeAll);
    Pool.async(CloneAll);
    Pool.wait();
  }

  if (Options.Statistics) {
    // Create a vector sorted in descending order by output size.
    std::vector<std::pair<StringRef, DebugInfoSize>> Sorted;
    for (auto &E : SizeByObject)
      Sorted.emplace_back(E.first(), E.second);
    llvm::sort(Sorted, [](auto &LHS, auto &RHS) {
      return LHS.second.Output > RHS.second.Output;
    });

    auto ComputePercentange = [](int64_t Input, int64_t Output) -> float {
      const float Difference = Output - Input;
      const float Sum = Input + Output;
      if (Sum == 0)
        return 0;
      return (Difference / (Sum / 2));
    };

    int64_t InputTotal = 0;
    int64_t OutputTotal = 0;
    const char *FormatStr = "{0,-45} {1,10}b  {2,10}b {3,8:P}\n";

    // Print header.
    outs() << ".debug_info section size (in bytes)\n";
    outs() << "----------------------------------------------------------------"
              "---------------\n";
    outs() << "Filename                                           Object       "
              "  dSYM   Change\n";
    outs() << "----------------------------------------------------------------"
              "---------------\n";

    // Print body.
    for (auto &E : Sorted) {
      InputTotal += E.second.Input;
      OutputTotal += E.second.Output;
      llvm::outs() << formatv(
          FormatStr, sys::path::filename(E.first).take_back(45), E.second.Input,
          E.second.Output, ComputePercentange(E.second.Input, E.second.Output));
    }
    // Print total and footer.
    outs() << "----------------------------------------------------------------"
              "---------------\n";
    llvm::outs() << formatv(FormatStr, "Total", InputTotal, OutputTotal,
                            ComputePercentange(InputTotal, OutputTotal));
    outs() << "----------------------------------------------------------------"
              "---------------\n\n";
  }

  return true;
}

} // namespace llvm
