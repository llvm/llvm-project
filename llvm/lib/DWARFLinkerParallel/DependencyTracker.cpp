//=== DependencyTracker.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DependencyTracker.h"
#include "llvm/Support/FormatVariadic.h"

namespace llvm {
namespace dwarflinker_parallel {

#ifndef NDEBUG
/// A broken link in the keep chain. By recording both the parent and the child
/// we can show only broken links for DIEs with multiple children.
struct BrokenLink {
  BrokenLink(DWARFDie Parent, DWARFDie Child) : Parent(Parent), Child(Child) {}
  DWARFDie Parent;
  DWARFDie Child;
};

/// Verify the keep chain by looking for DIEs that are kept but who's parent
/// isn't.
void DependencyTracker::verifyKeepChain(CompileUnit &CU) {
  SmallVector<DWARFDie> Worklist;
  Worklist.push_back(CU.getOrigUnit().getUnitDIE());

  // List of broken links.
  SmallVector<BrokenLink> BrokenLinks;

  while (!Worklist.empty()) {
    const DWARFDie Current = Worklist.back();
    Worklist.pop_back();

    if (!Current.isValid())
      continue;

    const bool CurrentDieIsKept = CU.getDIEInfo(Current).getKeep() ||
                                  CU.getDIEInfo(Current).getKeepChildren();

    for (DWARFDie Child : reverse(Current.children())) {
      Worklist.push_back(Child);

      const bool ChildDieIsKept = CU.getDIEInfo(Child).getKeep() ||
                                  CU.getDIEInfo(Child).getKeepChildren();
      if (!CurrentDieIsKept && ChildDieIsKept)
        BrokenLinks.emplace_back(Current, Child);
    }
  }

  if (!BrokenLinks.empty()) {
    for (BrokenLink Link : BrokenLinks) {
      WithColor::error() << formatv(
          "Found invalid link in keep chain between {0:x} and {1:x}\n",
          Link.Parent.getOffset(), Link.Child.getOffset());

      errs() << "Parent:";
      Link.Parent.dump(errs(), 0, {});
      CU.getDIEInfo(Link.Parent).dump();

      errs() << "Child:";
      Link.Child.dump(errs(), 2, {});
      CU.getDIEInfo(Link.Child).dump();
    }
    report_fatal_error("invalid keep chain");
  }
}
#endif

bool DependencyTracker::resolveDependenciesAndMarkLiveness(CompileUnit &CU) {
  // We do not track liveness inside Clang modules. We also do not track
  // liveness if UpdateIndexTablesOnly is requested.
  TrackLiveness = !(CU.isClangModule() ||
                    CU.getGlobalData().getOptions().UpdateIndexTablesOnly);
  RootEntriesWorkList.clear();

  // Search for live root DIEs.
  collectRootsToKeep(CU, CU.getDebugInfoEntry(0));

  // Mark live DIEs as kept.
  return markLiveRootsAsKept();
}

void DependencyTracker::collectRootsToKeep(CompileUnit &CU,
                                           const DWARFDebugInfoEntry *Entry) {
  if (!TrackLiveness) {
    addItemToWorklist(CU, Entry);
    return;
  }

  switch (Entry->getTag()) {
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_label:
    if (isLiveSubprogramEntry(CU, Entry)) {
      addItemToWorklist(CU, Entry);
      break;
    }
    [[fallthrough]];
  case dwarf::DW_TAG_compile_unit:
  case dwarf::DW_TAG_namespace:
  case dwarf::DW_TAG_module:
  case dwarf::DW_TAG_lexical_block: {
    for (const DWARFDebugInfoEntry *CurChild = CU.getFirstChildEntry(Entry);
         CurChild && CurChild->getAbbreviationDeclarationPtr();
         CurChild = CU.getSiblingEntry(CurChild))
      collectRootsToKeep(CU, CurChild);
  } break;
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_variable: {
    if (isLiveVariableEntry(CU, Entry))
      addItemToWorklist(CU, Entry);
  } break;
  case dwarf::DW_TAG_base_type: {
    addItemToWorklist(CU, Entry);
  } break;
  case dwarf::DW_TAG_imported_module:
  case dwarf::DW_TAG_imported_declaration:
  case dwarf::DW_TAG_imported_unit: {
    addItemToWorklist(CU, Entry);
  } break;
  default:
    // Nothing to do.
    break;
  }
}

bool DependencyTracker::markLiveRootsAsKept() {
  bool Res = true;

  while (!RootEntriesWorkList.empty()) {
    RootEntryTy CurrentItem = RootEntriesWorkList.pop_back_val();

    if (!markDIEEntryAsKeptRec(CurrentItem, CurrentItem.CU,
                               CurrentItem.RootEntry))
      Res = false;
  }

  return Res;
}

bool DependencyTracker::markDIEEntryAsKeptRec(
    const RootEntryTy &RootItem, CompileUnit &CU,
    const DWARFDebugInfoEntry *Entry) {
  if (Entry->getAbbreviationDeclarationPtr() == nullptr)
    return true;

  CompileUnit::DIEInfo &Info = CU.getDIEInfo(Entry);

  if (Info.getKeep())
    return true;

  // Mark parents as 'KeepChildren'.
  std::optional<uint32_t> ParentIdx = Entry->getParentIdx();
  while (ParentIdx) {
    const DWARFDebugInfoEntry *ParentEntry = CU.getDebugInfoEntry(*ParentIdx);
    CompileUnit::DIEInfo &ParentInfo = CU.getDIEInfo(*ParentIdx);
    if (ParentInfo.getKeepChildren())
      break;
    ParentInfo.setKeepChildren();
    ParentIdx = ParentEntry->getParentIdx();
  }

  // Mark current DIE as kept.
  Info.setKeep();
  setDIEPlacementAndTypename(Info);

  // Set liveness information.
  switch (Entry->getTag()) {
  case dwarf::DW_TAG_constant:
  case dwarf::DW_TAG_variable: {
    isLiveVariableEntry(CU, Entry);
  } break;
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_label: {
    isLiveSubprogramEntry(CU, Entry);
  } break;
  default:
    // Nothing to do.
    break;
  }

  // Analyse referenced DIEs.
  bool Res = true;
  if (!maybeAddReferencedRoots(RootItem, CU, Entry))
    Res = false;

  // Navigate children.
  for (const DWARFDebugInfoEntry *CurChild = CU.getFirstChildEntry(Entry);
       CurChild && CurChild->getAbbreviationDeclarationPtr();
       CurChild = CU.getSiblingEntry(CurChild)) {
    if (!markDIEEntryAsKeptRec(RootItem, CU, CurChild))
      Res = false;
  }

  return Res;
}

bool DependencyTracker::maybeAddReferencedRoots(
    const RootEntryTy &RootItem, CompileUnit &CU,
    const DWARFDebugInfoEntry *Entry) {
  const auto *Abbrev = Entry->getAbbreviationDeclarationPtr();
  if (Abbrev == nullptr)
    return true;

  DWARFUnit &Unit = CU.getOrigUnit();
  DWARFDataExtractor Data = Unit.getDebugInfoExtractor();
  uint64_t Offset = Entry->getOffset() + getULEB128Size(Abbrev->getCode());

  // For each DIE attribute...
  for (const auto &AttrSpec : Abbrev->attributes()) {
    DWARFFormValue Val(AttrSpec.Form);
    if (!Val.isFormClass(DWARFFormValue::FC_Reference) ||
        AttrSpec.Attr == dwarf::DW_AT_sibling) {
      DWARFFormValue::skipValue(AttrSpec.Form, Data, &Offset,
                                Unit.getFormParams());
      continue;
    }
    Val.extractValue(Data, &Offset, Unit.getFormParams(), &Unit);

    // Resolve reference.
    std::optional<std::pair<CompileUnit *, uint32_t>> RefDie =
        CU.resolveDIEReference(Val);
    if (!RefDie) {
      CU.warn("cann't find referenced DIE", Entry);
      continue;
    }

    if (CU.getUniqueID() == RefDie->first->getUniqueID()) {
      // Check if referenced DIE entry is already kept.
      if (RefDie->first->getDIEInfo(RefDie->second).getKeep())
        continue;

      // If referenced DIE is inside current compilation unit.
      const DWARFDebugInfoEntry *RefEntry =
          RefDie->first->getDebugInfoEntry(RefDie->second);

      if (RootItem.RootEntry->getTag() == dwarf::DW_TAG_compile_unit)
        addItemToWorklist(*RefDie->first, RefEntry);
      else {
        uint64_t RootStartOffset = RootItem.RootEntry->getOffset();
        uint64_t RootEndOffset;
        if (std::optional<uint32_t> SiblingIdx =
                RootItem.RootEntry->getSiblingIdx()) {
          RootEndOffset =
              RootItem.CU.getDebugInfoEntry(*SiblingIdx)->getOffset();
        } else {
          RootEndOffset = RootItem.CU.getOrigUnit().getNextUnitOffset();
        }

        // Do not put item in work list if it is an ancestor of RootItem.
        // (since we will visit and mark it as kept during normal traversing of
        // RootItem children)
        if (RootStartOffset > RefEntry->getOffset() ||
            RefEntry->getOffset() >= RootEndOffset)
          addItemToWorklist(*RefDie->first, RefEntry);
      }
    } else if (Context.InterCUProcessingStarted && RefDie->second != 0) {
      // If referenced DIE is in other compilation unit and
      // it is safe to navigate other units DIEs.
      addItemToWorklist(*RefDie->first,
                        RefDie->first->getDebugInfoEntry(RefDie->second));
    } else {
      // Delay resolving reference.
      RefDie->first->setInterconnectedCU();
      CU.setInterconnectedCU();
      Context.HasNewInterconnectedCUs = true;
      return false;
    }
  }

  return true;
}

// Returns true if the specified DIE type allows removing children.
static bool childrenCanBeRemoved(uint32_t Tag) {
  switch (Tag) {
  default:
    return true;
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_common_block:
  case dwarf::DW_TAG_lexical_block:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_subprogram:
  case dwarf::DW_TAG_subroutine_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_array_type:
    return false;
  }
  llvm_unreachable("Invalid Tag");
}

void DependencyTracker::addItemToWorklist(CompileUnit &CU,
                                          const DWARFDebugInfoEntry *Entry) {
  if (Entry->getAbbreviationDeclarationPtr() == nullptr)
    return;

  const DWARFDebugInfoEntry *EntryToAdd = Entry;

  // If parent does not allow children removing then use that parent as a root
  // DIE.
  std::optional<uint32_t> ParentIdx = Entry->getParentIdx();
  while (ParentIdx) {
    const DWARFDebugInfoEntry *ParentEntry = CU.getDebugInfoEntry(*ParentIdx);
    if (childrenCanBeRemoved(ParentEntry->getTag()))
      break;
    EntryToAdd = ParentEntry;
    ParentIdx = ParentEntry->getParentIdx();
  }

  // Check if the DIE entry is already kept.
  if (CU.getDIEInfo(EntryToAdd).getKeep())
    return;

  RootEntriesWorkList.emplace_back(CU, EntryToAdd);
}

bool DependencyTracker::isLiveVariableEntry(CompileUnit &CU,
                                            const DWARFDebugInfoEntry *Entry) {
  DWARFDie DIE = CU.getDIE(Entry);
  CompileUnit::DIEInfo &Info = CU.getDIEInfo(DIE);

  if (TrackLiveness) {
    const auto *Abbrev = DIE.getAbbreviationDeclarationPtr();

    // Global variables with constant value can always be kept.
    if (!Info.getIsInFunctionScope() &&
        Abbrev->findAttributeIndex(dwarf::DW_AT_const_value))
      return true;

    // See if there is a relocation to a valid debug map entry inside this
    // variable's location. The order is important here. We want to always check
    // if the variable has a location expression address.
    // However, we don't want a static variable in a function to force us to
    // keep the enclosing function, unless requested explicitly.
    std::pair<bool, std::optional<int64_t>> LocExprAddrAndRelocAdjustment =
        CU.getContaingFile().Addresses->getVariableRelocAdjustment(DIE);

    if (!LocExprAddrAndRelocAdjustment.second)
      return false;

    if ((Info.getIsInFunctionScope()) &&
        !LLVM_UNLIKELY(CU.getGlobalData().getOptions().KeepFunctionForStatic))
      return false;
  }

  if (CU.getGlobalData().getOptions().Verbose) {
    outs() << "Keeping variable DIE:";
    DIDumpOptions DumpOpts;
    DumpOpts.ChildRecurseDepth = 0;
    DumpOpts.Verbose = CU.getGlobalData().getOptions().Verbose;
    DIE.dump(outs(), 8 /* Indent */, DumpOpts);
  }

  return true;
}

bool DependencyTracker::isLiveSubprogramEntry(
    CompileUnit &CU, const DWARFDebugInfoEntry *Entry) {
  DWARFDie DIE = CU.getDIE(Entry);

  std::optional<uint64_t> LowPc;
  std::optional<uint64_t> HighPc;
  std::optional<int64_t> RelocAdjustment;

  if (TrackLiveness) {
    LowPc = dwarf::toAddress(DIE.find(dwarf::DW_AT_low_pc));
    if (!LowPc)
      return false;

    RelocAdjustment =
        CU.getContaingFile().Addresses->getSubprogramRelocAdjustment(DIE);
    if (!RelocAdjustment)
      return false;

    if (DIE.getTag() == dwarf::DW_TAG_subprogram) {
      // Validate subprogram address range.

      HighPc = DIE.getHighPC(*LowPc);
      if (!HighPc) {
        CU.warn("function without high_pc. Range will be discarded.", &DIE);
        return false;
      }

      if (*LowPc > *HighPc) {
        CU.warn("low_pc greater than high_pc. Range will be discarded.", &DIE);
        return false;
      }
    } else if (DIE.getTag() == dwarf::DW_TAG_variable) {
      if (CU.hasLabelAt(*LowPc))
        return false;

      // FIXME: dsymutil-classic compat. dsymutil-classic doesn't consider
      // labels that don't fall into the CU's aranges. This is wrong IMO. Debug
      // info generation bugs aside, this is really wrong in the case of labels,
      // where a label marking the end of a function will have a PC == CU's
      // high_pc.
      if (dwarf::toAddress(
              CU.getOrigUnit().getUnitDIE().find(dwarf::DW_AT_high_pc))
              .value_or(UINT64_MAX) <= LowPc)
        return false;

      CU.addLabelLowPc(*LowPc, *RelocAdjustment);
    }
  }

  if (CU.getGlobalData().getOptions().Verbose) {
    outs() << "Keeping subprogram DIE:";
    DIDumpOptions DumpOpts;
    DumpOpts.ChildRecurseDepth = 0;
    DumpOpts.Verbose = CU.getGlobalData().getOptions().Verbose;
    DIE.dump(outs(), 8 /* Indent */, DumpOpts);
  }

  if (!TrackLiveness || DIE.getTag() == dwarf::DW_TAG_label)
    return true;

  CU.addFunctionRange(*LowPc, *HighPc, *RelocAdjustment);
  return true;
}

void DependencyTracker::setDIEPlacementAndTypename(CompileUnit::DIEInfo &Info) {
  Info.setPlacement(CompileUnit::PlainDwarf);
}

} // end of namespace dwarflinker_parallel
} // namespace llvm
