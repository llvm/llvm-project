//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDiff.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

using namespace llvm;
using namespace llvm::dwarf;

/// Tags for which unnamed instances get a structural identity key (e.g.,
/// an unnamed "const int *" becomes "*Kint"). This lets unnamed type DIEs
/// participate in identity-based matching.
static bool hasStructuralTypeKey(dwarf::Tag Tag) {
  switch (Tag) {
  case DW_TAG_base_type:
  case DW_TAG_class_type:
  case DW_TAG_enumeration_type:
  case DW_TAG_structure_type:
  case DW_TAG_typedef:
  case DW_TAG_union_type:
  case DW_TAG_pointer_type:
  case DW_TAG_reference_type:
  case DW_TAG_rvalue_reference_type:
  case DW_TAG_const_type:
  case DW_TAG_volatile_type:
  case DW_TAG_restrict_type:
  case DW_TAG_array_type:
  case DW_TAG_subroutine_type:
  case DW_TAG_ptr_to_member_type:
  case DW_TAG_atomic_type:
  case DW_TAG_unspecified_type:
    return true;
  default:
    return false;
  }
}

static bool isTypeModifierTag(dwarf::Tag Tag) {
  switch (Tag) {
  case DW_TAG_pointer_type:
  case DW_TAG_reference_type:
  case DW_TAG_rvalue_reference_type:
  case DW_TAG_const_type:
  case DW_TAG_volatile_type:
  case DW_TAG_restrict_type:
  case DW_TAG_atomic_type:
    return true;
  default:
    return false;
  }
}

/// Attributes that encode file-level structure rather than semantic content.
/// These always differ between files and are never compared.
static bool isStructuralAttribute(dwarf::Attribute Attr) {
  switch (Attr) {
  case DW_AT_comp_dir:
  case DW_AT_producer:
  case DW_AT_language:
  case DW_AT_stmt_list:
  case DW_AT_low_pc:
  case DW_AT_high_pc:
  case DW_AT_ranges:
  case DW_AT_addr_base:
  case DW_AT_str_offsets_base:
  case DW_AT_rnglists_base:
  case DW_AT_loclists_base:
  case DW_AT_GNU_addr_base:
  case DW_AT_GNU_ranges_base:
  case DW_AT_macro_info:
  case DW_AT_macros:
  case DW_AT_GNU_pubnames:
  case DW_AT_sibling:
  case DW_AT_object_pointer:
  case DW_AT_location:
  case DW_AT_frame_base:
  case DW_AT_segment:
  case DW_AT_static_link:
  case DW_AT_use_location:
  case DW_AT_vtable_elem_location:
  case DW_AT_data_location:
  case DW_AT_call_value:
  case DW_AT_GNU_call_site_value:
  case DW_AT_call_origin:
  case DW_AT_call_return_pc:
  case DW_AT_call_pc:
  case DW_AT_call_all_calls:
  case DW_AT_call_all_source_calls:
  case DW_AT_call_all_tail_calls:
    return true;
  default:
    return false;
  }
}

/// References that should be compared by following the type graph.
static bool isTypeReferenceAttribute(dwarf::Attribute Attr) {
  switch (Attr) {
  case DW_AT_type:
  case DW_AT_containing_type:
  case DW_AT_friend:
  case DW_AT_default_value:
  case DW_AT_signature:
    return true;
  default:
    return false;
  }
}

/// References to declarations/definitions of the same entity. These are
/// compared by identity (qualified name) rather than structural comparison,
/// because the referent may have been reorganized across CUs.
static bool isIdentityReferenceAttribute(dwarf::Attribute Attr) {
  switch (Attr) {
  case DW_AT_specification:
  case DW_AT_abstract_origin:
  case DW_AT_import:
    return true;
  default:
    return false;
  }
}

static std::string buildQualifiedName(StringRef Name, DWARFDie Die) {
  std::string Result;
  SmallVector<StringRef, 8> Parts;
  Parts.push_back(Name);

  DWARFDie Parent = Die.getParent();
  while (Parent.isValid()) {
    dwarf::Tag PTag = Parent.getTag();
    if (PTag == DW_TAG_compile_unit || PTag == DW_TAG_type_unit ||
        PTag == DW_TAG_partial_unit)
      break;
    const char *PName = Parent.getName(DINameKind::ShortName);
    if (PName && PName[0])
      Parts.push_back(PName);
    else if (PTag == DW_TAG_namespace)
      Parts.push_back("(anonymous namespace)");
    Parent = Parent.getParent();
  }

  for (auto I = Parts.rbegin(), E = Parts.rend(); I != E; ++I) {
    if (!Result.empty())
      Result += "::";
    Result += *I;
  }
  return Result;
}

static void sortPreferDefinitions(SmallVectorImpl<DWARFDie> &Dies) {
  llvm::stable_sort(Dies, [](DWARFDie A, DWARFDie B) {
    bool AIsDecl = A.find(DW_AT_declaration).has_value();
    bool BIsDecl = B.find(DW_AT_declaration).has_value();
    return !AIsDecl && BIsDecl;
  });
}

static bool compareFormValues(const DWARFFormValue &LHS,
                              const DWARFFormValue &RHS) {
  // The else-after-return is needed: LStr is an Expected<> that must be
  // consumed in both success and error paths before falling through.
  if (auto LStr = LHS.getAsCString()) {
    auto RStr = RHS.getAsCString();
    if (!RStr) {
      consumeError(RStr.takeError());
      return false;
    }
    return StringRef(*LStr) == StringRef(*RStr);
  } else {
    consumeError(LStr.takeError());
  }

  if (auto LV = LHS.getAsUnsignedConstant()) {
    auto RV = RHS.getAsUnsignedConstant();
    return RV && *LV == *RV;
  }

  if (auto LV = LHS.getAsSignedConstant()) {
    auto RV = RHS.getAsSignedConstant();
    return RV && *LV == *RV;
  }

  if (auto LV = LHS.getAsBlock()) {
    auto RV = RHS.getAsBlock();
    return RV && *LV == *RV;
  }

  if (auto LV = LHS.getAsAddress()) {
    auto RV = RHS.getAsAddress();
    return RV && *LV == *RV;
  }

  if (LHS.isFormClass(DWARFFormValue::FC_Flag))
    return LHS.getAsUnsignedConstant() == RHS.getAsUnsignedConstant();

  if (auto LV = LHS.getAsSectionOffset()) {
    auto RV = RHS.getAsSectionOffset();
    return RV && *LV == *RV;
  }

  // Be conservative and treat unrecognized form classes as equal rather than
  // reporting false differences for vendor extensions or future DWARF forms.
  return true;
}

bool DWARFDiff::isSkippedAttribute(dwarf::Attribute Attr) const {
  if (isStructuralAttribute(Attr))
    return true;
  if (Attr == DW_AT_decl_file || Attr == DW_AT_declaration)
    return true;
  if (Opts.IgnoreLines && (Attr == DW_AT_decl_line || Attr == DW_AT_call_line))
    return true;
  return false;
}

std::string DWARFDiff::getTypeKey(DWARFDie Die, DenseSet<uint64_t> &Visited,
                                  unsigned Depth) {
  if (!Die.isValid())
    return "void";
  if (Depth > 128)
    return "?";

  uint64_t Offset = Die.getOffset();
  if (!Visited.insert(Offset).second)
    return "^";

  auto GetInner = [&]() -> std::string {
    DWARFDie Inner = Die.getAttributeValueAsReferencedDie(DW_AT_type);
    return getTypeKey(Inner, Visited, Depth + 1);
  };

  dwarf::Tag Tag = Die.getTag();
  std::string Result;

  switch (Tag) {
  case DW_TAG_pointer_type:
    Result = "*" + GetInner();
    break;
  case DW_TAG_reference_type:
    Result = "&" + GetInner();
    break;
  case DW_TAG_rvalue_reference_type:
    Result = "&&" + GetInner();
    break;
  case DW_TAG_const_type:
    Result = "K" + GetInner();
    break;
  case DW_TAG_volatile_type:
    Result = "V" + GetInner();
    break;
  case DW_TAG_restrict_type:
    Result = "R" + GetInner();
    break;
  case DW_TAG_atomic_type:
    Result = "A" + GetInner();
    break;
  case DW_TAG_array_type:
    Result = "[" + GetInner();
    break;
  case DW_TAG_subroutine_type: {
    Result = "F(";
    for (DWARFDie C : Die.children()) {
      if (C.getTag() == DW_TAG_formal_parameter) {
        DWARFDie PT = C.getAttributeValueAsReferencedDie(DW_AT_type);
        Result += getTypeKey(PT, Visited, Depth + 1) + ",";
      } else if (C.getTag() == DW_TAG_unspecified_parameters) {
        Result += "...,";
      }
    }
    Result += ")" + GetInner();
    break;
  }
  case DW_TAG_ptr_to_member_type: {
    DWARFDie Cont = Die.getAttributeValueAsReferencedDie(DW_AT_containing_type);
    Result = "M<" + getTypeKey(Cont, Visited, Depth + 1) + ">" + GetInner();
    break;
  }
  default: {
    const char *Name = Die.getName(DINameKind::ShortName);
    if (Name && Name[0])
      Result = buildQualifiedName(Name, Die);
    else
      Result = TagString(Tag).str();
    break;
  }
  }

  Visited.erase(Offset);
  return Result;
}

std::string DWARFDiff::getQualifiedName(DWARFDie Die) {
  if (!Die.isValid())
    return "";

  dwarf::Tag Tag = Die.getTag();

  if (Tag == DW_TAG_subprogram || Tag == DW_TAG_inlined_subroutine ||
      Tag == DW_TAG_variable) {
    const char *Name = Die.getLinkageName();
    if (Name && Name[0])
      return Name;
  }

  if (Tag == DW_TAG_inheritance) {
    DWARFDie BaseDie = Die.getAttributeValueAsReferencedDie(DW_AT_type);
    if (BaseDie.isValid())
      return getQualifiedName(BaseDie);
    return "";
  }

  if (Tag == DW_TAG_imported_declaration || Tag == DW_TAG_imported_module) {
    DWARFDie Imported = Die.getAttributeValueAsReferencedDie(DW_AT_import);
    if (Imported.isValid())
      return "import:" + getQualifiedName(Imported);
    return "";
  }

  const char *Name = Die.getName(DINameKind::ShortName);
  if (!Name || !Name[0]) {
    if (hasStructuralTypeKey(Tag)) {
      DenseSet<uint64_t> Visited;
      return getTypeKey(Die, Visited);
    }
    return "";
  }

  return buildQualifiedName(Name, Die);
}

DIEIdentity DWARFDiff::getIdentity(DWARFDie Die) {
  if (!Die.isValid())
    return {};

  dwarf::Tag Tag = Die.getTag();

  switch (Tag) {
  case DW_TAG_compile_unit:
  case DW_TAG_type_unit:
  case DW_TAG_partial_unit: {
    const char *Name = Die.getName(DINameKind::ShortName);
    return {Name ? Name : "", Tag};
  }
  default:
    break;
  }

  std::string QName = getQualifiedName(Die);
  if (QName.empty())
    return {};
  return {std::move(QName), Tag};
}

bool DWARFDiff::compareTypes(DWARFDie LHS, DWARFDie RHS,
                             DenseSet<std::pair<uint64_t, uint64_t>> &Visited) {
  if (!LHS.isValid() && !RHS.isValid())
    return true;
  if (!LHS.isValid() || !RHS.isValid())
    return false;
  dwarf::Tag Tag = LHS.getTag();
  if (Tag != RHS.getTag())
    return false;

  auto Key = std::make_pair(LHS.getOffset(), RHS.getOffset());
  if (!Visited.insert(Key).second)
    return true;

  auto Guard = llvm::scope_exit([&] { Visited.erase(Key); });

  auto CompareInner = [&]() {
    DWARFDie L = LHS.getAttributeValueAsReferencedDie(DW_AT_type);
    DWARFDie R = RHS.getAttributeValueAsReferencedDie(DW_AT_type);
    return compareTypes(L, R, Visited);
  };

  if (isTypeModifierTag(Tag))
    return CompareInner();

  const char *LName = LHS.getName(DINameKind::ShortName);
  const char *RName = RHS.getName(DINameKind::ShortName);
  bool LHasName = LName && LName[0];
  bool RHasName = RName && RName[0];
  if (LHasName != RHasName)
    return false;
  if (LHasName && StringRef(LName) != StringRef(RName))
    return false;

  // A forward declaration matches its definition if names and tags agree.
  bool LIsDecl = LHS.find(DW_AT_declaration).has_value();
  bool RIsDecl = RHS.find(DW_AT_declaration).has_value();
  if (LHasName && (LIsDecl || RIsDecl))
    return true;

  if (Tag == DW_TAG_array_type) {
    if (!CompareInner())
      return false;
    SmallVector<DWARFDie, 2> LSubs, RSubs;
    for (DWARFDie C : LHS.children())
      if (C.getTag() == DW_TAG_subrange_type)
        LSubs.push_back(C);
    for (DWARFDie C : RHS.children())
      if (C.getTag() == DW_TAG_subrange_type)
        RSubs.push_back(C);
    if (LSubs.size() != RSubs.size())
      return false;
    for (size_t I = 0, E = LSubs.size(); I < E; ++I) {
      auto LC = LSubs[I].find(DW_AT_count);
      auto RC = RSubs[I].find(DW_AT_count);
      if (LC.has_value() != RC.has_value())
        return false;
      if (LC && RC &&
          LC->getAsUnsignedConstant() != RC->getAsUnsignedConstant())
        return false;
    }
    return true;
  }

  if (Tag == DW_TAG_subroutine_type) {
    if (!CompareInner())
      return false;
    SmallVector<DWARFDie, 4> LP, RP;
    for (DWARFDie C : LHS.children())
      if (C.getTag() == DW_TAG_formal_parameter ||
          C.getTag() == DW_TAG_unspecified_parameters)
        LP.push_back(C);
    for (DWARFDie C : RHS.children())
      if (C.getTag() == DW_TAG_formal_parameter ||
          C.getTag() == DW_TAG_unspecified_parameters)
        RP.push_back(C);
    if (LP.size() != RP.size())
      return false;
    for (size_t I = 0, E = LP.size(); I < E; ++I) {
      if (LP[I].getTag() != RP[I].getTag())
        return false;
      if (LP[I].getTag() == DW_TAG_formal_parameter) {
        DWARFDie LT = LP[I].getAttributeValueAsReferencedDie(DW_AT_type);
        DWARFDie RT = RP[I].getAttributeValueAsReferencedDie(DW_AT_type);
        if (!compareTypes(LT, RT, Visited))
          return false;
      }
    }
    return true;
  }

  if (Tag == DW_TAG_ptr_to_member_type) {
    DWARFDie LC = LHS.getAttributeValueAsReferencedDie(DW_AT_containing_type);
    DWARFDie RC = RHS.getAttributeValueAsReferencedDie(DW_AT_containing_type);
    return compareTypes(LC, RC, Visited) && CompareInner();
  }

  if (Tag == DW_TAG_typedef)
    return CompareInner();

  if (Tag == DW_TAG_enumeration_type)
    return CompareInner();

  return true;
}

bool DWARFDiff::compareDIEs(DWARFDie LHS, DWARFDie RHS) {
  if (!LHS.isValid() || !RHS.isValid())
    return LHS.isValid() == RHS.isValid();
  if (LHS.getTag() != RHS.getTag())
    return false;

  // A forward declaration matches its definition. This is safe because the
  // caller already matched these DIEs by identity (qualified name + tag).
  bool LIsDecl = LHS.find(DW_AT_declaration).has_value();
  bool RIsDecl = RHS.find(DW_AT_declaration).has_value();
  if (LIsDecl || RIsDecl)
    return true;

  SmallDenseMap<dwarf::Attribute, DWARFFormValue, 16> LAttrs, RAttrs;
  for (const DWARFAttribute &A : LHS.attributes())
    LAttrs[A.Attr] = A.Value;
  for (const DWARFAttribute &A : RHS.attributes())
    RAttrs[A.Attr] = A.Value;

  for (auto &[Attr, LVal] : LAttrs) {
    if (isSkippedAttribute(Attr))
      continue;
    auto RIt = RAttrs.find(Attr);
    if (RIt == RAttrs.end())
      return false;
    if (isIdentityReferenceAttribute(Attr)) {
      DWARFDie LRef = LHS.getAttributeValueAsReferencedDie(Attr);
      DWARFDie RRef = RHS.getAttributeValueAsReferencedDie(Attr);
      if (getQualifiedName(LRef) != getQualifiedName(RRef))
        return false;
    } else if (isTypeReferenceAttribute(Attr)) {
      DWARFDie LRef = LHS.getAttributeValueAsReferencedDie(Attr);
      DWARFDie RRef = RHS.getAttributeValueAsReferencedDie(Attr);
      DenseSet<std::pair<uint64_t, uint64_t>> Visited;
      if (!compareTypes(LRef, RRef, Visited))
        return false;
    } else if (!compareFormValues(LVal, RIt->second)) {
      return false;
    }
  }

  for (auto &[Attr, RVal] : RAttrs) {
    if (isSkippedAttribute(Attr))
      continue;
    if (!LAttrs.contains(Attr))
      return false;
  }

  DenseMap<DIEIdentity, SmallVector<DWARFDie, 2>> LNamed, RNamed;
  DenseMap<dwarf::Tag, SmallVector<DWARFDie, 2>> LUnnamed, RUnnamed;

  for (DWARFDie C : LHS.children()) {
    DIEIdentity ID = getIdentity(C);
    if (ID.isValid())
      LNamed[ID].push_back(C);
    else if (C.getTag() != DW_TAG_null)
      LUnnamed[C.getTag()].push_back(C);
  }
  for (DWARFDie C : RHS.children()) {
    DIEIdentity ID = getIdentity(C);
    if (ID.isValid())
      RNamed[ID].push_back(C);
    else if (C.getTag() != DW_TAG_null)
      RUnnamed[C.getTag()].push_back(C);
  }

  for (auto &[ID, Dies] : LNamed)
    sortPreferDefinitions(Dies);
  for (auto &[ID, Dies] : RNamed)
    sortPreferDefinitions(Dies);

  for (auto &[ID, LDies] : LNamed) {
    auto RIt = RNamed.find(ID);
    if (RIt == RNamed.end())
      return false;
    if (!compareDIEs(LDies.front(), RIt->second.front()))
      return false;
  }
  for (auto &[ID, RDies] : RNamed)
    if (!LNamed.contains(ID))
      return false;

  for (auto &[Tag, LDies] : LUnnamed) {
    auto RIt = RUnnamed.find(Tag);
    if (RIt == RUnnamed.end())
      return false;
    auto &RDies = RIt->second;
    if (LDies.size() != RDies.size())
      return false;
    for (size_t I = 0, Count = LDies.size(); I < Count; ++I)
      if (!compareDIEs(LDies[I], RDies[I]))
        return false;
  }
  for (auto &[Tag, RDies] : RUnnamed)
    if (!LUnnamed.contains(Tag))
      return false;

  return true;
}

CUMap DWARFDiff::collectCUs(DWARFContext &Ctx) {
  CUMap CUs;
  for (auto &Unit : Ctx.normal_units()) {
    DWARFDie UDie = Unit->getUnitDIE(false);
    if (UDie.isValid())
      CUs[getIdentity(UDie)] = UDie;
  }
  return CUs;
}

void DWARFDiff::indexDIE(DWARFDie Die, DIEIndexMap &Index) {
  if (!Die.isValid())
    return;

  DIEIdentity ID = getIdentity(Die);
  if (ID.isValid())
    Index[ID].push_back(Die);

  for (DWARFDie Child : Die.children())
    indexDIE(Child, Index);
}

DIEIndexMap DWARFDiff::buildCUIndex(DWARFDie UnitDie) {
  DIEIndexMap Index;
  indexDIE(UnitDie, Index);
  for (auto &[ID, Dies] : Index)
    sortPreferDefinitions(Dies);
  return Index;
}

DiffDIERef DWARFDiff::makeDIERef(DWARFDie Die) {
  return {Die.getOffset(), Die.getTag(), getQualifiedName(Die)};
}

DiffResult DWARFDiff::diff(const DiffInput &LHS, const DiffInput &RHS) {
  DiffResult Result;

  CUMap LCUs = collectCUs(LHS.Context);
  CUMap RCUs = collectCUs(RHS.Context);

  auto isContainerTag = [](dwarf::Tag T) {
    return T == DW_TAG_compile_unit || T == DW_TAG_type_unit ||
           T == DW_TAG_partial_unit || T == DW_TAG_namespace;
  };

  CUMap LUnmatched, RUnmatched;

  for (auto &[CUID, CUDie] : LCUs) {
    auto RIt = RCUs.find(CUID);
    if (RIt == RCUs.end()) {
      DIEIndexMap Idx = buildCUIndex(CUDie);
      for (auto &[ID, Dies] : Idx)
        LUnmatched[ID] = Dies.front();
      continue;
    }
    DIEIndexMap LIdx = buildCUIndex(CUDie);
    DIEIndexMap RIdx = buildCUIndex(RIt->second);

    for (auto &[ID, LDies] : LIdx) {
      if (isContainerTag(ID.Tag))
        continue;
      auto RFind = RIdx.find(ID);
      if (RFind == RIdx.end()) {
        LUnmatched[ID] = LDies.front();
        continue;
      }
      if (compareDIEs(LDies.front(), RFind->second.front())) {
        ++Result.NumMatched;
      } else {
        Result.Different.push_back(
            {makeDIERef(LDies.front()), makeDIERef(RFind->second.front())});
      }
    }
    for (auto &[ID, RDies] : RIdx)
      if (!isContainerTag(ID.Tag) && !LIdx.contains(ID))
        RUnmatched[ID] = RDies.front();
  }
  for (auto &[CUID, CUDie] : RCUs) {
    if (LCUs.contains(CUID))
      continue;
    DIEIndexMap Idx = buildCUIndex(CUDie);
    for (auto &[ID, Dies] : Idx)
      RUnmatched[ID] = Dies.front();
  }

  // Phase 2: DIEs unmatched within their CU may have moved to a different
  // CU. Match them by identity alone.
  for (auto &[ID, Die] : make_early_inc_range(LUnmatched)) {
    if (RUnmatched.erase(ID)) {
      ++Result.NumMatched;
      LUnmatched.erase(ID);
    }
  }

  for (auto &[ID, Die] : LUnmatched)
    Result.OnlyInLHS.push_back(makeDIERef(Die));
  for (auto &[ID, Die] : RUnmatched)
    Result.OnlyInRHS.push_back(makeDIERef(Die));

  auto CmpRef = [](const DiffDIERef &A, const DiffDIERef &B) {
    if (A.QualifiedName != B.QualifiedName)
      return A.QualifiedName < B.QualifiedName;
    return A.Tag < B.Tag;
  };
  llvm::sort(Result.OnlyInLHS, CmpRef);
  llvm::sort(Result.OnlyInRHS, CmpRef);
  llvm::sort(Result.Different, [&](const DiffEntry &A, const DiffEntry &B) {
    return CmpRef(A.LHS, B.LHS);
  });

  return Result;
}

static constexpr llvm::StringLiteral SepLine =
    "----------------------------------------------"
    "----------------------------------";

static void printRow(raw_ostream &OS, const DiffDIERef *LHS,
                     const DiffDIERef *RHS) {
  if (LHS)
    OS << formatv("{0:x8}        ", LHS->Offset);
  else
    OS << "                  ";
  if (RHS)
    OS << formatv("{0:x8}        ", RHS->Offset);
  else
    OS << "                  ";

  const DiffDIERef &Ref = LHS ? *LHS : *RHS;
  OS << formatv("{0,-31}", TagString(Ref.Tag));
  if (!Ref.QualifiedName.empty())
    OS << Ref.QualifiedName;
  OS << "\n";
}

void llvm::printDiffResult(raw_ostream &OS, const DiffResult &Result) {
  bool HasOutput = !Result.OnlyInLHS.empty() || !Result.OnlyInRHS.empty() ||
                   !Result.Different.empty();
  if (HasOutput) {
    OS << SepLine << "\n";
    OS << formatv("{0,-18}{1,-18}{2,-31}{3}\n", "LHS", "RHS", "TAG", "NAME");
    OS << SepLine << "\n";
  }

  for (auto &Ref : Result.OnlyInLHS)
    printRow(OS, &Ref, nullptr);
  for (auto &Ref : Result.OnlyInRHS)
    printRow(OS, nullptr, &Ref);
  for (auto &E : Result.Different)
    printRow(OS, &E.LHS, &E.RHS);

  if (HasOutput)
    OS << SepLine << "\n";

  unsigned NumMissing = Result.OnlyInLHS.size();
  unsigned NumAdded = Result.OnlyInRHS.size();
  unsigned NumDiff = Result.Different.size();
  unsigned Total = Result.NumMatched + NumMissing + NumAdded + NumDiff;
  OS << formatv("Matched: {0}  Only in LHS: {1}  Only in RHS: {2}  "
                "Different: {3}  Total: {4}\n",
                Result.NumMatched, NumMissing, NumAdded, NumDiff, Total);
}
