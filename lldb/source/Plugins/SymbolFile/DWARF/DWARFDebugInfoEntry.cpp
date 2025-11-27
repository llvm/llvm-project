//===-- DWARFDebugInfoEntry.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugInfoEntry.h"

#include <cassert>

#include <algorithm>
#include <limits>
#include <optional>

#include "LogChannelDWARF.h"
#include "lldb/Core/Module.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/ObjectFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFAddressRange.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/LEB128.h"

#include "DWARFCompileUnit.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDeclContext.h"
#include "DWARFFormValue.h"
#include "DWARFUnit.h"
#include "SymbolFileDWARF.h"
#include "SymbolFileDWARFDwo.h"

using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace llvm::dwarf;
extern int g_verbose;

// Extract a debug info entry for a given DWARFUnit from the data
// starting at the offset in offset_ptr
bool DWARFDebugInfoEntry::Extract(const DWARFDataExtractor &data,
                                  const DWARFUnit &unit,
                                  lldb::offset_t *offset_ptr) {
  m_offset = *offset_ptr;
  auto report_error = [&](const char *fmt, const auto &...vals) {
    unit.GetSymbolFileDWARF().GetObjectFile()->GetModule()->ReportError(
        "[{0:x16}]: {1}, please file a bug and "
        "attach the file at the start of this error message",
        static_cast<uint64_t>(m_offset), llvm::formatv(fmt, vals...));
    *offset_ptr = std::numeric_limits<lldb::offset_t>::max();
    return false;
  };

  m_parent_idx = 0;
  m_sibling_idx = 0;
  const uint64_t abbr_idx = data.GetULEB128(offset_ptr);
  if (abbr_idx > std::numeric_limits<uint16_t>::max())
    return report_error("abbreviation code {0} too big", abbr_idx);
  m_abbr_idx = abbr_idx;

  if (m_abbr_idx == 0) {
    m_tag = llvm::dwarf::DW_TAG_null;
    m_has_children = false;
    return true; // NULL debug tag entry
  }

  const auto *abbrevDecl = GetAbbreviationDeclarationPtr(&unit);
  if (abbrevDecl == nullptr)
    return report_error("invalid abbreviation code {0}", abbr_idx);

  m_tag = abbrevDecl->getTag();
  m_has_children = abbrevDecl->hasChildren();
  // Skip all data in the .debug_info or .debug_types for the attributes
  for (const auto &attribute : abbrevDecl->attributes()) {
    if (DWARFFormValue::SkipValue(attribute.Form, data, offset_ptr, &unit))
      continue;

    return report_error("Unsupported DW_FORM_{1:x}", attribute.Form);
  }
  return true;
}

static llvm::Expected<llvm::DWARFAddressRangesVector>
GetRanges(DWARFUnit &unit, const DWARFFormValue &value) {
  return (value.Form() == DW_FORM_rnglistx)
             ? unit.FindRnglistFromIndex(value.Unsigned())
             : unit.FindRnglistFromOffset(value.Unsigned());
}

static void ExtractAttrAndFormValue(
    const llvm::DWARFAbbreviationDeclaration::AttributeSpec &attr_spec,
    dw_attr_t &attr, DWARFFormValue &form_value) {
  attr = attr_spec.Attr;
  form_value.FormRef() = attr_spec.Form;
  if (attr_spec.isImplicitConst())
    form_value.SetSigned(attr_spec.getImplicitConstValue());
}

// GetDIENamesAndRanges
//
// Gets the valid address ranges for a given DIE by looking for a
// DW_AT_low_pc/DW_AT_high_pc pair, DW_AT_entry_pc, or DW_AT_ranges attributes.
bool DWARFDebugInfoEntry::GetDIENamesAndRanges(
    DWARFUnit *cu, const char *&name, const char *&mangled,
    llvm::DWARFAddressRangesVector &ranges, std::optional<int> &decl_file,
    std::optional<int> &decl_line, std::optional<int> &decl_column,
    std::optional<int> &call_file, std::optional<int> &call_line,
    std::optional<int> &call_column, DWARFExpressionList *frame_base) const {
  dw_addr_t lo_pc = LLDB_INVALID_ADDRESS;
  dw_addr_t hi_pc = LLDB_INVALID_ADDRESS;
  std::vector<DWARFDIE> dies;
  bool set_frame_base_loclist_addr = false;

  SymbolFileDWARF &dwarf = cu->GetSymbolFileDWARF();
  lldb::ModuleSP module = dwarf.GetObjectFile()->GetModule();

  if (const auto *abbrevDecl = GetAbbreviationDeclarationPtr(cu)) {
    const DWARFDataExtractor &data = cu->GetData();
    lldb::offset_t offset = GetFirstAttributeOffset();

    if (!data.ValidOffset(offset))
      return false;

    bool do_offset = false;

    for (const auto &attribute : abbrevDecl->attributes()) {
      DWARFFormValue form_value(cu);
      dw_attr_t attr;
      ExtractAttrAndFormValue(attribute, attr, form_value);

      if (form_value.ExtractValue(data, &offset)) {
        switch (attr) {
        case DW_AT_low_pc:
          lo_pc = form_value.Address();

          if (do_offset)
            hi_pc += lo_pc;
          do_offset = false;
          break;

        case DW_AT_entry_pc:
          lo_pc = form_value.Address();
          break;

        case DW_AT_high_pc:
          if (form_value.Form() == DW_FORM_addr ||
              form_value.Form() == DW_FORM_addrx ||
              form_value.Form() == DW_FORM_GNU_addr_index) {
            hi_pc = form_value.Address();
          } else {
            hi_pc = form_value.Unsigned();
            if (lo_pc == LLDB_INVALID_ADDRESS)
              do_offset = hi_pc != LLDB_INVALID_ADDRESS;
            else
              hi_pc += lo_pc; // DWARF 4 introduces <offset-from-lo-pc> to save
                              // on relocations
          }
          break;

        case DW_AT_ranges:
          if (llvm::Expected<llvm::DWARFAddressRangesVector> r =
                  GetRanges(*cu, form_value)) {
            ranges = std::move(*r);
          } else {
            module->ReportError(
                "[{0:x16}]: DIE has DW_AT_ranges({1} {2:x16}) attribute, but "
                "range extraction failed ({3}), please file a bug "
                "and attach the file at the start of this error message",
                GetOffset(), llvm::dwarf::FormEncodingString(form_value.Form()),
                form_value.Unsigned(), fmt_consume(r.takeError()));
          }
          break;

        case DW_AT_name:
          if (name == nullptr)
            name = form_value.AsCString();
          break;

        case DW_AT_MIPS_linkage_name:
        case DW_AT_linkage_name:
          if (mangled == nullptr)
            mangled = form_value.AsCString();
          break;

        case DW_AT_abstract_origin:
          dies.push_back(form_value.Reference());
          break;

        case DW_AT_specification:
          dies.push_back(form_value.Reference());
          break;

        case DW_AT_decl_file:
          if (!decl_file)
            decl_file = form_value.Unsigned();
          break;

        case DW_AT_decl_line:
          if (!decl_line)
            decl_line = form_value.Unsigned();
          break;

        case DW_AT_decl_column:
          if (!decl_column)
            decl_column = form_value.Unsigned();
          break;

        case DW_AT_call_file:
          if (!call_file)
            call_file = form_value.Unsigned();
          break;

        case DW_AT_call_line:
          if (!call_line)
            call_line = form_value.Unsigned();
          break;

        case DW_AT_call_column:
          if (!call_column)
            call_column = form_value.Unsigned();
          break;

        case DW_AT_frame_base:
          if (frame_base) {
            if (form_value.BlockData()) {
              uint64_t block_offset =
                  form_value.BlockData() - data.GetDataStart();
              uint64_t block_length = form_value.Unsigned();
              *frame_base =
                  DWARFExpressionList(module,
                                      DWARFExpression(DataExtractor(
                                          data, block_offset, block_length)),
                                      cu);
            } else {
              DataExtractor data = cu->GetLocationData();
              const dw_offset_t offset = form_value.Unsigned();
              if (data.ValidOffset(offset)) {
                data = DataExtractor(data, offset, data.GetByteSize() - offset);
                if (lo_pc != LLDB_INVALID_ADDRESS) {
                  assert(lo_pc >= cu->GetBaseAddress());
                  cu->ParseDWARFLocationList(data, *frame_base);
                  frame_base->SetFuncFileAddress(lo_pc);
                } else
                  set_frame_base_loclist_addr = true;
              }
            }
          }
          break;

        default:
          break;
        }
      }
    }
  }

  if (ranges.empty() && lo_pc != LLDB_INVALID_ADDRESS) {
    lldb::addr_t range_hi_pc =
        (hi_pc != LLDB_INVALID_ADDRESS && hi_pc > lo_pc) ? hi_pc : lo_pc;
    ranges.emplace_back(lo_pc, range_hi_pc);
  }

  if (set_frame_base_loclist_addr && !ranges.empty()) {
    dw_addr_t file_addr = ranges.begin()->LowPC;
    assert(file_addr >= cu->GetBaseAddress());
    frame_base->SetFuncFileAddress(file_addr);
  }

  if (ranges.empty() || name == nullptr || mangled == nullptr) {
    for (const DWARFDIE &die : dies) {
      if (die) {
        die.GetDIE()->GetDIENamesAndRanges(die.GetCU(), name, mangled, ranges,
                                           decl_file, decl_line, decl_column,
                                           call_file, call_line, call_column);
      }
    }
  }
  return !ranges.empty();
}

/// Helper for the public \ref DWARFDebugInfoEntry::GetAttributes API.
/// Adds all attributes of the DIE at the top of the \c worklist to the
/// \c attributes list. Specifcations and abstract origins are added
/// to the \c worklist if the referenced DIE has not been seen before.
static bool
GetAttributes(llvm::SmallVectorImpl<DWARFDIE> &worklist,
              llvm::SmallPtrSet<DWARFDebugInfoEntry const *, 3> &seen,
              DWARFAttributes &attributes) {
  assert(!worklist.empty() && "Need at least one DIE to visit.");
  assert(seen.size() >= 1 &&
         "Need to have seen at least the currently visited entry.");

  DWARFDIE current = worklist.pop_back_val();

  const auto *cu = current.GetCU();
  assert(cu);

  const auto *entry = current.GetDIE();
  assert(entry);

  const auto *abbrevDecl =
      entry->GetAbbreviationDeclarationPtr(current.GetCU());
  if (!abbrevDecl)
    return false;

  const DWARFDataExtractor &data = cu->GetData();
  lldb::offset_t offset = current.GetDIE()->GetFirstAttributeOffset();

  const bool is_first_die = seen.size() == 1;

  for (const auto &attribute : abbrevDecl->attributes()) {
    DWARFFormValue form_value(cu);
    dw_attr_t attr;
    ExtractAttrAndFormValue(attribute, attr, form_value);

    // If we are tracking down DW_AT_specification or DW_AT_abstract_origin
    // attributes, the depth will be non-zero. We need to omit certain
    // attributes that don't make sense.
    switch (attr) {
    case DW_AT_sibling:
    case DW_AT_declaration:
      if (!is_first_die) {
        // This attribute doesn't make sense when combined with the DIE that
        // references this DIE. We know a DIE is referencing this DIE because
        // we've visited more than one DIE already.
        break;
      }
      [[fallthrough]];
    default:
      attributes.Append(form_value, offset, attr);
      break;
    }

    if (attr == DW_AT_specification || attr == DW_AT_abstract_origin) {
      if (form_value.ExtractValue(data, &offset)) {
        if (DWARFDIE spec_die = form_value.Reference()) {
          if (seen.insert(spec_die.GetDIE()).second)
            worklist.push_back(spec_die);
        }
      }
    } else {
      const dw_form_t form = form_value.Form();
      std::optional<uint8_t> fixed_skip_size =
          DWARFFormValue::GetFixedSize(form, cu);
      if (fixed_skip_size)
        offset += *fixed_skip_size;
      else
        DWARFFormValue::SkipValue(form, data, &offset, cu);
    }
  }

  return true;
}

DWARFAttributes DWARFDebugInfoEntry::GetAttributes(const DWARFUnit *cu,
                                                   Recurse recurse) const {
  // FIXME: use ElaboratingDIEIterator to follow specifications/abstract origins
  // instead of maintaining our own worklist/seen list.

  DWARFAttributes attributes;

  llvm::SmallVector<DWARFDIE, 3> worklist;
  worklist.emplace_back(cu, this);

  // Keep track if DIEs already seen to prevent infinite recursion.
  // Value of '3' was picked for the same reason that
  // DWARFDie::findRecursively does.
  llvm::SmallPtrSet<DWARFDebugInfoEntry const *, 3> seen;
  seen.insert(this);

  do {
    if (!::GetAttributes(worklist, seen, attributes)) {
      attributes.Clear();
      break;
    }
  } while (!worklist.empty() && recurse == Recurse::yes);

  return attributes;
}

// GetAttributeValue
//
// Get the value of an attribute and return the .debug_info or .debug_types
// offset of the attribute if it was properly extracted into form_value,
// or zero if we fail since an offset of zero is invalid for an attribute (it
// would be a compile unit header).
dw_offset_t DWARFDebugInfoEntry::GetAttributeValue(
    const DWARFUnit *cu, const dw_attr_t attr, DWARFFormValue &form_value,
    dw_offset_t *end_attr_offset_ptr, bool check_elaborating_dies) const {
  if (const auto *abbrevDecl = GetAbbreviationDeclarationPtr(cu)) {
    std::optional<uint32_t> attr_idx = abbrevDecl->findAttributeIndex(attr);

    if (attr_idx) {
      const DWARFDataExtractor &data = cu->GetData();
      lldb::offset_t offset = GetFirstAttributeOffset();

      uint32_t idx = 0;
      while (idx < *attr_idx)
        DWARFFormValue::SkipValue(abbrevDecl->getFormByIndex(idx++), data,
                                  &offset, cu);

      const dw_offset_t attr_offset = offset;
      form_value.SetUnit(cu);
      form_value.SetForm(abbrevDecl->getFormByIndex(idx));
      if (abbrevDecl->getAttrIsImplicitConstByIndex(idx))
        form_value.SetValue(abbrevDecl->getAttrImplicitConstValueByIndex(idx));

      if (form_value.ExtractValue(data, &offset)) {
        if (end_attr_offset_ptr)
          *end_attr_offset_ptr = offset;
        return attr_offset;
      }
    }
  }

  if (check_elaborating_dies) {
    for (dw_attr_t elaborating_attr :
         {DW_AT_specification, DW_AT_abstract_origin, DW_AT_signature}) {
      if (!GetAttributeValue(cu, elaborating_attr, form_value))
        continue;
      DWARFDIE die = form_value.Reference();
      if (!die)
        continue;
      dw_offset_t die_offset = die.GetDIE()->GetAttributeValue(
          die.GetCU(), attr, form_value, end_attr_offset_ptr, false);
      if (die_offset)
        return die_offset;
    }
  }
  return 0;
}

// GetAttributeValueAsString
//
// Get the value of an attribute as a string return it. The resulting pointer
// to the string data exists within the supplied SymbolFileDWARF and will only
// be available as long as the SymbolFileDWARF is still around and it's content
// doesn't change.
const char *DWARFDebugInfoEntry::GetAttributeValueAsString(
    const DWARFUnit *cu, const dw_attr_t attr, const char *fail_value,
    bool check_elaborating_dies) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr, check_elaborating_dies))
    return form_value.AsCString();
  return fail_value;
}

// GetAttributeValueAsUnsigned
//
// Get the value of an attribute as unsigned and return it.
uint64_t DWARFDebugInfoEntry::GetAttributeValueAsUnsigned(
    const DWARFUnit *cu, const dw_attr_t attr, uint64_t fail_value,
    bool check_elaborating_dies) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr, check_elaborating_dies))
    return form_value.Unsigned();
  return fail_value;
}

std::optional<uint64_t>
DWARFDebugInfoEntry::GetAttributeValueAsOptionalUnsigned(
    const DWARFUnit *cu, const dw_attr_t attr,
    bool check_elaborating_dies) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr, check_elaborating_dies))
    return form_value.Unsigned();
  return std::nullopt;
}

// GetAttributeValueAsReference
//
// Get the value of an attribute as reference and fix up and compile unit
// relative offsets as needed.
DWARFDIE DWARFDebugInfoEntry::GetAttributeValueAsReference(
    const DWARFUnit *cu, const dw_attr_t attr,
    bool check_elaborating_dies) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr, check_elaborating_dies))
    return form_value.Reference();
  return {};
}

uint64_t DWARFDebugInfoEntry::GetAttributeValueAsAddress(
    const DWARFUnit *cu, const dw_attr_t attr, uint64_t fail_value,
    bool check_elaborating_dies) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr, check_elaborating_dies))
    return form_value.Address();
  return fail_value;
}

// GetAttributeHighPC
//
// Get the hi_pc, adding hi_pc to lo_pc when specified as an <offset-from-low-
// pc>.
//
// Returns the hi_pc or fail_value.
dw_addr_t
DWARFDebugInfoEntry::GetAttributeHighPC(const DWARFUnit *cu, dw_addr_t lo_pc,
                                        uint64_t fail_value,
                                        bool check_elaborating_dies) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, DW_AT_high_pc, form_value, nullptr,
                        check_elaborating_dies)) {
    dw_form_t form = form_value.Form();
    if (form == DW_FORM_addr || form == DW_FORM_addrx ||
        form == DW_FORM_GNU_addr_index)
      return form_value.Address();

    // DWARF4 can specify the hi_pc as an <offset-from-lowpc>
    return lo_pc + form_value.Unsigned();
  }
  return fail_value;
}

// GetAttributeAddressRange
//
// Get the lo_pc and hi_pc, adding hi_pc to lo_pc when specified as an <offset-
// from-low-pc>.
//
// Returns true or sets lo_pc and hi_pc to fail_value.
bool DWARFDebugInfoEntry::GetAttributeAddressRange(
    const DWARFUnit *cu, dw_addr_t &lo_pc, dw_addr_t &hi_pc,
    uint64_t fail_value, bool check_elaborating_dies) const {
  lo_pc = GetAttributeValueAsAddress(cu, DW_AT_low_pc, fail_value,
                                     check_elaborating_dies);
  if (lo_pc != fail_value) {
    hi_pc = GetAttributeHighPC(cu, lo_pc, fail_value, check_elaborating_dies);
    if (hi_pc != fail_value)
      return true;
  }
  lo_pc = fail_value;
  hi_pc = fail_value;
  return false;
}

llvm::Expected<llvm::DWARFAddressRangesVector>
DWARFDebugInfoEntry::GetAttributeAddressRanges(
    DWARFUnit *cu, bool check_hi_lo_pc, bool check_elaborating_dies) const {

  DWARFFormValue form_value;
  if (GetAttributeValue(cu, DW_AT_ranges, form_value))
    return GetRanges(*cu, form_value);

  if (check_hi_lo_pc) {
    dw_addr_t lo_pc = LLDB_INVALID_ADDRESS;
    dw_addr_t hi_pc = LLDB_INVALID_ADDRESS;
    if (GetAttributeAddressRange(cu, lo_pc, hi_pc, LLDB_INVALID_ADDRESS,
                                 check_elaborating_dies) &&
        lo_pc < hi_pc)
      return llvm::DWARFAddressRangesVector{{lo_pc, hi_pc}};
  }
  return llvm::createStringError("DIE has no address range information");
}

// GetName
//
// Get value of the DW_AT_name attribute and return it if one exists, else
// return NULL.
const char *DWARFDebugInfoEntry::GetName(const DWARFUnit *cu) const {
  return GetAttributeValueAsString(cu, DW_AT_name, nullptr, true);
}

// GetMangledName
//
// Get value of the DW_AT_MIPS_linkage_name attribute and return it if one
// exists, else return the value of the DW_AT_name attribute
const char *
DWARFDebugInfoEntry::GetMangledName(const DWARFUnit *cu,
                                    bool substitute_name_allowed) const {
  const char *name = nullptr;

  name = GetAttributeValueAsString(cu, DW_AT_MIPS_linkage_name, nullptr, true);
  if (name)
    return name;

  name = GetAttributeValueAsString(cu, DW_AT_linkage_name, nullptr, true);
  if (name)
    return name;

  if (!substitute_name_allowed)
    return nullptr;

  name = GetAttributeValueAsString(cu, DW_AT_name, nullptr, true);
  return name;
}

// GetPubname
//
// Get value the name for a DIE as it should appear for a .debug_pubnames or
// .debug_pubtypes section.
const char *DWARFDebugInfoEntry::GetPubname(const DWARFUnit *cu) const {
  const char *name = nullptr;
  if (!cu)
    return name;

  name = GetAttributeValueAsString(cu, DW_AT_MIPS_linkage_name, nullptr, true);
  if (name)
    return name;

  name = GetAttributeValueAsString(cu, DW_AT_linkage_name, nullptr, true);
  if (name)
    return name;

  name = GetAttributeValueAsString(cu, DW_AT_name, nullptr, true);
  return name;
}

/// This function is builds a table very similar to the standard .debug_aranges
/// table, except that the actual DIE offset for the function is placed in the
/// table instead of the compile unit offset.
void DWARFDebugInfoEntry::BuildFunctionAddressRangeTable(
    DWARFUnit *cu, DWARFDebugAranges *debug_aranges) const {
  Log *log = GetLog(DWARFLog::DebugInfo);
  if (m_tag) {
    // Subprogram forward declarations don't have
    // DW_AT_ranges/DW_AT_low_pc/DW_AT_high_pc attributes, so don't even try
    // getting address range information for them.
    if (m_tag == DW_TAG_subprogram &&
        !GetAttributeValueAsOptionalUnsigned(cu, DW_AT_declaration)) {
      if (llvm::Expected<llvm::DWARFAddressRangesVector> ranges =
              GetAttributeAddressRanges(cu, /*check_hi_lo_pc=*/true)) {
        for (const auto &r : *ranges)
          debug_aranges->AppendRange(GetOffset(), r.LowPC, r.HighPC);
      } else {
        LLDB_LOG_ERROR(log, ranges.takeError(), "DIE({1:x}): {0}", GetOffset());
      }
    }

    const DWARFDebugInfoEntry *child = GetFirstChild();
    while (child) {
      child->BuildFunctionAddressRangeTable(cu, debug_aranges);
      child = child->GetSibling();
    }
  }
}

lldb::offset_t DWARFDebugInfoEntry::GetFirstAttributeOffset() const {
  return GetOffset() + llvm::getULEB128Size(m_abbr_idx);
}

const llvm::DWARFAbbreviationDeclaration *
DWARFDebugInfoEntry::GetAbbreviationDeclarationPtr(const DWARFUnit *cu) const {
  if (!cu)
    return nullptr;

  const llvm::DWARFAbbreviationDeclarationSet *abbrev_set =
      cu->GetAbbreviations();
  if (!abbrev_set)
    return nullptr;

  return abbrev_set->getAbbreviationDeclaration(m_abbr_idx);
}

bool DWARFDebugInfoEntry::IsGlobalOrStaticScopeVariable() const {
  if (Tag() != DW_TAG_variable && Tag() != DW_TAG_member)
    return false;
  const DWARFDebugInfoEntry *parent_die = GetParent();
  while (parent_die != nullptr) {
    switch (parent_die->Tag()) {
    case DW_TAG_subprogram:
    case DW_TAG_lexical_block:
    case DW_TAG_inlined_subroutine:
      return false;

    case DW_TAG_compile_unit:
    case DW_TAG_partial_unit:
      return true;

    default:
      break;
    }
    parent_die = parent_die->GetParent();
  }
  return false;
}

bool DWARFDebugInfoEntry::operator==(const DWARFDebugInfoEntry &rhs) const {
  return m_offset == rhs.m_offset && m_parent_idx == rhs.m_parent_idx &&
         m_sibling_idx == rhs.m_sibling_idx &&
         m_abbr_idx == rhs.m_abbr_idx && m_has_children == rhs.m_has_children &&
         m_tag == rhs.m_tag;
}

bool DWARFDebugInfoEntry::operator!=(const DWARFDebugInfoEntry &rhs) const {
  return !(*this == rhs);
}
