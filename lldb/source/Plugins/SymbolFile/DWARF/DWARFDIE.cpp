//===-- DWARFDIE.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDIE.h"

#include "DWARFASTParser.h"
#include "DWARFDebugInfo.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFDeclContext.h"
#include "DWARFUnit.h"

using namespace lldb_private;

namespace {

/// Iterate through all DIEs elaborating (i.e. reachable by a chain of
/// DW_AT_specification and DW_AT_abstract_origin attributes) a given DIE. For
/// convenience, the starting die is included in the sequence as the first
/// item.
class ElaboratingDIEIterator
    : public std::iterator<std::input_iterator_tag, DWARFDIE> {

  // The operating invariant is: top of m_worklist contains the "current" item
  // and the rest of the list are items yet to be visited. An empty worklist
  // means we've reached the end.
  // Infinite recursion is prevented by maintaining a list of seen DIEs.
  // Container sizes are optimized for the case of following DW_AT_specification
  // and DW_AT_abstract_origin just once.
  llvm::SmallVector<DWARFDIE, 2> m_worklist;
  llvm::SmallSet<DWARFDebugInfoEntry *, 3> m_seen;

  void Next() {
    assert(!m_worklist.empty() && "Incrementing end iterator?");

    // Pop the current item from the list.
    DWARFDIE die = m_worklist.back();
    m_worklist.pop_back();

    // And add back any items that elaborate it.
    for (dw_attr_t attr : {DW_AT_specification, DW_AT_abstract_origin}) {
      if (DWARFDIE d = die.GetReferencedDIE(attr))
        if (m_seen.insert(die.GetDIE()).second)
          m_worklist.push_back(d);
    }
  }

public:
  /// An iterator starting at die d.
  explicit ElaboratingDIEIterator(DWARFDIE d) : m_worklist(1, d) {}

  /// End marker
  ElaboratingDIEIterator() = default;

  const DWARFDIE &operator*() const { return m_worklist.back(); }
  ElaboratingDIEIterator &operator++() {
    Next();
    return *this;
  }
  ElaboratingDIEIterator operator++(int) {
    ElaboratingDIEIterator I = *this;
    Next();
    return I;
  }

  friend bool operator==(const ElaboratingDIEIterator &a,
                         const ElaboratingDIEIterator &b) {
    if (a.m_worklist.empty() || b.m_worklist.empty())
      return a.m_worklist.empty() == b.m_worklist.empty();
    return a.m_worklist.back() == b.m_worklist.back();
  }
  friend bool operator!=(const ElaboratingDIEIterator &a,
                         const ElaboratingDIEIterator &b) {
    return !(a == b);
  }
};

llvm::iterator_range<ElaboratingDIEIterator>
elaborating_dies(const DWARFDIE &die) {
  return llvm::make_range(ElaboratingDIEIterator(die),
                          ElaboratingDIEIterator());
}
} // namespace

DWARFDIE
DWARFDIE::GetParent() const {
  if (IsValid())
    return DWARFDIE(m_cu, m_die->GetParent());
  else
    return DWARFDIE();
}

DWARFDIE
DWARFDIE::GetFirstChild() const {
  if (IsValid())
    return DWARFDIE(m_cu, m_die->GetFirstChild());
  else
    return DWARFDIE();
}

DWARFDIE
DWARFDIE::GetSibling() const {
  if (IsValid())
    return DWARFDIE(m_cu, m_die->GetSibling());
  else
    return DWARFDIE();
}

DWARFDIE
DWARFDIE::GetReferencedDIE(const dw_attr_t attr) const {
  if (IsValid())
    return m_die->GetAttributeValueAsReference(GetCU(), attr);
  else
    return {};
}

DWARFDIE
DWARFDIE::GetDIE(dw_offset_t die_offset) const {
  if (IsValid())
    return m_cu->GetDIE(die_offset);
  else
    return DWARFDIE();
}

DWARFDIE
DWARFDIE::GetAttributeValueAsReferenceDIE(const dw_attr_t attr) const {
  if (IsValid()) {
    DWARFUnit *cu = GetCU();
    const bool check_specification_or_abstract_origin = true;
    DWARFFormValue form_value;
    if (m_die->GetAttributeValue(cu, attr, form_value, nullptr,
                                 check_specification_or_abstract_origin))
      return form_value.Reference();
  }
  return DWARFDIE();
}

DWARFDIE
DWARFDIE::LookupDeepestBlock(lldb::addr_t address) const {
  if (!IsValid())
    return DWARFDIE();

  DWARFDIE result;
  bool check_children = false;
  bool match_addr_range = false;
  switch (Tag()) {
  case DW_TAG_class_type:
  case DW_TAG_namespace:
  case DW_TAG_structure_type:
  case DW_TAG_common_block:
    check_children = true;
    break;
  case DW_TAG_compile_unit:
  case DW_TAG_module:
  case DW_TAG_catch_block:
  case DW_TAG_subprogram:
  case DW_TAG_try_block:
  case DW_TAG_partial_unit:
    match_addr_range = true;
    break;
  case DW_TAG_lexical_block:
  case DW_TAG_inlined_subroutine:
    check_children = true;
    match_addr_range = true;
    break;
  default:
    break;
  }

  if (match_addr_range) {
    DWARFRangeList ranges;
    if (m_die->GetAttributeAddressRanges(m_cu, ranges,
                                         /*check_hi_lo_pc=*/true) &&
        ranges.FindEntryThatContains(address)) {
      check_children = true;
      switch (Tag()) {
      default:
        break;

      case DW_TAG_inlined_subroutine: // Inlined Function
      case DW_TAG_lexical_block:      // Block { } in code
        result = *this;
        break;
      }
    } else {
      check_children = false;
    }
  }

  if (check_children) {
    for (DWARFDIE child = GetFirstChild(); child; child = child.GetSibling()) {
      if (DWARFDIE child_result = child.LookupDeepestBlock(address))
        return child_result;
    }
  }
  return result;
}

const char *DWARFDIE::GetMangledName() const {
  if (IsValid())
    return m_die->GetMangledName(m_cu);
  else
    return nullptr;
}

const char *DWARFDIE::GetPubname() const {
  if (IsValid())
    return m_die->GetPubname(m_cu);
  else
    return nullptr;
}

const char *DWARFDIE::GetQualifiedName(std::string &storage) const {
  if (IsValid())
    return m_die->GetQualifiedName(m_cu, storage);
  else
    return nullptr;
}

// GetName
//
// Get value of the DW_AT_name attribute and place that value into the supplied
// stream object. If the DIE is a NULL object "NULL" is placed into the stream,
// and if no DW_AT_name attribute exists for the DIE then nothing is printed.
void DWARFDIE::GetName(Stream &s) const {
  if (!IsValid())
    return;
  if (GetDIE()->IsNULL()) {
    s.PutCString("NULL");
    return;
  }
  const char *name = GetDIE()->GetAttributeValueAsString(GetCU(), DW_AT_name, nullptr, true);
  if (!name)
    return;
  s.PutCString(name);
}

// AppendTypeName
//
// Follows the type name definition down through all needed tags to end up with
// a fully qualified type name and dump the results to the supplied stream.
// This is used to show the name of types given a type identifier.
void DWARFDIE::AppendTypeName(Stream &s) const {
  if (!IsValid())
    return;
  if (GetDIE()->IsNULL()) {
    s.PutCString("NULL");
    return;
  }
  if (const char *name = GetPubname()) {
    s.PutCString(name);
    return;
  }
  switch (Tag()) {
  case DW_TAG_array_type:
    break; // print out a "[]" after printing the full type of the element
           // below
  case DW_TAG_base_type:
    s.PutCString("base ");
    break;
  case DW_TAG_class_type:
    s.PutCString("class ");
    break;
  case DW_TAG_const_type:
    s.PutCString("const ");
    break;
  case DW_TAG_enumeration_type:
    s.PutCString("enum ");
    break;
  case DW_TAG_file_type:
    s.PutCString("file ");
    break;
  case DW_TAG_interface_type:
    s.PutCString("interface ");
    break;
  case DW_TAG_packed_type:
    s.PutCString("packed ");
    break;
  case DW_TAG_pointer_type:
    break; // print out a '*' after printing the full type below
  case DW_TAG_ptr_to_member_type:
    break; // print out a '*' after printing the full type below
  case DW_TAG_reference_type:
    break; // print out a '&' after printing the full type below
  case DW_TAG_restrict_type:
    s.PutCString("restrict ");
    break;
  case DW_TAG_set_type:
    s.PutCString("set ");
    break;
  case DW_TAG_shared_type:
    s.PutCString("shared ");
    break;
  case DW_TAG_string_type:
    s.PutCString("string ");
    break;
  case DW_TAG_structure_type:
    s.PutCString("struct ");
    break;
  case DW_TAG_subrange_type:
    s.PutCString("subrange ");
    break;
  case DW_TAG_subroutine_type:
    s.PutCString("function ");
    break;
  case DW_TAG_thrown_type:
    s.PutCString("thrown ");
    break;
  case DW_TAG_union_type:
    s.PutCString("union ");
    break;
  case DW_TAG_unspecified_type:
    s.PutCString("unspecified ");
    break;
  case DW_TAG_volatile_type:
    s.PutCString("volatile ");
    break;
  default:
    return;
  }

  // Follow the DW_AT_type if possible
  if (DWARFDIE next_die = GetAttributeValueAsReferenceDIE(DW_AT_type))
    next_die.AppendTypeName(s);

  switch (Tag()) {
  case DW_TAG_array_type:
    s.PutCString("[]");
    break;
  case DW_TAG_pointer_type:
    s.PutChar('*');
    break;
  case DW_TAG_ptr_to_member_type:
    s.PutChar('*');
    break;
  case DW_TAG_reference_type:
    s.PutChar('&');
    break;
  default:
    break;
  }
}

lldb_private::Type *DWARFDIE::ResolveType() const {
  if (IsValid())
    return GetDWARF()->ResolveType(*this, true);
  else
    return nullptr;
}

lldb_private::Type *DWARFDIE::ResolveTypeUID(const DWARFDIE &die) const {
  if (SymbolFileDWARF *dwarf = GetDWARF())
    return dwarf->ResolveTypeUID(die, true);
  return nullptr;
}

std::vector<DWARFDIE> DWARFDIE::GetDeclContextDIEs() const {
  if (!IsValid())
    return {};

  std::vector<DWARFDIE> result;
  DWARFDIE parent = GetParentDeclContextDIE();
  while (parent.IsValid() && parent.GetDIE() != GetDIE()) {
    result.push_back(std::move(parent));
    parent = parent.GetParentDeclContextDIE();
  }

  return result;
}

void DWARFDIE::GetDeclContext(
    llvm::SmallVectorImpl<lldb_private::CompilerContext> &context) const {
  const dw_tag_t tag = Tag();
  if (tag == DW_TAG_compile_unit || tag == DW_TAG_partial_unit)
    return;
  DWARFDIE parent = GetParent();
  if (parent)
    parent.GetDeclContext(context);
  switch (tag) {
  case DW_TAG_module:
    context.push_back({CompilerContextKind::Module, ConstString(GetName())});
    break;
  case DW_TAG_namespace:
    context.push_back({CompilerContextKind::Namespace, ConstString(GetName())});
    break;
  case DW_TAG_structure_type:
    context.push_back({CompilerContextKind::Struct, ConstString(GetName())});
    break;
  case DW_TAG_union_type:
    context.push_back({CompilerContextKind::Union, ConstString(GetName())});
    break;
  case DW_TAG_class_type:
    context.push_back({CompilerContextKind::Class, ConstString(GetName())});
    break;
  case DW_TAG_enumeration_type:
    context.push_back({CompilerContextKind::Enum, ConstString(GetName())});
    break;
  case DW_TAG_subprogram:
    context.push_back(
        {CompilerContextKind::Function, ConstString(GetPubname())});
    break;
  case DW_TAG_variable:
    context.push_back(
        {CompilerContextKind::Variable, ConstString(GetPubname())});
    break;
  case DW_TAG_typedef:
    context.push_back({CompilerContextKind::Typedef, ConstString(GetName())});
    break;
  default:
    break;
  }
}

DWARFDIE
DWARFDIE::GetParentDeclContextDIE() const {
  if (IsValid())
    return m_die->GetParentDeclContextDIE(m_cu);
  else
    return DWARFDIE();
}

bool DWARFDIE::IsStructUnionOrClass() const {
  const dw_tag_t tag = Tag();
  return tag == DW_TAG_class_type || tag == DW_TAG_structure_type ||
         tag == DW_TAG_union_type;
}

bool DWARFDIE::IsMethod() const {
  for (DWARFDIE d : elaborating_dies(*this))
    if (d.GetParent().IsStructUnionOrClass())
      return true;
  return false;
}

bool DWARFDIE::GetDIENamesAndRanges(
    const char *&name, const char *&mangled, DWARFRangeList &ranges,
    int &decl_file, int &decl_line, int &decl_column, int &call_file,
    int &call_line, int &call_column,
    lldb_private::DWARFExpression *frame_base) const {
  if (IsValid()) {
    return m_die->GetDIENamesAndRanges(
        GetCU(), name, mangled, ranges, decl_file, decl_line, decl_column,
        call_file, call_line, call_column, frame_base);
  } else
    return false;
}
