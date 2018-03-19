//===-- DWARFDIE.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDIE.h"

#include "DWARFASTParser.h"
#include "DWARFUnit.h"
#include "DWARFDIECollection.h"
#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFDebugRanges.h"
#include "DWARFDeclContext.h"
#include "DWARFFormValue.h"
#include "SymbolFileDWARF.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeSystem.h"

using namespace lldb_private;

DIERef DWARFDIE::GetDIERef() const {
  if (!IsValid())
    return DIERef();

  dw_offset_t cu_offset = m_cu->GetOffset();
  if (m_cu->GetBaseObjOffset() != DW_INVALID_OFFSET)
    cu_offset = m_cu->GetBaseObjOffset();
  return DIERef(cu_offset, m_die->GetOffset());
}

dw_tag_t DWARFDIE::Tag() const {
  if (m_die)
    return m_die->Tag();
  else
    return 0;
}

const char *DWARFDIE::GetTagAsCString() const {
  return lldb_private::DW_TAG_value_to_name(Tag());
}

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
  const dw_offset_t die_offset =
      GetAttributeValueAsReference(attr, DW_INVALID_OFFSET);
  if (die_offset != DW_INVALID_OFFSET)
    return GetDIE(die_offset);
  else
    return DWARFDIE();
}

DWARFDIE
DWARFDIE::GetDIE(dw_offset_t die_offset) const {
  if (IsValid())
    return m_cu->GetDIE(die_offset);
  else
    return DWARFDIE();
}

const char *DWARFDIE::GetAttributeValueAsString(const dw_attr_t attr,
                                                const char *fail_value) const {
  if (IsValid())
    return m_die->GetAttributeValueAsString(GetDWARF(), GetCU(), attr,
                                            fail_value);
  else
    return fail_value;
}

uint64_t DWARFDIE::GetAttributeValueAsUnsigned(const dw_attr_t attr,
                                               uint64_t fail_value) const {
  if (IsValid())
    return m_die->GetAttributeValueAsUnsigned(GetDWARF(), GetCU(), attr,
                                              fail_value);
  else
    return fail_value;
}

int64_t DWARFDIE::GetAttributeValueAsSigned(const dw_attr_t attr,
                                            int64_t fail_value) const {
  if (IsValid())
    return m_die->GetAttributeValueAsSigned(GetDWARF(), GetCU(), attr,
                                            fail_value);
  else
    return fail_value;
}

DWARFDIE
DWARFDIE::GetAttributeValueAsReferenceDIE(const dw_attr_t attr) const {
  if (IsValid()) {
    DWARFUnit *cu = GetCU();
    SymbolFileDWARF *dwarf = cu->GetSymbolFileDWARF();
    const bool check_specification_or_abstract_origin = true;
    DWARFFormValue form_value;
    if (m_die->GetAttributeValue(dwarf, cu, attr, form_value, nullptr,
                                 check_specification_or_abstract_origin))
      return dwarf->GetDIE(DIERef(form_value));
  }
  return DWARFDIE();
}

uint64_t DWARFDIE::GetAttributeValueAsReference(const dw_attr_t attr,
                                                uint64_t fail_value) const {
  if (IsValid())
    return m_die->GetAttributeValueAsReference(GetDWARF(), GetCU(), attr,
                                               fail_value);
  else
    return fail_value;
}

uint64_t DWARFDIE::GetAttributeValueAsAddress(const dw_attr_t attr,
                                              uint64_t fail_value) const {
  if (IsValid())
    return m_die->GetAttributeValueAsAddress(GetDWARF(), GetCU(), attr,
                                             fail_value);
  else
    return fail_value;
}

DWARFDIE
DWARFDIE::LookupDeepestBlock(lldb::addr_t file_addr) const {
  if (IsValid()) {
    SymbolFileDWARF *dwarf = GetDWARF();
    DWARFUnit *cu = GetCU();
    DWARFDebugInfoEntry *function_die = nullptr;
    DWARFDebugInfoEntry *block_die = nullptr;
    if (m_die->LookupAddress(file_addr, dwarf, cu, &function_die, &block_die)) {
      if (block_die && block_die != function_die) {
        if (cu->ContainsDIEOffset(block_die->GetOffset()))
          return DWARFDIE(cu, block_die);
        else
          return DWARFDIE(dwarf->DebugInfo()->GetCompileUnit(
                              DIERef(cu->GetOffset(), block_die->GetOffset())),
                          block_die);
      }
    }
  }
  return DWARFDIE();
}

lldb::user_id_t DWARFDIE::GetID() const {
  return GetDIERef().GetUID(GetDWARF());
}

const char *DWARFDIE::GetName() const {
  if (IsValid())
    return m_die->GetName(GetDWARF(), m_cu);
  else
    return nullptr;
}

const char *DWARFDIE::GetMangledName() const {
  if (IsValid())
    return m_die->GetMangledName(GetDWARF(), m_cu);
  else
    return nullptr;
}

const char *DWARFDIE::GetPubname() const {
  if (IsValid())
    return m_die->GetPubname(GetDWARF(), m_cu);
  else
    return nullptr;
}

const char *DWARFDIE::GetQualifiedName(std::string &storage) const {
  if (IsValid())
    return m_die->GetQualifiedName(GetDWARF(), m_cu, storage);
  else
    return nullptr;
}

lldb::LanguageType DWARFDIE::GetLanguage() const {
  if (IsValid())
    return m_cu->GetLanguageType();
  else
    return lldb::eLanguageTypeUnknown;
}

lldb::ModuleSP DWARFDIE::GetModule() const {
  SymbolFileDWARF *dwarf = GetDWARF();
  if (dwarf)
    return dwarf->GetObjectFile()->GetModule();
  else
    return lldb::ModuleSP();
}

lldb_private::CompileUnit *DWARFDIE::GetLLDBCompileUnit() const {
  if (IsValid())
    return GetDWARF()->GetCompUnitForDWARFCompUnit(GetCU());
  else
    return nullptr;
}

lldb_private::Type *DWARFDIE::ResolveType() const {
  if (IsValid())
    return GetDWARF()->ResolveType(*this, true);
  else
    return nullptr;
}

lldb_private::Type *DWARFDIE::ResolveTypeUID(const DIERef &die_ref) const {
  SymbolFileDWARF *dwarf = GetDWARF();
  if (dwarf)
    return dwarf->ResolveTypeUID(dwarf->GetDIE(die_ref), true);
  else
    return nullptr;
}

void DWARFDIE::GetDeclContextDIEs(DWARFDIECollection &decl_context_dies) const {
  if (IsValid()) {
    DWARFDIE parent_decl_ctx_die =
        m_die->GetParentDeclContextDIE(GetDWARF(), GetCU());
    if (parent_decl_ctx_die && parent_decl_ctx_die.GetDIE() != GetDIE()) {
      decl_context_dies.Append(parent_decl_ctx_die);
      parent_decl_ctx_die.GetDeclContextDIEs(decl_context_dies);
    }
  }
}

void DWARFDIE::GetDWARFDeclContext(DWARFDeclContext &dwarf_decl_ctx) const {
  if (IsValid()) {
    dwarf_decl_ctx.SetLanguage(GetLanguage());
    m_die->GetDWARFDeclContext(GetDWARF(), GetCU(), dwarf_decl_ctx);
  } else {
    dwarf_decl_ctx.Clear();
  }
}

void DWARFDIE::GetDWOContext(std::vector<CompilerContext> &context) const {
  const dw_tag_t tag = Tag();
  if (tag == DW_TAG_compile_unit)
    return;
  DWARFDIE parent = GetParent();
  if (parent)
    parent.GetDWOContext(context);
  switch (tag) {
  case DW_TAG_module:
    context.push_back(
        CompilerContext(CompilerContextKind::Module, ConstString(GetName())));
    break;
  case DW_TAG_namespace:
    context.push_back(CompilerContext(CompilerContextKind::Namespace,
                                      ConstString(GetName())));
    break;
  case DW_TAG_structure_type:
    context.push_back(CompilerContext(CompilerContextKind::Structure,
                                      ConstString(GetName())));
    break;
  case DW_TAG_union_type:
    context.push_back(
        CompilerContext(CompilerContextKind::Union, ConstString(GetName())));
    break;
  case DW_TAG_class_type:
    context.push_back(
        CompilerContext(CompilerContextKind::Class, ConstString(GetName())));
    break;
  case DW_TAG_enumeration_type:
    context.push_back(CompilerContext(CompilerContextKind::Enumeration,
                                      ConstString(GetName())));
    break;
  case DW_TAG_subprogram:
    context.push_back(CompilerContext(CompilerContextKind::Function,
                                      ConstString(GetPubname())));
    break;
  case DW_TAG_variable:
    context.push_back(CompilerContext(CompilerContextKind::Variable,
                                      ConstString(GetPubname())));
    break;
  case DW_TAG_typedef:
    context.push_back(
        CompilerContext(CompilerContextKind::Typedef, ConstString(GetName())));
    break;
  default:
    break;
  }
}

DWARFDIE
DWARFDIE::GetParentDeclContextDIE() const {
  if (IsValid())
    return m_die->GetParentDeclContextDIE(GetDWARF(), m_cu);
  else
    return DWARFDIE();
}

dw_offset_t DWARFDIE::GetOffset() const {
  if (IsValid())
    return m_die->GetOffset();
  else
    return DW_INVALID_OFFSET;
}

dw_offset_t DWARFDIE::GetCompileUnitRelativeOffset() const {
  if (IsValid())
    return m_die->GetOffset() - m_cu->GetOffset();
  else
    return DW_INVALID_OFFSET;
}

SymbolFileDWARF *DWARFDIE::GetDWARF() const {
  if (m_cu)
    return m_cu->GetSymbolFileDWARF();
  else
    return nullptr;
}

lldb_private::TypeSystem *DWARFDIE::GetTypeSystem() const {
  if (m_cu)
    return m_cu->GetTypeSystem();
  else
    return nullptr;
}

DWARFASTParser *DWARFDIE::GetDWARFParser() const {
  lldb_private::TypeSystem *type_system = GetTypeSystem();
  if (type_system)
    return type_system->GetDWARFParser();
  else
    return nullptr;
}

bool DWARFDIE::IsStructOrClass() const {
  const dw_tag_t tag = Tag();
  return tag == DW_TAG_class_type || tag == DW_TAG_structure_type;
}

DWARFDIE
DWARFDIE::GetContainingDWOModuleDIE() const {
  if (IsValid()) {
    DWARFDIE top_module_die;
    // Now make sure this DIE is scoped in a DW_TAG_module tag and return true
    // if so
    for (DWARFDIE parent = GetParent(); parent.IsValid();
         parent = parent.GetParent()) {
      const dw_tag_t tag = parent.Tag();
      if (tag == DW_TAG_module)
        top_module_die = parent;
      else if (tag == DW_TAG_compile_unit)
        break;
    }

    return top_module_die;
  }
  return DWARFDIE();
}

lldb::ModuleSP DWARFDIE::GetContainingDWOModule() const {
  if (IsValid()) {
    DWARFDIE dwo_module_die = GetContainingDWOModuleDIE();

    if (dwo_module_die) {
      const char *module_name = dwo_module_die.GetName();
      if (module_name)
        return GetDWARF()->GetDWOModule(lldb_private::ConstString(module_name));
    }
  }
  return lldb::ModuleSP();
}

bool DWARFDIE::HasChildren() const {
  return m_die && m_die->HasChildren();
}

bool DWARFDIE::Supports_DW_AT_APPLE_objc_complete_type() const {
  return IsValid() && GetDWARF()->Supports_DW_AT_APPLE_objc_complete_type(m_cu);
}

size_t DWARFDIE::GetAttributes(DWARFAttributes &attributes,
                               uint32_t depth) const {
  if (IsValid()) {
    return m_die->GetAttributes(m_cu, m_cu->GetFixedFormSizes(), attributes,
                                depth);
  }
  if (depth == 0)
    attributes.Clear();
  return 0;
}

bool DWARFDIE::GetDIENamesAndRanges(
    const char *&name, const char *&mangled, DWARFRangeList &ranges,
    int &decl_file, int &decl_line, int &decl_column, int &call_file,
    int &call_line, int &call_column,
    lldb_private::DWARFExpression *frame_base) const {
  if (IsValid()) {
    return m_die->GetDIENamesAndRanges(
        GetDWARF(), GetCU(), name, mangled, ranges, decl_file, decl_line,
        decl_column, call_file, call_line, call_column, frame_base);
  } else
    return false;
}

void DWARFDIE::Dump(lldb_private::Stream *s,
                    const uint32_t recurse_depth) const {
  if (s && IsValid())
    m_die->Dump(GetDWARF(), GetCU(), *s, recurse_depth);
}

CompilerDecl DWARFDIE::GetDecl() const {
  DWARFASTParser *dwarf_ast = GetDWARFParser();
  if (dwarf_ast)
    return dwarf_ast->GetDeclForUIDFromDWARF(*this);
  else
    return CompilerDecl();
}

CompilerDeclContext DWARFDIE::GetDeclContext() const {
  DWARFASTParser *dwarf_ast = GetDWARFParser();
  if (dwarf_ast)
    return dwarf_ast->GetDeclContextForUIDFromDWARF(*this);
  else
    return CompilerDeclContext();
}

CompilerDeclContext DWARFDIE::GetContainingDeclContext() const {
  DWARFASTParser *dwarf_ast = GetDWARFParser();
  if (dwarf_ast)
    return dwarf_ast->GetDeclContextContainingUIDFromDWARF(*this);
  else
    return CompilerDeclContext();
}

bool operator==(const DWARFDIE &lhs, const DWARFDIE &rhs) {
  return lhs.GetDIE() == rhs.GetDIE() && lhs.GetCU() == rhs.GetCU();
}

bool operator!=(const DWARFDIE &lhs, const DWARFDIE &rhs) {
  return !(lhs == rhs);
}
