//===-- DWARFASTParserJava.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFASTParserJava.h"
#include "DWARFAttribute.h"
#include "DWARFCompileUnit.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFDeclContext.h"
#include "SymbolFileDWARF.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContextScope.h"
#include "lldb/Symbol/TypeList.h"

using namespace lldb;
using namespace lldb_private;

DWARFASTParserJava::DWARFASTParserJava(JavaASTContext &ast) : m_ast(ast) {}

DWARFASTParserJava::~DWARFASTParserJava() {}

TypeSP DWARFASTParserJava::ParseBaseTypeFromDIE(const DWARFDIE &die) {
  SymbolFileDWARF *dwarf = die.GetDWARF();
  dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;

  ConstString type_name;
  uint64_t byte_size = 0;

  DWARFAttributes attributes;
  const size_t num_attributes = die.GetAttributes(attributes);
  for (uint32_t i = 0; i < num_attributes; ++i) {
    DWARFFormValue form_value;
    dw_attr_t attr = attributes.AttributeAtIndex(i);
    if (attributes.ExtractFormValueAtIndex(i, form_value)) {
      switch (attr) {
      case DW_AT_name:
        type_name.SetCString(form_value.AsCString());
        break;
      case DW_AT_byte_size:
        byte_size = form_value.Unsigned();
        break;
      case DW_AT_encoding:
        break;
      default:
        assert(false && "Unsupported attribute for DW_TAG_base_type");
      }
    }
  }

  Declaration decl;
  CompilerType compiler_type = m_ast.CreateBaseType(type_name);
  return std::make_shared<Type>(die.GetID(), dwarf, type_name, byte_size,
                                nullptr, LLDB_INVALID_UID, Type::eEncodingIsUID,
                                decl, compiler_type, Type::eResolveStateFull);
}

TypeSP DWARFASTParserJava::ParseArrayTypeFromDIE(const DWARFDIE &die) {
  SymbolFileDWARF *dwarf = die.GetDWARF();
  dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;

  ConstString linkage_name;
  DWARFFormValue type_attr_value;
  lldb::addr_t data_offset = LLDB_INVALID_ADDRESS;
  DWARFExpression length_expression(die.GetCU());

  DWARFAttributes attributes;
  const size_t num_attributes = die.GetAttributes(attributes);
  for (uint32_t i = 0; i < num_attributes; ++i) {
    DWARFFormValue form_value;
    dw_attr_t attr = attributes.AttributeAtIndex(i);
    if (attributes.ExtractFormValueAtIndex(i, form_value)) {
      switch (attr) {
      case DW_AT_linkage_name:
        linkage_name.SetCString(form_value.AsCString());
        break;
      case DW_AT_type:
        type_attr_value = form_value;
        break;
      case DW_AT_data_member_location:
        data_offset = form_value.Unsigned();
        break;
      case DW_AT_declaration:
        break;
      default:
        assert(false && "Unsupported attribute for DW_TAG_array_type");
      }
    }
  }

  for (DWARFDIE child_die = die.GetFirstChild(); child_die.IsValid();
       child_die = child_die.GetSibling()) {
    if (child_die.Tag() == DW_TAG_subrange_type) {
      DWARFAttributes attributes;
      const size_t num_attributes = child_die.GetAttributes(attributes);
      for (uint32_t i = 0; i < num_attributes; ++i) {
        DWARFFormValue form_value;
        dw_attr_t attr = attributes.AttributeAtIndex(i);
        if (attributes.ExtractFormValueAtIndex(i, form_value)) {
          switch (attr) {
          case DW_AT_count:
            if (form_value.BlockData())
              length_expression.CopyOpcodeData(
                  form_value.BlockData(), form_value.Unsigned(),
                  child_die.GetCU()->GetByteOrder(),
                  child_die.GetCU()->GetAddressByteSize());
            break;
          default:
            assert(false && "Unsupported attribute for DW_TAG_subrange_type");
          }
        }
      }
    } else {
      assert(false && "Unsupported child for DW_TAG_array_type");
    }
  }

  DIERef type_die_ref(type_attr_value);
  Type *element_type = dwarf->ResolveTypeUID(type_die_ref);
  if (!element_type)
    return nullptr;

  CompilerType element_compiler_type = element_type->GetForwardCompilerType();
  CompilerType array_compiler_type = m_ast.CreateArrayType(
      linkage_name, element_compiler_type, length_expression, data_offset);

  Declaration decl;
  TypeSP type_sp(new Type(die.GetID(), dwarf, array_compiler_type.GetTypeName(),
                          -1, nullptr, type_die_ref.GetUID(dwarf),
                          Type::eEncodingIsUID, &decl, array_compiler_type,
                          Type::eResolveStateFull));
  type_sp->SetEncodingType(element_type);
  return type_sp;
}

TypeSP DWARFASTParserJava::ParseReferenceTypeFromDIE(const DWARFDIE &die) {
  SymbolFileDWARF *dwarf = die.GetDWARF();
  dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;

  Declaration decl;
  DWARFFormValue type_attr_value;

  DWARFAttributes attributes;
  const size_t num_attributes = die.GetAttributes(attributes);
  for (uint32_t i = 0; i < num_attributes; ++i) {
    DWARFFormValue form_value;
    dw_attr_t attr = attributes.AttributeAtIndex(i);
    if (attributes.ExtractFormValueAtIndex(i, form_value)) {
      switch (attr) {
      case DW_AT_type:
        type_attr_value = form_value;
        break;
      default:
        assert(false && "Unsupported attribute for DW_TAG_array_type");
      }
    }
  }

  DIERef type_die_ref(type_attr_value);
  Type *pointee_type = dwarf->ResolveTypeUID(type_die_ref);
  if (!pointee_type)
    return nullptr;

  CompilerType pointee_compiler_type = pointee_type->GetForwardCompilerType();
  CompilerType reference_compiler_type =
      m_ast.CreateReferenceType(pointee_compiler_type);
  TypeSP type_sp(
      new Type(die.GetID(), dwarf, reference_compiler_type.GetTypeName(), -1,
               nullptr, type_die_ref.GetUID(dwarf), Type::eEncodingIsUID, &decl,
               reference_compiler_type, Type::eResolveStateFull));
  type_sp->SetEncodingType(pointee_type);
  return type_sp;
}

lldb::TypeSP DWARFASTParserJava::ParseClassTypeFromDIE(const DWARFDIE &die,
                                                       bool &is_new_type) {
  SymbolFileDWARF *dwarf = die.GetDWARF();
  dwarf->m_die_to_type[die.GetDIE()] = DIE_IS_BEING_PARSED;

  Declaration decl;
  ConstString name;
  ConstString linkage_name;
  bool is_forward_declaration = false;
  uint32_t byte_size = 0;

  DWARFAttributes attributes;
  const size_t num_attributes = die.GetAttributes(attributes);
  for (uint32_t i = 0; i < num_attributes; ++i) {
    DWARFFormValue form_value;
    dw_attr_t attr = attributes.AttributeAtIndex(i);
    if (attributes.ExtractFormValueAtIndex(i, form_value)) {
      switch (attr) {
      case DW_AT_name:
        name.SetCString(form_value.AsCString());
        break;
      case DW_AT_declaration:
        is_forward_declaration = form_value.Boolean();
        break;
      case DW_AT_byte_size:
        byte_size = form_value.Unsigned();
        break;
      case DW_AT_linkage_name:
        linkage_name.SetCString(form_value.AsCString());
        break;
      default:
        assert(false && "Unsupported attribute for DW_TAG_class_type");
      }
    }
  }

  UniqueDWARFASTType unique_ast_entry;
  if (name) {
    std::string qualified_name;
    if (die.GetQualifiedName(qualified_name)) {
      name.SetCString(qualified_name.c_str());
      if (dwarf->GetUniqueDWARFASTTypeMap().Find(name, die, Declaration(), -1,
                                                 unique_ast_entry)) {
        if (unique_ast_entry.m_type_sp) {
          dwarf->GetDIEToType()[die.GetDIE()] =
              unique_ast_entry.m_type_sp.get();
          is_new_type = false;
          return unique_ast_entry.m_type_sp;
        }
      }
    }
  }

  if (is_forward_declaration) {
    DWARFDeclContext die_decl_ctx;
    die.GetDWARFDeclContext(die_decl_ctx);

    TypeSP type_sp = dwarf->FindDefinitionTypeForDWARFDeclContext(die_decl_ctx);
    if (type_sp) {
      // We found a real definition for this type elsewhere so lets use it
      dwarf->GetDIEToType()[die.GetDIE()] = type_sp.get();
      is_new_type = false;
      return type_sp;
    }
  }

  CompilerType compiler_type(
      &m_ast, dwarf->GetForwardDeclDieToClangType().lookup(die.GetDIE()));
  if (!compiler_type)
    compiler_type = m_ast.CreateObjectType(name, linkage_name, byte_size);

  is_new_type = true;
  TypeSP type_sp(new Type(die.GetID(), dwarf, name,
                          -1, // byte size isn't specified
                          nullptr, LLDB_INVALID_UID, Type::eEncodingIsUID,
                          &decl, compiler_type, Type::eResolveStateForward));

  // Add our type to the unique type map
  unique_ast_entry.m_type_sp = type_sp;
  unique_ast_entry.m_die = die;
  unique_ast_entry.m_declaration = decl;
  unique_ast_entry.m_byte_size = -1;
  dwarf->GetUniqueDWARFASTTypeMap().Insert(name, unique_ast_entry);

  if (!is_forward_declaration) {
    // Leave this as a forward declaration until we need to know the details of
    // the type
    dwarf->GetForwardDeclDieToClangType()[die.GetDIE()] =
        compiler_type.GetOpaqueQualType();
    dwarf->GetForwardDeclClangTypeToDie()[compiler_type.GetOpaqueQualType()] =
        die.GetDIERef();
  }
  return type_sp;
}

lldb::TypeSP DWARFASTParserJava::ParseTypeFromDWARF(
    const lldb_private::SymbolContext &sc, const DWARFDIE &die,
    lldb_private::Log *log, bool *type_is_new_ptr) {
  if (type_is_new_ptr)
    *type_is_new_ptr = false;

  if (!die)
    return nullptr;

  SymbolFileDWARF *dwarf = die.GetDWARF();

  Type *type_ptr = dwarf->m_die_to_type.lookup(die.GetDIE());
  if (type_ptr == DIE_IS_BEING_PARSED)
    return nullptr;
  if (type_ptr != nullptr)
    return type_ptr->shared_from_this();

  TypeSP type_sp;
  if (type_is_new_ptr)
    *type_is_new_ptr = true;

  switch (die.Tag()) {
  case DW_TAG_base_type: {
    type_sp = ParseBaseTypeFromDIE(die);
    break;
  }
  case DW_TAG_array_type: {
    type_sp = ParseArrayTypeFromDIE(die);
    break;
  }
  case DW_TAG_class_type: {
    bool is_new_type = false;
    type_sp = ParseClassTypeFromDIE(die, is_new_type);
    if (!is_new_type)
      return type_sp;
    break;
  }
  case DW_TAG_reference_type: {
    type_sp = ParseReferenceTypeFromDIE(die);
    break;
  }
  }

  if (!type_sp)
    return nullptr;

  DWARFDIE sc_parent_die = SymbolFileDWARF::GetParentSymbolContextDIE(die);
  dw_tag_t sc_parent_tag = sc_parent_die.Tag();

  SymbolContextScope *symbol_context_scope = nullptr;
  if (sc_parent_tag == DW_TAG_compile_unit) {
    symbol_context_scope = sc.comp_unit;
  } else if (sc.function != nullptr && sc_parent_die) {
    symbol_context_scope =
        sc.function->GetBlock(true).FindBlockByID(sc_parent_die.GetID());
    if (symbol_context_scope == nullptr)
      symbol_context_scope = sc.function;
  }

  if (symbol_context_scope != nullptr)
    type_sp->SetSymbolContextScope(symbol_context_scope);

  dwarf->GetTypeList()->Insert(type_sp);
  dwarf->m_die_to_type[die.GetDIE()] = type_sp.get();

  return type_sp;
}

lldb_private::Function *DWARFASTParserJava::ParseFunctionFromDWARF(
    const lldb_private::SymbolContext &sc, const DWARFDIE &die) {
  assert(die.Tag() == DW_TAG_subprogram);

  const char *name = nullptr;
  const char *mangled = nullptr;
  int decl_file = 0;
  int decl_line = 0;
  int decl_column = 0;
  int call_file = 0;
  int call_line = 0;
  int call_column = 0;
  DWARFRangeList func_ranges;
  DWARFExpression frame_base(die.GetCU());

  if (die.GetDIENamesAndRanges(name, mangled, func_ranges, decl_file, decl_line,
                               decl_column, call_file, call_line, call_column,
                               &frame_base)) {
    // Union of all ranges in the function DIE (if the function is
    // discontiguous)
    AddressRange func_range;
    lldb::addr_t lowest_func_addr = func_ranges.GetMinRangeBase(0);
    lldb::addr_t highest_func_addr = func_ranges.GetMaxRangeEnd(0);
    if (lowest_func_addr != LLDB_INVALID_ADDRESS &&
        lowest_func_addr <= highest_func_addr) {
      ModuleSP module_sp(die.GetModule());
      func_range.GetBaseAddress().ResolveAddressUsingFileSections(
          lowest_func_addr, module_sp->GetSectionList());
      if (func_range.GetBaseAddress().IsValid())
        func_range.SetByteSize(highest_func_addr - lowest_func_addr);
    }

    if (func_range.GetBaseAddress().IsValid()) {
      std::unique_ptr<Declaration> decl_ap;
      if (decl_file != 0 || decl_line != 0 || decl_column != 0)
        decl_ap.reset(new Declaration(
            sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(decl_file),
            decl_line, decl_column));

      if (die.GetDWARF()->FixupAddress(func_range.GetBaseAddress())) {
        FunctionSP func_sp(new Function(sc.comp_unit, die.GetID(), die.GetID(),
                                        Mangled(ConstString(name), false),
                                        nullptr, // No function types in java
                                        func_range));
        if (frame_base.IsValid())
          func_sp->GetFrameBaseExpression() = frame_base;
        sc.comp_unit->AddFunction(func_sp);

        return func_sp.get();
      }
    }
  }
  return nullptr;
}

bool DWARFASTParserJava::CompleteTypeFromDWARF(
    const DWARFDIE &die, lldb_private::Type *type,
    lldb_private::CompilerType &java_type) {
  switch (die.Tag()) {
  case DW_TAG_class_type: {
    if (die.GetAttributeValueAsUnsigned(DW_AT_declaration, 0) == 0) {
      if (die.HasChildren())
        ParseChildMembers(die, java_type);
      m_ast.CompleteObjectType(java_type);
      return java_type.IsValid();
    }
  } break;
  default:
    assert(false && "Not a forward java type declaration!");
    break;
  }
  return false;
}

void DWARFASTParserJava::ParseChildMembers(const DWARFDIE &parent_die,
                                           CompilerType &compiler_type) {
  DWARFCompileUnit *dwarf_cu = parent_die.GetCU();
  for (DWARFDIE die = parent_die.GetFirstChild(); die.IsValid();
       die = die.GetSibling()) {
    switch (die.Tag()) {
    case DW_TAG_member: {
      const char *name = nullptr;
      DWARFFormValue encoding_uid;
      uint32_t member_byte_offset = UINT32_MAX;
      DWARFExpression member_location_expression(dwarf_cu);

      DWARFAttributes attributes;
      size_t num_attributes = die.GetAttributes(attributes);
      for (size_t i = 0; i < num_attributes; ++i) {
        DWARFFormValue form_value;
        if (attributes.ExtractFormValueAtIndex(i, form_value)) {
          switch (attributes.AttributeAtIndex(i)) {
          case DW_AT_name:
            name = form_value.AsCString();
            break;
          case DW_AT_type:
            encoding_uid = form_value;
            break;
          case DW_AT_data_member_location:
            if (form_value.BlockData())
              member_location_expression.CopyOpcodeData(
                  form_value.BlockData(), form_value.Unsigned(),
                  dwarf_cu->GetByteOrder(), dwarf_cu->GetAddressByteSize());
            else
              member_byte_offset = form_value.Unsigned();
            break;
          case DW_AT_artificial:
            static_cast<void>(form_value.Boolean());
            break;
          case DW_AT_accessibility:
            // TODO: Handle when needed
            break;
          default:
            assert(false && "Unhandled attribute for DW_TAG_member");
            break;
          }
        }
      }

      if (strcmp(name, ".dynamic_type") == 0)
        m_ast.SetDynamicTypeId(compiler_type, member_location_expression);
      else {
        if (Type *member_type = die.ResolveTypeUID(DIERef(encoding_uid)))
          m_ast.AddMemberToObject(compiler_type, ConstString(name),
                                  member_type->GetFullCompilerType(),
                                  member_byte_offset);
      }
      break;
    }
    case DW_TAG_inheritance: {
      DWARFFormValue encoding_uid;
      uint32_t member_byte_offset = UINT32_MAX;

      DWARFAttributes attributes;
      size_t num_attributes = die.GetAttributes(attributes);
      for (size_t i = 0; i < num_attributes; ++i) {
        DWARFFormValue form_value;
        if (attributes.ExtractFormValueAtIndex(i, form_value)) {
          switch (attributes.AttributeAtIndex(i)) {
          case DW_AT_type:
            encoding_uid = form_value;
            break;
          case DW_AT_data_member_location:
            member_byte_offset = form_value.Unsigned();
            break;
          case DW_AT_accessibility:
            // In java all base class is public so we can ignore this attribute
            break;
          default:
            assert(false && "Unhandled attribute for DW_TAG_member");
            break;
          }
        }
      }
      if (Type *base_type = die.ResolveTypeUID(DIERef(encoding_uid)))
        m_ast.AddBaseClassToObject(compiler_type,
                                   base_type->GetFullCompilerType(),
                                   member_byte_offset);
      break;
    }
    default:
      break;
    }
  }
}
