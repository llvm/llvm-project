//===-- DWARFASTParserSwift.cpp ---------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "DWARFASTParserSwift.h"

#include "DWARFASTParserClang.h"
#include "DWARFCompileUnit.h"
#include "DWARFDIE.h"
#include "DWARFDebugInfo.h"
#include "DWARFDefines.h"
#include "SymbolFileDWARF.h"

#include "swift/AST/ASTContext.h"
#include "swift/AST/Decl.h"
#include "swift/Demangling/Demangle.h"

#include "clang/AST/DeclObjC.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Status.h"

using namespace lldb;
using namespace lldb_private;

DWARFASTParserSwift::DWARFASTParserSwift(
    TypeSystemSwiftTypeRef &swift_typesystem)
    : m_swift_typesystem(swift_typesystem) {}

DWARFASTParserSwift::~DWARFASTParserSwift() {}

static llvm::StringRef GetTypedefName(const DWARFDIE &die) {
  if (die.Tag() != DW_TAG_typedef)
    return {};
  DWARFDIE type_die = die.GetAttributeValueAsReferenceDIE(DW_AT_type);
  if (!type_die.IsValid())
    return {};
  if (!type_die.GetName())
    return {};
  return llvm::StringRef(type_die.GetName());
}

lldb::TypeSP DWARFASTParserSwift::ParseTypeFromDWARF(const SymbolContext &sc,
                                                     const DWARFDIE &die,
                                                     bool *type_is_new_ptr) {
  lldb::TypeSP type_sp;
  CompilerType compiler_type;
  Status error;

  Declaration decl;
  ConstString mangled_name;
  ConstString name;
  ConstString preferred_name;

  llvm::Optional<uint64_t> dwarf_byte_size;

  DWARFAttributes attributes;
  const size_t num_attributes = die.GetAttributes(attributes);
  DWARFFormValue type_attr;

  if (num_attributes > 0) {
    uint32_t i;
    for (i = 0; i < num_attributes; ++i) {
      const dw_attr_t attr = attributes.AttributeAtIndex(i);
      DWARFFormValue form_value;
      if (attributes.ExtractFormValueAtIndex(i, form_value)) {
        switch (attr) {
        case DW_AT_decl_file:
          decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(
              form_value.Unsigned()));
          break;
        case DW_AT_decl_line:
          decl.SetLine(form_value.Unsigned());
          break;
        case DW_AT_decl_column:
          decl.SetColumn(form_value.Unsigned());
          break;
        case DW_AT_name:
          name.SetCString(form_value.AsCString());
          break;
        case DW_AT_linkage_name:
        case DW_AT_MIPS_linkage_name:
          mangled_name.SetCString(form_value.AsCString());
          break;
        case DW_AT_byte_size:
          dwarf_byte_size = form_value.Unsigned();
          break;
        case DW_AT_type:
          if (die.Tag() == DW_TAG_const_type)
            // This is how let bindings are represented. This doesn't
            // change the underlying Swift type.
            return ParseTypeFromDWARF(sc, die.GetReferencedDIE(attr),
                                      type_is_new_ptr);
          break;
        default:
          break;
        }
      }
    }
  }

  // Helper to retrieve the DW_AT_type as a lldb::TypeSP.
  auto get_type = [&](DWARFDIE die) -> TypeSP {
    if (DWARFDIE type_die = die.GetAttributeValueAsReferenceDIE(DW_AT_type))
      return ParseTypeFromDWARF(sc, type_die, type_is_new_ptr);
    return {};
  };

  if (!name && !mangled_name && die.Tag() == DW_TAG_structure_type) {
    // This is a sized container for a bound generic.
    return get_type(die.GetFirstChild());
  }

  if (!mangled_name && name) {
    if (name.GetStringRef().equals("$swift.fixedbuffer")) {
      if (auto wrapped_type = get_type(die.GetFirstChild())) {
        // Create a unique pointer for the type + fixed buffer flag.
        type_sp.reset(new Type(*wrapped_type));
        type_sp->SetPayload(TypePayloadSwift(true));
        return type_sp;
      }
    }
    if (SwiftLanguageRuntime::IsSwiftMangledName(name.GetStringRef())) {
      mangled_name = name;
      if (die.Tag() == DW_TAG_typedef)
        if (TypeSP desugared_type = get_type(die)) {
          // For a typedef, store the once desugared type as the name.
          CompilerType type = desugared_type->GetForwardCompilerType();
          if (auto swift_ast_ctx =
                  llvm::dyn_cast_or_null<TypeSystemSwift>(type.GetTypeSystem()))
            preferred_name =
                swift_ast_ctx->GetMangledTypeName(type.GetOpaqueQualType());
        }
    }
  }

  if (mangled_name) {
    type_sp = m_swift_typesystem.GetCachedType(mangled_name);
    if (type_sp)
      return type_sp;

    // Because of DWARFImporter, we may search for this type again while
    // resolving the mangled name.
    die.GetDWARF()->GetDIEToType()[die.GetDIE()] = DIE_IS_BEING_PARSED;

    // Try to import the type from one of the loaded Swift modules.
    if (SwiftLanguageRuntime::IsSwiftMangledName(mangled_name.GetCString()))
      compiler_type =
          m_swift_typesystem.GetTypeFromMangledTypename(mangled_name);
  }

  if (!compiler_type && name) {
    // Handle Archetypes, which are typedefs to RawPointerType.
    llvm::StringRef typedef_name = GetTypedefName(die);
    if (typedef_name.startswith("$sBp")) {
      preferred_name = name;
      compiler_type = m_swift_typesystem.GetTypeFromMangledTypename(
          ConstString(typedef_name));
    }
  }

  switch (die.Tag()) {
  case DW_TAG_inlined_subroutine:
  case DW_TAG_subprogram:
  case DW_TAG_subroutine_type:
    if (!compiler_type || !compiler_type.IsFunctionType()) {
      // Make sure we at least have some function type. The mangling for
      // the "top_level_code" is returning the empty tuple type "()",
      // which is not a function type.
      compiler_type = m_swift_typesystem.GetVoidFunctionType();
    }
    break;
  default:
    break;
  }

  if (compiler_type) {
    type_sp = TypeSP(new Type(
        die.GetID(), die.GetDWARF(),
        preferred_name ? preferred_name : compiler_type.GetTypeName(),
        // We don't have an exe_scope here by design, so we need to
        // read the size from DWARF.
        dwarf_byte_size, nullptr, LLDB_INVALID_UID, Type::eEncodingIsUID, &decl,
        compiler_type, Type::ResolveState::Full));
  }

  // Cache this type.
  if (type_sp && mangled_name &&
      SwiftLanguageRuntime::IsSwiftMangledName(mangled_name.GetStringRef()))
    m_swift_typesystem.SetCachedType(mangled_name, type_sp);
  die.GetDWARF()->GetDIEToType()[die.GetDIE()] = type_sp.get();

  return type_sp;
}

Function *DWARFASTParserSwift::ParseFunctionFromDWARF(
    lldb_private::CompileUnit &comp_unit, const DWARFDIE &die) {
  DWARFRangeList func_ranges;
  const char *name = NULL;
  const char *mangled = NULL;
  int decl_file = 0;
  int decl_line = 0;
  int decl_column = 0;
  int call_file = 0;
  int call_line = 0;
  int call_column = 0;
  DWARFExpression frame_base;

  if (die.Tag() != DW_TAG_subprogram)
    return NULL;

  if (die.GetDIENamesAndRanges(name, mangled, func_ranges, decl_file, decl_line,
                               decl_column, call_file, call_line, call_column,
                               &frame_base)) {
    // Union of all ranges in the function DIE (if the function is
    // discontiguous)
    SymbolFileDWARF *dwarf = die.GetDWARF();
    AddressRange func_range;
    lldb::addr_t lowest_func_addr = func_ranges.GetMinRangeBase(0);
    lldb::addr_t highest_func_addr = func_ranges.GetMaxRangeEnd(0);
    if (lowest_func_addr != LLDB_INVALID_ADDRESS &&
        lowest_func_addr <= highest_func_addr) {
      ModuleSP module_sp(dwarf->GetObjectFile()->GetModule());
      func_range.GetBaseAddress().ResolveAddressUsingFileSections(
          lowest_func_addr, module_sp->GetSectionList());
      if (func_range.GetBaseAddress().IsValid())
        func_range.SetByteSize(highest_func_addr - lowest_func_addr);
    }

    if (func_range.GetBaseAddress().IsValid()) {
      Mangled func_name;
      if (mangled)
        func_name.SetValue(ConstString(mangled), true);
      else
        func_name.SetValue(ConstString(name), false);

      // See if this function can throw.  We can't get that from the
      // mangled name (even though the information is often there)
      // because Swift reserves the right to omit it from the name
      // if it doesn't need it.  So instead we look for the
      // DW_TAG_thrown_type:

      bool can_throw = false;

      DWARFDebugInfoEntry *child(die.GetFirstChild().GetDIE());
      while (child) {
        if (child->Tag() == DW_TAG_thrown_type) {
          can_throw = true;
          break;
        }
        child = child->GetSibling();
      }

      FunctionSP func_sp;
      std::unique_ptr<Declaration> decl_ap;
      if (decl_file != 0 || decl_line != 0 || decl_column != 0)
        decl_ap.reset(new Declaration(
            comp_unit.GetSupportFiles().GetFileSpecAtIndex(decl_file),
            decl_line, decl_column));

      if (dwarf->FixupAddress(func_range.GetBaseAddress())) {
        const user_id_t func_user_id = die.GetID();
        func_sp.reset(new Function(&comp_unit, func_user_id, func_user_id,
                                   func_name, nullptr, func_range,
                                   can_throw)); // first address range

        if (func_sp.get() != NULL) {
          if (frame_base.IsValid())
            func_sp->GetFrameBaseExpression() = frame_base;
          comp_unit.AddFunction(func_sp);
          return func_sp.get();
        }
      }
    }
  }
  return NULL;
}

lldb_private::CompilerDeclContext
DWARFASTParserSwift::GetDeclContextForUIDFromDWARF(const DWARFDIE &die) {
  return CompilerDeclContext();
}

lldb_private::CompilerDeclContext
DWARFASTParserSwift::GetDeclContextContainingUIDFromDWARF(const DWARFDIE &die) {
  return CompilerDeclContext();
}
