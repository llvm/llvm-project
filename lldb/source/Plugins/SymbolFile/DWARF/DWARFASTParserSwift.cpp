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

#include "llvm/DebugInfo/DWARF/DWARFAddressRange.h"

#include "clang/AST/DeclObjC.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;

DWARFASTParserSwift::DWARFASTParserSwift(
    TypeSystemSwiftTypeRef &swift_typesystem)
    : DWARFASTParser(Kind::DWARFASTParserSwift),
      m_swift_typesystem(swift_typesystem) {}

DWARFASTParserSwift::~DWARFASTParserSwift() {}

static llvm::StringRef GetTypedefName(const DWARFDIE &die) {
  if (die.Tag() != llvm::dwarf::DW_TAG_typedef)
    return {};
  DWARFDIE type_die =
      die.GetAttributeValueAsReferenceDIE(llvm::dwarf::DW_AT_type);
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

  std::optional<uint64_t> dwarf_byte_size;

  DWARFAttributes attributes = die.GetAttributes();
  const size_t num_attributes = attributes.Size();
  DWARFFormValue type_attr;

  if (num_attributes > 0) {
    uint32_t i;
    bool has_specification_of = false;
    for (i = 0; i < num_attributes; ++i) {
      const dw_attr_t attr = attributes.AttributeAtIndex(i);
      DWARFFormValue form_value;
      if (attributes.ExtractFormValueAtIndex(i, form_value)) {
        switch (attr) {
        case llvm::dwarf::DW_AT_decl_file:
          decl.SetFile(sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(
              form_value.Unsigned()));
          break;
        case llvm::dwarf::DW_AT_decl_line:
          decl.SetLine(form_value.Unsigned());
          break;
        case llvm::dwarf::DW_AT_decl_column:
          decl.SetColumn(form_value.Unsigned());
          break;
        case llvm::dwarf::DW_AT_name:
          name.SetCString(form_value.AsCString());
          break;
        case llvm::dwarf::DW_AT_specification:
          has_specification_of = true;
          break;
        case llvm::dwarf::DW_AT_linkage_name:
        case llvm::dwarf::DW_AT_MIPS_linkage_name: {
          mangled_name.SetCString(form_value.AsCString());
          auto HasSpecificationOf = [&](){
            if (has_specification_of)
              return true;
            for (uint32_t j = i+1; j < num_attributes; ++j)
              if (attributes.AttributeAtIndex(j) ==
                  llvm::dwarf::DW_AT_specification)
                return true;
            return false;
          };
          // Is this a sized container with a specification?  If yes,
          // the linkage name we just got is the one of the
          // specification die, which would be the unsubsituted
          // type. The child contains the linkage name of the
          // specialized type.  We should define appropriate DWARF for
          // this instead of relying on this heuristic.
          if (die.Tag() == llvm::dwarf::DW_TAG_structure_type &&
              die.HasChildren() && HasSpecificationOf()) {
            DWARFDIE member_die = die.GetFirstChild();
            if (member_die.Tag() != llvm::dwarf::DW_TAG_member ||
                member_die.GetName())
              break;
            if (DWARFDIE inner_type_die =
                    member_die.GetAttributeValueAsReferenceDIE(
                        llvm::dwarf::DW_AT_type))
              if (const char *s = inner_type_die.GetAttributeValueAsString(
                      llvm::dwarf::DW_AT_name, nullptr))
                mangled_name.SetCString(s);
          }
        } break;
        case llvm::dwarf::DW_AT_byte_size:
          dwarf_byte_size = form_value.Unsigned();
          break;
        case llvm::dwarf::DW_AT_type:
          if (die.Tag() == llvm::dwarf::DW_TAG_const_type)
            // This is how let bindings are represented. This doesn't
            // change the underlying Swift type.
            return ParseTypeFromDWARF(sc, form_value.Reference(),
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
    if (DWARFDIE type_die =
            die.GetAttributeValueAsReferenceDIE(llvm::dwarf::DW_AT_type))
      return ParseTypeFromDWARF(sc, type_die, type_is_new_ptr);
    return {};
  };

  if (!name && !mangled_name &&
      die.Tag() == llvm::dwarf::DW_TAG_structure_type) {
    // This is a sized container for a bound generic.
    return get_type(die.GetFirstChild());
  }

  if (!mangled_name && name) {
    if (name.GetStringRef() == "$swift.fixedbuffer") {
      if (auto wrapped_type = get_type(die.GetFirstChild())) {
        // Create a unique pointer for the type + fixed buffer flag.
        type_sp = wrapped_type->GetSymbolFile()->CopyType(wrapped_type);
        type_sp->SetPayload(TypePayloadSwift(true));
        return type_sp;
      }
    }
    if (SwiftLanguageRuntime::IsSwiftMangledName(name.GetStringRef())) {
      mangled_name = name;
      if (die.Tag() == llvm::dwarf::DW_TAG_typedef)
        if (TypeSP desugared_type = get_type(die)) {
          // For a typedef, store the once desugared type as the name.
          CompilerType type = desugared_type->GetForwardCompilerType();
          if (auto swift_ast_ctx =
                  type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>())
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

  if (!compiler_type && die.Tag() == llvm::dwarf::DW_TAG_typedef) {
    // Handle Archetypes, which are typedefs to RawPointerType.
    llvm::StringRef typedef_name = GetTypedefName(die);
    if (typedef_name.starts_with("$sBp")) {
      preferred_name = name;
      compiler_type = m_swift_typesystem.GetTypeFromMangledTypename(
          ConstString(typedef_name));
    } else {
      // Otherwise ignore the typedef name and resolve the pointee.
      if (TypeSP desugared_type = get_type(die)) {
        preferred_name = name;
        compiler_type = desugared_type->GetForwardCompilerType();
      }
    }
  }

  switch (die.Tag()) {
  case llvm::dwarf::DW_TAG_inlined_subroutine:
  case llvm::dwarf::DW_TAG_subprogram:
  case llvm::dwarf::DW_TAG_subroutine_type:
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
    type_sp = die.GetDWARF()->MakeType(
        die.GetID(),
        preferred_name ? preferred_name : compiler_type.GetTypeName(),
        // We don't have an exe_scope here by design, so we need to
        // read the size from DWARF.
        dwarf_byte_size, nullptr, LLDB_INVALID_UID, Type::eEncodingIsUID, &decl,
        compiler_type, Type::ResolveState::Full);
  }

  // Cache this type.
  if (type_sp && mangled_name &&
      SwiftLanguageRuntime::IsSwiftMangledName(mangled_name.GetStringRef()))
    m_swift_typesystem.SetCachedType(mangled_name, type_sp);
  die.GetDWARF()->GetDIEToType()[die.GetDIE()] = type_sp.get();

  return type_sp;
}

ConstString
DWARFASTParserSwift::ConstructDemangledNameFromDWARF(const DWARFDIE &die) {
  // FIXME: Implement me.
  return {};
}

Function *DWARFASTParserSwift::ParseFunctionFromDWARF(
    lldb_private::CompileUnit &comp_unit, const DWARFDIE &die,
    lldb_private::AddressRanges ranges) {
  llvm::DWARFAddressRangesVector unused_ranges;
  const char *name = NULL;
  const char *mangled = NULL;
  std::optional<int> decl_file = 0;
  std::optional<int> decl_line = 0;
  std::optional<int> decl_column = 0;
  std::optional<int> call_file = 0;
  std::optional<int> call_line = 0;
  std::optional<int> call_column = 0;
  DWARFExpressionList frame_base;

  if (die.Tag() != llvm::dwarf::DW_TAG_subprogram)
    return NULL;

  if (die.GetDIENamesAndRanges(name, mangled, unused_ranges, decl_file,
                               decl_line, decl_column, call_file, call_line,
                               call_column, &frame_base)) {
    // Union of all ranges in the function DIE (if the function is
    // discontiguous)

    Mangled func_name;
    if (mangled)
      func_name.SetValue(ConstString(mangled));
    else
      func_name.SetValue(ConstString(name));

    // See if this function can throw.  We can't get that from the
    // mangled name (even though the information is often there)
    // because Swift reserves the right to omit it from the name
    // if it doesn't need it.  So instead we look for the
    // DW_TAG_thrown_type:

    bool can_throw = false;

    DWARFDebugInfoEntry *child(die.GetFirstChild().GetDIE());
    while (child) {
      if (child->Tag() == llvm::dwarf::DW_TAG_thrown_type) {
        can_throw = true;
        break;
      }
      child = child->GetSibling();
    }

    FunctionSP func_sp;
    std::unique_ptr<Declaration> decl_ap;
    if (decl_file != 0 || decl_line != 0 || decl_column != 0)
      decl_ap.reset(new Declaration(
          comp_unit.GetSupportFiles().GetFileSpecAtIndex(*decl_file), *decl_line,
          *decl_column));

    const user_id_t func_user_id = die.GetID();
    bool is_generic_trampoline = die.IsGenericTrampoline();

    // The base address of the scope for any of the debugging information
    // entries listed above is given by either the DW_AT_low_pc attribute or the
    // first address in the first range entry in the list of ranges given by the
    // DW_AT_ranges attribute.
    //   -- DWARFv5, Section 2.17 Code Addresses, Ranges and Base Addresses
    //
    // If no DW_AT_entry_pc attribute is present, then the entry address is
    // assumed to be the same as the base address of the containing scope.
    //   -- DWARFv5, Section 2.18 Entry Address
    //
    // We currently don't support Debug Info Entries with
    // DW_AT_low_pc/DW_AT_entry_pc and DW_AT_ranges attributes (the latter
    // attributes are ignored even though they should be used for the address of
    // the function), but compilers also don't emit that kind of information. If
    // this becomes a problem we need to plumb these attributes separately.
    Address func_addr = ranges[0].GetBaseAddress();

    func_sp.reset(new Function(&comp_unit, func_user_id, func_user_id,
                               func_name, nullptr, std::move(func_addr),
                               std::move(ranges), can_throw,
                               is_generic_trampoline));

    if (func_sp.get() != NULL) {
      if (frame_base.IsValid())
        func_sp->GetFrameBaseExpression() = frame_base;
      comp_unit.AddFunction(func_sp);
      return func_sp.get();
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
