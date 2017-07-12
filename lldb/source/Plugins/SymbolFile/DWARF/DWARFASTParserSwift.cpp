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

#include "DWARFCompileUnit.h"
#include "DWARFDIE.h"
#include "DWARFDebugInfo.h"
#include "DWARFDefines.h"
#include "SymbolFileDWARF.h"

#include "swift/AST/ASTContext.h"

#include "lldb/Utility/Status.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/SwiftLanguageRuntime.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

DWARFASTParserSwift::DWARFASTParserSwift(SwiftASTContext &ast) : m_ast(ast) {}

DWARFASTParserSwift::~DWARFASTParserSwift() {}

lldb::TypeSP DWARFASTParserSwift::ParseTypeFromDWARF(const SymbolContext &sc,
                                                     const DWARFDIE &die,
                                                     Log *log,
                                                     bool *type_is_new_ptr) {
  lldb::TypeSP type_sp;
  CompilerType compiler_type;
  Status error;

  Declaration decl;
  ConstString mangled_name;
  ConstString name;

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
        default:
          break;
        }
      }
    }
  }

  if (!mangled_name && name) {
    if (SwiftLanguageRuntime::IsSwiftMangledName(name.GetCString()))
      mangled_name = name;
    else {
      const char *type_name_cstr = name.GetCString();
      // TODO: remove this once all mangled names are always included for all
      // types in DWARF
      swift::ModuleDecl *swift_module = m_ast.GetModule(decl.GetFile(), error);
      if (swift_module)
        compiler_type = m_ast.FindType(type_name_cstr, swift_module);

      if (!compiler_type) {
        // Anything from the swift module might be in a DW_TAG_typedef with a
        // name of "Int"
        // so we shuld also check the swift module if we fail to find our type
        // until we get
        // <rdar://problem/15290346> fixed.
        compiler_type =
            m_ast.FindFirstType(type_name_cstr, ConstString("Swift"));
      }
    }
  }

  if (mangled_name) {
    // see if we parsed this type already
    type_sp = m_ast.GetCachedType(mangled_name);
    if (type_sp)
      return type_sp;

    // otherwise figure it out yourself
    compiler_type =
        m_ast.GetTypeFromMangledTypename(mangled_name.GetCString(), error);
  }

  if (!compiler_type && name) {
    if (name.GetStringRef().startswith("$swift.") ||
        name.GetStringRef().startswith(SwiftLanguageRuntime::GetCurrentMangledName("_TtBp").c_str())) {  // This is the RawPointerType, need to figure out its name from the AST.
      swift::ASTContext *swift_ast_ctx = m_ast.GetASTContext();
      if (swift_ast_ctx)
        compiler_type =
            CompilerType(swift_ast_ctx, swift_ast_ctx->TheRawPointerType);
      else {
        if (log) {
          const char *file_name = "<unknown>";
          SymbolFile *sym_file = m_ast.GetSymbolFile();
          if (sym_file) {
            ObjectFile *obj_file = sym_file->GetObjectFile();
            if (obj_file) {
              ModuleSP module_sp = obj_file->GetModule();
              if (module_sp)
                file_name = module_sp->GetFileSpec().GetFilename().AsCString();
            }
          }
          log->Printf("Got null AST context while looking up %s in %s.",
                      name.AsCString(), file_name);
        }
        return TypeSP();
      }
    }
  }

  switch (die.Tag()) {
  case DW_TAG_inlined_subroutine:
  case DW_TAG_subprogram:
  case DW_TAG_subroutine_type:
    if (!compiler_type || !compiler_type.IsFunctionType()) {
      // Make sure we at least have some function type. The mangling for the
      // "top_level_code"
      // is currently returning the emptyTupleType (originally "_TtT_") which is not a function type...
      compiler_type = m_ast.GetVoidFunctionType();
    }
    break;
  default:
    break;
  }

  if (compiler_type) {
    type_sp = TypeSP(new Type(
        die.GetID(), die.GetDWARF(), compiler_type.GetTypeName(),
        compiler_type.GetByteSize(nullptr), NULL, LLDB_INVALID_UID,
        Type::eEncodingIsUID, &decl, compiler_type, Type::eResolveStateFull));
  }

  // cache this type
  if (type_sp && mangled_name
      && SwiftLanguageRuntime::IsSwiftMangledName(mangled_name.GetCString()))
    m_ast.SetCachedType(mangled_name, type_sp);

  return type_sp;
}

Function *DWARFASTParserSwift::ParseFunctionFromDWARF(const SymbolContext &sc,
                                                      const DWARFDIE &die) {
  DWARFRangeList func_ranges;
  const char *name = NULL;
  const char *mangled = NULL;
  int decl_file = 0;
  int decl_line = 0;
  int decl_column = 0;
  int call_file = 0;
  int call_line = 0;
  int call_column = 0;
  DWARFExpression frame_base(die.GetCU());

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
      // DW_TAG_thrown_error:
      
      bool can_throw = false;
      
      DWARFDebugInfoEntry *child(die.GetFirstChild().GetDIE());
      while (child)
      {
        if (child->Tag() == DW_TAG_thrown_type)
        {
          can_throw = true;
          break;
        }
        child = child->GetSibling();
      }

      FunctionSP func_sp;
      std::unique_ptr<Declaration> decl_ap;
      if (decl_file != 0 || decl_line != 0 || decl_column != 0)
        decl_ap.reset(new Declaration(
            sc.comp_unit->GetSupportFiles().GetFileSpecAtIndex(decl_file),
            decl_line, decl_column));

      if (dwarf->FixupAddress(func_range.GetBaseAddress())) {
        const user_id_t func_user_id = die.GetID();
        func_sp.reset(new Function(sc.comp_unit, func_user_id, func_user_id,
                                   func_name, nullptr,
                                   func_range, can_throw)); // first address range

        if (func_sp.get() != NULL) {
          if (frame_base.IsValid())
            func_sp->GetFrameBaseExpression() = frame_base;
          sc.comp_unit->AddFunction(func_sp);
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
