//===-- DWARFASTParserClang.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFASTPARSERCLANG_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFASTPARSERCLANG_H

#include "clang/AST/CharUnits.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "DWARFASTParser.h"
#include "DWARFDIE.h"
#include "DWARFDefines.h"
#include "DWARFFormValue.h"
#include "LogChannelDWARF.h"
#include "lldb/Core/PluginInterface.h"

#include "Plugins/ExpressionParser/Clang/ClangASTImporter.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include <vector>

namespace lldb_private {
class CompileUnit;
}
class DWARFDebugInfoEntry;
class SymbolFileDWARF;

struct ParsedDWARFTypeAttributes;

class DWARFASTParserClang : public DWARFASTParser {
public:
  DWARFASTParserClang(lldb_private::TypeSystemClang &ast);

  ~DWARFASTParserClang() override;

  // DWARFASTParser interface.
  lldb::TypeSP ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                                  const DWARFDIE &die,
                                  bool *type_is_new_ptr) override;

  lldb_private::Function *
  ParseFunctionFromDWARF(lldb_private::CompileUnit &comp_unit,
                         const DWARFDIE &die) override;

  bool
  CompleteTypeFromDWARF(const DWARFDIE &die, lldb_private::Type *type,
                        lldb_private::CompilerType &compiler_type) override;

  lldb_private::CompilerDecl
  GetDeclForUIDFromDWARF(const DWARFDIE &die) override;

  void EnsureAllDIEsInDeclContextHaveBeenParsed(
      lldb_private::CompilerDeclContext decl_context) override;

  lldb_private::CompilerDeclContext
  GetDeclContextForUIDFromDWARF(const DWARFDIE &die) override;

  lldb_private::CompilerDeclContext
  GetDeclContextContainingUIDFromDWARF(const DWARFDIE &die) override;

  lldb_private::ClangASTImporter &GetClangASTImporter();

protected:
  /// Protected typedefs and members.
  /// @{
  class DelayedAddObjCClassProperty;
  typedef std::vector<DelayedAddObjCClassProperty> DelayedPropertyList;

  typedef llvm::SmallPtrSet<const DWARFDebugInfoEntry *, 4> DIEPointerSet;
  typedef llvm::DenseMap<const DWARFDebugInfoEntry *, clang::DeclContext *>
      DIEToDeclContextMap;
  typedef std::multimap<const clang::DeclContext *, const DWARFDIE>
      DeclContextToDIEMap;
  typedef llvm::DenseMap<const DWARFDebugInfoEntry *,
                         lldb_private::OptionalClangModuleID>
      DIEToModuleMap;
  typedef llvm::DenseMap<const DWARFDebugInfoEntry *, clang::Decl *>
      DIEToDeclMap;
  typedef llvm::DenseMap<const clang::Decl *, DIEPointerSet> DeclToDIEMap;

  lldb_private::TypeSystemClang &m_ast;
  DIEToDeclMap m_die_to_decl;
  DeclToDIEMap m_decl_to_die;
  DIEToDeclContextMap m_die_to_decl_ctx;
  DeclContextToDIEMap m_decl_ctx_to_die;
  DIEToModuleMap m_die_to_module;
  std::unique_ptr<lldb_private::ClangASTImporter> m_clang_ast_importer_up;
  /// @}

  clang::DeclContext *GetDeclContextForBlock(const DWARFDIE &die);

  clang::BlockDecl *ResolveBlockDIE(const DWARFDIE &die);

  clang::NamespaceDecl *ResolveNamespaceDIE(const DWARFDIE &die);

  bool ParseTemplateDIE(const DWARFDIE &die,
                        lldb_private::TypeSystemClang::TemplateParameterInfos
                            &template_param_infos);
  bool ParseTemplateParameterInfos(
      const DWARFDIE &parent_die,
      lldb_private::TypeSystemClang::TemplateParameterInfos
          &template_param_infos);

  bool ParseChildMembers(
      const DWARFDIE &die, lldb_private::CompilerType &class_compiler_type,
      std::vector<std::unique_ptr<clang::CXXBaseSpecifier>> &base_classes,
      std::vector<DWARFDIE> &member_function_dies,
      DelayedPropertyList &delayed_properties,
      lldb::AccessType &default_accessibility,
      lldb_private::ClangASTImporter::LayoutInfo &layout_info);

  size_t
  ParseChildParameters(clang::DeclContext *containing_decl_ctx,
                       const DWARFDIE &parent_die, bool skip_artificial,
                       bool &is_static, bool &is_variadic,
                       bool &has_template_params,
                       std::vector<lldb_private::CompilerType> &function_args,
                       std::vector<clang::ParmVarDecl *> &function_param_decls,
                       unsigned &type_quals);

  size_t ParseChildEnumerators(lldb_private::CompilerType &compiler_type,
                               bool is_signed, uint32_t enumerator_byte_size,
                               const DWARFDIE &parent_die);

  /// Parse a structure, class, or union type DIE.
  lldb::TypeSP ParseStructureLikeDIE(const lldb_private::SymbolContext &sc,
                                     const DWARFDIE &die,
                                     ParsedDWARFTypeAttributes &attrs);

  lldb_private::Type *GetTypeForDIE(const DWARFDIE &die);

  clang::Decl *GetClangDeclForDIE(const DWARFDIE &die);

  clang::DeclContext *GetClangDeclContextForDIE(const DWARFDIE &die);

  clang::DeclContext *GetClangDeclContextContainingDIE(const DWARFDIE &die,
                                                       DWARFDIE *decl_ctx_die);
  lldb_private::OptionalClangModuleID GetOwningClangModule(const DWARFDIE &die);

  bool CopyUniqueClassMethodTypes(const DWARFDIE &src_class_die,
                                  const DWARFDIE &dst_class_die,
                                  lldb_private::Type *class_type,
                                  std::vector<DWARFDIE> &failures);

  clang::DeclContext *GetCachedClangDeclContextForDIE(const DWARFDIE &die);

  void LinkDeclContextToDIE(clang::DeclContext *decl_ctx, const DWARFDIE &die);

  void LinkDeclToDIE(clang::Decl *decl, const DWARFDIE &die);

  /// If \p type_sp is valid, calculate and set its symbol context scope, and
  /// update the type list for its backing symbol file.
  ///
  /// Returns \p type_sp.
  lldb::TypeSP
  UpdateSymbolContextScopeForType(const lldb_private::SymbolContext &sc,
                                  const DWARFDIE &die, lldb::TypeSP type_sp);

  /// Follow Clang Module Skeleton CU references to find a type definition.
  lldb::TypeSP ParseTypeFromClangModule(const lldb_private::SymbolContext &sc,
                                        const DWARFDIE &die,
                                        lldb_private::Log *log);

  // Return true if this type is a declaration to a type in an external
  // module.
  lldb::ModuleSP GetModuleForType(const DWARFDIE &die);

private:
  struct FieldInfo {
    uint64_t bit_size = 0;
    uint64_t bit_offset = 0;
    bool is_bitfield = false;

    FieldInfo() = default;

    void SetIsBitfield(bool flag) { is_bitfield = flag; }
    bool IsBitfield() { return is_bitfield; }

    bool NextBitfieldOffsetIsValid(const uint64_t next_bit_offset) const {
      // Any subsequent bitfields must not overlap and must be at a higher
      // bit offset than any previous bitfield + size.
      return (bit_size + bit_offset) <= next_bit_offset;
    }
  };

  void
  ParseSingleMember(const DWARFDIE &die, const DWARFDIE &parent_die,
                    const lldb_private::CompilerType &class_clang_type,
                    lldb::AccessType default_accessibility,
                    DelayedPropertyList &delayed_properties,
                    lldb_private::ClangASTImporter::LayoutInfo &layout_info,
                    FieldInfo &last_field_info);

  bool CompleteRecordType(const DWARFDIE &die, lldb_private::Type *type,
                          lldb_private::CompilerType &clang_type);
  bool CompleteEnumType(const DWARFDIE &die, lldb_private::Type *type,
                        lldb_private::CompilerType &clang_type);

  lldb::TypeSP ParseTypeModifier(const lldb_private::SymbolContext &sc,
                                 const DWARFDIE &die,
                                 ParsedDWARFTypeAttributes &attrs);
  lldb::TypeSP ParseEnum(const lldb_private::SymbolContext &sc,
                         const DWARFDIE &die, ParsedDWARFTypeAttributes &attrs);
  lldb::TypeSP ParseSubroutine(const DWARFDIE &die,
                               ParsedDWARFTypeAttributes &attrs);
  // FIXME: attrs should be passed as a const reference.
  lldb::TypeSP ParseArrayType(const DWARFDIE &die,
                              ParsedDWARFTypeAttributes &attrs);
  lldb::TypeSP ParsePointerToMemberType(const DWARFDIE &die,
                                        const ParsedDWARFTypeAttributes &attrs);
};

/// Parsed form of all attributes that are relevant for type reconstruction.
/// Some attributes are relevant for all kinds of types (declaration), while
/// others are only meaningful to a specific type (is_virtual)
struct ParsedDWARFTypeAttributes {
  explicit ParsedDWARFTypeAttributes(const DWARFDIE &die);

  lldb::AccessType accessibility = lldb::eAccessNone;
  bool is_artificial = false;
  bool is_complete_objc_class = false;
  bool is_explicit = false;
  bool is_forward_declaration = false;
  bool is_inline = false;
  bool is_scoped_enum = false;
  bool is_vector = false;
  bool is_virtual = false;
  bool is_objc_direct_call = false;
  bool exports_symbols = false;
  clang::StorageClass storage = clang::SC_None;
  const char *mangled_name = nullptr;
  lldb_private::ConstString name;
  lldb_private::Declaration decl;
  DWARFDIE object_pointer;
  DWARFFormValue abstract_origin;
  DWARFFormValue containing_type;
  DWARFFormValue signature;
  DWARFFormValue specification;
  DWARFFormValue type;
  lldb::LanguageType class_language = lldb::eLanguageTypeUnknown;
  llvm::Optional<uint64_t> byte_size;
  size_t calling_convention = llvm::dwarf::DW_CC_normal;
  uint32_t bit_stride = 0;
  uint32_t byte_stride = 0;
  uint32_t encoding = 0;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFASTPARSERCLANG_H
