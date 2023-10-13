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
#include "clang/AST/Type.h"
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

#include <optional>
#include <vector>

namespace lldb_private {
class CompileUnit;
}
namespace lldb_private::plugin {
namespace dwarf {
class DWARFDebugInfoEntry;
class SymbolFileDWARF;
} // namespace dwarf
} // namespace lldb_private::plugin

struct ParsedDWARFTypeAttributes;

class DWARFASTParserClang : public lldb_private::plugin::dwarf::DWARFASTParser {
public:
  DWARFASTParserClang(lldb_private::TypeSystemClang &ast);

  ~DWARFASTParserClang() override;

  // DWARFASTParser interface.
  lldb::TypeSP
  ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                     const lldb_private::plugin::dwarf::DWARFDIE &die,
                     bool *type_is_new_ptr) override;

  lldb_private::ConstString ConstructDemangledNameFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override;

  lldb_private::Function *
  ParseFunctionFromDWARF(lldb_private::CompileUnit &comp_unit,
                         const lldb_private::plugin::dwarf::DWARFDIE &die,
                         const lldb_private::AddressRange &func_range) override;

  bool
  CompleteTypeFromDWARF(const lldb_private::plugin::dwarf::DWARFDIE &die,
                        lldb_private::Type *type,
                        lldb_private::CompilerType &compiler_type) override;

  lldb_private::CompilerDecl GetDeclForUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override;

  void EnsureAllDIEsInDeclContextHaveBeenParsed(
      lldb_private::CompilerDeclContext decl_context) override;

  lldb_private::CompilerDeclContext GetDeclContextForUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override;

  lldb_private::CompilerDeclContext GetDeclContextContainingUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override;

  lldb_private::ClangASTImporter &GetClangASTImporter();

  /// Extracts an value for a given Clang integer type from a DWARFFormValue.
  ///
  /// \param int_type The Clang type that defines the bit size and signedness
  ///                 of the integer that should be extracted. Has to be either
  ///                 an integer type or an enum type. For enum types the
  ///                 underlying integer type will be considered as the
  ///                 expected integer type that should be extracted.
  /// \param form_value The DWARFFormValue that contains the integer value.
  /// \return An APInt containing the same integer value as the given
  ///         DWARFFormValue with the bit width of the given integer type.
  ///         Returns an error if the value in the DWARFFormValue does not fit
  ///         into the given integer type or the integer type isn't supported.
  llvm::Expected<llvm::APInt> ExtractIntFromFormValue(
      const lldb_private::CompilerType &int_type,
      const lldb_private::plugin::dwarf::DWARFFormValue &form_value) const;

  /// Returns the template parameters of a class DWARFDIE as a string.
  ///
  /// This is mostly useful for -gsimple-template-names which omits template
  /// parameters from the DIE name and instead always adds template parameter
  /// children DIEs.
  ///
  /// \param die The struct/class DWARFDIE containing template parameters.
  /// \return A string, including surrounding '<>', of the template parameters.
  /// If the DIE's name already has '<>', returns an empty ConstString because
  /// it's assumed that the caller is using the DIE name anyway.
  lldb_private::ConstString GetDIEClassTemplateParams(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override;

protected:
  /// Protected typedefs and members.
  /// @{
  class DelayedAddObjCClassProperty;
  typedef std::vector<DelayedAddObjCClassProperty> DelayedPropertyList;

  typedef llvm::DenseMap<
      const lldb_private::plugin::dwarf::DWARFDebugInfoEntry *,
      clang::DeclContext *>
      DIEToDeclContextMap;
  typedef std::multimap<const clang::DeclContext *,
                        const lldb_private::plugin::dwarf::DWARFDIE>
      DeclContextToDIEMap;
  typedef llvm::DenseMap<
      const lldb_private::plugin::dwarf::DWARFDebugInfoEntry *,
      lldb_private::OptionalClangModuleID>
      DIEToModuleMap;
  typedef llvm::DenseMap<
      const lldb_private::plugin::dwarf::DWARFDebugInfoEntry *, clang::Decl *>
      DIEToDeclMap;

  lldb_private::TypeSystemClang &m_ast;
  DIEToDeclMap m_die_to_decl;
  DIEToDeclContextMap m_die_to_decl_ctx;
  DeclContextToDIEMap m_decl_ctx_to_die;
  DIEToModuleMap m_die_to_module;
  std::unique_ptr<lldb_private::ClangASTImporter> m_clang_ast_importer_up;
  /// @}

  clang::DeclContext *
  GetDeclContextForBlock(const lldb_private::plugin::dwarf::DWARFDIE &die);

  clang::BlockDecl *
  ResolveBlockDIE(const lldb_private::plugin::dwarf::DWARFDIE &die);

  clang::NamespaceDecl *
  ResolveNamespaceDIE(const lldb_private::plugin::dwarf::DWARFDIE &die);

  /// Returns the namespace decl that a DW_TAG_imported_declaration imports.
  ///
  /// \param[in] die The import declaration to resolve. If the DIE is not a
  ///                DW_TAG_imported_declaration the behaviour is undefined.
  ///
  /// \returns The decl corresponding to the namespace that the specified
  ///          'die' imports. If the imported entity is not a namespace
  ///          or another import declaration, returns nullptr. If an error
  ///          occurs, returns nullptr.
  clang::NamespaceDecl *ResolveImportedDeclarationDIE(
      const lldb_private::plugin::dwarf::DWARFDIE &die);

  bool ParseTemplateDIE(const lldb_private::plugin::dwarf::DWARFDIE &die,
                        lldb_private::TypeSystemClang::TemplateParameterInfos
                            &template_param_infos);

  bool ParseTemplateParameterInfos(
      const lldb_private::plugin::dwarf::DWARFDIE &parent_die,
      lldb_private::TypeSystemClang::TemplateParameterInfos
          &template_param_infos);

  std::string
  GetCPlusPlusQualifiedName(const lldb_private::plugin::dwarf::DWARFDIE &die);

  bool ParseChildMembers(
      const lldb_private::plugin::dwarf::DWARFDIE &die,
      lldb_private::CompilerType &class_compiler_type,
      std::vector<std::unique_ptr<clang::CXXBaseSpecifier>> &base_classes,
      std::vector<lldb_private::plugin::dwarf::DWARFDIE> &member_function_dies,
      DelayedPropertyList &delayed_properties,
      const lldb::AccessType default_accessibility,
      lldb_private::ClangASTImporter::LayoutInfo &layout_info);

  size_t
  ParseChildParameters(clang::DeclContext *containing_decl_ctx,
                       const lldb_private::plugin::dwarf::DWARFDIE &parent_die,
                       bool skip_artificial, bool &is_static, bool &is_variadic,
                       bool &has_template_params,
                       std::vector<lldb_private::CompilerType> &function_args,
                       std::vector<clang::ParmVarDecl *> &function_param_decls,
                       unsigned &type_quals);

  size_t ParseChildEnumerators(
      lldb_private::CompilerType &compiler_type, bool is_signed,
      uint32_t enumerator_byte_size,
      const lldb_private::plugin::dwarf::DWARFDIE &parent_die);

  /// Parse a structure, class, or union type DIE.
  lldb::TypeSP
  ParseStructureLikeDIE(const lldb_private::SymbolContext &sc,
                        const lldb_private::plugin::dwarf::DWARFDIE &die,
                        ParsedDWARFTypeAttributes &attrs);

  lldb_private::Type *
  GetTypeForDIE(const lldb_private::plugin::dwarf::DWARFDIE &die);

  clang::Decl *
  GetClangDeclForDIE(const lldb_private::plugin::dwarf::DWARFDIE &die);

  clang::DeclContext *
  GetClangDeclContextForDIE(const lldb_private::plugin::dwarf::DWARFDIE &die);

  clang::DeclContext *GetClangDeclContextContainingDIE(
      const lldb_private::plugin::dwarf::DWARFDIE &die,
      lldb_private::plugin::dwarf::DWARFDIE *decl_ctx_die);
  lldb_private::OptionalClangModuleID
  GetOwningClangModule(const lldb_private::plugin::dwarf::DWARFDIE &die);

  bool CopyUniqueClassMethodTypes(
      const lldb_private::plugin::dwarf::DWARFDIE &src_class_die,
      const lldb_private::plugin::dwarf::DWARFDIE &dst_class_die,
      lldb_private::Type *class_type,
      std::vector<lldb_private::plugin::dwarf::DWARFDIE> &failures);

  clang::DeclContext *GetCachedClangDeclContextForDIE(
      const lldb_private::plugin::dwarf::DWARFDIE &die);

  void LinkDeclContextToDIE(clang::DeclContext *decl_ctx,
                            const lldb_private::plugin::dwarf::DWARFDIE &die);

  void LinkDeclToDIE(clang::Decl *decl,
                     const lldb_private::plugin::dwarf::DWARFDIE &die);

  /// If \p type_sp is valid, calculate and set its symbol context scope, and
  /// update the type list for its backing symbol file.
  ///
  /// Returns \p type_sp.
  lldb::TypeSP UpdateSymbolContextScopeForType(
      const lldb_private::SymbolContext &sc,
      const lldb_private::plugin::dwarf::DWARFDIE &die, lldb::TypeSP type_sp);

  /// Follow Clang Module Skeleton CU references to find a type definition.
  lldb::TypeSP
  ParseTypeFromClangModule(const lldb_private::SymbolContext &sc,
                           const lldb_private::plugin::dwarf::DWARFDIE &die,
                           lldb_private::Log *log);

  // Return true if this type is a declaration to a type in an external
  // module.
  lldb::ModuleSP
  GetModuleForType(const lldb_private::plugin::dwarf::DWARFDIE &die);

private:
  struct FieldInfo {
    uint64_t bit_size = 0;
    uint64_t bit_offset = 0;
    bool is_bitfield = false;
    bool is_artificial = false;

    FieldInfo() = default;

    void SetIsBitfield(bool flag) { is_bitfield = flag; }
    bool IsBitfield() { return is_bitfield; }

    void SetIsArtificial(bool flag) { is_artificial = flag; }
    bool IsArtificial() const { return is_artificial; }

    bool NextBitfieldOffsetIsValid(const uint64_t next_bit_offset) const {
      // Any subsequent bitfields must not overlap and must be at a higher
      // bit offset than any previous bitfield + size.
      return (bit_size + bit_offset) <= next_bit_offset;
    }
  };

  /// Returns 'true' if we should create an unnamed bitfield
  /// and add it to the parser's current AST.
  ///
  /// \param[in] last_field_info FieldInfo of the previous DW_TAG_member
  ///            we parsed.
  /// \param[in] last_field_end Offset (in bits) where the last parsed field
  ///            ended.
  /// \param[in] this_field_info FieldInfo of the current DW_TAG_member
  ///            being parsed.
  /// \param[in] layout_info Layout information of all decls parsed by the
  ///            current parser.
  bool ShouldCreateUnnamedBitfield(
      FieldInfo const &last_field_info, uint64_t last_field_end,
      FieldInfo const &this_field_info,
      lldb_private::ClangASTImporter::LayoutInfo const &layout_info) const;

  /// Parses a DW_TAG_APPLE_property DIE and appends the parsed data to the
  /// list of delayed Objective-C properties.
  ///
  /// Note: The delayed property needs to be finalized to actually create the
  /// property declarations in the module AST.
  ///
  /// \param die The DW_TAG_APPLE_property DIE that will be parsed.
  /// \param parent_die The parent DIE.
  /// \param class_clang_type The Objective-C class that will contain the
  /// created property.
  /// \param delayed_properties The list of delayed properties that the result
  /// will be appended to.
  void
  ParseObjCProperty(const lldb_private::plugin::dwarf::DWARFDIE &die,
                    const lldb_private::plugin::dwarf::DWARFDIE &parent_die,
                    const lldb_private::CompilerType &class_clang_type,
                    DelayedPropertyList &delayed_properties);

  void
  ParseSingleMember(const lldb_private::plugin::dwarf::DWARFDIE &die,
                    const lldb_private::plugin::dwarf::DWARFDIE &parent_die,
                    const lldb_private::CompilerType &class_clang_type,
                    lldb::AccessType default_accessibility,
                    lldb_private::ClangASTImporter::LayoutInfo &layout_info,
                    FieldInfo &last_field_info);

  bool CompleteRecordType(const lldb_private::plugin::dwarf::DWARFDIE &die,
                          lldb_private::Type *type,
                          lldb_private::CompilerType &clang_type);
  bool CompleteEnumType(const lldb_private::plugin::dwarf::DWARFDIE &die,
                        lldb_private::Type *type,
                        lldb_private::CompilerType &clang_type);

  lldb::TypeSP
  ParseTypeModifier(const lldb_private::SymbolContext &sc,
                    const lldb_private::plugin::dwarf::DWARFDIE &die,
                    ParsedDWARFTypeAttributes &attrs);
  lldb::TypeSP ParseEnum(const lldb_private::SymbolContext &sc,
                         const lldb_private::plugin::dwarf::DWARFDIE &die,
                         ParsedDWARFTypeAttributes &attrs);
  lldb::TypeSP ParseSubroutine(const lldb_private::plugin::dwarf::DWARFDIE &die,
                               ParsedDWARFTypeAttributes &attrs);
  lldb::TypeSP ParseArrayType(const lldb_private::plugin::dwarf::DWARFDIE &die,
                              const ParsedDWARFTypeAttributes &attrs);
  lldb::TypeSP
  ParsePointerToMemberType(const lldb_private::plugin::dwarf::DWARFDIE &die,
                           const ParsedDWARFTypeAttributes &attrs);

  /// Parses a DW_TAG_inheritance DIE into a base/super class.
  ///
  /// \param die The DW_TAG_inheritance DIE to parse.
  /// \param parent_die The parent DIE of the given DIE.
  /// \param class_clang_type The C++/Objective-C class representing parent_die.
  /// For an Objective-C class this method sets the super class on success. For
  /// a C++ class this will *not* add the result as a base class.
  /// \param default_accessibility The default accessibility that is given to
  /// base classes if they don't have an explicit accessibility set.
  /// \param module_sp The current Module.
  /// \param base_classes The list of C++ base classes that will be appended
  /// with the parsed base class on success.
  /// \param layout_info The layout information that will be updated for C++
  /// base classes with the base offset.
  void ParseInheritance(
      const lldb_private::plugin::dwarf::DWARFDIE &die,
      const lldb_private::plugin::dwarf::DWARFDIE &parent_die,
      const lldb_private::CompilerType class_clang_type,
      const lldb::AccessType default_accessibility,
      const lldb::ModuleSP &module_sp,
      std::vector<std::unique_ptr<clang::CXXBaseSpecifier>> &base_classes,
      lldb_private::ClangASTImporter::LayoutInfo &layout_info);

  /// Parses DW_TAG_variant_part DIE into a structure that encodes all variants
  /// Note that this is currently being emitted by rustc and not Clang
  /// \param die DW_TAG_variant_part DIE to parse
  /// \param parent_die The parent DW_TAG_structure_type to parse
  /// \param class_clang_type The Rust struct representing parent_die.
  /// \param default_accesibility The default accessibility that is given to
  ///  base classes if they don't have an explicit accessibility set
  /// \param layout_info The layout information that will be updated for
  //   base classes with the base offset
  void
  ParseRustVariantPart(lldb_private::plugin::dwarf::DWARFDIE &die,
                       const lldb_private::plugin::dwarf::DWARFDIE &parent_die,
                       lldb_private::CompilerType &class_clang_type,
                       const lldb::AccessType default_accesibility,
                       lldb_private::ClangASTImporter::LayoutInfo &layout_info);
};

/// Parsed form of all attributes that are relevant for type reconstruction.
/// Some attributes are relevant for all kinds of types (declaration), while
/// others are only meaningful to a specific type (is_virtual)
struct ParsedDWARFTypeAttributes {
  explicit ParsedDWARFTypeAttributes(
      const lldb_private::plugin::dwarf::DWARFDIE &die);

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
  lldb_private::plugin::dwarf::DWARFDIE object_pointer;
  lldb_private::plugin::dwarf::DWARFFormValue abstract_origin;
  lldb_private::plugin::dwarf::DWARFFormValue containing_type;
  lldb_private::plugin::dwarf::DWARFFormValue signature;
  lldb_private::plugin::dwarf::DWARFFormValue specification;
  lldb_private::plugin::dwarf::DWARFFormValue type;
  lldb::LanguageType class_language = lldb::eLanguageTypeUnknown;
  std::optional<uint64_t> byte_size;
  size_t calling_convention = llvm::dwarf::DW_CC_normal;
  uint32_t bit_stride = 0;
  uint32_t byte_stride = 0;
  uint32_t encoding = 0;
  clang::RefQualifierKind ref_qual =
      clang::RQ_None; ///< Indicates ref-qualifier of
                      ///< C++ member function if present.
                      ///< Is RQ_None otherwise.
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFASTPARSERCLANG_H
