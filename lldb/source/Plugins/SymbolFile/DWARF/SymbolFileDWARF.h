//===-- SymbolFileDWARF.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARF_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARF_H

#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Threading.h"

#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Expression/DWARFExpressionList.h"
#include "lldb/Symbol/DebugMacros.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Flags.h"
#include "lldb/Utility/RangeMap.h"
#include "lldb/lldb-private.h"

#include "DWARFContext.h"
#include "DWARFDataExtractor.h"
#include "DWARFDefines.h"
#include "DWARFIndex.h"
#include "UniqueDWARFASTType.h"

// Forward Declarations for this DWARF plugin
class DebugMapModule;
class DWARFAbbreviationDeclaration;
class DWARFAbbreviationDeclarationSet;
class DWARFCompileUnit;
class DWARFDebugAbbrev;
class DWARFDebugAranges;
class DWARFDebugInfo;
class DWARFDebugInfoEntry;
class DWARFDebugLine;
class DWARFDebugRanges;
class DWARFDeclContext;
class DWARFFormValue;
class DWARFTypeUnit;
class SymbolFileDWARFDebugMap;
class SymbolFileDWARFDwo;
class SymbolFileDWARFDwp;

namespace lldb_private {
  class ClangASTImporter;
  class SwiftASTContext;
}

#define DIE_IS_BEING_PARSED ((lldb_private::Type *)1)

class SymbolFileDWARF : public lldb_private::SymbolFileCommon,
                        public lldb_private::UserID {
  /// LLVM RTTI support.
  static char ID;

public:
  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SymbolFileCommon::isA(ClassID);
  }
  static bool classof(const SymbolFile *obj) { return obj->isA(&ID); }
  /// \}

  friend class SymbolFileDWARFDebugMap;
  friend class SymbolFileDWARFDwo;
  friend class DebugMapModule;
  friend class DWARFCompileUnit;
  friend class DWARFDIE;
  friend class DWARFASTParserClang;
  friend class DWARFASTParserSwift;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static void DebuggerInitialize(lldb_private::Debugger &debugger);

  static llvm::StringRef GetPluginNameStatic() { return "dwarf"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::SymbolFile *
  CreateInstance(lldb::ObjectFileSP objfile_sp);

  // Constructors and Destructors

  SymbolFileDWARF(lldb::ObjectFileSP objfile_sp,
                  lldb_private::SectionList *dwo_section_list);

  ~SymbolFileDWARF() override;

  uint32_t CalculateAbilities() override;

  void InitializeObject() override;

  // Compile Unit function calls
  llvm::VersionTuple
  GetProducerVersion(lldb_private::CompileUnit &comp_unit) override;

  lldb::LanguageType
  ParseLanguage(lldb_private::CompileUnit &comp_unit) override;

  lldb_private::XcodeSDK
  ParseXcodeSDK(lldb_private::CompileUnit &comp_unit) override;

  size_t ParseFunctions(lldb_private::CompileUnit &comp_unit) override;

  bool ParseLineTable(lldb_private::CompileUnit &comp_unit) override;

  bool ParseDebugMacros(lldb_private::CompileUnit &comp_unit) override;

  bool ForEachExternalModule(
      lldb_private::CompileUnit &, llvm::DenseSet<lldb_private::SymbolFile *> &,
      llvm::function_ref<bool(lldb_private::Module &)>) override;

  bool ParseSupportFiles(lldb_private::CompileUnit &comp_unit,
                         lldb_private::FileSpecList &support_files) override;

  bool ParseIsOptimized(lldb_private::CompileUnit &comp_unit) override;

  size_t ParseTypes(lldb_private::CompileUnit &comp_unit) override;

  bool ParseImportedModules(
      const lldb_private::SymbolContext &sc,
      std::vector<lldb_private::SourceModule> &imported_modules) override;

  size_t ParseBlocksRecursive(lldb_private::Function &func) override;

  size_t
  ParseVariablesForContext(const lldb_private::SymbolContext &sc) override;

  lldb_private::Type *ResolveTypeUID(lldb::user_id_t type_uid) override;
  llvm::Optional<ArrayInfo> GetDynamicArrayInfoForUID(
      lldb::user_id_t type_uid,
      const lldb_private::ExecutionContext *exe_ctx) override;

  bool CompleteType(lldb_private::CompilerType &compiler_type) override;

  lldb_private::Type *ResolveType(const DWARFDIE &die,
                                  bool assert_not_being_parsed = true,
                                  bool resolve_function_context = false);

  lldb_private::CompilerDecl GetDeclForUID(lldb::user_id_t uid) override;

  lldb_private::CompilerDeclContext
  GetDeclContextForUID(lldb::user_id_t uid) override;

  lldb_private::CompilerDeclContext
  GetDeclContextContainingUID(lldb::user_id_t uid) override;

  void GetDeclContextForUID(
      llvm::SmallVectorImpl<lldb_private::CompilerContext> &context,
      lldb::user_id_t type_uid) override;

  void
  ParseDeclsForContext(lldb_private::CompilerDeclContext decl_ctx) override;

  uint32_t ResolveSymbolContext(const lldb_private::Address &so_addr,
                                lldb::SymbolContextItem resolve_scope,
                                lldb_private::SymbolContext &sc) override;

  lldb_private::Status
  CalculateFrameVariableError(lldb_private::StackFrame &frame) override;

  uint32_t ResolveSymbolContext(
      const lldb_private::SourceLocationSpec &src_location_spec,
      lldb::SymbolContextItem resolve_scope,
      lldb_private::SymbolContextList &sc_list) override;

  void
  FindGlobalVariables(lldb_private::ConstString name,
                      const lldb_private::CompilerDeclContext &parent_decl_ctx,
                      uint32_t max_matches,
                      lldb_private::VariableList &variables) override;

  void FindGlobalVariables(const lldb_private::RegularExpression &regex,
                           uint32_t max_matches,
                           lldb_private::VariableList &variables) override;

  void FindFunctions(const lldb_private::Module::LookupInfo &lookup_info,
                     const lldb_private::CompilerDeclContext &parent_decl_ctx,
                     bool include_inlines,
                     lldb_private::SymbolContextList &sc_list) override;

  void FindFunctions(const lldb_private::RegularExpression &regex,
                     bool include_inlines,
                     lldb_private::SymbolContextList &sc_list) override;

  void GetMangledNamesForFunction(
      const std::string &scope_qualified_name,
      std::vector<lldb_private::ConstString> &mangled_names) override;

  void
  FindTypes(lldb_private::ConstString name,
            const lldb_private::CompilerDeclContext &parent_decl_ctx,
            uint32_t max_matches,
            llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
            lldb_private::TypeMap &types) override;

  void FindTypes(llvm::ArrayRef<lldb_private::CompilerContext> pattern,
                 lldb_private::LanguageSet languages,
                 llvm::DenseSet<SymbolFile *> &searched_symbol_files,
                 lldb_private::TypeMap &types) override;

  void GetTypes(lldb_private::SymbolContextScope *sc_scope,
                lldb::TypeClass type_mask,
                lldb_private::TypeList &type_list) override;

  llvm::Expected<lldb::TypeSystemSP>
  GetTypeSystemForLanguage(lldb::LanguageType language) override;

  lldb_private::CompilerDeclContext FindNamespace(
      lldb_private::ConstString name,
      const lldb_private::CompilerDeclContext &parent_decl_ctx) override;

  bool GetCompileOption(const char *option, std::string &value,
                        lldb_private::CompileUnit *cu = nullptr) override;

  void PreloadSymbols() override;

  std::recursive_mutex &GetModuleMutex() const override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  DWARFDebugAbbrev *DebugAbbrev();

  DWARFDebugInfo &DebugInfo();

  DWARFDebugRanges *GetDebugRanges();

  static bool SupportedVersion(uint16_t version);

  DWARFDIE
  GetDeclContextDIEContainingDIE(const DWARFDIE &die);

  bool
  HasForwardDeclForClangType(const lldb_private::CompilerType &compiler_type);

  lldb_private::CompileUnit *
  GetCompUnitForDWARFCompUnit(DWARFCompileUnit &dwarf_cu);

  virtual void GetObjCMethods(lldb_private::ConstString class_name,
                              llvm::function_ref<bool(DWARFDIE die)> callback);

  bool Supports_DW_AT_APPLE_objc_complete_type(DWARFUnit *cu);

  lldb_private::DebugMacrosSP ParseDebugMacros(lldb::offset_t *offset);

  static DWARFDIE GetParentSymbolContextDIE(const DWARFDIE &die);

  lldb::ModuleSP GetExternalModule(lldb_private::ConstString name);

  typedef std::map<lldb_private::ConstString, lldb::ModuleSP>
      ExternalTypeModuleMap;

  /// Return the list of Clang modules imported by this SymbolFile.
  const ExternalTypeModuleMap &getExternalTypeModules() const {
    return m_external_type_modules;
  }

  virtual DWARFDIE GetDIE(const DIERef &die_ref);

  DWARFDIE GetDIE(lldb::user_id_t uid);

  lldb::user_id_t GetUID(const DWARFBaseDIE &die) {
    return GetUID(die.GetDIERef());
  }

  lldb::user_id_t GetUID(const llvm::Optional<DIERef> &ref) {
    return ref ? GetUID(*ref) : LLDB_INVALID_UID;
  }

  lldb::user_id_t GetUID(DIERef ref);

  std::shared_ptr<SymbolFileDWARFDwo>
  GetDwoSymbolFileForCompileUnit(DWARFUnit &dwarf_cu,
                                 const DWARFDebugInfoEntry &cu_die);

  virtual llvm::Optional<uint32_t> GetDwoNum() { return llvm::None; }

  /// If this is a DWARF object with a single CU, return its DW_AT_dwo_id.
  llvm::Optional<uint64_t> GetDWOId();

  static bool
  DIEInDeclContext(const lldb_private::CompilerDeclContext &parent_decl_ctx,
                   const DWARFDIE &die);

  std::vector<std::unique_ptr<lldb_private::CallEdge>>
  ParseCallEdgesInFunction(UserID func_id) override;

  void Dump(lldb_private::Stream &s) override;

  void DumpClangAST(lldb_private::Stream &s) override;

  lldb_private::DWARFContext &GetDWARFContext() { return m_context; }

  const std::shared_ptr<SymbolFileDWARFDwo> &GetDwpSymbolFile();

  lldb_private::FileSpec GetFile(DWARFUnit &unit, size_t file_idx);

  static llvm::Expected<lldb::TypeSystemSP> GetTypeSystem(DWARFUnit &unit);

  static DWARFASTParser *GetDWARFParser(DWARFUnit &unit);

  // CompilerDecl related functions

  static lldb_private::CompilerDecl GetDecl(const DWARFDIE &die);

  static lldb_private::CompilerDeclContext GetDeclContext(const DWARFDIE &die);

  static lldb_private::CompilerDeclContext
  GetContainingDeclContext(const DWARFDIE &die);

  static DWARFDeclContext GetDWARFDeclContext(const DWARFDIE &die);

  static lldb::LanguageType LanguageTypeFromDWARF(uint64_t val);

  static lldb::LanguageType GetLanguage(DWARFUnit &unit);
  /// Same as GetLanguage() but reports all C++ versions as C++ (no version).
  static lldb::LanguageType GetLanguageFamily(DWARFUnit &unit);

  lldb_private::StatsDuration::Duration GetDebugInfoParseTime() override {
    return m_parse_time;
  }
  lldb_private::StatsDuration::Duration GetDebugInfoIndexTime() override;

  lldb_private::StatsDuration &GetDebugInfoParseTimeRef() {
    return m_parse_time;
  }

  lldb_private::ConstString ConstructFunctionDemangledName(const DWARFDIE &die);

protected:
  typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb_private::Type *>
      DIEToTypePtr;
  typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::VariableSP>
      DIEToVariableSP;
  typedef llvm::DenseMap<const DWARFDebugInfoEntry *,
                         lldb::opaque_compiler_type_t>
      DIEToClangType;
  typedef llvm::DenseMap<lldb::opaque_compiler_type_t, DIERef> ClangTypeToDIE;

  SymbolFileDWARF(const SymbolFileDWARF &) = delete;
  const SymbolFileDWARF &operator=(const SymbolFileDWARF &) = delete;

  virtual void LoadSectionData(lldb::SectionType sect_type,
                               lldb_private::DWARFDataExtractor &data);

  bool DeclContextMatchesThisSymbolFile(
      const lldb_private::CompilerDeclContext &decl_ctx);

  uint32_t CalculateNumCompileUnits() override;

  lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index) override;

  lldb_private::TypeList &GetTypeList() override;

  lldb::CompUnitSP ParseCompileUnit(DWARFCompileUnit &dwarf_cu);

  virtual DWARFCompileUnit *
  GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit);

  DWARFUnit *GetNextUnparsedDWARFCompileUnit(DWARFUnit *prev_cu);

  bool GetFunction(const DWARFDIE &die, lldb_private::SymbolContext &sc);

  lldb_private::Function *ParseFunction(lldb_private::CompileUnit &comp_unit,
                                        const DWARFDIE &die);

  size_t ParseBlocksRecursive(lldb_private::CompileUnit &comp_unit,
                              lldb_private::Block *parent_block,
                              const DWARFDIE &die,
                              lldb::addr_t subprogram_low_pc, uint32_t depth);

  size_t ParseTypes(const lldb_private::SymbolContext &sc, const DWARFDIE &die,
                    bool parse_siblings, bool parse_children);

  lldb::TypeSP ParseType(const lldb_private::SymbolContext &sc,
                         const DWARFDIE &die, bool *type_is_new);

  bool ParseSupportFiles(DWARFUnit &dwarf_cu, const lldb::ModuleSP &module,
                         lldb_private::FileSpecList &support_files);

  lldb_private::Type *ResolveTypeUID(const DWARFDIE &die,
                                     bool assert_not_being_parsed);

  lldb_private::Type *ResolveTypeUID(const DIERef &die_ref);

  lldb::VariableSP ParseVariableDIE(const lldb_private::SymbolContext &sc,
                                    const DWARFDIE &die,
                                    const lldb::addr_t func_low_pc);
  lldb::VariableSP ParseVariableDIECached(const lldb_private::SymbolContext &sc,
                                          const DWARFDIE &die);

  void
  ParseAndAppendGlobalVariable(const lldb_private::SymbolContext &sc,
                               const DWARFDIE &die,
                               lldb_private::VariableList &cc_variable_list);

  size_t ParseVariablesInFunctionContext(const lldb_private::SymbolContext &sc,
                                         const DWARFDIE &die,
                                         const lldb::addr_t func_low_pc);

  size_t ParseVariablesInFunctionContextRecursive(
      const lldb_private::SymbolContext &sc, const DWARFDIE &die,
      lldb::addr_t func_low_pc, DIEArray &accumulator);

  size_t PopulateBlockVariableList(lldb_private::VariableList &variable_list,
                                   const lldb_private::SymbolContext &sc,
                                   llvm::ArrayRef<DIERef> variable_dies,
                                   lldb::addr_t func_low_pc);

  DIEArray MergeBlockAbstractParameters(const DWARFDIE &block_die,
                                        DIEArray &&variable_dies);

  bool ClassOrStructIsVirtual(const DWARFDIE &die);

  // Given a die_offset, figure out the symbol context representing that die.
  bool ResolveFunction(const DWARFDIE &die, bool include_inlines,
                       lldb_private::SymbolContextList &sc_list);

  /// Resolve functions and (possibly) blocks for the given file address and a
  /// compile unit. The compile unit comes from the sc argument and it must be
  /// set. The results of the lookup (if any) are written back to the symbol
  /// context.
  void ResolveFunctionAndBlock(lldb::addr_t file_vm_addr, bool lookup_block,
                               lldb_private::SymbolContext &sc);

  virtual lldb::TypeSP
  FindDefinitionTypeForDWARFDeclContext(const DWARFDeclContext &die_decl_ctx);

  virtual lldb::TypeSP
  FindCompleteObjCDefinitionTypeForDIE(const DWARFDIE &die,
                                       lldb_private::ConstString type_name,
                                       bool must_be_implementation);

  lldb_private::Symbol *
  GetObjCClassSymbol(lldb_private::ConstString objc_class_name);

  lldb::TypeSP GetTypeForDIE(const DWARFDIE &die,
                             bool resolve_function_context = false);

  void SetDebugMapModule(const lldb::ModuleSP &module_sp) {
    m_debug_map_module_wp = module_sp;
  }

  SymbolFileDWARFDebugMap *GetDebugMapSymfile();

  DWARFDIE
  FindBlockContainingSpecification(const DIERef &func_die_ref,
                                   dw_offset_t spec_block_die_offset);

  DWARFDIE
  FindBlockContainingSpecification(const DWARFDIE &die,
                                   dw_offset_t spec_block_die_offset);

  virtual UniqueDWARFASTTypeMap &GetUniqueDWARFASTTypeMap();

  bool DIEDeclContextsMatch(const DWARFDIE &die1, const DWARFDIE &die2);

  bool ClassContainsSelector(const DWARFDIE &class_die,
                             lldb_private::ConstString selector);

  /// Parse call site entries (DW_TAG_call_site), including any nested call site
  /// parameters (DW_TAG_call_site_parameter).
  std::vector<std::unique_ptr<lldb_private::CallEdge>>
  CollectCallEdges(lldb::ModuleSP module, DWARFDIE function_die);

  /// If this symbol file is linked to by a debug map (see
  /// SymbolFileDWARFDebugMap), and \p file_addr is a file address relative to
  /// an object file, adjust \p file_addr so that it is relative to the main
  /// binary. Returns the adjusted address, or \p file_addr if no adjustment is
  /// needed, on success and LLDB_INVALID_ADDRESS otherwise.
  lldb::addr_t FixupAddress(lldb::addr_t file_addr);

  bool FixupAddress(lldb_private::Address &addr);

  typedef llvm::SetVector<lldb_private::Type *> TypeSet;

  void GetTypes(const DWARFDIE &die, dw_offset_t min_die_offset,
                dw_offset_t max_die_offset, uint32_t type_mask,
                TypeSet &type_set);

  typedef lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t,
                                        lldb_private::Variable *>
      GlobalVariableMap;

  GlobalVariableMap &GetGlobalAranges();

  void UpdateExternalModuleListIfNeeded();

  lldb_private::ClangASTImporter &GetClangASTImporter();

  lldb_private::SwiftASTContext *
  GetSwiftASTContextForCU(lldb_private::Status *error, DWARFCompileUnit &cu);

  lldb::user_id_t GetTypeUIDFromTypeAttribute(const DWARFFormValue &type_attr);

  lldb::TypeSP ResolveTypeFromAttribute(const DWARFFormValue &type_attr);

  virtual DIEToTypePtr &GetDIEToType() { return m_die_to_type; }

  virtual DIEToVariableSP &GetDIEToVariable() { return m_die_to_variable_sp; }

  virtual DIEToClangType &GetForwardDeclDieToClangType() {
    return m_forward_decl_die_to_clang_type;
  }

  virtual ClangTypeToDIE &GetForwardDeclClangTypeToDie() {
    return m_forward_decl_clang_type_to_die;
  }

  void BuildCuTranslationTable();
  llvm::Optional<uint32_t> GetDWARFUnitIndex(uint32_t cu_idx);

  struct DecodedUID {
    SymbolFileDWARF &dwarf;
    DIERef ref;
  };
  llvm::Optional<DecodedUID> DecodeUID(lldb::user_id_t uid);

  void FindDwpSymbolFile();

  const lldb_private::FileSpecList &GetTypeUnitSupportFiles(DWARFTypeUnit &tu);

  void InitializeFirstCodeAddressRecursive(
      const lldb_private::SectionList &section_list);

  void InitializeFirstCodeAddress();

  void GetCompileOptions(
      std::unordered_map<lldb::CompUnitSP, lldb_private::Args> &args) override;

  lldb::ModuleWP m_debug_map_module_wp;
  SymbolFileDWARFDebugMap *m_debug_map_symfile;

  llvm::once_flag m_dwp_symfile_once_flag;
  std::shared_ptr<SymbolFileDWARFDwo> m_dwp_symfile;

  lldb_private::DWARFContext m_context;

  llvm::once_flag m_info_once_flag;
  std::unique_ptr<DWARFDebugInfo> m_info;

  std::unique_ptr<DWARFDebugAbbrev> m_abbr;
  std::unique_ptr<GlobalVariableMap> m_global_aranges_up;
  std::unique_ptr<lldb_private::ClangASTImporter> m_clang_ast_importer_up;

  typedef std::unordered_map<lldb::offset_t, lldb_private::DebugMacrosSP>
      DebugMacrosMap;
  DebugMacrosMap m_debug_macros_map;

  ExternalTypeModuleMap m_external_type_modules;
  std::unique_ptr<lldb_private::DWARFIndex> m_index;
  bool m_fetched_external_modules : 1;
  lldb_private::LazyBool m_supports_DW_AT_APPLE_objc_complete_type;

  typedef std::set<DIERef> DIERefSet;
  typedef llvm::StringMap<DIERefSet> NameToOffsetMap;
  NameToOffsetMap m_function_scope_qualified_name_map;
  std::unique_ptr<DWARFDebugRanges> m_ranges;
  UniqueDWARFASTTypeMap m_unique_ast_type_map;
  DIEToTypePtr m_die_to_type;
  DIEToVariableSP m_die_to_variable_sp;
  DIEToClangType m_forward_decl_die_to_clang_type;
  ClangTypeToDIE m_forward_decl_clang_type_to_die;
  llvm::DenseMap<dw_offset_t, lldb_private::FileSpecList>
      m_type_unit_support_files;
  std::vector<uint32_t> m_lldb_cu_to_dwarf_unit;
  /// DWARF does not provide a good way for traditional (concatenating) linkers
  /// to invalidate debug info describing dead-stripped code. These linkers will
  /// keep the debug info but resolve any addresses referring to such code as
  /// zero (BFD) or a small positive integer (zero + relocation addend -- GOLD).
  /// Try to filter out this debug info by comparing it to the lowest code
  /// address in the module.
  lldb::addr_t m_first_code_address = LLDB_INVALID_ADDRESS;
  lldb_private::StatsDuration m_parse_time;
  std::atomic_flag m_dwo_warning_issued = ATOMIC_FLAG_INIT;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARF_H
