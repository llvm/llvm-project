//===-- SymbolFilePDB.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Plugins_SymbolFile_PDB_SymbolFilePDB_h_
#define lldb_Plugins_SymbolFile_PDB_SymbolFilePDB_h_

#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Utility/UserID.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/PDB/IPDBSession.h"
#include "llvm/DebugInfo/PDB/PDB.h"

class SymbolFilePDB : public lldb_private::SymbolFile {
public:
  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static void DebuggerInitialize(lldb_private::Debugger &debugger);

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static lldb_private::SymbolFile *
  CreateInstance(lldb_private::ObjectFile *obj_file);

  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  SymbolFilePDB(lldb_private::ObjectFile *ofile);

  ~SymbolFilePDB() override;

  uint32_t CalculateAbilities() override;

  void InitializeObject() override;

  //------------------------------------------------------------------
  // Compile Unit function calls
  //------------------------------------------------------------------

  uint32_t GetNumCompileUnits() override;

  lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index) override;

  lldb::LanguageType
  ParseCompileUnitLanguage(const lldb_private::SymbolContext &sc) override;

  size_t
  ParseCompileUnitFunctions(const lldb_private::SymbolContext &sc) override;

  bool
  ParseCompileUnitLineTable(const lldb_private::SymbolContext &sc) override;

  bool
  ParseCompileUnitDebugMacros(const lldb_private::SymbolContext &sc) override;

  bool ParseCompileUnitSupportFiles(
      const lldb_private::SymbolContext &sc,
      lldb_private::FileSpecList &support_files) override;

  bool ParseImportedModules(
      const lldb_private::SymbolContext &sc,
      std::vector<lldb_private::ConstString> &imported_modules) override;

  size_t ParseFunctionBlocks(const lldb_private::SymbolContext &sc) override;

  size_t ParseTypes(const lldb_private::SymbolContext &sc) override;

  size_t
  ParseVariablesForContext(const lldb_private::SymbolContext &sc) override;

  lldb_private::Type *ResolveTypeUID(lldb::user_id_t type_uid) override;

  bool CompleteType(lldb_private::CompilerType &compiler_type) override;

  lldb_private::CompilerDecl GetDeclForUID(lldb::user_id_t uid) override;

  lldb_private::CompilerDeclContext
  GetDeclContextForUID(lldb::user_id_t uid) override;

  lldb_private::CompilerDeclContext
  GetDeclContextContainingUID(lldb::user_id_t uid) override;

  void
  ParseDeclsForContext(lldb_private::CompilerDeclContext decl_ctx) override;

  uint32_t ResolveSymbolContext(const lldb_private::Address &so_addr,
                                uint32_t resolve_scope,
                                lldb_private::SymbolContext &sc) override;

  uint32_t
  ResolveSymbolContext(const lldb_private::FileSpec &file_spec, uint32_t line,
                       bool check_inlines, uint32_t resolve_scope,
                       lldb_private::SymbolContextList &sc_list) override;

  uint32_t
  FindGlobalVariables(const lldb_private::ConstString &name,
                      const lldb_private::CompilerDeclContext *parent_decl_ctx,
                      bool append, uint32_t max_matches,
                      lldb_private::VariableList &variables) override;

  uint32_t FindGlobalVariables(const lldb_private::RegularExpression &regex,
                               bool append, uint32_t max_matches,
                               lldb_private::VariableList &variables) override;

  uint32_t
  FindFunctions(const lldb_private::ConstString &name,
                const lldb_private::CompilerDeclContext *parent_decl_ctx,
                uint32_t name_type_mask, bool include_inlines, bool append,
                lldb_private::SymbolContextList &sc_list) override;

  uint32_t FindFunctions(const lldb_private::RegularExpression &regex,
                         bool include_inlines, bool append,
                         lldb_private::SymbolContextList &sc_list) override;

  void GetMangledNamesForFunction(
      const std::string &scope_qualified_name,
      std::vector<lldb_private::ConstString> &mangled_names) override;

  uint32_t
  FindTypes(const lldb_private::SymbolContext &sc,
            const lldb_private::ConstString &name,
            const lldb_private::CompilerDeclContext *parent_decl_ctx,
            bool append, uint32_t max_matches,
            llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
            lldb_private::TypeMap &types) override;

  size_t FindTypes(const std::vector<lldb_private::CompilerContext> &context,
                   bool append, lldb_private::TypeMap &types) override;

  void FindTypesByRegex(const lldb_private::RegularExpression &regex,
                        uint32_t max_matches,
                        lldb_private::TypeMap &types);

  lldb_private::TypeList *GetTypeList() override;

  size_t GetTypes(lldb_private::SymbolContextScope *sc_scope,
                  uint32_t type_mask,
                  lldb_private::TypeList &type_list) override;

  lldb_private::TypeSystem *
  GetTypeSystemForLanguage(lldb::LanguageType language) override;

  lldb_private::CompilerDeclContext FindNamespace(
      const lldb_private::SymbolContext &sc,
      const lldb_private::ConstString &name,
      const lldb_private::CompilerDeclContext *parent_decl_ctx) override;

  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  llvm::pdb::IPDBSession &GetPDBSession();

  const llvm::pdb::IPDBSession &GetPDBSession() const;

private:
  lldb::CompUnitSP ParseCompileUnitForSymIndex(uint32_t id);

  bool ParseCompileUnitLineTable(const lldb_private::SymbolContext &sc,
                                 uint32_t match_line);

  void BuildSupportFileIdToSupportFileIndexMap(
      const llvm::pdb::PDBSymbolCompiland &cu,
      llvm::DenseMap<uint32_t, uint32_t> &index_map) const;

  void FindTypesByName(const std::string &name, uint32_t max_matches,
                       lldb_private::TypeMap &types);

  llvm::DenseMap<uint32_t, lldb::CompUnitSP> m_comp_units;
  llvm::DenseMap<uint32_t, lldb::TypeSP> m_types;

  std::vector<lldb::TypeSP> m_builtin_types;
  std::unique_ptr<llvm::pdb::IPDBSession> m_session_up;
  uint32_t m_cached_compile_unit_count;
  std::unique_ptr<lldb_private::CompilerDeclContext> m_tu_decl_ctx_up;
};

#endif // lldb_Plugins_SymbolFile_PDB_SymbolFilePDB_h_
