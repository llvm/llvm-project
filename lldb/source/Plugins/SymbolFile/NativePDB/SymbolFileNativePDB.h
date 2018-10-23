//===-- SymbolFileNativePDB.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILE_NATIVEPDB_SYMBOLFILENATIVEPDB_H
#define LLDB_PLUGINS_SYMBOLFILE_NATIVEPDB_SYMBOLFILENATIVEPDB_H

#include "lldb/Symbol/ClangASTImporter.h"
#include "lldb/Symbol/SymbolFile.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"

#include "CompileUnitIndex.h"
#include "PdbIndex.h"

namespace clang {
class TagDecl;
}

namespace llvm {
namespace codeview {
class ClassRecord;
class EnumRecord;
class ModifierRecord;
class PointerRecord;
struct UnionRecord;
} // namespace codeview
} // namespace llvm

namespace lldb_private {
class ClangASTImporter;

namespace npdb {

struct DeclStatus {
  DeclStatus() = default;
  DeclStatus(lldb::user_id_t uid, Type::ResolveStateTag status)
      : uid(uid), status(status) {}
  lldb::user_id_t uid = 0;
  Type::ResolveStateTag status = Type::eResolveStateForward;
};

class SymbolFileNativePDB : public SymbolFile {
  friend class UdtRecordCompleter;

public:
  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static void DebuggerInitialize(Debugger &debugger);

  static ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static SymbolFile *CreateInstance(ObjectFile *obj_file);

  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  SymbolFileNativePDB(ObjectFile *ofile);

  ~SymbolFileNativePDB() override;

  uint32_t CalculateAbilities() override;

  void InitializeObject() override;

  //------------------------------------------------------------------
  // Compile Unit function calls
  //------------------------------------------------------------------

  uint32_t GetNumCompileUnits() override;

  lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index) override;

  lldb::LanguageType ParseCompileUnitLanguage(const SymbolContext &sc) override;

  size_t ParseCompileUnitFunctions(const SymbolContext &sc) override;

  bool ParseCompileUnitLineTable(const SymbolContext &sc) override;

  bool ParseCompileUnitDebugMacros(const SymbolContext &sc) override;

  bool ParseCompileUnitSupportFiles(const SymbolContext &sc,
                                    FileSpecList &support_files) override;

  bool
  ParseImportedModules(const SymbolContext &sc,
                       std::vector<ConstString> &imported_modules) override;

  size_t ParseFunctionBlocks(const SymbolContext &sc) override;

  size_t ParseTypes(const SymbolContext &sc) override;
  size_t ParseVariablesForContext(const SymbolContext &sc) override {
    return 0;
  }
  Type *ResolveTypeUID(lldb::user_id_t type_uid) override;
  bool CompleteType(CompilerType &compiler_type) override;
  uint32_t ResolveSymbolContext(const Address &so_addr, uint32_t resolve_scope,
                                SymbolContext &sc) override;

  size_t GetTypes(SymbolContextScope *sc_scope, uint32_t type_mask,
                  TypeList &type_list) override;

  uint32_t FindFunctions(const ConstString &name,
                         const CompilerDeclContext *parent_decl_ctx,
                         uint32_t name_type_mask, bool include_inlines,
                         bool append, SymbolContextList &sc_list) override;

  uint32_t FindFunctions(const RegularExpression &regex, bool include_inlines,
                         bool append, SymbolContextList &sc_list) override;

  uint32_t FindTypes(const SymbolContext &sc, const ConstString &name,
                     const CompilerDeclContext *parent_decl_ctx, bool append,
                     uint32_t max_matches,
                     llvm::DenseSet<SymbolFile *> &searched_symbol_files,
                     TypeMap &types) override;

  size_t FindTypes(const std::vector<CompilerContext> &context, bool append,
                   TypeMap &types) override;

  TypeSystem *GetTypeSystemForLanguage(lldb::LanguageType language) override;

  CompilerDeclContext
  FindNamespace(const SymbolContext &sc, const ConstString &name,
                const CompilerDeclContext *parent_decl_ctx) override;

  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  llvm::pdb::PDBFile &GetPDBFile() { return m_index->pdb(); }
  const llvm::pdb::PDBFile &GetPDBFile() const { return m_index->pdb(); }

  ClangASTContext &GetASTContext() { return *m_clang; }
  ClangASTImporter &GetASTImporter() { return *m_importer; }

private:
  size_t FindTypesByName(llvm::StringRef name, uint32_t max_matches,
                         TypeMap &types);

  lldb::TypeSP CreateModifierType(PdbSymUid type_uid,
                                  const llvm::codeview::ModifierRecord &mr);
  lldb::TypeSP CreatePointerType(PdbSymUid type_uid,
                                 const llvm::codeview::PointerRecord &pr);
  lldb::TypeSP CreateSimpleType(llvm::codeview::TypeIndex ti);
  lldb::TypeSP CreateTagType(PdbSymUid type_uid,
                             const llvm::codeview::ClassRecord &cr);
  lldb::TypeSP CreateTagType(PdbSymUid type_uid,
                             const llvm::codeview::EnumRecord &er);
  lldb::TypeSP CreateTagType(PdbSymUid type_uid,
                             const llvm::codeview::UnionRecord &ur);
  lldb::TypeSP
  CreateClassStructUnion(PdbSymUid type_uid, llvm::StringRef name, size_t size,
                         clang::TagTypeKind ttk,
                         clang::MSInheritanceAttr::Spelling inheritance);

  lldb::FunctionSP GetOrCreateFunction(PdbSymUid func_uid,
                                       const SymbolContext &sc);
  lldb::CompUnitSP GetOrCreateCompileUnit(const CompilandIndexItem &cci);
  lldb::TypeSP GetOrCreateType(PdbSymUid type_uid);
  lldb::TypeSP GetOrCreateType(llvm::codeview::TypeIndex ti);

  lldb::FunctionSP CreateFunction(PdbSymUid func_uid, const SymbolContext &sc);
  lldb::CompUnitSP CreateCompileUnit(const CompilandIndexItem &cci);
  lldb::TypeSP CreateType(PdbSymUid type_uid);
  lldb::TypeSP CreateAndCacheType(PdbSymUid type_uid);

  llvm::BumpPtrAllocator m_allocator;

  lldb::addr_t m_obj_load_address = 0;

  std::unique_ptr<PdbIndex> m_index;
  std::unique_ptr<ClangASTImporter> m_importer;
  ClangASTContext *m_clang = nullptr;

  llvm::DenseMap<clang::TagDecl *, DeclStatus> m_decl_to_status;

  llvm::DenseMap<lldb::user_id_t, clang::TagDecl *> m_uid_to_decl;

  llvm::DenseMap<lldb::user_id_t, lldb::FunctionSP> m_functions;
  llvm::DenseMap<lldb::user_id_t, lldb::CompUnitSP> m_compilands;
  llvm::DenseMap<lldb::user_id_t, lldb::TypeSP> m_types;
};

} // namespace npdb
} // namespace lldb_private

#endif // lldb_Plugins_SymbolFile_PDB_SymbolFilePDB_h_
