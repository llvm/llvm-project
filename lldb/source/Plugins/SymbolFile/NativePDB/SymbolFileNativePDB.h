//===-- SymbolFileNativePDB.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Plugins_SymbolFile_PDB_SymbolFileNativePDB_h_
#define lldb_Plugins_SymbolFile_PDB_SymbolFileNativePDB_h_

#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Utility/UserID.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/StringsAndChecksums.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"

#include "CompileUnitIndex.h"
#include "PdbIndex.h"

#include <unordered_map>

namespace llvm {
namespace pdb {
class PDBFile;
class PDBSymbol;
class PDBSymbolCompiland;
class PDBSymbolData;
class PDBSymbolFunc;

class DbiStream;
class TpiStream;
class TpiStream;
class InfoStream;
class PublicsStream;
class GlobalsStream;
class SymbolStream;
class ModuleDebugStreamRef;
} // namespace pdb
} // namespace llvm

namespace lldb_private {
namespace npdb {

class SymbolFileNativePDB : public lldb_private::SymbolFile {
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
  SymbolFileNativePDB(lldb_private::ObjectFile *ofile);

  ~SymbolFileNativePDB() override;

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

  size_t ParseTypes(const lldb_private::SymbolContext &sc) override {
    return 0;
  }
  size_t
  ParseVariablesForContext(const lldb_private::SymbolContext &sc) override {
    return 0;
  }
  lldb_private::Type *ResolveTypeUID(lldb::user_id_t type_uid) override {
    return nullptr;
  }
  bool CompleteType(lldb_private::CompilerType &compiler_type) override {
    return false;
  }
  uint32_t ResolveSymbolContext(const lldb_private::Address &so_addr,
                                uint32_t resolve_scope,
                                lldb_private::SymbolContext &sc) override;

  virtual size_t GetTypes(lldb_private::SymbolContextScope *sc_scope,
                          uint32_t type_mask,
                          lldb_private::TypeList &type_list) override {
    return 0;
  }

  uint32_t
  FindFunctions(const lldb_private::ConstString &name,
                const lldb_private::CompilerDeclContext *parent_decl_ctx,
                uint32_t name_type_mask, bool include_inlines, bool append,
                lldb_private::SymbolContextList &sc_list) override;

  uint32_t FindFunctions(const lldb_private::RegularExpression &regex,
                         bool include_inlines, bool append,
                         lldb_private::SymbolContextList &sc_list) override;

  lldb_private::TypeSystem *
  GetTypeSystemForLanguage(lldb::LanguageType language) override;

  lldb_private::CompilerDeclContext FindNamespace(
      const lldb_private::SymbolContext &sc,
      const lldb_private::ConstString &name,
      const lldb_private::CompilerDeclContext *parent_decl_ctx) override;

  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  llvm::pdb::PDBFile &GetPDBFile() { return m_index->pdb(); }
  const llvm::pdb::PDBFile &GetPDBFile() const { return m_index->pdb(); }

private:
  lldb::FunctionSP GetOrCreateFunction(PdbSymUid func_uid,
                                       const SymbolContext &sc);
  lldb::CompUnitSP GetOrCreateCompileUnit(const CompilandIndexItem &cci);

  lldb::FunctionSP CreateFunction(PdbSymUid func_uid, const SymbolContext &sc);
  lldb::CompUnitSP CreateCompileUnit(const CompilandIndexItem &cci);

  llvm::BumpPtrAllocator m_allocator;

  lldb::addr_t m_obj_load_address = 0;

  std::unique_ptr<PdbIndex> m_index;

  llvm::DenseMap<lldb::user_id_t, lldb::FunctionSP> m_functions;
  llvm::DenseMap<lldb::user_id_t, lldb::CompUnitSP> m_compilands;
};

} // namespace npdb
} // namespace lldb_private

#endif // lldb_Plugins_SymbolFile_PDB_SymbolFilePDB_h_
