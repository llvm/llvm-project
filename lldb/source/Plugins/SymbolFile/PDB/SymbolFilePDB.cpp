//===-- SymbolFilePDB.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFilePDB.h"

#include "clang/Lex/Lexer.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Utility/RegularExpression.h"

#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/IPDBDataStream.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBLineNumber.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/IPDBTable.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandDetails.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugEnd.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

#include "Plugins/SymbolFile/PDB/PDBASTParser.h"

#include <regex>

using namespace lldb;
using namespace lldb_private;
using namespace llvm::pdb;

namespace {
lldb::LanguageType TranslateLanguage(PDB_Lang lang) {
  switch (lang) {
  case PDB_Lang::Cpp:
    return lldb::LanguageType::eLanguageTypeC_plus_plus;
  case PDB_Lang::C:
    return lldb::LanguageType::eLanguageTypeC;
  default:
    return lldb::LanguageType::eLanguageTypeUnknown;
  }
}

bool ShouldAddLine(uint32_t requested_line, uint32_t actual_line,
                   uint32_t addr_length) {
  return ((requested_line == 0 || actual_line == requested_line) &&
          addr_length > 0);
}
}

void SymbolFilePDB::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInitialize);
}

void SymbolFilePDB::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

void SymbolFilePDB::DebuggerInitialize(lldb_private::Debugger &debugger) {}

lldb_private::ConstString SymbolFilePDB::GetPluginNameStatic() {
  static ConstString g_name("pdb");
  return g_name;
}

const char *SymbolFilePDB::GetPluginDescriptionStatic() {
  return "Microsoft PDB debug symbol file reader.";
}

lldb_private::SymbolFile *
SymbolFilePDB::CreateInstance(lldb_private::ObjectFile *obj_file) {
  return new SymbolFilePDB(obj_file);
}

SymbolFilePDB::SymbolFilePDB(lldb_private::ObjectFile *object_file)
    : SymbolFile(object_file), m_session_up(), m_global_scope_up(),
      m_cached_compile_unit_count(0), m_tu_decl_ctx_up() {}

SymbolFilePDB::~SymbolFilePDB() {}

uint32_t SymbolFilePDB::CalculateAbilities() {
  uint32_t abilities = 0;
  if (!m_obj_file)
    return 0;

  if (!m_session_up) {
    // Lazily load and match the PDB file, but only do this once.
    std::string exePath = m_obj_file->GetFileSpec().GetPath();
    auto error = loadDataForEXE(PDB_ReaderType::DIA, llvm::StringRef(exePath),
                                m_session_up);
    if (error) {
      llvm::consumeError(std::move(error));
      auto module_sp = m_obj_file->GetModule();
      if (!module_sp)
        return 0;
      // See if any symbol file is specified through `--symfile` option.
      FileSpec symfile = module_sp->GetSymbolFileFileSpec();
      if (!symfile)
        return 0;
      error = loadDataForPDB(PDB_ReaderType::DIA,
                             llvm::StringRef(symfile.GetPath()),
                             m_session_up);
      if (error) {
        llvm::consumeError(std::move(error));
        return 0;
      }
    }
  }
  if (!m_session_up.get())
    return 0;

  auto enum_tables_up = m_session_up->getEnumTables();
  if (!enum_tables_up)
    return 0;
  while (auto table_up = enum_tables_up->getNext()) {
    if (table_up->getItemCount() == 0)
      continue;
    auto type = table_up->getTableType();
    switch (type) {
    case PDB_TableType::Symbols:
      // This table represents a store of symbols with types listed in
      // PDBSym_Type
      abilities |= (CompileUnits | Functions | Blocks |
                    GlobalVariables | LocalVariables | VariableTypes);
      break;
    case PDB_TableType::LineNumbers:
      abilities |= LineTables;
      break;
    default: break;
    }
  }
  return abilities;
}

void SymbolFilePDB::InitializeObject() {
  lldb::addr_t obj_load_address = m_obj_file->GetFileOffset();
  lldbassert(obj_load_address &&
             obj_load_address != LLDB_INVALID_ADDRESS);
  m_session_up->setLoadAddress(obj_load_address);
  if (!m_global_scope_up)
    m_global_scope_up = m_session_up->getGlobalScope();
  lldbassert(m_global_scope_up.get());

  TypeSystem *type_system =
      GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus);
  ClangASTContext *clang_type_system =
      llvm::dyn_cast_or_null<ClangASTContext>(type_system);
  lldbassert(clang_type_system);
  m_tu_decl_ctx_up = llvm::make_unique<CompilerDeclContext>(
      type_system, clang_type_system->GetTranslationUnitDecl());
}

uint32_t SymbolFilePDB::GetNumCompileUnits() {
  if (m_cached_compile_unit_count == 0) {
    auto compilands = m_global_scope_up->findAllChildren<PDBSymbolCompiland>();
    if (!compilands)
      return 0;

    // The linker could link *.dll (compiland language = LINK), or import
    // *.dll. For example, a compiland with name `Import:KERNEL32.dll`
    // could be found as a child of the global scope (PDB executable).
    // Usually, such compilands contain `thunk` symbols in which we are not
    // interested for now. However we still count them in the compiland list.
    // If we perform any compiland related activity, like finding symbols
    // through llvm::pdb::IPDBSession methods, such compilands will all be
    // searched automatically no matter whether we include them or not.
    m_cached_compile_unit_count = compilands->getChildCount();

    // The linker can inject an additional "dummy" compilation unit into the
    // PDB. Ignore this special compile unit for our purposes, if it is there.
    // It is always the last one.
    auto last_compiland_up =
        compilands->getChildAtIndex(m_cached_compile_unit_count - 1);
    lldbassert(last_compiland_up.get());
    std::string name = last_compiland_up->getName();
    if (name == "* Linker *")
      --m_cached_compile_unit_count;
  }
  return m_cached_compile_unit_count;
}

void SymbolFilePDB::GetCompileUnitIndex(
    const llvm::pdb::PDBSymbolCompiland *pdb_compiland,
    uint32_t &index) {
  if (!pdb_compiland)
    return;

  auto results_up = m_global_scope_up->findAllChildren<PDBSymbolCompiland>();
  if (!results_up)
    return;
  auto uid = pdb_compiland->getSymIndexId();
  for (uint32_t cu_idx = 0; cu_idx < GetNumCompileUnits(); ++cu_idx) {
    auto compiland_up = results_up->getChildAtIndex(cu_idx);
    if (!compiland_up)
      continue;
    if (compiland_up->getSymIndexId() == uid) {
      index = cu_idx;
      return;
    }
  }
  index = UINT32_MAX;
  return;
}

std::unique_ptr<llvm::pdb::PDBSymbolCompiland>
SymbolFilePDB::GetPDBCompilandByUID(uint32_t uid) {
  return m_session_up->getConcreteSymbolById<PDBSymbolCompiland>(uid);
}

lldb::CompUnitSP SymbolFilePDB::ParseCompileUnitAtIndex(uint32_t index) {
  if (index >= GetNumCompileUnits())
    return CompUnitSP();

  // Assuming we always retrieve same compilands listed in same order through
  // `PDBSymbolExe::findAllChildren` method, otherwise using `index` to get a
  // compile unit makes no sense.
  auto results = m_global_scope_up->findAllChildren<PDBSymbolCompiland>();
  if (!results)
    return CompUnitSP();
  auto compiland_up = results->getChildAtIndex(index);
  if (!compiland_up)
    return CompUnitSP();
  return ParseCompileUnitForUID(compiland_up->getSymIndexId(), index);
}

lldb::LanguageType
SymbolFilePDB::ParseCompileUnitLanguage(const lldb_private::SymbolContext &sc) {
  // What fields should I expect to be filled out on the SymbolContext?  Is it
  // safe to assume that `sc.comp_unit` is valid?
  if (!sc.comp_unit)
    return lldb::eLanguageTypeUnknown;

  auto compiland_up = GetPDBCompilandByUID(sc.comp_unit->GetID());
  if (!compiland_up)
    return lldb::eLanguageTypeUnknown;
  auto details = compiland_up->findOneChild<PDBSymbolCompilandDetails>();
  if (!details)
    return lldb::eLanguageTypeUnknown;
  return TranslateLanguage(details->getLanguage());
}

size_t SymbolFilePDB::ParseCompileUnitFunctions(
    const lldb_private::SymbolContext &sc) {
  // TODO: Implement this
  return size_t();
}

bool SymbolFilePDB::ParseCompileUnitLineTable(
    const lldb_private::SymbolContext &sc) {
  lldbassert(sc.comp_unit);
  if (sc.comp_unit->GetLineTable())
    return true;
  return ParseCompileUnitLineTable(sc, 0);
}

bool SymbolFilePDB::ParseCompileUnitDebugMacros(
    const lldb_private::SymbolContext &sc) {
  // PDB doesn't contain information about macros
  return false;
}

bool SymbolFilePDB::ParseCompileUnitSupportFiles(
    const lldb_private::SymbolContext &sc,
    lldb_private::FileSpecList &support_files) {
  lldbassert(sc.comp_unit);

  // In theory this is unnecessary work for us, because all of this information
  // is easily (and quickly) accessible from DebugInfoPDB, so caching it a
  // second time seems like a waste.  Unfortunately, there's no good way around
  // this short of a moderate refactor since SymbolVendor depends on being able
  // to cache this list.
  auto compiland_up = GetPDBCompilandByUID(sc.comp_unit->GetID());
  if (!compiland_up)
    return false;
  auto files = m_session_up->getSourceFilesForCompiland(*compiland_up);
  if (!files || files->getChildCount() == 0)
    return false;

  while (auto file = files->getNext()) {
    FileSpec spec(file->getFileName(), false, FileSpec::ePathSyntaxWindows);
    support_files.AppendIfUnique(spec);
  }
  return true;
}

bool SymbolFilePDB::ParseImportedModules(
    const lldb_private::SymbolContext &sc,
    std::vector<lldb_private::ConstString> &imported_modules) {
  // PDB does not yet support module debug info
  return false;
}

size_t
SymbolFilePDB::ParseFunctionBlocks(const lldb_private::SymbolContext &sc) {
  // TODO: Implement this
  return size_t();
}

size_t SymbolFilePDB::ParseTypes(const lldb_private::SymbolContext &sc) {
  lldbassert(sc.module_sp.get());
  size_t num_added = 0;
  auto results_up = m_session_up->getGlobalScope()->findAllChildren();
  if (!results_up)
    return 0;
  while (auto symbol_up = results_up->getNext()) {
    switch (symbol_up->getSymTag()) {
    case PDB_SymType::Enum:
    case PDB_SymType::UDT:
    case PDB_SymType::Typedef:
      break;
    default:
      continue;
    }

    auto type_uid = symbol_up->getSymIndexId();
    if (m_types.find(type_uid) != m_types.end())
      continue;

    // This should cause the type to get cached and stored in the `m_types`
    // lookup.
    if (!ResolveTypeUID(symbol_up->getSymIndexId()))
      continue;

    ++num_added;
  }
  return num_added;
}

size_t
SymbolFilePDB::ParseVariablesForContext(const lldb_private::SymbolContext &sc) {
  // TODO: Implement this
  return size_t();
}

lldb_private::Type *SymbolFilePDB::ResolveTypeUID(lldb::user_id_t type_uid) {
  auto find_result = m_types.find(type_uid);
  if (find_result != m_types.end())
    return find_result->second.get();

  TypeSystem *type_system =
      GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus);
  ClangASTContext *clang_type_system =
      llvm::dyn_cast_or_null<ClangASTContext>(type_system);
  if (!clang_type_system)
    return nullptr;
  PDBASTParser *pdb =
      llvm::dyn_cast<PDBASTParser>(clang_type_system->GetPDBParser());
  if (!pdb)
    return nullptr;

  auto pdb_type = m_session_up->getSymbolById(type_uid);
  if (pdb_type == nullptr)
    return nullptr;

  lldb::TypeSP result = pdb->CreateLLDBTypeFromPDBType(*pdb_type);
  if (result.get()) {
    m_types.insert(std::make_pair(type_uid, result));
    auto type_list = GetTypeList();
    type_list->Insert(result);
  }
  return result.get();
}

bool SymbolFilePDB::CompleteType(lldb_private::CompilerType &compiler_type) {
  // TODO: Implement this
  return false;
}

lldb_private::CompilerDecl SymbolFilePDB::GetDeclForUID(lldb::user_id_t uid) {
  return lldb_private::CompilerDecl();
}

lldb_private::CompilerDeclContext
SymbolFilePDB::GetDeclContextForUID(lldb::user_id_t uid) {
  // PDB always uses the translation unit decl context for everything.  We can
  // improve this later but it's not easy because PDB doesn't provide a high
  // enough level of type fidelity in this area.
  return *m_tu_decl_ctx_up;
}

lldb_private::CompilerDeclContext
SymbolFilePDB::GetDeclContextContainingUID(lldb::user_id_t uid) {
  return *m_tu_decl_ctx_up;
}

void SymbolFilePDB::ParseDeclsForContext(
    lldb_private::CompilerDeclContext decl_ctx) {}

uint32_t
SymbolFilePDB::ResolveSymbolContext(const lldb_private::Address &so_addr,
                                    uint32_t resolve_scope,
                                    lldb_private::SymbolContext &sc) {
  return uint32_t();
}

std::string SymbolFilePDB::GetSourceFileNameForPDBCompiland(
    const PDBSymbolCompiland *pdb_compiland) {
  if (!pdb_compiland)
    return std::string();

  std::string source_file_name;
  // `getSourceFileName` returns the basename of the original source file
  // used to generate this compiland.  It does not return the full path.
  // Currently the only way to get that is to do a basename lookup to get the
  // IPDBSourceFile, but this is ambiguous in the case of two source files
  // with the same name contributing to the same compiland. This is an edge
  // case that we ignore for now, although we need to a long-term solution.
  std::string file_name = pdb_compiland->getSourceFileName();
  if (!file_name.empty()) {
    auto one_src_file_up =
      m_session_up->findOneSourceFile(pdb_compiland, file_name,
                                      PDB_NameSearchFlags::NS_CaseInsensitive);
    if (one_src_file_up)
      source_file_name = one_src_file_up->getFileName();
  }
  // For some reason, source file name could be empty, so we will walk through
  // all source files of this compiland, and determine the right source file
  // if any that is used to generate this compiland based on language
  // indicated in compilanddetails language field.
  if (!source_file_name.empty())
    return source_file_name;

  auto details_up = pdb_compiland->findOneChild<PDBSymbolCompilandDetails>();
  PDB_Lang pdb_lang = details_up ? details_up->getLanguage() : PDB_Lang::Cpp;
  auto src_files_up =
    m_session_up->getSourceFilesForCompiland(*pdb_compiland);
  if (src_files_up) {
    while (auto file_up = src_files_up->getNext()) {
      FileSpec file_spec(file_up->getFileName(), false,
                         FileSpec::ePathSyntaxWindows);
      auto file_extension = file_spec.GetFileNameExtension();
      if (pdb_lang == PDB_Lang::Cpp || pdb_lang == PDB_Lang::C) {
        static const char* exts[] = { "cpp", "c", "cc", "cxx" };
        if (llvm::is_contained(exts, file_extension.GetStringRef().lower()))
          source_file_name = file_up->getFileName();
        break;
      } else if (pdb_lang == PDB_Lang::Masm &&
                 ConstString::Compare(file_extension, ConstString("ASM"),
                                      false) == 0) {
        source_file_name = file_up->getFileName();
        break;
      }
    }
  }
  return source_file_name;
}

uint32_t SymbolFilePDB::ResolveSymbolContext(
    const lldb_private::FileSpec &file_spec, uint32_t line, bool check_inlines,
    uint32_t resolve_scope, lldb_private::SymbolContextList &sc_list) {
  const size_t old_size = sc_list.GetSize();
  if (resolve_scope & lldb::eSymbolContextCompUnit) {
    // Locate all compilation units with line numbers referencing the specified
    // file.  For example, if `file_spec` is <vector>, then this should return
    // all source files and header files that reference <vector>, either
    // directly or indirectly.
    auto compilands = m_session_up->findCompilandsForSourceFile(
        file_spec.GetPath(), PDB_NameSearchFlags::NS_CaseInsensitive);

    if (!compilands)
      return 0;

    // For each one, either find its previously parsed data or parse it afresh
    // and add it to the symbol context list.
    while (auto compiland = compilands->getNext()) {
      // If we're not checking inlines, then don't add line information for this
      // file unless the FileSpec matches.
      if (!check_inlines) {
        // `getSourceFileName` returns the basename of the original source file
        // used to generate this compiland.  It does not return the full path.
        // Currently the only way to get that is to do a basename lookup to get
        // the IPDBSourceFile, but this is ambiguous in the case of two source
        // files with the same name contributing to the same compiland.  This is
        // a moderately extreme edge case, so we consider this OK for now,
        // although we need to find a long-term solution.
        std::string source_file =
            GetSourceFileNameForPDBCompiland(compiland.get());
        if (source_file.empty())
          continue;
        FileSpec this_spec(source_file, false, FileSpec::ePathSyntaxWindows);
        bool need_full_match = !file_spec.GetDirectory().IsEmpty();
        if (FileSpec::Compare(file_spec, this_spec, need_full_match) != 0)
          continue;
      }

      SymbolContext sc;
      auto cu = ParseCompileUnitForUID(compiland->getSymIndexId());
      if (!cu.get())
        continue;
      sc.comp_unit = cu.get();
      sc.module_sp = cu->GetModule();
      sc_list.Append(sc);

      // If we were asked to resolve line entries, add all entries to the line
      // table that match the requested line (or all lines if `line` == 0).
      if (resolve_scope & lldb::eSymbolContextLineEntry)
        ParseCompileUnitLineTable(sc, line);
    }
  }
  return sc_list.GetSize() - old_size;
}

uint32_t SymbolFilePDB::FindGlobalVariables(
    const lldb_private::ConstString &name,
    const lldb_private::CompilerDeclContext *parent_decl_ctx, bool append,
    uint32_t max_matches, lldb_private::VariableList &variables) {
  return uint32_t();
}

uint32_t
SymbolFilePDB::FindGlobalVariables(const lldb_private::RegularExpression &regex,
                                   bool append, uint32_t max_matches,
                                   lldb_private::VariableList &variables) {
  return uint32_t();
}

uint32_t SymbolFilePDB::FindFunctions(
    const lldb_private::ConstString &name,
    const lldb_private::CompilerDeclContext *parent_decl_ctx,
    uint32_t name_type_mask, bool include_inlines, bool append,
    lldb_private::SymbolContextList &sc_list) {
  return uint32_t();
}

uint32_t
SymbolFilePDB::FindFunctions(const lldb_private::RegularExpression &regex,
                             bool include_inlines, bool append,
                             lldb_private::SymbolContextList &sc_list) {
  return uint32_t();
}

void SymbolFilePDB::GetMangledNamesForFunction(
    const std::string &scope_qualified_name,
    std::vector<lldb_private::ConstString> &mangled_names) {}

uint32_t SymbolFilePDB::FindTypes(
    const lldb_private::SymbolContext &sc,
    const lldb_private::ConstString &name,
    const lldb_private::CompilerDeclContext *parent_decl_ctx, bool append,
    uint32_t max_matches,
    llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
    lldb_private::TypeMap &types) {
  if (!append)
    types.Clear();
  if (!name)
    return 0;

  searched_symbol_files.clear();
  searched_symbol_files.insert(this);

  std::string name_str = name.AsCString();

  // There is an assumption 'name' is not a regex
  FindTypesByName(name_str, max_matches, types);
   
  return types.GetSize();
}

void
SymbolFilePDB::FindTypesByRegex(const lldb_private::RegularExpression &regex,
                                uint32_t max_matches,
                                lldb_private::TypeMap &types) {
  // When searching by regex, we need to go out of our way to limit the search
  // space as much as possible since this searches EVERYTHING in the PDB,
  // manually doing regex comparisons.  PDB library isn't optimized for regex
  // searches or searches across multiple symbol types at the same time, so the
  // best we can do is to search enums, then typedefs, then classes one by one,
  // and do a regex comparison against each of them.
  PDB_SymType tags_to_search[] = {PDB_SymType::Enum, PDB_SymType::Typedef,
                                  PDB_SymType::UDT};
  std::unique_ptr<IPDBEnumSymbols> results;

  uint32_t matches = 0;

  for (auto tag : tags_to_search) {
    results = m_global_scope_up->findAllChildren(tag);
    if (!results)
      continue;

    while (auto result = results->getNext()) {
      if (max_matches > 0 && matches >= max_matches)
        break;

      std::string type_name;
      if (auto enum_type = llvm::dyn_cast<PDBSymbolTypeEnum>(result.get()))
        type_name = enum_type->getName();
      else if (auto typedef_type =
                   llvm::dyn_cast<PDBSymbolTypeTypedef>(result.get()))
        type_name = typedef_type->getName();
      else if (auto class_type = llvm::dyn_cast<PDBSymbolTypeUDT>(result.get()))
        type_name = class_type->getName();
      else {
        // We're looking only for types that have names.  Skip symbols, as well
        // as unnamed types such as arrays, pointers, etc.
        continue;
      }

      if (!regex.Execute(type_name))
        continue;

      // This should cause the type to get cached and stored in the `m_types`
      // lookup.
      if (!ResolveTypeUID(result->getSymIndexId()))
        continue;

      auto iter = m_types.find(result->getSymIndexId());
      if (iter == m_types.end())
        continue;
      types.Insert(iter->second);
      ++matches;
    }
  }
}

void SymbolFilePDB::FindTypesByName(const std::string &name,
                                    uint32_t max_matches,
                                    lldb_private::TypeMap &types) {
  std::unique_ptr<IPDBEnumSymbols> results;
  results = m_global_scope_up->findChildren(PDB_SymType::None, name,
                                            PDB_NameSearchFlags::NS_Default);
  if (!results)
    return;

  uint32_t matches = 0;

  while (auto result = results->getNext()) {
    if (max_matches > 0 && matches >= max_matches)
      break;
    switch (result->getSymTag()) {
    case PDB_SymType::Enum:
    case PDB_SymType::UDT:
    case PDB_SymType::Typedef:
      break;
    default:
      // We're looking only for types that have names.  Skip symbols, as well as
      // unnamed types such as arrays, pointers, etc.
      continue;
    }

    // This should cause the type to get cached and stored in the `m_types`
    // lookup.
    if (!ResolveTypeUID(result->getSymIndexId()))
      continue;

    auto iter = m_types.find(result->getSymIndexId());
    if (iter == m_types.end())
      continue;
    types.Insert(iter->second);
    ++matches;
  }
}

size_t SymbolFilePDB::FindTypes(
    const std::vector<lldb_private::CompilerContext> &contexts, bool append,
    lldb_private::TypeMap &types) {
  return 0;
}

lldb_private::TypeList *SymbolFilePDB::GetTypeList() {
  return m_obj_file->GetModule()->GetTypeList();
}

size_t SymbolFilePDB::GetTypes(lldb_private::SymbolContextScope *sc_scope,
                               uint32_t type_mask,
                               lldb_private::TypeList &type_list) {
  return size_t();
}

lldb_private::TypeSystem *
SymbolFilePDB::GetTypeSystemForLanguage(lldb::LanguageType language) {
  auto type_system =
      m_obj_file->GetModule()->GetTypeSystemForLanguage(language);
  if (type_system)
    type_system->SetSymbolFile(this);
  return type_system;
}

lldb_private::CompilerDeclContext SymbolFilePDB::FindNamespace(
    const lldb_private::SymbolContext &sc,
    const lldb_private::ConstString &name,
    const lldb_private::CompilerDeclContext *parent_decl_ctx) {
  return lldb_private::CompilerDeclContext();
}

lldb_private::ConstString SymbolFilePDB::GetPluginName() {
  static ConstString g_name("pdb");
  return g_name;
}

uint32_t SymbolFilePDB::GetPluginVersion() { return 1; }

IPDBSession &SymbolFilePDB::GetPDBSession() { return *m_session_up; }

const IPDBSession &SymbolFilePDB::GetPDBSession() const {
  return *m_session_up;
}

lldb::CompUnitSP
SymbolFilePDB::ParseCompileUnitForUID(uint32_t id, uint32_t index) {
  auto found_cu = m_comp_units.find(id);
  if (found_cu != m_comp_units.end())
    return found_cu->second;

  auto compiland_up = GetPDBCompilandByUID(id);
  if (!compiland_up)
    return CompUnitSP();
  std::string path = GetSourceFileNameForPDBCompiland(compiland_up.get());
  if (path.empty())
    return CompUnitSP();

  lldb::LanguageType lang;
  auto details = compiland_up->findOneChild<PDBSymbolCompilandDetails>();
  if (!details)
    lang = lldb::eLanguageTypeC_plus_plus;
  else
    lang = TranslateLanguage(details->getLanguage());

  // Don't support optimized code for now, DebugInfoPDB does not return this
  // information.
  LazyBool optimized = eLazyBoolNo;
  auto cu_sp = std::make_shared<CompileUnit>(
      m_obj_file->GetModule(), nullptr, path.c_str(), id, lang, optimized);

  if (!cu_sp)
    return CompUnitSP();

  m_comp_units.insert(std::make_pair(id, cu_sp));
  if (index == UINT32_MAX)
    GetCompileUnitIndex(compiland_up.get(), index);
  lldbassert(index != UINT32_MAX);
  m_obj_file->GetModule()->GetSymbolVendor()->SetCompileUnitAtIndex(
      index, cu_sp);
  return cu_sp;
}

bool SymbolFilePDB::ParseCompileUnitLineTable(
    const lldb_private::SymbolContext &sc, uint32_t match_line) {
  lldbassert(sc.comp_unit);

  auto compiland_up = GetPDBCompilandByUID(sc.comp_unit->GetID());
  if (!compiland_up)
    return false;

  // LineEntry needs the *index* of the file into the list of support files
  // returned by ParseCompileUnitSupportFiles.  But the underlying SDK gives us
  // a globally unique idenfitifier in the namespace of the PDB.  So, we have to
  // do a mapping so that we can hand out indices.
  llvm::DenseMap<uint32_t, uint32_t> index_map;
  BuildSupportFileIdToSupportFileIndexMap(*compiland_up, index_map);
  auto line_table = llvm::make_unique<LineTable>(sc.comp_unit);

  // Find contributions to `compiland` from all source and header files.
  std::string path = sc.comp_unit->GetPath();
  auto files = m_session_up->getSourceFilesForCompiland(*compiland_up);
  if (!files)
    return false;

  // For each source and header file, create a LineSequence for contributions to
  // the compiland from that file, and add the sequence.
  while (auto file = files->getNext()) {
    std::unique_ptr<LineSequence> sequence(
        line_table->CreateLineSequenceContainer());
    auto lines = m_session_up->findLineNumbers(*compiland_up, *file);
    if (!lines)
      continue;
    int entry_count = lines->getChildCount();

    uint64_t prev_addr;
    uint32_t prev_length;
    uint32_t prev_line;
    uint32_t prev_source_idx;

    for (int i = 0; i < entry_count; ++i) {
      auto line = lines->getChildAtIndex(i);

      uint64_t lno = line->getLineNumber();
      uint64_t addr = line->getVirtualAddress();
      uint32_t length = line->getLength();
      uint32_t source_id = line->getSourceFileId();
      uint32_t col = line->getColumnNumber();
      uint32_t source_idx = index_map[source_id];

      // There was a gap between the current entry and the previous entry if the
      // addresses don't perfectly line up.
      bool is_gap = (i > 0) && (prev_addr + prev_length < addr);

      // Before inserting the current entry, insert a terminal entry at the end
      // of the previous entry's address range if the current entry resulted in
      // a gap from the previous entry.
      if (is_gap && ShouldAddLine(match_line, prev_line, prev_length)) {
        line_table->AppendLineEntryToSequence(
            sequence.get(), prev_addr + prev_length, prev_line, 0,
            prev_source_idx, false, false, false, false, true);
      }

      if (ShouldAddLine(match_line, lno, length)) {
        bool is_statement = line->isStatement();
        bool is_prologue = false;
        bool is_epilogue = false;
        auto func =
            m_session_up->findSymbolByAddress(addr, PDB_SymType::Function);
        if (func) {
          auto prologue = func->findOneChild<PDBSymbolFuncDebugStart>();
          if (prologue)
            is_prologue = (addr == prologue->getVirtualAddress());

          auto epilogue = func->findOneChild<PDBSymbolFuncDebugEnd>();
          if (epilogue)
            is_epilogue = (addr == epilogue->getVirtualAddress());
        }

        line_table->AppendLineEntryToSequence(sequence.get(), addr, lno, col,
                                              source_idx, is_statement, false,
                                              is_prologue, is_epilogue, false);
      }

      prev_addr = addr;
      prev_length = length;
      prev_line = lno;
      prev_source_idx = source_idx;
    }

    if (entry_count > 0 && ShouldAddLine(match_line, prev_line, prev_length)) {
      // The end is always a terminal entry, so insert it regardless.
      line_table->AppendLineEntryToSequence(
          sequence.get(), prev_addr + prev_length, prev_line, 0,
          prev_source_idx, false, false, false, false, true);
    }

    line_table->InsertSequence(sequence.release());
  }

  if (line_table->GetSize()) {
    sc.comp_unit->SetLineTable(line_table.release());
    return true;
  }
  return false;
}

void SymbolFilePDB::BuildSupportFileIdToSupportFileIndexMap(
    const PDBSymbolCompiland &compiland,
    llvm::DenseMap<uint32_t, uint32_t> &index_map) const {
  // This is a hack, but we need to convert the source id into an index into the
  // support files array.  We don't want to do path comparisons to avoid
  // basename / full path issues that may or may not even be a problem, so we
  // use the globally unique source file identifiers.  Ideally we could use the
  // global identifiers everywhere, but LineEntry currently assumes indices.
  auto source_files = m_session_up->getSourceFilesForCompiland(compiland);
  if (!source_files)
    return;
  int index = 0;

  while (auto file = source_files->getNext()) {
    uint32_t source_id = file->getUniqueId();
    index_map[source_id] = index++;
  }
}
