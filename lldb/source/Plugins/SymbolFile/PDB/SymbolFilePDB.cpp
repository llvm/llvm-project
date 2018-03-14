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
#include "llvm/DebugInfo/PDB/PDBSymbolBlock.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandDetails.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugEnd.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"
#include "llvm/DebugInfo/PDB/PDBSymbolPublicSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

#include "Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
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

lldb_private::Function *
SymbolFilePDB::ParseCompileUnitFunctionForPDBFunc(
    const PDBSymbolFunc *pdb_func,
    const lldb_private::SymbolContext &sc) {
  assert(pdb_func != nullptr);
  lldbassert(sc.comp_unit && sc.module_sp.get());

  auto file_vm_addr = pdb_func->getVirtualAddress();
  if (file_vm_addr == LLDB_INVALID_ADDRESS)
    return nullptr;

  auto func_length = pdb_func->getLength();
  AddressRange func_range = AddressRange(file_vm_addr,
                                         func_length,
                                         sc.module_sp->GetSectionList());
  if (!func_range.GetBaseAddress().IsValid())
    return nullptr;

  lldb_private::Type* func_type = ResolveTypeUID(pdb_func->getSymIndexId());
  if (!func_type)
    return nullptr;

  user_id_t func_type_uid = pdb_func->getSignatureId();

  Mangled mangled = GetMangledForPDBFunc(pdb_func);

  FunctionSP func_sp = std::make_shared<Function>(sc.comp_unit,
                                                  pdb_func->getSymIndexId(),
                                                  func_type_uid,
                                                  mangled,
                                                  func_type,
                                                  func_range);

  sc.comp_unit->AddFunction(func_sp);
  return func_sp.get();
}

size_t SymbolFilePDB::ParseCompileUnitFunctions(
    const lldb_private::SymbolContext &sc) {
  lldbassert(sc.comp_unit);
  size_t func_added = 0;
  auto compiland_up = GetPDBCompilandByUID(sc.comp_unit->GetID());
  if (!compiland_up)
    return 0;
  auto results_up = compiland_up->findAllChildren<PDBSymbolFunc>();
  if (!results_up)
    return 0;
  while (auto pdb_func_up = results_up->getNext()) {
    auto func_sp =
        sc.comp_unit->FindFunctionByUID(pdb_func_up->getSymIndexId());
    if (!func_sp) {
      if (ParseCompileUnitFunctionForPDBFunc(pdb_func_up.get(), sc))
        ++func_added;
    }
  }
  return func_added;
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

static size_t
ParseFunctionBlocksForPDBSymbol(const lldb_private::SymbolContext &sc,
                                uint64_t func_file_vm_addr,
                                const llvm::pdb::PDBSymbol *pdb_symbol,
                                lldb_private::Block *parent_block,
                                bool is_top_parent) {
  assert(pdb_symbol && parent_block);

  size_t num_added = 0;
  switch (pdb_symbol->getSymTag()) {
  case PDB_SymType::Block:
  case PDB_SymType::Function: {
    Block *block = nullptr;
    auto &raw_sym = pdb_symbol->getRawSymbol();
    if (auto *pdb_func = llvm::dyn_cast<PDBSymbolFunc>(pdb_symbol)) {
      if (pdb_func->hasNoInlineAttribute())
        break;
      if (is_top_parent)
        block = parent_block;
      else
        break;
    } else if (llvm::dyn_cast<PDBSymbolBlock>(pdb_symbol)) {
      auto uid = pdb_symbol->getSymIndexId();
      if (parent_block->FindBlockByID(uid))
        break;
      if (raw_sym.getVirtualAddress() < func_file_vm_addr)
        break;

      auto block_sp = std::make_shared<Block>(pdb_symbol->getSymIndexId());
      parent_block->AddChild(block_sp);
      block = block_sp.get();
    } else
      llvm_unreachable("Unexpected PDB symbol!");

    block->AddRange(
        Block::Range(raw_sym.getVirtualAddress() - func_file_vm_addr,
                     raw_sym.getLength()));
    block->FinalizeRanges();
    ++num_added;

    auto results_up = pdb_symbol->findAllChildren();
    if (!results_up)
      break;
    while (auto symbol_up = results_up->getNext()) {
      num_added += ParseFunctionBlocksForPDBSymbol(sc, func_file_vm_addr,
                                                   symbol_up.get(),
                                                   block, false);
    }
  } break;
  default: break;
  }
  return num_added;
}

size_t
SymbolFilePDB::ParseFunctionBlocks(const lldb_private::SymbolContext &sc) {
  lldbassert(sc.comp_unit && sc.function);
  size_t num_added = 0;
  auto uid = sc.function->GetID();
  auto pdb_func_up = m_session_up->getConcreteSymbolById<PDBSymbolFunc>(uid);
  if (!pdb_func_up)
    return 0;
  Block &parent_block = sc.function->GetBlock(false);
  num_added =
      ParseFunctionBlocksForPDBSymbol(sc, pdb_func_up->getVirtualAddress(),
                                      pdb_func_up.get(), &parent_block, true);
  return num_added;
}

size_t SymbolFilePDB::ParseTypes(const lldb_private::SymbolContext &sc) {
  lldbassert(sc.module_sp.get());
  if (!sc.comp_unit)
    return 0;

  size_t num_added = 0;
  auto compiland = GetPDBCompilandByUID(sc.comp_unit->GetID());
  if (!compiland)
    return 0;

  auto ParseTypesByTagFn = [&num_added, this](const PDBSymbol &raw_sym) {
    std::unique_ptr<IPDBEnumSymbols> results;
    PDB_SymType tags_to_search[] = { PDB_SymType::Enum, PDB_SymType::Typedef,
        PDB_SymType::UDT };
    for (auto tag : tags_to_search) {
      results = raw_sym.findAllChildren(tag);
      if (!results || results->getChildCount() == 0)
        continue;
      while (auto symbol = results->getNext()) {
        switch (symbol->getSymTag()) {
        case PDB_SymType::Enum:
        case PDB_SymType::UDT:
        case PDB_SymType::Typedef:
          break;
        default:
          continue;
        }

        // This should cause the type to get cached and stored in the `m_types`
        // lookup.
        if (!ResolveTypeUID(symbol->getSymIndexId()))
          continue;

        ++num_added;
      }
    }
  };

  if (sc.function) {
    auto pdb_func =
        m_session_up->getConcreteSymbolById<PDBSymbolFunc>(sc.function->GetID());
    if (!pdb_func)
      return 0;
    ParseTypesByTagFn(*pdb_func);
  } else {
    ParseTypesByTagFn(*compiland);

    // Also parse global types particularly coming from this compiland.
    // Unfortunately, PDB has no compiland information for each global type.
    // We have to parse them all. But ensure we only do this once.
    static bool parse_all_global_types = false;
    if (!parse_all_global_types) {
      ParseTypesByTagFn(*m_global_scope_up);
      parse_all_global_types = true;
    }
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
    if (type_list)
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
  uint32_t resolved_flags = 0;
  if (resolve_scope & eSymbolContextCompUnit ||
      resolve_scope & eSymbolContextVariable ||
      resolve_scope & eSymbolContextFunction ||
      resolve_scope & eSymbolContextBlock ||
      resolve_scope & eSymbolContextLineEntry) {
    addr_t file_vm_addr = so_addr.GetFileAddress();
    auto symbol_up =
        m_session_up->findSymbolByAddress(file_vm_addr, PDB_SymType::None);
    if (!symbol_up)
      return 0;

    auto cu_sp = GetCompileUnitContainsAddress(so_addr);
    if (!cu_sp) {
      if (resolved_flags | eSymbolContextVariable) {
        // TODO: Resolve variables
      }
      return 0;
    }
    sc.comp_unit = cu_sp.get();
    resolved_flags |= eSymbolContextCompUnit;
    lldbassert(sc.module_sp == cu_sp->GetModule());

    switch (symbol_up->getSymTag()) {
    case PDB_SymType::Function:
      if (resolve_scope & eSymbolContextFunction) {
        auto *pdb_func = llvm::dyn_cast<PDBSymbolFunc>(symbol_up.get());
        assert(pdb_func);
        auto func_uid = pdb_func->getSymIndexId();
        sc.function = sc.comp_unit->FindFunctionByUID(func_uid).get();
        if (sc.function == nullptr)
          sc.function = ParseCompileUnitFunctionForPDBFunc(pdb_func, sc);
        if (sc.function) {
          resolved_flags |= eSymbolContextFunction;
          if (resolve_scope & eSymbolContextBlock) {
            Block &block = sc.function->GetBlock(true);
            sc.block = block.FindBlockByID(sc.function->GetID());
            if (sc.block)
              resolved_flags |= eSymbolContextBlock;
          }
        }
      }
      break;
    default:
      break;
    }

    if (resolve_scope & eSymbolContextLineEntry) {
      if (auto *line_table = sc.comp_unit->GetLineTable()) {
        Address addr(so_addr);
        if (line_table->FindLineEntryByAddress(addr, sc.line_entry))
          resolved_flags |= eSymbolContextLineEntry;
      }
    }
  }
  return resolved_flags;
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
        if (llvm::is_contained(exts, file_extension.GetStringRef().lower())) {
          source_file_name = file_up->getFileName();
          break;
        }
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
      // For inline functions, we don't have to match the FileSpec since they
      // could be defined in headers other than file specified in FileSpec.
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

      // If we were asked to resolve line entries, add all entries to the line
      // table that match the requested line (or all lines if `line` == 0).
      if (resolve_scope & (eSymbolContextFunction | eSymbolContextBlock |
                           eSymbolContextLineEntry)) {
        bool has_line_table = ParseCompileUnitLineTable(sc, line);

        if ((resolve_scope & eSymbolContextLineEntry) && !has_line_table) {
          // The query asks for line entries, but we can't get them for the
          // compile unit. This is not normal for `line` = 0. So just assert it.
          assert(line && "Couldn't get all line entries!\n");

          // Current compiland does not have the requested line. Search next.
          continue;
        }

        if (resolve_scope & (eSymbolContextFunction | eSymbolContextBlock)) {
          if (!has_line_table)
            continue;

          auto *line_table = sc.comp_unit->GetLineTable();
          lldbassert(line_table);

          uint32_t num_line_entries = line_table->GetSize();
          // Skip the terminal line entry.
          --num_line_entries;

          // If `line `!= 0, see if we can resolve function for each line
          // entry in the line table.
          for (uint32_t line_idx = 0; line && line_idx < num_line_entries;
               ++line_idx) {
            if (!line_table->GetLineEntryAtIndex(line_idx, sc.line_entry))
              continue;

            auto file_vm_addr =
                sc.line_entry.range.GetBaseAddress().GetFileAddress();
            if (file_vm_addr == LLDB_INVALID_ADDRESS)
              continue;

            auto symbol_up =
                m_session_up->findSymbolByAddress(file_vm_addr,
                                                  PDB_SymType::Function);
            if (symbol_up) {
              auto func_uid = symbol_up->getSymIndexId();
              sc.function = sc.comp_unit->FindFunctionByUID(func_uid).get();
              if (sc.function == nullptr) {
                auto pdb_func = llvm::dyn_cast<PDBSymbolFunc>(symbol_up.get());
                assert(pdb_func);
                sc.function = ParseCompileUnitFunctionForPDBFunc(pdb_func, sc);
              }
              if (sc.function && (resolve_scope & eSymbolContextBlock)) {
                Block &block = sc.function->GetBlock(true);
                sc.block = block.FindBlockByID(sc.function->GetID());
              }
            }
            sc_list.Append(sc);
          }
        } else if (has_line_table) {
          // We can parse line table for the compile unit. But no query to
          // resolve function or block. We append `sc` to the list anyway.
          sc_list.Append(sc);
        }
      } else {
        // No query for line entry, function or block. But we have a valid
        // compile unit, append `sc` to the list.
        sc_list.Append(sc);
      }
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

bool SymbolFilePDB::ResolveFunction(llvm::pdb::PDBSymbolFunc *pdb_func,
                                    bool include_inlines,
                                    lldb_private::SymbolContextList &sc_list) {
  if (!pdb_func)
    return false;
  lldb_private::SymbolContext sc;
  auto file_vm_addr = pdb_func->getVirtualAddress();
  if (file_vm_addr == LLDB_INVALID_ADDRESS)
    return false;

  Address so_addr(file_vm_addr);
  sc.comp_unit = GetCompileUnitContainsAddress(so_addr).get();
  if (!sc.comp_unit)
    return false;
  sc.module_sp = sc.comp_unit->GetModule();
  auto symbol_up =
      m_session_up->findSymbolByAddress(file_vm_addr, PDB_SymType::Function);
  if (!symbol_up)
    return false;

  auto *func = llvm::dyn_cast<PDBSymbolFunc>(symbol_up.get());
  assert(func);
  sc.function = ParseCompileUnitFunctionForPDBFunc(func, sc);
  if (!sc.function)
    return false;

  sc_list.Append(sc);
  return true;
}

bool SymbolFilePDB::ResolveFunction(uint32_t uid, bool include_inlines,
                                    lldb_private::SymbolContextList &sc_list) {
  auto pdb_func_up =
      m_session_up->getConcreteSymbolById<PDBSymbolFunc>(uid);
  if (!pdb_func_up && !(include_inlines && pdb_func_up->hasInlineAttribute()))
    return false;
  return ResolveFunction(pdb_func_up.get(), include_inlines, sc_list);
}

void SymbolFilePDB::CacheFunctionNames() {
  if (!m_func_full_names.IsEmpty())
    return;

  std::map<uint64_t, uint32_t> addr_ids;

  if (auto results_up = m_global_scope_up->findAllChildren<PDBSymbolFunc>()) {
    while (auto pdb_func_up = results_up->getNext()) {
      if (pdb_func_up->isCompilerGenerated())
        continue;

      auto name = pdb_func_up->getName();
      auto demangled_name = pdb_func_up->getUndecoratedName();
      if (name.empty() && demangled_name.empty())
        continue;

      auto uid = pdb_func_up->getSymIndexId();
      if (!demangled_name.empty() && pdb_func_up->getVirtualAddress())
        addr_ids.insert(std::make_pair(pdb_func_up->getVirtualAddress(), uid));

      if (auto parent = pdb_func_up->getClassParent()) {

        // PDB have symbols for class/struct methods or static methods in Enum
        // Class. We won't bother to check if the parent is UDT or Enum here.
        m_func_method_names.Append(ConstString(name), uid);

        ConstString cstr_name(name);

        // To search a method name, like NS::Class:MemberFunc, LLDB searches its
        // base name, i.e. MemberFunc by default. Since PDBSymbolFunc does not
        // have inforamtion of this, we extract base names and cache them by our
        // own effort.
        llvm::StringRef basename;
        CPlusPlusLanguage::MethodName cpp_method(cstr_name);
        if (cpp_method.IsValid()) {
          llvm::StringRef context;
          basename = cpp_method.GetBasename();
          if (basename.empty())
            CPlusPlusLanguage::ExtractContextAndIdentifier(name.c_str(),
                                                           context, basename);
        }

        if (!basename.empty())
          m_func_base_names.Append(ConstString(basename), uid);
        else {
          m_func_base_names.Append(ConstString(name), uid);
        }

        if (!demangled_name.empty())
          m_func_full_names.Append(ConstString(demangled_name), uid);

      } else {
        // Handle not-method symbols.

        // The function name might contain namespace, or its lexical scope. It
        // is not safe to get its base name by applying same scheme as we deal
        // with the method names.
        // FIXME: Remove namespace if function is static in a scope.
        m_func_base_names.Append(ConstString(name), uid);

        if (name == "main") {
          m_func_full_names.Append(ConstString(name), uid);

          if (!demangled_name.empty() && name != demangled_name) {
            m_func_full_names.Append(ConstString(demangled_name), uid);
            m_func_base_names.Append(ConstString(demangled_name), uid);
          }
        } else if (!demangled_name.empty()) {
          m_func_full_names.Append(ConstString(demangled_name), uid);
        } else {
          m_func_full_names.Append(ConstString(name), uid);
        }
      }
    }
  }

  if (auto results_up =
      m_global_scope_up->findAllChildren<PDBSymbolPublicSymbol>()) {
    while (auto pub_sym_up = results_up->getNext()) {
      if (!pub_sym_up->isFunction())
        continue;
      auto name = pub_sym_up->getName();
      if (name.empty())
        continue;

      if (CPlusPlusLanguage::IsCPPMangledName(name.c_str())) {
        auto vm_addr = pub_sym_up->getVirtualAddress();

        // PDB public symbol has mangled name for its associated function.
        if (vm_addr && addr_ids.find(vm_addr) != addr_ids.end()) {
          // Cache mangled name.
          m_func_full_names.Append(ConstString(name), addr_ids[vm_addr]);
        }
      }
    }
  }
  // Sort them before value searching is working properly
  m_func_full_names.Sort();
  m_func_full_names.SizeToFit();
  m_func_method_names.Sort();
  m_func_method_names.SizeToFit();
  m_func_base_names.Sort();
  m_func_base_names.SizeToFit();
}

uint32_t SymbolFilePDB::FindFunctions(
    const lldb_private::ConstString &name,
    const lldb_private::CompilerDeclContext *parent_decl_ctx,
    uint32_t name_type_mask, bool include_inlines, bool append,
    lldb_private::SymbolContextList &sc_list) {
  if (!append)
    sc_list.Clear();
  lldbassert((name_type_mask & eFunctionNameTypeAuto) == 0);

  if (name_type_mask == eFunctionNameTypeNone)
    return 0;
  if (!DeclContextMatchesThisSymbolFile(parent_decl_ctx))
    return 0;
  if (name.IsEmpty())
    return 0;

  auto old_size = sc_list.GetSize();
  if (name_type_mask & eFunctionNameTypeFull ||
      name_type_mask & eFunctionNameTypeBase ||
      name_type_mask & eFunctionNameTypeMethod) {
    CacheFunctionNames();

    std::set<uint32_t> resolved_ids;
    auto ResolveFn = [include_inlines, &name, &sc_list, &resolved_ids, this] (
        UniqueCStringMap<uint32_t> &Names)
    {
      std::vector<uint32_t> ids;
      if (Names.GetValues(name, ids)) {
        for (auto id : ids) {
          if (resolved_ids.find(id) == resolved_ids.end()) {
            if (ResolveFunction(id, include_inlines, sc_list))
              resolved_ids.insert(id);
          }
        }
      }
    };
    if (name_type_mask & eFunctionNameTypeFull) {
      ResolveFn(m_func_full_names);
    }
    if (name_type_mask & eFunctionNameTypeBase) {
      ResolveFn(m_func_base_names);
    }
    if (name_type_mask & eFunctionNameTypeMethod) {
      ResolveFn(m_func_method_names);
    }
  }
  return sc_list.GetSize() - old_size;
}

uint32_t
SymbolFilePDB::FindFunctions(const lldb_private::RegularExpression &regex,
                             bool include_inlines, bool append,
                             lldb_private::SymbolContextList &sc_list) {
  if (!append)
    sc_list.Clear();
  if (!regex.IsValid())
    return 0;

  auto old_size = sc_list.GetSize();
  CacheFunctionNames();

  std::set<uint32_t> resolved_ids;
  auto ResolveFn = [&regex, include_inlines, &sc_list, &resolved_ids, this] (
      UniqueCStringMap<uint32_t> &Names)
  {
    std::vector<uint32_t> ids;
    if (Names.GetValues(regex, ids)) {
      for (auto id : ids) {
        if (resolved_ids.find(id) == resolved_ids.end())
          if (ResolveFunction(id, include_inlines, sc_list))
            resolved_ids.insert(id);
      }
    }
  };
  ResolveFn(m_func_full_names);
  ResolveFn(m_func_base_names);

  return sc_list.GetSize() - old_size;
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
  if (!DeclContextMatchesThisSymbolFile(parent_decl_ctx))
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
  if (name.empty())
    return;
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

void
SymbolFilePDB::GetTypesForPDBSymbol(const llvm::pdb::PDBSymbol *pdb_symbol,
                                    uint32_t type_mask,
                                    TypeCollection &type_collection) {
  if (!pdb_symbol)
    return;

  bool can_parse = false;
  switch (pdb_symbol->getSymTag()) {
  case PDB_SymType::ArrayType:
    can_parse = ((type_mask & eTypeClassArray) != 0);
    break;
  case PDB_SymType::BuiltinType:
    can_parse = ((type_mask & eTypeClassBuiltin) != 0);
    break;
  case PDB_SymType::Enum:
    can_parse = ((type_mask & eTypeClassEnumeration) != 0);
    break;
  case PDB_SymType::Function:
  case PDB_SymType::FunctionSig:
    can_parse = ((type_mask & eTypeClassFunction) != 0);
    break;
  case PDB_SymType::PointerType:
    can_parse = ((type_mask & (eTypeClassPointer | eTypeClassBlockPointer |
                               eTypeClassMemberPointer)) != 0);
    break;
  case PDB_SymType::Typedef:
    can_parse = ((type_mask & eTypeClassTypedef) != 0);
    break;
  case PDB_SymType::UDT: {
    auto *udt = llvm::dyn_cast<PDBSymbolTypeUDT>(pdb_symbol);
    assert(udt);
    can_parse = (udt->getUdtKind() != PDB_UdtType::Interface &&
        ((type_mask & (eTypeClassClass | eTypeClassStruct |
                       eTypeClassUnion)) != 0));
  } break;
  default:break;
  }

  if (can_parse) {
    if (auto *type = ResolveTypeUID(pdb_symbol->getSymIndexId())) {
      auto result =
          std::find(type_collection.begin(), type_collection.end(), type);
      if (result == type_collection.end())
        type_collection.push_back(type);
    }
  }

  auto results_up = pdb_symbol->findAllChildren();
  while (auto symbol_up = results_up->getNext())
    GetTypesForPDBSymbol(symbol_up.get(), type_mask, type_collection);
}

size_t SymbolFilePDB::GetTypes(lldb_private::SymbolContextScope *sc_scope,
                               uint32_t type_mask,
                               lldb_private::TypeList &type_list) {
  TypeCollection type_collection;
  uint32_t old_size = type_list.GetSize();
  CompileUnit *cu = sc_scope ?
      sc_scope->CalculateSymbolContextCompileUnit() : nullptr;
  if (cu) {
    auto compiland_up = GetPDBCompilandByUID(cu->GetID());
    GetTypesForPDBSymbol(compiland_up.get(), type_mask, type_collection);
  } else {
    for (uint32_t cu_idx = 0; cu_idx < GetNumCompileUnits(); ++cu_idx) {
      auto cu_sp = ParseCompileUnitAtIndex(cu_idx);
      if (cu_sp.get()) {
        auto compiland_up = GetPDBCompilandByUID(cu_sp->GetID());
        GetTypesForPDBSymbol(compiland_up.get(), type_mask, type_collection);
      }
    }
  }

  for (auto type : type_collection) {
    type->GetForwardCompilerType();
    type_list.Insert(type->shared_from_this());
  }
  return type_list.GetSize() - old_size;
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

  lldb::LanguageType lang;
  auto details = compiland_up->findOneChild<PDBSymbolCompilandDetails>();
  if (!details)
    lang = lldb::eLanguageTypeC_plus_plus;
  else
    lang = TranslateLanguage(details->getLanguage());

  if (lang == lldb::LanguageType::eLanguageTypeUnknown)
    return CompUnitSP();

  std::string path = GetSourceFileNameForPDBCompiland(compiland_up.get());
  if (path.empty())
    return CompUnitSP();

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

lldb::CompUnitSP SymbolFilePDB::GetCompileUnitContainsAddress(
     const lldb_private::Address &so_addr) {
  lldb::addr_t file_vm_addr = so_addr.GetFileAddress();
  if (file_vm_addr == LLDB_INVALID_ADDRESS)
    return nullptr;

  auto lines_up =
      m_session_up->findLineNumbersByAddress(file_vm_addr, /*Length=*/200);
  if (!lines_up)
    return nullptr;

  auto first_line_up = lines_up->getNext();
  if (!first_line_up)
    return nullptr;
  auto compiland_up = GetPDBCompilandByUID(first_line_up->getCompilandId());
  if (compiland_up) {
    return ParseCompileUnitForUID(compiland_up->getSymIndexId());
  }

  return nullptr;
}

Mangled
SymbolFilePDB::GetMangledForPDBFunc(const llvm::pdb::PDBSymbolFunc *pdb_func) {
  Mangled mangled;
  if (!pdb_func)
    return mangled;

  auto func_name = pdb_func->getName();
  auto func_undecorated_name = pdb_func->getUndecoratedName();
  std::string func_decorated_name;

  // Seek from public symbols for non-static function's decorated name if any.
  // For static functions, they don't have undecorated names and aren't exposed
  // in Public Symbols either.
  if (!func_undecorated_name.empty()) {
    auto result_up =
        m_global_scope_up->findChildren(PDB_SymType::PublicSymbol,
                                        func_undecorated_name,
                                        PDB_NameSearchFlags::NS_UndecoratedName);
    if (result_up) {
      while (auto symbol_up = result_up->getNext()) {
        // For a public symbol, it is unique.
        lldbassert(result_up->getChildCount() == 1);
        if (auto *pdb_public_sym =
            llvm::dyn_cast_or_null<PDBSymbolPublicSymbol>(symbol_up.get())) {
          if (pdb_public_sym->isFunction()) {
            func_decorated_name = pdb_public_sym->getName();
            break;
          }
        }
      }
    }
  }
  if (!func_decorated_name.empty()) {
    mangled.SetMangledName(ConstString(func_decorated_name));

    // For MSVC, format of C funciton's decorated name depends on calling
    // conventon. Unfortunately none of the format is recognized by current
    // LLDB. For example, `_purecall` is a __cdecl C function. From PDB,
    // `__purecall` is retrieved as both its decorated and
    // undecorated name (using PDBSymbolFunc::getUndecoratedName method).
    // However `__purecall` string is not treated as mangled in LLDB
    // (neither `?` nor `_Z` prefix). Mangled::GetDemangledName method
    // will fail internally and caches an empty string as its undecorated
    // name. So we will face a contradition here for the same symbol:
    //   non-empty undecorated name from PDB
    //   empty undecorated name from LLDB
    if (!func_undecorated_name.empty() &&
        mangled.GetDemangledName(mangled.GuessLanguage()).IsEmpty())
      mangled.SetDemangledName(ConstString(func_undecorated_name));

    // LLDB uses several flags to control how a C++ decorated name is
    // undecorated for MSVC. See `safeUndecorateName` in Class Mangled.
    // So the yielded name could be different from what we retrieve from
    // PDB source unless we also apply same flags in getting undecorated
    // name through PDBSymbolFunc::getUndecoratedNameEx method.
    if (!func_undecorated_name.empty() &&
        mangled.GetDemangledName(mangled.GuessLanguage()) !=
            ConstString(func_undecorated_name))
      mangled.SetDemangledName(ConstString(func_undecorated_name));
  } else if (!func_undecorated_name.empty()) {
    mangled.SetDemangledName(ConstString(func_undecorated_name));
  } else if (!func_name.empty())
    mangled.SetValue(ConstString(func_name), false);

  return mangled;
}

bool SymbolFilePDB::DeclContextMatchesThisSymbolFile(
    const lldb_private::CompilerDeclContext *decl_ctx) {
  if (decl_ctx == nullptr || !decl_ctx->IsValid())
    return true;

  TypeSystem *decl_ctx_type_system = decl_ctx->GetTypeSystem();
  if (!decl_ctx_type_system)
    return false;
  TypeSystem *type_system = GetTypeSystemForLanguage(
      decl_ctx_type_system->GetMinimumLanguage(nullptr));
  if (decl_ctx_type_system == type_system)
    return true; // The type systems match, return true

  return false;
}
