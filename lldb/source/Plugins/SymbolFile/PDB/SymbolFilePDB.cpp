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
#include "lldb/Symbol/TypeMap.h"

#include "llvm/DebugInfo/PDB/GenericError.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"
#include "llvm/DebugInfo/PDB/IPDBLineNumber.h"
#include "llvm/DebugInfo/PDB/IPDBSourceFile.h"
#include "llvm/DebugInfo/PDB/PDBSymbol.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompiland.h"
#include "llvm/DebugInfo/PDB/PDBSymbolCompilandDetails.h"
#include "llvm/DebugInfo/PDB/PDBSymbolExe.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFunc.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugEnd.h"
#include "llvm/DebugInfo/PDB/PDBSymbolFuncDebugStart.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeEnum.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeTypedef.h"
#include "llvm/DebugInfo/PDB/PDBSymbolTypeUDT.h"

#include "Plugins/SymbolFile/PDB/PDBASTParser.h"

#include <regex>

using namespace lldb_private;
using namespace llvm::pdb;

namespace
{
lldb::LanguageType
TranslateLanguage(PDB_Lang lang)
{
    switch (lang)
    {
        case PDB_Lang::Cpp:
            return lldb::LanguageType::eLanguageTypeC_plus_plus;
        case PDB_Lang::C:
            return lldb::LanguageType::eLanguageTypeC;
        default:
            return lldb::LanguageType::eLanguageTypeUnknown;
        }
    }

    bool
    ShouldAddLine(uint32_t requested_line, uint32_t actual_line, uint32_t addr_length)
    {
        return ((requested_line == 0 || actual_line == requested_line) && addr_length > 0);
    }
}

void
SymbolFilePDB::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
                                  DebuggerInitialize);
}

void
SymbolFilePDB::Terminate()
{
    PluginManager::UnregisterPlugin(CreateInstance);
}

void
SymbolFilePDB::DebuggerInitialize(lldb_private::Debugger &debugger)
{
}

lldb_private::ConstString
SymbolFilePDB::GetPluginNameStatic()
{
    static ConstString g_name("pdb");
    return g_name;
}

const char *
SymbolFilePDB::GetPluginDescriptionStatic()
{
    return "Microsoft PDB debug symbol file reader.";
}

lldb_private::SymbolFile *
SymbolFilePDB::CreateInstance(lldb_private::ObjectFile *obj_file)
{
    return new SymbolFilePDB(obj_file);
}

SymbolFilePDB::SymbolFilePDB(lldb_private::ObjectFile *object_file)
    : SymbolFile(object_file), m_cached_compile_unit_count(0)
{
}

SymbolFilePDB::~SymbolFilePDB()
{
}

uint32_t
SymbolFilePDB::CalculateAbilities()
{
    if (!m_session_up)
    {
        // Lazily load and match the PDB file, but only do this once.
        std::string exePath = m_obj_file->GetFileSpec().GetPath();
        auto error = loadDataForEXE(PDB_ReaderType::DIA, llvm::StringRef(exePath), m_session_up);
        if (error)
        {
            llvm::consumeError(std::move(error));
            return 0;
        }
    }
    return CompileUnits | LineTables;
}

void
SymbolFilePDB::InitializeObject()
{
    lldb::addr_t obj_load_address = m_obj_file->GetFileOffset();
    m_session_up->setLoadAddress(obj_load_address);

    TypeSystem *type_system = GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus);
    ClangASTContext *clang_type_system = llvm::dyn_cast_or_null<ClangASTContext>(type_system);
    m_tu_decl_ctx_up = llvm::make_unique<CompilerDeclContext>(type_system, clang_type_system->GetTranslationUnitDecl());
}

uint32_t
SymbolFilePDB::GetNumCompileUnits()
{
    if (m_cached_compile_unit_count == 0)
    {
        auto global = m_session_up->getGlobalScope();
        auto compilands = global->findAllChildren<PDBSymbolCompiland>();
        m_cached_compile_unit_count = compilands->getChildCount();

        // The linker can inject an additional "dummy" compilation unit into the PDB.
        // Ignore this special compile unit for our purposes, if it is there.  It is
        // always the last one.
        auto last_cu = compilands->getChildAtIndex(m_cached_compile_unit_count - 1);
        std::string name = last_cu->getName();
        if (name == "* Linker *")
            --m_cached_compile_unit_count;
    }
    return m_cached_compile_unit_count;
}

lldb::CompUnitSP
SymbolFilePDB::ParseCompileUnitAtIndex(uint32_t index)
{
    auto global = m_session_up->getGlobalScope();
    auto compilands = global->findAllChildren<PDBSymbolCompiland>();
    auto cu = compilands->getChildAtIndex(index);

    uint32_t id = cu->getSymIndexId();

    return ParseCompileUnitForSymIndex(id);
}

lldb::LanguageType
SymbolFilePDB::ParseCompileUnitLanguage(const lldb_private::SymbolContext &sc)
{
    // What fields should I expect to be filled out on the SymbolContext?  Is it
    // safe to assume that `sc.comp_unit` is valid?
    if (!sc.comp_unit)
        return lldb::eLanguageTypeUnknown;

    auto cu = m_session_up->getConcreteSymbolById<PDBSymbolCompiland>(sc.comp_unit->GetID());
    if (!cu)
        return lldb::eLanguageTypeUnknown;
    auto details = cu->findOneChild<PDBSymbolCompilandDetails>();
    if (!details)
        return lldb::eLanguageTypeUnknown;
    return TranslateLanguage(details->getLanguage());
}

size_t
SymbolFilePDB::ParseCompileUnitFunctions(const lldb_private::SymbolContext &sc)
{
    // TODO: Implement this
    return size_t();
}

bool
SymbolFilePDB::ParseCompileUnitLineTable(const lldb_private::SymbolContext &sc)
{
    return ParseCompileUnitLineTable(sc, 0);
}

bool
SymbolFilePDB::ParseCompileUnitDebugMacros(const lldb_private::SymbolContext &sc)
{
    // PDB doesn't contain information about macros
    return false;
}

bool
SymbolFilePDB::ParseCompileUnitSupportFiles(const lldb_private::SymbolContext &sc,
                                            lldb_private::FileSpecList &support_files)
{
    if (!sc.comp_unit)
        return false;

    // In theory this is unnecessary work for us, because all of this information is easily
    // (and quickly) accessible from DebugInfoPDB, so caching it a second time seems like a waste.
    // Unfortunately, there's no good way around this short of a moderate refactor, since SymbolVendor
    // depends on being able to cache this list.
    auto cu = m_session_up->getConcreteSymbolById<PDBSymbolCompiland>(sc.comp_unit->GetID());
    if (!cu)
        return false;
    auto files = m_session_up->getSourceFilesForCompiland(*cu);
    if (!files || files->getChildCount() == 0)
        return false;

    while (auto file = files->getNext())
    {
        FileSpec spec(file->getFileName(), false);
        support_files.Append(spec);
    }
    return true;
}

bool
SymbolFilePDB::ParseImportedModules(const lldb_private::SymbolContext &sc,
                                    std::vector<lldb_private::ConstString> &imported_modules)
{
    // PDB does not yet support module debug info
    return false;
}

size_t
SymbolFilePDB::ParseFunctionBlocks(const lldb_private::SymbolContext &sc)
{
    // TODO: Implement this
    return size_t();
}

size_t
SymbolFilePDB::ParseTypes(const lldb_private::SymbolContext &sc)
{
    // TODO: Implement this
    return size_t();
}

size_t
SymbolFilePDB::ParseVariablesForContext(const lldb_private::SymbolContext &sc)
{
    // TODO: Implement this
    return size_t();
}

lldb_private::Type *
SymbolFilePDB::ResolveTypeUID(lldb::user_id_t type_uid)
{
    auto find_result = m_types.find(type_uid);
    if (find_result != m_types.end())
        return find_result->second.get();

    TypeSystem *type_system = GetTypeSystemForLanguage(lldb::eLanguageTypeC_plus_plus);
    ClangASTContext *clang_type_system = llvm::dyn_cast_or_null<ClangASTContext>(type_system);
    if (!clang_type_system)
        return nullptr;
    PDBASTParser *pdb = llvm::dyn_cast<PDBASTParser>(clang_type_system->GetPDBParser());
    if (!pdb)
        return nullptr;

    auto pdb_type = m_session_up->getSymbolById(type_uid);
    if (pdb_type == nullptr)
        return nullptr;

    lldb::TypeSP result = pdb->CreateLLDBTypeFromPDBType(*pdb_type);
    m_types.insert(std::make_pair(type_uid, result));
    return result.get();
}

bool
SymbolFilePDB::CompleteType(lldb_private::CompilerType &compiler_type)
{
    // TODO: Implement this
    return false;
}

lldb_private::CompilerDecl
SymbolFilePDB::GetDeclForUID(lldb::user_id_t uid)
{
    return lldb_private::CompilerDecl();
}

lldb_private::CompilerDeclContext
SymbolFilePDB::GetDeclContextForUID(lldb::user_id_t uid)
{
    // PDB always uses the translation unit decl context for everything.  We can improve this later
    // but it's not easy because PDB doesn't provide a high enough level of type fidelity in this area.
    return *m_tu_decl_ctx_up;
}

lldb_private::CompilerDeclContext
SymbolFilePDB::GetDeclContextContainingUID(lldb::user_id_t uid)
{
    return *m_tu_decl_ctx_up;
}

void
SymbolFilePDB::ParseDeclsForContext(lldb_private::CompilerDeclContext decl_ctx)
{
}

uint32_t
SymbolFilePDB::ResolveSymbolContext(const lldb_private::Address &so_addr, uint32_t resolve_scope,
                                    lldb_private::SymbolContext &sc)
{
    return uint32_t();
}

uint32_t
SymbolFilePDB::ResolveSymbolContext(const lldb_private::FileSpec &file_spec, uint32_t line, bool check_inlines,
                                    uint32_t resolve_scope, lldb_private::SymbolContextList &sc_list)
{
    if (resolve_scope & lldb::eSymbolContextCompUnit)
    {
        // Locate all compilation units with line numbers referencing the specified file.  For example, if
        // `file_spec` is <vector>, then this should return all source files and header files that reference
        // <vector>, either directly or indirectly.
        auto compilands =
            m_session_up->findCompilandsForSourceFile(file_spec.GetPath(), PDB_NameSearchFlags::NS_CaseInsensitive);

        // For each one, either find get its previously parsed data, or parse it afresh and add it to
        // the symbol context list.
        while (auto compiland = compilands->getNext())
        {
            // If we're not checking inlines, then don't add line information for this file unless the FileSpec
            // matches.
            if (!check_inlines)
            {
                // `getSourceFileName` returns the basename of the original source file used to generate this compiland.
                // It does not return the full path.  Currently the only way to get that is to do a basename lookup to
                // get the IPDBSourceFile, but this is ambiguous in the case of two source files with the same name
                // contributing to the same compiland.  This is a moderately extreme edge case, so we consider this ok
                // for now, although we need to find a long term solution.
                std::string source_file = compiland->getSourceFileName();
                auto pdb_file = m_session_up->findOneSourceFile(compiland.get(), source_file,
                                                                PDB_NameSearchFlags::NS_CaseInsensitive);
                source_file = pdb_file->getFileName();
                FileSpec this_spec(source_file, false, FileSpec::ePathSyntaxWindows);
                if (!file_spec.FileEquals(this_spec))
                    continue;
            }

            SymbolContext sc;
            auto cu = ParseCompileUnitForSymIndex(compiland->getSymIndexId());
            sc.comp_unit = cu.get();
            sc.module_sp = cu->GetModule();
            sc_list.Append(sc);

            // If we were asked to resolve line entries, add all entries to the line table that match the requested
            // line (or all lines if `line` == 0)
            if (resolve_scope & lldb::eSymbolContextLineEntry)
                ParseCompileUnitLineTable(sc, line);
        }
    }
    return sc_list.GetSize();
}

uint32_t
SymbolFilePDB::FindGlobalVariables(const lldb_private::ConstString &name,
                                   const lldb_private::CompilerDeclContext *parent_decl_ctx, bool append,
                                   uint32_t max_matches, lldb_private::VariableList &variables)
{
    return uint32_t();
}

uint32_t
SymbolFilePDB::FindGlobalVariables(const lldb_private::RegularExpression &regex, bool append, uint32_t max_matches,
                                   lldb_private::VariableList &variables)
{
    return uint32_t();
}

uint32_t
SymbolFilePDB::FindFunctions(const lldb_private::ConstString &name,
                             const lldb_private::CompilerDeclContext *parent_decl_ctx, uint32_t name_type_mask,
                             bool include_inlines, bool append, lldb_private::SymbolContextList &sc_list)
{
    return uint32_t();
}

uint32_t
SymbolFilePDB::FindFunctions(const lldb_private::RegularExpression &regex, bool include_inlines, bool append,
                             lldb_private::SymbolContextList &sc_list)
{
    return uint32_t();
}

void
SymbolFilePDB::GetMangledNamesForFunction(const std::string &scope_qualified_name,
                                          std::vector<lldb_private::ConstString> &mangled_names)
{
}

uint32_t
SymbolFilePDB::FindTypes(const lldb_private::SymbolContext &sc, const lldb_private::ConstString &name,
                         const lldb_private::CompilerDeclContext *parent_decl_ctx, bool append, uint32_t max_matches,
                         llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
                         lldb_private::TypeMap &types)
{
    if (!append)
        types.Clear();
    if (!name)
        return 0;

    searched_symbol_files.clear();
    searched_symbol_files.insert(this);

    std::string name_str = name.AsCString();

    // If this might be a regex, we have to return EVERY symbol and process them one by one, which is going
    // to destroy performance on large PDB files.  So try really hard not to use a regex match.
    if (name_str.find_first_of("[]?*.-+\\") != std::string::npos)
        FindTypesByRegex(name_str, max_matches, types);
    else
        FindTypesByName(name_str, max_matches, types);
    return types.GetSize();
}

void
SymbolFilePDB::FindTypesByRegex(const std::string &regex, uint32_t max_matches, lldb_private::TypeMap &types)
{
    // When searching by regex, we need to go out of our way to limit the search space as much as possible, since
    // the way this is implemented is by searching EVERYTHING in the PDB and manually doing a regex compare.  PDB
    // library isn't optimized for regex searches or searches across multiple symbol types at the same time, so the
    // best we can do is to search enums, then typedefs, then classes one by one, and do a regex compare against all
    // of them.
    PDB_SymType tags_to_search[] = {PDB_SymType::Enum, PDB_SymType::Typedef, PDB_SymType::UDT};
    auto global = m_session_up->getGlobalScope();
    std::unique_ptr<IPDBEnumSymbols> results;

    std::regex re(regex);

    uint32_t matches = 0;

    for (auto tag : tags_to_search)
    {
        results = global->findAllChildren(tag);
        while (auto result = results->getNext())
        {
            if (max_matches > 0 && matches >= max_matches)
                break;

            std::string type_name;
            if (auto enum_type = llvm::dyn_cast<PDBSymbolTypeEnum>(result.get()))
                type_name = enum_type->getName();
            else if (auto typedef_type = llvm::dyn_cast<PDBSymbolTypeTypedef>(result.get()))
                type_name = typedef_type->getName();
            else if (auto class_type = llvm::dyn_cast<PDBSymbolTypeUDT>(result.get()))
                type_name = class_type->getName();
            else
            {
                // We're only looking for types that have names.  Skip symbols, as well as
                // unnamed types such as arrays, pointers, etc.
                continue;
            }

            if (!std::regex_match(type_name, re))
                continue;

            // This should cause the type to get cached and stored in the `m_types` lookup.
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

void
SymbolFilePDB::FindTypesByName(const std::string &name, uint32_t max_matches, lldb_private::TypeMap &types)
{
    auto global = m_session_up->getGlobalScope();
    std::unique_ptr<IPDBEnumSymbols> results;
    results = global->findChildren(PDB_SymType::None, name.c_str(), PDB_NameSearchFlags::NS_Default);

    uint32_t matches = 0;

    while (auto result = results->getNext())
    {
        if (max_matches > 0 && matches >= max_matches)
            break;
        switch (result->getSymTag())
        {
            case PDB_SymType::Enum:
            case PDB_SymType::UDT:
            case PDB_SymType::Typedef:
                break;
            default:
                // We're only looking for types that have names.  Skip symbols, as well as
                // unnamed types such as arrays, pointers, etc.
                continue;
        }

        // This should cause the type to get cached and stored in the `m_types` lookup.
        if (!ResolveTypeUID(result->getSymIndexId()))
            continue;

        auto iter = m_types.find(result->getSymIndexId());
        if (iter == m_types.end())
            continue;
        types.Insert(iter->second);
        ++matches;
    }
}

size_t
SymbolFilePDB::FindTypes(const std::vector<lldb_private::CompilerContext> &contexts, bool append,
                         lldb_private::TypeMap &types)
{
    return 0;
}

lldb_private::TypeList *
SymbolFilePDB::GetTypeList()
{
    return nullptr;
}

size_t
SymbolFilePDB::GetTypes(lldb_private::SymbolContextScope *sc_scope, uint32_t type_mask,
                        lldb_private::TypeList &type_list)
{
    return size_t();
}

lldb_private::TypeSystem *
SymbolFilePDB::GetTypeSystemForLanguage(lldb::LanguageType language)
{
    auto type_system = m_obj_file->GetModule()->GetTypeSystemForLanguage(language);
    if (type_system)
        type_system->SetSymbolFile(this);
    return type_system;
}

lldb_private::CompilerDeclContext
SymbolFilePDB::FindNamespace(const lldb_private::SymbolContext &sc, const lldb_private::ConstString &name,
                             const lldb_private::CompilerDeclContext *parent_decl_ctx)
{
    return lldb_private::CompilerDeclContext();
}

lldb_private::ConstString
SymbolFilePDB::GetPluginName()
{
    static ConstString g_name("pdb");
    return g_name;
}

uint32_t
SymbolFilePDB::GetPluginVersion()
{
    return 1;
}

IPDBSession &
SymbolFilePDB::GetPDBSession()
{
    return *m_session_up;
}

const IPDBSession &
SymbolFilePDB::GetPDBSession() const
{
    return *m_session_up;
}

lldb::CompUnitSP
SymbolFilePDB::ParseCompileUnitForSymIndex(uint32_t id)
{
    auto found_cu = m_comp_units.find(id);
    if (found_cu != m_comp_units.end())
        return found_cu->second;

    auto cu = m_session_up->getConcreteSymbolById<PDBSymbolCompiland>(id);

    // `getSourceFileName` returns the basename of the original source file used to generate this compiland.  It does
    // not return the full path.  Currently the only way to get that is to do a basename lookup to get the
    // IPDBSourceFile, but this is ambiguous in the case of two source files with the same name contributing to the
    // same compiland. This is a moderately extreme edge case, so we consider this ok for now, although we need to find
    // a long term solution.
    auto file =
        m_session_up->findOneSourceFile(cu.get(), cu->getSourceFileName(), PDB_NameSearchFlags::NS_CaseInsensitive);
    std::string path = file->getFileName();

    lldb::LanguageType lang;
    auto details = cu->findOneChild<PDBSymbolCompilandDetails>();
    if (!details)
        lang = lldb::eLanguageTypeC_plus_plus;
    else
        lang = TranslateLanguage(details->getLanguage());

    // Don't support optimized code for now, DebugInfoPDB does not return this information.
    LazyBool optimized = eLazyBoolNo;
    auto result = std::make_shared<CompileUnit>(m_obj_file->GetModule(), nullptr, path.c_str(), id, lang, optimized);
    m_comp_units.insert(std::make_pair(id, result));
    return result;
}

bool
SymbolFilePDB::ParseCompileUnitLineTable(const lldb_private::SymbolContext &sc, uint32_t match_line)
{
    auto global = m_session_up->getGlobalScope();
    auto cu = m_session_up->getConcreteSymbolById<PDBSymbolCompiland>(sc.comp_unit->GetID());

    // LineEntry needs the *index* of the file into the list of support files returned by
    // ParseCompileUnitSupportFiles.  But the underlying SDK gives us a globally unique
    // idenfitifier in the namespace of the PDB.  So, we have to do a mapping so that we
    // can hand out indices.
    llvm::DenseMap<uint32_t, uint32_t> index_map;
    BuildSupportFileIdToSupportFileIndexMap(*cu, index_map);
    auto line_table = llvm::make_unique<LineTable>(sc.comp_unit);

    // Find contributions to `cu` from all source and header files.
    std::string path = sc.comp_unit->GetPath();
    auto files = m_session_up->getSourceFilesForCompiland(*cu);

    // For each source and header file, create a LineSequence for contributions to the cu
    // from that file, and add the sequence.
    while (auto file = files->getNext())
    {
        std::unique_ptr<LineSequence> sequence(line_table->CreateLineSequenceContainer());
        auto lines = m_session_up->findLineNumbers(*cu, *file);
        int entry_count = lines->getChildCount();

        uint64_t prev_addr;
        uint32_t prev_length;
        uint32_t prev_line;
        uint32_t prev_source_idx;

        for (int i = 0; i < entry_count; ++i)
        {
            auto line = lines->getChildAtIndex(i);

            uint64_t lno = line->getLineNumber();
            uint64_t addr = line->getVirtualAddress();
            uint32_t length = line->getLength();
            uint32_t source_id = line->getSourceFileId();
            uint32_t col = line->getColumnNumber();
            uint32_t source_idx = index_map[source_id];

            // There was a gap between the current entry and the previous entry if the addresses don't perfectly line
            // up.
            bool is_gap = (i > 0) && (prev_addr + prev_length < addr);

            // Before inserting the current entry, insert a terminal entry at the end of the previous entry's address
            // range if the current entry resulted in a gap from the previous entry.
            if (is_gap && ShouldAddLine(match_line, prev_line, prev_length))
            {
                line_table->AppendLineEntryToSequence(sequence.get(), prev_addr + prev_length, prev_line, 0,
                                                      prev_source_idx, false, false, false, false, true);
            }

            if (ShouldAddLine(match_line, lno, length))
            {
                bool is_statement = line->isStatement();
                bool is_prologue = false;
                bool is_epilogue = false;
                auto func = m_session_up->findSymbolByAddress(addr, PDB_SymType::Function);
                if (func)
                {
                    auto prologue = func->findOneChild<PDBSymbolFuncDebugStart>();
                    is_prologue = (addr == prologue->getVirtualAddress());

                    auto epilogue = func->findOneChild<PDBSymbolFuncDebugEnd>();
                    is_epilogue = (addr == epilogue->getVirtualAddress());
                }

                line_table->AppendLineEntryToSequence(sequence.get(), addr, lno, col, source_idx, is_statement, false,
                                                      is_prologue, is_epilogue, false);
            }

            prev_addr = addr;
            prev_length = length;
            prev_line = lno;
            prev_source_idx = source_idx;
        }

        if (entry_count > 0 && ShouldAddLine(match_line, prev_line, prev_length))
        {
            // The end is always a terminal entry, so insert it regardless.
            line_table->AppendLineEntryToSequence(sequence.get(), prev_addr + prev_length, prev_line, 0,
                                                  prev_source_idx, false, false, false, false, true);
        }

        line_table->InsertSequence(sequence.release());
    }

    sc.comp_unit->SetLineTable(line_table.release());
    return true;
}

void
SymbolFilePDB::BuildSupportFileIdToSupportFileIndexMap(const PDBSymbolCompiland &cu,
                                                       llvm::DenseMap<uint32_t, uint32_t> &index_map) const
{
    // This is a hack, but we need to convert the source id into an index into the support
    // files array.  We don't want to do path comparisons to avoid basename / full path
    // issues that may or may not even be a problem, so we use the globally unique source
    // file identifiers.  Ideally we could use the global identifiers everywhere, but LineEntry
    // currently assumes indices.
    auto source_files = m_session_up->getSourceFilesForCompiland(cu);
    int index = 0;

    while (auto file = source_files->getNext())
    {
        uint32_t source_id = file->getUniqueId();
        index_map[source_id] = index++;
    }
}
