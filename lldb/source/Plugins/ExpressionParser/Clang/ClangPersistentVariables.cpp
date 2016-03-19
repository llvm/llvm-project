//===-- ClangPersistentVariables.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/IRExecutionUnit.h"
#include "ClangPersistentVariables.h"

#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Value.h"

#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Symbol/SwiftASTContext.h" // Needed for llvm::isa<SwiftASTContext>(...)

#include "clang/AST/Decl.h"
#include "swift/AST/Decl.h"
#include "swift/AST/Pattern.h"

#include "llvm/ADT/StringMap.h"

using namespace lldb;
using namespace lldb_private;

ClangPersistentVariables::ClangPersistentVariables () :
    lldb_private::PersistentExpressionState(LLVMCastKind::eKindClang),
    m_next_persistent_variable_id (0),
    m_next_persistent_error_id (0)
{
}

ExpressionVariableSP
ClangPersistentVariables::CreatePersistentVariable (const lldb::ValueObjectSP &valobj_sp)
{
    return AddNewlyConstructedVariable(new ClangExpressionVariable(valobj_sp));
}

ExpressionVariableSP
ClangPersistentVariables::CreatePersistentVariable (ExecutionContextScope *exe_scope, 
                                                    const ConstString &name, 
                                                    const CompilerType& compiler_type,
                                                    lldb::ByteOrder byte_order, 
                                                    uint32_t addr_byte_size)
{
    return AddNewlyConstructedVariable(new ClangExpressionVariable(exe_scope, name, compiler_type, byte_order, addr_byte_size));
}

void
ClangPersistentVariables::RemovePersistentVariable (lldb::ExpressionVariableSP variable)
{
    if (!variable)
        return;

    RemoveVariable(variable);
    
    const char *name = variable->GetName().AsCString();
    
    if (*name != '$')
        return;
    name++;

    bool is_error = false;

    if (llvm::isa<SwiftASTContext>(variable->GetCompilerType().GetTypeSystem()))
    {
        switch (*name)
        {
        case 'R':
            break;
        case 'E':
            is_error = true;
            break;
        default:
            return;
        }
        name++;
    }

    uint32_t value = strtoul(name, NULL, 0);
    if (is_error)
    {
        if (value == m_next_persistent_error_id - 1)
            m_next_persistent_error_id--;
    }
    else
    {
        if (value == m_next_persistent_variable_id - 1)
            m_next_persistent_variable_id--;
    }
}

ConstString
ClangPersistentVariables::GetNextPersistentVariableName (bool is_error)
{
    char name_cstr[256];
    
    const char *prefix = "$";
    
/* THIS NEEDS TO BE HANDLED BY SWIFT-SPECIFIC CODE
    switch (language_type)
    {
    default:
        break;
    case lldb::eLanguageTypePLI:
    case lldb::eLanguageTypeSwift:
        if (is_error)
            prefix = "$E";
        else
            prefix = "$R";
        break;
    }
 */

    ::snprintf (name_cstr,
                sizeof(name_cstr),
                "%s%u",
                prefix,
                is_error? m_next_persistent_error_id++ : m_next_persistent_variable_id++);

    ConstString name(name_cstr);
    return name;
}

void
ClangPersistentVariables::RegisterPersistentDecl (const ConstString &name,
                                                  clang::NamedDecl *decl)
{
    m_persistent_decls.insert(std::pair<const char*, clang::NamedDecl*>(name.GetCString(), decl));
    
    if (clang::EnumDecl *enum_decl = llvm::dyn_cast<clang::EnumDecl>(decl))
    {
        for (clang::EnumConstantDecl *enumerator_decl : enum_decl->enumerators())
        {
            m_persistent_decls.insert(std::pair<const char*, clang::NamedDecl*>(ConstString(enumerator_decl->getNameAsString()).GetCString(), enumerator_decl));
        }
    }
}

clang::NamedDecl *
ClangPersistentVariables::GetPersistentDecl (const ConstString &name)
{
    PersistentDeclMap::const_iterator i = m_persistent_decls.find(name.GetCString());
    
    if (i == m_persistent_decls.end())
        return NULL;
    else
        return i->second;
}


void
ClangPersistentVariables::RegisterExecutionUnit (lldb::IRExecutionUnitSP &execution_unit_sp)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    m_execution_units.insert(execution_unit_sp);
    
    if (log)
        log->Printf ("Registering JITted Functions:\n");
    
    for (const IRExecutionUnit::JittedFunction &jitted_function : execution_unit_sp->GetJittedFunctions())
    {
        if (jitted_function.m_name != execution_unit_sp->GetFunctionName() &&
            jitted_function.m_remote_addr != LLDB_INVALID_ADDRESS)
        {
            m_symbol_map[jitted_function.m_name.GetCString()] = jitted_function.m_remote_addr;
            if (log)
                log->Printf ("  Function: %s at 0x%" PRIx64 ".", jitted_function.m_name.GetCString(), jitted_function.m_remote_addr);
        }
    }
    
    if (log)
        log->Printf ("Registering JIIted Symbols:\n");
    
    for (const IRExecutionUnit::JittedGlobalVariable &global_var : execution_unit_sp->GetJittedGlobalVariables())
    {
        if (global_var.m_remote_addr != LLDB_INVALID_ADDRESS)
        {
            // Demangle the name before inserting it, so that lookups by the ConstStr of the demangled name
            // will find the mangled one (needed for looking up metadata pointers.)
            Mangled mangler(global_var.m_name);
            mangler.GetDemangledName(lldb::eLanguageTypeUnknown);
            m_symbol_map[global_var.m_name.GetCString()] = global_var.m_remote_addr;
            if (log)
                log->Printf ("  Symbol: %s at 0x%" PRIx64 ".", global_var.m_name.GetCString(), global_var.m_remote_addr);
        }
    }
}

void
ClangPersistentVariables::RegisterSymbol (const ConstString &name, lldb::addr_t addr)
{
    m_symbol_map[name.GetCString()] = addr;
}

lldb::addr_t
ClangPersistentVariables::LookupSymbol (const ConstString &name)
{
    SymbolMap::iterator si = m_symbol_map.find(name.GetCString());
    
    if (si != m_symbol_map.end())
        return si->second;
    else
        return LLDB_INVALID_ADDRESS;    
}
