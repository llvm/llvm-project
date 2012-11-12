//===-- ClangPersistentVariables.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangPersistentVariables.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Value.h"

#include "llvm/ADT/StringMap.h"

using namespace lldb;
using namespace lldb_private;

ClangPersistentVariables::ClangPersistentVariables () :
    ClangExpressionVariableList(),
    m_next_persistent_variable_id (0)
{
}

ClangExpressionVariableSP
ClangPersistentVariables::CreatePersistentVariable (const lldb::ValueObjectSP &valobj_sp)
{
    ClangExpressionVariableSP var_sp (CreateVariable(valobj_sp));
    return var_sp;
}

ClangExpressionVariableSP
ClangPersistentVariables::CreatePersistentVariable (ExecutionContextScope *exe_scope, 
                                                    const ConstString &name, 
                                                    const TypeFromUser& user_type, 
                                                    lldb::ByteOrder byte_order, 
                                                    uint32_t addr_byte_size)
{
    ClangExpressionVariableSP var_sp (GetVariable(name));
    
    if (!var_sp)
        var_sp = CreateVariable(exe_scope, name, user_type, byte_order, addr_byte_size);

    return var_sp;
}

void
ClangPersistentVariables::RemovePersistentVariable (lldb::ClangExpressionVariableSP variable)
{
    RemoveVariable(variable);
    
    const char *name = variable->GetName().AsCString();
    
    if (*name != '$')
        return;
    name++;
    
    if (strtoul(name, NULL, 0) == m_next_persistent_variable_id - 1)
        m_next_persistent_variable_id--;
}

ConstString
ClangPersistentVariables::GetNextPersistentVariableName ()
{
    char name_cstr[256];
    ::snprintf (name_cstr, sizeof(name_cstr), "$%u", m_next_persistent_variable_id++);
    ConstString name(name_cstr);
    return name;
}

void
ClangPersistentVariables::RegisterPersistentType (const ConstString &name,
                                                  clang::TypeDecl *type_decl)
{
    m_persistent_types.insert(std::pair<const char*, clang::TypeDecl*>(name.GetCString(), type_decl));
}

clang::TypeDecl *
ClangPersistentVariables::GetPersistentType (const ConstString &name)
{
    PersistentTypeMap::const_iterator i = m_persistent_types.find(name.GetCString());
    
    if (i == m_persistent_types.end())
        return NULL;
    else
        return i->second;
}
