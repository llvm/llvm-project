//===-- CompilerDecl.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/CompilerDecl.h"
#include "lldb/Symbol/CompilerDeclContext.h"
#include "lldb/Symbol/TypeSystem.h"

using namespace lldb_private;

bool
CompilerDecl::IsClang () const
{
    return IsValid() && m_type_system->getKind() == TypeSystem::eKindClang;
}

ConstString
CompilerDecl::GetName() const
{
    return m_type_system->DeclGetName(m_opaque_decl);
}

lldb::VariableSP
CompilerDecl::GetAsVariable ()
{
    return m_type_system->DeclGetVariable(m_opaque_decl);
}

bool
lldb_private::operator == (const lldb_private::CompilerDecl &lhs, const lldb_private::CompilerDecl &rhs)
{
    return lhs.GetTypeSystem() == rhs.GetTypeSystem() && lhs.GetOpaqueDecl() == rhs.GetOpaqueDecl();
}


bool
lldb_private::operator != (const lldb_private::CompilerDecl &lhs, const lldb_private::CompilerDecl &rhs)
{
    return lhs.GetTypeSystem() != rhs.GetTypeSystem() || lhs.GetOpaqueDecl() != rhs.GetOpaqueDecl();
}

