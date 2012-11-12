//===-- TypeList.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"

#include "clang/Basic/Builtins.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

// Project includes
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"

using namespace lldb;
using namespace lldb_private;
using namespace clang;

TypeList::TypeList() :
    m_types ()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
TypeList::~TypeList()
{
}

void
TypeList::Insert (const TypeSP& type_sp)
{
    // Just push each type on the back for now. We will worry about uniquing later
    if (type_sp)
        m_types.insert(std::make_pair(type_sp->GetID(), type_sp));
}


bool
TypeList::InsertUnique (const TypeSP& type_sp)
{
    if (type_sp)
    {
        user_id_t type_uid = type_sp->GetID();
        iterator pos, end = m_types.end();
        
        for (pos = m_types.find(type_uid); pos != end && pos->second->GetID() == type_uid; ++pos)
        {
            if (pos->second.get() == type_sp.get())
                return false;
        }
    }
    Insert (type_sp);
    return true;
}

//----------------------------------------------------------------------
// Find a base type by its unique ID.
//----------------------------------------------------------------------
//TypeSP
//TypeList::FindType(lldb::user_id_t uid)
//{
//    iterator pos = m_types.find(uid);
//    if (pos != m_types.end())
//        return pos->second;
//    return TypeSP();
//}

//----------------------------------------------------------------------
// Find a type by name.
//----------------------------------------------------------------------
//TypeList
//TypeList::FindTypes (const ConstString &name)
//{
//    // Do we ever need to make a lookup by name map? Here we are doing
//    // a linear search which isn't going to be fast.
//    TypeList types(m_ast.getTargetInfo()->getTriple().getTriple().c_str());
//    iterator pos, end;
//    for (pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
//        if (pos->second->GetName() == name)
//            types.Insert (pos->second);
//    return types;
//}

void
TypeList::Clear()
{
    m_types.clear();
}

uint32_t
TypeList::GetSize() const
{
    return m_types.size();
}

// GetTypeAtIndex isn't used a lot for large type lists, currently only for
// type lists that are returned for "image dump -t TYPENAME" commands and other
// simple symbol queries that grab the first result...

TypeSP
TypeList::GetTypeAtIndex(uint32_t idx)
{
    iterator pos, end;
    uint32_t i = idx;
    for (pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
    {
        if (i == 0)
            return pos->second;
        --i;
    }
    return TypeSP();
}

bool
TypeList::RemoveTypeWithUID (user_id_t uid)
{
    iterator pos = m_types.find(uid);
    
    if (pos != m_types.end())
    {
        m_types.erase(pos);
        return true;
    }
    return false;
}


void
TypeList::Dump(Stream *s, bool show_context)
{
    for (iterator pos = m_types.begin(), end = m_types.end(); pos != end; ++pos)
    {
        pos->second->Dump(s, show_context);
    }
}

// depending on implementation details, type lookup might fail because of
// embedded spurious namespace:: prefixes. this call strips them, paying
// attention to the fact that a type might have namespace'd type names as
// arguments to templates, and those must not be stripped off
static bool
GetTypeScopeAndBasename(const char* name_cstr, std::string &scope, std::string &basename, bool *exact_ptr)
{
    // Protect against null c string.
    
    if (name_cstr && name_cstr[0])
    {
        const char *basename_cstr = name_cstr;
        const char* namespace_separator = ::strstr (basename_cstr, "::");
        if (namespace_separator)
        {
            const char* template_arg_char = ::strchr (basename_cstr, '<');
            while (namespace_separator != NULL)
            {
                if (template_arg_char && namespace_separator > template_arg_char) // but namespace'd template arguments are still good to go
                    break;
                basename_cstr = namespace_separator + 2;
                namespace_separator = strstr(basename_cstr, "::");
            }
            if (basename_cstr > name_cstr)
            {
                scope.assign (name_cstr, basename_cstr - name_cstr);
                if (scope.size() >= 2 && scope[0] == ':' && scope[1] == ':')
                {
                    // The typename passed in started with "::" so make sure we only do exact matches
                    if (exact_ptr)
                        *exact_ptr = true;
                    // Strip the leading "::" as this won't ever show in qualified typenames we get
                    // from clang.
                    scope.erase(0,2);
                }
                basename.assign (basename_cstr);
                return true;
            }
        }
    }
    return false;
}

void
TypeList::RemoveMismatchedTypes (const char *qualified_typename,
                                 bool exact_match)
{
    std::string type_scope;
    std::string type_basename;
    TypeClass type_class = eTypeClassAny;
    if (!Type::GetTypeScopeAndBasename (qualified_typename, type_scope, type_basename, type_class))
    {
        type_basename = qualified_typename;
        type_scope.clear();
    }
    return RemoveMismatchedTypes (type_scope, type_basename, type_class, exact_match);
}

void
TypeList::RemoveMismatchedTypes (const std::string &type_scope,
                                 const std::string &type_basename,
                                 TypeClass type_class,
                                 bool exact_match)
{
    // Our "collection" type currently is a std::map which doesn't
    // have any good way to iterate and remove items from the map
    // so we currently just make a new list and add all of the matching
    // types to it, and then swap it into m_types at the end
    collection matching_types;

    iterator pos, end = m_types.end();
    
    for (pos = m_types.begin(); pos != end; ++pos)
    {
        Type* the_type = pos->second.get();
        bool keep_match = false;
        TypeClass match_type_class = eTypeClassAny;

        if (type_class != eTypeClassAny)
        {
            match_type_class = ClangASTType::GetTypeClass (the_type->GetClangAST(),
                                                           the_type->GetClangForwardType());
            if ((match_type_class & type_class) == 0)
                continue;
        }

        ConstString match_type_name_const_str (the_type->GetQualifiedName());
        if (match_type_name_const_str)
        {
            const char *match_type_name = match_type_name_const_str.GetCString();
            std::string match_type_scope;
            std::string match_type_basename;
            if (Type::GetTypeScopeAndBasename (match_type_name,
                                               match_type_scope,
                                               match_type_basename,
                                               match_type_class))
            {
                if (match_type_basename == type_basename)
                {
                    const size_t type_scope_size = type_scope.size();
                    const size_t match_type_scope_size = match_type_scope.size();
                    if (exact_match || (type_scope_size == match_type_scope_size))
                    {
                        keep_match = match_type_scope == type_scope;
                    }
                    else
                    {
                        if (match_type_scope_size > type_scope_size)
                        {
                            const size_t type_scope_pos = match_type_scope.rfind(type_scope);
                            if (type_scope_pos == match_type_scope_size - type_scope_size)
                            {
                                if (type_scope_pos >= 2)
                                {
                                    // Our match scope ends with the type scope we were lookikng for,
                                    // but we need to make sure what comes before the matching
                                    // type scope is a namepace boundary in case we are trying to match:
                                    // type_basename = "d"
                                    // type_scope = "b::c::"
                                    // We want to match:
                                    //  match_type_scope "a::b::c::"
                                    // But not:
                                    //  match_type_scope "a::bb::c::"
                                    // So below we make sure what comes before "b::c::" in match_type_scope
                                    // is "::", or the namespace boundary
                                    if (match_type_scope[type_scope_pos - 1] == ':' &&
                                        match_type_scope[type_scope_pos - 2] == ':')
                                    {
                                        keep_match = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // The type we are currently looking at doesn't exists
                // in a namespace or class, so it only matches if there
                // is no type scope...
                keep_match = type_scope.empty() && type_basename.compare(match_type_name) == 0;
            }
        }
        
        if (keep_match)
        {
            matching_types.insert (*pos);
        }
    }
    m_types.swap(matching_types);
}

void
TypeList::RemoveMismatchedTypes (TypeClass type_class)
{
    if (type_class == eTypeClassAny)
        return;

    // Our "collection" type currently is a std::map which doesn't
    // have any good way to iterate and remove items from the map
    // so we currently just make a new list and add all of the matching
    // types to it, and then swap it into m_types at the end
    collection matching_types;
    
    iterator pos, end = m_types.end();
    
    for (pos = m_types.begin(); pos != end; ++pos)
    {
        Type* the_type = pos->second.get();
        TypeClass match_type_class = ClangASTType::GetTypeClass (the_type->GetClangAST(),
                                                                 the_type->GetClangForwardType());
        if (match_type_class & type_class)
            matching_types.insert (*pos);
    }
    m_types.swap(matching_types);
}

//void *
//TypeList::CreateClangPointerType (Type *type)
//{
//    assert(type);
//    return m_ast.CreatePointerType(type->GetClangForwardType());
//}
//
//void *
//TypeList::CreateClangTypedefType (Type *typedef_type, Type *base_type)
//{
//    assert(typedef_type && base_type);
//    return m_ast.CreateTypedefType (typedef_type->GetName().AsCString(), 
//                                    base_type->GetClangForwardType(), 
//                                    typedef_type->GetSymbolFile()->GetClangDeclContextForTypeUID(typedef_type->GetID()));
//}
//
//void *
//TypeList::CreateClangLValueReferenceType (Type *type)
//{
//    assert(type);
//    return m_ast.CreateLValueReferenceType(type->GetClangForwardType());
//}
//
//void *
//TypeList::CreateClangRValueReferenceType (Type *type)
//{
//    assert(type);
//    return m_ast.CreateRValueReferenceType (type->GetClangForwardType());
//}
//


