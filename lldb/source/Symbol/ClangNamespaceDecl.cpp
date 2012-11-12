//===-- ClangNamespaceDecl.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangNamespaceDecl.h"

#include "clang/AST/Decl.h"

namespace lldb_private {
    
std::string
ClangNamespaceDecl::GetQualifiedName () const
{
    if (m_namespace_decl)
        return m_namespace_decl->getQualifiedNameAsString();
    return std::string();
}


}
