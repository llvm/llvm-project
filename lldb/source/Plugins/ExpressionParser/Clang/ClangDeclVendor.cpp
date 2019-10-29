//===-- ClangDeclVendor.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangDeclVendor.h"

#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Utility/ConstString.h"

using namespace lldb_private;

uint32_t ClangDeclVendor::FindDecls(ConstString name, bool append,
                                    uint32_t max_matches,
                                    std::vector<CompilerDecl> &decls) {
  if (!append)
    decls.clear();

  std::vector<clang::NamedDecl *> named_decls;
  uint32_t ret = FindDecls(name, /*append*/ false, max_matches, named_decls);
  for (auto *named_decl : named_decls) {
    decls.push_back(CompilerDecl(
        ClangASTContext::GetASTContext(&named_decl->getASTContext()),
        named_decl));
  }
  return ret;
}
