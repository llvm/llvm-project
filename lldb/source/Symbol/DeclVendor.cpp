//===-- DeclVendor.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/DeclVendor.h"

#include "lldb/Symbol/ClangASTContext.h"

#include <vector>

using namespace lldb;
using namespace lldb_private;

std::vector<CompilerType> DeclVendor::FindTypes(ConstString name,
                                                uint32_t max_matches) {
  // FIXME: This depends on clang, but should be able to support any
  // TypeSystem.
  std::vector<CompilerType> ret;
  std::vector<clang::NamedDecl *> decls;
  if (FindDecls(name, /*append*/ true, max_matches, decls))
    for (auto *decl : decls)
      if (auto type = ClangASTContext::GetTypeForDecl(decl))
        ret.push_back(type);
  return ret;
}
