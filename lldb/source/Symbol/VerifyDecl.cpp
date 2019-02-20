//===-- VerifyDecl.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/VerifyDecl.h"
#include "clang/AST/DeclBase.h"

void lldb_private::VerifyDecl(clang::Decl *decl) { decl->getAccess(); }
