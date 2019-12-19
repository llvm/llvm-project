//===-- ClangExternalASTSourceCommon.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Utility/Stream.h"

#include <mutex>

using namespace lldb_private;

char ClangExternalASTSourceCommon::ID;

ClangExternalASTSourceCommon::~ClangExternalASTSourceCommon() {}

ClangASTMetadata *
ClangExternalASTSourceCommon::GetMetadata(const clang::Decl *object) {
  auto It = m_decl_metadata.find(object);
  if (It != m_decl_metadata.end())
    return &It->second;
  return nullptr;
}

ClangASTMetadata *
ClangExternalASTSourceCommon::GetMetadata(const clang::Type *object) {
  auto It = m_type_metadata.find(object);
  if (It != m_type_metadata.end())
    return &It->second;
  return nullptr;
}
