//===-- ImporterBackedASTSource.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLANG_IMPORTERBACKEDASTSOURCE
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLANG_IMPORTERBACKEDASTSOURCE

#include "clang/AST/ASTContext.h"
#include "clang/Sema/ExternalSemaSource.h"

namespace lldb_private {

/// The base class of all ExternalASTSources in LLDB that use the
/// ClangASTImporter to move declarations from other ASTs to the ASTContext they
/// are attached to.
class ImporterBackedASTSource : public clang::ExternalSemaSource {
  /// LLVM RTTI support.
  static char ID;

public:
  /// LLVM RTTI support.
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || ExternalSemaSource::isA(ClassID);
  }
  static bool classof(const clang::ExternalASTSource *s) { return s->isA(&ID); }

  /// This marks all redeclaration chains in the ASTContext as out-of-date and
  /// that this ExternalASTSource should be consulted to get the complete
  /// redeclaration chain.
  ///
  /// \see ExternalASTSource::CompleteRedeclChain
  void MarkRedeclChainsAsOutOfDate(clang::ASTContext &c) {
    // This invalidates redeclaration chains but also other things such as
    // identifiers. There isn't a more precise way at the moment that only
    // affects redecl chains.
    incrementGeneration(c);
  }
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLANG_IMPORTERBACKEDASTSOURCE
