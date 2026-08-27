//===-- PdbAstBuilder.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDER_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDER_H

#include "lldb/Symbol/CompilerDecl.h"
#include "lldb/Symbol/CompilerDeclContext.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringRef.h"

#include "PdbSymUid.h"

namespace lldb_private {
class Stream;

namespace npdb {

class PdbAstBuilder {
public:
  virtual ~PdbAstBuilder() = default;

  virtual CompilerDecl GetOrCreateDeclForUid(PdbSymUid uid) = 0;
  virtual CompilerDeclContext GetOrCreateDeclContextForUid(PdbSymUid uid) = 0;
  virtual CompilerDeclContext GetParentDeclContext(PdbSymUid uid) = 0;

  virtual void EnsureFunction(PdbCompilandSymId func_id) = 0;
  virtual void EnsureInlinedFunction(PdbCompilandSymId inlinesite_id) = 0;
  virtual void EnsureBlock(PdbCompilandSymId block_id) = 0;
  virtual void EnsureVariable(PdbCompilandSymId scope_id,
                              PdbCompilandSymId var_id) = 0;
  virtual void EnsureVariable(PdbGlobalSymId var_id) = 0;

  virtual CompilerType GetOrCreateType(PdbTypeSymId type) = 0;
  virtual CompilerType GetOrCreateTypedefType(PdbGlobalSymId id) = 0;
  virtual bool CompleteType(CompilerType ct) = 0;

  virtual void ParseDeclsForContext(CompilerDeclContext context) = 0;

  virtual CompilerDeclContext FindNamespaceDecl(CompilerDeclContext parent_ctx,
                                                llvm::StringRef name) = 0;

  virtual void Dump(Stream &stream, llvm::StringRef filter,
                    bool show_color) = 0;
};

} // namespace npdb
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDER_H
