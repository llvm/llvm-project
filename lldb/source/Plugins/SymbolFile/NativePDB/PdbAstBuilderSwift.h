//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDERSWIFT_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDERSWIFT_H

#include "PdbAstBuilder.h"

#include "llvm/ADT/DenseMap.h"

namespace llvm::pdb {
class TpiStream;
}

namespace lldb_private {

class TypeSystemSwiftTypeRef;

namespace npdb {

class PdbAstBuilderSwift : public PdbAstBuilder {
public:
  PdbAstBuilderSwift(TypeSystemSwiftTypeRef &swift_ts);

  CompilerDecl GetOrCreateDeclForUid(PdbSymUid uid) override { return {}; }
  CompilerDeclContext GetOrCreateDeclContextForUid(PdbSymUid uid) override {
    return {};
  }
  CompilerDeclContext GetParentDeclContext(PdbSymUid uid) override {
    return {};
  }

  void EnsureFunction(PdbCompilandSymId func_id) override {}
  void EnsureInlinedFunction(PdbCompilandSymId inlinesite_id) override {}
  void EnsureBlock(PdbCompilandSymId block_id) override {}
  void EnsureVariable(PdbCompilandSymId scope_id,
                      PdbCompilandSymId var_id) override {}
  void EnsureVariable(PdbGlobalSymId var_id) override {}

  CompilerType GetOrCreateType(PdbTypeSymId type) override;
  CompilerType GetOrCreateTypedefType(PdbGlobalSymId id) override {
    return {};
  }
  bool CompleteType(CompilerType ct) override { return true; }

  void ParseDeclsForContext(CompilerDeclContext context) override {}

  void Dump(Stream &stream, llvm::StringRef filter) override;

private:
  CompilerType CreateType(PdbTypeSymId type, llvm::pdb::TpiStream &tpi);

  TypeSystemSwiftTypeRef &m_swift_ts;
  llvm::DenseMap<lldb::user_id_t, CompilerType> m_uid_to_type;
};

} // namespace npdb
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDERSWIFT_H
