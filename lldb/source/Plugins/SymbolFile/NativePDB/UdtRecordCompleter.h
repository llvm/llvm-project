//===-- UdtRecordCompleter.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILE_NATIVEPDB_UDTRECORDCOMPLETER_H
#define LLDB_PLUGINS_SYMBOLFILE_NATIVEPDB_UDTRECORDCOMPLETER_H

#include "lldb/Symbol/ClangASTImporter.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"

#include "PdbSymUid.h"

namespace clang {
class CXXBaseSpecifier;
class QualType;
class TagDecl;
} // namespace clang

namespace llvm {
namespace pdb {
class TpiStream;
}
} // namespace llvm

namespace lldb_private {
class Type;
class CompilerType;
namespace npdb {
class PdbAstBuilder;

class UdtRecordCompleter : public llvm::codeview::TypeVisitorCallbacks {
  using IndexedBase =
      std::pair<uint64_t, std::unique_ptr<clang::CXXBaseSpecifier>>;

  union UdtTagRecord {
    UdtTagRecord() {}
    llvm::codeview::UnionRecord ur;
    llvm::codeview::ClassRecord cr;
    llvm::codeview::EnumRecord er;
  } m_cvr;

  PdbTypeSymId m_id;
  CompilerType &m_derived_ct;
  clang::TagDecl &m_tag_decl;
  PdbAstBuilder &m_ast_builder;
  llvm::pdb::TpiStream &m_tpi;
  std::vector<IndexedBase> m_bases;
  ClangASTImporter::LayoutInfo m_layout;

public:
  UdtRecordCompleter(PdbTypeSymId id, CompilerType &derived_ct,
                     clang::TagDecl &tag_decl, PdbAstBuilder &ast_builder,
                     llvm::pdb::TpiStream &tpi);

#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  llvm::Error visitKnownMember(llvm::codeview::CVMemberRecord &CVR,            \
                               llvm::codeview::Name##Record &Record) override;
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"

  void complete();

private:
  clang::QualType AddBaseClassForTypeIndex(
      llvm::codeview::TypeIndex ti, llvm::codeview::MemberAccess access,
      llvm::Optional<uint64_t> vtable_idx = llvm::Optional<uint64_t>());
  void AddMethod(llvm::StringRef name, llvm::codeview::TypeIndex type_idx,
                 llvm::codeview::MemberAccess access,
                 llvm::codeview::MethodOptions options,
                 llvm::codeview::MemberAttributes attrs);
};

} // namespace npdb
} // namespace lldb_private

#endif // LLDB_PLUGINS_SYMBOLFILE_NATIVEPDB_UDTRECORDCOMPLETER_H
