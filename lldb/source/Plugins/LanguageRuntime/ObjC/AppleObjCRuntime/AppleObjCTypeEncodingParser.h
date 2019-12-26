//===-- AppleObjCTypeEncodingParser.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCTypeEncodingParser_h_
#define liblldb_AppleObjCTypeEncodingParser_h_

#include "clang/AST/ASTContext.h"

#include "lldb/lldb-private.h"

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"

namespace lldb_private {
class StringLexer;
class AppleObjCTypeEncodingParser : public ObjCLanguageRuntime::EncodingToType {
public:
  AppleObjCTypeEncodingParser(ObjCLanguageRuntime &runtime);
  ~AppleObjCTypeEncodingParser() override = default;

  CompilerType RealizeType(ClangASTContext &ast_ctx, const char *name,
                           bool for_expression) override;

private:
  struct StructElement {
    std::string name;
    clang::QualType type;
    uint32_t bitfield;

    StructElement();
    ~StructElement() = default;
  };

  clang::QualType BuildType(ClangASTContext &clang_ast_ctx, StringLexer &type,
                            bool for_expression,
                            uint32_t *bitfield_bit_size = nullptr);

  clang::QualType BuildStruct(ClangASTContext &ast_ctx, StringLexer &type,
                              bool for_expression);

  clang::QualType BuildAggregate(ClangASTContext &clang_ast_ctx,
                                 StringLexer &type, bool for_expression,
                                 char opener, char closer, uint32_t kind);

  clang::QualType BuildUnion(ClangASTContext &ast_ctx, StringLexer &type,
                             bool for_expression);

  clang::QualType BuildArray(ClangASTContext &ast_ctx, StringLexer &type,
                             bool for_expression);

  std::string ReadStructName(StringLexer &type);

  StructElement ReadStructElement(ClangASTContext &ast_ctx, StringLexer &type,
                                  bool for_expression);

  clang::QualType BuildObjCObjectPointerType(ClangASTContext &clang_ast_ctx,
                                             StringLexer &type,
                                             bool for_expression);

  uint32_t ReadNumber(StringLexer &type);

  std::string ReadQuotedString(StringLexer &type);

  ObjCLanguageRuntime &m_runtime;
};

} // namespace lldb_private

#endif // liblldb_AppleObjCTypeEncodingParser_h_
