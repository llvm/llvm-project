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

#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/lldb-private.h"

namespace lldb_utility {
class StringLexer;
}

namespace lldb_private {

class AppleObjCTypeEncodingParser : public ObjCLanguageRuntime::EncodingToType {
public:
  AppleObjCTypeEncodingParser(ObjCLanguageRuntime &runtime);
  ~AppleObjCTypeEncodingParser() override = default;

  CompilerType RealizeType(clang::ASTContext &ast_ctx, const char *name,
                           bool for_expression) override;

private:
  struct StructElement {
    std::string name;
    clang::QualType type;
    uint32_t bitfield;

    StructElement();
    ~StructElement() = default;
  };

  clang::QualType BuildType(clang::ASTContext &ast_ctx,
                            lldb_utility::StringLexer &type,
                            bool for_expression,
                            uint32_t *bitfield_bit_size = nullptr);

  clang::QualType BuildStruct(clang::ASTContext &ast_ctx,
                              lldb_utility::StringLexer &type,
                              bool for_expression);

  clang::QualType BuildAggregate(clang::ASTContext &ast_ctx,
                                 lldb_utility::StringLexer &type,
                                 bool for_expression, char opener, char closer,
                                 uint32_t kind);

  clang::QualType BuildUnion(clang::ASTContext &ast_ctx,
                             lldb_utility::StringLexer &type,
                             bool for_expression);

  clang::QualType BuildArray(clang::ASTContext &ast_ctx,
                             lldb_utility::StringLexer &type,
                             bool for_expression);

  std::string ReadStructName(lldb_utility::StringLexer &type);

  StructElement ReadStructElement(clang::ASTContext &ast_ctx,
                                  lldb_utility::StringLexer &type,
                                  bool for_expression);

  clang::QualType BuildObjCObjectPointerType(clang::ASTContext &ast_ctx,
                                             lldb_utility::StringLexer &type,
                                             bool for_expression);

  uint32_t ReadNumber(lldb_utility::StringLexer &type);

  std::string ReadQuotedString(lldb_utility::StringLexer &type);

  ObjCLanguageRuntime &m_runtime;
};

} // namespace lldb_private

#endif // liblldb_AppleObjCTypeEncodingParser_h_
