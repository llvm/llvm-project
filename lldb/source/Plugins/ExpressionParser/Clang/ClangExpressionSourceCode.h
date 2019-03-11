//===-- ClangExpressionSourceCode.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangExpressionSourceCode_h
#define liblldb_ClangExpressionSourceCode_h

#include "lldb/Expression/Expression.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace lldb_private {

class ExecutionContext;

class ClangExpressionSourceCode : public ExpressionSourceCode {
public:
  static const char *g_expression_prefix;

  static ClangExpressionSourceCode *CreateWrapped(const char *prefix,
                                             const char *body) {
    return new ClangExpressionSourceCode("$__lldb_expr", prefix, body, true);
  }

  uint32_t GetNumBodyLines();

  bool GetText(std::string &text, lldb::LanguageType wrapping_language,
               bool static_method,
               ExecutionContext &exe_ctx,
               bool add_locals) const;

  // Given a string returned by GetText, find the beginning and end of the body
  // passed to CreateWrapped. Return true if the bounds could be found.  This
  // will also work on text with FixItHints applied.
  static bool GetOriginalBodyBounds(std::string transformed_text,
                                    lldb::LanguageType wrapping_language,
                                    size_t &start_loc, size_t &end_loc);

protected:
  ClangExpressionSourceCode(const char *name, const char *prefix, const char *body,
                       bool wrap) :
      ExpressionSourceCode(name, prefix, body, wrap) {}
};

} // namespace lldb_private

#endif
