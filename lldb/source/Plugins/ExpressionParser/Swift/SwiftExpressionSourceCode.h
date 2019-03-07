//===-- SwiftExpressionSourceCode.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftExpressionSourceCode_h
#define liblldb_SwiftExpressionSourceCode_h

#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/lldb-enumerations.h"

namespace lldb_private {

class SwiftExpressionSourceCode : public ExpressionSourceCode {
public:

  static SwiftExpressionSourceCode *CreateWrapped(const char *prefix,
                                             const char *body) {
    return new SwiftExpressionSourceCode("$__lldb_expr", prefix, body, true);
  }

  static SwiftExpressionSourceCode *CreateUnwrapped(const char *name,
                                               const char *body) {
    return new SwiftExpressionSourceCode(name, "", body, false);
  }

  // Given a string returned by GetText, find the beginning and end of the body
  // passed to CreateWrapped. Return true if the bounds could be found.  This
  // will also work on text with FixItHints applied.
  static bool GetOriginalBodyBounds(std::string transformed_text,
                                    size_t &start_loc, size_t &end_loc);

  uint32_t GetNumBodyLines();

  bool GetText(std::string &text, 
               lldb::LanguageType wrapping_language,
               bool needs_object_ptr,
               bool static_method,
               bool is_class,
               bool weak_self,
               const EvaluateExpressionOptions &options,
               ExecutionContext &exe_ctx,
               const Expression::SwiftGenericInfo &generic_info,
               uint32_t &first_body_line) const;

private:
  SwiftExpressionSourceCode(const char *name, const char *prefix, const char *body,
                       bool wrap)
      : ExpressionSourceCode(name, prefix, body, wrap) {}
  uint32_t m_num_body_lines = 0;
};

} // namespace lldb_private

#endif
