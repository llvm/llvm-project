//===-- SwiftExpressionSourceCode.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftExpressionSourceCode_h
#define liblldb_SwiftExpressionSourceCode_h

#include "SwiftASTManipulator.h"
#include "lldb/Expression/Expression.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/lldb-enumerations.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"

namespace lldb_private {

/// Parse a name such as "$τ_0_0".
std::optional<std::pair<unsigned, unsigned>>
ParseSwiftGenericParameter(llvm::StringRef name);

class SwiftExpressionSourceCode : public ExpressionSourceCode {
public:

  static SwiftExpressionSourceCode *CreateWrapped(const char *prefix,
                                             const char *body) {
    return new SwiftExpressionSourceCode("$__lldb_expr", prefix, body, Wrap);
  }

  static SwiftExpressionSourceCode *CreateUnwrapped(const char *name,
                                               const char *body) {
    return new SwiftExpressionSourceCode(name, "", body, NoWrap);
  }

  // Given a string returned by GetText, find the beginning and end of the body
  // passed to CreateWrapped. Return true if the bounds could be found.  This
  // will also work on text with FixItHints applied.
  static bool GetOriginalBodyBounds(std::string transformed_text,
                                    size_t &start_loc, size_t &end_loc);

  uint32_t GetNumBodyLines();

  Status GetText(
      std::string &text, lldb::LanguageType wrapping_language,
      bool needs_object_ptr, bool static_method, bool is_class, bool weak_self,
      const EvaluateExpressionOptions &options,
      const std::optional<SwiftLanguageRuntime::GenericSignature> &generic_sig,
      ExecutionContext &exe_ctx, uint32_t &first_body_line,
      llvm::ArrayRef<SwiftASTManipulator::VariableInfo> local_variables) const;

private:
  SwiftExpressionSourceCode(const char *name, const char *prefix, const char *body,
                       Wrapping wrap)
      : ExpressionSourceCode(name, prefix, body, wrap) {}
  uint32_t m_num_body_lines = 0;
};

} // namespace lldb_private

#endif
