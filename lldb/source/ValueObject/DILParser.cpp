//===-- DILParser.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This implements the recursive descent parser for the Data Inspection
// Language (DIL), and its helper functions, which will eventually underlie the
// 'frame variable' command. The language that this parser recognizes is
// described in lldb/docs/dil-expr-lang.ebnf
//
//===----------------------------------------------------------------------===//

#include "lldb/ValueObject/DILParser.h"

#include <limits.h>
#include <stdlib.h>

#include <memory>
#include <sstream>
#include <string>

#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILEval.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatAdapters.h"

namespace lldb_private {

namespace dil {

inline void TokenKindsJoinImpl(std::ostringstream &os, dil::TokenKind k) {
  os << "'" << DILToken::getTokenName(k) << "'";
}

template <typename... Ts>
inline void TokenKindsJoinImpl(std::ostringstream &os, dil::TokenKind k,
                               Ts... ks) {
  TokenKindsJoinImpl(os, k);
  os << ", ";
  TokenKindsJoinImpl(os, ks...);
}

template <typename... Ts>
inline std::string TokenKindsJoin(dil::TokenKind k, Ts... ks) {
  std::ostringstream os;
  TokenKindsJoinImpl(os, k, ks...);

  return os.str();
}

std::string FormatDiagnostics(std::shared_ptr<std::string> input_expr,
                              const std::string &message, uint32_t loc) {
  // Get the source buffer and the location of the current token.
  llvm::StringRef text(*input_expr);
  size_t loc_offset = (size_t)loc;

  // Look for the start of the line.
  size_t line_start = text.rfind('\n', loc_offset);
  line_start = line_start == llvm::StringRef::npos ? 0 : line_start + 1;

  // Look for the end of the line.
  size_t line_end = text.find('\n', loc_offset);
  line_end = line_end == llvm::StringRef::npos ? text.size() : line_end;

  // Get a view of the current line in the source code and the position of the
  // diagnostics pointer.
  llvm::StringRef line = text.slice(line_start, line_end);
  int32_t arrow = loc + 1; // Column offset starts at 1, not 0.

  // Calculate the padding in case we point outside of the expression (this can
  // happen if the parser expected something, but got EOF).˚
  size_t expr_rpad = std::max(0, arrow - static_cast<int32_t>(line.size()));
  size_t arrow_rpad = std::max(0, static_cast<int32_t>(line.size()) - arrow);

  return llvm::formatv("<expr:1:{0}>: {1}\n{2}\n{3}", loc, message,
                       llvm::fmt_pad(line, 0, expr_rpad),
                       llvm::fmt_pad("^", arrow - 1, arrow_rpad));
}

DILParser::DILParser(std::shared_ptr<std::string> dil_input_expr,
                     std::shared_ptr<ExecutionContextScope> exe_ctx_scope,
                     lldb::DynamicValueType use_dynamic, bool use_synthetic,
                     bool fragile_ivar, bool check_ptr_vs_member)
    : m_ctx_scope(exe_ctx_scope), m_input_expr(dil_input_expr),
      m_use_dynamic(use_dynamic), m_use_synthetic(use_synthetic),
      m_fragile_ivar(fragile_ivar), m_check_ptr_vs_member(check_ptr_vs_member),
      m_dil_lexer(DILLexer(dil_input_expr)) {
  // Initialize the token.
  m_dil_token.setKind(dil::TokenKind::unknown);
}

DILASTNodeUP DILParser::Run(Status &error) {
  ConsumeToken();

  DILASTNodeUP expr;

  expr = ParseExpression();

  Expect(dil::TokenKind::eof);

  error = std::move(m_error);
  m_error.Clear();

  // Explicitly return ErrorNode if there was an error during the parsing.
  // Some routines raise an error, but don't change the return value (e.g.
  // Expect).
  if (error.Fail()) {
    CompilerType bad_type;
    return std::make_unique<ErrorNode>(bad_type);
  }
  return expr;
}

// Parse an expression.
//
//  expression:
//    primary_expression
//
DILASTNodeUP DILParser::ParseExpression() { return ParsePrimaryExpression(); }

// Parse a primary_expression.
//
//  primary_expression:
//    id_expression
//    "this"
//    "(" expression ")"
//
DILASTNodeUP DILParser::ParsePrimaryExpression() {
  CompilerType bad_type;
  if (m_dil_token.isOneOf(dil::TokenKind::coloncolon,
                          dil::TokenKind::identifier)) {
    // Save the source location for the diagnostics message.
    uint32_t loc = m_dil_token.getLocation();
    auto identifier = ParseIdExpression();

    return std::make_unique<IdentifierNode>(loc, identifier, m_use_dynamic,
                                            m_ctx_scope);
  } else if (m_dil_token.is(dil::TokenKind::kw_this)) {
    // Save the source location for the diagnostics message.
    uint32_t loc = m_dil_token.getLocation();
    ConsumeToken();

    // Special case for "this" pointer. As per C++ standard, it's a prvalue.
    return std::make_unique<IdentifierNode>(loc, "this", m_use_dynamic,
                                            m_ctx_scope);
  } else if (m_dil_token.is(dil::TokenKind::l_paren)) {
    ConsumeToken();
    auto expr = ParseExpression();
    Expect(dil::TokenKind::r_paren);
    ConsumeToken();
    return expr;
  }

  BailOut(ErrorCode::kInvalidExpressionSyntax,
          llvm::formatv("Unexpected token: {0}", TokenDescription(m_dil_token)),
          m_dil_token.getLocation());
  return std::make_unique<ErrorNode>(bad_type);
}

// Parse nested_name_specifier.
//
//  nested_name_specifier:
//    type_name "::"
//    namespace_name "::"
//    nested_name_specifier identifier "::"
//
std::string DILParser::ParseNestedNameSpecifier() {
  // The first token in nested_name_specifier is always an identifier, or
  // '(anonymous namespace)'.
  if (m_dil_token.isNot(dil::TokenKind::identifier) &&
      m_dil_token.isNot(dil::TokenKind::l_paren)) {
    return "";
  }

  // Anonymous namespaces need to be treated specially: They are represented
  // the the string '(anonymous namespace)', which has a space in it (throwing
  // off normal parsing) and is not actually proper C++> Check to see if we're
  // looking at '(anonymous namespace)::...'
  if (m_dil_token.is(dil::TokenKind::l_paren)) {
    // Look for all the pieces, in order:
    // l_paren 'anonymous' 'namespace' r_paren coloncolon
    if (m_dil_lexer.LookAhead(0).is(dil::TokenKind::identifier) &&
        (((DILToken)m_dil_lexer.LookAhead(0)).getSpelling() == "anonymous") &&
        m_dil_lexer.LookAhead(1).is(dil::TokenKind::kw_namespace) &&
        m_dil_lexer.LookAhead(2).is(dil::TokenKind::r_paren) &&
        m_dil_lexer.LookAhead(3).is(dil::TokenKind::coloncolon)) {
      m_dil_token = m_dil_lexer.AcceptLookAhead(3);

      assert((m_dil_token.is(dil::TokenKind::identifier) ||
              m_dil_token.is(dil::TokenKind::l_paren)) &&
             "Expected an identifier or anonymous namespace, but not found.");
      // Continue parsing the nested_namespace_specifier.
      std::string identifier2 = ParseNestedNameSpecifier();
      if (identifier2.empty()) {
        Expect(dil::TokenKind::identifier);
        identifier2 = m_dil_token.getSpelling();
        ConsumeToken();
      }
      return "(anonymous namespace)::" + identifier2;
    } else {
      return "";
    }
  } // end of special handling for '(anonymous namespace)'

  // If the next token is scope ("::"), then this is indeed a
  // nested_name_specifier
  if (m_dil_lexer.LookAhead(0).is(dil::TokenKind::coloncolon)) {
    // This nested_name_specifier is a single identifier.
    std::string identifier = m_dil_token.getSpelling();
    m_dil_token = m_dil_lexer.AcceptLookAhead(0);
    Expect(dil::TokenKind::coloncolon);
    ConsumeToken();
    // Continue parsing the nested_name_specifier.
    return identifier + "::" + ParseNestedNameSpecifier();
  }

  return "";
}

// Parse an id_expression.
//
//  id_expression:
//    unqualified_id
//    qualified_id
//
//  qualified_id:
//    ["::"] [nested_name_specifier] unqualified_id
//    ["::"] identifier
//
//  identifier:
//    ? dil::TokenKind::identifier ?
//
std::string DILParser::ParseIdExpression() {
  // Try parsing optional global scope operator.
  bool global_scope = false;
  if (m_dil_token.is(dil::TokenKind::coloncolon)) {
    global_scope = true;
    ConsumeToken();
  }

  // Try parsing optional nested_name_specifier.
  auto nested_name_specifier = ParseNestedNameSpecifier();

  // If nested_name_specifier is present, then it's qualified_id production.
  // Follow the first production rule.
  if (!nested_name_specifier.empty()) {
    // Parse unqualified_id and construct a fully qualified id expression.
    auto unqualified_id = ParseUnqualifiedId();

    return llvm::formatv("{0}{1}{2}", global_scope ? "::" : "",
                         nested_name_specifier, unqualified_id);
  }

  // No nested_name_specifier, but with global scope -- this is also a
  // qualified_id production. Follow the second production rule.
  else if (global_scope) {
    Expect(dil::TokenKind::identifier);
    std::string identifier = m_dil_token.getSpelling();
    ConsumeToken();
    return llvm::formatv("{0}{1}", global_scope ? "::" : "", identifier);
  }

  // This is unqualified_id production.
  return ParseUnqualifiedId();
}

// Parse an unqualified_id.
//
//  unqualified_id:
//    identifier
//
//  identifier:
//    ? dil::TokenKind::identifier ?
//
std::string DILParser::ParseUnqualifiedId() {
  Expect(dil::TokenKind::identifier);
  std::string identifier = m_dil_token.getSpelling();
  ConsumeToken();
  return identifier;
}

void DILParser::BailOut(ErrorCode code, const std::string &error,
                        uint32_t loc) {
  if (m_error.Fail()) {
    // If error is already set, then the parser is in the "bail-out" mode. Don't
    // do anything and keep the original error.
    return;
  }

  m_error = Status((uint32_t)code, lldb::eErrorTypeGeneric,
                   FormatDiagnostics(m_input_expr, error, loc));
  m_dil_token.setKind(dil::TokenKind::eof);
}

void DILParser::BailOut(Status error) {
  if (m_error.Fail()) {
    // If error is already set, then the parser is in the "bail-out" mode. Don't
    // do anything and keep the original error.
    return;
  }
  m_error = std::move(error);
  m_dil_token.setKind(dil::TokenKind::eof);
}

void DILParser::ConsumeToken() {
  if (m_dil_token.is(dil::TokenKind::eof)) {
    // Don't do anything if we're already at eof. This can happen if an error
    // occurred during parsing and we're trying to bail out.
    return;
  }
  bool all_ok;
  m_dil_lexer.Lex(m_dil_token);
  if (m_dil_lexer.GetCurrentTokenIdx() == UINT_MAX)
    all_ok = m_dil_lexer.ResetTokenIdx(0);
  else
    all_ok = m_dil_lexer.IncrementTokenIdx();
  if (!all_ok)
    BailOut(ErrorCode::kUnknown, "Invalid lexer token index", 0);
}

void DILParser::Expect(dil::TokenKind kind) {
  if (m_dil_token.isNot(kind)) {
    BailOut(ErrorCode::kUnknown,
            llvm::formatv("expected {0}, got: {1}", TokenKindsJoin(kind),
                          TokenDescription(m_dil_token)),
            m_dil_token.getLocation());
  }
}

template <typename... Ts>
void DILParser::ExpectOneOf(dil::TokenKind k, Ts... ks) {
  static_assert((std::is_same_v<Ts, dil::TokenKind> && ...),
                "ExpectOneOf can be only called with values of type "
                "dil::TokenKind");

  if (!m_dil_token.isOneOf(k, ks...)) {
    BailOut(ErrorCode::kUnknown,
            llvm::formatv("expected any of ({0}), got: {1}",
                          TokenKindsJoin(k, ks...),
                          TokenDescription(m_dil_token)),
            m_dil_token.getLocation());
  }
}

std::string DILParser::TokenDescription(const DILToken &token) {
  const auto &spelling = ((DILToken)token).getSpelling();
  const std::string kind_name =
      DILToken::getTokenName(((DILToken)token).getKind());
  return llvm::formatv("<'{0}' ({1})>", spelling, kind_name);
}

} // namespace dil

} // namespace lldb_private
