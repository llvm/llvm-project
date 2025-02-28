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
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILEval.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatAdapters.h"
#include <limits.h>
#include <memory>
#include <sstream>
#include <stdlib.h>
#include <string>

namespace lldb_private::dil {

std::string FormatDiagnostics(llvm::StringRef text, const std::string &message,
                              uint32_t loc) {
  // Get the position, in the current line of text, of the diagnostics pointer.
  // ('loc' is the location of the start of the current token/error within the
  // overal text line).
  int32_t arrow = loc + 1; // Column offset starts at 1, not 0.

  return llvm::formatv("<expr:1:{0}>: {1}\n{2}\n{3}", loc, message,
                       llvm::fmt_pad(text, 0, 0),
                       llvm::fmt_pad("^", arrow - 1, 0));
}

llvm::Expected<ASTNodeUP>
DILParser::Parse(llvm::StringRef dil_input_expr, DILLexer lexer,
                 std::shared_ptr<StackFrame> frame_sp,
                 lldb::DynamicValueType use_dynamic, bool use_synthetic,
                 bool fragile_ivar, bool check_ptr_vs_member) {
  Status error;
  DILParser parser(dil_input_expr, lexer, frame_sp, use_dynamic, use_synthetic,
                   fragile_ivar, check_ptr_vs_member, error);
  return parser.Run();
}

DILParser::DILParser(llvm::StringRef dil_input_expr, DILLexer lexer,
                     std::shared_ptr<StackFrame> frame_sp,
                     lldb::DynamicValueType use_dynamic, bool use_synthetic,
                     bool fragile_ivar, bool check_ptr_vs_member, Status &error)
    : m_ctx_scope(frame_sp), m_input_expr(dil_input_expr), m_dil_lexer(lexer),
      m_error(error), m_use_dynamic(use_dynamic),
      m_use_synthetic(use_synthetic), m_fragile_ivar(fragile_ivar),
      m_check_ptr_vs_member(check_ptr_vs_member) {}

llvm::Expected<ASTNodeUP> DILParser::Run() {
  ASTNodeUP expr;

  expr = ParseExpression();

  Expect(Token::Kind::eof);

  if (m_error.Fail())
    return m_error.ToError();

  return expr;
}

// Parse an expression.
//
//  expression:
//    primary_expression
//
ASTNodeUP DILParser::ParseExpression() { return ParsePrimaryExpression(); }

// Parse a primary_expression.
//
//  primary_expression:
//    id_expression
//    "(" expression ")"
//
ASTNodeUP DILParser::ParsePrimaryExpression() {
  if (CurToken().IsOneOf(Token::coloncolon, Token::identifier)) {
    // Save the source location for the diagnostics message.
    uint32_t loc = CurToken().GetLocation();
    auto identifier = ParseIdExpression();

    return std::make_unique<IdentifierNode>(loc, identifier, m_use_dynamic,
                                            m_ctx_scope);
  } else if (CurToken().Is(Token::l_paren)) {
    m_dil_lexer.Advance();
    auto expr = ParseExpression();
    Expect(Token::r_paren);
    m_dil_lexer.Advance();
    return expr;
  }

  BailOut(ErrorCode::kInvalidExpressionSyntax,
          llvm::formatv("Unexpected token: {0}", CurToken()),
          CurToken().GetLocation());
  return std::make_unique<ErrorNode>();
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
  if (CurToken().IsNot(Token::identifier) && CurToken().IsNot(Token::l_paren)) {
    return "";
  }

  // Anonymous namespaces need to be treated specially: They are represented
  // the the string '(anonymous namespace)', which has a space in it (throwing
  // off normal parsing) and is not actually proper C++> Check to see if we're
  // looking at '(anonymous namespace)::...'
  if (CurToken().Is(Token::l_paren)) {
    // Look for all the pieces, in order:
    // l_paren 'anonymous' 'namespace' r_paren coloncolon
    if (m_dil_lexer.LookAhead(1).Is(Token::identifier) &&
        (m_dil_lexer.LookAhead(1).GetSpelling() == "anonymous") &&
        m_dil_lexer.LookAhead(2).Is(Token::identifier) &&
        (m_dil_lexer.LookAhead(2).GetSpelling() == "namespace") &&
        m_dil_lexer.LookAhead(3).Is(Token::r_paren) &&
        m_dil_lexer.LookAhead(4).Is(Token::coloncolon)) {
      m_dil_lexer.Advance(4);

      assert(
          (CurToken().Is(Token::identifier) || CurToken().Is(Token::l_paren)) &&
          "Expected an identifier or anonymous namespace, but not found.");
      // Continue parsing the nested_namespace_specifier.
      std::string identifier2 = ParseNestedNameSpecifier();
      if (identifier2.empty()) {
        Expect(Token::identifier);
        identifier2 = CurToken().GetSpelling();
        m_dil_lexer.Advance();
      }
      return "(anonymous namespace)::" + identifier2;
    }

    return "";
  } // end of special handling for '(anonymous namespace)'

  // If the next token is scope ("::"), then this is indeed a
  // nested_name_specifier
  if (m_dil_lexer.LookAhead(1).Is(Token::coloncolon)) {
    // This nested_name_specifier is a single identifier.
    std::string identifier = CurToken().GetSpelling();
    m_dil_lexer.Advance(1);
    Expect(Token::coloncolon);
    m_dil_lexer.Advance();
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
//    ? Token::identifier ?
//
std::string DILParser::ParseIdExpression() {
  // Try parsing optional global scope operator.
  bool global_scope = false;
  if (CurToken().Is(Token::coloncolon)) {
    global_scope = true;
    m_dil_lexer.Advance();
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
    Expect(Token::identifier);
    std::string identifier = CurToken().GetSpelling();
    m_dil_lexer.Advance();
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
//    ? Token::identifier ?
//
std::string DILParser::ParseUnqualifiedId() {
  Expect(Token::identifier);
  std::string identifier = CurToken().GetSpelling();
  m_dil_lexer.Advance();
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
  // Advance the lexer token index to the end of the lexed tokens vector.
  m_dil_lexer.ResetTokenIdx(m_dil_lexer.NumLexedTokens() - 1);
}

void DILParser::BailOut(Status error) {
  if (m_error.Fail()) {
    // If error is already set, then the parser is in the "bail-out" mode. Don't
    // do anything and keep the original error.
    return;
  }
  m_error = std::move(error);
  // Advance the lexer token index to the end of the lexed tokens vector.
  m_dil_lexer.ResetTokenIdx(m_dil_lexer.NumLexedTokens() - 1);
}

void DILParser::Expect(Token::Kind kind) {
  if (CurToken().IsNot(kind)) {
    BailOut(ErrorCode::kUnknown,
            llvm::formatv("expected {0}, got: {1}", kind, CurToken()),
            CurToken().GetLocation());
  }
}

} // namespace lldb_private::dil
