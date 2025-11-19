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
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Utility/DiagnosticsRendering.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/DILEval.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatAdapters.h"
#include <cstdlib>
#include <limits.h>
#include <memory>
#include <sstream>
#include <string>

namespace lldb_private::dil {

DILDiagnosticError::DILDiagnosticError(llvm::StringRef expr,
                                       const std::string &message, uint32_t loc,
                                       uint16_t err_len)
    : ErrorInfo(make_error_code(std::errc::invalid_argument)) {
  DiagnosticDetail::SourceLocation sloc = {
      FileSpec{}, /*line=*/1, static_cast<uint16_t>(loc + 1),
      err_len,    false,      /*in_user_input=*/true};
  std::string rendered_msg =
      llvm::formatv("<user expression 0>:1:{0}: {1}\n   1 | {2}\n     | ^",
                    loc + 1, message, expr);
  m_detail.source_location = sloc;
  m_detail.severity = lldb::eSeverityError;
  m_detail.message = message;
  m_detail.rendered = std::move(rendered_msg);
}

llvm::Expected<ASTNodeUP>
DILParser::Parse(llvm::StringRef dil_input_expr, DILLexer lexer,
                 std::shared_ptr<StackFrame> frame_sp,
                 lldb::DynamicValueType use_dynamic, bool use_synthetic,
                 bool fragile_ivar, bool check_ptr_vs_member) {
  llvm::Error error = llvm::Error::success();
  DILParser parser(dil_input_expr, lexer, frame_sp, use_dynamic, use_synthetic,
                   fragile_ivar, check_ptr_vs_member, error);

  ASTNodeUP node_up = parser.Run();

  if (error)
    return error;

  return node_up;
}

DILParser::DILParser(llvm::StringRef dil_input_expr, DILLexer lexer,
                     std::shared_ptr<StackFrame> frame_sp,
                     lldb::DynamicValueType use_dynamic, bool use_synthetic,
                     bool fragile_ivar, bool check_ptr_vs_member,
                     llvm::Error &error)
    : m_ctx_scope(frame_sp), m_input_expr(dil_input_expr),
      m_dil_lexer(std::move(lexer)), m_error(error), m_use_dynamic(use_dynamic),
      m_use_synthetic(use_synthetic), m_fragile_ivar(fragile_ivar),
      m_check_ptr_vs_member(check_ptr_vs_member) {}

ASTNodeUP DILParser::Run() {
  ASTNodeUP expr = ParseExpression();

  Expect(Token::Kind::eof);

  return expr;
}

// Parse an expression.
//
//  expression:
//    cast_expression
//
ASTNodeUP DILParser::ParseExpression() { return ParseCastExpression(); }

// Parse a cast_expression.
//
// cast_expression:
//   unary_expression
//   "(" type_id ")" cast_expression

ASTNodeUP DILParser::ParseCastExpression() {
  if (!CurToken().Is(Token::l_paren))
    return ParseUnaryExpression();

  // This could be a type cast, try parsing the contents as a type declaration.
  Token token = CurToken();
  uint32_t loc = token.GetLocation();

  // Enable lexer backtracking, so that we can rollback in case it's not
  // actually a type declaration.

  // Start tentative parsing (save token location/idx, for possible rollback).
  uint32_t save_token_idx = m_dil_lexer.GetCurrentTokenIdx();

  // Consume the token only after enabling the backtracking.
  m_dil_lexer.Advance();

  // Try parsing the type declaration. If the returned value is not valid,
  // then we should rollback and try parsing the expression.
  auto type_id = ParseTypeId();
  if (type_id) {
    // Successfully parsed the type declaration. Commit the backtracked
    // tokens and parse the cast_expression.

    if (!type_id.value().IsValid())
      return std::make_unique<ErrorNode>();

    Expect(Token::r_paren);
    m_dil_lexer.Advance();
    auto rhs = ParseCastExpression();

    return std::make_unique<CastNode>(
        loc, type_id.value(), std::move(rhs), CastKind::eNone);
  }

  // Failed to parse the contents of the parentheses as a type declaration.
  // Rollback the lexer and try parsing it as unary_expression.
  TentativeParsingRollback(save_token_idx);

  return ParseUnaryExpression();
}

// Parse an unary_expression.
//
//  unary_expression:
//    postfix_expression
//    unary_operator cast_expression
//
//  unary_operator:
//    "&"
//    "*"
//
ASTNodeUP DILParser::ParseUnaryExpression() {
  if (CurToken().IsOneOf({Token::amp, Token::star})) {
    Token token = CurToken();
    uint32_t loc = token.GetLocation();
    m_dil_lexer.Advance();
    auto rhs = ParseCastExpression();
    switch (token.GetKind()) {
    case Token::star:
      return std::make_unique<UnaryOpNode>(loc, UnaryOpKind::Deref,
                                           std::move(rhs));
    case Token::amp:
      return std::make_unique<UnaryOpNode>(loc, UnaryOpKind::AddrOf,
                                           std::move(rhs));

    default:
      llvm_unreachable("invalid token kind");
    }
  }
  return ParsePostfixExpression();
}

// Parse a postfix_expression.
//
//  postfix_expression:
//    primary_expression
//    postfix_expression "[" integer_literal "]"
//    postfix_expression "[" integer_literal "-" integer_literal "]"
//    postfix_expression "." id_expression
//    postfix_expression "->" id_expression
//
ASTNodeUP DILParser::ParsePostfixExpression() {
  ASTNodeUP lhs = ParsePrimaryExpression();
  while (CurToken().IsOneOf({Token::l_square, Token::period, Token::arrow})) {
    uint32_t loc = CurToken().GetLocation();
    Token token = CurToken();
    switch (token.GetKind()) {
    case Token::l_square: {
      m_dil_lexer.Advance();
      std::optional<int64_t> index = ParseIntegerConstant();
      if (!index) {
        BailOut(
            llvm::formatv("failed to parse integer constant: {0}", CurToken()),
            CurToken().GetLocation(), CurToken().GetSpelling().length());
        return std::make_unique<ErrorNode>();
      }
      if (CurToken().GetKind() == Token::minus) {
        m_dil_lexer.Advance();
        std::optional<int64_t> last_index = ParseIntegerConstant();
        if (!last_index) {
          BailOut(llvm::formatv("failed to parse integer constant: {0}",
                                CurToken()),
                  CurToken().GetLocation(), CurToken().GetSpelling().length());
          return std::make_unique<ErrorNode>();
        }
        lhs = std::make_unique<BitFieldExtractionNode>(
            loc, std::move(lhs), std::move(*index), std::move(*last_index));
      } else {
        lhs = std::make_unique<ArraySubscriptNode>(loc, std::move(lhs),
                                                   std::move(*index));
      }
      Expect(Token::r_square);
      m_dil_lexer.Advance();
      break;
    }
    case Token::period:
    case Token::arrow: {
      m_dil_lexer.Advance();
      Token member_token = CurToken();
      std::string member_id = ParseIdExpression();
      lhs = std::make_unique<MemberOfNode>(
          member_token.GetLocation(), std::move(lhs),
          token.GetKind() == Token::arrow, member_id);
      break;
    }
    default:
      llvm_unreachable("invalid token");
    }
  }

  return lhs;
}

// Parse a primary_expression.
//
//  primary_expression:
//    numeric_literal
//    boolean_literal
//    id_expression
//    "(" expression ")"
//
ASTNodeUP DILParser::ParsePrimaryExpression() {
  if (CurToken().IsOneOf({Token::integer_constant, Token::float_constant}))
    return ParseNumericLiteral();
  if (CurToken().IsOneOf({Token::kw_true, Token::kw_false}))
    return ParseBooleanLiteral();
  if (CurToken().IsOneOf(
          {Token::coloncolon, Token::identifier, Token::l_paren})) {
    // Save the source location for the diagnostics message.
    uint32_t loc = CurToken().GetLocation();
    std::string identifier = ParseIdExpression();

    if (!identifier.empty())
      return std::make_unique<IdentifierNode>(loc, identifier);
  }

  if (CurToken().Is(Token::l_paren)) {
    m_dil_lexer.Advance();
    auto expr = ParseExpression();
    Expect(Token::r_paren);
    m_dil_lexer.Advance();
    return expr;
  }

  BailOut(llvm::formatv("Unexpected token: {0}", CurToken()),
          CurToken().GetLocation(), CurToken().GetSpelling().length());
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
  switch (CurToken().GetKind()) {
  case Token::l_paren: {
    // Anonymous namespaces need to be treated specially: They are
    // represented the the string '(anonymous namespace)', which has a
    // space in it (throwing off normal parsing) and is not actually
    // proper C++> Check to see if we're looking at
    // '(anonymous namespace)::...'

    // Look for all the pieces, in order:
    // l_paren 'anonymous' 'namespace' r_paren coloncolon
    if (m_dil_lexer.LookAhead(1).Is(Token::identifier) &&
        (m_dil_lexer.LookAhead(1).GetSpelling() == "anonymous") &&
        m_dil_lexer.LookAhead(2).Is(Token::identifier) &&
        (m_dil_lexer.LookAhead(2).GetSpelling() == "namespace") &&
        m_dil_lexer.LookAhead(3).Is(Token::r_paren) &&
        m_dil_lexer.LookAhead(4).Is(Token::coloncolon)) {
      m_dil_lexer.Advance(4);

      Expect(Token::coloncolon);
      m_dil_lexer.Advance();
      if (!CurToken().Is(Token::identifier) && !CurToken().Is(Token::l_paren)) {
        BailOut("Expected an identifier or anonymous namespace, but not found.",
                CurToken().GetLocation(), CurToken().GetSpelling().length());
      }
      // Continue parsing the nested_namespace_specifier.
      std::string identifier2 = ParseNestedNameSpecifier();

      return "(anonymous namespace)::" + identifier2;
    }

    return "";
  } // end of special handling for '(anonymous namespace)'
  case Token::identifier: {
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
  default:
    return "";
  }
}

// Parse a type_id.
//
//  type_id:
//    type_specifier_seq [abstract_declarator]
//
//  type_specifier_seq:
//    type_specifier [type_specifier]
//
//  type_specifier:
//    ["::"] [nested_name_specifier] type_name // not handled for now!
//    builtin_typename
//
std::optional<CompilerType> DILParser::ParseTypeId() {
  CompilerType type;
  // For now only allow builtin types -- will expand add to this later.
  auto maybe_builtin_type = ParseBuiltinType();
  if (maybe_builtin_type) {
    type = *maybe_builtin_type;
  } else
    return {};

  //
  //  abstract_declarator:
  //    ptr_operator [abstract_declarator]
  //
  std::vector<Token> ptr_operators;
  while (CurToken().IsOneOf({Token::star, Token::amp})) {
    Token tok = CurToken();
    ptr_operators.push_back(std::move(tok));
    m_dil_lexer.Advance();
  }
  type = ResolveTypeDeclarators(type, ptr_operators);

  return type;
}

// Parse a built-in type
//
// builtin_typename:
//   identifer_seq
//
//  identifier_seq
//    identifer [identifier_seq]
//
// A built-in type can be a single identifier or a space-separated
// list of identifiers (e.g. "short" or "long long").
std::optional<CompilerType> DILParser::ParseBuiltinType() {
  std::string type_name = "";
  uint32_t save_token_idx = m_dil_lexer.GetCurrentTokenIdx();
  bool first_word = true;
  while (CurToken().GetKind() == Token::identifier) {
    if (CurToken().GetSpelling() == "const" ||
        CurToken().GetSpelling() == "volatile")
      continue;
    if (!first_word)
      type_name.push_back(' ');
    else
      first_word = false;
    type_name.append(CurToken().GetSpelling());
    m_dil_lexer.Advance();
  }

  if (type_name.size() > 0) {
    lldb::TargetSP target_sp = m_ctx_scope->CalculateTarget();
    ConstString const_type_name(type_name.c_str());
    for (auto type_system_sp : target_sp->GetScratchTypeSystems())
      if (auto compiler_type =
              type_system_sp->GetBuiltinTypeByName(const_type_name))
        return compiler_type;
  }

  TentativeParsingRollback(save_token_idx);
  return {};
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
  std::string nested_name_specifier = ParseNestedNameSpecifier();

  // If nested_name_specifier is present, then it's qualified_id production.
  // Follow the first production rule.
  if (!nested_name_specifier.empty()) {
    // Parse unqualified_id and construct a fully qualified id expression.
    auto unqualified_id = ParseUnqualifiedId();

    return llvm::formatv("{0}{1}{2}", global_scope ? "::" : "",
                         nested_name_specifier, unqualified_id);
  }

  if (!CurToken().Is(Token::identifier))
    return "";

  // No nested_name_specifier, but with global scope -- this is also a
  // qualified_id production. Follow the second production rule.
  if (global_scope) {
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

CompilerType
DILParser::ResolveTypeDeclarators(CompilerType type,
                                  const std::vector<Token> &ptr_operators) {
  // Resolve pointers/references.
  for (Token tk : ptr_operators) {
    uint32_t loc = tk.GetLocation();
    if (tk.GetKind() == Token::star) {
      // Pointers to reference types are forbidden.
      if (type.IsReferenceType()) {
        BailOut(llvm::formatv("'type name' declared as a pointer to a "
                              "reference of type {0}",
                              type.TypeDescription()),
                loc, CurToken().GetSpelling().length());
        return {};
      }
      // Get pointer type for the base type: e.g. int* -> int**.
      type = type.GetPointerType();

    } else if (tk.GetKind() == Token::amp) {
      // References to references are forbidden.
      // FIXME: In future we may want to allow rvalue references (i.e. &&).
      if (type.IsReferenceType()) {
        BailOut("type name declared as a reference to a reference", loc,
                CurToken().GetSpelling().length());
        return {};
      }
      // Get reference type for the base type: e.g. int -> int&.
      type = type.GetLValueReferenceType();
    }
  }

  return type;
}

// Parse an boolean_literal.
//
//  boolean_literal:
//    "true"
//    "false"
//
ASTNodeUP DILParser::ParseBooleanLiteral() {
  ExpectOneOf(std::vector<Token::Kind>{Token::kw_true, Token::kw_false});
  uint32_t loc = CurToken().GetLocation();
  bool literal_value = CurToken().Is(Token::kw_true);
  m_dil_lexer.Advance();
  return std::make_unique<BooleanLiteralNode>(loc, literal_value);
}

void DILParser::BailOut(const std::string &error, uint32_t loc,
                        uint16_t err_len) {
  if (m_error)
    // If error is already set, then the parser is in the "bail-out" mode. Don't
    // do anything and keep the original error.
    return;

  m_error =
      llvm::make_error<DILDiagnosticError>(m_input_expr, error, loc, err_len);
  // Advance the lexer token index to the end of the lexed tokens vector.
  m_dil_lexer.ResetTokenIdx(m_dil_lexer.NumLexedTokens() - 1);
}

// FIXME: Remove this once subscript operator uses ScalarLiteralNode.
// Parse a integer_literal.
//
//  integer_literal:
//    ? Integer constant ?
//
std::optional<int64_t> DILParser::ParseIntegerConstant() {
  std::string number_spelling;
  if (CurToken().GetKind() == Token::minus) {
    // StringRef::getAsInteger<>() can parse negative numbers.
    // FIXME: Remove this once unary minus operator is added.
    number_spelling = "-";
    m_dil_lexer.Advance();
  }
  number_spelling.append(CurToken().GetSpelling());
  llvm::StringRef spelling_ref = number_spelling;
  int64_t raw_value;
  if (!spelling_ref.getAsInteger<int64_t>(0, raw_value)) {
    m_dil_lexer.Advance();
    return raw_value;
  }

  return std::nullopt;
}

// Parse a numeric_literal.
//
//  numeric_literal:
//    ? Token::integer_constant ?
//    ? Token::floating_constant ?
//
ASTNodeUP DILParser::ParseNumericLiteral() {
  ASTNodeUP numeric_constant;
  if (CurToken().Is(Token::integer_constant))
    numeric_constant = ParseIntegerLiteral();
  else
    numeric_constant = ParseFloatingPointLiteral();
  if (!numeric_constant) {
    BailOut(llvm::formatv("Failed to parse token as numeric-constant: {0}",
                          CurToken()),
            CurToken().GetLocation(), CurToken().GetSpelling().length());
    return std::make_unique<ErrorNode>();
  }
  m_dil_lexer.Advance();
  return numeric_constant;
}

ASTNodeUP DILParser::ParseIntegerLiteral() {
  Token token = CurToken();
  auto spelling = token.GetSpelling();
  llvm::StringRef spelling_ref = spelling;

  auto radix = llvm::getAutoSenseRadix(spelling_ref);
  IntegerTypeSuffix type = IntegerTypeSuffix::None;
  bool is_unsigned = false;
  if (spelling_ref.consume_back_insensitive("u"))
    is_unsigned = true;
  if (spelling_ref.consume_back_insensitive("ll"))
    type = IntegerTypeSuffix::LongLong;
  else if (spelling_ref.consume_back_insensitive("l"))
    type = IntegerTypeSuffix::Long;
  // Suffix 'u' can be only specified only once, before or after 'l'
  if (!is_unsigned && spelling_ref.consume_back_insensitive("u"))
    is_unsigned = true;

  llvm::APInt raw_value;
  if (!spelling_ref.getAsInteger(radix, raw_value))
    return std::make_unique<IntegerLiteralNode>(token.GetLocation(), raw_value,
                                                radix, is_unsigned, type);
  return nullptr;
}

ASTNodeUP DILParser::ParseFloatingPointLiteral() {
  Token token = CurToken();
  auto spelling = token.GetSpelling();
  llvm::StringRef spelling_ref = spelling;

  llvm::APFloat raw_float(llvm::APFloat::IEEEdouble());
  if (spelling_ref.consume_back_insensitive("f"))
    raw_float = llvm::APFloat(llvm::APFloat::IEEEsingle());

  auto StatusOrErr = raw_float.convertFromString(
      spelling_ref, llvm::APFloat::rmNearestTiesToEven);
  if (!errorToBool(StatusOrErr.takeError()))
    return std::make_unique<FloatLiteralNode>(token.GetLocation(), raw_float);
  return nullptr;
}

void DILParser::Expect(Token::Kind kind) {
  if (CurToken().IsNot(kind)) {
    BailOut(llvm::formatv("expected {0}, got: {1}", kind, CurToken()),
            CurToken().GetLocation(), CurToken().GetSpelling().length());
  }
}

void DILParser::ExpectOneOf(std::vector<Token::Kind> kinds_vec) {
  if (!CurToken().IsOneOf(kinds_vec)) {
    BailOut(llvm::formatv("expected any of ({0}), got: {1}",
                          llvm::iterator_range(kinds_vec), CurToken()),
            CurToken().GetLocation(), CurToken().GetSpelling().length());
  }
}

} // namespace lldb_private::dil
