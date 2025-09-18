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

llvm::Expected<lldb::TypeSystemSP>
DILGetTypeSystemFromCU(std::shared_ptr<StackFrame> ctx) {
  SymbolContext symbol_context =
      ctx->GetSymbolContext(lldb::eSymbolContextCompUnit);
  lldb::LanguageType language = symbol_context.comp_unit->GetLanguage();

  symbol_context = ctx->GetSymbolContext(lldb::eSymbolContextModule);
  return symbol_context.module_sp->GetTypeSystemForLanguage(language);
}

CompilerType
ResolveTypeByName(const std::string &name,
                  std::shared_ptr<ExecutionContextScope> ctx_scope) {
  // Internally types don't have global scope qualifier in their names and
  // LLDB doesn't support queries with it too.
  llvm::StringRef name_ref(name);

  if (name_ref.starts_with("::"))
    name_ref = name_ref.drop_front(2);

  std::vector<CompilerType> result_type_list;
  lldb::TargetSP target_sp = ctx_scope->CalculateTarget();
  const char *type_name = name_ref.data();
  if (type_name && type_name[0] && target_sp) {
    ModuleList &images = target_sp->GetImages();
    ConstString const_type_name(type_name);
    TypeQuery query(type_name);
    TypeResults results;
    images.FindTypes(nullptr, query, results);
    for (const lldb::TypeSP &type_sp : results.GetTypeMap().Types())
      if (type_sp)
        result_type_list.push_back(type_sp->GetFullCompilerType());

    if (auto process_sp = target_sp->GetProcessSP()) {
      for (auto *runtime : process_sp->GetLanguageRuntimes()) {
        if (auto *vendor = runtime->GetDeclVendor()) {
          auto types = vendor->FindTypes(const_type_name, UINT32_MAX);
          for (auto type : types)
            result_type_list.push_back(type);
        }
      }
    }

    if (result_type_list.empty()) {
      for (auto type_system_sp : target_sp->GetScratchTypeSystems())
        if (auto compiler_type =
                type_system_sp->GetBuiltinTypeByName(const_type_name))
          result_type_list.push_back(compiler_type);
    }
  }

  // We've found multiple types, try finding the "correct" one.
  CompilerType full_match;
  std::vector<CompilerType> partial_matches;

  for (uint32_t i = 0; i < result_type_list.size(); ++i) {
    CompilerType type = result_type_list[i];
    llvm::StringRef type_name_ref = type.GetTypeName().GetStringRef();
    ;

    if (type_name_ref == name_ref)
      full_match = type;
    else if (type_name_ref.ends_with(name_ref))
      partial_matches.push_back(type);
  }

  // Full match is always correct.
  if (full_match.IsValid())
    return full_match;

  // If we have partial matches, pick a "random" one.
  if (partial_matches.size() > 0)
    return partial_matches.back();

  return {};
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
  // This can be a C-style cast, try parsing the contents as a type declaration.
  if (CurToken().Is(Token::l_paren)) {
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

      // return BuildCStyleCast(type_id.value(), std::move(rhs),
      //                        token.GetLocation());
      return std::make_unique<CStyleCastNode>(
          loc, type_id.value(), std::move(rhs), CStyleCastKind::eNone);
    }

    // Failed to parse the contents of the parentheses as a type declaration.
    // Rollback the lexer and try parsing it as unary_expression.
    TentativeParsingRollback(save_token_idx);
  }

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
std::optional<CompilerType> DILParser::ParseTypeId(bool must_be_type_id) {
  uint32_t type_loc = CurToken().GetLocation();
  TypeDeclaration type_decl;

  // type_specifier_seq is required here, start with trying to parse it.
  ParseTypeSpecifierSeq(&type_decl);

  if (type_decl.IsEmpty()) {
    // TODO: Should we bail out if `must_be_type_id` is set?
    return {};
  }

  if (type_decl.m_has_error) {
    if (type_decl.m_is_builtin) {
      return {};
    }

    assert(type_decl.m_is_user_type && "type_decl must be a user type");
    // Found something looking like a user type, but failed to parse it.
    // Return invalid type if we expect to have a type here, otherwise nullopt.
    if (must_be_type_id) {
      return {};
    }
    return {};
  }

  // Try to resolve the base type.
  CompilerType type;
  if (type_decl.m_is_builtin) {
    llvm::Expected<lldb::TypeSystemSP> type_system =
        DILGetTypeSystemFromCU(m_ctx_scope);
    if (!type_system)
      return {};
    // type = GetBasicType(m_ctx_scope, type_decl.GetBasicType());
    // type = DILGetBasicType(*type_system, type_decl.GetBasicType());
    type = (*type_system).get()->GetBasicTypeFromAST(type_decl.GetBasicType());
    assert(type.IsValid() && "cannot resolve basic type");

  } else {
    assert(type_decl.m_is_user_type && "type_decl must be a user type");
    type = ResolveTypeByName(type_decl.m_user_typename, m_ctx_scope);
    if (!type.IsValid()) {
      if (must_be_type_id) {
        BailOut(
            llvm::formatv("unknown type name '{0}'", type_decl.m_user_typename),
            type_loc, CurToken().GetSpelling().length());
        return {};
      }
      return {};
    }

    if (LookupIdentifier(type_decl.m_user_typename, m_ctx_scope,
                         m_use_dynamic)) {
      // Same-name identifiers should be preferred over typenames.
      // TODO: Make type accessible with 'class', 'struct' and 'union' keywords.
      if (must_be_type_id) {
        BailOut(llvm::formatv(
                    "must use '{0}' tag to refer to type '{1}' in this scope",
                    type.GetTypeTag(), type_decl.m_user_typename),
                type_loc, CurToken().GetSpelling().length());
        return {};
      }
      return {};
    }

    if (LookupGlobalIdentifier(type_decl.m_user_typename, m_ctx_scope,
                               m_ctx_scope->CalculateTarget(), m_use_dynamic)) {
      // Same-name identifiers should be preferred over typenames.
      // TODO: Make type accessible with 'class', 'struct' and 'union' keywords.
      if (must_be_type_id) {
        BailOut(llvm::formatv(
                    "must use '{0}' tag to refer to type '{1}' in this scope",
                    type.GetTypeTag(), type_decl.m_user_typename),
                type_loc, CurToken().GetSpelling().length());
        return {};
      }
      return {};
    }
  }

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

// Parse a type_specifier_seq.
//
//  type_specifier_seq:
//    type_specifier [type_specifier_seq]
//
void DILParser::ParseTypeSpecifierSeq(TypeDeclaration *type_decl) {
  while (true) {
    bool type_specifier = ParseTypeSpecifier(type_decl);
    if (!type_specifier) {
      break;
    }
  }
}

// Parse a type_specifier.
//
//  type_specifier:
//    simple_type_specifier
//    cv_qualifier
//
//  simple_type_specifier:
//    ["::"] [nested_name_specifier] type_name
//    "char"
//    "bool"
//    "integer"
//    "float"
//    "void"
//
// Returns TRUE if a type_specifier was successfully parsed at this location.
//
bool DILParser::ParseTypeSpecifier(TypeDeclaration *type_decl) {
  if (IsSimpleTypeSpecifierKeyword(CurToken())) {
    // User-defined typenames can't be combined with builtin keywords.
    if (type_decl->m_is_user_type) {
      BailOut("cannot combine with previous declaration specifier",
              CurToken().GetLocation(), CurToken().GetSpelling().length());
      type_decl->m_has_error = true;
      return false;
    }

    // From now on this type declaration must describe a builtin type.
    // TODO: Should this be allowed -- `unsigned myint`?
    type_decl->m_is_builtin = true;

    if (!HandleSimpleTypeSpecifier(type_decl)) {
      type_decl->m_has_error = true;
      return false;
    }
    m_dil_lexer.Advance();
    return true;
  }

  // The type_specifier must be a user-defined type. Try parsing a
  // simple_type_specifier.
  {
    // Try parsing optional global scope operator.
    bool global_scope = false;
    if (CurToken().Is(Token::coloncolon)) {
      global_scope = true;
      m_dil_lexer.Advance();
    }

    uint32_t loc = CurToken().GetLocation();

    // Try parsing optional nested_name_specifier.
    auto nested_name_specifier = ParseNestedNameSpecifier();

    // Try parsing required type_name.
    auto type_name = ParseTypeName();

    // If there is a type_name, then this is indeed a simple_type_specifier.
    // Global and qualified (namespace/class) scopes can be empty, since they're
    // optional. In this case type_name is type we're looking for.
    if (!type_name.empty()) {
      // User-defined typenames can't be combined with builtin keywords.
      if (type_decl->m_is_builtin) {
        BailOut("cannot combine with previous declaration specifier", loc,
                CurToken().GetSpelling().length());
        type_decl->m_has_error = true;
        return false;
      }
      // There should be only one user-defined typename.
      if (type_decl->m_is_user_type) {
        BailOut("two or more data types in declaration of 'type name'", loc,
                CurToken().GetSpelling().length());
        type_decl->m_has_error = true;
        return false;
      }

      // Construct the fully qualified typename.
      type_decl->m_is_user_type = true;
      type_decl->m_user_typename =
          llvm::formatv("{0}{1}{2}", global_scope ? "::" : "",
                        nested_name_specifier, type_name);
      return true;
    }
  }

  // No type_specifier was found here.
  return false;
}

// Parse a type_name.
//
//  type_name:
//    class_name
//    enum_name
//    typedef_name
//
//  class_name
//    identifier
//
//  enum_name
//    identifier
//
//  typedef_name
//    identifier
//
std::string DILParser::ParseTypeName() {
  // Typename always starts with an identifier.
  if (CurToken().IsNot(Token::identifier)) {
    return "";
  }

  // Otherwise look for a class_name, enum_name or a typedef_name.
  std::string identifier = CurToken().GetSpelling();
  m_dil_lexer.Advance();

  return identifier;
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
  CompilerType bad_type;
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
        return bad_type;
      }
      // Get pointer type for the base type: e.g. int* -> int**.
      type = type.GetPointerType();

    } else if (tk.GetKind() == Token::amp) {
      // References to references are forbidden.
      if (type.IsReferenceType()) {
        BailOut("type name declared as a reference to a reference", loc,
                CurToken().GetSpelling().length());
        return bad_type;
      }
      // Get reference type for the base type: e.g. int -> int&.
      type = type.GetLValueReferenceType();
    }
  }

  return type;
}

bool DILParser::IsSimpleTypeSpecifierKeyword(Token token) const {
  if (token.GetKind() != Token::identifier)
    return false;
  if (token.GetSpelling() == "bool" || token.GetSpelling() == "char" ||
      token.GetSpelling() == "int" || token.GetSpelling() == "float" ||
      token.GetSpelling() == "void" || token.GetSpelling() == "short" ||
      token.GetSpelling() == "long" || token.GetSpelling() == "signed" ||
      token.GetSpelling() == "unsigned" || token.GetSpelling() == "double")
    return true;
  return false;
}

bool DILParser::HandleSimpleTypeSpecifier(TypeDeclaration *type_decl) {
  using TypeSpecifier = TypeDeclaration::TypeSpecifier;
  using SignSpecifier = TypeDeclaration::SignSpecifier;

  TypeSpecifier type_spec = type_decl->m_type_specifier;
  uint32_t loc = CurToken().GetLocation();
  std::string kind = CurToken().GetSpelling();

  // switch (kind) {
  if (kind == "int") {
    // case Token::kw_int: {
    //  "int" can have signedness and be combined with "short", "long" and
    //  "long long" (but not with another "int").
    if (type_decl->m_has_int_specifier) {
      BailOut("cannot combine with previous 'int' declaration specifier", loc,
              CurToken().GetSpelling().length());
      return false;
    }
    if (type_spec == TypeSpecifier::kShort ||
        type_spec == TypeSpecifier::kLong ||
        type_spec == TypeSpecifier::kLongLong) {
      type_decl->m_has_int_specifier = true;
      return true;
    } else if (type_spec == TypeSpecifier::kUnknown) {
      type_decl->m_type_specifier = TypeSpecifier::kInt;
      type_decl->m_has_int_specifier = true;
      return true;
    }
    BailOut(llvm::formatv(
                "cannot combine with previous '{0}' declaration specifier",
                type_spec),
            loc, CurToken().GetSpelling().length());
    return false;
  }

  if (kind == "long") {
    // "long" can have signedness and be combined with "int" or "long" to
    // form "long long".
    if (type_spec == TypeSpecifier::kUnknown ||
        type_spec == TypeSpecifier::kInt) {
      type_decl->m_type_specifier = TypeSpecifier::kLong;
      return true;
    } else if (type_spec == TypeSpecifier::kLong) {
      type_decl->m_type_specifier = TypeSpecifier::kLongLong;
      return true;
    } else if (type_spec == TypeSpecifier::kDouble) {
      type_decl->m_type_specifier = TypeSpecifier::kLongDouble;
      return true;
    }
    BailOut(llvm::formatv(
                "cannot combine with previous '{0}' declaration specifier",
                type_spec),
            loc, CurToken().GetSpelling().length());
    return false;
  }

  if (kind == "short") {
    // "short" can have signedness and be combined with "int".
    if (type_spec == TypeSpecifier::kUnknown ||
        type_spec == TypeSpecifier::kInt) {
      type_decl->m_type_specifier = TypeSpecifier::kShort;
      return true;
    }
    BailOut(llvm::formatv(
                "cannot combine with previous '{0}' declaration specifier",
                type_spec),
            loc, CurToken().GetSpelling().length());
    return false;
  }

  if (kind == "char") {
    // "char" can have signedness, but it cannot be combined with any other
    // type specifier.
    if (type_spec == TypeSpecifier::kUnknown) {
      type_decl->m_type_specifier = TypeSpecifier::kChar;
      return true;
    }
    BailOut(llvm::formatv(
                "cannot combine with previous '{0}' declaration specifier",
                type_spec),
            loc, CurToken().GetSpelling().length());
    return false;
  }

  if (kind == "double") {
    // "double" can be combined with "long" to form "long double", but it
    // cannot be combined with signedness specifier.
    if (type_decl->m_sign_specifier != SignSpecifier::kUnknown) {
      BailOut("'double' cannot be signed or unsigned", loc,
              CurToken().GetSpelling().length());
      return false;
    }
    if (type_spec == TypeSpecifier::kUnknown) {
      type_decl->m_type_specifier = TypeSpecifier::kDouble;
      return true;
    } else if (type_spec == TypeSpecifier::kLong) {
      type_decl->m_type_specifier = TypeSpecifier::kLongDouble;
      return true;
    }
    BailOut(llvm::formatv(
                "cannot combine with previous '{0}' declaration specifier",
                type_spec),
            loc, CurToken().GetSpelling().length());
    return false;
  }

  if (kind == "bool" || kind == "void" || kind == "float") {
    // These types cannot have signedness or be combined with any other type
    // specifiers.
    if (type_decl->m_sign_specifier != SignSpecifier::kUnknown) {
      BailOut(llvm::formatv("'{0}' cannot be signed or unsigned", kind), loc,
              CurToken().GetSpelling().length());
      return false;
    }
    if (type_spec != TypeSpecifier::kUnknown) {
      BailOut(llvm::formatv(
                  "cannot combine with previous '{0}' declaration specifier",
                  type_spec),
              loc, CurToken().GetSpelling().length());
    }
    if (kind == "bool")
      type_decl->m_type_specifier = TypeSpecifier::kBool;
    else if (kind == "void")
      type_decl->m_type_specifier = TypeSpecifier::kVoid;
    else if (kind == "float")
      type_decl->m_type_specifier = TypeSpecifier::kFloat;
    return true;
  }

  if (kind == "signed" || kind == "unsigned") {
    // "signed" and "unsigned" cannot be combined with another signedness
    // specifier.
    if (type_spec == TypeSpecifier::kVoid ||
        type_spec == TypeSpecifier::kBool ||
        type_spec == TypeSpecifier::kFloat ||
        type_spec == TypeSpecifier::kDouble ||
        type_spec == TypeSpecifier::kLongDouble) {
      BailOut(llvm::formatv("'{0}' cannot be signed or unsigned", type_spec),
              loc, CurToken().GetSpelling().length());
      return false;
    }

    type_decl->m_sign_specifier =
        (kind == "signed") ? SignSpecifier::kSigned : SignSpecifier::kUnsigned;
    return true;
  }

  BailOut(llvm::formatv("invalid simple type specifier kind"), loc,
          CurToken().GetSpelling().length());
  return false;
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


lldb::BasicType TypeDeclaration::GetBasicType() const {
  if (!m_is_builtin)
    return lldb::eBasicTypeInvalid;

  if (m_sign_specifier == SignSpecifier::kSigned &&
      m_type_specifier == TypeSpecifier::kChar) {
    // "signed char" isn't the same as "char".
    return lldb::eBasicTypeSignedChar;
  }

  if (m_sign_specifier == SignSpecifier::kUnsigned) {
    switch (m_type_specifier) {
    // "unsigned" is "unsigned int"
    case TypeSpecifier::kUnknown:
      return lldb::eBasicTypeUnsignedInt;
    case TypeSpecifier::kChar:
      return lldb::eBasicTypeUnsignedChar;
    case TypeSpecifier::kShort:
      return lldb::eBasicTypeUnsignedShort;
    case TypeSpecifier::kInt:
      return lldb::eBasicTypeUnsignedInt;
    case TypeSpecifier::kLong:
      return lldb::eBasicTypeUnsignedLong;
    case TypeSpecifier::kLongLong:
      return lldb::eBasicTypeUnsignedLongLong;
    default:
      // assert(false && "unknown unsigned basic type");
      return lldb::eBasicTypeInvalid;
    }
  }

  switch (m_type_specifier) {
  case TypeSpecifier::kUnknown:
    // "signed" is "signed int"
    if (m_sign_specifier != SignSpecifier::kSigned)
      return lldb::eBasicTypeInvalid;
    return lldb::eBasicTypeInt;
  case TypeSpecifier::kVoid:
    return lldb::eBasicTypeVoid;
  case TypeSpecifier::kBool:
    return lldb::eBasicTypeBool;
  case TypeSpecifier::kChar:
    return lldb::eBasicTypeChar;
  case TypeSpecifier::kShort:
    return lldb::eBasicTypeShort;
  case TypeSpecifier::kInt:
    return lldb::eBasicTypeInt;
  case TypeSpecifier::kLong:
    return lldb::eBasicTypeLong;
  case TypeSpecifier::kLongLong:
    return lldb::eBasicTypeLongLong;
  case TypeSpecifier::kFloat:
    return lldb::eBasicTypeFloat;
  case TypeSpecifier::kDouble:
    return lldb::eBasicTypeDouble;
  case TypeSpecifier::kLongDouble:
    return lldb::eBasicTypeLongDouble;
  }

  return lldb::eBasicTypeInvalid;
}

} // namespace lldb_private::dil
