//===-- GoParser.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "GoParser.h"

#include "Plugins/ExpressionParser/Go/GoAST.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/SmallString.h"

using namespace lldb_private;
using namespace lldb;

namespace {
llvm::StringRef DescribeToken(GoLexer::TokenType t) {
  switch (t) {
  case GoLexer::TOK_EOF:
    return "<eof>";
  case GoLexer::TOK_IDENTIFIER:
    return "identifier";
  case GoLexer::LIT_FLOAT:
    return "float";
  case GoLexer::LIT_IMAGINARY:
    return "imaginary";
  case GoLexer::LIT_INTEGER:
    return "integer";
  case GoLexer::LIT_RUNE:
    return "rune";
  case GoLexer::LIT_STRING:
    return "string";
  default:
    return GoLexer::LookupToken(t);
  }
}
} // namespace

class GoParser::Rule {
public:
  Rule(llvm::StringRef name, GoParser *p)
      : m_name(name), m_parser(p), m_pos(p->m_pos) {}

  std::nullptr_t error() {
    if (!m_parser->m_failed) {
      // Set m_error in case this is the top level.
      if (m_parser->m_last_tok == GoLexer::TOK_INVALID)
        m_parser->m_error = m_parser->m_last;
      else
        m_parser->m_error = DescribeToken(m_parser->m_last_tok);
      // And set m_last in case it isn't.
      m_parser->m_last = m_name;
      m_parser->m_last_tok = GoLexer::TOK_INVALID;
      m_parser->m_pos = m_pos;
    }
    return nullptr;
  }

private:
  llvm::StringRef m_name;
  GoParser *m_parser;
  size_t m_pos;
};

GoParser::GoParser(const char *src) : m_lexer(src), m_pos(0), m_failed(false) {}

GoASTStmt *GoParser::Statement() {
  Rule r("Statement", this);
  GoLexer::TokenType t = peek();
  GoASTStmt *ret = nullptr;
  switch (t) {
  case GoLexer::TOK_EOF:
  case GoLexer::OP_SEMICOLON:
  case GoLexer::OP_RPAREN:
  case GoLexer::OP_RBRACE:
  case GoLexer::TOK_INVALID:
    return EmptyStmt();
  case GoLexer::OP_LBRACE:
    return Block();

  /*      TODO:
case GoLexer::KEYWORD_GO:
  return GoStmt();
case GoLexer::KEYWORD_RETURN:
  return ReturnStmt();
case GoLexer::KEYWORD_BREAK:
case GoLexer::KEYWORD_CONTINUE:
case GoLexer::KEYWORD_GOTO:
case GoLexer::KEYWORD_FALLTHROUGH:
  return BranchStmt();
case GoLexer::KEYWORD_IF:
  return IfStmt();
case GoLexer::KEYWORD_SWITCH:
  return SwitchStmt();
case GoLexer::KEYWORD_SELECT:
  return SelectStmt();
case GoLexer::KEYWORD_FOR:
  return ForStmt();
case GoLexer::KEYWORD_DEFER:
  return DeferStmt();
case GoLexer::KEYWORD_CONST:
case GoLexer::KEYWORD_TYPE:
case GoLexer::KEYWORD_VAR:
  return DeclStmt();
case GoLexer::TOK_IDENTIFIER:
  if ((ret = LabeledStmt()) ||
      (ret = ShortVarDecl()))
  {
      return ret;
  }
*/
  default:
    break;
  }
  GoASTExpr *expr = Expression();
  if (expr == nullptr)
    return r.error();
  if (/*(ret = SendStmt(expr)) ||*/
      (ret = IncDecStmt(expr)) || (ret = Assignment(expr)) ||
      (ret = ExpressionStmt(expr))) {
    return ret;
  }
  delete expr;
  return r.error();
}

GoASTStmt *GoParser::ExpressionStmt(GoASTExpr *e) {
  if (Semicolon())
    return new GoASTExprStmt(e);
  return nullptr;
}

GoASTStmt *GoParser::IncDecStmt(GoASTExpr *e) {
  Rule r("IncDecStmt", this);
  if (match(GoLexer::OP_PLUS_PLUS))
    return Semicolon() ? new GoASTIncDecStmt(e, GoLexer::OP_PLUS_PLUS)
                       : r.error();
  if (match(GoLexer::OP_MINUS_MINUS))
    return Semicolon() ? new GoASTIncDecStmt(e, GoLexer::OP_MINUS_MINUS)
                       : r.error();
  return nullptr;
}

GoASTStmt *GoParser::Assignment(lldb_private::GoASTExpr *e) {
  Rule r("Assignment", this);
  std::vector<std::unique_ptr<GoASTExpr>> lhs;
  for (GoASTExpr *l = MoreExpressionList(); l; l = MoreExpressionList())
    lhs.push_back(std::unique_ptr<GoASTExpr>(l));
  switch (peek()) {
  case GoLexer::OP_EQ:
  case GoLexer::OP_PLUS_EQ:
  case GoLexer::OP_MINUS_EQ:
  case GoLexer::OP_PIPE_EQ:
  case GoLexer::OP_CARET_EQ:
  case GoLexer::OP_STAR_EQ:
  case GoLexer::OP_SLASH_EQ:
  case GoLexer::OP_PERCENT_EQ:
  case GoLexer::OP_LSHIFT_EQ:
  case GoLexer::OP_RSHIFT_EQ:
  case GoLexer::OP_AMP_EQ:
  case GoLexer::OP_AMP_CARET_EQ:
    break;
  default:
    return r.error();
  }
  // We don't want to own e until we know this is an assignment.
  std::unique_ptr<GoASTAssignStmt> stmt(new GoASTAssignStmt(false));
  stmt->AddLhs(e);
  for (auto &l : lhs)
    stmt->AddLhs(l.release());
  for (GoASTExpr *r = Expression(); r; r = MoreExpressionList())
    stmt->AddRhs(r);
  if (!Semicolon() || stmt->NumRhs() == 0)
    return new GoASTBadStmt;
  return stmt.release();
}

GoASTStmt *GoParser::EmptyStmt() {
  if (match(GoLexer::TOK_EOF))
    return nullptr;
  if (Semicolon())
    return new GoASTEmptyStmt;
  return nullptr;
}

GoASTStmt *GoParser::GoStmt() {
  if (match(GoLexer::KEYWORD_GO)) {
    if (GoASTCallExpr *e =
            llvm::dyn_cast_or_null<GoASTCallExpr>(Expression())) {
      return FinishStmt(new GoASTGoStmt(e));
    }
    m_last = "call expression";
    m_failed = true;
    return new GoASTBadStmt();
  }
  return nullptr;
}

GoASTStmt *GoParser::ReturnStmt() {
  if (match(GoLexer::KEYWORD_RETURN)) {
    std::unique_ptr<GoASTReturnStmt> r(new GoASTReturnStmt());
    for (GoASTExpr *e = Expression(); e; e = MoreExpressionList())
      r->AddResults(e);
    return FinishStmt(r.release());
  }
  return nullptr;
}

GoASTStmt *GoParser::BranchStmt() {
  GoLexer::Token *tok;
  if ((tok = match(GoLexer::KEYWORD_BREAK)) ||
      (tok = match(GoLexer::KEYWORD_CONTINUE)) ||
      (tok = match(GoLexer::KEYWORD_GOTO))) {
    auto *e = Identifier();
    if (tok->m_type == GoLexer::KEYWORD_GOTO && !e)
      return syntaxerror();
    return FinishStmt(new GoASTBranchStmt(e, tok->m_type));
  }
  if ((tok = match(GoLexer::KEYWORD_FALLTHROUGH)))
    return FinishStmt(new GoASTBranchStmt(nullptr, tok->m_type));

  return nullptr;
}

GoASTIdent *GoParser::Identifier() {
  if (auto *tok = match(GoLexer::TOK_IDENTIFIER))
    return new GoASTIdent(*tok);
  return nullptr;
}

GoASTExpr *GoParser::MoreExpressionList() {
  if (match(GoLexer::OP_COMMA)) {
    auto *e = Expression();
    if (!e)
      return syntaxerror();
    return e;
  }
  return nullptr;
}

GoASTIdent *GoParser::MoreIdentifierList() {
  if (match(GoLexer::OP_COMMA)) {
    auto *i = Identifier();
    if (!i)
      return syntaxerror();
    return i;
  }
  return nullptr;
}

GoASTExpr *GoParser::Expression() {
  Rule r("Expression", this);
  if (GoASTExpr *ret = OrExpr())
    return ret;
  return r.error();
}

GoASTExpr *GoParser::UnaryExpr() {
  switch (peek()) {
  case GoLexer::OP_PLUS:
  case GoLexer::OP_MINUS:
  case GoLexer::OP_BANG:
  case GoLexer::OP_CARET:
  case GoLexer::OP_STAR:
  case GoLexer::OP_AMP:
  case GoLexer::OP_LT_MINUS: {
    const GoLexer::Token t = next();
    if (GoASTExpr *e = UnaryExpr()) {
      if (t.m_type == GoLexer::OP_STAR)
        return new GoASTStarExpr(e);
      else
        return new GoASTUnaryExpr(t.m_type, e);
    }
    return syntaxerror();
  }
  default:
    return PrimaryExpr();
  }
}

GoASTExpr *GoParser::OrExpr() {
  std::unique_ptr<GoASTExpr> l(AndExpr());
  if (l) {
    while (match(GoLexer::OP_PIPE_PIPE)) {
      GoASTExpr *r = AndExpr();
      if (r)
        l.reset(new GoASTBinaryExpr(l.release(), r, GoLexer::OP_PIPE_PIPE));
      else
        return syntaxerror();
    }
    return l.release();
  }
  return nullptr;
}

GoASTExpr *GoParser::AndExpr() {
  std::unique_ptr<GoASTExpr> l(RelExpr());
  if (l) {
    while (match(GoLexer::OP_AMP_AMP)) {
      GoASTExpr *r = RelExpr();
      if (r)
        l.reset(new GoASTBinaryExpr(l.release(), r, GoLexer::OP_AMP_AMP));
      else
        return syntaxerror();
    }
    return l.release();
  }
  return nullptr;
}

GoASTExpr *GoParser::RelExpr() {
  std::unique_ptr<GoASTExpr> l(AddExpr());
  if (l) {
    for (GoLexer::Token *t;
         (t = match(GoLexer::OP_EQ_EQ)) || (t = match(GoLexer::OP_BANG_EQ)) ||
         (t = match(GoLexer::OP_LT)) || (t = match(GoLexer::OP_LT_EQ)) ||
         (t = match(GoLexer::OP_GT)) || (t = match(GoLexer::OP_GT_EQ));) {
      GoLexer::TokenType op = t->m_type;
      GoASTExpr *r = AddExpr();
      if (r)
        l.reset(new GoASTBinaryExpr(l.release(), r, op));
      else
        return syntaxerror();
    }
    return l.release();
  }
  return nullptr;
}

GoASTExpr *GoParser::AddExpr() {
  std::unique_ptr<GoASTExpr> l(MulExpr());
  if (l) {
    for (GoLexer::Token *t;
         (t = match(GoLexer::OP_PLUS)) || (t = match(GoLexer::OP_MINUS)) ||
         (t = match(GoLexer::OP_PIPE)) || (t = match(GoLexer::OP_CARET));) {
      GoLexer::TokenType op = t->m_type;
      GoASTExpr *r = MulExpr();
      if (r)
        l.reset(new GoASTBinaryExpr(l.release(), r, op));
      else
        return syntaxerror();
    }
    return l.release();
  }
  return nullptr;
}

GoASTExpr *GoParser::MulExpr() {
  std::unique_ptr<GoASTExpr> l(UnaryExpr());
  if (l) {
    for (GoLexer::Token *t;
         (t = match(GoLexer::OP_STAR)) || (t = match(GoLexer::OP_SLASH)) ||
         (t = match(GoLexer::OP_PERCENT)) || (t = match(GoLexer::OP_LSHIFT)) ||
         (t = match(GoLexer::OP_RSHIFT)) || (t = match(GoLexer::OP_AMP)) ||
         (t = match(GoLexer::OP_AMP_CARET));) {
      GoLexer::TokenType op = t->m_type;
      GoASTExpr *r = UnaryExpr();
      if (r)
        l.reset(new GoASTBinaryExpr(l.release(), r, op));
      else
        return syntaxerror();
    }
    return l.release();
  }
  return nullptr;
}

GoASTExpr *GoParser::PrimaryExpr() {
  GoASTExpr *l;
  GoASTExpr *r;
  (l = Conversion()) || (l = Operand());
  if (!l)
    return nullptr;
  while ((r = Selector(l)) || (r = IndexOrSlice(l)) || (r = TypeAssertion(l)) ||
         (r = Arguments(l))) {
    l = r;
  }
  return l;
}

GoASTExpr *GoParser::Operand() {
  GoLexer::Token *lit;
  if ((lit = match(GoLexer::LIT_INTEGER)) ||
      (lit = match(GoLexer::LIT_FLOAT)) ||
      (lit = match(GoLexer::LIT_IMAGINARY)) ||
      (lit = match(GoLexer::LIT_RUNE)) || (lit = match(GoLexer::LIT_STRING)))
    return new GoASTBasicLit(*lit);
  if (match(GoLexer::OP_LPAREN)) {
    GoASTExpr *e;
    if (!((e = Expression()) && match(GoLexer::OP_RPAREN)))
      return syntaxerror();
    return e;
  }
  // MethodExpr should be handled by Selector
  if (GoASTExpr *e = CompositeLit())
    return e;
  if (GoASTExpr *n = Name())
    return n;
  return FunctionLit();
}

GoASTExpr *GoParser::FunctionLit() {
  if (!match(GoLexer::KEYWORD_FUNC))
    return nullptr;
  auto *sig = Signature();
  if (!sig)
    return syntaxerror();
  auto *body = Block();
  if (!body) {
    delete sig;
    return syntaxerror();
  }
  return new GoASTFuncLit(sig, body);
}

GoASTBlockStmt *GoParser::Block() {
  if (!match(GoLexer::OP_LBRACE))
    return nullptr;
  std::unique_ptr<GoASTBlockStmt> block(new GoASTBlockStmt);
  for (auto *s = Statement(); s; s = Statement())
    block->AddList(s);
  if (!match(GoLexer::OP_RBRACE))
    return syntaxerror();
  return block.release();
}

GoASTExpr *GoParser::CompositeLit() {
  Rule r("CompositeLit", this);
  GoASTExpr *type;
  (type = StructType()) || (type = ArrayOrSliceType(true)) ||
      (type = MapType()) || (type = Name());
  if (!type)
    return r.error();
  GoASTCompositeLit *lit = LiteralValue();
  if (!lit)
    return r.error();
  lit->SetType(type);
  return lit;
}

GoASTCompositeLit *GoParser::LiteralValue() {
  if (!match(GoLexer::OP_LBRACE))
    return nullptr;
  std::unique_ptr<GoASTCompositeLit> lit(new GoASTCompositeLit);
  for (GoASTExpr *e = Element(); e; e = Element()) {
    lit->AddElts(e);
    if (!match(GoLexer::OP_COMMA))
      break;
  }
  if (!mustMatch(GoLexer::OP_RBRACE))
    return nullptr;
  return lit.release();
}

GoASTExpr *GoParser::Element() {
  GoASTExpr *key;
  if (!((key = Expression()) || (key = LiteralValue())))
    return nullptr;
  if (!match(GoLexer::OP_COLON))
    return key;
  GoASTExpr *value;
  if ((value = Expression()) || (value = LiteralValue()))
    return new GoASTKeyValueExpr(key, value);
  delete key;
  return syntaxerror();
}

GoASTExpr *GoParser::Selector(GoASTExpr *e) {
  Rule r("Selector", this);
  if (match(GoLexer::OP_DOT)) {
    if (auto *name = Identifier())
      return new GoASTSelectorExpr(e, name);
  }
  return r.error();
}

GoASTExpr *GoParser::IndexOrSlice(GoASTExpr *e) {
  Rule r("IndexOrSlice", this);
  if (match(GoLexer::OP_LBRACK)) {
    std::unique_ptr<GoASTExpr> i1(Expression()), i2, i3;
    bool slice = false;
    if (match(GoLexer::OP_COLON)) {
      slice = true;
      i2.reset(Expression());
      if (i2 && match(GoLexer::OP_COLON)) {
        i3.reset(Expression());
        if (!i3)
          return syntaxerror();
      }
    }
    if (!(slice || i1))
      return syntaxerror();
    if (!mustMatch(GoLexer::OP_RBRACK))
      return nullptr;
    if (slice) {
      bool slice3 = i3.get();
      return new GoASTSliceExpr(e, i1.release(), i2.release(), i3.release(),
                                slice3);
    }
    return new GoASTIndexExpr(e, i1.release());
  }
  return r.error();
}

GoASTExpr *GoParser::TypeAssertion(GoASTExpr *e) {
  Rule r("TypeAssertion", this);
  if (match(GoLexer::OP_DOT) && match(GoLexer::OP_LPAREN)) {
    if (auto *t = Type()) {
      if (!mustMatch(GoLexer::OP_RPAREN))
        return nullptr;
      return new GoASTTypeAssertExpr(e, t);
    }
    return syntaxerror();
  }
  return r.error();
}

GoASTExpr *GoParser::Arguments(GoASTExpr *e) {
  if (match(GoLexer::OP_LPAREN)) {
    std::unique_ptr<GoASTCallExpr> call(new GoASTCallExpr(false));
    GoASTExpr *arg;
    // ( ExpressionList | Type [ "," ExpressionList ] )
    for ((arg = Expression()) || (arg = Type()); arg;
         arg = MoreExpressionList()) {
      call->AddArgs(arg);
    }
    if (match(GoLexer::OP_DOTS))
      call->SetEllipsis(true);

    // Eat trailing comma
    match(GoLexer::OP_COMMA);

    if (!mustMatch(GoLexer::OP_RPAREN))
      return nullptr;
    call->SetFun(e);
    return call.release();
  }
  return nullptr;
}

GoASTExpr *GoParser::Conversion() {
  Rule r("Conversion", this);
  if (GoASTExpr *t = Type2()) {
    if (match(GoLexer::OP_LPAREN)) {
      GoASTExpr *v = Expression();
      if (!v)
        return syntaxerror();
      match(GoLexer::OP_COMMA);
      if (!mustMatch(GoLexer::OP_RPAREN))
        return r.error();
      GoASTCallExpr *call = new GoASTCallExpr(false);
      call->SetFun(t);
      call->AddArgs(v);
      return call;
    }
  }
  return r.error();
}

GoASTExpr *GoParser::Type2() {
  switch (peek()) {
  case GoLexer::OP_LBRACK:
    return ArrayOrSliceType(false);
  case GoLexer::KEYWORD_STRUCT:
    return StructType();
  case GoLexer::KEYWORD_FUNC:
    return FunctionType();
  case GoLexer::KEYWORD_INTERFACE:
    return InterfaceType();
  case GoLexer::KEYWORD_MAP:
    return MapType();
  case GoLexer::KEYWORD_CHAN:
    return ChanType2();
  default:
    return nullptr;
  }
}

GoASTExpr *GoParser::ArrayOrSliceType(bool allowEllipsis) {
  Rule r("ArrayType", this);
  if (match(GoLexer::OP_LBRACK)) {
    std::unique_ptr<GoASTExpr> len;
    if (allowEllipsis && match(GoLexer::OP_DOTS)) {
      len.reset(new GoASTEllipsis(nullptr));
    } else {
      len.reset(Expression());
    }

    if (!match(GoLexer::OP_RBRACK))
      return r.error();
    GoASTExpr *elem = Type();
    if (!elem)
      return syntaxerror();
    return new GoASTArrayType(len.release(), elem);
  }
  return r.error();
}

GoASTExpr *GoParser::StructType() {
  if (!(match(GoLexer::KEYWORD_STRUCT) && mustMatch(GoLexer::OP_LBRACE)))
    return nullptr;
  std::unique_ptr<GoASTFieldList> fields(new GoASTFieldList);
  while (auto *field = FieldDecl())
    fields->AddList(field);
  if (!mustMatch(GoLexer::OP_RBRACE))
    return nullptr;
  return new GoASTStructType(fields.release());
}

GoASTField *GoParser::FieldDecl() {
  std::unique_ptr<GoASTField> f(new GoASTField);
  GoASTExpr *t = FieldNamesAndType(f.get());
  if (!t)
    t = AnonymousFieldType();
  if (!t)
    return nullptr;

  if (auto *tok = match(GoLexer::LIT_STRING))
    f->SetTag(new GoASTBasicLit(*tok));
  if (!Semicolon())
    return syntaxerror();
  return f.release();
}

GoASTExpr *GoParser::FieldNamesAndType(GoASTField *field) {
  Rule r("FieldNames", this);
  for (auto *id = Identifier(); id; id = MoreIdentifierList())
    field->AddNames(id);
  if (m_failed)
    return nullptr;
  GoASTExpr *t = Type();
  if (t)
    return t;
  return r.error();
}

GoASTExpr *GoParser::AnonymousFieldType() {
  bool pointer = match(GoLexer::OP_STAR);
  GoASTExpr *t = Type();
  if (!t)
    return nullptr;
  if (pointer)
    return new GoASTStarExpr(t);
  return t;
}

GoASTExpr *GoParser::FunctionType() {
  if (!match(GoLexer::KEYWORD_FUNC))
    return nullptr;
  return Signature();
}

GoASTFuncType *GoParser::Signature() {
  auto *params = Params();
  if (!params)
    return syntaxerror();
  auto *result = Params();
  if (!result) {
    if (auto *t = Type()) {
      result = new GoASTFieldList;
      auto *f = new GoASTField;
      f->SetType(t);
      result->AddList(f);
    }
  }
  return new GoASTFuncType(params, result);
}

GoASTFieldList *GoParser::Params() {
  if (!match(GoLexer::OP_LPAREN))
    return nullptr;
  std::unique_ptr<GoASTFieldList> l(new GoASTFieldList);
  while (GoASTField *p = ParamDecl()) {
    l->AddList(p);
    if (!match(GoLexer::OP_COMMA))
      break;
  }
  if (!mustMatch(GoLexer::OP_RPAREN))
    return nullptr;
  return l.release();
}

GoASTField *GoParser::ParamDecl() {
  std::unique_ptr<GoASTField> field(new GoASTField);
  GoASTIdent *id = Identifier();
  if (id) {
    // Try `IdentifierList [ "..." ] Type`.
    // If that fails, backtrack and try `[ "..." ] Type`.
    Rule r("NamedParam", this);
    for (; id; id = MoreIdentifierList())
      field->AddNames(id);
    GoASTExpr *t = ParamType();
    if (t) {
      field->SetType(t);
      return field.release();
    }
    field.reset(new GoASTField);
    r.error();
  }
  GoASTExpr *t = ParamType();
  if (t) {
    field->SetType(t);
    return field.release();
  }
  return nullptr;
}

GoASTExpr *GoParser::ParamType() {
  bool dots = match(GoLexer::OP_DOTS);
  GoASTExpr *t = Type();
  if (!dots)
    return t;
  if (!t)
    return syntaxerror();
  return new GoASTEllipsis(t);
}

GoASTExpr *GoParser::InterfaceType() {
  if (!match(GoLexer::KEYWORD_INTERFACE) || !mustMatch(GoLexer::OP_LBRACE))
    return nullptr;
  std::unique_ptr<GoASTFieldList> methods(new GoASTFieldList);
  while (true) {
    Rule r("MethodSpec", this);
    // ( identifier Signature | TypeName ) ;
    std::unique_ptr<GoASTIdent> id(Identifier());
    if (!id)
      break;
    GoASTExpr *type = Signature();
    if (!type) {
      r.error();
      id.reset();
      type = Name();
    }
    if (!Semicolon())
      return syntaxerror();
    auto *f = new GoASTField;
    if (id)
      f->AddNames(id.release());
    f->SetType(type);
    methods->AddList(f);
  }
  if (!mustMatch(GoLexer::OP_RBRACE))
    return nullptr;
  return new GoASTInterfaceType(methods.release());
}

GoASTExpr *GoParser::MapType() {
  if (!(match(GoLexer::KEYWORD_MAP) && mustMatch(GoLexer::OP_LBRACK)))
    return nullptr;
  std::unique_ptr<GoASTExpr> key(Type());
  if (!key)
    return syntaxerror();
  if (!mustMatch(GoLexer::OP_RBRACK))
    return nullptr;
  auto *elem = Type();
  if (!elem)
    return syntaxerror();
  return new GoASTMapType(key.release(), elem);
}

GoASTExpr *GoParser::ChanType() {
  Rule r("chan", this);
  if (match(GoLexer::OP_LT_MINUS)) {
    if (match(GoLexer::KEYWORD_CHAN)) {
      auto *elem = Type();
      if (!elem)
        return syntaxerror();
      return new GoASTChanType(GoASTNode::eChanRecv, elem);
    }
    return r.error();
  }
  return ChanType2();
}

GoASTExpr *GoParser::ChanType2() {
  if (!match(GoLexer::KEYWORD_CHAN))
    return nullptr;
  auto dir = GoASTNode::eChanBidir;
  if (match(GoLexer::OP_LT_MINUS))
    dir = GoASTNode::eChanSend;
  auto *elem = Type();
  if (!elem)
    return syntaxerror();
  return new GoASTChanType(dir, elem);
}

GoASTExpr *GoParser::Type() {
  if (GoASTExpr *t = Type2())
    return t;
  if (GoASTExpr *t = Name())
    return t;
  if (GoASTExpr *t = ChanType())
    return t;
  if (match(GoLexer::OP_STAR)) {
    GoASTExpr *t = Type();
    if (!t)
      return syntaxerror();
    return new GoASTStarExpr(t);
  }
  if (match(GoLexer::OP_LPAREN)) {
    std::unique_ptr<GoASTExpr> t(Type());
    if (!t || !match(GoLexer::OP_RPAREN))
      return syntaxerror();
    return t.release();
  }
  return nullptr;
}

bool GoParser::Semicolon() {
  if (match(GoLexer::OP_SEMICOLON))
    return true;
  switch (peek()) {
  case GoLexer::OP_RPAREN:
  case GoLexer::OP_RBRACE:
  case GoLexer::TOK_EOF:
    return true;
  default:
    return false;
  }
}

GoASTExpr *GoParser::Name() {
  if (auto *id = Identifier()) {
    if (GoASTExpr *qual = QualifiedIdent(id))
      return qual;
    return id;
  }
  return nullptr;
}

GoASTExpr *GoParser::QualifiedIdent(lldb_private::GoASTIdent *p) {
  Rule r("QualifiedIdent", this);
  llvm::SmallString<32> path(p->GetName().m_value);
  GoLexer::Token *next;
  bool have_slashes = false;
  // LLDB extension: support full/package/path.name
  while (match(GoLexer::OP_SLASH) && (next = match(GoLexer::TOK_IDENTIFIER))) {
    have_slashes = true;
    path.append("/");
    path.append(next->m_value);
  }
  if (match(GoLexer::OP_DOT)) {
    auto *name = Identifier();
    if (name) {
      if (have_slashes) {
        p->SetName(GoLexer::Token(GoLexer::TOK_IDENTIFIER, CopyString(path)));
      }
      return new GoASTSelectorExpr(p, name);
    }
  }
  return r.error();
}

llvm::StringRef GoParser::CopyString(llvm::StringRef s) {
  return m_strings.insert(std::make_pair(s, 'x')).first->getKey();
}

void GoParser::GetError(Status &error) {
  llvm::StringRef want;
  if (m_failed)
    want =
        m_last_tok == GoLexer::TOK_INVALID ? DescribeToken(m_last_tok) : m_last;
  else
    want = m_error;
  size_t len = m_lexer.BytesRemaining();
  if (len > 10)
    len = 10;
  llvm::StringRef got;
  if (len == 0)
    got = "<eof>";
  else
    got = m_lexer.GetString(len);
  error.SetErrorStringWithFormat("Syntax error: expected %s before '%s'.",
                                 want.str().c_str(), got.str().c_str());
}
