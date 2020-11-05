//===- mlir-linalg-ods-gen.cpp - Linalg ODS generation from math form -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for the Tensor Comprehension-inspired
// parser and ODS pretty-printer for specifying Linalg "named ops" from a
// mathematical form.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "linalg-ods-gen"

static llvm::cl::OptionCategory ODSGenCat("Linalg ODS Gen");

// Commandline options
static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("filename"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool>
    genODSDecl("gen-ods-decl", llvm::cl::desc("Emit the ODS ops declarations."),
               llvm::cl::cat(ODSGenCat));

static llvm::cl::opt<bool>
    genODSImpl("gen-impl", llvm::cl::desc("Emit the ops implementations"),
               llvm::cl::init(false), llvm::cl::cat(ODSGenCat));

static llvm::cl::opt<bool> testEmitIncludeTdHeader(
    "test-emit-include-td-header",
    llvm::cl::desc("Include LinalgStructuredOps.td for end-to-end "
                   "tblgen testing."),
    llvm::cl::init(false), llvm::cl::cat(ODSGenCat));

using llvm::SetVector;
using llvm::SMLoc;
using llvm::StringRef;
using llvm::Twine;

using namespace mlir;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a specific token in the input format.
class Token {
public:
  enum class Kind {
    // Markers.
    eof,
    error,

    // Tokens with no info.
    colon,
    comma,
    equal,
    gt,
    l_brace,
    l_paren,
    lt,
    minus,
    plus,
    r_brace,
    r_paren,
    semicolon,
    star,

    // Keywords.
    kw_def,
    FIRST_KEYWORD = kw_def,
    kw_ods_def,
    kw_floordiv,
    kw_ceildiv,
    kw_mod,
    LAST_KEYWORD = kw_mod,

    // String valued tokens.
    id,
    integer,
  };

  Token(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  /// Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  /// Return the kind of this token.
  Kind getKind() const { return kind; }

  /// Return a location for this token.
  llvm::SMLoc getLoc() const {
    return llvm::SMLoc::getFromPointer(spelling.data());
  }

  /// Return if this token is a keyword.
  bool isKeyword() const {
    return kind >= Kind::FIRST_KEYWORD && kind <= Kind::LAST_KEYWORD;
  }
  bool is(Kind k) const { return kind == k; }
  bool isNot(Kind k) const { return kind != k; }

  Optional<uint64_t> getUInt64IntegerValue() const {
    bool isHex = spelling.size() > 1 && spelling[1] == 'x';

    uint64_t result = 0;
    if (spelling.getAsInteger(isHex ? 0 : 10, result))
      return None;
    return result;
  }

private:
  /// Discriminator that indicates the kind of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

/// This class implements a simple lexer.
class Lexer {
public:
  Lexer(llvm::SourceMgr &mgr);

  /// Lex the next token and return it.
  Token lexToken();

  /// Emit an error to the lexer with the given location and message.
  Token emitError(llvm::SMLoc loc, const Twine &msg);
  Token emitError(const char *loc, const Twine &msg);

private:
  Token formToken(Token::Kind kind, const char *tokStart) {
    return Token(kind, StringRef(tokStart, curPtr - tokStart));
  }

  /// Return the next character in the stream.
  int getNextChar();

  /// Lex an identifier.
  Token lexIdentifier(const char *tokStart);

  // Lex an integer.
  Token lexInteger(const char *tokStart);

  // Skip a comment line, starting with a '//'.
  void skipComment();

  llvm::SourceMgr &srcMgr;
  StringRef curBuffer;
  const char *curPtr;
};
} // end anonymous namespace

Lexer::Lexer(llvm::SourceMgr &mgr) : srcMgr(mgr) {
  curBuffer = srcMgr.getMemoryBuffer(mgr.getMainFileID())->getBuffer();
  curPtr = curBuffer.begin();
}

Token Lexer::emitError(llvm::SMLoc loc, const Twine &msg) {
  srcMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
  return formToken(Token::Kind::error, loc.getPointer());
}
Token Lexer::emitError(const char *loc, const Twine &msg) {
  return emitError(llvm::SMLoc::getFromPointer(loc), msg);
}

int Lexer::getNextChar() {
  char curChar = *curPtr++;
  switch (curChar) {
  default:
    return (unsigned char)curChar;
  case 0: {
    // A nul character in the stream is either the end of the current buffer
    // or a random nul in the file. Disambiguate that here.
    if (curPtr - 1 != curBuffer.end())
      return 0;

    // Otherwise, return end of file.
    --curPtr;
    return EOF;
  }
  case '\n':
  case '\r':
    // Handle the newline character by ignoring it and incrementing the line
    // count. However, be careful about 'dos style' files with \n\r in them.
    // Only treat a \n\r or \r\n as a single line.
    if ((*curPtr == '\n' || (*curPtr == '\r')) && *curPtr != curChar)
      ++curPtr;
    return '\n';
  }
}

Token Lexer::lexToken() {
  while (true) {
    const char *tokStart = curPtr;

    // This always consumes at least one character.
    int curChar = getNextChar();
    switch (curChar) {
    default:
      // Handle identifiers: [a-zA-Z_]
      if (isalpha(curChar) || curChar == '_')
        return lexIdentifier(tokStart);

      // Handle integers: [0-9]
      if (isdigit(curChar))
        return lexInteger(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case EOF:
      // Return EOF denoting the end of lexing.
      return formToken(Token::Kind::eof, tokStart);

    // Lex punctuation.
    case ':':
      return formToken(Token::Kind::colon, tokStart);
    case ',':
      return formToken(Token::Kind::comma, tokStart);
    case '=':
      return formToken(Token::Kind::equal, tokStart);
    case '{':
      return formToken(Token::Kind::l_brace, tokStart);
    case '(':
      return formToken(Token::Kind::l_paren, tokStart);
    case '}':
      return formToken(Token::Kind::r_brace, tokStart);
    case ')':
      return formToken(Token::Kind::r_paren, tokStart);
    case '<':
      return formToken(Token::Kind::lt, tokStart);
    case '>':
      return formToken(Token::Kind::gt, tokStart);
    case '+':
      return formToken(Token::Kind::plus, tokStart);
    case '-':
      return formToken(Token::Kind::minus, tokStart);
    case ';':
      return formToken(Token::Kind::semicolon, tokStart);
    case '*':
      return formToken(Token::Kind::star, tokStart);
    case '/':
      if (*curPtr == '/') {
        skipComment();
        continue;
      }
      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character: not a comment");

    // Ignore whitespace characters.
    case 0:
    case ' ':
    case '\t':
    case '\n':
      return lexToken();
    }
  }
}

Token Lexer::lexIdentifier(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_\-]*
  while (isalnum(*curPtr) || *curPtr == '_' || *curPtr == '-')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef str(tokStart, curPtr - tokStart);
  Token::Kind kind = StringSwitch<Token::Kind>(str)
                         .Case("def", Token::Kind::kw_def)
                         .Case("ods_def", Token::Kind::kw_ods_def)
                         .Case("floordiv", Token::Kind::kw_floordiv)
                         .Case("ceildiv", Token::Kind::kw_ceildiv)
                         .Case("mod", Token::Kind::kw_mod)
                         .Default(Token::Kind::id);

  return Token(kind, str);
}

Token Lexer::lexInteger(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_\-]*
  while (isdigit(*curPtr))
    ++curPtr;

  StringRef str(tokStart, curPtr - tokStart);
  return Token(Token::Kind::integer, str);
}

/// Skip a comment line, starting with a '//'.
void Lexer::skipComment() {
  // Advance over the second '/' in a '//' comment.
  assert(*curPtr == '/');
  ++curPtr;

  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      LLVM_FALLTHROUGH;
    default:
      // Skip over other characters.
      break;
    }
  }
}

namespace {

class Parser {
public:
  Parser(llvm::SourceMgr &mgr, MLIRContext *ctx)
      : lexer(mgr), curToken(lexer.lexToken()), context(ctx) {}

  //===--------------------------------------------------------------------===//
  // Lexer Utilities
  //===--------------------------------------------------------------------===//

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(curToken.getKind() != Token::Kind::eof &&
           curToken.getKind() != Token::Kind::error &&
           "shouldn't advance past EOF or errors");
    curToken = lexer.lexToken();
  }
  void consumeToken(Token::Kind kind) {
    assert(curToken.getKind() == kind && "unexpected token");
    curToken = lexer.lexToken();
  }
  LogicalResult parseToken(Token::Kind kind, const Twine &msg) {
    if (curToken.getKind() != kind)
      return emitError(curToken.getLoc(), msg);
    consumeToken();
    return success();
  }
  LogicalResult emitError(llvm::SMLoc loc, const Twine &msg) {
    lexer.emitError(loc, msg);
    return failure();
  }
  LogicalResult emitError(const Twine &msg) {
    return emitError(curToken.getLoc(), msg);
  }
  bool consumeIf(Token::Kind kind) {
    if (curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }
  LogicalResult
  parseCommaSeparatedList(llvm::function_ref<ParseResult()> parseElement) {
    // Non-empty case starts with an element.
    if (parseElement())
      return failure();

    // Otherwise we have a list of comma separated elements.
    while (consumeIf(Token::Kind::comma)) {
      if (parseElement())
        return failure();
    }
    return success();
  }
  LogicalResult
  parseCommaSeparatedListUntil(Token::Kind rightToken,
                               llvm::function_ref<ParseResult()> parseElement,
                               bool allowEmptyList) {
    // Handle the empty case.
    if (curToken.is(rightToken)) {
      if (!allowEmptyList)
        return emitError("expected list element");
      consumeToken(rightToken);
      return success();
    }

    if (failed(parseCommaSeparatedList(parseElement)) ||
        failed(
            parseToken(rightToken, "expected ',' or right-terminating token")))
      return failure();

    return success();
  }

  Lexer lexer;
  Token curToken;
  MLIRContext *context;
};
} // namespace

//===----------------------------------------------------------------------===//
// Affine parsing.
//===----------------------------------------------------------------------===//

namespace {

/// Lower precedence ops (all at the same precedence level). LNoOp is false in
/// the boolean sense.
enum AffineLowPrecOp {
  /// Null value.
  LNoOp,
  Add,
  Sub
};

/// Higher precedence ops - all at the same precedence level. HNoOp is false
/// in the boolean sense.
enum AffineHighPrecOp {
  /// Null value.
  HNoOp,
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

using AffineDimList = SmallVector<std::pair<StringRef, AffineExpr>, 4>;
using AffineSymbolList = SmallVector<std::pair<StringRef, AffineExpr>, 4>;

/// This is a specialized parser for affine expressions.
class AffineParser {
public:
  explicit AffineParser(Parser &p,
                        std::function<AffineExpr(StringRef)> bareIdParsingHook,
                        AffineDimList &dimList, AffineSymbolList &symbolList)
      : parser(p), bareIdFallback(bareIdParsingHook), dims(dimList),
        symbols(symbolList) {}

  /// Parse a comma-separated list of affine exprs.
  SmallVector<AffineExpr, 4>
  parseAffineExprs(Token::Kind lDelim = Token::Kind::l_paren,
                   Token::Kind rDelim = Token::Kind::r_paren);

  /// Parse a single affine expr.`.
  AffineExpr parseAffineExpr();

private:
  // Binary affine op parsing.
  AffineLowPrecOp consumeIfLowPrecOp();
  AffineHighPrecOp consumeIfHighPrecOp();

  // AffineExpr parsing.
  AffineExpr parseParentheticalExpr();
  AffineExpr parseNegateExpression(AffineExpr lhs);
  AffineExpr parseIntegerExpr();
  AffineExpr parseBareIdExpr();

  AffineExpr getAffineBinaryOpExpr(AffineHighPrecOp op, AffineExpr lhs,
                                   AffineExpr rhs, SMLoc opLoc);
  AffineExpr getAffineBinaryOpExpr(AffineLowPrecOp op, AffineExpr lhs,
                                   AffineExpr rhs);
  AffineExpr parseAffineOperandExpr(AffineExpr lhs);
  AffineExpr parseAffineLowPrecOpExpr(AffineExpr llhs, AffineLowPrecOp llhsOp);
  AffineExpr parseAffineHighPrecOpExpr(AffineExpr llhs, AffineHighPrecOp llhsOp,
                                       SMLoc llhsOpLoc);

  Parser &parser;
  std::function<AffineExpr(StringRef)> bareIdFallback;
  AffineDimList &dims;
  AffineSymbolList &symbols;
};
} // end anonymous namespace

/// Create an affine binary high precedence op expression (mul's, div's, mod).
/// opLoc is the location of the op token to be used to report errors
/// for non-conforming expressions.
AffineExpr AffineParser::getAffineBinaryOpExpr(AffineHighPrecOp op,
                                               AffineExpr lhs, AffineExpr rhs,
                                               SMLoc opLoc) {
  switch (op) {
  case Mul:
    if (!lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant()) {
      parser.emitError(opLoc,
                       "non-affine expression: at least one of the multiply "
                       "operands has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs * rhs;
  case FloorDiv:
    if (!rhs.isSymbolicOrConstant()) {
      parser.emitError(opLoc,
                       "non-affine expression: right operand of floordiv "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.floorDiv(rhs);
  case CeilDiv:
    if (!rhs.isSymbolicOrConstant()) {
      parser.emitError(opLoc, "non-affine expression: right operand of ceildiv "
                              "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.ceilDiv(rhs);
  case Mod:
    if (!rhs.isSymbolicOrConstant()) {
      parser.emitError(opLoc, "non-affine expression: right operand of mod "
                              "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs % rhs;
  case HNoOp:
    llvm_unreachable("can't create affine expression for null high prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineHighPrecOp");
}

/// Create an affine binary low precedence op expression (add, sub).
AffineExpr AffineParser::getAffineBinaryOpExpr(AffineLowPrecOp op,
                                               AffineExpr lhs, AffineExpr rhs) {
  switch (op) {
  case AffineLowPrecOp::Add:
    return lhs + rhs;
  case AffineLowPrecOp::Sub:
    return lhs - rhs;
  case AffineLowPrecOp::LNoOp:
    llvm_unreachable("can't create affine expression for null low prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineLowPrecOp");
}

/// Consume this token if it is a lower precedence affine op (there are only
/// two precedence levels).
AffineLowPrecOp AffineParser::consumeIfLowPrecOp() {
  switch (parser.curToken.getKind()) {
  case Token::Kind::plus:
    parser.consumeToken();
    return AffineLowPrecOp::Add;
  case Token::Kind::minus:
    parser.consumeToken();
    return AffineLowPrecOp::Sub;
  default:
    return AffineLowPrecOp::LNoOp;
  }
}

/// Consume this token if it is a higher precedence affine op (there are only
/// two precedence levels)
AffineHighPrecOp AffineParser::consumeIfHighPrecOp() {
  switch (parser.curToken.getKind()) {
  case Token::Kind::star:
    parser.consumeToken(Token::Kind::star);
    return Mul;
  case Token::Kind::kw_floordiv:
    parser.consumeToken(Token::Kind::kw_floordiv);
    return FloorDiv;
  case Token::Kind::kw_ceildiv:
    parser.consumeToken(Token::Kind::kw_ceildiv);
    return CeilDiv;
  case Token::Kind::kw_mod:
    parser.consumeToken(Token::Kind::kw_mod);
    return Mod;
  default:
    return HNoOp;
  }
}

/// Parse a high precedence op expression list: mul, div, and mod are high
/// precedence binary ops, i.e., parse a
///   expr_1 op_1 expr_2 op_2 ... expr_n
/// where op_1, op_2 are all a AffineHighPrecOp (mul, div, mod).
/// All affine binary ops are left associative.
/// Given llhs, returns (llhs llhsOp lhs) op rhs, or (lhs op rhs) if llhs is
/// null. If no rhs can be found, returns (llhs llhsOp lhs) or lhs if llhs is
/// null. llhsOpLoc is the location of the llhsOp token that will be used to
/// report an error for non-conforming expressions.
AffineExpr AffineParser::parseAffineHighPrecOpExpr(AffineExpr llhs,
                                                   AffineHighPrecOp llhsOp,
                                                   SMLoc llhsOpLoc) {
  AffineExpr lhs = parseAffineOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  auto opLoc = parser.curToken.getLoc();
  if (AffineHighPrecOp op = consumeIfHighPrecOp()) {
    if (llhs) {
      AffineExpr expr = getAffineBinaryOpExpr(llhsOp, llhs, lhs, opLoc);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(expr, op, opLoc);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(lhs, op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, llhs, lhs, llhsOpLoc);

  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an affine expression inside parentheses.
///
///   affine-expr ::= `(` affine-expr `)`
AffineExpr AffineParser::parseParentheticalExpr() {
  if (failed(parser.parseToken(Token::Kind::l_paren, "expected '('")))
    return nullptr;
  if (parser.curToken.is(Token::Kind::r_paren))
    return (parser.emitError("no expression inside parentheses"), nullptr);

  auto expr = parseAffineExpr();
  if (!expr)
    return nullptr;
  if (failed(parser.parseToken(Token::Kind::r_paren, "expected ')'")))
    return nullptr;

  return expr;
}

/// Parse the negation expression.
///
///   affine-expr ::= `-` affine-expr
AffineExpr AffineParser::parseNegateExpression(AffineExpr lhs) {
  if (failed(parser.parseToken(Token::Kind::minus, "expected '-'")))
    return nullptr;

  AffineExpr operand = parseAffineOperandExpr(lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseAffineOperandExpr instead of parseAffineExpr here.
  if (!operand)
    // Extra error message although parseAffineOperandExpr would have
    // complained. Leads to a better diagnostic.
    return (parser.emitError("missing operand of negation"), nullptr);
  return (-1) * operand;
}

/// Parse a bare id that may appear in an affine expression.
///
///   affine-expr ::= bare-id
AffineExpr AffineParser::parseBareIdExpr() {
  if (parser.curToken.isNot(Token::Kind::id))
    return (parser.emitError("expected id"), nullptr);

  StringRef sRef = parser.curToken.getSpelling();
  for (auto &list : {dims, symbols}) {
    for (auto entry : list) {
      if (entry.first == sRef) {
        parser.consumeToken(Token::Kind::id);
        return entry.second;
      }
    }
  }

  // Not found, check fallback path.
  AffineExpr expr = bareIdFallback(sRef);
  if (expr) {
    parser.consumeToken(Token::Kind::id);
    return expr;
  }

  return (parser.emitError("use of undeclared id"), nullptr);
}

/// Parse a positive integral constant appearing in an affine expression.
///
///   affine-expr ::= integer-literal
AffineExpr AffineParser::parseIntegerExpr() {
  auto val = parser.curToken.getUInt64IntegerValue();
  if (!val.hasValue() || (int64_t)val.getValue() < 0)
    return (parser.emitError("constant too large for index"), nullptr);

  parser.consumeToken(Token::Kind::integer);
  return getAffineConstantExpr((int64_t)val.getValue(), parser.context);
}

/// Parses an expression that can be a valid operand of an affine expression.
/// lhs: if non-null, lhs is an affine expression that is the lhs of a binary
/// operator, the rhs of which is being parsed. This is used to determine
/// whether an error should be emitted for a missing right operand.
//  Eg: for an expression without parentheses (like i + j + k + l), each
//  of the four identifiers is an operand. For i + j*k + l, j*k is not an
//  operand expression, it's an op expression and will be parsed via
//  parseAffineHighPrecOpExpression(). However, for i + (j*k) + -l, (j*k) and
//  -l are valid operands that will be parsed by this function.
AffineExpr AffineParser::parseAffineOperandExpr(AffineExpr lhs) {
  switch (parser.curToken.getKind()) {
  case Token::Kind::id:
    return parseBareIdExpr();
  case Token::Kind::integer:
    return parseIntegerExpr();
  case Token::Kind::l_paren:
    return parseParentheticalExpr();
  case Token::Kind::minus:
    return parseNegateExpression(lhs);
  case Token::Kind::kw_ceildiv:
  case Token::Kind::kw_floordiv:
  case Token::Kind::kw_mod:
  case Token::Kind::plus:
  case Token::Kind::star:
    if (lhs)
      parser.emitError("missing right operand of binary operator");
    else
      parser.emitError("missing left operand of binary operator");
    return nullptr;
  default:
    if (lhs)
      parser.emitError("missing right operand of binary operator");
    else
      parser.emitError("expected affine expression");
    return nullptr;
  }
}

/// Parse affine expressions that are bare-id's, integer constants,
/// parenthetical affine expressions, and affine op expressions that are a
/// composition of those.
///
/// All binary op's associate from left to right.
///
/// {add, sub} have lower precedence than {mul, div, and mod}.
///
/// Add, sub'are themselves at the same precedence level. Mul, floordiv,
/// ceildiv, and mod are at the same higher precedence level. Negation has
/// higher precedence than any binary op.
///
/// llhs: the affine expression appearing on the left of the one being parsed.
/// This function will return ((llhs llhsOp lhs) op rhs) if llhs is non null,
/// and lhs op rhs otherwise; if there is no rhs, llhs llhsOp lhs is returned
/// if llhs is non-null; otherwise lhs is returned. This is to deal with left
/// associativity.
///
/// Eg: when the expression is e1 + e2*e3 + e4, with e1 as llhs, this function
/// will return the affine expr equivalent of (e1 + (e2*e3)) + e4, where
/// (e2*e3) will be parsed using parseAffineHighPrecOpExpr().
AffineExpr AffineParser::parseAffineLowPrecOpExpr(AffineExpr llhs,
                                                  AffineLowPrecOp llhsOp) {
  AffineExpr lhs;
  if (!(lhs = parseAffineOperandExpr(llhs)))
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (AffineLowPrecOp lOp = consumeIfLowPrecOp()) {
    if (llhs) {
      AffineExpr sum = getAffineBinaryOpExpr(llhsOp, llhs, lhs);
      return parseAffineLowPrecOpExpr(sum, lOp);
    }
    // No LLHS, get RHS and form the expression.
    return parseAffineLowPrecOpExpr(lhs, lOp);
  }
  auto opLoc = parser.curToken.getLoc();
  if (AffineHighPrecOp hOp = consumeIfHighPrecOp()) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    AffineExpr highRes = parseAffineHighPrecOpExpr(lhs, hOp, opLoc);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    AffineExpr expr =
        llhs ? getAffineBinaryOpExpr(llhsOp, llhs, highRes) : highRes;

    // Recurse for subsequent low prec op's after the affine high prec op
    // expression.
    if (AffineLowPrecOp nextOp = consumeIfLowPrecOp())
      return parseAffineLowPrecOpExpr(expr, nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, llhs, lhs);
  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an affine expression.
///  affine-expr ::= `(` affine-expr `)`
///                | `-` affine-expr
///                | affine-expr `+` affine-expr
///                | affine-expr `-` affine-expr
///                | affine-expr `*` affine-expr
///                | affine-expr `floordiv` affine-expr
///                | affine-expr `ceildiv` affine-expr
///                | affine-expr `mod` affine-expr
///                | bare-id
///                | integer-literal
///
/// Additional conditions are checked depending on the production. For eg.,
/// one of the operands for `*` has to be either constant/symbolic; the second
/// operand for floordiv, ceildiv, and mod has to be a positive integer.
AffineExpr AffineParser::parseAffineExpr() {
  return parseAffineLowPrecOpExpr(nullptr, AffineLowPrecOp::LNoOp);
}

SmallVector<AffineExpr, 4> AffineParser::parseAffineExprs(Token::Kind lDelim,
                                                          Token::Kind rDelim) {
  parser.parseToken(lDelim, "expected lDelim at start of affine expr list");

  SmallVector<AffineExpr, 4> exprs;
  auto parseElt = [&]() -> LogicalResult {
    auto elt = parseAffineExpr();
    exprs.push_back(elt);
    return elt ? success() : failure();
  };

  if (failed(parser.parseCommaSeparatedListUntil(rDelim, parseElt,
                                                 /*allowEmptyList=*/true)))
    llvm_unreachable("Failed AffineExpr parsing");

  return exprs;
}

//===----------------------------------------------------------------------===//
// TC parsing.
//===----------------------------------------------------------------------===//

namespace {

/// Base class for expressions involved in TC parsing.
struct Expression {
  enum class Kind {
    Uninitialized = 0,
    TensorExpr = 1,
    TensorUse = 2,
  };

  explicit Expression(Kind k = Kind::Uninitialized) : kind(k) {}
  virtual ~Expression() = default;

  operator bool() const { return kind != Kind::Uninitialized; }

  Kind kind;
};

/// Encodes a tensor use of the form:
///
///   affine-expr-list ::= affine-expr (`,` affine-expr)*
///   tensor-use ::= bare-id `(` `)`
///                | bare-id `(` affine-expr-list `)`
///
/// The affine-expr-list is stored as an AffineMap.
struct TensorUse : public Expression {
  TensorUse() : TensorUse("", AffineMap()) {}
  TensorUse(StringRef name, AffineMap map)
      : Expression(Kind::TensorUse), tensorId(name), indexingMap(map) {}
  TensorUse(const TensorUse &use) = default;

  static bool classof(const Expression *e) {
    return e->kind == Kind::TensorUse;
  }

  bool operator==(const TensorUse &other) const {
    return tensorId == other.tensorId && indexingMap == other.indexingMap;
  }

  /// Visitation function. Performs preorder or postorder traversal depending on
  /// `PreOrder` and applies `callback` on each node.
  template <typename Lambda, bool PreOrder>
  void visit(Lambda callback) const;

  StringRef tensorId;
  AffineMap indexingMap;
};

/// Encodes a tensor expression of the form:
///
///   op-spec ::= bare-id `<` reduction-dims-list `>`
///             | bare-id
///   op-arg ::= tensor-expr
///            | tensor-use
///   op-arg-list ::= op-arg (`,` op-arg)*
///   tensor-expr ::= op-spec `(` op-arg-list `)`
///
/// Underlying op-arg are stored by unique_ptr to base class.
struct TensorExpr : public Expression {
  TensorExpr(StringRef name,
             SmallVectorImpl<std::unique_ptr<Expression>> &&exprs,
             ArrayRef<unsigned> reductionDims)
      : Expression(Kind::TensorExpr), operationName(name),
        expressions(std::move(exprs)),
        reductionDimensions(reductionDims.begin(), reductionDims.end()) {}

  static bool classof(const Expression *e) {
    return e->kind == Kind::TensorExpr;
  }

  bool operator==(const TensorExpr &other) const {
    if (operationName != other.operationName)
      return false;
    if (expressions.size() != other.expressions.size())
      return false;
    for (unsigned i = 0, e = expressions.size(); i < e; ++i)
      if (*expressions[i] != *other.expressions[i])
        return false;
    for (unsigned i = 0, e = reductionDimensions.size(); i < e; ++i)
      if (reductionDimensions[i] != other.reductionDimensions[i])
        return false;
    return true;
  }

  /// Visitation function. Performs preorder or postorder traversal depending on
  /// `PreOrder` and applies `callback` on each node.
  template <typename Lambda, bool PreOrder>
  void visit(Lambda callback) const;

  StringRef operationName;
  SmallVector<std::unique_ptr<Expression>, 4> expressions;
  SetVector<unsigned> reductionDimensions;
};

/// This is a specialized parser for a TCDef.
/// This maintains the dims it finds in an eager fashion.
class TCParser {
  enum class EagerDiscoveryMode { None = 0, Symbols, Dimensions };

public:
  explicit TCParser(Parser &p);

  /// Uses the AffineParser to parse the affine exprs used in a tensor
  /// definition. If `discoveryMode` is set to Symbols (resp. Dimensions), new
  /// symbols (resp. dimensions) are added eagerly. Otherwise, an error is
  /// emitted on new identifiers.
  SmallVector<AffineExpr, 4>
  parseAffineExprs(EagerDiscoveryMode discoveryMode, AffineDimList &dims,
                   Token::Kind lDelim = Token::Kind::l_paren,
                   Token::Kind rDelim = Token::Kind::r_paren);

  /// Parse the information for a tensor def.
  /// All the affine-expr must be dimensionless (i.e. contain only expressions
  /// involving symbols and constants), but can otherwise contain arbitrary
  /// affine expressions.
  LogicalResult parseTensorDef(bool isOutput);

  /// Parses a tensor use.
  struct ComprehensionParsingState {
    AffineDimList dims;
    SmallVector<std::unique_ptr<Expression>, 4> expressions;
    llvm::DenseMap<TensorUse, unsigned> orderedTensorArgs;
  };
  LogicalResult parseTensorUse(TensorUse &result,
                               ComprehensionParsingState &state);

  /// Parses a tensor expression.
  LogicalResult parseExpression(TensorUse currentDefinition,
                                std::unique_ptr<Expression> &result,
                                ComprehensionParsingState &state);

  /// Parse a single comprehension.
  LogicalResult parseOneComprehension(StringRef cppOpName,
                                      StringRef linalgOpName,
                                      ComprehensionParsingState &state);

  /// Parse and print the information for a TC def.
  /// When `gen-ods-decl` is used, this prints the ODS declaration for the TC.
  /// When `gen-impl` is used, this prints the C++ implementation for the extra
  /// methods defined in ODS (`iterator_types`, `indexing_maps` and
  /// `regionBuilder`).
  LogicalResult parseAndEmitODSDef(llvm::raw_ostream &os);

  /// Print the ODS class that defines a new `cppOpName` for a `linalgOpName`.
  void printODS(llvm::raw_ostream &os, StringRef cppOpName,
                StringRef linalgOpName, ComprehensionParsingState &state);

  /// Print the C++ StructuredOpsInterface impl of `iterator_types`.
  void printReferenceIterators(llvm::raw_ostream &os, StringRef cppOpName,
                               ComprehensionParsingState &state);

  /// Print the C++ StructuredOpsInterface impl of `indexing_maps`.
  void printReferenceIndexingMaps(llvm::raw_ostream &os, StringRef cppOpName,
                                  ComprehensionParsingState &state);

  /// Print the C++ StructuredOpsInterface impl of `regionBuilder`.
  void printRegionBuilder(llvm::raw_ostream &os, StringRef cppOpName,
                          ComprehensionParsingState &state);

  /// Print the C++ impl for named ops canonicalizers and fodlers.
  void printCanonicalizersAndFolders(llvm::raw_ostream &os,
                                     StringRef cppOpName);

private:
  //===--------------------------------------------------------------------===//
  // Internal bookkeeping of tensors.
  //===--------------------------------------------------------------------===//
  struct RegisteredTensor {
    StringRef type;
    AffineMap shape;
    bool isOutput;
    AffineMap indexingMap;
    unsigned index;
  };

  //===--------------------------------------------------------------------===//
  // Per-TC def state.
  //===--------------------------------------------------------------------===//
  /// Symbols are per TC def.
  AffineSymbolList symbols;
  /// Tensors are per TC def.
  llvm::StringMap<RegisteredTensor> registeredTensors;
  unsigned nextRegisteredTensorIndex;

  Parser &parser;
};
} // namespace

namespace llvm {

template <>
struct DenseMapInfo<TensorUse> {
  static TensorUse getEmptyKey() { return TensorUse("", AffineMap()); }
  static TensorUse getTombstoneKey() {
    return TensorUse(DenseMapInfo<StringRef>::getTombstoneKey(),
                     DenseMapInfo<AffineMap>::getTombstoneKey());
  }
  static unsigned getHashValue(const TensorUse &val) {
    return ::llvm::hash_value(val.tensorId); // don't care about collisions.
  }
  static bool isEqual(const TensorUse &LHS, const TensorUse &RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
// Visitation functions.
//===----------------------------------------------------------------------===//

template <typename Lambda, bool PreOrder>
void visit(const Expression &expr, Lambda callback) {
  switch (expr.kind) {
  default:
    llvm_unreachable("Unexpected kind");
  case Expression::Kind::TensorExpr:
    static_cast<const TensorExpr &>(expr).visit<Lambda, PreOrder>(callback);
    break;
  case Expression::Kind::TensorUse:
    static_cast<const TensorUse &>(expr).visit<Lambda, PreOrder>(callback);
    break;
  }
}

template <typename Lambda>
void visitPreorder(const Expression &expr, Lambda callback) {
  visit<Lambda, false>(expr, callback);
}

template <typename Lambda>
void visitPostorder(Expression &expr, Lambda callback) {
  visit<Lambda, true>(expr, callback);
}

template <typename Lambda, bool PreOrder>
void TensorExpr::visit(Lambda callback) const {
  if (!PreOrder)
    callback(*this);
  for (auto &e : expressions)
    ::visit<Lambda, PreOrder>(*e, callback);
  if (PreOrder)
    callback(*this);
}

template <typename Lambda, bool PreOrder>
void TensorUse::visit(Lambda callback) const {
  callback(*this);
}

//===----------------------------------------------------------------------===//
// TC parsing functions.
//===----------------------------------------------------------------------===//
TCParser::TCParser(Parser &p)
    : symbols(), registeredTensors(), nextRegisteredTensorIndex(0), parser(p) {}

/// Uses the AffineParser to parse the affine exprs used in a tensor
/// definition. All identifiers are interpreted as symbols, new symbols are
/// added eagerly.
SmallVector<AffineExpr, 4>
TCParser::parseAffineExprs(EagerDiscoveryMode discoveryMode,
                           AffineDimList &dims, Token::Kind lDelim,
                           Token::Kind rDelim) {
  AffineParser affineParser(
      parser,
      [&](StringRef sRef) {
        AffineExpr expr;
        if (discoveryMode == EagerDiscoveryMode::Symbols) {
          expr = getAffineSymbolExpr(symbols.size(), parser.context);
          symbols.emplace_back(sRef, expr);
        } else if (discoveryMode == EagerDiscoveryMode::Dimensions) {
          expr = getAffineDimExpr(dims.size(), parser.context);
          dims.emplace_back(sRef, expr);
        }
        return expr;
      },
      dims, symbols);
  return affineParser.parseAffineExprs(lDelim, rDelim);
}

/// Parse the information for a tensor def of the form:
///
///   affine-expr-list ::= affine-expr (`,` affine-expr )*
///   tensor-typedef ::= type `(` `)`
///                    | type `(` affine-expr-list `)`
///   tensor-def ::= bare-id `:` tensor-typedef
LogicalResult TCParser::parseTensorDef(bool isOutput) {
  StringRef tensorId = parser.curToken.getSpelling();
  if (failed(parser.parseToken(Token::Kind::id, "expected an id")) ||
      failed(parser.parseToken(Token::Kind::colon, "expected colon")))
    return failure();

  StringRef tensorType = parser.curToken.getSpelling();
  if (failed(parser.parseToken(Token::Kind::id, "expected an id")))
    return failure();

  AffineDimList emptyDims;
  auto exprs = parseAffineExprs(EagerDiscoveryMode::Symbols, emptyDims);
  assert(emptyDims.empty() && "Unexpected dimension in tensor def");
  AffineMap map =
      AffineMap::get(/*dimCount=*/0, symbols.size(), exprs, parser.context);

  auto iterBoolPair = registeredTensors.try_emplace(
      tensorId, RegisteredTensor{tensorType, map, isOutput, AffineMap(),
                                 nextRegisteredTensorIndex++});
  (void)iterBoolPair;
  assert(iterBoolPair.second && "Could not emplace tensor registration");
  LLVM_DEBUG(llvm::dbgs() << "Recorded: " << tensorId << " "
                          << "with typeString: " << tensorType << " "
                          << "and shape: " << map << "\n");

  return success();
}

/// Parses a tensor use of the form:
///
///   affine-expr-list ::= affine-expr (`,` affine-expr)*
///   tensor-use ::= bare-id `(` `)`
///                | bare-id `(` affine-expr-list `)`
LogicalResult TCParser::parseTensorUse(TensorUse &result,
                                       ComprehensionParsingState &state) {
  StringRef tensorId = parser.curToken.getSpelling();
  if (failed(parser.parseToken(Token::Kind::id, "expected an id")))
    return failure();

  auto exprs = parseAffineExprs(EagerDiscoveryMode::Dimensions, state.dims);
  AffineMap map =
      AffineMap::get(state.dims.size(), symbols.size(), exprs, parser.context);
  LLVM_DEBUG(llvm::dbgs() << "Use of tensor: " << tensorId << " map: " << map
                          << "\n");

  result = TensorUse(tensorId, map);
  return success();
}

/// Parses a tensor expression of the form:
///
///   op-spec ::= bare-id `<` reduction-dims-list `>`
///             | bare-id
///   op-arg ::= tensor-expr
///            | tensor-use
///   op-arg-list ::= op-arg (`,` op-arg)*
///   tensor-expr ::= op-spec `(` op-arg-list `)`
LogicalResult TCParser::parseExpression(TensorUse currentDefinition,
                                        std::unique_ptr<Expression> &result,
                                        ComprehensionParsingState &state) {
  StringRef opOrTensor = parser.curToken.getSpelling();
  if (registeredTensors.count(opOrTensor) > 0) {
    TensorUse use;
    auto res = parseTensorUse(use, state);
    if (failed(res))
      return res;
    result = std::make_unique<TensorUse>(use);
    return success();
  }

  if (failed(parser.parseToken(Token::Kind::id, "expected an operation")))
    return failure();

  // This is an op.
  SmallVector<unsigned, 4> reductionDims;
  SmallVector<std::unique_ptr<Expression>, 4> expressions;

  // Check if it has a reduction set, discover dimensions eagerly.
  if (parser.curToken.is(Token::Kind::lt)) {
    auto iters = parseAffineExprs(EagerDiscoveryMode::Dimensions, state.dims,
                                  Token::Kind::lt, Token::Kind::gt);
    for (auto iter : iters)
      reductionDims.push_back(iter.cast<AffineDimExpr>().getPosition());
  }

  // If this op is a reduction, it's first argument is the `currentDefinition`
  // tensor use.
  if (!reductionDims.empty())
    expressions.push_back(std::make_unique<TensorUse>(currentDefinition));
  LLVM_DEBUG(llvm::dbgs() << "op: " << opOrTensor << "\n");

  auto parseExpr = [&]() -> LogicalResult {
    std::unique_ptr<Expression> e;
    if (failed(parseExpression(currentDefinition, e, state)))
      return failure();
    expressions.push_back(std::move(e));
    return success();
  };
  if (failed(parser.parseToken(Token::Kind::l_paren, "expected '('")) ||
      failed(parser.parseCommaSeparatedListUntil(
          Token::Kind::r_paren, parseExpr, /*allowEmptyList=*/true)))
    return failure();

  result = std::make_unique<TensorExpr>(opOrTensor, std::move(expressions),
                                        reductionDims);

  return success();
}

//===----------------------------------------------------------------------===//
// Parse and Emit functions.
//===----------------------------------------------------------------------===//

/// Parse the information for a single comprehension.
///
///   tensor-def-list ::= tensor-def (`,` tensor-def)*
///   tensor-expr-list ::= tensor-expr (`,` tensor-expr)*
///   comprehension ::= tensor-def-list `=` tensor-expr-list `;`
LogicalResult
TCParser::parseOneComprehension(StringRef cppOpName, StringRef linalgOpName,
                                ComprehensionParsingState &state) {
  // 1. Parse LHS of `=`, these become the definitions that appear as the output
  // tensors or read/write buffers.
  SmallVector<TensorUse, 4> definitions;
  auto parseUse = [&]() -> LogicalResult {
    TensorUse use;
    if (failed(parseTensorUse(use, state)))
      return failure();
    definitions.push_back(use);
    return success();
  };
  if (failed(parser.parseCommaSeparatedListUntil(Token::Kind::equal, parseUse,
                                                 /*allowEmptyList=*/true)))
    return failure();

  // 2. Parse RHS of `=`, this becomes the expressions from which we emit
  // computations.
  unsigned idx = 0;
  auto parseExpr = [&]() -> LogicalResult {
    std::unique_ptr<Expression> expr;
    if (idx >= definitions.size()) {
      parser.emitError("Fewer LHS definitions than RHS expressions");
      return failure();
    }
    if (failed(parseExpression(definitions[idx++], expr, state)))
      return failure();
    state.expressions.push_back(std::move(expr));
    return success();
  };
  if (failed(parser.parseCommaSeparatedListUntil(
          Token::Kind::semicolon, parseExpr, /*allowEmptyList=*/true)))
    return failure();
  if (idx != definitions.size()) {
    parser.emitError("Fewer RHS expressions than LHS definitions");
    return failure();
  }

  // 3. Postprocess.
  // 3.a. Normalize all maps to the proper state.dims and symbols counts.
  SmallVector<TensorUse, 4> allUses;
  allUses.reserve(registeredTensors.size());
  for (auto &def : definitions)
    allUses.push_back(def);
  for (auto &pExpr : state.expressions)
    visitPostorder(*pExpr, [&](const Expression &e) {
      if (auto *use = dyn_cast<TensorUse>(&e))
        allUses.push_back(*use);
    });
  for (auto &use : allUses)
    use.indexingMap =
        AffineMap::get(state.dims.size(), symbols.size(),
                       use.indexingMap.getResults(), parser.context);

  // 3.b. Traverse definitions
  llvm::DenseSet<StringRef> seenDefs;
  for (auto &def : definitions) {
    if (seenDefs.count(def.tensorId) > 0) {
      parser.emitError("Unexpected multi-write to a single tensor");
      return failure();
    }
    seenDefs.insert(def.tensorId);
    auto tensorIter = registeredTensors.find(def.tensorId);
    assert(tensorIter != registeredTensors.end() && "unregistered tensor");
    auto &tensor = tensorIter->getValue();
    tensor.indexingMap = def.indexingMap;
    state.orderedTensorArgs[def] = tensor.index;
  }

  bool failed = false;
  for (auto &pExpr : state.expressions)
    visitPostorder(*pExpr, [&](const Expression &e) {
      auto *pUse = dyn_cast<TensorUse>(&e);
      if (failed || !pUse)
        return;
      auto &use = *pUse;
      LLVM_DEBUG(llvm::dbgs()
                 << "\nuse: " << use.tensorId << " map: " << use.indexingMap);
      auto tensorIter = registeredTensors.find(use.tensorId);
      assert(tensorIter != registeredTensors.end() && "unregistered tensor");
      auto &tensor = tensorIter->getValue();
      if (tensor.indexingMap && state.orderedTensorArgs.count(use) == 0) {
        LLVM_DEBUG(llvm::dbgs() << "\nexisting: " << tensor.indexingMap);
        parser.emitError(
            "Unexpected multi-read of a tensor with different accesses");
        failed = true;
        return;
      }
      seenDefs.insert(use.tensorId);
      tensor.indexingMap = use.indexingMap;
      state.orderedTensorArgs[use] = tensor.index;
    });
  if (failed)
    return failure();

  return success();
}

/// Parse and print the information for a ODS def.
///
///   tensor-def-list ::= tensor-def (`,` tensor-def )*
///
///   comprehension-list ::= comprehension comprehension*
///
///   tc-def ::= `def` bare-id `(`tensor-def-list`)` `->` `(` tensor-def-list`)`
///     `{` comprehension-list `}`
///
///   ods-def ::= `ods_def` `<` bare-id `>` `:` tc-def
///
/// All the affine-expr in a `tensor-typedef` must be dimensionless (i.e.
/// contain only expressions involving symbols and constants), but can
/// otherwise contain arbitrary affine expressions.
LogicalResult TCParser::parseAndEmitODSDef(llvm::raw_ostream &os) {
  if (failed(parser.parseToken(Token::Kind::kw_ods_def,
                               "expected 'ods_def' to define a TC ODS")) ||
      failed(parser.parseToken(Token::Kind::lt, "expected '<'")))
    return failure();
  StringRef cppOpName = parser.curToken.getSpelling();
  LLVM_DEBUG(llvm::dbgs() << "\n\nStart parsing ODS: " << cppOpName << "\n");

  if (failed(parser.parseToken(Token::Kind::id, "expected id")) ||
      failed(parser.parseToken(Token::Kind::gt, "expected '>'")) ||
      failed(parser.parseToken(Token::Kind::colon, "expected ':'")))
    return failure();
  if (failed(parser.parseToken(Token::Kind::kw_def,
                               "expected 'def' to define a TC")))
    return failure();

  StringRef tcName = parser.curToken.getSpelling();
  LLVM_DEBUG(llvm::dbgs() << "\n\nStart parsing TC: " << tcName << "\n");
  if (failed(parser.parseToken(Token::Kind::id, "expected id")) ||
      failed(parser.parseToken(Token::Kind::l_paren, "expected '('")))
    return failure();

  auto parseInputDef = [&]() -> LogicalResult {
    return parseTensorDef(/*isOutput=*/false);
  };
  if (failed(parser.parseCommaSeparatedListUntil(
          Token::Kind::r_paren, parseInputDef, /*allowEmptyList=*/false)))
    return failure();

  if (failed(parser.parseToken(Token::Kind::minus, "expected '-'")) ||
      failed(parser.parseToken(Token::Kind::gt, "expected '>'")) ||
      failed(parser.parseToken(Token::Kind::l_paren, "expected '('")))
    return failure();
  auto parseOutputDef = [&]() -> LogicalResult {
    return parseTensorDef(/*isOutput=*/true);
  };
  if (failed(parser.parseCommaSeparatedListUntil(
          Token::Kind::r_paren, parseOutputDef, /*allowEmptyList=*/false)))
    return failure();

  // Since we don't declare symbols separately, we discover them eagerly: each
  // newly encountered id in a tensor shape expression is treated as a new
  // symbolic. At this point, all tensors have been parsed and all the symbols
  // that could be discovered eagerly are now known. Resize all AffineMaps to
  // normalize the number of eagerly discovered symbols.
  for (auto &tensor : registeredTensors) {
    auto &map = tensor.getValue().shape;
    map = AffineMap::get(/*dimCount=*/0, symbols.size(), map.getResults(),
                         parser.context);
  }

  if (failed(parser.parseToken(Token::Kind::l_brace, "expected '{'")))
    return failure();

  SmallVector<ComprehensionParsingState, 4> perComprehensionStates;
  while (parser.curToken.isNot(Token::Kind::r_brace)) {
    perComprehensionStates.push_back(ComprehensionParsingState());
    if (failed(parseOneComprehension(cppOpName, tcName,
                                     perComprehensionStates.back())))
      return failure();
  };
  parser.parseToken(Token::Kind::r_brace, "expected '}'");

  // Print.
  auto nComprehensions = perComprehensionStates.size();
  if (nComprehensions != 1) {
    parser.emitError("only 1 comprehension supported for now, got: " +
                     llvm::Twine(nComprehensions));
    return failure();
  }
  if (genODSDecl) {
    auto &state = perComprehensionStates.back();
    printODS(os, cppOpName, tcName, state);
    os << "\n";
  }
  if (genODSImpl) {
    auto &state = perComprehensionStates.back();
    std::string extraMethods;
    llvm::raw_string_ostream ss(extraMethods);
    printReferenceIterators(ss, cppOpName, state);
    printReferenceIndexingMaps(ss, cppOpName, state);
    printRegionBuilder(ss, cppOpName, state);
    printCanonicalizersAndFolders(ss, cppOpName);
    ss.flush();
    os << extraMethods << "\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Printing functions
//===----------------------------------------------------------------------===//

/// Print the ODS class that defines a new `cppOpName` for a `linalgOpName`.
void TCParser::printODS(llvm::raw_ostream &os, StringRef cppOpName,
                        StringRef linalgOpName,
                        ComprehensionParsingState &state) {
  const char *header = R"FMT(  def {0} : LinalgStructuredBase_Op<"{1}", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    NamedStructuredOpTrait,
    SingleBlockImplicitTerminator<"YieldOp">]> {
      let arguments = (ins Variadic<AnyShaped>:$inputs,
                           Variadic<AnyMemRef>:$output_buffers,
                           Variadic<AnyRankedTensor>:$init_tensors);
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [ OpBuilderDAG<
        (ins "ValueRange":$inputs, "ValueRange":$outputBuffers),
        [{{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({{
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<{0}>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        }]>, OpBuilderDAG<
        (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
             "ValueRange":$outputBuffers, "ValueRange":$initTensors),
        [{{
          $_state.addOperands(inputs);
          $_state.addOperands(outputBuffers);
          $_state.addOperands(initTensors);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({{
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<{0}>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        }]>, OpBuilderDAG<
        (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
             CArg<"ArrayRef<NamedAttribute>", "{{}">:$attributes),
        [{{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
      ];
      let printer = [{{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{{ return ::parseNamedStructuredOp<{0}>(parser, result); }];
      let verifier = [{{ return ::verifyNamedStructuredOp(*this); }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = [{{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() {{ return {4}; }
        std::string getLibraryCallName() {{
          return generateLibraryCallName(getOperation());
        }
      }];
  })FMT";

  unsigned nInputs = 0, nOutputs = 0;
  for (auto &t : registeredTensors) {
    if (t.getValue().isOutput)
      nOutputs++;
    else
      nInputs++;
  }

  os << llvm::formatv(header, cppOpName, linalgOpName, nInputs, nOutputs,
                      state.orderedTensorArgs.size());
}

/// Print the C++ StructuredOpsInterface impl of `iterator_types`.
void TCParser::printReferenceIterators(llvm::raw_ostream &os,
                                       StringRef cppOpName,
                                       ComprehensionParsingState &state) {
  const char *referenceReferenceIteratorsFmt =
      R"FMT(
    ArrayAttr {0}::iterator_types() {
      return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef, 8>{{ {1} });
    })FMT";

  std::string iteratorsStr;
  llvm::raw_string_ostream ss(iteratorsStr);
  unsigned pos = 0;
  llvm::interleaveComma(
      state.dims, ss, [&](std::pair<StringRef, AffineExpr> p) {
        bool reduction = false;
        for (auto &expr : state.expressions) {
          visitPostorder(*expr, [&](const Expression &e) {
            if (auto *pTensorExpr = dyn_cast<TensorExpr>(&e)) {
              if (pTensorExpr->reductionDimensions.count(pos) > 0)
                reduction = true;
            }
          });
          if (reduction)
            break;
        }
        ss << (reduction ? "getReductionIteratorTypeName()"
                         : "getParallelIteratorTypeName()");
        pos++;
      });
  ss.flush();

  os << llvm::formatv(referenceReferenceIteratorsFmt, cppOpName, iteratorsStr);
}

void TCParser::printCanonicalizersAndFolders(llvm::raw_ostream &os,
                                             StringRef cppOpName) {
  const char *canonicalizersAndFoldersFmt = R"FMT(
    void {0}::getCanonicalizationPatterns(
        OwningRewritePatternList &results,
        MLIRContext *context) {{
      results.insert<EraseDeadLinalgOp>();
      results.insert<FoldTensorCastOp>();
    }
    LogicalResult {0}::fold(ArrayRef<Attribute>,
                            SmallVectorImpl<OpFoldResult> &) {{
      return foldMemRefCast(*this);
    }
    void {0}::getEffects(SmallVectorImpl<
        SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {{
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
    })FMT";
  os << llvm::formatv(canonicalizersAndFoldersFmt, cppOpName);
}

/// Print the C++ StructuredOpsInterface impl of `referenceIndexingMaps`.
void TCParser::printReferenceIndexingMaps(llvm::raw_ostream &os,
                                          StringRef cppOpName,
                                          ComprehensionParsingState &state) {
  // 1. Generic string template for specifying reference indexing maps.
  const char *referenceIndexingMapsFmt =
      R"FMT(
  // This is temporary until we transition out of manually specified ops that
  // should be auto-generated with linalg-ods-gen.
  ArrayAttr {0}::indexing_maps() {
    MLIRContext *context = getContext();
    AffineExpr {1};
    bindDims(context, {1});
    return Builder(context).getAffineMapArrayAttr({ {2} });
  })FMT";

  // 2. Print a comma-separated list of identifiers for the AffineExpr in
  // `state.dims`. These will replace the `{1}` placeholder in both
  // `AffineExpr {1}` and `bindDims(context, {1})` ensuring the AffineExpr
  // identifiers are bound in the right order to the proper AffineDimExpr.
  std::string dimsStr;
  llvm::raw_string_ostream ss(dimsStr);
  llvm::interleaveComma(
      state.dims, ss,
      [&](std::pair<StringRef, AffineExpr> p) { ss << p.second; });
  ss.flush();

  // 3. Print a comma-separated list of AffineMap constructors that use the
  // identifiers from 1. The AffineExpr use the common arithmetic operators on
  // AffineExpr. These AffineMap constructors will replace the `{2}` placeholder
  // in return `SmallVector<AffineMap, 8>{{ {2} };`.
  std::string mapsStr;
  llvm::raw_string_ostream mapsStringStream(mapsStr);
  SmallVector<TensorUse, 4> orderedUses(state.orderedTensorArgs.size());
  for (const auto &it : state.orderedTensorArgs)
    orderedUses[it.second] = it.first;
  llvm::interleaveComma(orderedUses, mapsStringStream, [&](TensorUse u) {
    assert(u.indexingMap);
    const char *mapFmt = "\n\tAffineMap::get({0}, 0, {1}, context)";
    if (u.indexingMap.isEmpty()) {
      mapsStringStream << llvm::formatv(mapFmt, state.dims.size(), "context");
      return;
    }

    std::string exprsStr;
    llvm::raw_string_ostream exprsStringStream(exprsStr);
    exprsStringStream << "{";
    llvm::interleaveComma(u.indexingMap.getResults(), exprsStringStream);
    exprsStringStream << "}";
    exprsStringStream.flush();

    mapsStringStream << llvm::formatv(mapFmt, state.dims.size(), exprsStr);
  });
  mapsStringStream.flush();

  // 4. Apply format to 1. using 2. and 3.
  os << llvm::formatv(referenceIndexingMapsFmt, cppOpName, dimsStr, mapsStr);
}

/// Print the C++ StructuredOpsInterface impl of `regionBuilder`.
void TCParser::printRegionBuilder(llvm::raw_ostream &os, StringRef cppOpName,
                                  ComprehensionParsingState &state) {
  unsigned count = state.orderedTensorArgs.size();
  llvm::DenseMap<const TensorExpr *, unsigned> subExprsMap;
  std::function<void(llvm::raw_ostream & os, const Expression &)> printExpr;
  printExpr = [&](llvm::raw_ostream &os, const Expression &e) -> void {
    if (auto *pUse = dyn_cast<TensorUse>(&e)) {
      os << "_" << state.orderedTensorArgs.find(*pUse)->second;
      return;
    }
    auto *pTensorExpr = cast<TensorExpr>(&e);
    if (subExprsMap.count(pTensorExpr) > 0) {
      os << "_" << subExprsMap[pTensorExpr];
    } else {
      std::string subExprs;
      llvm::raw_string_ostream subExprsStringStream(subExprs);
      llvm::interleaveComma(pTensorExpr->expressions, subExprsStringStream,
                            [&](const std::unique_ptr<Expression> &e) {
                              printExpr(subExprsStringStream, *e);
                            });
      subExprsStringStream.flush();
      const char *tensorExprFmt = "\n    Value _{0} = {1}({2});";
      os << llvm::formatv(tensorExprFmt, ++count, pTensorExpr->operationName,
                          subExprs);
      subExprsMap[pTensorExpr] = count;
    }
  };

  const char *regionBuilderFmt = R"FMT(
  void {0}::regionBuilder(Block &block) {
    using namespace edsc;
    using namespace intrinsics;
    auto args = block.getArguments();
    Value {1};
    {2}
    (linalg_yield(ValueRange{ {3} }));
  })FMT";

  unsigned idx = 0;
  std::string valueHandleStr;
  llvm::raw_string_ostream valueHandleStringStream(valueHandleStr);
  llvm::interleaveComma(
      state.orderedTensorArgs, valueHandleStringStream, [&](auto) {
        valueHandleStringStream << "_" << idx << "(args[" << idx << "])";
        idx++;
      });

  std::string expressionsStr;
  llvm::raw_string_ostream expressionStringStream(expressionsStr);
  for (auto &expr : state.expressions)
    visitPostorder(*expr, [&](const Expression &e) {
      if (e.kind == Expression::Kind::TensorExpr)
        printExpr(expressionStringStream, e);
    });

  std::string yieldStr;
  llvm::raw_string_ostream yieldStringStream(yieldStr);
  llvm::interleaveComma(state.expressions, yieldStringStream,
                        [&](const std::unique_ptr<Expression> &e) {
                          printExpr(yieldStringStream, *e);
                        });

  valueHandleStringStream.flush();
  expressionStringStream.flush();
  yieldStringStream.flush();

  os << llvm::formatv(regionBuilderFmt, cppOpName, valueHandleStr,
                      expressionsStr, yieldStr);
}

/// Iterate over each Tensor Comprehension def.
LogicalResult parseAndEmitAllTensorComprehensions(llvm::raw_ostream &os,
                                                  Parser &parser) {
  while (parser.curToken.getKind() != Token::Kind::eof) {
    TCParser tcParser(parser);
    if (failed(tcParser.parseAndEmitODSDef(os)))
      return failure();
  }
  return success();
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Linalg ODS Gen");

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Include the proper Linalg header for end-to-end tblgen testing without
  // resorting to non-portable shell manipulations.
  if (testEmitIncludeTdHeader)
    output->os() << "include \"mlir/Dialect/Linalg/IR/LinalgStructuredOps.td\"";

  MLIRContext context;
  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  Parser parser(mgr, &context);
  parseAndEmitAllTensorComprehensions(output->os(), parser);
  output->keep();

  return 0;
}
