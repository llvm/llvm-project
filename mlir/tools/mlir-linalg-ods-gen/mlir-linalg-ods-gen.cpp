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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"

#include <map>

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
    doc_str,
    equal,
    gt,
    l_brace,
    l_paren,
    l_square,
    lt,
    minus,
    plus,
    question,
    r_brace,
    r_paren,
    r_square,
    semicolon,
    star,

    // Keywords.
    kw_def,
    FIRST_KEYWORD = kw_def,
    kw_ods_def,
    kw_implements_interface,
    kw_attr_def,
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

  /// Change the position of the lexer cursor. The next token we lex will start
  /// at the designated point in the input.
  void resetPointer(const char *newPtr) { curPtr = newPtr; }

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

  // Lex a string.
  Token lexString(const char *tokStart);

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
    case '[':
      return formToken(Token::Kind::l_square, tokStart);
    case '}':
      return formToken(Token::Kind::r_brace, tokStart);
    case ')':
      return formToken(Token::Kind::r_paren, tokStart);
    case ']':
      return formToken(Token::Kind::r_square, tokStart);
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
    case '?':
      return formToken(Token::Kind::question, tokStart);
    case '"':
      return lexString(tokStart);
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
  Token::Kind kind =
      StringSwitch<Token::Kind>(str)
          .Case("attr", Token::Kind::kw_attr_def)
          .Case("def", Token::Kind::kw_def)
          .Case("ods_def", Token::Kind::kw_ods_def)
          .Case("implements_interface", Token::Kind::kw_implements_interface)
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

Token Lexer::lexString(const char *tokStart) {
  assert(curPtr[-1] == '"');

  if (*curPtr == '"' && *(curPtr + 1) == '"') {
    curPtr += 2;
    while (true) {
      switch (*curPtr++) {
      case '"':
        if (*curPtr == '"' && *(curPtr + 1) == '"') {
          Token token(Token::Kind::doc_str,
                      StringRef(tokStart + 3, curPtr - tokStart - 4));
          curPtr += 2;
          return token;
        }
        continue;
      case 0:
        // If this is a random nul character in the middle of the doc string,
        // just include it.  If it is the end of file, then it is an error.
        if (curPtr - 1 != curBuffer.end())
          continue;
        return emitError(curPtr - 1, "expected '\"\"\"' to end doc string");
      default:
        continue;
      }
    }
  }

  return emitError(curPtr - 1, "expected '\"\"\"' to start doc string");
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

  LogicalResult parseInteger(uint64_t &value) {
    if (!curToken.is(Token::Kind::integer))
      return emitError(curToken.getLoc(), "expected integer");
    value = curToken.getUInt64IntegerValue().getValue();
    consumeToken();
    return success();
  }

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

  /// Parses an optional token and returns failure if failed to parse.
  LogicalResult parseOptionalToken(Token::Kind kind) {
    return success(consumeIf(kind));
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

/// Encodes an attribute use of the form:
///
///   index-list ::= integer-literal (`,` integer-literal)*
///   attr-use ::= bare-id `[` index-list `]`
struct AttrUse {
  // Referenced attribute
  StringRef attrName;
  // Indices into the attribute
  SmallVector<uint64_t, 4> indices;
  /// Affine symbol for this usage.
  /// This is represented as an affine symbol because at the time of parsing the
  /// spec and generating the op's ODS/C++, we don't know the concrete constant
  /// value. But they should be replaced with constants read from the attribute
  /// and thus folded away for concrete op instances.
  AffineExpr symbol;

  std::string getKey() {
    SmallVector<std::string, 4> indexStrs;
    for (uint64_t index : indices)
      indexStrs.push_back(std::to_string(index));
    return llvm::formatv("{0}[{1}]", attrName, llvm::join(indexStrs, ","));
  }
};

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
  /// Creates an affine parser that parses tokens from `p`.
  ///
  /// The affine parser introduces new dimensions and symbols eagerly as new
  /// `id` are discovered. To additionally support attribute use `id`s, for a
  /// parsed `id`, the resolution mechanism proceeds as follows:
  /// 1. Try to parse `id` as an attribute use (using the `attrUseParsingHook`).
  /// 2. If unsuccessful, try to match `id` to a known dim or symbol.
  /// 3. If still unsuccessful, eagerly create a new dim or symbol and add it to
  ///    the known dims or symbols (using the `bareIdParsingHook`).
  explicit AffineParser(
      Parser &p, std::function<AffineExpr(StringRef)> bareIdParsingHook,
      std::function<llvm::Optional<AffineExpr>()> attrUseParsingHook,
      AffineDimList &dimList, AffineSymbolList &symbolList)
      : parser(p), bareIdFallback(bareIdParsingHook),
        attrUseCallback(attrUseParsingHook), dims(dimList),
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
  AffineExpr parseAttrUseOrBareIdExpr();
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
  std::function<llvm::Optional<AffineExpr>()> attrUseCallback;
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
      (void)parser.emitError(
          opLoc, "non-affine expression: at least one of the multiply "
                 "operands has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs * rhs;
  case FloorDiv:
    if (!rhs.isSymbolicOrConstant()) {
      (void)parser.emitError(opLoc,
                             "non-affine expression: right operand of floordiv "
                             "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.floorDiv(rhs);
  case CeilDiv:
    if (!rhs.isSymbolicOrConstant()) {
      (void)parser.emitError(opLoc,
                             "non-affine expression: right operand of ceildiv "
                             "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.ceilDiv(rhs);
  case Mod:
    if (!rhs.isSymbolicOrConstant()) {
      (void)parser.emitError(opLoc,
                             "non-affine expression: right operand of mod "
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
    return ((void)parser.emitError("no expression inside parentheses"),
            nullptr);

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
    return ((void)parser.emitError("missing operand of negation"), nullptr);
  return (-1) * operand;
}

AffineExpr AffineParser::parseAttrUseOrBareIdExpr() {
  if (llvm::Optional<AffineExpr> attrUse = attrUseCallback())
    return attrUse.getValue();
  return parseBareIdExpr();
}

/// Parse a bare id that may appear in an affine expression.
///
///   affine-expr ::= bare-id
AffineExpr AffineParser::parseBareIdExpr() {
  if (parser.curToken.isNot(Token::Kind::id))
    return ((void)parser.emitError("expected id"), nullptr);

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

  return ((void)parser.emitError("use of undeclared id"), nullptr);
}

/// Parse a positive integral constant appearing in an affine expression.
///
///   affine-expr ::= integer-literal
AffineExpr AffineParser::parseIntegerExpr() {
  auto val = parser.curToken.getUInt64IntegerValue();
  if (!val.hasValue() || (int64_t)val.getValue() < 0)
    return ((void)parser.emitError("constant too large for index"), nullptr);

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
    return parseAttrUseOrBareIdExpr();
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
      (void)parser.emitError("missing right operand of binary operator");
    else
      (void)parser.emitError("missing left operand of binary operator");
    return nullptr;
  default:
    if (lhs)
      (void)parser.emitError("missing right operand of binary operator");
    else
      (void)parser.emitError("expected affine expression");
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
  if (failed(parser.parseToken(lDelim,
                               "expected lDelim at start of affine expr list")))
    return {};

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

  /// Parses an attribute definition.
  LogicalResult parseAttrDef();

  /// Parses an optional attribute use.
  LogicalResult parseAttrUse(AttrUse &result);

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
                StringRef linalgOpName, ArrayRef<StringRef> interfaces,
                ComprehensionParsingState &state);

  /// Print the C++ StructuredOpsInterface impl of `iterator_types`.
  void printReferenceIterators(llvm::raw_ostream &os, StringRef cppOpName,
                               ComprehensionParsingState &state);

  /// Print methods related to indexing map required attributes.
  ///
  /// Specifically, this prints the definitions for the following methods:
  ///   bool hasDynamicIndexingMaps();
  ///   LogicalResult verifyIndexingMapRequiredAttributes();
  void printIndexingMapRequiredAttrMethods(llvm::raw_ostream &os,
                                           StringRef cppOpName,
                                           ComprehensionParsingState &state);

  /// Print the C++ StructuredOpsInterface impl of `indexing_maps`.
  void printReferenceIndexingMaps(llvm::raw_ostream &os, StringRef cppOpName,
                                  ComprehensionParsingState &state);

  /// Print the C++ StructuredOpsInterface impl of `regionBuilder`.
  void printRegionBuilder(llvm::raw_ostream &os, StringRef cppOpName,
                          ComprehensionParsingState &state);

  /// Print the C++ impl for named ops canonicalizers and folders.
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
  // Internal bookkeeping of attributes.
  //===--------------------------------------------------------------------===//
  struct RegisteredAttr {
    StringRef elementType;
    SmallVector<uint64_t, 4> vectorDims;
    bool isArray;
    bool isOptional;

    // Returns the function to get values at the given indices from this
    // attribute.
    std::string getValueFn(ArrayRef<uint64_t> indices) const;
  };

  //===--------------------------------------------------------------------===//
  // Per-TC def state.
  //===--------------------------------------------------------------------===//
  /// Symbols are per TC def.
  AffineSymbolList symbols;

  /// Attribute usages in all affine expressions.
  SmallVector<AttrUse, 8> attrUses;

  /// Tensors are per TC def.
  llvm::StringMap<RegisteredTensor> registeredTensors;
  unsigned nextRegisteredTensorIndex;

  /// Attributes are per TC def.
  std::map<std::string, RegisteredAttr> registeredAttrs;

  StringRef docString;

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
  auto createAffineBareId = [&](StringRef sRef) {
    AffineExpr expr;
    if (discoveryMode == EagerDiscoveryMode::Symbols) {
      expr = getAffineSymbolExpr(symbols.size(), parser.context);
      symbols.emplace_back(sRef, expr);
    } else if (discoveryMode == EagerDiscoveryMode::Dimensions) {
      expr = getAffineDimExpr(dims.size(), parser.context);
      dims.emplace_back(sRef, expr);
    }
    return expr;
  };

  auto tryToParseAttrUse = [&]() -> llvm::Optional<AffineExpr> {
    if (!parser.curToken.is(Token::Kind::id))
      return llvm::None;

    StringRef attrName = parser.curToken.getSpelling();
    auto it = registeredAttrs.find(attrName.str());
    if (it == registeredAttrs.end())
      return llvm::None;

    AttrUse result;
    if (failed(parseAttrUse(result)))
      return llvm::None;

    // We create a new symbol for each attribute usage without reuse. This is
    // fine given these symbols will be replaced with constants and folded away
    // for concrete op instances.
    result.symbol = getAffineSymbolExpr(symbols.size(), parser.context);
    // Merely for taking the index. We don't reuse anyway.
    symbols.emplace_back("<attr-use>", result.symbol);

    attrUses.push_back(result);

    return result.symbol;
  };

  AffineParser affineParser(parser, createAffineBareId, tryToParseAttrUse, dims,
                            symbols);
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

/// Parse the information for an attribute def of the form:
///
///   affine-expr-list ::= affine-expr (`,` affine-expr )*
///   attr-id ::= bare-id (`?`)?
///   dim-list ::= (integer-literal 'x')+
///   attr-typedef ::= dim-list? type (`[` `]`)?
///   attr-def ::= attr-id `:` attr-typedef
LogicalResult TCParser::parseAttrDef() {
  auto attrLoc = parser.curToken.getLoc();
  StringRef attrName = parser.curToken.getSpelling();
  if (failed(parser.parseToken(Token::Kind::id, "expected an id")))
    return failure();
  bool isOptional = succeeded(parser.parseOptionalToken(Token::Kind::question));
  if (failed(parser.parseToken(Token::Kind::colon, "expected colon")))
    return failure();

  // Parse the attribute's type. We don't expect the type to be arbitrary
  // complex, so just use this ad-hoc handling here.

  // Parse potential dimension list
  SmallVector<uint64_t, 4> vectorDims;
  while (parser.curToken.is(Token::Kind::integer)) {
    uint64_t value;
    if (failed(parser.parseInteger(value)))
      return failure();
    vectorDims.push_back(value);

    StringRef spelling = parser.curToken.getSpelling();
    if (spelling[0] != 'x')
      return parser.emitError(parser.curToken.getLoc(),
                              "expected 'x' in dimension list");

    // If we had a prefix of 'x', lex the next token immediately after the 'x'.
    if (spelling.size() != 1)
      parser.lexer.resetPointer(spelling.data() + 1);

    parser.consumeToken();
  }

  StringRef elementType = parser.curToken.getSpelling();
  if (failed(parser.parseToken(Token::Kind::id, "expected an id")))
    return failure();

  bool isArray = false;
  auto arrayLoc = parser.curToken.getLoc();
  if (succeeded(parser.parseOptionalToken(Token::Kind::l_square))) {
    isArray = true;
    if (failed(parser.parseToken(Token::Kind::r_square, "expected ']'")))
      return failure();
  }

  if (!vectorDims.empty() && isArray)
    return parser.emitError(arrayLoc, "unsupported vector array attribute");

  auto iterBoolPair = registeredAttrs.emplace(
      attrName.str(),
      RegisteredAttr{elementType, vectorDims, isArray, isOptional});
  if (!iterBoolPair.second)
    return parser.emitError(attrLoc,
                            "Failed to register attribute '" + attrName + "'");

  LLVM_DEBUG(llvm::dbgs() << "Recorded: " << (isOptional ? "[optional]" : "")
                          << " " << attrName << " "
                          << "with type: " << elementType
                          << (isArray ? "[]" : "") << "\n");

  return success();
}

LogicalResult TCParser::parseAttrUse(AttrUse &result) {
  result.attrName = parser.curToken.getSpelling();
  if (failed(parser.parseToken(Token::Kind::id, "expected an id")))
    return failure();

  auto it = registeredAttrs.find(result.attrName.str());
  assert(it != registeredAttrs.end());
  const RegisteredAttr &attr = it->second;

  if (!attr.vectorDims.empty() || attr.isArray) {
    // This is a vector/array attribute. Parse indices for it.
    auto indexLoc = parser.curToken.getLoc();

    if (failed(parser.parseToken(Token::Kind::l_square, "expected '['")))
      return failure();

    auto parseIndex = [&]() {
      uint64_t value;
      if (failed(parser.parseInteger(value)))
        return failure();
      result.indices.push_back(value);
      return success();
    };
    if (failed(parser.parseCommaSeparatedListUntil(
            Token::Kind::r_square, parseIndex, /*allowEmptyList=*/false)))
      return failure();

    size_t rank = attr.isArray ? 1 : attr.vectorDims.size();
    if (result.indices.size() != rank)
      return parser.emitError(indexLoc,
                              "number of indices mismatch: expected " +
                                  std::to_string(rank) + ", but found " +
                                  std::to_string(result.indices.size()));
  }

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
    if (idx >= definitions.size())
      return parser.emitError("Fewer LHS definitions than RHS expressions");
    if (failed(parseExpression(definitions[idx++], expr, state)))
      return failure();
    state.expressions.push_back(std::move(expr));
    return success();
  };
  if (failed(parser.parseCommaSeparatedListUntil(
          Token::Kind::semicolon, parseExpr, /*allowEmptyList=*/true)))
    return failure();
  if (idx != definitions.size())
    return parser.emitError("Fewer RHS expressions than LHS definitions");

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
    if (seenDefs.count(def.tensorId) > 0)
      return parser.emitError("Unexpected multi-write to a single tensor");
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
        (void)parser.emitError(
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
///   attr-def-list ::= attr-def (`,` attr-def )*
///
///   comprehension-list ::= comprehension comprehension*
///
///   tc-attr-def ::= `attr` `(` attr-def-list `)`
///   tc-def ::= `def` bare-id `(`tensor-def-list`)` `->` `(` tensor-def-list`)`
///     (tc-attr-def)?
///     `{` comprehension-list `}`
///
///   implements-interface ::=
///     `implements_interface` `<` bare-id (`,` bare-id)* `>` `:` tc-def
///
///   ods-def ::= `ods_def` `<` bare-id `>`
///               (implements-interface)? `:`
///               tc-def
///
/// All the affine-expr in a `tensor-typedef` must be dimensionless (i.e.
/// contain only expressions involving symbols and constants), but can
/// otherwise contain arbitrary affine expressions.
LogicalResult TCParser::parseAndEmitODSDef(llvm::raw_ostream &os) {
  // Parse ods-def header (including C++ op name)
  if (failed(parser.parseToken(Token::Kind::kw_ods_def,
                               "expected 'ods_def' to define a TC ODS")) ||
      failed(parser.parseToken(Token::Kind::lt, "expected '<'")))
    return failure();
  StringRef cppOpName = parser.curToken.getSpelling();
  LLVM_DEBUG(llvm::dbgs() << "\n\nStart parsing ODS: " << cppOpName << "\n");
  if (failed(parser.parseToken(Token::Kind::id, "expected id")) ||
      failed(parser.parseToken(Token::Kind::gt, "expected '>'")))
    return failure();

  // Parse optional implements-interface header (including C++ op names)
  SmallVector<StringRef> interfaces;
  bool implementsInterface = succeeded(
      parser.parseOptionalToken(Token::Kind::kw_implements_interface));
  if (implementsInterface) {
    auto parseInterfaceString = [&]() -> LogicalResult {
      StringRef interfaceName = parser.curToken.getSpelling();
      if (failed(parser.parseToken(Token::Kind::id, "expected id")))
        return failure();
      interfaces.push_back(interfaceName);
      return success();
    };
    if (failed(parser.parseToken(Token::Kind::lt, "expected '<'")) ||
        failed(parser.parseCommaSeparatedListUntil(
            Token::Kind::gt, parseInterfaceString, /*allowEmptyList=*/false)))
      return failure();
  }

  // Parse column.
  if (failed(parser.parseToken(Token::Kind::colon, "expected ':'")))
    return failure();

  // Parse TC op name.
  if (failed(parser.parseToken(Token::Kind::kw_def,
                               "expected 'def' to define a TC")))
    return failure();
  StringRef tcName = parser.curToken.getSpelling();
  LLVM_DEBUG(llvm::dbgs() << "\n\nStart parsing TC: " << tcName << "\n");

  // Parse input/output tensor definitions
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

  // Parse optional attribute definitions
  if (succeeded(parser.parseOptionalToken(Token::Kind::kw_attr_def))) {
    if (failed(parser.parseToken(Token::Kind::l_paren, "expected '('")))
      return failure();
    if (failed(parser.parseCommaSeparatedListUntil(
            Token::Kind::r_paren, std::bind(&TCParser::parseAttrDef, this),
            /*allowEmptyList=*/false)))
      return failure();
  }

  // Parse optional doc string
  if (parser.curToken.is(Token::Kind::doc_str)) {
    docString = parser.curToken.getSpelling();
    parser.consumeToken();
    LLVM_DEBUG(llvm::dbgs()
               << "parsed doc string: '''" << docString << "'''\n");
  }

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
  if (failed(parser.parseToken(Token::Kind::r_brace, "expected '}'")))
    return failure();

  // Print.
  auto nComprehensions = perComprehensionStates.size();
  if (nComprehensions != 1)
    return parser.emitError("only 1 comprehension supported for now, got: " +
                            llvm::Twine(nComprehensions));
  if (genODSDecl) {
    auto &state = perComprehensionStates.back();
    printODS(os, cppOpName, tcName, interfaces, state);
    os << "\n";
  }
  if (genODSImpl) {
    auto &state = perComprehensionStates.back();
    std::string extraMethods;
    llvm::raw_string_ostream ss(extraMethods);
    printReferenceIterators(ss, cppOpName, state);
    printIndexingMapRequiredAttrMethods(ss, cppOpName, state);
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
                        StringRef linalgOpName, ArrayRef<StringRef> interfaces,
                        ComprehensionParsingState &state) {
  SmallVector<std::string, 4> attributes;
  for (const auto &attr : registeredAttrs) {
    llvm::StringRef name = attr.first;

    llvm::StringRef elementType = attr.second.elementType;
    std::string odsType = llvm::StringSwitch<std::string>(elementType)
                              .Case("f32", "F32")
                              .Case("i32", "I32")
                              .Case("i64", "I64")
                              .Default("");
    if (odsType.empty()) {
      (void)parser.emitError(
          "unimplemented support for attribute element type: " + elementType);
      return;
    }

    const auto &dims = attr.second.vectorDims;
    if (!dims.empty()) {
      SmallVector<std::string, 4> dimStrs;
      for (uint64_t dim : dims)
        dimStrs.push_back(std::to_string(dim));
      odsType = llvm::formatv("Ranked{0}ElementsAttr<[{1}]>", odsType,
                              llvm::join(dimStrs, ", "));
    }

    assert(dims.empty() || !attr.second.isArray);
    if (attr.second.isArray)
      odsType = llvm::formatv("{0}ArrayAttr", odsType);

    if (attr.second.isOptional)
      odsType = llvm::formatv("OptionalAttr<{0}>", odsType);

    attributes.push_back(llvm::formatv("{0}:${1}", odsType, name));
  }

  std::string attrList = llvm::join(attributes, ",\n");
  if (!attrList.empty())
    attrList = ",\n" + attrList;

  // Template for Linalg named ops' ODS definitions. Parameters:
  // {0}: ODS/C++ op name
  // {1}: assembly op mnemonic
  // {2}: op interface list
  // {3}: documentation (summary + description)
  // {4}: op attribute list
  // {5}: the number of arguments for the op region
  // {6}: builder methods taking standalone attribute parameters
  // {7}: additional methods for attributes used by indexing maps
  const char *header = R"FMT(  def {0} : LinalgStructuredBase_Op<"{1}", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"YieldOp">
    /*extraInterfaces=*/{2}]> {
      {3}
      let arguments = (ins
        Variadic<AnyShaped>:$inputs,
        Variadic<AnyShaped>:$outputs{4}
      );
      let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
      let regions = (region AnyRegion:$region);

      let skipDefaultBuilders = 1;
      let builders = [
        OpBuilderDAG<
        (ins "ValueRange":$inputs, "ValueRange":$outputs),
        [{{
          $_state.addOperands(inputs);
          $_state.addOperands(outputs);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({{
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<{0}>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        }]>,
        OpBuilderDAG<
        (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
             "ValueRange":$outputs),
        [{{
          $_state.addOperands(inputs);
          $_state.addOperands(outputs);
          $_state.addTypes(resultTensorTypes);
          $_state.addAttribute(
            "operand_segment_sizes",
            $_builder.getI32VectorAttr({{
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputs.size())}));
          createAndFillStructuredOpRegion<{0}>(
            $_builder,
            $_state,
            TypeRange(inputs),
            TypeRange(outputs)/*, TODO: support captures*/);
        }]>,
        OpBuilderDAG<
        (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
             CArg<"ArrayRef<NamedAttribute>", "{{}">:$attributes),
        [{{
          $_state.addOperands(operands);
          $_state.addAttributes(attributes);
          $_state.addTypes(resultTensorTypes);
          (void)$_state.addRegion();
        }]>
        {6}
      ];
      let printer = [{{ return ::printNamedStructuredOp(p, *this); }];
      let parser = [{{
        return ::parseNamedStructuredOp<{0}>(parser, result/*TODO:, captures*/);
      }];
      let hasFolder = 1;
      let hasCanonicalizer = 1;

      let extraClassDeclaration = structuredOpsBaseDecls # [{{
        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block, ValueRange captures);
        static std::function<void(Block &, ValueRange)> getRegionBuilder() {{
          return regionBuilder;
        }

        // Generic methods.
        static unsigned getNumRegionArgs() {{ return {5}; }
        std::string getLibraryCallName() {{
          return generateLibraryCallName(getOperation());
        }

        {7}
      }];
  })FMT";

  // Generate the list of extra implemented interfaces.
  std::string interfaceNameList;
  if (!interfaces.empty()) {
    llvm::raw_string_ostream ss(interfaceNameList);
    ss << ", "; // Leading comma to concat to existing list of interfaces.
    llvm::interleaveComma(interfaces, ss);
    ss.flush();
  }

  // Generate documentation.
  std::string doc;
  if (!docString.empty()) {
    const char *docFmt = R"FMT(
      let summary = [{ {0} }];
      let description = [{
        {1}
      }];
    )FMT";

    StringRef summary, description;
    std::tie(summary, description) = docString.trim().split('\n');
    doc = llvm::formatv(docFmt, summary.trim(), description.trim());
  }

  // Generate an additional builder that has parameters for attributes.
  std::string attrBuilder;
  if (!registeredAttrs.empty()) {
    SmallVector<std::string, 4> attrParams, attrStmts;
    for (const auto &attr : registeredAttrs) {
      llvm::StringRef name = attr.first;
      attrParams.push_back(llvm::formatv("\"Attribute\":${0}", name));
      attrStmts.push_back(
          llvm::formatv("$_state.addAttribute(\"{0}\", {0});", name));
    }
    std::string attrParamsList = llvm::join(attrParams, ", ");
    std::string attrStmtsList = llvm::join(attrStmts, "\n");

    const char *builderFmt = R"FMT(
      , OpBuilderDAG<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
           "ValueRange":$outputs, {1}),
      [{{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({{
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<{0}>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs)/*, TODO: support captures*/);
        {2}
      }]>
    )FMT";
    attrBuilder =
        llvm::formatv(builderFmt, cppOpName, attrParamsList, attrStmtsList);
  }

  std::string attrMethods;
  if (!registeredAttrs.empty()) {
    attrMethods = R"(
      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();
    )";
  }

  // Finally put everything together.
  os << llvm::formatv(header, cppOpName, linalgOpName, interfaceNameList, doc,
                      attrList, state.orderedTensorArgs.size(), attrBuilder,
                      attrMethods);
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

// Prints methods for querying whether the current named op has attributes that
// are used by its indexing maps and for verifying those attributes have the
// expected type.
void TCParser::printIndexingMapRequiredAttrMethods(
    llvm::raw_ostream &os, StringRef cppOpName,
    ComprehensionParsingState &state) {
  // If there are no attribute used by the whole definition, then we are done.
  if (registeredAttrs.empty())
    return;

  // Otherwise, go through each attribute and generate code to verify it's
  // valid per the spec.
  SmallVector<std::string, 4> attributes;
  for (const auto &attr : registeredAttrs) {
    if (attr.second.isOptional)
      continue;

    llvm::StringRef name = attr.first;
    llvm::StringRef elementType = attr.second.elementType;
    const auto &dims = attr.second.vectorDims;

    // Get the method call to check the element type is of the expected kind.
    std::string elemTypeCheck = llvm::StringSwitch<std::string>(elementType)
                                    .Case("f32", "isF32()")
                                    .Case("i32", "isInteger(32)")
                                    .Case("i64", "isInteger(64)")
                                    .Default("");
    if (elemTypeCheck.empty()) {
      (void)parser.emitError(
          "unimplemented support for attribute element type: " + elementType);
      return;
    }

    // Scalar case.
    if (dims.empty() && !attr.second.isArray) {
      const char *attrFmt = R"FMT(
        if (auto attr = op->getAttr("{0}")) {{
          if (!attr.getType().{1}) return op->emitError(
            "incorrect type for indexing map required attribute '{0}'");
        } else {{
          return op->emitError(
            "missing indexing map required attribute '{0}'");
        }
      )FMT";

      attributes.push_back(llvm::formatv(attrFmt, name, elemTypeCheck));
      continue;
    }

    // Vector case.
    if (!dims.empty()) {
      SmallVector<std::string, 4> dimStrs;
      for (uint64_t dim : dims)
        dimStrs.push_back(std::to_string(dim));

      const char *attrFmt = R"FMT(
        if (auto attr = op->getAttrOfType<DenseElementsAttr>("{0}")) {{
          if (!attr.getType().getElementType().{1}) return op->emitError(
            "incorrect element type for indexing map required attribute '{0}'");
          if (attr.getType().getShape() != ArrayRef<int64_t>{{ {2} })
            return op->emitError(
              "incorrect shape for indexing map required attribute '{0}'");
        } else {
          return op->emitError(
            "missing indexing map required attribute '{0}'");
        }
      )FMT";

      attributes.push_back(llvm::formatv(attrFmt, name, elemTypeCheck,
                                         llvm::join(dimStrs, ", ")));
      continue;
    }

    // Array case.
    {
      const char *attrFmt = R"FMT(
        if (auto attr = op->getAttrOfType<ArrayAttr>("{0}")) {{
          for (Attribute element : attr) {{
            if (!element.getType().{1}) return emitError(
              "incorrect element type for indexing map required attribute '{0}'");
          }
        } else {{
          return op->emitError(
            "missing indexing map required attribute '{0}'");
        }
      )FMT";

      attributes.push_back(llvm::formatv(attrFmt, name, elemTypeCheck));
    }
  }

  const char *methodFmt = R"FMT(
  bool {0}::hasDynamicIndexingMaps() {{ return true; }

  LogicalResult {0}::verifyIndexingMapRequiredAttributes() {{
    Operation *op = getOperation();
    {1}
    return success();
  }
  )FMT";

  // Print everything out.
  os << llvm::formatv(methodFmt, cppOpName, llvm::join(attributes, "\n"));
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
    {2}
    return Builder(context).getAffineMapArrayAttr({ {3} });
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

  // 3. Get the list of affine maps for each input/output. The AffineExpr use
  // the common arithmetic operators on AffineExpr. These affine maps will
  // replace the `{2}` placeholder.
  std::string mapsStr;
  llvm::raw_string_ostream mapsStringStream(mapsStr);

  SmallVector<TensorUse, 4> orderedUses(state.orderedTensorArgs.size());
  for (const auto &it : state.orderedTensorArgs)
    orderedUses[it.second] = it.first;

  // Create a list of all symbols.
  SmallVector<std::string, 4> symbolReplacements;
  symbolReplacements.reserve(symbols.size());
  for (unsigned i = 0; i < symbols.size(); ++i) {
    const char *symFmt =
        "\n\tauto s{0} = getAffineSymbolExpr({0}, context); (void)s{0};";
    mapsStringStream << llvm::formatv(symFmt, i);
    symbolReplacements.push_back(llvm::formatv("s{0}", i));
  }

  // Create the affine constant expressions to replace symbols for attributes.
  for (auto attrUse : llvm::enumerate(attrUses)) {
    StringRef attrName = attrUse.value().attrName;
    auto it = registeredAttrs.find(attrName.str());
    assert(it != registeredAttrs.end() && "uses should point to valid attr!");
    std::string getValueFn = it->second.getValueFn(attrUse.value().indices);
    if (getValueFn.empty()) {
      (void)parser.emitError("unimplemented getValueFn for attribute: " +
                             attrName);
      return;
    }
    std::string cstVal = llvm::formatv("{0}().{1}", attrName, getValueFn);
    const char *cstFmt =
        "\n\tauto cst{0} = getAffineConstantExpr({1}, context);";
    mapsStringStream << llvm::formatv(cstFmt, attrUse.index(), cstVal);

    unsigned position =
        attrUse.value().symbol.cast<AffineSymbolExpr>().getPosition();
    symbolReplacements[position] = llvm::formatv("cst{0}", attrUse.index());
  }

  // For each tensor use, construct the affine map, replace symbols by the
  // corresponding attribute values, and simplify the affine map.
  for (auto tensorUse : llvm::enumerate(orderedUses)) {
    auto indexingMap = tensorUse.value().indexingMap;
    const char *mapFmt =
        "\n\tauto map{0} = AffineMap::get({1}, {2}, {3}, context);";

    std::string exprsStr;
    llvm::raw_string_ostream exprsStringStream(exprsStr);
    exprsStringStream << "{";
    llvm::interleaveComma(indexingMap.getResults(), exprsStringStream);
    exprsStringStream << "}";
    exprsStringStream.flush();
    mapsStringStream << llvm::formatv(mapFmt, tensorUse.index(),
                                      state.dims.size(),
                                      indexingMap.getNumSymbols(), exprsStr);

    std::string replaceSymbolList =
        llvm::formatv("{ {0} }", llvm::join(symbolReplacements, ", "));

    // Note that we use `0` as the result affine map's number of symbols. All
    // symbols representing attribute usages should be folded away. But there
    // may exist additional symbols for tensor dimension upper bounds. Linalg
    // does not handle such cases right now. This needs to be fixed once we
    // need that.
    const char *replaceFmt =
        "\n\tmap{0} = map{0}.replaceDimsAndSymbols({{}, {1}, {2}, 0);";
    mapsStringStream << llvm::formatv(replaceFmt, tensorUse.index(),
                                      replaceSymbolList, state.dims.size());
    const char *simplifyFmt = "\n\tmap{0} = simplifyAffineMap(map{0});";
    mapsStringStream << llvm::formatv(simplifyFmt, tensorUse.index());
  }

  mapsStringStream.flush();

  SmallVector<std::string, 4> mapList;
  mapList.reserve(orderedUses.size());
  for (unsigned i = 0; i < orderedUses.size(); ++i)
    mapList.push_back(llvm::formatv("map{0}", i));

  // 4. Apply format to 1. using 2. and 3.
  os << llvm::formatv(referenceIndexingMapsFmt, cppOpName, dimsStr, mapsStr,
                      llvm::join(mapList, ", "));
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
  void {0}::regionBuilder(Block &block, ValueRange captures) {
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

std::string
TCParser::RegisteredAttr::getValueFn(ArrayRef<uint64_t> indices) const {
  if (isArray)
    return "";

  if (!vectorDims.empty()) {
    SmallVector<std::string, 4> indexStrs;
    for (uint64_t index : indices)
      indexStrs.push_back(std::to_string(index));
    std::string indexList = llvm::join(indexStrs, ", ");
    if (elementType == "f32")
      return llvm::formatv("getValue<float>({ {0} })", indexList);
    if (elementType == "i32")
      return llvm::formatv("getValue<int>({ {0} })", indexList);
    if (elementType == "i64")
      return llvm::formatv("getValue<int64_t>({ {0} })", indexList);

    return "";
  }

  if (elementType == "f32")
    return "getValue().convertToFloat()";
  if (elementType == "i32" || elementType == "i64")
    return "getInt()";
  return "";
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
  (void)parseAndEmitAllTensorComprehensions(output->os(), parser);
  output->keep();

  return 0;
}
