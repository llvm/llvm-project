//===- OpFormatGen.cpp - MLIR operation asm format generator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpFormatGen.h"
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/OpClass.h"
#include "mlir/TableGen/OpInterfaces.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#define DEBUG_TYPE "mlir-tblgen-opformatgen"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Element
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single format element.
class Element {
public:
  enum class Kind {
    /// This element is a directive.
    AttrDictDirective,
    FunctionalTypeDirective,
    OperandsDirective,
    ResultsDirective,
    TypeDirective,

    /// This element is a literal.
    Literal,

    /// This element is an variable value.
    AttributeVariable,
    OperandVariable,
    ResultVariable,
  };
  Element(Kind kind) : kind(kind) {}
  virtual ~Element() = default;

  /// Return the kind of this element.
  Kind getKind() const { return kind; }

private:
  /// The kind of this element.
  Kind kind;
};
} // namespace

//===----------------------------------------------------------------------===//
// VariableElement

namespace {
/// This class represents an instance of an variable element. A variable refers
/// to something registered on the operation itself, e.g. an argument, result,
/// etc.
template <typename VarT, Element::Kind kindVal>
class VariableElement : public Element {
public:
  VariableElement(const VarT *var) : Element(kindVal), var(var) {}
  static bool classof(const Element *element) {
    return element->getKind() == kindVal;
  }
  const VarT *getVar() { return var; }

private:
  const VarT *var;
};

/// This class represents a variable that refers to an attribute argument.
using AttributeVariable =
    VariableElement<NamedAttribute, Element::Kind::AttributeVariable>;

/// This class represents a variable that refers to an operand argument.
using OperandVariable =
    VariableElement<NamedTypeConstraint, Element::Kind::OperandVariable>;

/// This class represents a variable that refers to a result.
using ResultVariable =
    VariableElement<NamedTypeConstraint, Element::Kind::ResultVariable>;
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// DirectiveElement

namespace {
/// This class implements single kind directives.
template <Element::Kind type>
class DirectiveElement : public Element {
public:
  DirectiveElement() : Element(type){};
  static bool classof(const Element *ele) { return ele->getKind() == type; }
};
/// This class represents the `attr-dict` directive. This directive represents
/// the attribute dictionary of the operation.
using AttrDictDirective = DirectiveElement<Element::Kind::AttrDictDirective>;

/// This class represents the `operands` directive. This directive represents
/// all of the operands of an operation.
using OperandsDirective = DirectiveElement<Element::Kind::OperandsDirective>;

/// This class represents the `results` directive. This directive represents
/// all of the results of an operation.
using ResultsDirective = DirectiveElement<Element::Kind::ResultsDirective>;

/// This class represents the `functional-type` directive. This directive takes
/// two arguments and formats them, respectively, as the inputs and results of a
/// FunctionType.
struct FunctionalTypeDirective
    : public DirectiveElement<Element::Kind::FunctionalTypeDirective> {
public:
  FunctionalTypeDirective(std::unique_ptr<Element> inputs,
                          std::unique_ptr<Element> results)
      : inputs(std::move(inputs)), results(std::move(results)) {}
  Element *getInputs() const { return inputs.get(); }
  Element *getResults() const { return results.get(); }

private:
  /// The input and result arguments.
  std::unique_ptr<Element> inputs, results;
};

/// This class represents the `type` directive.
struct TypeDirective : public DirectiveElement<Element::Kind::TypeDirective> {
public:
  TypeDirective(std::unique_ptr<Element> arg) : operand(std::move(arg)) {}
  Element *getOperand() const { return operand.get(); }

private:
  /// The operand that is used to format the directive.
  std::unique_ptr<Element> operand;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LiteralElement

namespace {
/// This class represents an instance of a literal element.
class LiteralElement : public Element {
public:
  LiteralElement(StringRef literal)
      : Element{Kind::Literal}, literal(literal){};
  static bool classof(const Element *element) {
    return element->getKind() == Kind::Literal;
  }

  /// Return the literal for this element.
  StringRef getLiteral() const { return literal; }

  /// Returns true if the given string is a valid literal.
  static bool isValidLiteral(StringRef value);

private:
  /// The spelling of the literal for this element.
  StringRef literal;
};
} // end anonymous namespace

bool LiteralElement::isValidLiteral(StringRef value) {
  if (value.empty())
    return false;
  char front = value.front();

  // If there is only one character, this must either be punctuation or a
  // single character bare identifier.
  if (value.size() == 1)
    return isalpha(front) || StringRef("_:,=<>()[]").contains(front);

  // Check the punctuation that are larger than a single character.
  if (value == "->")
    return true;

  // Otherwise, this must be an identifier.
  if (!isalpha(front) && front != '_')
    return false;
  return llvm::all_of(value.drop_front(), [](char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

//===----------------------------------------------------------------------===//
// OperationFormat
//===----------------------------------------------------------------------===//

namespace {
struct OperationFormat {
  OperationFormat(const Operator &op)
      : allOperandTypes(false), allResultTypes(false) {
    buildableOperandTypes.resize(op.getNumOperands(), llvm::None);
    buildableResultTypes.resize(op.getNumResults(), llvm::None);
  }

  /// The various elements in this format.
  std::vector<std::unique_ptr<Element>> elements;

  /// A flag indicating if all operand/result types were seen. If the format
  /// contains these, it can not contain individual type resolvers.
  bool allOperandTypes, allResultTypes;

  /// A map of buildable types to indices.
  llvm::MapVector<StringRef, int, llvm::StringMap<int>> buildableTypes;

  /// The index of the buildable type, if valid, for every operand and result.
  std::vector<Optional<int>> buildableOperandTypes, buildableResultTypes;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// FormatLexer
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a specific token in the input format.
class Token {
public:
  enum Kind {
    // Markers.
    eof,
    error,

    // Tokens with no info.
    l_paren,
    r_paren,
    comma,
    equal,

    // Keywords.
    keyword_start,
    kw_attr_dict,
    kw_functional_type,
    kw_operands,
    kw_results,
    kw_type,
    keyword_end,

    // String valued tokens.
    identifier,
    literal,
    variable,
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
  bool isKeyword() const { return kind > keyword_start && kind < keyword_end; }

private:
  /// Discriminator that indicates the kind of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

/// This class implements a simple lexer for operation assembly format strings.
class FormatLexer {
public:
  FormatLexer(llvm::SourceMgr &mgr);

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

  /// Lex an identifier, literal, or variable.
  Token lexIdentifier(const char *tokStart);
  Token lexLiteral(const char *tokStart);
  Token lexVariable(const char *tokStart);

  llvm::SourceMgr &srcMgr;
  StringRef curBuffer;
  const char *curPtr;
};
} // end anonymous namespace

FormatLexer::FormatLexer(llvm::SourceMgr &mgr) : srcMgr(mgr) {
  curBuffer = srcMgr.getMemoryBuffer(mgr.getMainFileID())->getBuffer();
  curPtr = curBuffer.begin();
}

Token FormatLexer::emitError(llvm::SMLoc loc, const Twine &msg) {
  srcMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
  return formToken(Token::error, loc.getPointer());
}
Token FormatLexer::emitError(const char *loc, const Twine &msg) {
  return emitError(llvm::SMLoc::getFromPointer(loc), msg);
}

int FormatLexer::getNextChar() {
  char curChar = *curPtr++;
  switch (curChar) {
  default:
    return (unsigned char)curChar;
  case 0: {
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file. Disambiguate that here.
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

Token FormatLexer::lexToken() {
  const char *tokStart = curPtr;

  // This always consumes at least one character.
  int curChar = getNextChar();
  switch (curChar) {
  default:
    // Handle identifiers: [a-zA-Z_]
    if (isalpha(curChar) || curChar == '_')
      return lexIdentifier(tokStart);

    // Unknown character, emit an error.
    return emitError(tokStart, "unexpected character");
  case EOF:
    // Return EOF denoting the end of lexing.
    return formToken(Token::eof, tokStart);

  // Lex punctuation.
  case ',':
    return formToken(Token::comma, tokStart);
  case '=':
    return formToken(Token::equal, tokStart);
  case '(':
    return formToken(Token::l_paren, tokStart);
  case ')':
    return formToken(Token::r_paren, tokStart);

  // Ignore whitespace characters.
  case 0:
  case ' ':
  case '\t':
  case '\n':
    return lexToken();

  case '`':
    return lexLiteral(tokStart);
  case '$':
    return lexVariable(tokStart);
  }
}

Token FormatLexer::lexLiteral(const char *tokStart) {
  assert(curPtr[-1] == '`');

  // Lex a literal surrounded by ``.
  while (const char curChar = *curPtr++) {
    if (curChar == '`')
      return formToken(Token::literal, tokStart);
  }
  return emitError(curPtr - 1, "unexpected end of file in literal");
}

Token FormatLexer::lexVariable(const char *tokStart) {
  if (!isalpha(curPtr[0]) && curPtr[0] != '_')
    return emitError(curPtr - 1, "expected variable name");

  // Otherwise, consume the rest of the characters.
  while (isalnum(*curPtr) || *curPtr == '_')
    ++curPtr;
  return formToken(Token::variable, tokStart);
}

Token FormatLexer::lexIdentifier(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_\-]*
  while (isalnum(*curPtr) || *curPtr == '_' || *curPtr == '-')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef str(tokStart, curPtr - tokStart);
  Token::Kind kind = llvm::StringSwitch<Token::Kind>(str)
                         .Case("attr-dict", Token::kw_attr_dict)
                         .Case("functional-type", Token::kw_functional_type)
                         .Case("operands", Token::kw_operands)
                         .Case("results", Token::kw_results)
                         .Case("type", Token::kw_type)
                         .Default(Token::identifier);
  return Token(kind, str);
}

//===----------------------------------------------------------------------===//
// FormatParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a parser for an instance of an operation assembly
/// format.
class FormatParser {
public:
  FormatParser(llvm::SourceMgr &mgr, OperationFormat &format, Operator &op)
      : lexer(mgr), curToken(lexer.lexToken()), fmt(format), op(op),
        seenOperandTypes(op.getNumOperands()),
        seenResultTypes(op.getNumResults()) {}

  /// Parse the operation assembly format.
  LogicalResult parse();

private:
  /// Parse a specific element.
  LogicalResult parseElement(std::unique_ptr<Element> &element,
                             bool isTopLevel);
  LogicalResult parseVariable(std::unique_ptr<Element> &element,
                              bool isTopLevel);
  LogicalResult parseDirective(std::unique_ptr<Element> &element,
                               bool isTopLevel);
  LogicalResult parseLiteral(std::unique_ptr<Element> &element);

  /// Parse the various different directives.
  LogicalResult parseAttrDictDirective(std::unique_ptr<Element> &element,
                                       llvm::SMLoc loc, bool isTopLevel);
  LogicalResult parseFunctionalTypeDirective(std::unique_ptr<Element> &element,
                                             Token tok, bool isTopLevel);
  LogicalResult parseOperandsDirective(std::unique_ptr<Element> &element,
                                       llvm::SMLoc loc, bool isTopLevel);
  LogicalResult parseResultsDirective(std::unique_ptr<Element> &element,
                                      llvm::SMLoc loc, bool isTopLevel);
  LogicalResult parseTypeDirective(std::unique_ptr<Element> &element, Token tok,
                                   bool isTopLevel);
  LogicalResult parseTypeDirectiveOperand(std::unique_ptr<Element> &element);

  //===--------------------------------------------------------------------===//
  // Lexer Utilities
  //===--------------------------------------------------------------------===//

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(curToken.getKind() != Token::eof &&
           curToken.getKind() != Token::error &&
           "shouldn't advance past EOF or errors");
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

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  FormatLexer lexer;
  Token curToken;
  OperationFormat &fmt;
  Operator &op;

  // The following are various bits of format state used for verification during
  // parsing.
  bool hasAllOperands = false, hasAttrDict = false;
  llvm::SmallBitVector seenOperandTypes, seenResultTypes;
  llvm::DenseSet<const NamedTypeConstraint *> seenOperands;
  llvm::DenseSet<const NamedAttribute *> seenAttrs;
};
} // end anonymous namespace

LogicalResult FormatParser::parse() {
  llvm::SMLoc loc = curToken.getLoc();

  // Parse each of the format elements into the main format.
  while (curToken.getKind() != Token::eof) {
    std::unique_ptr<Element> element;
    if (failed(parseElement(element, /*isTopLevel=*/true)))
      return failure();
    fmt.elements.push_back(std::move(element));
  }

  // Check that the attribute dictionary is in the format.
  if (!hasAttrDict)
    return emitError(loc, "format missing 'attr-dict' directive");

  // Check that all of the result types can be inferred.
  auto &buildableTypes = fmt.buildableTypes;
  if (!fmt.allResultTypes) {
    for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
      if (seenResultTypes.test(i))
        continue;

      // If the result is not variadic, allow for the case where the type has a
      // builder that we can use.
      NamedTypeConstraint &result = op.getResult(i);
      Optional<StringRef> builder = result.constraint.getBuilderCall();
      if (!builder || result.constraint.isVariadic()) {
        return emitError(loc, "format missing instance of result #" + Twine(i) +
                                  "('" + result.name + "') type");
      }
      // Note in the format that this result uses the custom builder.
      auto it = buildableTypes.insert({*builder, buildableTypes.size()});
      fmt.buildableResultTypes[i] = it.first->second;
    }
  }

  // Check that all of the operands are within the format, and their types can
  // be inferred.
  for (unsigned i = 0, e = op.getNumOperands(); i != e; ++i) {
    NamedTypeConstraint &operand = op.getOperand(i);

    // Check that the operand itself is in the format.
    if (!hasAllOperands && !seenOperands.count(&operand)) {
      return emitError(loc, "format missing instance of operand #" + Twine(i) +
                                "('" + operand.name + "')");
    }

    // Check that the operand type is in the format, or that it can be inferred.
    if (!fmt.allOperandTypes && !seenOperandTypes.test(i)) {
      // Similarly to results, allow a custom builder for resolving the type if
      // we aren't using the 'operands' directive.
      Optional<StringRef> builder = operand.constraint.getBuilderCall();
      if (!builder || (hasAllOperands && operand.isVariadic())) {
        return emitError(loc, "format missing instance of operand #" +
                                  Twine(i) + "('" + operand.name + "') type");
      }
      auto it = buildableTypes.insert({*builder, buildableTypes.size()});
      fmt.buildableOperandTypes[i] = it.first->second;
    }
  }
  return success();
}

LogicalResult FormatParser::parseElement(std::unique_ptr<Element> &element,
                                         bool isTopLevel) {
  // Directives.
  if (curToken.isKeyword())
    return parseDirective(element, isTopLevel);
  // Literals.
  if (curToken.getKind() == Token::literal)
    return parseLiteral(element);
  // Variables.
  if (curToken.getKind() == Token::variable)
    return parseVariable(element, isTopLevel);
  return emitError(curToken.getLoc(),
                   "expected directive, literal, or variable");
}

LogicalResult FormatParser::parseVariable(std::unique_ptr<Element> &element,
                                          bool isTopLevel) {
  Token varTok = curToken;
  consumeToken();

  StringRef name = varTok.getSpelling().drop_front();
  llvm::SMLoc loc = varTok.getLoc();

  // Functor used to find an element within the given range that has the same
  // name as 'name'.
  auto findArg = [&](auto &&range) {
    auto it = llvm::find_if(range, [=](auto &arg) { return arg.name == name; });
    return it != range.end() ? &*it : nullptr;
  };

  // Check that the parsed argument is something actually registered on the op.
  /// Attributes
  if (const NamedAttribute *attr = findArg(op.getAttributes())) {
    if (isTopLevel && !seenAttrs.insert(attr).second)
      return emitError(loc, "attribute '" + name + "' is already bound");
    element = std::make_unique<AttributeVariable>(attr);
    return success();
  }
  /// Operands
  if (const NamedTypeConstraint *operand = findArg(op.getOperands())) {
    if (isTopLevel) {
      if (hasAllOperands || !seenOperands.insert(operand).second)
        return emitError(loc, "operand '" + name + "' is already bound");
    }
    element = std::make_unique<OperandVariable>(operand);
    return success();
  }
  /// Results.
  if (const NamedTypeConstraint *result = findArg(op.getResults())) {
    if (isTopLevel)
      return emitError(loc, "results can not be used at the top level");
    element = std::make_unique<ResultVariable>(result);
    return success();
  }
  return emitError(loc, "expected variable to refer to a argument or result");
}

LogicalResult FormatParser::parseDirective(std::unique_ptr<Element> &element,
                                           bool isTopLevel) {
  Token dirTok = curToken;
  consumeToken();

  switch (dirTok.getKind()) {
  case Token::kw_attr_dict:
    return parseAttrDictDirective(element, dirTok.getLoc(), isTopLevel);
  case Token::kw_functional_type:
    return parseFunctionalTypeDirective(element, dirTok, isTopLevel);
  case Token::kw_operands:
    return parseOperandsDirective(element, dirTok.getLoc(), isTopLevel);
  case Token::kw_results:
    return parseResultsDirective(element, dirTok.getLoc(), isTopLevel);
  case Token::kw_type:
    return parseTypeDirective(element, dirTok, isTopLevel);

  default:
    llvm_unreachable("unknown directive token");
  }
}

LogicalResult FormatParser::parseLiteral(std::unique_ptr<Element> &element) {
  Token literalTok = curToken;
  consumeToken();

  // Check that the parsed literal is valid.
  StringRef value = literalTok.getSpelling().drop_front().drop_back();
  if (!LiteralElement::isValidLiteral(value))
    return emitError(literalTok.getLoc(), "expected valid literal");

  element = std::make_unique<LiteralElement>(value);
  return success();
}

LogicalResult
FormatParser::parseAttrDictDirective(std::unique_ptr<Element> &element,
                                     llvm::SMLoc loc, bool isTopLevel) {
  if (!isTopLevel)
    return emitError(loc, "'attr-dict' directive can only be used as a "
                          "top-level directive");
  if (hasAttrDict)
    return emitError(loc, "'attr-dict' directive has already been seen");

  hasAttrDict = true;
  element = std::make_unique<AttrDictDirective>();
  return success();
}

LogicalResult
FormatParser::parseFunctionalTypeDirective(std::unique_ptr<Element> &element,
                                           Token tok, bool isTopLevel) {
  llvm::SMLoc loc = tok.getLoc();
  if (!isTopLevel)
    return emitError(
        loc, "'functional-type' is only valid as a top-level directive");

  // Parse the main operand.
  std::unique_ptr<Element> inputs, results;
  if (failed(parseToken(Token::l_paren, "expected '(' before argument list")) ||
      failed(parseTypeDirectiveOperand(inputs)) ||
      failed(parseToken(Token::comma, "expected ',' after inputs argument")) ||
      failed(parseTypeDirectiveOperand(results)) ||
      failed(parseToken(Token::r_paren, "expected ')' after argument list")))
    return failure();

  // Get the proper directive kind and create it.
  element = std::make_unique<FunctionalTypeDirective>(std::move(inputs),
                                                      std::move(results));
  return success();
}

LogicalResult
FormatParser::parseOperandsDirective(std::unique_ptr<Element> &element,
                                     llvm::SMLoc loc, bool isTopLevel) {
  if (isTopLevel && (hasAllOperands || !seenOperands.empty()))
    return emitError(loc, "'operands' directive creates overlap in format");
  hasAllOperands = true;
  element = std::make_unique<OperandsDirective>();
  return success();
}

LogicalResult
FormatParser::parseResultsDirective(std::unique_ptr<Element> &element,
                                    llvm::SMLoc loc, bool isTopLevel) {
  if (isTopLevel)
    return emitError(loc, "'results' directive can not be used as a "
                          "top-level directive");
  element = std::make_unique<ResultsDirective>();
  return success();
}

LogicalResult
FormatParser::parseTypeDirective(std::unique_ptr<Element> &element, Token tok,
                                 bool isTopLevel) {
  llvm::SMLoc loc = tok.getLoc();
  if (!isTopLevel)
    return emitError(loc, "'type' is only valid as a top-level directive");

  std::unique_ptr<Element> operand;
  if (failed(parseToken(Token::l_paren, "expected '(' before argument list")) ||
      failed(parseTypeDirectiveOperand(operand)) ||
      failed(parseToken(Token::r_paren, "expected ')' after argument list")))
    return failure();
  element = std::make_unique<TypeDirective>(std::move(operand));
  return success();
}

LogicalResult
FormatParser::parseTypeDirectiveOperand(std::unique_ptr<Element> &element) {
  llvm::SMLoc loc = curToken.getLoc();
  if (failed(parseElement(element, /*isTopLevel=*/false)))
    return failure();
  if (isa<LiteralElement>(element.get()))
    return emitError(
        loc, "'type' directive operand expects variable or directive operand");

  if (auto *var = dyn_cast<OperandVariable>(element.get())) {
    unsigned opIdx = var->getVar() - op.operand_begin();
    if (fmt.allOperandTypes || seenOperandTypes.test(opIdx))
      return emitError(loc, "'type' of '" + var->getVar()->name +
                                "' is already bound");
    seenOperandTypes.set(opIdx);
  } else if (auto *var = dyn_cast<ResultVariable>(element.get())) {
    unsigned resIdx = var->getVar() - op.result_begin();
    if (fmt.allResultTypes || seenResultTypes.test(resIdx))
      return emitError(loc, "'type' of '" + var->getVar()->name +
                                "' is already bound");
    seenResultTypes.set(resIdx);
  } else if (isa<OperandsDirective>(&*element)) {
    if (fmt.allOperandTypes || seenOperandTypes.any())
      return emitError(loc, "'operands' 'type' is already bound");
    fmt.allOperandTypes = true;
  } else if (isa<ResultsDirective>(&*element)) {
    if (fmt.allResultTypes || seenResultTypes.any())
      return emitError(loc, "'results' 'type' is already bound");
    fmt.allResultTypes = true;
  } else {
    return emitError(loc, "invalid argument to 'type' directive");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateOpFormat(const Operator &constOp, OpClass &opClass) {
  // TODO(riverriddle) Operator doesn't expose all necessary functionality via
  // the const interface.
  Operator &op = const_cast<Operator &>(constOp);

  // Check if the operation specified the format field.
  StringRef formatStr;
  TypeSwitch<llvm::Init *>(op.getDef().getValueInit("assemblyFormat"))
      .Case<llvm::StringInit, llvm::CodeInit>(
          [&](auto *init) { formatStr = init->getValue(); });
  if (formatStr.empty())
    return;

  // Parse the format description.
  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(formatStr),
                         llvm::SMLoc());
  OperationFormat format(op);
  if (failed(FormatParser(mgr, format, op).parse()))
    return;
}
