//===- FormatGen.h - Utilities for custom assembly formats ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common classes for building custom assembly format parsers
// and generators.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_
#define MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SMLoc.h"
#include <vector>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// FormatToken
//===----------------------------------------------------------------------===//

/// This class represents a specific token in the input format.
class FormatToken {
public:
  /// Basic token kinds.
  enum Kind {
    // Markers.
    eof,
    error,

    // Tokens with no info.
    l_paren,
    r_paren,
    caret,
    colon,
    comma,
    equal,
    less,
    greater,
    question,
    star,

    // Keywords.
    keyword_start,
    kw_attr_dict,
    kw_attr_dict_w_keyword,
    kw_custom,
    kw_functional_type,
    kw_operands,
    kw_params,
    kw_qualified,
    kw_ref,
    kw_regions,
    kw_results,
    kw_struct,
    kw_successors,
    kw_type,
    keyword_end,

    // String valued tokens.
    identifier,
    literal,
    variable,
  };

  FormatToken(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  /// Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  /// Return the kind of this token.
  Kind getKind() const { return kind; }

  /// Return a location for this token.
  SMLoc getLoc() const;

  /// Returns true if the token is of the given kind.
  bool is(Kind kind) { return getKind() == kind; }

  /// Return if this token is a keyword.
  bool isKeyword() const {
    return getKind() > Kind::keyword_start && getKind() < Kind::keyword_end;
  }

private:
  /// Discriminator that indicates the kind of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

//===----------------------------------------------------------------------===//
// FormatLexer
//===----------------------------------------------------------------------===//

/// This class implements a simple lexer for operation assembly format strings.
class FormatLexer {
public:
  FormatLexer(llvm::SourceMgr &mgr, SMLoc loc);

  /// Lex the next token and return it.
  FormatToken lexToken();

  /// Emit an error to the lexer with the given location and message.
  FormatToken emitError(SMLoc loc, const Twine &msg);
  FormatToken emitError(const char *loc, const Twine &msg);

  FormatToken emitErrorAndNote(SMLoc loc, const Twine &msg, const Twine &note);

private:
  /// Return the next character in the stream.
  int getNextChar();

  /// Lex an identifier, literal, or variable.
  FormatToken lexIdentifier(const char *tokStart);
  FormatToken lexLiteral(const char *tokStart);
  FormatToken lexVariable(const char *tokStart);

  /// Create a token with the current pointer and a start pointer.
  FormatToken formToken(FormatToken::Kind kind, const char *tokStart) {
    return FormatToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  /// The source manager containing the format string.
  llvm::SourceMgr &mgr;
  /// Location of the format string.
  SMLoc loc;
  /// Buffer containing the format string.
  StringRef curBuffer;
  /// Current pointer in the buffer.
  const char *curPtr;
};

//===----------------------------------------------------------------------===//
// FormatElement
//===----------------------------------------------------------------------===//

/// This class represents a single format element.
///
/// If you squint and take a close look, you can see the outline of a `Format`
/// dialect.
class FormatElement {
public:
  virtual ~FormatElement();

  // The top-level kinds of format elements.
  enum Kind { Literal, Variable, Whitespace, Directive, Optional };

  /// Support LLVM-style RTTI.
  static bool classof(const FormatElement *el) { return true; }

  /// Get the element kind.
  Kind getKind() const { return kind; }

protected:
  /// Create a format element with the given kind.
  FormatElement(Kind kind) : kind(kind) {}

private:
  /// The kind of the element.
  Kind kind;
};

/// The base class for all format elements. This class implements common methods
/// for LLVM-style RTTI.
template <FormatElement::Kind ElementKind>
class FormatElementBase : public FormatElement {
public:
  /// Support LLVM-style RTTI.
  static bool classof(const FormatElement *el) {
    return ElementKind == el->getKind();
  }

protected:
  /// Create a format element with the given kind.
  FormatElementBase() : FormatElement(ElementKind) {}
};

/// This class represents a literal element. A literal is either one of the
/// supported punctuation characters (e.g. `(` or `,`) or a string literal (e.g.
/// `literal`).
class LiteralElement : public FormatElementBase<FormatElement::Literal> {
public:
  /// Create a literal element with the given spelling.
  explicit LiteralElement(StringRef spelling) : spelling(spelling) {}

  /// Get the spelling of the literal.
  StringRef getSpelling() const { return spelling; }

private:
  /// The spelling of the variable, i.e. the string contained within the
  /// backticks.
  StringRef spelling;
};

/// This class represents a variable element. A variable refers to some part of
/// the object being parsed, e.g. an attribute or operand on an operation or a
/// parameter on an attribute.
class VariableElement : public FormatElementBase<FormatElement::Variable> {
public:
  /// These are the kinds of variables.
  enum Kind { Attribute, Operand, Region, Result, Successor, Parameter };

  /// Get the kind of variable.
  Kind getKind() const { return kind; }

protected:
  /// Create a variable with a kind.
  VariableElement(Kind kind) : kind(kind) {}

private:
  /// The kind of variable.
  Kind kind;
};

/// Base class for variable elements. This class implements common methods for
/// LLVM-style RTTI.
template <VariableElement::Kind VariableKind>
class VariableElementBase : public VariableElement {
public:
  /// An element is of this class if it is a variable and has the same variable
  /// type.
  static bool classof(const FormatElement *el) {
    if (auto *varEl = dyn_cast<VariableElement>(el))
      return VariableKind == varEl->getKind();
    return false;
  }

protected:
  /// Create a variable element with the given variable kind.
  VariableElementBase() : VariableElement(VariableKind) {}
};

/// This class represents a whitespace element, e.g. a newline or space. It is a
/// literal that is printed but never parsed. When the value is empty, i.e. ``,
/// a space is elided where one would have been printed automatically.
class WhitespaceElement : public FormatElementBase<FormatElement::Whitespace> {
public:
  /// Create a whitespace element.
  explicit WhitespaceElement(StringRef value) : value(value) {}

  /// Get the whitespace value.
  StringRef getValue() const { return value; }

private:
  /// The value of the whitespace element. Can be empty.
  StringRef value;
};

class DirectiveElement : public FormatElementBase<FormatElement::Directive> {
public:
  /// These are the kinds of directives.
  enum Kind {
    AttrDict,
    Custom,
    FunctionalType,
    Operands,
    Ref,
    Regions,
    Results,
    Successors,
    Type,
    Params,
    Struct
  };

  /// Get the directive kind.
  Kind getKind() const { return kind; }

protected:
  /// Create a directive element with a kind.
  DirectiveElement(Kind kind) : kind(kind) {}

private:
  /// The directive kind.
  Kind kind;
};

/// Base class for directive elements. This class implements common methods for
/// LLVM-style RTTI.
template <DirectiveElement::Kind DirectiveKind>
class DirectiveElementBase : public DirectiveElement {
public:
  /// Create a directive element with the specified kind.
  DirectiveElementBase() : DirectiveElement(DirectiveKind) {}

  /// A format element is of this class if it is a directive element and has the
  /// same kind.
  static bool classof(const FormatElement *el) {
    if (auto *directiveEl = dyn_cast<DirectiveElement>(el))
      return DirectiveKind == directiveEl->getKind();
    return false;
  }
};

/// This class represents a custom format directive that is implemented by the
/// user in C++. The directive accepts a list of arguments that is passed to the
/// C++ function.
class CustomDirective : public DirectiveElementBase<DirectiveElement::Custom> {
public:
  /// Create a custom directive with a name and list of arguments.
  CustomDirective(StringRef name, std::vector<FormatElement *> &&arguments)
      : name(name), arguments(std::move(arguments)) {}

  /// Get the custom directive name.
  StringRef getName() const { return name; }

  /// Get the arguments to the custom directive.
  ArrayRef<FormatElement *> getArguments() const { return arguments; }

private:
  /// The name of the custom directive. The name is used to call two C++
  /// methods: `parse{name}` and `print{name}` with the given arguments.
  StringRef name;
  /// The arguments with which to call the custom functions. These are either
  /// variables (for which the functions are responsible for populating) or
  /// references to variables.
  std::vector<FormatElement *> arguments;
};

/// This class represents a group of elements that are optionally emitted based
/// on an optional variable "anchor" and a group of elements that are emitted
/// when the anchor element is not present.
class OptionalElement : public FormatElementBase<FormatElement::Optional> {
public:
  /// Create an optional group with the given child elements.
  OptionalElement(std::vector<FormatElement *> &&thenElements,
                  std::vector<FormatElement *> &&elseElements,
                  unsigned anchorIndex, unsigned parseStart)
      : thenElements(std::move(thenElements)),
        elseElements(std::move(elseElements)), anchorIndex(anchorIndex),
        parseStart(parseStart) {}

  /// Return the `then` elements of the optional group.
  ArrayRef<FormatElement *> getThenElements() const { return thenElements; }

  /// Return the `else` elements of the optional group.
  ArrayRef<FormatElement *> getElseElements() const { return elseElements; }

  /// Return the anchor of the optional group.
  FormatElement *getAnchor() const { return thenElements[anchorIndex]; }

  /// Return the index of the first element to be parsed.
  unsigned getParseStart() const { return parseStart; }

private:
  /// The child elements emitted when the anchor is present.
  std::vector<FormatElement *> thenElements;
  /// The child elements emitted when the anchor is not present.
  std::vector<FormatElement *> elseElements;
  /// The index of the anchor element of the optional group within
  /// `thenElements`.
  unsigned anchorIndex;
  /// The index of the first element that is parsed in `thenElements`. That is,
  /// the first non-whitespace element.
  unsigned parseStart;
};

//===----------------------------------------------------------------------===//
// FormatParserBase
//===----------------------------------------------------------------------===//

/// Base class for a parser that implements an assembly format. This class
/// defines a common assembly format syntax and the creation of format elements.
/// Subclasses will need to implement parsing for the format elements they
/// support.
class FormatParser {
public:
  /// Vtable anchor.
  virtual ~FormatParser();

  /// Parse the assembly format.
  FailureOr<std::vector<FormatElement *>> parse();

protected:
  /// The current context of the parser when parsing an element.
  enum Context {
    /// The element is being parsed in a "top-level" context, i.e. at the top of
    /// the format or in an optional group.
    TopLevelContext,
    /// The element is being parsed as a custom directive child.
    CustomDirectiveContext,
    /// The element is being parsed as a type directive child.
    TypeDirectiveContext,
    /// The element is being parsed as a reference directive child.
    RefDirectiveContext,
    /// The element is being parsed as a struct directive child.
    StructDirectiveContext
  };

  /// Create a format parser with the given source manager and a location.
  explicit FormatParser(llvm::SourceMgr &mgr, llvm::SMLoc loc)
      : lexer(mgr, loc), curToken(lexer.lexToken()) {}

  /// Allocate and construct a format element.
  template <typename FormatElementT, typename... Args>
  FormatElementT *create(Args &&...args) {
    // FormatElementT *ptr = allocator.Allocate<FormatElementT>();
    // ::new (ptr) FormatElementT(std::forward<Args>(args)...);
    // return ptr;
    auto mem = std::make_unique<FormatElementT>(std::forward<Args>(args)...);
    FormatElementT *ptr = mem.get();
    allocator.push_back(std::move(mem));
    return ptr;
  }

  //===--------------------------------------------------------------------===//
  // Element Parsing

  /// Parse a single element of any kind.
  FailureOr<FormatElement *> parseElement(Context ctx);
  /// Parse a literal.
  FailureOr<FormatElement *> parseLiteral(Context ctx);
  /// Parse a variable.
  FailureOr<FormatElement *> parseVariable(Context ctx);
  /// Parse a directive.
  FailureOr<FormatElement *> parseDirective(Context ctx);
  /// Parse an optional group.
  FailureOr<FormatElement *> parseOptionalGroup(Context ctx);

  /// Parse a custom directive.
  FailureOr<FormatElement *> parseCustomDirective(llvm::SMLoc loc, Context ctx);

  /// Parse a format-specific variable kind.
  virtual FailureOr<FormatElement *>
  parseVariableImpl(llvm::SMLoc loc, StringRef name, Context ctx) = 0;
  /// Parse a format-specific directive kind.
  virtual FailureOr<FormatElement *>
  parseDirectiveImpl(llvm::SMLoc loc, FormatToken::Kind kind, Context ctx) = 0;

  //===--------------------------------------------------------------------===//
  // Format Verification

  /// Verify that the format is well-formed.
  virtual LogicalResult verify(llvm::SMLoc loc,
                               ArrayRef<FormatElement *> elements) = 0;
  /// Verify the arguments to a custom directive.
  virtual LogicalResult
  verifyCustomDirectiveArguments(llvm::SMLoc loc,
                                 ArrayRef<FormatElement *> arguments) = 0;
  /// Verify the elements of an optional group.
  virtual LogicalResult
  verifyOptionalGroupElements(llvm::SMLoc loc,
                              ArrayRef<FormatElement *> elements,
                              Optional<unsigned> anchorIndex) = 0;

  //===--------------------------------------------------------------------===//
  // Lexer Utilities

  /// Emit an error at the given location.
  LogicalResult emitError(llvm::SMLoc loc, const Twine &msg) {
    lexer.emitError(loc, msg);
    return failure();
  }

  /// Emit an error and a note at the given notation.
  LogicalResult emitErrorAndNote(llvm::SMLoc loc, const Twine &msg,
                                 const Twine &note) {
    lexer.emitErrorAndNote(loc, msg, note);
    return failure();
  }

  /// Parse a single token of the expected kind.
  FailureOr<FormatToken> parseToken(FormatToken::Kind kind, const Twine &msg) {
    if (!curToken.is(kind))
      return emitError(curToken.getLoc(), msg);
    FormatToken tok = curToken;
    consumeToken();
    return tok;
  }

  /// Advance the lexer to the next token.
  void consumeToken() {
    assert(!curToken.is(FormatToken::eof) && !curToken.is(FormatToken::error) &&
           "shouldn't advance past EOF or errors");
    curToken = lexer.lexToken();
  }

  /// Get the current token.
  FormatToken peekToken() { return curToken; }

private:
  /// The format parser retains ownership of the format elements in a bump
  /// pointer allocator.
  // FIXME: FormatElement with `std::vector` need to be converted to use
  // trailing objects.
  // llvm::BumpPtrAllocator allocator;
  std::vector<std::unique_ptr<FormatElement>> allocator;
  /// The format lexer to use.
  FormatLexer lexer;
  /// The current token in the lexer.
  FormatToken curToken;
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Whether a space needs to be emitted before a literal. E.g., two keywords
/// back-to-back require a space separator, but a keyword followed by '<' does
/// not require a space.
bool shouldEmitSpaceBefore(StringRef value, bool lastWasPunctuation);

/// Returns true if the given string can be formatted as a keyword.
bool canFormatStringAsKeyword(StringRef value,
                              function_ref<void(Twine)> emitError = nullptr);

/// Returns true if the given string is valid format literal element.
/// If `emitError` is provided, it is invoked with the reason for the failure.
bool isValidLiteral(StringRef value,
                    function_ref<void(Twine)> emitError = nullptr);

/// Whether a failure in parsing the assembly format should be a fatal error.
extern llvm::cl::opt<bool> formatErrorIsFatal;

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_
