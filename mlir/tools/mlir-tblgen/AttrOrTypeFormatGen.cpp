//===- AttrOrTypeFormatGen.cpp - MLIR attribute and type format generator -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AttrOrTypeFormatGen.h"
#include "FormatGen.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace mlir;
using namespace mlir::tblgen;

using llvm::formatv;

//===----------------------------------------------------------------------===//
// Element
//===----------------------------------------------------------------------===//

namespace {

/// This class represents a single format element.
class Element {
public:
  /// LLVM-style RTTI.
  enum class Kind {
    /// This element is a directive.
    ParamsDirective,
    StructDirective,

    /// This element is a literal.
    Literal,

    /// This element is a variable.
    Variable,
  };
  Element(Kind kind) : kind(kind) {}
  virtual ~Element() = default;

  /// Return the kind of this element.
  Kind getKind() const { return kind; }

private:
  /// The kind of this element.
  Kind kind;
};

/// This class represents an instance of a literal element.
class LiteralElement : public Element {
public:
  LiteralElement(StringRef literal)
      : Element(Kind::Literal), literal(literal) {}

  static bool classof(const Element *el) {
    return el->getKind() == Kind::Literal;
  }

  /// Get the literal spelling.
  StringRef getSpelling() const { return literal; }

private:
  /// The spelling of the literal for this element.
  StringRef literal;
};

/// This class represents an instance of a variable element. A variable refers
/// to an attribute or type parameter.
class VariableElement : public Element {
public:
  VariableElement(AttrOrTypeParameter param)
      : Element(Kind::Variable), param(param) {}

  static bool classof(const Element *el) {
    return el->getKind() == Kind::Variable;
  }

  /// Get the parameter in the element.
  const AttrOrTypeParameter &getParam() const { return param; }

private:
  AttrOrTypeParameter param;
};

/// Base class for a directive that contains references to multiple variables.
template <Element::Kind ElementKind>
class ParamsDirectiveBase : public Element {
public:
  using Base = ParamsDirectiveBase<ElementKind>;

  ParamsDirectiveBase(SmallVector<std::unique_ptr<Element>> &&params)
      : Element(ElementKind), params(std::move(params)) {}

  static bool classof(const Element *el) {
    return el->getKind() == ElementKind;
  }

  /// Get the parameters contained in this directive.
  auto getParams() const {
    return llvm::map_range(params, [](auto &el) {
      return cast<VariableElement>(el.get())->getParam();
    });
  }

  /// Get the number of parameters.
  unsigned getNumParams() const { return params.size(); }

  /// Take all of the parameters from this directive.
  SmallVector<std::unique_ptr<Element>> takeParams() {
    return std::move(params);
  }

private:
  /// The parameters captured by this directive.
  SmallVector<std::unique_ptr<Element>> params;
};

/// This class represents a `params` directive that refers to all parameters
/// of an attribute or type. When used as a top-level directive, it generates
/// a format of the form:
///
///   (param-value (`,` param-value)*)?
///
/// When used as an argument to another directive that accepts variables,
/// `params` can be used in place of manually listing all parameters of an
/// attribute or type.
class ParamsDirective
    : public ParamsDirectiveBase<Element::Kind::ParamsDirective> {
public:
  using Base::Base;
};

/// This class represents a `struct` directive that generates a struct format
/// of the form:
///
///   `{` param-name `=` param-value (`,` param-name `=` param-value)* `}`
///
class StructDirective
    : public ParamsDirectiveBase<Element::Kind::StructDirective> {
public:
  using Base::Base;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Format Strings
//===----------------------------------------------------------------------===//

/// Format for defining an attribute parser.
///
/// $0: The attribute C++ class name.
static const char *const attrParserDefn = R"(
::mlir::Attribute $0::parse(::mlir::AsmParser &$_parser,
                             ::mlir::Type $_type) {
)";

/// Format for defining a type parser.
///
/// $0: The type C++ class name.
static const char *const typeParserDefn = R"(
::mlir::Type $0::parse(::mlir::AsmParser &$_parser) {
)";

/// Default parser for attribute or type parameters.
static const char *const defaultParameterParser =
    "::mlir::FieldParser<$0>::parse($_parser)";

/// Default printer for attribute or type parameters.
static const char *const defaultParameterPrinter = "$_printer << $_self";

/// Print an error when failing to parse an element.
///
/// $0: The parameter C++ class name.
static const char *const parseErrorStr =
    "$_parser.emitError($_parser.getCurrentLocation(), ";

/// Format for defining an attribute or type printer.
///
/// $0: The attribute or type C++ class name.
static const char *const attrOrTypePrinterDefn = R"(
void $0::print(::mlir::AsmPrinter &$_printer) const {
)";

/// Loop declaration for struct parser.
///
/// $0: Number of expected parameters.
static const char *const structParseLoopStart = R"(
  for (unsigned _index = 0; _index < $0; ++_index) {
    StringRef _paramKey;
    if ($_parser.parseKeyword(&_paramKey)) {
      $_parser.emitError($_parser.getCurrentLocation(),
                         "expected a parameter name in struct");
      return {};
    }
)";

/// Terminator code segment for the struct parser loop. Check for duplicate or
/// unknown parameters. Parse a comma except on the last element.
///
/// {0}: Code template for printing an error.
/// {1}: Number of elements in the struct.
static const char *const structParseLoopEnd = R"({{
      {0}"duplicate or unknown struct parameter name: ") << _paramKey;
      return {{};
    }
    if ((_index != {1} - 1) && parser.parseComma())
      return {{};
  }
)";

/// Code format to parse a variable. Separate by lines because variable parsers
/// may be generated inside other directives, which requires indentation.
///
/// {0}: The parameter name.
/// {1}: The parse code for the parameter.
/// {2}: Code template for printing an error.
/// {3}: Name of the attribute or type.
/// {4}: C++ class of the parameter.
static const char *const variableParser[] = {
    "  // Parse variable '{0}'",
    "  _result_{0} = {1};",
    "  if (failed(_result_{0})) {{",
    "    {2}\"failed to parse {3} parameter '{0}' which is to be a `{4}`\");",
    "    return {{};",
    "  }",
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Get a list of an attribute's or type's parameters. These can be wrapper
/// objects around `AttrOrTypeParameter` or string inits.
static auto getParameters(const AttrOrTypeDef &def) {
  SmallVector<AttrOrTypeParameter> params;
  def.getParameters(params);
  return params;
}

//===----------------------------------------------------------------------===//
// AttrOrTypeFormat
//===----------------------------------------------------------------------===//

namespace {
class AttrOrTypeFormat {
public:
  AttrOrTypeFormat(const AttrOrTypeDef &def,
                   std::vector<std::unique_ptr<Element>> &&elements)
      : def(def), elements(std::move(elements)) {}

  /// Generate the attribute or type parser.
  void genParser(raw_ostream &os);
  /// Generate the attribute or type printer.
  void genPrinter(raw_ostream &os);

private:
  /// Generate the parser code for a specific format element.
  void genElementParser(Element *el, FmtContext &ctx, raw_ostream &os);
  /// Generate the parser code for a literal.
  void genLiteralParser(StringRef value, FmtContext &ctx, raw_ostream &os,
                        unsigned indent = 0);
  /// Generate the parser code for a variable.
  void genVariableParser(const AttrOrTypeParameter &param, FmtContext &ctx,
                         raw_ostream &os, unsigned indent = 0);
  /// Generate the parser code for a `params` directive.
  void genParamsParser(ParamsDirective *el, FmtContext &ctx, raw_ostream &os);
  /// Generate the parser code for a `struct` directive.
  void genStructParser(StructDirective *el, FmtContext &ctx, raw_ostream &os);

  /// Generate the printer code for a specific format element.
  void genElementPrinter(Element *el, FmtContext &ctx, raw_ostream &os);
  /// Generate the printer code for a literal.
  void genLiteralPrinter(StringRef value, FmtContext &ctx, raw_ostream &os);
  /// Generate the printer code for a variable.
  void genVariablePrinter(const AttrOrTypeParameter &param, FmtContext &ctx,
                          raw_ostream &os);
  /// Generate the printer code for a `params` directive.
  void genParamsPrinter(ParamsDirective *el, FmtContext &ctx, raw_ostream &os);
  /// Generate the printer code for a `struct` directive.
  void genStructPrinter(StructDirective *el, FmtContext &ctx, raw_ostream &os);

  /// The ODS definition of the attribute or type whose format is being used to
  /// generate a parser and printer.
  const AttrOrTypeDef &def;
  /// The list of top-level format elements returned by the assembly format
  /// parser.
  std::vector<std::unique_ptr<Element>> elements;

  /// Flags for printing spaces.
  bool shouldEmitSpace;
  bool lastWasPunctuation;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ParserGen
//===----------------------------------------------------------------------===//

void AttrOrTypeFormat::genParser(raw_ostream &os) {
  FmtContext ctx;
  ctx.addSubst("_parser", "parser");

  /// Generate the definition.
  if (isa<AttrDef>(def)) {
    ctx.addSubst("_type", "attrType");
    os << tgfmt(attrParserDefn, &ctx, def.getCppClassName());
  } else {
    os << tgfmt(typeParserDefn, &ctx, def.getCppClassName());
  }

  /// Declare variables to store all of the parameters. Allocated parameters
  /// such as `ArrayRef` and `StringRef` must provide a `storageType`. Store
  /// FailureOr<T> to defer type construction for parameters that are parsed in
  /// a loop (parsers return FailureOr anyways).
  SmallVector<AttrOrTypeParameter> params = getParameters(def);
  for (const AttrOrTypeParameter &param : params) {
    os << formatv("  ::mlir::FailureOr<{0}> _result_{1};\n",
                  param.getCppStorageType(), param.getName());
  }

  /// Store the initial location of the parser.
  ctx.addSubst("_loc", "loc");
  os << tgfmt("  ::llvm::SMLoc $_loc = $_parser.getCurrentLocation();\n"
              "  (void) $_loc;\n",
              &ctx);

  /// Generate call to each parameter parser.
  for (auto &el : elements)
    genElementParser(el.get(), ctx, os);

  /// Generate call to the attribute or type builder. Use the checked getter
  /// if one was generated.
  if (def.genVerifyDecl()) {
    os << tgfmt("  return $_parser.getChecked<$0>($_loc, $_parser.getContext()",
                &ctx, def.getCppClassName());
  } else {
    os << tgfmt("  return $0::get($_parser.getContext()", &ctx,
                def.getCppClassName());
  }
  for (const AttrOrTypeParameter &param : params)
    os << formatv(",\n    _result_{0}.getValue()", param.getName());
  os << ");\n}\n\n";
}

void AttrOrTypeFormat::genElementParser(Element *el, FmtContext &ctx,
                                        raw_ostream &os) {
  if (auto *literal = dyn_cast<LiteralElement>(el))
    return genLiteralParser(literal->getSpelling(), ctx, os);
  if (auto *var = dyn_cast<VariableElement>(el))
    return genVariableParser(var->getParam(), ctx, os);
  if (auto *params = dyn_cast<ParamsDirective>(el))
    return genParamsParser(params, ctx, os);
  if (auto *strct = dyn_cast<StructDirective>(el))
    return genStructParser(strct, ctx, os);

  llvm_unreachable("unknown format element");
}

void AttrOrTypeFormat::genLiteralParser(StringRef value, FmtContext &ctx,
                                        raw_ostream &os, unsigned indent) {
  os.indent(indent) << "  // Parse literal '" << value << "'\n";
  os.indent(indent) << tgfmt("  if ($_parser.parse", &ctx);
  if (value.front() == '_' || isalpha(value.front())) {
    os << "Keyword(\"" << value << "\")";
  } else {
    os << StringSwitch<StringRef>(value)
              .Case("->", "Arrow")
              .Case(":", "Colon")
              .Case(",", "Comma")
              .Case("=", "Equal")
              .Case("<", "Less")
              .Case(">", "Greater")
              .Case("{", "LBrace")
              .Case("}", "RBrace")
              .Case("(", "LParen")
              .Case(")", "RParen")
              .Case("[", "LSquare")
              .Case("]", "RSquare")
              .Case("?", "Question")
              .Case("+", "Plus")
              .Case("*", "Star")
       << "()";
  }
  os << ")\n";
  // Parser will emit an error
  os.indent(indent) << "    return {};\n";
}

void AttrOrTypeFormat::genVariableParser(const AttrOrTypeParameter &param,
                                         FmtContext &ctx, raw_ostream &os,
                                         unsigned indent) {
  /// Check for a custom parser. Use the default attribute parser otherwise.
  auto customParser = param.getParser();
  auto parser =
      customParser ? *customParser : StringRef(defaultParameterParser);
  for (const char *line : variableParser) {
    os.indent(indent) << formatv(line, param.getName(),
                                 tgfmt(parser, &ctx, param.getCppStorageType()),
                                 tgfmt(parseErrorStr, &ctx), def.getName(),
                                 param.getCppType())
                      << "\n";
  }
}

void AttrOrTypeFormat::genParamsParser(ParamsDirective *el, FmtContext &ctx,
                                       raw_ostream &os) {
  os << "  // Parse parameter list\n";
  llvm::interleave(
      el->getParams(),
      [&](auto param) { this->genVariableParser(param, ctx, os); },
      [&]() { this->genLiteralParser(",", ctx, os); });
}

void AttrOrTypeFormat::genStructParser(StructDirective *el, FmtContext &ctx,
                                       raw_ostream &os) {
  os << "  // Parse parameter struct\n";

  /// Declare a "seen" variable for each key.
  for (const AttrOrTypeParameter &param : el->getParams())
    os << formatv("  bool _seen_{0} = false;\n", param.getName());

  /// Generate the parsing loop.
  os << tgfmt(structParseLoopStart, &ctx, el->getNumParams());
  genLiteralParser("=", ctx, os, 2);
  os << "    ";
  for (const AttrOrTypeParameter &param : el->getParams()) {
    os << formatv("if (!_seen_{0} && _paramKey == \"{0}\") {\n"
                  "      _seen_{0} = true;\n",
                  param.getName());
    genVariableParser(param, ctx, os, 4);
    os << "    } else ";
  }

  /// Duplicate or unknown parameter.
  os << formatv(structParseLoopEnd, tgfmt(parseErrorStr, &ctx),
                el->getNumParams());

  /// Because the loop loops N times and each non-failing iteration sets 1 of
  /// N flags, successfully exiting the loop means that all parameters have been
  /// seen. `parseOptionalComma` would cause issues with any formats that use
  /// "struct(...) `,`" beacuse structs aren't sounded by braces.
}

//===----------------------------------------------------------------------===//
// PrinterGen
//===----------------------------------------------------------------------===//

void AttrOrTypeFormat::genPrinter(raw_ostream &os) {
  FmtContext ctx;
  ctx.addSubst("_printer", "printer");

  /// Generate the definition.
  os << tgfmt(attrOrTypePrinterDefn, &ctx, def.getCppClassName());

  /// Generate printers.
  shouldEmitSpace = true;
  lastWasPunctuation = false;
  for (auto &el : elements)
    genElementPrinter(el.get(), ctx, os);

  os << "}\n\n";
}

void AttrOrTypeFormat::genElementPrinter(Element *el, FmtContext &ctx,
                                         raw_ostream &os) {
  if (auto *literal = dyn_cast<LiteralElement>(el))
    return genLiteralPrinter(literal->getSpelling(), ctx, os);
  if (auto *params = dyn_cast<ParamsDirective>(el))
    return genParamsPrinter(params, ctx, os);
  if (auto *strct = dyn_cast<StructDirective>(el))
    return genStructPrinter(strct, ctx, os);
  if (auto *var = dyn_cast<VariableElement>(el))
    return genVariablePrinter(var->getParam(), ctx, os);

  llvm_unreachable("unknown format element");
}

void AttrOrTypeFormat::genLiteralPrinter(StringRef value, FmtContext &ctx,
                                         raw_ostream &os) {
  /// Don't insert a space before certain punctuation.
  bool needSpace =
      shouldEmitSpace && shouldEmitSpaceBefore(value, lastWasPunctuation);
  os << tgfmt("  $_printer$0 << \"$1\";\n", &ctx, needSpace ? " << ' '" : "",
              value);

  /// Update the flags.
  shouldEmitSpace =
      value.size() != 1 || !StringRef("<({[").contains(value.front());
  lastWasPunctuation = !(value.front() == '_' || isalpha(value.front()));
}

void AttrOrTypeFormat::genVariablePrinter(const AttrOrTypeParameter &param,
                                          FmtContext &ctx, raw_ostream &os) {
  /// Insert a space before the next parameter, if necessary.
  if (shouldEmitSpace || !lastWasPunctuation)
    os << tgfmt("  $_printer << ' ';\n", &ctx);
  shouldEmitSpace = true;
  lastWasPunctuation = false;

  ctx.withSelf(getParameterAccessorName(param.getName()) + "()");
  os << "  ";
  if (auto printer = param.getPrinter())
    os << tgfmt(*printer, &ctx) << ";\n";
  else
    os << tgfmt(defaultParameterPrinter, &ctx) << ";\n";
}

void AttrOrTypeFormat::genParamsPrinter(ParamsDirective *el, FmtContext &ctx,
                                        raw_ostream &os) {
  llvm::interleave(
      el->getParams(),
      [&](auto param) { this->genVariablePrinter(param, ctx, os); },
      [&]() { this->genLiteralPrinter(",", ctx, os); });
}

void AttrOrTypeFormat::genStructPrinter(StructDirective *el, FmtContext &ctx,
                                        raw_ostream &os) {
  llvm::interleave(
      el->getParams(),
      [&](auto param) {
        this->genLiteralPrinter(param.getName(), ctx, os);
        this->genLiteralPrinter("=", ctx, os);
        os << tgfmt("  $_printer << ' ';\n", &ctx);
        this->genVariablePrinter(param, ctx, os);
      },
      [&]() { this->genLiteralPrinter(",", ctx, os); });
}

//===----------------------------------------------------------------------===//
// FormatParser
//===----------------------------------------------------------------------===//

namespace {
class FormatParser {
public:
  FormatParser(llvm::SourceMgr &mgr, const AttrOrTypeDef &def)
      : lexer(mgr, def.getLoc()[0]), curToken(lexer.lexToken()), def(def),
        seenParams(def.getNumParameters()) {}

  /// Parse the attribute or type format and create the format elements.
  FailureOr<AttrOrTypeFormat> parse();

private:
  /// The current context of the parser when parsing an element.
  enum ParserContext {
    /// The element is being parsed in the default context - at the top of the
    /// format
    TopLevelContext,
    /// The element is being parsed as a child to a `struct` directive.
    StructDirective,
  };

  /// Emit an error.
  LogicalResult emitError(const Twine &msg) {
    lexer.emitError(curToken.getLoc(), msg);
    return failure();
  }

  /// Parse an expected token.
  LogicalResult parseToken(FormatToken::Kind kind, const Twine &msg) {
    if (curToken.getKind() != kind)
      return emitError(msg);
    consumeToken();
    return success();
  }

  /// Advance the lexer to the next token.
  void consumeToken() {
    assert(curToken.getKind() != FormatToken::eof &&
           curToken.getKind() != FormatToken::error &&
           "shouldn't advance past EOF or errors");
    curToken = lexer.lexToken();
  }

  /// Parse any element.
  FailureOr<std::unique_ptr<Element>> parseElement(ParserContext ctx);
  /// Parse a literal element.
  FailureOr<std::unique_ptr<Element>> parseLiteral(ParserContext ctx);
  /// Parse a variable element.
  FailureOr<std::unique_ptr<Element>> parseVariable(ParserContext ctx);
  /// Parse a directive.
  FailureOr<std::unique_ptr<Element>> parseDirective(ParserContext ctx);
  /// Parse a `params` directive.
  FailureOr<std::unique_ptr<Element>> parseParamsDirective();
  /// Parse a `struct` directive.
  FailureOr<std::unique_ptr<Element>> parseStructDirective();

  /// The current format lexer.
  FormatLexer lexer;
  /// The current token in the stream.
  FormatToken curToken;
  /// Attribute or type tablegen def.
  const AttrOrTypeDef &def;

  /// Seen attribute or type parameters.
  llvm::BitVector seenParams;
};
} // end anonymous namespace

FailureOr<AttrOrTypeFormat> FormatParser::parse() {
  std::vector<std::unique_ptr<Element>> elements;
  elements.reserve(16);

  /// Parse the format elements.
  while (curToken.getKind() != FormatToken::eof) {
    auto element = parseElement(TopLevelContext);
    if (failed(element))
      return failure();

    /// Add the format element and continue.
    elements.push_back(std::move(*element));
  }

  /// Check that all parameters have been seen.
  SmallVector<AttrOrTypeParameter> params = getParameters(def);
  for (auto it : llvm::enumerate(params)) {
    if (!seenParams.test(it.index())) {
      return emitError("format is missing reference to parameter: " +
                       it.value().getName());
    }
  }

  return AttrOrTypeFormat(def, std::move(elements));
}

FailureOr<std::unique_ptr<Element>>
FormatParser::parseElement(ParserContext ctx) {
  if (curToken.getKind() == FormatToken::literal)
    return parseLiteral(ctx);
  if (curToken.getKind() == FormatToken::variable)
    return parseVariable(ctx);
  if (curToken.isKeyword())
    return parseDirective(ctx);

  return emitError("expected literal, directive, or variable");
}

FailureOr<std::unique_ptr<Element>>
FormatParser::parseLiteral(ParserContext ctx) {
  if (ctx != TopLevelContext) {
    return emitError(
        "literals may only be used in the top-level section of the format");
  }

  /// Get the literal spelling without the surrounding "`".
  auto value = curToken.getSpelling().drop_front().drop_back();
  if (!isValidLiteral(value))
    return emitError("literal '" + value + "' is not valid");

  consumeToken();
  return {std::make_unique<LiteralElement>(value)};
}

FailureOr<std::unique_ptr<Element>>
FormatParser::parseVariable(ParserContext ctx) {
  /// Get the parameter name without the preceding "$".
  auto name = curToken.getSpelling().drop_front();

  /// Lookup the parameter.
  SmallVector<AttrOrTypeParameter> params = getParameters(def);
  auto *it = llvm::find_if(
      params, [&](auto &param) { return param.getName() == name; });

  /// Check that the parameter reference is valid.
  if (it == params.end())
    return emitError(def.getName() + " has no parameter named '" + name + "'");
  auto idx = std::distance(params.begin(), it);
  if (seenParams.test(idx))
    return emitError("duplicate parameter '" + name + "'");
  seenParams.set(idx);

  consumeToken();
  return {std::make_unique<VariableElement>(*it)};
}

FailureOr<std::unique_ptr<Element>>
FormatParser::parseDirective(ParserContext ctx) {

  switch (curToken.getKind()) {
  case FormatToken::kw_params:
    return parseParamsDirective();
  case FormatToken::kw_struct:
    if (ctx != TopLevelContext) {
      return emitError(
          "`struct` may only be used in the top-level section of the format");
    }
    return parseStructDirective();
  default:
    return emitError("unknown directive in format: " + curToken.getSpelling());
  }
}

FailureOr<std::unique_ptr<Element>> FormatParser::parseParamsDirective() {
  consumeToken();
  /// Collect all of the attribute's or type's parameters.
  SmallVector<AttrOrTypeParameter> params = getParameters(def);
  SmallVector<std::unique_ptr<Element>> vars;
  /// Ensure that none of the parameters have already been captured.
  for (auto it : llvm::enumerate(params)) {
    if (seenParams.test(it.index())) {
      return emitError("`params` captures duplicate parameter: " +
                       it.value().getName());
    }
    seenParams.set(it.index());
    vars.push_back(std::make_unique<VariableElement>(it.value()));
  }
  return {std::make_unique<ParamsDirective>(std::move(vars))};
}

FailureOr<std::unique_ptr<Element>> FormatParser::parseStructDirective() {
  consumeToken();
  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before `struct` argument list")))
    return failure();

  /// Parse variables captured by `struct`.
  SmallVector<std::unique_ptr<Element>> vars;

  /// Parse first captured parameter or a `params` directive.
  FailureOr<std::unique_ptr<Element>> var = parseElement(StructDirective);
  if (failed(var) || !isa<VariableElement, ParamsDirective>(*var))
    return emitError("`struct` argument list expected a variable or directive");
  if (isa<VariableElement>(*var)) {
    /// Parse any other parameters.
    vars.push_back(std::move(*var));
    while (curToken.getKind() == FormatToken::comma) {
      consumeToken();
      var = parseElement(StructDirective);
      if (failed(var) || !isa<VariableElement>(*var))
        return emitError("expected a variable in `struct` argument list");
      vars.push_back(std::move(*var));
    }
  } else {
    /// `struct(params)` captures all parameters in the attribute or type.
    vars = cast<ParamsDirective>(var->get())->takeParams();
  }

  if (curToken.getKind() != FormatToken::r_paren)
    return emitError("expected ')' at the end of an argument list");

  consumeToken();
  return {std::make_unique<::StructDirective>(std::move(vars))};
}

//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateAttrOrTypeFormat(const AttrOrTypeDef &def,
                                            raw_ostream &os) {
  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(*def.getAssemblyFormat()),
      llvm::SMLoc());

  /// Parse the custom assembly format>
  FormatParser parser(mgr, def);
  FailureOr<AttrOrTypeFormat> format = parser.parse();
  if (failed(format)) {
    if (formatErrorIsFatal)
      PrintFatalError(def.getLoc(), "failed to parse assembly format");
    return;
  }

  /// Generate the parser and printer.
  format->genParser(os);
  format->genPrinter(os);
}
