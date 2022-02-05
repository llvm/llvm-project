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
#include "llvm/ADT/StringSwitch.h"
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
/// This class represents an instance of a variable element. A variable refers
/// to an attribute or type parameter.
class ParameterElement
    : public VariableElementBase<VariableElement::Parameter> {
public:
  ParameterElement(AttrOrTypeParameter param) : param(param) {}

  /// Get the parameter in the element.
  const AttrOrTypeParameter &getParam() const { return param; }

  /// Indicate if this variable is printed "qualified" (that is it is
  /// prefixed with the `#dialect.mnemonic`).
  bool shouldBeQualified() { return shouldBeQualifiedFlag; }
  void setShouldBeQualified(bool qualified = true) {
    shouldBeQualifiedFlag = qualified;
  }

private:
  bool shouldBeQualifiedFlag = false;
  AttrOrTypeParameter param;
};

/// Base class for a directive that contains references to multiple variables.
template <DirectiveElement::Kind DirectiveKind>
class ParamsDirectiveBase : public DirectiveElementBase<DirectiveKind> {
public:
  using Base = ParamsDirectiveBase<DirectiveKind>;

  ParamsDirectiveBase(std::vector<FormatElement *> &&params)
      : params(std::move(params)) {}

  /// Get the parameters contained in this directive.
  auto getParams() const {
    return llvm::map_range(params, [](FormatElement *el) {
      return cast<ParameterElement>(el)->getParam();
    });
  }

  /// Get the number of parameters.
  unsigned getNumParams() const { return params.size(); }

  /// Take all of the parameters from this directive.
  std::vector<FormatElement *> takeParams() { return std::move(params); }

private:
  /// The parameters captured by this directive.
  std::vector<FormatElement *> params;
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
class ParamsDirective : public ParamsDirectiveBase<DirectiveElement::Params> {
public:
  using Base::Base;
};

/// This class represents a `struct` directive that generates a struct format
/// of the form:
///
///   `{` param-name `=` param-value (`,` param-name `=` param-value)* `}`
///
class StructDirective : public ParamsDirectiveBase<DirectiveElement::Struct> {
public:
  using Base::Base;
};

} // namespace

//===----------------------------------------------------------------------===//
// Format Strings
//===----------------------------------------------------------------------===//

/// Default parser for attribute or type parameters.
static const char *const defaultParameterParser =
    "::mlir::FieldParser<$0>::parse($_parser)";

/// Default printer for attribute or type parameters.
static const char *const defaultParameterPrinter =
    "$_printer.printStrippedAttrOrType($_self)";

/// Qualified printer for attribute or type parameters: it does not elide
/// dialect and mnemonic.
static const char *const qualifiedParameterPrinter = "$_printer << $_self";

/// Print an error when failing to parse an element.
///
/// $0: The parameter C++ class name.
static const char *const parseErrorStr =
    "$_parser.emitError($_parser.getCurrentLocation(), ";

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
static const char *const variableParser = R"(
// Parse variable '{0}'
_result_{0} = {1};
if (failed(_result_{0})) {{
  {2}"failed to parse {3} parameter '{0}' which is to be a `{4}`");
  return {{};
}
)";

//===----------------------------------------------------------------------===//
// AttrOrTypeFormat
//===----------------------------------------------------------------------===//

namespace {
class AttrOrTypeFormat {
public:
  AttrOrTypeFormat(const AttrOrTypeDef &def,
                   std::vector<FormatElement *> &&elements)
      : def(def), elements(std::move(elements)) {}

  /// Generate the attribute or type parser.
  void genParser(MethodBody &os);
  /// Generate the attribute or type printer.
  void genPrinter(MethodBody &os);

private:
  /// Generate the parser code for a specific format element.
  void genElementParser(FormatElement *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a literal.
  void genLiteralParser(StringRef value, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a variable.
  void genVariableParser(const AttrOrTypeParameter &param, FmtContext &ctx,
                         MethodBody &os);
  /// Generate the parser code for a `params` directive.
  void genParamsParser(ParamsDirective *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a `struct` directive.
  void genStructParser(StructDirective *el, FmtContext &ctx, MethodBody &os);

  /// Generate the printer code for a specific format element.
  void genElementPrinter(FormatElement *el, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for a literal.
  void genLiteralPrinter(StringRef value, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for a variable.
  void genVariablePrinter(const AttrOrTypeParameter &param, FmtContext &ctx,
                          MethodBody &os, bool printQualified = false);
  /// Generate the printer code for a `params` directive.
  void genParamsPrinter(ParamsDirective *el, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for a `struct` directive.
  void genStructPrinter(StructDirective *el, FmtContext &ctx, MethodBody &os);

  /// The ODS definition of the attribute or type whose format is being used to
  /// generate a parser and printer.
  const AttrOrTypeDef &def;
  /// The list of top-level format elements returned by the assembly format
  /// parser.
  std::vector<FormatElement *> elements;

  /// Flags for printing spaces.
  bool shouldEmitSpace = false;
  bool lastWasPunctuation = false;
};
} // namespace

//===----------------------------------------------------------------------===//
// ParserGen
//===----------------------------------------------------------------------===//

void AttrOrTypeFormat::genParser(MethodBody &os) {
  FmtContext ctx;
  ctx.addSubst("_parser", "parser");
  if (isa<AttrDef>(def))
    ctx.addSubst("_type", "type");
  os.indent();

  /// Declare variables to store all of the parameters. Allocated parameters
  /// such as `ArrayRef` and `StringRef` must provide a `storageType`. Store
  /// FailureOr<T> to defer type construction for parameters that are parsed in
  /// a loop (parsers return FailureOr anyways).
  ArrayRef<AttrOrTypeParameter> params = def.getParameters();
  for (const AttrOrTypeParameter &param : params) {
    os << formatv("  ::mlir::FailureOr<{0}> _result_{1};\n",
                  param.getCppStorageType(), param.getName());
  }

  /// Store the initial location of the parser.
  ctx.addSubst("_loc", "loc");
  os << tgfmt("::llvm::SMLoc $_loc = $_parser.getCurrentLocation();\n"
              "(void) $_loc;\n",
              &ctx);

  /// Generate call to each parameter parser.
  for (FormatElement *el : elements)
    genElementParser(el, ctx, os);

  /// Generate call to the attribute or type builder. Use the checked getter
  /// if one was generated.
  if (def.genVerifyDecl()) {
    os << tgfmt("return $_parser.getChecked<$0>($_loc, $_parser.getContext()",
                &ctx, def.getCppClassName());
  } else {
    os << tgfmt("return $0::get($_parser.getContext()", &ctx,
                def.getCppClassName());
  }
  for (const AttrOrTypeParameter &param : params)
    os << formatv(",\n    _result_{0}.getValue()", param.getName());
  os << ");";
}

void AttrOrTypeFormat::genElementParser(FormatElement *el, FmtContext &ctx,
                                        MethodBody &os) {
  if (auto *literal = dyn_cast<LiteralElement>(el))
    return genLiteralParser(literal->getSpelling(), ctx, os);
  if (auto *var = dyn_cast<ParameterElement>(el))
    return genVariableParser(var->getParam(), ctx, os);
  if (auto *params = dyn_cast<ParamsDirective>(el))
    return genParamsParser(params, ctx, os);
  if (auto *strct = dyn_cast<StructDirective>(el))
    return genStructParser(strct, ctx, os);

  llvm_unreachable("unknown format element");
}

void AttrOrTypeFormat::genLiteralParser(StringRef value, FmtContext &ctx,
                                        MethodBody &os) {
  os << "// Parse literal '" << value << "'\n";
  os << tgfmt("if ($_parser.parse", &ctx);
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
  os << "  return {};\n";
}

void AttrOrTypeFormat::genVariableParser(const AttrOrTypeParameter &param,
                                         FmtContext &ctx, MethodBody &os) {
  /// Check for a custom parser. Use the default attribute parser otherwise.
  auto customParser = param.getParser();
  auto parser =
      customParser ? *customParser : StringRef(defaultParameterParser);
  os << formatv(variableParser, param.getName(),
                tgfmt(parser, &ctx, param.getCppStorageType()),
                tgfmt(parseErrorStr, &ctx), def.getName(), param.getCppType());
}

void AttrOrTypeFormat::genParamsParser(ParamsDirective *el, FmtContext &ctx,
                                       MethodBody &os) {
  os << "// Parse parameter list\n";
  llvm::interleave(
      el->getParams(),
      [&](auto param) { this->genVariableParser(param, ctx, os); },
      [&]() { this->genLiteralParser(",", ctx, os); });
}

void AttrOrTypeFormat::genStructParser(StructDirective *el, FmtContext &ctx,
                                       MethodBody &os) {
  os << "// Parse parameter struct\n";

  /// Declare a "seen" variable for each key.
  for (const AttrOrTypeParameter &param : el->getParams())
    os << formatv("bool _seen_{0} = false;\n", param.getName());

  /// Generate the parsing loop.
  os.getStream().printReindented(
      tgfmt(structParseLoopStart, &ctx, el->getNumParams()).str());
  os.indent();
  genLiteralParser("=", ctx, os);
  for (const AttrOrTypeParameter &param : el->getParams()) {
    os << formatv("if (!_seen_{0} && _paramKey == \"{0}\") {\n"
                  "  _seen_{0} = true;\n",
                  param.getName());
    genVariableParser(param, ctx, os.indent());
    os.unindent() << "} else ";
  }
  os.unindent();

  /// Duplicate or unknown parameter.
  os.getStream().printReindented(strfmt(
      structParseLoopEnd, tgfmt(parseErrorStr, &ctx), el->getNumParams()));

  /// Because the loop loops N times and each non-failing iteration sets 1 of
  /// N flags, successfully exiting the loop means that all parameters have been
  /// seen. `parseOptionalComma` would cause issues with any formats that use
  /// "struct(...) `,`" beacuse structs aren't sounded by braces.
}

//===----------------------------------------------------------------------===//
// PrinterGen
//===----------------------------------------------------------------------===//

void AttrOrTypeFormat::genPrinter(MethodBody &os) {
  FmtContext ctx;
  ctx.addSubst("_printer", "printer");

  /// Generate printers.
  shouldEmitSpace = true;
  lastWasPunctuation = false;
  for (FormatElement *el : elements)
    genElementPrinter(el, ctx, os);
}

void AttrOrTypeFormat::genElementPrinter(FormatElement *el, FmtContext &ctx,
                                         MethodBody &os) {
  if (auto *literal = dyn_cast<LiteralElement>(el))
    return genLiteralPrinter(literal->getSpelling(), ctx, os);
  if (auto *params = dyn_cast<ParamsDirective>(el))
    return genParamsPrinter(params, ctx, os);
  if (auto *strct = dyn_cast<StructDirective>(el))
    return genStructPrinter(strct, ctx, os);
  if (auto *var = dyn_cast<ParameterElement>(el))
    return genVariablePrinter(var->getParam(), ctx, os,
                              var->shouldBeQualified());

  llvm_unreachable("unknown format element");
}

void AttrOrTypeFormat::genLiteralPrinter(StringRef value, FmtContext &ctx,
                                         MethodBody &os) {
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
                                          FmtContext &ctx, MethodBody &os,
                                          bool printQualified) {
  /// Insert a space before the next parameter, if necessary.
  if (shouldEmitSpace || !lastWasPunctuation)
    os << tgfmt("  $_printer << ' ';\n", &ctx);
  shouldEmitSpace = true;
  lastWasPunctuation = false;

  ctx.withSelf(getParameterAccessorName(param.getName()) + "()");
  os << "  ";
  if (printQualified)
    os << tgfmt(qualifiedParameterPrinter, &ctx) << ";\n";
  else if (auto printer = param.getPrinter())
    os << tgfmt(*printer, &ctx) << ";\n";
  else
    os << tgfmt(defaultParameterPrinter, &ctx) << ";\n";
}

void AttrOrTypeFormat::genParamsPrinter(ParamsDirective *el, FmtContext &ctx,
                                        MethodBody &os) {
  llvm::interleave(
      el->getParams(),
      [&](auto param) { this->genVariablePrinter(param, ctx, os); },
      [&] { this->genLiteralPrinter(",", ctx, os); });
}

void AttrOrTypeFormat::genStructPrinter(StructDirective *el, FmtContext &ctx,
                                        MethodBody &os) {
  llvm::interleave(
      el->getParams(),
      [&](auto param) {
        this->genLiteralPrinter(param.getName(), ctx, os);
        this->genLiteralPrinter("=", ctx, os);
        this->genVariablePrinter(param, ctx, os);
      },
      [&] { this->genLiteralPrinter(",", ctx, os); });
}

//===----------------------------------------------------------------------===//
// DefFormatParser
//===----------------------------------------------------------------------===//

namespace {
class DefFormatParser : public FormatParser {
public:
  DefFormatParser(llvm::SourceMgr &mgr, const AttrOrTypeDef &def)
      : FormatParser(mgr, def.getLoc()[0]), def(def),
        seenParams(def.getNumParameters()) {}

  /// Parse the attribute or type format and create the format elements.
  FailureOr<AttrOrTypeFormat> parse();

protected:
  /// Verify the parsed elements.
  LogicalResult verify(SMLoc loc, ArrayRef<FormatElement *> elements) override;
  /// Verify the elements of a custom directive.
  LogicalResult
  verifyCustomDirectiveArguments(SMLoc loc,
                                 ArrayRef<FormatElement *> arguments) override {
    return emitError(loc, "'custom' not supported (yet)");
  }
  /// Verify the elements of an optional group.
  LogicalResult
  verifyOptionalGroupElements(SMLoc loc, ArrayRef<FormatElement *> elements,
                              Optional<unsigned> anchorIndex) override {
    return emitError(loc, "optional groups not (yet) supported");
  }

  /// Parse an attribute or type variable.
  FailureOr<FormatElement *> parseVariableImpl(SMLoc loc, StringRef name,
                                               Context ctx) override;
  /// Parse an attribute or type format directive.
  FailureOr<FormatElement *>
  parseDirectiveImpl(SMLoc loc, FormatToken::Kind kind, Context ctx) override;

private:
  /// Parse a `params` directive.
  FailureOr<FormatElement *> parseParamsDirective(SMLoc loc);
  /// Parse a `qualified` directive.
  FailureOr<FormatElement *> parseQualifiedDirective(SMLoc loc, Context ctx);
  /// Parse a `struct` directive.
  FailureOr<FormatElement *> parseStructDirective(SMLoc loc);

  /// Attribute or type tablegen def.
  const AttrOrTypeDef &def;

  /// Seen attribute or type parameters.
  BitVector seenParams;
};
} // namespace

LogicalResult DefFormatParser::verify(SMLoc loc,
                                      ArrayRef<FormatElement *> elements) {
  for (auto &it : llvm::enumerate(def.getParameters())) {
    if (!seenParams.test(it.index())) {
      return emitError(loc, "format is missing reference to parameter: " +
                                it.value().getName());
    }
  }
  return success();
}

FailureOr<AttrOrTypeFormat> DefFormatParser::parse() {
  FailureOr<std::vector<FormatElement *>> elements = FormatParser::parse();
  if (failed(elements))
    return failure();
  return AttrOrTypeFormat(def, std::move(*elements));
}

FailureOr<FormatElement *>
DefFormatParser::parseVariableImpl(SMLoc loc, StringRef name, Context ctx) {
  /// Lookup the parameter.
  ArrayRef<AttrOrTypeParameter> params = def.getParameters();
  auto *it = llvm::find_if(
      params, [&](auto &param) { return param.getName() == name; });

  /// Check that the parameter reference is valid.
  if (it == params.end()) {
    return emitError(loc,
                     def.getName() + " has no parameter named '" + name + "'");
  }
  auto idx = std::distance(params.begin(), it);
  if (seenParams.test(idx))
    return emitError(loc, "duplicate parameter '" + name + "'");
  seenParams.set(idx);

  return create<ParameterElement>(*it);
}

FailureOr<FormatElement *>
DefFormatParser::parseDirectiveImpl(SMLoc loc, FormatToken::Kind kind,
                                    Context ctx) {

  switch (kind) {
  case FormatToken::kw_qualified:
    return parseQualifiedDirective(loc, ctx);
  case FormatToken::kw_params:
    return parseParamsDirective(loc);
  case FormatToken::kw_struct:
    if (ctx != TopLevelContext) {
      return emitError(
          loc,
          "`struct` may only be used in the top-level section of the format");
    }
    return parseStructDirective(loc);

  default:
    return emitError(loc, "unsupported directive kind");
  }
}

FailureOr<FormatElement *>
DefFormatParser::parseQualifiedDirective(SMLoc loc, Context ctx) {
  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before argument list")))
    return failure();
  FailureOr<FormatElement *> var = parseElement(ctx);
  if (failed(var))
    return var;
  if (!isa<ParameterElement>(*var))
    return emitError(loc, "`qualified` argument list expected a variable");
  cast<ParameterElement>(*var)->setShouldBeQualified();
  if (failed(
          parseToken(FormatToken::r_paren, "expected ')' after argument list")))
    return failure();
  return var;
}

FailureOr<FormatElement *> DefFormatParser::parseParamsDirective(SMLoc loc) {
  /// Collect all of the attribute's or type's parameters.
  std::vector<FormatElement *> vars;
  /// Ensure that none of the parameters have already been captured.
  for (const auto &it : llvm::enumerate(def.getParameters())) {
    if (seenParams.test(it.index())) {
      return emitError(loc, "`params` captures duplicate parameter: " +
                                it.value().getName());
    }
    seenParams.set(it.index());
    vars.push_back(create<ParameterElement>(it.value()));
  }
  return create<ParamsDirective>(std::move(vars));
}

FailureOr<FormatElement *> DefFormatParser::parseStructDirective(SMLoc loc) {
  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before `struct` argument list")))
    return failure();

  /// Parse variables captured by `struct`.
  std::vector<FormatElement *> vars;

  /// Parse first captured parameter or a `params` directive.
  FailureOr<FormatElement *> var = parseElement(StructDirectiveContext);
  if (failed(var) || !isa<VariableElement, ParamsDirective>(*var)) {
    return emitError(loc,
                     "`struct` argument list expected a variable or directive");
  }
  if (isa<VariableElement>(*var)) {
    /// Parse any other parameters.
    vars.push_back(std::move(*var));
    while (peekToken().is(FormatToken::comma)) {
      consumeToken();
      var = parseElement(StructDirectiveContext);
      if (failed(var) || !isa<VariableElement>(*var))
        return emitError(loc, "expected a variable in `struct` argument list");
      vars.push_back(std::move(*var));
    }
  } else {
    /// `struct(params)` captures all parameters in the attribute or type.
    vars = cast<ParamsDirective>(*var)->takeParams();
  }

  if (failed(parseToken(FormatToken::r_paren,
                        "expected ')' at the end of an argument list")))
    return failure();

  return create<StructDirective>(std::move(vars));
}

//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateAttrOrTypeFormat(const AttrOrTypeDef &def,
                                            MethodBody &parser,
                                            MethodBody &printer) {
  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(*def.getAssemblyFormat()), SMLoc());

  /// Parse the custom assembly format>
  DefFormatParser fmtParser(mgr, def);
  FailureOr<AttrOrTypeFormat> format = fmtParser.parse();
  if (failed(format)) {
    if (formatErrorIsFatal)
      PrintFatalError(def.getLoc(), "failed to parse assembly format");
    return;
  }

  /// Generate the parser and printer.
  format->genParser(parser);
  format->genPrinter(printer);
}
