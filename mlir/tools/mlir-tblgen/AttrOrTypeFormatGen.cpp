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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SaveAndRestore.h"
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

  /// Returns true if the element contains an optional parameter.
  bool isOptional() const { return param.isOptional(); }

  /// Returns the name of the parameter.
  StringRef getName() const { return param.getName(); }

  /// Generate the code to check whether the parameter should be printed.
  MethodBody &genPrintGuard(FmtContext &ctx, MethodBody &os) const {
    std::string self = param.getAccessorName() + "()";
    ctx.withSelf(self);
    os << tgfmt("($_self", &ctx);
    if (llvm::Optional<StringRef> defaultValue = getParam().getDefaultValue()) {
      // Use the `comparator` field if it exists, else the equality operator.
      std::string valueStr = tgfmt(*defaultValue, &ctx).str();
      ctx.addSubst("_lhs", self).addSubst("_rhs", valueStr);
      os << " && !(" << tgfmt(getParam().getComparator(), &ctx) << ")";
    }
    return os << ")";
  }

private:
  bool shouldBeQualifiedFlag = false;
  AttrOrTypeParameter param;
};

/// Shorthand functions that can be used with ranged-based conditions.
static bool paramIsOptional(ParameterElement *el) { return el->isOptional(); }
static bool paramNotOptional(ParameterElement *el) { return !el->isOptional(); }

/// Base class for a directive that contains references to multiple variables.
template <DirectiveElement::Kind DirectiveKind>
class ParamsDirectiveBase : public DirectiveElementBase<DirectiveKind> {
public:
  using Base = ParamsDirectiveBase<DirectiveKind>;

  ParamsDirectiveBase(std::vector<ParameterElement *> &&params)
      : params(std::move(params)) {}

  /// Get the parameters contained in this directive.
  ArrayRef<ParameterElement *> getParams() const { return params; }

  /// Get the number of parameters.
  unsigned getNumParams() const { return params.size(); }

  /// Take all of the parameters from this directive.
  std::vector<ParameterElement *> takeParams() { return std::move(params); }

  /// Returns true if there are optional parameters present.
  bool hasOptionalParams() const {
    return llvm::any_of(getParams(), paramIsOptional);
  }

private:
  /// The parameters captured by this directive.
  std::vector<ParameterElement *> params;
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
static const char *const parserErrorStr =
    "$_parser.emitError($_parser.getCurrentLocation(), ";

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
if (::mlir::failed(_result_{0})) {{
  {2}"failed to parse {3} parameter '{0}' which is to be a `{4}`");
  return {{};
}
)";

//===----------------------------------------------------------------------===//
// DefFormat
//===----------------------------------------------------------------------===//

namespace {
class DefFormat {
public:
  DefFormat(const AttrOrTypeDef &def, std::vector<FormatElement *> &&elements)
      : def(def), elements(std::move(elements)) {}

  /// Generate the attribute or type parser.
  void genParser(MethodBody &os);
  /// Generate the attribute or type printer.
  void genPrinter(MethodBody &os);

private:
  /// Generate the parser code for a specific format element.
  void genElementParser(FormatElement *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a literal.
  void genLiteralParser(StringRef value, FmtContext &ctx, MethodBody &os,
                        bool isOptional = false);
  /// Generate the parser code for a variable.
  void genVariableParser(ParameterElement *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a `params` directive.
  void genParamsParser(ParamsDirective *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a `struct` directive.
  void genStructParser(StructDirective *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for a `custom` directive.
  void genCustomParser(CustomDirective *el, FmtContext &ctx, MethodBody &os);
  /// Generate the parser code for an optional group.
  void genOptionalGroupParser(OptionalElement *el, FmtContext &ctx,
                              MethodBody &os);

  /// Generate the printer code for a specific format element.
  void genElementPrinter(FormatElement *el, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for a literal.
  void genLiteralPrinter(StringRef value, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for a variable.
  void genVariablePrinter(ParameterElement *el, FmtContext &ctx, MethodBody &os,
                          bool skipGuard = false);
  /// Generate a printer for comma-separated parameters.
  void genCommaSeparatedPrinter(ArrayRef<ParameterElement *> params,
                                FmtContext &ctx, MethodBody &os,
                                function_ref<void(ParameterElement *)> extra);
  /// Generate the printer code for a `params` directive.
  void genParamsPrinter(ParamsDirective *el, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for a `struct` directive.
  void genStructPrinter(StructDirective *el, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for a `custom` directive.
  void genCustomPrinter(CustomDirective *el, FmtContext &ctx, MethodBody &os);
  /// Generate the printer code for an optional group.
  void genOptionalGroupPrinter(OptionalElement *el, FmtContext &ctx,
                               MethodBody &os);
  /// Generate a printer (or space eraser) for a whitespace element.
  void genWhitespacePrinter(WhitespaceElement *el, FmtContext &ctx,
                            MethodBody &os);

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

/// Generate a special-case "parser" for an attribute's self type parameter. The
/// self type parameter has special handling in the assembly format in that it
/// is derived from the optional trailing colon type after the attribute.
static void genAttrSelfTypeParser(MethodBody &os, const FmtContext &ctx,
                                  const AttributeSelfTypeParameter &param) {
  // "Parser" for an attribute self type parameter that checks the
  // optionally-parsed trailing colon type.
  //
  // $0: The C++ storage class of the type parameter.
  // $1: The self type parameter name.
  const char *const selfTypeParser = R"(
if ($_type) {
  if (auto reqType = $_type.dyn_cast<$0>()) {
    _result_$1 = reqType;
  } else {
    $_parser.emitError($_loc, "invalid kind of type specified");
    return {};
  }
})";

  // If the attribute self type parameter is required, emit code that emits an
  // error if the trailing type was not parsed.
  const char *const selfTypeRequired = R"( else {
  $_parser.emitError($_loc, "expected a trailing type");
  return {};
})";

  os << tgfmt(selfTypeParser, &ctx, param.getCppStorageType(), param.getName());
  if (!param.isOptional())
    os << tgfmt(selfTypeRequired, &ctx);
  os << "\n";
}

void DefFormat::genParser(MethodBody &os) {
  FmtContext ctx;
  ctx.addSubst("_parser", "odsParser");
  ctx.addSubst("_ctxt", "odsParser.getContext()");
  ctx.withBuilder("odsBuilder");
  if (isa<AttrDef>(def))
    ctx.addSubst("_type", "odsType");
  os.indent();
  os << "::mlir::Builder odsBuilder(odsParser.getContext());\n";

  // Declare variables to store all of the parameters. Allocated parameters
  // such as `ArrayRef` and `StringRef` must provide a `storageType`. Store
  // FailureOr<T> to defer type construction for parameters that are parsed in
  // a loop (parsers return FailureOr anyways).
  ArrayRef<AttrOrTypeParameter> params = def.getParameters();
  for (const AttrOrTypeParameter &param : params) {
    os << formatv("::mlir::FailureOr<{0}> _result_{1};\n",
                  param.getCppStorageType(), param.getName());
  }

  // Store the initial location of the parser.
  ctx.addSubst("_loc", "odsLoc");
  os << tgfmt("::llvm::SMLoc $_loc = $_parser.getCurrentLocation();\n"
              "(void) $_loc;\n",
              &ctx);

  // Generate call to each parameter parser.
  for (FormatElement *el : elements)
    genElementParser(el, ctx, os);

  // Emit an assert for each mandatory parameter. Triggering an assert means
  // the generated parser is incorrect (i.e. there is a bug in this code).
  for (const AttrOrTypeParameter &param : params) {
    if (auto *selfTypeParam = dyn_cast<AttributeSelfTypeParameter>(&param))
      genAttrSelfTypeParser(os, ctx, *selfTypeParam);
    if (param.isOptional())
      continue;
    os << formatv("assert(::mlir::succeeded(_result_{0}));\n", param.getName());
  }

  // Generate call to the attribute or type builder. Use the checked getter
  // if one was generated.
  if (def.genVerifyDecl()) {
    os << tgfmt("return $_parser.getChecked<$0>($_loc, $_parser.getContext()",
                &ctx, def.getCppClassName());
  } else {
    os << tgfmt("return $0::get($_parser.getContext()", &ctx,
                def.getCppClassName());
  }
  for (const AttrOrTypeParameter &param : params) {
    os << ",\n    ";
    std::string paramSelfStr;
    llvm::raw_string_ostream selfOs(paramSelfStr);
    if (param.isOptional()) {
      selfOs << formatv("(_result_{0}.value_or(", param.getName());
      if (Optional<StringRef> defaultValue = param.getDefaultValue())
        selfOs << tgfmt(*defaultValue, &ctx);
      else
        selfOs << param.getCppStorageType() << "()";
      selfOs << "))";
    } else {
      selfOs << formatv("(*_result_{0})", param.getName());
    }
    ctx.addSubst(param.getName(), selfOs.str());
    os << param.getCppType() << "("
       << tgfmt(param.getConvertFromStorage(), &ctx.withSelf(selfOs.str()))
       << ")";
  }
  os << ");";
}

void DefFormat::genElementParser(FormatElement *el, FmtContext &ctx,
                                 MethodBody &os) {
  if (auto *literal = dyn_cast<LiteralElement>(el))
    return genLiteralParser(literal->getSpelling(), ctx, os);
  if (auto *var = dyn_cast<ParameterElement>(el))
    return genVariableParser(var, ctx, os);
  if (auto *params = dyn_cast<ParamsDirective>(el))
    return genParamsParser(params, ctx, os);
  if (auto *strct = dyn_cast<StructDirective>(el))
    return genStructParser(strct, ctx, os);
  if (auto *custom = dyn_cast<CustomDirective>(el))
    return genCustomParser(custom, ctx, os);
  if (auto *optional = dyn_cast<OptionalElement>(el))
    return genOptionalGroupParser(optional, ctx, os);
  if (isa<WhitespaceElement>(el))
    return;

  llvm_unreachable("unknown format element");
}

void DefFormat::genLiteralParser(StringRef value, FmtContext &ctx,
                                 MethodBody &os, bool isOptional) {
  os << "// Parse literal '" << value << "'\n";
  os << tgfmt("if ($_parser.parse", &ctx);
  if (isOptional)
    os << "Optional";
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
  if (isOptional) {
    // Leave the `if` unclosed to guard optional groups.
    return;
  }
  // Parser will emit an error
  os << ") return {};\n";
}

void DefFormat::genVariableParser(ParameterElement *el, FmtContext &ctx,
                                  MethodBody &os) {
  // Check for a custom parser. Use the default attribute parser otherwise.
  const AttrOrTypeParameter &param = el->getParam();
  auto customParser = param.getParser();
  auto parser =
      customParser ? *customParser : StringRef(defaultParameterParser);
  os << formatv(variableParser, param.getName(),
                tgfmt(parser, &ctx, param.getCppStorageType()),
                tgfmt(parserErrorStr, &ctx), def.getName(), param.getCppType());
}

void DefFormat::genParamsParser(ParamsDirective *el, FmtContext &ctx,
                                MethodBody &os) {
  os << "// Parse parameter list\n";

  // If there are optional parameters, we need to switch to `parseOptionalComma`
  // if there are no more required parameters after a certain point.
  bool hasOptional = el->hasOptionalParams();
  if (hasOptional) {
    // Wrap everything in a do-while so that we can `break`.
    os << "do {\n";
    os.indent();
  }

  ArrayRef<ParameterElement *> params = el->getParams();
  using IteratorT = ParameterElement *const *;
  IteratorT it = params.begin();

  // Find the last required parameter. Commas become optional aftewards.
  // Note: IteratorT's copy assignment is deleted.
  ParameterElement *lastReq = nullptr;
  for (ParameterElement *param : params)
    if (!param->isOptional())
      lastReq = param;
  IteratorT lastReqIt = lastReq ? llvm::find(params, lastReq) : params.begin();

  auto eachFn = [&](ParameterElement *el) { genVariableParser(el, ctx, os); };
  auto betweenFn = [&](IteratorT it) {
    ParameterElement *el = *std::prev(it);
    // Parse a comma if the last optional parameter had a value.
    if (el->isOptional()) {
      os << formatv("if (::mlir::succeeded(_result_{0}) && *_result_{0}) {{\n",
                    el->getName());
      os.indent();
    }
    if (it <= lastReqIt) {
      genLiteralParser(",", ctx, os);
    } else {
      genLiteralParser(",", ctx, os, /*isOptional=*/true);
      os << ") break;\n";
    }
    if (el->isOptional())
      os.unindent() << "}\n";
  };

  // llvm::interleave
  if (it != params.end()) {
    eachFn(*it++);
    for (IteratorT e = params.end(); it != e; ++it) {
      betweenFn(it);
      eachFn(*it);
    }
  }

  if (hasOptional)
    os.unindent() << "} while(false);\n";
}

void DefFormat::genStructParser(StructDirective *el, FmtContext &ctx,
                                MethodBody &os) {
  // Loop declaration for struct parser with only required parameters.
  //
  // $0: Number of expected parameters.
  const char *const loopHeader = R"(
  for (unsigned odsStructIndex = 0; odsStructIndex < $0; ++odsStructIndex) {
)";

  // Loop body start for struct parser.
  const char *const loopStart = R"(
    ::llvm::StringRef _paramKey;
    if ($_parser.parseKeyword(&_paramKey)) {
      $_parser.emitError($_parser.getCurrentLocation(),
                         "expected a parameter name in struct");
      return {};
    }
    if (!_loop_body(_paramKey)) return {};
)";

  // Struct parser loop end. Check for duplicate or unknown struct parameters.
  //
  // {0}: Code template for printing an error.
  const char *const loopEnd = R"({{
  {0}"duplicate or unknown struct parameter name: ") << _paramKey;
  return {{};
}
)";

  // Struct parser loop terminator. Parse a comma except on the last element.
  //
  // {0}: Number of elements in the struct.
  const char *const loopTerminator = R"(
  if ((odsStructIndex != {0} - 1) && odsParser.parseComma())
    return {{};
}
)";

  // Check that a mandatory parameter was parse.
  //
  // {0}: Name of the parameter.
  const char *const checkParam = R"(
    if (!_seen_{0}) {
      {1}"struct is missing required parameter: ") << "{0}";
      return {{};
    }
)";

  // Optional parameters in a struct must be parsed successfully if the
  // keyword is present.
  //
  // {0}: Name of the parameter.
  // {1}: Emit error string
  const char *const checkOptionalParam = R"(
    if (::mlir::succeeded(_result_{0}) && !*_result_{0}) {{
      {1}"expected a value for parameter '{0}'");
      return {{};
    }
)";

  // First iteration of the loop parsing an optional struct.
  const char *const optionalStructFirst = R"(
  ::llvm::StringRef _paramKey;
  if (!$_parser.parseOptionalKeyword(&_paramKey)) {
    if (!_loop_body(_paramKey)) return {};
    while (!$_parser.parseOptionalComma()) {
)";

  os << "// Parse parameter struct\n";

  // Declare a "seen" variable for each key.
  for (ParameterElement *param : el->getParams())
    os << formatv("bool _seen_{0} = false;\n", param->getName());

  // Generate the body of the parsing loop inside a lambda.
  os << "{\n";
  os.indent()
      << "const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {\n";
  genLiteralParser("=", ctx, os.indent());
  for (ParameterElement *param : el->getParams()) {
    os << formatv("if (!_seen_{0} && _paramKey == \"{0}\") {\n"
                  "  _seen_{0} = true;\n",
                  param->getName());
    genVariableParser(param, ctx, os.indent());
    if (param->isOptional()) {
      os.getStream().printReindented(strfmt(checkOptionalParam,
                                            param->getName(),
                                            tgfmt(parserErrorStr, &ctx).str()));
    }
    os.unindent() << "} else ";
    // Print the check for duplicate or unknown parameter.
  }
  os.getStream().printReindented(strfmt(loopEnd, tgfmt(parserErrorStr, &ctx)));
  os << "return true;\n";
  os.unindent() << "};\n";

  // Generate the parsing loop. If optional parameters are present, then the
  // parse loop is guarded by commas.
  unsigned numOptional = llvm::count_if(el->getParams(), paramIsOptional);
  if (numOptional) {
    // If the struct itself is optional, pull out the first iteration.
    if (numOptional == el->getNumParams()) {
      os.getStream().printReindented(tgfmt(optionalStructFirst, &ctx).str());
      os.indent();
    } else {
      os << "do {\n";
    }
  } else {
    os.getStream().printReindented(
        tgfmt(loopHeader, &ctx, el->getNumParams()).str());
  }
  os.indent();
  os.getStream().printReindented(tgfmt(loopStart, &ctx).str());
  os.unindent();

  // Print the loop terminator. For optional parameters, we have to check that
  // all mandatory parameters have been parsed.
  // The whole struct is optional if all its parameters are optional.
  if (numOptional) {
    if (numOptional == el->getNumParams()) {
      os << "}\n";
      os.unindent() << "}\n";
    } else {
      os << tgfmt("} while(!$_parser.parseOptionalComma());\n", &ctx);
      for (ParameterElement *param : el->getParams()) {
        if (param->isOptional())
          continue;
        os.getStream().printReindented(
            strfmt(checkParam, param->getName(), tgfmt(parserErrorStr, &ctx)));
      }
    }
  } else {
    // Because the loop loops N times and each non-failing iteration sets 1 of
    // N flags, successfully exiting the loop means that all parameters have
    // been seen. `parseOptionalComma` would cause issues with any formats that
    // use "struct(...) `,`" beacuse structs aren't sounded by braces.
    os.getStream().printReindented(strfmt(loopTerminator, el->getNumParams()));
  }
  os.unindent() << "}\n";
}

void DefFormat::genCustomParser(CustomDirective *el, FmtContext &ctx,
                                MethodBody &os) {
  os << "{\n";
  os.indent();

  // Bound variables are passed directly to the parser as `FailureOr<T> &`.
  // Referenced variables are passed as `T`. The custom parser fails if it
  // returns failure or if any of the required parameters failed.
  os << tgfmt("auto odsCustomLoc = $_parser.getCurrentLocation();\n", &ctx);
  os << "(void)odsCustomLoc;\n";
  os << tgfmt("auto odsCustomResult = parse$0($_parser", &ctx, el->getName());
  os.indent();
  for (FormatElement *arg : el->getArguments()) {
    os << ",\n";
    if (auto *param = dyn_cast<ParameterElement>(arg))
      os << "_result_" << param->getName();
    else if (auto *ref = dyn_cast<RefDirective>(arg))
      os << "*_result_" << cast<ParameterElement>(ref->getArg())->getName();
    else
      os << tgfmt(cast<StringElement>(arg)->getValue(), &ctx);
  }
  os.unindent() << ");\n";
  os << "if (::mlir::failed(odsCustomResult)) return {};\n";
  for (FormatElement *arg : el->getArguments()) {
    if (auto *param = dyn_cast<ParameterElement>(arg)) {
      if (param->isOptional())
        continue;
      os << formatv("if (::mlir::failed(_result_{0})) {{\n", param->getName());
      os.indent() << tgfmt("$_parser.emitError(odsCustomLoc, ", &ctx)
                  << "\"custom parser failed to parse parameter '"
                  << param->getName() << "'\");\n";
      os << "return {};\n";
      os.unindent() << "}\n";
    }
  }

  os.unindent() << "}\n";
}

void DefFormat::genOptionalGroupParser(OptionalElement *el, FmtContext &ctx,
                                       MethodBody &os) {
  ArrayRef<FormatElement *> thenElements =
      el->getThenElements(/*parseable=*/true);

  FormatElement *first = thenElements.front();
  const auto guardOn = [&](auto params) {
    os << "if (!(";
    llvm::interleave(
        params, os,
        [&](ParameterElement *el) {
          os << formatv("(::mlir::succeeded(_result_{0}) && *_result_{0})",
                        el->getName());
        },
        " || ");
    os << ")) {\n";
  };
  if (auto *literal = dyn_cast<LiteralElement>(first)) {
    genLiteralParser(literal->getSpelling(), ctx, os, /*isOptional=*/true);
    os << ") {\n";
  } else if (auto *param = dyn_cast<ParameterElement>(first)) {
    genVariableParser(param, ctx, os);
    guardOn(llvm::makeArrayRef(param));
  } else if (auto *params = dyn_cast<ParamsDirective>(first)) {
    genParamsParser(params, ctx, os);
    guardOn(params->getParams());
  } else {
    auto *strct = cast<StructDirective>(first);
    genStructParser(strct, ctx, os);
    guardOn(params->getParams());
  }
  os.indent();

  // Generate the parsers for the rest of the thenElements.
  for (FormatElement *element : el->getElseElements(/*parseable=*/true))
    genElementParser(element, ctx, os);
  os.unindent() << "} else {\n";
  os.indent();
  for (FormatElement *element : thenElements.drop_front())
    genElementParser(element, ctx, os);
  os.unindent() << "}\n";
}

//===----------------------------------------------------------------------===//
// PrinterGen
//===----------------------------------------------------------------------===//

void DefFormat::genPrinter(MethodBody &os) {
  FmtContext ctx;
  ctx.addSubst("_printer", "odsPrinter");
  ctx.addSubst("_ctxt", "getContext()");
  ctx.withBuilder("odsBuilder");
  os.indent();
  os << "::mlir::Builder odsBuilder(getContext());\n";

  // Generate printers.
  shouldEmitSpace = true;
  lastWasPunctuation = false;
  for (FormatElement *el : elements)
    genElementPrinter(el, ctx, os);
}

void DefFormat::genElementPrinter(FormatElement *el, FmtContext &ctx,
                                  MethodBody &os) {
  if (auto *literal = dyn_cast<LiteralElement>(el))
    return genLiteralPrinter(literal->getSpelling(), ctx, os);
  if (auto *params = dyn_cast<ParamsDirective>(el))
    return genParamsPrinter(params, ctx, os);
  if (auto *strct = dyn_cast<StructDirective>(el))
    return genStructPrinter(strct, ctx, os);
  if (auto *custom = dyn_cast<CustomDirective>(el))
    return genCustomPrinter(custom, ctx, os);
  if (auto *var = dyn_cast<ParameterElement>(el))
    return genVariablePrinter(var, ctx, os);
  if (auto *optional = dyn_cast<OptionalElement>(el))
    return genOptionalGroupPrinter(optional, ctx, os);
  if (auto *whitespace = dyn_cast<WhitespaceElement>(el))
    return genWhitespacePrinter(whitespace, ctx, os);

  llvm::PrintFatalError("unsupported format element");
}

void DefFormat::genLiteralPrinter(StringRef value, FmtContext &ctx,
                                  MethodBody &os) {
  // Don't insert a space before certain punctuation.
  bool needSpace =
      shouldEmitSpace && shouldEmitSpaceBefore(value, lastWasPunctuation);
  os << tgfmt("$_printer$0 << \"$1\";\n", &ctx, needSpace ? " << ' '" : "",
              value);

  // Update the flags.
  shouldEmitSpace =
      value.size() != 1 || !StringRef("<({[").contains(value.front());
  lastWasPunctuation = value.front() != '_' && !isalpha(value.front());
}

void DefFormat::genVariablePrinter(ParameterElement *el, FmtContext &ctx,
                                   MethodBody &os, bool skipGuard) {
  const AttrOrTypeParameter &param = el->getParam();
  ctx.withSelf(param.getAccessorName() + "()");

  // Guard the printer on the presence of optional parameters and that they
  // aren't equal to their default values (if they have one).
  if (el->isOptional() && !skipGuard) {
    el->genPrintGuard(ctx, os << "if (") << ") {\n";
    os.indent();
  }

  // Insert a space before the next parameter, if necessary.
  if (shouldEmitSpace || !lastWasPunctuation)
    os << tgfmt("$_printer << ' ';\n", &ctx);
  shouldEmitSpace = true;
  lastWasPunctuation = false;

  if (el->shouldBeQualified())
    os << tgfmt(qualifiedParameterPrinter, &ctx) << ";\n";
  else if (auto printer = param.getPrinter())
    os << tgfmt(*printer, &ctx) << ";\n";
  else
    os << tgfmt(defaultParameterPrinter, &ctx) << ";\n";

  if (el->isOptional() && !skipGuard)
    os.unindent() << "}\n";
}

/// Generate code to guard printing on the presence of any optional parameters.
template <typename ParameterRange>
static void guardOnAny(FmtContext &ctx, MethodBody &os, ParameterRange &&params,
                       bool inverted = false) {
  os << "if (";
  if (inverted)
    os << "!(";
  llvm::interleave(
      params, os,
      [&](ParameterElement *param) { param->genPrintGuard(ctx, os); }, " || ");
  if (inverted)
    os << ")";
  os << ") {\n";
  os.indent();
}

void DefFormat::genCommaSeparatedPrinter(
    ArrayRef<ParameterElement *> params, FmtContext &ctx, MethodBody &os,
    function_ref<void(ParameterElement *)> extra) {
  // Emit a space if necessary, but only if the struct is present.
  if (shouldEmitSpace || !lastWasPunctuation) {
    bool allOptional = llvm::all_of(params, paramIsOptional);
    if (allOptional)
      guardOnAny(ctx, os, params);
    os << tgfmt("$_printer << ' ';\n", &ctx);
    if (allOptional)
      os.unindent() << "}\n";
  }

  // The first printed element does not need to emit a comma.
  os << "{\n";
  os.indent() << "bool _firstPrinted = true;\n";
  for (ParameterElement *param : params) {
    if (param->isOptional()) {
      param->genPrintGuard(ctx, os << "if (") << ") {\n";
      os.indent();
    }
    os << tgfmt("if (!_firstPrinted) $_printer << \", \";\n", &ctx);
    os << "_firstPrinted = false;\n";
    extra(param);
    shouldEmitSpace = false;
    lastWasPunctuation = true;
    genVariablePrinter(param, ctx, os);
    if (param->isOptional())
      os.unindent() << "}\n";
  }
  os.unindent() << "}\n";
}

void DefFormat::genParamsPrinter(ParamsDirective *el, FmtContext &ctx,
                                 MethodBody &os) {
  genCommaSeparatedPrinter(llvm::to_vector(el->getParams()), ctx, os,
                           [&](ParameterElement *param) {});
}

void DefFormat::genStructPrinter(StructDirective *el, FmtContext &ctx,
                                 MethodBody &os) {
  genCommaSeparatedPrinter(
      llvm::to_vector(el->getParams()), ctx, os, [&](ParameterElement *param) {
        os << tgfmt("$_printer << \"$0 = \";\n", &ctx, param->getName());
      });
}

void DefFormat::genCustomPrinter(CustomDirective *el, FmtContext &ctx,
                                 MethodBody &os) {
  os << tgfmt("print$0($_printer", &ctx, el->getName());
  os.indent();
  for (FormatElement *arg : el->getArguments()) {
    os << ",\n";
    if (auto *param = dyn_cast<ParameterElement>(arg)) {
      os << param->getParam().getAccessorName() << "()";
    } else if (auto *ref = dyn_cast<RefDirective>(arg)) {
      os << cast<ParameterElement>(ref->getArg())->getParam().getAccessorName()
         << "()";
    } else {
      os << tgfmt(cast<StringElement>(arg)->getValue(), &ctx);
    }
  }
  os.unindent() << ");\n";
}

void DefFormat::genOptionalGroupPrinter(OptionalElement *el, FmtContext &ctx,
                                        MethodBody &os) {
  FormatElement *anchor = el->getAnchor();
  if (auto *param = dyn_cast<ParameterElement>(anchor)) {
    guardOnAny(ctx, os, llvm::makeArrayRef(param), el->isInverted());
  } else if (auto *params = dyn_cast<ParamsDirective>(anchor)) {
    guardOnAny(ctx, os, params->getParams(), el->isInverted());
  } else {
    auto *strct = cast<StructDirective>(anchor);
    guardOnAny(ctx, os, strct->getParams(), el->isInverted());
  }
  // Generate the printer for the contained elements.
  {
    llvm::SaveAndRestore<bool> shouldEmitSpaceFlag(shouldEmitSpace);
    llvm::SaveAndRestore<bool> lastWasPunctuationFlag(lastWasPunctuation);
    for (FormatElement *element : el->getThenElements())
      genElementPrinter(element, ctx, os);
  }
  os.unindent() << "} else {\n";
  os.indent();
  for (FormatElement *element : el->getElseElements())
    genElementPrinter(element, ctx, os);
  os.unindent() << "}\n";
}

void DefFormat::genWhitespacePrinter(WhitespaceElement *el, FmtContext &ctx,
                                     MethodBody &os) {
  if (el->getValue() == "\\n") {
    // FIXME: The newline should be `printer.printNewLine()`, i.e., handled by
    // the printer.
    os << tgfmt("$_printer << '\\n';\n", &ctx);
  } else if (!el->getValue().empty()) {
    os << tgfmt("$_printer << \"$0\";\n", &ctx, el->getValue());
  } else {
    lastWasPunctuation = true;
  }
  shouldEmitSpace = false;
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
  FailureOr<DefFormat> parse();

protected:
  /// Verify the parsed elements.
  LogicalResult verify(SMLoc loc, ArrayRef<FormatElement *> elements) override;
  /// Verify the elements of a custom directive.
  LogicalResult
  verifyCustomDirectiveArguments(SMLoc loc,
                                 ArrayRef<FormatElement *> arguments) override;
  /// Verify the elements of an optional group.
  LogicalResult verifyOptionalGroupElements(SMLoc loc,
                                            ArrayRef<FormatElement *> elements,
                                            FormatElement *anchor) override;

  /// Parse an attribute or type variable.
  FailureOr<FormatElement *> parseVariableImpl(SMLoc loc, StringRef name,
                                               Context ctx) override;
  /// Parse an attribute or type format directive.
  FailureOr<FormatElement *>
  parseDirectiveImpl(SMLoc loc, FormatToken::Kind kind, Context ctx) override;

private:
  /// Parse a `params` directive.
  FailureOr<FormatElement *> parseParamsDirective(SMLoc loc, Context ctx);
  /// Parse a `qualified` directive.
  FailureOr<FormatElement *> parseQualifiedDirective(SMLoc loc, Context ctx);
  /// Parse a `struct` directive.
  FailureOr<FormatElement *> parseStructDirective(SMLoc loc, Context ctx);
  /// Parse a `ref` directive.
  FailureOr<FormatElement *> parseRefDirective(SMLoc loc, Context ctx);

  /// Attribute or type tablegen def.
  const AttrOrTypeDef &def;

  /// Seen attribute or type parameters.
  BitVector seenParams;
};
} // namespace

LogicalResult DefFormatParser::verify(SMLoc loc,
                                      ArrayRef<FormatElement *> elements) {
  // Check that all parameters are referenced in the format.
  for (auto &it : llvm::enumerate(def.getParameters())) {
    if (it.value().isOptional())
      continue;
    if (!seenParams.test(it.index())) {
      if (isa<AttributeSelfTypeParameter>(it.value()))
        continue;
      return emitError(loc, "format is missing reference to parameter: " +
                                it.value().getName());
    }
    if (isa<AttributeSelfTypeParameter>(it.value())) {
      return emitError(loc,
                       "unexpected self type parameter in assembly format");
    }
  }
  if (elements.empty())
    return success();
  // A `struct` directive that contains optional parameters cannot be followed
  // by a comma literal, which is ambiguous.
  for (auto it : llvm::zip(elements.drop_back(), elements.drop_front())) {
    auto *structEl = dyn_cast<StructDirective>(std::get<0>(it));
    auto *literalEl = dyn_cast<LiteralElement>(std::get<1>(it));
    if (!structEl || !literalEl)
      continue;
    if (literalEl->getSpelling() == "," && structEl->hasOptionalParams()) {
      return emitError(loc, "`struct` directive with optional parameters "
                            "cannot be followed by a comma literal");
    }
  }
  return success();
}

LogicalResult DefFormatParser::verifyCustomDirectiveArguments(
    SMLoc loc, ArrayRef<FormatElement *> arguments) {
  // Arguments are fully verified by the parser context.
  return success();
}

LogicalResult
DefFormatParser::verifyOptionalGroupElements(llvm::SMLoc loc,
                                             ArrayRef<FormatElement *> elements,
                                             FormatElement *anchor) {
  // `params` and `struct` directives are allowed only if all the contained
  // parameters are optional.
  for (FormatElement *el : elements) {
    if (auto *param = dyn_cast<ParameterElement>(el)) {
      if (!param->isOptional()) {
        return emitError(loc,
                         "parameters in an optional group must be optional");
      }
    } else if (auto *params = dyn_cast<ParamsDirective>(el)) {
      if (llvm::any_of(params->getParams(), paramNotOptional)) {
        return emitError(loc, "`params` directive allowed in optional group "
                              "only if all parameters are optional");
      }
    } else if (auto *strct = dyn_cast<StructDirective>(el)) {
      if (llvm::any_of(strct->getParams(), paramNotOptional)) {
        return emitError(loc, "`struct` is only allowed in an optional group "
                              "if all captured parameters are optional");
      }
    }
  }
  // The anchor must be a parameter or one of the aforementioned directives.
  if (anchor &&
      !isa<ParameterElement, ParamsDirective, StructDirective>(anchor)) {
    return emitError(loc,
                     "optional group anchor must be a parameter or directive");
  }
  return success();
}

FailureOr<DefFormat> DefFormatParser::parse() {
  FailureOr<std::vector<FormatElement *>> elements = FormatParser::parse();
  if (failed(elements))
    return failure();
  return DefFormat(def, std::move(*elements));
}

FailureOr<FormatElement *>
DefFormatParser::parseVariableImpl(SMLoc loc, StringRef name, Context ctx) {
  // Lookup the parameter.
  ArrayRef<AttrOrTypeParameter> params = def.getParameters();
  auto *it = llvm::find_if(
      params, [&](auto &param) { return param.getName() == name; });

  // Check that the parameter reference is valid.
  if (it == params.end()) {
    return emitError(loc,
                     def.getName() + " has no parameter named '" + name + "'");
  }
  auto idx = std::distance(params.begin(), it);

  if (ctx != RefDirectiveContext) {
    // Check that the variable has not already been bound.
    if (seenParams.test(idx))
      return emitError(loc, "duplicate parameter '" + name + "'");
    seenParams.set(idx);

    // Otherwise, to be referenced, a variable must have been bound.
  } else if (!seenParams.test(idx)) {
    return emitError(loc, "parameter '" + name +
                              "' must be bound before it is referenced");
  }

  return create<ParameterElement>(*it);
}

FailureOr<FormatElement *>
DefFormatParser::parseDirectiveImpl(SMLoc loc, FormatToken::Kind kind,
                                    Context ctx) {

  switch (kind) {
  case FormatToken::kw_qualified:
    return parseQualifiedDirective(loc, ctx);
  case FormatToken::kw_params:
    return parseParamsDirective(loc, ctx);
  case FormatToken::kw_struct:
    return parseStructDirective(loc, ctx);
  case FormatToken::kw_ref:
    return parseRefDirective(loc, ctx);
  case FormatToken::kw_custom:
    return parseCustomDirective(loc, ctx);

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

FailureOr<FormatElement *> DefFormatParser::parseParamsDirective(SMLoc loc,
                                                                 Context ctx) {
  // It doesn't make sense to allow references to all parameters in a custom
  // directive because parameters are the only things that can be bound.
  if (ctx != TopLevelContext && ctx != StructDirectiveContext) {
    return emitError(loc, "`params` can only be used at the top-level context "
                          "or within a `struct` directive");
  }

  // Collect all of the attribute's or type's parameters and ensure that none of
  // the parameters have already been captured.
  std::vector<ParameterElement *> vars;
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

FailureOr<FormatElement *> DefFormatParser::parseStructDirective(SMLoc loc,
                                                                 Context ctx) {
  if (ctx != TopLevelContext)
    return emitError(loc, "`struct` can only be used at the top-level context");

  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before `struct` argument list")))
    return failure();

  // Parse variables captured by `struct`.
  std::vector<ParameterElement *> vars;

  // Parse first captured parameter or a `params` directive.
  FailureOr<FormatElement *> var = parseElement(StructDirectiveContext);
  if (failed(var) || !isa<VariableElement, ParamsDirective>(*var)) {
    return emitError(loc,
                     "`struct` argument list expected a variable or directive");
  }
  if (isa<VariableElement>(*var)) {
    // Parse any other parameters.
    vars.push_back(cast<ParameterElement>(*var));
    while (peekToken().is(FormatToken::comma)) {
      consumeToken();
      var = parseElement(StructDirectiveContext);
      if (failed(var) || !isa<VariableElement>(*var))
        return emitError(loc, "expected a variable in `struct` argument list");
      vars.push_back(cast<ParameterElement>(*var));
    }
  } else {
    // `struct(params)` captures all parameters in the attribute or type.
    vars = cast<ParamsDirective>(*var)->takeParams();
  }

  if (failed(parseToken(FormatToken::r_paren,
                        "expected ')' at the end of an argument list")))
    return failure();

  return create<StructDirective>(std::move(vars));
}

FailureOr<FormatElement *> DefFormatParser::parseRefDirective(SMLoc loc,
                                                              Context ctx) {
  if (ctx != CustomDirectiveContext)
    return emitError(loc, "`ref` is only allowed inside custom directives");

  // Parse the child parameter element.
  FailureOr<FormatElement *> child;
  if (failed(parseToken(FormatToken::l_paren, "expected '('")) ||
      failed(child = parseElement(RefDirectiveContext)) ||
      failed(parseToken(FormatToken::r_paren, "expeced ')'")))
    return failure();

  // Only parameter elements are allowed to be parsed under a `ref` directive.
  return create<RefDirective>(*child);
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

  // Parse the custom assembly format>
  DefFormatParser fmtParser(mgr, def);
  FailureOr<DefFormat> format = fmtParser.parse();
  if (failed(format)) {
    if (formatErrorIsFatal)
      PrintFatalError(def.getLoc(), "failed to parse assembly format");
    return;
  }

  // Generate the parser and printer.
  format->genParser(parser);
  format->genPrinter(printer);
}
