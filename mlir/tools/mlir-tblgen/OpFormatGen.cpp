//===- OpFormatGen.cpp - MLIR operation asm format generator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpFormatGen.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/OpClass.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#define DEBUG_TYPE "mlir-tblgen-opformatgen"

using namespace mlir;
using namespace mlir::tblgen;

static llvm::cl::opt<bool> formatErrorIsFatal(
    "asmformat-error-is-fatal",
    llvm::cl::desc("Emit a fatal error if format parsing fails"),
    llvm::cl::init(true));

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
    SuccessorsDirective,
    TypeDirective,

    /// This element is a literal.
    Literal,

    /// This element is an variable value.
    AttributeVariable,
    OperandVariable,
    ResultVariable,
    SuccessorVariable,

    /// This element is an optional element.
    Optional,
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

protected:
  const VarT *var;
};

/// This class represents a variable that refers to an attribute argument.
struct AttributeVariable
    : public VariableElement<NamedAttribute, Element::Kind::AttributeVariable> {
  using VariableElement<NamedAttribute,
                        Element::Kind::AttributeVariable>::VariableElement;

  /// Return the constant builder call for the type of this attribute, or None
  /// if it doesn't have one.
  Optional<StringRef> getTypeBuilder() const {
    Optional<Type> attrType = var->attr.getValueType();
    return attrType ? attrType->getBuilderCall() : llvm::None;
  }
};

/// This class represents a variable that refers to an operand argument.
using OperandVariable =
    VariableElement<NamedTypeConstraint, Element::Kind::OperandVariable>;

/// This class represents a variable that refers to a result.
using ResultVariable =
    VariableElement<NamedTypeConstraint, Element::Kind::ResultVariable>;

/// This class represents a variable that refers to a successor.
using SuccessorVariable =
    VariableElement<NamedSuccessor, Element::Kind::SuccessorVariable>;
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
/// This class represents the `operands` directive. This directive represents
/// all of the operands of an operation.
using OperandsDirective = DirectiveElement<Element::Kind::OperandsDirective>;

/// This class represents the `results` directive. This directive represents
/// all of the results of an operation.
using ResultsDirective = DirectiveElement<Element::Kind::ResultsDirective>;

/// This class represents the `successors` directive. This directive represents
/// all of the successors of an operation.
using SuccessorsDirective =
    DirectiveElement<Element::Kind::SuccessorsDirective>;

/// This class represents the `attr-dict` directive. This directive represents
/// the attribute dictionary of the operation.
class AttrDictDirective
    : public DirectiveElement<Element::Kind::AttrDictDirective> {
public:
  explicit AttrDictDirective(bool withKeyword) : withKeyword(withKeyword) {}
  bool isWithKeyword() const { return withKeyword; }

private:
  /// If the dictionary should be printed with the 'attributes' keyword.
  bool withKeyword;
};

/// This class represents the `functional-type` directive. This directive takes
/// two arguments and formats them, respectively, as the inputs and results of a
/// FunctionType.
class FunctionalTypeDirective
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
class TypeDirective : public DirectiveElement<Element::Kind::TypeDirective> {
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
      : Element{Kind::Literal}, literal(literal) {}
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
// OptionalElement

namespace {
/// This class represents a group of elements that are optionally emitted based
/// upon an optional variable of the operation.
class OptionalElement : public Element {
public:
  OptionalElement(std::vector<std::unique_ptr<Element>> &&elements,
                  unsigned anchor)
      : Element{Kind::Optional}, elements(std::move(elements)), anchor(anchor) {
  }
  static bool classof(const Element *element) {
    return element->getKind() == Kind::Optional;
  }

  /// Return the nested elements of this grouping.
  auto getElements() const { return llvm::make_pointee_range(elements); }

  /// Return the anchor of this optional group.
  Element *getAnchor() const { return elements[anchor].get(); }

private:
  /// The child elements of this optional.
  std::vector<std::unique_ptr<Element>> elements;
  /// The index of the element that acts as the anchor for the optional group.
  unsigned anchor;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// OperationFormat
//===----------------------------------------------------------------------===//

namespace {

using ConstArgument =
    llvm::PointerUnion<const NamedAttribute *, const NamedTypeConstraint *>;

struct OperationFormat {
  /// This class represents a specific resolver for an operand or result type.
  class TypeResolution {
  public:
    TypeResolution() = default;

    /// Get the index into the buildable types for this type, or None.
    Optional<int> getBuilderIdx() const { return builderIdx; }
    void setBuilderIdx(int idx) { builderIdx = idx; }

    /// Get the variable this type is resolved to, or nullptr.
    const NamedTypeConstraint *getVariable() const {
      return resolver.dyn_cast<const NamedTypeConstraint *>();
    }
    /// Get the attribute this type is resolved to, or nullptr.
    const NamedAttribute *getAttribute() const {
      return resolver.dyn_cast<const NamedAttribute *>();
    }
    /// Get the transformer for the type of the variable, or None.
    Optional<StringRef> getVarTransformer() const {
      return variableTransformer;
    }
    void setResolver(ConstArgument arg, Optional<StringRef> transformer) {
      resolver = arg;
      variableTransformer = transformer;
      assert(getVariable() || getAttribute());
    }

  private:
    /// If the type is resolved with a buildable type, this is the index into
    /// 'buildableTypes' in the parent format.
    Optional<int> builderIdx;
    /// If the type is resolved based upon another operand or result, this is
    /// the variable or the attribute that this type is resolved to.
    ConstArgument resolver;
    /// If the type is resolved based upon another operand or result, this is
    /// a transformer to apply to the variable when resolving.
    Optional<StringRef> variableTransformer;
  };

  OperationFormat(const Operator &op)
      : allOperands(false), allOperandTypes(false), allResultTypes(false) {
    operandTypes.resize(op.getNumOperands(), TypeResolution());
    resultTypes.resize(op.getNumResults(), TypeResolution());
  }

  /// Generate the operation parser from this format.
  void genParser(Operator &op, OpClass &opClass);
  /// Generate the c++ to resolve the types of operands and results during
  /// parsing.
  void genParserTypeResolution(Operator &op, OpMethodBody &body);
  /// Generate the c++ to resolve successors during parsing.
  void genParserSuccessorResolution(Operator &op, OpMethodBody &body);
  /// Generate the c++ to handling variadic segment size traits.
  void genParserVariadicSegmentResolution(Operator &op, OpMethodBody &body);

  /// Generate the operation printer from this format.
  void genPrinter(Operator &op, OpClass &opClass);

  /// The various elements in this format.
  std::vector<std::unique_ptr<Element>> elements;

  /// A flag indicating if all operand/result types were seen. If the format
  /// contains these, it can not contain individual type resolvers.
  bool allOperands, allOperandTypes, allResultTypes;

  /// A map of buildable types to indices.
  llvm::MapVector<StringRef, int, llvm::StringMap<int>> buildableTypes;

  /// The index of the buildable type, if valid, for every operand and result.
  std::vector<TypeResolution> operandTypes, resultTypes;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Parser Gen

/// Returns if we can format the given attribute as an EnumAttr in the parser
/// format.
static bool canFormatEnumAttr(const NamedAttribute *attr) {
  const EnumAttr *enumAttr = dyn_cast<EnumAttr>(&attr->attr);
  if (!enumAttr)
    return false;

  // The attribute must have a valid underlying type and a constant builder.
  return !enumAttr->getUnderlyingType().empty() &&
         !enumAttr->getConstBuilderTemplate().empty();
}

/// The code snippet used to generate a parser call for an attribute.
///
/// {0}: The storage type of the attribute.
/// {1}: The name of the attribute.
/// {2}: The type for the attribute.
const char *const attrParserCode = R"(
  {0} {1}Attr;
  if (parser.parseAttribute({1}Attr{2}, "{1}", result.attributes))
    return failure();
)";
const char *const optionalAttrParserCode = R"(
  {0} {1}Attr;
  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute({1}Attr{2}, "{1}", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return failure();
  }
)";

/// The code snippet used to generate a parser call for an enum attribute.
///
/// {0}: The name of the attribute.
/// {1}: The c++ namespace for the enum symbolize functions.
/// {2}: The function to symbolize a string of the enum.
/// {3}: The constant builder call to create an attribute of the enum type.
const char *const enumAttrParserCode = R"(
  {
    ::mlir::StringAttr attrVal;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "{0}", attrStorage))
      return failure();

    auto attrOptional = {1}::{2}(attrVal.getValue());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "{0} attribute specification: " << attrVal;

    result.addAttribute("{0}", {3});
  }
)";
const char *const optionalEnumAttrParserCode = R"(
  Attribute {0}Attr;
  {
    ::mlir::StringAttr attrVal;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();

    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(attrVal, parser.getBuilder().getNoneType(),
                                    "{0}", attrStorage);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return failure();

      auto attrOptional = {1}::{2}(attrVal.getValue());
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "{0} attribute specification: " << attrVal;

      {0}Attr = {3};
      result.addAttribute("{0}", {0}Attr);
    }
  }
)";

/// The code snippet used to generate a parser call for an operand.
///
/// {0}: The name of the operand.
const char *const variadicOperandParserCode = R"(
  if (parser.parseOperandList({0}Operands))
    return failure();
)";
const char *const optionalOperandParserCode = R"(
  {
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return failure();
      {0}Operands.push_back(operand);
    }
  }
)";
const char *const operandParserCode = R"(
  if (parser.parseOperand({0}RawOperands[0]))
    return failure();
)";

/// The code snippet used to generate a parser call for a type list.
///
/// {0}: The name for the type list.
const char *const variadicTypeParserCode = R"(
  if (parser.parseTypeList({0}Types))
    return failure();
)";
const char *const optionalTypeParserCode = R"(
  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return failure();
      {0}Types.push_back(optionalType);
    }
  }
)";
const char *const typeParserCode = R"(
  if (parser.parseType({0}RawTypes[0]))
    return failure();
)";

/// The code snippet used to generate a parser call for a functional type.
///
/// {0}: The name for the input type list.
/// {1}: The name for the result type list.
const char *const functionalTypeParserCode = R"(
  ::mlir::FunctionType {0}__{1}_functionType;
  if (parser.parseType({0}__{1}_functionType))
    return failure();
  {0}Types = {0}__{1}_functionType.getInputs();
  {1}Types = {0}__{1}_functionType.getResults();
)";

/// The code snippet used to generate a parser call for a successor list.
///
/// {0}: The name for the successor list.
const char *successorListParserCode = R"(
  ::llvm::SmallVector<::mlir::Block *, 2> {0}Successors;
  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return failure();
      {0}Successors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return failure();
        {0}Successors.emplace_back(succ);
      }
    }
  }
)";

/// The code snippet used to generate a parser call for a successor.
///
/// {0}: The name of the successor.
const char *successorParserCode = R"(
  ::mlir::Block *{0}Successor = nullptr;
  if (parser.parseSuccessor({0}Successor))
    return failure();
)";

namespace {
/// The type of length for a given parse argument.
enum class ArgumentLengthKind {
  /// The argument is variadic, and may contain 0->N elements.
  Variadic,
  /// The argument is optional, and may contain 0 or 1 elements.
  Optional,
  /// The argument is a single element, i.e. always represents 1 element.
  Single
};
} // end anonymous namespace

/// Get the length kind for the given constraint.
static ArgumentLengthKind
getArgumentLengthKind(const NamedTypeConstraint *var) {
  if (var->isOptional())
    return ArgumentLengthKind::Optional;
  if (var->isVariadic())
    return ArgumentLengthKind::Variadic;
  return ArgumentLengthKind::Single;
}

/// Get the name used for the type list for the given type directive operand.
/// 'lengthKind' to the corresponding kind for the given argument.
static StringRef getTypeListName(Element *arg, ArgumentLengthKind &lengthKind) {
  if (auto *operand = dyn_cast<OperandVariable>(arg)) {
    lengthKind = getArgumentLengthKind(operand->getVar());
    return operand->getVar()->name;
  }
  if (auto *result = dyn_cast<ResultVariable>(arg)) {
    lengthKind = getArgumentLengthKind(result->getVar());
    return result->getVar()->name;
  }
  lengthKind = ArgumentLengthKind::Variadic;
  if (isa<OperandsDirective>(arg))
    return "allOperand";
  if (isa<ResultsDirective>(arg))
    return "allResult";
  llvm_unreachable("unknown 'type' directive argument");
}

/// Generate the parser for a literal value.
static void genLiteralParser(StringRef value, OpMethodBody &body) {
  // Handle the case of a keyword/identifier.
  if (value.front() == '_' || isalpha(value.front())) {
    body << "Keyword(\"" << value << "\")";
    return;
  }
  body << (StringRef)llvm::StringSwitch<StringRef>(value)
              .Case("->", "Arrow()")
              .Case(":", "Colon()")
              .Case(",", "Comma()")
              .Case("=", "Equal()")
              .Case("<", "Less()")
              .Case(">", "Greater()")
              .Case("(", "LParen()")
              .Case(")", "RParen()")
              .Case("[", "LSquare()")
              .Case("]", "RSquare()");
}

/// Generate the storage code required for parsing the given element.
static void genElementParserStorage(Element *element, OpMethodBody &body) {
  if (auto *optional = dyn_cast<OptionalElement>(element)) {
    for (auto &childElement : optional->getElements())
      genElementParserStorage(&childElement, body);
  } else if (auto *operand = dyn_cast<OperandVariable>(element)) {
    StringRef name = operand->getVar()->name;
    if (operand->getVar()->isVariableLength()) {
      body << "  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> "
           << name << "Operands;\n";
    } else {
      body << "  ::mlir::OpAsmParser::OperandType " << name
           << "RawOperands[1];\n"
           << "  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> " << name
           << "Operands(" << name << "RawOperands);";
    }
    body << llvm::formatv(
        "  ::llvm::SMLoc {0}OperandsLoc = parser.getCurrentLocation();\n"
        "  (void){0}OperandsLoc;\n",
        name);
  } else if (auto *dir = dyn_cast<TypeDirective>(element)) {
    ArgumentLengthKind lengthKind;
    StringRef name = getTypeListName(dir->getOperand(), lengthKind);
    if (lengthKind != ArgumentLengthKind::Single)
      body << "  ::mlir::SmallVector<::mlir::Type, 1> " << name << "Types;\n";
    else
      body << llvm::formatv("  ::mlir::Type {0}RawTypes[1];\n", name)
           << llvm::formatv(
                  "  ::llvm::ArrayRef<::mlir::Type> {0}Types({0}RawTypes);\n",
                  name);
  } else if (auto *dir = dyn_cast<FunctionalTypeDirective>(element)) {
    ArgumentLengthKind ignored;
    body << "  ::llvm::ArrayRef<::mlir::Type> "
         << getTypeListName(dir->getInputs(), ignored) << "Types;\n";
    body << "  ::llvm::ArrayRef<::mlir::Type> "
         << getTypeListName(dir->getResults(), ignored) << "Types;\n";
  }
}

/// Generate the parser for a single format element.
static void genElementParser(Element *element, OpMethodBody &body,
                             FmtContext &attrTypeCtx) {
  /// Optional Group.
  if (auto *optional = dyn_cast<OptionalElement>(element)) {
    auto elements = optional->getElements();

    // Generate a special optional parser for the first element to gate the
    // parsing of the rest of the elements.
    Element *firstElement = &*elements.begin();
    if (auto *attrVar = dyn_cast<AttributeVariable>(firstElement)) {
      genElementParser(attrVar, body, attrTypeCtx);
      body << "  if (" << attrVar->getVar()->name << "Attr) {\n";
    } else if (auto *literal = dyn_cast<LiteralElement>(firstElement)) {
      body << "  if (succeeded(parser.parseOptional";
      genLiteralParser(literal->getLiteral(), body);
      body << ")) {\n";
    } else if (auto *opVar = dyn_cast<OperandVariable>(firstElement)) {
      genElementParser(opVar, body, attrTypeCtx);
      body << "  if (!" << opVar->getVar()->name << "Operands.empty()) {\n";
    }

    // Generate the rest of the elements normally.
    for (auto &childElement : llvm::drop_begin(elements, 1))
      genElementParser(&childElement, body, attrTypeCtx);
    body << "  }\n";

    /// Literals.
  } else if (LiteralElement *literal = dyn_cast<LiteralElement>(element)) {
    body << "  if (parser.parse";
    genLiteralParser(literal->getLiteral(), body);
    body << ")\n    return failure();\n";

    /// Arguments.
  } else if (auto *attr = dyn_cast<AttributeVariable>(element)) {
    const NamedAttribute *var = attr->getVar();

    // Check to see if we can parse this as an enum attribute.
    if (canFormatEnumAttr(var)) {
      const EnumAttr &enumAttr = cast<EnumAttr>(var->attr);

      // Generate the code for building an attribute for this enum.
      std::string attrBuilderStr;
      {
        llvm::raw_string_ostream os(attrBuilderStr);
        os << tgfmt(enumAttr.getConstBuilderTemplate(), &attrTypeCtx,
                    "attrOptional.getValue()");
      }

      body << formatv(var->attr.isOptional() ? optionalEnumAttrParserCode
                                             : enumAttrParserCode,
                      var->name, enumAttr.getCppNamespace(),
                      enumAttr.getStringToSymbolFnName(), attrBuilderStr);
      return;
    }

    // If this attribute has a buildable type, use that when parsing the
    // attribute.
    std::string attrTypeStr;
    if (Optional<StringRef> typeBuilder = attr->getTypeBuilder()) {
      llvm::raw_string_ostream os(attrTypeStr);
      os << ", " << tgfmt(*typeBuilder, &attrTypeCtx);
    }

    body << formatv(var->attr.isOptional() ? optionalAttrParserCode
                                           : attrParserCode,
                    var->attr.getStorageType(), var->name, attrTypeStr);
  } else if (auto *operand = dyn_cast<OperandVariable>(element)) {
    ArgumentLengthKind lengthKind = getArgumentLengthKind(operand->getVar());
    StringRef name = operand->getVar()->name;
    if (lengthKind == ArgumentLengthKind::Variadic)
      body << llvm::formatv(variadicOperandParserCode, name);
    else if (lengthKind == ArgumentLengthKind::Optional)
      body << llvm::formatv(optionalOperandParserCode, name);
    else
      body << formatv(operandParserCode, name);
  } else if (auto *successor = dyn_cast<SuccessorVariable>(element)) {
    bool isVariadic = successor->getVar()->isVariadic();
    body << formatv(isVariadic ? successorListParserCode : successorParserCode,
                    successor->getVar()->name);

    /// Directives.
  } else if (auto *attrDict = dyn_cast<AttrDictDirective>(element)) {
    body << "  if (parser.parseOptionalAttrDict"
         << (attrDict->isWithKeyword() ? "WithKeyword" : "")
         << "(result.attributes))\n"
         << "    return failure();\n";
  } else if (isa<OperandsDirective>(element)) {
    body << "  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();\n"
         << "  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> "
            "allOperands;\n"
         << "  if (parser.parseOperandList(allOperands))\n"
         << "    return failure();\n";
  } else if (isa<SuccessorsDirective>(element)) {
    body << llvm::formatv(successorListParserCode, "full");
  } else if (auto *dir = dyn_cast<TypeDirective>(element)) {
    ArgumentLengthKind lengthKind;
    StringRef listName = getTypeListName(dir->getOperand(), lengthKind);
    if (lengthKind == ArgumentLengthKind::Variadic)
      body << llvm::formatv(variadicTypeParserCode, listName);
    else if (lengthKind == ArgumentLengthKind::Optional)
      body << llvm::formatv(optionalTypeParserCode, listName);
    else
      body << formatv(typeParserCode, listName);
  } else if (auto *dir = dyn_cast<FunctionalTypeDirective>(element)) {
    ArgumentLengthKind ignored;
    body << formatv(functionalTypeParserCode,
                    getTypeListName(dir->getInputs(), ignored),
                    getTypeListName(dir->getResults(), ignored));
  } else {
    llvm_unreachable("unknown format element");
  }
}

void OperationFormat::genParser(Operator &op, OpClass &opClass) {
  auto &method = opClass.newMethod(
      "::mlir::ParseResult", "parse",
      "::mlir::OpAsmParser &parser, ::mlir::OperationState &result",
      OpMethod::MP_Static);
  auto &body = method.body();

  // Generate variables to store the operands and type within the format. This
  // allows for referencing these variables in the presence of optional
  // groupings.
  for (auto &element : elements)
    genElementParserStorage(&*element, body);

  // A format context used when parsing attributes with buildable types.
  FmtContext attrTypeCtx;
  attrTypeCtx.withBuilder("parser.getBuilder()");

  // Generate parsers for each of the elements.
  for (auto &element : elements)
    genElementParser(element.get(), body, attrTypeCtx);

  // Generate the code to resolve the operand/result types and successors now
  // that they have been parsed.
  genParserTypeResolution(op, body);
  genParserSuccessorResolution(op, body);
  genParserVariadicSegmentResolution(op, body);

  body << "  return success();\n";
}

void OperationFormat::genParserTypeResolution(Operator &op,
                                              OpMethodBody &body) {
  // If any of type resolutions use transformed variables, make sure that the
  // types of those variables are resolved.
  SmallPtrSet<const NamedTypeConstraint *, 8> verifiedVariables;
  FmtContext verifierFCtx;
  for (TypeResolution &resolver :
       llvm::concat<TypeResolution>(resultTypes, operandTypes)) {
    Optional<StringRef> transformer = resolver.getVarTransformer();
    if (!transformer)
      continue;
    // Ensure that we don't verify the same variables twice.
    const NamedTypeConstraint *variable = resolver.getVariable();
    if (!variable || !verifiedVariables.insert(variable).second)
      continue;

    auto constraint = variable->constraint;
    body << "  for (::mlir::Type type : " << variable->name << "Types) {\n"
         << "    (void)type;\n"
         << "    if (!("
         << tgfmt(constraint.getConditionTemplate(),
                  &verifierFCtx.withSelf("type"))
         << ")) {\n"
         << formatv("      return parser.emitError(parser.getNameLoc()) << "
                    "\"'{0}' must be {1}, but got \" << type;\n",
                    variable->name, constraint.getDescription())
         << "    }\n"
         << "  }\n";
  }

  // Initialize the set of buildable types.
  if (!buildableTypes.empty()) {
    FmtContext typeBuilderCtx;
    typeBuilderCtx.withBuilder("parser.getBuilder()");
    for (auto &it : buildableTypes)
      body << "  ::mlir::Type odsBuildableType" << it.second << " = "
           << tgfmt(it.first, &typeBuilderCtx) << ";\n";
  }

  // Emit the code necessary for a type resolver.
  auto emitTypeResolver = [&](TypeResolution &resolver, StringRef curVar) {
    if (Optional<int> val = resolver.getBuilderIdx()) {
      body << "odsBuildableType" << *val;
    } else if (const NamedTypeConstraint *var = resolver.getVariable()) {
      if (Optional<StringRef> tform = resolver.getVarTransformer())
        body << tgfmt(*tform, &FmtContext().withSelf(var->name + "Types[0]"));
      else
        body << var->name << "Types";
    } else if (const NamedAttribute *attr = resolver.getAttribute()) {
      if (Optional<StringRef> tform = resolver.getVarTransformer())
        body << tgfmt(*tform,
                      &FmtContext().withSelf(attr->name + "Attr.getType()"));
      else
        body << attr->name << "Attr.getType()";
    } else {
      body << curVar << "Types";
    }
  };

  // Resolve each of the result types.
  if (allResultTypes) {
    body << "  result.addTypes(allResultTypes);\n";
  } else {
    for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
      body << "  result.addTypes(";
      emitTypeResolver(resultTypes[i], op.getResultName(i));
      body << ");\n";
    }
  }

  // Early exit if there are no operands.
  if (op.getNumOperands() == 0)
    return;

  // Handle the case where all operand types are in one group.
  if (allOperandTypes) {
    // If we have all operands together, use the full operand list directly.
    if (allOperands) {
      body << "  if (parser.resolveOperands(allOperands, allOperandTypes, "
              "allOperandLoc, result.operands))\n"
              "    return failure();\n";
      return;
    }

    // Otherwise, use llvm::concat to merge the disjoint operand lists together.
    // llvm::concat does not allow the case of a single range, so guard it here.
    body << "  if (parser.resolveOperands(";
    if (op.getNumOperands() > 1) {
      body << "::llvm::concat<const ::mlir::OpAsmParser::OperandType>(";
      llvm::interleaveComma(op.getOperands(), body, [&](auto &operand) {
        body << operand.name << "Operands";
      });
      body << ")";
    } else {
      body << op.operand_begin()->name << "Operands";
    }
    body << ", allOperandTypes, parser.getNameLoc(), result.operands))\n"
         << "    return failure();\n";
    return;
  }
  // Handle the case where all of the operands were grouped together.
  if (allOperands) {
    body << "  if (parser.resolveOperands(allOperands, ";

    // Group all of the operand types together to perform the resolution all at
    // once. Use llvm::concat to perform the merge. llvm::concat does not allow
    // the case of a single range, so guard it here.
    if (op.getNumOperands() > 1) {
      body << "::llvm::concat<const Type>(";
      llvm::interleaveComma(
          llvm::seq<int>(0, op.getNumOperands()), body, [&](int i) {
            body << "::llvm::ArrayRef<::mlir::Type>(";
            emitTypeResolver(operandTypes[i], op.getOperand(i).name);
            body << ")";
          });
      body << ")";
    } else {
      emitTypeResolver(operandTypes.front(), op.getOperand(0).name);
    }

    body << ", allOperandLoc, result.operands))\n"
         << "    return failure();\n";
    return;
  }

  // The final case is the one where each of the operands types are resolved
  // separately.
  for (unsigned i = 0, e = op.getNumOperands(); i != e; ++i) {
    NamedTypeConstraint &operand = op.getOperand(i);
    body << "  if (parser.resolveOperands(" << operand.name << "Operands, ";

    // Resolve the type of this operand.
    TypeResolution &operandType = operandTypes[i];
    emitTypeResolver(operandType, operand.name);

    // If the type is resolved by a non-variadic variable, index into the
    // resolved type list. This allows for resolving the types of a variadic
    // operand list from a non-variadic variable.
    bool verifyOperandAndTypeSize = true;
    if (auto *resolverVar = operandType.getVariable()) {
      if (!resolverVar->isVariadic() && !operandType.getVarTransformer()) {
        body << "[0]";
        verifyOperandAndTypeSize = false;
      }
    } else {
      verifyOperandAndTypeSize = !operandType.getBuilderIdx();
    }

    // Check to see if the sizes between the types and operands must match. If
    // they do, provide the operand location to select the proper resolution
    // overload.
    if (verifyOperandAndTypeSize)
      body << ", " << operand.name << "OperandsLoc";
    body << ", result.operands))\n    return failure();\n";
  }
}

void OperationFormat::genParserSuccessorResolution(Operator &op,
                                                   OpMethodBody &body) {
  // Check for the case where all successors were parsed.
  bool hasAllSuccessors = llvm::any_of(
      elements, [](auto &elt) { return isa<SuccessorsDirective>(elt.get()); });
  if (hasAllSuccessors) {
    body << "  result.addSuccessors(fullSuccessors);\n";
    return;
  }

  // Otherwise, handle each successor individually.
  for (const NamedSuccessor &successor : op.getSuccessors()) {
    if (successor.isVariadic())
      body << "  result.addSuccessors(" << successor.name << "Successors);\n";
    else
      body << "  result.addSuccessors(" << successor.name << "Successor);\n";
  }
}

void OperationFormat::genParserVariadicSegmentResolution(Operator &op,
                                                         OpMethodBody &body) {
  if (!allOperands && op.getTrait("OpTrait::AttrSizedOperandSegments")) {
    body << "  result.addAttribute(\"operand_segment_sizes\", "
         << "parser.getBuilder().getI32VectorAttr({";
    auto interleaveFn = [&](const NamedTypeConstraint &operand) {
      // If the operand is variadic emit the parsed size.
      if (operand.isVariableLength())
        body << "static_cast<int32_t>(" << operand.name << "Operands.size())";
      else
        body << "1";
    };
    llvm::interleaveComma(op.getOperands(), body, interleaveFn);
    body << "}));\n";
  }
}

//===----------------------------------------------------------------------===//
// PrinterGen

/// Generate the printer for the 'attr-dict' directive.
static void genAttrDictPrinter(OperationFormat &fmt, Operator &op,
                               OpMethodBody &body, bool withKeyword) {
  // Collect all of the attributes used in the format, these will be elided.
  SmallVector<const NamedAttribute *, 1> usedAttributes;
  for (auto &it : fmt.elements) {
    if (auto *attr = dyn_cast<AttributeVariable>(it.get()))
      usedAttributes.push_back(attr->getVar());
    // Collect the optional attributes.
    if (auto *opt = dyn_cast<OptionalElement>(it.get())) {
      for (auto &elem : opt->getElements()) {
        if (auto *attr = dyn_cast<AttributeVariable>(&elem))
          usedAttributes.push_back(attr->getVar());
      }
    }
  }

  body << "  p.printOptionalAttrDict" << (withKeyword ? "WithKeyword" : "")
       << "(getAttrs(), /*elidedAttrs=*/{";
  // Elide the variadic segment size attributes if necessary.
  if (!fmt.allOperands && op.getTrait("OpTrait::AttrSizedOperandSegments"))
    body << "\"operand_segment_sizes\", ";
  llvm::interleaveComma(usedAttributes, body, [&](const NamedAttribute *attr) {
    body << "\"" << attr->name << "\"";
  });
  body << "});\n";
}

/// Generate the printer for a literal value. `shouldEmitSpace` is true if a
/// space should be emitted before this element. `lastWasPunctuation` is true if
/// the previous element was a punctuation literal.
static void genLiteralPrinter(StringRef value, OpMethodBody &body,
                              bool &shouldEmitSpace, bool &lastWasPunctuation) {
  body << "  p";

  // Don't insert a space for certain punctuation.
  auto shouldPrintSpaceBeforeLiteral = [&] {
    if (value.size() != 1 && value != "->")
      return true;
    if (lastWasPunctuation)
      return !StringRef(">)}],").contains(value.front());
    return !StringRef("<>(){}[],").contains(value.front());
  };
  if (shouldEmitSpace && shouldPrintSpaceBeforeLiteral())
    body << " << \" \"";
  body << " << \"" << value << "\";\n";

  // Insert a space after certain literals.
  shouldEmitSpace =
      value.size() != 1 || !StringRef("<({[").contains(value.front());
  lastWasPunctuation = !(value.front() == '_' || isalpha(value.front()));
}

/// Generate the C++ for an operand to a (*-)type directive.
static OpMethodBody &genTypeOperandPrinter(Element *arg, OpMethodBody &body) {
  if (isa<OperandsDirective>(arg))
    return body << "getOperation()->getOperandTypes()";
  if (isa<ResultsDirective>(arg))
    return body << "getOperation()->getResultTypes()";
  auto *operand = dyn_cast<OperandVariable>(arg);
  auto *var = operand ? operand->getVar() : cast<ResultVariable>(arg)->getVar();
  if (var->isVariadic())
    return body << var->name << "().getTypes()";
  if (var->isOptional())
    return body << llvm::formatv(
               "({0}() ? ::llvm::ArrayRef<::mlir::Type>({0}().getType()) : "
               "::llvm::ArrayRef<::mlir::Type>())",
               var->name);
  return body << "::llvm::ArrayRef<::mlir::Type>(" << var->name
              << "().getType())";
}

/// Generate the code for printing the given element.
static void genElementPrinter(Element *element, OpMethodBody &body,
                              OperationFormat &fmt, Operator &op,
                              bool &shouldEmitSpace, bool &lastWasPunctuation) {
  if (LiteralElement *literal = dyn_cast<LiteralElement>(element))
    return genLiteralPrinter(literal->getLiteral(), body, shouldEmitSpace,
                             lastWasPunctuation);

  // Emit an optional group.
  if (OptionalElement *optional = dyn_cast<OptionalElement>(element)) {
    // Emit the check for the presence of the anchor element.
    Element *anchor = optional->getAnchor();
    if (auto *operand = dyn_cast<OperandVariable>(anchor)) {
      const NamedTypeConstraint *var = operand->getVar();
      if (var->isOptional())
        body << "  if (" << var->name << "()) {\n";
      else if (var->isVariadic())
        body << "  if (!" << var->name << "().empty()) {\n";
    } else {
      body << "  if (getAttr(\""
           << cast<AttributeVariable>(anchor)->getVar()->name << "\")) {\n";
    }

    // Emit each of the elements.
    for (Element &childElement : optional->getElements())
      genElementPrinter(&childElement, body, fmt, op, shouldEmitSpace,
                        lastWasPunctuation);
    body << "  }\n";
    return;
  }

  // Emit the attribute dictionary.
  if (auto *attrDict = dyn_cast<AttrDictDirective>(element)) {
    genAttrDictPrinter(fmt, op, body, attrDict->isWithKeyword());
    lastWasPunctuation = false;
    return;
  }

  // Optionally insert a space before the next element. The AttrDict printer
  // already adds a space as necessary.
  if (shouldEmitSpace || !lastWasPunctuation)
    body << "  p << \" \";\n";
  lastWasPunctuation = false;
  shouldEmitSpace = true;

  if (auto *attr = dyn_cast<AttributeVariable>(element)) {
    const NamedAttribute *var = attr->getVar();

    // If we are formatting as an enum, symbolize the attribute as a string.
    if (canFormatEnumAttr(var)) {
      const EnumAttr &enumAttr = cast<EnumAttr>(var->attr);
      body << "  p << \"\\\"\" << " << enumAttr.getSymbolToStringFnName() << "("
           << var->name << "()) << \"\\\"\";\n";
      return;
    }

    // Elide the attribute type if it is buildable.
    if (attr->getTypeBuilder())
      body << "  p.printAttributeWithoutType(" << var->name << "Attr());\n";
    else
      body << "  p.printAttribute(" << var->name << "Attr());\n";
  } else if (auto *operand = dyn_cast<OperandVariable>(element)) {
    if (operand->getVar()->isOptional()) {
      body << "  if (::mlir::Value value = " << operand->getVar()->name
           << "())\n"
           << "    p << value;\n";
    } else {
      body << "  p << " << operand->getVar()->name << "();\n";
    }
  } else if (auto *successor = dyn_cast<SuccessorVariable>(element)) {
    const NamedSuccessor *var = successor->getVar();
    if (var->isVariadic())
      body << "  ::llvm::interleaveComma(" << var->name << "(), p);\n";
    else
      body << "  p << " << var->name << "();\n";
  } else if (isa<OperandsDirective>(element)) {
    body << "  p << getOperation()->getOperands();\n";
  } else if (isa<SuccessorsDirective>(element)) {
    body << "  ::llvm::interleaveComma(getOperation()->getSuccessors(), p);\n";
  } else if (auto *dir = dyn_cast<TypeDirective>(element)) {
    body << "  p << ";
    genTypeOperandPrinter(dir->getOperand(), body) << ";\n";
  } else if (auto *dir = dyn_cast<FunctionalTypeDirective>(element)) {
    body << "  p.printFunctionalType(";
    genTypeOperandPrinter(dir->getInputs(), body) << ", ";
    genTypeOperandPrinter(dir->getResults(), body) << ");\n";
  } else {
    llvm_unreachable("unknown format element");
  }
}

void OperationFormat::genPrinter(Operator &op, OpClass &opClass) {
  auto &method = opClass.newMethod("void", "print", "OpAsmPrinter &p");
  auto &body = method.body();

  // Emit the operation name, trimming the prefix if this is the standard
  // dialect.
  body << "  p << \"";
  std::string opName = op.getOperationName();
  if (op.getDialectName() == "std")
    body << StringRef(opName).drop_front(4);
  else
    body << opName;
  body << "\";\n";

  // Flags for if we should emit a space, and if the last element was
  // punctuation.
  bool shouldEmitSpace = true, lastWasPunctuation = false;
  for (auto &element : elements)
    genElementPrinter(element.get(), body, *this, op, shouldEmitSpace,
                      lastWasPunctuation);
}

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
    caret,
    comma,
    equal,
    question,

    // Keywords.
    keyword_start,
    kw_attr_dict,
    kw_attr_dict_w_keyword,
    kw_functional_type,
    kw_operands,
    kw_results,
    kw_successors,
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
  FormatLexer(llvm::SourceMgr &mgr, Operator &op);

  /// Lex the next token and return it.
  Token lexToken();

  /// Emit an error to the lexer with the given location and message.
  Token emitError(llvm::SMLoc loc, const Twine &msg);
  Token emitError(const char *loc, const Twine &msg);

  Token emitErrorAndNote(llvm::SMLoc loc, const Twine &msg, const Twine &note);

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
  Operator &op;
  StringRef curBuffer;
  const char *curPtr;
};
} // end anonymous namespace

FormatLexer::FormatLexer(llvm::SourceMgr &mgr, Operator &op)
    : srcMgr(mgr), op(op) {
  curBuffer = srcMgr.getMemoryBuffer(mgr.getMainFileID())->getBuffer();
  curPtr = curBuffer.begin();
}

Token FormatLexer::emitError(llvm::SMLoc loc, const Twine &msg) {
  srcMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
  llvm::SrcMgr.PrintMessage(op.getLoc()[0], llvm::SourceMgr::DK_Note,
                            "in custom assembly format for this operation");
  return formToken(Token::error, loc.getPointer());
}
Token FormatLexer::emitErrorAndNote(llvm::SMLoc loc, const Twine &msg,
                                    const Twine &note) {
  srcMgr.PrintMessage(loc, llvm::SourceMgr::DK_Error, msg);
  llvm::SrcMgr.PrintMessage(op.getLoc()[0], llvm::SourceMgr::DK_Note,
                            "in custom assembly format for this operation");
  srcMgr.PrintMessage(loc, llvm::SourceMgr::DK_Note, note);
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
  case '^':
    return formToken(Token::caret, tokStart);
  case ',':
    return formToken(Token::comma, tokStart);
  case '=':
    return formToken(Token::equal, tokStart);
  case '?':
    return formToken(Token::question, tokStart);
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
  Token::Kind kind =
      llvm::StringSwitch<Token::Kind>(str)
          .Case("attr-dict", Token::kw_attr_dict)
          .Case("attr-dict-with-keyword", Token::kw_attr_dict_w_keyword)
          .Case("functional-type", Token::kw_functional_type)
          .Case("operands", Token::kw_operands)
          .Case("results", Token::kw_results)
          .Case("successors", Token::kw_successors)
          .Case("type", Token::kw_type)
          .Default(Token::identifier);
  return Token(kind, str);
}

//===----------------------------------------------------------------------===//
// FormatParser
//===----------------------------------------------------------------------===//

/// Function to find an element within the given range that has the same name as
/// 'name'.
template <typename RangeT>
static auto findArg(RangeT &&range, StringRef name) {
  auto it = llvm::find_if(range, [=](auto &arg) { return arg.name == name; });
  return it != range.end() ? &*it : nullptr;
}

namespace {
/// This class implements a parser for an instance of an operation assembly
/// format.
class FormatParser {
public:
  FormatParser(llvm::SourceMgr &mgr, OperationFormat &format, Operator &op)
      : lexer(mgr, op), curToken(lexer.lexToken()), fmt(format), op(op),
        seenOperandTypes(op.getNumOperands()),
        seenResultTypes(op.getNumResults()) {}

  /// Parse the operation assembly format.
  LogicalResult parse();

private:
  /// This struct represents a type resolution instance. It includes a specific
  /// type as well as an optional transformer to apply to that type in order to
  /// properly resolve the type of a variable.
  struct TypeResolutionInstance {
    ConstArgument resolver;
    Optional<StringRef> transformer;
  };

  /// An iterator over the elements of a format group.
  using ElementsIterT = llvm::pointee_iterator<
      std::vector<std::unique_ptr<Element>>::const_iterator>;

  /// Verify the state of operation attributes within the format.
  LogicalResult verifyAttributes(llvm::SMLoc loc);
  /// Verify the attribute elements at the back of the given stack of iterators.
  LogicalResult verifyAttributes(
      llvm::SMLoc loc,
      SmallVectorImpl<std::pair<ElementsIterT, ElementsIterT>> &iteratorStack);

  /// Verify the state of operation operands within the format.
  LogicalResult
  verifyOperands(llvm::SMLoc loc,
                 llvm::StringMap<TypeResolutionInstance> &variableTyResolver);

  /// Verify the state of operation results within the format.
  LogicalResult
  verifyResults(llvm::SMLoc loc,
                llvm::StringMap<TypeResolutionInstance> &variableTyResolver);

  /// Verify the state of operation successors within the format.
  LogicalResult verifySuccessors(llvm::SMLoc loc);

  /// Given the values of an `AllTypesMatch` trait, check for inferable type
  /// resolution.
  void handleAllTypesMatchConstraint(
      ArrayRef<StringRef> values,
      llvm::StringMap<TypeResolutionInstance> &variableTyResolver);
  /// Check for inferable type resolution given all operands, and or results,
  /// have the same type. If 'includeResults' is true, the results also have the
  /// same type as all of the operands.
  void handleSameTypesConstraint(
      llvm::StringMap<TypeResolutionInstance> &variableTyResolver,
      bool includeResults);
  /// Check for inferable type resolution based on another operand, result, or
  /// attribute.
  void handleTypesMatchConstraint(
      llvm::StringMap<TypeResolutionInstance> &variableTyResolver,
      llvm::Record def);

  /// Returns an argument or attribute with the given name that has been seen
  /// within the format.
  ConstArgument findSeenArg(StringRef name);

  /// Parse a specific element.
  LogicalResult parseElement(std::unique_ptr<Element> &element,
                             bool isTopLevel);
  LogicalResult parseVariable(std::unique_ptr<Element> &element,
                              bool isTopLevel);
  LogicalResult parseDirective(std::unique_ptr<Element> &element,
                               bool isTopLevel);
  LogicalResult parseLiteral(std::unique_ptr<Element> &element);
  LogicalResult parseOptional(std::unique_ptr<Element> &element,
                              bool isTopLevel);
  LogicalResult parseOptionalChildElement(
      std::vector<std::unique_ptr<Element>> &childElements,
      SmallPtrSetImpl<const NamedTypeConstraint *> &seenVariables,
      Optional<unsigned> &anchorIdx);

  /// Parse the various different directives.
  LogicalResult parseAttrDictDirective(std::unique_ptr<Element> &element,
                                       llvm::SMLoc loc, bool isTopLevel,
                                       bool withKeyword);
  LogicalResult parseFunctionalTypeDirective(std::unique_ptr<Element> &element,
                                             Token tok, bool isTopLevel);
  LogicalResult parseOperandsDirective(std::unique_ptr<Element> &element,
                                       llvm::SMLoc loc, bool isTopLevel);
  LogicalResult parseResultsDirective(std::unique_ptr<Element> &element,
                                      llvm::SMLoc loc, bool isTopLevel);
  LogicalResult parseSuccessorsDirective(std::unique_ptr<Element> &element,
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
  LogicalResult emitErrorAndNote(llvm::SMLoc loc, const Twine &msg,
                                 const Twine &note) {
    lexer.emitErrorAndNote(loc, msg, note);
    return failure();
  }

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  FormatLexer lexer;
  Token curToken;
  OperationFormat &fmt;
  Operator &op;

  // The following are various bits of format state used for verification
  // during parsing.
  bool hasAllOperands = false, hasAttrDict = false;
  bool hasAllSuccessors = false;
  llvm::SmallBitVector seenOperandTypes, seenResultTypes;
  llvm::DenseSet<const NamedTypeConstraint *> seenOperands;
  llvm::DenseSet<const NamedAttribute *> seenAttrs;
  llvm::DenseSet<const NamedSuccessor *> seenSuccessors;
  llvm::DenseSet<const NamedTypeConstraint *> optionalVariables;
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
    return emitError(loc, "'attr-dict' directive not found in "
                          "custom assembly format");

  // Check for any type traits that we can use for inferring types.
  llvm::StringMap<TypeResolutionInstance> variableTyResolver;
  for (const OpTrait &trait : op.getTraits()) {
    const llvm::Record &def = trait.getDef();
    if (def.isSubClassOf("AllTypesMatch")) {
      handleAllTypesMatchConstraint(def.getValueAsListOfStrings("values"),
                                    variableTyResolver);
    } else if (def.getName() == "SameTypeOperands") {
      handleSameTypesConstraint(variableTyResolver, /*includeResults=*/false);
    } else if (def.getName() == "SameOperandsAndResultType") {
      handleSameTypesConstraint(variableTyResolver, /*includeResults=*/true);
    } else if (def.isSubClassOf("TypesMatchWith")) {
      handleTypesMatchConstraint(variableTyResolver, def);
    }
  }

  // Verify the state of the various operation components.
  if (failed(verifyAttributes(loc)) ||
      failed(verifyResults(loc, variableTyResolver)) ||
      failed(verifyOperands(loc, variableTyResolver)) ||
      failed(verifySuccessors(loc)))
    return failure();

  // Check to see if we are formatting all of the operands.
  fmt.allOperands = llvm::any_of(fmt.elements, [](auto &elt) {
    return isa<OperandsDirective>(elt.get());
  });
  return success();
}

LogicalResult FormatParser::verifyAttributes(llvm::SMLoc loc) {
  // Check that there are no `:` literals after an attribute without a constant
  // type. The attribute grammar contains an optional trailing colon type, which
  // can lead to unexpected and generally unintended behavior. Given that, it is
  // better to just error out here instead.
  using ElementsIterT = llvm::pointee_iterator<
      std::vector<std::unique_ptr<Element>>::const_iterator>;
  SmallVector<std::pair<ElementsIterT, ElementsIterT>, 1> iteratorStack;
  iteratorStack.emplace_back(fmt.elements.begin(), fmt.elements.end());
  while (!iteratorStack.empty())
    if (failed(verifyAttributes(loc, iteratorStack)))
      return failure();
  return success();
}
/// Verify the attribute elements at the back of the given stack of iterators.
LogicalResult FormatParser::verifyAttributes(
    llvm::SMLoc loc,
    SmallVectorImpl<std::pair<ElementsIterT, ElementsIterT>> &iteratorStack) {
  auto &stackIt = iteratorStack.back();
  ElementsIterT &it = stackIt.first, e = stackIt.second;
  while (it != e) {
    Element *element = &*(it++);

    // Traverse into optional groups.
    if (auto *optional = dyn_cast<OptionalElement>(element)) {
      auto elements = optional->getElements();
      iteratorStack.emplace_back(elements.begin(), elements.end());
      return success();
    }

    // We are checking for an attribute element followed by a `:`, so there is
    // no need to check the end.
    if (it == e && iteratorStack.size() == 1)
      break;

    // Check for an attribute with a constant type builder, followed by a `:`.
    auto *prevAttr = dyn_cast<AttributeVariable>(element);
    if (!prevAttr || prevAttr->getTypeBuilder())
      continue;

    // Check the next iterator within the stack for literal elements.
    for (auto &nextItPair : iteratorStack) {
      ElementsIterT nextIt = nextItPair.first, nextE = nextItPair.second;
      for (; nextIt != nextE; ++nextIt) {
        // Skip any trailing optional groups or attribute dictionaries.
        if (isa<AttrDictDirective>(*nextIt) || isa<OptionalElement>(*nextIt))
          continue;

        // We are only interested in `:` literals.
        auto *literal = dyn_cast<LiteralElement>(&*nextIt);
        if (!literal || literal->getLiteral() != ":")
          break;

        // TODO: Use the location of the literal element itself.
        return emitError(
            loc, llvm::formatv("format ambiguity caused by `:` literal found "
                               "after attribute `{0}` which does not have "
                               "a buildable type",
                               prevAttr->getVar()->name));
      }
    }
  }
  iteratorStack.pop_back();
  return success();
}

LogicalResult FormatParser::verifyOperands(
    llvm::SMLoc loc,
    llvm::StringMap<TypeResolutionInstance> &variableTyResolver) {
  // Check that all of the operands are within the format, and their types can
  // be inferred.
  auto &buildableTypes = fmt.buildableTypes;
  for (unsigned i = 0, e = op.getNumOperands(); i != e; ++i) {
    NamedTypeConstraint &operand = op.getOperand(i);

    // Check that the operand itself is in the format.
    if (!hasAllOperands && !seenOperands.count(&operand)) {
      return emitErrorAndNote(loc,
                              "operand #" + Twine(i) + ", named '" +
                                  operand.name + "', not found",
                              "suggest adding a '$" + operand.name +
                                  "' directive to the custom assembly format");
    }

    // Check that the operand type is in the format, or that it can be inferred.
    if (fmt.allOperandTypes || seenOperandTypes.test(i))
      continue;

    // Check to see if we can infer this type from another variable.
    auto varResolverIt = variableTyResolver.find(op.getOperand(i).name);
    if (varResolverIt != variableTyResolver.end()) {
      TypeResolutionInstance &resolver = varResolverIt->second;
      fmt.operandTypes[i].setResolver(resolver.resolver, resolver.transformer);
      continue;
    }

    // Similarly to results, allow a custom builder for resolving the type if
    // we aren't using the 'operands' directive.
    Optional<StringRef> builder = operand.constraint.getBuilderCall();
    if (!builder || (hasAllOperands && operand.isVariableLength())) {
      return emitErrorAndNote(
          loc,
          "type of operand #" + Twine(i) + ", named '" + operand.name +
              "', is not buildable and a buildable type cannot be inferred",
          "suggest adding a type constraint to the operation or adding a "
          "'type($" +
              operand.name + ")' directive to the " + "custom assembly format");
    }
    auto it = buildableTypes.insert({*builder, buildableTypes.size()});
    fmt.operandTypes[i].setBuilderIdx(it.first->second);
  }
  return success();
}

LogicalResult FormatParser::verifyResults(
    llvm::SMLoc loc,
    llvm::StringMap<TypeResolutionInstance> &variableTyResolver) {
  // If we format all of the types together, there is nothing to check.
  if (fmt.allResultTypes)
    return success();

  // Check that all of the result types can be inferred.
  auto &buildableTypes = fmt.buildableTypes;
  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
    if (seenResultTypes.test(i))
      continue;

    // Check to see if we can infer this type from another variable.
    auto varResolverIt = variableTyResolver.find(op.getResultName(i));
    if (varResolverIt != variableTyResolver.end()) {
      TypeResolutionInstance resolver = varResolverIt->second;
      fmt.resultTypes[i].setResolver(resolver.resolver, resolver.transformer);
      continue;
    }

    // If the result is not variable length, allow for the case where the type
    // has a builder that we can use.
    NamedTypeConstraint &result = op.getResult(i);
    Optional<StringRef> builder = result.constraint.getBuilderCall();
    if (!builder || result.isVariableLength()) {
      return emitErrorAndNote(
          loc,
          "type of result #" + Twine(i) + ", named '" + result.name +
              "', is not buildable and a buildable type cannot be inferred",
          "suggest adding a type constraint to the operation or adding a "
          "'type($" +
              result.name + ")' directive to the " + "custom assembly format");
    }
    // Note in the format that this result uses the custom builder.
    auto it = buildableTypes.insert({*builder, buildableTypes.size()});
    fmt.resultTypes[i].setBuilderIdx(it.first->second);
  }
  return success();
}

LogicalResult FormatParser::verifySuccessors(llvm::SMLoc loc) {
  // Check that all of the successors are within the format.
  if (hasAllSuccessors)
    return success();

  for (unsigned i = 0, e = op.getNumSuccessors(); i != e; ++i) {
    const NamedSuccessor &successor = op.getSuccessor(i);
    if (!seenSuccessors.count(&successor)) {
      return emitErrorAndNote(loc,
                              "successor #" + Twine(i) + ", named '" +
                                  successor.name + "', not found",
                              "suggest adding a '$" + successor.name +
                                  "' directive to the custom assembly format");
    }
  }
  return success();
}

void FormatParser::handleAllTypesMatchConstraint(
    ArrayRef<StringRef> values,
    llvm::StringMap<TypeResolutionInstance> &variableTyResolver) {
  for (unsigned i = 0, e = values.size(); i != e; ++i) {
    // Check to see if this value matches a resolved operand or result type.
    ConstArgument arg = findSeenArg(values[i]);
    if (!arg)
      continue;

    // Mark this value as the type resolver for the other variables.
    for (unsigned j = 0; j != i; ++j)
      variableTyResolver[values[j]] = {arg, llvm::None};
    for (unsigned j = i + 1; j != e; ++j)
      variableTyResolver[values[j]] = {arg, llvm::None};
  }
}

void FormatParser::handleSameTypesConstraint(
    llvm::StringMap<TypeResolutionInstance> &variableTyResolver,
    bool includeResults) {
  const NamedTypeConstraint *resolver = nullptr;
  int resolvedIt = -1;

  // Check to see if there is an operand or result to use for the resolution.
  if ((resolvedIt = seenOperandTypes.find_first()) != -1)
    resolver = &op.getOperand(resolvedIt);
  else if (includeResults && (resolvedIt = seenResultTypes.find_first()) != -1)
    resolver = &op.getResult(resolvedIt);
  else
    return;

  // Set the resolvers for each operand and result.
  for (unsigned i = 0, e = op.getNumOperands(); i != e; ++i)
    if (!seenOperandTypes.test(i) && !op.getOperand(i).name.empty())
      variableTyResolver[op.getOperand(i).name] = {resolver, llvm::None};
  if (includeResults) {
    for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
      if (!seenResultTypes.test(i) && !op.getResultName(i).empty())
        variableTyResolver[op.getResultName(i)] = {resolver, llvm::None};
  }
}

void FormatParser::handleTypesMatchConstraint(
    llvm::StringMap<TypeResolutionInstance> &variableTyResolver,
    llvm::Record def) {
  StringRef lhsName = def.getValueAsString("lhs");
  StringRef rhsName = def.getValueAsString("rhs");
  StringRef transformer = def.getValueAsString("transformer");
  if (ConstArgument arg = findSeenArg(lhsName))
    variableTyResolver[rhsName] = {arg, transformer};
}

ConstArgument FormatParser::findSeenArg(StringRef name) {
  if (const NamedTypeConstraint *arg = findArg(op.getOperands(), name))
    return seenOperandTypes.test(arg - op.operand_begin()) ? arg : nullptr;
  if (const NamedTypeConstraint *arg = findArg(op.getResults(), name))
    return seenResultTypes.test(arg - op.result_begin()) ? arg : nullptr;
  if (const NamedAttribute *attr = findArg(op.getAttributes(), name))
    return seenAttrs.find_as(attr) != seenAttrs.end() ? attr : nullptr;
  return nullptr;
}

LogicalResult FormatParser::parseElement(std::unique_ptr<Element> &element,
                                         bool isTopLevel) {
  // Directives.
  if (curToken.isKeyword())
    return parseDirective(element, isTopLevel);
  // Literals.
  if (curToken.getKind() == Token::literal)
    return parseLiteral(element);
  // Optionals.
  if (curToken.getKind() == Token::l_paren)
    return parseOptional(element, isTopLevel);
  // Variables.
  if (curToken.getKind() == Token::variable)
    return parseVariable(element, isTopLevel);
  return emitError(curToken.getLoc(),
                   "expected directive, literal, variable, or optional group");
}

LogicalResult FormatParser::parseVariable(std::unique_ptr<Element> &element,
                                          bool isTopLevel) {
  Token varTok = curToken;
  consumeToken();

  StringRef name = varTok.getSpelling().drop_front();
  llvm::SMLoc loc = varTok.getLoc();

  // Check that the parsed argument is something actually registered on the
  // op.
  /// Attributes
  if (const NamedAttribute *attr = findArg(op.getAttributes(), name)) {
    if (isTopLevel && !seenAttrs.insert(attr).second)
      return emitError(loc, "attribute '" + name + "' is already bound");
    element = std::make_unique<AttributeVariable>(attr);
    return success();
  }
  /// Operands
  if (const NamedTypeConstraint *operand = findArg(op.getOperands(), name)) {
    if (isTopLevel) {
      if (hasAllOperands || !seenOperands.insert(operand).second)
        return emitError(loc, "operand '" + name + "' is already bound");
    }
    element = std::make_unique<OperandVariable>(operand);
    return success();
  }
  /// Results.
  if (const auto *result = findArg(op.getResults(), name)) {
    if (isTopLevel)
      return emitError(loc, "results can not be used at the top level");
    element = std::make_unique<ResultVariable>(result);
    return success();
  }
  /// Successors.
  if (const auto *successor = findArg(op.getSuccessors(), name)) {
    if (!isTopLevel)
      return emitError(loc, "successors can only be used at the top level");
    if (hasAllSuccessors || !seenSuccessors.insert(successor).second)
      return emitError(loc, "successor '" + name + "' is already bound");
    element = std::make_unique<SuccessorVariable>(successor);
    return success();
  }
  return emitError(
      loc, "expected variable to refer to an argument, result, or successor");
}

LogicalResult FormatParser::parseDirective(std::unique_ptr<Element> &element,
                                           bool isTopLevel) {
  Token dirTok = curToken;
  consumeToken();

  switch (dirTok.getKind()) {
  case Token::kw_attr_dict:
    return parseAttrDictDirective(element, dirTok.getLoc(), isTopLevel,
                                  /*withKeyword=*/false);
  case Token::kw_attr_dict_w_keyword:
    return parseAttrDictDirective(element, dirTok.getLoc(), isTopLevel,
                                  /*withKeyword=*/true);
  case Token::kw_functional_type:
    return parseFunctionalTypeDirective(element, dirTok, isTopLevel);
  case Token::kw_operands:
    return parseOperandsDirective(element, dirTok.getLoc(), isTopLevel);
  case Token::kw_results:
    return parseResultsDirective(element, dirTok.getLoc(), isTopLevel);
  case Token::kw_successors:
    return parseSuccessorsDirective(element, dirTok.getLoc(), isTopLevel);
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

LogicalResult FormatParser::parseOptional(std::unique_ptr<Element> &element,
                                          bool isTopLevel) {
  llvm::SMLoc curLoc = curToken.getLoc();
  if (!isTopLevel)
    return emitError(curLoc, "optional groups can only be used as top-level "
                             "elements");
  consumeToken();

  // Parse the child elements for this optional group.
  std::vector<std::unique_ptr<Element>> elements;
  SmallPtrSet<const NamedTypeConstraint *, 8> seenVariables;
  Optional<unsigned> anchorIdx;
  do {
    if (failed(parseOptionalChildElement(elements, seenVariables, anchorIdx)))
      return failure();
  } while (curToken.getKind() != Token::r_paren);
  consumeToken();
  if (failed(parseToken(Token::question, "expected '?' after optional group")))
    return failure();

  // The optional group is required to have an anchor.
  if (!anchorIdx)
    return emitError(curLoc, "optional group specified no anchor element");

  // The first element of the group must be one that can be parsed/printed in an
  // optional fashion.
  Element *firstElement = &*elements.front();
  if (!isa<AttributeVariable>(firstElement) &&
      !isa<LiteralElement>(firstElement) && !isa<OperandVariable>(firstElement))
    return emitError(curLoc, "first element of an operand group must be an "
                             "attribute, literal, or operand");

  // After parsing all of the elements, ensure that all type directives refer
  // only to elements within the group.
  auto checkTypeOperand = [&](Element *typeEle) {
    auto *opVar = dyn_cast<OperandVariable>(typeEle);
    const NamedTypeConstraint *var = opVar ? opVar->getVar() : nullptr;
    if (!seenVariables.count(var))
      return emitError(curLoc, "type directive can only refer to variables "
                               "within the optional group");
    return success();
  };
  for (auto &ele : elements) {
    if (auto *typeEle = dyn_cast<TypeDirective>(ele.get())) {
      if (failed(checkTypeOperand(typeEle->getOperand())))
        return failure();
    } else if (auto *typeEle = dyn_cast<FunctionalTypeDirective>(ele.get())) {
      if (failed(checkTypeOperand(typeEle->getInputs())) ||
          failed(checkTypeOperand(typeEle->getResults())))
        return failure();
    }
  }

  optionalVariables.insert(seenVariables.begin(), seenVariables.end());
  element = std::make_unique<OptionalElement>(std::move(elements), *anchorIdx);
  return success();
}

LogicalResult FormatParser::parseOptionalChildElement(
    std::vector<std::unique_ptr<Element>> &childElements,
    SmallPtrSetImpl<const NamedTypeConstraint *> &seenVariables,
    Optional<unsigned> &anchorIdx) {
  llvm::SMLoc childLoc = curToken.getLoc();
  childElements.push_back({});
  if (failed(parseElement(childElements.back(), /*isTopLevel=*/true)))
    return failure();

  // Check to see if this element is the anchor of the optional group.
  bool isAnchor = curToken.getKind() == Token::caret;
  if (isAnchor) {
    if (anchorIdx)
      return emitError(childLoc, "only one element can be marked as the anchor "
                                 "of an optional group");
    anchorIdx = childElements.size() - 1;
    consumeToken();
  }

  return TypeSwitch<Element *, LogicalResult>(childElements.back().get())
      // All attributes can be within the optional group, but only optional
      // attributes can be the anchor.
      .Case([&](AttributeVariable *attrEle) {
        if (isAnchor && !attrEle->getVar()->attr.isOptional())
          return emitError(childLoc, "only optional attributes can be used to "
                                     "anchor an optional group");
        return success();
      })
      // Only optional-like(i.e. variadic) operands can be within an optional
      // group.
      .Case<OperandVariable>([&](OperandVariable *ele) {
        if (!ele->getVar()->isVariableLength())
          return emitError(childLoc, "only variable length operands can be "
                                     "used within an optional group");
        seenVariables.insert(ele->getVar());
        return success();
      })
      // Literals and type directives may be used, but they can't anchor the
      // group.
      .Case<LiteralElement, TypeDirective, FunctionalTypeDirective>(
          [&](Element *) {
            if (isAnchor)
              return emitError(childLoc, "only variables can be used to anchor "
                                         "an optional group");
            return success();
          })
      .Default([&](Element *) {
        return emitError(childLoc, "only literals, types, and variables can be "
                                   "used within an optional group");
      });
}

LogicalResult
FormatParser::parseAttrDictDirective(std::unique_ptr<Element> &element,
                                     llvm::SMLoc loc, bool isTopLevel,
                                     bool withKeyword) {
  if (!isTopLevel)
    return emitError(loc, "'attr-dict' directive can only be used as a "
                          "top-level directive");
  if (hasAttrDict)
    return emitError(loc, "'attr-dict' directive has already been seen");

  hasAttrDict = true;
  element = std::make_unique<AttrDictDirective>(withKeyword);
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
FormatParser::parseSuccessorsDirective(std::unique_ptr<Element> &element,
                                       llvm::SMLoc loc, bool isTopLevel) {
  if (!isTopLevel)
    return emitError(loc,
                     "'successors' is only valid as a top-level directive");
  if (hasAllSuccessors || !seenSuccessors.empty())
    return emitError(loc, "'successors' directive creates overlap in format");
  hasAllSuccessors = true;
  element = std::make_unique<SuccessorsDirective>();
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
  // TODO: Operator doesn't expose all necessary functionality via
  // the const interface.
  Operator &op = const_cast<Operator &>(constOp);
  if (!op.hasAssemblyFormat())
    return;

  // Parse the format description.
  llvm::SourceMgr mgr;
  mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(op.getAssemblyFormat()), llvm::SMLoc());
  OperationFormat format(op);
  if (failed(FormatParser(mgr, format, op).parse())) {
    // Exit the process if format errors are treated as fatal.
    if (formatErrorIsFatal) {
      // Invoke the interrupt handlers to run the file cleanup handlers.
      llvm::sys::RunInterruptHandlers();
      std::exit(1);
    }
    return;
  }

  // Generate the printer and parser based on the parsed format.
  format.genParser(op, opClass);
  format.genPrinter(op, opClass);
}
