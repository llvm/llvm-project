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
  /// This class represents a specific resolver for an operand or result type.
  class TypeResolution {
  public:
    TypeResolution() = default;

    /// Get the index into the buildable types for this type, or None.
    Optional<int> getBuilderIdx() const { return builderIdx; }
    void setBuilderIdx(int idx) { builderIdx = idx; }

    /// Get the variable this type is resolved to, or None.
    Optional<StringRef> getVariable() const { return variableName; }
    void setVariable(StringRef variable) { variableName = variable; }

  private:
    /// If the type is resolved with a buildable type, this is the index into
    /// 'buildableTypes' in the parent format.
    Optional<int> builderIdx;
    /// If the type is resolved based upon another operand or result, this is
    /// the name of the variable that this type is resolved to.
    Optional<StringRef> variableName;
  };

  OperationFormat(const Operator &op)
      : allOperandTypes(false), allResultTypes(false) {
    operandTypes.resize(op.getNumOperands(), TypeResolution());
    resultTypes.resize(op.getNumResults(), TypeResolution());
  }

  /// Generate the operation parser from this format.
  void genParser(Operator &op, OpClass &opClass);
  /// Generate the c++ to resolve the types of operands and results during
  /// parsing.
  void genParserTypeResolution(Operator &op, OpMethodBody &body);

  /// Generate the operation printer from this format.
  void genPrinter(Operator &op, OpClass &opClass);

  /// The various elements in this format.
  std::vector<std::unique_ptr<Element>> elements;

  /// A flag indicating if all operand/result types were seen. If the format
  /// contains these, it can not contain individual type resolvers.
  bool allOperandTypes, allResultTypes;

  /// A map of buildable types to indices.
  llvm::MapVector<StringRef, int, llvm::StringMap<int>> buildableTypes;

  /// The index of the buildable type, if valid, for every operand and result.
  std::vector<TypeResolution> operandTypes, resultTypes;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Parser Gen

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

/// The code snippet used to generate a parser call for an operand.
///
/// {0}: The name of the operand.
const char *const variadicOperandParserCode = R"(
  llvm::SMLoc {0}OperandsLoc = parser.getCurrentLocation();
  (void){0}OperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> {0}Operands;
  if (parser.parseOperandList({0}Operands))
    return failure();
)";
const char *const operandParserCode = R"(
  llvm::SMLoc {0}OperandsLoc = parser.getCurrentLocation();
  (void){0}OperandsLoc;
  OpAsmParser::OperandType {0}RawOperands[1];
  if (parser.parseOperand({0}RawOperands[0]))
    return failure();
  ArrayRef<OpAsmParser::OperandType> {0}Operands({0}RawOperands);
)";

/// The code snippet used to generate a parser call for a type list.
///
/// {0}: The name for the type list.
const char *const variadicTypeParserCode = R"(
  SmallVector<Type, 1> {0}Types;
  if (parser.parseTypeList({0}Types))
    return failure();
)";
const char *const typeParserCode = R"(
  Type {0}RawTypes[1] = {{nullptr};
  if (parser.parseType({0}RawTypes[0]))
    return failure();
  ArrayRef<Type> {0}Types({0}RawTypes);
)";

/// The code snippet used to generate a parser call for a functional type.
///
/// {0}: The name for the input type list.
/// {1}: The name for the result type list.
const char *const functionalTypeParserCode = R"(
  FunctionType {0}__{1}_functionType;
  if (parser.parseType({0}__{1}_functionType))
    return failure();
  ArrayRef<Type> {0}Types = {0}__{1}_functionType.getInputs();
  ArrayRef<Type> {1}Types = {0}__{1}_functionType.getResults();
)";

/// Get the name used for the type list for the given type directive operand.
/// 'isVariadic' is set to true if the operand has variadic types.
static StringRef getTypeListName(Element *arg, bool &isVariadic) {
  if (auto *operand = dyn_cast<OperandVariable>(arg)) {
    isVariadic = operand->getVar()->isVariadic();
    return operand->getVar()->name;
  }
  if (auto *result = dyn_cast<ResultVariable>(arg)) {
    isVariadic = result->getVar()->isVariadic();
    return result->getVar()->name;
  }
  isVariadic = true;
  if (isa<OperandsDirective>(arg))
    return "allOperand";
  if (isa<ResultsDirective>(arg))
    return "allResult";
  llvm_unreachable("unknown 'type' directive argument");
}

/// Generate the parser for a literal value.
static void genLiteralParser(StringRef value, OpMethodBody &body) {
  body << "  if (parser.parse";

  // Handle the case of a keyword/identifier.
  if (value.front() == '_' || isalpha(value.front())) {
    body << "Keyword(\"" << value << "\")";
  } else {
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
  body << ")\n    return failure();\n";
}

void OperationFormat::genParser(Operator &op, OpClass &opClass) {
  auto &method = opClass.newMethod(
      "ParseResult", "parse", "OpAsmParser &parser, OperationState &result",
      OpMethod::MP_Static);
  auto &body = method.body();

  // A format context used when parsing attributes with buildable types.
  FmtContext attrTypeCtx;
  attrTypeCtx.withBuilder("parser.getBuilder()");

  // Generate parsers for each of the elements.
  for (auto &element : elements) {
    /// Literals.
    if (LiteralElement *literal = dyn_cast<LiteralElement>(element.get())) {
      genLiteralParser(literal->getLiteral(), body);

      /// Arguments.
    } else if (auto *attr = dyn_cast<AttributeVariable>(element.get())) {
      const NamedAttribute *var = attr->getVar();

      // If this attribute has a buildable type, use that when parsing the
      // attribute.
      std::string attrTypeStr;
      if (Optional<Type> attrType = var->attr.getValueType()) {
        if (Optional<StringRef> typeBuilder = attrType->getBuilderCall()) {
          llvm::raw_string_ostream os(attrTypeStr);
          os << ", " << tgfmt(*typeBuilder, &attrTypeCtx);
        }
      }

      body << formatv(attrParserCode, var->attr.getStorageType(), var->name,
                      attrTypeStr);
    } else if (auto *operand = dyn_cast<OperandVariable>(element.get())) {
      bool isVariadic = operand->getVar()->isVariadic();
      body << formatv(isVariadic ? variadicOperandParserCode
                                 : operandParserCode,
                      operand->getVar()->name);

      /// Directives.
    } else if (isa<AttrDictDirective>(element.get())) {
      body << "  if (parser.parseOptionalAttrDict(result.attributes))\n"
           << "    return failure();\n";
    } else if (isa<OperandsDirective>(element.get())) {
      body << "  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();\n"
           << "  SmallVector<OpAsmParser::OperandType, 4> allOperands;\n"
           << "  if (parser.parseOperandList(allOperands))\n"
           << "    return failure();\n";
    } else if (auto *dir = dyn_cast<TypeDirective>(element.get())) {
      bool isVariadic = false;
      StringRef listName = getTypeListName(dir->getOperand(), isVariadic);
      body << formatv(isVariadic ? variadicTypeParserCode : typeParserCode,
                      listName);
    } else if (auto *dir = dyn_cast<FunctionalTypeDirective>(element.get())) {
      bool ignored = false;
      body << formatv(functionalTypeParserCode,
                      getTypeListName(dir->getInputs(), ignored),
                      getTypeListName(dir->getResults(), ignored));
    } else {
      llvm_unreachable("unknown format element");
    }
  }

  // Generate the code to resolve the operand and result types now that they
  // have been parsed.
  genParserTypeResolution(op, body);
  body << "  return success();\n";
}

void OperationFormat::genParserTypeResolution(Operator &op,
                                              OpMethodBody &body) {
  // Initialize the set of buildable types.
  if (!buildableTypes.empty()) {
    body << "  Builder &builder = parser.getBuilder();\n";

    FmtContext typeBuilderCtx;
    typeBuilderCtx.withBuilder("builder");
    for (auto &it : buildableTypes)
      body << "  Type odsBuildableType" << it.second << " = "
           << tgfmt(it.first, &typeBuilderCtx) << ";\n";
  }

  // Resolve each of the result types.
  if (allResultTypes) {
    body << "  result.addTypes(allResultTypes);\n";
  } else {
    for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
      body << "  result.addTypes(";
      if (Optional<int> val = resultTypes[i].getBuilderIdx())
        body << "odsBuildableType" << *val;
      else if (Optional<StringRef> var = resultTypes[i].getVariable())
        body << *var << "Types";
      else
        body << op.getResultName(i) << "Types";
      body << ");\n";
    }
  }

  // Early exit if there are no operands.
  if (op.getNumOperands() == 0)
    return;

  // Flag indicating if operands were dumped all together in a group.
  bool hasAllOperands = llvm::any_of(
      elements, [](auto &elt) { return isa<OperandsDirective>(elt.get()); });

  // Handle the case where all operand types are in one group.
  if (allOperandTypes) {
    // If we have all operands together, use the full operand list directly.
    if (hasAllOperands) {
      body << "  if (parser.resolveOperands(allOperands, allOperandTypes, "
              "allOperandLoc, result.operands))\n"
              "    return failure();\n";
      return;
    }

    // Otherwise, use llvm::concat to merge the disjoint operand lists together.
    // llvm::concat does not allow the case of a single range, so guard it here.
    body << "  if (parser.resolveOperands(";
    if (op.getNumOperands() > 1) {
      body << "llvm::concat<const OpAsmParser::OperandType>(";
      interleaveComma(op.getOperands(), body, [&](auto &operand) {
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
  if (hasAllOperands) {
    body << "  if (parser.resolveOperands(allOperands, ";

    auto emitOperandType = [&](int idx) {
      if (Optional<int> val = operandTypes[idx].getBuilderIdx())
        body << "ArrayRef<Type>(odsBuildableType" << *val << ")";
      else if (Optional<StringRef> var = operandTypes[idx].getVariable())
        body << *var << "Types";
      else
        body << op.getOperand(idx).name << "Types";
    };

    // Group all of the operand types together to perform the resolution all at
    // once. Use llvm::concat to perform the merge. llvm::concat does not allow
    // the case of a single range, so guard it here.
    if (op.getNumOperands() > 1) {
      body << "llvm::concat<const Type>(";
      interleaveComma(llvm::seq<int>(0, op.getNumOperands()), body,
                      emitOperandType);
      body << ")";
    } else {
      emitOperandType(/*idx=*/0);
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
    if (Optional<int> val = operandTypes[i].getBuilderIdx())
      body << "odsBuildableType" << *val << ", ";
    else if (Optional<StringRef> var = operandTypes[i].getVariable())
      body << *var << "Types, " << operand.name << "OperandsLoc, ";
    else
      body << operand.name << "Types, " << operand.name << "OperandsLoc, ";
    body << "result.operands))\n    return failure();\n";
  }
}

//===----------------------------------------------------------------------===//
// PrinterGen

/// Generate the printer for the 'attr-dict' directive.
static void genAttrDictPrinter(OperationFormat &fmt, OpMethodBody &body) {
  // Collect all of the attributes used in the format, these will be elided.
  SmallVector<const NamedAttribute *, 1> usedAttributes;
  for (auto &it : fmt.elements)
    if (auto *attr = dyn_cast<AttributeVariable>(it.get()))
      usedAttributes.push_back(attr->getVar());

  body << "  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{";
  interleaveComma(usedAttributes, body, [&](const NamedAttribute *attr) {
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

/// Generate the c++ for an operand to a (*-)type directive.
static OpMethodBody &genTypeOperandPrinter(Element *arg, OpMethodBody &body) {
  if (isa<OperandsDirective>(arg))
    return body << "getOperation()->getOperandTypes()";
  if (isa<ResultsDirective>(arg))
    return body << "getOperation()->getResultTypes()";
  auto *operand = dyn_cast<OperandVariable>(arg);
  auto *var = operand ? operand->getVar() : cast<ResultVariable>(arg)->getVar();
  if (var->isVariadic())
    return body << var->name << "().getTypes()";
  return body << "ArrayRef<Type>(" << var->name << "().getType())";
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
  for (auto &element : elements) {
    // Emit a literal element.
    if (LiteralElement *literal = dyn_cast<LiteralElement>(element.get())) {
      genLiteralPrinter(literal->getLiteral(), body, shouldEmitSpace,
                        lastWasPunctuation);
      continue;
    }

    // Emit the attribute dictionary.
    if (isa<AttrDictDirective>(element.get())) {
      genAttrDictPrinter(*this, body);
      lastWasPunctuation = false;
      continue;
    }

    // Optionally insert a space before the next element. The AttrDict printer
    // already adds a space as necessary.
    if (shouldEmitSpace || !lastWasPunctuation)
      body << "  p << \" \";\n";
    lastWasPunctuation = false;
    shouldEmitSpace = true;

    if (auto *attr = dyn_cast<AttributeVariable>(element.get())) {
      const NamedAttribute *var = attr->getVar();

      // Elide the attribute type if it is buildable..
      Optional<Type> attrType = var->attr.getValueType();
      if (attrType && attrType->getBuilderCall())
        body << "  p.printAttributeWithoutType(" << var->name << "Attr());\n";
      else
        body << "  p.printAttribute(" << var->name << "Attr());\n";
    } else if (auto *operand = dyn_cast<OperandVariable>(element.get())) {
      body << "  p << " << operand->getVar()->name << "();\n";
    } else if (isa<OperandsDirective>(element.get())) {
      body << "  p << getOperation()->getOperands();\n";
    } else if (auto *dir = dyn_cast<TypeDirective>(element.get())) {
      body << "  p << ";
      genTypeOperandPrinter(dir->getOperand(), body) << ";\n";
    } else if (auto *dir = dyn_cast<FunctionalTypeDirective>(element.get())) {
      body << "  p.printFunctionalType(";
      genTypeOperandPrinter(dir->getInputs(), body) << ", ";
      genTypeOperandPrinter(dir->getResults(), body) << ");\n";
    } else {
      llvm_unreachable("unknown format element");
    }
  }
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

/// Function to find an element within the given range that has the same name as
/// 'name'.
template <typename RangeT> static auto findArg(RangeT &&range, StringRef name) {
  auto it = llvm::find_if(range, [=](auto &arg) { return arg.name == name; });
  return it != range.end() ? &*it : nullptr;
}

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
  /// Given the values of an `AllTypesMatch` trait, check for inferrable type
  /// resolution.
  void handleAllTypesMatchConstraint(
      ArrayRef<StringRef> values,
      llvm::StringMap<const NamedTypeConstraint *> &variableTyResolver);
  /// Check for inferrable type resolution given all operands, and or results,
  /// have the same type. If 'includeResults' is true, the results also have the
  /// same type as all of the operands.
  void handleSameTypesConstraint(
      llvm::StringMap<const NamedTypeConstraint *> &variableTyResolver,
      bool includeResults);

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

  // The following are various bits of format state used for verification
  // during parsing.
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

  // Check for any type traits that we can use for inferring types.
  llvm::StringMap<const NamedTypeConstraint *> variableTyResolver;
  for (const OpTrait &trait : op.getTraits()) {
    const llvm::Record &def = trait.getDef();
    if (def.isSubClassOf("AllTypesMatch"))
      handleAllTypesMatchConstraint(def.getValueAsListOfStrings("values"),
                                    variableTyResolver);
    else if (def.getName() == "SameTypeOperands")
      handleSameTypesConstraint(variableTyResolver, /*includeResults=*/false);
    else if (def.getName() == "SameOperandsAndResultType")
      handleSameTypesConstraint(variableTyResolver, /*includeResults=*/true);
  }

  // Check that all of the result types can be inferred.
  auto &buildableTypes = fmt.buildableTypes;
  if (!fmt.allResultTypes) {
    for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
      if (seenResultTypes.test(i))
        continue;

      // Check to see if we can infer this type from another variable.
      auto varResolverIt = variableTyResolver.find(op.getResultName(i));
      if (varResolverIt != variableTyResolver.end()) {
        fmt.resultTypes[i].setVariable(varResolverIt->second->name);
        continue;
      }

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
      fmt.resultTypes[i].setBuilderIdx(it.first->second);
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
    if (fmt.allOperandTypes || seenOperandTypes.test(i))
      continue;

    // Check to see if we can infer this type from another variable.
    auto varResolverIt = variableTyResolver.find(op.getOperand(i).name);
    if (varResolverIt != variableTyResolver.end()) {
      fmt.operandTypes[i].setVariable(varResolverIt->second->name);
      continue;
    }

    // Similarly to results, allow a custom builder for resolving the type if
    // we aren't using the 'operands' directive.
    Optional<StringRef> builder = operand.constraint.getBuilderCall();
    if (!builder || (hasAllOperands && operand.isVariadic())) {
      return emitError(loc, "format missing instance of operand #" + Twine(i) +
                                "('" + operand.name + "') type");
    }
    auto it = buildableTypes.insert({*builder, buildableTypes.size()});
    fmt.operandTypes[i].setBuilderIdx(it.first->second);
  }
  return success();
}

void FormatParser::handleAllTypesMatchConstraint(
    ArrayRef<StringRef> values,
    llvm::StringMap<const NamedTypeConstraint *> &variableTyResolver) {
  for (unsigned i = 0, e = values.size(); i != e; ++i) {
    // Check to see if this value matches a resolved operand or result type.
    const NamedTypeConstraint *arg = nullptr;
    if ((arg = findArg(op.getOperands(), values[i]))) {
      if (!seenOperandTypes.test(arg - op.operand_begin()))
        continue;
    } else if ((arg = findArg(op.getResults(), values[i]))) {
      if (!seenResultTypes.test(arg - op.result_begin()))
        continue;
    } else {
      continue;
    }

    // Mark this value as the type resolver for the other variables.
    for (unsigned j = 0; j != i; ++j)
      variableTyResolver[values[j]] = arg;
    for (unsigned j = i + 1; j != e; ++j)
      variableTyResolver[values[j]] = arg;
  }
}

void FormatParser::handleSameTypesConstraint(
    llvm::StringMap<const NamedTypeConstraint *> &variableTyResolver,
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
      variableTyResolver[op.getOperand(i).name] = resolver;
  if (includeResults) {
    for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
      if (!seenResultTypes.test(i) && !op.getResultName(i).empty())
        variableTyResolver[op.getResultName(i)] = resolver;
  }
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

  // Check that the parsed argument is something actually registered on the op.
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
