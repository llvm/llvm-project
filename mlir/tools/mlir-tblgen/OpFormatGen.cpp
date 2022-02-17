//===- OpFormatGen.cpp - MLIR operation asm format generator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpFormatGen.h"
#include "FormatGen.h"
#include "OpClass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#define DEBUG_TYPE "mlir-tblgen-opformatgen"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// VariableElement

namespace {
/// This class represents an instance of an op variable element. A variable
/// refers to something registered on the operation itself, e.g. an operand,
/// result, attribute, region, or successor.
template <typename VarT, VariableElement::Kind VariableKind>
class OpVariableElement : public VariableElementBase<VariableKind> {
public:
  using Base = OpVariableElement<VarT, VariableKind>;

  /// Create an op variable element with the variable value.
  OpVariableElement(const VarT *var) : var(var) {}

  /// Get the variable.
  const VarT *getVar() { return var; }

protected:
  /// The op variable, e.g. a type or attribute constraint.
  const VarT *var;
};

/// This class represents a variable that refers to an attribute argument.
struct AttributeVariable
    : public OpVariableElement<NamedAttribute, VariableElement::Attribute> {
  using Base::Base;

  /// Return the constant builder call for the type of this attribute, or None
  /// if it doesn't have one.
  llvm::Optional<StringRef> getTypeBuilder() const {
    llvm::Optional<Type> attrType = var->attr.getValueType();
    return attrType ? attrType->getBuilderCall() : llvm::None;
  }

  /// Return if this attribute refers to a UnitAttr.
  bool isUnitAttr() const {
    return var->attr.getBaseAttr().getAttrDefName() == "UnitAttr";
  }

  /// Indicate if this attribute is printed "qualified" (that is it is
  /// prefixed with the `#dialect.mnemonic`).
  bool shouldBeQualified() { return shouldBeQualifiedFlag; }
  void setShouldBeQualified(bool qualified = true) {
    shouldBeQualifiedFlag = qualified;
  }

private:
  bool shouldBeQualifiedFlag = false;
};

/// This class represents a variable that refers to an operand argument.
using OperandVariable =
    OpVariableElement<NamedTypeConstraint, VariableElement::Operand>;

/// This class represents a variable that refers to a result.
using ResultVariable =
    OpVariableElement<NamedTypeConstraint, VariableElement::Result>;

/// This class represents a variable that refers to a region.
using RegionVariable = OpVariableElement<NamedRegion, VariableElement::Region>;

/// This class represents a variable that refers to a successor.
using SuccessorVariable =
    OpVariableElement<NamedSuccessor, VariableElement::Successor>;
} // namespace

//===----------------------------------------------------------------------===//
// DirectiveElement

namespace {
/// This class represents the `operands` directive. This directive represents
/// all of the operands of an operation.
using OperandsDirective = DirectiveElementBase<DirectiveElement::Operands>;

/// This class represents the `results` directive. This directive represents
/// all of the results of an operation.
using ResultsDirective = DirectiveElementBase<DirectiveElement::Results>;

/// This class represents the `regions` directive. This directive represents
/// all of the regions of an operation.
using RegionsDirective = DirectiveElementBase<DirectiveElement::Regions>;

/// This class represents the `successors` directive. This directive represents
/// all of the successors of an operation.
using SuccessorsDirective = DirectiveElementBase<DirectiveElement::Successors>;

/// This class represents the `attr-dict` directive. This directive represents
/// the attribute dictionary of the operation.
class AttrDictDirective
    : public DirectiveElementBase<DirectiveElement::AttrDict> {
public:
  explicit AttrDictDirective(bool withKeyword) : withKeyword(withKeyword) {}

  /// Return whether the dictionary should be printed with the 'attributes'
  /// keyword.
  bool isWithKeyword() const { return withKeyword; }

private:
  /// If the dictionary should be printed with the 'attributes' keyword.
  bool withKeyword;
};

/// This class represents the `functional-type` directive. This directive takes
/// two arguments and formats them, respectively, as the inputs and results of a
/// FunctionType.
class FunctionalTypeDirective
    : public DirectiveElementBase<DirectiveElement::FunctionalType> {
public:
  FunctionalTypeDirective(FormatElement *inputs, FormatElement *results)
      : inputs(inputs), results(results) {}

  FormatElement *getInputs() const { return inputs; }
  FormatElement *getResults() const { return results; }

private:
  /// The input and result arguments.
  FormatElement *inputs, *results;
};

/// This class represents the `ref` directive.
class RefDirective : public DirectiveElementBase<DirectiveElement::Ref> {
public:
  RefDirective(FormatElement *arg) : arg(arg) {}

  FormatElement *getArg() const { return arg; }

private:
  /// The argument that is used to format the directive.
  FormatElement *arg;
};

/// This class represents the `type` directive.
class TypeDirective : public DirectiveElementBase<DirectiveElement::Type> {
public:
  TypeDirective(FormatElement *arg) : arg(arg) {}

  FormatElement *getArg() const { return arg; }

  /// Indicate if this type is printed "qualified" (that is it is
  /// prefixed with the `!dialect.mnemonic`).
  bool shouldBeQualified() { return shouldBeQualifiedFlag; }
  void setShouldBeQualified(bool qualified = true) {
    shouldBeQualifiedFlag = qualified;
  }

private:
  /// The argument that is used to format the directive.
  FormatElement *arg;

  bool shouldBeQualifiedFlag = false;
};

/// This class represents a group of order-independent optional clauses. Each
/// clause starts with a literal element and has a coressponding parsing
/// element. A parsing element is a continous sequence of format elements.
/// Each clause can appear 0 or 1 time.
class OIListElement : public DirectiveElementBase<DirectiveElement::OIList> {
public:
  OIListElement(std::vector<FormatElement *> &&literalElements,
                std::vector<std::vector<FormatElement *>> &&parsingElements)
      : literalElements(std::move(literalElements)),
        parsingElements(std::move(parsingElements)) {}

  /// Returns a range to iterate over the LiteralElements.
  auto getLiteralElements() const {
    // The use of std::function is unfortunate but necessary here. Lambda
    // functions cannot be copied but std::function can be copied. This copy
    // constructor is used in llvm::zip.
    std::function<LiteralElement *(FormatElement * el)>
        literalElementCastConverter =
            [](FormatElement *el) { return cast<LiteralElement>(el); };
    return llvm::map_range(literalElements, literalElementCastConverter);
  }

  /// Returns a range to iterate over the parsing elements corresponding to the
  /// clauses.
  ArrayRef<std::vector<FormatElement *>> getParsingElements() const {
    return parsingElements;
  }

  /// Returns a range to iterate over tuples of parsing and literal elements.
  auto getClauses() const {
    return llvm::zip(getLiteralElements(), getParsingElements());
  }

private:
  /// A vector of `LiteralElement` objects. Each element stores the keyword
  /// for one case of oilist element. For example, an oilist element along with
  /// the `literalElements` vector:
  /// ```
  ///  oilist [ `keyword` `=` `(` $arg0 `)` | `otherKeyword` `<` $arg1 `>`]
  ///  literalElements = { `keyword`, `otherKeyword` }
  /// ```
  std::vector<FormatElement *> literalElements;

  /// A vector of valid declarative assembly format vectors. Each object in
  /// parsing elements is a vector of elements in assembly format syntax.
  /// For example, an oilist element along with the parsingElements vector:
  /// ```
  ///  oilist [ `keyword` `=` `(` $arg0 `)` | `otherKeyword` `<` $arg1 `>`]
  ///  parsingElements = {
  ///    { `=`, `(`, $arg0, `)` },
  ///    { `<`, $arg1, `>` }
  ///  }
  /// ```
  std::vector<std::vector<FormatElement *>> parsingElements;
};
} // namespace

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

  /// The context in which an element is generated.
  enum class GenContext {
    /// The element is generated at the top-level or with the same behaviour.
    Normal,
    /// The element is generated inside an optional group.
    Optional
  };

  OperationFormat(const Operator &op)
      : allOperands(false), allOperandTypes(false), allResultTypes(false),
        infersResultTypes(false) {
    operandTypes.resize(op.getNumOperands(), TypeResolution());
    resultTypes.resize(op.getNumResults(), TypeResolution());

    hasImplicitTermTrait = llvm::any_of(op.getTraits(), [](const Trait &trait) {
      return trait.getDef().isSubClassOf("SingleBlockImplicitTerminator");
    });

    hasSingleBlockTrait =
        hasImplicitTermTrait || op.getTrait("::mlir::OpTrait::SingleBlock");
  }

  /// Generate the operation parser from this format.
  void genParser(Operator &op, OpClass &opClass);
  /// Generate the parser code for a specific format element.
  void genElementParser(FormatElement *element, MethodBody &body,
                        FmtContext &attrTypeCtx,
                        GenContext genCtx = GenContext::Normal);
  /// Generate the C++ to resolve the types of operands and results during
  /// parsing.
  void genParserTypeResolution(Operator &op, MethodBody &body);
  /// Generate the C++ to resolve the types of the operands during parsing.
  void genParserOperandTypeResolution(
      Operator &op, MethodBody &body,
      function_ref<void(TypeResolution &, StringRef)> emitTypeResolver);
  /// Generate the C++ to resolve regions during parsing.
  void genParserRegionResolution(Operator &op, MethodBody &body);
  /// Generate the C++ to resolve successors during parsing.
  void genParserSuccessorResolution(Operator &op, MethodBody &body);
  /// Generate the C++ to handling variadic segment size traits.
  void genParserVariadicSegmentResolution(Operator &op, MethodBody &body);

  /// Generate the operation printer from this format.
  void genPrinter(Operator &op, OpClass &opClass);

  /// Generate the printer code for a specific format element.
  void genElementPrinter(FormatElement *element, MethodBody &body, Operator &op,
                         bool &shouldEmitSpace, bool &lastWasPunctuation);

  /// The various elements in this format.
  std::vector<FormatElement *> elements;

  /// A flag indicating if all operand/result types were seen. If the format
  /// contains these, it can not contain individual type resolvers.
  bool allOperands, allOperandTypes, allResultTypes;

  /// A flag indicating if this operation infers its result types
  bool infersResultTypes;

  /// A flag indicating if this operation has the SingleBlockImplicitTerminator
  /// trait.
  bool hasImplicitTermTrait;

  /// A flag indicating if this operation has the SingleBlock trait.
  bool hasSingleBlockTrait;

  /// A map of buildable types to indices.
  llvm::MapVector<StringRef, int, llvm::StringMap<int>> buildableTypes;

  /// The index of the buildable type, if valid, for every operand and result.
  std::vector<TypeResolution> operandTypes, resultTypes;

  /// The set of attributes explicitly used within the format.
  SmallVector<const NamedAttribute *, 8> usedAttributes;
  llvm::StringSet<> inferredAttributes;
};
} // namespace

//===----------------------------------------------------------------------===//
// Parser Gen

/// Returns true if we can format the given attribute as an EnumAttr in the
/// parser format.
static bool canFormatEnumAttr(const NamedAttribute *attr) {
  Attribute baseAttr = attr->attr.getBaseAttr();
  const EnumAttr *enumAttr = dyn_cast<EnumAttr>(&baseAttr);
  if (!enumAttr)
    return false;

  // The attribute must have a valid underlying type and a constant builder.
  return !enumAttr->getUnderlyingType().empty() &&
         !enumAttr->getConstBuilderTemplate().empty();
}

/// Returns if we should format the given attribute as an SymbolNameAttr.
static bool shouldFormatSymbolNameAttr(const NamedAttribute *attr) {
  return attr->attr.getBaseAttr().getAttrDefName() == "SymbolNameAttr";
}

/// The code snippet used to generate a parser call for an attribute.
///
/// {0}: The name of the attribute.
/// {1}: The type for the attribute.
const char *const attrParserCode = R"(
  if (parser.parseCustomAttributeWithFallback({0}Attr, {1}, "{0}",
          result.attributes)) {{
    return ::mlir::failure();
  }
)";

/// The code snippet used to generate a parser call for an attribute.
///
/// {0}: The name of the attribute.
/// {1}: The type for the attribute.
const char *const genericAttrParserCode = R"(
  if (parser.parseAttribute({0}Attr, {1}, "{0}", result.attributes))
    return ::mlir::failure();
)";

const char *const optionalAttrParserCode = R"(
  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute({0}Attr, {1}, "{0}", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
)";

/// The code snippet used to generate a parser call for a symbol name attribute.
///
/// {0}: The name of the attribute.
const char *const symbolNameAttrParserCode = R"(
  if (parser.parseSymbolName({0}Attr, "{0}", result.attributes))
    return ::mlir::failure();
)";
const char *const optionalSymbolNameAttrParserCode = R"(
  // Parsing an optional symbol name doesn't fail, so no need to check the
  // result.
  (void)parser.parseOptionalSymbolName({0}Attr, "{0}", result.attributes);
)";

/// The code snippet used to generate a parser call for an enum attribute.
///
/// {0}: The name of the attribute.
/// {1}: The c++ namespace for the enum symbolize functions.
/// {2}: The function to symbolize a string of the enum.
/// {3}: The constant builder call to create an attribute of the enum type.
/// {4}: The set of allowed enum keywords.
/// {5}: The error message on failure when the enum isn't present.
const char *const enumAttrParserCode = R"(
  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {4})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "{0}", attrStorage);
      if (parseResult.hasValue()) {{
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        {5}
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = {1}::{2}(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "{0} attribute specification: \"" << attrStr << '"';;

      {0}Attr = {3};
      result.addAttribute("{0}", {0}Attr);
    }
  }
)";

/// The code snippet used to generate a parser call for an operand.
///
/// {0}: The name of the operand.
const char *const variadicOperandParserCode = R"(
  {0}OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList({0}Operands))
    return ::mlir::failure();
)";
const char *const optionalOperandParserCode = R"(
  {
    {0}OperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      {0}Operands.push_back(operand);
    }
  }
)";
const char *const operandParserCode = R"(
  {0}OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand({0}RawOperands[0]))
    return ::mlir::failure();
)";
/// The code snippet used to generate a parser call for a VariadicOfVariadic
/// operand.
///
/// {0}: The name of the operand.
/// {1}: The name of segment size attribute.
const char *const variadicOfVariadicOperandParserCode = R"(
  {
    {0}OperandsLoc = parser.getCurrentLocation();
    int32_t curSize = 0;
    do {
      if (parser.parseOptionalLParen())
        break;
      if (parser.parseOperandList({0}Operands) || parser.parseRParen())
        return ::mlir::failure();
      {0}OperandGroupSizes.push_back({0}Operands.size() - curSize);
      curSize = {0}Operands.size();
    } while (succeeded(parser.parseOptionalComma()));
  }
)";

/// The code snippet used to generate a parser call for a type list.
///
/// {0}: The name for the type list.
const char *const variadicOfVariadicTypeParserCode = R"(
  do {
    if (parser.parseOptionalLParen())
      break;
    if (parser.parseOptionalRParen() &&
        (parser.parseTypeList({0}Types) || parser.parseRParen()))
      return ::mlir::failure();
  } while (succeeded(parser.parseOptionalComma()));
)";
const char *const variadicTypeParserCode = R"(
  if (parser.parseTypeList({0}Types))
    return ::mlir::failure();
)";
const char *const optionalTypeParserCode = R"(
  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      {0}Types.push_back(optionalType);
    }
  }
)";
const char *const typeParserCode = R"(
  {
    {0} type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    {1}RawTypes[0] = type;
  }
)";
const char *const qualifiedTypeParserCode = R"(
  if (parser.parseType({1}RawTypes[0]))
    return ::mlir::failure();
)";

/// The code snippet used to generate a parser call for a functional type.
///
/// {0}: The name for the input type list.
/// {1}: The name for the result type list.
const char *const functionalTypeParserCode = R"(
  ::mlir::FunctionType {0}__{1}_functionType;
  if (parser.parseType({0}__{1}_functionType))
    return ::mlir::failure();
  {0}Types = {0}__{1}_functionType.getInputs();
  {1}Types = {0}__{1}_functionType.getResults();
)";

/// The code snippet used to generate a parser call to infer return types.
///
/// {0}: The operation class name
const char *const inferReturnTypesParserCode = R"(
  ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
  if (::mlir::failed({0}::inferReturnTypes(parser.getContext(),
      result.location, result.operands,
      result.attributes.getDictionary(parser.getContext()),
      result.regions, inferredReturnTypes)))
    return ::mlir::failure();
  result.addTypes(inferredReturnTypes);
)";

/// The code snippet used to generate a parser call for a region list.
///
/// {0}: The name for the region list.
const char *regionListParserCode = R"(
  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.hasValue()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      {0}Regions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        {0}Regions.emplace_back(std::move(region));
      }
    }
  }
)";

/// The code snippet used to ensure a list of regions have terminators.
///
/// {0}: The name of the region list.
const char *regionListEnsureTerminatorParserCode = R"(
  for (auto &region : {0}Regions)
    ensureTerminator(*region, parser.getBuilder(), result.location);
)";

/// The code snippet used to ensure a list of regions have a block.
///
/// {0}: The name of the region list.
const char *regionListEnsureSingleBlockParserCode = R"(
  for (auto &region : {0}Regions)
    if (region->empty()) region->emplaceBlock();
)";

/// The code snippet used to generate a parser call for an optional region.
///
/// {0}: The name of the region.
const char *optionalRegionParserCode = R"(
  {
     auto parseResult = parser.parseOptionalRegion(*{0}Region);
     if (parseResult.hasValue() && failed(*parseResult))
       return ::mlir::failure();
  }
)";

/// The code snippet used to generate a parser call for a region.
///
/// {0}: The name of the region.
const char *regionParserCode = R"(
  if (parser.parseRegion(*{0}Region))
    return ::mlir::failure();
)";

/// The code snippet used to ensure a region has a terminator.
///
/// {0}: The name of the region.
const char *regionEnsureTerminatorParserCode = R"(
  ensureTerminator(*{0}Region, parser.getBuilder(), result.location);
)";

/// The code snippet used to ensure a region has a block.
///
/// {0}: The name of the region.
const char *regionEnsureSingleBlockParserCode = R"(
  if ({0}Region->empty()) {0}Region->emplaceBlock();
)";

/// The code snippet used to generate a parser call for a successor list.
///
/// {0}: The name for the successor list.
const char *successorListParserCode = R"(
  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      {0}Successors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        {0}Successors.emplace_back(succ);
      }
    }
  }
)";

/// The code snippet used to generate a parser call for a successor.
///
/// {0}: The name of the successor.
const char *successorParserCode = R"(
  if (parser.parseSuccessor({0}Successor))
    return ::mlir::failure();
)";

/// The code snippet used to generate a parser for OIList
///
/// {0}: literal keyword corresponding to a case for oilist
const char *oilistParserCode = R"(
  if ({0}Clause) {
    return parser.emitError(parser.getNameLoc())
          << "`{0}` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  {0}Clause = true;
  result.addAttribute("{0}", UnitAttr::get(parser.getContext()));
)";

namespace {
/// The type of length for a given parse argument.
enum class ArgumentLengthKind {
  /// The argument is a variadic of a variadic, and may contain 0->N range
  /// elements.
  VariadicOfVariadic,
  /// The argument is variadic, and may contain 0->N elements.
  Variadic,
  /// The argument is optional, and may contain 0 or 1 elements.
  Optional,
  /// The argument is a single element, i.e. always represents 1 element.
  Single
};
} // namespace

/// Get the length kind for the given constraint.
static ArgumentLengthKind
getArgumentLengthKind(const NamedTypeConstraint *var) {
  if (var->isOptional())
    return ArgumentLengthKind::Optional;
  if (var->isVariadicOfVariadic())
    return ArgumentLengthKind::VariadicOfVariadic;
  if (var->isVariadic())
    return ArgumentLengthKind::Variadic;
  return ArgumentLengthKind::Single;
}

/// Get the name used for the type list for the given type directive operand.
/// 'lengthKind' to the corresponding kind for the given argument.
static StringRef getTypeListName(FormatElement *arg,
                                 ArgumentLengthKind &lengthKind) {
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
static void genLiteralParser(StringRef value, MethodBody &body) {
  // Handle the case of a keyword/identifier.
  if (value.front() == '_' || isalpha(value.front())) {
    body << "Keyword(\"" << value << "\")";
    return;
  }
  body << (StringRef)StringSwitch<StringRef>(value)
              .Case("->", "Arrow()")
              .Case(":", "Colon()")
              .Case(",", "Comma()")
              .Case("=", "Equal()")
              .Case("<", "Less()")
              .Case(">", "Greater()")
              .Case("{", "LBrace()")
              .Case("}", "RBrace()")
              .Case("(", "LParen()")
              .Case(")", "RParen()")
              .Case("[", "LSquare()")
              .Case("]", "RSquare()")
              .Case("?", "Question()")
              .Case("+", "Plus()")
              .Case("*", "Star()");
}

/// Generate the storage code required for parsing the given element.
static void genElementParserStorage(FormatElement *element, const Operator &op,
                                    MethodBody &body) {
  if (auto *optional = dyn_cast<OptionalElement>(element)) {
    ArrayRef<FormatElement *> elements = optional->getThenElements();

    // If the anchor is a unit attribute, it won't be parsed directly so elide
    // it.
    auto *anchor = dyn_cast<AttributeVariable>(optional->getAnchor());
    FormatElement *elidedAnchorElement = nullptr;
    if (anchor && anchor != elements.front() && anchor->isUnitAttr())
      elidedAnchorElement = anchor;
    for (FormatElement *childElement : elements)
      if (childElement != elidedAnchorElement)
        genElementParserStorage(childElement, op, body);
    for (FormatElement *childElement : optional->getElseElements())
      genElementParserStorage(childElement, op, body);

  } else if (auto *oilist = dyn_cast<OIListElement>(element)) {
    for (ArrayRef<FormatElement *> pelement : oilist->getParsingElements())
      for (FormatElement *element : pelement)
        genElementParserStorage(element, op, body);

  } else if (auto *custom = dyn_cast<CustomDirective>(element)) {
    for (FormatElement *paramElement : custom->getArguments())
      genElementParserStorage(paramElement, op, body);

  } else if (isa<OperandsDirective>(element)) {
    body << "  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> "
            "allOperands;\n";

  } else if (isa<RegionsDirective>(element)) {
    body << "  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> "
            "fullRegions;\n";

  } else if (isa<SuccessorsDirective>(element)) {
    body << "  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;\n";

  } else if (auto *attr = dyn_cast<AttributeVariable>(element)) {
    const NamedAttribute *var = attr->getVar();
    body << llvm::formatv("  {0} {1}Attr;\n", var->attr.getStorageType(),
                          var->name);

  } else if (auto *operand = dyn_cast<OperandVariable>(element)) {
    StringRef name = operand->getVar()->name;
    if (operand->getVar()->isVariableLength()) {
      body << "  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> "
           << name << "Operands;\n";
      if (operand->getVar()->isVariadicOfVariadic()) {
        body << "    llvm::SmallVector<int32_t> " << name
             << "OperandGroupSizes;\n";
      }
    } else {
      body << "  ::mlir::OpAsmParser::OperandType " << name
           << "RawOperands[1];\n"
           << "  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> " << name
           << "Operands(" << name << "RawOperands);";
    }
    body << llvm::formatv("  ::llvm::SMLoc {0}OperandsLoc;\n"
                          "  (void){0}OperandsLoc;\n",
                          name);

  } else if (auto *region = dyn_cast<RegionVariable>(element)) {
    StringRef name = region->getVar()->name;
    if (region->getVar()->isVariadic()) {
      body << llvm::formatv(
          "  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> "
          "{0}Regions;\n",
          name);
    } else {
      body << llvm::formatv("  std::unique_ptr<::mlir::Region> {0}Region = "
                            "std::make_unique<::mlir::Region>();\n",
                            name);
    }

  } else if (auto *successor = dyn_cast<SuccessorVariable>(element)) {
    StringRef name = successor->getVar()->name;
    if (successor->getVar()->isVariadic()) {
      body << llvm::formatv("  ::llvm::SmallVector<::mlir::Block *, 2> "
                            "{0}Successors;\n",
                            name);
    } else {
      body << llvm::formatv("  ::mlir::Block *{0}Successor = nullptr;\n", name);
    }

  } else if (auto *dir = dyn_cast<TypeDirective>(element)) {
    ArgumentLengthKind lengthKind;
    StringRef name = getTypeListName(dir->getArg(), lengthKind);
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

/// Generate the parser for a parameter to a custom directive.
static void genCustomParameterParser(FormatElement *param, MethodBody &body) {
  if (auto *attr = dyn_cast<AttributeVariable>(param)) {
    body << attr->getVar()->name << "Attr";
  } else if (isa<AttrDictDirective>(param)) {
    body << "result.attributes";
  } else if (auto *operand = dyn_cast<OperandVariable>(param)) {
    StringRef name = operand->getVar()->name;
    ArgumentLengthKind lengthKind = getArgumentLengthKind(operand->getVar());
    if (lengthKind == ArgumentLengthKind::VariadicOfVariadic)
      body << llvm::formatv("{0}OperandGroups", name);
    else if (lengthKind == ArgumentLengthKind::Variadic)
      body << llvm::formatv("{0}Operands", name);
    else if (lengthKind == ArgumentLengthKind::Optional)
      body << llvm::formatv("{0}Operand", name);
    else
      body << formatv("{0}RawOperands[0]", name);

  } else if (auto *region = dyn_cast<RegionVariable>(param)) {
    StringRef name = region->getVar()->name;
    if (region->getVar()->isVariadic())
      body << llvm::formatv("{0}Regions", name);
    else
      body << llvm::formatv("*{0}Region", name);

  } else if (auto *successor = dyn_cast<SuccessorVariable>(param)) {
    StringRef name = successor->getVar()->name;
    if (successor->getVar()->isVariadic())
      body << llvm::formatv("{0}Successors", name);
    else
      body << llvm::formatv("{0}Successor", name);

  } else if (auto *dir = dyn_cast<RefDirective>(param)) {
    genCustomParameterParser(dir->getArg(), body);

  } else if (auto *dir = dyn_cast<TypeDirective>(param)) {
    ArgumentLengthKind lengthKind;
    StringRef listName = getTypeListName(dir->getArg(), lengthKind);
    if (lengthKind == ArgumentLengthKind::VariadicOfVariadic)
      body << llvm::formatv("{0}TypeGroups", listName);
    else if (lengthKind == ArgumentLengthKind::Variadic)
      body << llvm::formatv("{0}Types", listName);
    else if (lengthKind == ArgumentLengthKind::Optional)
      body << llvm::formatv("{0}Type", listName);
    else
      body << formatv("{0}RawTypes[0]", listName);
  } else {
    llvm_unreachable("unknown custom directive parameter");
  }
}

/// Generate the parser for a custom directive.
static void genCustomDirectiveParser(CustomDirective *dir, MethodBody &body) {
  body << "  {\n";

  // Preprocess the directive variables.
  // * Add a local variable for optional operands and types. This provides a
  //   better API to the user defined parser methods.
  // * Set the location of operand variables.
  for (FormatElement *param : dir->getArguments()) {
    if (auto *operand = dyn_cast<OperandVariable>(param)) {
      auto *var = operand->getVar();
      body << "    " << var->name
           << "OperandsLoc = parser.getCurrentLocation();\n";
      if (var->isOptional()) {
        body << llvm::formatv(
            "    ::llvm::Optional<::mlir::OpAsmParser::OperandType> "
            "{0}Operand;\n",
            var->name);
      } else if (var->isVariadicOfVariadic()) {
        body << llvm::formatv("    "
                              "::llvm::SmallVector<::llvm::SmallVector<::mlir::"
                              "OpAsmParser::OperandType>> "
                              "{0}OperandGroups;\n",
                              var->name);
      }
    } else if (auto *dir = dyn_cast<TypeDirective>(param)) {
      ArgumentLengthKind lengthKind;
      StringRef listName = getTypeListName(dir->getArg(), lengthKind);
      if (lengthKind == ArgumentLengthKind::Optional) {
        body << llvm::formatv("    ::mlir::Type {0}Type;\n", listName);
      } else if (lengthKind == ArgumentLengthKind::VariadicOfVariadic) {
        body << llvm::formatv(
            "    ::llvm::SmallVector<llvm::SmallVector<::mlir::Type>> "
            "{0}TypeGroups;\n",
            listName);
      }
    } else if (auto *dir = dyn_cast<RefDirective>(param)) {
      FormatElement *input = dir->getArg();
      if (auto *operand = dyn_cast<OperandVariable>(input)) {
        if (!operand->getVar()->isOptional())
          continue;
        body << llvm::formatv(
            "    {0} {1}Operand = {1}Operands.empty() ? {0}() : "
            "{1}Operands[0];\n",
            "::llvm::Optional<::mlir::OpAsmParser::OperandType>",
            operand->getVar()->name);

      } else if (auto *type = dyn_cast<TypeDirective>(input)) {
        ArgumentLengthKind lengthKind;
        StringRef listName = getTypeListName(type->getArg(), lengthKind);
        if (lengthKind == ArgumentLengthKind::Optional) {
          body << llvm::formatv("    ::mlir::Type {0}Type = {0}Types.empty() ? "
                                "::mlir::Type() : {0}Types[0];\n",
                                listName);
        }
      }
    }
  }

  body << "    if (parse" << dir->getName() << "(parser";
  for (FormatElement *param : dir->getArguments()) {
    body << ", ";
    genCustomParameterParser(param, body);
  }

  body << "))\n"
       << "      return ::mlir::failure();\n";

  // After parsing, add handling for any of the optional constructs.
  for (FormatElement *param : dir->getArguments()) {
    if (auto *attr = dyn_cast<AttributeVariable>(param)) {
      const NamedAttribute *var = attr->getVar();
      if (var->attr.isOptional())
        body << llvm::formatv("    if ({0}Attr)\n  ", var->name);

      body << llvm::formatv("    result.addAttribute(\"{0}\", {0}Attr);\n",
                            var->name);
    } else if (auto *operand = dyn_cast<OperandVariable>(param)) {
      const NamedTypeConstraint *var = operand->getVar();
      if (var->isOptional()) {
        body << llvm::formatv("    if ({0}Operand.hasValue())\n"
                              "      {0}Operands.push_back(*{0}Operand);\n",
                              var->name);
      } else if (var->isVariadicOfVariadic()) {
        body << llvm::formatv(
            "    for (const auto &subRange : {0}OperandGroups) {{\n"
            "      {0}Operands.append(subRange.begin(), subRange.end());\n"
            "      {0}OperandGroupSizes.push_back(subRange.size());\n"
            "    }\n",
            var->name, var->constraint.getVariadicOfVariadicSegmentSizeAttr());
      }
    } else if (auto *dir = dyn_cast<TypeDirective>(param)) {
      ArgumentLengthKind lengthKind;
      StringRef listName = getTypeListName(dir->getArg(), lengthKind);
      if (lengthKind == ArgumentLengthKind::Optional) {
        body << llvm::formatv("    if ({0}Type)\n"
                              "      {0}Types.push_back({0}Type);\n",
                              listName);
      } else if (lengthKind == ArgumentLengthKind::VariadicOfVariadic) {
        body << llvm::formatv(
            "    for (const auto &subRange : {0}TypeGroups)\n"
            "      {0}Types.append(subRange.begin(), subRange.end());\n",
            listName);
      }
    }
  }

  body << "  }\n";
}

/// Generate the parser for a enum attribute.
static void genEnumAttrParser(const NamedAttribute *var, MethodBody &body,
                              FmtContext &attrTypeCtx) {
  Attribute baseAttr = var->attr.getBaseAttr();
  const EnumAttr &enumAttr = cast<EnumAttr>(baseAttr);
  std::vector<EnumAttrCase> cases = enumAttr.getAllCases();

  // Generate the code for building an attribute for this enum.
  std::string attrBuilderStr;
  {
    llvm::raw_string_ostream os(attrBuilderStr);
    os << tgfmt(enumAttr.getConstBuilderTemplate(), &attrTypeCtx,
                "attrOptional.getValue()");
  }

  // Build a string containing the cases that can be formatted as a keyword.
  std::string validCaseKeywordsStr = "{";
  llvm::raw_string_ostream validCaseKeywordsOS(validCaseKeywordsStr);
  for (const EnumAttrCase &attrCase : cases)
    if (canFormatStringAsKeyword(attrCase.getStr()))
      validCaseKeywordsOS << '"' << attrCase.getStr() << "\",";
  validCaseKeywordsOS.str().back() = '}';

  // If the attribute is not optional, build an error message for the missing
  // attribute.
  std::string errorMessage;
  if (!var->attr.isOptional()) {
    llvm::raw_string_ostream errorMessageOS(errorMessage);
    errorMessageOS
        << "return parser.emitError(loc, \"expected string or "
           "keyword containing one of the following enum values for attribute '"
        << var->name << "' [";
    llvm::interleaveComma(cases, errorMessageOS, [&](const auto &attrCase) {
      errorMessageOS << attrCase.getStr();
    });
    errorMessageOS << "]\");";
  }

  body << formatv(enumAttrParserCode, var->name, enumAttr.getCppNamespace(),
                  enumAttr.getStringToSymbolFnName(), attrBuilderStr,
                  validCaseKeywordsStr, errorMessage);
}

void OperationFormat::genParser(Operator &op, OpClass &opClass) {
  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("::mlir::OpAsmParser &", "parser");
  paramList.emplace_back("::mlir::OperationState &", "result");

  auto *method = opClass.addStaticMethod("::mlir::ParseResult", "parse",
                                         std::move(paramList));
  auto &body = method->body();

  // Generate variables to store the operands and type within the format. This
  // allows for referencing these variables in the presence of optional
  // groupings.
  for (FormatElement *element : elements)
    genElementParserStorage(element, op, body);

  // A format context used when parsing attributes with buildable types.
  FmtContext attrTypeCtx;
  attrTypeCtx.withBuilder("parser.getBuilder()");

  // Generate parsers for each of the elements.
  for (FormatElement *element : elements)
    genElementParser(element, body, attrTypeCtx);

  // Generate the code to resolve the operand/result types and successors now
  // that they have been parsed.
  genParserRegionResolution(op, body);
  genParserSuccessorResolution(op, body);
  genParserVariadicSegmentResolution(op, body);
  genParserTypeResolution(op, body);

  body << "  return ::mlir::success();\n";
}

void OperationFormat::genElementParser(FormatElement *element, MethodBody &body,
                                       FmtContext &attrTypeCtx,
                                       GenContext genCtx) {
  /// Optional Group.
  if (auto *optional = dyn_cast<OptionalElement>(element)) {
    ArrayRef<FormatElement *> elements =
        optional->getThenElements().drop_front(optional->getParseStart());

    // Generate a special optional parser for the first element to gate the
    // parsing of the rest of the elements.
    FormatElement *firstElement = elements.front();
    if (auto *attrVar = dyn_cast<AttributeVariable>(firstElement)) {
      genElementParser(attrVar, body, attrTypeCtx);
      body << "  if (" << attrVar->getVar()->name << "Attr) {\n";
    } else if (auto *literal = dyn_cast<LiteralElement>(firstElement)) {
      body << "  if (succeeded(parser.parseOptional";
      genLiteralParser(literal->getSpelling(), body);
      body << ")) {\n";
    } else if (auto *opVar = dyn_cast<OperandVariable>(firstElement)) {
      genElementParser(opVar, body, attrTypeCtx);
      body << "  if (!" << opVar->getVar()->name << "Operands.empty()) {\n";
    } else if (auto *regionVar = dyn_cast<RegionVariable>(firstElement)) {
      const NamedRegion *region = regionVar->getVar();
      if (region->isVariadic()) {
        genElementParser(regionVar, body, attrTypeCtx);
        body << "  if (!" << region->name << "Regions.empty()) {\n";
      } else {
        body << llvm::formatv(optionalRegionParserCode, region->name);
        body << "  if (!" << region->name << "Region->empty()) {\n  ";
        if (hasImplicitTermTrait)
          body << llvm::formatv(regionEnsureTerminatorParserCode, region->name);
        else if (hasSingleBlockTrait)
          body << llvm::formatv(regionEnsureSingleBlockParserCode,
                                region->name);
      }
    }

    // If the anchor is a unit attribute, we don't need to print it. When
    // parsing, we will add this attribute if this group is present.
    FormatElement *elidedAnchorElement = nullptr;
    auto *anchorAttr = dyn_cast<AttributeVariable>(optional->getAnchor());
    if (anchorAttr && anchorAttr != firstElement && anchorAttr->isUnitAttr()) {
      elidedAnchorElement = anchorAttr;

      // Add the anchor unit attribute to the operation state.
      body << "    result.addAttribute(\"" << anchorAttr->getVar()->name
           << "\", parser.getBuilder().getUnitAttr());\n";
    }

    // Generate the rest of the elements inside an optional group. Elements in
    // an optional group after the guard are parsed as required.
    for (FormatElement *childElement : llvm::drop_begin(elements, 1))
      if (childElement != elidedAnchorElement)
        genElementParser(childElement, body, attrTypeCtx, GenContext::Optional);
    body << "  }";

    // Generate the else elements.
    auto elseElements = optional->getElseElements();
    if (!elseElements.empty()) {
      body << " else {\n";
      for (FormatElement *childElement : elseElements)
        genElementParser(childElement, body, attrTypeCtx);
      body << "  }";
    }
    body << "\n";

    /// OIList Directive
  } else if (OIListElement *oilist = dyn_cast<OIListElement>(element)) {
    for (LiteralElement *le : oilist->getLiteralElements())
      body << "  bool " << le->getSpelling() << "Clause = false;\n";

    // Generate the parsing loop
    body << "  while(true) {\n";
    for (auto clause : oilist->getClauses()) {
      LiteralElement *lelement = std::get<0>(clause);
      ArrayRef<FormatElement *> pelement = std::get<1>(clause);
      body << "if (succeeded(parser.parseOptional";
      genLiteralParser(lelement->getSpelling(), body);
      body << ")) {\n";
      StringRef attrName = lelement->getSpelling();
      body << formatv(oilistParserCode, attrName);
      inferredAttributes.insert(attrName);
      for (FormatElement *el : pelement)
        genElementParser(el, body, attrTypeCtx);
      body << "    } else ";
    }
    body << " {\n";
    body << "    break;\n";
    body << "  }\n";
    body << "}\n";

    /// Literals.
  } else if (LiteralElement *literal = dyn_cast<LiteralElement>(element)) {
    body << "  if (parser.parse";
    genLiteralParser(literal->getSpelling(), body);
    body << ")\n    return ::mlir::failure();\n";

    /// Whitespaces.
  } else if (isa<WhitespaceElement>(element)) {
    // Nothing to parse.

    /// Arguments.
  } else if (auto *attr = dyn_cast<AttributeVariable>(element)) {
    const NamedAttribute *var = attr->getVar();

    // Check to see if we can parse this as an enum attribute.
    if (canFormatEnumAttr(var))
      return genEnumAttrParser(var, body, attrTypeCtx);

    // Check to see if we should parse this as a symbol name attribute.
    if (shouldFormatSymbolNameAttr(var)) {
      body << formatv(var->attr.isOptional() ? optionalSymbolNameAttrParserCode
                                             : symbolNameAttrParserCode,
                      var->name);
      return;
    }

    // If this attribute has a buildable type, use that when parsing the
    // attribute.
    std::string attrTypeStr;
    if (Optional<StringRef> typeBuilder = attr->getTypeBuilder()) {
      llvm::raw_string_ostream os(attrTypeStr);
      os << tgfmt(*typeBuilder, &attrTypeCtx);
    } else {
      attrTypeStr = "::mlir::Type{}";
    }
    if (genCtx == GenContext::Normal && var->attr.isOptional()) {
      body << formatv(optionalAttrParserCode, var->name, attrTypeStr);
    } else {
      if (attr->shouldBeQualified() ||
          var->attr.getStorageType() == "::mlir::Attribute")
        body << formatv(genericAttrParserCode, var->name, attrTypeStr);
      else
        body << formatv(attrParserCode, var->name, attrTypeStr);
    }

  } else if (auto *operand = dyn_cast<OperandVariable>(element)) {
    ArgumentLengthKind lengthKind = getArgumentLengthKind(operand->getVar());
    StringRef name = operand->getVar()->name;
    if (lengthKind == ArgumentLengthKind::VariadicOfVariadic)
      body << llvm::formatv(
          variadicOfVariadicOperandParserCode, name,
          operand->getVar()->constraint.getVariadicOfVariadicSegmentSizeAttr());
    else if (lengthKind == ArgumentLengthKind::Variadic)
      body << llvm::formatv(variadicOperandParserCode, name);
    else if (lengthKind == ArgumentLengthKind::Optional)
      body << llvm::formatv(optionalOperandParserCode, name);
    else
      body << formatv(operandParserCode, name);

  } else if (auto *region = dyn_cast<RegionVariable>(element)) {
    bool isVariadic = region->getVar()->isVariadic();
    body << llvm::formatv(isVariadic ? regionListParserCode : regionParserCode,
                          region->getVar()->name);
    if (hasImplicitTermTrait)
      body << llvm::formatv(isVariadic ? regionListEnsureTerminatorParserCode
                                       : regionEnsureTerminatorParserCode,
                            region->getVar()->name);
    else if (hasSingleBlockTrait)
      body << llvm::formatv(isVariadic ? regionListEnsureSingleBlockParserCode
                                       : regionEnsureSingleBlockParserCode,
                            region->getVar()->name);

  } else if (auto *successor = dyn_cast<SuccessorVariable>(element)) {
    bool isVariadic = successor->getVar()->isVariadic();
    body << formatv(isVariadic ? successorListParserCode : successorParserCode,
                    successor->getVar()->name);

    /// Directives.
  } else if (auto *attrDict = dyn_cast<AttrDictDirective>(element)) {
    body << "  if (parser.parseOptionalAttrDict"
         << (attrDict->isWithKeyword() ? "WithKeyword" : "")
         << "(result.attributes))\n"
         << "    return ::mlir::failure();\n";
  } else if (auto *customDir = dyn_cast<CustomDirective>(element)) {
    genCustomDirectiveParser(customDir, body);

  } else if (isa<OperandsDirective>(element)) {
    body << "  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();\n"
         << "  if (parser.parseOperandList(allOperands))\n"
         << "    return ::mlir::failure();\n";

  } else if (isa<RegionsDirective>(element)) {
    body << llvm::formatv(regionListParserCode, "full");
    if (hasImplicitTermTrait)
      body << llvm::formatv(regionListEnsureTerminatorParserCode, "full");
    else if (hasSingleBlockTrait)
      body << llvm::formatv(regionListEnsureSingleBlockParserCode, "full");

  } else if (isa<SuccessorsDirective>(element)) {
    body << llvm::formatv(successorListParserCode, "full");

  } else if (auto *dir = dyn_cast<TypeDirective>(element)) {
    ArgumentLengthKind lengthKind;
    StringRef listName = getTypeListName(dir->getArg(), lengthKind);
    if (lengthKind == ArgumentLengthKind::VariadicOfVariadic) {
      body << llvm::formatv(variadicOfVariadicTypeParserCode, listName);
    } else if (lengthKind == ArgumentLengthKind::Variadic) {
      body << llvm::formatv(variadicTypeParserCode, listName);
    } else if (lengthKind == ArgumentLengthKind::Optional) {
      body << llvm::formatv(optionalTypeParserCode, listName);
    } else {
      const char *parserCode =
          dir->shouldBeQualified() ? qualifiedTypeParserCode : typeParserCode;
      TypeSwitch<FormatElement *>(dir->getArg())
          .Case<OperandVariable, ResultVariable>([&](auto operand) {
            body << formatv(parserCode,
                            operand->getVar()->constraint.getCPPClassName(),
                            listName);
          })
          .Default([&](auto operand) {
            body << formatv(parserCode, "::mlir::Type", listName);
          });
    }
  } else if (auto *dir = dyn_cast<FunctionalTypeDirective>(element)) {
    ArgumentLengthKind ignored;
    body << formatv(functionalTypeParserCode,
                    getTypeListName(dir->getInputs(), ignored),
                    getTypeListName(dir->getResults(), ignored));
  } else {
    llvm_unreachable("unknown format element");
  }
}

void OperationFormat::genParserTypeResolution(Operator &op, MethodBody &body) {
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
                    variable->name, constraint.getSummary())
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
      if (Optional<StringRef> tform = resolver.getVarTransformer()) {
        FmtContext fmtContext;
        fmtContext.addSubst("_ctxt", "parser.getContext()");
        if (var->isVariadic())
          fmtContext.withSelf(var->name + "Types");
        else
          fmtContext.withSelf(var->name + "Types[0]");
        body << tgfmt(*tform, &fmtContext);
      } else {
        body << var->name << "Types";
      }
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
  if (!infersResultTypes) {
    if (allResultTypes) {
      body << "  result.addTypes(allResultTypes);\n";
    } else {
      for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
        body << "  result.addTypes(";
        emitTypeResolver(resultTypes[i], op.getResultName(i));
        body << ");\n";
      }
    }
  }

  // Emit the operand type resolutions.
  genParserOperandTypeResolution(op, body, emitTypeResolver);

  // Handle return type inference once all operands have been resolved
  if (infersResultTypes)
    body << formatv(inferReturnTypesParserCode, op.getCppClassName());
}

void OperationFormat::genParserOperandTypeResolution(
    Operator &op, MethodBody &body,
    function_ref<void(TypeResolution &, StringRef)> emitTypeResolver) {
  // Early exit if there are no operands.
  if (op.getNumOperands() == 0)
    return;

  // Handle the case where all operand types are grouped together with
  // "types(operands)".
  if (allOperandTypes) {
    // If `operands` was specified, use the full operand list directly.
    if (allOperands) {
      body << "  if (parser.resolveOperands(allOperands, allOperandTypes, "
              "allOperandLoc, result.operands))\n"
              "    return ::mlir::failure();\n";
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
         << "    return ::mlir::failure();\n";
    return;
  }

  // Handle the case where all operands are grouped together with "operands".
  if (allOperands) {
    body << "  if (parser.resolveOperands(allOperands, ";

    // Group all of the operand types together to perform the resolution all at
    // once. Use llvm::concat to perform the merge. llvm::concat does not allow
    // the case of a single range, so guard it here.
    if (op.getNumOperands() > 1) {
      body << "::llvm::concat<const ::mlir::Type>(";
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
         << "    return ::mlir::failure();\n";
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
    body << ", result.operands))\n    return ::mlir::failure();\n";
  }
}

void OperationFormat::genParserRegionResolution(Operator &op,
                                                MethodBody &body) {
  // Check for the case where all regions were parsed.
  bool hasAllRegions = llvm::any_of(
      elements, [](FormatElement *elt) { return isa<RegionsDirective>(elt); });
  if (hasAllRegions) {
    body << "  result.addRegions(fullRegions);\n";
    return;
  }

  // Otherwise, handle each region individually.
  for (const NamedRegion &region : op.getRegions()) {
    if (region.isVariadic())
      body << "  result.addRegions(" << region.name << "Regions);\n";
    else
      body << "  result.addRegion(std::move(" << region.name << "Region));\n";
  }
}

void OperationFormat::genParserSuccessorResolution(Operator &op,
                                                   MethodBody &body) {
  // Check for the case where all successors were parsed.
  bool hasAllSuccessors = llvm::any_of(elements, [](FormatElement *elt) {
    return isa<SuccessorsDirective>(elt);
  });
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
                                                         MethodBody &body) {
  if (!allOperands) {
    if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
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
    for (const NamedTypeConstraint &operand : op.getOperands()) {
      if (!operand.isVariadicOfVariadic())
        continue;
      body << llvm::formatv(
          "  result.addAttribute(\"{0}\", "
          "parser.getBuilder().getI32TensorAttr({1}OperandGroupSizes));\n",
          operand.constraint.getVariadicOfVariadicSegmentSizeAttr(),
          operand.name);
    }
  }

  if (!allResultTypes &&
      op.getTrait("::mlir::OpTrait::AttrSizedResultSegments")) {
    body << "  result.addAttribute(\"result_segment_sizes\", "
         << "parser.getBuilder().getI32VectorAttr({";
    auto interleaveFn = [&](const NamedTypeConstraint &result) {
      // If the result is variadic emit the parsed size.
      if (result.isVariableLength())
        body << "static_cast<int32_t>(" << result.name << "Types.size())";
      else
        body << "1";
    };
    llvm::interleaveComma(op.getResults(), body, interleaveFn);
    body << "}));\n";
  }
}

//===----------------------------------------------------------------------===//
// PrinterGen

/// The code snippet used to generate a printer call for a region of an
// operation that has the SingleBlockImplicitTerminator trait.
///
/// {0}: The name of the region.
const char *regionSingleBlockImplicitTerminatorPrinterCode = R"(
  {
    bool printTerminator = true;
    if (auto *term = {0}.empty() ? nullptr : {0}.begin()->getTerminator()) {{
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion({0}, /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
)";

/// The code snippet used to generate a printer call for an enum that has cases
/// that can't be represented with a keyword.
///
/// {0}: The name of the enum attribute.
/// {1}: The name of the enum attributes symbolToString function.
const char *enumAttrBeginPrinterCode = R"(
  {
    auto caseValue = {0}();
    auto caseValueStr = {1}(caseValue);
)";

/// Generate the printer for the 'attr-dict' directive.
static void genAttrDictPrinter(OperationFormat &fmt, Operator &op,
                               MethodBody &body, bool withKeyword) {
  body << "  _odsPrinter.printOptionalAttrDict"
       << (withKeyword ? "WithKeyword" : "")
       << "((*this)->getAttrs(), /*elidedAttrs=*/{";
  // Elide the variadic segment size attributes if necessary.
  if (!fmt.allOperands &&
      op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments"))
    body << "\"operand_segment_sizes\", ";
  if (!fmt.allResultTypes &&
      op.getTrait("::mlir::OpTrait::AttrSizedResultSegments"))
    body << "\"result_segment_sizes\", ";
  if (!fmt.inferredAttributes.empty()) {
    for (const auto &attr : fmt.inferredAttributes)
      body << "\"" << attr.getKey() << "\", ";
  }
  llvm::interleaveComma(
      fmt.usedAttributes, body,
      [&](const NamedAttribute *attr) { body << "\"" << attr->name << "\""; });
  body << "});\n";
}

/// Generate the printer for a literal value. `shouldEmitSpace` is true if a
/// space should be emitted before this element. `lastWasPunctuation` is true if
/// the previous element was a punctuation literal.
static void genLiteralPrinter(StringRef value, MethodBody &body,
                              bool &shouldEmitSpace, bool &lastWasPunctuation) {
  body << "  _odsPrinter";

  // Don't insert a space for certain punctuation.
  if (shouldEmitSpace && shouldEmitSpaceBefore(value, lastWasPunctuation))
    body << " << ' '";
  body << " << \"" << value << "\";\n";

  // Insert a space after certain literals.
  shouldEmitSpace =
      value.size() != 1 || !StringRef("<({[").contains(value.front());
  lastWasPunctuation = !(value.front() == '_' || isalpha(value.front()));
}

/// Generate the printer for a space. `shouldEmitSpace` and `lastWasPunctuation`
/// are set to false.
static void genSpacePrinter(bool value, MethodBody &body, bool &shouldEmitSpace,
                            bool &lastWasPunctuation) {
  if (value) {
    body << "  _odsPrinter << ' ';\n";
    lastWasPunctuation = false;
  } else {
    lastWasPunctuation = true;
  }
  shouldEmitSpace = false;
}

/// Generate the printer for a custom directive parameter.
static void genCustomDirectiveParameterPrinter(FormatElement *element,
                                               const Operator &op,
                                               MethodBody &body) {
  if (auto *attr = dyn_cast<AttributeVariable>(element)) {
    body << op.getGetterName(attr->getVar()->name) << "Attr()";

  } else if (isa<AttrDictDirective>(element)) {
    body << "getOperation()->getAttrDictionary()";

  } else if (auto *operand = dyn_cast<OperandVariable>(element)) {
    body << op.getGetterName(operand->getVar()->name) << "()";

  } else if (auto *region = dyn_cast<RegionVariable>(element)) {
    body << op.getGetterName(region->getVar()->name) << "()";

  } else if (auto *successor = dyn_cast<SuccessorVariable>(element)) {
    body << op.getGetterName(successor->getVar()->name) << "()";

  } else if (auto *dir = dyn_cast<RefDirective>(element)) {
    genCustomDirectiveParameterPrinter(dir->getArg(), op, body);

  } else if (auto *dir = dyn_cast<TypeDirective>(element)) {
    auto *typeOperand = dir->getArg();
    auto *operand = dyn_cast<OperandVariable>(typeOperand);
    auto *var = operand ? operand->getVar()
                        : cast<ResultVariable>(typeOperand)->getVar();
    std::string name = op.getGetterName(var->name);
    if (var->isVariadic())
      body << name << "().getTypes()";
    else if (var->isOptional())
      body << llvm::formatv("({0}() ? {0}().getType() : Type())", name);
    else
      body << name << "().getType()";
  } else {
    llvm_unreachable("unknown custom directive parameter");
  }
}

/// Generate the printer for a custom directive.
static void genCustomDirectivePrinter(CustomDirective *customDir,
                                      const Operator &op, MethodBody &body) {
  body << "  print" << customDir->getName() << "(_odsPrinter, *this";
  for (FormatElement *param : customDir->getArguments()) {
    body << ", ";
    genCustomDirectiveParameterPrinter(param, op, body);
  }
  body << ");\n";
}

/// Generate the printer for a region with the given variable name.
static void genRegionPrinter(const Twine &regionName, MethodBody &body,
                             bool hasImplicitTermTrait) {
  if (hasImplicitTermTrait)
    body << llvm::formatv(regionSingleBlockImplicitTerminatorPrinterCode,
                          regionName);
  else
    body << "  _odsPrinter.printRegion(" << regionName << ");\n";
}
static void genVariadicRegionPrinter(const Twine &regionListName,
                                     MethodBody &body,
                                     bool hasImplicitTermTrait) {
  body << "    llvm::interleaveComma(" << regionListName
       << ", _odsPrinter, [&](::mlir::Region &region) {\n      ";
  genRegionPrinter("region", body, hasImplicitTermTrait);
  body << "    });\n";
}

/// Generate the C++ for an operand to a (*-)type directive.
static MethodBody &genTypeOperandPrinter(FormatElement *arg, const Operator &op,
                                         MethodBody &body,
                                         bool useArrayRef = true) {
  if (isa<OperandsDirective>(arg))
    return body << "getOperation()->getOperandTypes()";
  if (isa<ResultsDirective>(arg))
    return body << "getOperation()->getResultTypes()";
  auto *operand = dyn_cast<OperandVariable>(arg);
  auto *var = operand ? operand->getVar() : cast<ResultVariable>(arg)->getVar();
  if (var->isVariadicOfVariadic())
    return body << llvm::formatv("{0}().join().getTypes()",
                                 op.getGetterName(var->name));
  if (var->isVariadic())
    return body << op.getGetterName(var->name) << "().getTypes()";
  if (var->isOptional())
    return body << llvm::formatv(
               "({0}() ? ::llvm::ArrayRef<::mlir::Type>({0}().getType()) : "
               "::llvm::ArrayRef<::mlir::Type>())",
               op.getGetterName(var->name));
  if (useArrayRef)
    return body << "::llvm::ArrayRef<::mlir::Type>("
                << op.getGetterName(var->name) << "().getType())";
  return body << op.getGetterName(var->name) << "().getType()";
}

/// Generate the printer for an enum attribute.
static void genEnumAttrPrinter(const NamedAttribute *var, const Operator &op,
                               MethodBody &body) {
  Attribute baseAttr = var->attr.getBaseAttr();
  const EnumAttr &enumAttr = cast<EnumAttr>(baseAttr);
  std::vector<EnumAttrCase> cases = enumAttr.getAllCases();

  body << llvm::formatv(enumAttrBeginPrinterCode,
                        (var->attr.isOptional() ? "*" : "") +
                            op.getGetterName(var->name),
                        enumAttr.getSymbolToStringFnName());

  // Get a string containing all of the cases that can't be represented with a
  // keyword.
  BitVector nonKeywordCases(cases.size());
  bool hasStrCase = false;
  for (auto &it : llvm::enumerate(cases)) {
    hasStrCase = it.value().isStrCase();
    if (!canFormatStringAsKeyword(it.value().getStr()))
      nonKeywordCases.set(it.index());
  }

  // If this is a string enum, use the case string to determine which cases
  // need to use the string form.
  if (hasStrCase) {
    if (nonKeywordCases.any()) {
      body << "    if (llvm::is_contained(llvm::ArrayRef<llvm::StringRef>(";
      llvm::interleaveComma(nonKeywordCases.set_bits(), body, [&](unsigned it) {
        body << '"' << cases[it].getStr() << '"';
      });
      body << ")))\n"
              "      _odsPrinter << '\"' << caseValueStr << '\"';\n"
              "    else\n  ";
    }
    body << "    _odsPrinter << caseValueStr;\n"
            "  }\n";
    return;
  }

  // Otherwise if this is a bit enum attribute, don't allow cases that may
  // overlap with other cases. For simplicity sake, only allow cases with a
  // single bit value.
  if (enumAttr.isBitEnum()) {
    for (auto &it : llvm::enumerate(cases)) {
      int64_t value = it.value().getValue();
      if (value < 0 || !llvm::isPowerOf2_64(value))
        nonKeywordCases.set(it.index());
    }
  }

  // If there are any cases that can't be used with a keyword, switch on the
  // case value to determine when to print in the string form.
  if (nonKeywordCases.any()) {
    body << "    switch (caseValue) {\n";
    StringRef cppNamespace = enumAttr.getCppNamespace();
    StringRef enumName = enumAttr.getEnumClassName();
    for (auto &it : llvm::enumerate(cases)) {
      if (nonKeywordCases.test(it.index()))
        continue;
      StringRef symbol = it.value().getSymbol();
      body << llvm::formatv("    case {0}::{1}::{2}:\n", cppNamespace, enumName,
                            llvm::isDigit(symbol.front()) ? ("_" + symbol)
                                                          : symbol);
    }
    body << "      _odsPrinter << caseValueStr;\n"
            "      break;\n"
            "    default:\n"
            "      _odsPrinter << '\"' << caseValueStr << '\"';\n"
            "      break;\n"
            "    }\n"
            "  }\n";
    return;
  }

  body << "    _odsPrinter << caseValueStr;\n"
          "  }\n";
}

/// Generate the check for the anchor of an optional group.
static void genOptionalGroupPrinterAnchor(FormatElement *anchor,
                                          const Operator &op,
                                          MethodBody &body) {
  TypeSwitch<FormatElement *>(anchor)
      .Case<OperandVariable, ResultVariable>([&](auto *element) {
        const NamedTypeConstraint *var = element->getVar();
        std::string name = op.getGetterName(var->name);
        if (var->isOptional())
          body << "  if (" << name << "()) {\n";
        else if (var->isVariadic())
          body << "  if (!" << name << "().empty()) {\n";
      })
      .Case<RegionVariable>([&](RegionVariable *element) {
        const NamedRegion *var = element->getVar();
        std::string name = op.getGetterName(var->name);
        // TODO: Add a check for optional regions here when ODS supports it.
        body << "  if (!" << name << "().empty()) {\n";
      })
      .Case<TypeDirective>([&](TypeDirective *element) {
        genOptionalGroupPrinterAnchor(element->getArg(), op, body);
      })
      .Case<FunctionalTypeDirective>([&](FunctionalTypeDirective *element) {
        genOptionalGroupPrinterAnchor(element->getInputs(), op, body);
      })
      .Case<AttributeVariable>([&](AttributeVariable *attr) {
        body << "  if ((*this)->getAttr(\"" << attr->getVar()->name
             << "\")) {\n";
      });
}

void OperationFormat::genElementPrinter(FormatElement *element,
                                        MethodBody &body, Operator &op,
                                        bool &shouldEmitSpace,
                                        bool &lastWasPunctuation) {
  if (LiteralElement *literal = dyn_cast<LiteralElement>(element))
    return genLiteralPrinter(literal->getSpelling(), body, shouldEmitSpace,
                             lastWasPunctuation);

  // Emit a whitespace element.
  if (auto *space = dyn_cast<WhitespaceElement>(element)) {
    if (space->getValue() == "\\n") {
      body << "  _odsPrinter.printNewline();\n";
    } else {
      genSpacePrinter(!space->getValue().empty(), body, shouldEmitSpace,
                      lastWasPunctuation);
    }
    return;
  }

  // Emit an optional group.
  if (OptionalElement *optional = dyn_cast<OptionalElement>(element)) {
    // Emit the check for the presence of the anchor element.
    FormatElement *anchor = optional->getAnchor();
    genOptionalGroupPrinterAnchor(anchor, op, body);

    // If the anchor is a unit attribute, we don't need to print it. When
    // parsing, we will add this attribute if this group is present.
    auto elements = optional->getThenElements();
    FormatElement *elidedAnchorElement = nullptr;
    auto *anchorAttr = dyn_cast<AttributeVariable>(anchor);
    if (anchorAttr && anchorAttr != elements.front() &&
        anchorAttr->isUnitAttr()) {
      elidedAnchorElement = anchorAttr;
    }

    // Emit each of the elements.
    for (FormatElement *childElement : elements) {
      if (childElement != elidedAnchorElement) {
        genElementPrinter(childElement, body, op, shouldEmitSpace,
                          lastWasPunctuation);
      }
    }
    body << "  }";

    // Emit each of the else elements.
    auto elseElements = optional->getElseElements();
    if (!elseElements.empty()) {
      body << " else {\n";
      for (FormatElement *childElement : elseElements) {
        genElementPrinter(childElement, body, op, shouldEmitSpace,
                          lastWasPunctuation);
      }
      body << "  }";
    }

    body << "\n";
    return;
  }

  // Emit the OIList
  if (auto *oilist = dyn_cast<OIListElement>(element)) {
    genLiteralPrinter(" ", body, shouldEmitSpace, lastWasPunctuation);
    for (auto clause : oilist->getClauses()) {
      LiteralElement *lelement = std::get<0>(clause);
      ArrayRef<FormatElement *> pelement = std::get<1>(clause);

      body << "  if ((*this)->hasAttrOfType<UnitAttr>(\""
           << lelement->getSpelling() << "\")) {\n";
      genLiteralPrinter(lelement->getSpelling(), body, shouldEmitSpace,
                        lastWasPunctuation);
      for (FormatElement *element : pelement) {
        genElementPrinter(element, body, op, shouldEmitSpace,
                          lastWasPunctuation);
      }
      body << "  }\n";
    }
    return;
  }

  // Emit the attribute dictionary.
  if (auto *attrDict = dyn_cast<AttrDictDirective>(element)) {
    genAttrDictPrinter(*this, op, body, attrDict->isWithKeyword());
    lastWasPunctuation = false;
    return;
  }

  // Optionally insert a space before the next element. The AttrDict printer
  // already adds a space as necessary.
  if (shouldEmitSpace || !lastWasPunctuation)
    body << "  _odsPrinter << ' ';\n";
  lastWasPunctuation = false;
  shouldEmitSpace = true;

  if (auto *attr = dyn_cast<AttributeVariable>(element)) {
    const NamedAttribute *var = attr->getVar();

    // If we are formatting as an enum, symbolize the attribute as a string.
    if (canFormatEnumAttr(var))
      return genEnumAttrPrinter(var, op, body);

    // If we are formatting as a symbol name, handle it as a symbol name.
    if (shouldFormatSymbolNameAttr(var)) {
      body << "  _odsPrinter.printSymbolName(" << op.getGetterName(var->name)
           << "Attr().getValue());\n";
      return;
    }

    // Elide the attribute type if it is buildable.
    if (attr->getTypeBuilder())
      body << "  _odsPrinter.printAttributeWithoutType("
           << op.getGetterName(var->name) << "Attr());\n";
    else if (attr->shouldBeQualified() ||
             var->attr.getStorageType() == "::mlir::Attribute")
      body << "  _odsPrinter.printAttribute(" << op.getGetterName(var->name)
           << "Attr());\n";
    else
      body << "_odsPrinter.printStrippedAttrOrType("
           << op.getGetterName(var->name) << "Attr());\n";
  } else if (auto *operand = dyn_cast<OperandVariable>(element)) {
    if (operand->getVar()->isVariadicOfVariadic()) {
      body << "  ::llvm::interleaveComma("
           << op.getGetterName(operand->getVar()->name)
           << "(), _odsPrinter, [&](const auto &operands) { _odsPrinter << "
              "\"(\" << operands << "
              "\")\"; });\n";

    } else if (operand->getVar()->isOptional()) {
      body << "  if (::mlir::Value value = "
           << op.getGetterName(operand->getVar()->name) << "())\n"
           << "    _odsPrinter << value;\n";
    } else {
      body << "  _odsPrinter << " << op.getGetterName(operand->getVar()->name)
           << "();\n";
    }
  } else if (auto *region = dyn_cast<RegionVariable>(element)) {
    const NamedRegion *var = region->getVar();
    std::string name = op.getGetterName(var->name);
    if (var->isVariadic()) {
      genVariadicRegionPrinter(name + "()", body, hasImplicitTermTrait);
    } else {
      genRegionPrinter(name + "()", body, hasImplicitTermTrait);
    }
  } else if (auto *successor = dyn_cast<SuccessorVariable>(element)) {
    const NamedSuccessor *var = successor->getVar();
    std::string name = op.getGetterName(var->name);
    if (var->isVariadic())
      body << "  ::llvm::interleaveComma(" << name << "(), _odsPrinter);\n";
    else
      body << "  _odsPrinter << " << name << "();\n";
  } else if (auto *dir = dyn_cast<CustomDirective>(element)) {
    genCustomDirectivePrinter(dir, op, body);
  } else if (isa<OperandsDirective>(element)) {
    body << "  _odsPrinter << getOperation()->getOperands();\n";
  } else if (isa<RegionsDirective>(element)) {
    genVariadicRegionPrinter("getOperation()->getRegions()", body,
                             hasImplicitTermTrait);
  } else if (isa<SuccessorsDirective>(element)) {
    body << "  ::llvm::interleaveComma(getOperation()->getSuccessors(), "
            "_odsPrinter);\n";
  } else if (auto *dir = dyn_cast<TypeDirective>(element)) {
    if (auto *operand = dyn_cast<OperandVariable>(dir->getArg())) {
      if (operand->getVar()->isVariadicOfVariadic()) {
        body << llvm::formatv(
            "  ::llvm::interleaveComma({0}().getTypes(), _odsPrinter, "
            "[&](::mlir::TypeRange types) {{ _odsPrinter << \"(\" << "
            "types << \")\"; });\n",
            op.getGetterName(operand->getVar()->name));
        return;
      }
    }
    const NamedTypeConstraint *var = nullptr;
    {
      if (auto *operand = dyn_cast<OperandVariable>(dir->getArg()))
        var = operand->getVar();
      else if (auto *operand = dyn_cast<ResultVariable>(dir->getArg()))
        var = operand->getVar();
    }
    if (var && !var->isVariadicOfVariadic() && !var->isVariadic() &&
        !var->isOptional()) {
      std::string cppClass = var->constraint.getCPPClassName();
      if (dir->shouldBeQualified()) {
        body << "   _odsPrinter << " << op.getGetterName(var->name)
             << "().getType();\n";
        return;
      }
      body << "  {\n"
           << "    auto type = " << op.getGetterName(var->name)
           << "().getType();\n"
           << "    if (auto validType = type.dyn_cast<" << cppClass << ">())\n"
           << "      _odsPrinter.printStrippedAttrOrType(validType);\n"
           << "   else\n"
           << "     _odsPrinter << type;\n"
           << "  }\n";
      return;
    }
    body << "  _odsPrinter << ";
    genTypeOperandPrinter(dir->getArg(), op, body, /*useArrayRef=*/false)
        << ";\n";
  } else if (auto *dir = dyn_cast<FunctionalTypeDirective>(element)) {
    body << "  _odsPrinter.printFunctionalType(";
    genTypeOperandPrinter(dir->getInputs(), op, body) << ", ";
    genTypeOperandPrinter(dir->getResults(), op, body) << ");\n";
  } else {
    llvm_unreachable("unknown format element");
  }
}

void OperationFormat::genPrinter(Operator &op, OpClass &opClass) {
  auto *method = opClass.addMethod(
      "void", "print",
      MethodParameter("::mlir::OpAsmPrinter &", "_odsPrinter"));
  auto &body = method->body();

  // Flags for if we should emit a space, and if the last element was
  // punctuation.
  bool shouldEmitSpace = true, lastWasPunctuation = false;
  for (FormatElement *element : elements)
    genElementPrinter(element, body, op, shouldEmitSpace, lastWasPunctuation);
}

//===----------------------------------------------------------------------===//
// OpFormatParser
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
class OpFormatParser : public FormatParser {
public:
  OpFormatParser(llvm::SourceMgr &mgr, OperationFormat &format, Operator &op)
      : FormatParser(mgr, op.getLoc()[0]), fmt(format), op(op),
        seenOperandTypes(op.getNumOperands()),
        seenResultTypes(op.getNumResults()) {}

protected:
  /// Verify the format elements.
  LogicalResult verify(SMLoc loc, ArrayRef<FormatElement *> elements) override;
  /// Verify the arguments to a custom directive.
  LogicalResult
  verifyCustomDirectiveArguments(SMLoc loc,
                                 ArrayRef<FormatElement *> arguments) override;
  /// Verify the elements of an optional group.
  LogicalResult
  verifyOptionalGroupElements(SMLoc loc, ArrayRef<FormatElement *> elements,
                              Optional<unsigned> anchorIndex) override;
  LogicalResult verifyOptionalGroupElement(SMLoc loc, FormatElement *element,
                                           bool isAnchor);

  /// Parse an operation variable.
  FailureOr<FormatElement *> parseVariableImpl(SMLoc loc, StringRef name,
                                               Context ctx) override;
  /// Parse an operation format directive.
  FailureOr<FormatElement *>
  parseDirectiveImpl(SMLoc loc, FormatToken::Kind kind, Context ctx) override;

private:
  /// This struct represents a type resolution instance. It includes a specific
  /// type as well as an optional transformer to apply to that type in order to
  /// properly resolve the type of a variable.
  struct TypeResolutionInstance {
    ConstArgument resolver;
    Optional<StringRef> transformer;
  };

  using ElementsItT = ArrayRef<FormatElement *>::iterator;

  /// Verify the state of operation attributes within the format.
  LogicalResult verifyAttributes(SMLoc loc, ArrayRef<FormatElement *> elements);
  /// Verify the attribute elements at the back of the given stack of iterators.
  LogicalResult verifyAttributes(
      SMLoc loc,
      SmallVectorImpl<std::pair<ElementsItT, ElementsItT>> &iteratorStack);

  /// Verify the state of operation operands within the format.
  LogicalResult
  verifyOperands(SMLoc loc,
                 llvm::StringMap<TypeResolutionInstance> &variableTyResolver);

  /// Verify the state of operation regions within the format.
  LogicalResult verifyRegions(SMLoc loc);

  /// Verify the state of operation results within the format.
  LogicalResult
  verifyResults(SMLoc loc,
                llvm::StringMap<TypeResolutionInstance> &variableTyResolver);

  /// Verify the state of operation successors within the format.
  LogicalResult verifySuccessors(SMLoc loc);

  LogicalResult verifyOIListElements(SMLoc loc,
                                     ArrayRef<FormatElement *> elements);

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
      const llvm::Record &def);

  /// Returns an argument or attribute with the given name that has been seen
  /// within the format.
  ConstArgument findSeenArg(StringRef name);

  /// Parse the various different directives.
  FailureOr<FormatElement *> parseAttrDictDirective(SMLoc loc, Context context,
                                                    bool withKeyword);
  FailureOr<FormatElement *> parseFunctionalTypeDirective(SMLoc loc,
                                                          Context context);
  FailureOr<FormatElement *> parseOIListDirective(SMLoc loc, Context context);
  LogicalResult verifyOIListParsingElement(FormatElement *element, SMLoc loc);
  FailureOr<FormatElement *> parseOperandsDirective(SMLoc loc, Context context);
  FailureOr<FormatElement *> parseQualifiedDirective(SMLoc loc,
                                                     Context context);
  FailureOr<FormatElement *> parseReferenceDirective(SMLoc loc,
                                                     Context context);
  FailureOr<FormatElement *> parseRegionsDirective(SMLoc loc, Context context);
  FailureOr<FormatElement *> parseResultsDirective(SMLoc loc, Context context);
  FailureOr<FormatElement *> parseSuccessorsDirective(SMLoc loc,
                                                      Context context);
  FailureOr<FormatElement *> parseTypeDirective(SMLoc loc, Context context);
  FailureOr<FormatElement *> parseTypeDirectiveOperand(SMLoc loc,
                                                       bool isRefChild = false);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  OperationFormat &fmt;
  Operator &op;

  // The following are various bits of format state used for verification
  // during parsing.
  bool hasAttrDict = false;
  bool hasAllRegions = false, hasAllSuccessors = false;
  bool canInferResultTypes = false;
  llvm::SmallBitVector seenOperandTypes, seenResultTypes;
  llvm::SmallSetVector<const NamedAttribute *, 8> seenAttrs;
  llvm::DenseSet<const NamedTypeConstraint *> seenOperands;
  llvm::DenseSet<const NamedRegion *> seenRegions;
  llvm::DenseSet<const NamedSuccessor *> seenSuccessors;
};
} // namespace

LogicalResult OpFormatParser::verify(SMLoc loc,
                                     ArrayRef<FormatElement *> elements) {
  // Check that the attribute dictionary is in the format.
  if (!hasAttrDict)
    return emitError(loc, "'attr-dict' directive not found in "
                          "custom assembly format");

  // Check for any type traits that we can use for inferring types.
  llvm::StringMap<TypeResolutionInstance> variableTyResolver;
  for (const Trait &trait : op.getTraits()) {
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
    } else if (!op.allResultTypesKnown()) {
      // This doesn't check the name directly to handle
      //    DeclareOpInterfaceMethods<InferTypeOpInterface>
      // and the like.
      // TODO: Add hasCppInterface check.
      if (auto name = def.getValueAsOptionalString("cppClassName")) {
        if (*name == "InferTypeOpInterface" &&
            def.getValueAsString("cppNamespace") == "::mlir")
          canInferResultTypes = true;
      }
    }
  }

  // Verify the state of the various operation components.
  if (failed(verifyAttributes(loc, elements)) ||
      failed(verifyResults(loc, variableTyResolver)) ||
      failed(verifyOperands(loc, variableTyResolver)) ||
      failed(verifyRegions(loc)) || failed(verifySuccessors(loc)) ||
      failed(verifyOIListElements(loc, elements)))
    return failure();

  // Collect the set of used attributes in the format.
  fmt.usedAttributes = seenAttrs.takeVector();
  return success();
}

LogicalResult
OpFormatParser::verifyAttributes(SMLoc loc,
                                 ArrayRef<FormatElement *> elements) {
  // Check that there are no `:` literals after an attribute without a constant
  // type. The attribute grammar contains an optional trailing colon type, which
  // can lead to unexpected and generally unintended behavior. Given that, it is
  // better to just error out here instead.
  SmallVector<std::pair<ElementsItT, ElementsItT>, 1> iteratorStack;
  iteratorStack.emplace_back(elements.begin(), elements.end());
  while (!iteratorStack.empty())
    if (failed(verifyAttributes(loc, iteratorStack)))
      return ::failure();

  // Check for VariadicOfVariadic variables. The segment attribute of those
  // variables will be infered.
  for (const NamedTypeConstraint *var : seenOperands) {
    if (var->constraint.isVariadicOfVariadic()) {
      fmt.inferredAttributes.insert(
          var->constraint.getVariadicOfVariadicSegmentSizeAttr());
    }
  }

  return success();
}
/// Verify the attribute elements at the back of the given stack of iterators.
LogicalResult OpFormatParser::verifyAttributes(
    SMLoc loc,
    SmallVectorImpl<std::pair<ElementsItT, ElementsItT>> &iteratorStack) {
  auto &stackIt = iteratorStack.back();
  ElementsItT &it = stackIt.first, e = stackIt.second;
  while (it != e) {
    FormatElement *element = *(it++);

    // Traverse into optional groups.
    if (auto *optional = dyn_cast<OptionalElement>(element)) {
      auto thenElements = optional->getThenElements();
      iteratorStack.emplace_back(thenElements.begin(), thenElements.end());

      auto elseElements = optional->getElseElements();
      iteratorStack.emplace_back(elseElements.begin(), elseElements.end());
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
      ElementsItT nextIt = nextItPair.first, nextE = nextItPair.second;
      for (; nextIt != nextE; ++nextIt) {
        // Skip any trailing whitespace, attribute dictionaries, or optional
        // groups.
        if (isa<WhitespaceElement>(*nextIt) ||
            isa<AttrDictDirective>(*nextIt) || isa<OptionalElement>(*nextIt))
          continue;

        // We are only interested in `:` literals.
        auto *literal = dyn_cast<LiteralElement>(*nextIt);
        if (!literal || literal->getSpelling() != ":")
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

LogicalResult OpFormatParser::verifyOperands(
    SMLoc loc, llvm::StringMap<TypeResolutionInstance> &variableTyResolver) {
  // Check that all of the operands are within the format, and their types can
  // be inferred.
  auto &buildableTypes = fmt.buildableTypes;
  for (unsigned i = 0, e = op.getNumOperands(); i != e; ++i) {
    NamedTypeConstraint &operand = op.getOperand(i);

    // Check that the operand itself is in the format.
    if (!fmt.allOperands && !seenOperands.count(&operand)) {
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
    if (!builder || (fmt.allOperands && operand.isVariableLength())) {
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

LogicalResult OpFormatParser::verifyRegions(SMLoc loc) {
  // Check that all of the regions are within the format.
  if (hasAllRegions)
    return success();

  for (unsigned i = 0, e = op.getNumRegions(); i != e; ++i) {
    const NamedRegion &region = op.getRegion(i);
    if (!seenRegions.count(&region)) {
      return emitErrorAndNote(loc,
                              "region #" + Twine(i) + ", named '" +
                                  region.name + "', not found",
                              "suggest adding a '$" + region.name +
                                  "' directive to the custom assembly format");
    }
  }
  return success();
}

LogicalResult OpFormatParser::verifyResults(
    SMLoc loc, llvm::StringMap<TypeResolutionInstance> &variableTyResolver) {
  // If we format all of the types together, there is nothing to check.
  if (fmt.allResultTypes)
    return success();

  // If no result types are specified and we can infer them, infer all result
  // types
  if (op.getNumResults() > 0 && seenResultTypes.count() == 0 &&
      canInferResultTypes) {
    fmt.infersResultTypes = true;
    return success();
  }

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

LogicalResult OpFormatParser::verifySuccessors(SMLoc loc) {
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

LogicalResult
OpFormatParser::verifyOIListElements(SMLoc loc,
                                     ArrayRef<FormatElement *> elements) {
  // Check that all of the successors are within the format.
  SmallVector<StringRef> prohibitedLiterals;
  for (FormatElement *it : elements) {
    if (auto *oilist = dyn_cast<OIListElement>(it)) {
      if (!prohibitedLiterals.empty()) {
        // We just saw an oilist element in last iteration. Literals should not
        // match.
        for (LiteralElement *literal : oilist->getLiteralElements()) {
          if (find(prohibitedLiterals, literal->getSpelling()) !=
              prohibitedLiterals.end()) {
            return emitError(
                loc, "format ambiguity because " + literal->getSpelling() +
                         " is used in two adjacent oilist elements.");
          }
        }
      }
      for (LiteralElement *literal : oilist->getLiteralElements())
        prohibitedLiterals.push_back(literal->getSpelling());
    } else if (auto *literal = dyn_cast<LiteralElement>(it)) {
      if (find(prohibitedLiterals, literal->getSpelling()) !=
          prohibitedLiterals.end()) {
        return emitError(
            loc,
            "format ambiguity because " + literal->getSpelling() +
                " is used both in oilist element and the adjacent literal.");
      }
      prohibitedLiterals.clear();
    } else {
      prohibitedLiterals.clear();
    }
  }
  return success();
}

void OpFormatParser::handleAllTypesMatchConstraint(
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

void OpFormatParser::handleSameTypesConstraint(
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

void OpFormatParser::handleTypesMatchConstraint(
    llvm::StringMap<TypeResolutionInstance> &variableTyResolver,
    const llvm::Record &def) {
  StringRef lhsName = def.getValueAsString("lhs");
  StringRef rhsName = def.getValueAsString("rhs");
  StringRef transformer = def.getValueAsString("transformer");
  if (ConstArgument arg = findSeenArg(lhsName))
    variableTyResolver[rhsName] = {arg, transformer};
}

ConstArgument OpFormatParser::findSeenArg(StringRef name) {
  if (const NamedTypeConstraint *arg = findArg(op.getOperands(), name))
    return seenOperandTypes.test(arg - op.operand_begin()) ? arg : nullptr;
  if (const NamedTypeConstraint *arg = findArg(op.getResults(), name))
    return seenResultTypes.test(arg - op.result_begin()) ? arg : nullptr;
  if (const NamedAttribute *attr = findArg(op.getAttributes(), name))
    return seenAttrs.count(attr) ? attr : nullptr;
  return nullptr;
}

FailureOr<FormatElement *>
OpFormatParser::parseVariableImpl(SMLoc loc, StringRef name, Context ctx) {
  // Check that the parsed argument is something actually registered on the op.
  // Attributes
  if (const NamedAttribute *attr = findArg(op.getAttributes(), name)) {
    if (ctx == TypeDirectiveContext)
      return emitError(
          loc, "attributes cannot be used as children to a `type` directive");
    if (ctx == RefDirectiveContext) {
      if (!seenAttrs.count(attr))
        return emitError(loc, "attribute '" + name +
                                  "' must be bound before it is referenced");
    } else if (!seenAttrs.insert(attr)) {
      return emitError(loc, "attribute '" + name + "' is already bound");
    }

    return create<AttributeVariable>(attr);
  }
  // Operands
  if (const NamedTypeConstraint *operand = findArg(op.getOperands(), name)) {
    if (ctx == TopLevelContext || ctx == CustomDirectiveContext) {
      if (fmt.allOperands || !seenOperands.insert(operand).second)
        return emitError(loc, "operand '" + name + "' is already bound");
    } else if (ctx == RefDirectiveContext && !seenOperands.count(operand)) {
      return emitError(loc, "operand '" + name +
                                "' must be bound before it is referenced");
    }
    return create<OperandVariable>(operand);
  }
  // Regions
  if (const NamedRegion *region = findArg(op.getRegions(), name)) {
    if (ctx == TopLevelContext || ctx == CustomDirectiveContext) {
      if (hasAllRegions || !seenRegions.insert(region).second)
        return emitError(loc, "region '" + name + "' is already bound");
    } else if (ctx == RefDirectiveContext && !seenRegions.count(region)) {
      return emitError(loc, "region '" + name +
                                "' must be bound before it is referenced");
    } else {
      return emitError(loc, "regions can only be used at the top level");
    }
    return create<RegionVariable>(region);
  }
  // Results.
  if (const auto *result = findArg(op.getResults(), name)) {
    if (ctx != TypeDirectiveContext)
      return emitError(loc, "result variables can can only be used as a child "
                            "to a 'type' directive");
    return create<ResultVariable>(result);
  }
  // Successors.
  if (const auto *successor = findArg(op.getSuccessors(), name)) {
    if (ctx == TopLevelContext || ctx == CustomDirectiveContext) {
      if (hasAllSuccessors || !seenSuccessors.insert(successor).second)
        return emitError(loc, "successor '" + name + "' is already bound");
    } else if (ctx == RefDirectiveContext && !seenSuccessors.count(successor)) {
      return emitError(loc, "successor '" + name +
                                "' must be bound before it is referenced");
    } else {
      return emitError(loc, "successors can only be used at the top level");
    }

    return create<SuccessorVariable>(successor);
  }
  return emitError(loc, "expected variable to refer to an argument, region, "
                        "result, or successor");
}

FailureOr<FormatElement *>
OpFormatParser::parseDirectiveImpl(SMLoc loc, FormatToken::Kind kind,
                                   Context ctx) {
  switch (kind) {
  case FormatToken::kw_attr_dict:
    return parseAttrDictDirective(loc, ctx,
                                  /*withKeyword=*/false);
  case FormatToken::kw_attr_dict_w_keyword:
    return parseAttrDictDirective(loc, ctx,
                                  /*withKeyword=*/true);
  case FormatToken::kw_functional_type:
    return parseFunctionalTypeDirective(loc, ctx);
  case FormatToken::kw_operands:
    return parseOperandsDirective(loc, ctx);
  case FormatToken::kw_qualified:
    return parseQualifiedDirective(loc, ctx);
  case FormatToken::kw_regions:
    return parseRegionsDirective(loc, ctx);
  case FormatToken::kw_results:
    return parseResultsDirective(loc, ctx);
  case FormatToken::kw_successors:
    return parseSuccessorsDirective(loc, ctx);
  case FormatToken::kw_ref:
    return parseReferenceDirective(loc, ctx);
  case FormatToken::kw_type:
    return parseTypeDirective(loc, ctx);
  case FormatToken::kw_oilist:
    return parseOIListDirective(loc, ctx);

  default:
    return emitError(loc, "unsupported directive kind");
  }
}

FailureOr<FormatElement *>
OpFormatParser::parseAttrDictDirective(SMLoc loc, Context context,
                                       bool withKeyword) {
  if (context == TypeDirectiveContext)
    return emitError(loc, "'attr-dict' directive can only be used as a "
                          "top-level directive");

  if (context == RefDirectiveContext) {
    if (!hasAttrDict)
      return emitError(loc, "'ref' of 'attr-dict' is not bound by a prior "
                            "'attr-dict' directive");

    // Otherwise, this is a top-level context.
  } else {
    if (hasAttrDict)
      return emitError(loc, "'attr-dict' directive has already been seen");
    hasAttrDict = true;
  }

  return create<AttrDictDirective>(withKeyword);
}

LogicalResult OpFormatParser::verifyCustomDirectiveArguments(
    SMLoc loc, ArrayRef<FormatElement *> arguments) {
  for (FormatElement *argument : arguments) {
    if (!isa<RefDirective, TypeDirective, AttrDictDirective, AttributeVariable,
             OperandVariable, RegionVariable, SuccessorVariable>(argument)) {
      // TODO: FormatElement should have location info attached.
      return emitError(loc, "only variables and types may be used as "
                            "parameters to a custom directive");
    }
    if (auto *type = dyn_cast<TypeDirective>(argument)) {
      if (!isa<OperandVariable, ResultVariable>(type->getArg())) {
        return emitError(loc, "type directives within a custom directive may "
                              "only refer to variables");
      }
    }
  }
  return success();
}

FailureOr<FormatElement *>
OpFormatParser::parseFunctionalTypeDirective(SMLoc loc, Context context) {
  if (context != TopLevelContext)
    return emitError(
        loc, "'functional-type' is only valid as a top-level directive");

  // Parse the main operand.
  FailureOr<FormatElement *> inputs, results;
  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before argument list")) ||
      failed(inputs = parseTypeDirectiveOperand(loc)) ||
      failed(parseToken(FormatToken::comma,
                        "expected ',' after inputs argument")) ||
      failed(results = parseTypeDirectiveOperand(loc)) ||
      failed(
          parseToken(FormatToken::r_paren, "expected ')' after argument list")))
    return failure();
  return create<FunctionalTypeDirective>(*inputs, *results);
}

FailureOr<FormatElement *>
OpFormatParser::parseOperandsDirective(SMLoc loc, Context context) {
  if (context == RefDirectiveContext) {
    if (!fmt.allOperands)
      return emitError(loc, "'ref' of 'operands' is not bound by a prior "
                            "'operands' directive");

  } else if (context == TopLevelContext || context == CustomDirectiveContext) {
    if (fmt.allOperands || !seenOperands.empty())
      return emitError(loc, "'operands' directive creates overlap in format");
    fmt.allOperands = true;
  }
  return create<OperandsDirective>();
}

FailureOr<FormatElement *>
OpFormatParser::parseReferenceDirective(SMLoc loc, Context context) {
  if (context != CustomDirectiveContext)
    return emitError(loc, "'ref' is only valid within a `custom` directive");

  FailureOr<FormatElement *> arg;
  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before argument list")) ||
      failed(arg = parseElement(RefDirectiveContext)) ||
      failed(
          parseToken(FormatToken::r_paren, "expected ')' after argument list")))
    return failure();

  return create<RefDirective>(*arg);
}

FailureOr<FormatElement *>
OpFormatParser::parseRegionsDirective(SMLoc loc, Context context) {
  if (context == TypeDirectiveContext)
    return emitError(loc, "'regions' is only valid as a top-level directive");
  if (context == RefDirectiveContext) {
    if (!hasAllRegions)
      return emitError(loc, "'ref' of 'regions' is not bound by a prior "
                            "'regions' directive");

    // Otherwise, this is a TopLevel directive.
  } else {
    if (hasAllRegions || !seenRegions.empty())
      return emitError(loc, "'regions' directive creates overlap in format");
    hasAllRegions = true;
  }
  return create<RegionsDirective>();
}

FailureOr<FormatElement *>
OpFormatParser::parseResultsDirective(SMLoc loc, Context context) {
  if (context != TypeDirectiveContext)
    return emitError(loc, "'results' directive can can only be used as a child "
                          "to a 'type' directive");
  return create<ResultsDirective>();
}

FailureOr<FormatElement *>
OpFormatParser::parseSuccessorsDirective(SMLoc loc, Context context) {
  if (context == TypeDirectiveContext)
    return emitError(loc,
                     "'successors' is only valid as a top-level directive");
  if (context == RefDirectiveContext) {
    if (!hasAllSuccessors)
      return emitError(loc, "'ref' of 'successors' is not bound by a prior "
                            "'successors' directive");

    // Otherwise, this is a TopLevel directive.
  } else {
    if (hasAllSuccessors || !seenSuccessors.empty())
      return emitError(loc, "'successors' directive creates overlap in format");
    hasAllSuccessors = true;
  }
  return create<SuccessorsDirective>();
}

FailureOr<FormatElement *>
OpFormatParser::parseOIListDirective(SMLoc loc, Context context) {
  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before oilist argument list")))
    return failure();
  std::vector<FormatElement *> literalElements;
  std::vector<std::vector<FormatElement *>> parsingElements;
  do {
    FailureOr<FormatElement *> lelement = parseLiteral(context);
    if (failed(lelement))
      return failure();
    literalElements.push_back(*lelement);
    parsingElements.push_back(std::vector<FormatElement *>());
    std::vector<FormatElement *> &currParsingElements = parsingElements.back();
    while (peekToken().getKind() != FormatToken::pipe &&
           peekToken().getKind() != FormatToken::r_paren) {
      FailureOr<FormatElement *> pelement = parseElement(context);
      if (failed(pelement) ||
          failed(verifyOIListParsingElement(*pelement, loc)))
        return failure();
      currParsingElements.push_back(*pelement);
    }
    if (peekToken().getKind() == FormatToken::pipe) {
      consumeToken();
      continue;
    }
    if (peekToken().getKind() == FormatToken::r_paren) {
      consumeToken();
      break;
    }
  } while (true);

  return create<OIListElement>(std::move(literalElements),
                               std::move(parsingElements));
}

LogicalResult OpFormatParser::verifyOIListParsingElement(FormatElement *element,
                                                         SMLoc loc) {
  return TypeSwitch<FormatElement *, LogicalResult>(element)
      // Only optional attributes can be within an oilist parsing group.
      .Case([&](AttributeVariable *attrEle) {
        if (!attrEle->getVar()->attr.isOptional())
          return emitError(loc, "only optional attributes can be used to "
                                "in an oilist parsing group");
        return success();
      })
      // Only optional-like(i.e. variadic) operands can be within an oilist
      // parsing group.
      .Case([&](OperandVariable *ele) {
        if (!ele->getVar()->isVariableLength())
          return emitError(loc, "only variable length operands can be "
                                "used within an oilist parsing group");
        return success();
      })
      // Only optional-like(i.e. variadic) results can be within an oilist
      // parsing group.
      .Case([&](ResultVariable *ele) {
        if (!ele->getVar()->isVariableLength())
          return emitError(loc, "only variable length results can be "
                                "used within an oilist parsing group");
        return success();
      })
      .Case([&](RegionVariable *) {
        // TODO: When ODS has proper support for marking "optional" regions, add
        // a check here.
        return success();
      })
      .Case([&](TypeDirective *ele) {
        return verifyOIListParsingElement(ele->getArg(), loc);
      })
      .Case([&](FunctionalTypeDirective *ele) {
        if (failed(verifyOIListParsingElement(ele->getInputs(), loc)))
          return failure();
        return verifyOIListParsingElement(ele->getResults(), loc);
      })
      // Literals, whitespace, and custom directives may be used.
      .Case<LiteralElement, WhitespaceElement, CustomDirective,
            FunctionalTypeDirective, OptionalElement>(
          [&](FormatElement *) { return success(); })
      .Default([&](FormatElement *) {
        return emitError(loc, "only literals, types, and variables can be "
                              "used within an oilist group");
      });
}

FailureOr<FormatElement *> OpFormatParser::parseTypeDirective(SMLoc loc,
                                                              Context context) {
  if (context == TypeDirectiveContext)
    return emitError(loc, "'type' cannot be used as a child of another `type`");

  bool isRefChild = context == RefDirectiveContext;
  FailureOr<FormatElement *> operand;
  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before argument list")) ||
      failed(operand = parseTypeDirectiveOperand(loc, isRefChild)) ||
      failed(
          parseToken(FormatToken::r_paren, "expected ')' after argument list")))
    return failure();

  return create<TypeDirective>(*operand);
}

FailureOr<FormatElement *>
OpFormatParser::parseQualifiedDirective(SMLoc loc, Context context) {
  FailureOr<FormatElement *> element;
  if (failed(parseToken(FormatToken::l_paren,
                        "expected '(' before argument list")) ||
      failed(element = parseElement(context)) ||
      failed(
          parseToken(FormatToken::r_paren, "expected ')' after argument list")))
    return failure();
  return TypeSwitch<FormatElement *, FailureOr<FormatElement *>>(*element)
      .Case<AttributeVariable, TypeDirective>([](auto *element) {
        element->setShouldBeQualified();
        return element;
      })
      .Default([&](auto *element) {
        return this->emitError(
            loc,
            "'qualified' directive expects an attribute or a `type` directive");
      });
}

FailureOr<FormatElement *>
OpFormatParser::parseTypeDirectiveOperand(SMLoc loc, bool isRefChild) {
  FailureOr<FormatElement *> result = parseElement(TypeDirectiveContext);
  if (failed(result))
    return failure();

  FormatElement *element = *result;
  if (isa<LiteralElement>(element))
    return emitError(
        loc, "'type' directive operand expects variable or directive operand");

  if (auto *var = dyn_cast<OperandVariable>(element)) {
    unsigned opIdx = var->getVar() - op.operand_begin();
    if (!isRefChild && (fmt.allOperandTypes || seenOperandTypes.test(opIdx)))
      return emitError(loc, "'type' of '" + var->getVar()->name +
                                "' is already bound");
    if (isRefChild && !(fmt.allOperandTypes || seenOperandTypes.test(opIdx)))
      return emitError(loc, "'ref' of 'type($" + var->getVar()->name +
                                ")' is not bound by a prior 'type' directive");
    seenOperandTypes.set(opIdx);
  } else if (auto *var = dyn_cast<ResultVariable>(element)) {
    unsigned resIdx = var->getVar() - op.result_begin();
    if (!isRefChild && (fmt.allResultTypes || seenResultTypes.test(resIdx)))
      return emitError(loc, "'type' of '" + var->getVar()->name +
                                "' is already bound");
    if (isRefChild && !(fmt.allResultTypes || seenResultTypes.test(resIdx)))
      return emitError(loc, "'ref' of 'type($" + var->getVar()->name +
                                ")' is not bound by a prior 'type' directive");
    seenResultTypes.set(resIdx);
  } else if (isa<OperandsDirective>(&*element)) {
    if (!isRefChild && (fmt.allOperandTypes || seenOperandTypes.any()))
      return emitError(loc, "'operands' 'type' is already bound");
    if (isRefChild && !fmt.allOperandTypes)
      return emitError(loc, "'ref' of 'type(operands)' is not bound by a prior "
                            "'type' directive");
    fmt.allOperandTypes = true;
  } else if (isa<ResultsDirective>(&*element)) {
    if (!isRefChild && (fmt.allResultTypes || seenResultTypes.any()))
      return emitError(loc, "'results' 'type' is already bound");
    if (isRefChild && !fmt.allResultTypes)
      return emitError(loc, "'ref' of 'type(results)' is not bound by a prior "
                            "'type' directive");
    fmt.allResultTypes = true;
  } else {
    return emitError(loc, "invalid argument to 'type' directive");
  }
  return element;
}

LogicalResult
OpFormatParser::verifyOptionalGroupElements(SMLoc loc,
                                            ArrayRef<FormatElement *> elements,
                                            Optional<unsigned> anchorIndex) {
  for (auto &it : llvm::enumerate(elements)) {
    if (failed(verifyOptionalGroupElement(
            loc, it.value(), anchorIndex && *anchorIndex == it.index())))
      return failure();
  }
  return success();
}

LogicalResult OpFormatParser::verifyOptionalGroupElement(SMLoc loc,
                                                         FormatElement *element,
                                                         bool isAnchor) {
  return TypeSwitch<FormatElement *, LogicalResult>(element)
      // All attributes can be within the optional group, but only optional
      // attributes can be the anchor.
      .Case([&](AttributeVariable *attrEle) {
        if (isAnchor && !attrEle->getVar()->attr.isOptional())
          return emitError(loc, "only optional attributes can be used to "
                                "anchor an optional group");
        return success();
      })
      // Only optional-like(i.e. variadic) operands can be within an optional
      // group.
      .Case([&](OperandVariable *ele) {
        if (!ele->getVar()->isVariableLength())
          return emitError(loc, "only variable length operands can be used "
                                "within an optional group");
        return success();
      })
      // Only optional-like(i.e. variadic) results can be within an optional
      // group.
      .Case([&](ResultVariable *ele) {
        if (!ele->getVar()->isVariableLength())
          return emitError(loc, "only variable length results can be used "
                                "within an optional group");
        return success();
      })
      .Case([&](RegionVariable *) {
        // TODO: When ODS has proper support for marking "optional" regions, add
        // a check here.
        return success();
      })
      .Case([&](TypeDirective *ele) {
        return verifyOptionalGroupElement(loc, ele->getArg(),
                                          /*isAnchor=*/false);
      })
      .Case([&](FunctionalTypeDirective *ele) {
        if (failed(verifyOptionalGroupElement(loc, ele->getInputs(),
                                              /*isAnchor=*/false)))
          return failure();
        return verifyOptionalGroupElement(loc, ele->getResults(),
                                          /*isAnchor=*/false);
      })
      // Literals, whitespace, and custom directives may be used, but they can't
      // anchor the group.
      .Case<LiteralElement, WhitespaceElement, CustomDirective,
            FunctionalTypeDirective, OptionalElement>([&](FormatElement *) {
        if (isAnchor)
          return emitError(loc, "only variables and types can be used "
                                "to anchor an optional group");
        return success();
      })
      .Default([&](FormatElement *) {
        return emitError(loc, "only literals, types, and variables can be "
                              "used within an optional group");
      });
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
      llvm::MemoryBuffer::getMemBuffer(op.getAssemblyFormat()), SMLoc());
  OperationFormat format(op);
  OpFormatParser parser(mgr, format, op);
  FailureOr<std::vector<FormatElement *>> elements = parser.parse();
  if (failed(elements)) {
    // Exit the process if format errors are treated as fatal.
    if (formatErrorIsFatal) {
      // Invoke the interrupt handlers to run the file cleanup handlers.
      llvm::sys::RunInterruptHandlers();
      std::exit(1);
    }
    return;
  }
  format.elements = std::move(*elements);

  // Generate the printer and parser based on the parsed format.
  format.genParser(op, opClass);
  format.genPrinter(op, opClass);
}
