//===- IRDLToCpp.cpp - Converts IRDL definitions to C++ -------------------===//
//
// Part of the LLVM Project, under the A0ache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/IRDLToCpp/IRDLToCpp.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "TemplatingUtils.h"

using namespace mlir;

constexpr char headerTemplateText[] =
#include "Templates/Header.txt"
    ;

constexpr char declarationMacroFlag[] = "GEN_DIALECT_DECL_HEADER";
constexpr char definitionMacroFlag[] = "GEN_DIALECT_DEF";

namespace {

/// The set of strings that can be generated from a Dialect declaraiton
struct DialectStrings {
  std::string dialectName;
  std::string dialectCppName;
  std::string dialectCppShortName;
  std::string dialectBaseTypeName;

  std::string namespaceOpen;
  std::string namespaceClose;
  std::string namespacePath;
};

/// The set of strings that can be generated from a Type declaraiton
struct TypeStrings {
  StringRef typeName;
  std::string typeCppName;
};

/// The set of strings that can be generated from an Operation declaraiton
struct OpStrings {
  StringRef opName;
  std::string opCppName;
  SmallVector<std::string> opResultNames;
  SmallVector<std::string> opOperandNames;
  SmallVector<std::string> opRegionNames;
};

static std::string joinNameList(llvm::ArrayRef<std::string> names) {
  std::string nameArray;
  llvm::raw_string_ostream nameArrayStream(nameArray);
  nameArrayStream << "{\"" << llvm::join(names, "\", \"") << "\"}";

  return nameArray;
}

/// Generates the C++ type name for a TypeOp
static std::string typeToCppName(irdl::TypeOp type) {
  return llvm::formatv("{0}Type",
                       convertToCamelFromSnakeCase(type.getSymName(), true));
}

/// Generates the C++ class name for an OperationOp
static std::string opToCppName(irdl::OperationOp op) {
  return llvm::formatv("{0}Op",
                       convertToCamelFromSnakeCase(op.getSymName(), true));
}

/// Generates TypeStrings from a TypeOp
static TypeStrings getStrings(irdl::TypeOp type) {
  TypeStrings strings;
  strings.typeName = type.getSymName();
  strings.typeCppName = typeToCppName(type);
  return strings;
}

/// Generates OpStrings from an OperatioOp
static OpStrings getStrings(irdl::OperationOp op) {
  auto operandOp = op.getOp<irdl::OperandsOp>();
  auto resultOp = op.getOp<irdl::ResultsOp>();
  auto regionsOp = op.getOp<irdl::RegionsOp>();

  OpStrings strings;
  strings.opName = op.getSymName();
  strings.opCppName = opToCppName(op);

  if (operandOp) {
    strings.opOperandNames = SmallVector<std::string>(
        llvm::map_range(operandOp->getNames(), [](Attribute attr) {
          return llvm::formatv("{0}", cast<StringAttr>(attr));
        }));
  }

  if (resultOp) {
    strings.opResultNames = SmallVector<std::string>(
        llvm::map_range(resultOp->getNames(), [](Attribute attr) {
          return llvm::formatv("{0}", cast<StringAttr>(attr));
        }));
  }

  if (regionsOp) {
    strings.opRegionNames = SmallVector<std::string>(
        llvm::map_range(regionsOp->getNames(), [](Attribute attr) {
          return llvm::formatv("{0}", cast<StringAttr>(attr));
        }));
  }

  return strings;
}

/// Fills a dictionary with values from TypeStrings
static void fillDict(irdl::detail::dictionary &dict,
                     const TypeStrings &strings) {
  dict["TYPE_NAME"] = strings.typeName;
  dict["TYPE_CPP_NAME"] = strings.typeCppName;
}

/// Fills a dictionary with values from OpStrings
static void fillDict(irdl::detail::dictionary &dict, const OpStrings &strings) {
  const auto operandCount = strings.opOperandNames.size();
  const auto resultCount = strings.opResultNames.size();
  const auto regionCount = strings.opRegionNames.size();

  dict["OP_NAME"] = strings.opName;
  dict["OP_CPP_NAME"] = strings.opCppName;
  dict["OP_OPERAND_COUNT"] = std::to_string(strings.opOperandNames.size());
  dict["OP_RESULT_COUNT"] = std::to_string(strings.opResultNames.size());
  dict["OP_OPERAND_INITIALIZER_LIST"] =
      operandCount ? joinNameList(strings.opOperandNames) : "{\"\"}";
  dict["OP_RESULT_INITIALIZER_LIST"] =
      resultCount ? joinNameList(strings.opResultNames) : "{\"\"}";
  dict["OP_REGION_COUNT"] = std::to_string(regionCount);
}

/// Fills a dictionary with values from DialectStrings
static void fillDict(irdl::detail::dictionary &dict,
                     const DialectStrings &strings) {
  dict["DIALECT_NAME"] = strings.dialectName;
  dict["DIALECT_BASE_TYPE_NAME"] = strings.dialectBaseTypeName;
  dict["DIALECT_CPP_NAME"] = strings.dialectCppName;
  dict["DIALECT_CPP_SHORT_NAME"] = strings.dialectCppShortName;
  dict["NAMESPACE_OPEN"] = strings.namespaceOpen;
  dict["NAMESPACE_CLOSE"] = strings.namespaceClose;
  dict["NAMESPACE_PATH"] = strings.namespacePath;
}

static LogicalResult generateTypedefList(irdl::DialectOp &dialect,
                                         SmallVector<std::string> &typeNames) {
  auto typeOps = dialect.getOps<irdl::TypeOp>();
  auto range = llvm::map_range(typeOps, typeToCppName);
  typeNames = SmallVector<std::string>(range);
  return success();
}

static LogicalResult generateOpList(irdl::DialectOp &dialect,
                                    SmallVector<std::string> &opNames) {
  auto operationOps = dialect.getOps<irdl::OperationOp>();
  auto range = llvm::map_range(operationOps, opToCppName);
  opNames = SmallVector<std::string>(range);
  return success();
}

} // namespace

static LogicalResult generateTypeInclude(irdl::TypeOp type, raw_ostream &output,
                                         irdl::detail::dictionary &dict) {
  static const auto typeDeclTemplate = irdl::detail::Template(
#include "Templates/TypeDecl.txt"
  );

  fillDict(dict, getStrings(type));
  typeDeclTemplate.render(output, dict);

  return success();
}

static void generateOpGetterDeclarations(irdl::detail::dictionary &dict,
                                         const OpStrings &opStrings) {
  auto opGetters = std::string{};
  auto resGetters = std::string{};
  auto regionGetters = std::string{};
  auto regionAdaptorGetters = std::string{};

  for (size_t i = 0, end = opStrings.opOperandNames.size(); i < end; ++i) {
    const auto op =
        llvm::convertToCamelFromSnakeCase(opStrings.opOperandNames[i], true);
    opGetters += llvm::formatv("::mlir::Value get{0}() { return "
                               "getStructuredOperands({1}).front(); }\n  ",
                               op, i);
  }
  for (size_t i = 0, end = opStrings.opResultNames.size(); i < end; ++i) {
    const auto op =
        llvm::convertToCamelFromSnakeCase(opStrings.opResultNames[i], true);
    resGetters += llvm::formatv(
        R"(::mlir::Value get{0}() { return ::llvm::cast<::mlir::Value>(getStructuredResults({1}).front()); }
  )",
        op, i);
  }

  for (size_t i = 0, end = opStrings.opRegionNames.size(); i < end; ++i) {
    const auto op =
        llvm::convertToCamelFromSnakeCase(opStrings.opRegionNames[i], true);
    regionAdaptorGetters += llvm::formatv(
        R"(::mlir::Region &get{0}() { return *getRegions()[{1}]; }
  )",
        op, i);
    regionGetters += llvm::formatv(
        R"(::mlir::Region &get{0}() { return (*this)->getRegion({1}); }
  )",
        op, i);
  }

  dict["OP_OPERAND_GETTER_DECLS"] = opGetters;
  dict["OP_RESULT_GETTER_DECLS"] = resGetters;
  dict["OP_REGION_ADAPTER_GETTER_DECLS"] = regionAdaptorGetters;
  dict["OP_REGION_GETTER_DECLS"] = regionGetters;
}

static void generateOpBuilderDeclarations(irdl::detail::dictionary &dict,
                                          const OpStrings &opStrings) {
  std::string buildDecls;
  llvm::raw_string_ostream stream{buildDecls};

  auto resultParams =
      llvm::join(llvm::map_range(opStrings.opResultNames,
                                 [](StringRef name) -> std::string {
                                   return llvm::formatv(
                                       "::mlir::Type {0}, ",
                                       llvm::convertToCamelFromSnakeCase(name));
                                 }),
                 "");

  auto operandParams =
      llvm::join(llvm::map_range(opStrings.opOperandNames,
                                 [](StringRef name) -> std::string {
                                   return llvm::formatv(
                                       "::mlir::Value {0}, ",
                                       llvm::convertToCamelFromSnakeCase(name));
                                 }),
                 "");

  stream << llvm::formatv(
      R"(static void build(::mlir::OpBuilder &opBuilder, ::mlir::OperationState &opState, {0} {1} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {{});)",
      resultParams, operandParams);
  stream << "\n";
  stream << llvm::formatv(
      R"(static {0} create(::mlir::OpBuilder &opBuilder, ::mlir::Location location, {1} {2} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {{});)",
      opStrings.opCppName, resultParams, operandParams);
  stream << "\n";
  stream << llvm::formatv(
      R"(static {0} create(::mlir::ImplicitLocOpBuilder &opBuilder, {1} {2} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {{});)",
      opStrings.opCppName, resultParams, operandParams);
  stream << "\n";
  dict["OP_BUILD_DECLS"] = buildDecls;
}

// add traits to the dictionary, return true if any were added
static SmallVector<std::string> generateTraits(irdl::OperationOp op,
                                               const OpStrings &strings) {
  SmallVector<std::string> cppTraitNames;
  if (!strings.opRegionNames.empty()) {
    cppTraitNames.push_back(
        llvm::formatv("::mlir::OpTrait::NRegions<{0}>::Impl",
                      strings.opRegionNames.size())
            .str());

    // Requires verifyInvariantsImpl is implemented on the op
    cppTraitNames.emplace_back("::mlir::OpTrait::OpInvariants");
  }
  return cppTraitNames;
}

static LogicalResult generateOperationInclude(irdl::OperationOp op,
                                              raw_ostream &output,
                                              irdl::detail::dictionary &dict) {
  static const auto perOpDeclTemplate = irdl::detail::Template(
#include "Templates/PerOperationDecl.txt"
  );
  const auto opStrings = getStrings(op);
  fillDict(dict, opStrings);

  SmallVector<std::string> traitNames = generateTraits(op, opStrings);
  if (traitNames.empty())
    dict["OP_TEMPLATE_ARGS"] = opStrings.opCppName;
  else
    dict["OP_TEMPLATE_ARGS"] = llvm::formatv("{0}, {1}", opStrings.opCppName,
                                             llvm::join(traitNames, ", "));

  generateOpGetterDeclarations(dict, opStrings);
  generateOpBuilderDeclarations(dict, opStrings);

  perOpDeclTemplate.render(output, dict);
  return success();
}

static LogicalResult generateInclude(irdl::DialectOp dialect,
                                     raw_ostream &output,
                                     DialectStrings &dialectStrings) {
  static const auto dialectDeclTemplate = irdl::detail::Template(
#include "Templates/DialectDecl.txt"
  );
  static const auto typeHeaderDeclTemplate = irdl::detail::Template(
#include "Templates/TypeHeaderDecl.txt"
  );

  irdl::detail::dictionary dict;
  fillDict(dict, dialectStrings);

  dialectDeclTemplate.render(output, dict);
  typeHeaderDeclTemplate.render(output, dict);

  auto typeOps = dialect.getOps<irdl::TypeOp>();
  auto operationOps = dialect.getOps<irdl::OperationOp>();

  for (auto &&typeOp : typeOps) {
    if (failed(generateTypeInclude(typeOp, output, dict)))
      return failure();
  }

  SmallVector<std::string> opNames;
  if (failed(generateOpList(dialect, opNames)))
    return failure();

  auto classDeclarations =
      llvm::join(llvm::map_range(opNames,
                                 [](llvm::StringRef name) -> std::string {
                                   return llvm::formatv("class {0};", name);
                                 }),
                 "\n");
  const auto forwardDeclarations = llvm::formatv(
      "{1}\n{0}\n{2}", std::move(classDeclarations),
      dialectStrings.namespaceOpen, dialectStrings.namespaceClose);

  output << forwardDeclarations;
  for (auto &&operationOp : operationOps) {
    if (failed(generateOperationInclude(operationOp, output, dict)))
      return failure();
  }

  return success();
}

static void generateRegionConstraintVerifiers(
    irdl::detail::dictionary &dict, irdl::OperationOp op,
    const OpStrings &strings, SmallVectorImpl<std::string> &verifierHelpers,
    SmallVectorImpl<std::string> &verifierCalls) {
  auto regionsOp = op.getOp<irdl::RegionsOp>();
  if (strings.opRegionNames.empty() || !regionsOp)
    return;

  for (size_t i = 0; i < strings.opRegionNames.size(); ++i) {
    std::string regionName = strings.opRegionNames[i];
    std::string helperFnName =
        llvm::formatv("__mlir_irdl_local_region_constraint_{0}_{1}",
                      strings.opCppName, regionName)
            .str();

    // Extract the actual region constraint from the IRDL RegionOp
    std::string condition = "true";
    std::string textualConditionName = "any region";

    if (auto regionDefOp =
            dyn_cast<irdl::RegionOp>(regionsOp->getArgs()[i].getDefiningOp())) {
      // Generate constraint condition based on RegionOp attributes
      SmallVector<std::string> conditionParts;
      SmallVector<std::string> descriptionParts;

      // Check number of blocks constraint
      if (auto blockCount = regionDefOp.getNumberOfBlocks()) {
        conditionParts.push_back(
            llvm::formatv("region.getBlocks().size() == {0}",
                          blockCount.value())
                .str());
        descriptionParts.push_back(
            llvm::formatv("exactly {0} block(s)", blockCount.value()).str());
      }

      // Check entry block arguments constraint
      if (regionDefOp.getConstrainedArguments()) {
        size_t expectedArgCount = regionDefOp.getEntryBlockArgs().size();
        conditionParts.push_back(
            llvm::formatv("region.getNumArguments() == {0}", expectedArgCount)
                .str());
        descriptionParts.push_back(
            llvm::formatv("{0} entry block argument(s)", expectedArgCount)
                .str());
      }

      // Combine conditions
      if (!conditionParts.empty()) {
        condition = llvm::join(conditionParts, " && ");
      }

      // Generate descriptive error message
      if (!descriptionParts.empty()) {
        textualConditionName =
            llvm::formatv("region with {0}",
                          llvm::join(descriptionParts, " and "))
                .str();
      }
    }

    verifierHelpers.push_back(llvm::formatv(
        R"(static ::llvm::LogicalResult {0}(::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName, unsigned regionIndex) {{
    if (!({1})) {{
      return op->emitOpError("region #") << regionIndex
          << (regionName.empty() ? " " : " ('" + regionName + "') ")
          << "failed to verify constraint: {2}";
      }
  return ::mlir::success();
})",
        helperFnName, condition, textualConditionName));

    verifierCalls.push_back(llvm::formatv(R"(
  if (::mlir::failed({0}(*this, (*this)->getRegion({1}), "{2}", {1})))
    return ::mlir::failure();)",
                                          helperFnName, i, regionName)
                                .str());
  }
}

static void generateVerifiers(irdl::detail::dictionary &dict,
                              irdl::OperationOp op, const OpStrings &strings) {
  SmallVector<std::string> verifierHelpers;
  SmallVector<std::string> verifierCalls;

  generateRegionConstraintVerifiers(dict, op, strings, verifierHelpers,
                                    verifierCalls);

  // Add an overall verifier that sequences the helper calls
  std::string verifierDef =
      llvm::formatv(R"(
::llvm::LogicalResult {0}::verifyInvariantsImpl() {{
  if(::mlir::failed(verify()))
    return ::mlir::failure();

  {1}

  return ::mlir::success();
})",
                    strings.opCppName, llvm::join(verifierCalls, "\n"));

  dict["OP_VERIFIER_HELPERS"] = llvm::join(verifierHelpers, "\n");
  dict["OP_VERIFIER"] = verifierDef;
}

static std::string generateOpDefinition(irdl::detail::dictionary &dict,
                                        irdl::OperationOp op) {
  static const auto perOpDefTemplate = mlir::irdl::detail::Template{
#include "Templates/PerOperationDef.txt"
  };

  auto opStrings = getStrings(op);
  fillDict(dict, opStrings);

  auto resultTypes = llvm::join(
      llvm::map_range(opStrings.opResultNames,
                      [](StringRef attr) -> std::string {
                        return llvm::formatv("::mlir::Type {0}, ", attr);
                      }),
      "");
  auto operandTypes = llvm::join(
      llvm::map_range(opStrings.opOperandNames,
                      [](StringRef attr) -> std::string {
                        return llvm::formatv("::mlir::Value {0}, ", attr);
                      }),
      "");
  auto operandAdder =
      llvm::join(llvm::map_range(opStrings.opOperandNames,
                                 [](StringRef attr) -> std::string {
                                   return llvm::formatv(
                                       "  opState.addOperands({0});", attr);
                                 }),
                 "\n");
  auto resultAdder = llvm::join(
      llvm::map_range(opStrings.opResultNames,
                      [](StringRef attr) -> std::string {
                        return llvm::formatv("  opState.addTypes({0});", attr);
                      }),
      "\n");

  const auto buildDefinition = llvm::formatv(
      R"(
void {0}::build(::mlir::OpBuilder &opBuilder, ::mlir::OperationState &opState, {1} {2} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {{
{3}
{4}
}

{0} {0}::create(::mlir::OpBuilder &opBuilder, ::mlir::Location location, {1} {2} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {{
  ::mlir::OperationState __state__(location, getOperationName());
  build(opBuilder, __state__, {5} {6} attributes);
  auto __res__ = ::llvm::dyn_cast<{0}>(opBuilder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

{0} {0}::create(::mlir::ImplicitLocOpBuilder &opBuilder, {1} {2} ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {{
  return create(opBuilder, opBuilder.getLoc(), {5} {6} attributes);
}
)",
      opStrings.opCppName, std::move(resultTypes), std::move(operandTypes),
      std::move(operandAdder), std::move(resultAdder),
      llvm::join(opStrings.opResultNames, ",") +
          (!opStrings.opResultNames.empty() ? "," : ""),
      llvm::join(opStrings.opOperandNames, ",") +
          (!opStrings.opOperandNames.empty() ? "," : ""));

  dict["OP_BUILD_DEFS"] = buildDefinition;

  generateVerifiers(dict, op, opStrings);

  std::string str;
  llvm::raw_string_ostream stream{str};
  perOpDefTemplate.render(stream, dict);
  return str;
}

static std::string
generateTypeVerifierCase(StringRef name, const DialectStrings &dialectStrings) {
  return llvm::formatv(
      R"(.Case({1}::{0}::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
value = {1}::{0}::get(parser.getContext());
return ::mlir::success(!!value);
}))",
      name, dialectStrings.namespacePath);
}

static LogicalResult generateLib(irdl::DialectOp dialect, raw_ostream &output,
                                 DialectStrings &dialectStrings) {

  static const auto typeHeaderDefTemplate = mlir::irdl::detail::Template{
#include "Templates/TypeHeaderDef.txt"
  };
  static const auto typeDefTemplate = mlir::irdl::detail::Template{
#include "Templates/TypeDef.txt"
  };
  static const auto dialectDefTemplate = mlir::irdl::detail::Template{
#include "Templates/DialectDef.txt"
  };

  irdl::detail::dictionary dict;
  fillDict(dict, dialectStrings);

  typeHeaderDefTemplate.render(output, dict);

  SmallVector<std::string> typeNames;
  if (failed(generateTypedefList(dialect, typeNames)))
    return failure();

  dict["TYPE_LIST"] = llvm::join(
      llvm::map_range(typeNames,
                      [&dialectStrings](llvm::StringRef name) -> std::string {
                        return llvm::formatv(
                            "{0}::{1}", dialectStrings.namespacePath, name);
                      }),
      ",\n");

  auto typeVerifierGenerator =
      [&dialectStrings](llvm::StringRef name) -> std::string {
    return generateTypeVerifierCase(name, dialectStrings);
  };

  auto typeCase =
      llvm::join(llvm::map_range(typeNames, typeVerifierGenerator), "\n");

  dict["TYPE_PARSER"] = llvm::formatv(
      R"(static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    {0}
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {{
      *mnemonic = keyword;
      return std::nullopt;
    });
})",
      std::move(typeCase));

  auto typePrintCase =
      llvm::join(llvm::map_range(typeNames,
                                 [&](llvm::StringRef name) -> std::string {
                                   return llvm::formatv(
                                       R"(.Case<{1}::{0}>([&](auto t) {
      printer << {1}::{0}::getMnemonic();
      return ::mlir::success();
    }))",
                                       name, dialectStrings.namespacePath);
                                 }),
                 "\n");
  dict["TYPE_PRINTER"] = llvm::formatv(
      R"(static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)
    {0}
    .Default([](auto) {{ return ::mlir::failure(); });
})",
      std::move(typePrintCase));

  dict["TYPE_DEFINES"] =
      join(map_range(typeNames,
                     [&](StringRef name) -> std::string {
                       return formatv("MLIR_DEFINE_EXPLICIT_TYPE_ID({1}::{0})",
                                      name, dialectStrings.namespacePath);
                     }),
           "\n");

  typeDefTemplate.render(output, dict);

  auto operations = dialect.getOps<irdl::OperationOp>();
  SmallVector<std::string> opNames;
  if (failed(generateOpList(dialect, opNames)))
    return failure();

  const auto commaSeparatedOpList = llvm::join(
      map_range(opNames,
                [&dialectStrings](llvm::StringRef name) -> std::string {
                  return llvm::formatv("{0}::{1}", dialectStrings.namespacePath,
                                       name);
                }),
      ",\n");

  const auto opDefinitionGenerator = [&dict](irdl::OperationOp op) {
    return generateOpDefinition(dict, op);
  };

  const auto perOpDefinitions =
      llvm::join(llvm::map_range(operations, opDefinitionGenerator), "\n");

  dict["OP_LIST"] = commaSeparatedOpList;
  dict["OP_CLASSES"] = perOpDefinitions;
  output << perOpDefinitions;
  dialectDefTemplate.render(output, dict);

  return success();
}

static LogicalResult verifySupported(irdl::DialectOp dialect) {
  LogicalResult res = success();
  dialect.walk([&](mlir::Operation *op) {
    res =
        llvm::TypeSwitch<Operation *, LogicalResult>(op)
            .Case<irdl::DialectOp>(([](irdl::DialectOp) { return success(); }))
            .Case<irdl::OperationOp>(
                ([](irdl::OperationOp) { return success(); }))
            .Case<irdl::TypeOp>(([](irdl::TypeOp) { return success(); }))
            .Case<irdl::OperandsOp>(([](irdl::OperandsOp op) -> LogicalResult {
              if (llvm::all_of(
                      op.getVariadicity(), [](irdl::VariadicityAttr attr) {
                        return attr.getValue() == irdl::Variadicity::single;
                      }))
                return success();
              return op.emitError("IRDL C++ translation does not yet support "
                                  "variadic operations");
            }))
            .Case<irdl::ResultsOp>(([](irdl::ResultsOp op) -> LogicalResult {
              if (llvm::all_of(
                      op.getVariadicity(), [](irdl::VariadicityAttr attr) {
                        return attr.getValue() == irdl::Variadicity::single;
                      }))
                return success();
              return op.emitError(
                  "IRDL C++ translation does not yet support variadic results");
            }))
            .Case<irdl::AnyOp>(([](irdl::AnyOp) { return success(); }))
            .Case<irdl::RegionOp>(([](irdl::RegionOp) { return success(); }))
            .Case<irdl::RegionsOp>(([](irdl::RegionsOp) { return success(); }))
            .Default([](mlir::Operation *op) -> LogicalResult {
              return op->emitError("IRDL C++ translation does not yet support "
                                   "translation of ")
                     << op->getName() << " operation";
            });

    if (failed(res))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return res;
}

LogicalResult
irdl::translateIRDLDialectToCpp(llvm::ArrayRef<irdl::DialectOp> dialects,
                                raw_ostream &output) {
  static const auto typeDefTempl = detail::Template(
#include "Templates/TypeDef.txt"
  );

  llvm::SmallMapVector<DialectOp, DialectStrings, 2> dialectStringTable;

  for (auto dialect : dialects) {
    if (failed(verifySupported(dialect)))
      return failure();

    StringRef dialectName = dialect.getSymName();

    SmallVector<SmallString<8>> namespaceAbsolutePath{{"mlir"}, dialectName};
    std::string namespaceOpen;
    std::string namespaceClose;
    std::string namespacePath;
    llvm::raw_string_ostream namespaceOpenStream(namespaceOpen);
    llvm::raw_string_ostream namespaceCloseStream(namespaceClose);
    llvm::raw_string_ostream namespacePathStream(namespacePath);
    for (auto &pathElement : namespaceAbsolutePath) {
      namespaceOpenStream << "namespace " << pathElement << " {\n";
      namespaceCloseStream << "} // namespace " << pathElement << "\n";
      namespacePathStream << "::" << pathElement;
    }

    std::string cppShortName =
        llvm::convertToCamelFromSnakeCase(dialectName, true);
    std::string dialectBaseTypeName = llvm::formatv("{0}Type", cppShortName);
    std::string cppName = llvm::formatv("{0}Dialect", cppShortName);

    DialectStrings dialectStrings;
    dialectStrings.dialectName = dialectName;
    dialectStrings.dialectBaseTypeName = dialectBaseTypeName;
    dialectStrings.dialectCppName = cppName;
    dialectStrings.dialectCppShortName = cppShortName;
    dialectStrings.namespaceOpen = namespaceOpen;
    dialectStrings.namespaceClose = namespaceClose;
    dialectStrings.namespacePath = namespacePath;

    dialectStringTable[dialect] = std::move(dialectStrings);
  }

  // generate the actual header
  output << headerTemplateText;

  output << llvm::formatv("#ifdef {0}\n#undef {0}\n", declarationMacroFlag);
  for (auto dialect : dialects) {

    auto &dialectStrings = dialectStringTable[dialect];
    auto &dialectName = dialectStrings.dialectName;

    if (failed(generateInclude(dialect, output, dialectStrings)))
      return dialect->emitError("Error in Dialect " + dialectName +
                                " while generating headers");
  }
  output << llvm::formatv("#endif // #ifdef {}\n", declarationMacroFlag);

  output << llvm::formatv("#ifdef {0}\n#undef {0}\n ", definitionMacroFlag);
  for (auto &dialect : dialects) {
    auto &dialectStrings = dialectStringTable[dialect];
    auto &dialectName = dialectStrings.dialectName;

    if (failed(generateLib(dialect, output, dialectStrings)))
      return dialect->emitError("Error in Dialect " + dialectName +
                                " while generating library");
  }
  output << llvm::formatv("#endif // #ifdef {}\n", definitionMacroFlag);

  return success();
}
