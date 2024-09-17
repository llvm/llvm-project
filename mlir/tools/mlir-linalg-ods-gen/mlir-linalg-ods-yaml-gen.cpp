//===- mlir-linalg-ods-yaml-gen.cpp - Linalg ODS generation from yaml  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an ODS (and C++) generator from a YAML form
// derived from the mathematical expression of linalg named ops. Typically a
// math oriented DSL will be used to export the essential representation to
// this form, and maintaining the SOT at the math level (versus recreating it
// in MLIR) is deemed to have systemic value.
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include <optional>

using namespace mlir;

using llvm::yaml::Input;
using llvm::yaml::MappingTraits;
using llvm::yaml::ScalarEnumerationTraits;
using llvm::yaml::ScalarTraits;

#define DEBUG_TYPE "linalg-ods-gen"

//===----------------------------------------------------------------------===//
// Mapping structs (correspond to data types in the YAML description).
// TODO: Since this is a schema/part of the contract, it should be moved to
// a real header.
//===----------------------------------------------------------------------===//

namespace {

struct LinalgYAMLContext {
  MLIRContext *mlirContext;
};

struct LinalgOpMetadata {
  std::string name;
  std::string cppClassName;
  std::optional<std::string> doc;
  SmallVector<std::string> implements;
  SmallVector<std::string> defines;
};

struct SerializedAffineMap {
  AffineMapAttr affineMapAttr;

  AffineMap affineMap() { return affineMapAttr.getValue(); }
};

enum class LinalgOperandDefKind {
  InputTensor,
  Scalar,
  OutputTensor,
  IndexAttr,
  UnaryFnAttr,
  BinaryFnAttr,
  TernaryFnAttr,
  TypeFnAttr
};

struct LinalgOperandDef {
  std::string name;
  LinalgOperandDefKind kind;
  std::optional<std::string> typeVar;
  std::optional<SerializedAffineMap> shapeMap;
  std::optional<SerializedAffineMap> indexAttrMap;
  std::optional<SmallVector<int64_t>> defaultIndices;
  std::optional<std::string> defaultFn;
};

enum class LinalgIteratorTypeDef {
  parallel,
  reduction,
};

struct LinalgIndexingMapsConfig {
  std::optional<SmallVector<SerializedAffineMap>> staticIndexingMaps;
};

struct ScalarExpression;

enum class ScalarFnKind { Unary, Binary, Ternary, Type };

struct ScalarFn {
  ScalarFnKind kind;
  std::optional<std::string> fnName;
  std::optional<std::string> attrName;
  std::optional<std::string> typeVar;
  // NOTE: This must be of arity 1, but to break the self-referential cycle,
  // we use a heap allocated vector.
  std::vector<ScalarExpression> operands;
};

struct ScalarExpression {
  std::optional<std::string> arg;
  std::optional<std::string> constant;
  std::optional<int64_t> index;
  std::optional<ScalarFn> scalarFn;
};

struct ScalarAssign {
  std::string arg;
  ScalarExpression value;
};

struct LinalgStructuredOpConfig {
  SmallVector<LinalgOperandDef> args;
  LinalgIndexingMapsConfig indexingMaps;
  SmallVector<LinalgIteratorTypeDef> iteratorTypes;
  std::vector<ScalarAssign> assignments;
};

struct LinalgOpConfig {
  std::optional<LinalgOpMetadata> metadata;
  std::optional<LinalgStructuredOpConfig> structuredOp;
};

} // namespace

//===----------------------------------------------------------------------===//
// Mapping traits.
//===----------------------------------------------------------------------===//

LLVM_YAML_IS_SEQUENCE_VECTOR(LinalgOperandDef)
LLVM_YAML_IS_SEQUENCE_VECTOR(SerializedAffineMap)
LLVM_YAML_IS_SEQUENCE_VECTOR(LinalgIteratorTypeDef)
LLVM_YAML_IS_SEQUENCE_VECTOR(ScalarAssign)
LLVM_YAML_IS_SEQUENCE_VECTOR(ScalarExpression)
LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(LinalgOpConfig)

namespace llvm {
namespace yaml {

/// Top-level type containing op metadata and one of a concrete op type.
/// Currently, the only defined op type is `structured_op` (maps to
/// `LinalgStructuredOpConfig`).
template <>
struct MappingTraits<LinalgOpConfig> {
  static void mapping(IO &io, LinalgOpConfig &info) {
    io.mapOptional("metadata", info.metadata);
    io.mapOptional("structured_op", info.structuredOp);
  }
};

/// A structured op models (at most) a single contraction by modeling
///   - A list of named arguments (`LinalgOperandDef`), which can be inputs,
///     outputs, or index attributes.
///   - List of indexing maps (see `LinalgIndexingMaps`).
///   - Iterator types (see `LinalgIteratorTypeDef`).
///   - List of scalar level assignment (see `ScalarAssign`).
template <>
struct MappingTraits<LinalgStructuredOpConfig> {
  static void mapping(IO &io, LinalgStructuredOpConfig &info) {
    io.mapRequired("args", info.args);
    io.mapRequired("indexing_maps", info.indexingMaps);
    io.mapRequired("iterator_types", info.iteratorTypes);
    io.mapRequired("assignments", info.assignments);
  }
};

/// Maps a named tensor, scalar or attribute argument to an operation,
/// consisting of:
///   - `name`: Must be unique within the operation.
///   - `usage`: How the argument is used (input, output, attribute, etc).
///   - `type_var`: The symbolic type variable that binds to the element or self
///     type of the tensor or scalar argument, respectively.
///   - `shape_map`: An optional AffineMap from all op symbols to the shape of
///     the argument. Only tensor arguments have a `shape_map`. Each shape must
///     be normalized over the same list of symbols and have no dimension
///     inputs.
///   - `index_attr_map`: An optional AffineMap from all op symbols to the
///     index attribute symbols. During op creation these symbols are replaced
///     by the corresponding `name` index attribue values. Only index attribute
///     arguments have an `index_attr_map`.
///   - `default_indices`: An optional default initialization for index
///     attribute arguments.
///   - `default_fn`: An optional default initialization for function attribute
///     arguments.
template <>
struct MappingTraits<LinalgOperandDef> {
  static void mapping(IO &io, LinalgOperandDef &info) {
    io.mapRequired("name", info.name);
    io.mapRequired("kind", info.kind);
    io.mapOptional("type_var", info.typeVar);
    io.mapOptional("shape_map", info.shapeMap);
    io.mapOptional("index_attr_map", info.indexAttrMap);
    io.mapOptional("default_indices", info.defaultIndices);
    io.mapOptional("default_fn", info.defaultFn);
  }
};

/// Usage enum for a named argument.
template <>
struct ScalarEnumerationTraits<LinalgOperandDefKind> {
  static void enumeration(IO &io, LinalgOperandDefKind &value) {
    io.enumCase(value, "input_tensor", LinalgOperandDefKind::InputTensor);
    io.enumCase(value, "scalar", LinalgOperandDefKind::Scalar);
    io.enumCase(value, "output_tensor", LinalgOperandDefKind::OutputTensor);
    io.enumCase(value, "index_attr", LinalgOperandDefKind::IndexAttr);
    io.enumCase(value, "unary_fn_attr", LinalgOperandDefKind::UnaryFnAttr);
    io.enumCase(value, "binary_fn_attr", LinalgOperandDefKind::BinaryFnAttr);
    io.enumCase(value, "ternary_fn_attr", LinalgOperandDefKind::TernaryFnAttr);
    io.enumCase(value, "type_fn_attr", LinalgOperandDefKind::TypeFnAttr);
  }
};

/// Iterator type enum.
template <>
struct ScalarEnumerationTraits<LinalgIteratorTypeDef> {
  static void enumeration(IO &io, LinalgIteratorTypeDef &value) {
    io.enumCase(value, "parallel", LinalgIteratorTypeDef::parallel);
    io.enumCase(value, "reduction", LinalgIteratorTypeDef::reduction);
  }
};

/// Metadata about the op (name, C++ name, and documentation).
template <>
struct MappingTraits<LinalgOpMetadata> {
  static void mapping(IO &io, LinalgOpMetadata &info) {
    io.mapRequired("name", info.name);
    io.mapRequired("cpp_class_name", info.cppClassName);
    io.mapOptional("doc", info.doc);
    io.mapOptional("implements", info.implements);
    io.mapOptional("defines", info.defines);
  }
};

/// How the ops indexing maps are produced. Must be one of:
///   - static_indexing_maps: A static list of AffineMaps, possibly with
///     some symbols that bind to attributes of the op. Each indexing map must
///     be normalized over the same list of dimensions, and its symbols must
///     match the symbols for argument shapes.
template <>
struct MappingTraits<LinalgIndexingMapsConfig> {
  static void mapping(IO &io, LinalgIndexingMapsConfig &info) {
    io.mapOptional("static_indexing_maps", info.staticIndexingMaps);
  }
};

/// Models an assignment to a named output.
///   - The `arg` name must match a named output.
///   - The `value` is a scalar expression for computing the value to
///     assign (see `ScalarExpression`).
template <>
struct MappingTraits<ScalarAssign> {
  static void mapping(IO &io, ScalarAssign &info) {
    io.mapRequired("arg", info.arg);
    io.mapRequired("value", info.value);
  }
};

/// A scalar expression (RHS of an assignment). Must be one of:
///   - `scalar_arg`: An operation argument.
///   - `scalar_const`: A constant definition.
///   - `scalar_index`: An iteration index.
///   - `scalar_fn`: A named function (see `ScalarFn`).
template <>
struct MappingTraits<ScalarExpression> {
  static void mapping(IO &io, ScalarExpression &info) {
    io.mapOptional("scalar_arg", info.arg);
    io.mapOptional("scalar_const", info.constant);
    io.mapOptional("scalar_index", info.index);
    io.mapOptional("scalar_fn", info.scalarFn);
  }
};

/// Scalar function kind enum.
template <>
struct ScalarEnumerationTraits<ScalarFnKind> {
  static void enumeration(IO &io, ScalarFnKind &value) {
    io.enumCase(value, "unary", ScalarFnKind::Unary);
    io.enumCase(value, "binary", ScalarFnKind::Binary);
    io.enumCase(value, "ternary", ScalarFnKind::Ternary);
    io.enumCase(value, "type", ScalarFnKind::Type);
  }
};

/// A scalar expression that evaluates a named function.
/// Functions are generally "math" level and type polymorphic. Builtin
/// functions include:
///   - `add(lhs, rhs)`
///   - `mul(lhs, rhs)`
template <>
struct MappingTraits<ScalarFn> {
  static void mapping(IO &io, ScalarFn &info) {
    io.mapRequired("kind", info.kind);
    io.mapOptional("fn_name", info.fnName);
    io.mapOptional("attr_name", info.attrName);
    io.mapOptional("type_var", info.typeVar);
    io.mapRequired("operands", info.operands);
  }
};

/// Helper mapping which accesses an AffineMapAttr as a serialized string of
/// the same.
template <>
struct ScalarTraits<SerializedAffineMap> {
  static void output(const SerializedAffineMap &value, void *rawYamlContext,
                     raw_ostream &out) {
    assert(value.affineMapAttr);
    value.affineMapAttr.print(out);
  }
  static StringRef input(StringRef scalar, void *rawYamlContext,
                         SerializedAffineMap &value) {
    assert(rawYamlContext);
    auto *yamlContext = static_cast<LinalgYAMLContext *>(rawYamlContext);
    if (auto attr = dyn_cast_or_null<AffineMapAttr>(
            mlir::parseAttribute(scalar, yamlContext->mlirContext)))
      value.affineMapAttr = attr;
    else if (!value.affineMapAttr || !isa<AffineMapAttr>(value.affineMapAttr))
      return "could not parse as an affine map attribute";
    return StringRef();
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

} // namespace yaml
} // namespace llvm

namespace {

//===----------------------------------------------------------------------===//
// Generation utilities
//===----------------------------------------------------------------------===//

class GenerationContext {
public:
  GenerationContext(MLIRContext *context, raw_ostream *odsOut,
                    raw_ostream *defnOut)
      : context(context), loc(UnknownLoc::get(context)), odsOut(odsOut),
        defnOut(defnOut) {}

  MLIRContext *getContext() { return context; }

  void setLoc(Location loc) { this->loc = loc; }
  Location getLoc() { return loc; }

  bool shouldGenerateOds() { return odsOut; }
  bool shouldGenerateDefns() { return defnOut; }

  raw_ostream &odss() {
    assert(odsOut && "ODS stream not defined");
    return *odsOut;
  }

  raw_ostream &defns() {
    assert(defnOut && "Definition stream not defined");
    return *defnOut;
  }

private:
  MLIRContext *context;
  Location loc;
  raw_ostream *odsOut;
  raw_ostream *defnOut;
};

} // namespace

static std::string generateCppExpression(SerializedAffineMap self,
                                         StringRef contextName) {
  std::string printedStr;
  llvm::raw_string_ostream printedSs(printedStr);
  self.affineMapAttr.print(printedSs);

  static const char exprFormat[] =
      R"FMT(llvm::cast<AffineMapAttr>(mlir::parseAttribute("{0}", {1})).getValue())FMT";
  return llvm::formatv(exprFormat, printedStr, contextName);
}

template <typename Container>
static std::string interleaveToString(Container &container,
                                      StringRef separator) {
  std::string result;
  llvm::raw_string_ostream ss(result);
  llvm::interleave(container, ss, separator);
  return result;
}

static std::optional<int>
findTensorDefArgIndex(StringRef name, SmallVectorImpl<LinalgOperandDef> &args) {
  for (const auto &it : llvm::enumerate(args)) {
    if (it.value().name == name)
      return it.index();
  }
  return std::nullopt;
}

// Try to map the TypeVar to a predefined or an argument type.
static std::optional<std::string>
findTypeValue(StringRef typeVar, SmallVectorImpl<LinalgOperandDef> &args) {
  // Handle all predefined types.
  if (typeVar == "I32")
    return std::string("helper.getIntegerType(32)");
  if (typeVar == "I64")
    return std::string("helper.getIntegerType(64)");
  if (typeVar == "F32")
    return std::string("helper.getFloat32Type()");
  if (typeVar == "F64")
    return std::string("helper.getFloat64Type()");

  // Search all argument types.
  for (const auto &it : llvm::enumerate(args)) {
    if (it.value().kind != LinalgOperandDefKind::InputTensor &&
        it.value().kind != LinalgOperandDefKind::Scalar &&
        it.value().kind != LinalgOperandDefKind::OutputTensor)
      continue;
    if (*it.value().typeVar == typeVar)
      return llvm::formatv("block.getArgument({0}).getType()", it.index())
          .str();
  }

  return std::nullopt;
}

static ScalarAssign *findAssignment(StringRef name,
                                    std::vector<ScalarAssign> &assignments) {
  for (auto &assign : assignments) {
    if (assign.arg == name)
      return &assign;
  }
  return nullptr;
}

// Return true if the operand is a function attribute.
static bool isFunctionAttribute(LinalgOperandDefKind kind) {
  return kind == LinalgOperandDefKind::UnaryFnAttr ||
         kind == LinalgOperandDefKind::BinaryFnAttr ||
         kind == LinalgOperandDefKind::TernaryFnAttr ||
         kind == LinalgOperandDefKind::TypeFnAttr;
}

// Return true if the operand is an attribute.
static bool isAttribute(LinalgOperandDefKind kind) {
  return kind == LinalgOperandDefKind::IndexAttr || isFunctionAttribute(kind);
}

// Get the enum name for the given operand kind.
std::string convertOperandKindToEnumName(LinalgOperandDefKind kind) {
  switch (kind) {
  case LinalgOperandDefKind::UnaryFnAttr:
    return std::string("UnaryFn");
  case LinalgOperandDefKind::BinaryFnAttr:
    return std::string("BinaryFn");
  case LinalgOperandDefKind::TernaryFnAttr:
    return std::string("TernaryFn");
  case LinalgOperandDefKind::TypeFnAttr:
    return std::string("TypeFn");
  default:
    break;
  }
  llvm_unreachable("unsupported function attribute kind");
}

// Get the enum name for the given function kind.
std::string convertFunctionKindToEnumName(ScalarFnKind kind) {
  switch (kind) {
  case ScalarFnKind::Unary:
    return std::string("UnaryFn");
  case ScalarFnKind::Binary:
    return std::string("BinaryFn");
  case ScalarFnKind::Ternary:
    return std::string("TernaryFn");
  case ScalarFnKind::Type:
    return std::string("TypeFn");
  }
  llvm_unreachable("unsupported function kind");
}

//===----------------------------------------------------------------------===//
// Templates
//===----------------------------------------------------------------------===//

// A single line banner format. Parameters:
// {0}: Single line comment
static const char bannerFormat[] = R"FMT(
//===----------------------------------------------------------------------===//
// {0}
//===----------------------------------------------------------------------===//
)FMT";

//===----------------------------------------------------------------------===//
// Named generic op generation.
// These ops map at most a single contraction that complies with the limitations
// of a linalg.generic.
//===----------------------------------------------------------------------===//

// Template for Linalg named ops' ODS definitions. Parameters:
// {0}: ODS/C++ op name
// {1}: assembly op mnemonic
// {2}: op interface list
// {3}: documentation (summary + description)
// {4}: op attribute list
// {5}: builder methods taking standalone attribute parameters
// {6}: additional method defintions
// {7}: additional methods for attributes used by indexing maps
static const char structuredOpOdsHeaderFormat[] = R"FMT(
//===----------------------------------------------------------------------===//
// Op definition for {0}
//===----------------------------------------------------------------------===//

def {0} : LinalgStructuredBase_Op<"{1}", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[{2}])> {
    {3}
    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs{4}
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{{}">:$attributes),
      [{{
        buildStructuredOp($_builder, $_state, std::nullopt, inputs, outputs,
          attributes, {0}::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{{}">:$attributes),
      [{{
        buildStructuredOp($_builder, $_state, resultTensorTypes,
          inputs, outputs, attributes, {0}::getRegionBuilder());
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{{}">:$attributes),
      [{{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>
      {5}
    ];
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    {6}

    let extraClassDeclaration = structuredOpsBaseDecls # [{{
      // Auto-generated.
      SmallVector<utils::IteratorType> getIteratorTypesArray();
      ArrayAttr getIndexingMaps();
      static void regionBuilder(ImplicitLocOpBuilder &b,
                                Block &block, ArrayRef<NamedAttribute> attrs);
      static std::function<void(ImplicitLocOpBuilder &,
                                Block &, ArrayRef<NamedAttribute>)>
      getRegionBuilder() {{
        return regionBuilder;
      }

      ::mlir::MutableOperandRange getDpsInitsMutable() {{
        return getOutputsMutable();
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      {7}
    }];
}
)FMT";

// Builder method taking attribute parameters. Parameters:
// {0}: Class name
// {1}: Comma interleaved attribute parameters
// {2}: Attribute initialization
static const char structuredOpBuilderFormat[] = R"FMT(
  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, {1},
       CArg<"ArrayRef<NamedAttribute>", "{{}">:$attributes),
  [{{
    {2}
    buildStructuredOp($_builder, $_state, resultTensorTypes, inputs, outputs,
      attributes, {0}::getRegionBuilder());
  }]>
)FMT";

// The getIteratorTypesArray() method for structured ops. Parameters:
// {0}: Class name
// {1}: Comma interleaved iterator type names.
static const char structuredOpIteratorTypesFormat[] =
    R"FMT(
SmallVector<utils::IteratorType> {0}::getIteratorTypesArray() {{
  return SmallVector<utils::IteratorType>{{ {1} };
}
)FMT";

// The getIteratorTypesArray() method for rank polymorphic structured ops.
// Parameters:
// {0}: Class name
static const char rankPolyStructuredOpIteratorTypesFormat[] =
    R"FMT(
SmallVector<utils::IteratorType> {0}::getIteratorTypesArray() {{
  int64_t rank = getRank(getDpsInitOperand(0));
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}
)FMT";

// The indexing_maps() method for structured ops. Parameters:
// {0}: Class name
// {1}: Comma-separated list of dimension variable names.
// {2}: Statements
static const char structuredOpIndexingMapsFormat[] = R"FMT(
ArrayAttr {0}::getIndexingMaps() {{
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  {1}
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}
)FMT";

// The indexing_maps() method for rank polymorphic structured ops. Parameters:
// {0}: Class name
static const char rankPolyStructuredOpIndexingMapsFormat[] = R"FMT(
ArrayAttr {0}::getIndexingMaps() {{
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    indexingMaps.push_back(getRank(&opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}
)FMT";

// Implementations of fold, getEffects and getSpeculatability.
// Parameters:
// {0}: Class name
const char structuredOpFoldersFormat[] = R"FMT(
LogicalResult {0}::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {{
  return memref::foldMemRefCast(*this);
}
void {0}::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {{
      if (hasPureTensorSemantics()) return;
      getGenericEffectsImpl(effects, cast<LinalgOp>(getOperation()));
}
Speculation::Speculatability {0}::getSpeculatability() {{
  return getGenericSpeculatabilityImpl(cast<LinalgOp>(getOperation()));
}
)FMT";

// Implementation of parse/print.
// Parameters:
// {0}: Class name
static const char structuredOpParserFormat[] = R"FMT(
ParseResult {0}::parse(OpAsmParser &parser, OperationState &result) {{
  return ::parseNamedStructuredOp(parser, result,
    {0}::getNumRegionArgs(), {0}::getRegionBuilder());
}
void {0}::print(OpAsmPrinter &p) {{
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}
)FMT";

static LogicalResult generateNamedGenericOpOds(LinalgOpConfig &opConfig,
                                               GenerationContext &genContext) {
  if (!genContext.shouldGenerateOds())
    return success();

  raw_ostream &os = genContext.odss();

  std::string interfaceNameList;
  std::string attrList;
  std::string attrMethods;
  std::string attrBuilder;

  std::string doc;
  if (opConfig.metadata->doc) {
    static const char structuredOpDocFmt[] = R"FMT(
  let summary = [{{{0}}];
  let description = [{{{1}}];
)FMT";
    StringRef summary, description;
    std::tie(summary, description) =
        StringRef(*opConfig.metadata->doc).trim().split("\n\n");

    doc = llvm::formatv(structuredOpDocFmt, summary.trim(), description.trim());
  }

  interfaceNameList = interleaveToString(opConfig.metadata->implements, ", ");

  std::string definitionList;
  for (const std::string &definition : opConfig.metadata->defines) {
    static const char definitionFmt[] = "let {0} = 1;\n";
    definitionList.append(llvm::formatv(definitionFmt, definition));
  }

  if (llvm::any_of(opConfig.structuredOp->args, [](LinalgOperandDef &arg) {
        return isAttribute(arg.kind);
      })) {
    SmallVector<std::string> attrDefs;
    SmallVector<std::string> attrParams;
    SmallVector<std::string> attrStmts;
    for (LinalgOperandDef &arg : opConfig.structuredOp->args) {
      static const char paramFmt[] = "\"Attribute\":${0}";
      static const char stmtFmt[] = "$_state.addAttribute(\"{0}\", {0});";
      // Add the type conversion attributes to the op definition and builders.
      if (isFunctionAttribute(arg.kind)) {
        assert(arg.defaultFn);
        std::string enumName = convertOperandKindToEnumName(arg.kind);
        static const char typeFmt[] = "{0}::{1}";
        static const char defFmt[] =
            "DefaultValuedOptionalAttr<{0}, \"{1}\">:${2}";
        attrDefs.push_back(llvm::formatv(
            defFmt, llvm::formatv("{0}Attr", enumName),
            llvm::formatv(typeFmt, enumName, arg.defaultFn), arg.name));
        attrParams.push_back(llvm::formatv(paramFmt, arg.name));
        attrStmts.push_back(llvm::formatv(stmtFmt, arg.name));
      }
      // Add the index attributes to the op definition and builders.
      if (arg.kind == LinalgOperandDefKind::IndexAttr) {
        assert(arg.indexAttrMap.has_value());
        assert(arg.defaultIndices.has_value());
        size_t size = arg.indexAttrMap->affineMap().getNumResults();
        assert(arg.defaultIndices->size() == size);
        static const char typeFmt[] = "RankedI64ElementsAttr<[{0}]>";
        static const char defFmt[] =
            "DefaultValuedOptionalAttr<{0}, \"{ {1} }\">:${2}";
        std::string defaultVals;
        llvm::raw_string_ostream ss(defaultVals);
        llvm::interleave(
            *arg.defaultIndices, ss,
            [&](int64_t val) { ss << "static_cast<int64_t>(" << val << ")"; },
            ", ");
        attrDefs.push_back(llvm::formatv(defFmt, llvm::formatv(typeFmt, size),
                                         ss.str(), arg.name));
        attrParams.push_back(llvm::formatv(paramFmt, arg.name));
        attrStmts.push_back(llvm::formatv(stmtFmt, arg.name));
      }
    }
    if (llvm::any_of(opConfig.structuredOp->args, [](LinalgOperandDef &arg) {
          return arg.kind == LinalgOperandDefKind::IndexAttr;
        })) {
      attrMethods = R"(
        bool hasDynamicIndexingMaps();
        LogicalResult verifyIndexingMapRequiredAttributes();
      )";
    }
    attrList = ",\n" + llvm::join(attrDefs, ",\n");
    attrBuilder = llvm::formatv(
        structuredOpBuilderFormat, opConfig.metadata->cppClassName,
        llvm::join(attrParams, ", "), llvm::join(attrStmts, "\n"));
  }

  os << llvm::formatv(structuredOpOdsHeaderFormat,
                      opConfig.metadata->cppClassName, opConfig.metadata->name,
                      interfaceNameList, doc, attrList, attrBuilder,
                      definitionList, attrMethods);

  return success();
}

static LogicalResult
generateNamedGenericOpDefns(LinalgOpConfig &opConfig,
                            GenerationContext &genContext) {
  if (!genContext.shouldGenerateDefns())
    return success();

  raw_ostream &os = genContext.defns();
  StringRef className = opConfig.metadata->cppClassName;

  // Implementation banner.
  std::string bannerComment = llvm::formatv("Implementation of {0}", className);
  os << llvm::formatv(bannerFormat, bannerComment);

  // Compute the number of scalar and tensor arguments.
  int64_t numOfArgs =
      llvm::count_if(opConfig.structuredOp->args, [](LinalgOperandDef &arg) {
        return arg.kind == LinalgOperandDefKind::InputTensor ||
               arg.kind == LinalgOperandDefKind::Scalar ||
               arg.kind == LinalgOperandDefKind::OutputTensor;
      });

  // An operation that accesses only scalars and scalar/rank zero tensors is
  // rank polymorhpic. We implement rank polymorphism by generating different
  // indexing maps and iterators that match the rank of the first output tensor.
  // An operation is rank polymorphic if the iteration domain has rank zero.
  bool isRankPolymorphic = opConfig.structuredOp->iteratorTypes.empty();

  // Generate the iterator_types() method.
  if (!isRankPolymorphic) {
    std::string iteratorsStr;
    llvm::raw_string_ostream ss(iteratorsStr);
    llvm::interleaveComma(opConfig.structuredOp->iteratorTypes, ss,
                          [&](LinalgIteratorTypeDef it) {
                            switch (it) {
                            case LinalgIteratorTypeDef::parallel:
                              ss << "utils::IteratorType::parallel";
                              break;
                            case LinalgIteratorTypeDef::reduction:
                              ss << "utils::IteratorType::reduction";
                              break;
                            }
                          });
    os << llvm::formatv(structuredOpIteratorTypesFormat, className,
                        iteratorsStr);
  } else {
    os << llvm::formatv(rankPolyStructuredOpIteratorTypesFormat, className);
  }

  // Generating the getIndexingMaps() method.
  if (auto &staticMaps =
          opConfig.structuredOp->indexingMaps.staticIndexingMaps) {
    if (staticMaps->empty())
      return emitError(genContext.getLoc()) << "op has no indexing maps";
    if (!isRankPolymorphic) {
      AffineMap firstMap = staticMaps->front().affineMap();

      // Symbol bindings.
      {
        // For each symbol, generate a declaration for it, either with an
        // AffineSymbolExpr or an AffineConstantExpr (if the symbol derives from
        // an attribute).
        // TODO: Possibly lift into a top-level method.
        static const char structuredOpSymbolBindingsFormat[] = R"FMT(
static SmallVector<AffineExpr> getSymbolBindings({0} self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
{1}
  return exprs;
}
)FMT";

        unsigned symbolCount = firstMap.getNumSymbols();
        SmallVector<std::string> symbolBindings;
        for (unsigned i = 0; i < symbolCount; ++i) {
          symbolBindings.push_back(llvm::formatv(
              "  exprs.push_back(getAffineSymbolExpr({0}, context));", i));
        }

        // Access an index attribute. Parameters:
        // {0}: Attribute name
        // {1}: Symbol position
        // {2}: Attribute index
        static const char structuredOpAccessAttrFormat[] = R"FMT(
int64_t cst{1} = self.get{0}().getValues<int64_t>()[{2}];
exprs.push_back(getAffineConstantExpr(cst{1}, context));
)FMT";
        // Update all symbol bindings mapped to an attribute.
        for (LinalgOperandDef &arg : opConfig.structuredOp->args) {
          if (arg.kind != LinalgOperandDefKind::IndexAttr)
            continue;
          assert(arg.indexAttrMap);
          for (auto [idx, result] :
               llvm::enumerate(arg.indexAttrMap->affineMap().getResults())) {
            if (auto symbol = dyn_cast<AffineSymbolExpr>(result)) {
              std::string argName = arg.name;
              argName[0] = toupper(argName[0]);
              symbolBindings[symbol.getPosition()] =
                  llvm::formatv(structuredOpAccessAttrFormat, argName,
                                symbol.getPosition(), idx);
            }
          }
        }

        std::string symbolBindingsStr;
        llvm::raw_string_ostream symbolBindingsSs(symbolBindingsStr);
        llvm::interleave(symbolBindings, symbolBindingsSs, "\n");

        os << llvm::formatv(structuredOpSymbolBindingsFormat, className,
                            symbolBindingsStr);
      }

      // Indexing maps.
      {
        unsigned dimCount = firstMap.getNumDims();

        // Generate a comma-separated list of dim identifiers to be passed to
        // bindDims, ensuring tht AffineExpr identifiers are bound in the right
        // order to the proper AffineDimExpr.
        // This results in vars in scope like: d0, d1, d2...
        SmallVector<unsigned> dimIndices;
        for (unsigned i = 0; i < dimCount; ++i)
          dimIndices.push_back(i);
        std::string dimIdentsStr;
        llvm::raw_string_ostream dimIdentsSs(dimIdentsStr);
        llvm::interleaveComma(dimIndices, dimIdentsSs,
                              [&](unsigned i) { dimIdentsSs << "d" << i; });

        // Statements to add and simplify each affine map.
        SmallVector<std::string> stmts;
        for (auto &indexingMap : *staticMaps) {
          // TODO: Assert that dim and symbol count match the first.
          stmts.push_back(
              llvm::formatv("maps.push_back({0});",
                            generateCppExpression(indexingMap, "context")));
          stmts.push_back(llvm::formatv(
              "maps.back() = "
              "simplifyAffineMap(maps.back().replaceDimsAndSymbols({{}, "
              "symbolBindings, {0}, 0));",
              dimCount));
        }

        // TODO: This needs to be memoized and/or converted to non-parser based
        // C++ codegen prior to real use.
        os << llvm::formatv(structuredOpIndexingMapsFormat, className,
                            interleaveToString(stmts, "\n  "));
      }
    } else {
      os << llvm::formatv(rankPolyStructuredOpIndexingMapsFormat, className);
    }
  } else {
    return emitError(genContext.getLoc())
           << "generating code for non static indexing maps not currently "
              "supported";
  }

  // getNumRegionArgs()
  {
    // Generates a getNumRegionArgs() method. Parameters:
    // {0}: Class name
    // {1}: Number of region args
    static const char structuredOpGetNumRegionArgsFormat[] = R"FMT(
unsigned {0}::getNumRegionArgs() {{ return {1}; }
)FMT";
    os << llvm::formatv(structuredOpGetNumRegionArgsFormat, className,
                        numOfArgs);
  }

  // getLibraryCallName()
  {
    // Generates a getLibraryCallName method. Parameters:
    // {0}: Class name
    static const char structuredOpGetLibraryCallFormat[] = R"FMT(
std::string {0}::getLibraryCallName() {{
  return generateLibraryCallName(getOperation());
}
)FMT";
    os << llvm::formatv(structuredOpGetLibraryCallFormat, className);
  }

  // hasDynamicIndexingMaps() and verifyIndexingMapRequiredAttributes()
  if (llvm::any_of(opConfig.structuredOp->args, [](LinalgOperandDef &arg) {
        return arg.kind == LinalgOperandDefKind::IndexAttr;
      })) {
    std::vector<std::string> attrVerifications;
    for (LinalgOperandDef &arg : opConfig.structuredOp->args) {
      if (arg.kind != LinalgOperandDefKind::IndexAttr)
        continue;
      assert(arg.indexAttrMap);
      // Verify index attribute. Paramters:
      // {0}: Attribute name
      // {1}: Attribute size
      static const char attrFmt[] = R"FMT(
if (auto attr = op->getAttrOfType<DenseElementsAttr>("{0}")) {{
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute '{0}'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{{ {1} })
    return op->emitError("incorrect shape for index attribute '{0}'");
}
)FMT";
      attrVerifications.push_back(llvm::formatv(
          attrFmt, arg.name, arg.indexAttrMap->affineMap().getNumResults()));
    }

    // Generates the verifyIndexingMapRequiredAttributes method. Parameters:
    // {0}: Class name
    // {1}: Attribute verification
    static const char structuredOpVerifyIndexingMapRequiredAttributes[] = R"FMT(
bool {0}::hasDynamicIndexingMaps() {{ return true; }
LogicalResult {0}::verifyIndexingMapRequiredAttributes() {{
  Operation *op = getOperation();
  {1}
  return success();
}
)FMT";
    os << llvm::formatv(structuredOpVerifyIndexingMapRequiredAttributes,
                        className, llvm::join(attrVerifications, "\n"));
  }

  // regionBuilder()
  {
    // Generates a regionBuilder method. Parameters.
    // {0}: Class name
    // {1}: Number of args
    // {2}: Attributes
    // {3}: Statements
    static const char structuredOpRegionBuilderFormat[] = R"FMT(
void {0}::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {{
  assert({1} > 0 && block.getNumArguments() == {1} &&
         "{0} regionBuilder expects {1} (>=0) args");
  RegionBuilderHelper helper(b, block);
  SmallVector<Value> yields;
  {2}
  {3}
  helper.yieldOutputs(yields);
}
)FMT";
    auto &args = opConfig.structuredOp->args;
    auto &assignments = opConfig.structuredOp->assignments;
    size_t generatedAssignmentCount = 0;
    int localCounter = 0;
    SmallVector<std::string> attrs;
    SmallVector<std::string> stmts;
    for (LinalgOperandDef &arg : args) {
      if (!isFunctionAttribute(arg.kind))
        continue;
      // Obtain the type function attribute values. Parameters.
      // {0}: enum name
      // {1}: attribute name
      // {2}: default type function name
      static const char attrDef[] = R"FMT(
  {0} {1}Val = {0}::{2};
  auto {1}Iter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {{
                                return attr.getName() == "{1}"; });
  if ({1}Iter != attrs.end()) {{
    if (auto attr = llvm::dyn_cast<{0}Attr>({1}Iter->getValue()))
      {1}Val = attr.getValue();
  }
)FMT";
      std::string enumName = convertOperandKindToEnumName(arg.kind);
      attrs.push_back(
          llvm::formatv(attrDef, enumName, arg.name, arg.defaultFn));
    }
    for (LinalgOperandDef &arg : args) {
      if (arg.kind != LinalgOperandDefKind::OutputTensor)
        continue;

      // Find the assignment that correlates with the argument.
      ScalarAssign *assignment = findAssignment(arg.name, assignments);
      if (!assignment)
        return emitError(genContext.getLoc())
               << "no assignment found for output argument " << arg.name;
      ++generatedAssignmentCount;

      // Recursively generate the expression.
      std::function<std::optional<std::string>(ScalarExpression &)>
          generateExpression =
              [&](ScalarExpression &expression) -> std::optional<std::string> {
        if (expression.arg) {
          // Argument reference.
          std::optional<int> argIndex =
              findTensorDefArgIndex(*expression.arg, args);
          if (!argIndex) {
            emitError(genContext.getLoc())
                << "scalar argument not defined on the op: " << *expression.arg;
            return std::nullopt;
          }
          return std::string(
              llvm::formatv("block.getArgument({0})", *argIndex));
        }
        if (expression.constant) {
          std::string cppIdent = llvm::formatv("value{0}", ++localCounter);
          stmts.push_back(
              llvm::formatv(R"FMT(Value {0} = helper.constant("{1}");)FMT",
                            cppIdent, expression.constant));
          return cppIdent;
        }
        if (expression.index) {
          // Access an iteration index.
          std::string cppIdent = llvm::formatv("value{0}", ++localCounter);
          stmts.push_back(llvm::formatv("Value {0} = helper.index({1});",
                                        cppIdent, *expression.index));
          return cppIdent;
        }
        if (expression.scalarFn) {
          std::string enumName =
              convertFunctionKindToEnumName(expression.scalarFn->kind);

          // Get the function or attribute name.
          assert(expression.scalarFn->fnName || expression.scalarFn->attrName);
          std::string funcType;
          if (expression.scalarFn->fnName) {
            funcType = llvm::formatv("{0}::{1}", enumName,
                                     *expression.scalarFn->fnName);
          }
          if (expression.scalarFn->attrName) {
            if (llvm::none_of(args, [&](LinalgOperandDef &arg) {
                  return isFunctionAttribute(arg.kind) &&
                         arg.name == *expression.scalarFn->attrName;
                })) {
              emitError(genContext.getLoc()) << "missing function attribute "
                                             << *expression.scalarFn->attrName;
            }
            funcType = llvm::formatv("{0}Val", *expression.scalarFn->attrName);
          }
          assert(!funcType.empty());

          // Add the optional type parameter to the operands.
          SmallVector<std::string> operandCppValues;
          if (expression.scalarFn->kind == ScalarFnKind::Type) {
            assert(expression.scalarFn->typeVar.has_value());
            std::optional<std::string> typeCppValue =
                findTypeValue(*expression.scalarFn->typeVar, args);
            if (!typeCppValue) {
              emitError(genContext.getLoc())
                  << "type variable " << *expression.scalarFn->typeVar
                  << ", used in a type conversion, must map to a predefined or "
                  << "an argument type but it does not";
              return std::nullopt;
            }
            operandCppValues.push_back(*typeCppValue);
          }

          // Collect the scalar operands.
          for (ScalarExpression &operand : expression.scalarFn->operands) {
            auto operandCppValue = generateExpression(operand);
            if (!operandCppValue)
              return std::nullopt;
            operandCppValues.push_back(*operandCppValue);
          }

          // Call the function builder.
          std::string cppIdent = llvm::formatv("value{0}", ++localCounter);
          stmts.push_back(llvm::formatv(
              "Value {0} = helper.build{1}({2}, {3});", cppIdent, enumName,
              funcType, interleaveToString(operandCppValues, ", ")));
          return cppIdent;
        }
        emitError(genContext.getLoc()) << "unknown ScalarExpression type";
        return std::nullopt;
      };
      std::optional<std::string> cppValue =
          generateExpression(assignment->value);
      if (!cppValue)
        return failure();
      stmts.push_back(llvm::formatv("yields.push_back({0});", *cppValue));
    }

    if (generatedAssignmentCount != assignments.size())
      return emitError(genContext.getLoc())
             << "mismatched number of assignments vs output arguments";

    os << llvm::formatv(structuredOpRegionBuilderFormat, className, numOfArgs,
                        interleaveToString(attrs, "\n  "),
                        interleaveToString(stmts, "\n  "));
  }

  // Parser and printer.
  os << llvm::formatv(structuredOpParserFormat, className);

  // Canonicalizers and folders.
  os << llvm::formatv(structuredOpFoldersFormat, className);

  return success();
}

static LogicalResult generateOp(LinalgOpConfig &opConfig,
                                GenerationContext &genContext) {
  // Switch on op type being generated.
  if (opConfig.structuredOp) {
    return success(
        succeeded(generateNamedGenericOpOds(opConfig, genContext)) &&
        succeeded(generateNamedGenericOpDefns(opConfig, genContext)));
  }
  return emitError(genContext.getLoc()) << "unsupported operation type";
}

//===----------------------------------------------------------------------===//
// Command line options and main
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("YAML filename"));

static llvm::cl::opt<std::string>
    outputOdsDeclFilename("o-ods-decl", llvm::cl::desc("ODS output filename"),
                          llvm::cl::value_desc("filename"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    outputCppImplFilename("o-impl",
                          llvm::cl::desc("C++ implementation file name"),
                          llvm::cl::value_desc("filename"), llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Linalg ODS Gen from YAML");

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  MLIRContext mlirContext;
  LinalgYAMLContext yamlContext{&mlirContext};

  std::vector<LinalgOpConfig> opConfigs;

  // Parse input.
  Input yin(file->getBuffer(), &yamlContext);
  yin >> opConfigs;

  if (yin.error())
    return 1;

  // Open output files.
  std::unique_ptr<llvm::ToolOutputFile> outputOdsDecl;
  if (!outputOdsDeclFilename.empty()) {
    outputOdsDecl = openOutputFile(outputOdsDeclFilename, &errorMessage);
    if (!outputOdsDecl) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  }

  std::unique_ptr<llvm::ToolOutputFile> outputCppImpl;
  if (!outputCppImplFilename.empty()) {
    outputCppImpl = openOutputFile(outputCppImplFilename, &errorMessage);
    if (!outputCppImpl) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  }

  if (!outputOdsDecl && !outputCppImpl) {
    llvm::errs() << "error: No output files specified\n";
    return 1;
  }

  // Generate.
  GenerationContext genContext(&mlirContext,
                               outputOdsDecl ? &outputOdsDecl->os() : nullptr,
                               outputCppImpl ? &outputCppImpl->os() : nullptr);

  for (auto &opConfig : opConfigs) {
    if (!opConfig.metadata) {
      emitError(genContext.getLoc())
          << "missing operation metadata on subsequent op";
      return 1;
    }

    genContext.setLoc(NameLoc::get(
        StringAttr::get(&mlirContext, opConfig.metadata->cppClassName)));
    if (failed(generateOp(opConfig, genContext))) {
      return 1;
    }
  }

  if (outputOdsDecl)
    outputOdsDecl->keep();
  if (outputCppImpl)
    outputCppImpl->keep();

  return 0;
}
