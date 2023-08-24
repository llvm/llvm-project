//===- OpDefinitionsGen.cpp - MLIR op definitions generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDefinitionsGen uses the description of operations to generate C++
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "OpClass.h"
#include "OpFormatGen.h"
#include "OpGenHelpers.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Property.h"
#include "mlir/TableGen/SideEffects.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-opdefgen"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

static const char *const tblgenNamePrefix = "tblgen_";
static const char *const generatedArgName = "odsArg";
static const char *const odsBuilder = "odsBuilder";
static const char *const builderOpState = "odsState";
static const char *const propertyStorage = "propStorage";
static const char *const propertyValue = "propValue";
static const char *const propertyAttr = "propAttr";
static const char *const propertyDiag = "propDiag";

/// The names of the implicit attributes that contain variadic operand and
/// result segment sizes.
static const char *const operandSegmentAttrName = "operandSegmentSizes";
static const char *const resultSegmentAttrName = "resultSegmentSizes";

/// Code for an Op to lookup an attribute. Uses cached identifiers and subrange
/// lookup.
///
/// {0}: Code snippet to get the attribute's name or identifier.
/// {1}: The lower bound on the sorted subrange.
/// {2}: The upper bound on the sorted subrange.
/// {3}: Code snippet to get the array of named attributes.
/// {4}: "Named" to get the named attribute.
static const char *const subrangeGetAttr =
    "::mlir::impl::get{4}AttrFromSortedRange({3}.begin() + {1}, {3}.end() - "
    "{2}, {0})";

/// The logic to calculate the actual value range for a declared operand/result
/// of an op with variadic operands/results. Note that this logic is not for
/// general use; it assumes all variadic operands/results must have the same
/// number of values.
///
/// {0}: The list of whether each declared operand/result is variadic.
/// {1}: The total number of non-variadic operands/results.
/// {2}: The total number of variadic operands/results.
/// {3}: The total number of actual values.
/// {4}: "operand" or "result".
static const char *const sameVariadicSizeValueRangeCalcCode = R"(
  bool isVariadic[] = {{{0}};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic {4} corresponds to.
  // This assumes all static variadic {4}s have the same dynamic value count.
  int variadicSize = ({3} - {1}) / {2};
  // `index` passed in as the parameter is the static index which counts each
  // {4} (variadic or not) as size 1. So here for each previous static variadic
  // {4}, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static {4} starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {{start, size};
)";

/// The logic to calculate the actual value range for a declared operand/result
/// of an op with variadic operands/results. Note that this logic is assumes
/// the op has an attribute specifying the size of each operand/result segment
/// (variadic or not).
static const char *const attrSizedSegmentValueRangeCalcCode = R"(
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
)";
/// The code snippet to initialize the sizes for the value range calculation.
///
/// {0}: The code to get the attribute.
static const char *const adapterSegmentSizeAttrInitCode = R"(
  assert({0} && "missing segment size attribute for op");
  auto sizeAttr = ::llvm::cast<::mlir::DenseI32ArrayAttr>({0});
)";
static const char *const adapterSegmentSizeAttrInitCodeProperties = R"(
  ::llvm::ArrayRef<int32_t> sizeAttr = {0};
)";

/// The code snippet to initialize the sizes for the value range calculation.
///
/// {0}: The code to get the attribute.
static const char *const opSegmentSizeAttrInitCode = R"(
  auto sizeAttr = ::llvm::cast<::mlir::DenseI32ArrayAttr>({0});
)";

/// The logic to calculate the actual value range for a declared operand
/// of an op with variadic of variadic operands within the OpAdaptor.
///
/// {0}: The name of the segment attribute.
/// {1}: The index of the main operand.
/// {2}: The range type of adaptor.
static const char *const variadicOfVariadicAdaptorCalcCode = R"(
  auto tblgenTmpOperands = getODSOperands({1});
  auto sizes = {0}();

  ::llvm::SmallVector<{2}> tblgenTmpOperandGroups;
  for (int i = 0, e = sizes.size(); i < e; ++i) {{
    tblgenTmpOperandGroups.push_back(tblgenTmpOperands.take_front(sizes[i]));
    tblgenTmpOperands = tblgenTmpOperands.drop_front(sizes[i]);
  }
  return tblgenTmpOperandGroups;
)";

/// The logic to build a range of either operand or result values.
///
/// {0}: The begin iterator of the actual values.
/// {1}: The call to generate the start and length of the value range.
static const char *const valueRangeReturnCode = R"(
  auto valueRange = {1};
  return {{std::next({0}, valueRange.first),
           std::next({0}, valueRange.first + valueRange.second)};
)";

/// Read operand/result segment_size from bytecode.
static const char *const readBytecodeSegmentSize = R"(
if ($_reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  ::mlir::DenseI32ArrayAttr attr;
  if (::mlir::failed($_reader.readAttribute(attr))) return ::mlir::failure();
  if (attr.size() > static_cast<int64_t>(sizeof($_storage) / sizeof(int32_t))) {
    $_reader.emitError("size mismatch for operand/result_segment_size");
    return ::mlir::failure();
  }
  ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), $_storage.begin());
} else {
  return $_reader.readSparseArray(::llvm::MutableArrayRef($_storage));
}
)";

/// Write operand/result segment_size to bytecode.
static const char *const writeBytecodeSegmentSize = R"(
if ($_writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6)
  $_writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(getContext(), $_storage));
else
  $_writer.writeSparseArray(::llvm::ArrayRef($_storage));
)";

/// A header for indicating code sections.
///
/// {0}: Some text, or a class name.
/// {1}: Some text.
static const char *const opCommentHeader = R"(
//===----------------------------------------------------------------------===//
// {0} {1}
//===----------------------------------------------------------------------===//

)";

//===----------------------------------------------------------------------===//
// Utility structs and functions
//===----------------------------------------------------------------------===//

// Replaces all occurrences of `match` in `str` with `substitute`.
static std::string replaceAllSubstrs(std::string str, const std::string &match,
                                     const std::string &substitute) {
  std::string::size_type scanLoc = 0, matchLoc = std::string::npos;
  while ((matchLoc = str.find(match, scanLoc)) != std::string::npos) {
    str = str.replace(matchLoc, match.size(), substitute);
    scanLoc = matchLoc + substitute.size();
  }
  return str;
}

// Returns whether the record has a value of the given name that can be returned
// via getValueAsString.
static inline bool hasStringAttribute(const Record &record,
                                      StringRef fieldName) {
  auto *valueInit = record.getValueInit(fieldName);
  return isa<StringInit>(valueInit);
}

static std::string getArgumentName(const Operator &op, int index) {
  const auto &operand = op.getOperand(index);
  if (!operand.name.empty())
    return std::string(operand.name);
  return std::string(formatv("{0}_{1}", generatedArgName, index));
}

// Returns true if we can use unwrapped value for the given `attr` in builders.
static bool canUseUnwrappedRawValue(const tblgen::Attribute &attr) {
  return attr.getReturnType() != attr.getStorageType() &&
         // We need to wrap the raw value into an attribute in the builder impl
         // so we need to make sure that the attribute specifies how to do that.
         !attr.getConstBuilderTemplate().empty();
}

/// Build an attribute from a parameter value using the constant builder.
static std::string constBuildAttrFromParam(const tblgen::Attribute &attr,
                                           FmtContext &fctx,
                                           StringRef paramName) {
  std::string builderTemplate = attr.getConstBuilderTemplate().str();

  // For StringAttr, its constant builder call will wrap the input in
  // quotes, which is correct for normal string literals, but incorrect
  // here given we use function arguments. So we need to strip the
  // wrapping quotes.
  if (StringRef(builderTemplate).contains("\"$0\""))
    builderTemplate = replaceAllSubstrs(builderTemplate, "\"$0\"", "$0");

  return tgfmt(builderTemplate, &fctx, paramName).str();
}

namespace {
/// Metadata on a registered attribute. Given that attributes are stored in
/// sorted order on operations, we can use information from ODS to deduce the
/// number of required attributes less and and greater than each attribute,
/// allowing us to search only a subrange of the attributes in ODS-generated
/// getters.
struct AttributeMetadata {
  /// The attribute name.
  StringRef attrName;
  /// Whether the attribute is required.
  bool isRequired;
  /// The ODS attribute constraint. Not present for implicit attributes.
  std::optional<Attribute> constraint;
  /// The number of required attributes less than this attribute.
  unsigned lowerBound = 0;
  /// The number of required attributes greater than this attribute.
  unsigned upperBound = 0;
};

/// Helper class to select between OpAdaptor and Op code templates.
class OpOrAdaptorHelper {
public:
  OpOrAdaptorHelper(const Operator &op, bool emitForOp)
      : op(op), emitForOp(emitForOp) {
    computeAttrMetadata();
  }

  /// Object that wraps a functor in a stream operator for interop with
  /// llvm::formatv.
  class Formatter {
  public:
    template <typename Functor>
    Formatter(Functor &&func) : func(std::forward<Functor>(func)) {}

    std::string str() const {
      std::string result;
      llvm::raw_string_ostream os(result);
      os << *this;
      return os.str();
    }

  private:
    std::function<raw_ostream &(raw_ostream &)> func;

    friend raw_ostream &operator<<(raw_ostream &os, const Formatter &fmt) {
      return fmt.func(os);
    }
  };

  // Generate code for getting an attribute.
  Formatter getAttr(StringRef attrName, bool isNamed = false) const {
    assert(attrMetadata.count(attrName) && "expected attribute metadata");
    return [this, attrName, isNamed](raw_ostream &os) -> raw_ostream & {
      const AttributeMetadata &attr = attrMetadata.find(attrName)->second;
      if (hasProperties()) {
        assert(!isNamed);
        return os << "getProperties()." << attrName;
      }
      return os << formatv(subrangeGetAttr, getAttrName(attrName),
                           attr.lowerBound, attr.upperBound, getAttrRange(),
                           isNamed ? "Named" : "");
    };
  }

  // Generate code for getting the name of an attribute.
  Formatter getAttrName(StringRef attrName) const {
    return [this, attrName](raw_ostream &os) -> raw_ostream & {
      if (emitForOp)
        return os << op.getGetterName(attrName) << "AttrName()";
      return os << formatv("{0}::{1}AttrName(*odsOpName)", op.getCppClassName(),
                           op.getGetterName(attrName));
    };
  }

  // Get the code snippet for getting the named attribute range.
  StringRef getAttrRange() const {
    return emitForOp ? "(*this)->getAttrs()" : "odsAttrs";
  }

  // Get the prefix code for emitting an error.
  Formatter emitErrorPrefix() const {
    return [this](raw_ostream &os) -> raw_ostream & {
      if (emitForOp)
        return os << "emitOpError(";
      return os << formatv("emitError(loc, \"'{0}' op \"",
                           op.getOperationName());
    };
  }

  // Get the call to get an operand or segment of operands.
  Formatter getOperand(unsigned index) const {
    return [this, index](raw_ostream &os) -> raw_ostream & {
      return os << formatv(op.getOperand(index).isVariadic()
                               ? "this->getODSOperands({0})"
                               : "(*this->getODSOperands({0}).begin())",
                           index);
    };
  }

  // Get the call to get a result of segment of results.
  Formatter getResult(unsigned index) const {
    return [this, index](raw_ostream &os) -> raw_ostream & {
      if (!emitForOp)
        return os << "<no results should be generated>";
      return os << formatv(op.getResult(index).isVariadic()
                               ? "this->getODSResults({0})"
                               : "(*this->getODSResults({0}).begin())",
                           index);
    };
  }

  // Return whether an op instance is available.
  bool isEmittingForOp() const { return emitForOp; }

  // Return the ODS operation wrapper.
  const Operator &getOp() const { return op; }

  // Get the attribute metadata sorted by name.
  const llvm::MapVector<StringRef, AttributeMetadata> &getAttrMetadata() const {
    return attrMetadata;
  }

  /// Returns whether to emit a `Properties` struct for this operation or not.
  bool hasProperties() const {
    if (!op.getProperties().empty())
      return true;
    if (!op.getDialect().usePropertiesForAttributes())
      return false;
    if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments") ||
        op.getTrait("::mlir::OpTrait::AttrSizedResultSegments"))
      return true;
    return llvm::any_of(getAttrMetadata(),
                        [](const std::pair<StringRef, AttributeMetadata> &it) {
                          return !it.second.constraint ||
                                 !it.second.constraint->isDerivedAttr();
                        });
  }

  std::optional<NamedProperty> &getOperandSegmentsSize() {
    return operandSegmentsSize;
  }

  std::optional<NamedProperty> &getResultSegmentsSize() {
    return resultSegmentsSize;
  }

private:
  // Compute the attribute metadata.
  void computeAttrMetadata();

  // The operation ODS wrapper.
  const Operator &op;
  // True if code is being generate for an op. False for an adaptor.
  const bool emitForOp;

  // The attribute metadata, mapped by name.
  llvm::MapVector<StringRef, AttributeMetadata> attrMetadata;

  // Property
  std::optional<NamedProperty> operandSegmentsSize;
  std::string operandSegmentsSizeStorage;
  std::optional<NamedProperty> resultSegmentsSize;
  std::string resultSegmentsSizeStorage;

  // The number of required attributes.
  unsigned numRequired;
};

} // namespace

void OpOrAdaptorHelper::computeAttrMetadata() {
  // Enumerate the attribute names of this op, ensuring the attribute names are
  // unique in case implicit attributes are explicitly registered.
  for (const NamedAttribute &namedAttr : op.getAttributes()) {
    Attribute attr = namedAttr.attr;
    bool isOptional =
        attr.hasDefaultValue() || attr.isOptional() || attr.isDerivedAttr();
    attrMetadata.insert(
        {namedAttr.name, AttributeMetadata{namedAttr.name, !isOptional, attr}});
  }

  auto makeProperty = [&](StringRef storageType) {
    return Property(
        /*storageType=*/storageType,
        /*interfaceType=*/"::llvm::ArrayRef<int32_t>",
        /*convertFromStorageCall=*/"$_storage",
        /*assignToStorageCall=*/
        "::llvm::copy($_value, $_storage.begin())",
        /*convertToAttributeCall=*/
        "::mlir::DenseI32ArrayAttr::get($_ctxt, $_storage)",
        /*convertFromAttributeCall=*/
        "return convertFromAttribute($_storage, $_attr, $_diag);",
        /*readFromMlirBytecodeCall=*/readBytecodeSegmentSize,
        /*writeToMlirBytecodeCall=*/writeBytecodeSegmentSize,
        /*hashPropertyCall=*/
        "::llvm::hash_combine_range(std::begin($_storage), "
        "std::end($_storage));",
        /*StringRef defaultValue=*/"");
  };
  // Include key attributes from several traits as implicitly registered.
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    if (op.getDialect().usePropertiesForAttributes()) {
      operandSegmentsSizeStorage =
          llvm::formatv("std::array<int32_t, {0}>", op.getNumOperands());
      operandSegmentsSize = {"operandSegmentSizes",
                             makeProperty(operandSegmentsSizeStorage)};
    } else {
      attrMetadata.insert(
          {operandSegmentAttrName, AttributeMetadata{operandSegmentAttrName,
                                                     /*isRequired=*/true,
                                                     /*attr=*/std::nullopt}});
    }
  }
  if (op.getTrait("::mlir::OpTrait::AttrSizedResultSegments")) {
    if (op.getDialect().usePropertiesForAttributes()) {
      resultSegmentsSizeStorage =
          llvm::formatv("std::array<int32_t, {0}>", op.getNumResults());
      resultSegmentsSize = {"resultSegmentSizes",
                            makeProperty(resultSegmentsSizeStorage)};
    } else {
      attrMetadata.insert(
          {resultSegmentAttrName,
           AttributeMetadata{resultSegmentAttrName, /*isRequired=*/true,
                             /*attr=*/std::nullopt}});
    }
  }

  // Store the metadata in sorted order.
  SmallVector<AttributeMetadata> sortedAttrMetadata =
      llvm::to_vector(llvm::make_second_range(attrMetadata.takeVector()));
  llvm::sort(sortedAttrMetadata,
             [](const AttributeMetadata &lhs, const AttributeMetadata &rhs) {
               return lhs.attrName < rhs.attrName;
             });

  // Compute the subrange bounds for each attribute.
  numRequired = 0;
  for (AttributeMetadata &attr : sortedAttrMetadata) {
    attr.lowerBound = numRequired;
    numRequired += attr.isRequired;
  };
  for (AttributeMetadata &attr : sortedAttrMetadata)
    attr.upperBound = numRequired - attr.lowerBound - attr.isRequired;

  // Store the results back into the map.
  for (const AttributeMetadata &attr : sortedAttrMetadata)
    attrMetadata.insert({attr.attrName, attr});
}

//===----------------------------------------------------------------------===//
// Op emitter
//===----------------------------------------------------------------------===//

namespace {
// Helper class to emit a record into the given output stream.
class OpEmitter {
  using ConstArgument =
      llvm::PointerUnion<const AttributeMetadata *, const NamedProperty *>;

public:
  static void
  emitDecl(const Operator &op, raw_ostream &os,
           const StaticVerifierFunctionEmitter &staticVerifierEmitter);
  static void
  emitDef(const Operator &op, raw_ostream &os,
          const StaticVerifierFunctionEmitter &staticVerifierEmitter);

private:
  OpEmitter(const Operator &op,
            const StaticVerifierFunctionEmitter &staticVerifierEmitter);

  void emitDecl(raw_ostream &os);
  void emitDef(raw_ostream &os);

  // Generate methods for accessing the attribute names of this operation.
  void genAttrNameGetters();

  // Generates the OpAsmOpInterface for this operation if possible.
  void genOpAsmInterface();

  // Generates the `getOperationName` method for this op.
  void genOpNameGetter();

  // Generates code to manage the properties, if any!
  void genPropertiesSupport();

  // Generates code to manage the encoding of properties to bytecode.
  void
  genPropertiesSupportForBytecode(ArrayRef<ConstArgument> attrOrProperties);

  // Generates getters for the attributes.
  void genAttrGetters();

  // Generates setter for the attributes.
  void genAttrSetters();

  // Generates removers for optional attributes.
  void genOptionalAttrRemovers();

  // Generates getters for named operands.
  void genNamedOperandGetters();

  // Generates setters for named operands.
  void genNamedOperandSetters();

  // Generates getters for named results.
  void genNamedResultGetters();

  // Generates getters for named regions.
  void genNamedRegionGetters();

  // Generates getters for named successors.
  void genNamedSuccessorGetters();

  // Generates the method to populate default attributes.
  void genPopulateDefaultAttributes();

  // Generates builder methods for the operation.
  void genBuilder();

  // Generates the build() method that takes each operand/attribute
  // as a stand-alone parameter.
  void genSeparateArgParamBuilder();

  // Generates the build() method that takes each operand/attribute as a
  // stand-alone parameter. The generated build() method uses first operand's
  // type as all results' types.
  void genUseOperandAsResultTypeSeparateParamBuilder();

  // Generates the build() method that takes all operands/attributes
  // collectively as one parameter. The generated build() method uses first
  // operand's type as all results' types.
  void genUseOperandAsResultTypeCollectiveParamBuilder();

  // Generates the build() method that takes aggregate operands/attributes
  // parameters. This build() method uses inferred types as result types.
  // Requires: The type needs to be inferable via InferTypeOpInterface.
  void genInferredTypeCollectiveParamBuilder();

  // Generates the build() method that takes each operand/attribute as a
  // stand-alone parameter. The generated build() method uses first attribute's
  // type as all result's types.
  void genUseAttrAsResultTypeBuilder();

  // Generates the build() method that takes all result types collectively as
  // one parameter. Similarly for operands and attributes.
  void genCollectiveParamBuilder();

  // The kind of parameter to generate for result types in builders.
  enum class TypeParamKind {
    None,       // No result type in parameter list.
    Separate,   // A separate parameter for each result type.
    Collective, // An ArrayRef<Type> for all result types.
  };

  // The kind of parameter to generate for attributes in builders.
  enum class AttrParamKind {
    WrappedAttr,    // A wrapped MLIR Attribute instance.
    UnwrappedValue, // A raw value without MLIR Attribute wrapper.
  };

  // Builds the parameter list for build() method of this op. This method writes
  // to `paramList` the comma-separated parameter list and updates
  // `resultTypeNames` with the names for parameters for specifying result
  // types. `inferredAttributes` is populated with any attributes that are
  // elided from the build list. The given `typeParamKind` and `attrParamKind`
  // controls how result types and attributes are placed in the parameter list.
  void buildParamList(SmallVectorImpl<MethodParameter> &paramList,
                      llvm::StringSet<> &inferredAttributes,
                      SmallVectorImpl<std::string> &resultTypeNames,
                      TypeParamKind typeParamKind,
                      AttrParamKind attrParamKind = AttrParamKind::WrappedAttr);

  // Adds op arguments and regions into operation state for build() methods.
  void
  genCodeForAddingArgAndRegionForBuilder(MethodBody &body,
                                         llvm::StringSet<> &inferredAttributes,
                                         bool isRawValueAttr = false);

  // Generates canonicalizer declaration for the operation.
  void genCanonicalizerDecls();

  // Generates the folder declaration for the operation.
  void genFolderDecls();

  // Generates the parser for the operation.
  void genParser();

  // Generates the printer for the operation.
  void genPrinter();

  // Generates verify method for the operation.
  void genVerifier();

  // Generates custom verify methods for the operation.
  void genCustomVerifier();

  // Generates verify statements for operands and results in the operation.
  // The generated code will be attached to `body`.
  void genOperandResultVerifier(MethodBody &body,
                                Operator::const_value_range values,
                                StringRef valueKind);

  // Generates verify statements for regions in the operation.
  // The generated code will be attached to `body`.
  void genRegionVerifier(MethodBody &body);

  // Generates verify statements for successors in the operation.
  // The generated code will be attached to `body`.
  void genSuccessorVerifier(MethodBody &body);

  // Generates the traits used by the object.
  void genTraits();

  // Generate the OpInterface methods for all interfaces.
  void genOpInterfaceMethods();

  // Generate op interface methods for the given interface.
  void genOpInterfaceMethods(const tblgen::InterfaceTrait *trait);

  // Generate op interface method for the given interface method. If
  // 'declaration' is true, generates a declaration, else a definition.
  Method *genOpInterfaceMethod(const tblgen::InterfaceMethod &method,
                               bool declaration = true);

  // Generate the side effect interface methods.
  void genSideEffectInterfaceMethods();

  // Generate the type inference interface methods.
  void genTypeInterfaceMethods();

private:
  // The TableGen record for this op.
  // TODO: OpEmitter should not have a Record directly,
  // it should rather go through the Operator for better abstraction.
  const Record &def;

  // The wrapper operator class for querying information from this op.
  const Operator &op;

  // The C++ code builder for this op
  OpClass opClass;

  // The format context for verification code generation.
  FmtContext verifyCtx;

  // The emitter containing all of the locally emitted verification functions.
  const StaticVerifierFunctionEmitter &staticVerifierEmitter;

  // Helper for emitting op code.
  OpOrAdaptorHelper emitHelper;
};

} // namespace

// Populate the format context `ctx` with substitutions of attributes, operands
// and results.
static void populateSubstitutions(const OpOrAdaptorHelper &emitHelper,
                                  FmtContext &ctx) {
  // Populate substitutions for attributes.
  auto &op = emitHelper.getOp();
  for (const auto &namedAttr : op.getAttributes())
    ctx.addSubst(namedAttr.name,
                 emitHelper.getOp().getGetterName(namedAttr.name) + "()");

  // Populate substitutions for named operands.
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    auto &value = op.getOperand(i);
    if (!value.name.empty())
      ctx.addSubst(value.name, emitHelper.getOperand(i).str());
  }

  // Populate substitutions for results.
  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    auto &value = op.getResult(i);
    if (!value.name.empty())
      ctx.addSubst(value.name, emitHelper.getResult(i).str());
  }
}

/// Generate verification on native traits requiring attributes.
static void genNativeTraitAttrVerifier(MethodBody &body,
                                       const OpOrAdaptorHelper &emitHelper) {
  // Check that the variadic segment sizes attribute exists and contains the
  // expected number of elements.
  //
  // {0}: Attribute name.
  // {1}: Expected number of elements.
  // {2}: "operand" or "result".
  // {3}: Emit error prefix.
  const char *const checkAttrSizedValueSegmentsCode = R"(
  {
    auto sizeAttr = ::llvm::cast<::mlir::DenseI32ArrayAttr>(tblgen_{0});
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != {1})
      return {3}"'{0}' attribute for specifying {2} segments must have {1} "
                "elements, but got ") << numElements;
  }
  )";

  // Verify a few traits first so that we can use getODSOperands() and
  // getODSResults() in the rest of the verifier.
  auto &op = emitHelper.getOp();
  if (!op.getDialect().usePropertiesForAttributes()) {
    if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
      body << formatv(checkAttrSizedValueSegmentsCode, operandSegmentAttrName,
                      op.getNumOperands(), "operand",
                      emitHelper.emitErrorPrefix());
    }
    if (op.getTrait("::mlir::OpTrait::AttrSizedResultSegments")) {
      body << formatv(checkAttrSizedValueSegmentsCode, resultSegmentAttrName,
                      op.getNumResults(), "result",
                      emitHelper.emitErrorPrefix());
    }
  }
}

// Return true if a verifier can be emitted for the attribute: it is not a
// derived attribute, it has a predicate, its condition is not empty, and, for
// adaptors, the condition does not reference the op.
static bool canEmitAttrVerifier(Attribute attr, bool isEmittingForOp) {
  if (attr.isDerivedAttr())
    return false;
  Pred pred = attr.getPredicate();
  if (pred.isNull())
    return false;
  std::string condition = pred.getCondition();
  return !condition.empty() &&
         (!StringRef(condition).contains("$_op") || isEmittingForOp);
}

// Generate attribute verification. If an op instance is not available, then
// attribute checks that require one will not be emitted.
//
// Attribute verification is performed as follows:
//
// 1. Verify that all required attributes are present in sorted order. This
// ensures that we can use subrange lookup even with potentially missing
// attributes.
// 2. Verify native trait attributes so that other attributes may call methods
// that depend on the validity of these attributes, e.g. segment size attributes
// and operand or result getters.
// 3. Verify the constraints on all present attributes.
static void
genAttributeVerifier(const OpOrAdaptorHelper &emitHelper, FmtContext &ctx,
                     MethodBody &body,
                     const StaticVerifierFunctionEmitter &staticVerifierEmitter,
                     bool useProperties) {
  if (emitHelper.getAttrMetadata().empty())
    return;

  // Verify the attribute if it is present. This assumes that default values
  // are valid. This code snippet pastes the condition inline.
  //
  // TODO: verify the default value is valid (perhaps in debug mode only).
  //
  // {0}: Attribute variable name.
  // {1}: Attribute condition code.
  // {2}: Emit error prefix.
  // {3}: Attribute name.
  // {4}: Attribute/constraint description.
  const char *const verifyAttrInline = R"(
  if ({0} && !({1}))
    return {2}"attribute '{3}' failed to satisfy constraint: {4}");
)";
  // Verify the attribute using a uniqued constraint. Can only be used within
  // the context of an op.
  //
  // {0}: Unique constraint name.
  // {1}: Attribute variable name.
  // {2}: Attribute name.
  const char *const verifyAttrUnique = R"(
  if (::mlir::failed({0}(*this, {1}, "{2}")))
    return ::mlir::failure();
)";

  // Traverse the array until the required attribute is found. Return an error
  // if the traversal reached the end.
  //
  // {0}: Code to get the name of the attribute.
  // {1}: The emit error prefix.
  // {2}: The name of the attribute.
  const char *const findRequiredAttr = R"(
while (true) {{
  if (namedAttrIt == namedAttrRange.end())
    return {1}"requires attribute '{2}'");
  if (namedAttrIt->getName() == {0}) {{
    tblgen_{2} = namedAttrIt->getValue();
    break;
  })";

  // Emit a check to see if the iteration has encountered an optional attribute.
  //
  // {0}: Code to get the name of the attribute.
  // {1}: The name of the attribute.
  const char *const checkOptionalAttr = R"(
  else if (namedAttrIt->getName() == {0}) {{
    tblgen_{1} = namedAttrIt->getValue();
  })";

  // Emit the start of the loop for checking trailing attributes.
  const char *const checkTrailingAttrs = R"(while (true) {
  if (namedAttrIt == namedAttrRange.end()) {
    break;
  })";

  // Emit the verifier for the attribute.
  const auto emitVerifier = [&](Attribute attr, StringRef attrName,
                                StringRef varName) {
    std::string condition = attr.getPredicate().getCondition();

    std::optional<StringRef> constraintFn;
    if (emitHelper.isEmittingForOp() &&
        (constraintFn = staticVerifierEmitter.getAttrConstraintFn(attr))) {
      body << formatv(verifyAttrUnique, *constraintFn, varName, attrName);
    } else {
      body << formatv(verifyAttrInline, varName,
                      tgfmt(condition, &ctx.withSelf(varName)),
                      emitHelper.emitErrorPrefix(), attrName,
                      escapeString(attr.getSummary()));
    }
  };

  // Prefix variables with `tblgen_` to avoid hiding the attribute accessor.
  const auto getVarName = [&](StringRef attrName) {
    return (tblgenNamePrefix + attrName).str();
  };

  body.indent();
  if (useProperties) {
    for (const std::pair<StringRef, AttributeMetadata> &it :
         emitHelper.getAttrMetadata()) {
      const AttributeMetadata &metadata = it.second;
      if (metadata.constraint && metadata.constraint->isDerivedAttr())
        continue;
      body << formatv(
          "auto tblgen_{0} = getProperties().{0}; (void)tblgen_{0};\n",
          it.first);
      if (metadata.isRequired)
        body << formatv(
            "if (!tblgen_{0}) return {1}\"requires attribute '{0}'\");\n",
            it.first, emitHelper.emitErrorPrefix());
    }
  } else {
    body << formatv("auto namedAttrRange = {0};\n", emitHelper.getAttrRange());
    body << "auto namedAttrIt = namedAttrRange.begin();\n";

    // Iterate over the attributes in sorted order. Keep track of the optional
    // attributes that may be encountered along the way.
    SmallVector<const AttributeMetadata *> optionalAttrs;

    for (const std::pair<StringRef, AttributeMetadata> &it :
         emitHelper.getAttrMetadata()) {
      const AttributeMetadata &metadata = it.second;
      if (!metadata.isRequired) {
        optionalAttrs.push_back(&metadata);
        continue;
      }

      body << formatv("::mlir::Attribute {0};\n", getVarName(it.first));
      for (const AttributeMetadata *optional : optionalAttrs) {
        body << formatv("::mlir::Attribute {0};\n",
                        getVarName(optional->attrName));
      }
      body << formatv(findRequiredAttr, emitHelper.getAttrName(it.first),
                      emitHelper.emitErrorPrefix(), it.first);
      for (const AttributeMetadata *optional : optionalAttrs) {
        body << formatv(checkOptionalAttr,
                        emitHelper.getAttrName(optional->attrName),
                        optional->attrName);
      }
      body << "\n  ++namedAttrIt;\n}\n";
      optionalAttrs.clear();
    }
    // Get trailing optional attributes.
    if (!optionalAttrs.empty()) {
      for (const AttributeMetadata *optional : optionalAttrs) {
        body << formatv("::mlir::Attribute {0};\n",
                        getVarName(optional->attrName));
      }
      body << checkTrailingAttrs;
      for (const AttributeMetadata *optional : optionalAttrs) {
        body << formatv(checkOptionalAttr,
                        emitHelper.getAttrName(optional->attrName),
                        optional->attrName);
      }
      body << "\n  ++namedAttrIt;\n}\n";
    }
  }
  body.unindent();

  // Emit the checks for segment attributes first so that the other
  // constraints can call operand and result getters.
  genNativeTraitAttrVerifier(body, emitHelper);

  bool isEmittingForOp = emitHelper.isEmittingForOp();
  for (const auto &namedAttr : emitHelper.getOp().getAttributes())
    if (canEmitAttrVerifier(namedAttr.attr, isEmittingForOp))
      emitVerifier(namedAttr.attr, namedAttr.name, getVarName(namedAttr.name));
}

/// Include declarations specified on NativeTrait
static std::string formatExtraDeclarations(const Operator &op) {
  SmallVector<StringRef> extraDeclarations;
  // Include extra class declarations from NativeTrait
  for (const auto &trait : op.getTraits()) {
    if (auto *opTrait = dyn_cast<tblgen::NativeTrait>(&trait)) {
      StringRef value = opTrait->getExtraConcreteClassDeclaration();
      if (value.empty())
        continue;
      extraDeclarations.push_back(value);
    }
  }
  extraDeclarations.push_back(op.getExtraClassDeclaration());
  return llvm::join(extraDeclarations, "\n");
}

/// Op extra class definitions have a `$cppClass` substitution that is to be
/// replaced by the C++ class name.
/// Include declarations specified on NativeTrait
static std::string formatExtraDefinitions(const Operator &op) {
  SmallVector<StringRef> extraDefinitions;
  // Include extra class definitions from NativeTrait
  for (const auto &trait : op.getTraits()) {
    if (auto *opTrait = dyn_cast<tblgen::NativeTrait>(&trait)) {
      StringRef value = opTrait->getExtraConcreteClassDefinition();
      if (value.empty())
        continue;
      extraDefinitions.push_back(value);
    }
  }
  extraDefinitions.push_back(op.getExtraClassDefinition());
  FmtContext ctx = FmtContext().addSubst("cppClass", op.getCppClassName());
  return tgfmt(llvm::join(extraDefinitions, "\n"), &ctx).str();
}

OpEmitter::OpEmitter(const Operator &op,
                     const StaticVerifierFunctionEmitter &staticVerifierEmitter)
    : def(op.getDef()), op(op),
      opClass(op.getCppClassName(), formatExtraDeclarations(op),
              formatExtraDefinitions(op)),
      staticVerifierEmitter(staticVerifierEmitter),
      emitHelper(op, /*emitForOp=*/true) {
  verifyCtx.addSubst("_op", "(*this->getOperation())");
  verifyCtx.addSubst("_ctxt", "this->getOperation()->getContext()");

  genTraits();

  // Generate C++ code for various op methods. The order here determines the
  // methods in the generated file.
  genAttrNameGetters();
  genOpAsmInterface();
  genOpNameGetter();
  genNamedOperandGetters();
  genNamedOperandSetters();
  genNamedResultGetters();
  genNamedRegionGetters();
  genNamedSuccessorGetters();
  genPropertiesSupport();
  genAttrGetters();
  genAttrSetters();
  genOptionalAttrRemovers();
  genBuilder();
  genPopulateDefaultAttributes();
  genParser();
  genPrinter();
  genVerifier();
  genCustomVerifier();
  genCanonicalizerDecls();
  genFolderDecls();
  genTypeInterfaceMethods();
  genOpInterfaceMethods();
  generateOpFormat(op, opClass);
  genSideEffectInterfaceMethods();
}
void OpEmitter::emitDecl(
    const Operator &op, raw_ostream &os,
    const StaticVerifierFunctionEmitter &staticVerifierEmitter) {
  OpEmitter(op, staticVerifierEmitter).emitDecl(os);
}

void OpEmitter::emitDef(
    const Operator &op, raw_ostream &os,
    const StaticVerifierFunctionEmitter &staticVerifierEmitter) {
  OpEmitter(op, staticVerifierEmitter).emitDef(os);
}

void OpEmitter::emitDecl(raw_ostream &os) {
  opClass.finalize();
  opClass.writeDeclTo(os);
}

void OpEmitter::emitDef(raw_ostream &os) {
  opClass.finalize();
  opClass.writeDefTo(os);
}

static void errorIfPruned(size_t line, Method *m, const Twine &methodName,
                          const Operator &op) {
  if (m)
    return;
  PrintFatalError(op.getLoc(), "Unexpected overlap when generating `" +
                                   methodName + "` for " +
                                   op.getOperationName() + " (from line " +
                                   Twine(line) + ")");
}

#define ERROR_IF_PRUNED(M, N, O) errorIfPruned(__LINE__, M, N, O)

void OpEmitter::genAttrNameGetters() {
  const llvm::MapVector<StringRef, AttributeMetadata> &attributes =
      emitHelper.getAttrMetadata();
  bool hasOperandSegmentsSize =
      op.getDialect().usePropertiesForAttributes() &&
      op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments");
  // Emit the getAttributeNames method.
  {
    auto *method = opClass.addStaticInlineMethod(
        "::llvm::ArrayRef<::llvm::StringRef>", "getAttributeNames");
    ERROR_IF_PRUNED(method, "getAttributeNames", op);
    auto &body = method->body();
    if (!hasOperandSegmentsSize && attributes.empty()) {
      body << "  return {};";
      // Nothing else to do if there are no registered attributes. Exit early.
      return;
    }
    body << "  static ::llvm::StringRef attrNames[] = {";
    llvm::interleaveComma(llvm::make_first_range(attributes), body,
                          [&](StringRef attrName) {
                            body << "::llvm::StringRef(\"" << attrName << "\")";
                          });
    if (hasOperandSegmentsSize) {
      if (!attributes.empty())
        body << ", ";
      body << "::llvm::StringRef(\"" << operandSegmentAttrName << "\")";
    }
    body << "};\n  return ::llvm::ArrayRef(attrNames);";
  }

  // Emit the getAttributeNameForIndex methods.
  {
    auto *method = opClass.addInlineMethod<Method::Private>(
        "::mlir::StringAttr", "getAttributeNameForIndex",
        MethodParameter("unsigned", "index"));
    ERROR_IF_PRUNED(method, "getAttributeNameForIndex", op);
    method->body()
        << "  return getAttributeNameForIndex((*this)->getName(), index);";
  }
  {
    auto *method = opClass.addStaticInlineMethod<Method::Private>(
        "::mlir::StringAttr", "getAttributeNameForIndex",
        MethodParameter("::mlir::OperationName", "name"),
        MethodParameter("unsigned", "index"));
    ERROR_IF_PRUNED(method, "getAttributeNameForIndex", op);

    if (attributes.empty()) {
      method->body() << "  return {};";
    } else {
      const char *const getAttrName = R"(
  assert(index < {0} && "invalid attribute index");
  assert(name.getStringRef() == getOperationName() && "invalid operation name");
  return name.getAttributeNames()[index];
)";
      method->body() << formatv(getAttrName, attributes.size());
    }
  }

  // Generate the <attr>AttrName methods, that expose the attribute names to
  // users.
  const char *attrNameMethodBody = "  return getAttributeNameForIndex({0});";
  for (auto [index, attr] :
       llvm::enumerate(llvm::make_first_range(attributes))) {
    std::string name = op.getGetterName(attr);
    std::string methodName = name + "AttrName";

    // Generate the non-static variant.
    {
      auto *method = opClass.addInlineMethod("::mlir::StringAttr", methodName);
      ERROR_IF_PRUNED(method, methodName, op);
      method->body() << llvm::formatv(attrNameMethodBody, index);
    }

    // Generate the static variant.
    {
      auto *method = opClass.addStaticInlineMethod(
          "::mlir::StringAttr", methodName,
          MethodParameter("::mlir::OperationName", "name"));
      ERROR_IF_PRUNED(method, methodName, op);
      method->body() << llvm::formatv(attrNameMethodBody,
                                      "name, " + Twine(index));
    }
  }
  if (hasOperandSegmentsSize) {
    std::string name = op.getGetterName(operandSegmentAttrName);
    std::string methodName = name + "AttrName";
    // Generate the non-static variant.
    {
      auto *method = opClass.addInlineMethod("::mlir::StringAttr", methodName);
      ERROR_IF_PRUNED(method, methodName, op);
      method->body()
          << " return (*this)->getName().getAttributeNames().back();";
    }

    // Generate the static variant.
    {
      auto *method = opClass.addStaticInlineMethod(
          "::mlir::StringAttr", methodName,
          MethodParameter("::mlir::OperationName", "name"));
      ERROR_IF_PRUNED(method, methodName, op);
      method->body() << " return name.getAttributeNames().back();";
    }
  }
}

// Emit the getter for an attribute with the return type specified.
// It is templated to be shared between the Op and the adaptor class.
template <typename OpClassOrAdaptor>
static void emitAttrGetterWithReturnType(FmtContext &fctx,
                                         OpClassOrAdaptor &opClass,
                                         const Operator &op, StringRef name,
                                         Attribute attr) {
  auto *method = opClass.addMethod(attr.getReturnType(), name);
  ERROR_IF_PRUNED(method, name, op);
  auto &body = method->body();
  body << "  auto attr = " << name << "Attr();\n";
  if (attr.hasDefaultValue() && attr.isOptional()) {
    // Returns the default value if not set.
    // TODO: this is inefficient, we are recreating the attribute for every
    // call. This should be set instead.
    if (!attr.isConstBuildable()) {
      PrintFatalError("DefaultValuedAttr of type " + attr.getAttrDefName() +
                      " must have a constBuilder");
    }
    std::string defaultValue = std::string(
        tgfmt(attr.getConstBuilderTemplate(), &fctx, attr.getDefaultValue()));
    body << "    if (!attr)\n      return "
         << tgfmt(attr.getConvertFromStorageCall(),
                  &fctx.withSelf(defaultValue))
         << ";\n";
  }
  body << "  return "
       << tgfmt(attr.getConvertFromStorageCall(), &fctx.withSelf("attr"))
       << ";\n";
}

void OpEmitter::genPropertiesSupport() {
  if (!emitHelper.hasProperties())
    return;

  SmallVector<ConstArgument> attrOrProperties;
  for (const std::pair<StringRef, AttributeMetadata> &it :
       emitHelper.getAttrMetadata()) {
    if (!it.second.constraint || !it.second.constraint->isDerivedAttr())
      attrOrProperties.push_back(&it.second);
  }
  for (const NamedProperty &prop : op.getProperties())
    attrOrProperties.push_back(&prop);
  if (emitHelper.getOperandSegmentsSize())
    attrOrProperties.push_back(&emitHelper.getOperandSegmentsSize().value());
  if (emitHelper.getResultSegmentsSize())
    attrOrProperties.push_back(&emitHelper.getResultSegmentsSize().value());
  if (attrOrProperties.empty())
    return;
  auto &setPropMethod =
      opClass
          .addStaticMethod(
              "::mlir::LogicalResult", "setPropertiesFromAttr",
              MethodParameter("Properties &", "prop"),
              MethodParameter("::mlir::Attribute", "attr"),
              MethodParameter("::mlir::InFlightDiagnostic *", "diag"))
          ->body();
  auto &getPropMethod =
      opClass
          .addStaticMethod("::mlir::Attribute", "getPropertiesAsAttr",
                           MethodParameter("::mlir::MLIRContext *", "ctx"),
                           MethodParameter("const Properties &", "prop"))
          ->body();
  auto &hashMethod =
      opClass
          .addStaticMethod("llvm::hash_code", "computePropertiesHash",
                           MethodParameter("const Properties &", "prop"))
          ->body();
  auto &getInherentAttrMethod =
      opClass
          .addStaticMethod("std::optional<mlir::Attribute>", "getInherentAttr",
                           MethodParameter("::mlir::MLIRContext *", "ctx"),
                           MethodParameter("const Properties &", "prop"),
                           MethodParameter("llvm::StringRef", "name"))
          ->body();
  auto &setInherentAttrMethod =
      opClass
          .addStaticMethod("void", "setInherentAttr",
                           MethodParameter("Properties &", "prop"),
                           MethodParameter("llvm::StringRef", "name"),
                           MethodParameter("mlir::Attribute", "value"))
          ->body();
  auto &populateInherentAttrsMethod =
      opClass
          .addStaticMethod("void", "populateInherentAttrs",
                           MethodParameter("::mlir::MLIRContext *", "ctx"),
                           MethodParameter("const Properties &", "prop"),
                           MethodParameter("::mlir::NamedAttrList &", "attrs"))
          ->body();
  auto &verifyInherentAttrsMethod =
      opClass
          .addStaticMethod(
              "::mlir::LogicalResult", "verifyInherentAttrs",
              MethodParameter("::mlir::OperationName", "opName"),
              MethodParameter("::mlir::NamedAttrList &", "attrs"),
              MethodParameter(
                  "llvm::function_ref<::mlir::InFlightDiagnostic()>",
                  "getDiag"))
          ->body();

  opClass.declare<UsingDeclaration>("Properties", "FoldAdaptor::Properties");

  // Convert the property to the attribute form.

  setPropMethod << R"decl(
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
    )decl";
  // TODO: properties might be optional as well.
  const char *propFromAttrFmt = R"decl(;
    {{
      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
                             ::mlir::InFlightDiagnostic *propDiag) {{
        {0};
      };
      {2};
      if (!attr) {{
        if (diag)
          *diag << "expected key entry for {1} in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      if (::mlir::failed(setFromAttr(prop.{1}, attr, diag)))
        return ::mlir::failure();
    }
)decl";

  for (const auto &attrOrProp : attrOrProperties) {
    if (const auto *namedProperty =
            llvm::dyn_cast_if_present<const NamedProperty *>(attrOrProp)) {
      StringRef name = namedProperty->name;
      auto &prop = namedProperty->prop;
      FmtContext fctx;

      std::string getAttr;
      llvm::raw_string_ostream os(getAttr);
      os << "   auto attr = dict.get(\"" << name << "\");";
      if (name == operandSegmentAttrName) {
        // Backward compat for now, TODO: Remove at some point.
        os << "   if (!attr) attr = dict.get(\"operand_segment_sizes\");";
      }
      if (name == resultSegmentAttrName) {
        // Backward compat for now, TODO: Remove at some point.
        os << "   if (!attr) attr = dict.get(\"result_segment_sizes\");";
      }
      os.flush();

      setPropMethod << formatv(propFromAttrFmt,
                               tgfmt(prop.getConvertFromAttributeCall(),
                                     &fctx.addSubst("_attr", propertyAttr)
                                          .addSubst("_storage", propertyStorage)
                                          .addSubst("_diag", propertyDiag)),
                               name, getAttr);

    } else {
      const auto *namedAttr =
          llvm::dyn_cast_if_present<const AttributeMetadata *>(attrOrProp);
      StringRef name = namedAttr->attrName;
      std::string getAttr;
      llvm::raw_string_ostream os(getAttr);
      os << "   auto attr = dict.get(\"" << name << "\");";
      if (name == operandSegmentAttrName) {
        // Backward compat for now
        os << "   if (!attr) attr = dict.get(\"operand_segment_sizes\");";
      }
      if (name == resultSegmentAttrName) {
        // Backward compat for now
        os << "   if (!attr) attr = dict.get(\"result_segment_sizes\");";
      }
      os.flush();

      setPropMethod << formatv(R"decl(
  {{
    auto &propStorage = prop.{0};
    {2}
    if (attr || /*isRequired=*/{1}) {{
      if (!attr) {{
        if (diag)
          *diag << "expected key entry for {0} in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {{
        propStorage = convertedAttr;
      } else {{
        if (diag)
          *diag << "Invalid attribute `{0}` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
)decl",
                               name, namedAttr->isRequired, getAttr);
    }
  }
  setPropMethod << "  return ::mlir::success();\n";

  // Convert the attribute form to the property.

  getPropMethod << "    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;\n"
                << "    ::mlir::Builder odsBuilder{ctx};\n";
  const char *propToAttrFmt = R"decl(
    {
      const auto &propStorage = prop.{0};
      attrs.push_back(odsBuilder.getNamedAttr("{0}",
                                              {1}));
    }
)decl";
  for (const auto &attrOrProp : attrOrProperties) {
    if (const auto *namedProperty =
            llvm::dyn_cast_if_present<const NamedProperty *>(attrOrProp)) {
      StringRef name = namedProperty->name;
      auto &prop = namedProperty->prop;
      FmtContext fctx;
      getPropMethod << formatv(
          propToAttrFmt, name,
          tgfmt(prop.getConvertToAttributeCall(),
                &fctx.addSubst("_ctxt", "ctx")
                     .addSubst("_storage", propertyStorage)));
      continue;
    }
    const auto *namedAttr =
        llvm::dyn_cast_if_present<const AttributeMetadata *>(attrOrProp);
    StringRef name = namedAttr->attrName;
    getPropMethod << formatv(R"decl(
    {{
      const auto &propStorage = prop.{0};
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("{0}",
                                       propStorage));
    }
)decl",
                             name);
  }
  getPropMethod << R"decl(
  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
)decl";

  // Hashing for the property

  const char *propHashFmt = R"decl(
  auto hash_{0} = [] (const auto &propStorage) -> llvm::hash_code {
    return {1};
  };
)decl";
  for (const auto &attrOrProp : attrOrProperties) {
    if (const auto *namedProperty =
            llvm::dyn_cast_if_present<const NamedProperty *>(attrOrProp)) {
      StringRef name = namedProperty->name;
      auto &prop = namedProperty->prop;
      FmtContext fctx;
      hashMethod << formatv(propHashFmt, name,
                            tgfmt(prop.getHashPropertyCall(),
                                  &fctx.addSubst("_storage", propertyStorage)));
    }
  }
  hashMethod << "  return llvm::hash_combine(";
  llvm::interleaveComma(
      attrOrProperties, hashMethod, [&](const ConstArgument &attrOrProp) {
        if (const auto *namedProperty =
                llvm::dyn_cast_if_present<const NamedProperty *>(attrOrProp)) {
          hashMethod << "\n    hash_" << namedProperty->name << "(prop."
                     << namedProperty->name << ")";
          return;
        }
        const auto *namedAttr =
            llvm::dyn_cast_if_present<const AttributeMetadata *>(attrOrProp);
        StringRef name = namedAttr->attrName;
        hashMethod << "\n    llvm::hash_value(prop." << name
                   << ".getAsOpaquePointer())";
      });
  hashMethod << ");\n";

  const char *getInherentAttrMethodFmt = R"decl(
    if (name == "{0}")
      return prop.{0};
)decl";
  const char *setInherentAttrMethodFmt = R"decl(
    if (name == "{0}") {{
       prop.{0} = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.{0})>>(value);
       return;
    }
)decl";
  const char *populateInherentAttrsMethodFmt = R"decl(
    if (prop.{0}) attrs.append("{0}", prop.{0});
)decl";
  for (const auto &attrOrProp : attrOrProperties) {
    if (const auto *namedAttr =
            llvm::dyn_cast_if_present<const AttributeMetadata *>(attrOrProp)) {
      StringRef name = namedAttr->attrName;
      getInherentAttrMethod << formatv(getInherentAttrMethodFmt, name);
      setInherentAttrMethod << formatv(setInherentAttrMethodFmt, name);
      populateInherentAttrsMethod
          << formatv(populateInherentAttrsMethodFmt, name);
      continue;
    }
    // The ODS segment size property is "special": we expose it as an attribute
    // even though it is a native property.
    const auto *namedProperty = cast<const NamedProperty *>(attrOrProp);
    StringRef name = namedProperty->name;
    if (name != operandSegmentAttrName && name != resultSegmentAttrName)
      continue;
    auto &prop = namedProperty->prop;
    FmtContext fctx;
    fctx.addSubst("_ctxt", "ctx");
    fctx.addSubst("_storage", Twine("prop.") + name);
    if (name == operandSegmentAttrName) {
      getInherentAttrMethod
          << formatv("    if (name == \"operand_segment_sizes\" || name == "
                     "\"{0}\") return ",
                     operandSegmentAttrName);
    } else {
      getInherentAttrMethod
          << formatv("    if (name == \"result_segment_sizes\" || name == "
                     "\"{0}\") return ",
                     resultSegmentAttrName);
    }
    getInherentAttrMethod << tgfmt(prop.getConvertToAttributeCall(), &fctx)
                          << ";\n";

    if (name == operandSegmentAttrName) {
      setInherentAttrMethod
          << formatv("        if (name == \"operand_segment_sizes\" || name == "
                     "\"{0}\") {{",
                     operandSegmentAttrName);
    } else {
      setInherentAttrMethod
          << formatv("        if (name == \"result_segment_sizes\" || name == "
                     "\"{0}\") {{",
                     resultSegmentAttrName);
    }
    setInherentAttrMethod << formatv(R"decl(
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.{0}) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.{0}.begin());
       return;
    }
)decl",
                                     name);
    if (name == operandSegmentAttrName) {
      populateInherentAttrsMethod
          << formatv("  attrs.append(\"{0}\", {1});\n", operandSegmentAttrName,
                     tgfmt(prop.getConvertToAttributeCall(), &fctx));
    } else {
      populateInherentAttrsMethod
          << formatv("  attrs.append(\"{0}\", {1});\n", resultSegmentAttrName,
                     tgfmt(prop.getConvertToAttributeCall(), &fctx));
    }
  }
  getInherentAttrMethod << "  return std::nullopt;\n";

  // Emit the verifiers method for backward compatibility with the generic
  // syntax. This method verifies the constraint on the properties attributes
  // before they are set, since dyn_cast<> will silently omit failures.
  for (const auto &attrOrProp : attrOrProperties) {
    const auto *namedAttr =
        llvm::dyn_cast_if_present<const AttributeMetadata *>(attrOrProp);
    if (!namedAttr || !namedAttr->constraint)
      continue;
    Attribute attr = *namedAttr->constraint;
    std::optional<StringRef> constraintFn =
        staticVerifierEmitter.getAttrConstraintFn(attr);
    if (!constraintFn)
      continue;
    if (canEmitAttrVerifier(attr,
                            /*isEmittingForOp=*/false)) {
      std::string name = op.getGetterName(namedAttr->attrName);
      verifyInherentAttrsMethod
          << formatv(R"(
    {{
      ::mlir::Attribute attr = attrs.get({0}AttrName(opName));
      if (attr && ::mlir::failed({1}(attr, "{2}", getDiag)))
        return ::mlir::failure();
    }
)",
                     name, constraintFn, namedAttr->attrName);
    }
  }
  verifyInherentAttrsMethod << "    return ::mlir::success();";

  // Generate methods to interact with bytecode.
  genPropertiesSupportForBytecode(attrOrProperties);
}

void OpEmitter::genPropertiesSupportForBytecode(
    ArrayRef<ConstArgument> attrOrProperties) {
  if (op.useCustomPropertiesEncoding()) {
    opClass.declareStaticMethod(
        "::mlir::LogicalResult", "readProperties",
        MethodParameter("::mlir::DialectBytecodeReader &", "reader"),
        MethodParameter("::mlir::OperationState &", "state"));
    opClass.declareMethod(
        "void", "writeProperties",
        MethodParameter("::mlir::DialectBytecodeWriter &", "writer"));
    return;
  }

  auto &readPropertiesMethod =
      opClass
          .addStaticMethod(
              "::mlir::LogicalResult", "readProperties",
              MethodParameter("::mlir::DialectBytecodeReader &", "reader"),
              MethodParameter("::mlir::OperationState &", "state"))
          ->body();

  auto &writePropertiesMethod =
      opClass
          .addMethod(
              "void", "writeProperties",
              MethodParameter("::mlir::DialectBytecodeWriter &", "writer"))
          ->body();

  // Populate bytecode serialization logic.
  readPropertiesMethod
      << "  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;";
  writePropertiesMethod << "  auto &prop = getProperties(); (void)prop;\n";
  for (const auto &attrOrProp : attrOrProperties) {
    if (const auto *namedProperty =
            attrOrProp.dyn_cast<const NamedProperty *>()) {
      StringRef name = namedProperty->name;
      FmtContext fctx;
      fctx.addSubst("_reader", "reader")
          .addSubst("_writer", "writer")
          .addSubst("_storage", propertyStorage)
          .addSubst("_ctxt", "this->getContext()");
      readPropertiesMethod << formatv(
          R"(
  {{
    auto &propStorage = prop.{0};
    auto readProp = [&]() {
      {1};
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
)",
          name,
          tgfmt(namedProperty->prop.getReadFromMlirBytecodeCall(), &fctx));
      writePropertiesMethod << formatv(
          R"(
  {{
    auto &propStorage = prop.{0};
    {1};
  }
)",
          name, tgfmt(namedProperty->prop.getWriteToMlirBytecodeCall(), &fctx));
      continue;
    }
    const auto *namedAttr = attrOrProp.dyn_cast<const AttributeMetadata *>();
    StringRef name = namedAttr->attrName;
    if (namedAttr->isRequired) {
      readPropertiesMethod << formatv(R"(
  if (::mlir::failed(reader.readAttribute(prop.{0})))
    return ::mlir::failure();
)",
                                      name);
      writePropertiesMethod
          << formatv("  writer.writeAttribute(prop.{0});\n", name);
    } else {
      readPropertiesMethod << formatv(R"(
  if (::mlir::failed(reader.readOptionalAttribute(prop.{0})))
    return ::mlir::failure();
)",
                                      name);
      writePropertiesMethod << formatv(R"(
  writer.writeOptionalAttribute(prop.{0});
)",
                                       name);
    }
  }
  readPropertiesMethod << "  return ::mlir::success();";
}

void OpEmitter::genAttrGetters() {
  FmtContext fctx;
  fctx.withBuilder("::mlir::Builder((*this)->getContext())");

  // Emit the derived attribute body.
  auto emitDerivedAttr = [&](StringRef name, Attribute attr) {
    if (auto *method = opClass.addMethod(attr.getReturnType(), name))
      method->body() << "  " << attr.getDerivedCodeBody() << "\n";
  };

  // Generate named accessor with Attribute return type. This is a wrapper
  // class that allows referring to the attributes via accessors instead of
  // having to use the string interface for better compile time verification.
  auto emitAttrWithStorageType = [&](StringRef name, StringRef attrName,
                                     Attribute attr) {
    auto *method = opClass.addMethod(attr.getStorageType(), name + "Attr");
    if (!method)
      return;
    method->body() << formatv(
        "  return ::llvm::{1}<{2}>({0});", emitHelper.getAttr(attrName),
        attr.isOptional() || attr.hasDefaultValue() ? "dyn_cast_or_null"
                                                    : "cast",
        attr.getStorageType());
  };

  for (const NamedAttribute &namedAttr : op.getAttributes()) {
    std::string name = op.getGetterName(namedAttr.name);
    if (namedAttr.attr.isDerivedAttr()) {
      emitDerivedAttr(name, namedAttr.attr);
    } else {
      emitAttrWithStorageType(name, namedAttr.name, namedAttr.attr);
      emitAttrGetterWithReturnType(fctx, opClass, op, name, namedAttr.attr);
    }
  }

  auto derivedAttrs = make_filter_range(op.getAttributes(),
                                        [](const NamedAttribute &namedAttr) {
                                          return namedAttr.attr.isDerivedAttr();
                                        });
  if (derivedAttrs.empty())
    return;

  opClass.addTrait("::mlir::DerivedAttributeOpInterface::Trait");
  // Generate helper method to query whether a named attribute is a derived
  // attribute. This enables, for example, avoiding adding an attribute that
  // overlaps with a derived attribute.
  {
    auto *method =
        opClass.addStaticMethod("bool", "isDerivedAttribute",
                                MethodParameter("::llvm::StringRef", "name"));
    ERROR_IF_PRUNED(method, "isDerivedAttribute", op);
    auto &body = method->body();
    for (auto namedAttr : derivedAttrs)
      body << "  if (name == \"" << namedAttr.name << "\") return true;\n";
    body << " return false;";
  }
  // Generate method to materialize derived attributes as a DictionaryAttr.
  {
    auto *method = opClass.addMethod("::mlir::DictionaryAttr",
                                     "materializeDerivedAttributes");
    ERROR_IF_PRUNED(method, "materializeDerivedAttributes", op);
    auto &body = method->body();

    auto nonMaterializable =
        make_filter_range(derivedAttrs, [](const NamedAttribute &namedAttr) {
          return namedAttr.attr.getConvertFromStorageCall().empty();
        });
    if (!nonMaterializable.empty()) {
      std::string attrs;
      llvm::raw_string_ostream os(attrs);
      interleaveComma(nonMaterializable, os, [&](const NamedAttribute &attr) {
        os << op.getGetterName(attr.name);
      });
      PrintWarning(
          op.getLoc(),
          formatv(
              "op has non-materializable derived attributes '{0}', skipping",
              os.str()));
      body << formatv("  emitOpError(\"op has non-materializable derived "
                      "attributes '{0}'\");\n",
                      attrs);
      body << "  return nullptr;";
      return;
    }

    body << "  ::mlir::MLIRContext* ctx = getContext();\n";
    body << "  ::mlir::Builder odsBuilder(ctx); (void)odsBuilder;\n";
    body << "  return ::mlir::DictionaryAttr::get(";
    body << "  ctx, {\n";
    interleave(
        derivedAttrs, body,
        [&](const NamedAttribute &namedAttr) {
          auto tmpl = namedAttr.attr.getConvertFromStorageCall();
          std::string name = op.getGetterName(namedAttr.name);
          body << "    {" << name << "AttrName(),\n"
               << tgfmt(tmpl, &fctx.withSelf(name + "()")
                                   .withBuilder("odsBuilder")
                                   .addSubst("_ctxt", "ctx")
                                   .addSubst("_storage", "ctx"))
               << "}";
        },
        ",\n");
    body << "});";
  }
}

void OpEmitter::genAttrSetters() {
  // Generate raw named setter type. This is a wrapper class that allows setting
  // to the attributes via setters instead of having to use the string interface
  // for better compile time verification.
  auto emitAttrWithStorageType = [&](StringRef setterName, StringRef getterName,
                                     Attribute attr) {
    auto *method =
        opClass.addMethod("void", setterName + "Attr",
                          MethodParameter(attr.getStorageType(), "attr"));
    if (method)
      method->body() << formatv("  (*this)->setAttr({0}AttrName(), attr);",
                                getterName);
  };

  // Generate a setter that accepts the underlying C++ type as opposed to the
  // attribute type.
  auto emitAttrWithReturnType = [&](StringRef setterName, StringRef getterName,
                                    Attribute attr) {
    Attribute baseAttr = attr.getBaseAttr();
    if (!canUseUnwrappedRawValue(baseAttr))
      return;
    FmtContext fctx;
    fctx.withBuilder("::mlir::Builder((*this)->getContext())");
    bool isUnitAttr = attr.getAttrDefName() == "UnitAttr";
    bool isOptional = attr.isOptional();

    auto createMethod = [&](const Twine &paramType) {
      return opClass.addMethod("void", setterName,
                               MethodParameter(paramType.str(), "attrValue"));
    };

    // Build the method using the correct parameter type depending on
    // optionality.
    Method *method = nullptr;
    if (isUnitAttr)
      method = createMethod("bool");
    else if (isOptional)
      method =
          createMethod("::std::optional<" + baseAttr.getReturnType() + ">");
    else
      method = createMethod(attr.getReturnType());
    if (!method)
      return;

    // If the value isn't optional, just set it directly.
    if (!isOptional) {
      method->body() << formatv(
          "  (*this)->setAttr({0}AttrName(), {1});", getterName,
          constBuildAttrFromParam(attr, fctx, "attrValue"));
      return;
    }

    // Otherwise, we only set if the provided value is valid. If it isn't, we
    // remove the attribute.

    // TODO: Handle unit attr parameters specially, given that it is treated as
    // optional but not in the same way as the others (i.e. it uses bool over
    // std::optional<>).
    StringRef paramStr = isUnitAttr ? "attrValue" : "*attrValue";
    const char *optionalCodeBody = R"(
    if (attrValue)
      return (*this)->setAttr({0}AttrName(), {1});
    (*this)->removeAttr({0}AttrName());)";
    method->body() << formatv(
        optionalCodeBody, getterName,
        constBuildAttrFromParam(baseAttr, fctx, paramStr));
  };

  for (const NamedAttribute &namedAttr : op.getAttributes()) {
    if (namedAttr.attr.isDerivedAttr())
      continue;
    std::string setterName = op.getSetterName(namedAttr.name);
    std::string getterName = op.getGetterName(namedAttr.name);
    emitAttrWithStorageType(setterName, getterName, namedAttr.attr);
    emitAttrWithReturnType(setterName, getterName, namedAttr.attr);
  }
}

void OpEmitter::genOptionalAttrRemovers() {
  // Generate methods for removing optional attributes, instead of having to
  // use the string interface. Enables better compile time verification.
  auto emitRemoveAttr = [&](StringRef name, bool useProperties) {
    auto upperInitial = name.take_front().upper();
    auto *method = opClass.addMethod("::mlir::Attribute",
                                     op.getRemoverName(name) + "Attr");
    if (!method)
      return;
    if (useProperties) {
      method->body() << formatv(R"(
    auto &attr = getProperties().{0};
    attr = {{};
    return attr;
)",
                                name);
      return;
    }
    method->body() << formatv("return (*this)->removeAttr({0}AttrName());",
                              op.getGetterName(name));
  };

  for (const NamedAttribute &namedAttr : op.getAttributes())
    if (namedAttr.attr.isOptional())
      emitRemoveAttr(namedAttr.name,
                     op.getDialect().usePropertiesForAttributes());
}

// Generates the code to compute the start and end index of an operand or result
// range.
template <typename RangeT>
static void generateValueRangeStartAndEnd(
    Class &opClass, bool isGenericAdaptorBase, StringRef methodName,
    int numVariadic, int numNonVariadic, StringRef rangeSizeCall,
    bool hasAttrSegmentSize, StringRef sizeAttrInit, RangeT &&odsValues) {

  SmallVector<MethodParameter> parameters{MethodParameter("unsigned", "index")};
  if (isGenericAdaptorBase) {
    parameters.emplace_back("unsigned", "odsOperandsSize");
    // The range size is passed per parameter for generic adaptor bases as
    // using the rangeSizeCall would require the operands, which are not
    // accessible in the base class.
    rangeSizeCall = "odsOperandsSize";
  }

  auto *method = opClass.addMethod("std::pair<unsigned, unsigned>", methodName,
                                   parameters);
  if (!method)
    return;
  auto &body = method->body();
  if (numVariadic == 0) {
    body << "  return {index, 1};\n";
  } else if (hasAttrSegmentSize) {
    body << sizeAttrInit << attrSizedSegmentValueRangeCalcCode;
  } else {
    // Because the op can have arbitrarily interleaved variadic and non-variadic
    // operands, we need to embed a list in the "sink" getter method for
    // calculation at run-time.
    SmallVector<StringRef, 4> isVariadic;
    isVariadic.reserve(llvm::size(odsValues));
    for (auto &it : odsValues)
      isVariadic.push_back(it.isVariableLength() ? "true" : "false");
    std::string isVariadicList = llvm::join(isVariadic, ", ");
    body << formatv(sameVariadicSizeValueRangeCalcCode, isVariadicList,
                    numNonVariadic, numVariadic, rangeSizeCall, "operand");
  }
}

static std::string generateTypeForGetter(const NamedTypeConstraint &value) {
  std::string str = "::mlir::Value";
  /// If the CPPClassName is not a fully qualified type. Uses of types
  /// across Dialect fail because they are not in the correct namespace. So we
  /// dont generate TypedValue unless the type is fully qualified.
  /// getCPPClassName doesn't return the fully qualified path for
  /// `mlir::pdl::OperationType` see
  /// https://github.com/llvm/llvm-project/issues/57279.
  /// Adaptor will have values that are not from the type of their operation and
  /// this is expected, so we dont generate TypedValue for Adaptor
  if (value.constraint.getCPPClassName() != "::mlir::Type" &&
      StringRef(value.constraint.getCPPClassName()).startswith("::"))
    str = llvm::formatv("::mlir::TypedValue<{0}>",
                        value.constraint.getCPPClassName())
              .str();
  return str;
}

// Generates the named operand getter methods for the given Operator `op` and
// puts them in `opClass`.  Uses `rangeType` as the return type of getters that
// return a range of operands (individual operands are `Value ` and each
// element in the range must also be `Value `); use `rangeBeginCall` to get
// an iterator to the beginning of the operand range; use `rangeSizeCall` to
// obtain the number of operands. `getOperandCallPattern` contains the code
// necessary to obtain a single operand whose position will be substituted
// instead of
// "{0}" marker in the pattern.  Note that the pattern should work for any kind
// of ops, in particular for one-operand ops that may not have the
// `getOperand(unsigned)` method.
static void
generateNamedOperandGetters(const Operator &op, Class &opClass,
                            Class *genericAdaptorBase, StringRef sizeAttrInit,
                            StringRef rangeType, StringRef rangeElementType,
                            StringRef rangeBeginCall, StringRef rangeSizeCall,
                            StringRef getOperandCallPattern) {
  const int numOperands = op.getNumOperands();
  const int numVariadicOperands = op.getNumVariableLengthOperands();
  const int numNormalOperands = numOperands - numVariadicOperands;

  const auto *sameVariadicSize =
      op.getTrait("::mlir::OpTrait::SameVariadicOperandSize");
  const auto *attrSizedOperands =
      op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments");

  if (numVariadicOperands > 1 && !sameVariadicSize && !attrSizedOperands) {
    PrintFatalError(op.getLoc(), "op has multiple variadic operands but no "
                                 "specification over their sizes");
  }

  if (numVariadicOperands < 2 && attrSizedOperands) {
    PrintFatalError(op.getLoc(), "op must have at least two variadic operands "
                                 "to use 'AttrSizedOperandSegments' trait");
  }

  if (attrSizedOperands && sameVariadicSize) {
    PrintFatalError(op.getLoc(),
                    "op cannot have both 'AttrSizedOperandSegments' and "
                    "'SameVariadicOperandSize' traits");
  }

  // First emit a few "sink" getter methods upon which we layer all nicer named
  // getter methods.
  // If generating for an adaptor, the method is put into the non-templated
  // generic base class, to not require being defined in the header.
  // Since the operand size can't be determined from the base class however,
  // it has to be passed as an additional argument. The trampoline below
  // generates the function with the same signature as the Op in the generic
  // adaptor.
  bool isGenericAdaptorBase = genericAdaptorBase != nullptr;
  generateValueRangeStartAndEnd(
      /*opClass=*/isGenericAdaptorBase ? *genericAdaptorBase : opClass,
      isGenericAdaptorBase,
      /*methodName=*/"getODSOperandIndexAndLength", numVariadicOperands,
      numNormalOperands, rangeSizeCall, attrSizedOperands, sizeAttrInit,
      const_cast<Operator &>(op).getOperands());
  if (isGenericAdaptorBase) {
    // Generate trampoline for calling 'getODSOperandIndexAndLength' with just
    // the index. This just calls the implementation in the base class but
    // passes the operand size as parameter.
    Method *method = opClass.addMethod("std::pair<unsigned, unsigned>",
                                       "getODSOperandIndexAndLength",
                                       MethodParameter("unsigned", "index"));
    ERROR_IF_PRUNED(method, "getODSOperandIndexAndLength", op);
    MethodBody &body = method->body();
    body.indent() << formatv(
        "return Base::getODSOperandIndexAndLength(index, {0});", rangeSizeCall);
  }

  auto *m = opClass.addMethod(rangeType, "getODSOperands",
                              MethodParameter("unsigned", "index"));
  ERROR_IF_PRUNED(m, "getODSOperands", op);
  auto &body = m->body();
  body << formatv(valueRangeReturnCode, rangeBeginCall,
                  "getODSOperandIndexAndLength(index)");

  // Then we emit nicer named getter methods by redirecting to the "sink" getter
  // method.
  for (int i = 0; i != numOperands; ++i) {
    const auto &operand = op.getOperand(i);
    if (operand.name.empty())
      continue;
    std::string name = op.getGetterName(operand.name);
    if (operand.isOptional()) {
      m = opClass.addMethod(isGenericAdaptorBase
                                ? rangeElementType
                                : generateTypeForGetter(operand),
                            name);
      ERROR_IF_PRUNED(m, name, op);
      m->body().indent() << formatv("auto operands = getODSOperands({0});\n"
                                    "return operands.empty() ? {1}{{} : ",
                                    i, m->getReturnType());
      if (!isGenericAdaptorBase)
        m->body() << llvm::formatv("::llvm::cast<{0}>", m->getReturnType());
      m->body() << "(*operands.begin());";
    } else if (operand.isVariadicOfVariadic()) {
      std::string segmentAttr = op.getGetterName(
          operand.constraint.getVariadicOfVariadicSegmentSizeAttr());
      if (genericAdaptorBase) {
        m = opClass.addMethod("::llvm::SmallVector<" + rangeType + ">", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << llvm::formatv(variadicOfVariadicAdaptorCalcCode,
                                   segmentAttr, i, rangeType);
        continue;
      }

      m = opClass.addMethod("::mlir::OperandRangeRange", name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << "  return getODSOperands(" << i << ").split(" << segmentAttr
                << "Attr());";
    } else if (operand.isVariadic()) {
      m = opClass.addMethod(rangeType, name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << "  return getODSOperands(" << i << ");";
    } else {
      m = opClass.addMethod(isGenericAdaptorBase
                                ? rangeElementType
                                : generateTypeForGetter(operand),
                            name);
      ERROR_IF_PRUNED(m, name, op);
      m->body().indent() << "return ";
      if (!isGenericAdaptorBase)
        m->body() << llvm::formatv("::llvm::cast<{0}>", m->getReturnType());
      m->body() << llvm::formatv("(*getODSOperands({0}).begin());", i);
    }
  }
}

void OpEmitter::genNamedOperandGetters() {
  // Build the code snippet used for initializing the operand_segment_size)s
  // array.
  std::string attrSizeInitCode;
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    if (op.getDialect().usePropertiesForAttributes())
      attrSizeInitCode = formatv(adapterSegmentSizeAttrInitCodeProperties,
                                 "getProperties().operandSegmentSizes");

    else
      attrSizeInitCode = formatv(opSegmentSizeAttrInitCode,
                                 emitHelper.getAttr(operandSegmentAttrName));
  }

  generateNamedOperandGetters(
      op, opClass,
      /*genericAdaptorBase=*/nullptr,
      /*sizeAttrInit=*/attrSizeInitCode,
      /*rangeType=*/"::mlir::Operation::operand_range",
      /*rangeElementType=*/"::mlir::Value",
      /*rangeBeginCall=*/"getOperation()->operand_begin()",
      /*rangeSizeCall=*/"getOperation()->getNumOperands()",
      /*getOperandCallPattern=*/"getOperation()->getOperand({0})");
}

void OpEmitter::genNamedOperandSetters() {
  auto *attrSizedOperands =
      op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments");
  for (int i = 0, e = op.getNumOperands(); i != e; ++i) {
    const auto &operand = op.getOperand(i);
    if (operand.name.empty())
      continue;
    std::string name = op.getGetterName(operand.name);

    auto *m = opClass.addMethod(operand.isVariadicOfVariadic()
                                    ? "::mlir::MutableOperandRangeRange"
                                    : "::mlir::MutableOperandRange",
                                name + "Mutable");
    ERROR_IF_PRUNED(m, name, op);
    auto &body = m->body();
    body << "  auto range = getODSOperandIndexAndLength(" << i << ");\n"
         << "  auto mutableRange = "
            "::mlir::MutableOperandRange(getOperation(), "
            "range.first, range.second";
    if (attrSizedOperands) {
      if (emitHelper.hasProperties())
        body << formatv(", ::mlir::MutableOperandRange::OperandSegment({0}u, "
                        "{{getOperandSegmentSizesAttrName(), "
                        "::mlir::DenseI32ArrayAttr::get(getContext(), "
                        "getProperties().operandSegmentSizes)})",
                        i);
      else
        body << formatv(
            ", ::mlir::MutableOperandRange::OperandSegment({0}u, *{1})", i,
            emitHelper.getAttr(operandSegmentAttrName, /*isNamed=*/true));
    }
    body << ");\n";

    // If this operand is a nested variadic, we split the range into a
    // MutableOperandRangeRange that provides a range over all of the
    // sub-ranges.
    if (operand.isVariadicOfVariadic()) {
      body << "  return "
              "mutableRange.split(*(*this)->getAttrDictionary().getNamed("
           << op.getGetterName(
                  operand.constraint.getVariadicOfVariadicSegmentSizeAttr())
           << "AttrName()));\n";
    } else {
      // Otherwise, we use the full range directly.
      body << "  return mutableRange;\n";
    }
  }
}

void OpEmitter::genNamedResultGetters() {
  const int numResults = op.getNumResults();
  const int numVariadicResults = op.getNumVariableLengthResults();
  const int numNormalResults = numResults - numVariadicResults;

  // If we have more than one variadic results, we need more complicated logic
  // to calculate the value range for each result.

  const auto *sameVariadicSize =
      op.getTrait("::mlir::OpTrait::SameVariadicResultSize");
  const auto *attrSizedResults =
      op.getTrait("::mlir::OpTrait::AttrSizedResultSegments");

  if (numVariadicResults > 1 && !sameVariadicSize && !attrSizedResults) {
    PrintFatalError(op.getLoc(), "op has multiple variadic results but no "
                                 "specification over their sizes");
  }

  if (numVariadicResults < 2 && attrSizedResults) {
    PrintFatalError(op.getLoc(), "op must have at least two variadic results "
                                 "to use 'AttrSizedResultSegments' trait");
  }

  if (attrSizedResults && sameVariadicSize) {
    PrintFatalError(op.getLoc(),
                    "op cannot have both 'AttrSizedResultSegments' and "
                    "'SameVariadicResultSize' traits");
  }

  // Build the initializer string for the result segment size attribute.
  std::string attrSizeInitCode;
  if (attrSizedResults) {
    if (op.getDialect().usePropertiesForAttributes())
      attrSizeInitCode = formatv(adapterSegmentSizeAttrInitCodeProperties,
                                 "getProperties().resultSegmentSizes");

    else
      attrSizeInitCode = formatv(opSegmentSizeAttrInitCode,
                                 emitHelper.getAttr(resultSegmentAttrName));
  }

  generateValueRangeStartAndEnd(
      opClass, /*isGenericAdaptorBase=*/false, "getODSResultIndexAndLength",
      numVariadicResults, numNormalResults, "getOperation()->getNumResults()",
      attrSizedResults, attrSizeInitCode, op.getResults());

  auto *m =
      opClass.addMethod("::mlir::Operation::result_range", "getODSResults",
                        MethodParameter("unsigned", "index"));
  ERROR_IF_PRUNED(m, "getODSResults", op);
  m->body() << formatv(valueRangeReturnCode, "getOperation()->result_begin()",
                       "getODSResultIndexAndLength(index)");

  for (int i = 0; i != numResults; ++i) {
    const auto &result = op.getResult(i);
    if (result.name.empty())
      continue;
    std::string name = op.getGetterName(result.name);
    if (result.isOptional()) {
      m = opClass.addMethod(generateTypeForGetter(result), name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << "  auto results = getODSResults(" << i << ");\n"
                << llvm::formatv("  return results.empty()"
                                 " ? {0}()"
                                 " : ::llvm::cast<{0}>(*results.begin());",
                                 m->getReturnType());
    } else if (result.isVariadic()) {
      m = opClass.addMethod("::mlir::Operation::result_range", name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << "  return getODSResults(" << i << ");";
    } else {
      m = opClass.addMethod(generateTypeForGetter(result), name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << llvm::formatv(
          "  return ::llvm::cast<{0}>(*getODSResults({1}).begin());",
          m->getReturnType(), i);
    }
  }
}

void OpEmitter::genNamedRegionGetters() {
  unsigned numRegions = op.getNumRegions();
  for (unsigned i = 0; i < numRegions; ++i) {
    const auto &region = op.getRegion(i);
    if (region.name.empty())
      continue;
    std::string name = op.getGetterName(region.name);

    // Generate the accessors for a variadic region.
    if (region.isVariadic()) {
      auto *m =
          opClass.addMethod("::mlir::MutableArrayRef<::mlir::Region>", name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << formatv("  return (*this)->getRegions().drop_front({0});",
                           i);
      continue;
    }

    auto *m = opClass.addMethod("::mlir::Region &", name);
    ERROR_IF_PRUNED(m, name, op);
    m->body() << formatv("  return (*this)->getRegion({0});", i);
  }
}

void OpEmitter::genNamedSuccessorGetters() {
  unsigned numSuccessors = op.getNumSuccessors();
  for (unsigned i = 0; i < numSuccessors; ++i) {
    const NamedSuccessor &successor = op.getSuccessor(i);
    if (successor.name.empty())
      continue;
    std::string name = op.getGetterName(successor.name);
    // Generate the accessors for a variadic successor list.
    if (successor.isVariadic()) {
      auto *m = opClass.addMethod("::mlir::SuccessorRange", name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << formatv(
          "  return {std::next((*this)->successor_begin(), {0}), "
          "(*this)->successor_end()};",
          i);
      continue;
    }

    auto *m = opClass.addMethod("::mlir::Block *", name);
    ERROR_IF_PRUNED(m, name, op);
    m->body() << formatv("  return (*this)->getSuccessor({0});", i);
  }
}

static bool canGenerateUnwrappedBuilder(const Operator &op) {
  // If this op does not have native attributes at all, return directly to avoid
  // redefining builders.
  if (op.getNumNativeAttributes() == 0)
    return false;

  bool canGenerate = false;
  // We are generating builders that take raw values for attributes. We need to
  // make sure the native attributes have a meaningful "unwrapped" value type
  // different from the wrapped mlir::Attribute type to avoid redefining
  // builders. This checks for the op has at least one such native attribute.
  for (int i = 0, e = op.getNumNativeAttributes(); i < e; ++i) {
    const NamedAttribute &namedAttr = op.getAttribute(i);
    if (canUseUnwrappedRawValue(namedAttr.attr)) {
      canGenerate = true;
      break;
    }
  }
  return canGenerate;
}

static bool canInferType(const Operator &op) {
  return op.getTrait("::mlir::InferTypeOpInterface::Trait");
}

void OpEmitter::genSeparateArgParamBuilder() {
  SmallVector<AttrParamKind, 2> attrBuilderType;
  attrBuilderType.push_back(AttrParamKind::WrappedAttr);
  if (canGenerateUnwrappedBuilder(op))
    attrBuilderType.push_back(AttrParamKind::UnwrappedValue);

  // Emit with separate builders with or without unwrapped attributes and/or
  // inferring result type.
  auto emit = [&](AttrParamKind attrType, TypeParamKind paramKind,
                  bool inferType) {
    SmallVector<MethodParameter> paramList;
    SmallVector<std::string, 4> resultNames;
    llvm::StringSet<> inferredAttributes;
    buildParamList(paramList, inferredAttributes, resultNames, paramKind,
                   attrType);

    auto *m = opClass.addStaticMethod("void", "build", std::move(paramList));
    // If the builder is redundant, skip generating the method.
    if (!m)
      return;
    auto &body = m->body();
    genCodeForAddingArgAndRegionForBuilder(body, inferredAttributes,
                                           /*isRawValueAttr=*/attrType ==
                                               AttrParamKind::UnwrappedValue);

    // Push all result types to the operation state

    if (inferType) {
      // Generate builder that infers type too.
      // TODO: Subsume this with general checking if type can be
      // inferred automatically.
      body << formatv(R"(
        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded({0}::inferReturnTypes(odsBuilder.getContext(),
                      {1}.location, {1}.operands,
                      {1}.attributes.getDictionary({1}.getContext()),
                      {1}.getRawProperties(),
                      {1}.regions, inferredReturnTypes)))
          {1}.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");)",
                      opClass.getClassName(), builderOpState);
      return;
    }

    switch (paramKind) {
    case TypeParamKind::None:
      return;
    case TypeParamKind::Separate:
      for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        if (op.getResult(i).isOptional())
          body << "  if (" << resultNames[i] << ")\n  ";
        body << "  " << builderOpState << ".addTypes(" << resultNames[i]
             << ");\n";
      }

      // Automatically create the 'resultSegmentSizes' attribute using
      // the length of the type ranges.
      if (op.getTrait("::mlir::OpTrait::AttrSizedResultSegments")) {
        if (op.getDialect().usePropertiesForAttributes()) {
          body << "  ::llvm::copy(::llvm::ArrayRef<int32_t>({";
        } else {
          std::string getterName = op.getGetterName(resultSegmentAttrName);
          body << " " << builderOpState << ".addAttribute(" << getterName
               << "AttrName(" << builderOpState << ".name), "
               << "odsBuilder.getDenseI32ArrayAttr({";
        }
        interleaveComma(
            llvm::seq<int>(0, op.getNumResults()), body, [&](int i) {
              const NamedTypeConstraint &result = op.getResult(i);
              if (!result.isVariableLength()) {
                body << "1";
              } else if (result.isOptional()) {
                body << "(" << resultNames[i] << " ? 1 : 0)";
              } else {
                // VariadicOfVariadic of results are currently unsupported in
                // MLIR, hence it can only be a simple variadic.
                // TODO: Add implementation for VariadicOfVariadic results here
                //       once supported.
                assert(result.isVariadic());
                body << "static_cast<int32_t>(" << resultNames[i] << ".size())";
              }
            });
        if (op.getDialect().usePropertiesForAttributes()) {
          body << "}), " << builderOpState
               << ".getOrAddProperties<Properties>()."
                  "resultSegmentSizes.begin());\n";
        } else {
          body << "}));\n";
        }
      }

      return;
    case TypeParamKind::Collective: {
      int numResults = op.getNumResults();
      int numVariadicResults = op.getNumVariableLengthResults();
      int numNonVariadicResults = numResults - numVariadicResults;
      bool hasVariadicResult = numVariadicResults != 0;

      // Avoid emitting "resultTypes.size() >= 0u" which is always true.
      if (!hasVariadicResult || numNonVariadicResults != 0)
        body << "  "
             << "assert(resultTypes.size() "
             << (hasVariadicResult ? ">=" : "==") << " "
             << numNonVariadicResults
             << "u && \"mismatched number of results\");\n";
      body << "  " << builderOpState << ".addTypes(resultTypes);\n";
    }
      return;
    }
    llvm_unreachable("unhandled TypeParamKind");
  };

  // Some of the build methods generated here may be ambiguous, but TableGen's
  // ambiguous function detection will elide those ones.
  for (auto attrType : attrBuilderType) {
    emit(attrType, TypeParamKind::Separate, /*inferType=*/false);
    if (canInferType(op))
      emit(attrType, TypeParamKind::None, /*inferType=*/true);
    emit(attrType, TypeParamKind::Collective, /*inferType=*/false);
  }
}

void OpEmitter::genUseOperandAsResultTypeCollectiveParamBuilder() {
  int numResults = op.getNumResults();

  // Signature
  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("::mlir::OpBuilder &", "odsBuilder");
  paramList.emplace_back("::mlir::OperationState &", builderOpState);
  paramList.emplace_back("::mlir::ValueRange", "operands");
  // Provide default value for `attributes` when its the last parameter
  StringRef attributesDefaultValue = op.getNumVariadicRegions() ? "" : "{}";
  paramList.emplace_back("::llvm::ArrayRef<::mlir::NamedAttribute>",
                         "attributes", attributesDefaultValue);
  if (op.getNumVariadicRegions())
    paramList.emplace_back("unsigned", "numRegions");

  auto *m = opClass.addStaticMethod("void", "build", std::move(paramList));
  // If the builder is redundant, skip generating the method
  if (!m)
    return;
  auto &body = m->body();

  // Operands
  body << "  " << builderOpState << ".addOperands(operands);\n";

  // Attributes
  body << "  " << builderOpState << ".addAttributes(attributes);\n";

  // Create the correct number of regions
  if (int numRegions = op.getNumRegions()) {
    body << llvm::formatv(
        "  for (unsigned i = 0; i != {0}; ++i)\n",
        (op.getNumVariadicRegions() ? "numRegions" : Twine(numRegions)));
    body << "    (void)" << builderOpState << ".addRegion();\n";
  }

  // Result types
  SmallVector<std::string, 2> resultTypes(numResults, "operands[0].getType()");
  body << "  " << builderOpState << ".addTypes({"
       << llvm::join(resultTypes, ", ") << "});\n\n";
}

void OpEmitter::genPopulateDefaultAttributes() {
  // All done if no attributes, except optional ones, have default values.
  if (llvm::all_of(op.getAttributes(), [](const NamedAttribute &named) {
        return !named.attr.hasDefaultValue() || named.attr.isOptional();
      }))
    return;

  if (op.getDialect().usePropertiesForAttributes()) {
    SmallVector<MethodParameter> paramList;
    paramList.emplace_back("::mlir::OperationName", "opName");
    paramList.emplace_back("Properties &", "properties");
    auto *m =
        opClass.addStaticMethod("void", "populateDefaultProperties", paramList);
    ERROR_IF_PRUNED(m, "populateDefaultProperties", op);
    auto &body = m->body();
    body.indent();
    body << "::mlir::Builder " << odsBuilder << "(opName.getContext());\n";
    for (const NamedAttribute &namedAttr : op.getAttributes()) {
      auto &attr = namedAttr.attr;
      if (!attr.hasDefaultValue() || attr.isOptional())
        continue;
      StringRef name = namedAttr.name;
      FmtContext fctx;
      fctx.withBuilder(odsBuilder);
      body << "if (!properties." << name << ")\n"
           << "  properties." << name << " = "
           << std::string(tgfmt(attr.getConstBuilderTemplate(), &fctx,
                                tgfmt(attr.getDefaultValue(), &fctx)))
           << ";\n";
    }
    return;
  }

  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("const ::mlir::OperationName &", "opName");
  paramList.emplace_back("::mlir::NamedAttrList &", "attributes");
  auto *m = opClass.addStaticMethod("void", "populateDefaultAttrs", paramList);
  ERROR_IF_PRUNED(m, "populateDefaultAttrs", op);
  auto &body = m->body();
  body.indent();

  // Set default attributes that are unset.
  body << "auto attrNames = opName.getAttributeNames();\n";
  body << "::mlir::Builder " << odsBuilder
       << "(attrNames.front().getContext());\n";
  StringMap<int> attrIndex;
  for (const auto &it : llvm::enumerate(emitHelper.getAttrMetadata())) {
    attrIndex[it.value().first] = it.index();
  }
  for (const NamedAttribute &namedAttr : op.getAttributes()) {
    auto &attr = namedAttr.attr;
    if (!attr.hasDefaultValue() || attr.isOptional())
      continue;
    auto index = attrIndex[namedAttr.name];
    body << "if (!attributes.get(attrNames[" << index << "])) {\n";
    FmtContext fctx;
    fctx.withBuilder(odsBuilder);

    std::string defaultValue =
        std::string(tgfmt(attr.getConstBuilderTemplate(), &fctx,
                          tgfmt(attr.getDefaultValue(), &fctx)));
    body.indent() << formatv("attributes.append(attrNames[{0}], {1});\n", index,
                             defaultValue);
    body.unindent() << "}\n";
  }
}

void OpEmitter::genInferredTypeCollectiveParamBuilder() {
  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("::mlir::OpBuilder &", "odsBuilder");
  paramList.emplace_back("::mlir::OperationState &", builderOpState);
  paramList.emplace_back("::mlir::ValueRange", "operands");
  StringRef attributesDefaultValue = op.getNumVariadicRegions() ? "" : "{}";
  paramList.emplace_back("::llvm::ArrayRef<::mlir::NamedAttribute>",
                         "attributes", attributesDefaultValue);
  if (op.getNumVariadicRegions())
    paramList.emplace_back("unsigned", "numRegions");

  auto *m = opClass.addStaticMethod("void", "build", std::move(paramList));
  // If the builder is redundant, skip generating the method
  if (!m)
    return;
  auto &body = m->body();

  int numResults = op.getNumResults();
  int numVariadicResults = op.getNumVariableLengthResults();
  int numNonVariadicResults = numResults - numVariadicResults;

  int numOperands = op.getNumOperands();
  int numVariadicOperands = op.getNumVariableLengthOperands();
  int numNonVariadicOperands = numOperands - numVariadicOperands;

  // Operands
  if (numVariadicOperands == 0 || numNonVariadicOperands != 0)
    body << "  assert(operands.size()"
         << (numVariadicOperands != 0 ? " >= " : " == ")
         << numNonVariadicOperands
         << "u && \"mismatched number of parameters\");\n";
  body << "  " << builderOpState << ".addOperands(operands);\n";
  body << "  " << builderOpState << ".addAttributes(attributes);\n";

  // Create the correct number of regions
  if (int numRegions = op.getNumRegions()) {
    body << llvm::formatv(
        "  for (unsigned i = 0; i != {0}; ++i)\n",
        (op.getNumVariadicRegions() ? "numRegions" : Twine(numRegions)));
    body << "    (void)" << builderOpState << ".addRegion();\n";
  }

  // Result types
  body << formatv(R"(
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded({0}::inferReturnTypes(odsBuilder.getContext(),
          {1}.location, operands,
          {1}.attributes.getDictionary({1}.getContext()),
          {1}.getRawProperties(),
          {1}.regions, inferredReturnTypes))) {{)",
                  opClass.getClassName(), builderOpState);
  if (numVariadicResults == 0 || numNonVariadicResults != 0)
    body << "\n    assert(inferredReturnTypes.size()"
         << (numVariadicResults != 0 ? " >= " : " == ") << numNonVariadicResults
         << "u && \"mismatched number of return types\");";
  body << "\n    " << builderOpState << ".addTypes(inferredReturnTypes);";

  body << formatv(R"(
  } else {{
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  })",
                  opClass.getClassName(), builderOpState);
}

void OpEmitter::genUseOperandAsResultTypeSeparateParamBuilder() {
  auto emit = [&](AttrParamKind attrType) {
    SmallVector<MethodParameter> paramList;
    SmallVector<std::string, 4> resultNames;
    llvm::StringSet<> inferredAttributes;
    buildParamList(paramList, inferredAttributes, resultNames,
                   TypeParamKind::None, attrType);

    auto *m = opClass.addStaticMethod("void", "build", std::move(paramList));
    // If the builder is redundant, skip generating the method
    if (!m)
      return;
    auto &body = m->body();
    genCodeForAddingArgAndRegionForBuilder(body, inferredAttributes,
                                           /*isRawValueAttr=*/attrType ==
                                               AttrParamKind::UnwrappedValue);

    auto numResults = op.getNumResults();
    if (numResults == 0)
      return;

    // Push all result types to the operation state
    const char *index = op.getOperand(0).isVariadic() ? ".front()" : "";
    std::string resultType =
        formatv("{0}{1}.getType()", getArgumentName(op, 0), index).str();
    body << "  " << builderOpState << ".addTypes({" << resultType;
    for (int i = 1; i != numResults; ++i)
      body << ", " << resultType;
    body << "});\n\n";
  };

  emit(AttrParamKind::WrappedAttr);
  // Generate additional builder(s) if any attributes can be "unwrapped"
  if (canGenerateUnwrappedBuilder(op))
    emit(AttrParamKind::UnwrappedValue);
}

void OpEmitter::genUseAttrAsResultTypeBuilder() {
  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("::mlir::OpBuilder &", "odsBuilder");
  paramList.emplace_back("::mlir::OperationState &", builderOpState);
  paramList.emplace_back("::mlir::ValueRange", "operands");
  paramList.emplace_back("::llvm::ArrayRef<::mlir::NamedAttribute>",
                         "attributes", "{}");
  auto *m = opClass.addStaticMethod("void", "build", std::move(paramList));
  // If the builder is redundant, skip generating the method
  if (!m)
    return;

  auto &body = m->body();

  // Push all result types to the operation state
  std::string resultType;
  const auto &namedAttr = op.getAttribute(0);

  body << "  auto attrName = " << op.getGetterName(namedAttr.name)
       << "AttrName(" << builderOpState
       << ".name);\n"
          "  for (auto attr : attributes) {\n"
          "    if (attr.getName() != attrName) continue;\n";
  if (namedAttr.attr.isTypeAttr()) {
    resultType = "::llvm::cast<::mlir::TypeAttr>(attr.getValue()).getValue()";
  } else {
    resultType = "::llvm::cast<::mlir::TypedAttr>(attr.getValue()).getType()";
  }

  // Operands
  body << "  " << builderOpState << ".addOperands(operands);\n";

  // Attributes
  body << "  " << builderOpState << ".addAttributes(attributes);\n";

  // Result types
  SmallVector<std::string, 2> resultTypes(op.getNumResults(), resultType);
  body << "    " << builderOpState << ".addTypes({"
       << llvm::join(resultTypes, ", ") << "});\n";
  body << "  }\n";
}

/// Returns a signature of the builder. Updates the context `fctx` to enable
/// replacement of $_builder and $_state in the body.
static SmallVector<MethodParameter>
getBuilderSignature(const Builder &builder) {
  ArrayRef<Builder::Parameter> params(builder.getParameters());

  // Inject builder and state arguments.
  SmallVector<MethodParameter> arguments;
  arguments.reserve(params.size() + 2);
  arguments.emplace_back("::mlir::OpBuilder &", odsBuilder);
  arguments.emplace_back("::mlir::OperationState &", builderOpState);

  for (unsigned i = 0, e = params.size(); i < e; ++i) {
    // If no name is provided, generate one.
    std::optional<StringRef> paramName = params[i].getName();
    std::string name =
        paramName ? paramName->str() : "odsArg" + std::to_string(i);

    StringRef defaultValue;
    if (std::optional<StringRef> defaultParamValue =
            params[i].getDefaultValue())
      defaultValue = *defaultParamValue;

    arguments.emplace_back(params[i].getCppType(), std::move(name),
                           defaultValue);
  }

  return arguments;
}

void OpEmitter::genBuilder() {
  // Handle custom builders if provided.
  for (const Builder &builder : op.getBuilders()) {
    SmallVector<MethodParameter> arguments = getBuilderSignature(builder);

    std::optional<StringRef> body = builder.getBody();
    auto properties = body ? Method::Static : Method::StaticDeclaration;
    auto *method =
        opClass.addMethod("void", "build", properties, std::move(arguments));
    if (body)
      ERROR_IF_PRUNED(method, "build", op);

    if (method)
      method->setDeprecated(builder.getDeprecatedMessage());

    FmtContext fctx;
    fctx.withBuilder(odsBuilder);
    fctx.addSubst("_state", builderOpState);
    if (body)
      method->body() << tgfmt(*body, &fctx);
  }

  // Generate default builders that requires all result type, operands, and
  // attributes as parameters.
  if (op.skipDefaultBuilders())
    return;

  // We generate three classes of builders here:
  // 1. one having a stand-alone parameter for each operand / attribute, and
  genSeparateArgParamBuilder();
  // 2. one having an aggregated parameter for all result types / operands /
  //    attributes, and
  genCollectiveParamBuilder();
  // 3. one having a stand-alone parameter for each operand and attribute,
  //    use the first operand or attribute's type as all result types
  //    to facilitate different call patterns.
  if (op.getNumVariableLengthResults() == 0) {
    if (op.getTrait("::mlir::OpTrait::SameOperandsAndResultType")) {
      genUseOperandAsResultTypeSeparateParamBuilder();
      genUseOperandAsResultTypeCollectiveParamBuilder();
    }
    if (op.getTrait("::mlir::OpTrait::FirstAttrDerivedResultType"))
      genUseAttrAsResultTypeBuilder();
  }
}

void OpEmitter::genCollectiveParamBuilder() {
  int numResults = op.getNumResults();
  int numVariadicResults = op.getNumVariableLengthResults();
  int numNonVariadicResults = numResults - numVariadicResults;

  int numOperands = op.getNumOperands();
  int numVariadicOperands = op.getNumVariableLengthOperands();
  int numNonVariadicOperands = numOperands - numVariadicOperands;

  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("::mlir::OpBuilder &", "");
  paramList.emplace_back("::mlir::OperationState &", builderOpState);
  paramList.emplace_back("::mlir::TypeRange", "resultTypes");
  paramList.emplace_back("::mlir::ValueRange", "operands");
  // Provide default value for `attributes` when its the last parameter
  StringRef attributesDefaultValue = op.getNumVariadicRegions() ? "" : "{}";
  paramList.emplace_back("::llvm::ArrayRef<::mlir::NamedAttribute>",
                         "attributes", attributesDefaultValue);
  if (op.getNumVariadicRegions())
    paramList.emplace_back("unsigned", "numRegions");

  auto *m = opClass.addStaticMethod("void", "build", std::move(paramList));
  // If the builder is redundant, skip generating the method
  if (!m)
    return;
  auto &body = m->body();

  // Operands
  if (numVariadicOperands == 0 || numNonVariadicOperands != 0)
    body << "  assert(operands.size()"
         << (numVariadicOperands != 0 ? " >= " : " == ")
         << numNonVariadicOperands
         << "u && \"mismatched number of parameters\");\n";
  body << "  " << builderOpState << ".addOperands(operands);\n";

  // Attributes
  body << "  " << builderOpState << ".addAttributes(attributes);\n";

  // Create the correct number of regions
  if (int numRegions = op.getNumRegions()) {
    body << llvm::formatv(
        "  for (unsigned i = 0; i != {0}; ++i)\n",
        (op.getNumVariadicRegions() ? "numRegions" : Twine(numRegions)));
    body << "    (void)" << builderOpState << ".addRegion();\n";
  }

  // Result types
  if (numVariadicResults == 0 || numNonVariadicResults != 0)
    body << "  assert(resultTypes.size()"
         << (numVariadicResults != 0 ? " >= " : " == ") << numNonVariadicResults
         << "u && \"mismatched number of return types\");\n";
  body << "  " << builderOpState << ".addTypes(resultTypes);\n";

  // Generate builder that infers type too.
  // TODO: Expand to handle successors.
  if (canInferType(op) && op.getNumSuccessors() == 0)
    genInferredTypeCollectiveParamBuilder();
}

void OpEmitter::buildParamList(SmallVectorImpl<MethodParameter> &paramList,
                               llvm::StringSet<> &inferredAttributes,
                               SmallVectorImpl<std::string> &resultTypeNames,
                               TypeParamKind typeParamKind,
                               AttrParamKind attrParamKind) {
  resultTypeNames.clear();
  auto numResults = op.getNumResults();
  resultTypeNames.reserve(numResults);

  paramList.emplace_back("::mlir::OpBuilder &", odsBuilder);
  paramList.emplace_back("::mlir::OperationState &", builderOpState);

  switch (typeParamKind) {
  case TypeParamKind::None:
    break;
  case TypeParamKind::Separate: {
    // Add parameters for all return types
    for (int i = 0; i < numResults; ++i) {
      const auto &result = op.getResult(i);
      std::string resultName = std::string(result.name);
      if (resultName.empty())
        resultName = std::string(formatv("resultType{0}", i));

      StringRef type =
          result.isVariadic() ? "::mlir::TypeRange" : "::mlir::Type";

      paramList.emplace_back(type, resultName, result.isOptional());
      resultTypeNames.emplace_back(std::move(resultName));
    }
  } break;
  case TypeParamKind::Collective: {
    paramList.emplace_back("::mlir::TypeRange", "resultTypes");
    resultTypeNames.push_back("resultTypes");
  } break;
  }

  // Add parameters for all arguments (operands and attributes).
  int defaultValuedAttrStartIndex = op.getNumArgs();
  // Successors and variadic regions go at the end of the parameter list, so no
  // default arguments are possible.
  bool hasTrailingParams = op.getNumSuccessors() || op.getNumVariadicRegions();
  if (attrParamKind == AttrParamKind::UnwrappedValue && !hasTrailingParams) {
    // Calculate the start index from which we can attach default values in the
    // builder declaration.
    for (int i = op.getNumArgs() - 1; i >= 0; --i) {
      auto *namedAttr =
          llvm::dyn_cast_if_present<tblgen::NamedAttribute *>(op.getArg(i));
      if (!namedAttr || !namedAttr->attr.hasDefaultValue())
        break;

      if (!canUseUnwrappedRawValue(namedAttr->attr))
        break;

      // Creating an APInt requires us to provide bitwidth, value, and
      // signedness, which is complicated compared to others. Similarly
      // for APFloat.
      // TODO: Adjust the 'returnType' field of such attributes
      // to support them.
      StringRef retType = namedAttr->attr.getReturnType();
      if (retType == "::llvm::APInt" || retType == "::llvm::APFloat")
        break;

      defaultValuedAttrStartIndex = i;
    }
  }

  /// Collect any inferred attributes.
  for (const NamedTypeConstraint &operand : op.getOperands()) {
    if (operand.isVariadicOfVariadic()) {
      inferredAttributes.insert(
          operand.constraint.getVariadicOfVariadicSegmentSizeAttr());
    }
  }

  for (int i = 0, e = op.getNumArgs(), numOperands = 0; i < e; ++i) {
    Argument arg = op.getArg(i);
    if (const auto *operand =
            llvm::dyn_cast_if_present<NamedTypeConstraint *>(arg)) {
      StringRef type;
      if (operand->isVariadicOfVariadic())
        type = "::llvm::ArrayRef<::mlir::ValueRange>";
      else if (operand->isVariadic())
        type = "::mlir::ValueRange";
      else
        type = "::mlir::Value";

      paramList.emplace_back(type, getArgumentName(op, numOperands++),
                             operand->isOptional());
      continue;
    }
    if (const auto *operand = llvm::dyn_cast_if_present<NamedProperty *>(arg)) {
      // TODO
      continue;
    }
    const NamedAttribute &namedAttr = *arg.get<NamedAttribute *>();
    const Attribute &attr = namedAttr.attr;

    // Inferred attributes don't need to be added to the param list.
    if (inferredAttributes.contains(namedAttr.name))
      continue;

    StringRef type;
    switch (attrParamKind) {
    case AttrParamKind::WrappedAttr:
      type = attr.getStorageType();
      break;
    case AttrParamKind::UnwrappedValue:
      if (canUseUnwrappedRawValue(attr))
        type = attr.getReturnType();
      else
        type = attr.getStorageType();
      break;
    }

    // Attach default value if requested and possible.
    std::string defaultValue;
    if (attrParamKind == AttrParamKind::UnwrappedValue &&
        i >= defaultValuedAttrStartIndex) {
      defaultValue += attr.getDefaultValue();
    }
    paramList.emplace_back(type, namedAttr.name, StringRef(defaultValue),
                           attr.isOptional());
  }

  /// Insert parameters for each successor.
  for (const NamedSuccessor &succ : op.getSuccessors()) {
    StringRef type =
        succ.isVariadic() ? "::mlir::BlockRange" : "::mlir::Block *";
    paramList.emplace_back(type, succ.name);
  }

  /// Insert parameters for variadic regions.
  for (const NamedRegion &region : op.getRegions())
    if (region.isVariadic())
      paramList.emplace_back("unsigned",
                             llvm::formatv("{0}Count", region.name).str());
}

void OpEmitter::genCodeForAddingArgAndRegionForBuilder(
    MethodBody &body, llvm::StringSet<> &inferredAttributes,
    bool isRawValueAttr) {
  // Push all operands to the result.
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    std::string argName = getArgumentName(op, i);
    const NamedTypeConstraint &operand = op.getOperand(i);
    if (operand.constraint.isVariadicOfVariadic()) {
      body << "  for (::mlir::ValueRange range : " << argName << ")\n   "
           << builderOpState << ".addOperands(range);\n";

      // Add the segment attribute.
      body << "  {\n"
           << "    ::llvm::SmallVector<int32_t> rangeSegments;\n"
           << "    for (::mlir::ValueRange range : " << argName << ")\n"
           << "      rangeSegments.push_back(range.size());\n"
           << "    auto rangeAttr = " << odsBuilder
           << ".getDenseI32ArrayAttr(rangeSegments);\n";
      if (op.getDialect().usePropertiesForAttributes()) {
        body << "    " << builderOpState << ".getOrAddProperties<Properties>()."
             << operand.constraint.getVariadicOfVariadicSegmentSizeAttr()
             << " = rangeAttr;";
      } else {
        body << "    " << builderOpState << ".addAttribute("
             << op.getGetterName(
                    operand.constraint.getVariadicOfVariadicSegmentSizeAttr())
             << "AttrName(" << builderOpState << ".name), rangeAttr);";
      }
      body << "  }\n";
      continue;
    }

    if (operand.isOptional())
      body << "  if (" << argName << ")\n  ";
    body << "  " << builderOpState << ".addOperands(" << argName << ");\n";
  }

  // If the operation has the operand segment size attribute, add it here.
  auto emitSegment = [&]() {
    interleaveComma(llvm::seq<int>(0, op.getNumOperands()), body, [&](int i) {
      const NamedTypeConstraint &operand = op.getOperand(i);
      if (!operand.isVariableLength()) {
        body << "1";
        return;
      }

      std::string operandName = getArgumentName(op, i);
      if (operand.isOptional()) {
        body << "(" << operandName << " ? 1 : 0)";
      } else if (operand.isVariadicOfVariadic()) {
        body << llvm::formatv(
            "static_cast<int32_t>(std::accumulate({0}.begin(), {0}.end(), 0, "
            "[](int32_t curSum, ::mlir::ValueRange range) {{ return curSum + "
            "range.size(); }))",
            operandName);
      } else {
        body << "static_cast<int32_t>(" << getArgumentName(op, i) << ".size())";
      }
    });
  };
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    std::string sizes = op.getGetterName(operandSegmentAttrName);
    if (op.getDialect().usePropertiesForAttributes()) {
      body << "  ::llvm::copy(::llvm::ArrayRef<int32_t>({";
      emitSegment();
      body << "}), " << builderOpState
           << ".getOrAddProperties<Properties>()."
              "operandSegmentSizes.begin());\n";
    } else {
      body << "  " << builderOpState << ".addAttribute(" << sizes << "AttrName("
           << builderOpState << ".name), "
           << "odsBuilder.getDenseI32ArrayAttr({";
      emitSegment();
      body << "}));\n";
    }
  }

  // Push all attributes to the result.
  for (const auto &namedAttr : op.getAttributes()) {
    auto &attr = namedAttr.attr;
    if (attr.isDerivedAttr() || inferredAttributes.contains(namedAttr.name))
      continue;

    // TODO: The wrapping of optional is different for default or not, so don't
    // unwrap for default ones that would fail below.
    bool emitNotNullCheck =
        (attr.isOptional() && !attr.hasDefaultValue()) ||
        (attr.hasDefaultValue() && !isRawValueAttr) ||
        // TODO: UnitAttr is optional, not wrapped, but needs to be guarded as
        // the constant materialization is only for true case.
        (isRawValueAttr && attr.getAttrDefName() == "UnitAttr");
    if (emitNotNullCheck)
      body.indent() << formatv("if ({0}) ", namedAttr.name) << "{\n";

    if (isRawValueAttr && canUseUnwrappedRawValue(attr)) {
      // If this is a raw value, then we need to wrap it in an Attribute
      // instance.
      FmtContext fctx;
      fctx.withBuilder("odsBuilder");
      if (op.getDialect().usePropertiesForAttributes()) {
        body << formatv("  {0}.getOrAddProperties<Properties>().{1} = {2};\n",
                        builderOpState, namedAttr.name,
                        constBuildAttrFromParam(attr, fctx, namedAttr.name));
      } else {
        body << formatv("  {0}.addAttribute({1}AttrName({0}.name), {2});\n",
                        builderOpState, op.getGetterName(namedAttr.name),
                        constBuildAttrFromParam(attr, fctx, namedAttr.name));
      }
    } else {
      if (op.getDialect().usePropertiesForAttributes()) {
        body << formatv("  {0}.getOrAddProperties<Properties>().{1} = {1};\n",
                        builderOpState, namedAttr.name);
      } else {
        body << formatv("  {0}.addAttribute({1}AttrName({0}.name), {2});\n",
                        builderOpState, op.getGetterName(namedAttr.name),
                        namedAttr.name);
      }
    }
    if (emitNotNullCheck)
      body.unindent() << "  }\n";
  }

  // Create the correct number of regions.
  for (const NamedRegion &region : op.getRegions()) {
    if (region.isVariadic())
      body << formatv("  for (unsigned i = 0; i < {0}Count; ++i)\n  ",
                      region.name);

    body << "  (void)" << builderOpState << ".addRegion();\n";
  }

  // Push all successors to the result.
  for (const NamedSuccessor &namedSuccessor : op.getSuccessors()) {
    body << formatv("  {0}.addSuccessors({1});\n", builderOpState,
                    namedSuccessor.name);
  }
}

void OpEmitter::genCanonicalizerDecls() {
  bool hasCanonicalizeMethod = def.getValueAsBit("hasCanonicalizeMethod");
  if (hasCanonicalizeMethod) {
    // static LogicResult FooOp::
    // canonicalize(FooOp op, PatternRewriter &rewriter);
    SmallVector<MethodParameter> paramList;
    paramList.emplace_back(op.getCppClassName(), "op");
    paramList.emplace_back("::mlir::PatternRewriter &", "rewriter");
    auto *m = opClass.declareStaticMethod("::mlir::LogicalResult",
                                          "canonicalize", std::move(paramList));
    ERROR_IF_PRUNED(m, "canonicalize", op);
  }

  // We get a prototype for 'getCanonicalizationPatterns' if requested directly
  // or if using a 'canonicalize' method.
  bool hasCanonicalizer = def.getValueAsBit("hasCanonicalizer");
  if (!hasCanonicalizeMethod && !hasCanonicalizer)
    return;

  // We get a body for 'getCanonicalizationPatterns' when using a 'canonicalize'
  // method, but not implementing 'getCanonicalizationPatterns' manually.
  bool hasBody = hasCanonicalizeMethod && !hasCanonicalizer;

  // Add a signature for getCanonicalizationPatterns if implemented by the
  // dialect or if synthesized to call 'canonicalize'.
  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("::mlir::RewritePatternSet &", "results");
  paramList.emplace_back("::mlir::MLIRContext *", "context");
  auto kind = hasBody ? Method::Static : Method::StaticDeclaration;
  auto *method = opClass.addMethod("void", "getCanonicalizationPatterns", kind,
                                   std::move(paramList));

  // If synthesizing the method, fill it it.
  if (hasBody) {
    ERROR_IF_PRUNED(method, "getCanonicalizationPatterns", op);
    method->body() << "  results.add(canonicalize);\n";
  }
}

void OpEmitter::genFolderDecls() {
  if (!op.hasFolder())
    return;

  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("FoldAdaptor", "adaptor");

  StringRef retType;
  bool hasSingleResult =
      op.getNumResults() == 1 && op.getNumVariableLengthResults() == 0;
  if (hasSingleResult) {
    retType = "::mlir::OpFoldResult";
  } else {
    paramList.emplace_back("::llvm::SmallVectorImpl<::mlir::OpFoldResult> &",
                           "results");
    retType = "::mlir::LogicalResult";
  }

  auto *m = opClass.declareMethod(retType, "fold", std::move(paramList));
  ERROR_IF_PRUNED(m, "fold", op);
}

void OpEmitter::genOpInterfaceMethods(const tblgen::InterfaceTrait *opTrait) {
  Interface interface = opTrait->getInterface();

  // Get the set of methods that should always be declared.
  auto alwaysDeclaredMethodsVec = opTrait->getAlwaysDeclaredMethods();
  llvm::StringSet<> alwaysDeclaredMethods;
  alwaysDeclaredMethods.insert(alwaysDeclaredMethodsVec.begin(),
                               alwaysDeclaredMethodsVec.end());

  for (const InterfaceMethod &method : interface.getMethods()) {
    // Don't declare if the method has a body.
    if (method.getBody())
      continue;
    // Don't declare if the method has a default implementation and the op
    // didn't request that it always be declared.
    if (method.getDefaultImplementation() &&
        !alwaysDeclaredMethods.count(method.getName()))
      continue;
    // Interface methods are allowed to overlap with existing methods, so don't
    // check if pruned.
    (void)genOpInterfaceMethod(method);
  }
}

Method *OpEmitter::genOpInterfaceMethod(const InterfaceMethod &method,
                                        bool declaration) {
  SmallVector<MethodParameter> paramList;
  for (const InterfaceMethod::Argument &arg : method.getArguments())
    paramList.emplace_back(arg.type, arg.name);

  auto props = (method.isStatic() ? Method::Static : Method::None) |
               (declaration ? Method::Declaration : Method::None);
  return opClass.addMethod(method.getReturnType(), method.getName(), props,
                           std::move(paramList));
}

void OpEmitter::genOpInterfaceMethods() {
  for (const auto &trait : op.getTraits()) {
    if (const auto *opTrait = dyn_cast<tblgen::InterfaceTrait>(&trait))
      if (opTrait->shouldDeclareMethods())
        genOpInterfaceMethods(opTrait);
  }
}

void OpEmitter::genSideEffectInterfaceMethods() {
  enum EffectKind { Operand, Result, Symbol, Static };
  struct EffectLocation {
    /// The effect applied.
    SideEffect effect;

    /// The index if the kind is not static.
    unsigned index;

    /// The kind of the location.
    unsigned kind;
  };

  StringMap<SmallVector<EffectLocation, 1>> interfaceEffects;
  auto resolveDecorators = [&](Operator::var_decorator_range decorators,
                               unsigned index, unsigned kind) {
    for (auto decorator : decorators)
      if (SideEffect *effect = dyn_cast<SideEffect>(&decorator)) {
        opClass.addTrait(effect->getInterfaceTrait());
        interfaceEffects[effect->getBaseEffectName()].push_back(
            EffectLocation{*effect, index, kind});
      }
  };

  // Collect effects that were specified via:
  /// Traits.
  for (const auto &trait : op.getTraits()) {
    const auto *opTrait = dyn_cast<tblgen::SideEffectTrait>(&trait);
    if (!opTrait)
      continue;
    auto &effects = interfaceEffects[opTrait->getBaseEffectName()];
    for (auto decorator : opTrait->getEffects())
      effects.push_back(EffectLocation{cast<SideEffect>(decorator),
                                       /*index=*/0, EffectKind::Static});
  }
  /// Attributes and Operands.
  for (unsigned i = 0, operandIt = 0, e = op.getNumArgs(); i != e; ++i) {
    Argument arg = op.getArg(i);
    if (arg.is<NamedTypeConstraint *>()) {
      resolveDecorators(op.getArgDecorators(i), operandIt, EffectKind::Operand);
      ++operandIt;
      continue;
    }
    if (arg.is<NamedProperty *>())
      continue;
    const NamedAttribute *attr = arg.get<NamedAttribute *>();
    if (attr->attr.getBaseAttr().isSymbolRefAttr())
      resolveDecorators(op.getArgDecorators(i), i, EffectKind::Symbol);
  }
  /// Results.
  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    resolveDecorators(op.getResultDecorators(i), i, EffectKind::Result);

  // The code used to add an effect instance.
  // {0}: The effect class.
  // {1}: Optional value or symbol reference.
  // {1}: The resource class.
  const char *addEffectCode =
      "  effects.emplace_back({0}::get(), {1}{2}::get());\n";

  for (auto &it : interfaceEffects) {
    // Generate the 'getEffects' method.
    std::string type = llvm::formatv("::llvm::SmallVectorImpl<::mlir::"
                                     "SideEffects::EffectInstance<{0}>> &",
                                     it.first())
                           .str();
    auto *getEffects = opClass.addMethod("void", "getEffects",
                                         MethodParameter(type, "effects"));
    ERROR_IF_PRUNED(getEffects, "getEffects", op);
    auto &body = getEffects->body();

    // Add effect instances for each of the locations marked on the operation.
    for (auto &location : it.second) {
      StringRef effect = location.effect.getName();
      StringRef resource = location.effect.getResource();
      if (location.kind == EffectKind::Static) {
        // A static instance has no attached value.
        body << llvm::formatv(addEffectCode, effect, "", resource).str();
      } else if (location.kind == EffectKind::Symbol) {
        // A symbol reference requires adding the proper attribute.
        const auto *attr = op.getArg(location.index).get<NamedAttribute *>();
        std::string argName = op.getGetterName(attr->name);
        if (attr->attr.isOptional()) {
          body << "  if (auto symbolRef = " << argName << "Attr())\n  "
               << llvm::formatv(addEffectCode, effect, "symbolRef, ", resource)
                      .str();
        } else {
          body << llvm::formatv(addEffectCode, effect, argName + "Attr(), ",
                                resource)
                      .str();
        }
      } else {
        // Otherwise this is an operand/result, so we need to attach the Value.
        body << "  for (::mlir::Value value : getODS"
             << (location.kind == EffectKind::Operand ? "Operands" : "Results")
             << "(" << location.index << "))\n  "
             << llvm::formatv(addEffectCode, effect, "value, ", resource).str();
      }
    }
  }
}

void OpEmitter::genTypeInterfaceMethods() {
  if (!op.allResultTypesKnown())
    return;
  // Generate 'inferReturnTypes' method declaration using the interface method
  // declared in 'InferTypeOpInterface' op interface.
  const auto *trait =
      cast<InterfaceTrait>(op.getTrait("::mlir::InferTypeOpInterface::Trait"));
  Interface interface = trait->getInterface();
  Method *method = [&]() -> Method * {
    for (const InterfaceMethod &interfaceMethod : interface.getMethods()) {
      if (interfaceMethod.getName() == "inferReturnTypes") {
        return genOpInterfaceMethod(interfaceMethod, /*declaration=*/false);
      }
    }
    assert(0 && "unable to find inferReturnTypes interface method");
    return nullptr;
  }();
  ERROR_IF_PRUNED(method, "inferReturnTypes", op);
  auto &body = method->body();
  body << "  inferredReturnTypes.resize(" << op.getNumResults() << ");\n";

  FmtContext fctx;
  fctx.withBuilder("odsBuilder");
  fctx.addSubst("_ctxt", "context");
  body << "  ::mlir::Builder odsBuilder(context);\n";

  // Process the type inference graph in topological order, starting from types
  // that are always fully-inferred: operands and results with constructible
  // types. The type inference graph here will always be a DAG, so this gives
  // us the correct order for generating the types. -1 is a placeholder to
  // indicate the type for a result has not been generated.
  SmallVector<int> constructedIndices(op.getNumResults(), -1);
  int inferredTypeIdx = 0;
  for (int numResults = op.getNumResults(); inferredTypeIdx != numResults;) {
    for (int i = 0, e = op.getNumResults(); i != e; ++i) {
      if (constructedIndices[i] >= 0)
        continue;
      const InferredResultType &infer = op.getInferredResultType(i);
      std::string typeStr;
      if (infer.isArg()) {
        // If this is an operand, just index into operand list to access the
        // type.
        auto arg = op.getArgToOperandOrAttribute(infer.getIndex());
        if (arg.kind() == Operator::OperandOrAttribute::Kind::Operand) {
          typeStr = ("operands[" + Twine(arg.operandOrAttributeIndex()) +
                     "].getType()")
                        .str();

          // If this is an attribute, index into the attribute dictionary.
        } else {
          auto *attr =
              op.getArg(arg.operandOrAttributeIndex()).get<NamedAttribute *>();
          body << "  ::mlir::TypedAttr odsInferredTypeAttr" << inferredTypeIdx
               << " = ";
          if (op.getDialect().usePropertiesForAttributes()) {
            body << "(properties ? properties.as<Properties *>()->"
                 << attr->name
                 << " : "
                    "::llvm::dyn_cast_or_null<::mlir::TypedAttr>(attributes."
                    "get(\"" +
                        attr->name + "\")));\n";
          } else {
            body << "::llvm::dyn_cast_or_null<::mlir::TypedAttr>(attributes."
                    "get(\"" +
                        attr->name + "\"));\n";
          }
          body << "  if (!odsInferredTypeAttr" << inferredTypeIdx
               << ") return ::mlir::failure();\n";
          typeStr =
              ("odsInferredTypeAttr" + Twine(inferredTypeIdx) + ".getType()")
                  .str();
        }
      } else if (std::optional<StringRef> builder =
                     op.getResult(infer.getResultIndex())
                         .constraint.getBuilderCall()) {
        typeStr = tgfmt(*builder, &fctx).str();
      } else if (int index = constructedIndices[infer.getResultIndex()];
                 index >= 0) {
        typeStr = ("odsInferredType" + Twine(index)).str();
      } else {
        continue;
      }
      body << "  ::mlir::Type odsInferredType" << inferredTypeIdx++ << " = "
           << tgfmt(infer.getTransformer(), &fctx.withSelf(typeStr)) << ";\n";
      constructedIndices[i] = inferredTypeIdx - 1;
    }
  }
  for (auto [i, index] : llvm::enumerate(constructedIndices))
    body << "  inferredReturnTypes[" << i << "] = odsInferredType" << index
         << ";\n";
  body << "  return ::mlir::success();";
}

void OpEmitter::genParser() {
  if (hasStringAttribute(def, "assemblyFormat"))
    return;

  if (!def.getValueAsBit("hasCustomAssemblyFormat"))
    return;

  SmallVector<MethodParameter> paramList;
  paramList.emplace_back("::mlir::OpAsmParser &", "parser");
  paramList.emplace_back("::mlir::OperationState &", "result");

  auto *method = opClass.declareStaticMethod("::mlir::ParseResult", "parse",
                                             std::move(paramList));
  ERROR_IF_PRUNED(method, "parse", op);
}

void OpEmitter::genPrinter() {
  if (hasStringAttribute(def, "assemblyFormat"))
    return;

  // Check to see if this op uses a c++ format.
  if (!def.getValueAsBit("hasCustomAssemblyFormat"))
    return;
  auto *method = opClass.declareMethod(
      "void", "print", MethodParameter("::mlir::OpAsmPrinter &", "p"));
  ERROR_IF_PRUNED(method, "print", op);
}

void OpEmitter::genVerifier() {
  auto *implMethod =
      opClass.addMethod("::mlir::LogicalResult", "verifyInvariantsImpl");
  ERROR_IF_PRUNED(implMethod, "verifyInvariantsImpl", op);
  auto &implBody = implMethod->body();
  bool useProperties = emitHelper.hasProperties();

  populateSubstitutions(emitHelper, verifyCtx);
  genAttributeVerifier(emitHelper, verifyCtx, implBody, staticVerifierEmitter,
                       useProperties);
  genOperandResultVerifier(implBody, op.getOperands(), "operand");
  genOperandResultVerifier(implBody, op.getResults(), "result");

  for (auto &trait : op.getTraits()) {
    if (auto *t = dyn_cast<tblgen::PredTrait>(&trait)) {
      implBody << tgfmt("  if (!($0))\n    "
                        "return emitOpError(\"failed to verify that $1\");\n",
                        &verifyCtx, tgfmt(t->getPredTemplate(), &verifyCtx),
                        t->getSummary());
    }
  }

  genRegionVerifier(implBody);
  genSuccessorVerifier(implBody);

  implBody << "  return ::mlir::success();\n";

  // TODO: Some places use the `verifyInvariants` to do operation verification.
  // This may not act as their expectation because this doesn't call any
  // verifiers of native/interface traits. Needs to review those use cases and
  // see if we should use the mlir::verify() instead.
  auto *method = opClass.addMethod("::mlir::LogicalResult", "verifyInvariants");
  ERROR_IF_PRUNED(method, "verifyInvariants", op);
  auto &body = method->body();
  if (def.getValueAsBit("hasVerifier")) {
    body << "  if(::mlir::succeeded(verifyInvariantsImpl()) && "
            "::mlir::succeeded(verify()))\n";
    body << "    return ::mlir::success();\n";
    body << "  return ::mlir::failure();";
  } else {
    body << "  return verifyInvariantsImpl();";
  }
}

void OpEmitter::genCustomVerifier() {
  if (def.getValueAsBit("hasVerifier")) {
    auto *method = opClass.declareMethod("::mlir::LogicalResult", "verify");
    ERROR_IF_PRUNED(method, "verify", op);
  }

  if (def.getValueAsBit("hasRegionVerifier")) {
    auto *method =
        opClass.declareMethod("::mlir::LogicalResult", "verifyRegions");
    ERROR_IF_PRUNED(method, "verifyRegions", op);
  }
}

void OpEmitter::genOperandResultVerifier(MethodBody &body,
                                         Operator::const_value_range values,
                                         StringRef valueKind) {
  // Check that an optional value is at most 1 element.
  //
  // {0}: Value index.
  // {1}: "operand" or "result"
  const char *const verifyOptional = R"(
    if (valueGroup{0}.size() > 1) {
      return emitOpError("{1} group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup{0}.size();
    }
)";
  // Check the types of a range of values.
  //
  // {0}: Value index.
  // {1}: Type constraint function.
  // {2}: "operand" or "result"
  const char *const verifyValues = R"(
    for (auto v : valueGroup{0}) {
      if (::mlir::failed({1}(*this, v.getType(), "{2}", index++)))
        return ::mlir::failure();
    }
)";

  const auto canSkip = [](const NamedTypeConstraint &value) {
    return !value.hasPredicate() && !value.isOptional() &&
           !value.isVariadicOfVariadic();
  };
  if (values.empty() || llvm::all_of(values, canSkip))
    return;

  FmtContext fctx;

  body << "  {\n    unsigned index = 0; (void)index;\n";

  for (const auto &staticValue : llvm::enumerate(values)) {
    const NamedTypeConstraint &value = staticValue.value();

    bool hasPredicate = value.hasPredicate();
    bool isOptional = value.isOptional();
    bool isVariadicOfVariadic = value.isVariadicOfVariadic();
    if (!hasPredicate && !isOptional && !isVariadicOfVariadic)
      continue;
    body << formatv("    auto valueGroup{2} = getODS{0}{1}s({2});\n",
                    // Capitalize the first letter to match the function name
                    valueKind.substr(0, 1).upper(), valueKind.substr(1),
                    staticValue.index());

    // If the constraint is optional check that the value group has at most 1
    // value.
    if (isOptional) {
      body << formatv(verifyOptional, staticValue.index(), valueKind);
    } else if (isVariadicOfVariadic) {
      body << formatv(
          "    if (::mlir::failed(::mlir::OpTrait::impl::verifyValueSizeAttr("
          "*this, \"{0}\", \"{1}\", valueGroup{2}.size())))\n"
          "      return ::mlir::failure();\n",
          value.constraint.getVariadicOfVariadicSegmentSizeAttr(), value.name,
          staticValue.index());
    }

    // Otherwise, if there is no predicate there is nothing left to do.
    if (!hasPredicate)
      continue;
    // Emit a loop to check all the dynamic values in the pack.
    StringRef constraintFn =
        staticVerifierEmitter.getTypeConstraintFn(value.constraint);
    body << formatv(verifyValues, staticValue.index(), constraintFn, valueKind);
  }

  body << "  }\n";
}

void OpEmitter::genRegionVerifier(MethodBody &body) {
  /// Code to verify a region.
  ///
  /// {0}: Getter for the regions.
  /// {1}: The region constraint.
  /// {2}: The region's name.
  /// {3}: The region description.
  const char *const verifyRegion = R"(
    for (auto &region : {0})
      if (::mlir::failed({1}(*this, region, "{2}", index++)))
        return ::mlir::failure();
)";
  /// Get a single region.
  ///
  /// {0}: The region's index.
  const char *const getSingleRegion =
      "::llvm::MutableArrayRef((*this)->getRegion({0}))";

  // If we have no regions, there is nothing more to do.
  const auto canSkip = [](const NamedRegion &region) {
    return region.constraint.getPredicate().isNull();
  };
  auto regions = op.getRegions();
  if (regions.empty() && llvm::all_of(regions, canSkip))
    return;

  body << "  {\n    unsigned index = 0; (void)index;\n";
  for (const auto &it : llvm::enumerate(regions)) {
    const auto &region = it.value();
    if (canSkip(region))
      continue;

    auto getRegion = region.isVariadic()
                         ? formatv("{0}()", op.getGetterName(region.name)).str()
                         : formatv(getSingleRegion, it.index()).str();
    auto constraintFn =
        staticVerifierEmitter.getRegionConstraintFn(region.constraint);
    body << formatv(verifyRegion, getRegion, constraintFn, region.name);
  }
  body << "  }\n";
}

void OpEmitter::genSuccessorVerifier(MethodBody &body) {
  const char *const verifySuccessor = R"(
    for (auto *successor : {0})
      if (::mlir::failed({1}(*this, successor, "{2}", index++)))
        return ::mlir::failure();
)";
  /// Get a single successor.
  ///
  /// {0}: The successor's name.
  const char *const getSingleSuccessor = "::llvm::MutableArrayRef({0}())";

  // If we have no successors, there is nothing more to do.
  const auto canSkip = [](const NamedSuccessor &successor) {
    return successor.constraint.getPredicate().isNull();
  };
  auto successors = op.getSuccessors();
  if (successors.empty() && llvm::all_of(successors, canSkip))
    return;

  body << "  {\n    unsigned index = 0; (void)index;\n";

  for (auto it : llvm::enumerate(successors)) {
    const auto &successor = it.value();
    if (canSkip(successor))
      continue;

    auto getSuccessor =
        formatv(successor.isVariadic() ? "{0}()" : getSingleSuccessor,
                successor.name, it.index())
            .str();
    auto constraintFn =
        staticVerifierEmitter.getSuccessorConstraintFn(successor.constraint);
    body << formatv(verifySuccessor, getSuccessor, constraintFn,
                    successor.name);
  }
  body << "  }\n";
}

/// Add a size count trait to the given operation class.
static void addSizeCountTrait(OpClass &opClass, StringRef traitKind,
                              int numTotal, int numVariadic) {
  if (numVariadic != 0) {
    if (numTotal == numVariadic)
      opClass.addTrait("::mlir::OpTrait::Variadic" + traitKind + "s");
    else
      opClass.addTrait("::mlir::OpTrait::AtLeastN" + traitKind + "s<" +
                       Twine(numTotal - numVariadic) + ">::Impl");
    return;
  }
  switch (numTotal) {
  case 0:
    opClass.addTrait("::mlir::OpTrait::Zero" + traitKind + "s");
    break;
  case 1:
    opClass.addTrait("::mlir::OpTrait::One" + traitKind);
    break;
  default:
    opClass.addTrait("::mlir::OpTrait::N" + traitKind + "s<" + Twine(numTotal) +
                     ">::Impl");
    break;
  }
}

void OpEmitter::genTraits() {
  // Add region size trait.
  unsigned numRegions = op.getNumRegions();
  unsigned numVariadicRegions = op.getNumVariadicRegions();
  addSizeCountTrait(opClass, "Region", numRegions, numVariadicRegions);

  // Add result size traits.
  int numResults = op.getNumResults();
  int numVariadicResults = op.getNumVariableLengthResults();
  addSizeCountTrait(opClass, "Result", numResults, numVariadicResults);

  // For single result ops with a known specific type, generate a OneTypedResult
  // trait.
  if (numResults == 1 && numVariadicResults == 0) {
    auto cppName = op.getResults().begin()->constraint.getCPPClassName();
    opClass.addTrait("::mlir::OpTrait::OneTypedResult<" + cppName + ">::Impl");
  }

  // Add successor size trait.
  unsigned numSuccessors = op.getNumSuccessors();
  unsigned numVariadicSuccessors = op.getNumVariadicSuccessors();
  addSizeCountTrait(opClass, "Successor", numSuccessors, numVariadicSuccessors);

  // Add variadic size trait and normal op traits.
  int numOperands = op.getNumOperands();
  int numVariadicOperands = op.getNumVariableLengthOperands();

  // Add operand size trait.
  addSizeCountTrait(opClass, "Operand", numOperands, numVariadicOperands);

  // The op traits defined internal are ensured that they can be verified
  // earlier.
  for (const auto &trait : op.getTraits()) {
    if (auto *opTrait = dyn_cast<tblgen::NativeTrait>(&trait)) {
      if (opTrait->isStructuralOpTrait())
        opClass.addTrait(opTrait->getFullyQualifiedTraitName());
    }
  }

  // OpInvariants wrapps the verifyInvariants which needs to be run before
  // native/interface traits and after all the traits with `StructuralOpTrait`.
  opClass.addTrait("::mlir::OpTrait::OpInvariants");

  if (emitHelper.hasProperties())
    opClass.addTrait("::mlir::BytecodeOpInterface::Trait");

  // Add the native and interface traits.
  for (const auto &trait : op.getTraits()) {
    if (auto *opTrait = dyn_cast<tblgen::NativeTrait>(&trait)) {
      if (!opTrait->isStructuralOpTrait())
        opClass.addTrait(opTrait->getFullyQualifiedTraitName());
    } else if (auto *opTrait = dyn_cast<tblgen::InterfaceTrait>(&trait)) {
      opClass.addTrait(opTrait->getFullyQualifiedTraitName());
    }
  }
}

void OpEmitter::genOpNameGetter() {
  auto *method = opClass.addStaticMethod<Method::Constexpr>(
      "::llvm::StringLiteral", "getOperationName");
  ERROR_IF_PRUNED(method, "getOperationName", op);
  method->body() << "  return ::llvm::StringLiteral(\"" << op.getOperationName()
                 << "\");";
}

void OpEmitter::genOpAsmInterface() {
  // If the user only has one results or specifically added the Asm trait,
  // then don't generate it for them. We specifically only handle multi result
  // operations, because the name of a single result in the common case is not
  // interesting(generally 'result'/'output'/etc.).
  // TODO: We could also add a flag to allow operations to opt in to this
  // generation, even if they only have a single operation.
  int numResults = op.getNumResults();
  if (numResults <= 1 || op.getTrait("::mlir::OpAsmOpInterface::Trait"))
    return;

  SmallVector<StringRef, 4> resultNames(numResults);
  for (int i = 0; i != numResults; ++i)
    resultNames[i] = op.getResultName(i);

  // Don't add the trait if none of the results have a valid name.
  if (llvm::all_of(resultNames, [](StringRef name) { return name.empty(); }))
    return;
  opClass.addTrait("::mlir::OpAsmOpInterface::Trait");

  // Generate the right accessor for the number of results.
  auto *method = opClass.addMethod(
      "void", "getAsmResultNames",
      MethodParameter("::mlir::OpAsmSetValueNameFn", "setNameFn"));
  ERROR_IF_PRUNED(method, "getAsmResultNames", op);
  auto &body = method->body();
  for (int i = 0; i != numResults; ++i) {
    body << "  auto resultGroup" << i << " = getODSResults(" << i << ");\n"
         << "  if (!resultGroup" << i << ".empty())\n"
         << "    setNameFn(*resultGroup" << i << ".begin(), \""
         << resultNames[i] << "\");\n";
  }
}

//===----------------------------------------------------------------------===//
// OpOperandAdaptor emitter
//===----------------------------------------------------------------------===//

namespace {
// Helper class to emit Op operand adaptors to an output stream.  Operand
// adaptors are wrappers around random access ranges that provide named operand
// getters identical to those defined in the Op.
// This currently generates 3 classes per Op:
// * A Base class within the 'detail' namespace, which contains all logic and
//   members independent of the random access range that is indexed into.
//   In other words, it contains all the attribute and region getters.
// * A templated class named '{OpName}GenericAdaptor' with a template parameter
//   'RangeT' that is indexed into by the getters to access the operands.
//   It contains all getters to access operands and inherits from the previous
//   class.
// * A class named '{OpName}Adaptor', which inherits from the 'GenericAdaptor'
//   with 'mlir::ValueRange' as template parameter. It adds a constructor from
//   an instance of the op type and a verify function.
class OpOperandAdaptorEmitter {
public:
  static void
  emitDecl(const Operator &op,
           const StaticVerifierFunctionEmitter &staticVerifierEmitter,
           raw_ostream &os);
  static void
  emitDef(const Operator &op,
          const StaticVerifierFunctionEmitter &staticVerifierEmitter,
          raw_ostream &os);

private:
  explicit OpOperandAdaptorEmitter(
      const Operator &op,
      const StaticVerifierFunctionEmitter &staticVerifierEmitter);

  // Add verification function. This generates a verify method for the adaptor
  // which verifies all the op-independent attribute constraints.
  void addVerification();

  // The operation for which to emit an adaptor.
  const Operator &op;

  // The generated adaptor classes.
  Class genericAdaptorBase;
  Class genericAdaptor;
  Class adaptor;

  // The emitter containing all of the locally emitted verification functions.
  const StaticVerifierFunctionEmitter &staticVerifierEmitter;

  // Helper for emitting adaptor code.
  OpOrAdaptorHelper emitHelper;
};
} // namespace

OpOperandAdaptorEmitter::OpOperandAdaptorEmitter(
    const Operator &op,
    const StaticVerifierFunctionEmitter &staticVerifierEmitter)
    : op(op), genericAdaptorBase(op.getGenericAdaptorName() + "Base"),
      genericAdaptor(op.getGenericAdaptorName()), adaptor(op.getAdaptorName()),
      staticVerifierEmitter(staticVerifierEmitter),
      emitHelper(op, /*emitForOp=*/false) {

  genericAdaptorBase.declare<VisibilityDeclaration>(Visibility::Public);
  bool useProperties = emitHelper.hasProperties();
  if (useProperties) {
    // Define the properties struct with multiple members.
    using ConstArgument =
        llvm::PointerUnion<const AttributeMetadata *, const NamedProperty *>;
    SmallVector<ConstArgument> attrOrProperties;
    for (const std::pair<StringRef, AttributeMetadata> &it :
         emitHelper.getAttrMetadata()) {
      if (!it.second.constraint || !it.second.constraint->isDerivedAttr())
        attrOrProperties.push_back(&it.second);
    }
    for (const NamedProperty &prop : op.getProperties())
      attrOrProperties.push_back(&prop);
    if (emitHelper.getOperandSegmentsSize())
      attrOrProperties.push_back(&emitHelper.getOperandSegmentsSize().value());
    if (emitHelper.getResultSegmentsSize())
      attrOrProperties.push_back(&emitHelper.getResultSegmentsSize().value());
    assert(!attrOrProperties.empty());
    std::string declarations = "  struct Properties {\n";
    llvm::raw_string_ostream os(declarations);
    std::string comparator =
        "    bool operator==(const Properties &rhs) const {\n"
        "      return \n";
    llvm::raw_string_ostream comparatorOs(comparator);
    for (const auto &attrOrProp : attrOrProperties) {
      if (const auto *namedProperty =
              llvm::dyn_cast_if_present<const NamedProperty *>(attrOrProp)) {
        StringRef name = namedProperty->name;
        if (name.empty())
          report_fatal_error("missing name for property");
        std::string camelName =
            convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);
        auto &prop = namedProperty->prop;
        // Generate the data member using the storage type.
        os << "    using " << name << "Ty = " << prop.getStorageType() << ";\n"
           << "    " << name << "Ty " << name;
        if (prop.hasDefaultValue())
          os << " = " << prop.getDefaultValue();
        comparatorOs << "        rhs." << name << " == this->" << name
                     << " &&\n";
        // Emit accessors using the interface type.
        const char *accessorFmt = R"decl(;
    {0} get{1}() {
      auto &propStorage = this->{2};
      return {3};
    }
    void set{1}(const {0} &propValue) {
      auto &propStorage = this->{2};
      {4};
    }
)decl";
        FmtContext fctx;
        os << formatv(accessorFmt, prop.getInterfaceType(), camelName, name,
                      tgfmt(prop.getConvertFromStorageCall(),
                            &fctx.addSubst("_storage", propertyStorage)),
                      tgfmt(prop.getAssignToStorageCall(),
                            &fctx.addSubst("_value", propertyValue)
                                 .addSubst("_storage", propertyStorage)));
        continue;
      }
      const auto *namedAttr =
          llvm::dyn_cast_if_present<const AttributeMetadata *>(attrOrProp);
      const Attribute *attr = nullptr;
      if (namedAttr->constraint)
        attr = &*namedAttr->constraint;
      StringRef name = namedAttr->attrName;
      if (name.empty())
        report_fatal_error("missing name for property attr");
      std::string camelName =
          convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);
      // Generate the data member using the storage type.
      StringRef storageType;
      if (attr) {
        storageType = attr->getStorageType();
      } else {
        if (name != operandSegmentAttrName && name != resultSegmentAttrName) {
          report_fatal_error("unexpected AttributeMetadata");
        }
        // TODO: update to use native integers.
        storageType = "::mlir::DenseI32ArrayAttr";
      }
      os << "    using " << name << "Ty = " << storageType << ";\n"
         << "    " << name << "Ty " << name << ";\n";
      comparatorOs << "        rhs." << name << " == this->" << name << " &&\n";

      // Emit accessors using the interface type.
      if (attr) {
        const char *accessorFmt = R"decl(
    auto get{0}() {
      auto &propStorage = this->{1};
      return ::llvm::{2}<{3}>(propStorage);
    }
    void set{0}(const {3} &propValue) {
      this->{1} = propValue;
    }
)decl";
        os << formatv(accessorFmt, camelName, name,
                      attr->isOptional() || attr->hasDefaultValue()
                          ? "dyn_cast_or_null"
                          : "cast",
                      storageType);
      }
    }
    comparatorOs << "        true;\n    }\n"
                    "    bool operator!=(const Properties &rhs) const {\n"
                    "      return !(*this == rhs);\n"
                    "    }\n";
    comparatorOs.flush();
    os << comparator;
    os << "  };\n";
    os.flush();

    genericAdaptorBase.declare<ExtraClassDeclaration>(std::move(declarations));
  }
  genericAdaptorBase.declare<VisibilityDeclaration>(Visibility::Protected);
  genericAdaptorBase.declare<Field>("::mlir::DictionaryAttr", "odsAttrs");
  genericAdaptorBase.declare<Field>("::std::optional<::mlir::OperationName>",
                                    "odsOpName");
  if (useProperties)
    genericAdaptorBase.declare<Field>("Properties", "properties");
  genericAdaptorBase.declare<Field>("::mlir::RegionRange", "odsRegions");

  genericAdaptor.addTemplateParam("RangeT");
  genericAdaptor.addField("RangeT", "odsOperands");
  genericAdaptor.addParent(
      ParentClass("detail::" + genericAdaptorBase.getClassName()));
  genericAdaptor.declare<UsingDeclaration>(
      "ValueT", "::llvm::detail::ValueOfRange<RangeT>");
  genericAdaptor.declare<UsingDeclaration>(
      "Base", "detail::" + genericAdaptorBase.getClassName());

  const auto *attrSizedOperands =
      op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments");
  {
    SmallVector<MethodParameter> paramList;
    paramList.emplace_back("::mlir::DictionaryAttr", "attrs",
                           attrSizedOperands ? "" : "nullptr");
    if (useProperties)
      paramList.emplace_back("const Properties &", "properties", "{}");
    else
      paramList.emplace_back("const ::mlir::EmptyProperties &", "properties",
                             "{}");
    paramList.emplace_back("::mlir::RegionRange", "regions", "{}");
    auto *baseConstructor = genericAdaptorBase.addConstructor(paramList);
    baseConstructor->addMemberInitializer("odsAttrs", "attrs");
    if (useProperties)
      baseConstructor->addMemberInitializer("properties", "properties");
    baseConstructor->addMemberInitializer("odsRegions", "regions");

    MethodBody &body = baseConstructor->body();
    body.indent() << "if (odsAttrs)\n";
    body.indent() << formatv(
        "odsOpName.emplace(\"{0}\", odsAttrs.getContext());\n",
        op.getOperationName());

    paramList.insert(paramList.begin(), MethodParameter("RangeT", "values"));
    auto *constructor = genericAdaptor.addConstructor(paramList);
    constructor->addMemberInitializer("Base", "attrs, properties, regions");
    constructor->addMemberInitializer("odsOperands", "values");

    // Add a forwarding constructor to the previous one that accepts
    // OpaqueProperties instead and check for null and perform the cast to the
    // actual properties type.
    paramList[1] = MethodParameter("::mlir::DictionaryAttr", "attrs");
    paramList[2] = MethodParameter("::mlir::OpaqueProperties", "properties");
    auto *opaquePropertiesConstructor =
        genericAdaptor.addConstructor(std::move(paramList));
    if (useProperties) {
      opaquePropertiesConstructor->addMemberInitializer(
          genericAdaptor.getClassName(),
          "values, "
          "attrs, "
          "(properties ? *properties.as<Properties *>() : Properties{}), "
          "regions");
    } else {
      opaquePropertiesConstructor->addMemberInitializer(
          genericAdaptor.getClassName(),
          "values, "
          "attrs, "
          "(properties ? *properties.as<::mlir::EmptyProperties *>() : "
          "::mlir::EmptyProperties{}), "
          "regions");
    }
  }

  // Create constructors constructing the adaptor from an instance of the op.
  // This takes the attributes, properties and regions from the op instance
  // and the value range from the parameter.
  {
    // Base class is in the cpp file and can simply access the members of the op
    // class to initialize the template independent fields.
    auto *constructor = genericAdaptorBase.addConstructor(
        MethodParameter(op.getCppClassName(), "op"));
    constructor->addMemberInitializer(
        genericAdaptorBase.getClassName(),
        llvm::Twine(!useProperties ? "op->getAttrDictionary()"
                                   : "op->getDiscardableAttrDictionary()") +
            ", op.getProperties(), op->getRegions()");

    // Generic adaptor is templated and therefore defined inline in the header.
    // We cannot use the Op class here as it is an incomplete type (we have a
    // circular reference between the two).
    // Use a template trick to make the constructor be instantiated at call site
    // when the op class is complete.
    constructor = genericAdaptor.addConstructor(
        MethodParameter("RangeT", "values"), MethodParameter("LateInst", "op"));
    constructor->addTemplateParam("LateInst = " + op.getCppClassName());
    constructor->addTemplateParam(
        "= std::enable_if_t<std::is_same_v<LateInst, " + op.getCppClassName() +
        ">>");
    constructor->addMemberInitializer("Base", "op");
    constructor->addMemberInitializer("odsOperands", "values");
  }

  std::string sizeAttrInit;
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    if (op.getDialect().usePropertiesForAttributes())
      sizeAttrInit =
          formatv(adapterSegmentSizeAttrInitCodeProperties,
                  llvm::formatv("getProperties().operandSegmentSizes"));
    else
      sizeAttrInit = formatv(adapterSegmentSizeAttrInitCode,
                             emitHelper.getAttr(operandSegmentAttrName));
  }
  generateNamedOperandGetters(op, genericAdaptor,
                              /*genericAdaptorBase=*/&genericAdaptorBase,
                              /*sizeAttrInit=*/sizeAttrInit,
                              /*rangeType=*/"RangeT",
                              /*rangeElementType=*/"ValueT",
                              /*rangeBeginCall=*/"odsOperands.begin()",
                              /*rangeSizeCall=*/"odsOperands.size()",
                              /*getOperandCallPattern=*/"odsOperands[{0}]");

  // Any invalid overlap for `getOperands` will have been diagnosed before
  // here already.
  if (auto *m = genericAdaptor.addMethod("RangeT", "getOperands"))
    m->body() << "  return odsOperands;";

  FmtContext fctx;
  fctx.withBuilder("::mlir::Builder(odsAttrs.getContext())");

  // Generate named accessor with Attribute return type.
  auto emitAttrWithStorageType = [&](StringRef name, StringRef emitName,
                                     Attribute attr) {
    auto *method =
        genericAdaptorBase.addMethod(attr.getStorageType(), emitName + "Attr");
    ERROR_IF_PRUNED(method, "Adaptor::" + emitName + "Attr", op);
    auto &body = method->body().indent();
    if (!useProperties)
      body << "assert(odsAttrs && \"no attributes when constructing "
              "adapter\");\n";
    body << formatv(
        "auto attr = ::llvm::{1}<{2}>({0});\n", emitHelper.getAttr(name),
        attr.hasDefaultValue() || attr.isOptional() ? "dyn_cast_or_null"
                                                    : "cast",
        attr.getStorageType());

    if (attr.hasDefaultValue() && attr.isOptional()) {
      // Use the default value if attribute is not set.
      // TODO: this is inefficient, we are recreating the attribute for every
      // call. This should be set instead.
      std::string defaultValue = std::string(
          tgfmt(attr.getConstBuilderTemplate(), &fctx, attr.getDefaultValue()));
      body << "if (!attr)\n  attr = " << defaultValue << ";\n";
    }
    body << "return attr;\n";
  };

  if (useProperties) {
    auto *m = genericAdaptorBase.addInlineMethod("const Properties &",
                                                 "getProperties");
    ERROR_IF_PRUNED(m, "Adaptor::getProperties", op);
    m->body() << "  return properties;";
  }
  {
    auto *m =
        genericAdaptorBase.addMethod("::mlir::DictionaryAttr", "getAttributes");
    ERROR_IF_PRUNED(m, "Adaptor::getAttributes", op);
    m->body() << "  return odsAttrs;";
  }
  for (auto &namedAttr : op.getAttributes()) {
    const auto &name = namedAttr.name;
    const auto &attr = namedAttr.attr;
    if (attr.isDerivedAttr())
      continue;
    std::string emitName = op.getGetterName(name);
    emitAttrWithStorageType(name, emitName, attr);
    emitAttrGetterWithReturnType(fctx, genericAdaptorBase, op, emitName, attr);
  }

  unsigned numRegions = op.getNumRegions();
  for (unsigned i = 0; i < numRegions; ++i) {
    const auto &region = op.getRegion(i);
    if (region.name.empty())
      continue;

    // Generate the accessors for a variadic region.
    std::string name = op.getGetterName(region.name);
    if (region.isVariadic()) {
      auto *m = genericAdaptorBase.addMethod("::mlir::RegionRange", name);
      ERROR_IF_PRUNED(m, "Adaptor::" + name, op);
      m->body() << formatv("  return odsRegions.drop_front({0});", i);
      continue;
    }

    auto *m = genericAdaptorBase.addMethod("::mlir::Region &", name);
    ERROR_IF_PRUNED(m, "Adaptor::" + name, op);
    m->body() << formatv("  return *odsRegions[{0}];", i);
  }
  if (numRegions > 0) {
    // Any invalid overlap for `getRegions` will have been diagnosed before
    // here already.
    if (auto *m =
            genericAdaptorBase.addMethod("::mlir::RegionRange", "getRegions"))
      m->body() << "  return odsRegions;";
  }

  StringRef genericAdaptorClassName = genericAdaptor.getClassName();
  adaptor.addParent(ParentClass(genericAdaptorClassName))
      .addTemplateParam("::mlir::ValueRange");
  adaptor.declare<VisibilityDeclaration>(Visibility::Public);
  adaptor.declare<UsingDeclaration>(genericAdaptorClassName +
                                    "::" + genericAdaptorClassName);
  {
    // Constructor taking the Op as single parameter.
    auto *constructor =
        adaptor.addConstructor(MethodParameter(op.getCppClassName(), "op"));
    constructor->addMemberInitializer(genericAdaptorClassName,
                                      "op->getOperands(), op");
  }

  // Add verification function.
  addVerification();

  genericAdaptorBase.finalize();
  genericAdaptor.finalize();
  adaptor.finalize();
}

void OpOperandAdaptorEmitter::addVerification() {
  auto *method = adaptor.addMethod("::mlir::LogicalResult", "verify",
                                   MethodParameter("::mlir::Location", "loc"));
  ERROR_IF_PRUNED(method, "verify", op);
  auto &body = method->body();
  bool useProperties = emitHelper.hasProperties();

  FmtContext verifyCtx;
  populateSubstitutions(emitHelper, verifyCtx);
  genAttributeVerifier(emitHelper, verifyCtx, body, staticVerifierEmitter,
                       useProperties);

  body << "  return ::mlir::success();";
}

void OpOperandAdaptorEmitter::emitDecl(
    const Operator &op,
    const StaticVerifierFunctionEmitter &staticVerifierEmitter,
    raw_ostream &os) {
  OpOperandAdaptorEmitter emitter(op, staticVerifierEmitter);
  {
    NamespaceEmitter ns(os, "detail");
    emitter.genericAdaptorBase.writeDeclTo(os);
  }
  emitter.genericAdaptor.writeDeclTo(os);
  emitter.adaptor.writeDeclTo(os);
}

void OpOperandAdaptorEmitter::emitDef(
    const Operator &op,
    const StaticVerifierFunctionEmitter &staticVerifierEmitter,
    raw_ostream &os) {
  OpOperandAdaptorEmitter emitter(op, staticVerifierEmitter);
  {
    NamespaceEmitter ns(os, "detail");
    emitter.genericAdaptorBase.writeDefTo(os);
  }
  emitter.genericAdaptor.writeDefTo(os);
  emitter.adaptor.writeDefTo(os);
}

// Emits the opcode enum and op classes.
static void emitOpClasses(const RecordKeeper &recordKeeper,
                          const std::vector<Record *> &defs, raw_ostream &os,
                          bool emitDecl) {
  // First emit forward declaration for each class, this allows them to refer
  // to each others in traits for example.
  if (emitDecl) {
    os << "#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)\n";
    os << "#undef GET_OP_FWD_DEFINES\n";
    for (auto *def : defs) {
      Operator op(*def);
      NamespaceEmitter emitter(os, op.getCppNamespace());
      os << "class " << op.getCppClassName() << ";\n";
    }
    os << "#endif\n\n";
  }

  IfDefScope scope("GET_OP_CLASSES", os);
  if (defs.empty())
    return;

  // Generate all of the locally instantiated methods first.
  StaticVerifierFunctionEmitter staticVerifierEmitter(os, recordKeeper);
  os << formatv(opCommentHeader, "Local Utility Method", "Definitions");
  staticVerifierEmitter.emitOpConstraints(defs, emitDecl);

  for (auto *def : defs) {
    Operator op(*def);
    if (emitDecl) {
      {
        NamespaceEmitter emitter(os, op.getCppNamespace());
        os << formatv(opCommentHeader, op.getQualCppClassName(),
                      "declarations");
        OpOperandAdaptorEmitter::emitDecl(op, staticVerifierEmitter, os);
        OpEmitter::emitDecl(op, os, staticVerifierEmitter);
      }
      // Emit the TypeID explicit specialization to have a single definition.
      if (!op.getCppNamespace().empty())
        os << "MLIR_DECLARE_EXPLICIT_TYPE_ID(" << op.getCppNamespace()
           << "::" << op.getCppClassName() << ")\n\n";
    } else {
      {
        NamespaceEmitter emitter(os, op.getCppNamespace());
        os << formatv(opCommentHeader, op.getQualCppClassName(), "definitions");
        OpOperandAdaptorEmitter::emitDef(op, staticVerifierEmitter, os);
        OpEmitter::emitDef(op, os, staticVerifierEmitter);
      }
      // Emit the TypeID explicit specialization to have a single definition.
      if (!op.getCppNamespace().empty())
        os << "MLIR_DEFINE_EXPLICIT_TYPE_ID(" << op.getCppNamespace()
           << "::" << op.getCppClassName() << ")\n\n";
    }
  }
}

// Emits a comma-separated list of the ops.
static void emitOpList(const std::vector<Record *> &defs, raw_ostream &os) {
  IfDefScope scope("GET_OP_LIST", os);

  interleave(
      // TODO: We are constructing the Operator wrapper instance just for
      // getting it's qualified class name here. Reduce the overhead by having a
      // lightweight version of Operator class just for that purpose.
      defs, [&os](Record *def) { os << Operator(def).getQualCppClassName(); },
      [&os]() { os << ",\n"; });
}

static bool emitOpDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op Declarations", os);

  std::vector<Record *> defs = getRequestedOpDefinitions(recordKeeper);
  emitOpClasses(recordKeeper, defs, os, /*emitDecl=*/true);

  return false;
}

static bool emitOpDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op Definitions", os);

  std::vector<Record *> defs = getRequestedOpDefinitions(recordKeeper);
  emitOpList(defs, os);
  emitOpClasses(recordKeeper, defs, os, /*emitDecl=*/false);

  return false;
}

static mlir::GenRegistration
    genOpDecls("gen-op-decls", "Generate op declarations",
               [](const RecordKeeper &records, raw_ostream &os) {
                 return emitOpDecls(records, os);
               });

static mlir::GenRegistration genOpDefs("gen-op-defs", "Generate op definitions",
                                       [](const RecordKeeper &records,
                                          raw_ostream &os) {
                                         return emitOpDefs(records, os);
                                       });
