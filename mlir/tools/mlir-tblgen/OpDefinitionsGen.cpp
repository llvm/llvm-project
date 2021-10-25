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

#include "OpFormatGen.h"
#include "OpGenHelpers.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/OpClass.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/SideEffects.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Signals.h"
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

// The logic to calculate the actual value range for a declared operand/result
// of an op with variadic operands/results. Note that this logic is not for
// general use; it assumes all variadic operands/results must have the same
// number of values.
//
// {0}: The list of whether each declared operand/result is variadic.
// {1}: The total number of non-variadic operands/results.
// {2}: The total number of variadic operands/results.
// {3}: The total number of actual values.
// {4}: "operand" or "result".
const char *sameVariadicSizeValueRangeCalcCode = R"(
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

// The logic to calculate the actual value range for a declared operand/result
// of an op with variadic operands/results. Note that this logic is assumes
// the op has an attribute specifying the size of each operand/result segment
// (variadic or not).
//
// {0}: The name of the attribute specifying the segment sizes.
const char *adapterSegmentSizeAttrInitCode = R"(
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("{0}").cast<::mlir::DenseIntElementsAttr>();
)";
const char *opSegmentSizeAttrInitCode = R"(
  auto sizeAttr = (*this)->getAttr({0}).cast<::mlir::DenseIntElementsAttr>();
)";
const char *attrSizedSegmentValueRangeCalcCode = R"(
  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
)";
// The logic to calculate the actual value range for a declared operand
// of an op with variadic of variadic operands within the OpAdaptor.
//
// {0}: The name of the segment attribute.
// {1}: The index of the main operand.
const char *variadicOfVariadicAdaptorCalcCode = R"(
  auto tblgenTmpOperands = getODSOperands({1});
  auto sizeAttrValues = {0}().getValues<uint32_t>();
  auto sizeAttrIt = sizeAttrValues.begin();

  ::llvm::SmallVector<::mlir::ValueRange> tblgenTmpOperandGroups;
  for (int i = 0, e = ::llvm::size(sizeAttrValues); i < e; ++i, ++sizeAttrIt) {{
    tblgenTmpOperandGroups.push_back(tblgenTmpOperands.take_front(*sizeAttrIt));
    tblgenTmpOperands = tblgenTmpOperands.drop_front(*sizeAttrIt);
  }
  return tblgenTmpOperandGroups;
)";

// The logic to build a range of either operand or result values.
//
// {0}: The begin iterator of the actual values.
// {1}: The call to generate the start and length of the value range.
const char *valueRangeReturnCode = R"(
  auto valueRange = {1};
  return {{std::next({0}, valueRange.first),
           std::next({0}, valueRange.first + valueRange.second)};
)";

const char *typeVerifierSignature =
    "static ::mlir::LogicalResult {0}(::mlir::Operation *op, ::mlir::Type "
    "type, ::llvm::StringRef valueKind, unsigned valueGroupStartIndex)";

const char *typeVerifierErrorHandler =
    " op->emitOpError(valueKind) << \" #\" << valueGroupStartIndex << \" must "
    "be {0}, but got \" << type";

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
  auto valueInit = record.getValueInit(fieldName);
  return isa<StringInit>(valueInit);
}

static std::string getArgumentName(const Operator &op, int index) {
  const auto &operand = op.getOperand(index);
  if (!operand.name.empty())
    return std::string(operand.name);
  else
    return std::string(formatv("{0}_{1}", generatedArgName, index));
}

// Returns true if we can use unwrapped value for the given `attr` in builders.
static bool canUseUnwrappedRawValue(const tblgen::Attribute &attr) {
  return attr.getReturnType() != attr.getStorageType() &&
         // We need to wrap the raw value into an attribute in the builder impl
         // so we need to make sure that the attribute specifies how to do that.
         !attr.getConstBuilderTemplate().empty();
}

//===----------------------------------------------------------------------===//
// Op emitter
//===----------------------------------------------------------------------===//

namespace {
// Helper class to emit a record into the given output stream.
class OpEmitter {
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
  void buildParamList(llvm::SmallVectorImpl<OpMethodParameter> &paramList,
                      llvm::StringSet<> &inferredAttributes,
                      SmallVectorImpl<std::string> &resultTypeNames,
                      TypeParamKind typeParamKind,
                      AttrParamKind attrParamKind = AttrParamKind::WrappedAttr);

  // Adds op arguments and regions into operation state for build() methods.
  void
  genCodeForAddingArgAndRegionForBuilder(OpMethodBody &body,
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

  // Generates verify statements for operands and results in the operation.
  // The generated code will be attached to `body`.
  void genOperandResultVerifier(OpMethodBody &body,
                                Operator::value_range values,
                                StringRef valueKind);

  // Generates verify statements for regions in the operation.
  // The generated code will be attached to `body`.
  void genRegionVerifier(OpMethodBody &body);

  // Generates verify statements for successors in the operation.
  // The generated code will be attached to `body`.
  void genSuccessorVerifier(OpMethodBody &body);

  // Generates the traits used by the object.
  void genTraits();

  // Generate the OpInterface methods for all interfaces.
  void genOpInterfaceMethods();

  // Generate op interface methods for the given interface.
  void genOpInterfaceMethods(const tblgen::InterfaceTrait *trait);

  // Generate op interface method for the given interface method. If
  // 'declaration' is true, generates a declaration, else a definition.
  OpMethod *genOpInterfaceMethod(const tblgen::InterfaceMethod &method,
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
  Operator op;

  // The C++ code builder for this op
  OpClass opClass;

  // The format context for verification code generation.
  FmtContext verifyCtx;

  // The emitter containing all of the locally emitted verification functions.
  const StaticVerifierFunctionEmitter &staticVerifierEmitter;
};

} // end anonymous namespace

// Populate the format context `ctx` with substitutions of attributes, operands
// and results.
// - attrGet corresponds to the name of the function to call to get value of
//   attribute (the generated function call returns an Attribute);
// - operandGet corresponds to the name of the function with which to retrieve
//   an operand (the generated function call returns an OperandRange);
// - resultGet corresponds to the name of the function to get an result (the
//   generated function call returns a ValueRange);
static void populateSubstitutions(const Operator &op, const char *attrGet,
                                  const char *operandGet, const char *resultGet,
                                  FmtContext &ctx) {
  // Populate substitutions for attributes and named operands.
  for (const auto &namedAttr : op.getAttributes())
    ctx.addSubst(namedAttr.name,
                 formatv("{0}(\"{1}\")", attrGet, namedAttr.name));
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    auto &value = op.getOperand(i);
    if (value.name.empty())
      continue;

    if (value.isVariadic())
      ctx.addSubst(value.name, formatv("{0}({1})", operandGet, i));
    else
      ctx.addSubst(value.name, formatv("(*{0}({1}).begin())", operandGet, i));
  }

  // Populate substitutions for results.
  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    auto &value = op.getResult(i);
    if (value.name.empty())
      continue;

    if (value.isVariadic())
      ctx.addSubst(value.name, formatv("{0}({1})", resultGet, i));
    else
      ctx.addSubst(value.name, formatv("(*{0}({1}).begin())", resultGet, i));
  }
}

// Generate attribute verification. If emitVerificationRequiringOp is set then
// only verification for attributes whose value depend on op being known are
// emitted, else only verification that doesn't depend on the op being known are
// generated.
// - emitErrorPrefix is the prefix for the error emitting call which consists
//   of the entire function call up to start of error message fragment;
// - emitVerificationRequiringOp specifies whether verification should be
//   emitted for verification that require the op to exist;
static void genAttributeVerifier(const Operator &op, const char *attrGet,
                                 const Twine &emitErrorPrefix,
                                 bool emitVerificationRequiringOp,
                                 FmtContext &ctx, OpMethodBody &body) {
  for (const auto &namedAttr : op.getAttributes()) {
    const auto &attr = namedAttr.attr;
    if (attr.isDerivedAttr())
      continue;

    auto attrName = namedAttr.name;
    bool allowMissingAttr = attr.hasDefaultValue() || attr.isOptional();
    auto attrPred = attr.getPredicate();
    auto condition = attrPred.isNull() ? "" : attrPred.getCondition();
    // There is a condition to emit only if the use of $_op and whether to
    // emit verifications for op matches.
    bool hasConditionToEmit = (!(condition.find("$_op") != StringRef::npos) ^
                               emitVerificationRequiringOp);

    // Prefix with `tblgen_` to avoid hiding the attribute accessor.
    auto varName = tblgenNamePrefix + attrName;

    // If the attribute is
    //  1. Required (not allowed missing) and not in op verification, or
    //  2. Has a condition that will get verified
    // then the variable will be used.
    //
    // Therefore, for optional attributes whose verification requires that an
    // op already exists for verification/emitVerificationRequiringOp is set
    // has nothing that can be verified here.
    if ((allowMissingAttr || emitVerificationRequiringOp) &&
        !hasConditionToEmit)
      continue;

    body << formatv("  {\n  auto {0} = {1}(\"{2}\");\n", varName, attrGet,
                    attrName);

    if (!emitVerificationRequiringOp && !allowMissingAttr) {
      body << "  if (!" << varName << ") return " << emitErrorPrefix
           << "\"requires attribute '" << attrName << "'\");\n";
    }

    if (!hasConditionToEmit) {
      body << "  }\n";
      continue;
    }

    if (allowMissingAttr) {
      // If the attribute has a default value, then only verify the predicate if
      // set. This does effectively assume that the default value is valid.
      // TODO: verify the debug value is valid (perhaps in debug mode only).
      body << "  if (" << varName << ") {\n";
    }

    body << tgfmt("    if (!($0)) return $1\"attribute '$2' "
                  "failed to satisfy constraint: $3\");\n",
                  /*ctx=*/nullptr, tgfmt(condition, &ctx.withSelf(varName)),
                  emitErrorPrefix, attrName, escapeString(attr.getSummary()));
    if (allowMissingAttr)
      body << "  }\n";
    body << "  }\n";
  }
}

OpEmitter::OpEmitter(const Operator &op,
                     const StaticVerifierFunctionEmitter &staticVerifierEmitter)
    : def(op.getDef()), op(op),
      opClass(op.getCppClassName(), op.getExtraClassDeclaration()),
      staticVerifierEmitter(staticVerifierEmitter) {
  verifyCtx.withOp("(*this->getOperation())");
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
  genAttrGetters();
  genAttrSetters();
  genOptionalAttrRemovers();
  genBuilder();
  genParser();
  genPrinter();
  genVerifier();
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

void OpEmitter::emitDecl(raw_ostream &os) { opClass.writeDeclTo(os); }

void OpEmitter::emitDef(raw_ostream &os) { opClass.writeDefTo(os); }

static void errorIfPruned(size_t line, OpMethod *m, const Twine &methodName,
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
  // A map of attribute names (including implicit attributes) registered to the
  // current operation, to the relative order in which they were registered.
  llvm::MapVector<StringRef, unsigned> attributeNames;

  // Enumerate the attribute names of this op, assigning each a relative
  // ordering.
  auto addAttrName = [&](StringRef name) {
    unsigned index = attributeNames.size();
    attributeNames.insert({name, index});
  };
  for (const NamedAttribute &namedAttr : op.getAttributes())
    addAttrName(namedAttr.name);
  // Include key attributes from several traits as implicitly registered.
  std::string operandSizes = "operand_segment_sizes";
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments"))
    addAttrName(operandSizes);
  std::string attrSizes = "result_segment_sizes";
  if (op.getTrait("::mlir::OpTrait::AttrSizedResultSegments"))
    addAttrName(attrSizes);

  // Emit the getAttributeNames method.
  {
    auto *method = opClass.addMethodAndPrune(
        "::llvm::ArrayRef<::llvm::StringRef>", "getAttributeNames",
        OpMethod::Property(OpMethod::MP_Static | OpMethod::MP_Inline));
    ERROR_IF_PRUNED(method, "getAttributeNames", op);
    auto &body = method->body();
    if (attributeNames.empty()) {
      body << "  return {};";
    } else {
      body << "  static ::llvm::StringRef attrNames[] = {";
      llvm::interleaveComma(llvm::make_first_range(attributeNames), body,
                            [&](StringRef attrName) {
                              body << "::llvm::StringRef(\"" << attrName
                                   << "\")";
                            });
      body << "};\n  return ::llvm::makeArrayRef(attrNames);";
    }
  }
  if (attributeNames.empty())
    return;

  // Emit the getAttributeNameForIndex methods.
  {
    auto *method = opClass.addMethodAndPrune(
        "::mlir::Identifier", "getAttributeNameForIndex",
        OpMethod::Property(OpMethod::MP_Private | OpMethod::MP_Inline),
        "unsigned", "index");
    ERROR_IF_PRUNED(method, "getAttributeNameForIndex", op);
    method->body()
        << "  return getAttributeNameForIndex((*this)->getName(), index);";
  }
  {
    auto *method = opClass.addMethodAndPrune(
        "::mlir::Identifier", "getAttributeNameForIndex",
        OpMethod::Property(OpMethod::MP_Private | OpMethod::MP_Inline |
                           OpMethod::MP_Static),
        "::mlir::OperationName name, unsigned index");
    ERROR_IF_PRUNED(method, "getAttributeNameForIndex", op);
    method->body() << "assert(index < " << attributeNames.size()
                   << " && \"invalid attribute index\");\n"
                      "  return name.getAbstractOperation()"
                      "->getAttributeNames()[index];";
  }

  // Generate the <attr>AttrName methods, that expose the attribute names to
  // users.
  const char *attrNameMethodBody = "  return getAttributeNameForIndex({0});";
  for (const std::pair<StringRef, unsigned> &attrIt : attributeNames) {
    for (StringRef name : op.getGetterNames(attrIt.first)) {
      std::string methodName = (name + "AttrName").str();

      // Generate the non-static variant.
      {
        auto *method =
            opClass.addMethodAndPrune("::mlir::Identifier", methodName,
                                      OpMethod::Property(OpMethod::MP_Inline));
        ERROR_IF_PRUNED(method, methodName, op);
        method->body()
            << llvm::formatv(attrNameMethodBody, attrIt.second).str();
      }

      // Generate the static variant.
      {
        auto *method = opClass.addMethodAndPrune(
            "::mlir::Identifier", methodName,
            OpMethod::Property(OpMethod::MP_Inline | OpMethod::MP_Static),
            "::mlir::OperationName", "name");
        ERROR_IF_PRUNED(method, methodName, op);
        method->body() << llvm::formatv(attrNameMethodBody,
                                        "name, " + Twine(attrIt.second))
                              .str();
      }
    }
  }
}

void OpEmitter::genAttrGetters() {
  FmtContext fctx;
  fctx.withBuilder("::mlir::Builder((*this)->getContext())");

  // Emit the derived attribute body.
  auto emitDerivedAttr = [&](StringRef name, Attribute attr) {
    if (auto *method = opClass.addMethodAndPrune(attr.getReturnType(), name))
      method->body() << "  " << attr.getDerivedCodeBody() << "\n";
  };

  // Emit with return type specified.
  auto emitAttrWithReturnType = [&](StringRef name, Attribute attr) {
    auto *method = opClass.addMethodAndPrune(attr.getReturnType(), name);
    ERROR_IF_PRUNED(method, name, op);
    auto &body = method->body();
    body << "  auto attr = " << name << "Attr();\n";
    if (attr.hasDefaultValue()) {
      // Returns the default value if not set.
      // TODO: this is inefficient, we are recreating the attribute for every
      // call. This should be set instead.
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
  };

  // Generate named accessor with Attribute return type. This is a wrapper class
  // that allows referring to the attributes via accessors instead of having to
  // use the string interface for better compile time verification.
  auto emitAttrWithStorageType = [&](StringRef name, Attribute attr) {
    auto *method =
        opClass.addMethodAndPrune(attr.getStorageType(), (name + "Attr").str());
    if (!method)
      return;
    auto &body = method->body();
    body << "  return (*this)->getAttr(" << name << "AttrName()).template ";
    if (attr.isOptional() || attr.hasDefaultValue())
      body << "dyn_cast_or_null<";
    else
      body << "cast<";
    body << attr.getStorageType() << ">();";
  };

  for (const NamedAttribute &namedAttr : op.getAttributes()) {
    for (StringRef name : op.getGetterNames(namedAttr.name)) {
      if (namedAttr.attr.isDerivedAttr()) {
        emitDerivedAttr(name, namedAttr.attr);
      } else {
        emitAttrWithStorageType(name, namedAttr.attr);
        emitAttrWithReturnType(name, namedAttr.attr);
      }
    }
  }

  auto derivedAttrs = make_filter_range(op.getAttributes(),
                                        [](const NamedAttribute &namedAttr) {
                                          return namedAttr.attr.isDerivedAttr();
                                        });
  if (!derivedAttrs.empty()) {
    opClass.addTrait("::mlir::DerivedAttributeOpInterface::Trait");
    // Generate helper method to query whether a named attribute is a derived
    // attribute. This enables, for example, avoiding adding an attribute that
    // overlaps with a derived attribute.
    {
      auto *method = opClass.addMethodAndPrune("bool", "isDerivedAttribute",
                                               OpMethod::MP_Static,
                                               "::llvm::StringRef", "name");
      ERROR_IF_PRUNED(method, "isDerivedAttribute", op);
      auto &body = method->body();
      for (auto namedAttr : derivedAttrs)
        body << "  if (name == \"" << namedAttr.name << "\") return true;\n";
      body << " return false;";
    }
    // Generate method to materialize derived attributes as a DictionaryAttr.
    {
      auto *method = opClass.addMethodAndPrune("::mlir::DictionaryAttr",
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
                                     .addSubst("_ctx", "ctx"))
                 << "}";
          },
          ",\n");
      body << "});";
    }
  }
}

void OpEmitter::genAttrSetters() {
  // Generate raw named setter type. This is a wrapper class that allows setting
  // to the attributes via setters instead of having to use the string interface
  // for better compile time verification.
  auto emitAttrWithStorageType = [&](StringRef setterName, StringRef getterName,
                                     Attribute attr) {
    auto *method = opClass.addMethodAndPrune(
        "void", (setterName + "Attr").str(), attr.getStorageType(), "attr");
    if (method)
      method->body() << "  (*this)->setAttr(" << getterName
                     << "AttrName(), attr);";
  };

  for (const NamedAttribute &namedAttr : op.getAttributes()) {
    if (!namedAttr.attr.isDerivedAttr())
      for (auto names : llvm::zip(op.getSetterNames(namedAttr.name),
                                  op.getGetterNames(namedAttr.name)))
        emitAttrWithStorageType(std::get<0>(names), std::get<1>(names),
                                namedAttr.attr);
  }
}

void OpEmitter::genOptionalAttrRemovers() {
  // Generate methods for removing optional attributes, instead of having to
  // use the string interface. Enables better compile time verification.
  auto emitRemoveAttr = [&](StringRef name) {
    auto upperInitial = name.take_front().upper();
    auto suffix = name.drop_front();
    auto *method = opClass.addMethodAndPrune(
        "::mlir::Attribute", ("remove" + upperInitial + suffix + "Attr").str());
    if (!method)
      return;
    method->body() << "  return (*this)->removeAttr(" << op.getGetterName(name)
                   << "AttrName());";
  };

  for (const NamedAttribute &namedAttr : op.getAttributes())
    if (namedAttr.attr.isOptional())
      emitRemoveAttr(namedAttr.name);
}

// Generates the code to compute the start and end index of an operand or result
// range.
template <typename RangeT>
static void
generateValueRangeStartAndEnd(Class &opClass, StringRef methodName,
                              int numVariadic, int numNonVariadic,
                              StringRef rangeSizeCall, bool hasAttrSegmentSize,
                              StringRef sizeAttrInit, RangeT &&odsValues) {
  auto *method = opClass.addMethodAndPrune("std::pair<unsigned, unsigned>",
                                           methodName, "unsigned", "index");
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
    llvm::SmallVector<StringRef, 4> isVariadic;
    isVariadic.reserve(llvm::size(odsValues));
    for (auto &it : odsValues)
      isVariadic.push_back(it.isVariableLength() ? "true" : "false");
    std::string isVariadicList = llvm::join(isVariadic, ", ");
    body << formatv(sameVariadicSizeValueRangeCalcCode, isVariadicList,
                    numNonVariadic, numVariadic, rangeSizeCall, "operand");
  }
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
static void generateNamedOperandGetters(const Operator &op, Class &opClass,
                                        bool isAdaptor, StringRef sizeAttrInit,
                                        StringRef rangeType,
                                        StringRef rangeBeginCall,
                                        StringRef rangeSizeCall,
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
  generateValueRangeStartAndEnd(opClass, "getODSOperandIndexAndLength",
                                numVariadicOperands, numNormalOperands,
                                rangeSizeCall, attrSizedOperands, sizeAttrInit,
                                const_cast<Operator &>(op).getOperands());

  auto *m = opClass.addMethodAndPrune(rangeType, "getODSOperands", "unsigned",
                                      "index");
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
    for (StringRef name : op.getGetterNames(operand.name)) {
      if (operand.isOptional()) {
        m = opClass.addMethodAndPrune("::mlir::Value", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << "  auto operands = getODSOperands(" << i << ");\n"
                  << "  return operands.empty() ? ::mlir::Value() : "
                     "*operands.begin();";
      } else if (operand.isVariadicOfVariadic()) {
        std::string segmentAttr = op.getGetterName(
            operand.constraint.getVariadicOfVariadicSegmentSizeAttr());
        if (isAdaptor) {
          m = opClass.addMethodAndPrune(
              "::llvm::SmallVector<::mlir::ValueRange>", name);
          ERROR_IF_PRUNED(m, name, op);
          m->body() << llvm::formatv(variadicOfVariadicAdaptorCalcCode,
                                     segmentAttr, i);
          continue;
        }

        m = opClass.addMethodAndPrune("::mlir::OperandRangeRange", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << "  return getODSOperands(" << i << ").split("
                  << segmentAttr << "Attr());";
      } else if (operand.isVariadic()) {
        m = opClass.addMethodAndPrune(rangeType, name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << "  return getODSOperands(" << i << ");";
      } else {
        m = opClass.addMethodAndPrune("::mlir::Value", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << "  return *getODSOperands(" << i << ").begin();";
      }
    }
  }
}

void OpEmitter::genNamedOperandGetters() {
  // Build the code snippet used for initializing the operand_segment_size)s
  // array.
  std::string attrSizeInitCode;
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    std::string attr = op.getGetterName("operand_segment_sizes") + "AttrName()";
    attrSizeInitCode = formatv(opSegmentSizeAttrInitCode, attr).str();
  }

  generateNamedOperandGetters(
      op, opClass,
      /*isAdaptor=*/false,
      /*sizeAttrInit=*/attrSizeInitCode,
      /*rangeType=*/"::mlir::Operation::operand_range",
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
    for (StringRef name : op.getGetterNames(operand.name)) {
      auto *m = opClass.addMethodAndPrune(
          operand.isVariadicOfVariadic() ? "::mlir::MutableOperandRangeRange"
                                         : "::mlir::MutableOperandRange",
          (name + "Mutable").str());
      ERROR_IF_PRUNED(m, name, op);
      auto &body = m->body();
      body << "  auto range = getODSOperandIndexAndLength(" << i << ");\n"
           << "  auto mutableRange = "
              "::mlir::MutableOperandRange(getOperation(), "
              "range.first, range.second";
      if (attrSizedOperands)
        body << ", ::mlir::MutableOperandRange::OperandSegment(" << i
             << "u, *getOperation()->getAttrDictionary().getNamed("
             << op.getGetterName("operand_segment_sizes") << "AttrName()))";
      body << ");\n";

      // If this operand is a nested variadic, we split the range into a
      // MutableOperandRangeRange that provides a range over all of the
      // sub-ranges.
      if (operand.isVariadicOfVariadic()) {
        //
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
    std::string attr = op.getGetterName("result_segment_sizes") + "AttrName()";
    attrSizeInitCode = formatv(opSegmentSizeAttrInitCode, attr).str();
  }

  generateValueRangeStartAndEnd(
      opClass, "getODSResultIndexAndLength", numVariadicResults,
      numNormalResults, "getOperation()->getNumResults()", attrSizedResults,
      attrSizeInitCode, op.getResults());

  auto *m = opClass.addMethodAndPrune("::mlir::Operation::result_range",
                                      "getODSResults", "unsigned", "index");
  ERROR_IF_PRUNED(m, "getODSResults", op);
  m->body() << formatv(valueRangeReturnCode, "getOperation()->result_begin()",
                       "getODSResultIndexAndLength(index)");

  for (int i = 0; i != numResults; ++i) {
    const auto &result = op.getResult(i);
    if (result.name.empty())
      continue;
    for (StringRef name : op.getGetterNames(result.name)) {
      if (result.isOptional()) {
        m = opClass.addMethodAndPrune("::mlir::Value", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body()
            << "  auto results = getODSResults(" << i << ");\n"
            << "  return results.empty() ? ::mlir::Value() : *results.begin();";
      } else if (result.isVariadic()) {
        m = opClass.addMethodAndPrune("::mlir::Operation::result_range", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << "  return getODSResults(" << i << ");";
      } else {
        m = opClass.addMethodAndPrune("::mlir::Value", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << "  return *getODSResults(" << i << ").begin();";
      }
    }
  }
}

void OpEmitter::genNamedRegionGetters() {
  unsigned numRegions = op.getNumRegions();
  for (unsigned i = 0; i < numRegions; ++i) {
    const auto &region = op.getRegion(i);
    if (region.name.empty())
      continue;

    for (StringRef name : op.getGetterNames(region.name)) {
      // Generate the accessors for a variadic region.
      if (region.isVariadic()) {
        auto *m = opClass.addMethodAndPrune(
            "::mlir::MutableArrayRef<::mlir::Region>", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << formatv("  return (*this)->getRegions().drop_front({0});",
                             i);
        continue;
      }

      auto *m = opClass.addMethodAndPrune("::mlir::Region &", name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << formatv("  return (*this)->getRegion({0});", i);
    }
  }
}

void OpEmitter::genNamedSuccessorGetters() {
  unsigned numSuccessors = op.getNumSuccessors();
  for (unsigned i = 0; i < numSuccessors; ++i) {
    const NamedSuccessor &successor = op.getSuccessor(i);
    if (successor.name.empty())
      continue;

    for (StringRef name : op.getGetterNames(successor.name)) {
      // Generate the accessors for a variadic successor list.
      if (successor.isVariadic()) {
        auto *m = opClass.addMethodAndPrune("::mlir::SuccessorRange", name);
        ERROR_IF_PRUNED(m, name, op);
        m->body() << formatv(
            "  return {std::next((*this)->successor_begin(), {0}), "
            "(*this)->successor_end()};",
            i);
        continue;
      }

      auto *m = opClass.addMethodAndPrune("::mlir::Block *", name);
      ERROR_IF_PRUNED(m, name, op);
      m->body() << formatv("  return (*this)->getSuccessor({0});", i);
    }
  }
}

static bool canGenerateUnwrappedBuilder(Operator &op) {
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
    NamedAttribute &namedAttr = op.getAttribute(i);
    if (canUseUnwrappedRawValue(namedAttr.attr)) {
      canGenerate = true;
      break;
    }
  }
  return canGenerate;
}

static bool canInferType(Operator &op) {
  return op.getTrait("::mlir::InferTypeOpInterface::Trait") &&
         op.getNumRegions() == 0;
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
    llvm::SmallVector<OpMethodParameter, 4> paramList;
    llvm::SmallVector<std::string, 4> resultNames;
    llvm::StringSet<> inferredAttributes;
    buildParamList(paramList, inferredAttributes, resultNames, paramKind,
                   attrType);

    auto *m = opClass.addMethodAndPrune("void", "build", OpMethod::MP_Static,
                                        std::move(paramList));
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
      // TODO: Expand to handle regions.
      body << formatv(R"(
        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded({0}::inferReturnTypes(odsBuilder.getContext(),
                      {1}.location, {1}.operands,
                      {1}.attributes.getDictionary({1}.getContext()),
                      /*regions=*/{{}, inferredReturnTypes)))
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
      return;
    case TypeParamKind::Collective: {
      int numResults = op.getNumResults();
      int numVariadicResults = op.getNumVariableLengthResults();
      int numNonVariadicResults = numResults - numVariadicResults;
      bool hasVariadicResult = numVariadicResults != 0;

      // Avoid emitting "resultTypes.size() >= 0u" which is always true.
      if (!(hasVariadicResult && numNonVariadicResults == 0))
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
  llvm::SmallVector<OpMethodParameter, 4> paramList;
  paramList.emplace_back("::mlir::OpBuilder &", "odsBuilder");
  paramList.emplace_back("::mlir::OperationState &", builderOpState);
  paramList.emplace_back("::mlir::ValueRange", "operands");
  // Provide default value for `attributes` when its the last parameter
  StringRef attributesDefaultValue = op.getNumVariadicRegions() ? "" : "{}";
  paramList.emplace_back("::llvm::ArrayRef<::mlir::NamedAttribute>",
                         "attributes", attributesDefaultValue);
  if (op.getNumVariadicRegions())
    paramList.emplace_back("unsigned", "numRegions");

  auto *m = opClass.addMethodAndPrune("void", "build", OpMethod::MP_Static,
                                      std::move(paramList));
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

void OpEmitter::genInferredTypeCollectiveParamBuilder() {
  // TODO: Expand to support regions.
  SmallVector<OpMethodParameter, 4> paramList;
  paramList.emplace_back("::mlir::OpBuilder &", "odsBuilder");
  paramList.emplace_back("::mlir::OperationState &", builderOpState);
  paramList.emplace_back("::mlir::ValueRange", "operands");
  paramList.emplace_back("::llvm::ArrayRef<::mlir::NamedAttribute>",
                         "attributes", "{}");
  auto *m = opClass.addMethodAndPrune("void", "build", OpMethod::MP_Static,
                                      std::move(paramList));
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
    ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
    if (::mlir::succeeded({0}::inferReturnTypes(odsBuilder.getContext(),
                  {1}.location, operands,
                  {1}.attributes.getDictionary({1}.getContext()),
                  /*regions=*/{{}, inferredReturnTypes))) {{)",
                  opClass.getClassName(), builderOpState);
  if (numVariadicResults == 0 || numNonVariadicResults != 0)
    body << "  assert(inferredReturnTypes.size()"
         << (numVariadicResults != 0 ? " >= " : " == ") << numNonVariadicResults
         << "u && \"mismatched number of return types\");\n";
  body << "      " << builderOpState << ".addTypes(inferredReturnTypes);";

  body << formatv(R"(
    } else
      ::llvm::report_fatal_error("Failed to infer result type(s).");)",
                  opClass.getClassName(), builderOpState);
}

void OpEmitter::genUseOperandAsResultTypeSeparateParamBuilder() {
  llvm::SmallVector<OpMethodParameter, 4> paramList;
  llvm::SmallVector<std::string, 4> resultNames;
  llvm::StringSet<> inferredAttributes;
  buildParamList(paramList, inferredAttributes, resultNames,
                 TypeParamKind::None);

  auto *m = opClass.addMethodAndPrune("void", "build", OpMethod::MP_Static,
                                      std::move(paramList));
  // If the builder is redundant, skip generating the method
  if (!m)
    return;
  auto &body = m->body();
  genCodeForAddingArgAndRegionForBuilder(body, inferredAttributes);

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
}

void OpEmitter::genUseAttrAsResultTypeBuilder() {
  SmallVector<OpMethodParameter, 4> paramList;
  paramList.emplace_back("::mlir::OpBuilder &", "odsBuilder");
  paramList.emplace_back("::mlir::OperationState &", builderOpState);
  paramList.emplace_back("::mlir::ValueRange", "operands");
  paramList.emplace_back("::llvm::ArrayRef<::mlir::NamedAttribute>",
                         "attributes", "{}");
  auto *m = opClass.addMethodAndPrune("void", "build", OpMethod::MP_Static,
                                      std::move(paramList));
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
          "    if (attr.first != attrName) continue;\n";
  if (namedAttr.attr.isTypeAttr()) {
    resultType = "attr.second.cast<::mlir::TypeAttr>().getValue()";
  } else {
    resultType = "attr.second.getType()";
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
static std::string getBuilderSignature(const Builder &builder) {
  ArrayRef<Builder::Parameter> params(builder.getParameters());

  // Inject builder and state arguments.
  llvm::SmallVector<std::string, 8> arguments;
  arguments.reserve(params.size() + 2);
  arguments.push_back(
      llvm::formatv("::mlir::OpBuilder &{0}", odsBuilder).str());
  arguments.push_back(
      llvm::formatv("::mlir::OperationState &{0}", builderOpState).str());

  for (unsigned i = 0, e = params.size(); i < e; ++i) {
    // If no name is provided, generate one.
    Optional<StringRef> paramName = params[i].getName();
    std::string name =
        paramName ? paramName->str() : "odsArg" + std::to_string(i);

    std::string defaultValue;
    if (Optional<StringRef> defaultParamValue = params[i].getDefaultValue())
      defaultValue = llvm::formatv(" = {0}", *defaultParamValue).str();
    arguments.push_back(
        llvm::formatv("{0} {1}{2}", params[i].getCppType(), name, defaultValue)
            .str());
  }

  return llvm::join(arguments, ", ");
}

void OpEmitter::genBuilder() {
  // Handle custom builders if provided.
  for (const Builder &builder : op.getBuilders()) {
    std::string paramStr = getBuilderSignature(builder);

    Optional<StringRef> body = builder.getBody();
    OpMethod::Property properties =
        body ? OpMethod::MP_Static : OpMethod::MP_StaticDeclaration;
    auto *method =
        opClass.addMethodAndPrune("void", "build", properties, paramStr);
    if (body)
      ERROR_IF_PRUNED(method, "build", op);

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

  SmallVector<OpMethodParameter, 4> paramList;
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

  auto *m = opClass.addMethodAndPrune("void", "build", OpMethod::MP_Static,
                                      std::move(paramList));
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
  // TODO: Expand to handle regions and successors.
  if (canInferType(op) && op.getNumSuccessors() == 0)
    genInferredTypeCollectiveParamBuilder();
}

void OpEmitter::buildParamList(SmallVectorImpl<OpMethodParameter> &paramList,
                               llvm::StringSet<> &inferredAttributes,
                               SmallVectorImpl<std::string> &resultTypeNames,
                               TypeParamKind typeParamKind,
                               AttrParamKind attrParamKind) {
  resultTypeNames.clear();
  auto numResults = op.getNumResults();
  resultTypeNames.reserve(numResults);

  paramList.emplace_back("::mlir::OpBuilder &", "odsBuilder");
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
      OpMethodParameter::Property properties = OpMethodParameter::PP_None;
      if (result.isOptional())
        properties = OpMethodParameter::PP_Optional;

      paramList.emplace_back(type, resultName, properties);
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
      auto *namedAttr = op.getArg(i).dyn_cast<tblgen::NamedAttribute *>();
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
    if (const auto *operand = arg.dyn_cast<NamedTypeConstraint *>()) {
      StringRef type;
      if (operand->isVariadicOfVariadic())
        type = "::llvm::ArrayRef<::mlir::ValueRange>";
      else if (operand->isVariadic())
        type = "::mlir::ValueRange";
      else
        type = "::mlir::Value";

      OpMethodParameter::Property properties = OpMethodParameter::PP_None;
      if (operand->isOptional())
        properties = OpMethodParameter::PP_Optional;
      paramList.emplace_back(type, getArgumentName(op, numOperands++),
                             properties);
      continue;
    }
    const NamedAttribute &namedAttr = *arg.get<NamedAttribute *>();
    const Attribute &attr = namedAttr.attr;

    // inferred attributes don't need to be added to the param list.
    if (inferredAttributes.contains(namedAttr.name))
      continue;

    OpMethodParameter::Property properties = OpMethodParameter::PP_None;
    if (attr.isOptional())
      properties = OpMethodParameter::PP_Optional;

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
    paramList.emplace_back(type, namedAttr.name, defaultValue, properties);
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
    OpMethodBody &body, llvm::StringSet<> &inferredAttributes,
    bool isRawValueAttr) {
  // Push all operands to the result.
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    std::string argName = getArgumentName(op, i);
    NamedTypeConstraint &operand = op.getOperand(i);
    if (operand.constraint.isVariadicOfVariadic()) {
      body << "  for (::mlir::ValueRange range : " << argName << ")\n   "
           << builderOpState << ".addOperands(range);\n";

      // Add the segment attribute.
      body << "  {\n"
           << "    SmallVector<int32_t> rangeSegments;\n"
           << "    for (::mlir::ValueRange range : " << argName << ")\n"
           << "      rangeSegments.push_back(range.size());\n"
           << "    " << builderOpState << ".addAttribute("
           << op.getGetterName(
                  operand.constraint.getVariadicOfVariadicSegmentSizeAttr())
           << "AttrName(" << builderOpState << ".name), " << odsBuilder
           << ".getI32TensorAttr(rangeSegments));"
           << "  }\n";
      continue;
    }

    if (operand.isOptional())
      body << "  if (" << argName << ")\n  ";
    body << "  " << builderOpState << ".addOperands(" << argName << ");\n";
  }

  // If the operation has the operand segment size attribute, add it here.
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    std::string sizes = op.getGetterName("operand_segment_sizes");
    body << "  " << builderOpState << ".addAttribute(" << sizes << "AttrName("
         << builderOpState << ".name), "
         << "odsBuilder.getI32VectorAttr({";
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
    body << "}));\n";
  }

  // Push all attributes to the result.
  for (const auto &namedAttr : op.getAttributes()) {
    auto &attr = namedAttr.attr;
    if (attr.isDerivedAttr() || inferredAttributes.contains(namedAttr.name))
      continue;

    bool emitNotNullCheck = attr.isOptional();
    if (emitNotNullCheck)
      body << formatv("  if ({0}) ", namedAttr.name) << "{\n";

    if (isRawValueAttr && canUseUnwrappedRawValue(attr)) {
      // If this is a raw value, then we need to wrap it in an Attribute
      // instance.
      FmtContext fctx;
      fctx.withBuilder("odsBuilder");

      std::string builderTemplate = std::string(attr.getConstBuilderTemplate());

      // For StringAttr, its constant builder call will wrap the input in
      // quotes, which is correct for normal string literals, but incorrect
      // here given we use function arguments. So we need to strip the
      // wrapping quotes.
      if (StringRef(builderTemplate).contains("\"$0\""))
        builderTemplate = replaceAllSubstrs(builderTemplate, "\"$0\"", "$0");

      std::string value =
          std::string(tgfmt(builderTemplate, &fctx, namedAttr.name));
      body << formatv("  {0}.addAttribute({1}AttrName({0}.name), {2});\n",
                      builderOpState, op.getGetterName(namedAttr.name), value);
    } else {
      body << formatv("  {0}.addAttribute({1}AttrName({0}.name), {2});\n",
                      builderOpState, op.getGetterName(namedAttr.name),
                      namedAttr.name);
    }
    if (emitNotNullCheck)
      body << "  }\n";
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
    SmallVector<OpMethodParameter, 2> paramList;
    paramList.emplace_back(op.getCppClassName(), "op");
    paramList.emplace_back("::mlir::PatternRewriter &", "rewriter");
    auto *m = opClass.addMethodAndPrune("::mlir::LogicalResult", "canonicalize",
                                        OpMethod::MP_StaticDeclaration,
                                        std::move(paramList));
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
  SmallVector<OpMethodParameter, 2> paramList;
  paramList.emplace_back("::mlir::RewritePatternSet &", "results");
  paramList.emplace_back("::mlir::MLIRContext *", "context");
  auto kind = hasBody ? OpMethod::MP_Static : OpMethod::MP_StaticDeclaration;
  auto *method = opClass.addMethodAndPrune(
      "void", "getCanonicalizationPatterns", kind, std::move(paramList));

  // If synthesizing the method, fill it it.
  if (hasBody) {
    ERROR_IF_PRUNED(method, "getCanonicalizationPatterns", op);
    method->body() << "  results.add(canonicalize);\n";
  }
}

void OpEmitter::genFolderDecls() {
  bool hasSingleResult =
      op.getNumResults() == 1 && op.getNumVariableLengthResults() == 0;

  if (def.getValueAsBit("hasFolder")) {
    if (hasSingleResult) {
      auto *m = opClass.addMethodAndPrune(
          "::mlir::OpFoldResult", "fold", OpMethod::MP_Declaration,
          "::llvm::ArrayRef<::mlir::Attribute>", "operands");
      ERROR_IF_PRUNED(m, "operands", op);
    } else {
      SmallVector<OpMethodParameter, 2> paramList;
      paramList.emplace_back("::llvm::ArrayRef<::mlir::Attribute>", "operands");
      paramList.emplace_back("::llvm::SmallVectorImpl<::mlir::OpFoldResult> &",
                             "results");
      auto *m = opClass.addMethodAndPrune("::mlir::LogicalResult", "fold",
                                          OpMethod::MP_Declaration,
                                          std::move(paramList));
      ERROR_IF_PRUNED(m, "fold", op);
    }
  }
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

OpMethod *OpEmitter::genOpInterfaceMethod(const InterfaceMethod &method,
                                          bool declaration) {
  SmallVector<OpMethodParameter, 4> paramList;
  for (const InterfaceMethod::Argument &arg : method.getArguments())
    paramList.emplace_back(arg.type, arg.name);

  auto properties = method.isStatic() ? OpMethod::MP_Static : OpMethod::MP_None;
  if (declaration)
    properties =
        static_cast<OpMethod::Property>(properties | OpMethod::MP_Declaration);
  return opClass.addMethodAndPrune(method.getReturnType(), method.getName(),
                                   properties, std::move(paramList));
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
    unsigned index : 30;

    /// The kind of the location.
    unsigned kind : 2;
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
    std::string type = llvm::formatv("::mlir::SmallVectorImpl<::mlir::"
                                     "SideEffects::EffectInstance<{0}>> &",
                                     it.first())
                           .str();
    auto *getEffects =
        opClass.addMethodAndPrune("void", "getEffects", type, "effects");
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
        if (attr->attr.isOptional()) {
          body << "  if (auto symbolRef = " << attr->name << "Attr())\n  "
               << llvm::formatv(addEffectCode, effect, "symbolRef, ", resource)
                      .str();
        } else {
          body << llvm::formatv(addEffectCode, effect, attr->name + "(), ",
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
  const auto *trait = dyn_cast<InterfaceTrait>(
      op.getTrait("::mlir::InferTypeOpInterface::Trait"));
  Interface interface = trait->getInterface();
  OpMethod *method = [&]() -> OpMethod * {
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
  body << "  ::mlir::Builder odsBuilder(context);\n";

  auto emitType =
      [&](const tblgen::Operator::ArgOrType &type) -> OpMethodBody & {
    if (type.isArg()) {
      auto argIndex = type.getArg();
      assert(!op.getArg(argIndex).is<NamedAttribute *>());
      auto arg = op.getArgToOperandOrAttribute(argIndex);
      if (arg.kind() == Operator::OperandOrAttribute::Kind::Operand)
        return body << "operands[" << arg.operandOrAttributeIndex()
                    << "].getType()";
      return body << "attributes[" << arg.operandOrAttributeIndex()
                  << "].getType()";
    } else {
      return body << tgfmt(*type.getType().getBuilderCall(), &fctx);
    }
  };

  for (int i = 0, e = op.getNumResults(); i != e; ++i) {
    body << "  inferredReturnTypes[" << i << "] = ";
    auto types = op.getSameTypeAsResult(i);
    emitType(types[0]) << ";\n";
    if (types.size() == 1)
      continue;
    // TODO: We could verify equality here, but skipping that for verification.
  }
  body << "  return ::mlir::success();";
}

void OpEmitter::genParser() {
  if (!hasStringAttribute(def, "parser") ||
      hasStringAttribute(def, "assemblyFormat"))
    return;

  SmallVector<OpMethodParameter, 2> paramList;
  paramList.emplace_back("::mlir::OpAsmParser &", "parser");
  paramList.emplace_back("::mlir::OperationState &", "result");
  auto *method =
      opClass.addMethodAndPrune("::mlir::ParseResult", "parse",
                                OpMethod::MP_Static, std::move(paramList));
  ERROR_IF_PRUNED(method, "parse", op);

  FmtContext fctx;
  fctx.addSubst("cppClass", opClass.getClassName());
  auto parser = def.getValueAsString("parser").ltrim().rtrim(" \t\v\f\r");
  method->body() << "  " << tgfmt(parser, &fctx);
}

void OpEmitter::genPrinter() {
  if (hasStringAttribute(def, "assemblyFormat"))
    return;

  auto valueInit = def.getValueInit("printer");
  StringInit *stringInit = dyn_cast<StringInit>(valueInit);
  if (!stringInit)
    return;

  auto *method =
      opClass.addMethodAndPrune("void", "print", "::mlir::OpAsmPrinter &", "p");
  ERROR_IF_PRUNED(method, "print", op);
  FmtContext fctx;
  fctx.addSubst("cppClass", opClass.getClassName());
  auto printer = stringInit->getValue().ltrim().rtrim(" \t\v\f\r");
  method->body() << "  " << tgfmt(printer, &fctx);
}

void OpEmitter::genVerifier() {
  auto *method = opClass.addMethodAndPrune("::mlir::LogicalResult", "verify");
  ERROR_IF_PRUNED(method, "verify", op);
  auto &body = method->body();
  body << "  if (::mlir::failed(" << op.getAdaptorName()
       << "(*this).verify((*this)->getLoc()))) "
       << "return ::mlir::failure();\n";

  auto *valueInit = def.getValueInit("verifier");
  StringInit *stringInit = dyn_cast<StringInit>(valueInit);
  bool hasCustomVerify = stringInit && !stringInit->getValue().empty();
  populateSubstitutions(op, "(*this)->getAttr", "this->getODSOperands",
                        "this->getODSResults", verifyCtx);

  genAttributeVerifier(op, "(*this)->getAttr", "emitOpError(",
                       /*emitVerificationRequiringOp=*/true, verifyCtx, body);
  genOperandResultVerifier(body, op.getOperands(), "operand");
  genOperandResultVerifier(body, op.getResults(), "result");

  for (auto &trait : op.getTraits()) {
    if (auto *t = dyn_cast<tblgen::PredTrait>(&trait)) {
      body << tgfmt("  if (!($0))\n    "
                    "return emitOpError(\"failed to verify that $1\");\n",
                    &verifyCtx, tgfmt(t->getPredTemplate(), &verifyCtx),
                    t->getSummary());
    }
  }

  genRegionVerifier(body);
  genSuccessorVerifier(body);

  if (hasCustomVerify) {
    FmtContext fctx;
    fctx.addSubst("cppClass", opClass.getClassName());
    auto printer = stringInit->getValue().ltrim().rtrim(" \t\v\f\r");
    body << "  " << tgfmt(printer, &fctx);
  } else {
    body << "  return ::mlir::success();\n";
  }
}

void OpEmitter::genOperandResultVerifier(OpMethodBody &body,
                                         Operator::value_range values,
                                         StringRef valueKind) {
  FmtContext fctx;

  body << "  {\n";
  body << "    unsigned index = 0; (void)index;\n";

  for (auto staticValue : llvm::enumerate(values)) {
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
      body << formatv("    if (valueGroup{0}.size() > 1)\n"
                      "      return emitOpError(\"{1} group starting at #\") "
                      "<< index << \" requires 0 or 1 element, but found \" << "
                      "valueGroup{0}.size();\n",
                      staticValue.index(), valueKind);
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
    body << "    for (::mlir::Value v : valueGroup" << staticValue.index()
         << ") {\n"
         << "      if (::mlir::failed(" << constraintFn
         << "(getOperation(), v.getType(), \"" << valueKind << "\", index)))\n"
         << "        return ::mlir::failure();\n"
         << "      ++index;\n"
         << "    }\n";
  }

  body << "  }\n";
}

void OpEmitter::genRegionVerifier(OpMethodBody &body) {
  // If we have no regions, there is nothing more to do.
  unsigned numRegions = op.getNumRegions();
  if (numRegions == 0)
    return;

  body << "{\n";
  body << "    unsigned index = 0; (void)index;\n";

  for (unsigned i = 0; i < numRegions; ++i) {
    const auto &region = op.getRegion(i);
    if (region.constraint.getPredicate().isNull())
      continue;

    body << "    for (::mlir::Region &region : ";
    body << formatv(region.isVariadic()
                        ? "{0}()"
                        : "::mlir::MutableArrayRef<::mlir::Region>((*this)"
                          "->getRegion({1}))",
                    op.getGetterName(region.name), i);
    body << ") {\n";
    auto constraint = tgfmt(region.constraint.getConditionTemplate(),
                            &verifyCtx.withSelf("region"))
                          .str();

    body << formatv("      (void)region;\n"
                    "      if (!({0})) {\n        "
                    "return emitOpError(\"region #\") << index << \" {1}"
                    "failed to "
                    "verify constraint: {2}\";\n      }\n",
                    constraint,
                    region.name.empty() ? "" : "('" + region.name + "') ",
                    region.constraint.getSummary())
         << "      ++index;\n"
         << "    }\n";
  }
  body << "  }\n";
}

void OpEmitter::genSuccessorVerifier(OpMethodBody &body) {
  // If we have no successors, there is nothing more to do.
  unsigned numSuccessors = op.getNumSuccessors();
  if (numSuccessors == 0)
    return;

  body << "{\n";
  body << "    unsigned index = 0; (void)index;\n";

  for (unsigned i = 0; i < numSuccessors; ++i) {
    const auto &successor = op.getSuccessor(i);
    if (successor.constraint.getPredicate().isNull())
      continue;

    if (successor.isVariadic()) {
      body << formatv("    for (::mlir::Block *successor : {0}()) {\n",
                      successor.name);
    } else {
      body << "    {\n";
      body << formatv("      ::mlir::Block *successor = {0}();\n",
                      successor.name);
    }
    auto constraint = tgfmt(successor.constraint.getConditionTemplate(),
                            &verifyCtx.withSelf("successor"))
                          .str();

    body << formatv("      (void)successor;\n"
                    "      if (!({0})) {\n        "
                    "return emitOpError(\"successor #\") << index << \"('{1}') "
                    "failed to "
                    "verify constraint: {2}\";\n      }\n",
                    constraint, successor.name,
                    successor.constraint.getSummary())
         << "      ++index;\n"
         << "    }\n";
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
    opClass.addTrait("::mlir::OpTrait::Zero" + traitKind);
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
  if (numVariadicOperands != 0) {
    if (numOperands == numVariadicOperands)
      opClass.addTrait("::mlir::OpTrait::VariadicOperands");
    else
      opClass.addTrait("::mlir::OpTrait::AtLeastNOperands<" +
                       Twine(numOperands - numVariadicOperands) + ">::Impl");
  } else {
    switch (numOperands) {
    case 0:
      opClass.addTrait("::mlir::OpTrait::ZeroOperands");
      break;
    case 1:
      opClass.addTrait("::mlir::OpTrait::OneOperand");
      break;
    default:
      opClass.addTrait("::mlir::OpTrait::NOperands<" + Twine(numOperands) +
                       ">::Impl");
      break;
    }
  }

  // Add the native and interface traits.
  for (const auto &trait : op.getTraits()) {
    if (auto opTrait = dyn_cast<tblgen::NativeTrait>(&trait))
      opClass.addTrait(opTrait->getFullyQualifiedTraitName());
    else if (auto opTrait = dyn_cast<tblgen::InterfaceTrait>(&trait))
      opClass.addTrait(opTrait->getFullyQualifiedTraitName());
  }
}

void OpEmitter::genOpNameGetter() {
  auto *method = opClass.addMethodAndPrune(
      "::llvm::StringLiteral", "getOperationName",
      OpMethod::Property(OpMethod::MP_Static | OpMethod::MP_Constexpr));
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
  auto *method = opClass.addMethodAndPrune(
      "void", "getAsmResultNames", "::mlir::OpAsmSetValueNameFn", "setNameFn");
  ERROR_IF_PRUNED(method, "getAsmResultNames", op);
  auto &body = method->body();
  for (int i = 0; i != numResults; ++i) {
    body << "  auto resultGroup" << i << " = getODSResults(" << i << ");\n"
         << "  if (!llvm::empty(resultGroup" << i << "))\n"
         << "    setNameFn(*resultGroup" << i << ".begin(), \""
         << resultNames[i] << "\");\n";
  }
}

//===----------------------------------------------------------------------===//
// OpOperandAdaptor emitter
//===----------------------------------------------------------------------===//

namespace {
// Helper class to emit Op operand adaptors to an output stream.  Operand
// adaptors are wrappers around ArrayRef<Value> that provide named operand
// getters identical to those defined in the Op.
class OpOperandAdaptorEmitter {
public:
  static void emitDecl(const Operator &op, raw_ostream &os);
  static void emitDef(const Operator &op, raw_ostream &os);

private:
  explicit OpOperandAdaptorEmitter(const Operator &op);

  // Add verification function. This generates a verify method for the adaptor
  // which verifies all the op-independent attribute constraints.
  void addVerification();

  const Operator &op;
  Class adaptor;
};
} // end namespace

OpOperandAdaptorEmitter::OpOperandAdaptorEmitter(const Operator &op)
    : op(op), adaptor(op.getAdaptorName()) {
  adaptor.newField("::mlir::ValueRange", "odsOperands");
  adaptor.newField("::mlir::DictionaryAttr", "odsAttrs");
  adaptor.newField("::mlir::RegionRange", "odsRegions");
  const auto *attrSizedOperands =
      op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments");
  {
    SmallVector<OpMethodParameter, 2> paramList;
    paramList.emplace_back("::mlir::ValueRange", "values");
    paramList.emplace_back("::mlir::DictionaryAttr", "attrs",
                           attrSizedOperands ? "" : "nullptr");
    paramList.emplace_back("::mlir::RegionRange", "regions", "{}");
    auto *constructor = adaptor.addConstructorAndPrune(std::move(paramList));

    constructor->addMemberInitializer("odsOperands", "values");
    constructor->addMemberInitializer("odsAttrs", "attrs");
    constructor->addMemberInitializer("odsRegions", "regions");
  }

  {
    auto *constructor = adaptor.addConstructorAndPrune(
        llvm::formatv("{0}&", op.getCppClassName()).str(), "op");
    constructor->addMemberInitializer("odsOperands", "op->getOperands()");
    constructor->addMemberInitializer("odsAttrs", "op->getAttrDictionary()");
    constructor->addMemberInitializer("odsRegions", "op->getRegions()");
  }

  {
    auto *m = adaptor.addMethodAndPrune("::mlir::ValueRange", "getOperands");
    ERROR_IF_PRUNED(m, "getOperands", op);
    m->body() << "  return odsOperands;";
  }
  std::string attr = op.getGetterName("operand_segment_sizes");
  std::string sizeAttrInit = formatv(adapterSegmentSizeAttrInitCode, attr);
  generateNamedOperandGetters(op, adaptor,
                              /*isAdaptor=*/true, sizeAttrInit,
                              /*rangeType=*/"::mlir::ValueRange",
                              /*rangeBeginCall=*/"odsOperands.begin()",
                              /*rangeSizeCall=*/"odsOperands.size()",
                              /*getOperandCallPattern=*/"odsOperands[{0}]");

  FmtContext fctx;
  fctx.withBuilder("::mlir::Builder(odsAttrs.getContext())");

  auto emitAttr = [&](StringRef name, Attribute attr) {
    auto *method = adaptor.addMethodAndPrune(attr.getStorageType(), name);
    ERROR_IF_PRUNED(method, "Adaptor::" + name, op);
    auto &body = method->body();
    body << "  assert(odsAttrs && \"no attributes when constructing adapter\");"
         << "\n  " << attr.getStorageType() << " attr = "
         << "odsAttrs.get(\"" << name << "\").";
    if (attr.hasDefaultValue() || attr.isOptional())
      body << "dyn_cast_or_null<";
    else
      body << "cast<";
    body << attr.getStorageType() << ">();\n";

    if (attr.hasDefaultValue()) {
      // Use the default value if attribute is not set.
      // TODO: this is inefficient, we are recreating the attribute for every
      // call. This should be set instead.
      std::string defaultValue = std::string(
          tgfmt(attr.getConstBuilderTemplate(), &fctx, attr.getDefaultValue()));
      body << "  if (!attr)\n    attr = " << defaultValue << ";\n";
    }
    body << "  return attr;\n";
  };

  {
    auto *m =
        adaptor.addMethodAndPrune("::mlir::DictionaryAttr", "getAttributes");
    ERROR_IF_PRUNED(m, "Adaptor::getAttributes", op);
    m->body() << "  return odsAttrs;";
  }
  for (auto &namedAttr : op.getAttributes()) {
    const auto &name = namedAttr.name;
    const auto &attr = namedAttr.attr;
    if (!attr.isDerivedAttr()) {
      for (auto emitName : op.getGetterNames(name))
        emitAttr(emitName, attr);
    }
  }

  unsigned numRegions = op.getNumRegions();
  if (numRegions > 0) {
    auto *m = adaptor.addMethodAndPrune("::mlir::RegionRange", "getRegions");
    ERROR_IF_PRUNED(m, "Adaptor::getRegions", op);
    m->body() << "  return odsRegions;";
  }
  for (unsigned i = 0; i < numRegions; ++i) {
    const auto &region = op.getRegion(i);
    if (region.name.empty())
      continue;

    // Generate the accessors for a variadic region.
    if (region.isVariadic()) {
      auto *m = adaptor.addMethodAndPrune("::mlir::RegionRange", region.name);
      ERROR_IF_PRUNED(m, "Adaptor::" + region.name, op);
      m->body() << formatv("  return odsRegions.drop_front({0});", i);
      continue;
    }

    auto *m = adaptor.addMethodAndPrune("::mlir::Region &", region.name);
    ERROR_IF_PRUNED(m, "Adaptor::" + region.name, op);
    m->body() << formatv("  return *odsRegions[{0}];", i);
  }

  // Add verification function.
  addVerification();
}

void OpOperandAdaptorEmitter::addVerification() {
  auto *method = adaptor.addMethodAndPrune("::mlir::LogicalResult", "verify",
                                           "::mlir::Location", "loc");
  ERROR_IF_PRUNED(method, "verify", op);
  auto &body = method->body();

  const char *checkAttrSizedValueSegmentsCode = R"(
  {
    auto sizeAttr = odsAttrs.get("{0}").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != {1})
      return emitError(loc, "'{0}' attribute for specifying {2} segments "
                       "must have {1} elements, but got ") << numElements;
  }
  )";

  // Verify a few traits first so that we can use
  // getODSOperands()/getODSResults() in the rest of the verifier.
  for (auto &trait : op.getTraits()) {
    if (auto *t = dyn_cast<tblgen::NativeTrait>(&trait)) {
      if (t->getFullyQualifiedTraitName() ==
          "::mlir::OpTrait::AttrSizedOperandSegments") {
        body << formatv(checkAttrSizedValueSegmentsCode,
                        "operand_segment_sizes", op.getNumOperands(),
                        "operand");
      } else if (t->getFullyQualifiedTraitName() ==
                 "::mlir::OpTrait::AttrSizedResultSegments") {
        body << formatv(checkAttrSizedValueSegmentsCode, "result_segment_sizes",
                        op.getNumResults(), "result");
      }
    }
  }

  FmtContext verifyCtx;
  populateSubstitutions(op, "odsAttrs.get", "getODSOperands",
                        "<no results should be generated>", verifyCtx);
  genAttributeVerifier(op, "odsAttrs.get",
                       Twine("emitError(loc, \"'") + op.getOperationName() +
                           "' op \"",
                       /*emitVerificationRequiringOp*/ false, verifyCtx, body);

  body << "  return ::mlir::success();";
}

void OpOperandAdaptorEmitter::emitDecl(const Operator &op, raw_ostream &os) {
  OpOperandAdaptorEmitter(op).adaptor.writeDeclTo(os);
}

void OpOperandAdaptorEmitter::emitDef(const Operator &op, raw_ostream &os) {
  OpOperandAdaptorEmitter(op).adaptor.writeDefTo(os);
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
  StaticVerifierFunctionEmitter staticVerifierEmitter(recordKeeper, os);
  os << formatv(opCommentHeader, "Local Utility Method", "Definitions");
  staticVerifierEmitter.emitFunctionsFor(
      typeVerifierSignature, typeVerifierErrorHandler, /*typeArgName=*/"type",
      defs, emitDecl);

  for (auto *def : defs) {
    Operator op(*def);
    if (emitDecl) {
      {
        NamespaceEmitter emitter(os, op.getCppNamespace());
        os << formatv(opCommentHeader, op.getQualCppClassName(),
                      "declarations");
        OpOperandAdaptorEmitter::emitDecl(op, os);
        OpEmitter::emitDecl(op, os, staticVerifierEmitter);
      }
      // Emit the TypeID explicit specialization to have a single definition.
      if (!op.getCppNamespace().empty())
        os << "DECLARE_EXPLICIT_TYPE_ID(" << op.getCppNamespace()
           << "::" << op.getCppClassName() << ")\n\n";
    } else {
      {
        NamespaceEmitter emitter(os, op.getCppNamespace());
        os << formatv(opCommentHeader, op.getQualCppClassName(), "definitions");
        OpOperandAdaptorEmitter::emitDef(op, os);
        OpEmitter::emitDef(op, os, staticVerifierEmitter);
      }
      // Emit the TypeID explicit specialization to have a single definition.
      if (!op.getCppNamespace().empty())
        os << "DEFINE_EXPLICIT_TYPE_ID(" << op.getCppNamespace()
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
