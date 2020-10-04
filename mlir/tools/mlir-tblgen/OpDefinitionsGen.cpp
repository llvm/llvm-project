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
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/OpClass.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/SideEffects.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#define DEBUG_TYPE "mlir-tblgen-opdefgen"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

cl::OptionCategory opDefGenCat("Options for -gen-op-defs and -gen-op-decls");

static cl::opt<std::string> opIncFilter(
    "op-include-regex",
    cl::desc("Regex of name of op's to include (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<std::string> opExcFilter(
    "op-exclude-regex",
    cl::desc("Regex of name of op's to exclude (no filter if empty)"),
    cl::cat(opDefGenCat));

static const char *const tblgenNamePrefix = "tblgen_";
static const char *const generatedArgName = "odsArg";
static const char *const builder = "odsBuilder";
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
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("{0}");
)";
const char *attrSizedSegmentValueRangeCalcCode = R"(
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
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
  return isa<CodeInit, StringInit>(valueInit);
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
  static void emitDecl(const Operator &op, raw_ostream &os);
  static void emitDef(const Operator &op, raw_ostream &os);

private:
  OpEmitter(const Operator &op);

  void emitDecl(raw_ostream &os);
  void emitDef(raw_ostream &os);

  // Generates the OpAsmOpInterface for this operation if possible.
  void genOpAsmInterface();

  // Generates the `getOperationName` method for this op.
  void genOpNameGetter();

  // Generates getters for the attributes.
  void genAttrGetters();

  // Generates setter for the attributes.
  void genAttrSetters();

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
  // types. The given `typeParamKind` and `attrParamKind` controls how result
  // types and attributes are placed in the parameter list.
  void buildParamList(llvm::SmallVectorImpl<OpMethodParameter> &paramList,
                      SmallVectorImpl<std::string> &resultTypeNames,
                      TypeParamKind typeParamKind,
                      AttrParamKind attrParamKind = AttrParamKind::WrappedAttr);

  // Adds op arguments and regions into operation state for build() methods.
  void genCodeForAddingArgAndRegionForBuilder(OpMethodBody &body,
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

  // Generate the OpInterface methods.
  void genOpInterfaceMethods();

  // Generate op interface method.
  void genOpInterfaceMethod(const tblgen::InterfaceOpTrait *trait);

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
};
} // end anonymous namespace

// Populate the format context `ctx` with substitutions of attributes, operands
// and results.
// - attrGet corresponds to the name of the function to call to get value of
//   attribute (the generated function call returns an Attribute);
// - operandGet corresponds to the name of the function with which to retrieve
//   an operand (the generated function call returns an OperandRange);
// - reultGet corresponds to the name of the function to get an result (the
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
                  emitErrorPrefix, attrName, attr.getDescription());
    if (allowMissingAttr)
      body << "  }\n";
    body << "  }\n";
  }
}

OpEmitter::OpEmitter(const Operator &op)
    : def(op.getDef()), op(op),
      opClass(op.getCppClassName(), op.getExtraClassDeclaration()) {
  verifyCtx.withOp("(*this->getOperation())");

  genTraits();

  // Generate C++ code for various op methods. The order here determines the
  // methods in the generated file.
  genOpAsmInterface();
  genOpNameGetter();
  genNamedOperandGetters();
  genNamedOperandSetters();
  genNamedResultGetters();
  genNamedRegionGetters();
  genNamedSuccessorGetters();
  genAttrGetters();
  genAttrSetters();
  genBuilder();
  genParser();
  genPrinter();
  genVerifier();
  genCanonicalizerDecls();
  genFolderDecls();
  genOpInterfaceMethods();
  generateOpFormat(op, opClass);
  genSideEffectInterfaceMethods();
  genTypeInterfaceMethods();
}

void OpEmitter::emitDecl(const Operator &op, raw_ostream &os) {
  OpEmitter(op).emitDecl(os);
}

void OpEmitter::emitDef(const Operator &op, raw_ostream &os) {
  OpEmitter(op).emitDef(os);
}

void OpEmitter::emitDecl(raw_ostream &os) { opClass.writeDeclTo(os); }

void OpEmitter::emitDef(raw_ostream &os) { opClass.writeDefTo(os); }

void OpEmitter::genAttrGetters() {
  FmtContext fctx;
  fctx.withBuilder("::mlir::Builder(this->getContext())");

  Dialect opDialect = op.getDialect();
  // Emit the derived attribute body.
  auto emitDerivedAttr = [&](StringRef name, Attribute attr) {
    auto *method = opClass.addMethodAndPrune(attr.getReturnType(), name);
    if (!method)
      return;
    auto &body = method->body();
    body << "  " << attr.getDerivedCodeBody() << "\n";
  };

  // Emit with return type specified.
  auto emitAttrWithReturnType = [&](StringRef name, Attribute attr) {
    Dialect attrDialect = attr.getDialect();
    // Does the current operation have a different namespace than the attribute?
    bool differentNamespace =
        attrDialect && opDialect && attrDialect != opDialect;
    std::string returnType = differentNamespace
                                 ? (llvm::Twine(attrDialect.getCppNamespace()) +
                                    "::" + attr.getReturnType())
                                       .str()
                                 : attr.getReturnType().str();
    auto *method = opClass.addMethodAndPrune(returnType, name);
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

  // Generate raw named accessor type. This is a wrapper class that allows
  // referring to the attributes via accessors instead of having to use
  // the string interface for better compile time verification.
  auto emitAttrWithStorageType = [&](StringRef name, Attribute attr) {
    auto *method =
        opClass.addMethodAndPrune(attr.getStorageType(), (name + "Attr").str());
    if (!method)
      return;
    auto &body = method->body();
    body << "  return this->getAttr(\"" << name << "\").";
    if (attr.isOptional() || attr.hasDefaultValue())
      body << "dyn_cast_or_null<";
    else
      body << "cast<";
    body << attr.getStorageType() << ">();";
  };

  for (auto &namedAttr : op.getAttributes()) {
    const auto &name = namedAttr.name;
    const auto &attr = namedAttr.attr;
    if (attr.isDerivedAttr()) {
      emitDerivedAttr(name, attr);
    } else {
      emitAttrWithStorageType(name, attr);
      emitAttrWithReturnType(name, attr);
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
      auto &body = method->body();
      for (auto namedAttr : derivedAttrs)
        body << "  if (name == \"" << namedAttr.name << "\") return true;\n";
      body << " return false;";
    }
    // Generate method to materialize derived attributes as a DictionaryAttr.
    {
      auto *method = opClass.addMethodAndPrune("::mlir::DictionaryAttr",
                                               "materializeDerivedAttributes");
      auto &body = method->body();

      auto nonMaterializable =
          make_filter_range(derivedAttrs, [](const NamedAttribute &namedAttr) {
            return namedAttr.attr.getConvertFromStorageCall().empty();
          });
      if (!nonMaterializable.empty()) {
        std::string attrs;
        llvm::raw_string_ostream os(attrs);
        interleaveComma(nonMaterializable, os,
                        [&](const NamedAttribute &attr) { os << attr.name; });
        PrintWarning(
            op.getLoc(),
            formatv(
                "op has non-materialzable derived attributes '{0}', skipping",
                os.str()));
        body << formatv("  emitOpError(\"op has non-materializable derived "
                        "attributes '{0}'\");\n",
                        attrs);
        body << "  return nullptr;";
        return;
      }

      body << "  ::mlir::MLIRContext* ctx = getContext();\n";
      body << "  ::mlir::Builder odsBuilder(ctx); (void)odsBuilder;\n";
      body << "  return ::mlir::DictionaryAttr::get({\n";
      interleave(
          derivedAttrs, body,
          [&](const NamedAttribute &namedAttr) {
            auto tmpl = namedAttr.attr.getConvertFromStorageCall();
            body << "    {::mlir::Identifier::get(\"" << namedAttr.name
                 << "\", ctx),\n"
                 << tgfmt(tmpl, &fctx.withSelf(namedAttr.name + "()")
                                     .withBuilder("odsBuilder")
                                     .addSubst("_ctx", "ctx"))
                 << "}";
          },
          ",\n");
      body << "\n    }, ctx);";
    }
  }
}

void OpEmitter::genAttrSetters() {
  // Generate raw named setter type. This is a wrapper class that allows setting
  // to the attributes via setters instead of having to use the string interface
  // for better compile time verification.
  auto emitAttrWithStorageType = [&](StringRef name, Attribute attr) {
    auto *method = opClass.addMethodAndPrune("void", (name + "Attr").str(),
                                             attr.getStorageType(), "attr");
    if (!method)
      return;
    auto &body = method->body();
    body << "  this->getOperation()->setAttr(\"" << name << "\", attr);";
  };

  for (auto &namedAttr : op.getAttributes()) {
    const auto &name = namedAttr.name;
    const auto &attr = namedAttr.attr;
    if (!attr.isDerivedAttr())
      emitAttrWithStorageType(name, attr);
  }
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
                                        StringRef sizeAttrInit,
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
  auto &body = m->body();
  body << formatv(valueRangeReturnCode, rangeBeginCall,
                  "getODSOperandIndexAndLength(index)");

  // Then we emit nicer named getter methods by redirecting to the "sink" getter
  // method.
  for (int i = 0; i != numOperands; ++i) {
    const auto &operand = op.getOperand(i);
    if (operand.name.empty())
      continue;

    if (operand.isOptional()) {
      m = opClass.addMethodAndPrune("::mlir::Value", operand.name);
      m->body() << "  auto operands = getODSOperands(" << i << ");\n"
                << "  return operands.empty() ? Value() : *operands.begin();";
    } else if (operand.isVariadic()) {
      m = opClass.addMethodAndPrune(rangeType, operand.name);
      m->body() << "  return getODSOperands(" << i << ");";
    } else {
      m = opClass.addMethodAndPrune("::mlir::Value", operand.name);
      m->body() << "  return *getODSOperands(" << i << ").begin();";
    }
  }
}

void OpEmitter::genNamedOperandGetters() {
  generateNamedOperandGetters(
      op, opClass,
      /*sizeAttrInit=*/
      formatv(opSegmentSizeAttrInitCode, "operand_segment_sizes").str(),
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
    auto *m = opClass.addMethodAndPrune("::mlir::MutableOperandRange",
                                        (operand.name + "Mutable").str());
    auto &body = m->body();
    body << "  auto range = getODSOperandIndexAndLength(" << i << ");\n"
         << "  return ::mlir::MutableOperandRange(getOperation(), "
            "range.first, range.second";
    if (attrSizedOperands)
      body << ", ::mlir::MutableOperandRange::OperandSegment(" << i
           << "u, *getOperation()->getMutableAttrDict().getNamed("
              "\"operand_segment_sizes\"))";
    body << ");\n";
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

  generateValueRangeStartAndEnd(
      opClass, "getODSResultIndexAndLength", numVariadicResults,
      numNormalResults, "getOperation()->getNumResults()", attrSizedResults,
      formatv(opSegmentSizeAttrInitCode, "result_segment_sizes").str(),
      op.getResults());

  auto *m = opClass.addMethodAndPrune("::mlir::Operation::result_range",
                                      "getODSResults", "unsigned", "index");
  m->body() << formatv(valueRangeReturnCode, "getOperation()->result_begin()",
                       "getODSResultIndexAndLength(index)");

  for (int i = 0; i != numResults; ++i) {
    const auto &result = op.getResult(i);
    if (result.name.empty())
      continue;

    if (result.isOptional()) {
      m = opClass.addMethodAndPrune("::mlir::Value", result.name);
      m->body()
          << "  auto results = getODSResults(" << i << ");\n"
          << "  return results.empty() ? ::mlir::Value() : *results.begin();";
    } else if (result.isVariadic()) {
      m = opClass.addMethodAndPrune("::mlir::Operation::result_range",
                                    result.name);
      m->body() << "  return getODSResults(" << i << ");";
    } else {
      m = opClass.addMethodAndPrune("::mlir::Value", result.name);
      m->body() << "  return *getODSResults(" << i << ").begin();";
    }
  }
}

void OpEmitter::genNamedRegionGetters() {
  unsigned numRegions = op.getNumRegions();
  for (unsigned i = 0; i < numRegions; ++i) {
    const auto &region = op.getRegion(i);
    if (region.name.empty())
      continue;

    // Generate the accessors for a varidiadic region.
    if (region.isVariadic()) {
      auto *m = opClass.addMethodAndPrune("::mlir::MutableArrayRef<Region>",
                                          region.name);
      m->body() << formatv(
          "  return this->getOperation()->getRegions().drop_front({0});", i);
      continue;
    }

    auto *m = opClass.addMethodAndPrune("::mlir::Region &", region.name);
    m->body() << formatv("  return this->getOperation()->getRegion({0});", i);
  }
}

void OpEmitter::genNamedSuccessorGetters() {
  unsigned numSuccessors = op.getNumSuccessors();
  for (unsigned i = 0; i < numSuccessors; ++i) {
    const NamedSuccessor &successor = op.getSuccessor(i);
    if (successor.name.empty())
      continue;

    // Generate the accessors for a variadic successor list.
    if (successor.isVariadic()) {
      auto *m =
          opClass.addMethodAndPrune("::mlir::SuccessorRange", successor.name);
      m->body() << formatv(
          "  return {std::next(this->getOperation()->successor_begin(), {0}), "
          "this->getOperation()->successor_end()};",
          i);
      continue;
    }

    auto *m = opClass.addMethodAndPrune("::mlir::Block *", successor.name);
    m->body() << formatv("  return this->getOperation()->getSuccessor({0});",
                         i);
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
    buildParamList(paramList, resultNames, paramKind, attrType);

    auto *m = opClass.addMethodAndPrune("void", "build", OpMethod::MP_Static,
                                        std::move(paramList));
    // If the builder is redundant, skip generating the method.
    if (!m)
      return;
    auto &body = m->body();
    genCodeForAddingArgAndRegionForBuilder(
        body, /*isRawValueAttr=*/attrType == AttrParamKind::UnwrappedValue);

    // Push all result types to the operation state

    if (inferType) {
      // Generate builder that infers type too.
      // TODO: Subsume this with general checking if type can be
      // inferred automatically.
      // TODO: Expand to handle regions.
      body << formatv(R"(
        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (succeeded({0}::inferReturnTypes(odsBuilder.getContext(),
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

  // Some of the build methods generated here may be amiguous, but TableGen's
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
    if (succeeded({0}::inferReturnTypes(odsBuilder.getContext(),
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
  buildParamList(paramList, resultNames, TypeParamKind::None);

  auto *m = opClass.addMethodAndPrune("void", "build", OpMethod::MP_Static,
                                      std::move(paramList));
  // If the builder is redundant, skip generating the method
  if (!m)
    return;
  auto &body = m->body();
  genCodeForAddingArgAndRegionForBuilder(body);

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

  body << "  for (auto attr : attributes) {\n";
  body << "    if (attr.first != \"" << namedAttr.name << "\") continue;\n";
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

void OpEmitter::genBuilder() {
  // Handle custom builders if provided.
  // TODO: Create wrapper class for OpBuilder to hide the native
  // TableGen API calls here.
  {
    auto *listInit = dyn_cast_or_null<ListInit>(def.getValueInit("builders"));
    if (listInit) {
      for (Init *init : listInit->getValues()) {
        Record *builderDef = cast<DefInit>(init)->getDef();
        StringRef params = builderDef->getValueAsString("params").trim();
        // TODO: Remove this and just generate the builder/state always.
        bool skipParamGen = params.startswith("OpBuilder") ||
                            params.startswith("mlir::OpBuilder") ||
                            params.startswith("::mlir::OpBuilder");
        StringRef body = builderDef->getValueAsString("body");
        bool hasBody = !body.empty();

        OpMethod::Property properties =
            hasBody ? OpMethod::MP_Static : OpMethod::MP_StaticDeclaration;
        std::string paramStr =
            skipParamGen ? params.str()
                         : llvm::formatv("::mlir::OpBuilder &{0}, "
                                         "::mlir::OperationState &{1}{2}{3}",
                                         builder, builderOpState,
                                         params.empty() ? "" : ", ", params)
                               .str();
        auto *method =
            opClass.addMethodAndPrune("void", "build", properties, paramStr);
        if (hasBody) {
          if (skipParamGen) {
            method->body() << body;
          } else {
            FmtContext fctx;
            fctx.withBuilder(builder);
            fctx.addSubst("_state", builderOpState);
            method->body() << tgfmt(body, &fctx);
          }
        }
      }
    }
    if (op.skipDefaultBuilders()) {
      if (!listInit || listInit->empty())
        PrintFatalError(
            op.getLoc(),
            "default builders are skipped and no custom builders provided");
      return;
    }
  }

  // Generate default builders that requires all result type, operands, and
  // attributes as parameters.

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

  int numOperands = 0;
  int numAttrs = 0;

  int defaultValuedAttrStartIndex = op.getNumArgs();
  if (attrParamKind == AttrParamKind::UnwrappedValue) {
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

  for (int i = 0, e = op.getNumArgs(); i < e; ++i) {
    auto argument = op.getArg(i);
    if (argument.is<tblgen::NamedTypeConstraint *>()) {
      const auto &operand = op.getOperand(numOperands);
      StringRef type =
          operand.isVariadic() ? "::mlir::ValueRange" : "::mlir::Value";
      OpMethodParameter::Property properties = OpMethodParameter::PP_None;
      if (operand.isOptional())
        properties = OpMethodParameter::PP_Optional;

      paramList.emplace_back(type, getArgumentName(op, numOperands),
                             properties);
      ++numOperands;
    } else {
      const auto &namedAttr = op.getAttribute(numAttrs);
      const auto &attr = namedAttr.attr;

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

      std::string defaultValue;
      // Attach default value if requested and possible.
      if (attrParamKind == AttrParamKind::UnwrappedValue &&
          i >= defaultValuedAttrStartIndex) {
        bool isString = attr.getReturnType() == "::llvm::StringRef";
        if (isString)
          defaultValue.append("\"");
        defaultValue += attr.getDefaultValue();
        if (isString)
          defaultValue.append("\"");
      }
      paramList.emplace_back(type, namedAttr.name, defaultValue, properties);
      ++numAttrs;
    }
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

void OpEmitter::genCodeForAddingArgAndRegionForBuilder(OpMethodBody &body,
                                                       bool isRawValueAttr) {
  // Push all operands to the result.
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    std::string argName = getArgumentName(op, i);
    if (op.getOperand(i).isOptional())
      body << "  if (" << argName << ")\n  ";
    body << "  " << builderOpState << ".addOperands(" << argName << ");\n";
  }

  // If the operation has the operand segment size attribute, add it here.
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    body << "  " << builderOpState
         << ".addAttribute(\"operand_segment_sizes\", "
            "odsBuilder.getI32VectorAttr({";
    interleaveComma(llvm::seq<int>(0, op.getNumOperands()), body, [&](int i) {
      if (op.getOperand(i).isOptional())
        body << "(" << getArgumentName(op, i) << " ? 1 : 0)";
      else if (op.getOperand(i).isVariadic())
        body << "static_cast<int32_t>(" << getArgumentName(op, i) << ".size())";
      else
        body << "1";
    });
    body << "}));\n";
  }

  // Push all attributes to the result.
  for (const auto &namedAttr : op.getAttributes()) {
    auto &attr = namedAttr.attr;
    if (!attr.isDerivedAttr()) {
      bool emitNotNullCheck = attr.isOptional();
      if (emitNotNullCheck) {
        body << formatv("  if ({0}) ", namedAttr.name) << "{\n";
      }
      if (isRawValueAttr && canUseUnwrappedRawValue(attr)) {
        // If this is a raw value, then we need to wrap it in an Attribute
        // instance.
        FmtContext fctx;
        fctx.withBuilder("odsBuilder");

        std::string builderTemplate =
            std::string(attr.getConstBuilderTemplate());

        // For StringAttr, its constant builder call will wrap the input in
        // quotes, which is correct for normal string literals, but incorrect
        // here given we use function arguments. So we need to strip the
        // wrapping quotes.
        if (StringRef(builderTemplate).contains("\"$0\""))
          builderTemplate = replaceAllSubstrs(builderTemplate, "\"$0\"", "$0");

        std::string value =
            std::string(tgfmt(builderTemplate, &fctx, namedAttr.name));
        body << formatv("  {0}.addAttribute(\"{1}\", {2});\n", builderOpState,
                        namedAttr.name, value);
      } else {
        body << formatv("  {0}.addAttribute(\"{1}\", {1});\n", builderOpState,
                        namedAttr.name);
      }
      if (emitNotNullCheck) {
        body << "  }\n";
      }
    }
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
  if (!def.getValueAsBit("hasCanonicalizer"))
    return;

  SmallVector<OpMethodParameter, 2> paramList;
  paramList.emplace_back("::mlir::OwningRewritePatternList &", "results");
  paramList.emplace_back("::mlir::MLIRContext *", "context");
  opClass.addMethodAndPrune("void", "getCanonicalizationPatterns",
                            OpMethod::MP_StaticDeclaration,
                            std::move(paramList));
}

void OpEmitter::genFolderDecls() {
  bool hasSingleResult =
      op.getNumResults() == 1 && op.getNumVariableLengthResults() == 0;

  if (def.getValueAsBit("hasFolder")) {
    if (hasSingleResult) {
      opClass.addMethodAndPrune(
          "::mlir::OpFoldResult", "fold", OpMethod::MP_Declaration,
          "::llvm::ArrayRef<::mlir::Attribute>", "operands");
    } else {
      SmallVector<OpMethodParameter, 2> paramList;
      paramList.emplace_back("::llvm::ArrayRef<::mlir::Attribute>", "operands");
      paramList.emplace_back("::llvm::SmallVectorImpl<::mlir::OpFoldResult> &",
                             "results");
      opClass.addMethodAndPrune("::mlir::LogicalResult", "fold",
                                OpMethod::MP_Declaration, std::move(paramList));
    }
  }
}

void OpEmitter::genOpInterfaceMethod(const tblgen::InterfaceOpTrait *opTrait) {
  auto interface = opTrait->getOpInterface();

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

    SmallVector<OpMethodParameter, 4> paramList;
    for (const InterfaceMethod::Argument &arg : method.getArguments())
      paramList.emplace_back(arg.type, arg.name);

    auto properties = method.isStatic() ? OpMethod::MP_StaticDeclaration
                                        : OpMethod::MP_Declaration;
    opClass.addMethodAndPrune(method.getReturnType(), method.getName(),
                              properties, std::move(paramList));
  }
}

void OpEmitter::genOpInterfaceMethods() {
  for (const auto &trait : op.getTraits()) {
    if (const auto *opTrait = dyn_cast<tblgen::InterfaceOpTrait>(&trait))
      if (opTrait->shouldDeclareMethods())
        genOpInterfaceMethod(opTrait);
  }
}

void OpEmitter::genSideEffectInterfaceMethods() {
  enum EffectKind { Operand, Result, Static };
  struct EffectLocation {
    /// The effect applied.
    SideEffect effect;

    /// The index if the kind is either operand or result.
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
  /// Operands.
  for (unsigned i = 0, operandIt = 0, e = op.getNumArgs(); i != e; ++i) {
    if (op.getArg(i).is<NamedTypeConstraint *>()) {
      resolveDecorators(op.getArgDecorators(i), operandIt, EffectKind::Operand);
      ++operandIt;
    }
  }
  /// Results.
  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    resolveDecorators(op.getResultDecorators(i), i, EffectKind::Result);

  for (auto &it : interfaceEffects) {
    // Generate the 'getEffects' method.
    std::string type = llvm::formatv("::mlir::SmallVectorImpl<::mlir::"
                                     "SideEffects::EffectInstance<{0}>> &",
                                     it.first())
                           .str();
    auto *getEffects =
        opClass.addMethodAndPrune("void", "getEffects", type, "effects");
    auto &body = getEffects->body();

    // Add effect instances for each of the locations marked on the operation.
    for (auto &location : it.second) {
      if (location.kind != EffectKind::Static) {
        body << "  for (::mlir::Value value : getODS"
             << (location.kind == EffectKind::Operand ? "Operands" : "Results")
             << "(" << location.index << "))\n  ";
      }

      body << "  effects.emplace_back(" << location.effect.getName()
           << "::get()";

      // If the effect isn't static, it has a specific value attached to it.
      if (location.kind != EffectKind::Static)
        body << ", value";
      body << ", " << location.effect.getResource() << "::get());\n";
    }
  }
}

void OpEmitter::genTypeInterfaceMethods() {
  if (!op.allResultTypesKnown())
    return;

  SmallVector<OpMethodParameter, 4> paramList;
  paramList.emplace_back("::mlir::MLIRContext *", "context");
  paramList.emplace_back("::llvm::Optional<::mlir::Location>", "location");
  paramList.emplace_back("::mlir::ValueRange", "operands");
  paramList.emplace_back("::mlir::DictionaryAttr", "attributes");
  paramList.emplace_back("::mlir::RegionRange", "regions");
  paramList.emplace_back("::llvm::SmallVectorImpl<::mlir::Type>&",
                         "inferredReturnTypes");
  auto *method =
      opClass.addMethodAndPrune("::mlir::LogicalResult", "inferReturnTypes",
                                OpMethod::MP_Static, std::move(paramList));

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

  FmtContext fctx;
  fctx.addSubst("cppClass", opClass.getClassName());
  auto parser = def.getValueAsString("parser").ltrim().rtrim(" \t\v\f\r");
  method->body() << "  " << tgfmt(parser, &fctx);
}

void OpEmitter::genPrinter() {
  if (hasStringAttribute(def, "assemblyFormat"))
    return;

  auto valueInit = def.getValueInit("printer");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  if (!codeInit)
    return;

  auto *method =
      opClass.addMethodAndPrune("void", "print", "::mlir::OpAsmPrinter &", "p");
  FmtContext fctx;
  fctx.addSubst("cppClass", opClass.getClassName());
  auto printer = codeInit->getValue().ltrim().rtrim(" \t\v\f\r");
  method->body() << "  " << tgfmt(printer, &fctx);
}

void OpEmitter::genVerifier() {
  auto *method = opClass.addMethodAndPrune("::mlir::LogicalResult", "verify");
  auto &body = method->body();
  body << "  if (failed(" << op.getAdaptorName()
       << "(*this).verify(this->getLoc()))) "
       << "return ::mlir::failure();\n";

  auto *valueInit = def.getValueInit("verifier");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  bool hasCustomVerify = codeInit && !codeInit->getValue().empty();
  populateSubstitutions(op, "this->getAttr", "this->getODSOperands",
                        "this->getODSResults", verifyCtx);

  genAttributeVerifier(op, "this->getAttr", "emitOpError(",
                       /*emitVerificationRequiringOp=*/true, verifyCtx, body);
  genOperandResultVerifier(body, op.getOperands(), "operand");
  genOperandResultVerifier(body, op.getResults(), "result");

  for (auto &trait : op.getTraits()) {
    if (auto *t = dyn_cast<tblgen::PredOpTrait>(&trait)) {
      body << tgfmt("  if (!($0))\n    "
                    "return emitOpError(\"failed to verify that $1\");\n",
                    &verifyCtx, tgfmt(t->getPredTemplate(), &verifyCtx),
                    t->getDescription());
    }
  }

  genRegionVerifier(body);
  genSuccessorVerifier(body);

  if (hasCustomVerify) {
    FmtContext fctx;
    fctx.addSubst("cppClass", opClass.getClassName());
    auto printer = codeInit->getValue().ltrim().rtrim(" \t\v\f\r");
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
    bool hasPredicate = staticValue.value().hasPredicate();
    bool isOptional = staticValue.value().isOptional();
    if (!hasPredicate && !isOptional)
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
    }

    // Otherwise, if there is no predicate there is nothing left to do.
    if (!hasPredicate)
      continue;

    // Emit a loop to check all the dynamic values in the pack.
    body << "    for (::mlir::Value v : valueGroup" << staticValue.index()
         << ") {\n";

    auto constraint = staticValue.value().constraint;
    body << "      (void)v;\n"
         << "      if (!("
         << tgfmt(constraint.getConditionTemplate(),
                  &fctx.withSelf("v.getType()"))
         << ")) {\n"
         << formatv("        return emitOpError(\"{0} #\") << index "
                    "<< \" must be {1}, but got \" << v.getType();\n",
                    valueKind, constraint.getDescription())
         << "      }\n" // if
         << "      ++index;\n"
         << "    }\n"; // for
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
                        : "::mlir::MutableArrayRef<::mlir::Region>(this->"
                          "getOperation()->getRegion({1}))",
                    region.name, i);
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
                    region.constraint.getDescription())
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
                    successor.constraint.getDescription())
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

  // Add result size trait.
  int numResults = op.getNumResults();
  int numVariadicResults = op.getNumVariableLengthResults();
  addSizeCountTrait(opClass, "Result", numResults, numVariadicResults);

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
    if (auto opTrait = dyn_cast<tblgen::NativeOpTrait>(&trait))
      opClass.addTrait(opTrait->getTrait());
    else if (auto opTrait = dyn_cast<tblgen::InterfaceOpTrait>(&trait))
      opClass.addTrait(opTrait->getTrait());
  }
}

void OpEmitter::genOpNameGetter() {
  auto *method = opClass.addMethodAndPrune(
      "::llvm::StringRef", "getOperationName", OpMethod::MP_Static);
  method->body() << "  return \"" << op.getOperationName() << "\";\n";
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
  auto *method = opClass.addMethodAndPrune("void", "getAsmResultNames",
                                           "OpAsmSetValueNameFn", "setNameFn");
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
  const auto *attrSizedOperands =
      op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments");
  {
    SmallVector<OpMethodParameter, 2> paramList;
    paramList.emplace_back("::mlir::ValueRange", "values");
    paramList.emplace_back("::mlir::DictionaryAttr", "attrs",
                           attrSizedOperands ? "" : "nullptr");
    auto *constructor = adaptor.addConstructorAndPrune(std::move(paramList));

    constructor->addMemberInitializer("odsOperands", "values");
    constructor->addMemberInitializer("odsAttrs", "attrs");
  }

  {
    auto *constructor = adaptor.addConstructorAndPrune(
        llvm::formatv("{0}&", op.getCppClassName()).str(), "op");
    constructor->addMemberInitializer("odsOperands",
                                      "op.getOperation()->getOperands()");
    constructor->addMemberInitializer("odsAttrs",
                                      "op.getOperation()->getAttrDictionary()");
  }

  std::string sizeAttrInit =
      formatv(adapterSegmentSizeAttrInitCode, "operand_segment_sizes");
  generateNamedOperandGetters(op, adaptor, sizeAttrInit,
                              /*rangeType=*/"::mlir::ValueRange",
                              /*rangeBeginCall=*/"odsOperands.begin()",
                              /*rangeSizeCall=*/"odsOperands.size()",
                              /*getOperandCallPattern=*/"odsOperands[{0}]");

  FmtContext fctx;
  fctx.withBuilder("::mlir::Builder(odsAttrs.getContext())");

  auto emitAttr = [&](StringRef name, Attribute attr) {
    auto &body = adaptor.addMethodAndPrune(attr.getStorageType(), name)->body();
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

  for (auto &namedAttr : op.getAttributes()) {
    const auto &name = namedAttr.name;
    const auto &attr = namedAttr.attr;
    if (!attr.isDerivedAttr())
      emitAttr(name, attr);
  }

  // Add verification function.
  addVerification();
}

void OpOperandAdaptorEmitter::addVerification() {
  auto *method = adaptor.addMethodAndPrune("::mlir::LogicalResult", "verify",
                                           "::mlir::Location", "loc");
  auto &body = method->body();

  const char *checkAttrSizedValueSegmentsCode = R"(
  {
    auto sizeAttr = odsAttrs.get("{0}").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != {1})
      return emitError(loc, "'{0}' attribute for specifying {2} segments "
                       "must have {1} elements");
  }
  )";

  // Verify a few traits first so that we can use
  // getODSOperands()/getODSResults() in the rest of the verifier.
  for (auto &trait : op.getTraits()) {
    if (auto *t = dyn_cast<tblgen::NativeOpTrait>(&trait)) {
      if (t->getTrait() == "::mlir::OpTrait::AttrSizedOperandSegments") {
        body << formatv(checkAttrSizedValueSegmentsCode,
                        "operand_segment_sizes", op.getNumOperands(),
                        "operand");
      } else if (t->getTrait() == "::mlir::OpTrait::AttrSizedResultSegments") {
        body << formatv(checkAttrSizedValueSegmentsCode, "result_segment_sizes",
                        op.getNumResults(), "result");
      }
    }
  }

  FmtContext verifyCtx;
  populateSubstitutions(op, "odsAttrs.get", "getODSOperands",
                        "<no results should be genarated>", verifyCtx);
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
static void emitOpClasses(const std::vector<Record *> &defs, raw_ostream &os,
                          bool emitDecl) {
  // First emit forward declaration for each class, this allows them to refer
  // to each others in traits for example.
  if (emitDecl) {
    os << "#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)\n";
    os << "#undef GET_OP_FWD_DEFINES\n";
    for (auto *def : defs) {
      Operator op(*def);
      NamespaceEmitter emitter(os, op.getDialect());
      os << "class " << op.getCppClassName() << ";\n";
    }
    os << "#endif\n\n";
  }

  IfDefScope scope("GET_OP_CLASSES", os);
  for (auto *def : defs) {
    Operator op(*def);
    NamespaceEmitter emitter(os, op.getDialect());
    if (emitDecl) {
      os << formatv(opCommentHeader, op.getQualCppClassName(), "declarations");
      OpOperandAdaptorEmitter::emitDecl(op, os);
      OpEmitter::emitDecl(op, os);
    } else {
      os << formatv(opCommentHeader, op.getQualCppClassName(), "definitions");
      OpOperandAdaptorEmitter::emitDef(op, os);
      OpEmitter::emitDef(op, os);
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

static std::string getOperationName(const Record &def) {
  auto prefix = def.getValueAsDef("opDialect")->getValueAsString("name");
  auto opName = def.getValueAsString("opName");
  if (prefix.empty())
    return std::string(opName);
  return std::string(llvm::formatv("{0}.{1}", prefix, opName));
}

static std::vector<Record *>
getAllDerivedDefinitions(const RecordKeeper &recordKeeper,
                         StringRef className) {
  Record *classDef = recordKeeper.getClass(className);
  if (!classDef)
    PrintFatalError("ERROR: Couldn't find the `" + className + "' class!\n");

  llvm::Regex includeRegex(opIncFilter), excludeRegex(opExcFilter);
  std::vector<Record *> defs;
  for (const auto &def : recordKeeper.getDefs()) {
    if (!def.second->isSubClassOf(classDef))
      continue;
    // Include if no include filter or include filter matches.
    if (!opIncFilter.empty() &&
        !includeRegex.match(getOperationName(*def.second)))
      continue;
    // Unless there is an exclude filter and it matches.
    if (!opExcFilter.empty() &&
        excludeRegex.match(getOperationName(*def.second)))
      continue;
    defs.push_back(def.second.get());
  }

  return defs;
}

static bool emitOpDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op Declarations", os);

  const auto &defs = getAllDerivedDefinitions(recordKeeper, "Op");
  emitOpClasses(defs, os, /*emitDecl=*/true);

  return false;
}

static bool emitOpDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op Definitions", os);

  const auto &defs = getAllDerivedDefinitions(recordKeeper, "Op");
  emitOpList(defs, os);
  emitOpClasses(defs, os, /*emitDecl=*/false);

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
