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
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/OpClass.h"
#include "mlir/TableGen/OpInterfaces.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/SideEffects.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
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
  auto sizeAttr = odsAttrs.get("{0}").cast<DenseIntElementsAttr>();
)";
const char *opSegmentSizeAttrInitCode = R"(
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("{0}");
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
  return isa<CodeInit>(valueInit) || isa<StringInit>(valueInit);
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
// Simple RAII helper for defining ifdef-undef-endif scopes.
class IfDefScope {
public:
  IfDefScope(StringRef name, raw_ostream &os) : name(name), os(os) {
    os << "#ifdef " << name << "\n"
       << "#undef " << name << "\n\n";
  }

  ~IfDefScope() { os << "\n#endif  // " << name << "\n\n"; }

private:
  StringRef name;
  raw_ostream &os;
};
} // end anonymous namespace

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
  void buildParamList(std::string &paramList,
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

  // Generate the side effect interface methods.
  void genSideEffectInterfaceMethods();

private:
  // The TableGen record for this op.
  // TODO(antiagainst,zinenko): OpEmitter should not have a Record directly,
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
  fctx.withBuilder("mlir::Builder(this->getContext())");

  // Emit the derived attribute body.
  auto emitDerivedAttr = [&](StringRef name, Attribute attr) {
    auto &method = opClass.newMethod(attr.getReturnType(), name);
    auto &body = method.body();
    body << "  " << attr.getDerivedCodeBody() << "\n";
  };

  // Emit with return type specified.
  auto emitAttrWithReturnType = [&](StringRef name, Attribute attr) {
    auto &method = opClass.newMethod(attr.getReturnType(), name);
    auto &body = method.body();
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
    auto &method =
        opClass.newMethod(attr.getStorageType(), (name + "Attr").str());
    auto &body = method.body();
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
    opClass.addTrait("DerivedAttributeOpInterface::Trait");
    // Generate helper method to query whether a named attribute is a derived
    // attribute. This enables, for example, avoiding adding an attribute that
    // overlaps with a derived attribute.
    {
      auto &method = opClass.newMethod("bool", "isDerivedAttribute",
                                       "StringRef name", OpMethod::MP_Static);
      auto &body = method.body();
      for (auto namedAttr : derivedAttrs)
        body << "  if (name == \"" << namedAttr.name << "\") return true;\n";
      body << " return false;";
    }
    // Generate method to materialize derived attributes as a DictionaryAttr.
    {
      OpMethod &method =
          opClass.newMethod("DictionaryAttr", "materializeDerivedAttributes");
      auto &body = method.body();

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

      body << "  MLIRContext* ctx = getContext();\n";
      body << "  Builder odsBuilder(ctx); (void)odsBuilder;\n";
      body << "  return DictionaryAttr::get({\n";
      interleave(
          derivedAttrs, body,
          [&](const NamedAttribute &namedAttr) {
            auto tmpl = namedAttr.attr.getConvertFromStorageCall();
            body << "    {Identifier::get(\"" << namedAttr.name << "\", ctx),\n"
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
    auto &method = opClass.newMethod("void", (name + "Attr").str(),
                                     (attr.getStorageType() + " attr").str());
    auto &body = method.body();
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
  auto &method = opClass.newMethod("std::pair<unsigned, unsigned>", methodName,
                                   "unsigned index");

  if (numVariadic == 0) {
    method.body() << "  return {index, 1};\n";
  } else if (hasAttrSegmentSize) {
    method.body() << sizeAttrInit << attrSizedSegmentValueRangeCalcCode;
  } else {
    // Because the op can have arbitrarily interleaved variadic and non-variadic
    // operands, we need to embed a list in the "sink" getter method for
    // calculation at run-time.
    llvm::SmallVector<StringRef, 4> isVariadic;
    isVariadic.reserve(llvm::size(odsValues));
    for (auto &it : odsValues)
      isVariadic.push_back(it.isVariableLength() ? "true" : "false");
    std::string isVariadicList = llvm::join(isVariadic, ", ");
    method.body() << formatv(sameVariadicSizeValueRangeCalcCode, isVariadicList,
                             numNonVariadic, numVariadic, rangeSizeCall,
                             "operand");
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
      op.getTrait("OpTrait::SameVariadicOperandSize");
  const auto *attrSizedOperands =
      op.getTrait("OpTrait::AttrSizedOperandSegments");

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

  auto &m = opClass.newMethod(rangeType, "getODSOperands", "unsigned index");
  m.body() << formatv(valueRangeReturnCode, rangeBeginCall,
                      "getODSOperandIndexAndLength(index)");

  // Then we emit nicer named getter methods by redirecting to the "sink" getter
  // method.
  for (int i = 0; i != numOperands; ++i) {
    const auto &operand = op.getOperand(i);
    if (operand.name.empty())
      continue;

    if (operand.isOptional()) {
      auto &m = opClass.newMethod("Value", operand.name);
      m.body() << "  auto operands = getODSOperands(" << i << ");\n"
               << "  return operands.empty() ? Value() : *operands.begin();";
    } else if (operand.isVariadic()) {
      auto &m = opClass.newMethod(rangeType, operand.name);
      m.body() << "  return getODSOperands(" << i << ");";
    } else {
      auto &m = opClass.newMethod("Value", operand.name);
      m.body() << "  return *getODSOperands(" << i << ").begin();";
    }
  }
}

void OpEmitter::genNamedOperandGetters() {
  generateNamedOperandGetters(
      op, opClass,
      /*sizeAttrInit=*/
      formatv(opSegmentSizeAttrInitCode, "operand_segment_sizes").str(),
      /*rangeType=*/"Operation::operand_range",
      /*rangeBeginCall=*/"getOperation()->operand_begin()",
      /*rangeSizeCall=*/"getOperation()->getNumOperands()",
      /*getOperandCallPattern=*/"getOperation()->getOperand({0})");
}

void OpEmitter::genNamedOperandSetters() {
  auto *attrSizedOperands = op.getTrait("OpTrait::AttrSizedOperandSegments");
  for (int i = 0, e = op.getNumOperands(); i != e; ++i) {
    const auto &operand = op.getOperand(i);
    if (operand.name.empty())
      continue;
    auto &m = opClass.newMethod("::mlir::MutableOperandRange",
                                (operand.name + "Mutable").str());
    auto &body = m.body();
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

  const auto *sameVariadicSize = op.getTrait("OpTrait::SameVariadicResultSize");
  const auto *attrSizedResults =
      op.getTrait("OpTrait::AttrSizedResultSegments");

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
  auto &m = opClass.newMethod("Operation::result_range", "getODSResults",
                              "unsigned index");
  m.body() << formatv(valueRangeReturnCode, "getOperation()->result_begin()",
                      "getODSResultIndexAndLength(index)");

  for (int i = 0; i != numResults; ++i) {
    const auto &result = op.getResult(i);
    if (result.name.empty())
      continue;

    if (result.isOptional()) {
      auto &m = opClass.newMethod("Value", result.name);
      m.body() << "  auto results = getODSResults(" << i << ");\n"
               << "  return results.empty() ? Value() : *results.begin();";
    } else if (result.isVariadic()) {
      auto &m = opClass.newMethod("Operation::result_range", result.name);
      m.body() << "  return getODSResults(" << i << ");";
    } else {
      auto &m = opClass.newMethod("Value", result.name);
      m.body() << "  return *getODSResults(" << i << ").begin();";
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
      auto &m = opClass.newMethod("MutableArrayRef<Region>", region.name);
      m.body() << formatv(
          "  return this->getOperation()->getRegions().drop_front({0});", i);
      continue;
    }

    auto &m = opClass.newMethod("Region &", region.name);
    m.body() << formatv("  return this->getOperation()->getRegion({0});", i);
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
      auto &m = opClass.newMethod("SuccessorRange", successor.name);
      m.body() << formatv(
          "  return {std::next(this->getOperation()->successor_begin(), {0}), "
          "this->getOperation()->successor_end()};",
          i);
      continue;
    }

    auto &m = opClass.newMethod("Block *", successor.name);
    m.body() << formatv("  return this->getOperation()->getSuccessor({0});", i);
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

void OpEmitter::genSeparateArgParamBuilder() {
  SmallVector<AttrParamKind, 2> attrBuilderType;
  attrBuilderType.push_back(AttrParamKind::WrappedAttr);
  if (canGenerateUnwrappedBuilder(op))
    attrBuilderType.push_back(AttrParamKind::UnwrappedValue);

  // Emit with separate builders with or without unwrapped attributes and/or
  // inferring result type.
  auto emit = [&](AttrParamKind attrType, TypeParamKind paramKind,
                  bool inferType) {
    std::string paramList;
    llvm::SmallVector<std::string, 4> resultNames;
    buildParamList(paramList, resultNames, paramKind, attrType);

    auto &m =
        opClass.newMethod("void", "build", paramList, OpMethod::MP_Static);
    auto &body = m.body();

    genCodeForAddingArgAndRegionForBuilder(
        body, /*isRawValueAttr=*/attrType == AttrParamKind::UnwrappedValue);

    // Push all result types to the operation state

    if (inferType) {
      // Generate builder that infers type too.
      // TODO(jpienaar): Subsume this with general checking if type can be
      // inferred automatically.
      // TODO(jpienaar): Expand to handle regions.
      body << formatv(R"(
        SmallVector<Type, 2> inferredReturnTypes;
        if (succeeded({0}::inferReturnTypes(odsBuilder.getContext(),
                      {1}.location, {1}.operands,
                      {1}.attributes.getDictionary({1}.getContext()),
                      /*regions=*/{{}, inferredReturnTypes)))
          {1}.addTypes(inferredReturnTypes);
        else
          llvm::report_fatal_error("Failed to infer result type(s).");)",
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
    case TypeParamKind::Collective:
      body << "  "
           << "assert(resultTypes.size() "
           << (op.getNumVariableLengthResults() == 0 ? "==" : ">=") << " "
           << (op.getNumResults() - op.getNumVariableLengthResults())
           << "u && \"mismatched number of results\");\n";
      body << "  " << builderOpState << ".addTypes(resultTypes);\n";
      return;
    }
    llvm_unreachable("unhandled TypeParamKind");
  };

  bool canInferType =
      op.getTrait("InferTypeOpInterface::Trait") && op.getNumRegions() == 0;
  for (auto attrType : attrBuilderType) {
    emit(attrType, TypeParamKind::Separate, /*inferType=*/false);
    if (canInferType)
      emit(attrType, TypeParamKind::None, /*inferType=*/true);
    // Emit separate arg build with collective type, unless there is only one
    // variadic result, in which case the above would have already generated
    // the same build method.
    if (!(op.getNumResults() == 1 && op.getResult(0).isVariableLength()))
      emit(attrType, TypeParamKind::Collective, /*inferType=*/false);
  }
}

void OpEmitter::genUseOperandAsResultTypeCollectiveParamBuilder() {
  // If this op has a variadic result, we cannot generate this builder because
  // we don't know how many results to create.
  if (op.getNumVariableLengthResults() != 0)
    return;

  int numResults = op.getNumResults();

  // Signature
  std::string params =
      std::string("OpBuilder &odsBuilder, OperationState &") + builderOpState +
      ", ValueRange operands, ArrayRef<NamedAttribute> attributes";
  if (op.getNumVariadicRegions())
    params += ", unsigned numRegions";
  auto &m = opClass.newMethod("void", "build", params, OpMethod::MP_Static);
  auto &body = m.body();

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
  // TODO(jpienaar): Expand to support regions.
  const char *params =
      "OpBuilder &odsBuilder, OperationState &{0}, "
      "ValueRange operands, ArrayRef<NamedAttribute> attributes";
  auto &m =
      opClass.newMethod("void", "build", formatv(params, builderOpState).str(),
                        OpMethod::MP_Static);
  auto &body = m.body();

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
    SmallVector<Type, 2> inferredReturnTypes;
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
      llvm::report_fatal_error("Failed to infer result type(s).");)",
                  opClass.getClassName(), builderOpState);
}

void OpEmitter::genUseOperandAsResultTypeSeparateParamBuilder() {
  std::string paramList;
  llvm::SmallVector<std::string, 4> resultNames;
  buildParamList(paramList, resultNames, TypeParamKind::None);

  auto &m = opClass.newMethod("void", "build", paramList, OpMethod::MP_Static);
  genCodeForAddingArgAndRegionForBuilder(m.body());

  auto numResults = op.getNumResults();
  if (numResults == 0)
    return;

  // Push all result types to the operation state
  const char *index = op.getOperand(0).isVariadic() ? ".front()" : "";
  std::string resultType =
      formatv("{0}{1}.getType()", getArgumentName(op, 0), index).str();
  m.body() << "  " << builderOpState << ".addTypes({" << resultType;
  for (int i = 1; i != numResults; ++i)
    m.body() << ", " << resultType;
  m.body() << "});\n\n";
}

void OpEmitter::genUseAttrAsResultTypeBuilder() {
  std::string params =
      std::string("OpBuilder &odsBuilder, OperationState &") + builderOpState +
      ", ValueRange operands, ArrayRef<NamedAttribute> attributes";
  auto &m = opClass.newMethod("void", "build", params, OpMethod::MP_Static);
  auto &body = m.body();

  // Push all result types to the operation state
  std::string resultType;
  const auto &namedAttr = op.getAttribute(0);

  body << "  for (auto attr : attributes) {\n";
  body << "    if (attr.first != \"" << namedAttr.name << "\") continue;\n";
  if (namedAttr.attr.isTypeAttr()) {
    resultType = "attr.second.cast<TypeAttr>().getValue()";
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
  // TODO(antiagainst): Create wrapper class for OpBuilder to hide the native
  // TableGen API calls here.
  {
    auto *listInit = dyn_cast_or_null<ListInit>(def.getValueInit("builders"));
    if (listInit) {
      for (Init *init : listInit->getValues()) {
        Record *builderDef = cast<DefInit>(init)->getDef();
        StringRef params = builderDef->getValueAsString("params");
        StringRef body = builderDef->getValueAsString("body");
        bool hasBody = !body.empty();

        auto &method =
            opClass.newMethod("void", "build", params, OpMethod::MP_Static,
                              /*declOnly=*/!hasBody);
        if (hasBody)
          method.body() << body;
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
    if (op.getTrait("OpTrait::SameOperandsAndResultType")) {
      genUseOperandAsResultTypeSeparateParamBuilder();
      genUseOperandAsResultTypeCollectiveParamBuilder();
    }
    if (op.getTrait("OpTrait::FirstAttrDerivedResultType"))
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
  // Signature
  std::string params = std::string("OpBuilder &, OperationState &") +
                       builderOpState +
                       ", ArrayRef<Type> resultTypes, ValueRange operands, "
                       "ArrayRef<NamedAttribute> attributes";
  if (op.getNumVariadicRegions())
    params += ", unsigned numRegions";
  auto &m = opClass.newMethod("void", "build", params, OpMethod::MP_Static);
  auto &body = m.body();

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
  // TODO(jpienaar): Subsume this with general checking if type can be inferred
  // automatically.
  // TODO(jpienaar): Expand to handle regions and successors.
  if (op.getTrait("InferTypeOpInterface::Trait") && op.getNumRegions() == 0 &&
      op.getNumSuccessors() == 0)
    genInferredTypeCollectiveParamBuilder();
}

void OpEmitter::buildParamList(std::string &paramList,
                               SmallVectorImpl<std::string> &resultTypeNames,
                               TypeParamKind typeParamKind,
                               AttrParamKind attrParamKind) {
  resultTypeNames.clear();
  auto numResults = op.getNumResults();
  resultTypeNames.reserve(numResults);

  paramList = "OpBuilder &odsBuilder, OperationState &";
  paramList.append(builderOpState);

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

      if (result.isOptional())
        paramList.append(", /*optional*/Type ");
      else if (result.isVariadic())
        paramList.append(", ArrayRef<Type> ");
      else
        paramList.append(", Type ");
      paramList.append(resultName);

      resultTypeNames.emplace_back(std::move(resultName));
    }
  } break;
  case TypeParamKind::Collective: {
    paramList.append(", ArrayRef<Type> resultTypes");
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
      // TODO(b/144412160) Adjust the 'returnType' field of such attributes
      // to support them.
      StringRef retType = namedAttr->attr.getReturnType();
      if (retType == "APInt" || retType == "APFloat")
        break;

      defaultValuedAttrStartIndex = i;
    }
  }

  for (int i = 0, e = op.getNumArgs(); i < e; ++i) {
    auto argument = op.getArg(i);
    if (argument.is<tblgen::NamedTypeConstraint *>()) {
      const auto &operand = op.getOperand(numOperands);
      if (operand.isOptional())
        paramList.append(", /*optional*/Value ");
      else if (operand.isVariadic())
        paramList.append(", ValueRange ");
      else
        paramList.append(", Value ");
      paramList.append(getArgumentName(op, numOperands));
      ++numOperands;
    } else {
      const auto &namedAttr = op.getAttribute(numAttrs);
      const auto &attr = namedAttr.attr;
      paramList.append(", ");

      if (attr.isOptional())
        paramList.append("/*optional*/");

      switch (attrParamKind) {
      case AttrParamKind::WrappedAttr:
        paramList.append(std::string(attr.getStorageType()));
        break;
      case AttrParamKind::UnwrappedValue:
        if (canUseUnwrappedRawValue(attr)) {
          paramList.append(std::string(attr.getReturnType()));
        } else {
          paramList.append(std::string(attr.getStorageType()));
        }
        break;
      }
      paramList.append(" ");
      paramList.append(std::string(namedAttr.name));

      // Attach default value if requested and possible.
      if (attrParamKind == AttrParamKind::UnwrappedValue &&
          i >= defaultValuedAttrStartIndex) {
        bool isString = attr.getReturnType() == "StringRef";
        paramList.append(" = ");
        if (isString)
          paramList.append("\"");
        paramList.append(std::string(attr.getDefaultValue()));
        if (isString)
          paramList.append("\"");
      }
      ++numAttrs;
    }
  }

  /// Insert parameters for each successor.
  for (const NamedSuccessor &succ : op.getSuccessors()) {
    paramList += (succ.isVariadic() ? ", ArrayRef<Block *> " : ", Block *");
    paramList += succ.name;
  }

  /// Insert parameters for variadic regions.
  for (const NamedRegion &region : op.getRegions()) {
    if (region.isVariadic())
      paramList += llvm::formatv(", unsigned {0}Count", region.name).str();
  }
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
  if (op.getTrait("OpTrait::AttrSizedOperandSegments")) {
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

  const char *const params =
      "OwningRewritePatternList &results, MLIRContext *context";
  opClass.newMethod("void", "getCanonicalizationPatterns", params,
                    OpMethod::MP_Static, /*declOnly=*/true);
}

void OpEmitter::genFolderDecls() {
  bool hasSingleResult =
      op.getNumResults() == 1 && op.getNumVariableLengthResults() == 0;

  if (def.getValueAsBit("hasFolder")) {
    if (hasSingleResult) {
      const char *const params = "ArrayRef<Attribute> operands";
      opClass.newMethod("OpFoldResult", "fold", params, OpMethod::MP_None,
                        /*declOnly=*/true);
    } else {
      const char *const params = "ArrayRef<Attribute> operands, "
                                 "SmallVectorImpl<OpFoldResult> &results";
      opClass.newMethod("LogicalResult", "fold", params, OpMethod::MP_None,
                        /*declOnly=*/true);
    }
  }
}

void OpEmitter::genOpInterfaceMethods() {
  for (const auto &trait : op.getTraits()) {
    auto opTrait = dyn_cast<tblgen::InterfaceOpTrait>(&trait);
    if (!opTrait || !opTrait->shouldDeclareMethods())
      continue;
    auto interface = opTrait->getOpInterface();

    // Get the set of methods that should always be declared.
    auto alwaysDeclaredMethodsVec = opTrait->getAlwaysDeclaredMethods();
    llvm::StringSet<> alwaysDeclaredMethods;
    alwaysDeclaredMethods.insert(alwaysDeclaredMethodsVec.begin(),
                                 alwaysDeclaredMethodsVec.end());

    for (const OpInterfaceMethod &method : interface.getMethods()) {
      // Don't declare if the method has a body.
      if (method.getBody())
        continue;
      // Don't declare if the method has a default implementation and the op
      // didn't request that it always be declared.
      if (method.getDefaultImplementation() &&
          !alwaysDeclaredMethods.count(method.getName()))
        continue;

      std::string args;
      llvm::raw_string_ostream os(args);
      interleaveComma(method.getArguments(), os,
                      [&](const OpInterfaceMethod::Argument &arg) {
                        os << arg.type << " " << arg.name;
                      });
      opClass.newMethod(method.getReturnType(), method.getName(), os.str(),
                        method.isStatic() ? OpMethod::MP_Static
                                          : OpMethod::MP_None,
                        /*declOnly=*/true);
    }
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
    auto effectsParam =
        llvm::formatv(
            "SmallVectorImpl<SideEffects::EffectInstance<{0}>> &effects",
            it.first())
            .str();

    // Generate the 'getEffects' method.
    auto &getEffects = opClass.newMethod("void", "getEffects", effectsParam);
    auto &body = getEffects.body();

    // Add effect instances for each of the locations marked on the operation.
    for (auto &location : it.second) {
      if (location.kind != EffectKind::Static) {
        body << "  for (Value value : getODS"
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

void OpEmitter::genParser() {
  if (!hasStringAttribute(def, "parser") ||
      hasStringAttribute(def, "assemblyFormat"))
    return;

  auto &method = opClass.newMethod(
      "ParseResult", "parse", "OpAsmParser &parser, OperationState &result",
      OpMethod::MP_Static);
  FmtContext fctx;
  fctx.addSubst("cppClass", opClass.getClassName());
  auto parser = def.getValueAsString("parser").ltrim().rtrim(" \t\v\f\r");
  method.body() << "  " << tgfmt(parser, &fctx);
}

void OpEmitter::genPrinter() {
  if (hasStringAttribute(def, "assemblyFormat"))
    return;

  auto valueInit = def.getValueInit("printer");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  if (!codeInit)
    return;

  auto &method = opClass.newMethod("void", "print", "OpAsmPrinter &p");
  FmtContext fctx;
  fctx.addSubst("cppClass", opClass.getClassName());
  auto printer = codeInit->getValue().ltrim().rtrim(" \t\v\f\r");
  method.body() << "  " << tgfmt(printer, &fctx);
}

void OpEmitter::genVerifier() {
  auto valueInit = def.getValueInit("verifier");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  bool hasCustomVerify = codeInit && !codeInit->getValue().empty();

  auto &method = opClass.newMethod("LogicalResult", "verify", /*params=*/"");
  auto &body = method.body();

  const char *checkAttrSizedValueSegmentsCode = R"(
  {
    auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("{0}");
    auto numElements = sizeAttr.getType().cast<ShapedType>().getNumElements();
    if (numElements != {1}) {{
      return emitOpError("'{0}' attribute for specifying {2} segments "
                         "must have {1} elements");
    }
  }
  )";

  // Verify a few traits first so that we can use
  // getODSOperands()/getODSResults() in the rest of the verifier.
  for (auto &trait : op.getTraits()) {
    if (auto *t = dyn_cast<tblgen::NativeOpTrait>(&trait)) {
      if (t->getTrait() == "OpTrait::AttrSizedOperandSegments") {
        body << formatv(checkAttrSizedValueSegmentsCode,
                        "operand_segment_sizes", op.getNumOperands(),
                        "operand");
      } else if (t->getTrait() == "OpTrait::AttrSizedResultSegments") {
        body << formatv(checkAttrSizedValueSegmentsCode, "result_segment_sizes",
                        op.getNumResults(), "result");
      }
    }
  }

  // Populate substitutions for attributes and named operands and results.
  for (const auto &namedAttr : op.getAttributes())
    verifyCtx.addSubst(namedAttr.name,
                       formatv("this->getAttr(\"{0}\")", namedAttr.name));
  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    auto &value = op.getOperand(i);
    if (value.name.empty())
      continue;

    if (value.isVariadic())
      verifyCtx.addSubst(value.name, formatv("this->getODSOperands({0})", i));
    else
      verifyCtx.addSubst(value.name,
                         formatv("(*this->getODSOperands({0}).begin())", i));
  }
  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    auto &value = op.getResult(i);
    if (value.name.empty())
      continue;

    if (value.isVariadic())
      verifyCtx.addSubst(value.name, formatv("this->getODSResults({0})", i));
    else
      verifyCtx.addSubst(value.name,
                         formatv("(*this->getODSResults({0}).begin())", i));
  }

  // Verify the attributes have the correct type.
  for (const auto &namedAttr : op.getAttributes()) {
    const auto &attr = namedAttr.attr;
    if (attr.isDerivedAttr())
      continue;

    auto attrName = namedAttr.name;
    // Prefix with `tblgen_` to avoid hiding the attribute accessor.
    auto varName = tblgenNamePrefix + attrName;
    body << formatv("  auto {0} = this->getAttr(\"{1}\");\n", varName,
                    attrName);

    bool allowMissingAttr = attr.hasDefaultValue() || attr.isOptional();
    if (allowMissingAttr) {
      // If the attribute has a default value, then only verify the predicate if
      // set. This does effectively assume that the default value is valid.
      // TODO: verify the debug value is valid (perhaps in debug mode only).
      body << "  if (" << varName << ") {\n";
    } else {
      body << "  if (!" << varName
           << ") return emitOpError(\"requires attribute '" << attrName
           << "'\");\n  {\n";
    }

    auto attrPred = attr.getPredicate();
    if (!attrPred.isNull()) {
      body << tgfmt(
          "    if (!($0)) return emitOpError(\"attribute '$1' "
          "failed to satisfy constraint: $2\");\n",
          /*ctx=*/nullptr,
          tgfmt(attrPred.getCondition(), &verifyCtx.withSelf(varName)),
          attrName, attr.getDescription());
    }

    body << "  }\n";
  }

  genOperandResultVerifier(body, op.getOperands(), "operand");
  genOperandResultVerifier(body, op.getResults(), "result");

  for (auto &trait : op.getTraits()) {
    if (auto *t = dyn_cast<tblgen::PredOpTrait>(&trait)) {
      body << tgfmt("  if (!($0)) {\n    "
                    "return emitOpError(\"failed to verify that $1\");\n  }\n",
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
    body << "  return mlir::success();\n";
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
    body << "    for (Value v : valueGroup" << staticValue.index() << ") {\n";

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

    body << "    for (Region &region : ";
    body << formatv(
        region.isVariadic()
            ? "{0}()"
            : "MutableArrayRef<Region>(this->getOperation()->getRegion({1}))",
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

    body << "    for (Block *successor : ";
    body << formatv(successor.isVariadic() ? "{0}()"
                                           : "ArrayRef<Block *>({0}())",
                    successor.name);
    body << ") {\n";
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
      opClass.addTrait("OpTrait::Variadic" + traitKind + "s");
    else
      opClass.addTrait("OpTrait::AtLeastN" + traitKind + "s<" +
                       Twine(numTotal - numVariadic) + ">::Impl");
    return;
  }
  switch (numTotal) {
  case 0:
    opClass.addTrait("OpTrait::Zero" + traitKind);
    break;
  case 1:
    opClass.addTrait("OpTrait::One" + traitKind);
    break;
  default:
    opClass.addTrait("OpTrait::N" + traitKind + "s<" + Twine(numTotal) +
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
      opClass.addTrait("OpTrait::VariadicOperands");
    else
      opClass.addTrait("OpTrait::AtLeastNOperands<" +
                       Twine(numOperands - numVariadicOperands) + ">::Impl");
  } else {
    switch (numOperands) {
    case 0:
      opClass.addTrait("OpTrait::ZeroOperands");
      break;
    case 1:
      opClass.addTrait("OpTrait::OneOperand");
      break;
    default:
      opClass.addTrait("OpTrait::NOperands<" + Twine(numOperands) + ">::Impl");
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
  auto &method = opClass.newMethod("StringRef", "getOperationName",
                                   /*params=*/"", OpMethod::MP_Static);
  method.body() << "  return \"" << op.getOperationName() << "\";\n";
}

void OpEmitter::genOpAsmInterface() {
  // If the user only has one results or specifically added the Asm trait,
  // then don't generate it for them. We specifically only handle multi result
  // operations, because the name of a single result in the common case is not
  // interesting(generally 'result'/'output'/etc.).
  // TODO: We could also add a flag to allow operations to opt in to this
  // generation, even if they only have a single operation.
  int numResults = op.getNumResults();
  if (numResults <= 1 || op.getTrait("OpAsmOpInterface::Trait"))
    return;

  SmallVector<StringRef, 4> resultNames(numResults);
  for (int i = 0; i != numResults; ++i)
    resultNames[i] = op.getResultName(i);

  // Don't add the trait if none of the results have a valid name.
  if (llvm::all_of(resultNames, [](StringRef name) { return name.empty(); }))
    return;
  opClass.addTrait("OpAsmOpInterface::Trait");

  // Generate the right accessor for the number of results.
  auto &method = opClass.newMethod("void", "getAsmResultNames",
                                   "OpAsmSetValueNameFn setNameFn");
  auto &body = method.body();
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

  Class adapterClass;
};
} // end namespace

OpOperandAdaptorEmitter::OpOperandAdaptorEmitter(const Operator &op)
    : adapterClass(op.getCppClassName().str() + "OperandAdaptor") {
  adapterClass.newField("ArrayRef<Value>", "odsOperands");
  adapterClass.newField("DictionaryAttr", "odsAttrs");
  const auto *attrSizedOperands =
      op.getTrait("OpTrait::AttrSizedOperandSegments");
  auto &constructor = adapterClass.newConstructor(
      attrSizedOperands
          ? "ArrayRef<Value> values, DictionaryAttr attrs"
          : "ArrayRef<Value> values, DictionaryAttr attrs = nullptr");
  constructor.body() << "  odsOperands = values;\n";
  constructor.body() << "  odsAttrs = attrs;\n";

  std::string sizeAttrInit =
      formatv(adapterSegmentSizeAttrInitCode, "operand_segment_sizes");
  generateNamedOperandGetters(op, adapterClass, sizeAttrInit,
                              /*rangeType=*/"ArrayRef<Value>",
                              /*rangeBeginCall=*/"odsOperands.begin()",
                              /*rangeSizeCall=*/"odsOperands.size()",
                              /*getOperandCallPattern=*/"odsOperands[{0}]");

  FmtContext fctx;
  fctx.withBuilder("mlir::Builder(odsAttrs.getContext())");

  auto emitAttr = [&](StringRef name, Attribute attr) {
    auto &body = adapterClass.newMethod(attr.getStorageType(), name).body();
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
}

void OpOperandAdaptorEmitter::emitDecl(const Operator &op, raw_ostream &os) {
  OpOperandAdaptorEmitter(op).adapterClass.writeDeclTo(os);
}

void OpOperandAdaptorEmitter::emitDef(const Operator &op, raw_ostream &os) {
  OpOperandAdaptorEmitter(op).adapterClass.writeDefTo(os);
}

// Emits the opcode enum and op classes.
static void emitOpClasses(const std::vector<Record *> &defs, raw_ostream &os,
                          bool emitDecl) {
  IfDefScope scope("GET_OP_CLASSES", os);
  // First emit forward declaration for each class, this allows them to refer
  // to each others in traits for example.
  if (emitDecl) {
    for (auto *def : defs) {
      Operator op(*def);
      os << "class " << op.getCppClassName() << ";\n";
    }
  }
  for (auto *def : defs) {
    Operator op(*def);
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

static bool emitOpDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op Declarations", os);

  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");
  emitOpClasses(defs, os, /*emitDecl=*/true);

  return false;
}

static bool emitOpDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op Definitions", os);

  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");
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
