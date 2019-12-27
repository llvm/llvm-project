//===- SPIRVSerializationGen.cpp - SPIR-V serialization utility generator -===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPIRVSerializationGen generates common utility functions for SPIR-V
// serialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/StringExtras.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::ArrayRef;
using llvm::formatv;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::SmallVector;
using llvm::SMLoc;
using llvm::StringMap;
using llvm::StringRef;
using llvm::Twine;
using mlir::tblgen::Attribute;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::NamedAttribute;
using mlir::tblgen::NamedTypeConstraint;
using mlir::tblgen::Operator;

//===----------------------------------------------------------------------===//
// Availability Wrapper Class
//===----------------------------------------------------------------------===//

namespace {
// Wrapper class with helper methods for accessing availability defined in
// TableGen.
class Availability {
public:
  explicit Availability(const Record *def);

  // Returns the name of the direct TableGen class for this availability
  // instance.
  StringRef getClass() const;

  // Returns the generated C++ interface's class name.
  StringRef getInterfaceClassName() const;

  // Returns the generated C++ interface's description.
  StringRef getInterfaceDescription() const;

  // Returns the name of the query function insided the generated C++ interface.
  StringRef getQueryFnName() const;

  // Returns the return type of the query function insided the generated C++
  // interface.
  StringRef getQueryFnRetType() const;

  // Returns the code for merging availability requirements.
  StringRef getMergeActionCode() const;

  // Returns the initializer expression for initializing the final availability
  // requirements.
  StringRef getMergeInitializer() const;

  // Returns the C++ type for an availability instance.
  StringRef getMergeInstanceType() const;

  // Returns the concrete availability instance carried in this case.
  StringRef getMergeInstance() const;

private:
  // The TableGen definition of this availability.
  const llvm::Record *def;
};
} // namespace

Availability::Availability(const llvm::Record *def) : def(def) {
  assert(def->isSubClassOf("Availability") &&
         "must be subclass of TableGen 'Availability' class");
}

StringRef Availability::getClass() const {
  SmallVector<Record *, 1> parentClass;
  def->getDirectSuperClasses(parentClass);
  if (parentClass.size() != 1) {
    PrintFatalError(def->getLoc(),
                    "expected to only have one direct superclass");
  }
  return parentClass.front()->getName();
}

StringRef Availability::getInterfaceClassName() const {
  return def->getValueAsString("interfaceName");
}

StringRef Availability::getInterfaceDescription() const {
  return def->getValueAsString("interfaceDescription");
}

StringRef Availability::getQueryFnRetType() const {
  return def->getValueAsString("queryFnRetType");
}

StringRef Availability::getQueryFnName() const {
  return def->getValueAsString("queryFnName");
}

StringRef Availability::getMergeActionCode() const {
  return def->getValueAsString("mergeAction");
}

StringRef Availability::getMergeInitializer() const {
  return def->getValueAsString("initializer");
}

StringRef Availability::getMergeInstanceType() const {
  return def->getValueAsString("instanceType");
}

StringRef Availability::getMergeInstance() const {
  return def->getValueAsString("instance");
}

//===----------------------------------------------------------------------===//
// Availability Interface Definitions AutoGen
//===----------------------------------------------------------------------===//

static void emitInterfaceDef(const Availability &availability,
                             raw_ostream &os) {
  StringRef methodName = availability.getQueryFnName();
  os << availability.getQueryFnRetType() << " "
     << availability.getInterfaceClassName() << "::" << methodName << "() {\n"
     << "  return getImpl()->" << methodName << "(getOperation());\n"
     << "}\n";
}

static bool emitInterfaceDefs(const RecordKeeper &recordKeeper,
                              raw_ostream &os) {
  llvm::emitSourceFileHeader("Availability Interface Definitions", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("Availability");
  SmallVector<const Record *, 1> handledClasses;
  for (const Record *def : defs) {
    SmallVector<Record *, 1> parent;
    def->getDirectSuperClasses(parent);
    if (parent.size() != 1) {
      PrintFatalError(def->getLoc(),
                      "expected to only have one direct superclass");
    }
    if (llvm::is_contained(handledClasses, parent.front()))
      continue;

    Availability availability(def);
    emitInterfaceDef(availability, os);
    handledClasses.push_back(parent.front());
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Availability Interface Declarations AutoGen
//===----------------------------------------------------------------------===//

static void emitConceptDecl(const Availability &availability, raw_ostream &os) {
  os << "  class Concept {\n"
     << "  public:\n"
     << "    virtual ~Concept() = default;\n"
     << "    virtual " << availability.getQueryFnRetType() << " "
     << availability.getQueryFnName() << "(Operation *tblgen_opaque_op) = 0;\n"
     << "  };\n";
}

static void emitModelDecl(const Availability &availability, raw_ostream &os) {
  os << "  template<typename ConcreteOp>\n";
  os << "  class Model : public Concept {\n"
     << "  public:\n"
     << "    " << availability.getQueryFnRetType() << " "
     << availability.getQueryFnName()
     << "(Operation *tblgen_opaque_op) final {\n"
     << "      auto op = llvm::cast<ConcreteOp>(tblgen_opaque_op);\n"
     << "      (void)op;\n"
     // Forward to the method on the concrete operation type.
     << "      return op." << availability.getQueryFnName() << "();\n"
     << "    }\n"
     << "  };\n";
}

static void emitInterfaceDecl(const Availability &availability,
                              raw_ostream &os) {
  StringRef interfaceName = availability.getInterfaceClassName();
  std::string interfaceTraitsName = formatv("{0}Traits", interfaceName);

  // Emit the traits struct containing the concept and model declarations.
  os << "namespace detail {\n"
     << "struct " << interfaceTraitsName << " {\n";
  emitConceptDecl(availability, os);
  os << '\n';
  emitModelDecl(availability, os);
  os << "};\n} // end namespace detail\n\n";

  // Emit the main interface class declaration.
  os << "/*\n" << availability.getInterfaceDescription().trim() << "\n*/\n";
  os << llvm::formatv("class {0} : public OpInterface<{1}, detail::{2}> {\n"
                      "public:\n"
                      "  using OpInterface<{1}, detail::{2}>::OpInterface;\n",
                      interfaceName, interfaceName, interfaceTraitsName);

  // Emit query function declaration.
  os << "  " << availability.getQueryFnRetType() << " "
     << availability.getQueryFnName() << "();\n";
  os << "};\n\n";
}

static bool emitInterfaceDecls(const RecordKeeper &recordKeeper,
                               raw_ostream &os) {
  llvm::emitSourceFileHeader("Availability Interface Declarations", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("Availability");
  SmallVector<const Record *, 4> handledClasses;
  for (const Record *def : defs) {
    SmallVector<Record *, 1> parent;
    def->getDirectSuperClasses(parent);
    if (parent.size() != 1) {
      PrintFatalError(def->getLoc(),
                      "expected to only have one direct superclass");
    }
    if (llvm::is_contained(handledClasses, parent.front()))
      continue;

    Availability avail(def);
    emitInterfaceDecl(avail, os);
    handledClasses.push_back(parent.front());
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Availability Interface Hook Registration
//===----------------------------------------------------------------------===//

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genInterfaceDecls("gen-avail-interface-decls",
                      "Generate availability interface declarations",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return emitInterfaceDecls(records, os);
                      });

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genInterfaceDefs("gen-avail-interface-defs",
                     "Generate op interface definitions",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       return emitInterfaceDefs(records, os);
                     });

//===----------------------------------------------------------------------===//
// Serialization AutoGen
//===----------------------------------------------------------------------===//

// Writes the following function to `os`:
//   inline uint32_t getOpcode(<op-class-name>) { return <opcode>; }
static void emitGetOpcodeFunction(const Record *record, Operator const &op,
                                  raw_ostream &os) {
  os << formatv("template <> constexpr inline ::mlir::spirv::Opcode "
                "getOpcode<{0}>() {{\n",
                op.getQualCppClassName());
  os << formatv("  return ::mlir::spirv::Opcode::{0};\n",
                record->getValueAsString("spirvOpName"));
  os << "}\n";
}

/// Forward declaration of function to return the SPIR-V opcode corresponding to
/// an operation. This function will be generated for all SPV_Op instances that
/// have hasOpcode = 1.
static void declareOpcodeFn(raw_ostream &os) {
  os << "template <typename OpClass> inline constexpr ::mlir::spirv::Opcode "
        "getOpcode();\n";
}

/// Generates code to serialize attributes of a SPV_Op `op` into `os`. The
/// generates code extracts the attribute with name `attrName` from
/// `operandList` of `op`.
static void emitAttributeSerialization(const Attribute &attr,
                                       ArrayRef<SMLoc> loc, StringRef tabs,
                                       StringRef opVar, StringRef operandList,
                                       StringRef attrName, raw_ostream &os) {
  os << tabs << formatv("auto attr = {0}.getAttr(\"{1}\");\n", opVar, attrName);
  os << tabs << "if (attr) {\n";
  if (attr.getAttrDefName() == "SPV_ScopeAttr" ||
      attr.getAttrDefName() == "SPV_MemorySemanticsAttr") {
    os << tabs
       << formatv("  {0}.push_back(prepareConstantInt({1}.getLoc(), "
                  "attr.cast<IntegerAttr>()));\n",
                  operandList, opVar);
  } else if (attr.getAttrDefName() == "I32ArrayAttr") {
    // Serialize all the elements of the array
    os << tabs << "  for (auto attrElem : attr.cast<ArrayAttr>()) {\n";
    os << tabs
       << formatv("    {0}.push_back(static_cast<uint32_t>("
                  "attrElem.cast<IntegerAttr>().getValue().getZExtValue()));\n",
                  operandList);
    os << tabs << "  }\n";
  } else if (attr.isEnumAttr() || attr.getAttrDefName() == "I32Attr") {
    os << tabs
       << formatv("  {0}.push_back(static_cast<uint32_t>("
                  "attr.cast<IntegerAttr>().getValue().getZExtValue()));\n",
                  operandList);
  } else {
    PrintFatalError(
        loc,
        llvm::Twine(
            "unhandled attribute type in SPIR-V serialization generation : '") +
            attr.getAttrDefName() + llvm::Twine("'"));
  }
  os << tabs << "}\n";
}

/// Generates code to serialize the operands of a SPV_Op `op` into `os`. The
/// generated queries the SSA-ID if operand is a SSA-Value, or serializes the
/// attributes. The `operands` vector is updated appropriately. `elidedAttrs`
/// updated as well to include the serialized attributes.
static void emitOperandSerialization(const Operator &op, ArrayRef<SMLoc> loc,
                                     StringRef tabs, StringRef opVar,
                                     StringRef operands, StringRef elidedAttrs,
                                     raw_ostream &os) {
  auto operandNum = 0;
  for (unsigned i = 0, e = op.getNumArgs(); i < e; ++i) {
    auto argument = op.getArg(i);
    os << tabs << "{\n";
    if (argument.is<NamedTypeConstraint *>()) {
      os << tabs
         << formatv("  for (auto arg : {0}.getODSOperands({1})) {{\n", opVar,
                    operandNum);
      os << tabs << "    auto argID = getValueID(arg);\n";
      os << tabs << "    if (!argID) {\n";
      os << tabs
         << formatv("      return emitError({0}.getLoc(), "
                    "\"operand #{1} has a use before def\");\n",
                    opVar, operandNum);
      os << tabs << "    }\n";
      os << tabs << formatv("    {0}.push_back(argID);\n", operands);
      os << "    }\n";
      operandNum++;
    } else {
      auto attr = argument.get<NamedAttribute *>();
      auto newtabs = tabs.str() + "  ";
      emitAttributeSerialization(
          (attr->attr.isOptional() ? attr->attr.getBaseAttr() : attr->attr),
          loc, newtabs, opVar, operands, attr->name, os);
      os << newtabs
         << formatv("{0}.push_back(\"{1}\");\n", elidedAttrs, attr->name);
    }
    os << tabs << "}\n";
  }
}

/// Generates code to serializes the result of SPV_Op `op` into `os`. The
/// generated gets the ID for the type of the result (if any), the SSA-ID of
/// the result and updates `resultID` with the SSA-ID.
static void emitResultSerialization(const Operator &op, ArrayRef<SMLoc> loc,
                                    StringRef tabs, StringRef opVar,
                                    StringRef operands, StringRef resultID,
                                    raw_ostream &os) {
  if (op.getNumResults() == 1) {
    StringRef resultTypeID("resultTypeID");
    os << tabs << formatv("uint32_t {0} = 0;\n", resultTypeID);
    os << tabs
       << formatv(
              "if (failed(processType({0}.getLoc(), {0}.getType(), {1}))) {{\n",
              opVar, resultTypeID);
    os << tabs << "  return failure();\n";
    os << tabs << "}\n";
    os << tabs << formatv("{0}.push_back({1});\n", operands, resultTypeID);
    // Create an SSA result <id> for the op
    os << tabs << formatv("{0} = getNextID();\n", resultID);
    os << tabs
       << formatv("valueIDMap[{0}.getResult()] = {1};\n", opVar, resultID);
    os << tabs << formatv("{0}.push_back({1});\n", operands, resultID);
  } else if (op.getNumResults() != 0) {
    PrintFatalError(loc, "SPIR-V ops can only have zero or one result");
  }
}

/// Generates code to serialize attributes of SPV_Op `op` that become
/// decorations on the `resultID` of the serialized operation `opVar` in the
/// SPIR-V binary.
static void emitDecorationSerialization(const Operator &op, StringRef tabs,
                                        StringRef opVar, StringRef elidedAttrs,
                                        StringRef resultID, raw_ostream &os) {
  if (op.getNumResults() == 1) {
    // All non-argument attributes translated into OpDecorate instruction
    os << tabs << formatv("for (auto attr : {0}.getAttrs()) {{\n", opVar);
    os << tabs
       << formatv("  if (llvm::any_of({0}, [&](StringRef elided)", elidedAttrs);
    os << " {return attr.first.is(elided);})) {\n";
    os << tabs << "    continue;\n";
    os << tabs << "  }\n";
    os << tabs
       << formatv(
              "  if (failed(processDecoration({0}.getLoc(), {1}, attr))) {{\n",
              opVar, resultID);
    os << tabs << "    return failure();\n";
    os << tabs << "  }\n";
    os << tabs << "}\n";
  }
}

/// Generates code to serialize an SPV_Op `op` into `os`.
static void emitSerializationFunction(const Record *attrClass,
                                      const Record *record, const Operator &op,
                                      raw_ostream &os) {
  // If the record has 'autogenSerialization' set to 0, nothing to do
  if (!record->getValueAsBit("autogenSerialization")) {
    return;
  }
  StringRef opVar("op"), operands("operands"), elidedAttrs("elidedAttrs"),
      resultID("resultID");
  os << formatv(
      "template <> LogicalResult\nSerializer::processOp<{0}>({0} {1}) {{\n",
      op.getQualCppClassName(), opVar);
  os << formatv("  SmallVector<uint32_t, 4> {0};\n", operands);
  os << formatv("  SmallVector<StringRef, 2> {0};\n", elidedAttrs);

  // Serialize result information.
  if (op.getNumResults() == 1) {
    os << formatv("  uint32_t {0} = 0;\n", resultID);
    emitResultSerialization(op, record->getLoc(), "  ", opVar, operands,
                            resultID, os);
  }

  // Process arguments.
  emitOperandSerialization(op, record->getLoc(), "  ", opVar, operands,
                           elidedAttrs, os);

  if (record->isSubClassOf("SPV_ExtInstOp")) {
    os << formatv("  encodeExtensionInstruction({0}, \"{1}\", {2}, {3});\n",
                  opVar, record->getValueAsString("extendedInstSetName"),
                  record->getValueAsInt("extendedInstOpcode"), operands);
  } else {
    os << formatv("  encodeInstructionInto("
                  "functionBody, spirv::getOpcode<{0}>(), {1});\n",
                  op.getQualCppClassName(), operands);
  }

  // Process decorations.
  emitDecorationSerialization(op, "  ", opVar, elidedAttrs, resultID, os);

  os << "  return success();\n";
  os << "}\n\n";
}

/// Generates the prologue for the function that dispatches the serialization of
/// the operation `opVar` based on its opcode.
static void initDispatchSerializationFn(StringRef opVar, raw_ostream &os) {
  os << formatv(
      "LogicalResult Serializer::dispatchToAutogenSerialization(Operation "
      "*{0}) {{\n ",
      opVar);
}

/// Generates the body of the dispatch function. This function generates the
/// check that if satisfied, will call the serialization function generated for
/// the `op`.
static void emitSerializationDispatch(const Operator &op, StringRef tabs,
                                      StringRef opVar, raw_ostream &os) {
  os << tabs
     << formatv("if (isa<{0}>({1})) {{\n", op.getQualCppClassName(), opVar);
  os << tabs
     << formatv("  return processOp(cast<{0}>({1}));\n",
                op.getQualCppClassName(), opVar);
  os << tabs << "} else";
}

/// Generates the epilogue for the function that dispatches the serialization of
/// the operation.
static void finalizeDispatchSerializationFn(StringRef opVar, raw_ostream &os) {
  os << " {\n";
  os << formatv(
      "    return {0}->emitError(\"unhandled operation serialization\");\n",
      opVar);
  os << "  }\n";
  os << "  return success();\n";
  os << "}\n\n";
}

/// Generates code to deserialize the attribute of a SPV_Op into `os`. The
/// generated code reads the `words` of the serialized instruction at
/// position `wordIndex` and adds the deserialized attribute into `attrList`.
static void emitAttributeDeserialization(const Attribute &attr,
                                         ArrayRef<SMLoc> loc, StringRef tabs,
                                         StringRef attrList, StringRef attrName,
                                         StringRef words, StringRef wordIndex,
                                         raw_ostream &os) {
  if (attr.getAttrDefName() == "SPV_ScopeAttr" ||
      attr.getAttrDefName() == "SPV_MemorySemanticsAttr") {
    os << tabs
       << formatv("{0}.push_back(opBuilder.getNamedAttr(\"{1}\", "
                  "getConstantInt({2}[{3}++])));\n",
                  attrList, attrName, words, wordIndex);
  } else if (attr.getAttrDefName() == "I32ArrayAttr") {
    os << tabs << "SmallVector<Attribute, 4> attrListElems;\n";
    os << tabs << formatv("while ({0} < {1}.size()) {{\n", wordIndex, words);
    os << tabs
       << formatv(
              "  "
              "attrListElems.push_back(opBuilder.getI32IntegerAttr({0}[{1}++]))"
              ";\n",
              words, wordIndex);
    os << tabs << "}\n";
    os << tabs
       << formatv("{0}.push_back(opBuilder.getNamedAttr(\"{1}\", "
                  "opBuilder.getArrayAttr(attrListElems)));\n",
                  attrList, attrName);
  } else if (attr.isEnumAttr() || attr.getAttrDefName() == "I32Attr") {
    os << tabs
       << formatv("{0}.push_back(opBuilder.getNamedAttr(\"{1}\", "
                  "opBuilder.getI32IntegerAttr({2}[{3}++])));\n",
                  attrList, attrName, words, wordIndex);
  } else {
    PrintFatalError(
        loc, llvm::Twine(
                 "unhandled attribute type in deserialization generation : '") +
                 attr.getAttrDefName() + llvm::Twine("'"));
  }
}

/// Generates the code to deserialize the result of an SPV_Op `op` into
/// `os`. The generated code gets the type of the result specified at
/// `words`[`wordIndex`], the SSA ID for the result at position `wordIndex` + 1
/// and updates the `resultType` and `valueID` with the parsed type and SSA ID,
/// respectively.
static void emitResultDeserialization(const Operator &op, ArrayRef<SMLoc> loc,
                                      StringRef tabs, StringRef words,
                                      StringRef wordIndex,
                                      StringRef resultTypes, StringRef valueID,
                                      raw_ostream &os) {
  // Deserialize result information if it exists
  if (op.getNumResults() == 1) {
    os << tabs << "{\n";
    os << tabs << formatv("  if ({0} >= {1}.size()) {{\n", wordIndex, words);
    os << tabs
       << formatv(
              "    return emitError(unknownLoc, \"expected result type <id> "
              "while deserializing {0}\");\n",
              op.getQualCppClassName());
    os << tabs << "  }\n";
    os << tabs << formatv("  auto ty = getType({0}[{1}]);\n", words, wordIndex);
    os << tabs << "  if (!ty) {\n";
    os << tabs
       << formatv(
              "    return emitError(unknownLoc, \"unknown type result <id> : "
              "\") << {0}[{1}];\n",
              words, wordIndex);
    os << tabs << "  }\n";
    os << tabs << formatv("  {0}.push_back(ty);\n", resultTypes);
    os << tabs << formatv("  {0}++;\n", wordIndex);
    os << tabs << formatv("  if ({0} >= {1}.size()) {{\n", wordIndex, words);
    os << tabs
       << formatv(
              "    return emitError(unknownLoc, \"expected result <id> while "
              "deserializing {0}\");\n",
              op.getQualCppClassName());
    os << tabs << "  }\n";
    os << tabs << "}\n";
    os << tabs << formatv("{0} = {1}[{2}++];\n", valueID, words, wordIndex);
  } else if (op.getNumResults() != 0) {
    PrintFatalError(loc, "SPIR-V ops can have only zero or one result");
  }
}

/// Generates the code to deserialize the operands of an SPV_Op `op` into
/// `os`. The generated code reads the `words` of the binary instruction, from
/// position `wordIndex` to the end, and either gets the Value corresponding to
/// the ID encoded, or deserializes the attributes encoded. The parsed
/// operand(attribute) is added to the `operands` list or `attributes` list.
static void emitOperandDeserialization(const Operator &op, ArrayRef<SMLoc> loc,
                                       StringRef tabs, StringRef words,
                                       StringRef wordIndex, StringRef operands,
                                       StringRef attributes, raw_ostream &os) {
  // Process operands/attributes
  unsigned operandNum = 0;
  for (unsigned i = 0, e = op.getNumArgs(); i < e; ++i) {
    auto argument = op.getArg(i);
    if (auto valueArg = argument.dyn_cast<NamedTypeConstraint *>()) {
      if (valueArg->isVariadic()) {
        if (i != e - 1) {
          PrintFatalError(loc,
                          "SPIR-V ops can have Variadic<..> argument only if "
                          "it's the last argument");
        }
        os << tabs
           << formatv("for (; {0} < {1}.size(); ++{0})", wordIndex, words);
      } else {
        os << tabs << formatv("if ({0} < {1}.size())", wordIndex, words);
      }
      os << " {\n";
      os << tabs
         << formatv("  auto arg = getValue({0}[{1}]);\n", words, wordIndex);
      os << tabs << "  if (!arg) {\n";
      os << tabs
         << formatv(
                "    return emitError(unknownLoc, \"unknown result <id> : \") "
                "<< {0}[{1}];\n",
                words, wordIndex);
      os << tabs << "  }\n";
      os << tabs << formatv("  {0}.push_back(arg);\n", operands);
      if (!valueArg->isVariadic()) {
        os << tabs << formatv("  {0}++;\n", wordIndex);
      }
      operandNum++;
      os << tabs << "}\n";
    } else {
      os << tabs << formatv("if ({0} < {1}.size()) {{\n", wordIndex, words);
      auto attr = argument.get<NamedAttribute *>();
      auto newtabs = tabs.str() + "  ";
      emitAttributeDeserialization(
          (attr->attr.isOptional() ? attr->attr.getBaseAttr() : attr->attr),
          loc, newtabs, attributes, attr->name, words, wordIndex, os);
      os << "  }\n";
    }
  }

  os << tabs << formatv("if ({0} != {1}.size()) {{\n", wordIndex, words);
  os << tabs
     << formatv(
            "  return emitError(unknownLoc, \"found more operands than "
            "expected when deserializing {0}, only \") << {1} << \" of \" << "
            "{2}.size() << \" processed\";\n",
            op.getQualCppClassName(), wordIndex, words);
  os << tabs << "}\n\n";
}

/// Generates code to update the `attributes` vector with the attributes
/// obtained from parsing the decorations in the SPIR-V binary associated with
/// an <id> `valueID`
static void emitDecorationDeserialization(const Operator &op, StringRef tabs,
                                          StringRef valueID,
                                          StringRef attributes,
                                          raw_ostream &os) {
  // Import decorations parsed
  if (op.getNumResults() == 1) {
    os << tabs << formatv("if (decorations.count({0})) {{\n", valueID);
    os << tabs
       << formatv("  auto attrs = decorations[{0}].getAttrs();\n", valueID);
    os << tabs
       << formatv("  {0}.append(attrs.begin(), attrs.end());\n", attributes);
    os << tabs << "}\n";
  }
}

/// Generates code to deserialize an SPV_Op `op` into `os`.
static void emitDeserializationFunction(const Record *attrClass,
                                        const Record *record,
                                        const Operator &op, raw_ostream &os) {
  // If the record has 'autogenSerialization' set to 0, nothing to do
  if (!record->getValueAsBit("autogenSerialization")) {
    return;
  }
  StringRef resultTypes("resultTypes"), valueID("valueID"), words("words"),
      wordIndex("wordIndex"), opVar("op"), operands("operands"),
      attributes("attributes");
  os << formatv("template <> "
                "LogicalResult\nDeserializer::processOp<{0}>(ArrayRef<"
                "uint32_t> {1}) {{\n",
                op.getQualCppClassName(), words);
  os << formatv("  SmallVector<Type, 1> {0};\n", resultTypes);
  os << formatv("  size_t {0} = 0; (void){0};\n", wordIndex);
  os << formatv("  uint32_t {0} = 0; (void){0};\n", valueID);

  // Deserialize result information
  emitResultDeserialization(op, record->getLoc(), "  ", words, wordIndex,
                            resultTypes, valueID, os);

  os << formatv("  SmallVector<Value, 4> {0};\n", operands);
  os << formatv("  SmallVector<NamedAttribute, 4> {0};\n", attributes);
  // Operand deserialization
  emitOperandDeserialization(op, record->getLoc(), "  ", words, wordIndex,
                             operands, attributes, os);

  os << formatv(
      "  auto {1} = opBuilder.create<{0}>(unknownLoc, {2}, {3}, {4}); "
      "(void){1};\n",
      op.getQualCppClassName(), opVar, resultTypes, operands, attributes);
  if (op.getNumResults() == 1) {
    os << formatv("  valueMap[{0}] = {1}.getResult();\n\n", valueID, opVar);
  }

  // Decorations
  emitDecorationDeserialization(op, "  ", valueID, attributes, os);
  os << "  return success();\n";
  os << "}\n\n";
}

/// Generates the prologue for the function that dispatches the deserialization
/// based on the `opcode`.
static void initDispatchDeserializationFn(StringRef opcode, StringRef words,
                                          raw_ostream &os) {
  os << formatv(
      "LogicalResult "
      "Deserializer::dispatchToAutogenDeserialization(spirv::Opcode {0}, "
      "ArrayRef<uint32_t> {1}) {{\n",
      opcode, words);
  os << formatv("  switch ({0}) {{\n", opcode);
}

/// Generates the body of the dispatch function, by generating the case label
/// for an opcode and the call to the method to perform the deserialization.
static void emitDeserializationDispatch(const Operator &op, const Record *def,
                                        StringRef tabs, StringRef words,
                                        raw_ostream &os) {
  os << tabs
     << formatv("case spirv::Opcode::{0}:\n",
                def->getValueAsString("spirvOpName"));
  os << tabs
     << formatv("  return processOp<{0}>({1});\n", op.getQualCppClassName(),
                words);
}

/// Generates the epilogue for the function that dispatches the deserialization
/// of the operation.
static void finalizeDispatchDeserializationFn(StringRef opcode,
                                              raw_ostream &os) {
  os << "  default:\n";
  os << "    ;\n";
  os << "  }\n";
  StringRef opcodeVar("opcodeString");
  os << formatv("  auto {0} = spirv::stringifyOpcode({1});\n", opcodeVar,
                opcode);
  os << formatv("  if (!{0}.empty()) {{\n", opcodeVar);
  os << formatv("    return emitError(unknownLoc, \"unhandled deserialization "
                "of \") << {0};\n",
                opcodeVar);
  os << "  } else {\n";
  os << formatv("   return emitError(unknownLoc, \"unhandled opcode \") << "
                "static_cast<uint32_t>({0});\n",
                opcode);
  os << "  }\n";
  os << "}\n";
}

static void initExtendedSetDeserializationDispatch(StringRef extensionSetName,
                                                   StringRef instructionID,
                                                   StringRef words,
                                                   raw_ostream &os) {
  os << formatv("LogicalResult "
                "Deserializer::dispatchToExtensionSetAutogenDeserialization("
                "StringRef {0}, uint32_t {1}, ArrayRef<uint32_t> {2}) {{\n",
                extensionSetName, instructionID, words);
}

static void
emitExtendedSetDeserializationDispatch(const RecordKeeper &recordKeeper,
                                       raw_ostream &os) {
  StringRef extensionSetName("extensionSetName"),
      instructionID("instructionID"), words("words");

  // First iterate over all ops derived from SPV_ExtensionSetOps to get all
  // extensionSets.

  // For each of the extensions a separate raw_string_ostream is used to
  // generate code into. These are then concatenated at the end. Since
  // raw_string_ostream needs a string&, use a vector to store all the string
  // that are captured by reference within raw_string_ostream.
  StringMap<raw_string_ostream> extensionSets;
  SmallVector<std::string, 1> extensionSetNames;

  initExtendedSetDeserializationDispatch(extensionSetName, instructionID, words,
                                         os);
  auto defs = recordKeeper.getAllDerivedDefinitions("SPV_ExtInstOp");
  for (const auto *def : defs) {
    if (!def->getValueAsBit("autogenSerialization")) {
      continue;
    }
    Operator op(def);
    auto setName = def->getValueAsString("extendedInstSetName");
    if (!extensionSets.count(setName)) {
      extensionSetNames.push_back("");
      extensionSets.try_emplace(setName, extensionSetNames.back());
      auto &setos = extensionSets.find(setName)->second;
      setos << formatv("  if ({0} == \"{1}\") {{\n", extensionSetName, setName);
      setos << formatv("    switch ({0}) {{\n", instructionID);
    }
    auto &setos = extensionSets.find(setName)->second;
    setos << formatv("    case {0}:\n",
                     def->getValueAsInt("extendedInstOpcode"));
    setos << formatv("      return processOp<{0}>({1});\n",
                     op.getQualCppClassName(), words);
  }

  // Append the dispatch code for all the extended sets.
  for (auto &extensionSet : extensionSets) {
    os << extensionSet.second.str();
    os << "    default:\n";
    os << formatv(
        "      return emitError(unknownLoc, \"unhandled deserializations of "
        "\") << {0} << \" from extension set \" << {1};\n",
        instructionID, extensionSetName);
    os << "    }\n";
    os << "  }\n";
  }

  os << formatv("  return emitError(unknownLoc, \"unhandled deserialization of "
                "extended instruction set {0}\");\n",
                extensionSetName);
  os << "}\n";
}

/// Emits all the autogenerated serialization/deserializations functions for the
/// SPV_Ops.
static bool emitSerializationFns(const RecordKeeper &recordKeeper,
                                 raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Serialization Utilities/Functions", os);

  std::string dSerFnString, dDesFnString, serFnString, deserFnString,
      utilsString;
  raw_string_ostream dSerFn(dSerFnString), dDesFn(dDesFnString),
      serFn(serFnString), deserFn(deserFnString), utils(utilsString);
  auto attrClass = recordKeeper.getClass("Attr");

  // Emit the serialization and deserialization functions simultaneously.
  declareOpcodeFn(utils);
  StringRef opVar("op");
  StringRef opcode("opcode"), words("words");

  // Handle the SPIR-V ops.
  initDispatchSerializationFn(opVar, dSerFn);
  initDispatchDeserializationFn(opcode, words, dDesFn);
  auto defs = recordKeeper.getAllDerivedDefinitions("SPV_Op");
  for (const auto *def : defs) {
    Operator op(def);
    emitSerializationFunction(attrClass, def, op, serFn);
    emitDeserializationFunction(attrClass, def, op, deserFn);
    if (def->getValueAsBit("hasOpcode") || def->isSubClassOf("SPV_ExtInstOp")) {
      emitSerializationDispatch(op, "  ", opVar, dSerFn);
    }
    if (def->getValueAsBit("hasOpcode")) {
      emitGetOpcodeFunction(def, op, utils);
      emitDeserializationDispatch(op, def, "  ", words, dDesFn);
    }
  }
  finalizeDispatchSerializationFn(opVar, dSerFn);
  finalizeDispatchDeserializationFn(opcode, dDesFn);

  emitExtendedSetDeserializationDispatch(recordKeeper, dDesFn);

  os << "#ifdef GET_SPIRV_SERIALIZATION_UTILS\n";
  os << utils.str();
  os << "#endif // GET_SPIRV_SERIALIZATION_UTILS\n\n";

  os << "#ifdef GET_SERIALIZATION_FNS\n\n";
  os << serFn.str();
  os << dSerFn.str();
  os << "#endif // GET_SERIALIZATION_FNS\n\n";

  os << "#ifdef GET_DESERIALIZATION_FNS\n\n";
  os << deserFn.str();
  os << dDesFn.str();
  os << "#endif // GET_DESERIALIZATION_FNS\n\n";

  return false;
}

//===----------------------------------------------------------------------===//
// Serialization Hook Registration
//===----------------------------------------------------------------------===//

static mlir::GenRegistration genSerialization(
    "gen-spirv-serialization",
    "Generate SPIR-V (de)serialization utilities and functions",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitSerializationFns(records, os);
    });

//===----------------------------------------------------------------------===//
// Op Utils AutoGen
//===----------------------------------------------------------------------===//

static void emitEnumGetAttrNameFnDecl(raw_ostream &os) {
  os << formatv("template <typename EnumClass> inline constexpr StringRef "
                "attributeName();\n");
}

static void emitEnumGetSymbolizeFnDecl(raw_ostream &os) {
  os << "template <typename EnumClass> using SymbolizeFnTy = "
        "llvm::Optional<EnumClass> (*)(StringRef);\n";
  os << "template <typename EnumClass> inline constexpr "
        "SymbolizeFnTy<EnumClass> symbolizeEnum();\n";
}

static void emitEnumGetAttrNameFnDefn(const EnumAttr &enumAttr,
                                      raw_ostream &os) {
  auto enumName = enumAttr.getEnumClassName();
  os << formatv("template <> inline StringRef attributeName<{0}>() {{\n",
                enumName);
  os << "  "
     << formatv("static constexpr const char attrName[] = \"{0}\";\n",
                mlir::convertToSnakeCase(enumName));
  os << "  return attrName;\n";
  os << "}\n";
}

static void emitEnumGetSymbolizeFnDefn(const EnumAttr &enumAttr,
                                       raw_ostream &os) {
  auto enumName = enumAttr.getEnumClassName();
  auto strToSymFnName = enumAttr.getStringToSymbolFnName();
  os << formatv(
      "template <> inline SymbolizeFnTy<{0}> symbolizeEnum<{0}>() {{\n",
      enumName);
  os << "  return " << strToSymFnName << ";\n";
  os << "}\n";
}

static bool emitOpUtils(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Op Utilities", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  os << "#ifndef SPIRV_OP_UTILS_H_\n";
  os << "#define SPIRV_OP_UTILS_H_\n";
  emitEnumGetAttrNameFnDecl(os);
  emitEnumGetSymbolizeFnDecl(os);
  for (const auto *def : defs) {
    EnumAttr enumAttr(*def);
    emitEnumGetAttrNameFnDefn(enumAttr, os);
    emitEnumGetSymbolizeFnDefn(enumAttr, os);
  }
  os << "#endif // SPIRV_OP_UTILS_H\n";
  return false;
}

//===----------------------------------------------------------------------===//
// Op Utils Hook Registration
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genOpUtils("gen-spirv-op-utils",
               "Generate SPIR-V operation utility definitions",
               [](const RecordKeeper &records, raw_ostream &os) {
                 return emitOpUtils(records, os);
               });

//===----------------------------------------------------------------------===//
// SPIR-V Availability Impl AutoGen
//===----------------------------------------------------------------------===//

// Returns the availability spec of the given `def`.
std::vector<Availability> getAvailabilities(const Record &def) {
  std::vector<Availability> availabilities;
  if (auto *availListInit = def.getValueAsListInit("availability")) {
    availabilities.reserve(availListInit->size());
    for (auto *availInit : *availListInit)
      availabilities.emplace_back(
          llvm::cast<llvm::DefInit>(availInit)->getDef());
  }
  return availabilities;
}

static void emitAvailabilityImpl(const Operator &srcOp, raw_ostream &os) {
  mlir::tblgen::FmtContext fctx;
  fctx.addSubst("overall", "overall");

  std::vector<Availability> opAvailabilities =
      getAvailabilities(srcOp.getDef());

  // First collect all availablity classes this op should implement.
  // All availablity instances keep information for the generated interface and
  // the instance's specific requirement. Here we remember a random instance so
  // we can get the information regarding the generated interface.
  llvm::StringMap<Availability> availClasses;
  for (const Availability &avail : opAvailabilities)
    availClasses.try_emplace(avail.getClass(), avail);

  // Then generate implementation for each availability class.
  for (const auto &availClass : availClasses) {
    StringRef availClassName = availClass.getKey();
    Availability avail = availClass.getValue();

    // Generate the implementation method signature.
    os << formatv("{0} {1}::{2}() {{\n", avail.getQueryFnRetType(),
                  srcOp.getCppClassName(), avail.getQueryFnName());

    // Create the variable for the final requirement and initialize it.
    os << formatv("  {0} overall = {1};\n", avail.getQueryFnRetType(),
                  avail.getMergeInitializer());

    // Update with the op's specific availability spec.
    for (const Availability &avail : opAvailabilities)
      if (avail.getClass() == availClassName) {
        os << "  "
           << tgfmt(avail.getMergeActionCode(),
                    &fctx.addSubst("instance", avail.getMergeInstance()))
           << ";\n";
      }
    os << "  return overall;\n";
    os << "}\n";
  }
}

static bool emitAvailabilityImpl(const RecordKeeper &recordKeeper,
                                 raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Op Availability Implementations", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("SPV_Op");
  for (const auto *def : defs) {
    Operator op(def);
    emitAvailabilityImpl(op, os);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Op Availability Implementation Hook Registration
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genOpAvailabilityImpl("gen-spirv-avail-impls",
                          "Generate SPIR-V operation utility definitions",
                          [](const RecordKeeper &records, raw_ostream &os) {
                            return emitAvailabilityImpl(records, os);
                          });
