//===- SPIRVSerializationGen.cpp - SPIR-V serialization utility generator -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SPIRVSerializationGen generates common utility functions for SPIR-V
// serialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
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
using mlir::tblgen::EnumAttrCase;
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

  // Returns the C++ statements for preparing availability instance.
  StringRef getMergeInstancePreparation() const;

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

StringRef Availability::getMergeInstancePreparation() const {
  return def->getValueAsString("instancePreparation");
}

StringRef Availability::getMergeInstance() const {
  return def->getValueAsString("instance");
}

// Returns the availability spec of the given `def`.
std::vector<Availability> getAvailabilities(const Record &def) {
  std::vector<Availability> availabilities;

  if (def.getValue("availability")) {
    std::vector<Record *> availDefs = def.getValueAsListOfDefs("availability");
    availabilities.reserve(availDefs.size());
    for (const Record *avail : availDefs)
      availabilities.emplace_back(avail);
  }

  return availabilities;
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
  std::string interfaceTraitsName =
      std::string(formatv("{0}Traits", interfaceName));

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
// Enum Availability Query AutoGen
//===----------------------------------------------------------------------===//

static void emitAvailabilityQueryForIntEnum(const Record &enumDef,
                                            raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  std::vector<EnumAttrCase> enumerants = enumAttr.getAllCases();

  // Mapping from availability class name to (enumerant, availability
  // specification) pairs.
  llvm::StringMap<llvm::SmallVector<std::pair<EnumAttrCase, Availability>, 1>>
      classCaseMap;

  // Place all availability specifications to their corresponding
  // availability classes.
  for (const EnumAttrCase &enumerant : enumerants)
    for (const Availability &avail : getAvailabilities(enumerant.getDef()))
      classCaseMap[avail.getClass()].push_back({enumerant, avail});

  for (const auto &classCasePair : classCaseMap) {
    Availability avail = classCasePair.getValue().front().second;

    os << formatv("llvm::Optional<{0}> {1}({2} value) {{\n",
                  avail.getMergeInstanceType(), avail.getQueryFnName(),
                  enumName);

    os << "  switch (value) {\n";
    for (const auto &caseSpecPair : classCasePair.getValue()) {
      EnumAttrCase enumerant = caseSpecPair.first;
      Availability avail = caseSpecPair.second;
      os << formatv("  case {0}::{1}: { {2} return {3}({4}); }\n", enumName,
                    enumerant.getSymbol(), avail.getMergeInstancePreparation(),
                    avail.getMergeInstanceType(), avail.getMergeInstance());
    }
    // Only emit default if uncovered cases.
    if (classCasePair.getValue().size() < enumAttr.getAllCases().size())
      os << "  default: break;\n";
    os << "  }\n"
       << "  return llvm::None;\n"
       << "}\n";
  }
}

static void emitAvailabilityQueryForBitEnum(const Record &enumDef,
                                            raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  std::string underlyingType = std::string(enumAttr.getUnderlyingType());
  std::vector<EnumAttrCase> enumerants = enumAttr.getAllCases();

  // Mapping from availability class name to (enumerant, availability
  // specification) pairs.
  llvm::StringMap<llvm::SmallVector<std::pair<EnumAttrCase, Availability>, 1>>
      classCaseMap;

  // Place all availability specifications to their corresponding
  // availability classes.
  for (const EnumAttrCase &enumerant : enumerants)
    for (const Availability &avail : getAvailabilities(enumerant.getDef()))
      classCaseMap[avail.getClass()].push_back({enumerant, avail});

  for (const auto &classCasePair : classCaseMap) {
    Availability avail = classCasePair.getValue().front().second;

    os << formatv("llvm::Optional<{0}> {1}({2} value) {{\n",
                  avail.getMergeInstanceType(), avail.getQueryFnName(),
                  enumName);

    os << formatv(
        "  assert(::llvm::countPopulation(static_cast<{0}>(value)) <= 1"
        " && \"cannot have more than one bit set\");\n",
        underlyingType);

    os << "  switch (value) {\n";
    for (const auto &caseSpecPair : classCasePair.getValue()) {
      EnumAttrCase enumerant = caseSpecPair.first;
      Availability avail = caseSpecPair.second;
      os << formatv("  case {0}::{1}: { {2} return {3}({4}); }\n", enumName,
                    enumerant.getSymbol(), avail.getMergeInstancePreparation(),
                    avail.getMergeInstanceType(), avail.getMergeInstance());
    }
    os << "  default: break;\n";
    os << "  }\n"
       << "  return llvm::None;\n"
       << "}\n";
  }
}

static void emitEnumDecl(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  StringRef cppNamespace = enumAttr.getCppNamespace();
  auto enumerants = enumAttr.getAllCases();

  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  llvm::StringSet<> handledClasses;

  // Place all availability specifications to their corresponding
  // availability classes.
  for (const EnumAttrCase &enumerant : enumerants)
    for (const Availability &avail : getAvailabilities(enumerant.getDef())) {
      StringRef className = avail.getClass();
      if (handledClasses.count(className))
        continue;
      os << formatv("llvm::Optional<{0}> {1}({2} value);\n",
                    avail.getMergeInstanceType(), avail.getQueryFnName(),
                    enumName);
      handledClasses.insert(className);
    }

  for (auto ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
}

static bool emitEnumDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Enum Availability Declarations", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  for (const auto *def : defs)
    emitEnumDecl(*def, os);

  return false;
}

static void emitEnumDef(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef cppNamespace = enumAttr.getCppNamespace();

  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  if (enumAttr.isBitEnum()) {
    emitAvailabilityQueryForBitEnum(enumDef, os);
  } else {
    emitAvailabilityQueryForIntEnum(enumDef, os);
  }

  for (auto ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
  os << "\n";
}

static bool emitEnumDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Enum Availability Definitions", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  for (const auto *def : defs)
    emitEnumDef(*def, os);

  return false;
}

//===----------------------------------------------------------------------===//
// Enum Availability Query Hook Registration
//===----------------------------------------------------------------------===//

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDecls("gen-spirv-enum-avail-decls",
                 "Generate SPIR-V enum availability declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return emitEnumDecls(records, os);
                 });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDefs("gen-spirv-enum-avail-defs",
                "Generate SPIR-V enum availability definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitEnumDefs(records, os);
                });

//===----------------------------------------------------------------------===//
// Serialization AutoGen
//===----------------------------------------------------------------------===//

/// Generates code to serialize attributes of a SPV_Op `op` into `os`. The
/// generates code extracts the attribute with name `attrName` from
/// `operandList` of `op`.
static void emitAttributeSerialization(const Attribute &attr,
                                       ArrayRef<SMLoc> loc, StringRef tabs,
                                       StringRef opVar, StringRef operandList,
                                       StringRef attrName, raw_ostream &os) {
  os << tabs
     << formatv("if (auto attr = {0}.getAttr(\"{1}\")) {{\n", opVar, attrName);
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
  } else if (attr.isEnumAttr() || attr.getAttrDefName() == "TypeAttr") {
    os << tabs
       << formatv("  {0}.push_back(static_cast<uint32_t>("
                  "getTypeID(attr.cast<TypeAttr>().getValue())));\n",
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
static void emitArgumentSerialization(const Operator &op, ArrayRef<SMLoc> loc,
                                      StringRef tabs, StringRef opVar,
                                      StringRef operands, StringRef elidedAttrs,
                                      raw_ostream &os) {
  using mlir::tblgen::Argument;

  // SPIR-V ops can mix operands and attributes in the definition. These
  // operands and attributes are serialized in the exact order of the definition
  // to match SPIR-V binary format requirements. It can cause excessive
  // generated code bloat because we are emitting code to handle each
  // operand/attribute separately. So here we probe first to check whether all
  // the operands are ahead of attributes. Then we can serialize all operands
  // together.

  // Whether all operands are ahead of all attributes in the op's spec.
  bool areOperandsAheadOfAttrs = true;
  // Find the first attribute.
  const Argument *it = llvm::find_if(op.getArgs(), [](const Argument &arg) {
    return arg.is<NamedAttribute *>();
  });
  // Check whether all following arguments are attributes.
  for (const Argument *ie = op.arg_end(); it != ie; ++it) {
    if (!it->is<NamedAttribute *>()) {
      areOperandsAheadOfAttrs = false;
      break;
    }
  }

  // Serialize all operands together.
  if (areOperandsAheadOfAttrs) {
    if (op.getNumOperands() != 0) {
      os << tabs
         << formatv(
                "for (Value operand : {0}.getOperation()->getOperands()) {{\n",
                opVar);
      os << tabs << "  auto id = getValueID(operand);\n";
      os << tabs << "  assert(id && \"use before def!\");\n";
      os << tabs << formatv("  {0}.push_back(id);\n", operands);
      os << tabs << "}\n";
    }
    for (const NamedAttribute &attr : op.getAttributes()) {
      emitAttributeSerialization(
          (attr.attr.isOptional() ? attr.attr.getBaseAttr() : attr.attr), loc,
          tabs, opVar, operands, attr.name, os);
      os << tabs
         << formatv("{0}.push_back(\"{1}\");\n", elidedAttrs, attr.name);
    }
    return;
  }

  // Serialize operands separately.
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
      NamedAttribute *attr = argument.get<NamedAttribute *>();
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
    os << " {return attr.first == elided;})) {\n";
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
  emitArgumentSerialization(op, record->getLoc(), "  ", opVar, operands,
                            elidedAttrs, os);

  if (record->isSubClassOf("SPV_ExtInstOp")) {
    os << formatv("  encodeExtensionInstruction({0}, \"{1}\", {2}, {3});\n",
                  opVar, record->getValueAsString("extendedInstSetName"),
                  record->getValueAsInt("extendedInstOpcode"), operands);
  } else {
    // Emit debug info.
    os << formatv("  emitDebugLine(functionBody, {0}.getLoc());\n", opVar);
    os << formatv("  encodeInstructionInto("
                  "functionBody, spirv::Opcode::{1}, {2});\n",
                  op.getQualCppClassName(),
                  record->getValueAsString("spirvOpName"), operands);
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
      "*{0}) {{\n",
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
  os << tabs << "}\n";
}

/// Generates the epilogue for the function that dispatches the serialization of
/// the operation.
static void finalizeDispatchSerializationFn(StringRef opVar, raw_ostream &os) {
  os << formatv(
      "  return {0}->emitError(\"unhandled operation serialization\");\n",
      opVar);
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
  } else if (attr.isEnumAttr() || attr.getAttrDefName() == "TypeAttr") {
    os << tabs
       << formatv("{0}.push_back(opBuilder.getNamedAttr(\"{1}\", "
                  "TypeAttr::get(getType({2}[{3}++]))));\n",
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
      if (valueArg->isVariableLength()) {
        if (i != e - 1) {
          PrintFatalError(loc, "SPIR-V ops can have Variadic<..> or "
                               "Optional<...> arguments only if "
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
      if (!valueArg->isVariableLength()) {
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

  os << formatv("  Location loc = createFileLineColLoc(opBuilder);\n");
  os << formatv("  auto {1} = opBuilder.create<{0}>(loc, {2}, {3}, {4}); "
                "(void){1};\n",
                op.getQualCppClassName(), opVar, resultTypes, operands,
                attributes);
  if (op.getNumResults() == 1) {
    os << formatv("  valueMap[{0}] = {1}.getResult();\n\n", valueID, opVar);
  }

  // According to SPIR-V spec:
  // This location information applies to the instructions physically following
  // this instruction, up to the first occurrence of any of the following: the
  // next end of block.
  os << formatv("  if ({0}.hasTrait<OpTrait::IsTerminator>())\n", opVar);
  os << formatv("    clearDebugLine();\n");

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
      serFn(serFnString), deserFn(deserFnString);
  Record *attrClass = recordKeeper.getClass("Attr");

  // Emit the serialization and deserialization functions simultaneously.
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
      emitDeserializationDispatch(op, def, "  ", words, dDesFn);
    }
  }
  finalizeDispatchSerializationFn(opVar, dSerFn);
  finalizeDispatchDeserializationFn(opcode, dDesFn);

  emitExtendedSetDeserializationDispatch(recordKeeper, dDesFn);

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

static void emitEnumGetAttrNameFnDefn(const EnumAttr &enumAttr,
                                      raw_ostream &os) {
  auto enumName = enumAttr.getEnumClassName();
  os << formatv("template <> inline StringRef attributeName<{0}>() {{\n",
                enumName);
  os << "  "
     << formatv("static constexpr const char attrName[] = \"{0}\";\n",
                llvm::convertToSnakeFromCamelCase(enumName));
  os << "  return attrName;\n";
  os << "}\n";
}

static bool emitOpUtils(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Op Utilities", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  os << "#ifndef SPIRV_OP_UTILS_H_\n";
  os << "#define SPIRV_OP_UTILS_H_\n";
  emitEnumGetAttrNameFnDecl(os);
  for (const auto *def : defs) {
    EnumAttr enumAttr(*def);
    emitEnumGetAttrNameFnDefn(enumAttr, os);
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

static void emitAvailabilityImpl(const Operator &srcOp, raw_ostream &os) {
  mlir::tblgen::FmtContext fctx;
  fctx.addSubst("overall", "tblgen_overall");

  std::vector<Availability> opAvailabilities =
      getAvailabilities(srcOp.getDef());

  // First collect all availability classes this op should implement.
  // All availability instances keep information for the generated interface and
  // the instance's specific requirement. Here we remember a random instance so
  // we can get the information regarding the generated interface.
  llvm::StringMap<Availability> availClasses;
  for (const Availability &avail : opAvailabilities)
    availClasses.try_emplace(avail.getClass(), avail);
  for (const NamedAttribute &namedAttr : srcOp.getAttributes()) {
    const auto *enumAttr = llvm::dyn_cast<EnumAttr>(&namedAttr.attr);
    if (!enumAttr)
      continue;

    for (const EnumAttrCase &enumerant : enumAttr->getAllCases())
      for (const Availability &caseAvail :
           getAvailabilities(enumerant.getDef()))
        availClasses.try_emplace(caseAvail.getClass(), caseAvail);
  }

  // Then generate implementation for each availability class.
  for (const auto &availClass : availClasses) {
    StringRef availClassName = availClass.getKey();
    Availability avail = availClass.getValue();

    // Generate the implementation method signature.
    os << formatv("{0} {1}::{2}() {{\n", avail.getQueryFnRetType(),
                  srcOp.getCppClassName(), avail.getQueryFnName());

    // Create the variable for the final requirement and initialize it.
    os << formatv("  {0} tblgen_overall = {1};\n", avail.getQueryFnRetType(),
                  avail.getMergeInitializer());

    // Update with the op's specific availability spec.
    for (const Availability &avail : opAvailabilities)
      if (avail.getClass() == availClassName &&
          (!avail.getMergeInstancePreparation().empty() ||
           !avail.getMergeActionCode().empty())) {
        os << "  {\n    "
           // Prepare this instance.
           << avail.getMergeInstancePreparation()
           << "\n    "
           // Merge this instance.
           << std::string(
                  tgfmt(avail.getMergeActionCode(),
                        &fctx.addSubst("instance", avail.getMergeInstance())))
           << ";\n  }\n";
      }

    // Update with enum attributes' specific availability spec.
    for (const NamedAttribute &namedAttr : srcOp.getAttributes()) {
      const auto *enumAttr = llvm::dyn_cast<EnumAttr>(&namedAttr.attr);
      if (!enumAttr)
        continue;

      // (enumerant, availability specification) pairs for this availability
      // class.
      SmallVector<std::pair<EnumAttrCase, Availability>, 1> caseSpecs;

      // Collect all cases' availability specs.
      for (const EnumAttrCase &enumerant : enumAttr->getAllCases())
        for (const Availability &caseAvail :
             getAvailabilities(enumerant.getDef()))
          if (availClassName == caseAvail.getClass())
            caseSpecs.push_back({enumerant, caseAvail});

      // If this attribute kind does not have any availability spec from any of
      // its cases, no more work to do.
      if (caseSpecs.empty())
        continue;

      if (enumAttr->isBitEnum()) {
        // For BitEnumAttr, we need to iterate over each bit to query its
        // availability spec.
        os << formatv("  for (unsigned i = 0; "
                      "i < std::numeric_limits<{0}>::digits; ++i) {{\n",
                      enumAttr->getUnderlyingType());
        os << formatv("    {0}::{1} tblgen_attrVal = this->{2}() & "
                      "static_cast<{0}::{1}>(1 << i);\n",
                      enumAttr->getCppNamespace(), enumAttr->getEnumClassName(),
                      namedAttr.name);
        os << formatv(
            "    if (static_cast<{0}>(tblgen_attrVal) == 0) continue;\n",
            enumAttr->getUnderlyingType());
      } else {
        // For IntEnumAttr, we just need to query the value as a whole.
        os << "  {\n";
        os << formatv("    auto tblgen_attrVal = this->{0}();\n",
                      namedAttr.name);
      }
      os << formatv("    auto tblgen_instance = {0}::{1}(tblgen_attrVal);\n",
                    enumAttr->getCppNamespace(), avail.getQueryFnName());
      os << "    if (tblgen_instance) "
         // TODO(antiagainst): use `avail.getMergeCode()` here once ODS supports
         // dialect-specific contents so that we can use not implementing the
         // availability interface as indication of no requirements.
         << std::string(tgfmt(caseSpecs.front().second.getMergeActionCode(),
                              &fctx.addSubst("instance", "*tblgen_instance")))
         << ";\n";
      os << "  }\n";
    }

    os << "  return tblgen_overall;\n";
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

//===----------------------------------------------------------------------===//
// SPIR-V Capability Implication AutoGen
//===----------------------------------------------------------------------===//

static bool emitCapabilityImplication(const RecordKeeper &recordKeeper,
                                      raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Capability Implication", os);

  EnumAttr enumAttr(recordKeeper.getDef("SPV_CapabilityAttr"));

  os << "ArrayRef<Capability> "
        "spirv::getDirectImpliedCapabilities(Capability cap) {\n"
     << "  switch (cap) {\n"
     << "  default: return {};\n";
  for (const EnumAttrCase &enumerant : enumAttr.getAllCases()) {
    const Record &def = enumerant.getDef();
    if (!def.getValue("implies"))
      continue;

    std::vector<Record *> impliedCapsDefs = def.getValueAsListOfDefs("implies");
    os << "  case Capability::" << enumerant.getSymbol()
       << ": {static const Capability implies[" << impliedCapsDefs.size()
       << "] = {";
    llvm::interleaveComma(impliedCapsDefs, os, [&](const Record *capDef) {
      os << "Capability::" << EnumAttrCase(capDef).getSymbol();
    });
    os << "}; return ArrayRef<Capability>(implies, " << impliedCapsDefs.size()
       << "); }\n";
  }
  os << "  }\n";
  os << "}\n";

  return false;
}

//===----------------------------------------------------------------------===//
// SPIR-V Capability Implication Hook Registration
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genCapabilityImplication("gen-spirv-capability-implication",
                             "Generate utility function to return implied "
                             "capabilities for a given capability",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               return emitCapabilityImplication(records, os);
                             });
