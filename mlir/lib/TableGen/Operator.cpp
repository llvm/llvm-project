//===- Operator.cpp - Operator class --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Operator wrapper to simplify using TableGen Record defining a MLIR Op.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Predicate.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#define DEBUG_TYPE "mlir-tblgen-operator"

using namespace mlir;

using llvm::DagInit;
using llvm::DefInit;
using llvm::Record;

tblgen::Operator::Operator(const llvm::Record &def)
    : dialect(def.getValueAsDef("opDialect")), def(def) {
  // The first `_` in the op's TableGen def name is treated as separating the
  // dialect prefix and the op class name. The dialect prefix will be ignored if
  // not empty. Otherwise, if def name starts with a `_`, the `_` is considered
  // as part of the class name.
  StringRef prefix;
  std::tie(prefix, cppClassName) = def.getName().split('_');
  if (prefix.empty()) {
    // Class name with a leading underscore and without dialect prefix
    cppClassName = def.getName();
  } else if (cppClassName.empty()) {
    // Class name without dialect prefix
    cppClassName = prefix;
  }

  populateOpStructure();
}

std::string tblgen::Operator::getOperationName() const {
  auto prefix = dialect.getName();
  auto opName = def.getValueAsString("opName");
  if (prefix.empty())
    return std::string(opName);
  return std::string(llvm::formatv("{0}.{1}", prefix, opName));
}

std::string tblgen::Operator::getAdaptorName() const {
  return std::string(llvm::formatv("{0}OperandAdaptor", getCppClassName()));
}

StringRef tblgen::Operator::getDialectName() const { return dialect.getName(); }

StringRef tblgen::Operator::getCppClassName() const { return cppClassName; }

std::string tblgen::Operator::getQualCppClassName() const {
  auto prefix = dialect.getCppNamespace();
  if (prefix.empty())
    return std::string(cppClassName);
  return std::string(llvm::formatv("{0}::{1}", prefix, cppClassName));
}

int tblgen::Operator::getNumResults() const {
  DagInit *results = def.getValueAsDag("results");
  return results->getNumArgs();
}

StringRef tblgen::Operator::getExtraClassDeclaration() const {
  constexpr auto attr = "extraClassDeclaration";
  if (def.isValueUnset(attr))
    return {};
  return def.getValueAsString(attr);
}

const llvm::Record &tblgen::Operator::getDef() const { return def; }

bool tblgen::Operator::skipDefaultBuilders() const {
  return def.getValueAsBit("skipDefaultBuilders");
}

auto tblgen::Operator::result_begin() -> value_iterator {
  return results.begin();
}

auto tblgen::Operator::result_end() -> value_iterator { return results.end(); }

auto tblgen::Operator::getResults() -> value_range {
  return {result_begin(), result_end()};
}

tblgen::TypeConstraint
tblgen::Operator::getResultTypeConstraint(int index) const {
  DagInit *results = def.getValueAsDag("results");
  return TypeConstraint(cast<DefInit>(results->getArg(index)));
}

StringRef tblgen::Operator::getResultName(int index) const {
  DagInit *results = def.getValueAsDag("results");
  return results->getArgNameStr(index);
}

auto tblgen::Operator::getResultDecorators(int index) const
    -> var_decorator_range {
  Record *result =
      cast<DefInit>(def.getValueAsDag("results")->getArg(index))->getDef();
  if (!result->isSubClassOf("OpVariable"))
    return var_decorator_range(nullptr, nullptr);
  return *result->getValueAsListInit("decorators");
}

unsigned tblgen::Operator::getNumVariableLengthResults() const {
  return llvm::count_if(results, [](const NamedTypeConstraint &c) {
    return c.constraint.isVariableLength();
  });
}

unsigned tblgen::Operator::getNumVariableLengthOperands() const {
  return llvm::count_if(operands, [](const NamedTypeConstraint &c) {
    return c.constraint.isVariableLength();
  });
}

tblgen::Operator::arg_iterator tblgen::Operator::arg_begin() const {
  return arguments.begin();
}

tblgen::Operator::arg_iterator tblgen::Operator::arg_end() const {
  return arguments.end();
}

tblgen::Operator::arg_range tblgen::Operator::getArgs() const {
  return {arg_begin(), arg_end()};
}

StringRef tblgen::Operator::getArgName(int index) const {
  DagInit *argumentValues = def.getValueAsDag("arguments");
  return argumentValues->getArgName(index)->getValue();
}

auto tblgen::Operator::getArgDecorators(int index) const
    -> var_decorator_range {
  Record *arg =
      cast<DefInit>(def.getValueAsDag("arguments")->getArg(index))->getDef();
  if (!arg->isSubClassOf("OpVariable"))
    return var_decorator_range(nullptr, nullptr);
  return *arg->getValueAsListInit("decorators");
}

const tblgen::OpTrait *tblgen::Operator::getTrait(StringRef trait) const {
  for (const auto &t : traits) {
    if (const auto *opTrait = dyn_cast<tblgen::NativeOpTrait>(&t)) {
      if (opTrait->getTrait() == trait)
        return opTrait;
    } else if (const auto *opTrait = dyn_cast<tblgen::InternalOpTrait>(&t)) {
      if (opTrait->getTrait() == trait)
        return opTrait;
    } else if (const auto *opTrait = dyn_cast<tblgen::InterfaceOpTrait>(&t)) {
      if (opTrait->getTrait() == trait)
        return opTrait;
    }
  }
  return nullptr;
}

auto tblgen::Operator::region_begin() const -> const_region_iterator {
  return regions.begin();
}
auto tblgen::Operator::region_end() const -> const_region_iterator {
  return regions.end();
}
auto tblgen::Operator::getRegions() const
    -> llvm::iterator_range<const_region_iterator> {
  return {region_begin(), region_end()};
}

unsigned tblgen::Operator::getNumRegions() const { return regions.size(); }

const tblgen::NamedRegion &tblgen::Operator::getRegion(unsigned index) const {
  return regions[index];
}

unsigned tblgen::Operator::getNumVariadicRegions() const {
  return llvm::count_if(regions,
                        [](const NamedRegion &c) { return c.isVariadic(); });
}

auto tblgen::Operator::successor_begin() const -> const_successor_iterator {
  return successors.begin();
}
auto tblgen::Operator::successor_end() const -> const_successor_iterator {
  return successors.end();
}
auto tblgen::Operator::getSuccessors() const
    -> llvm::iterator_range<const_successor_iterator> {
  return {successor_begin(), successor_end()};
}

unsigned tblgen::Operator::getNumSuccessors() const {
  return successors.size();
}

const tblgen::NamedSuccessor &
tblgen::Operator::getSuccessor(unsigned index) const {
  return successors[index];
}

unsigned tblgen::Operator::getNumVariadicSuccessors() const {
  return llvm::count_if(successors,
                        [](const NamedSuccessor &c) { return c.isVariadic(); });
}

auto tblgen::Operator::trait_begin() const -> const_trait_iterator {
  return traits.begin();
}
auto tblgen::Operator::trait_end() const -> const_trait_iterator {
  return traits.end();
}
auto tblgen::Operator::getTraits() const
    -> llvm::iterator_range<const_trait_iterator> {
  return {trait_begin(), trait_end()};
}

auto tblgen::Operator::attribute_begin() const -> attribute_iterator {
  return attributes.begin();
}
auto tblgen::Operator::attribute_end() const -> attribute_iterator {
  return attributes.end();
}
auto tblgen::Operator::getAttributes() const
    -> llvm::iterator_range<attribute_iterator> {
  return {attribute_begin(), attribute_end()};
}

auto tblgen::Operator::operand_begin() -> value_iterator {
  return operands.begin();
}
auto tblgen::Operator::operand_end() -> value_iterator {
  return operands.end();
}
auto tblgen::Operator::getOperands() -> value_range {
  return {operand_begin(), operand_end()};
}

auto tblgen::Operator::getArg(int index) const -> Argument {
  return arguments[index];
}

// Mapping from result index to combined argument and result index. Arguments
// are indexed to match getArg index, while the result indexes are mapped to
// avoid overlap.
static int resultIndex(int i) { return -1 - i; }

bool tblgen::Operator::isVariadic() const {
  return any_of(llvm::concat<const NamedTypeConstraint>(operands, results),
                [](const NamedTypeConstraint &op) { return op.isVariadic(); });
}

void tblgen::Operator::populateTypeInferenceInfo(
    const llvm::StringMap<int> &argumentsAndResultsIndex) {
  // If the type inference op interface is not registered, then do not attempt
  // to determine if the result types an be inferred.
  auto &recordKeeper = def.getRecords();
  auto *inferTrait = recordKeeper.getDef(inferTypeOpInterface);
  allResultsHaveKnownTypes = false;
  if (!inferTrait)
    return;

  // If there are no results, the skip this else the build method generated
  // overlaps with another autogenerated builder.
  if (getNumResults() == 0)
    return;

  // Skip for ops with variadic operands/results.
  // TODO: This can be relaxed.
  if (isVariadic())
    return;

  // Skip cases currently being custom generated.
  // TODO: Remove special cases.
  if (getTrait("OpTrait::SameOperandsAndResultType"))
    return;

  // We create equivalence classes of argument/result types where arguments
  // and results are mapped into the same index space and indices corresponding
  // to the same type are in the same equivalence class.
  llvm::EquivalenceClasses<int> ecs;
  resultTypeMapping.resize(getNumResults());
  // Captures the argument whose type matches a given result type. Preference
  // towards capturing operands first before attributes.
  auto captureMapping = [&](int i) {
    bool found = false;
    ecs.insert(resultIndex(i));
    auto mi = ecs.findLeader(resultIndex(i));
    for (auto me = ecs.member_end(); mi != me; ++mi) {
      if (*mi < 0) {
        auto tc = getResultTypeConstraint(i);
        if (tc.getBuilderCall().hasValue()) {
          resultTypeMapping[i].emplace_back(tc);
          found = true;
        }
        continue;
      }

      if (getArg(*mi).isa<NamedAttribute *>()) {
        // TODO: Handle attributes.
        continue;
      } else {
        resultTypeMapping[i].emplace_back(*mi);
        found = true;
      }
    }
    return found;
  };

  for (const OpTrait &trait : traits) {
    const llvm::Record &def = trait.getDef();
    // If the infer type op interface was manually added, then treat it as
    // intention that the op needs special handling.
    // TODO: Reconsider whether to always generate, this is more conservative
    // and keeps existing behavior so starting that way for now.
    if (def.isSubClassOf(
            llvm::formatv("{0}::Trait", inferTypeOpInterface).str()))
      return;
    if (const auto *opTrait = dyn_cast<tblgen::InterfaceOpTrait>(&trait))
      if (opTrait->getTrait().startswith(inferTypeOpInterface))
        return;

    if (!def.isSubClassOf("AllTypesMatch"))
      continue;

    auto values = def.getValueAsListOfStrings("values");
    auto root = argumentsAndResultsIndex.lookup(values.front());
    for (StringRef str : values)
      ecs.unionSets(argumentsAndResultsIndex.lookup(str), root);
  }

  // Verifies that all output types have a corresponding known input type
  // and chooses matching operand or attribute (in that order) that
  // matches it.
  allResultsHaveKnownTypes =
      all_of(llvm::seq<int>(0, getNumResults()), captureMapping);

  // If the types could be computed, then add type inference trait.
  if (allResultsHaveKnownTypes)
    traits.push_back(OpTrait::create(inferTrait->getDefInit()));
}

void tblgen::Operator::populateOpStructure() {
  auto &recordKeeper = def.getRecords();
  auto *typeConstraintClass = recordKeeper.getClass("TypeConstraint");
  auto *attrClass = recordKeeper.getClass("Attr");
  auto *derivedAttrClass = recordKeeper.getClass("DerivedAttr");
  auto *opVarClass = recordKeeper.getClass("OpVariable");
  numNativeAttributes = 0;

  DagInit *argumentValues = def.getValueAsDag("arguments");
  unsigned numArgs = argumentValues->getNumArgs();

  // Mapping from name of to argument or result index. Arguments are indexed
  // to match getArg index, while the results are negatively indexed.
  llvm::StringMap<int> argumentsAndResultsIndex;

  // Handle operands and native attributes.
  for (unsigned i = 0; i != numArgs; ++i) {
    auto *arg = argumentValues->getArg(i);
    auto givenName = argumentValues->getArgNameStr(i);
    auto *argDefInit = dyn_cast<DefInit>(arg);
    if (!argDefInit)
      PrintFatalError(def.getLoc(),
                      Twine("undefined type for argument #") + Twine(i));
    Record *argDef = argDefInit->getDef();
    if (argDef->isSubClassOf(opVarClass))
      argDef = argDef->getValueAsDef("constraint");

    if (argDef->isSubClassOf(typeConstraintClass)) {
      operands.push_back(
          NamedTypeConstraint{givenName, TypeConstraint(argDef)});
    } else if (argDef->isSubClassOf(attrClass)) {
      if (givenName.empty())
        PrintFatalError(argDef->getLoc(), "attributes must be named");
      if (argDef->isSubClassOf(derivedAttrClass))
        PrintFatalError(argDef->getLoc(),
                        "derived attributes not allowed in argument list");
      attributes.push_back({givenName, Attribute(argDef)});
      ++numNativeAttributes;
    } else {
      PrintFatalError(def.getLoc(), "unexpected def type; only defs deriving "
                                    "from TypeConstraint or Attr are allowed");
    }
    if (!givenName.empty())
      argumentsAndResultsIndex[givenName] = i;
  }

  // Handle derived attributes.
  for (const auto &val : def.getValues()) {
    if (auto *record = dyn_cast<llvm::RecordRecTy>(val.getType())) {
      if (!record->isSubClassOf(attrClass))
        continue;
      if (!record->isSubClassOf(derivedAttrClass))
        PrintFatalError(def.getLoc(),
                        "unexpected Attr where only DerivedAttr is allowed");

      if (record->getClasses().size() != 1) {
        PrintFatalError(
            def.getLoc(),
            "unsupported attribute modelling, only single class expected");
      }
      attributes.push_back(
          {cast<llvm::StringInit>(val.getNameInit())->getValue(),
           Attribute(cast<DefInit>(val.getValue()))});
    }
  }

  // Populate `arguments`. This must happen after we've finalized `operands` and
  // `attributes` because we will put their elements' pointers in `arguments`.
  // SmallVector may perform re-allocation under the hood when adding new
  // elements.
  int operandIndex = 0, attrIndex = 0;
  for (unsigned i = 0; i != numArgs; ++i) {
    Record *argDef = dyn_cast<DefInit>(argumentValues->getArg(i))->getDef();
    if (argDef->isSubClassOf(opVarClass))
      argDef = argDef->getValueAsDef("constraint");

    if (argDef->isSubClassOf(typeConstraintClass)) {
      arguments.emplace_back(&operands[operandIndex++]);
    } else {
      assert(argDef->isSubClassOf(attrClass));
      arguments.emplace_back(&attributes[attrIndex++]);
    }
  }

  auto *resultsDag = def.getValueAsDag("results");
  auto *outsOp = dyn_cast<DefInit>(resultsDag->getOperator());
  if (!outsOp || outsOp->getDef()->getName() != "outs") {
    PrintFatalError(def.getLoc(), "'results' must have 'outs' directive");
  }

  // Handle results.
  for (unsigned i = 0, e = resultsDag->getNumArgs(); i < e; ++i) {
    auto name = resultsDag->getArgNameStr(i);
    auto *resultInit = dyn_cast<DefInit>(resultsDag->getArg(i));
    if (!resultInit) {
      PrintFatalError(def.getLoc(),
                      Twine("undefined type for result #") + Twine(i));
    }
    auto *resultDef = resultInit->getDef();
    if (resultDef->isSubClassOf(opVarClass))
      resultDef = resultDef->getValueAsDef("constraint");
    results.push_back({name, TypeConstraint(resultDef)});
    if (!name.empty())
      argumentsAndResultsIndex[name] = resultIndex(i);
  }

  // Handle successors
  auto *successorsDag = def.getValueAsDag("successors");
  auto *successorsOp = dyn_cast<DefInit>(successorsDag->getOperator());
  if (!successorsOp || successorsOp->getDef()->getName() != "successor") {
    PrintFatalError(def.getLoc(),
                    "'successors' must have 'successor' directive");
  }

  for (unsigned i = 0, e = successorsDag->getNumArgs(); i < e; ++i) {
    auto name = successorsDag->getArgNameStr(i);
    auto *successorInit = dyn_cast<DefInit>(successorsDag->getArg(i));
    if (!successorInit) {
      PrintFatalError(def.getLoc(),
                      Twine("undefined kind for successor #") + Twine(i));
    }
    Successor successor(successorInit->getDef());

    // Only support variadic successors if it is the last one for now.
    if (i != e - 1 && successor.isVariadic())
      PrintFatalError(def.getLoc(), "only the last successor can be variadic");
    successors.push_back({name, successor});
  }

  // Create list of traits, skipping over duplicates: appending to lists in
  // tablegen is easy, making them unique less so, so dedupe here.
  if (auto *traitList = def.getValueAsListInit("traits")) {
    // This is uniquing based on pointers of the trait.
    SmallPtrSet<const llvm::Init *, 32> traitSet;
    traits.reserve(traitSet.size());
    for (auto *traitInit : *traitList) {
      // Keep traits in the same order while skipping over duplicates.
      if (traitSet.insert(traitInit).second)
        traits.push_back(OpTrait::create(traitInit));
    }
  }

  populateTypeInferenceInfo(argumentsAndResultsIndex);

  // Handle regions
  auto *regionsDag = def.getValueAsDag("regions");
  auto *regionsOp = dyn_cast<DefInit>(regionsDag->getOperator());
  if (!regionsOp || regionsOp->getDef()->getName() != "region") {
    PrintFatalError(def.getLoc(), "'regions' must have 'region' directive");
  }

  for (unsigned i = 0, e = regionsDag->getNumArgs(); i < e; ++i) {
    auto name = regionsDag->getArgNameStr(i);
    auto *regionInit = dyn_cast<DefInit>(regionsDag->getArg(i));
    if (!regionInit) {
      PrintFatalError(def.getLoc(),
                      Twine("undefined kind for region #") + Twine(i));
    }
    Region region(regionInit->getDef());
    if (region.isVariadic()) {
      // Only support variadic regions if it is the last one for now.
      if (i != e - 1)
        PrintFatalError(def.getLoc(), "only the last region can be variadic");
      if (name.empty())
        PrintFatalError(def.getLoc(), "variadic regions must be named");
    }

    regions.push_back({name, region});
  }

  LLVM_DEBUG(print(llvm::dbgs()));
}

auto tblgen::Operator::getSameTypeAsResult(int index) const
    -> ArrayRef<ArgOrType> {
  assert(allResultTypesKnown());
  return resultTypeMapping[index];
}

ArrayRef<llvm::SMLoc> tblgen::Operator::getLoc() const { return def.getLoc(); }

bool tblgen::Operator::hasDescription() const {
  return def.getValue("description") != nullptr;
}

StringRef tblgen::Operator::getDescription() const {
  return def.getValueAsString("description");
}

bool tblgen::Operator::hasSummary() const {
  return def.getValue("summary") != nullptr;
}

StringRef tblgen::Operator::getSummary() const {
  return def.getValueAsString("summary");
}

bool tblgen::Operator::hasAssemblyFormat() const {
  auto *valueInit = def.getValueInit("assemblyFormat");
  return isa<llvm::CodeInit>(valueInit) || isa<llvm::StringInit>(valueInit);
}

StringRef tblgen::Operator::getAssemblyFormat() const {
  return TypeSwitch<llvm::Init *, StringRef>(def.getValueInit("assemblyFormat"))
      .Case<llvm::StringInit, llvm::CodeInit>(
          [&](auto *init) { return init->getValue(); });
}

void tblgen::Operator::print(llvm::raw_ostream &os) const {
  os << "op '" << getOperationName() << "'\n";
  for (Argument arg : arguments) {
    if (auto *attr = arg.dyn_cast<NamedAttribute *>())
      os << "[attribute] " << attr->name << '\n';
    else
      os << "[operand] " << arg.get<NamedTypeConstraint *>()->name << '\n';
  }
}

auto tblgen::Operator::VariableDecoratorIterator::unwrap(llvm::Init *init)
    -> VariableDecorator {
  return VariableDecorator(cast<llvm::DefInit>(init)->getDef());
}
