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
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Predicate.h"
#include "mlir/TableGen/Trait.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <list>

#define DEBUG_TYPE "mlir-tblgen-operator"

using namespace mlir;
using namespace mlir::tblgen;

using llvm::DagInit;
using llvm::DefInit;
using llvm::Init;
using llvm::ListInit;
using llvm::Record;
using llvm::StringInit;

Operator::Operator(const Record &def)
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

  cppNamespace = def.getValueAsString("cppNamespace");

  populateOpStructure();
  assertInvariants();
}

std::string Operator::getOperationName() const {
  auto prefix = dialect.getName();
  auto opName = def.getValueAsString("opName");
  if (prefix.empty())
    return std::string(opName);
  return std::string(llvm::formatv("{0}.{1}", prefix, opName));
}

std::string Operator::getAdaptorName() const {
  return std::string(llvm::formatv("{0}Adaptor", getCppClassName()));
}

std::string Operator::getGenericAdaptorName() const {
  return std::string(llvm::formatv("{0}GenericAdaptor", getCppClassName()));
}

/// Assert the invariants of accessors generated for the given name.
static void assertAccessorInvariants(const Operator &op, StringRef name) {
  std::string accessorName =
      convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);

  // Functor used to detect when an accessor will cause an overlap with an
  // operation API.
  //
  // There are a little bit more invasive checks possible for cases where not
  // all ops have the trait that would cause overlap. For many cases here,
  // renaming would be better (e.g., we can only guard in limited manner
  // against methods from traits and interfaces here, so avoiding these in op
  // definition is safer).
  auto nameOverlapsWithOpAPI = [&](StringRef newName) {
    if (newName == "AttributeNames" || newName == "Attributes" ||
        newName == "Operation")
      return true;
    if (newName == "Operands")
      return op.getNumOperands() != 1 || op.getNumVariableLengthOperands() != 1;
    if (newName == "Regions")
      return op.getNumRegions() != 1 || op.getNumVariadicRegions() != 1;
    if (newName == "Type")
      return op.getNumResults() != 1;
    return false;
  };
  if (nameOverlapsWithOpAPI(accessorName)) {
    // This error could be avoided in situations where the final function is
    // identical, but preferably the op definition should avoid using generic
    // names.
    PrintFatalError(op.getLoc(), "generated accessor for `" + name +
                                     "` overlaps with a default one; please "
                                     "rename to avoid overlap");
  }
}

void Operator::assertInvariants() const {
  // Check that the name of arguments/results/regions/successors don't overlap.
  DenseMap<StringRef, StringRef> existingNames;
  auto checkName = [&](StringRef name, StringRef entity) {
    if (name.empty())
      return;
    auto insertion = existingNames.insert({name, entity});
    if (insertion.second) {
      // Assert invariants for accessors generated for this name.
      assertAccessorInvariants(*this, name);
      return;
    }
    if (entity == insertion.first->second)
      PrintFatalError(getLoc(), "op has a conflict with two " + entity +
                                    " having the same name '" + name + "'");
    PrintFatalError(getLoc(), "op has a conflict with " +
                                  insertion.first->second + " and " + entity +
                                  " both having an entry with the name '" +
                                  name + "'");
  };
  // Check operands amongst themselves.
  for (int i : llvm::seq<int>(0, getNumOperands()))
    checkName(getOperand(i).name, "operands");

  // Check results amongst themselves and against operands.
  for (int i : llvm::seq<int>(0, getNumResults()))
    checkName(getResult(i).name, "results");

  // Check regions amongst themselves and against operands and results.
  for (int i : llvm::seq<int>(0, getNumRegions()))
    checkName(getRegion(i).name, "regions");

  // Check successors amongst themselves and against operands, results, and
  // regions.
  for (int i : llvm::seq<int>(0, getNumSuccessors()))
    checkName(getSuccessor(i).name, "successors");
}

StringRef Operator::getDialectName() const { return dialect.getName(); }

StringRef Operator::getCppClassName() const { return cppClassName; }

std::string Operator::getQualCppClassName() const {
  if (cppNamespace.empty())
    return std::string(cppClassName);
  return std::string(llvm::formatv("{0}::{1}", cppNamespace, cppClassName));
}

StringRef Operator::getCppNamespace() const { return cppNamespace; }

int Operator::getNumResults() const {
  const DagInit *results = def.getValueAsDag("results");
  return results->getNumArgs();
}

StringRef Operator::getExtraClassDeclaration() const {
  constexpr auto attr = "extraClassDeclaration";
  if (def.isValueUnset(attr))
    return {};
  return def.getValueAsString(attr);
}

StringRef Operator::getExtraClassDefinition() const {
  constexpr auto attr = "extraClassDefinition";
  if (def.isValueUnset(attr))
    return {};
  return def.getValueAsString(attr);
}

const Record &Operator::getDef() const { return def; }

bool Operator::skipDefaultBuilders() const {
  return def.getValueAsBit("skipDefaultBuilders");
}

auto Operator::result_begin() const -> const_value_iterator {
  return results.begin();
}

auto Operator::result_end() const -> const_value_iterator {
  return results.end();
}

auto Operator::getResults() const -> const_value_range {
  return {result_begin(), result_end()};
}

TypeConstraint Operator::getResultTypeConstraint(int index) const {
  const DagInit *results = def.getValueAsDag("results");
  return TypeConstraint(cast<DefInit>(results->getArg(index)));
}

StringRef Operator::getResultName(int index) const {
  const DagInit *results = def.getValueAsDag("results");
  return results->getArgNameStr(index);
}

auto Operator::getResultDecorators(int index) const -> var_decorator_range {
  const Record *result =
      cast<DefInit>(def.getValueAsDag("results")->getArg(index))->getDef();
  if (!result->isSubClassOf("OpVariable"))
    return var_decorator_range(nullptr, nullptr);
  return *result->getValueAsListInit("decorators");
}

unsigned Operator::getNumVariableLengthResults() const {
  return llvm::count_if(results, [](const NamedTypeConstraint &c) {
    return c.constraint.isVariableLength();
  });
}

unsigned Operator::getNumVariableLengthOperands() const {
  return llvm::count_if(operands, [](const NamedTypeConstraint &c) {
    return c.constraint.isVariableLength();
  });
}

bool Operator::hasSingleVariadicArg() const {
  return getNumArgs() == 1 && isa<NamedTypeConstraint *>(getArg(0)) &&
         getOperand(0).isVariadic();
}

Operator::arg_iterator Operator::arg_begin() const { return arguments.begin(); }

Operator::arg_iterator Operator::arg_end() const { return arguments.end(); }

Operator::arg_range Operator::getArgs() const {
  return {arg_begin(), arg_end()};
}

StringRef Operator::getArgName(int index) const {
  const DagInit *argumentValues = def.getValueAsDag("arguments");
  return argumentValues->getArgNameStr(index);
}

auto Operator::getArgDecorators(int index) const -> var_decorator_range {
  const Record *arg =
      cast<DefInit>(def.getValueAsDag("arguments")->getArg(index))->getDef();
  if (!arg->isSubClassOf("OpVariable"))
    return var_decorator_range(nullptr, nullptr);
  return *arg->getValueAsListInit("decorators");
}

const Trait *Operator::getTrait(StringRef trait) const {
  for (const auto &t : traits) {
    if (const auto *traitDef = dyn_cast<NativeTrait>(&t)) {
      if (traitDef->getFullyQualifiedTraitName() == trait)
        return traitDef;
    } else if (const auto *traitDef = dyn_cast<InternalTrait>(&t)) {
      if (traitDef->getFullyQualifiedTraitName() == trait)
        return traitDef;
    } else if (const auto *traitDef = dyn_cast<InterfaceTrait>(&t)) {
      if (traitDef->getFullyQualifiedTraitName() == trait)
        return traitDef;
    }
  }
  return nullptr;
}

auto Operator::region_begin() const -> const_region_iterator {
  return regions.begin();
}
auto Operator::region_end() const -> const_region_iterator {
  return regions.end();
}
auto Operator::getRegions() const
    -> llvm::iterator_range<const_region_iterator> {
  return {region_begin(), region_end()};
}

unsigned Operator::getNumRegions() const { return regions.size(); }

const NamedRegion &Operator::getRegion(unsigned index) const {
  return regions[index];
}

unsigned Operator::getNumVariadicRegions() const {
  return llvm::count_if(regions,
                        [](const NamedRegion &c) { return c.isVariadic(); });
}

auto Operator::successor_begin() const -> const_successor_iterator {
  return successors.begin();
}
auto Operator::successor_end() const -> const_successor_iterator {
  return successors.end();
}
auto Operator::getSuccessors() const
    -> llvm::iterator_range<const_successor_iterator> {
  return {successor_begin(), successor_end()};
}

unsigned Operator::getNumSuccessors() const { return successors.size(); }

const NamedSuccessor &Operator::getSuccessor(unsigned index) const {
  return successors[index];
}

unsigned Operator::getNumVariadicSuccessors() const {
  return llvm::count_if(successors,
                        [](const NamedSuccessor &c) { return c.isVariadic(); });
}

auto Operator::trait_begin() const -> const_trait_iterator {
  return traits.begin();
}
auto Operator::trait_end() const -> const_trait_iterator {
  return traits.end();
}
auto Operator::getTraits() const -> llvm::iterator_range<const_trait_iterator> {
  return {trait_begin(), trait_end()};
}

auto Operator::attribute_begin() const -> const_attribute_iterator {
  return attributes.begin();
}
auto Operator::attribute_end() const -> const_attribute_iterator {
  return attributes.end();
}
auto Operator::getAttributes() const
    -> llvm::iterator_range<const_attribute_iterator> {
  return {attribute_begin(), attribute_end()};
}
auto Operator::attribute_begin() -> attribute_iterator {
  return attributes.begin();
}
auto Operator::attribute_end() -> attribute_iterator {
  return attributes.end();
}
auto Operator::getAttributes() -> llvm::iterator_range<attribute_iterator> {
  return {attribute_begin(), attribute_end()};
}

auto Operator::operand_begin() const -> const_value_iterator {
  return operands.begin();
}
auto Operator::operand_end() const -> const_value_iterator {
  return operands.end();
}
auto Operator::getOperands() const -> const_value_range {
  return {operand_begin(), operand_end()};
}

auto Operator::getArg(int index) const -> Argument { return arguments[index]; }

bool Operator::isVariadic() const {
  return any_of(llvm::concat<const NamedTypeConstraint>(operands, results),
                [](const NamedTypeConstraint &op) { return op.isVariadic(); });
}

void Operator::populateTypeInferenceInfo(
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

  // Skip ops with variadic or optional results.
  if (getNumVariableLengthResults() > 0)
    return;

  // Skip cases currently being custom generated.
  // TODO: Remove special cases.
  if (getTrait("::mlir::OpTrait::SameOperandsAndResultType")) {
    // Check for a non-variable length operand to use as the type anchor.
    auto *operandI = llvm::find_if(arguments, [](const Argument &arg) {
      NamedTypeConstraint *operand = llvm::dyn_cast_if_present<NamedTypeConstraint *>(arg);
      return operand && !operand->isVariableLength();
    });
    if (operandI == arguments.end())
      return;

    // All result types are inferred from the operand type.
    int operandIdx = operandI - arguments.begin();
    for (int i = 0; i < getNumResults(); ++i)
      resultTypeMapping.emplace_back(operandIdx, "$_self");

    allResultsHaveKnownTypes = true;
    traits.push_back(Trait::create(inferTrait->getDefInit()));
    return;
  }

  /// This struct represents a node in this operation's result type inferenece
  /// graph. Each node has a list of incoming type inference edges `sources`.
  /// Each edge represents a "source" from which the result type can be
  /// inferred, either an operand (leaf) or another result (node). When a node
  /// is known to have a fully-inferred type, `inferred` is set to true.
  struct ResultTypeInference {
    /// The list of incoming type inference edges.
    SmallVector<InferredResultType> sources;
    /// This flag is set to true when the result type is known to be inferrable.
    bool inferred = false;
  };

  // This vector represents the type inference graph, with one node for each
  // operation result. The nth element is the node for the nth result.
  SmallVector<ResultTypeInference> inference(getNumResults(), {});

  // For all results whose types are buildable, initialize their type inference
  // nodes with an edge to themselves. Mark those nodes are fully-inferred.
  for (auto [idx, infer] : llvm::enumerate(inference)) {
    if (getResult(idx).constraint.getBuilderCall()) {
      infer.sources.emplace_back(InferredResultType::mapResultIndex(idx),
                                 "$_self");
      infer.inferred = true;
    }
  }

  // Use `AllTypesMatch` and `TypesMatchWith` operation traits to build the
  // result type inference graph.
  for (const Trait &trait : traits) {
    const Record &def = trait.getDef();

    // If the infer type op interface was manually added, then treat it as
    // intention that the op needs special handling.
    // TODO: Reconsider whether to always generate, this is more conservative
    // and keeps existing behavior so starting that way for now.
    if (def.isSubClassOf(
            llvm::formatv("{0}::Trait", inferTypeOpInterface).str()))
      return;
    if (const auto *traitDef = dyn_cast<InterfaceTrait>(&trait))
      if (&traitDef->getDef() == inferTrait)
        return;

    // The `TypesMatchWith` trait represents a 1 -> 1 type inference edge with a
    // type transformer.
    if (def.isSubClassOf("TypesMatchWith")) {
      int target = argumentsAndResultsIndex.lookup(def.getValueAsString("rhs"));
      // Ignore operand type inference.
      if (InferredResultType::isArgIndex(target))
        continue;
      int resultIndex = InferredResultType::unmapResultIndex(target);
      ResultTypeInference &infer = inference[resultIndex];
      // If the type of the result has already been inferred, do nothing.
      if (infer.inferred)
        continue;
      int sourceIndex =
          argumentsAndResultsIndex.lookup(def.getValueAsString("lhs"));
      infer.sources.emplace_back(sourceIndex,
                                 def.getValueAsString("transformer").str());
      // Locally propagate inferredness.
      infer.inferred =
          InferredResultType::isArgIndex(sourceIndex) ||
          inference[InferredResultType::unmapResultIndex(sourceIndex)].inferred;
      continue;
    }

    if (!def.isSubClassOf("AllTypesMatch"))
      continue;

    auto values = def.getValueAsListOfStrings("values");
    // The `AllTypesMatch` trait represents an N <-> N fanin and fanout. That
    // is, every result type has an edge from every other type. However, if any
    // one of the values refers to an operand or a result with a fully-inferred
    // type, we can infer all other types from that value. Try to find a
    // fully-inferred type in the list.
    std::optional<int> fullyInferredIndex;
    SmallVector<int> resultIndices;
    for (StringRef name : values) {
      int index = argumentsAndResultsIndex.lookup(name);
      if (InferredResultType::isResultIndex(index))
        resultIndices.push_back(InferredResultType::unmapResultIndex(index));
      if (InferredResultType::isArgIndex(index) ||
          inference[InferredResultType::unmapResultIndex(index)].inferred)
        fullyInferredIndex = index;
    }
    if (fullyInferredIndex) {
      // Make the fully-inferred type the only source for all results that
      // aren't already inferred -- a 1 -> N fanout.
      for (int resultIndex : resultIndices) {
        ResultTypeInference &infer = inference[resultIndex];
        if (!infer.inferred) {
          infer.sources.assign(1, {*fullyInferredIndex, "$_self"});
          infer.inferred = true;
        }
      }
    } else {
      // Add an edge between every result and every other type; N <-> N.
      for (int resultIndex : resultIndices) {
        for (int otherResultIndex : resultIndices) {
          if (resultIndex == otherResultIndex)
            continue;
          inference[resultIndex].sources.emplace_back(
              InferredResultType::unmapResultIndex(otherResultIndex), "$_self");
        }
      }
    }
  }

  // Propagate inferredness until a fixed point.
  std::vector<ResultTypeInference *> worklist;
  for (ResultTypeInference &infer : inference)
    if (!infer.inferred)
      worklist.push_back(&infer);
  bool changed;
  do {
    changed = false;
    for (auto cur = worklist.begin(); cur != worklist.end();) {
      ResultTypeInference &infer = **cur;

      InferredResultType *iter =
          llvm::find_if(infer.sources, [&](const InferredResultType &source) {
            assert(InferredResultType::isResultIndex(source.getIndex()));
            return inference[InferredResultType::unmapResultIndex(
                                 source.getIndex())]
                .inferred;
          });
      if (iter == infer.sources.end()) {
        ++cur;
        continue;
      }

      changed = true;
      infer.inferred = true;
      // Make this the only source for the result. This breaks any cycles.
      infer.sources.assign(1, *iter);
      cur = worklist.erase(cur);
    }
  } while (changed);

  allResultsHaveKnownTypes = worklist.empty();

  // If the types could be computed, then add type inference trait.
  if (allResultsHaveKnownTypes) {
    traits.push_back(Trait::create(inferTrait->getDefInit()));
    for (const ResultTypeInference &infer : inference)
      resultTypeMapping.push_back(infer.sources.front());
  }
}

void Operator::populateOpStructure() {
  auto &recordKeeper = def.getRecords();
  auto *typeConstraintClass = recordKeeper.getClass("TypeConstraint");
  auto *attrClass = recordKeeper.getClass("Attr");
  auto *propertyClass = recordKeeper.getClass("Property");
  auto *derivedAttrClass = recordKeeper.getClass("DerivedAttr");
  auto *opVarClass = recordKeeper.getClass("OpVariable");
  numNativeAttributes = 0;

  const DagInit *argumentValues = def.getValueAsDag("arguments");
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
    const Record *argDef = argDefInit->getDef();
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
    } else if (argDef->isSubClassOf(propertyClass)) {
      if (givenName.empty())
        PrintFatalError(argDef->getLoc(), "properties must be named");
      properties.push_back({givenName, Property(argDef)});
    } else {
      PrintFatalError(def.getLoc(),
                      "unexpected def type; only defs deriving "
                      "from TypeConstraint or Attr or Property are allowed");
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
      attributes.push_back({cast<StringInit>(val.getNameInit())->getValue(),
                            Attribute(cast<DefInit>(val.getValue()))});
    }
  }

  // Populate `arguments`. This must happen after we've finalized `operands` and
  // `attributes` because we will put their elements' pointers in `arguments`.
  // SmallVector may perform re-allocation under the hood when adding new
  // elements.
  int operandIndex = 0, attrIndex = 0, propIndex = 0;
  for (unsigned i = 0; i != numArgs; ++i) {
    const Record *argDef =
        dyn_cast<DefInit>(argumentValues->getArg(i))->getDef();
    if (argDef->isSubClassOf(opVarClass))
      argDef = argDef->getValueAsDef("constraint");

    if (argDef->isSubClassOf(typeConstraintClass)) {
      attrOrOperandMapping.push_back(
          {OperandOrAttribute::Kind::Operand, operandIndex});
      arguments.emplace_back(&operands[operandIndex++]);
    } else if (argDef->isSubClassOf(attrClass)) {
      attrOrOperandMapping.push_back(
          {OperandOrAttribute::Kind::Attribute, attrIndex});
      arguments.emplace_back(&attributes[attrIndex++]);
    } else {
      assert(argDef->isSubClassOf(propertyClass));
      arguments.emplace_back(&properties[propIndex++]);
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
      argumentsAndResultsIndex[name] = InferredResultType::mapResultIndex(i);

    // We currently only support VariadicOfVariadic operands.
    if (results.back().constraint.isVariadicOfVariadic()) {
      PrintFatalError(
          def.getLoc(),
          "'VariadicOfVariadic' results are currently not supported");
    }
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
    SmallPtrSet<const Init *, 32> traitSet;
    traits.reserve(traitSet.size());

    // The declaration order of traits imply the verification order of traits.
    // Some traits may require other traits to be verified first then they can
    // do further verification based on those verified facts. If you see this
    // error, fix the traits declaration order by checking the `dependentTraits`
    // field.
    auto verifyTraitValidity = [&](const Record *trait) {
      auto *dependentTraits = trait->getValueAsListInit("dependentTraits");
      for (auto *traitInit : *dependentTraits)
        if (!traitSet.contains(traitInit))
          PrintFatalError(
              def.getLoc(),
              trait->getValueAsString("trait") + " requires " +
                  cast<DefInit>(traitInit)->getDef()->getValueAsString(
                      "trait") +
                  " to precede it in traits list");
    };

    std::function<void(const ListInit *)> insert;
    insert = [&](const ListInit *traitList) {
      for (auto *traitInit : *traitList) {
        auto *def = cast<DefInit>(traitInit)->getDef();
        if (def->isSubClassOf("TraitList")) {
          insert(def->getValueAsListInit("traits"));
          continue;
        }

        // Ignore duplicates.
        if (!traitSet.insert(traitInit).second)
          continue;

        // If this is an interface with base classes, add the bases to the
        // trait list.
        if (def->isSubClassOf("Interface"))
          insert(def->getValueAsListInit("baseInterfaces"));

        // Verify if the trait has all the dependent traits declared before
        // itself.
        verifyTraitValidity(def);
        traits.push_back(Trait::create(traitInit));
      }
    };
    insert(traitList);
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

  // Populate the builders.
  auto *builderList = dyn_cast_or_null<ListInit>(def.getValueInit("builders"));
  if (builderList && !builderList->empty()) {
    for (const Init *init : builderList->getValues())
      builders.emplace_back(cast<DefInit>(init)->getDef(), def.getLoc());
  } else if (skipDefaultBuilders()) {
    PrintFatalError(
        def.getLoc(),
        "default builders are skipped and no custom builders provided");
  }

  LLVM_DEBUG(print(llvm::dbgs()));
}

const InferredResultType &Operator::getInferredResultType(int index) const {
  assert(allResultTypesKnown());
  return resultTypeMapping[index];
}

ArrayRef<SMLoc> Operator::getLoc() const { return def.getLoc(); }

bool Operator::hasDescription() const {
  return !getDescription().trim().empty();
}

StringRef Operator::getDescription() const {
  return def.getValueAsString("description");
}

bool Operator::hasSummary() const { return !getSummary().trim().empty(); }

StringRef Operator::getSummary() const {
  return def.getValueAsString("summary");
}

bool Operator::hasAssemblyFormat() const {
  auto *valueInit = def.getValueInit("assemblyFormat");
  return isa<StringInit>(valueInit);
}

StringRef Operator::getAssemblyFormat() const {
  return TypeSwitch<const Init *, StringRef>(def.getValueInit("assemblyFormat"))
      .Case<StringInit>([&](auto *init) { return init->getValue(); });
}

void Operator::print(llvm::raw_ostream &os) const {
  os << "op '" << getOperationName() << "'\n";
  for (Argument arg : arguments) {
    if (auto *attr = llvm::dyn_cast_if_present<NamedAttribute *>(arg))
      os << "[attribute] " << attr->name << '\n';
    else
      os << "[operand] " << cast<NamedTypeConstraint *>(arg)->name << '\n';
  }
}

auto Operator::VariableDecoratorIterator::unwrap(const Init *init)
    -> VariableDecorator {
  return VariableDecorator(cast<DefInit>(init)->getDef());
}

auto Operator::getArgToOperandOrAttribute(int index) const
    -> OperandOrAttribute {
  return attrOrOperandMapping[index];
}

std::string Operator::getGetterName(StringRef name) const {
  return "get" + convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);
}

std::string Operator::getSetterName(StringRef name) const {
  return "set" + convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);
}

std::string Operator::getRemoverName(StringRef name) const {
  return "remove" + convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);
}

bool Operator::hasFolder() const { return def.getValueAsBit("hasFolder"); }

bool Operator::useCustomPropertiesEncoding() const {
  return def.getValueAsBit("useCustomPropertiesEncoding");
}
