//===- Operator.h - Operator class ------------------------------*- C++ -*-===//
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

#ifndef MLIR_TABLEGEN_OPERATOR_H_
#define MLIR_TABLEGEN_OPERATOR_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Builder.h"
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/Successor.h"
#include "mlir/TableGen/Trait.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
class DefInit;
class Record;
class StringInit;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class that contains a MLIR op's information (e.g., operands,
// attributes) defined in TableGen and provides helper methods for
// accessing them.
class Operator {
public:
  explicit Operator(const llvm::Record &def);
  explicit Operator(const llvm::Record *def) : Operator(*def) {}

  // Returns this op's dialect name.
  StringRef getDialectName() const;

  // Returns the operation name. The name will follow the "<dialect>.<op-name>"
  // format if its dialect name is not empty.
  std::string getOperationName() const;

  // Returns this op's C++ class name.
  StringRef getCppClassName() const;

  // Returns this op's C++ class name prefixed with namespaces.
  std::string getQualCppClassName() const;

  // Returns this op's C++ namespace.
  StringRef getCppNamespace() const;

  // Returns the name of op's adaptor C++ class.
  std::string getAdaptorName() const;

  // Check invariants (like no duplicated or conflicted names) and abort the
  // process if any invariant is broken.
  void assertInvariants() const;

  /// A class used to represent the decorators of an operator variable, i.e.
  /// argument or result.
  struct VariableDecorator {
  public:
    explicit VariableDecorator(const llvm::Record *def) : def(def) {}
    const llvm::Record &getDef() const { return *def; }

  protected:
    // The TableGen definition of this decorator.
    const llvm::Record *def;
  };

  // A utility iterator over a list of variable decorators.
  struct VariableDecoratorIterator
      : public llvm::mapped_iterator<llvm::Init *const *,
                                     VariableDecorator (*)(llvm::Init *)> {
    /// Initializes the iterator to the specified iterator.
    VariableDecoratorIterator(llvm::Init *const *it)
        : llvm::mapped_iterator<llvm::Init *const *,
                                VariableDecorator (*)(llvm::Init *)>(it,
                                                                     &unwrap) {}
    static VariableDecorator unwrap(llvm::Init *init);
  };
  using var_decorator_iterator = VariableDecoratorIterator;
  using var_decorator_range = llvm::iterator_range<VariableDecoratorIterator>;

  using value_iterator = NamedTypeConstraint *;
  using const_value_iterator = const NamedTypeConstraint *;
  using value_range = llvm::iterator_range<value_iterator>;
  using const_value_range = llvm::iterator_range<const_value_iterator>;

  // Returns true if this op has variable length operands or results.
  bool isVariadic() const;

  // Returns true if default builders should not be generated.
  bool skipDefaultBuilders() const;

  // Op result iterators.
  const_value_iterator result_begin() const;
  const_value_iterator result_end() const;
  const_value_range getResults() const;

  // Returns the number of results this op produces.
  int getNumResults() const;

  // Returns the op result at the given `index`.
  NamedTypeConstraint &getResult(int index) { return results[index]; }
  const NamedTypeConstraint &getResult(int index) const {
    return results[index];
  }

  // Returns the `index`-th result's type constraint.
  TypeConstraint getResultTypeConstraint(int index) const;
  // Returns the `index`-th result's name.
  StringRef getResultName(int index) const;
  // Returns the `index`-th result's decorators.
  var_decorator_range getResultDecorators(int index) const;

  // Returns the number of variable length results in this operation.
  unsigned getNumVariableLengthResults() const;

  // Op attribute iterators.
  using attribute_iterator = const NamedAttribute *;
  attribute_iterator attribute_begin() const;
  attribute_iterator attribute_end() const;
  llvm::iterator_range<attribute_iterator> getAttributes() const;

  int getNumAttributes() const { return attributes.size(); }
  int getNumNativeAttributes() const { return numNativeAttributes; }

  // Op attribute accessors.
  NamedAttribute &getAttribute(int index) { return attributes[index]; }
  const NamedAttribute &getAttribute(int index) const {
    return attributes[index];
  }

  // Op operand iterators.
  const_value_iterator operand_begin() const;
  const_value_iterator operand_end() const;
  const_value_range getOperands() const;

  int getNumOperands() const { return operands.size(); }
  NamedTypeConstraint &getOperand(int index) { return operands[index]; }
  const NamedTypeConstraint &getOperand(int index) const {
    return operands[index];
  }

  // Returns the number of variadic operands in this operation.
  unsigned getNumVariableLengthOperands() const;

  // Returns the total number of arguments.
  int getNumArgs() const { return arguments.size(); }

  // Returns true of the operation has a single variadic arg.
  bool hasSingleVariadicArg() const;

  // Returns true if the operation has a single variadic result.
  bool hasSingleVariadicResult() const {
    return getNumResults() == 1 && getResult(0).isVariadic();
  }

  // Returns true of the operation has no variadic regions.
  bool hasNoVariadicRegions() const { return getNumVariadicRegions() == 0; }

  using arg_iterator = const Argument *;
  using arg_range = llvm::iterator_range<arg_iterator>;

  // Op argument (attribute or operand) iterators.
  arg_iterator arg_begin() const;
  arg_iterator arg_end() const;
  arg_range getArgs() const;

  // Op argument (attribute or operand) accessors.
  Argument getArg(int index) const;
  StringRef getArgName(int index) const;
  var_decorator_range getArgDecorators(int index) const;

  // Returns the trait wrapper for the given MLIR C++ `trait`.
  const Trait *getTrait(llvm::StringRef trait) const;

  // Regions.
  using const_region_iterator = const NamedRegion *;
  const_region_iterator region_begin() const;
  const_region_iterator region_end() const;
  llvm::iterator_range<const_region_iterator> getRegions() const;

  // Returns the number of regions.
  unsigned getNumRegions() const;
  // Returns the `index`-th region.
  const NamedRegion &getRegion(unsigned index) const;

  // Returns the number of variadic regions in this operation.
  unsigned getNumVariadicRegions() const;

  // Successors.
  using const_successor_iterator = const NamedSuccessor *;
  const_successor_iterator successor_begin() const;
  const_successor_iterator successor_end() const;
  llvm::iterator_range<const_successor_iterator> getSuccessors() const;

  // Returns the number of successors.
  unsigned getNumSuccessors() const;
  // Returns the `index`-th successor.
  const NamedSuccessor &getSuccessor(unsigned index) const;

  // Returns the number of variadic successors in this operation.
  unsigned getNumVariadicSuccessors() const;

  // Trait.
  using const_trait_iterator = const Trait *;
  const_trait_iterator trait_begin() const;
  const_trait_iterator trait_end() const;
  llvm::iterator_range<const_trait_iterator> getTraits() const;

  ArrayRef<SMLoc> getLoc() const;

  // Query functions for the documentation of the operator.
  bool hasDescription() const;
  StringRef getDescription() const;
  bool hasSummary() const;
  StringRef getSummary() const;

  // Query functions for the assembly format of the operator.
  bool hasAssemblyFormat() const;
  StringRef getAssemblyFormat() const;

  // Returns this op's extra class declaration code.
  StringRef getExtraClassDeclaration() const;

  // Returns this op's extra class definition code.
  StringRef getExtraClassDefinition() const;

  // Returns the Tablegen definition this operator was constructed from.
  // TODO: do not expose the TableGen record, this is a temporary solution to
  // OpEmitter requiring a Record because Operator does not provide enough
  // methods.
  const llvm::Record &getDef() const;

  // Returns the dialect of the op.
  const Dialect &getDialect() const { return dialect; }

  // Prints the contents in this operator to the given `os`. This is used for
  // debugging purposes.
  void print(llvm::raw_ostream &os) const;

  // Return whether all the result types are known.
  bool allResultTypesKnown() const { return allResultsHaveKnownTypes; };

  // Pair representing either a index to an argument or a type constraint. Only
  // one of these entries should have the non-default value.
  struct ArgOrType {
    explicit ArgOrType(int index) : index(index), constraint(None) {}
    explicit ArgOrType(TypeConstraint constraint)
        : index(None), constraint(constraint) {}
    bool isArg() const {
      assert(constraint.has_value() ^ index.has_value());
      return index.has_value();
    }
    bool isType() const {
      assert(constraint.has_value() ^ index.has_value());
      return constraint.has_value();
    }

    int getArg() const { return *index; }
    TypeConstraint getType() const { return *constraint; }

  private:
    Optional<int> index;
    Optional<TypeConstraint> constraint;
  };

  // Return all arguments or type constraints with same type as result[index].
  // Requires: all result types are known.
  ArrayRef<ArgOrType> getSameTypeAsResult(int index) const;

  // Pair consisting kind of argument and index into operands or attributes.
  struct OperandOrAttribute {
    enum class Kind { Operand, Attribute };
    OperandOrAttribute(Kind kind, int index) {
      packed = (index << 1) | (kind == Kind::Attribute);
    }
    int operandOrAttributeIndex() const { return (packed >> 1); }
    Kind kind() { return (packed & 0x1) ? Kind::Attribute : Kind::Operand; }

  private:
    int packed;
  };

  // Returns the OperandOrAttribute corresponding to the index.
  OperandOrAttribute getArgToOperandOrAttribute(int index) const;

  // Returns the builders of this operation.
  ArrayRef<Builder> getBuilders() const { return builders; }

  // Returns the preferred getter name for the accessor.
  std::string getGetterName(StringRef name) const {
    return getGetterNames(name).front();
  }

  // Returns the getter names for the accessor.
  SmallVector<std::string, 2> getGetterNames(StringRef name) const;

  // Returns the setter names for the accessor.
  SmallVector<std::string, 2> getSetterNames(StringRef name) const;

private:
  // Populates the vectors containing operands, attributes, results and traits.
  void populateOpStructure();

  // Populates type inference info (mostly equality) with input a mapping from
  // names to indices for arguments and results.
  void populateTypeInferenceInfo(
      const llvm::StringMap<int> &argumentsAndResultsIndex);

  // The dialect of this op.
  Dialect dialect;

  // The unqualified C++ class name of the op.
  StringRef cppClassName;

  // The C++ namespace for this op.
  StringRef cppNamespace;

  // The operands of the op.
  SmallVector<NamedTypeConstraint, 4> operands;

  // The attributes of the op.  Contains native attributes (corresponding to the
  // actual stored attributed of the operation) followed by derived attributes
  // (corresponding to dynamic properties of the operation that are computed
  // upon request).
  SmallVector<NamedAttribute, 4> attributes;

  // The arguments of the op (operands and native attributes).
  SmallVector<Argument, 4> arguments;

  // The results of the op.
  SmallVector<NamedTypeConstraint, 4> results;

  // The successors of this op.
  SmallVector<NamedSuccessor, 0> successors;

  // The traits of the op.
  SmallVector<Trait, 4> traits;

  // The regions of this op.
  SmallVector<NamedRegion, 1> regions;

  // The argument with the same type as the result.
  SmallVector<SmallVector<ArgOrType, 2>, 4> resultTypeMapping;

  // Map from argument to attribute or operand number.
  SmallVector<OperandOrAttribute, 4> attrOrOperandMapping;

  // The builders of this operator.
  SmallVector<Builder> builders;

  // The number of native attributes stored in the leading positions of
  // `attributes`.
  int numNativeAttributes;

  // The TableGen definition of this op.
  const llvm::Record &def;

  // Whether the type of all results are known.
  bool allResultsHaveKnownTypes;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_OPERATOR_H_
