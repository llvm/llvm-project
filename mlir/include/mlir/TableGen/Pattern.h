//===- Pattern.h - Pattern wrapper class ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pattern wrapper class to simplify using TableGen Record defining a MLIR
// Pattern.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PATTERN_H_
#define MLIR_TABLEGEN_PATTERN_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <optional>
#include <unordered_map>

namespace llvm {
class DagInit;
class Init;
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Mapping from TableGen Record to Operator wrapper object.
//
// We allocate each wrapper object in heap to make sure the pointer to it is
// valid throughout the lifetime of this map. This is important because this map
// is shared among multiple patterns to avoid creating the wrapper object for
// the same op again and again. But this map will continuously grow.
using RecordOperatorMap =
    DenseMap<const llvm::Record *, std::unique_ptr<Operator>>;

class Pattern;

// Wrapper class providing helper methods for accessing TableGen DAG leaves
// used inside Patterns. This class is lightweight and designed to be used like
// values.
//
// A TableGen DAG construct is of the syntax
//   `(operator, arg0, arg1, ...)`.
//
// This class provides getters to retrieve `arg*` as tblgen:: wrapper objects
// for handy helper methods. It only works on `arg*`s that are not nested DAG
// constructs.
class DagLeaf {
public:
  explicit DagLeaf(const llvm::Init *def) : def(def) {}

  // Returns true if this DAG leaf is not specified in the pattern. That is, it
  // places no further constraints/transforms and just carries over the original
  // value.
  bool isUnspecified() const;

  // Returns true if this DAG leaf is matching an operand. That is, it specifies
  // a type constraint.
  bool isOperandMatcher() const;

  // Returns true if this DAG leaf is matching an attribute. That is, it
  // specifies an attribute constraint.
  bool isAttrMatcher() const;

  // Returns true if this DAG leaf is wrapping native code call.
  bool isNativeCodeCall() const;

  // Returns true if this DAG leaf is specifying a constant attribute.
  bool isConstantAttr() const;

  // Returns true if this DAG leaf is specifying an enum attribute case.
  bool isEnumAttrCase() const;

  // Returns true if this DAG leaf is specifying a string attribute.
  bool isStringAttr() const;

  // Returns this DAG leaf as a constraint. Asserts if fails.
  Constraint getAsConstraint() const;

  // Returns this DAG leaf as an constant attribute. Asserts if fails.
  ConstantAttr getAsConstantAttr() const;

  // Returns this DAG leaf as an enum attribute case.
  // Precondition: isEnumAttrCase()
  EnumAttrCase getAsEnumAttrCase() const;

  // Returns the matching condition template inside this DAG leaf. Assumes the
  // leaf is an operand/attribute matcher and asserts otherwise.
  std::string getConditionTemplate() const;

  // Returns the native code call template inside this DAG leaf.
  // Precondition: isNativeCodeCall()
  StringRef getNativeCodeTemplate() const;

  // Returns the number of values will be returned by the native helper
  // function.
  // Precondition: isNativeCodeCall()
  int getNumReturnsOfNativeCode() const;

  // Returns the string associated with the leaf.
  // Precondition: isStringAttr()
  std::string getStringAttr() const;

  void print(raw_ostream &os) const;

private:
  friend llvm::DenseMapInfo<DagLeaf>;
  const void *getAsOpaquePointer() const { return def; }

  // Returns true if the TableGen Init `def` in this DagLeaf is a DefInit and
  // also a subclass of the given `superclass`.
  bool isSubClassOf(StringRef superclass) const;

  const llvm::Init *def;
};

// Wrapper class providing helper methods for accessing TableGen DAG constructs
// used inside Patterns. This class is lightweight and designed to be used like
// values.
//
// A TableGen DAG construct is of the syntax
//   `(operator, arg0, arg1, ...)`.
//
// When used inside Patterns, `operator` corresponds to some dialect op, or
// a known list of verbs that defines special transformation actions. This
// `arg*` can be a nested DAG construct. This class provides getters to
// retrieve `operator` and `arg*` as tblgen:: wrapper objects for handy helper
// methods.
//
// A null DagNode contains a nullptr and converts to false implicitly.
class DagNode {
public:
  explicit DagNode(const llvm::DagInit *node) : node(node) {}

  // Implicit bool converter that returns true if this DagNode is not a null
  // DagNode.
  operator bool() const { return node != nullptr; }

  // Returns the symbol bound to this DAG node.
  StringRef getSymbol() const;

  // Returns the operator wrapper object corresponding to the dialect op matched
  // by this DAG. The operator wrapper will be queried from the given `mapper`
  // and created in it if not existing.
  Operator &getDialectOp(RecordOperatorMap *mapper) const;

  // Returns the number of operations recursively involved in the DAG tree
  // rooted from this node.
  int getNumOps() const;

  // Returns the number of immediate arguments to this DAG node.
  int getNumArgs() const;

  // Returns true if the `index`-th argument is a nested DAG construct.
  bool isNestedDagArg(unsigned index) const;

  // Gets the `index`-th argument as a nested DAG construct if possible. Returns
  // null DagNode otherwise.
  DagNode getArgAsNestedDag(unsigned index) const;

  // Gets the `index`-th argument as a DAG leaf.
  DagLeaf getArgAsLeaf(unsigned index) const;

  // Returns the specified name of the `index`-th argument.
  StringRef getArgName(unsigned index) const;

  // Returns true if this DAG construct means to replace with an existing SSA
  // value.
  bool isReplaceWithValue() const;

  // Returns whether this DAG represents the location of an op creation.
  bool isLocationDirective() const;

  // Returns whether this DAG is a return type specifier.
  bool isReturnTypeDirective() const;

  // Returns true if this DAG node is wrapping native code call.
  bool isNativeCodeCall() const;

  // Returns whether this DAG is an `either` specifier.
  bool isEither() const;

  // Returns whether this DAG is an `variadic` specifier.
  bool isVariadic() const;

  // Returns true if this DAG node is an operation.
  bool isOperation() const;

  // Returns the native code call template inside this DAG node.
  // Precondition: isNativeCodeCall()
  StringRef getNativeCodeTemplate() const;

  // Returns the number of values will be returned by the native helper
  // function.
  // Precondition: isNativeCodeCall()
  int getNumReturnsOfNativeCode() const;

  void print(raw_ostream &os) const;

private:
  friend class SymbolInfoMap;
  friend llvm::DenseMapInfo<DagNode>;
  const void *getAsOpaquePointer() const { return node; }

  const llvm::DagInit *node; // nullptr means null DagNode
};

// A class for maintaining information for symbols bound in patterns and
// provides methods for resolving them according to specific use cases.
//
// Symbols can be bound to
//
// * Op arguments and op results in the source pattern and
// * Op results in result patterns.
//
// Symbols can be referenced in result patterns and additional constraints to
// the pattern.
//
// For example, in
//
// ```
// def : Pattern<
//     (SrcOp:$results1 $arg0, %arg1),
//     [(ResOp1:$results2), (ResOp2 $results2 (ResOp3 $arg0, $arg1))]>;
// ```
//
// `$argN` is bound to the `SrcOp`'s N-th argument. `$results1` is bound to
// `SrcOp`. `$results2` is bound to `ResOp1`. $result2 is referenced to build
// `ResOp2`. `$arg0` and `$arg1` are referenced to build `ResOp3`.
//
// If a symbol binds to a multi-result op and it does not have the `__N`
// suffix, the symbol is expanded to represent all results generated by the
// multi-result op. If the symbol has a `__N` suffix, then it will expand to
// only the N-th *static* result as declared in ODS, and that can still
// corresponds to multiple *dynamic* values if the N-th *static* result is
// variadic.
//
// This class keeps track of such symbols and resolves them into their bound
// values in a suitable way.
class SymbolInfoMap {
public:
  explicit SymbolInfoMap(ArrayRef<SMLoc> loc) : loc(loc) {}

  // Class for information regarding a symbol.
  class SymbolInfo {
  public:
    // Returns a type string of a variable.
    std::string getVarTypeStr(StringRef name) const;

    // Returns a string for defining a variable named as `name` to store the
    // value bound by this symbol.
    std::string getVarDecl(StringRef name) const;

    // Returns a string for defining an argument which passes the reference of
    // the variable.
    std::string getArgDecl(StringRef name) const;

    // Returns a variable name for the symbol named as `name`.
    std::string getVarName(StringRef name) const;

  private:
    // Allow SymbolInfoMap to access private methods.
    friend class SymbolInfoMap;

    // Structure to uniquely distinguish different locations of the symbols.
    //
    // * If a symbol is defined as an operand of an operation, `dag` specifies
    //   the DAG of the operation, `operandIndexOrNumValues` specifies the
    //   operand index, and `variadicSubIndex` must be set to `std::nullopt`.
    //
    // * If a symbol is defined in a `variadic` DAG, `dag` specifies the DAG
    //   of the parent operation, `operandIndexOrNumValues` specifies the
    //   declared operand index of the variadic operand in the parent
    //   operation.
    //
    //   - If the symbol is defined as a result of `variadic` DAG, the
    //     `variadicSubIndex` must be set to `std::nullopt`, which means that
    //     the symbol binds to the full operand range.
    //
    //   - If the symbol is defined as a operand, the `variadicSubIndex` must
    //     be set to the index within the variadic sub-operand list.
    //
    // * If a symbol is defined in a `either` DAG, `dag` specifies the DAG
    //   of the parent operation, `operandIndexOrNumValues` specifies the
    //   operand index in the parent operation (not necessary the index in the
    //   DAG).
    //
    // * If a symbol is defined as a result, specifies the number of returning
    //   value.
    //
    // Example 1:
    //
    //   def : Pat<(OpA $input0, $input1), ...>;
    //
    //   $input0: (OpA, 0, nullopt)
    //   $input1: (OpA, 1, nullopt)
    //
    // Example 2:
    //
    //   def : Pat<(OpB (variadic:$input0 $input0a, $input0b),
    //                  (variadic:$input1 $input1a, $input1b, $input1c)),
    //             ...>;
    //
    //   $input0:  (OpB, 0, nullopt)
    //   $input0a: (OpB, 0, 0)
    //   $input0b: (OpB, 0, 1)
    //   $input1:  (OpB, 1, nullopt)
    //   $input1a: (OpB, 1, 0)
    //   $input1b: (OpB, 1, 1)
    //   $input1c: (OpB, 1, 2)
    //
    // Example 3:
    //
    //   def : Pat<(OpC $input0, (either $input1, $input2)), ...>;
    //
    //   $input0: (OpC, 0, nullopt)
    //   $input1: (OpC, 1, nullopt)
    //   $input2: (OpC, 2, nullopt)
    //
    // Example 4:
    //
    //   def ThreeResultOp : TEST_Op<...> {
    //     let results = (outs
    //       AnyType:$result1,
    //       AnyType:$result2,
    //       AnyType:$result3
    //     );
    //   }
    //
    //   def : Pat<...,
    //             (ThreeResultOp:$result ...)>;
    //
    //   $result: (nullptr, 3, nullopt)
    //
    struct DagAndConstant {
      // DagNode and DagLeaf are accessed by value which means it can't be used
      // as identifier here. Use an opaque pointer type instead.
      const void *dag;
      int operandIndexOrNumValues;
      std::optional<int> variadicSubIndex;

      DagAndConstant(const void *dag, int operandIndexOrNumValues,
                     std::optional<int> variadicSubIndex)
          : dag(dag), operandIndexOrNumValues(operandIndexOrNumValues),
            variadicSubIndex(variadicSubIndex) {}

      bool operator==(const DagAndConstant &rhs) const {
        return dag == rhs.dag &&
               operandIndexOrNumValues == rhs.operandIndexOrNumValues &&
               variadicSubIndex == rhs.variadicSubIndex;
      }
    };

    // What kind of entity this symbol represents:
    // * Attr: op attribute
    // * Operand: op operand
    // * Result: op result
    // * Value: a value not attached to an op (e.g., from NativeCodeCall)
    // * MultipleValues: a pack of values not attached to an op (e.g., from
    //   NativeCodeCall). This kind supports indexing.
    enum class Kind : uint8_t { Attr, Operand, Result, Value, MultipleValues };

    // Creates a SymbolInfo instance. `dagAndConstant` is only used for `Attr`
    // and `Operand` so should be std::nullopt for `Result` and `Value` kind.
    SymbolInfo(const Operator *op, Kind kind,
               std::optional<DagAndConstant> dagAndConstant);

    // Static methods for creating SymbolInfo.
    static SymbolInfo getAttr(const Operator *op, int index) {
      return SymbolInfo(op, Kind::Attr,
                        DagAndConstant(nullptr, index, std::nullopt));
    }
    static SymbolInfo getAttr() {
      return SymbolInfo(nullptr, Kind::Attr, std::nullopt);
    }
    static SymbolInfo
    getOperand(DagNode node, const Operator *op, int operandIndex,
               std::optional<int> variadicSubIndex = std::nullopt) {
      return SymbolInfo(op, Kind::Operand,
                        DagAndConstant(node.getAsOpaquePointer(), operandIndex,
                                       variadicSubIndex));
    }
    static SymbolInfo getResult(const Operator *op) {
      return SymbolInfo(op, Kind::Result, std::nullopt);
    }
    static SymbolInfo getValue() {
      return SymbolInfo(nullptr, Kind::Value, std::nullopt);
    }
    static SymbolInfo getMultipleValues(int numValues) {
      return SymbolInfo(nullptr, Kind::MultipleValues,
                        DagAndConstant(nullptr, numValues, std::nullopt));
    }

    // Returns the number of static values this symbol corresponds to.
    // A static value is an operand/result declared in ODS. Normally a symbol
    // only represents one static value, but symbols bound to op results can
    // represent more than one if the op is a multi-result op.
    int getStaticValueCount() const;

    // Returns a string containing the C++ expression for referencing this
    // symbol as a value (if this symbol represents one static value) or a value
    // range (if this symbol represents multiple static values). `name` is the
    // name of the C++ variable that this symbol bounds to. `index` should only
    // be used for indexing results.  `fmt` is used to format each value.
    // `separator` is used to separate values if this is a value range.
    std::string getValueAndRangeUse(StringRef name, int index, const char *fmt,
                                    const char *separator) const;

    // Returns a string containing the C++ expression for referencing this
    // symbol as a value range regardless of how many static values this symbol
    // represents. `name` is the name of the C++ variable that this symbol
    // bounds to. `index` should only be used for indexing results. `fmt` is
    // used to format each value. `separator` is used to separate values in the
    // range.
    std::string getAllRangeUse(StringRef name, int index, const char *fmt,
                               const char *separator) const;

    // The argument index (for `Attr` and `Operand` only)
    int getArgIndex() const { return dagAndConstant->operandIndexOrNumValues; }

    // The number of values in the MultipleValue
    int getSize() const { return dagAndConstant->operandIndexOrNumValues; }

    // The variadic sub-operands index (for variadic `Operand` only)
    std::optional<int> getVariadicSubIndex() const {
      return dagAndConstant->variadicSubIndex;
    }

    const Operator *op; // The op where the bound entity belongs
    Kind kind;          // The kind of the bound entity

    // The tuple of DagNode pointer and two constant values (for `Attr`,
    // `Operand` and the size of MultipleValue symbol). Note that operands may
    // be bound to the same symbol, use the DagNode and index to distinguish
    // them. For `Attr` and MultipleValue, the Dag part will be nullptr.
    std::optional<DagAndConstant> dagAndConstant;

    // Alternative name for the symbol. It is used in case the name
    // is not unique. Applicable for `Operand` only.
    std::optional<std::string> alternativeName;
  };

  using BaseT = std::unordered_multimap<std::string, SymbolInfo>;

  // Iterators for accessing all symbols.
  using iterator = BaseT::iterator;
  iterator begin() { return symbolInfoMap.begin(); }
  iterator end() { return symbolInfoMap.end(); }

  // Const iterators for accessing all symbols.
  using const_iterator = BaseT::const_iterator;
  const_iterator begin() const { return symbolInfoMap.begin(); }
  const_iterator end() const { return symbolInfoMap.end(); }

  // Binds the given `symbol` to the `argIndex`-th argument to the given `op`.
  // Returns false if `symbol` is already bound and symbols are not operands.
  bool bindOpArgument(DagNode node, StringRef symbol, const Operator &op,
                      int argIndex,
                      std::optional<int> variadicSubIndex = std::nullopt);

  // Binds the given `symbol` to the results the given `op`. Returns false if
  // `symbol` is already bound.
  bool bindOpResult(StringRef symbol, const Operator &op);

  // A helper function for dispatching target value binding functions.
  bool bindValues(StringRef symbol, int numValues = 1);

  // Registers the given `symbol` as bound to the Value(s). Returns false if
  // `symbol` is already bound.
  bool bindValue(StringRef symbol);

  // Registers the given `symbol` as bound to a MultipleValue. Return false if
  // `symbol` is already bound.
  bool bindMultipleValues(StringRef symbol, int numValues);

  // Registers the given `symbol` as bound to an attr. Returns false if `symbol`
  // is already bound.
  bool bindAttr(StringRef symbol);

  // Returns true if the given `symbol` is bound.
  bool contains(StringRef symbol) const;

  // Returns an iterator to the information of the given symbol named as `key`.
  const_iterator find(StringRef key) const;

  // Returns an iterator to the information of the given symbol named as `key`,
  // with index `argIndex` for operator `op`.
  const_iterator findBoundSymbol(StringRef key, DagNode node,
                                 const Operator &op, int argIndex,
                                 std::optional<int> variadicSubIndex) const;
  const_iterator findBoundSymbol(StringRef key,
                                 const SymbolInfo &symbolInfo) const;

  // Returns the bounds of a range that includes all the elements which
  // bind to the `key`.
  std::pair<iterator, iterator> getRangeOfEqualElements(StringRef key);

  // Returns number of times symbol named as `key` was used.
  int count(StringRef key) const;

  // Returns the number of static values of the given `symbol` corresponds to.
  // A static value is an operand/result declared in ODS. Normally a symbol only
  // represents one static value, but symbols bound to op results can represent
  // more than one if the op is a multi-result op.
  int getStaticValueCount(StringRef symbol) const;

  // Returns a string containing the C++ expression for referencing this
  // symbol as a value (if this symbol represents one static value) or a value
  // range (if this symbol represents multiple static values). `fmt` is used to
  // format each value. `separator` is used to separate values if `symbol`
  // represents a value range.
  std::string getValueAndRangeUse(StringRef symbol, const char *fmt = "{0}",
                                  const char *separator = ", ") const;

  // Returns a string containing the C++ expression for referencing this
  // symbol as a value range regardless of how many static values this symbol
  // represents. `fmt` is used to format each value. `separator` is used to
  // separate values in the range.
  std::string getAllRangeUse(StringRef symbol, const char *fmt = "{0}",
                             const char *separator = ", ") const;

  // Assign alternative unique names to Operands that have equal names.
  void assignUniqueAlternativeNames();

  // Splits the given `symbol` into a value pack name and an index. Returns the
  // value pack name and writes the index to `index` on success. Returns
  // `symbol` itself if it does not contain an index.
  //
  // We can use `name__N` to access the `N`-th value in the value pack bound to
  // `name`. `name` is typically the results of an multi-result op.
  static StringRef getValuePackName(StringRef symbol, int *index = nullptr);

private:
  BaseT symbolInfoMap;

  // Pattern instantiation location. This is intended to be used as parameter
  // to PrintFatalError() to report errors.
  ArrayRef<SMLoc> loc;
};

// Wrapper class providing helper methods for accessing MLIR Pattern defined
// in TableGen. This class should closely reflect what is defined as class
// `Pattern` in TableGen. This class contains maps so it is not intended to be
// used as values.
class Pattern {
public:
  explicit Pattern(const llvm::Record *def, RecordOperatorMap *mapper);

  // Returns the source pattern to match.
  DagNode getSourcePattern() const;

  // Returns the number of result patterns generated by applying this rewrite
  // rule.
  int getNumResultPatterns() const;

  // Returns the DAG tree root node of the `index`-th result pattern.
  DagNode getResultPattern(unsigned index) const;

  // Collects all symbols bound in the source pattern into `infoMap`.
  void collectSourcePatternBoundSymbols(SymbolInfoMap &infoMap);

  // Collects all symbols bound in result patterns into `infoMap`.
  void collectResultPatternBoundSymbols(SymbolInfoMap &infoMap);

  // Returns the op that the root node of the source pattern matches.
  const Operator &getSourceRootOp();

  // Returns the operator wrapper object corresponding to the given `node`'s DAG
  // operator.
  Operator &getDialectOp(DagNode node);

  // Returns the constraints.
  std::vector<AppliedConstraint> getConstraints() const;

  // Returns the number of supplemental auxiliary patterns generated by applying
  // this rewrite rule.
  int getNumSupplementalPatterns() const;

  // Returns the DAG tree root node of the `index`-th supplemental result
  // pattern.
  DagNode getSupplementalPattern(unsigned index) const;

  // Returns the benefit score of the pattern.
  int getBenefit() const;

  using IdentifierLine = std::pair<StringRef, unsigned>;

  // Returns the file location of the pattern (buffer identifier + line number
  // pair).
  std::vector<IdentifierLine> getLocation() const;

  // Recursively collects all bound symbols inside the DAG tree rooted
  // at `tree` and updates the given `infoMap`.
  void collectBoundSymbols(DagNode tree, SymbolInfoMap &infoMap,
                           bool isSrcPattern);

private:
  // Helper function to verify variable binding.
  void verifyBind(bool result, StringRef symbolName);

  // The TableGen definition of this pattern.
  const llvm::Record &def;

  // All operators.
  // TODO: we need a proper context manager, like MLIRContext, for managing the
  // lifetime of shared entities.
  RecordOperatorMap *recordOpMap;
};

} // namespace tblgen
} // namespace mlir

namespace llvm {
template <>
struct DenseMapInfo<mlir::tblgen::DagNode> {
  static mlir::tblgen::DagNode getEmptyKey() {
    return mlir::tblgen::DagNode(
        llvm::DenseMapInfo<llvm::DagInit *>::getEmptyKey());
  }
  static mlir::tblgen::DagNode getTombstoneKey() {
    return mlir::tblgen::DagNode(
        llvm::DenseMapInfo<llvm::DagInit *>::getTombstoneKey());
  }
  static unsigned getHashValue(mlir::tblgen::DagNode node) {
    return llvm::hash_value(node.getAsOpaquePointer());
  }
  static bool isEqual(mlir::tblgen::DagNode lhs, mlir::tblgen::DagNode rhs) {
    return lhs.node == rhs.node;
  }
};

template <>
struct DenseMapInfo<mlir::tblgen::DagLeaf> {
  static mlir::tblgen::DagLeaf getEmptyKey() {
    return mlir::tblgen::DagLeaf(
        llvm::DenseMapInfo<llvm::Init *>::getEmptyKey());
  }
  static mlir::tblgen::DagLeaf getTombstoneKey() {
    return mlir::tblgen::DagLeaf(
        llvm::DenseMapInfo<llvm::Init *>::getTombstoneKey());
  }
  static unsigned getHashValue(mlir::tblgen::DagLeaf leaf) {
    return llvm::hash_value(leaf.getAsOpaquePointer());
  }
  static bool isEqual(mlir::tblgen::DagLeaf lhs, mlir::tblgen::DagLeaf rhs) {
    return lhs.def == rhs.def;
  }
};
} // namespace llvm

#endif // MLIR_TABLEGEN_PATTERN_H_
