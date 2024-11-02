//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating C++ from tablegen
// structures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CODEGENHELPERS_H
#define MLIR_TABLEGEN_CODEGENHELPERS_H

#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/Format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {
class Constraint;
class DagLeaf;

// Format into a std::string
template <typename... Parameters>
std::string strfmt(const char *fmt, Parameters &&...parameters) {
  return llvm::formatv(fmt, std::forward<Parameters>(parameters)...).str();
}

// Simple RAII helper for defining ifdef-undef-endif scopes.
class IfDefScope {
public:
  IfDefScope(llvm::StringRef name, llvm::raw_ostream &os)
      : name(name.str()), os(os) {
    os << "#ifdef " << name << "\n"
       << "#undef " << name << "\n\n";
  }
  ~IfDefScope() { os << "\n#endif  // " << name << "\n\n"; }

private:
  std::string name;
  llvm::raw_ostream &os;
};

// A helper RAII class to emit nested namespaces for this op.
class NamespaceEmitter {
public:
  NamespaceEmitter(raw_ostream &os, const Dialect &dialect) : os(os) {
    if (!dialect)
      return;
    emitNamespaceStarts(os, dialect.getCppNamespace());
  }
  NamespaceEmitter(raw_ostream &os, StringRef cppNamespace) : os(os) {
    emitNamespaceStarts(os, cppNamespace);
  }

  ~NamespaceEmitter() {
    for (StringRef ns : llvm::reverse(namespaces))
      os << "} // namespace " << ns << "\n";
  }

private:
  void emitNamespaceStarts(raw_ostream &os, StringRef cppNamespace) {
    llvm::SplitString(cppNamespace, namespaces, "::");
    for (StringRef ns : namespaces)
      os << "namespace " << ns << " {\n";
  }
  raw_ostream &os;
  SmallVector<StringRef, 2> namespaces;
};

/// This class deduplicates shared operation verification code by emitting
/// static functions alongside the op definitions. These methods are local to
/// the definition file, and are invoked within the operation verify methods.
/// An example is shown below:
///
/// static LogicalResult localVerify(...)
///
/// LogicalResult OpA::verify(...) {
///  if (failed(localVerify(...)))
///    return failure();
///  ...
/// }
///
/// LogicalResult OpB::verify(...) {
///  if (failed(localVerify(...)))
///    return failure();
///  ...
/// }
///
class StaticVerifierFunctionEmitter {
public:
  StaticVerifierFunctionEmitter(raw_ostream &os,
                                const llvm::RecordKeeper &records);

  /// Collect and unique all compatible type, attribute, successor, and region
  /// constraints from the operations in the file and emit them at the top of
  /// the generated file.
  ///
  /// Constraints that do not meet the restriction that they can only reference
  /// `$_self` and `$_op` are not uniqued.
  void emitOpConstraints(ArrayRef<llvm::Record *> opDefs, bool emitDecl);

  /// Unique all compatible type and attribute constraints from a pattern file
  /// and emit them at the top of the generated file.
  ///
  /// Constraints that do not meet the restriction that they can only reference
  /// `$_self`, `$_op`, and `$_builder` are not uniqued.
  void emitPatternConstraints(const ArrayRef<DagLeaf> constraints);

  /// Get the name of the static function used for the given type constraint.
  /// These functions are used for operand and result constraints and have the
  /// form:
  ///
  ///   LogicalResult(Operation *op, Type type, StringRef valueKind,
  ///                 unsigned valueIndex);
  ///
  /// Pattern constraints have the form:
  ///
  ///   LogicalResult(PatternRewriter &rewriter, Operation *op, Type type,
  ///                 StringRef failureStr);
  ///
  StringRef getTypeConstraintFn(const Constraint &constraint) const;

  /// Get the name of the static function used for the given attribute
  /// constraint. These functions are in the form:
  ///
  ///   LogicalResult(Operation *op, Attribute attr, StringRef attrName);
  ///
  /// If a uniqued constraint was not found, this function returns None. The
  /// uniqued constraints cannot be used in the context of an OpAdaptor.
  ///
  /// Pattern constraints have the form:
  ///
  ///   LogicalResult(PatternRewriter &rewriter, Operation *op, Attribute attr,
  ///                 StringRef failureStr);
  ///
  Optional<StringRef> getAttrConstraintFn(const Constraint &constraint) const;

  /// Get the name of the static function used for the given successor
  /// constraint. These functions are in the form:
  ///
  ///   LogicalResult(Operation *op, Block *successor, StringRef successorName,
  ///                 unsigned successorIndex);
  ///
  StringRef getSuccessorConstraintFn(const Constraint &constraint) const;

  /// Get the name of the static function used for the given region constraint.
  /// These functions are in the form:
  ///
  ///   LogicalResult(Operation *op, Region &region, StringRef regionName,
  ///                 unsigned regionIndex);
  ///
  /// The region name may be empty.
  StringRef getRegionConstraintFn(const Constraint &constraint) const;

private:
  /// Emit static type constraint functions.
  void emitTypeConstraints();
  /// Emit static attribute constraint functions.
  void emitAttrConstraints();
  /// Emit static successor constraint functions.
  void emitSuccessorConstraints();
  /// Emit static region constraint functions.
  void emitRegionConstraints();

  /// Emit pattern constraints.
  void emitPatternConstraints();

  /// Collect and unique all the constraints used by operations.
  void collectOpConstraints(ArrayRef<llvm::Record *> opDefs);
  /// Collect and unique all pattern constraints.
  void collectPatternConstraints(ArrayRef<DagLeaf> constraints);

  /// The output stream.
  raw_ostream &os;

  /// A unique label for the file currently being generated. This is used to
  /// ensure that the static functions have a unique name.
  std::string uniqueOutputLabel;

  /// Use a MapVector to ensure that functions are generated deterministically.
  using ConstraintMap = llvm::MapVector<Constraint, std::string,
                                        llvm::DenseMap<Constraint, unsigned>>;

  /// A generic function to emit constraints
  void emitConstraints(const ConstraintMap &constraints, StringRef selfName,
                       const char *codeTemplate);

  /// Assign a unique name to a unique constraint.
  std::string getUniqueName(StringRef kind, unsigned index);
  /// Unique a constraint in the map.
  void collectConstraint(ConstraintMap &map, StringRef kind,
                         Constraint constraint);

  /// The set of type constraints used for operand and result verification in
  /// the current file.
  ConstraintMap typeConstraints;
  /// The set of attribute constraints used in the current file.
  ConstraintMap attrConstraints;
  /// The set of successor constraints used in the current file.
  ConstraintMap successorConstraints;
  /// The set of region constraints used in the current file.
  ConstraintMap regionConstraints;
};

/// Escape a string using C++ encoding. E.g. foo"bar -> foo\x22bar.
std::string escapeString(StringRef value);

namespace detail {
template <typename> struct stringifier {
  template <typename T> static std::string apply(T &&t) {
    return std::string(std::forward<T>(t));
  }
};
template <> struct stringifier<Twine> {
  static std::string apply(const Twine &twine) {
    return twine.str();
  }
};
template <typename OptionalT>
struct stringifier<Optional<OptionalT>> {
  static std::string apply(Optional<OptionalT> optional) {
    return optional.hasValue() ? stringifier<OptionalT>::apply(*optional)
                               : std::string();
  }
};
} // namespace detail

/// Generically convert a value to a std::string.
template <typename T> std::string stringify(T &&t) {
  return detail::stringifier<std::remove_reference_t<std::remove_const_t<T>>>::
      apply(std::forward<T>(t));
}

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CODEGENHELPERS_H
