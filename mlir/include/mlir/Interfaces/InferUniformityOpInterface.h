//===- InferUniformityOpInterface.h - Uniformity ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the uniformity inference interface
// defined in `InferUniformityOpInterface.td`, and the lattice value it works
// with.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERUNIFORMITYOPINTERFACE_H
#define MLIR_INTERFACES_INFERUNIFORMITYOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace mlir {

/// The widest group of threads of a SIMT launch within which an SSA value is
/// known to be the same for every thread, from the narrowest to the widest.
///
/// The groups nest: a thread belongs to a subgroup (warp, wave), a subgroup to
/// a workgroup (thread block), a workgroup to a cluster, and every cluster to
/// the launch. A value that is the same within a group is also the same within
/// every narrower group, so the scopes are totally ordered and the meet of two
/// scopes is the narrower one.
enum class UniformityScope : uint8_t {
  /// Two threads of the same subgroup may observe different values.
  Divergent = 0,
  /// The same for every thread of a subgroup.
  Subgroup,
  /// The same for every thread of a workgroup.
  Workgroup,
  /// The same for every thread of a cluster of workgroups.
  Cluster,
  /// The same for every thread of the launch.
  Uniform,
};

/// Returns the narrower of two scopes.
inline UniformityScope meet(UniformityScope lhs, UniformityScope rhs) {
  return lhs < rhs ? lhs : rhs;
}

/// Returns the name of a scope as written in IR and diagnostics.
StringRef stringifyUniformityScope(UniformityScope scope);

/// Returns the scope with the given name, or nullopt if there is none.
std::optional<UniformityScope> symbolizeUniformityScope(StringRef name);

raw_ostream &operator<<(raw_ostream &os, UniformityScope scope);

/// The uniformity lattice value of an SSA value: a scope, or uninitialized
/// while the analysis has not reached the value yet.
class Uniformity {
public:
  Uniformity() = default;
  Uniformity(UniformityScope scope) : scope(scope) {}

  static Uniformity getUniform() {
    return Uniformity(UniformityScope::Uniform);
  }
  static Uniformity getDivergent() {
    return Uniformity(UniformityScope::Divergent);
  }

  bool isUninitialized() const { return !scope.has_value(); }

  UniformityScope getScope() const {
    assert(!isUninitialized() && "querying an uninitialized uniformity");
    return *scope;
  }

  bool operator==(const Uniformity &rhs) const { return scope == rhs.scope; }
  bool operator!=(const Uniformity &rhs) const { return !(*this == rhs); }

  /// The join of two lattice values: uninitialized is the neutral element,
  /// otherwise the narrower scope. The pessimistic fixpoint is `Divergent`.
  static Uniformity join(const Uniformity &lhs, const Uniformity &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    return Uniformity(meet(*lhs.scope, *rhs.scope));
  }

  /// The join of a list of values; uninitialized if the list is empty or all
  /// of its elements are uninitialized.
  static Uniformity join(ArrayRef<Uniformity> values) {
    Uniformity result;
    for (const Uniformity &value : values)
      result = join(result, value);
    return result;
  }

  void print(raw_ostream &os) const;

private:
  std::optional<UniformityScope> scope;
};

inline raw_ostream &operator<<(raw_ostream &os, const Uniformity &uniformity) {
  uniformity.print(os);
  return os;
}

/// The callback through which an operation implementing
/// `InferUniformityOpInterface` reports the uniformity of a value it defines.
using SetUniformityFn = llvm::function_ref<void(Value, UniformityScope)>;

} // namespace mlir

#include "mlir/Interfaces/InferUniformityOpInterface.h.inc"

#endif // MLIR_INTERFACES_INFERUNIFORMITYOPINTERFACE_H
