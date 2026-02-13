//===- ControlFlowInterfaces.h - ControlFlow Interfaces ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the branch interfaces defined in
// `ControlFlowInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CONTROLFLOWINTERFACES_H
#define MLIR_INTERFACES_CONTROLFLOWINTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class BranchOpInterface;
class RegionBranchOpInterface;
class RegionBranchTerminatorOpInterface;

/// This class models how operands are forwarded to block arguments in control
/// flow. It consists of a number, denoting how many of the successors block
/// arguments are produced by the operation, followed by a range of operands
/// that are forwarded. The produced operands are passed to the first few
/// block arguments of the successor, followed by the forwarded operands.
/// It is unsupported to pass them in a different order.
///
/// An example operation with both of these concepts would be a branch-on-error
/// operation, that internally produces an error object on the error path:
///
///   invoke %function(%0)
///     label ^success ^error(%1 : i32)
///
///     ^error(%e: !error, %arg0 : i32):
///       ...
///
/// This operation would return an instance of SuccessorOperands with a produced
/// operand count of 1 (mapped to %e in the successor) and a forwarded
/// operands range consisting of %1 in the example above (mapped to %arg0 in the
/// successor).
class SuccessorOperands {
public:
  /// Constructs a SuccessorOperands with no produced operands that simply
  /// forwards operands to the successor.
  explicit SuccessorOperands(MutableOperandRange forwardedOperands);

  /// Constructs a SuccessorOperands with the given amount of produced operands
  /// and forwarded operands.
  SuccessorOperands(unsigned producedOperandCount,
                    MutableOperandRange forwardedOperands);

  /// Returns the amount of operands passed to the successor. This consists both
  /// of produced operands by the operation as well as forwarded ones.
  unsigned size() const {
    return producedOperandCount + forwardedOperands.size();
  }

  /// Returns true if there are no successor operands.
  bool empty() const { return size() == 0; }

  /// Returns the amount of operands that are produced internally by the
  /// operation. These are passed to the first few block arguments.
  unsigned getProducedOperandCount() const { return producedOperandCount; }

  /// Returns true if the successor operand denoted by `index` is produced by
  /// the operation.
  bool isOperandProduced(unsigned index) const {
    return index < producedOperandCount;
  }

  /// Returns the Value that is passed to the successors block argument denoted
  /// by `index`. If it is produced by the operation, no such value exists and
  /// a null Value is returned.
  Value operator[](unsigned index) const {
    if (isOperandProduced(index))
      return Value();
    return forwardedOperands[index - producedOperandCount].get();
  }

  /// Get the range of operands that are simply forwarded to the successor.
  OperandRange getForwardedOperands() const { return forwardedOperands; }

  /// Get the range of operands that are simply forwarded to the successor.
  MutableOperandRange getMutableForwardedOperands() const {
    return forwardedOperands;
  }

  /// Get a slice of the operands forwarded to the successor. The given range
  /// must not contain any operands produced by the operation.
  MutableOperandRange slice(unsigned subStart, unsigned subLen) const {
    assert(!isOperandProduced(subStart) &&
           "can't slice operands produced by the operation");
    return forwardedOperands.slice(subStart - producedOperandCount, subLen);
  }

  /// Erase operands forwarded to the successor. The given range must
  /// not contain any operands produced by the operation.
  void erase(unsigned subStart, unsigned subLen = 1) {
    assert(!isOperandProduced(subStart) &&
           "can't erase operands produced by the operation");
    forwardedOperands.erase(subStart - producedOperandCount, subLen);
  }

  /// Add new operands that are forwarded to the successor.
  void append(ValueRange valueRange) { forwardedOperands.append(valueRange); }

  /// Gets the index of the forwarded operand within the operation which maps
  /// to the block argument denoted by `blockArgumentIndex`. The block argument
  /// must be mapped to a forwarded operand.
  unsigned getOperandIndex(unsigned blockArgumentIndex) const {
    assert(!isOperandProduced(blockArgumentIndex) &&
           "can't map operand produced by the operation");
    OperandRange operands = forwardedOperands;
    return operands.getBeginOperandIndex() +
           (blockArgumentIndex - producedOperandCount);
  }

private:
  /// Amount of operands that are produced internally within the operation and
  /// passed to the first few block arguments.
  unsigned producedOperandCount;
  /// Range of operands that are forwarded to the remaining block arguments.
  MutableOperandRange forwardedOperands;
};

//===----------------------------------------------------------------------===//
// BranchOpInterface
//===----------------------------------------------------------------------===//

namespace detail {
/// Return the `BlockArgument` corresponding to operand `operandIndex` in some
/// successor if `operandIndex` is within the range of `operands`, or
/// std::nullopt if `operandIndex` isn't a successor operand index.
std::optional<BlockArgument>
getBranchSuccessorArgument(const SuccessorOperands &operands,
                           unsigned operandIndex, Block *successor);

/// Verify that the given operands match those of the given successor block.
LogicalResult verifyBranchSuccessorOperands(Operation *op, unsigned succNo,
                                            const SuccessorOperands &operands);
} // namespace detail

//===----------------------------------------------------------------------===//
// WeightedBranchOpInterface
//===----------------------------------------------------------------------===//

namespace detail {
/// Verify that the branch weights attached to an operation
/// implementing WeightedBranchOpInterface are correct.
LogicalResult verifyBranchWeights(Operation *op);
} // namespace detail

//===----------------------------------------------------------------------===//
// WeightedRegiobBranchOpInterface
//===----------------------------------------------------------------------===//

namespace detail {
/// Verify that the region weights attached to an operation
/// implementing WeightedRegiobBranchOpInterface are correct.
LogicalResult verifyRegionBranchWeights(Operation *op);
} // namespace detail

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface
//===----------------------------------------------------------------------===//

namespace detail {
/// Verify that types match along control flow edges described the given op.
LogicalResult verifyRegionBranchOpInterface(Operation *op);
} //  namespace detail

/// A mapping from successor operands to successor inputs.
///
/// * A successor operand is an operand of a region branch op or region
///   branch terminator, that is forwarded to a successor input.
/// * A successor input is a block argument of a region or a result of the
///   region branch op, that is populated by a successor operand.
///
/// The mapping is 1:N. Each successor operand may be forwarded to multiple
/// successor inputs. (Because the control flow can dispatch to multiple
/// possible successors.) Operands that not forwarded at all are not present in
/// the mapping.
using RegionBranchSuccessorMapping = DenseMap<OpOperand *, SmallVector<Value>>;
using RegionBranchInverseSuccessorMapping =
    DenseMap<Value, SmallVector<OpOperand *>>;

/// This class represents a successor of a region. A region successor can either
/// be another region, or the parent operation (i.e., the operation that
/// implements the `RegionBranchOpInterface`). In the latter case, the control
/// flow branches after/out of the region branch operation.
class RegionSuccessor {
public:
  /// Initialize a successor that branches to a region of the parent operation.
  RegionSuccessor(Region *region) : successor(region) {
    assert(region && "Region must not be null");
  }

  /// Initialize a successor that branches after/out of the parent operation.
  static RegionSuccessor parent() { return RegionSuccessor(); }

  /// Return the given region successor. Returns nullptr if the successor is the
  /// parent operation.
  Region *getSuccessor() const { return successor; }

  /// Return true if the successor is the parent operation.
  bool isParent() const { return successor == nullptr; }

  bool operator==(RegionSuccessor rhs) const {
    return successor == rhs.successor;
  }

  bool operator==(const Region *region) const { return successor == region; }

  friend bool operator!=(RegionSuccessor lhs, RegionSuccessor rhs) {
    return !(lhs == rhs);
  }

private:
  /// Private constructor to encourage the use of `RegionSuccessor::parent`.
  RegionSuccessor() : successor(nullptr) {}

  Region *successor = nullptr;
};

/// This class represents a point being branched from in the methods of the
/// `RegionBranchOpInterface`.
/// One can branch from one of two kinds of places:
/// * The parent operation (aka the `RegionBranchOpInterface` implementation)
/// * A RegionBranchTerminatorOpInterface inside a region within the parent
//    operation.
class RegionBranchPoint {
public:
  /// Returns an instance of `RegionBranchPoint` representing the parent
  /// operation.
  static constexpr RegionBranchPoint parent() { return RegionBranchPoint(); }

  /// Creates a `RegionBranchPoint` that branches from the given terminator.
  inline RegionBranchPoint(RegionBranchTerminatorOpInterface predecessor);

  /// Explicitly stops users from constructing with `nullptr`.
  RegionBranchPoint(std::nullptr_t) = delete;

  /// Returns true if branching from the parent op.
  bool isParent() const { return predecessor == nullptr; }

  /// Returns the terminator if branching from a region.
  /// A "null" operation otherwise.
  inline RegionBranchTerminatorOpInterface
  getTerminatorPredecessorOrNull() const;

  /// Returns true if the two branch points are equal.
  friend bool operator==(RegionBranchPoint lhs, RegionBranchPoint rhs) {
    return lhs.predecessor == rhs.predecessor;
  }

private:
  // Private constructor to encourage the use of `RegionBranchPoint::parent`.
  constexpr RegionBranchPoint() = default;

  /// Internal encoding. Uses nullptr for representing branching from the parent
  /// op and the region terminator being branched from otherwise.
  Operation *predecessor = nullptr;
};

/// This class represents upper and lower bounds on the number of times a region
/// of a `RegionBranchOpInterface` can be invoked. The lower bound is at least
/// zero, but the upper bound may not be known.
class InvocationBounds {
public:
  /// Create invocation bounds. The lower bound must be at least 0 and only the
  /// upper bound can be unknown.
  InvocationBounds(unsigned lb, std::optional<unsigned> ub)
      : lower(lb), upper(ub) {
    assert((!ub || ub >= lb) && "upper bound cannot be less than lower bound");
  }

  /// Return the lower bound.
  unsigned getLowerBound() const { return lower; }

  /// Return the upper bound.
  std::optional<unsigned> getUpperBound() const { return upper; }

  /// Returns the unknown invocation bounds, i.e., there is no information on
  /// how many times a region may be invoked.
  static InvocationBounds getUnknown() { return {0, std::nullopt}; }

private:
  /// The minimum number of times the successor region will be invoked.
  unsigned lower;
  /// The maximum number of times the successor region will be invoked or
  /// `std::nullopt` if an upper bound is not known.
  std::optional<unsigned> upper;
};

/// Return `true` if `a` and `b` are in mutually exclusive regions as per
/// RegionBranchOpInterface.
bool insideMutuallyExclusiveRegions(Operation *a, Operation *b);

/// Return the first enclosing region of the given op that may be executed
/// repetitively as per RegionBranchOpInterface or `nullptr` if no such region
/// exists.
Region *getEnclosingRepetitiveRegion(Operation *op);

/// Return the first enclosing region of the given Value that may be executed
/// repetitively as per RegionBranchOpInterface or `nullptr` if no such region
/// exists.
Region *getEnclosingRepetitiveRegion(Value value);

/// Populate canonicalization patterns that simplify successor operands/inputs
/// of region branch operations. Only operations with the given name are
/// matched.
void populateRegionBranchOpInterfaceCanonicalizationPatterns(
    RewritePatternSet &patterns, StringRef opName, PatternBenefit benefit = 1);

/// Helper function for the region branch op inlining pattern that builds
/// replacement values for non-successor-input values.
using NonSuccessorInputReplacementBuilderFn =
    std::function<Value(OpBuilder &, Location, Value)>;
/// Helper function for the region branch op inlining pattern that checks if the
/// pattern is applicable to the given operation.
using PatternMatcherFn = std::function<LogicalResult(Operation *)>;

namespace detail {
/// Default implementation of the non-successor-input replacement builder
/// function. This default implemention assumes that all block arguments and
/// op results are successor inputs.
static inline Value defaultReplBuilderFn(OpBuilder &builder, Location loc,
                                         Value value) {
  llvm_unreachable("defaultReplBuilderFn not implemented");
}

/// Default implementation of the pattern matcher function.
static inline LogicalResult defaultMatcherFn(Operation *op) {
  return success();
}
} // namespace detail

/// Populate a pattern that inlines the body of region branch ops when there is
/// a single acyclic path through the region branch op, starting from "parent"
/// and ending at "parent". For details, refer to the documentation of the
/// pattern.
///
/// `replBuilderFn` is a function that builds replacement values for
/// non-successor-input values of the region branch op. `matcherFn` is a
/// function that checks if the pattern is applicable to the given operation.
/// Both functions are optional.
void populateRegionBranchOpInterfaceInliningPattern(
    RewritePatternSet &patterns, StringRef opName,
    NonSuccessorInputReplacementBuilderFn replBuilderFn =
        detail::defaultReplBuilderFn,
    PatternMatcherFn matcherFn = detail::defaultMatcherFn,
    PatternBenefit benefit = 1);

//===----------------------------------------------------------------------===//
// ControlFlow Traits
//===----------------------------------------------------------------------===//

namespace OpTrait {
/// This trait indicates that a terminator operation is "return-like". This
/// means that it exits its current region and forwards its operands as "exit"
/// values to the parent region. Operations with this trait are not permitted to
/// contain successors or produce results.
template <typename ConcreteType>
struct ReturnLike : public TraitBase<ConcreteType, ReturnLike> {
  static LogicalResult verifyTrait(Operation *op) {
    static_assert(ConcreteType::template hasTrait<IsTerminator>(),
                  "expected operation to be a terminator");
    static_assert(ConcreteType::template hasTrait<ZeroResults>(),
                  "expected operation to have zero results");
    static_assert(ConcreteType::template hasTrait<ZeroSuccessors>(),
                  "expected operation to have zero successors");
    return success();
  }
};
} // namespace OpTrait

} // namespace mlir

//===----------------------------------------------------------------------===//
// ControlFlow Interfaces
//===----------------------------------------------------------------------===//

/// Include the generated interface declarations.
#include "mlir/Interfaces/ControlFlowInterfaces.h.inc"

namespace mlir {
inline RegionBranchPoint::RegionBranchPoint(
    RegionBranchTerminatorOpInterface predecessor)
    : predecessor(predecessor.getOperation()) {}

inline RegionBranchTerminatorOpInterface
RegionBranchPoint::getTerminatorPredecessorOrNull() const {
  if (!predecessor)
    return nullptr;
  return cast<RegionBranchTerminatorOpInterface>(predecessor);
}

inline bool operator!=(RegionBranchPoint lhs, RegionBranchPoint rhs) {
  return !(lhs == rhs);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     RegionBranchPoint point) {
  if (point.isParent())
    return os << "<from parent>";
  return os << "<region #"
            << point.getTerminatorPredecessorOrNull()
                   ->getParentRegion()
                   ->getRegionNumber()
            << ", terminator "
            << OpWithFlags(point.getTerminatorPredecessorOrNull(),
                           OpPrintingFlags().skipRegions())
            << ">";
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     RegionSuccessor successor) {
  if (successor.isParent())
    return os << "<to parent>";
  return os << "<to region #" << successor.getSuccessor()->getRegionNumber()
            << ">";
}
} // namespace mlir

#endif // MLIR_INTERFACES_CONTROLFLOWINTERFACES_H
