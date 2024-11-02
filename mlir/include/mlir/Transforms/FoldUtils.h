//===- FoldUtils.h - Operation Fold Utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares various operation folding utilities. These
// utilities are intended to be used by passes to unify and simply their logic.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_FOLDUTILS_H
#define MLIR_TRANSFORMS_FOLDUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Interfaces/FoldInterfaces.h"

namespace mlir {
class Operation;
class Value;

//===--------------------------------------------------------------------===//
// OperationFolder
//===--------------------------------------------------------------------===//

/// A utility class for folding operations, and unifying duplicated constants
/// generated along the way.
class OperationFolder {
public:
  OperationFolder(MLIRContext *ctx) : interfaces(ctx) {}

  /// Tries to perform folding on the given `op`, including unifying
  /// deduplicated constants. If successful, replaces `op`'s uses with
  /// folded results, and returns success. `preReplaceAction` is invoked on `op`
  /// before it is replaced. 'processGeneratedConstants' is invoked for any new
  /// operations generated when folding. If the op was completely folded it is
  /// erased. If it is just updated in place, `inPlaceUpdate` is set to true.
  LogicalResult
  tryToFold(Operation *op,
            function_ref<void(Operation *)> processGeneratedConstants = nullptr,
            function_ref<void(Operation *)> preReplaceAction = nullptr,
            bool *inPlaceUpdate = nullptr);

  /// Tries to fold a pre-existing constant operation. `constValue` represents
  /// the value of the constant, and can be optionally passed if the value is
  /// already known (e.g. if the constant was discovered by m_Constant). This is
  /// purely an optimization opportunity for callers that already know the value
  /// of the constant. Returns false if an existing constant for `op` already
  /// exists in the folder, in which case `op` is replaced and erased.
  /// Otherwise, returns true and `op` is inserted into the folder (and
  /// hoisted if necessary).
  bool insertKnownConstant(Operation *op, Attribute constValue = {});

  /// Notifies that the given constant `op` should be remove from this
  /// OperationFolder's internal bookkeeping.
  ///
  /// Note: this method must be called if a constant op is to be deleted
  /// externally to this OperationFolder. `op` must be a constant op.
  void notifyRemoval(Operation *op);

  /// Create an operation of specific op type with the given builder,
  /// and immediately try to fold it. This function populates 'results' with
  /// the results after folding the operation.
  template <typename OpTy, typename... Args>
  void create(OpBuilder &builder, SmallVectorImpl<Value> &results,
              Location location, Args &&... args) {
    // The op needs to be inserted only if the fold (below) fails, or the number
    // of results produced by the successful folding is zero (which is treated
    // as an in-place fold). Using create methods of the builder will insert the
    // op, so not using it here.
    OperationState state(location, OpTy::getOperationName());
    OpTy::build(builder, state, std::forward<Args>(args)...);
    Operation *op = Operation::create(state);

    if (failed(tryToFold(builder, op, results)) || results.empty()) {
      builder.insert(op);
      results.assign(op->result_begin(), op->result_end());
      return;
    }
    op->destroy();
  }

  /// Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  typename std::enable_if<OpTy::template hasTrait<OpTrait::OneResult>(),
                          Value>::type
  create(OpBuilder &builder, Location location, Args &&... args) {
    SmallVector<Value, 1> results;
    create<OpTy>(builder, results, location, std::forward<Args>(args)...);
    return results.front();
  }

  /// Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  typename std::enable_if<OpTy::template hasTrait<OpTrait::ZeroResults>(),
                          OpTy>::type
  create(OpBuilder &builder, Location location, Args &&... args) {
    auto op = builder.create<OpTy>(location, std::forward<Args>(args)...);
    SmallVector<Value, 0> unused;
    (void)tryToFold(op.getOperation(), unused);

    // Folding cannot remove a zero-result operation, so for convenience we
    // continue to return it.
    return op;
  }

  /// Clear out any constants cached inside of the folder.
  void clear();

  /// Get or create a constant using the given builder. On success this returns
  /// the constant operation, nullptr otherwise.
  Value getOrCreateConstant(OpBuilder &builder, Dialect *dialect,
                            Attribute value, Type type, Location loc);

private:
  /// This map keeps track of uniqued constants by dialect, attribute, and type.
  /// A constant operation materializes an attribute with a type. Dialects may
  /// generate different constants with the same input attribute and type, so we
  /// also need to track per-dialect.
  using ConstantMap =
      DenseMap<std::tuple<Dialect *, Attribute, Type>, Operation *>;

  /// Returns true if the given operation is an already folded constant that is
  /// owned by this folder.
  bool isFolderOwnedConstant(Operation *op) const;

  /// Tries to perform folding on the given `op`. If successful, populates
  /// `results` with the results of the folding.
  LogicalResult tryToFold(
      OpBuilder &builder, Operation *op, SmallVectorImpl<Value> &results,
      function_ref<void(Operation *)> processGeneratedConstants = nullptr);

  /// Try to process a set of fold results, generating constants as necessary.
  /// Populates `results` on success, otherwise leaves it unchanged.
  LogicalResult
  processFoldResults(OpBuilder &builder, Operation *op,
                     SmallVectorImpl<Value> &results,
                     ArrayRef<OpFoldResult> foldResults,
                     function_ref<void(Operation *)> processGeneratedConstants);

  /// Try to get or create a new constant entry. On success this returns the
  /// constant operation, nullptr otherwise.
  Operation *tryGetOrCreateConstant(ConstantMap &uniquedConstants,
                                    Dialect *dialect, OpBuilder &builder,
                                    Attribute value, Type type, Location loc);

  /// A mapping between an insertion region and the constants that have been
  /// created within it.
  DenseMap<Region *, ConstantMap> foldScopes;

  /// This map tracks all of the dialects that an operation is referenced by;
  /// given that many dialects may generate the same constant.
  DenseMap<Operation *, SmallVector<Dialect *, 2>> referencedDialects;

  /// A collection of dialect folder interfaces.
  DialectInterfaceCollection<DialectFoldInterface> interfaces;
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_FOLDUTILS_H
