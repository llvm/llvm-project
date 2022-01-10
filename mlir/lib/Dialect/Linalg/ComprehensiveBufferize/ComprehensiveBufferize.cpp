//===- ComprehensiveBufferize.cpp - Single pass bufferization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Comprehensive Bufferize bufferizes function bodies. Function boundaries
// (FuncOp bbArgs, CallOps, ReturnOps) are treated as "unknown" ops.
// ModuleBufferization.cpp is an extension of Comprehensive Bufferize for simple
// call graphs.
//
// Comprehensive Bufferize consists of two phases.
//
// 1. Analyze ops to decide which OpResults can bufferize inplace, i.e., without
//    inserting buffer copies. The analysis queries op bufferization semantics
//    via `BufferizableOpInterface`.
// 2. Bufferize ops by calling `BufferizableOpInterface::bufferize`. This
//    function does not generate buffer copies for OpResults that were decided
//    to bufferize inplace during the analysis phase.
//
// Inplace bufferization decisions are passed from the analysis to the
// bufferization phase via `BufferizationState` and `BufferizationAliasInfo`.
// They can be printed for debugging purposes with `testAnalysisOnly`.
//
// Ops that do not implement `BufferizableOpInterface` can be analyzed but are
// treated conservatively. E.g., the analysis has to assume that their
// OpOperands bufferize to memory writes. While such ops can be analyzed, they
// are not bufferized and remain in the IR. to_tensor and to_memref ops are
// inserted at the bufferization boundary.
//
// Note: If `allowUnknownOps` is set to false, bufferization fails when an
// unknown op (that does not implement `BufferizableOpInterface`) is found. No
// to_tensor/to_memref ops are inserted.
//
// This pass caters to high-performance codegen where buffer reuse is deemed
// critical: the pass should fail if the bufferized form of the function needs
// to return any buffer, unless `allowReturnMemref` is enabled.
//
//  Lastly, note that layout map chosen to bufferize is the most dynamic
//  canonical strided layout of the proper rank. This ensures compatibility with
//  expected layouts after transformations. Combinations of memref.cast +
//  canonicalization are responsible for clean ups.

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"

#include <random>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace linalg;
using namespace tensor;
using namespace comprehensive_bufferize;

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

//===----------------------------------------------------------------------===//
// Bufferization-specific attribute manipulation.
// These are for testing and debugging only. Bufferization information is
// stored in BufferizationAliasInfo. When run with `testAnalysisOnly`, the IR
// is annotated with the results of the analysis (copied from
// BufferizationAliasInfo), so that they can be checked in tests.
//===----------------------------------------------------------------------===//

/// Attribute marker to specify op results that can be bufferized inPlace.
constexpr StringLiteral kInPlaceResultsAttrName = "__inplace_operands_attr__";

/// Mark whether OpOperand will be bufferized inplace.
static void setInPlaceOpOperand(OpOperand &opOperand, bool inPlace) {
  Operation *op = opOperand.getOwner();
  auto attr =
      op->getAttr(kInPlaceResultsAttrName).dyn_cast_or_null<ArrayAttr>();
  SmallVector<StringRef> inPlaceVector;
  if (attr) {
    inPlaceVector = SmallVector<StringRef>(
        llvm::to_vector<4>(attr.getAsValueRange<StringAttr>()));
  } else {
    inPlaceVector = SmallVector<StringRef>(op->getNumOperands(), "none");
    for (OpOperand &opOperand : op->getOpOperands())
      if (opOperand.get().getType().isa<TensorType>())
        inPlaceVector[opOperand.getOperandNumber()] = "false";
  }

  inPlaceVector[opOperand.getOperandNumber()] = inPlace ? "true" : "false";
  op->setAttr(kInPlaceResultsAttrName,
              OpBuilder(op).getStrArrayAttr(inPlaceVector));
}

//===----------------------------------------------------------------------===//
// Bufferization-specific alias analysis.
//===----------------------------------------------------------------------===//

/// Return true if opOperand has been decided to bufferize in-place.
static bool isInplaceMemoryWrite(OpOperand &opOperand,
                                 const BufferizationAliasInfo &aliasInfo,
                                 BufferizationState &state) {
  // OpOperands that do not bufferize to a memory write do not write in-place.
  if (!state.bufferizesToMemoryWrite(opOperand))
    return false;
  // Check current bufferization decisions.
  return aliasInfo.isInPlace(opOperand);
}

/// Return true if, under current bufferization decisions, the buffer of `value`
/// is not writable.
static bool aliasesNonWritableBuffer(Value value,
                                     const BufferizationAliasInfo &aliasInfo,
                                     BufferizationState &state) {
  bool foundNonWritableBuffer = false;
  aliasInfo.applyOnAliases(value, [&](Value v) {
    // Query BufferizableOpInterface to see if the value is writable.
    // TODO: Out-of-place bufferized value could be considered writable.
    if (auto bufferizableOp = state.getOptions().dynCastBufferizableOp(v))
      if (bufferizableOp && bufferizableOp.isWritable(v, state))
        return;

    // Query BufferizableOpInterface to see if the BlockArgument is writable.
    if (auto bbArg = v.dyn_cast<BlockArgument>())
      if (auto bufferizableOp = state.getOptions().dynCastBufferizableOp(
              bbArg.getOwner()->getParentOp()))
        if (bufferizableOp.isWritable(bbArg, state))
          return;

    foundNonWritableBuffer = true;
  });

  return foundNonWritableBuffer;
}

/// Return true if the buffer to which `operand` would bufferize is equivalent
/// to some buffer write.
static bool aliasesInPlaceWrite(Value value,
                                const BufferizationAliasInfo &aliasInfo,
                                BufferizationState &state) {
  bool foundInplaceWrite = false;
  aliasInfo.applyOnAliases(value, [&](Value v) {
    for (auto &use : v.getUses()) {
      if (isInplaceMemoryWrite(use, aliasInfo, state)) {
        foundInplaceWrite = true;
        return;
      }
    }
  });
  return foundInplaceWrite;
}

/// Return true if `a` happens before `b`, i.e., `a` or one of its ancestors
/// properly dominates `b` and `b` is not inside `a`.
static bool happensBefore(Operation *a, Operation *b,
                          const DominanceInfo &domInfo) {
  do {
    // TODO: Instead of isProperAncestor + properlyDominates, we should use
    // properlyDominatesImpl(a, b, /*enclosingOpOk=*/false)
    if (a->isProperAncestor(b))
      return false;
    if (domInfo.properlyDominates(a, b))
      return true;
  } while ((a = a->getParentOp()));
  return false;
}

/// Annotate IR with details about the detected RaW conflict.
static void annotateConflict(OpOperand *uRead, OpOperand *uConflictingWrite,
                             Value lastWrite) {
  static uint64_t counter = 0;
  Operation *readingOp = uRead->getOwner();
  Operation *conflictingWritingOp = uConflictingWrite->getOwner();

  OpBuilder b(conflictingWritingOp->getContext());
  std::string id = "C_" + std::to_string(counter++);

  std::string conflictingWriteAttr =
      id +
      "[CONFL-WRITE: " + std::to_string(uConflictingWrite->getOperandNumber()) +
      "]";
  conflictingWritingOp->setAttr(conflictingWriteAttr, b.getUnitAttr());

  std::string readAttr =
      id + "[READ: " + std::to_string(uRead->getOperandNumber()) + "]";
  readingOp->setAttr(readAttr, b.getUnitAttr());

  if (auto opResult = lastWrite.dyn_cast<OpResult>()) {
    std::string lastWriteAttr = id + "[LAST-WRITE: result " +
                                std::to_string(opResult.getResultNumber()) +
                                "]";
    opResult.getDefiningOp()->setAttr(lastWriteAttr, b.getUnitAttr());
  } else {
    auto bbArg = lastWrite.cast<BlockArgument>();
    std::string lastWriteAttr =
        id + "[LAST-WRITE: bbArg " + std::to_string(bbArg.getArgNumber()) + "]";
    bbArg.getOwner()->getParentOp()->setAttr(lastWriteAttr, b.getUnitAttr());
  }
}

/// Given sets of uses and writes, return true if there is a RaW conflict under
/// the assumption that all given reads/writes alias the same buffer and that
/// all given writes bufferize inplace.
///
/// A conflict is: According to SSA use-def chains, a read R is supposed to read
/// the result of a write W1. But because of bufferization decisions, R actually
/// reads another write W2.
static bool hasReadAfterWriteInterference(
    const DenseSet<OpOperand *> &usesRead,
    const DenseSet<OpOperand *> &usesWrite, const DominanceInfo &domInfo,
    BufferizationState &state, const BufferizationAliasInfo &aliasInfo) {
  const BufferizationOptions &options = state.getOptions();

  for (OpOperand *uRead : usesRead) {
    Operation *readingOp = uRead->getOwner();

    // Find most recent write of uRead by following the SSA use-def chain. E.g.:
    //
    // %0 = "writing_op"(%t) : tensor<?x32> -> tensor<?xf32>
    // %1 = "aliasing_op"(%0) : tensor<?x32> -> tensor<?xf32>
    // %2 = "reading_op"(%1) : : tensor<?x32> -> not_a_tensor_type
    //
    // In the above example, if uRead is the OpOperand of reading_op, lastWrite
    // is %0. Note that operations that create an alias but do not write (such
    // as ExtractSliceOp) are skipped.
    Value lastWrite = state.findLastPrecedingWrite(uRead->get());

    // Look for conflicting memory writes. Potential conflicts are writes to an
    // alias that have been decided to bufferize inplace.
    for (OpOperand *uConflictingWrite : usesWrite) {
      // Throughout this loop, check for multiple requirements that have to be
      // met for uConflictingWrite to be an actual conflict.
      Operation *conflictingWritingOp = uConflictingWrite->getOwner();

      // No conflict if the readingOp dominates conflictingWritingOp, i.e., the
      // write is not visible when reading.
      if (happensBefore(readingOp, conflictingWritingOp, domInfo))
        continue;

      // No conflict if the reading use equals the use of the conflicting write.
      // A use cannot conflict with itself. Note: Just being the same op is not
      // enough. It has to be the same use.
      if (uConflictingWrite == uRead)
        continue;

      // No conflict if the op interface says so.
      if (auto bufferizableOp = options.dynCastBufferizableOp(readingOp))
        if (bufferizableOp.isNotConflicting(uRead, uConflictingWrite, state,
                                            aliasInfo))
          continue;

      if (conflictingWritingOp != readingOp)
        if (auto bufferizableOp =
                options.dynCastBufferizableOp(conflictingWritingOp))
          if (bufferizableOp.isNotConflicting(uRead, uConflictingWrite, state,
                                              aliasInfo))
            continue;

      // Ops are not conflicting if they are in mutually exclusive regions.
      if (insideMutuallyExclusiveRegions(readingOp, conflictingWritingOp))
        continue;

      // No conflict if the conflicting write happens before the last
      // write.
      if (Operation *writingOp = lastWrite.getDefiningOp()) {
        if (happensBefore(conflictingWritingOp, writingOp, domInfo))
          // conflictingWritingOp happens before writingOp. No conflict.
          continue;
        // No conflict if conflictingWritingOp is contained in writingOp.
        if (writingOp->isProperAncestor(conflictingWritingOp))
          continue;
      } else {
        auto bbArg = lastWrite.cast<BlockArgument>();
        Block *block = bbArg.getOwner();
        if (!block->findAncestorOpInBlock(*conflictingWritingOp))
          // conflictingWritingOp happens outside of the block. No
          // conflict.
          continue;
      }

      // No conflict if the conflicting write and the last write are the same
      // use.
      if (state.getAliasingOpResult(*uConflictingWrite) == lastWrite)
        continue;

      // All requirements are met. Conflict found!

      if (options.printConflicts)
        annotateConflict(uRead, uConflictingWrite, lastWrite);

      return true;
    }
  }

  return false;
}

/// Return true if bufferizing `operand` inplace would create a conflict. A read
/// R and a write W of the same alias set is a conflict if inplace bufferization
/// of W changes the value read by R to a value different from the one that
/// would be expected by tracing back R's origin through SSA use-def chains.
/// A conflict can only be introduced by a new alias and/or an inplace
/// bufferization decision.
///
/// Example:
/// %0 = tensor.extract_slice %t[...][...][1, 1] {inplace?}
/// %1 = vector.transfer_write %v1, %t {inplace} : vector<5xf32>, tensor<?xf32>
/// %e = tensor.extract_slice %1
/// %2 = vector.transfer_write %v2, %0 {inplace} : vector<6xf32>, tensor<?xf32>
/// %3 = vector.transfer_read %e, %cst : tensor<?xf32>, vector<7xf32>
///
/// In the above example, the two TransferWriteOps have already been decided to
/// bufferize inplace. Bufferizing the ExtractSliceOp inplace would create a
/// conflict because:
/// * According to SSA use-def chains, we expect to read the result of %1.
/// * However, adding an alias {%0, %t} would mean that the second
///   TransferWriteOp overwrites the first one. Therefore, the TransferReadOp
///   would no longer be reading the result of %1.
///
/// If `checkConsistencyOnly` is true, this function checks if there is a
/// read-after-write conflict without bufferizing `operand` inplace. This would
/// indicate a problem with the current inplace bufferization decisions.
///
/// Note: If `checkConsistencyOnly`, this function may be called with a null
/// OpResult. In that case, only the consistency of bufferization decisions
/// involving aliases of the given OpOperand are checked.
static bool wouldCreateReadAfterWriteInterference(
    OpOperand &operand, const DominanceInfo &domInfo, BufferizationState &state,
    const BufferizationAliasInfo &aliasInfo,
    bool checkConsistencyOnly = false) {
  // Helper function to iterate on aliases of `root` and capture the reads.
  auto getAliasingReads = [&](DenseSet<OpOperand *> &res, Value root) {
    aliasInfo.applyOnAliases(root, [&](Value alias) {
      for (auto &use : alias.getUses())
        // Read to a value that aliases root.
        if (state.bufferizesToMemoryRead(use))
          res.insert(&use);
    });
  };

  // Helper function to iterate on aliases of `root` and capture the writes.
  auto getAliasingInplaceWrites = [&](DenseSet<OpOperand *> &res, Value root) {
    aliasInfo.applyOnAliases(root, [&](Value alias) {
      for (auto &use : alias.getUses())
        // Inplace write to a value that aliases root.
        if (isInplaceMemoryWrite(use, aliasInfo, state))
          res.insert(&use);
    });
  };

  // Collect reads and writes of all aliases of OpOperand and OpResult.
  DenseSet<OpOperand *> usesRead, usesWrite;
  getAliasingReads(usesRead, operand.get());
  getAliasingInplaceWrites(usesWrite, operand.get());
  if (OpResult result = state.getAliasingOpResult(operand)) {
    getAliasingReads(usesRead, result);
    getAliasingInplaceWrites(usesWrite, result);
  }
  if (!checkConsistencyOnly && state.bufferizesToMemoryWrite(operand))
    usesWrite.insert(&operand);

  return hasReadAfterWriteInterference(usesRead, usesWrite, domInfo, state,
                                       aliasInfo);
}

/// Return true if bufferizing `opOperand` inplace would create a write to a
/// non-writable buffer.
static bool
wouldCreateWriteToNonWritableBuffer(OpOperand &opOperand,
                                    const BufferizationAliasInfo &aliasInfo,
                                    BufferizationState &state) {
  // Certain buffers are not writeable:
  //   1. A function bbArg that is not inplaceable or
  //   2. A constant op.
  bool nonWritable =
      aliasesNonWritableBuffer(opOperand.get(), aliasInfo, state);
  if (!nonWritable)
    return false;

  // This is a problem only if the buffer is written to via some alias.
  bool hasWrite = aliasesInPlaceWrite(opOperand.get(), aliasInfo, state) ||
                  state.bufferizesToMemoryWrite(opOperand);

  if (OpResult opResult = state.getAliasingOpResult(opOperand))
    hasWrite |= aliasesInPlaceWrite(opResult, aliasInfo, state);

  return hasWrite;
}

//===----------------------------------------------------------------------===//
// Bufferization analyses.
//===----------------------------------------------------------------------===//

/// Determine if `operand` can be bufferized in-place.
static LogicalResult bufferizableInPlaceAnalysisImpl(
    OpOperand &operand, BufferizationAliasInfo &aliasInfo,
    BufferizationState &state, const DominanceInfo &domInfo) {
  bool foundInterference =
      wouldCreateWriteToNonWritableBuffer(operand, aliasInfo, state) ||
      wouldCreateReadAfterWriteInterference(operand, domInfo, state, aliasInfo);

  if (foundInterference)
    aliasInfo.bufferizeOutOfPlace(operand);
  else
    aliasInfo.bufferizeInPlace(operand, state);

  return success();
}

/// Analyze the `ops` to determine which OpOperands are inplaceable. Walk ops in
/// reverse and bufferize ops greedily. This is a good starter heuristic.
///
/// Even if an op does not read or write, it may still create an alias when
/// bufferized in-place. An example of such ops is tensor.extract_slice.
///
/// Rationale for bufferizing `%1 = tensor.extract_slice %0[...]` inplace:
///
/// When bufferized out of place, an ExtractSliceOp lowers to alloc + copy. This
/// cannot change the flow of information for either the source or the
/// result buffers.
///
/// When bufferized inplace, an ExtractSliceOp does not by itself create any
/// read or write from memory. Instead, it has the effect of merging the alias
/// sets of the source and the result buffers.
///
/// An analysis is required to ensure inplace bufferization would not result in
/// RaW dependence violations.
static LogicalResult inPlaceAnalysis(SmallVector<Operation *> &ops,
                                     BufferizationAliasInfo &aliasInfo,
                                     BufferizationState &state,
                                     const DominanceInfo &domInfo,
                                     unsigned analysisFuzzerSeed = 0) {
  if (analysisFuzzerSeed) {
    // This is a fuzzer. For testing purposes only. Randomize the order in which
    // operations are analyzed. The bufferization quality is likely worse, but
    // we want to make sure that no assertions are triggered anywhere.
    std::mt19937 g(analysisFuzzerSeed);
    llvm::shuffle(ops.begin(), ops.end(), g);
  }

  // Walk ops in reverse for better interference analysis.
  for (Operation *op : reverse(ops))
    for (OpOperand &opOperand : op->getOpOperands())
      if (opOperand.get().getType().isa<TensorType>())
        if (auto bufferizableOp = state.getOptions().dynCastBufferizableOp(op))
          if (failed(bufferizableInPlaceAnalysisImpl(opOperand, aliasInfo,
                                                     state, domInfo)))
            return failure();

  return success();
}

/// Return true if the given op has a tensor result or a tensor operand.
static bool hasTensorSemantics(Operation *op) {
  bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
  bool hasTensorOperand = any_of(op->getOperandTypes(), isaTensor);
  return hasTensorResult || hasTensorOperand;
}

/// Analyze all ops that are contained in `op`.
static LogicalResult inPlaceAnalysis(Operation *op,
                                     BufferizationAliasInfo &aliasInfo,
                                     BufferizationState &state,
                                     const DominanceInfo &domInfo,
                                     unsigned analysisFuzzerSeed = 0) {
  // Collect ops so we can build our own reverse traversal.
  SmallVector<Operation *> ops;
  op->walk([&](Operation *op) {
    // No tensors => no buffers.
    if (!hasTensorSemantics(op))
      return;
    ops.push_back(op);
  });

  return inPlaceAnalysis(ops, aliasInfo, state, domInfo, analysisFuzzerSeed);
}

/// Analyze equivalence of tied OpResult/OpOperand pairs of the given ops.
static void equivalenceAnalysis(SmallVector<Operation *> &ops,
                                BufferizationAliasInfo &aliasInfo,
                                BufferizationState &state) {
  for (Operation *op : ops)
    if (auto bufferizableOp = state.getOptions().dynCastBufferizableOp(op))
      for (OpResult opResult : op->getOpResults())
        if (opResult.getType().isa<TensorType>())
          for (OpOperand *opOperand :
               bufferizableOp.getAliasingOpOperand(opResult, state))
            if (state.isInPlace(*opOperand))
              if (bufferizableOp.bufferRelation(opResult, aliasInfo, state) ==
                  BufferRelation::Equivalent)
                aliasInfo.unionEquivalenceClasses(opResult, opOperand->get());
}

/// Analyze equivalence of tied OpResult/OpOperand pairs of all ops contained
/// in `op`.
static void equivalenceAnalysis(Operation *op,
                                BufferizationAliasInfo &aliasInfo,
                                BufferizationState &state) {
  // Traverse ops in PostOrder: Nested ops first, then enclosing ops.
  SmallVector<Operation *> ops;
  op->walk<WalkOrder::PostOrder>([&](Operation *op) {
    // No tensors => no buffers.
    if (none_of(op->getResultTypes(), isaTensor))
      return;
    ops.push_back(op);
  });

  equivalenceAnalysis(ops, aliasInfo, state);
}

/// Assert that the current bufferization decisions are consistent.
static LogicalResult
checkAliasInfoConsistency(Operation *op, const DominanceInfo &domInfo,
                          BufferizationState &state,
                          const BufferizationAliasInfo &aliasInfo) {
  const BufferizationOptions &options = state.getOptions();
  Operation *inconsistentOp = nullptr;
  WalkResult walkResult = op->walk([&](Operation *op) {
    if (auto bufferizableOp = options.dynCastBufferizableOp(op))
      for (OpOperand &opOperand : op->getOpOperands())
        if (opOperand.get().getType().isa<TensorType>()) {
          if (wouldCreateReadAfterWriteInterference(
                  opOperand, domInfo, state, aliasInfo,
                  /*checkConsistencyOnly=*/true)) {
            // This error can happen if certain "mustBufferizeInPlace" interface
            // methods are implemented incorrectly, such that the IR already has
            // a RaW conflict before making any bufferization decisions.
            inconsistentOp = op;
            return WalkResult::interrupt();
          }
        }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return inconsistentOp->emitError("input IR has RaW conflict");
  return success();
}

/// Annotate the IR with the result of the analysis. For testing/debugging only.
static void
annotateOpsWithBufferizationMarkers(Operation *op,
                                    const BufferizationAliasInfo &aliasInfo,
                                    BufferizationState &state) {
  op->walk([&](Operation *op) {
    if (auto bufferizableOp = state.getOptions().dynCastBufferizableOp(op))
      for (OpOperand &opOperand : op->getOpOperands())
        if (opOperand.get().getType().isa<TensorType>())
          setInPlaceOpOperand(opOperand, aliasInfo.isInPlace(opOperand));
  });
}

/// Rewrite pattern that bufferizes bufferizable ops.
struct BufferizationPattern
    : public OpInterfaceRewritePattern<BufferizableOpInterface> {
  BufferizationPattern(MLIRContext *context, BufferizationState &state,
                       PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<BufferizableOpInterface>(context, benefit),
        state(state) {}

  LogicalResult matchAndRewrite(BufferizableOpInterface bufferizableOp,
                                PatternRewriter &rewriter) const override {
    // No tensors => no buffers.
    if (!hasTensorSemantics(bufferizableOp.getOperation()))
      return failure();
    if (!state.getOptions().isOpAllowed(bufferizableOp.getOperation()))
      return failure();
    return bufferizableOp.bufferize(rewriter, state);
  }

private:
  const BufferizationState &state;
};

/// Check the result of bufferization. Return an error if an op was not
/// bufferized, unless partial bufferization is allowed.
static LogicalResult
checkBufferizationResult(Operation *op, const BufferizationOptions &options) {
  if (!options.allowUnknownOps) {
    // Check if all ops were bufferized.
    LogicalResult status = success();
    op->walk([&](Operation *op) {
      if (!hasTensorSemantics(op))
        return WalkResult::advance();

      // Bufferization dialect ops will canonicalize away if all other ops are
      // bufferized.
      if (isa<bufferization::ToMemrefOp, bufferization::ToTensorOp>(op))
        return WalkResult::advance();

      // Ops that are not in the allow list can be ignored.
      if (!options.isOpAllowed(op))
        return WalkResult::advance();

      // Ops without any uses and no side effects will fold away.
      if (op->getUses().empty() && MemoryEffectOpInterface::hasNoEffect(op))
        return WalkResult::advance();

      status = op->emitError("op was not bufferized");
      return WalkResult::interrupt();
    });

    if (failed(status))
      return status;
  }

  return success();
}

LogicalResult
mlir::linalg::comprehensive_bufferize::analyzeOp(Operation *op,
                                                 BufferizationState &state) {
  DominanceInfo domInfo(op);
  BufferizationAliasInfo &aliasInfo = state.getAliasInfo();
  const BufferizationOptions &options = state.getOptions();

  if (failed(checkAliasInfoConsistency(op, domInfo, state, aliasInfo)))
    return failure();

  // If the analysis fails, just return.
  if (failed(inPlaceAnalysis(op, aliasInfo, state, domInfo,
                             options.analysisFuzzerSeed)))
    return failure();
  equivalenceAnalysis(op, aliasInfo, state);

  for (const std::unique_ptr<PostAnalysisStep> &step :
       options.postAnalysisSteps) {
    SmallVector<Operation *> newOps;
    if (failed(step->run(op, state, aliasInfo, newOps)))
      return failure();
    // Analyze ops that were created by the PostAnalysisStep.
    if (failed(inPlaceAnalysis(newOps, aliasInfo, state, domInfo)))
      return failure();
    equivalenceAnalysis(newOps, aliasInfo, state);
  }

  // Annotate operations if we only want to report the analysis.
  if (options.testAnalysisOnly)
    annotateOpsWithBufferizationMarkers(op, aliasInfo, state);

  return success();
}

LogicalResult
mlir::linalg::comprehensive_bufferize::bufferizeOp(Operation *op,
                                                   BufferizationState &state) {
  // Bufferize the op and its nested ops.
  OwningRewritePatternList patterns(op->getContext());
  patterns.add<BufferizationPattern>(op->getContext(), state);
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
    return failure();

  return checkBufferizationResult(op, state.getOptions());
}

LogicalResult mlir::linalg::comprehensive_bufferize::runComprehensiveBufferize(
    Operation *op, std::unique_ptr<BufferizationOptions> options) {
  BufferizationState state(op, *options);
  if (failed(analyzeOp(op, state)))
    return failure();
  if (options->testAnalysisOnly)
    return success();
  if (failed(bufferizeOp(op, state)))
    return failure();
  return success();
}
