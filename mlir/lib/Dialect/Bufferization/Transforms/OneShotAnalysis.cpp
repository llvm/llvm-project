//===- OneShotAnalysis.cpp - One-Shot (Single Pass) Analysis --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One-Shot Analysis analyzes function bodies. By default, function boundaries
// (FuncOp bbArgs, CallOps, ReturnOps) are treated as "unknown" ops.
// OneShotModuleBufferization.cpp is an extension of One-Shot Analysis for
// simple call graphs without loops.
//
// One-Shot Bufferize consists of three phases.
//
// 1. Analyze ops to decide which OpOperands can bufferize inplace, i.e.,
//    without inserting buffer copies. The analysis queries op bufferization
//    semantics via `BufferizableOpInterface`.
// 2. Insert copies for OpOperands that were decided to bufferize out-of-place
//    in tensor land during `TensorCopyInsertion`.
// 3. Bufferize ops by calling `BufferizableOpInterface::bufferize`.
//
// This file contains only the analysis. For convenience, this file also
// contains a helper function `runOneShotBufferize` that analyzes an op (and its
// nested ops) and then bufferizes it.
//
// Inplace bufferization decisions are passed from the analysis to the
// `TensorCopyInsertion` phase via `AnalysisState`. They can be printed for
// debugging purposes with `testAnalysisOnly`.
//
// Ops that do not implement `BufferizableOpInterface` can be analyzed but are
// treated conservatively. E.g., the analysis has to assume that their tensor
// OpOperands bufferize to memory writes. While such ops can be analyzed, they
// are not bufferized and remain in the IR. to_tensor and to_memref ops are
// inserted at the bufferization boundary.
//
// This analysis caters to high-performance codegen where buffer reuse is deemed
// critical: the analysis should fail if the bufferized form of the function
// needs to return a buffer, unless `allowReturnAllocs` is enabled.

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

#include <optional>
#include <random>

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::bufferization::OneShotAnalysisState)

// Run mlir-opt with `-debug-only="one-shot-analysis"` for detailed debug
// output.
#define DEBUG_TYPE "one-shot-analysis"

using namespace mlir;
using namespace mlir::bufferization;

static bool isaTensor(Type t) { return isa<TensorType>(t); }

//===----------------------------------------------------------------------===//
// Bufferization-specific attribute manipulation.
// These are for testing and debugging only. Bufferization information is stored
// in OneShotBufferizationState. When run with `testAnalysisOnly`, the IR is
// annotated with the results of the analysis, so that they can be checked in
// tests.
//===----------------------------------------------------------------------===//

/// Attribute marker to specify op operands that bufferize in-place.
constexpr StringLiteral kInPlaceOperandsAttrName = "__inplace_operands_attr__";

constexpr StringLiteral kOpResultAliasSetAttrName =
    "__opresult_alias_set_attr__";

constexpr StringLiteral kBbArgAliasSetAttrName = "__bbarg_alias_set_attr__";

/// Mark whether OpOperand will be bufferized inplace.
static void setInPlaceOpOperand(OpOperand &opOperand, bool inPlace) {
  Operation *op = opOperand.getOwner();
  SmallVector<StringRef> inPlaceVector;
  if (auto attr = op->getAttr(kInPlaceOperandsAttrName)) {
    inPlaceVector = SmallVector<StringRef>(llvm::to_vector<4>(
        cast<ArrayAttr>(attr).getAsValueRange<StringAttr>()));
  } else {
    inPlaceVector = SmallVector<StringRef>(op->getNumOperands(), "none");
    for (OpOperand &opOperand : op->getOpOperands())
      if (isa<TensorType>(opOperand.get().getType()))
        inPlaceVector[opOperand.getOperandNumber()] = "false";
  }
  inPlaceVector[opOperand.getOperandNumber()] = inPlace ? "true" : "false";
  op->setAttr(kInPlaceOperandsAttrName,
              OpBuilder(op).getStrArrayAttr(inPlaceVector));
}

//===----------------------------------------------------------------------===//
// OneShotAnalysisState
//===----------------------------------------------------------------------===//

OneShotAnalysisState::OneShotAnalysisState(
    Operation *op, const OneShotBufferizationOptions &options)
    : AnalysisState(options, TypeID::get<OneShotAnalysisState>()) {
  // Set up alias sets.
  op->walk([&](Operation *op) {
    for (Value v : op->getResults())
      if (isa<TensorType>(v.getType()))
        createAliasInfoEntry(v);
    for (Region &r : op->getRegions())
      for (Block &b : r.getBlocks())
        for (auto bbArg : b.getArguments())
          if (isa<TensorType>(bbArg.getType()))
            createAliasInfoEntry(bbArg);
  });

  // Mark OpOperands in-place that must bufferize in-place.
  op->walk([&](BufferizableOpInterface bufferizableOp) {
    if (!options.isOpAllowed(bufferizableOp))
      return WalkResult::skip();
    for (OpOperand &opOperand : bufferizableOp->getOpOperands())
      if (isa<TensorType>(opOperand.get().getType()))
        if (bufferizableOp.mustBufferizeInPlace(opOperand, *this))
          bufferizeInPlace(opOperand);
    return WalkResult::advance();
  });
}

void OneShotAnalysisState::applyOnEquivalenceClass(
    Value v, function_ref<void(Value)> fun) const {
  auto leaderIt = equivalentInfo.findLeader(v);
  for (auto mit = leaderIt, meit = equivalentInfo.member_end(); mit != meit;
       ++mit) {
    fun(*mit);
  }
}

void OneShotAnalysisState::applyOnAliases(Value v,
                                          function_ref<void(Value)> fun) const {
  auto leaderIt = aliasInfo.findLeader(v);
  for (auto mit = leaderIt, meit = aliasInfo.member_end(); mit != meit; ++mit) {
    fun(*mit);
  }
}

bool OneShotAnalysisState::areEquivalentBufferizedValues(Value v1,
                                                         Value v2) const {
  return equivalentInfo.isEquivalent(v1, v2);
}

bool OneShotAnalysisState::areAliasingBufferizedValues(Value v1,
                                                       Value v2) const {
  return aliasInfo.isEquivalent(v1, v2);
}

void OneShotAnalysisState::bufferizeInPlace(OpOperand &operand) {
  if (inplaceBufferized.contains(&operand))
    return;
  inplaceBufferized.insert(&operand);
  for (AliasingValue alias : getAliasingValues(operand))
    aliasInfo.unionSets(alias.value, operand.get());
  ++statNumTensorInPlace;
}

void OneShotAnalysisState::bufferizeOutOfPlace(OpOperand &operand) {
  assert(!inplaceBufferized.contains(&operand) &&
         "OpOperand was already decided to bufferize inplace");
  ++statNumTensorOutOfPlace;
}

void OneShotAnalysisState::createAliasInfoEntry(Value v) {
  aliasInfo.insert(v);
  equivalentInfo.insert(v);
}

void OneShotAnalysisState::gatherUndefinedTensorUses(Operation *op) {
  op->walk([&](Operation *op) {
    // Skip unknown ops.
    auto bufferizableOp = getOptions().dynCastBufferizableOp(op);
    if (!bufferizableOp)
      return WalkResult::skip();

    // Check all tensor OpResults.
    for (OpResult opResult : op->getOpResults()) {
      if (!isa<TensorType>(opResult.getType()))
        continue;

      // If there is no preceding definition, the tensor contents are
      // undefined.
      if (findDefinitionsCached(opResult).empty())
        for (OpOperand &use : opResult.getUses())
          undefinedTensorUses.insert(&use);
    }

    return WalkResult::advance();
  });
}

bool OneShotAnalysisState::hasUndefinedContents(OpOperand *opOperand) const {
  return undefinedTensorUses.contains(opOperand);
}

bool OneShotAnalysisState::isInPlace(OpOperand &opOperand) const {
  return inplaceBufferized.contains(&opOperand);
}

bool OneShotAnalysisState::isValueWritten(Value value) const {
  bool isWritten = false;
  applyOnAliases(value, [&](Value val) {
    for (OpOperand &use : val.getUses())
      if (isInPlace(use) && bufferizesToMemoryWrite(use))
        isWritten = true;
  });
  return isWritten;
}

bool OneShotAnalysisState::isWritable(Value value) const {
  // TODO: Out-of-place bufferized value could be considered writable.
  // Query BufferizableOpInterface to see if the BlockArgument is writable.
  if (auto bufferizableOp =
          getOptions().dynCastBufferizableOp(getOwnerOfValue(value)))
    return bufferizableOp.isWritable(value, *this);

  // Not a bufferizable op: The conservative answer is "not writable".
  return false;
}

void OneShotAnalysisState::unionAliasSets(Value v1, Value v2) {
  aliasInfo.unionSets(v1, v2);
}

void OneShotAnalysisState::unionEquivalenceClasses(Value v1, Value v2) {
  equivalentInfo.unionSets(v1, v2);
}

OneShotAnalysisState::Extension::~Extension() = default;

//===----------------------------------------------------------------------===//
// Bufferization-specific alias analysis.
//===----------------------------------------------------------------------===//

/// Return true if opOperand has been decided to bufferize in-place.
static bool isInplaceMemoryWrite(OpOperand &opOperand,
                                 const OneShotAnalysisState &state) {
  // OpOperands that do not bufferize to a memory write do not write in-place.
  if (!state.bufferizesToMemoryWrite(opOperand))
    return false;
  // Check current bufferization decisions.
  return state.isInPlace(opOperand);
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

static bool isReachable(Block *from, Block *to, ArrayRef<Block *> except) {
  DenseSet<Block *> visited;
  SmallVector<Block *> worklist;
  for (Block *succ : from->getSuccessors())
    worklist.push_back(succ);
  while (!worklist.empty()) {
    Block *next = worklist.pop_back_val();
    if (llvm::find(except, next) != except.end())
      continue;
    if (next == to)
      return true;
    if (visited.contains(next))
      continue;
    visited.insert(next);
    for (Block *succ : next->getSuccessors())
      worklist.push_back(succ);
  }
  return false;
}

/// Return `true` if op dominance can be used to rule out a read-after-write
/// conflicts based on the ordering of ops. Returns `false` if op dominance
/// cannot be used to due region-based loops.
///
/// Generalized op dominance can often be used to rule out potential conflicts
/// due to "read happens before write". E.g., the following IR is not a RaW
/// conflict because the read happens *before* the write.
///
/// Example 1:
/// %0 = ... : tensor<?xf32>                                // DEF
/// "reading_op"(%0) : tensor<?xf32>                        // READ
/// %1 = "writing_op"(%0) : tensor<?xf32> -> tensor<?xf32>  // WRITE
///
/// This is no longer true inside loops (or repetitive regions). In such cases,
/// there may not be a meaningful `happensBefore` relationship because ops
/// could be executed multiple times. E.g.:
///
/// Example 2:
/// %0 = ... : tensor<?xf32>                                  // DEF
/// scf.for ... {
///   "reading_op"(%0) : tensor<?xf32>                        // READ
///   %1 = "writing_op"(%0) : tensor<?xf32> -> tensor<?xf32>  // WRITE
///   ...
/// }
///
/// In the above example, reading_op happens before writing_op according to
/// op dominance. However, both ops may happen multiple times; in
/// particular, the second execution of reading_op happens after the first
/// execution of writing_op. This is problematic because the tensor %0 they
/// operate on (i.e., the "definition") is defined outside of the loop.
///
/// On a high-level, there is a potential RaW in a program if there exists a
/// possible program execution such that there is a sequence of DEF, followed
/// by WRITE, followed by READ. Each additional DEF resets the sequence.
///
/// E.g.:
/// No conflict:        DEF, WRITE, DEF, READ
/// Potential conflict: DEF, READ, WRITE, READ, WRITE
///
/// Example 1 has no conflict:          DEF, READ, WRITE
/// Example 2 has a potential conflict: DEF, (READ, WRITE)*
//
/// Example 3:
/// scf.for ... {
///   %0 = ... : tensor<?xf32>
///   "reading_op"(%0) : tensor<?xf32>
///   %1 = "writing_op"(%0) : tensor<?xf32> -> tensor<?xf32>
///   ...
/// }
/// This has no conflict: (DEF, READ, WRITE)*
///
/// Example 4:
/// %0 = ... : tensor<?xf32>
/// scf.for ... {
///   scf.for ... { "reading_op"(%0) }
///   %1 = "writing_op"(%0)
/// }
/// This has a potential conflict: DEF, ((READ)*, WRITE)*
///
/// Example 5:
/// %0 = ... : tensor<?xf32>
/// scf.for ... { %1 = "writing_op"(%0) }
/// scf.for ... { "reading_op"(%0) }
/// This has a potential conflict: DEF, WRITE*, READ*
///
/// The following rules are used to rule out RaW conflicts via ordering of ops:
///
/// 1. If the closest enclosing repetitive region of DEF is a proper ancestor of
///    a repetitive region that enclosing both READ and WRITE, we cannot rule
///    out RaW conflict due to the ordering of ops.
/// 2. Otherwise: There are no loops that interfere with our analysis; for
///    analysis purposes, we can assume that there are no loops/repetitive
///    regions. I.e., we can rule out a RaW conflict if READ happensBefore WRITE
///    or WRITE happensBefore DEF. (Checked in `hasReadAfterWriteInterference`.)
///
static bool canUseOpDominanceDueToRegions(OpOperand *uRead, OpOperand *uWrite,
                                          const SetVector<Value> &definitions,
                                          AnalysisState &state) {
  const BufferizationOptions &options = state.getOptions();
  for (Value def : definitions) {
    Region *rRead =
        state.getEnclosingRepetitiveRegion(uRead->getOwner(), options);
    Region *rDef = state.getEnclosingRepetitiveRegion(def, options);

    // READ and DEF are in the same repetitive region. `happensBefore` can be
    // used to rule out RaW conflicts due to op ordering.
    if (rRead == rDef)
      continue;

    // Find the enclosing repetitive region of READ that is closest to DEF but
    // not the repetitive region of DEF itself.
    while (true) {
      Region *nextRegion = getNextEnclosingRepetitiveRegion(rRead, options);
      if (nextRegion == rDef)
        break;
      assert(nextRegion && "expected to find another repetitive region");
      rRead = nextRegion;
    }

    // We cannot use op dominance if WRITE is inside the same repetitive region.
    if (rRead->getParentOp()->isAncestor(uWrite->getOwner()))
      return false;
  }

  return true;
}

/// Return `true` if op dominance can be used to rule out a read-after-write
/// conflicts based on the ordering of ops. Returns `false` if op dominance
/// cannot be used to due block-based loops within a region.
///
/// Refer to the `canUseOpDominanceDueToRegions` documentation for details on
/// how op domiance is used during RaW conflict detection.
///
/// On a high-level, there is a potential RaW in a program if there exists a
/// possible program execution such that there is a sequence of DEF, followed
/// by WRITE, followed by READ. Each additional DEF resets the sequence.
///
/// Op dominance cannot be used if there is a path from block(READ) to
/// block(WRITE) and a path from block(WRITE) to block(READ). block(DEF) should
/// not appear on that path.
static bool canUseOpDominanceDueToBlocks(OpOperand *uRead, OpOperand *uWrite,
                                         const SetVector<Value> &definitions,
                                         AnalysisState &state) {
  // Fast path: If READ and WRITE are in different regions, their block cannot
  // be reachable just via unstructured control flow. (Loops due to regions are
  // covered by `canUseOpDominanceDueToRegions`.)
  if (uRead->getOwner()->getParentRegion() !=
      uWrite->getOwner()->getParentRegion())
    return true;

  Block *readBlock = uRead->getOwner()->getBlock();
  Block *writeBlock = uWrite->getOwner()->getBlock();
  for (Value def : definitions) {
    Block *defBlock = def.getParentBlock();
    if (isReachable(readBlock, writeBlock, {defBlock}) &&
        isReachable(writeBlock, readBlock, {defBlock}))
      return false;
  }

  return true;
}

static bool canUseOpDominance(OpOperand *uRead, OpOperand *uWrite,
                              const SetVector<Value> &definitions,
                              AnalysisState &state) {
  return canUseOpDominanceDueToRegions(uRead, uWrite, definitions, state) &&
         canUseOpDominanceDueToBlocks(uRead, uWrite, definitions, state);
}

/// Annotate IR with details about the detected RaW conflict.
static void annotateConflict(OpOperand *uRead, OpOperand *uConflictingWrite,
                             Value definition) {
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

  if (auto opResult = dyn_cast<OpResult>(definition)) {
    std::string defAttr =
        id + "[DEF: result " + std::to_string(opResult.getResultNumber()) + "]";
    opResult.getDefiningOp()->setAttr(defAttr, b.getUnitAttr());
  } else {
    auto bbArg = cast<BlockArgument>(definition);
    std::string defAttr =
        id + "[DEF: bbArg " + std::to_string(bbArg.getArgNumber()) + "]";
    bbArg.getOwner()->getParentOp()->setAttr(defAttr, b.getUnitAttr());
  }
}

/// Return 'true' if a tensor that is equivalent to `other` can be found in the
/// reverse use-def chain of `start`. Note: If an OpOperand bufferizes out of
/// place along that use-def chain, the two tensors may not materialize as
/// equivalent buffers (but separate allocations).
///
/// Note: This function also requires that the two tensors have equivalent
/// indexing. I.e., the tensor types do not change along the use-def chain,
/// apart from static <-> dynamic dim casts.
static bool hasEquivalentValueInReverseUseDefChain(AnalysisState &state,
                                                   Value start, Value other) {
  TraversalConfig config;
  config.followEquivalentOnly = true;
  config.alwaysIncludeLeaves = false;
  config.followSameTypeOrCastsOnly = true;
  return !state
              .findValueInReverseUseDefChain(
                  start, [&](Value v) { return v == other; }, config)
              .empty();
}

/// Return "true" if `value` is originating from a subset that is equivalent to
/// the subset that `subsetOp` inserts into.
static bool matchesInsertDestination(const AnalysisState &state, Value value,
                                     SubsetInsertionOpInterface subsetOp) {
  auto matchingSubset = [&](Value val) {
    if (auto opResult = dyn_cast<OpResult>(val))
      if (subsetOp.isEquivalentSubset(opResult, [&](Value v1, Value v2) {
            return state.areEquivalentBufferizedValues(v1, v2);
          }))
        return true;
    return false;
  };
  // There may be multiple leaves at which the reverse SSA use-def chain lookup
  // terminates. All of them must be equivalent subsets.
  SetVector<Value> backwardSlice =
      state.findValueInReverseUseDefChain(value, matchingSubset);
  return static_cast<bool>(llvm::all_of(backwardSlice, matchingSubset));
}

/// Return "true" if the given "read" and potentially conflicting "write" are
/// not conflicting due to their subset relationship. The comments in this
/// function are expressed in terms of tensor.extract_slice/tensor.insert_slice
/// pairs, but apply to any subset ops that implement the
/// `SubsetInsertionOpInterface`.
static bool areNonConflictingSubsets(OpOperand *uRead,
                                     OpOperand *uConflictingWrite,
                                     const AnalysisState &state) {
  Operation *readingOp = uRead->getOwner();
  Operation *conflictingWritingOp = uConflictingWrite->getOwner();

  // Special rules for matching ExtractSliceOp/InsertSliceOp pairs. If
  // uRead is an InsertSliceOp...
  if (auto subsetOp = dyn_cast<SubsetInsertionOpInterface>(readingOp)) {
    // As an example, consider the following IR.
    //
    // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
    // %1 = linalg.fill %cst, %0 {inplace= [true] }
    // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
    //     {inplace= [true] }

    if (uRead == &subsetOp.getDestinationOperand() &&
        matchesInsertDestination(state, uConflictingWrite->get(), subsetOp))
      // Case 1: The main insight is that InsertSliceOp reads only part of
      // the destination tensor. The overwritten area is not read. If
      // uConflictingWrite writes into exactly the memory location that is
      // being read by uRead, this is not a conflict.
      //
      // In the above example:
      // uRead             = OpOperand 1 (%t) of tensor.insert_slice
      // uConflictingWrite = OpOperand 1 (%0) of linalg.fill
      //
      // The read of %t does not conflict with the write of the FillOp
      // (same aliases!) because the area that the FillOp operates on is
      // exactly the one that is *not* read via %t.
      return true;

    if (uRead == &subsetOp.getSourceOperand() &&
        uConflictingWrite == &subsetOp.getDestinationOperand() &&
        matchesInsertDestination(state, uRead->get(), subsetOp))
      // Case 2: The read of the source tensor and the write to the dest
      // tensor via an InsertSliceOp is not a conflict if the read is
      // reading exactly that part of an equivalent tensor that the
      // InsertSliceOp is writing.
      //
      // In the above example:
      // uRead             = OpOperand 0 (%1) of tensor.insert_slice
      // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
      return true;
  }

  // If uConflictingWrite is an InsertSliceOp...
  if (auto subsetOp =
          dyn_cast<SubsetInsertionOpInterface>(conflictingWritingOp))
    // As an example, consider the following IR.
    //
    // %0 = tensor.extract_slice %t[%a, %b][%c, %d][1, 1] {inplace = [true] }
    // %1 = linalg.fill %cst, %0 {inplace= [true] }
    // %2 = tensor.insert_slice %1 into %t[%a, %b][%c, %d][1, 1]
    //     {inplace= [true] }
    // %3 = vector.transfer_read %1, %cst
    //
    // In the above example:
    // uRead             = OpOperand 0 (%1) of vector.transfer_read
    // uConflictingWrite = OpOperand 1 (%t) of tensor.insert_slice
    // definition        = %1
    //
    // This is not a conflict because the InsertSliceOp overwrites the
    // memory segment of %1 with the exact same data. (Effectively, there
    // is no memory write here.)
    if (uConflictingWrite == &subsetOp.getDestinationOperand() &&
        state.areEquivalentBufferizedValues(
            uRead->get(), subsetOp.getSourceOperand().get()) &&
        matchesInsertDestination(state, subsetOp.getSourceOperand().get(),
                                 subsetOp))
      return true;

  return false;
}

/// Given sets of uses and writes, return true if there is a RaW conflict under
/// the assumption that all given reads/writes alias the same buffer and that
/// all given writes bufferize inplace.
///
/// A conflict is: According to SSA use-def chains, a read R is supposed to read
/// the result of a definition W1. But because of bufferization decisions, R
/// actually reads another definition W2.
static bool
hasReadAfterWriteInterference(const DenseSet<OpOperand *> &usesRead,
                              const DenseSet<OpOperand *> &usesWrite,
                              const DominanceInfo &domInfo,
                              OneShotAnalysisState &state) {
  const BufferizationOptions &options = state.getOptions();

  // Before going through the main RaW analysis, find cases where a buffer must
  // be privatized due to parallelism. If the result of a write is never read,
  // privatization is not necessary (and large parts of the IR are likely dead).
  if (!usesRead.empty()) {
    for (OpOperand *uConflictingWrite : usesWrite) {
      // Find the allocation point or last write (definition) of the buffer.
      // Note: In contrast to `findDefinitions`, this also returns results of
      // ops that do not bufferize to memory write when no other definition
      // could be found. E.g., "bufferization.alloc_tensor" would be included,
      // even though that op just bufferizes to an allocation but does define
      // the contents of the buffer.
      SetVector<Value> definitionsOrLeaves =
          state.findValueInReverseUseDefChain(
              uConflictingWrite->get(),
              [&](Value v) { return state.bufferizesToMemoryWrite(v); });
      assert(!definitionsOrLeaves.empty() &&
             "expected at least one definition or leaf");

      // The writing op must bufferize out-of-place if the definition is in a
      // different parallel region than this write.
      for (Value def : definitionsOrLeaves) {
        if (getParallelRegion(def.getParentRegion(), options) !=
            getParallelRegion(uConflictingWrite->getOwner()->getParentRegion(),
                              options)) {
          LLVM_DEBUG(
              llvm::dbgs()
              << "\n- bufferizes out-of-place due to parallel region:\n");
          LLVM_DEBUG(llvm::dbgs()
                     << "  unConflictingWrite = operand "
                     << uConflictingWrite->getOperandNumber() << " of "
                     << *uConflictingWrite->getOwner() << "\n");
          return true;
        }
      }
    }
  }

  for (OpOperand *uRead : usesRead) {
    Operation *readingOp = uRead->getOwner();
    LLVM_DEBUG(llvm::dbgs() << "\n- check conflict:\n");
    LLVM_DEBUG(llvm::dbgs() << "  uRead = operand " << uRead->getOperandNumber()
                            << " of " << *readingOp << "\n");

    // Find the definition of uRead by following the SSA use-def chain.
    // E.g.:
    //
    // %0 = "writing_op"(%t) : tensor<?x32> -> tensor<?xf32>
    // %1 = "aliasing_op"(%0) : tensor<?x32> -> tensor<?xf32>
    // %2 = "reading_op"(%1) : : tensor<?x32> -> not_a_tensor_type
    //
    // In the above example, if uRead is the OpOperand of reading_op, the
    // definition is %0. Note that operations that create an alias but do not
    // bufferize to a memory write (such as ExtractSliceOp) are skipped.
    const SetVector<Value> &definitions =
        state.findDefinitionsCached(uRead->get());
    if (definitions.empty()) {
      // Fast path: No conflict if there are no definitions.
      LLVM_DEBUG(llvm::dbgs()
                 << "  no conflict: read value has no definitions\n");
      continue;
    }

    // Look for conflicting memory writes. Potential conflicts are writes to an
    // alias that have been decided to bufferize inplace.
    for (OpOperand *uConflictingWrite : usesWrite) {
      LLVM_DEBUG(llvm::dbgs() << "  unConflictingWrite = operand "
                              << uConflictingWrite->getOperandNumber() << " of "
                              << *uConflictingWrite->getOwner() << "\n");

      // Check if op dominance can be used to rule out read-after-write
      // conflicts.
      bool useDominance =
          canUseOpDominance(uRead, uConflictingWrite, definitions, state);
      LLVM_DEBUG(llvm::dbgs() << "\n- useDominance = " << useDominance << "\n");

      // Throughout this loop, check for multiple requirements that have to be
      // met for uConflictingWrite to be an actual conflict.
      Operation *conflictingWritingOp = uConflictingWrite->getOwner();

      // Inside of repetitive regions, ops may be executed multiple times and op
      // dominance cannot be used to rule out conflicts.
      if (useDominance) {
        // No conflict if the readingOp dominates conflictingWritingOp, i.e.,
        // the write is not visible when reading.
        //
        // Note: If ops are executed multiple times (e.g., because they are
        //       inside a loop), there may be no meaningful `happensBefore`
        //       relationship.
        if (happensBefore(readingOp, conflictingWritingOp, domInfo)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  no conflict: read happens before write\n");
          continue;
        }

        // No conflict if the reading use equals the use of the conflicting
        // write. A use cannot conflict with itself.
        //
        // Note: Just being the same op is not enough. It has to be the same
        //       use.
        // Note: If the op is executed multiple times (e.g., because it is
        //       inside a loop), it may be conflicting with itself.
        if (uConflictingWrite == uRead) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  no conflict: read and write are same use\n");
          continue;
        }

        // Ops are not conflicting if they are in mutually exclusive regions.
        //
        // Note: If ops are executed multiple times (e.g., because they are
        //       inside a loop), mutually exclusive regions may be executed
        //       multiple times.
        if (insideMutuallyExclusiveRegions(readingOp, conflictingWritingOp)) {
          LLVM_DEBUG(llvm::dbgs() << "  no conflict: read and write are in "
                                     "mutually exclusive regions\n");
          continue;
        }
      }

      // Two equivalent operands of the same op are not conflicting if the op
      // bufferizes to element-wise access. I.e., all loads at a position happen
      // before all stores to the same position.
      if (conflictingWritingOp == readingOp) {
        if (auto bufferizableOp = options.dynCastBufferizableOp(readingOp)) {
          if (bufferizableOp.bufferizesToElementwiseAccess(
                  state, {uRead, uConflictingWrite})) {
            if (hasEquivalentValueInReverseUseDefChain(
                    state, uRead->get(), uConflictingWrite->get()) ||
                hasEquivalentValueInReverseUseDefChain(
                    state, uConflictingWrite->get(), uRead->get())) {
              LLVM_DEBUG(
                  llvm::dbgs()
                  << "  no conflict: op bufferizes to element-wise access\n");
              continue;
            }
          }
        }
      }

      // No conflict if the operands are non-conflicting subsets.
      if (areNonConflictingSubsets(uRead, uConflictingWrite, state)) {
        LLVM_DEBUG(llvm::dbgs() << "  no conflict: non-conflicting subsets\n");
        continue;
      }

      // No conflict if the op interface says so.
      if (auto bufferizableOp = options.dynCastBufferizableOp(readingOp)) {
        if (bufferizableOp.isNotConflicting(uRead, uConflictingWrite, state)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  no conflict: op interace of reading op says 'no'\n");
          continue;
        }
      }

      if (conflictingWritingOp != readingOp) {
        if (auto bufferizableOp =
                options.dynCastBufferizableOp(conflictingWritingOp)) {
          if (bufferizableOp.isNotConflicting(uRead, uConflictingWrite,
                                              state)) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "  no conflict: op interace of writing op says 'no'\n");
            continue;
          }
        }
      }

      // Check all possible definitions.
      for (Value definition : definitions) {
        LLVM_DEBUG(llvm::dbgs() << "  * definition = " << definition << "\n");

        // No conflict if the conflicting write happens before the definition.
        if (Operation *defOp = definition.getDefiningOp()) {
          if (happensBefore(conflictingWritingOp, defOp, domInfo)) {
            // conflictingWritingOp happens before defOp. No conflict.
            LLVM_DEBUG(llvm::dbgs()
                       << "    no conflict: write happens before definition\n");
            continue;
          }
          // No conflict if conflictingWritingOp is contained in defOp.
          if (defOp->isProperAncestor(conflictingWritingOp)) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "    no conflict: write is contained in definition\n");
            continue;
          }
        } else {
          auto bbArg = cast<BlockArgument>(definition);
          Block *block = bbArg.getOwner();
          if (!block->findAncestorOpInBlock(*conflictingWritingOp)) {
            LLVM_DEBUG(llvm::dbgs() << "    no conflict: definition is bbArg "
                                       "and write happens outside of block\n");
            // conflictingWritingOp happens outside of the block. No
            // conflict.
            continue;
          }
        }

        // No conflict if the conflicting write and the definition are the same
        // use.
        AliasingValueList aliases = state.getAliasingValues(*uConflictingWrite);
        if (aliases.getNumAliases() == 1 &&
            aliases.getAliases()[0].value == definition) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    no conflict: definition and write are same\n");
          continue;
        }

        // All requirements are met. Conflict found!

        if (options.printConflicts)
          annotateConflict(uRead, uConflictingWrite, definition);
        LLVM_DEBUG(llvm::dbgs() << "  => RaW CONFLICT FOUND\n");
        return true;
      }
    }
  }

  return false;
}

// Helper function to iterate on aliases of `root` and capture the writes.
static void getAliasingInplaceWrites(DenseSet<OpOperand *> &res, Value root,
                                     const OneShotAnalysisState &state) {
  state.applyOnAliases(root, [&](Value alias) {
    for (auto &use : alias.getUses())
      // Inplace write to a value that aliases root.
      if (isInplaceMemoryWrite(use, state))
        res.insert(&use);
  });
}

// Helper function to iterate on aliases of `root` and capture the reads.
static void getAliasingReads(DenseSet<OpOperand *> &res, Value root,
                             const OneShotAnalysisState &state) {
  state.applyOnAliases(root, [&](Value alias) {
    for (auto &use : alias.getUses()) {
      // Read of a value that aliases root.
      if (state.bufferizesToMemoryRead(use)) {
        res.insert(&use);
        continue;
      }

      // Read of a dependent value in the SSA use-def chain. E.g.:
      //
      // %0 = ...
      // %1 = tensor.extract_slice %0 {not_analyzed_yet}
      // "read"(%1)
      //
      // In the above example, getAliasingReads(%0) includes the first OpOperand
      // of the tensor.extract_slice op. The extract_slice itself does not read
      // but its aliasing result is eventually fed into an op that does.
      //
      // Note: This is considered a "read" only if the use does not bufferize to
      // a memory write. (We already ruled out memory reads. In case of a memory
      // write, the buffer would be entirely overwritten; in the above example
      // there would then be no flow of data from the extract_slice operand to
      // its result's uses.)
      if (!state.bufferizesToMemoryWrite(use)) {
        AliasingValueList aliases = state.getAliasingValues(use);
        if (llvm::any_of(aliases, [&](AliasingValue a) {
              return state.isValueRead(a.value);
            }))
          res.insert(&use);
      }
    }
  });
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
///   TransferWriteOp overwrites the result of the first one. Therefore, the
///   TransferReadOp would no longer be reading the result of %1.
///
/// If `checkConsistencyOnly` is true, this function checks if there is a
/// read-after-write conflict without bufferizing `operand` inplace. This would
/// indicate a problem with the current inplace bufferization decisions.
///
/// Note: If `checkConsistencyOnly`, this function may be called with a null
/// OpResult. In that case, only the consistency of bufferization decisions
/// involving aliases of the given OpOperand are checked.
static bool wouldCreateReadAfterWriteInterference(
    OpOperand &operand, const DominanceInfo &domInfo,
    OneShotAnalysisState &state, bool checkConsistencyOnly = false) {
  // Collect reads and writes of all aliases of OpOperand and OpResult.
  DenseSet<OpOperand *> usesRead, usesWrite;
  getAliasingReads(usesRead, operand.get(), state);
  getAliasingInplaceWrites(usesWrite, operand.get(), state);
  for (AliasingValue alias : state.getAliasingValues(operand)) {
    getAliasingReads(usesRead, alias.value, state);
    getAliasingInplaceWrites(usesWrite, alias.value, state);
  }
  if (!checkConsistencyOnly && state.bufferizesToMemoryWrite(operand))
    usesWrite.insert(&operand);

  return hasReadAfterWriteInterference(usesRead, usesWrite, domInfo, state);
}

/// Annotate IR with details about the detected non-writability conflict.
static void annotateNonWritableTensor(Value value) {
  static int64_t counter = 0;
  OpBuilder b(value.getContext());
  std::string id = "W_" + std::to_string(counter++);
  if (auto opResult = dyn_cast<OpResult>(value)) {
    std::string attr = id + "[NOT-WRITABLE: result " +
                       std::to_string(opResult.getResultNumber()) + "]";
    opResult.getDefiningOp()->setAttr(attr, b.getUnitAttr());
  } else {
    auto bbArg = cast<BlockArgument>(value);
    std::string attr = id + "[NOT-WRITABLE: bbArg " +
                       std::to_string(bbArg.getArgNumber()) + "]";
    bbArg.getOwner()->getParentOp()->setAttr(attr, b.getUnitAttr());
  }
}

/// Return true if bufferizing `operand` inplace would create a write to a
/// non-writable buffer.
static bool
wouldCreateWriteToNonWritableBuffer(OpOperand &operand,
                                    OneShotAnalysisState &state,
                                    bool checkConsistencyOnly = false) {
  bool foundWrite =
      !checkConsistencyOnly && state.bufferizesToMemoryWrite(operand);

  if (!foundWrite) {
    // Collect writes of all aliases of OpOperand and OpResult.
    DenseSet<OpOperand *> usesWrite;
    getAliasingInplaceWrites(usesWrite, operand.get(), state);
    for (AliasingValue alias : state.getAliasingValues(operand))
      getAliasingInplaceWrites(usesWrite, alias.value, state);
    foundWrite = !usesWrite.empty();
  }

  if (!foundWrite)
    return false;

  // Look for a read-only tensor among all aliases.
  bool foundReadOnly = false;
  auto checkReadOnly = [&](Value v) {
    if (!state.isWritable(v)) {
      foundReadOnly = true;
      if (state.getOptions().printConflicts)
        annotateNonWritableTensor(v);
    }
  };
  state.applyOnAliases(operand.get(), checkReadOnly);
  for (AliasingValue alias : state.getAliasingValues(operand))
    state.applyOnAliases(alias.value, checkReadOnly);
  if (foundReadOnly) {
    LLVM_DEBUG(llvm::dbgs() << "=> NOT WRITABLE\n");
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Bufferization analyses.
//===----------------------------------------------------------------------===//

// Find the values that define the contents of the given value.
const llvm::SetVector<Value> &
OneShotAnalysisState::findDefinitionsCached(Value value) {
  if (!cachedDefinitions.count(value))
    cachedDefinitions[value] = findDefinitions(value);
  return cachedDefinitions[value];
}

void OneShotAnalysisState::resetCache() {
  AnalysisState::resetCache();
  cachedDefinitions.clear();
}

/// Determine if `operand` can be bufferized in-place.
static LogicalResult
bufferizableInPlaceAnalysisImpl(OpOperand &operand, OneShotAnalysisState &state,
                                const DominanceInfo &domInfo) {
  LLVM_DEBUG(
      llvm::dbgs() << "//===-------------------------------------------===//\n"
                   << "Analyzing operand #" << operand.getOperandNumber()
                   << " of " << *operand.getOwner() << "\n");

  bool foundInterference =
      wouldCreateWriteToNonWritableBuffer(operand, state) ||
      wouldCreateReadAfterWriteInterference(operand, domInfo, state);

  if (foundInterference)
    state.bufferizeOutOfPlace(operand);
  else
    state.bufferizeInPlace(operand);

  LLVM_DEBUG(llvm::dbgs()
             << "//===-------------------------------------------===//\n");
  return success();
}

LogicalResult
OneShotAnalysisState::analyzeSingleOp(Operation *op,
                                      const DominanceInfo &domInfo) {
  for (OpOperand &opOperand : op->getOpOperands())
    if (isa<TensorType>(opOperand.get().getType()))
      if (failed(bufferizableInPlaceAnalysisImpl(opOperand, *this, domInfo)))
        return failure();
  return success();
}

/// Return true if the given op has a tensor result or a tensor operand.
static bool hasTensorSemantics(Operation *op) {
  bool hasTensorResult = any_of(op->getResultTypes(), isaTensor);
  bool hasTensorOperand = any_of(op->getOperandTypes(), isaTensor);
  return hasTensorResult || hasTensorOperand;
}

/// Analyze equivalence of tied OpResult/OpOperand pairs of the given ops.
static void equivalenceAnalysis(SmallVector<Operation *> &ops,
                                OneShotAnalysisState &state) {
  for (Operation *op : ops) {
    if (auto bufferizableOp = state.getOptions().dynCastBufferizableOp(op)) {
      for (OpResult opResult : op->getOpResults()) {
        if (!isa<TensorType>(opResult.getType()))
          continue;
        AliasingOpOperandList aliases = state.getAliasingOpOperands(opResult);
        if (aliases.getNumAliases() == 0)
          // Nothing to do if there are no aliasing OpOperands.
          continue;

        Value firstOperand = aliases.begin()->opOperand->get();
        bool allEquivalent = true;
        for (AliasingOpOperand alias : aliases) {
          bool isEquiv = alias.relation == BufferRelation::Equivalent;
          bool isInPlace = state.isInPlace(*alias.opOperand);
          Value operand = alias.opOperand->get();
          if (isEquiv && isInPlace && alias.isDefinite) {
            // Found a definite, equivalent alias. Merge equivalence sets.
            // There can only be one definite alias, so we can stop here.
            state.unionEquivalenceClasses(opResult, operand);
            allEquivalent = false;
            break;
          }
          if (!isEquiv || !isInPlace)
            allEquivalent = false;
          if (!state.areEquivalentBufferizedValues(operand, firstOperand))
            allEquivalent = false;
        }

        // If all "maybe" aliases are equivalent and the OpResult is not a new
        // allocation, it is a definite, equivalent alias. E.g.:
        //
        // aliasingOpOperands(%r) = {(%t0, EQUIV, MAYBE), (%t1, EQUIV, MAYBE)}
        // aliasingValues(%t0) = {(%r, EQUIV, MAYBE)}
        // aliasingValues(%t1) = {(%r, EQUIV, MAYBE)}
        // %r = arith.select %c, %t0, %t1 : tensor<?xf32>
        //
        // If %t0 and %t1 are equivalent, it is safe to union the equivalence
        // classes of %r, %t0 and %t1.
        if (allEquivalent && !bufferizableOp.bufferizesToAllocation(opResult))
          state.unionEquivalenceClasses(opResult, firstOperand);
      }
    }
  }
}

/// Analyze equivalence of tied OpResult/OpOperand pairs of all ops contained
/// in `op`.
static void equivalenceAnalysis(Operation *op, OneShotAnalysisState &state) {
  // Traverse ops in PostOrder: Nested ops first, then enclosing ops.
  SmallVector<Operation *> ops;
  op->walk<WalkOrder::PostOrder>([&](Operation *op) {
    // No tensors => no buffers.
    if (none_of(op->getResultTypes(), isaTensor))
      return;
    ops.push_back(op);
  });

  equivalenceAnalysis(ops, state);
}

LogicalResult OneShotAnalysisState::analyzeOp(Operation *op,
                                              const DominanceInfo &domInfo) {
  // Collect ops so we can build our own reverse traversal.
  SmallVector<Operation *> ops;
  op->walk([&](Operation *op) {
    // No tensors => no buffers.
    if (!hasTensorSemantics(op))
      return;
    ops.push_back(op);
  });

  if (getOptions().analysisFuzzerSeed) {
    // This is a fuzzer. For testing purposes only. Randomize the order in which
    // operations are analyzed. The bufferization quality is likely worse, but
    // we want to make sure that no assertions are triggered anywhere.
    std::mt19937 g(getOptions().analysisFuzzerSeed);
    llvm::shuffle(ops.begin(), ops.end(), g);
  }

  OneShotBufferizationOptions::AnalysisHeuristic heuristic =
      getOptions().analysisHeuristic;
  if (heuristic == OneShotBufferizationOptions::AnalysisHeuristic::BottomUp) {
    // Default: Walk ops in reverse for better interference analysis.
    for (Operation *op : reverse(ops))
      if (failed(analyzeSingleOp(op, domInfo)))
        return failure();
  } else if (heuristic ==
             OneShotBufferizationOptions::AnalysisHeuristic::TopDown) {
    for (Operation *op : ops)
      if (failed(analyzeSingleOp(op, domInfo)))
        return failure();
  } else {
    llvm_unreachable("unsupported heuristic");
  }

  equivalenceAnalysis(op, *this);
  return success();
}

/// Perform various checks on the input IR to see if it contains IR constructs
/// that are unsupported by One-Shot Bufferize.
static LogicalResult
checkPreBufferizationAssumptions(Operation *op, const DominanceInfo &domInfo,
                                 OneShotAnalysisState &state) {
  const BufferizationOptions &options = state.getOptions();

  // Note: This walk cannot be combined with the one below because interface
  // methods of invalid/unsupported ops may be called during the second walk.
  // (On ops different from `op`.)
  WalkResult walkResult = op->walk([&](BufferizableOpInterface op) {
    // Skip ops that are not in the filter.
    if (!options.isOpAllowed(op.getOperation()))
      return WalkResult::advance();

    // Check for unsupported unstructured control flow.
    if (!op.supportsUnstructuredControlFlow()) {
      for (Region &r : op->getRegions()) {
        if (r.getBlocks().size() > 1) {
          op->emitOpError("op or BufferizableOpInterface implementation does "
                          "not support unstructured control flow, but at least "
                          "one region has multiple blocks");
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  walkResult = op->walk([&](BufferizableOpInterface op) {
    // Skip ops that are not in the filter.
    if (!options.isOpAllowed(op.getOperation()))
      return WalkResult::advance();

    // Input IR may not contain any ToTensorOps without the "restrict"
    // attribute. Such tensors may alias any other tensor, which is currently
    // not handled in the analysis.
    if (auto toTensorOp = dyn_cast<ToTensorOp>(op.getOperation())) {
      if (!toTensorOp.getRestrict() && !toTensorOp->getUses().empty()) {
        op->emitOpError("to_tensor ops without `restrict` are not supported by "
                        "One-Shot Analysis");
        return WalkResult::interrupt();
      }
    }

    for (OpOperand &opOperand : op->getOpOperands()) {
      if (isa<TensorType>(opOperand.get().getType())) {
        if (wouldCreateReadAfterWriteInterference(
                opOperand, domInfo, state,
                /*checkConsistencyOnly=*/true)) {
          // This error can happen if certain "mustBufferizeInPlace" interface
          // methods are implemented incorrectly, such that the IR already has
          // a RaW conflict before making any bufferization decisions. It can
          // also happen if the bufferization.materialize_in_destination is used
          // in such a way that a RaW conflict is not avoidable.
          op->emitOpError("not bufferizable under the given constraints: "
                          "cannot avoid RaW conflict");
          return WalkResult::interrupt();
        }

        if (state.isInPlace(opOperand) &&
            wouldCreateWriteToNonWritableBuffer(
                opOperand, state, /*checkConsistencyOnly=*/true)) {
          op->emitOpError("not bufferizable under the given constraints: would "
                          "write to read-only buffer");
          return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}

/// Annotate the IR with the result of the analysis. For testing/debugging only.
static void
annotateOpsWithBufferizationMarkers(Operation *op,
                                    const OneShotAnalysisState &state) {
  // Add __inplace_operands_attr__.
  op->walk([&](Operation *op) {
    for (OpOperand &opOperand : op->getOpOperands())
      if (isa<TensorType>(opOperand.get().getType()))
        setInPlaceOpOperand(opOperand, state.isInPlace(opOperand));
  });
}

static void annotateOpsWithAliasSets(Operation *op,
                                     const OneShotAnalysisState &state) {
  AsmState asmState(op);
  Builder b(op->getContext());
  // Helper function to build an array attribute of aliasing SSA value strings.
  auto buildAliasesArray = [&](Value v) {
    SmallVector<Attribute> aliases;
    state.applyOnAliases(v, [&](Value alias) {
      std::string buffer;
      llvm::raw_string_ostream stream(buffer);
      alias.printAsOperand(stream, asmState);
      aliases.push_back(b.getStringAttr(stream.str()));
    });
    return b.getArrayAttr(aliases);
  };

  op->walk([&](Operation *op) {
    // Build alias set array for every OpResult.
    SmallVector<Attribute> opResultAliasSets;
    for (OpResult opResult : op->getOpResults()) {
      if (llvm::isa<TensorType>(opResult.getType())) {
        opResultAliasSets.push_back(buildAliasesArray(opResult));
      }
    }
    if (!opResultAliasSets.empty())
      op->setAttr(kOpResultAliasSetAttrName, b.getArrayAttr(opResultAliasSets));

    // Build alias set array for every BlockArgument.
    SmallVector<Attribute> regionAliasSets;
    bool hasTensorBbArg = false;
    for (Region &r : op->getRegions()) {
      SmallVector<Attribute> blockAliasSets;
      for (Block &block : r.getBlocks()) {
        SmallVector<Attribute> bbArgAliasSets;
        for (BlockArgument bbArg : block.getArguments()) {
          if (llvm::isa<TensorType>(bbArg.getType())) {
            bbArgAliasSets.push_back(buildAliasesArray(bbArg));
            hasTensorBbArg = true;
          }
        }
        blockAliasSets.push_back(b.getArrayAttr(bbArgAliasSets));
      }
      regionAliasSets.push_back(b.getArrayAttr(blockAliasSets));
    }
    if (hasTensorBbArg)
      op->setAttr(kBbArgAliasSetAttrName, b.getArrayAttr(regionAliasSets));
  });
}

LogicalResult bufferization::analyzeOp(Operation *op,
                                       OneShotAnalysisState &state,
                                       BufferizationStatistics *statistics) {
  DominanceInfo domInfo(op);
  const OneShotBufferizationOptions &options = state.getOptions();

  if (failed(checkPreBufferizationAssumptions(op, domInfo, state)))
    return failure();

  // If the analysis fails, just return.
  if (failed(state.analyzeOp(op, domInfo)))
    return failure();

  if (statistics) {
    statistics->numTensorInPlace = state.getStatNumTensorInPlace();
    statistics->numTensorOutOfPlace = state.getStatNumTensorOutOfPlace();
  }

  bool failedAnalysis = false;

  // Gather some extra analysis data.
  state.gatherUndefinedTensorUses(op);

  // Analysis verification: After setting up alias/equivalence sets, each op
  // can check for expected invariants/limitations and fail the analysis if
  // necessary.
  op->walk([&](Operation *op) {
    if (BufferizableOpInterface bufferizableOp =
            options.dynCastBufferizableOp(op))
      failedAnalysis |= failed(bufferizableOp.verifyAnalysis(state));
  });

  // Annotate operations if we only want to report the analysis.
  if (options.testAnalysisOnly)
    annotateOpsWithBufferizationMarkers(op, state);
  if (options.dumpAliasSets)
    annotateOpsWithAliasSets(op, state);

  return success(!failedAnalysis);
}

LogicalResult
bufferization::runOneShotBufferize(Operation *op,
                                   const OneShotBufferizationOptions &options,
                                   BufferizationStatistics *statistics) {
  assert(!(options.copyBeforeWrite && options.testAnalysisOnly) &&
         "invalid combination of bufferization flags");
  if (!options.copyBeforeWrite) {
    // If a buffer is copied before every write, no analysis is needed.
    if (failed(insertTensorCopies(op, options, statistics)))
      return failure();
  }
  if (options.testAnalysisOnly)
    return success();
  return bufferizeOp(op, options, statistics);
}
