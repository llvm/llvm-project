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

#include <random>
#include <optional>

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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::bufferization::OneShotAnalysisState)

// Run mlir-opt with `-debug-only="one-shot-analysis"` for detailed debug
// output.
#define DEBUG_TYPE "one-shot-analysis"

using namespace mlir;
using namespace mlir::bufferization;

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

//===----------------------------------------------------------------------===//
// Bufferization-specific attribute manipulation.
// These are for testing and debugging only. Bufferization information is stored
// in BufferizationAliasInfo. When run with `testAnalysisOnly`, the IR is
// annotated with the results of the analysis, so that they can be checked in
// tests.
//===----------------------------------------------------------------------===//

/// Attribute marker to specify op operands that bufferize in-place.
constexpr StringLiteral kInPlaceOperandsAttrName = "__inplace_operands_attr__";

/// Mark whether OpOperand will be bufferized inplace.
static void setInPlaceOpOperand(OpOperand &opOperand, bool inPlace) {
  Operation *op = opOperand.getOwner();
  SmallVector<StringRef> inPlaceVector;
  if (auto attr = op->getAttr(kInPlaceOperandsAttrName)) {
    inPlaceVector = SmallVector<StringRef>(llvm::to_vector<4>(
        attr.cast<ArrayAttr>().getAsValueRange<StringAttr>()));
  } else {
    inPlaceVector = SmallVector<StringRef>(op->getNumOperands(), "none");
    for (OpOperand &opOperand : op->getOpOperands())
      if (opOperand.get().getType().isa<TensorType>())
        inPlaceVector[opOperand.getOperandNumber()] = "false";
  }
  inPlaceVector[opOperand.getOperandNumber()] = inPlace ? "true" : "false";
  op->setAttr(kInPlaceOperandsAttrName,
              OpBuilder(op).getStrArrayAttr(inPlaceVector));
}

//===----------------------------------------------------------------------===//
// BufferizationAliasInfo
//===----------------------------------------------------------------------===//

BufferizationAliasInfo::BufferizationAliasInfo(Operation *rootOp) {
  rootOp->walk([&](Operation *op) {
    for (Value v : op->getResults())
      if (v.getType().isa<TensorType>())
        createAliasInfoEntry(v);
    for (Region &r : op->getRegions())
      for (Block &b : r.getBlocks())
        for (auto bbArg : b.getArguments())
          if (bbArg.getType().isa<TensorType>())
            createAliasInfoEntry(bbArg);
  });
}

/// Add a new entry for `v` in the `aliasInfo` and `equivalentInfo`. In the
/// beginning the alias and equivalence sets only contain `v` itself.
void BufferizationAliasInfo::createAliasInfoEntry(Value v) {
  aliasInfo.insert(v);
  equivalentInfo.insert(v);
}

/// Insert an info entry for `newValue` and merge its alias set with that of
/// `alias`.
void BufferizationAliasInfo::insertNewBufferAlias(Value newValue, Value alias) {
  createAliasInfoEntry(newValue);
  aliasInfo.unionSets(newValue, alias);
}

/// Insert an info entry for `newValue` and merge its alias set with that of
/// `alias`. Additionally, merge their equivalence classes.
void BufferizationAliasInfo::insertNewBufferEquivalence(Value newValue,
                                                        Value alias) {
  insertNewBufferAlias(newValue, alias);
  equivalentInfo.unionSets(newValue, alias);
}

/// Return `true` if a value was marked as in-place bufferized.
bool BufferizationAliasInfo::isInPlace(OpOperand &operand) const {
  return inplaceBufferized.contains(&operand);
}

/// Set the inPlace bufferization spec to true.
void BufferizationAliasInfo::bufferizeInPlace(OpOperand &operand,
                                              AnalysisState &state) {
  if (inplaceBufferized.contains(&operand))
    return;
  markInPlace(operand);
  for (OpResult result : state.getAliasingOpResult(operand))
    aliasInfo.unionSets(result, operand.get());
  ++statNumTensorInPlace;
}

/// Set the inPlace bufferization spec to false.
void BufferizationAliasInfo::bufferizeOutOfPlace(OpOperand &operand) {
  assert(!inplaceBufferized.contains(&operand) &&
         "OpOperand was already decided to bufferize inplace");
  ++statNumTensorOutOfPlace;
}

/// Apply `fun` to all the members of the equivalence class of `v`.
void BufferizationAliasInfo::applyOnEquivalenceClass(
    Value v, function_ref<void(Value)> fun) const {
  auto leaderIt = equivalentInfo.findLeader(v);
  for (auto mit = leaderIt, meit = equivalentInfo.member_end(); mit != meit;
       ++mit) {
    fun(*mit);
  }
}

/// Apply `fun` to all aliases of `v`.
void BufferizationAliasInfo::applyOnAliases(
    Value v, function_ref<void(Value)> fun) const {
  auto leaderIt = aliasInfo.findLeader(v);
  for (auto mit = leaderIt, meit = aliasInfo.member_end(); mit != meit; ++mit) {
    fun(*mit);
  }
}

BufferizationAliasInfo::EquivalenceClassRangeType
BufferizationAliasInfo::getAliases(Value v) const {
  DenseSet<Value> res;
  auto it = aliasInfo.findValue(aliasInfo.getLeaderValue(v));
  for (auto mit = aliasInfo.member_begin(it), meit = aliasInfo.member_end();
       mit != meit; ++mit) {
    res.insert(static_cast<Value>(*mit));
  }
  return BufferizationAliasInfo::EquivalenceClassRangeType(
      aliasInfo.member_begin(it), aliasInfo.member_end());
}

//===----------------------------------------------------------------------===//
// OneShotAnalysisState
//===----------------------------------------------------------------------===//

OneShotAnalysisState::OneShotAnalysisState(
    Operation *op, const OneShotBufferizationOptions &options)
    : AnalysisState(options, TypeID::get<OneShotAnalysisState>()),
      aliasInfo(op) {
  // Set up alias sets for OpResults that must bufferize in-place. This should
  // be done before making any other bufferization decisions.
  op->walk([&](BufferizableOpInterface bufferizableOp) {
    if (!options.isOpAllowed(bufferizableOp))
      return WalkResult::skip();
    for (OpOperand &opOperand : bufferizableOp->getOpOperands())
      if (opOperand.get().getType().isa<TensorType>())
        if (bufferizableOp.mustBufferizeInPlace(opOperand, *this))
          aliasInfo.bufferizeInPlace(opOperand, *this);
    return WalkResult::advance();
  });
}

bool OneShotAnalysisState::isInPlace(OpOperand &opOperand) const {
  return aliasInfo.isInPlace(opOperand);
}

bool OneShotAnalysisState::areEquivalentBufferizedValues(Value v1,
                                                         Value v2) const {
  return aliasInfo.areEquivalentBufferizedValues(v1, v2);
}

bool OneShotAnalysisState::areAliasingBufferizedValues(Value v1,
                                                       Value v2) const {
  return aliasInfo.areAliasingBufferizedValues(v1, v2);
}

// Gather yielded tensors in `yieldedTensors` by querying all aliases. This is
// to ensure that such information is available during bufferization time.
// Alias information can no longer be queried through BufferizationAliasInfo
// once we have started modifying the IR.
void OneShotAnalysisState::gatherYieldedTensors(Operation *op) {
  op->walk([&](Operation *returnOp) {
    if (!isRegionReturnLike(returnOp) || !getOptions().isOpAllowed(returnOp))
      return WalkResult::advance();

    for (OpOperand &returnValOperand : returnOp->getOpOperands()) {
      Value returnVal = returnValOperand.get();
      // Skip non-tensor values.
      if (!returnVal.getType().isa<TensorType>())
        continue;

      // Add all aliases of the returned value. But only the ones that are in
      // the same block.
      aliasInfo.applyOnAliases(returnVal, [&](Value v) {
        if (auto bbArg = v.dyn_cast<BlockArgument>()) {
          if (bbArg.getOwner()->getParentOp() == returnOp->getParentOp())
            yieldedTensors.insert(bbArg);
          return;
        }
        Operation *definingOp = v.getDefiningOp();
        if (definingOp->getParentOp() == returnOp->getParentOp())
          yieldedTensors.insert(v);
      });
    }

    return WalkResult::advance();
  });
}

void OneShotAnalysisState::gatherUndefinedTensorUses(Operation *op) {
  op->walk([&](Operation *op) {
    // Skip unknown ops.
    auto bufferizableOp = getOptions().dynCastBufferizableOp(op);
    if (!bufferizableOp)
      return WalkResult::skip();

    // Check all tensor OpResults.
    for (OpResult opResult : op->getOpResults()) {
      if (!opResult.getType().isa<TensorType>())
        continue;

      // If there is no preceding definition, the tensor contents are
      // undefined.
      if (findDefinitions(opResult, /*alwaysIncludeLeaves=*/false).empty())
        for (OpOperand &use : opResult.getUses())
          undefinedTensorUses.insert(&use);
    }

    return WalkResult::advance();
  });
}

bool OneShotAnalysisState::hasUndefinedContents(OpOperand *opOperand) const {
  return undefinedTensorUses.contains(opOperand);
}

bool OneShotAnalysisState::isTensorYielded(Value tensor) const {
  return yieldedTensors.contains(tensor);
}

bool OneShotAnalysisState::isValueWritten(Value value) const {
  bool isWritten = false;
  aliasInfo.applyOnAliases(value, [&](Value val) {
    for (OpOperand &use : val.getUses())
      if (isInPlace(use) && bufferizesToMemoryWrite(use))
        isWritten = true;
  });
  return isWritten;
}

bool OneShotAnalysisState::isWritable(Value value) const {
  // TODO: Out-of-place bufferized value could be considered writable.
  if (auto bufferizableOp = getOptions().dynCastBufferizableOp(value))
    return bufferizableOp.isWritable(value, *this);

  // Query BufferizableOpInterface to see if the BlockArgument is writable.
  if (auto bbArg = value.dyn_cast<BlockArgument>())
    if (auto bufferizableOp =
            getOptions().dynCastBufferizableOp(bbArg.getOwner()->getParentOp()))
      return bufferizableOp.isWritable(bbArg, *this);

  // Not a bufferizable op: The conservative answer is "not writable".
  return false;
}

OneShotAnalysisState::Extension::~Extension() = default;

//===----------------------------------------------------------------------===//
// Bufferization-specific alias analysis.
//===----------------------------------------------------------------------===//

/// Return true if opOperand has been decided to bufferize in-place.
static bool isInplaceMemoryWrite(OpOperand &opOperand,
                                 const BufferizationAliasInfo &aliasInfo,
                                 const AnalysisState &state) {
  // OpOperands that do not bufferize to a memory write do not write in-place.
  if (!state.bufferizesToMemoryWrite(opOperand))
    return false;
  // Check current bufferization decisions.
  return aliasInfo.isInPlace(opOperand);
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

/// Return `true` if op dominance can be used to rule out read-after-write
/// conflicts wrt. the given reads and writes.
///
/// Op dominance can often be used to rule out potential conflicts such as
/// "read" happens before "write". E.g., the following IR is not a RaW conflict
/// because the the read happens *before* the write.
///
/// %0 = ... : tensor<?xf32>
/// "reading_op"(%0) : tensor<?xf32>
/// %1 = "writing_op"(%0) : tensor<?xf32> -> tensor<?xf32>
///
/// This is no longer true inside loops (or repetitive regions). In such cases,
/// there may not be a meaningful `happensBefore` relationship because ops
/// could be executed multiple times. E.g.:
///
/// %0 = ... : tensor<?xf32>
/// scf.for ... {
///   "reading_op"(%0) : tensor<?xf32>
///   %1 = "writing_op"(%0) : tensor<?xf32> -> tensor<?xf32>
///   ...
/// }
///
/// In the above example, reading_op happens before writing_op according to
/// op dominance. However, both ops may happen multiple times; in
/// particular, the second execution of reading_op happens after the first
/// execution of writing_op. This is problematic because the tensor %0 they
/// operate on (i.e., the "definition") is defined outside of the loop.
///
/// Counter example:
///
/// scf.for ... {
///   %0 = ... : tensor<?xf32>
///   "reading_op"(%0) : tensor<?xf32>
///   %1 = "writing_op"(%0) : tensor<?xf32> -> tensor<?xf32>
///   ...
/// }
///
/// In this example, the definition %0 is in the same repetitive region as
/// "writing_op", so op dominance can be used to compute the `happensBefore`
/// relationship.
///
/// Whether op dominance can be used or not is decided as follows: Find the
/// closest enclosing repetitive region of all buffer writes wrt. the given
/// tensor reads and writes. (The given sets of reads and writes contain the
/// entire alias set.) In case of a read, we look at the op that defines the
/// read value. In case of a write, we look at the op that is writing. If all of
/// those ops are in the same closest enclosing repetitive region (nullptr in
/// case of "no repetitive region" found at all), then op dominance can be used.
/// Otherwise, it cannot be used.
///
/// Example: The common enclosing repetitive region is the scf.for loop.
///          Op dominance can be used.
/// scf.for ... {
///   %0 = tensor.generate
///   "read"(%0)
/// }
///
/// Example: The common enclosing repetitive region is nullptr: There is no
///          repetitive region around the tensor.generate. Op dominance can be
///          used.
/// %0 = tensor.generate
/// scf.for ... { "read"(%0) }
///
/// Example: The common enclosing repetitive regions of tensor.generate and
///          "write" differ. Op dominance cannot be used.
/// %0 = tensor.generate
/// scf.for ... {
///   "read"(%0)
///   "write"(%0)
/// }
///
/// Example: The common enclosing repetitive regions of tensor.generate and
///          "write" differ, but there is no read of %0, so op dominance can be
///          used.
/// %0 = tensor.generate
/// scf.for ... {
///   "write"(%0)
/// }
///
/// Note: iter_args of loops are not aliases of their respective block
/// arguments, so op domanice can be used when analyzing ops that operate
/// on them.
bool canUseOpDominance(const DenseSet<OpOperand *> &usesRead,
                       const DenseSet<OpOperand *> &usesWrite,
                       const AnalysisState &state) {
  const BufferizationOptions &options = state.getOptions();
  std::optional<Region *> commonEnclosingRegion;

  // In case of a write, take the region in which the write takes place.
  for (OpOperand *uWrite : usesWrite) {
    Region *r = getEnclosingRepetitiveRegion(uWrite->getOwner(), options);
    if (!commonEnclosingRegion.has_value()) {
      commonEnclosingRegion = r;
      continue;
    }
    if (*commonEnclosingRegion != r)
      return false;
  }

  // In case of a read, take the region which the read value is defined.
  for (OpOperand *uRead : usesRead) {
    // Optimization: Skip reads of values that have no defined contents.
    if (!state.bufferizesToMemoryWrite(uRead->get()))
      continue;
    Region *r = getEnclosingRepetitiveRegion(uRead->get(), options);
    if (!commonEnclosingRegion.has_value()) {
      commonEnclosingRegion = r;
      continue;
    }
    if (*commonEnclosingRegion != r)
      return false;
  }

  return commonEnclosingRegion.has_value();
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

  if (auto opResult = definition.dyn_cast<OpResult>()) {
    std::string defAttr =
        id + "[DEF: result " + std::to_string(opResult.getResultNumber()) + "]";
    opResult.getDefiningOp()->setAttr(defAttr, b.getUnitAttr());
  } else {
    auto bbArg = definition.cast<BlockArgument>();
    std::string defAttr =
        id + "[DEF: bbArg " + std::to_string(bbArg.getArgNumber()) + "]";
    bbArg.getOwner()->getParentOp()->setAttr(defAttr, b.getUnitAttr());
  }
}

/// Given sets of uses and writes, return true if there is a RaW conflict under
/// the assumption that all given reads/writes alias the same buffer and that
/// all given writes bufferize inplace.
///
/// A conflict is: According to SSA use-def chains, a read R is supposed to read
/// the result of a definition W1. But because of bufferization decisions, R
/// actually reads another definition W2.
static bool hasReadAfterWriteInterference(
    const DenseSet<OpOperand *> &usesRead,
    const DenseSet<OpOperand *> &usesWrite, const DominanceInfo &domInfo,
    AnalysisState &state, const BufferizationAliasInfo &aliasInfo) {
  const BufferizationOptions &options = state.getOptions();

  // Check if op dominance can be used to rule out read-after-write conflicts.
  bool useDominance = canUseOpDominance(usesRead, usesWrite, state);
  LLVM_DEBUG(llvm::dbgs() << "\n- useDominance = " << useDominance << "\n");

  for (OpOperand *uRead : usesRead) {
    Operation *readingOp = uRead->getOwner();

    // Find most recent writes of uRead by following the SSA use-def chain.
    // E.g.:
    //
    // %0 = "writing_op"(%t) : tensor<?x32> -> tensor<?xf32>
    // %1 = "aliasing_op"(%0) : tensor<?x32> -> tensor<?xf32>
    // %2 = "reading_op"(%1) : : tensor<?x32> -> not_a_tensor_type
    //
    // In the above example, if uRead is the OpOperand of reading_op, the
    // definition is %0. Note that operations that create an alias but do not
    // bufferize to a memory write (such as ExtractSliceOp) are skipped.
    SetVector<Value> definitions = state.findDefinitions(uRead->get());

    // Look for conflicting memory writes. Potential conflicts are writes to an
    // alias that have been decided to bufferize inplace.
    for (OpOperand *uConflictingWrite : usesWrite) {
      LLVM_DEBUG(llvm::dbgs() << "\n- check conflict:\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "  uRead = operand " << uRead->getOperandNumber() << " of "
                 << *uRead->getOwner() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "  unConflictingWrite = operand "
                              << uConflictingWrite->getOperandNumber() << " of "
                              << *uConflictingWrite->getOwner() << "\n");

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
        if (Operation *writingOp = definition.getDefiningOp()) {
          if (happensBefore(conflictingWritingOp, writingOp, domInfo)) {
            // conflictingWritingOp happens before writingOp. No conflict.
            LLVM_DEBUG(llvm::dbgs()
                       << "    no conflict: write happens before definition\n");
            continue;
          }
          // No conflict if conflictingWritingOp is contained in writingOp.
          if (writingOp->isProperAncestor(conflictingWritingOp)) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "    no conflict: write is contained in definition\n");
            continue;
          }
        } else {
          auto bbArg = definition.cast<BlockArgument>();
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
        SmallVector<OpResult> aliasingOpResult =
            state.getAliasingOpResult(*uConflictingWrite);
        if (aliasingOpResult.size() == 1 && aliasingOpResult[0] == definition) {
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
                                     const BufferizationAliasInfo &aliasInfo,
                                     const AnalysisState &state) {
  aliasInfo.applyOnAliases(root, [&](Value alias) {
    for (auto &use : alias.getUses())
      // Inplace write to a value that aliases root.
      if (isInplaceMemoryWrite(use, aliasInfo, state))
        res.insert(&use);
  });
}

// Helper function to iterate on aliases of `root` and capture the reads.
static void getAliasingReads(DenseSet<OpOperand *> &res, Value root,
                             const BufferizationAliasInfo &aliasInfo,
                             const AnalysisState &state) {
  aliasInfo.applyOnAliases(root, [&](Value alias) {
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
        SmallVector<OpResult> opResults = state.getAliasingOpResult(use);
        if (llvm::any_of(opResults,
                         [&](OpResult r) { return state.isValueRead(r); }))
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
    OpOperand &operand, const DominanceInfo &domInfo, AnalysisState &state,
    const BufferizationAliasInfo &aliasInfo,
    bool checkConsistencyOnly = false) {
  // Collect reads and writes of all aliases of OpOperand and OpResult.
  DenseSet<OpOperand *> usesRead, usesWrite;
  getAliasingReads(usesRead, operand.get(), aliasInfo, state);
  getAliasingInplaceWrites(usesWrite, operand.get(), aliasInfo, state);
  for (OpResult result : state.getAliasingOpResult(operand)) {
    getAliasingReads(usesRead, result, aliasInfo, state);
    getAliasingInplaceWrites(usesWrite, result, aliasInfo, state);
  }
  if (!checkConsistencyOnly && state.bufferizesToMemoryWrite(operand))
    usesWrite.insert(&operand);

  return hasReadAfterWriteInterference(usesRead, usesWrite, domInfo, state,
                                       aliasInfo);
}

/// Annotate IR with details about the detected non-writability conflict.
static void annotateNonWritableTensor(Value value) {
  static int64_t counter = 0;
  OpBuilder b(value.getContext());
  std::string id = "W_" + std::to_string(counter++);
  if (auto opResult = value.dyn_cast<OpResult>()) {
    std::string attr = id + "[NOT-WRITABLE: result " +
                       std::to_string(opResult.getResultNumber()) + "]";
    opResult.getDefiningOp()->setAttr(attr, b.getUnitAttr());
  } else {
    auto bbArg = value.cast<BlockArgument>();
    std::string attr = id + "[NOT-WRITABLE: bbArg " +
                       std::to_string(bbArg.getArgNumber()) + "]";
    bbArg.getOwner()->getParentOp()->setAttr(attr, b.getUnitAttr());
  }
}

/// Check the reverse SSA use-def chain (following aliasing OpOperands) for
/// non-writable tensor values. Stop searching when an out-of-place bufferized
/// OpOperand was found (or when the OpOperand was not bufferized yet).
/// `currentOpOperand` is assumed to be in-place, even if that decision was not
/// materialized in `aliasInfo` yet.
static bool
hasPrecedingAliasingNonWritableTensor(Value value, OpOperand *currentOpOperand,
                                      const BufferizationAliasInfo &aliasInfo,
                                      const OneShotAnalysisState &state) {
  SmallVector<Value> worklist;
  worklist.push_back(value);
  while (!worklist.empty()) {
    Value nextVal = worklist.pop_back_val();
    if (!state.isWritable(nextVal)) {
      if (state.getOptions().printConflicts)
        annotateNonWritableTensor(nextVal);
      return true;
    }

    // If `nextVal` is not a BlockArgument: End of use-def chain reached.
    auto opResult = nextVal.dyn_cast<OpResult>();
    if (!opResult)
      continue;

    // Follow reverse SSA use-def chain.
    SmallVector<OpOperand *> aliasingOpOperands =
        state.getAliasingOpOperand(opResult);
    for (OpOperand *opOperand : aliasingOpOperands)
      if (aliasInfo.isInPlace(*opOperand) || currentOpOperand == opOperand)
        worklist.push_back(opOperand->get());
  }
  return false;
}

/// Return true if bufferizing `operand` inplace would create a write to a
/// non-writable buffer.
static bool wouldCreateWriteToNonWritableBuffer(
    OpOperand &operand, const BufferizationAliasInfo &aliasInfo,
    OneShotAnalysisState &state, bool checkConsistencyOnly = false) {
  // Collect writes of all aliases of OpOperand and OpResult.
  DenseSet<OpOperand *> usesWrite;
  getAliasingInplaceWrites(usesWrite, operand.get(), aliasInfo, state);
  for (OpResult result : state.getAliasingOpResult(operand)) {
    getAliasingInplaceWrites(usesWrite, result, aliasInfo, state);
  }
  if (!checkConsistencyOnly && state.bufferizesToMemoryWrite(operand))
    usesWrite.insert(&operand);

  // Assuming that `operand` bufferizes in-place: For each write (to each
  // alias), check if there is a non-writable tensor in the reverse SSA use-def
  // chain.
  for (OpOperand *uWrite : usesWrite) {
    if (hasPrecedingAliasingNonWritableTensor(uWrite->get(), &operand,
                                              aliasInfo, state)) {
      LLVM_DEBUG(llvm::dbgs() << "=> NOT WRITABLE\n");
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Bufferization analyses.
//===----------------------------------------------------------------------===//

/// Determine if `operand` can be bufferized in-place.
static LogicalResult bufferizableInPlaceAnalysisImpl(
    OpOperand &operand, BufferizationAliasInfo &aliasInfo,
    OneShotAnalysisState &state, const DominanceInfo &domInfo) {
  LLVM_DEBUG(
      llvm::dbgs() << "//===-------------------------------------------===//\n"
                   << "Analyzing operand #" << operand.getOperandNumber()
                   << " of " << *operand.getOwner() << "\n");

  bool foundInterference =
      wouldCreateWriteToNonWritableBuffer(operand, aliasInfo, state) ||
      wouldCreateReadAfterWriteInterference(operand, domInfo, state, aliasInfo);

  if (foundInterference)
    aliasInfo.bufferizeOutOfPlace(operand);
  else
    aliasInfo.bufferizeInPlace(operand, state);

  LLVM_DEBUG(llvm::dbgs()
             << "//===-------------------------------------------===//\n");
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
                                     OneShotAnalysisState &state,
                                     const DominanceInfo &domInfo,
                                     unsigned analysisFuzzerSeed = 0) {
  if (analysisFuzzerSeed) {
    // This is a fuzzer. For testing purposes only. Randomize the order in which
    // operations are analyzed. The bufferization quality is likely worse, but
    // we want to make sure that no assertions are triggered anywhere.
    std::mt19937 g(analysisFuzzerSeed);
    llvm::shuffle(ops.begin(), ops.end(), g);
  }

  // Analyze a single op.
  auto analyzeOp = [&](Operation *op) {
    for (OpOperand &opOperand : op->getOpOperands())
      if (opOperand.get().getType().isa<TensorType>())
        if (failed(bufferizableInPlaceAnalysisImpl(opOperand, aliasInfo, state,
                                                   domInfo)))
          return failure();
    return success();
  };

  OneShotBufferizationOptions::AnalysisHeuristic heuristic =
      state.getOptions().analysisHeuristic;
  if (heuristic == OneShotBufferizationOptions::AnalysisHeuristic::BottomUp) {
    // Default: Walk ops in reverse for better interference analysis.
    for (Operation *op : reverse(ops))
      if (failed(analyzeOp(op)))
        return failure();
  } else if (heuristic ==
             OneShotBufferizationOptions::AnalysisHeuristic::TopDown) {
    for (Operation *op : ops)
      if (failed(analyzeOp(op)))
        return failure();
  } else {
    llvm_unreachable("unsupported heuristic");
  }

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
                                     OneShotAnalysisState &state,
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
                                AnalysisState &state) {
  for (Operation *op : ops)
    if (auto bufferizableOp = state.getOptions().dynCastBufferizableOp(op))
      for (OpResult opResult : op->getOpResults())
        if (opResult.getType().isa<TensorType>())
          for (OpOperand *opOperand :
               bufferizableOp.getAliasingOpOperand(opResult, state))
            if (state.isInPlace(*opOperand))
              if (bufferizableOp.bufferRelation(opResult, state) ==
                  BufferRelation::Equivalent)
                aliasInfo.unionEquivalenceClasses(opResult, opOperand->get());
}

/// Analyze equivalence of tied OpResult/OpOperand pairs of all ops contained
/// in `op`.
static void equivalenceAnalysis(Operation *op,
                                BufferizationAliasInfo &aliasInfo,
                                AnalysisState &state) {
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
                          AnalysisState &state,
                          const BufferizationAliasInfo &aliasInfo) {
  const BufferizationOptions &options = state.getOptions();

  WalkResult walkResult = op->walk([&](BufferizableOpInterface op) {
    // Skip ops that are not in the filter.
    if (!options.isOpAllowed(op.getOperation()))
      return WalkResult::advance();

    // Input IR may not contain any ToMemrefOps. These are not supported because
    // the analysis cannot follow the data flow through memrefs.
    if (isa<ToMemrefOp>(op.getOperation())) {
      op->emitError("to_memref ops not supported during One-Shot Analysis");
      return WalkResult::interrupt();
    }

    for (OpOperand &opOperand : op->getOpOperands()) {
      if (opOperand.get().getType().isa<TensorType>()) {
        if (wouldCreateReadAfterWriteInterference(
                opOperand, domInfo, state, aliasInfo,
                /*checkConsistencyOnly=*/true)) {
          // This error can happen if certain "mustBufferizeInPlace" interface
          // methods are implemented incorrectly, such that the IR already has
          // a RaW conflict before making any bufferization decisions.
          op->emitError("input IR has RaW conflict");
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
                                    const BufferizationAliasInfo &aliasInfo,
                                    const BufferizationOptions &options) {
  // Add __inplace_operands_attr__.
  op->walk([&](Operation *op) {
    for (OpOperand &opOperand : op->getOpOperands())
      if (opOperand.get().getType().isa<TensorType>())
        setInPlaceOpOperand(opOperand, aliasInfo.isInPlace(opOperand));
  });
}

/// Assert that every allocation can be deallocated in the same block. I.e.,
/// every value that is returned or yielded from a block is:
/// * guaranteed to be aliasing a bbArg of that block or a parent block, or
/// * guaranteed to be aliasing an OpResult of a op in a parent block.
///
/// In that case, buffer deallocation is simple: Every allocated buffer can be
/// deallocated in the same block. Otherwise, the buffer deallocation pass must
/// be run.
///
/// Note: The current implementation checks for equivalent values instead of
/// aliasing values, which is stricter than needed. We can currently not check
/// for aliasing values because the analysis is a maybe-alias analysis and we
/// need a must-alias analysis here.
///
/// Example:
/// ```
/// %0 = "some_op" : tensor<?xf32>
/// %1 = scf.if %c -> (tensor<?xf32>) {
///   scf.yield %0 : tensor<?xf32>
/// } else {
///   %t = linalg.alloc_tensor : tensor<?xf32>
///   scf.yield %t : tensor<?xf32>
/// }
/// ```
///
/// In the above example, the second scf.yield op is problematic because the
/// yielded value %t is defined in the same block as the scf.yield op and
/// and bufferizes to a new allocation.
// TODO: Remove buffer deallocation from One-Shot Bufferize and fix the buffer
// deallocation pass.
static LogicalResult assertNoAllocsReturned(Operation *op,
                                            const BufferizationOptions &options,
                                            BufferizationAliasInfo &aliasInfo) {
  LogicalResult status = success();
  DominanceInfo domInfo(op);
  op->walk([&](Operation *returnOp) {
    if (!isRegionReturnLike(returnOp) || !options.isOpAllowed(returnOp))
      return WalkResult::advance();

    for (OpOperand &returnValOperand : returnOp->getOpOperands()) {
      Value returnVal = returnValOperand.get();
      // Skip non-tensor values.
      if (!returnVal.getType().isa<TensorType>())
        continue;

      bool foundEquivValue = false;
      aliasInfo.applyOnEquivalenceClass(returnVal, [&](Value equivVal) {
        if (auto bbArg = equivVal.dyn_cast<BlockArgument>()) {
          Operation *definingOp = bbArg.getOwner()->getParentOp();
          if (definingOp->isProperAncestor(returnOp))
            foundEquivValue = true;
          return;
        }

        Operation *definingOp = equivVal.getDefiningOp();
        if (definingOp->getBlock()->findAncestorOpInBlock(
                *returnOp->getParentOp()))
          // Skip ops that happen after `returnOp` and parent ops.
          if (happensBefore(definingOp, returnOp, domInfo))
            foundEquivValue = true;
      });

      // Note: Returning/yielding buffer allocations is allowed only if
      // `allowReturnAllocs` is set.
      if (!foundEquivValue)
        status = returnOp->emitError()
                 << "operand #" << returnValOperand.getOperandNumber()
                 << " may return/yield a new buffer allocation";
    }

    return WalkResult::advance();
  });

  return status;
}

LogicalResult bufferization::analyzeOp(Operation *op,
                                       OneShotAnalysisState &state,
                                       BufferizationStatistics *statistics) {
  DominanceInfo domInfo(op);
  BufferizationAliasInfo &aliasInfo = state.getAliasInfo();
  const OneShotBufferizationOptions &options = state.getOptions();

  if (failed(checkAliasInfoConsistency(op, domInfo, state, aliasInfo)))
    return failure();

  // If the analysis fails, just return.
  if (failed(inPlaceAnalysis(op, aliasInfo, state, domInfo,
                             options.analysisFuzzerSeed)))
    return failure();

  if (statistics) {
    statistics->numTensorInPlace = aliasInfo.getStatNumTensorInPlace();
    statistics->numTensorOutOfPlace = aliasInfo.getStatNumTensorOutOfPlace();
  }

  equivalenceAnalysis(op, aliasInfo, state);

  bool failedAnalysis = false;
  if (!options.allowReturnAllocs)
    failedAnalysis |= failed(assertNoAllocsReturned(op, options, aliasInfo));

  // Gather some extra analysis data.
  state.gatherYieldedTensors(op);
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
    annotateOpsWithBufferizationMarkers(op, aliasInfo, options);

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
  return bufferizeOp(op, options, /*copyBeforeWrite=*/options.copyBeforeWrite,
                     /*opFilter=*/nullptr, statistics);
}
