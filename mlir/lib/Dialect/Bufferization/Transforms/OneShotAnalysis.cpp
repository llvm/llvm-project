//===- OneShotAnalysis.cpp - One-Shot (Single Pass) Analysis --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One-Shot Analysis analyzes function bodies. Function boundaries (FuncOp
// bbArgs, CallOps, ReturnOps) are treated as "unknown" ops.
// ModuleBufferization.cpp is an extension of One-Shot Analysis for simple
// call graphs.
//
// One-Shot Bufferize consists of two phases.
//
// 1. Analyze ops to decide which OpResults can bufferize inplace, i.e., without
//    inserting buffer copies. The analysis queries op bufferization semantics
//    via `BufferizableOpInterface`.
// 2. Bufferize ops by calling `BufferizableOpInterface::bufferize`. This
//    function does not generate buffer copies for OpResults that were decided
//    to bufferize inplace during the analysis phase.
//
// This file contains only the analysis. The actual bufferization is implemented
// via `bufferizeOp` (Bufferize.h). For convenience, this file also contains a
// helper function `runOneShotBufferize` that analyzes an op (and its nested
// ops) and then bufferizes it.
//
// Inplace bufferization decisions are passed from the analysis to the
// bufferization phase via `AnalysisState` and `BufferizationAliasInfo`.
// They can be printed for debugging purposes with `testAnalysisOnly`.
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

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/TensorCopyInsertion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::bufferization;

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
  markInPlace(operand);
  for (OpResult result : state.getAliasingOpResult(operand))
    aliasInfo.unionSets(result, operand.get());
}

/// Set the inPlace bufferization spec to false.
void BufferizationAliasInfo::bufferizeOutOfPlace(OpOperand &operand) {
  assert(!inplaceBufferized.contains(&operand) &&
         "OpOperand was already decided to bufferize inplace");
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
    : AnalysisState(options), aliasInfo(op) {
  // Set up alias sets for OpResults that must bufferize in-place. This should
  // be done before making any other bufferization decisions.
  op->walk([&](BufferizableOpInterface bufferizableOp) {
    if (!options.isOpAllowed(bufferizableOp))
      return WalkResult::skip();
    for (OpOperand &opOperand : bufferizableOp->getOpOperands()) {
      if (opOperand.get().getType().isa<TensorType>())
        if (bufferizableOp.mustBufferizeInPlace(opOperand, *this)) {
          for (OpResult opResult :
               bufferizableOp.getAliasingOpResult(opOperand, *this))
            aliasInfo.unionAliasSets(opOperand.get(), opResult);
          aliasInfo.markInPlace(opOperand);
        }
    }
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

      // If there is no preceding memory write, the tensor contents are
      // undefined.
      // Note: If `findLastPrecedingWrite` reaches the end of the reverse SSA
      // use-def chain, it returns that value, regardless of whether it is a
      // memory write or not.
      SetVector<Value> lastWrites = findLastPrecedingWrite(opResult);
      bool isUndefined = llvm::none_of(lastWrites, [&](Value lastWrite) {
        if (auto bufferizableOp = getOptions().dynCastBufferizableOp(lastWrite))
          return bufferizableOp.isMemoryWrite(lastWrite.cast<OpResult>(),
                                              *this);
        return true;
      });
      if (isUndefined)
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

/// For each given value, find the closest enclosing repetitive region. If this
/// is the same region for each value, return it. Otherwise return None.
/// Note: If there is no enclosing repetitive region, return nullptr.
static Optional<Region *>
getCommonEnclosingRepetitiveRegion(ArrayRef<Value> values) {
  if (values.empty())
    return None;
  Region *r = getEnclosingRepetitiveRegion(values.front());
  for (Value value : values.drop_front())
    if (getEnclosingRepetitiveRegion(value) != r)
      return None;
  return r;
}

/// Return `true` if the given tensor value is a memory write. Most values are
/// tensor writes, but ops that define a tensor SSA value without specifying its
/// contents (e.g., alloc_tensor) are not.
static bool isMemoryWrite(Value value, const AnalysisState &state) {
  auto opResult = value.dyn_cast<OpResult>();
  if (!opResult)
    return true;
  auto bufferizableOp = state.getOptions().dynCastBufferizableOp(value);
  if (!bufferizableOp)
    return true;
  return bufferizableOp.isMemoryWrite(opResult, state);
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
    AnalysisState &state, const BufferizationAliasInfo &aliasInfo) {
  const BufferizationOptions &options = state.getOptions();

  // Gather all written aliases. Skip over aliases that are not actual writes.
  SmallVector<Value> writtenAliases;
  for (OpOperand *uWrite : usesWrite)
    if (isMemoryWrite(uWrite->get(), state))
      writtenAliases.push_back(uWrite->get());
  // Find the inner-most enclosing repetitive region of each alias. If this is
  // the same region for every alias, save it in `repetitiveRegionOfWrites`.
  Optional<Region *> repetitiveRegionOfWrites =
      getCommonEnclosingRepetitiveRegion(writtenAliases);

  for (OpOperand *uRead : usesRead) {
    Operation *readingOp = uRead->getOwner();

    // Find most recent writes of uRead by following the SSA use-def chain.
    // E.g.:
    //
    // %0 = "writing_op"(%t) : tensor<?x32> -> tensor<?xf32>
    // %1 = "aliasing_op"(%0) : tensor<?x32> -> tensor<?xf32>
    // %2 = "reading_op"(%1) : : tensor<?x32> -> not_a_tensor_type
    //
    // In the above example, if uRead is the OpOperand of reading_op, lastWrite
    // is %0. Note that operations that create an alias but do not write (such
    // as ExtractSliceOp) are skipped.
    SetVector<Value> lastWrites = state.findLastPrecedingWrite(uRead->get());

    // Look for conflicting memory writes. Potential conflicts are writes to an
    // alias that have been decided to bufferize inplace.
    for (OpOperand *uConflictingWrite : usesWrite) {
      // Throughout this loop, check for multiple requirements that have to be
      // met for uConflictingWrite to be an actual conflict.
      Operation *conflictingWritingOp = uConflictingWrite->getOwner();

      // Check if conflictingWritingOp is in the same repetitive region as all
      // written aliases. If this is not the case, there is no meaningful
      // `happensBefore` relationship because conflictingWritingOp may be
      // executed multiple times. E.g.:
      //
      // %0 = ... : tensor<?xf32>
      // scf.for ... {
      //   "reading_op"(%0) : tensor<?xf32>
      //   %1 = "writing_op"(%0) : tensor<?xf32> -> tensor<?xf32>
      //   ...
      // }
      //
      // In the above example, reading_op happens before writing_op according to
      // op dominance. However, both ops may happen multiple times; in
      // particular, the second execution of reading_op happens after the first
      // execution of writing_op. This is problematic if the tensor they operate
      // on (%0) is defined outside of the loop.
      //
      // Counter example:
      //
      // scf.for ... {
      //   %0 = ... : tensor<?xf32>
      //   "reading_op"(%0) : tensor<?xf32>
      //   %1 = "writing_op"(%0) : tensor<?xf32> -> tensor<?xf32>
      //   ...
      // }
      //
      // In this example, %0 is in the same repetitive region as
      // conflictingWritingOp, so op dominance can be used to compute the
      // `happensBefore` relationship.
      //
      // Note: iter_args of loops are not aliases of their respective block
      // arguments, so op domanice can be used when analyzing ops that operate
      // on them.
      //
      // Note: If `writtenAliases` is empty, there are no memory writes outside
      // of the repetitive region of conflictingWritingOp, which means that all
      // relevant aliases are inside the same repetitive region.
      bool canUseOpDominance =
          writtenAliases.empty() ||
          repetitiveRegionOfWrites ==
              getEnclosingRepetitiveRegion(conflictingWritingOp);

      // No conflict if the readingOp dominates conflictingWritingOp, i.e., the
      // write is not visible when reading.
      //
      // Note: If ops are executed multiple times (e.g., because they are inside
      //       a loop), there may be no meaningful `happensBefore` relationship.
      if (canUseOpDominance &&
          happensBefore(readingOp, conflictingWritingOp, domInfo))
        continue;

      // No conflict if the reading use equals the use of the conflicting write.
      // A use cannot conflict with itself.
      //
      // Note: Just being the same op is not enough. It has to be the same use.
      // Note: If the op is executed multiple times (e.g., because it is inside
      //       a loop), it may be conflicting with itself.
      if (canUseOpDominance && uConflictingWrite == uRead)
        continue;

      // No conflict if the op interface says so.
      if (auto bufferizableOp = options.dynCastBufferizableOp(readingOp))
        if (bufferizableOp.isNotConflicting(uRead, uConflictingWrite, state))
          continue;

      if (conflictingWritingOp != readingOp)
        if (auto bufferizableOp =
                options.dynCastBufferizableOp(conflictingWritingOp))
          if (bufferizableOp.isNotConflicting(uRead, uConflictingWrite, state))
            continue;

      // Ops are not conflicting if they are in mutually exclusive regions.
      //
      // Note: If ops are executed multiple times (e.g., because they are inside
      //       a loop), mutually exclusive regions may be executed multiple
      //       times.
      if (canUseOpDominance &&
          insideMutuallyExclusiveRegions(readingOp, conflictingWritingOp))
        continue;

      // Check all possible last writes.
      for (Value lastWrite : lastWrites) {
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
        SmallVector<OpResult> aliasingOpResult =
            state.getAliasingOpResult(*uConflictingWrite);
        if (aliasingOpResult.size() == 1 && aliasingOpResult[0] == lastWrite)
          continue;

        // All requirements are met. Conflict found!

        if (options.printConflicts)
          annotateConflict(uRead, uConflictingWrite, lastWrite);

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
    for (auto &use : alias.getUses())
      // Read to a value that aliases root.
      if (state.bufferizesToMemoryRead(use))
        res.insert(&use);
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
    if (!state.isWritable(nextVal))
      return true;

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
  for (OpOperand *uWrite : usesWrite)
    if (hasPrecedingAliasingNonWritableTensor(uWrite->get(), &operand,
                                              aliasInfo, state))
      return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Bufferization analyses.
//===----------------------------------------------------------------------===//

/// Determine if `operand` can be bufferized in-place.
static LogicalResult bufferizableInPlaceAnalysisImpl(
    OpOperand &operand, BufferizationAliasInfo &aliasInfo,
    OneShotAnalysisState &state, const DominanceInfo &domInfo) {
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
                                    AnalysisState &state) {
  op->walk([&](Operation *op) {
    if (auto bufferizableOp = state.getOptions().dynCastBufferizableOp(op))
      for (OpOperand &opOperand : op->getOpOperands())
        if (opOperand.get().getType().isa<TensorType>())
          setInPlaceOpOperand(opOperand, aliasInfo.isInPlace(opOperand));
  });
}

/// Assert that IR is in destination-passing style. I.e., every value that is
/// returned or yielded from a block is:
/// * aliasing a bbArg of that block or a parent block, or
/// * aliasing an OpResult of a op in a parent block.
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
/// In the above example, the first scf.yield op satifies destination-passing
/// style because the yielded value %0 is defined in the parent block. The
/// second scf.yield op does not satisfy destination-passing style because the
/// yielded value %t is defined in the same block as the scf.yield op.
// TODO: The current implementation checks for equivalent values instead of
// aliasing values, which is stricter than needed. We can currently not check
// for aliasing values because the analysis is a maybe-alias analysis and we
// need a must-alias analysis here.
static LogicalResult
assertDestinationPassingStyle(Operation *op, AnalysisState &state,
                              BufferizationAliasInfo &aliasInfo,
                              SmallVector<Operation *> &newOps) {
  LogicalResult status = success();
  DominanceInfo domInfo(op);
  op->walk([&](Operation *returnOp) {
    if (!isRegionReturnLike(returnOp) ||
        !state.getOptions().isOpAllowed(returnOp))
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

      if (!foundEquivValue)
        status =
            returnOp->emitError()
            << "operand #" << returnValOperand.getOperandNumber()
            << " of ReturnLike op does not satisfy destination passing style";
    }

    return WalkResult::advance();
  });

  return status;
}

LogicalResult bufferization::analyzeOp(Operation *op,
                                       OneShotAnalysisState &state) {
  DominanceInfo domInfo(op);
  BufferizationAliasInfo &aliasInfo = state.getAliasInfo();
  const auto &options =
      static_cast<const OneShotBufferizationOptions &>(state.getOptions());

  // Catch incorrect API usage.
  assert((state.hasDialectState(func::FuncDialect::getDialectNamespace()) ||
          !options.bufferizeFunctionBoundaries) &&
         "must use ModuleBufferize to bufferize function boundaries");

  if (failed(checkAliasInfoConsistency(op, domInfo, state, aliasInfo)))
    return failure();

  // If the analysis fails, just return.
  if (failed(inPlaceAnalysis(op, aliasInfo, state, domInfo,
                             options.analysisFuzzerSeed)))
    return failure();
  equivalenceAnalysis(op, aliasInfo, state);

  bool failedAnalysis = false;
  if (!options.allowReturnAllocs) {
    SmallVector<Operation *> newOps;
    failedAnalysis |=
        failed(assertDestinationPassingStyle(op, state, aliasInfo, newOps));
  }

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
    annotateOpsWithBufferizationMarkers(op, aliasInfo, state);

  return success(!failedAnalysis);
}

LogicalResult
bufferization::runOneShotBufferize(Operation *op,
                                   const OneShotBufferizationOptions &options) {
  OneShotAnalysisState state(op, options);
  if (failed(insertTensorCopies(op, options)))
    return failure();
  if (options.testAnalysisOnly)
    return success();
  return bufferizeOp(op, options, /*copyBeforeWrite=*/false);
}
