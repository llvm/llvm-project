//===- BufferizableOpInterface.cpp - Comprehensive Bufferize --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.cpp.inc"

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#define DEBUG_TYPE "bufferizable-op-interface"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X)

using namespace mlir;
using namespace linalg::comprehensive_bufferize;

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

  // Set up alias sets for OpResults that must bufferize in-place. This should
  // be done before making any other bufferization decisions.
  rootOp->walk([&](BufferizableOpInterface bufferizableOp) {
    for (OpResult opResult : bufferizableOp->getOpResults()) {
      if (opResult.getType().isa<TensorType>())
        if (bufferizableOp.mustBufferizeInPlace(opResult)) {
          SmallVector<OpOperand *> operands =
              bufferizableOp.getAliasingOpOperand(opResult);
          assert(!operands.empty() &&
                 "expected that OpResult has aliasing OpOperand");
          for (OpOperand *operand : operands)
            aliasInfo.unionSets(operand->get(), opResult);
          markInPlace(opResult);
        }
    }
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

bool BufferizationAliasInfo::bufferizesToWritableMemory(Value v) const {
  return bufferizeToWritableMemory.count(v) > 0;
}

/// Specify that the value is known to bufferize to writable memory.
void BufferizationAliasInfo::setBufferizesToWritableMemory(Value v) {
  bufferizeToWritableMemory.insert(v);
}

/// Return `true` if a value was marked as in-place bufferized.
bool BufferizationAliasInfo::isInPlace(OpResult opResult) const {
  bool inplace = inplaceBufferized.contains(opResult);
#ifndef NDEBUG
  if (inplace) {
    auto bufferizableOp =
        dyn_cast<BufferizableOpInterface>(opResult.getDefiningOp());
    assert(bufferizableOp &&
           "expected that in-place bufferized op is bufferizable");
    SmallVector<OpOperand *> operands =
        bufferizableOp.getAliasingOpOperand(opResult);
    for (OpOperand *operand : operands)
      assert(areAliasingBufferizedValues(operand->get(), opResult) &&
             "expected that in-place bufferized OpResult aliases with "
             "aliasing OpOperand");
  }
#endif // NDEBUG
  return inplace;
}

/// Set the inPlace bufferization spec to true.
void BufferizationAliasInfo::bufferizeInPlace(OpResult result,
                                              OpOperand &operand) {
  LLVM_DEBUG(llvm::dbgs() << "bufferizeInPlace: ");
  LLVM_DEBUG(result.print(llvm::dbgs()));

  markInPlace(result);
  aliasInfo.unionSets(result, operand.get());
  if (bufferRelation(operand) == BufferRelation::Equivalent)
    equivalentInfo.unionSets(result, operand.get());
}

/// Set the inPlace bufferization spec to false.
void BufferizationAliasInfo::bufferizeOutOfPlace(OpResult result) {
  LLVM_DEBUG(llvm::dbgs() << "bufferizeOutOfPlace: ");
  LLVM_DEBUG(result.print(llvm::dbgs()));

  if (inplaceBufferized.contains(result))
    inplaceBufferized.erase(result);
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
// Helper functions for BufferizableOpInterface
//===----------------------------------------------------------------------===//

/// Determine which OpOperand* will alias with `result` if the op is bufferized
/// in place. Return an empty vector if the op is not bufferizable.
SmallVector<OpOperand *>
mlir::linalg::comprehensive_bufferize::getAliasingOpOperand(OpResult result) {
  if (Operation *op = result.getDefiningOp())
    if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op))
      return bufferizableOp.getAliasingOpOperand(result);
  return {};
}

/// Determine which OpResult will alias with `opOperand` if the op is bufferized
/// in place. Return an empty OpResult if the op is not bufferizable.
OpResult mlir::linalg::comprehensive_bufferize::getAliasingOpResult(
    OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.getAliasingOpResult(opOperand);
  return OpResult();
}

/// Return true if `opOperand` bufferizes to a memory read. Return `true` if the
/// op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::bufferizesToMemoryRead(
    OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToMemoryRead(opOperand);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return true.
  return true;
}

/// Return true if `opOperand` bufferizes to a memory write. Return
/// `true` if the op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::bufferizesToMemoryWrite(
    OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToMemoryWrite(opOperand);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return true.
  return true;
}

/// Return true if `opOperand` does neither read nor write but bufferizes to an
/// alias. Return false if the op is not bufferizable.
bool mlir::linalg::comprehensive_bufferize::bufferizesToAliasOnly(
    OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferizesToAliasOnly(opOperand);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return false.
  return false;
}

/// Return true if the given value is read by an op that bufferizes to a memory
/// read. Also takes into account ops that create an alias but do not read by
/// themselves (e.g., ExtractSliceOp).
bool mlir::linalg::comprehensive_bufferize::isValueRead(Value value) {
  SmallVector<OpOperand *> workingSet;
  for (OpOperand &use : value.getUses())
    workingSet.push_back(&use);

  while (!workingSet.empty()) {
    OpOperand *uMaybeReading = workingSet.pop_back_val();
    // Skip over all ops that neither read nor write (but create an alias).
    if (bufferizesToAliasOnly(*uMaybeReading))
      for (OpOperand &use : getAliasingOpResult(*uMaybeReading).getUses())
        workingSet.push_back(&use);
    if (bufferizesToMemoryRead(*uMaybeReading))
      return true;
  }

  return false;
}

/// Return the relationship between the operand and the its corresponding
/// OpResult that it may alias with. Return None if the op is not bufferizable.
BufferRelation
mlir::linalg::comprehensive_bufferize::bufferRelation(OpOperand &opOperand) {
  if (auto bufferizableOp =
          dyn_cast<BufferizableOpInterface>(opOperand.getOwner()))
    return bufferizableOp.bufferRelation(opOperand);

  // Unknown op that returns a tensor. The inplace analysis does not support it.
  // Conservatively return None.
  return BufferRelation::None;
}
