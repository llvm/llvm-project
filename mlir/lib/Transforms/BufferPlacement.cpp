//===- BufferPlacement.cpp - the impl for buffer placement ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for computing correct alloc and dealloc positions.
// The main class is the BufferPlacementPass class that implements the
// underlying algorithm. In order to put allocations and deallocations at safe
// positions, it is significantly important to put them into the correct blocks.
// However, the liveness analysis does not pay attention to aliases, which can
// occur due to branches (and their associated block arguments) in general. For
// this purpose, BufferPlacement firstly finds all possible aliases for a single
// value (using the BufferPlacementAliasAnalysis class). Consider the following
// example:
//
// ^bb0(%arg0):
//   cond_br %cond, ^bb1, ^bb2
// ^bb1:
//   br ^exit(%arg0)
// ^bb2:
//   %new_value = ...
//   br ^exit(%new_value)
// ^exit(%arg1):
//   return %arg1;
//
// Using liveness information on its own would cause us to place the allocs and
// deallocs in the wrong block. This is due to the fact that %new_value will not
// be liveOut of its block. Instead, we have to place the alloc for %new_value
// in bb0 and its associated dealloc in exit. Using the class
// BufferPlacementAliasAnalysis, we will find out that %new_value has a
// potential alias %arg1. In order to find the dealloc position we have to find
// all potential aliases, iterate over their uses and find the common
// post-dominator block. In this block we can safely be sure that %new_value
// will die and can use liveness information to determine the exact operation
// after which we have to insert the dealloc. Finding the alloc position is
// highly similar and non- obvious. Again, we have to consider all potential
// aliases and find the common dominator block to place the alloc.
//
// TODO:
// The current implementation does not support loops and the resulting code will
// be invalid with respect to program semantics. The only thing that is
// currently missing is a high-level loop analysis that allows us to move allocs
// and deallocs outside of the loop blocks. Furthermore, it doesn't also accept
// functions which return buffers already.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/BufferPlacement.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// BufferPlacementAliasAnalysis
//===----------------------------------------------------------------------===//

/// A straight-forward alias analysis which ensures that all aliases of all
/// values will be determined. This is a requirement for the BufferPlacement
/// class since you need to determine safe positions to place alloc and
/// deallocs.
class BufferPlacementAliasAnalysis {
public:
  using ValueSetT = SmallPtrSet<Value, 16>;

public:
  /// Constructs a new alias analysis using the op provided.
  BufferPlacementAliasAnalysis(Operation *op) { build(op->getRegions()); }

  /// Finds all immediate and indirect aliases this value could potentially
  /// have. Note that the resulting set will also contain the value provided as
  /// it is an alias of itself.
  ValueSetT resolve(Value value) const {
    ValueSetT result;
    resolveRecursive(value, result);
    return result;
  }

private:
  /// Recursively determines alias information for the given value. It stores
  /// all newly found potential aliases in the given result set.
  void resolveRecursive(Value value, ValueSetT &result) const {
    if (!result.insert(value).second)
      return;
    auto it = aliases.find(value);
    if (it == aliases.end())
      return;
    for (Value alias : it->second)
      resolveRecursive(alias, result);
  }

  /// This function constructs a mapping from values to its immediate aliases.
  /// It iterates over all blocks, gets their predecessors, determines the
  /// values that will be passed to the corresponding block arguments and
  /// inserts them into the underlying map.
  void build(MutableArrayRef<Region> regions) {
    for (Region &region : regions) {
      for (Block &block : region) {
        // Iterate over all predecessor and get the mapped values to their
        // corresponding block arguments values.
        for (auto it = block.pred_begin(), e = block.pred_end(); it != e;
             ++it) {
          unsigned successorIndex = it.getSuccessorIndex();
          // Get the terminator and the values that will be passed to our block.
          auto branchInterface =
              dyn_cast<BranchOpInterface>((*it)->getTerminator());
          if (!branchInterface)
            continue;
          // Query the branch op interace to get the successor operands.
          auto successorOperands =
              branchInterface.getSuccessorOperands(successorIndex);
          if (successorOperands.hasValue()) {
            // Build the actual mapping of values to their immediate aliases.
            for (auto argPair : llvm::zip(block.getArguments(),
                                          successorOperands.getValue())) {
              aliases[std::get<1>(argPair)].insert(std::get<0>(argPair));
            }
          }
        }
      }
    }
  }

  /// Maps values to all immediate aliases this value can have.
  llvm::DenseMap<Value, ValueSetT> aliases;
};

//===----------------------------------------------------------------------===//
// BufferPlacementPositions
//===----------------------------------------------------------------------===//

/// Stores correct alloc and dealloc positions to place dialect-specific alloc
/// and dealloc operations.
struct BufferPlacementPositions {
public:
  BufferPlacementPositions()
      : allocPosition(nullptr), deallocPosition(nullptr) {}

  /// Creates a new positions tuple including alloc and dealloc positions.
  BufferPlacementPositions(Operation *allocPosition, Operation *deallocPosition)
      : allocPosition(allocPosition), deallocPosition(deallocPosition) {}

  /// Returns the alloc position before which the alloc operation has to be
  /// inserted.
  Operation *getAllocPosition() const { return allocPosition; }

  /// Returns the dealloc position after which the dealloc operation has to be
  /// inserted.
  Operation *getDeallocPosition() const { return deallocPosition; }

private:
  Operation *allocPosition;
  Operation *deallocPosition;
};

//===----------------------------------------------------------------------===//
// BufferPlacementAnalysis
//===----------------------------------------------------------------------===//

// The main buffer placement analysis used to place allocs and deallocs.
class BufferPlacementAnalysis {
public:
  using DeallocSetT = SmallPtrSet<Operation *, 2>;

public:
  BufferPlacementAnalysis(Operation *op)
      : operation(op), liveness(op), dominators(op), postDominators(op),
        aliases(op) {}

  /// Computes the actual positions to place allocs and deallocs for the given
  /// value.
  BufferPlacementPositions
  computeAllocAndDeallocPositions(OpResult result) const {
    if (result.use_empty())
      return BufferPlacementPositions(result.getOwner(), result.getOwner());
    // Get all possible aliases.
    auto possibleValues = aliases.resolve(result);
    return BufferPlacementPositions(getAllocPosition(result, possibleValues),
                                    getDeallocPosition(result, possibleValues));
  }

  /// Finds all associated dealloc nodes for the alloc nodes using alias
  /// information.
  DeallocSetT findAssociatedDeallocs(OpResult allocResult) const {
    DeallocSetT result;
    auto possibleValues = aliases.resolve(allocResult);
    for (Value alias : possibleValues)
      for (Operation *op : alias.getUsers()) {
        // Check for an existing memory effect interface.
        auto effectInstance = dyn_cast<MemoryEffectOpInterface>(op);
        if (!effectInstance)
          continue;
        // Check whether the associated value will be freed using the current
        // operation.
        SmallVector<MemoryEffects::EffectInstance, 2> effects;
        effectInstance.getEffectsOnValue(alias, effects);
        if (llvm::any_of(effects, [=](MemoryEffects::EffectInstance &it) {
              return isa<MemoryEffects::Free>(it.getEffect());
            }))
          result.insert(op);
      }
    return result;
  }

  /// Dumps the buffer placement information to the given stream.
  void print(raw_ostream &os) const {
    os << "// ---- Buffer Placement -----\n";

    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (Operation &operation : block)
          for (OpResult result : operation.getResults()) {
            BufferPlacementPositions positions =
                computeAllocAndDeallocPositions(result);
            os << "Positions for ";
            result.print(os);
            os << "\n Alloc: ";
            positions.getAllocPosition()->print(os);
            os << "\n Dealloc: ";
            positions.getDeallocPosition()->print(os);
            os << "\n";
          }
  }

private:
  /// Finds a correct placement block to store alloc/dealloc node according to
  /// the algorithm described at the top of the file. It supports dominator and
  /// post-dominator analyses via template arguments.
  template <typename DominatorT>
  Block *
  findPlacementBlock(OpResult result,
                     const BufferPlacementAliasAnalysis::ValueSetT &aliases,
                     const DominatorT &doms) const {
    // Start with the current block the value is defined in.
    Block *dom = result.getOwner()->getBlock();
    // Iterate over all aliases and their uses to find a safe placement block
    // according to the given dominator information.
    for (Value alias : aliases)
      for (Operation *user : alias.getUsers()) {
        // Move upwards in the dominator tree to find an appropriate
        // dominator block that takes the current use into account.
        dom = doms.findNearestCommonDominator(dom, user->getBlock());
      }
    return dom;
  }

  /// Finds a correct alloc position according to the algorithm described at
  /// the top of the file.
  Operation *getAllocPosition(
      OpResult result,
      const BufferPlacementAliasAnalysis::ValueSetT &aliases) const {
    // Determine the actual block to place the alloc and get liveness
    // information.
    Block *placementBlock = findPlacementBlock(result, aliases, dominators);
    const LivenessBlockInfo *livenessInfo =
        liveness.getLiveness(placementBlock);

    // We have to ensure that the alloc will be before the first use of all
    // aliases of the given value. We first assume that there are no uses in the
    // placementBlock and that we can safely place the alloc before the
    // terminator at the end of the block.
    Operation *startOperation = placementBlock->getTerminator();
    // Iterate over all aliases and ensure that the startOperation will point to
    // the first operation of all potential aliases in the placementBlock.
    for (Value alias : aliases) {
      Operation *aliasStartOperation = livenessInfo->getStartOperation(alias);
      // Check whether the aliasStartOperation lies in the desired block and
      // whether it is before the current startOperation. If yes, this will be
      // the new startOperation.
      if (aliasStartOperation->getBlock() == placementBlock &&
          aliasStartOperation->isBeforeInBlock(startOperation))
        startOperation = aliasStartOperation;
    }
    // startOperation is the first operation before which we can safely store
    // the alloc taking all potential aliases into account.
    return startOperation;
  }

  /// Finds a correct dealloc position according to the algorithm described at
  /// the top of the file.
  Operation *getDeallocPosition(
      OpResult result,
      const BufferPlacementAliasAnalysis::ValueSetT &aliases) const {
    // Determine the actual block to place the dealloc and get liveness
    // information.
    Block *placementBlock = findPlacementBlock(result, aliases, postDominators);
    const LivenessBlockInfo *livenessInfo =
        liveness.getLiveness(placementBlock);

    // We have to ensure that the dealloc will be after the last use of all
    // aliases of the given value. We first assume that there are no uses in the
    // placementBlock and that we can safely place the dealloc at the beginning.
    Operation *endOperation = &placementBlock->front();
    // Iterate over all aliases and ensure that the endOperation will point to
    // the last operation of all potential aliases in the placementBlock.
    for (Value alias : aliases) {
      Operation *aliasEndOperation =
          livenessInfo->getEndOperation(alias, endOperation);
      // Check whether the aliasEndOperation lies in the desired block and
      // whether it is behind the current endOperation. If yes, this will be the
      // new endOperation.
      if (aliasEndOperation->getBlock() == placementBlock &&
          endOperation->isBeforeInBlock(aliasEndOperation))
        endOperation = aliasEndOperation;
    }
    // endOperation is the last operation behind which we can safely store the
    // dealloc taking all potential aliases into account.
    return endOperation;
  }

  /// The operation this transformation was constructed from.
  Operation *operation;

  /// The underlying liveness analysis to compute fine grained information about
  /// alloc and dealloc positions.
  Liveness liveness;

  /// The dominator analysis to place allocs in the appropriate blocks.
  DominanceInfo dominators;

  /// The post dominator analysis to place deallocs in the appropriate blocks.
  PostDominanceInfo postDominators;

  /// The internal alias analysis to ensure that allocs and deallocs take all
  /// their potential aliases into account.
  BufferPlacementAliasAnalysis aliases;
};

//===----------------------------------------------------------------------===//
// BufferPlacementPass
//===----------------------------------------------------------------------===//

/// The actual buffer placement pass that moves alloc and dealloc nodes into
/// the right positions. It uses the algorithm described at the top of the file.
struct BufferPlacementPass
    : mlir::PassWrapper<BufferPlacementPass, FunctionPass> {
  void runOnFunction() override {
    // Get required analysis information first.
    auto &analysis = getAnalysis<BufferPlacementAnalysis>();

    // Compute an initial placement of all nodes.
    llvm::SmallVector<std::pair<OpResult, BufferPlacementPositions>, 16>
        placements;
    getFunction().walk([&](MemoryEffectOpInterface op) {
      // Try to find a single allocation result.
      SmallVector<MemoryEffects::EffectInstance, 2> effects;
      op.getEffects(effects);

      SmallVector<MemoryEffects::EffectInstance, 2> allocateResultEffects;
      llvm::copy_if(effects, std::back_inserter(allocateResultEffects),
                    [=](MemoryEffects::EffectInstance &it) {
                      Value value = it.getValue();
                      return isa<MemoryEffects::Allocate>(it.getEffect()) &&
                             value && value.isa<OpResult>();
                    });
      // If there is one result only, we will be able to move the allocation and
      // (possibly existing) deallocation ops.
      if (allocateResultEffects.size() == 1) {
        // Insert allocation result.
        auto allocResult = allocateResultEffects[0].getValue().cast<OpResult>();
        placements.emplace_back(
            allocResult, analysis.computeAllocAndDeallocPositions(allocResult));
      }
    });

    // Move alloc (and dealloc - if any) nodes into the right places and insert
    // dealloc nodes if necessary.
    for (auto &entry : placements) {
      // Find already associated dealloc nodes.
      OpResult alloc = entry.first;
      auto deallocs = analysis.findAssociatedDeallocs(alloc);
      if (deallocs.size() > 1) {
        emitError(alloc.getLoc(),
                  "not supported number of associated dealloc operations");
        return;
      }

      // Move alloc node to the right place.
      BufferPlacementPositions &positions = entry.second;
      Operation *allocOperation = alloc.getOwner();
      allocOperation->moveBefore(positions.getAllocPosition());

      // If there is an existing dealloc, move it to the right place.
      Operation *nextOp = positions.getDeallocPosition()->getNextNode();
      // If the Dealloc position is at the terminator operation of the block,
      // then the value should escape from a deallocation.
      if (!nextOp) {
        assert(deallocs.empty() &&
               "There should be no dealloc for the returned buffer");
        continue;
      }
      if (deallocs.size()) {
        (*deallocs.begin())->moveBefore(nextOp);
      } else {
        // If there is no dealloc node, insert one in the right place.
        OpBuilder builder(nextOp);
        builder.create<DeallocOp>(allocOperation->getLoc(), alloc);
      }
    }
  };
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BufferAssignmentPlacer
//===----------------------------------------------------------------------===//

/// Creates a new assignment placer.
BufferAssignmentPlacer::BufferAssignmentPlacer(Operation *op) : operation(op) {}

/// Computes the actual position to place allocs for the given value.
OpBuilder::InsertPoint
BufferAssignmentPlacer::computeAllocPosition(OpResult result) {
  Operation *owner = result.getOwner();
  return OpBuilder::InsertPoint(owner->getBlock(), Block::iterator(owner));
}

//===----------------------------------------------------------------------===//
// FunctionAndBlockSignatureConverter
//===----------------------------------------------------------------------===//

// Performs the actual signature rewriting step.
LogicalResult FunctionAndBlockSignatureConverter::matchAndRewrite(
    FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!converter) {
    funcOp.emitError("The type converter has not been defined for "
                     "FunctionAndBlockSignatureConverter");
    return failure();
  }
  auto funcType = funcOp.getType();

  // Convert function arguments using the provided TypeConverter.
  TypeConverter::SignatureConversion conversion(funcType.getNumInputs());
  for (auto argType : llvm::enumerate(funcType.getInputs()))
    conversion.addInputs(argType.index(),
                         converter->convertType(argType.value()));

  // If a function result type is not a memref but it would be a memref after
  // type conversion, a new argument should be appended to the function
  // arguments list for this result. Otherwise, it remains unchanged as a
  // function result.
  SmallVector<Type, 2> newResultTypes;
  newResultTypes.reserve(funcOp.getNumResults());
  for (Type resType : funcType.getResults()) {
    Type convertedType = converter->convertType(resType);
    if (BufferAssignmentTypeConverter::isConvertedMemref(convertedType,
                                                         resType))
      conversion.addInputs(convertedType);
    else
      newResultTypes.push_back(convertedType);
  }

  // Update the signature of the function.
  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                            newResultTypes));
    rewriter.applySignatureConversion(&funcOp.getBody(), conversion);
  });
  return success();
}

//===----------------------------------------------------------------------===//
// BufferAssignmentTypeConverter
//===----------------------------------------------------------------------===//

/// Registers conversions into BufferAssignmentTypeConverter
BufferAssignmentTypeConverter::BufferAssignmentTypeConverter() {
  // Keep all types unchanged.
  addConversion([](Type type) { return type; });
  // A type conversion that converts ranked-tensor type to memref type.
  addConversion([](RankedTensorType type) {
    return (Type)MemRefType::get(type.getShape(), type.getElementType());
  });
}

/// Checks if `type` has been converted from non-memref type to memref.
bool BufferAssignmentTypeConverter::isConvertedMemref(Type type, Type before) {
  return type.isa<MemRefType>() && !before.isa<MemRefType>();
}

//===----------------------------------------------------------------------===//
// BufferPlacementPass construction
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::createBufferPlacementPass() {
  return std::make_unique<BufferPlacementPass>();
}
