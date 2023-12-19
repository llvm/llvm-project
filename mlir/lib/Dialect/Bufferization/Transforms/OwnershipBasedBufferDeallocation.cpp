//===- OwnershipBasedBufferDeallocation.cpp - impl. for buffer dealloc. ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for computing correct `bufferization.dealloc`
// positions. Furthermore, buffer deallocation also adds required new clone
// operations to ensure that memrefs returned by functions never alias an
// argument.
//
// TODO:
// The current implementation does not support explicit-control-flow loops and
// the resulting code will be invalid with respect to program semantics.
// However, structured control-flow loops are fully supported.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_OWNERSHIPBASEDBUFFERDEALLOCATION
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value buildBoolValue(OpBuilder &builder, Location loc, bool value) {
  return builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(value));
}

static bool isMemref(Value v) { return v.getType().isa<BaseMemRefType>(); }

//===----------------------------------------------------------------------===//
// Backedges analysis
//===----------------------------------------------------------------------===//

namespace {

/// A straight-forward program analysis which detects loop backedges induced by
/// explicit control flow.
class Backedges {
public:
  using BlockSetT = SmallPtrSet<Block *, 16>;
  using BackedgeSetT = llvm::DenseSet<std::pair<Block *, Block *>>;

public:
  /// Constructs a new backedges analysis using the op provided.
  Backedges(Operation *op) { recurse(op); }

  /// Returns the number of backedges formed by explicit control flow.
  size_t size() const { return edgeSet.size(); }

  /// Returns the start iterator to loop over all backedges.
  BackedgeSetT::const_iterator begin() const { return edgeSet.begin(); }

  /// Returns the end iterator to loop over all backedges.
  BackedgeSetT::const_iterator end() const { return edgeSet.end(); }

private:
  /// Enters the current block and inserts a backedge into the `edgeSet` if we
  /// have already visited the current block. The inserted edge links the given
  /// `predecessor` with the `current` block.
  bool enter(Block &current, Block *predecessor) {
    bool inserted = visited.insert(&current).second;
    if (!inserted)
      edgeSet.insert(std::make_pair(predecessor, &current));
    return inserted;
  }

  /// Leaves the current block.
  void exit(Block &current) { visited.erase(&current); }

  /// Recurses into the given operation while taking all attached regions into
  /// account.
  void recurse(Operation *op) {
    Block *current = op->getBlock();
    // If the current op implements the `BranchOpInterface`, there can be
    // cycles in the scope of all successor blocks.
    if (isa<BranchOpInterface>(op)) {
      for (Block *succ : current->getSuccessors())
        recurse(*succ, current);
    }
    // Recurse into all distinct regions and check for explicit control-flow
    // loops.
    for (Region &region : op->getRegions()) {
      if (!region.empty())
        recurse(region.front(), current);
    }
  }

  /// Recurses into explicit control-flow structures that are given by
  /// the successor relation defined on the block level.
  void recurse(Block &block, Block *predecessor) {
    // Try to enter the current block. If this is not possible, we are
    // currently processing this block and can safely return here.
    if (!enter(block, predecessor))
      return;

    // Recurse into all operations and successor blocks.
    for (Operation &op : block.getOperations())
      recurse(&op);

    // Leave the current block.
    exit(block);
  }

  /// Stores all blocks that are currently visited and on the processing stack.
  BlockSetT visited;

  /// Stores all backedges in the format (source, target).
  BackedgeSetT edgeSet;
};

} // namespace

//===----------------------------------------------------------------------===//
// BufferDeallocation
//===----------------------------------------------------------------------===//

namespace {
/// The buffer deallocation transformation which ensures that all allocs in the
/// program have a corresponding de-allocation.
class BufferDeallocation {
public:
  BufferDeallocation(Operation *op, bool privateFuncDynamicOwnership)
      : state(op) {
    options.privateFuncDynamicOwnership = privateFuncDynamicOwnership;
  }

  /// Performs the actual placement/creation of all dealloc operations.
  LogicalResult deallocate(FunctionOpInterface op);

private:
  /// The base case for the recursive template below.
  template <typename... T>
  typename std::enable_if<sizeof...(T) == 0, FailureOr<Operation *>>::type
  handleOp(Operation *op) {
    return op;
  }

  /// Applies all the handlers of the interfaces in the template list
  /// implemented by 'op'. In particular, if an operation implements more than
  /// one of the interfaces in the template list, all the associated handlers
  /// will be applied to the operation in the same order as the template list
  /// specifies. If a handler reports a failure or removes the operation without
  /// replacement (indicated by returning 'nullptr'), no further handlers are
  /// applied and the return value is propagated to the caller of 'handleOp'.
  ///
  /// The interface handlers job is to update the deallocation state, most
  /// importantly the ownership map and list of memrefs to potentially be
  /// deallocated per block, but also to insert `bufferization.dealloc`
  /// operations where needed. Obviously, no MemRefs that may be used at a later
  /// point in the control-flow may be deallocated and the ownership map has to
  /// be updated to reflect potential ownership changes caused by the dealloc
  /// operation (e.g., if two interfaces on the same op insert a dealloc
  /// operation each, the second one should query the ownership map and use them
  /// as deallocation condition such that MemRefs already deallocated in the
  /// first dealloc operation are not deallocated a second time (double-free)).
  /// Note that currently only the interfaces on terminators may insert dealloc
  /// operations and it is verified as a precondition that a terminator op must
  /// implement exactly one of the interfaces handling dealloc insertion.
  ///
  /// The return value of the 'handleInterface' functions should be a
  /// FailureOr<Operation *> indicating whether there was a failure or otherwise
  /// returning the operation itself or a replacement operation.
  ///
  /// Note: The difference compared to `TypeSwitch` is that all
  /// matching cases are applied instead of just the first match.
  template <typename InterfaceT, typename... InterfacesU>
  FailureOr<Operation *> handleOp(Operation *op) {
    Operation *next = op;
    if (auto concreteOp = dyn_cast<InterfaceT>(op)) {
      FailureOr<Operation *> result = handleInterface(concreteOp);
      if (failed(result))
        return failure();
      next = *result;
    }
    if (!next)
      return FailureOr<Operation *>(nullptr);
    return handleOp<InterfacesU...>(next);
  }

  /// Apply all supported interface handlers to the given op.
  FailureOr<Operation *> handleAllInterfaces(Operation *op) {
    if (auto deallocOpInterface = dyn_cast<BufferDeallocationOpInterface>(op))
      return deallocOpInterface.process(state, options);

    if (failed(verifyOperationPreconditions(op)))
      return failure();

    return handleOp<MemoryEffectOpInterface, RegionBranchOpInterface,
                    CallOpInterface, BranchOpInterface,
                    RegionBranchTerminatorOpInterface>(op);
  }

  /// Make sure that for each forwarded MemRef value, an ownership indicator
  /// `i1` value is forwarded as well such that the successor block knows
  /// whether the MemRef has to be deallocated.
  ///
  /// Example:
  /// ```
  /// ^bb1:
  ///   <more ops...>
  ///   cf.br ^bb2(<forward-to-bb2>)
  /// ```
  /// becomes
  /// ```
  /// // let (m, c) = getMemrefsAndConditionsToDeallocate(bb1)
  /// // let r = getMemrefsToRetain(bb1, bb2, <forward-to-bb2>)
  /// ^bb1:
  ///   <more ops...>
  ///   o = bufferization.dealloc m if c retain r
  ///   // replace ownership(r) with o element-wise
  ///   cf.br ^bb2(<forward-to-bb2>, o)
  /// ```
  FailureOr<Operation *> handleInterface(BranchOpInterface op);

  /// Add an ownership indicator for every forwarding MemRef operand and result.
  /// Nested regions never take ownership of MemRefs owned by a parent region
  /// (neither via forwarding operand nor when captured implicitly when the
  /// region is not isolated from above). Ownerships will only be passed to peer
  /// regions (when an operation has multiple regions, such as scf.while), or to
  /// parent regions.
  /// Note that the block arguments in the nested region are currently handled
  /// centrally in the 'dealloc' function, but better interface support could
  /// allow us to do this here for the nested region specifically to reduce the
  /// amount of assumptions we make on the structure of ops implementing this
  /// interface.
  ///
  /// Example:
  /// ```
  /// %ret = scf.for %i = %c0 to %c10 step %c1 iter_args(%m = %memref) {
  ///   <more ops...>
  ///   scf.yield %m : memref<2xi32>, i1
  /// }
  /// ```
  /// becomes
  /// ```
  /// %ret:2 = scf.for %i = %c0 to %c10 step %c1
  ///     iter_args(%m = %memref, %own = %false) {
  ///   <more ops...>
  ///   // Note that the scf.yield is handled by the
  ///   // RegionBranchTerminatorOpInterface (not this handler)
  ///   // let o = getMemrefWithUniqueOwnership(%own)
  ///   scf.yield %m, o : memref<2xi32>, i1
  /// }
  /// ```
  FailureOr<Operation *> handleInterface(RegionBranchOpInterface op);

  /// If the private-function-dynamic-ownership pass option is enabled and the
  /// called function is private, additional arguments and results are added for
  /// each MemRef argument/result to pass the dynamic ownership indicator along.
  /// Otherwise, updates the ownership map and list of memrefs to be deallocated
  /// according to the function boundary ABI, i.e., assume ownership of all
  /// returned MemRefs.
  ///
  /// Example (assume `private-function-dynamic-ownership` is enabled):
  /// ```
  /// func.func @f(%arg0: memref<2xi32>) -> memref<2xi32> {...}
  /// func.func private @g(%arg0: memref<2xi32>) -> memref<2xi32> {...}
  ///
  /// %ret_f = func.call @f(%memref) : (memref<2xi32>) -> memref<2xi32>
  /// %ret_g = func.call @g(%memref) : (memref<2xi32>) -> memref<2xi32>
  /// ```
  /// becomes
  /// ```
  /// func.func @f(%arg0: memref<2xi32>) -> memref<2xi32> {...}
  /// func.func private @g(%arg0: memref<2xi32>) -> memref<2xi32> {...}
  ///
  /// %ret_f = func.call @f(%memref) : (memref<2xi32>) -> memref<2xi32>
  /// // set ownership(%ret_f) := true
  /// // remember to deallocate %ret_f
  ///
  /// // (new_memref, own) = getmemrefWithUniqueOwnership(%memref)
  /// %ret_g:2 = func.call @g(new_memref, own) :
  ///   (memref<2xi32>, i1) -> (memref<2xi32>, i1)
  /// // set ownership(%ret_g#0) := %ret_g#1
  /// // remember to deallocate %ret_g
  /// ```
  FailureOr<Operation *> handleInterface(CallOpInterface op);

  /// Takes care of allocation and free side-effects. It collects allocated
  /// MemRefs that we have to add to manually deallocate, but also removes
  /// values again that are already deallocated before the end of the block. It
  /// also updates the ownership map accordingly.
  ///
  /// Example:
  /// ```
  /// %alloc = memref.alloc()
  /// %alloca = memref.alloca()
  /// ```
  /// becomes
  /// ```
  /// %alloc = memref.alloc()
  /// %alloca = memref.alloca()
  /// // set ownership(alloc) := true
  /// // set ownership(alloca) := false
  /// // remember to deallocate %alloc
  /// ```
  FailureOr<Operation *> handleInterface(MemoryEffectOpInterface op);

  /// Takes care that the function boundary ABI is adhered to if the parent
  /// operation implements FunctionOpInterface, inserting a
  /// `bufferization.clone` if necessary, and inserts the
  /// `bufferization.dealloc` operation according to the ops operands.
  ///
  /// Example:
  /// ```
  /// ^bb1:
  ///   <more ops...>
  ///   func.return <return-vals>
  /// ```
  /// becomes
  /// ```
  /// // let (m, c) = getMemrefsAndConditionsToDeallocate(bb1)
  /// // let r = getMemrefsToRetain(bb1, nullptr, <return-vals>)
  /// ^bb1:
  ///   <more ops...>
  ///   o = bufferization.dealloc m if c retain r
  ///   func.return <return-vals>
  ///     (if !isFunctionWithoutDynamicOwnership: append o)
  /// ```
  FailureOr<Operation *> handleInterface(RegionBranchTerminatorOpInterface op);

  /// Construct a new operation which is exactly the same as the passed 'op'
  /// except that the OpResults list is appended by new results of the passed
  /// 'types'.
  /// TODO: ideally, this would be implemented using an OpInterface because it
  /// is used to append function results, loop iter_args, etc. and thus makes
  /// some assumptions that the variadic list of those is at the end of the
  /// OpResults range.
  Operation *appendOpResults(Operation *op, ArrayRef<Type> types);

  /// A convenience template for the generic 'appendOpResults' function above to
  /// avoid manual casting of the result.
  template <typename OpTy>
  OpTy appendOpResults(OpTy op, ArrayRef<Type> types) {
    return cast<OpTy>(appendOpResults(op.getOperation(), types));
  }

  /// Performs deallocation of a single basic block. This is a private function
  /// because some internal data structures have to be set up beforehand and
  /// this function has to be called on blocks in a region in dominance order.
  LogicalResult deallocate(Block *block);

  /// After all relevant interfaces of an operation have been processed by the
  /// 'handleInterface' functions, this function sets the ownership of operation
  /// results that have not been set yet by the 'handleInterface' functions. It
  /// generally assumes that each result can alias with every operand of the
  /// operation, if there are MemRef typed results but no MemRef operands it
  /// assigns 'false' as ownership. This happens, e.g., for the
  /// memref.get_global operation. It would also be possible to query some alias
  /// analysis to get more precise ownerships, however, the analysis would have
  /// to be updated according to the IR modifications this pass performs (e.g.,
  /// re-building operations to have more result values, inserting clone
  /// operations, etc.).
  void populateRemainingOwnerships(Operation *op);

  /// Given an SSA value of MemRef type, returns the same of a new SSA value
  /// which has 'Unique' ownership where the ownership indicator is guaranteed
  /// to be always 'true'.
  Value materializeMemrefWithGuaranteedOwnership(OpBuilder &builder,
                                                 Value memref, Block *block);

  /// Returns whether the given operation implements FunctionOpInterface, has
  /// private visibility, and the private-function-dynamic-ownership pass option
  /// is enabled.
  bool isFunctionWithoutDynamicOwnership(Operation *op);

  /// Given an SSA value of MemRef type, this function queries the
  /// BufferDeallocationOpInterface of the defining operation of 'memref' for a
  /// materialized ownership indicator for 'memref'.  If the op does not
  /// implement the interface or if the block for which the materialized value
  /// is requested does not match the block in which 'memref' is defined, the
  /// default implementation in
  /// `DeallocationState::getMemrefWithUniqueOwnership` is queried instead.
  std::pair<Value, Value>
  materializeUniqueOwnership(OpBuilder &builder, Value memref, Block *block);

  /// Checks all the preconditions for operations implementing the
  /// FunctionOpInterface that have to hold for the deallocation to be
  /// applicable:
  /// (1) Checks that there are not explicit control flow loops.
  static LogicalResult verifyFunctionPreconditions(FunctionOpInterface op);

  /// Checks all the preconditions for operations inside the region of
  /// operations implementing the FunctionOpInterface that have to hold for the
  /// deallocation to be applicable:
  /// (1) Checks if all operations that have at least one attached region
  /// implement the RegionBranchOpInterface. This is not required in edge cases,
  /// where we have a single attached region and the parent operation has no
  /// results.
  /// (2) Checks that no deallocations already exist. Especially deallocations
  /// in nested regions are not properly supported yet since this requires
  /// ownership of the memref to be transferred to the nested region, which does
  /// not happen by default.  This constrained can be lifted in the future.
  /// (3) Checks that terminators with more than one successor except
  /// `cf.cond_br` are not present and that either BranchOpInterface or
  /// RegionBranchTerminatorOpInterface is implemented.
  static LogicalResult verifyOperationPreconditions(Operation *op);

  /// When the 'private-function-dynamic-ownership' pass option is enabled,
  /// additional `i1` arguments and return values are added for each MemRef
  /// value in the function signature. This function takes care of updating the
  /// `function_type` attribute of the function according to the actually
  /// returned values from the terminators.
  static LogicalResult updateFunctionSignature(FunctionOpInterface op);

private:
  ///  Collects all analysis state and including liveness, caches, ownerships of
  ///  already processed values and operations, and the MemRefs that have to be
  ///  deallocated at the end of each block.
  DeallocationState state;

  /// Collects all pass options in a single place.
  DeallocationOptions options;
};

} // namespace

//===----------------------------------------------------------------------===//
// BufferDeallocation Implementation
//===----------------------------------------------------------------------===//

std::pair<Value, Value>
BufferDeallocation::materializeUniqueOwnership(OpBuilder &builder, Value memref,
                                               Block *block) {
  // The interface can only materialize ownership indicators in the same block
  // as the defining op.
  if (memref.getParentBlock() != block)
    return state.getMemrefWithUniqueOwnership(builder, memref, block);

  Operation *owner = memref.getDefiningOp();
  if (!owner)
    owner = memref.getParentBlock()->getParentOp();

  // If the op implements the interface, query it for a materialized ownership
  // value.
  if (auto deallocOpInterface = dyn_cast<BufferDeallocationOpInterface>(owner))
    return deallocOpInterface.materializeUniqueOwnershipForMemref(
        state, options, builder, memref);

  // Otherwise use the default implementation.
  return state.getMemrefWithUniqueOwnership(builder, memref, block);
}

static bool regionOperatesOnMemrefValues(Region &region) {
  auto checkBlock = [](Block *block) {
    if (llvm::any_of(block->getArguments(), isMemref))
      return WalkResult::interrupt();
    for (Operation &op : *block) {
      if (llvm::any_of(op.getOperands(), isMemref))
        return WalkResult::interrupt();
      if (llvm::any_of(op.getResults(), isMemref))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  WalkResult result = region.walk(checkBlock);
  if (result.wasInterrupted())
    return true;

  // Note: Block::walk/Region::walk visits only blocks that are nested under
  // nested operations, but not direct children.
  for (Block &block : region)
    if (checkBlock(&block).wasInterrupted())
      return true;

  return false;
}

LogicalResult
BufferDeallocation::verifyFunctionPreconditions(FunctionOpInterface op) {
  // (1) Ensure that there are supported loops only (no explicit control flow
  // loops).
  Backedges backedges(op);
  if (backedges.size()) {
    op->emitError("Only structured control-flow loops are supported.");
    return failure();
  }

  return success();
}

LogicalResult BufferDeallocation::verifyOperationPreconditions(Operation *op) {
  // (1) Check that the control flow structures are supported.
  auto regions = op->getRegions();
  // Check that if the operation has at
  // least one region it implements the RegionBranchOpInterface. If there
  // is an operation that does not fulfill this condition, we cannot apply
  // the deallocation steps. Furthermore, we accept cases, where we have a
  // region that returns no results, since, in that case, the intra-region
  // control flow does not affect the transformation.
  size_t size = regions.size();
  if (((size == 1 && !op->getResults().empty()) || size > 1) &&
      !dyn_cast<RegionBranchOpInterface>(op)) {
    if (llvm::any_of(regions, regionOperatesOnMemrefValues))
      return op->emitError("All operations with attached regions need to "
                           "implement the RegionBranchOpInterface.");
  }

  // (2) The pass does not work properly when deallocations are already present.
  // Alternatively, we could also remove all deallocations as a pre-pass.
  if (isa<DeallocOp>(op))
    return op->emitError(
        "No deallocation operations must be present when running this pass!");

  // (3) Check that terminators with more than one successor except `cf.cond_br`
  // are not present and that either BranchOpInterface or
  // RegionBranchTerminatorOpInterface is implemented.
  if (op->hasTrait<OpTrait::NoTerminator>())
    return op->emitError("NoTerminator trait is not supported");

  if (op->hasTrait<OpTrait::IsTerminator>()) {
    // Either one of those interfaces has to be implemented on terminators, but
    // not both.
    if (!isa<BranchOpInterface, RegionBranchTerminatorOpInterface>(op) ||
        (isa<BranchOpInterface>(op) &&
         isa<RegionBranchTerminatorOpInterface>(op)))

      return op->emitError(
          "Terminators must implement either BranchOpInterface or "
          "RegionBranchTerminatorOpInterface (but not both)!");

    // We only support terminators with 0 or 1 successors for now and
    // special-case the conditional branch op.
    if (op->getSuccessors().size() > 1)

      return op->emitError("Terminators with more than one successor "
                           "are not supported!");
  }

  return success();
}

LogicalResult
BufferDeallocation::updateFunctionSignature(FunctionOpInterface op) {
  SmallVector<TypeRange> returnOperandTypes(llvm::map_range(
      op.getFunctionBody().getOps<RegionBranchTerminatorOpInterface>(),
      [](RegionBranchTerminatorOpInterface op) {
        return op.getSuccessorOperands(RegionBranchPoint::parent()).getTypes();
      }));
  if (!llvm::all_equal(returnOperandTypes))
    return op->emitError(
        "there are multiple return operations with different operand types");

  TypeRange resultTypes = op.getResultTypes();
  // Check if we found a return operation because that doesn't necessarily
  // always have to be the case, e.g., consider a function with one block that
  // has a cf.br at the end branching to itself again (i.e., an infinite loop).
  // In that case we don't want to crash but just not update the return types.
  if (!returnOperandTypes.empty())
    resultTypes = returnOperandTypes[0];

  // TODO: it would be nice if the FunctionOpInterface had a method to not only
  // get the function type but also set it.
  op->setAttr(
      "function_type",
      TypeAttr::get(FunctionType::get(
          op->getContext(), op.getFunctionBody().front().getArgumentTypes(),
          resultTypes)));

  return success();
}

LogicalResult BufferDeallocation::deallocate(FunctionOpInterface op) {
  // Stop and emit a proper error message if we don't support the input IR.
  if (failed(verifyFunctionPreconditions(op)))
    return failure();

  // Process the function block by block.
  auto result = op->walk<WalkOrder::PostOrder, ForwardDominanceIterator<>>(
      [&](Block *block) {
        if (failed(deallocate(block)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (result.wasInterrupted())
    return failure();

  // Update the function signature if the function is private, dynamic ownership
  // is enabled, and the function has memrefs as arguments or results.
  return updateFunctionSignature(op);
}

LogicalResult BufferDeallocation::deallocate(Block *block) {
  OpBuilder builder = OpBuilder::atBlockBegin(block);

  // Compute liveness transfers of ownership to this block.
  SmallVector<Value> liveMemrefs;
  state.getLiveMemrefsIn(block, liveMemrefs);
  for (auto li : liveMemrefs) {
    // Ownership of implicitly captured memrefs from other regions is never
    // taken, but ownership of memrefs in the same region (but different block)
    // is taken.
    if (li.getParentRegion() == block->getParent()) {
      state.updateOwnership(li, state.getOwnership(li, li.getParentBlock()),
                            block);
      state.addMemrefToDeallocate(li, block);
      continue;
    }

    if (li.getParentRegion()->isProperAncestor(block->getParent())) {
      Value falseVal = buildBoolValue(builder, li.getLoc(), false);
      state.updateOwnership(li, falseVal, block);
    }
  }

  for (unsigned i = 0, e = block->getNumArguments(); i < e; ++i) {
    BlockArgument arg = block->getArgument(i);
    if (!isMemref(arg))
      continue;

    // Adhere to function boundary ABI: no ownership of function argument
    // MemRefs is taken.
    if (isFunctionWithoutDynamicOwnership(block->getParentOp()) &&
        block->isEntryBlock()) {
      Value newArg = buildBoolValue(builder, arg.getLoc(), false);
      state.updateOwnership(arg, newArg);
      state.addMemrefToDeallocate(arg, block);
      continue;
    }

    // Pass MemRef ownerships along via `i1` values.
    Value newArg = block->addArgument(builder.getI1Type(), arg.getLoc());
    state.updateOwnership(arg, newArg);
    state.addMemrefToDeallocate(arg, block);
  }

  // For each operation in the block, handle the interfaces that affect aliasing
  // and ownership of memrefs.
  for (Operation &op : llvm::make_early_inc_range(*block)) {
    FailureOr<Operation *> result = handleAllInterfaces(&op);
    if (failed(result))
      return failure();
    if (!*result)
      continue;

    populateRemainingOwnerships(*result);
  }

  // TODO: if block has no terminator, handle dealloc insertion here.
  return success();
}

Operation *BufferDeallocation::appendOpResults(Operation *op,
                                               ArrayRef<Type> types) {
  SmallVector<Type> newTypes(op->getResultTypes());
  newTypes.append(types.begin(), types.end());
  auto *newOp = Operation::create(op->getLoc(), op->getName(), newTypes,
                                  op->getOperands(), op->getAttrDictionary(),
                                  op->getPropertiesStorage(),
                                  op->getSuccessors(), op->getNumRegions());
  for (auto [oldRegion, newRegion] :
       llvm::zip(op->getRegions(), newOp->getRegions()))
    newRegion.takeBody(oldRegion);

  OpBuilder(op).insert(newOp);
  op->replaceAllUsesWith(newOp->getResults().take_front(op->getNumResults()));
  op->erase();

  return newOp;
}

FailureOr<Operation *>
BufferDeallocation::handleInterface(RegionBranchOpInterface op) {
  OpBuilder builder = OpBuilder::atBlockBegin(op->getBlock());

  // TODO: the RegionBranchOpInterface does not provide all the necessary
  // methods to perform this transformation without additional assumptions on
  // the structure. In particular, that
  // * additional values to be passed to the next region can be added to the end
  //   of the operand list, the end of the block argument list, and the end of
  //   the result value list. However, it seems to be the general guideline for
  //   operations implementing this interface to follow this structure.
  // * and that the block arguments and result values match the forwarded
  //   operands one-to-one (i.e., that there are no other values appended to the
  //   front).
  // These assumptions are satisfied by the `scf.if`, `scf.for`, and `scf.while`
  // operations.

  SmallVector<RegionSuccessor> regions;
  op.getSuccessorRegions(RegionBranchPoint::parent(), regions);
  assert(!regions.empty() && "Must have at least one successor region");
  SmallVector<Value> entryOperands(
      op.getEntrySuccessorOperands(regions.front()));
  unsigned numMemrefOperands = llvm::count_if(entryOperands, isMemref);

  // No ownership is acquired for any MemRefs that are passed to the region from
  // the outside.
  Value falseVal = buildBoolValue(builder, op.getLoc(), false);
  op->insertOperands(op->getNumOperands(),
                     SmallVector<Value>(numMemrefOperands, falseVal));

  int counter = op->getNumResults();
  unsigned numMemrefResults = llvm::count_if(op->getResults(), isMemref);
  SmallVector<Type> ownershipResults(numMemrefResults, builder.getI1Type());
  RegionBranchOpInterface newOp = appendOpResults(op, ownershipResults);

  for (auto result : llvm::make_filter_range(newOp->getResults(), isMemref)) {
    state.updateOwnership(result, newOp->getResult(counter++));
    state.addMemrefToDeallocate(result, newOp->getBlock());
  }

  return newOp.getOperation();
}

Value BufferDeallocation::materializeMemrefWithGuaranteedOwnership(
    OpBuilder &builder, Value memref, Block *block) {
  // First, make sure we at least have 'Unique' ownership already.
  std::pair<Value, Value> newMemrefAndOnwership =
      materializeUniqueOwnership(builder, memref, block);
  Value newMemref = newMemrefAndOnwership.first;
  Value condition = newMemrefAndOnwership.second;

  // Avoid inserting additional IR if ownership is already guaranteed. In
  // particular, this is already the case when we had 'Unknown' ownership
  // initially and a clone was inserted to get to 'Unique' ownership.
  if (matchPattern(condition, m_One()))
    return newMemref;

  // Insert a runtime check and only clone if we still don't have ownership at
  // runtime.
  Value maybeClone =
      builder
          .create<scf::IfOp>(
              memref.getLoc(), condition,
              [&](OpBuilder &builder, Location loc) {
                builder.create<scf::YieldOp>(loc, newMemref);
              },
              [&](OpBuilder &builder, Location loc) {
                Value clone =
                    builder.create<bufferization::CloneOp>(loc, newMemref);
                builder.create<scf::YieldOp>(loc, clone);
              })
          .getResult(0);
  Value trueVal = buildBoolValue(builder, memref.getLoc(), true);
  state.updateOwnership(maybeClone, trueVal);
  state.addMemrefToDeallocate(maybeClone, maybeClone.getParentBlock());
  return maybeClone;
}

FailureOr<Operation *>
BufferDeallocation::handleInterface(BranchOpInterface op) {
  if (op->getNumSuccessors() > 1)
    return op->emitError("BranchOpInterface operations with multiple "
                         "successors are not supported yet");

  if (op->getNumSuccessors() != 1)
    return emitError(op.getLoc(),
                     "only BranchOpInterface operations with exactly "
                     "one successor are supported yet");

  if (op.getSuccessorOperands(0).getProducedOperandCount() > 0)
    return op.emitError("produced operands are not supported");

  // Collect the values to deallocate and retain and use them to create the
  // dealloc operation.
  Block *block = op->getBlock();
  OpBuilder builder(op);
  SmallVector<Value> memrefs, conditions, toRetain;
  if (failed(state.getMemrefsAndConditionsToDeallocate(
          builder, op.getLoc(), block, memrefs, conditions)))
    return failure();

  OperandRange forwardedOperands =
      op.getSuccessorOperands(0).getForwardedOperands();
  state.getMemrefsToRetain(block, op->getSuccessor(0), forwardedOperands,
                           toRetain);

  auto deallocOp = builder.create<bufferization::DeallocOp>(
      op.getLoc(), memrefs, conditions, toRetain);

  // We want to replace the current ownership of the retained values with the
  // result values of the dealloc operation as they are always unique.
  state.resetOwnerships(deallocOp.getRetained(), block);
  for (auto [retained, ownership] :
       llvm::zip(deallocOp.getRetained(), deallocOp.getUpdatedConditions())) {
    state.updateOwnership(retained, ownership, block);
  }

  unsigned numAdditionalReturns = llvm::count_if(forwardedOperands, isMemref);
  SmallVector<Value> newOperands(forwardedOperands);
  auto additionalConditions =
      deallocOp.getUpdatedConditions().take_front(numAdditionalReturns);
  newOperands.append(additionalConditions.begin(), additionalConditions.end());
  op.getSuccessorOperands(0).getMutableForwardedOperands().assign(newOperands);

  return op.getOperation();
}

FailureOr<Operation *> BufferDeallocation::handleInterface(CallOpInterface op) {
  OpBuilder builder(op);

  // Lookup the function operation and check if it has private visibility. If
  // the function is referenced by SSA value instead of a Symbol, it's assumed
  // to be always private.
  Operation *funcOp = op.resolveCallable(state.getSymbolTable());
  bool isPrivate = true;
  if (auto symbol = dyn_cast<SymbolOpInterface>(funcOp))
    isPrivate = symbol.isPrivate() && !symbol.isDeclaration();

  // If the private-function-dynamic-ownership option is enabled and we are
  // calling a private function, we need to add an additional `i1`
  // argument/result for each MemRef argument/result to dynamically pass the
  // current ownership indicator rather than adhering to the function boundary
  // ABI.
  if (options.privateFuncDynamicOwnership && isPrivate) {
    SmallVector<Value> newOperands, ownershipIndicatorsToAdd;
    for (Value operand : op.getArgOperands()) {
      if (!isMemref(operand)) {
        newOperands.push_back(operand);
        continue;
      }
      auto [memref, condition] =
          materializeUniqueOwnership(builder, operand, op->getBlock());
      newOperands.push_back(memref);
      ownershipIndicatorsToAdd.push_back(condition);
    }
    newOperands.append(ownershipIndicatorsToAdd.begin(),
                       ownershipIndicatorsToAdd.end());
    op.getArgOperandsMutable().assign(newOperands);

    unsigned numMemrefs = llvm::count_if(op->getResults(), isMemref);
    SmallVector<Type> ownershipTypesToAppend(numMemrefs, builder.getI1Type());
    unsigned ownershipCounter = op->getNumResults();
    op = appendOpResults(op, ownershipTypesToAppend);

    for (auto result : llvm::make_filter_range(op->getResults(), isMemref)) {
      state.updateOwnership(result, op->getResult(ownershipCounter++));
      state.addMemrefToDeallocate(result, result.getParentBlock());
    }

    return op.getOperation();
  }

  // According to the function boundary ABI we are guaranteed to get ownership
  // of all MemRefs returned by the function. Thus we set ownership to constant
  // 'true' and remember to deallocate it.
  Value trueVal = buildBoolValue(builder, op.getLoc(), true);
  for (auto result : llvm::make_filter_range(op->getResults(), isMemref)) {
    state.updateOwnership(result, trueVal);
    state.addMemrefToDeallocate(result, result.getParentBlock());
  }

  return op.getOperation();
}

FailureOr<Operation *>
BufferDeallocation::handleInterface(MemoryEffectOpInterface op) {
  auto *block = op->getBlock();
  OpBuilder builder = OpBuilder::atBlockBegin(block);

  for (auto operand : llvm::make_filter_range(op->getOperands(), isMemref)) {
    if (op.getEffectOnValue<MemoryEffects::Free>(operand).has_value()) {
      if (!op->hasAttr(BufferizationDialect::kManualDeallocation))
        return op->emitError(
            "memory free side-effect on MemRef value not supported!");

      // Buffers that were allocated under "manual deallocation" may be
      // manually deallocated. We insert a runtime assertion to cover certain
      // cases of invalid IR where an automatically managed buffer allocation
      // is manually deallocated. This is not a bulletproof check!
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(op);
      Ownership ownership = state.getOwnership(operand, block);
      if (ownership.isUnique()) {
        Value ownershipInverted = builder.create<arith::XOrIOp>(
            op.getLoc(), ownership.getIndicator(),
            buildBoolValue(builder, op.getLoc(), true));
        builder.create<cf::AssertOp>(
            op.getLoc(), ownershipInverted,
            "expected that the block does not have ownership");
      }
    }
  }

  for (auto res : llvm::make_filter_range(op->getResults(), isMemref)) {
    auto allocEffect = op.getEffectOnValue<MemoryEffects::Allocate>(res);
    if (allocEffect.has_value()) {
      if (isa<SideEffects::AutomaticAllocationScopeResource>(
              allocEffect->getResource())) {
        // Make sure that the ownership of auto-managed allocations is set to
        // false. This is important for operations that have at least one memref
        // typed operand. E.g., consider an operation like `bufferization.clone`
        // that lowers to a `memref.alloca + memref.copy` instead of a
        // `memref.alloc`. If we wouldn't set the ownership of the result here,
        // the default ownership population in `populateRemainingOwnerships`
        // would assume aliasing with the MemRef operand.
        state.resetOwnerships(res, block);
        state.updateOwnership(res, buildBoolValue(builder, op.getLoc(), false));
        continue;
      }

      if (op->hasAttr(BufferizationDialect::kManualDeallocation)) {
        // This allocation will be deallocated manually. Assign an ownership of
        // "false", so that it will never be deallocated by the buffer
        // deallocation pass.
        state.resetOwnerships(res, block);
        state.updateOwnership(res, buildBoolValue(builder, op.getLoc(), false));
        continue;
      }

      state.updateOwnership(res, buildBoolValue(builder, op.getLoc(), true));
      state.addMemrefToDeallocate(res, block);
    }
  }

  return op.getOperation();
}

FailureOr<Operation *>
BufferDeallocation::handleInterface(RegionBranchTerminatorOpInterface op) {
  OpBuilder builder(op);

  // If this is a return operation of a function that is not private or the
  // dynamic function boundary ownership is disabled, we need to return memref
  // values for which we have guaranteed ownership to pass on to adhere to the
  // function boundary ABI.
  bool funcWithoutDynamicOwnership =
      isFunctionWithoutDynamicOwnership(op->getParentOp());
  if (funcWithoutDynamicOwnership) {
    for (OpOperand &val : op->getOpOperands()) {
      if (!isMemref(val.get()))
        continue;

      val.set(materializeMemrefWithGuaranteedOwnership(builder, val.get(),
                                                       op->getBlock()));
    }
  }

  // TODO: getSuccessorRegions is not implemented by all operations we care
  // about, but we would need to check how many successors there are and under
  // which condition they are taken, etc.

  MutableOperandRange operands =
      op.getMutableSuccessorOperands(RegionBranchPoint::parent());

  SmallVector<Value> updatedOwnerships;
  auto result = deallocation_impl::insertDeallocOpForReturnLike(
      state, op, OperandRange(operands), updatedOwnerships);
  if (failed(result) || !*result)
    return result;

  // Add an additional operand for every MemRef for the ownership indicator.
  if (!funcWithoutDynamicOwnership) {
    SmallVector<Value> newOperands{OperandRange(operands)};
    newOperands.append(updatedOwnerships.begin(), updatedOwnerships.end());
    operands.assign(newOperands);
  }

  return op.getOperation();
}

bool BufferDeallocation::isFunctionWithoutDynamicOwnership(Operation *op) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  return funcOp && (!options.privateFuncDynamicOwnership ||
                    !funcOp.isPrivate() || funcOp.isExternal());
}

void BufferDeallocation::populateRemainingOwnerships(Operation *op) {
  for (auto res : op->getResults()) {
    if (!isMemref(res))
      continue;
    if (!state.getOwnership(res, op->getBlock()).isUninitialized())
      continue;

    // The op does not allocate memory, otherwise, it would have been assigned
    // an ownership during `handleInterface`. Assume the result may alias with
    // any memref operand and thus combine all their ownerships.
    for (auto operand : op->getOperands()) {
      if (!isMemref(operand))
        continue;

      state.updateOwnership(
          res, state.getOwnership(operand, operand.getParentBlock()),
          op->getBlock());
    }

    // If the ownership value is still uninitialized (e.g., because the op has
    // no memref operands), assume that no ownership is taken. E.g., this is the
    // case for "memref.get_global".
    //
    // Note: This can lead to memory leaks if memory side effects are not
    // properly specified on the op.
    if (state.getOwnership(res, op->getBlock()).isUninitialized()) {
      OpBuilder builder(op);
      state.updateOwnership(res, buildBoolValue(builder, op->getLoc(), false));
    }
  }
}

//===----------------------------------------------------------------------===//
// OwnershipBasedBufferDeallocationPass
//===----------------------------------------------------------------------===//

namespace {

/// The actual buffer deallocation pass that inserts and moves dealloc nodes
/// into the right positions. Furthermore, it inserts additional clones if
/// necessary. It uses the algorithm described at the top of the file.
struct OwnershipBasedBufferDeallocationPass
    : public bufferization::impl::OwnershipBasedBufferDeallocationBase<
          OwnershipBasedBufferDeallocationPass> {
  OwnershipBasedBufferDeallocationPass() = default;
  OwnershipBasedBufferDeallocationPass(bool privateFuncDynamicOwnership)
      : OwnershipBasedBufferDeallocationPass() {
    this->privateFuncDynamicOwnership.setValue(privateFuncDynamicOwnership);
  }
  void runOnOperation() override {
    auto status = getOperation()->walk([&](func::FuncOp func) {
      if (func.isExternal())
        return WalkResult::skip();

      if (failed(deallocateBuffersOwnershipBased(func,
                                                 privateFuncDynamicOwnership)))
        return WalkResult::interrupt();

      return WalkResult::advance();
    });
    if (status.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Implement bufferization API
//===----------------------------------------------------------------------===//

LogicalResult bufferization::deallocateBuffersOwnershipBased(
    FunctionOpInterface op, bool privateFuncDynamicOwnership) {
  // Gather all required allocation nodes and prepare the deallocation phase.
  BufferDeallocation deallocation(op, privateFuncDynamicOwnership);

  // Place all required temporary clone and dealloc nodes.
  return deallocation.deallocate(op);
}

//===----------------------------------------------------------------------===//
// OwnershipBasedBufferDeallocationPass construction
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
mlir::bufferization::createOwnershipBasedBufferDeallocationPass(
    bool privateFuncDynamicOwnership) {
  return std::make_unique<OwnershipBasedBufferDeallocationPass>(
      privateFuncDynamicOwnership);
}
