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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SetOperations.h"

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
/// This class is used to track the ownership of values. The ownership can
/// either be not initialized yet ('Uninitialized' state), set to a unique SSA
/// value which indicates the ownership at runtime (or statically if it is a
/// constant value) ('Unique' state), or it cannot be represented in a single
/// SSA value ('Unknown' state). An artificial example of a case where ownership
/// cannot be represented in a single i1 SSA value could be the following:
/// `%0 = test.non_deterministic_select %arg0, %arg1 : i32`
/// Since the operation does not provide us a separate boolean indicator on
/// which of the two operands was selected, we would need to either insert an
/// alias check at runtime to determine if `%0` aliases with `%arg0` or `%arg1`,
/// or insert a `bufferization.clone` operation to get a fresh buffer which we
/// could assign ownership to.
///
/// The three states this class can represent form a lattice on a partial order:
/// forall X in SSA values. uninitialized < unique(X) < unknown
/// forall X, Y in SSA values.
///   unique(X) == unique(Y) iff X and Y always evaluate to the same value
///   unique(X) != unique(Y) otherwise
class Ownership {
public:
  /// Constructor that creates an 'Uninitialized' ownership. This is needed for
  /// default-construction when used in DenseMap.
  Ownership() = default;

  /// Constructor that creates an 'Unique' ownership. This is a non-explicit
  /// constructor to allow implicit conversion from 'Value'.
  Ownership(Value indicator) : indicator(indicator), state(State::Unique) {}

  /// Get an ownership value in 'Unknown' state.
  static Ownership getUnknown() {
    Ownership unknown;
    unknown.indicator = Value();
    unknown.state = State::Unknown;
    return unknown;
  }
  /// Get an ownership value in 'Unique' state with 'indicator' as parameter.
  static Ownership getUnique(Value indicator) { return Ownership(indicator); }
  /// Get an ownership value in 'Uninitialized' state.
  static Ownership getUninitialized() { return Ownership(); }

  /// Check if this ownership value is in the 'Uninitialized' state.
  bool isUninitialized() const { return state == State::Uninitialized; }
  /// Check if this ownership value is in the 'Unique' state.
  bool isUnique() const { return state == State::Unique; }
  /// Check if this ownership value is in the 'Unknown' state.
  bool isUnknown() const { return state == State::Unknown; }

  /// If this ownership value is in 'Unique' state, this function can be used to
  /// get the indicator parameter. Using this function in any other state is UB.
  Value getIndicator() const {
    assert(isUnique() && "must have unique ownership to get the indicator");
    return indicator;
  }

  /// Get the join of the two-element subset {this,other}. Does not modify
  /// 'this'.
  Ownership getCombined(Ownership other) const {
    if (other.isUninitialized())
      return *this;
    if (isUninitialized())
      return other;

    if (!isUnique() || !other.isUnique())
      return getUnknown();

    // Since we create a new constant i1 value for (almost) each use-site, we
    // should compare the actual value rather than just the SSA Value to avoid
    // unnecessary invalidations.
    if (isEqualConstantIntOrValue(indicator, other.indicator))
      return *this;

    // Return the join of the lattice if the indicator of both ownerships cannot
    // be merged.
    return getUnknown();
  }

  /// Modify 'this' ownership to be the join of the current 'this' and 'other'.
  void combine(Ownership other) { *this = getCombined(other); }

private:
  enum class State {
    Uninitialized,
    Unique,
    Unknown,
  };

  // The indicator value is only relevant in the 'Unique' state.
  Value indicator;
  State state = State::Uninitialized;
};

/// The buffer deallocation transformation which ensures that all allocs in the
/// program have a corresponding de-allocation.
class BufferDeallocation {
public:
  BufferDeallocation(Operation *op, bool privateFuncDynamicOwnership)
      : liveness(op), privateFuncDynamicOwnership(privateFuncDynamicOwnership) {
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
      return nullptr;
    return handleOp<InterfacesU...>(next);
  }

  /// Apply all supported interface handlers to the given op.
  FailureOr<Operation *> handleAllInterfaces(Operation *op) {
    if (failed(verifyOperationPreconditions(op)))
      return failure();

    return handleOp<MemoryEffectOpInterface, RegionBranchOpInterface,
                    CallOpInterface, BranchOpInterface, cf::CondBranchOp,
                    RegionBranchTerminatorOpInterface>(op);
  }

  /// While CondBranchOp also implements the BranchOpInterface, we add a
  /// special-case implementation here because the BranchOpInterface does not
  /// offer all of the functionality we need to insert dealloc operations in an
  /// efficient way. More precisely, there is no way to extract the branch
  /// condition without casting to CondBranchOp specifically. It would still be
  /// possible to implement deallocation for cases where we don't know to which
  /// successor the terminator branches before the actual branch happens by
  /// inserting auxiliary blocks and putting the dealloc op there, however, this
  /// can lead to less efficient code.
  /// This function inserts two dealloc operations (one for each successor) and
  /// adjusts the dealloc conditions according to the branch condition, then the
  /// ownerships of the retained MemRefs are updated by combining the result
  /// values of the two dealloc operations.
  ///
  /// Example:
  /// ```
  /// ^bb1:
  ///   <more ops...>
  ///   cf.cond_br cond, ^bb2(<forward-to-bb2>), ^bb3(<forward-to-bb2>)
  /// ```
  /// becomes
  /// ```
  /// // let (m, c) = getMemrefsAndConditionsToDeallocate(bb1)
  /// // let r0 = getMemrefsToRetain(bb1, bb2, <forward-to-bb2>)
  /// // let r1 = getMemrefsToRetain(bb1, bb3, <forward-to-bb3>)
  /// ^bb1:
  ///   <more ops...>
  ///   let thenCond = map(c, (c) -> arith.andi cond, c)
  ///   let elseCond = map(c, (c) -> arith.andi (arith.xori cond, true), c)
  ///   o0 = bufferization.dealloc m if thenCond retain r0
  ///   o1 = bufferization.dealloc m if elseCond retain r1
  ///   // replace ownership(r0) with o0 element-wise
  ///   // replace ownership(r1) with o1 element-wise
  ///   // let ownership0 := (r) -> o in o0 corresponding to r
  ///   // let ownership1 := (r) -> o in o1 corresponding to r
  ///   // let cmn := intersection(r0, r1)
  ///   foreach (a, b) in zip(map(cmn, ownership0), map(cmn, ownership1)):
  ///     forall r in r0: replace ownership0(r) with arith.select cond, a, b)
  ///     forall r in r1: replace ownership1(r) with arith.select cond, a, b)
  ///   cf.cond_br cond, ^bb2(<forward-to-bb2>, o0), ^bb3(<forward-to-bb3>, o1)
  /// ```
  FailureOr<Operation *> handleInterface(cf::CondBranchOp op);

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

  /// Small helper function to update the ownership map by taking the current
  /// ownership ('Uninitialized' state if not yet present), computing the join
  /// with the passed ownership and storing this new value in the map. By
  /// default, it will be performed for the block where 'owned' is defined. If
  /// the ownership of the given value should be updated for another block, the
  /// 'block' argument can be explicitly passed.
  void joinOwnership(Value owned, Ownership ownership, Block *block = nullptr);

  /// Removes ownerships associated with all values in the passed range for
  /// 'block'.
  void clearOwnershipOf(ValueRange values, Block *block);

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

  /// Given two basic blocks and the values passed via block arguments to the
  /// destination block, compute the list of MemRefs that have to be retained in
  /// the 'fromBlock' to not run into a use-after-free situation.
  /// This list consists of the MemRefs in the successor operand list of the
  /// terminator and the MemRefs in the 'out' set of the liveness analysis
  /// intersected with the 'in' set of the destination block.
  ///
  /// toRetain = filter(successorOperands + (liveOut(fromBlock) insersect
  ///   liveIn(toBlock)), isMemRef)
  void getMemrefsToRetain(Block *fromBlock, Block *toBlock,
                          ValueRange destOperands,
                          SmallVectorImpl<Value> &toRetain) const;

  /// For a given block, computes the list of MemRefs that potentially need to
  /// be deallocated at the end of that block. This list also contains values
  /// that have to be retained (and are thus part of the list returned by
  /// `getMemrefsToRetain`) and is computed by taking the MemRefs in the 'in'
  /// set of the liveness analysis of 'block'  appended by the set of MemRefs
  /// allocated in 'block' itself and subtracted by the set of MemRefs
  /// deallocated in 'block'.
  /// Note that we don't have to take the intersection of the liveness 'in' set
  /// with the 'out' set of the predecessor block because a value that is in the
  /// 'in' set must be defined in an ancestor block that dominates all direct
  /// predecessors and thus the 'in' set of this block is a subset of the 'out'
  /// sets of each predecessor.
  ///
  /// memrefs = filter((liveIn(block) U
  ///   allocated(block) U arguments(block)) \ deallocated(block), isMemRef)
  ///
  /// The list of conditions is then populated by querying the internal
  /// datastructures for the ownership value of that MemRef.
  LogicalResult
  getMemrefsAndConditionsToDeallocate(OpBuilder &builder, Location loc,
                                      Block *block,
                                      SmallVectorImpl<Value> &memrefs,
                                      SmallVectorImpl<Value> &conditions) const;

  /// Given an SSA value of MemRef type, this function queries the ownership and
  /// if it is not already in the 'Unique' state, potentially inserts IR to get
  /// a new SSA value, returned as the first element of the pair, which has
  /// 'Unique' ownership and can be used instead of the passed Value with the
  /// the ownership indicator returned as the second element of the pair.
  std::pair<Value, Value> getMemrefWithUniqueOwnership(OpBuilder &builder,
                                                       Value memref);

  /// Given an SSA value of MemRef type, returns the same of a new SSA value
  /// which has 'Unique' ownership where the ownership indicator is guaranteed
  /// to be always 'true'.
  Value getMemrefWithGuaranteedOwnership(OpBuilder &builder, Value memref);

  /// Returns whether the given operation implements FunctionOpInterface, has
  /// private visibility, and the private-function-dynamic-ownership pass option
  /// is enabled.
  bool isFunctionWithoutDynamicOwnership(Operation *op);

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
  // Mapping from each SSA value with MemRef type to the associated ownership in
  // each block.
  DenseMap<std::pair<Value, Block *>, Ownership> ownershipMap;

  // Collects the list of MemRef values that potentially need to be deallocated
  // per block. It is also fine (albeit not efficient) to add MemRef values that
  // don't have to be deallocated, but only when the ownership is not 'Unknown'.
  DenseMap<Block *, SmallVector<Value>> memrefsToDeallocatePerBlock;

  // Symbol cache to lookup functions from call operations to check attributes
  // on the function operation.
  SymbolTableCollection symbolTable;

  // The underlying liveness analysis to compute fine grained information about
  // alloc and dealloc positions.
  Liveness liveness;

  // A pass option indicating whether private functions should be modified to
  // pass the ownership of MemRef values instead of adhering to the function
  // boundary ABI.
  bool privateFuncDynamicOwnership;
};

} // namespace

//===----------------------------------------------------------------------===//
// BufferDeallocation Implementation
//===----------------------------------------------------------------------===//

void BufferDeallocation::joinOwnership(Value owned, Ownership ownership,
                                       Block *block) {
  // In most cases we care about the block where the value is defined.
  if (block == nullptr)
    block = owned.getParentBlock();

  // Update ownership of current memref itself.
  ownershipMap[{owned, block}].combine(ownership);
}

void BufferDeallocation::clearOwnershipOf(ValueRange values, Block *block) {
  for (Value val : values) {
    ownershipMap[{val, block}] = Ownership::getUninitialized();
  }
}

static bool regionOperatesOnMemrefValues(Region &region) {
  WalkResult result = region.walk([](Block *block) {
    if (llvm::any_of(block->getArguments(), isMemref))
      return WalkResult::interrupt();
    for (Operation &op : *block) {
      if (llvm::any_of(op.getOperands(), isMemref))
        return WalkResult::interrupt();
      if (llvm::any_of(op.getResults(), isMemref))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted();
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
    if (op->getSuccessors().size() > 1 && !isa<cf::CondBranchOp>(op))

      return op->emitError("Terminators with more than one successor "
                           "are not supported (except cf.cond_br)!");
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

void BufferDeallocation::getMemrefsToRetain(
    Block *fromBlock, Block *toBlock, ValueRange destOperands,
    SmallVectorImpl<Value> &toRetain) const {
  for (Value operand : destOperands) {
    if (!isMemref(operand))
      continue;
    toRetain.push_back(operand);
  }

  SmallPtrSet<Value, 16> liveOut;
  for (auto val : liveness.getLiveOut(fromBlock))
    if (isMemref(val))
      liveOut.insert(val);

  if (toBlock)
    llvm::set_intersect(liveOut, liveness.getLiveIn(toBlock));

  // liveOut has non-deterministic order because it was constructed by iterating
  // over a hash-set.
  SmallVector<Value> retainedByLiveness(liveOut.begin(), liveOut.end());
  std::sort(retainedByLiveness.begin(), retainedByLiveness.end(),
            ValueComparator());
  toRetain.append(retainedByLiveness);
}

LogicalResult BufferDeallocation::getMemrefsAndConditionsToDeallocate(
    OpBuilder &builder, Location loc, Block *block,
    SmallVectorImpl<Value> &memrefs, SmallVectorImpl<Value> &conditions) const {

  for (auto [i, memref] :
       llvm::enumerate(memrefsToDeallocatePerBlock.lookup(block))) {
    Ownership ownership = ownershipMap.lookup({memref, block});
    assert(ownership.isUnique() && "MemRef value must have valid ownership");

    // Simply cast unranked MemRefs to ranked memrefs with 0 dimensions such
    // that we can call extract_strided_metadata on it.
    if (auto unrankedMemRefTy = dyn_cast<UnrankedMemRefType>(memref.getType()))
      memref = builder.create<memref::ReinterpretCastOp>(
          loc, MemRefType::get({}, unrankedMemRefTy.getElementType()), memref,
          0, SmallVector<int64_t>{}, SmallVector<int64_t>{});

    // Use the `memref.extract_strided_metadata` operation to get the base
    // memref. This is needed because the same MemRef that was produced by the
    // alloc operation has to be passed to the dealloc operation. Passing
    // subviews, etc. to a dealloc operation is not allowed.
    memrefs.push_back(
        builder.create<memref::ExtractStridedMetadataOp>(loc, memref)
            .getResult(0));
    conditions.push_back(ownership.getIndicator());
  }

  return success();
}

LogicalResult BufferDeallocation::deallocate(Block *block) {
  OpBuilder builder = OpBuilder::atBlockBegin(block);

  // Compute liveness transfers of ownership to this block.
  for (auto li : liveness.getLiveIn(block)) {
    if (!isMemref(li))
      continue;

    // Ownership of implicitly captured memrefs from other regions is never
    // taken, but ownership of memrefs in the same region (but different block)
    // is taken.
    if (li.getParentRegion() == block->getParent()) {
      joinOwnership(li, ownershipMap[{li, li.getParentBlock()}], block);
      memrefsToDeallocatePerBlock[block].push_back(li);
      continue;
    }

    if (li.getParentRegion()->isProperAncestor(block->getParent())) {
      Value falseVal = buildBoolValue(builder, li.getLoc(), false);
      joinOwnership(li, falseVal, block);
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
      joinOwnership(arg, newArg);
      continue;
    }

    // Pass MemRef ownerships along via `i1` values.
    Value newArg = block->addArgument(builder.getI1Type(), arg.getLoc());
    joinOwnership(arg, newArg);
    memrefsToDeallocatePerBlock[block].push_back(arg);
  }

  // For each operation in the block, handle the interfaces that affect aliasing
  // and ownership of memrefs.
  for (Operation &op : llvm::make_early_inc_range(*block)) {
    FailureOr<Operation *> result = handleAllInterfaces(&op);
    if (failed(result))
      return failure();

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
BufferDeallocation::handleInterface(cf::CondBranchOp op) {
  OpBuilder builder(op);

  // The list of memrefs to pass to the `bufferization.dealloc` op as "memrefs
  // to deallocate" in this block is independent of which branch is taken.
  SmallVector<Value> memrefs, ownerships;
  if (failed(getMemrefsAndConditionsToDeallocate(
          builder, op.getLoc(), op->getBlock(), memrefs, ownerships)))
    return failure();

  // Helper lambda to factor out common logic for inserting the dealloc
  // operations for each successor.
  auto insertDeallocForBranch =
      [&](Block *target, MutableOperandRange destOperands,
          ArrayRef<Value> conditions,
          DenseMap<Value, Value> &ownershipMapping) -> DeallocOp {
    SmallVector<Value> toRetain;
    getMemrefsToRetain(op->getBlock(), target, OperandRange(destOperands),
                       toRetain);
    auto deallocOp = builder.create<bufferization::DeallocOp>(
        op.getLoc(), memrefs, conditions, toRetain);
    clearOwnershipOf(deallocOp.getRetained(), op->getBlock());
    for (auto [retained, ownership] :
         llvm::zip(deallocOp.getRetained(), deallocOp.getUpdatedConditions())) {
      joinOwnership(retained, ownership, op->getBlock());
      ownershipMapping[retained] = ownership;
    }
    SmallVector<Value> replacements, ownerships;
    for (Value operand : destOperands) {
      replacements.push_back(operand);
      if (isMemref(operand)) {
        assert(ownershipMapping.contains(operand) &&
               "Should be contained at this point");
        ownerships.push_back(ownershipMapping[operand]);
      }
    }
    replacements.append(ownerships);
    destOperands.assign(replacements);
    return deallocOp;
  };

  // Call the helper lambda and make sure the dealloc conditions are properly
  // modified to reflect the branch condition as well.
  DenseMap<Value, Value> thenOwnershipMap, elseOwnershipMap;

  // Retain `trueDestOperands` if "true" branch is taken.
  SmallVector<Value> thenOwnerships(
      llvm::map_range(ownerships, [&](Value cond) {
        return builder.create<arith::AndIOp>(op.getLoc(), cond,
                                             op.getCondition());
      }));
  DeallocOp thenTakenDeallocOp =
      insertDeallocForBranch(op.getTrueDest(), op.getTrueDestOperandsMutable(),
                             thenOwnerships, thenOwnershipMap);

  // Retain `elseDestOperands` if "false" branch is taken.
  SmallVector<Value> elseOwnerships(
      llvm::map_range(ownerships, [&](Value cond) {
        Value trueVal = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getBoolAttr(true));
        Value negation = builder.create<arith::XOrIOp>(op.getLoc(), trueVal,
                                                       op.getCondition());
        return builder.create<arith::AndIOp>(op.getLoc(), cond, negation);
      }));
  DeallocOp elseTakenDeallocOp = insertDeallocForBranch(
      op.getFalseDest(), op.getFalseDestOperandsMutable(), elseOwnerships,
      elseOwnershipMap);

  // We specifically need to update the ownerships of values that are retained
  // in both dealloc operations again to get a combined 'Unique' ownership
  // instead of an 'Unknown' ownership.
  SmallPtrSet<Value, 16> thenValues(thenTakenDeallocOp.getRetained().begin(),
                                    thenTakenDeallocOp.getRetained().end());
  SetVector<Value> commonValues;
  for (Value val : elseTakenDeallocOp.getRetained()) {
    if (thenValues.contains(val))
      commonValues.insert(val);
  }

  for (Value retained : commonValues) {
    clearOwnershipOf(retained, op->getBlock());
    Value combinedOwnership = builder.create<arith::SelectOp>(
        op.getLoc(), op.getCondition(), thenOwnershipMap[retained],
        elseOwnershipMap[retained]);
    joinOwnership(retained, combinedOwnership, op->getBlock());
  }

  return op.getOperation();
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
    joinOwnership(result, newOp->getResult(counter++));
    memrefsToDeallocatePerBlock[newOp->getBlock()].push_back(result);
  }

  return newOp.getOperation();
}

std::pair<Value, Value>
BufferDeallocation::getMemrefWithUniqueOwnership(OpBuilder &builder,
                                                 Value memref) {
  auto iter = ownershipMap.find({memref, memref.getParentBlock()});
  assert(iter != ownershipMap.end() &&
         "Value must already have been registered in the ownership map");

  Ownership ownership = iter->second;
  if (ownership.isUnique())
    return {memref, ownership.getIndicator()};

  // Instead of inserting a clone operation we could also insert a dealloc
  // operation earlier in the block and use the updated ownerships returned by
  // the op for the retained values. Alternatively, we could insert code to
  // check aliasing at runtime and use this information to combine two unique
  // ownerships more intelligently to not end up with an 'Unknown' ownership in
  // the first place.
  auto cloneOp =
      builder.create<bufferization::CloneOp>(memref.getLoc(), memref);
  Value condition = buildBoolValue(builder, memref.getLoc(), true);
  Value newMemref = cloneOp.getResult();
  joinOwnership(newMemref, condition);
  memrefsToDeallocatePerBlock[newMemref.getParentBlock()].push_back(newMemref);
  return {newMemref, condition};
}

Value BufferDeallocation::getMemrefWithGuaranteedOwnership(OpBuilder &builder,
                                                           Value memref) {
  // First, make sure we at least have 'Unique' ownership already.
  std::pair<Value, Value> newMemrefAndOnwership =
      getMemrefWithUniqueOwnership(builder, memref);
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
  joinOwnership(maybeClone, trueVal);
  memrefsToDeallocatePerBlock[maybeClone.getParentBlock()].push_back(
      maybeClone);
  return maybeClone;
}

FailureOr<Operation *>
BufferDeallocation::handleInterface(BranchOpInterface op) {
  // Skip conditional branches since we special case them for now.
  if (isa<cf::CondBranchOp>(op.getOperation()))
    return op.getOperation();

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
  if (failed(getMemrefsAndConditionsToDeallocate(builder, op.getLoc(), block,
                                                 memrefs, conditions)))
    return failure();

  OperandRange forwardedOperands =
      op.getSuccessorOperands(0).getForwardedOperands();
  getMemrefsToRetain(block, op->getSuccessor(0), forwardedOperands, toRetain);

  auto deallocOp = builder.create<bufferization::DeallocOp>(
      op.getLoc(), memrefs, conditions, toRetain);

  // We want to replace the current ownership of the retained values with the
  // result values of the dealloc operation as they are always unique.
  clearOwnershipOf(deallocOp.getRetained(), block);
  for (auto [retained, ownership] :
       llvm::zip(deallocOp.getRetained(), deallocOp.getUpdatedConditions())) {
    joinOwnership(retained, ownership, block);
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
  Operation *funcOp = op.resolveCallable(&symbolTable);
  bool isPrivate = true;
  if (auto symbol = dyn_cast<SymbolOpInterface>(funcOp))
    isPrivate &= (symbol.getVisibility() == SymbolTable::Visibility::Private);

  // If the private-function-dynamic-ownership option is enabled and we are
  // calling a private function, we need to add an additional `i1`
  // argument/result for each MemRef argument/result to dynamically pass the
  // current ownership indicator rather than adhering to the function boundary
  // ABI.
  if (privateFuncDynamicOwnership && isPrivate) {
    SmallVector<Value> newOperands, ownershipIndicatorsToAdd;
    for (Value operand : op.getArgOperands()) {
      if (!isMemref(operand)) {
        newOperands.push_back(operand);
        continue;
      }
      auto [memref, condition] = getMemrefWithUniqueOwnership(builder, operand);
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
      joinOwnership(result, op->getResult(ownershipCounter++));
      memrefsToDeallocatePerBlock[result.getParentBlock()].push_back(result);
    }

    return op.getOperation();
  }

  // According to the function boundary ABI we are guaranteed to get ownership
  // of all MemRefs returned by the function. Thus we set ownership to constant
  // 'true' and remember to deallocate it.
  Value trueVal = buildBoolValue(builder, op.getLoc(), true);
  for (auto result : llvm::make_filter_range(op->getResults(), isMemref)) {
    joinOwnership(result, trueVal);
    memrefsToDeallocatePerBlock[result.getParentBlock()].push_back(result);
  }

  return op.getOperation();
}

FailureOr<Operation *>
BufferDeallocation::handleInterface(MemoryEffectOpInterface op) {
  auto *block = op->getBlock();

  for (auto operand : llvm::make_filter_range(op->getOperands(), isMemref))
    if (op.getEffectOnValue<MemoryEffects::Free>(operand).has_value())
      return op->emitError(
          "memory free side-effect on MemRef value not supported!");

  OpBuilder builder = OpBuilder::atBlockBegin(block);
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
        clearOwnershipOf(res, block);
        joinOwnership(res, buildBoolValue(builder, op.getLoc(), false));
        continue;
      }

      joinOwnership(res, buildBoolValue(builder, op.getLoc(), true));
      memrefsToDeallocatePerBlock[block].push_back(res);
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

      val.set(getMemrefWithGuaranteedOwnership(builder, val.get()));
    }
  }

  // TODO: getSuccessorRegions is not implemented by all operations we care
  // about, but we would need to check how many successors there are and under
  // which condition they are taken, etc.

  MutableOperandRange operands =
      op.getMutableSuccessorOperands(RegionBranchPoint::parent());

  // Collect the values to deallocate and retain and use them to create the
  // dealloc operation.
  Block *block = op->getBlock();
  SmallVector<Value> memrefs, conditions, toRetain;
  if (failed(getMemrefsAndConditionsToDeallocate(builder, op.getLoc(), block,
                                                 memrefs, conditions)))
    return failure();

  getMemrefsToRetain(block, nullptr, OperandRange(operands), toRetain);
  if (memrefs.empty() && toRetain.empty())
    return op.getOperation();

  auto deallocOp = builder.create<bufferization::DeallocOp>(
      op.getLoc(), memrefs, conditions, toRetain);

  // We want to replace the current ownership of the retained values with the
  // result values of the dealloc operation as they are always unique.
  clearOwnershipOf(deallocOp.getRetained(), block);
  for (auto [retained, ownership] :
       llvm::zip(deallocOp.getRetained(), deallocOp.getUpdatedConditions()))
    joinOwnership(retained, ownership, block);

  // Add an additional operand for every MemRef for the ownership indicator.
  if (!funcWithoutDynamicOwnership) {
    unsigned numMemRefs = llvm::count_if(operands, isMemref);
    SmallVector<Value> newOperands{OperandRange(operands)};
    auto ownershipValues =
        deallocOp.getUpdatedConditions().take_front(numMemRefs);
    newOperands.append(ownershipValues.begin(), ownershipValues.end());
    operands.assign(newOperands);
  }

  return op.getOperation();
}

bool BufferDeallocation::isFunctionWithoutDynamicOwnership(Operation *op) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  return funcOp && (!privateFuncDynamicOwnership ||
                    funcOp.getVisibility() != SymbolTable::Visibility::Private);
}

void BufferDeallocation::populateRemainingOwnerships(Operation *op) {
  for (auto res : op->getResults()) {
    if (!isMemref(res))
      continue;
    if (ownershipMap.count({res, op->getBlock()}))
      continue;

    // Don't take ownership of a returned memref if no allocate side-effect is
    // present, relevant for memref.get_global, for example.
    if (op->getNumOperands() == 0) {
      OpBuilder builder(op);
      joinOwnership(res, buildBoolValue(builder, op->getLoc(), false));
      continue;
    }

    // Assume the result may alias with any operand and thus combine all their
    // ownerships.
    for (auto operand : op->getOperands()) {
      if (!isMemref(operand))
        continue;

      ownershipMap[{res, op->getBlock()}].combine(
          ownershipMap[{operand, operand.getParentBlock()}]);
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
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    if (failed(
            deallocateBuffersOwnershipBased(func, privateFuncDynamicOwnership)))
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
mlir::bufferization::createOwnershipBasedBufferDeallocationPass() {
  return std::make_unique<OwnershipBasedBufferDeallocationPass>();
}
