//===- ComprehensiveBufferize.cpp - Single pass bufferization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Perform inplace bufferization within function boundaries.
// This is a specialized pass that supports inplace analysis for a fixed subset
// of ops that have well-defined inplace semantics.
// This pass caters to high-performance codegen where buffer reuse is deemed
// critical: the pass should fail if the bufferized form of the function needs
// to return any buffer.
// Generic control-flow and branching are unsupported.
// Composability with extensible set of ops is not a first-class concern.
//
// Bufferization occurs by:
//  a. performing an inPlace analysis `inPlaceAnalysisFuncOpInternals`
//     which marks each operation within the function with the
//     `kInPlaceResultsAttrName` attribute.
//  b. traversing each operation in the function and rewriting it in
//     buffer form and keeping a BlockAndValueMapping mapping of the
//     rewrites. New allocations are introduced during this step.
//     TODO: Allocation + depending op hoisting to outermost enclosing
//     sequential scope.
//  c. at the end of this bufferization, 3 cases may occur:
//     i. inplaceable function arguments may be reused in place after the
//        function itself has been bufferized. This is encoded by IR resembling:
//
//        ```
//          #map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//           func @foo(%A: tensor<?xf32> {linalg.inplaceable = true})
//              -> tensor<?xf32> {
//            %0 = memref.buffer_cast %A : memref<?xf32, #map>
//            // ... uses of %0
//            %res = memref.tensor_load %0 : memref<?xf32, #map>
//            return %res : tensor<?xf32>
//          }
//        ```
//
//        this is the cue for the bufferization of the function foo (and calls
//        to it) may bufferize to `func @foo(%A: memref<?xf32, some_layout>)`.
//        To fully achieve bufferization, an additional analysis is needed to
//        determine whether function argument/operand pairs bufferize to a
//        single inplace buffer argument (i.e. functions may return tensors in
//        arbitrary order that may not match argument numbers).
//
//    ii. results that don't map to an inplaceable function argument are
//        generally allocated. Since memref semantics wrt ownership of the
//        underlying memory region are not well-defined, comprehensive
//        bufferization chooses to perform allocations in a scoped fashion:
//        returning memrefs is always considered illegal.
//        Such scenarios are encoded by IR resembling:
//
//        ```
//          #map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//          func @foo(%A: tensor<?xf32> {linalg.inplaceable = true})
//              -> tensor<?xf32> {
//            %0 = memref.buffer_cast %A : memref<?xf32, #map>
//            %1 = memref.dim %0, %c0 : memref<?xf32, #map>
//            %2 = memref.alloc(%1) : memref<?xf32>
//            %3 = memref.cast %2 : memref<?xf32> to memref<?xf32, #map>
//            // ... uses of %3
//            memref.dealloc %2 : memref<?xf32, #map>
//            %res = memref.tensor_load %3 : memref<?xf32, #map>
//            return %res : tensor<?xf32>
//          }
//       ```
//
//        this is the cue for the bufferization of the function foo (and calls
//        to it) that it must bufferize to `func @foo(%A: memref<?xf32,
//        some_layout>,
//                   %B: memref<?xf32, some_layout>)` (i.e. make a cloned
//        allocation of the result tensor)
//        To fully achieve bufferization, the alloc/dealloc pair must be lifted
//        out of the function at each call site.
//
//   iii. as an optimization over ii., it may be possible to reuse an argument
//        and only want to return a slice.
//        This may forego allocation by letting *all* callers decide whether to
//        pass a new *aliasing* memref function argument (i.e. a subview).
//        Without loss of generality, callers may agree to allocate a new buffer
//        to avoid this aliasing. Such scenarios are encoded by IR resembling:
//
//        ```
//          #map = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//          func @foo(%arg0: tensor<?xf32> {linalg.inplaceable = true})
//              -> tensor<4xf32> {
//            %0 = memref.buffer_cast %arg0 : memref<?xf32, #map>
//            %1 = memref.subview %0[0] [4] [1] : memref<?xf32, #map> to
//                                                memref<4xf32, #map>
//            // ... inplace computes into %1
//            %3 = memref.tensor_load %1 : memref<4xf32, #map>
//            return %3 : tensor<4xf32>
//          }
//        ```
//
//  Note: In the future, it may be worthwhile to design special bufferization
//  ops to encode the desired semantics at function boundaries for i., ii. and
//  iii.
//
//  Lastly, note that layout map chosen to bufferize is the most dynamic
//  canonical strided layout of the proper rank. This ensures compatibility with
//  expected layouts after transformations. Combinations of memref.cast +
//  canonicalization are responsible for clean ups.

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "comprehensive-func-bufferize"

using namespace mlir;
using namespace linalg;
using namespace tensor;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X)

//===----------------------------------------------------------------------===//
// Bufferization-specific BlockAndValueMapping support with debugging.
//===----------------------------------------------------------------------===//

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, ValueRange keys, ValueRange values) {
  assert(!keys.empty() && "Unexpected empty keys");
  LDBG("Map: " << keys.front() << " to " << values.front() << '\n');
  return bvm.map(keys, values);
}

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, Value key, Value value) {
  LDBG("Map: " << key << " to " << value << '\n');
  return bvm.map(key, value);
}

/// Wrapper for better debugging.
static Value lookup(BlockAndValueMapping &bvm, Value key) {
  // TODO: if key comes from bbArg, forward.
  assert(key.getType().isa<TensorType>());
  Value v = bvm.lookupOrNull(key);
  if (v)
    return v;

  Operation *parentOp;
  if (auto bbArg = key.dyn_cast<BlockArgument>()) {
    if (isa<FuncOp>(key.getParentBlock()->getParentOp()))
      parentOp = key.getParentBlock()->getParentOp();
    else
      parentOp = key.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();
  } else {
    parentOp = key.getDefiningOp()->getParentOfType<FuncOp>();
  }
  LDBG("In func:\n" << *parentOp << "NO VALUE FOR KEY: " << key << '\n');
  return Value();
}

//===----------------------------------------------------------------------===//
// Bufferization-specific attribute manipulation.
// These could be simplified with helper structs on the side, for now attributes
// allow simple embedding in the IR which simplifies testing.
// This could also be folded in BufferizationAliasInfo or a Bufferizer class
// that uses BufferizationAliasInfo.
//===----------------------------------------------------------------------===//

/// Attribute marker to specify op results that can be bufferized inPlace.
constexpr StringLiteral kInPlaceResultsAttrName = "__inplace_results_attr__";

// TODO: proper enum.
enum class InPlaceSpec {
  False,
  True,
  None,
};

static StringRef stringify(InPlaceSpec val) {
  switch (val) {
  case InPlaceSpec::False:
    return "false";
  case InPlaceSpec::True:
    return "true";
  case InPlaceSpec::None:
    return "none";
  }
  return "";
}

static Optional<InPlaceSpec> symbolize(StringRef str) {
  return StringSwitch<Optional<InPlaceSpec>>(str)
      .Case("false", InPlaceSpec::False)
      .Case("true", InPlaceSpec::True)
      .Case("none", InPlaceSpec::None)
      .Default(None);
}

/// Mark whether OpResult can actually be bufferized inplace.
/// If `inPlace` is `InPlaceSpec::True`, the use-def chain analysis has
/// guaranteed that no subsequent write would occur to the bufferized
/// tensor value (i.e. the result can be bufferized inPlace).
static void setInPlaceOpResult(OpResult opResult,
                               InPlaceSpec inPlace = InPlaceSpec::True) {
  if (!opResult)
    return;

  Operation *op = opResult.getOwner();
  auto attr =
      op->getAttr(kInPlaceResultsAttrName).dyn_cast_or_null<ArrayAttr>();
  SmallVector<StringRef> inPlaceVector =
      attr ? SmallVector<StringRef>(
                 llvm::to_vector<4>(attr.getAsValueRange<StringAttr>()))
           : SmallVector<StringRef>(op->getNumResults(),
                                    stringify(InPlaceSpec::None));
  LDBG("->set inPlace=" << stringify(inPlace) << ": " << *op
                        << " @idx=" << opResult.getResultNumber() << '\n');
  inPlaceVector[opResult.getResultNumber()] = stringify(inPlace);
  op->setAttr(kInPlaceResultsAttrName,
              OpBuilder(op).getStrArrayAttr(inPlaceVector));
}

/// Get the InPlaceSpec attribute entry `kInPlaceResultsAttrName` for
/// `opResult`. If the result is `InPlaceSpec::True`, the use-def chain analysis
/// has guaranteed that no subsequent read of the tensor value occurs and the
/// result can be buferized inPlace.
/// If no InPlaceSpec attribute has been set for `opResult`, return
/// InPlaceSpec::None.
static InPlaceSpec getInPlace(OpResult opResult) {
  if (!opResult)
    return InPlaceSpec::None;

  Operation *op = opResult.getOwner();
  auto attr =
      op->getAttr(kInPlaceResultsAttrName).dyn_cast_or_null<ArrayAttr>();
  if (!attr)
    return InPlaceSpec::None;

  // Must return a proper value.
  return *symbolize(*(attr.getAsValueRange<StringAttr>().begin() +
                      opResult.getResultNumber()));
}

/// Get inPlace information for `bbArg`.
/// FuncOp allow argument attributes, we use those to encode the information.
/// BlockArgument of other ops delegate to their owner's parent op.
static InPlaceSpec getInPlace(BlockArgument bbArg) {
  if (auto funcOp = dyn_cast<FuncOp>(bbArg.getOwner()->getParentOp())) {
    BoolAttr inplaceAttr = funcOp.getArgAttrOfType<BoolAttr>(
        bbArg.getArgNumber(), LinalgDialect::kInplaceableAttrName);
    if (!inplaceAttr)
      return InPlaceSpec::None;
    return inplaceAttr.getValue() ? InPlaceSpec::True : InPlaceSpec::False;
  }
  // Interestingly, scf::ForOp's bbArg can **always** be viewed inplace from the
  // perspective of ops nested under it:
  //   1. Either the matching iter operand is not bufferized inplace and an
  //      alloc + optional copy makes the bbArg itself inplaceable.
  //   2. Or the matching iter operand is bufferized inplace and bbArg just
  //      bufferizes to that too.
  if (auto forOp = dyn_cast<scf::ForOp>(bbArg.getOwner()->getParentOp()))
    return InPlaceSpec::True;
  // Unknown cases.
  return InPlaceSpec::None;
}

LLVM_ATTRIBUTE_UNUSED static InPlaceSpec getInPlace(Value v) {
  if (auto bbArg = v.dyn_cast<BlockArgument>())
    return getInPlace(bbArg);
  return getInPlace(v.cast<OpResult>());
}

//===----------------------------------------------------------------------===//
// Op-specific semantics helper to retrieve matching inplaceable result.
// These should become proper interfaces interfaces when the time is right.
// Modulo better naming, these helpers / interfaces comprise information on:
//   1. Whether an op has a known bufferization behavior (i.e. an instance of
//      BufferizableOpInterface).
//   2. Whether an op, when bufferized inplace, can guarantee an
//      (OpOperand, OpResult) pair bufferizes to equivalent (i.e. the same)
//      buffers in memory.
//   3. Whether an op operand, when bufferized inplace, aliases a return value.
//   4. Whether an op return value, when bufferized inplace, aliases an operand.
//   5. Wheher an op bufferizes to a memory read.
//   6. Wheher an op bufferizes to a memory write.
// These interfaces are necessary to distinguish between various cases and allow
// special inplace behavior for (ExtractSliceOp, InsertSliceOp) pairs.
//===----------------------------------------------------------------------===//

/// Return `true` if the op is explicitly supported by bufferization or if it
/// has no result tensors.
/// Other cases must be conservative.
static bool hasKnownBufferizationAliasingBehavior(Operation *op) {
  return
      // clang-format off
      isa<scf::ForOp,
          LinalgOp,
          ReturnOp,
          ExtractSliceOp,
          InsertSliceOp,
          VectorTransferOpInterface,
          scf::YieldOp>(op)
      // clang-format on
      || (none_of(op->getResultTypes(),
                  [](Type t) { return t.isa<TensorType>(); }) &&
          none_of(op->getOperandTypes(),
                  [](Type t) { return t.isa<TensorType>(); }));
}

/// Return the OpResult that may bufferize into the same buffer as `opOperand`
/// when the op is bufferized inplace.
/// Return null if no such result exists.
static OpResult getInplaceableOpResult(scf::ForOp forOp, OpOperand &opOperand) {
  if (!opOperand.get().getType().isa<RankedTensorType>())
    return OpResult();
  return forOp.getResultForOpOperand(opOperand);
}

/// Return the OpResult that may bufferize into the same buffer as `opOperand`
/// when the op is bufferized inplace.
/// Return null if no such result exists.
static OpResult getInplaceableOpResult(LinalgOp linalgOp,
                                       OpOperand &opOperand) {
  if (!opOperand.get().getType().isa<RankedTensorType>())
    return OpResult();
  // For now assume inputs are never inplaceable.
  // TODO: refine this.
  if (opOperand.getOperandNumber() < linalgOp.getNumInputs())
    return OpResult();
  int64_t outputOperandIndex =
      opOperand.getOperandNumber() - linalgOp.getNumInputs();
  int64_t numOutputBuffers = 0;
  for (unsigned idx = 0; idx < outputOperandIndex; ++idx)
    if (!linalgOp.getOutputOperand(idx)->get().getType().isa<TensorType>())
      ++numOutputBuffers;
  return linalgOp->getResult(outputOperandIndex - numOutputBuffers);
}

/// Return the OpResult that may bufferize into the same buffer as `opOperand`
/// when the op is bufferized inplace.
/// Return null if no such result exists.
static OpResult getInplaceableOpResult(VectorTransferOpInterface op,
                                       OpOperand &opOperand) {
  if (opOperand.get() != op.source() ||
      !op.source().getType().isa<TensorType>())
    return OpResult();
  return op->getResult(0);
}

/// Return the OpResult that may bufferize into the same buffer as `opOperand`
/// when the op is bufferized inplace.
/// Return null if no such result exists.
static OpResult getInplaceableOpResult(InsertSliceOp op, OpOperand &opOperand) {
  if (opOperand.get() != op.dest())
    return OpResult();
  return op->getResult(0);
}

/// Return the OpResult that may bufferize into the same buffer as `opOperand`
/// when the op is bufferized inplace.
/// The inplace analysis uses this information along with interfering read
/// analysis to determine which op results reuse the same buffer as some
/// operand.
static OpResult getInplaceableOpResult(OpOperand &opOperand) {
  return TypeSwitch<Operation *, OpResult>(opOperand.getOwner())
      // clang-format off
        // Ops that perform destructive updates on operand(s) to produce
        // result(s).
        .Case<scf::ForOp,
              LinalgOp,
              InsertSliceOp,
              VectorTransferOpInterface>(
            [&](auto op) { return getInplaceableOpResult(op, opOperand); })
        // ExtractSliceOp is special, when bufferized inplace it just returns an
        // alias to its operand. Its result is never inplaceable on its operand.
        .Case([&](ExtractSliceOp op) { return OpResult(); })
        // Other ops.
        .Default([&](Operation *op) { return OpResult(); });
  // clang-format on
}

/// Determine which OpResult will alias with `opOperand` if the op is bufferized
/// in place. This is a superset of `getInplaceableOpResult`.
/// Return None if the owner of `opOperand` does not have known
/// bufferization aliasing behavior, which indicates that the op must allocate
/// all of its tensor results.
/// TODO: in the future this may need to evolve towards a list of OpResult.
static Optional<OpResult> getAliasingOpResult(OpOperand &opOperand) {
  if (!hasKnownBufferizationAliasingBehavior(opOperand.getOwner()))
    return None;
  return TypeSwitch<Operation *, OpResult>(opOperand.getOwner())
      // These terminators legitimately have no result.
      .Case<ReturnOp, linalg::YieldOp, scf::YieldOp>(
          [&](auto op) { return OpResult(); })
      // ExtractSliceOp is different: its result is not inplaceable on op.source
      // but when bufferized inplace, the result is an aliasing subregion of
      // op.source.
      .Case([&](ExtractSliceOp op) { return op->getResult(0); })
      // All other ops, including scf::ForOp, return the result of
      // `getInplaceableOpResult`.
      .Default(
          [&](Operation *op) { return getInplaceableOpResult(opOperand); });
}

/// Return true if `opOperand` bufferizes to a memory read.
static bool bufferizesToMemoryRead(OpOperand &opOperand) {
  Optional<OpResult> maybeOpResult = getAliasingOpResult(opOperand);
  // Unknown op that returns a tensor. The inplace analysis does not support
  // it. Conservatively return true.
  if (!maybeOpResult)
    return true;
  // ExtractSliceOp alone doesn't bufferize to a memory read, one of its uses
  // may.
  if (isa<ExtractSliceOp>(opOperand.getOwner()))
    return false;
  // scf::ForOp alone doesn't bufferize to a memory read, one of the uses of its
  // matching bbArg may.
  if (isa<scf::ForOp>(opOperand.getOwner()))
    return false;
  if (auto linalgOp = dyn_cast<LinalgOp>(opOperand.getOwner()))
    return linalgOp.isInputTensor(&opOperand) ||
           linalgOp.isInitTensor(&opOperand);
  // All other cases are considered to bufferize to memory reads.
  // In particular, terminators are often the last use and need to be considered
  // as reads to return the proper value and avoid WAW clobbers.
  return true;
}

/// Return true if `opOperand` bufferizes to a memory write.
/// If inPlaceSpec is different from InPlaceSpec::None, additionally require the
/// write to match the inplace specification.
static bool
bufferizesToMemoryWrite(OpOperand &opOperand,
                        InPlaceSpec inPlaceSpec = InPlaceSpec::None) {
  Optional<OpResult> maybeOpResult = getAliasingOpResult(opOperand);
  // Unknown op that returns a tensor. The inplace analysis does not support
  // it. Conservatively return true.
  if (!maybeOpResult)
    return true;
  // Supported op without a matching result for opOperand (e.g. ReturnOp).
  // This does not bufferize to a write.
  if (!*maybeOpResult)
    return false;
  // These terminators are not writes.
  if (isa<ReturnOp, linalg::YieldOp, scf::YieldOp>(opOperand.getOwner()))
    return false;
  // ExtractSliceOp alone doesn't bufferize to a memory write, one of its uses
  // may.
  if (maybeOpResult->getDefiningOp<ExtractSliceOp>())
    return false;
  // If we have a matching OpResult, this is a write.
  // Additionally allow to restrict to only inPlace write, if so specified.
  return inPlaceSpec == InPlaceSpec::None ||
         getInPlace(*maybeOpResult) == inPlaceSpec;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific alias analysis.
//===----------------------------------------------------------------------===//

namespace {

/// The BufferizationAliasInfo class maintains a list of buffer aliases and
/// equivalence classes to support bufferization.
/// ExtractSliceOps have special behavior, they act as a level of indirection
/// for bufferization. They don't create reads or writes themselves and analysis
/// needs to look through their uses.
/// ExtractSliceOp + InsertSliceOp have special joint behavior: they may
/// bufferize to the same buffer (i.e. subview), which is what introduces the
/// need for bufferization classes.
/// Some of these functionalities could be refactored in a Bufferizer class that
/// uses BufferizationAliasInfo.
class BufferizationAliasInfo {
public:
  /// Specify fine-grain relationship between buffers to enable more analysis.
  enum class BufferRelation {
    None,
    // TODO: ResultContainsOperand,
    // TODO: OperandContainsResult,
    Equivalent
  };

  explicit BufferizationAliasInfo(FuncOp funcOp);

  /// Return true if the buffer to which `operand` would bufferize aliases a
  /// buffer that is known to not be writeable. This implies that the matching
  /// OpResult cannot be bufferized inplace.
  bool aliasesNonWriteableBuffer(OpOperand &operand) const;

  /// Return true if the buffer to which `operand` would bufferize is equivalent
  /// to some use that would bufferize to a write to a buffer.
  bool aliasesInPlaceWrite(ExtractSliceOp extractSliceOp) const;

  /// Set the inPlace bufferization spec to true.
  /// Merge result's and operand's aliasing sets and iterate to a fixed point.
  void bufferizeInPlace(OpResult result, OpOperand &operand,
                        BufferRelation bufferRelation = BufferRelation::None);

  /// Set the inPlace bufferization spec to false.
  void bufferizeOutOfPlace(OpResult result);

  /// Return true if it is possible to find an inplace write W among the uses of
  /// aliasInfo[rootWrite], and a read R among the uses of aliasInfo[rootRead],
  /// such that W and R interfere.
  /// Such a (W, R) pair is an interference to the inplace bufferization of
  /// rootWrite when:
  ///   1. R is not known properly dominate W (i.e. the effects of the write may
  ///      be visible from R).
  ///   2. one cannot find an intermediate clobbering write `C` to W, such that
  ///      C interleaved between W and R (i.e. W -> C -> R where -> denotes
  ///      dominance).
  bool
  wouldCreateReadAfterWriteInterference(Value rootWrite, Value rootRead,
                                        Operation *opToBufferize,
                                        const DominanceInfo &domInfo) const;

  /// Return true if we find any read to opOperand.get() or any of its aliases,
  /// that does not dominate opOperand.getOwner().
  bool existsNonDominatingRead(OpOperand &opOperand,
                               const DominanceInfo &domInfo) const;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const {
    return equivalentInfo.getLeaderValue(v1) ==
           equivalentInfo.getLeaderValue(v2);
  }

  /// Return true if the source of an `insertSliceOp` bufferizes to an
  /// equivalent ExtractSliceOp.
  bool isSourceEquivalentToAMatchingExtractSliceOp(
      InsertSliceOp insertSliceOp) const;

  /// Print to `os`.
  void print(raw_ostream &os) const;

  /// Print to `errs()`.
  void dump() const { print(llvm::errs()); }

private:
  /// Check aliasInfo for `v` exists and return a reference to it.
  DenseSet<Value> &getAliasInfoRef(Value v);
  const DenseSet<Value> &getAliasInfoRef(Value v) const {
    return const_cast<BufferizationAliasInfo *>(this)->getAliasInfoRef(v);
  }

  /// Union all the aliasing sets of all aliases of v1 and v2.
  bool mergeAliases(Value v1, Value v2);

  /// Iteratively merge alias sets until a fixed-point.
  void mergeAliasesToFixedPoint();

  /// Return true if the (ExtractSliceOp, InsertSliceOp) pair match (i.e.
  /// equivalent operand / result and same offset/sizes/strides specification).
  ///
  /// This is one particular type of relationship between ops on tensors that
  /// reduce to an equivalence on buffers. This should be generalized and
  /// exposed as interfaces on the proper types.
  bool areEquivalentExtractSliceOps(ExtractSliceOp st, InsertSliceOp sti) const;

  /// Return true if there is a `candidateOp` that would write to memory after
  /// bufferization and such that:
  ///   1. The written buffer is equivalent to either `aliasingRead` or
  ///      `aliasingWrite` under the inPlace bufferization decisions taken
  ///      so far.
  ///   2. `aliasingWrite` properly dominates `candidateOp`.
  ///   3. `candidateOp` properly dominates `aliasingReadOp`.
  // TODO: richer clobbering analysis with container-containee relationship
  // instead of equivalence.
  bool existsInterleavedValueClobber(OpOperand &aliasingRead,
                                     OpOperand &aliasingWrite,
                                     const DominanceInfo &domInfo) const;

  /// Return true if there is a write that:
  ///   1. Properly dominates aliasingReadOp.
  ///   2. Is properly dominated by aliasingWriteOp.
  ///   3. Clobbers the write that would be interfering with the read.
  ///
  /// Case discussion:
  /// ================
  /// Case 1: rootRead is produced by opToBufferize,
  /// Case 2: rootWrite is produced by opToBufferize,
  /// Common case:
  ///   - aliasingReadOp is a read to an alias of rootRead.
  ///   - aliasingWriteOp is an inplace write to an alias of rootWrite.
  ///   - aliasingWriteOp dominates aliasingReadOp.
  ///
  /// ```
  ///    // Either case 1:
  ///    %rootRead = opToBufferize(%rootWrite)
  ///    aliasingWriteOp(%aliasingWrite = alias(%rootWrite)) // inplace
  ///     aliasingReadOp( %aliasingRead = alias(%rootRead))
  /// ```
  ///
  /// ```
  ///    // Or case 2:
  ///    %rootWrite = opToBufferize(%rootRead)
  ///    aliasingWriteOp(%aliasingWrite = alias(%rootWrite)) // inplace
  ///     aliasingReadOp( %aliasingRead = alias(%rootRead))
  /// ```
  ///
  /// Capture possible cases where `aliasingWriteOp(alias(%rootWrite))` has no
  /// visible effect on `aliasingReadOp(alias(%rootRead))`.
  bool isClobberedWriteBeforeRead(Operation *opToBufferize, Value rootRead,
                                  Value rootWrite, OpOperand &aliasingRead,
                                  OpOperand &aliasingWrite,
                                  const DominanceInfo &domInfo) const;

  /// EquivalenceClasses wants comparable elements because it uses std::set.
  /// ValueWrapper wraps Value and uses pointer comparison on the defining op.
  /// This is a poor man's comparison but it's not like UnionFind needs ordering
  /// anyway ..
  struct ValueWrapper {
    ValueWrapper(Value val) : v(val) {}
    operator Value() const { return v; }
    bool operator<(const ValueWrapper &wrap) const {
      return v.getImpl() < wrap.v.getImpl();
    }
    bool operator==(const ValueWrapper &wrap) const { return v == wrap.v; }
    Value v;
  };

  /// Auxiliary structure to store all the values a given value aliases with.
  /// These are the conservative cases that can further decompose into
  /// "equivalent" buffer relationships.
  DenseMap<Value, DenseSet<Value>> aliasInfo;

  /// Auxiliary structure to store all the equivalent buffer classes.
  llvm::EquivalenceClasses<ValueWrapper> equivalentInfo;
};
} // namespace

BufferizationAliasInfo::BufferizationAliasInfo(FuncOp funcOp) {
  funcOp.walk([&](Operation *op) {
    for (Value v : op->getResults()) {
      if (!v.getType().isa<TensorType>())
        continue;
      assert(getInPlace(v) == InPlaceSpec::None &&
             "unexpected inplace in analysis.");
      DenseSet<Value> selfSet;
      selfSet.insert(v);
      aliasInfo.try_emplace(v, selfSet);
      equivalentInfo.insert(v);
    }
    for (Region &r : op->getRegions()) {
      for (Block &b : r.getBlocks()) {
        for (auto bbArg : b.getArguments()) {
          if (!bbArg.getType().isa<TensorType>())
            continue;
          DenseSet<Value> selfSet;
          selfSet.insert(bbArg);
          aliasInfo.try_emplace(bbArg, selfSet);
          equivalentInfo.insert(bbArg);
        }
      }
    }
  });
}

/// Return true if the buffer to which `operand` would bufferize aliases a
/// buffer that is known to not be writeable. This implies that the matching
/// OpResult cannot be bufferized inplace.
bool BufferizationAliasInfo::aliasesNonWriteableBuffer(
    OpOperand &operand) const {
  LDBG("----Start aliasesNonWriteableBuffer\n");
  LDBG("-------for operand #" << operand.getOperandNumber() << ": "
                              << *(operand.getOwner()) << '\n');
  for (Value v : getAliasInfoRef(operand.get())) {
    LDBG("-----------examine: " << v << '\n');
    if (auto bbArg = v.dyn_cast<BlockArgument>()) {
      if (getInPlace(bbArg) == InPlaceSpec::True) {
        LDBG("-----------bbArg is writeable -> skip: " << bbArg << '\n');
        continue;
      }
      LDBG("-----------notWriteable: " << v << '\n');
      return true;
    }

    if (Operation *op = v.getDefiningOp()) {
      if (isa<ConstantOp>(op) || !hasKnownBufferizationAliasingBehavior(op)) {
        LDBG("-----------notWriteable: " << v << '\n');
        return true;
      }
    }
  }
  LDBG("---->operand is writeable\n");
  return false;
}

/// Return true if the buffer to which `operand` would bufferize is equivalent
/// to some use that would bufferize to a write to a buffer.
bool BufferizationAliasInfo::aliasesInPlaceWrite(
    ExtractSliceOp extractSliceOp) const {
  LDBG("----Start aliasesInPlaceWrite\n");
  LDBG("-------for op: " << *extractSliceOp.getOperation() << '\n');
  for (Value v : getAliasInfoRef(extractSliceOp.result())) {
    for (auto &use : v.getUses()) {
      if (bufferizesToMemoryWrite(use, InPlaceSpec::True)) {
        LDBG("-----------wants to bufferize to inPlace write: "
             << *use.getOwner() << '\n');
        return true;
      }
    }
  }
  LDBG("----------->extract_slice does not alias an inplace write");
  return false;
}

/// Set the inPlace bufferization spec to true.
/// Merge result's and operand's aliasing sets and iterates to a fixed point.
void BufferizationAliasInfo::bufferizeInPlace(OpResult result,
                                              OpOperand &operand,
                                              BufferRelation bufferRelation) {
  setInPlaceOpResult(result, InPlaceSpec::True);
  if (mergeAliases(result, operand.get()))
    mergeAliasesToFixedPoint();
  if (bufferRelation == BufferRelation::Equivalent)
    equivalentInfo.unionSets(result, operand.get());
  // Dump the updated analysis.
  LLVM_DEBUG(dump());
}

/// Set the inPlace bufferization spec to false.
void BufferizationAliasInfo::bufferizeOutOfPlace(OpResult result) {
  setInPlaceOpResult(result, InPlaceSpec::False);
}

/// Return true if merging the alias sets of `rootWrite` and `rootRead` would
/// result in a semantic change in the program (i.e. RAW violation).
///
/// This is the case when one can find an inplace write W among the aliases
/// `rootWrite`, that may become an interference if W were to be bufferized
/// inplace. A potential interference would be with respect to a read R among
/// the aliases of `rootRead`.
///
/// Such a (W, R) pair is an interference to the inplace bufferization of
/// rootWrite when R does not properly dominate W (i.e. W may come before R
/// along some control-flow path).
bool BufferizationAliasInfo::wouldCreateReadAfterWriteInterference(
    Value rootWrite, Value rootRead, Operation *opToBufferize,
    const DominanceInfo &domInfo) const {
  LDBG("----Start wouldCreateReadAfterWriteInterference\n");

  // Collect all the inplace write uses of some alias of `rootWrite`.
  DenseSet<OpOperand *> usesWrite;
  auto &aliasListWrite = getAliasInfoRef(rootWrite);
  for (Value vWrite : aliasListWrite) {
    for (auto &uWrite : vWrite.getUses()) {
      if (!bufferizesToMemoryWrite(uWrite, InPlaceSpec::True))
        continue;
      usesWrite.insert(&uWrite);
    }
  }

  // Collect all the read uses of some alias of `rootRead`.
  DenseSet<OpOperand *> usesRead;
  auto &aliasListRead = getAliasInfoRef(rootRead);
  for (Value vRead : aliasListRead) {
    for (auto &uRead : vRead.getUses()) {
      if (!bufferizesToMemoryRead(uRead))
        continue;
      usesRead.insert(&uRead);
    }
  }

  for (OpOperand *uRead : usesRead) {
    Operation *aliasingReadOp = uRead->getOwner();
    LDBG("----++++aliasRead  #" << uRead->getOperandNumber()
                                << " in: " << *aliasingReadOp << '\n');
    for (OpOperand *uWrite : usesWrite) {
      // Don't consider self-use of the same operand.
      // Uses within the same op is fine though.
      if (uWrite == uRead)
        continue;
      Operation *aliasingWriteOp = uWrite->getOwner();
      LDBG("----    aliasWrite #" << uWrite->getOperandNumber()
                                  << " in: " << *aliasingWriteOp << '\n');
      // If read and written value already alias, no interference would be added
      // by bufferizing inplace.
      if (getAliasInfoRef(uRead->get()).contains(uWrite->get()))
        continue;
      // If aliasingReadOp properly dominates aliasingWriteOp, the read cannot
      // be affected by the write: there is no interference.
      if (domInfo.properlyDominates(aliasingReadOp, aliasingWriteOp))
        continue;
      // At this point, aliasingWriteOp properly dominates aliasingReadOp or
      // there is no clear dominance and we need to be conservative.
      LDBG("---->found RaW interference\n");
      LDBG("     Interfering  read (op #" << uRead->getOperandNumber()
                                          << "): " << *aliasingReadOp << '\n');
      LDBG("     Interfering write (op #" << uWrite->getOperandNumber()
                                          << "): " << *aliasingWriteOp << '\n');
      LDBG("         aliases rootRead: " << rootRead << '\n');
      LDBG("         aliases rootWrite: " << rootWrite << '\n');
      LDBG("---->opportunity to clobber RaW interference\n");
      if (isClobberedWriteBeforeRead(opToBufferize, rootRead, rootWrite, *uRead,
                                     *uWrite, domInfo)) {

        LDBG("---->clobbered! -> skip\n");
        continue;
      }

      LDBG("---->not clobbered -> found an interference\n");
      return true;
    }
  }
  LDBG("----No interference found\n");
  return false;
}

/// Return true if we find any read to opOperand.get() or any of its aliases,
/// that does not dominate opOperand.getOwner().
bool BufferizationAliasInfo::existsNonDominatingRead(
    OpOperand &opOperand, const DominanceInfo &domInfo) const {
  LDBG("----Start existsNonDominatingRead\n");
  Operation *op = opOperand.getOwner();
  for (Value alias : getAliasInfoRef(opOperand.get())) {
    for (OpOperand &wantReadUse : alias.getUses()) {
      LDBG("--------current operand #" << wantReadUse.getOperandNumber() << ": "
                                       << *(wantReadUse.getOwner()) << '\n');
      if (!bufferizesToMemoryRead(wantReadUse)) {
        LDBG("------------not a read -> skip\n");
        continue;
      }
      if (&wantReadUse == &opOperand) {
        LDBG("------------self-read is not an interference -> skip\n");
        continue;
      }
      if (domInfo.properlyDominates(wantReadUse.getOwner(), op)) {
        LDBG("------------read properly dominates -> skip\n");
        continue;
      }
      LDBG("----found interfering read of " << wantReadUse.get() << '\n');
      return true;
    }
  }
  return false;
}

/// Return true if the source of a `insertSliceOp` bufferizes to an
/// equivalent ExtractSliceOp.
bool BufferizationAliasInfo::isSourceEquivalentToAMatchingExtractSliceOp(
    InsertSliceOp insertSliceOp) const {
  auto leaderIt = equivalentInfo.findLeader(insertSliceOp.source());
  for (auto mit = leaderIt, meit = equivalentInfo.member_end(); mit != meit;
       ++mit) {
    if (areEquivalentExtractSliceOps(
            dyn_cast_or_null<ExtractSliceOp>(mit->v.getDefiningOp()),
            insertSliceOp))
      return true;
  }
  return false;
}

void BufferizationAliasInfo::print(raw_ostream &os) const {
  os << "\n/========================== AliasInfo "
        "==========================\n";
  for (auto it : aliasInfo) {
    os << "|\n| -- source: " << it.getFirst() << '\n';
    for (auto v : it.getSecond())
      os << "| ---- target: " << v << '\n';
  }
  os << "|\n\\====================== End AliasInfo "
        "======================\n\n";
  os << "\n/********************* Equivalent Buffers *********************\n";
  for (auto it = equivalentInfo.begin(), eit = equivalentInfo.end(); it != eit;
       ++it) {
    if (!it->isLeader())
      continue;
    Value leader = it->getData();
    os << "|\n| -- leader: " << leader << '\n';
    for (auto mit = equivalentInfo.member_begin(it),
              meit = equivalentInfo.member_end();
         mit != meit; ++mit) {
      Value v = static_cast<Value>(*mit);
      os << "| ---- equivalent member: " << v << '\n';
    }
  }
  os << "|\n\\***************** End Equivalent Buffers *****************\n\n";
}

DenseSet<Value> &BufferizationAliasInfo::getAliasInfoRef(Value v) {
  auto it = aliasInfo.find(v);
  if (it == aliasInfo.end())
    llvm_unreachable("Missing alias");
  return it->getSecond();
}

/// Union all the aliasing sets of all aliases of v1 and v2.
bool BufferizationAliasInfo::mergeAliases(Value v1, Value v2) {
  // Avoid invalidation of iterators by pre unioning the aliases for v1 and v2.
  bool changed = set_union(getAliasInfoRef(v1), getAliasInfoRef(v2)) ||
                 set_union(getAliasInfoRef(v2), getAliasInfoRef(v1));
  for (auto v : getAliasInfoRef(v1))
    if (v != v1)
      changed |= set_union(getAliasInfoRef(v), getAliasInfoRef(v2));
  for (auto v : getAliasInfoRef(v2))
    if (v != v2)
      changed |= set_union(getAliasInfoRef(v), getAliasInfoRef(v1));
  return changed;
}

/// Iteratively merge alias sets until a fixed-point.
void BufferizationAliasInfo::mergeAliasesToFixedPoint() {
  while (true) {
    bool changed = false;
    for (auto it : aliasInfo)
      for (auto v : it.getSecond())
        changed |= mergeAliases(it.getFirst(), v);
    if (!changed)
      break;
  }
}

/// This is one particular type of relationship between ops on tensors that
/// reduce to an equivalence on buffers. This should be generalized and exposed
/// as interfaces on the proper types.
bool BufferizationAliasInfo::areEquivalentExtractSliceOps(
    ExtractSliceOp st, InsertSliceOp sti) const {
  if (!st || !sti)
    return false;
  if (!equivalentInfo.isEquivalent(st.source(), sti.dest()))
    return false;
  if (!sameOffsetsSizesAndStrides(st, sti, isEqualConstantIntOrValue))
    return false;
  if (!equivalentInfo.isEquivalent(st.result(), sti.source()))
    return false;
  return true;
}

/// Return true if there is a `candidateOp` that would write to memory after
/// bufferization and such that:
///   1. The written buffer is equivalent to either `aliasingRead` or
///      `aliasingWrite` under the inPlace bufferization decisions taken
///      so far.
///   2. `aliasingWrite` properly dominates `candidateOp`.
///   3. `candidateOp` properly dominates `aliasingReadOp`.
// TODO: richer clobbering analysis with container-containee relationship
// instead of equivalence.
bool BufferizationAliasInfo::existsInterleavedValueClobber(
    OpOperand &aliasingRead, OpOperand &aliasingWrite,
    const DominanceInfo &domInfo) const {
  Operation *aliasingReadOp = aliasingRead.getOwner();
  Operation *aliasingWriteOp = aliasingWrite.getOwner();
  assert(!domInfo.properlyDominates(aliasingReadOp, aliasingWriteOp) &&
         "Unexpected aliasingReadOp properly dominates aliasingWriteOp");

  for (Value valueToClobber : {aliasingRead.get(), aliasingWrite.get()}) {
    auto leaderIt = equivalentInfo.findLeader(valueToClobber);
    for (auto mit = leaderIt, meit = equivalentInfo.member_end(); mit != meit;
         ++mit) {
      /// Note: the "would write to memory after bufferization" condition is
      /// verified by `candidateOp` since it would produce a value that
      /// bufferizes to an equivalent buffer.
      Operation *candidateOp = mit->v.getDefiningOp();
      if (!candidateOp)
        continue;
      LDBG("---->clobbering candidate: " << *candidateOp << '\n');
      if (domInfo.properlyDominates(aliasingWriteOp, candidateOp) &&
          domInfo.properlyDominates(candidateOp, aliasingReadOp))
        return true;
    }
  }
  return false;
}

/// Return true if there is a write that:
///   1. Properly dominates aliasingReadOp.
///   2. Is properly dominated by aliasingWriteOp.
///   3. Clobbers the write that would be interfering with the read.
///
bool BufferizationAliasInfo::isClobberedWriteBeforeRead(
    Operation *opToBufferize, Value rootRead, Value rootWrite,
    OpOperand &aliasingRead, OpOperand &aliasingWrite,
    const DominanceInfo &domInfo) const {
  Operation *aliasingReadOp = aliasingRead.getOwner();
  Operation *aliasingWriteOp = aliasingWrite.getOwner();
  assert(!domInfo.properlyDominates(aliasingReadOp, aliasingWriteOp) &&
         "Unexpected aliasingReadOp properly dominates aliasingWriteOp");

  assert(((rootRead.isa<OpResult>() &&
           rootRead.getDefiningOp() == opToBufferize) ||
          (rootWrite.isa<OpResult>() &&
           rootWrite.getDefiningOp() == opToBufferize)) &&
         "Expected rootRead or rootWrite to be produced by opToBufferize");

  // Bail if the write does not dominate the read: it may clobber but only on
  // a strict subset of paths, which is not enough for safety.
  if (!domInfo.dominates(aliasingWriteOp, aliasingReadOp)) {
    LDBG("---->no clobbering: write does not dominate read\n");
    return false;
  }

  // The case `opToBufferize` isa ExtractSliceOp is important enough that we
  // look for it specifically. The key information to discover is whether the
  // aliasing read or write come from a matching InsertSliceOp.
  // Such a pattern is introduced by tiling and is the key inplace condition
  // not to miss.
  if (auto extractSliceOp = dyn_cast<ExtractSliceOp>(opToBufferize)) {
    if (auto insertSliceOp = dyn_cast<InsertSliceOp>(aliasingReadOp)) {
      // %1 = extract_slice %0[%offset_sizes_and_strides_1]
      //
      // ... // 0 or more of inplace compute that reduces to: %X is an
      //     // aliasingWrite equivalent to %1.
      // %W = inplace_write(%1)
      //
      // // aliasingRead %Y in insert_slice
      // ... = insert_slice %W into %R[%offset_sizes_and_strides_1]
      if (aliasingRead.get() == insertSliceOp.dest() &&
          // TODO: This is currently too restrictive and misses clobberings.
          // When available, use container-containee analysis: the condition
          // should be that the `aliasingWrite` is contained within
          // `insertSliceOp.source()`.
          equivalentInfo.isEquivalent(aliasingWrite.get(),
                                      insertSliceOp.source()) &&
          areEquivalentExtractSliceOps(extractSliceOp, insertSliceOp)) {
        LDBG("---->clobbering matching extract_slice/insert_slice\n");
        return true;
      }
      // %1 = extract_slice %0[%offset_sizes_and_strides_1]
      //
      // ... // bunch of inplace ops that reduce to %X, equivalent to %1.
      // %X = inplace_write(%1)
      //
      // // aliasingRead %X in insert_slice
      // // aliasingWrite %Y in insert_slice
      // ... = insert_slice %X into %Y[%offset_sizes_and_strides_1]
      if (aliasingReadOp == aliasingWriteOp) {
        assert(aliasingRead.get() == insertSliceOp.source() &&
               "expected read to source of insert_slice");
        assert(aliasingWrite.get() == insertSliceOp.dest() &&
               "expected write to dest of insert_slice");
        if (areEquivalentExtractSliceOps(extractSliceOp, insertSliceOp)) {
          LDBG("---->clobbering matching extract_slice/insert_slice\n");
          return true;
        }
      }
    }
  }

  // General case: look for a properly interleaved clobber of either exactly
  // `aliasingRead` or `aliasingWrite`.
  // TODO: Relax this to inclusion instead of double inclusion (a.k.a
  // equivalence). We will need to compute container-containee relationship.
  return existsInterleavedValueClobber(aliasingRead, aliasingWrite, domInfo);
}

//===----------------------------------------------------------------------===//
// Bufferization-specific MemRefType support.
//===----------------------------------------------------------------------===//

/// Return a contiguous MemRefType (i.e. with canonical/empty layout map)
/// with the same shape as `shapedType` and specified `layout` and
/// `addressSpace`.
static MemRefType getContiguousMemRefType(ShapedType shapedType,
                                          ArrayRef<AffineMap> layout = {},
                                          unsigned addressSpace = 0) {
  if (RankedTensorType tensorType = shapedType.dyn_cast<RankedTensorType>())
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                           layout, addressSpace);
  MemRefType memrefType = shapedType.cast<MemRefType>();
  return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                         layout, addressSpace);
}

/// Return a contiguous MemRefType (i.e. with canonical/empty layout map)
/// with the same shape as `shapedType` and specified `layout` and
/// `addressSpace` or an UnrankedMemRefType otherwise.
static Type getContiguousOrUnrankedMemRefType(Type type,
                                              ArrayRef<AffineMap> layout = {},
                                              unsigned addressSpace = 0) {
  if (type.isa<RankedTensorType, MemRefType>())
    return getContiguousMemRefType(type.cast<ShapedType>(), layout,
                                   addressSpace);
  assert(layout.empty() && "expected empty layout with UnrankedMemRefType");
  return UnrankedMemRefType::get(getElementTypeOrSelf(type), addressSpace);
}

/// Return a MemRefType to which the `tensorType` can be bufferized in a
/// composable fashion. The layout must be the most dynamic possible and
/// canonicalize away once bufferization is finished.
static MemRefType getDynamicMemRefType(RankedTensorType tensorType,
                                       unsigned addressSpace = 0) {
  // TODO: address space decisions to connect with the actual alloc.
  int64_t dynamicOffset = ShapedType::kDynamicStrideOrOffset;
  SmallVector<int64_t> dynamicStrides(tensorType.getRank(),
                                      ShapedType::kDynamicStrideOrOffset);
  AffineMap stridedLayout = makeStridedLinearLayoutMap(
      dynamicStrides, dynamicOffset, tensorType.getContext());
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         stridedLayout, addressSpace);
}

//===----------------------------------------------------------------------===//
// Bufferization-specific scoped alloc/dealloc insertion support.
//===----------------------------------------------------------------------===//

/// Create an Allocop/DeAllocOp pair, where the AllocOp is after
/// `shapedValue.getDefiningOp` (or at the top of the block in case of a
/// bbArg) and the DeallocOp is at the end of the block.
static Value createNewAllocDeallocPairForShapedValue(OpBuilder &b, Location loc,
                                                     Value shapedValue) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // TODO: non-zero address space.
  // TODO: layout information if relevant.
  // Cannot allocate an unranked memref so just always go for the contiguous
  // form.
  MemRefType allocMemRefType =
      getContiguousMemRefType(shapedValue.getType().cast<ShapedType>());
  assert(shapedValue.getType().isa<ShapedType>());
  MemRefType memRefType = shapedValue.getType().dyn_cast<MemRefType>();
  memRefType = memRefType ? memRefType : allocMemRefType;

  if (auto bbArg = shapedValue.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
    loc = bbArg.getOwner()->getParentOp()->getLoc();
  } else {
    b.setInsertionPointAfter(shapedValue.getDefiningOp());
    loc = shapedValue.getDefiningOp()->getLoc();
  }

  // Compute the dynamic part of the shape.
  SmallVector<Value> dynShape;
  for (auto dim : enumerate(memRefType.getShape()))
    if (dim.value() == ShapedType::kDynamicSize)
      dynShape.push_back(
          b.create<memref::DimOp>(loc, shapedValue, dim.index()));

  Value allocated = b.create<memref::AllocOp>(loc, allocMemRefType, dynShape);
  Value casted = allocated;
  if (memRefType != allocMemRefType)
    casted = b.create<memref::CastOp>(loc, memRefType, allocated);
  b.setInsertionPoint(allocated.getParentBlock()->getTerminator());
  b.create<memref::DeallocOp>(loc, allocated);
  return casted;
}

//===----------------------------------------------------------------------===//
// Bufferization as simple BlockAndValueMapping rewrites.
//===----------------------------------------------------------------------===//

/// Helper function for LinalgOp bufferization.
/// Examines each result and determines whether it bufferizes inplace on an
/// operand.
/// If the opResult bufferizes inplace, just reuse the existing buffer.
/// Otherwise allocate a new buffer to hold the result.
/// When allocating a new buffer, analyze whether `op` want to read form that
/// buffer. In such a case, insert a copy to ensure the newly allocated buffer
/// is properly initialiazed.
static LogicalResult
allocateBuffersForResults(OpBuilder &b, Location loc, LinalgOp op,
                          SmallVectorImpl<Value> &resultBuffers,
                          BlockAndValueMapping &bvm) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // TODO: provide the proper interface to iterate on OpResults and get the
  // matching OpOperands.
  for (OpOperand *opOperand : op.getOutputOperands()) {
    Value output = opOperand->get();
    assert(output.getType().isa<TensorType>() && "expected tensor type");

    // If output tensor is marked inPlace, just use the buffer.
    // The following uses internal knowledge of the position of inplaceable
    // operand / results.
    OpResult opResult = getInplaceableOpResult(*opOperand);
    if (getInPlace(opResult) == InPlaceSpec::True) {
      Value v = lookup(bvm, output);
      if (!v)
        return failure();
      resultBuffers.push_back(v);
      continue;
    }

    // Otherwise, `op` is not inplaceable and we need to allocate its result.
    Value dimTensor = bvm.lookupOrDefault(output);
    Value alloc = createNewAllocDeallocPairForShapedValue(b, loc, dimTensor);
    b.setInsertionPointAfter(alloc.getDefiningOp());
    resultBuffers.push_back(alloc);

    // Additionally, if the output buffer is used, clone its value for now.
    if (op.payloadUsesValueFromOperand(opOperand)) {
      if (Value v = lookup(bvm, output))
        b.create<CopyOp>(loc, v, alloc);
      else
        return failure();
    }
  }

  if (op->getNumResults())
    map(bvm, op->getResults(), resultBuffers);

  return success();
}

/// Generic conversion for any LinalgOp on tensors.
static LogicalResult bufferize(OpBuilder &b, LinalgOp op,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasTensorSemantics())
    return failure();

  LDBG("bufferize: " << *op << '\n');

  b.setInsertionPoint(op);
  Location loc = op.getLoc();
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    newInputBuffers.push_back(lookup(bvm, opOperand->get()));
    if (!newInputBuffers.back())
      return failure();
  }
  SmallVector<Value> newOutputBuffers;
  // Try to allocate new buffers depending on op's inplace semantics.
  if (failed(allocateBuffersForResults(b, loc, op, newOutputBuffers, bvm)))
    return failure();

  // Clone the newly bufferized op.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());
  auto otherOperands = op.getAssumedNonShapedOperands();
  newOperands.append(otherOperands.begin(), otherOperands.end());
  op.clone(b, loc, /*resultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  if (op->getNumResults())
    map(bvm, op->getResults(), newOutputBuffers);

  // The original op will be DCE'd away later.

  return success();
}

/// DimOp tensor operand is modified inplace. This allows leaving dead
/// tensors behind that will get DCE'd.
static LogicalResult bufferize(OpBuilder &b, memref::DimOp dimOp,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  if (dimOp.memrefOrTensor().getType().isa<RankedTensorType>()) {
    Value v = lookup(bvm, dimOp.memrefOrTensor());
    if (!v)
      return failure();
    dimOp.memrefOrTensorMutable().assign(v);
  }
  return success();
}

static LogicalResult bufferize(OpBuilder &b, scf::ForOp forOp,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  Location loc = forOp.getLoc();

  LLVM_DEBUG(DBGS() << "bufferize: " << *forOp << "\n");

  // If inPlace, just forward the buffer.
  // Otherwise alloc and copy.
  b.setInsertionPoint(forOp);
  for (OpResult opResult : forOp->getResults()) {
    // TODO: Atm we bail on unranked TensorType because we don't know how to
    // alloc an UnrankedMemRefType + its underlying ranked MemRefType.
    if (!opResult.getType().isa<RankedTensorType>())
      return failure();
    OpOperand &opOperand = forOp.getOpOperandForResult(opResult);
    Value operand = opOperand.get();
    Value operandBuffer = lookup(bvm, operand);
    Value resultBuffer = operandBuffer;
    if (getInPlace(opResult) != InPlaceSpec::True) {
      resultBuffer = createNewAllocDeallocPairForShapedValue(b, loc, operand);
      // If the tensor comes from `linalg::InitTensorOp`, the value is
      // unitialized and we do not need to copy.
      // TODO: if the matching bbArg does not bufferize to a read is more
      // general.
      if (!operand.getDefiningOp<linalg::InitTensorOp>())
        b.create<linalg::CopyOp>(forOp.getLoc(), operandBuffer, resultBuffer);
    }
    BlockArgument bbArg = forOp.getRegionIterArgForOpOperand(opOperand);
    map(bvm, bbArg, resultBuffer);
    map(bvm, opResult, resultBuffer);
  }

  return success();
}

/// FuncOp always creates TensorToMemRef ops.
static LogicalResult bufferize(OpBuilder &b, FuncOp funcOp,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(&funcOp.body().front());
  for (auto bbArg : funcOp.getArguments()) {
    auto tensorType = bbArg.getType().dyn_cast<TensorType>();
    if (!tensorType)
      continue;
    auto rankedTensorType = tensorType.dyn_cast<RankedTensorType>();
    // Cast the tensor to the most dynamic buffer possible. Further
    // canonicalizations will clean up.
    Type memRefType = rankedTensorType
                          ? getDynamicMemRefType(rankedTensorType)
                          : getContiguousOrUnrankedMemRefType(tensorType);
    Value tensorToMemref =
        b.create<memref::BufferCastOp>(funcOp.getLoc(), memRefType, bbArg);
    map(bvm, bbArg, tensorToMemref);
  }
  return success();
}

/// ReturnOp always creates memref::TensorLoadOp.
static LogicalResult bufferize(OpBuilder &b, ReturnOp returnOp,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(returnOp);

  assert(isa<FuncOp>(returnOp->getParentOp()) &&
         "only support FuncOp parent for ReturnOp");
  for (OpOperand &operand : returnOp->getOpOperands()) {
    auto tensorType = operand.get().getType().dyn_cast<TensorType>();
    if (!tensorType)
      continue;
    Value v = lookup(bvm, operand.get());
    if (!v)
      return failure();
    operand.set(b.create<memref::TensorLoadOp>(returnOp.getLoc(), v));
  }
  return success();
}

/// Bufferize ExtractSliceOp to subview with optional alloc + copy depending on
/// whether or not it is marked inplaceable.
/// Note that `getInplaceableOpResult` on a ExtractSliceOp always returns null.
/// As consequence a ExtractSliceOp always alloc + copy when taken in
/// isolation.
static LogicalResult bufferize(OpBuilder &b, ExtractSliceOp extractSliceOp,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  LDBG("bufferize: " << *extractSliceOp << '\n');

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(extractSliceOp);

  Location loc = extractSliceOp.getLoc();
  // Bail if source was not bufferized.
  Value srcMemref = lookup(bvm, extractSliceOp.source());
  if (!srcMemref)
    return failure();
  auto srcMemrefType = srcMemref.getType().cast<MemRefType>();
  auto dstTensorType =
      extractSliceOp.result().getType().cast<RankedTensorType>();

  // If not inplaceable, alloc.
  Value alloc;
  auto inPlace = getInPlace(extractSliceOp->getResult(0));
  if (inPlace != InPlaceSpec::True) {
    alloc = createNewAllocDeallocPairForShapedValue(b, loc,
                                                    extractSliceOp.result());
    b.setInsertionPointAfter(alloc.getDefiningOp());
  }

  // Bufferize to subview.
  auto subviewMemRefType =
      memref::SubViewOp::inferRankReducedResultType(
          dstTensorType.getRank(), srcMemrefType,
          extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
          extractSliceOp.getMixedStrides())
          .cast<MemRefType>();
  Value subView = b.create<memref::SubViewOp>(
      loc, subviewMemRefType, srcMemref, extractSliceOp.getMixedOffsets(),
      extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides());

  /// If not inplaceable, copy.
  if (alloc) {
    b.create<CopyOp>(extractSliceOp.getLoc(), subView, alloc);
    subView = alloc;
  }

  map(bvm, extractSliceOp.result(), subView);
  return success();
}

static LogicalResult bufferize(OpBuilder &b, InsertSliceOp insertSliceOp,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  LDBG("bufferize: " << *insertSliceOp << '\n');

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(insertSliceOp);
  Location loc = insertSliceOp.getLoc();

  Value dstMemref = lookup(bvm, insertSliceOp.dest());
  if (!dstMemref)
    return failure();
  auto inPlace = getInPlace(insertSliceOp->getResult(0));
  if (inPlace != InPlaceSpec::True) {
    // Since insert_slice arise from tiling and introducing loops, this
    // case is generally a deal breaker. When used with loops, this ends up
    // cloning the whole tensor on every single iteration and is a symptom
    // of a catastrophically bad scheduling decision.
    // TODO: be very loud about it or even consider failing the pass.
    Value newDstMemref =
        createNewAllocDeallocPairForShapedValue(b, loc, insertSliceOp.result());
    b.setInsertionPointAfter(newDstMemref.getDefiningOp());
    b.create<CopyOp>(insertSliceOp.getLoc(), dstMemref, newDstMemref);
    dstMemref = newDstMemref;
  }
  auto dstMemrefType = dstMemref.getType().cast<MemRefType>();

  Value srcMemref = lookup(bvm, insertSliceOp.source());
  if (!srcMemref)
    return failure();
  auto subviewMemRefType =
      memref::SubViewOp::inferRankReducedResultType(
          insertSliceOp.getSourceType().getRank(), dstMemrefType,
          insertSliceOp.getMixedOffsets(), insertSliceOp.getMixedSizes(),
          insertSliceOp.getMixedStrides())
          .cast<MemRefType>();

  // A copy of the source buffer is needed if either:
  //   - The producer of `source` is not inplace. This is the case where a
  //     slice is computed out of place into the inplace full tensor.
  //   - The result is not inplace. This is the case where the whole tensor is
  //     cloned and the clone needs to be updated.
  if (!aliasInfo.isSourceEquivalentToAMatchingExtractSliceOp(insertSliceOp) ||
      inPlace != InPlaceSpec::True) {
    LDBG("insert_slice needs extra source copy: " << insertSliceOp.source()
                                                  << " -> copy\n");
    // Take a subview of the dst.
    Value subView = b.create<memref::SubViewOp>(
        loc, subviewMemRefType, dstMemref, insertSliceOp.getMixedOffsets(),
        insertSliceOp.getMixedSizes(), insertSliceOp.getMixedStrides());
    b.create<CopyOp>(insertSliceOp.getLoc(), srcMemref, subView);
  }

  map(bvm, insertSliceOp.result(), dstMemref);

  return success();
}

static LogicalResult bufferize(OpBuilder &b, VectorTransferOpInterface op,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op.getLoc();

  if (op.getShapedType().isa<MemRefType>())
    return failure();

  LDBG("bufferize: " << *op << '\n');

  /// transfer_read from buffer always reads from the bufferized
  /// op.source().
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op.getOperation())) {
    Value v = lookup(bvm, op.source());
    if (!v)
      return failure();
    readOp.sourceMutable().assign(v);
    return success();
  }

  auto inPlace = getInPlace(op->getResult(0));
  auto writeOp = cast<vector::TransferWriteOp>(op.getOperation());

  // If transfer_write is not inPlace, allocate a new buffer.
  Value newInputBuffer;
  if (inPlace != InPlaceSpec::True) {
    newInputBuffer =
        createNewAllocDeallocPairForShapedValue(b, loc, writeOp.result());
    b.setInsertionPointAfter(newInputBuffer.getDefiningOp());
    map(bvm, writeOp.result(), newInputBuffer);
  } else {
    // InPlace write will result in memref.tensor_load(x) which must
    // canonicalize away with one of it uses.
    newInputBuffer = lookup(bvm, writeOp.source());
    if (!newInputBuffer)
      return failure();
  }

  // Create a new transfer_write on buffer that doesn't have a return value.
  // Leave the previous transfer_write to dead code as it still has uses at
  // this point.
  b.create<vector::TransferWriteOp>(
      loc, writeOp.vector(), newInputBuffer, writeOp.indices(),
      writeOp.permutation_map(),
      writeOp.in_bounds() ? *writeOp.in_bounds() : ArrayAttr());

  map(bvm, op->getResult(0), newInputBuffer);

  return success();
}

static LogicalResult bufferize(OpBuilder &b, scf::YieldOp yieldOp,
                               BlockAndValueMapping &bvm,
                               const BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(yieldOp);

  scf::ForOp forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
  assert(forOp && "only support scf::ForOp parent for scf::YieldOp");
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    auto tensorType = operand.get().getType().dyn_cast<TensorType>();
    if (!tensorType)
      continue;
    OpOperand &forOperand = forOp.getOpOperandForResult(
        forOp->getResult(operand.getOperandNumber()));
    auto bbArg = forOp.getRegionIterArgForOpOperand(forOperand);
    if (getInPlace(bbArg) == InPlaceSpec::True)
      operand.set(bbArg);
    else
      operand.set(
          b.create<memref::TensorLoadOp>(yieldOp.getLoc(), lookup(bvm, bbArg)));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Bufferization analyses.
//===----------------------------------------------------------------------===//

///
/// Rationale for bufferizing `%1 = tensor.extract_slice %0[...]` inplace.
/// ===========================================================
///
/// When bufferized out of place, a ExtractSlice lowers to alloc + copy. This
/// cannot change the flow of information for either the source or the
/// result buffers.
///
/// When bufferized inplace, a ExtractSliceOp does not by itself create any read
/// or write from memory. Instead, it has the effect of merging the alias sets
/// of the source and the result buffers.
///
/// An analysis is required to ensure inplace bufferization would not result in
/// RaW dependence violations.
static LogicalResult
bufferizableInPlaceAnalysis(ExtractSliceOp extractSliceOp,
                            BufferizationAliasInfo &aliasInfo,
                            const DominanceInfo &domInfo) {
  LDBG('\n');
  LDBG("Inplace analysis for extract_slice: " << *extractSliceOp << '\n');

  // If `extractSliceOp` were to be bufferized inplace, it cannot end up
  // aliasing a write into a non-writeable buffer.
  bool wouldCreateAliasingWriteToNonWriteableBuffer =
      aliasInfo.aliasesInPlaceWrite(extractSliceOp) &&
      aliasInfo.aliasesNonWriteableBuffer(extractSliceOp->getOpOperand(0));

  if (wouldCreateAliasingWriteToNonWriteableBuffer)
    LDBG("->the corresponding buffer is not writeable\n");
  else
    LDBG("->bufferizes to writeable inplace buffer\n");

  // In any of extractSliceOp.result's aliases, can we find 2 such that we hit
  // an interfering write?
  OpResult r = extractSliceOp->getResult(0);
  OpOperand &s = extractSliceOp->getOpOperand(0);
  bool foundInterference = wouldCreateAliasingWriteToNonWriteableBuffer ||
                           // Do not consider (s, s) and (r, r) as all the
                           // aliasings already exist by construction; we are
                           // interested in new interfering aliases only.
                           aliasInfo.wouldCreateReadAfterWriteInterference(
                               s.get(), r, extractSliceOp, domInfo) ||
                           aliasInfo.wouldCreateReadAfterWriteInterference(
                               r, s.get(), extractSliceOp, domInfo);
  if (foundInterference)
    aliasInfo.bufferizeOutOfPlace(r);
  else
    aliasInfo.bufferizeInPlace(r, s);

  LDBG("Done inplace analysis for extract_slice\n");

  return success();
}

/// Analyze the (opOperand, result) pair to determine whether the result can
/// be bufferized inPlace. If successful, InPlaceSpec::True is set for
/// `result`. Otherwise, InPlaceSpec::False is set for `result`.
static LogicalResult
bufferizableInPlaceAnalysis(OpOperand &operand, OpResult result,
                            BufferizationAliasInfo &aliasInfo,
                            const DominanceInfo &domInfo) {
  Operation *op = result.getDefiningOp();
  assert(result && !isa<ExtractSliceOp>(op) &&
         "expected OpResult not coming from a ExtractSliceOp");

  int64_t resultNumber = result.getResultNumber();
  (void)resultNumber;
  LDBG('\n');
  LDBG("Inplace analysis for result #" << resultNumber << " (operand #"
                                       << operand.getOperandNumber() << ") in "
                                       << result << '\n');

  // `result` must bufferize to a writeable buffer to be a candidate.
  // This means the use->def chain not backpropagate to a function that is
  // not inplaceable or to a constant op to be considered.
  bool wouldCreateAliasingWriteToNonWriteableBuffer =
      aliasInfo.aliasesNonWriteableBuffer(operand);
  if (wouldCreateAliasingWriteToNonWriteableBuffer)
    LDBG("->the corresponding buffer is not writeable\n");
  else
    LDBG("->bufferizes to writeable inplace buffer\n");

  Value s = operand.get(), r = result;
  bool foundInterference =
      wouldCreateAliasingWriteToNonWriteableBuffer ||
      aliasInfo.existsNonDominatingRead(operand, domInfo) ||
      // Do not consider (s, s) and (r, r) as all the aliasings already
      // exist by construction; we are interested in new interfering aliases
      // only.
      aliasInfo.wouldCreateReadAfterWriteInterference(s, r, op, domInfo) ||
      aliasInfo.wouldCreateReadAfterWriteInterference(r, s, op, domInfo);

  if (foundInterference)
    aliasInfo.bufferizeOutOfPlace(result);
  else
    // TODO: Atm, all inplace bufferizations yield equivalent tensors. Support
    // more cases on a per-need basis.
    aliasInfo.bufferizeInPlace(
        result, operand, BufferizationAliasInfo::BufferRelation::Equivalent);

  LDBG("Done inplace analysis for result #" << resultNumber << '\n');

  return success();
}

/// Return `failure()` if either
/// scf::YieldOp are not explicitly bufferized and we need to perform a separate
/// sanity check for now.
static LogicalResult
bufferizationSanityCheck(scf::YieldOp yieldOp,
                         const BufferizationAliasInfo &aliasInfo) {
  auto parentForOp = yieldOp->getParentOfType<scf::ForOp>();
  if (!parentForOp)
    return failure();

  for (OpOperand &operand : yieldOp->getOpOperands()) {
    OpResult matchingForOpResult =
        parentForOp->getResult(operand.getOperandNumber());
    // Nothing to do if operand bufferizes out of place.
    if (getInPlace(matchingForOpResult) != InPlaceSpec::True)
      continue;
    OpOperand &machingForOpOperand =
        parentForOp.getOpOperandForResult(matchingForOpResult);
    BlockArgument matchingForOpIterArg =
        parentForOp.getRegionIterArgForOpOperand(machingForOpOperand);
    if (!aliasInfo.areEquivalentBufferizedValues(matchingForOpIterArg,
                                                 operand.get())) {
      yieldOp->emitError()
          << "Yield operand #" << operand.getOperandNumber()
          << " does not bufferize to an equivalent buffer to the matching"
          << " enclosing scf::for operand -> Fail the pass\n";
      return failure();
    }
  }

  return success();
}

/// Analyze the `funcOp` body to determine which OpResults are inplaceable.
static LogicalResult
inPlaceAnalysisFuncOpInternals(FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
                               const DominanceInfo &domInfo) {
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  LDBG("Begin InPlaceAnalysisFuncOpInternals:\n" << funcOp << '\n');
  assert(funcOp && funcOp->getNumRegions() > 0 && !funcOp.body().empty() &&
         "expected a funcOp definition with a body");

  // Collect ops so we can build our own traversal.
  SmallVector<ExtractSliceOp> extractSliceOps;
  SmallVector<InsertSliceOp> insertSliceOps;
  SmallVector<Operation *> nonSliceOps;
  funcOp.walk([&](Operation *op) {
    if (auto extractSliceOp = dyn_cast<ExtractSliceOp>(op))
      return extractSliceOps.push_back(extractSliceOp);
    if (auto insertSliceOp = dyn_cast<InsertSliceOp>(op))
      return insertSliceOps.push_back(insertSliceOp);
    auto isaTensor = [](Type t) { return t.isa<TensorType>(); };
    // No tensors => no buffers.
    if (none_of(op->getOperandTypes(), isaTensor) &&
        none_of(op->getResultTypes(), isaTensor))
      return;
    nonSliceOps.push_back(op);
  });

  // Bufferize InsertSliceOp greedily: we almost never want to bufferize
  // the tensor "inserted into" to become out-of-place. This implementation
  // does not distinguish between different InsertSliceOp. If we want
  // finer-grained behavior, we could order the InsertSliceOp with some metric.
  // Walk InsertSliceOp in reverse for better interference behavior.
  for (InsertSliceOp insertSliceOp : reverse(insertSliceOps)) {
    OpOperand &destOpOperand = insertSliceOp->getOpOperand(1);
    if (failed(bufferizableInPlaceAnalysis(
            destOpOperand, getInplaceableOpResult(destOpOperand), aliasInfo,
            domInfo)))
      return failure();
  }

  // Bufferize all ops except ExtractSliceOp and InsertSliceOp which are handled
  // separately.
  // Walk other ops in reverse for better interference behavior.
  for (Operation *op : reverse(nonSliceOps))
    for (OpOperand &opOperand : op->getOpOperands())
      if (OpResult result = getInplaceableOpResult(opOperand))
        if (failed(bufferizableInPlaceAnalysis(opOperand, result, aliasInfo,
                                               domInfo)))
          return failure();

  // Finally, bufferize ExtractSliceOp.
  // Walk ExtractSliceOps in reverse for better clobbering behavior: it is
  // easier to detect clobbers of smaller slices before larger ones.
  for (ExtractSliceOp extractSliceOp : reverse(extractSliceOps))
    if (failed(bufferizableInPlaceAnalysis(extractSliceOp, aliasInfo, domInfo)))
      return failure();

  // Sanity checks.
  auto walkResult = funcOp.walk([&](scf::YieldOp yieldOp) -> WalkResult {
    return bufferizationSanityCheck(yieldOp, aliasInfo);
  });

  LDBG("End InPlaceAnalysisFuncOpInternals:\n" << funcOp << '\n');

  return success(!walkResult.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Bufferization entry-point.
//===----------------------------------------------------------------------===//

static LogicalResult
bufferizeFuncOpInternals(FuncOp funcOp, BlockAndValueMapping &bvm,
                         const BufferizationAliasInfo &aliasInfo) {
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  LDBG("Begin BufferizeFuncOpInternals:\n" << funcOp << '\n');
  OpBuilder b(funcOp->getContext());
  /// Start by bufferizing `funcOp` arguments.
  if (failed(bufferize(b, funcOp, bvm, aliasInfo)))
    return failure();
  // Walk in PreOrder to ensure ops with regions are handled before their body.
  WalkResult result = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    LogicalResult status =
        TypeSwitch<Operation *, LogicalResult>(op)
            // Skip BufferCast and TensorLoad ops.
            // clang-format off
            .Case<memref::BufferCastOp,
                  memref::TensorLoadOp>(
                [&](auto) { return success(); })
            .Case<memref::DimOp,
                  scf::ForOp,
                  LinalgOp,
                  ReturnOp,
                  ExtractSliceOp,
                  InsertSliceOp,
                  VectorTransferOpInterface,
                  scf::YieldOp>(
                [&](auto op) {
                  LDBG("Begin buferize:\n" << op << '\n');
                  return bufferize(b, op, bvm, aliasInfo);
                })
            // clang-format on
            .Default([&](Operation *op) {
              auto isaTensor = [](Type t) { return t.isa<TensorType>(); };
              if (any_of(op->getOperandTypes(), isaTensor) ||
                  any_of(op->getResultTypes(), isaTensor))
                return failure();
              return success();
            });
    if (failed(status)) {
      op->emitError("Failed bufferization");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  LDBG("End BufferizeFuncOpInternals:\n" << funcOp << '\n');

  return failure(result.wasInterrupted());
}

namespace {
struct LinalgComprehensiveFuncBufferize
    : public LinalgComprehensiveFuncBufferizeBase<
          LinalgComprehensiveFuncBufferize> {
  void runOnFunction() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect>();
  }
};
} // end namespace

void LinalgComprehensiveFuncBufferize::runOnFunction() {
  auto funcOp = getFunction();

  // Analysis phase.
  DominanceInfo domInfo(funcOp);
  BufferizationAliasInfo aliasInfo(funcOp);
  // If the analysis fails, just return. This is expected to reset the IR and no
  // single OpResult should be marked inPlace.
  if (failed(inPlaceAnalysisFuncOpInternals(funcOp, aliasInfo, domInfo))) {
    signalPassFailure();
    return;
  }

  if (testAnalysisOnly)
    return;

  // Bufferization phase.
  BlockAndValueMapping bvm;
  if (failed(bufferizeFuncOpInternals(funcOp, bvm, aliasInfo)))
    signalPassFailure();

  // Post-pass cleanup of inplaceable attributes.
  funcOp.walk([&](Operation *op) { op->removeAttr(kInPlaceResultsAttrName); });
}

std::unique_ptr<Pass> mlir::createLinalgComprehensiveFuncBufferizePass() {
  return std::make_unique<LinalgComprehensiveFuncBufferize>();
}
