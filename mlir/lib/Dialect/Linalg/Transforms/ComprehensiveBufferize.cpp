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
//  a. performing an inPlace analysis `inPlaceAnalysisFuncOpBody`
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

#include "mlir/Dialect/Linalg/Transforms/ComprehensiveBufferize.h"

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "comprehensive-module-bufferize"

using namespace mlir;
using namespace linalg;
using namespace tensor;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X)

// TODO: from some HW description.
static constexpr int64_t kBufferAlignments = 128;

// Forward declarations.
static std::string printOperationInfo(Operation *, bool prefix = true);
static std::string printValueInfo(Value, bool prefix = true);

//===----------------------------------------------------------------------===//
// Generic helpers.
//===----------------------------------------------------------------------===//

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

/// Return the FuncOp called by `callOp`.
static FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// Return the unique ReturnOp that terminates `funcOp`.
/// Return nullptr if there is no such unique ReturnOp.
static ReturnOp getAssumedUniqueReturnOp(FuncOp funcOp) {
  ReturnOp returnOp;
  for (Block &b : funcOp.body()) {
    if (auto candidateOp = dyn_cast<ReturnOp>(b.getTerminator())) {
      if (returnOp)
        return nullptr;
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

/// Return true if `value` is the result of an InitTensorOp or a cast thereof.
static bool isInitTensorOp(Value value) {
  tensor::CastOp castOp;
  while ((castOp = value.getDefiningOp<tensor::CastOp>()))
    value = castOp.source();
  return value.getDefiningOp<InitTensorOp>();
}

//===----------------------------------------------------------------------===//
// Bufferization-specific BlockAndValueMapping support with debugging.
//===----------------------------------------------------------------------===//

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, ValueRange keys, ValueRange values) {
  assert(!keys.empty() && "Unexpected empty keys");
  LDBG("\n\tMap: " << printValueInfo(keys.front())
                   << "\n\tto: " << printValueInfo(values.front()) << '\n');
  return bvm.map(keys, values);
}

/// Wrapper for better debugging.
static void map(BlockAndValueMapping &bvm, Value key, Value value) {
  LDBG("\n\tMap: " << printValueInfo(key) << "\n\tto: " << printValueInfo(value)
                   << '\n');
  return bvm.map(key, value);
}

/// Wrapper for better debugging.
static Value lookup(const BlockAndValueMapping &bvm, Value key) {
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
  LDBG("In func:\n" << *parentOp << "\nNO VALUE FOR KEY: " << key << '\n');
  (void)parentOp;
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
  LDBG("->set inPlace=" << stringify(inPlace) << " <- #"
                        << opResult.getResultNumber() << ": "
                        << printOperationInfo(op) << "\n");
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
  // Interestingly, scf::ForOp's and TiledLoop's bbArg can **always** be viewed
  // inplace from the perspective of ops nested under:
  //   1. Either the matching iter operand is not bufferized inplace and an
  //      alloc + optional copy makes the bbArg itself inplaceable.
  //   2. Or the matching iter operand is bufferized inplace and bbArg just
  //      bufferizes to that too.
  if (isa<scf::ForOp, TiledLoopOp>(bbArg.getOwner()->getParentOp()))
    return InPlaceSpec::True;
  // Unknown cases.
  return InPlaceSpec::None;
}

/// Set the attribute that triggers inplace bufferization on a FuncOp argument
/// `bbArg`.
static void
setInPlaceFuncArgument(BlockArgument bbArg,
                       InPlaceSpec inPlaceSpec = InPlaceSpec::True) {
  auto funcOp = cast<FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.setArgAttr(
      bbArg.getArgNumber(), LinalgDialect::kInplaceableAttrName,
      BoolAttr::get(bbArg.getContext(), inPlaceSpec == InPlaceSpec::True));
}

/// Remove the attribute that triggers inplace bufferization on a FuncOp
/// argument `bbArg`.
static void removeBufferizationFuncArguments(BlockArgument bbArg) {
  auto funcOp = cast<FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       LinalgDialect::kBufferLayoutAttrName);
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       LinalgDialect::kInplaceableAttrName);
}

LLVM_ATTRIBUTE_UNUSED static InPlaceSpec getInPlace(Value v) {
  if (auto bbArg = v.dyn_cast<BlockArgument>())
    return getInPlace(bbArg);
  return getInPlace(v.cast<OpResult>());
}

//===----------------------------------------------------------------------===//
// Printing helpers.
//===----------------------------------------------------------------------===//

/// Helper method printing the bufferization information of a buffer / tensor.
static void printTensorOrBufferInfo(std::string prefix, Value value,
                                    AsmState &state, llvm::raw_ostream &os) {
  if (!value.getType().isa<ShapedType>())
    return;
  os << prefix;
  value.printAsOperand(os, state);
  os << " : " << value.getType();
  if (getInPlace(value) == InPlaceSpec::None)
    return;
  os << " [InPlace=" << stringify(getInPlace(value)) << "]";
}

/// Print the operation name and bufferization information.
static std::string printOperationInfo(Operation *op, bool prefix) {
  std::string result;
  llvm::raw_string_ostream os(result);
  AsmState state(op->getParentOfType<mlir::FuncOp>());
  StringRef tab = prefix ? "\n[" DEBUG_TYPE "]\t" : "";
  os << tab << op->getName();
  SmallVector<Value> shapedOperands;
  for (OpOperand &opOperand : op->getOpOperands()) {
    std::string prefix =
        llvm::formatv("{0}  -> #{1} ", tab, opOperand.getOperandNumber());
    printTensorOrBufferInfo(prefix, opOperand.get(), state, os);
  }
  for (OpResult opResult : op->getOpResults()) {
    std::string prefix =
        llvm::formatv("{0}  <- #{1} ", tab, opResult.getResultNumber());
    printTensorOrBufferInfo(prefix, opResult, state, os);
  }
  return result;
}

/// Print the bufferization information for the defining op or block argument.
static std::string printValueInfo(Value value, bool prefix) {
  auto *op = value.getDefiningOp();
  if (op)
    return printOperationInfo(op, prefix);
  // Print the block argument bufferization information.
  std::string result;
  llvm::raw_string_ostream os(result);
  AsmState state(value.getParentRegion()->getParentOfType<mlir::FuncOp>());
  os << value;
  printTensorOrBufferInfo("\n\t - ", value, state, os);
  return result;
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
      isa<CallOpInterface,
          tensor::CastOp,
          ConstantOp,
          tensor::DimOp,
          ExtractSliceOp,
          scf::ForOp,
          InsertSliceOp,
          InitTensorOp,
          LinalgOp,
          ReturnOp,
          TiledLoopOp,
          VectorTransferOpInterface,
          linalg::YieldOp,
          scf::YieldOp>(op)
      // clang-format on
      || (none_of(op->getResultTypes(), isaTensor) &&
          none_of(op->getOperandTypes(), isaTensor));
}

/// Return the OpResult that may bufferize into the same buffer as `opOperand`
/// when the op is bufferized inplace.
/// Return null if no such result exists.
static OpResult getInplaceableOpResult(TiledLoopOp op, OpOperand &opOperand) {
  return op.getTiedOpResult(opOperand);
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
/// Return null if no such result exists.
static OpResult getInplaceableOpResult(tensor::CastOp op,
                                       OpOperand &opOperand) {
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
        .Case<tensor::CastOp,
              scf::ForOp,
              InsertSliceOp,
              LinalgOp,
              TiledLoopOp,
              VectorTransferOpInterface>(
            [&](auto op) { return getInplaceableOpResult(op, opOperand); })
        // ExtractSliceOp is special, when bufferized inplace it just returns an
        // alias to its operand. Its result is never inplaceable on its operand.
        .Case([&](ExtractSliceOp op) { return OpResult(); })
        // CallOpInterface is special, it needs to wait for the callee to be
        // bufferized and needs to inspect the BufferAliasInfo object. It can't
        // make a proper determination by itself and needs to be conservative.
        .Case([&](CallOpInterface op) { return OpResult(); })
        // Other ops.
        .Default([&](Operation *op) { return OpResult(); });
  // clang-format on
}

/// Determine which OpOperand* will alias with `result` if the op is bufferized
/// in place.
/// Return None if the owner of `opOperand` does not have known
/// bufferization aliasing behavior, which indicates that the op must allocate
/// all of its tensor results.
/// TODO: in the future this may need to evolve towards a list of OpOperand*.
static Optional<OpOperand *> getAliasingOpOperand(OpResult result) {
  if (!hasKnownBufferizationAliasingBehavior(result.getDefiningOp()))
    return None;
  return TypeSwitch<Operation *, OpOperand *>(result.getDefiningOp())
      .Case([&](tensor::CastOp op) { return &op->getOpOperand(0); })
      .Case([&](ConstantOp op) { return &op->getOpOperand(0); })
      .Case([&](ExtractSliceOp op) { return &op->getOpOperand(0); })
      // In the case of scf::ForOp, this currently assumes the iter_args / yield
      // are 1-1. This may fail and is verified at the end.
      // TODO: update this.
      .Case([&](scf::ForOp op) {
        return &op.getIterOpOperands()[result.getResultNumber()];
      })
      .Case([&](InitTensorOp op) { return nullptr; })
      .Case([&](InsertSliceOp op) { return &op->getOpOperand(1); })
      .Case([&](LinalgOp op) {
        return op.getOutputTensorOperands()[result.getResultNumber()];
      })
      .Case([&](TiledLoopOp op) {
        // TODO: TiledLoopOp helper method to avoid leaking impl details.
        return &op->getOpOperand(op.getNumControlOperands() +
                                 op.getNumInputs() + result.getResultNumber());
      })
      .Case([&](vector::TransferWriteOp op) { return &op->getOpOperand(1); })
      .Default([&](Operation *op) {
        op->dump();
        llvm_unreachable("unexpected defining op");
        return nullptr;
      });
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
      .Case<ReturnOp, linalg::InitTensorOp, linalg::YieldOp, scf::YieldOp>(
          [&](auto op) { return OpResult(); })
      // DimOp has no tensor result.
      .Case<tensor::DimOp>([&](auto op) { return None; })
      // ConstantOp is never inplaceable.
      .Case([&](ConstantOp op) { return op->getResult(0); })
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
  if (auto forOp = dyn_cast<scf::ForOp>(opOperand.getOwner())) {
    for (OpOperand &use :
         forOp.getRegionIterArgForOpOperand(opOperand).getUses())
      if (bufferizesToMemoryRead(use))
        return true;
    return false;
  }
  // TiledLoop alone doesn't bufferize to a memory read, one of the uses of its
  // matching bbArg may.
  if (auto tiledLoopOp = dyn_cast<TiledLoopOp>(opOperand.getOwner())) {
    for (OpOperand &use : tiledLoopOp.getTiedBlockArgument(opOperand).getUses())
      if (bufferizesToMemoryRead(use))
        return true;
    return false;
  }
  // CallOpInterface alone doesn't bufferize to a memory read, one of the uses
  // of the matching bbArg may. It is the responsibility of the caller to
  // inspect bbArgs. In the absence of a BufferizationAliasInfo, we need to be
  // conservative.
  if (auto callOp = dyn_cast<CallOpInterface>(opOperand.getOwner()))
    return true;
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
  // These terminators are not writes.
  if (isa<ReturnOp, linalg::YieldOp, scf::YieldOp>(opOperand.getOwner()))
    return false;
  // ExtractSliceOp alone doesn't bufferize to a memory write, one of its uses
  // may.
  if (isa<ExtractSliceOp>(opOperand.getOwner()))
    return false;
  // CallOpInterface alone doesn't bufferize to a memory write, one of the uses
  // of the matching bbArg may. It is the responsibility of the caller to
  // inspect bbArgs. In the absence of a BufferizationAliasInfo, we need to be
  // conservative.
  if (auto callOp = dyn_cast<CallOpInterface>(opOperand.getOwner()))
    return true;
  Optional<OpResult> maybeOpResult = getAliasingOpResult(opOperand);
  // Unknown op that returns a tensor. The inplace analysis does not support
  // it. Conservatively return true.
  if (!maybeOpResult)
    return true;
  // Supported op without a matching result for opOperand (e.g. ReturnOp).
  // This does not bufferize to a write.
  if (!*maybeOpResult)
    return false;
  // If we have a matching OpResult, this is a write.
  // Additionally allow to restrict to only inPlace write, if so specified.
  return inPlaceSpec == InPlaceSpec::None ||
         getInPlace(*maybeOpResult) == inPlaceSpec;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific alias analysis.
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

/// Return true if the buffer to which `operand` would bufferize aliases a
/// buffer that is known to not be writable. This implies that the matching
/// OpResult cannot be bufferized inplace.
bool BufferizationAliasInfo::aliasesNonWritableBuffer(
    OpOperand &operand) const {
  LDBG("----Start aliasesNonWritableBuffer\n");
  LDBG("-------for -> #" << operand.getOperandNumber() << ": "
                         << printOperationInfo(operand.getOwner()) << '\n');
  for (Value v : getAliases(operand.get())) {
    LDBG("-----------examine: " << printValueInfo(v) << '\n');
    if (bufferizesToWritableMemory(v)) {
      LDBG("-----------Value is known to be writeable -> skip: "
           << printValueInfo(v) << '\n');
      continue;
    }

    if (auto bbArg = v.dyn_cast<BlockArgument>()) {
      if (getInPlace(bbArg) == InPlaceSpec::True) {
        LDBG("-----------bbArg is writeable -> skip: " << printValueInfo(bbArg)
                                                       << '\n');
        continue;
      }
      LDBG("-----------notWriteable\n");
      return true;
    }

    if (Operation *op = v.getDefiningOp()) {
      if (isa<ConstantOp>(op) || !hasKnownBufferizationAliasingBehavior(op)) {
        LDBG("-----------notWritable\n");
        return true;
      }
    }
  }
  LDBG("---->operand is writable\n");
  return false;
}

bool BufferizationAliasInfo::bufferizesToWritableMemory(Value v) const {
  return bufferizeToWritableMemory.count(v) > 0;
}

/// Specify that the value is known to bufferize to writable memory.
void BufferizationAliasInfo::setBufferizesToWritableMemory(Value v) {
  bufferizeToWritableMemory.insert(v);
}

/// Return true if the buffer to which `operand` would bufferize is equivalent
/// to some buffer write.
bool BufferizationAliasInfo::aliasesInPlaceWrite(Value value) const {
  LDBG("----Start aliasesInPlaceWrite\n");
  LDBG("-------for : " << printValueInfo(value) << '\n');
  for (Value v : getAliases(value)) {
    for (auto &use : v.getUses()) {
      if (bufferizesToMemoryWrite(use, InPlaceSpec::True)) {
        LDBG("-----------wants to bufferize to inPlace write: "
             << printOperationInfo(use.getOwner()) << '\n');
        return true;
      }
    }
  }
  LDBG("----------->does not alias an inplace write\n");
  return false;
}

/// Set the inPlace bufferization spec to true.
void BufferizationAliasInfo::bufferizeInPlace(OpResult result,
                                              OpOperand &operand,
                                              BufferRelation bufferRelation) {
  setInPlaceOpResult(result, InPlaceSpec::True);
  aliasInfo.unionSets(result, operand.get());
  // Dump the updated alias analysis.
  LLVM_DEBUG(dumpAliases());
  if (bufferRelation == BufferRelation::Equivalent)
    equivalentInfo.unionSets(result, operand.get());
  // Dump the updated equivalence analysis.
  LLVM_DEBUG(dumpEquivalences());
}

/// Set the inPlace bufferization spec to false.
void BufferizationAliasInfo::bufferizeOutOfPlace(OpResult result) {
  setInPlaceOpResult(result, InPlaceSpec::False);
}

/// Return true if it is possible to find an inplace write W among `usesWrite`
/// and a read R among `usesRead`, such that W and R interfere.
bool BufferizationAliasInfo::wouldCreateReadAfterWriteInterference(
    Operation *opToBufferize, DenseSet<OpOperand *> &usesRead,
    DenseSet<OpOperand *> &usesWrite, const DominanceInfo &domInfo) const {
  for (OpOperand *uRead : usesRead) {
    Operation *aliasingReadOp = uRead->getOwner();
    LDBG("----++++aliasRead -> #"
         << uRead->getOperandNumber()
         << " in: " << printOperationInfo(aliasingReadOp) << '\n');
    for (OpOperand *uWrite : usesWrite) {
      // The same operand may both read and write.
      // Don't consider self-use of the same operand for interference.
      // Multiple different uses within the same op is fair game though.
      if (uWrite == uRead)
        continue;

      Operation *aliasingWriteOp = uWrite->getOwner();
      LDBG("----    aliasWrite -> #"
           << uWrite->getOperandNumber()
           << " in: " << printOperationInfo(aliasingWriteOp) << '\n');
      // If the candidate write is the one that produces the read value (in the
      // SSA def-use sense), this is not considered an interference.
      if (getInplaceableOpResult(*uWrite) == uRead->get())
        continue;
      // If aliasingReadOp properly dominates aliasingWriteOp, the read cannot
      // be affected by the write: there is no interference.
      if (domInfo.properlyDominates(aliasingReadOp, aliasingWriteOp))
        continue;
      // At this point, aliasingWriteOp properly dominates aliasingReadOp or
      // there is no clear dominance and we need to be conservative.
      LDBG("---->found RaW interference between:\n");
      LDBG("       OpToBufferize -> " << printOperationInfo(opToBufferize)
                                      << '\n');
      LDBG("       Interfering write -> #"
           << uWrite->getOperandNumber() << ":"
           << printOperationInfo(aliasingWriteOp) << '\n');
      LDBG("       Target read -> #" << uRead->getOperandNumber() << ":"
                                     << printOperationInfo(aliasingReadOp)
                                     << '\n');
      LDBG("---->opportunity to clobber RaW interference\n");
      if (isClobberedWriteBeforeRead(opToBufferize, *uRead, *uWrite, domInfo)) {
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

/// Return true if it is possible to find an inplace write W among the uses of
/// aliasInfo[result], and a read R among the uses of aliasInfo[result],
/// such that W and R interfere.
/// Such a (W, R) pair is an interference to the inplace bufferization of
/// opResult when:
///   1. R is not known to properly dominate W (i.e. the effects of the write
///      may be visible from R).
///   2. one cannot find an intermediate clobbering write `C` to W, such that
///      C interleaved between W and R (i.e. W -> C -> R where -> denotes
///      dominance).
bool BufferizationAliasInfo::wouldCreateReadAfterWriteInterference(
    OpResult result, const DominanceInfo &domInfo) const {
  Optional<OpOperand *> maybeAliasingOperand = getAliasingOpOperand(result);
  if (!maybeAliasingOperand)
    return false;

  Operation *opToBufferize = result.getDefiningOp();
  Value opResult = result;
  Value opOperand = (*maybeAliasingOperand)->get();

  LDBG("----Start wouldCreateReadAfterWriteInterference\n");
  LDBG("--------consider all aliases to root read: "
       << printValueInfo(opOperand) << "\n");
  LDBG("--------consider all aliases to root write: "
       << printValueInfo(opResult) << "\n");

  /// Helper function to iterate on aliases of `root` and capture the reads.
  auto getAliasingReads = [&](DenseSet<OpOperand *> &res, Value root) {
    for (Value alias : getAliases(root)) {
      for (auto &use : alias.getUses()) {
        // Read to a value that aliases root.
        if (bufferizesToMemoryRead(use)) {
          LDBG("------------bufferizesToMemoryRead: "
               << use.getOwner()->getName().getStringRef() << "\n");
          res.insert(&use);
        }
      }
    }
  };

  /// Helper function to iterate on aliases of `root` and capture the writes.
  auto getAliasingInplaceWrites = [&](DenseSet<OpOperand *> &res, Value root) {
    for (Value alias : getAliases(root)) {
      for (auto &use : alias.getUses()) {
        // Inplace write to a value that aliases root.
        if (bufferizesToMemoryWrite(use, InPlaceSpec::True)) {
          LDBG("------------bufferizesToMemoryWrite: "
               << use.getOwner()->getName().getStringRef() << "\n");
          res.insert(&use);
        }
      }
    }
  };

  // Check if we can find any interference between reads to aliases[`opOperand`]
  // and writes to aliases[`opResult`]. This handles the case:
  //
  // ```
  //  %0 = op_to_bufferize_maybe_inplace(%1)
  //  %2 = some_alias(%0)
  //  inplace_write(%2)
  //  %3 = some_alias(%1)
  //  read(%3)
  // ```
  DenseSet<OpOperand *> usesRead, usesWrite;
  LDBG("--------\n");
  LDBG("--------Test reads(opOperand) vs writes(opResult)\n");
  getAliasingReads(usesRead, opOperand);
  getAliasingInplaceWrites(usesWrite, opResult);
  // Additionally, `result` is not yet bufferized and we need to check for
  // interferences as if it were bufferized inplace: add `maybeAliasingOperand`
  // if it is a write. This handles the case:
  //
  // ```
  //  %0 = op_to_bufferize_maybe_inplace(%1)
  //  %2 = some_alias(%1)
  //  read(%2)
  // ```
  if (bufferizesToMemoryWrite(**maybeAliasingOperand))
    usesWrite.insert(*maybeAliasingOperand);
  if (wouldCreateReadAfterWriteInterference(opToBufferize, usesRead, usesWrite,
                                            domInfo))
    return true;

  // Check if we can find any interference between writes to
  // aliases[`opOperand`] and reads to aliases[`opResult`]. This handles the
  // case:
  //
  // ```
  //  %0 = op_to_bufferize_maybe_inplace(%1)
  //  %2 = some_alias(%1)
  //  inplace_write(%2)
  //  %3 = some_alias(%0)
  //  read(%3)
  // ```
  LDBG("--------\n");
  LDBG("--------Test reads(opResult) vs writes(opOperand)\n");
  usesRead.clear();
  usesWrite.clear();
  getAliasingReads(usesRead, opResult);
  getAliasingInplaceWrites(usesWrite, opOperand);
  return wouldCreateReadAfterWriteInterference(opToBufferize, usesRead,
                                               usesWrite, domInfo);
}

/// Return true if the source of a `insertSliceOp` bufferizes to an
/// equivalent ExtractSliceOp that bufferizes inplace.
bool BufferizationAliasInfo::isSourceEquivalentToAMatchingInplaceExtractSliceOp(
    InsertSliceOp insertSliceOp) const {
  LDBG("isSourceEquivalentToAMatchingInplaceExtractSliceOp: " << *insertSliceOp
                                                              << '\n');
  auto leaderIt = equivalentInfo.findLeader(insertSliceOp.source());
  for (auto mit = leaderIt, meit = equivalentInfo.member_end(); mit != meit;
       ++mit) {
    auto extractSliceOp =
        dyn_cast_or_null<ExtractSliceOp>(mit->v.getDefiningOp());
    if (extractSliceOp &&
        areEquivalentExtractSliceOps(extractSliceOp, insertSliceOp) &&
        getInPlace(extractSliceOp.result()) == InPlaceSpec::True) {
      LDBG("\tfound: " << *mit->v.getDefiningOp() << '\n');
      return true;
    }
  }
  LDBG("\tnot equivalent\n");
  return false;
}

/// Apply `fun` to all the members of the equivalence class of `v`.
void BufferizationAliasInfo::applyOnEquivalenceClass(
    Value v, function_ref<void(Value)> fun) const {
  auto leaderIt = equivalentInfo.findLeader(v);
  for (auto mit = leaderIt, meit = equivalentInfo.member_end(); mit != meit;
       ++mit) {
    fun(mit->v);
  }
}

void BufferizationAliasInfo::printAliases(raw_ostream &os) const {
  os << "\n/===================== AliasInfo =====================\n";
  for (auto it = aliasInfo.begin(), eit = aliasInfo.end(); it != eit; ++it) {
    if (!it->isLeader())
      continue;
    Value leader = it->getData();
    os << "|\n| -- leader: " << printValueInfo(leader, /*prefix=*/false)
       << '\n';
    for (auto mit = aliasInfo.member_begin(it), meit = aliasInfo.member_end();
         mit != meit; ++mit) {
      Value v = static_cast<Value>(*mit);
      os << "| ---- aliasing member: " << printValueInfo(v, /*prefix=*/false)
         << '\n';
    }
  }
  os << "\n/===================== End AliasInfo =====================\n\n";
}

void BufferizationAliasInfo::printEquivalences(raw_ostream &os) const {
  os << "\n/********************* Equivalent Buffers *********************\n";
  for (auto it = equivalentInfo.begin(), eit = equivalentInfo.end(); it != eit;
       ++it) {
    if (!it->isLeader())
      continue;
    Value leader = it->getData();
    os << "|\n| -- leader: " << printValueInfo(leader, /*prefix=*/false)
       << '\n';
    for (auto mit = equivalentInfo.member_begin(it),
              meit = equivalentInfo.member_end();
         mit != meit; ++mit) {
      Value v = static_cast<Value>(*mit);
      os << "| ---- equivalent member: " << printValueInfo(v, /*prefix=*/false)
         << '\n';
    }
  }
  os << "|\n\\***************** End Equivalent Buffers *****************\n\n";
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

void BufferizationAliasInfo::dumpAliases() const { printAliases(llvm::errs()); }

void BufferizationAliasInfo::dumpEquivalences() const {
  printEquivalences(llvm::errs());
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
      Operation *candidateOp = mit->v.getDefiningOp();
      if (!candidateOp)
        continue;
      auto maybeAliasingOperand = getAliasingOpOperand(mit->v.cast<OpResult>());
      if (!maybeAliasingOperand || !*maybeAliasingOperand ||
          !bufferizesToMemoryWrite(**maybeAliasingOperand))
        continue;
      LDBG("---->clobbering candidate: " << printOperationInfo(candidateOp)
                                         << '\n');
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
    Operation *opToBufferize, OpOperand &aliasingRead, OpOperand &aliasingWrite,
    const DominanceInfo &domInfo) const {
  Operation *aliasingReadOp = aliasingRead.getOwner();
  Operation *aliasingWriteOp = aliasingWrite.getOwner();
  assert(!domInfo.properlyDominates(aliasingReadOp, aliasingWriteOp) &&
         "Unexpected aliasingReadOp properly dominates aliasingWriteOp");

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
// Forward declarations.
//===----------------------------------------------------------------------===//

/// Return the op with Allocate MemoryEffect if `v` is equivalent to an such
/// an op. Return null otherwise.
static Operation *getEquivalentAlloc(Value value,
                                     const BufferizationAliasInfo &aliasInfo);

/// Return the first argument of the enclosing FuncOp that is equivalent to `v`.
/// Return null if no such bbArg can be found.
static BlockArgument
getEquivalentEnclosingFuncBBArg(Value v,
                                const BufferizationAliasInfo &aliasInfo);

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

/// Return the FunctionType with `argumentTypes` and `resultTypes` where each
/// tensor is replaced by the corresponding buffer type.
/// In order for all the callers to agree, this *must* bufferize to the most
/// dynamic buffer type supported.
/// A later pass across all CallOps in the module can decide whether to simplify
/// the types of to version according to some cost model.
static FunctionType getBufferizedFunctionType(MLIRContext *ctx,
                                              TypeRange argumentTypes,
                                              TypeRange resultTypes) {
  auto rewrite = [](Type t) -> Type {
    // TODO: non-zero address space.
    // TODO: layout information if relevant.
    if (auto rankedTensorType = t.dyn_cast<RankedTensorType>())
      return getDynamicMemRefType(rankedTensorType);
    if (auto tensorType = t.dyn_cast<TensorType>())
      return getContiguousOrUnrankedMemRefType(tensorType);
    return t;
  };
  auto argTypes = llvm::to_vector<4>(llvm::map_range(argumentTypes, rewrite));
  auto retTypes = llvm::to_vector<4>(llvm::map_range(resultTypes, rewrite));
  return FunctionType::get(ctx, argTypes, retTypes);
}

/// If an entry for `funcOp` is available in `bufferizedFunctionTypes`, return
/// it. Otherwise, construct a new entry based on `argumentTypes` and
/// `resultTypes`.
// TODO: improve the layering.
static FunctionType getOrCreateBufferizedFunctionType(
    FuncOp funcOp, TypeRange argumentTypes, TypeRange resultTypes,
    DenseMap<FuncOp, FunctionType> &bufferizedFunctionTypes) {
  auto it = bufferizedFunctionTypes.find(funcOp);
  if (it != bufferizedFunctionTypes.end())
    return it->second;

  auto it2 = bufferizedFunctionTypes.try_emplace(
      funcOp, getBufferizedFunctionType(funcOp.getContext(), argumentTypes,
                                        resultTypes));
  LDBG("FT: " << funcOp.getType() << " -> " << it2.first->second << "\n");
  return it2.first->second;
}

//===----------------------------------------------------------------------===//
// Bufferization-specific scoped alloc/dealloc insertion support.
//===----------------------------------------------------------------------===//

template <typename... Args>
Operation *getFirstParentOfType(Value v) {
  Operation *parent;
  if (auto bbArg = v.dyn_cast<BlockArgument>())
    parent = bbArg.getOwner()->getParentOp();
  else
    parent = v.getDefiningOp()->getParentOp();
  while (parent) {
    if (isa<Args...>(parent))
      return parent;
    parent = parent->getParentOp();
  }
  return nullptr;
}

/// Create an Allocop/DeAllocOp pair, where the AllocOp is after
/// `shapedValue.getDefiningOp` (or at the top of the block in case of a
/// bbArg) and the DeallocOp is at the end of the block.
static Value
createNewAllocDeallocPairForShapedValue(OpBuilder &b, Location loc,
                                        Value shapedValue,
                                        BufferizationAliasInfo &aliasInfo) {
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
      dynShape.push_back(createOrFoldDimOp(b, loc, shapedValue, dim.index()));

  // If the buffer is statically shaped, try to hoist it to the first enclosing
  // parallel region.
  // TODO: this concept of parallel region and threadlocal needs interfaces.
  // TODO: also hoist in the dynamic case. For now this relies on subsequent
  // calls to LICM and buffer hoisting which will most likely not succeed.
  // TODO: when packing, allocate a static bounding box which will enable more
  // hoisting.
  Value allocated;
  { // Guarded insertion point to potentially hoist the AllocOp.
    OpBuilder::InsertionGuard g(b);
    if (dynShape.empty()) {
      Operation *parent =
          getFirstParentOfType<FuncOp, TiledLoopOp, scf::ParallelOp,
                               AffineParallelOp>(shapedValue);
      if (parent)
        b.setInsertionPointToStart(&(parent->getRegion(0).front()));
    }
    allocated = b.create<memref::AllocOp>(
        loc, allocMemRefType, dynShape, b.getI64IntegerAttr(kBufferAlignments));
    aliasInfo.createAliasInfoEntry(allocated);
  }
  Value casted = allocated;
  if (memRefType != allocMemRefType) {
    casted = b.create<memref::CastOp>(loc, memRefType, allocated);
    aliasInfo.insertNewBufferEquivalence(casted, allocated);
  }
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
static void allocateBuffersForResults(OpBuilder &b, Location loc, LinalgOp op,
                                      SmallVectorImpl<Value> &resultBuffers,
                                      BlockAndValueMapping &bvm,
                                      BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);

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
      assert(v && "missing buffer");
      resultBuffers.push_back(v);
      continue;
    }

    // Otherwise, `op` is not inplaceable and we need to allocate its result.
    Value dimTensor = bvm.lookupOrDefault(output);
    Value alloc =
        createNewAllocDeallocPairForShapedValue(b, loc, dimTensor, aliasInfo);
    resultBuffers.push_back(alloc);

    // Additionally, if the output buffer is used, clone its value for now.
    if (op.payloadUsesValueFromOperand(opOperand)) {
      Value v = lookup(bvm, output);
      b.create<CopyOp>(loc, v, alloc);
    }
  }

  if (op->getNumResults())
    map(bvm, op->getResults(), resultBuffers);
}

/// Generic conversion for any LinalgOp on tensors.
static LogicalResult bufferize(OpBuilder &b, LinalgOp op,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  Location loc = op.getLoc();
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    newInputBuffers.push_back(lookup(bvm, opOperand->get()));
    assert(newInputBuffers.back() && "missing buffer");
  }
  SmallVector<Value> newOutputBuffers;
  // Try to allocate new buffers depending on op's inplace semantics.
  allocateBuffersForResults(b, loc, op, newOutputBuffers, bvm, aliasInfo);

  // Clone the newly bufferized op.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  b.setInsertionPoint(op);
  op.clone(b, loc, /*resultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  if (op->getNumResults())
    map(bvm, op->getResults(), newOutputBuffers);

  // The original op will be DCE'd away later.

  return success();
}

/// In a first approximation, all the function arguments of a FuncOp are marked
/// inplaceable. For now, it is the responsibility of the `callOp` bufferization
/// to allow FuncOp that are inplaceable to write inPlace.
static LogicalResult
bufferize(OpBuilder &b, CallOpInterface callOp, BlockAndValueMapping &bvm,
          BufferizationAliasInfo &aliasInfo,
          DenseMap<FuncOp, FunctionType> &bufferizedFunctionTypes) {
  FuncOp funcOp = getCalledFunction(callOp);
  assert(isa<CallOp>(callOp.getOperation()) && funcOp &&
         "expected Callop to a FuncOp");

  // If nothing to do then we are done.
  if (!llvm::any_of(funcOp.getType().getInputs(), isaTensor) &&
      !llvm::any_of(funcOp.getType().getResults(), isaTensor))
    return success();

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(callOp);

  // 1. Filter return types:
  //    - if the callee is bodiless / external, we cannot inspect it and we
  //      cannot assume anything. We can just assert that it does not return a
  //      tensor as this would have to bufferize to "return a memref", whose
  //      semantics is ill-defined.
  //    - if the callee has a body, we perform inter-procedural equivalence
  //      analysis. When successful, a result folds onto an operand. When
  //      unsuccessful, additional work is needed to either:
  //        * hoist a result into an inplaceable operand or
  //        * devise a better representation to truly return a buffer.
  SmallVector<Type> resultTypes;
  SmallVector<Value> hoistedArguments;
  if (funcOp.body().empty()) {
    if (llvm::any_of(funcOp.getType().getResults(), isaTensor))
      return callOp->emitError()
             << "cannot bufferize bodiless function that returns a tensor";
  } else {
    ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
    assert(returnOp && "expected func with single return op");

    // For each FuncOp result, keep track of which inplace argument it reuses.
    for (OpOperand &returnOperand : returnOp->getOpOperands()) {
      Type returnType = returnOperand.get().getType();
      if (!isaTensor(returnType)) {
        resultTypes.push_back(returnType);
        continue;
      }

      // If return operand is equivalent to some bbArg, no need to return it.
      Value returnVal = returnOperand.get();
      if (BlockArgument bbArg =
              getEquivalentEnclosingFuncBBArg(returnVal, aliasInfo)) {
        Value oldRes = callOp->getResult(returnOperand.getOperandNumber());
        int64_t idx = bbArg.getArgNumber();
        Value buffer = lookup(bvm, callOp->getOperand(idx));
        assert(buffer && "expected bufferized value");
        // Add CallOp operand/result equivalence: this is interprocedural info.
        aliasInfo.insertNewBufferEquivalence(oldRes, buffer);
        map(bvm, oldRes, buffer);
        // Add a TensorLoadOp to kill all uses of the CallOp return.
        // Replace all uses of the CallOp results so we can erase the CallOp.
        // This TensorLoadOp must fold/DCE away or bufferization should be
        // considered failed.
        Value tensorLoad =
            b.create<memref::TensorLoadOp>(callOp.getLoc(), buffer);
        oldRes.replaceAllUsesWith(tensorLoad);
        // Add new op equivalence info.
        aliasInfo.insertNewBufferEquivalence(tensorLoad, buffer);
        map(bvm, tensorLoad, buffer);
        continue;
      }

      // TODO: Need to hoist above function boundary.
      if (Operation *allocOp = getEquivalentAlloc(returnVal, aliasInfo)) {
        hoistedArguments.push_back(allocOp->getResult(0));
        continue;
      }

      // Other cases legitimately need to return a tensor, this is currently not
      // supported. For instance, if hoisting across function boundary has
      // failed, it may be due to e.g. data-dependent sizes. In such a case, we
      // would we need a better type than memref.
      resultTypes.push_back(returnType);

      int64_t returnIdx = returnOperand.getOperandNumber();
      return returnOp->emitError()
             << "buffer result #" << returnIdx << " not produced by an alloc\n";
    }
  }

  // 2. Compute bufferized FunctionType.
  SmallVector<Type> argumentTypes{callOp->getOperandTypes()};
  ValueRange hoistedArgs{hoistedArguments};
  llvm::append_range(argumentTypes, hoistedArgs.getTypes());
  // Get the bufferized FunctionType for funcOp or construct it if not yet
  // available.
  FunctionType bufferizedFuncType = getOrCreateBufferizedFunctionType(
      funcOp, argumentTypes, resultTypes, bufferizedFunctionTypes);

  // 3. Rewrite tensor operands as memrefs based on `bufferizedFuncType`.
  SmallVector<Value> newOperands;
  newOperands.reserve(callOp->getNumOperands());
  for (OpOperand &opOperand : callOp->getOpOperands()) {
    Value tensorOperand = opOperand.get();
    // Non-tensor operands are just copied.
    if (!tensorOperand.getType().isa<TensorType>()) {
      newOperands.push_back(tensorOperand);
      continue;
    }

    // Tensor operands are guaranteed to have been buferized.
    int64_t idx = opOperand.getOperandNumber();
    Value buffer = lookup(bvm, tensorOperand);
    assert(buffer && "expected bufferized value");

    // Caller / callee type mistmatch is handled with a CastOp.
    auto memRefType = bufferizedFuncType.getInput(idx);
    // Since we don't yet have a clear layout story, buffer_cast may
    // conservatively turn tensors into more dynamic memref than necessary.
    // If the memref type of the callee fails, introduce an extra memref.cast
    // that will either canonicalize away or fail compilation until we can do
    // something better.
    if (buffer.getType() != memRefType) {
      Value castBuffer =
          b.create<memref::CastOp>(callOp.getLoc(), memRefType, buffer);
      // Add new op equivalence info.
      aliasInfo.insertNewBufferEquivalence(castBuffer, buffer);
      map(bvm, tensorOperand, castBuffer);
      buffer = castBuffer;
    }
    newOperands.push_back(buffer);
  }

  // 4. Create the new CallOp.
  Operation *newCallOp = b.create<CallOp>(callOp.getLoc(), funcOp.sym_name(),
                                          resultTypes, newOperands);
  newCallOp->setAttrs(callOp->getAttrs());
  return success();
}

/// tensor::CastOp bufferizes to memref::CastOp.
static LogicalResult bufferize(OpBuilder &b, tensor::CastOp castOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(castOp);

  Type sourceType = lookup(bvm, castOp.source()).getType();
  auto rankedMemRefType = sourceType.dyn_cast<MemRefType>();
  auto unrankedMemRefType = sourceType.dyn_cast<UnrankedMemRefType>();
  assert(rankedMemRefType || unrankedMemRefType);
  unsigned memorySpace = rankedMemRefType
                             ? rankedMemRefType.getMemorySpaceAsInt()
                             : unrankedMemRefType.getMemorySpaceAsInt();
  TensorType tensorType = castOp.getResult().getType().cast<TensorType>();
  ArrayRef<AffineMap> affineMaps =
      rankedMemRefType && tensorType.isa<RankedTensorType>()
          ? rankedMemRefType.getAffineMaps()
          : ArrayRef<AffineMap>{};
  Type memRefType = getContiguousOrUnrankedMemRefType(
      castOp.getResult().getType(), affineMaps, memorySpace);
  Value res = b.create<memref::CastOp>(castOp.getLoc(), memRefType,
                                       lookup(bvm, castOp.source()));
  aliasInfo.insertNewBufferEquivalence(res, castOp.getResult());
  map(bvm, castOp.getResult(), res);
  return success();
}

static LogicalResult bufferize(OpBuilder &b, ConstantOp constantOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo,
                               GlobalCreator &globalCreator) {
  assert(constantOp.getType().dyn_cast<RankedTensorType>() &&
         "not a constant ranked tensor");

  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(constantOp);

  auto globalMemref = globalCreator.getGlobalFor(constantOp);
  Value memref = b.create<memref::GetGlobalOp>(
      constantOp.getLoc(), globalMemref.type(), globalMemref.getName());
  aliasInfo.insertNewBufferEquivalence(memref, constantOp.getResult());
  map(bvm, constantOp, memref);

  return success();
}

/// DimOp tensor operand is modified inplace. This allows leaving dead
/// tensors behind that will get DCE'd.
static LogicalResult bufferize(OpBuilder &b, tensor::DimOp dimOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(dimOp);

  if (dimOp.source().getType().isa<RankedTensorType>()) {
    Value v = lookup(bvm, dimOp.source());
    assert(v && "missing buffer");
    dimOp.result().replaceAllUsesWith(
        b.create<memref::DimOp>(dimOp.getLoc(), v, dimOp.index()));
  }
  return success();
}

static LogicalResult bufferize(OpBuilder &b, scf::ForOp forOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // If inPlace, just forward the buffer.
  // Otherwise alloc and copy.
  Location loc = forOp.getLoc();
  for (OpResult opResult : forOp->getResults()) {
    if (!opResult.getType().isa<TensorType>())
      continue;
    // TODO: Atm we bail on unranked TensorType because we don't know how to
    // alloc an UnrankedMemRefType + its underlying ranked MemRefType.
    assert(opResult.getType().isa<RankedTensorType>() &&
           "unsupported unranked tensor");
    OpOperand &opOperand = forOp.getOpOperandForResult(opResult);
    Value operand = opOperand.get();
    Value operandBuffer = lookup(bvm, operand);
    Value resultBuffer = operandBuffer;
    if (getInPlace(opResult) != InPlaceSpec::True) {
      resultBuffer =
          createNewAllocDeallocPairForShapedValue(b, loc, operand, aliasInfo);
      // If the tensor comes from either:
      //   - linalg.init_tensor
      //   - tensor.cast(linalg.init_tensor())
      // Then the value is unitialized and we do not need to copy. This is a
      // pragmatic simplification of "matching bbArg does not bufferize to a
      // read".
      // TODO: "matching bbArg does not bufferize to a read" is a more general
      // check.
      if (!isInitTensorOp(operand)) {
        OpBuilder::InsertionGuard g(b);
        // Set insertion point now that potential alloc/dealloc are introduced.
        // Copy is inserted just before the forOp.
        b.setInsertionPoint(forOp);
        b.create<linalg::CopyOp>(forOp.getLoc(), operandBuffer, resultBuffer);
      }
    }
    BlockArgument bbArg = forOp.getRegionIterArgForOpOperand(opOperand);
    aliasInfo.createAliasInfoEntry(resultBuffer);
    aliasInfo.insertNewBufferEquivalence(bbArg, resultBuffer);
    map(bvm, bbArg, resultBuffer);
    map(bvm, opResult, resultBuffer);
  }

  return success();
}

/// FuncOp always creates TensorToMemRef ops.
static LogicalResult bufferize(OpBuilder &b, FuncOp funcOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
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
    Value bufferCast =
        b.create<memref::BufferCastOp>(funcOp.getLoc(), memRefType, bbArg);
    aliasInfo.insertNewBufferEquivalence(bufferCast, bbArg);
    map(bvm, bbArg, bufferCast);
  }
  return success();
}

/// InitTensor always allocates.
/// TODO: consider hoisting across function boundaries prior to bufferization.
static LogicalResult bufferize(OpBuilder &b, InitTensorOp initTensorOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(initTensorOp);

  Value alloc = createNewAllocDeallocPairForShapedValue(
      b, initTensorOp->getLoc(), initTensorOp.result(), aliasInfo);
  map(bvm, initTensorOp.result(), alloc);
  return success();
}

/// ReturnOp always creates memref::TensorLoadOp.
static LogicalResult bufferize(OpBuilder &b, ReturnOp returnOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  // Cannot insert after returnOp.
  b.setInsertionPoint(returnOp);

  assert(isa<FuncOp>(returnOp->getParentOp()) &&
         "only support FuncOp parent for ReturnOp");
  for (OpOperand &operand : returnOp->getOpOperands()) {
    auto tensorType = operand.get().getType().dyn_cast<TensorType>();
    if (!tensorType)
      continue;
    Value v = lookup(bvm, operand.get());
    assert(v && "missing buffer for result");
    Value returnTensor = b.create<memref::TensorLoadOp>(returnOp.getLoc(), v);
    operand.set(returnTensor);
    aliasInfo.insertNewBufferEquivalence(returnTensor, v);
    map(bvm, returnTensor, v);
  }
  return success();
}

/// Bufferization for TiledLoopOp..
static LogicalResult bufferize(OpBuilder &b, TiledLoopOp tiledLoopOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  // Allocate output buffers if needed, forward output tensor args to the
  // terminator.
  Operation *yieldOp = tiledLoopOp.getBody()->getTerminator();
  Block *body = tiledLoopOp.getBody();

  // Take copies of the old input and output operands, so we can insert inplace
  // easily.
  auto oldInputs = llvm::to_vector<4>(tiledLoopOp.inputs());
  auto oldOutputs = llvm::to_vector<4>(tiledLoopOp.outputs());

  int numLoops = tiledLoopOp.getNumLoops();
  int numControlOperands = tiledLoopOp.getNumControlOperands();

  // Add buffers for outputs and the corresponding block arguments.
  // Keep separate iterators to increment without further leaking impl. details.
  // Start with outputs to avoid interference from new input buffers.
  int numNewOutputBuffers = 0;
  int resultIndex = 0;
  int oldOutputBBArgIndex = numLoops + oldInputs.size();
  int nextOutputBBArgIndex = numLoops + oldInputs.size() + oldOutputs.size();
  int nextOutputOperandIndex =
      numControlOperands + oldInputs.size() + oldOutputs.size();
  for (Value oldOutputTensor : oldOutputs) {
    if (!oldOutputTensor.getType().isa<TensorType>()) {
      // Skip and increment the old bbarg index only.
      ++oldOutputBBArgIndex;
      // Do not increment resultIndex as only tensors are returned.
      // TODO: better interface to avoid leaking such impl details.
      continue;
    }

    assert(oldOutputTensor.getType().isa<RankedTensorType>() &&
           "bufferizable output must be a ranked tensor");

    Value outputBuffer = lookup(bvm, oldOutputTensor);
    const OpResult &opResult = tiledLoopOp->getResult(resultIndex);
    OpOperand &yieldOperand = yieldOp->getOpOperand(resultIndex);
    // If the result is not inplaceable, need to allocate a copy for it.
    if (getInPlace(opResult) != InPlaceSpec::True) {
      auto loc = tiledLoopOp.getLoc();
      Value alloc = createNewAllocDeallocPairForShapedValue(
          b, loc, oldOutputTensor, aliasInfo);
      // If the tensor comes from either:
      //   - linalg.init_tensor
      //   - tensor.cast(linalg.init_tensor())
      // Then the value is unitialized and we do not need to copy. This is a
      // pragmatic simplification of "matching bbArg does not bufferize to a
      // read".
      // TODO: "matching bbArg does not bufferize to a read" is a more general
      // check.
      if (!isInitTensorOp(oldOutputTensor)) {
        OpBuilder::InsertionGuard g(b);
        // Set insertion point now that potential alloc/dealloc are introduced.
        // Copy is inserted just before the tiledLoopOp.
        b.setInsertionPoint(tiledLoopOp);
        b.create<linalg::CopyOp>(loc, outputBuffer, alloc);
      }
      outputBuffer = alloc;
    }
    // Insert mapping and aliasing info.
    aliasInfo.createAliasInfoEntry(outputBuffer);
    aliasInfo.insertNewBufferEquivalence(opResult, outputBuffer);
    map(bvm, opResult, outputBuffer);

    // Insert new operand and bbArg.
    tiledLoopOp->insertOperands(nextOutputOperandIndex, outputBuffer);
    BlockArgument newBufferBBArg =
        body->insertArgument(nextOutputBBArgIndex, outputBuffer.getType());
    BlockArgument oldTensorBBArg = body->getArgument(oldOutputBBArgIndex);
    // Insert mapping and aliasing info.
    aliasInfo.createAliasInfoEntry(newBufferBBArg);
    aliasInfo.insertNewBufferEquivalence(oldTensorBBArg, newBufferBBArg);
    map(bvm, oldTensorBBArg, newBufferBBArg);

    // Set operand of `linalg.yield` to the bbArg so it just canonicalizes away
    // later.
    yieldOperand.set(oldTensorBBArg);

    // Increment indices.
    ++numNewOutputBuffers;
    ++resultIndex;
    ++oldOutputBBArgIndex;
    ++nextOutputBBArgIndex;
    ++nextOutputOperandIndex;
  }

  // Add buffers for inputs and the corresponding block arguments.
  // Keep separate iterators to increment without further leaking impl. details.
  int numNewInputBuffers = 0;
  int oldInputBBArgIndex = numLoops;
  int nextInputBBArgIndex = numLoops + oldInputs.size();
  int nextInputOperandIndex = numControlOperands + oldInputs.size();
  for (Value oldInputTensor : oldInputs) {
    if (!oldInputTensor.getType().isa<TensorType>()) {
      // Skip and increment the old bbarg index only.
      ++oldInputBBArgIndex;
      continue;
    }

    Value inputBuffer = lookup(bvm, oldInputTensor);
    assert(inputBuffer && " missing buffer for operand");

    // Insert new operand and bbArg.
    tiledLoopOp->insertOperands(nextInputOperandIndex, inputBuffer);
    BlockArgument newBufferBBArg =
        body->insertArgument(nextInputBBArgIndex, inputBuffer.getType());
    BlockArgument oldTensorBBArg = body->getArgument(oldInputBBArgIndex);

    // Insert mapping and aliasing info.
    aliasInfo.createAliasInfoEntry(newBufferBBArg);
    aliasInfo.insertNewBufferEquivalence(oldTensorBBArg, newBufferBBArg);
    map(bvm, oldTensorBBArg, newBufferBBArg);

    // Increment indices.
    ++numNewInputBuffers;
    ++oldInputBBArgIndex;
    ++nextInputBBArgIndex;
    ++nextInputOperandIndex;
  }

  // Update segment sizes.
  // TODO: Helper method to avoid leaking impl details.
  tiledLoopOp->setAttr(
      TiledLoopOp::getOperandSegmentSizeAttr(),
      b.getI32VectorAttr(
          {numLoops, numLoops, numLoops,
           static_cast<int>(oldInputs.size()) + numNewInputBuffers,
           static_cast<int>(oldOutputs.size()) + numNewOutputBuffers}));

  return success();
}

/// Bufferize ExtractSliceOp to subview with optional alloc + copy depending on
/// whether or not it is marked inplaceable.
/// Note that `getInplaceableOpResult` on a ExtractSliceOp always returns null.
/// As consequence a ExtractSliceOp always alloc + copy when taken in
/// isolation.
static LogicalResult bufferize(OpBuilder &b, ExtractSliceOp extractSliceOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);

  LDBG("bufferize: " << *extractSliceOp << '\n');

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
  if (inPlace != InPlaceSpec::True)
    alloc = createNewAllocDeallocPairForShapedValue(
        b, loc, extractSliceOp.result(), aliasInfo);

  // Set insertion point now that potential alloc/dealloc are introduced.
  b.setInsertionPoint(extractSliceOp);

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
  // Insert new alias.
  aliasInfo.insertNewBufferAlias(subView, srcMemref);

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
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(insertSliceOp);

  LDBG("bufferize: " << *insertSliceOp << '\n');

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
    // Alloc a copy for `insertSliceOp.dest()`, it will become the result
    // buffer.
    Value newDstMemref = createNewAllocDeallocPairForShapedValue(
        b, loc, insertSliceOp.dest(), aliasInfo);
    // Set insertion point now that potential alloc/dealloc are introduced.
    b.setInsertionPoint(insertSliceOp);
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
  if (!aliasInfo.isSourceEquivalentToAMatchingInplaceExtractSliceOp(
          insertSliceOp) ||
      inPlace != InPlaceSpec::True) {
    LDBG("insert_slice needs extra source copy: " << insertSliceOp.source()
                                                  << " -> copy\n");
    // Take a subview of the dst.
    Value subView = b.create<memref::SubViewOp>(
        loc, subviewMemRefType, dstMemref, insertSliceOp.getMixedOffsets(),
        insertSliceOp.getMixedSizes(), insertSliceOp.getMixedStrides());
    // Insert new alias.
    aliasInfo.insertNewBufferAlias(subView, dstMemref);
    b.create<CopyOp>(insertSliceOp.getLoc(), srcMemref, subView);
  }

  map(bvm, insertSliceOp.result(), dstMemref);

  return success();
}

static LogicalResult bufferize(OpBuilder &b, VectorTransferOpInterface op,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);

  if (op.getShapedType().isa<MemRefType>())
    return failure();

  /// transfer_read from buffer always reads from the bufferized
  /// op.source().
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op.getOperation())) {
    Value v = lookup(bvm, op.source());
    assert(v && "missing buffer");
    readOp.sourceMutable().assign(v);
    return success();
  }

  auto inPlace = getInPlace(op->getResult(0));
  auto writeOp = cast<vector::TransferWriteOp>(op.getOperation());

  // If transfer_write is not inPlace, allocate a new buffer.
  Value newInputBuffer;
  Location loc = op.getLoc();
  if (inPlace != InPlaceSpec::True) {
    // Alloc a copy for `writeOp.source()`, it will become the result buffer.
    newInputBuffer = createNewAllocDeallocPairForShapedValue(
        b, loc, writeOp.source(), aliasInfo);
    Value v = lookup(bvm, writeOp.source());
    if (!isInitTensorOp(writeOp.source())) {
      // Set insertion point now that potential alloc/dealloc are introduced.
      b.setInsertionPoint(op);
      b.create<CopyOp>(loc, v, newInputBuffer);
    }
  } else {
    // InPlace write will result in memref.tensor_load(x) which must
    // canonicalize away with one of it uses.
    newInputBuffer = lookup(bvm, writeOp.source());
    assert(newInputBuffer && "missing buffer");
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
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  // Cannot create IR past a yieldOp.
  b.setInsertionPoint(yieldOp);

  if (auto execOp = dyn_cast<scf::ExecuteRegionOp>(yieldOp->getParentOp())) {
    if (execOp->getNumResults() != 0)
      return execOp->emitError(
          "expected result-less scf.execute_region containing op");
    return success();
  }

  scf::ForOp forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
  if (!forOp)
    return yieldOp->emitError("expected scf::ForOp parent for scf::YieldOp");
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    auto tensorType = operand.get().getType().dyn_cast<TensorType>();
    if (!tensorType)
      continue;

    OpOperand &forOperand = forOp.getOpOperandForResult(
        forOp->getResult(operand.getOperandNumber()));
    auto bbArg = forOp.getRegionIterArgForOpOperand(forOperand);
    Value yieldedBuffer = lookup(bvm, operand.get());
    Value bbArgBuffer = lookup(bvm, bbArg);
    if (!aliasInfo.areEquivalentBufferizedValues(yieldedBuffer, bbArgBuffer)) {
      // TODO: this could get resolved with copies but it can also turn into
      // swaps so we need to be careful about order of copies.
      return yieldOp->emitError()
             << "Yield operand #" << operand.getOperandNumber()
             << " does not bufferize to an equivalent buffer to the matching"
             << " enclosing scf::for operand";
    }

    // Buffers are equivalent so the work is already done and we just yield the
    // bbArg so that it later canonicalizes away.
    operand.set(bbArg);
  }
  return success();
}

/// Bufferization for linalg::YieldOp either does not involve tensors or just
/// results in later canonicalization. In either case it does nothing.
static LogicalResult bufferize(OpBuilder &b, linalg::YieldOp yieldOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  // Cannot create IR past a yieldOp.
  b.setInsertionPoint(yieldOp);

  // No tensors -> success.
  if (!llvm::any_of(yieldOp.getOperandTypes(), isaTensor))
    return success();
  // linalg::YieldOp nested under TiledLoop must just canonicalize.
  if (yieldOp->getParentOfType<TiledLoopOp>())
    return success();
  llvm_unreachable("unexpected yieldOp");
}

/// Bufferization for tensor::ExtractOp just translate to memref.load, it only
/// reads the tensor.
static LogicalResult bufferize(OpBuilder &b, tensor::ExtractOp extractOp,
                               BlockAndValueMapping &bvm,
                               BufferizationAliasInfo &aliasInfo) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(extractOp);

  Location loc = extractOp.getLoc();
  Value srcMemref = lookup(bvm, extractOp.tensor());
  Value l = b.create<memref::LoadOp>(loc, srcMemref, extractOp.indices());
  extractOp.replaceAllUsesWith(l);
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
  LDBG("Inplace analysis for extract_slice: "
       << printOperationInfo(extractSliceOp) << '\n');

  // If `extractSliceOp` were to be bufferized inplace, it cannot end up
  // aliasing a write into a non-writable buffer.
  bool wouldCreateAliasingWriteToNonWritableBuffer =
      aliasInfo.aliasesInPlaceWrite(extractSliceOp.result()) &&
      aliasInfo.aliasesNonWritableBuffer(extractSliceOp->getOpOperand(0));

  if (wouldCreateAliasingWriteToNonWritableBuffer)
    LDBG("->the corresponding buffer is not writable\n");
  else
    LDBG("->bufferizes to writable inplace buffer\n");

  // In any of extractSliceOp.result's aliases, can we find 2 such that we hit
  // an interfering write?
  OpResult r = extractSliceOp->getResult(0);
  OpOperand &s = extractSliceOp->getOpOperand(0);
  bool foundInterference =
      wouldCreateAliasingWriteToNonWritableBuffer ||
      aliasInfo.wouldCreateReadAfterWriteInterference(r, domInfo);
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
  (void)op;

  int64_t resultNumber = result.getResultNumber();
  (void)resultNumber;
  LDBG('\n');
  LDBG("Inplace analysis for <- #" << resultNumber << " -> #"
                                   << operand.getOperandNumber() << " in "
                                   << printValueInfo(result) << '\n');

  // `result` must bufferize to a writable buffer to be a candidate.
  // This means the operand  must not alias either:
  //   1. a function bbArg that is not inplaceable or
  //   2. a constant op.
  // to be considered for inplace bufferization
  bool wouldCreateAliasingWriteToNonWritableBuffer =
      aliasInfo.aliasesNonWritableBuffer(operand);
  if (wouldCreateAliasingWriteToNonWritableBuffer)
    LDBG("->the corresponding buffer is not writable\n");
  else
    LDBG("->bufferizes to writable inplace buffer\n");

  assert(result == getInplaceableOpResult(operand));
  bool foundInterference =
      wouldCreateAliasingWriteToNonWritableBuffer ||
      aliasInfo.wouldCreateReadAfterWriteInterference(result, domInfo);

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

/// Analyze the `ops` to determine which OpResults are inplaceable:
///   1. First, analyze InsertSliceOp greedily: we almost never want to
///      bufferize the tensor "inserted into" to become out-of-place.
///   2. Walk the other ops in reverse. This is a good starter heuristic.
///      ExtractSliceOps are interleaved with other ops in traversal order.
///
LogicalResult mlir::linalg::inPlaceAnalysis(SmallVector<Operation *> &ops,
                                            BufferizationAliasInfo &aliasInfo,
                                            const DominanceInfo &domInfo) {
  // Walk ops in reverse for better interference analysis.
  for (Operation *op : reverse(ops)) {
    for (OpOperand &opOperand : op->getOpOperands()) {
      if (OpResult result = getInplaceableOpResult(opOperand))
        if (result.getType().isa<TensorType>() &&
            failed(bufferizableInPlaceAnalysis(opOperand, result, aliasInfo,
                                               domInfo)))
          op->emitWarning() << "Inplace analysis treated conservatively";
    }
    // Special logic to analyze ExtractSliceOp.
    // Note that ExtractSliceOp analysis needs to be interleaved with other ops
    // to properly capture aliases.
    // Walk ExtractSliceOps in reverse for better clobbering analysis behavior:
    // it is easier to detect clobbers of smaller slices before larger ones.
    if (auto extractSliceOp = dyn_cast<ExtractSliceOp>(op))
      if (failed(
              bufferizableInPlaceAnalysis(extractSliceOp, aliasInfo, domInfo)))
        op->emitWarning() << "Inplace analysis treated conservatively";
  }
  return success();
}

/// Analyze the `funcOp` body to determine which OpResults are inplaceable.
static LogicalResult
inPlaceAnalysisFuncOpBody(FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
                          const DominanceInfo &domInfo) {
  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  LDBG("Begin InPlaceAnalysisFuncOpInternals:\n" << funcOp << '\n');
  assert(funcOp && funcOp->getNumRegions() > 0 && !funcOp.body().empty() &&
         "expected a funcOp definition with a body");

  // Collect ops so we can build our own reverse traversal.
  SmallVector<Operation *> ops;
  funcOp.walk([&](Operation *op) {
    // No tensors => no buffers.
    if (none_of(op->getOperandTypes(), isaTensor) &&
        none_of(op->getResultTypes(), isaTensor))
      return;
    ops.push_back(op);
  });

  // Set the function arguments marked with inplaceable to be known as
  // bufferizing to a writeable memory.
  for (BlockArgument bbArg : funcOp.getArguments()) {
    BoolAttr inplaceAttr = funcOp.getArgAttrOfType<BoolAttr>(
        bbArg.getArgNumber(), LinalgDialect::kInplaceableAttrName);
    if (inplaceAttr && inplaceAttr.getValue())
      aliasInfo.setBufferizesToWritableMemory(bbArg);
  }

  LogicalResult res = inPlaceAnalysis(ops, aliasInfo, domInfo);
  LDBG("End InPlaceAnalysisFuncOpInternals:\n" << funcOp << '\n');

  return res;
}

//===----------------------------------------------------------------------===//
// Bufferization entry-point for functions.
//===----------------------------------------------------------------------===//

LogicalResult mlir::linalg::bufferizeOp(
    Operation *op, BlockAndValueMapping &bvm, BufferizationAliasInfo &aliasInfo,
    DenseMap<FuncOp, FunctionType> *bufferizedFunctionTypes,
    GlobalCreator *globalCreator) {
  OpBuilder b(op->getContext());
  return TypeSwitch<Operation *, LogicalResult>(op)
      // Skip BufferCast and TensorLoad ops.
      .Case<memref::BufferCastOp, memref::TensorLoadOp>(
          [&](auto) { return success(); })
      .Case<tensor::CastOp, tensor::DimOp, ExtractSliceOp, scf::ForOp,
            InitTensorOp, InsertSliceOp, tensor::ExtractOp, LinalgOp, ReturnOp,
            TiledLoopOp, VectorTransferOpInterface, linalg::YieldOp,
            scf::YieldOp>([&](auto op) {
        LDBG("Begin bufferize:\n" << op << '\n');
        return bufferize(b, op, bvm, aliasInfo);
      })
      .Case([&](CallOpInterface op) {
        LDBG("Begin bufferize:\n" << op << '\n');
        if (!bufferizedFunctionTypes)
          llvm_unreachable(
              "null bufferizedFunctionTypes when bufferizing CallOpInterface");
        return bufferize(b, op, bvm, aliasInfo, *bufferizedFunctionTypes);
      })
      .Case([&](ConstantOp op) {
        if (!isaTensor(op.getResult().getType()))
          return success();
        LDBG("Begin bufferize:\n" << op << '\n');
        if (!globalCreator)
          llvm_unreachable("null globalCreator when bufferizing ConstantOp");
        return bufferize(b, op, bvm, aliasInfo, *globalCreator);
      })
      .Default([&](Operation *op) -> LogicalResult {
        auto isaTensor = [](Type t) { return t.isa<TensorType>(); };
        if (any_of(op->getOperandTypes(), isaTensor) ||
            any_of(op->getResultTypes(), isaTensor))
          return op->emitError() << "unsupported op with tensors";
        return success();
      });
}

static LogicalResult bufferizeFuncOpInternals(
    FuncOp funcOp, BlockAndValueMapping &bvm, BufferizationAliasInfo &aliasInfo,
    DenseMap<FuncOp, FunctionType> &bufferizedFunctionTypes,
    GlobalCreator &globalCreator) {

  LLVM_DEBUG(llvm::dbgs() << "\n\n");
  LDBG("Begin BufferizeFuncOpInternals:\n" << funcOp << '\n');
  OpBuilder b(funcOp->getContext());
  /// Start by bufferizing `funcOp` arguments.
  if (failed(bufferize(b, funcOp, bvm, aliasInfo)))
    return failure();

  // Walk in PreOrder to ensure ops with regions are handled before their body.
  // Since walk has to be PreOrder, we need to erase ops that require it
  // separately: this is the case for CallOp
  SmallVector<Operation *> toErase;
  if (funcOp
          .walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
            if (failed(bufferizeOp(op, bvm, aliasInfo, &bufferizedFunctionTypes,
                                   &globalCreator)))
              return failure();
            // Register post-walk erasure, if necessary.
            if (isa<CallOpInterface>(op))
              if (llvm::any_of(op->getOperandTypes(), isaTensor) ||
                  llvm::any_of(op->getResultTypes(), isaTensor))
                toErase.push_back(op);
            return success();
          })
          .wasInterrupted())
    return failure();

  LDBG("End BufferizeFuncOpInternals:\n" << funcOp << '\n');

  for (Operation *op : toErase)
    op->erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Bufferization entry-point for modules.
//===----------------------------------------------------------------------===//

/// Return the op with Allocate MemoryEffect if `v` is equivalent to such an
/// an op. Return null otherwise.
static Operation *getEquivalentAlloc(Value value,
                                     const BufferizationAliasInfo &aliasInfo) {
  Operation *res = nullptr;
  aliasInfo.applyOnEquivalenceClass(value, [&](Value v) {
    if (!res)
      if (auto interface =
              dyn_cast_or_null<MemoryEffectOpInterface>(v.getDefiningOp()))
        if (auto effect =
                interface.getEffectOnValue<MemoryEffects::Allocate>(v))
          res = v.getDefiningOp();
  });
  return res;
}

/// Return the first argument of the enclosing FuncOp that is equivalent to `v`.
/// Return null if no such bbArg can be found.
static BlockArgument
getEquivalentEnclosingFuncBBArg(Value v,
                                const BufferizationAliasInfo &aliasInfo) {
  if (!v.getType().isa<RankedTensorType>())
    return nullptr;
  Operation *op = v.getParentBlock()->getParentOp();
  FuncOp funcOp = dyn_cast<FuncOp>(op);
  if (!funcOp)
    funcOp = op->getParentOfType<FuncOp>();
  assert(funcOp && "expected non-null FuncOp");
  for (BlockArgument bbArg : funcOp.getArguments()) {
    if (!bbArg.getType().isa<RankedTensorType>())
      continue;
    if (aliasInfo.areEquivalentBufferizedValues(v, bbArg))
      return bbArg;
  }
  return nullptr;
}

/// Rewrite the `funcOp` arguments analysis return values and terminator into
/// buffer form (using the canonical memref layout for now), according to the
/// inPlace-bufferizable information of the function arguments.
/// This relies on a buffer equivalence analysis of each return operand. When a
/// result buffer is equivalent to:
///   1. a BlockArgument of `funcOp`, it can be dropped from the return values
///      and becomes inplaceable at all callers. This assumes all CallOp perform
///      the necessary work to clone operands so as to make them inplaceable.
//       Reliance on this logic will need to be relaxed in thefuture.
///   2. an op with an Alloc effect, this currently fails bufferization but is a
///      candidate for hoisting and creating a new inplace operand at all caller
///      sites.
///   3. if such a hoisting for 2. is not possible (e.g. data-dependent that
///      prevents hoisting), this is currently unsupported and will require a
///      refcounted buffer type.
static LogicalResult bufferizeFuncOpBoundary(
    FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
    DenseMap<FuncOp, FunctionType> &bufferizedFunctionTypes) {
  LLVM_DEBUG(DBGS() << "Begin bufferizeFuncOpBoundary:\n" << funcOp << "\n");

  // If nothing to do then we are done.
  if (!llvm::any_of(funcOp.getType().getInputs(), isaTensor) &&
      !llvm::any_of(funcOp.getType().getResults(), isaTensor))
    return success();

  // Get the bufferized FunctionType for funcOp or construct it if not yet
  // available.
  // TODO: Atm we have 3 cases:
  // 1. if a function is called from within the Module, it must have bufferized
  //    to inplaceable tensor results.
  // 2. if it is bodiless, it must have bufferized and is not allowed to have
  //    result tensors.
  // 3. if it is not called internally, it still must bufferize to inplaceable
  //    tensor results and we construct it now (e.g. top-level function called
  //    externally).
  // -> Figure out a better layering.
  TypeRange resultTypes;

  // Corner case: Bodiless FuncOp
  // ============================
  // The body of such functions is assumed opaque and we can't know the
  // bufferization contract they want to enforce atm.
  // As a consequence, only support functions that don't return any tensor atm.
  if (funcOp.getBody().empty()) {
    if (llvm::any_of(funcOp.getType().getResults(), isaTensor))
      return funcOp->emitError() << "cannot bufferize bodiless function that "
                                 << "returns a tensor";
    FunctionType bufferizedFuncType =
        getOrCreateBufferizedFunctionType(funcOp, funcOp.getType().getInputs(),
                                          TypeRange{}, bufferizedFunctionTypes);
    funcOp.setType(bufferizedFuncType);
    LLVM_DEBUG(DBGS() << "End bufferizeFuncOpBoundary no fun body: " << funcOp);
    return success();
  }

  // Support only single return-terminated block in the function.
  ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
  assert(returnOp && "expected func with single return op");

  // 1. For each FuncOp result, keep track of which inplace argument it reuses.
  SmallVector<Value> returnValues;
  for (OpOperand &returnOperand : returnOp->getOpOperands()) {
    // If not a renturn tensor type just forward it.
    if (!returnOperand.get().getType().isa<RankedTensorType>()) {
      returnValues.push_back(returnOperand.get());
      continue;
    }

    // If return operand is equivalent to some bbArg, no need to return it.
    Value returnVal = returnOperand.get();
    if (getEquivalentEnclosingFuncBBArg(returnVal, aliasInfo))
      continue;

    // TODO: Need to hoist above function boundary.
    if (Operation *allocOp = getEquivalentAlloc(returnVal, aliasInfo)) {
      returnValues.push_back(allocOp->getResult(0));
      continue;
    }

    // Other cases legitimately need to return a tensor, this is currently not
    // supported. For instance, if hoisting across function boundary has
    // failed, it may be due to e.g. data-dependent sizes. In such a case, we
    // would need a better type than memref.
    int64_t returnIdx = returnOperand.getOperandNumber();
    return returnOp->emitError()
           << "buffer result #" << returnIdx << " not produced by an alloc\n";
  }

  // 2. Rewrite the terminator without the inPlace bufferizable values.
  ValueRange retValues{returnValues};
  FunctionType bufferizedFuncType = getOrCreateBufferizedFunctionType(
      funcOp, funcOp.getType().getInputs(), retValues.getTypes(),
      bufferizedFunctionTypes);
  OpBuilder b(returnOp);
  b.create<ReturnOp>(returnOp.getLoc(), returnValues);
  returnOp->erase();

  // 3. Rewrite the bbArgs.
  // Iterate on the original `numArgs` and replace them in order.
  // This guarantees the argument order still matches after the rewrite.
  Block &frontBlock = funcOp.body().front();
  unsigned numArgs = frontBlock.getNumArguments();
  for (unsigned idx = 0; idx < numArgs; ++idx) {
    auto bbArg = frontBlock.getArgument(0);
    auto tensorType = bbArg.getType().dyn_cast<TensorType>();
    // Non-tensor types are just forwarded.
    if (!tensorType) {
      frontBlock.addArgument(bbArg.getType());
      bbArg.replaceAllUsesWith(frontBlock.getArguments().back());
      frontBlock.eraseArgument(0);
      continue;
    }

    // Get the buffer type from the bufferized function type.
    Type memrefType = bufferizedFuncType.getInput(idx);
    Value memref = frontBlock.addArgument(memrefType);
    OpBuilder b(funcOp->getContext());
    b.setInsertionPointToStart(&frontBlock);
    // Replace all uses of bbArg through a BufferCastOp by a memref::CastOp.
    for (auto &use : llvm::make_early_inc_range(bbArg.getUses())) {
      if (auto bufferCastOp = dyn_cast<memref::BufferCastOp>(use.getOwner())) {
        auto castOp = b.create<memref::CastOp>(
            funcOp.getLoc(), bufferCastOp.memref().getType(), memref);
        bufferCastOp.memref().replaceAllUsesWith(castOp);
        aliasInfo.insertNewBufferEquivalence(castOp.dest(),
                                             bufferCastOp.memref());
      }
    }
    // Replace all remaining uses by a tensor_load.
    if (!bbArg.use_empty()) {
      auto tensorLoadOp =
          b.create<memref::TensorLoadOp>(funcOp.getLoc(), memref);
      aliasInfo.insertNewBufferEquivalence(tensorLoadOp, bbArg);
      bbArg.replaceAllUsesWith(tensorLoadOp);
    }
    frontBlock.eraseArgument(0);
    // TODO: add support to erase aliasInfo entries if deemed necessary.
  }

  // 4. Rewrite the FuncOp type to buffer form.
  funcOp.setType(bufferizedFuncType);

  LLVM_DEBUG(DBGS() << "End bufferizeFuncOpBoundary:\n" << funcOp);

  return success();
}

/// Store all functions of the `moduleOp` in `orderedFuncOps`, sorted by
/// callee-caller order (i.e. callees without callers first).
/// Store the map of FuncOp to all its callers in `callerMap`.
/// Return `failure()` if a cycle of calls is detected or if we are unable to
/// retrieve the called FuncOp from any CallOpInterface.
static LogicalResult
getFuncOpsOrderedByCalls(ModuleOp moduleOp,
                         SmallVectorImpl<FuncOp> &orderedFuncOps,
                         DenseMap<FuncOp, DenseSet<Operation *>> &callerMap) {
  // For each FuncOp, the set of functions called by it (i.e. the union of
  // symbols of all nested CallOpInterfaceOp).
  DenseMap<FuncOp, DenseSet<FuncOp>> calledBy;
  // For each FuncOp, the number of CallOpInterface it contains.
  DenseMap<FuncOp, unsigned> numberCallOpsContainedInFuncOp;
  WalkResult res = moduleOp.walk([&](FuncOp funcOp) -> WalkResult {
    if (!funcOp.body().empty()) {
      ReturnOp returnOp = getAssumedUniqueReturnOp(funcOp);
      if (!returnOp)
        return funcOp->emitError()
               << "cannot bufferize a FuncOp with tensors and "
                  "without a unique ReturnOp";
    }

    numberCallOpsContainedInFuncOp[funcOp] = 0;
    return funcOp.walk([&](CallOpInterface callOp) -> WalkResult {
      // Only support CallOp for now.
      if (!isa<CallOp>(callOp.getOperation()))
        return callOp->emitError() << "expected a CallOp";
      FuncOp calledFunction = getCalledFunction(callOp);
      assert(calledFunction && "could not retrieved called FuncOp");
      auto it = callerMap.try_emplace(calledFunction, DenseSet<Operation *>{});
      it.first->getSecond().insert(callOp);
      if (calledBy[calledFunction].count(funcOp) == 0) {
        calledBy[calledFunction].insert(funcOp);
        numberCallOpsContainedInFuncOp[funcOp]++;
      }
      return WalkResult::advance();
    });
  });
  if (res.wasInterrupted())
    return failure();
  // Iteratively remove function operation that do not call any of the
  // functions remaining in the callCounter map and add them to the worklist.
  while (!numberCallOpsContainedInFuncOp.empty()) {
    auto it = llvm::find_if(numberCallOpsContainedInFuncOp,
                            [](auto entry) { return entry.getSecond() == 0; });
    if (it == numberCallOpsContainedInFuncOp.end())
      return moduleOp.emitOpError(
          "expected callgraph to be free of circular dependencies.");
    orderedFuncOps.push_back(it->getFirst());
    for (auto callee : calledBy[it->getFirst()])
      numberCallOpsContainedInFuncOp[callee]--;
    numberCallOpsContainedInFuncOp.erase(it);
  }
  return success();
}

namespace {
struct LinalgComprehensiveModuleBufferize
    : public LinalgComprehensiveModuleBufferizeBase<
          LinalgComprehensiveModuleBufferize> {

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect>();
  }
};
} // end namespace

static void applyEnablingTransformations(ModuleOp moduleOp) {
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<GeneralizePadTensorOpPattern>(moduleOp.getContext());
  (void)applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

static void
foreachCaller(const DenseMap<FuncOp, DenseSet<Operation *>> &callerMap,
              FuncOp callee, llvm::function_ref<void(Operation *)> doit) {
  auto itCallers = callerMap.find(callee);
  if (itCallers == callerMap.end())
    return;
  for (Operation *caller : itCallers->second)
    doit(caller);
}

/// Postprocess the linalg.buffer_layout annotation across function boundaries.
/// This is a purely mechanical process that may later become part of a
/// separate pass with its own layout assignment heuristic.
static void layoutPostProcessing(ModuleOp moduleOp) {
  SmallVector<FuncOp> orderedFuncOps;
  DenseMap<FuncOp, DenseSet<Operation *>> callerMap;
  auto res = getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps, callerMap);
  (void)res;
  assert(succeeded(res) && "unexpected getFuncOpsOrderedByCalls failure");

  for (FuncOp funcOp : orderedFuncOps) {
    DenseMap<Operation *, SmallVector<Value>> operandsPerCaller;
    foreachCaller(callerMap, funcOp, [&](Operation *caller) {
      operandsPerCaller.try_emplace(caller, SmallVector<Value>());
    });

    SmallVector<Type> argumentTypes;
    // Iterate on each function argument and check it it was marked with a
    // desired layout.
    for (auto it : llvm::enumerate(funcOp.getType().getInputs())) {
      int argNumber = it.index();
      Type inputType = it.value();
      auto memrefType = inputType.dyn_cast<MemRefType>();
      auto layoutAttr = funcOp.getArgAttrOfType<AffineMapAttr>(
          argNumber, LinalgDialect::kBufferLayoutAttrName);
      AffineMap desiredLayoutMap =
          layoutAttr ? layoutAttr.getValue() : AffineMap();
      AffineMap currentLayoutMap =
          memrefType ? getStridedLinearLayoutMap(memrefType) : AffineMap();
      if (!memrefType || !layoutAttr || desiredLayoutMap == currentLayoutMap) {
        argumentTypes.push_back(inputType);
        foreachCaller(callerMap, funcOp, [&](Operation *caller) {
          operandsPerCaller.find(caller)->getSecond().push_back(
              caller->getOperand(argNumber));
        });
        continue;
      }

      // Compute the buffer type with desired layout and add to input argument
      // types.
      MemRefType desiredMemrefType = MemRefType::get(
          memrefType.getShape(), memrefType.getElementType(), desiredLayoutMap);
      argumentTypes.push_back(desiredMemrefType);

      // If funcOp's body is not empty, change the bbArg type and propagate.
      if (!funcOp.body().empty()) {
        BlockArgument bbArg = funcOp.getArgument(argNumber);
        bbArg.setType(desiredMemrefType);
        OpBuilder b(bbArg.getContext());
        b.setInsertionPointToStart(bbArg.getOwner());
        // Cast back to the original memrefType and let it canonicalize.
        Value cast =
            b.create<memref::CastOp>(funcOp.getLoc(), memrefType, bbArg);
        bbArg.replaceAllUsesExcept(cast, cast.getDefiningOp());
      }

      // Cast to desired buffer type on all callers to `funcOp`.
      // TODO: on the callee side, this may even have to trigger a copy to
      // change the layout. For now let the memref::CastOp fail to verify in
      // such cases.
      auto castArg = [&](Operation *caller) {
        OpBuilder b(caller);
        Value newOperand = b.create<memref::CastOp>(
            funcOp.getLoc(), desiredMemrefType, caller->getOperand(argNumber));
        operandsPerCaller.find(caller)->getSecond().push_back(newOperand);
      };
      foreachCaller(callerMap, funcOp, castArg);
    }

    // Set operands with cast buffer on all callers to `funcOp`.
    foreachCaller(callerMap, funcOp, [&](Operation *caller) {
      caller->setOperands(operandsPerCaller.lookup(caller));
    });

    // Finally set the funcOp type to update the arguments.
    auto newFuncType = FunctionType::get(moduleOp.getContext(), argumentTypes,
                                         funcOp.getType().getResults());
    funcOp.setType(newFuncType);
  }
}

void LinalgComprehensiveModuleBufferize::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  applyEnablingTransformations(moduleOp);

  SmallVector<FuncOp> orderedFuncOps;
  DenseMap<FuncOp, DenseSet<Operation *>> callerMap;
  DenseMap<FuncOp, FunctionType> bufferizedFunctionTypes;
  if (failed(getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps, callerMap)))
    return signalPassFailure();

  GlobalCreator globalCreator(moduleOp);
  DominanceInfo domInfo(moduleOp);
  BufferizationAliasInfo aliasInfo(moduleOp);
  // Interestingly, all function args that are not visible outside of a module
  // can be fully bufferized inplace by guaranteeing the CallOp is bufferized
  // inplace. Therefore, we just bufferize funcOp as if none of its results were
  // inplaceable, detect which operands are cloned internally and decide what to
  // do at call sites.
  for (FuncOp funcOp : orderedFuncOps) {
    // No body => no analysis.
    if (funcOp.body().empty())
      continue;

    // In a first approximation:
    // =========================
    // If the function is called, we can allocate on the caller side which lets
    // us force inplace arguments at function boundaries.
    // TODO: do not rely on this behavior.
    if (callerMap.find(funcOp) != callerMap.end())
      for (BlockArgument bbArg : funcOp.getArguments())
        if (bbArg.getType().isa<TensorType>())
          setInPlaceFuncArgument(bbArg);

    // If the analysis fails, just return.
    if (failed(inPlaceAnalysisFuncOpBody(funcOp, aliasInfo, domInfo))) {
      signalPassFailure();
      return;
    }

    // Bufferization phase.
    if (!testAnalysisOnly) {
      BlockAndValueMapping tensorToBufferMap;
      if (failed(bufferizeFuncOpInternals(funcOp, tensorToBufferMap, aliasInfo,
                                          bufferizedFunctionTypes,
                                          globalCreator))) {
        signalPassFailure();
        return;
      }
    }
  }
  // Don't drop the attributes if we only want to report the analysis.
  if (testAnalysisOnly)
    return;

  for (FuncOp funcOp : orderedFuncOps) {
    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.
    if (failed(bufferizeFuncOpBoundary(funcOp, aliasInfo,
                                       bufferizedFunctionTypes))) {
      signalPassFailure();
      return;
    }
    if (!allowReturnMemref &&
        llvm::any_of(funcOp.getType().getResults(), [](Type t) {
          return t.isa<MemRefType, UnrankedMemRefType>();
        })) {
      funcOp->emitError("memref return type is unsupported");
      signalPassFailure();
      return;
    }
  }

  // Perform a post-processing pass of layout modification at function boundary
  // according to the kBufferLayoutAttrName.
  layoutPostProcessing(moduleOp);

  // Post-pass cleanup of inplaceable and buffer_layout attributes.
  moduleOp.walk(
      [&](Operation *op) { op->removeAttr(kInPlaceResultsAttrName); });
  moduleOp.walk([&](FuncOp op) {
    for (BlockArgument bbArg : op.getArguments())
      removeBufferizationFuncArguments(bbArg);
  });

  OpPassManager cleanupPipeline("builtin.module");
  cleanupPipeline.addPass(createCanonicalizerPass());
  cleanupPipeline.addPass(createCSEPass());
  cleanupPipeline.addPass(createLoopInvariantCodeMotionPass());
  (void)runPipeline(cleanupPipeline, moduleOp);
}

std::unique_ptr<Pass> mlir::createLinalgComprehensiveModuleBufferizePass() {
  return std::make_unique<LinalgComprehensiveModuleBufferize>();
}
