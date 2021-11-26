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
//            %0 = bufferization.to_memref %A : memref<?xf32, #map>
//            // ... uses of %0
//            %res = bufferization.to_tensor %0 : memref<?xf32, #map>
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
//            %0 = bufferization.to_memref %A : memref<?xf32, #map>
//            %1 = memref.dim %0, %c0 : memref<?xf32, #map>
//            %2 = memref.alloc(%1) : memref<?xf32>
//            %3 = memref.cast %2 : memref<?xf32> to memref<?xf32, #map>
//            // ... uses of %3
//            memref.dealloc %2 : memref<?xf32, #map>
//            %res = bufferization.to_tensor %3 : memref<?xf32, #map>
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
//            %0 = bufferization.to_memref %arg0 : memref<?xf32, #map>
//            %1 = memref.subview %0[0] [4] [1] : memref<?xf32, #map> to
//                                                memref<4xf32, #map>
//            // ... inplace computes into %1
//            %3 = bufferization.to_tensor %1 : memref<4xf32, #map>
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

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"

#include <random>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "comprehensive-module-bufferize"

using namespace mlir;
using namespace linalg;
using namespace tensor;
using namespace comprehensive_bufferize;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X)

// Forward declarations.
#ifndef NDEBUG
static std::string printOperationInfo(Operation *, bool prefix = true);
static std::string printValueInfo(Value, bool prefix = true);
#endif

static bool isaTensor(Type t) { return t.isa<TensorType>(); }

//===----------------------------------------------------------------------===//
// Bufferization-specific attribute manipulation.
// These are for testing and debugging only. Bufferization information is
// stored in BufferizationAliasInfo. When run with `testAnalysisOnly`, the IR
// is annotated with the results of the analysis (copied from
// BufferizationAliasInfo), so that they can be checked in tests.
//===----------------------------------------------------------------------===//

/// Attribute marker to specify op results that can be bufferized inPlace.
constexpr StringLiteral kInPlaceResultsAttrName = "__inplace_results_attr__";

/// Mark whether OpResult can actually be bufferized inplace.
/// If `inPlace` is `true`, the use-def chain analysis has guaranteed that no
/// subsequent write would occur to the bufferized tensor value (i.e. the result
/// can be bufferized inplace).
static void setInPlaceOpResult(OpResult opResult, bool inPlace) {
  if (!opResult)
    return;

  Operation *op = opResult.getOwner();
  auto attr =
      op->getAttr(kInPlaceResultsAttrName).dyn_cast_or_null<ArrayAttr>();
  SmallVector<StringRef> inPlaceVector =
      attr ? SmallVector<StringRef>(
                 llvm::to_vector<4>(attr.getAsValueRange<StringAttr>()))
           : SmallVector<StringRef>(op->getNumResults(), "false");
  LDBG("->set inPlace=" << inPlace << " <- #" << opResult.getResultNumber()
                        << ": " << printOperationInfo(op) << "\n");
  inPlaceVector[opResult.getResultNumber()] = inPlace ? "true" : "false";
  op->setAttr(kInPlaceResultsAttrName,
              OpBuilder(op).getStrArrayAttr(inPlaceVector));
}

/// Set the attribute that triggers inplace bufferization on a FuncOp argument
/// `bbArg`.
static void setInPlaceFuncArgument(BlockArgument bbArg, bool inPlace) {
  auto funcOp = cast<FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.setArgAttr(bbArg.getArgNumber(),
                    BufferizableOpInterface::kInplaceableAttrName,
                    BoolAttr::get(bbArg.getContext(), inPlace));
}

//===----------------------------------------------------------------------===//
// Printing helpers.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
/// Helper method printing the bufferization information of a buffer / tensor.
static void printTensorOrBufferInfo(std::string prefix, Value value,
                                    AsmState &state, llvm::raw_ostream &os) {
  if (!value.getType().isa<ShapedType>())
    return;
  os << prefix;
  value.printAsOperand(os, state);
  os << " : " << value.getType();
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
#endif

//===----------------------------------------------------------------------===//
// Bufferization-specific alias analysis.
//===----------------------------------------------------------------------===//

/// Return true if opOperand has been decided to bufferize in-place.
static bool isInplaceMemoryWrite(OpOperand &opOperand,
                                 const BufferizationAliasInfo &aliasInfo) {
  // Ops that do not bufferize to a memory write, cannot be write in-place.
  if (!bufferizesToMemoryWrite(opOperand))
    return false;
  OpResult opResult = getAliasingOpResult(opOperand);
  return opResult && aliasInfo.isInPlace(opResult);
}

/// Return true if, under current bufferization decisions, the buffer of `value`
/// is not writable.
static bool aliasesNonWritableBuffer(Value value,
                                     const BufferizationAliasInfo &aliasInfo) {
  LDBG("WRITABILITY ANALYSIS FOR " << printValueInfo(value) << "\n");
  bool foundNonWritableBuffer = false;
  aliasInfo.applyOnAliases(value, [&](Value v) {
    // Some values are known to be writable.
    if (aliasInfo.bufferizesToWritableMemory(v))
      return;

    // Query BufferizableOpInterface to see if the OpResult is writable.
    // TODO: Out-of-place bufferized OpResult could be considered writable.
    if (auto bufferizableOp = v.getDefiningOp<BufferizableOpInterface>())
      if (bufferizableOp && bufferizableOp.isWritable(v))
        return;

    // Query BufferizableOpInterface to see if the BlockArgument is writable.
    if (auto bbArg = v.dyn_cast<BlockArgument>())
      if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(
              bbArg.getOwner()->getParentOp()))
        if (bufferizableOp.isWritable(bbArg))
          return;

    foundNonWritableBuffer = true;
  });

  if (foundNonWritableBuffer)
    LDBG("--> NON WRITABLE\n");
  else
    LDBG("--> WRITABLE\n");

  return foundNonWritableBuffer;
}

/// Return true if the buffer to which `operand` would bufferize is equivalent
/// to some buffer write.
static bool aliasesInPlaceWrite(Value value,
                                const BufferizationAliasInfo &aliasInfo) {
  LDBG("----Start aliasesInPlaceWrite\n");
  LDBG("-------for : " << printValueInfo(value) << '\n');
  bool foundInplaceWrite = false;
  aliasInfo.applyOnAliases(value, [&](Value v) {
    for (auto &use : v.getUses()) {
      if (isInplaceMemoryWrite(use, aliasInfo)) {
        LDBG("-----------wants to bufferize to inPlace write: "
             << printOperationInfo(use.getOwner()) << '\n');
        foundInplaceWrite = true;
        return;
      }
    }
  });

  if (!foundInplaceWrite)
    LDBG("----------->does not alias an inplace write\n");

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

/// Given sets of uses and writes, return true if there is a RaW conflict under
/// the assumption that all given reads/writes alias the same buffer and that
/// all given writes bufferize inplace.
///
/// A conflict is: According to SSA use-def chains, a read R is supposed to read
/// the result of a write W1. But because of bufferization decisions, R actually
/// reads another write W2.
static bool
hasReadAfterWriteInterference(const DenseSet<OpOperand *> &usesRead,
                              const DenseSet<OpOperand *> &usesWrite,
                              const DominanceInfo &domInfo,
                              const BufferizationAliasInfo &aliasInfo) {
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
    Value lastWrite = findLastPrecedingWrite(uRead->get());

    // Look for conflicting memory writes. Potential conflicts are writes to an
    // alias that have been decided to bufferize inplace.
    for (OpOperand *uConflictingWrite : usesWrite) {
      // Throughout this loop, check for multiple requirements that have to be
      // met for uConflictingWrite to be an actual conflict.
      Operation *conflictingWritingOp = uConflictingWrite->getOwner();

      // Print some debug info.
      LDBG("Found potential conflict:\n");
      LDBG("READ = #" << uRead->getOperandNumber() << " of "
                      << printOperationInfo(readingOp) << "\n");
      LDBG("CONFLICTING WRITE = #"
           << uConflictingWrite->getOperandNumber() << " of "
           << printOperationInfo(conflictingWritingOp) << "\n");

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
      if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(readingOp))
        if (bufferizableOp.isNotConflicting(uRead, uConflictingWrite,
                                            aliasInfo))
          continue;

      if (conflictingWritingOp != readingOp)
        if (auto bufferizableOp =
                dyn_cast<BufferizableOpInterface>(conflictingWritingOp))
          if (bufferizableOp.isNotConflicting(uRead, uConflictingWrite,
                                              aliasInfo))
            continue;

      // Ops are not conflicting if they are in mutually exclusive regions.
      if (insideMutuallyExclusiveRegions(readingOp, conflictingWritingOp))
        continue;

      LDBG("WRITE = #" << printValueInfo(lastWrite) << "\n");

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
      if (getAliasingOpResult(*uConflictingWrite) == lastWrite)
        continue;

      // All requirements are met. Conflict found!
      LDBG("CONFLICT CONFIRMED!\n\n");
      return true;
    }
  }

  LDBG("NOT A CONFLICT!\n\n");
  return false;
}

/// Return true if bufferizing result inplace would create a conflict. A read R
/// and a write W of the same alias set is a conflict if inplace bufferization
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
bool wouldCreateReadAfterWriteInterference(
    OpOperand &operand, OpResult result, const DominanceInfo &domInfo,
    const BufferizationAliasInfo &aliasInfo,
    bool checkConsistencyOnly = false) {
#ifndef NDEBUG
  SmallVector<OpOperand *> opOperands = getAliasingOpOperand(result);
  assert(llvm::find(opOperands, &operand) != opOperands.end() &&
         "operand and result do not match");
#endif // NDEBUG

  // Helper function to iterate on aliases of `root` and capture the reads.
  auto getAliasingReads = [&](DenseSet<OpOperand *> &res, Value root) {
    aliasInfo.applyOnAliases(root, [&](Value alias) {
      for (auto &use : alias.getUses())
        // Read to a value that aliases root.
        if (bufferizesToMemoryRead(use))
          res.insert(&use);
    });
  };

  // Helper function to iterate on aliases of `root` and capture the writes.
  auto getAliasingInplaceWrites = [&](DenseSet<OpOperand *> &res, Value root) {
    aliasInfo.applyOnAliases(root, [&](Value alias) {
      for (auto &use : alias.getUses())
        // Inplace write to a value that aliases root.
        if (isInplaceMemoryWrite(use, aliasInfo))
          res.insert(&use);
    });
  };

  // Collect reads and writes of all aliases of OpOperand and OpResult.
  DenseSet<OpOperand *> usesRead, usesWrite;
  getAliasingReads(usesRead, operand.get());
  getAliasingReads(usesRead, result);
  getAliasingInplaceWrites(usesWrite, operand.get());
  getAliasingInplaceWrites(usesWrite, result);
  if (!checkConsistencyOnly && bufferizesToMemoryWrite(operand))
    usesWrite.insert(&operand);

  return hasReadAfterWriteInterference(usesRead, usesWrite, domInfo, aliasInfo);
}

/// Return true if bufferizing `opOperand` inplace with `opResult` would create
/// a write to a non-writable buffer.
static bool
wouldCreateWriteToNonWritableBuffer(OpOperand &opOperand, OpResult opResult,
                                    const BufferizationAliasInfo &aliasInfo) {
#ifndef NDEBUG
  SmallVector<OpOperand *> opOperands = getAliasingOpOperand(opResult);
  assert(llvm::find(opOperands, &opOperand) != opOperands.end() &&
         "operand and result do not match");
#endif // NDEBUG

  // Certain buffers are not writeable:
  //   1. A function bbArg that is not inplaceable or
  //   2. A constant op.
  assert(!aliasesNonWritableBuffer(opResult, aliasInfo) &&
         "expected that opResult does not alias non-writable buffer");
  bool nonWritable = aliasesNonWritableBuffer(opOperand.get(), aliasInfo);
  if (!nonWritable)
    return false;

  // This is a problem only if the buffer is written to via some alias.
  bool hasWrite = aliasesInPlaceWrite(opResult, aliasInfo) ||
                  aliasesInPlaceWrite(opOperand.get(), aliasInfo) ||
                  bufferizesToMemoryWrite(opOperand);
  if (!hasWrite)
    return false;

  LDBG("->the corresponding buffer is not writeable\n");
  return true;
}

//===----------------------------------------------------------------------===//
// Bufferization as simple BlockAndValueMapping rewrites.
//===----------------------------------------------------------------------===//

/// FuncOp always creates TensorToMemRef ops.
static LogicalResult bufferizeFuncOp(FuncOp funcOp, BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder b(funcOp->getContext());
  b.setInsertionPointToStart(&funcOp.body().front());

  // Create BufferCastOps for function args.
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
        b.create<bufferization::ToMemrefOp>(funcOp.getLoc(), memRefType, bbArg);
    state.aliasInfo.insertNewBufferEquivalence(bufferCast, bbArg);
    state.mapBuffer(bbArg, bufferCast);
  }

  // Bufferize function body.
  return bufferize(&funcOp.body(), state);
}

//===----------------------------------------------------------------------===//
// Bufferization analyses.
//===----------------------------------------------------------------------===//

/// Determine if `operand` can be bufferized in-place with `result`.
static LogicalResult
bufferizableInPlaceAnalysisImpl(OpOperand &operand, OpResult result,
                                BufferizationAliasInfo &aliasInfo,
                                const DominanceInfo &domInfo) {
#ifndef NDEBUG
  SmallVector<OpOperand *> opOperands = getAliasingOpOperand(result);
  assert(llvm::find(opOperands, &operand) != opOperands.end() &&
         "operand and result do not match");
#endif // NDEBUG

  int64_t resultNumber = result.getResultNumber();
  (void)resultNumber;
  LDBG('\n');
  LDBG("Inplace analysis for <- #" << resultNumber << " -> #"
                                   << operand.getOperandNumber() << " in "
                                   << printValueInfo(result) << '\n');

  bool foundInterference =
      wouldCreateWriteToNonWritableBuffer(operand, result, aliasInfo) ||
      wouldCreateReadAfterWriteInterference(operand, result, domInfo,
                                            aliasInfo);

  if (foundInterference)
    aliasInfo.bufferizeOutOfPlace(result);
  else
    aliasInfo.bufferizeInPlace(result, operand);

  LDBG("Done inplace analysis for result #" << resultNumber << '\n');

  return success();
}

/// Analyze the `ops` to determine which OpResults are inplaceable. Walk ops in
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
        if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op))
          if (OpResult opResult = bufferizableOp.getAliasingOpResult(opOperand))
            if (failed(bufferizableInPlaceAnalysisImpl(opOperand, opResult,
                                                       aliasInfo, domInfo)))
              return failure();

  return success();
}

/// Analyze the `funcOp` body to determine which OpResults are inplaceable.
static LogicalResult
inPlaceAnalysisFuncOpBody(FuncOp funcOp, BufferizationAliasInfo &aliasInfo,
                          const DominanceInfo &domInfo,
                          unsigned analysisFuzzerSeed = 0) {
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
        bbArg.getArgNumber(), BufferizableOpInterface::kInplaceableAttrName);
    if (inplaceAttr && inplaceAttr.getValue())
      aliasInfo.setBufferizesToWritableMemory(bbArg);
  }

  LogicalResult res =
      inPlaceAnalysis(ops, aliasInfo, domInfo, analysisFuzzerSeed);
  LDBG("End InPlaceAnalysisFuncOpInternals:\n" << funcOp << '\n');

  return res;
}

#ifndef NDEBUG
/// Assert that the current bufferization decisions are consistent.
static void checkAliasInfoConsistency(FuncOp funcOp,
                                      const DominanceInfo &domInfo,
                                      const BufferizationAliasInfo &aliasInfo) {
  funcOp.walk([&](Operation *op) {
    if (auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op))
      for (OpOperand &opOperand : op->getOpOperands())
        if (opOperand.get().getType().isa<TensorType>())
          if (OpResult opResult = bufferizableOp.getAliasingOpResult(opOperand))
            // If this assertion fails, there is probably an inconsistent
            // combination of "mustBufferizeInPlace" decisions.
            assert(!wouldCreateReadAfterWriteInterference(
                       opOperand, opResult, domInfo, aliasInfo,
                       /*checkConsistencyOnly=*/true) &&
                   "found read after write conflict before running analysis");
  });
}
#endif

/// Annotate the IR with the result of the analysis. For testing/debugging only.
static void
annotateOpsWithBufferizationMarkers(Operation *op,
                                    const BufferizationAliasInfo &aliasInfo) {
  op->walk([&](Operation *op) {
    for (OpResult opResult : op->getResults()) {
      if (opResult.getType().isa<TensorType>())
        setInPlaceOpResult(opResult, aliasInfo.isInPlace(opResult));
      if (auto funcOp = dyn_cast<FuncOp>(op))
        for (BlockArgument bbArg : funcOp.getArguments())
          if (bbArg.getType().isa<TensorType>())
            setInPlaceFuncArgument(bbArg,
                                   aliasInfo.bufferizesToWritableMemory(bbArg));
    }
  });
}

LogicalResult mlir::linalg::comprehensive_bufferize::runComprehensiveBufferize(
    FuncOp funcOp, const BufferizationOptions &options,
    BufferizationState &state) {

  DominanceInfo domInfo(funcOp);
  BufferizationAliasInfo &aliasInfo = state.aliasInfo;

  if (funcOp.body().empty())
    return success();

#ifndef NDEBUG
  checkAliasInfoConsistency(funcOp, domInfo, aliasInfo);
#endif // NDEBUG

  // If the analysis fails, just return.
  if (failed(inPlaceAnalysisFuncOpBody(funcOp, aliasInfo, domInfo,
                                       options.analysisFuzzerSeed)))
    return failure();

  for (const std::unique_ptr<PostAnalysisStep> &step :
       options.postAnalysisSteps) {
    SmallVector<Operation *> newOps;
    if (failed(step->run(funcOp, aliasInfo, domInfo, newOps)))
      return failure();
    // Analyze ops that were created by the PostAnalysisStep.
    if (failed(inPlaceAnalysis(newOps, aliasInfo, domInfo)))
      return failure();
  }

  // Annotate operations if we only want to report the analysis.
  if (options.testAnalysisOnly) {
    annotateOpsWithBufferizationMarkers(funcOp, aliasInfo);
    return success();
  }

  // Bufferize all ops in funcOp.
  if (failed(bufferizeFuncOp(funcOp, state)))
    return failure();

  // Erase all obsolete ops.
  state.eraseObsoleteOps();

  return success();
}

/// Default allocation function that is used by the comprehensive bufferization
/// pass. The default currently creates a ranked memref using `memref.alloc`.
static Optional<Value> defaultAllocationFn(OpBuilder &b, Location loc,
                                           MemRefType type,
                                           ArrayRef<Value> dynShape) {
  Value allocated = b.create<memref::AllocOp>(
      loc, type, dynShape, b.getI64IntegerAttr(kBufferAlignments));
  return allocated;
}

/// Default deallocation function that is used by the comprehensive
/// bufferization pass. It expects to recieve back the value called from the
/// `defaultAllocationFn`.
static void defaultDeallocationFn(OpBuilder &b, Location loc,
                                  Value allocatedBuffer) {
  b.create<memref::DeallocOp>(loc, allocatedBuffer);
}

/// Default memory copy function that is used by the comprehensive bufferization
/// pass. Creates a `memref.copy` op.
static void defaultMemCpyFn(OpBuilder &b, Location loc, Value from, Value to) {
  b.create<memref::CopyOp>(loc, from, to);
}

std::unique_ptr<AllocationCallbacks>
mlir::linalg::comprehensive_bufferize::defaultAllocationCallbacks() {
  return std::make_unique<AllocationCallbacks>(
      defaultAllocationFn, defaultDeallocationFn, defaultMemCpyFn);
}

// Default constructor for BufferizationOptions that sets all allocation
// callbacks to their default functions.
BufferizationOptions::BufferizationOptions()
    : allocationFns(defaultAllocationCallbacks()) {}

