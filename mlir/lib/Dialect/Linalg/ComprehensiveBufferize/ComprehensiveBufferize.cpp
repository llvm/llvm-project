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

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"

#include <random>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/BufferUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
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

/// Remove the attribute that triggers inplace bufferization on a FuncOp
/// argument `bbArg`.
static void removeBufferizationFuncArguments(BlockArgument bbArg) {
  auto funcOp = cast<FuncOp>(bbArg.getOwner()->getParentOp());
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       BufferizableOpInterface::kBufferLayoutAttrName);
  funcOp.removeArgAttr(bbArg.getArgNumber(),
                       BufferizableOpInterface::kInplaceableAttrName);
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

      // Special rules for branches.
      // TODO: Use an interface.
      if (scf::insideMutuallyExclusiveBranches(readingOp, conflictingWritingOp))
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
        b.create<memref::BufferCastOp>(funcOp.getLoc(), memRefType, bbArg);
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
          argNumber, BufferizableOpInterface::kBufferLayoutAttrName);
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
    ModuleOp moduleOp, const BufferizationOptions &options) {
  SmallVector<FuncOp> orderedFuncOps;
  DenseMap<FuncOp, DenseSet<Operation *>> callerMap;
  if (failed(getFuncOpsOrderedByCalls(moduleOp, orderedFuncOps, callerMap)))
    return failure();

  DominanceInfo domInfo(moduleOp);
  BufferizationState state(moduleOp, *options.allocationFns);
  BufferizationAliasInfo &aliasInfo = state.aliasInfo;

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
          aliasInfo.setBufferizesToWritableMemory(bbArg);

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

    // Bufferization phase.
    if (!options.testAnalysisOnly) {
      // Bufferize all ops in funcOp.
      if (failed(bufferizeFuncOp(funcOp, state)))
        return failure();

      // Erase all obsolete ops.
      state.eraseObsoleteOps();
    }
  }
  // Annotate operations if we only want to report the analysis.
  if (options.testAnalysisOnly) {
    annotateOpsWithBufferizationMarkers(moduleOp, aliasInfo);
    return success();
  }

  for (FuncOp funcOp : orderedFuncOps) {
    // Note: It would be good to apply cleanups here but we cannot as aliasInfo
    // would be invalidated.
    if (failed(bufferizeFuncOpBoundary(funcOp, aliasInfo,
                                       state.bufferizedFunctionTypes)))
      return failure();

    if (!options.allowReturnMemref &&
        llvm::any_of(funcOp.getType().getResults(), [](Type t) {
          return t.isa<MemRefType, UnrankedMemRefType>();
        })) {
      funcOp->emitError("memref return type is unsupported");
      return failure();
    }
  }

  // Perform a post-processing pass of layout modification at function boundary
  // according to the kBufferLayoutAttrName.
  layoutPostProcessing(moduleOp);

  // Post-pass cleanup of inplaceable and buffer_layout attributes.
  moduleOp.walk([&](FuncOp op) {
    for (BlockArgument bbArg : op.getArguments())
      removeBufferizationFuncArguments(bbArg);
  });

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

//===----------------------------------------------------------------------===//
// BufferizableOpInterface Implementations
//===----------------------------------------------------------------------===//

// TODO: Move these to a different file and BUILD target, so that they are
// decoupled from ComprehensiveBufferize.

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {
namespace arith_ext {

struct ConstantOpInterface
    : public BufferizableOpInterface::ExternalModel<ConstantOpInterface,
                                                    arith::ConstantOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto constantOp = cast<arith::ConstantOp>(op);
    if (!isaTensor(constantOp.getResult().getType()))
      return success();
    assert(constantOp.getType().dyn_cast<RankedTensorType>() &&
           "not a constant ranked tensor");
    auto moduleOp = constantOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return constantOp.emitError(
          "cannot bufferize constants not within builtin.module op");
    }
    GlobalCreator globalCreator(moduleOp);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(constantOp);

    auto globalMemref = globalCreator.getGlobalFor(constantOp);
    Value memref = b.create<memref::GetGlobalOp>(
        constantOp.getLoc(), globalMemref.type(), globalMemref.getName());
    state.aliasInfo.insertNewBufferEquivalence(memref, constantOp.getResult());
    state.mapBuffer(constantOp, memref);

    return success();
  }

  bool isWritable(Operation *op, Value value) const {
    // Memory locations returned by memref::GetGlobalOp may not be written to.
    assert(value.isa<OpResult>());
    return false;
  }
};

} // namespace arith_ext

namespace scf_ext {

struct ExecuteRegionOpInterface
    : public BufferizableOpInterface::ExternalModel<ExecuteRegionOpInterface,
                                                    scf::ExecuteRegionOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    // ExecuteRegionOps do not have tensor OpOperands. The yielded value can be
    // any SSA value that is in scope. To allow for use-def chain traversal
    // through ExecuteRegionOps in the analysis, the corresponding yield value
    // is considered to be aliasing with the result.
    auto executeRegionOp = cast<scf::ExecuteRegionOp>(op);
    size_t resultNum = std::distance(op->getOpResults().begin(),
                                     llvm::find(op->getOpResults(), opResult));
    assert(executeRegionOp.region().getBlocks().size() == 1 &&
           "expected exactly 1 block");
    auto yieldOp = dyn_cast<scf::YieldOp>(
        executeRegionOp.region().front().getTerminator());
    assert(yieldOp && "expected scf.yield terminator in scf.execute_region");
    return {&yieldOp->getOpOperand(resultNum)};
  }

  bool mustBufferizeInPlace(Operation *op, OpResult opResult) const {
    // ExecuteRegionOp results always bufferize in-place. Since they have no
    // OpOperands, they are mostly ignored by the analysis once alias sets are
    // set up.
    return true;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    // TODO: Add bufferization support when needed. scf.execute_region should be
    // bufferized similar to scf.if.
    auto executeRegionOp = cast<scf::ExecuteRegionOp>(op);
    bool hasTensorReturnType = any_of(
        op->getResultTypes(), [](Type t) { return t.isa<TensorType>(); });
    if (hasTensorReturnType)
      return op->emitError(
          "scf.execute_region with tensor result not supported");
    return comprehensive_bufferize::bufferize(&executeRegionOp.region(), state);
  }
};

struct IfOpInterface
    : public BufferizableOpInterface::ExternalModel<IfOpInterface, scf::IfOp> {
  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    // IfOps do not have tensor OpOperands. The yielded value can be any SSA
    // value that is in scope. To allow for use-def chain traversal through
    // IfOps in the analysis, both corresponding yield values from the then/else
    // branches are considered to be aliasing with the result.
    auto ifOp = cast<scf::IfOp>(op);
    size_t resultNum = std::distance(op->getOpResults().begin(),
                                     llvm::find(op->getOpResults(), opResult));
    return {&ifOp.thenYield()->getOpOperand(resultNum),
            &ifOp.elseYield()->getOpOperand(resultNum)};
  }

  // TODO: For better bufferization results, this could return `true` only if
  // there is a memory write in one (or both) of the branches. Since this is not
  // allowed at the moment, we should never encounter scf.ifs that yield
  // unmodified tensors. Such scf.yield ops could just fold away.
  bool isMemoryWrite(Operation *op, OpResult opResult) const {
    // IfOp results are always considered memory writes in the analysis. This
    // design decision simplifies the analysis considerably. E.g., consider the
    // following test case:
    //
    // %0 = "some_writing_op" : tensor<?xf32>
    // %r = scf.if %c -> (tensor<?xf32>) {
    //   scf.yield %0
    // } else {
    //   %1 = "another_writing_op"(%0) : tensor<?xf32>
    // }
    // "some_reading_op"(%r)
    //
    // "another_writing_op" in the above example should be able to bufferize
    // inplace in the absence of another read of %0. However, if the scf.if op
    // would not be considered a "write", the analysis would detect the
    // following conflict:
    //
    // * read = some_reading_op
    // * lastWrite = %0  (Note: The last write of %r would be a set: {%0, %1}.)
    // * conflictingWrite = %1
    //
    // For more details, check the "scf.IfOp" section of the design document.
    return true;
  }

  bool mustBufferizeInPlace(Operation *op, OpResult opResult) const {
    // IfOp results always bufferize in-place. Since they have no OpOperands,
    // they are mostly ignored by the analysis once alias sets are set up.
    return true;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto ifOp = cast<scf::IfOp>(op);

    // Bufferize then/else blocks.
    if (failed(comprehensive_bufferize::bufferize(ifOp.thenBlock(), state)))
      return failure();
    if (failed(comprehensive_bufferize::bufferize(ifOp.elseBlock(), state)))
      return failure();

    for (OpResult opResult : ifOp->getResults()) {
      if (!opResult.getType().isa<TensorType>())
        continue;
      // TODO: Atm we bail on unranked TensorType because we don't know how to
      // alloc an UnrankedMemRefType + its underlying ranked MemRefType.
      assert(opResult.getType().isa<RankedTensorType>() &&
             "unsupported unranked tensor");

      Value resultBuffer = getResultBuffer(b, opResult, state);
      if (!resultBuffer)
        return failure();

      state.aliasInfo.createAliasInfoEntry(resultBuffer);
      state.mapBuffer(opResult, resultBuffer);
    }

    return success();
  }
};

struct ForOpInterface
    : public BufferizableOpInterface::ExternalModel<ForOpInterface,
                                                    scf::ForOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    // scf::ForOp alone doesn't bufferize to a memory read, one of the uses of
    // its matching bbArg may.
    auto forOp = cast<scf::ForOp>(op);
    return isValueRead(forOp.getRegionIterArgForOpOperand(opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    // Tensor iter_args of scf::ForOps are always considered as a write. This is
    // to simplify the analysis.
    // TODO: Consider doing sth. like isValueWritten.
    return true;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    auto forOp = cast<scf::ForOp>(op);
    return {&forOp.getIterOpOperands()[opResult.getResultNumber()]};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    auto forOp = cast<scf::ForOp>(op);
    if (!opOperand.get().getType().isa<RankedTensorType>())
      return OpResult();
    return forOp.getResultForOpOperand(opOperand);
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  bool isWritable(Operation *op, Value value) const {
    // Interestingly, scf::ForOp's bbArg can **always** be viewed
    // inplace from the perspective of ops nested under:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto forOp = cast<scf::ForOp>(op);

    // Take a guard before anything else.
    OpBuilder::InsertionGuard g(b);

    for (OpResult opResult : forOp->getResults()) {
      if (!opResult.getType().isa<TensorType>())
        continue;
      // TODO: Atm we bail on unranked TensorType because we don't know how to
      // alloc an UnrankedMemRefType + its underlying ranked MemRefType.
      assert(opResult.getType().isa<RankedTensorType>() &&
             "unsupported unranked tensor");

      // TODO: More general: Matching bbArg does not bufferize to a read.
      Value resultBuffer = getResultBuffer(b, opResult, state);
      if (!resultBuffer)
        return failure();

      OpOperand &opOperand = forOp.getOpOperandForResult(opResult);
      BlockArgument bbArg = forOp.getRegionIterArgForOpOperand(opOperand);
      state.aliasInfo.createAliasInfoEntry(resultBuffer);
      state.aliasInfo.insertNewBufferEquivalence(bbArg, resultBuffer);
      state.mapBuffer(bbArg, resultBuffer);
      state.mapBuffer(opResult, resultBuffer);
    }

    // Bufferize loop body.
    if (failed(comprehensive_bufferize::bufferize(&forOp.region(), state)))
      return failure();

    // Finish bufferizing scf::ForOp.
    auto yieldOp = cast<scf::YieldOp>(&forOp.region().front().back());
    for (OpOperand &operand : yieldOp->getOpOperands()) {
      auto tensorType = operand.get().getType().dyn_cast<TensorType>();
      if (!tensorType)
        continue;

      OpOperand &forOperand = forOp.getOpOperandForResult(
          forOp->getResult(operand.getOperandNumber()));
      auto bbArg = forOp.getRegionIterArgForOpOperand(forOperand);
      Value yieldedBuffer = state.lookupBuffer(operand.get());
      Value bbArgBuffer = state.lookupBuffer(bbArg);
      if (!state.aliasInfo.areEquivalentBufferizedValues(yieldedBuffer,
                                                         bbArgBuffer)) {
        // TODO: this could get resolved with copies but it can also turn into
        // swaps so we need to be careful about order of copies.
        return yieldOp->emitError()
               << "Yield operand #" << operand.getOperandNumber()
               << " does not bufferize to an equivalent buffer to the matching"
               << " enclosing scf::for operand";
      }

      // Buffers are equivalent so the work is already done and we just yield
      // the bbArg so that it later canonicalizes away.
      operand.set(bbArg);
    }
    return success();
  }
};

struct YieldOpInterface
    : public BufferizableOpInterface::ExternalModel<YieldOpInterface,
                                                    scf::YieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto yieldOp = cast<scf::YieldOp>(op);
    if (!isa<scf::ExecuteRegionOp, scf::IfOp, scf::ForOp>(
            yieldOp->getParentOp()))
      return yieldOp->emitError("unsupported scf::YieldOp parent");
    return success();
  }
};

} // namespace scf_ext

namespace std_ext {

struct CallOpInterface
    : public BufferizableOpInterface::ExternalModel<CallOpInterface, CallOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    // CallOpInterface alone doesn't bufferize to a memory read, one of the uses
    // of the matching bbArg may. It is the responsibility of the caller to
    // inspect bbArgs. In the absence of a BufferizationAliasInfo, we need to be
    // conservative.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    // CallOpInterface alone doesn't bufferize to a memory write, one of the
    // uses of the matching bbArg may. It is the responsibility of the caller to
    // inspect bbArgs. In the absence of a BufferizationAliasInfo, we need to be
    // conservative.
    return true;
  }

  SmallVector<OpOperand *> getAliasingOpOperand(Operation *op,
                                                OpResult opResult) const {
    // TODO: Can we do better?
    return {};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    // CallOpInterface is special, it needs to wait for the callee to be
    // bufferized and needs to inspect the BufferAliasInfo object. It can't
    // make a proper determination by itself and needs to be conservative.
    return OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpOperand &opOperand) const {
    return BufferRelation::Equivalent;
  }

  /// In a first approximation, all the function arguments of a FuncOp are
  /// marked inplaceable. For now, it is the responsibility of the `callOp`
  /// bufferization to allow FuncOp that are inplaceable to write inPlace.
  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    CallOp callOp = cast<CallOp>(op);
    FuncOp funcOp = getCalledFunction(callOp);
    assert(isa<CallOp>(callOp.getOperation()) && funcOp &&
           "expected Callop to a FuncOp");

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
                getEquivalentEnclosingFuncBBArg(returnVal, state.aliasInfo)) {
          Value oldRes = callOp->getResult(returnOperand.getOperandNumber());
          int64_t idx = bbArg.getArgNumber();
          Value buffer = state.lookupBuffer(callOp->getOperand(idx));
          // Add CallOp operand/result equivalence: this is interprocedural
          // info.
          state.aliasInfo.insertNewBufferEquivalence(oldRes, buffer);
          state.mapBuffer(oldRes, buffer);
          // Add a TensorLoadOp to kill all uses of the CallOp return.
          // Replace all uses of the CallOp results so we can erase the CallOp.
          // This TensorLoadOp must fold/DCE away or bufferization should be
          // considered failed.
          Value tensorLoad =
              b.create<memref::TensorLoadOp>(callOp.getLoc(), buffer);
          oldRes.replaceAllUsesWith(tensorLoad);
          // Add new op equivalence info.
          state.aliasInfo.insertNewBufferEquivalence(tensorLoad, buffer);
          state.mapBuffer(tensorLoad, buffer);
          continue;
        }

        // TODO: Need to hoist above function boundary.
        if (Operation *allocOp =
                getEquivalentAlloc(returnVal, state.aliasInfo)) {
          hoistedArguments.push_back(allocOp->getResult(0));
          continue;
        }

        // Other cases legitimately need to return a tensor, this is currently
        // not supported. For instance, if hoisting across function boundary has
        // failed, it may be due to e.g. data-dependent sizes. In such a case,
        // we would we need a better type than memref.
        resultTypes.push_back(returnType);

        int64_t returnIdx = returnOperand.getOperandNumber();
        return returnOp->emitError() << "buffer result #" << returnIdx
                                     << " not produced by an alloc\n";
      }
    }

    // 2. Compute bufferized FunctionType.
    SmallVector<Type> argumentTypes{callOp->getOperandTypes()};
    ValueRange hoistedArgs{hoistedArguments};
    llvm::append_range(argumentTypes, hoistedArgs.getTypes());
    // Get the bufferized FunctionType for funcOp or construct it if not yet
    // available.
    FunctionType bufferizedFuncType = getOrCreateBufferizedFunctionType(
        funcOp, argumentTypes, resultTypes, state.bufferizedFunctionTypes);

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
      Value buffer = state.lookupBuffer(tensorOperand);

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
        state.aliasInfo.insertNewBufferEquivalence(castBuffer, buffer);
        state.mapBuffer(tensorOperand, castBuffer);
        buffer = castBuffer;
      }
      newOperands.push_back(buffer);
    }

    // 4. Create the new CallOp.
    Operation *newCallOp = b.create<CallOp>(callOp.getLoc(), funcOp.sym_name(),
                                            resultTypes, newOperands);
    newCallOp->setAttrs(callOp->getAttrs());

    // 5. Delete the op at the end of bufferization.
    state.markOpObsolete(callOp);

    return success();
  }
};

struct ReturnOpInterface
    : public BufferizableOpInterface::ExternalModel<ReturnOpInterface,
                                                    ReturnOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand) const {
    return false;
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand) const {
    return OpResult();
  }

  LogicalResult bufferize(Operation *op, OpBuilder &b,
                          BufferizationState &state) const {
    auto returnOp = cast<ReturnOp>(op);

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
      Value v = state.lookupBuffer(operand.get());
      Value returnTensor = b.create<memref::TensorLoadOp>(returnOp.getLoc(), v);
      operand.set(returnTensor);
      state.aliasInfo.insertNewBufferEquivalence(returnTensor, v);
      state.mapBuffer(returnTensor, v);
    }
    return success();
  }
};

} // namespace std_ext

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addOpInterface<arith::ConstantOp, arith_ext::ConstantOpInterface>();
  registry.addOpInterface<scf::ExecuteRegionOp,
                          scf_ext::ExecuteRegionOpInterface>();
  registry.addOpInterface<scf::ForOp, scf_ext::ForOpInterface>();
  registry.addOpInterface<scf::IfOp, scf_ext::IfOpInterface>();
  registry.addOpInterface<scf::YieldOp, scf_ext::YieldOpInterface>();
  registry.addOpInterface<CallOp, std_ext::CallOpInterface>();
  registry.addOpInterface<ReturnOp, std_ext::ReturnOpInterface>();

  // Ops that are not bufferizable but are allocation hoisting barriers.
  registry.addOpInterface<FuncOp, AllocationHoistingBarrierOnly<FuncOp>>();
  registry.addOpInterface<scf::ParallelOp,
                          AllocationHoistingBarrierOnly<scf::ParallelOp>>();
  registry.addOpInterface<AffineParallelOp,
                          AllocationHoistingBarrierOnly<AffineParallelOp>>();
}

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir
