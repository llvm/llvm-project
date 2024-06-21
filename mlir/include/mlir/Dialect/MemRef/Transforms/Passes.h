//===- Passes.h - MemRef Patterns and Passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on MemRef operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class AffineDialect;
class ModuleOp;

namespace func {
namespace arith {
class ArithDialect;
} // namespace arith
class FuncDialect;
} // namespace func
namespace scf {
class SCFDialect;
} // namespace scf
namespace tensor {
class TensorDialect;
} // namespace tensor
namespace vector {
class VectorDialect;
} // namespace vector

class BufferViewFlowAnalysis;
namespace memref {
//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

/// Creates an instance of the ExpandOps pass that legalizes memref dialect ops
/// to be convertible to LLVM. For example, `memref.reshape` gets converted to
/// `memref_reinterpret_cast`.
std::unique_ptr<Pass> createExpandOpsPass();

/// Creates an operation pass to fold memref aliasing ops into consumer
/// load/store ops into `patterns`.
std::unique_ptr<Pass> createFoldMemRefAliasOpsPass();

/// Creates an interprocedural pass to normalize memrefs to have a trivial
/// (identity) layout map.
std::unique_ptr<OperationPass<ModuleOp>> createNormalizeMemRefsPass();

/// Creates an operation pass to resolve `memref.dim` operations with values
/// that are defined by operations that implement the
/// `ReifyRankedShapedTypeOpInterface`, in terms of shapes of its input
/// operands.
std::unique_ptr<Pass> createResolveRankedShapeTypeResultDimsPass();

/// Creates an operation pass to resolve `memref.dim` operations with values
/// that are defined by operations that implement the
/// `InferShapedTypeOpInterface` or the `ReifyRankedShapedTypeOpInterface`,
/// in terms of shapes of its input operands.
std::unique_ptr<Pass> createResolveShapedTypeResultDimsPass();

/// Creates an operation pass to expand some memref operation into
/// easier to reason about operations.
std::unique_ptr<Pass> createExpandStridedMetadataPass();

/// Creates an operation pass to expand `memref.realloc` operations into their
/// components.
std::unique_ptr<Pass> createExpandReallocPass(bool emitDeallocs = true);

/// abstract base class for lifetime of buffers in the same "allocation scope".
/// It should hold the lifetime informantion of buffers that are to be merged in
/// the same allocation in an "allocation scope". TraceCollectorFunc decides
/// which buffers are put into which "allocation scope".
class LifetimeTrace {
public:
  enum TraceKind { TK_TICK };
  virtual ~LifetimeTrace() = default;
  LifetimeTrace(TraceKind kind) : kind{kind} {}
  TraceKind getKind() const { return kind; }
  virtual Block *getAllocScope() const = 0;
  virtual Attribute getMemorySpace() const = 0;

private:
  TraceKind kind;
};

/// top level memory trace info for multiple scopes. Each element of scopeTraces
/// should contain an "allocation scope" and the implementation-defined lifetime
/// data
struct MemoryTraceScopes {
  llvm::SmallVector<std::unique_ptr<LifetimeTrace>> scopeTraces;
  MemoryTraceScopes() = default;
};

/// the memory scheduling result for allocations in the same allocation scope.
/// allocation => offset map. All Operation* in the map should be
/// memref::AllocOp which are in the same LifetimeTrace.
struct MemorySchedule {
  size_t totalSize;
  Attribute memorySpace;
  llvm::DenseMap<Operation *, int64_t> allocToOffset;
  MemorySchedule() : totalSize{0} {}
};

struct MergeAllocationOptions;
using TraceCollectorFunc = std::function<FailureOr<MemoryTraceScopes>(
    Operation *, const BufferViewFlowAnalysis &,
    const MergeAllocationOptions &)>;
using MemoryPlannerFunc = std::function<FailureOr<MemorySchedule>(
    Operation *, const LifetimeTrace &, const MergeAllocationOptions &)>;
using MemoryMergeMutatorFunc = std::function<LogicalResult(
    Operation *toplevel, Block *scope, const MemorySchedule &,
    const MergeAllocationOptions &)>;

struct MergeAllocationOptions {
  bool checkOnly = false;
  std::string plannerOptions;
  int64_t alignment = 64;
  TraceCollectorFunc tracer;
  MemoryPlannerFunc planner;
  MemoryMergeMutatorFunc mutator;
};

/// Creates an operation pass to merge the local memref allocations
std::unique_ptr<Pass> createMergeAllocPass(const MergeAllocationOptions &o);
std::unique_ptr<Pass> createMergeAllocPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
