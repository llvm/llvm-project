//===- BufferizableOpInterface.h - Bufferizable Ops -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

#include "mlir/Dialect/Bufferization/IR/BufferizationEnums.h.inc"

namespace mlir {
class OpBuilder;

namespace bufferization {

class AnalysisState;
class BufferizableOpInterface;

class OpFilter {
public:
  /// An op filter entry. Filters can be used to specify which ops should be
  /// processed by the bufferization.
  struct Entry {
    /// If the filter function evaluates to `true`, the filter matches.
    using FilterFn = std::function<bool(Operation *)>;

    /// Filter type: A filter can either be a DENY filter or an ALLOW filter.
    enum FilterType : int8_t { DENY = 0, ALLOW = 1 };

    FilterFn fn;
    FilterType type;
  };

  /// Return whether the op is allowed or not.
  ///
  /// If the filter does not have an ALLOW rule, ops are allowed by default,
  /// unless they are explicitly marked as DENY. If the filter has at least one
  /// ALLOW rule, ops are denied by default and only allowed if they match
  /// an ALLOW rule and no DENY rule.
  bool isOpAllowed(Operation *op) const;

  /// Allow the given dialects.
  ///
  /// This function adds one or multiple ALLOW entries.
  template <typename... DialectTs>
  void allowDialect() {
    // The following expands a call to allowDialectImpl for each dialect
    // in 'DialectTs'.
    (allowDialectImpl<DialectTs>(), ...);
  }

  /// Deny the given dialects.
  ///
  /// This function adds one or multiple DENY entries.
  template <typename... DialectTs>
  void denyDialect() {
    (denyDialectImpl<DialectTs>(), ...);
  }

  /// Allow the given dialect.
  ///
  /// This function adds an ALLOW entry.
  void allowDialect(StringRef dialectNamespace) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getDialect()->getNamespace() == dialectNamespace;
    };
    entries.push_back(Entry{filterFn, Entry::FilterType::ALLOW});
  }

  /// Allow the given ops.
  ///
  /// This function adds one or multiple ALLOW entries.
  template <typename... OpTys>
  void allowOperation() {
    (allowOperationImpl<OpTys>(), ...);
  }

  /// Deny the given ops.
  ///
  /// This function adds one or multiple DENY entries.
  template <typename... OpTys>
  void denyOperation() {
    (denyOperationImpl<OpTys>(), ...);
  }

  /// Allow the given op.
  ///
  /// This function adds an ALLOW entry.
  void allowOperation(StringRef opName) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getStringRef() == opName;
    };
    allowOperation(filterFn);
  }

  /// Deny the given op.
  ///
  /// This function adds a DENY entry.
  void denyOperation(StringRef opName) {
    Entry::FilterFn filterFn = [=](Operation *op) {
      return op->getName().getStringRef() == opName;
    };
    denyOperation(filterFn);
  }

  /// Allow ops that are matched by `fn`.
  ///
  /// This function adds an ALLOW entry.
  void allowOperation(Entry::FilterFn fn) {
    entries.push_back(Entry{fn, Entry::FilterType::ALLOW});
  }

  /// Deny ops that are matched by `fn`.
  ///
  /// This function adds a DENY entry.
  void denyOperation(Entry::FilterFn fn) {
    entries.push_back(Entry{fn, Entry::FilterType::DENY});
  }

private:
  /// Return `true` if the filter has at least one ALLOW rule.
  bool hasAllowRule() const {
    for (const Entry &e : entries)
      if (e.type == Entry::FilterType::ALLOW)
        return true;
    return false;
  }

  /// Allow a dialect.
  template <typename DialectT>
  void allowDialectImpl() {
    allowDialect(DialectT::getDialectNamespace());
  }

  /// Deny a dialect.
  template <typename DialectT>
  void denyDialectImpl() {
    denyDialect(DialectT::getDialectNamespace());
  }

  /// Allow an op.
  template <typename OpTy>
  void allowOperationImpl() {
    allowOperation(OpTy::getOperationName());
  }

  /// Deny an op.
  template <typename OpTy>
  void denyOperationImpl() {
    denyOperation(OpTy::getOperationName());
  }

  /// A list of filter entries that determine whether an op should be allowed or
  /// denied. If the filter has an ALLOW rule, only ops that are allowed and not
  /// denied are allowed. If the filter does not have an ALLOW rule, only ops
  /// that are not denied are allowed.
  SmallVector<Entry> entries;
};

/// Options for BufferizableOpInterface-based bufferization.
struct BufferizationOptions {
  /// Allocator function: Generate a memref allocation with the given type,
  /// dynamic extents and alignment.
  using AllocationFn = std::function<FailureOr<Value>(
      OpBuilder &, Location, MemRefType, ValueRange, unsigned int)>;
  /// Deallocator function: Deallocate a buffer that was allocated with
  /// AllocatorFn.
  using DeallocationFn =
      std::function<LogicalResult(OpBuilder &, Location, Value)>;
  /// Memcpy function: Generate a memcpy between two buffers.
  using MemCpyFn =
      std::function<LogicalResult(OpBuilder &, Location, Value, Value)>;
  /// Initializer function for analysis state.
  using AnalysisStateInitFn = std::function<void(AnalysisState &)>;
  /// Tensor -> MemRef type converter.
  /// Parameters: Value, memory space, bufferization options
  using UnknownTypeConverterFn = std::function<BaseMemRefType(
      Value, Attribute memorySpace, const BufferizationOptions &)>;

  BufferizationOptions();

  /// Try to cast the given op to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Operation *op) const;

  /// Try to cast the given value to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Value value) const;

  /// A filter that specifies which ops should be bufferized and which ops
  /// should be ignored.
  OpFilter opFilter;

  /// Return `true` if the given op should be bufferized.
  bool isOpAllowed(Operation *op) const;

  /// Helper functions for allocation, deallocation, memory copying.
  std::optional<AllocationFn> allocationFn;
  std::optional<DeallocationFn> deallocationFn;
  std::optional<MemCpyFn> memCpyFn;

  /// Create a memref allocation with the given type and dynamic extents.
  FailureOr<Value> createAlloc(OpBuilder &b, Location loc, MemRefType type,
                               ValueRange dynShape) const;

  /// Creates a memref deallocation. The given memref buffer must have been
  /// allocated using `createAlloc`.
  LogicalResult createDealloc(OpBuilder &b, Location loc,
                              Value allocatedBuffer) const;

  /// Creates a memcpy between two given buffers.
  LogicalResult createMemCpy(OpBuilder &b, Location loc, Value from,
                             Value to) const;

  /// Specifies whether not bufferizable ops are allowed in the input. If so,
  /// bufferization.to_memref and bufferization.to_tensor ops are inserted at
  /// the boundaries.
  bool allowUnknownOps = false;

  /// Specifies whether function boundaries (ops in the func dialect) should be
  /// bufferized or not.
  bool bufferizeFunctionBoundaries = false;

  /// The default memory space that should be used when it cannot be inferred
  /// from the context. If case of std::nullopt, bufferization fails when the
  /// memory space cannot be inferred at any point.
  std::optional<Attribute> defaultMemorySpace = Attribute();

  /// Certain ops have aliasing OpOperand/OpResult invariants (e.g., scf.for).
  /// If this flag is set to `false`, those invariants are no longer enforced
  /// with buffer copies.
  ///
  /// Note: Deactivating this flag can lead to incorrect bufferization results
  /// when used incorrectly. This flag is useful with
  /// `AlwaysCopyAnalysisState` which bufferizes all writing tensor
  /// OpOperands out-of-place.
  bool enforceAliasingInvariants = true;

  /// This flag controls buffer types on function signatures.
  ///
  /// * InferLayoutMap: All function parameter types have a fully dynamic layout
  ///   map, but function result types are inferred from the body of the
  ///   function.
  /// * FullyDynamicLayoutMap: All function parameter types and result types
  ///   have a fully dynamic layout map. This option is most efficient because
  ///   any layout map can be casted to a fully dynamic one.
  /// * IdentityLayoutMap: All function parameter types and result types have a
  ///   static identity layout (i.e., no layout map). This option may introduce
  ///   additional buffer allocs and copies because layout maps cannot be casted
  ///   away.
  ///
  /// If `bufferizeFunctionBoundaries` is not set, this flag has no effect.
  ///
  /// Note: Inferred layout maps may not be desireable when interacting with
  /// external functions, because the generated function signatures will be less
  /// predictable.
  LayoutMapOption functionBoundaryTypeConversion =
      LayoutMapOption::InferLayoutMap;

  /// Type converter from tensors to memrefs. This type converter is used if no
  /// memref type could be inferred during bufferization. By default, a type
  /// converter that returns a memref type with a fully dynamic layout map is
  /// used.
  UnknownTypeConverterFn unknownTypeConverterFn = nullptr;

  /// Specifies whether dealloc ops should be generated along with alloc ops. If
  /// not, new memory allocations will leak.
  bool createDeallocs = true;

  /// Seed for the analysis fuzzer. If set to `0`, the fuzzer is deactivated.
  /// Should be used only with `testAnalysisOnly = true`.
  unsigned analysisFuzzerSeed = 0;

  /// If set to `true`, the analysis is skipped. A buffer is copied before every
  /// write. This flag cannot be used together with `testAnalysisOnly = true`.
  bool copyBeforeWrite = false;

  /// If set to `true`, does not modify the IR apart from adding attributes (for
  /// checking the results of the analysis) and post analysis steps.
  bool testAnalysisOnly = false;

  /// If set to `true`, the IR is annotated with details about RaW conflicts.
  /// For debugging only. Should be used together with `testAnalysisOnly`.
  bool printConflicts = false;

  /// Buffer alignment for new memory allocations.
  unsigned int bufferAlignment = 64;

  /// Initializer functions for analysis state. These can be used to
  /// initialize dialect-specific analysis state.
  SmallVector<AnalysisStateInitFn> stateInitializers;
};

/// Specify fine-grain relationship between buffers to enable more analysis.
enum class BufferRelation {
  None,
  // TODO: ResultContainsOperand,
  // TODO: OperandContainsResult,
  Equivalent
};

/// Return `true` if the given value is a BlockArgument of a func::FuncOp.
bool isFunctionArgument(Value value);

/// AnalysisState provides a variety of helper functions for dealing with
/// tensor values.
class AnalysisState {
public:
  /// Determine which OpOperand* will alias with `opResult` if the op is
  /// bufferized in place. Return all tensor OpOperand* if the op is not
  /// bufferizable.
  SmallVector<OpOperand *> getAliasingOpOperand(OpResult opResult) const;

  /// Determine which OpResult will alias with `opOperand` if the op is
  /// bufferized in place. Return all tensor OpResults if the op is not
  /// bufferizable.
  SmallVector<OpResult> getAliasingOpResult(OpOperand &opOperand) const;

  /// Return true if `opOperand` bufferizes to a memory read. Return `true` if
  /// the op is not bufferizable.
  bool bufferizesToMemoryRead(OpOperand &opOperand) const;

  /// Return true if `opOperand` bufferizes to a memory write. Return true` if
  /// the op is not bufferizable.
  bool bufferizesToMemoryWrite(OpOperand &opOperand) const;

  /// Return true if the given `value` bufferizes to a memory write. Return
  /// true if the value is a block argument. Return `true` if the defining op is
  /// not bufferizable. Otherwise, consult the BufferizableOpInterface.
  bool bufferizesToMemoryWrite(Value value) const;

  /// Return true if `opOperand` does neither read nor write but bufferizes to
  /// an alias. Return false if the op is not bufferizable.
  bool bufferizesToAliasOnly(OpOperand &opOperand) const;

  /// Return true if a copy can always be avoided when allocating a new tensor
  /// for the given OpOperand.
  bool canOmitTensorCopy(OpOperand &opOperand) const;

  /// Return true if the given value is read by an op that bufferizes to a
  /// memory read. Also takes into account ops that create an alias but do not
  /// read by themselves (e.g., ExtractSliceOp).
  bool isValueRead(Value value) const;

  /// Starting from `value`, follow the use-def chain in reverse, always
  /// selecting the aliasing OpOperands. Find and return Values for which
  /// `condition` evaluates to true. OpOperands of such matching Values are not
  /// traversed any further.
  ///
  /// When reaching the end of a chain (BlockArgument or Value without aliasing
  /// OpOperands), also return the last Value of that chain if
  /// `alwaysIncludeLeaves` is set.
  ///
  /// Example:
  ///
  ///                               8
  ///                               |
  ///   6*         7*         +-----+----+
  ///   |          |          |          |
  ///   2*         3          4*         5
  ///   |          |          |          |
  ///   +----------+----------+----------+
  ///              |
  ///              1
  ///
  /// In the above example, Values with a star satisfy the condition. When
  /// starting the traversal from Value 1, the resulting SetVector is:
  /// { 2, 7, 8, 5 }
  ///
  /// If `followEquivalentOnly` is set, only equivalent OpOperands are selected.
  SetVector<Value> findValueInReverseUseDefChain(
      Value value, llvm::function_ref<bool(Value)> condition,
      bool followEquivalentOnly = false, bool alwaysIncludeLeaves = true) const;

  /// Find the values that may define the contents of the given value at
  /// runtime. A block argument is always a definition. An OpResult is a
  /// definition if it bufferizes to memory write. If it does not bufferize to
  /// a memory write but has aliasing operands, we continue the lookup on these
  /// values.
  ///
  /// Example: %r = tensor.insert %f into %t[%c0] : tensor<?xf32>
  /// findDefinitions(%r) = {%r} because %r bufferizes to memory write.
  ///
  /// Example: %r = tensor.empty() : tensor<10xf32>
  /// findDefinitions(%r) = {} because tensor.empty does not the define the
  /// contents of its result (i.e., it does not bufferize to a memory write)
  /// and it has no aliasing OpOperands.
  ///
  /// Example:
  /// %a = arith.constant ... : tensor<10xf32>
  /// %b1 = tensor.insert %f into %t : tensor<50xf32>
  /// %b2 = tensor.extract_slice %b1[0][10][1] : tensor<50xf32> tensor<10xf32>
  /// %r = arith.select %cond, %a, %b : tensor<10xf32>
  /// findDefinitions(%r) = {%a, %b1}. %r and %b2 are skipped (lookup continues
  /// in the operands) because their defining ops do not define the contents of
  /// the tensor.
  ///
  /// Note: OpResults of unknown ops are handled conservatively and assumed to
  /// be definitions.
  ///
  /// Note: When reaching an end of the reverse SSA use-def chain, that value
  /// is included regardless of whether it is a definition or not unless
  /// `alwaysIncludeLeaves` is unset.
  SetVector<Value> findDefinitions(Value value,
                                   bool alwaysIncludeLeaves = true) const;

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  virtual bool isInPlace(OpOperand &opOperand) const;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  virtual bool areEquivalentBufferizedValues(Value v1, Value v2) const;

  /// Return true if `v1` and `v2` may bufferize to aliasing buffers.
  virtual bool areAliasingBufferizedValues(Value v1, Value v2) const;

  /// Return `true` if the given tensor has undefined contents.
  virtual bool hasUndefinedContents(OpOperand *opOperand) const;

  /// Return true if the given tensor (or an aliasing tensor) is yielded from
  /// the containing block. Also include all aliasing tensors in the same block.
  ///
  /// Note: In the absence of an analysis, an implementation may return true for
  /// any given tensor.
  virtual bool isTensorYielded(Value tensor) const;

  /// Return a reference to the BufferizationOptions.
  const BufferizationOptions &getOptions() const { return options; }

  AnalysisState(const BufferizationOptions &options);

  // AnalysisState should be passed as a reference.
  AnalysisState(const AnalysisState &) = delete;

  virtual ~AnalysisState() = default;

  static bool classof(const AnalysisState *base) { return true; }

  TypeID getType() const { return type; }

protected:
  AnalysisState(const BufferizationOptions &options, TypeID type);

private:
  /// A reference to current bufferization options.
  const BufferizationOptions &options;

  /// The type of analysis.
  TypeID type;
};

/// Create an AllocTensorOp for the given shaped value (memref or tensor).
/// If `copy` is set, the shaped value is copied. Otherwise, a tensor with
/// undefined contents is allocated.
FailureOr<Value>
allocateTensorForShapedValue(OpBuilder &b, Location loc, Value shapedValue,
                             bool escape, const BufferizationOptions &options,
                             bool copy = true);

/// Return `true` if the allocation of the given op is guaranteed to not escape
/// the containing block.
bool allocationDoesNotEscape(OpResult opResult);

/// Lookup the buffer for the given value. If the value was not bufferized
/// yet, wrap it in a ToMemrefOp. Otherwise, it is the result of a ToTensorOp,
/// from which the memref operand is returned.
FailureOr<Value> getBuffer(RewriterBase &rewriter, Value value,
                           const BufferizationOptions &options);

/// Return the buffer type for a given Value (tensor) after bufferization
/// without bufferizing any IR.
///
/// Note: It should be sufficient to call `getBuffer()->getType()` in most
/// cases. However, when a buffer type should be predicted without modifying any
/// IR, this function can be used.
///
/// This function is a wrapper around BufferizableOpInterface::getBufferType.
FailureOr<BaseMemRefType> getBufferType(Value value,
                                        const BufferizationOptions &options);

/// Return the buffer type for a given Value (tensor) after bufferization
/// without bufferizing any IR. If at any point during the type computation, the
/// type of a value in `fixedTypes` in required, the mapped type is used.
///
/// Note: It should be sufficient to call `getBuffer()->getType()` in most
/// cases. However, when a buffer type should be predicted without modifying any
/// IR, this function can be used.
///
/// This function is a wrapper around BufferizableOpInterface::getBufferType.
FailureOr<BaseMemRefType>
getBufferType(Value value, const BufferizationOptions &options,
              const DenseMap<Value, BaseMemRefType> &fixedTypes);

/// Replace an op with replacement values. The op is deleted. Tensor OpResults
/// must be replaced with memref values.
void replaceOpWithBufferizedValues(RewriterBase &rewriter, Operation *op,
                                   ValueRange values);

/// Replace an op with a new op. The new op must have the same number of
/// results as the replaced op. The new op may not return any tensor values.
template <typename OpTy, typename... Args>
OpTy replaceOpWithNewBufferizedOp(RewriterBase &rewriter, Operation *op,
                                  Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
  return newOp;
}

/// Return `true` if the buffer of given OpResult should be deallocated. This
/// function should be called during `BufferizableOpInterface::bufferize`
/// implementations that allocate a new buffer for the given OpResult.
bool shouldDeallocateOpResult(OpResult opResult,
                              const BufferizationOptions &options);

/// Return a MemRefType to which the type of the given value can be bufferized.
///
/// If possible, op bufferization implementations should not use this function
/// and instead infer precise memref types for tensor results by themselves.
///
/// Unless a layout map was specified, `options.unknownTypeConverterFn`
/// determines what kind of layout map will be used. For best composability
/// (without copies), the fully dynamic layout map is used by default.
///
/// Note: Canonicalization patterns could clean up layout maps and infer more
/// precise layout maps after bufferization. However, many possible
/// canonicalizations are currently not implemented.
BaseMemRefType getMemRefType(Value value, const BufferizationOptions &options,
                             MemRefLayoutAttrInterface layout = {},
                             Attribute memorySpace = nullptr);

/// Return a MemRef type with fully dynamic layout. If the given tensor type
/// is unranked, return an unranked MemRef type.
BaseMemRefType
getMemRefTypeWithFullyDynamicLayout(TensorType tensorType,
                                    Attribute memorySpace = nullptr);

/// Return a MemRef type with a static identity layout (i.e., no layout map). If
/// the given tensor type is unranked, return an unranked MemRef type.
BaseMemRefType
getMemRefTypeWithStaticIdentityLayout(TensorType tensorType,
                                      Attribute memorySpace = nullptr);

/// Return the owner of the given value. In case of a BlockArgument that is the
/// owner of the block. In case of an OpResult that is the defining op.
Operation *getOwnerOfValue(Value value);

/// Return the closest enclosing repetitive region around the given op.
Region *getEnclosingRepetitiveRegion(Operation *op,
                                     const BufferizationOptions &options);

/// Return the closest enclosing repetitive region around the place where the
/// given value is defined.
Region *getEnclosingRepetitiveRegion(Value value,
                                     const BufferizationOptions &options);

/// Return the closest enclosing repetitive region around the given block.
Region *getEnclosingRepetitiveRegion(Block *block,
                                     const BufferizationOptions &options);

namespace detail {
/// This is the default implementation of
/// BufferizableOpInterface::getBufferType. Should not be called from other
/// places.
FailureOr<BaseMemRefType>
defaultGetBufferType(Value value, const BufferizationOptions &options,
                     const DenseMap<Value, BaseMemRefType> &fixedTypes);

/// This is the default implementation of
/// BufferizableOpInterface::resultBufferizesToMemoryWrite. Should not be called
/// from other places.
bool defaultResultBufferizesToMemoryWrite(OpResult opResult,
                                          const AnalysisState &state);

/// This is the default implementation of
/// BufferizableOpInterface::isRepetitiveRegion. Should not be called from other
/// places.
bool defaultIsRepetitiveRegion(BufferizableOpInterface bufferizableOp,
                               unsigned index);

/// This is the default implementation of getAliasingOpOperand in case the
/// defining op does not implement the BufferizableOpInterface.
SmallVector<OpOperand *> unknownGetAliasingOpOperand(OpResult opResult);

/// This is the default implementation of getAliasingOpResult in case the
/// defining op does not implement the BufferizableOpInterface.
SmallVector<OpResult> unknownGetAliasingOpResult(OpOperand &opOperand);
} // namespace detail

} // namespace bufferization
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::bufferization::AnalysisState)

//===----------------------------------------------------------------------===//
// Bufferization Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZABLEOPINTERFACE_H_
