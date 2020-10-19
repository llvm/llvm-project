//===- Bufferize.h - Bufferization utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// We use the term "bufferize" to mean conversion from tensor types to
// memref types.
//
// Generally speaking, for each op that operates on tensor types, a conversion
// pattern needs to be written. The infrastructure in this file assists in
// defining these conversion patterns in a composable way.
//
// Bufferization conversion patterns should generally use the ordinary
// conversion pattern classes (e.g. OpConversionPattern). A TypeConverter
// (accessible with getTypeConverter()) available on such patterns is sufficient
// for most cases (if needed at all).
//
// But some patterns require access to the extra functions on
// BufferizeTypeConverter that don't exist on the base TypeConverter class. For
// those cases, BufferizeConversionPattern and its related classes should be
// used, which provide access to a BufferizeTypeConverter directly.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_BUFFERIZE_H
#define MLIR_TRANSFORMS_BUFFERIZE_H

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

/// A helper type converter class for using inside Buffer Assignment operation
/// conversion patterns. The default constructor keeps all the types intact
/// except for the ranked-tensor types which is converted to memref types.
class BufferizeTypeConverter : public TypeConverter {
public:
  /// This enum is for showing how buffer placement operation converters should
  /// conduct with certain result type after type conversion. This value can be
  /// set/get for each specific type using setResultConversionKind or
  /// getResultConversionKind.
  enum ResultConversionKind { AppendToArgumentsList, KeepAsFunctionResult };

  BufferizeTypeConverter();

  /// This method tries to decompose a value of a certain type using provided
  /// decompose callback functions. If it is unable to do so, the original value
  /// is returned.
  void tryDecomposeValue(OpBuilder &, Location, Type, Value,
                         SmallVectorImpl<Value> &);

  /// This method tries to decompose a type using provided decompose callback
  /// functions. If it is unable to do so, the original type is returned.
  void tryDecomposeType(Type, SmallVectorImpl<Type> &);

  /// This method registers a callback function that will be called to decompose
  /// a value of a certain type into several values.
  template <typename FnT,
            typename T = typename llvm::function_traits<FnT>::template arg_t<2>>
  void addDecomposeValueConversion(FnT &&callback) {
    decomposeValueConversions.emplace_back(
        wrapDecomposeValueConversionCallback<T>(std::forward<FnT>(callback)));
  }

  /// This method registers a callback function that will be called to decompose
  /// a type into several types.
  template <typename FnT,
            typename T = typename llvm::function_traits<FnT>::template arg_t<0>>
  void addDecomposeTypeConversion(FnT &&callback) {
    auto wrapper =
        wrapDecomposeTypeConversionCallback<T>(std::forward<FnT>(callback));
    decomposeTypeConversions.emplace_back(wrapper);
    addConversion(std::forward<FnT>(callback));
  }

  /// This method returns ResultConversionKind for the mapping from `origin`
  /// type to `input` type.
  ResultConversionKind getResultConversionKind(Type origin, Type input);

  /// This method registers ResultConversionKind for the mapping from type 'T'
  /// to type 'U'.
  template <typename T, typename U>
  void setResultConversionKind(ResultConversionKind kind) {
    assert((kind != AppendToArgumentsList ||
            llvm::is_one_of<U, MemRefType, UnrankedMemRefType>::value) &&
           "Only the memref typed values can be set to be appended to the "
           "function argument list at the moment");
    resultTypeConversions.emplace_back(
        [=](Type origin, Type input) -> Optional<ResultConversionKind> {
          if (origin.template isa<T>() && input.template isa<U>())
            return kind;
          return llvm::None;
        });
  }

private:
  using DecomposeValueConversionCallFn = std::function<Optional<LogicalResult>(
      OpBuilder &, Location, Type, Value, SmallVectorImpl<Value> &)>;

  using DecomposeTypeConversionCallFn =
      std::function<Optional<LogicalResult>(Type, SmallVectorImpl<Type> &)>;

  using ResultConversionKindFn =
      std::function<Optional<ResultConversionKind>(Type, Type)>;

  /// Generate a wrapper for the given decompose value conversion callback.
  template <typename T, typename FnT>
  DecomposeValueConversionCallFn
  wrapDecomposeValueConversionCallback(FnT &&callback) {
    return [callback = std::forward<FnT>(callback)](
               OpBuilder &builder, Location loc, Type type, Value value,
               SmallVectorImpl<Value> &newValues) -> Optional<LogicalResult> {
      if (T derivedType = type.dyn_cast<T>())
        return callback(builder, loc, derivedType, value, newValues);
      return llvm::None;
    };
  }

  /// Generate a wrapper for the given decompose type conversion callback.
  template <typename T, typename FnT>
  DecomposeTypeConversionCallFn
  wrapDecomposeTypeConversionCallback(FnT &&callback) {
    return [callback = std::forward<FnT>(callback)](
               Type type,
               SmallVectorImpl<Type> &results) -> Optional<LogicalResult> {
      T derivedType = type.dyn_cast<T>();
      if (!derivedType)
        return llvm::None;
      return callback(derivedType, results);
    };
  }

  SmallVector<ResultConversionKindFn, 2> resultTypeConversions;
  SmallVector<DecomposeValueConversionCallFn, 2> decomposeValueConversions;
  SmallVector<DecomposeTypeConversionCallFn, 2> decomposeTypeConversions;
};

/// Helper conversion pattern that encapsulates a BufferizeTypeConverter
/// instance.
template <typename SourceOp>
class BufferizeOpConversionPattern : public OpConversionPattern<SourceOp> {
public:
  explicit BufferizeOpConversionPattern(MLIRContext *context,
                                        BufferizeTypeConverter &converter,
                                        PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit), converter(converter) {}

protected:
  BufferizeTypeConverter &converter;
};

/// Helper conversion pattern that encapsulates a BufferizeTypeConverter
/// instance and that operates on Operation* to be compatible with OpInterfaces.
/// This allows avoiding to instantiate N patterns for ops that can be subsumed
/// by a single op interface (e.g. Linalg named ops).
class BufferizeConversionPattern : public ConversionPattern {
public:
  explicit BufferizeConversionPattern(MLIRContext *context,
                                      BufferizeTypeConverter &converter,
                                      PatternBenefit benefit = 1)
      : ConversionPattern(benefit, converter, MatchAnyOpTypeTag()),
        converter(converter) {}

protected:
  BufferizeTypeConverter &converter;
};

/// Converts the signature of the function using BufferizeTypeConverter.
/// Each result type of the function is kept as a function result or appended to
/// the function arguments list based on ResultConversionKind for the converted
/// result type.
class BufferizeFuncOpConverter : public BufferizeOpConversionPattern<FuncOp> {
public:
  using BufferizeOpConversionPattern<FuncOp>::BufferizeOpConversionPattern;

  /// Performs the actual signature rewriting step.
  LogicalResult matchAndRewrite(mlir::FuncOp, ArrayRef<Value>,
                                ConversionPatternRewriter &) const override;
};

/// Rewrites the `ReturnOp` to conform with the changed function signature.
/// Operands that correspond to return values and their types have been set to
/// AppendToArgumentsList are dropped. In their place, a corresponding copy
/// operation from the operand to the target function argument is inserted.
template <typename ReturnOpSourceTy, typename ReturnOpTargetTy,
          typename CopyOpTy>
class BufferizeReturnOpConverter
    : public BufferizeOpConversionPattern<ReturnOpSourceTy> {
public:
  using BufferizeOpConversionPattern<
      ReturnOpSourceTy>::BufferizeOpConversionPattern;

  /// Performs the actual return-op conversion step.
  LogicalResult
  matchAndRewrite(ReturnOpSourceTy returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = returnOp.getLoc();

    // Split the operands depending on whether they need a copy operation or
    // they remain as operands of the return operation. If an operand is
    // decomposable and a decompose callback function has been provided by the
    // user, it will be unpacked.
    SmallVector<Value, 2> newOperands, needCopyOperands;
    OpBuilder builder(returnOp);
    for (auto operand : llvm::enumerate(operands)) {
      SmallVector<Value, 2> values;
      this->converter.tryDecomposeValue(builder, loc, operand.value().getType(),
                                        operand.value(), values);
      Type type = returnOp.getOperand(operand.index()).getType();
      SmallVector<Type, 2> originTypes;
      this->converter.tryDecomposeType(type, originTypes);
      for (auto value : llvm::enumerate(values)) {
        Type origin = originTypes[value.index()];
        Type converted = value.value().getType();
        auto kind = this->converter.getResultConversionKind(origin, converted);
        if (kind == BufferizeTypeConverter::KeepAsFunctionResult)
          newOperands.push_back(value.value());
        else
          // kind = BufferizeTypeConverter::AppendToArgumentsList
          needCopyOperands.push_back(value.value());
      }
    }

    // Insert Copy operations instead for the operands that have been removed
    // from operand list and appended to the function arguments list.
    Block &entryBlock = returnOp.getParentRegion()->front();
    unsigned numFuncArgs = entryBlock.getNumArguments();
    if (needCopyOperands.size() > numFuncArgs)
      return returnOp.emitError(
          "The number of operands that need Copy operations is more "
          "than the number of target function arguments.");
    unsigned destArgNum = numFuncArgs - needCopyOperands.size();
    rewriter.setInsertionPoint(returnOp);
    for (Value operand : needCopyOperands) {
      rewriter.create<CopyOpTy>(loc, operand,
                                entryBlock.getArgument(destArgNum));
      ++destArgNum;
    }
    rewriter.replaceOpWithNewOp<ReturnOpTargetTy>(returnOp, newOperands);
    return success();
  }
};

/// Rewrites the `CallOp` to match its operands and results with the signature
/// of the callee after rewriting the callee with
/// BufferizeFuncOpConverter.
class BufferizeCallOpConverter : public BufferizeOpConversionPattern<CallOp> {
public:
  using BufferizeOpConversionPattern<CallOp>::BufferizeOpConversionPattern;

  /// Performs the actual rewriting step.
  LogicalResult matchAndRewrite(CallOp, ArrayRef<Value>,
                                ConversionPatternRewriter &) const override;
};

/// Populates `patterns` with the conversion patterns of buffer
/// assignment.
template <typename ReturnOpSourceTy, typename ReturnOpTargetTy,
          typename CopyOpTy>
static void
populateWithBufferizeOpConversionPatterns(MLIRContext *context,
                                          BufferizeTypeConverter &converter,
                                          OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.insert<
    BufferizeCallOpConverter,
    BufferizeFuncOpConverter,
    BufferizeReturnOpConverter
      <ReturnOpSourceTy, ReturnOpTargetTy, CopyOpTy>
  >(context, converter);
  // clang-format on
}

/// A straight-forward alias analysis which ensures that all aliases of all
/// values will be determined. This is a requirement for the BufferPlacement
/// class since you need to determine safe positions to place alloc and
/// deallocs.
class BufferPlacementAliasAnalysis {
public:
  using ValueSetT = SmallPtrSet<Value, 16>;
  using ValueMapT = llvm::DenseMap<Value, ValueSetT>;

public:
  /// Constructs a new alias analysis using the op provided.
  BufferPlacementAliasAnalysis(Operation *op);

  /// Find all immediate aliases this value could potentially have.
  ValueMapT::const_iterator find(Value value) const {
    return aliases.find(value);
  }

  /// Returns the begin iterator to iterate over all aliases.
  ValueMapT::const_iterator begin() const { return aliases.begin(); }

  /// Returns the end iterator that can be used in combination with find.
  ValueMapT::const_iterator end() const { return aliases.end(); }

  /// Find all immediate and indirect aliases this value could potentially
  /// have. Note that the resulting set will also contain the value provided as
  /// it is an alias of itself.
  ValueSetT resolve(Value value) const;

  /// Removes the given values from all alias sets.
  void remove(const SmallPtrSetImpl<Value> &aliasValues);

private:
  /// This function constructs a mapping from values to its immediate aliases.
  void build(Operation *op);

  /// Maps values to all immediate aliases this value can have.
  ValueMapT aliases;
};

/// A simple analysis that detects allocation operations.
class BufferPlacementAllocs {
public:
  /// Represents a tuple of allocValue and deallocOperation.
  using AllocEntry = std::tuple<Value, Operation *>;

  /// Represents a list containing all alloc entries.
  using AllocEntryList = SmallVector<AllocEntry, 8>;

  /// Get the start operation to place the given alloc value withing the
  // specified placement block.
  static Operation *getStartOperation(Value allocValue, Block *placementBlock,
                                      const Liveness &liveness);

  /// Find an associated dealloc operation that is linked to the given
  /// allocation node (if any).
  static Operation *findDealloc(Value allocValue);

public:
  /// Initializes the internal list by discovering all supported allocation
  /// nodes.
  BufferPlacementAllocs(Operation *op);

  /// Returns the begin iterator to iterate over all allocations.
  AllocEntryList::const_iterator begin() const { return allocs.begin(); }

  /// Returns the end iterator that can be used in combination with begin.
  AllocEntryList::const_iterator end() const { return allocs.end(); }

  /// Returns the begin iterator to iterate over all allocations.
  AllocEntryList::iterator begin() { return allocs.begin(); }

  /// Returns the end iterator that can be used in combination with begin.
  AllocEntryList::iterator end() { return allocs.end(); }

  /// Registers a new allocation entry.
  void registerAlloc(const AllocEntry &entry) { allocs.push_back(entry); }

private:
  /// Searches for and registers all supported allocation entries.
  void build(Operation *op);

private:
  /// Maps allocation nodes to their associated blocks.
  AllocEntryList allocs;
};

/// The base class for all BufferPlacement transformations.
class BufferPlacementTransformationBase {
public:
  using ValueSetT = BufferPlacementAliasAnalysis::ValueSetT;

  /// Finds a common dominator for the given value while taking the positions
  /// of the values in the value set into account. It supports dominator and
  /// post-dominator analyses via template arguments.
  template <typename DominatorT>
  static Block *findCommonDominator(Value value, const ValueSetT &values,
                                    const DominatorT &doms) {
    // Start with the current block the value is defined in.
    Block *dom = value.getParentBlock();
    // Iterate over all aliases and their uses to find a safe placement block
    // according to the given dominator information.
    for (Value childValue : values) {
      for (Operation *user : childValue.getUsers()) {
        // Move upwards in the dominator tree to find an appropriate
        // dominator block that takes the current use into account.
        dom = doms.findNearestCommonDominator(dom, user->getBlock());
      }
      // Take values without any users into account.
      dom = doms.findNearestCommonDominator(dom, childValue.getParentBlock());
    }
    return dom;
  }

  /// Returns true if the given operation represents a loop by testing whether
  /// it implements the `LoopLikeOpInterface` or the `RegionBranchOpInterface`.
  /// In the case of a `RegionBranchOpInterface`, it checks all region-based
  /// control-flow edges for cycles.
  static bool isLoop(Operation *op);

  /// Constructs a new operation base using the given root operation.
  BufferPlacementTransformationBase(Operation *op);

protected:
  /// Alias information that can be updated during the insertion of copies.
  BufferPlacementAliasAnalysis aliases;

  /// Stores all internally managed allocations.
  BufferPlacementAllocs allocs;

  /// The underlying liveness analysis to compute fine grained information
  /// about alloc and dealloc positions.
  Liveness liveness;
};

} // end namespace mlir

#endif // MLIR_TRANSFORMS_BUFFERIZE_H
