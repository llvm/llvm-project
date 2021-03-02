//===- DialectConversion.h - MLIR dialect conversion pass -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a generic pass for converting between MLIR dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_DIALECTCONVERSION_H_
#define MLIR_TRANSFORMS_DIALECTCONVERSION_H_

#include "mlir/Rewrite/FrozenRewritePatternList.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {

// Forward declarations.
class Block;
class ConversionPatternRewriter;
class FuncOp;
class MLIRContext;
class Operation;
class Type;
class Value;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Type conversion class. Specific conversions and materializations can be
/// registered using addConversion and addMaterialization, respectively.
class TypeConverter {
public:
  /// This class provides all of the information necessary to convert a type
  /// signature.
  class SignatureConversion {
  public:
    SignatureConversion(unsigned numOrigInputs)
        : remappedInputs(numOrigInputs) {}

    /// This struct represents a range of new types or a single value that
    /// remaps an existing signature input.
    struct InputMapping {
      size_t inputNo, size;
      Value replacementValue;
    };

    /// Return the argument types for the new signature.
    ArrayRef<Type> getConvertedTypes() const { return argTypes; }

    /// Get the input mapping for the given argument.
    Optional<InputMapping> getInputMapping(unsigned input) const {
      return remappedInputs[input];
    }

    //===------------------------------------------------------------------===//
    // Conversion Hooks
    //===------------------------------------------------------------------===//

    /// Remap an input of the original signature with a new set of types. The
    /// new types are appended to the new signature conversion.
    void addInputs(unsigned origInputNo, ArrayRef<Type> types);

    /// Append new input types to the signature conversion, this should only be
    /// used if the new types are not intended to remap an existing input.
    void addInputs(ArrayRef<Type> types);

    /// Remap an input of the original signature to another `replacement`
    /// value. This drops the original argument.
    void remapInput(unsigned origInputNo, Value replacement);

  private:
    /// Remap an input of the original signature with a range of types in the
    /// new signature.
    void remapInput(unsigned origInputNo, unsigned newInputNo,
                    unsigned newInputCount = 1);

    /// The remapping information for each of the original arguments.
    SmallVector<Optional<InputMapping>, 4> remappedInputs;

    /// The set of new argument types.
    SmallVector<Type, 4> argTypes;
  };

  /// Register a conversion function. A conversion function must be convertible
  /// to any of the following forms(where `T` is a class derived from `Type`:
  ///   * Optional<Type>(T)
  ///     - This form represents a 1-1 type conversion. It should return nullptr
  ///       or `llvm::None` to signify failure. If `llvm::None` is returned, the
  ///       converter is allowed to try another conversion function to perform
  ///       the conversion.
  ///   * Optional<LogicalResult>(T, SmallVectorImpl<Type> &)
  ///     - This form represents a 1-N type conversion. It should return
  ///       `failure` or `llvm::None` to signify a failed conversion. If the new
  ///       set of types is empty, the type is removed and any usages of the
  ///       existing value are expected to be removed during conversion. If
  ///       `llvm::None` is returned, the converter is allowed to try another
  ///       conversion function to perform the conversion.
  /// Note: When attempting to convert a type, e.g. via 'convertType', the
  ///       mostly recently added conversions will be invoked first.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<0>>
  void addConversion(FnT &&callback) {
    registerConversion(wrapCallback<T>(std::forward<FnT>(callback)));
  }

  /// Register a materialization function, which must be convertible to the
  /// following form:
  ///   `Optional<Value>(OpBuilder &, T, ValueRange, Location)`,
  /// where `T` is any subclass of `Type`. This function is responsible for
  /// creating an operation, using the OpBuilder and Location provided, that
  /// "casts" a range of values into a single value of the given type `T`. It
  /// must return a Value of the converted type on success, an `llvm::None` if
  /// it failed but other materialization can be attempted, and `nullptr` on
  /// unrecoverable failure. It will only be called for (sub)types of `T`.
  /// Materialization functions must be provided when a type conversion
  /// results in more than one type, or if a type conversion may persist after
  /// the conversion has finished.
  ///
  /// This method registers a materialization that will be called when
  /// converting an illegal block argument type, to a legal type.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addArgumentMaterialization(FnT &&callback) {
    argumentMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }
  /// This method registers a materialization that will be called when
  /// converting a legal type to an illegal source type. This is used when
  /// conversions to an illegal type must persist beyond the main conversion.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addSourceMaterialization(FnT &&callback) {
    sourceMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }
  /// This method registers a materialization that will be called when
  /// converting type from an illegal, or source, type to a legal type.
  template <typename FnT, typename T = typename llvm::function_traits<
                              std::decay_t<FnT>>::template arg_t<1>>
  void addTargetMaterialization(FnT &&callback) {
    targetMaterializations.emplace_back(
        wrapMaterialization<T>(std::forward<FnT>(callback)));
  }

  /// Convert the given type. This function should return failure if no valid
  /// conversion exists, success otherwise. If the new set of types is empty,
  /// the type is removed and any usages of the existing value are expected to
  /// be removed during conversion.
  LogicalResult convertType(Type t, SmallVectorImpl<Type> &results);

  /// This hook simplifies defining 1-1 type conversions. This function returns
  /// the type to convert to on success, and a null type on failure.
  Type convertType(Type t);

  /// Convert the given set of types, filling 'results' as necessary. This
  /// returns failure if the conversion of any of the types fails, success
  /// otherwise.
  LogicalResult convertTypes(ArrayRef<Type> types,
                             SmallVectorImpl<Type> &results);

  /// Return true if the given type is legal for this type converter, i.e. the
  /// type converts to itself.
  bool isLegal(Type type);
  /// Return true if all of the given types are legal for this type converter.
  template <typename RangeT>
  std::enable_if_t<!std::is_convertible<RangeT, Type>::value &&
                       !std::is_convertible<RangeT, Operation *>::value,
                   bool>
  isLegal(RangeT &&range) {
    return llvm::all_of(range, [this](Type type) { return isLegal(type); });
  }
  /// Return true if the given operation has legal operand and result types.
  bool isLegal(Operation *op);

  /// Return true if the types of block arguments within the region are legal.
  bool isLegal(Region *region);

  /// Return true if the inputs and outputs of the given function type are
  /// legal.
  bool isSignatureLegal(FunctionType ty);

  /// This method allows for converting a specific argument of a signature. It
  /// takes as inputs the original argument input number, type.
  /// On success, it populates 'result' with any new mappings.
  LogicalResult convertSignatureArg(unsigned inputNo, Type type,
                                    SignatureConversion &result);
  LogicalResult convertSignatureArgs(TypeRange types,
                                     SignatureConversion &result,
                                     unsigned origInputOffset = 0);

  /// This function converts the type signature of the given block, by invoking
  /// 'convertSignatureArg' for each argument. This function should return a
  /// valid conversion for the signature on success, None otherwise.
  Optional<SignatureConversion> convertBlockSignature(Block *block);

  /// Materialize a conversion from a set of types into one result type by
  /// generating a cast sequence of some kind. See the respective
  /// `add*Materialization` for more information on the context for these
  /// methods.
  Value materializeArgumentConversion(OpBuilder &builder, Location loc,
                                      Type resultType, ValueRange inputs) {
    return materializeConversion(argumentMaterializations, builder, loc,
                                 resultType, inputs);
  }
  Value materializeSourceConversion(OpBuilder &builder, Location loc,
                                    Type resultType, ValueRange inputs) {
    return materializeConversion(sourceMaterializations, builder, loc,
                                 resultType, inputs);
  }
  Value materializeTargetConversion(OpBuilder &builder, Location loc,
                                    Type resultType, ValueRange inputs) {
    return materializeConversion(targetMaterializations, builder, loc,
                                 resultType, inputs);
  }

private:
  /// The signature of the callback used to convert a type. If the new set of
  /// types is empty, the type is removed and any usages of the existing value
  /// are expected to be removed during conversion.
  using ConversionCallbackFn =
      std::function<Optional<LogicalResult>(Type, SmallVectorImpl<Type> &)>;

  /// The signature of the callback used to materialize a conversion.
  using MaterializationCallbackFn =
      std::function<Optional<Value>(OpBuilder &, Type, ValueRange, Location)>;

  /// Attempt to materialize a conversion using one of the provided
  /// materialization functions.
  Value materializeConversion(
      MutableArrayRef<MaterializationCallbackFn> materializations,
      OpBuilder &builder, Location loc, Type resultType, ValueRange inputs);

  /// Generate a wrapper for the given callback. This allows for accepting
  /// different callback forms, that all compose into a single version.
  /// With callback of form: `Optional<Type>(T)`
  template <typename T, typename FnT>
  std::enable_if_t<llvm::is_invocable<FnT, T>::value, ConversionCallbackFn>
  wrapCallback(FnT &&callback) {
    return wrapCallback<T>([callback = std::forward<FnT>(callback)](
                               T type, SmallVectorImpl<Type> &results) {
      if (Optional<Type> resultOpt = callback(type)) {
        bool wasSuccess = static_cast<bool>(resultOpt.getValue());
        if (wasSuccess)
          results.push_back(resultOpt.getValue());
        return Optional<LogicalResult>(success(wasSuccess));
      }
      return Optional<LogicalResult>();
    });
  }
  /// With callback of form: `Optional<LogicalResult>(T, SmallVectorImpl<> &)`
  template <typename T, typename FnT>
  std::enable_if_t<!llvm::is_invocable<FnT, T>::value, ConversionCallbackFn>
  wrapCallback(FnT &&callback) {
    return [callback = std::forward<FnT>(callback)](
               Type type,
               SmallVectorImpl<Type> &results) -> Optional<LogicalResult> {
      T derivedType = type.dyn_cast<T>();
      if (!derivedType)
        return llvm::None;
      return callback(derivedType, results);
    };
  }

  /// Register a type conversion.
  void registerConversion(ConversionCallbackFn callback) {
    conversions.emplace_back(std::move(callback));
    cachedDirectConversions.clear();
    cachedMultiConversions.clear();
  }

  /// Generate a wrapper for the given materialization callback. The callback
  /// may take any subclass of `Type` and the wrapper will check for the target
  /// type to be of the expected class before calling the callback.
  template <typename T, typename FnT>
  MaterializationCallbackFn wrapMaterialization(FnT &&callback) {
    return [callback = std::forward<FnT>(callback)](
               OpBuilder &builder, Type resultType, ValueRange inputs,
               Location loc) -> Optional<Value> {
      if (T derivedType = resultType.dyn_cast<T>())
        return callback(builder, derivedType, inputs, loc);
      return llvm::None;
    };
  }

  /// The set of registered conversion functions.
  SmallVector<ConversionCallbackFn, 4> conversions;

  /// The list of registered materialization functions.
  SmallVector<MaterializationCallbackFn, 2> argumentMaterializations;
  SmallVector<MaterializationCallbackFn, 2> sourceMaterializations;
  SmallVector<MaterializationCallbackFn, 2> targetMaterializations;

  /// A set of cached conversions to avoid recomputing in the common case.
  /// Direct 1-1 conversions are the most common, so this cache stores the
  /// successful 1-1 conversions as well as all failed conversions.
  DenseMap<Type, Type> cachedDirectConversions;
  /// This cache stores the successful 1->N conversions, where N != 1.
  DenseMap<Type, SmallVector<Type, 2>> cachedMultiConversions;
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Base class for the conversion patterns. This pattern class enables type
/// conversions, and other uses specific to the conversion framework. As such,
/// patterns of this type can only be used with the 'apply*' methods below.
class ConversionPattern : public RewritePattern {
public:
  /// Hook for derived classes to implement rewriting. `op` is the (first)
  /// operation matched by the pattern, `operands` is a list of the rewritten
  /// operand values that are passed to `op`, `rewriter` can be used to emit the
  /// new operations. This function should not fail. If some specific cases of
  /// the operation are not supported, these cases should not be matched.
  virtual void rewrite(Operation *op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite");
  }

  /// Hook for derived classes to implement combined matching and rewriting.
  virtual LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, operands, rewriter);
    return success();
  }

  /// Attempt to match and rewrite the IR root at the specified operation.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final;

  /// Return the type converter held by this pattern, or nullptr if the pattern
  /// does not require type conversion.
  TypeConverter *getTypeConverter() const { return typeConverter; }

  template <typename ConverterTy>
  std::enable_if_t<std::is_base_of<TypeConverter, ConverterTy>::value,
                   ConverterTy *>
  getTypeConverter() const {
    return static_cast<ConverterTy *>(typeConverter);
  }

protected:
  /// See `RewritePattern::RewritePattern` for information on the other
  /// available constructors.
  using RewritePattern::RewritePattern;
  /// Construct a conversion pattern that matches an operation with the given
  /// root name. This constructor allows for providing a type converter to use
  /// within the pattern.
  ConversionPattern(StringRef rootName, PatternBenefit benefit,
                    TypeConverter &typeConverter, MLIRContext *ctx)
      : RewritePattern(rootName, benefit, ctx), typeConverter(&typeConverter) {}
  /// Construct a conversion pattern that matches any operation type. This
  /// constructor allows for providing a type converter to use within the
  /// pattern. `MatchAnyOpTypeTag` is just a tag to ensure that the "match any"
  /// behavior is what the user actually desired, `MatchAnyOpTypeTag()` should
  /// always be supplied here.
  ConversionPattern(PatternBenefit benefit, TypeConverter &typeConverter,
                    MatchAnyOpTypeTag tag)
      : RewritePattern(benefit, tag), typeConverter(&typeConverter) {}

protected:
  /// An optional type converter for use by this pattern.
  TypeConverter *typeConverter = nullptr;

private:
  using RewritePattern::rewrite;
};

/// OpConversionPattern is a wrapper around ConversionPattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
template <typename SourceOp>
struct OpConversionPattern : public ConversionPattern {
  OpConversionPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(SourceOp::getOperationName(), benefit, context) {}
  OpConversionPattern(TypeConverter &typeConverter, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : ConversionPattern(SourceOp::getOperationName(), benefit, typeConverter,
                          context) {}

  /// Wrappers around the ConversionPattern methods that pass the derived op
  /// type.
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    rewrite(cast<SourceOp>(op), operands, rewriter);
  }
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(cast<SourceOp>(op), operands, rewriter);
  }

  // TODO: Use OperandAdaptor when it supports access to unnamed operands.

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual void rewrite(SourceOp op, ArrayRef<Value> operands,
                       ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }

  virtual LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, operands, rewriter);
    return success();
  }

private:
  using ConversionPattern::matchAndRewrite;
};

/// Add a pattern to the given pattern list to convert the signature of a
/// FunctionLike op with the given type converter. This only supports
/// FunctionLike ops which use FunctionType to represent their type.
void populateFunctionLikeTypeConversionPattern(
    StringRef functionLikeOpName, OwningRewritePatternList &patterns,
    MLIRContext *ctx, TypeConverter &converter);

template <typename FuncOpT>
void populateFunctionLikeTypeConversionPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx,
    TypeConverter &converter) {
  populateFunctionLikeTypeConversionPattern(FuncOpT::getOperationName(),
                                            patterns, ctx, converter);
}

/// Add a pattern to the given pattern list to convert the signature of a FuncOp
/// with the given type converter.
void populateFuncOpTypeConversionPattern(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx,
                                         TypeConverter &converter);

//===----------------------------------------------------------------------===//
// Conversion PatternRewriter
//===----------------------------------------------------------------------===//

namespace detail {
struct ConversionPatternRewriterImpl;
} // end namespace detail

/// This class implements a pattern rewriter for use with ConversionPatterns. It
/// extends the base PatternRewriter and provides special conversion specific
/// hooks.
class ConversionPatternRewriter final : public PatternRewriter {
public:
  ConversionPatternRewriter(MLIRContext *ctx);
  ~ConversionPatternRewriter() override;

  /// Apply a signature conversion to the entry block of the given region. This
  /// replaces the entry block with a new block containing the updated
  /// signature. The new entry block to the region is returned for convenience.
  Block *
  applySignatureConversion(Region *region,
                           TypeConverter::SignatureConversion &conversion);

  /// Convert the types of block arguments within the given region. This
  /// replaces each block with a new block containing the updated signature. The
  /// entry block may have a special conversion if `entryConversion` is
  /// provided. On success, the new entry block to the region is returned for
  /// convenience. Otherwise, failure is returned.
  FailureOr<Block *> convertRegionTypes(
      Region *region, TypeConverter &converter,
      TypeConverter::SignatureConversion *entryConversion = nullptr);

  /// Convert the types of block arguments within the given region except for
  /// the entry region. This replaces each non-entry block with a new block
  /// containing the updated signature.
  LogicalResult convertNonEntryRegionTypes(Region *region,
                                           TypeConverter &converter);

  /// Replace all the uses of the block argument `from` with value `to`.
  void replaceUsesOfBlockArgument(BlockArgument from, Value to);

  /// Return the converted value that replaces 'key'. Return 'key' if there is
  /// no such a converted value.
  Value getRemappedValue(Value key);

  //===--------------------------------------------------------------------===//
  // PatternRewriter Hooks
  //===--------------------------------------------------------------------===//

  /// PatternRewriter hook for replacing the results of an operation when the
  /// given functor returns true.
  void replaceOpWithIf(
      Operation *op, ValueRange newValues, bool *allUsesReplaced,
      llvm::unique_function<bool(OpOperand &) const> functor) override;

  /// PatternRewriter hook for replacing the results of an operation.
  void replaceOp(Operation *op, ValueRange newValues) override;
  using PatternRewriter::replaceOp;

  /// PatternRewriter hook for erasing a dead operation. The uses of this
  /// operation *must* be made dead by the end of the conversion process,
  /// otherwise an assert will be issued.
  void eraseOp(Operation *op) override;

  /// PatternRewriter hook for erase all operations in a block. This is not yet
  /// implemented for dialect conversion.
  void eraseBlock(Block *block) override;

  /// PatternRewriter hook creating a new block.
  void notifyBlockCreated(Block *block) override;

  /// PatternRewriter hook for splitting a block into two parts.
  Block *splitBlock(Block *block, Block::iterator before) override;

  /// PatternRewriter hook for merging a block into another.
  void mergeBlocks(Block *source, Block *dest, ValueRange argValues) override;

  /// PatternRewriter hook for moving blocks out of a region.
  void inlineRegionBefore(Region &region, Region &parent,
                          Region::iterator before) override;
  using PatternRewriter::inlineRegionBefore;

  /// PatternRewriter hook for cloning blocks of one region into another. The
  /// given region to clone *must* not have been modified as part of conversion
  /// yet, i.e. it must be within an operation that is either in the process of
  /// conversion, or has not yet been converted.
  void cloneRegionBefore(Region &region, Region &parent,
                         Region::iterator before,
                         BlockAndValueMapping &mapping) override;
  using PatternRewriter::cloneRegionBefore;

  /// PatternRewriter hook for inserting a new operation.
  void notifyOperationInserted(Operation *op) override;

  /// PatternRewriter hook for updating the root operation in-place.
  /// Note: These methods only track updates to the top-level operation itself,
  /// and not nested regions. Updates to regions will still require notification
  /// through other more specific hooks above.
  void startRootUpdate(Operation *op) override;

  /// PatternRewriter hook for updating the root operation in-place.
  void finalizeRootUpdate(Operation *op) override;

  /// PatternRewriter hook for updating the root operation in-place.
  void cancelRootUpdate(Operation *op) override;

  /// PatternRewriter hook for notifying match failure reasons.
  LogicalResult
  notifyMatchFailure(Operation *op,
                     function_ref<void(Diagnostic &)> reasonCallback) override;
  using PatternRewriter::notifyMatchFailure;

  /// Return a reference to the internal implementation.
  detail::ConversionPatternRewriterImpl &getImpl();

private:
  std::unique_ptr<detail::ConversionPatternRewriterImpl> impl;
};

//===----------------------------------------------------------------------===//
// ConversionTarget
//===----------------------------------------------------------------------===//

/// This class describes a specific conversion target.
class ConversionTarget {
public:
  /// This enumeration corresponds to the specific action to take when
  /// considering an operation legal for this conversion target.
  enum class LegalizationAction {
    /// The target supports this operation.
    Legal,

    /// This operation has dynamic legalization constraints that must be checked
    /// by the target.
    Dynamic,

    /// The target explicitly does not support this operation.
    Illegal,
  };

  /// A structure containing additional information describing a specific legal
  /// operation instance.
  struct LegalOpDetails {
    /// A flag that indicates if this operation is 'recursively' legal. This
    /// means that if an operation is legal, either statically or dynamically,
    /// all of the operations nested within are also considered legal.
    bool isRecursivelyLegal = false;
  };

  /// The signature of the callback used to determine if an operation is
  /// dynamically legal on the target.
  using DynamicLegalityCallbackFn = std::function<bool(Operation *)>;

  ConversionTarget(MLIRContext &ctx)
      : unknownOpsDynamicallyLegal(false), ctx(ctx) {}
  virtual ~ConversionTarget() = default;

  //===--------------------------------------------------------------------===//
  // Legality Registration
  //===--------------------------------------------------------------------===//

  /// Register a legality action for the given operation.
  void setOpAction(OperationName op, LegalizationAction action);
  template <typename OpT> void setOpAction(LegalizationAction action) {
    setOpAction(OperationName(OpT::getOperationName(), &ctx), action);
  }

  /// Register the given operations as legal.
  template <typename OpT> void addLegalOp() {
    setOpAction<OpT>(LegalizationAction::Legal);
  }
  template <typename OpT, typename OpT2, typename... OpTs> void addLegalOp() {
    addLegalOp<OpT>();
    addLegalOp<OpT2, OpTs...>();
  }

  /// Register the given operation as dynamically legal, i.e. requiring custom
  /// handling by the target via 'isDynamicallyLegal'.
  template <typename OpT> void addDynamicallyLegalOp() {
    setOpAction<OpT>(LegalizationAction::Dynamic);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void addDynamicallyLegalOp() {
    addDynamicallyLegalOp<OpT>();
    addDynamicallyLegalOp<OpT2, OpTs...>();
  }

  /// Register the given operation as dynamically legal and set the dynamic
  /// legalization callback to the one provided.
  template <typename OpT>
  void addDynamicallyLegalOp(const DynamicLegalityCallbackFn &callback) {
    OperationName opName(OpT::getOperationName(), &ctx);
    setOpAction(opName, LegalizationAction::Dynamic);
    setLegalityCallback(opName, callback);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void addDynamicallyLegalOp(const DynamicLegalityCallbackFn &callback) {
    addDynamicallyLegalOp<OpT>(callback);
    addDynamicallyLegalOp<OpT2, OpTs...>(callback);
  }
  template <typename OpT, class Callable>
  typename std::enable_if<
      !llvm::is_invocable<Callable, Operation *>::value>::type
  addDynamicallyLegalOp(Callable &&callback) {
    addDynamicallyLegalOp<OpT>(
        [=](Operation *op) { return callback(cast<OpT>(op)); });
  }

  /// Register the given operation as illegal, i.e. this operation is known to
  /// not be supported by this target.
  template <typename OpT> void addIllegalOp() {
    setOpAction<OpT>(LegalizationAction::Illegal);
  }
  template <typename OpT, typename OpT2, typename... OpTs> void addIllegalOp() {
    addIllegalOp<OpT>();
    addIllegalOp<OpT2, OpTs...>();
  }

  /// Mark an operation, that *must* have either been set as `Legal` or
  /// `DynamicallyLegal`, as being recursively legal. This means that in
  /// addition to the operation itself, all of the operations nested within are
  /// also considered legal. An optional dynamic legality callback may be
  /// provided to mark subsets of legal instances as recursively legal.
  template <typename OpT>
  void markOpRecursivelyLegal(const DynamicLegalityCallbackFn &callback = {}) {
    OperationName opName(OpT::getOperationName(), &ctx);
    markOpRecursivelyLegal(opName, callback);
  }
  template <typename OpT, typename OpT2, typename... OpTs>
  void markOpRecursivelyLegal(const DynamicLegalityCallbackFn &callback = {}) {
    markOpRecursivelyLegal<OpT>(callback);
    markOpRecursivelyLegal<OpT2, OpTs...>(callback);
  }
  template <typename OpT, class Callable>
  typename std::enable_if<
      !llvm::is_invocable<Callable, Operation *>::value>::type
  markOpRecursivelyLegal(Callable &&callback) {
    markOpRecursivelyLegal<OpT>(
        [=](Operation *op) { return callback(cast<OpT>(op)); });
  }

  /// Register a legality action for the given dialects.
  void setDialectAction(ArrayRef<StringRef> dialectNames,
                        LegalizationAction action);

  /// Register the operations of the given dialects as legal.
  template <typename... Names>
  void addLegalDialect(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Legal);
  }
  template <typename... Args> void addLegalDialect() {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Legal);
  }

  /// Register the operations of the given dialects as dynamically legal, i.e.
  /// requiring custom handling by the target via 'isDynamicallyLegal'.
  template <typename... Names>
  void addDynamicallyLegalDialect(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Dynamic);
  }
  template <typename... Args>
  void addDynamicallyLegalDialect(
      Optional<DynamicLegalityCallbackFn> callback = llvm::None) {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Dynamic);
    if (callback)
      setLegalityCallback(dialectNames, *callback);
  }
  template <typename... Args>
  void addDynamicallyLegalDialect(DynamicLegalityCallbackFn callback) {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Dynamic);
    setLegalityCallback(dialectNames, callback);
  }

  /// Register unknown operations as dynamically legal. For operations(and
  /// dialects) that do not have a set legalization action, treat them as
  /// dynamically legal and invoke the given callback if valid or
  /// 'isDynamicallyLegal'.
  void markUnknownOpDynamicallyLegal(const DynamicLegalityCallbackFn &fn) {
    unknownOpsDynamicallyLegal = true;
    unknownLegalityFn = fn;
  }
  void markUnknownOpDynamicallyLegal() { unknownOpsDynamicallyLegal = true; }

  /// Register the operations of the given dialects as illegal, i.e.
  /// operations of this dialect are not supported by the target.
  template <typename... Names>
  void addIllegalDialect(StringRef name, Names... names) {
    SmallVector<StringRef, 2> dialectNames({name, names...});
    setDialectAction(dialectNames, LegalizationAction::Illegal);
  }
  template <typename... Args> void addIllegalDialect() {
    SmallVector<StringRef, 2> dialectNames({Args::getDialectNamespace()...});
    setDialectAction(dialectNames, LegalizationAction::Illegal);
  }

  //===--------------------------------------------------------------------===//
  // Legality Querying
  //===--------------------------------------------------------------------===//

  /// Get the legality action for the given operation.
  Optional<LegalizationAction> getOpAction(OperationName op) const;

  /// If the given operation instance is legal on this target, a structure
  /// containing legality information is returned. If the operation is not
  /// legal, None is returned.
  Optional<LegalOpDetails> isLegal(Operation *op) const;

protected:
  /// Runs a custom legalization query for the given operation. This should
  /// return true if the given operation is legal, otherwise false.
  virtual bool isDynamicallyLegal(Operation *op) const {
    llvm_unreachable(
        "targets with custom legalization must override 'isDynamicallyLegal'");
  }

private:
  /// Set the dynamic legality callback for the given operation.
  void setLegalityCallback(OperationName name,
                           const DynamicLegalityCallbackFn &callback);

  /// Set the dynamic legality callback for the given dialects.
  void setLegalityCallback(ArrayRef<StringRef> dialects,
                           const DynamicLegalityCallbackFn &callback);

  /// Set the recursive legality callback for the given operation and mark the
  /// operation as recursively legal.
  void markOpRecursivelyLegal(OperationName name,
                              const DynamicLegalityCallbackFn &callback);

  /// The set of information that configures the legalization of an operation.
  struct LegalizationInfo {
    /// The legality action this operation was given.
    LegalizationAction action;

    /// If some legal instances of this operation may also be recursively legal.
    bool isRecursivelyLegal;

    /// The legality callback if this operation is dynamically legal.
    Optional<DynamicLegalityCallbackFn> legalityFn;
  };

  /// Get the legalization information for the given operation.
  Optional<LegalizationInfo> getOpInfo(OperationName op) const;

  /// A deterministic mapping of operation name and its respective legality
  /// information.
  llvm::MapVector<OperationName, LegalizationInfo> legalOperations;

  /// A set of legality callbacks for given operation names that are used to
  /// check if an operation instance is recursively legal.
  DenseMap<OperationName, DynamicLegalityCallbackFn> opRecursiveLegalityFns;

  /// A deterministic mapping of dialect name to the specific legality action to
  /// take.
  llvm::StringMap<LegalizationAction> legalDialects;

  /// A set of dynamic legality callbacks for given dialect names.
  llvm::StringMap<DynamicLegalityCallbackFn> dialectLegalityFns;

  /// An optional legality callback for unknown operations.
  Optional<DynamicLegalityCallbackFn> unknownLegalityFn;

  /// Flag indicating if unknown operations should be treated as dynamically
  /// legal.
  bool unknownOpsDynamicallyLegal;

  /// The current context this target applies to.
  MLIRContext &ctx;
};

//===----------------------------------------------------------------------===//
// Op Conversion Entry Points
//===----------------------------------------------------------------------===//

/// Below we define several entry points for operation conversion. It is
/// important to note that the patterns provided to the conversion framework may
/// have additional constraints. See the `PatternRewriter Hooks` section of the
/// ConversionPatternRewriter, to see what additional constraints are imposed on
/// the use of the PatternRewriter.

/// Apply a partial conversion on the given operations and all nested
/// operations. This method converts as many operations to the target as
/// possible, ignoring operations that failed to legalize. This method only
/// returns failure if there ops explicitly marked as illegal. If an
/// `unconvertedOps` set is provided, all operations that are found not to be
/// legalizable to the given `target` are placed within that set. (Note that if
/// there is an op explicitly marked as illegal, the conversion terminates and
/// the `unconvertedOps` set will not necessarily be complete.)
LogicalResult
applyPartialConversion(ArrayRef<Operation *> ops, ConversionTarget &target,
                       const FrozenRewritePatternList &patterns,
                       DenseSet<Operation *> *unconvertedOps = nullptr);
LogicalResult
applyPartialConversion(Operation *op, ConversionTarget &target,
                       const FrozenRewritePatternList &patterns,
                       DenseSet<Operation *> *unconvertedOps = nullptr);

/// Apply a complete conversion on the given operations, and all nested
/// operations. This method returns failure if the conversion of any operation
/// fails, or if there are unreachable blocks in any of the regions nested
/// within 'ops'.
LogicalResult applyFullConversion(ArrayRef<Operation *> ops,
                                  ConversionTarget &target,
                                  const FrozenRewritePatternList &patterns);
LogicalResult applyFullConversion(Operation *op, ConversionTarget &target,
                                  const FrozenRewritePatternList &patterns);

/// Apply an analysis conversion on the given operations, and all nested
/// operations. This method analyzes which operations would be successfully
/// converted to the target if a conversion was applied. All operations that
/// were found to be legalizable to the given 'target' are placed within the
/// provided 'convertedOps' set; note that no actual rewrites are applied to the
/// operations on success and only pre-existing operations are added to the set.
/// This method only returns failure if there are unreachable blocks in any of
/// the regions nested within 'ops'.
LogicalResult applyAnalysisConversion(ArrayRef<Operation *> ops,
                                      ConversionTarget &target,
                                      const FrozenRewritePatternList &patterns,
                                      DenseSet<Operation *> &convertedOps);
LogicalResult applyAnalysisConversion(Operation *op, ConversionTarget &target,
                                      const FrozenRewritePatternList &patterns,
                                      DenseSet<Operation *> &convertedOps);
} // end namespace mlir

#endif // MLIR_TRANSFORMS_DIALECTCONVERSION_H_
