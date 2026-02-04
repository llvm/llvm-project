//===-- mlir-c/Rewrite.h - Helpers for C API to Rewrites ----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the registration and creation method for
// rewrite patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_REWRITE_H
#define MLIR_C_REWRITE_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Config/mlir-config.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations (see mlir-c/IR.h for more details).
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirRewriterBase, void);
DEFINE_C_API_STRUCT(MlirFrozenRewritePatternSet, void);
DEFINE_C_API_STRUCT(MlirGreedyRewriteDriverConfig, void);

/// Greedy rewrite strictness levels.
typedef enum {
  /// No restrictions wrt. which ops are processed.
  MLIR_GREEDY_REWRITE_STRICTNESS_ANY_OP,
  /// Only pre-existing and newly created ops are processed.
  MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS,
  /// Only pre-existing ops are processed.
  MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS
} MlirGreedyRewriteStrictness;

/// Greedy simplify region levels.
typedef enum {
  /// Disable region control-flow simplification.
  MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED,
  /// Run the normal simplification (e.g. dead args elimination).
  MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL,
  /// Run extra simplifications (e.g. block merging).
  MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE
} MlirGreedySimplifyRegionLevel;
DEFINE_C_API_STRUCT(MlirRewritePatternSet, void);
DEFINE_C_API_STRUCT(MlirPatternRewriter, void);
DEFINE_C_API_STRUCT(MlirRewritePattern, const void);
DEFINE_C_API_STRUCT(MlirConversionTarget, void);
DEFINE_C_API_STRUCT(MlirConversionPattern, const void);
DEFINE_C_API_STRUCT(MlirTypeConverter, void);
DEFINE_C_API_STRUCT(MlirConversionPatternRewriter, void);
DEFINE_C_API_STRUCT(MlirConversionConfig, void);

//===----------------------------------------------------------------------===//
/// RewriterBase API inherited from OpBuilder
//===----------------------------------------------------------------------===//

/// Get the MLIR context referenced by the rewriter.
MLIR_CAPI_EXPORTED MlirContext
mlirRewriterBaseGetContext(MlirRewriterBase rewriter);

//===----------------------------------------------------------------------===//
/// Insertion points methods
//===----------------------------------------------------------------------===//

// These do not include functions using Block::iterator or Region::iterator, as
// they are not exposed by the C API yet. Similarly for methods using
// `InsertPoint` directly.

/// Reset the insertion point to no location.  Creating an operation without a
/// set insertion point is an error, but this can still be useful when the
/// current insertion point a builder refers to is being removed.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseClearInsertionPoint(MlirRewriterBase rewriter);

/// Sets the insertion point to the specified operation, which will cause
/// subsequent insertions to go right before it.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseSetInsertionPointBefore(MlirRewriterBase rewriter,
                                        MlirOperation op);

/// Sets the insertion point to the node after the specified operation, which
/// will cause subsequent insertions to go right after it.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseSetInsertionPointAfter(MlirRewriterBase rewriter,
                                       MlirOperation op);

/// Sets the insertion point to the node after the specified value. If value
/// has a defining operation, sets the insertion point to the node after such
/// defining operation. This will cause subsequent insertions to go right
/// after it. Otherwise, value is a BlockArgument. Sets the insertion point to
/// the start of its block.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseSetInsertionPointAfterValue(MlirRewriterBase rewriter,
                                            MlirValue value);

/// Sets the insertion point to the start of the specified block.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseSetInsertionPointToStart(MlirRewriterBase rewriter,
                                         MlirBlock block);

/// Sets the insertion point to the end of the specified block.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseSetInsertionPointToEnd(MlirRewriterBase rewriter,
                                       MlirBlock block);

/// Return the block the current insertion point belongs to.  Note that the
/// insertion point is not necessarily the end of the block.
MLIR_CAPI_EXPORTED MlirBlock
mlirRewriterBaseGetInsertionBlock(MlirRewriterBase rewriter);

/// Returns the current block of the rewriter.
MLIR_CAPI_EXPORTED MlirBlock
mlirRewriterBaseGetBlock(MlirRewriterBase rewriter);

/// Returns the operation right after the current insertion point
/// of the rewriter. A null MlirOperation will be returned
// if the current insertion point is at the end of the block.
MLIR_CAPI_EXPORTED MlirOperation
mlirRewriterBaseGetOperationAfterInsertion(MlirRewriterBase rewriter);

//===----------------------------------------------------------------------===//
/// Block and operation creation/insertion/cloning
//===----------------------------------------------------------------------===//

// These functions do not include the IRMapper, as it is not yet exposed by the
// C API.

/// Add new block with 'argTypes' arguments and set the insertion point to the
/// end of it. The block is placed before 'insertBefore'. `locs` contains the
/// locations of the inserted arguments, and should match the size of
/// `argTypes`.
MLIR_CAPI_EXPORTED MlirBlock mlirRewriterBaseCreateBlockBefore(
    MlirRewriterBase rewriter, MlirBlock insertBefore, intptr_t nArgTypes,
    MlirType const *argTypes, MlirLocation const *locations);

/// Insert the given operation at the current insertion point and return it.
MLIR_CAPI_EXPORTED MlirOperation
mlirRewriterBaseInsert(MlirRewriterBase rewriter, MlirOperation op);

/// Creates a deep copy of the specified operation.
MLIR_CAPI_EXPORTED MlirOperation
mlirRewriterBaseClone(MlirRewriterBase rewriter, MlirOperation op);

/// Creates a deep copy of this operation but keep the operation regions
/// empty.
MLIR_CAPI_EXPORTED MlirOperation mlirRewriterBaseCloneWithoutRegions(
    MlirRewriterBase rewriter, MlirOperation op);

/// Clone the blocks that belong to "region" before the given position in
/// another region "parent".
MLIR_CAPI_EXPORTED void
mlirRewriterBaseCloneRegionBefore(MlirRewriterBase rewriter, MlirRegion region,
                                  MlirBlock before);

//===----------------------------------------------------------------------===//
/// RewriterBase API
//===----------------------------------------------------------------------===//

/// Move the blocks that belong to "region" before the given position in
/// another region "parent". The two regions must be different. The caller
/// is responsible for creating or updating the operation transferring flow
/// of control to the region and passing it the correct block arguments.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseInlineRegionBefore(MlirRewriterBase rewriter, MlirRegion region,
                                   MlirBlock before);

/// Replace the results of the given (original) operation with the specified
/// list of values (replacements). The result types of the given op and the
/// replacements must match. The original op is erased.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseReplaceOpWithValues(MlirRewriterBase rewriter, MlirOperation op,
                                    intptr_t nValues, MlirValue const *values);

/// Replace the results of the given (original) operation with the specified
/// new op (replacement). The result types of the two ops must match. The
/// original op is erased.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseReplaceOpWithOperation(MlirRewriterBase rewriter,
                                       MlirOperation op, MlirOperation newOp);

/// Erases an operation that is known to have no uses.
MLIR_CAPI_EXPORTED void mlirRewriterBaseEraseOp(MlirRewriterBase rewriter,
                                                MlirOperation op);

/// Erases a block along with all operations inside it.
MLIR_CAPI_EXPORTED void mlirRewriterBaseEraseBlock(MlirRewriterBase rewriter,
                                                   MlirBlock block);

/// Inline the operations of block 'source' before the operation 'op'. The
/// source block will be deleted and must have no uses. 'argValues' is used to
/// replace the block arguments of 'source'
///
/// The source block must have no successors. Otherwise, the resulting IR
/// would have unreachable operations.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseInlineBlockBefore(MlirRewriterBase rewriter, MlirBlock source,
                                  MlirOperation op, intptr_t nArgValues,
                                  MlirValue const *argValues);

/// Inline the operations of block 'source' into the end of block 'dest'. The
/// source block will be deleted and must have no uses. 'argValues' is used to
/// replace the block arguments of 'source'
///
/// The dest block must have no successors. Otherwise, the resulting IR would
/// have unreachable operation.
MLIR_CAPI_EXPORTED void mlirRewriterBaseMergeBlocks(MlirRewriterBase rewriter,
                                                    MlirBlock source,
                                                    MlirBlock dest,
                                                    intptr_t nArgValues,
                                                    MlirValue const *argValues);

/// Unlink this operation from its current block and insert it right before
/// `existingOp` which may be in the same or another block in the same
/// function.
MLIR_CAPI_EXPORTED void mlirRewriterBaseMoveOpBefore(MlirRewriterBase rewriter,
                                                     MlirOperation op,
                                                     MlirOperation existingOp);

/// Unlink this operation from its current block and insert it right after
/// `existingOp` which may be in the same or another block in the same
/// function.
MLIR_CAPI_EXPORTED void mlirRewriterBaseMoveOpAfter(MlirRewriterBase rewriter,
                                                    MlirOperation op,
                                                    MlirOperation existingOp);

/// Unlink this block and insert it right before `existingBlock`.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseMoveBlockBefore(MlirRewriterBase rewriter, MlirBlock block,
                                MlirBlock existingBlock);

/// This method is used to notify the rewriter that an in-place operation
/// modification is about to happen. A call to this function *must* be
/// followed by a call to either `finalizeOpModification` or
/// `cancelOpModification`. This is a minor efficiency win (it avoids creating
/// a new operation and removing the old one) but also often allows simpler
/// code in the client.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseStartOpModification(MlirRewriterBase rewriter,
                                    MlirOperation op);

/// This method is used to signal the end of an in-place modification of the
/// given operation. This can only be called on operations that were provided
/// to a call to `startOpModification`.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseFinalizeOpModification(MlirRewriterBase rewriter,
                                       MlirOperation op);

/// This method cancels a pending in-place modification. This can only be
/// called on operations that were provided to a call to
/// `startOpModification`.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseCancelOpModification(MlirRewriterBase rewriter,
                                     MlirOperation op);

/// Find uses of `from` and replace them with `to`. Also notify the listener
/// about every in-place op modification (for every use that was replaced).
MLIR_CAPI_EXPORTED void
mlirRewriterBaseReplaceAllUsesWith(MlirRewriterBase rewriter, MlirValue from,
                                   MlirValue to);

/// Find uses of `from` and replace them with `to`. Also notify the listener
/// about every in-place op modification (for every use that was replaced).
MLIR_CAPI_EXPORTED void mlirRewriterBaseReplaceAllValueRangeUsesWith(
    MlirRewriterBase rewriter, intptr_t nValues, MlirValue const *from,
    MlirValue const *to);

/// Find uses of `from` and replace them with `to`. Also notify the listener
/// about every in-place op modification (for every use that was replaced)
/// and that the `from` operation is about to be replaced.
MLIR_CAPI_EXPORTED void
mlirRewriterBaseReplaceAllOpUsesWithValueRange(MlirRewriterBase rewriter,
                                               MlirOperation from, intptr_t nTo,
                                               MlirValue const *to);

/// Find uses of `from` and replace them with `to`. Also notify the listener
/// about every in-place op modification (for every use that was replaced)
/// and that the `from` operation is about to be replaced.
MLIR_CAPI_EXPORTED void mlirRewriterBaseReplaceAllOpUsesWithOperation(
    MlirRewriterBase rewriter, MlirOperation from, MlirOperation to);

/// Find uses of `from` within `block` and replace them with `to`. Also notify
/// the listener about every in-place op modification (for every use that was
/// replaced). The optional `allUsesReplaced` flag is set to "true" if all
/// uses were replaced.
MLIR_CAPI_EXPORTED void mlirRewriterBaseReplaceOpUsesWithinBlock(
    MlirRewriterBase rewriter, MlirOperation op, intptr_t nNewValues,
    MlirValue const *newValues, MlirBlock block);

/// Find uses of `from` and replace them with `to` except if the user is
/// `exceptedUser`. Also notify the listener about every in-place op
/// modification (for every use that was replaced).
MLIR_CAPI_EXPORTED void
mlirRewriterBaseReplaceAllUsesExcept(MlirRewriterBase rewriter, MlirValue from,
                                     MlirValue to, MlirOperation exceptedUser);

//===----------------------------------------------------------------------===//
/// IRRewriter API
//===----------------------------------------------------------------------===//

/// Create an IRRewriter and transfer ownership to the caller.
MLIR_CAPI_EXPORTED MlirRewriterBase mlirIRRewriterCreate(MlirContext context);

/// Create an IRRewriter and transfer ownership to the caller. Additionally
/// set the insertion point before the operation.
MLIR_CAPI_EXPORTED MlirRewriterBase
mlirIRRewriterCreateFromOp(MlirOperation op);

/// Takes an IRRewriter owned by the caller and destroys it. It is the
/// responsibility of the user to only pass an IRRewriter class.
MLIR_CAPI_EXPORTED void mlirIRRewriterDestroy(MlirRewriterBase rewriter);

//===----------------------------------------------------------------------===//
/// FrozenRewritePatternSet API
//===----------------------------------------------------------------------===//

/// Freeze the given MlirRewritePatternSet to a MlirFrozenRewritePatternSet.
/// Note that the ownership of the input set is transferred into the frozen set
/// after this call.
MLIR_CAPI_EXPORTED MlirFrozenRewritePatternSet
mlirFreezeRewritePattern(MlirRewritePatternSet set);

/// Destroy the given MlirFrozenRewritePatternSet.
MLIR_CAPI_EXPORTED void
mlirFrozenRewritePatternSetDestroy(MlirFrozenRewritePatternSet set);

MLIR_CAPI_EXPORTED MlirLogicalResult mlirApplyPatternsAndFoldGreedilyWithOp(
    MlirOperation op, MlirFrozenRewritePatternSet patterns,
    MlirGreedyRewriteDriverConfig);

MLIR_CAPI_EXPORTED MlirLogicalResult mlirApplyPatternsAndFoldGreedily(
    MlirModule op, MlirFrozenRewritePatternSet patterns,
    MlirGreedyRewriteDriverConfig config);

//===----------------------------------------------------------------------===//
/// GreedyRewriteDriverConfig API
//===----------------------------------------------------------------------===//

/// Creates a greedy rewrite driver configuration with default settings.
MLIR_CAPI_EXPORTED MlirGreedyRewriteDriverConfig
mlirGreedyRewriteDriverConfigCreate(void);

/// Destroys a greedy rewrite driver configuration.
MLIR_CAPI_EXPORTED void
mlirGreedyRewriteDriverConfigDestroy(MlirGreedyRewriteDriverConfig config);

/// Sets the maximum number of iterations for the greedy rewrite driver.
/// Use -1 for no limit.
MLIR_CAPI_EXPORTED void mlirGreedyRewriteDriverConfigSetMaxIterations(
    MlirGreedyRewriteDriverConfig config, int64_t maxIterations);

/// Sets the maximum number of rewrites within an iteration.
/// Use -1 for no limit.
MLIR_CAPI_EXPORTED void mlirGreedyRewriteDriverConfigSetMaxNumRewrites(
    MlirGreedyRewriteDriverConfig config, int64_t maxNumRewrites);

/// Sets whether to use top-down traversal for the initial population of the
/// worklist.
MLIR_CAPI_EXPORTED void mlirGreedyRewriteDriverConfigSetUseTopDownTraversal(
    MlirGreedyRewriteDriverConfig config, bool useTopDownTraversal);

/// Enables or disables folding during greedy rewriting.
MLIR_CAPI_EXPORTED void
mlirGreedyRewriteDriverConfigEnableFolding(MlirGreedyRewriteDriverConfig config,
                                           bool enable);

/// Sets the strictness level for the greedy rewrite driver.
MLIR_CAPI_EXPORTED void mlirGreedyRewriteDriverConfigSetStrictness(
    MlirGreedyRewriteDriverConfig config,
    MlirGreedyRewriteStrictness strictness);

/// Sets the region simplification level.
MLIR_CAPI_EXPORTED void
mlirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
    MlirGreedyRewriteDriverConfig config, MlirGreedySimplifyRegionLevel level);

/// Enables or disables constant CSE.
MLIR_CAPI_EXPORTED void mlirGreedyRewriteDriverConfigEnableConstantCSE(
    MlirGreedyRewriteDriverConfig config, bool enable);

/// Gets the maximum number of iterations for the greedy rewrite driver.
MLIR_CAPI_EXPORTED int64_t mlirGreedyRewriteDriverConfigGetMaxIterations(
    MlirGreedyRewriteDriverConfig config);

/// Gets the maximum number of rewrites within an iteration.
MLIR_CAPI_EXPORTED int64_t mlirGreedyRewriteDriverConfigGetMaxNumRewrites(
    MlirGreedyRewriteDriverConfig config);

/// Gets whether top-down traversal is used for initial worklist population.
MLIR_CAPI_EXPORTED bool mlirGreedyRewriteDriverConfigGetUseTopDownTraversal(
    MlirGreedyRewriteDriverConfig config);

/// Gets whether folding is enabled during greedy rewriting.
MLIR_CAPI_EXPORTED bool mlirGreedyRewriteDriverConfigIsFoldingEnabled(
    MlirGreedyRewriteDriverConfig config);

/// Gets the strictness level for the greedy rewrite driver.
MLIR_CAPI_EXPORTED MlirGreedyRewriteStrictness
mlirGreedyRewriteDriverConfigGetStrictness(
    MlirGreedyRewriteDriverConfig config);

/// Gets the region simplification level.
MLIR_CAPI_EXPORTED MlirGreedySimplifyRegionLevel
mlirGreedyRewriteDriverConfigGetRegionSimplificationLevel(
    MlirGreedyRewriteDriverConfig config);

/// Gets whether constant CSE is enabled.
MLIR_CAPI_EXPORTED bool mlirGreedyRewriteDriverConfigIsConstantCSEEnabled(
    MlirGreedyRewriteDriverConfig config);

/// Applies the given patterns to the given op by a fast walk-based pattern
/// rewrite driver.
MLIR_CAPI_EXPORTED void
mlirWalkAndApplyPatterns(MlirOperation op,
                         MlirFrozenRewritePatternSet patterns);

/// Apply a partial conversion on the given operation.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirApplyPartialConversion(
    MlirOperation op, MlirConversionTarget target,
    MlirFrozenRewritePatternSet patterns, MlirConversionConfig config);

/// Apply a full conversion on the given operation.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirApplyFullConversion(
    MlirOperation op, MlirConversionTarget target,
    MlirFrozenRewritePatternSet patterns, MlirConversionConfig config);

//===----------------------------------------------------------------------===//
/// ConversionConfig API
//===----------------------------------------------------------------------===//

/// Create a default ConversionConfig.
MLIR_CAPI_EXPORTED MlirConversionConfig mlirConversionConfigCreate(void);

/// Destroy the given ConversionConfig.
MLIR_CAPI_EXPORTED void
mlirConversionConfigDestroy(MlirConversionConfig config);

typedef enum {
  MLIR_DIALECT_CONVERSION_FOLDING_MODE_NEVER,
  MLIR_DIALECT_CONVERSION_FOLDING_MODE_BEFORE_PATTERNS,
  MLIR_DIALECT_CONVERSION_FOLDING_MODE_AFTER_PATTERNS,
} MlirDialectConversionFoldingMode;

/// Set the folding mode for the given ConversionConfig.
MLIR_CAPI_EXPORTED void
mlirConversionConfigSetFoldingMode(MlirConversionConfig config,
                                   MlirDialectConversionFoldingMode mode);

/// Get the folding mode for the given ConversionConfig.
MLIR_CAPI_EXPORTED MlirDialectConversionFoldingMode
mlirConversionConfigGetFoldingMode(MlirConversionConfig config);

/// Enable or disable building materializations during conversion.
MLIR_CAPI_EXPORTED void
mlirConversionConfigEnableBuildMaterializations(MlirConversionConfig config,
                                                bool enable);

/// Check if building materializations during conversion is enabled.
MLIR_CAPI_EXPORTED bool
mlirConversionConfigIsBuildMaterializationsEnabled(MlirConversionConfig config);

//===----------------------------------------------------------------------===//
/// PatternRewriter API
//===----------------------------------------------------------------------===//

/// Cast the PatternRewriter to a RewriterBase
MLIR_CAPI_EXPORTED MlirRewriterBase
mlirPatternRewriterAsBase(MlirPatternRewriter rewriter);

//===----------------------------------------------------------------------===//
/// ConversionPatternRewriter API
//===----------------------------------------------------------------------===//

/// Cast the ConversionPatternRewriter to a PatternRewriter
MLIR_CAPI_EXPORTED MlirPatternRewriter
mlirConversionPatternRewriterAsPatternRewriter(
    MlirConversionPatternRewriter rewriter);

//===----------------------------------------------------------------------===//
/// ConversionTarget API
//===----------------------------------------------------------------------===//

/// Create an empty ConversionTarget.
MLIR_CAPI_EXPORTED MlirConversionTarget
mlirConversionTargetCreate(MlirContext context);

/// Destroy the given ConversionTarget.
MLIR_CAPI_EXPORTED void
mlirConversionTargetDestroy(MlirConversionTarget target);

/// Register the given operations as legal.
MLIR_CAPI_EXPORTED void
mlirConversionTargetAddLegalOp(MlirConversionTarget target,
                               MlirStringRef opName);

/// Register the given operations as illegal.
MLIR_CAPI_EXPORTED void
mlirConversionTargetAddIllegalOp(MlirConversionTarget target,
                                 MlirStringRef opName);

/// Register the operations of the given dialect as legal.
MLIR_CAPI_EXPORTED void
mlirConversionTargetAddLegalDialect(MlirConversionTarget target,
                                    MlirStringRef dialectName);

/// Register the operations of the given dialect as illegal.
MLIR_CAPI_EXPORTED void
mlirConversionTargetAddIllegalDialect(MlirConversionTarget target,
                                      MlirStringRef dialectName);

//===----------------------------------------------------------------------===//
/// TypeConverter API
//===----------------------------------------------------------------------===//

/// Create a TypeConverter.
MLIR_CAPI_EXPORTED MlirTypeConverter mlirTypeConverterCreate(void);

/// Destroy the given TypeConverter.
MLIR_CAPI_EXPORTED void
mlirTypeConverterDestroy(MlirTypeConverter typeConverter);

/// Callback type for type conversion functions.
/// Returns failure or sets convertedType to MlirType{NULL} to indicate failure.
/// If failure is returned, the converter is allowed to try another
/// conversion function to perform the conversion.
typedef MlirLogicalResult (*MlirTypeConverterConversionCallback)(
    MlirType type, MlirType *convertedType, void *userData);

/// Add a type conversion function to the given TypeConverter.
MLIR_CAPI_EXPORTED void
mlirTypeConverterAddConversion(MlirTypeConverter typeConverter,
                               MlirTypeConverterConversionCallback convertType,
                               void *userData);

//===----------------------------------------------------------------------===//
/// ConversionPattern API
//===----------------------------------------------------------------------===//

typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// The callback function to match against code rooted at the specified
  /// operation, and perform the conversion rewrite if the match is successful,
  /// corresponding to ConversionPattern::matchAndRewrite.
  MlirLogicalResult (*matchAndRewrite)(MlirConversionPattern pattern,
                                       MlirOperation op, intptr_t nOperands,
                                       MlirValue *operands,
                                       MlirConversionPatternRewriter rewriter,
                                       void *userData);
} MlirConversionPatternCallbacks;

/// Create a conversion pattern that matches the operation with the given
/// rootName, corresponding to mlir::OpConversionPattern.
MLIR_CAPI_EXPORTED MlirConversionPattern mlirOpConversionPatternCreate(
    MlirStringRef rootName, unsigned benefit, MlirContext context,
    MlirTypeConverter typeConverter, MlirConversionPatternCallbacks callbacks,
    void *userData, size_t nGeneratedNames, MlirStringRef *generatedNames);

/// Get the type converter used by this conversion pattern.
MLIR_CAPI_EXPORTED MlirTypeConverter
mlirConversionPatternGetTypeConverter(MlirConversionPattern pattern);

/// Cast the ConversionPattern to a RewritePattern.
MLIR_CAPI_EXPORTED MlirRewritePattern
mlirConversionPatternAsRewritePattern(MlirConversionPattern pattern);

//===----------------------------------------------------------------------===//
/// RewritePattern API
//===----------------------------------------------------------------------===//

/// Callbacks to construct a rewrite pattern.
typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// The callback function to match against code rooted at the specified
  /// operation, and perform the rewrite if the match is successful,
  /// corresponding to RewritePattern::matchAndRewrite.
  MlirLogicalResult (*matchAndRewrite)(MlirRewritePattern pattern,
                                       MlirOperation op,
                                       MlirPatternRewriter rewriter,
                                       void *userData);
} MlirRewritePatternCallbacks;

/// Create a rewrite pattern that matches the operation
/// with the given rootName, corresponding to mlir::OpRewritePattern.
MLIR_CAPI_EXPORTED MlirRewritePattern mlirOpRewritePatternCreate(
    MlirStringRef rootName, unsigned benefit, MlirContext context,
    MlirRewritePatternCallbacks callbacks, void *userData,
    size_t nGeneratedNames, MlirStringRef *generatedNames);

//===----------------------------------------------------------------------===//
/// RewritePatternSet API
//===----------------------------------------------------------------------===//

/// Create an empty MlirRewritePatternSet.
MLIR_CAPI_EXPORTED MlirRewritePatternSet
mlirRewritePatternSetCreate(MlirContext context);

/// Destruct the given MlirRewritePatternSet.
MLIR_CAPI_EXPORTED void mlirRewritePatternSetDestroy(MlirRewritePatternSet set);

/// Add the given MlirRewritePattern into a MlirRewritePatternSet.
/// Note that the ownership of the pattern is transferred to the set after this
/// call.
MLIR_CAPI_EXPORTED void mlirRewritePatternSetAdd(MlirRewritePatternSet set,
                                                 MlirRewritePattern pattern);

//===----------------------------------------------------------------------===//
/// PDLPatternModule API
//===----------------------------------------------------------------------===//

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
DEFINE_C_API_STRUCT(MlirPDLPatternModule, void);
DEFINE_C_API_STRUCT(MlirPDLValue, const void);
DEFINE_C_API_STRUCT(MlirPDLResultList, void);

MLIR_CAPI_EXPORTED MlirPDLPatternModule
mlirPDLPatternModuleFromModule(MlirModule op);

MLIR_CAPI_EXPORTED void mlirPDLPatternModuleDestroy(MlirPDLPatternModule op);

MLIR_CAPI_EXPORTED MlirRewritePatternSet
mlirRewritePatternSetFromPDLPatternModule(MlirPDLPatternModule op);

/// Cast the MlirPDLValue to an MlirValue.
/// Return a null value if the cast fails, just like llvm::dyn_cast.
MLIR_CAPI_EXPORTED MlirValue mlirPDLValueAsValue(MlirPDLValue value);

/// Cast the MlirPDLValue to an MlirType.
/// Return a null value if the cast fails, just like llvm::dyn_cast.
MLIR_CAPI_EXPORTED MlirType mlirPDLValueAsType(MlirPDLValue value);

/// Cast the MlirPDLValue to an MlirOperation.
/// Return a null value if the cast fails, just like llvm::dyn_cast.
MLIR_CAPI_EXPORTED MlirOperation mlirPDLValueAsOperation(MlirPDLValue value);

/// Cast the MlirPDLValue to an MlirAttribute.
/// Return a null value if the cast fails, just like llvm::dyn_cast.
MLIR_CAPI_EXPORTED MlirAttribute mlirPDLValueAsAttribute(MlirPDLValue value);

/// Push the MlirValue into the given MlirPDLResultList.
MLIR_CAPI_EXPORTED void
mlirPDLResultListPushBackValue(MlirPDLResultList results, MlirValue value);

/// Push the MlirType into the given MlirPDLResultList.
MLIR_CAPI_EXPORTED void mlirPDLResultListPushBackType(MlirPDLResultList results,
                                                      MlirType value);

/// Push the MlirOperation into the given MlirPDLResultList.
MLIR_CAPI_EXPORTED void
mlirPDLResultListPushBackOperation(MlirPDLResultList results,
                                   MlirOperation value);

/// Push the MlirAttribute into the given MlirPDLResultList.
MLIR_CAPI_EXPORTED void
mlirPDLResultListPushBackAttribute(MlirPDLResultList results,
                                   MlirAttribute value);

/// This function type is used as callbacks for PDL native rewrite functions.
/// Input values can be accessed by `values` with its size `nValues`;
/// output values can be added into `results` by `mlirPDLResultListPushBack*`
/// APIs. And the return value indicates whether the rewrite succeeds.
typedef MlirLogicalResult (*MlirPDLRewriteFunction)(
    MlirPatternRewriter rewriter, MlirPDLResultList results, size_t nValues,
    MlirPDLValue *values, void *userData);

/// Register a rewrite function into the given PDL pattern module.
/// `userData` will be provided as an argument to the rewrite function.
MLIR_CAPI_EXPORTED void mlirPDLPatternModuleRegisterRewriteFunction(
    MlirPDLPatternModule pdlModule, MlirStringRef name,
    MlirPDLRewriteFunction rewriteFn, void *userData);

/// This function type is used as callbacks for PDL native constraint functions.
/// Input values can be accessed by `values` with its size `nValues`;
/// output values can be added into `results` by `mlirPDLResultListPushBack*`
/// APIs. And the return value indicates whether the constraint holds.
typedef MlirLogicalResult (*MlirPDLConstraintFunction)(
    MlirPatternRewriter rewriter, MlirPDLResultList results, size_t nValues,
    MlirPDLValue *values, void *userData);

/// Register a constraint function into the given PDL pattern module.
/// `userData` will be provided as an argument to the constraint function.
MLIR_CAPI_EXPORTED void mlirPDLPatternModuleRegisterConstraintFunction(
    MlirPDLPatternModule pdlModule, MlirStringRef name,
    MlirPDLConstraintFunction constraintFn, void *userData);

#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH

#undef DEFINE_C_API_STRUCT

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_REWRITE_H
