//===- DialectConversion.cpp - MLIR dialect conversion generic pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace mlir;
using namespace mlir::detail;

#define DEBUG_TYPE "dialect-conversion"

/// Recursively collect all of the operations to convert from within 'region'.
/// If 'target' is nonnull, operations that are recursively legal have their
/// regions pre-filtered to avoid considering them for legalization.
static LogicalResult
computeConversionSet(iterator_range<Region::iterator> region,
                     Location regionLoc, std::vector<Operation *> &toConvert,
                     ConversionTarget *target = nullptr) {
  if (llvm::empty(region))
    return success();

  // Traverse starting from the entry block.
  SmallVector<Block *, 16> worklist(1, &*region.begin());
  DenseSet<Block *> visitedBlocks;
  visitedBlocks.insert(worklist.front());
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();

    // Compute the conversion set of each of the nested operations.
    for (Operation &op : *block) {
      toConvert.emplace_back(&op);

      // Don't check this operation's children for conversion if the operation
      // is recursively legal.
      auto legalityInfo = target ? target->isLegal(&op)
                                 : Optional<ConversionTarget::LegalOpDetails>();
      if (legalityInfo && legalityInfo->isRecursivelyLegal)
        continue;
      for (auto &region : op.getRegions()) {
        if (failed(computeConversionSet(region.getBlocks(), region.getLoc(),
                                        toConvert, target)))
          return failure();
      }
    }

    // Recurse to children that haven't been visited.
    for (Block *succ : block->getSuccessors())
      if (visitedBlocks.insert(succ).second)
        worklist.push_back(succ);
  }

  // Check that all blocks in the region were visited.
  if (llvm::any_of(llvm::drop_begin(region, 1),
                   [&](Block &block) { return !visitedBlocks.count(&block); }))
    return emitError(regionLoc, "unreachable blocks were not converted");
  return success();
}

/// A utility function to log a successful result for the given reason.
template <typename... Args>
static void logSuccess(llvm::ScopedPrinter &os, StringRef fmt, Args &&...args) {
  LLVM_DEBUG({
    os.unindent();
    os.startLine() << "} -> SUCCESS";
    if (!fmt.empty())
      os.getOStream() << " : "
                      << llvm::formatv(fmt.data(), std::forward<Args>(args)...);
    os.getOStream() << "\n";
  });
}

/// A utility function to log a failure result for the given reason.
template <typename... Args>
static void logFailure(llvm::ScopedPrinter &os, StringRef fmt, Args &&...args) {
  LLVM_DEBUG({
    os.unindent();
    os.startLine() << "} -> FAILURE : "
                   << llvm::formatv(fmt.data(), std::forward<Args>(args)...)
                   << "\n";
  });
}

//===----------------------------------------------------------------------===//
// ConversionValueMapping
//===----------------------------------------------------------------------===//

namespace {
/// This class wraps a BlockAndValueMapping to provide recursive lookup
/// functionality, i.e. we will traverse if the mapped value also has a mapping.
struct ConversionValueMapping {
  /// Lookup a mapped value within the map. If a mapping for the provided value
  /// does not exist then return the provided value. If `desiredType` is
  /// non-null, returns the most recently mapped value with that type. If an
  /// operand of that type does not exist, defaults to normal behavior.
  Value lookupOrDefault(Value from, Type desiredType = nullptr) const;

  /// Lookup a mapped value within the map, or return null if a mapping does not
  /// exist. If a mapping exists, this follows the same behavior of
  /// `lookupOrDefault`.
  Value lookupOrNull(Value from) const;

  /// Map a value to the one provided.
  void map(Value oldVal, Value newVal) { mapping.map(oldVal, newVal); }

  /// Drop the last mapping for the given value.
  void erase(Value value) { mapping.erase(value); }

  /// Returns the inverse raw value mapping (without recursive query support).
  BlockAndValueMapping getInverse() const { return mapping.getInverse(); }

private:
  /// Current value mappings.
  BlockAndValueMapping mapping;
};
} // end anonymous namespace

Value ConversionValueMapping::lookupOrDefault(Value from,
                                              Type desiredType) const {
  // If there was no desired type, simply find the leaf value.
  if (!desiredType) {
    // If this value had a valid mapping, unmap that value as well in the case
    // that it was also replaced.
    while (auto mappedValue = mapping.lookupOrNull(from))
      from = mappedValue;
    return from;
  }

  // Otherwise, try to find the deepest value that has the desired type.
  Value desiredValue;
  do {
    if (from.getType() == desiredType)
      desiredValue = from;

    Value mappedValue = mapping.lookupOrNull(from);
    if (!mappedValue)
      break;
    from = mappedValue;
  } while (true);

  // If the desired value was found use it, otherwise default to the leaf value.
  return desiredValue ? desiredValue : from;
}

Value ConversionValueMapping::lookupOrNull(Value from) const {
  Value result = lookupOrDefault(from);
  return result == from ? nullptr : result;
}

//===----------------------------------------------------------------------===//
// ArgConverter
//===----------------------------------------------------------------------===//
namespace {
/// This class provides a simple interface for converting the types of block
/// arguments. This is done by creating a new block that contains the new legal
/// types and extracting the block that contains the old illegal types to allow
/// for undoing pending rewrites in the case of failure.
struct ArgConverter {
  ArgConverter(PatternRewriter &rewriter) : rewriter(rewriter) {}

  /// This structure contains the information pertaining to an argument that has
  /// been converted.
  struct ConvertedArgInfo {
    ConvertedArgInfo(unsigned newArgIdx, unsigned newArgSize,
                     Value castValue = nullptr)
        : newArgIdx(newArgIdx), newArgSize(newArgSize), castValue(castValue) {}

    /// The start index of in the new argument list that contains arguments that
    /// replace the original.
    unsigned newArgIdx;

    /// The number of arguments that replaced the original argument.
    unsigned newArgSize;

    /// The cast value that was created to cast from the new arguments to the
    /// old. This only used if 'newArgSize' > 1.
    Value castValue;
  };

  /// This structure contains information pertaining to a block that has had its
  /// signature converted.
  struct ConvertedBlockInfo {
    ConvertedBlockInfo(Block *origBlock, TypeConverter &converter)
        : origBlock(origBlock), converter(&converter) {}

    /// The original block that was requested to have its signature converted.
    Block *origBlock;

    /// The conversion information for each of the arguments. The information is
    /// None if the argument was dropped during conversion.
    SmallVector<Optional<ConvertedArgInfo>, 1> argInfo;

    /// The type converter used to convert the arguments.
    TypeConverter *converter;
  };

  /// Return if the signature of the given block has already been converted.
  bool hasBeenConverted(Block *block) const {
    return conversionInfo.count(block) || convertedBlocks.count(block);
  }

  /// Set the type converter to use for the given region.
  void setConverter(Region *region, TypeConverter *typeConverter) {
    assert(typeConverter && "expected valid type converter");
    regionToConverter[region] = typeConverter;
  }

  /// Return the type converter to use for the given region, or null if there
  /// isn't one.
  TypeConverter *getConverter(Region *region) {
    return regionToConverter.lookup(region);
  }

  //===--------------------------------------------------------------------===//
  // Rewrite Application
  //===--------------------------------------------------------------------===//

  /// Erase any rewrites registered for the blocks within the given operation
  /// which is about to be removed. This merely drops the rewrites without
  /// undoing them.
  void notifyOpRemoved(Operation *op);

  /// Cleanup and undo any generated conversions for the arguments of block.
  /// This method replaces the new block with the original, reverting the IR to
  /// its original state.
  void discardRewrites(Block *block);

  /// Fully replace uses of the old arguments with the new.
  void applyRewrites(ConversionValueMapping &mapping);

  /// Materialize any necessary conversions for converted arguments that have
  /// live users, using the provided `findLiveUser` to search for a user that
  /// survives the conversion process.
  LogicalResult
  materializeLiveConversions(ConversionValueMapping &mapping,
                             OpBuilder &builder,
                             function_ref<Operation *(Value)> findLiveUser);

  //===--------------------------------------------------------------------===//
  // Conversion
  //===--------------------------------------------------------------------===//

  /// Attempt to convert the signature of the given block, if successful a new
  /// block is returned containing the new arguments. Returns `block` if it did
  /// not require conversion.
  FailureOr<Block *>
  convertSignature(Block *block, TypeConverter &converter,
                   ConversionValueMapping &mapping,
                   SmallVectorImpl<BlockArgument> &argReplacements);

  /// Apply the given signature conversion on the given block. The new block
  /// containing the updated signature is returned. If no conversions were
  /// necessary, e.g. if the block has no arguments, `block` is returned.
  /// `converter` is used to generate any necessary cast operations that
  /// translate between the origin argument types and those specified in the
  /// signature conversion.
  Block *applySignatureConversion(
      Block *block, TypeConverter &converter,
      TypeConverter::SignatureConversion &signatureConversion,
      ConversionValueMapping &mapping,
      SmallVectorImpl<BlockArgument> &argReplacements);

  /// Insert a new conversion into the cache.
  void insertConversion(Block *newBlock, ConvertedBlockInfo &&info);

  /// A collection of blocks that have had their arguments converted. This is a
  /// map from the new replacement block, back to the original block.
  llvm::MapVector<Block *, ConvertedBlockInfo> conversionInfo;

  /// The set of original blocks that were converted.
  DenseSet<Block *> convertedBlocks;

  /// A mapping from valid regions, to those containing the original blocks of a
  /// conversion.
  DenseMap<Region *, std::unique_ptr<Region>> regionMapping;

  /// A mapping of regions to type converters that should be used when
  /// converting the arguments of blocks within that region.
  DenseMap<Region *, TypeConverter *> regionToConverter;

  /// The pattern rewriter to use when materializing conversions.
  PatternRewriter &rewriter;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Rewrite Application

void ArgConverter::notifyOpRemoved(Operation *op) {
  if (conversionInfo.empty())
    return;

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      // Drop any rewrites from within.
      for (Operation &nestedOp : block)
        if (nestedOp.getNumRegions())
          notifyOpRemoved(&nestedOp);

      // Check if this block was converted.
      auto it = conversionInfo.find(&block);
      if (it == conversionInfo.end())
        continue;

      // Drop all uses of the original arguments and delete the original block.
      Block *origBlock = it->second.origBlock;
      for (BlockArgument arg : origBlock->getArguments())
        arg.dropAllUses();
      conversionInfo.erase(it);
    }
  }
}

void ArgConverter::discardRewrites(Block *block) {
  auto it = conversionInfo.find(block);
  if (it == conversionInfo.end())
    return;
  Block *origBlock = it->second.origBlock;

  // Drop all uses of the new block arguments and replace uses of the new block.
  for (int i = block->getNumArguments() - 1; i >= 0; --i)
    block->getArgument(i).dropAllUses();
  block->replaceAllUsesWith(origBlock);

  // Move the operations back the original block and the delete the new block.
  origBlock->getOperations().splice(origBlock->end(), block->getOperations());
  origBlock->moveBefore(block);
  block->erase();

  convertedBlocks.erase(origBlock);
  conversionInfo.erase(it);
}

void ArgConverter::applyRewrites(ConversionValueMapping &mapping) {
  for (auto &info : conversionInfo) {
    ConvertedBlockInfo &blockInfo = info.second;
    Block *origBlock = blockInfo.origBlock;

    // Process the remapping for each of the original arguments.
    for (unsigned i = 0, e = origBlock->getNumArguments(); i != e; ++i) {
      Optional<ConvertedArgInfo> &argInfo = blockInfo.argInfo[i];
      BlockArgument origArg = origBlock->getArgument(i);

      // Handle the case of a 1->0 value mapping.
      if (!argInfo) {
        if (Value newArg = mapping.lookupOrNull(origArg))
          origArg.replaceAllUsesWith(newArg);
        continue;
      }

      // Otherwise this is a 1->1+ value mapping.
      Value castValue = argInfo->castValue;
      assert(argInfo->newArgSize >= 1 && castValue && "expected 1->1+ mapping");

      // If the argument is still used, replace it with the generated cast.
      if (!origArg.use_empty())
        origArg.replaceAllUsesWith(mapping.lookupOrDefault(castValue));
    }
  }
}

LogicalResult ArgConverter::materializeLiveConversions(
    ConversionValueMapping &mapping, OpBuilder &builder,
    function_ref<Operation *(Value)> findLiveUser) {
  for (auto &info : conversionInfo) {
    Block *newBlock = info.first;
    ConvertedBlockInfo &blockInfo = info.second;
    Block *origBlock = blockInfo.origBlock;

    // Process the remapping for each of the original arguments.
    for (unsigned i = 0, e = origBlock->getNumArguments(); i != e; ++i) {
      // FIXME: We should run the below checks even if the type conversion was
      // 1->N, but a lot of existing lowering rely on the block argument being
      // blindly replaced. Those usages should be updated, and this if should be
      // removed.
      if (blockInfo.argInfo[i])
        continue;

      // If the type of this argument changed and the argument is still live, we
      // need to materialize a conversion.
      BlockArgument origArg = origBlock->getArgument(i);
      auto argReplacementValue = mapping.lookupOrDefault(origArg);
      bool isDroppedArg = argReplacementValue == origArg;
      if (argReplacementValue.getType() == origArg.getType() && !isDroppedArg)
        continue;
      Operation *liveUser = findLiveUser(origArg);
      if (!liveUser)
        continue;

      if (OpResult result = argReplacementValue.dyn_cast<OpResult>())
        rewriter.setInsertionPointAfter(result.getOwner());
      else
        rewriter.setInsertionPointToStart(newBlock);
      Value newArg = blockInfo.converter->materializeSourceConversion(
          rewriter, origArg.getLoc(), origArg.getType(),
          isDroppedArg ? ValueRange() : ValueRange(argReplacementValue));
      if (!newArg) {
        InFlightDiagnostic diag =
            emitError(origArg.getLoc())
            << "failed to materialize conversion for block argument #" << i
            << " that remained live after conversion, type was "
            << origArg.getType();
        if (!isDroppedArg)
          diag << ", with target type " << argReplacementValue.getType();
        diag.attachNote(liveUser->getLoc())
            << "see existing live user here: " << *liveUser;
        return failure();
      }
      mapping.map(origArg, newArg);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Conversion

FailureOr<Block *> ArgConverter::convertSignature(
    Block *block, TypeConverter &converter, ConversionValueMapping &mapping,
    SmallVectorImpl<BlockArgument> &argReplacements) {
  // Check if the block was already converted. If the block is detached,
  // conservatively assume it is going to be deleted.
  if (hasBeenConverted(block) || !block->getParent())
    return block;

  // Try to convert the signature for the block with the provided converter.
  if (auto conversion = converter.convertBlockSignature(block))
    return applySignatureConversion(block, converter, *conversion, mapping,
                                    argReplacements);
  return failure();
}

Block *ArgConverter::applySignatureConversion(
    Block *block, TypeConverter &converter,
    TypeConverter::SignatureConversion &signatureConversion,
    ConversionValueMapping &mapping,
    SmallVectorImpl<BlockArgument> &argReplacements) {
  // If no arguments are being changed or added, there is nothing to do.
  unsigned origArgCount = block->getNumArguments();
  auto convertedTypes = signatureConversion.getConvertedTypes();
  if (origArgCount == 0 && convertedTypes.empty())
    return block;

  // Split the block at the beginning to get a new block to use for the updated
  // signature.
  Block *newBlock = block->splitBlock(block->begin());
  block->replaceAllUsesWith(newBlock);

  SmallVector<Value, 4> newArgRange(newBlock->addArguments(convertedTypes));
  ArrayRef<Value> newArgs(newArgRange);

  // Remap each of the original arguments as determined by the signature
  // conversion.
  ConvertedBlockInfo info(block, converter);
  info.argInfo.resize(origArgCount);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newBlock);
  for (unsigned i = 0; i != origArgCount; ++i) {
    auto inputMap = signatureConversion.getInputMapping(i);
    if (!inputMap)
      continue;
    BlockArgument origArg = block->getArgument(i);

    // If inputMap->replacementValue is not nullptr, then the argument is
    // dropped and a replacement value is provided to be the remappedValue.
    if (inputMap->replacementValue) {
      assert(inputMap->size == 0 &&
             "invalid to provide a replacement value when the argument isn't "
             "dropped");
      mapping.map(origArg, inputMap->replacementValue);
      argReplacements.push_back(origArg);
      continue;
    }

    // Otherwise, this is a 1->1+ mapping. Call into the provided type converter
    // to pack the new values. For 1->1 mappings, if there is no materialization
    // provided, use the argument directly instead.
    auto replArgs = newArgs.slice(inputMap->inputNo, inputMap->size);
    Value newArg;

    // If this is a 1->1 mapping and the types of new and replacement arguments
    // match (i.e. it's an identity map), then the argument is mapped to its
    // original type.
    if (replArgs.size() == 1 && replArgs[0].getType() == origArg.getType())
      newArg = replArgs[0];
    else
      newArg = converter.materializeArgumentConversion(
          rewriter, origArg.getLoc(), origArg.getType(), replArgs);

    if (!newArg) {
      assert(replArgs.size() == 1 &&
             "couldn't materialize the result of 1->N conversion");
      newArg = replArgs.front();
    }
    mapping.map(origArg, newArg);
    argReplacements.push_back(origArg);
    info.argInfo[i] =
        ConvertedArgInfo(inputMap->inputNo, inputMap->size, newArg);
  }

  // Remove the original block from the region and return the new one.
  insertConversion(newBlock, std::move(info));
  return newBlock;
}

void ArgConverter::insertConversion(Block *newBlock,
                                    ConvertedBlockInfo &&info) {
  // Get a region to insert the old block.
  Region *region = newBlock->getParent();
  std::unique_ptr<Region> &mappedRegion = regionMapping[region];
  if (!mappedRegion)
    mappedRegion = std::make_unique<Region>(region->getParentOp());

  // Move the original block to the mapped region and emplace the conversion.
  mappedRegion->getBlocks().splice(mappedRegion->end(), region->getBlocks(),
                                   info.origBlock->getIterator());
  convertedBlocks.insert(info.origBlock);
  conversionInfo.insert({newBlock, std::move(info)});
}

//===----------------------------------------------------------------------===//
// Rewriter and Translation State
//===----------------------------------------------------------------------===//
namespace {
/// This class contains a snapshot of the current conversion rewriter state.
/// This is useful when saving and undoing a set of rewrites.
struct RewriterState {
  RewriterState(unsigned numCreatedOps, unsigned numReplacements,
                unsigned numArgReplacements, unsigned numBlockActions,
                unsigned numIgnoredOperations, unsigned numRootUpdates)
      : numCreatedOps(numCreatedOps), numReplacements(numReplacements),
        numArgReplacements(numArgReplacements),
        numBlockActions(numBlockActions),
        numIgnoredOperations(numIgnoredOperations),
        numRootUpdates(numRootUpdates) {}

  /// The current number of created operations.
  unsigned numCreatedOps;

  /// The current number of replacements queued.
  unsigned numReplacements;

  /// The current number of argument replacements queued.
  unsigned numArgReplacements;

  /// The current number of block actions performed.
  unsigned numBlockActions;

  /// The current number of ignored operations.
  unsigned numIgnoredOperations;

  /// The current number of operations that were updated in place.
  unsigned numRootUpdates;
};

/// The state of an operation that was updated by a pattern in-place. This
/// contains all of the necessary information to reconstruct an operation that
/// was updated in place.
class OperationTransactionState {
public:
  OperationTransactionState() = default;
  OperationTransactionState(Operation *op)
      : op(op), loc(op->getLoc()), attrs(op->getAttrDictionary()),
        operands(op->operand_begin(), op->operand_end()),
        successors(op->successor_begin(), op->successor_end()) {}

  /// Discard the transaction state and reset the state of the original
  /// operation.
  void resetOperation() const {
    op->setLoc(loc);
    op->setAttrs(attrs);
    op->setOperands(operands);
    for (auto it : llvm::enumerate(successors))
      op->setSuccessor(it.value(), it.index());
  }

  /// Return the original operation of this state.
  Operation *getOperation() const { return op; }

private:
  Operation *op;
  LocationAttr loc;
  DictionaryAttr attrs;
  SmallVector<Value, 8> operands;
  SmallVector<Block *, 2> successors;
};

/// This class represents one requested operation replacement via 'replaceOp' or
/// 'eraseOp`.
struct OpReplacement {
  OpReplacement() = default;
  OpReplacement(TypeConverter *converter) : converter(converter) {}

  /// An optional type converter that can be used to materialize conversions
  /// between the new and old values if necessary.
  TypeConverter *converter = nullptr;
};

/// The kind of the block action performed during the rewrite.  Actions can be
/// undone if the conversion fails.
enum class BlockActionKind {
  Create,
  Erase,
  Merge,
  Move,
  Split,
  TypeConversion
};

/// Original position of the given block in its parent region. During undo
/// actions, the block needs to be placed after `insertAfterBlock`.
struct BlockPosition {
  Region *region;
  Block *insertAfterBlock;
};

/// Information needed to undo the merge actions.
/// - the source block, and
/// - the Operation that was the last operation in the dest block before the
///   merge (could be null if the dest block was empty).
struct MergeInfo {
  Block *sourceBlock;
  Operation *destBlockLastInst;
};

/// The storage class for an undoable block action (one of BlockActionKind),
/// contains the information necessary to undo this action.
struct BlockAction {
  static BlockAction getCreate(Block *block) {
    return {BlockActionKind::Create, block, {}};
  }
  static BlockAction getErase(Block *block, BlockPosition originalPosition) {
    return {BlockActionKind::Erase, block, {originalPosition}};
  }
  static BlockAction getMerge(Block *block, Block *sourceBlock) {
    BlockAction action{BlockActionKind::Merge, block, {}};
    action.mergeInfo = {sourceBlock, block->empty() ? nullptr : &block->back()};
    return action;
  }
  static BlockAction getMove(Block *block, BlockPosition originalPosition) {
    return {BlockActionKind::Move, block, {originalPosition}};
  }
  static BlockAction getSplit(Block *block, Block *originalBlock) {
    BlockAction action{BlockActionKind::Split, block, {}};
    action.originalBlock = originalBlock;
    return action;
  }
  static BlockAction getTypeConversion(Block *block) {
    return BlockAction{BlockActionKind::TypeConversion, block, {}};
  }

  // The action kind.
  BlockActionKind kind;

  // A pointer to the block that was created by the action.
  Block *block;

  union {
    // In use if kind == BlockActionKind::Move or BlockActionKind::Erase, and
    // contains a pointer to the region that originally contained the block as
    // well as the position of the block in that region.
    BlockPosition originalPosition;
    // In use if kind == BlockActionKind::Split and contains a pointer to the
    // block that was split into two parts.
    Block *originalBlock;
    // In use if kind == BlockActionKind::Merge, and contains the information
    // needed to undo the merge.
    MergeInfo mergeInfo;
  };
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConversionPatternRewriterImpl
//===----------------------------------------------------------------------===//
namespace mlir {
namespace detail {
struct ConversionPatternRewriterImpl {
  ConversionPatternRewriterImpl(PatternRewriter &rewriter)
      : argConverter(rewriter) {}

  /// Cleanup and destroy any generated rewrite operations. This method is
  /// invoked when the conversion process fails.
  void discardRewrites();

  /// Apply all requested operation rewrites. This method is invoked when the
  /// conversion process succeeds.
  void applyRewrites();

  //===--------------------------------------------------------------------===//
  // State Management
  //===--------------------------------------------------------------------===//

  /// Return the current state of the rewriter.
  RewriterState getCurrentState();

  /// Reset the state of the rewriter to a previously saved point.
  void resetState(RewriterState state);

  /// Erase any blocks that were unlinked from their regions and stored in block
  /// actions.
  void eraseDanglingBlocks();

  /// Undo the block actions (motions, splits) one by one in reverse order until
  /// "numActionsToKeep" actions remains.
  void undoBlockActions(unsigned numActionsToKeep = 0);

  /// Remap the given operands to those with potentially different types. The
  /// provided type converter is used to ensure that the remapped types are
  /// legal. Returns success if the operands could be remapped, failure
  /// otherwise.
  LogicalResult remapValues(Location loc, PatternRewriter &rewriter,
                            TypeConverter *converter,
                            Operation::operand_range operands,
                            SmallVectorImpl<Value> &remapped);

  /// Returns true if the given operation is ignored, and does not need to be
  /// converted.
  bool isOpIgnored(Operation *op) const;

  /// Recursively marks the nested operations under 'op' as ignored. This
  /// removes them from being considered for legalization.
  void markNestedOpsIgnored(Operation *op);

  //===--------------------------------------------------------------------===//
  // Type Conversion
  //===--------------------------------------------------------------------===//

  /// Convert the signature of the given block.
  FailureOr<Block *> convertBlockSignature(
      Block *block, TypeConverter &converter,
      TypeConverter::SignatureConversion *conversion = nullptr);

  /// Apply a signature conversion on the given region, using `converter` for
  /// materializations if not null.
  Block *
  applySignatureConversion(Region *region,
                           TypeConverter::SignatureConversion &conversion,
                           TypeConverter *converter);

  /// Convert the types of block arguments within the given region.
  FailureOr<Block *>
  convertRegionTypes(Region *region, TypeConverter &converter,
                     TypeConverter::SignatureConversion *entryConversion);

  /// Convert the types of non-entry block arguments within the given region.
  LogicalResult convertNonEntryRegionTypes(
      Region *region, TypeConverter &converter,
      ArrayRef<TypeConverter::SignatureConversion> blockConversions = {});

  //===--------------------------------------------------------------------===//
  // Rewriter Notification Hooks
  //===--------------------------------------------------------------------===//

  /// PatternRewriter hook for replacing the results of an operation.
  void notifyOpReplaced(Operation *op, ValueRange newValues);

  /// Notifies that a block is about to be erased.
  void notifyBlockIsBeingErased(Block *block);

  /// Notifies that a block was created.
  void notifyCreatedBlock(Block *block);

  /// Notifies that a block was split.
  void notifySplitBlock(Block *block, Block *continuation);

  /// Notifies that `block` is being merged with `srcBlock`.
  void notifyBlocksBeingMerged(Block *block, Block *srcBlock);

  /// Notifies that the blocks of a region are about to be moved.
  void notifyRegionIsBeingInlinedBefore(Region &region, Region &parent,
                                        Region::iterator before);

  /// Notifies that the blocks of a region were cloned into another.
  void notifyRegionWasClonedBefore(iterator_range<Region::iterator> &blocks,
                                   Location origRegionLoc);

  /// Notifies that a pattern match failed for the given reason.
  LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback);

  //===--------------------------------------------------------------------===//
  // State
  //===--------------------------------------------------------------------===//

  // Mapping between replaced values that differ in type. This happens when
  // replacing a value with one of a different type.
  ConversionValueMapping mapping;

  /// Utility used to convert block arguments.
  ArgConverter argConverter;

  /// Ordered vector of all of the newly created operations during conversion.
  std::vector<Operation *> createdOps;

  /// Ordered map of requested operation replacements.
  llvm::MapVector<Operation *, OpReplacement> replacements;

  /// Ordered vector of any requested block argument replacements.
  SmallVector<BlockArgument, 4> argReplacements;

  /// Ordered list of block operations (creations, splits, motions).
  SmallVector<BlockAction, 4> blockActions;

  /// A set of operations that should no longer be considered for legalization,
  /// but were not directly replace/erased/etc. by a pattern. These are
  /// generally child operations of other operations who were
  /// replaced/erased/etc. This is not meant to be an exhaustive list of all
  /// operations, but the minimal set that can be used to detect if a given
  /// operation should be `ignored`. For example, we may add the operations that
  /// define non-empty regions to the set, but not any of the others. This
  /// simplifies the amount of memory needed as we can query if the parent
  /// operation was ignored.
  SetVector<Operation *> ignoredOps;

  /// A transaction state for each of operations that were updated in-place.
  SmallVector<OperationTransactionState, 4> rootUpdates;

  /// A vector of indices into `replacements` of operations that were replaced
  /// with values with different result types than the original operation, e.g.
  /// 1->N conversion of some kind.
  SmallVector<unsigned, 4> operationsWithChangedResults;

  /// A default type converter, used when block conversions do not have one
  /// explicitly provided.
  TypeConverter defaultTypeConverter;

  /// The current conversion pattern that is being rewritten, or nullptr if
  /// called from outside of a conversion pattern rewrite.
  const ConversionPattern *currentConversionPattern = nullptr;

#ifndef NDEBUG
  /// A set of operations that have pending updates. This tracking isn't
  /// strictly necessary, and is thus only active during debug builds for extra
  /// verification.
  SmallPtrSet<Operation *, 1> pendingRootUpdates;

  /// A logger used to emit diagnostics during the conversion process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};
} // end namespace detail
} // end namespace mlir

/// Detach any operations nested in the given operation from their parent
/// blocks, and erase the given operation. This can be used when the nested
/// operations are scheduled for erasure themselves, so deleting the regions of
/// the given operation together with their content would result in double-free.
/// This happens, for example, when rolling back op creation in the reverse
/// order and if the nested ops were created before the parent op. This function
/// does not need to collect nested ops recursively because it is expected to
/// also be called for each nested op when it is about to be deleted.
static void detachNestedAndErase(Operation *op) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      while (!block.getOperations().empty())
        block.getOperations().remove(block.getOperations().begin());
      block.dropAllDefinedValueUses();
    }
  }
  op->dropAllUses();
  op->erase();
}

void ConversionPatternRewriterImpl::discardRewrites() {
  // Reset any operations that were updated in place.
  for (auto &state : rootUpdates)
    state.resetOperation();

  undoBlockActions();

  // Remove any newly created ops.
  for (auto *op : llvm::reverse(createdOps))
    detachNestedAndErase(op);
}

void ConversionPatternRewriterImpl::applyRewrites() {
  // Apply all of the rewrites replacements requested during conversion.
  for (auto &repl : replacements) {
    for (OpResult result : repl.first->getResults())
      if (Value newValue = mapping.lookupOrNull(result))
        result.replaceAllUsesWith(newValue);

    // If this operation defines any regions, drop any pending argument
    // rewrites.
    if (repl.first->getNumRegions())
      argConverter.notifyOpRemoved(repl.first);
  }

  // Apply all of the requested argument replacements.
  for (BlockArgument arg : argReplacements) {
    Value repl = mapping.lookupOrDefault(arg);
    if (repl.isa<BlockArgument>()) {
      arg.replaceAllUsesWith(repl);
      continue;
    }

    // If the replacement value is an operation, we check to make sure that we
    // don't replace uses that are within the parent operation of the
    // replacement value.
    Operation *replOp = repl.cast<OpResult>().getOwner();
    Block *replBlock = replOp->getBlock();
    arg.replaceUsesWithIf(repl, [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      return user->getBlock() != replBlock || replOp->isBeforeInBlock(user);
    });
  }

  // In a second pass, erase all of the replaced operations in reverse. This
  // allows processing nested operations before their parent region is
  // destroyed. Because we process in reverse order, producers may be deleted
  // before their users (a pattern deleting a producer and then the consumer)
  // so we first drop all uses explicitly.
  for (auto &repl : llvm::reverse(replacements)) {
    repl.first->dropAllUses();
    repl.first->erase();
  }

  argConverter.applyRewrites(mapping);

  // Now that the ops have been erased, also erase dangling blocks.
  eraseDanglingBlocks();
}

//===----------------------------------------------------------------------===//
// State Management

RewriterState ConversionPatternRewriterImpl::getCurrentState() {
  return RewriterState(createdOps.size(), replacements.size(),
                       argReplacements.size(), blockActions.size(),
                       ignoredOps.size(), rootUpdates.size());
}

void ConversionPatternRewriterImpl::resetState(RewriterState state) {
  // Reset any operations that were updated in place.
  for (unsigned i = state.numRootUpdates, e = rootUpdates.size(); i != e; ++i)
    rootUpdates[i].resetOperation();
  rootUpdates.resize(state.numRootUpdates);

  // Reset any replaced arguments.
  for (BlockArgument replacedArg :
       llvm::drop_begin(argReplacements, state.numArgReplacements))
    mapping.erase(replacedArg);
  argReplacements.resize(state.numArgReplacements);

  // Undo any block actions.
  undoBlockActions(state.numBlockActions);

  // Reset any replaced operations and undo any saved mappings.
  for (auto &repl : llvm::drop_begin(replacements, state.numReplacements))
    for (auto result : repl.first->getResults())
      mapping.erase(result);
  while (replacements.size() != state.numReplacements)
    replacements.pop_back();

  // Pop all of the newly created operations.
  while (createdOps.size() != state.numCreatedOps) {
    detachNestedAndErase(createdOps.back());
    createdOps.pop_back();
  }

  // Pop all of the recorded ignored operations that are no longer valid.
  while (ignoredOps.size() != state.numIgnoredOperations)
    ignoredOps.pop_back();

  // Reset operations with changed results.
  while (!operationsWithChangedResults.empty() &&
         operationsWithChangedResults.back() >= state.numReplacements)
    operationsWithChangedResults.pop_back();
}

void ConversionPatternRewriterImpl::eraseDanglingBlocks() {
  for (auto &action : blockActions)
    if (action.kind == BlockActionKind::Erase)
      delete action.block;
}

void ConversionPatternRewriterImpl::undoBlockActions(
    unsigned numActionsToKeep) {
  for (auto &action :
       llvm::reverse(llvm::drop_begin(blockActions, numActionsToKeep))) {
    switch (action.kind) {
    // Delete the created block.
    case BlockActionKind::Create: {
      // Unlink all of the operations within this block, they will be deleted
      // separately.
      auto &blockOps = action.block->getOperations();
      while (!blockOps.empty())
        blockOps.remove(blockOps.begin());
      action.block->dropAllDefinedValueUses();
      action.block->erase();
      break;
    }
    // Put the block (owned by action) back into its original position.
    case BlockActionKind::Erase: {
      auto &blockList = action.originalPosition.region->getBlocks();
      Block *insertAfterBlock = action.originalPosition.insertAfterBlock;
      blockList.insert((insertAfterBlock
                            ? std::next(Region::iterator(insertAfterBlock))
                            : blockList.begin()),
                       action.block);
      break;
    }
    // Split the block at the position which was originally the end of the
    // destination block (owned by action), and put the instructions back into
    // the block used before the merge.
    case BlockActionKind::Merge: {
      Block *sourceBlock = action.mergeInfo.sourceBlock;
      Block::iterator splitPoint =
          (action.mergeInfo.destBlockLastInst
               ? ++Block::iterator(action.mergeInfo.destBlockLastInst)
               : action.block->begin());
      sourceBlock->getOperations().splice(sourceBlock->begin(),
                                          action.block->getOperations(),
                                          splitPoint, action.block->end());
      break;
    }
    // Move the block back to its original position.
    case BlockActionKind::Move: {
      Region *originalRegion = action.originalPosition.region;
      Block *insertAfterBlock = action.originalPosition.insertAfterBlock;
      originalRegion->getBlocks().splice(
          (insertAfterBlock ? std::next(Region::iterator(insertAfterBlock))
                            : originalRegion->end()),
          action.block->getParent()->getBlocks(), action.block);
      break;
    }
    // Merge back the block that was split out.
    case BlockActionKind::Split: {
      action.originalBlock->getOperations().splice(
          action.originalBlock->end(), action.block->getOperations());
      action.block->dropAllDefinedValueUses();
      action.block->erase();
      break;
    }
    // Undo the type conversion.
    case BlockActionKind::TypeConversion: {
      argConverter.discardRewrites(action.block);
      break;
    }
    }
  }
  blockActions.resize(numActionsToKeep);
}

LogicalResult ConversionPatternRewriterImpl::remapValues(
    Location loc, PatternRewriter &rewriter, TypeConverter *converter,
    Operation::operand_range operands, SmallVectorImpl<Value> &remapped) {
  remapped.reserve(llvm::size(operands));

  SmallVector<Type, 1> legalTypes;
  for (auto it : llvm::enumerate(operands)) {
    Value operand = it.value();
    Type origType = operand.getType();

    // If a converter was provided, get the desired legal types for this
    // operand.
    Type desiredType;
    if (converter) {
      // If there is no legal conversion, fail to match this pattern.
      legalTypes.clear();
      if (failed(converter->convertType(origType, legalTypes))) {
        return notifyMatchFailure(loc, [=](Diagnostic &diag) {
          diag << "unable to convert type for operand #" << it.index()
               << ", type was " << origType;
        });
      }
      // TODO: There currently isn't any mechanism to do 1->N type conversion
      // via the PatternRewriter replacement API, so for now we just ignore it.
      if (legalTypes.size() == 1)
        desiredType = legalTypes.front();
    } else {
      // TODO: What we should do here is just set `desiredType` to `origType`
      // and then handle the necessary type conversions after the conversion
      // process has finished. Unfortunately a lot of patterns currently rely on
      // receiving the new operands even if the types change, so we keep the
      // original behavior here for now until all of the patterns relying on
      // this get updated.
    }
    Value newOperand = mapping.lookupOrDefault(operand, desiredType);

    // Handle the case where the conversion was 1->1 and the new operand type
    // isn't legal.
    Type newOperandType = newOperand.getType();
    if (converter && desiredType && newOperandType != desiredType) {
      // Attempt to materialize a conversion for this new value.
      newOperand = converter->materializeTargetConversion(
          rewriter, loc, desiredType, newOperand);
      if (!newOperand) {
        return notifyMatchFailure(loc, [=](Diagnostic &diag) {
          diag << "unable to materialize a conversion for "
                  "operand #"
               << it.index() << ", from " << newOperandType << " to "
               << desiredType;
        });
      }
    }
    remapped.push_back(newOperand);
  }
  return success();
}

bool ConversionPatternRewriterImpl::isOpIgnored(Operation *op) const {
  // Check to see if this operation was replaced or its parent ignored.
  return replacements.count(op) || ignoredOps.count(op->getParentOp());
}

void ConversionPatternRewriterImpl::markNestedOpsIgnored(Operation *op) {
  // Walk this operation and collect nested operations that define non-empty
  // regions. We mark such operations as 'ignored' so that we know we don't have
  // to convert them, or their nested ops.
  if (op->getNumRegions() == 0)
    return;
  op->walk([&](Operation *op) {
    if (llvm::any_of(op->getRegions(),
                     [](Region &region) { return !region.empty(); }))
      ignoredOps.insert(op);
  });
}

//===----------------------------------------------------------------------===//
// Type Conversion

FailureOr<Block *> ConversionPatternRewriterImpl::convertBlockSignature(
    Block *block, TypeConverter &converter,
    TypeConverter::SignatureConversion *conversion) {
  FailureOr<Block *> result =
      conversion ? argConverter.applySignatureConversion(
                       block, converter, *conversion, mapping, argReplacements)
                 : argConverter.convertSignature(block, converter, mapping,
                                                 argReplacements);
  if (Block *newBlock = result.getValue()) {
    if (newBlock != block)
      blockActions.push_back(BlockAction::getTypeConversion(newBlock));
  }
  return result;
}

Block *ConversionPatternRewriterImpl::applySignatureConversion(
    Region *region, TypeConverter::SignatureConversion &conversion,
    TypeConverter *converter) {
  if (!region->empty()) {
    return *convertBlockSignature(&region->front(),
                                  converter ? *converter : defaultTypeConverter,
                                  &conversion);
  }
  return nullptr;
}

FailureOr<Block *> ConversionPatternRewriterImpl::convertRegionTypes(
    Region *region, TypeConverter &converter,
    TypeConverter::SignatureConversion *entryConversion) {
  argConverter.setConverter(region, &converter);
  if (region->empty())
    return nullptr;

  if (failed(convertNonEntryRegionTypes(region, converter)))
    return failure();

  FailureOr<Block *> newEntry =
      convertBlockSignature(&region->front(), converter, entryConversion);
  return newEntry;
}

LogicalResult ConversionPatternRewriterImpl::convertNonEntryRegionTypes(
    Region *region, TypeConverter &converter,
    ArrayRef<TypeConverter::SignatureConversion> blockConversions) {
  argConverter.setConverter(region, &converter);
  if (region->empty())
    return success();

  // Convert the arguments of each block within the region.
  int blockIdx = 0;
  assert((blockConversions.empty() ||
          blockConversions.size() == region->getBlocks().size() - 1) &&
         "expected either to provide no SignatureConversions at all or to "
         "provide a SignatureConversion for each non-entry block");

  for (Block &block :
       llvm::make_early_inc_range(llvm::drop_begin(*region, 1))) {
    TypeConverter::SignatureConversion *blockConversion =
        blockConversions.empty()
            ? nullptr
            : const_cast<TypeConverter::SignatureConversion *>(
                  &blockConversions[blockIdx++]);

    if (failed(convertBlockSignature(&block, converter, blockConversion)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Rewriter Notification Hooks

void ConversionPatternRewriterImpl::notifyOpReplaced(Operation *op,
                                                     ValueRange newValues) {
  assert(newValues.size() == op->getNumResults());
  assert(!replacements.count(op) && "operation was already replaced");

  // Track if any of the results changed, e.g. erased and replaced with null.
  bool resultChanged = false;

  // Create mappings for each of the new result values.
  Value newValue, result;
  for (auto it : llvm::zip(newValues, op->getResults())) {
    std::tie(newValue, result) = it;
    if (!newValue) {
      resultChanged = true;
      continue;
    }
    // Remap, and check for any result type changes.
    mapping.map(result, newValue);
    resultChanged |= (newValue.getType() != result.getType());
  }
  if (resultChanged)
    operationsWithChangedResults.push_back(replacements.size());

  // Record the requested operation replacement.
  TypeConverter *converter = nullptr;
  if (currentConversionPattern)
    converter = currentConversionPattern->getTypeConverter();
  replacements.insert(std::make_pair(op, OpReplacement(converter)));

  // Mark this operation as recursively ignored so that we don't need to
  // convert any nested operations.
  markNestedOpsIgnored(op);
}

void ConversionPatternRewriterImpl::notifyBlockIsBeingErased(Block *block) {
  Region *region = block->getParent();
  Block *origPrevBlock = block->getPrevNode();
  blockActions.push_back(BlockAction::getErase(block, {region, origPrevBlock}));
}

void ConversionPatternRewriterImpl::notifyCreatedBlock(Block *block) {
  blockActions.push_back(BlockAction::getCreate(block));
}

void ConversionPatternRewriterImpl::notifySplitBlock(Block *block,
                                                     Block *continuation) {
  blockActions.push_back(BlockAction::getSplit(continuation, block));
}

void ConversionPatternRewriterImpl::notifyBlocksBeingMerged(Block *block,
                                                            Block *srcBlock) {
  blockActions.push_back(BlockAction::getMerge(block, srcBlock));
}

void ConversionPatternRewriterImpl::notifyRegionIsBeingInlinedBefore(
    Region &region, Region &parent, Region::iterator before) {
  if (region.empty())
    return;
  Block *laterBlock = &region.back();
  for (auto &earlierBlock : llvm::drop_begin(llvm::reverse(region), 1)) {
    blockActions.push_back(
        BlockAction::getMove(laterBlock, {&region, &earlierBlock}));
    laterBlock = &earlierBlock;
  }
  blockActions.push_back(BlockAction::getMove(laterBlock, {&region, nullptr}));
}

void ConversionPatternRewriterImpl::notifyRegionWasClonedBefore(
    iterator_range<Region::iterator> &blocks, Location origRegionLoc) {
  for (Block &block : blocks)
    blockActions.push_back(BlockAction::getCreate(&block));

  // Compute the conversion set for the inlined region.
  auto result = computeConversionSet(blocks, origRegionLoc, createdOps);

  // This original region has already had its conversion set computed, so there
  // shouldn't be any new failures.
  (void)result;
  assert(succeeded(result) && "expected region to have no unreachable blocks");
}

LogicalResult ConversionPatternRewriterImpl::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    logger.startLine() << "** Failure : " << diag.str() << "\n";
  });
  return failure();
}

//===----------------------------------------------------------------------===//
// ConversionPatternRewriter
//===----------------------------------------------------------------------===//

ConversionPatternRewriter::ConversionPatternRewriter(MLIRContext *ctx)
    : PatternRewriter(ctx),
      impl(new detail::ConversionPatternRewriterImpl(*this)) {}
ConversionPatternRewriter::~ConversionPatternRewriter() {}

/// PatternRewriter hook for replacing the results of an operation when the
/// given functor returns true.
void ConversionPatternRewriter::replaceOpWithIf(
    Operation *op, ValueRange newValues, bool *allUsesReplaced,
    llvm::unique_function<bool(OpOperand &) const> functor) {
  // TODO: To support this we will need to rework a bit of how replacements are
  // tracked, given that this isn't guranteed to replace all of the uses of an
  // operation. The main change is that now an operation can be replaced
  // multiple times, in parts. The current "set" based tracking is mainly useful
  // for tracking if a replaced operation should be ignored, i.e. if all of the
  // uses will be replaced.
  llvm_unreachable(
      "replaceOpWithIf is currently not supported by DialectConversion");
}

/// PatternRewriter hook for replacing the results of an operation.
void ConversionPatternRewriter::replaceOp(Operation *op, ValueRange newValues) {
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Replace : '" << op->getName() << "'(" << op << ")\n";
  });
  impl->notifyOpReplaced(op, newValues);
}

/// PatternRewriter hook for erasing a dead operation. The uses of this
/// operation *must* be made dead by the end of the conversion process,
/// otherwise an assert will be issued.
void ConversionPatternRewriter::eraseOp(Operation *op) {
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Erase   : '" << op->getName() << "'(" << op << ")\n";
  });
  SmallVector<Value, 1> nullRepls(op->getNumResults(), nullptr);
  impl->notifyOpReplaced(op, nullRepls);
}

void ConversionPatternRewriter::eraseBlock(Block *block) {
  impl->notifyBlockIsBeingErased(block);

  // Mark all ops for erasure.
  for (Operation &op : *block)
    eraseOp(&op);

  // Unlink the block from its parent region. The block is kept in the block
  // action and will be actually destroyed when rewrites are applied. This
  // allows us to keep the operations in the block live and undo the removal by
  // re-inserting the block.
  block->getParent()->getBlocks().remove(block);
}

Block *ConversionPatternRewriter::applySignatureConversion(
    Region *region, TypeConverter::SignatureConversion &conversion,
    TypeConverter *converter) {
  return impl->applySignatureConversion(region, conversion, converter);
}

FailureOr<Block *> ConversionPatternRewriter::convertRegionTypes(
    Region *region, TypeConverter &converter,
    TypeConverter::SignatureConversion *entryConversion) {
  return impl->convertRegionTypes(region, converter, entryConversion);
}

LogicalResult ConversionPatternRewriter::convertNonEntryRegionTypes(
    Region *region, TypeConverter &converter,
    ArrayRef<TypeConverter::SignatureConversion> blockConversions) {
  return impl->convertNonEntryRegionTypes(region, converter, blockConversions);
}

void ConversionPatternRewriter::replaceUsesOfBlockArgument(BlockArgument from,
                                                           Value to) {
  LLVM_DEBUG({
    Operation *parentOp = from.getOwner()->getParentOp();
    impl->logger.startLine() << "** Replace Argument : '" << from
                             << "'(in region of '" << parentOp->getName()
                             << "'(" << from.getOwner()->getParentOp() << ")\n";
  });
  impl->argReplacements.push_back(from);
  impl->mapping.map(impl->mapping.lookupOrDefault(from), to);
}

/// Return the converted value that replaces 'key'. Return 'key' if there is
/// no such a converted value.
Value ConversionPatternRewriter::getRemappedValue(Value key) {
  return impl->mapping.lookupOrDefault(key);
}

/// PatternRewriter hook for creating a new block with the given arguments.
void ConversionPatternRewriter::notifyBlockCreated(Block *block) {
  impl->notifyCreatedBlock(block);
}

/// PatternRewriter hook for splitting a block into two parts.
Block *ConversionPatternRewriter::splitBlock(Block *block,
                                             Block::iterator before) {
  auto *continuation = PatternRewriter::splitBlock(block, before);
  impl->notifySplitBlock(block, continuation);
  return continuation;
}

/// PatternRewriter hook for merging a block into another.
void ConversionPatternRewriter::mergeBlocks(Block *source, Block *dest,
                                            ValueRange argValues) {
  impl->notifyBlocksBeingMerged(dest, source);
  assert(llvm::all_of(source->getPredecessors(),
                      [dest](Block *succ) { return succ == dest; }) &&
         "expected 'source' to have no predecessors or only 'dest'");
  assert(argValues.size() == source->getNumArguments() &&
         "incorrect # of argument replacement values");
  for (auto it : llvm::zip(source->getArguments(), argValues))
    replaceUsesOfBlockArgument(std::get<0>(it), std::get<1>(it));
  dest->getOperations().splice(dest->end(), source->getOperations());
  eraseBlock(source);
}

/// PatternRewriter hook for moving blocks out of a region.
void ConversionPatternRewriter::inlineRegionBefore(Region &region,
                                                   Region &parent,
                                                   Region::iterator before) {
  impl->notifyRegionIsBeingInlinedBefore(region, parent, before);
  PatternRewriter::inlineRegionBefore(region, parent, before);
}

/// PatternRewriter hook for cloning blocks of one region into another.
void ConversionPatternRewriter::cloneRegionBefore(
    Region &region, Region &parent, Region::iterator before,
    BlockAndValueMapping &mapping) {
  if (region.empty())
    return;
  PatternRewriter::cloneRegionBefore(region, parent, before, mapping);

  // Collect the range of the cloned blocks.
  auto clonedBeginIt = mapping.lookup(&region.front())->getIterator();
  auto clonedBlocks = llvm::make_range(clonedBeginIt, before);
  impl->notifyRegionWasClonedBefore(clonedBlocks, region.getLoc());
}

/// PatternRewriter hook for creating a new operation.
void ConversionPatternRewriter::notifyOperationInserted(Operation *op) {
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Insert  : '" << op->getName() << "'(" << op << ")\n";
  });
  impl->createdOps.push_back(op);
}

/// PatternRewriter hook for updating the root operation in-place.
void ConversionPatternRewriter::startRootUpdate(Operation *op) {
#ifndef NDEBUG
  impl->pendingRootUpdates.insert(op);
#endif
  impl->rootUpdates.emplace_back(op);
}

/// PatternRewriter hook for updating the root operation in-place.
void ConversionPatternRewriter::finalizeRootUpdate(Operation *op) {
  // There is nothing to do here, we only need to track the operation at the
  // start of the update.
#ifndef NDEBUG
  assert(impl->pendingRootUpdates.erase(op) &&
         "operation did not have a pending in-place update");
#endif
}

/// PatternRewriter hook for updating the root operation in-place.
void ConversionPatternRewriter::cancelRootUpdate(Operation *op) {
#ifndef NDEBUG
  assert(impl->pendingRootUpdates.erase(op) &&
         "operation did not have a pending in-place update");
#endif
  // Erase the last update for this operation.
  auto stateHasOp = [op](const auto &it) { return it.getOperation() == op; };
  auto &rootUpdates = impl->rootUpdates;
  auto it = llvm::find_if(llvm::reverse(rootUpdates), stateHasOp);
  rootUpdates.erase(rootUpdates.begin() + (rootUpdates.rend() - it));
}

/// PatternRewriter hook for notifying match failure reasons.
LogicalResult ConversionPatternRewriter::notifyMatchFailure(
    Operation *op, function_ref<void(Diagnostic &)> reasonCallback) {
  return impl->notifyMatchFailure(op->getLoc(), reasonCallback);
}

/// Return a reference to the internal implementation.
detail::ConversionPatternRewriterImpl &ConversionPatternRewriter::getImpl() {
  return *impl;
}

//===----------------------------------------------------------------------===//
// ConversionPattern
//===----------------------------------------------------------------------===//

/// Attempt to match and rewrite the IR root at the specified operation.
LogicalResult
ConversionPattern::matchAndRewrite(Operation *op,
                                   PatternRewriter &rewriter) const {
  auto &dialectRewriter = static_cast<ConversionPatternRewriter &>(rewriter);
  auto &rewriterImpl = dialectRewriter.getImpl();

  // Track the current conversion pattern in the rewriter.
  assert(!rewriterImpl.currentConversionPattern &&
         "already inside of a pattern rewrite");
  llvm::SaveAndRestore<const ConversionPattern *> currentPatternGuard(
      rewriterImpl.currentConversionPattern, this);

  // Remap the operands of the operation.
  SmallVector<Value, 4> operands;
  if (failed(rewriterImpl.remapValues(op->getLoc(), rewriter,
                                      getTypeConverter(), op->getOperands(),
                                      operands))) {
    return failure();
  }
  return matchAndRewrite(op, operands, dialectRewriter);
}

//===----------------------------------------------------------------------===//
// OperationLegalizer
//===----------------------------------------------------------------------===//

namespace {
/// A set of rewrite patterns that can be used to legalize a given operation.
using LegalizationPatterns = SmallVector<const Pattern *, 1>;

/// This class defines a recursive operation legalizer.
class OperationLegalizer {
public:
  using LegalizationAction = ConversionTarget::LegalizationAction;

  OperationLegalizer(ConversionTarget &targetInfo,
                     const FrozenRewritePatternSet &patterns);

  /// Returns true if the given operation is known to be illegal on the target.
  bool isIllegal(Operation *op) const;

  /// Attempt to legalize the given operation. Returns success if the operation
  /// was legalized, failure otherwise.
  LogicalResult legalize(Operation *op, ConversionPatternRewriter &rewriter);

  /// Returns the conversion target in use by the legalizer.
  ConversionTarget &getTarget() { return target; }

private:
  /// Attempt to legalize the given operation by folding it.
  LogicalResult legalizeWithFold(Operation *op,
                                 ConversionPatternRewriter &rewriter);

  /// Attempt to legalize the given operation by applying a pattern. Returns
  /// success if the operation was legalized, failure otherwise.
  LogicalResult legalizeWithPattern(Operation *op,
                                    ConversionPatternRewriter &rewriter);

  /// Return true if the given pattern may be applied to the given operation,
  /// false otherwise.
  bool canApplyPattern(Operation *op, const Pattern &pattern,
                       ConversionPatternRewriter &rewriter);

  /// Legalize the resultant IR after successfully applying the given pattern.
  LogicalResult legalizePatternResult(Operation *op, const Pattern &pattern,
                                      ConversionPatternRewriter &rewriter,
                                      RewriterState &curState);

  /// Legalizes the actions registered during the execution of a pattern.
  LogicalResult legalizePatternBlockActions(Operation *op,
                                            ConversionPatternRewriter &rewriter,
                                            ConversionPatternRewriterImpl &impl,
                                            RewriterState &state,
                                            RewriterState &newState);
  LogicalResult legalizePatternCreatedOperations(
      ConversionPatternRewriter &rewriter, ConversionPatternRewriterImpl &impl,
      RewriterState &state, RewriterState &newState);
  LogicalResult legalizePatternRootUpdates(ConversionPatternRewriter &rewriter,
                                           ConversionPatternRewriterImpl &impl,
                                           RewriterState &state,
                                           RewriterState &newState);

  //===--------------------------------------------------------------------===//
  // Cost Model
  //===--------------------------------------------------------------------===//

  /// Build an optimistic legalization graph given the provided patterns. This
  /// function populates 'anyOpLegalizerPatterns' and 'legalizerPatterns' with
  /// patterns for operations that are not directly legal, but may be
  /// transitively legal for the current target given the provided patterns.
  void buildLegalizationGraph(
      LegalizationPatterns &anyOpLegalizerPatterns,
      DenseMap<OperationName, LegalizationPatterns> &legalizerPatterns);

  /// Compute the benefit of each node within the computed legalization graph.
  /// This orders the patterns within 'legalizerPatterns' based upon two
  /// criteria:
  ///  1) Prefer patterns that have the lowest legalization depth, i.e.
  ///     represent the more direct mapping to the target.
  ///  2) When comparing patterns with the same legalization depth, prefer the
  ///     pattern with the highest PatternBenefit. This allows for users to
  ///     prefer specific legalizations over others.
  void computeLegalizationGraphBenefit(
      LegalizationPatterns &anyOpLegalizerPatterns,
      DenseMap<OperationName, LegalizationPatterns> &legalizerPatterns);

  /// Compute the legalization depth when legalizing an operation of the given
  /// type.
  unsigned computeOpLegalizationDepth(
      OperationName op, DenseMap<OperationName, unsigned> &minOpPatternDepth,
      DenseMap<OperationName, LegalizationPatterns> &legalizerPatterns);

  /// Apply the conversion cost model to the given set of patterns, and return
  /// the smallest legalization depth of any of the patterns. See
  /// `computeLegalizationGraphBenefit` for the breakdown of the cost model.
  unsigned applyCostModelToPatterns(
      LegalizationPatterns &patterns,
      DenseMap<OperationName, unsigned> &minOpPatternDepth,
      DenseMap<OperationName, LegalizationPatterns> &legalizerPatterns);

  /// The current set of patterns that have been applied.
  SmallPtrSet<const Pattern *, 8> appliedPatterns;

  /// The legalization information provided by the target.
  ConversionTarget &target;

  /// The pattern applicator to use for conversions.
  PatternApplicator applicator;
};
} // namespace

OperationLegalizer::OperationLegalizer(ConversionTarget &targetInfo,
                                       const FrozenRewritePatternSet &patterns)
    : target(targetInfo), applicator(patterns) {
  // The set of patterns that can be applied to illegal operations to transform
  // them into legal ones.
  DenseMap<OperationName, LegalizationPatterns> legalizerPatterns;
  LegalizationPatterns anyOpLegalizerPatterns;

  buildLegalizationGraph(anyOpLegalizerPatterns, legalizerPatterns);
  computeLegalizationGraphBenefit(anyOpLegalizerPatterns, legalizerPatterns);
}

bool OperationLegalizer::isIllegal(Operation *op) const {
  // Check if the target explicitly marked this operation as illegal.
  return target.getOpAction(op->getName()) == LegalizationAction::Illegal;
}

LogicalResult
OperationLegalizer::legalize(Operation *op,
                             ConversionPatternRewriter &rewriter) {
#ifndef NDEBUG
  const char *logLineComment =
      "//===-------------------------------------------===//\n";

  auto &rewriterImpl = rewriter.getImpl();
#endif
  LLVM_DEBUG({
    auto &os = rewriterImpl.logger;
    os.getOStream() << "\n";
    os.startLine() << logLineComment;
    os.startLine() << "Legalizing operation : '" << op->getName() << "'(" << op
                   << ") {\n";
    os.indent();

    // If the operation has no regions, just print it here.
    if (op->getNumRegions() == 0) {
      op->print(os.startLine(), OpPrintingFlags().printGenericOpForm());
      os.getOStream() << "\n\n";
    }
  });

  // Check if this operation is legal on the target.
  if (auto legalityInfo = target.isLegal(op)) {
    LLVM_DEBUG({
      logSuccess(
          rewriterImpl.logger, "operation marked legal by the target{0}",
          legalityInfo->isRecursivelyLegal
              ? "; NOTE: operation is recursively legal; skipping internals"
              : "");
      rewriterImpl.logger.startLine() << logLineComment;
    });

    // If this operation is recursively legal, mark its children as ignored so
    // that we don't consider them for legalization.
    if (legalityInfo->isRecursivelyLegal)
      rewriter.getImpl().markNestedOpsIgnored(op);
    return success();
  }

  // Check to see if the operation is ignored and doesn't need to be converted.
  if (rewriter.getImpl().isOpIgnored(op)) {
    LLVM_DEBUG({
      logSuccess(rewriterImpl.logger,
                 "operation marked 'ignored' during conversion");
      rewriterImpl.logger.startLine() << logLineComment;
    });
    return success();
  }

  // If the operation isn't legal, try to fold it in-place.
  // TODO: Should we always try to do this, even if the op is
  // already legal?
  if (succeeded(legalizeWithFold(op, rewriter))) {
    LLVM_DEBUG({
      logSuccess(rewriterImpl.logger, "operation was folded");
      rewriterImpl.logger.startLine() << logLineComment;
    });
    return success();
  }

  // Otherwise, we need to apply a legalization pattern to this operation.
  if (succeeded(legalizeWithPattern(op, rewriter))) {
    LLVM_DEBUG({
      logSuccess(rewriterImpl.logger, "");
      rewriterImpl.logger.startLine() << logLineComment;
    });
    return success();
  }

  LLVM_DEBUG({
    logFailure(rewriterImpl.logger, "no matched legalization pattern");
    rewriterImpl.logger.startLine() << logLineComment;
  });
  return failure();
}

LogicalResult
OperationLegalizer::legalizeWithFold(Operation *op,
                                     ConversionPatternRewriter &rewriter) {
  auto &rewriterImpl = rewriter.getImpl();
  RewriterState curState = rewriterImpl.getCurrentState();

  LLVM_DEBUG({
    rewriterImpl.logger.startLine() << "* Fold {\n";
    rewriterImpl.logger.indent();
  });

  // Try to fold the operation.
  SmallVector<Value, 2> replacementValues;
  rewriter.setInsertionPoint(op);
  if (failed(rewriter.tryFold(op, replacementValues))) {
    LLVM_DEBUG(logFailure(rewriterImpl.logger, "unable to fold"));
    return failure();
  }

  // Insert a replacement for 'op' with the folded replacement values.
  rewriter.replaceOp(op, replacementValues);

  // Recursively legalize any new constant operations.
  for (unsigned i = curState.numCreatedOps, e = rewriterImpl.createdOps.size();
       i != e; ++i) {
    Operation *cstOp = rewriterImpl.createdOps[i];
    if (failed(legalize(cstOp, rewriter))) {
      LLVM_DEBUG(logFailure(rewriterImpl.logger,
                            "generated constant '{0}' was illegal",
                            cstOp->getName()));
      rewriterImpl.resetState(curState);
      return failure();
    }
  }

  LLVM_DEBUG(logSuccess(rewriterImpl.logger, ""));
  return success();
}

LogicalResult
OperationLegalizer::legalizeWithPattern(Operation *op,
                                        ConversionPatternRewriter &rewriter) {
  auto &rewriterImpl = rewriter.getImpl();

  // Functor that returns if the given pattern may be applied.
  auto canApply = [&](const Pattern &pattern) {
    return canApplyPattern(op, pattern, rewriter);
  };

  // Functor that cleans up the rewriter state after a pattern failed to match.
  RewriterState curState = rewriterImpl.getCurrentState();
  auto onFailure = [&](const Pattern &pattern) {
    LLVM_DEBUG(logFailure(rewriterImpl.logger, "pattern failed to match"));
    rewriterImpl.resetState(curState);
    appliedPatterns.erase(&pattern);
  };

  // Functor that performs additional legalization when a pattern is
  // successfully applied.
  auto onSuccess = [&](const Pattern &pattern) {
    auto result = legalizePatternResult(op, pattern, rewriter, curState);
    appliedPatterns.erase(&pattern);
    if (failed(result))
      rewriterImpl.resetState(curState);
    return result;
  };

  // Try to match and rewrite a pattern on this operation.
  return applicator.matchAndRewrite(op, rewriter, canApply, onFailure,
                                    onSuccess);
}

bool OperationLegalizer::canApplyPattern(Operation *op, const Pattern &pattern,
                                         ConversionPatternRewriter &rewriter) {
  LLVM_DEBUG({
    auto &os = rewriter.getImpl().logger;
    os.getOStream() << "\n";
    os.startLine() << "* Pattern : '" << op->getName() << " -> (";
    llvm::interleaveComma(pattern.getGeneratedOps(), llvm::dbgs());
    os.getOStream() << ")' {\n";
    os.indent();
  });

  // Ensure that we don't cycle by not allowing the same pattern to be
  // applied twice in the same recursion stack if it is not known to be safe.
  if (!pattern.hasBoundedRewriteRecursion() &&
      !appliedPatterns.insert(&pattern).second) {
    LLVM_DEBUG(
        logFailure(rewriter.getImpl().logger, "pattern was already applied"));
    return false;
  }
  return true;
}

LogicalResult
OperationLegalizer::legalizePatternResult(Operation *op, const Pattern &pattern,
                                          ConversionPatternRewriter &rewriter,
                                          RewriterState &curState) {
  auto &impl = rewriter.getImpl();

#ifndef NDEBUG
  assert(impl.pendingRootUpdates.empty() && "dangling root updates");
#endif

  // Check that the root was either replaced or updated in place.
  auto replacedRoot = [&] {
    return llvm::any_of(
        llvm::drop_begin(impl.replacements, curState.numReplacements),
        [op](auto &it) { return it.first == op; });
  };
  auto updatedRootInPlace = [&] {
    return llvm::any_of(
        llvm::drop_begin(impl.rootUpdates, curState.numRootUpdates),
        [op](auto &state) { return state.getOperation() == op; });
  };
  (void)replacedRoot;
  (void)updatedRootInPlace;
  assert((replacedRoot() || updatedRootInPlace()) &&
         "expected pattern to replace the root operation");

  // Legalize each of the actions registered during application.
  RewriterState newState = impl.getCurrentState();
  if (failed(legalizePatternBlockActions(op, rewriter, impl, curState,
                                         newState)) ||
      failed(legalizePatternRootUpdates(rewriter, impl, curState, newState)) ||
      failed(legalizePatternCreatedOperations(rewriter, impl, curState,
                                              newState))) {
    return failure();
  }

  LLVM_DEBUG(logSuccess(impl.logger, "pattern applied successfully"));
  return success();
}

LogicalResult OperationLegalizer::legalizePatternBlockActions(
    Operation *op, ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &impl, RewriterState &state,
    RewriterState &newState) {
  SmallPtrSet<Operation *, 16> operationsToIgnore;

  // If the pattern moved or created any blocks, make sure the types of block
  // arguments get legalized.
  for (int i = state.numBlockActions, e = newState.numBlockActions; i != e;
       ++i) {
    auto &action = impl.blockActions[i];
    if (action.kind == BlockActionKind::TypeConversion ||
        action.kind == BlockActionKind::Erase)
      continue;
    // Only check blocks outside of the current operation.
    Operation *parentOp = action.block->getParentOp();
    if (!parentOp || parentOp == op || action.block->getNumArguments() == 0)
      continue;

    // If the region of the block has a type converter, try to convert the block
    // directly.
    if (auto *converter =
            impl.argConverter.getConverter(action.block->getParent())) {
      if (failed(impl.convertBlockSignature(action.block, *converter))) {
        LLVM_DEBUG(logFailure(impl.logger, "failed to convert types of moved "
                                           "block"));
        return failure();
      }
      continue;
    }

    // Otherwise, check that this operation isn't one generated by this pattern.
    // This is because we will attempt to legalize the parent operation, and
    // blocks in regions created by this pattern will already be legalized later
    // on. If we haven't built the set yet, build it now.
    if (operationsToIgnore.empty()) {
      auto createdOps = ArrayRef<Operation *>(impl.createdOps)
                            .drop_front(state.numCreatedOps);
      operationsToIgnore.insert(createdOps.begin(), createdOps.end());
    }

    // If this operation should be considered for re-legalization, try it.
    if (operationsToIgnore.insert(parentOp).second &&
        failed(legalize(parentOp, rewriter))) {
      LLVM_DEBUG(logFailure(
          impl.logger, "operation '{0}'({1}) became illegal after block action",
          parentOp->getName(), parentOp));
      return failure();
    }
  }
  return success();
}
LogicalResult OperationLegalizer::legalizePatternCreatedOperations(
    ConversionPatternRewriter &rewriter, ConversionPatternRewriterImpl &impl,
    RewriterState &state, RewriterState &newState) {
  for (int i = state.numCreatedOps, e = newState.numCreatedOps; i != e; ++i) {
    Operation *op = impl.createdOps[i];
    if (failed(legalize(op, rewriter))) {
      LLVM_DEBUG(logFailure(impl.logger,
                            "generated operation '{0}'({1}) was illegal",
                            op->getName(), op));
      return failure();
    }
  }
  return success();
}
LogicalResult OperationLegalizer::legalizePatternRootUpdates(
    ConversionPatternRewriter &rewriter, ConversionPatternRewriterImpl &impl,
    RewriterState &state, RewriterState &newState) {
  for (int i = state.numRootUpdates, e = newState.numRootUpdates; i != e; ++i) {
    Operation *op = impl.rootUpdates[i].getOperation();
    if (failed(legalize(op, rewriter))) {
      LLVM_DEBUG(logFailure(impl.logger,
                            "operation updated in-place '{0}' was illegal",
                            op->getName()));
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Cost Model

void OperationLegalizer::buildLegalizationGraph(
    LegalizationPatterns &anyOpLegalizerPatterns,
    DenseMap<OperationName, LegalizationPatterns> &legalizerPatterns) {
  // A mapping between an operation and a set of operations that can be used to
  // generate it.
  DenseMap<OperationName, SmallPtrSet<OperationName, 2>> parentOps;
  // A mapping between an operation and any currently invalid patterns it has.
  DenseMap<OperationName, SmallPtrSet<const Pattern *, 2>> invalidPatterns;
  // A worklist of patterns to consider for legality.
  SetVector<const Pattern *> patternWorklist;

  // Build the mapping from operations to the parent ops that may generate them.
  applicator.walkAllPatterns([&](const Pattern &pattern) {
    Optional<OperationName> root = pattern.getRootKind();

    // If the pattern has no specific root, we can't analyze the relationship
    // between the root op and generated operations. Given that, add all such
    // patterns to the legalization set.
    if (!root) {
      anyOpLegalizerPatterns.push_back(&pattern);
      return;
    }

    // Skip operations that are always known to be legal.
    if (target.getOpAction(*root) == LegalizationAction::Legal)
      return;

    // Add this pattern to the invalid set for the root op and record this root
    // as a parent for any generated operations.
    invalidPatterns[*root].insert(&pattern);
    for (auto op : pattern.getGeneratedOps())
      parentOps[op].insert(*root);

    // Add this pattern to the worklist.
    patternWorklist.insert(&pattern);
  });

  // If there are any patterns that don't have a specific root kind, we can't
  // make direct assumptions about what operations will never be legalized.
  // Note: Technically we could, but it would require an analysis that may
  // recurse into itself. It would be better to perform this kind of filtering
  // at a higher level than here anyways.
  if (!anyOpLegalizerPatterns.empty()) {
    for (const Pattern *pattern : patternWorklist)
      legalizerPatterns[*pattern->getRootKind()].push_back(pattern);
    return;
  }

  while (!patternWorklist.empty()) {
    auto *pattern = patternWorklist.pop_back_val();

    // Check to see if any of the generated operations are invalid.
    if (llvm::any_of(pattern->getGeneratedOps(), [&](OperationName op) {
          Optional<LegalizationAction> action = target.getOpAction(op);
          return !legalizerPatterns.count(op) &&
                 (!action || action == LegalizationAction::Illegal);
        }))
      continue;

    // Otherwise, if all of the generated operation are valid, this op is now
    // legal so add all of the child patterns to the worklist.
    legalizerPatterns[*pattern->getRootKind()].push_back(pattern);
    invalidPatterns[*pattern->getRootKind()].erase(pattern);

    // Add any invalid patterns of the parent operations to see if they have now
    // become legal.
    for (auto op : parentOps[*pattern->getRootKind()])
      patternWorklist.set_union(invalidPatterns[op]);
  }
}

void OperationLegalizer::computeLegalizationGraphBenefit(
    LegalizationPatterns &anyOpLegalizerPatterns,
    DenseMap<OperationName, LegalizationPatterns> &legalizerPatterns) {
  // The smallest pattern depth, when legalizing an operation.
  DenseMap<OperationName, unsigned> minOpPatternDepth;

  // For each operation that is transitively legal, compute a cost for it.
  for (auto &opIt : legalizerPatterns)
    if (!minOpPatternDepth.count(opIt.first))
      computeOpLegalizationDepth(opIt.first, minOpPatternDepth,
                                 legalizerPatterns);

  // Apply the cost model to the patterns that can match any operation. Those
  // with a specific operation type are already resolved when computing the op
  // legalization depth.
  if (!anyOpLegalizerPatterns.empty())
    applyCostModelToPatterns(anyOpLegalizerPatterns, minOpPatternDepth,
                             legalizerPatterns);

  // Apply a cost model to the pattern applicator. We order patterns first by
  // depth then benefit. `legalizerPatterns` contains per-op patterns by
  // decreasing benefit.
  applicator.applyCostModel([&](const Pattern &pattern) {
    ArrayRef<const Pattern *> orderedPatternList;
    if (Optional<OperationName> rootName = pattern.getRootKind())
      orderedPatternList = legalizerPatterns[*rootName];
    else
      orderedPatternList = anyOpLegalizerPatterns;

    // If the pattern is not found, then it was removed and cannot be matched.
    auto it = llvm::find(orderedPatternList, &pattern);
    if (it == orderedPatternList.end())
      return PatternBenefit::impossibleToMatch();

    // Patterns found earlier in the list have higher benefit.
    return PatternBenefit(std::distance(it, orderedPatternList.end()));
  });
}

unsigned OperationLegalizer::computeOpLegalizationDepth(
    OperationName op, DenseMap<OperationName, unsigned> &minOpPatternDepth,
    DenseMap<OperationName, LegalizationPatterns> &legalizerPatterns) {
  // Check for existing depth.
  auto depthIt = minOpPatternDepth.find(op);
  if (depthIt != minOpPatternDepth.end())
    return depthIt->second;

  // If a mapping for this operation does not exist, then this operation
  // is always legal. Return 0 as the depth for a directly legal operation.
  auto opPatternsIt = legalizerPatterns.find(op);
  if (opPatternsIt == legalizerPatterns.end() || opPatternsIt->second.empty())
    return 0u;

  // Record this initial depth in case we encounter this op again when
  // recursively computing the depth.
  minOpPatternDepth.try_emplace(op, std::numeric_limits<unsigned>::max());

  // Apply the cost model to the operation patterns, and update the minimum
  // depth.
  unsigned minDepth = applyCostModelToPatterns(
      opPatternsIt->second, minOpPatternDepth, legalizerPatterns);
  minOpPatternDepth[op] = minDepth;
  return minDepth;
}

unsigned OperationLegalizer::applyCostModelToPatterns(
    LegalizationPatterns &patterns,
    DenseMap<OperationName, unsigned> &minOpPatternDepth,
    DenseMap<OperationName, LegalizationPatterns> &legalizerPatterns) {
  unsigned minDepth = std::numeric_limits<unsigned>::max();

  // Compute the depth for each pattern within the set.
  SmallVector<std::pair<const Pattern *, unsigned>, 4> patternsByDepth;
  patternsByDepth.reserve(patterns.size());
  for (const Pattern *pattern : patterns) {
    unsigned depth = 0;
    for (auto generatedOp : pattern->getGeneratedOps()) {
      unsigned generatedOpDepth = computeOpLegalizationDepth(
          generatedOp, minOpPatternDepth, legalizerPatterns);
      depth = std::max(depth, generatedOpDepth + 1);
    }
    patternsByDepth.emplace_back(pattern, depth);

    // Update the minimum depth of the pattern list.
    minDepth = std::min(minDepth, depth);
  }

  // If the operation only has one legalization pattern, there is no need to
  // sort them.
  if (patternsByDepth.size() == 1)
    return minDepth;

  // Sort the patterns by those likely to be the most beneficial.
  llvm::array_pod_sort(patternsByDepth.begin(), patternsByDepth.end(),
                       [](const std::pair<const Pattern *, unsigned> *lhs,
                          const std::pair<const Pattern *, unsigned> *rhs) {
                         // First sort by the smaller pattern legalization
                         // depth.
                         if (lhs->second != rhs->second)
                           return llvm::array_pod_sort_comparator<unsigned>(
                               &lhs->second, &rhs->second);

                         // Then sort by the larger pattern benefit.
                         auto lhsBenefit = lhs->first->getBenefit();
                         auto rhsBenefit = rhs->first->getBenefit();
                         return llvm::array_pod_sort_comparator<PatternBenefit>(
                             &rhsBenefit, &lhsBenefit);
                       });

  // Update the legalization pattern to use the new sorted list.
  patterns.clear();
  for (auto &patternIt : patternsByDepth)
    patterns.push_back(patternIt.first);
  return minDepth;
}

//===----------------------------------------------------------------------===//
// OperationConverter
//===----------------------------------------------------------------------===//
namespace {
enum OpConversionMode {
  // In this mode, the conversion will ignore failed conversions to allow
  // illegal operations to co-exist in the IR.
  Partial,

  // In this mode, all operations must be legal for the given target for the
  // conversion to succeed.
  Full,

  // In this mode, operations are analyzed for legality. No actual rewrites are
  // applied to the operations on success.
  Analysis,
};

// This class converts operations to a given conversion target via a set of
// rewrite patterns. The conversion behaves differently depending on the
// conversion mode.
struct OperationConverter {
  explicit OperationConverter(ConversionTarget &target,
                              const FrozenRewritePatternSet &patterns,
                              OpConversionMode mode,
                              DenseSet<Operation *> *trackedOps = nullptr)
      : opLegalizer(target, patterns), mode(mode), trackedOps(trackedOps) {}

  /// Converts the given operations to the conversion target.
  LogicalResult convertOperations(ArrayRef<Operation *> ops);

private:
  /// Converts an operation with the given rewriter.
  LogicalResult convert(ConversionPatternRewriter &rewriter, Operation *op);

  /// This method is called after the conversion process to legalize any
  /// remaining artifacts and complete the conversion.
  LogicalResult finalize(ConversionPatternRewriter &rewriter);

  /// Legalize the types of converted block arguments.
  LogicalResult
  legalizeConvertedArgumentTypes(ConversionPatternRewriter &rewriter,
                                 ConversionPatternRewriterImpl &rewriterImpl);

  /// Legalize an operation result that was marked as "erased".
  LogicalResult
  legalizeErasedResult(Operation *op, OpResult result,
                       ConversionPatternRewriterImpl &rewriterImpl);

  /// Legalize an operation result that was replaced with a value of a different
  /// type.
  LogicalResult
  legalizeChangedResultType(Operation *op, OpResult result, Value newValue,
                            TypeConverter *replConverter,
                            ConversionPatternRewriter &rewriter,
                            ConversionPatternRewriterImpl &rewriterImpl,
                            const BlockAndValueMapping &inverseMapping);

  /// The legalizer to use when converting operations.
  OperationLegalizer opLegalizer;

  /// The conversion mode to use when legalizing operations.
  OpConversionMode mode;

  /// A set of pre-existing operations. When mode == OpConversionMode::Analysis,
  /// this is populated with ops found to be legalizable to the target.
  /// When mode == OpConversionMode::Partial, this is populated with ops found
  /// *not* to be legalizable to the target.
  DenseSet<Operation *> *trackedOps;
};
} // end anonymous namespace

LogicalResult OperationConverter::convert(ConversionPatternRewriter &rewriter,
                                          Operation *op) {
  // Legalize the given operation.
  if (failed(opLegalizer.legalize(op, rewriter))) {
    // Handle the case of a failed conversion for each of the different modes.
    // Full conversions expect all operations to be converted.
    if (mode == OpConversionMode::Full)
      return op->emitError()
             << "failed to legalize operation '" << op->getName() << "'";
    // Partial conversions allow conversions to fail iff the operation was not
    // explicitly marked as illegal. If the user provided a nonlegalizableOps
    // set, non-legalizable ops are included.
    if (mode == OpConversionMode::Partial) {
      if (opLegalizer.isIllegal(op))
        return op->emitError()
               << "failed to legalize operation '" << op->getName()
               << "' that was explicitly marked illegal";
      if (trackedOps)
        trackedOps->insert(op);
    }
  } else if (mode == OpConversionMode::Analysis) {
    // Analysis conversions don't fail if any operations fail to legalize,
    // they are only interested in the operations that were successfully
    // legalized.
    trackedOps->insert(op);
  }
  return success();
}

LogicalResult OperationConverter::convertOperations(ArrayRef<Operation *> ops) {
  if (ops.empty())
    return success();
  ConversionTarget &target = opLegalizer.getTarget();

  // Compute the set of operations and blocks to convert.
  std::vector<Operation *> toConvert;
  for (auto *op : ops) {
    toConvert.emplace_back(op);
    for (auto &region : op->getRegions())
      if (failed(computeConversionSet(region.getBlocks(), region.getLoc(),
                                      toConvert, &target)))
        return failure();
  }

  // Convert each operation and discard rewrites on failure.
  ConversionPatternRewriter rewriter(ops.front()->getContext());
  ConversionPatternRewriterImpl &rewriterImpl = rewriter.getImpl();
  for (auto *op : toConvert)
    if (failed(convert(rewriter, op)))
      return rewriterImpl.discardRewrites(), failure();

  // Now that all of the operations have been converted, finalize the conversion
  // process to ensure any lingering conversion artifacts are cleaned up and
  // legalized.
  if (failed(finalize(rewriter)))
    return rewriterImpl.discardRewrites(), failure();
  // After a successful conversion, apply rewrites if this is not an analysis
  // conversion.
  if (mode == OpConversionMode::Analysis)
    rewriterImpl.discardRewrites();
  else {
    rewriterImpl.applyRewrites();

    // It is possible for a later pattern to erase an op that was originally
    // identified as illegal and added to the trackedOps, remove it now after
    // replacements have been computed.
    if (trackedOps)
      for (auto &repl : rewriterImpl.replacements)
        trackedOps->erase(repl.first);
  }
  return success();
}

LogicalResult
OperationConverter::finalize(ConversionPatternRewriter &rewriter) {
  ConversionPatternRewriterImpl &rewriterImpl = rewriter.getImpl();

  // Legalize converted block arguments.
  if (failed(legalizeConvertedArgumentTypes(rewriter, rewriterImpl)))
    return failure();

  if (rewriterImpl.operationsWithChangedResults.empty())
    return success();

  Optional<BlockAndValueMapping> inverseMapping;

  // Process requested operation replacements.
  for (unsigned i = 0, e = rewriterImpl.operationsWithChangedResults.size();
       i != e; ++i) {
    unsigned replIdx = rewriterImpl.operationsWithChangedResults[i];
    auto &repl = *(rewriterImpl.replacements.begin() + replIdx);
    for (OpResult result : repl.first->getResults()) {
      Value newValue = rewriterImpl.mapping.lookupOrNull(result);

      // If the operation result was replaced with null, all of the uses of this
      // value should be replaced.
      if (!newValue) {
        if (failed(legalizeErasedResult(repl.first, result, rewriterImpl)))
          return failure();
        continue;
      }

      // Otherwise, check to see if the type of the result changed.
      if (result.getType() == newValue.getType())
        continue;

      // Compute the inverse mapping only if it is really needed.
      if (!inverseMapping)
        inverseMapping = rewriterImpl.mapping.getInverse();

      // Legalize this result.
      rewriter.setInsertionPoint(repl.first);
      if (failed(legalizeChangedResultType(repl.first, result, newValue,
                                           repl.second.converter, rewriter,
                                           rewriterImpl, *inverseMapping)))
        return failure();

      // Update the end iterator for this loop in the case it was updated
      // when legalizing generated conversion operations.
      e = rewriterImpl.operationsWithChangedResults.size();
    }
  }
  return success();
}

LogicalResult OperationConverter::legalizeConvertedArgumentTypes(
    ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &rewriterImpl) {
  // Functor used to check if all users of a value will be dead after
  // conversion.
  auto findLiveUser = [&](Value val) {
    auto liveUserIt = llvm::find_if_not(val.getUsers(), [&](Operation *user) {
      return rewriterImpl.isOpIgnored(user);
    });
    return liveUserIt == val.user_end() ? nullptr : *liveUserIt;
  };

  // Materialize any necessary conversions for converted block arguments that
  // are still live.
  size_t numCreatedOps = rewriterImpl.createdOps.size();
  if (failed(rewriterImpl.argConverter.materializeLiveConversions(
          rewriterImpl.mapping, rewriter, findLiveUser)))
    return failure();

  // Legalize any newly created operations during argument materialization.
  for (int i : llvm::seq<int>(numCreatedOps, rewriterImpl.createdOps.size())) {
    if (failed(opLegalizer.legalize(rewriterImpl.createdOps[i], rewriter))) {
      return rewriterImpl.createdOps[i]->emitError()
             << "failed to legalize conversion operation generated for block "
                "argument that remained live after conversion";
    }
  }
  return success();
}

LogicalResult OperationConverter::legalizeErasedResult(
    Operation *op, OpResult result,
    ConversionPatternRewriterImpl &rewriterImpl) {
  // If the operation result was replaced with null, all of the uses of this
  // value should be replaced.
  auto liveUserIt = llvm::find_if_not(result.getUsers(), [&](Operation *user) {
    return rewriterImpl.isOpIgnored(user);
  });
  if (liveUserIt != result.user_end()) {
    InFlightDiagnostic diag = op->emitError("failed to legalize operation '")
                              << op->getName() << "' marked as erased";
    diag.attachNote(liveUserIt->getLoc())
        << "found live user of result #" << result.getResultNumber() << ": "
        << *liveUserIt;
    return failure();
  }
  return success();
}

/// Finds a user of the given value, or of any other value that the given value
/// replaced, that was not replaced in the conversion process.
static Operation *
findLiveUserOfReplaced(Value value, ConversionPatternRewriterImpl &rewriterImpl,
                       const BlockAndValueMapping &inverseMapping) {
  do {
    // Walk the users of this value to see if there are any live users that
    // weren't replaced during conversion.
    auto liveUserIt = llvm::find_if_not(value.getUsers(), [&](Operation *user) {
      return rewriterImpl.isOpIgnored(user);
    });
    if (liveUserIt != value.user_end())
      return *liveUserIt;
    value = inverseMapping.lookupOrNull(value);
  } while (value != nullptr);
  return nullptr;
}

LogicalResult OperationConverter::legalizeChangedResultType(
    Operation *op, OpResult result, Value newValue,
    TypeConverter *replConverter, ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &rewriterImpl,
    const BlockAndValueMapping &inverseMapping) {
  Operation *liveUser =
      findLiveUserOfReplaced(result, rewriterImpl, inverseMapping);
  if (!liveUser)
    return success();

  // If the replacement has a type converter, attempt to materialize a
  // conversion back to the original type.
  if (!replConverter) {
    // TODO: We should emit an error here, similarly to the case where the
    // result is replaced with null. Unfortunately a lot of existing
    // patterns rely on this behavior, so until those patterns are updated
    // we keep the legacy behavior here of just forwarding the new value.
    return success();
  }

  // Track the number of created operations so that new ones can be legalized.
  size_t numCreatedOps = rewriterImpl.createdOps.size();

  // Materialize a conversion for this live result value.
  Type resultType = result.getType();
  Value convertedValue = replConverter->materializeSourceConversion(
      rewriter, op->getLoc(), resultType, newValue);
  if (!convertedValue) {
    InFlightDiagnostic diag = op->emitError()
                              << "failed to materialize conversion for result #"
                              << result.getResultNumber() << " of operation '"
                              << op->getName()
                              << "' that remained live after conversion";
    diag.attachNote(liveUser->getLoc())
        << "see existing live user here: " << *liveUser;
    return failure();
  }

  // Legalize all of the newly created conversion operations.
  for (int i : llvm::seq<int>(numCreatedOps, rewriterImpl.createdOps.size())) {
    if (failed(opLegalizer.legalize(rewriterImpl.createdOps[i], rewriter))) {
      return op->emitError("failed to legalize conversion operation generated ")
             << "for result #" << result.getResultNumber() << " of operation '"
             << op->getName() << "' that remained live after conversion";
    }
  }

  rewriterImpl.mapping.map(result, convertedValue);
  return success();
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Remap an input of the original signature with a new set of types. The
/// new types are appended to the new signature conversion.
void TypeConverter::SignatureConversion::addInputs(unsigned origInputNo,
                                                   ArrayRef<Type> types) {
  assert(!types.empty() && "expected valid types");
  remapInput(origInputNo, /*newInputNo=*/argTypes.size(), types.size());
  addInputs(types);
}

/// Append new input types to the signature conversion, this should only be
/// used if the new types are not intended to remap an existing input.
void TypeConverter::SignatureConversion::addInputs(ArrayRef<Type> types) {
  assert(!types.empty() &&
         "1->0 type remappings don't need to be added explicitly");
  argTypes.append(types.begin(), types.end());
}

/// Remap an input of the original signature with a range of types in the
/// new signature.
void TypeConverter::SignatureConversion::remapInput(unsigned origInputNo,
                                                    unsigned newInputNo,
                                                    unsigned newInputCount) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  assert(newInputCount != 0 && "expected valid input count");
  remappedInputs[origInputNo] =
      InputMapping{newInputNo, newInputCount, /*replacementValue=*/nullptr};
}

/// Remap an input of the original signature to another `replacementValue`
/// value. This would make the signature converter drop this argument.
void TypeConverter::SignatureConversion::remapInput(unsigned origInputNo,
                                                    Value replacementValue) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  remappedInputs[origInputNo] =
      InputMapping{origInputNo, /*size=*/0, replacementValue};
}

/// This hooks allows for converting a type.
LogicalResult TypeConverter::convertType(Type t,
                                         SmallVectorImpl<Type> &results) {
  auto existingIt = cachedDirectConversions.find(t);
  if (existingIt != cachedDirectConversions.end()) {
    if (existingIt->second)
      results.push_back(existingIt->second);
    return success(existingIt->second != nullptr);
  }
  auto multiIt = cachedMultiConversions.find(t);
  if (multiIt != cachedMultiConversions.end()) {
    results.append(multiIt->second.begin(), multiIt->second.end());
    return success();
  }

  // Walk the added converters in reverse order to apply the most recently
  // registered first.
  size_t currentCount = results.size();
  for (ConversionCallbackFn &converter : llvm::reverse(conversions)) {
    if (Optional<LogicalResult> result = converter(t, results)) {
      if (!succeeded(*result)) {
        cachedDirectConversions.try_emplace(t, nullptr);
        return failure();
      }
      auto newTypes = ArrayRef<Type>(results).drop_front(currentCount);
      if (newTypes.size() == 1)
        cachedDirectConversions.try_emplace(t, newTypes.front());
      else
        cachedMultiConversions.try_emplace(t, llvm::to_vector<2>(newTypes));
      return success();
    }
  }
  return failure();
}

/// This hook simplifies defining 1-1 type conversions. This function returns
/// the type to convert to on success, and a null type on failure.
Type TypeConverter::convertType(Type t) {
  // Use the multi-type result version to convert the type.
  SmallVector<Type, 1> results;
  if (failed(convertType(t, results)))
    return nullptr;

  // Check to ensure that only one type was produced.
  return results.size() == 1 ? results.front() : nullptr;
}

/// Convert the given set of types, filling 'results' as necessary. This
/// returns failure if the conversion of any of the types fails, success
/// otherwise.
LogicalResult TypeConverter::convertTypes(TypeRange types,
                                          SmallVectorImpl<Type> &results) {
  for (Type type : types)
    if (failed(convertType(type, results)))
      return failure();
  return success();
}

/// Return true if the given type is legal for this type converter, i.e. the
/// type converts to itself.
bool TypeConverter::isLegal(Type type) { return convertType(type) == type; }
/// Return true if the given operation has legal operand and result types.
bool TypeConverter::isLegal(Operation *op) {
  return isLegal(op->getOperandTypes()) && isLegal(op->getResultTypes());
}

/// Return true if the types of block arguments within the region are legal.
bool TypeConverter::isLegal(Region *region) {
  return llvm::all_of(*region, [this](Block &block) {
    return isLegal(block.getArgumentTypes());
  });
}

/// Return true if the inputs and outputs of the given function type are
/// legal.
bool TypeConverter::isSignatureLegal(FunctionType ty) {
  return isLegal(llvm::concat<const Type>(ty.getInputs(), ty.getResults()));
}

/// This hook allows for converting a specific argument of a signature.
LogicalResult TypeConverter::convertSignatureArg(unsigned inputNo, Type type,
                                                 SignatureConversion &result) {
  // Try to convert the given input type.
  SmallVector<Type, 1> convertedTypes;
  if (failed(convertType(type, convertedTypes)))
    return failure();

  // If this argument is being dropped, there is nothing left to do.
  if (convertedTypes.empty())
    return success();

  // Otherwise, add the new inputs.
  result.addInputs(inputNo, convertedTypes);
  return success();
}
LogicalResult TypeConverter::convertSignatureArgs(TypeRange types,
                                                  SignatureConversion &result,
                                                  unsigned origInputOffset) {
  for (unsigned i = 0, e = types.size(); i != e; ++i)
    if (failed(convertSignatureArg(origInputOffset + i, types[i], result)))
      return failure();
  return success();
}

Value TypeConverter::materializeConversion(
    MutableArrayRef<MaterializationCallbackFn> materializations,
    OpBuilder &builder, Location loc, Type resultType, ValueRange inputs) {
  for (MaterializationCallbackFn &fn : llvm::reverse(materializations))
    if (Optional<Value> result = fn(builder, resultType, inputs, loc))
      return result.getValue();
  return nullptr;
}

/// This function converts the type signature of the given block, by invoking
/// 'convertSignatureArg' for each argument. This function should return a valid
/// conversion for the signature on success, None otherwise.
auto TypeConverter::convertBlockSignature(Block *block)
    -> Optional<SignatureConversion> {
  SignatureConversion conversion(block->getNumArguments());
  if (failed(convertSignatureArgs(block->getArgumentTypes(), conversion)))
    return llvm::None;
  return conversion;
}

/// Create a default conversion pattern that rewrites the type signature of a
/// FunctionLike op. This only supports FunctionLike ops which use FunctionType
/// to represent their type.
namespace {
struct FunctionLikeSignatureConversion : public ConversionPattern {
  FunctionLikeSignatureConversion(StringRef functionLikeOpName,
                                  MLIRContext *ctx, TypeConverter &converter)
      : ConversionPattern(converter, functionLikeOpName, /*benefit=*/1, ctx) {}

  /// Hook to implement combined matching and rewriting for FunctionLike ops.
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = function_like_impl::getFunctionType(op);

    // Convert the original function types.
    TypeConverter::SignatureConversion result(type.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(typeConverter->convertSignatureArgs(type.getInputs(), result)) ||
        failed(typeConverter->convertTypes(type.getResults(), newResults)) ||
        failed(rewriter.convertRegionTypes(
            &function_like_impl::getFunctionBody(op), *typeConverter, &result)))
      return failure();

    // Update the function signature in-place.
    auto newType = FunctionType::get(rewriter.getContext(),
                                     result.getConvertedTypes(), newResults);

    rewriter.updateRootInPlace(
        op, [&] { function_like_impl::setFunctionType(op, newType); });

    return success();
  }
};
} // end anonymous namespace

void mlir::populateFunctionLikeTypeConversionPattern(
    StringRef functionLikeOpName, RewritePatternSet &patterns,
    TypeConverter &converter) {
  patterns.add<FunctionLikeSignatureConversion>(
      functionLikeOpName, patterns.getContext(), converter);
}

void mlir::populateFuncOpTypeConversionPattern(RewritePatternSet &patterns,
                                               TypeConverter &converter) {
  populateFunctionLikeTypeConversionPattern<FuncOp>(patterns, converter);
}

//===----------------------------------------------------------------------===//
// ConversionTarget
//===----------------------------------------------------------------------===//

/// Register a legality action for the given operation.
void ConversionTarget::setOpAction(OperationName op,
                                   LegalizationAction action) {
  legalOperations[op] = {action, /*isRecursivelyLegal=*/false, llvm::None};
}

/// Register a legality action for the given dialects.
void ConversionTarget::setDialectAction(ArrayRef<StringRef> dialectNames,
                                        LegalizationAction action) {
  for (StringRef dialect : dialectNames)
    legalDialects[dialect] = action;
}

/// Get the legality action for the given operation.
auto ConversionTarget::getOpAction(OperationName op) const
    -> Optional<LegalizationAction> {
  Optional<LegalizationInfo> info = getOpInfo(op);
  return info ? info->action : Optional<LegalizationAction>();
}

/// If the given operation instance is legal on this target, a structure
/// containing legality information is returned. If the operation is not legal,
/// None is returned.
auto ConversionTarget::isLegal(Operation *op) const
    -> Optional<LegalOpDetails> {
  Optional<LegalizationInfo> info = getOpInfo(op->getName());
  if (!info)
    return llvm::None;

  // Returns true if this operation instance is known to be legal.
  auto isOpLegal = [&] {
    // Handle dynamic legality either with the provided legality function, or
    // the default hook on the derived instance.
    if (info->action == LegalizationAction::Dynamic)
      return info->legalityFn ? (*info->legalityFn)(op)
                              : isDynamicallyLegal(op);

    // Otherwise, the operation is only legal if it was marked 'Legal'.
    return info->action == LegalizationAction::Legal;
  };
  if (!isOpLegal())
    return llvm::None;

  // This operation is legal, compute any additional legality information.
  LegalOpDetails legalityDetails;
  if (info->isRecursivelyLegal) {
    auto legalityFnIt = opRecursiveLegalityFns.find(op->getName());
    if (legalityFnIt != opRecursiveLegalityFns.end())
      legalityDetails.isRecursivelyLegal = legalityFnIt->second(op);
    else
      legalityDetails.isRecursivelyLegal = true;
  }
  return legalityDetails;
}

/// Set the dynamic legality callback for the given operation.
void ConversionTarget::setLegalityCallback(
    OperationName name, const DynamicLegalityCallbackFn &callback) {
  assert(callback && "expected valid legality callback");
  auto infoIt = legalOperations.find(name);
  assert(infoIt != legalOperations.end() &&
         infoIt->second.action == LegalizationAction::Dynamic &&
         "expected operation to already be marked as dynamically legal");
  infoIt->second.legalityFn = callback;
}

/// Set the recursive legality callback for the given operation and mark the
/// operation as recursively legal.
void ConversionTarget::markOpRecursivelyLegal(
    OperationName name, const DynamicLegalityCallbackFn &callback) {
  auto infoIt = legalOperations.find(name);
  assert(infoIt != legalOperations.end() &&
         infoIt->second.action != LegalizationAction::Illegal &&
         "expected operation to already be marked as legal");
  infoIt->second.isRecursivelyLegal = true;
  if (callback)
    opRecursiveLegalityFns[name] = callback;
  else
    opRecursiveLegalityFns.erase(name);
}

/// Set the dynamic legality callback for the given dialects.
void ConversionTarget::setLegalityCallback(
    ArrayRef<StringRef> dialects, const DynamicLegalityCallbackFn &callback) {
  assert(callback && "expected valid legality callback");
  for (StringRef dialect : dialects)
    dialectLegalityFns[dialect] = callback;
}

/// Get the legalization information for the given operation.
auto ConversionTarget::getOpInfo(OperationName op) const
    -> Optional<LegalizationInfo> {
  // Check for info for this specific operation.
  auto it = legalOperations.find(op);
  if (it != legalOperations.end())
    return it->second;
  // Check for info for the parent dialect.
  auto dialectIt = legalDialects.find(op.getDialectNamespace());
  if (dialectIt != legalDialects.end()) {
    Optional<DynamicLegalityCallbackFn> callback;
    auto dialectFn = dialectLegalityFns.find(op.getDialectNamespace());
    if (dialectFn != dialectLegalityFns.end())
      callback = dialectFn->second;
    return LegalizationInfo{dialectIt->second, /*isRecursivelyLegal=*/false,
                            callback};
  }
  // Otherwise, check if we mark unknown operations as dynamic.
  if (unknownOpsDynamicallyLegal)
    return LegalizationInfo{LegalizationAction::Dynamic,
                            /*isRecursivelyLegal=*/false, unknownLegalityFn};
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// Op Conversion Entry Points
//===----------------------------------------------------------------------===//

/// Apply a partial conversion on the given operations and all nested
/// operations. This method converts as many operations to the target as
/// possible, ignoring operations that failed to legalize. This method only
/// returns failure if there ops explicitly marked as illegal.
/// If an `unconvertedOps` set is provided, all operations that are found not
/// to be legalizable to the given `target` are placed within that set. (Note
/// that if there is an op explicitly marked as illegal, the conversion
/// terminates and the `unconvertedOps` set will not necessarily be complete.)
LogicalResult
mlir::applyPartialConversion(ArrayRef<Operation *> ops,
                             ConversionTarget &target,
                             const FrozenRewritePatternSet &patterns,
                             DenseSet<Operation *> *unconvertedOps) {
  OperationConverter opConverter(target, patterns, OpConversionMode::Partial,
                                 unconvertedOps);
  return opConverter.convertOperations(ops);
}
LogicalResult
mlir::applyPartialConversion(Operation *op, ConversionTarget &target,
                             const FrozenRewritePatternSet &patterns,
                             DenseSet<Operation *> *unconvertedOps) {
  return applyPartialConversion(llvm::makeArrayRef(op), target, patterns,
                                unconvertedOps);
}

/// Apply a complete conversion on the given operations, and all nested
/// operations. This method will return failure if the conversion of any
/// operation fails.
LogicalResult
mlir::applyFullConversion(ArrayRef<Operation *> ops, ConversionTarget &target,
                          const FrozenRewritePatternSet &patterns) {
  OperationConverter opConverter(target, patterns, OpConversionMode::Full);
  return opConverter.convertOperations(ops);
}
LogicalResult
mlir::applyFullConversion(Operation *op, ConversionTarget &target,
                          const FrozenRewritePatternSet &patterns) {
  return applyFullConversion(llvm::makeArrayRef(op), target, patterns);
}

/// Apply an analysis conversion on the given operations, and all nested
/// operations. This method analyzes which operations would be successfully
/// converted to the target if a conversion was applied. All operations that
/// were found to be legalizable to the given 'target' are placed within the
/// provided 'convertedOps' set; note that no actual rewrites are applied to the
/// operations on success and only pre-existing operations are added to the set.
LogicalResult
mlir::applyAnalysisConversion(ArrayRef<Operation *> ops,
                              ConversionTarget &target,
                              const FrozenRewritePatternSet &patterns,
                              DenseSet<Operation *> &convertedOps) {
  OperationConverter opConverter(target, patterns, OpConversionMode::Analysis,
                                 &convertedOps);
  return opConverter.convertOperations(ops);
}
LogicalResult
mlir::applyAnalysisConversion(Operation *op, ConversionTarget &target,
                              const FrozenRewritePatternSet &patterns,
                              DenseSet<Operation *> &convertedOps) {
  return applyAnalysisConversion(llvm::makeArrayRef(op), target, patterns,
                                 convertedOps);
}
