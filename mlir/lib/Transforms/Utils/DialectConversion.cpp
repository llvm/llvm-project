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
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/ScopeExit.h"
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
                     Location regionLoc,
                     SmallVectorImpl<Operation *> &toConvert,
                     ConversionTarget *target = nullptr) {
  if (region.empty())
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
      auto legalityInfo =
          target ? target->isLegal(&op)
                 : std::optional<ConversionTarget::LegalOpDetails>();
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
  Value lookupOrNull(Value from, Type desiredType = nullptr) const;

  /// Map a value to the one provided.
  void map(Value oldVal, Value newVal) {
    LLVM_DEBUG({
      for (Value it = newVal; it; it = mapping.lookupOrNull(it))
        assert(it != oldVal && "inserting cyclic mapping");
    });
    mapping.map(oldVal, newVal);
  }

  /// Try to map a value to the one provided. Returns false if a transitive
  /// mapping from the new value to the old value already exists, true if the
  /// map was updated.
  bool tryMap(Value oldVal, Value newVal);

  /// Drop the last mapping for the given value.
  void erase(Value value) { mapping.erase(value); }

  /// Returns the inverse raw value mapping (without recursive query support).
  DenseMap<Value, SmallVector<Value>> getInverse() const {
    DenseMap<Value, SmallVector<Value>> inverse;
    for (auto &it : mapping.getValueMap())
      inverse[it.second].push_back(it.first);
    return inverse;
  }

private:
  /// Current value mappings.
  BlockAndValueMapping mapping;
};
} // namespace

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

Value ConversionValueMapping::lookupOrNull(Value from, Type desiredType) const {
  Value result = lookupOrDefault(from, desiredType);
  if (result == from || (desiredType && result.getType() != desiredType))
    return nullptr;
  return result;
}

bool ConversionValueMapping::tryMap(Value oldVal, Value newVal) {
  for (Value it = newVal; it; it = mapping.lookupOrNull(it))
    if (it == oldVal)
      return false;
  map(oldVal, newVal);
  return true;
}

//===----------------------------------------------------------------------===//
// Rewriter and Translation State
//===----------------------------------------------------------------------===//
namespace {
/// This class contains a snapshot of the current conversion rewriter state.
/// This is useful when saving and undoing a set of rewrites.
struct RewriterState {
  RewriterState(unsigned numCreatedOps, unsigned numUnresolvedMaterializations,
                unsigned numReplacements, unsigned numArgReplacements,
                unsigned numBlockActions, unsigned numIgnoredOperations,
                unsigned numRootUpdates)
      : numCreatedOps(numCreatedOps),
        numUnresolvedMaterializations(numUnresolvedMaterializations),
        numReplacements(numReplacements),
        numArgReplacements(numArgReplacements),
        numBlockActions(numBlockActions),
        numIgnoredOperations(numIgnoredOperations),
        numRootUpdates(numRootUpdates) {}

  /// The current number of created operations.
  unsigned numCreatedOps;

  /// The current number of unresolved materializations.
  unsigned numUnresolvedMaterializations;

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

//===----------------------------------------------------------------------===//
// OperationTransactionState

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
    for (const auto &it : llvm::enumerate(successors))
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

//===----------------------------------------------------------------------===//
// OpReplacement

/// This class represents one requested operation replacement via 'replaceOp' or
/// 'eraseOp`.
struct OpReplacement {
  OpReplacement(TypeConverter *converter = nullptr) : converter(converter) {}

  /// An optional type converter that can be used to materialize conversions
  /// between the new and old values if necessary.
  TypeConverter *converter;
};

//===----------------------------------------------------------------------===//
// BlockAction

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

//===----------------------------------------------------------------------===//
// UnresolvedMaterialization

/// This class represents an unresolved materialization, i.e. a materialization
/// that was inserted during conversion that needs to be legalized at the end of
/// the conversion process.
class UnresolvedMaterialization {
public:
  /// The type of materialization.
  enum Kind {
    /// This materialization materializes a conversion for an illegal block
    /// argument type, to a legal one.
    Argument,

    /// This materialization materializes a conversion from an illegal type to a
    /// legal one.
    Target
  };

  UnresolvedMaterialization(UnrealizedConversionCastOp op = nullptr,
                            TypeConverter *converter = nullptr,
                            Kind kind = Target, Type origOutputType = nullptr)
      : op(op), converterAndKind(converter, kind),
        origOutputType(origOutputType) {}

  /// Return the temporary conversion operation inserted for this
  /// materialization.
  UnrealizedConversionCastOp getOp() const { return op; }

  /// Return the type converter of this materialization (which may be null).
  TypeConverter *getConverter() const { return converterAndKind.getPointer(); }

  /// Return the kind of this materialization.
  Kind getKind() const { return converterAndKind.getInt(); }

  /// Set the kind of this materialization.
  void setKind(Kind kind) { converterAndKind.setInt(kind); }

  /// Return the original illegal output type of the input values.
  Type getOrigOutputType() const { return origOutputType; }

private:
  /// The unresolved materialization operation created during conversion.
  UnrealizedConversionCastOp op;

  /// The corresponding type converter to use when resolving this
  /// materialization, and the kind of this materialization.
  llvm::PointerIntPair<TypeConverter *, 1, Kind> converterAndKind;

  /// The original output type. This is only used for argument conversions.
  Type origOutputType;
};
} // namespace

/// Build an unresolved materialization operation given an output type and set
/// of input operands.
static Value buildUnresolvedMaterialization(
    UnresolvedMaterialization::Kind kind, Block *insertBlock,
    Block::iterator insertPt, Location loc, ValueRange inputs, Type outputType,
    Type origOutputType, TypeConverter *converter,
    SmallVectorImpl<UnresolvedMaterialization> &unresolvedMaterializations) {
  // Avoid materializing an unnecessary cast.
  if (inputs.size() == 1 && inputs.front().getType() == outputType)
    return inputs.front();

  // Create an unresolved materialization. We use a new OpBuilder to avoid
  // tracking the materialization like we do for other operations.
  OpBuilder builder(insertBlock, insertPt);
  auto convertOp =
      builder.create<UnrealizedConversionCastOp>(loc, outputType, inputs);
  unresolvedMaterializations.emplace_back(convertOp, converter, kind,
                                          origOutputType);
  return convertOp.getResult(0);
}
static Value buildUnresolvedArgumentMaterialization(
    PatternRewriter &rewriter, Location loc, ValueRange inputs,
    Type origOutputType, Type outputType, TypeConverter *converter,
    SmallVectorImpl<UnresolvedMaterialization> &unresolvedMaterializations) {
  return buildUnresolvedMaterialization(
      UnresolvedMaterialization::Argument, rewriter.getInsertionBlock(),
      rewriter.getInsertionPoint(), loc, inputs, outputType, origOutputType,
      converter, unresolvedMaterializations);
}
static Value buildUnresolvedTargetMaterialization(
    Location loc, Value input, Type outputType, TypeConverter *converter,
    SmallVectorImpl<UnresolvedMaterialization> &unresolvedMaterializations) {
  Block *insertBlock = input.getParentBlock();
  Block::iterator insertPt = insertBlock->begin();
  if (OpResult inputRes = input.dyn_cast<OpResult>())
    insertPt = ++inputRes.getOwner()->getIterator();

  return buildUnresolvedMaterialization(
      UnresolvedMaterialization::Target, insertBlock, insertPt, loc, input,
      outputType, outputType, converter, unresolvedMaterializations);
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
  ArgConverter(
      PatternRewriter &rewriter,
      SmallVectorImpl<UnresolvedMaterialization> &unresolvedMaterializations)
      : rewriter(rewriter),
        unresolvedMaterializations(unresolvedMaterializations) {}

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
    ConvertedBlockInfo(Block *origBlock, TypeConverter *converter)
        : origBlock(origBlock), converter(converter) {}

    /// The original block that was requested to have its signature converted.
    Block *origBlock;

    /// The conversion information for each of the arguments. The information is
    /// std::nullopt if the argument was dropped during conversion.
    SmallVector<std::optional<ConvertedArgInfo>, 1> argInfo;

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
  convertSignature(Block *block, TypeConverter *converter,
                   ConversionValueMapping &mapping,
                   SmallVectorImpl<BlockArgument> &argReplacements);

  /// Apply the given signature conversion on the given block. The new block
  /// containing the updated signature is returned. If no conversions were
  /// necessary, e.g. if the block has no arguments, `block` is returned.
  /// `converter` is used to generate any necessary cast operations that
  /// translate between the origin argument types and those specified in the
  /// signature conversion.
  Block *applySignatureConversion(
      Block *block, TypeConverter *converter,
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

  /// An ordered set of unresolved materializations during conversion.
  SmallVectorImpl<UnresolvedMaterialization> &unresolvedMaterializations;
};
} // namespace

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
      std::optional<ConvertedArgInfo> &argInfo = blockInfo.argInfo[i];
      BlockArgument origArg = origBlock->getArgument(i);

      // Handle the case of a 1->0 value mapping.
      if (!argInfo) {
        if (Value newArg = mapping.lookupOrNull(origArg, origArg.getType()))
          origArg.replaceAllUsesWith(newArg);
        continue;
      }

      // Otherwise this is a 1->1+ value mapping.
      Value castValue = argInfo->castValue;
      assert(argInfo->newArgSize >= 1 && castValue && "expected 1->1+ mapping");

      // If the argument is still used, replace it with the generated cast.
      if (!origArg.use_empty()) {
        origArg.replaceAllUsesWith(
            mapping.lookupOrDefault(castValue, origArg.getType()));
      }
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
      // If the type of this argument changed and the argument is still live, we
      // need to materialize a conversion.
      BlockArgument origArg = origBlock->getArgument(i);
      if (mapping.lookupOrNull(origArg, origArg.getType()))
        continue;
      Operation *liveUser = findLiveUser(origArg);
      if (!liveUser)
        continue;

      Value replacementValue = mapping.lookupOrDefault(origArg);
      bool isDroppedArg = replacementValue == origArg;
      if (isDroppedArg)
        rewriter.setInsertionPointToStart(newBlock);
      else
        rewriter.setInsertionPointAfterValue(replacementValue);
      Value newArg;
      if (blockInfo.converter) {
        newArg = blockInfo.converter->materializeSourceConversion(
            rewriter, origArg.getLoc(), origArg.getType(),
            isDroppedArg ? ValueRange() : ValueRange(replacementValue));
        assert((!newArg || newArg.getType() == origArg.getType()) &&
               "materialization hook did not provide a value of the expected "
               "type");
      }
      if (!newArg) {
        InFlightDiagnostic diag =
            emitError(origArg.getLoc())
            << "failed to materialize conversion for block argument #" << i
            << " that remained live after conversion, type was "
            << origArg.getType();
        if (!isDroppedArg)
          diag << ", with target type " << replacementValue.getType();
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
    Block *block, TypeConverter *converter, ConversionValueMapping &mapping,
    SmallVectorImpl<BlockArgument> &argReplacements) {
  // Check if the block was already converted. If the block is detached,
  // conservatively assume it is going to be deleted.
  if (hasBeenConverted(block) || !block->getParent())
    return block;
  // If a converter wasn't provided, and the block wasn't already converted,
  // there is nothing we can do.
  if (!converter)
    return failure();

  // Try to convert the signature for the block with the provided converter.
  if (auto conversion = converter->convertBlockSignature(block))
    return applySignatureConversion(block, converter, *conversion, mapping,
                                    argReplacements);
  return failure();
}

Block *ArgConverter::applySignatureConversion(
    Block *block, TypeConverter *converter,
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

  // FIXME: We should map the new arguments to proper locations.
  SmallVector<Location> newLocs(convertedTypes.size(),
                                rewriter.getUnknownLoc());
  SmallVector<Value, 4> newArgRange(
      newBlock->addArguments(convertedTypes, newLocs));
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

    // Otherwise, this is a 1->1+ mapping.
    auto replArgs = newArgs.slice(inputMap->inputNo, inputMap->size);
    Value newArg;

    // If this is a 1->1 mapping and the types of new and replacement arguments
    // match (i.e. it's an identity map), then the argument is mapped to its
    // original type.
    // FIXME: We simply pass through the replacement argument if there wasn't a
    // converter, which isn't great as it allows implicit type conversions to
    // appear. We should properly restructure this code to handle cases where a
    // converter isn't provided and also to properly handle the case where an
    // argument materialization is actually a temporary source materialization
    // (e.g. in the case of 1->N).
    if (replArgs.size() == 1 &&
        (!converter || replArgs[0].getType() == origArg.getType())) {
      newArg = replArgs.front();
    } else {
      Type origOutputType = origArg.getType();

      // Legalize the argument output type.
      Type outputType = origOutputType;
      if (Type legalOutputType = converter->convertType(outputType))
        outputType = legalOutputType;

      newArg = buildUnresolvedArgumentMaterialization(
          rewriter, origArg.getLoc(), replArgs, origOutputType, outputType,
          converter, unresolvedMaterializations);
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
// ConversionPatternRewriterImpl
//===----------------------------------------------------------------------===//
namespace mlir {
namespace detail {
struct ConversionPatternRewriterImpl {
  explicit ConversionPatternRewriterImpl(PatternRewriter &rewriter)
      : argConverter(rewriter, unresolvedMaterializations),
        notifyCallback(nullptr) {}

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

  /// Remap the given values to those with potentially different types. Returns
  /// success if the values could be remapped, failure otherwise. `valueDiagTag`
  /// is the tag used when describing a value within a diagnostic, e.g.
  /// "operand".
  LogicalResult remapValues(StringRef valueDiagTag,
                            std::optional<Location> inputLoc,
                            PatternRewriter &rewriter, ValueRange values,
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
      Block *block, TypeConverter *converter,
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
  SmallVector<Operation *> createdOps;

  /// Ordered vector of all unresolved type conversion materializations during
  /// conversion.
  SmallVector<UnresolvedMaterialization> unresolvedMaterializations;

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

  /// The current type converter, or nullptr if no type converter is currently
  /// active.
  TypeConverter *currentTypeConverter = nullptr;

  /// This allows the user to collect the match failure message.
  function_ref<void(Diagnostic &)> notifyCallback;

#ifndef NDEBUG
  /// A set of operations that have pending updates. This tracking isn't
  /// strictly necessary, and is thus only active during debug builds for extra
  /// verification.
  SmallPtrSet<Operation *, 1> pendingRootUpdates;

  /// A logger used to emit diagnostics during the conversion process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};
} // namespace detail
} // namespace mlir

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
  for (UnresolvedMaterialization &materialization : unresolvedMaterializations)
    detachNestedAndErase(materialization.getOp());
  for (auto *op : llvm::reverse(createdOps))
    detachNestedAndErase(op);
}

void ConversionPatternRewriterImpl::applyRewrites() {
  // Apply all of the rewrites replacements requested during conversion.
  for (auto &repl : replacements) {
    for (OpResult result : repl.first->getResults())
      if (Value newValue = mapping.lookupOrNull(result, result.getType()))
        result.replaceAllUsesWith(newValue);

    // If this operation defines any regions, drop any pending argument
    // rewrites.
    if (repl.first->getNumRegions())
      argConverter.notifyOpRemoved(repl.first);
  }

  // Apply all of the requested argument replacements.
  for (BlockArgument arg : argReplacements) {
    Value repl = mapping.lookupOrNull(arg, arg.getType());
    if (!repl)
      continue;

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

  // Drop all of the unresolved materialization operations created during
  // conversion.
  for (auto &mat : unresolvedMaterializations) {
    mat.getOp()->dropAllUses();
    mat.getOp()->erase();
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
  return RewriterState(createdOps.size(), unresolvedMaterializations.size(),
                       replacements.size(), argReplacements.size(),
                       blockActions.size(), ignoredOps.size(),
                       rootUpdates.size());
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

  // Pop all of the newly inserted materializations.
  while (unresolvedMaterializations.size() !=
         state.numUnresolvedMaterializations) {
    UnresolvedMaterialization mat = unresolvedMaterializations.pop_back_val();
    UnrealizedConversionCastOp op = mat.getOp();

    // If this was a target materialization, drop the mapping that was inserted.
    if (mat.getKind() == UnresolvedMaterialization::Target) {
      for (Value input : op->getOperands())
        mapping.erase(input);
    }
    detachNestedAndErase(op);
  }

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
    StringRef valueDiagTag, std::optional<Location> inputLoc,
    PatternRewriter &rewriter, ValueRange values,
    SmallVectorImpl<Value> &remapped) {
  remapped.reserve(llvm::size(values));

  SmallVector<Type, 1> legalTypes;
  for (const auto &it : llvm::enumerate(values)) {
    Value operand = it.value();
    Type origType = operand.getType();

    // If a converter was provided, get the desired legal types for this
    // operand.
    Type desiredType;
    if (currentTypeConverter) {
      // If there is no legal conversion, fail to match this pattern.
      legalTypes.clear();
      if (failed(currentTypeConverter->convertType(origType, legalTypes))) {
        Location operandLoc = inputLoc ? *inputLoc : operand.getLoc();
        return notifyMatchFailure(operandLoc, [=](Diagnostic &diag) {
          diag << "unable to convert type for " << valueDiagTag << " #"
               << it.index() << ", type was " << origType;
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
    if (currentTypeConverter && desiredType && newOperandType != desiredType) {
      Location operandLoc = inputLoc ? *inputLoc : operand.getLoc();
      Value castValue = buildUnresolvedTargetMaterialization(
          operandLoc, newOperand, desiredType, currentTypeConverter,
          unresolvedMaterializations);
      mapping.map(mapping.lookupOrDefault(newOperand), castValue);
      newOperand = castValue;
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
    Block *block, TypeConverter *converter,
    TypeConverter::SignatureConversion *conversion) {
  FailureOr<Block *> result =
      conversion ? argConverter.applySignatureConversion(
                       block, converter, *conversion, mapping, argReplacements)
                 : argConverter.convertSignature(block, converter, mapping,
                                                 argReplacements);
  if (failed(result))
    return failure();
  if (Block *newBlock = *result) {
    if (newBlock != block)
      blockActions.push_back(BlockAction::getTypeConversion(newBlock));
  }
  return result;
}

Block *ConversionPatternRewriterImpl::applySignatureConversion(
    Region *region, TypeConverter::SignatureConversion &conversion,
    TypeConverter *converter) {
  if (!region->empty())
    return *convertBlockSignature(&region->front(), converter, &conversion);
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
      convertBlockSignature(&region->front(), &converter, entryConversion);
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

    if (failed(convertBlockSignature(&block, &converter, blockConversion)))
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
  for (auto [newValue, result] : llvm::zip(newValues, op->getResults())) {
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
  replacements.insert(std::make_pair(op, OpReplacement(currentTypeConverter)));

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
    if (notifyCallback)
      notifyCallback(diag);
  });
  return failure();
}

//===----------------------------------------------------------------------===//
// ConversionPatternRewriter
//===----------------------------------------------------------------------===//

ConversionPatternRewriter::ConversionPatternRewriter(MLIRContext *ctx)
    : PatternRewriter(ctx),
      impl(new detail::ConversionPatternRewriterImpl(*this)) {}
ConversionPatternRewriter::~ConversionPatternRewriter() = default;

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

void ConversionPatternRewriter::replaceOp(Operation *op, ValueRange newValues) {
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Replace : '" << op->getName() << "'(" << op << ")\n";
  });
  impl->notifyOpReplaced(op, newValues);
}

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

Value ConversionPatternRewriter::getRemappedValue(Value key) {
  SmallVector<Value> remappedValues;
  if (failed(impl->remapValues("value", /*inputLoc=*/std::nullopt, *this, key,
                               remappedValues)))
    return nullptr;
  return remappedValues.front();
}

LogicalResult
ConversionPatternRewriter::getRemappedValues(ValueRange keys,
                                             SmallVectorImpl<Value> &results) {
  if (keys.empty())
    return success();
  return impl->remapValues("value", /*inputLoc=*/std::nullopt, *this, keys,
                           results);
}

void ConversionPatternRewriter::notifyBlockCreated(Block *block) {
  impl->notifyCreatedBlock(block);
}

Block *ConversionPatternRewriter::splitBlock(Block *block,
                                             Block::iterator before) {
  auto *continuation = PatternRewriter::splitBlock(block, before);
  impl->notifySplitBlock(block, continuation);
  return continuation;
}

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

void ConversionPatternRewriter::inlineRegionBefore(Region &region,
                                                   Region &parent,
                                                   Region::iterator before) {
  impl->notifyRegionIsBeingInlinedBefore(region, parent, before);
  PatternRewriter::inlineRegionBefore(region, parent, before);
}

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

void ConversionPatternRewriter::notifyOperationInserted(Operation *op) {
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Insert  : '" << op->getName() << "'(" << op << ")\n";
  });
  impl->createdOps.push_back(op);
}

void ConversionPatternRewriter::startRootUpdate(Operation *op) {
#ifndef NDEBUG
  impl->pendingRootUpdates.insert(op);
#endif
  impl->rootUpdates.emplace_back(op);
}

void ConversionPatternRewriter::finalizeRootUpdate(Operation *op) {
  // There is nothing to do here, we only need to track the operation at the
  // start of the update.
#ifndef NDEBUG
  assert(impl->pendingRootUpdates.erase(op) &&
         "operation did not have a pending in-place update");
#endif
}

void ConversionPatternRewriter::cancelRootUpdate(Operation *op) {
#ifndef NDEBUG
  assert(impl->pendingRootUpdates.erase(op) &&
         "operation did not have a pending in-place update");
#endif
  // Erase the last update for this operation.
  auto stateHasOp = [op](const auto &it) { return it.getOperation() == op; };
  auto &rootUpdates = impl->rootUpdates;
  auto it = llvm::find_if(llvm::reverse(rootUpdates), stateHasOp);
  assert(it != rootUpdates.rend() && "no root update started on op");
  (*it).resetOperation();
  int updateIdx = std::prev(rootUpdates.rend()) - it;
  rootUpdates.erase(rootUpdates.begin() + updateIdx);
}

LogicalResult ConversionPatternRewriter::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  return impl->notifyMatchFailure(loc, reasonCallback);
}

detail::ConversionPatternRewriterImpl &ConversionPatternRewriter::getImpl() {
  return *impl;
}

//===----------------------------------------------------------------------===//
// ConversionPattern
//===----------------------------------------------------------------------===//

LogicalResult
ConversionPattern::matchAndRewrite(Operation *op,
                                   PatternRewriter &rewriter) const {
  auto &dialectRewriter = static_cast<ConversionPatternRewriter &>(rewriter);
  auto &rewriterImpl = dialectRewriter.getImpl();

  // Track the current conversion pattern type converter in the rewriter.
  llvm::SaveAndRestore currentConverterGuard(rewriterImpl.currentTypeConverter,
                                             getTypeConverter());

  // Remap the operands of the operation.
  SmallVector<Value, 4> operands;
  if (failed(rewriterImpl.remapValues("operand", op->getLoc(), rewriter,
                                      op->getOperands(), operands))) {
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
  return target.isIllegal(op);
}

LogicalResult
OperationLegalizer::legalize(Operation *op,
                             ConversionPatternRewriter &rewriter) {
#ifndef NDEBUG
  const char *logLineComment =
      "//===-------------------------------------------===//\n";

  auto &logger = rewriter.getImpl().logger;
#endif
  LLVM_DEBUG({
    logger.getOStream() << "\n";
    logger.startLine() << logLineComment;
    logger.startLine() << "Legalizing operation : '" << op->getName() << "'("
                       << op << ") {\n";
    logger.indent();

    // If the operation has no regions, just print it here.
    if (op->getNumRegions() == 0) {
      op->print(logger.startLine(), OpPrintingFlags().printGenericOpForm());
      logger.getOStream() << "\n\n";
    }
  });

  // Check if this operation is legal on the target.
  if (auto legalityInfo = target.isLegal(op)) {
    LLVM_DEBUG({
      logSuccess(
          logger, "operation marked legal by the target{0}",
          legalityInfo->isRecursivelyLegal
              ? "; NOTE: operation is recursively legal; skipping internals"
              : "");
      logger.startLine() << logLineComment;
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
      logSuccess(logger, "operation marked 'ignored' during conversion");
      logger.startLine() << logLineComment;
    });
    return success();
  }

  // If the operation isn't legal, try to fold it in-place.
  // TODO: Should we always try to do this, even if the op is
  // already legal?
  if (succeeded(legalizeWithFold(op, rewriter))) {
    LLVM_DEBUG({
      logSuccess(logger, "operation was folded");
      logger.startLine() << logLineComment;
    });
    return success();
  }

  // Otherwise, we need to apply a legalization pattern to this operation.
  if (succeeded(legalizeWithPattern(op, rewriter))) {
    LLVM_DEBUG({
      logSuccess(logger, "");
      logger.startLine() << logLineComment;
    });
    return success();
  }

  LLVM_DEBUG({
    logFailure(logger, "no matched legalization pattern");
    logger.startLine() << logLineComment;
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
                            "failed to legalize generated constant '{0}'",
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
    LLVM_DEBUG({
      logFailure(rewriterImpl.logger, "pattern failed to match");
      if (rewriterImpl.notifyCallback) {
        Diagnostic diag(op->getLoc(), DiagnosticSeverity::Remark);
        diag << "Failed to apply pattern \"" << pattern.getDebugName()
             << "\" on op:\n"
             << *op;
        rewriterImpl.notifyCallback(diag);
      }
    });
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
    llvm::interleaveComma(pattern.getGeneratedOps(), os.getOStream());
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
      if (failed(impl.convertBlockSignature(action.block, converter))) {
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
                            "failed to legalize generated operation '{0}'({1})",
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
      LLVM_DEBUG(logFailure(
          impl.logger, "failed to legalize operation updated in-place '{0}'",
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
    std::optional<OperationName> root = pattern.getRootKind();

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
          std::optional<LegalizationAction> action = target.getOpAction(op);
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
    if (std::optional<OperationName> rootName = pattern.getRootKind())
      orderedPatternList = legalizerPatterns[*rootName];
    else
      orderedPatternList = anyOpLegalizerPatterns;

    // If the pattern is not found, then it was removed and cannot be matched.
    auto *it = llvm::find(orderedPatternList, &pattern);
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
    unsigned depth = 1;
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
  /// In this mode, the conversion will ignore failed conversions to allow
  /// illegal operations to co-exist in the IR.
  Partial,

  /// In this mode, all operations must be legal for the given target for the
  /// conversion to succeed.
  Full,

  /// In this mode, operations are analyzed for legality. No actual rewrites are
  /// applied to the operations on success.
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
  LogicalResult
  convertOperations(ArrayRef<Operation *> ops,
                    function_ref<void(Diagnostic &)> notifyCallback = nullptr);

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

  /// Legalize any unresolved type materializations.
  LogicalResult legalizeUnresolvedMaterializations(
      ConversionPatternRewriter &rewriter,
      ConversionPatternRewriterImpl &rewriterImpl,
      std::optional<DenseMap<Value, SmallVector<Value>>> &inverseMapping);

  /// Legalize an operation result that was marked as "erased".
  LogicalResult
  legalizeErasedResult(Operation *op, OpResult result,
                       ConversionPatternRewriterImpl &rewriterImpl);

  /// Legalize an operation result that was replaced with a value of a different
  /// type.
  LogicalResult legalizeChangedResultType(
      Operation *op, OpResult result, Value newValue,
      TypeConverter *replConverter, ConversionPatternRewriter &rewriter,
      ConversionPatternRewriterImpl &rewriterImpl,
      const DenseMap<Value, SmallVector<Value>> &inverseMapping);

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
} // namespace

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

LogicalResult OperationConverter::convertOperations(
    ArrayRef<Operation *> ops,
    function_ref<void(Diagnostic &)> notifyCallback) {
  if (ops.empty())
    return success();
  ConversionTarget &target = opLegalizer.getTarget();

  // Compute the set of operations and blocks to convert.
  SmallVector<Operation *> toConvert;
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
  rewriterImpl.notifyCallback = notifyCallback;

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
  if (mode == OpConversionMode::Analysis) {
    rewriterImpl.discardRewrites();
  } else {
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
  std::optional<DenseMap<Value, SmallVector<Value>>> inverseMapping;
  ConversionPatternRewriterImpl &rewriterImpl = rewriter.getImpl();
  if (failed(legalizeUnresolvedMaterializations(rewriter, rewriterImpl,
                                                inverseMapping)) ||
      failed(legalizeConvertedArgumentTypes(rewriter, rewriterImpl)))
    return failure();

  if (rewriterImpl.operationsWithChangedResults.empty())
    return success();

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
  return rewriterImpl.argConverter.materializeLiveConversions(
      rewriterImpl.mapping, rewriter, findLiveUser);
}

/// Replace the results of a materialization operation with the given values.
static void
replaceMaterialization(ConversionPatternRewriterImpl &rewriterImpl,
                       ResultRange matResults, ValueRange values,
                       DenseMap<Value, SmallVector<Value>> &inverseMapping) {
  matResults.replaceAllUsesWith(values);

  // For each of the materialization results, update the inverse mappings to
  // point to the replacement values.
  for (auto [matResult, newValue] : llvm::zip(matResults, values)) {
    auto inverseMapIt = inverseMapping.find(matResult);
    if (inverseMapIt == inverseMapping.end())
      continue;

    // Update the reverse mapping, or remove the mapping if we couldn't update
    // it. Not being able to update signals that the mapping would have become
    // circular (i.e. %foo -> newValue -> %foo), which may occur as values are
    // propagated through temporary materializations. We simply drop the
    // mapping, and let the post-conversion replacement logic handle updating
    // uses.
    for (Value inverseMapVal : inverseMapIt->second)
      if (!rewriterImpl.mapping.tryMap(inverseMapVal, newValue))
        rewriterImpl.mapping.erase(inverseMapVal);
  }
}

/// Compute all of the unresolved materializations that will persist beyond the
/// conversion process, and require inserting a proper user materialization for.
static void computeNecessaryMaterializations(
    DenseMap<Operation *, UnresolvedMaterialization *> &materializationOps,
    ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &rewriterImpl,
    DenseMap<Value, SmallVector<Value>> &inverseMapping,
    SetVector<UnresolvedMaterialization *> &necessaryMaterializations) {
  auto isLive = [&](Value value) {
    auto findFn = [&](Operation *user) {
      auto matIt = materializationOps.find(user);
      if (matIt != materializationOps.end())
        return !necessaryMaterializations.count(matIt->second);
      return rewriterImpl.isOpIgnored(user);
    };
    // This value may be replacing another value that has a live user.
    for (Value inv : inverseMapping.lookup(value))
      if (llvm::find_if_not(inv.getUsers(), findFn) != inv.user_end())
        return true;
    // Or have live users itself.
    return llvm::find_if_not(value.getUsers(), findFn) != value.user_end();
  };

  llvm::unique_function<Value(Value, Value, Type)> lookupRemappedValue =
      [&](Value invalidRoot, Value value, Type type) {
        // Check to see if the input operation was remapped to a variant of the
        // output.
        Value remappedValue = rewriterImpl.mapping.lookupOrDefault(value, type);
        if (remappedValue.getType() == type && remappedValue != invalidRoot)
          return remappedValue;

        // Check to see if the input is a materialization operation that
        // provides an inverse conversion. We just check blindly for
        // UnrealizedConversionCastOp here, but it has no effect on correctness.
        auto inputCastOp = value.getDefiningOp<UnrealizedConversionCastOp>();
        if (inputCastOp && inputCastOp->getNumOperands() == 1)
          return lookupRemappedValue(invalidRoot, inputCastOp->getOperand(0),
                                     type);

        return Value();
      };

  SetVector<UnresolvedMaterialization *> worklist;
  for (auto &mat : rewriterImpl.unresolvedMaterializations) {
    materializationOps.try_emplace(mat.getOp(), &mat);
    worklist.insert(&mat);
  }
  while (!worklist.empty()) {
    UnresolvedMaterialization *mat = worklist.pop_back_val();
    UnrealizedConversionCastOp op = mat->getOp();

    // We currently only handle target materializations here.
    assert(op->getNumResults() == 1 && "unexpected materialization type");
    OpResult opResult = op->getOpResult(0);
    Type outputType = opResult.getType();
    Operation::operand_range inputOperands = op.getOperands();

    // Try to forward propagate operands for user conversion casts that result
    // in the input types of the current cast.
    for (Operation *user : llvm::make_early_inc_range(opResult.getUsers())) {
      auto castOp = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!castOp)
        continue;
      if (castOp->getResultTypes() == inputOperands.getTypes()) {
        replaceMaterialization(rewriterImpl, opResult, inputOperands,
                               inverseMapping);
        necessaryMaterializations.remove(materializationOps.lookup(user));
      }
    }

    // Try to avoid materializing a resolved materialization if possible.
    // Handle the case of a 1-1 materialization.
    if (inputOperands.size() == 1) {
      // Check to see if the input operation was remapped to a variant of the
      // output.
      Value remappedValue =
          lookupRemappedValue(opResult, inputOperands[0], outputType);
      if (remappedValue && remappedValue != opResult) {
        replaceMaterialization(rewriterImpl, opResult, remappedValue,
                               inverseMapping);
        necessaryMaterializations.remove(mat);
        continue;
      }
    } else {
      // TODO: Avoid materializing other types of conversions here.
    }

    // Check to see if this is an argument materialization.
    auto isBlockArg = [](Value v) { return v.isa<BlockArgument>(); };
    if (llvm::any_of(op->getOperands(), isBlockArg) ||
        llvm::any_of(inverseMapping[op->getResult(0)], isBlockArg)) {
      mat->setKind(UnresolvedMaterialization::Argument);
    }

    // If the materialization does not have any live users, we don't need to
    // generate a user materialization for it.
    // FIXME: For argument materializations, we currently need to check if any
    // of the inverse mapped values are used because some patterns expect blind
    // value replacement even if the types differ in some cases. When those
    // patterns are fixed, we can drop the argument special case here.
    bool isMaterializationLive = isLive(opResult);
    if (mat->getKind() == UnresolvedMaterialization::Argument)
      isMaterializationLive |= llvm::any_of(inverseMapping[opResult], isLive);
    if (!isMaterializationLive)
      continue;
    if (!necessaryMaterializations.insert(mat))
      continue;

    // Reprocess input materializations to see if they have an updated status.
    for (Value input : inputOperands) {
      if (auto parentOp = input.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (auto *mat = materializationOps.lookup(parentOp))
          worklist.insert(mat);
      }
    }
  }
}

/// Legalize the given unresolved materialization. Returns success if the
/// materialization was legalized, failure otherise.
static LogicalResult legalizeUnresolvedMaterialization(
    UnresolvedMaterialization &mat,
    DenseMap<Operation *, UnresolvedMaterialization *> &materializationOps,
    ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &rewriterImpl,
    DenseMap<Value, SmallVector<Value>> &inverseMapping) {
  auto findLiveUser = [&](auto &&users) {
    auto liveUserIt = llvm::find_if_not(
        users, [&](Operation *user) { return rewriterImpl.isOpIgnored(user); });
    return liveUserIt == users.end() ? nullptr : *liveUserIt;
  };

  llvm::unique_function<Value(Value, Type)> lookupRemappedValue =
      [&](Value value, Type type) {
        // Check to see if the input operation was remapped to a variant of the
        // output.
        Value remappedValue = rewriterImpl.mapping.lookupOrDefault(value, type);
        if (remappedValue.getType() == type)
          return remappedValue;
        return Value();
      };

  UnrealizedConversionCastOp op = mat.getOp();
  if (!rewriterImpl.ignoredOps.insert(op))
    return success();

  // We currently only handle target materializations here.
  OpResult opResult = op->getOpResult(0);
  Operation::operand_range inputOperands = op.getOperands();
  Type outputType = opResult.getType();

  // If any input to this materialization is another materialization, resolve
  // the input first.
  for (Value value : op->getOperands()) {
    auto valueCast = value.getDefiningOp<UnrealizedConversionCastOp>();
    if (!valueCast)
      continue;

    auto matIt = materializationOps.find(valueCast);
    if (matIt != materializationOps.end())
      if (failed(legalizeUnresolvedMaterialization(
              *matIt->second, materializationOps, rewriter, rewriterImpl,
              inverseMapping)))
        return failure();
  }

  // Perform a last ditch attempt to avoid materializing a resolved
  // materialization if possible.
  // Handle the case of a 1-1 materialization.
  if (inputOperands.size() == 1) {
    // Check to see if the input operation was remapped to a variant of the
    // output.
    Value remappedValue = lookupRemappedValue(inputOperands[0], outputType);
    if (remappedValue && remappedValue != opResult) {
      replaceMaterialization(rewriterImpl, opResult, remappedValue,
                             inverseMapping);
      return success();
    }
  } else {
    // TODO: Avoid materializing other types of conversions here.
  }

  // Try to materialize the conversion.
  if (TypeConverter *converter = mat.getConverter()) {
    // FIXME: Determine a suitable insertion location when there are multiple
    // inputs.
    if (inputOperands.size() == 1)
      rewriter.setInsertionPointAfterValue(inputOperands.front());
    else
      rewriter.setInsertionPoint(op);

    Value newMaterialization;
    switch (mat.getKind()) {
    case UnresolvedMaterialization::Argument:
      // Try to materialize an argument conversion.
      // FIXME: The current argument materialization hook expects the original
      // output type, even though it doesn't use that as the actual output type
      // of the generated IR. The output type is just used as an indicator of
      // the type of materialization to do. This behavior is really awkward in
      // that it diverges from the behavior of the other hooks, and can be
      // easily misunderstood. We should clean up the argument hooks to better
      // represent the desired invariants we actually care about.
      newMaterialization = converter->materializeArgumentConversion(
          rewriter, op->getLoc(), mat.getOrigOutputType(), inputOperands);
      if (newMaterialization)
        break;

      // If an argument materialization failed, fallback to trying a target
      // materialization.
      [[fallthrough]];
    case UnresolvedMaterialization::Target:
      newMaterialization = converter->materializeTargetConversion(
          rewriter, op->getLoc(), outputType, inputOperands);
      break;
    }
    if (newMaterialization) {
      replaceMaterialization(rewriterImpl, opResult, newMaterialization,
                             inverseMapping);
      return success();
    }
  }

  InFlightDiagnostic diag = op->emitError()
                            << "failed to legalize unresolved materialization "
                               "from "
                            << inputOperands.getTypes() << " to " << outputType
                            << " that remained live after conversion";
  if (Operation *liveUser = findLiveUser(op->getUsers())) {
    diag.attachNote(liveUser->getLoc())
        << "see existing live user here: " << *liveUser;
  }
  return failure();
}

LogicalResult OperationConverter::legalizeUnresolvedMaterializations(
    ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &rewriterImpl,
    std::optional<DenseMap<Value, SmallVector<Value>>> &inverseMapping) {
  if (rewriterImpl.unresolvedMaterializations.empty())
    return success();
  inverseMapping = rewriterImpl.mapping.getInverse();

  // As an initial step, compute all of the inserted materializations that we
  // expect to persist beyond the conversion process.
  DenseMap<Operation *, UnresolvedMaterialization *> materializationOps;
  SetVector<UnresolvedMaterialization *> necessaryMaterializations;
  computeNecessaryMaterializations(materializationOps, rewriter, rewriterImpl,
                                   *inverseMapping, necessaryMaterializations);

  // Once computed, legalize any necessary materializations.
  for (auto *mat : necessaryMaterializations) {
    if (failed(legalizeUnresolvedMaterialization(
            *mat, materializationOps, rewriter, rewriterImpl, *inverseMapping)))
      return failure();
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
static Operation *findLiveUserOfReplaced(
    Value initialValue, ConversionPatternRewriterImpl &rewriterImpl,
    const DenseMap<Value, SmallVector<Value>> &inverseMapping) {
  SmallVector<Value> worklist(1, initialValue);
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();

    // Walk the users of this value to see if there are any live users that
    // weren't replaced during conversion.
    auto liveUserIt = llvm::find_if_not(value.getUsers(), [&](Operation *user) {
      return rewriterImpl.isOpIgnored(user);
    });
    if (liveUserIt != value.user_end())
      return *liveUserIt;
    auto mapIt = inverseMapping.find(value);
    if (mapIt != inverseMapping.end())
      worklist.append(mapIt->second);
  }
  return nullptr;
}

LogicalResult OperationConverter::legalizeChangedResultType(
    Operation *op, OpResult result, Value newValue,
    TypeConverter *replConverter, ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &rewriterImpl,
    const DenseMap<Value, SmallVector<Value>> &inverseMapping) {
  Operation *liveUser =
      findLiveUserOfReplaced(result, rewriterImpl, inverseMapping);
  if (!liveUser)
    return success();

  // Functor used to emit a conversion error for a failed materialization.
  auto emitConversionError = [&] {
    InFlightDiagnostic diag = op->emitError()
                              << "failed to materialize conversion for result #"
                              << result.getResultNumber() << " of operation '"
                              << op->getName()
                              << "' that remained live after conversion";
    diag.attachNote(liveUser->getLoc())
        << "see existing live user here: " << *liveUser;
    return failure();
  };

  // If the replacement has a type converter, attempt to materialize a
  // conversion back to the original type.
  if (!replConverter)
    return emitConversionError();

  // Materialize a conversion for this live result value.
  Type resultType = result.getType();
  Value convertedValue = replConverter->materializeSourceConversion(
      rewriter, op->getLoc(), resultType, newValue);
  if (!convertedValue)
    return emitConversionError();

  rewriterImpl.mapping.map(result, convertedValue);
  return success();
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

void TypeConverter::SignatureConversion::addInputs(unsigned origInputNo,
                                                   ArrayRef<Type> types) {
  assert(!types.empty() && "expected valid types");
  remapInput(origInputNo, /*newInputNo=*/argTypes.size(), types.size());
  addInputs(types);
}

void TypeConverter::SignatureConversion::addInputs(ArrayRef<Type> types) {
  assert(!types.empty() &&
         "1->0 type remappings don't need to be added explicitly");
  argTypes.append(types.begin(), types.end());
}

void TypeConverter::SignatureConversion::remapInput(unsigned origInputNo,
                                                    unsigned newInputNo,
                                                    unsigned newInputCount) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  assert(newInputCount != 0 && "expected valid input count");
  remappedInputs[origInputNo] =
      InputMapping{newInputNo, newInputCount, /*replacementValue=*/nullptr};
}

void TypeConverter::SignatureConversion::remapInput(unsigned origInputNo,
                                                    Value replacementValue) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  remappedInputs[origInputNo] =
      InputMapping{origInputNo, /*size=*/0, replacementValue};
}

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
  conversionCallStack.push_back(t);
  auto popConversionCallStack =
      llvm::make_scope_exit([this]() { conversionCallStack.pop_back(); });
  for (ConversionCallbackFn &converter : llvm::reverse(conversions)) {
    if (std::optional<LogicalResult> result =
            converter(t, results, conversionCallStack)) {
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

Type TypeConverter::convertType(Type t) {
  // Use the multi-type result version to convert the type.
  SmallVector<Type, 1> results;
  if (failed(convertType(t, results)))
    return nullptr;

  // Check to ensure that only one type was produced.
  return results.size() == 1 ? results.front() : nullptr;
}

LogicalResult TypeConverter::convertTypes(TypeRange types,
                                          SmallVectorImpl<Type> &results) {
  for (Type type : types)
    if (failed(convertType(type, results)))
      return failure();
  return success();
}

bool TypeConverter::isLegal(Type type) { return convertType(type) == type; }
bool TypeConverter::isLegal(Operation *op) {
  return isLegal(op->getOperandTypes()) && isLegal(op->getResultTypes());
}

bool TypeConverter::isLegal(Region *region) {
  return llvm::all_of(*region, [this](Block &block) {
    return isLegal(block.getArgumentTypes());
  });
}

bool TypeConverter::isSignatureLegal(FunctionType ty) {
  return isLegal(llvm::concat<const Type>(ty.getInputs(), ty.getResults()));
}

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
    if (std::optional<Value> result = fn(builder, resultType, inputs, loc))
      return *result;
  return nullptr;
}

auto TypeConverter::convertBlockSignature(Block *block)
    -> std::optional<SignatureConversion> {
  SignatureConversion conversion(block->getNumArguments());
  if (failed(convertSignatureArgs(block->getArgumentTypes(), conversion)))
    return std::nullopt;
  return conversion;
}

//===----------------------------------------------------------------------===//
// FunctionOpInterfaceSignatureConversion
//===----------------------------------------------------------------------===//

static LogicalResult convertFuncOpTypes(FunctionOpInterface funcOp,
                                        TypeConverter &typeConverter,
                                        ConversionPatternRewriter &rewriter) {
  FunctionType type = funcOp.getFunctionType().cast<FunctionType>();

  // Convert the original function types.
  TypeConverter::SignatureConversion result(type.getNumInputs());
  SmallVector<Type, 1> newResults;
  if (failed(typeConverter.convertSignatureArgs(type.getInputs(), result)) ||
      failed(typeConverter.convertTypes(type.getResults(), newResults)) ||
      failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                         typeConverter, &result)))
    return failure();

  // Update the function signature in-place.
  auto newType = FunctionType::get(rewriter.getContext(),
                                   result.getConvertedTypes(), newResults);

  rewriter.updateRootInPlace(funcOp, [&] { funcOp.setType(newType); });

  return success();
}

/// Create a default conversion pattern that rewrites the type signature of a
/// FunctionOpInterface op. This only supports ops which use FunctionType to
/// represent their type.
namespace {
struct FunctionOpInterfaceSignatureConversion : public ConversionPattern {
  FunctionOpInterfaceSignatureConversion(StringRef functionLikeOpName,
                                         MLIRContext *ctx,
                                         TypeConverter &converter)
      : ConversionPattern(converter, functionLikeOpName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionOpInterface funcOp = cast<FunctionOpInterface>(op);
    return convertFuncOpTypes(funcOp, *typeConverter, rewriter);
  }
};

struct AnyFunctionOpInterfaceSignatureConversion
    : public OpInterfaceConversionPattern<FunctionOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(FunctionOpInterface funcOp, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    return convertFuncOpTypes(funcOp, *typeConverter, rewriter);
  }
};
} // namespace

void mlir::populateFunctionOpInterfaceTypeConversionPattern(
    StringRef functionLikeOpName, RewritePatternSet &patterns,
    TypeConverter &converter) {
  patterns.add<FunctionOpInterfaceSignatureConversion>(
      functionLikeOpName, patterns.getContext(), converter);
}

void mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(
    RewritePatternSet &patterns, TypeConverter &converter) {
  patterns.add<AnyFunctionOpInterfaceSignatureConversion>(
      converter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// ConversionTarget
//===----------------------------------------------------------------------===//

void ConversionTarget::setOpAction(OperationName op,
                                   LegalizationAction action) {
  legalOperations[op].action = action;
}

void ConversionTarget::setDialectAction(ArrayRef<StringRef> dialectNames,
                                        LegalizationAction action) {
  for (StringRef dialect : dialectNames)
    legalDialects[dialect] = action;
}

auto ConversionTarget::getOpAction(OperationName op) const
    -> std::optional<LegalizationAction> {
  std::optional<LegalizationInfo> info = getOpInfo(op);
  return info ? info->action : std::optional<LegalizationAction>();
}

auto ConversionTarget::isLegal(Operation *op) const
    -> std::optional<LegalOpDetails> {
  std::optional<LegalizationInfo> info = getOpInfo(op->getName());
  if (!info)
    return std::nullopt;

  // Returns true if this operation instance is known to be legal.
  auto isOpLegal = [&] {
    // Handle dynamic legality either with the provided legality function.
    if (info->action == LegalizationAction::Dynamic) {
      std::optional<bool> result = info->legalityFn(op);
      if (result)
        return *result;
    }

    // Otherwise, the operation is only legal if it was marked 'Legal'.
    return info->action == LegalizationAction::Legal;
  };
  if (!isOpLegal())
    return std::nullopt;

  // This operation is legal, compute any additional legality information.
  LegalOpDetails legalityDetails;
  if (info->isRecursivelyLegal) {
    auto legalityFnIt = opRecursiveLegalityFns.find(op->getName());
    if (legalityFnIt != opRecursiveLegalityFns.end()) {
      legalityDetails.isRecursivelyLegal =
          legalityFnIt->second(op).value_or(true);
    } else {
      legalityDetails.isRecursivelyLegal = true;
    }
  }
  return legalityDetails;
}

bool ConversionTarget::isIllegal(Operation *op) const {
  std::optional<LegalizationInfo> info = getOpInfo(op->getName());
  if (!info)
    return false;

  if (info->action == LegalizationAction::Dynamic) {
    std::optional<bool> result = info->legalityFn(op);
    if (!result)
      return false;

    return !(*result);
  }

  return info->action == LegalizationAction::Illegal;
}

static ConversionTarget::DynamicLegalityCallbackFn composeLegalityCallbacks(
    ConversionTarget::DynamicLegalityCallbackFn oldCallback,
    ConversionTarget::DynamicLegalityCallbackFn newCallback) {
  if (!oldCallback)
    return newCallback;

  auto chain = [oldCl = std::move(oldCallback), newCl = std::move(newCallback)](
                   Operation *op) -> std::optional<bool> {
    if (std::optional<bool> result = newCl(op))
      return *result;

    return oldCl(op);
  };
  return chain;
}

void ConversionTarget::setLegalityCallback(
    OperationName name, const DynamicLegalityCallbackFn &callback) {
  assert(callback && "expected valid legality callback");
  auto infoIt = legalOperations.find(name);
  assert(infoIt != legalOperations.end() &&
         infoIt->second.action == LegalizationAction::Dynamic &&
         "expected operation to already be marked as dynamically legal");
  infoIt->second.legalityFn =
      composeLegalityCallbacks(std::move(infoIt->second.legalityFn), callback);
}

void ConversionTarget::markOpRecursivelyLegal(
    OperationName name, const DynamicLegalityCallbackFn &callback) {
  auto infoIt = legalOperations.find(name);
  assert(infoIt != legalOperations.end() &&
         infoIt->second.action != LegalizationAction::Illegal &&
         "expected operation to already be marked as legal");
  infoIt->second.isRecursivelyLegal = true;
  if (callback)
    opRecursiveLegalityFns[name] = composeLegalityCallbacks(
        std::move(opRecursiveLegalityFns[name]), callback);
  else
    opRecursiveLegalityFns.erase(name);
}

void ConversionTarget::setLegalityCallback(
    ArrayRef<StringRef> dialects, const DynamicLegalityCallbackFn &callback) {
  assert(callback && "expected valid legality callback");
  for (StringRef dialect : dialects)
    dialectLegalityFns[dialect] = composeLegalityCallbacks(
        std::move(dialectLegalityFns[dialect]), callback);
}

void ConversionTarget::setLegalityCallback(
    const DynamicLegalityCallbackFn &callback) {
  assert(callback && "expected valid legality callback");
  unknownLegalityFn = composeLegalityCallbacks(unknownLegalityFn, callback);
}

auto ConversionTarget::getOpInfo(OperationName op) const
    -> std::optional<LegalizationInfo> {
  // Check for info for this specific operation.
  auto it = legalOperations.find(op);
  if (it != legalOperations.end())
    return it->second;
  // Check for info for the parent dialect.
  auto dialectIt = legalDialects.find(op.getDialectNamespace());
  if (dialectIt != legalDialects.end()) {
    DynamicLegalityCallbackFn callback;
    auto dialectFn = dialectLegalityFns.find(op.getDialectNamespace());
    if (dialectFn != dialectLegalityFns.end())
      callback = dialectFn->second;
    return LegalizationInfo{dialectIt->second, /*isRecursivelyLegal=*/false,
                            callback};
  }
  // Otherwise, check if we mark unknown operations as dynamic.
  if (unknownLegalityFn)
    return LegalizationInfo{LegalizationAction::Dynamic,
                            /*isRecursivelyLegal=*/false, unknownLegalityFn};
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// PDL Configuration
//===----------------------------------------------------------------------===//

void PDLConversionConfig::notifyRewriteBegin(PatternRewriter &rewriter) {
  auto &rewriterImpl =
      static_cast<ConversionPatternRewriter &>(rewriter).getImpl();
  rewriterImpl.currentTypeConverter = getTypeConverter();
}

void PDLConversionConfig::notifyRewriteEnd(PatternRewriter &rewriter) {
  auto &rewriterImpl =
      static_cast<ConversionPatternRewriter &>(rewriter).getImpl();
  rewriterImpl.currentTypeConverter = nullptr;
}

/// Remap the given value using the rewriter and the type converter in the
/// provided config.
static FailureOr<SmallVector<Value>>
pdllConvertValues(ConversionPatternRewriter &rewriter, ValueRange values) {
  SmallVector<Value> mappedValues;
  if (failed(rewriter.getRemappedValues(values, mappedValues)))
    return failure();
  return std::move(mappedValues);
}

void mlir::registerConversionPDLFunctions(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerRewriteFunction(
      "convertValue",
      [](PatternRewriter &rewriter, Value value) -> FailureOr<Value> {
        auto results = pdllConvertValues(
            static_cast<ConversionPatternRewriter &>(rewriter), value);
        if (failed(results))
          return failure();
        return results->front();
      });
  patterns.getPDLPatterns().registerRewriteFunction(
      "convertValues", [](PatternRewriter &rewriter, ValueRange values) {
        return pdllConvertValues(
            static_cast<ConversionPatternRewriter &>(rewriter), values);
      });
  patterns.getPDLPatterns().registerRewriteFunction(
      "convertType",
      [](PatternRewriter &rewriter, Type type) -> FailureOr<Type> {
        auto &rewriterImpl =
            static_cast<ConversionPatternRewriter &>(rewriter).getImpl();
        if (TypeConverter *converter = rewriterImpl.currentTypeConverter) {
          if (Type newType = converter->convertType(type))
            return newType;
          return failure();
        }
        return type;
      });
  patterns.getPDLPatterns().registerRewriteFunction(
      "convertTypes",
      [](PatternRewriter &rewriter,
         TypeRange types) -> FailureOr<SmallVector<Type>> {
        auto &rewriterImpl =
            static_cast<ConversionPatternRewriter &>(rewriter).getImpl();
        TypeConverter *converter = rewriterImpl.currentTypeConverter;
        if (!converter)
          return SmallVector<Type>(types);

        SmallVector<Type> remappedTypes;
        if (failed(converter->convertTypes(types, remappedTypes)))
          return failure();
        return std::move(remappedTypes);
      });
}

//===----------------------------------------------------------------------===//
// Op Conversion Entry Points
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Partial Conversion

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
  return applyPartialConversion(llvm::ArrayRef(op), target, patterns,
                                unconvertedOps);
}

//===----------------------------------------------------------------------===//
// Full Conversion

LogicalResult
mlir::applyFullConversion(ArrayRef<Operation *> ops, ConversionTarget &target,
                          const FrozenRewritePatternSet &patterns) {
  OperationConverter opConverter(target, patterns, OpConversionMode::Full);
  return opConverter.convertOperations(ops);
}
LogicalResult
mlir::applyFullConversion(Operation *op, ConversionTarget &target,
                          const FrozenRewritePatternSet &patterns) {
  return applyFullConversion(llvm::ArrayRef(op), target, patterns);
}

//===----------------------------------------------------------------------===//
// Analysis Conversion

LogicalResult
mlir::applyAnalysisConversion(ArrayRef<Operation *> ops,
                              ConversionTarget &target,
                              const FrozenRewritePatternSet &patterns,
                              DenseSet<Operation *> &convertedOps,
                              function_ref<void(Diagnostic &)> notifyCallback) {
  OperationConverter opConverter(target, patterns, OpConversionMode::Analysis,
                                 &convertedOps);
  return opConverter.convertOperations(ops, notifyCallback);
}
LogicalResult
mlir::applyAnalysisConversion(Operation *op, ConversionTarget &target,
                              const FrozenRewritePatternSet &patterns,
                              DenseSet<Operation *> &convertedOps,
                              function_ref<void(Diagnostic &)> notifyCallback) {
  return applyAnalysisConversion(llvm::ArrayRef(op), target, patterns,
                                 convertedOps, notifyCallback);
}
