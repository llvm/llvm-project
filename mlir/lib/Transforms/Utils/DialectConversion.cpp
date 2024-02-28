//===- DialectConversion.cpp - MLIR dialect conversion generic pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ScopedPrinter.h"
#include <optional>

using namespace mlir;
using namespace mlir::detail;

#define DEBUG_TYPE "dialect-conversion"

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
/// This class wraps a IRMapping to provide recursive lookup
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
  IRMapping mapping;
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
  RewriterState(unsigned numRewrites, unsigned numIgnoredOperations,
                unsigned numErased, unsigned numReplacedOps)
      : numRewrites(numRewrites), numIgnoredOperations(numIgnoredOperations),
        numErased(numErased), numReplacedOps(numReplacedOps) {}

  /// The current number of rewrites performed.
  unsigned numRewrites;

  /// The current number of ignored operations.
  unsigned numIgnoredOperations;

  /// The current number of erased operations/blocks.
  unsigned numErased;

  /// The current number of replaced ops that are scheduled for erasure.
  unsigned numReplacedOps;
};

//===----------------------------------------------------------------------===//
// IR rewrites
//===----------------------------------------------------------------------===//

/// An IR rewrite that can be committed (upon success) or rolled back (upon
/// failure).
///
/// The dialect conversion keeps track of IR modifications (requested by the
/// user through the rewriter API) in `IRRewrite` objects. Some kind of rewrites
/// are directly applied to the IR as the rewriter API is used, some are applied
/// partially, and some are delayed until the `IRRewrite` objects are committed.
class IRRewrite {
public:
  /// The kind of the rewrite. Rewrites can be undone if the conversion fails.
  /// Enum values are ordered, so that they can be used in `classof`: first all
  /// block rewrites, then all operation rewrites.
  enum class Kind {
    // Block rewrites
    CreateBlock,
    EraseBlock,
    InlineBlock,
    MoveBlock,
    SplitBlock,
    BlockTypeConversion,
    ReplaceBlockArg,
    // Operation rewrites
    MoveOperation,
    ModifyOperation,
    ReplaceOperation,
    CreateOperation,
    UnresolvedMaterialization
  };

  virtual ~IRRewrite() = default;

  /// Roll back the rewrite. Operations may be erased during rollback.
  virtual void rollback() = 0;

  /// Commit the rewrite. Operations may be unlinked from their blocks during
  /// the commit phase, but they must not be erased yet. This is because
  /// internal dialect conversion state (such as `mapping`) may still be using
  /// them. Operations must be erased during cleanup.
  virtual void commit() {}

  /// Cleanup operations. Cleanup is called after commit.
  virtual void cleanup() {}

  Kind getKind() const { return kind; }

  static bool classof(const IRRewrite *rewrite) { return true; }

protected:
  IRRewrite(Kind kind, ConversionPatternRewriterImpl &rewriterImpl)
      : kind(kind), rewriterImpl(rewriterImpl) {}

  /// Erase the given op (unless it was already erased).
  void eraseOp(Operation *op);

  /// Erase the given block (unless it was already erased).
  void eraseBlock(Block *block);

  const ConversionConfig &getConfig() const;

  const Kind kind;
  ConversionPatternRewriterImpl &rewriterImpl;
};

/// A block rewrite.
class BlockRewrite : public IRRewrite {
public:
  /// Return the block that this rewrite operates on.
  Block *getBlock() const { return block; }

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() >= Kind::CreateBlock &&
           rewrite->getKind() <= Kind::ReplaceBlockArg;
  }

protected:
  BlockRewrite(Kind kind, ConversionPatternRewriterImpl &rewriterImpl,
               Block *block)
      : IRRewrite(kind, rewriterImpl), block(block) {}

  // The block that this rewrite operates on.
  Block *block;
};

/// Creation of a block. Block creations are immediately reflected in the IR.
/// There is no extra work to commit the rewrite. During rollback, the newly
/// created block is erased.
class CreateBlockRewrite : public BlockRewrite {
public:
  CreateBlockRewrite(ConversionPatternRewriterImpl &rewriterImpl, Block *block)
      : BlockRewrite(Kind::CreateBlock, rewriterImpl, block) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::CreateBlock;
  }

  void rollback() override {
    // Unlink all of the operations within this block, they will be deleted
    // separately.
    auto &blockOps = block->getOperations();
    while (!blockOps.empty())
      blockOps.remove(blockOps.begin());
    if (block->getParent())
      eraseBlock(block);
    else
      delete block;
  }
};

/// Erasure of a block. Block erasures are partially reflected in the IR. Erased
/// blocks are immediately unlinked, but only erased when the rewrite is
/// committed. This makes it easier to rollback a block erasure: the block is
/// simply inserted into its original location.
class EraseBlockRewrite : public BlockRewrite {
public:
  EraseBlockRewrite(ConversionPatternRewriterImpl &rewriterImpl, Block *block,
                    Region *region, Block *insertBeforeBlock)
      : BlockRewrite(Kind::EraseBlock, rewriterImpl, block), region(region),
        insertBeforeBlock(insertBeforeBlock) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::EraseBlock;
  }

  ~EraseBlockRewrite() override {
    assert(!block && "rewrite was neither rolled back nor committed");
  }

  void rollback() override {
    // The block (owned by this rewrite) was not actually erased yet. It was
    // just unlinked. Put it back into its original position.
    assert(block && "expected block");
    auto &blockList = region->getBlocks();
    Region::iterator before = insertBeforeBlock
                                  ? Region::iterator(insertBeforeBlock)
                                  : blockList.end();
    blockList.insert(before, block);
    block = nullptr;
  }

  void commit() override {
    // Erase the block.
    assert(block && "expected block");
    assert(block->empty() && "expected empty block");
    block->dropAllDefinedValueUses();
    delete block;
    block = nullptr;
  }

private:
  // The region in which this block was previously contained.
  Region *region;

  // The original successor of this block before it was unlinked. "nullptr" if
  // this block was the only block in the region.
  Block *insertBeforeBlock;
};

/// Inlining of a block. This rewrite is immediately reflected in the IR.
/// Note: This rewrite represents only the inlining of the operations. The
/// erasure of the inlined block is a separate rewrite.
class InlineBlockRewrite : public BlockRewrite {
public:
  InlineBlockRewrite(ConversionPatternRewriterImpl &rewriterImpl, Block *block,
                     Block *sourceBlock, Block::iterator before)
      : BlockRewrite(Kind::InlineBlock, rewriterImpl, block),
        sourceBlock(sourceBlock),
        firstInlinedInst(sourceBlock->empty() ? nullptr
                                              : &sourceBlock->front()),
        lastInlinedInst(sourceBlock->empty() ? nullptr : &sourceBlock->back()) {
  }

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::InlineBlock;
  }

  void rollback() override {
    // Put the operations from the destination block (owned by the rewrite)
    // back into the source block.
    if (firstInlinedInst) {
      assert(lastInlinedInst && "expected operation");
      sourceBlock->getOperations().splice(sourceBlock->begin(),
                                          block->getOperations(),
                                          Block::iterator(firstInlinedInst),
                                          ++Block::iterator(lastInlinedInst));
    }
  }

private:
  // The block that originally contained the operations.
  Block *sourceBlock;

  // The first inlined operation.
  Operation *firstInlinedInst;

  // The last inlined operation.
  Operation *lastInlinedInst;
};

/// Moving of a block. This rewrite is immediately reflected in the IR.
class MoveBlockRewrite : public BlockRewrite {
public:
  MoveBlockRewrite(ConversionPatternRewriterImpl &rewriterImpl, Block *block,
                   Region *region, Block *insertBeforeBlock)
      : BlockRewrite(Kind::MoveBlock, rewriterImpl, block), region(region),
        insertBeforeBlock(insertBeforeBlock) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::MoveBlock;
  }

  void rollback() override {
    // Move the block back to its original position.
    Region::iterator before =
        insertBeforeBlock ? Region::iterator(insertBeforeBlock) : region->end();
    region->getBlocks().splice(before, block->getParent()->getBlocks(), block);
  }

private:
  // The region in which this block was previously contained.
  Region *region;

  // The original successor of this block before it was moved. "nullptr" if
  // this block was the only block in the region.
  Block *insertBeforeBlock;
};

/// Splitting of a block. This rewrite is immediately reflected in the IR.
class SplitBlockRewrite : public BlockRewrite {
public:
  SplitBlockRewrite(ConversionPatternRewriterImpl &rewriterImpl, Block *block,
                    Block *originalBlock)
      : BlockRewrite(Kind::SplitBlock, rewriterImpl, block),
        originalBlock(originalBlock) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::SplitBlock;
  }

  void rollback() override {
    // Merge back the block that was split out.
    originalBlock->getOperations().splice(originalBlock->end(),
                                          block->getOperations());
    eraseBlock(block);
  }

private:
  // The original block from which this block was split.
  Block *originalBlock;
};

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

/// Block type conversion. This rewrite is partially reflected in the IR.
class BlockTypeConversionRewrite : public BlockRewrite {
public:
  BlockTypeConversionRewrite(
      ConversionPatternRewriterImpl &rewriterImpl, Block *block,
      Block *origBlock, SmallVector<std::optional<ConvertedArgInfo>, 1> argInfo,
      const TypeConverter *converter)
      : BlockRewrite(Kind::BlockTypeConversion, rewriterImpl, block),
        origBlock(origBlock), argInfo(argInfo), converter(converter) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::BlockTypeConversion;
  }

  /// Materialize any necessary conversions for converted arguments that have
  /// live users, using the provided `findLiveUser` to search for a user that
  /// survives the conversion process.
  LogicalResult
  materializeLiveConversions(function_ref<Operation *(Value)> findLiveUser);

  void commit() override;

  void rollback() override;

private:
  /// The original block that was requested to have its signature converted.
  Block *origBlock;

  /// The conversion information for each of the arguments. The information is
  /// std::nullopt if the argument was dropped during conversion.
  SmallVector<std::optional<ConvertedArgInfo>, 1> argInfo;

  /// The type converter used to convert the arguments.
  const TypeConverter *converter;
};

/// Replacing a block argument. This rewrite is not immediately reflected in the
/// IR. An internal IR mapping is updated, but the actual replacement is delayed
/// until the rewrite is committed.
class ReplaceBlockArgRewrite : public BlockRewrite {
public:
  ReplaceBlockArgRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                         Block *block, BlockArgument arg)
      : BlockRewrite(Kind::ReplaceBlockArg, rewriterImpl, block), arg(arg) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::ReplaceBlockArg;
  }

  void commit() override;

  void rollback() override;

private:
  BlockArgument arg;
};

/// An operation rewrite.
class OperationRewrite : public IRRewrite {
public:
  /// Return the operation that this rewrite operates on.
  Operation *getOperation() const { return op; }

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() >= Kind::MoveOperation &&
           rewrite->getKind() <= Kind::UnresolvedMaterialization;
  }

protected:
  OperationRewrite(Kind kind, ConversionPatternRewriterImpl &rewriterImpl,
                   Operation *op)
      : IRRewrite(kind, rewriterImpl), op(op) {}

  // The operation that this rewrite operates on.
  Operation *op;
};

/// Moving of an operation. This rewrite is immediately reflected in the IR.
class MoveOperationRewrite : public OperationRewrite {
public:
  MoveOperationRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                       Operation *op, Block *block, Operation *insertBeforeOp)
      : OperationRewrite(Kind::MoveOperation, rewriterImpl, op), block(block),
        insertBeforeOp(insertBeforeOp) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::MoveOperation;
  }

  void rollback() override {
    // Move the operation back to its original position.
    Block::iterator before =
        insertBeforeOp ? Block::iterator(insertBeforeOp) : block->end();
    block->getOperations().splice(before, op->getBlock()->getOperations(), op);
  }

private:
  // The block in which this operation was previously contained.
  Block *block;

  // The original successor of this operation before it was moved. "nullptr"
  // if this operation was the only operation in the region.
  Operation *insertBeforeOp;
};

/// In-place modification of an op. This rewrite is immediately reflected in
/// the IR. The previous state of the operation is stored in this object.
class ModifyOperationRewrite : public OperationRewrite {
public:
  ModifyOperationRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                         Operation *op)
      : OperationRewrite(Kind::ModifyOperation, rewriterImpl, op),
        name(op->getName()), loc(op->getLoc()), attrs(op->getAttrDictionary()),
        operands(op->operand_begin(), op->operand_end()),
        successors(op->successor_begin(), op->successor_end()) {
    if (OpaqueProperties prop = op->getPropertiesStorage()) {
      // Make a copy of the properties.
      propertiesStorage = operator new(op->getPropertiesStorageSize());
      OpaqueProperties propCopy(propertiesStorage);
      name.initOpProperties(propCopy, /*init=*/prop);
    }
  }

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::ModifyOperation;
  }

  ~ModifyOperationRewrite() override {
    assert(!propertiesStorage &&
           "rewrite was neither committed nor rolled back");
  }

  void commit() override {
    if (propertiesStorage) {
      OpaqueProperties propCopy(propertiesStorage);
      // Note: The operation may have been erased in the mean time, so
      // OperationName must be stored in this object.
      name.destroyOpProperties(propCopy);
      operator delete(propertiesStorage);
      propertiesStorage = nullptr;
    }
  }

  void rollback() override {
    op->setLoc(loc);
    op->setAttrs(attrs);
    op->setOperands(operands);
    for (const auto &it : llvm::enumerate(successors))
      op->setSuccessor(it.value(), it.index());
    if (propertiesStorage) {
      OpaqueProperties propCopy(propertiesStorage);
      op->copyProperties(propCopy);
      name.destroyOpProperties(propCopy);
      operator delete(propertiesStorage);
      propertiesStorage = nullptr;
    }
  }

private:
  OperationName name;
  LocationAttr loc;
  DictionaryAttr attrs;
  SmallVector<Value, 8> operands;
  SmallVector<Block *, 2> successors;
  void *propertiesStorage = nullptr;
};

/// Replacing an operation. Erasing an operation is treated as a special case
/// with "null" replacements. This rewrite is not immediately reflected in the
/// IR. An internal IR mapping is updated, but values are not replaced and the
/// original op is not erased until the rewrite is committed.
class ReplaceOperationRewrite : public OperationRewrite {
public:
  ReplaceOperationRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                          Operation *op, const TypeConverter *converter,
                          bool changedResults)
      : OperationRewrite(Kind::ReplaceOperation, rewriterImpl, op),
        converter(converter), changedResults(changedResults) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::ReplaceOperation;
  }

  void commit() override;

  void rollback() override;

  void cleanup() override;

  const TypeConverter *getConverter() const { return converter; }

  bool hasChangedResults() const { return changedResults; }

private:
  /// An optional type converter that can be used to materialize conversions
  /// between the new and old values if necessary.
  const TypeConverter *converter;

  /// A boolean flag that indicates whether result types have changed or not.
  bool changedResults;
};

class CreateOperationRewrite : public OperationRewrite {
public:
  CreateOperationRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                         Operation *op)
      : OperationRewrite(Kind::CreateOperation, rewriterImpl, op) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::CreateOperation;
  }

  void rollback() override;
};

/// The type of materialization.
enum MaterializationKind {
  /// This materialization materializes a conversion for an illegal block
  /// argument type, to a legal one.
  Argument,

  /// This materialization materializes a conversion from an illegal type to a
  /// legal one.
  Target
};

/// An unresolved materialization, i.e., a "builtin.unrealized_conversion_cast"
/// op. Unresolved materializations are erased at the end of the dialect
/// conversion.
class UnresolvedMaterializationRewrite : public OperationRewrite {
public:
  UnresolvedMaterializationRewrite(
      ConversionPatternRewriterImpl &rewriterImpl,
      UnrealizedConversionCastOp op, const TypeConverter *converter = nullptr,
      MaterializationKind kind = MaterializationKind::Target,
      Type origOutputType = nullptr)
      : OperationRewrite(Kind::UnresolvedMaterialization, rewriterImpl, op),
        converterAndKind(converter, kind), origOutputType(origOutputType) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::UnresolvedMaterialization;
  }

  UnrealizedConversionCastOp getOperation() const {
    return cast<UnrealizedConversionCastOp>(op);
  }

  void rollback() override;

  void cleanup() override;

  /// Return the type converter of this materialization (which may be null).
  const TypeConverter *getConverter() const {
    return converterAndKind.getPointer();
  }

  /// Return the kind of this materialization.
  MaterializationKind getMaterializationKind() const {
    return converterAndKind.getInt();
  }

  /// Set the kind of this materialization.
  void setMaterializationKind(MaterializationKind kind) {
    converterAndKind.setInt(kind);
  }

  /// Return the original illegal output type of the input values.
  Type getOrigOutputType() const { return origOutputType; }

private:
  /// The corresponding type converter to use when resolving this
  /// materialization, and the kind of this materialization.
  llvm::PointerIntPair<const TypeConverter *, 1, MaterializationKind>
      converterAndKind;

  /// The original output type. This is only used for argument conversions.
  Type origOutputType;
};
} // namespace

/// Return "true" if there is an operation rewrite that matches the specified
/// rewrite type and operation among the given rewrites.
template <typename RewriteTy, typename R>
static bool hasRewrite(R &&rewrites, Operation *op) {
  return any_of(std::move(rewrites), [&](auto &rewrite) {
    auto *rewriteTy = dyn_cast<RewriteTy>(rewrite.get());
    return rewriteTy && rewriteTy->getOperation() == op;
  });
}

/// Find the single rewrite object of the specified type and block among the
/// given rewrites. In debug mode, asserts that there is mo more than one such
/// object. Return "nullptr" if no object was found.
template <typename RewriteTy, typename R>
static RewriteTy *findSingleRewrite(R &&rewrites, Block *block) {
  RewriteTy *result = nullptr;
  for (auto &rewrite : rewrites) {
    auto *rewriteTy = dyn_cast<RewriteTy>(rewrite.get());
    if (rewriteTy && rewriteTy->getBlock() == block) {
#ifndef NDEBUG
      assert(!result && "expected single matching rewrite");
      result = rewriteTy;
#else
      return rewriteTy;
#endif // NDEBUG
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// ConversionPatternRewriterImpl
//===----------------------------------------------------------------------===//
namespace mlir {
namespace detail {
struct ConversionPatternRewriterImpl : public RewriterBase::Listener {
  explicit ConversionPatternRewriterImpl(MLIRContext *ctx,
                                         const ConversionConfig &config)
      : eraseRewriter(ctx), config(config) {}

  //===--------------------------------------------------------------------===//
  // State Management
  //===--------------------------------------------------------------------===//

  /// Return the current state of the rewriter.
  RewriterState getCurrentState();

  /// Apply all requested operation rewrites. This method is invoked when the
  /// conversion process succeeds.
  void applyRewrites();

  /// Reset the state of the rewriter to a previously saved point.
  void resetState(RewriterState state);

  /// Append a rewrite. Rewrites are committed upon success and rolled back upon
  /// failure.
  template <typename RewriteTy, typename... Args>
  void appendRewrite(Args &&...args) {
    rewrites.push_back(
        std::make_unique<RewriteTy>(*this, std::forward<Args>(args)...));
  }

  /// Undo the rewrites (motions, splits) one by one in reverse order until
  /// "numRewritesToKeep" rewrites remains.
  void undoRewrites(unsigned numRewritesToKeep = 0);

  /// Remap the given values to those with potentially different types. Returns
  /// success if the values could be remapped, failure otherwise. `valueDiagTag`
  /// is the tag used when describing a value within a diagnostic, e.g.
  /// "operand".
  LogicalResult remapValues(StringRef valueDiagTag,
                            std::optional<Location> inputLoc,
                            PatternRewriter &rewriter, ValueRange values,
                            SmallVectorImpl<Value> &remapped);

  /// Return "true" if the given operation is ignored, and does not need to be
  /// converted.
  bool isOpIgnored(Operation *op) const;

  /// Return "true" if the given operation was replaced or erased.
  bool wasOpReplaced(Operation *op) const;

  //===--------------------------------------------------------------------===//
  // Type Conversion
  //===--------------------------------------------------------------------===//

  /// Attempt to convert the signature of the given block, if successful a new
  /// block is returned containing the new arguments. Returns `block` if it did
  /// not require conversion.
  FailureOr<Block *> convertBlockSignature(
      Block *block, const TypeConverter *converter,
      TypeConverter::SignatureConversion *conversion = nullptr);

  /// Convert the types of non-entry block arguments within the given region.
  LogicalResult convertNonEntryRegionTypes(
      Region *region, const TypeConverter &converter,
      ArrayRef<TypeConverter::SignatureConversion> blockConversions = {});

  /// Apply a signature conversion on the given region, using `converter` for
  /// materializations if not null.
  Block *
  applySignatureConversion(Region *region,
                           TypeConverter::SignatureConversion &conversion,
                           const TypeConverter *converter);

  /// Convert the types of block arguments within the given region.
  FailureOr<Block *>
  convertRegionTypes(Region *region, const TypeConverter &converter,
                     TypeConverter::SignatureConversion *entryConversion);

  /// Apply the given signature conversion on the given block. The new block
  /// containing the updated signature is returned. If no conversions were
  /// necessary, e.g. if the block has no arguments, `block` is returned.
  /// `converter` is used to generate any necessary cast operations that
  /// translate between the origin argument types and those specified in the
  /// signature conversion.
  Block *applySignatureConversion(
      Block *block, const TypeConverter *converter,
      TypeConverter::SignatureConversion &signatureConversion);

  //===--------------------------------------------------------------------===//
  // Materializations
  //===--------------------------------------------------------------------===//
  /// Build an unresolved materialization operation given an output type and set
  /// of input operands.
  Value buildUnresolvedMaterialization(MaterializationKind kind,
                                       Block *insertBlock,
                                       Block::iterator insertPt, Location loc,
                                       ValueRange inputs, Type outputType,
                                       Type origOutputType,
                                       const TypeConverter *converter);

  Value buildUnresolvedArgumentMaterialization(Block *block, Location loc,
                                               ValueRange inputs,
                                               Type origOutputType,
                                               Type outputType,
                                               const TypeConverter *converter);

  Value buildUnresolvedTargetMaterialization(Location loc, Value input,
                                             Type outputType,
                                             const TypeConverter *converter);

  //===--------------------------------------------------------------------===//
  // Rewriter Notification Hooks
  //===--------------------------------------------------------------------===//

  //// Notifies that an op was inserted.
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;

  /// Notifies that an op is about to be replaced with the given values.
  void notifyOpReplaced(Operation *op, ValueRange newValues);

  /// Notifies that a block is about to be erased.
  void notifyBlockIsBeingErased(Block *block);

  /// Notifies that a block was inserted.
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override;

  /// Notifies that a block was split.
  void notifySplitBlock(Block *block, Block *continuation);

  /// Notifies that a block is being inlined into another block.
  void notifyBlockBeingInlined(Block *block, Block *srcBlock,
                               Block::iterator before);

  /// Notifies that a pattern match failed for the given reason.
  void
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

  //===--------------------------------------------------------------------===//
  // IR Erasure
  //===--------------------------------------------------------------------===//

  /// A rewriter that keeps track of erased ops and blocks. It ensures that no
  /// operation or block is erased multiple times. This rewriter assumes that
  /// no new IR is created between calls to `eraseOp`/`eraseBlock`.
  struct SingleEraseRewriter : public RewriterBase, RewriterBase::Listener {
  public:
    SingleEraseRewriter(MLIRContext *context)
        : RewriterBase(context, /*listener=*/this) {}

    /// Erase the given op (unless it was already erased).
    void eraseOp(Operation *op) override {
      if (erased.contains(op))
        return;
      op->dropAllUses();
      RewriterBase::eraseOp(op);
    }

    /// Erase the given block (unless it was already erased).
    void eraseBlock(Block *block) override {
      if (erased.contains(block))
        return;
      assert(block->empty() && "expected empty block");
      block->dropAllDefinedValueUses();
      RewriterBase::eraseBlock(block);
    }

    void notifyOperationErased(Operation *op) override { erased.insert(op); }
    void notifyBlockErased(Block *block) override { erased.insert(block); }

    /// Pointers to all erased operations and blocks.
    SetVector<void *> erased;
  };

  //===--------------------------------------------------------------------===//
  // State
  //===--------------------------------------------------------------------===//

  /// This rewriter must be used for erasing ops/blocks.
  SingleEraseRewriter eraseRewriter;

  // Mapping between replaced values that differ in type. This happens when
  // replacing a value with one of a different type.
  ConversionValueMapping mapping;

  /// Ordered list of block operations (creations, splits, motions).
  SmallVector<std::unique_ptr<IRRewrite>> rewrites;

  /// A set of operations that should no longer be considered for legalization.
  /// E.g., ops that are recursively legal. Ops that were replaced/erased are
  /// tracked separately.
  SetVector<Operation *> ignoredOps;

  /// A set of operations that were replaced/erased. Such ops are not erased
  /// immediately but only when the dialect conversion succeeds. In the mean
  /// time, they should no longer be considered for legalization and any attempt
  /// to modify/access them is invalid rewriter API usage.
  SetVector<Operation *> replacedOps;

  /// The current type converter, or nullptr if no type converter is currently
  /// active.
  const TypeConverter *currentTypeConverter = nullptr;

  /// A mapping of regions to type converters that should be used when
  /// converting the arguments of blocks within that region.
  DenseMap<Region *, const TypeConverter *> regionToConverter;

  /// Dialect conversion configuration.
  const ConversionConfig &config;

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

void IRRewrite::eraseOp(Operation *op) {
  rewriterImpl.eraseRewriter.eraseOp(op);
}

void IRRewrite::eraseBlock(Block *block) {
  rewriterImpl.eraseRewriter.eraseBlock(block);
}

const ConversionConfig &IRRewrite::getConfig() const {
  return rewriterImpl.config;
}

void BlockTypeConversionRewrite::commit() {
  // Process the remapping for each of the original arguments.
  for (auto [origArg, info] :
       llvm::zip_equal(origBlock->getArguments(), argInfo)) {
    // Handle the case of a 1->0 value mapping.
    if (!info) {
      if (Value newArg =
              rewriterImpl.mapping.lookupOrNull(origArg, origArg.getType()))
        origArg.replaceAllUsesWith(newArg);
      continue;
    }

    // Otherwise this is a 1->1+ value mapping.
    Value castValue = info->castValue;
    assert(info->newArgSize >= 1 && castValue && "expected 1->1+ mapping");

    // If the argument is still used, replace it with the generated cast.
    if (!origArg.use_empty()) {
      origArg.replaceAllUsesWith(
          rewriterImpl.mapping.lookupOrDefault(castValue, origArg.getType()));
    }
  }

  assert(origBlock->empty() && "expected empty block");
  origBlock->dropAllDefinedValueUses();
  delete origBlock;
  origBlock = nullptr;
}

void BlockTypeConversionRewrite::rollback() {
  // Drop all uses of the new block arguments and replace uses of the new block.
  for (int i = block->getNumArguments() - 1; i >= 0; --i)
    block->getArgument(i).dropAllUses();
  block->replaceAllUsesWith(origBlock);

  // Move the operations back the original block, move the original block back
  // into its original location and the delete the new block.
  origBlock->getOperations().splice(origBlock->end(), block->getOperations());
  block->getParent()->getBlocks().insert(Region::iterator(block), origBlock);
  eraseBlock(block);
}

LogicalResult BlockTypeConversionRewrite::materializeLiveConversions(
    function_ref<Operation *(Value)> findLiveUser) {
  // Process the remapping for each of the original arguments.
  for (auto it : llvm::enumerate(origBlock->getArguments())) {
    BlockArgument origArg = it.value();
    // Note: `block` may be detached, so OpBuilder::atBlockBegin cannot be used.
    OpBuilder builder(it.value().getContext(), /*listener=*/&rewriterImpl);
    builder.setInsertionPointToStart(block);

    // If the type of this argument changed and the argument is still live, we
    // need to materialize a conversion.
    if (rewriterImpl.mapping.lookupOrNull(origArg, origArg.getType()))
      continue;
    Operation *liveUser = findLiveUser(origArg);
    if (!liveUser)
      continue;

    Value replacementValue = rewriterImpl.mapping.lookupOrDefault(origArg);
    bool isDroppedArg = replacementValue == origArg;
    if (!isDroppedArg)
      builder.setInsertionPointAfterValue(replacementValue);
    Value newArg;
    if (converter) {
      newArg = converter->materializeSourceConversion(
          builder, origArg.getLoc(), origArg.getType(),
          isDroppedArg ? ValueRange() : ValueRange(replacementValue));
      assert((!newArg || newArg.getType() == origArg.getType()) &&
             "materialization hook did not provide a value of the expected "
             "type");
    }
    if (!newArg) {
      InFlightDiagnostic diag =
          emitError(origArg.getLoc())
          << "failed to materialize conversion for block argument #"
          << it.index() << " that remained live after conversion, type was "
          << origArg.getType();
      if (!isDroppedArg)
        diag << ", with target type " << replacementValue.getType();
      diag.attachNote(liveUser->getLoc())
          << "see existing live user here: " << *liveUser;
      return failure();
    }
    rewriterImpl.mapping.map(origArg, newArg);
  }
  return success();
}

void ReplaceBlockArgRewrite::commit() {
  Value repl = rewriterImpl.mapping.lookupOrNull(arg, arg.getType());
  if (!repl)
    return;

  if (isa<BlockArgument>(repl)) {
    arg.replaceAllUsesWith(repl);
    return;
  }

  // If the replacement value is an operation, we check to make sure that we
  // don't replace uses that are within the parent operation of the
  // replacement value.
  Operation *replOp = cast<OpResult>(repl).getOwner();
  Block *replBlock = replOp->getBlock();
  arg.replaceUsesWithIf(repl, [&](OpOperand &operand) {
    Operation *user = operand.getOwner();
    return user->getBlock() != replBlock || replOp->isBeforeInBlock(user);
  });
}

void ReplaceBlockArgRewrite::rollback() { rewriterImpl.mapping.erase(arg); }

void ReplaceOperationRewrite::commit() {
  for (OpResult result : op->getResults())
    if (Value newValue =
            rewriterImpl.mapping.lookupOrNull(result, result.getType()))
      result.replaceAllUsesWith(newValue);
  if (getConfig().unlegalizedOps)
    getConfig().unlegalizedOps->erase(op);
  // Do not erase the operation yet. It may still be referenced in `mapping`.
  op->getBlock()->getOperations().remove(op);
}

void ReplaceOperationRewrite::rollback() {
  for (auto result : op->getResults())
    rewriterImpl.mapping.erase(result);
}

void ReplaceOperationRewrite::cleanup() { eraseOp(op); }

void CreateOperationRewrite::rollback() {
  for (Region &region : op->getRegions()) {
    while (!region.getBlocks().empty())
      region.getBlocks().remove(region.getBlocks().begin());
  }
  op->dropAllUses();
  eraseOp(op);
}

void UnresolvedMaterializationRewrite::rollback() {
  if (getMaterializationKind() == MaterializationKind::Target) {
    for (Value input : op->getOperands())
      rewriterImpl.mapping.erase(input);
  }
  eraseOp(op);
}

void UnresolvedMaterializationRewrite::cleanup() { eraseOp(op); }

void ConversionPatternRewriterImpl::applyRewrites() {
  // Commit all rewrites.
  for (auto &rewrite : rewrites)
    rewrite->commit();
  for (auto &rewrite : rewrites)
    rewrite->cleanup();
}

//===----------------------------------------------------------------------===//
// State Management

RewriterState ConversionPatternRewriterImpl::getCurrentState() {
  return RewriterState(rewrites.size(), ignoredOps.size(),
                       eraseRewriter.erased.size(), replacedOps.size());
}

void ConversionPatternRewriterImpl::resetState(RewriterState state) {
  // Undo any rewrites.
  undoRewrites(state.numRewrites);

  // Pop all of the recorded ignored operations that are no longer valid.
  while (ignoredOps.size() != state.numIgnoredOperations)
    ignoredOps.pop_back();

  while (eraseRewriter.erased.size() != state.numErased)
    eraseRewriter.erased.pop_back();

  while (replacedOps.size() != state.numReplacedOps)
    replacedOps.pop_back();
}

void ConversionPatternRewriterImpl::undoRewrites(unsigned numRewritesToKeep) {
  for (auto &rewrite :
       llvm::reverse(llvm::drop_begin(rewrites, numRewritesToKeep)))
    rewrite->rollback();
  rewrites.resize(numRewritesToKeep);
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
        notifyMatchFailure(operandLoc, [=](Diagnostic &diag) {
          diag << "unable to convert type for " << valueDiagTag << " #"
               << it.index() << ", type was " << origType;
        });
        return failure();
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
          operandLoc, newOperand, desiredType, currentTypeConverter);
      mapping.map(mapping.lookupOrDefault(newOperand), castValue);
      newOperand = castValue;
    }
    remapped.push_back(newOperand);
  }
  return success();
}

bool ConversionPatternRewriterImpl::isOpIgnored(Operation *op) const {
  // Check to see if this operation is ignored or was replaced.
  return replacedOps.count(op) || ignoredOps.count(op);
}

bool ConversionPatternRewriterImpl::wasOpReplaced(Operation *op) const {
  // Check to see if this operation was replaced.
  return replacedOps.count(op);
}

//===----------------------------------------------------------------------===//
// Type Conversion

FailureOr<Block *> ConversionPatternRewriterImpl::convertBlockSignature(
    Block *block, const TypeConverter *converter,
    TypeConverter::SignatureConversion *conversion) {
  if (conversion)
    return applySignatureConversion(block, converter, *conversion);

  // If a converter wasn't provided, and the block wasn't already converted,
  // there is nothing we can do.
  if (!converter)
    return failure();

  // Try to convert the signature for the block with the provided converter.
  if (auto conversion = converter->convertBlockSignature(block))
    return applySignatureConversion(block, converter, *conversion);
  return failure();
}

Block *ConversionPatternRewriterImpl::applySignatureConversion(
    Region *region, TypeConverter::SignatureConversion &conversion,
    const TypeConverter *converter) {
  if (!region->empty())
    return *convertBlockSignature(&region->front(), converter, &conversion);
  return nullptr;
}

FailureOr<Block *> ConversionPatternRewriterImpl::convertRegionTypes(
    Region *region, const TypeConverter &converter,
    TypeConverter::SignatureConversion *entryConversion) {
  regionToConverter[region] = &converter;
  if (region->empty())
    return nullptr;

  if (failed(convertNonEntryRegionTypes(region, converter)))
    return failure();

  FailureOr<Block *> newEntry =
      convertBlockSignature(&region->front(), &converter, entryConversion);
  return newEntry;
}

LogicalResult ConversionPatternRewriterImpl::convertNonEntryRegionTypes(
    Region *region, const TypeConverter &converter,
    ArrayRef<TypeConverter::SignatureConversion> blockConversions) {
  regionToConverter[region] = &converter;
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

Block *ConversionPatternRewriterImpl::applySignatureConversion(
    Block *block, const TypeConverter *converter,
    TypeConverter::SignatureConversion &signatureConversion) {
  MLIRContext *ctx = eraseRewriter.getContext();

  // If no arguments are being changed or added, there is nothing to do.
  unsigned origArgCount = block->getNumArguments();
  auto convertedTypes = signatureConversion.getConvertedTypes();
  if (llvm::equal(block->getArgumentTypes(), convertedTypes))
    return block;

  // Split the block at the beginning to get a new block to use for the updated
  // signature.
  Block *newBlock = block->splitBlock(block->begin());
  block->replaceAllUsesWith(newBlock);
  // Unlink the block, but do not erase it yet, so that the change can be rolled
  // back.
  block->getParent()->getBlocks().remove(block);

  // Map all new arguments to the location of the argument they originate from.
  SmallVector<Location> newLocs(convertedTypes.size(),
                                Builder(ctx).getUnknownLoc());
  for (unsigned i = 0; i < origArgCount; ++i) {
    auto inputMap = signatureConversion.getInputMapping(i);
    if (!inputMap || inputMap->replacementValue)
      continue;
    Location origLoc = block->getArgument(i).getLoc();
    for (unsigned j = 0; j < inputMap->size; ++j)
      newLocs[inputMap->inputNo + j] = origLoc;
  }

  SmallVector<Value, 4> newArgRange(
      newBlock->addArguments(convertedTypes, newLocs));
  ArrayRef<Value> newArgs(newArgRange);

  // Remap each of the original arguments as determined by the signature
  // conversion.
  SmallVector<std::optional<ConvertedArgInfo>, 1> argInfo;
  argInfo.resize(origArgCount);

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
      appendRewrite<ReplaceBlockArgRewrite>(block, origArg);
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
          newBlock, origArg.getLoc(), replArgs, origOutputType, outputType,
          converter);
    }

    mapping.map(origArg, newArg);
    appendRewrite<ReplaceBlockArgRewrite>(block, origArg);
    argInfo[i] = ConvertedArgInfo(inputMap->inputNo, inputMap->size, newArg);
  }

  appendRewrite<BlockTypeConversionRewrite>(newBlock, block, argInfo,
                                            converter);
  return newBlock;
}

//===----------------------------------------------------------------------===//
// Materializations
//===----------------------------------------------------------------------===//

/// Build an unresolved materialization operation given an output type and set
/// of input operands.
Value ConversionPatternRewriterImpl::buildUnresolvedMaterialization(
    MaterializationKind kind, Block *insertBlock, Block::iterator insertPt,
    Location loc, ValueRange inputs, Type outputType, Type origOutputType,
    const TypeConverter *converter) {
  // Avoid materializing an unnecessary cast.
  if (inputs.size() == 1 && inputs.front().getType() == outputType)
    return inputs.front();

  // Create an unresolved materialization. We use a new OpBuilder to avoid
  // tracking the materialization like we do for other operations.
  OpBuilder builder(insertBlock, insertPt);
  auto convertOp =
      builder.create<UnrealizedConversionCastOp>(loc, outputType, inputs);
  appendRewrite<UnresolvedMaterializationRewrite>(convertOp, converter, kind,
                                                  origOutputType);
  return convertOp.getResult(0);
}
Value ConversionPatternRewriterImpl::buildUnresolvedArgumentMaterialization(
    Block *block, Location loc, ValueRange inputs, Type origOutputType,
    Type outputType, const TypeConverter *converter) {
  return buildUnresolvedMaterialization(MaterializationKind::Argument, block,
                                        block->begin(), loc, inputs, outputType,
                                        origOutputType, converter);
}
Value ConversionPatternRewriterImpl::buildUnresolvedTargetMaterialization(
    Location loc, Value input, Type outputType,
    const TypeConverter *converter) {
  Block *insertBlock = input.getParentBlock();
  Block::iterator insertPt = insertBlock->begin();
  if (OpResult inputRes = dyn_cast<OpResult>(input))
    insertPt = ++inputRes.getOwner()->getIterator();

  return buildUnresolvedMaterialization(MaterializationKind::Target,
                                        insertBlock, insertPt, loc, input,
                                        outputType, outputType, converter);
}

//===----------------------------------------------------------------------===//
// Rewriter Notification Hooks

void ConversionPatternRewriterImpl::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  assert(!wasOpReplaced(op->getParentOp()) &&
         "attempting to insert into a block within a replaced/erased op");

  if (!previous.isSet()) {
    // This is a newly created op.
    appendRewrite<CreateOperationRewrite>(op);
    return;
  }
  Operation *prevOp = previous.getPoint() == previous.getBlock()->end()
                          ? nullptr
                          : &*previous.getPoint();
  appendRewrite<MoveOperationRewrite>(op, previous.getBlock(), prevOp);
}

void ConversionPatternRewriterImpl::notifyOpReplaced(Operation *op,
                                                     ValueRange newValues) {
  assert(newValues.size() == op->getNumResults());
  assert(!ignoredOps.contains(op) && "operation was already replaced");

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

  appendRewrite<ReplaceOperationRewrite>(op, currentTypeConverter,
                                         resultChanged);

  // Mark this operation and all nested ops as replaced.
  op->walk([&](Operation *op) { replacedOps.insert(op); });
}

void ConversionPatternRewriterImpl::notifyBlockIsBeingErased(Block *block) {
  Region *region = block->getParent();
  Block *origNextBlock = block->getNextNode();
  appendRewrite<EraseBlockRewrite>(block, region, origNextBlock);
}

void ConversionPatternRewriterImpl::notifyBlockInserted(
    Block *block, Region *previous, Region::iterator previousIt) {
  assert(!wasOpReplaced(block->getParentOp()) &&
         "attempting to insert into a region within a replaced/erased op");

  if (!previous) {
    // This is a newly created block.
    appendRewrite<CreateBlockRewrite>(block);
    return;
  }
  Block *prevBlock = previousIt == previous->end() ? nullptr : &*previousIt;
  appendRewrite<MoveBlockRewrite>(block, previous, prevBlock);
}

void ConversionPatternRewriterImpl::notifySplitBlock(Block *block,
                                                     Block *continuation) {
  appendRewrite<SplitBlockRewrite>(continuation, block);
}

void ConversionPatternRewriterImpl::notifyBlockBeingInlined(
    Block *block, Block *srcBlock, Block::iterator before) {
  appendRewrite<InlineBlockRewrite>(block, srcBlock, before);
}

void ConversionPatternRewriterImpl::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    logger.startLine() << "** Failure : " << diag.str() << "\n";
    if (config.notifyCallback)
      config.notifyCallback(diag);
  });
}

//===----------------------------------------------------------------------===//
// ConversionPatternRewriter
//===----------------------------------------------------------------------===//

ConversionPatternRewriter::ConversionPatternRewriter(
    MLIRContext *ctx, const ConversionConfig &config)
    : PatternRewriter(ctx),
      impl(new detail::ConversionPatternRewriterImpl(ctx, config)) {
  setListener(impl.get());
}

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

void ConversionPatternRewriter::replaceOp(Operation *op, Operation *newOp) {
  assert(op && newOp && "expected non-null op");
  replaceOp(op, newOp->getResults());
}

void ConversionPatternRewriter::replaceOp(Operation *op, ValueRange newValues) {
  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");
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
  assert(!impl->wasOpReplaced(block->getParentOp()) &&
         "attempting to erase a block within a replaced/erased op");

  // Mark all ops for erasure.
  for (Operation &op : *block)
    eraseOp(&op);

  // Unlink the block from its parent region. The block is kept in the rewrite
  // object and will be actually destroyed when rewrites are applied. This
  // allows us to keep the operations in the block live and undo the removal by
  // re-inserting the block.
  impl->notifyBlockIsBeingErased(block);
  block->getParent()->getBlocks().remove(block);
}

Block *ConversionPatternRewriter::applySignatureConversion(
    Region *region, TypeConverter::SignatureConversion &conversion,
    const TypeConverter *converter) {
  assert(!impl->wasOpReplaced(region->getParentOp()) &&
         "attempting to apply a signature conversion to a block within a "
         "replaced/erased op");
  return impl->applySignatureConversion(region, conversion, converter);
}

FailureOr<Block *> ConversionPatternRewriter::convertRegionTypes(
    Region *region, const TypeConverter &converter,
    TypeConverter::SignatureConversion *entryConversion) {
  assert(!impl->wasOpReplaced(region->getParentOp()) &&
         "attempting to apply a signature conversion to a block within a "
         "replaced/erased op");
  return impl->convertRegionTypes(region, converter, entryConversion);
}

LogicalResult ConversionPatternRewriter::convertNonEntryRegionTypes(
    Region *region, const TypeConverter &converter,
    ArrayRef<TypeConverter::SignatureConversion> blockConversions) {
  assert(!impl->wasOpReplaced(region->getParentOp()) &&
         "attempting to apply a signature conversion to a block within a "
         "replaced/erased op");
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
  impl->appendRewrite<ReplaceBlockArgRewrite>(from.getOwner(), from);
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

Block *ConversionPatternRewriter::splitBlock(Block *block,
                                             Block::iterator before) {
  assert(!impl->wasOpReplaced(block->getParentOp()) &&
         "attempting to split a block within a replaced/erased op");
  auto *continuation = block->splitBlock(before);
  impl->notifySplitBlock(block, continuation);
  return continuation;
}

void ConversionPatternRewriter::inlineBlockBefore(Block *source, Block *dest,
                                                  Block::iterator before,
                                                  ValueRange argValues) {
#ifndef NDEBUG
  assert(argValues.size() == source->getNumArguments() &&
         "incorrect # of argument replacement values");
  assert(!impl->wasOpReplaced(source->getParentOp()) &&
         "attempting to inline a block from a replaced/erased op");
  assert(!impl->wasOpReplaced(dest->getParentOp()) &&
         "attempting to inline a block into a replaced/erased op");
  auto opIgnored = [&](Operation *op) { return impl->isOpIgnored(op); };
  // The source block will be deleted, so it should not have any users (i.e.,
  // there should be no predecessors).
  assert(llvm::all_of(source->getUsers(), opIgnored) &&
         "expected 'source' to have no predecessors");
#endif // NDEBUG

  impl->notifyBlockBeingInlined(dest, source, before);
  for (auto it : llvm::zip(source->getArguments(), argValues))
    replaceUsesOfBlockArgument(std::get<0>(it), std::get<1>(it));
  dest->getOperations().splice(before, source->getOperations());
  eraseBlock(source);
}

void ConversionPatternRewriter::startOpModification(Operation *op) {
  assert(!impl->wasOpReplaced(op) &&
         "attempting to modify a replaced/erased op");
#ifndef NDEBUG
  impl->pendingRootUpdates.insert(op);
#endif
  impl->appendRewrite<ModifyOperationRewrite>(op);
}

void ConversionPatternRewriter::finalizeOpModification(Operation *op) {
  assert(!impl->wasOpReplaced(op) &&
         "attempting to modify a replaced/erased op");
  PatternRewriter::finalizeOpModification(op);
  // There is nothing to do here, we only need to track the operation at the
  // start of the update.
#ifndef NDEBUG
  assert(impl->pendingRootUpdates.erase(op) &&
         "operation did not have a pending in-place update");
#endif
}

void ConversionPatternRewriter::cancelOpModification(Operation *op) {
#ifndef NDEBUG
  assert(impl->pendingRootUpdates.erase(op) &&
         "operation did not have a pending in-place update");
#endif
  // Erase the last update for this operation.
  auto it = llvm::find_if(
      llvm::reverse(impl->rewrites), [&](std::unique_ptr<IRRewrite> &rewrite) {
        auto *modifyRewrite = dyn_cast<ModifyOperationRewrite>(rewrite.get());
        return modifyRewrite && modifyRewrite->getOperation() == op;
      });
  assert(it != impl->rewrites.rend() && "no root update started on op");
  (*it)->rollback();
  int updateIdx = std::prev(impl->rewrites.rend()) - it;
  impl->rewrites.erase(impl->rewrites.begin() + updateIdx);
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

  OperationLegalizer(const ConversionTarget &targetInfo,
                     const FrozenRewritePatternSet &patterns);

  /// Returns true if the given operation is known to be illegal on the target.
  bool isIllegal(Operation *op) const;

  /// Attempt to legalize the given operation. Returns success if the operation
  /// was legalized, failure otherwise.
  LogicalResult legalize(Operation *op, ConversionPatternRewriter &rewriter);

  /// Returns the conversion target in use by the legalizer.
  const ConversionTarget &getTarget() { return target; }

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
  LogicalResult
  legalizePatternBlockRewrites(Operation *op,
                               ConversionPatternRewriter &rewriter,
                               ConversionPatternRewriterImpl &impl,
                               RewriterState &state, RewriterState &newState);
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
  const ConversionTarget &target;

  /// The pattern applicator to use for conversions.
  PatternApplicator applicator;
};
} // namespace

OperationLegalizer::OperationLegalizer(const ConversionTarget &targetInfo,
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
    if (legalityInfo->isRecursivelyLegal) {
      op->walk([&](Operation *nested) {
        if (op != nested)
          rewriter.getImpl().ignoredOps.insert(nested);
      });
    }

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
  for (unsigned i = curState.numRewrites, e = rewriterImpl.rewrites.size();
       i != e; ++i) {
    auto *createOp =
        dyn_cast<CreateOperationRewrite>(rewriterImpl.rewrites[i].get());
    if (!createOp)
      continue;
    if (failed(legalize(createOp->getOperation(), rewriter))) {
      LLVM_DEBUG(logFailure(rewriterImpl.logger,
                            "failed to legalize generated constant '{0}'",
                            createOp->getOperation()->getName()));
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
    assert(rewriterImpl.pendingRootUpdates.empty() && "dangling root updates");
    LLVM_DEBUG({
      logFailure(rewriterImpl.logger, "pattern failed to match");
      if (rewriterImpl.config.notifyCallback) {
        Diagnostic diag(op->getLoc(), DiagnosticSeverity::Remark);
        diag << "Failed to apply pattern \"" << pattern.getDebugName()
             << "\" on op:\n"
             << *op;
        rewriterImpl.config.notifyCallback(diag);
      }
    });
    rewriterImpl.resetState(curState);
    appliedPatterns.erase(&pattern);
  };

  // Functor that performs additional legalization when a pattern is
  // successfully applied.
  auto onSuccess = [&](const Pattern &pattern) {
    assert(rewriterImpl.pendingRootUpdates.empty() && "dangling root updates");
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
  // Check that the root was either replaced or updated in place.
  auto newRewrites = llvm::drop_begin(impl.rewrites, curState.numRewrites);
  auto replacedRoot = [&] {
    return hasRewrite<ReplaceOperationRewrite>(newRewrites, op);
  };
  auto updatedRootInPlace = [&] {
    return hasRewrite<ModifyOperationRewrite>(newRewrites, op);
  };
  assert((replacedRoot() || updatedRootInPlace()) &&
         "expected pattern to replace the root operation");
#endif // NDEBUG

  // Legalize each of the actions registered during application.
  RewriterState newState = impl.getCurrentState();
  if (failed(legalizePatternBlockRewrites(op, rewriter, impl, curState,
                                          newState)) ||
      failed(legalizePatternRootUpdates(rewriter, impl, curState, newState)) ||
      failed(legalizePatternCreatedOperations(rewriter, impl, curState,
                                              newState))) {
    return failure();
  }

  LLVM_DEBUG(logSuccess(impl.logger, "pattern applied successfully"));
  return success();
}

LogicalResult OperationLegalizer::legalizePatternBlockRewrites(
    Operation *op, ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &impl, RewriterState &state,
    RewriterState &newState) {
  SmallPtrSet<Operation *, 16> operationsToIgnore;

  // If the pattern moved or created any blocks, make sure the types of block
  // arguments get legalized.
  for (int i = state.numRewrites, e = newState.numRewrites; i != e; ++i) {
    BlockRewrite *rewrite = dyn_cast<BlockRewrite>(impl.rewrites[i].get());
    if (!rewrite)
      continue;
    Block *block = rewrite->getBlock();
    if (isa<BlockTypeConversionRewrite, EraseBlockRewrite,
            ReplaceBlockArgRewrite>(rewrite))
      continue;
    // Only check blocks outside of the current operation.
    Operation *parentOp = block->getParentOp();
    if (!parentOp || parentOp == op || block->getNumArguments() == 0)
      continue;

    // If the region of the block has a type converter, try to convert the block
    // directly.
    if (auto *converter = impl.regionToConverter.lookup(block->getParent())) {
      if (failed(impl.convertBlockSignature(block, converter))) {
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
      for (unsigned i = state.numRewrites, e = impl.rewrites.size(); i != e;
           ++i) {
        auto *createOp =
            dyn_cast<CreateOperationRewrite>(impl.rewrites[i].get());
        if (!createOp)
          continue;
        operationsToIgnore.insert(createOp->getOperation());
      }
    }

    // If this operation should be considered for re-legalization, try it.
    if (operationsToIgnore.insert(parentOp).second &&
        failed(legalize(parentOp, rewriter))) {
      LLVM_DEBUG(logFailure(impl.logger,
                            "operation '{0}'({1}) became illegal after rewrite",
                            parentOp->getName(), parentOp));
      return failure();
    }
  }
  return success();
}

LogicalResult OperationLegalizer::legalizePatternCreatedOperations(
    ConversionPatternRewriter &rewriter, ConversionPatternRewriterImpl &impl,
    RewriterState &state, RewriterState &newState) {
  for (int i = state.numRewrites, e = newState.numRewrites; i != e; ++i) {
    auto *createOp = dyn_cast<CreateOperationRewrite>(impl.rewrites[i].get());
    if (!createOp)
      continue;
    Operation *op = createOp->getOperation();
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
  for (int i = state.numRewrites, e = newState.numRewrites; i != e; ++i) {
    auto *rewrite = dyn_cast<ModifyOperationRewrite>(impl.rewrites[i].get());
    if (!rewrite)
      continue;
    Operation *op = rewrite->getOperation();
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
  std::stable_sort(patternsByDepth.begin(), patternsByDepth.end(),
                   [](const std::pair<const Pattern *, unsigned> &lhs,
                      const std::pair<const Pattern *, unsigned> &rhs) {
                     // First sort by the smaller pattern legalization
                     // depth.
                     if (lhs.second != rhs.second)
                       return lhs.second < rhs.second;

                     // Then sort by the larger pattern benefit.
                     auto lhsBenefit = lhs.first->getBenefit();
                     auto rhsBenefit = rhs.first->getBenefit();
                     return lhsBenefit > rhsBenefit;
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
} // namespace

namespace mlir {
// This class converts operations to a given conversion target via a set of
// rewrite patterns. The conversion behaves differently depending on the
// conversion mode.
struct OperationConverter {
  explicit OperationConverter(const ConversionTarget &target,
                              const FrozenRewritePatternSet &patterns,
                              const ConversionConfig &config,
                              OpConversionMode mode)
      : opLegalizer(target, patterns), config(config), mode(mode) {}

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
      const TypeConverter *replConverter, ConversionPatternRewriter &rewriter,
      ConversionPatternRewriterImpl &rewriterImpl,
      const DenseMap<Value, SmallVector<Value>> &inverseMapping);

  /// The legalizer to use when converting operations.
  OperationLegalizer opLegalizer;

  /// Dialect conversion configuration.
  ConversionConfig config;

  /// The conversion mode to use when legalizing operations.
  OpConversionMode mode;
};
} // namespace mlir

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
    // explicitly marked as illegal. If the user provided a `unlegalizedOps`
    // set, non-legalizable ops are added to that set.
    if (mode == OpConversionMode::Partial) {
      if (opLegalizer.isIllegal(op))
        return op->emitError()
               << "failed to legalize operation '" << op->getName()
               << "' that was explicitly marked illegal";
      if (config.unlegalizedOps)
        config.unlegalizedOps->insert(op);
    }
  } else if (mode == OpConversionMode::Analysis) {
    // Analysis conversions don't fail if any operations fail to legalize,
    // they are only interested in the operations that were successfully
    // legalized.
    if (config.legalizableOps)
      config.legalizableOps->insert(op);
  }
  return success();
}

LogicalResult OperationConverter::convertOperations(ArrayRef<Operation *> ops) {
  if (ops.empty())
    return success();
  const ConversionTarget &target = opLegalizer.getTarget();

  // Compute the set of operations and blocks to convert.
  SmallVector<Operation *> toConvert;
  for (auto *op : ops) {
    op->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [&](Operation *op) {
          toConvert.push_back(op);
          // Don't check this operation's children for conversion if the
          // operation is recursively legal.
          auto legalityInfo = target.isLegal(op);
          if (legalityInfo && legalityInfo->isRecursivelyLegal)
            return WalkResult::skip();
          return WalkResult::advance();
        });
  }

  // Convert each operation and discard rewrites on failure.
  ConversionPatternRewriter rewriter(ops.front()->getContext(), config);
  ConversionPatternRewriterImpl &rewriterImpl = rewriter.getImpl();

  for (auto *op : toConvert)
    if (failed(convert(rewriter, op)))
      return rewriterImpl.undoRewrites(), failure();

  // Now that all of the operations have been converted, finalize the conversion
  // process to ensure any lingering conversion artifacts are cleaned up and
  // legalized.
  if (failed(finalize(rewriter)))
    return rewriterImpl.undoRewrites(), failure();

  // After a successful conversion, apply rewrites if this is not an analysis
  // conversion.
  if (mode == OpConversionMode::Analysis) {
    rewriterImpl.undoRewrites();
  } else {
    rewriterImpl.applyRewrites();
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

  // Process requested operation replacements.
  for (unsigned i = 0; i < rewriterImpl.rewrites.size(); ++i) {
    auto *opReplacement =
        dyn_cast<ReplaceOperationRewrite>(rewriterImpl.rewrites[i].get());
    if (!opReplacement || !opReplacement->hasChangedResults())
      continue;
    Operation *op = opReplacement->getOperation();
    for (OpResult result : op->getResults()) {
      Value newValue = rewriterImpl.mapping.lookupOrNull(result);

      // If the operation result was replaced with null, all of the uses of this
      // value should be replaced.
      if (!newValue) {
        if (failed(legalizeErasedResult(op, result, rewriterImpl)))
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
      rewriter.setInsertionPoint(op);
      if (failed(legalizeChangedResultType(
              op, result, newValue, opReplacement->getConverter(), rewriter,
              rewriterImpl, *inverseMapping)))
        return failure();
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
  // Note: `rewrites` may be reallocated as the loop is running.
  for (int64_t i = 0; i < static_cast<int64_t>(rewriterImpl.rewrites.size());
       ++i) {
    auto &rewrite = rewriterImpl.rewrites[i];
    if (auto *blockTypeConversionRewrite =
            dyn_cast<BlockTypeConversionRewrite>(rewrite.get()))
      if (failed(blockTypeConversionRewrite->materializeLiveConversions(
              findLiveUser)))
        return failure();
  }
  return success();
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
    DenseMap<Operation *, UnresolvedMaterializationRewrite *>
        &materializationOps,
    ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &rewriterImpl,
    DenseMap<Value, SmallVector<Value>> &inverseMapping,
    SetVector<UnresolvedMaterializationRewrite *> &necessaryMaterializations) {
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

  SetVector<UnresolvedMaterializationRewrite *> worklist;
  for (auto &rewrite : rewriterImpl.rewrites) {
    auto *mat = dyn_cast<UnresolvedMaterializationRewrite>(rewrite.get());
    if (!mat)
      continue;
    materializationOps.try_emplace(mat->getOperation(), mat);
    worklist.insert(mat);
  }
  while (!worklist.empty()) {
    UnresolvedMaterializationRewrite *mat = worklist.pop_back_val();
    UnrealizedConversionCastOp op = mat->getOperation();

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
    auto isBlockArg = [](Value v) { return isa<BlockArgument>(v); };
    if (llvm::any_of(op->getOperands(), isBlockArg) ||
        llvm::any_of(inverseMapping[op->getResult(0)], isBlockArg)) {
      mat->setMaterializationKind(MaterializationKind::Argument);
    }

    // If the materialization does not have any live users, we don't need to
    // generate a user materialization for it.
    // FIXME: For argument materializations, we currently need to check if any
    // of the inverse mapped values are used because some patterns expect blind
    // value replacement even if the types differ in some cases. When those
    // patterns are fixed, we can drop the argument special case here.
    bool isMaterializationLive = isLive(opResult);
    if (mat->getMaterializationKind() == MaterializationKind::Argument)
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
    UnresolvedMaterializationRewrite &mat,
    DenseMap<Operation *, UnresolvedMaterializationRewrite *>
        &materializationOps,
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

  UnrealizedConversionCastOp op = mat.getOperation();
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
  if (const TypeConverter *converter = mat.getConverter()) {
    // FIXME: Determine a suitable insertion location when there are multiple
    // inputs.
    if (inputOperands.size() == 1)
      rewriter.setInsertionPointAfterValue(inputOperands.front());
    else
      rewriter.setInsertionPoint(op);

    Value newMaterialization;
    switch (mat.getMaterializationKind()) {
    case MaterializationKind::Argument:
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
    case MaterializationKind::Target:
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
  inverseMapping = rewriterImpl.mapping.getInverse();

  // As an initial step, compute all of the inserted materializations that we
  // expect to persist beyond the conversion process.
  DenseMap<Operation *, UnresolvedMaterializationRewrite *> materializationOps;
  SetVector<UnresolvedMaterializationRewrite *> necessaryMaterializations;
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
    const TypeConverter *replConverter, ConversionPatternRewriter &rewriter,
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
                                         SmallVectorImpl<Type> &results) const {
  {
    std::shared_lock<decltype(cacheMutex)> cacheReadLock(cacheMutex,
                                                         std::defer_lock);
    if (t.getContext()->isMultithreadingEnabled())
      cacheReadLock.lock();
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
  }
  // Walk the added converters in reverse order to apply the most recently
  // registered first.
  size_t currentCount = results.size();

  std::unique_lock<decltype(cacheMutex)> cacheWriteLock(cacheMutex,
                                                        std::defer_lock);

  for (const ConversionCallbackFn &converter : llvm::reverse(conversions)) {
    if (std::optional<LogicalResult> result = converter(t, results)) {
      if (t.getContext()->isMultithreadingEnabled())
        cacheWriteLock.lock();
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

Type TypeConverter::convertType(Type t) const {
  // Use the multi-type result version to convert the type.
  SmallVector<Type, 1> results;
  if (failed(convertType(t, results)))
    return nullptr;

  // Check to ensure that only one type was produced.
  return results.size() == 1 ? results.front() : nullptr;
}

LogicalResult
TypeConverter::convertTypes(TypeRange types,
                            SmallVectorImpl<Type> &results) const {
  for (Type type : types)
    if (failed(convertType(type, results)))
      return failure();
  return success();
}

bool TypeConverter::isLegal(Type type) const {
  return convertType(type) == type;
}
bool TypeConverter::isLegal(Operation *op) const {
  return isLegal(op->getOperandTypes()) && isLegal(op->getResultTypes());
}

bool TypeConverter::isLegal(Region *region) const {
  return llvm::all_of(*region, [this](Block &block) {
    return isLegal(block.getArgumentTypes());
  });
}

bool TypeConverter::isSignatureLegal(FunctionType ty) const {
  return isLegal(llvm::concat<const Type>(ty.getInputs(), ty.getResults()));
}

LogicalResult
TypeConverter::convertSignatureArg(unsigned inputNo, Type type,
                                   SignatureConversion &result) const {
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
LogicalResult
TypeConverter::convertSignatureArgs(TypeRange types,
                                    SignatureConversion &result,
                                    unsigned origInputOffset) const {
  for (unsigned i = 0, e = types.size(); i != e; ++i)
    if (failed(convertSignatureArg(origInputOffset + i, types[i], result)))
      return failure();
  return success();
}

Value TypeConverter::materializeConversion(
    ArrayRef<MaterializationCallbackFn> materializations, OpBuilder &builder,
    Location loc, Type resultType, ValueRange inputs) const {
  for (const MaterializationCallbackFn &fn : llvm::reverse(materializations))
    if (std::optional<Value> result = fn(builder, resultType, inputs, loc))
      return *result;
  return nullptr;
}

std::optional<TypeConverter::SignatureConversion>
TypeConverter::convertBlockSignature(Block *block) const {
  SignatureConversion conversion(block->getNumArguments());
  if (failed(convertSignatureArgs(block->getArgumentTypes(), conversion)))
    return std::nullopt;
  return conversion;
}

//===----------------------------------------------------------------------===//
// Type attribute conversion
//===----------------------------------------------------------------------===//
TypeConverter::AttributeConversionResult
TypeConverter::AttributeConversionResult::result(Attribute attr) {
  return AttributeConversionResult(attr, resultTag);
}

TypeConverter::AttributeConversionResult
TypeConverter::AttributeConversionResult::na() {
  return AttributeConversionResult(nullptr, naTag);
}

TypeConverter::AttributeConversionResult
TypeConverter::AttributeConversionResult::abort() {
  return AttributeConversionResult(nullptr, abortTag);
}

bool TypeConverter::AttributeConversionResult::hasResult() const {
  return impl.getInt() == resultTag;
}

bool TypeConverter::AttributeConversionResult::isNa() const {
  return impl.getInt() == naTag;
}

bool TypeConverter::AttributeConversionResult::isAbort() const {
  return impl.getInt() == abortTag;
}

Attribute TypeConverter::AttributeConversionResult::getResult() const {
  assert(hasResult() && "Cannot get result from N/A or abort");
  return impl.getPointer();
}

std::optional<Attribute>
TypeConverter::convertTypeAttribute(Type type, Attribute attr) const {
  for (const TypeAttributeConversionCallbackFn &fn :
       llvm::reverse(typeAttributeConversions)) {
    AttributeConversionResult res = fn(type, attr);
    if (res.hasResult())
      return res.getResult();
    if (res.isAbort())
      return std::nullopt;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// FunctionOpInterfaceSignatureConversion
//===----------------------------------------------------------------------===//

static LogicalResult convertFuncOpTypes(FunctionOpInterface funcOp,
                                        const TypeConverter &typeConverter,
                                        ConversionPatternRewriter &rewriter) {
  FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (!type)
    return failure();

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

  rewriter.modifyOpInPlace(funcOp, [&] { funcOp.setType(newType); });

  return success();
}

/// Create a default conversion pattern that rewrites the type signature of a
/// FunctionOpInterface op. This only supports ops which use FunctionType to
/// represent their type.
namespace {
struct FunctionOpInterfaceSignatureConversion : public ConversionPattern {
  FunctionOpInterfaceSignatureConversion(StringRef functionLikeOpName,
                                         MLIRContext *ctx,
                                         const TypeConverter &converter)
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

FailureOr<Operation *>
mlir::convertOpResultTypes(Operation *op, ValueRange operands,
                           const TypeConverter &converter,
                           ConversionPatternRewriter &rewriter) {
  assert(op && "Invalid op");
  Location loc = op->getLoc();
  if (converter.isLegal(op))
    return rewriter.notifyMatchFailure(loc, "op already legal");

  OperationState newOp(loc, op->getName());
  newOp.addOperands(operands);

  SmallVector<Type> newResultTypes;
  if (failed(converter.convertTypes(op->getResultTypes(), newResultTypes)))
    return rewriter.notifyMatchFailure(loc, "couldn't convert return types");

  newOp.addTypes(newResultTypes);
  newOp.addAttributes(op->getAttrs());
  return rewriter.create(newOp);
}

void mlir::populateFunctionOpInterfaceTypeConversionPattern(
    StringRef functionLikeOpName, RewritePatternSet &patterns,
    const TypeConverter &converter) {
  patterns.add<FunctionOpInterfaceSignatureConversion>(
      functionLikeOpName, patterns.getContext(), converter);
}

void mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(
    RewritePatternSet &patterns, const TypeConverter &converter) {
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
  auto *infoIt = legalOperations.find(name);
  assert(infoIt != legalOperations.end() &&
         infoIt->second.action == LegalizationAction::Dynamic &&
         "expected operation to already be marked as dynamically legal");
  infoIt->second.legalityFn =
      composeLegalityCallbacks(std::move(infoIt->second.legalityFn), callback);
}

void ConversionTarget::markOpRecursivelyLegal(
    OperationName name, const DynamicLegalityCallbackFn &callback) {
  auto *infoIt = legalOperations.find(name);
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
  const auto *it = legalOperations.find(op);
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

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
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
        if (const TypeConverter *converter =
                rewriterImpl.currentTypeConverter) {
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
        const TypeConverter *converter = rewriterImpl.currentTypeConverter;
        if (!converter)
          return SmallVector<Type>(types);

        SmallVector<Type> remappedTypes;
        if (failed(converter->convertTypes(types, remappedTypes)))
          return failure();
        return std::move(remappedTypes);
      });
}
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH

//===----------------------------------------------------------------------===//
// Op Conversion Entry Points
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Partial Conversion

LogicalResult mlir::applyPartialConversion(
    ArrayRef<Operation *> ops, const ConversionTarget &target,
    const FrozenRewritePatternSet &patterns, ConversionConfig config) {
  OperationConverter opConverter(target, patterns, config,
                                 OpConversionMode::Partial);
  return opConverter.convertOperations(ops);
}
LogicalResult
mlir::applyPartialConversion(Operation *op, const ConversionTarget &target,
                             const FrozenRewritePatternSet &patterns,
                             ConversionConfig config) {
  return applyPartialConversion(llvm::ArrayRef(op), target, patterns, config);
}

//===----------------------------------------------------------------------===//
// Full Conversion

LogicalResult mlir::applyFullConversion(ArrayRef<Operation *> ops,
                                        const ConversionTarget &target,
                                        const FrozenRewritePatternSet &patterns,
                                        ConversionConfig config) {
  OperationConverter opConverter(target, patterns, config,
                                 OpConversionMode::Full);
  return opConverter.convertOperations(ops);
}
LogicalResult mlir::applyFullConversion(Operation *op,
                                        const ConversionTarget &target,
                                        const FrozenRewritePatternSet &patterns,
                                        ConversionConfig config) {
  return applyFullConversion(llvm::ArrayRef(op), target, patterns, config);
}

//===----------------------------------------------------------------------===//
// Analysis Conversion

LogicalResult mlir::applyAnalysisConversion(
    ArrayRef<Operation *> ops, ConversionTarget &target,
    const FrozenRewritePatternSet &patterns, ConversionConfig config) {
  OperationConverter opConverter(target, patterns, config,
                                 OpConversionMode::Analysis);
  return opConverter.convertOperations(ops);
}
LogicalResult
mlir::applyAnalysisConversion(Operation *op, ConversionTarget &target,
                              const FrozenRewritePatternSet &patterns,
                              ConversionConfig config) {
  return applyAnalysisConversion(llvm::ArrayRef(op), target, patterns, config);
}
