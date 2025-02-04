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
#include "mlir/IR/Dominance.h"
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

/// Helper function that computes an insertion point where the given value is
/// defined and can be used without a dominance violation.
static OpBuilder::InsertPoint computeInsertPoint(Value value) {
  Block *insertBlock = value.getParentBlock();
  Block::iterator insertPt = insertBlock->begin();
  if (OpResult inputRes = dyn_cast<OpResult>(value))
    insertPt = ++inputRes.getOwner()->getIterator();
  return OpBuilder::InsertPoint(insertBlock, insertPt);
}

/// Helper function that computes an insertion point where the given values are
/// defined and can be used without a dominance violation.
static OpBuilder::InsertPoint computeInsertPoint(ArrayRef<Value> vals) {
  assert(!vals.empty() && "expected at least one value");
  DominanceInfo domInfo;
  OpBuilder::InsertPoint pt = computeInsertPoint(vals.front());
  for (Value v : vals.drop_front()) {
    // Choose the "later" insertion point.
    OpBuilder::InsertPoint nextPt = computeInsertPoint(v);
    if (domInfo.dominates(pt.getBlock(), pt.getPoint(), nextPt.getBlock(),
                          nextPt.getPoint())) {
      // pt is before nextPt => choose nextPt.
      pt = nextPt;
    } else {
#ifndef NDEBUG
      // nextPt should be before pt => choose pt.
      // If pt, nextPt are no dominance relationship, then there is no valid
      // insertion point at which all given values are defined.
      bool dom = domInfo.dominates(nextPt.getBlock(), nextPt.getPoint(),
                                   pt.getBlock(), pt.getPoint());
      assert(dom && "unable to find valid insertion point");
#endif // NDEBUG
    }
  }
  return pt;
}

//===----------------------------------------------------------------------===//
// ConversionValueMapping
//===----------------------------------------------------------------------===//

/// A vector of SSA values, optimized for the most common case of a single
/// value.
using ValueVector = SmallVector<Value, 1>;

namespace {

/// Helper class to make it possible to use `ValueVector` as a key in DenseMap.
struct ValueVectorMapInfo {
  static ValueVector getEmptyKey() { return ValueVector{Value()}; }
  static ValueVector getTombstoneKey() { return ValueVector{Value(), Value()}; }
  static ::llvm::hash_code getHashValue(const ValueVector &val) {
    return ::llvm::hash_combine_range(val.begin(), val.end());
  }
  static bool isEqual(const ValueVector &LHS, const ValueVector &RHS) {
    return LHS == RHS;
  }
};

/// This class wraps a IRMapping to provide recursive lookup
/// functionality, i.e. we will traverse if the mapped value also has a mapping.
struct ConversionValueMapping {
  /// Return "true" if an SSA value is mapped to the given value. May return
  /// false positives.
  bool isMappedTo(Value value) const { return mappedTo.contains(value); }

  /// Lookup the most recently mapped values with the desired types in the
  /// mapping.
  ///
  /// Special cases:
  /// - If the desired type range is empty, simply return the most recently
  ///   mapped values.
  /// - If there is no mapping to the desired types, also return the most
  ///   recently mapped values.
  /// - If there is no mapping for the given values at all, return the given
  ///   value.
  ValueVector lookupOrDefault(Value from, TypeRange desiredTypes = {}) const;

  /// Lookup the given value within the map, or return an empty vector if the
  /// value is not mapped. If it is mapped, this follows the same behavior
  /// as `lookupOrDefault`.
  ValueVector lookupOrNull(Value from, TypeRange desiredTypes = {}) const;

  template <typename T>
  struct IsValueVector : std::is_same<std::decay_t<T>, ValueVector> {};

  /// Map a value vector to the one provided.
  template <typename OldVal, typename NewVal>
  std::enable_if_t<IsValueVector<OldVal>::value && IsValueVector<NewVal>::value>
  map(OldVal &&oldVal, NewVal &&newVal) {
    LLVM_DEBUG({
      ValueVector next(newVal);
      while (true) {
        assert(next != oldVal && "inserting cyclic mapping");
        auto it = mapping.find(next);
        if (it == mapping.end())
          break;
        next = it->second;
      }
    });
    for (Value v : newVal)
      mappedTo.insert(v);

    mapping[std::forward<OldVal>(oldVal)] = std::forward<NewVal>(newVal);
  }

  /// Map a value vector or single value to the one provided.
  template <typename OldVal, typename NewVal>
  std::enable_if_t<!IsValueVector<OldVal>::value ||
                   !IsValueVector<NewVal>::value>
  map(OldVal &&oldVal, NewVal &&newVal) {
    if constexpr (IsValueVector<OldVal>{}) {
      map(std::forward<OldVal>(oldVal), ValueVector{newVal});
    } else if constexpr (IsValueVector<NewVal>{}) {
      map(ValueVector{oldVal}, std::forward<NewVal>(newVal));
    } else {
      map(ValueVector{oldVal}, ValueVector{newVal});
    }
  }

  /// Drop the last mapping for the given values.
  void erase(const ValueVector &value) { mapping.erase(value); }

private:
  /// Current value mappings.
  DenseMap<ValueVector, ValueVector, ValueVectorMapInfo> mapping;

  /// All SSA values that are mapped to. May contain false positives.
  DenseSet<Value> mappedTo;
};
} // namespace

ValueVector
ConversionValueMapping::lookupOrDefault(Value from,
                                        TypeRange desiredTypes) const {
  // Try to find the deepest values that have the desired types. If there is no
  // such mapping, simply return the deepest values.
  ValueVector desiredValue;
  ValueVector current{from};
  do {
    // Store the current value if the types match.
    if (TypeRange(ValueRange(current)) == desiredTypes)
      desiredValue = current;

    // If possible, Replace each value with (one or multiple) mapped values.
    ValueVector next;
    for (Value v : current) {
      auto it = mapping.find({v});
      if (it != mapping.end()) {
        llvm::append_range(next, it->second);
      } else {
        next.push_back(v);
      }
    }
    if (next != current) {
      // If at least one value was replaced, continue the lookup from there.
      current = std::move(next);
      continue;
    }

    // Otherwise: Check if there is a mapping for the entire vector. Such
    // mappings are materializations. (N:M mapping are not supported for value
    // replacements.)
    //
    // Note: From a correctness point of view, materializations do not have to
    // be stored (and looked up) in the mapping. But for performance reasons,
    // we choose to reuse existing IR (when possible) instead of creating it
    // multiple times.
    auto it = mapping.find(current);
    if (it == mapping.end()) {
      // No mapping found: The lookup stops here.
      break;
    }
    current = it->second;
  } while (true);

  // If the desired values were found use them, otherwise default to the leaf
  // values.
  // Note: If `desiredTypes` is empty, this function always returns `current`.
  return !desiredValue.empty() ? std::move(desiredValue) : std::move(current);
}

ValueVector ConversionValueMapping::lookupOrNull(Value from,
                                                 TypeRange desiredTypes) const {
  ValueVector result = lookupOrDefault(from, desiredTypes);
  if (result == ValueVector{from} ||
      (!desiredTypes.empty() && TypeRange(ValueRange(result)) != desiredTypes))
    return {};
  return result;
}

//===----------------------------------------------------------------------===//
// Rewriter and Translation State
//===----------------------------------------------------------------------===//
namespace {
/// This class contains a snapshot of the current conversion rewriter state.
/// This is useful when saving and undoing a set of rewrites.
struct RewriterState {
  RewriterState(unsigned numRewrites, unsigned numIgnoredOperations,
                unsigned numReplacedOps)
      : numRewrites(numRewrites), numIgnoredOperations(numIgnoredOperations),
        numReplacedOps(numReplacedOps) {}

  /// The current number of rewrites performed.
  unsigned numRewrites;

  /// The current number of ignored operations.
  unsigned numIgnoredOperations;

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

  /// Commit the rewrite. At this point, it is certain that the dialect
  /// conversion will succeed. All IR modifications, except for operation/block
  /// erasure, must be performed through the given rewriter.
  ///
  /// Instead of erasing operations/blocks, they should merely be unlinked
  /// commit phase and finally be erased during the cleanup phase. This is
  /// because internal dialect conversion state (such as `mapping`) may still
  /// be using them.
  ///
  /// Any IR modification that was already performed before the commit phase
  /// (e.g., insertion of an op) must be communicated to the listener that may
  /// be attached to the given rewriter.
  virtual void commit(RewriterBase &rewriter) {}

  /// Cleanup operations/blocks. Cleanup is called after commit.
  virtual void cleanup(RewriterBase &rewriter) {}

  Kind getKind() const { return kind; }

  static bool classof(const IRRewrite *rewrite) { return true; }

protected:
  IRRewrite(Kind kind, ConversionPatternRewriterImpl &rewriterImpl)
      : kind(kind), rewriterImpl(rewriterImpl) {}

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

  void commit(RewriterBase &rewriter) override {
    // The block was already created and inserted. Just inform the listener.
    if (auto *listener = rewriter.getListener())
      listener->notifyBlockInserted(block, /*previous=*/{}, /*previousIt=*/{});
  }

  void rollback() override {
    // Unlink all of the operations within this block, they will be deleted
    // separately.
    auto &blockOps = block->getOperations();
    while (!blockOps.empty())
      blockOps.remove(blockOps.begin());
    block->dropAllUses();
    if (block->getParent())
      block->erase();
    else
      delete block;
  }
};

/// Erasure of a block. Block erasures are partially reflected in the IR. Erased
/// blocks are immediately unlinked, but only erased during cleanup. This makes
/// it easier to rollback a block erasure: the block is simply inserted into its
/// original location.
class EraseBlockRewrite : public BlockRewrite {
public:
  EraseBlockRewrite(ConversionPatternRewriterImpl &rewriterImpl, Block *block)
      : BlockRewrite(Kind::EraseBlock, rewriterImpl, block),
        region(block->getParent()), insertBeforeBlock(block->getNextNode()) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::EraseBlock;
  }

  ~EraseBlockRewrite() override {
    assert(!block &&
           "rewrite was neither rolled back nor committed/cleaned up");
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

  void commit(RewriterBase &rewriter) override {
    // Erase the block.
    assert(block && "expected block");
    assert(block->empty() && "expected empty block");

    // Notify the listener that the block is about to be erased.
    if (auto *listener =
            dyn_cast_or_null<RewriterBase::Listener>(rewriter.getListener()))
      listener->notifyBlockErased(block);
  }

  void cleanup(RewriterBase &rewriter) override {
    // Erase the block.
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
    // If a listener is attached to the dialect conversion, ops must be moved
    // one-by-one. When they are moved in bulk, notifications cannot be sent
    // because the ops that used to be in the source block at the time of the
    // inlining (before the "commit" phase) are unknown at the time when
    // notifications are sent (which is during the "commit" phase).
    assert(!getConfig().listener &&
           "InlineBlockRewrite not supported if listener is attached");
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

  void commit(RewriterBase &rewriter) override {
    // The block was already moved. Just inform the listener.
    if (auto *listener = rewriter.getListener()) {
      // Note: `previousIt` cannot be passed because this is a delayed
      // notification and iterators into past IR state cannot be represented.
      listener->notifyBlockInserted(block, /*previous=*/region,
                                    /*previousIt=*/{});
    }
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

/// Block type conversion. This rewrite is partially reflected in the IR.
class BlockTypeConversionRewrite : public BlockRewrite {
public:
  BlockTypeConversionRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                             Block *origBlock, Block *newBlock)
      : BlockRewrite(Kind::BlockTypeConversion, rewriterImpl, origBlock),
        newBlock(newBlock) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::BlockTypeConversion;
  }

  Block *getOrigBlock() const { return block; }

  Block *getNewBlock() const { return newBlock; }

  void commit(RewriterBase &rewriter) override;

  void rollback() override;

private:
  /// The new block that was created as part of this signature conversion.
  Block *newBlock;
};

/// Replacing a block argument. This rewrite is not immediately reflected in the
/// IR. An internal IR mapping is updated, but the actual replacement is delayed
/// until the rewrite is committed.
class ReplaceBlockArgRewrite : public BlockRewrite {
public:
  ReplaceBlockArgRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                         Block *block, BlockArgument arg,
                         const TypeConverter *converter)
      : BlockRewrite(Kind::ReplaceBlockArg, rewriterImpl, block), arg(arg),
        converter(converter) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::ReplaceBlockArg;
  }

  void commit(RewriterBase &rewriter) override;

  void rollback() override;

private:
  BlockArgument arg;

  /// The current type converter when the block argument was replaced.
  const TypeConverter *converter;
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

  void commit(RewriterBase &rewriter) override {
    // The operation was already moved. Just inform the listener.
    if (auto *listener = rewriter.getListener()) {
      // Note: `previousIt` cannot be passed because this is a delayed
      // notification and iterators into past IR state cannot be represented.
      listener->notifyOperationInserted(
          op, /*previous=*/OpBuilder::InsertPoint(/*insertBlock=*/block,
                                                  /*insertPt=*/{}));
    }
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

  void commit(RewriterBase &rewriter) override {
    // Notify the listener that the operation was modified in-place.
    if (auto *listener =
            dyn_cast_or_null<RewriterBase::Listener>(rewriter.getListener()))
      listener->notifyOperationModified(op);

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
                          Operation *op, const TypeConverter *converter)
      : OperationRewrite(Kind::ReplaceOperation, rewriterImpl, op),
        converter(converter) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::ReplaceOperation;
  }

  void commit(RewriterBase &rewriter) override;

  void rollback() override;

  void cleanup(RewriterBase &rewriter) override;

private:
  /// An optional type converter that can be used to materialize conversions
  /// between the new and old values if necessary.
  const TypeConverter *converter;
};

class CreateOperationRewrite : public OperationRewrite {
public:
  CreateOperationRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                         Operation *op)
      : OperationRewrite(Kind::CreateOperation, rewriterImpl, op) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::CreateOperation;
  }

  void commit(RewriterBase &rewriter) override {
    // The operation was already created and inserted. Just inform the listener.
    if (auto *listener = rewriter.getListener())
      listener->notifyOperationInserted(op, /*previous=*/{});
  }

  void rollback() override;
};

/// The type of materialization.
enum MaterializationKind {
  /// This materialization materializes a conversion from an illegal type to a
  /// legal one.
  Target,

  /// This materialization materializes a conversion from a legal type back to
  /// an illegal one.
  Source
};

/// An unresolved materialization, i.e., a "builtin.unrealized_conversion_cast"
/// op. Unresolved materializations are erased at the end of the dialect
/// conversion.
class UnresolvedMaterializationRewrite : public OperationRewrite {
public:
  UnresolvedMaterializationRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                                   UnrealizedConversionCastOp op,
                                   const TypeConverter *converter,
                                   MaterializationKind kind, Type originalType,
                                   ValueVector mappedValues);

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::UnresolvedMaterialization;
  }

  void rollback() override;

  UnrealizedConversionCastOp getOperation() const {
    return cast<UnrealizedConversionCastOp>(op);
  }

  /// Return the type converter of this materialization (which may be null).
  const TypeConverter *getConverter() const {
    return converterAndKind.getPointer();
  }

  /// Return the kind of this materialization.
  MaterializationKind getMaterializationKind() const {
    return converterAndKind.getInt();
  }

  /// Return the original type of the SSA value.
  Type getOriginalType() const { return originalType; }

private:
  /// The corresponding type converter to use when resolving this
  /// materialization, and the kind of this materialization.
  llvm::PointerIntPair<const TypeConverter *, 2, MaterializationKind>
      converterAndKind;

  /// The original type of the SSA value. Only used for target
  /// materializations.
  Type originalType;

  /// The values in the conversion value mapping that are being replaced by the
  /// results of this unresolved materialization.
  ValueVector mappedValues;
};
} // namespace

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
/// Return "true" if there is an operation rewrite that matches the specified
/// rewrite type and operation among the given rewrites.
template <typename RewriteTy, typename R>
static bool hasRewrite(R &&rewrites, Operation *op) {
  return any_of(std::forward<R>(rewrites), [&](auto &rewrite) {
    auto *rewriteTy = dyn_cast<RewriteTy>(rewrite.get());
    return rewriteTy && rewriteTy->getOperation() == op;
  });
}

/// Return "true" if there is a block rewrite that matches the specified
/// rewrite type and block among the given rewrites.
template <typename RewriteTy, typename R>
static bool hasRewrite(R &&rewrites, Block *block) {
  return any_of(std::forward<R>(rewrites), [&](auto &rewrite) {
    auto *rewriteTy = dyn_cast<RewriteTy>(rewrite.get());
    return rewriteTy && rewriteTy->getBlock() == block;
  });
}
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

//===----------------------------------------------------------------------===//
// ConversionPatternRewriterImpl
//===----------------------------------------------------------------------===//
namespace mlir {
namespace detail {
struct ConversionPatternRewriterImpl : public RewriterBase::Listener {
  explicit ConversionPatternRewriterImpl(MLIRContext *ctx,
                                         const ConversionConfig &config)
      : context(ctx), eraseRewriter(ctx), config(config) {}

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
                            SmallVector<ValueVector> &remapped);

  /// Return "true" if the given operation is ignored, and does not need to be
  /// converted.
  bool isOpIgnored(Operation *op) const;

  /// Return "true" if the given operation was replaced or erased.
  bool wasOpReplaced(Operation *op) const;

  //===--------------------------------------------------------------------===//
  // Type Conversion
  //===--------------------------------------------------------------------===//

  /// Convert the types of block arguments within the given region.
  FailureOr<Block *>
  convertRegionTypes(ConversionPatternRewriter &rewriter, Region *region,
                     const TypeConverter &converter,
                     TypeConverter::SignatureConversion *entryConversion);

  /// Apply the given signature conversion on the given block. The new block
  /// containing the updated signature is returned. If no conversions were
  /// necessary, e.g. if the block has no arguments, `block` is returned.
  /// `converter` is used to generate any necessary cast operations that
  /// translate between the origin argument types and those specified in the
  /// signature conversion.
  Block *applySignatureConversion(
      ConversionPatternRewriter &rewriter, Block *block,
      const TypeConverter *converter,
      TypeConverter::SignatureConversion &signatureConversion);

  //===--------------------------------------------------------------------===//
  // Materializations
  //===--------------------------------------------------------------------===//

  /// Build an unresolved materialization operation given a range of output
  /// types and a list of input operands. Returns the inputs if they their
  /// types match the output types.
  ///
  /// If a cast op was built, it can optionally be returned with the `castOp`
  /// output argument.
  ///
  /// If `valuesToMap` is set to a non-null Value, then that value is mapped to
  /// the results of the unresolved materialization in the conversion value
  /// mapping.
  ValueRange buildUnresolvedMaterialization(
      MaterializationKind kind, OpBuilder::InsertPoint ip, Location loc,
      ValueVector valuesToMap, ValueRange inputs, TypeRange outputTypes,
      Type originalType, const TypeConverter *converter,
      UnrealizedConversionCastOp *castOp = nullptr);

  /// Find a replacement value for the given SSA value in the conversion value
  /// mapping. The replacement value must have the same type as the given SSA
  /// value. If there is no replacement value with the correct type, find the
  /// latest replacement value (regardless of the type) and build a source
  /// materialization.
  Value findOrBuildReplacementValue(Value value,
                                    const TypeConverter *converter);

  //===--------------------------------------------------------------------===//
  // Rewriter Notification Hooks
  //===--------------------------------------------------------------------===//

  //// Notifies that an op was inserted.
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;

  /// Notifies that an op is about to be replaced with the given values.
  void notifyOpReplaced(Operation *op, ArrayRef<ValueRange> newValues);

  /// Notifies that a block is about to be erased.
  void notifyBlockIsBeingErased(Block *block);

  /// Notifies that a block was inserted.
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override;

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
      if (wasErased(op))
        return;
      op->dropAllUses();
      RewriterBase::eraseOp(op);
    }

    /// Erase the given block (unless it was already erased).
    void eraseBlock(Block *block) override {
      if (wasErased(block))
        return;
      assert(block->empty() && "expected empty block");
      block->dropAllDefinedValueUses();
      RewriterBase::eraseBlock(block);
    }

    bool wasErased(void *ptr) const { return erased.contains(ptr); }

    void notifyOperationErased(Operation *op) override { erased.insert(op); }

    void notifyBlockErased(Block *block) override { erased.insert(block); }

  private:
    /// Pointers to all erased operations and blocks.
    DenseSet<void *> erased;
  };

  //===--------------------------------------------------------------------===//
  // State
  //===--------------------------------------------------------------------===//

  /// MLIR context.
  MLIRContext *context;

  /// A rewriter that keeps track of ops/block that were already erased and
  /// skips duplicate op/block erasures. This rewriter is used during the
  /// "cleanup" phase.
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

  /// A mapping of all unresolved materializations (UnrealizedConversionCastOp)
  /// to the corresponding rewrite objects.
  DenseMap<UnrealizedConversionCastOp, UnresolvedMaterializationRewrite *>
      unresolvedMaterializations;

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

const ConversionConfig &IRRewrite::getConfig() const {
  return rewriterImpl.config;
}

void BlockTypeConversionRewrite::commit(RewriterBase &rewriter) {
  // Inform the listener about all IR modifications that have already taken
  // place: References to the original block have been replaced with the new
  // block.
  if (auto *listener =
          dyn_cast_or_null<RewriterBase::Listener>(rewriter.getListener()))
    for (Operation *op : getNewBlock()->getUsers())
      listener->notifyOperationModified(op);
}

void BlockTypeConversionRewrite::rollback() {
  getNewBlock()->replaceAllUsesWith(getOrigBlock());
}

void ReplaceBlockArgRewrite::commit(RewriterBase &rewriter) {
  Value repl = rewriterImpl.findOrBuildReplacementValue(arg, converter);
  if (!repl)
    return;

  if (isa<BlockArgument>(repl)) {
    rewriter.replaceAllUsesWith(arg, repl);
    return;
  }

  // If the replacement value is an operation, we check to make sure that we
  // don't replace uses that are within the parent operation of the
  // replacement value.
  Operation *replOp = cast<OpResult>(repl).getOwner();
  Block *replBlock = replOp->getBlock();
  rewriter.replaceUsesWithIf(arg, repl, [&](OpOperand &operand) {
    Operation *user = operand.getOwner();
    return user->getBlock() != replBlock || replOp->isBeforeInBlock(user);
  });
}

void ReplaceBlockArgRewrite::rollback() { rewriterImpl.mapping.erase({arg}); }

void ReplaceOperationRewrite::commit(RewriterBase &rewriter) {
  auto *listener =
      dyn_cast_or_null<RewriterBase::Listener>(rewriter.getListener());

  // Compute replacement values.
  SmallVector<Value> replacements =
      llvm::map_to_vector(op->getResults(), [&](OpResult result) {
        return rewriterImpl.findOrBuildReplacementValue(result, converter);
      });

  // Notify the listener that the operation is about to be replaced.
  if (listener)
    listener->notifyOperationReplaced(op, replacements);

  // Replace all uses with the new values.
  for (auto [result, newValue] :
       llvm::zip_equal(op->getResults(), replacements))
    if (newValue)
      rewriter.replaceAllUsesWith(result, newValue);

  // The original op will be erased, so remove it from the set of unlegalized
  // ops.
  if (getConfig().unlegalizedOps)
    getConfig().unlegalizedOps->erase(op);

  // Notify the listener that the operation (and its nested operations) was
  // erased.
  if (listener) {
    op->walk<WalkOrder::PostOrder>(
        [&](Operation *op) { listener->notifyOperationErased(op); });
  }

  // Do not erase the operation yet. It may still be referenced in `mapping`.
  // Just unlink it for now and erase it during cleanup.
  op->getBlock()->getOperations().remove(op);
}

void ReplaceOperationRewrite::rollback() {
  for (auto result : op->getResults())
    rewriterImpl.mapping.erase({result});
}

void ReplaceOperationRewrite::cleanup(RewriterBase &rewriter) {
  rewriter.eraseOp(op);
}

void CreateOperationRewrite::rollback() {
  for (Region &region : op->getRegions()) {
    while (!region.getBlocks().empty())
      region.getBlocks().remove(region.getBlocks().begin());
  }
  op->dropAllUses();
  op->erase();
}

UnresolvedMaterializationRewrite::UnresolvedMaterializationRewrite(
    ConversionPatternRewriterImpl &rewriterImpl, UnrealizedConversionCastOp op,
    const TypeConverter *converter, MaterializationKind kind, Type originalType,
    ValueVector mappedValues)
    : OperationRewrite(Kind::UnresolvedMaterialization, rewriterImpl, op),
      converterAndKind(converter, kind), originalType(originalType),
      mappedValues(std::move(mappedValues)) {
  assert((!originalType || kind == MaterializationKind::Target) &&
         "original type is valid only for target materializations");
  rewriterImpl.unresolvedMaterializations[op] = this;
}

void UnresolvedMaterializationRewrite::rollback() {
  if (!mappedValues.empty())
    rewriterImpl.mapping.erase(mappedValues);
  rewriterImpl.unresolvedMaterializations.erase(getOperation());
  op->erase();
}

void ConversionPatternRewriterImpl::applyRewrites() {
  // Commit all rewrites.
  IRRewriter rewriter(context, config.listener);
  // Note: New rewrites may be added during the "commit" phase and the
  // `rewrites` vector may reallocate.
  for (size_t i = 0; i < rewrites.size(); ++i)
    rewrites[i]->commit(rewriter);

  // Clean up all rewrites.
  for (auto &rewrite : rewrites)
    rewrite->cleanup(eraseRewriter);
}

//===----------------------------------------------------------------------===//
// State Management

RewriterState ConversionPatternRewriterImpl::getCurrentState() {
  return RewriterState(rewrites.size(), ignoredOps.size(), replacedOps.size());
}

void ConversionPatternRewriterImpl::resetState(RewriterState state) {
  // Undo any rewrites.
  undoRewrites(state.numRewrites);

  // Pop all of the recorded ignored operations that are no longer valid.
  while (ignoredOps.size() != state.numIgnoredOperations)
    ignoredOps.pop_back();

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
    SmallVector<ValueVector> &remapped) {
  remapped.reserve(llvm::size(values));

  for (const auto &it : llvm::enumerate(values)) {
    Value operand = it.value();
    Type origType = operand.getType();
    Location operandLoc = inputLoc ? *inputLoc : operand.getLoc();

    if (!currentTypeConverter) {
      // The current pattern does not have a type converter. I.e., it does not
      // distinguish between legal and illegal types. For each operand, simply
      // pass through the most recently mapped values.
      remapped.push_back(mapping.lookupOrDefault(operand));
      continue;
    }

    // If there is no legal conversion, fail to match this pattern.
    SmallVector<Type, 1> legalTypes;
    if (failed(currentTypeConverter->convertType(origType, legalTypes))) {
      notifyMatchFailure(operandLoc, [=](Diagnostic &diag) {
        diag << "unable to convert type for " << valueDiagTag << " #"
             << it.index() << ", type was " << origType;
      });
      return failure();
    }
    // If a type is converted to 0 types, there is nothing to do.
    if (legalTypes.empty()) {
      remapped.push_back({});
      continue;
    }

    ValueVector repl = mapping.lookupOrDefault(operand, legalTypes);
    if (!repl.empty() && TypeRange(ValueRange(repl)) == legalTypes) {
      // Mapped values have the correct type or there is an existing
      // materialization. Or the operand is not mapped at all and has the
      // correct type.
      remapped.push_back(std::move(repl));
      continue;
    }

    // Create a materialization for the most recently mapped values.
    repl = mapping.lookupOrDefault(operand);
    ValueRange castValues = buildUnresolvedMaterialization(
        MaterializationKind::Target, computeInsertPoint(repl), operandLoc,
        /*valuesToMap=*/repl, /*inputs=*/repl, /*outputTypes=*/legalTypes,
        /*originalType=*/origType, currentTypeConverter);
    remapped.push_back(castValues);
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

FailureOr<Block *> ConversionPatternRewriterImpl::convertRegionTypes(
    ConversionPatternRewriter &rewriter, Region *region,
    const TypeConverter &converter,
    TypeConverter::SignatureConversion *entryConversion) {
  regionToConverter[region] = &converter;
  if (region->empty())
    return nullptr;

  // Convert the arguments of each non-entry block within the region.
  for (Block &block :
       llvm::make_early_inc_range(llvm::drop_begin(*region, 1))) {
    // Compute the signature for the block with the provided converter.
    std::optional<TypeConverter::SignatureConversion> conversion =
        converter.convertBlockSignature(&block);
    if (!conversion)
      return failure();
    // Convert the block with the computed signature.
    applySignatureConversion(rewriter, &block, &converter, *conversion);
  }

  // Convert the entry block. If an entry signature conversion was provided,
  // use that one. Otherwise, compute the signature with the type converter.
  if (entryConversion)
    return applySignatureConversion(rewriter, &region->front(), &converter,
                                    *entryConversion);
  std::optional<TypeConverter::SignatureConversion> conversion =
      converter.convertBlockSignature(&region->front());
  if (!conversion)
    return failure();
  return applySignatureConversion(rewriter, &region->front(), &converter,
                                  *conversion);
}

Block *ConversionPatternRewriterImpl::applySignatureConversion(
    ConversionPatternRewriter &rewriter, Block *block,
    const TypeConverter *converter,
    TypeConverter::SignatureConversion &signatureConversion) {
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  // A block cannot be converted multiple times.
  if (hasRewrite<BlockTypeConversionRewrite>(rewrites, block))
    llvm::report_fatal_error("block was already converted");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

  OpBuilder::InsertionGuard g(rewriter);

  // If no arguments are being changed or added, there is nothing to do.
  unsigned origArgCount = block->getNumArguments();
  auto convertedTypes = signatureConversion.getConvertedTypes();
  if (llvm::equal(block->getArgumentTypes(), convertedTypes))
    return block;

  // Compute the locations of all block arguments in the new block.
  SmallVector<Location> newLocs(convertedTypes.size(),
                                rewriter.getUnknownLoc());
  for (unsigned i = 0; i < origArgCount; ++i) {
    auto inputMap = signatureConversion.getInputMapping(i);
    if (!inputMap || inputMap->replacementValue)
      continue;
    Location origLoc = block->getArgument(i).getLoc();
    for (unsigned j = 0; j < inputMap->size; ++j)
      newLocs[inputMap->inputNo + j] = origLoc;
  }

  // Insert a new block with the converted block argument types and move all ops
  // from the old block to the new block.
  Block *newBlock =
      rewriter.createBlock(block->getParent(), std::next(block->getIterator()),
                           convertedTypes, newLocs);

  // If a listener is attached to the dialect conversion, ops cannot be moved
  // to the destination block in bulk ("fast path"). This is because at the time
  // the notifications are sent, it is unknown which ops were moved. Instead,
  // ops should be moved one-by-one ("slow path"), so that a separate
  // `MoveOperationRewrite` is enqueued for each moved op. Moving ops in bulk is
  // a bit more efficient, so we try to do that when possible.
  bool fastPath = !config.listener;
  if (fastPath) {
    appendRewrite<InlineBlockRewrite>(newBlock, block, newBlock->end());
    newBlock->getOperations().splice(newBlock->end(), block->getOperations());
  } else {
    while (!block->empty())
      rewriter.moveOpBefore(&block->front(), newBlock, newBlock->end());
  }

  // Replace all uses of the old block with the new block.
  block->replaceAllUsesWith(newBlock);

  for (unsigned i = 0; i != origArgCount; ++i) {
    BlockArgument origArg = block->getArgument(i);
    Type origArgType = origArg.getType();

    std::optional<TypeConverter::SignatureConversion::InputMapping> inputMap =
        signatureConversion.getInputMapping(i);
    if (!inputMap) {
      // This block argument was dropped and no replacement value was provided.
      // Materialize a replacement value "out of thin air".
      buildUnresolvedMaterialization(
          MaterializationKind::Source,
          OpBuilder::InsertPoint(newBlock, newBlock->begin()), origArg.getLoc(),
          /*valuesToMap=*/{origArg}, /*inputs=*/ValueRange(),
          /*outputType=*/origArgType, /*originalType=*/Type(), converter);
      appendRewrite<ReplaceBlockArgRewrite>(block, origArg, converter);
      continue;
    }

    if (Value repl = inputMap->replacementValue) {
      // This block argument was dropped and a replacement value was provided.
      assert(inputMap->size == 0 &&
             "invalid to provide a replacement value when the argument isn't "
             "dropped");
      mapping.map(origArg, repl);
      appendRewrite<ReplaceBlockArgRewrite>(block, origArg, converter);
      continue;
    }

    // This is a 1->1+ mapping.
    auto replArgs =
        newBlock->getArguments().slice(inputMap->inputNo, inputMap->size);
    ValueVector replArgVals = llvm::to_vector_of<Value, 1>(replArgs);
    mapping.map(origArg, std::move(replArgVals));
    appendRewrite<ReplaceBlockArgRewrite>(block, origArg, converter);
  }

  appendRewrite<BlockTypeConversionRewrite>(/*origBlock=*/block, newBlock);

  // Erase the old block. (It is just unlinked for now and will be erased during
  // cleanup.)
  rewriter.eraseBlock(block);

  return newBlock;
}

//===----------------------------------------------------------------------===//
// Materializations
//===----------------------------------------------------------------------===//

/// Build an unresolved materialization operation given an output type and set
/// of input operands.
ValueRange ConversionPatternRewriterImpl::buildUnresolvedMaterialization(
    MaterializationKind kind, OpBuilder::InsertPoint ip, Location loc,
    ValueVector valuesToMap, ValueRange inputs, TypeRange outputTypes,
    Type originalType, const TypeConverter *converter,
    UnrealizedConversionCastOp *castOp) {
  assert((!originalType || kind == MaterializationKind::Target) &&
         "original type is valid only for target materializations");
  assert(TypeRange(inputs) != outputTypes &&
         "materialization is not necessary");

  // Create an unresolved materialization. We use a new OpBuilder to avoid
  // tracking the materialization like we do for other operations.
  OpBuilder builder(outputTypes.front().getContext());
  builder.setInsertionPoint(ip.getBlock(), ip.getPoint());
  auto convertOp =
      builder.create<UnrealizedConversionCastOp>(loc, outputTypes, inputs);
  if (!valuesToMap.empty())
    mapping.map(valuesToMap, convertOp.getResults());
  if (castOp)
    *castOp = convertOp;
  appendRewrite<UnresolvedMaterializationRewrite>(
      convertOp, converter, kind, originalType, std::move(valuesToMap));
  return convertOp.getResults();
}

Value ConversionPatternRewriterImpl::findOrBuildReplacementValue(
    Value value, const TypeConverter *converter) {
  // Try to find a replacement value with the same type in the conversion value
  // mapping. This includes cached materializations. We try to reuse those
  // instead of generating duplicate IR.
  ValueVector repl = mapping.lookupOrNull(value, value.getType());
  if (!repl.empty())
    return repl.front();

  // Check if the value is dead. No replacement value is needed in that case.
  // This is an approximate check that may have false negatives but does not
  // require computing and traversing an inverse mapping. (We may end up
  // building source materializations that are never used and that fold away.)
  if (llvm::all_of(value.getUsers(),
                   [&](Operation *op) { return replacedOps.contains(op); }) &&
      !mapping.isMappedTo(value))
    return Value();

  // No replacement value was found. Get the latest replacement value
  // (regardless of the type) and build a source materialization to the
  // original type.
  repl = mapping.lookupOrNull(value);
  if (repl.empty()) {
    // No replacement value is registered in the mapping. This means that the
    // value is dropped and no longer needed. (If the value were still needed,
    // a source materialization producing a replacement value "out of thin air"
    // would have already been created during `replaceOp` or
    // `applySignatureConversion`.)
    return Value();
  }

  // Note: `computeInsertPoint` computes the "earliest" insertion point at
  // which all values in `repl` are defined. It is important to emit the
  // materialization at that location because the same materialization may be
  // reused in a different context. (That's because materializations are cached
  // in the conversion value mapping.) The insertion point of the
  // materialization must be valid for all future users that may be created
  // later in the conversion process.
  Value castValue =
      buildUnresolvedMaterialization(MaterializationKind::Source,
                                     computeInsertPoint(repl), value.getLoc(),
                                     /*valuesToMap=*/repl, /*inputs=*/repl,
                                     /*outputType=*/value.getType(),
                                     /*originalType=*/Type(), converter)
          .front();
  return castValue;
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

void ConversionPatternRewriterImpl::notifyOpReplaced(
    Operation *op, ArrayRef<ValueRange> newValues) {
  assert(newValues.size() == op->getNumResults());
  assert(!ignoredOps.contains(op) && "operation was already replaced");

  // Check if replaced op is an unresolved materialization, i.e., an
  // unrealized_conversion_cast op that was created by the conversion driver.
  bool isUnresolvedMaterialization = false;
  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op))
    if (unresolvedMaterializations.contains(castOp))
      isUnresolvedMaterialization = true;

  // Create mappings for each of the new result values.
  for (auto [repl, result] : llvm::zip_equal(newValues, op->getResults())) {
    if (repl.empty()) {
      // This result was dropped and no replacement value was provided.
      if (isUnresolvedMaterialization) {
        // Do not create another materializations if we are erasing a
        // materialization.
        continue;
      }

      // Materialize a replacement value "out of thin air".
      buildUnresolvedMaterialization(
          MaterializationKind::Source, computeInsertPoint(result),
          result.getLoc(), /*valuesToMap=*/{result}, /*inputs=*/ValueRange(),
          /*outputType=*/result.getType(), /*originalType=*/Type(),
          currentTypeConverter);
      continue;
    } else {
      // Make sure that the user does not mess with unresolved materializations
      // that were inserted by the conversion driver. We keep track of these
      // ops in internal data structures. Erasing them must be allowed because
      // this can happen when the user is erasing an entire block (including
      // its body). But replacing them with another value should be forbidden
      // to avoid problems with the `mapping`.
      assert(!isUnresolvedMaterialization &&
             "attempting to replace an unresolved materialization");
    }

    // Remap result to replacement value.
    if (repl.empty())
      continue;
    mapping.map(result, repl);
  }

  appendRewrite<ReplaceOperationRewrite>(op, currentTypeConverter);
  // Mark this operation and all nested ops as replaced.
  op->walk([&](Operation *op) { replacedOps.insert(op); });
}

void ConversionPatternRewriterImpl::notifyBlockIsBeingErased(Block *block) {
  appendRewrite<EraseBlockRewrite>(block);
}

void ConversionPatternRewriterImpl::notifyBlockInserted(
    Block *block, Region *previous, Region::iterator previousIt) {
  assert(!wasOpReplaced(block->getParentOp()) &&
         "attempting to insert into a region within a replaced/erased op");
  LLVM_DEBUG(
      {
        Operation *parent = block->getParentOp();
        if (parent) {
          logger.startLine() << "** Insert Block into : '" << parent->getName()
                             << "'(" << parent << ")\n";
        } else {
          logger.startLine()
              << "** Insert Block into detached Region (nullptr parent op)'";
        }
      });

  if (!previous) {
    // This is a newly created block.
    appendRewrite<CreateBlockRewrite>(block);
    return;
  }
  Block *prevBlock = previousIt == previous->end() ? nullptr : &*previousIt;
  appendRewrite<MoveBlockRewrite>(block, previous, prevBlock);
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
  SmallVector<ValueRange> newVals;
  for (size_t i = 0; i < newValues.size(); ++i) {
    if (newValues[i]) {
      newVals.push_back(newValues.slice(i, 1));
    } else {
      newVals.push_back(ValueRange());
    }
  }
  impl->notifyOpReplaced(op, newVals);
}

void ConversionPatternRewriter::replaceOpWithMultiple(
    Operation *op, ArrayRef<ValueRange> newValues) {
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
  SmallVector<ValueRange> nullRepls(op->getNumResults(), {});
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
    Block *block, TypeConverter::SignatureConversion &conversion,
    const TypeConverter *converter) {
  assert(!impl->wasOpReplaced(block->getParentOp()) &&
         "attempting to apply a signature conversion to a block within a "
         "replaced/erased op");
  return impl->applySignatureConversion(*this, block, converter, conversion);
}

FailureOr<Block *> ConversionPatternRewriter::convertRegionTypes(
    Region *region, const TypeConverter &converter,
    TypeConverter::SignatureConversion *entryConversion) {
  assert(!impl->wasOpReplaced(region->getParentOp()) &&
         "attempting to apply a signature conversion to a block within a "
         "replaced/erased op");
  return impl->convertRegionTypes(*this, region, converter, entryConversion);
}

void ConversionPatternRewriter::replaceUsesOfBlockArgument(BlockArgument from,
                                                           Value to) {
  LLVM_DEBUG({
    impl->logger.startLine() << "** Replace Argument : '" << from << "'";
    if (Operation *parentOp = from.getOwner()->getParentOp()) {
      impl->logger.getOStream() << " (in region of '" << parentOp->getName()
                                << "' (" << parentOp << ")\n";
    } else {
      impl->logger.getOStream() << " (unlinked block)\n";
    }
  });
  impl->appendRewrite<ReplaceBlockArgRewrite>(from.getOwner(), from,
                                              impl->currentTypeConverter);
  impl->mapping.map(impl->mapping.lookupOrDefault(from), to);
}

Value ConversionPatternRewriter::getRemappedValue(Value key) {
  SmallVector<ValueVector> remappedValues;
  if (failed(impl->remapValues("value", /*inputLoc=*/std::nullopt, *this, key,
                               remappedValues)))
    return nullptr;
  assert(remappedValues.front().size() == 1 && "1:N conversion not supported");
  return remappedValues.front().front();
}

LogicalResult
ConversionPatternRewriter::getRemappedValues(ValueRange keys,
                                             SmallVectorImpl<Value> &results) {
  if (keys.empty())
    return success();
  SmallVector<ValueVector> remapped;
  if (failed(impl->remapValues("value", /*inputLoc=*/std::nullopt, *this, keys,
                               remapped)))
    return failure();
  for (const auto &values : remapped) {
    assert(values.size() == 1 && "1:N conversion not supported");
    results.push_back(values.front());
  }
  return success();
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

  // If a listener is attached to the dialect conversion, ops cannot be moved
  // to the destination block in bulk ("fast path"). This is because at the time
  // the notifications are sent, it is unknown which ops were moved. Instead,
  // ops should be moved one-by-one ("slow path"), so that a separate
  // `MoveOperationRewrite` is enqueued for each moved op. Moving ops in bulk is
  // a bit more efficient, so we try to do that when possible.
  bool fastPath = !impl->config.listener;

  if (fastPath)
    impl->notifyBlockBeingInlined(dest, source, before);

  // Replace all uses of block arguments.
  for (auto it : llvm::zip(source->getArguments(), argValues))
    replaceUsesOfBlockArgument(std::get<0>(it), std::get<1>(it));

  if (fastPath) {
    // Move all ops at once.
    dest->getOperations().splice(before, source->getOperations());
  } else {
    // Move op by op.
    while (!source->empty())
      moveOpBefore(&source->front(), dest, before);
  }

  // Erase the source block.
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

SmallVector<Value> ConversionPattern::getOneToOneAdaptorOperands(
    ArrayRef<ValueRange> operands) const {
  SmallVector<Value> oneToOneOperands;
  oneToOneOperands.reserve(operands.size());
  for (ValueRange operand : operands) {
    if (operand.size() != 1)
      llvm::report_fatal_error("pattern '" + getDebugName() +
                               "' does not support 1:N conversion");
    oneToOneOperands.push_back(operand.front());
  }
  return oneToOneOperands;
}

LogicalResult
ConversionPattern::matchAndRewrite(Operation *op,
                                   PatternRewriter &rewriter) const {
  auto &dialectRewriter = static_cast<ConversionPatternRewriter &>(rewriter);
  auto &rewriterImpl = dialectRewriter.getImpl();

  // Track the current conversion pattern type converter in the rewriter.
  llvm::SaveAndRestore currentConverterGuard(rewriterImpl.currentTypeConverter,
                                             getTypeConverter());

  // Remap the operands of the operation.
  SmallVector<ValueVector> remapped;
  if (failed(rewriterImpl.remapValues("operand", op->getLoc(), rewriter,
                                      op->getOperands(), remapped))) {
    return failure();
  }
  SmallVector<ValueRange> remappedAsRange =
      llvm::to_vector_of<ValueRange>(remapped);
  return matchAndRewrite(op, remappedAsRange, dialectRewriter);
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
                     const FrozenRewritePatternSet &patterns,
                     const ConversionConfig &config);

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

  /// Dialect conversion configuration.
  const ConversionConfig &config;
};
} // namespace

OperationLegalizer::OperationLegalizer(const ConversionTarget &targetInfo,
                                       const FrozenRewritePatternSet &patterns,
                                       const ConversionConfig &config)
    : target(targetInfo), applicator(patterns), config(config) {
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
  // An empty list of replacement values indicates that the fold was in-place.
  // As the operation changed, a new legalization needs to be attempted.
  if (replacementValues.empty())
    return legalize(op, rewriter);

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
    bool canApply = canApplyPattern(op, pattern, rewriter);
    if (canApply && config.listener)
      config.listener->notifyPatternBegin(pattern, op);
    return canApply;
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
    if (config.listener)
      config.listener->notifyPatternEnd(pattern, failure());
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
    if (config.listener)
      config.listener->notifyPatternEnd(pattern, result);
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
  assert(impl.pendingRootUpdates.empty() && "dangling root updates");

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  // Check that the root was either replaced or updated in place.
  auto newRewrites = llvm::drop_begin(impl.rewrites, curState.numRewrites);
  auto replacedRoot = [&] {
    return hasRewrite<ReplaceOperationRewrite>(newRewrites, op);
  };
  auto updatedRootInPlace = [&] {
    return hasRewrite<ModifyOperationRewrite>(newRewrites, op);
  };
  if (!replacedRoot() && !updatedRootInPlace())
    llvm::report_fatal_error("expected pattern to replace the root operation");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

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
      std::optional<TypeConverter::SignatureConversion> conversion =
          converter->convertBlockSignature(block);
      if (!conversion) {
        LLVM_DEBUG(logFailure(impl.logger, "failed to convert types of moved "
                                           "block"));
        return failure();
      }
      impl.applySignatureConversion(rewriter, block, converter, *conversion);
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
      : config(config), opLegalizer(target, patterns, this->config),
        mode(mode) {}

  /// Converts the given operations to the conversion target.
  LogicalResult convertOperations(ArrayRef<Operation *> ops);

private:
  /// Converts an operation with the given rewriter.
  LogicalResult convert(ConversionPatternRewriter &rewriter, Operation *op);

  /// Dialect conversion configuration.
  ConversionConfig config;

  /// The legalizer to use when converting operations.
  OperationLegalizer opLegalizer;

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

static LogicalResult
legalizeUnresolvedMaterialization(RewriterBase &rewriter,
                                  UnresolvedMaterializationRewrite *rewrite) {
  UnrealizedConversionCastOp op = rewrite->getOperation();
  assert(!op.use_empty() &&
         "expected that dead materializations have already been DCE'd");
  Operation::operand_range inputOperands = op.getOperands();

  // Try to materialize the conversion.
  if (const TypeConverter *converter = rewrite->getConverter()) {
    rewriter.setInsertionPoint(op);
    SmallVector<Value> newMaterialization;
    switch (rewrite->getMaterializationKind()) {
    case MaterializationKind::Target:
      newMaterialization = converter->materializeTargetConversion(
          rewriter, op->getLoc(), op.getResultTypes(), inputOperands,
          rewrite->getOriginalType());
      break;
    case MaterializationKind::Source:
      assert(op->getNumResults() == 1 && "expected single result");
      Value sourceMat = converter->materializeSourceConversion(
          rewriter, op->getLoc(), op.getResultTypes().front(), inputOperands);
      if (sourceMat)
        newMaterialization.push_back(sourceMat);
      break;
    }
    if (!newMaterialization.empty()) {
#ifndef NDEBUG
      ValueRange newMaterializationRange(newMaterialization);
      assert(TypeRange(newMaterializationRange) == op.getResultTypes() &&
             "materialization callback produced value of incorrect type");
#endif // NDEBUG
      rewriter.replaceOp(op, newMaterialization);
      return success();
    }
  }

  InFlightDiagnostic diag = op->emitError()
                            << "failed to legalize unresolved materialization "
                               "from ("
                            << inputOperands.getTypes() << ") to ("
                            << op.getResultTypes()
                            << ") that remained live after conversion";
  diag.attachNote(op->getUsers().begin()->getLoc())
      << "see existing live user here: " << *op->getUsers().begin();
  return failure();
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

  // After a successful conversion, apply rewrites.
  rewriterImpl.applyRewrites();

  // Gather all unresolved materializations.
  SmallVector<UnrealizedConversionCastOp> allCastOps;
  const DenseMap<UnrealizedConversionCastOp, UnresolvedMaterializationRewrite *>
      &materializations = rewriterImpl.unresolvedMaterializations;
  for (auto it : materializations) {
    if (rewriterImpl.eraseRewriter.wasErased(it.first))
      continue;
    allCastOps.push_back(it.first);
  }

  // Reconcile all UnrealizedConversionCastOps that were inserted by the
  // dialect conversion frameworks. (Not the one that were inserted by
  // patterns.)
  SmallVector<UnrealizedConversionCastOp> remainingCastOps;
  reconcileUnrealizedCasts(allCastOps, &remainingCastOps);

  // Try to legalize all unresolved materializations.
  if (config.buildMaterializations) {
    IRRewriter rewriter(rewriterImpl.context, config.listener);
    for (UnrealizedConversionCastOp castOp : remainingCastOps) {
      auto it = materializations.find(castOp);
      assert(it != materializations.end() && "inconsistent state");
      if (failed(legalizeUnresolvedMaterialization(rewriter, it->second)))
        return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Reconcile Unrealized Casts
//===----------------------------------------------------------------------===//

void mlir::reconcileUnrealizedCasts(
    ArrayRef<UnrealizedConversionCastOp> castOps,
    SmallVectorImpl<UnrealizedConversionCastOp> *remainingCastOps) {
  SetVector<UnrealizedConversionCastOp> worklist(castOps.begin(),
                                                 castOps.end());
  // This set is maintained only if `remainingCastOps` is provided.
  DenseSet<Operation *> erasedOps;

  // Helper function that adds all operands to the worklist that are an
  // unrealized_conversion_cast op result.
  auto enqueueOperands = [&](UnrealizedConversionCastOp castOp) {
    for (Value v : castOp.getInputs())
      if (auto inputCastOp = v.getDefiningOp<UnrealizedConversionCastOp>())
        worklist.insert(inputCastOp);
  };

  // Helper function that return the unrealized_conversion_cast op that
  // defines all inputs of the given op (in the same order). Return "nullptr"
  // if there is no such op.
  auto getInputCast =
      [](UnrealizedConversionCastOp castOp) -> UnrealizedConversionCastOp {
    if (castOp.getInputs().empty())
      return {};
    auto inputCastOp =
        castOp.getInputs().front().getDefiningOp<UnrealizedConversionCastOp>();
    if (!inputCastOp)
      return {};
    if (inputCastOp.getOutputs() != castOp.getInputs())
      return {};
    return inputCastOp;
  };

  // Process ops in the worklist bottom-to-top.
  while (!worklist.empty()) {
    UnrealizedConversionCastOp castOp = worklist.pop_back_val();
    if (castOp->use_empty()) {
      // DCE: If the op has no users, erase it. Add the operands to the
      // worklist to find additional DCE opportunities.
      enqueueOperands(castOp);
      if (remainingCastOps)
        erasedOps.insert(castOp.getOperation());
      castOp->erase();
      continue;
    }

    // Traverse the chain of input cast ops to see if an op with the same
    // input types can be found.
    UnrealizedConversionCastOp nextCast = castOp;
    while (nextCast) {
      if (nextCast.getInputs().getTypes() == castOp.getResultTypes()) {
        // Found a cast where the input types match the output types of the
        // matched op. We can directly use those inputs and the matched op can
        // be removed.
        enqueueOperands(castOp);
        castOp.replaceAllUsesWith(nextCast.getInputs());
        if (remainingCastOps)
          erasedOps.insert(castOp.getOperation());
        castOp->erase();
        break;
      }
      nextCast = getInputCast(nextCast);
    }
  }

  if (remainingCastOps)
    for (UnrealizedConversionCastOp op : castOps)
      if (!erasedOps.contains(op.getOperation()))
        remainingCastOps->push_back(op);
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
  assert(t && "expected non-null type");

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

Value TypeConverter::materializeArgumentConversion(OpBuilder &builder,
                                                   Location loc,
                                                   Type resultType,
                                                   ValueRange inputs) const {
  for (const MaterializationCallbackFn &fn :
       llvm::reverse(argumentMaterializations))
    if (Value result = fn(builder, resultType, inputs, loc))
      return result;
  return nullptr;
}

Value TypeConverter::materializeSourceConversion(OpBuilder &builder,
                                                 Location loc, Type resultType,
                                                 ValueRange inputs) const {
  for (const MaterializationCallbackFn &fn :
       llvm::reverse(sourceMaterializations))
    if (Value result = fn(builder, resultType, inputs, loc))
      return result;
  return nullptr;
}

Value TypeConverter::materializeTargetConversion(OpBuilder &builder,
                                                 Location loc, Type resultType,
                                                 ValueRange inputs,
                                                 Type originalType) const {
  SmallVector<Value> result = materializeTargetConversion(
      builder, loc, TypeRange(resultType), inputs, originalType);
  if (result.empty())
    return nullptr;
  assert(result.size() == 1 && "expected single result");
  return result.front();
}

SmallVector<Value> TypeConverter::materializeTargetConversion(
    OpBuilder &builder, Location loc, TypeRange resultTypes, ValueRange inputs,
    Type originalType) const {
  for (const TargetMaterializationCallbackFn &fn :
       llvm::reverse(targetMaterializations)) {
    SmallVector<Value> result =
        fn(builder, resultTypes, inputs, loc, originalType);
    if (result.empty())
      continue;
    assert(TypeRange(ValueRange(result)) == resultTypes &&
           "callback produced incorrect number of values or values with "
           "incorrect types");
    return result;
  }
  return {};
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

/// Find a common IsolatedFromAbove ancestor of the given ops. If at least one
/// op is a top-level module op (which is expected to be isolated from above),
/// return that op.
static Operation *findCommonAncestor(ArrayRef<Operation *> ops) {
  // Check if there is a top-level operation within `ops`. If so, return that
  // op.
  for (Operation *op : ops) {
    if (!op->getParentOp()) {
#ifndef NDEBUG
      assert(op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
             "expected top-level op to be isolated from above");
      for (Operation *other : ops)
        assert(op->isAncestor(other) &&
               "expected ops to have a common ancestor");
#endif // NDEBUG
      return op;
    }
  }

  // No top-level op. Find a common ancestor.
  Operation *commonAncestor =
      ops.front()->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
  for (Operation *op : ops.drop_front()) {
    while (!commonAncestor->isProperAncestor(op)) {
      commonAncestor =
          commonAncestor->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
      assert(commonAncestor &&
             "expected to find a common isolated from above ancestor");
    }
  }

  return commonAncestor;
}

LogicalResult mlir::applyAnalysisConversion(
    ArrayRef<Operation *> ops, ConversionTarget &target,
    const FrozenRewritePatternSet &patterns, ConversionConfig config) {
#ifndef NDEBUG
  if (config.legalizableOps)
    assert(config.legalizableOps->empty() && "expected empty set");
#endif // NDEBUG

  // Clone closted common ancestor that is isolated from above.
  Operation *commonAncestor = findCommonAncestor(ops);
  IRMapping mapping;
  Operation *clonedAncestor = commonAncestor->clone(mapping);
  // Compute inverse IR mapping.
  DenseMap<Operation *, Operation *> inverseOperationMap;
  for (auto &it : mapping.getOperationMap())
    inverseOperationMap[it.second] = it.first;

  // Convert the cloned operations. The original IR will remain unchanged.
  SmallVector<Operation *> opsToConvert = llvm::map_to_vector(
      ops, [&](Operation *op) { return mapping.lookup(op); });
  OperationConverter opConverter(target, patterns, config,
                                 OpConversionMode::Analysis);
  LogicalResult status = opConverter.convertOperations(opsToConvert);

  // Remap `legalizableOps`, so that they point to the original ops and not the
  // cloned ops.
  if (config.legalizableOps) {
    DenseSet<Operation *> originalLegalizableOps;
    for (Operation *op : *config.legalizableOps)
      originalLegalizableOps.insert(inverseOperationMap[op]);
    *config.legalizableOps = std::move(originalLegalizableOps);
  }

  // Erase the cloned IR.
  clonedAncestor->erase();
  return status;
}

LogicalResult
mlir::applyAnalysisConversion(Operation *op, ConversionTarget &target,
                              const FrozenRewritePatternSet &patterns,
                              ConversionConfig config) {
  return applyAnalysisConversion(llvm::ArrayRef(op), target, patterns, config);
}
