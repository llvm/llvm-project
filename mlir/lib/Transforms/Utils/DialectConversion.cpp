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
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
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
    return ::llvm::hash_combine_range(val);
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

  /// Lookup a value in the mapping.
  ValueVector lookup(const ValueVector &from) const;

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
    mappedTo.insert_range(newVal);

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

  void map(Value oldVal, SmallVector<Value> &&newVal) {
    map(ValueVector{oldVal}, ValueVector(std::move(newVal)));
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

/// Marker attribute for pure type conversions. I.e., mappings whose only
/// purpose is to resolve a type mismatch. (In contrast, mappings that point to
/// the replacement values of a "replaceOp" call, etc., are not pure type
/// conversions.)
static const StringRef kPureTypeConversionMarker = "__pure_type_conversion__";

/// A vector of values is a pure type conversion if all values are defined by
/// the same operation and the operation has the `kPureTypeConversionMarker`
/// attribute.
static bool isPureTypeConversion(const ValueVector &values) {
  assert(!values.empty() && "expected non-empty value vector");
  Operation *op = values.front().getDefiningOp();
  for (Value v : llvm::drop_begin(values))
    if (v.getDefiningOp() != op)
      return false;
  return op && op->hasAttr(kPureTypeConversionMarker);
}

ValueVector ConversionValueMapping::lookup(const ValueVector &from) const {
  auto it = mapping.find(from);
  if (it == mapping.end()) {
    // No mapping found: The lookup stops here.
    return {};
  }
  return it->second;
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

static void notifyIRErased(RewriterBase::Listener *listener, Operation &op);

/// Notify the listener that the given block and its contents are being erased.
static void notifyIRErased(RewriterBase::Listener *listener, Block &b) {
  for (Operation &op : b)
    notifyIRErased(listener, op);
  listener->notifyBlockErased(&b);
}

/// Notify the listener that the given operation and its contents are being
/// erased.
static void notifyIRErased(RewriterBase::Listener *listener, Operation &op) {
  for (Region &r : op.getRegions()) {
    for (Block &b : r) {
      notifyIRErased(listener, b);
    }
  }
  listener->notifyOperationErased(&op);
}

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
    assert(block && "expected block");

    // Notify the listener that the block and its contents are being erased.
    if (auto *listener =
            dyn_cast_or_null<RewriterBase::Listener>(rewriter.getListener()))
      notifyIRErased(listener, *block);
  }

  void cleanup(RewriterBase &rewriter) override {
    // Erase the contents of the block.
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block)))
      rewriter.eraseOp(&op);
    assert(block->empty() && "expected empty block");

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
                   Region *previousRegion, Region::iterator previousIt)
      : BlockRewrite(Kind::MoveBlock, rewriterImpl, block),
        region(previousRegion),
        insertBeforeBlock(previousIt == previousRegion->end() ? nullptr
                                                              : &*previousIt) {}

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
                       Operation *op, OpBuilder::InsertPoint previous)
      : OperationRewrite(Kind::MoveOperation, rewriterImpl, op),
        block(previous.getBlock()),
        insertBeforeOp(previous.getPoint() == previous.getBlock()->end()
                           ? nullptr
                           : &*previous.getPoint()) {}

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

/// Helper class that stores metadata about an unresolved materialization.
class UnresolvedMaterializationInfo {
public:
  UnresolvedMaterializationInfo() = default;
  UnresolvedMaterializationInfo(const TypeConverter *converter,
                                MaterializationKind kind, Type originalType)
      : converterAndKind(converter, kind), originalType(originalType) {}

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
};

/// An unresolved materialization, i.e., a "builtin.unrealized_conversion_cast"
/// op. Unresolved materializations fold away or are replaced with
/// source/target materializations at the end of the dialect conversion.
class UnresolvedMaterializationRewrite : public OperationRewrite {
public:
  UnresolvedMaterializationRewrite(ConversionPatternRewriterImpl &rewriterImpl,
                                   UnrealizedConversionCastOp op,
                                   ValueVector mappedValues)
      : OperationRewrite(Kind::UnresolvedMaterialization, rewriterImpl, op),
        mappedValues(std::move(mappedValues)) {}

  static bool classof(const IRRewrite *rewrite) {
    return rewrite->getKind() == Kind::UnresolvedMaterialization;
  }

  void rollback() override;

  UnrealizedConversionCastOp getOperation() const {
    return cast<UnrealizedConversionCastOp>(op);
  }

private:
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
      : context(ctx), config(config) {}

  //===--------------------------------------------------------------------===//
  // State Management
  //===--------------------------------------------------------------------===//

  /// Return the current state of the rewriter.
  RewriterState getCurrentState();

  /// Apply all requested operation rewrites. This method is invoked when the
  /// conversion process succeeds.
  void applyRewrites();

  /// Reset the state of the rewriter to a previously saved point. Optionally,
  /// the name of the pattern that triggered the rollback can specified for
  /// debugging purposes.
  void resetState(RewriterState state, StringRef patternName = "");

  /// Append a rewrite. Rewrites are committed upon success and rolled back upon
  /// failure.
  template <typename RewriteTy, typename... Args>
  void appendRewrite(Args &&...args) {
    rewrites.push_back(
        std::make_unique<RewriteTy>(*this, std::forward<Args>(args)...));
  }

  /// Undo the rewrites (motions, splits) one by one in reverse order until
  /// "numRewritesToKeep" rewrites remains. Optionally, the name of the pattern
  /// that triggered the rollback can specified for debugging purposes.
  void undoRewrites(unsigned numRewritesToKeep = 0, StringRef patternName = "");

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
  ///
  /// If `skipPureTypeConversions` is "true", materializations that are pure
  /// type conversions are not considered.
  ValueVector lookupOrDefault(Value from, TypeRange desiredTypes = {},
                              bool skipPureTypeConversions = false) const;

  /// Lookup the given value within the map, or return an empty vector if the
  /// value is not mapped. If it is mapped, this follows the same behavior
  /// as `lookupOrDefault`.
  ValueVector lookupOrNull(Value from, TypeRange desiredTypes = {}) const;

  //===--------------------------------------------------------------------===//
  // IR Rewrites / Type Conversion
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

  /// Replace the results of the given operation with the given values and
  /// erase the operation.
  ///
  /// There can be multiple replacement values for each result (1:N
  /// replacement). If the replacement values are empty, the respective result
  /// is dropped and a source materialization is built if the result still has
  /// uses.
  void replaceOp(Operation *op, SmallVector<SmallVector<Value>> &&newValues);

  /// Replace the given block argument with the given values. The specified
  /// converter is used to build materializations (if necessary).
  void replaceUsesOfBlockArgument(BlockArgument from, ValueRange to,
                                  const TypeConverter *converter);

  /// Erase the given block and its contents.
  void eraseBlock(Block *block);

  /// Inline the source block into the destination block before the given
  /// iterator.
  void inlineBlockBefore(Block *source, Block *dest, Block::iterator before);

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
  ///
  /// If `isPureTypeConversion` is "true", the materialization is created only
  /// to resolve a type mismatch. That means it is not a regular value
  /// replacement issued by the user. (Replacement values that are created
  /// "out of thin air" appear like unresolved materializations because they are
  /// unrealized_conversion_cast ops. However, they must be treated like
  /// regular value replacements.)
  ValueRange buildUnresolvedMaterialization(
      MaterializationKind kind, OpBuilder::InsertPoint ip, Location loc,
      ValueVector valuesToMap, ValueRange inputs, TypeRange outputTypes,
      Type originalType, const TypeConverter *converter,
      UnrealizedConversionCastOp *castOp = nullptr,
      bool isPureTypeConversion = true);

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

  /// Notifies that a block was inserted.
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override;

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
    SingleEraseRewriter(
        MLIRContext *context,
        std::function<void(Operation *)> opErasedCallback = nullptr)
        : RewriterBase(context, /*listener=*/this),
          opErasedCallback(opErasedCallback) {}

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

    void notifyOperationErased(Operation *op) override {
      erased.insert(op);
      if (opErasedCallback)
        opErasedCallback(op);
    }

    void notifyBlockErased(Block *block) override { erased.insert(block); }

  private:
    /// Pointers to all erased operations and blocks.
    DenseSet<void *> erased;

    /// A callback that is invoked when an operation is erased.
    std::function<void(Operation *)> opErasedCallback;
  };

  //===--------------------------------------------------------------------===//
  // State
  //===--------------------------------------------------------------------===//

  /// MLIR context.
  MLIRContext *context;

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

  /// A set of operations that were created by the current pattern.
  SetVector<Operation *> patternNewOps;

  /// A set of operations that were modified by the current pattern.
  SetVector<Operation *> patternModifiedOps;

  /// A set of blocks that were inserted (newly-created blocks or moved blocks)
  /// by the current pattern.
  SetVector<Block *> patternInsertedBlocks;

  /// A mapping for looking up metadata of unresolved materializations.
  DenseMap<UnrealizedConversionCastOp, UnresolvedMaterializationInfo>
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

  /// A raw output stream used to prefix the debug log.
  llvm::impl::raw_ldbg_ostream os{(Twine("[") + DEBUG_TYPE + "] ").str(),
                                  llvm::dbgs(), /*HasPendingNewline=*/false};

  /// A logger used to emit diagnostics during the conversion process.
  llvm::ScopedPrinter logger{os};
  std::string logPrefix;
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

  // Notify the listener that the operation and its contents are being erased.
  if (listener)
    notifyIRErased(listener, *op);

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
  SingleEraseRewriter eraseRewriter(
      context, /*opErasedCallback=*/[&](Operation *op) {
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op))
          unresolvedMaterializations.erase(castOp);
      });
  for (auto &rewrite : rewrites)
    rewrite->cleanup(eraseRewriter);
}

//===----------------------------------------------------------------------===//
// State Management
//===----------------------------------------------------------------------===//

ValueVector ConversionPatternRewriterImpl::lookupOrDefault(
    Value from, TypeRange desiredTypes, bool skipPureTypeConversions) const {
  // Helper function that looks up each value in `values` individually and then
  // composes the results. If that fails, it tries to look up the entire vector
  // at once.
  auto composedLookup = [&](const ValueVector &values) -> ValueVector {
    // If possible, replace each value with (one or multiple) mapped values.
    ValueVector next;
    for (Value v : values) {
      ValueVector r = mapping.lookup({v});
      if (!r.empty()) {
        llvm::append_range(next, r);
      } else {
        next.push_back(v);
      }
    }
    if (next != values) {
      // At least one value was replaced.
      return next;
    }

    // Otherwise: Check if there is a mapping for the entire vector. Such
    // mappings are materializations. (N:M mapping are not supported for value
    // replacements.)
    //
    // Note: From a correctness point of view, materializations do not have to
    // be stored (and looked up) in the mapping. But for performance reasons,
    // we choose to reuse existing IR (when possible) instead of creating it
    // multiple times.
    ValueVector r = mapping.lookup(values);
    if (r.empty()) {
      // No mapping found: The lookup stops here.
      return {};
    }
    return r;
  };

  // Try to find the deepest values that have the desired types. If there is no
  // such mapping, simply return the deepest values.
  ValueVector desiredValue;
  ValueVector current{from};
  ValueVector lastNonMaterialization{from};
  do {
    // Store the current value if the types match.
    bool match = TypeRange(ValueRange(current)) == desiredTypes;
    if (skipPureTypeConversions) {
      // Skip pure type conversions, if requested.
      bool pureConversion = isPureTypeConversion(current);
      match &= !pureConversion;
      // Keep track of the last mapped value that was not a pure type
      // conversion.
      if (!pureConversion)
        lastNonMaterialization = current;
    }
    if (match)
      desiredValue = current;

    // Lookup next value in the mapping.
    ValueVector next = composedLookup(current);
    if (next.empty())
      break;
    current = std::move(next);
  } while (true);

  // If the desired values were found use them, otherwise default to the leaf
  // values. (Skip pure type conversions, if requested.)
  if (!desiredTypes.empty())
    return desiredValue;
  if (skipPureTypeConversions)
    return lastNonMaterialization;
  return current;
}

ValueVector
ConversionPatternRewriterImpl::lookupOrNull(Value from,
                                            TypeRange desiredTypes) const {
  ValueVector result = lookupOrDefault(from, desiredTypes);
  if (result == ValueVector{from} ||
      (!desiredTypes.empty() && TypeRange(ValueRange(result)) != desiredTypes))
    return {};
  return result;
}

RewriterState ConversionPatternRewriterImpl::getCurrentState() {
  return RewriterState(rewrites.size(), ignoredOps.size(), replacedOps.size());
}

void ConversionPatternRewriterImpl::resetState(RewriterState state,
                                               StringRef patternName) {
  // Undo any rewrites.
  undoRewrites(state.numRewrites, patternName);

  // Pop all of the recorded ignored operations that are no longer valid.
  while (ignoredOps.size() != state.numIgnoredOperations)
    ignoredOps.pop_back();

  while (replacedOps.size() != state.numReplacedOps)
    replacedOps.pop_back();
}

void ConversionPatternRewriterImpl::undoRewrites(unsigned numRewritesToKeep,
                                                 StringRef patternName) {
  for (auto &rewrite :
       llvm::reverse(llvm::drop_begin(rewrites, numRewritesToKeep))) {
    if (!config.allowPatternRollback &&
        !isa<UnresolvedMaterializationRewrite>(rewrite)) {
      // Unresolved materializations can always be rolled back (erased).
      llvm::report_fatal_error("pattern '" + patternName +
                               "' rollback of IR modifications requested");
    }
    rewrite->rollback();
  }
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
      // The current pattern does not have a type converter. Pass the most
      // recently mapped values, excluding materializations. Materializations
      // are intentionally excluded because their presence may depend on other
      // patterns. Including materializations would make the lookup fragile
      // and unpredictable.
      remapped.push_back(lookupOrDefault(operand, /*desiredTypes=*/{},
                                         /*skipPureTypeConversions=*/true));
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

    ValueVector repl = lookupOrDefault(operand, legalTypes);
    if (!repl.empty() && TypeRange(ValueRange(repl)) == legalTypes) {
      // Mapped values have the correct type or there is an existing
      // materialization. Or the operand is not mapped at all and has the
      // correct type.
      remapped.push_back(std::move(repl));
      continue;
    }

    // Create a materialization for the most recently mapped values.
    repl = lookupOrDefault(operand, /*desiredTypes=*/{},
                           /*skipPureTypeConversions=*/true);
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
//===----------------------------------------------------------------------===//

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
    if (!inputMap || inputMap->replacedWithValues())
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
      Value mat =
          buildUnresolvedMaterialization(
              MaterializationKind::Source,
              OpBuilder::InsertPoint(newBlock, newBlock->begin()),
              origArg.getLoc(),
              /*valuesToMap=*/{}, /*inputs=*/ValueRange(),
              /*outputTypes=*/origArgType, /*originalType=*/Type(), converter,
              /*castOp=*/nullptr, /*isPureTypeConversion=*/false)
              .front();
      replaceUsesOfBlockArgument(origArg, mat, converter);
      continue;
    }

    if (inputMap->replacedWithValues()) {
      // This block argument was dropped and replacement values were provided.
      assert(inputMap->size == 0 &&
             "invalid to provide a replacement value when the argument isn't "
             "dropped");
      replaceUsesOfBlockArgument(origArg, inputMap->replacementValues,
                                 converter);
      continue;
    }

    // This is a 1->1+ mapping.
    auto replArgs =
        newBlock->getArguments().slice(inputMap->inputNo, inputMap->size);
    replaceUsesOfBlockArgument(origArg, replArgs, converter);
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
    UnrealizedConversionCastOp *castOp, bool isPureTypeConversion) {
  assert((!originalType || kind == MaterializationKind::Target) &&
         "original type is valid only for target materializations");
  assert(TypeRange(inputs) != outputTypes &&
         "materialization is not necessary");

  // Create an unresolved materialization. We use a new OpBuilder to avoid
  // tracking the materialization like we do for other operations.
  OpBuilder builder(outputTypes.front().getContext());
  builder.setInsertionPoint(ip.getBlock(), ip.getPoint());
  auto convertOp =
      UnrealizedConversionCastOp::create(builder, loc, outputTypes, inputs);
  if (isPureTypeConversion)
    convertOp->setAttr(kPureTypeConversionMarker, builder.getUnitAttr());
  if (!valuesToMap.empty())
    mapping.map(valuesToMap, convertOp.getResults());
  if (castOp)
    *castOp = convertOp;
  unresolvedMaterializations[convertOp] =
      UnresolvedMaterializationInfo(converter, kind, originalType);
  appendRewrite<UnresolvedMaterializationRewrite>(convertOp,
                                                  std::move(valuesToMap));
  return convertOp.getResults();
}

Value ConversionPatternRewriterImpl::findOrBuildReplacementValue(
    Value value, const TypeConverter *converter) {
  // Try to find a replacement value with the same type in the conversion value
  // mapping. This includes cached materializations. We try to reuse those
  // instead of generating duplicate IR.
  ValueVector repl = lookupOrNull(value, value.getType());
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
  repl = lookupOrNull(value);
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
                                     /*outputTypes=*/value.getType(),
                                     /*originalType=*/Type(), converter)
          .front();
  return castValue;
}

//===----------------------------------------------------------------------===//
// Rewriter Notification Hooks
//===----------------------------------------------------------------------===//

void ConversionPatternRewriterImpl::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
  // If no previous insertion point is provided, the op used to be detached.
  bool wasDetached = !previous.isSet();
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "' (" << op
                       << ")";
    if (wasDetached)
      logger.getOStream() << " (was detached)";
    logger.getOStream() << "\n";
  });
  assert(!wasOpReplaced(op->getParentOp()) &&
         "attempting to insert into a block within a replaced/erased op");

  if (wasDetached) {
    // If the op was detached, it is most likely a newly created op.
    // TODO: If the same op is inserted multiple times from a detached state,
    // the rollback mechanism may erase the same op multiple times. This is a
    // bug in the rollback-based dialect conversion driver.
    appendRewrite<CreateOperationRewrite>(op);
    patternNewOps.insert(op);
    return;
  }

  // The op was moved from one place to another.
  appendRewrite<MoveOperationRewrite>(op, previous);
}

void ConversionPatternRewriterImpl::replaceOp(
    Operation *op, SmallVector<SmallVector<Value>> &&newValues) {
  assert(newValues.size() == op->getNumResults());
  assert(!ignoredOps.contains(op) && "operation was already replaced");

  // Check if replaced op is an unresolved materialization, i.e., an
  // unrealized_conversion_cast op that was created by the conversion driver.
  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    // Make sure that the user does not mess with unresolved materializations
    // that were inserted by the conversion driver. We keep track of these
    // ops in internal data structures.
    assert(!unresolvedMaterializations.contains(castOp) &&
           "attempting to replace/erase an unresolved materialization");
  }

  // Create mappings for each of the new result values.
  for (auto [repl, result] : llvm::zip_equal(newValues, op->getResults())) {
    if (repl.empty()) {
      // This result was dropped and no replacement value was provided.
      // Materialize a replacement value "out of thin air".
      buildUnresolvedMaterialization(
          MaterializationKind::Source, computeInsertPoint(result),
          result.getLoc(), /*valuesToMap=*/{result}, /*inputs=*/ValueRange(),
          /*outputTypes=*/result.getType(), /*originalType=*/Type(),
          currentTypeConverter, /*castOp=*/nullptr,
          /*isPureTypeConversion=*/false);
      continue;
    }

    // Remap result to replacement value.
    if (repl.empty())
      continue;
    mapping.map(static_cast<Value>(result), std::move(repl));
  }

  appendRewrite<ReplaceOperationRewrite>(op, currentTypeConverter);
  // Mark this operation and all nested ops as replaced.
  op->walk([&](Operation *op) { replacedOps.insert(op); });
}

void ConversionPatternRewriterImpl::replaceUsesOfBlockArgument(
    BlockArgument from, ValueRange to, const TypeConverter *converter) {
  appendRewrite<ReplaceBlockArgRewrite>(from.getOwner(), from, converter);
  mapping.map(from, to);
}

void ConversionPatternRewriterImpl::eraseBlock(Block *block) {
  assert(!wasOpReplaced(block->getParentOp()) &&
         "attempting to erase a block within a replaced/erased op");
  appendRewrite<EraseBlockRewrite>(block);

  // Unlink the block from its parent region. The block is kept in the rewrite
  // object and will be actually destroyed when rewrites are applied. This
  // allows us to keep the operations in the block live and undo the removal by
  // re-inserting the block.
  block->getParent()->getBlocks().remove(block);

  // Mark all nested ops as erased.
  block->walk([&](Operation *op) { replacedOps.insert(op); });
}

void ConversionPatternRewriterImpl::notifyBlockInserted(
    Block *block, Region *previous, Region::iterator previousIt) {
  // If no previous insertion point is provided, the block used to be detached.
  bool wasDetached = !previous;
  Operation *newParentOp = block->getParentOp();
  LLVM_DEBUG(
      {
        Operation *parent = newParentOp;
        if (parent) {
          logger.startLine() << "** Insert Block into : '" << parent->getName()
                             << "' (" << parent << ")";
        } else {
          logger.startLine()
              << "** Insert Block into detached Region (nullptr parent op)";
        }
        if (wasDetached)
          logger.getOStream() << " (was detached)";
        logger.getOStream() << "\n";
      });
  assert(!wasOpReplaced(newParentOp) &&
         "attempting to insert into a region within a replaced/erased op");
  (void)newParentOp;

  patternInsertedBlocks.insert(block);

  if (wasDetached) {
    // If the block was detached, it is most likely a newly created block.
    // TODO: If the same block is inserted multiple times from a detached state,
    // the rollback mechanism may erase the same block multiple times. This is a
    // bug in the rollback-based dialect conversion driver.
    appendRewrite<CreateBlockRewrite>(block);
    return;
  }

  // The block was moved from one place to another.
  appendRewrite<MoveBlockRewrite>(block, previous, previousIt);
}

void ConversionPatternRewriterImpl::inlineBlockBefore(Block *source,
                                                      Block *dest,
                                                      Block::iterator before) {
  appendRewrite<InlineBlockRewrite>(dest, source, before);
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

const ConversionConfig &ConversionPatternRewriter::getConfig() const {
  return impl->config;
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

  // If the current insertion point is before the erased operation, we adjust
  // the insertion point to be after the operation.
  if (getInsertionPoint() == op->getIterator())
    setInsertionPointAfter(op);

  SmallVector<SmallVector<Value>> newVals =
      llvm::map_to_vector(newValues, [](Value v) -> SmallVector<Value> {
        return v ? SmallVector<Value>{v} : SmallVector<Value>();
      });
  impl->replaceOp(op, std::move(newVals));
}

void ConversionPatternRewriter::replaceOpWithMultiple(
    Operation *op, SmallVector<SmallVector<Value>> &&newValues) {
  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Replace : '" << op->getName() << "'(" << op << ")\n";
  });

  // If the current insertion point is before the erased operation, we adjust
  // the insertion point to be after the operation.
  if (getInsertionPoint() == op->getIterator())
    setInsertionPointAfter(op);

  impl->replaceOp(op, std::move(newValues));
}

void ConversionPatternRewriter::eraseOp(Operation *op) {
  LLVM_DEBUG({
    impl->logger.startLine()
        << "** Erase   : '" << op->getName() << "'(" << op << ")\n";
  });

  // If the current insertion point is before the erased operation, we adjust
  // the insertion point to be after the operation.
  if (getInsertionPoint() == op->getIterator())
    setInsertionPointAfter(op);

  SmallVector<SmallVector<Value>> nullRepls(op->getNumResults(), {});
  impl->replaceOp(op, std::move(nullRepls));
}

void ConversionPatternRewriter::eraseBlock(Block *block) {
  impl->eraseBlock(block);
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
                                                           ValueRange to) {
  LLVM_DEBUG({
    impl->logger.startLine() << "** Replace Argument : '" << from << "'";
    if (Operation *parentOp = from.getOwner()->getParentOp()) {
      impl->logger.getOStream() << " (in region of '" << parentOp->getName()
                                << "' (" << parentOp << ")\n";
    } else {
      impl->logger.getOStream() << " (unlinked block)\n";
    }
  });
  impl->replaceUsesOfBlockArgument(from, to, impl->currentTypeConverter);
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
  bool fastPath = !getConfig().listener;

  if (fastPath)
    impl->inlineBlockBefore(source, dest, before);

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

  // If the current insertion point is within the source block, adjust the
  // insertion point to the destination block.
  if (getInsertionBlock() == source)
    setInsertionPoint(dest, getInsertionPoint());

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
  impl->patternModifiedOps.insert(op);

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
                                      const RewriterState &curState,
                                      const SetVector<Operation *> &newOps,
                                      const SetVector<Operation *> &modifiedOps,
                                      const SetVector<Block *> &insertedBlocks);

  /// Legalizes the actions registered during the execution of a pattern.
  LogicalResult
  legalizePatternBlockRewrites(Operation *op,
                               ConversionPatternRewriter &rewriter,
                               ConversionPatternRewriterImpl &impl,
                               const SetVector<Block *> &insertedBlocks,
                               const SetVector<Operation *> &newOps);
  LogicalResult
  legalizePatternCreatedOperations(ConversionPatternRewriter &rewriter,
                                   ConversionPatternRewriterImpl &impl,
                                   const SetVector<Operation *> &newOps);
  LogicalResult
  legalizePatternRootUpdates(ConversionPatternRewriter &rewriter,
                             ConversionPatternRewriterImpl &impl,
                             const SetVector<Operation *> &modifiedOps);

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

  // Check to see if the operation is ignored and doesn't need to be converted.
  bool isIgnored = rewriter.getImpl().isOpIgnored(op);

  LLVM_DEBUG({
    logger.getOStream() << "\n";
    logger.startLine() << logLineComment;
    logger.startLine() << "Legalizing operation : ";
    // Do not print the operation name if the operation is ignored. Ignored ops
    // may have been erased and should not be accessed. The pointer can be
    // printed safely.
    if (!isIgnored)
      logger.getOStream() << "'" << op->getName() << "' ";
    logger.getOStream() << "(" << op << ") {\n";
    logger.indent();

    // If the operation has no regions, just print it here.
    if (!isIgnored && op->getNumRegions() == 0) {
      logger.startLine() << OpWithFlags(op,
                                        OpPrintingFlags().printGenericOpForm())
                         << "\n";
    }
  });

  if (isIgnored) {
    LLVM_DEBUG({
      logSuccess(logger, "operation marked 'ignored' during conversion");
      logger.startLine() << logLineComment;
    });
    return success();
  }

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

/// Helper function that moves and returns the given object. Also resets the
/// original object, so that it is in a valid, empty state again.
template <typename T>
static T moveAndReset(T &obj) {
  T result = std::move(obj);
  obj = T();
  return result;
}

LogicalResult
OperationLegalizer::legalizeWithFold(Operation *op,
                                     ConversionPatternRewriter &rewriter) {
  auto &rewriterImpl = rewriter.getImpl();
  LLVM_DEBUG({
    rewriterImpl.logger.startLine() << "* Fold {\n";
    rewriterImpl.logger.indent();
  });

  // Clear pattern state, so that the next pattern application starts with a
  // clean slate. (The op/block sets are populated by listener notifications.)
  auto cleanup = llvm::make_scope_exit([&]() {
    rewriterImpl.patternNewOps.clear();
    rewriterImpl.patternModifiedOps.clear();
    rewriterImpl.patternInsertedBlocks.clear();
  });

  // Upon failure, undo all changes made by the folder.
  RewriterState curState = rewriterImpl.getCurrentState();

  // Try to fold the operation.
  StringRef opName = op->getName().getStringRef();
  SmallVector<Value, 2> replacementValues;
  SmallVector<Operation *, 2> newOps;
  rewriter.setInsertionPoint(op);
  rewriter.startOpModification(op);
  if (failed(rewriter.tryFold(op, replacementValues, &newOps))) {
    LLVM_DEBUG(logFailure(rewriterImpl.logger, "unable to fold"));
    rewriter.cancelOpModification(op);
    return failure();
  }
  rewriter.finalizeOpModification(op);

  // An empty list of replacement values indicates that the fold was in-place.
  // As the operation changed, a new legalization needs to be attempted.
  if (replacementValues.empty())
    return legalize(op, rewriter);

  // Insert a replacement for 'op' with the folded replacement values.
  rewriter.replaceOp(op, replacementValues);

  // Recursively legalize any new constant operations.
  for (Operation *newOp : newOps) {
    if (failed(legalize(newOp, rewriter))) {
      LLVM_DEBUG(logFailure(rewriterImpl.logger,
                            "failed to legalize generated constant '{0}'",
                            newOp->getName()));
      if (!rewriter.getConfig().allowPatternRollback) {
        // Rolling back a folder is like rolling back a pattern.
        llvm::report_fatal_error(
            "op '" + opName +
            "' folder rollback of IR modifications requested");
      }
      rewriterImpl.resetState(
          curState, std::string(op->getName().getStringRef()) + " folder");
      return failure();
    }
  }

  LLVM_DEBUG(logSuccess(rewriterImpl.logger, ""));
  return success();
}

/// Report a fatal error indicating that newly produced or modified IR could
/// not be legalized.
static void
reportNewIrLegalizationFatalError(const Pattern &pattern,
                                  const SetVector<Operation *> &newOps,
                                  const SetVector<Operation *> &modifiedOps,
                                  const SetVector<Block *> &insertedBlocks) {
  auto newOpNames = llvm::map_range(
      newOps, [](Operation *op) { return op->getName().getStringRef(); });
  auto modifiedOpNames = llvm::map_range(
      modifiedOps, [](Operation *op) { return op->getName().getStringRef(); });
  StringRef detachedBlockStr = "(detached block)";
  auto insertedBlockNames = llvm::map_range(insertedBlocks, [&](Block *block) {
    if (block->getParentOp())
      return block->getParentOp()->getName().getStringRef();
    return detachedBlockStr;
  });
  llvm::report_fatal_error(
      "pattern '" + pattern.getDebugName() +
      "' produced IR that could not be legalized. " + "new ops: {" +
      llvm::join(newOpNames, ", ") + "}, " + "modified ops: {" +
      llvm::join(modifiedOpNames, ", ") + "}, " + "inserted block into ops: {" +
      llvm::join(insertedBlockNames, ", ") + "}");
}

LogicalResult
OperationLegalizer::legalizeWithPattern(Operation *op,
                                        ConversionPatternRewriter &rewriter) {
  auto &rewriterImpl = rewriter.getImpl();
  const ConversionConfig &config = rewriter.getConfig();

#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  Operation *checkOp;
  std::optional<OperationFingerPrint> topLevelFingerPrint;
  if (!rewriterImpl.config.allowPatternRollback) {
    // The op may be getting erased, so we have to check the parent op.
    // (In rare cases, a pattern may even erase the parent op, which will cause
    // a crash here. Expensive checks are "best effort".) Skip the check if the
    // op does not have a parent op.
    if ((checkOp = op->getParentOp())) {
      if (!op->getContext()->isMultithreadingEnabled()) {
        topLevelFingerPrint = OperationFingerPrint(checkOp);
      } else {
        // Another thread may be modifying a sibling operation. Therefore, the
        // fingerprinting mechanism of the parent op works only in
        // single-threaded mode.
        LLVM_DEBUG({
          rewriterImpl.logger.startLine()
              << "WARNING: Multi-threadeding is enabled. Some dialect "
                 "conversion expensive checks are skipped in multithreading "
                 "mode!\n";
        });
      }
    }
  }
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

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
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
    if (!rewriterImpl.config.allowPatternRollback) {
      // Returning "failure" after modifying IR is not allowed.
      if (checkOp) {
        OperationFingerPrint fingerPrintAfterPattern(checkOp);
        if (fingerPrintAfterPattern != *topLevelFingerPrint)
          llvm::report_fatal_error("pattern '" + pattern.getDebugName() +
                                   "' returned failure but IR did change");
      }
    }
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
    rewriterImpl.patternNewOps.clear();
    rewriterImpl.patternModifiedOps.clear();
    rewriterImpl.patternInsertedBlocks.clear();
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
    rewriterImpl.resetState(curState, pattern.getDebugName());
    appliedPatterns.erase(&pattern);
  };

  // Functor that performs additional legalization when a pattern is
  // successfully applied.
  auto onSuccess = [&](const Pattern &pattern) {
    assert(rewriterImpl.pendingRootUpdates.empty() && "dangling root updates");
    SetVector<Operation *> newOps = moveAndReset(rewriterImpl.patternNewOps);
    SetVector<Operation *> modifiedOps =
        moveAndReset(rewriterImpl.patternModifiedOps);
    SetVector<Block *> insertedBlocks =
        moveAndReset(rewriterImpl.patternInsertedBlocks);
    auto result = legalizePatternResult(op, pattern, rewriter, curState, newOps,
                                        modifiedOps, insertedBlocks);
    appliedPatterns.erase(&pattern);
    if (failed(result)) {
      if (!rewriterImpl.config.allowPatternRollback)
        reportNewIrLegalizationFatalError(pattern, newOps, modifiedOps,
                                          insertedBlocks);
      rewriterImpl.resetState(curState, pattern.getDebugName());
    }
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

LogicalResult OperationLegalizer::legalizePatternResult(
    Operation *op, const Pattern &pattern, ConversionPatternRewriter &rewriter,
    const RewriterState &curState, const SetVector<Operation *> &newOps,
    const SetVector<Operation *> &modifiedOps,
    const SetVector<Block *> &insertedBlocks) {
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
    llvm::report_fatal_error(
        "expected pattern to replace the root operation or modify it in place");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

  // Legalize each of the actions registered during application.
  if (failed(legalizePatternBlockRewrites(op, rewriter, impl, insertedBlocks,
                                          newOps)) ||
      failed(legalizePatternRootUpdates(rewriter, impl, modifiedOps)) ||
      failed(legalizePatternCreatedOperations(rewriter, impl, newOps))) {
    return failure();
  }

  LLVM_DEBUG(logSuccess(impl.logger, "pattern applied successfully"));
  return success();
}

LogicalResult OperationLegalizer::legalizePatternBlockRewrites(
    Operation *op, ConversionPatternRewriter &rewriter,
    ConversionPatternRewriterImpl &impl,
    const SetVector<Block *> &insertedBlocks,
    const SetVector<Operation *> &newOps) {
  SmallPtrSet<Operation *, 16> alreadyLegalized;

  // If the pattern moved or created any blocks, make sure the types of block
  // arguments get legalized.
  for (Block *block : insertedBlocks) {
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

    // Otherwise, try to legalize the parent operation if it was not generated
    // by this pattern. This is because we will attempt to legalize the parent
    // operation, and blocks in regions created by this pattern will already be
    // legalized later on.
    if (!newOps.count(parentOp) && alreadyLegalized.insert(parentOp).second) {
      if (failed(legalize(parentOp, rewriter))) {
        LLVM_DEBUG(logFailure(
            impl.logger, "operation '{0}'({1}) became illegal after rewrite",
            parentOp->getName(), parentOp));
        return failure();
      }
    }
  }
  return success();
}

LogicalResult OperationLegalizer::legalizePatternCreatedOperations(
    ConversionPatternRewriter &rewriter, ConversionPatternRewriterImpl &impl,
    const SetVector<Operation *> &newOps) {
  for (Operation *op : newOps) {
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
    const SetVector<Operation *> &modifiedOps) {
  for (Operation *op : modifiedOps) {
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
//===----------------------------------------------------------------------===//

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
  llvm::stable_sort(patternsByDepth,
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
      : config(config), opLegalizer(target, patterns), mode(mode) {}

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
                                  UnrealizedConversionCastOp op,
                                  const UnresolvedMaterializationInfo &info) {
  assert(!op.use_empty() &&
         "expected that dead materializations have already been DCE'd");
  Operation::operand_range inputOperands = op.getOperands();

  // Try to materialize the conversion.
  if (const TypeConverter *converter = info.getConverter()) {
    rewriter.setInsertionPoint(op);
    SmallVector<Value> newMaterialization;
    switch (info.getMaterializationKind()) {
    case MaterializationKind::Target:
      newMaterialization = converter->materializeTargetConversion(
          rewriter, op->getLoc(), op.getResultTypes(), inputOperands,
          info.getOriginalType());
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
  assert(!ops.empty() && "expected at least one operation");
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

  for (auto *op : toConvert) {
    if (failed(convert(rewriter, op))) {
      // Dialect conversion failed.
      if (rewriterImpl.config.allowPatternRollback) {
        // Rollback is allowed: restore the original IR.
        rewriterImpl.undoRewrites();
      } else {
        // Rollback is not allowed: apply all modifications that have been
        // performed so far.
        rewriterImpl.applyRewrites();
      }
      return failure();
    }
  }

  // After a successful conversion, apply rewrites.
  rewriterImpl.applyRewrites();

  // Gather all unresolved materializations.
  SmallVector<UnrealizedConversionCastOp> allCastOps;
  const DenseMap<UnrealizedConversionCastOp, UnresolvedMaterializationInfo>
      &materializations = rewriterImpl.unresolvedMaterializations;
  for (auto it : materializations)
    allCastOps.push_back(it.first);

  // Reconcile all UnrealizedConversionCastOps that were inserted by the
  // dialect conversion frameworks. (Not the one that were inserted by
  // patterns.)
  SmallVector<UnrealizedConversionCastOp> remainingCastOps;
  reconcileUnrealizedCasts(allCastOps, &remainingCastOps);

  // Drop markers.
  for (UnrealizedConversionCastOp castOp : remainingCastOps)
    castOp->removeAttr(kPureTypeConversionMarker);

  // Try to legalize all unresolved materializations.
  if (config.buildMaterializations) {
    IRRewriter rewriter(rewriterImpl.context, config.listener);
    for (UnrealizedConversionCastOp castOp : remainingCastOps) {
      auto it = materializations.find(castOp);
      assert(it != materializations.end() && "inconsistent state");
      if (failed(
              legalizeUnresolvedMaterialization(rewriter, castOp, it->second)))
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
  SetVector<UnrealizedConversionCastOp> worklist(llvm::from_range, castOps);
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
      InputMapping{newInputNo, newInputCount, /*replacementValues=*/{}};
}

void TypeConverter::SignatureConversion::remapInput(
    unsigned origInputNo, ArrayRef<Value> replacements) {
  assert(!remappedInputs[origInputNo] && "input has already been remapped");
  remappedInputs[origInputNo] = InputMapping{
      origInputNo, /*size=*/0,
      SmallVector<Value, 1>(replacements.begin(), replacements.end())};
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
        assert(results.size() == currentCount &&
               "failed type conversion should not change results");
        cachedDirectConversions.try_emplace(t, nullptr);
        return failure();
      }
      auto newTypes = ArrayRef<Type>(results).drop_front(currentCount);
      if (newTypes.size() == 1)
        cachedDirectConversions.try_emplace(t, newTypes.front());
      else
        cachedMultiConversions.try_emplace(t, llvm::to_vector<2>(newTypes));
      return success();
    } else {
      assert(results.size() == currentCount &&
             "failed type conversion should not change results");
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

Value TypeConverter::materializeSourceConversion(OpBuilder &builder,
                                                 Location loc, Type resultType,
                                                 ValueRange inputs) const {
  for (const SourceMaterializationCallbackFn &fn :
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

/// This is the type of Action that is dispatched when a conversion is applied.
class ApplyConversionAction
    : public tracing::ActionImpl<ApplyConversionAction> {
public:
  using Base = tracing::ActionImpl<ApplyConversionAction>;
  ApplyConversionAction(ArrayRef<IRUnit> irUnits) : Base(irUnits) {}
  static constexpr StringLiteral tag = "apply-conversion";
  static constexpr StringLiteral desc =
      "Encapsulate the application of a dialect conversion";

  void print(raw_ostream &os) const override { os << tag; }
};

static LogicalResult applyConversion(ArrayRef<Operation *> ops,
                                     const ConversionTarget &target,
                                     const FrozenRewritePatternSet &patterns,
                                     ConversionConfig config,
                                     OpConversionMode mode) {
  if (ops.empty())
    return success();
  MLIRContext *ctx = ops.front()->getContext();
  LogicalResult status = success();
  SmallVector<IRUnit> irUnits(ops.begin(), ops.end());
  ctx->executeAction<ApplyConversionAction>(
      [&] {
        OperationConverter opConverter(target, patterns, config, mode);
        status = opConverter.convertOperations(ops);
      },
      irUnits);
  return status;
}

//===----------------------------------------------------------------------===//
// Partial Conversion
//===----------------------------------------------------------------------===//

LogicalResult mlir::applyPartialConversion(
    ArrayRef<Operation *> ops, const ConversionTarget &target,
    const FrozenRewritePatternSet &patterns, ConversionConfig config) {
  return applyConversion(ops, target, patterns, config,
                         OpConversionMode::Partial);
}
LogicalResult
mlir::applyPartialConversion(Operation *op, const ConversionTarget &target,
                             const FrozenRewritePatternSet &patterns,
                             ConversionConfig config) {
  return applyPartialConversion(llvm::ArrayRef(op), target, patterns, config);
}

//===----------------------------------------------------------------------===//
// Full Conversion
//===----------------------------------------------------------------------===//

LogicalResult mlir::applyFullConversion(ArrayRef<Operation *> ops,
                                        const ConversionTarget &target,
                                        const FrozenRewritePatternSet &patterns,
                                        ConversionConfig config) {
  return applyConversion(ops, target, patterns, config, OpConversionMode::Full);
}
LogicalResult mlir::applyFullConversion(Operation *op,
                                        const ConversionTarget &target,
                                        const FrozenRewritePatternSet &patterns,
                                        ConversionConfig config) {
  return applyFullConversion(llvm::ArrayRef(op), target, patterns, config);
}

//===----------------------------------------------------------------------===//
// Analysis Conversion
//===----------------------------------------------------------------------===//

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
  LogicalResult status = applyConversion(opsToConvert, target, patterns, config,
                                         OpConversionMode::Analysis);

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
