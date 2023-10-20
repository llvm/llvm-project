//===- IRNumbering.cpp - MLIR Bytecode IR numbering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRNumbering.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::bytecode::detail;

//===----------------------------------------------------------------------===//
// NumberingDialectWriter
//===----------------------------------------------------------------------===//

struct IRNumberingState::NumberingDialectWriter : public DialectBytecodeWriter {
  NumberingDialectWriter(IRNumberingState &state) : state(state) {}

  void writeAttribute(Attribute attr) override { state.number(attr); }
  void writeOptionalAttribute(Attribute attr) override {
    if (attr)
      state.number(attr);
  }
  void writeType(Type type) override { state.number(type); }
  void writeResourceHandle(const AsmDialectResourceHandle &resource) override {
    state.number(resource.getDialect(), resource);
  }

  /// Stubbed out methods that are not used for numbering.
  void writeVarInt(uint64_t) override {}
  void writeSignedVarInt(int64_t value) override {}
  void writeAPIntWithKnownWidth(const APInt &value) override {}
  void writeAPFloatWithKnownSemantics(const APFloat &value) override {}
  void writeOwnedString(StringRef) override {
    // TODO: It might be nice to prenumber strings and sort by the number of
    // references. This could potentially be useful for optimizing things like
    // file locations.
  }
  void writeOwnedBlob(ArrayRef<char> blob) override {}
  void writeOwnedBool(bool value) override {}

  int64_t getBytecodeVersion() const override {
    return state.getDesiredBytecodeVersion();
  }

  /// The parent numbering state that is populated by this writer.
  IRNumberingState &state;
};

//===----------------------------------------------------------------------===//
// IR Numbering
//===----------------------------------------------------------------------===//

/// Group and sort the elements of the given range by their parent dialect. This
/// grouping is applied to sub-sections of the ranged defined by how many bytes
/// it takes to encode a varint index to that sub-section.
template <typename T>
static void groupByDialectPerByte(T range) {
  if (range.empty())
    return;

  // A functor used to sort by a given dialect, with a desired dialect to be
  // ordered first (to better enable sharing of dialects across byte groups).
  auto sortByDialect = [](unsigned dialectToOrderFirst, const auto &lhs,
                          const auto &rhs) {
    if (lhs->dialect->number == dialectToOrderFirst)
      return rhs->dialect->number != dialectToOrderFirst;
    if (rhs->dialect->number == dialectToOrderFirst)
      return false;
    return lhs->dialect->number < rhs->dialect->number;
  };

  unsigned dialectToOrderFirst = 0;
  size_t elementsInByteGroup = 0;
  auto iterRange = range;
  for (unsigned i = 1; i < 9; ++i) {
    // Update the number of elements in the current byte grouping. Reminder
    // that varint encodes 7-bits per byte, so that's how we compute the
    // number of elements in each byte grouping.
    elementsInByteGroup = (1ULL << (7ULL * i)) - elementsInByteGroup;

    // Slice out the sub-set of elements that are in the current byte grouping
    // to be sorted.
    auto byteSubRange = iterRange.take_front(elementsInByteGroup);
    iterRange = iterRange.drop_front(byteSubRange.size());

    // Sort the sub range for this byte.
    llvm::stable_sort(byteSubRange, [&](const auto &lhs, const auto &rhs) {
      return sortByDialect(dialectToOrderFirst, lhs, rhs);
    });

    // Update the dialect to order first to be the dialect at the end of the
    // current grouping. This seeks to allow larger dialect groupings across
    // byte boundaries.
    dialectToOrderFirst = byteSubRange.back()->dialect->number;

    // If the data range is now empty, we are done.
    if (iterRange.empty())
      break;
  }

  // Assign the entry numbers based on the sort order.
  for (auto [idx, value] : llvm::enumerate(range))
    value->number = idx;
}

IRNumberingState::IRNumberingState(Operation *op,
                                   const BytecodeWriterConfig &config)
    : config(config) {
  computeGlobalNumberingState(op);

  // Number the root operation.
  number(*op);

  // A worklist of region contexts to number and the next value id before that
  // region.
  SmallVector<std::pair<Region *, unsigned>, 8> numberContext;

  // Functor to push the regions of the given operation onto the numbering
  // context.
  auto addOpRegionsToNumber = [&](Operation *op) {
    MutableArrayRef<Region> regions = op->getRegions();
    if (regions.empty())
      return;

    // Isolated regions don't share value numbers with their parent, so we can
    // start numbering these regions at zero.
    unsigned opFirstValueID = isIsolatedFromAbove(op) ? 0 : nextValueID;
    for (Region &region : regions)
      numberContext.emplace_back(&region, opFirstValueID);
  };
  addOpRegionsToNumber(op);

  // Iteratively process each of the nested regions.
  while (!numberContext.empty()) {
    Region *region;
    std::tie(region, nextValueID) = numberContext.pop_back_val();
    number(*region);

    // Traverse into nested regions.
    for (Operation &op : region->getOps())
      addOpRegionsToNumber(&op);
  }

  // Number each of the dialects. For now this is just in the order they were
  // found, given that the number of dialects on average is small enough to fit
  // within a singly byte (128). If we ever have real world use cases that have
  // a huge number of dialects, this could be made more intelligent.
  for (auto [idx, dialect] : llvm::enumerate(dialects))
    dialect.second->number = idx;

  // Number each of the recorded components within each dialect.

  // First sort by ref count so that the most referenced elements are first. We
  // try to bias more heavily used elements to the front. This allows for more
  // frequently referenced things to be encoded using smaller varints.
  auto sortByRefCountFn = [](const auto &lhs, const auto &rhs) {
    return lhs->refCount > rhs->refCount;
  };
  llvm::stable_sort(orderedAttrs, sortByRefCountFn);
  llvm::stable_sort(orderedOpNames, sortByRefCountFn);
  llvm::stable_sort(orderedTypes, sortByRefCountFn);

  // After that, we apply a secondary ordering based on the parent dialect. This
  // ordering is applied to sub-sections of the element list defined by how many
  // bytes it takes to encode a varint index to that sub-section. This allows
  // for more efficiently encoding components of the same dialect (e.g. we only
  // have to encode the dialect reference once).
  groupByDialectPerByte(llvm::MutableArrayRef(orderedAttrs));
  groupByDialectPerByte(llvm::MutableArrayRef(orderedOpNames));
  groupByDialectPerByte(llvm::MutableArrayRef(orderedTypes));

  // Finalize the numbering of the dialect resources.
  finalizeDialectResourceNumberings(op);
}

void IRNumberingState::computeGlobalNumberingState(Operation *rootOp) {
  // A simple state struct tracking data used when walking operations.
  struct StackState {
    /// The operation currently being walked.
    Operation *op;

    /// The numbering of the operation.
    OperationNumbering *numbering;

    /// A flag indicating if the current state or one of its parents has
    /// unresolved isolation status. This is tracked separately from the
    /// isIsolatedFromAbove bit on `numbering` because we need to be able to
    /// handle the given case:
    ///   top.op {
    ///     %value = ...
    ///     middle.op {
    ///       %value2 = ...
    ///       inner.op {
    ///         // Here we mark `inner.op` as not isolated. Note `middle.op`
    ///         // isn't known not isolated yet.
    ///         use.op %value2
    ///
    ///         // Here inner.op is already known to be non-isolated, but
    ///         // `middle.op` is now also discovered to be non-isolated.
    ///         use.op %value
    ///       }
    ///     }
    ///   }
    bool hasUnresolvedIsolation;
  };

  // Compute a global operation ID numbering according to the pre-order walk of
  // the IR. This is used as reference to construct use-list orders.
  unsigned operationID = 0;

  // Walk each of the operations within the IR, tracking a stack of operations
  // as we recurse into nested regions. This walk method hooks in at two stages
  // during the walk:
  //
  //   BeforeAllRegions:
  //     Here we generate a numbering for the operation and push it onto the
  //     stack if it has regions. We also compute the isolation status of parent
  //     regions at this stage. This is done by checking the parent regions of
  //     operands used by the operation, and marking each region between the
  //     the operand region and the current as not isolated. See
  //     StackState::hasUnresolvedIsolation above for an example.
  //
  //   AfterAllRegions:
  //     Here we pop the operation from the stack, and if it hasn't been marked
  //     as non-isolated, we mark it as so. A non-isolated use would have been
  //     found while walking the regions, so it is safe to mark the operation at
  //     this point.
  //
  SmallVector<StackState> opStack;
  rootOp->walk([&](Operation *op, const WalkStage &stage) {
    // After visiting all nested regions, we pop the operation from the stack.
    if (op->getNumRegions() && stage.isAfterAllRegions()) {
      // If no non-isolated uses were found, we can safely mark this operation
      // as isolated from above.
      OperationNumbering *numbering = opStack.pop_back_val().numbering;
      if (!numbering->isIsolatedFromAbove.has_value())
        numbering->isIsolatedFromAbove = true;
      return;
    }

    // When visiting before nested regions, we process "IsolatedFromAbove"
    // checks and compute the number for this operation.
    if (!stage.isBeforeAllRegions())
      return;
    // Update the isolation status of parent regions if any have yet to be
    // resolved.
    if (!opStack.empty() && opStack.back().hasUnresolvedIsolation) {
      Region *parentRegion = op->getParentRegion();
      for (Value operand : op->getOperands()) {
        Region *operandRegion = operand.getParentRegion();
        if (operandRegion == parentRegion)
          continue;
        // We've found a use of an operand outside of the current region,
        // walk the operation stack searching for the parent operation,
        // marking every region on the way as not isolated.
        Operation *operandContainerOp = operandRegion->getParentOp();
        auto it = std::find_if(
            opStack.rbegin(), opStack.rend(), [=](const StackState &it) {
              // We only need to mark up to the container region, or the first
              // that has an unresolved status.
              return !it.hasUnresolvedIsolation || it.op == operandContainerOp;
            });
        assert(it != opStack.rend() && "expected to find the container");
        for (auto &state : llvm::make_range(opStack.rbegin(), it)) {
          // If we stopped at a region that knows its isolation status, we can
          // stop updating the isolation status for the parent regions.
          state.hasUnresolvedIsolation = it->hasUnresolvedIsolation;
          state.numbering->isIsolatedFromAbove = false;
        }
      }
    }

    // Compute the number for this op and push it onto the stack.
    auto *numbering =
        new (opAllocator.Allocate()) OperationNumbering(operationID++);
    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      numbering->isIsolatedFromAbove = true;
    operations.try_emplace(op, numbering);
    if (op->getNumRegions()) {
      opStack.emplace_back(StackState{
          op, numbering, !numbering->isIsolatedFromAbove.has_value()});
    }
  });
}

void IRNumberingState::number(Attribute attr) {
  auto it = attrs.insert({attr, nullptr});
  if (!it.second) {
    ++it.first->second->refCount;
    return;
  }
  auto *numbering = new (attrAllocator.Allocate()) AttributeNumbering(attr);
  it.first->second = numbering;
  orderedAttrs.push_back(numbering);

  // Check for OpaqueAttr, which is a dialect-specific attribute that didn't
  // have a registered dialect when it got created. We don't want to encode this
  // as the builtin OpaqueAttr, we want to encode it as if the dialect was
  // actually loaded.
  if (OpaqueAttr opaqueAttr = dyn_cast<OpaqueAttr>(attr)) {
    numbering->dialect = &numberDialect(opaqueAttr.getDialectNamespace());
    return;
  }
  numbering->dialect = &numberDialect(&attr.getDialect());

  // If this attribute will be emitted using the bytecode format, perform a
  // dummy writing to number any nested components.
  // TODO: We don't allow custom encodings for mutable attributes right now.
  if (!attr.hasTrait<AttributeTrait::IsMutable>()) {
    // Try overriding emission with callbacks.
    for (const auto &callback : config.getAttributeWriterCallbacks()) {
      NumberingDialectWriter writer(*this);
      // The client has the ability to override the group name through the
      // callback.
      std::optional<StringRef> groupNameOverride;
      if (succeeded(callback->write(attr, groupNameOverride, writer))) {
        if (groupNameOverride.has_value())
          numbering->dialect = &numberDialect(*groupNameOverride);
        return;
      }
    }

    if (const auto *interface = numbering->dialect->interface) {
      NumberingDialectWriter writer(*this);
      if (succeeded(interface->writeAttribute(attr, writer)))
        return;
    }
  }
  // If this attribute will be emitted using the fallback, number the nested
  // dialect resources. We don't number everything (e.g. no nested
  // attributes/types), because we don't want to encode things we won't decode
  // (the textual format can't really share much).
  AsmState tempState(attr.getContext());
  llvm::raw_null_ostream dummyOS;
  attr.print(dummyOS, tempState);

  // Number the used dialect resources.
  for (const auto &it : tempState.getDialectResources())
    number(it.getFirst(), it.getSecond().getArrayRef());
}

void IRNumberingState::number(Block &block) {
  // Number the arguments of the block.
  for (BlockArgument arg : block.getArguments()) {
    valueIDs.try_emplace(arg, nextValueID++);
    number(arg.getLoc());
    number(arg.getType());
  }

  // Number the operations in this block.
  unsigned &numOps = blockOperationCounts[&block];
  for (Operation &op : block) {
    number(op);
    ++numOps;
  }
}

auto IRNumberingState::numberDialect(Dialect *dialect) -> DialectNumbering & {
  DialectNumbering *&numbering = registeredDialects[dialect];
  if (!numbering) {
    numbering = &numberDialect(dialect->getNamespace());
    numbering->interface = dyn_cast<BytecodeDialectInterface>(dialect);
    numbering->asmInterface = dyn_cast<OpAsmDialectInterface>(dialect);
  }
  return *numbering;
}

auto IRNumberingState::numberDialect(StringRef dialect) -> DialectNumbering & {
  DialectNumbering *&numbering = dialects[dialect];
  if (!numbering) {
    numbering = new (dialectAllocator.Allocate())
        DialectNumbering(dialect, dialects.size() - 1);
  }
  return *numbering;
}

void IRNumberingState::number(Region &region) {
  if (region.empty())
    return;
  size_t firstValueID = nextValueID;

  // Number the blocks within this region.
  size_t blockCount = 0;
  for (auto it : llvm::enumerate(region)) {
    blockIDs.try_emplace(&it.value(), it.index());
    number(it.value());
    ++blockCount;
  }

  // Remember the number of blocks and values in this region.
  regionBlockValueCounts.try_emplace(&region, blockCount,
                                     nextValueID - firstValueID);
}

void IRNumberingState::number(Operation &op) {
  // Number the components of an operation that won't be numbered elsewhere
  // (e.g. we don't number operands, regions, or successors here).
  number(op.getName());
  for (OpResult result : op.getResults()) {
    valueIDs.try_emplace(result, nextValueID++);
    number(result.getType());
  }

  // Only number the operation's dictionary if it isn't empty.
  DictionaryAttr dictAttr = op.getDiscardableAttrDictionary();
  // Prior to version 5 we need to number also the merged dictionnary
  // containing both the inherent and discardable attribute.
  if (config.getDesiredBytecodeVersion() < 5)
    dictAttr = op.getAttrDictionary();
  if (!dictAttr.empty())
    number(dictAttr);

  // Visit the operation properties (if any) to make sure referenced attributes
  // are numbered.
  if (config.getDesiredBytecodeVersion() >= 5 &&
      op.getPropertiesStorageSize()) {
    if (op.isRegistered()) {
      // Operation that have properties *must* implement this interface.
      auto iface = cast<BytecodeOpInterface>(op);
      NumberingDialectWriter writer(*this);
      iface.writeProperties(writer);
    } else {
      // Unregistered op are storing properties as an optional attribute.
      if (Attribute prop = *op.getPropertiesStorage().as<Attribute *>())
        number(prop);
    }
  }

  number(op.getLoc());
}

void IRNumberingState::number(OperationName opName) {
  OpNameNumbering *&numbering = opNames[opName];
  if (numbering) {
    ++numbering->refCount;
    return;
  }
  DialectNumbering *dialectNumber = nullptr;
  if (Dialect *dialect = opName.getDialect())
    dialectNumber = &numberDialect(dialect);
  else
    dialectNumber = &numberDialect(opName.getDialectNamespace());

  numbering =
      new (opNameAllocator.Allocate()) OpNameNumbering(dialectNumber, opName);
  orderedOpNames.push_back(numbering);
}

void IRNumberingState::number(Type type) {
  auto it = types.insert({type, nullptr});
  if (!it.second) {
    ++it.first->second->refCount;
    return;
  }
  auto *numbering = new (typeAllocator.Allocate()) TypeNumbering(type);
  it.first->second = numbering;
  orderedTypes.push_back(numbering);

  // Check for OpaqueType, which is a dialect-specific type that didn't have a
  // registered dialect when it got created. We don't want to encode this as the
  // builtin OpaqueType, we want to encode it as if the dialect was actually
  // loaded.
  if (OpaqueType opaqueType = dyn_cast<OpaqueType>(type)) {
    numbering->dialect = &numberDialect(opaqueType.getDialectNamespace());
    return;
  }
  numbering->dialect = &numberDialect(&type.getDialect());

  // If this type will be emitted using the bytecode format, perform a dummy
  // writing to number any nested components.
  // TODO: We don't allow custom encodings for mutable types right now.
  if (!type.hasTrait<TypeTrait::IsMutable>()) {
    // Try overriding emission with callbacks.
    for (const auto &callback : config.getTypeWriterCallbacks()) {
      NumberingDialectWriter writer(*this);
      // The client has the ability to override the group name through the
      // callback.
      std::optional<StringRef> groupNameOverride;
      if (succeeded(callback->write(type, groupNameOverride, writer))) {
        if (groupNameOverride.has_value())
          numbering->dialect = &numberDialect(*groupNameOverride);
        return;
      }
    }

    // If this attribute will be emitted using the bytecode format, perform a
    // dummy writing to number any nested components.
    if (const auto *interface = numbering->dialect->interface) {
      NumberingDialectWriter writer(*this);
      if (succeeded(interface->writeType(type, writer)))
        return;
    }
  }
  // If this type will be emitted using the fallback, number the nested dialect
  // resources. We don't number everything (e.g. no nested attributes/types),
  // because we don't want to encode things we won't decode (the textual format
  // can't really share much).
  AsmState tempState(type.getContext());
  llvm::raw_null_ostream dummyOS;
  type.print(dummyOS, tempState);

  // Number the used dialect resources.
  for (const auto &it : tempState.getDialectResources())
    number(it.getFirst(), it.getSecond().getArrayRef());
}

void IRNumberingState::number(Dialect *dialect,
                              ArrayRef<AsmDialectResourceHandle> resources) {
  DialectNumbering &dialectNumber = numberDialect(dialect);
  assert(
      dialectNumber.asmInterface &&
      "expected dialect owning a resource to implement OpAsmDialectInterface");

  for (const auto &resource : resources) {
    // Check if this is a newly seen resource.
    if (!dialectNumber.resources.insert(resource))
      return;

    auto *numbering =
        new (resourceAllocator.Allocate()) DialectResourceNumbering(
            dialectNumber.asmInterface->getResourceKey(resource));
    dialectNumber.resourceMap.insert({numbering->key, numbering});
    dialectResources.try_emplace(resource, numbering);
  }
}

int64_t IRNumberingState::getDesiredBytecodeVersion() const {
  return config.getDesiredBytecodeVersion();
}

namespace {
/// A dummy resource builder used to number dialect resources.
struct NumberingResourceBuilder : public AsmResourceBuilder {
  NumberingResourceBuilder(DialectNumbering *dialect, unsigned &nextResourceID)
      : dialect(dialect), nextResourceID(nextResourceID) {}
  ~NumberingResourceBuilder() override = default;

  void buildBlob(StringRef key, ArrayRef<char>, uint32_t) final {
    numberEntry(key);
  }
  void buildBool(StringRef key, bool) final { numberEntry(key); }
  void buildString(StringRef key, StringRef) final {
    // TODO: We could pre-number the value string here as well.
    numberEntry(key);
  }

  /// Number the dialect entry for the given key.
  void numberEntry(StringRef key) {
    // TODO: We could pre-number resource key strings here as well.

    auto *it = dialect->resourceMap.find(key);
    if (it != dialect->resourceMap.end()) {
      it->second->number = nextResourceID++;
      it->second->isDeclaration = false;
    }
  }

  DialectNumbering *dialect;
  unsigned &nextResourceID;
};
} // namespace

void IRNumberingState::finalizeDialectResourceNumberings(Operation *rootOp) {
  unsigned nextResourceID = 0;
  for (DialectNumbering &dialect : getDialects()) {
    if (!dialect.asmInterface)
      continue;
    NumberingResourceBuilder entryBuilder(&dialect, nextResourceID);
    dialect.asmInterface->buildResources(rootOp, dialect.resources,
                                         entryBuilder);

    // Number any resources that weren't added by the dialect. This can happen
    // if there was no backing data to the resource, but we still want these
    // resource references to roundtrip, so we number them and indicate that the
    // data is missing.
    for (const auto &it : dialect.resourceMap)
      if (it.second->isDeclaration)
        it.second->number = nextResourceID++;
  }
}
