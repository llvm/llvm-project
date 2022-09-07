//===- IRNumbering.cpp - MLIR Bytecode IR numbering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRNumbering.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

using namespace mlir;
using namespace mlir::bytecode::detail;

//===----------------------------------------------------------------------===//
// NumberingDialectWriter
//===----------------------------------------------------------------------===//

struct IRNumberingState::NumberingDialectWriter : public DialectBytecodeWriter {
  NumberingDialectWriter(IRNumberingState &state) : state(state) {}

  void writeAttribute(Attribute attr) override { state.number(attr); }
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
  for (auto &entry : llvm::enumerate(range))
    entry.value()->number = entry.index();
}

IRNumberingState::IRNumberingState(Operation *op) {
  // Number the root operation.
  number(*op);

  // Push all of the regions of the root operation onto the worklist.
  SmallVector<std::pair<Region *, unsigned>, 8> numberContext;
  for (Region &region : op->getRegions())
    numberContext.emplace_back(&region, nextValueID);

  // Iteratively process each of the nested regions.
  while (!numberContext.empty()) {
    Region *region;
    std::tie(region, nextValueID) = numberContext.pop_back_val();
    number(*region);

    // Traverse into nested regions.
    for (Operation &op : region->getOps()) {
      // Isolated regions don't share value numbers with their parent, so we can
      // start numbering these regions at zero.
      unsigned opFirstValueID =
          op.hasTrait<OpTrait::IsIsolatedFromAbove>() ? 0 : nextValueID;
      for (Region &region : op.getRegions())
        numberContext.emplace_back(&region, opFirstValueID);
    }
  }

  // Number each of the dialects. For now this is just in the order they were
  // found, given that the number of dialects on average is small enough to fit
  // within a singly byte (128). If we ever have real world use cases that have
  // a huge number of dialects, this could be made more intelligent.
  for (auto &it : llvm::enumerate(dialects))
    it.value().second->number = it.index();

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
  groupByDialectPerByte(llvm::makeMutableArrayRef(orderedAttrs));
  groupByDialectPerByte(llvm::makeMutableArrayRef(orderedOpNames));
  groupByDialectPerByte(llvm::makeMutableArrayRef(orderedTypes));

  // Finalize the numbering of the dialect resources.
  finalizeDialectResourceNumberings(op);
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
  if (OpaqueAttr opaqueAttr = attr.dyn_cast<OpaqueAttr>()) {
    numbering->dialect = &numberDialect(opaqueAttr.getDialectNamespace());
    return;
  }
  numbering->dialect = &numberDialect(&attr.getDialect());

  // If this attribute will be emitted using the bytecode format, perform a
  // dummy writing to number any nested components.
  if (const auto *interface = numbering->dialect->interface) {
    // TODO: We don't allow custom encodings for mutable attributes right now.
    if (!attr.hasTrait<AttributeTrait::IsMutable>()) {
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
  for (auto &it : llvm::enumerate(region)) {
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
  DictionaryAttr dictAttr = op.getAttrDictionary();
  if (!dictAttr.empty())
    number(dictAttr);

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
  if (OpaqueType opaqueType = type.dyn_cast<OpaqueType>()) {
    numbering->dialect = &numberDialect(opaqueType.getDialectNamespace());
    return;
  }
  numbering->dialect = &numberDialect(&type.getDialect());

  // If this type will be emitted using the bytecode format, perform a dummy
  // writing to number any nested components.
  if (const auto *interface = numbering->dialect->interface) {
    // TODO: We don't allow custom encodings for mutable types right now.
    if (!type.hasTrait<TypeTrait::IsMutable>()) {
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

    auto it = dialect->resourceMap.find(key);
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
