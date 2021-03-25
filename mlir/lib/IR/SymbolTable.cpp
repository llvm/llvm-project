//===- SymbolTable.cpp - MLIR Symbol Table Class --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

/// Return true if the given operation is unknown and may potentially define a
/// symbol table.
static bool isPotentiallyUnknownSymbolTable(Operation *op) {
  return op->getNumRegions() == 1 && !op->getDialect();
}

/// Returns the string name of the given symbol, or None if this is not a
/// symbol.
static Optional<StringRef> getNameIfSymbol(Operation *symbol) {
  auto nameAttr =
      symbol->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  return nameAttr ? nameAttr.getValue() : Optional<StringRef>();
}
static Optional<StringRef> getNameIfSymbol(Operation *symbol,
                                           Identifier symbolAttrNameId) {
  auto nameAttr = symbol->getAttrOfType<StringAttr>(symbolAttrNameId);
  return nameAttr ? nameAttr.getValue() : Optional<StringRef>();
}

/// Computes the nested symbol reference attribute for the symbol 'symbolName'
/// that are usable within the symbol table operations from 'symbol' as far up
/// to the given operation 'within', where 'within' is an ancestor of 'symbol'.
/// Returns success if all references up to 'within' could be computed.
static LogicalResult
collectValidReferencesFor(Operation *symbol, StringRef symbolName,
                          Operation *within,
                          SmallVectorImpl<SymbolRefAttr> &results) {
  assert(within->isAncestor(symbol) && "expected 'within' to be an ancestor");
  MLIRContext *ctx = symbol->getContext();

  auto leafRef = FlatSymbolRefAttr::get(ctx, symbolName);
  results.push_back(leafRef);

  // Early exit for when 'within' is the parent of 'symbol'.
  Operation *symbolTableOp = symbol->getParentOp();
  if (within == symbolTableOp)
    return success();

  // Collect references until 'symbolTableOp' reaches 'within'.
  SmallVector<FlatSymbolRefAttr, 1> nestedRefs(1, leafRef);
  Identifier symbolNameId =
      Identifier::get(SymbolTable::getSymbolAttrName(), ctx);
  do {
    // Each parent of 'symbol' should define a symbol table.
    if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>())
      return failure();
    // Each parent of 'symbol' should also be a symbol.
    Optional<StringRef> symbolTableName =
        getNameIfSymbol(symbolTableOp, symbolNameId);
    if (!symbolTableName)
      return failure();
    results.push_back(SymbolRefAttr::get(ctx, *symbolTableName, nestedRefs));

    symbolTableOp = symbolTableOp->getParentOp();
    if (symbolTableOp == within)
      break;
    nestedRefs.insert(nestedRefs.begin(),
                      FlatSymbolRefAttr::get(ctx, *symbolTableName));
  } while (true);
  return success();
}

/// Walk all of the operations within the given set of regions, without
/// traversing into any nested symbol tables. Stops walking if the result of the
/// callback is anything other than `WalkResult::advance`.
static Optional<WalkResult>
walkSymbolTable(MutableArrayRef<Region> regions,
                function_ref<Optional<WalkResult>(Operation *)> callback) {
  SmallVector<Region *, 1> worklist(llvm::make_pointer_range(regions));
  while (!worklist.empty()) {
    for (Operation &op : worklist.pop_back_val()->getOps()) {
      Optional<WalkResult> result = callback(&op);
      if (result != WalkResult::advance())
        return result;

      // If this op defines a new symbol table scope, we can't traverse. Any
      // symbol references nested within 'op' are different semantically.
      if (!op.hasTrait<OpTrait::SymbolTable>()) {
        for (Region &region : op.getRegions())
          worklist.push_back(&region);
      }
    }
  }
  return WalkResult::advance();
}

//===----------------------------------------------------------------------===//
// SymbolTable
//===----------------------------------------------------------------------===//

/// Build a symbol table with the symbols within the given operation.
SymbolTable::SymbolTable(Operation *symbolTableOp)
    : symbolTableOp(symbolTableOp) {
  assert(symbolTableOp->hasTrait<OpTrait::SymbolTable>() &&
         "expected operation to have SymbolTable trait");
  assert(symbolTableOp->getNumRegions() == 1 &&
         "expected operation to have a single region");
  assert(llvm::hasSingleElement(symbolTableOp->getRegion(0)) &&
         "expected operation to have a single block");

  Identifier symbolNameId = Identifier::get(SymbolTable::getSymbolAttrName(),
                                            symbolTableOp->getContext());
  for (auto &op : symbolTableOp->getRegion(0).front()) {
    Optional<StringRef> name = getNameIfSymbol(&op, symbolNameId);
    if (!name)
      continue;

    auto inserted = symbolTable.insert({*name, &op});
    (void)inserted;
    assert(inserted.second &&
           "expected region to contain uniquely named symbol operations");
  }
}

/// Look up a symbol with the specified name, returning null if no such name
/// exists. Names never include the @ on them.
Operation *SymbolTable::lookup(StringRef name) const {
  return symbolTable.lookup(name);
}

/// Erase the given symbol from the table.
void SymbolTable::erase(Operation *symbol) {
  Optional<StringRef> name = getNameIfSymbol(symbol);
  assert(name && "expected valid 'name' attribute");
  assert(symbol->getParentOp() == symbolTableOp &&
         "expected this operation to be inside of the operation with this "
         "SymbolTable");

  auto it = symbolTable.find(*name);
  if (it != symbolTable.end() && it->second == symbol) {
    symbolTable.erase(it);
    symbol->erase();
  }
}

// TODO: Consider if this should be renamed to something like insertOrUpdate
/// Insert a new symbol into the table and associated operation if not already
/// there and rename it as necessary to avoid collisions.
void SymbolTable::insert(Operation *symbol, Block::iterator insertPt) {
  // The symbol cannot be the child of another op and must be the child of the
  // symbolTableOp after this.
  //
  // TODO: consider if SymbolTable's constructor should behave the same.
  if (!symbol->getParentOp()) {
    auto &body = symbolTableOp->getRegion(0).front();
    if (insertPt == Block::iterator()) {
      insertPt = Block::iterator(body.end());
    } else {
      assert((insertPt == body.end() ||
              insertPt->getParentOp() == symbolTableOp) &&
             "expected insertPt to be in the associated module operation");
    }
    // Insert before the terminator, if any.
    if (insertPt == Block::iterator(body.end()) && !body.empty() &&
        std::prev(body.end())->hasTrait<OpTrait::IsTerminator>())
      insertPt = std::prev(body.end());

    body.getOperations().insert(insertPt, symbol);
  }
  assert(symbol->getParentOp() == symbolTableOp &&
         "symbol is already inserted in another op");

  // Add this symbol to the symbol table, uniquing the name if a conflict is
  // detected.
  StringRef name = getSymbolName(symbol);
  if (symbolTable.insert({name, symbol}).second)
    return;
  // If the symbol was already in the table, also return.
  if (symbolTable.lookup(name) == symbol)
    return;
  // If a conflict was detected, then the symbol will not have been added to
  // the symbol table. Try suffixes until we get to a unique name that works.
  SmallString<128> nameBuffer(name);
  unsigned originalLength = nameBuffer.size();

  // Iteratively try suffixes until we find one that isn't used.
  do {
    nameBuffer.resize(originalLength);
    nameBuffer += '_';
    nameBuffer += std::to_string(uniquingCounter++);
  } while (!symbolTable.insert({nameBuffer, symbol}).second);
  setSymbolName(symbol, nameBuffer);
}

/// Returns the name of the given symbol operation.
StringRef SymbolTable::getSymbolName(Operation *symbol) {
  Optional<StringRef> name = getNameIfSymbol(symbol);
  assert(name && "expected valid symbol name");
  return *name;
}
/// Sets the name of the given symbol operation.
void SymbolTable::setSymbolName(Operation *symbol, StringRef name) {
  symbol->setAttr(getSymbolAttrName(),
                  StringAttr::get(symbol->getContext(), name));
}

/// Returns the visibility of the given symbol operation.
SymbolTable::Visibility SymbolTable::getSymbolVisibility(Operation *symbol) {
  // If the attribute doesn't exist, assume public.
  StringAttr vis = symbol->getAttrOfType<StringAttr>(getVisibilityAttrName());
  if (!vis)
    return Visibility::Public;

  // Otherwise, switch on the string value.
  return StringSwitch<Visibility>(vis.getValue())
      .Case("private", Visibility::Private)
      .Case("nested", Visibility::Nested)
      .Case("public", Visibility::Public);
}
/// Sets the visibility of the given symbol operation.
void SymbolTable::setSymbolVisibility(Operation *symbol, Visibility vis) {
  MLIRContext *ctx = symbol->getContext();

  // If the visibility is public, just drop the attribute as this is the
  // default.
  if (vis == Visibility::Public) {
    symbol->removeAttr(Identifier::get(getVisibilityAttrName(), ctx));
    return;
  }

  // Otherwise, update the attribute.
  assert((vis == Visibility::Private || vis == Visibility::Nested) &&
         "unknown symbol visibility kind");

  StringRef visName = vis == Visibility::Private ? "private" : "nested";
  symbol->setAttr(getVisibilityAttrName(), StringAttr::get(ctx, visName));
}

/// Returns the nearest symbol table from a given operation `from`. Returns
/// nullptr if no valid parent symbol table could be found.
Operation *SymbolTable::getNearestSymbolTable(Operation *from) {
  assert(from && "expected valid operation");
  if (isPotentiallyUnknownSymbolTable(from))
    return nullptr;

  while (!from->hasTrait<OpTrait::SymbolTable>()) {
    from = from->getParentOp();

    // Check that this is a valid op and isn't an unknown symbol table.
    if (!from || isPotentiallyUnknownSymbolTable(from))
      return nullptr;
  }
  return from;
}

/// Walks all symbol table operations nested within, and including, `op`. For
/// each symbol table operation, the provided callback is invoked with the op
/// and a boolean signifying if the symbols within that symbol table can be
/// treated as if all uses are visible. `allSymUsesVisible` identifies whether
/// all of the symbol uses of symbols within `op` are visible.
void SymbolTable::walkSymbolTables(
    Operation *op, bool allSymUsesVisible,
    function_ref<void(Operation *, bool)> callback) {
  bool isSymbolTable = op->hasTrait<OpTrait::SymbolTable>();
  if (isSymbolTable) {
    SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op);
    allSymUsesVisible |= !symbol || symbol.isPrivate();
  } else {
    // Otherwise if 'op' is not a symbol table, any nested symbols are
    // guaranteed to be hidden.
    allSymUsesVisible = true;
  }

  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nestedOp : block)
        walkSymbolTables(&nestedOp, allSymUsesVisible, callback);

  // If 'op' had the symbol table trait, visit it after any nested symbol
  // tables.
  if (isSymbolTable)
    callback(op, allSymUsesVisible);
}

/// Returns the operation registered with the given symbol name with the
/// regions of 'symbolTableOp'. 'symbolTableOp' is required to be an operation
/// with the 'OpTrait::SymbolTable' trait. Returns nullptr if no valid symbol
/// was found.
Operation *SymbolTable::lookupSymbolIn(Operation *symbolTableOp,
                                       StringRef symbol) {
  assert(symbolTableOp->hasTrait<OpTrait::SymbolTable>());
  Region &region = symbolTableOp->getRegion(0);
  if (region.empty())
    return nullptr;

  // Look for a symbol with the given name.
  Identifier symbolNameId = Identifier::get(SymbolTable::getSymbolAttrName(),
                                            symbolTableOp->getContext());
  for (auto &op : region.front())
    if (getNameIfSymbol(&op, symbolNameId) == symbol)
      return &op;
  return nullptr;
}
Operation *SymbolTable::lookupSymbolIn(Operation *symbolTableOp,
                                       SymbolRefAttr symbol) {
  SmallVector<Operation *, 4> resolvedSymbols;
  if (failed(lookupSymbolIn(symbolTableOp, symbol, resolvedSymbols)))
    return nullptr;
  return resolvedSymbols.back();
}

/// Internal implementation of `lookupSymbolIn` that allows for specialized
/// implementations of the lookup function.
static LogicalResult lookupSymbolInImpl(
    Operation *symbolTableOp, SymbolRefAttr symbol,
    SmallVectorImpl<Operation *> &symbols,
    function_ref<Operation *(Operation *, StringRef)> lookupSymbolFn) {
  assert(symbolTableOp->hasTrait<OpTrait::SymbolTable>());

  // Lookup the root reference for this symbol.
  symbolTableOp = lookupSymbolFn(symbolTableOp, symbol.getRootReference());
  if (!symbolTableOp)
    return failure();
  symbols.push_back(symbolTableOp);

  // If there are no nested references, just return the root symbol directly.
  ArrayRef<FlatSymbolRefAttr> nestedRefs = symbol.getNestedReferences();
  if (nestedRefs.empty())
    return success();

  // Verify that the root is also a symbol table.
  if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>())
    return failure();

  // Otherwise, lookup each of the nested non-leaf references and ensure that
  // each corresponds to a valid symbol table.
  for (FlatSymbolRefAttr ref : nestedRefs.drop_back()) {
    symbolTableOp = lookupSymbolFn(symbolTableOp, ref.getValue());
    if (!symbolTableOp || !symbolTableOp->hasTrait<OpTrait::SymbolTable>())
      return failure();
    symbols.push_back(symbolTableOp);
  }
  symbols.push_back(lookupSymbolFn(symbolTableOp, symbol.getLeafReference()));
  return success(symbols.back());
}

LogicalResult
SymbolTable::lookupSymbolIn(Operation *symbolTableOp, SymbolRefAttr symbol,
                            SmallVectorImpl<Operation *> &symbols) {
  auto lookupFn = [](Operation *symbolTableOp, StringRef symbol) {
    return lookupSymbolIn(symbolTableOp, symbol);
  };
  return lookupSymbolInImpl(symbolTableOp, symbol, symbols, lookupFn);
}

/// Returns the operation registered with the given symbol name within the
/// closes parent operation with the 'OpTrait::SymbolTable' trait. Returns
/// nullptr if no valid symbol was found.
Operation *SymbolTable::lookupNearestSymbolFrom(Operation *from,
                                                StringRef symbol) {
  Operation *symbolTableOp = getNearestSymbolTable(from);
  return symbolTableOp ? lookupSymbolIn(symbolTableOp, symbol) : nullptr;
}
Operation *SymbolTable::lookupNearestSymbolFrom(Operation *from,
                                                SymbolRefAttr symbol) {
  Operation *symbolTableOp = getNearestSymbolTable(from);
  return symbolTableOp ? lookupSymbolIn(symbolTableOp, symbol) : nullptr;
}

//===----------------------------------------------------------------------===//
// SymbolTable Trait Types
//===----------------------------------------------------------------------===//

LogicalResult detail::verifySymbolTable(Operation *op) {
  if (op->getNumRegions() != 1)
    return op->emitOpError()
           << "Operations with a 'SymbolTable' must have exactly one region";
  if (!llvm::hasSingleElement(op->getRegion(0)))
    return op->emitOpError()
           << "Operations with a 'SymbolTable' must have exactly one block";

  // Check that all symbols are uniquely named within child regions.
  DenseMap<Attribute, Location> nameToOrigLoc;
  for (auto &block : op->getRegion(0)) {
    for (auto &op : block) {
      // Check for a symbol name attribute.
      auto nameAttr =
          op.getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
      if (!nameAttr)
        continue;

      // Try to insert this symbol into the table.
      auto it = nameToOrigLoc.try_emplace(nameAttr, op.getLoc());
      if (!it.second)
        return op.emitError()
            .append("redefinition of symbol named '", nameAttr.getValue(), "'")
            .attachNote(it.first->second)
            .append("see existing symbol definition here");
    }
  }

  // Verify any nested symbol user operations.
  SymbolTableCollection symbolTable;
  auto verifySymbolUserFn = [&](Operation *op) -> Optional<WalkResult> {
    if (SymbolUserOpInterface user = dyn_cast<SymbolUserOpInterface>(op))
      return WalkResult(user.verifySymbolUses(symbolTable));
    return WalkResult::advance();
  };

  Optional<WalkResult> result =
      walkSymbolTable(op->getRegions(), verifySymbolUserFn);
  return success(result && !result->wasInterrupted());
}

LogicalResult detail::verifySymbol(Operation *op) {
  // Verify the name attribute.
  if (!op->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName()))
    return op->emitOpError() << "requires string attribute '"
                             << mlir::SymbolTable::getSymbolAttrName() << "'";

  // Verify the visibility attribute.
  if (Attribute vis = op->getAttr(mlir::SymbolTable::getVisibilityAttrName())) {
    StringAttr visStrAttr = vis.dyn_cast<StringAttr>();
    if (!visStrAttr)
      return op->emitOpError() << "requires visibility attribute '"
                               << mlir::SymbolTable::getVisibilityAttrName()
                               << "' to be a string attribute, but got " << vis;

    if (!llvm::is_contained(ArrayRef<StringRef>{"public", "private", "nested"},
                            visStrAttr.getValue()))
      return op->emitOpError()
             << "visibility expected to be one of [\"public\", \"private\", "
                "\"nested\"], but got "
             << visStrAttr;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Symbol Use Lists
//===----------------------------------------------------------------------===//

/// Walk all of the symbol references within the given operation, invoking the
/// provided callback for each found use. The callbacks takes as arguments: the
/// use of the symbol, and the nested access chain to the attribute within the
/// operation dictionary. An access chain is a set of indices into nested
/// container attributes. For example, a symbol use in an attribute dictionary
/// that looks like the following:
///
///    {use = [{other_attr, @symbol}]}
///
/// May have the following access chain:
///
///     [0, 0, 1]
///
static WalkResult walkSymbolRefs(
    Operation *op,
    function_ref<WalkResult(SymbolTable::SymbolUse, ArrayRef<int>)> callback) {
  // Check to see if the operation has any attributes.
  DictionaryAttr attrDict = op->getAttrDictionary();
  if (attrDict.empty())
    return WalkResult::advance();

  // A worklist of a container attribute and the current index into the held
  // attribute list.
  SmallVector<Attribute, 1> attrWorklist(1, attrDict);
  SmallVector<int, 1> curAccessChain(1, /*Value=*/-1);

  // Process the symbol references within the given nested attribute range.
  auto processAttrs = [&](int &index, auto attrRange) -> WalkResult {
    for (Attribute attr : llvm::drop_begin(attrRange, index)) {
      /// Check for a nested container attribute, these will also need to be
      /// walked.
      if (attr.isa<ArrayAttr, DictionaryAttr>()) {
        attrWorklist.push_back(attr);
        curAccessChain.push_back(-1);
        return WalkResult::advance();
      }

      // Invoke the provided callback if we find a symbol use and check for a
      // requested interrupt.
      if (auto symbolRef = attr.dyn_cast<SymbolRefAttr>())
        if (callback({op, symbolRef}, curAccessChain).wasInterrupted())
          return WalkResult::interrupt();

      // Make sure to keep the index counter in sync.
      ++index;
    }

    // Pop this container attribute from the worklist.
    attrWorklist.pop_back();
    curAccessChain.pop_back();
    return WalkResult::advance();
  };

  WalkResult result = WalkResult::advance();
  do {
    Attribute attr = attrWorklist.back();
    int &index = curAccessChain.back();
    ++index;

    // Process the given attribute, which is guaranteed to be a container.
    if (auto dict = attr.dyn_cast<DictionaryAttr>())
      result = processAttrs(index, make_second_range(dict.getValue()));
    else
      result = processAttrs(index, attr.cast<ArrayAttr>().getValue());
  } while (!attrWorklist.empty() && !result.wasInterrupted());
  return result;
}

/// Walk all of the uses, for any symbol, that are nested within the given
/// regions, invoking the provided callback for each. This does not traverse
/// into any nested symbol tables.
static Optional<WalkResult> walkSymbolUses(
    MutableArrayRef<Region> regions,
    function_ref<WalkResult(SymbolTable::SymbolUse, ArrayRef<int>)> callback) {
  return walkSymbolTable(regions, [&](Operation *op) -> Optional<WalkResult> {
    // Check that this isn't a potentially unknown symbol table.
    if (isPotentiallyUnknownSymbolTable(op))
      return llvm::None;

    return walkSymbolRefs(op, callback);
  });
}
/// Walk all of the uses, for any symbol, that are nested within the given
/// operation 'from', invoking the provided callback for each. This does not
/// traverse into any nested symbol tables.
static Optional<WalkResult> walkSymbolUses(
    Operation *from,
    function_ref<WalkResult(SymbolTable::SymbolUse, ArrayRef<int>)> callback) {
  // If this operation has regions, and it, as well as its dialect, isn't
  // registered then conservatively fail. The operation may define a
  // symbol table, so we can't opaquely know if we should traverse to find
  // nested uses.
  if (isPotentiallyUnknownSymbolTable(from))
    return llvm::None;

  // Walk the uses on this operation.
  if (walkSymbolRefs(from, callback).wasInterrupted())
    return WalkResult::interrupt();

  // Only recurse if this operation is not a symbol table. A symbol table
  // defines a new scope, so we can't walk the attributes from within the symbol
  // table op.
  if (!from->hasTrait<OpTrait::SymbolTable>())
    return walkSymbolUses(from->getRegions(), callback);
  return WalkResult::advance();
}

namespace {
/// This class represents a single symbol scope. A symbol scope represents the
/// set of operations nested within a symbol table that may reference symbols
/// within that table. A symbol scope does not contain the symbol table
/// operation itself, just its contained operations. A scope ends at leaf
/// operations or another symbol table operation.
struct SymbolScope {
  /// Walk the symbol uses within this scope, invoking the given callback.
  /// This variant is used when the callback type matches that expected by
  /// 'walkSymbolUses'.
  template <typename CallbackT,
            typename std::enable_if_t<!std::is_same<
                typename llvm::function_traits<CallbackT>::result_t,
                void>::value> * = nullptr>
  Optional<WalkResult> walk(CallbackT cback) {
    if (Region *region = limit.dyn_cast<Region *>())
      return walkSymbolUses(*region, cback);
    return walkSymbolUses(limit.get<Operation *>(), cback);
  }
  /// This variant is used when the callback type matches a stripped down type:
  /// void(SymbolTable::SymbolUse use)
  template <typename CallbackT,
            typename std::enable_if_t<std::is_same<
                typename llvm::function_traits<CallbackT>::result_t,
                void>::value> * = nullptr>
  Optional<WalkResult> walk(CallbackT cback) {
    return walk([=](SymbolTable::SymbolUse use, ArrayRef<int>) {
      return cback(use), WalkResult::advance();
    });
  }

  /// The representation of the symbol within this scope.
  SymbolRefAttr symbol;

  /// The IR unit representing this scope.
  llvm::PointerUnion<Operation *, Region *> limit;
};
} // end anonymous namespace

/// Collect all of the symbol scopes from 'symbol' to (inclusive) 'limit'.
static SmallVector<SymbolScope, 2> collectSymbolScopes(Operation *symbol,
                                                       Operation *limit) {
  StringRef symName = SymbolTable::getSymbolName(symbol);
  assert(!symbol->hasTrait<OpTrait::SymbolTable>() || symbol != limit);

  // Compute the ancestors of 'limit'.
  llvm::SetVector<Operation *, SmallVector<Operation *, 4>,
                  SmallPtrSet<Operation *, 4>>
      limitAncestors;
  Operation *limitAncestor = limit;
  do {
    // Check to see if 'symbol' is an ancestor of 'limit'.
    if (limitAncestor == symbol) {
      // Check that the nearest symbol table is 'symbol's parent. SymbolRefAttr
      // doesn't support parent references.
      if (SymbolTable::getNearestSymbolTable(limit->getParentOp()) ==
          symbol->getParentOp())
        return {{SymbolRefAttr::get(symbol->getContext(), symName), limit}};
      return {};
    }

    limitAncestors.insert(limitAncestor);
  } while ((limitAncestor = limitAncestor->getParentOp()));

  // Try to find the first ancestor of 'symbol' that is an ancestor of 'limit'.
  Operation *commonAncestor = symbol->getParentOp();
  do {
    if (limitAncestors.count(commonAncestor))
      break;
  } while ((commonAncestor = commonAncestor->getParentOp()));
  assert(commonAncestor && "'limit' and 'symbol' have no common ancestor");

  // Compute the set of valid nested references for 'symbol' as far up to the
  // common ancestor as possible.
  SmallVector<SymbolRefAttr, 2> references;
  bool collectedAllReferences = succeeded(
      collectValidReferencesFor(symbol, symName, commonAncestor, references));

  // Handle the case where the common ancestor is 'limit'.
  if (commonAncestor == limit) {
    SmallVector<SymbolScope, 2> scopes;

    // Walk each of the ancestors of 'symbol', calling the compute function for
    // each one.
    Operation *limitIt = symbol->getParentOp();
    for (size_t i = 0, e = references.size(); i != e;
         ++i, limitIt = limitIt->getParentOp()) {
      assert(limitIt->hasTrait<OpTrait::SymbolTable>());
      scopes.push_back({references[i], &limitIt->getRegion(0)});
    }
    return scopes;
  }

  // Otherwise, we just need the symbol reference for 'symbol' that will be
  // used within 'limit'. This is the last reference in the list we computed
  // above if we were able to collect all references.
  if (!collectedAllReferences)
    return {};
  return {{references.back(), limit}};
}
static SmallVector<SymbolScope, 2> collectSymbolScopes(Operation *symbol,
                                                       Region *limit) {
  auto scopes = collectSymbolScopes(symbol, limit->getParentOp());

  // If we collected some scopes to walk, make sure to constrain the one for
  // limit to the specific region requested.
  if (!scopes.empty())
    scopes.back().limit = limit;
  return scopes;
}
template <typename IRUnit>
static SmallVector<SymbolScope, 1> collectSymbolScopes(StringRef symbol,
                                                       IRUnit *limit) {
  return {{SymbolRefAttr::get(limit->getContext(), symbol), limit}};
}

/// Returns true if the given reference 'SubRef' is a sub reference of the
/// reference 'ref', i.e. 'ref' is a further qualified reference.
static bool isReferencePrefixOf(SymbolRefAttr subRef, SymbolRefAttr ref) {
  if (ref == subRef)
    return true;

  // If the references are not pointer equal, check to see if `subRef` is a
  // prefix of `ref`.
  if (ref.isa<FlatSymbolRefAttr>() ||
      ref.getRootReference() != subRef.getRootReference())
    return false;

  auto refLeafs = ref.getNestedReferences();
  auto subRefLeafs = subRef.getNestedReferences();
  return subRefLeafs.size() < refLeafs.size() &&
         subRefLeafs == refLeafs.take_front(subRefLeafs.size());
}

//===----------------------------------------------------------------------===//
// SymbolTable::getSymbolUses

/// The implementation of SymbolTable::getSymbolUses below.
template <typename FromT>
static Optional<SymbolTable::UseRange> getSymbolUsesImpl(FromT from) {
  std::vector<SymbolTable::SymbolUse> uses;
  auto walkFn = [&](SymbolTable::SymbolUse symbolUse, ArrayRef<int>) {
    uses.push_back(symbolUse);
    return WalkResult::advance();
  };
  auto result = walkSymbolUses(from, walkFn);
  return result ? Optional<SymbolTable::UseRange>(std::move(uses)) : llvm::None;
}

/// Get an iterator range for all of the uses, for any symbol, that are nested
/// within the given operation 'from'. This does not traverse into any nested
/// symbol tables, and will also only return uses on 'from' if it does not
/// also define a symbol table. This is because we treat the region as the
/// boundary of the symbol table, and not the op itself. This function returns
/// None if there are any unknown operations that may potentially be symbol
/// tables.
auto SymbolTable::getSymbolUses(Operation *from) -> Optional<UseRange> {
  return getSymbolUsesImpl(from);
}
auto SymbolTable::getSymbolUses(Region *from) -> Optional<UseRange> {
  return getSymbolUsesImpl(MutableArrayRef<Region>(*from));
}

//===----------------------------------------------------------------------===//
// SymbolTable::getSymbolUses

/// The implementation of SymbolTable::getSymbolUses below.
template <typename SymbolT, typename IRUnitT>
static Optional<SymbolTable::UseRange> getSymbolUsesImpl(SymbolT symbol,
                                                         IRUnitT *limit) {
  std::vector<SymbolTable::SymbolUse> uses;
  for (SymbolScope &scope : collectSymbolScopes(symbol, limit)) {
    if (!scope.walk([&](SymbolTable::SymbolUse symbolUse) {
          if (isReferencePrefixOf(scope.symbol, symbolUse.getSymbolRef()))
            uses.push_back(symbolUse);
        }))
      return llvm::None;
  }
  return SymbolTable::UseRange(std::move(uses));
}

/// Get all of the uses of the given symbol that are nested within the given
/// operation 'from', invoking the provided callback for each. This does not
/// traverse into any nested symbol tables. This function returns None if there
/// are any unknown operations that may potentially be symbol tables.
auto SymbolTable::getSymbolUses(StringRef symbol, Operation *from)
    -> Optional<UseRange> {
  return getSymbolUsesImpl(symbol, from);
}
auto SymbolTable::getSymbolUses(Operation *symbol, Operation *from)
    -> Optional<UseRange> {
  return getSymbolUsesImpl(symbol, from);
}
auto SymbolTable::getSymbolUses(StringRef symbol, Region *from)
    -> Optional<UseRange> {
  return getSymbolUsesImpl(symbol, from);
}
auto SymbolTable::getSymbolUses(Operation *symbol, Region *from)
    -> Optional<UseRange> {
  return getSymbolUsesImpl(symbol, from);
}

//===----------------------------------------------------------------------===//
// SymbolTable::symbolKnownUseEmpty

/// The implementation of SymbolTable::symbolKnownUseEmpty below.
template <typename SymbolT, typename IRUnitT>
static bool symbolKnownUseEmptyImpl(SymbolT symbol, IRUnitT *limit) {
  for (SymbolScope &scope : collectSymbolScopes(symbol, limit)) {
    // Walk all of the symbol uses looking for a reference to 'symbol'.
    if (scope.walk([&](SymbolTable::SymbolUse symbolUse, ArrayRef<int>) {
          return isReferencePrefixOf(scope.symbol, symbolUse.getSymbolRef())
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        }) != WalkResult::advance())
      return false;
  }
  return true;
}

/// Return if the given symbol is known to have no uses that are nested within
/// the given operation 'from'. This does not traverse into any nested symbol
/// tables. This function will also return false if there are any unknown
/// operations that may potentially be symbol tables.
bool SymbolTable::symbolKnownUseEmpty(StringRef symbol, Operation *from) {
  return symbolKnownUseEmptyImpl(symbol, from);
}
bool SymbolTable::symbolKnownUseEmpty(Operation *symbol, Operation *from) {
  return symbolKnownUseEmptyImpl(symbol, from);
}
bool SymbolTable::symbolKnownUseEmpty(StringRef symbol, Region *from) {
  return symbolKnownUseEmptyImpl(symbol, from);
}
bool SymbolTable::symbolKnownUseEmpty(Operation *symbol, Region *from) {
  return symbolKnownUseEmptyImpl(symbol, from);
}

//===----------------------------------------------------------------------===//
// SymbolTable::replaceAllSymbolUses

/// Rebuild the given attribute container after replacing all references to a
/// symbol with the updated attribute in 'accesses'.
static Attribute rebuildAttrAfterRAUW(
    Attribute container,
    ArrayRef<std::pair<SmallVector<int, 1>, SymbolRefAttr>> accesses,
    unsigned depth) {
  // Given a range of Attributes, update the ones referred to by the given
  // access chains to point to the new symbol attribute.
  auto updateAttrs = [&](auto &&attrRange) {
    auto attrBegin = std::begin(attrRange);
    for (unsigned i = 0, e = accesses.size(); i != e;) {
      ArrayRef<int> access = accesses[i].first;
      Attribute &attr = *std::next(attrBegin, access[depth]);

      // Check to see if this is a leaf access, i.e. a SymbolRef.
      if (access.size() == depth + 1) {
        attr = accesses[i].second;
        ++i;
        continue;
      }

      // Otherwise, this is a container. Collect all of the accesses for this
      // index and recurse. The recursion here is bounded by the size of the
      // largest access array.
      auto nestedAccesses = accesses.drop_front(i).take_while([&](auto &it) {
        ArrayRef<int> nextAccess = it.first;
        return nextAccess.size() > depth + 1 &&
               nextAccess[depth] == access[depth];
      });
      attr = rebuildAttrAfterRAUW(attr, nestedAccesses, depth + 1);

      // Skip over all of the accesses that refer to the nested container.
      i += nestedAccesses.size();
    }
  };

  if (auto dictAttr = container.dyn_cast<DictionaryAttr>()) {
    auto newAttrs = llvm::to_vector<4>(dictAttr.getValue());
    updateAttrs(make_second_range(newAttrs));
    return DictionaryAttr::get(dictAttr.getContext(), newAttrs);
  }
  auto newAttrs = llvm::to_vector<4>(container.cast<ArrayAttr>().getValue());
  updateAttrs(newAttrs);
  return ArrayAttr::get(container.getContext(), newAttrs);
}

/// Generates a new symbol reference attribute with a new leaf reference.
static SymbolRefAttr generateNewRefAttr(SymbolRefAttr oldAttr,
                                        FlatSymbolRefAttr newLeafAttr) {
  if (oldAttr.isa<FlatSymbolRefAttr>())
    return newLeafAttr;
  auto nestedRefs = llvm::to_vector<2>(oldAttr.getNestedReferences());
  nestedRefs.back() = newLeafAttr;
  return SymbolRefAttr::get(oldAttr.getContext(), oldAttr.getRootReference(),
                            nestedRefs);
}

/// The implementation of SymbolTable::replaceAllSymbolUses below.
template <typename SymbolT, typename IRUnitT>
static LogicalResult
replaceAllSymbolUsesImpl(SymbolT symbol, StringRef newSymbol, IRUnitT *limit) {
  // A collection of operations along with their new attribute dictionary.
  std::vector<std::pair<Operation *, DictionaryAttr>> updatedAttrDicts;

  // The current operation being processed.
  Operation *curOp = nullptr;

  // The set of access chains into the attribute dictionary of the current
  // operation, as well as the replacement attribute to use.
  SmallVector<std::pair<SmallVector<int, 1>, SymbolRefAttr>, 1> accessChains;

  // Generate a new attribute dictionary for the current operation by replacing
  // references to the old symbol.
  auto generateNewAttrDict = [&] {
    auto oldDict = curOp->getAttrDictionary();
    auto newDict = rebuildAttrAfterRAUW(oldDict, accessChains, /*depth=*/0);
    return newDict.cast<DictionaryAttr>();
  };

  // Generate a new attribute to replace the given attribute.
  MLIRContext *ctx = limit->getContext();
  FlatSymbolRefAttr newLeafAttr = FlatSymbolRefAttr::get(ctx, newSymbol);
  for (SymbolScope &scope : collectSymbolScopes(symbol, limit)) {
    SymbolRefAttr newAttr = generateNewRefAttr(scope.symbol, newLeafAttr);
    auto walkFn = [&](SymbolTable::SymbolUse symbolUse,
                      ArrayRef<int> accessChain) {
      SymbolRefAttr useRef = symbolUse.getSymbolRef();
      if (!isReferencePrefixOf(scope.symbol, useRef))
        return WalkResult::advance();

      // If we have a valid match, check to see if this is a proper
      // subreference. If it is, then we will need to generate a different new
      // attribute specifically for this use.
      SymbolRefAttr replacementRef = newAttr;
      if (useRef != scope.symbol) {
        if (scope.symbol.isa<FlatSymbolRefAttr>()) {
          replacementRef =
              SymbolRefAttr::get(ctx, newSymbol, useRef.getNestedReferences());
        } else {
          auto nestedRefs = llvm::to_vector<4>(useRef.getNestedReferences());
          nestedRefs[scope.symbol.getNestedReferences().size() - 1] =
              newLeafAttr;
          replacementRef =
              SymbolRefAttr::get(ctx, useRef.getRootReference(), nestedRefs);
        }
      }

      // If there was a previous operation, generate a new attribute dict
      // for it. This means that we've finished processing the current
      // operation, so generate a new dictionary for it.
      if (curOp && symbolUse.getUser() != curOp) {
        updatedAttrDicts.push_back({curOp, generateNewAttrDict()});
        accessChains.clear();
      }

      // Record this access.
      curOp = symbolUse.getUser();
      accessChains.push_back({llvm::to_vector<1>(accessChain), replacementRef});
      return WalkResult::advance();
    };
    if (!scope.walk(walkFn))
      return failure();

    // Check to see if we have a dangling op that needs to be processed.
    if (curOp) {
      updatedAttrDicts.push_back({curOp, generateNewAttrDict()});
      curOp = nullptr;
    }
  }

  // Update the attribute dictionaries as necessary.
  for (auto &it : updatedAttrDicts)
    it.first->setAttrs(it.second);
  return success();
}

/// Attempt to replace all uses of the given symbol 'oldSymbol' with the
/// provided symbol 'newSymbol' that are nested within the given operation
/// 'from'. This does not traverse into any nested symbol tables. If there are
/// any unknown operations that may potentially be symbol tables, no uses are
/// replaced and failure is returned.
LogicalResult SymbolTable::replaceAllSymbolUses(StringRef oldSymbol,
                                                StringRef newSymbol,
                                                Operation *from) {
  return replaceAllSymbolUsesImpl(oldSymbol, newSymbol, from);
}
LogicalResult SymbolTable::replaceAllSymbolUses(Operation *oldSymbol,
                                                StringRef newSymbol,
                                                Operation *from) {
  return replaceAllSymbolUsesImpl(oldSymbol, newSymbol, from);
}
LogicalResult SymbolTable::replaceAllSymbolUses(StringRef oldSymbol,
                                                StringRef newSymbol,
                                                Region *from) {
  return replaceAllSymbolUsesImpl(oldSymbol, newSymbol, from);
}
LogicalResult SymbolTable::replaceAllSymbolUses(Operation *oldSymbol,
                                                StringRef newSymbol,
                                                Region *from) {
  return replaceAllSymbolUsesImpl(oldSymbol, newSymbol, from);
}

//===----------------------------------------------------------------------===//
// SymbolTableCollection
//===----------------------------------------------------------------------===//

Operation *SymbolTableCollection::lookupSymbolIn(Operation *symbolTableOp,
                                                 StringRef symbol) {
  return getSymbolTable(symbolTableOp).lookup(symbol);
}
Operation *SymbolTableCollection::lookupSymbolIn(Operation *symbolTableOp,
                                                 SymbolRefAttr name) {
  SmallVector<Operation *, 4> symbols;
  if (failed(lookupSymbolIn(symbolTableOp, name, symbols)))
    return nullptr;
  return symbols.back();
}
/// A variant of 'lookupSymbolIn' that returns all of the symbols referenced by
/// a given SymbolRefAttr. Returns failure if any of the nested references could
/// not be resolved.
LogicalResult
SymbolTableCollection::lookupSymbolIn(Operation *symbolTableOp,
                                      SymbolRefAttr name,
                                      SmallVectorImpl<Operation *> &symbols) {
  auto lookupFn = [this](Operation *symbolTableOp, StringRef symbol) {
    return lookupSymbolIn(symbolTableOp, symbol);
  };
  return lookupSymbolInImpl(symbolTableOp, name, symbols, lookupFn);
}

/// Returns the operation registered with the given symbol name within the
/// closest parent operation of, or including, 'from' with the
/// 'OpTrait::SymbolTable' trait. Returns nullptr if no valid symbol was
/// found.
Operation *SymbolTableCollection::lookupNearestSymbolFrom(Operation *from,
                                                          StringRef symbol) {
  Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(from);
  return symbolTableOp ? lookupSymbolIn(symbolTableOp, symbol) : nullptr;
}
Operation *
SymbolTableCollection::lookupNearestSymbolFrom(Operation *from,
                                               SymbolRefAttr symbol) {
  Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(from);
  return symbolTableOp ? lookupSymbolIn(symbolTableOp, symbol) : nullptr;
}

/// Lookup, or create, a symbol table for an operation.
SymbolTable &SymbolTableCollection::getSymbolTable(Operation *op) {
  auto it = symbolTables.try_emplace(op, nullptr);
  if (it.second)
    it.first->second = std::make_unique<SymbolTable>(op);
  return *it.first->second;
}

//===----------------------------------------------------------------------===//
// SymbolUserMap
//===----------------------------------------------------------------------===//

SymbolUserMap::SymbolUserMap(SymbolTableCollection &symbolTable,
                             Operation *symbolTableOp)
    : symbolTable(symbolTable) {
  // Walk each of the symbol tables looking for discardable callgraph nodes.
  SmallVector<Operation *> symbols;
  auto walkFn = [&](Operation *symbolTableOp, bool allUsesVisible) {
    for (Operation &nestedOp : symbolTableOp->getRegion(0).getOps()) {
      auto symbolUses = SymbolTable::getSymbolUses(&nestedOp);
      assert(symbolUses && "expected uses to be valid");

      for (const SymbolTable::SymbolUse &use : *symbolUses) {
        symbols.clear();
        (void)symbolTable.lookupSymbolIn(symbolTableOp, use.getSymbolRef(),
                                         symbols);
        for (Operation *symbolOp : symbols)
          symbolToUsers[symbolOp].insert(use.getUser());
      }
    }
  };
  // We just set `allSymUsesVisible` to false here because it isn't necessary
  // for building the user map.
  SymbolTable::walkSymbolTables(symbolTableOp, /*allSymUsesVisible=*/false,
                                walkFn);
}

void SymbolUserMap::replaceAllUsesWith(Operation *symbol,
                                       StringRef newSymbolName) {
  auto it = symbolToUsers.find(symbol);
  if (it == symbolToUsers.end())
    return;
  llvm::SetVector<Operation *> &users = it->second;

  // Replace the uses within the users of `symbol`.
  for (Operation *user : users)
    (void)SymbolTable::replaceAllSymbolUses(symbol, newSymbolName, user);

  // Move the current users of `symbol` to the new symbol if it is in the
  // symbol table.
  Operation *newSymbol =
      symbolTable.lookupSymbolIn(symbol->getParentOp(), newSymbolName);
  if (newSymbol != symbol) {
    // Transfer over the users to the new symbol.
    auto newIt = symbolToUsers.find(newSymbol);
    if (newIt == symbolToUsers.end())
      symbolToUsers.try_emplace(newSymbol, std::move(users));
    else
      newIt->second.set_union(users);
    symbolToUsers.erase(symbol);
  }
}

//===----------------------------------------------------------------------===//
// Visibility parsing implementation.
//===----------------------------------------------------------------------===//

ParseResult impl::parseOptionalVisibilityKeyword(OpAsmParser &parser,
                                                 NamedAttrList &attrs) {
  StringRef visibility;
  if (parser.parseOptionalKeyword(&visibility, {"public", "private", "nested"}))
    return failure();

  StringAttr visibilityAttr = parser.getBuilder().getStringAttr(visibility);
  attrs.push_back(parser.getBuilder().getNamedAttr(
      SymbolTable::getVisibilityAttrName(), visibilityAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// Symbol Interfaces
//===----------------------------------------------------------------------===//

/// Include the generated symbol interfaces.
#include "mlir/IR/SymbolInterfaces.cpp.inc"
