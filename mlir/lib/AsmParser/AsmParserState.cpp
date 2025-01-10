//===- AsmParserState.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cctype>
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// AsmParserState::Impl
//===----------------------------------------------------------------------===//

struct AsmParserState::Impl {
  /// A map from a SymbolRefAttr to a range of uses.
  using SymbolUseMap =
      DenseMap<Attribute, SmallVector<SmallVector<SMRange>, 0>>;

  struct PartialOpDef {
    explicit PartialOpDef(const OperationName &opName) {
      if (opName.hasTrait<OpTrait::SymbolTable>())
        symbolTable = std::make_unique<SymbolUseMap>();
    }

    /// Return if this operation is a symbol table.
    bool isSymbolTable() const { return symbolTable.get(); }

    /// If this operation is a symbol table, the following contains symbol uses
    /// within this operation.
    std::unique_ptr<SymbolUseMap> symbolTable;
  };

  /// Resolve any symbol table uses in the IR.
  void resolveSymbolUses();

  /// A mapping from operations in the input source file to their parser state.
  SmallVector<std::unique_ptr<OperationDefinition>> operations;
  DenseMap<Operation *, unsigned> operationToIdx;

  /// A mapping from blocks in the input source file to their parser state.
  SmallVector<std::unique_ptr<BlockDefinition>> blocks;
  DenseMap<Block *, unsigned> blocksToIdx;

  /// A mapping from aliases in the input source file to their parser state.
  SmallVector<std::unique_ptr<AttributeAliasDefinition>> attrAliases;
  SmallVector<std::unique_ptr<TypeAliasDefinition>> typeAliases;
  llvm::StringMap<unsigned> attrAliasToIdx;
  llvm::StringMap<unsigned> typeAliasToIdx;

  /// A set of value definitions that are placeholders for forward references.
  /// This map should be empty if the parser finishes successfully.
  DenseMap<Value, SmallVector<SMLoc>> placeholderValueUses;

  /// The symbol table operations within the IR.
  SmallVector<std::pair<Operation *, std::unique_ptr<SymbolUseMap>>>
      symbolTableOperations;

  /// A stack of partial operation definitions that have been started but not
  /// yet finalized.
  SmallVector<PartialOpDef> partialOperations;

  /// A stack of symbol use scopes. This is used when collecting symbol table
  /// uses during parsing.
  SmallVector<SymbolUseMap *> symbolUseScopes;

  /// A symbol table containing all of the symbol table operations in the IR.
  SymbolTableCollection symbolTable;
};

void AsmParserState::Impl::resolveSymbolUses() {
  SmallVector<Operation *> symbolOps;
  for (auto &opAndUseMapIt : symbolTableOperations) {
    for (auto &it : *opAndUseMapIt.second) {
      symbolOps.clear();
      if (failed(symbolTable.lookupSymbolIn(
              opAndUseMapIt.first, cast<SymbolRefAttr>(it.first), symbolOps)))
        continue;

      for (ArrayRef<SMRange> useRange : it.second) {
        for (const auto &symIt : llvm::zip(symbolOps, useRange)) {
          auto opIt = operationToIdx.find(std::get<0>(symIt));
          if (opIt != operationToIdx.end())
            operations[opIt->second]->symbolUses.push_back(std::get<1>(symIt));
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// AsmParserState
//===----------------------------------------------------------------------===//

AsmParserState::AsmParserState() : impl(std::make_unique<Impl>()) {}
AsmParserState::~AsmParserState() = default;
AsmParserState &AsmParserState::operator=(AsmParserState &&other) {
  impl = std::move(other.impl);
  return *this;
}

//===----------------------------------------------------------------------===//
// Access State

auto AsmParserState::getBlockDefs() const -> iterator_range<BlockDefIterator> {
  return llvm::make_pointee_range(llvm::ArrayRef(impl->blocks));
}

auto AsmParserState::getBlockDef(Block *block) const
    -> const BlockDefinition * {
  auto it = impl->blocksToIdx.find(block);
  return it == impl->blocksToIdx.end() ? nullptr : &*impl->blocks[it->second];
}

auto AsmParserState::getOpDefs() const -> iterator_range<OperationDefIterator> {
  return llvm::make_pointee_range(llvm::ArrayRef(impl->operations));
}

auto AsmParserState::getOpDef(Operation *op) const
    -> const OperationDefinition * {
  auto it = impl->operationToIdx.find(op);
  return it == impl->operationToIdx.end() ? nullptr
                                          : &*impl->operations[it->second];
}

auto AsmParserState::getAttributeAliasDefs() const
    -> iterator_range<AttributeDefIterator> {
  return llvm::make_pointee_range(ArrayRef(impl->attrAliases));
}

auto AsmParserState::getAttributeAliasDef(StringRef name) const
    -> const AttributeAliasDefinition * {
  auto it = impl->attrAliasToIdx.find(name);
  return it == impl->attrAliasToIdx.end() ? nullptr
                                          : &*impl->attrAliases[it->second];
}

auto AsmParserState::getTypeAliasDefs() const
    -> iterator_range<TypeDefIterator> {
  return llvm::make_pointee_range(ArrayRef(impl->typeAliases));
}

auto AsmParserState::getTypeAliasDef(StringRef name) const
    -> const TypeAliasDefinition * {
  auto it = impl->typeAliasToIdx.find(name);
  return it == impl->typeAliasToIdx.end() ? nullptr
                                          : &*impl->typeAliases[it->second];
}

/// Lex a string token whose contents start at the given `curPtr`. Returns the
/// position at the end of the string, after a terminal or invalid character
/// (e.g. `"` or `\0`).
static const char *lexLocStringTok(const char *curPtr) {
  while (char c = *curPtr++) {
    // Check for various terminal characters.
    if (StringRef("\"\n\v\f").contains(c))
      return curPtr;

    // Check for escape sequences.
    if (c == '\\') {
      // Check a few known escapes and \xx hex digits.
      if (*curPtr == '"' || *curPtr == '\\' || *curPtr == 'n' || *curPtr == 't')
        ++curPtr;
      else if (llvm::isHexDigit(*curPtr) && llvm::isHexDigit(curPtr[1]))
        curPtr += 2;
      else
        return curPtr;
    }
  }

  // If we hit this point, we've reached the end of the buffer. Update the end
  // pointer to not point past the buffer.
  return curPtr - 1;
}

SMRange AsmParserState::convertIdLocToRange(SMLoc loc) {
  if (!loc.isValid())
    return SMRange();
  const char *curPtr = loc.getPointer();

  // Check if this is a string token.
  if (*curPtr == '"') {
    curPtr = lexLocStringTok(curPtr + 1);

    // Otherwise, default to handling an identifier.
  } else {
    // Return if the given character is a valid identifier character.
    auto isIdentifierChar = [](char c) {
      return isalnum(c) || c == '$' || c == '.' || c == '_' || c == '-';
    };

    while (*curPtr && isIdentifierChar(*(++curPtr)))
      continue;
  }

  return SMRange(loc, SMLoc::getFromPointer(curPtr));
}

//===----------------------------------------------------------------------===//
// Populate State

void AsmParserState::initialize(Operation *topLevelOp) {
  startOperationDefinition(topLevelOp->getName());

  // If the top-level operation is a symbol table, push a new symbol scope.
  Impl::PartialOpDef &partialOpDef = impl->partialOperations.back();
  if (partialOpDef.isSymbolTable())
    impl->symbolUseScopes.push_back(partialOpDef.symbolTable.get());
}

void AsmParserState::finalize(Operation *topLevelOp) {
  assert(!impl->partialOperations.empty() &&
         "expected valid partial operation definition");
  Impl::PartialOpDef partialOpDef = impl->partialOperations.pop_back_val();

  // If this operation is a symbol table, resolve any symbol uses.
  if (partialOpDef.isSymbolTable()) {
    impl->symbolTableOperations.emplace_back(
        topLevelOp, std::move(partialOpDef.symbolTable));
  }
  impl->resolveSymbolUses();
}

void AsmParserState::startOperationDefinition(const OperationName &opName) {
  impl->partialOperations.emplace_back(opName);
}

void AsmParserState::finalizeOperationDefinition(
    Operation *op, SMRange nameLoc, SMLoc endLoc,
    ArrayRef<std::pair<unsigned, SMLoc>> resultGroups) {
  assert(!impl->partialOperations.empty() &&
         "expected valid partial operation definition");
  Impl::PartialOpDef partialOpDef = impl->partialOperations.pop_back_val();

  // Build the full operation definition.
  std::unique_ptr<OperationDefinition> def =
      std::make_unique<OperationDefinition>(op, nameLoc, endLoc);
  for (auto &resultGroup : resultGroups)
    def->resultGroups.emplace_back(resultGroup.first,
                                   convertIdLocToRange(resultGroup.second));
  impl->operationToIdx.try_emplace(op, impl->operations.size());
  impl->operations.emplace_back(std::move(def));

  // If this operation is a symbol table, resolve any symbol uses.
  if (partialOpDef.isSymbolTable()) {
    impl->symbolTableOperations.emplace_back(
        op, std::move(partialOpDef.symbolTable));
  }
}

void AsmParserState::startRegionDefinition() {
  assert(!impl->partialOperations.empty() &&
         "expected valid partial operation definition");

  // If the parent operation of this region is a symbol table, we also push a
  // new symbol scope.
  Impl::PartialOpDef &partialOpDef = impl->partialOperations.back();
  if (partialOpDef.isSymbolTable())
    impl->symbolUseScopes.push_back(partialOpDef.symbolTable.get());
}

void AsmParserState::finalizeRegionDefinition() {
  assert(!impl->partialOperations.empty() &&
         "expected valid partial operation definition");

  // If the parent operation of this region is a symbol table, pop the symbol
  // scope for this region.
  Impl::PartialOpDef &partialOpDef = impl->partialOperations.back();
  if (partialOpDef.isSymbolTable())
    impl->symbolUseScopes.pop_back();
}

void AsmParserState::addDefinition(Block *block, SMLoc location) {
  auto [it, inserted] =
      impl->blocksToIdx.try_emplace(block, impl->blocks.size());
  if (inserted) {
    impl->blocks.emplace_back(std::make_unique<BlockDefinition>(
        block, convertIdLocToRange(location)));
    return;
  }

  // If an entry already exists, this was a forward declaration that now has a
  // proper definition.
  impl->blocks[it->second]->definition.loc = convertIdLocToRange(location);
}

void AsmParserState::addDefinition(BlockArgument blockArg, SMLoc location) {
  auto it = impl->blocksToIdx.find(blockArg.getOwner());
  assert(it != impl->blocksToIdx.end() &&
         "expected owner block to have an entry");
  BlockDefinition &def = *impl->blocks[it->second];
  unsigned argIdx = blockArg.getArgNumber();

  if (def.arguments.size() <= argIdx)
    def.arguments.resize(argIdx + 1);
  def.arguments[argIdx] = SMDefinition(convertIdLocToRange(location));
}

void AsmParserState::addAttrAliasDefinition(StringRef name, SMRange location,
                                            Attribute value) {
  auto [it, inserted] =
      impl->attrAliasToIdx.try_emplace(name, impl->attrAliases.size());
  // Location aliases may be referenced before they are defined.
  if (inserted) {
    impl->attrAliases.push_back(
        std::make_unique<AttributeAliasDefinition>(name, location, value));
  } else {
    AttributeAliasDefinition &attr = *impl->attrAliases[it->second];
    attr.definition.loc = location;
    attr.value = value;
  }
}

void AsmParserState::addTypeAliasDefinition(StringRef name, SMRange location,
                                            Type value) {
  [[maybe_unused]] auto [it, inserted] =
      impl->typeAliasToIdx.try_emplace(name, impl->typeAliases.size());
  assert(inserted && "unexpected attribute alias redefinition");
  impl->typeAliases.push_back(
      std::make_unique<TypeAliasDefinition>(name, location, value));
}

void AsmParserState::addUses(Value value, ArrayRef<SMLoc> locations) {
  // Handle the case where the value is an operation result.
  if (OpResult result = dyn_cast<OpResult>(value)) {
    // Check to see if a definition for the parent operation has been recorded.
    // If one hasn't, we treat the provided value as a placeholder value that
    // will be refined further later.
    Operation *parentOp = result.getOwner();
    auto existingIt = impl->operationToIdx.find(parentOp);
    if (existingIt == impl->operationToIdx.end()) {
      impl->placeholderValueUses[value].append(locations.begin(),
                                               locations.end());
      return;
    }

    // If a definition does exist, locate the value's result group and add the
    // use. The result groups are ordered by increasing start index, so we just
    // need to find the last group that has a smaller/equal start index.
    unsigned resultNo = result.getResultNumber();
    OperationDefinition &def = *impl->operations[existingIt->second];
    for (auto &resultGroup : llvm::reverse(def.resultGroups)) {
      if (resultNo >= resultGroup.startIndex) {
        for (SMLoc loc : locations)
          resultGroup.definition.uses.push_back(convertIdLocToRange(loc));
        return;
      }
    }
    llvm_unreachable("expected valid result group for value use");
  }

  // Otherwise, this is a block argument.
  BlockArgument arg = cast<BlockArgument>(value);
  auto existingIt = impl->blocksToIdx.find(arg.getOwner());
  assert(existingIt != impl->blocksToIdx.end() &&
         "expected valid block definition for block argument");
  BlockDefinition &blockDef = *impl->blocks[existingIt->second];
  SMDefinition &argDef = blockDef.arguments[arg.getArgNumber()];
  for (SMLoc loc : locations)
    argDef.uses.emplace_back(convertIdLocToRange(loc));
}

void AsmParserState::addUses(Block *block, ArrayRef<SMLoc> locations) {
  auto [it, inserted] =
      impl->blocksToIdx.try_emplace(block, impl->blocks.size());
  if (inserted)
    impl->blocks.emplace_back(std::make_unique<BlockDefinition>(block));

  BlockDefinition &def = *impl->blocks[it->second];
  for (SMLoc loc : locations)
    def.definition.uses.push_back(convertIdLocToRange(loc));
}

void AsmParserState::addUses(SymbolRefAttr refAttr,
                             ArrayRef<SMRange> locations) {
  // Ignore this symbol if no scopes are active.
  if (impl->symbolUseScopes.empty())
    return;

  assert((refAttr.getNestedReferences().size() + 1) == locations.size() &&
         "expected the same number of references as provided locations");
  (*impl->symbolUseScopes.back())[refAttr].emplace_back(locations.begin(),
                                                        locations.end());
}

void AsmParserState::addAttrAliasUses(StringRef name, SMRange location) {
  auto it = impl->attrAliasToIdx.find(name);
  // Location aliases may be referenced before they are defined.
  if (it == impl->attrAliasToIdx.end()) {
    it = impl->attrAliasToIdx.try_emplace(name, impl->attrAliases.size()).first;
    impl->attrAliases.push_back(
        std::make_unique<AttributeAliasDefinition>(name));
  }
  AttributeAliasDefinition &def = *impl->attrAliases[it->second];
  def.definition.uses.push_back(location);
}

void AsmParserState::addTypeAliasUses(StringRef name, SMRange location) {
  auto it = impl->typeAliasToIdx.find(name);
  // Location aliases may be referenced before they are defined.
  assert(it != impl->typeAliasToIdx.end() &&
         "expected valid type alias definition");
  TypeAliasDefinition &def = *impl->typeAliases[it->second];
  def.definition.uses.push_back(location);
}

void AsmParserState::refineDefinition(Value oldValue, Value newValue) {
  auto it = impl->placeholderValueUses.find(oldValue);
  assert(it != impl->placeholderValueUses.end() &&
         "expected `oldValue` to be a placeholder");
  addUses(newValue, it->second);
  impl->placeholderValueUses.erase(oldValue);
}
