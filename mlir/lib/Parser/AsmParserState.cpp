//===- AsmParserState.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Parser/AsmParserState.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

/// Given a SMLoc corresponding to an identifier location, return a location
/// representing the full range of the identifier.
static llvm::SMRange convertIdLocToRange(llvm::SMLoc loc) {
  if (!loc.isValid())
    return llvm::SMRange();

  // Return if the given character is a valid identifier character.
  auto isIdentifierChar = [](char c) {
    return isalnum(c) || c == '$' || c == '.' || c == '_' || c == '-';
  };

  const char *curPtr = loc.getPointer();
  while (isIdentifierChar(*(++curPtr)))
    continue;
  return llvm::SMRange(loc, llvm::SMLoc::getFromPointer(curPtr));
}

//===----------------------------------------------------------------------===//
// AsmParserState::Impl
//===----------------------------------------------------------------------===//

struct AsmParserState::Impl {
  /// A mapping from operations in the input source file to their parser state.
  SmallVector<std::unique_ptr<OperationDefinition>> operations;
  DenseMap<Operation *, unsigned> operationToIdx;

  /// A mapping from blocks in the input source file to their parser state.
  SmallVector<std::unique_ptr<BlockDefinition>> blocks;
  DenseMap<Block *, unsigned> blocksToIdx;

  /// A set of value definitions that are placeholders for forward references.
  /// This map should be empty if the parser finishes successfully.
  DenseMap<Value, SmallVector<llvm::SMLoc>> placeholderValueUses;
};

//===----------------------------------------------------------------------===//
// AsmParserState
//===----------------------------------------------------------------------===//

AsmParserState::AsmParserState() : impl(std::make_unique<Impl>()) {}
AsmParserState::~AsmParserState() {}
AsmParserState &AsmParserState::operator=(AsmParserState &&other) {
  impl = std::move(other.impl);
  return *this;
}

//===----------------------------------------------------------------------===//
// Access State

auto AsmParserState::getBlockDefs() const -> iterator_range<BlockDefIterator> {
  return llvm::make_pointee_range(llvm::makeArrayRef(impl->blocks));
}

auto AsmParserState::getBlockDef(Block *block) const
    -> const BlockDefinition * {
  auto it = impl->blocksToIdx.find(block);
  return it == impl->blocksToIdx.end() ? nullptr : &*impl->blocks[it->second];
}

auto AsmParserState::getOpDefs() const -> iterator_range<OperationDefIterator> {
  return llvm::make_pointee_range(llvm::makeArrayRef(impl->operations));
}

//===----------------------------------------------------------------------===//
// Populate State

void AsmParserState::addDefinition(
    Operation *op, llvm::SMRange location,
    ArrayRef<std::pair<unsigned, llvm::SMLoc>> resultGroups) {
  std::unique_ptr<OperationDefinition> def =
      std::make_unique<OperationDefinition>(op, location);
  for (auto &resultGroup : resultGroups)
    def->resultGroups.emplace_back(resultGroup.first,
                                   convertIdLocToRange(resultGroup.second));

  impl->operationToIdx.try_emplace(op, impl->operations.size());
  impl->operations.emplace_back(std::move(def));
}

void AsmParserState::addDefinition(Block *block, llvm::SMLoc location) {
  auto it = impl->blocksToIdx.find(block);
  if (it == impl->blocksToIdx.end()) {
    impl->blocksToIdx.try_emplace(block, impl->blocks.size());
    impl->blocks.emplace_back(std::make_unique<BlockDefinition>(
        block, convertIdLocToRange(location)));
    return;
  }

  // If an entry already exists, this was a forward declaration that now has a
  // proper definition.
  impl->blocks[it->second]->definition.loc = convertIdLocToRange(location);
}

void AsmParserState::addDefinition(BlockArgument blockArg,
                                   llvm::SMLoc location) {
  auto it = impl->blocksToIdx.find(blockArg.getOwner());
  assert(it != impl->blocksToIdx.end() &&
         "expected owner block to have an entry");
  BlockDefinition &def = *impl->blocks[it->second];
  unsigned argIdx = blockArg.getArgNumber();

  if (def.arguments.size() <= argIdx)
    def.arguments.resize(argIdx + 1);
  def.arguments[argIdx] = SMDefinition(convertIdLocToRange(location));
}

void AsmParserState::addUses(Value value, ArrayRef<llvm::SMLoc> locations) {
  // Handle the case where the value is an operation result.
  if (OpResult result = value.dyn_cast<OpResult>()) {
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
      if (resultNo >= resultGroup.first) {
        for (llvm::SMLoc loc : locations)
          resultGroup.second.uses.push_back(convertIdLocToRange(loc));
        return;
      }
    }
    llvm_unreachable("expected valid result group for value use");
  }

  // Otherwise, this is a block argument.
  BlockArgument arg = value.cast<BlockArgument>();
  auto existingIt = impl->blocksToIdx.find(arg.getOwner());
  assert(existingIt != impl->blocksToIdx.end() &&
         "expected valid block definition for block argument");
  BlockDefinition &blockDef = *impl->blocks[existingIt->second];
  SMDefinition &argDef = blockDef.arguments[arg.getArgNumber()];
  for (llvm::SMLoc loc : locations)
    argDef.uses.emplace_back(convertIdLocToRange(loc));
}

void AsmParserState::addUses(Block *block, ArrayRef<llvm::SMLoc> locations) {
  auto it = impl->blocksToIdx.find(block);
  if (it == impl->blocksToIdx.end()) {
    it = impl->blocksToIdx.try_emplace(block, impl->blocks.size()).first;
    impl->blocks.emplace_back(std::make_unique<BlockDefinition>(block));
  }

  BlockDefinition &def = *impl->blocks[it->second];
  for (llvm::SMLoc loc : locations)
    def.definition.uses.push_back(convertIdLocToRange(loc));
}

void AsmParserState::refineDefinition(Value oldValue, Value newValue) {
  auto it = impl->placeholderValueUses.find(oldValue);
  assert(it != impl->placeholderValueUses.end() &&
         "expected `oldValue` to be a placeholder");
  addUses(newValue, it->second);
  impl->placeholderValueUses.erase(oldValue);
}
