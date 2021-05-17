//===- AsmParserState.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PARSER_ASMPARSERSTATE_H
#define MLIR_PARSER_ASMPARSERSTATE_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include <cstddef>

namespace mlir {
class Block;
class BlockArgument;
class FileLineColLoc;
class Operation;
class Value;

/// This class represents state from a parsed MLIR textual format string. It is
/// useful for building additional analysis and language utilities on top of
/// textual MLIR. This should generally not be used for traditional compilation.
class AsmParserState {
public:
  /// This class represents a definition within the source manager, containing
  /// it's defining location and locations of any uses. SMDefinitions are only
  /// provided for entities that have uses within an input file, e.g. SSA
  /// values, Blocks, and Symbols.
  struct SMDefinition {
    SMDefinition() = default;
    SMDefinition(llvm::SMRange loc) : loc(loc) {}

    /// The source location of the definition.
    llvm::SMRange loc;
    /// The source location of all uses of the definition.
    SmallVector<llvm::SMRange> uses;
  };

  /// This class represents the information for an operation definition within
  /// an input file.
  struct OperationDefinition {
    struct ResultGroupDefinition {
      /// The result number that starts this group.
      unsigned startIndex;
      /// The source definition of the result group.
      SMDefinition definition;
    };

    OperationDefinition(Operation *op, llvm::SMRange loc) : op(op), loc(loc) {}

    /// The operation representing this definition.
    Operation *op;

    /// The source location for the operation, i.e. the location of its name.
    llvm::SMRange loc;

    /// Source definitions for any result groups of this operation.
    SmallVector<std::pair<unsigned, SMDefinition>> resultGroups;
  };

  /// This class represents the information for a block definition within the
  /// input file.
  struct BlockDefinition {
    BlockDefinition(Block *block, llvm::SMRange loc = {})
        : block(block), definition(loc) {}

    /// The block representing this definition.
    Block *block;

    /// The source location for the block, i.e. the location of its name, and
    /// any uses it has.
    SMDefinition definition;

    /// Source definitions for any arguments of this block.
    SmallVector<SMDefinition> arguments;
  };

  AsmParserState();
  ~AsmParserState();
  AsmParserState &operator=(AsmParserState &&other);

  //===--------------------------------------------------------------------===//
  // Access State
  //===--------------------------------------------------------------------===//

  using BlockDefIterator = llvm::pointee_iterator<
      ArrayRef<std::unique_ptr<BlockDefinition>>::iterator>;
  using OperationDefIterator = llvm::pointee_iterator<
      ArrayRef<std::unique_ptr<OperationDefinition>>::iterator>;

  /// Return a range of the BlockDefinitions held by the current parser state.
  iterator_range<BlockDefIterator> getBlockDefs() const;

  /// Return the definition for the given block, or nullptr if the given
  /// block does not have a definition.
  const BlockDefinition *getBlockDef(Block *block) const;

  /// Return a range of the OperationDefinitions held by the current parser
  /// state.
  iterator_range<OperationDefIterator> getOpDefs() const;

  //===--------------------------------------------------------------------===//
  // Populate State
  //===--------------------------------------------------------------------===//

  /// Add a definition of the given operation.
  void addDefinition(
      Operation *op, llvm::SMRange location,
      ArrayRef<std::pair<unsigned, llvm::SMLoc>> resultGroups = llvm::None);
  void addDefinition(Block *block, llvm::SMLoc location);
  void addDefinition(BlockArgument blockArg, llvm::SMLoc location);

  /// Add a source uses of the given value.
  void addUses(Value value, ArrayRef<llvm::SMLoc> locations);
  void addUses(Block *block, ArrayRef<llvm::SMLoc> locations);

  /// Refine the `oldValue` to the `newValue`. This is used to indicate that
  /// `oldValue` was a placeholder, and the uses of it should really refer to
  /// `newValue`.
  void refineDefinition(Value oldValue, Value newValue);

private:
  struct Impl;

  /// A pointer to the internal implementation of this class.
  std::unique_ptr<Impl> impl;
};

} // end namespace mlir

#endif // MLIR_PARSER_ASMPARSERSTATE_H
