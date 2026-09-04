//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASMPARSER_ASMPARSERCONTEXT_H
#define LLVM_ASMPARSER_ASMPARSERCONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/AsmParser/FileLoc.h"
#include "llvm/IR/Value.h"
#include <optional>

namespace llvm {
class BasicBlock;

/// Registry of file location information for LLVM IR constructs.
///
/// This class provides access to the file location information
/// for various LLVM IR constructs. Currently, it supports Function,
/// BasicBlock and Instruction locations.
///
/// When available, it can answer queries about what is at a given
/// file location, as well as where in a file a given IR construct
/// is.
///
/// This information is optionally emitted by the LLParser while
/// it reads LLVM textual IR.
class AsmParserContext {
  using FMap =
      IntervalMap<FileLoc, Function *,
                  IntervalMapImpl::NodeSizer<FileLoc, Function *>::LeafSize,
                  IntervalMapHalfOpenInfo<FileLoc>>;

  DenseMap<Function *, FileLocRange> Functions;
  FMap::Allocator FAllocator;
  FMap FunctionsInverse = FMap(FAllocator);

  DenseMap<BasicBlock *, FileLocRange> Blocks;
  using BBMap =
      IntervalMap<FileLoc, BasicBlock *,
                  IntervalMapImpl::NodeSizer<FileLoc, BasicBlock *>::LeafSize,
                  IntervalMapHalfOpenInfo<FileLoc>>;
  BBMap::Allocator BBAllocator;
  BBMap BlocksInverse = BBMap(BBAllocator);
  DenseMap<Value *, FileLocRange> InstructionsAndArguments;
  using VMap =
      IntervalMap<FileLoc, Value *,
                  IntervalMapImpl::NodeSizer<FileLoc, Value *>::LeafSize,
                  IntervalMapHalfOpenInfo<FileLoc>>;
  VMap::Allocator VAllocator;
  VMap InstructionsAndArgumentsInverse = VMap(VAllocator);

  VMap ReferencedValues = VMap(VAllocator);

public:
  LLVM_ABI std::optional<FileLocRange>
  getFunctionLocation(const Function *) const;
  LLVM_ABI std::optional<FileLocRange>
  getBlockLocation(const BasicBlock *) const;
  LLVM_ABI std::optional<FileLocRange>
  getInstructionOrArgumentLocation(const Value *) const;
  /// Get the function at the requested location range.
  /// If no single function occupies the queried range, or the record is
  /// missing, a nullptr is returned.
  LLVM_ABI Function *getFunctionAtLocation(const FileLocRange &) const;
  /// Get the function at the requested location.
  /// If no function occupies the queried location, or the record is missing, a
  /// nullptr is returned.
  LLVM_ABI Function *getFunctionAtLocation(const FileLoc &) const;
  /// Get the block at the requested location range.
  /// If no single block occupies the queried range, or the record is missing, a
  /// nullptr is returned.
  LLVM_ABI BasicBlock *getBlockAtLocation(const FileLocRange &) const;
  /// Get the block at the requested location.
  /// If no block occupies the queried location, or the record is missing, a
  /// nullptr is returned.
  LLVM_ABI BasicBlock *getBlockAtLocation(const FileLoc &) const;
  /// Get the instruction or function argument at the requested location range.
  /// If no single instruction occupies the queried range, or the record is
  /// missing, a nullptr is returned.
  LLVM_ABI Value *
  getInstructionOrArgumentAtLocation(const FileLocRange &) const;
  /// Get the instruction or function argument at the requested location.
  /// If no instruction occupies the queried location, or the record is missing,
  /// a nullptr is returned.
  LLVM_ABI Value *getInstructionOrArgumentAtLocation(const FileLoc &) const;
  /// Get value referenced at the requested location.
  /// If no value occupies the queried location, or the record is missing,
  /// a nullptr is returned.
  LLVM_ABI Value *getValueReferencedAtLocation(const FileLoc &) const;
  /// Get value referenced at the requested location range.
  /// If no value occupies the queried location, or the record is missing,
  /// a nullptr is returned.
  LLVM_ABI Value *getValueReferencedAtLocation(const FileLocRange &) const;
  LLVM_ABI bool addFunctionLocation(Function *, const FileLocRange &);
  LLVM_ABI bool addBlockLocation(BasicBlock *, const FileLocRange &);
  LLVM_ABI bool addInstructionOrArgumentLocation(Value *, const FileLocRange &);
  LLVM_ABI bool addValueReferenceAtLocation(Value *, const FileLocRange &);
};
} // namespace llvm

#endif
