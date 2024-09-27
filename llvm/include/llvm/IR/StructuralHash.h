//===- llvm/IR/StructuralHash.h - IR Hashing --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides hashing of the LLVM IR structure to be used to check
// Passes modification status.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_STRUCTURALHASH_H
#define LLVM_IR_STRUCTURALHASH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/IR/Instruction.h"
#include <cstdint>

namespace llvm {

class Function;
class Module;

using IRHash = stable_hash;
using OpndHash = stable_hash;

/// Returns a hash of the function \p F.
/// \param F The function to hash.
/// \param DetailedHash Whether or not to encode additional information in the
/// hash. The additional information added into the hash when this flag is set
/// to true includes instruction and operand type information.
IRHash StructuralHash(const Function &F, bool DetailedHash = false);

/// Returns a hash of the module \p M by hashing all functions and global
/// variables contained within. \param M The module to hash. \param DetailedHash
/// Whether or not to encode additional information in the function hashes that
/// composed the module hash.
IRHash StructuralHash(const Module &M, bool DetailedHash = false);

/// The pair of an instruction index and a operand index.
using IndexPair = std::pair<unsigned, unsigned>;

/// A map from an instruction index to an instruction pointer.
using IndexInstrMap = MapVector<unsigned, Instruction *>;

/// A map from an IndexPair to an OpndHash.
using IndexOperandHashMapType = DenseMap<IndexPair, OpndHash>;

/// A function that takes an instruction and an operand index and returns true
/// if the operand should be ignored in the function hash computation.
using IgnoreOperandFunc = std::function<bool(const Instruction *, unsigned)>;

struct FunctionHashInfo {
  /// A hash value representing the structural content of the function
  IRHash FunctionHash;
  /// A mapping from instruction indices to instruction pointers
  std::unique_ptr<IndexInstrMap> IndexInstruction;
  /// A mapping from pairs of instruction indices and operand indices
  /// to the hashes of the operands. This can be used to analyze or
  /// reconstruct the differences in ignored operands
  std::unique_ptr<IndexOperandHashMapType> IndexOperandHashMap;

  FunctionHashInfo(IRHash FuntionHash,
                   std::unique_ptr<IndexInstrMap> IndexInstruction,
                   std::unique_ptr<IndexOperandHashMapType> IndexOperandHashMap)
      : FunctionHash(FuntionHash),
        IndexInstruction(std::move(IndexInstruction)),
        IndexOperandHashMap(std::move(IndexOperandHashMap)) {}
#if 0 // TODO: delete
  // Disable copy operations
  FunctionHashInfo(const FunctionHashInfo &) = delete;
  FunctionHashInfo &operator=(const FunctionHashInfo &) = delete;
  // Enable move operations
  FunctionHashInfo(FunctionHashInfo &&) noexcept = default;
  FunctionHashInfo &operator=(FunctionHashInfo &&) noexcept = default;
#endif
};

/// Computes a structural hash of a given function, considering the structure
/// and content of the function's instructions while allowing for selective
/// ignoring of certain operands based on custom criteria. This hash can be used
/// to identify functions that are structurally similar or identical, which is
/// useful in optimizations, deduplication, or analysis tasks.
/// \param F The function to hash.
/// \param IgnoreOp A callable that takes an instruction and one of its operands
/// and returns true if an operand should be ignored in the hash computation.
/// for the hash computation.
/// \return A FunctionHashInfo structure

FunctionHashInfo StructuralHashWithDifferences(const Function &F,
                                               IgnoreOperandFunc IgnoreOp);

} // end namespace llvm

#endif
