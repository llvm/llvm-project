//===- Common.h - Shared types for IR2Vec/MIR2Vec tools -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains shared types and constants used by both the IR2Vec
/// and MIR2Vec tool implementations. It has no dependency on either the
/// LLVM IR or Machine IR APIs, making it safe to include from either side.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_IR2VEC_UTILS_COMMON_H
#define LLVM_TOOLS_LLVM_IR2VEC_UTILS_COMMON_H

#include <string>
#include <vector>

namespace llvm {

/// Tool name for error reporting
static const char *ToolName = "llvm-ir2vec";

/// Specifies the granularity at which embeddings are generated.
enum EmbeddingLevel {
  InstructionLevel, // Generate instruction-level embeddings
  BasicBlockLevel,  // Generate basic block-level embeddings
  FunctionLevel     // Generate function-level embeddings
};

/// Represents a single knowledge graph triplet (Head, Relation, Tail)
/// where indices reference entities in an EntityList
struct Triplet {
  unsigned Head = 0;     ///< Index of the head entity in the entity list
  unsigned Tail = 0;     ///< Index of the tail entity in the entity list
  unsigned Relation = 0; ///< Relation type (see RelationType enum)
};

/// Result structure containing all generated triplets and metadata
struct TripletResult {
  unsigned MaxRelation =
      0; ///< Highest relation index used (for ArgRelation + N)
  std::vector<Triplet> Triplets; ///< Collection of all generated triplets
};

/// Entity mappings: [entity_name]
using EntityList = std::vector<std::string>;

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_IR2VEC_UTILS_COMMON_H
