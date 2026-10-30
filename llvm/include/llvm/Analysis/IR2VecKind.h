//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IR2VECKIND_H
#define LLVM_ANALYSIS_IR2VECKIND_H

namespace llvm {

/// IR2Vec computes two kinds of embeddings: Symbolic and Flow-aware.
/// Symbolic embeddings capture the "syntactic" and "statistical correlation"
/// of the IR entities. Flow-aware embeddings build on top of symbolic
/// embeddings and additionally capture the flow information in the IR.
/// IR2VecKind is used to specify the type of embeddings to generate.
/// Note: Implementation of FlowAware embeddings is not same as the one
/// described in the paper. The current implementation is a simplified version
/// that captures the flow information (SSA-based use-defs) without tracing
/// through memory level use-defs in the embedding computation described in the
/// paper.
enum class IR2VecKind { Symbolic, FlowAware };

} // namespace llvm

#endif // LLVM_ANALYSIS_IR2VECKIND_H
