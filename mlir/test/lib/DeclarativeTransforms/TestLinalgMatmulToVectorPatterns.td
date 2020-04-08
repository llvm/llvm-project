//===- TestLinalgMatmulToVectorPatterns.td - Test patterns -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the pattern definition file for declarative Linalg transformations
// tests.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LINALG_MATMUL_TO_VECTOR_PATTERNS
#define TEST_LINALG_MATMUL_TO_VECTOR_PATTERNS

include "mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.td"
include "mlir/Dialect/Vector/VectorTransformPatterns.td"

//===----------------------------------------------------------------------===//
// Linalg tiling and permutation patterns.
//===----------------------------------------------------------------------===//
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[768, 264, 768], "L2__with_perm__", [1, 2, 0]>),
         [(Constraint<HasLinalgTransformMarker<"__with_perm__">>)]>;
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[8, 12, 16], "L1__with_perm__", [1, 0, 2]>),
         [(Constraint<HasLinalgTransformMarker<"L2__with_perm__">>)]>;
def : Pat<(MatmulOp:$op $_, $_, $_),
          (PromoteSubviewsLinalgOp),
         [(Constraint<HasOperandsOfType<"SubViewOp">>),
          (Constraint<HasLinalgTransformMarker<"L1__with_perm__">>)]>;

//===----------------------------------------------------------------------===//
// Linalg to vector contraction patterns.
//===----------------------------------------------------------------------===//
def : Pattern<(MatmulOp:$op $_, $_, $_),
              [(VectorizeLinalgOp)],
              [(Constraint<And<[
                HasLinalgTransformMarker<"L1__with_perm__">,
                PreconditionVectorizeLinalgOp]>>)]>;

#endif // TEST_LINALG_MATMUL_TO_VECTOR_PATTERNS
