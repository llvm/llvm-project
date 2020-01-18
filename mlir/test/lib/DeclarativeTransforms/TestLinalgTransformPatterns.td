//===- TestLinalgTransformPatterns.td - Test patterns --*- tablegen ----*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the pattern definition file for declarative Linalg transformations
// tests.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LINALG_TRANSFORMS_PATTERNS
#define TEST_LINALG_TRANSFORMS_PATTERNS

include "mlir/Dialect/Linalg/Transforms/LinalgTransformPatterns.td"

//===----------------------------------------------------------------------===//
// Test Linalg fusion patterns.
//===----------------------------------------------------------------------===//
def : Pat<(MatmulOp:$op $A, $_, $_),
          (TileAndFuseLinalgOp<[100, 150], [0], "L1">),
          [
            (Constraint<HasNoLinalgTransformMarker>),
            (Constraint<IsProducedByOpOfType<"MatmulOp">> $A),
          ],
          // In the buffer world there is no use-def chains or dags so benefits
          // cannot be computed automatically from the length of the matched
          // pattern. Instead we specify the benefit ourselves for now.
          // This is not expected to be a big challenge long-term because
          // pattern benefits are akin to feature engineering: features should
          // be learned.
          (addBenefit 1)>;

//===----------------------------------------------------------------------===//
// Linalg tiling patterns.
//===----------------------------------------------------------------------===//
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[2000, 3000, 4000], "L3">),
          [(Constraint<Or<[HasNoLinalgTransformMarker,
                           HasLinalgTransformMarker<"MEM">]>>)]>;
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[200, 300, 400], "L2">),
          [(Constraint<HasLinalgTransformMarker<"L3">>)]>;
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[20, 30, 40], "L1">),
          [(Constraint<HasLinalgTransformMarker<"L2">>)]>;
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[2, 3, 4], "REG">),
          [(Constraint<HasLinalgTransformMarker<"L1">>)]>;

def : Pattern<(MatvecOp:$op $_, $_, $_),
              [(TileLinalgOp<[5, 6], "L1">)],
              [(Constraint<HasNoLinalgTransformMarker>)]>;

def : Pattern<(DotOp:$op $_, $_, $_),
              [(TileLinalgOp<[8000], "L1">)],
              [(Constraint<Or<[HasNoLinalgTransformMarker,
                               HasLinalgTransformMarker<"MEM">,
                               HasLinalgTransformMarker<"L3">,
                               HasLinalgTransformMarker<"L2">]>>)]>;
def : Pattern<(DotOp:$op $_, $_, $_),
              [(TileLinalgOp<[8], "REG">)],
              [(Constraint<HasLinalgTransformMarker<"L1">>)]>;

//===----------------------------------------------------------------------===//
// Linalg tiling and permutation patterns.
//===----------------------------------------------------------------------===//
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[2000, 3000, 4000], "L2__with_perm__", [1,2,0]>),
          [(Constraint<HasLinalgTransformMarker<"__with_perm__">>)]>;
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[200, 300, 400], "L1__with_perm__", [1,0,2]>),
          [(Constraint<HasLinalgTransformMarker<"L2__with_perm__">>)]>;
def : Pat<(MatmulOp:$op $_, $_, $_),
          (TileLinalgOp<[20, 30, 40], "REG__with_perm__">),
          [(Constraint<HasLinalgTransformMarker<"L1__with_perm__">>)]>;


def : Pattern<(MatvecOp:$op $_, $_, $_),
              [(TileLinalgOp<[5, 6], "L1__with_perm__", [1,0]>)],
              [(Constraint<HasLinalgTransformMarker<"__with_perm__">>)]>;

def : Pattern<(DotOp:$op $_, $_, $_),
              [(TileLinalgOp<[8000], "L1__with_perm__">)],
              [(Constraint<HasLinalgTransformMarker<"__with_perm__">>)]>;
def : Pattern<(DotOp:$op $_, $_, $_),
              [(TileLinalgOp<[8], "REG__with_perm__">)],
              [(Constraint<HasLinalgTransformMarker<"L1__with_perm__">>)]>;

//===----------------------------------------------------------------------===//
// Linalg to loops patterns.
//===----------------------------------------------------------------------===//
def : Pattern<(DotOp:$op $_, $_, $_),
              [(LinalgOpToLoops<"DotOp">)],
              [(Constraint<HasLinalgTransformMarker<"REG">>)]>;

//===----------------------------------------------------------------------===//
// Linalg to vector contraction patterns.
//===----------------------------------------------------------------------===//
def : Pattern<(MatmulOp:$op $_, $_, $_),
              [(VectorizeLinalgOp)],
              [(Constraint<And<[
                HasLinalgTransformMarker<"VECTORIZE">,
                PreconditionVectorizeLinalgOp
               ]>>)]>;
def : Pattern<(GenericOp:$op $_, $_, $_, $_, $_, $_, $_, $_),
              [(VectorizeLinalgOp)],
              [(Constraint<And<[
                HasLinalgTransformMarker<"VECTORIZE">,
                PreconditionVectorizeLinalgOp
               ]>>)]>;

//===----------------------------------------------------------------------===//
// Linalg generic permutation patterns.
//===----------------------------------------------------------------------===//
def : Pat<(GenericOp:$op $_, $_, $_, $_, $_, $_, $_, $_),
              (PermuteGenericLinalgOp<[1, 2, 0], "PERMUTE"> $op),
              [(Constraint<And<[
                HasNoLinalgTransformMarker,
                AffineMapDomainHasDim<3>,
                PreconditionPermuteGenericLinalgOp<[1, 2, 0]>
              ]>>)]>;

def : Pat<(IndexedGenericOp:$op $_, $_, $_, $_, $_, $_, $_, $_),
              (PermuteGenericLinalgOp<[1, 2, 0], "PERMUTE"> $op),
              [(Constraint<And<[
                HasNoLinalgTransformMarker,
                AffineMapDomainHasDim<3>,
                PreconditionPermuteGenericLinalgOp<[1, 2, 0]>
              ]>>)]>;

//===----------------------------------------------------------------------===//
// Linalg subview operands promotion.
//===----------------------------------------------------------------------===//
def : Pat<(MatmulOp:$op $_, $_, $_),
          (PromoteSubviewsLinalgOp),
          [(Constraint<And<[
              PreconditionPromoteSubviewsLinalgOp,
              HasOperandsOfType<"SubViewOp">,
              HasLinalgTransformMarker<"_promote_views_">]>>
           )]>;

#endif // TEST_LINALG_TRANSFORMS_PATTERNS
