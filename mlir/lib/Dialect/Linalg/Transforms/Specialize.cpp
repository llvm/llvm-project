//===- Specialize.cpp - linalg generic ops to named ops  ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a method to specialize generic operations to named
// operations. Conceptually it is the opposite of generalize.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-specialization"

#define REPLACE_BINARY_OP(NEWOP, OPERANDS_SWAP)                                \
  (rewriter.replaceOpWithNewOp<NEWOP>(                                         \
      genericOp,                                                               \
      ValueRange{genericOp.getDpsInputs()[(OPERANDS_SWAP) ? 1 : 0],            \
                 genericOp.getDpsInputs()[(OPERANDS_SWAP) ? 0 : 1]},           \
      ValueRange{genericOp.getDpsInits()[0]}))

#define REPLACE_UNARY_OP(NEWOP)                                                \
  (rewriter.replaceOpWithNewOp<NEWOP>(genericOp,                               \
                                      ValueRange{genericOp.getDpsInputs()[0]}, \
                                      ValueRange{genericOp.getDpsInits()[0]}))

using namespace mlir;
using namespace mlir::linalg;

// Given a elementwise single binary linalg generic op, checks whether the
// binary op accesses operands as swapped. e.g.
// this differentiates between a linalg-generic body that contains:
//    ^bb0(%a: f32, %b: f32, %c : f32):
//         %0 = arith.subf %a, %b : f32
//         linalg.yield %0: f32
// against:
//    ^bb0(%a: f32, %b: f32, %c : f32):
//         %0 = arith.subf %b, %a : f32
//         linalg.yield %0: f32
// Former is linalg.sub(a,b), latter is linalg.sub(b,a).
static bool areBinOpsSwapped(GenericOp genericOp) {
  Block *body = genericOp.getBody();
  Operation *op = &body->front();
  bool swapped = false;
  if (op->getOpOperand(0).get() != body->getArgument(0)) {
    swapped = true;
    assert(op->getOpOperand(0).get() == body->getArgument(1) &&
           op->getOpOperand(1).get() == body->getArgument(0) &&
           "binary op uses just one block arg");
  }
  return swapped;
}

FailureOr<LinalgOp> mlir::linalg::specializeGenericOp(RewriterBase &rewriter,
                                                      GenericOp genericOp) {
  if (isaCopyOpInterface(genericOp)) {
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<CopyOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0]);
    return namedOp;
  }

  if (isaFillOpInterface(genericOp)) {
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<FillOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0]);
    return namedOp;
  }

  if (isaElemwiseSingleUnaryOpInterface(genericOp)) {
    Operation *op = &genericOp.getBody()->front();
    if (isa<math::ExpOp>(op)) {
      LinalgOp namedOp = REPLACE_UNARY_OP(ExpOp);
      return namedOp;
    }
  }

  if (isaElemwiseSingleBinaryOpInterface(genericOp)) {
    bool swap = areBinOpsSwapped(genericOp);
    Operation *op = &genericOp.getBody()->front();
    if (isa<arith::AddFOp>(op)) {
      LinalgOp namedOp = REPLACE_BINARY_OP(AddOp, swap);
      return namedOp;
    }
    if (isa<arith::SubFOp>(op)) {
      LinalgOp namedOp = REPLACE_BINARY_OP(SubOp, swap);
      return namedOp;
    }
    if (isa<arith::MulFOp>(op)) {
      LinalgOp namedOp = REPLACE_BINARY_OP(MulOp, swap);
      return namedOp;
    }
    if (isa<arith::DivFOp>(op)) {
      LinalgOp namedOp = REPLACE_BINARY_OP(DivOp, swap);
      return namedOp;
    }
  }
  return failure();
}
