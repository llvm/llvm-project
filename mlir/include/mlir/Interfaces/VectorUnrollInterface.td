//===- VectorUnrollInterface.td - VectorUnroll interface ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for operations on vectors that can be unrolled.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VECTORUNROLLINTERFACE
#define MLIR_INTERFACES_VECTORUNROLLINTERFACE

include "mlir/IR/OpBase.td"

def VectorUnrollOpInterface : OpInterface<"VectorUnrollOpInterface"> {
  let description = [{
    Encodes properties of an operation on vectors that can be unrolled.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the shape ratio of unrolling to the target vector shape
        `targetShape`. Returns `None` if the op cannot be unrolled to the target
        vector shape.
      }],
      "Optional<SmallVector<int64_t, 4>>",
      "getShapeForUnroll",
      (ins),
      /*methodBody=*/[{}],
      [{
        auto vt = this->getOperation()->getResult(0).getType().
          template dyn_cast<VectorType>();
        if (!vt)
          return None;
        SmallVector<int64_t, 4> res(vt.getShape().begin(), vt.getShape().end());
        return res;
      }]
    >,
  ];
}

#endif // MLIR_INTERFACES_VECTORUNROLLINTERFACE
