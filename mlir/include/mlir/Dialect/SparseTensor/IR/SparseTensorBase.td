//===- SparseTensorBase.td - Sparse tensor dialect base ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPARSETENSOR_BASE
#define SPARSETENSOR_BASE

include "mlir/IR/OpBase.td"

def SparseTensor_Dialect : Dialect {
  let name = "sparse_tensor";
  let cppNamespace = "::mlir::sparse_tensor";
  let description = [{
    The `sparse tensor` dialect is intended to hold primitives that
    form a bridge between high-level operations on sparse tensors
    and lower-level operations on the actual sparse storage schemes
    consisting of pointers, indices, and values. This bridge
    simplifies a `sparse compiler` pass by postponing actual
    code generation for the supported primitives to a later phase,
    either by generating calls into a runtime support library
    or by further lowering the primitives into actual code.
  }];
}

#endif // SPARSETENSOR_BASE
