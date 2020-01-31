//===-- OpenMPOps.td - OpenMP dialect operation definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the OpenMP dialect.
//
//===----------------------------------------------------------------------===//


#ifndef OPENMP_OPS
#define OPENMP_OPS

include "mlir/IR/OpBase.td"

def OpenMP_Dialect : Dialect {
  let name = "omp";
}

class OpenMP_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<OpenMP_Dialect, mnemonic, traits>;

def BarrierOp : OpenMP_Op<"barrier"> {
  let summary = "barrier construct";
  let description = [{
    The barrier construct specifies an explicit barrier at the point at which
    the construct appears.
  }];

  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
}

#endif // OPENMP_OPS
