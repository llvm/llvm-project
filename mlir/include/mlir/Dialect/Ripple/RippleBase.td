//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//
// This file contains definitions for the Ripple dialect.
//
//==============================================================================

#ifndef RIPPLE_BASE
#define RIPPLE_BASE

include "mlir/IR/OpBase.td"

def Ripple_Dialect : Dialect {
  let name = "ripple";
  let cppNamespace = "::mlir::ripple";
  let description = [{
        "Ripple is a compiler-interpreted API to express SPMD (Single Program, Multiple Data, as with CUDA and OpenCL) and loop-annotation parallelism (as with OpenMP)."
    }];
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

def Ripple_ScalarType
    : Type<Or<[AnySignlessInteger.predicate, AnyFloat.predicate]>,
           "signless integer or float">;

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

def Ripple_SameScalarShape
    : PredOpTrait<"input and result have matching scalar shape",
                  And<[TypeIsPred<"input", Ripple_ScalarType>,
                       TypeIsPred<"result", Ripple_ScalarType>,
                       AllShapesMatch<["input", "result"]>.predicate]>>;

def Ripple_SetSizeShape
    : PredOpTrait<"peid, dim, index have matching signless int shape",
                  And<[TypeIsPred<"peid", AnySignlessInteger>,
                       TypeIsPred<"dim", AnySignlessInteger>,
                       TypeIsPred<"index", AnySignlessInteger>,
                       AllShapesMatch<["peid", "dim", "index"]>.predicate]>>;

#endif // RIPPLE_BASE
