//===-- ValueBoundsOpInterface.td - Value Bounds -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VALUEBOUNDSOPINTERFACE
#define VALUEBOUNDSOPINTERFACE

include "mlir/IR/OpBase.td"

def ValueBoundsOpInterface : OpInterface<"ValueBoundsOpInterface"> {
  let description = [{
    This interface allows operations with index-typed and/or shaped value-typed
    results/block arguments to specify range bounds. These bounds are stored in
    a constraint set. The constraint set can then be queried to compute bounds
    in terms of other values that are stored in the constraint set.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Populate the constraint set with bounds for the given index-typed
        value.

        Note: If `value` is a block argument, it must belong to an entry block
        of a region. Unstructured control flow graphs are not supported at the
        moment.
      }],
      /*retType=*/"void",
      /*methodName=*/"populateBoundsForIndexValue",
      /*args=*/(ins "::mlir::Value":$value,
                    "::mlir::ValueBoundsConstraintSet &":$cstr),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Populate the constraint set with bounds for the size of the specified
        dimension of the given shaped value.

        Note: If `value` is a block argument, it must belong to an entry block
        of a region. Unstructured control flow graphs are not supported at the
        moment.
      }],
      /*retType=*/"void",
      /*methodName=*/"populateBoundsForShapedValueDim",
      /*args=*/(ins "::mlir::Value":$value,
                    "int64_t":$dim,
                    "::mlir::ValueBoundsConstraintSet &":$cstr),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("not implemented");
      }]
    >,
  ];
}

#endif  // VALUEBOUNDSOPINTERFACE
