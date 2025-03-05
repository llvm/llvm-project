//===- InferIntRangeInterface.td - Integer Range Inference --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//
//
// Defines the interface for range analysis on scalar integers
//
//===-----------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERINTRANGEINTERFACE
#define MLIR_INTERFACES_INFERINTRANGEINTERFACE

include "mlir/IR/OpBase.td"

def InferIntRangeInterface : OpInterface<"InferIntRangeInterface"> {
  let description = [{
    Allows operations to participate in range analysis for scalar integer values by
    providing a methods that allows them to specify lower and upper bounds on their
    result(s) given lower and upper bounds on their input(s) if known.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
      Infer the bounds on the results of this op given the bounds on its arguments.
      For each result value or block argument (that isn't a branch argument,
      since the dataflow analysis handles those case), the method should call
      `setValueRange` with that `Value` as an argument. When implemented,
      `setValueRange` should be called on all result values for the operation.
      When operations take non-integer inputs, the
     `inferResultRangesFromOptional` method should be implemented instead.

      When called on an op that also implements the RegionBranchOpInterface
      or BranchOpInterface, this method should not attempt to infer the values
      of the branch results, as this will be handled by the analyses that use
      this interface.

      This function will only be called when at least one result of the op is a
      scalar integer value or the op has a region.
    }],
    /*retTy=*/"void",
    /*methodName=*/"inferResultRanges",
    /*args=*/(ins "::llvm::ArrayRef<::mlir::ConstantIntRanges>":$argRanges,
                  "::mlir::SetIntRangeFn":$setResultRanges),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      ::mlir::intrange::detail::defaultInferResultRangesFromOptional($_op,
                                                                     argRanges,
                                                                     setResultRanges);
    }]>,

    InterfaceMethod<[{
      Infer the bounds on the results of this op given the lattice representation
      of the bounds for its arguments. For each result value or block argument
      (that isn't a branch argument, since the dataflow analysis handles
      those case), the method should call `setValueRange` with that `Value`
      as an argument. When implemented, `setValueRange` should be called on
      all result values for the operation.

      This method allows for more precise implementations when operations
      want to reason about inputs which may be undefined during the analysis.
    }],
    /*retTy=*/"void",
    /*methodName=*/"inferResultRangesFromOptional",
    /*args=*/(ins "::llvm::ArrayRef<::mlir::IntegerValueRange>":$argRanges,
                  "::mlir::SetIntLatticeFn":$setResultRanges),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      ::mlir::intrange::detail::defaultInferResultRanges($_op,
                                                         argRanges,
                                                         setResultRanges);
    }]>
  ];
}
#endif // MLIR_INTERFACES_INFERINTRANGEINTERFACE
