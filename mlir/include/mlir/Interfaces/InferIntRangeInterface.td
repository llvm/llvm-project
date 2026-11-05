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
    Allows operations to specify lower and upper bounds on their integer results
    from known bounds on their inputs.

    An implementation may defer a result while information needed to infer its
    range is pending. This information can include operand ranges, values yielded
    by regions, and other operation-specific dependencies. Initialized operand
    ranges alone do not establish that all required information is available.

    Once the required information is available, the implementation must report
    an initialized range for each integer result or block argument that it is
    responsible for. If it cannot infer bounds, it must report a maximal range.
    An empty range represents poison. An uninitialized range must not represent
    an unknown but otherwise arbitrary integer value.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
      Infer the bounds on the results of this op given the bounds on its arguments.
      Call `setResultRanges` for each integer result or non-branch block argument
      whose required information is available. The dataflow analysis handles
      branch arguments.
      This method requires integer ranges for all operands. Operations with
      non-integer operands must implement `inferResultRangesFromOptional`.
      Poison is represented by an empty range. Each operation defines how
      poison propagates to its results.

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
      of the bounds for its arguments. Call `setResultRanges` for each integer
      result or non-branch block argument whose required information is available.
      The dataflow analysis handles branch arguments.

      Each input has the same position as its operand. Non-integer operands
      have no range; use operand types to identify them. For integer operands,
      an uninitialized range means that the analysis has no result yet. It must
      not be treated as poison or as an arbitrary integer. Use this method when
      results can be inferred before all integer operand ranges are initialized.
      Only report ranges for integer results or block arguments.
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
