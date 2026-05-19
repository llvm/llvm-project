//===- InferStridedMetadataInterface.td - Strided MD Inference ----------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for strided metadata range analysis
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE
#define MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE

include "mlir/IR/OpBase.td"

def InferStridedMetadataOpInterface :
    OpInterface<"InferStridedMetadataOpInterface"> {
  let description = [{
    Allows operations to participate in strided metadata analysis by providing
    methods that allow them to specify bounds on offsets, sizes, and strides
    of their result(s) given bounds on their input(s) if known.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
      Infer the strided metadata bounds on the results of this op given
      the bounds on its operands.
      For each result value or block argument of interest, the method should
      call `setMetadata` with that `Value` as an argument.
      The `operands` parameter contains the strided metadata ranges for all the
      operands of the operation in order.
      The `getIntRange` callback is provided for obtaining the int-range
      analysis result for a given value.
    }],
    "void", "inferStridedMetadataRanges",
    (ins "::llvm::ArrayRef<::mlir::StridedMetadataRange>":$operands,
         "::mlir::GetIntRangeFn":$getIntRange,
         "::mlir::SetStridedMetadataRangeFn":$setMetadata,
         "int32_t":$indexBitwidth)>
  ];
}
#endif // MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE
