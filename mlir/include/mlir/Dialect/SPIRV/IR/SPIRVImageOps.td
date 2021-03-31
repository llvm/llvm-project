//===-- SPIRVImageOps.td - MLIR SPIR-V Image Ops ------*- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains image ops for the SPIR-V dialect. It corresponds
// to "3.37.10. Image Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_IMAGE_OPS
#define MLIR_DIALECT_SPIRV_IR_IMAGE_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

def SPV_ImageOp : SPV_Op<"Image",
    [NoSideEffect,
     TypesMatchWith<"type of 'result' matches image type of 'sampledimage'",
                    "sampledimage", "result",
                    "$_self.cast<spirv::SampledImageType>().getImageType()">]> {
  let summary = "Extract the image from a sampled image.";

  let description = [{
    Result Type must be OpTypeImage.

    Sampled Image must have type OpTypeSampledImage whose Image Type is the
    same as Result Type.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spv.Image %1 : !spv.sampled_image<!spv.image<f32, Cube, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>
    ```
  }];

  let arguments = (ins
    SPV_AnySampledImage:$sampledimage
  );

  let results = (outs
    SPV_AnyImage:$result
  );

  let assemblyFormat = "attr-dict $sampledimage `:` type($sampledimage)";

  let verifier = ?;

}

#endif // MLIR_DIALECT_SPIRV_IR_GLSL_OPS
