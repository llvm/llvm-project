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

def SPV_ImageDrefGatherOp : SPV_Op<"ImageDrefGather", [NoSideEffect]> {
  let summary = "Gathers the requested depth-comparison from four texels.";

  let description = [{
    Result Type must be a vector of four components of floating-point type
    or integer type.  Its components must be the same as Sampled Type of the
    underlying OpTypeImage (unless that underlying Sampled Type is
    OpTypeVoid). It has one component per gathered texel.

    Sampled Image must be an object whose type is OpTypeSampledImage. Its
    OpTypeImage must have a Dim of 2D, Cube, or Rect. The MS operand of the
    underlying OpTypeImage must be 0.

    Coordinate  must be a scalar or vector of floating-point type.  It
    contains (u[, v] … [, array layer]) as needed by the definition of
    Sampled Image.

    Dref is the depth-comparison reference value. It must be a 32-bit
    floating-point type scalar.

    Image Operands encodes what operands follow, as per Image Operands.

    <!-- End of AutoGen section -->
    ```
    image-operands ::= `"None"` | `"Bias"` | `"Lod"` | `"Grad"`
                      | `"ConstOffset"` | `"Offser"` | `"ConstOffsets"`
                      | `"Sample"` | `"MinLod"` | `"MakeTexelAvailable"`
                      | `"MakeTexelVisible"` | `"NonPrivateTexel"`
                      | `"VolatileTexel"` | `"SignExtend"` | `"ZeroExtend"`
    #### Example:
    ```

    ```mlir
    %0 = spv.ImageDrefGather %1 : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %2 : vector<4xf32>, %3 : f32 -> vector<4xi32>
    %0 = spv.ImageDrefGather %1 : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %2 : vector<4xf32>, %3 : f32 ["NonPrivateTexel"] : f32, f32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Shader]>
  ];

  let arguments = (ins
    SPV_AnySampledImage:$sampledimage,
    SPV_ScalarOrVectorOf<SPV_Float>:$coordinate,
    SPV_Float:$dref,
    OptionalAttr<SPV_ImageOperandAttr>:$imageoperands,
    Variadic<SPV_Type>:$operand_arguments
  );

  let results = (outs
    SPV_Vector:$result
  );

  let assemblyFormat = [{$sampledimage `:` type($sampledimage) `,`
                         $coordinate `:` type($coordinate) `,` $dref `:` type($dref)
                         custom<ImageOperands>($imageoperands)
                         ( `(` $operand_arguments^ `:` type($operand_arguments) `)`)?
                         attr-dict
                         `->` type($result)}];

  let verifier = [{ return ::verify(*this); }];
}

// -----

def SPV_ImageQuerySizeOp : SPV_Op<"ImageQuerySize", [NoSideEffect]> {
  let summary = "Query the dimensions of Image, with no level of detail.";

  let description = [{
    Result Type must be an integer type scalar or vector.  The number of
    components must be:

    1 for the 1D and Buffer dimensionalities,

    2 for the 2D, Cube, and Rect dimensionalities,

    3 for the 3D dimensionality,

    plus 1 more if the image type is arrayed. This vector is filled in with
    (width [, height] [, elements]) where elements is the number of layers
    in an image array or the number of cubes in a cube-map array.

    Image must be an object whose type is OpTypeImage. Its Dim operand must
    be one of those listed under Result Type, above. Additionally, if its
    Dim is 1D, 2D, 3D, or Cube, it must also have either an MS of 1 or a
    Sampled of 0 or 2. There is no implicit level-of-detail consumed by this
    instruction. See OpImageQuerySizeLod for querying images having level of
    detail. This operation is allowed on an image decorated as NonReadable.
    See the client API specification for additional image type restrictions.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %3 = spv.ImageQuerySize %0 : !spv.image<i32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> i32
    %4 = spv.ImageQuerySize %1 : !spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
    %5 = spv.ImageQuerySize %2 : !spv.image<i32, Dim2D, NoDepth, Arrayed, SingleSampled, NoSampler, Unknown> -> vector<3xi32>
    ```

  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_ImageQuery, SPV_C_Kernel]>
  ];

  let arguments = (ins
    SPV_AnyImage:$image
  );

  let results = (outs
    SPV_ScalarOrVectorOf<SPV_Integer>:$result
  );

  let assemblyFormat = "attr-dict $image `:` type($image) `->` type($result)";

  let verifier = [{return ::verify(*this);}];
}

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
