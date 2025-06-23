//===-- SPIRVImageOps.td - MLIR SPIR-V Image Ops ------*- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains image ops for the SPIR-V dialect. It corresponds
// to "3.56.10. Image Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_IMAGE_OPS
#define MLIR_DIALECT_SPIRV_IR_IMAGE_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Dialect/SPIRV/Interfaces/SPIRVImageInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

class SPIRV_ValuesAreContained<string operand, list<string> values, string transform, string type, string getter> :
  CPred<"::llvm::is_contained("
    "{::mlir::spirv::" # type # "::" # !interleave(values, ", ::mlir::spirv::" # type # "::") # "},"
    "::llvm::cast<::mlir::spirv::ImageType>(" # !subst("$_self", "$" # operand # ".getType()", transform) # ")." # getter # "()"
  ")"
>;

class SPIRV_SampledOperandIs<string operand, list<string> values, string transform="$_self"> : PredOpTrait<
  "the sampled operand of the underlying image must be " # !interleave(values, " or "),
  SPIRV_ValuesAreContained<operand, values, transform, "ImageSamplerUseInfo", "getSamplerUseInfo"> 
>;

class SPIRV_MSOperandIs<string operand, list<string> values, string transform="$_self"> : PredOpTrait<
  "the MS operand of the underlying image type must be " # !interleave(values, " or "),
  SPIRV_ValuesAreContained<operand, values, transform, "ImageSamplingInfo", "getSamplingInfo"> 
>;

class SPIRV_DimIs<string operand, list<string> values, string transform="$_self"> : PredOpTrait<
  "the Dim operand of the underlying image must be " # !interleave(values, " or "),
  SPIRV_ValuesAreContained<operand, values, transform, "Dim", "getDim">
>;

class SPIRV_DimIsNot<string operand, list<string> values, string transform="$_self"> : PredOpTrait<
  "the Dim operand of the underlying image must not be " # !interleave(values, " or "),
  Neg<SPIRV_ValuesAreContained<operand, values, transform, "Dim", "getDim">>
>;

class SPIRV_NoneOrElementMatchImage<string operand, string image, string transform="$_self"> : PredOpTrait<
  "the " # operand # " component type must match the image sampled type",
  CPred<"::llvm::isa<NoneType>(cast<::mlir::spirv::ImageType>(" # !subst("$_self", "$" # image # ".getType()", transform) # ").getElementType()) ||"
        "(getElementTypeOrSelf($" # operand # ")"
          "=="
        "cast<::mlir::spirv::ImageType>(" # !subst("$_self", "$" # image # ".getType()", transform) # ").getElementType())"
  >
>;

class SPIRV_ImageOperandIsPresent<string operand, list<string> values> : PredOpTrait<
  "either " # !interleave(values, " or ") # " image operands must be present",
  CPred<"::mlir::spirv::bitEnumContainsAny($" # operand # ", " # "::mlir::spirv::ImageOperands::" # !interleave(values, " | ::mlir::spirv::ImageOperands::") # ")">
>;

def SPIRV_SampledImageTransform : StrFunc<"llvm::cast<::mlir::spirv::SampledImageType>($_self).getImageType()">;

// -----

def SPIRV_ImageDrefGatherOp : SPIRV_Op<"ImageDrefGather", 
    [Pure,
     SPIRV_DimIs<"sampled_image", ["Dim2D", "Cube", "Rect"], SPIRV_SampledImageTransform.result>,
     SPIRV_MSOperandIs<"sampled_image", ["SingleSampled"], SPIRV_SampledImageTransform.result>,
     SPIRV_NoneOrElementMatchImage<"result", "sampled_image", SPIRV_SampledImageTransform.result>]>{
  let summary = "Gathers the requested depth-comparison from four texels.";

  let description = [{
    Result Type must be a vector of four components of floating-point type
    or integer type. Its components must be the same as Sampled Type of the
    underlying OpTypeImage (unless that underlying Sampled Type is
    OpTypeVoid). It has one component per gathered texel.

    Sampled Image must be an object whose type is OpTypeSampledImage. Its
    OpTypeImage must have a Dim of 2D, Cube, or Rect. The MS operand of the
    underlying OpTypeImage must be 0.

    Coordinate must be a scalar or vector of floating-point type. It
    contains (u[, v] ... [, array layer]) as needed by the definition of
    Sampled Image.

    Dref is the depth-comparison reference value. It must be a 32-bit
    floating-point type scalar.

    Image Operands encodes what operands follow, as per Image Operands.

    #### Example:

    ```mlir
    %0 = spirv.ImageDrefGather %1, %2, %3 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
    %0 = spirv.ImageDrefGather %1, %2, %3 ["NonPrivateTexel"] : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Shader]>
  ];

  let arguments = (ins
    SPIRV_AnySampledImage:$sampled_image,
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$coordinate,
    SPIRV_Float32:$dref,
    OptionalAttr<SPIRV_ImageOperandsAttr>:$image_operands,
    Variadic<SPIRV_Type>:$operand_arguments
  );

  let results = (outs
    AnyTypeOf<[SPIRV_Vec4<SPIRV_Integer>, SPIRV_Vec4<SPIRV_Float>]>:$result
  );


  let assemblyFormat = [{
    $sampled_image `,` $coordinate `,` $dref custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
    `:` type($sampled_image) `,` type($coordinate) `,` type($dref) ( `,` type($operand_arguments)^ )?
    `->` type($result) 
  }];

}

// -----

def SPIRV_ImageQuerySizeOp : SPIRV_Op<"ImageQuerySize", [Pure]> {
  let summary = "Query the dimensions of Image, with no level of detail.";

  let description = [{
    Result Type must be an integer type scalar or vector. The number of
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
    %3 = spirv.ImageQuerySize %0 : !spirv.image<i32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> i32
    %4 = spirv.ImageQuerySize %1 : !spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
    %5 = spirv.ImageQuerySize %2 : !spirv.image<i32, Dim2D, NoDepth, Arrayed, SingleSampled, NoSampler, Unknown> -> vector<3xi32>
    ```

  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_ImageQuery, SPIRV_C_Kernel]>
  ];

  let arguments = (ins
    SPIRV_AnyImage:$image
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$result
  );

  let assemblyFormat = "$image attr-dict `:` type($image) `->` type($result)";
}

// -----

def SPIRV_ImageReadOp : SPIRV_Op<"ImageRead",
    [SPIRV_SampledOperandIs<"image", ["SamplerUnknown", "NoSampler"]>,
     SPIRV_NoneOrElementMatchImage<"result", "image">]> {
  let summary = "Read a texel from an image without a sampler.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type or integer
    type. It must be a scalar or vector with component type the same as Sampled
    Type of the OpTypeImage (unless that Sampled Type is OpTypeVoid).

    Image must be an object whose type is OpTypeImage with a Sampled operand of
    0 or 2. If the Arrayed operand is 1, then additional capabilities may be
    required; e.g., ImageCubeArray, or ImageMSArray.

    Coordinate must be a scalar or vector of floating-point type or integer
    type. It contains non-normalized texel coordinates (u[, v] ... [, array
    layer]) as needed by the definition of Image. See the client API
    specification for handling of coordinates outside the image.

    If the Image Dim operand is SubpassData, Coordinate is relative to the
    current fragment location. See the client API specification for more detail
    on how these coordinates are applied.

    If the Image Dim operand is not SubpassData, the Image Format must not be
    Unknown, unless the StorageImageReadWithoutFormat Capability was declared.

    Image Operands encodes what operands follow, as per Image Operands.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.ImageRead %1, %2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, R32f>, vector<2xsi32> -> vector<4xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_AnyImage:$image,
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>]>:$coordinate,
    OptionalAttr<SPIRV_ImageOperandsAttr>:$image_operands,
    Variadic<SPIRV_Type>:$operand_arguments
  );

  let results = (outs
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>]>:$result
  );

  let assemblyFormat = [{
    $image `,` $coordinate custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
    `:` type($image) `,` type($coordinate) ( `,` type($operand_arguments)^ )?
    `->` type($result)
  }];
}

// -----

def SPIRV_ImageWriteOp : SPIRV_Op<"ImageWrite",
    [SPIRV_SampledOperandIs<"image", ["SamplerUnknown", "NoSampler"]>,
     SPIRV_DimIsNot<"image", ["SubpassData"]>,
     SPIRV_NoneOrElementMatchImage<"texel", "image">]> {
  let summary = "Write a texel to an image without a sampler.";

  let description = [{
    Image must be an object whose type is OpTypeImage with a Sampled operand
    of 0 or 2. If the Arrayed operand is 1, then additional capabilities may
    be required; e.g., ImageCubeArray, or ImageMSArray. Its Dim operand
    must not be SubpassData.

    Coordinate must be a scalar or vector of floating-point type or integer
    type. It contains non-normalized texel coordinates (u[, v] ... [, array
    layer]) as needed by the definition of Image. See the client API
    specification for handling of coordinates outside the image.

    Texel is the data to write. It must be a scalar or vector with component
    type the same as Sampled Type of the OpTypeImage (unless that Sampled
    Type is OpTypeVoid).

    The Image Format must not be Unknown, unless the
    StorageImageWriteWithoutFormat Capability was declared.

    Image Operands encodes what operands follow, as per Image Operands.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    spirv.ImageWrite %0, %1, %2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, vector<2xsi32>, vector<4xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_AnyImage:$image,
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>]>:$coordinate,
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>]>:$texel,
    OptionalAttr<SPIRV_ImageOperandsAttr>:$image_operands,
    Variadic<SPIRV_Type>:$operand_arguments
  );

  let results = (outs);

  let assemblyFormat = [{
    $image `,` $coordinate `,` $texel custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
    `:` type($image) `,` type($coordinate) `,` type($texel) ( `,` type($operand_arguments)^ )?
  }];
}

// -----

def SPIRV_ImageOp : SPIRV_Op<"Image",
    [Pure,
     TypesMatchWith<"type of 'result' matches image type of 'sampled_image'",
                    "sampled_image", "result",
                    "::llvm::cast<spirv::SampledImageType>($_self).getImageType()">]> {
  let summary = "Extract the image from a sampled image.";

  let description = [{
    Result Type must be OpTypeImage.

    Sampled Image must have type OpTypeSampledImage whose Image Type is the
    same as Result Type.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.Image %1 : !spirv.sampled_image<!spirv.image<f32, Cube, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>
    ```
  }];

  let arguments = (ins
    SPIRV_AnySampledImage:$sampled_image
  );

  let results = (outs
    SPIRV_AnyImage:$result
  );

  let assemblyFormat = "$sampled_image attr-dict `:` type($sampled_image)";

  let hasVerifier = 0;
}

// -----

def SPIRV_ImageSampleExplicitLodOp : SPIRV_Op<"ImageSampleExplicitLod",
    [SPIRV_DimIsNot<"sampled_image", ["Buffer"], SPIRV_SampledImageTransform.result>,
     SPIRV_MSOperandIs<"sampled_image", ["SingleSampled"], SPIRV_SampledImageTransform.result>,
     SPIRV_NoneOrElementMatchImage<"result", "sampled_image", SPIRV_SampledImageTransform.result>,
     SPIRV_ImageOperandIsPresent<"image_operands", ["Lod", "Grad"]>,
     DeclareOpInterfaceMethods<SPIRV_ExplicitLodOpInterface>]> {
  let summary = "Sample an image using an explicit level of detail.";

  let description = [{
    Result Type must be a vector of four components of floating-point type
    or integer type. Its components must be the same as Sampled Type of the
    underlying OpTypeImage (unless that underlying Sampled Type is
    OpTypeVoid).

    Sampled Image must be an object whose type is OpTypeSampledImage. Its
    OpTypeImage must not have a Dim of Buffer. The MS operand of the
    underlying OpTypeImage must be 0.

    Coordinate must be a scalar or vector of floating-point type or integer
    type. It contains (u[, v] ... [, array layer]) as needed by the
    definition of Sampled Image. Unless the Kernel capability is declared,
    it must be floating point. It may be a vector larger than needed, but
    all unused components appear after all used components.

    Image Operands encodes what operands follow, as per Image Operands.
    Either Lod or Grad image operands must be present.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %result = spirv.ImageSampleExplicitLod %image, %coord ["Lod"](%lod) :
      !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>,
      vector<2xf32> (f32) -> vector<4xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_AnySampledImage:$sampled_image,
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>]>:$coordinate,
    SPIRV_ImageOperandsAttr:$image_operands,
    Variadic<SPIRV_Type>:$operand_arguments
  );

  let results = (outs
    AnyTypeOf<[SPIRV_Vec4<SPIRV_Integer>, SPIRV_Vec4<SPIRV_Float>]>:$result
  );

  let assemblyFormat = [{
    $sampled_image `,` $coordinate custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
      `:` type($sampled_image) `,` type($coordinate) ( `,` type($operand_arguments)^ )?
      `->` type($result)
  }];
}

// -----

def SPIRV_ImageSampleImplicitLodOp : SPIRV_Op<"ImageSampleImplicitLod",
    [SPIRV_DimIsNot<"sampled_image", ["Buffer"], SPIRV_SampledImageTransform.result>,
     SPIRV_MSOperandIs<"sampled_image", ["SingleSampled"], SPIRV_SampledImageTransform.result>,
     SPIRV_NoneOrElementMatchImage<"result", "sampled_image", SPIRV_SampledImageTransform.result>,
     DeclareOpInterfaceMethods<SPIRV_ImplicitLodOpInterface>]> {
  let summary = "Sample an image with an implicit level of detail.";

  let description = [{
    An invocation will not execute a dynamic instance of this instruction
    (X') until all invocations in its derivative group have executed all
    dynamic instances that are program-ordered before X'.

    Result Type must be a vector of four components of floating-point type
    or integer type. Its components must be the same as Sampled Type of the
    underlying OpTypeImage (unless that underlying Sampled Type is
    OpTypeVoid).

    Sampled Image must be an object whose type is OpTypeSampledImage. Its
    OpTypeImage must not have a Dim of Buffer. The MS operand of the
    underlying OpTypeImage must be 0.

    Coordinate must be a scalar or vector of floating-point type. It
    contains (u[, v] ... [, array layer]) as needed by the definition of
    Sampled Image. It may be a vector larger than needed, but all unused
    components appear after all used components.

    Image Operands encodes what operands follow, as per Image Operands.

    This instruction is only valid in the Fragment Execution Model. In
    addition, it consumes an implicit derivative that can be affected by
    code motion.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %result = spirv.ImageSampleImplicitLod %image, %coord :
      !spirv.sampled_image<!spirv.image<f32, Cube, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>,
      vector<3xf32> -> vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Shader]>
  ];

  let arguments = (ins
    SPIRV_AnySampledImage:$sampled_image,
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$coordinate,
    OptionalAttr<SPIRV_ImageOperandsAttr>:$image_operands,
    Variadic<SPIRV_Type>:$operand_arguments
  );

  let results = (outs
    AnyTypeOf<[SPIRV_Vec4<SPIRV_Integer>, SPIRV_Vec4<SPIRV_Float>]>:$result
  );

  let assemblyFormat = [{
    $sampled_image `,` $coordinate custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
      `:` type($sampled_image) `,` type($coordinate) ( `,` type($operand_arguments)^ )?
      `->` type($result)
  }];
}

// -----

def SPIRV_ImageSampleProjDrefImplicitLodOp : SPIRV_Op<"ImageSampleProjDrefImplicitLod",
    [SPIRV_DimIsNot<"sampled_image", ["Buffer"], SPIRV_SampledImageTransform.result>,
     SPIRV_MSOperandIs<"sampled_image", ["SingleSampled"], SPIRV_SampledImageTransform.result>,
     TypesMatchWith<"type of 'result' matches image type of 'sampled_image'",
                    "sampled_image", "result",
                    "::llvm::cast<::mlir::spirv::ImageType>(::llvm::cast<spirv::SampledImageType>($_self).getImageType()).getElementType()">,
     DeclareOpInterfaceMethods<SPIRV_ImplicitLodOpInterface>]> {

  let summary = [{
    Sample an image with a project coordinate, doing depth-comparison, with
    an implicit level of detail.
  }];

  let description = [{
    An invocation will not execute a dynamic instance of this instruction
    (X') until all invocations in its derivative group have executed all
    dynamic instances that are program-ordered before X'.

    Result Type must be a scalar of integer type or floating-point type. It
    must be the same as Sampled Type of the underlying OpTypeImage.

    Sampled Image must be an object whose type is OpTypeSampledImage. The
    Dim operand of the underlying OpTypeImage must be 1D, 2D, 3D, or Rect,
    and the Arrayed and MS operands must be 0.

    Coordinate must be a vector of floating-point type. It contains (u[,
    v] [, w], q), as needed by the definition of Sampled Image, with the q
    component consumed for the projective division. That is, the actual
    sample coordinate is (u/q [, v/q] [, w/q]), as needed by the definition
    of Sampled Image. It may be a vector larger than needed, but all unused
    components appear after all used components.

    Dref/q is the depth-comparison reference value. Dref must be a 32-bit
    floating-point type scalar.

    Image Operands encodes what operands follow, as per Image Operands.

    This instruction is only valid in the Fragment Execution Model. In
    addition, it consumes an implicit derivative that can be affected by
    code motion.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %result = spirv.ImageSampleProjDrefImplicitLod %image, %coord, %dref :
      !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>,
      vector<4xf16>, f32 -> f32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Shader]>
  ];

  let arguments = (ins
    SPIRV_AnySampledImage:$sampled_image,
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>]>:$coordinate,
    SPIRV_Float32:$dref,
    OptionalAttr<SPIRV_ImageOperandsAttr>:$image_operands,
    Variadic<SPIRV_Type>:$operand_arguments
  );

  let results = (outs
    AnyTypeOf<[SPIRV_Integer, SPIRV_Float]>:$result
  );

  let assemblyFormat = [{
    $sampled_image `,` $coordinate `,` $dref custom<ImageOperands>($image_operands) ( `,` $operand_arguments^ )? attr-dict
      `:` type($sampled_image) `,` type($coordinate) `,` type($dref) ( `,` type($operand_arguments)^ )?
      `->` type($result)
  }];
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_IMAGE_OPS
