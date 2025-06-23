// RUN: mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ImageDrefGather
//===----------------------------------------------------------------------===//

func.func @image_dref_gather(%arg0 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // CHECK: spirv.ImageDrefGather {{.*}}, {{.*}}, {{.*}} : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
  %0 = spirv.ImageDrefGather %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
  spirv.Return
}

// -----

func.func @image_dref_gather_with_single_imageoperands(%arg0 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // CHECK: spirv.ImageDrefGather {{.*}}, {{.*}}, {{.*}} ["NonPrivateTexel"] : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
  %0 = spirv.ImageDrefGather %arg0, %arg1, %arg2 ["NonPrivateTexel"] : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
  spirv.Return
}

// -----

func.func @image_dref_gather_with_mismatch_imageoperands(%arg0 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the Image Operands should encode what operands follow, as per Image Operands}}
  %0 = spirv.ImageDrefGather %arg0, %arg1, %arg2, %arg2, %arg2 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32, f32, f32 -> vector<4xi32>
  spirv.Return
}

// -----

func.func @image_dref_gather_error_result_type(%arg0 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{must be vector of 8/16/32/64-bit integer values of length 4 or vector of 16/32/64-bit float values of length 4}}
  %0 = spirv.ImageDrefGather %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<3xi32>
  spirv.Return
}

// -----

func.func @image_dref_gather_error_same_type(%arg0 : !spirv.sampled_image<!spirv.image<i32, Rect, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the result component type must match the image sampled type}}
  %0 = spirv.ImageDrefGather %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<i32, Rect, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @image_dref_gather_error_dim(%arg0 : !spirv.sampled_image<!spirv.image<i32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the Dim operand of the underlying image must be Dim2D or Cube or Rect}}
  %0 = spirv.ImageDrefGather %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<i32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
  spirv.Return
}

// -----

func.func @image_dref_gather_error_ms(%arg0 : !spirv.sampled_image<!spirv.image<i32, Cube, NoDepth, NonArrayed, MultiSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the MS operand of the underlying image type must be SingleSampled}}
  %0 = spirv.ImageDrefGather %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<i32, Cube, NoDepth, NonArrayed, MultiSampled, NoSampler, Unknown>>, vector<4xf32>, f32 -> vector<4xi32>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.Image
//===----------------------------------------------------------------------===//

func.func @image(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>) -> () {
  // CHECK: spirv.Image {{.*}} : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
  %0 = spirv.Image %arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
  return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageQuerySize
//===----------------------------------------------------------------------===//

func.func @image_query_size(%arg0 : !spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>) -> () {
  // CHECK:  {{%.*}} = spirv.ImageQuerySize %arg0 : !spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> i32
  %0 = spirv.ImageQuerySize %arg0 : !spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> i32
  spirv.Return
}

// -----

func.func @image_query_size_error_dim(%arg0 : !spirv.image<f32, SubpassData, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>) -> () {
  //  expected-error @+1 {{the Dim operand of the image type must be 1D, 2D, 3D, Buffer, Cube, or Rect}}
  %0 = spirv.ImageQuerySize %arg0 : !spirv.image<f32, SubpassData, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> i32
  spirv.Return
}

// -----

func.func @image_query_size_error_dim_sample(%arg0 : !spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>) -> () {
  //  expected-error @+1 {{if Dim is 1D, 2D, 3D, or Cube, it must also have either an MS of 1 or a Sampled of 0 or 2}}
  %0 = spirv.ImageQuerySize %arg0 : !spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown> -> i32
  spirv.Return
}

// -----

func.func @image_query_size_error_result1(%arg0 : !spirv.image<f32, Dim3D, NoDepth, Arrayed, SingleSampled, NoSampler, Unknown>) -> () {
  //  expected-error @+1 {{expected the result to have 4 component(s), but found 3 component(s)}}
  %0 = spirv.ImageQuerySize %arg0 : !spirv.image<f32, Dim3D, NoDepth, Arrayed, SingleSampled, NoSampler, Unknown> -> vector<3xi32>
  spirv.Return
}

// -----

func.func @image_query_size_error_result2(%arg0 : !spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>) -> () {
  //  expected-error @+1 {{expected the result to have 1 component(s), but found 2 component(s)}}
  %0 = spirv.ImageQuerySize %arg0 : !spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown> -> vector<2xi32>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageRead
//===----------------------------------------------------------------------===//

func.func @image_read(%arg0: !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba8>, %arg1: vector<2xsi32>) -> () {
  // CHECK: {{%.*}} = spirv.ImageRead {{%.*}}, {{%.*}} : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba8>, vector<2xsi32> -> vector<4xf32>
  %0 = spirv.ImageRead %arg0, %arg1 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba8>, vector<2xsi32> -> vector<4xf32>
  spirv.Return
}

// -----

func.func @image_read_type_mismatch(%arg0: !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba8>, %arg1: vector<2xsi32>) -> () {
  // expected-error @+1 {{op failed to verify that the result component type must match the image sampled type}}
  %0 = spirv.ImageRead %arg0, %arg1 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba8>, vector<2xsi32> -> vector<4xf16>
  spirv.Return
}

// -----

func.func @image_read_need_sampler(%arg0: !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>, %arg1: vector<2xsi32>) -> () {
  // expected-error @+1 {{op failed to verify that the sampled operand of the underlying image must be SamplerUnknown or NoSampler}}
  %0 = spirv.ImageRead %arg0, %arg1 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>, vector<2xsi32> -> vector<4xf16>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageWrite
//===----------------------------------------------------------------------===//

func.func @image_write(%arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>) -> () {
  // CHECK:  spirv.ImageWrite {{%.*}}, {{%.*}}, {{%.*}} : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, vector<2xsi32>, vector<4xf32>
  spirv.ImageWrite %arg0, %arg1, %arg2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, vector<2xsi32>, vector<4xf32>
  spirv.Return
}

// -----

func.func @image_write_scalar_texel(%arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, %arg1 : vector<2xsi32>, %arg2 : f32) -> () {
  // CHECK:  spirv.ImageWrite {{%.*}}, {{%.*}}, {{%.*}} : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, vector<2xsi32>, f32
  spirv.ImageWrite %arg0, %arg1, %arg2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, vector<2xsi32>, f32
  spirv.Return
}

// -----

func.func @image_write_need_sampler(%arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba16>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>) -> () {
  // expected-error @+1 {{the sampled operand of the underlying image must be SamplerUnknown or NoSampler}}
  spirv.ImageWrite %arg0, %arg1, %arg2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba16>, vector<2xsi32>, vector<4xf32>
  spirv.Return
}

// -----

func.func @image_write_subpass_data(%arg0 : !spirv.image<f32, SubpassData, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, %arg1 : vector<2xsi32>, %arg2 : vector<4xf32>) -> () {
  // expected-error @+1 {{the Dim operand of the underlying image must not be SubpassData}}
  spirv.ImageWrite %arg0, %arg1, %arg2 : !spirv.image<f32, SubpassData, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, vector<2xsi32>, vector<4xf32>
  spirv.Return
}

// -----

func.func @image_write_texel_type_mismatch(%arg0 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, %arg1 : vector<2xsi32>, %arg2 : vector<4xi32>) -> () {
  // expected-error @+1 {{the texel component type must match the image sampled type}}
  spirv.ImageWrite %arg0, %arg1, %arg2 : !spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Rgba16>, vector<2xsi32>, vector<4xi32>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageSampleExplicitLod
//===----------------------------------------------------------------------===//

func.func @sample_explicit_lod(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // CHECK: {{%.*}} = spirv.ImageSampleExplicitLod {{%.*}}, {{%.*}} ["Lod"], {{%.*}} : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32 -> vector<4xf32>
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Lod"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @sample_explicit_lod_buffer_dim(%arg0 : !spirv.sampled_image<!spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the Dim operand of the underlying image must not be Buffer}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Lod"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @sample_explicit_lod_multi_sampled(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, MultiSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the MS operand of the underlying image type must be SingleSampled}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Lod"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, MultiSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @sample_explicit_lod_wrong_result_type(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the result component type must match the image sampled type}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Lod"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32 -> vector<4xsi32>
  spirv.Return
}

// -----

func.func @sample_explicit_lod_no_lod(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{either Lod or Grad image operands must be present}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Bias"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32 -> vector<4xf32>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageSampleImplicitLod
//===----------------------------------------------------------------------===//

func.func @sample_implicit_lod(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>) -> () {
  // CHECK: {{%.*}} = spirv.ImageSampleImplicitLod {{%.*}}, {{%.*}} : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32> -> vector<4xf32>
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32> -> vector<4xf32>
  spirv.Return
}

// -----

func.func @sample_implicit_lod_buffer(%arg0 : !spirv.sampled_image<!spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>) -> () {
  // expected-error @+1 {{the Dim operand of the underlying image must not be Buffer}}
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 : !spirv.sampled_image<!spirv.image<f32, Buffer, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32> -> vector<4xf32>
  spirv.Return
}

// -----

func.func @sample_implicit_lod_multi_sampled(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, MultiSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>) -> () {
  // expected-error @+1 {{the MS operand of the underlying image type must be SingleSampled}}
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, MultiSampled, NeedSampler, Rgba8>>, vector<2xf32> -> vector<4xf32>
  spirv.Return
}

// -----

func.func @sample_implicit_lod_wrong_result(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>) -> () {
  // expected-error @+1 {{the result component type must match the image sampled type}}
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32> -> vector<4xi32>
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageSampleProjDrefImplicitLod
//===----------------------------------------------------------------------===//

func.func @sample_implicit_proj_dref(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // CHECK: {{%.*}} = spirv.ImageSampleProjDrefImplicitLod {{%.*}}, {{%.*}}, {{%.*}} : !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<4xf32>, f32 -> f32
  %0 = spirv.ImageSampleProjDrefImplicitLod %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<4xf32>, f32 -> f32
  spirv.Return
}

// -----

func.func @sample_implicit_proj_dref_buffer_dim(%arg0 : !spirv.sampled_image<!spirv.image<f32, Buffer, IsDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the Dim operand of the underlying image must not be Buffer}}
  %0 = spirv.ImageSampleProjDrefImplicitLod %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<f32, Buffer, IsDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<4xf32>, f32 -> f32
  spirv.Return
}

// -----

func.func @sample_implicit_proj_dref_multi_sampled(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, MultiSampled, NeedSampler, Rgba8>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the MS operand of the underlying image type must be SingleSampled}}
  %0 = spirv.ImageSampleProjDrefImplicitLod %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, MultiSampled, NeedSampler, Rgba8>>, vector<4xf32>, f32 -> f32
  spirv.Return
}

// -----

func.func @sample_implicit_proj_dref(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{type of 'result' matches image type of 'sampled_image'}}
  %0 = spirv.ImageSampleProjDrefImplicitLod %arg0, %arg1, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, IsDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<4xf32>, f32 -> i32
  spirv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageOperands: Bias
//===----------------------------------------------------------------------===//

func.func @bias_too_many_arguments(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : f32, %arg2 : f32) -> () {
  // expected-error @+1 {{too many image operand arguments have been provided}}
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 ["Bias"], %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, f32, f32, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @bias_too_many_arguments(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : f32, %arg2 : i32) -> () {
  // expected-error @+1 {{Bias must be a floating-point type scalar}}
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 ["Bias"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, f32, i32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @bias_with_rect(%arg0 : !spirv.sampled_image<!spirv.image<f32, Rect, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : f32, %arg2 : f32) -> () {
  // expected-error @+1 {{Bias must only be used with an image type that has a dim operand of 1D, 2D, 3D, or Cube}}
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 ["Bias"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Rect, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, f32, f32 -> vector<4xf32>
  spirv.Return
}

// TODO: We cannot currently test Bias with MS != 0 as all implemented implicit operations already check for that.

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageOperands: Lod
//===----------------------------------------------------------------------===//

func.func @grad_and_lod_set(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : f32, %arg2 : f32) -> () {
  // expected-error @+1 {{it is invalid to set both the Lod and Grad bits}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Lod | Grad"], %arg2, %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim1D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, f32, f32, f32, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @lod_with_implict_sample(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{Lod is only valid with explicit-lod and fetch instructions}}
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 ["Lod"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @lod_too_many_arguments(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{too many image operand arguments have been provided}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Lod"], %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @lod_with_rect(%arg0 : !spirv.sampled_image<!spirv.image<f32, Rect, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{Lod must only be used with an image type that has a dim operand of 1D, 2D, 3D, or Cube}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Lod"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Rect, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32 -> vector<4xf32>
  spirv.Return
}

// TODO: We cannot currently test Lod with MS != 0 as all implemented explicit operations already check for that.

// TODO: Add Lod tests for fetch operations once available.

// -----

//===----------------------------------------------------------------------===//
// spirv.ImageOperands: Grad
//===----------------------------------------------------------------------===//

func.func @gard_with_implicit_sample(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : vector<2xf32>) -> () {
  // expected-error @+1 {{Grad is only valid with explicit-lod instructions}}
  %0 = spirv.ImageSampleImplicitLod %arg0, %arg1 ["Grad"], %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, vector<2xf32>, vector<2xf32> -> vector<4xf32>
  spirv.Return
}

// -----

func.func @gard_not_enough_args(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : vector<2xf32>) -> () {
  // expected-error @+1 {{Grad operand requires 2 arguments (scalars or vectors)}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Grad"], %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, vector<2xf32> -> vector<4xf32>
  spirv.Return
}

// TODO: We cannot currently test Grad with MS != 0 as all implemented explicit operations already check for that.

// -----

func.func @grad_arg_size_mismatch(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : vector<3xf32>) -> () {
  // expected-error @+1 {{number of components of each Grad argument must equal the number of components in coordinate, minus the array layer component, if present}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Grad"], %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, vector<3xf32>, vector<3xf32> -> vector<4xf32>
  spirv.Return
}

// -----

func.func @gard_arg_wrong_type(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : vector<2xsi32>) -> () {
  // expected-error @+1 {{Grad arguments must be a vector of floating-point type}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Grad"], %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, vector<2xsi32>, vector<2xsi32> -> vector<4xf32>
  spirv.Return
}

// -----

func.func @gard_arg_size_mismatch_scalar(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{number of components of each Grad argument must equal the number of components in coordinate, minus the array layer component, if present}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Grad"], %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, f32, f32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @gard_int_args(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : i32) -> () {
  // expected-error @+1 {{Grad arguments must be a scalar or vector of floating-point type}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Grad"], %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, i32, i32 -> vector<4xf32>
  spirv.Return
}

// -----

func.func @gard_too_many_args(%arg0 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, %arg1 : vector<2xf32>, %arg2 : vector<2xf32>) -> () {
  // expected-error @+1 {{too many image operand arguments have been provided}}
  %0 = spirv.ImageSampleExplicitLod %arg0, %arg1 ["Grad"], %arg2, %arg2, %arg2 : !spirv.sampled_image<!spirv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Rgba8>>, vector<2xf32>, vector<2xf32>, vector<2xf32>, vector<2xf32> -> vector<4xf32>
  spirv.Return
}
