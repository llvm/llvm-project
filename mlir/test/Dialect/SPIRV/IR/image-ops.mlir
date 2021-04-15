// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s


//===----------------------------------------------------------------------===//
// spv.ImageDrefGather
//===----------------------------------------------------------------------===//

func @image_dref_gather(%arg0 : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // CHECK: spv.ImageDrefGather {{.*}} : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, {{.*}} : vector<4xf32>, {{.*}} : f32 -> vector<4xi32>
  %0 = spv.ImageDrefGather %arg0 : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32 -> vector<4xi32>
  spv.Return
}

// -----

func @image_dref_gather_error_result_type(%arg0 : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{result type must be a vector of four components}}
  %0 = spv.ImageDrefGather %arg0 : !spv.sampled_image<!spv.image<i32, Dim2D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32 -> vector<3xi32>
  spv.Return
}

// -----

func @image_dref_gather_error_same_type(%arg0 : !spv.sampled_image<!spv.image<i32, Rect, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the component type of result must be the same as sampled type of the underlying image type}}
  %0 = spv.ImageDrefGather %arg0 : !spv.sampled_image<!spv.image<i32, Rect, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32 -> vector<4xf32>
  spv.Return
}

// -----

func @image_dref_gather_error_dim(%arg0 : !spv.sampled_image<!spv.image<i32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the Dim operand of the underlying image type must be 2D, Cube, or Rect}}
  %0 = spv.ImageDrefGather %arg0 : !spv.sampled_image<!spv.image<i32, Dim1D, NoDepth, NonArrayed, SingleSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32 -> vector<4xi32>
  spv.Return
}

// -----

func @image_dref_gather_error_ms(%arg0 : !spv.sampled_image<!spv.image<i32, Cube, NoDepth, NonArrayed, MultiSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32) -> () {
  // expected-error @+1 {{the MS operand of the underlying image type must be 0}}
  %0 = spv.ImageDrefGather %arg0 : !spv.sampled_image<!spv.image<i32, Cube, NoDepth, NonArrayed, MultiSampled, NoSampler, Unknown>>, %arg1 : vector<4xf32>, %arg2 : f32 -> vector<4xi32>
  spv.Return
}

// -----

//===----------------------------------------------------------------------===//
// spv.Image
//===----------------------------------------------------------------------===//

func @image(%arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>) -> () {
  // CHECK: spv.Image {{.*}} : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
  %0 = spv.Image %arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
  return
}