// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Image
//===----------------------------------------------------------------------===//

func @image(%arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>) -> () {
  // CHECK: spv.Image {{.*}} : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
  %0 = spv.Image %arg0 : !spv.sampled_image<!spv.image<f32, Dim2D, NoDepth, NonArrayed, SingleSampled, NeedSampler, Unknown>>
  return
}