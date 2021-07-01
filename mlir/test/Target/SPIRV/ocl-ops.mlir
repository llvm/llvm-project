// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Physical64 OpenCL requires #spv.vce<v1.0, [Kernel, Addresses], []> {
  spv.func @float_insts(%arg0 : f32) "None" {
    // CHECK: {{%.*}} = spv.OCL.exp {{%.*}} : f32
    %0 = spv.OCL.exp %arg0 : f32
    // CHECK: {{%.*}} = spv.OCL.fabs {{%.*}} : f32
    %1 = spv.OCL.fabs %arg0 : f32
    // CHECK: {{%.*}} = spv.OCL.sin {{%.*}} : f32
    %2 = spv.OCL.sin %arg0 : f32
    // CHECK: {{%.*}} = spv.OCL.cos {{%.*}} : f32
    %3 = spv.OCL.cos %arg0 : f32
    // CHECK: {{%.*}} = spv.OCL.log {{%.*}} : f32
    %4 = spv.OCL.log %arg0 : f32
    // CHECK: {{%.*}} = spv.OCL.sqrt {{%.*}} : f32
    %5 = spv.OCL.sqrt %arg0 : f32
    spv.Return
  }

  spv.func @integer_insts(%arg0 : i32) "None" {
    // CHECK: {{%.*}} = spv.OCL.s_abs {{%.*}} : i32
    %0 = spv.OCL.s_abs %arg0 : i32
    spv.Return
  }
  
  spv.func @vector_size16(%arg0 : vector<16xf32>) "None" {
    // CHECK: {{%.*}} = spv.OCL.fabs {{%.*}} : vector<16xf32>
    %0 = spv.OCL.fabs %arg0 : vector<16xf32>
    spv.Return
  }
}
