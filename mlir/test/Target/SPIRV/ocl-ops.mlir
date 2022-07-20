// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Physical64 OpenCL requires #spv.vce<v1.0, [Kernel, Addresses], []> {
  spv.func @float_insts(%arg0 : f32) "None" {
    // CHECK: {{%.*}} = spv.CL.exp {{%.*}} : f32
    %0 = spv.CL.exp %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.fabs {{%.*}} : f32
    %1 = spv.CL.fabs %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.sin {{%.*}} : f32
    %2 = spv.CL.sin %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.cos {{%.*}} : f32
    %3 = spv.CL.cos %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.log {{%.*}} : f32
    %4 = spv.CL.log %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.sqrt {{%.*}} : f32
    %5 = spv.CL.sqrt %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.ceil {{%.*}} : f32
    %6 = spv.CL.ceil %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.floor {{%.*}} : f32
    %7 = spv.CL.floor %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.pow {{%.*}}, {{%.*}} : f32
    %8 = spv.CL.pow %arg0, %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.rsqrt {{%.*}} : f32
    %9 = spv.CL.rsqrt %arg0 : f32
    // CHECK: {{%.*}} = spv.CL.erf {{%.*}} : f32
    %10 = spv.CL.erf %arg0 : f32
    spv.Return
  }

  spv.func @integer_insts(%arg0 : i32) "None" {
    // CHECK: {{%.*}} = spv.CL.s_abs {{%.*}} : i32
    %0 = spv.CL.s_abs %arg0 : i32
    spv.Return
  }
  
  spv.func @vector_size16(%arg0 : vector<16xf32>) "None" {
    // CHECK: {{%.*}} = spv.CL.fabs {{%.*}} : vector<16xf32>
    %0 = spv.CL.fabs %arg0 : vector<16xf32>
    spv.Return
  }

  spv.func @fma(%arg0 : f32, %arg1 : f32, %arg2 : f32) "None" {
    // CHECK: spv.CL.fma {{%[^,]*}}, {{%[^,]*}}, {{%[^,]*}} : f32
    %13 = spv.CL.fma %arg0, %arg1, %arg2 : f32
    spv.Return
  }
}
