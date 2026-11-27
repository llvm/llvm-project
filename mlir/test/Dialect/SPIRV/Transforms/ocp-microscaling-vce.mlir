// RUN: mlir-opt --spirv-update-vce --split-input-file %s | FileCheck %s

// Check each format independently so one type cannot supply another's requirements.
// CHECK: requires #spirv.vce<v1.0, [Float4EXT, Shader, Matrix], [SPV_EXT_ocp_microscaling_types, SPV_EXT_float8]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Float4EXT],
                [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types]>,
    #spirv.resource_limits<>>
} {
  spirv.func @fp4(%scalar: f4E2M1FN, %vector: vector<2xf4E2M1FN>) "None" {
    spirv.Return
  }
}

// -----

// CHECK: requires #spirv.vce<v1.0, [Float6EXT, Shader, Matrix], [SPV_EXT_ocp_microscaling_types, SPV_EXT_float8]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Float6EXT],
                [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types]>,
    #spirv.resource_limits<>>
} {
  spirv.func @fp6e2m3(%scalar: f6E2M3FN, %vector: vector<2xf6E2M3FN>) "None" {
    spirv.Return
  }
}

// -----

// CHECK: requires #spirv.vce<v1.0, [Float6EXT, Shader, Matrix], [SPV_EXT_ocp_microscaling_types, SPV_EXT_float8]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Float6EXT],
                [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types]>,
    #spirv.resource_limits<>>
} {
  spirv.func @fp6e3m2(%scalar: f6E3M2FN, %vector: vector<2xf6E3M2FN>) "None" {
    spirv.Return
  }
}

// -----

// CHECK: requires #spirv.vce<v1.0, [Float8UnsignedE8M0EXT, Shader, Matrix], [SPV_EXT_ocp_microscaling_types, SPV_EXT_float8]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Float8UnsignedE8M0EXT],
                [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types]>,
    #spirv.resource_limits<>>
} {
  spirv.func @fp8e8m0(%scalar: f8E8M0FNU, %vector: vector<2xf8E8M0FNU>) "None" {
    spirv.Return
  }
}

// -----

// BitcastExtractEXT requires SPV_EXT_float8 even when no float8 type is used.
// CHECK: requires #spirv.vce<v1.0, [{{[^]]*}}BitcastExtractEXT{{[^]]*}}], [SPV_EXT_ocp_microscaling_types, SPV_EXT_float8]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Int16, BitcastExtractEXT],
                [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types]>,
    #spirv.resource_limits<>>
} {
  spirv.func @integer_bitcast_extract(%base: i32, %offset: i32) -> i16 "None" {
    %value = spirv.BitcastExtractEXT %base, %offset : i32, i32 to i16
    spirv.ReturnValue %value : i16
  }
}

// -----

// CHECK: requires #spirv.vce<v1.0, [{{.*}}Float8UnsignedE8M0EXT{{.*}}], [{{.*}}SPV_EXT_ocp_microscaling_types{{.*}}]>
spirv.module Logical GLSL450 attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader, Float8UnsignedE8M0EXT, StorageBuffer8BitAccess],
                [SPV_EXT_float8, SPV_EXT_ocp_microscaling_types, SPV_KHR_8bit_storage,
                 SPV_KHR_storage_buffer_storage_class]>,
    #spirv.resource_limits<>>
} {
  spirv.func @storage(%ptr: !spirv.ptr<f8E8M0FNU, StorageBuffer>) -> () "None" {
    spirv.Return
  }
}
