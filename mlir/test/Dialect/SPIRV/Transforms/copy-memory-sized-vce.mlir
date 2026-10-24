// RUN: mlir-opt --spirv-update-vce --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK: requires #spirv.vce<v1.0, [Addresses, Kernel], []>
spirv.module Physical64 OpenCL attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses], []>, #spirv.resource_limits<>>
} {
  spirv.func @copy(%ptr: !spirv.ptr<i32, Function>, %size: i32) "None" {
    spirv.CopyMemorySized %ptr, %ptr, %size : !spirv.ptr<i32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.Return
  }
}

// -----

// CHECK: requires #spirv.vce<v1.4, [Addresses, Kernel], []>
spirv.module Physical64 OpenCL attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses], []>, #spirv.resource_limits<>>
} {
  spirv.func @copy(%ptr: !spirv.ptr<i32, Function>, %size: i32) "None" {
    spirv.CopyMemorySized %ptr, %ptr, %size ["None"], ["None"] : !spirv.ptr<i32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.Return
  }
}

// -----

spirv.module Physical64 OpenCL attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses], []>, #spirv.resource_limits<>>
} {
  spirv.func @copy(%ptr: !spirv.ptr<i32, Function>, %size: i32) "None" {
    // expected-error @+1 {{'spirv.CopyMemorySized' requires min version v1.4 but target environment allows up to v1.3}}
    spirv.CopyMemorySized %ptr, %ptr, %size ["None"], ["None"] : !spirv.ptr<i32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.Return
  }
}
