// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{found unsupported 'spv.something' attribute on operation}}
func @unknown_attr_on_op() attributes {
  spv.something = 64
} { return }

// -----

// expected-error @+1 {{found unsupported 'spv.something' attribute on region argument}}
func @unknown_attr_on_region(%arg: i32 {spv.something}) {
  return
}

// -----

// expected-error @+1 {{cannot attach SPIR-V attributes to region result}}
func @unknown_attr_on_region() -> (i32 {spv.something}) {
  %0 = constant 10.0 : f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.entry_point_abi
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spv.entry_point_abi' attribute must be a dictionary attribute containing one 32-bit integer elements attribute: 'local_size'}}
func @spv_entry_point() attributes {
  spv.entry_point_abi = 64
} { return }

// -----

// expected-error @+1 {{'spv.entry_point_abi' attribute must be a dictionary attribute containing one 32-bit integer elements attribute: 'local_size'}}
func @spv_entry_point() attributes {
  spv.entry_point_abi = {local_size = 64}
} { return }

// -----

func @spv_entry_point() attributes {
  // CHECK: {spv.entry_point_abi = {local_size = dense<[64, 1, 1]> : vector<3xi32>}}
  spv.entry_point_abi = {local_size = dense<[64, 1, 1]>: vector<3xi32>}
} { return }

// -----

//===----------------------------------------------------------------------===//
// spv.interface_var_abi
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spv.interface_var_abi' must be a spirv::InterfaceVarABIAttr}}
func @interface_var(
  %arg0 : f32 {spv.interface_var_abi = 64}
) { return }

// -----

func @interface_var(
// expected-error @+1 {{missing descriptor set}}
  %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<()>}
) { return }

// -----

func @interface_var(
// expected-error @+1 {{missing binding}}
  %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<(1,)>}
) { return }

// -----

func @interface_var(
// expected-error @+1 {{unknown storage class: }}
  %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<(1,2), Foo>}
) { return }

// -----

// CHECK: {spv.interface_var_abi = #spv.interface_var_abi<(0, 1), Uniform>}
func @interface_var(
    %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 1), Uniform>}
) { return }

// -----

// CHECK: {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}
func @interface_var(
    %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}
) { return }

// -----

// expected-error @+1 {{'spv.interface_var_abi' attribute cannot specify storage class when attaching to a non-scalar value}}
func @interface_var(
  %arg0 : memref<4xf32> {spv.interface_var_abi = #spv.interface_var_abi<(0, 1), Uniform>}
) { return }

// -----

//===----------------------------------------------------------------------===//
// spv.target_env
//===----------------------------------------------------------------------===//

func @target_env_wrong_limits() attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    // expected-error @+1 {{limits must be a dictionary attribute containing two 32-bit integer attributes 'max_compute_workgroup_invocations' and 'max_compute_workgroup_size'}}
    {max_compute_workgroup_invocations = 128 : i64, max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>}>
} { return }

// -----

func @target_env() attributes {
  // CHECK:      spv.target_env = #spv.target_env<
  // CHECK-SAME:   #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
  // CHECK-SAME:   {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>}>
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {
      max_compute_workgroup_invocations = 128 : i32,
      max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>
    }>
} { return }

// -----

func @target_env_extra_fields() attributes {
  // expected-error @+6 {{expected '>'}}
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {
      max_compute_workgroup_invocations = 128 : i32,
      max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>
    },
    more_stuff
  >
} { return }

// -----

//===----------------------------------------------------------------------===//
// spv.vce
//===----------------------------------------------------------------------===//

func @vce_wrong_type() attributes {
  // expected-error @+1 {{expected valid keyword}}
  vce = #spv.vce<64>
} { return }

// -----

func @vce_missing_fields() attributes {
  // expected-error @+1 {{expected ','}}
  vce = #spv.vce<v1.0>
} { return }

// -----

func @vce_wrong_version() attributes {
  // expected-error @+1 {{unknown version: V_x_y}}
  vce = #spv.vce<V_x_y, []>
} { return }

// -----

func @vce_wrong_extension_type() attributes {
  // expected-error @+1 {{expected valid keyword}}
  vce = #spv.vce<v1.0, [32: i32], [Shader]>
} { return }

// -----

func @vce_wrong_extension() attributes {
  // expected-error @+1 {{unknown extension: SPV_Something}}
  vce = #spv.vce<v1.0, [Shader], [SPV_Something]>
} { return }

// -----

func @vce_wrong_capability() attributes {
  // expected-error @+1 {{unknown capability: Something}}
  vce = #spv.vce<v1.0, [Something], []>
} { return }

// -----

func @vce() attributes {
  // CHECK: #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>
  vce = #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>
} { return }
