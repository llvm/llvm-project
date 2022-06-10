// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{found unsupported 'spv.something' attribute on operation}}
func.func @unknown_attr_on_op() attributes {
  spv.something = 64
} { return }

// -----

// expected-error @+1 {{found unsupported 'spv.something' attribute on region argument}}
func.func @unknown_attr_on_region(%arg: i32 {spv.something}) {
  return
}

// -----

// expected-error @+1 {{cannot attach SPIR-V attributes to region result}}
func.func @unknown_attr_on_region() -> (i32 {spv.something}) {
  %0 = arith.constant 10.0 : f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.entry_point_abi
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spv.entry_point_abi' attribute must be an entry point ABI attribute}}
func.func @spv_entry_point() attributes {
  spv.entry_point_abi = 64
} { return }

// -----

func.func @spv_entry_point() attributes {
  // expected-error @+2 {{failed to parse SPV_EntryPointABIAttr parameter 'local_size' which is to be a `DenseIntElementsAttr`}}
  // expected-error @+1 {{invalid kind of attribute specified}}
  spv.entry_point_abi = #spv.entry_point_abi<local_size = 64>
} { return }

// -----

func.func @spv_entry_point() attributes {
  // CHECK: {spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[64, 1, 1]> : vector<3xi32>>}
  spv.entry_point_abi = #spv.entry_point_abi<local_size = dense<[64, 1, 1]>: vector<3xi32>>
} { return }

// -----

//===----------------------------------------------------------------------===//
// spv.interface_var_abi
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spv.interface_var_abi' must be a spirv::InterfaceVarABIAttr}}
func.func @interface_var(
  %arg0 : f32 {spv.interface_var_abi = 64}
) { return }

// -----

func.func @interface_var(
// expected-error @+1 {{missing descriptor set}}
  %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<()>}
) { return }

// -----

func.func @interface_var(
// expected-error @+1 {{missing binding}}
  %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<(1,)>}
) { return }

// -----

func.func @interface_var(
// expected-error @+1 {{unknown storage class: }}
  %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<(1,2), Foo>}
) { return }

// -----

// CHECK: {spv.interface_var_abi = #spv.interface_var_abi<(0, 1), Uniform>}
func.func @interface_var(
    %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 1), Uniform>}
) { return }

// -----

// CHECK: {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}
func.func @interface_var(
    %arg0 : f32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}
) { return }

// -----

// expected-error @+1 {{'spv.interface_var_abi' attribute cannot specify storage class when attaching to a non-scalar value}}
func.func @interface_var(
  %arg0 : memref<4xf32> {spv.interface_var_abi = #spv.interface_var_abi<(0, 1), Uniform>}
) { return }

// -----

//===----------------------------------------------------------------------===//
// spv.target_env
//===----------------------------------------------------------------------===//

func.func @target_env() attributes {
  // CHECK:      spv.target_env = #spv.target_env<
  // CHECK-SAME:   #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
  // CHECK-SAME:   #spv.resource_limits<max_compute_workgroup_size = [128, 64, 64]>>
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    #spv.resource_limits<
      max_compute_workgroup_size = [128, 64, 64]
    >>
} { return }

// -----

func.func @target_env_vendor_id() attributes {
  // CHECK:      spv.target_env = #spv.target_env<
  // CHECK-SAME:   #spv.vce<v1.0, [], []>,
  // CHECK-SAME:   NVIDIA,
  // CHECK-SAME:   #spv.resource_limits<>>
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, NVIDIA, #spv.resource_limits<>>
} { return }

// -----

func.func @target_env_vendor_id_device_type() attributes {
  // CHECK:      spv.target_env = #spv.target_env<
  // CHECK-SAME:   #spv.vce<v1.0, [], []>,
  // CHECK-SAME:   AMD:DiscreteGPU,
  // CHECK-SAME:   #spv.resource_limits<>>
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, AMD:DiscreteGPU, #spv.resource_limits<>>
} { return }

// -----

func.func @target_env_vendor_id_device_type_device_id() attributes {
  // CHECK:      spv.target_env = #spv.target_env<
  // CHECK-SAME:   #spv.vce<v1.0, [], []>,
  // CHECK-SAME:   Qualcomm:IntegratedGPU:100925441,
  // CHECK-SAME:   #spv.resource_limits<>>
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, Qualcomm:IntegratedGPU:0x6040001, #spv.resource_limits<>>
} { return }

// -----

func.func @target_env_extra_fields() attributes {
  // expected-error @+3 {{expected '>'}}
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    #spv.resource_limits<>,
    more_stuff
  >
} { return }

// -----

func.func @target_env_cooperative_matrix() attributes{
  // CHECK:      spv.target_env = #spv.target_env<
  // CHECK-SAME:   SPV_NV_cooperative_matrix
  // CHECK-SAME: #spv.coop_matrix_props<
  // CHECK-SAME:   m_size = 8, n_size = 8, k_size = 32,
  // CHECK-SAME:   a_type = i8, b_type = i8, c_type = i32,
  // CHECK-SAME:   result_type = i32, scope = 3 : i32>
  // CHECK-SAME: #spv.coop_matrix_props<
  // CHECK-SAME:   m_size = 8, n_size = 8, k_size = 16,
  // CHECK-SAME:   a_type = f16, b_type = f16, c_type = f16,
  // CHECK-SAME:   result_type = f16, scope = 3 : i32>
  spv.target_env = #spv.target_env<
  #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class,
                            SPV_NV_cooperative_matrix]>,
  #spv.resource_limits<
    cooperative_matrix_properties_nv = [#spv.coop_matrix_props<
      m_size = 8,
      n_size = 8,
      k_size = 32,
      a_type = i8,
      b_type = i8,
      c_type = i32,
      result_type = i32,
      scope = 3 : i32
    >, #spv.coop_matrix_props<
      m_size = 8,
      n_size = 8,
      k_size = 16,
      a_type = f16,
      b_type = f16,
      c_type = f16,
      result_type = f16,
      scope = 3 : i32
    >]
  >>
} { return }

// -----

//===----------------------------------------------------------------------===//
// spv.vce
//===----------------------------------------------------------------------===//

func.func @vce_wrong_type() attributes {
  // expected-error @+1 {{expected valid keyword}}
  vce = #spv.vce<64>
} { return }

// -----

func.func @vce_missing_fields() attributes {
  // expected-error @+1 {{expected ','}}
  vce = #spv.vce<v1.0>
} { return }

// -----

func.func @vce_wrong_version() attributes {
  // expected-error @+1 {{unknown version: V_x_y}}
  vce = #spv.vce<V_x_y, []>
} { return }

// -----

func.func @vce_wrong_extension_type() attributes {
  // expected-error @+1 {{expected valid keyword}}
  vce = #spv.vce<v1.0, [32: i32], [Shader]>
} { return }

// -----

func.func @vce_wrong_extension() attributes {
  // expected-error @+1 {{unknown extension: SPV_Something}}
  vce = #spv.vce<v1.0, [Shader], [SPV_Something]>
} { return }

// -----

func.func @vce_wrong_capability() attributes {
  // expected-error @+1 {{unknown capability: Something}}
  vce = #spv.vce<v1.0, [Something], []>
} { return }

// -----

func.func @vce() attributes {
  // CHECK: #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>
  vce = #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>
} { return }
