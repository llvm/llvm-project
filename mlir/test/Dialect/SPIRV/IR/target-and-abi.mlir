// RUN: mlir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{found unsupported 'spirv.something' attribute on operation}}
func.func @unknown_attr_on_op() attributes {
  spirv.something = 64
} { return }

// -----

// expected-error @+1 {{found unsupported 'spirv.something' attribute on region argument}}
func.func @unknown_attr_on_region(%arg: i32 {spirv.something}) {
  return
}

// -----

// expected-error @+1 {{cannot attach SPIR-V attributes to region result}}
func.func @unknown_attr_on_region() -> (i32 {spirv.something}) {
  %0 = arith.constant 10.0 : f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spirv.entry_point_abi
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spirv.entry_point_abi' attribute must be an entry point ABI attribute}}
func.func @spv_entry_point() attributes {
  spirv.entry_point_abi = 64
} { return }

// -----

func.func @spv_entry_point() attributes {
  // expected-error @+2 {{failed to parse SPIRV_EntryPointABIAttr parameter 'workgroup_size' which is to be a `DenseI32ArrayAttr`}}
  // expected-error @+1 {{expected '['}}
  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = 64>
} { return }

// -----

func.func @spv_entry_point() attributes {
  // CHECK: {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>}
  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>
} { return }

// -----

//===----------------------------------------------------------------------===//
// spirv.interface_var_abi
//===----------------------------------------------------------------------===//

// expected-error @+1 {{'spirv.interface_var_abi' must be a spirv::InterfaceVarABIAttr}}
func.func @interface_var(
  %arg0 : f32 {spirv.interface_var_abi = 64}
) { return }

// -----

func.func @interface_var(
// expected-error @+1 {{missing descriptor set}}
  %arg0 : f32 {spirv.interface_var_abi = #spirv.interface_var_abi<()>}
) { return }

// -----

func.func @interface_var(
// expected-error @+1 {{missing binding}}
  %arg0 : f32 {spirv.interface_var_abi = #spirv.interface_var_abi<(1,)>}
) { return }

// -----

func.func @interface_var(
// expected-error @+1 {{unknown storage class: }}
  %arg0 : f32 {spirv.interface_var_abi = #spirv.interface_var_abi<(1,2), Foo>}
) { return }

// -----

// CHECK: {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1), Uniform>}
func.func @interface_var(
    %arg0 : f32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1), Uniform>}
) { return }

// -----

// CHECK: {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
func.func @interface_var(
    %arg0 : f32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
) { return }

// -----

// expected-error @+1 {{'spirv.interface_var_abi' attribute cannot specify storage class when attaching to a non-scalar value}}
func.func @interface_var(
  %arg0 : memref<4xf32> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1), Uniform>}
) { return }

// -----

//===----------------------------------------------------------------------===//
// spirv.resource_limits
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @resource_limits_all_default()
func.func @resource_limits_all_default() attributes {
  // CHECK-SAME: #spirv.resource_limits<>
  limits = #spirv.resource_limits<>
} { return }

// -----

// CHECK-LABEL: func @resource_limits_min_max_subgroup_size()
func.func @resource_limits_min_max_subgroup_size() attributes {
  // CHECK-SAME: #spirv.resource_limits<min_subgroup_size = 32, max_subgroup_size = 64>
  limits = #spirv.resource_limits<min_subgroup_size = 32, max_subgroup_size=64>
} { return }

// -----

//===----------------------------------------------------------------------===//
// spirv.target_env
//===----------------------------------------------------------------------===//

func.func @target_env() attributes {
  // CHECK:      spirv.target_env = #spirv.target_env<
  // CHECK-SAME:   #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
  // CHECK-SAME:   #spirv.resource_limits<max_compute_workgroup_size = [128, 64, 64]>>
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    #spirv.resource_limits<
      max_compute_workgroup_size = [128, 64, 64]
    >>
} { return }

// -----

func.func @target_env_client_api() attributes {
  // CHECK:      spirv.target_env = #spirv.target_env<
  // CHECK-SAME:   #spirv.vce<v1.0, [], []>,
  // CHECK-SAME:   api=Metal,
  // CHECK-SAME:   #spirv.resource_limits<>>
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, api=Metal, #spirv.resource_limits<>>
} { return }

// -----

func.func @target_env_client_api() attributes {
  // CHECK:      spirv.target_env = #spirv.target_env
  // CHECK-NOT:   api=
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, api=Unknown, #spirv.resource_limits<>>
} { return }

// -----

func.func @target_env_vendor_id() attributes {
  // CHECK:      spirv.target_env = #spirv.target_env<
  // CHECK-SAME:   #spirv.vce<v1.0, [], []>,
  // CHECK-SAME:   NVIDIA,
  // CHECK-SAME:   #spirv.resource_limits<>>
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, NVIDIA, #spirv.resource_limits<>>
} { return }

// -----

func.func @target_env_vendor_id_device_type() attributes {
  // CHECK:      spirv.target_env = #spirv.target_env<
  // CHECK-SAME:   #spirv.vce<v1.0, [], []>,
  // CHECK-SAME:   AMD:DiscreteGPU,
  // CHECK-SAME:   #spirv.resource_limits<>>
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, AMD:DiscreteGPU, #spirv.resource_limits<>>
} { return }

// -----

func.func @target_env_vendor_id_device_type_device_id() attributes {
  // CHECK:      spirv.target_env = #spirv.target_env<
  // CHECK-SAME:   #spirv.vce<v1.0, [], []>,
  // CHECK-SAME:   Qualcomm:IntegratedGPU:100925441,
  // CHECK-SAME:   #spirv.resource_limits<>>
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, Qualcomm:IntegratedGPU:0x6040001, #spirv.resource_limits<>>
} { return }

// -----

func.func @target_env_client_api_vendor_id_device_type_device_id() attributes {
  // CHECK:      spirv.target_env = #spirv.target_env<
  // CHECK-SAME:   #spirv.vce<v1.0, [], []>,
  // CHECK-SAME:   api=Vulkan,
  // CHECK-SAME:   Qualcomm:IntegratedGPU:100925441,
  // CHECK-SAME:   #spirv.resource_limits<>>
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [], []>, api=Vulkan, Qualcomm:IntegratedGPU:0x6040001, #spirv.resource_limits<>>
} { return }

// -----

func.func @target_env_extra_fields() attributes {
  // expected-error @+3 {{expected '>'}}
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    #spirv.resource_limits<>,
    more_stuff
  >
} { return }

// -----

func.func @target_env_cooperative_matrix_nv() attributes{
  // CHECK:      spirv.target_env = #spirv.target_env<
  // CHECK-SAME:   SPV_NV_cooperative_matrix
  // CHECK-SAME: #spirv.coop_matrix_props_nv<
  // CHECK-SAME:   m_size = 8, n_size = 8, k_size = 32,
  // CHECK-SAME:   a_type = i8, b_type = i8, c_type = i32,
  // CHECK-SAME:   result_type = i32, scope = <Subgroup>>
  // CHECK-SAME: #spirv.coop_matrix_props_nv<
  // CHECK-SAME:   m_size = 8, n_size = 8, k_size = 16,
  // CHECK-SAME:   a_type = f16, b_type = f16, c_type = f16,
  // CHECK-SAME:   result_type = f16, scope = <Subgroup>>
  spirv.target_env = #spirv.target_env<
  #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class,
                            SPV_NV_cooperative_matrix]>,
  #spirv.resource_limits<
    cooperative_matrix_properties_nv = [#spirv.coop_matrix_props_nv<
      m_size = 8,
      n_size = 8,
      k_size = 32,
      a_type = i8,
      b_type = i8,
      c_type = i32,
      result_type = i32,
      scope = #spirv.scope<Subgroup>
    >, #spirv.coop_matrix_props_nv<
      m_size = 8,
      n_size = 8,
      k_size = 16,
      a_type = f16,
      b_type = f16,
      c_type = f16,
      result_type = f16,
      scope = #spirv.scope<Subgroup>
    >]
  >>
} { return }

// -----

//===----------------------------------------------------------------------===//
// spirv.vce
//===----------------------------------------------------------------------===//

func.func @vce_wrong_type() attributes {
  // expected-error @+1 {{expected valid keyword}}
  vce = #spirv.vce<64>
} { return }

// -----

func.func @vce_missing_fields() attributes {
  // expected-error @+1 {{expected ','}}
  vce = #spirv.vce<v1.0>
} { return }

// -----

func.func @vce_wrong_version() attributes {
  // expected-error @+1 {{unknown version: V_x_y}}
  vce = #spirv.vce<V_x_y, []>
} { return }

// -----

func.func @vce_wrong_extension_type() attributes {
  // expected-error @+1 {{expected valid keyword}}
  vce = #spirv.vce<v1.0, [32: i32], [Shader]>
} { return }

// -----

func.func @vce_wrong_extension() attributes {
  // expected-error @+1 {{unknown extension: SPIRV_Something}}
  vce = #spirv.vce<v1.0, [Shader], [SPIRV_Something]>
} { return }

// -----

func.func @vce_wrong_capability() attributes {
  // expected-error @+1 {{unknown capability: Something}}
  vce = #spirv.vce<v1.0, [Something], []>
} { return }

// -----

func.func @vce() attributes {
  // CHECK: #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>
  vce = #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>
} { return }
