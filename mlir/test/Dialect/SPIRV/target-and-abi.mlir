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

// expected-error @+1 {{found unsupported 'spv.something' attribute on region result}}
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

// expected-error @+1 {{'spv.interface_var_abi' attribute must be a dictionary attribute containing three 32-bit integer attributes: 'descriptor_set', 'binding', and 'storage_class'}}
func @interface_var(
  %arg0 : f32 {spv.interface_var_abi = 64}
) { return }

// -----

// expected-error @+1 {{'spv.interface_var_abi' attribute must be a dictionary attribute containing three 32-bit integer attributes: 'descriptor_set', 'binding', and 'storage_class'}}
func @interface_var(
  %arg0 : f32 {spv.interface_var_abi = {binding = 0: i32}}
) { return }

// -----

// CHECK: {spv.interface_var_abi = {binding = 0 : i32, descriptor_set = 0 : i32, storage_class = 12 : i32}}
func @interface_var(
  %arg0 : f32 {spv.interface_var_abi = {binding = 0 : i32,
                                        descriptor_set = 0 : i32,
                                        storage_class = 12 : i32}}
) { return }

// -----

// expected-error @+1 {{'spv.interface_var_abi' attribute must be a dictionary attribute containing three 32-bit integer attributes: 'descriptor_set', 'binding', and 'storage_class'}}
func @interface_var() -> (f32 {spv.interface_var_abi = 64})
{
  %0 = constant 10.0 : f32
  return %0: f32
}

// -----

// expected-error @+1 {{'spv.interface_var_abi' attribute must be a dictionary attribute containing three 32-bit integer attributes: 'descriptor_set', 'binding', and 'storage_class'}}
func @interface_var() -> (f32 {spv.interface_var_abi = {binding = 0: i32}})
{
  %0 = constant 10.0 : f32
  return %0: f32
}

// -----

// CHECK: {spv.interface_var_abi = {binding = 0 : i32, descriptor_set = 0 : i32, storage_class = 12 : i32}}
func @interface_var() -> (f32 {spv.interface_var_abi = {
    binding = 0 : i32, descriptor_set = 0 : i32, storage_class = 12 : i32}})
{
  %0 = constant 10.0 : f32
  return %0: f32
}

// -----

//===----------------------------------------------------------------------===//
// spv.target_env
//===----------------------------------------------------------------------===//

func @target_env_wrong_type() attributes {
  // expected-error @+1 {{expected valid keyword}}
  spv.target_env = #spv.target_env<64>
} { return }

// -----

func @target_env_missing_fields() attributes {
  // expected-error @+1 {{expected ','}}
  spv.target_env = #spv.target_env<V_1_0>
} { return }

// -----

func @target_env_wrong_version() attributes {
  // expected-error @+1 {{unknown version: V_x_y}}
  spv.target_env = #spv.target_env<V_x_y, []>
} { return }

// -----

func @target_env_wrong_extension_type() attributes {
  // expected-error @+1 {{expected valid keyword}}
  spv.target_env = #spv.target_env<V_1_0, [32: i32], [Shader]>
} { return }

// -----

func @target_env_wrong_extension() attributes {
  // expected-error @+1 {{unknown extension: SPV_Something}}
  spv.target_env = #spv.target_env<V_1_0, [SPV_Something], [Shader]>
} { return }

// -----

func @target_env_wrong_capability() attributes {
  // expected-error @+1 {{unknown capability: Something}}
  spv.target_env = #spv.target_env<V_1_0, [], [Something]>
} { return }

// -----

func @target_env_missing_limits() attributes {
  spv.target_env = #spv.target_env<
    V_1_0, [SPV_KHR_storage_buffer_storage_class], [Shader],
    // expected-error @+1 {{limits must be a dictionary attribute containing two 32-bit integer attributes 'max_compute_workgroup_invocations' and 'max_compute_workgroup_size'}}
    {max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>}>
} { return }

// -----

func @target_env_wrong_limits() attributes {
  spv.target_env = #spv.target_env<
    V_1_0, [SPV_KHR_storage_buffer_storage_class], [Shader],
    // expected-error @+1 {{limits must be a dictionary attribute containing two 32-bit integer attributes 'max_compute_workgroup_invocations' and 'max_compute_workgroup_size'}}
    {max_compute_workgroup_invocations = 128 : i64, max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>}>
} { return }

// -----

func @target_env() attributes {

  // CHECK: spv.target_env = #spv.target_env<V_1_0, [SPV_KHR_storage_buffer_storage_class], [Shader], {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>}>
  spv.target_env = #spv.target_env<
    V_1_0, [SPV_KHR_storage_buffer_storage_class], [Shader],
    {
      max_compute_workgroup_invocations = 128 : i32,
      max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>
    }>
} { return }

// -----

func @target_env_extra_fields() attributes {
  // expected-error @+6 {{expected '>'}}
  spv.target_env = #spv.target_env<
    V_1_0, [SPV_KHR_storage_buffer_storage_class], [Shader],
    {
      max_compute_workgroup_invocations = 128 : i32,
      max_compute_workgroup_size = dense<[128, 64, 64]> : vector<3xi32>
    },
    more_stuff
  >
} { return }
