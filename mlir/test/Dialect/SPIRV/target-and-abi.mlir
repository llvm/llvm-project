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

// expected-error @+1 {{'spv.entry_point_abi' attribute must be a dictionary attribute containing one integer elements attribute: 'local_size'}}
func @spv_entry_point() attributes {
  spv.entry_point_abi = 64
} { return }

// -----

// expected-error @+1 {{'spv.entry_point_abi' attribute must be a dictionary attribute containing one integer elements attribute: 'local_size'}}
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

// expected-error @+1 {{'spv.interface_var_abi' attribute must be a dictionary attribute containing three integer attributes: 'descriptor_set', 'binding', and 'storage_class'}}
func @interface_var(
  %arg0 : f32 {spv.interface_var_abi = 64}
) { return }

// -----

// expected-error @+1 {{'spv.interface_var_abi' attribute must be a dictionary attribute containing three integer attributes: 'descriptor_set', 'binding', and 'storage_class'}}
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

// expected-error @+1 {{'spv.interface_var_abi' attribute must be a dictionary attribute containing three integer attributes: 'descriptor_set', 'binding', and 'storage_class'}}
func @interface_var() -> (f32 {spv.interface_var_abi = 64})
{
  %0 = constant 10.0 : f32
  return %0: f32
}

// -----

// expected-error @+1 {{'spv.interface_var_abi' attribute must be a dictionary attribute containing three integer attributes: 'descriptor_set', 'binding', and 'storage_class'}}
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
