// RUN: mlir-opt -split-input-file %s | FileCheck %s

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 128 : i32>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096: i32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128: i32>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", 8192 : i32>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096: i32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 8192: i32>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i64>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", 8192 : i64>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096: i64>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 8192: i64>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 64 : i32>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 128 : i32>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 64: i32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128: i32>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 64 : i64>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 128 : i64>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 64: i64>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128: i64>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 64 : i64>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 128 : i64>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 64>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128>>
  >} {}
