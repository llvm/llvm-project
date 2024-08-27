// RUN: mlir-opt -split-input-file %s | FileCheck %s


// CHECK:      module attributes {
// CHECK-SAME:   dlti.map = #dlti.map<
// CHECK-SAME:     #dlti.dl_entry<"magic_num", 42 : i32>,
// CHECK-SAME:     #dlti.dl_entry<"magic_num_float", 4.242000e+01 : f32>,
// CHECK-SAME:     #dlti.dl_entry<"magic_type", i32>,
// CHECK-SAME:     #dlti.dl_entry<i32,
// CHECK-SAME:       #dlti.map<#dlti.dl_entry<"bitwidth", 32 : i32>>>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.map = #dlti.map<
    #dlti.dl_entry<"magic_num", 42 : i32>,
    #dlti.dl_entry<"magic_num_float", 42.42 : f32>,
    #dlti.dl_entry<"magic_type", i32>,
    #dlti.dl_entry<i32, #dlti.map<#dlti.dl_entry<"bitwidth", 32 : i32>>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.map = #dlti.map<
// CHECK-SAME:    #dlti.dl_entry<"CPU", #dlti.map<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>>,
// CHECK-SAME:    #dlti.dl_entry<"GPU", #dlti.map<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 128 : i32>>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.map = #dlti.map<
    #dlti.dl_entry<"CPU", #dlti.map<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>>,
    #dlti.dl_entry<"GPU", #dlti.map<
      #dlti.dl_entry<"max_vector_op_width", 128 : i32>>>
  >} {}

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
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128 : i32>>
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
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 8192 : i32>>
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
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i64>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 8192 : i64>>
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
      #dlti.dl_entry<"max_vector_op_width", 64 : i32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128 : i32>>
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
      #dlti.dl_entry<"max_vector_op_width", 64 : i64>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128 : i64>>
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
      #dlti.dl_entry<"max_vector_op_width", 64 : i64>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128 : i64>>
  >} {}

// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : ui32>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", "128">>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : ui32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", "128">>
  >} {}

// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"max_vector_op_width", 4.096000e+03 : f32>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", "128">>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 4096.0 : f32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", "128">>
  >} {}


// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"vector_unit", #dlti.map<
// CHECK-SAME:        #dlti.dl_entry<"max_op_width", 4.096000e+03 : f32>>>>,
// CHECK-SAME:    "GPU" : #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"L1_cache_size_in_bytes", "128">>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"vector_unit", #dlti.map<
        #dlti.dl_entry<"max_op_width", 4096.0 : f32>>>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", "128">>
  >} {}


// -----

// CHECK: "test.op_with_dlti_map"() ({
// CHECK: }) {dlti.map = #dlti.map<#dlti.dl_entry<"dlti.unknown_id", 42 : i64>>}
"test.op_with_dlti_map"() ({
}) { dlti.map = #dlti.map<#dlti.dl_entry<"dlti.unknown_id", 42>> } : () -> ()

// -----

// CHECK: "test.op_with_dlti_map"() ({
// CHECK: }) {dlti.map = #dlti.map<#dlti.dl_entry<i32, 42 : i64>>}
"test.op_with_dlti_map"() ({
}) { dlti.map = #dlti.map<#dlti.dl_entry<i32, 42>> } : () -> ()
