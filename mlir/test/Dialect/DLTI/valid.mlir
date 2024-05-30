// RUN: mlir-opt -split-input-file %s | FileCheck %s

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_id", 0 : ui32>,
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_type", "CPU">>,
// CHECK-SAME:    #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_id", 1 : ui32>,
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_type", "GPU">>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.device_id", 0: ui32>,
      #dlti.dl_entry<"dlti.device_type", "CPU">>,
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.device_id", 1: ui32>,
      #dlti.dl_entry<"dlti.device_type", "GPU">>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_id", 0 : ui32>,
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_type", "CPU">,
// CHECK-SAME:      #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 4096 : ui32>>,
// CHECK-SAME:    #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_id", 1 : ui32>,
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_type", "GPU">,
// CHECK-SAME:      #dlti.dl_entry<"dlti.max_vector_op_width", 128 : ui32>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.device_id", 0: ui32>,
      #dlti.dl_entry<"dlti.device_type", "CPU">,
      #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 4096: ui32>>,
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.device_id", 1: ui32>,
      #dlti.dl_entry<"dlti.device_type", "GPU">,
      #dlti.dl_entry<"dlti.max_vector_op_width", 128: ui32>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_id", 0 : ui32>,
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_type", "CPU">,
// CHECK-SAME:      #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 4096 : ui32>>,
// CHECK-SAME:    #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_id", 1 : ui32>,
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_type", "GPU">,
// CHECK-SAME:      #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 8192 : ui32>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.device_id", 0: ui32>,
      #dlti.dl_entry<"dlti.device_type", "CPU">,
      #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 4096: ui32>>,
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.device_id", 1: ui32>,
      #dlti.dl_entry<"dlti.device_type", "GPU">,
      #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 8192: ui32>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_id", 0 : ui32>,
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_type", "CPU">,
// CHECK-SAME:      #dlti.dl_entry<"dlti.max_vector_op_width", 64 : ui32>>,
// CHECK-SAME:    #dlti.target_device_spec<
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_id", 1 : ui32>,
// CHECK-SAME:      #dlti.dl_entry<"dlti.device_type", "GPU">,
// CHECK-SAME:      #dlti.dl_entry<"dlti.max_vector_op_width", 128 : ui32>>
// CHECK-SAME:  >} {
// CHECK:        }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.device_id", 0: ui32>,
      #dlti.dl_entry<"dlti.device_type", "CPU">,
      #dlti.dl_entry<"dlti.max_vector_op_width", 64: ui32>>,
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.device_id", 1: ui32>,
      #dlti.dl_entry<"dlti.device_type", "GPU">,
      #dlti.dl_entry<"dlti.max_vector_op_width", 128: ui32>>
  >} {}
