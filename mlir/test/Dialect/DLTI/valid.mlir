// RUN: mlir-opt -split-input-file %s | FileCheck %s
// RUN: mlir-opt -split-input-file --test-data-layout-query %s | FileCheck %s --check-prefix=BUILTIN


// CHECK:      module attributes {
// CHECK-SAME:   dlti.map = #dlti.map<
// CHECK-SAME:     "magic_num" = 42 : i32,
// CHECK-SAME:     "magic_num_float" = 4.242000e+01 : f32,
// CHECK-SAME:     "magic_type" = i32,
// CHECK-SAME:     i32 = #dlti.map<"bitwidth" = 32 : i32>
// CHECK:        >} {
// CHECK:      }
module attributes {
  dlti.map = #dlti.map<"magic_num" = 42 : i32,
                       "magic_num_float" = 42.42 : f32,
                       "magic_type" = i32,
                        i32 = #dlti.map<"bitwidth" = 32 : i32>>
  } {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.map = #dlti.map<
// CHECK-SAME:     "magic_num" = 42 : i32,
// CHECK-SAME:     "magic_num_float" = 4.242000e+01 : f32,
// CHECK-SAME:     "magic_type" = i32,
// CHECK-SAME:     i32 = #dlti.map<"bitwidth" = 32 : i32>
// CHECK:        >} {
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
// CHECK-SAME:   dlti.map = #dlti.map<
// CHECK-SAME:     "CPU" = #dlti.map<"L1_cache_size_in_bytes" = 4096 : i32>,
// CHECK-SAME:     "GPU" = #dlti.map<"max_vector_op_width" = 128 : i32>
// CHECK-SAME:   >} {
// CHECK:      }
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
// CHECK-SAME:    "CPU" = #dlti.target_device_spec<
// CHECK-SAME:      "L1_cache_size_in_bytes" = 4096 : i32>,
// CHECK-SAME:    "GPU" = #dlti.target_device_spec<
// CHECK-SAME:      "max_vector_op_width" = 128 : i32>
// CHECK-SAME:  >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128 : i32>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:  dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:    "CPU" = #dlti.target_device_spec<
// CHECK-SAME:      "L1_cache_size_in_bytes" = 4096 : i32>,
// CHECK-SAME:    "GPU" = #dlti.target_device_spec<
// CHECK-SAME:      "L1_cache_size_in_bytes" = 8192 : i32>
// CHECK-SAME:  >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 8192 : i32>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = 4096 : i64>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = 8192 : i64>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i64>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 8192 : i64>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 64 : i32>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 128 : i32>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 64 : i32>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128 : i32>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 64 : i64>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 128 : i64>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 64 : i64>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128 : i64>>
  >} {}

// -----

// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 64 : i64>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 128 : i64>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 64 : i64>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 128 : i64>>
  >} {}

// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = 4096 : ui32>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = "128">
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : ui32>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", "128">>
  >} {}

// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 4.096000e+03 : f32>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = "128">
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"max_vector_op_width", 4096.0 : f32>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", "128">>
  >} {}


// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "vector_unit" = #dlti.map<
// CHECK-SAME:         "max_op_width" = 4.096000e+03 : f32>>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = "128">
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"vector_unit", #dlti.map<
        #dlti.dl_entry<"max_op_width", 4096.0 : f32>>>>,
    "GPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", "128">>
  >} {}

// -----

// Check values of mixed type
//
// CHECK:      module attributes {
// CHECK-SAME:   dlti.target_system_spec = #dlti.target_system_spec<
// CHECK-SAME:     "CPU" = #dlti.target_device_spec<
// CHECK-SAME:       "L1_cache_size_in_bytes" = 4096 : ui32>,
// CHECK-SAME:     "GPU" = #dlti.target_device_spec<
// CHECK-SAME:       "max_vector_op_width" = 128 : i64>
// CHECK-SAME:   >} {
// CHECK:      }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<"L1_cache_size_in_bytes" = 4096 : ui32>,
    "GPU" = #dlti.target_device_spec<"max_vector_op_width" = 128>
  >} {}

// -----

// CHECK: "test.op_with_dlti_map"() ({
// CHECK: }) {dlti.map = #dlti.map<"dlti.unknown_id" = 42 : i64>}
"test.op_with_dlti_map"() ({
}) { dlti.map = #dlti.map<#dlti.dl_entry<"dlti.unknown_id", 42>> } : () -> ()

// -----

// CHECK: "test.op_with_dlti_map"() ({
// CHECK: }) {dlti.map = #dlti.map<i32 = 42 : i64>}
"test.op_with_dlti_map"() ({
}) { dlti.map = #dlti.map<#dlti.dl_entry<i32, 42>> } : () -> ()

// -----

// BUILTIN-LABEL: module @builtin_integer_entries
module @builtin_integer_entries attributes { dlti.dl_spec = #dlti.dl_spec<
  i32 = dense<32> : vector<2xi64>
>} {
  func.func @query() {
    "test.op_with_data_layout"() ({
        // BUILTIN: alignment = 4
        "test.data_layout_query"() : () -> i32
        // BUILTIN: alignment = 8
        "test.data_layout_query"() : () -> i64
        "test.maybe_terminator"() : () -> ()
      }) { dlti.dl_spec = #dlti.dl_spec<
        i32 = dense<32> : vector<2xi64>,
        i64 = dense<64> : vector<2xi64>
      >} : () -> ()
    return
  }
}

// -----

// BUILTIN-LABEL: module @builtin_integer_compatible_abi
module @builtin_integer_compatible_abi attributes { dlti.dl_spec = #dlti.dl_spec<
  i32 = dense<64> : vector<2xi64>
>} {
  func.func @query() {
    "test.op_with_data_layout"() ({
      // BUILTIN: alignment = 4
      "test.data_layout_query"() : () -> i32
      "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
      i32 = dense<32> : vector<2xi64>
    >} : () -> ()
    return
  }
}

// -----

// BUILTIN-LABEL: module @builtin_index_entry
module @builtin_index_entry attributes { dlti.dl_spec = #dlti.dl_spec<
  index = 32
>} {
  func.func @query() {
    "test.op_with_data_layout"() ({
      // BUILTIN: bitsize = 32
      // BUILTIN: index = 32
      "test.data_layout_query"() : () -> index
      "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
      index = 32
    >} : () -> ()
    return
  }
}

// -----

module attributes { dlti.dl_spec = #dlti.dl_spec<
  i32 = dense<0> : vector<2xi64>
>} {
  module attributes { dlti.dl_spec = #dlti.dl_spec<
    i32 = dense<0> : vector<2xi64>
  >} {
  }
}

// -----

// BUILTIN-LABEL: module @builtin_float_entry
module @builtin_float_entry attributes { dlti.dl_spec = #dlti.dl_spec<
  f32 = dense<32> : vector<2xi64>
>} {
  func.func @query() {
    "test.op_with_data_layout"() ({
      // BUILTIN: alignment = 4
      "test.data_layout_query"() : () -> f32
      "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
      f32 = dense<32> : vector<2xi64>
    >} : () -> ()
    return
  }
}
