// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error@below {{attribute 'dlti.unknown' not supported by dialect}}
"test.unknown_op"() { dlti.unknown } : () -> ()

// -----

// expected-error@below {{'dlti.map' is expected to be a #dlti.map attribute}}
"test.unknown_op"() { dlti.map = 42 } : () -> ()

// -----

// expected-error@below {{'dlti.dl_spec' is expected to be a #dlti.dl_spec attribute}}
"test.unknown_op"() { dlti.dl_spec = 42 } : () -> ()

// -----

// expected-error@below {{invalid kind of attribute specified}}
"test.unknown_op"() { dlti.dl_spec = #dlti.dl_spec<[]> } : () -> ()

// -----

// expected-error@below {{expected a type or a quoted string}}
"test.unknown_op"() { test.unknown_attr = #dlti.dl_entry<42, 42> } : () -> ()

// -----

// expected-error@below {{empty string as DLTI key is not allowed}}
"test.unknown_op"() { test.unknown_attr = #dlti.map<"" = 42> } : () -> ()

// -----

// expected-error@below {{repeated DLTI key: "test.id"}}
"test.unknown_op"() { test.unknown_attr = #dlti.dl_spec<
  #dlti.dl_entry<"test.id", 42>,
  #dlti.dl_entry<"test.id", 43>
>} : () -> ()

// -----

// expected-error@below {{repeated DLTI key: i32}}
"test.unknown_op"() { test.unknown_attr = #dlti.map<
  #dlti.dl_entry<i32, 42>,
  #dlti.dl_entry<i32, 42>
>} : () -> ()

// -----

// expected-error@below {{repeated DLTI key: i32}}
"test.unknown_op"() { test.unknown_attr = #dlti.dl_spec<
  #dlti.dl_entry<i32, 42>,
  #dlti.dl_entry<i32, 42>
>} : () -> ()

// -----

// expected-error@below {{unknown attribute `unknown` in dialect `dlti`}}
"test.unknown_op"() { test.unknown_attr = #dlti.unknown } : () -> ()

// -----

// expected-error@below {{unknown data layout entry name: dlti.unknown_id}}
"test.op_with_data_layout"() ({
}) { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.unknown_id", 42>> } : () -> ()

// -----

// expected-error@below {{'dlti.endianness' data layout entry is expected to be either 'big' or 'little'}}
"test.op_with_data_layout"() ({
}) { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "some">> } : () -> ()

// -----

// Mismatching entries don't combine.
"test.op_with_data_layout"() ({
  // expected-error@below {{data layout does not combine with layouts of enclosing ops}}
  // expected-note@above {{enclosing op with data layout}}
  "test.op_with_data_layout"() { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 32>> } : () -> ()
  "test.maybe_terminator_op"() : () -> ()
}) { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 33>> } : () -> ()

// -----

// Layout not supported some built-in types.
// expected-error@below {{unexpected data layout for a built-in type}}
"test.op_with_data_layout"() { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<tensor<f32>, 32>> } : () -> ()

// -----

// expected-error@below {{data layout specified for a type that does not support it}}
"test.op_with_data_layout"() { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!test.test_type, 32>> } : () -> ()

// -----

// Mismatching entries are checked on module ops as well.
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 33>>} {
  // expected-error@below {{data layout does not combine with layouts of enclosing ops}}
  // expected-note@above {{enclosing op with data layout}}
  module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 32>>} {
  }
}

// -----

// Mismatching entries are checked on a combination of modules and other ops.
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 33>>} {
  // expected-error@below {{data layout does not combine with layouts of enclosing ops}}
  // expected-note@above {{enclosing op with data layout}}
  "test.op_with_data_layout"() { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 32>>} : () -> ()
}

// -----

// expected-error@below {{'dlti.target_system_spec' is expected to be a #dlti.target_system_spec attribute}}
"test.unknown_op"() { dlti.target_system_spec = 42 } : () -> ()

// -----

// expected-error@below {{invalid kind of attribute specified}}
"test.unknown_op"() { dlti.target_system_spec = #dlti.target_system_spec<[]> } : () -> ()

// -----

module attributes {
  // Device ID is missing
  //
  // expected-error@below {{expected attribute value}}
  dlti.target_system_spec = #dlti.target_system_spec<
    = #dlti.target_device_spec<
      #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>
  >} {}

// -----

module attributes {
  // Device ID is wrong type
  //
  // expected-error@+2 {{invalid kind of attribute specified}}
  dlti.target_system_spec = #dlti.target_system_spec<
    0 = #dlti.target_device_spec<
        #dlti.dl_entry<"L1_cache_size_in_bytes", 4096 : i32>>
  >} {}

// -----

module attributes {
  // Repeated Device ID
  //
  // expected-error@+1 {{repeated device ID in dlti.target_system_spec: "CPU}}
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
            #dlti.dl_entry<"L1_cache_size_in_bytes", 4096>>,
    "CPU" = #dlti.target_device_spec<
            #dlti.dl_entry<"L1_cache_size_in_bytes", 8192>>
  >} {}

// -----

module attributes {
  // Repeated DLTI entry
  //
  // expected-error@+2 {{repeated DLTI key: "L1_cache_size_in_bytes"}}
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<"L1_cache_size_in_bytes" = 4096,
                                     "L1_cache_size_in_bytes" = 8192>
  >} {}
