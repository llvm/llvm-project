// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// expected-error@below {{attribute 'dlti.unknown' not supported by dialect}}
"test.unknown_op"() { dlti.unknown } : () -> ()

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

// expected-error@below {{repeated layout entry key: test.id}}
"test.unknown_op"() { test.unknown_attr = #dlti.dl_spec<
  #dlti.dl_entry<"test.id", 42>,
  #dlti.dl_entry<"test.id", 43>
>} : () -> ()

// -----

// expected-error@below {{repeated layout entry key: 'i32'}}
"test.unknown_op"() { test.unknown_attr = #dlti.dl_spec<
  #dlti.dl_entry<i32, 42>,
  #dlti.dl_entry<i32, 42>
>} : () -> ()

// -----

// expected-error@below {{unknown attrribute type: unknown}}
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

// Layout not supported for built-in types.
// expected-error@below {{unexpected data layout for a built-in type}}
"test.op_with_data_layout"() { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, 32>> } : () -> ()

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
