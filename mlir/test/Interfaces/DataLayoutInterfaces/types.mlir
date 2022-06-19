// RUN: mlir-opt --split-input-file --verify-diagnostics --test-data-layout-query %s | FileCheck %s

// expected-error@below {{expected integer attribute in the data layout entry for 'index'}}
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<index, [32]>>} {
}

// -----

// expected-error@below {{expected a dense i32 elements attribute}}
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<i32, dense<[64,128]> : vector<2xi64>>>
} {}

// -----

// expected-error@below {{expected 1 or 2 elements}}
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<i32, dense<[64,64,64]> : vector<3xi32>>>
} {}

// -----

// expected-error@below {{preferred alignment is expected to be greater than or equal to the abi alignment}}
module attributes {dlti.dl_spec = #dlti.dl_spec<
#dlti.dl_entry<i32, dense<[64,32]> : vector<2xi32>>>
} {}

// -----

// expected-error@below {{the 'test' dialect does not support identifier data layout entries}}
"test.op_with_data_layout"() { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<index, 32>,
  #dlti.dl_entry<"test.foo", [32]>>} : () -> ()

// -----

// CHECK-LABEL: @index
module @index attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<index, 32>>} {
  func.func @query() {
    // CHECK: bitsize = 32
    "test.data_layout_query"() : () -> index
    return
  }
}

// -----

// CHECK-LABEL: @index_default
module @index_default {
  func.func @query() {
    // CHECK: bitsize = 64
    "test.data_layout_query"() : () -> index
    return
  }
}
