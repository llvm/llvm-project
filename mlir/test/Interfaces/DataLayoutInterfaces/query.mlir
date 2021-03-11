// RUN: mlir-opt --test-data-layout-query %s | FileCheck %s

// CHECK-LABEL: @no_layout_builtin
func @no_layout_builtin() {
  // CHECK: alignment = 4
  // CHECK: preferred = 4
  // CHECK: size = 4
  "test.data_layout_query"() : () -> i32
  // CHECK: alignment = 8
  // CHECK: preferred = 8
  // CHECK: size = 8
  "test.data_layout_query"() : () -> f64
  return
}

// CHECK-LABEL: @no_layout_custom
func @no_layout_custom() {
  // CHECK: alignment = 1
  // CHECK: preferred = 1
  // CHECK: size = 1
  "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
  return
}

// CHECK-LABEL: @layout_op_no_layout
func @layout_op_no_layout() {
  "test.op_with_data_layout"() ({
    // CHECK: alignment = 1
    // CHECK: preferred = 1
    // CHECK: size = 1
    "test.data_layout_query"() : () -> !test.test_type_with_layout<1000>
    "test.maybe_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: @layout_op
func @layout_op() {
  "test.op_with_data_layout"() ({
    // CHECK: alignment = 20
    // CHECK: preferred = 1
    // CHECK: size = 10
    "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
      #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>
  >} : () -> ()
  return
}

// Make sure the outer op with layout may be missing the spec.
// CHECK-LABEL: @nested_inner_only
func @nested_inner_only() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      // CHECK: alignment = 20
      // CHECK: preferred = 1
      // CHECK: size = 10
      "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
      "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
        #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>
    >} : () -> ()
    "test.maybe_terminator"() : () -> ()
  }) : () -> ()
  return
}

// Make sure the inner op with layout may be missing the spec.
// CHECK-LABEL: @nested_outer_only
func @nested_outer_only() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      // CHECK: alignment = 20
      // CHECK: preferred = 1
      // CHECK: size = 10
      "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
      "test.maybe_terminator"() : () -> ()
    }) : () -> ()
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
      #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>
    >} : () -> ()
  return
}

// CHECK-LABEL: @nested_middle_only
func @nested_middle_only() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      "test.op_with_data_layout"() ({
        // CHECK: alignment = 20
        // CHECK: preferred = 1
        // CHECK: size = 10
        "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
        "test.maybe_terminator"() : () -> ()
    }) : () -> ()
    "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
        #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>
      >} : () -> ()
    "test.maybe_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: @nested_combine_with_missing
func @nested_combine_with_missing() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      "test.op_with_data_layout"() ({
        // CHECK: alignment = 20
        // CHECK: preferred = 30
        // CHECK: size = 10
        "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
        "test.maybe_terminator"() : () -> ()
      }) : () -> ()
    "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
        #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>
      >} : () -> ()
    // CHECK: alignment = 1
    // CHECK: preferred = 30
    // CHECK: size = 42
    "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 42]>,
      #dlti.dl_entry<!test.test_type_with_layout<30>, ["preferred", 30]>
  >}: () -> ()
  return
}

// CHECK-LABEL: @nested_combine_all
func @nested_combine_all() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      "test.op_with_data_layout"() ({
        // CHECK: alignment = 20
        // CHECK: preferred = 30
        // CHECK: size = 3
        "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
        "test.maybe_terminator"() : () -> ()
      }) { dlti.dl_spec = #dlti.dl_spec<
          #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 3]>,
          #dlti.dl_entry<!test.test_type_with_layout<30>, ["preferred", 30]>
        >} : () -> ()
      // CHECK: alignment = 20
      // CHECK: preferred = 30
      // CHECK: size = 10
      "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
      "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
        #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>
      >} : () -> ()
    // CHECK: alignment = 1
    // CHECK: preferred = 30
    // CHECK: size = 42
    "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 42]>,
      #dlti.dl_entry<!test.test_type_with_layout<30>, ["preferred", 30]>
  >}: () -> ()
  return
}
