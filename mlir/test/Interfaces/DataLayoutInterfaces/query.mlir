// RUN: mlir-opt --test-data-layout-query %s | FileCheck %s

// CHECK-LABEL: @no_layout_builtin
func.func @no_layout_builtin() {
  // CHECK: alignment = 4
  // CHECK: bitsize = 32
  // CHECK: index = 0
  // CHECK: preferred = 4
  // CHECK: size = 4
  "test.data_layout_query"() : () -> i32
  // CHECK: alignment = 8
  // CHECK: bitsize = 64
  // CHECK: index = 0
  // CHECK: preferred = 8
  // CHECK: size = 8
  "test.data_layout_query"() : () -> f64
  // CHECK: alignment = 4
  // CHECK: bitsize = 64
  // CHECK: index = 0
  // CHECK: preferred = 4
  // CHECK: size = 8
  "test.data_layout_query"() : () -> complex<f32>
  // CHECK: alignment = 1
  // CHECK: bitsize = 14
  // CHECK: index = 0
  // CHECK: preferred = 1
  // CHECK: size = 2
  "test.data_layout_query"() : () -> complex<i6>
  // CHECK: alignment = 4
  // CHECK: bitsize = 64
  // CHECK: index = 64
  // CHECK: preferred = 8
  // CHECK: size = 8
  "test.data_layout_query"() : () -> index
  return

}

// CHECK-LABEL: @no_layout_custom
func.func @no_layout_custom() {
  // CHECK: alignment = 1
  // CHECK: bitsize = 1
  // CHECK: index = 1
  // CHECK: preferred = 1
  // CHECK: size = 1
  "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
  return
}

// CHECK-LABEL: @layout_op_no_layout
func.func @layout_op_no_layout() {
  "test.op_with_data_layout"() ({
    // CHECK: alignment = 1
    // CHECK: bitsize = 1
    // CHECK: index = 1
    // CHECK: preferred = 1
    // CHECK: size = 1
    "test.data_layout_query"() : () -> !test.test_type_with_layout<1000>
    "test.maybe_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: @layout_op
func.func @layout_op() {
  "test.op_with_data_layout"() ({
    // CHECK: alignment = 20
    // CHECK: bitsize = 10
    // CHECK: index = 30
    // CHECK: preferred = 1
    // CHECK: size = 2
    "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
      #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>,
      #dlti.dl_entry<!test.test_type_with_layout<30>, ["index", 30]>
  >} : () -> ()
  return
}

// Make sure the outer op with layout may be missing the spec.
// CHECK-LABEL: @nested_inner_only
func.func @nested_inner_only() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      // CHECK: alignment = 20
      // CHECK: bitsize = 10
      // CHECK: index = 30
      // CHECK: preferred = 1
      // CHECK: size = 2
      "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
      "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
        #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>,
        #dlti.dl_entry<!test.test_type_with_layout<30>, ["index", 30]>
    >} : () -> ()
    "test.maybe_terminator"() : () -> ()
  }) : () -> ()
  return
}

// Make sure the inner op with layout may be missing the spec.
// CHECK-LABEL: @nested_outer_only
func.func @nested_outer_only() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      // CHECK: alignment = 20
      // CHECK: bitsize = 10
      // CHECK: index = 30
      // CHECK: preferred = 1
      // CHECK: size = 2
      "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
      "test.maybe_terminator"() : () -> ()
    }) : () -> ()
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
      #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>,
      #dlti.dl_entry<!test.test_type_with_layout<30>, ["index", 30]>
    >} : () -> ()
  return
}

// CHECK-LABEL: @nested_middle_only
func.func @nested_middle_only() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      "test.op_with_data_layout"() ({
        // CHECK: alignment = 20
        // CHECK: bitsize = 10
        // CHECK: index = 30
        // CHECK: preferred = 1
        // CHECK: size = 2
        "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
        "test.maybe_terminator"() : () -> ()
    }) : () -> ()
    "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
        #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>,
        #dlti.dl_entry<!test.test_type_with_layout<30>, ["index", 30]>
      >} : () -> ()
    "test.maybe_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: @nested_combine_with_missing
func.func @nested_combine_with_missing() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      "test.op_with_data_layout"() ({
        // CHECK: alignment = 20
        // CHECK: bitsize = 10
        // CHECK: index = 21
        // CHECK: preferred = 30
        // CHECK: size = 2
        "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
        "test.maybe_terminator"() : () -> ()
      }) : () -> ()
    "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
        #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>
      >} : () -> ()
    // CHECK: alignment = 1
    // CHECK: bitsize = 42
    // CHECK: index = 21
    // CHECK: preferred = 30
    // CHECK: size = 6
    "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 42]>,
      #dlti.dl_entry<!test.test_type_with_layout<30>, ["preferred", 30]>,
      #dlti.dl_entry<!test.test_type_with_layout<40>, ["index", 21]>
  >}: () -> ()
  return
}

// CHECK-LABEL: @nested_combine_all
func.func @nested_combine_all() {
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"() ({
      "test.op_with_data_layout"() ({
        // CHECK: alignment = 20
        // CHECK: bitsize = 3
        // CHECK: index = 40
        // CHECK: preferred = 30
        // CHECK: size = 1
        "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
        "test.maybe_terminator"() : () -> ()
      }) { dlti.dl_spec = #dlti.dl_spec<
          #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 3]>,
          #dlti.dl_entry<!test.test_type_with_layout<30>, ["preferred", 30]>
        >} : () -> ()
      // CHECK: alignment = 20
      // CHECK: bitsize = 10
      // CHECK: index = 40
      // CHECK: preferred = 30
      // CHECK: size = 2
      "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
      "test.maybe_terminator"() : () -> ()
    }) { dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 10]>,
        #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 20]>,
        #dlti.dl_entry<!test.test_type_with_layout<40>, ["index", 40]>
      >} : () -> ()
    // CHECK: alignment = 1
    // CHECK: bitsize = 42
    // CHECK: index = 1
    // CHECK: preferred = 30
    // CHECK: size = 6
    "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 42]>,
      #dlti.dl_entry<!test.test_type_with_layout<30>, ["preferred", 30]>
  >}: () -> ()
  return
}

// CHECK-LABEL: @integers
func.func @integers() {
  "test.op_with_data_layout"() ({
    // CHECK: alignment = 8
    // CHECK: bitsize = 32
    // CHECK: index = 0
    // CHECK: preferred = 8
    "test.data_layout_query"() : () -> i32
    // CHECK: alignment = 16
    // CHECK: bitsize = 56
    // CHECK: index = 0
    // CHECK: preferred = 16
    "test.data_layout_query"() : () -> i56
    // CHECK: alignment = 16
    // CHECK: bitsize = 64
    // CHECK: index = 0
    // CHECK: preferred = 16
    "test.data_layout_query"() : () -> i64
    // CHECK: alignment = 16
    // CHECK: bitsize = 128
    // CHECK: index = 0
    // CHECK: preferred = 16
    "test.data_layout_query"() : () -> i128
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<i32, dense<64> : vector<1xi64>>,
      #dlti.dl_entry<i64, dense<128> : vector<1xi64>>
    >} : () -> ()
  "test.op_with_data_layout"() ({
    // CHECK: alignment = 8
    // CHECK: bitsize = 32
    // CHECK: index = 0
    // CHECK: preferred = 16
    "test.data_layout_query"() : () -> i32
    // CHECK: alignment = 16
    // CHECK: bitsize = 56
    // CHECK: index = 0
    // CHECK: preferred = 32
    "test.data_layout_query"() : () -> i56
    // CHECK: alignment = 16
    // CHECK: bitsize = 64
    // CHECK: index = 0
    // CHECK: preferred = 32
    "test.data_layout_query"() : () -> i64
    // CHECK: alignment = 16
    // CHECK: bitsize = 128
    // CHECK: index = 0
    // CHECK: preferred = 32
    "test.data_layout_query"() : () -> i128
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<i32, dense<[64, 128]> : vector<2xi64>>,
      #dlti.dl_entry<i64, dense<[128, 256]> : vector<2xi64>>
    >} : () -> ()
  return
}

func.func @floats() {
  "test.op_with_data_layout"() ({
    // CHECK: alignment = 8
    // CHECK: bitsize = 32
    // CHECK: index = 0
    // CHECK: preferred = 8
    "test.data_layout_query"() : () -> f32
    // CHECK: alignment = 16
    // CHECK: bitsize = 80
    // CHECK: index = 0
    // CHECK: preferred = 16
    "test.data_layout_query"() : () -> f80
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<f32, dense<64> : vector<1xi64>>,
      #dlti.dl_entry<f80, dense<128> : vector<1xi64>>
    >} : () -> ()
  "test.op_with_data_layout"() ({
    // CHECK: alignment = 8
    // CHECK: bitsize = 32
    // CHECK: index = 0
    // CHECK: preferred = 16
    "test.data_layout_query"() : () -> f32
    // CHECK: alignment = 16
    // CHECK: bitsize = 80
    // CHECK: index = 0
    // CHECK: preferred = 32
    "test.data_layout_query"() : () -> f80
    "test.maybe_terminator"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<f32, dense<[64, 128]> : vector<2xi64>>,
      #dlti.dl_entry<f80, dense<[128, 256]> : vector<2xi64>>
    >} : () -> ()
  return
}
