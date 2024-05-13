// RUN: mlir-opt %s -split-input-file --test-operations-equality | FileCheck %s


// CHECK-LABEL: test.top_level_op
// CHECK-SAME: compares equals

"test.top_level_op"() : () -> ()
"test.top_level_op"() : () -> ()

// -----

// CHECK-LABEL: test.top_level_op_strict_loc
// CHECK-SAME: compares NOT equals

"test.top_level_op_strict_loc"() { strict_loc_check } : () -> ()
"test.top_level_op_strict_loc"() { strict_loc_check } : () -> ()

// -----

// CHECK-LABEL: test.top_level_op_loc_match
// CHECK-SAME: compares equals

"test.top_level_op_loc_match"() { strict_loc_check } : () -> () loc("foo")
"test.top_level_op_loc_match"() { strict_loc_check } : () -> () loc("foo")

// -----

// CHECK-LABEL: test.top_level_op_block_loc_mismatch
// CHECK-SAME: compares NOT equals

"test.top_level_op_block_loc_mismatch"() ({
 ^bb0(%a : i32):
}) { strict_loc_check } : () -> () loc("foo")
"test.top_level_op_block_loc_mismatch"() ({
 ^bb0(%a : i32):
}) { strict_loc_check } : () -> () loc("foo")

// -----

// CHECK-LABEL: test.top_level_op_block_loc_match
// CHECK-SAME: compares equals

"test.top_level_op_block_loc_match"() ({
 ^bb0(%a : i32 loc("bar")):
}) { strict_loc_check } : () -> () loc("foo")
"test.top_level_op_block_loc_match"() ({
 ^bb0(%a : i32 loc("bar")):
}) { strict_loc_check } : () -> () loc("foo")

// -----

// CHECK-LABEL: test.top_level_name_mismatch
// CHECK-SAME: compares NOT equals

"test.top_level_name_mismatch"() : () -> ()
"test.top_level_name_mismatch2"() : () -> ()

// -----

// CHECK-LABEL: test.top_level_op_attr_mismatch
// CHECK-SAME: compares NOT equals

"test.top_level_op_attr_mismatch"() { foo = "bar" } : () -> ()
"test.top_level_op_attr_mismatch"() { foo = "bar2"} : () -> ()

// -----

// CHECK-LABEL: test.top_level_op_cfg
// CHECK-SAME: compares equals

"test.top_level_op_cfg"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg0) [^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%arg2 : f32):
    "test.some_branching_op"() : () -> ()
  ^bb2(%arg3 : i32):
    "test.some_branching_op"() : () -> ()
  }, {
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg0) [^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%arg2 : f32):
    "test.some_branching_op"() : () -> ()
  ^bb2(%arg3 : i32):
    "test.some_branching_op"() : () -> ()
  })
   { attr = "foo" } : () -> ()
"test.top_level_op_cfg"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg0) [^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%arg2 : f32):
    "test.some_branching_op"() : () -> ()
  ^bb2(%arg3 : i32):
    "test.some_branching_op"() : () -> ()
  }, {
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg0) [^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%arg2 : f32):
    "test.some_branching_op"() : () -> ()
  ^bb2(%arg3 : i32):
    "test.some_branching_op"() : () -> ()
  })
   { attr = "foo" } : () -> ()

// -----

// CHECK-LABEL: test.operand_num_mismatch
// CHECK-SAME: compares NOT equals

"test.operand_num_mismatch"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg0) : (f32, i32) -> ()
  }) : () -> ()
"test.operand_num_mismatch"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1) : (f32) -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.operand_type_mismatch
// CHECK-SAME: compares NOT equals

"test.operand_type_mismatch"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg0) : (f32, i32) -> ()
  }) : () -> ()
"test.operand_type_mismatch"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg1) : (f32, f32) -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.block_type_mismatch
// CHECK-SAME: compares NOT equals

"test.block_type_mismatch"() ({
  ^bb0(%arg0 : f32, %arg1 : f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
"test.block_type_mismatch"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.block_arg_num_mismatch
// CHECK-SAME: compares NOT equals

"test.block_arg_num_mismatch"() ({
  ^bb0(%arg0 : f32, %arg1 : f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()
"test.block_arg_num_mismatch"() ({
  ^bb0(%arg0 : f32):
    "test.some_branching_op"() : () -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.dataflow_match
// CHECK-SAME: compares equals

"test.dataflow_match"() ({
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
"test.dataflow_match"() ({
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.dataflow_mismatch
// CHECK-SAME: compares NOT equals

"test.dataflow_mismatch"() ({
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
"test.dataflow_mismatch"() ({
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.consumer"(%0#1, %0#0) : (i32, i32) -> ()
  }) : () -> ()
