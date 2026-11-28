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

// -----

// CHECK-LABEL: test.commutatively_equal_permutation
// CHECK-SAME: compares equals

builtin.module attributes {test.includes_setup} {
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.commutatively_equal_permutation"() ({
    arith.addi %0#0, %0#1 : i32
  }) : () -> ()
  "test.commutatively_equal_permutation"() ({
    arith.addi %0#1, %0#0 : i32
  }) : () -> ()
}

// -----

// CHECK-LABEL: test.ignore_commutatively_equal_permutation
// CHECK-SAME: compares NOT equals

builtin.module attributes {test.includes_setup} {
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.ignore_commutatively_equal_permutation"() ({
    arith.addi %0#0, %0#1 : i32
  }) { ignore_commutativity } : () -> ()
  "test.ignore_commutatively_equal_permutation"() ({
    arith.addi %0#1, %0#0 : i32
  }) { ignore_commutativity } : () -> ()
}

// -----

// CHECK-LABEL: test.commutatively_equal
// CHECK-SAME: compares equals

"test.commutatively_equal"() ({
  ^bb0(%arg0 : i32, %arg1 : i32):
    arith.addi %arg0, %arg1 : i32
  }) : () -> ()
"test.commutatively_equal"() ({
  ^bb0(%arg0 : i32, %arg1 : i32):
    arith.addi %arg1, %arg0 : i32
  }) : () -> ()

// -----

// CHECK-LABEL: test.ignore_commutatively_equal
// CHECK-SAME: compares NOT equals

"test.ignore_commutatively_equal"() ({
  ^bb0(%arg0 : i32, %arg1 : i32):
    arith.addi %arg0, %arg1 : i32
  }) { ignore_commutativity } : () -> ()
"test.ignore_commutatively_equal"() ({
  ^bb0(%arg0 : i32, %arg1 : i32):
    arith.addi %arg1, %arg0 : i32
  }) { ignore_commutativity } : () -> ()

// -----

// Dominating block ^bb2 appears after ^bb1 in lexical order.
// CHECK-LABEL: test.cfg_non_lexical_dominance_match
// CHECK-SAME: compares equals

"test.cfg_non_lexical_dominance_match"() ({
  ^bb0(%arg0 : i32):
    "test.br"(%arg0) [^bb2] : (i32) -> ()
  ^bb1:
    %sum = arith.addi %dom_arg, %dom_val : i32
    "test.consumer"(%sum) : (i32) -> ()
  ^bb2(%dom_arg : i32):
    %dom_val = "test.producer"() : () -> i32
    "test.cond_br"() [^bb1, ^bb3] : () -> ()
  ^bb3:
    "test.end"() : () -> ()
  }) : () -> ()
"test.cfg_non_lexical_dominance_match"() ({
  ^bb0(%arg0 : i32):
    "test.br"(%arg0) [^bb2] : (i32) -> ()
  ^bb1:
    %sum = arith.addi %dom_val, %dom_arg : i32
    "test.consumer"(%sum) : (i32) -> ()
  ^bb2(%dom_arg : i32):
    %dom_val = "test.producer"() : () -> i32
    "test.cond_br"() [^bb1, ^bb3] : () -> ()
  ^bb3:
    "test.end"() : () -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.cfg_non_lexical_dominance_mismatch
// CHECK-SAME: compares NOT equals

"test.cfg_non_lexical_dominance_mismatch"() ({
  ^bb0:
    "test.br"() [^bb2] : () -> ()
  ^bb1:
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  ^bb2:
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.br"() [^bb1] : () -> ()
  }) : () -> ()
"test.cfg_non_lexical_dominance_mismatch"() ({
  ^bb0:
    "test.br"() [^bb2] : () -> ()
  ^bb1:
    "test.consumer"(%0#1, %0#0) : (i32, i32) -> ()
  ^bb2:
    %0:2 = "test.producer"() : () -> (i32, i32)
    "test.br"() [^bb1] : () -> ()
  }) : () -> ()

// -----

// Graph region with forward references and a self-loop.
// CHECK-LABEL: test.graph_region_forward_ref_match
// CHECK-SAME: compares equals

"test.graph_region_forward_ref_match"() ({
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
    %1 = "test.cycle"(%1) : (i32) -> i32
    %0:2 = "test.producer"(%1) : (i32) -> (i32, i32)
  }) : () -> ()
"test.graph_region_forward_ref_match"() ({
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
    %1 = "test.cycle"(%1) : (i32) -> i32
    %0:2 = "test.producer"(%1) : (i32) -> (i32, i32)
  }) : () -> ()

// -----

// CHECK-LABEL: test.graph_region_forward_ref_mismatch
// CHECK-SAME: compares NOT equals

"test.graph_region_forward_ref_mismatch"() ({
    "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
    %0:2 = "test.producer"() : () -> (i32, i32)
  }) : () -> ()
"test.graph_region_forward_ref_mismatch"() ({
    "test.consumer"(%0#1, %0#0) : (i32, i32) -> ()
    %0:2 = "test.producer"() : () -> (i32, i32)
  }) : () -> ()
