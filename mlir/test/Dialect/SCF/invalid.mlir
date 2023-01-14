// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func.func @loop_for_lb(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #0 must be index}}
  "scf.for"(%arg0, %arg1, %arg1) ({}) : (f32, index, index) -> ()
  return
}

// -----

func.func @loop_for_ub(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #1 must be index}}
  "scf.for"(%arg1, %arg0, %arg1) ({}) : (index, f32, index) -> ()
  return
}

// -----

func.func @loop_for_step(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #2 must be index}}
  "scf.for"(%arg1, %arg1, %arg0) ({}) : (index, index, f32) -> ()
  return
}

// -----

func.func @loop_for_step_positive(%arg0: index) {
  // expected-error@+2 {{constant step operand must be positive}}
  %c0 = arith.constant 0 : index
  "scf.for"(%arg0, %arg0, %c0) ({
    ^bb0(%arg1: index):
      scf.yield
  }) : (index, index, index) -> ()
  return
}

// -----

func.func @loop_for_one_region(%arg0: index) {
  // expected-error@+1 {{requires one region}}
  "scf.for"(%arg0, %arg0, %arg0) (
    {scf.yield},
    {scf.yield}
  ) : (index, index, index) -> ()
  return
}

// -----

func.func @loop_for_single_block(%arg0: index) {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  "scf.for"(%arg0, %arg0, %arg0) (
    {
    ^bb1:
      scf.yield
    ^bb2:
      scf.yield
    }
  ) : (index, index, index) -> ()
  return
}

// -----

func.func @loop_for_single_index_argument(%arg0: index) {
  // expected-error@+1 {{op expected body first argument to be an index argument for the induction variable}}
  "scf.for"(%arg0, %arg0, %arg0) (
    {
    ^bb0(%i0 : f32):
      scf.yield
    }
  ) : (index, index, index) -> ()
  return
}

// -----

func.func @loop_if_not_i1(%arg0: index) {
  // expected-error@+1 {{operand #0 must be 1-bit signless integer}}
  "scf.if"(%arg0) ({}, {}) : (index) -> ()
  return
}

// -----

func.func @loop_if_more_than_2_regions(%arg0: i1) {
  // expected-error@+1 {{expected 2 regions}}
  "scf.if"(%arg0) ({}, {}, {}): (i1) -> ()
  return
}

// -----

func.func @loop_if_not_one_block_per_region(%arg0: i1) {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  "scf.if"(%arg0) ({
    ^bb0:
      scf.yield
    ^bb1:
      scf.yield
  }, {}): (i1) -> ()
  return
}

// -----

func.func @loop_if_illegal_block_argument(%arg0: i1) {
  // expected-error@+1 {{region #0 should have no arguments}}
  "scf.if"(%arg0) ({
    ^bb0(%0 : index):
      scf.yield
  }, {}): (i1) -> ()
  return
}

// -----

func.func @parallel_arguments_different_tuple_size(
    %arg0: index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{custom op 'scf.parallel' expected 1 operands}}
  scf.parallel (%i0) = (%arg0) to (%arg1, %arg2) step () {
  }
  return
}

// -----

func.func @parallel_body_arguments_wrong_type(
    %arg0: index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{'scf.parallel' op expects arguments for the induction variable to be of index type}}
  "scf.parallel"(%arg0, %arg1, %arg2) ({
    ^bb0(%i0: f32):
      scf.yield
  }) {operand_segment_sizes = array<i32: 1, 1, 1, 0>}: (index, index, index) -> ()
  return
}

// -----

func.func @parallel_body_wrong_number_of_arguments(
    %arg0: index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{'scf.parallel' op expects the same number of induction variables: 2 as bound and step values: 1}}
  "scf.parallel"(%arg0, %arg1, %arg2) ({
    ^bb0(%i0: index, %i1: index):
      scf.yield
  }) {operand_segment_sizes = array<i32: 1, 1, 1, 0>}: (index, index, index) -> ()
  return
}

// -----

func.func @parallel_no_tuple_elements() {
  // expected-error@+1 {{'scf.parallel' op needs at least one tuple element for lowerBound, upperBound and step}}
  scf.parallel () = () to () step () {
  }
  return
}

// -----

func.func @parallel_step_not_positive(
    %arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  // expected-error@+3 {{constant step operand must be positive}}
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 0 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3) step (%c0, %c1) {
  }
  return
}

// -----

func.func @parallel_fewer_results_than_reduces(
    %arg0 : index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{expects number of results: 0 to be the same as number of reductions: 1}}
  scf.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) {
    %c0 = arith.constant 1.0 : f32
    scf.reduce(%c0) : f32 {
      ^bb0(%lhs: f32, %rhs: f32):
        scf.reduce.return %lhs : f32
    }
  }
  return
}

// -----

func.func @parallel_more_results_than_reduces(
    %arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+2 {{expects number of results: 1 to be the same as number of reductions: 0}}
  %zero = arith.constant 1.0 : f32
  %res = scf.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) init (%zero) -> f32 {
  }

  return
}

// -----

func.func @parallel_more_results_than_initial_values(
    %arg0 : index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{'scf.parallel' 0 operands present, but expected 1}}
  %res = scf.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) -> f32 {
    scf.reduce(%arg0) : index {
      ^bb0(%lhs: index, %rhs: index):
        scf.reduce.return %lhs : index
    }
  }
}

// -----

func.func @parallel_different_types_of_results_and_reduces(
    %arg0 : index, %arg1: index, %arg2: index) {
  %zero = arith.constant 0.0 : f32
  %res = scf.parallel (%i0) = (%arg0) to (%arg1)
                                       step (%arg2) init (%zero) -> f32 {
    // expected-error@+1 {{expects type of reduce: 'index' to be the same as result type: 'f32'}}
    scf.reduce(%arg0) : index {
      ^bb0(%lhs: index, %rhs: index):
        scf.reduce.return %lhs : index
    }
  }
  return
}

// -----

func.func @top_level_reduce(%arg0 : f32) {
  // expected-error@+1 {{expects parent op 'scf.parallel'}}
  scf.reduce(%arg0) : f32 {
    ^bb0(%lhs : f32, %rhs : f32):
      scf.reduce.return %lhs : f32
  }
  return
}

// -----

func.func @reduce_empty_block(%arg0 : index, %arg1 : f32) {
  %zero = arith.constant 0.0 : f32
  %res = scf.parallel (%i0) = (%arg0) to (%arg0)
                                       step (%arg0) init (%zero) -> f32 {
    // expected-error@+1 {{empty block: expect at least a terminator}}
    scf.reduce(%arg1) : f32 {
      ^bb0(%lhs : f32, %rhs : f32):
    }
  }
  return
}

// -----

func.func @reduce_too_many_args(%arg0 : index, %arg1 : f32) {
  %zero = arith.constant 0.0 : f32
  %res = scf.parallel (%i0) = (%arg0) to (%arg0)
                                       step (%arg0) init (%zero) -> f32 {
    // expected-error@+1 {{expects two arguments to reduce block of type 'f32'}}
    scf.reduce(%arg1) : f32 {
      ^bb0(%lhs : f32, %rhs : f32, %other : f32):
        scf.reduce.return %lhs : f32
    }
  }
  return
}

// -----

func.func @reduce_wrong_args(%arg0 : index, %arg1 : f32) {
  %zero = arith.constant 0.0 : f32
  %res = scf.parallel (%i0) = (%arg0) to (%arg0)
                                       step (%arg0) init (%zero) -> f32 {
    // expected-error@+1 {{expects two arguments to reduce block of type 'f32'}}
    scf.reduce(%arg1) : f32 {
      ^bb0(%lhs : f32, %rhs : i32):
        scf.reduce.return %lhs : f32
    }
  }
  return
}


// -----

func.func @reduce_wrong_terminator(%arg0 : index, %arg1 : f32) {
  %zero = arith.constant 0.0 : f32
  %res = scf.parallel (%i0) = (%arg0) to (%arg0)
                                       step (%arg0) init (%zero) -> f32 {
    // expected-error@+1 {{the block inside reduce should be terminated with a 'scf.reduce.return' op}}
    scf.reduce(%arg1) : f32 {
      ^bb0(%lhs : f32, %rhs : f32):
        "test.finish" () : () -> ()
    }
  }
  return
}

// -----

func.func @reduceReturn_wrong_type(%arg0 : index, %arg1: f32) {
  %zero = arith.constant 0.0 : f32
  %res = scf.parallel (%i0) = (%arg0) to (%arg0)
                                       step (%arg0) init (%zero) -> f32 {
    scf.reduce(%arg1) : f32 {
      ^bb0(%lhs : f32, %rhs : f32):
        %c0 = arith.constant 1 : index
        // expected-error@+1 {{needs to have type 'f32' (the type of the enclosing ReduceOp)}}
        scf.reduce.return %c0 : index
    }
  }
  return
}

// -----

func.func @reduceReturn_not_inside_reduce(%arg0 : f32) {
  "foo.region"() ({
    // expected-error@+1 {{expects parent op 'scf.reduce'}}
    scf.reduce.return %arg0 : f32
  }): () -> ()
  return
}

// -----

func.func @std_if_incorrect_yield(%arg0: i1, %arg1: f32)
{
  // expected-error@+1 {{region control flow edge from Region #0 to parent results: source has 1 operands, but target successor needs 2}}
  %x, %y = scf.if %arg0 -> (f32, f32) {
    %0 = arith.addf %arg1, %arg1 : f32
    scf.yield %0 : f32
  } else {
    %0 = arith.subf %arg1, %arg1 : f32
    scf.yield %0, %0 : f32, f32
  }
  return
}

// -----

func.func @std_if_missing_else(%arg0: i1, %arg1: f32)
{
  // expected-error@+1 {{must have an else block if defining values}}
  %x = scf.if %arg0 -> (f32) {
    %0 = arith.addf %arg1, %arg1 : f32
    scf.yield %0 : f32
  }
  return
}

// -----

func.func @std_for_operands_mismatch(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = arith.constant 0.0 : f32
  %t0 = arith.constant 1 : i32
  // expected-error@+1 {{mismatch in number of loop-carried values and defined values}}
  %result1:3 = scf.for %i0 = %arg0 to %arg1 step %arg2
                    iter_args(%si = %s0, %ti = %t0) -> (f32, i32, f32) {
    %sn = arith.addf %si, %si : f32
    %tn = arith.addi %ti, %ti : i32
    scf.yield %sn, %tn, %sn : f32, i32, f32
  }
  return
}

// -----

func.func @std_for_operands_mismatch_2(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = arith.constant 0.0 : f32
  %t0 = arith.constant 1 : i32
  %u0 = arith.constant 1.0 : f32
  // expected-error@+1 {{mismatch in number of loop-carried values and defined values}}
  %result1:2 = scf.for %i0 = %arg0 to %arg1 step %arg2
                    iter_args(%si = %s0, %ti = %t0, %ui = %u0) -> (f32, i32) {
    %sn = arith.addf %si, %si : f32
    %tn = arith.addi %ti, %ti : i32
    %un = arith.subf %ui, %ui : f32
    scf.yield %sn, %tn, %un : f32, i32, f32
  }
  return
}

// -----

func.func @std_for_operands_mismatch_3(%arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-note@+1 {{prior use here}}
  %s0 = arith.constant 0.0 : f32
  %t0 = arith.constant 1.0 : f32
  // expected-error@+2 {{expects different type than prior uses: 'i32' vs 'f32'}}
  %result1:2 = scf.for %i0 = %arg0 to %arg1 step %arg2
                    iter_args(%si = %s0, %ti = %t0) -> (i32, i32) {
    %sn = arith.addf %si, %si : i32
    %tn = arith.addf %ti, %ti : i32
    scf.yield %sn, %tn : i32, i32
  }
  return
}

// -----

func.func @std_for_operands_mismatch_4(%arg0 : index, %arg1 : index, %arg2 : index) {
  %s0 = arith.constant 0.0 : f32
  %t0 = arith.constant 1.0 : f32
  // expected-error @+1 {{along control flow edge from Region #0 to Region #0: source type #1 'i32' should match input type #1 'f32'}}
  %result1:2 = scf.for %i0 = %arg0 to %arg1 step %arg2
                    iter_args(%si = %s0, %ti = %t0) -> (f32, f32) {
    %sn = arith.addf %si, %si : f32
    %ic = arith.constant 1 : i32
    scf.yield %sn, %ic : f32, i32
  }
  return
}


// -----

func.func @parallel_invalid_yield(
    %arg0: index, %arg1: index, %arg2: index) {
  scf.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) {
    %c0 = arith.constant 1.0 : f32
    // expected-error@+1 {{'scf.yield' op not allowed to have operands inside 'scf.parallel'}}
    scf.yield %c0 : f32
  }
  return
}

// -----

func.func @yield_invalid_parent_op() {
  "my.op"() ({
   // expected-error@+1 {{'scf.yield' op expects parent op to be one of 'scf.execute_region, scf.for, scf.if, scf.index_switch, scf.parallel, scf.while'}}
   scf.yield
  }) : () -> ()
  return
}

// -----

func.func @while_parser_type_mismatch() {
  %true = arith.constant true
  // expected-error@+1 {{expected as many input types as operands (expected 0 got 1)}}
  scf.while : (i32) -> () {
    scf.condition(%true)
  } do {
    scf.yield
  }
}

// -----

func.func @while_bad_terminator() {
  // expected-error@+1 {{expects the 'before' region to terminate with 'scf.condition'}}
  scf.while : () -> () {
    // expected-note@+1 {{terminator here}}
    "some.other_terminator"() : () -> ()
  } do {
    scf.yield
  }
}

// -----

func.func @while_empty_region() {
  // expected-error@+1 {{'scf.while' op region #0 ('before') failed to verify constraint: region with 1 blocks}}
  scf.while : () -> () {
  } do {
  }
}

// -----

func.func @while_empty_block() {
  // expected-error@+1 {{expects the 'before' region to terminate with 'scf.condition'}}
  scf.while : () -> () {
   ^bb0:
  } do {
   ^bb0:
  }
}

// -----

func.func @while_cross_region_type_mismatch() {
  %true = arith.constant true
  // expected-error@+1 {{'scf.while' op  region control flow edge from Region #0 to Region #1: source has 0 operands, but target successor needs 1}}
  scf.while : () -> () {
    scf.condition(%true)
  } do {
  ^bb0(%arg0: i32):
    scf.yield
  }
}

// -----

func.func @while_cross_region_type_mismatch() {
  %true = arith.constant true
  // expected-error@+1 {{'scf.while' op  along control flow edge from Region #0 to Region #1: source type #0 'i1' should match input type #0 'i32'}}
  %0 = scf.while : () -> (i1) {
    scf.condition(%true) %true : i1
  } do {
  ^bb0(%arg0: i32):
    scf.yield
  }
}

// -----

func.func @while_result_type_mismatch() {
  %true = arith.constant true
  // expected-error@+1 {{'scf.while' op  region control flow edge from Region #0 to parent results: source has 1 operands, but target successor needs 0}}
  scf.while : () -> () {
    scf.condition(%true) %true : i1
  } do {
  ^bb0(%arg0: i1):
    scf.yield
  }
}

// -----

func.func @while_bad_terminator() {
  %true = arith.constant true
  // expected-error@+1 {{expects the 'after' region to terminate with 'scf.yield'}}
  scf.while : () -> () {
    scf.condition(%true)
  } do {
    // expected-note@+1 {{terminator here}}
    "some.other_terminator"() : () -> ()
  }
}

// -----

func.func @execute_region() {
  // expected-error @+1 {{region cannot have any arguments}}
  "scf.execute_region"() ({
  ^bb0(%i : i32):
    scf.yield
  }) : () -> ()
  return
}

// -----

func.func @wrong_num_results(%in: tensor<100xf32>, %out: tensor<100xf32>) {
  %c1 = arith.constant 1 : index
  %num_threads = arith.constant 100 : index

  // expected-error @+1 {{1 operands present, but expected 2}}
  %result:2 = scf.foreach_thread (%thread_idx) in (%num_threads) shared_outs(%o = %out) -> (tensor<100xf32>, tensor<100xf32>) {
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      scf.foreach_thread.perform_concurrently {
        tensor.parallel_insert_slice %1 into %o[%thread_idx][1][1] :
          tensor<1xf32> into tensor<100xf32>
      }
  }
  return
}

// -----

func.func @invalid_insert_dest(%in: tensor<100xf32>, %out: tensor<100xf32>) {
  %c1 = arith.constant 1 : index
  %num_threads = arith.constant 100 : index

  %result = scf.foreach_thread (%thread_idx) in (%num_threads) shared_outs(%o = %out) -> (tensor<100xf32>) {
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      scf.foreach_thread.perform_concurrently {
        // expected-error @+1 {{may only insert into an output block argument}}
        tensor.parallel_insert_slice %1 into %out[%thread_idx][1][1] :
          tensor<1xf32> into tensor<100xf32>
      }
  }
  return
}

// -----

func.func @wrong_terminator_op(%in: tensor<100xf32>, %out: tensor<100xf32>) {
  %c1 = arith.constant 1 : index
  %num_threads = arith.constant 100 : index

  %result = scf.foreach_thread (%thread_idx) in (%num_threads) shared_outs(%o = %out) -> (tensor<100xf32>) {
      %1 = tensor.extract_slice %in[%thread_idx][1][1] : tensor<100xf32> to tensor<1xf32>
      // expected-error @+1 {{expected only tensor.parallel_insert_slice ops}}
      scf.foreach_thread.perform_concurrently {
        tensor.parallel_insert_slice %1 into %o[%thread_idx][1][1] :
          tensor<1xf32> into tensor<100xf32>
        %0 = arith.constant 1: index
      }
  }
  return
}

// -----

func.func @mismatched_mapping(%x: memref<2 x 32 x f32>, %y: memref<2 x 32 x f32>, %t: memref<32 x f32>, %alpha : f32, %stream : !gpu.async.token) -> memref<2 x 32 x f32> {
  %one = arith.constant 1 : index
  %c65535 = arith.constant 65535 : index
  // expected-error @below {{'scf.foreach_thread' op mapping attribute size must match op rank}}
  scf.foreach_thread (%i, %j) in (%c65535, %c65535) {
      %4 = memref.load %x[%i, %j] : memref<2 x 32 x f32>
      %5 = memref.load %y[%i, %j] : memref<2 x 32 x f32>
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j] : memref<2 x 32 x f32>
  }  { mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>] }
  return %y : memref<2 x 32 x f32>
}

// -----

func.func @switch_wrong_case_count(%arg0: index) {
  // expected-error @below {{'scf.index_switch' op has 0 case regions but 1 case values}}
  "scf.index_switch"(%arg0) ({
    scf.yield
  }) {cases = array<i64: 1>} : (index) -> ()
  return
}

// -----

func.func @switch_duplicate_case(%arg0: index) {
  // expected-error @below {{'scf.index_switch' op has duplicate case value: 0}}
  scf.index_switch %arg0
  case 0 {
    scf.yield
  }
  case 0 {
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// -----

func.func @switch_wrong_types(%arg0: index) {
  // expected-error @below {{'scf.index_switch' op expected each region to return 0 values, but default region returns 1}}
  scf.index_switch %arg0
  default {
    // expected-note @below {{see yield operation here}}
    scf.yield %arg0 : index
  }
  return
}

// -----

func.func @switch_wrong_types(%arg0: index, %arg1: i32) {
  // expected-error @below {{'scf.index_switch' op expected result #0 of each region to be 'index'}}
  scf.index_switch %arg0 -> index
  case 0 {
    // expected-note @below {{case region #0 returns 'i32' here}}
    scf.yield %arg1 : i32
  }
  default {
    scf.yield %arg0 : index
  }
  return
}

// -----

func.func @switch_missing_terminator(%arg0: index, %arg1: i32) {
  // expected-error @below {{'scf.index_switch' op expected region to end with scf.yield, but got func.return}}
  "scf.index_switch"(%arg0) ({
    "scf.yield"() : () -> ()
  }, {
    return
  }) {cases = array<i64: 1>} : (index) -> ()
}
