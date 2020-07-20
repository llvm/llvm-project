// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @loop_for_lb(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #0 must be index}}
  "loop.for"(%arg0, %arg1, %arg1) : (f32, index, index) -> ()
  return
}

// -----

func @loop_for_ub(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #1 must be index}}
  "loop.for"(%arg1, %arg0, %arg1) : (index, f32, index) -> ()
  return
}

// -----

func @loop_for_step(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #2 must be index}}
  "loop.for"(%arg1, %arg1, %arg0) : (index, index, f32) -> ()
  return
}

// -----

func @loop_for_step_positive(%arg0: index) {
  // expected-error@+2 {{constant step operand must be positive}}
  %c0 = constant 0 : index
  "loop.for"(%arg0, %arg0, %c0) ({
    ^bb0(%arg1: index):
      "loop.terminator"() : () -> ()
  }) : (index, index, index) -> ()
  return
}

// -----

func @loop_for_one_region(%arg0: index) {
  // expected-error@+1 {{incorrect number of regions: expected 1 but found 2}}
  "loop.for"(%arg0, %arg0, %arg0) (
    {"loop.terminator"() : () -> ()},
    {"loop.terminator"() : () -> ()}
  ) : (index, index, index) -> ()
  return
}

// -----

func @loop_for_single_block(%arg0: index) {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  "loop.for"(%arg0, %arg0, %arg0) (
    {
    ^bb1:
      "loop.terminator"() : () -> ()
    ^bb2:
      "loop.terminator"() : () -> ()
    }
  ) : (index, index, index) -> ()
  return
}

// -----

func @loop_for_single_index_argument(%arg0: index) {
  // expected-error@+1 {{expected body to have a single index argument for the induction variable}}
  "loop.for"(%arg0, %arg0, %arg0) (
    {
    ^bb0(%i0 : f32):
      "loop.terminator"() : () -> ()
    }
  ) : (index, index, index) -> ()
  return
}

// -----

func @loop_if_not_i1(%arg0: index) {
  // expected-error@+1 {{operand #0 must be 1-bit integer}}
  "loop.if"(%arg0) : (index) -> ()
  return
}

// -----

func @loop_if_more_than_2_regions(%arg0: i1) {
  // expected-error@+1 {{op has incorrect number of regions: expected 2}}
  "loop.if"(%arg0) ({}, {}, {}): (i1) -> ()
  return
}

// -----

func @loop_if_not_one_block_per_region(%arg0: i1) {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  "loop.if"(%arg0) ({
    ^bb0:
      "loop.terminator"() : () -> ()
    ^bb1:
      "loop.terminator"() : () -> ()
  }, {}): (i1) -> ()
  return
}

// -----

func @loop_if_illegal_block_argument(%arg0: i1) {
  // expected-error@+1 {{requires that child entry blocks have no arguments}}
  "loop.if"(%arg0) ({
    ^bb0(%0 : index):
      "loop.terminator"() : () -> ()
  }, {}): (i1) -> ()
  return
}

// -----

func @parallel_arguments_different_tuple_size(
    %arg0: index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{custom op 'loop.parallel' expected 1 operands}}
  loop.parallel (%i0) = (%arg0) to (%arg1, %arg2) step () {
  }
  return
}

// -----

func @parallel_body_arguments_wrong_type(
    %arg0: index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{'loop.parallel' op expects arguments for the induction variable to be of index type}}
  "loop.parallel"(%arg0, %arg1, %arg2) ({
    ^bb0(%i0: f32):
      "loop.terminator"() : () -> ()
  }): (index, index, index) -> ()
  return
}

// -----

func @parallel_body_wrong_number_of_arguments(
    %arg0: index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{'loop.parallel' op expects the same number of induction variables as bound and step values}}
  "loop.parallel"(%arg0, %arg1, %arg2) ({
    ^bb0(%i0: index, %i1: index):
      "loop.terminator"() : () -> ()
  }): (index, index, index) -> ()
  return
}

// -----

func @parallel_no_tuple_elements() {
  // expected-error@+1 {{'loop.parallel' op needs at least one tuple element for lowerBound, upperBound and step}}
  loop.parallel () = () to () step () {
  }
  return
}

// -----

func @parallel_step_not_positive(
    %arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  // expected-error@+3 {{constant step operand must be positive}}
  %c0 = constant 1 : index
  %c1 = constant 0 : index
  loop.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3) step (%c0, %c1) {
  }
  return
}

// -----

func @parallel_fewer_results_than_reduces(
    %arg0 : index, %arg1: index, %arg2: index) {
  // expected-error@+1 {{expects number of results to be the same as number of reductions}}
  loop.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) {
    %c0 = constant 1.0 : f32
    loop.reduce(%c0) {
      ^bb0(%lhs: f32, %rhs: f32):
        loop.reduce.return %lhs : f32
    } : f32
  }
  return
}

// -----

func @parallel_more_results_than_reduces(
    %arg0 : index, %arg1 : index, %arg2 : index) {
  // expected-error@+1 {{expects number of results to be the same as number of reductions}}
  %res = loop.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) {
  } : f32

  return
}

// -----

func @parallel_different_types_of_results_and_reduces(
    %arg0 : index, %arg1: index, %arg2: index) {
  %res = loop.parallel (%i0) = (%arg0) to (%arg1) step (%arg2) {
    // expected-error@+1 {{expects type of reduce to be the same as result type: 'f32'}}
    loop.reduce(%arg0) {
      ^bb0(%lhs: index, %rhs: index):
        loop.reduce.return %lhs : index
    } : index
  } : f32
  return
}

// -----

func @top_level_reduce(%arg0 : f32) {
  // expected-error@+1 {{expects parent op 'loop.parallel'}}
  loop.reduce(%arg0) {
    ^bb0(%lhs : f32, %rhs : f32):
      loop.reduce.return %lhs : f32
  } : f32
  return
}

// -----

func @reduce_empty_block(%arg0 : index, %arg1 : f32) {
  %res = loop.parallel (%i0) = (%arg0) to (%arg0) step (%arg0) {
    // expected-error@+1 {{the block inside reduce should not be empty}}
    loop.reduce(%arg1) {
      ^bb0(%lhs : f32, %rhs : f32):
    } : f32
  } : f32
  return
}

// -----

func @reduce_too_many_args(%arg0 : index, %arg1 : f32) {
  %res = loop.parallel (%i0) = (%arg0) to (%arg0) step (%arg0) {
    // expected-error@+1 {{expects two arguments to reduce block of type 'f32'}}
    loop.reduce(%arg1) {
      ^bb0(%lhs : f32, %rhs : f32, %other : f32):
        loop.reduce.return %lhs : f32
    } : f32
  } : f32
  return
}

// -----

func @reduce_wrong_args(%arg0 : index, %arg1 : f32) {
  %res = loop.parallel (%i0) = (%arg0) to (%arg0) step (%arg0) {
    // expected-error@+1 {{expects two arguments to reduce block of type 'f32'}}
    loop.reduce(%arg1) {
      ^bb0(%lhs : f32, %rhs : i32):
        loop.reduce.return %lhs : f32
    } : f32
  } : f32
  return
}


// -----

func @reduce_wrong_terminator(%arg0 : index, %arg1 : f32) {
  %res = loop.parallel (%i0) = (%arg0) to (%arg0) step (%arg0) {
    // expected-error@+1 {{the block inside reduce should be terminated with a 'loop.reduce.return' op}}
    loop.reduce(%arg1) {
      ^bb0(%lhs : f32, %rhs : f32):
        "loop.terminator"(): () -> ()
    } : f32
  } : f32
  return
}

// -----

func @reduceReturn_wrong_type(%arg0 : index, %arg1: f32) {
  %res = loop.parallel (%i0) = (%arg0) to (%arg0) step (%arg0) {
    loop.reduce(%arg1) {
      ^bb0(%lhs : f32, %rhs : f32):
        %c0 = constant 1 : index
        // expected-error@+1 {{needs to have type 'f32' (the type of the enclosing ReduceOp)}}
        loop.reduce.return %c0 : index
    } : f32
  } : f32
  return
}

// -----

func @reduceReturn_not_inside_reduce(%arg0 : f32) {
  "foo.region"() ({
    // expected-error@+1 {{expects parent op 'loop.reduce'}}
    loop.reduce.return %arg0 : f32
  }): () -> ()
  return
}
