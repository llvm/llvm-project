// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// -----

%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c4 = arith.constant 4 : index
scf.parallel (%iv) = (%c0) to (%c4) step (%c1) {
  scf.reduce
// expected-error@+1 {{expected one of ::mlir::gpu::Processor enum names}}
} {acc.par_dims = #acc<par_dims[thread_x, bad_dim]>}

// -----

%c0_2 = arith.constant 0 : index
%c1_2 = arith.constant 1 : index
%c4_2 = arith.constant 4 : index
scf.parallel (%iv) = (%c0_2) to (%c4_2) step (%c1_2) {
  scf.reduce
// expected-error@+1 {{expected one of ::mlir::gpu::Processor enum names}}
} {acc.par_dims = #acc<par_dims[gang]>}

// -----

%c32 = arith.constant 32 : index
// expected-error@+1 {{'acc.compute_region' op launch arguments must be results of acc.par_width operations}}
acc.compute_region launch(%arg0 = %c32) {
  acc.yield
} {origin = "acc.parallel"}

// -----

// Use generic form to introduce an extra block argument.
%c64 = arith.constant 64 : index
%w = acc.par_width %c64 {par_dim = #acc.par_dim<thread_x>}
// expected-error@+1 {{'acc.compute_region' op expected 1 block arguments (launch + input), got 2}}
"acc.compute_region"(%w) <{operandSegmentSizes = array<i32: 1, 0, 0>}> ({
^bb0(%arg0: index, %extra: index):
  "acc.yield"() : () -> ()
}) {origin = "acc.parallel"} : (index) -> ()

// -----

func.func @reduction_accumulate_invalid_operator() {
  %partial = arith.constant 1.0 : f32
  %private = memref.alloca() : memref<f32>
  acc.reduction_accumulate %partial to %private <addi>
      : f32 -> memref<f32> {par_dims = #acc<par_dims[thread_x]>}
  // expected-error@-2 {{expected ::mlir::acc::ReductionOperator to be one of}}
  // expected-error@-3 {{failed to parse OpenACC_ReductionOperatorAttr}}
  return
}

// -----

func.func @reduction_accumulate_type_mismatch() {
  %wrong_ty = arith.constant 3.0 : f32
  %private_i32 = memref.alloca() : memref<i32>
  // expected-error@+1 {{pointer-like element type must match value type}}
  acc.reduction_accumulate %wrong_ty to %private_i32 <add>
      : f32 -> memref<i32> {par_dims = #acc<par_dims[thread_x]>}
  return
}

// -----

func.func @reduction_accumulate_empty_par_dims() {
  %partial3 = arith.constant 4 : i32
  %private4 = memref.alloca() : memref<i32>
  // expected-error@+1 {{par_dims must specify at least one parallel dimension}}
  acc.reduction_accumulate %partial3 to %private4 <add>
      : i32 -> memref<i32> {par_dims = #acc<par_dims[]>}
  return
}

// -----

func.func @predicate_region_empty() {
  acc.compute_region {
    // expected-error@+1 {{region needs to have at least one block}}
    acc.predicate_region {
    }
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

func.func @predicate_region_with_args() {
  acc.compute_region {
    // expected-error@+1 {{region cannot have any arguments}}
    acc.predicate_region {
    ^bb0(%arg0: index):
      %c0 = arith.constant 0 : index
    }
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// -----

func.func @predicate_region_outside_compute_region() {
  // expected-error@+1 {{must be nested within an acc.compute_region operation}}
  acc.predicate_region {
    %c0 = arith.constant 0 : i32
  }
  return
}
