/// Run once without applying the pattern and check the source of truth.
// RUN: mlir-opt %s --test-transform-dialect-erase-schedule -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

/// Run once with the pattern and compare.
// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule -test-lower-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @fcst_maskedload(%A: memref<?xi4>, %passthru: vector<6xi4>) -> vector<6xi4> {
  %c0 = arith.constant 0: index
  %mask = vector.constant_mask [3] : vector<6xi1>
  %1 = vector.maskedload %A[%c0], %mask, %passthru :
    memref<?xi4>, vector<6xi1>, vector<6xi4> into vector<6xi4>
  return %1 : vector<6xi4>
}

func.func @entry() {
  // Set up memory.
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c6 = arith.constant 6: index
  %A = memref.alloc(%c6) : memref<?xi4>
  scf.for %i = %c0 to %c6 step %c1 {
    %i4 = arith.index_cast %i : index to i4
    memref.store %i4, %A[%i] : memref<?xi4>
  }
  %passthru = arith.constant dense<[7, 8, 9, 10, 11, 12]> : vector<6xi4>
  %load = call @fcst_maskedload(%A, %passthru) : (memref<?xi4>, vector<6xi4>) -> (vector<6xi4>)
  vector.print %load : vector<6xi4>
  // CHECK: ( 0, 1, 2, -6, -5, -4 )
  memref.dealloc %A : memref<?xi4>

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
        : (!transform.any_op) -> !transform.any_op

    transform.apply_conversion_patterns to %f {
      transform.apply_conversion_patterns.vector.emulate_narrow_types
    } with type_converter {
      transform.apply_conversion_patterns.vector.emulate_narrow_type_converter
      {arith_compute_bitwidth = 1,
       load_store_emulate_bitwidth = 8}
    } {
      partial_conversion
    }: !transform.any_op
    transform.yield
  }
}
