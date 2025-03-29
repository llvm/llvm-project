/// Run once without applying the pattern and check the source of truth.
// RUN: mlir-opt %s --test-transform-dialect-erase-schedule -test-lower-to-llvm | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

/// Run once with the pattern and compare.
// RUN: mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule -test-lower-to-llvm | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @print_as_i1_16xi5(%v : vector<16xi5>) {
  %bitsi16 = vector.bitcast %v : vector<16xi5> to vector<80xi1>
  vector.print %bitsi16 : vector<80xi1>
  return
}

func.func @print_as_i1_10xi8(%v : vector<10xi8>) {
  %bitsi16 = vector.bitcast %v : vector<10xi8> to vector<80xi1>
  vector.print %bitsi16 : vector<80xi1>
  return
}

func.func @f(%v: vector<16xi16>) {
  %trunc = arith.trunci %v : vector<16xi16> to vector<16xi5>
  func.call @print_as_i1_16xi5(%trunc) : (vector<16xi5>) -> ()
  //      CHECK: (
  // CHECK-SAME: 1, 1, 1, 1, 1,
  // CHECK-SAME: 0, 1, 1, 1, 1,
  // CHECK-SAME: 1, 0, 1, 1, 1,
  // CHECK-SAME: 0, 0, 1, 1, 1,
  // CHECK-SAME: 1, 1, 0, 1, 1,
  // CHECK-SAME: 0, 1, 0, 1, 1,
  // CHECK-SAME: 1, 0, 0, 1, 1,
  // CHECK-SAME: 0, 0, 0, 1, 1,
  // CHECK-SAME: 1, 1, 1, 0, 1,
  // CHECK-SAME: 0, 1, 1, 0, 1,
  // CHECK-SAME: 1, 0, 1, 0, 1,
  // CHECK-SAME: 0, 0, 1, 0, 1,
  // CHECK-SAME: 1, 1, 0, 0, 1,
  // CHECK-SAME: 0, 1, 0, 0, 1,
  // CHECK-SAME: 1, 0, 0, 0, 1,
  // CHECK-SAME: 0, 0, 0, 0, 1 )

  %bitcast = vector.bitcast %trunc : vector<16xi5> to vector<10xi8>
  func.call @print_as_i1_10xi8(%bitcast) : (vector<10xi8>) -> ()
  //      CHECK: (
  // CHECK-SAME: 1, 1, 1, 1, 1, 0, 1, 1,
  // CHECK-SAME: 1, 1, 1, 0, 1, 1, 1, 0,
  // CHECK-SAME: 0, 1, 1, 1, 1, 1, 0, 1,
  // CHECK-SAME: 1, 0, 1, 0, 1, 1, 1, 0,
  // CHECK-SAME: 0, 1, 1, 0, 0, 0, 1, 1,
  // CHECK-SAME: 1, 1, 1, 0, 1, 0, 1, 1,
  // CHECK-SAME: 0, 1, 1, 0, 1, 0, 1, 0,
  // CHECK-SAME: 0, 1, 0, 1, 1, 1, 0, 0,
  // CHECK-SAME: 1, 0, 1, 0, 0, 1, 1, 0,
  // CHECK-SAME: 0, 0, 1, 0, 0, 0, 0, 1 )

  return
}

func.func @print_as_i1_8xi3(%v : vector<8xi3>) {
  %bitsi12 = vector.bitcast %v : vector<8xi3> to vector<24xi1>
  vector.print %bitsi12 : vector<24xi1>
  return
}

func.func @print_as_i1_3xi8(%v : vector<3xi8>) {
  %bitsi12 = vector.bitcast %v : vector<3xi8> to vector<24xi1>
  vector.print %bitsi12 : vector<24xi1>
  return
}

func.func @f2(%v: vector<8xi32>) {
  %trunc = arith.trunci %v : vector<8xi32> to vector<8xi3>
  func.call @print_as_i1_8xi3(%trunc) : (vector<8xi3>) -> ()
  //      CHECK: (
  // CHECK-SAME: 1, 1, 1,
  // CHECK-SAME: 0, 1, 1,
  // CHECK-SAME: 1, 0, 1,
  // CHECK-SAME: 0, 0, 1,
  // CHECK-SAME: 1, 1, 0,
  // CHECK-SAME: 0, 1, 0,
  // CHECK-SAME: 1, 0, 0,
  // CHECK-SAME: 0, 0, 0 )

  %bitcast = vector.bitcast %trunc : vector<8xi3> to vector<3xi8>
  func.call @print_as_i1_3xi8(%bitcast) : (vector<3xi8>) -> ()
  //      CHECK: (
  // CHECK-SAME: 1, 1, 1, 0, 1, 1, 1, 0,
  // CHECK-SAME: 1, 0, 0, 1, 1, 1, 0, 0,
  // CHECK-SAME: 1, 0, 1, 0, 0, 0, 0, 0 )

  return
}

func.func @print_as_i1_2xi24(%v : vector<2xi24>) {
  %bitsi48 = vector.bitcast %v : vector<2xi24> to vector<48xi1>
  vector.print %bitsi48 : vector<48xi1>
  return
}

func.func @print_as_i1_3xi16(%v : vector<3xi16>) {
  %bitsi48 = vector.bitcast %v : vector<3xi16> to vector<48xi1>
  vector.print %bitsi48 : vector<48xi1>
  return
}

func.func @f3(%v: vector<2xi48>) {
  %trunc = arith.trunci %v : vector<2xi48> to vector<2xi24>
  func.call @print_as_i1_2xi24(%trunc) : (vector<2xi24>) -> ()
  //      CHECK: (
  // CHECK-SAME: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  // CHECK-SAME: 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0 )

  %bitcast = vector.bitcast %trunc : vector<2xi24> to vector<3xi16>
  func.call @print_as_i1_3xi16(%bitcast) : (vector<3xi16>) -> ()
  //      CHECK: (
  // CHECK-SAME: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  // CHECK-SAME: 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
  // CHECK-SAME: 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0 )

  return
}

func.func @print_as_i1_8xi5(%v : vector<8xi5>) {
  %bitsi40 = vector.bitcast %v : vector<8xi5> to vector<40xi1>
  vector.print %bitsi40 : vector<40xi1>
  return
}

func.func @print_as_i1_8xi16(%v : vector<8xi16>) {
  %bitsi128 = vector.bitcast %v : vector<8xi16> to vector<128xi1>
  vector.print %bitsi128 : vector<128xi1>
  return
}

func.func @fext(%a: vector<5xi8>) {
  %0 = vector.bitcast %a : vector<5xi8> to vector<8xi5>
  func.call @print_as_i1_8xi5(%0) : (vector<8xi5>) -> ()
  //      CHECK: (
  // CHECK-SAME: 1, 1, 1, 1, 0,
  // CHECK-SAME: 1, 1, 1, 0, 1,
  // CHECK-SAME: 1, 1, 0, 1, 1,
  // CHECK-SAME: 1, 1, 0, 1, 1,
  // CHECK-SAME: 0, 1, 1, 1, 0,
  // CHECK-SAME: 0, 1, 1, 0, 1,
  // CHECK-SAME: 1, 1, 1, 1, 0,
  // CHECK-SAME: 1, 0, 1, 1, 1 )

  %1 = arith.extui %0 : vector<8xi5> to vector<8xi16>
  func.call @print_as_i1_8xi16(%1) : (vector<8xi16>) -> ()
  //      CHECK: (
  // CHECK-SAME: 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME: 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME: 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME: 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME: 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME: 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME: 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  // CHECK-SAME: 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )

  return
}

func.func @fcst_maskedload(%A: memref<?xi4>, %passthru: vector<6xi4>) -> vector<6xi4> {
  %c0 = arith.constant 0: index
  %mask = vector.constant_mask [3] : vector<6xi1>
  %1 = vector.maskedload %A[%c0], %mask, %passthru :
    memref<?xi4>, vector<6xi1>, vector<6xi4> into vector<6xi4>
  return %1 : vector<6xi4>
}

func.func @entry() {
  %v = arith.constant dense<[
    0xffff, 0xfffe, 0xfffd, 0xfffc, 0xfffb, 0xfffa, 0xfff9, 0xfff8,
    0xfff7, 0xfff6, 0xfff5, 0xfff4, 0xfff3, 0xfff2, 0xfff1, 0xfff0
  ]> : vector<16xi16>
  func.call @f(%v) : (vector<16xi16>) -> ()

  %v2 = arith.constant dense<[
    0xffff, 0xfffe, 0xfffd, 0xfffc, 0xfffb, 0xfffa, 0xfff9, 0xfff8
  ]> : vector<8xi32>
  func.call @f2(%v2) : (vector<8xi32>) -> ()

  %v3 = arith.constant dense<[
    0xf345aeffffff, 0xffff015f345a
  ]> : vector<2xi48>
  func.call @f3(%v3) : (vector<2xi48>) -> ()

  %v4 = arith.constant dense<[
    0xef, 0xee, 0xed, 0xec, 0xeb
  ]> : vector<5xi8>
  func.call @fext(%v4) : (vector<5xi8>) -> ()

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

    transform.apply_patterns to %f {
      transform.apply_patterns.vector.rewrite_narrow_types
    } : !transform.any_op
    transform.yield
  }
}
