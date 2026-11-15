// RUN: mlir-opt --int-range-optimizations %s | FileCheck %s --check-prefix=RANGE
// RUN: mlir-opt --scf-for-loop-canonicalization %s | FileCheck %s --check-prefix=AFFINE

func.func private @observe(index)
func.func private @observe_bit(i1)

// RANGE-LABEL: func.func @count_high(
// RANGE: scf.for unsigned
// RANGE: arith.cmpi slt
func.func @count_high(%upper: i8) -> i32 {
  %zero = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %init = arith.constant 0 : i32
  %count = scf.for unsigned %iv = %zero to %upper step %one
      iter_args(%acc = %init) -> (i32) : i8 {
    %is_high = arith.cmpi slt, %iv, %zero : i8
    %contribution = arith.extui %is_high : i1 to i32
    %next = arith.addi %acc, %contribution : i32
    scf.yield %next : i32
  }
  return %count : i32
}

// RANGE-LABEL: func.func @sum_high_bound(
// RANGE: scf.for unsigned {{.*}} iter_args(%[[ACC:[a-zA-Z0-9_]+]] =
// RANGE: arith.addi {{.*}}%[[ACC]]
func.func @sum_high_bound() -> i32 {
  %zero = arith.constant 0 : i8
  %upper = arith.constant 200 : i8
  %one = arith.constant 1 : i8
  %init = arith.constant 0 : i32
  %sum = scf.for unsigned %iv = %zero to %upper step %one
      iter_args(%acc = %init) -> (i32) : i8 {
    %value = arith.extui %iv : i8 to i32
    %next = arith.addi %acc, %value : i32
    scf.yield %next : i32
  }
  return %sum : i32
}

// RANGE-LABEL: func.func @parallel_dynamic(
// RANGE: scf.parallel unsigned
// RANGE: arith.cmpi slt
func.func @parallel_dynamic(%lower: index, %upper: index) {
  %one = arith.constant 1 : index
  %signed_max = arith.constant 9223372036854775807 : index
  scf.parallel unsigned (%iv) = (%lower) to (%upper) step (%one) {
    %is_less = arith.cmpi slt, %iv, %signed_max : index
    func.call @observe_bit(%is_less) : (i1) -> ()
    scf.reduce
  }
  return
}

// AFFINE-LABEL: func.func @affine_unsigned_max(
// AFFINE: scf.for unsigned
// AFFINE: affine.max
func.func @affine_unsigned_max(%upper: index) {
  %lower = arith.constant 9223372036854775806 : index
  %one = arith.constant 1 : index
  scf.for unsigned %iv = %lower to %upper step %one {
    %max = affine.max affine_map<(d0) -> (d0, 0)>(%iv)
    func.call @observe(%max) : (index) -> ()
  }
  return
}
