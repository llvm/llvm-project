// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s --check-prefixes=OPT,RAW
// RUN: mlir-opt %s -canonicalize -int-range-optimizations | FileCheck %s --check-prefixes=OPT,PRE
// RUN: mlir-opt %s -int-range-optimizations -canonicalize | FileCheck %s --check-prefix=CLEAN
// RUN: mlir-opt %s -arith-int-range-narrowing='int-bitwidths-supported=8,16,32' | FileCheck %s --check-prefix=NARROW

// A merge with poison preserves the bounds needed to remove the remainder.
// OPT-LABEL: func.func @remove_remainder(
// OPT: %[[BOUND:.*]] = arith.minsi
// RAW: %[[SELECT:.*]] = arith.select {{.*}}, %[[BOUND]], {{.*}} : i32
// RAW-NEXT: return %[[SELECT]] : i32
// PRE-NEXT: return %[[BOUND]] : i32
func.func @remove_remainder(%c: i1, %x: i32) -> i32 {
  %p = ub.poison : i32
  %zero = arith.constant 0 : i32
  %seven = arith.constant 7 : i32
  %eight = arith.constant 8 : i32
  %lo = arith.maxsi %x, %zero : i32
  %bounded = arith.minsi %lo, %seven : i32
  %v = arith.select %c, %bounded, %p : i32
  %r = arith.remui %v, %eight : i32
  return %r : i32
}

// Range inference proves the condition. Canonicalization removes the branch.
// OPT-LABEL: func.func @remove_branch(
// OPT: %[[TRUE:.*]] = arith.constant true
// RAW: %[[RESULT:.*]] = scf.if %[[TRUE]]
// PRE: %[[RESULT:.*]] = arith.select %[[TRUE]]
// OPT: return %[[RESULT]] : i32
// CLEAN-LABEL: func.func @remove_branch(
// CLEAN-SAME: %{{.*}}: i1, %[[X:.*]]: i32
// CLEAN-NEXT: return %[[X]] : i32
func.func @remove_branch(%c: i1, %x: i32) -> i32 {
  %p = ub.poison : i32
  %zero = arith.constant 0 : i32
  %lo = arith.maxsi %x, %zero : i32
  %v = arith.select %c, %lo, %p : i32
  %test = arith.cmpi sge, %v, %zero : i32
  %r = scf.if %test -> i32 {
    scf.yield %x : i32
  } else {
    scf.yield %zero : i32
  }
  return %r : i32
}

// A single narrowing pass uses the defined values from the poison merge.
// NARROW-LABEL: func.func @narrow_add(
// NARROW: %[[BOUND:.*]] = arith.minsi
// NARROW: %[[LHS:.*]] = arith.trunci %[[BOUND]] : i32 to i8
// NARROW: %[[RHS:.*]] = arith.trunci %[[BOUND]] : i32 to i8
// NARROW: %[[SUM:.*]] = arith.addi %[[LHS]], %[[RHS]] : i8
// NARROW-NEXT: %[[RESULT:.*]] = arith.extui %[[SUM]] : i8 to i32
// NARROW-NEXT: return %[[RESULT]] : i32
func.func @narrow_add(%c: i1, %x: i32) -> i32 {
  %p = ub.poison : i32
  %zero = arith.constant 0 : i32
  %seven = arith.constant 7 : i32
  %lo = arith.maxsi %x, %zero : i32
  %bounded = arith.minsi %lo, %seven : i32
  %v = arith.select %c, %bounded, %p : i32
  %r = arith.addi %v, %bounded : i32
  return %r : i32
}

// An integer extension preserves the empty range before the merge.
// OPT-LABEL: func.func @through_cast(
// OPT-NEXT: %[[TRUE:.*]] = arith.constant true
// OPT-NEXT: return %[[TRUE]] : i1
func.func @through_cast(%c: i1, %x: i32) -> i1 {
  %p = ub.poison : i16
  %wide = arith.extsi %p : i16 to i32
  %zero = arith.constant 0 : i32
  %lo = arith.maxsi %x, %zero : i32
  %v = arith.select %c, %lo, %wide : i32
  %test = arith.cmpi sge, %v, %zero : i32
  return %test : i1
}

// The comparison can refine the poison lane to true. The defined lane is true.
// This also folds when canonicalization runs before range inference.
// OPT-LABEL: func.func @scalar_to_vector(
// OPT-NEXT: %[[TRUE:.*]] = arith.constant dense<true> : vector<2xi1>
// OPT-NEXT: return %[[TRUE]] : vector<2xi1>
func.func @scalar_to_vector(%x: i32) -> vector<2xi1> {
  %p = ub.poison : vector<2xi32>
  %zero = arith.constant 0 : i32
  %vzero = arith.constant dense<0> : vector<2xi32>
  %lo = arith.maxsi %x, %zero : i32
  %v = vector.insert %lo, %p[0] : i32 into vector<2xi32>
  %test = arith.cmpi sge, %v, %vzero : vector<2xi32>
  return %test : vector<2xi1>
}

// The initial value contributes no defined integers to the loop-carried range.
// The inferred comparison also permits removal of the unused loop.
// OPT-LABEL: func.func @loop_carried_poison(
// OPT-NEXT: %[[TRUE:.*]] = arith.constant true
// OPT-NEXT: return %[[TRUE]] : i1
func.func @loop_carried_poison(%n: index, %c: i1, %x: i32) -> i1 {
  %z = arith.constant 0 : index
  %step = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %lo = arith.maxsi %x, %zero : i32
  %p = ub.poison : i32
  %v = scf.for %i = %z to %n step %step iter_args(%a = %p) -> i32 {
    %next = arith.select %c, %lo, %a : i32
    scf.yield %next : i32
  }
  %r = arith.cmpi sge, %v, %zero : i32
  return %r : i1
}
