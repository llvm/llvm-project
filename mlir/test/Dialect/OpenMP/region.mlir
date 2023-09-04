// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @basic_omp_region
// CHECK-NEXT: omp.parallel {
// CHECK-NEXT:   omp.region {
// CHECK-NEXT:     "test.foo"() : () -> ()
// CHECK-NEXT:     omp.terminator
// CHECK-NEXT:   }
// CHECK-NEXT:   omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @basic_omp_region() {
  omp.parallel {
    omp.region {
      "test.foo"() : () -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @omp_region_with_branch
// CHECK-NEXT: omp.task {
// CHECK-NEXT:   omp.region {
// CHECK-NEXT:     %[[c:.*]] = "test.foo"() : () -> i1
// CHECK-NEXT:     cf.cond_br %[[c]], ^[[bb1:.*]](%[[c]] : i1), ^[[bb2:.*]](%[[c]] : i1)
// CHECK-NEXT:   ^[[bb1]](%[[arg:.*]]: i1):
// CHECK-NEXT:     "test.bar"() : () -> ()
// CHECK-NEXT:     omp.terminator
// CHECK-NEXT:   ^[[bb2]](%[[arg2:.*]]: i1):
// CHECK-NEXT:     "test.baz"() : () -> ()
// CHECK-NEXT:     omp.terminator
// CHECK-NEXT:   }
// CHECK-NEXT:   omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @omp_region_with_branch(%a: i32) {
  omp.task {
    omp.region {
      %c = "test.foo"() : () -> i1
      cf.cond_br %c, ^bb1(%c: i1), ^bb2(%c: i1)
    ^bb1(%arg: i1):
      "test.bar"() : () -> ()
      omp.terminator
    ^bb2(%arg2: i1):
      "test.baz"() : () -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}
