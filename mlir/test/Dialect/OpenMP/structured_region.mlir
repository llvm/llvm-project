// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @basic_omp_region
// CHECK-NEXT: omp.parallel {
// CHECK-NEXT:   {{.+}} = omp.structured_region(i32) {
// CHECK-NEXT:     {{.+}} = "test.foo"() : () -> i32
// CHECK-NEXT:     omp.yield({{.+}})
// CHECK-NEXT:   }
// CHECK-NEXT:   omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @basic_omp_region() {
  omp.parallel {
    %x = omp.structured_region (i32) {
      %y = "test.foo"() : () -> i32
      omp.yield(%y : i32)
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL: @omp_region_with_branch
// CHECK-NEXT: omp.task {
// CHECK-NEXT:   {{.+}} = omp.structured_region(i32) {
// CHECK-NEXT:     %[[c:.*]] = "test.foo"() : () -> i1
// CHECK-NEXT:     cf.cond_br %[[c]], ^[[bb1:.*]](%[[c]] : i1), ^[[bb2:.*]](%[[c]] : i1)
// CHECK-NEXT:   ^[[bb1]](%[[arg:.*]]: i1):
// CHECK-NEXT:     {{.+}} = "test.bar"() : () -> i32
// CHECK-NEXT:     omp.yield({{.+}})
// CHECK-NEXT:   ^[[bb2]](%[[arg2:.*]]: i1):
// CHECK-NEXT:     {{.+}} = "test.baz"() : () -> i32
// CHECK-NEXT:     omp.yield({{.+}})
// CHECK-NEXT:   }
// CHECK-NEXT:   omp.terminator
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @omp_region_with_branch(%a: i32) {
  omp.task {
    %t = omp.structured_region (i32) {
      %c = "test.foo"() : () -> i1
      cf.cond_br %c, ^bb1(%c: i1), ^bb2(%c: i1)
    ^bb1(%arg: i1):
      %x = "test.bar"() : () -> i32
      omp.yield(%x : i32)
    ^bb2(%arg2: i1):
      %y = "test.baz"() : () -> i32
      omp.yield(%y: i32)
    }
    omp.terminator
  }
  return
}
