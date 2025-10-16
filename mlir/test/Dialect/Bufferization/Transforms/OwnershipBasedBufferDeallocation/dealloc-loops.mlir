// RUN: mlir-opt %s -expand-realloc="emit-deallocs=false" -ownership-based-buffer-deallocation="private-function-dynamic-ownership=true" -canonicalize -buffer-deallocation-simplification | FileCheck %s

// A function that reallocates two buffer inside of a loop. The simplification
// pass should be able to figure out that the iter_args are always originating
// from different allocations. IR like this one appears in the sparse compiler.

// CHECK-LABEL: func private @loop_with_realloc(
func.func private @loop_with_realloc(%lb: index, %ub: index, %step: index, %c: i1, %s1: index, %s2: index) -> (memref<?xf32>, memref<?xf32>) {
  // CHECK-DAG: %[[false:.*]] = arith.constant false
  // CHECK-DAG: %[[true:.*]] = arith.constant true

  // CHECK: %[[m0:.*]] = memref.alloc
  %m0 = memref.alloc(%s1) : memref<?xf32>
  // CHECK: %[[m1:.*]] = memref.alloc
  %m1 = memref.alloc(%s1) : memref<?xf32>

  // CHECK: %[[r:.*]]:4 = scf.for {{.*}} iter_args(%[[arg0:.*]] = %[[m0]], %[[arg1:.*]] = %[[m1]], %[[o0:.*]] = %[[false]], %[[o1:.*]] = %[[false]])
  %r0, %r1 = scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %m0, %arg1 = %m1) -> (memref<?xf32>, memref<?xf32>) {
    //      CHECK: %[[m2:.*]]:2 = scf.if %{{.*}} -> (memref<?xf32>, i1) {
    // CHECK-NEXT:   memref.alloc
    // CHECK-NEXT:   memref.subview
    // CHECK-NEXT:   memref.copy
    // CHECK-NEXT:   scf.yield %{{.*}}, %[[true]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   memref.reinterpret_cast
    // CHECK-NEXT:   scf.yield %{{.*}}, %[[false]]
    // CHECK-NEXT: }
    %m2 = memref.realloc %arg0(%s2) : memref<?xf32> to memref<?xf32>
    //      CHECK: %[[m3:.*]]:2 = scf.if %{{.*}} -> (memref<?xf32>, i1) {
    // CHECK-NEXT:   memref.alloc
    // CHECK-NEXT:   memref.subview
    // CHECK-NEXT:   memref.copy
    // CHECK-NEXT:   scf.yield %{{.*}}, %[[true]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   memref.reinterpret_cast
    // CHECK-NEXT:   scf.yield %{{.*}}, %[[false]]
    // CHECK-NEXT: }
    %m3 = memref.realloc %arg1(%s2) : memref<?xf32> to memref<?xf32>

    // CHECK: %[[base0:.*]], %{{.*}}, %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[arg0]]
    // CHECK: %[[base1:.*]], %{{.*}}, %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[arg1]]
    // CHECK: %[[d0:.*]] = bufferization.dealloc (%[[base0]] : memref<f32>) if (%[[o0]]) retain (%[[m2]]#0 : memref<?xf32>)
    // CHECK: %[[d1:.*]] = bufferization.dealloc (%[[base1]] : memref<f32>) if (%[[o1]]) retain (%[[m3]]#0 : memref<?xf32>)
    // CHECK-DAG: %[[o2:.*]] = arith.ori %[[d0]], %[[m2]]#1
    // CHECK-DAG: %[[o3:.*]] = arith.ori %[[d1]], %[[m3]]#1
    // CHECK: scf.yield %[[m2]]#0, %[[m3]]#0, %[[o2]], %[[o3]]
    scf.yield %m2, %m3 : memref<?xf32>, memref<?xf32>
  }

  // CHECK: %[[d2:.*]] = bufferization.dealloc (%[[m0]] : memref<?xf32>) if (%[[true]]) retain (%[[r]]#0 : memref<?xf32>)
  // CHECK: %[[d3:.*]] = bufferization.dealloc (%[[m1]] : memref<?xf32>) if (%[[true]]) retain (%[[r]]#1 : memref<?xf32>)
  // CHECK-DAG: %[[or0:.*]] = arith.ori %[[d2]], %[[r]]#2
  // CHECK-DAG: %[[or1:.*]] = arith.ori %[[d3]], %[[r]]#3
  // CHECK: return %[[r]]#0, %[[r]]#1, %[[or0]], %[[or1]]
  return %r0, %r1 : memref<?xf32>, memref<?xf32>
}

// -----

// The yielded values of the loop are swapped. Therefore, the
// bufferization.dealloc before the func.return can now longer be split,
// because %r0 could originate from either %m0 and %m1 (same for %r1).

// CHECK-LABEL: func private @swapping_loop_with_realloc(
func.func private @swapping_loop_with_realloc(%lb: index, %ub: index, %step: index, %c: i1, %s1: index, %s2: index) -> (memref<?xf32>, memref<?xf32>) {
  // CHECK-DAG: %[[false:.*]] = arith.constant false
  // CHECK-DAG: %[[true:.*]] = arith.constant true

  // CHECK: %[[m0:.*]] = memref.alloc
  %m0 = memref.alloc(%s1) : memref<?xf32>
  // CHECK: %[[m1:.*]] = memref.alloc
  %m1 = memref.alloc(%s1) : memref<?xf32>

  // CHECK: %[[r:.*]]:4 = scf.for {{.*}} iter_args(%[[arg0:.*]] = %[[m0]], %[[arg1:.*]] = %[[m1]], %[[o0:.*]] = %[[false]], %[[o1:.*]] = %[[false]])
  %r0, %r1 = scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %m0, %arg1 = %m1) -> (memref<?xf32>, memref<?xf32>) {
    %m2 = memref.realloc %arg0(%s2) : memref<?xf32> to memref<?xf32>
    %m3 = memref.realloc %arg1(%s2) : memref<?xf32> to memref<?xf32>
    scf.yield %m3, %m2 : memref<?xf32>, memref<?xf32>
  }

  // CHECK: %[[base0:.*]], %{{.*}}, %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[r]]#0
  // CHECK: %[[base1:.*]], %{{.*}}, %{{.*}}, %{{.*}} = memref.extract_strided_metadata %[[r]]#1
  // CHECK: %[[d:.*]]:2 = bufferization.dealloc (%[[m0]], %[[m1]], %[[base0]], %[[base1]] : {{.*}}) if (%[[true]], %[[true]], %[[r]]#2, %[[r]]#3) retain (%[[r]]#0, %[[r]]#1 : {{.*}})
  // CHECK: return %[[r]]#0, %[[r]]#1, %[[d]]#0, %[[d]]#1
  return %r0, %r1 : memref<?xf32>, memref<?xf32>
}
