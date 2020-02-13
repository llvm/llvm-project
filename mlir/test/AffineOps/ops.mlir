// RUN: mlir-opt -split-input-file %s | FileCheck %s
// RUN: mlir-opt %s -mlir-print-op-generic | FileCheck -check-prefix=GENERIC %s

// Check that the attributes for the affine operations are round-tripped.
// Check that `affine.terminator` is visible in the generic form.
// CHECK-LABEL: @empty
func @empty() {
  // CHECK: affine.for
  // CHECK-NEXT: } {some_attr = true}
  //
  // GENERIC:      "affine.for"()
  // GENERIC-NEXT: ^bb0(%{{.*}}: index):
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: })
  affine.for %i = 0 to 10 {
  } {some_attr = true}

  // CHECK: affine.if
  // CHECK-NEXT: } {some_attr = true}
  //
  // GENERIC:      "affine.if"()
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: },  {
  // GENERIC-NEXT: })
  affine.if affine_set<() : ()> () {
  } {some_attr = true}

  // CHECK: } else {
  // CHECK: } {some_attr = true}
  //
  // GENERIC:      "affine.if"()
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: },  {
  // GENERIC-NEXT:   "foo"() : () -> ()
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: })
  affine.if affine_set<() : ()> () {
  } else {
    "foo"() : () -> ()
  } {some_attr = true}

  return
}

// Check that an explicit affine terminator is not printed in custom format.
// Check that no extra terminator is introduced.
// CHECK-LABEL: @affine_terminator
func @affine_terminator() {
  // CHECK: affine.for
  // CHECK-NEXT: }
  //
  // GENERIC:      "affine.for"() ( {
  // GENERIC-NEXT: ^bb0(%{{.*}}: index):	// no predecessors
  // GENERIC-NEXT:   "affine.terminator"() : () -> ()
  // GENERIC-NEXT: }) {lower_bound = #map0, step = 1 : index, upper_bound = #map1} : () -> ()
  affine.for %i = 0 to 10 {
    "affine.terminator"() : () -> ()
  }
  return
}

// -----

// CHECK-DAG: #[[MAP0:map[0-9]+]] = affine_map<(d0)[s0] -> (1000, d0 + 512, s0)>
// CHECK-DAG: #[[MAP1:map[0-9]+]] = affine_map<(d0, d1)[s0] -> (d0 - d1, s0 + 512)>
// CHECK-DAG: #[[MAP2:map[0-9]+]] = affine_map<()[s0, s1] -> (s0 - s1, 11)>
// CHECK-DAG: #[[MAP3:map[0-9]+]] = affine_map<() -> (77, 78, 79)>

// CHECK-LABEL: @affine_min
func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: affine.min #[[MAP0]](%arg0)[%arg1]
  %0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
  // CHECK: affine.min #[[MAP1]](%arg0, %arg1)[%arg2]
  %1 = affine.min affine_map<(d0, d1)[s0] -> (d0 - d1, s0 + 512)> (%arg0, %arg1)[%arg2]
  // CHECK: affine.min #[[MAP2]]()[%arg1, %arg2]
  %2 = affine.min affine_map<()[s0, s1] -> (s0 - s1, 11)> ()[%arg1, %arg2]
  // CHECK: affine.min #[[MAP3]]()
  %3 = affine.min affine_map<()[] -> (77, 78, 79)> ()[]
  return
}

// CHECK-LABEL: @affine_max
func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: affine.max #[[MAP0]](%arg0)[%arg1]
  %0 = affine.max affine_map<(d0)[s0] -> (1000, d0 + 512, s0)> (%arg0)[%arg1]
  // CHECK: affine.max #[[MAP1]](%arg0, %arg1)[%arg2]
  %1 = affine.max affine_map<(d0, d1)[s0] -> (d0 - d1, s0 + 512)> (%arg0, %arg1)[%arg2]
  // CHECK: affine.max #[[MAP2]]()[%arg1, %arg2]
  %2 = affine.max affine_map<()[s0, s1] -> (s0 - s1, 11)> ()[%arg1, %arg2]
  // CHECK: affine.max #[[MAP3]]()
  %3 = affine.max affine_map<()[] -> (77, 78, 79)> ()[]
  return
}

// -----

func @valid_symbols(%arg0: index, %arg1: index, %arg2: index) {
  %c0 = constant 1 : index
  %c1 = constant 0 : index
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
  affine.for %arg3 = 0 to %arg2 step 768 {
    %13 = dim %0, 1 : memref<?x?xf32>
    affine.for %arg4 = 0 to %13 step 264 {
      %18 = dim %0, 0 : memref<?x?xf32>
      %20 = std.subview %0[%c0, %c0][%18,%arg4][%c1,%c1] : memref<?x?xf32>
                          to memref<?x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)>>
      %24 = dim %20, 0 : memref<?x?xf32, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + d1 * s2 + s0)>>
      affine.for %arg5 = 0 to %24 step 768 {
        "foo"() : () -> ()
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: @parallel
// CHECK-SAME: (%[[N:.*]]: index)
func @parallel(%N : index) {
  // CHECK: affine.parallel (%[[I0:.*]], %[[J0:.*]]) = (0, 0) to (symbol(%[[N]]), 100) step (10, 10)
  affine.parallel (%i0, %j0) = (0, 0) to (symbol(%N), 100) step (10, 10) {
    // CHECK-NEXT: affine.parallel (%{{.*}}, %{{.*}}) = (%[[I0]], %[[J0]]) to (%[[I0]] + 10, %[[J0]] + 10)
    affine.parallel (%i1, %j1) = (%i0, %j0) to (%i0 + 10, %j0 + 10) {
    }
  }
  return
}
