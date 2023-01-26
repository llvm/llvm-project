// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file -verify-diagnostics | FileCheck %s

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1:2 = transform.structured.split %0 after 42 { dimension = 0 } : !pdl.operation
}

func.func private @elem(%arg0: f32, %arg1: index, %arg2: index) -> f32

// CHECK: #[[$ADD_42_MAP:.+]] = affine_map<(d0) -> (d0 + 42)>

// CHECK-LABEL: @one_d_static
// CHECK-SAME:  %[[IN:.+]]: tensor<100xf32>, %[[OUT:.+]]: tensor<100xf32>
func.func @one_d_static(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  // CHECK: %[[IN_SLICE_LOW:.+]] = tensor.extract_slice %[[IN]][0] [42] [1] : tensor<100xf32> to tensor<42xf32>
  // CHECK: %[[OUT_SLICE_LOW:.+]] = tensor.extract_slice %[[OUT]][0] [42] [1] : tensor<100xf32> to tensor<42xf32>
  // CHECK: %[[RES_SLICE_LOW:.+]] = linalg.generic
  // CHECK:   ins(%[[IN_SLICE_LOW]]
  // CHECK:   outs(%[[OUT_SLICE_LOW]]
  // CHECK:   linalg.index 0
  // CHECK:   func.call @elem
  // CHECK: %[[RES_PARTIAL:.+]] = tensor.insert_slice %[[RES_SLICE_LOW]] into %[[OUT]][0] [42] [1]
  //
  // CHECK: %[[IN_SLICE_HIGH:.+]] = tensor.extract_slice %[[IN]][42] [58] [1] : tensor<100xf32> to tensor<58xf32>
  // CHECK: %[[OUT_SLICE_HIGH:.+]] = tensor.extract_slice %[[RES_PARTIAL]][42] [58] [1] : tensor<100xf32> to tensor<58xf32>
  // CHECK: %[[RES_SLICE_HIGH:.+]] = linalg.generic
  // CHECK:   ins(%[[IN_SLICE_HIGH]]
  // CHECK:   outs(%[[OUT_SLICE_HIGH]]
  // CHECK:   %[[IDX:.+]] = linalg.index 0
  // CHECK:   affine.apply #[[$ADD_42_MAP]](%[[IDX]])
  // CHECK:   func.call @elem
  // CHECK: %[[RES:.+]] = tensor.insert_slice %[[RES_SLICE_HIGH]] into %[[RES_PARTIAL]][42] [58] [1]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  }
  ins(%arg0: tensor<100xf32>) outs(%arg1: tensor<100xf32>) {
  ^bb0(%0: f32, %1: f32):
    %i = linalg.index 0 : index
    %call_res = func.call @elem(%0, %i, %i) : (f32, index, index) -> f32
    linalg.yield %call_res : f32
  } -> tensor<100xf32>

  // CHECK: return %[[RES]]
  return %0 : tensor<100xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1:2 = transform.structured.split %0 after 42 { dimension = 0 } : !pdl.operation
}

func.func private @elem(%arg0: f32, %arg1: index, %arg2: index) -> f32

// CHECK-LABEL: @one_d_static_overflow
// CHECK-SAME:  %[[IN:.+]]: tensor<10xf32>, %[[OUT:.+]]: tensor<10xf32>
func.func @one_d_static_overflow(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  // Folding is sufficiently powerful to detect the static overflow and avoid
  // the splitting altogether.
  // CHECK: %[[RES_SLICE_LOW:.+]] = linalg.generic
  // CHECK:   ins(%[[IN]]
  // CHECK:   outs(%[[OUT]]
  // CHECK:   linalg.index 0
  // CHECK:   func.call @elem
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  }
  ins(%arg0: tensor<10xf32>) outs(%arg1: tensor<10xf32>) {
  ^bb0(%0: f32, %1: f32):
    %i = linalg.index 0 : index
    %call_res = func.call @elem(%0, %i, %i) : (f32, index, index) -> f32
    linalg.yield %call_res : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = transform.structured.match ops{["func.call"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.split %0 after %1 { dimension = 0 } : !pdl.operation, !pdl.operation
}

func.func private @get_size() -> index

// CHECK: #[[$MAP_MIN_100:.+]] = affine_map<()[s0] -> (s0, 100)>
// CHECK: #[[$MAP_S_MINUS_100:.+]] = affine_map<()[s0] -> (-s0 + 100)>

// CHECK-LABEL: @dynamic
func.func @dynamic(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  // CHECK: %[[SPLIT:.+]] = call @get_size
  // CHECK: %[[SPLIT_LOW:.+]] = affine.min #[[$MAP_MIN_100]]()[%[[SPLIT]]
  // CHECK: %[[SPLIT_HIGH_1:.+]] = affine.apply #[[$MAP_S_MINUS_100]]()[%[[SPLIT_LOW]]]
  // CHECK: %[[IN_SLICE_LOW:.+]] = tensor.extract_slice %[[IN:.+]][0] [%[[SPLIT_LOW]]] [1] : tensor<100xf32> to tensor<?xf32>
  // CHECK: %[[OUT_SLICE_LOW:.+]] = tensor.extract_slice %[[OUT:.+]][0] [%[[SPLIT_LOW]]] [1] : tensor<100xf32> to tensor<?xf32>
  // CHECK: %[[RES_SLICE_LOW:.+]] = linalg.generic
  // CHECK:   ins(%[[IN_SLICE_LOW]]
  // CHECK:   outs(%[[OUT_SLICE_LOW]]
  // CHECK: %[[PARTIAL:.+]] = tensor.insert_slice %[[RES_SLICE_LOW]] into %[[OUT]][0] [%[[SPLIT_LOW]]] [1]
  //
  // CHECK: %[[SPLIT_HIGH_2:.+]] = affine.apply #[[$MAP_S_MINUS_100]]()[%[[SPLIT_LOW]]]
  // CHECK: %[[SPLIT_HIGH_3:.+]] = affine.apply #[[$MAP_S_MINUS_100]]()[%[[SPLIT_LOW]]]
  // CHECK: %[[IN_SLICE_HIGH:.+]] = tensor.extract_slice %[[IN:.+]][%[[SPLIT_LOW]]] [%[[SPLIT_HIGH_2]]] [1] : tensor<100xf32> to tensor<?xf32>
  // CHECK: %[[OUT_SLICE_HIGH:.+]] = tensor.extract_slice %[[PARTIAL:.+]][%[[SPLIT_LOW]]] [%[[SPLIT_HIGH_3]]] [1] : tensor<100xf32> to tensor<?xf32>
  // CHECK: %[[RES_SLICE_HIGH:.+]] = linalg.generic
  // CHECK:   ins(%[[IN_SLICE_HIGH]]
  // CHECK:   outs(%[[OUT_SLICE_HIGH]]
  // CHECK: %[[SPLIT_HIGH_4:.+]] = affine.apply #[[$MAP_S_MINUS_100]]()[%[[SPLIT_LOW]]]
  // CHECK: tensor.insert_slice %[[RES_SLICE_HIGH]] into %[[PARTIAL]][%[[SPLIT_LOW]]] [%[[SPLIT_HIGH_4]]] [1]
  %0 = func.call @get_size() : () -> index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  }
  ins(%arg0: tensor<100xf32>) outs(%arg1: tensor<100xf32>) {
  ^bb0(%3: f32, %4: f32):
    %5 = arith.addf %3, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<100xf32>
  return %1 : tensor<100xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1:2 = transform.structured.split %0 after 4 { dimension = 0 } : !pdl.operation
  %2:2 = transform.structured.split %1#1 after 16 { dimension = 1 } : !pdl.operation
}

func.func private @elem(%arg0: f32, %arg1: index, %arg2: index) -> f32

// CHECK-LABEL: @two_d
func.func @two_d(%arg0: tensor<10x34xf32>,
                 %arg1: tensor<10x34xf32>) -> tensor<10x34xf32> {
  // Check the overall structure: split along the dimension 0, and then split
  // the second half only along the dimension 1.
  // CHECK:      %[[IN_1:.+]] = tensor.extract_slice %[[IN:.+]][0, 0]
  // CHECK:      %[[OUT_1:.+]] = tensor.extract_slice %[[OUT:.+]][0, 0]
  // CHECK:      %[[RES_1:.+]] = linalg.generic
  // CHECK-SAME:   ins(%[[IN_1]] : tensor<4x34xf32>)
  // CHECK-SAME:   outs(%[[OUT_1]] : tensor<4x34xf32>)
  // CHECK:      %[[PARTIAL_1:.+]] = tensor.insert_slice %[[RES_1]] into %[[OUT]]
  //
  // CHECK:      %[[IN_2:.+]] = tensor.extract_slice %[[IN]]
  // CHECK:      %[[OUT_2:.+]] = tensor.extract_slice %[[PARTIAL_1]]
  // Note that `extract_slice` taking a slice from another `extract_slice` result
  // is folded to use the operand of the first `extract_slice`.
  // CHECK:      %[[IN_21:.+]] = tensor.extract_slice %[[IN_2]]
  // CHECK:      %[[OUT_21:.+]] = tensor.extract_slice %[[OUT_2]]
  // CHECK:      %[[RES_21:.+]] = linalg.generic
  // CHECK-SAME:   ins(%[[IN_21]] : tensor<6x16xf32>)
  // CHECK-SAME:   outs(%[[OUT_21]] : tensor<6x16xf32>)
  // CHECK:      %[[PARTIAL_21:.+]] = tensor.insert_slice %[[RES_21]] into %[[OUT_2]]
  //
  // CHECK:      %[[IN_22:.+]] = tensor.extract_slice %[[IN_2]]
  // CHECK:      %[[OUT_22:.+]] = tensor.extract_slice %[[PARTIAL_21]]
  // CHECK:      %[[RES_22:.+]] = linalg.generic
  // CHECK-SAME:   ins(%[[IN_22]] : tensor<6x18xf32>)
  // CHECK-SAME:   outs(%[[OUT_22]] : tensor<6x18xf32>)
  // CHECK:      %[[PARTIAL_22:.+]] = tensor.insert_slice %[[RES_22]] into %[[PARTIAL_21]]
  // CHECK:      %[[PARTIAL_2:.+]] = tensor.insert_slice %[[PARTIAL_22]] into %[[PARTIAL_1]]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  }
  ins(%arg0: tensor<10x34xf32>)
  outs(%arg1: tensor<10x34xf32>) {
  ^bb0(%0: f32, %1: f32):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %call_res = func.call @elem(%0, %i, %j) : (f32, index, index) -> f32
    linalg.yield %call_res : f32
  } -> tensor<10x34xf32>
  return %0 : tensor<10x34xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // expected-error @below {{expects either a dynamic or a static split point to be provided}}
  %0:2 = "transform.structured.split"(%arg1) { dimension = 1, static_split_point = -9223372036854775808 } : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = transform.structured.match ops{["func.call"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{expected dynamic split point handle to point to a single-result index-typed op}}
  transform.structured.split %0 after %1 { dimension = 0 } : !pdl.operation, !pdl.operation
}

func.func private @get_size() -> i64

func.func @dynamic(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  // expected-note @below {{dynamic split point}}
  %0 = func.call @get_size() : () -> i64
  %1 = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  }
  ins(%arg0: tensor<100xf32>) outs(%arg1: tensor<100xf32>) {
  ^bb0(%3: f32, %4: f32):
    linalg.yield %3 : f32
  } -> tensor<100xf32>
  return %1 : tensor<100xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = transform.structured.match ops{["func.call"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{expected the dynamic split point handle to point to as many operations (0) as the target handle (1)}}
  transform.structured.split %0 after %1 { dimension = 0 } : !pdl.operation, !pdl.operation
}

func.func private @get_size() -> i64

func.func @dynamic(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  %1 = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  }
  ins(%arg0: tensor<100xf32>) outs(%arg1: tensor<100xf32>) {
  ^bb0(%3: f32, %4: f32):
    linalg.yield %3 : f32
  } -> tensor<100xf32>
  return %1 : tensor<100xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["func.return"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{only applies to structured ops}}
  transform.structured.split %0 after 16 { dimension = 1 } : !pdl.operation
}

func.func @noop(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  // expected-note @below {{target op}}
  return %arg0 : tensor<100xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{dimension 1 does not exist in target op}}
  transform.structured.split %0 after 16 { dimension = 1 } : !pdl.operation
}

func.func @one_d_static(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> {
  // expected-note @below {{target op}}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  }
  ins(%arg0: tensor<100xf32>) outs(%arg1: tensor<100xf32>) {
  ^bb0(%0: f32, %1: f32):
    linalg.yield %0 : f32
  } -> tensor<100xf32>
  return %0 : tensor<100xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{splitting does not produce the second part for a subset of targets}}
  // expected-note @below {{expected splitting to produce the second part of all or none of the targets}}
  %1:2 = transform.structured.split %0 after 142 { dimension = 0 } : !pdl.operation
}

func.func private @elem(%arg0: f32, %arg1: index, %arg2: index) -> f32

func.func @split_one_but_not_other(
    %arg0: tensor<100xf32>, %arg1: tensor<100xf32>,
    %arg2: tensor<200xf32>, %arg3: tensor<200xf32>)
    -> (tensor<100xf32>, tensor<200xf32>) {
  // expected-note @below {{first target with no second part}}
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  }
  ins(%arg0: tensor<100xf32>) outs(%arg1: tensor<100xf32>) {
  ^bb0(%arg4: f32, %arg5: f32):
    %i = linalg.index 0 : index
    %call_res = func.call @elem(%arg4, %i, %i) : (f32, index, index) -> f32
    linalg.yield %call_res : f32
  } -> tensor<100xf32>

  %1 = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  }
  ins(%arg2: tensor<200xf32>) outs(%arg3: tensor<200xf32>) {
  ^bb0(%arg4: f32, %arg5: f32):
    %i = linalg.index 0 : index
    %call_res = func.call @elem(%arg4, %i, %i) : (f32, index, index) -> f32
    linalg.yield %call_res : f32
  } -> tensor<200xf32>

  return %0, %1 : tensor<100xf32>, tensor<200xf32>
}

