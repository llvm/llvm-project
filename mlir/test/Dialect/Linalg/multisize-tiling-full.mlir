// RUN: mlir-opt --test-transform-dialect-interpreter --canonicalize %s | FileCheck %s

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // This implements a 2D multisize tiling with target sizes [3, 10].
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1:3 = transform.structured.multitile_sizes %0 { dimension = 0, target_size = 3}
    %t:3 = transform.structured.multitile_sizes %0 { dimension = 1, target_size = 10}
    %2:2 = transform.structured.split %0 after %1#2 { dimension = 0 }
    %3:2 = transform.structured.tile %2#0 [%1#0]
    %4:2 = transform.structured.tile %2#1 [%1#1]
    %5 = merge_handles %3#0, %4#0
    %tt:3 = replicate num(%5) %t#0, %t#1, %t#2
    %6:2 = transform.structured.split %5 after %tt#2 { dimension = 1 }
    transform.structured.tile %6#0 [0, %tt#0]
    transform.structured.tile %6#1 [0, %tt#1]
  }
}

func.func private @elem(%arg0: f32, %arg1: index, %arg2: index) -> f32

// CHECK-LABEL: @two_d
// CHECK-SAME: %[[IN:.+]]: tensor<10x34xf32>, %[[OUT:.+]]: tensor<10x34xf32>
func.func @two_d(%arg0: tensor<10x34xf32>,
                 %arg1: tensor<10x34xf32>) -> tensor<10x34xf32> {
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

  // 2D multi-size tiling should produce for quadrants with sizes
  //   (2, 8), (2, 9), (3, 8), (3, 9)
  // respectively, and in this order.
  // Check the full code for the first quadrant, the data flow for the second
  // quadrant and only the overall code structure for the remaining quadrants.
  // The canonicalizer is able to recover static shapes of for linalg.generic
  // instances, use those to differentiate the quadrants.

  // CHECK:      %[[SLICE_1:.+]] = tensor.extract_slice %[[OUT]][0, 0] [4, 34] [1, 1]
  // CHECK:      scf.for %[[I1:.+]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITERARG_1:.+]] = %[[SLICE_1]])
  // CHECK:        %[[OUTSLICE_1:.+]] = tensor.extract_slice %[[ITERARG_1]][%[[I1]], 0] [2, 34] [1, 1]

  // CHECK:        %[[SLICE_2:.+]] = tensor.extract_slice %[[ITERARG_1]][%[[I1]], 0] [2, 16] [1, 1]
  // CHECK:        %[[LOOPRES:.+]] = scf.for %[[I2:.+]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITERARG_2:.+]] = %[[SLICE_2]])
  // CHECK:          %[[INSLICE_2:.+]] = tensor.extract_slice %[[IN]][%[[I1]], %[[I2]]] [2, 8] [1, 1]
  // CHECK:          %[[OUTSLICE_2:.+]] = tensor.extract_slice %[[ITERARG_2]][0, %[[I2]]] [2, 8] [1, 1]
  // CHECK:          %[[RESSLICE_1:.+]] = linalg.generic {{.*}} ins(%[[INSLICE_2]] : tensor<2x8xf32>) outs(%[[OUTSLICE_2]] : tensor<2x8xf32>)
  // CHECK:          %[[RESPARTIAL:.+]] = tensor.insert_slice %[[RESSLICE_1]] into %[[ITERARG_2]]
  // CHECK:          scf.yield %[[RESPARTIAL]]

  // CHECK:        %[[INSERTED:.+]] = tensor.insert_slice %[[LOOPRES]] into %[[OUTSLICE_1]][%[[I1]], 0] [2, 16] [1, 1]
  // CHECK:        %[[OUTSLICE_3:.+]] = tensor.extract_slice %[[INSERTED]][0, 16] [2, 18] [1, 1]
  // CHECK:        scf.for %{{.*}} iter_args(%{{.*}} = %[[OUTSLICE_3]])
  // CHECK-COUNT-2:  tensor.extract_slice
  // CHECK:          linalg.generic {{.*}} ins(%{{.*}} : tensor<2x9xf32>)
  // CHECK:          tensor.insert_slice
  // CHECK:          scf.yield
  // CHECK:        %[[INSERTED_2:.+]] = tensor.insert_slice %{{.*}} into %[[INSERTED]]
  // CHECK:        %[[INSERTED_3:.+]] = tensor.insert_slice %[[INSERTED_2]] into %[[ITERARG_1]]
  // CHECK:        scf.yield %[[INSERTED_3]]

  // CHECK:        tensor.insert_slice
  // CHECK:        tensor.extract_slice
  // CHECK:        scf.for
  // CHECK-COUNT-2:  tensor.extract_slice
  // CHECK:          scf.for
  // CHECK-COUNT-2:    tensor.extract_slice
  // CHECK:            linalg.generic {{.*}} ins(%{{.*}} : tensor<3x8xf32>)
  // CHECK:            tensor.insert_slice
  // CHECK:            scf.yield
  // CHECK:          tensor.insert_slice
  // CHECK:          tensor.extract_slice
  // CHECK:          scf.for
  // CHECK-COUNT-2:    tensor.extract_slice
  // CHECK:            linalg.generic {{.*}} ins(%{{.*}} : tensor<3x9xf32>)
  // CHECK:            tensor.insert_slice
  // CHECK:            scf.yield
  // CHECK-COUNT-2:  tensor.insert_slice
  // CHECK:          scf.yield
  // CHECK:        %[[RESULT:.+]] = tensor.insert_slice
  // CHECK:        return %[[RESULT]]

  return %0 : tensor<10x34xf32>
}
