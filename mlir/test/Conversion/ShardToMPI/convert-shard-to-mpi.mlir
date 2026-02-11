// RUN: mlir-opt %s -convert-shard-to-mpi -canonicalize -split-input-file | FileCheck %s

// -----
// CHECK: shard.grid @grid0
shard.grid @grid0(shape = 3x4x5)
func.func @process_multi_index() -> (index, index, index) {
  // CHECK: mpi.comm_rank
  // CHECK: [[v1:%.*]] = arith.index_cast
  // CHECK: [[res:%.*]]:3 = affine.delinearize_index [[v1]] into (3, 4, 5) : index, index, index 
  %0:3 = shard.process_multi_index on @grid0 axes = [] : index, index, index
  // CHECK: return [[res]]#0, [[res]]#1, [[res]]#2 : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @process_multi_index_reorder
func.func @process_multi_index_reorder() -> (index, index) {
  // CHECK: mpi.comm_rank
  // CHECK: [[v1:%.*]] = arith.index_cast
  // CHECK: [[v2:%.*]]:3 = affine.delinearize_index [[v1]] into (3, 4, 5) : index, index, index
  %0:2 = shard.process_multi_index on @grid0 axes = [2, 0] : index, index
  // CHECK: return [[v2]]#2, [[v2]]#0 : index, index
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func @process_linear_index
func.func @process_linear_index() -> index {
  // CHECK: %[[RES:.*]], %[[rank:.*]] = mpi.comm_rank
  // CHECK: %[[cast:.*]] = arith.index_cast %[[rank]] : i32 to index
  %0 = shard.process_linear_index on @grid0 : index
  // CHECK: return %[[cast]] : index
  return %0 : index
}

// CHECK-LABEL: func @neighbors_dim0
func.func @neighbors_dim0(%arg0 : tensor<120x120x120xi8>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK-DAG: [[up:%.*]] = arith.constant 44 : index
  // CHECK-DAG: [[down:%.*]] = arith.constant 4 : index
  %idx:2 = shard.neighbors_linear_indices on @grid0[%c1, %c0, %c4] split_axes = [0] : index, index
  // CHECK: return [[down]], [[up]] : index, index
  return %idx#0, %idx#1 : index, index
}

// CHECK-LABEL: func @neighbors_dim1
func.func @neighbors_dim1(%arg0 : tensor<120x120x120xi8>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK-DAG: [[up:%.*]] = arith.constant 29 : index
  // CHECK-DAG: [[down:%.*]] = arith.constant -1 : index
  %idx:2 = shard.neighbors_linear_indices on @grid0[%c1, %c0, %c4] split_axes = [1] : index, index
  // CHECK: return [[down]], [[up]] : index, index
  return %idx#0, %idx#1 : index, index
}

// CHECK-LABEL: func @neighbors_dim2
func.func @neighbors_dim2(%arg0 : tensor<120x120x120xi8>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK-DAG: [[up:%.*]] = arith.constant -1 : index
  // CHECK-DAG: [[down:%.*]] = arith.constant 23 : index
  %idx:2 = shard.neighbors_linear_indices on @grid0[%c1, %c0, %c4] split_axes = [2] : index, index
  // CHECK: return [[down]], [[up]] : index, index
  return %idx#0, %idx#1 : index, index
}

// -----
// CHECK: shard.grid @grid0
module attributes { mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 24> } {
  shard.grid @grid0(shape = 3x4x5)
  func.func @process_multi_index() -> (index, index, index) {
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    %0:3 = shard.process_multi_index on @grid0 axes = [] : index, index, index
    // CHECK: return %[[c1]], %[[c0]], %[[c4]] : index, index, index
    return %0#0, %0#1, %0#2 : index, index, index
  }

  // CHECK-LABEL: func @process_linear_index
  func.func @process_linear_index() -> index {
    // CHECK: %[[c24:.*]] = arith.constant 24 : index
    %0 = shard.process_linear_index on @grid0 : index
    // CHECK: return %[[c24]] : index
    return %0 : index
  }
}

// -----
// CHECK: shard.grid @grid0
module {
  shard.grid @grid0(shape = 3x4x5)
  // CHECK-LABEL: func @all_slice
  func.func @all_slice(%arg0 : tensor<3x5xf32>) -> tensor<3x1xf32> {
    // CHECK: [[v0:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[vretval:%.*]], [[vrank:%.*]] = mpi.comm_rank([[v0]]) : !mpi.retval, i32
    // CHECK: [[v1:%.*]] = arith.index_cast [[vrank]] : i32 to index
    // CHECK: [[v2:%.*]]:3 = affine.delinearize_index [[v1]] into (3, 4, 5) : index, index, index
    // CHECK: [[vextracted_slice:%.*]] = tensor.extract_slice
    // CHECK-SAME: [0, [[v2]]#2] [3, 1] [1, 1] : tensor<3x5xf32> to tensor<3x1xf32>
    %1 = shard.all_slice %arg0 on @grid0 grid_axes = [2] slice_axis = 1 : tensor<3x5xf32> -> tensor<3x1xf32>
    return %1 : tensor<3x1xf32>
  }
}

// -----
module attributes { mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 7> } {
  shard.grid @grid0(shape = 3x4x5)
  // CHECK-LABEL: func.func @allreduce_tensor(
  func.func @allreduce_tensor(
    // CHECK-SAME: [[varg0:%.*]]: tensor<3x4xf32>
    %arg0 : tensor<3x4xf32>) -> tensor<3x4xf32> {
    // CHECK-DAG: [[vc1_i32:%.*]] = arith.constant 1 : i32
    // CHECK-DAG: [[vc2_i32:%.*]] = arith.constant 2 : i32
    // CHECK: [[v0:%.*]] = bufferization.to_buffer [[varg0]] : tensor<3x4xf32> to memref<3x4xf32>
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<3x4xf32>
    // CHECK: linalg.copy ins([[v0]] : memref<3x4xf32>) outs([[valloc]] : memref<3x4xf32>)
    // CHECK: [[v1:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[vnewcomm:%.*]] = mpi.comm_split([[v1]], [[vc2_i32]], [[vc1_i32]]) : !mpi.comm
    // CHECK: mpi.allreduce([[valloc]], [[valloc]], MPI_MAX, [[vnewcomm]]) : memref<3x4xf32>, memref<3x4xf32>
    // CHECK: [[v2:%.*]] = bufferization.to_tensor [[valloc]] restrict : memref<3x4xf32> to tensor<3x4xf32>
    %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0, 1] reduction = max : tensor<3x4xf32> -> tensor<3x4xf32>
    // CHECK: return [[v2]] : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @allreduce_memref(
  func.func @allreduce_memref(
    // CHECK-SAME: [[varg0:%.*]]: memref<3x4xf32>
    %arg0 : memref<3x4xf32>) -> memref<3x4xf32> {
    // CHECK-DAG: [[vc1_i32:%.*]] = arith.constant 1 : i32
    // CHECK-DAG: [[vc2_i32:%.*]] = arith.constant 2 : i32
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<3x4xf32>
    // CHECK: linalg.copy ins([[varg0]] : memref<3x4xf32>) outs([[valloc]] : memref<3x4xf32>)
    // CHECK: [[v0:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[vnewcomm:%.*]] = mpi.comm_split([[v0]], [[vc2_i32]], [[vc1_i32]]) : !mpi.comm
    // CHECK: mpi.allreduce([[valloc]], [[valloc]], MPI_MAX, [[vnewcomm]]) : memref<3x4xf32>, memref<3x4xf32>
    %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0, 1] reduction = max : memref<3x4xf32> -> memref<3x4xf32>
    // CHECK: return [[valloc]] : memref<3x4xf32>
    return %0 : memref<3x4xf32>
  }

  // CHECK-LABEL: func.func @allreduce_new_type(
  func.func @allreduce_new_type(
    // CHECK-SAME: [[varg0:%.*]]: memref<3x4xf32>
    %arg0 : memref<3x4xf32>) -> memref<3x4xf64> {
    // CHECK-DAG: [[vc1_i32:%.*]] = arith.constant 1 : i32
    // CHECK-DAG: [[vc2_i32:%.*]] = arith.constant 2 : i32
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<3x4xf64>
    // CHECK: linalg.copy ins([[varg0]] : memref<3x4xf32>) outs([[valloc]] : memref<3x4xf64>)
    // CHECK: [[v0:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[vnewcomm:%.*]] = mpi.comm_split([[v0]], [[vc2_i32]], [[vc1_i32]]) : !mpi.comm
    // CHECK: mpi.allreduce([[valloc]], [[valloc]], MPI_MAX, [[vnewcomm]]) : memref<3x4xf64>, memref<3x4xf64>
    %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0, 1] reduction = max : memref<3x4xf32> -> memref<3x4xf64>
    // CHECK: return [[valloc]] : memref<3x4xf64>
    return %0 : memref<3x4xf64>
  }

  // CHECK-LABEL: func @allgather_tensor_0
  // CHECK-SAME: [[varg0:%.*]]: tensor<3x4xf32>
  func.func @allgather_tensor_0(%arg0 : tensor<3x4xf32>) -> tensor<12x4xf32> {
    // CHECK-DAG: [[vc1_i32:%.*]] = arith.constant 1 : i32
    // CHECK-DAG: [[vc2_i32:%.*]] = arith.constant 2 : i32
    // CHECK-DAG: [[vc4:%.*]] = arith.constant 4 : index
    // CHECK: [[v0:%.*]] = bufferization.to_buffer [[varg0]] : tensor<3x4xf32> to memref<3x4xf32>
    // CHECK: [[v1:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[vnewcomm:%.*]] = mpi.comm_split([[v1]], [[vc2_i32]], [[vc1_i32]]) : !mpi.comm
    // CHECK: [[vsize:%.*]] = mpi.comm_size([[vnewcomm]]) : i32
    // CHECK: [[v2:%.*]] = arith.index_cast [[vsize]] : i32 to index
    // CHECK: [[v3:%.*]] = arith.cmpi eq, [[v2]], [[vc4]] : index
    // CHECK: cf.assert [[v3]]
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<4x3x4xf32>
    // CHECK: mpi.allgather([[v0]], [[valloc]], [[vnewcomm]]) : memref<3x4xf32>, memref<4x3x4xf32>
    // CHECK: [[vcollapse_shape:%.*]] = memref.collapse_shape [[valloc]] {{\[\[}}0, 1], [2]] : memref<4x3x4xf32> into memref<12x4xf32>
    // CHECK: [[v4:%.*]] = bufferization.to_tensor [[vcollapse_shape]] restrict : memref<12x4xf32> to tensor<12x4xf32>
    %0 = shard.all_gather %arg0 on @grid0 grid_axes = [1] gather_axis = 0 : tensor<3x4xf32> -> tensor<12x4xf32>
    // CHECK: return [[v4]] : tensor<12x4xf32>
    return %0 : tensor<12x4xf32>
  }

  // CHECK-LABEL: func @allgather_tensor
  func.func @allgather_tensor(
      // CHECK-SAME: [[varg0:%.*]]: tensor<3x4xf32>
      // CHECK-SAME: -> tensor<3x20xf32>
      %arg0 : tensor<3x4xf32>) -> tensor<3x20xf32> {
    // CHECK-DAG: [[vc2_i32:%.*]] = arith.constant 2 : i32
    // CHECK-DAG: [[vc1_i32:%.*]] = arith.constant 1 : i32
    // CHECK-DAG: [[vc5:%.*]] = arith.constant 5 : index
    // CHECK: [[v0:%.*]] = bufferization.to_buffer [[varg0]] : tensor<3x4xf32> to memref<3x4xf32>
    // CHECK: [[v1:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[vnewcomm:%.*]] = mpi.comm_split([[v1]], [[vc1_i32]], [[vc2_i32]]) : !mpi.comm
    // CHECK: [[vsize:%.*]] = mpi.comm_size([[vnewcomm]]) : i32
    // CHECK: [[v2:%.*]] = arith.index_cast [[vsize]] : i32 to index
    // CHECK: [[v3:%.*]] = arith.cmpi eq, [[v2]], [[vc5]] : index
    // CHECK: cf.assert [[v3]]
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<5x3x4xf32>
    // CHECK: mpi.allgather([[v0]], [[valloc]], [[vnewcomm]]) : memref<3x4xf32>, memref<5x3x4xf32>
    // CHECK: [[v4:%.*]] = bufferization.to_tensor [[valloc]] restrict : memref<5x3x4xf32> to tensor<5x3x4xf32>
    // CHECK: [[v5:%.*]] = tensor.empty() : tensor<3x5x4xf32>
    // CHECK: [[vtransposed:%.*]] = linalg.transpose ins([[v4]] : tensor<5x3x4xf32>) outs([[v5]] : tensor<3x5x4xf32>) permutation = [1, 0, 2]
    // CHECK: [[vcollapsed:%.*]] = tensor.collapse_shape [[vtransposed]] {{\[\[}}0], [1, 2]] : tensor<3x5x4xf32> into tensor<3x20xf32>
    %0 = shard.all_gather %arg0 on @grid0 grid_axes = [2] gather_axis = 1 : tensor<3x4xf32> -> tensor<3x20xf32>
    // CHECK: return [[vcollapsed]] : tensor<3x20xf32>
    return %0 : tensor<3x20xf32>
  }

  // CHECK-LABEL: func @allgather_memref
  func.func @allgather_memref(
      // CHECK-SAME: [[varg0:%.*]]: memref<3x4xf32>
      // CHECK-SAME: -> memref<3x20xf32>
      %arg0 : memref<3x4xf32>) -> memref<3x20xf32> {
    // CHECK-DAG: [[vc2_i32:%.*]] = arith.constant 2 : i32
    // CHECK-DAG: [[vc1_i32:%.*]] = arith.constant 1 : i32
    // CHECK-DAG: [[vc5:%.*]] = arith.constant 5 : index
    // CHECK: [[v0:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[vnewcomm:%.*]] = mpi.comm_split([[v0]], [[vc1_i32]], [[vc2_i32]]) : !mpi.comm
    // CHECK: [[vsize:%.*]] = mpi.comm_size([[vnewcomm]]) : i32
    // CHECK: [[v1:%.*]] = arith.index_cast [[vsize]] : i32 to index
    // CHECK: [[v2:%.*]] = arith.cmpi eq, [[v1]], [[vc5]] : index
    // CHECK: cf.assert [[v2]]
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<5x3x4xf32>
    // CHECK: mpi.allgather([[varg0]], [[valloc]], [[vnewcomm]]) : memref<3x4xf32>, memref<5x3x4xf32>
    // CHECK: [[v3:%.*]] = bufferization.to_tensor [[valloc]] restrict : memref<5x3x4xf32> to tensor<5x3x4xf32>
    // CHECK: [[v4:%.*]] = tensor.empty() : tensor<3x5x4xf32>
    // CHECK: [[vtransposed:%.*]] = linalg.transpose ins([[v3]] : tensor<5x3x4xf32>) outs([[v4]] : tensor<3x5x4xf32>) permutation = [1, 0, 2] 
    // CHECK: [[vcollapsed:%.*]] = tensor.collapse_shape [[vtransposed]] {{\[\[}}0], [1, 2]] : tensor<3x5x4xf32> into tensor<3x20xf32>
    // CHECK: [[v5:%.*]] = bufferization.to_buffer [[vcollapsed]] : tensor<3x20xf32> to memref<3x20xf32>
    %0 = shard.all_gather %arg0 on @grid0 grid_axes = [2] gather_axis = 1 : memref<3x4xf32> -> memref<3x20xf32>
    // CHECK: return [[v5]] : memref<3x20xf32>
    return %0 : memref<3x20xf32>
  }
}

// -----
shard.grid @grid0(shape = 3x4x5)
// CHECK-LABEL: func @update_halo_1d_first
func.func @update_halo_1d_first(
  // CHECK-SAME: [[arg0:%.*]]: memref<120x120x120xi8>
  %arg0 : memref<120x120x120xi8>) -> memref<120x120x120xi8> {
  // CHECK: memref.subview [[arg0]][115, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8
  // CHECK: mpi.send(
  // CHECK-SAME: : memref<2x120x120xi8>, i32, i32
  // CHECK: mpi.recv(
  // CHECK-SAME: : memref<2x120x120xi8>, i32, i32
  // CHECK: memref.subview [[arg0]][0, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8
  // CHECK: memref.subview [[arg0]][2, 0, 0] [3, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<3x120x120xi8
  // CHECK: mpi.send(
  // CHECK-SAME: : memref<3x120x120xi8>, i32, i32
  // CHECK: mpi.recv(
  // CHECK-SAME: : memref<3x120x120xi8>, i32, i32
  // CHECK: memref.subview [[arg0]][117, 0, 0] [3, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<3x120x120xi8
  %res = shard.update_halo %arg0 on @grid0 split_axes = [[0]] halo_sizes = [2, 3] : memref<120x120x120xi8>
  // CHECK: return [[res:%.*]] : memref<120x120x120xi8>
  return %res : memref<120x120x120xi8>
}

// -----
module attributes { mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 1> } {
  shard.grid @grid0(shape = 4)
  // CHECK-LABEL: func @update_halo_1d_with_zero
  func.func @update_halo_1d_with_zero (
    // CHECK-SAME: [[varg0:%.*]]: memref<120x120x120xi8>
    %arg0 : memref<120x120x120xi8>) -> memref<120x120x120xi8> {
    // CHECK-DAG: [[vc91_i32:%.*]] = arith.constant 91 : i32
    // CHECK-DAG: [[vc0_i32:%.*]] = arith.constant 0 : i32
    // CHECK-DAG: [[vc2_i32:%.*]] = arith.constant 2 : i32
    // CHECK: [[v0:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<2x120x120xi8>
    // CHECK: [[vsubview:%.*]] = memref.subview [[varg0]][118, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 1699200>>
    // CHECK: memref.copy [[vsubview]], [[valloc]] : memref<2x120x120xi8, strided<[14400, 120, 1], offset: 1699200>> to memref<2x120x120xi8>
    // CHECK: mpi.send([[valloc]], [[vc91_i32]], [[vc2_i32]], [[v0]]) : memref<2x120x120xi8>, i32, i32
    // CHECK: mpi.recv([[valloc]], [[vc91_i32]], [[vc0_i32]], [[v0]]) : memref<2x120x120xi8>, i32, i32
    // CHECK: [[vsubview_0:%.*]] = memref.subview [[varg0]][0, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK: memref.copy [[valloc]], [[vsubview_0]] : memref<2x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK: memref.dealloc [[valloc]] : memref<2x120x120xi8>
    %res = shard.update_halo %arg0 on @grid0 split_axes = [[0]] halo_sizes = [2, 0] : memref<120x120x120xi8>
    // CHECK: return [[varg0]] : memref<120x120x120xi8>
    return %res : memref<120x120x120xi8>
  }
}

// -----
module attributes { mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 24> } {
  shard.grid @grid0(shape = 3x4x5)
  // CHECK-LABEL: func @update_halo_3d
  func.func @update_halo_3d(
    // CHECK-SAME: [[varg0:%.*]]: memref<120x120x120xi8>
    %arg0 : memref<120x120x120xi8>) -> memref<120x120x120xi8> {
    // CHECK-DAG: [[vc23_i32:%.*]] = arith.constant 23 : i32
    // CHECK-DAG: [[vc29_i32:%.*]] = arith.constant 29 : i32
    // CHECK-DAG: [[vc91_i32:%.*]] = arith.constant 91 : i32
    // CHECK-DAG: [[vc4_i32:%.*]] = arith.constant 4 : i32
    // CHECK-DAG: [[vc44_i32:%.*]] = arith.constant 44 : i32
    // CHECK: [[v0:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<117x113x5xi8>
    // CHECK: [[vsubview:%.*]] = memref.subview [[varg0]][1, 3, 109] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>>
    // CHECK: memref.copy [[vsubview]], [[valloc]] : memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>> to memref<117x113x5xi8>
    // CHECK: mpi.send([[valloc]], [[vc91_i32]], [[vc44_i32]], [[v0]]) : memref<117x113x5xi8>, i32, i32
    // CHECK: mpi.recv([[valloc]], [[vc91_i32]], [[vc4_i32]], [[v0]]) : memref<117x113x5xi8>, i32, i32
    // CHECK: [[vsubview_0:%.*]] = memref.subview [[varg0]][1, 3, 0] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
    // CHECK: memref.copy [[valloc]], [[vsubview_0]] : memref<117x113x5xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
    // CHECK: memref.dealloc [[valloc]] : memref<117x113x5xi8>
    // CHECK: [[valloc_1:%.*]] = memref.alloc() : memref<117x113x6xi8>
    // CHECK: [[vsubview_2:%.*]] = memref.subview [[varg0]][1, 3, 5] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>>
    // CHECK: memref.copy [[vsubview_2]], [[valloc_1]] : memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>> to memref<117x113x6xi8>
    // CHECK: mpi.send([[valloc_1]], [[vc91_i32]], [[vc4_i32]], [[v0]]) : memref<117x113x6xi8>, i32, i32
    // CHECK: mpi.recv([[valloc_1]], [[vc91_i32]], [[vc44_i32]], [[v0]]) : memref<117x113x6xi8>, i32, i32
    // CHECK: [[vsubview_3:%.*]] = memref.subview [[varg0]][1, 3, 114] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
    // CHECK: memref.copy [[valloc_1]], [[vsubview_3]] : memref<117x113x6xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
    // CHECK: memref.dealloc [[valloc_1]] : memref<117x113x6xi8>
    // CHECK: [[v1:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[valloc_4:%.*]] = memref.alloc() : memref<117x3x120xi8>
    // CHECK: [[vsubview_5:%.*]] = memref.subview [[varg0]][1, 113, 0] [117, 3, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x3x120xi8, strided<[14400, 120, 1], offset: 27960>>
    // CHECK: memref.copy [[vsubview_5]], [[valloc_4]] : memref<117x3x120xi8, strided<[14400, 120, 1], offset: 27960>> to memref<117x3x120xi8>
    // CHECK: mpi.send([[valloc_4]], [[vc91_i32]], [[vc29_i32]], [[v1]]) : memref<117x3x120xi8>, i32, i32
    // CHECK: memref.dealloc [[valloc_4]] : memref<117x3x120xi8>
    // CHECK: [[valloc_6:%.*]] = memref.alloc() : memref<117x4x120xi8>
    // CHECK: mpi.recv([[valloc_6]], [[vc91_i32]], [[vc29_i32]], [[v1]]) : memref<117x4x120xi8>, i32, i32
    // CHECK: [[vsubview_7:%.*]] = memref.subview [[varg0]][1, 116, 0] [117, 4, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 28320>>
    // CHECK: memref.copy [[valloc_6]], [[vsubview_7]] : memref<117x4x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 28320>>
    // CHECK: memref.dealloc [[valloc_6]] : memref<117x4x120xi8>
    // CHECK: [[v2:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[valloc_8:%.*]] = memref.alloc() : memref<1x120x120xi8>
    // CHECK: mpi.recv([[valloc_8]], [[vc91_i32]], [[vc23_i32]], [[v2]]) : memref<1x120x120xi8>, i32, i32
    // CHECK: [[vsubview_9:%.*]] = memref.subview [[varg0]][0, 0, 0] [1, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK: memref.copy [[valloc_8]], [[vsubview_9]] : memref<1x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK: memref.dealloc [[valloc_8]] : memref<1x120x120xi8>
    // CHECK: [[valloc_10:%.*]] = memref.alloc() : memref<2x120x120xi8>
    // CHECK: [[vsubview_11:%.*]] = memref.subview [[varg0]][1, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 14400>>
    // CHECK: memref.copy [[vsubview_11]], [[valloc_10]] : memref<2x120x120xi8, strided<[14400, 120, 1], offset: 14400>> to memref<2x120x120xi8>
    // CHECK: mpi.send([[valloc_10]], [[vc91_i32]], [[vc23_i32]], [[v2]]) : memref<2x120x120xi8>, i32, i32
    // CHECK: memref.dealloc [[valloc_10]] : memref<2x120x120xi8>
    %res = shard.update_halo %arg0 on @grid0 split_axes = [[2], [1], [0]] halo_sizes = [1, 2, 3, 4, 5, 6] : memref<120x120x120xi8>
    // CHECK: return [[varg0]] : memref<120x120x120xi8>
    return %res : memref<120x120x120xi8>
  }

  // CHECK-LABEL: func @update_halo_3d_tensor
  func.func @update_halo_3d_tensor(
    // CHECK-SAME: [[varg0:%.*]]: tensor<120x120x120xi8>
    %arg0 : tensor<120x120x120xi8>) -> tensor<120x120x120xi8> {
    // CHECK-DAG: [[vc23_i32:%.*]] = arith.constant 23 : i32
    // CHECK-DAG: [[vc29_i32:%.*]] = arith.constant 29 : i32
    // CHECK-DAG: [[vc44_i32:%.*]] = arith.constant 44 : i32
    // CHECK-DAG: [[vc4_i32:%.*]] = arith.constant 4 : i32
    // CHECK-DAG: [[vc91_i32:%.*]] = arith.constant 91 : i32
    // CHECK: [[v0:%.*]] = bufferization.to_buffer [[varg0]] : tensor<120x120x120xi8> to memref<120x120x120xi8>
    // CHECK: [[v1:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[valloc:%.*]] = memref.alloc() : memref<117x113x5xi8>
    // CHECK: [[vsubview:%.*]] = memref.subview [[v0]][1, 3, 109] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>>
    // CHECK: memref.copy [[vsubview]], [[valloc]] : memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>> to memref<117x113x5xi8>
    // CHECK: mpi.send([[valloc]], [[vc91_i32]], [[vc44_i32]], [[v1]]) : memref<117x113x5xi8>, i32, i32
    // CHECK: mpi.recv([[valloc]], [[vc91_i32]], [[vc4_i32]], [[v1]]) : memref<117x113x5xi8>, i32, i32
    // CHECK: [[vsubview_0:%.*]] = memref.subview [[v0]][1, 3, 0] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
    // CHECK: memref.copy [[valloc]], [[vsubview_0]] : memref<117x113x5xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
    // CHECK: memref.dealloc [[valloc]] : memref<117x113x5xi8>
    // CHECK: [[valloc_1:%.*]] = memref.alloc() : memref<117x113x6xi8>
    // CHECK: [[vsubview_2:%.*]] = memref.subview [[v0]][1, 3, 5] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>>
    // CHECK: memref.copy [[vsubview_2]], [[valloc_1]] : memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>> to memref<117x113x6xi8>
    // CHECK: mpi.send([[valloc_1]], [[vc91_i32]], [[vc4_i32]], [[v1]]) : memref<117x113x6xi8>, i32, i32
    // CHECK: mpi.recv([[valloc_1]], [[vc91_i32]], [[vc44_i32]], [[v1]]) : memref<117x113x6xi8>, i32, i32
    // CHECK: [[vsubview_3:%.*]] = memref.subview [[v0]][1, 3, 114] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
    // CHECK: memref.copy [[valloc_1]], [[vsubview_3]] : memref<117x113x6xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
    // CHECK: memref.dealloc [[valloc_1]] : memref<117x113x6xi8>
    // CHECK: [[v2:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[valloc_4:%.*]] = memref.alloc() : memref<117x3x120xi8>
    // CHECK: [[vsubview_5:%.*]] = memref.subview [[v0]][1, 113, 0] [117, 3, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x3x120xi8, strided<[14400, 120, 1], offset: 27960>>
    // CHECK: memref.copy [[vsubview_5]], [[valloc_4]] : memref<117x3x120xi8, strided<[14400, 120, 1], offset: 27960>> to memref<117x3x120xi8>
    // CHECK: mpi.send([[valloc_4]], [[vc91_i32]], [[vc29_i32]], [[v2]]) : memref<117x3x120xi8>, i32, i32
    // CHECK: memref.dealloc [[valloc_4]] : memref<117x3x120xi8>
    // CHECK: [[valloc_6:%.*]] = memref.alloc() : memref<117x4x120xi8>
    // CHECK: mpi.recv([[valloc_6]], [[vc91_i32]], [[vc29_i32]], [[v2]]) : memref<117x4x120xi8>, i32, i32
    // CHECK: [[vsubview_7:%.*]] = memref.subview [[v0]][1, 116, 0] [117, 4, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 28320>>
    // CHECK: memref.copy [[valloc_6]], [[vsubview_7]] : memref<117x4x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 28320>>
    // CHECK: memref.dealloc [[valloc_6]] : memref<117x4x120xi8>
    // CHECK: [[v3:%.*]] = mpi.comm_world : !mpi.comm
    // CHECK: [[valloc_8:%.*]] = memref.alloc() : memref<1x120x120xi8>
    // CHECK: mpi.recv([[valloc_8]], [[vc91_i32]], [[vc23_i32]], [[v3]]) : memref<1x120x120xi8>, i32, i32
    // CHECK: [[vsubview_9:%.*]] = memref.subview [[v0]][0, 0, 0] [1, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK: memref.copy [[valloc_8]], [[vsubview_9]] : memref<1x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK: memref.dealloc [[valloc_8]] : memref<1x120x120xi8>
    // CHECK: [[valloc_10:%.*]] = memref.alloc() : memref<2x120x120xi8>
    // CHECK: [[vsubview_11:%.*]] = memref.subview [[v0]][1, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 14400>>
    // CHECK: memref.copy [[vsubview_11]], [[valloc_10]] : memref<2x120x120xi8, strided<[14400, 120, 1], offset: 14400>> to memref<2x120x120xi8>
    // CHECK: mpi.send([[valloc_10]], [[vc91_i32]], [[vc23_i32]], [[v3]]) : memref<2x120x120xi8>, i32, i32
    // CHECK: memref.dealloc [[valloc_10]] : memref<2x120x120xi8>
    // CHECK: [[v4:%.*]] = bufferization.to_tensor [[v0]] restrict writable : memref<120x120x120xi8> to tensor<120x120x120xi8>
    %res = shard.update_halo %arg0 on @grid0 split_axes = [[2], [1], [0]] halo_sizes = [1, 2, 3, 4, 5, 6] : tensor<120x120x120xi8>
    // CHECK: return [[v4]] : tensor<120x120x120xi8>
    return %res : tensor<120x120x120xi8>
  }
}

// -----
shard.grid @grid0(shape = 2x2x4)
// CHECK-LABEL: func.func @return_sharding(
// CHECK-SAME: [[varg0:%.*]]: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>) {
func.func @return_sharding(%arg0: tensor<2x4xf32>) -> (tensor<2x4xf32>, !shard.sharding) {
  %sharding = shard.sharding @grid0 split_axes = [[0, 1], [2]] : !shard.sharding
  // CHECK-DAG: [[vcst:%.*]] = arith.constant dense<2> : tensor<1xi16>
  // CHECK-DAG: [[vcst_0:%.*]] = arith.constant dense<[0, 1]> : tensor<2xi16>
  // CHECK-DAG: [[vcm1_i16:%.*]] = arith.constant -1 : i16
  // CHECK: [[v0:%.*]] = tensor.empty() : tensor<2x2xi16>
  // CHECK: [[v1:%.*]] = linalg.fill ins([[vcm1_i16]] : i16) outs([[v0]] : tensor<2x2xi16>) -> tensor<2x2xi16>
  // CHECK: [[vinserted_slice:%.*]] = tensor.insert_slice [[vcst_0]] into [[v1]][0, 0] [1, 2] [1, 1] : tensor<2xi16> into tensor<2x2xi16>
  // CHECK: [[vinserted_slice_1:%.*]] = tensor.insert_slice [[vcst]] into [[vinserted_slice]][1, 0] [1, 1] [1, 1] : tensor<1xi16> into tensor<2x2xi16>
  // CHECK: [[v2:%.*]] = tensor.empty() : tensor<0x0xi64>
  // CHECK: [[v3:%.*]] = tensor.empty() : tensor<0x0xi64>
  // CHECK: [[vcast:%.*]] = tensor.cast [[vinserted_slice_1]] : tensor<2x2xi16> to tensor<?x?xi16>
  // CHECK: [[vcast_2:%.*]] = tensor.cast [[v2]] : tensor<0x0xi64> to tensor<?x?xi64>
  // CHECK: [[vcast_3:%.*]] = tensor.cast [[v3]] : tensor<0x0xi64> to tensor<?x?xi64>
  // CHECK: return [[varg0]], [[vcast]], [[vcast_2]], [[vcast_3]] : tensor<2x4xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>
  return %arg0, %sharding : tensor<2x4xf32>, !shard.sharding
}

// CHECK-LABEL: func.func @return_sharding_halos(
// CHECK-SAME: [[varg0:%.*]]: tensor<6x8xf32>) -> (tensor<6x8xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>) {
func.func @return_sharding_halos(%arg0: tensor<6x8xf32>) -> (tensor<6x8xf32>, !shard.sharding) {
  %sharding = shard.sharding @grid0 split_axes = [[0, 1], [2]] halo_sizes = [0, 4, 3, 1] : !shard.sharding
  // CHECK-DAG: [[vcst:%.*]] = arith.constant dense<{{\[\[}}0, 4], [3, 1]]> : tensor<2x2xi64>
  // CHECK-DAG: [[vcst_0:%.*]] = arith.constant dense<2> : tensor<1xi16>
  // CHECK-DAG: [[vcst_1:%.*]] = arith.constant dense<[0, 1]> : tensor<2xi16>
  // CHECK-DAG: [[vcm1_i16:%.*]] = arith.constant -1 : i16
  // CHECK: [[v0:%.*]] = tensor.empty() : tensor<2x2xi16>
  // CHECK: [[v1:%.*]] = linalg.fill ins([[vcm1_i16]] : i16) outs([[v0]] : tensor<2x2xi16>) -> tensor<2x2xi16>
  // CHECK: [[vinserted_slice:%.*]] = tensor.insert_slice [[vcst_1]] into [[v1]][0, 0] [1, 2] [1, 1] : tensor<2xi16> into tensor<2x2xi16>
  // CHECK: [[vinserted_slice_2:%.*]] = tensor.insert_slice [[vcst_0]] into [[vinserted_slice]][1, 0] [1, 1] [1, 1] : tensor<1xi16> into tensor<2x2xi16>
  // CHECK: [[v2:%.*]] = tensor.empty() : tensor<0x0xi64>
  // CHECK: [[vcast:%.*]] = tensor.cast [[vinserted_slice_2]] : tensor<2x2xi16> to tensor<?x?xi16>
  // CHECK: [[vcast_3:%.*]] = tensor.cast [[vcst]] : tensor<2x2xi64> to tensor<?x?xi64>
  // CHECK: [[vcast_4:%.*]] = tensor.cast [[v2]] : tensor<0x0xi64> to tensor<?x?xi64>
  // CHECK: return [[varg0]], [[vcast]], [[vcast_3]], [[vcast_4]] : tensor<6x8xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>
  return %arg0, %sharding : tensor<6x8xf32>, !shard.sharding
}

// CHECK-LABEL: func.func @return_sharding_offs(
// CHECK-SAME: [[varg0:%.*]]: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>) {
func.func @return_sharding_offs(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>, !shard.sharding) {
  %sharding = shard.sharding @grid0 split_axes = [[0, 1], [2]] sharded_dims_offsets = [0, 3, 5, 7, 8, 0, 0, 5, 10, 16] : !shard.sharding
  // CHECK-DAG: [[vcst:%.*]] = arith.constant dense<[0, 0, 5, 10, 16]> : tensor<5xi64>
  // CHECK-DAG: [[vcst_0:%.*]] = arith.constant dense<[0, 3, 5, 7, 8]> : tensor<5xi64>
  // CHECK-DAG: [[vcm9223372036854775808_i64:%.*]] = arith.constant -9223372036854775808 : i64
  // CHECK-DAG: [[vcst_1:%.*]] = arith.constant dense<2> : tensor<1xi16>
  // CHECK-DAG: [[vcst_2:%.*]] = arith.constant dense<[0, 1]> : tensor<2xi16>
  // CHECK-DAG: [[vcm1_i16:%.*]] = arith.constant -1 : i16
  // CHECK: [[v0:%.*]] = tensor.empty() : tensor<2x2xi16>
  // CHECK: [[v1:%.*]] = linalg.fill ins([[vcm1_i16]] : i16) outs([[v0]] : tensor<2x2xi16>) -> tensor<2x2xi16>
  // CHECK: [[vinserted_slice:%.*]] = tensor.insert_slice [[vcst_2]] into [[v1]][0, 0] [1, 2] [1, 1] : tensor<2xi16> into tensor<2x2xi16>
  // CHECK: [[vinserted_slice_3:%.*]] = tensor.insert_slice [[vcst_1]] into [[vinserted_slice]][1, 0] [1, 1] [1, 1] : tensor<1xi16> into tensor<2x2xi16>
  // CHECK: [[v2:%.*]] = tensor.empty() : tensor<0x0xi64>
  // CHECK: [[v3:%.*]] = tensor.empty() : tensor<2x5xi64>
  // CHECK: [[v4:%.*]] = linalg.fill ins([[vcm9223372036854775808_i64]] : i64) outs([[v3]] : tensor<2x5xi64>) -> tensor<2x5xi64>
  // CHECK: [[vinserted_slice_4:%.*]] = tensor.insert_slice [[vcst_0]] into [[v4]][0, 0] [1, 5] [1, 1] : tensor<5xi64> into tensor<2x5xi64>
  // CHECK: [[vinserted_slice_5:%.*]] = tensor.insert_slice [[vcst]] into [[vinserted_slice_4]][1, 0] [1, 5] [1, 1] : tensor<5xi64> into tensor<2x5xi64>
  // CHECK: [[vcast:%.*]] = tensor.cast [[vinserted_slice_3]] : tensor<2x2xi16> to tensor<?x?xi16>
  // CHECK: [[vcast_6:%.*]] = tensor.cast [[v2]] : tensor<0x0xi64> to tensor<?x?xi64>
  // CHECK: [[vcast_7:%.*]] = tensor.cast [[vinserted_slice_5]] : tensor<2x5xi64> to tensor<?x?xi64>
  // CHECK: return [[varg0]], [[vcast]], [[vcast_6]], [[vcast_7]] : tensor<?x?xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>
  return %arg0, %sharding : tensor<?x?xf32>, !shard.sharding
}

// -----
shard.grid @grid_1d_4(shape = 4)
// CHECK-LABEL: func.func @mlp_1dgrid(
// CHECK-SAME: [[varg0:%.*]]: tensor<512x512xf32>, [[varg1:%.*]]: tensor<2048x256xf32>, [[varg2:%.*]]: tensor<256x2048xf32>) -> tensor<512x2048xf32>
func.func @mlp_1dgrid(%arg0: tensor<512x512xf32>, %arg1: tensor<2048x256xf32>, %arg2: tensor<256x2048xf32>) -> tensor<512x2048xf32> attributes {llvm.emit_c_interface} {
  // CHECK-DAG: [[vcst:%.*]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[vc0:%.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: [[vc4:%.*]] = arith.constant 4 : index
  // CHECK: [[v0:%.*]] = bufferization.to_buffer [[varg0]] : tensor<512x512xf32> to memref<512x512xf32>
  // CHECK: [[v1:%.*]] = mpi.comm_world : !mpi.comm
  // CHECK: [[vsize:%.*]] = mpi.comm_size([[v1]]) : i32
  // CHECK: [[v2:%.*]] = arith.index_cast [[vsize]] : i32 to index
  // CHECK: [[v3:%.*]] = arith.cmpi eq, [[v2]], [[vc4]] : index
  // CHECK: cf.assert [[v3]]
  // CHECK: [[valloc:%.*]] = memref.alloc() : memref<4x512x512xf32>
  // CHECK: mpi.allgather([[v0]], [[valloc]], [[v1]]) : memref<512x512xf32>, memref<4x512x512xf32>
  // CHECK: [[v4:%.*]] = bufferization.to_tensor [[valloc]] restrict : memref<4x512x512xf32> to tensor<4x512x512xf32>
  // CHECK: [[v5:%.*]] = tensor.empty() : tensor<512x4x512xf32>
  // CHECK: [[vtransposed:%.*]] = linalg.transpose ins([[v4]] : tensor<4x512x512xf32>) outs([[v5]] : tensor<512x4x512xf32>) permutation = [1, 0, 2] 
  // CHECK: [[vcollapsed:%.*]] = tensor.collapse_shape [[vtransposed]] {{\[\[}}0], [1, 2]] : tensor<512x4x512xf32> into tensor<512x2048xf32>
  %all_gather = shard.all_gather %arg0 on @grid_1d_4 grid_axes = [0] gather_axis = 1 : tensor<512x512xf32> -> tensor<512x2048xf32>
  // CHECK: [[v6:%.*]] = tensor.empty() : tensor<512x256xf32>
  %0 = tensor.empty() : tensor<512x256xf32>
  // CHECK: [[v7:%.*]] = linalg.fill ins([[vcst]] : f32) outs([[v6]] : tensor<512x256xf32>) -> tensor<512x256xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<512x256xf32>) -> tensor<512x256xf32>
  // CHECK: [[v8:%.*]] = linalg.matmul ins([[vcollapsed]], [[varg1]] : tensor<512x2048xf32>, tensor<2048x256xf32>) outs([[v7]] : tensor<512x256xf32>) -> tensor<512x256xf32>
  %2 = linalg.matmul ins(%all_gather, %arg1 : tensor<512x2048xf32>, tensor<2048x256xf32>) outs(%1 : tensor<512x256xf32>) -> tensor<512x256xf32>
  // CHECK: [[v9:%.*]] = tosa.sigmoid [[v8]] : (tensor<512x256xf32>) -> tensor<512x256xf32>
  %3 = tosa.sigmoid %2 : (tensor<512x256xf32>) -> tensor<512x256xf32>
  %4 = tensor.empty() : tensor<512x2048xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<512x2048xf32>) -> tensor<512x2048xf32>
  %proc_linear_idx = shard.process_multi_index on @grid_1d_4 axes = [0] : index
  %grid_shape = shard.grid_shape @grid_1d_4 axes = [0] : index
  %6 = arith.cmpi eq, %proc_linear_idx, %c0 : index
  // CHECK: [[v15:%.*]] = scf.if
  %7 = scf.if %6 -> (tensor<512x2048xf32>) {
    scf.yield %5 : tensor<512x2048xf32>
  } else {
    %9 = tensor.empty() : tensor<512x2048xf32>
    %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<512x2048xf32>) -> tensor<512x2048xf32>
    scf.yield %10 : tensor<512x2048xf32>
  }
  // CHECK: [[v16:%.*]] = linalg.matmul ins([[v9]], [[varg2]] : tensor<512x256xf32>, tensor<256x2048xf32>) outs([[v15]] : tensor<512x2048xf32>) -> tensor<512x2048xf32>
  %8 = linalg.matmul ins(%3, %arg2 : tensor<512x256xf32>, tensor<256x2048xf32>) outs(%7 : tensor<512x2048xf32>) -> tensor<512x2048xf32>
  // CHECK: [[v17:%.*]] = bufferization.to_buffer [[v16]] : tensor<512x2048xf32> to memref<512x2048xf32>
  // CHECK: [[valloc_0:%.*]] = memref.alloc() : memref<512x2048xf32>
  // CHECK: linalg.copy ins([[v17]] : memref<512x2048xf32>) outs([[valloc_0]] : memref<512x2048xf32>)
  // CHECK: [[v18:%.*]] = mpi.comm_world : !mpi.comm
  // CHECK: mpi.allreduce([[valloc_0]], [[valloc_0]], MPI_SUM, [[v18]]) : memref<512x2048xf32>, memref<512x2048xf32>
  // CHECK: [[v19:%.*]] = bufferization.to_tensor [[valloc_0]] restrict : memref<512x2048xf32> to tensor<512x2048xf32>
  %all_reduce = shard.all_reduce %8 on @grid_1d_4 grid_axes = [0] : tensor<512x2048xf32> -> tensor<512x2048xf32>
  // CHECK: return [[v19]] : tensor<512x2048xf32>
  return %all_reduce : tensor<512x2048xf32>
}
