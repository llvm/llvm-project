// RUN: mlir-opt %s -convert-mesh-to-mpi -canonicalize -split-input-file | FileCheck %s

// -----
// CHECK: mesh.mesh @mesh0
mesh.mesh @mesh0(shape = 3x4x5)
func.func @process_multi_index() -> (index, index, index) {
  // CHECK: mpi.comm_rank : !mpi.retval, i32
  // CHECK-DAG: %[[v4:.*]] = arith.remsi
  // CHECK-DAG: %[[v0:.*]] = arith.remsi
  // CHECK-DAG: %[[v1:.*]] = arith.remsi
  %0:3 = mesh.process_multi_index on @mesh0 axes = [] : index, index, index
  // CHECK: return %[[v1]], %[[v0]], %[[v4]] : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// CHECK-LABEL: func @process_linear_index
func.func @process_linear_index() -> index {
  // CHECK: %[[RES:.*]], %[[rank:.*]] = mpi.comm_rank : !mpi.retval, i32
  // CHECK: %[[cast:.*]] = arith.index_cast %[[rank]] : i32 to index
  %0 = mesh.process_linear_index on @mesh0 : index
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
  %idx:2 = mesh.neighbors_linear_indices on @mesh0[%c1, %c0, %c4] split_axes = [0] : index, index
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
  %idx:2 = mesh.neighbors_linear_indices on @mesh0[%c1, %c0, %c4] split_axes = [1] : index, index
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
  %idx:2 = mesh.neighbors_linear_indices on @mesh0[%c1, %c0, %c4] split_axes = [2] : index, index
  // CHECK: return [[down]], [[up]] : index, index
  return %idx#0, %idx#1 : index, index
}

// -----
// CHECK: mesh.mesh @mesh0
module attributes { mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 24> } {
  mesh.mesh @mesh0(shape = 3x4x5)
  func.func @process_multi_index() -> (index, index, index) {
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    %0:3 = mesh.process_multi_index on @mesh0 axes = [] : index, index, index
    // CHECK: return %[[c1]], %[[c0]], %[[c4]] : index, index, index
    return %0#0, %0#1, %0#2 : index, index, index
  }

  // CHECK-LABEL: func @process_linear_index
  func.func @process_linear_index() -> index {
    // CHECK: %[[c24:.*]] = arith.constant 24 : index
    %0 = mesh.process_linear_index on @mesh0 : index
    // CHECK: return %[[c24]] : index
    return %0 : index
  }
}

// -----
mesh.mesh @mesh0(shape = 3x4x5)
// CHECK-LABEL: func @update_halo_1d_first
func.func @update_halo_1d_first(
  // CHECK-SAME: [[arg0:%.*]]: memref<120x120x120xi8>
  %arg0 : memref<120x120x120xi8>) -> memref<120x120x120xi8> {
  // CHECK: memref.subview [[arg0]][115, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8
  // CHECK: mpi.send(
  // CHECK-SAME: : memref<2x120x120xi8>, i32, i32
  // CHECK: mpi.recv(
  // CHECK-SAME: : memref<2x120x120xi8>, i32, i32
  // CHECK-NEXT: memref.subview [[arg0]][0, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8
  // CHECK: memref.subview [[arg0]][2, 0, 0] [3, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<3x120x120xi8
  // CHECK: mpi.send(
  // CHECK-SAME: : memref<3x120x120xi8>, i32, i32
  // CHECK: mpi.recv(
  // CHECK-SAME: : memref<3x120x120xi8>, i32, i32
  // CHECK-NEXT: memref.subview [[arg0]][117, 0, 0] [3, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<3x120x120xi8
  %res = mesh.update_halo %arg0 on @mesh0 split_axes = [[0]] halo_sizes = [2, 3] : memref<120x120x120xi8>
  // CHECK: return [[res:%.*]] : memref<120x120x120xi8>
  return %res : memref<120x120x120xi8>
}

// -----
module attributes { mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 1> } {
  mesh.mesh @mesh0(shape = 4)
  // CHECK-LABEL: func @update_halo_1d_with_zero
  func.func @update_halo_1d_with_zero (
    // CHECK-SAME: [[varg0:%.*]]: memref<120x120x120xi8>
    %arg0 : memref<120x120x120xi8>) -> memref<120x120x120xi8> {
    // CHECK: [[vc91_i32:%.*]] = arith.constant 91 : i32
    // CHECK-NEXT: [[vc0_i32:%.*]] = arith.constant 0 : i32
    // CHECK-NEXT: [[vc2_i32:%.*]] = arith.constant 2 : i32
    // CHECK-NEXT: [[valloc:%.*]] = memref.alloc() : memref<2x120x120xi8>
    // CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[varg0]][118, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8
    // CHECK-NEXT: memref.copy [[vsubview]], [[valloc]] : memref<2x120x120xi8
    // CHECK-SAME: to memref<2x120x120xi8>
    // CHECK-NEXT: mpi.send([[valloc]], [[vc91_i32]], [[vc2_i32]]) : memref<2x120x120xi8>, i32, i32
    // CHECK-NEXT: mpi.recv([[valloc]], [[vc91_i32]], [[vc0_i32]]) : memref<2x120x120xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_0:%.*]] = memref.subview [[varg0]][0, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8
    // CHECK-NEXT: memref.copy [[valloc]], [[vsubview_0]] : memref<2x120x120xi8> to memref<2x120x120xi8
    // CHECK-NEXT: memref.dealloc [[valloc]] : memref<2x120x120xi8>
    %res = mesh.update_halo %arg0 on @mesh0 split_axes = [[0]] halo_sizes = [2, 0] : memref<120x120x120xi8>
    // CHECK: return [[res:%.*]] : memref<120x120x120xi8>
    return %res : memref<120x120x120xi8>
  }
}

// -----
module attributes { mpi.dlti = #dlti.map<"MPI:comm_world_rank" = 24> } {
  mesh.mesh @mesh0(shape = 3x4x5)
  // CHECK-LABEL: func @update_halo_3d
  func.func @update_halo_3d(
    // CHECK-SAME: [[varg0:%.*]]: memref<120x120x120xi8>
    %arg0 : memref<120x120x120xi8>) -> memref<120x120x120xi8> {
    // CHECK-NEXT: [[vc23_i32:%.*]] = arith.constant 23 : i32
    // CHECK-NEXT: [[vc29_i32:%.*]] = arith.constant 29 : i32
    // CHECK-NEXT: [[vc91_i32:%.*]] = arith.constant 91 : i32
    // CHECK-NEXT: [[vc4_i32:%.*]] = arith.constant 4 : i32
    // CHECK-NEXT: [[vc44_i32:%.*]] = arith.constant 44 : i32
    // CHECK-NEXT: [[valloc:%.*]] = memref.alloc() : memref<117x113x5xi8>
    // CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[varg0]][1, 3, 109] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>>
    // CHECK-NEXT: memref.copy [[vsubview]], [[valloc]] : memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>> to memref<117x113x5xi8>
    // CHECK-NEXT: mpi.send([[valloc]], [[vc91_i32]], [[vc44_i32]]) : memref<117x113x5xi8>, i32, i32
    // CHECK-NEXT: mpi.recv([[valloc]], [[vc91_i32]], [[vc4_i32]]) : memref<117x113x5xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_0:%.*]] = memref.subview [[varg0]][1, 3, 0] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
    // CHECK-NEXT: memref.copy [[valloc]], [[vsubview_0]] : memref<117x113x5xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
    // CHECK-NEXT: memref.dealloc [[valloc]] : memref<117x113x5xi8>
    // CHECK-NEXT: [[valloc_1:%.*]] = memref.alloc() : memref<117x113x6xi8>
    // CHECK-NEXT: [[vsubview_2:%.*]] = memref.subview [[varg0]][1, 3, 5] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>>
    // CHECK-NEXT: memref.copy [[vsubview_2]], [[valloc_1]] : memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>> to memref<117x113x6xi8>
    // CHECK-NEXT: mpi.send([[valloc_1]], [[vc91_i32]], [[vc4_i32]]) : memref<117x113x6xi8>, i32, i32
    // CHECK-NEXT: mpi.recv([[valloc_1]], [[vc91_i32]], [[vc44_i32]]) : memref<117x113x6xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_3:%.*]] = memref.subview [[varg0]][1, 3, 114] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
    // CHECK-NEXT: memref.copy [[valloc_1]], [[vsubview_3]] : memref<117x113x6xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
    // CHECK-NEXT: memref.dealloc [[valloc_1]] : memref<117x113x6xi8>
    // CHECK-NEXT: [[valloc_4:%.*]] = memref.alloc() : memref<117x3x120xi8>
    // CHECK-NEXT: [[vsubview_5:%.*]] = memref.subview [[varg0]][1, 113, 0] [117, 3, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x3x120xi8, strided<[14400, 120, 1], offset: 27960>>
    // CHECK-NEXT: memref.copy [[vsubview_5]], [[valloc_4]] : memref<117x3x120xi8, strided<[14400, 120, 1], offset: 27960>> to memref<117x3x120xi8>
    // CHECK-NEXT: mpi.send([[valloc_4]], [[vc91_i32]], [[vc29_i32]]) : memref<117x3x120xi8>, i32, i32
    // CHECK-NEXT: memref.dealloc [[valloc_4]] : memref<117x3x120xi8>
    // CHECK-NEXT: [[valloc_6:%.*]] = memref.alloc() : memref<117x4x120xi8>
    // CHECK-NEXT: mpi.recv([[valloc_6]], [[vc91_i32]], [[vc29_i32]]) : memref<117x4x120xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_7:%.*]] = memref.subview [[varg0]][1, 116, 0] [117, 4, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 28320>>
    // CHECK-NEXT: memref.copy [[valloc_6]], [[vsubview_7]] : memref<117x4x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 28320>>
    // CHECK-NEXT: memref.dealloc [[valloc_6]] : memref<117x4x120xi8>
    // CHECK-NEXT: [[valloc_8:%.*]] = memref.alloc() : memref<1x120x120xi8>
    // CHECK-NEXT: mpi.recv([[valloc_8]], [[vc91_i32]], [[vc23_i32]]) : memref<1x120x120xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_9:%.*]] = memref.subview [[varg0]][0, 0, 0] [1, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK-NEXT: memref.copy [[valloc_8]], [[vsubview_9]] : memref<1x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK-NEXT: memref.dealloc [[valloc_8]] : memref<1x120x120xi8>
    // CHECK-NEXT: [[valloc_10:%.*]] = memref.alloc() : memref<2x120x120xi8>
    // CHECK-NEXT: [[vsubview_11:%.*]] = memref.subview [[varg0]][1, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 14400>>
    // CHECK-NEXT: memref.copy [[vsubview_11]], [[valloc_10]] : memref<2x120x120xi8, strided<[14400, 120, 1], offset: 14400>> to memref<2x120x120xi8>
    // CHECK-NEXT: mpi.send([[valloc_10]], [[vc91_i32]], [[vc23_i32]]) : memref<2x120x120xi8>, i32, i32
    // CHECK-NEXT: memref.dealloc [[valloc_10]] : memref<2x120x120xi8>
    %res = mesh.update_halo %arg0 on @mesh0 split_axes = [[2], [1], [0]] halo_sizes = [1, 2, 3, 4, 5, 6] : memref<120x120x120xi8>
    // CHECK: return [[varg0]] : memref<120x120x120xi8>
    return %res : memref<120x120x120xi8>
  }

  // CHECK-LABEL: func @update_halo_3d_tensor
  func.func @update_halo_3d_tensor(
    // CHECK-SAME: [[varg0:%.*]]: tensor<120x120x120xi8>
    %arg0 : tensor<120x120x120xi8>) -> tensor<120x120x120xi8> {
    // CHECK-NEXT: [[vc23_i32:%.*]] = arith.constant 23 : i32
    // CHECK-NEXT: [[vc29_i32:%.*]] = arith.constant 29 : i32
    // CHECK-NEXT: [[vc44_i32:%.*]] = arith.constant 44 : i32
    // CHECK-NEXT: [[vc4_i32:%.*]] = arith.constant 4 : i32
    // CHECK-NEXT: [[vc91_i32:%.*]] = arith.constant 91 : i32
    // CHECK-NEXT: [[v0:%.*]] = bufferization.to_memref [[varg0]] : tensor<120x120x120xi8> to memref<120x120x120xi8>
    // CHECK-NEXT: [[valloc:%.*]] = memref.alloc() : memref<117x113x5xi8>
    // CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[v0]][1, 3, 109] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>>
    // CHECK-NEXT: memref.copy [[vsubview]], [[valloc]] : memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>> to memref<117x113x5xi8>
    // CHECK-NEXT: mpi.send([[valloc]], [[vc91_i32]], [[vc44_i32]]) : memref<117x113x5xi8>, i32, i32
    // CHECK-NEXT: mpi.recv([[valloc]], [[vc91_i32]], [[vc4_i32]]) : memref<117x113x5xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_0:%.*]] = memref.subview [[v0]][1, 3, 0] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
    // CHECK-NEXT: memref.copy [[valloc]], [[vsubview_0]] : memref<117x113x5xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
    // CHECK-NEXT: memref.dealloc [[valloc]] : memref<117x113x5xi8>
    // CHECK-NEXT: [[valloc_1:%.*]] = memref.alloc() : memref<117x113x6xi8>
    // CHECK-NEXT: [[vsubview_2:%.*]] = memref.subview [[v0]][1, 3, 5] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>>
    // CHECK-NEXT: memref.copy [[vsubview_2]], [[valloc_1]] : memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>> to memref<117x113x6xi8>
    // CHECK-NEXT: mpi.send([[valloc_1]], [[vc91_i32]], [[vc4_i32]]) : memref<117x113x6xi8>, i32, i32
    // CHECK-NEXT: mpi.recv([[valloc_1]], [[vc91_i32]], [[vc44_i32]]) : memref<117x113x6xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_3:%.*]] = memref.subview [[v0]][1, 3, 114] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
    // CHECK-NEXT: memref.copy [[valloc_1]], [[vsubview_3]] : memref<117x113x6xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
    // CHECK-NEXT: memref.dealloc [[valloc_1]] : memref<117x113x6xi8>
    // CHECK-NEXT: [[valloc_4:%.*]] = memref.alloc() : memref<117x3x120xi8>
    // CHECK-NEXT: [[vsubview_5:%.*]] = memref.subview [[v0]][1, 113, 0] [117, 3, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x3x120xi8, strided<[14400, 120, 1], offset: 27960>>
    // CHECK-NEXT: memref.copy [[vsubview_5]], [[valloc_4]] : memref<117x3x120xi8, strided<[14400, 120, 1], offset: 27960>> to memref<117x3x120xi8>
    // CHECK-NEXT: mpi.send([[valloc_4]], [[vc91_i32]], [[vc29_i32]]) : memref<117x3x120xi8>, i32, i32
    // CHECK-NEXT: memref.dealloc [[valloc_4]] : memref<117x3x120xi8>
    // CHECK-NEXT: [[valloc_6:%.*]] = memref.alloc() : memref<117x4x120xi8>
    // CHECK-NEXT: mpi.recv([[valloc_6]], [[vc91_i32]], [[vc29_i32]]) : memref<117x4x120xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_7:%.*]] = memref.subview [[v0]][1, 116, 0] [117, 4, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 28320>>
    // CHECK-NEXT: memref.copy [[valloc_6]], [[vsubview_7]] : memref<117x4x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 28320>>
    // CHECK-NEXT: memref.dealloc [[valloc_6]] : memref<117x4x120xi8>
    // CHECK-NEXT: [[valloc_8:%.*]] = memref.alloc() : memref<1x120x120xi8>
    // CHECK-NEXT: mpi.recv([[valloc_8]], [[vc91_i32]], [[vc23_i32]]) : memref<1x120x120xi8>, i32, i32
    // CHECK-NEXT: [[vsubview_9:%.*]] = memref.subview [[v0]][0, 0, 0] [1, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK-NEXT: memref.copy [[valloc_8]], [[vsubview_9]] : memref<1x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1]>>
    // CHECK-NEXT: memref.dealloc [[valloc_8]] : memref<1x120x120xi8>
    // CHECK-NEXT: [[valloc_10:%.*]] = memref.alloc() : memref<2x120x120xi8>
    // CHECK-NEXT: [[vsubview_11:%.*]] = memref.subview [[v0]][1, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 14400>>
    // CHECK-NEXT: memref.copy [[vsubview_11]], [[valloc_10]] : memref<2x120x120xi8, strided<[14400, 120, 1], offset: 14400>> to memref<2x120x120xi8>
    // CHECK-NEXT: mpi.send([[valloc_10]], [[vc91_i32]], [[vc23_i32]]) : memref<2x120x120xi8>, i32, i32
    // CHECK-NEXT: memref.dealloc [[valloc_10]] : memref<2x120x120xi8>
    // CHECK-NEXT: [[v1:%.*]] = bufferization.to_tensor [[v0]] restrict writable : memref<120x120x120xi8> to tensor<120x120x120xi8>
    %res = mesh.update_halo %arg0 on @mesh0 split_axes = [[2], [1], [0]] halo_sizes = [1, 2, 3, 4, 5, 6] : tensor<120x120x120xi8>
    // CHECK: return [[v1]] : tensor<120x120x120xi8>
    return %res : tensor<120x120x120xi8>
  }
}

// -----
mesh.mesh @mesh0(shape = 2x2x4)
// CHECK-LABEL: func.func @return_sharding(
// CHECK-SAME: [[varg0:%.*]]: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>) {
func.func @return_sharding(%arg0: tensor<2x4xf32>) -> (tensor<2x4xf32>, !mesh.sharding) {
  %sharding = mesh.sharding @mesh0 split_axes = [[0, 1], [2]] : !mesh.sharding
  // CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<2> : tensor<1xi16>
  // CHECK-NEXT: [[vcst_0:%.*]] = arith.constant dense<[0, 1]> : tensor<2xi16>
  // CHECK-NEXT: [[vcm1_i16:%.*]] = arith.constant -1 : i16
  // CHECK-NEXT: [[v0:%.*]] = tensor.empty() : tensor<2x2xi16>
  // CHECK-NEXT: [[v1:%.*]] = linalg.fill ins([[vcm1_i16]] : i16) outs([[v0]] : tensor<2x2xi16>) -> tensor<2x2xi16>
  // CHECK-NEXT: [[vinserted_slice:%.*]] = tensor.insert_slice [[vcst_0]] into [[v1]][0, 0] [1, 2] [1, 1] : tensor<2xi16> into tensor<2x2xi16>
  // CHECK-NEXT: [[vinserted_slice_1:%.*]] = tensor.insert_slice [[vcst]] into [[vinserted_slice]][1, 0] [1, 1] [1, 1] : tensor<1xi16> into tensor<2x2xi16>
  // CHECK-NEXT: [[v2:%.*]] = tensor.empty() : tensor<0x0xi64>
  // CHECK-NEXT: [[v3:%.*]] = tensor.empty() : tensor<0x0xi64>
  // CHECK-NEXT: [[vcast:%.*]] = tensor.cast [[vinserted_slice_1]] : tensor<2x2xi16> to tensor<?x?xi16>
  // CHECK-NEXT: [[vcast_2:%.*]] = tensor.cast [[v2]] : tensor<0x0xi64> to tensor<?x?xi64>
  // CHECK-NEXT: [[vcast_3:%.*]] = tensor.cast [[v3]] : tensor<0x0xi64> to tensor<?x?xi64>
  // CHECK-NEXT: return [[varg0]], [[vcast]], [[vcast_2]], [[vcast_3]] : tensor<2x4xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>
  return %arg0, %sharding : tensor<2x4xf32>, !mesh.sharding
}

// CHECK-LABEL: func.func @return_sharding_halos(
// CHECK-SAME: [[varg0:%.*]]: tensor<6x8xf32>) -> (tensor<6x8xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>) {
func.func @return_sharding_halos(%arg0: tensor<6x8xf32>) -> (tensor<6x8xf32>, !mesh.sharding) {
  %sharding = mesh.sharding @mesh0 split_axes = [[0, 1], [2]] halo_sizes = [0, 4, 3, 1] : !mesh.sharding
  // CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<{{\[\[}}0, 4], [3, 1]]> : tensor<2x2xi64>
  // CHECK-NEXT: [[vcst_0:%.*]] = arith.constant dense<2> : tensor<1xi16>
  // CHECK-NEXT: [[vcst_1:%.*]] = arith.constant dense<[0, 1]> : tensor<2xi16>
  // CHECK-NEXT: [[vcm1_i16:%.*]] = arith.constant -1 : i16
  // CHECK-NEXT: [[v0:%.*]] = tensor.empty() : tensor<2x2xi16>
  // CHECK-NEXT: [[v1:%.*]] = linalg.fill ins([[vcm1_i16]] : i16) outs([[v0]] : tensor<2x2xi16>) -> tensor<2x2xi16>
  // CHECK-NEXT: [[vinserted_slice:%.*]] = tensor.insert_slice [[vcst_1]] into [[v1]][0, 0] [1, 2] [1, 1] : tensor<2xi16> into tensor<2x2xi16>
  // CHECK-NEXT: [[vinserted_slice_2:%.*]] = tensor.insert_slice [[vcst_0]] into [[vinserted_slice]][1, 0] [1, 1] [1, 1] : tensor<1xi16> into tensor<2x2xi16>
  // CHECK-NEXT: [[v2:%.*]] = tensor.empty() : tensor<0x0xi64>
  // CHECK-NEXT: [[vcast:%.*]] = tensor.cast [[vinserted_slice_2]] : tensor<2x2xi16> to tensor<?x?xi16>
  // CHECK-NEXT: [[vcast_3:%.*]] = tensor.cast [[vcst]] : tensor<2x2xi64> to tensor<?x?xi64>
  // CHECK-NEXT: [[vcast_4:%.*]] = tensor.cast [[v2]] : tensor<0x0xi64> to tensor<?x?xi64>
  // CHECK-NEXT: return [[varg0]], [[vcast]], [[vcast_3]], [[vcast_4]] : tensor<6x8xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>
  return %arg0, %sharding : tensor<6x8xf32>, !mesh.sharding
}

// CHECK-LABEL: func.func @return_sharding_offs(
// CHECK-SAME: [[varg0:%.*]]: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>) {
func.func @return_sharding_offs(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>, !mesh.sharding) {
  %sharding = mesh.sharding @mesh0 split_axes = [[0, 1], [2]] sharded_dims_offsets = [0, 3, 5, 7, 8, 0, 0, 5, 10, 16] : !mesh.sharding
  // CHECK-NEXT: [[vcst:%.*]] = arith.constant dense<[0, 0, 5, 10, 16]> : tensor<5xi64>
  // CHECK-NEXT: [[vcst_0:%.*]] = arith.constant dense<[0, 3, 5, 7, 8]> : tensor<5xi64>
  // CHECK-NEXT: [[vcm9223372036854775808_i64:%.*]] = arith.constant -9223372036854775808 : i64
  // CHECK-NEXT: [[vcst_1:%.*]] = arith.constant dense<2> : tensor<1xi16>
  // CHECK-NEXT: [[vcst_2:%.*]] = arith.constant dense<[0, 1]> : tensor<2xi16>
  // CHECK-NEXT: [[vcm1_i16:%.*]] = arith.constant -1 : i16
  // CHECK-NEXT: [[v0:%.*]] = tensor.empty() : tensor<2x2xi16>
  // CHECK-NEXT: [[v1:%.*]] = linalg.fill ins([[vcm1_i16]] : i16) outs([[v0]] : tensor<2x2xi16>) -> tensor<2x2xi16>
  // CHECK-NEXT: [[vinserted_slice:%.*]] = tensor.insert_slice [[vcst_2]] into [[v1]][0, 0] [1, 2] [1, 1] : tensor<2xi16> into tensor<2x2xi16>
  // CHECK-NEXT: [[vinserted_slice_3:%.*]] = tensor.insert_slice [[vcst_1]] into [[vinserted_slice]][1, 0] [1, 1] [1, 1] : tensor<1xi16> into tensor<2x2xi16>
  // CHECK-NEXT: [[v2:%.*]] = tensor.empty() : tensor<0x0xi64>
  // CHECK-NEXT: [[v3:%.*]] = tensor.empty() : tensor<2x5xi64>
  // CHECK-NEXT: [[v4:%.*]] = linalg.fill ins([[vcm9223372036854775808_i64]] : i64) outs([[v3]] : tensor<2x5xi64>) -> tensor<2x5xi64>
  // CHECK-NEXT: [[vinserted_slice_4:%.*]] = tensor.insert_slice [[vcst_0]] into [[v4]][0, 0] [1, 5] [1, 1] : tensor<5xi64> into tensor<2x5xi64>
  // CHECK-NEXT: [[vinserted_slice_5:%.*]] = tensor.insert_slice [[vcst]] into [[vinserted_slice_4]][1, 0] [1, 5] [1, 1] : tensor<5xi64> into tensor<2x5xi64>
  // CHECK-NEXT: [[vcast:%.*]] = tensor.cast [[vinserted_slice_3]] : tensor<2x2xi16> to tensor<?x?xi16>
  // CHECK-NEXT: [[vcast_6:%.*]] = tensor.cast [[v2]] : tensor<0x0xi64> to tensor<?x?xi64>
  // CHECK-NEXT: [[vcast_7:%.*]] = tensor.cast [[vinserted_slice_5]] : tensor<2x5xi64> to tensor<?x?xi64>
  // CHECK-NEXT: return [[varg0]], [[vcast]], [[vcast_6]], [[vcast_7]] : tensor<?x?xf32>, tensor<?x?xi16>, tensor<?x?xi64>, tensor<?x?xi64>
  return %arg0, %sharding : tensor<?x?xf32>, !mesh.sharding
}
