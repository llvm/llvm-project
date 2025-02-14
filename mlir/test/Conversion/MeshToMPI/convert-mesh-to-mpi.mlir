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
mesh.mesh @mesh0(shape = 3x4x5)
memref.global constant @static_mpi_rank : memref<index> = dense<24>
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
mesh.mesh @mesh0(shape = 3x4x5)
memref.global constant @static_mpi_rank : memref<index> = dense<24>
// CHECK-LABEL: func @update_halo_3d
func.func @update_halo_3d(
  // CHECK-SAME: [[varg0:%.*]]: memref<120x120x120xi8>
  %arg0 : memref<120x120x120xi8>) -> memref<120x120x120xi8> {
  // CHECK: [[vc23_i32:%.*]] = arith.constant 23 : i32
  // CHECK-NEXT: [[vc29_i32:%.*]] = arith.constant 29 : i32
  // CHECK-NEXT: [[vc91_i32:%.*]] = arith.constant 91 : i32
  // CHECK-NEXT: [[vc4_i32:%.*]] = arith.constant 4 : i32
  // CHECK-NEXT: [[vc44_i32:%.*]] = arith.constant 44 : i32
  // CHECK-NEXT: [[valloc:%.*]] = memref.alloc() : memref<117x113x5xi8>
  // CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[varg0]][1, 3, 109] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>>
  // CHECK-NEXT: memref.copy [[vsubview]], [[valloc]] : memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>> to memref<117x113x5xi8>
  // CHECK-NEXT: mpi.send([[valloc]], [[vc91_i32]], [[vc4_i32]]) : memref<117x113x5xi8>, i32, i32
  // CHECK-NEXT: mpi.recv([[valloc]], [[vc91_i32]], [[vc44_i32]]) : memref<117x113x5xi8>, i32, i32
  // CHECK-NEXT: [[vsubview_0:%.*]] = memref.subview [[varg0]][1, 3, 0] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
  // CHECK-NEXT: memref.copy [[valloc]], [[vsubview_0]] : memref<117x113x5xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
  // CHECK-NEXT: memref.dealloc [[valloc]] : memref<117x113x5xi8>
  // CHECK-NEXT: [[valloc_1:%.*]] = memref.alloc() : memref<117x113x6xi8>
  // CHECK-NEXT: [[vsubview_3:%.*]] = memref.subview [[varg0]][1, 3, 5] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>>
  // CHECK-NEXT: memref.copy [[vsubview_3]], [[valloc_1]] : memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>> to memref<117x113x6xi8>
  // CHECK-NEXT: mpi.send([[valloc_1]], [[vc91_i32]], [[vc44_i32]]) : memref<117x113x6xi8>, i32, i32
  // CHECK-NEXT: mpi.recv([[valloc_1]], [[vc91_i32]], [[vc4_i32]]) : memref<117x113x6xi8>, i32, i32
  // CHECK-NEXT: [[vsubview_4:%.*]] = memref.subview [[varg0]][1, 3, 114] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
  // CHECK-NEXT: memref.copy [[valloc_1]], [[vsubview_4]] : memref<117x113x6xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
  // CHECK-NEXT: memref.dealloc [[valloc_1]] : memref<117x113x6xi8>
  // CHECK-NEXT: [[valloc_5:%.*]] = memref.alloc() : memref<117x3x120xi8>
  // CHECK-NEXT: mpi.recv([[valloc_5]], [[vc91_i32]], [[vc29_i32]]) : memref<117x3x120xi8>, i32, i32
  // CHECK-NEXT: [[vsubview_7:%.*]] = memref.subview [[varg0]][1, 0, 0] [117, 3, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x3x120xi8, strided<[14400, 120, 1], offset: 14400>>
  // CHECK-NEXT: memref.copy [[valloc_5]], [[vsubview_7]] : memref<117x3x120xi8> to memref<117x3x120xi8, strided<[14400, 120, 1], offset: 14400>>
  // CHECK-NEXT: memref.dealloc [[valloc_5]] : memref<117x3x120xi8>
  // CHECK-NEXT: [[valloc_8:%.*]] = memref.alloc() : memref<117x4x120xi8>
  // CHECK-NEXT: [[vsubview_10:%.*]] = memref.subview [[varg0]][1, 3, 0] [117, 4, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 14760>>
  // CHECK-NEXT: memref.copy [[vsubview_10]], [[valloc_8]] : memref<117x4x120xi8, strided<[14400, 120, 1], offset: 14760>> to memref<117x4x120xi8>
  // CHECK-NEXT: mpi.send([[valloc_8]], [[vc91_i32]], [[vc29_i32]]) : memref<117x4x120xi8>, i32, i32
  // CHECK-NEXT: memref.dealloc [[valloc_8]] : memref<117x4x120xi8>
  // CHECK-NEXT: [[valloc_11:%.*]] = memref.alloc() : memref<1x120x120xi8>
  // CHECK-NEXT: [[vsubview_12:%.*]] = memref.subview [[varg0]][117, 0, 0] [1, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1], offset: 1684800>>
  // CHECK-NEXT: memref.copy [[vsubview_12]], [[valloc_11]] : memref<1x120x120xi8, strided<[14400, 120, 1], offset: 1684800>> to memref<1x120x120xi8>
  // CHECK-NEXT: mpi.send([[valloc_11]], [[vc91_i32]], [[vc23_i32]]) : memref<1x120x120xi8>, i32, i32
  // CHECK-NEXT: memref.dealloc [[valloc_11]] : memref<1x120x120xi8>
  // CHECK-NEXT: [[valloc_13:%.*]] = memref.alloc() : memref<2x120x120xi8>
  // CHECK-NEXT: mpi.recv([[valloc_13]], [[vc91_i32]], [[vc23_i32]]) : memref<2x120x120xi8>, i32, i32
  // CHECK-NEXT: [[vsubview_14:%.*]] = memref.subview [[varg0]][118, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 1699200>>
  // CHECK-NEXT: memref.copy [[valloc_13]], [[vsubview_14]] : memref<2x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 1699200>>
  // CHECK-NEXT: memref.dealloc [[valloc_13]] : memref<2x120x120xi8>
  %res = mesh.update_halo %arg0 on @mesh0 split_axes = [[2], [1], [0]] halo_sizes = [1, 2, 3, 4, 5, 6] : memref<120x120x120xi8>
  // CHECK: return [[varg0]] : memref<120x120x120xi8>
  return %res : memref<120x120x120xi8>
}

// CHECK-LABEL: func @update_halo_3d_tensor
func.func @update_halo_3d_tensor(
  // CHECK-SAME: [[varg0:%.*]]: tensor<120x120x120xi8>
  %arg0 : tensor<120x120x120xi8>) -> tensor<120x120x120xi8> {
  // CHECK: [[vc23_i32:%.*]] = arith.constant 23 : i32
  // CHECK-NEXT: [[vc29_i32:%.*]] = arith.constant 29 : i32
  // CHECK-NEXT: [[vc44_i32:%.*]] = arith.constant 44 : i32
  // CHECK-NEXT: [[vc4_i32:%.*]] = arith.constant 4 : i32
  // CHECK-NEXT: [[vc91_i32:%.*]] = arith.constant 91 : i32
  // CHECK-NEXT: [[v0:%.*]] = bufferization.to_memref [[varg0]] : tensor<120x120x120xi8> to memref<120x120x120xi8>
  // CHECK-NEXT: [[valloc:%.*]] = memref.alloc() : memref<117x113x5xi8>
  // CHECK-NEXT: [[vsubview:%.*]] = memref.subview [[v0]][1, 3, 109] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>>
  // CHECK-NEXT: memref.copy [[vsubview]], [[valloc]] : memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14869>> to memref<117x113x5xi8>
  // CHECK-NEXT: mpi.send([[valloc]], [[vc91_i32]], [[vc4_i32]]) : memref<117x113x5xi8>, i32, i32
  // CHECK-NEXT: mpi.recv([[valloc]], [[vc91_i32]], [[vc44_i32]]) : memref<117x113x5xi8>, i32, i32
  // CHECK-NEXT: [[vsubview_0:%.*]] = memref.subview [[v0]][1, 3, 0] [117, 113, 5] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
  // CHECK-NEXT: memref.copy [[valloc]], [[vsubview_0]] : memref<117x113x5xi8> to memref<117x113x5xi8, strided<[14400, 120, 1], offset: 14760>>
  // CHECK-NEXT: memref.dealloc [[valloc]] : memref<117x113x5xi8>
  // CHECK-NEXT: [[valloc_1:%.*]] = memref.alloc() : memref<117x113x6xi8>
  // CHECK-NEXT: [[vsubview_3:%.*]] = memref.subview [[v0]][1, 3, 5] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>>
  // CHECK-NEXT: memref.copy [[vsubview_3]], [[valloc_1]] : memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14765>> to memref<117x113x6xi8>
  // CHECK-NEXT: mpi.send([[valloc_1]], [[vc91_i32]], [[vc44_i32]]) : memref<117x113x6xi8>, i32, i32
  // CHECK-NEXT: mpi.recv([[valloc_1]], [[vc91_i32]], [[vc4_i32]]) : memref<117x113x6xi8>, i32, i32
  // CHECK-NEXT: [[vsubview_4:%.*]] = memref.subview [[v0]][1, 3, 114] [117, 113, 6] [1, 1, 1] : memref<120x120x120xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
  // CHECK-NEXT: memref.copy [[valloc_1]], [[vsubview_4]] : memref<117x113x6xi8> to memref<117x113x6xi8, strided<[14400, 120, 1], offset: 14874>>
  // CHECK-NEXT: memref.dealloc [[valloc_1]] : memref<117x113x6xi8>
  // CHECK-NEXT: [[valloc_5:%.*]] = memref.alloc() : memref<117x3x120xi8>
  // CHECK-NEXT: mpi.recv([[valloc_5]], [[vc91_i32]], [[vc29_i32]]) : memref<117x3x120xi8>, i32, i32
  // CHECK-NEXT: [[vsubview_7:%.*]] = memref.subview [[v0]][1, 0, 0] [117, 3, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x3x120xi8, strided<[14400, 120, 1], offset: 14400>>
  // CHECK-NEXT: memref.copy [[valloc_5]], [[vsubview_7]] : memref<117x3x120xi8> to memref<117x3x120xi8, strided<[14400, 120, 1], offset: 14400>>
  // CHECK-NEXT: memref.dealloc [[valloc_5]] : memref<117x3x120xi8>
  // CHECK-NEXT: [[valloc_8:%.*]] = memref.alloc() : memref<117x4x120xi8>
  // CHECK-NEXT: [[vsubview_10:%.*]] = memref.subview [[v0]][1, 3, 0] [117, 4, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<117x4x120xi8, strided<[14400, 120, 1], offset: 14760>>
  // CHECK-NEXT: memref.copy [[vsubview_10]], [[valloc_8]] : memref<117x4x120xi8, strided<[14400, 120, 1], offset: 14760>> to memref<117x4x120xi8>
  // CHECK-NEXT: mpi.send([[valloc_8]], [[vc91_i32]], [[vc29_i32]]) : memref<117x4x120xi8>, i32, i32
  // CHECK-NEXT: memref.dealloc [[valloc_8]] : memref<117x4x120xi8>
  // CHECK-NEXT: [[valloc_11:%.*]] = memref.alloc() : memref<1x120x120xi8>
  // CHECK-NEXT: [[vsubview_12:%.*]] = memref.subview [[v0]][117, 0, 0] [1, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<1x120x120xi8, strided<[14400, 120, 1], offset: 1684800>>
  // CHECK-NEXT: memref.copy [[vsubview_12]], [[valloc_11]] : memref<1x120x120xi8, strided<[14400, 120, 1], offset: 1684800>> to memref<1x120x120xi8>
  // CHECK-NEXT: mpi.send([[valloc_11]], [[vc91_i32]], [[vc23_i32]]) : memref<1x120x120xi8>, i32, i32
  // CHECK-NEXT: memref.dealloc [[valloc_11]] : memref<1x120x120xi8>
  // CHECK-NEXT: [[valloc_13:%.*]] = memref.alloc() : memref<2x120x120xi8>
  // CHECK-NEXT: mpi.recv([[valloc_13]], [[vc91_i32]], [[vc23_i32]]) : memref<2x120x120xi8>, i32, i32
  // CHECK-NEXT: [[vsubview_14:%.*]] = memref.subview [[v0]][118, 0, 0] [2, 120, 120] [1, 1, 1] : memref<120x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 1699200>>
  // CHECK-NEXT: memref.copy [[valloc_13]], [[vsubview_14]] : memref<2x120x120xi8> to memref<2x120x120xi8, strided<[14400, 120, 1], offset: 1699200>>
  // CHECK-NEXT: memref.dealloc [[valloc_13]] : memref<2x120x120xi8>
  // CHECK-NEXT: [[v1:%.*]] = bufferization.to_tensor [[v0]] restrict writable : memref<120x120x120xi8>
  %res = mesh.update_halo %arg0 on @mesh0 split_axes = [[2], [1], [0]] halo_sizes = [1, 2, 3, 4, 5, 6] : tensor<120x120x120xi8>
  // CHECK: return [[v1]] : tensor<120x120x120xi8>
  return %res : tensor<120x120x120xi8>
}
