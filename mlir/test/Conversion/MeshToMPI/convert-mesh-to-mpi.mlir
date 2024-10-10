// RUN: mlir-opt %s -convert-mesh-to-mpi | FileCheck %s

// CHECK: mesh.mesh @mesh0
mesh.mesh @mesh0(shape = 2x2x4)

// CHECK-LABEL: func @update_halo_1d_first
func.func @update_halo_1d_first(
  // CHECK-SAME: [[varg0:%.*]]: memref<12x12xi8>
  %arg0 : memref<12x12xi8>) {
  // CHECK-NEXT: [[vc7:%.*]] = arith.constant 7 : index
  // CHECK-NEXT: [[vc9:%.*]] = arith.constant 9 : index
  // CHECK-NEXT: [[vc2:%.*]] = arith.constant 2 : index
  // CHECK-NEXT: [[vc91_i32:%.*]] = arith.constant 91 : i32
  // CHECK-NEXT: [[vc0_i32:%.*]] = arith.constant 0 : i32
  // CHECK-NEXT: [[vproc_linear_idx:%.*]]:3 = mesh.process_multi_index on @mesh0 : index, index, index
  // CHECK-NEXT: [[vdown_linear_idx:%.*]], [[vup_linear_idx:%.*]] = mesh.neighbors_linear_indices on @mesh0[[[vproc_linear_idx]]#0, [[vproc_linear_idx]]#1, [[vproc_linear_idx]]#2] split_axes = [0] : index, index
  // CHECK-NEXT: [[v0:%.*]] = arith.index_cast [[vdown_linear_idx]] : index to i32
  // CHECK-NEXT: [[v1:%.*]] = arith.index_cast [[vup_linear_idx]] : index to i32
  // CHECK-NEXT: [[v2:%.*]] = arith.cmpi sge, [[v1]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[v3:%.*]] = arith.cmpi sge, [[v0]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[valloc:%.*]] = memref.alloc() : memref<2x12xi8>
  // CHECK-NEXT: scf.if [[v3]] {
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][[[vc7]], 0] [2, 12] [1, 1] : memref<12x12xi8> to memref<2x12xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[vsubview]], [[valloc]] : memref<2x12xi8, strided<[12, 1], offset: ?>> to memref<2x12xi8>
  // CHECK-NEXT:   mpi.send([[valloc]], [[vc91_i32]], [[v0]]) : memref<2x12xi8>, i32, i32
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.if [[v2]] {
  // CHECK-NEXT:   mpi.recv([[valloc]], [[vc91_i32]], [[v1]]) : memref<2x12xi8>, i32, i32
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][0, 0] [2, 12] [1, 1] : memref<12x12xi8> to memref<2x12xi8, strided<[12, 1]>>
  // CHECK-NEXT:   memref.copy [[valloc]], [[vsubview]] : memref<2x12xi8> to memref<2x12xi8, strided<[12, 1]>>
  // CHECK-NEXT: }
  // CHECK-NEXT: memref.dealloc [[valloc]] : memref<2x12xi8>
  // CHECK-NEXT: [[v4:%.*]] = arith.cmpi sge, [[v0]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[v5:%.*]] = arith.cmpi sge, [[v1]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[valloc_0:%.*]] = memref.alloc() : memref<3x12xi8>
  // CHECK-NEXT: scf.if [[v5]] {
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][[[vc2]], 0] [3, 12] [1, 1] : memref<12x12xi8> to memref<3x12xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[vsubview]], [[valloc_0]] : memref<3x12xi8, strided<[12, 1], offset: ?>> to memref<3x12xi8>
  // CHECK-NEXT:   mpi.send([[valloc_0]], [[vc91_i32]], [[v1]]) : memref<3x12xi8>, i32, i32
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.if [[v4]] {
  // CHECK-NEXT:   mpi.recv([[valloc_0]], [[vc91_i32]], [[v0]]) : memref<3x12xi8>, i32, i32
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][[[vc9]], 0] [3, 12] [1, 1] : memref<12x12xi8> to memref<3x12xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[valloc_0]], [[vsubview]] : memref<3x12xi8> to memref<3x12xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT: }
  // CHECK-NEXT: memref.dealloc [[valloc_0]] : memref<3x12xi8>
  mesh.update_halo %arg0 on @mesh0 split_axes = [[0]]
    halo_sizes = [2, 3] : memref<12x12xi8>
  // CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @update_halo_1d_second
func.func @update_halo_1d_second(
  // CHECK-SAME: [[varg0:%.*]]: memref<12x12xi8>
  %arg0 : memref<12x12xi8>) {
  //CHECK-NEXT: [[vc7:%.*]] = arith.constant 7 : index
  //CHECK-NEXT: [[vc9:%.*]] = arith.constant 9 : index
  //CHECK-NEXT: [[vc2:%.*]] = arith.constant 2 : index
  //CHECK-NEXT: [[vc91_i32:%.*]] = arith.constant 91 : i32
  //CHECK-NEXT: [[vc0_i32:%.*]] = arith.constant 0 : i32
  //CHECK-NEXT: [[vproc_linear_idx:%.*]]:3 = mesh.process_multi_index on @mesh0 : index, index, index
  //CHECK-NEXT: [[vdown_linear_idx:%.*]], [[vup_linear_idx:%.*]] = mesh.neighbors_linear_indices on @mesh0[[[vproc_linear_idx]]#0, [[vproc_linear_idx]]#1, [[vproc_linear_idx]]#2] split_axes = [3] : index, index
  //CHECK-NEXT: [[v0:%.*]] = arith.index_cast [[vdown_linear_idx]] : index to i32
  //CHECK-NEXT: [[v1:%.*]] = arith.index_cast [[vup_linear_idx]] : index to i32
  //CHECK-NEXT: [[v2:%.*]] = arith.cmpi sge, [[v1]], [[vc0_i32]] : i32
  //CHECK-NEXT: [[v3:%.*]] = arith.cmpi sge, [[v0]], [[vc0_i32]] : i32
  //CHECK-NEXT: [[valloc:%.*]] = memref.alloc() : memref<12x2xi8>
  //CHECK-NEXT: scf.if [[v3]] {
  //CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][0, %c7] [12, 2] [1, 1] : memref<12x12xi8> to memref<12x2xi8, strided<[12, 1], offset: ?>>
  //CHECK-NEXT:   memref.copy [[vsubview]], [[valloc]] : memref<12x2xi8, strided<[12, 1], offset: ?>> to memref<12x2xi8>
  //CHECK-NEXT:   mpi.send([[valloc]], [[vc91_i32]], [[v0]]) : memref<12x2xi8>, i32, i32
  //CHECK-NEXT: }
  //CHECK-NEXT: scf.if [[v2]] {
  //CHECK-NEXT:   mpi.recv([[valloc]], [[vc91_i32]], [[v1]]) : memref<12x2xi8>, i32, i32
  //CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][0, 0] [12, 2] [1, 1] : memref<12x12xi8> to memref<12x2xi8, strided<[12, 1]>>
  //CHECK-NEXT:   memref.copy [[valloc]], [[vsubview]] : memref<12x2xi8> to memref<12x2xi8, strided<[12, 1]>>
  //CHECK-NEXT: }
  //CHECK-NEXT: memref.dealloc [[valloc]] : memref<12x2xi8>
  //CHECK-NEXT: [[v4:%.*]] = arith.cmpi sge, [[v0]], [[vc0_i32]] : i32
  //CHECK-NEXT: [[v5:%.*]] = arith.cmpi sge, [[v1]], [[vc0_i32]] : i32
  //CHECK-NEXT: [[valloc_0:%.*]] = memref.alloc() : memref<12x3xi8>
  //CHECK-NEXT: scf.if [[v5]] {
  //CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][0, %c2] [12, 3] [1, 1] : memref<12x12xi8> to memref<12x3xi8, strided<[12, 1], offset: ?>>
  //CHECK-NEXT:   memref.copy [[vsubview]], [[valloc_0]] : memref<12x3xi8, strided<[12, 1], offset: ?>> to memref<12x3xi8>
  //CHECK-NEXT:   mpi.send([[valloc_0]], [[vc91_i32]], [[v1]]) : memref<12x3xi8>, i32, i32
  //CHECK-NEXT: }
  //CHECK-NEXT: scf.if [[v4]] {
  //CHECK-NEXT:   mpi.recv([[valloc_0]], [[vc91_i32]], [[v0]]) : memref<12x3xi8>, i32, i32
  //CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][0, %c9] [12, 3] [1, 1] : memref<12x12xi8> to memref<12x3xi8, strided<[12, 1], offset: ?>>
  //CHECK-NEXT:   memref.copy [[valloc_0]], [[vsubview]] : memref<12x3xi8> to memref<12x3xi8, strided<[12, 1], offset: ?>>
  //CHECK-NEXT: }
  //CHECK-NEXT: memref.dealloc [[valloc_0]] : memref<12x3xi8>
  mesh.update_halo %arg0 on @mesh0 split_axes = [[], [3]]
    halo_sizes = [2, 3] : memref<12x12xi8>
  //CHECK-NEXT: return
  return
}

// CHECK-LABEL: func @update_halo_2d
func.func @update_halo_2d(
    // CHECK-SAME: [[varg0:%.*]]: memref<12x12xi8>
    %arg0 : memref<12x12xi8>) {
  // CHECK-NEXT: [[vc10:%.*]] = arith.constant 10 : index
  // CHECK-NEXT: [[vc1:%.*]] = arith.constant 1 : index
  // CHECK-NEXT: [[vc5:%.*]] = arith.constant 5 : index
  // CHECK-NEXT: [[vc8:%.*]] = arith.constant 8 : index
  // CHECK-NEXT: [[vc3:%.*]] = arith.constant 3 : index
  // CHECK-NEXT: [[vc9:%.*]] = arith.constant 9 : index
  // CHECK-NEXT: [[vc91_i32:%.*]] = arith.constant 91 : i32
  // CHECK-NEXT: [[vc0_i32:%.*]] = arith.constant 0 : i32
  // CHECK-NEXT: [[vproc_linear_idx:%.*]]:3 = mesh.process_multi_index on @mesh0 : index, index, index
  // CHECK-NEXT: [[vdown_linear_idx:%.*]], [[vup_linear_idx:%.*]] = mesh.neighbors_linear_indices on @mesh0[[[vproc_linear_idx]]#0, [[vproc_linear_idx]]#1, [[vproc_linear_idx]]#2] split_axes = [1] : index, index
  // CHECK-NEXT: [[v0:%.*]] = arith.index_cast [[vdown_linear_idx]] : index to i32
  // CHECK-NEXT: [[v1:%.*]] = arith.index_cast [[vup_linear_idx]] : index to i32
  // CHECK-NEXT: [[v2:%.*]] = arith.cmpi sge, [[v1]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[v3:%.*]] = arith.cmpi sge, [[v0]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[valloc:%.*]] = memref.alloc([[vc9]]) : memref<?x3xi8>
  // CHECK-NEXT: scf.if [[v3]] {
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][1, %c5] [[[vc9]], 3] [1, 1] : memref<12x12xi8> to memref<?x3xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[vsubview]], [[valloc]] : memref<?x3xi8, strided<[12, 1], offset: ?>> to memref<?x3xi8>
  // CHECK-NEXT:   mpi.send([[valloc]], [[vc91_i32]], [[v0]]) : memref<?x3xi8>, i32, i32
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.if [[v2]] {
  // CHECK-NEXT:   mpi.recv([[valloc]], [[vc91_i32]], [[v1]]) : memref<?x3xi8>, i32, i32
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][1, 0] [[[vc9]], 3] [1, 1] : memref<12x12xi8> to memref<?x3xi8, strided<[12, 1], offset: 12>>
  // CHECK-NEXT:   memref.copy [[valloc]], [[vsubview]] : memref<?x3xi8> to memref<?x3xi8, strided<[12, 1], offset: 12>>
  // CHECK-NEXT: }
  // CHECK-NEXT: memref.dealloc [[valloc]] : memref<?x3xi8>
  // CHECK-NEXT: [[v4:%.*]] = arith.cmpi sge, [[v0]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[v5:%.*]] = arith.cmpi sge, [[v1]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[valloc_0:%.*]] = memref.alloc([[vc9]]) : memref<?x4xi8>
  // CHECK-NEXT: scf.if [[v5]] {
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][1, %c3] [[[vc9]], 4] [1, 1] : memref<12x12xi8> to memref<?x4xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[vsubview]], [[valloc_0]] : memref<?x4xi8, strided<[12, 1], offset: ?>> to memref<?x4xi8>
  // CHECK-NEXT:   mpi.send([[valloc_0]], [[vc91_i32]], [[v1]]) : memref<?x4xi8>, i32, i32
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.if [[v4]] {
  // CHECK-NEXT:   mpi.recv([[valloc_0]], [[vc91_i32]], [[v0]]) : memref<?x4xi8>, i32, i32
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][1, %c8] [[[vc9]], 4] [1, 1] : memref<12x12xi8> to memref<?x4xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[valloc_0]], [[vsubview]] : memref<?x4xi8> to memref<?x4xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT: }
  // CHECK-NEXT: memref.dealloc [[valloc_0]] : memref<?x4xi8>
  // CHECK-NEXT: [[vdown_linear_idx_1:%.*]], [[vup_linear_idx_2:%.*]] = mesh.neighbors_linear_indices on @mesh0[[[vproc_linear_idx]]#0, [[vproc_linear_idx]]#1, [[vproc_linear_idx]]#2] split_axes = [0] : index, index
  // CHECK-NEXT: [[v6:%.*]] = arith.index_cast [[vdown_linear_idx_1]] : index to i32
  // CHECK-NEXT: [[v7:%.*]] = arith.index_cast [[vup_linear_idx_2]] : index to i32
  // CHECK-NEXT: [[v8:%.*]] = arith.cmpi sge, [[v7]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[v9:%.*]] = arith.cmpi sge, [[v6]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[valloc_3:%.*]] = memref.alloc() : memref<1x12xi8>
  // CHECK-NEXT: scf.if [[v9]] {
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][[[vc9]], 0] [1, 12] [1, 1] : memref<12x12xi8> to memref<1x12xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[vsubview]], [[valloc_3]] : memref<1x12xi8, strided<[12, 1], offset: ?>> to memref<1x12xi8>
  // CHECK-NEXT:   mpi.send([[valloc_3]], [[vc91_i32]], [[v6]]) : memref<1x12xi8>, i32, i32
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.if [[v8]] {
  // CHECK-NEXT:   mpi.recv([[valloc_3]], [[vc91_i32]], [[v7]]) : memref<1x12xi8>, i32, i32
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][0, 0] [1, 12] [1, 1] : memref<12x12xi8> to memref<1x12xi8, strided<[12, 1]>>
  // CHECK-NEXT:   memref.copy [[valloc_3]], [[vsubview]] : memref<1x12xi8> to memref<1x12xi8, strided<[12, 1]>>
  // CHECK-NEXT: }
  // CHECK-NEXT: memref.dealloc [[valloc_3]] : memref<1x12xi8>
  // CHECK-NEXT: [[v10:%.*]] = arith.cmpi sge, [[v6]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[v11:%.*]] = arith.cmpi sge, [[v7]], [[vc0_i32]] : i32
  // CHECK-NEXT: [[valloc_4:%.*]] = memref.alloc() : memref<2x12xi8>
  // CHECK-NEXT: scf.if [[v11]] {
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][[[vc1]], 0] [2, 12] [1, 1] : memref<12x12xi8> to memref<2x12xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[vsubview]], [[valloc_4]] : memref<2x12xi8, strided<[12, 1], offset: ?>> to memref<2x12xi8>
  // CHECK-NEXT:   mpi.send([[valloc_4]], [[vc91_i32]], [[v7]]) : memref<2x12xi8>, i32, i32
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.if [[v10]] {
  // CHECK-NEXT:   mpi.recv([[valloc_4]], [[vc91_i32]], [[v6]]) : memref<2x12xi8>, i32, i32
  // CHECK-NEXT:   [[vsubview:%.*]] = memref.subview [[varg0]][[[vc10]], 0] [2, 12] [1, 1] : memref<12x12xi8> to memref<2x12xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT:   memref.copy [[valloc_4]], [[vsubview]] : memref<2x12xi8> to memref<2x12xi8, strided<[12, 1], offset: ?>>
  // CHECK-NEXT: }
  // CHECK-NEXT: memref.dealloc [[valloc_4]] : memref<2x12xi8>
  mesh.update_halo %arg0 on @mesh0 split_axes = [[0], [1]]
      halo_sizes = [1, 2, 3, 4]
      : memref<12x12xi8>
  // CHECK-NEXT: return
  return
}
