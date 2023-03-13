// RUN: mlir-opt -finalize-memref-to-llvm='use-opaque-pointers=1' %s -split-input-file | FileCheck %s
// RUN: mlir-opt -finalize-memref-to-llvm='index-bitwidth=32 use-opaque-pointers=1' %s -split-input-file | FileCheck --check-prefix=CHECK32 %s

// CHECK-LABEL: func @view(
// CHECK: %[[ARG0F:.*]]: index, %[[ARG1F:.*]]: index, %[[ARG2F:.*]]: index
func.func @view(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: %[[ARG2:.*]] = builtin.unrealized_conversion_cast %[[ARG2F:.*]]
  // CHECK: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0F:.*]]
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1F:.*]]
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %0 = memref.alloc() : memref<2048xi8>

  // Test two dynamic sizes.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR:.*]] = llvm.getelementptr %[[BASE_PTR]][%[[ARG2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.insertvalue %[[SHIFTED_BASE_PTR]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG0]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %1 = memref.view %0[%arg2][%arg0, %arg1] : memref<2048xi8> to memref<?x?xf32>

  // Test one dynamic size.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_2:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_2:.*]] = llvm.getelementptr %[[BASE_PTR_2]][%[[ARG2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.insertvalue %[[SHIFTED_BASE_PTR_2]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_2:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_2]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %3 = memref.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>

  // Test static sizes.
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_3:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_3:.*]] = llvm.getelementptr %[[BASE_PTR_3]][%[[ARG2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.insertvalue %[[SHIFTED_BASE_PTR_3]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_3:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_3]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %5 = memref.view %0[%arg2][] : memref<2048xi8> to memref<64x4xf32>

  // Test view memory space.
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<4>, ptr<4>, i64, array<1 x i64>, array<1 x i64>)>
  %6 = memref.alloc() : memref<2048xi8, 4>

  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_4:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_4:.*]] = llvm.getelementptr %[[BASE_PTR_4]][%[[ARG2]]] : (!llvm.ptr<4>, i64) -> !llvm.ptr<4>, i8
  // CHECK: llvm.insertvalue %[[SHIFTED_BASE_PTR_4]], %{{.*}}[1] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_4:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_4]], %{{.*}}[2] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(64 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<4>, ptr<4>, i64, array<2 x i64>, array<2 x i64>)>
  %7 = memref.view %6[%arg2][] : memref<2048xi8, 4> to memref<64x4xf32, 4>

  return
}

// -----

// CHECK-LABEL: func @view_empty_memref(
// CHECK:        %[[ARG0:.*]]: index,
// CHECK:        %[[ARG1:.*]]: memref<0xi8>)
func.func @view_empty_memref(%offset: index, %mem: memref<0xi8>) {

  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %0 = memref.view %mem[%offset][] : memref<0xi8> to memref<0x4xf32>

  return
}

// -----

// Subviews needs to be expanded outside of the memref-to-llvm pass.
// CHECK-LABEL: func @subview(
// CHECK:         %[[MEMREF:.*]]: memref<{{.*}}>,
// CHECK:         %[[ARG0:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG1:[a-zA-Z0-9]*]]: index,
// CHECK:         %[[ARG2:.*]]: index)
// CHECK32-LABEL: func @subview(
// CHECK32:         %[[MEMREF:.*]]: memref<{{.*}}>,
// CHECK32:         %[[ARG0:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG1:[a-zA-Z0-9]*]]: index,
// CHECK32:         %[[ARG2:.*]]: index)
func.func @subview(%0 : memref<64x4xf32, strided<[4, 1], offset: 0>>, %arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK: memref.subview %[[MEMREF]][%[[ARG0]], %[[ARG1]]] [%[[ARG0]], %[[ARG1]]]
  // CHECK32: memref.subview %[[MEMREF]][%[[ARG0]], %[[ARG1]]] [%[[ARG0]], %[[ARG1]]] [%[[ARG0]], %[[ARG1]]]
  %1 = memref.subview %0[%arg0, %arg1][%arg0, %arg1][%arg0, %arg1] :
    memref<64x4xf32, strided<[4, 1], offset: 0>>
  to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}

// -----

// CHECK-LABEL: func @assume_alignment
func.func @assume_alignment(%0 : memref<4x4xf16>) {
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[MEMREF:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-NEXT: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK-NEXT: %[[MASK:.*]] = llvm.mlir.constant(15 : index) : i64
  // CHECK-NEXT: %[[INT:.*]] = llvm.ptrtoint %[[PTR]] : !llvm.ptr to i64
  // CHECK-NEXT: %[[MASKED_PTR:.*]] = llvm.and %[[INT]], %[[MASK:.*]] : i64
  // CHECK-NEXT: %[[CONDITION:.*]] = llvm.icmp "eq" %[[MASKED_PTR]], %[[ZERO]] : i64
  // CHECK-NEXT: "llvm.intr.assume"(%[[CONDITION]]) : (i1) -> ()
  memref.assume_alignment %0, 16 : memref<4x4xf16>
  return
}

// -----

// CHECK-LABEL: func @dim_of_unranked
// CHECK32-LABEL: func @dim_of_unranked
func.func @dim_of_unranked(%unranked: memref<*xi32>) -> index {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %unranked, %c0 : memref<*xi32>
  return %dim : index
}
// CHECK: %[[UNRANKED_DESC:.*]] = builtin.unrealized_conversion_cast

// CHECK: %[[RANKED_DESC:.*]] = llvm.extractvalue %[[UNRANKED_DESC]][1]
// CHECK-SAME:   : !llvm.struct<(i64, ptr)>

// CHECK: %[[OFFSET_PTR:.*]] = llvm.getelementptr %[[RANKED_DESC]]{{\[}}
// CHECK-SAME:   0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64)>

// CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[INDEX_INC:.*]] = llvm.add %[[C1]], %{{.*}} : i64

// CHECK: %[[SIZE_PTR:.*]] = llvm.getelementptr %[[OFFSET_PTR]]{{\[}}
// CHECK-SAME:   %[[INDEX_INC]]] : (!llvm.ptr, i64) -> !llvm.ptr

// CHECK: %[[SIZE:.*]] = llvm.load %[[SIZE_PTR]] : !llvm.ptr -> i64

// CHECK32: %[[SIZE:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32

// -----

// CHECK-LABEL: func @address_space(
func.func @address_space(%arg0 : memref<32xf32, affine_map<(d0) -> (d0)>, 7>) {
  // CHECK: %[[MEMORY:.*]] = llvm.call @malloc(%{{.*}})
  // CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[MEMORY]] : !llvm.ptr to !llvm.ptr<5>
  // CHECK: llvm.insertvalue %[[CAST]], %{{[[:alnum:]]+}}[0]
  // CHECK: llvm.insertvalue %[[CAST]], %{{[[:alnum:]]+}}[1]
  %0 = memref.alloc() : memref<32xf32, affine_map<(d0) -> (d0)>, 5>
  %1 = arith.constant 7 : index
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  %2 = memref.load %0[%1] : memref<32xf32, affine_map<(d0) -> (d0)>, 5>
  func.return
}

// -----

// CHECK-LABEL: func @transpose
//       CHECK:   llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.insertvalue {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:   llvm.extractvalue {{.*}}[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//       CHECK:    llvm.insertvalue {{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
func.func @transpose(%arg0: memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>) {
  %0 = memref.transpose %arg0 (i, j, k) -> (k, i, j) : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>> to memref<?x?x?xf32, strided<[1, ?, ?], offset: ?>>
  return
}

// -----

// CHECK:   llvm.mlir.global external @gv0() {addr_space = 0 : i32} : !llvm.array<2 x f32> {
// CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.array<2 x f32>
// CHECK-NEXT:     llvm.return %0 : !llvm.array<2 x f32>
// CHECK-NEXT:   }
memref.global @gv0 : memref<2xf32> = uninitialized

// CHECK: llvm.mlir.global private @gv1() {addr_space = 0 : i32} : !llvm.array<2 x f32>
memref.global "private" @gv1 : memref<2xf32>

// CHECK: llvm.mlir.global external @gv2(dense<{{\[\[}}0.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00]]> : tensor<2x3xf32>) {addr_space = 0 : i32} : !llvm.array<2 x array<3 x f32>>
memref.global @gv2 : memref<2x3xf32> = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]>

// Test 1D memref.
// CHECK-LABEL: func @get_gv0_memref
func.func @get_gv0_memref() {
  %0 = memref.get_global @gv0 : memref<2xf32>
  // CHECK: %[[DIM:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv0 : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x f32>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM]], {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE]], {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  return
}

// Test 2D memref.
// CHECK-LABEL: func @get_gv2_memref
func.func @get_gv2_memref() {
  // CHECK: %[[DIM0:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[DIM1:.*]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: %[[STRIDE1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv2 : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<3 x f32>>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM0]], {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM1]], {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE1]], {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

  %0 = memref.get_global @gv2 : memref<2x3xf32>
  return
}

// Test scalar memref.
// CHECK: llvm.mlir.global external @gv3(1.000000e+00 : f32) {addr_space = 0 : i32} : f32
memref.global @gv3 : memref<f32> = dense<1.0>

// CHECK-LABEL: func @get_gv3_memref
func.func @get_gv3_memref() {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv3 : !llvm.ptr
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][0] : (!llvm.ptr) -> !llvm.ptr, f32
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr, ptr, i64)>
  %0 = memref.get_global @gv3 : memref<f32>
  return
}

// Test scalar memref with an alignment.
// CHECK: llvm.mlir.global private @gv4(1.000000e+00 : f32) {addr_space = 0 : i32, alignment = 64 : i64} : f32
memref.global "private" @gv4 : memref<f32> = dense<1.0> {alignment = 64}

// -----

// Expand shapes need to be expanded outside of the memref-to-llvm pass.
// CHECK-LABEL: func @expand_shape_static(
// CHECK-SAME:         %[[ARG:.*]]: memref<{{.*}}>)
func.func @expand_shape_static(%arg0: memref<3x4x5xf32>) -> memref<1x3x4x1x5xf32> {
  // CHECK: memref.expand_shape %[[ARG]] {{\[}}[0, 1], [2], [3, 4]]
  // Reshapes that expand a contiguous tensor with some 1's.
  %0 = memref.expand_shape %arg0 [[0, 1], [2], [3, 4]]
      : memref<3x4x5xf32> into memref<1x3x4x1x5xf32>
  return %0 : memref<1x3x4x1x5xf32>
}

// -----

// Collapse shapes need to be expanded outside of the memref-to-llvm pass.
// CHECK-LABEL: func @collapse_shape_static
// CHECK-SAME: %[[ARG:.*]]: memref<1x3x4x1x5xf32>) -> memref<3x4x5xf32> {
func.func @collapse_shape_static(%arg0: memref<1x3x4x1x5xf32>) -> memref<3x4x5xf32> {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2], [3, 4]] :
    memref<1x3x4x1x5xf32> into memref<3x4x5xf32>
  return %0 : memref<3x4x5xf32>
}

// -----

// CHECK-LABEL: func @rank_of_unranked
// CHECK32-LABEL: func @rank_of_unranked
func.func @rank_of_unranked(%unranked: memref<*xi32>) {
  %rank = memref.rank %unranked : memref<*xi32>
  return
}
// CHECK: %[[UNRANKED_DESC:.*]] = builtin.unrealized_conversion_cast
// CHECK-NEXT: llvm.extractvalue %[[UNRANKED_DESC]][0] : !llvm.struct<(i64, ptr)>
// CHECK32: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, ptr)>

// CHECK-LABEL: func @rank_of_ranked
// CHECK32-LABEL: func @rank_of_ranked
func.func @rank_of_ranked(%ranked: memref<?xi32>) {
  %rank = memref.rank %ranked : memref<?xi32>
  return
}
// CHECK: llvm.mlir.constant(1 : index) : i64
// CHECK32: llvm.mlir.constant(1 : index) : i32

// -----

// CHECK-LABEL: func @atomic_rmw
func.func @atomic_rmw(%I : memref<10xi32>, %ival : i32, %F : memref<10xf32>, %fval : f32, %i : index) {
  memref.atomic_rmw assign %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw xchg %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw addi %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw add %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw maxs %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw max %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw mins %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw min %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw maxu %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw umax %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw minu %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw umin %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw addf %fval, %F[%i] : (f32, memref<10xf32>) -> f32
  // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw ori %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw _or %{{.*}}, %{{.*}} acq_rel
  memref.atomic_rmw andi %ival, %I[%i] : (i32, memref<10xi32>) -> i32
  // CHECK: llvm.atomicrmw _and %{{.*}}, %{{.*}} acq_rel
  return
}

// -----

// CHECK-LABEL: func @generic_atomic_rmw
func.func @generic_atomic_rmw(%I : memref<10xi32>, %i : index) {
  %x = memref.generic_atomic_rmw %I[%i] : memref<10xi32> {
    ^bb0(%old_value : i32):
      memref.atomic_yield %old_value : i32
  }
  llvm.return
}
// CHECK:        %[[INIT:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK-NEXT:   llvm.br ^bb1(%[[INIT]] : i32)
// CHECK-NEXT: ^bb1(%[[LOADED:.*]]: i32):
// CHECK-NEXT:   %[[PAIR:.*]] = llvm.cmpxchg %{{.*}}, %[[LOADED]], %[[LOADED]]
// CHECK-SAME:                      acq_rel monotonic : !llvm.ptr, i32
// CHECK-NEXT:   %[[NEW:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK-NEXT:   %[[OK:.*]] = llvm.extractvalue %[[PAIR]][1]
// CHECK-NEXT:   llvm.cond_br %[[OK]], ^bb2, ^bb1(%[[NEW]] : i32)

// -----

// CHECK-LABEL: func @generic_atomic_rmw_in_alloca_scope
func.func @generic_atomic_rmw_in_alloca_scope(){
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() : memref<2x3xi32>
  memref.alloca_scope  {
    %0 = memref.generic_atomic_rmw %alloc[%c1, %c1] : memref<2x3xi32> {
    ^bb0(%arg0: i32):
      memref.atomic_yield %arg0 : i32
    }
  }
  return
}
// CHECK:        %[[STACK_SAVE:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-NEXT:   llvm.br ^bb1
// CHECK:      ^bb1:
// CHECK:        %[[INIT:.*]] = llvm.load %[[BUF:.*]] : !llvm.ptr -> i32
// CHECK-NEXT:   llvm.br ^bb2(%[[INIT]] : i32)
// CHECK-NEXT: ^bb2(%[[LOADED:.*]]: i32):
// CHECK-NEXT:   %[[PAIR:.*]] = llvm.cmpxchg %[[BUF]], %[[LOADED]], %[[LOADED]]
// CHECK-SAME:     acq_rel monotonic : !llvm.ptr, i32
// CHECK-NEXT:   %[[NEW:.*]] = llvm.extractvalue %[[PAIR]][0]
// CHECK-NEXT:   %[[OK:.*]] = llvm.extractvalue %[[PAIR]][1]
// CHECK-NEXT:   llvm.cond_br %[[OK]], ^bb3, ^bb2(%[[NEW]] : i32)
// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   llvm.intr.stackrestore %[[STACK_SAVE]] : !llvm.ptr
// CHECK-NEXT:   llvm.br ^bb4
// CHECK-NEXT: ^bb4:
// CHECK-NEXT:   return

// -----

// CHECK-LABEL: func @memref_copy_ranked
func.func @memref_copy_ranked() {
  %0 = memref.alloc() : memref<2xf32>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.cast %0 : memref<2xf32> to memref<?xf32>
  %2 = memref.alloc() : memref<2xf32>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %3 = memref.cast %2 : memref<2xf32> to memref<?xf32>
  memref.copy %1, %3 : memref<?xf32> to memref<?xf32>
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[EXTRACT0:%.*]] = llvm.extractvalue {{%.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[MUL:%.*]] = llvm.mul [[ONE]], [[EXTRACT0]] : i64
  // CHECK: [[NULL:%.*]] = llvm.mlir.null : !llvm.ptr
  // CHECK: [[GEP:%.*]] = llvm.getelementptr [[NULL]][1] : (!llvm.ptr) -> !llvm.ptr, f32
  // CHECK: [[PTRTOINT:%.*]] = llvm.ptrtoint [[GEP]] : !llvm.ptr to i64
  // CHECK: [[SIZE:%.*]] = llvm.mul [[MUL]], [[PTRTOINT]] : i64
  // CHECK: [[EXTRACT1P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[EXTRACT1O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[GEP1:%.*]] = llvm.getelementptr [[EXTRACT1P]][[[EXTRACT1O]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: [[EXTRACT2P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[EXTRACT2O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[GEP2:%.*]] = llvm.getelementptr [[EXTRACT2P]][[[EXTRACT2O]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: [[VOLATILE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK: "llvm.intr.memcpy"([[GEP2]], [[GEP1]], [[SIZE]], [[VOLATILE]]) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
  return
}


// -----

// CHECK-LABEL: func @memref_copy_contiguous
func.func @memref_copy_contiguous(%in: memref<16x2xi32>, %offset: index) {
  %buf = memref.alloc() : memref<1x2xi32>
  %sub = memref.subview %in[%offset, 0] [1, 2] [1, 1] : memref<16x2xi32> to memref<1x2xi32, strided<[2, 1], offset: ?>>
  memref.copy %sub, %buf : memref<1x2xi32, strided<[2, 1], offset: ?>> to memref<1x2xi32>
  // Skip the memref descriptor of the alloc.
  // CHECK: llvm.insertvalue {{%.*}}, {{%.*}}[4, 1]
  // Get the memref for the subview.
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %{{.*}}[%{{.*}}, 0] [1, 2] [1, 1] : memref<16x2xi32> to memref<1x2xi32, strided<[2, 1], offset: ?>>
  // CHECK: %[[DESC:.*]] = builtin.unrealized_conversion_cast %[[SUBVIEW]] : memref<1x2xi32, strided<[2, 1], offset: ?>> to !llvm.struct<(ptr
  // CHECK: [[EXTRACT0:%.*]] = llvm.extractvalue %[[DESC]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[MUL1:%.*]] = llvm.mul {{.*}}, [[EXTRACT0]] : i64
  // CHECK: [[EXTRACT1:%.*]] = llvm.extractvalue %[[DESC]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[MUL2:%.*]] = llvm.mul [[MUL1]], [[EXTRACT1]] : i64
  // CHECK: [[NULL:%.*]] = llvm.mlir.null : !llvm.ptr
  // CHECK: [[GEP:%.*]] = llvm.getelementptr [[NULL]][1] : (!llvm.ptr) -> !llvm.ptr, i32
  // CHECK: [[PTRTOINT:%.*]] = llvm.ptrtoint [[GEP]] : !llvm.ptr to i64
  // CHECK: [[SIZE:%.*]] = llvm.mul [[MUL2]], [[PTRTOINT]] : i64
  // CHECK: [[EXTRACT1P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[EXTRACT1O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[GEP1:%.*]] = llvm.getelementptr [[EXTRACT1P]][[[EXTRACT1O]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  // CHECK: [[EXTRACT2P:%.*]] = llvm.extractvalue {{%.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[EXTRACT2O:%.*]] = llvm.extractvalue {{%.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[GEP2:%.*]] = llvm.getelementptr [[EXTRACT2P]][[[EXTRACT2O]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  // CHECK: [[VOLATILE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK: "llvm.intr.memcpy"([[GEP2]], [[GEP1]], [[SIZE]], [[VOLATILE]]) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
  return
}

// -----

// CHECK-LABEL: func @memref_copy_0d_offset
func.func @memref_copy_0d_offset(%in: memref<2xi32>) {
  %buf = memref.alloc() : memref<i32>
  %sub = memref.subview %in[1] [1] [1] : memref<2xi32> to memref<1xi32, strided<[1], offset: 1>>
  %scalar = memref.collapse_shape %sub [] : memref<1xi32, strided<[1], offset: 1>> into memref<i32, strided<[], offset: 1>>
  memref.copy %scalar, %buf : memref<i32, strided<[], offset: 1>> to memref<i32>
  // CHECK: llvm.intr.memcpy
  return
}

// -----

// CHECK-LABEL: func @memref_copy_noncontiguous
func.func @memref_copy_noncontiguous(%in: memref<16x2xi32>, %offset: index) {
  %buf = memref.alloc() : memref<2x1xi32>
  %sub = memref.subview %in[%offset, 0] [2, 1] [1, 1] : memref<16x2xi32> to memref<2x1xi32, strided<[2, 1], offset: ?>>
  memref.copy %sub, %buf : memref<2x1xi32, strided<[2, 1], offset: ?>> to memref<2x1xi32>
  // CHECK: llvm.call @memrefCopy
  return
}

// -----

// CHECK-LABEL: func @memref_copy_unranked
func.func @memref_copy_unranked() {
  %0 = memref.alloc() : memref<2xi1>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.cast %0 : memref<2xi1> to memref<*xi1>
  %2 = memref.alloc() : memref<2xi1>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %3 = memref.cast %2 : memref<2xi1> to memref<*xi1>
  memref.copy %1, %3 : memref<*xi1> to memref<*xi1>
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[ALLOCA:%.*]] = llvm.alloca [[ONE]] x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
  // CHECK: llvm.store {{%.*}}, [[ALLOCA]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
  // CHECK: [[RANK:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[UNDEF:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
  // CHECK: [[INSERT:%.*]] = llvm.insertvalue [[RANK]], [[UNDEF]][0] : !llvm.struct<(i64, ptr)>
  // CHECK: [[INSERT2:%.*]] = llvm.insertvalue [[ALLOCA]], [[INSERT]][1] : !llvm.struct<(i64, ptr)>
  // CHECK: [[STACKSAVE:%.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: [[RANK2:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[ALLOCA2:%.*]] = llvm.alloca [[RANK2]] x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
  // CHECK: llvm.store {{%.*}}, [[ALLOCA2]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
  // CHECK: [[ALLOCA3:%.*]] = llvm.alloca [[RANK2]] x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
  // CHECK: llvm.store [[INSERT2]], [[ALLOCA3]] : !llvm.struct<(i64, ptr)>, !llvm.ptr
  // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.call @memrefCopy([[SIZE]], [[ALLOCA2]], [[ALLOCA3]]) : (i64, !llvm.ptr, !llvm.ptr) -> ()
  // CHECK: llvm.intr.stackrestore [[STACKSAVE]]
  return
}

// -----

// CHECK-LABEL: func @extract_aligned_pointer_as_index
func.func @extract_aligned_pointer_as_index(%m: memref<?xf32>) -> index {
  %0 = memref.extract_aligned_pointer_as_index %m: memref<?xf32> -> index
  // CHECK: %[[E:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[I64:.*]] = llvm.ptrtoint %[[E]] : !llvm.ptr to i64
  // CHECK: %[[R:.*]] = builtin.unrealized_conversion_cast %[[I64]] : i64 to index

  // CHECK: return %[[R:.*]] : index
  return %0: index
}

// -----

// CHECK-LABEL: func @extract_strided_metadata(
// CHECK-SAME: %[[ARG:.*]]: memref
// CHECK: %[[MEM_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : memref<?x?xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[ALIGNED_BASE:.*]] = llvm.extractvalue %[[MEM_DESC]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[DESC:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
// CHECK: %[[DESC0:.*]] = llvm.insertvalue %[[BASE]], %[[DESC]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[ALIGNED_BASE]], %[[DESC0]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK: %[[OFF0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: %[[BASE_BUFFER_DESC:.*]] = llvm.insertvalue %[[OFF0]], %[[DESC1]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK: %[[OFFSET:.*]] = llvm.extractvalue %[[MEM_DESC]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[SIZE0:.*]] = llvm.extractvalue %[[MEM_DESC]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[SIZE1:.*]] = llvm.extractvalue %[[MEM_DESC]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE0:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[STRIDE1:.*]] = llvm.extractvalue %[[MEM_DESC]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
func.func @extract_strided_metadata(
    %ref: memref<?x?xf32, strided<[?,?], offset: ?>>) {

  %base, %offset, %sizes:2, %strides:2 =
    memref.extract_strided_metadata %ref : memref<?x?xf32, strided<[?,?], offset: ?>>
    -> memref<f32>, index,
       index, index,
       index, index

  return
}

// -----

// CHECK-LABEL: func @load_non_temporal(
func.func @load_non_temporal(%arg0 : memref<32xf32, affine_map<(d0) -> (d0)>>) {  
  %1 = arith.constant 7 : index
  // CHECK: llvm.load %{{.*}} {nontemporal} : !llvm.ptr -> f32
  %2 = memref.load %arg0[%1] {nontemporal = true} : memref<32xf32, affine_map<(d0) -> (d0)>>
  func.return
}

// -----

// CHECK-LABEL: func @store_non_temporal(
func.func @store_non_temporal(%input : memref<32xf32, affine_map<(d0) -> (d0)>>, %output : memref<32xf32, affine_map<(d0) -> (d0)>>) {
  %1 = arith.constant 7 : index
  %2 = memref.load %input[%1] {nontemporal = true} : memref<32xf32, affine_map<(d0) -> (d0)>>
  // CHECK: llvm.store %{{.*}}, %{{.*}}  {nontemporal} : f32, !llvm.ptr
  memref.store %2, %output[%1] {nontemporal = true} : memref<32xf32, affine_map<(d0) -> (d0)>>
  func.return
}
