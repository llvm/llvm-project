// RUN: mlir-opt -finalize-memref-to-llvm='use-opaque-pointers=0' %s -split-input-file | FileCheck %s
// RUN: mlir-opt -finalize-memref-to-llvm='index-bitwidth=32 use-opaque-pointers=0' -split-input-file %s | FileCheck --check-prefix=CHECK32 %s

// CHECK-LABEL: func @view(
// CHECK: %[[ARG1F:.*]]: index, %[[ARG2F:.*]]: index
func.func @view(%arg1 : index, %arg2 : index) {
  // CHECK: %[[ARG2:.*]] = builtin.unrealized_conversion_cast %[[ARG2F:.*]]
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1F:.*]]
  // CHECK: llvm.mlir.constant(2048 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  %0 = memref.alloc() : memref<2048xi8>

  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[BASE_PTR_2:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[SHIFTED_BASE_PTR_2:.*]] = llvm.getelementptr %[[BASE_PTR_2]][%[[ARG2]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: %[[CAST_SHIFTED_BASE_PTR_2:.*]] = llvm.bitcast %[[SHIFTED_BASE_PTR_2]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: llvm.insertvalue %[[CAST_SHIFTED_BASE_PTR_2]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: %[[C0_2:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[C0_2]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ARG1]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.mul %{{.*}}, %[[ARG1]]
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %1 = memref.view %0[%arg2][%arg1] : memref<2048xi8> to memref<4x?xf32>
  return
}

// -----

// CHECK:   llvm.mlir.global external @gv0() {addr_space = 0 : i32} : !llvm.array<2 x f32> {
// CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.array<2 x f32>
// CHECK-NEXT:     llvm.return %0 : !llvm.array<2 x f32>
// CHECK-NEXT:   }
memref.global @gv0 : memref<2xf32> = uninitialized

// CHECK-LABEL: func @get_gv0_memref
func.func @get_gv0_memref() {
  %0 = memref.get_global @gv0 : memref<2xf32>
  // CHECK: %[[DIM:.*]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @gv0 : !llvm.ptr<array<2 x f32>>
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ADDR]][0, 0] : (!llvm.ptr<array<2 x f32>>) -> !llvm.ptr<f32>
  // CHECK: %[[DEADBEEF:.*]] = llvm.mlir.constant(3735928559 : index) : i64
  // CHECK: %[[DEADBEEFPTR:.*]] = llvm.inttoptr %[[DEADBEEF]] : i64 to !llvm.ptr<f32>
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DEADBEEFPTR]], {{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[GEP]], {{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue %[[OFFSET]], {{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[DIM]], {{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: llvm.insertvalue %[[STRIDE]], {{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  return
}

// -----

// CHECK-LABEL: func @memref_copy_unranked
func.func @memref_copy_unranked() {
  %0 = memref.alloc() : memref<2xi1>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
  %1 = memref.cast %0 : memref<2xi1> to memref<*xi1>
  %2 = memref.alloc() : memref<2xi1>
  // CHECK: llvm.mlir.constant(2 : index) : i64
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>
  %3 = memref.cast %2 : memref<2xi1> to memref<*xi1>
  memref.copy %1, %3 : memref<*xi1> to memref<*xi1>
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[ALLOCA:%.*]] = llvm.alloca %35 x !llvm.struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>>
  // CHECK: llvm.store {{%.*}}, [[ALLOCA]] : !llvm.ptr<struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>>
  // CHECK: [[BITCAST:%.*]] = llvm.bitcast [[ALLOCA]] : !llvm.ptr<struct<(ptr<i1>, ptr<i1>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
  // CHECK: [[RANK:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[UNDEF:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
  // CHECK: [[INSERT:%.*]] = llvm.insertvalue [[RANK]], [[UNDEF]][0] : !llvm.struct<(i64, ptr<i8>)>
  // CHECK: [[INSERT2:%.*]] = llvm.insertvalue [[BITCAST]], [[INSERT]][1] : !llvm.struct<(i64, ptr<i8>)>
  // CHECK: [[STACKSAVE:%.*]] = llvm.intr.stacksave : !llvm.ptr<i8>
  // CHECK: [[RANK2:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[ALLOCA2:%.*]] = llvm.alloca [[RANK2]] x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
  // CHECK: llvm.store {{%.*}}, [[ALLOCA2]] : !llvm.ptr<struct<(i64, ptr<i8>)>>
  // CHECK: [[ALLOCA3:%.*]] = llvm.alloca [[RANK2]] x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
  // CHECK: llvm.store [[INSERT2]], [[ALLOCA3]] : !llvm.ptr<struct<(i64, ptr<i8>)>>
  // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.call @memrefCopy([[SIZE]], [[ALLOCA2]], [[ALLOCA3]]) : (i64, !llvm.ptr<struct<(i64, ptr<i8>)>>, !llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
  // CHECK: llvm.intr.stackrestore [[STACKSAVE]]
  return
}

// -----

// CHECK-LABEL: func @mixed_alloc(
//       CHECK:   %[[Marg:.*]]: index, %[[Narg:.*]]: index)
func.func @mixed_alloc(%arg0: index, %arg1: index) -> memref<?x42x?xf32> {
//   CHECK-DAG:  %[[M:.*]] = builtin.unrealized_conversion_cast %[[Marg]]
//   CHECK-DAG:  %[[N:.*]] = builtin.unrealized_conversion_cast %[[Narg]]
//       CHECK:  %[[c42:.*]] = llvm.mlir.constant(42 : index) : i64
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mul %[[N]], %[[c42]] : i64
//  CHECK-NEXT:  %[[sz:.*]] = llvm.mul %[[st0]], %[[M]] : i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
//  CHECK-NEXT:  llvm.call @malloc(%[[sz_bytes]]) : (i64) -> !llvm.ptr<i8>
//  CHECK-NEXT:  llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<f32>
//  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[c42]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st0]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[one]], %{{.*}}[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
  %0 = memref.alloc(%arg0, %arg1) : memref<?x42x?xf32>
  return %0 : memref<?x42x?xf32>
}

// -----

// CHECK-LABEL: func @mixed_dealloc
func.func @mixed_dealloc(%arg0: memref<?x42x?xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:  %[[ptri8:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK-NEXT:  llvm.call @free(%[[ptri8]]) : (!llvm.ptr<i8>) -> ()
  memref.dealloc %arg0 : memref<?x42x?xf32>
  return
}

// -----

// CHECK-LABEL: func @dynamic_alloc(
//       CHECK:   %[[Marg:.*]]: index, %[[Narg:.*]]: index)
func.func @dynamic_alloc(%arg0: index, %arg1: index) -> memref<?x?xf32> {
//   CHECK-DAG:  %[[M:.*]] = builtin.unrealized_conversion_cast %[[Marg]]
//   CHECK-DAG:  %[[N:.*]] = builtin.unrealized_conversion_cast %[[Narg]]
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
//  CHECK-NEXT:  %[[sz:.*]] = llvm.mul %[[N]], %[[M]] : i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
//  CHECK-NEXT:  llvm.call @malloc(%[[sz_bytes]]) : (i64) -> !llvm.ptr<i8>
//  CHECK-NEXT:  llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<f32>
//  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[one]], %{{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_dealloc
func.func @dynamic_dealloc(%arg0: memref<?x?xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  %[[ptri8:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK-NEXT:  llvm.call @free(%[[ptri8]]) : (!llvm.ptr<i8>) -> ()
  memref.dealloc %arg0 : memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_static_to_dynamic
func.func @memref_cast_static_to_dynamic(%static : memref<10x42xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref.cast %static : memref<10x42xf32> to memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_static_to_mixed
func.func @memref_cast_static_to_mixed(%static : memref<10x42xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref.cast %static : memref<10x42xf32> to memref<?x42xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_dynamic_to_static
func.func @memref_cast_dynamic_to_static(%dynamic : memref<?x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref.cast %dynamic : memref<?x?xf32> to memref<10x12xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_dynamic_to_mixed
func.func @memref_cast_dynamic_to_mixed(%dynamic : memref<?x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref.cast %dynamic : memref<?x?xf32> to memref<?x12xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_mixed_to_dynamic
func.func @memref_cast_mixed_to_dynamic(%mixed : memref<42x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref.cast %mixed : memref<42x?xf32> to memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_mixed_to_static
func.func @memref_cast_mixed_to_static(%mixed : memref<42x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref.cast %mixed : memref<42x?xf32> to memref<42x1xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_mixed_to_mixed
func.func @memref_cast_mixed_to_mixed(%mixed : memref<42x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref.cast %mixed : memref<42x?xf32> to memref<?x1xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_ranked_to_unranked
// CHECK32-LABEL: func @memref_cast_ranked_to_unranked
func.func @memref_cast_ranked_to_unranked(%arg : memref<42x2x?xf32>) {
// CHECK-DAG:  %[[c:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:  %[[p:.*]] = llvm.alloca %[[c]] x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>>
// CHECK-DAG:  llvm.store %{{.*}}, %[[p]] : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>>
// CHECK-DAG:  %[[p2:.*]] = llvm.bitcast %[[p]] : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>> to !llvm.ptr<i8>
// CHECK-DAG:  %[[r:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK    :  llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK-DAG:  llvm.insertvalue %[[r]], %{{.*}}[0] : !llvm.struct<(i64, ptr<i8>)>
// CHECK-DAG:  llvm.insertvalue %[[p2]], %{{.*}}[1] : !llvm.struct<(i64, ptr<i8>)>
// CHECK32-DAG:  %[[c:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK32-DAG:  %[[p:.*]] = llvm.alloca %[[c]] x !llvm.struct<(ptr<f32>, ptr<f32>, i32, array<3 x i32>, array<3 x i32>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i32, array<3 x i32>, array<3 x i32>)>>
// CHECK32-DAG:  llvm.store %{{.*}}, %[[p]] : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i32, array<3 x i32>, array<3 x i32>)>>
// CHECK32-DAG:  %[[p2:.*]] = llvm.bitcast %[[p]] : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i32, array<3 x i32>, array<3 x i32>)>> to !llvm.ptr<i8>
// CHECK32-DAG:  %[[r:.*]] = llvm.mlir.constant(3 : index) : i32
// CHECK32    :  llvm.mlir.undef : !llvm.struct<(i32, ptr<i8>)>
// CHECK32-DAG:  llvm.insertvalue %[[r]], %{{.*}}[0] : !llvm.struct<(i32, ptr<i8>)>
// CHECK32-DAG:  llvm.insertvalue %[[p2]], %{{.*}}[1] : !llvm.struct<(i32, ptr<i8>)>
  %0 = memref.cast %arg : memref<42x2x?xf32> to memref<*xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_unranked_to_ranked
func.func @memref_cast_unranked_to_ranked(%arg : memref<*xf32>) {
//      CHECK: %[[p:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i64, ptr<i8>)>
// CHECK-NEXT: llvm.bitcast %[[p]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>>
  %0 = memref.cast %arg : memref<*xf32> to memref<?x?x10x2xf32>
  return
}

// -----

// CHECK-LABEL: @memref_reinterpret_cast_unranked_to_dynamic_shape
func.func @memref_reinterpret_cast_unranked_to_dynamic_shape(%offset: index,
                                                        %size_0 : index,
                                                        %size_1 : index,
                                                        %stride_0 : index,
                                                        %stride_1 : index,
                                                        %input : memref<*xf32>) {
  %output = memref.reinterpret_cast %input to
           offset: [%offset], sizes: [%size_0, %size_1],
           strides: [%stride_0, %stride_1]
           : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
  return
}
// CHECK-SAME: ([[OFFSETarg:%[a-z,0-9]+]]: index,
// CHECK-SAME: [[SIZE_0arg:%[a-z,0-9]+]]: index, [[SIZE_1arg:%[a-z,0-9]+]]: index,
// CHECK-SAME: [[STRIDE_0arg:%[a-z,0-9]+]]: index, [[STRIDE_1arg:%[a-z,0-9]+]]: index,
// CHECK-DAG: [[OFFSET:%.*]] = builtin.unrealized_conversion_cast [[OFFSETarg]]
// CHECK-DAG: [[SIZE_0:%.*]] = builtin.unrealized_conversion_cast [[SIZE_0arg]]
// CHECK-DAG: [[SIZE_1:%.*]] = builtin.unrealized_conversion_cast [[SIZE_1arg]]
// CHECK-DAG: [[STRIDE_0:%.*]] = builtin.unrealized_conversion_cast [[STRIDE_0arg]]
// CHECK-DAG: [[STRIDE_1:%.*]] = builtin.unrealized_conversion_cast [[STRIDE_1arg]]
// CHECK-DAG: [[INPUT:%.*]] = builtin.unrealized_conversion_cast
// CHECK: [[OUT_0:%.*]] = llvm.mlir.undef : [[TY:!.*]]
// CHECK: [[DESCRIPTOR:%.*]] = llvm.extractvalue [[INPUT]][1] : !llvm.struct<(i64, ptr<i8>)>
// CHECK: [[BASE_PTR_PTR:%.*]] = llvm.bitcast [[DESCRIPTOR]] : !llvm.ptr<i8> to !llvm.ptr<ptr<f32>>
// CHECK: [[BASE_PTR:%.*]] = llvm.load [[BASE_PTR_PTR]] : !llvm.ptr<ptr<f32>>
// CHECK: [[BASE_PTR_PTR_:%.*]] = llvm.bitcast [[DESCRIPTOR]] : !llvm.ptr<i8> to !llvm.ptr<ptr<f32>>
// CHECK: [[ALIGNED_PTR_PTR:%.*]] = llvm.getelementptr [[BASE_PTR_PTR_]]{{\[}}1]
// CHECK-SAME: : (!llvm.ptr<ptr<f32>>) -> !llvm.ptr<ptr<f32>>
// CHECK: [[ALIGNED_PTR:%.*]] = llvm.load [[ALIGNED_PTR_PTR]] : !llvm.ptr<ptr<f32>>
// CHECK: [[OUT_1:%.*]] = llvm.insertvalue [[BASE_PTR]], [[OUT_0]][0] : [[TY]]
// CHECK: [[OUT_2:%.*]] = llvm.insertvalue [[ALIGNED_PTR]], [[OUT_1]][1] : [[TY]]
// CHECK: [[OUT_3:%.*]] = llvm.insertvalue [[OFFSET]], [[OUT_2]][2] : [[TY]]
// CHECK: [[OUT_4:%.*]] = llvm.insertvalue [[SIZE_0]], [[OUT_3]][3, 0] : [[TY]]
// CHECK: [[OUT_5:%.*]] = llvm.insertvalue [[STRIDE_0]], [[OUT_4]][4, 0] : [[TY]]
// CHECK: [[OUT_6:%.*]] = llvm.insertvalue [[SIZE_1]], [[OUT_5]][3, 1] : [[TY]]
// CHECK: [[OUT_7:%.*]] = llvm.insertvalue [[STRIDE_1]], [[OUT_6]][4, 1] : [[TY]]

// -----

// CHECK-LABEL: @memref_reshape
func.func @memref_reshape(%input : memref<2x3xf32>, %shape : memref<?xindex>) {
  %output = memref.reshape %input(%shape)
                : (memref<2x3xf32>, memref<?xindex>) -> memref<*xf32>
  return
}
// CHECK: [[INPUT:%.*]] = builtin.unrealized_conversion_cast %{{.*}} to [[INPUT_TY:!.*]]
// CHECK: [[SHAPE:%.*]] = builtin.unrealized_conversion_cast %{{.*}} to [[SHAPE_TY:!.*]]
// CHECK: [[RANK:%.*]] = llvm.extractvalue [[SHAPE]][3, 0] : [[SHAPE_TY]]
// CHECK: [[UNRANKED_OUT_O:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK: [[UNRANKED_OUT_1:%.*]] = llvm.insertvalue [[RANK]], [[UNRANKED_OUT_O]][0] : !llvm.struct<(i64, ptr<i8>)>

// Compute size in bytes to allocate result ranked descriptor
// CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[C2:%.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK: [[INDEX_SIZE:%.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK: [[PTR_SIZE:%.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK: [[DOUBLE_PTR_SIZE:%.*]] = llvm.mul [[C2]], [[PTR_SIZE]] : i64
// CHECK: [[DESC_ALLOC_SIZE:%.*]] = llvm.add [[DOUBLE_PTR_SIZE]], %{{.*}}
// CHECK: [[UNDERLYING_DESC:%.*]] = llvm.alloca [[DESC_ALLOC_SIZE]] x i8
// CHECK: llvm.insertvalue [[UNDERLYING_DESC]], [[UNRANKED_OUT_1]][1]

// Set allocated, aligned pointers and offset.
// CHECK: [[ALLOC_PTR:%.*]] = llvm.extractvalue [[INPUT]][0] : [[INPUT_TY]]
// CHECK: [[ALIGN_PTR:%.*]] = llvm.extractvalue [[INPUT]][1] : [[INPUT_TY]]
// CHECK: [[OFFSET:%.*]] = llvm.extractvalue [[INPUT]][2] : [[INPUT_TY]]
// CHECK: [[BASE_PTR_PTR:%.*]] = llvm.bitcast [[UNDERLYING_DESC]]
// CHECK-SAME:                     !llvm.ptr<i8> to !llvm.ptr<ptr<f32>>
// CHECK: llvm.store [[ALLOC_PTR]], [[BASE_PTR_PTR]] : !llvm.ptr<ptr<f32>>
// CHECK: [[BASE_PTR_PTR_:%.*]] = llvm.bitcast [[UNDERLYING_DESC]] : !llvm.ptr<i8> to !llvm.ptr<ptr<f32>>
// CHECK: [[ALIGNED_PTR_PTR:%.*]] = llvm.getelementptr [[BASE_PTR_PTR_]]{{\[}}1]
// CHECK: llvm.store [[ALIGN_PTR]], [[ALIGNED_PTR_PTR]] : !llvm.ptr<ptr<f32>>
// CHECK: [[BASE_PTR_PTR__:%.*]] = llvm.bitcast [[UNDERLYING_DESC]] : !llvm.ptr<i8> to !llvm.ptr<ptr<f32>>
// CHECK: [[OFFSET_PTR_:%.*]] = llvm.getelementptr [[BASE_PTR_PTR__]]{{\[}}2]
// CHECK: [[OFFSET_PTR:%.*]] = llvm.bitcast [[OFFSET_PTR_]]
// CHECK: llvm.store [[OFFSET]], [[OFFSET_PTR]] : !llvm.ptr<i64>

// Iterate over shape operand in reverse order and set sizes and strides.
// CHECK: [[STRUCT_PTR:%.*]] = llvm.bitcast [[UNDERLYING_DESC]]
// CHECK-SAME: !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, i64)>>
// CHECK: [[SIZES_PTR:%.*]] = llvm.getelementptr [[STRUCT_PTR]]{{\[}}0, 3]
// CHECK: [[STRIDES_PTR:%.*]] = llvm.getelementptr [[SIZES_PTR]]{{\[}}[[RANK]]]
// CHECK: [[SHAPE_IN_PTR:%.*]] = llvm.extractvalue [[SHAPE]][1] : [[SHAPE_TY]]
// CHECK: [[C1_:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[RANK_MIN_1:%.*]] = llvm.sub [[RANK]], [[C1_]] : i64
// CHECK: llvm.br ^bb1([[RANK_MIN_1]], [[C1_]] : i64, i64)

// CHECK: ^bb1([[DIM:%.*]]: i64, [[CUR_STRIDE:%.*]]: i64):
// CHECK:   [[C0_:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:   [[COND:%.*]] = llvm.icmp "sge" [[DIM]], [[C0_]] : i64
// CHECK:   llvm.cond_br [[COND]], ^bb2, ^bb3

// CHECK: ^bb2:
// CHECK:   [[SIZE_PTR:%.*]] = llvm.getelementptr [[SHAPE_IN_PTR]]{{\[}}[[DIM]]]
// CHECK:   [[SIZE:%.*]] = llvm.load [[SIZE_PTR]] : !llvm.ptr<i64>
// CHECK:   [[TARGET_SIZE_PTR:%.*]] = llvm.getelementptr [[SIZES_PTR]]{{\[}}[[DIM]]]
// CHECK:   llvm.store [[SIZE]], [[TARGET_SIZE_PTR]] : !llvm.ptr<i64>
// CHECK:   [[TARGET_STRIDE_PTR:%.*]] = llvm.getelementptr [[STRIDES_PTR]]{{\[}}[[DIM]]]
// CHECK:   llvm.store [[CUR_STRIDE]], [[TARGET_STRIDE_PTR]] : !llvm.ptr<i64>
// CHECK:   [[UPDATE_STRIDE:%.*]] = llvm.mul [[CUR_STRIDE]], [[SIZE]] : i64
// CHECK:   [[STRIDE_COND:%.*]] = llvm.sub [[DIM]], [[C1_]] : i64
// CHECK:   llvm.br ^bb1([[STRIDE_COND]], [[UPDATE_STRIDE]] : i64, i64)

// CHECK: ^bb3:
// CHECK:   return
