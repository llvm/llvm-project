// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -split-input-file %s | FileCheck %s --check-prefix=BAREPTR

// BAREPTR-LABEL: func @check_noalias
// BAREPTR-SAME: %{{.*}}: !llvm.ptr<float> {llvm.noalias = true}, %{{.*}}: !llvm.ptr<float> {llvm.noalias = true}
func @check_noalias(%static : memref<2xf32> {llvm.noalias = true}, %other : memref<2xf32> {llvm.noalias = true}) {
    return
}

// -----

// CHECK-LABEL: func @check_static_return
// CHECK-COUNT-2: !llvm.ptr<float>
// CHECK-COUNT-5: !llvm.i64
// CHECK-SAME: -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-LABEL: func @check_static_return
// BAREPTR-SAME: (%[[arg:.*]]: !llvm.ptr<float>) -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
func @check_static_return(%static : memref<32x18xf32>) -> memref<32x18xf32> {
// CHECK:  llvm.return %{{.*}} : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

// BAREPTR: %[[udf:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base:.*]] = llvm.insertvalue %[[arg]], %[[udf]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[aligned:.*]] = llvm.insertvalue %[[arg]], %[[base]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins0:.*]] = llvm.insertvalue %[[val0]], %[[aligned]][2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins1:.*]] = llvm.insertvalue %[[val1]], %[[ins0]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val2:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins2:.*]] = llvm.insertvalue %[[val2]], %[[ins1]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val3:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins3:.*]] = llvm.insertvalue %[[val3]], %[[ins2]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val4:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins4:.*]] = llvm.insertvalue %[[val4]], %[[ins3]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: llvm.return %[[ins4]] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  return %static : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @check_static_return_with_offset
// CHECK-COUNT-2: !llvm.ptr<float>
// CHECK-COUNT-5: !llvm.i64
// CHECK-SAME: -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-LABEL: func @check_static_return_with_offset
// BAREPTR-SAME: (%[[arg:.*]]: !llvm.ptr<float>) -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
func @check_static_return_with_offset(%static : memref<32x18xf32, offset:7, strides:[22,1]>) -> memref<32x18xf32, offset:7, strides:[22,1]> {
// CHECK:  llvm.return %{{.*}} : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

// BAREPTR: %[[udf:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base:.*]] = llvm.insertvalue %[[arg]], %[[udf]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[aligned:.*]] = llvm.insertvalue %[[arg]], %[[base]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val0:.*]] = llvm.mlir.constant(7 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins0:.*]] = llvm.insertvalue %[[val0]], %[[aligned]][2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins1:.*]] = llvm.insertvalue %[[val1]], %[[ins0]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val2:.*]] = llvm.mlir.constant(22 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins2:.*]] = llvm.insertvalue %[[val2]], %[[ins1]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val3:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins3:.*]] = llvm.insertvalue %[[val3]], %[[ins2]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val4:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT: %[[ins4:.*]] = llvm.insertvalue %[[val4]], %[[ins3]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: llvm.return %[[ins4]] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  return %static : memref<32x18xf32, offset:7, strides:[22,1]>
}

// -----

// CHECK-LABEL: func @zero_d_alloc() -> !llvm.struct<(ptr<float>, ptr<float>, i64)> {
// BAREPTR-LABEL: func @zero_d_alloc() -> !llvm.struct<(ptr<float>, ptr<float>, i64)> {
func @zero_d_alloc() -> memref<f32> {
// CHECK-NEXT:  llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
// CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
// CHECK-NEXT:  llvm.mul %{{.*}}, %[[sizeof]] : !llvm.i64
// CHECK-NEXT:  %[[one_1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  llvm.call @malloc(%{{.*}}) : (!llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<float>
// CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// CHECK-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64)>

// BAREPTR-NEXT:  llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
// BAREPTR-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// BAREPTR-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
// BAREPTR-NEXT:  llvm.mul %{{.*}}, %[[sizeof]] : !llvm.i64
// BAREPTR-NEXT:  %[[one_1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT:  llvm.call @malloc(%{{.*}}) : (!llvm.i64) -> !llvm.ptr<i8>
// BAREPTR-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<float>
// BAREPTR-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// BAREPTR-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// BAREPTR-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// BAREPTR-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// BAREPTR-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
  %0 = alloc() : memref<f32>
  return %0 : memref<f32>
}

// -----

// CHECK-LABEL: func @zero_d_dealloc
// BAREPTR-LABEL: func @zero_d_dealloc(%{{.*}}: !llvm.ptr<float>) {
func @zero_d_dealloc(%arg0: memref<f32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// CHECK-NEXT:  %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<float> to !llvm.ptr<i8>
// CHECK-NEXT:  llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()

// BAREPTR: %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// BAREPTR-NEXT: %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<float> to !llvm.ptr<i8>
// BAREPTR-NEXT: llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()
  dealloc %arg0 : memref<f32>
  return
}

// -----

// CHECK-LABEL: func @aligned_1d_alloc(
// BAREPTR-LABEL: func @aligned_1d_alloc(
func @aligned_1d_alloc() -> memref<42xf32> {
//      CHECK:  llvm.mlir.constant(42 : index) : !llvm.i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
// CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
// CHECK-NEXT:  llvm.mul %{{.*}}, %[[sizeof]] : !llvm.i64
// CHECK-NEXT:  %[[one_1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[alignment:.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
// CHECK-NEXT:  %[[alignmentMinus1:.*]] = llvm.add {{.*}}, %[[alignment]] : !llvm.i64
// CHECK-NEXT:  %[[allocsize:.*]] = llvm.sub %[[alignmentMinus1]], %[[one_1]] : !llvm.i64
// CHECK-NEXT:  %[[allocated:.*]] = llvm.call @malloc(%[[allocsize]]) : (!llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<float>
// CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:  %[[allocatedAsInt:.*]] = llvm.ptrtoint %[[allocated]] : !llvm.ptr<i8> to !llvm.i64
// CHECK-NEXT:  %[[alignAdj1:.*]] = llvm.urem %[[allocatedAsInt]], %[[alignment]] : !llvm.i64
// CHECK-NEXT:  %[[alignAdj2:.*]] = llvm.sub %[[alignment]], %[[alignAdj1]] : !llvm.i64
// CHECK-NEXT:  %[[alignAdj3:.*]] = llvm.urem %[[alignAdj2]], %[[alignment]] : !llvm.i64
// CHECK-NEXT:  %[[aligned:.*]] = llvm.getelementptr %[[allocated]][%[[alignAdj3]]] : (!llvm.ptr<i8>, !llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  %[[alignedBitCast:.*]] = llvm.bitcast %[[aligned]] : !llvm.ptr<i8> to !llvm.ptr<float>
// CHECK-NEXT:  llvm.insertvalue %[[alignedBitCast]], %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>

// BAREPTR-NEXT:  llvm.mlir.constant(42 : index) : !llvm.i64
// BAREPTR-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
// BAREPTR-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// BAREPTR-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
// BAREPTR-NEXT:  llvm.mul %{{.*}}, %[[sizeof]] : !llvm.i64
// BAREPTR-NEXT:  %[[one_1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT:  %[[alignment:.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
// BAREPTR-NEXT:  %[[alignmentMinus1:.*]] = llvm.add {{.*}}, %[[alignment]] : !llvm.i64
// BAREPTR-NEXT:  %[[allocsize:.*]] = llvm.sub %[[alignmentMinus1]], %[[one_1]] : !llvm.i64
// BAREPTR-NEXT:  %[[allocated:.*]] = llvm.call @malloc(%[[allocsize]]) : (!llvm.i64) -> !llvm.ptr<i8>
// BAREPTR-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<float>
// BAREPTR-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// BAREPTR-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// BAREPTR-NEXT:  %[[allocatedAsInt:.*]] = llvm.ptrtoint %[[allocated]] : !llvm.ptr<i8> to !llvm.i64
// BAREPTR-NEXT:  %[[alignAdj1:.*]] = llvm.urem %[[allocatedAsInt]], %[[alignment]] : !llvm.i64
// BAREPTR-NEXT:  %[[alignAdj2:.*]] = llvm.sub %[[alignment]], %[[alignAdj1]] : !llvm.i64
// BAREPTR-NEXT:  %[[alignAdj3:.*]] = llvm.urem %[[alignAdj2]], %[[alignment]] : !llvm.i64
// BAREPTR-NEXT:  %[[aligned:.*]] = llvm.getelementptr %[[allocated]][%[[alignAdj3]]] : (!llvm.ptr<i8>, !llvm.i64) -> !llvm.ptr<i8>
// BAREPTR-NEXT:  %[[alignedBitCast:.*]] = llvm.bitcast %[[aligned]] : !llvm.ptr<i8> to !llvm.ptr<float>
// BAREPTR-NEXT:  llvm.insertvalue %[[alignedBitCast]], %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
// BAREPTR-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// BAREPTR-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<1 x i64>, array<1 x i64>)>
  %0 = alloc() {alignment = 8} : memref<42xf32>
  return %0 : memref<42xf32>
}

// -----

// CHECK-LABEL: func @static_alloc() -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
// BAREPTR-LABEL: func @static_alloc() -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
func @static_alloc() -> memref<32x18xf32> {
//      CHECK:  %[[sz1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// CHECK-NEXT:  %[[sz2:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// CHECK-NEXT:  %[[num_elems:.*]] = llvm.mul %0, %1 : !llvm.i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
// CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
// CHECK-NEXT:  %[[bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
// CHECK-NEXT:  %[[one_1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[allocated:.*]] = llvm.call @malloc(%[[bytes]]) : (!llvm.i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  llvm.bitcast %[[allocated]] : !llvm.ptr<i8> to !llvm.ptr<float>

// BAREPTR-NEXT: %[[sz1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// BAREPTR-NEXT: %[[sz2:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// BAREPTR-NEXT: %[[num_elems:.*]] = llvm.mul %[[sz1]], %[[sz2]] : !llvm.i64
// BAREPTR-NEXT: %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
// BAREPTR-NEXT: %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// BAREPTR-NEXT: %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
// BAREPTR-NEXT: %[[bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
// BAREPTR-NEXT:  %[[one_1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT: %[[allocated:.*]] = llvm.call @malloc(%[[bytes]]) : (!llvm.i64) -> !llvm.ptr<i8>
// BAREPTR-NEXT: llvm.bitcast %[[allocated]] : !llvm.ptr<i8> to !llvm.ptr<float>
 %0 = alloc() : memref<32x18xf32>
 return %0 : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @static_alloca() -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
func @static_alloca() -> memref<32x18xf32> {
// CHECK-NEXT:  %[[sz1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// CHECK-NEXT:  %[[sz2:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// CHECK-NEXT:  %[[num_elems:.*]] = llvm.mul %0, %1 : !llvm.i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
// CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
// CHECK-NEXT:  %[[bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
// CHECK-NEXT:  %[[one_1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[allocated:.*]] = llvm.alloca %[[bytes]] x !llvm.float : (!llvm.i64) -> !llvm.ptr<float>
 %0 = alloca() : memref<32x18xf32>

 // Test with explicitly specified alignment. llvm.alloca takes care of the
 // alignment. The same pointer is thus used for allocation and aligned
 // accesses.
 // CHECK: %[[alloca_aligned:.*]] = llvm.alloca %{{.*}} x !llvm.float {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<float>
 // CHECK: %[[desc:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
 // CHECK: %[[desc1:.*]] = llvm.insertvalue %[[alloca_aligned]], %[[desc]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
 // CHECK: llvm.insertvalue %[[alloca_aligned]], %[[desc1]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
 alloca() {alignment = 32} : memref<32x18xf32>
 return %0 : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @static_dealloc
// BAREPTR-LABEL: func @static_dealloc(%{{.*}}: !llvm.ptr<float>) {
func @static_dealloc(%static: memref<10x8xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<float> to !llvm.ptr<i8>
// CHECK-NEXT:  llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<float> to !llvm.ptr<i8>
// BAREPTR-NEXT: llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()
  dealloc %static : memref<10x8xf32>
  return
}

// -----

// CHECK-LABEL: func @zero_d_load
// BAREPTR-LABEL: func @zero_d_load(%{{.*}}: !llvm.ptr<float>) -> !llvm.float
func @zero_d_load(%arg0: memref<f32>) -> f32 {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// CHECK-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[c0]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK-NEXT:  %{{.*}} = llvm.load %[[addr]] : !llvm.ptr<float>

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// BAREPTR-NEXT: %[[c0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// BAREPTR-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[c0]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// BAREPTR-NEXT: llvm.load %[[addr:.*]] : !llvm.ptr<float>
  %0 = load %arg0[] : memref<f32>
  return %0 : f32
}

// -----

// CHECK-LABEL: func @static_load(
// CHECK-COUNT-2: !llvm.ptr<float>,
// CHECK-COUNT-5: {{%[a-zA-Z0-9]*}}: !llvm.i64
// CHECK:         %[[I:.*]]: !llvm.i64,
// CHECK:         %[[J:.*]]: !llvm.i64)
// BAREPTR-LABEL: func @static_load
// BAREPTR-SAME: (%[[A:.*]]: !llvm.ptr<float>, %[[I:.*]]: !llvm.i64, %[[J:.*]]: !llvm.i64) {
func @static_load(%static : memref<10x42xf32>, %i : index, %j : index) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK-NEXT:  llvm.load %[[addr]] : !llvm.ptr<float>

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// BAREPTR-NEXT: %[[st0:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
// BAREPTR-NEXT: %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
// BAREPTR-NEXT: %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
// BAREPTR-NEXT: %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT: %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
// BAREPTR-NEXT: %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
// BAREPTR-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// BAREPTR-NEXT: llvm.load %[[addr]] : !llvm.ptr<float>
  %0 = load %static[%i, %j] : memref<10x42xf32>
  return
}

// -----

// CHECK-LABEL: func @zero_d_store
// BAREPTR-LABEL: func @zero_d_store
// BAREPTR-SAME: (%[[A:.*]]: !llvm.ptr<float>, %[[val:.*]]: !llvm.float)
func @zero_d_store(%arg0: memref<f32>, %arg1: f32) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : !llvm.ptr<float>

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64)>
// BAREPTR-NEXT: %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// BAREPTR-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// BAREPTR-NEXT: llvm.store %[[val]], %[[addr]] : !llvm.ptr<float>
  store %arg1, %arg0[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func @static_store
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[I:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[J:[a-zA-Z0-9]*]]: !llvm.i64
// BAREPTR-LABEL: func @static_store
// BAREPTR-SAME: %[[A:.*]]: !llvm.ptr<float>
// BAREPTR-SAME:         %[[I:[a-zA-Z0-9]*]]: !llvm.i64
// BAREPTR-SAME:         %[[J:[a-zA-Z0-9]*]]: !llvm.i64
func @static_store(%static : memref<10x42xf32>, %i : index, %j : index, %val : f32) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : !llvm.ptr<float>

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// BAREPTR-NEXT: %[[st0:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
// BAREPTR-NEXT: %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
// BAREPTR-NEXT: %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
// BAREPTR-NEXT: %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// BAREPTR-NEXT: %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
// BAREPTR-NEXT: %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
// BAREPTR-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// BAREPTR-NEXT: llvm.store %{{.*}}, %[[addr]] : !llvm.ptr<float>
  store %val, %static[%i, %j] : memref<10x42xf32>
  return
}

// -----

// CHECK-LABEL: func @static_memref_dim
// BAREPTR-LABEL: func @static_memref_dim(%{{.*}}: !llvm.ptr<float>) {
func @static_memref_dim(%static : memref<42x32x15x13x27xf32>) {
// CHECK:        llvm.mlir.constant(42 : index) : !llvm.i64
// BAREPTR:      llvm.insertvalue %{{.*}}, %{{.*}}[4, 4] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<5 x i64>, array<5 x i64>)>
// BAREPTR: llvm.mlir.constant(42 : index) : !llvm.i64
  %c0 = constant 0 : index
  %0 = dim %static, %c0 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(32 : index) : !llvm.i64
// BAREPTR:  llvm.mlir.constant(32 : index) : !llvm.i64
  %c1 = constant 1 : index
  %1 = dim %static, %c1 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(15 : index) : !llvm.i64
// BAREPTR:  llvm.mlir.constant(15 : index) : !llvm.i64
  %c2 = constant 2 : index
  %2 = dim %static, %c2 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(13 : index) : !llvm.i64
// BAREPTR:  llvm.mlir.constant(13 : index) : !llvm.i64
  %c3 = constant 3 : index
  %3 = dim %static, %c3 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(27 : index) : !llvm.i64
// BAREPTR:  llvm.mlir.constant(27 : index) : !llvm.i64
  %c4 = constant 4 : index
  %4 = dim %static, %c4 : memref<42x32x15x13x27xf32>
  return
}
