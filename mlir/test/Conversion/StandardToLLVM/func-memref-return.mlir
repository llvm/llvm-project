// RUN: mlir-opt -convert-std-to-llvm -reconcile-unrealized-casts %s | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1'  -split-input-file %s | FileCheck %s --check-prefix=BAREPTR

// These tests were separated from func-memref.mlir because applying
// -reconcile-unrealized-casts resulted in `llvm.extractvalue` ops getting
// folded away.

// CHECK-LABEL: func @check_static_return
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
// CHECK-SAME: -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-LABEL: func @check_static_return
// BAREPTR-SAME: (%[[arg:.*]]: !llvm.ptr<f32>) -> !llvm.ptr<f32> {
func @check_static_return(%static : memref<32x18xf32>) -> memref<32x18xf32> {
// CHECK:  llvm.return %{{.*}} : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

// BAREPTR: %[[udf:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base0:.*]] = llvm.insertvalue %[[arg]], %[[udf]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[aligned:.*]] = llvm.insertvalue %[[arg]], %[[base0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val0:.*]] = llvm.mlir.constant(0 : index) : i64
// BAREPTR-NEXT: %[[ins0:.*]] = llvm.insertvalue %[[val0]], %[[aligned]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val1:.*]] = llvm.mlir.constant(32 : index) : i64
// BAREPTR-NEXT: %[[ins1:.*]] = llvm.insertvalue %[[val1]], %[[ins0]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val2:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins2:.*]] = llvm.insertvalue %[[val2]], %[[ins1]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val3:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins3:.*]] = llvm.insertvalue %[[val3]], %[[ins2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val4:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT: %[[ins4:.*]] = llvm.insertvalue %[[val4]], %[[ins3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base1:.*]] = llvm.extractvalue %[[ins4]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: llvm.return %[[base1]] : !llvm.ptr<f32>
  return %static : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @check_static_return_with_offset
// CHECK-COUNT-2: !llvm.ptr<f32>
// CHECK-COUNT-5: i64
// CHECK-SAME: -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-LABEL: func @check_static_return_with_offset
// BAREPTR-SAME: (%[[arg:.*]]: !llvm.ptr<f32>) -> !llvm.ptr<f32> {
func @check_static_return_with_offset(%static : memref<32x18xf32, offset:7, strides:[22,1]>) -> memref<32x18xf32, offset:7, strides:[22,1]> {
// CHECK:  llvm.return %{{.*}} : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

// BAREPTR: %[[udf:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base0:.*]] = llvm.insertvalue %[[arg]], %[[udf]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[aligned:.*]] = llvm.insertvalue %[[arg]], %[[base0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val0:.*]] = llvm.mlir.constant(7 : index) : i64
// BAREPTR-NEXT: %[[ins0:.*]] = llvm.insertvalue %[[val0]], %[[aligned]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val1:.*]] = llvm.mlir.constant(32 : index) : i64
// BAREPTR-NEXT: %[[ins1:.*]] = llvm.insertvalue %[[val1]], %[[ins0]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val2:.*]] = llvm.mlir.constant(22 : index) : i64
// BAREPTR-NEXT: %[[ins2:.*]] = llvm.insertvalue %[[val2]], %[[ins1]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val3:.*]] = llvm.mlir.constant(18 : index) : i64
// BAREPTR-NEXT: %[[ins3:.*]] = llvm.insertvalue %[[val3]], %[[ins2]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[val4:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT: %[[ins4:.*]] = llvm.insertvalue %[[val4]], %[[ins3]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[base1:.*]] = llvm.extractvalue %[[ins4]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: llvm.return %[[base1]] : !llvm.ptr<f32>
  return %static : memref<32x18xf32, offset:7, strides:[22,1]>
}

// -----

// BAREPTR: llvm.func @foo(!llvm.ptr<i8>) -> !llvm.ptr<i8>
func private @foo(memref<10xi8>) -> memref<20xi8>

// BAREPTR-LABEL: func @check_memref_func_call
// BAREPTR-SAME:    %[[in:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8>
func @check_memref_func_call(%in : memref<10xi8>) -> memref<20xi8> {
  // BAREPTR:         %[[inDesc:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0]
  // BAREPTR-NEXT:    %[[barePtr:.*]] = llvm.extractvalue %[[inDesc]][1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[call:.*]] = llvm.call @foo(%[[barePtr]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
  // BAREPTR-NEXT:    %[[desc0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[desc1:.*]] = llvm.insertvalue %[[call]], %[[desc0]][0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[desc2:.*]] = llvm.insertvalue %[[call]], %[[desc1]][1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
  // BAREPTR-NEXT:    %[[desc4:.*]] = llvm.insertvalue %[[c0]], %[[desc2]][2] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c20:.*]] = llvm.mlir.constant(20 : index) : i64
  // BAREPTR-NEXT:    %[[desc6:.*]] = llvm.insertvalue %[[c20]], %[[desc4]][3, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
  // BAREPTR-NEXT:    %[[outDesc:.*]] = llvm.insertvalue %[[c1]], %[[desc6]][4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  %res = call @foo(%in) : (memref<10xi8>) -> (memref<20xi8>)
  // BAREPTR-NEXT:    %[[res:.*]] = llvm.extractvalue %[[outDesc]][1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // BAREPTR-NEXT:    llvm.return %[[res]] : !llvm.ptr<i8>
  return %res : memref<20xi8>
}
