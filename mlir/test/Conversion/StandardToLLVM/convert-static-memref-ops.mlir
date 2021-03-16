// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -split-input-file %s | FileCheck %s --check-prefix=BAREPTR

// BAREPTR-LABEL: func @check_noalias
// BAREPTR-SAME: %{{.*}}: !llvm.ptr<f32> {llvm.noalias = true}, %{{.*}}: !llvm.ptr<f32> {llvm.noalias = true}
func @check_noalias(%static : memref<2xf32> {llvm.noalias = true}, %other : memref<2xf32> {llvm.noalias = true}) {
    return
}

// -----

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

// CHECK-LABEL: func @zero_d_alloc() -> !llvm.struct<(ptr<f32>, ptr<f32>, i64)> {
// BAREPTR-LABEL: func @zero_d_alloc() -> !llvm.ptr<f32> {
func @zero_d_alloc() -> memref<f32> {
// CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:  %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// CHECK-NEXT:  llvm.call @malloc(%[[size_bytes]]) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>

// BAREPTR-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// BAREPTR-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// BAREPTR-NEXT:  %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// BAREPTR-NEXT:  llvm.call @malloc(%[[size_bytes]]) : (i64) -> !llvm.ptr<i8>
// BAREPTR-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<f32>
// BAREPTR-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// BAREPTR-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// BAREPTR-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// BAREPTR-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
// BAREPTR-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

// -----

// CHECK-LABEL: func @zero_d_dealloc
// BAREPTR-LABEL: func @zero_d_dealloc(%{{.*}}: !llvm.ptr<f32>) {
func @zero_d_dealloc(%arg0: memref<f32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK-NEXT:  %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK-NEXT:  llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()

// BAREPTR: %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// BAREPTR-NEXT: %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// BAREPTR-NEXT: llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()
  memref.dealloc %arg0 : memref<f32>
  return
}

// -----

// CHECK-LABEL: func @aligned_1d_alloc(
// BAREPTR-LABEL: func @aligned_1d_alloc(
func @aligned_1d_alloc() -> memref<42xf32> {
// CHECK-NEXT:  %[[sz1:.*]] = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:  %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// CHECK-NEXT:  %[[alignment:.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK-NEXT:  %[[allocsize:.*]] = llvm.add %[[size_bytes]], %[[alignment]] : i64
// CHECK-NEXT:  %[[allocated:.*]] = llvm.call @malloc(%[[allocsize]]) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK-NEXT:  %[[allocatedAsInt:.*]] = llvm.ptrtoint %[[ptr]] : !llvm.ptr<f32> to i64
// CHECK-NEXT:  %[[one_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  %[[bump:.*]] = llvm.sub %[[alignment]], %[[one_1]] : i64
// CHECK-NEXT:  %[[bumped:.*]] = llvm.add %[[allocatedAsInt]], %[[bump]] : i64
// CHECK-NEXT:  %[[mod:.*]] = llvm.urem %[[bumped]], %[[alignment]] : i64
// CHECK-NEXT:  %[[aligned:.*]] = llvm.sub %[[bumped]], %[[mod]] : i64
// CHECK-NEXT:  %[[alignedBitCast:.*]] = llvm.inttoptr %[[aligned]] : i64 to !llvm.ptr<f32>
// CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:  llvm.insertvalue %[[alignedBitCast]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>

// BAREPTR-NEXT:  %[[sz1:.*]] = llvm.mlir.constant(42 : index) : i64
// BAREPTR-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// BAREPTR-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// BAREPTR-NEXT:  %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// BAREPTR-NEXT:  %[[alignment:.*]] = llvm.mlir.constant(8 : index) : i64
// BAREPTR-NEXT:  %[[allocsize:.*]] = llvm.add %[[size_bytes]], %[[alignment]] : i64
// BAREPTR-NEXT:  %[[allocated:.*]] = llvm.call @malloc(%[[allocsize]]) : (i64) -> !llvm.ptr<i8>
// BAREPTR-NEXT:  %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<f32>
// BAREPTR-NEXT:  %[[allocatedAsInt:.*]] = llvm.ptrtoint %[[ptr]] : !llvm.ptr<f32> to i64
// BAREPTR-NEXT:  %[[one_2:.*]] = llvm.mlir.constant(1 : index) : i64
// BAREPTR-NEXT:  %[[bump:.*]] = llvm.sub %[[alignment]], %[[one_2]] : i64
// BAREPTR-NEXT:  %[[bumped:.*]] = llvm.add %[[allocatedAsInt]], %[[bump]] : i64
// BAREPTR-NEXT:  %[[mod:.*]] = llvm.urem %[[bumped]], %[[alignment]] : i64
// BAREPTR-NEXT:  %[[aligned:.*]] = llvm.sub %[[bumped]], %[[mod]] : i64
// BAREPTR-NEXT:  %[[alignedBitCast:.*]] = llvm.inttoptr %[[aligned]] : i64 to !llvm.ptr<f32>
// BAREPTR-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// BAREPTR-NEXT:  llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// BAREPTR-NEXT:  llvm.insertvalue %[[alignedBitCast]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// BAREPTR-NEXT:  %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
// BAREPTR-NEXT:  llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %0 = memref.alloc() {alignment = 8} : memref<42xf32>
  return %0 : memref<42xf32>
}

// -----

// CHECK-LABEL: func @static_alloc() -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> {
// BAREPTR-LABEL: func @static_alloc() -> !llvm.ptr<f32> {
func @static_alloc() -> memref<32x18xf32> {
//      CHECK:  %[[num_elems:.*]] = llvm.mlir.constant(576 : index) : i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[num_elems]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:  %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// CHECK-NEXT:  %[[allocated:.*]] = llvm.call @malloc(%[[size_bytes]]) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:  llvm.bitcast %[[allocated]] : !llvm.ptr<i8> to !llvm.ptr<f32>

// BAREPTR:      %[[num_elems:.*]] = llvm.mlir.constant(576 : index) : i64
// BAREPTR-NEXT: %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// BAREPTR-NEXT: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[num_elems]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// BAREPTR-NEXT: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// BAREPTR-NEXT: %[[allocated:.*]] = llvm.call @malloc(%[[size_bytes]]) : (i64) -> !llvm.ptr<i8>
// BAREPTR-NEXT: llvm.bitcast %[[allocated]] : !llvm.ptr<i8> to !llvm.ptr<f32>
 %0 = memref.alloc() : memref<32x18xf32>
 return %0 : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @static_alloca() -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> {
func @static_alloca() -> memref<32x18xf32> {
// CHECK-NEXT:  %[[sz1:.*]] = llvm.mlir.constant(32 : index) : i64
// CHECK-NEXT:  %[[sz2:.*]] = llvm.mlir.constant(18 : index) : i64
// CHECK-NEXT:  %[[st2:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:  %[[num_elems:.*]] = llvm.mlir.constant(576 : index) : i64
// CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[num_elems]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:  %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// CHECK-NEXT:  %[[allocated:.*]] = llvm.alloca %[[size_bytes]] x f32 : (i64) -> !llvm.ptr<f32>
 %0 = memref.alloca() : memref<32x18xf32>

 // Test with explicitly specified alignment. llvm.alloca takes care of the
 // alignment. The same pointer is thus used for allocation and aligned
 // accesses.
 // CHECK: %[[alloca_aligned:.*]] = llvm.alloca %{{.*}} x f32 {alignment = 32 : i64} : (i64) -> !llvm.ptr<f32>
 // CHECK: %[[desc:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
 // CHECK: %[[desc1:.*]] = llvm.insertvalue %[[alloca_aligned]], %[[desc]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
 // CHECK: llvm.insertvalue %[[alloca_aligned]], %[[desc1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
 memref.alloca() {alignment = 32} : memref<32x18xf32>
 return %0 : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @static_dealloc
// BAREPTR-LABEL: func @static_dealloc(%{{.*}}: !llvm.ptr<f32>) {
func @static_dealloc(%static: memref<10x8xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK-NEXT:  llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// BAREPTR-NEXT: llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()
  memref.dealloc %static : memref<10x8xf32>
  return
}

// -----

// CHECK-LABEL: func @zero_d_load
// BAREPTR-LABEL: func @zero_d_load(%{{.*}}: !llvm.ptr<f32>) -> f32
func @zero_d_load(%arg0: memref<f32>) -> f32 {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK-NEXT:  %{{.*}} = llvm.load %[[ptr]] : !llvm.ptr<f32>

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// BAREPTR-NEXT: llvm.load %[[ptr:.*]] : !llvm.ptr<f32>
  %0 = memref.load %arg0[] : memref<f32>
  return %0 : f32
}

// -----

// CHECK-LABEL: func @static_load(
// CHECK-COUNT-2: !llvm.ptr<f32>,
// CHECK-COUNT-5: {{%[a-zA-Z0-9]*}}: i64
// CHECK:         %[[I:.*]]: i64,
// CHECK:         %[[J:.*]]: i64)
// BAREPTR-LABEL: func @static_load
// BAREPTR-SAME: (%[[A:.*]]: !llvm.ptr<f32>, %[[I:.*]]: i64, %[[J:.*]]: i64) {
func @static_load(%static : memref<10x42xf32>, %i : index, %j : index) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mlir.constant(42 : index) : i64
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK-NEXT:  llvm.load %[[addr]] : !llvm.ptr<f32>

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[st0:.*]] = llvm.mlir.constant(42 : index) : i64
// BAREPTR-NEXT: %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
// BAREPTR-NEXT: %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
// BAREPTR-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// BAREPTR-NEXT: llvm.load %[[addr]] : !llvm.ptr<f32>
  %0 = memref.load %static[%i, %j] : memref<10x42xf32>
  return
}

// -----

// CHECK-LABEL: func @zero_d_store
// BAREPTR-LABEL: func @zero_d_store
// BAREPTR-SAME: (%[[A:.*]]: !llvm.ptr<f32>, %[[val:.*]]: f32)
func @zero_d_store(%arg0: memref<f32>, %arg1: f32) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK-NEXT:  llvm.store %{{.*}}, %[[ptr]] : !llvm.ptr<f32>

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// BAREPTR-NEXT: llvm.store %[[val]], %[[ptr]] : !llvm.ptr<f32>
  memref.store %arg1, %arg0[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func @static_store
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<f32>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<f32>
// CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[I:[a-zA-Z0-9]*]]: i64
// CHECK-SAME:         %[[J:[a-zA-Z0-9]*]]: i64
// BAREPTR-LABEL: func @static_store
// BAREPTR-SAME: %[[A:.*]]: !llvm.ptr<f32>
// BAREPTR-SAME:         %[[I:[a-zA-Z0-9]*]]: i64
// BAREPTR-SAME:         %[[J:[a-zA-Z0-9]*]]: i64
func @static_store(%static : memref<10x42xf32>, %i : index, %j : index, %val : f32) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mlir.constant(42 : index) : i64
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : !llvm.ptr<f32>

// BAREPTR:      %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// BAREPTR-NEXT: %[[st0:.*]] = llvm.mlir.constant(42 : index) : i64
// BAREPTR-NEXT: %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
// BAREPTR-NEXT: %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
// BAREPTR-NEXT: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// BAREPTR-NEXT: llvm.store %{{.*}}, %[[addr]] : !llvm.ptr<f32>
  memref.store %val, %static[%i, %j] : memref<10x42xf32>
  return
}

// -----

// CHECK-LABEL: func @static_memref_dim
// BAREPTR-LABEL: func @static_memref_dim(%{{.*}}: !llvm.ptr<f32>) {
func @static_memref_dim(%static : memref<42x32x15x13x27xf32>) {
// CHECK:        llvm.mlir.constant(42 : index) : i64
// BAREPTR:      llvm.insertvalue %{{.*}}, %{{.*}}[4, 4] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<5 x i64>, array<5 x i64>)>
// BAREPTR: llvm.mlir.constant(42 : index) : i64
  %c0 = constant 0 : index
  %0 = memref.dim %static, %c0 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(32 : index) : i64
// BAREPTR:  llvm.mlir.constant(32 : index) : i64
  %c1 = constant 1 : index
  %1 = memref.dim %static, %c1 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(15 : index) : i64
// BAREPTR:  llvm.mlir.constant(15 : index) : i64
  %c2 = constant 2 : index
  %2 = memref.dim %static, %c2 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(13 : index) : i64
// BAREPTR:  llvm.mlir.constant(13 : index) : i64
  %c3 = constant 3 : index
  %3 = memref.dim %static, %c3 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(27 : index) : i64
// BAREPTR:  llvm.mlir.constant(27 : index) : i64
  %c4 = constant 4 : index
  %4 = memref.dim %static, %c4 : memref<42x32x15x13x27xf32>
  return
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

// -----

// BAREPTR: llvm.func @goo(f32) -> f32
func private @goo(f32) -> f32

// BAREPTR-LABEL: func @check_scalar_func_call
// BAREPTR-SAME:    %[[in:.*]]: f32)
func @check_scalar_func_call(%in : f32) {
  // BAREPTR-NEXT:    %[[call:.*]] = llvm.call @goo(%[[in]]) : (f32) -> f32
  %res = call @goo(%in) : (f32) -> (f32)
  return
}

// -----

// Unranked memrefs are currently not supported in the bare-ptr calling
// convention. Check that the conversion to the LLVM-IR dialect doesn't happen
// in the presence of unranked memrefs when using such a calling convention.

// BAREPTR: func private @hoo(memref<*xi8>) -> memref<*xi8>
func private @hoo(memref<*xi8>) -> memref<*xi8>

// BAREPTR-LABEL: func @check_unranked_memref_func_call(%{{.*}}: memref<*xi8>) -> memref<*xi8>
func @check_unranked_memref_func_call(%in: memref<*xi8>) -> memref<*xi8> {
  // BAREPTR-NEXT: call @hoo(%{{.*}}) : (memref<*xi8>) -> memref<*xi8>
  %res = call @hoo(%in) : (memref<*xi8>) -> memref<*xi8>
  // BAREPTR-NEXT: return %{{.*}} : memref<*xi8>
  return %res : memref<*xi8>
}
