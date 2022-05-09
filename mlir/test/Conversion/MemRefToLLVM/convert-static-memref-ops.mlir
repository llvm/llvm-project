// RUN: mlir-opt -convert-memref-to-llvm -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @zero_d_alloc()
func.func @zero_d_alloc() -> memref<f32> {
// CHECK: %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// CHECK: llvm.call @malloc(%[[size_bytes]]) : (i64) -> !llvm.ptr<i8>
// CHECK: %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: llvm.insertvalue %[[ptr]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: unrealized_conversion_cast %{{.*}}

  %0 = memref.alloc() : memref<f32>
  return %0 : memref<f32>
}

// -----

// CHECK-LABEL: func @zero_d_dealloc
func.func @zero_d_dealloc(%arg0: memref<f32>) {
// CHECK: unrealized_conversion_cast
// CHECK: %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK: llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()

  memref.dealloc %arg0 : memref<f32>
  return
}

// -----

// CHECK-LABEL: func @aligned_1d_alloc(
func.func @aligned_1d_alloc() -> memref<42xf32> {
// CHECK: %[[sz1:.*]] = llvm.mlir.constant(42 : index) : i64
// CHECK: %[[st1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// CHECK: %[[alignment:.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK: %[[allocsize:.*]] = llvm.add %[[size_bytes]], %[[alignment]] : i64
// CHECK: %[[allocated:.*]] = llvm.call @malloc(%[[allocsize]]) : (i64) -> !llvm.ptr<i8>
// CHECK: %[[ptr:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK: %[[allocatedAsInt:.*]] = llvm.ptrtoint %[[ptr]] : !llvm.ptr<f32> to i64
// CHECK: %[[one_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[bump:.*]] = llvm.sub %[[alignment]], %[[one_1]] : i64
// CHECK: %[[bumped:.*]] = llvm.add %[[allocatedAsInt]], %[[bump]] : i64
// CHECK: %[[mod:.*]] = llvm.urem %[[bumped]], %[[alignment]] : i64
// CHECK: %[[aligned:.*]] = llvm.sub %[[bumped]], %[[mod]] : i64
// CHECK: %[[alignedBitCast:.*]] = llvm.inttoptr %[[aligned]] : i64 to !llvm.ptr<f32>
// CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: llvm.insertvalue %[[ptr]], %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: llvm.insertvalue %[[alignedBitCast]], %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: llvm.insertvalue %[[c0]], %{{.*}}[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
  %0 = memref.alloc() {alignment = 8} : memref<42xf32>
  return %0 : memref<42xf32>
}

// -----

// CHECK-LABEL: func @static_alloc()
func.func @static_alloc() -> memref<32x18xf32> {
// CHECK: %[[num_elems:.*]] = llvm.mlir.constant(576 : index) : i64
// CHECK: %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[num_elems]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// CHECK: %[[allocated:.*]] = llvm.call @malloc(%[[size_bytes]]) : (i64) -> !llvm.ptr<i8>
// CHECK: llvm.bitcast %[[allocated]] : !llvm.ptr<i8> to !llvm.ptr<f32>
 %0 = memref.alloc() : memref<32x18xf32>
 return %0 : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @static_alloca()
func.func @static_alloca() -> memref<32x18xf32> {
// CHECK: %[[sz1:.*]] = llvm.mlir.constant(32 : index) : i64
// CHECK: %[[sz2:.*]] = llvm.mlir.constant(18 : index) : i64
// CHECK: %[[st2:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[num_elems:.*]] = llvm.mlir.constant(576 : index) : i64
// CHECK: %[[null:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK: %[[gep:.*]] = llvm.getelementptr %[[null]][%[[num_elems]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: %[[size_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<f32> to i64
// CHECK: %[[allocated:.*]] = llvm.alloca %[[size_bytes]] x f32 : (i64) -> !llvm.ptr<f32>
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
func.func @static_dealloc(%static: memref<10x8xf32>) {
// CHECK: %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[bc:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK: llvm.call @free(%[[bc]]) : (!llvm.ptr<i8>) -> ()
  memref.dealloc %static : memref<10x8xf32>
  return
}

// -----

// CHECK-LABEL: func @zero_d_load
func.func @zero_d_load(%arg0: memref<f32>) -> f32 {
// CHECK: %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: %{{.*}} = llvm.load %[[ptr]] : !llvm.ptr<f32>
  %0 = memref.load %arg0[] : memref<f32>
  return %0 : f32
}

// -----

// CHECK-LABEL: func @static_load
// CHECK:         %[[MEMREF:.*]]: memref<10x42xf32>,
// CHECK:         %[[I:.*]]: index,
// CHECK:         %[[J:.*]]: index)
func.func @static_load(%static : memref<10x42xf32>, %i : index, %j : index) {
// CHECK:  %[[II:.*]] = builtin.unrealized_conversion_cast %[[I]]
// CHECK:  %[[JJ:.*]] = builtin.unrealized_conversion_cast %[[J]]
// CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:  %[[st0:.*]] = llvm.mlir.constant(42 : index) : i64
// CHECK:  %[[offI:.*]] = llvm.mul %[[II]], %[[st0]] : i64
// CHECK:  %[[off1:.*]] = llvm.add %[[offI]], %[[JJ]] : i64
// CHECK:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK:  llvm.load %[[addr]] : !llvm.ptr<f32>
  %0 = memref.load %static[%i, %j] : memref<10x42xf32>
  return
}

// -----

// CHECK-LABEL: func @zero_d_store
func.func @zero_d_store(%arg0: memref<f32>, %arg1: f32) {
// CHECK: %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
// CHECK: llvm.store %{{.*}}, %[[ptr]] : !llvm.ptr<f32>
  memref.store %arg1, %arg0[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func @static_store
// CHECK:         %[[MEMREF:.*]]: memref<10x42xf32>,
// CHECK-SAME:    %[[I:.*]]: index, %[[J:.*]]: index,
func.func @static_store(%static : memref<10x42xf32>, %i : index, %j : index, %val : f32) {
// CHECK: %[[II:.*]] = builtin.unrealized_conversion_cast %[[I]]
// CHECK: %[[JJ:.*]] = builtin.unrealized_conversion_cast %[[J]]
// CHECK: %[[ptr:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[st0:.*]] = llvm.mlir.constant(42 : index) : i64
// CHECK: %[[offI:.*]] = llvm.mul %[[II]], %[[st0]] : i64
// CHECK: %[[off1:.*]] = llvm.add %[[offI]], %[[JJ]] : i64
// CHECK: %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: llvm.store %{{.*}}, %[[addr]] : !llvm.ptr<f32>

  memref.store %val, %static[%i, %j] : memref<10x42xf32>
  return
}

// -----

// CHECK-LABEL: func @static_memref_dim
func.func @static_memref_dim(%static : memref<42x32x15x13x27xf32>) {
// CHECK:  llvm.mlir.constant(42 : index) : i64
  %c0 = arith.constant 0 : index
  %0 = memref.dim %static, %c0 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(32 : index) : i64
  %c1 = arith.constant 1 : index
  %1 = memref.dim %static, %c1 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(15 : index) : i64
  %c2 = arith.constant 2 : index
  %2 = memref.dim %static, %c2 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(13 : index) : i64
  %c3 = arith.constant 3 : index
  %3 = memref.dim %static, %c3 : memref<42x32x15x13x27xf32>
// CHECK:  llvm.mlir.constant(27 : index) : i64
  %c4 = arith.constant 4 : index
  %4 = memref.dim %static, %c4 : memref<42x32x15x13x27xf32>
  return
}

// -----

// Check that consistent types are emitted in address arithemic in presence of
// a data layout specification.
module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {
  func.func @address() {
    %c1 = arith.constant 1 : index
    %0 = memref.alloc(%c1) : memref<? x vector<2xf32>>
    // CHECK: %[[CST_S:.*]] = arith.constant 1 : index
    // CHECK: %[[CST:.*]] = builtin.unrealized_conversion_cast
    // CHECK: llvm.mlir.null
    // CHECK: llvm.getelementptr %{{.*}}[[CST]]
    // CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<{{.*}}> to i32
    // CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<{{.*}}> to i32
    // CHECK: llvm.add %{{.*}} : i32
    // CHECK: llvm.call @malloc(%{{.*}}) : (i32) -> !llvm.ptr
    // CHECK: llvm.ptrtoint %{{.*}} : !llvm.ptr<{{.*}}> to i32
    // CHECK: llvm.sub {{.*}} : i32
    // CHECK: llvm.add {{.*}} : i32
    // CHECK: llvm.urem {{.*}} : i32
    // CHECK: llvm.sub {{.*}} : i32
    // CHECK: llvm.inttoptr %{{.*}} : i32 to !llvm.ptr
    return
  }
}

