// RUN: mlir-opt -split-input-file -finalize-memref-to-llvm='use-opaque-pointers=1' %s | FileCheck %s
// RUN: mlir-opt -split-input-file -finalize-memref-to-llvm='use-aligned-alloc=1 use-opaque-pointers=1' %s | FileCheck %s --check-prefix=ALIGNED-ALLOC
// RUN: mlir-opt -split-input-file -finalize-memref-to-llvm='index-bitwidth=32 use-opaque-pointers=1' %s | FileCheck --check-prefix=CHECK32 %s

// CHECK-LABEL: func @mixed_alloc(
//       CHECK:   %[[Marg:.*]]: index, %[[Narg:.*]]: index)
func.func @mixed_alloc(%arg0: index, %arg1: index) -> memref<?x42x?xf32> {
//   CHECK-DAG:  %[[M:.*]] = builtin.unrealized_conversion_cast %[[Marg]]
//   CHECK-DAG:  %[[N:.*]] = builtin.unrealized_conversion_cast %[[Narg]]
//       CHECK:  %[[c42:.*]] = llvm.mlir.constant(42 : index) : i64
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mul %[[N]], %[[c42]] : i64
//  CHECK-NEXT:  %[[sz:.*]] = llvm.mul %[[st0]], %[[M]] : i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
//  CHECK-NEXT:  llvm.call @malloc(%[[sz_bytes]]) : (i64) -> !llvm.ptr
//  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[c42]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st0]], %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[one]], %{{.*}}[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  %0 = memref.alloc(%arg0, %arg1) : memref<?x42x?xf32>
  return %0 : memref<?x42x?xf32>
}

// -----

// CHECK-LABEL: func @mixed_dealloc
func.func @mixed_dealloc(%arg0: memref<?x42x?xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:  llvm.call @free(%[[ptr]]) : (!llvm.ptr) -> ()
  memref.dealloc %arg0 : memref<?x42x?xf32>
  return
}

// -----

// CHECK-LABEL: func @unranked_dealloc
func.func @unranked_dealloc(%arg0: memref<*xf32>) {
//      CHECK: %[[memref:.*]] = llvm.extractvalue %{{.*}} : !llvm.struct<(i64, ptr)>
//      CHECK: %[[ptr:.*]] = llvm.load %[[memref]]
// CHECK-NEXT: llvm.call @free(%[[ptr]])
  memref.dealloc %arg0 : memref<*xf32>
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
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[sz]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
//  CHECK-NEXT:  llvm.call @malloc(%[[sz_bytes]]) : (i64) -> !llvm.ptr
//  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[one]], %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_alloca
// CHECK: %[[Marg:.*]]: index, %[[Narg:.*]]: index)
func.func @dynamic_alloca(%arg0: index, %arg1: index) -> memref<?x?xf32> {
//   CHECK-DAG:  %[[M:.*]] = builtin.unrealized_conversion_cast %[[Marg]]
//   CHECK-DAG:  %[[N:.*]] = builtin.unrealized_conversion_cast %[[Narg]]
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : i64
//  CHECK-NEXT:  %[[num_elems:.*]] = llvm.mul %[[N]], %[[M]] : i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[num_elems]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
//  CHECK-NEXT:  %[[allocated:.*]] = llvm.alloca %[[sz_bytes]] x f32 : (i64) -> !llvm.ptr
//  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[allocated]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[allocated]], %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st1]], %{{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %0 = memref.alloca(%arg0, %arg1) : memref<?x?xf32>

// Test with explicitly specified alignment. llvm.alloca takes care of the
// alignment. The same pointer is thus used for allocation and aligned
// accesses.
// CHECK: %[[alloca_aligned:.*]] = llvm.alloca %{{.*}} x f32 {alignment = 32 : i64} : (i64) -> !llvm.ptr
// CHECK: %[[desc:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[desc1:.*]] = llvm.insertvalue %[[alloca_aligned]], %[[desc]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.insertvalue %[[alloca_aligned]], %[[desc1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  memref.alloca(%arg0, %arg1) {alignment = 32} : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_dealloc
func.func @dynamic_dealloc(%arg0: memref<?x?xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  llvm.call @free(%[[ptr]]) : (!llvm.ptr) -> ()
  memref.dealloc %arg0 : memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @stdlib_aligned_alloc({{.*}})
// ALIGNED-ALLOC-LABEL: func @stdlib_aligned_alloc({{.*}})
func.func @stdlib_aligned_alloc(%N : index) -> memref<32x18xf32> {
// ALIGNED-ALLOC:       %[[sz1:.*]] = llvm.mlir.constant(32 : index) : i64
// ALIGNED-ALLOC-NEXT:  %[[sz2:.*]] = llvm.mlir.constant(18 : index) : i64
// ALIGNED-ALLOC-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : i64
// ALIGNED-ALLOC-NEXT:  %[[num_elems:.*]] = llvm.mlir.constant(576 : index) : i64
// ALIGNED-ALLOC-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr
// ALIGNED-ALLOC-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[num_elems]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// ALIGNED-ALLOC-NEXT:  %[[bytes:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr to i64
// ALIGNED-ALLOC-NEXT:  %[[alignment:.*]] = llvm.mlir.constant(32 : index) : i64
// ALIGNED-ALLOC-NEXT:  %[[allocated:.*]] = llvm.call @aligned_alloc(%[[alignment]], %[[bytes]]) : (i64, i64) -> !llvm.ptr
  %0 = memref.alloc() {alignment = 32} : memref<32x18xf32>
  // Do another alloc just to test that we have a unique declaration for
  // aligned_alloc.
  // ALIGNED-ALLOC:  llvm.call @aligned_alloc
  %1 = memref.alloc() {alignment = 64} : memref<4096xf32>

  // Alignment is to element type boundaries (minimum 16 bytes).
  // ALIGNED-ALLOC:  %[[c32:.*]] = llvm.mlir.constant(32 : index) : i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c32]]
  %2 = memref.alloc() : memref<4096xvector<8xf32>>
  // The minimum alignment is 16 bytes unless explicitly specified.
  // ALIGNED-ALLOC:  %[[c16:.*]] = llvm.mlir.constant(16 : index) : i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c16]],
  %3 = memref.alloc() : memref<4096xvector<2xf32>>
  // ALIGNED-ALLOC:  %[[c8:.*]] = llvm.mlir.constant(8 : index) : i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c8]],
  %4 = memref.alloc() {alignment = 8} : memref<1024xvector<4xf32>>
  // Bump the memref allocation size if its size is not a multiple of alignment.
  // ALIGNED-ALLOC:       %[[c32:.*]] = llvm.mlir.constant(32 : index) : i64
  // ALIGNED-ALLOC:       llvm.mlir.constant(1 : index) : i64
  // ALIGNED-ALLOC-NEXT:  llvm.sub
  // ALIGNED-ALLOC-NEXT:  llvm.add
  // ALIGNED-ALLOC-NEXT:  llvm.urem
  // ALIGNED-ALLOC-NEXT:  %[[SIZE_ALIGNED:.*]] = llvm.sub
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c32]], %[[SIZE_ALIGNED]])
  %5 = memref.alloc() {alignment = 32} : memref<100xf32>
  // Bump alignment to the next power of two if it isn't.
  // ALIGNED-ALLOC:  %[[c128:.*]] = llvm.mlir.constant(128 : index) : i64
  // ALIGNED-ALLOC:  llvm.call @aligned_alloc(%[[c128]]
  %6 = memref.alloc(%N) : memref<?xvector<18xf32>>
  return %0 : memref<32x18xf32>
}

// -----

// CHECK-LABEL: func @mixed_load(
// CHECK:         %{{.*}}, %[[Iarg:.*]]: index, %[[Jarg:.*]]: index)
func.func @mixed_load(%mixed : memref<42x?xf32>, %i : index, %j : index) {
//   CHECK-DAG:  %[[I:.*]] = builtin.unrealized_conversion_cast %[[Iarg]]
//   CHECK-DAG:  %[[J:.*]] = builtin.unrealized_conversion_cast %[[Jarg]]
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//  CHECK-NEXT:  llvm.load %[[addr]] : !llvm.ptr -> f32
  %0 = memref.load %mixed[%i, %j] : memref<42x?xf32>
  return
}

// -----

// CHECK-LABEL: func @dynamic_load(
// CHECK:         %{{.*}}, %[[Iarg:.*]]: index, %[[Jarg:.*]]: index)
func.func @dynamic_load(%dynamic : memref<?x?xf32>, %i : index, %j : index) {
//   CHECK-DAG:  %[[I:.*]] = builtin.unrealized_conversion_cast %[[Iarg]]
//   CHECK-DAG:  %[[J:.*]] = builtin.unrealized_conversion_cast %[[Jarg]]
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//  CHECK-NEXT:  llvm.load %[[addr]] : !llvm.ptr -> f32
  %0 = memref.load %dynamic[%i, %j] : memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @prefetch
// CHECK:         %{{.*}}, %[[Iarg:.*]]: index, %[[Jarg:.*]]: index)
func.func @prefetch(%A : memref<?x?xf32>, %i : index, %j : index) {
//      CHECK-DAG:  %[[I:.*]] = builtin.unrealized_conversion_cast %[[Iarg]]
//      CHECK-DAG:  %[[J:.*]] = builtin.unrealized_conversion_cast %[[Jarg]]
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
// CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
// CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:  [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:  [[C3:%.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT:  [[C1_1:%.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:  "llvm.intr.prefetch"(%[[addr]], [[C1]], [[C3]], [[C1_1]]) : (!llvm.ptr, i32, i32, i32) -> ()
  memref.prefetch %A[%i, %j], write, locality<3>, data : memref<?x?xf32>
// CHECK:  [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:  [[C0_1:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:  [[C1_2:%.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:  "llvm.intr.prefetch"(%{{.*}}, [[C0]], [[C0_1]], [[C1_2]]) : (!llvm.ptr, i32, i32, i32) -> ()
  memref.prefetch %A[%i, %j], read, locality<0>, data : memref<?x?xf32>
// CHECK:  [[C0_2:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:  [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:  [[C0_3:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:  "llvm.intr.prefetch"(%{{.*}}, [[C0_2]], [[C2]], [[C0_3]]) : (!llvm.ptr, i32, i32, i32) -> ()
  memref.prefetch %A[%i, %j], read, locality<2>, instr : memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @dynamic_store
// CHECK:         %{{.*}}, %[[Iarg:.*]]: index, %[[Jarg:.*]]: index
func.func @dynamic_store(%dynamic : memref<?x?xf32>, %i : index, %j : index, %val : f32) {
//   CHECK-DAG:  %[[I:.*]] = builtin.unrealized_conversion_cast %[[Iarg]]
//   CHECK-DAG:  %[[J:.*]] = builtin.unrealized_conversion_cast %[[Jarg]]
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : f32, !llvm.ptr
  memref.store %val, %dynamic[%i, %j] : memref<?x?xf32>
  return
}

// -----

// CHECK-LABEL: func @mixed_store
// CHECK:         %{{.*}}, %[[Iarg:.*]]: index, %[[Jarg:.*]]: index
func.func @mixed_store(%mixed : memref<42x?xf32>, %i : index, %j : index, %val : f32) {
//   CHECK-DAG:  %[[I:.*]] = builtin.unrealized_conversion_cast %[[Iarg]]
//   CHECK-DAG:  %[[J:.*]] = builtin.unrealized_conversion_cast %[[Jarg]]
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[offI]], %[[J]] : i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : f32, !llvm.ptr
  memref.store %val, %mixed[%i, %j] : memref<42x?xf32>
  return
}

// -----

// FIXME: the *ToLLVM passes don't use information from data layouts
// to set address spaces, so the constants below don't reflect the layout
// Update this test once that data layout attribute works how we'd expect it to.
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr, dense<[64, 64, 64]> : vector<3xi32>>,
  #dlti.dl_entry<!llvm.ptr<1>, dense<[32, 32, 32]> : vector<3xi32>>> }  {
  // CHECK-LABEL: @memref_memory_space_cast
  func.func @memref_memory_space_cast(%input : memref<*xf32>) -> memref<*xf32, 1> {
    %cast = memref.memory_space_cast %input : memref<*xf32> to memref<*xf32, 1>
    return %cast : memref<*xf32, 1>
  }
}
// CHECK: [[INPUT:%.*]] = builtin.unrealized_conversion_cast %{{.*}} to !llvm.struct<(i64, ptr)>
// CHECK: [[RANK:%.*]] = llvm.extractvalue [[INPUT]][0] : !llvm.struct<(i64, ptr)>
// CHECK: [[SOURCE_DESC:%.*]] = llvm.extractvalue [[INPUT]][1]
// CHECK: [[RESULT_0:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: [[RESULT_1:%.*]] = llvm.insertvalue [[RANK]], [[RESULT_0]][0] : !llvm.struct<(i64, ptr)>

// Compute size in bytes to allocate result ranked descriptor
// CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[C2:%.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK: [[INDEX_SIZE:%.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK: [[PTR_SIZE:%.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK: [[DOUBLE_PTR_SIZE:%.*]] = llvm.mul [[C2]], [[PTR_SIZE]]
// CHECK: [[DOUBLE_RANK:%.*]] = llvm.mul [[C2]], %{{.*}}
// CHECK: [[NUM_INDEX_VALS:%.*]] = llvm.add [[DOUBLE_RANK]], [[C1]]
// CHECK: [[INDEX_VALS_SIZE:%.*]] = llvm.mul [[NUM_INDEX_VALS]], [[INDEX_SIZE]]
// CHECK: [[DESC_ALLOC_SIZE:%.*]] = llvm.add [[DOUBLE_PTR_SIZE]], [[INDEX_VALS_SIZE]]
// CHECK: [[RESULT_DESC:%.*]] = llvm.alloca [[DESC_ALLOC_SIZE]] x i8
// CHECK: llvm.insertvalue [[RESULT_DESC]], [[RESULT_1]][1]

// Cast pointers
// CHECK: [[SOURCE_ALLOC:%.*]] = llvm.load [[SOURCE_DESC]]
// CHECK: [[SOURCE_ALIGN_GEP:%.*]] = llvm.getelementptr [[SOURCE_DESC]][1]
// CHECK: [[SOURCE_ALIGN:%.*]] = llvm.load [[SOURCE_ALIGN_GEP]] : !llvm.ptr
// CHECK: [[RESULT_ALLOC:%.*]] = llvm.addrspacecast [[SOURCE_ALLOC]] : !llvm.ptr to !llvm.ptr<1>
// CHECK: [[RESULT_ALIGN:%.*]] = llvm.addrspacecast [[SOURCE_ALIGN]] : !llvm.ptr to !llvm.ptr<1>
// CHECK: llvm.store [[RESULT_ALLOC]], [[RESULT_DESC]] : !llvm.ptr
// CHECK: [[RESULT_ALIGN_GEP:%.*]] = llvm.getelementptr [[RESULT_DESC]][1]
// CHECK: llvm.store [[RESULT_ALIGN]], [[RESULT_ALIGN_GEP]] : !llvm.ptr

// Memcpy remaniing values

// CHECK: [[SOURCE_OFFSET_GEP:%.*]] = llvm.getelementptr [[SOURCE_DESC]][2]
// CHECK: [[RESULT_OFFSET_GEP:%.*]] = llvm.getelementptr [[RESULT_DESC]][2]
// CHECK: [[SIZEOF_TWO_RESULT_PTRS:%.*]] = llvm.mlir.constant(16 : index) : i64
// CHECK: [[COPY_SIZE:%.*]] = llvm.sub [[DESC_ALLOC_SIZE]], [[SIZEOF_TWO_RESULT_PTRS]]
// CHECK: [[FALSE:%.*]] = llvm.mlir.constant(false) : i1
// CHECK: "llvm.intr.memcpy"([[RESULT_OFFSET_GEP]], [[SOURCE_OFFSET_GEP]], [[COPY_SIZE]], [[FALSE]])

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
// CHECK-DAG:  %[[p:.*]] = llvm.alloca %[[c]] x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
// CHECK-DAG:  llvm.store %{{.*}}, %[[p]] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
// CHECK-DAG:  %[[r:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK    :  llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK-DAG:  llvm.insertvalue %[[r]], %{{.*}}[0] : !llvm.struct<(i64, ptr)>
// CHECK-DAG:  llvm.insertvalue %[[p]], %{{.*}}[1] : !llvm.struct<(i64, ptr)>
// CHECK32-DAG:  %[[c:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK32-DAG:  %[[p:.*]] = llvm.alloca %[[c]] x !llvm.struct<(ptr, ptr, i32, array<3 x i32>, array<3 x i32>)> : (i64) -> !llvm.ptr
// CHECK32-DAG:  llvm.store %{{.*}}, %[[p]] : !llvm.struct<(ptr, ptr, i32, array<3 x i32>, array<3 x i32>)>, !llvm.ptr
// CHECK32-DAG:  %[[r:.*]] = llvm.mlir.constant(3 : index) : i32
// CHECK32    :  llvm.mlir.undef : !llvm.struct<(i32, ptr)>
// CHECK32-DAG:  llvm.insertvalue %[[r]], %{{.*}}[0] : !llvm.struct<(i32, ptr)>
// CHECK32-DAG:  llvm.insertvalue %[[p]], %{{.*}}[1] : !llvm.struct<(i32, ptr)>
  %0 = memref.cast %arg : memref<42x2x?xf32> to memref<*xf32>
  return
}

// -----

// CHECK-LABEL: func @memref_cast_unranked_to_ranked
func.func @memref_cast_unranked_to_ranked(%arg : memref<*xf32>) {
//      CHECK: %[[p:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i64, ptr)>
  %0 = memref.cast %arg : memref<*xf32> to memref<?x?x10x2xf32>
  return
}

// -----

// CHECK-LABEL: func @mixed_memref_dim
func.func @mixed_memref_dim(%mixed : memref<42x?x?x13x?xf32>) {
// CHECK: llvm.mlir.constant(42 : index) : i64
  %c0 = arith.constant 0 : index
  %0 = memref.dim %mixed, %c0 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %{{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)>
  %c1 = arith.constant 1 : index
  %1 = memref.dim %mixed, %c1 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %{{.*}}[3, 2] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)>
  %c2 = arith.constant 2 : index
  %2 = memref.dim %mixed, %c2 : memref<42x?x?x13x?xf32>
// CHECK: llvm.mlir.constant(13 : index) : i64
  %c3 = arith.constant 3 : index
  %3 = memref.dim %mixed, %c3 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %{{.*}}[3, 4] : !llvm.struct<(ptr, ptr, i64, array<5 x i64>, array<5 x i64>)>
  %c4 = arith.constant 4 : index
  %4 = memref.dim %mixed, %c4 : memref<42x?x?x13x?xf32>
  return
}

// -----

// CHECK-LABEL: @memref_dim_with_dyn_index
// CHECK: %{{.*}}, %[[IDXarg:.*]]: index
func.func @memref_dim_with_dyn_index(%arg : memref<3x?xf32>, %idx : index) -> index {
  // CHECK-DAG: %[[IDX:.*]] = builtin.unrealized_conversion_cast %[[IDXarg]]
  // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
  // CHECK-DAG: %[[SIZES:.*]] = llvm.extractvalue %{{.*}}[3] : ![[DESCR_TY:.*]]
  // CHECK-DAG: %[[SIZES_PTR:.*]] = llvm.alloca %[[C1]] x !llvm.array<2 x i64> : (i64) -> !llvm.ptr
  // CHECK-DAG: llvm.store %[[SIZES]], %[[SIZES_PTR]] : !llvm.array<2 x i64>, !llvm.ptr
  // CHECK-DAG: %[[RESULT_PTR:.*]] = llvm.getelementptr %[[SIZES_PTR]][0, %[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x i64>
  // CHECK-DAG: %[[RESULT:.*]] = llvm.load %[[RESULT_PTR]] : !llvm.ptr -> i64
  %result = memref.dim %arg, %idx : memref<3x?xf32>
  return %result : index
}

// -----

// CHECK-LABEL: @memref_reinterpret_cast_ranked_to_static_shape
func.func @memref_reinterpret_cast_ranked_to_static_shape(%input : memref<2x3xf32>) {
  %output = memref.reinterpret_cast %input to
           offset: [0], sizes: [6, 1], strides: [1, 1]
           : memref<2x3xf32> to memref<6x1xf32>
  return
}
// CHECK: [[INPUT:%.*]] = builtin.unrealized_conversion_cast %{{.*}} :
// CHECK: to [[TY:!.*]]
// CHECK: [[OUT_0:%.*]] = llvm.mlir.undef : [[TY]]
// CHECK: [[BASE_PTR:%.*]] = llvm.extractvalue [[INPUT]][0] : [[TY]]
// CHECK: [[ALIGNED_PTR:%.*]] = llvm.extractvalue [[INPUT]][1] : [[TY]]
// CHECK: [[OUT_1:%.*]] = llvm.insertvalue [[BASE_PTR]], [[OUT_0]][0] : [[TY]]
// CHECK: [[OUT_2:%.*]] = llvm.insertvalue [[ALIGNED_PTR]], [[OUT_1]][1] : [[TY]]
// CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: [[OUT_3:%.*]] = llvm.insertvalue [[OFFSET]], [[OUT_2]][2] : [[TY]]
// CHECK: [[SIZE_0:%.*]] = llvm.mlir.constant(6 : index) : i64
// CHECK: [[OUT_4:%.*]] = llvm.insertvalue [[SIZE_0]], [[OUT_3]][3, 0] : [[TY]]
// CHECK: [[SIZE_1:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[OUT_5:%.*]] = llvm.insertvalue [[SIZE_1]], [[OUT_4]][4, 0] : [[TY]]
// CHECK: [[STRIDE_0:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[OUT_6:%.*]] = llvm.insertvalue [[STRIDE_0]], [[OUT_5]][3, 1] : [[TY]]
// CHECK: [[STRIDE_1:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[OUT_7:%.*]] = llvm.insertvalue [[STRIDE_1]], [[OUT_6]][4, 1] : [[TY]]

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
// CHECK: [[DESCRIPTOR:%.*]] = llvm.extractvalue [[INPUT]][1] : !llvm.struct<(i64, ptr)>
// CHECK: [[BASE_PTR:%.*]] = llvm.load [[DESCRIPTOR]] : !llvm.ptr -> !llvm.ptr
// CHECK: [[ALIGNED_PTR_PTR:%.*]] = llvm.getelementptr [[DESCRIPTOR]]{{\[}}1]
// CHECK-SAME: : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK: [[ALIGNED_PTR:%.*]] = llvm.load [[ALIGNED_PTR_PTR]] : !llvm.ptr -> !llvm.ptr
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
// CHECK: [[UNRANKED_OUT_O:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: [[UNRANKED_OUT_1:%.*]] = llvm.insertvalue [[RANK]], [[UNRANKED_OUT_O]][0] : !llvm.struct<(i64, ptr)>

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
// CHECK: llvm.store [[ALLOC_PTR]], [[UNDERLYING_DESC]] : !llvm.ptr, !llvm.ptr
// CHECK: [[ALIGNED_PTR_PTR:%.*]] = llvm.getelementptr [[UNDERLYING_DESC]]{{\[}}1]
// CHECK: llvm.store [[ALIGN_PTR]], [[ALIGNED_PTR_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: [[OFFSET_PTR:%.*]] = llvm.getelementptr [[UNDERLYING_DESC]]{{\[}}2]
// CHECK: llvm.store [[OFFSET]], [[OFFSET_PTR]] : i64, !llvm.ptr

// Iterate over shape operand in reverse order and set sizes and strides.
// CHECK: [[SIZES_PTR:%.*]] = llvm.getelementptr [[UNDERLYING_DESC]]{{\[}}0, 3]
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
// CHECK:   [[SIZE:%.*]] = llvm.load [[SIZE_PTR]] : !llvm.ptr -> i64
// CHECK:   [[TARGET_SIZE_PTR:%.*]] = llvm.getelementptr [[SIZES_PTR]]{{\[}}[[DIM]]]
// CHECK:   llvm.store [[SIZE]], [[TARGET_SIZE_PTR]] : i64, !llvm.ptr
// CHECK:   [[TARGET_STRIDE_PTR:%.*]] = llvm.getelementptr [[STRIDES_PTR]]{{\[}}[[DIM]]]
// CHECK:   llvm.store [[CUR_STRIDE]], [[TARGET_STRIDE_PTR]] : i64, !llvm.ptr
// CHECK:   [[UPDATE_STRIDE:%.*]] = llvm.mul [[CUR_STRIDE]], [[SIZE]] : i64
// CHECK:   [[STRIDE_COND:%.*]] = llvm.sub [[DIM]], [[C1_]] : i64
// CHECK:   llvm.br ^bb1([[STRIDE_COND]], [[UPDATE_STRIDE]] : i64, i64)

// CHECK: ^bb3:
// CHECK:   return

// -----

// ALIGNED-ALLOC-LABEL: @memref_of_memref
func.func @memref_of_memref() {
  // Sizeof computation is as usual.
  // ALIGNED-ALLOC: %[[NULL:.*]] = llvm.mlir.null
  // ALIGNED-ALLOC: %[[PTR:.*]] = llvm.getelementptr
  // ALIGNED-ALLOC: %[[SIZEOF:.*]] = llvm.ptrtoint

  // Static alignment should be computed as ceilPowerOf2(2 * sizeof(pointer) +
  // (1 + 2 * rank) * sizeof(index) = ceilPowerOf2(2 * 8 + 3 * 8) = 64.
  // ALIGNED-ALLOC: llvm.mlir.constant(64 : index)

  // Check that the types are converted as expected.
  // ALIGNED-ALLOC: llvm.call @aligned_alloc
  // ALIGNED-ALLOC: llvm.mlir.undef
  // ALIGNED-ALLOC-SAME: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %0 = memref.alloc() : memref<1xmemref<1xf32>>
  return
}

// -----

module attributes { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>> } {
  // ALIGNED-ALLOC-LABEL: @memref_of_memref_32
  func.func @memref_of_memref_32() {
    // Sizeof computation is as usual.
    // ALIGNED-ALLOC: %[[NULL:.*]] = llvm.mlir.null
    // ALIGNED-ALLOC: %[[PTR:.*]] = llvm.getelementptr
    // ALIGNED-ALLOC: %[[SIZEOF:.*]] = llvm.ptrtoint

    // Static alignment should be computed as ceilPowerOf2(2 * sizeof(pointer) +
    // (1 + 2 * rank) * sizeof(index) = ceilPowerOf2(2 * 8 + 3 * 4) = 32.
    // ALIGNED-ALLOC: llvm.mlir.constant(32 : index)

    // Check that the types are converted as expected.
    // ALIGNED-ALLOC: llvm.call @aligned_alloc
    // ALIGNED-ALLOC: llvm.mlir.undef
    // ALIGNED-ALLOC-SAME: !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
    %0 = memref.alloc() : memref<1xmemref<1xf32>>
    return
  }
}


// -----

// ALIGNED-ALLOC-LABEL: @memref_of_memref_of_memref
func.func @memref_of_memref_of_memref() {
  // Sizeof computation is as usual, also check the type.
  // ALIGNED-ALLOC: %[[NULL:.*]] = llvm.mlir.null : !llvm.ptr
  // ALIGNED-ALLOC: %[[PTR:.*]] = llvm.getelementptr
  // ALIGNED-ALLOC: %[[SIZEOF:.*]] = llvm.ptrtoint

  // Static alignment should be computed as ceilPowerOf2(2 * sizeof(pointer) +
  // (1 + 2 * rank) * sizeof(index) = ceilPowerOf2(2 * 8 + 3 * 8) = 64.
  // ALIGNED-ALLOC: llvm.mlir.constant(64 : index)
  // ALIGNED-ALLOC: llvm.call @aligned_alloc
  %0 = memref.alloc() : memref<1 x memref<2 x memref<3 x f32>>>
  return
}

// -----

// ALIGNED-ALLOC-LABEL: @ranked_unranked
func.func @ranked_unranked() {
  // ALIGNED-ALLOC: llvm.mlir.null
  // ALIGNED-ALLOC-SAME: !llvm.ptr
  // ALIGNED-ALLOC: llvm.getelementptr
  // ALIGNED-ALLOC: llvm.ptrtoint

  // Static alignment should be computed as ceilPowerOf2(sizeof(index) +
  // sizeof(pointer)) = 16.
  // ALIGNED-ALLOC: llvm.mlir.constant(16 : index)
  // ALIGNED-ALLOC: llvm.call @aligned_alloc
  %0 = memref.alloc() : memref<1 x memref<* x f32>>
  memref.cast %0 : memref<1 x memref<* x f32>> to memref<* x memref<* x f32>>
  return
}

// -----

// CHECK-LABEL:     func.func @realloc_dynamic(
// CHECK-SAME:      %[[arg0:.*]]: memref<?xf32>,
// CHECK-SAME:      %[[arg1:.*]]: index) -> memref<?xf32> {
func.func @realloc_dynamic(%in: memref<?xf32>, %d: index) -> memref<?xf32>{
// CHECK:           %[[descriptor:.*]] = builtin.unrealized_conversion_cast %[[arg0]]
// CHECK:           %[[src_dim:.*]] = llvm.extractvalue %[[descriptor]][3, 0]
// CHECK:           %[[dst_dim:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i64
// CHECK:           %[[cond:.*]] = llvm.icmp "ugt" %[[dst_dim]], %[[src_dim]] : i64
// CHECK:           llvm.cond_br %[[cond]], ^bb1, ^bb2(%[[descriptor]]
// CHECK:           ^bb1:
// CHECK:           %[[dst_null:.*]] = llvm.mlir.null : !llvm.ptr
// CHECK:           %[[dst_gep:.*]] = llvm.getelementptr %[[dst_null]][1]
// CHECK:           %[[dst_es:.*]] = llvm.ptrtoint %[[dst_gep]] : !llvm.ptr to i64
// CHECK:           %[[dst_size:.*]] = llvm.mul %[[dst_dim]], %[[dst_es]]
// CHECK:           %[[src_size:.*]] = llvm.mul %[[src_dim]], %[[dst_es]]
// CHECK:           %[[new_buffer_raw:.*]] = llvm.call @malloc(%[[dst_size]])
// CHECK:           %[[old_buffer_aligned:.*]] = llvm.extractvalue %[[descriptor]][1]
// CHECK:           %[[volatile:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           "llvm.intr.memcpy"(%[[new_buffer_raw]], %[[old_buffer_aligned]], %[[src_size]], %[[volatile]])
// CHECK:           %[[old_buffer_unaligned:.*]] = llvm.extractvalue %[[descriptor]][0]
// CHECK:           llvm.call @free(%[[old_buffer_unaligned]])
// CHECK:           %[[descriptor_update1:.*]] = llvm.insertvalue %[[new_buffer_raw]], %[[descriptor]][0]
// CHECK:           %[[descriptor_update2:.*]] = llvm.insertvalue %[[new_buffer_raw]], %[[descriptor_update1]][1]
// CHECK:           llvm.br ^bb2(%[[descriptor_update2]]
// CHECK:           ^bb2(%[[descriptor_update3:.*]]: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):
// CHECK:           %[[descriptor_update4:.*]] = llvm.insertvalue %[[dst_dim]], %[[descriptor_update3]][3, 0]
// CHECK:           %[[descriptor_update5:.*]] = builtin.unrealized_conversion_cast %[[descriptor_update4]]
// CHECK:           return %[[descriptor_update5]] : memref<?xf32>

  %out = memref.realloc %in(%d) : memref<?xf32> to memref<?xf32>
  return %out : memref<?xf32>
}

// -----

// CHECK-LABEL:     func.func @realloc_dynamic_alignment(
// CHECK-SAME:      %[[arg0:.*]]: memref<?xf32>,
// CHECK-SAME:      %[[arg1:.*]]: index) -> memref<?xf32> {
// ALIGNED-ALLOC-LABEL:   func.func @realloc_dynamic_alignment(
// ALIGNED-ALLOC-SAME:    %[[arg0:.*]]: memref<?xf32>,
// ALIGNED-ALLOC-SAME:    %[[arg1:.*]]: index) -> memref<?xf32> {
func.func @realloc_dynamic_alignment(%in: memref<?xf32>, %d: index) -> memref<?xf32>{
// CHECK:           %[[descriptor:.*]] = builtin.unrealized_conversion_cast %[[arg0]]
// CHECK:           %[[drc_dim:.*]] = llvm.extractvalue %[[descriptor]][3, 0]
// CHECK:           %[[dst_dim:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i64
// CHECK:           %[[cond:.*]] = llvm.icmp "ugt" %[[dst_dim]], %[[drc_dim]] : i64
// CHECK:           llvm.cond_br %[[cond]], ^bb1, ^bb2(%[[descriptor]]
// CHECK:           ^bb1:
// CHECK:           %[[dst_null:.*]] = llvm.mlir.null : !llvm.ptr
// CHECK:           %[[dst_gep:.*]] = llvm.getelementptr %[[dst_null]][1]
// CHECK:           %[[dst_es:.*]] = llvm.ptrtoint %[[dst_gep]] : !llvm.ptr to i64
// CHECK:           %[[dst_size:.*]] = llvm.mul %[[dst_dim]], %[[dst_es]]
// CHECK:           %[[src_size:.*]] = llvm.mul %[[drc_dim]], %[[dst_es]]
// CHECK:           %[[alignment:.*]] = llvm.mlir.constant(8 : index) : i64
// CHECK:           %[[adjust_dst_size:.*]] = llvm.add %[[dst_size]], %[[alignment]]
// CHECK:           %[[new_buffer_raw:.*]] = llvm.call @malloc(%[[adjust_dst_size]])
// CHECK:           %[[new_buffer_int:.*]] = llvm.ptrtoint %[[new_buffer_raw]] : !llvm.ptr
// CHECK:           %[[const_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[alignment_m1:.*]] = llvm.sub %[[alignment]], %[[const_1]]
// CHECK:           %[[ptr_alignment_m1:.*]] = llvm.add %[[new_buffer_int]], %[[alignment_m1]]
// CHECK:           %[[padding:.*]] = llvm.urem %[[ptr_alignment_m1]], %[[alignment]]
// CHECK:           %[[new_buffer_aligned_int:.*]] = llvm.sub %[[ptr_alignment_m1]], %[[padding]]
// CHECK:           %[[new_buffer_aligned:.*]] = llvm.inttoptr %[[new_buffer_aligned_int]] : i64 to !llvm.ptr
// CHECK:           %[[old_buffer_aligned:.*]] = llvm.extractvalue %[[descriptor]][1]
// CHECK:           %[[volatile:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           "llvm.intr.memcpy"(%[[new_buffer_aligned]], %[[old_buffer_aligned]], %[[src_size]], %[[volatile]])
// CHECK:           %[[old_buffer_unaligned:.*]] = llvm.extractvalue %[[descriptor]][0]
// CHECK:           llvm.call @free(%[[old_buffer_unaligned]])
// CHECK:           %[[descriptor_update1:.*]] = llvm.insertvalue %[[new_buffer_raw]], %[[descriptor]][0]
// CHECK:           %[[descriptor_update2:.*]] = llvm.insertvalue %[[new_buffer_aligned]], %[[descriptor_update1]][1]
// CHECK:           llvm.br ^bb2(%[[descriptor_update2]]
// CHECK:           ^bb2(%[[descriptor_update3:.*]]: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):
// CHECK:           %[[descriptor_update4:.*]] = llvm.insertvalue %[[dst_dim]], %[[descriptor_update3]][3, 0]
// CHECK:           %[[descriptor_update5:.*]] = builtin.unrealized_conversion_cast %[[descriptor_update4]]
// CHECK:           return %[[descriptor_update5]] : memref<?xf32>

// ALIGNED-ALLOC:           %[[descriptor:.*]] = builtin.unrealized_conversion_cast %[[arg0]]
// ALIGNED-ALLOC:           %[[drc_dim:.*]] = llvm.extractvalue %[[descriptor]][3, 0]
// ALIGNED-ALLOC:           %[[dst_dim:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : index to i64
// ALIGNED-ALLOC:           %[[cond:.*]] = llvm.icmp "ugt" %[[dst_dim]], %[[drc_dim]] : i64
// ALIGNED-ALLOC:           llvm.cond_br %[[cond]], ^bb1, ^bb2(%[[descriptor]]
// ALIGNED-ALLOC:           ^bb1:
// ALIGNED-ALLOC:           %[[dst_null:.*]] = llvm.mlir.null : !llvm.ptr
// ALIGNED-ALLOC:           %[[dst_gep:.*]] = llvm.getelementptr %[[dst_null]][1]
// ALIGNED-ALLOC:           %[[dst_es:.*]] = llvm.ptrtoint %[[dst_gep]] : !llvm.ptr to i64
// ALIGNED-ALLOC:           %[[dst_size:.*]] = llvm.mul %[[dst_dim]], %[[dst_es]]
// ALIGNED-ALLOC:           %[[src_size:.*]] = llvm.mul %[[drc_dim]], %[[dst_es]]
// ALIGNED-ALLOC-DAG:       %[[alignment:.*]] = llvm.mlir.constant(8 : index) : i64
// ALIGNED-ALLOC-DAG:       %[[const_1:.*]] = llvm.mlir.constant(1 : index) : i64
// ALIGNED-ALLOC:           %[[alignment_m1:.*]] = llvm.sub %[[alignment]], %[[const_1]]
// ALIGNED-ALLOC:           %[[size_alignment_m1:.*]] = llvm.add %[[dst_size]], %[[alignment_m1]]
// ALIGNED-ALLOC:           %[[padding:.*]] = llvm.urem %[[size_alignment_m1]], %[[alignment]]
// ALIGNED-ALLOC:           %[[adjust_dst_size:.*]] = llvm.sub %[[size_alignment_m1]], %[[padding]]
// ALIGNED-ALLOC:           %[[new_buffer_raw:.*]] = llvm.call @aligned_alloc(%[[alignment]], %[[adjust_dst_size]])
// ALIGNED-ALLOC:           %[[old_buffer_aligned:.*]] = llvm.extractvalue %[[descriptor]][1]
// ALIGNED-ALLOC:           %[[volatile:.*]] = llvm.mlir.constant(false) : i1
// ALIGNED-ALLOC:           "llvm.intr.memcpy"(%[[new_buffer_raw]], %[[old_buffer_aligned]], %[[src_size]], %[[volatile]])
// ALIGNED-ALLOC:           %[[old_buffer_unaligned:.*]] = llvm.extractvalue %[[descriptor]][0]
// ALIGNED-ALLOC:           llvm.call @free(%[[old_buffer_unaligned]])
// ALIGNED-ALLOC:           %[[descriptor_update1:.*]] = llvm.insertvalue %[[new_buffer_raw]], %[[descriptor]][0]
// ALIGNED-ALLOC:           %[[descriptor_update2:.*]] = llvm.insertvalue %[[new_buffer_raw]], %[[descriptor_update1]][1]
// ALIGNED-ALLOC:           llvm.br ^bb2(%[[descriptor_update2]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
// ALIGNED-ALLOC:           ^bb2(%[[descriptor_update3:.*]]: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>):
// ALIGNED-ALLOC:           %[[descriptor_update4:.*]] = llvm.insertvalue %[[dst_dim]], %[[descriptor_update3]][3, 0]
// ALIGNED-ALLOC:           %[[descriptor_update5:.*]] = builtin.unrealized_conversion_cast %[[descriptor_update4]]
// ALIGNED-ALLOC:           return %[[descriptor_update5]] : memref<?xf32>

  %out = memref.realloc %in(%d)  {alignment = 8} : memref<?xf32> to memref<?xf32>
  return %out : memref<?xf32>
}

