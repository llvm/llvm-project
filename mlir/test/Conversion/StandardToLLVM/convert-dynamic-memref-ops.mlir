// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm='use-aligned-alloc=1' %s | FileCheck %s --check-prefix=ALIGNED-ALLOC

// CHECK-LABEL: func @check_strided_memref_arguments(
// CHECK-COUNT-2: !llvm.ptr<float>
// CHECK-COUNT-5: !llvm.i64
// CHECK-COUNT-2: !llvm.ptr<float>
// CHECK-COUNT-5: !llvm.i64
// CHECK-COUNT-2: !llvm.ptr<float>
// CHECK-COUNT-5: !llvm.i64
func @check_strided_memref_arguments(%static: memref<10x20xf32, affine_map<(i,j)->(20 * i + j + 1)>>,
                                     %dynamic : memref<?x?xf32, affine_map<(i,j)[M]->(M * i + j + 1)>>,
                                     %mixed : memref<10x?xf32, affine_map<(i,j)[M]->(M * i + j + 1)>>) {
  return
}

// CHECK-LABEL: func @check_arguments
// CHECK-COUNT-2: !llvm.ptr<float>
// CHECK-COUNT-5: !llvm.i64
// CHECK-COUNT-2: !llvm.ptr<float>
// CHECK-COUNT-5: !llvm.i64
// CHECK-COUNT-2: !llvm.ptr<float>
// CHECK-COUNT-5: !llvm.i64
func @check_arguments(%static: memref<10x20xf32>, %dynamic : memref<?x?xf32>, %mixed : memref<10x?xf32>) {
  return
}

// CHECK-LABEL: func @mixed_alloc(
//       CHECK:   %[[M:.*]]: !llvm.i64, %[[N:.*]]: !llvm.i64) -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)> {
func @mixed_alloc(%arg0: index, %arg1: index) -> memref<?x42x?xf32> {
//       CHECK:  %[[c42:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
//  CHECK-NEXT:  llvm.mul %[[M]], %[[c42]] : !llvm.i64
//  CHECK-NEXT:  %[[sz:.*]] = llvm.mul %{{.*}}, %[[N]] : !llvm.i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.mul %[[sz]], %[[sizeof]] : !llvm.i64
//  CHECK-NEXT:  llvm.call @malloc(%[[sz_bytes]]) : (!llvm.i64) -> !llvm.ptr<i8>
//  CHECK-NEXT:  llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<float>
//  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  %[[st2:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mul %{{.*}}, %[[N]] : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mul %{{.*}}, %[[c42]] : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st0]], %{{.*}}[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[c42]], %{{.*}}[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st2]], %{{.*}}[4, 2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
  %0 = alloc(%arg0, %arg1) : memref<?x42x?xf32>
//  CHECK-NEXT:  llvm.return %{{.*}} : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
  return %0 : memref<?x42x?xf32>
}

// CHECK-LABEL: func @mixed_dealloc
func @mixed_dealloc(%arg0: memref<?x42x?xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:  %[[ptri8:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<float> to !llvm.ptr<i8>
// CHECK-NEXT:  llvm.call @free(%[[ptri8]]) : (!llvm.ptr<i8>) -> ()
  dealloc %arg0 : memref<?x42x?xf32>
// CHECK-NEXT:  llvm.return
  return
}

// CHECK-LABEL: func @dynamic_alloc(
//       CHECK:   %[[M:.*]]: !llvm.i64, %[[N:.*]]: !llvm.i64) -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
func @dynamic_alloc(%arg0: index, %arg1: index) -> memref<?x?xf32> {
//       CHECK:  %[[sz:.*]] = llvm.mul %[[M]], %[[N]] : !llvm.i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.mul %[[sz]], %[[sizeof]] : !llvm.i64
//  CHECK-NEXT:  llvm.call @malloc(%[[sz_bytes]]) : (!llvm.i64) -> !llvm.ptr<i8>
//  CHECK-NEXT:  llvm.bitcast %{{.*}} : !llvm.ptr<i8> to !llvm.ptr<float>
//  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mul %{{.*}}, %[[N]] : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st0]], %{{.*}}[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
//  CHECK-NEXT:  llvm.return %{{.*}} : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  return %0 : memref<?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_alloca
// CHECK: %[[M:.*]]: !llvm.i64, %[[N:.*]]: !llvm.i64) -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
func @dynamic_alloca(%arg0: index, %arg1: index) -> memref<?x?xf32> {
//       CHECK:  %[[num_elems:.*]] = llvm.mul %[[M]], %[[N]] : !llvm.i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
//  CHECK-NEXT:  %[[allocated:.*]] = llvm.alloca %[[sz_bytes]] x !llvm.float : (!llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[allocated]], %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[allocated]], %{{.*}}[1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mul %{{.*}}, %[[N]] : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st0]], %{{.*}}[4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  llvm.insertvalue %[[st1]], %{{.*}}[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  %0 = alloca(%arg0, %arg1) : memref<?x?xf32>

// Test with explicitly specified alignment. llvm.alloca takes care of the
// alignment. The same pointer is thus used for allocation and aligned
// accesses.
// CHECK: %[[alloca_aligned:.*]] = llvm.alloca %{{.*}} x !llvm.float {alignment = 32 : i64} : (!llvm.i64) -> !llvm.ptr<float>
// CHECK: %[[desc:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[desc1:.*]] = llvm.insertvalue %[[alloca_aligned]], %[[desc]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: llvm.insertvalue %[[alloca_aligned]], %[[desc1]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  alloca(%arg0, %arg1) {alignment = 32} : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// CHECK-LABEL: func @dynamic_dealloc
func @dynamic_dealloc(%arg0: memref<?x?xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  %[[ptri8:.*]] = llvm.bitcast %[[ptr]] : !llvm.ptr<float> to !llvm.ptr<i8>
// CHECK-NEXT:  llvm.call @free(%[[ptri8]]) : (!llvm.ptr<i8>) -> ()
  dealloc %arg0 : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @stdlib_aligned_alloc({{.*}}) -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
// ALIGNED-ALLOC-LABEL: func @stdlib_aligned_alloc({{.*}}) -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {
func @stdlib_aligned_alloc(%N : index) -> memref<32x18xf32> {
// ALIGNED-ALLOC-NEXT:  %[[sz1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[sz2:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[num_elems:.*]] = llvm.mul %0, %1 : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm.ptr<float>
// ALIGNED-ALLOC-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// ALIGNED-ALLOC-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm.ptr<float> to !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[alignment:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[allocated:.*]] = llvm.call @aligned_alloc(%[[alignment]], %[[bytes]]) : (!llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
// ALIGNED-ALLOC-NEXT:  llvm.bitcast %[[allocated]] : !llvm.ptr<i8> to !llvm.ptr<float>
  %0 = alloc() {alignment = 32} : memref<32x18xf32>
  // Do another alloc just to test that we have a unique declaration for
  // aligned_alloc.
  // ALIGNED-ALLOC:  llvm.call @aligned_alloc
  %1 = alloc() {alignment = 64} : memref<4096xf32>

  // Alignment is to element type boundaries (minimum 16 bytes).
  // ALIGNED-ALLOC:  %[[c32:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c32]]
  %2 = alloc() : memref<4096xvector<8xf32>>
  // The minimum alignment is 16 bytes unless explicitly specified.
  // ALIGNED-ALLOC:  %[[c16:.*]] = llvm.mlir.constant(16 : index) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c16]],
  %3 = alloc() : memref<4096xvector<2xf32>>
  // ALIGNED-ALLOC:  %[[c8:.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c8]],
  %4 = alloc() {alignment = 8} : memref<1024xvector<4xf32>>
  // Bump the memref allocation size if its size is not a multiple of alignment.
  // ALIGNED-ALLOC:       %[[c32:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.mlir.constant(1 : index) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.sub
  // ALIGNED-ALLOC-NEXT:  llvm.add
  // ALIGNED-ALLOC-NEXT:  llvm.urem
  // ALIGNED-ALLOC-NEXT:  %[[SIZE_ALIGNED:.*]] = llvm.sub
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c32]], %[[SIZE_ALIGNED]])
  %5 = alloc() {alignment = 32} : memref<100xf32>
  // Bump alignment to the next power of two if it isn't.
  // ALIGNED-ALLOC:  %[[c128:.*]] = llvm.mlir.constant(128 : index) : !llvm.i64
  // ALIGNED-ALLOC:  llvm.call @aligned_alloc(%[[c128]]
  %6 = alloc(%N) : memref<?xvector<18xf32>>
  return %0 : memref<32x18xf32>
}

// CHECK-LABEL: func @mixed_load(
// CHECK-COUNT-2: !llvm.ptr<float>,
// CHECK-COUNT-5: {{%[a-zA-Z0-9]*}}: !llvm.i64
// CHECK:         %[[I:.*]]: !llvm.i64,
// CHECK:         %[[J:.*]]: !llvm.i64)
func @mixed_load(%mixed : memref<42x?xf32>, %i : index, %j : index) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  llvm.load %[[addr]] : !llvm.ptr<float>
  %0 = load %mixed[%i, %j] : memref<42x?xf32>
  return
}

// CHECK-LABEL: func @dynamic_load(
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[I:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[J:[a-zA-Z0-9]*]]: !llvm.i64
func @dynamic_load(%dynamic : memref<?x?xf32>, %i : index, %j : index) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  llvm.load %[[addr]] : !llvm.ptr<float>
  %0 = load %dynamic[%i, %j] : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @prefetch
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[I:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[J:[a-zA-Z0-9]*]]: !llvm.i64
func @prefetch(%A : memref<?x?xf32>, %i : index, %j : index) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
// CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
// CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
// CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
// CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK-NEXT:  [[C1:%.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:  [[C3:%.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT:  [[C1_1:%.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:  "llvm.intr.prefetch"(%[[addr]], [[C1]], [[C3]], [[C1_1]]) : (!llvm.ptr<float>, !llvm.i32, !llvm.i32, !llvm.i32) -> ()
  prefetch %A[%i, %j], write, locality<3>, data : memref<?x?xf32>
// CHECK:  [[C0:%.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:  [[C0_1:%.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:  [[C1_2:%.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:  "llvm.intr.prefetch"(%{{.*}}, [[C0]], [[C0_1]], [[C1_2]]) : (!llvm.ptr<float>, !llvm.i32, !llvm.i32, !llvm.i32) -> ()
  prefetch %A[%i, %j], read, locality<0>, data : memref<?x?xf32>
// CHECK:  [[C0_2:%.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:  [[C2:%.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:  [[C0_3:%.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:  "llvm.intr.prefetch"(%{{.*}}, [[C0_2]], [[C2]], [[C0_3]]) : (!llvm.ptr<float>, !llvm.i32, !llvm.i32, !llvm.i32) -> ()
  prefetch %A[%i, %j], read, locality<2>, instr : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @dynamic_store
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[I:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[J:[a-zA-Z0-9]*]]: !llvm.i64
func @dynamic_store(%dynamic : memref<?x?xf32>, %i : index, %j : index, %val : f32) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : !llvm.ptr<float>
  store %val, %dynamic[%i, %j] : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @mixed_store
// CHECK-SAME:         %[[ARG0:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG1:[a-zA-Z0-9]*]]: !llvm.ptr<float>
// CHECK-SAME:         %[[ARG2:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG3:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG4:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG5:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[ARG6:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[I:[a-zA-Z0-9]*]]: !llvm.i64
// CHECK-SAME:         %[[J:[a-zA-Z0-9]*]]: !llvm.i64
func @mixed_store(%mixed : memref<42x?xf32>, %i : index, %j : index, %val : f32) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : !llvm.ptr<float>
  store %val, %mixed[%i, %j] : memref<42x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_static_to_dynamic
func @memref_cast_static_to_dynamic(%static : memref<10x42xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref_cast %static : memref<10x42xf32> to memref<?x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_static_to_mixed
func @memref_cast_static_to_mixed(%static : memref<10x42xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref_cast %static : memref<10x42xf32> to memref<?x42xf32>
  return
}

// CHECK-LABEL: func @memref_cast_dynamic_to_static
func @memref_cast_dynamic_to_static(%dynamic : memref<?x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref_cast %dynamic : memref<?x?xf32> to memref<10x12xf32>
  return
}

// CHECK-LABEL: func @memref_cast_dynamic_to_mixed
func @memref_cast_dynamic_to_mixed(%dynamic : memref<?x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref_cast %dynamic : memref<?x?xf32> to memref<?x12xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_dynamic
func @memref_cast_mixed_to_dynamic(%mixed : memref<42x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<?x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_static
func @memref_cast_mixed_to_static(%mixed : memref<42x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<42x1xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_mixed
func @memref_cast_mixed_to_mixed(%mixed : memref<42x?xf32>) {
// CHECK-NOT: llvm.bitcast
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<?x1xf32>
  return
}

// CHECK-LABEL: func @memref_cast_ranked_to_unranked
func @memref_cast_ranked_to_unranked(%arg : memref<42x2x?xf32>) {
// CHECK-DAG:  %[[c:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-DAG:  %[[p:.*]] = llvm.alloca %[[c]] x !llvm.struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)> : (!llvm.i64) -> !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>>
// CHECK-DAG:  llvm.store %{{.*}}, %[[p]] : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>>
// CHECK-DAG:  %[[p2:.*]] = llvm.bitcast %[[p]] : !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<3 x i64>, array<3 x i64>)>> to !llvm.ptr<i8>
// CHECK-DAG:  %[[r:.*]] = llvm.mlir.constant(3 : i64) : !llvm.i64
// CHECK    :  llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK-DAG:  llvm.insertvalue %[[r]], %{{.*}}[0] : !llvm.struct<(i64, ptr<i8>)>
// CHECK-DAG:  llvm.insertvalue %[[p2]], %{{.*}}[1] : !llvm.struct<(i64, ptr<i8>)>
  %0 = memref_cast %arg : memref<42x2x?xf32> to memref<*xf32>
  return
}

// CHECK-LABEL: func @memref_cast_unranked_to_ranked
func @memref_cast_unranked_to_ranked(%arg : memref<*xf32>) {
//      CHECK: %[[p:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i64, ptr<i8>)>
// CHECK-NEXT: llvm.bitcast %[[p]] : !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, array<4 x i64>, array<4 x i64>)>>
  %0 = memref_cast %arg : memref<*xf32> to memref<?x?x10x2xf32>
  return
}

// CHECK-LABEL: func @mixed_memref_dim
func @mixed_memref_dim(%mixed : memref<42x?x?x13x?xf32>) {
// CHECK: llvm.mlir.constant(42 : index) : !llvm.i64
  %c0 = constant 0 : index
  %0 = dim %mixed, %c0 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %[[ld:.*]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<5 x i64>, array<5 x i64>)>
  %c1 = constant 1 : index
  %1 = dim %mixed, %c1 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %[[ld]][3, 2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<5 x i64>, array<5 x i64>)>
  %c2 = constant 2 : index
  %2 = dim %mixed, %c2 : memref<42x?x?x13x?xf32>
// CHECK: llvm.mlir.constant(13 : index) : !llvm.i64
  %c3 = constant 3 : index
  %3 = dim %mixed, %c3 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %[[ld]][3, 4] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<5 x i64>, array<5 x i64>)>
  %c4 = constant 4 : index
  %4 = dim %mixed, %c4 : memref<42x?x?x13x?xf32>
  return
}

// CHECK-LABEL: @memref_dim_with_dyn_index
// CHECK-SAME: %[[ALLOC_PTR:.*]]: !llvm.ptr<float>, %[[ALIGN_PTR:.*]]: !llvm.ptr<float>, %[[OFFSET:.*]]: !llvm.i64, %[[SIZE0:.*]]: !llvm.i64, %[[SIZE1:.*]]: !llvm.i64, %[[STRIDE0:.*]]: !llvm.i64, %[[STRIDE1:.*]]: !llvm.i64, %[[IDX:.*]]: !llvm.i64) -> !llvm.i64
func @memref_dim_with_dyn_index(%arg : memref<3x?xf32>, %idx : index) -> index {
  // CHECK-NEXT: %[[DESCR0:.*]] = llvm.mlir.undef : [[DESCR_TY:!llvm.struct<\(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>\)>]]
  // CHECK-NEXT: %[[DESCR1:.*]] = llvm.insertvalue %[[ALLOC_PTR]], %[[DESCR0]][0] : [[DESCR_TY]]
  // CHECK-NEXT: %[[DESCR2:.*]] = llvm.insertvalue %[[ALIGN_PTR]], %[[DESCR1]][1] : [[DESCR_TY]]
  // CHECK-NEXT: %[[DESCR3:.*]] = llvm.insertvalue %[[OFFSET]],    %[[DESCR2]][2] : [[DESCR_TY]]
  // CHECK-NEXT: %[[DESCR4:.*]] = llvm.insertvalue %[[SIZE0]],     %[[DESCR3]][3, 0] : [[DESCR_TY]]
  // CHECK-NEXT: %[[DESCR5:.*]] = llvm.insertvalue %[[STRIDE0]],   %[[DESCR4]][4, 0] : [[DESCR_TY]]
  // CHECK-NEXT: %[[DESCR6:.*]] = llvm.insertvalue %[[SIZE1]],     %[[DESCR5]][3, 1] : [[DESCR_TY]]
  // CHECK-NEXT: %[[DESCR7:.*]] = llvm.insertvalue %[[STRIDE1]],   %[[DESCR6]][4, 1] : [[DESCR_TY]]
  // CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK-DAG: %[[SIZES:.*]] = llvm.extractvalue %[[DESCR7]][3] : [[DESCR_TY]]
  // CHECK-DAG: %[[SIZES_PTR:.*]] = llvm.alloca %[[C1]] x !llvm.array<2 x i64> : (!llvm.i64) -> !llvm.ptr<array<2 x i64>>
  // CHECK-DAG: llvm.store %[[SIZES]], %[[SIZES_PTR]] : !llvm.ptr<array<2 x i64>>
  // CHECK-DAG: %[[RESULT_PTR:.*]] = llvm.getelementptr %[[SIZES_PTR]][%[[C0]], %[[IDX]]] : (!llvm.ptr<array<2 x i64>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i64>
  // CHECK-DAG: %[[RESULT:.*]] = llvm.load %[[RESULT_PTR]] : !llvm.ptr<i64>
  // CHECK-DAG: llvm.return %[[RESULT]] : !llvm.i64
  %result = dim %arg, %idx : memref<3x?xf32>
  return %result : index
}

// CHECK-LABEL: @memref_reinterpret_cast_ranked_to_static_shape
func @memref_reinterpret_cast_ranked_to_static_shape(%input : memref<2x3xf32>) {
  %output = memref_reinterpret_cast %input to
           offset: [0], sizes: [6, 1], strides: [1, 1]
           : memref<2x3xf32> to memref<6x1xf32>
  return
}
// CHECK: [[INPUT:%.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : [[TY:!.*]]
// CHECK: [[OUT_0:%.*]] = llvm.mlir.undef : [[TY]]
// CHECK: [[BASE_PTR:%.*]] = llvm.extractvalue [[INPUT]][0] : [[TY]]
// CHECK: [[ALIGNED_PTR:%.*]] = llvm.extractvalue [[INPUT]][1] : [[TY]]
// CHECK: [[OUT_1:%.*]] = llvm.insertvalue [[BASE_PTR]], [[OUT_0]][0] : [[TY]]
// CHECK: [[OUT_2:%.*]] = llvm.insertvalue [[ALIGNED_PTR]], [[OUT_1]][1] : [[TY]]
// CHECK: [[OFFSET:%.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK: [[OUT_3:%.*]] = llvm.insertvalue [[OFFSET]], [[OUT_2]][2] : [[TY]]
// CHECK: [[SIZE_0:%.*]] = llvm.mlir.constant(6 : index) : !llvm.i64
// CHECK: [[OUT_4:%.*]] = llvm.insertvalue [[SIZE_0]], [[OUT_3]][3, 0] : [[TY]]
// CHECK: [[SIZE_1:%.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: [[OUT_5:%.*]] = llvm.insertvalue [[SIZE_1]], [[OUT_4]][4, 0] : [[TY]]
// CHECK: [[STRIDE_0:%.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: [[OUT_6:%.*]] = llvm.insertvalue [[STRIDE_0]], [[OUT_5]][3, 1] : [[TY]]
// CHECK: [[STRIDE_1:%.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: [[OUT_7:%.*]] = llvm.insertvalue [[STRIDE_1]], [[OUT_6]][4, 1] : [[TY]]

// CHECK-LABEL: @memref_reinterpret_cast_unranked_to_dynamic_shape
func @memref_reinterpret_cast_unranked_to_dynamic_shape(%offset: index,
                                                        %size_0 : index,
                                                        %size_1 : index,
                                                        %stride_0 : index,
                                                        %stride_1 : index,
                                                        %input : memref<*xf32>) {
  %output = memref_reinterpret_cast %input to
           offset: [%offset], sizes: [%size_0, %size_1],
           strides: [%stride_0, %stride_1]
           : memref<*xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
  return
}
// CHECK-SAME: ([[OFFSET:%[a-z,0-9]+]]: !llvm.i64,
// CHECK-SAME: [[SIZE_0:%[a-z,0-9]+]]: !llvm.i64, [[SIZE_1:%[a-z,0-9]+]]: !llvm.i64,
// CHECK-SAME: [[STRIDE_0:%[a-z,0-9]+]]: !llvm.i64, [[STRIDE_1:%[a-z,0-9]+]]: !llvm.i64,
// CHECK: [[INPUT:%.*]] = llvm.insertvalue {{.*}}[1] : !llvm.struct<(i64, ptr<i8>)>
// CHECK: [[OUT_0:%.*]] = llvm.mlir.undef : [[TY:!.*]]
// CHECK: [[DESCRIPTOR:%.*]] = llvm.extractvalue [[INPUT]][1] : !llvm.struct<(i64, ptr<i8>)>
// CHECK: [[BASE_PTR_PTR:%.*]] = llvm.bitcast [[DESCRIPTOR]] : !llvm.ptr<i8> to !llvm.ptr<ptr<float>>
// CHECK: [[BASE_PTR:%.*]] = llvm.load [[BASE_PTR_PTR]] : !llvm.ptr<ptr<float>>
// CHECK: [[BASE_PTR_PTR_:%.*]] = llvm.bitcast [[DESCRIPTOR]] : !llvm.ptr<i8> to !llvm.ptr<ptr<float>>
// CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: [[ALIGNED_PTR_PTR:%.*]] = llvm.getelementptr [[BASE_PTR_PTR_]]{{\[}}[[C1]]]
// CHECK-SAME: : (!llvm.ptr<ptr<float>>, !llvm.i64) -> !llvm.ptr<ptr<float>>
// CHECK: [[ALIGNED_PTR:%.*]] = llvm.load [[ALIGNED_PTR_PTR]] : !llvm.ptr<ptr<float>>
// CHECK: [[OUT_1:%.*]] = llvm.insertvalue [[BASE_PTR]], [[OUT_0]][0] : [[TY]]
// CHECK: [[OUT_2:%.*]] = llvm.insertvalue [[ALIGNED_PTR]], [[OUT_1]][1] : [[TY]]
// CHECK: [[OUT_3:%.*]] = llvm.insertvalue [[OFFSET]], [[OUT_2]][2] : [[TY]]
// CHECK: [[OUT_4:%.*]] = llvm.insertvalue [[SIZE_0]], [[OUT_3]][3, 0] : [[TY]]
// CHECK: [[OUT_5:%.*]] = llvm.insertvalue [[STRIDE_0]], [[OUT_4]][4, 0] : [[TY]]
// CHECK: [[OUT_6:%.*]] = llvm.insertvalue [[SIZE_1]], [[OUT_5]][3, 1] : [[TY]]
// CHECK: [[OUT_7:%.*]] = llvm.insertvalue [[STRIDE_1]], [[OUT_6]][4, 1] : [[TY]]

// CHECK-LABEL: @memref_reshape
func @memref_reshape(%input : memref<2x3xf32>, %shape : memref<?xindex>) {
  %output = memref_reshape %input(%shape)
                : (memref<2x3xf32>, memref<?xindex>) -> memref<*xf32>
  return
}
// CHECK: [[INPUT:%.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 1] : [[INPUT_TY:!.*]]
// CHECK: [[SHAPE:%.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0] : [[SHAPE_TY:!.*]]
// CHECK: [[RANK:%.*]] = llvm.extractvalue [[SHAPE]][3, 0] : [[SHAPE_TY]]
// CHECK: [[UNRANKED_OUT_O:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK: [[UNRANKED_OUT_1:%.*]] = llvm.insertvalue [[RANK]], [[UNRANKED_OUT_O]][0] : !llvm.struct<(i64, ptr<i8>)>

// Compute size in bytes to allocate result ranked descriptor
// CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: [[C2:%.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK: [[PTR_SIZE:%.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
// CHECK: [[INDEX_SIZE:%.*]] = llvm.mlir.constant(8 : index) : !llvm.i64
// CHECK: [[DOUBLE_PTR_SIZE:%.*]] = llvm.mul [[C2]], [[PTR_SIZE]] : !llvm.i64
// CHECK: [[DESC_ALLOC_SIZE:%.*]] = llvm.add [[DOUBLE_PTR_SIZE]], %{{.*}}
// CHECK: [[UNDERLYING_DESC:%.*]] = llvm.alloca [[DESC_ALLOC_SIZE]] x !llvm.i8
// CHECK: llvm.insertvalue [[UNDERLYING_DESC]], [[UNRANKED_OUT_1]][1]

// Set allocated, aligned pointers and offset.
// CHECK: [[ALLOC_PTR:%.*]] = llvm.extractvalue [[INPUT]][0] : [[INPUT_TY]]
// CHECK: [[ALIGN_PTR:%.*]] = llvm.extractvalue [[INPUT]][1] : [[INPUT_TY]]
// CHECK: [[OFFSET:%.*]] = llvm.extractvalue [[INPUT]][2] : [[INPUT_TY]]
// CHECK: [[BASE_PTR_PTR:%.*]] = llvm.bitcast [[UNDERLYING_DESC]]
// CHECK-SAME:                     !llvm.ptr<i8> to !llvm.ptr<ptr<float>>
// CHECK: llvm.store [[ALLOC_PTR]], [[BASE_PTR_PTR]] : !llvm.ptr<ptr<float>>
// CHECK: [[BASE_PTR_PTR_:%.*]] = llvm.bitcast [[UNDERLYING_DESC]] : !llvm.ptr<i8> to !llvm.ptr<ptr<float>>
// CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: [[ALIGNED_PTR_PTR:%.*]] = llvm.getelementptr [[BASE_PTR_PTR_]]{{\[}}[[C1]]]
// CHECK: llvm.store [[ALIGN_PTR]], [[ALIGNED_PTR_PTR]] : !llvm.ptr<ptr<float>>
// CHECK: [[BASE_PTR_PTR__:%.*]] = llvm.bitcast [[UNDERLYING_DESC]] : !llvm.ptr<i8> to !llvm.ptr<ptr<float>>
// CHECK: [[C2:%.*]] = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK: [[OFFSET_PTR_:%.*]] = llvm.getelementptr [[BASE_PTR_PTR__]]{{\[}}[[C2]]]
// CHECK: [[OFFSET_PTR:%.*]] = llvm.bitcast [[OFFSET_PTR_]]
// CHECK: llvm.store [[OFFSET]], [[OFFSET_PTR]] : !llvm.ptr<i64>

// Iterate over shape operand in reverse order and set sizes and strides.
// CHECK: [[STRUCT_PTR:%.*]] = llvm.bitcast [[UNDERLYING_DESC]]
// CHECK-SAME: !llvm.ptr<i8> to !llvm.ptr<struct<(ptr<float>, ptr<float>, i64, i64)>>
// CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK: [[C3_I32:%.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK: [[SIZES_PTR:%.*]] = llvm.getelementptr [[STRUCT_PTR]]{{\[}}[[C0]], [[C3_I32]]]
// CHECK: [[STRIDES_PTR:%.*]] = llvm.getelementptr [[SIZES_PTR]]{{\[}}[[RANK]]]
// CHECK: [[SHAPE_IN_PTR:%.*]] = llvm.extractvalue [[SHAPE]][1] : [[SHAPE_TY]]
// CHECK: [[C1_:%.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: [[RANK_MIN_1:%.*]] = llvm.sub [[RANK]], [[C1_]] : !llvm.i64
// CHECK: llvm.br ^bb1([[RANK_MIN_1]], [[C1_]] : !llvm.i64, !llvm.i64)

// CHECK: ^bb1([[DIM:%.*]]: !llvm.i64, [[CUR_STRIDE:%.*]]: !llvm.i64):
// CHECK:   [[C0_:%.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK:   [[COND:%.*]] = llvm.icmp "sge" [[DIM]], [[C0_]] : !llvm.i64
// CHECK:   llvm.cond_br [[COND]], ^bb2, ^bb3

// CHECK: ^bb2:
// CHECK:   [[SIZE_PTR:%.*]] = llvm.getelementptr [[SHAPE_IN_PTR]]{{\[}}[[DIM]]]
// CHECK:   [[SIZE:%.*]] = llvm.load [[SIZE_PTR]] : !llvm.ptr<i64>
// CHECK:   [[TARGET_SIZE_PTR:%.*]] = llvm.getelementptr [[SIZES_PTR]]{{\[}}[[DIM]]]
// CHECK:   llvm.store [[SIZE]], [[TARGET_SIZE_PTR]] : !llvm.ptr<i64>
// CHECK:   [[TARGET_STRIDE_PTR:%.*]] = llvm.getelementptr [[STRIDES_PTR]]{{\[}}[[DIM]]]
// CHECK:   llvm.store [[CUR_STRIDE]], [[TARGET_STRIDE_PTR]] : !llvm.ptr<i64>
// CHECK:   [[UPDATE_STRIDE:%.*]] = llvm.mul [[CUR_STRIDE]], [[SIZE]] : !llvm.i64
// CHECK:   [[STRIDE_COND:%.*]] = llvm.sub [[DIM]], [[C1_]] : !llvm.i64
// CHECK:   llvm.br ^bb1([[STRIDE_COND]], [[UPDATE_STRIDE]] : !llvm.i64, !llvm.i64)

// CHECK: ^bb3:
// CHECK:   llvm.return
