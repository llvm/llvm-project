// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s
// RUN: mlir-opt -convert-std-to-llvm='use-aligned-alloc=1' %s | FileCheck %s --check-prefix=ALIGNED-ALLOC

// CHECK-LABEL: func @check_strided_memref_arguments(
// CHECK-COUNT-2: !llvm<"float*">
// CHECK-COUNT-5: !llvm.i64
// CHECK-COUNT-2: !llvm<"float*">
// CHECK-COUNT-5: !llvm.i64
// CHECK-COUNT-2: !llvm<"float*">
// CHECK-COUNT-5: !llvm.i64
func @check_strided_memref_arguments(%static: memref<10x20xf32, affine_map<(i,j)->(20 * i + j + 1)>>,
                                     %dynamic : memref<?x?xf32, affine_map<(i,j)[M]->(M * i + j + 1)>>,
                                     %mixed : memref<10x?xf32, affine_map<(i,j)[M]->(M * i + j + 1)>>) {
  return
}

// CHECK-LABEL: func @check_arguments
// CHECK-COUNT-2: !llvm<"float*">
// CHECK-COUNT-5: !llvm.i64
// CHECK-COUNT-2: !llvm<"float*">
// CHECK-COUNT-5: !llvm.i64
// CHECK-COUNT-2: !llvm<"float*">
// CHECK-COUNT-5: !llvm.i64
func @check_arguments(%static: memref<10x20xf32>, %dynamic : memref<?x?xf32>, %mixed : memref<10x?xf32>) {
  return
}

// CHECK-LABEL: func @mixed_alloc(
//       CHECK:   %[[M:.*]]: !llvm.i64, %[[N:.*]]: !llvm.i64) -> !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }"> {
func @mixed_alloc(%arg0: index, %arg1: index) -> memref<?x42x?xf32> {
//       CHECK:  %[[c42:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
//  CHECK-NEXT:  llvm.mul %[[M]], %[[c42]] : !llvm.i64
//  CHECK-NEXT:  %[[sz:.*]] = llvm.mul %{{.*}}, %[[N]] : !llvm.i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.mul %[[sz]], %[[sizeof]] : !llvm.i64
//  CHECK-NEXT:  llvm.call @malloc(%[[sz_bytes]]) : (!llvm.i64) -> !llvm<"i8*">
//  CHECK-NEXT:  llvm.bitcast %{{.*}} : !llvm<"i8*"> to !llvm<"float*">
//  CHECK-NEXT:  llvm.mlir.undef : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  %[[st2:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mul %{{.*}}, %[[N]] : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mul %{{.*}}, %[[c42]] : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[st0]], %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[c42]], %{{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[st1]], %{{.*}}[4, 1] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[st2]], %{{.*}}[4, 2] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
  %0 = alloc(%arg0, %arg1) : memref<?x42x?xf32>
//  CHECK-NEXT:  llvm.return %{{.*}} : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
  return %0 : memref<?x42x?xf32>
}

// CHECK-LABEL: func @mixed_dealloc
func @mixed_dealloc(%arg0: memref<?x42x?xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
// CHECK-NEXT:  %[[ptri8:.*]] = llvm.bitcast %[[ptr]] : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT:  llvm.call @free(%[[ptri8]]) : (!llvm<"i8*">) -> ()
  dealloc %arg0 : memref<?x42x?xf32>
// CHECK-NEXT:  llvm.return
  return
}

// CHECK-LABEL: func @dynamic_alloc(
//       CHECK:   %[[M:.*]]: !llvm.i64, %[[N:.*]]: !llvm.i64) -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> {
func @dynamic_alloc(%arg0: index, %arg1: index) -> memref<?x?xf32> {
//       CHECK:  %[[sz:.*]] = llvm.mul %[[M]], %[[N]] : !llvm.i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.mul %[[sz]], %[[sizeof]] : !llvm.i64
//  CHECK-NEXT:  llvm.call @malloc(%[[sz_bytes]]) : (!llvm.i64) -> !llvm<"i8*">
//  CHECK-NEXT:  llvm.bitcast %{{.*}} : !llvm<"i8*"> to !llvm<"float*">
//  CHECK-NEXT:  llvm.mlir.undef : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %{{.*}}, %{{.*}}[1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mul %{{.*}}, %[[N]] : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[st0]], %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[st1]], %{{.*}}[4, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
//  CHECK-NEXT:  llvm.return %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  return %0 : memref<?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_alloca
// CHECK: %[[M:.*]]: !llvm.i64, %[[N:.*]]: !llvm.i64) -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> {
func @dynamic_alloca(%arg0: index, %arg1: index) -> memref<?x?xf32> {
//       CHECK:  %[[num_elems:.*]] = llvm.mul %[[M]], %[[N]] : !llvm.i64
//  CHECK-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
//  CHECK-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
//  CHECK-NEXT:  %[[sz_bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
//  CHECK-NEXT:  %[[allocated:.*]] = llvm.alloca %[[sz_bytes]] x !llvm.float : (!llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  llvm.mlir.undef : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[allocated]], %{{.*}}[0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[allocated]], %{{.*}}[1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[off]], %{{.*}}[2] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.mul %{{.*}}, %[[N]] : !llvm.i64
//  CHECK-NEXT:  llvm.insertvalue %[[M]], %{{.*}}[3, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[st0]], %{{.*}}[4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[N]], %{{.*}}[3, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  llvm.insertvalue %[[st1]], %{{.*}}[4, 1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = alloca(%arg0, %arg1) : memref<?x?xf32>

// Test with explicitly specified alignment. llvm.alloca takes care of the
// alignment. The same pointer is thus used for allocation and aligned
// accesses.
// CHECK: %[[alloca_aligned:.*]] = llvm.alloca %{{.*}} x !llvm.float {alignment = 32 : i64} : (!llvm.i64) -> !llvm<"float*">
// CHECK: %[[desc:.*]] = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
// CHECK: %[[desc1:.*]] = llvm.insertvalue %[[alloca_aligned]], %[[desc]][0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
// CHECK: llvm.insertvalue %[[alloca_aligned]], %[[desc1]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  alloca(%arg0, %arg1) {alignment = 32} : memref<?x?xf32>
  return %0 : memref<?x?xf32>
}

// CHECK-LABEL: func @dynamic_dealloc
func @dynamic_dealloc(%arg0: memref<?x?xf32>) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
// CHECK-NEXT:  %[[ptri8:.*]] = llvm.bitcast %[[ptr]] : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT:  llvm.call @free(%[[ptri8]]) : (!llvm<"i8*">) -> ()
  dealloc %arg0 : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @stdlib_aligned_alloc({{.*}}) -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> {
// ALIGNED-ALLOC-LABEL: func @stdlib_aligned_alloc({{.*}}) -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> {
func @stdlib_aligned_alloc(%N : index) -> memref<32x18xf32> {
// ALIGNED-ALLOC-NEXT:  %[[sz1:.*]] = llvm.mlir.constant(32 : index) : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[sz2:.*]] = llvm.mlir.constant(18 : index) : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[num_elems:.*]] = llvm.mul %0, %1 : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[null:.*]] = llvm.mlir.null : !llvm<"float*">
// ALIGNED-ALLOC-NEXT:  %[[one:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[gep:.*]] = llvm.getelementptr %[[null]][%[[one]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// ALIGNED-ALLOC-NEXT:  %[[sizeof:.*]] = llvm.ptrtoint %[[gep]] : !llvm<"float*"> to !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[bytes:.*]] = llvm.mul %[[num_elems]], %[[sizeof]] : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[alignment:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
// ALIGNED-ALLOC-NEXT:  %[[allocated:.*]] = llvm.call @aligned_alloc(%[[alignment]], %[[bytes]]) : (!llvm.i64, !llvm.i64) -> !llvm<"i8*">
// ALIGNED-ALLOC-NEXT:  llvm.bitcast %[[allocated]] : !llvm<"i8*"> to !llvm<"float*">
  %0 = alloc() {alignment = 32} : memref<32x18xf32>
  // Do another alloc just to test that we have a unique declaration for
  // aligned_alloc.
  // ALIGNED-ALLOC:  llvm.call @aligned_alloc
  %1 = alloc() {alignment = 64} : memref<4096xf32>

  // Alignment is to element type boundaries (minimum 16 bytes).
  // ALIGNED-ALLOC:  %[[c32:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c32]]
  %2 = alloc() : memref<4096xvector<8xf32>>
  // The minimum alignment is 16 bytes unless explicitly specified.
  // ALIGNED-ALLOC:  %[[c16:.*]] = llvm.mlir.constant(16 : i64) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c16]],
  %3 = alloc() : memref<4096xvector<2xf32>>
  // ALIGNED-ALLOC:  %[[c8:.*]] = llvm.mlir.constant(8 : i64) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c8]],
  %4 = alloc() {alignment = 8} : memref<1024xvector<4xf32>>
  // Bump the memref allocation size if its size is not a multiple of alignment.
  // ALIGNED-ALLOC:       %[[c32:.*]] = llvm.mlir.constant(32 : i64) : !llvm.i64
  // ALIGNED-ALLOC-NEXT:  llvm.urem
  // ALIGNED-ALLOC-NEXT:  llvm.sub
  // ALIGNED-ALLOC-NEXT:  llvm.urem
  // ALIGNED-ALLOC-NEXT:  %[[SIZE_ALIGNED:.*]] = llvm.add
  // ALIGNED-ALLOC-NEXT:  llvm.call @aligned_alloc(%[[c32]], %[[SIZE_ALIGNED]])
  %5 = alloc() {alignment = 32} : memref<100xf32>
  // Bump alignment to the next power of two if it isn't.
  // ALIGNED-ALLOC:  %[[c128:.*]] = llvm.mlir.constant(128 : i64) : !llvm.i64
  // ALIGNED-ALLOC:  llvm.call @aligned_alloc(%[[c128]]
  %6 = alloc(%N) : memref<?xvector<18xf32>>
  return %0 : memref<32x18xf32>
}

// CHECK-LABEL: func @mixed_load(
// CHECK-COUNT-2: !llvm<"float*">,
// CHECK-COUNT-5: {{%[a-zA-Z0-9]*}}: !llvm.i64
// CHECK:         %[[I:.*]]: !llvm.i64,
// CHECK:         %[[J:.*]]: !llvm.i64)
func @mixed_load(%mixed : memref<42x?xf32>, %i : index, %j : index) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  llvm.load %[[addr]] : !llvm<"float*">
  %0 = load %mixed[%i, %j] : memref<42x?xf32>
  return
}

// CHECK-LABEL: func @dynamic_load(
func @dynamic_load(%dynamic : memref<?x?xf32>, %i : index, %j : index) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  llvm.load %[[addr]] : !llvm<"float*">
  %0 = load %dynamic[%i, %j] : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @prefetch
func @prefetch(%A : memref<?x?xf32>, %i : index, %j : index) {
//      CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
// CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
// CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
// CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
// CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
// CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
// CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
// CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
// CHECK-NEXT:  [[C1:%.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:  [[C3:%.*]] = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT:  [[C1_1:%.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT:  "llvm.intr.prefetch"(%[[addr]], [[C1]], [[C3]], [[C1_1]]) : (!llvm<"float*">, !llvm.i32, !llvm.i32, !llvm.i32) -> ()
  prefetch %A[%i, %j], write, locality<3>, data : memref<?x?xf32>
// CHECK:  [[C0:%.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:  [[C0_1:%.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:  [[C1_2:%.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK:  "llvm.intr.prefetch"(%{{.*}}, [[C0]], [[C0_1]], [[C1_2]]) : (!llvm<"float*">, !llvm.i32, !llvm.i32, !llvm.i32) -> ()
  prefetch %A[%i, %j], read, locality<0>, data : memref<?x?xf32>
// CHECK:  [[C0_2:%.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:  [[C2:%.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
// CHECK:  [[C0_3:%.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK:  "llvm.intr.prefetch"(%{{.*}}, [[C0_2]], [[C2]], [[C0_3]]) : (!llvm<"float*">, !llvm.i32, !llvm.i32, !llvm.i32) -> ()
  prefetch %A[%i, %j], read, locality<2>, instr : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @dynamic_store
func @dynamic_store(%dynamic : memref<?x?xf32>, %i : index, %j : index, %val : f32) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : !llvm<"float*">
  store %val, %dynamic[%i, %j] : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @mixed_store
func @mixed_store(%mixed : memref<42x?xf32>, %i : index, %j : index, %val : f32) {
//       CHECK:  %[[ptr:.*]] = llvm.extractvalue %[[ld:.*]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  llvm.store %{{.*}}, %[[addr]] : !llvm<"float*">
  store %val, %mixed[%i, %j] : memref<42x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_static_to_dynamic
func @memref_cast_static_to_dynamic(%static : memref<10x42xf32>) {
// CHECK: llvm.bitcast %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %static : memref<10x42xf32> to memref<?x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_static_to_mixed
func @memref_cast_static_to_mixed(%static : memref<10x42xf32>) {
// CHECK: llvm.bitcast %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %static : memref<10x42xf32> to memref<?x42xf32>
  return
}

// CHECK-LABEL: func @memref_cast_dynamic_to_static
func @memref_cast_dynamic_to_static(%dynamic : memref<?x?xf32>) {
// CHECK: llvm.bitcast %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %dynamic : memref<?x?xf32> to memref<10x12xf32>
  return
}

// CHECK-LABEL: func @memref_cast_dynamic_to_mixed
func @memref_cast_dynamic_to_mixed(%dynamic : memref<?x?xf32>) {
// CHECK: llvm.bitcast %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %dynamic : memref<?x?xf32> to memref<?x12xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_dynamic
func @memref_cast_mixed_to_dynamic(%mixed : memref<42x?xf32>) {
// CHECK: llvm.bitcast %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<?x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_static
func @memref_cast_mixed_to_static(%mixed : memref<42x?xf32>) {
// CHECK: llvm.bitcast %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<42x1xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_mixed
func @memref_cast_mixed_to_mixed(%mixed : memref<42x?xf32>) {
// CHECK: llvm.bitcast %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<?x1xf32>
  return
}

// CHECK-LABEL: func @memref_cast_ranked_to_unranked
func @memref_cast_ranked_to_unranked(%arg : memref<42x2x?xf32>) {
// CHECK-DAG:  %[[c:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-DAG:  %[[p:.*]] = llvm.alloca %[[c]] x !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">
// CHECK-DAG:  llvm.store %{{.*}}, %[[p]] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">
// CHECK-DAG:  %[[p2:.*]] = llvm.bitcast %[[p]] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*"> to !llvm<"i8*">
// CHECK-DAG:  %[[r:.*]] = llvm.mlir.constant(3 : i64) : !llvm.i64
// CHECK    :  llvm.mlir.undef : !llvm<"{ i64, i8* }">
// CHECK-DAG:  llvm.insertvalue %[[r]], %{{.*}}[0] : !llvm<"{ i64, i8* }">
// CHECK-DAG:  llvm.insertvalue %[[p2]], %{{.*}}[1] : !llvm<"{ i64, i8* }">
  %0 = memref_cast %arg : memref<42x2x?xf32> to memref<*xf32>
  return
}

// CHECK-LABEL: func @memref_cast_unranked_to_ranked
func @memref_cast_unranked_to_ranked(%arg : memref<*xf32>) {
//      CHECK: %[[p:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm<"{ i64, i8* }">
// CHECK-NEXT: llvm.bitcast %[[p]] : !llvm<"i8*"> to !llvm<"{ float*, float*, i64, [4 x i64], [4 x i64] }*">
  %0 = memref_cast %arg : memref<*xf32> to memref<?x?x10x2xf32>
  return
}

// CHECK-LABEL: func @mixed_memref_dim
func @mixed_memref_dim(%mixed : memref<42x?x?x13x?xf32>) {
// CHECK: llvm.mlir.constant(42 : index) : !llvm.i64
  %c0 = constant 0 : index
  %0 = dim %mixed, %c0 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %[[ld:.*]][3, 1] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
  %c1 = constant 1 : index
  %1 = dim %mixed, %c1 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %[[ld]][3, 2] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
  %c2 = constant 2 : index
  %2 = dim %mixed, %c2 : memref<42x?x?x13x?xf32>
// CHECK: llvm.mlir.constant(13 : index) : !llvm.i64
  %c3 = constant 3 : index
  %3 = dim %mixed, %c3 : memref<42x?x?x13x?xf32>
// CHECK: llvm.extractvalue %[[ld]][3, 4] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
  %c4 = constant 4 : index
  %4 = dim %mixed, %c4 : memref<42x?x?x13x?xf32>
  return
}
