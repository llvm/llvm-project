// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s

//   CHECK-LABEL: func @check_strided_memref_arguments(
// CHECK-COUNT-3:   !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
func @check_strided_memref_arguments(%static: memref<10x20xf32, affine_map<(i,j)->(20 * i + j + 1)>>,
                                     %dynamic : memref<?x?xf32, affine_map<(i,j)[M]->(M * i + j + 1)>>,
                                     %mixed : memref<10x?xf32, affine_map<(i,j)[M]->(M * i + j + 1)>>) {
  return
}

// CHECK-LABEL: func @check_arguments(%arg0: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">, %arg1: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">, %arg2: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">)
func @check_arguments(%static: memref<10x20xf32>, %dynamic : memref<?x?xf32>, %mixed : memref<10x?xf32>) {
  return
}

// CHECK-LABEL: func @mixed_alloc(
//       CHECK:   %[[M:.*]]: !llvm.i64, %[[N:.*]]: !llvm.i64) -> !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }"> {
func @mixed_alloc(%arg0: index, %arg1: index) -> memref<?x42x?xf32> {
//  CHECK-NEXT:  %[[c42:.*]] = llvm.mlir.constant(42 : index) : !llvm.i64
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

// CHECK-LABEL: func @mixed_dealloc(%arg0: !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">) {
func @mixed_dealloc(%arg0: memref<?x42x?xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][0] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }">
// CHECK-NEXT:  %[[ptri8:.*]] = llvm.bitcast %[[ptr]] : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT:  llvm.call @free(%[[ptri8]]) : (!llvm<"i8*">) -> ()
  dealloc %arg0 : memref<?x42x?xf32>
// CHECK-NEXT:  llvm.return
  return
}

// CHECK-LABEL: func @dynamic_alloc(
//       CHECK:   %[[M:.*]]: !llvm.i64, %[[N:.*]]: !llvm.i64) -> !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> {
func @dynamic_alloc(%arg0: index, %arg1: index) -> memref<?x?xf32> {
//  CHECK-NEXT:  %[[sz:.*]] = llvm.mul %[[M]], %[[N]] : !llvm.i64
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

// CHECK-LABEL: func @dynamic_dealloc(%arg0: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">) {
func @dynamic_dealloc(%arg0: memref<?x?xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
// CHECK-NEXT:  %[[ptri8:.*]] = llvm.bitcast %[[ptr]] : !llvm<"float*"> to !llvm<"i8*">
// CHECK-NEXT:  llvm.call @free(%[[ptri8]]) : (!llvm<"i8*">) -> ()
  dealloc %arg0 : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @mixed_load(
//       CHECK:   %[[A:.*]]: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">, %[[I:.*]]: !llvm.i64, %[[J:.*]]: !llvm.i64
func @mixed_load(%mixed : memref<42x?xf32>, %i : index, %j : index) {
//  CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//  CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
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
//       CHECK:   %[[A:.*]]: !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">, %[[I:.*]]: !llvm.i64, %[[J:.*]]: !llvm.i64
func @dynamic_load(%dynamic : memref<?x?xf32>, %i : index, %j : index) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//  CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
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
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
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
//  CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//  CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  llvm.store %arg3, %[[addr]] : !llvm<"float*">
  store %val, %dynamic[%i, %j] : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @mixed_store
func @mixed_store(%mixed : memref<42x?xf32>, %i : index, %j : index, %val : f32) {
//  CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
//  CHECK-NEXT:  %[[ptr:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[off:.*]] = llvm.mlir.constant(0 : index) : !llvm.i64
//  CHECK-NEXT:  %[[st0:.*]] = llvm.extractvalue %[[ld]][4, 0] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
//  CHECK-NEXT:  %[[offI:.*]] = llvm.mul %[[I]], %[[st0]] : !llvm.i64
//  CHECK-NEXT:  %[[off0:.*]] = llvm.add %[[off]], %[[offI]] : !llvm.i64
//  CHECK-NEXT:  %[[st1:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
//  CHECK-NEXT:  %[[offJ:.*]] = llvm.mul %[[J]], %[[st1]] : !llvm.i64
//  CHECK-NEXT:  %[[off1:.*]] = llvm.add %[[off0]], %[[offJ]] : !llvm.i64
//  CHECK-NEXT:  %[[addr:.*]] = llvm.getelementptr %[[ptr]][%[[off1]]] : (!llvm<"float*">, !llvm.i64) -> !llvm<"float*">
//  CHECK-NEXT:  llvm.store %arg3, %[[addr]] : !llvm<"float*">
  store %val, %mixed[%i, %j] : memref<42x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_static_to_dynamic
func @memref_cast_static_to_dynamic(%static : memref<10x42xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  llvm.bitcast %[[ld]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %static : memref<10x42xf32> to memref<?x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_static_to_mixed
func @memref_cast_static_to_mixed(%static : memref<10x42xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  llvm.bitcast %[[ld]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %static : memref<10x42xf32> to memref<?x42xf32>
  return
}

// CHECK-LABEL: func @memref_cast_dynamic_to_static
func @memref_cast_dynamic_to_static(%dynamic : memref<?x?xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  llvm.bitcast %[[ld]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %dynamic : memref<?x?xf32> to memref<10x12xf32>
  return
}

// CHECK-LABEL: func @memref_cast_dynamic_to_mixed
func @memref_cast_dynamic_to_mixed(%dynamic : memref<?x?xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  llvm.bitcast %[[ld]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %dynamic : memref<?x?xf32> to memref<?x12xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_dynamic
func @memref_cast_mixed_to_dynamic(%mixed : memref<42x?xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  llvm.bitcast %[[ld]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<?x?xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_static
func @memref_cast_mixed_to_static(%mixed : memref<42x?xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  llvm.bitcast %[[ld]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<42x1xf32>
  return
}

// CHECK-LABEL: func @memref_cast_mixed_to_mixed
func @memref_cast_mixed_to_mixed(%mixed : memref<42x?xf32>) {
// CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }*">
// CHECK-NEXT:  llvm.bitcast %[[ld]] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }"> to !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  %0 = memref_cast %mixed : memref<42x?xf32> to memref<?x1xf32>
  return
}

// CHECK-LABEL: func @memref_cast_ranked_to_unranked
func @memref_cast_ranked_to_unranked(%arg : memref<42x2x?xf32>) {
// CHECK-NEXT: %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">
// CHECK-DAG:  %[[c:.*]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-DAG:  %[[p:.*]] = llvm.alloca %[[c]] x !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }"> : (!llvm.i64) -> !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">
// CHECK-DAG:  llvm.store %[[ld]], %[[p]] : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*">
// CHECK-DAG:  %[[p2:.*]] = llvm.bitcast %2 : !llvm<"{ float*, float*, i64, [3 x i64], [3 x i64] }*"> to !llvm<"i8*">
// CHECK-DAG:  %[[r:.*]] = llvm.mlir.constant(3 : i64) : !llvm.i64
// CHECK    :  llvm.mlir.undef : !llvm<"{ i64, i8* }">
// CHECK-DAG:  llvm.insertvalue %[[r]], %{{.*}}[0] : !llvm<"{ i64, i8* }">
// CHECK-DAG:  llvm.insertvalue %[[p2]], %{{.*}}[1] : !llvm<"{ i64, i8* }">
  %0 = memref_cast %arg : memref<42x2x?xf32> to memref<*xf32>
  return
}

// CHECK-LABEL: func @memref_cast_unranked_to_ranked
func @memref_cast_unranked_to_ranked(%arg : memref<*xf32>) {
// CHECK: %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ i64, i8* }*">
// CHECK-NEXT: %[[p:.*]] = llvm.extractvalue %[[ld]][1] : !llvm<"{ i64, i8* }">
// CHECK-NEXT: llvm.bitcast %[[p]] : !llvm<"i8*"> to !llvm<"{ float*, float*, i64, [4 x i64], [4 x i64] }*">
  %0 = memref_cast %arg : memref<*xf32> to memref<?x?x10x2xf32>
  return
}

// CHECK-LABEL: func @mixed_memref_dim(%arg0: !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }*">) {
func @mixed_memref_dim(%mixed : memref<42x?x?x13x?xf32>) {
//  CHECK-NEXT:  %[[ld:.*]] = llvm.load %{{.*}} : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }*">
//  CHECK-NEXT:  llvm.mlir.constant(42 : index) : !llvm.i64
  %0 = dim %mixed, 0 : memref<42x?x?x13x?xf32>
//  CHECK-NEXT:  llvm.extractvalue %[[ld]][3, 1] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
  %1 = dim %mixed, 1 : memref<42x?x?x13x?xf32>
//  CHECK-NEXT:  llvm.extractvalue %[[ld]][3, 2] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
  %2 = dim %mixed, 2 : memref<42x?x?x13x?xf32>
//  CHECK-NEXT:  llvm.mlir.constant(13 : index) : !llvm.i64
  %3 = dim %mixed, 3 : memref<42x?x?x13x?xf32>
//  CHECK-NEXT:  llvm.extractvalue %[[ld]][3, 4] : !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
  %4 = dim %mixed, 4 : memref<42x?x?x13x?xf32>
  return
}
