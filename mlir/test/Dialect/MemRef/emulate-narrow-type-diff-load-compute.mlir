// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=4 memref-load-bitwidth=8" %s | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0, s1] -> ((s0 * s1) floordiv 2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 floordiv 2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0, s1, s2, s3] -> ((s0 * s1 + s2 * s3) floordiv 2)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<()[s0, s1] -> (s0 * s1)>

// Expect no conversions, i32 is supported.
// CHECK-LABEL: func @memref_i32
// CHECK:         [[M:%.+]] = memref.alloc() : memref<4xi32, 1>
// CHECK-NEXT:    [[V:%.+]] = memref.load [[M]][{{%.+}}] : memref<4xi32, 1>
// CHECK-NEXT:    memref.store {{%.+}}, [[M]][{{%.+}}] : memref<4xi32, 1>
// CHECK-NEXT:    return
func.func @memref_i32() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %m = memref.alloc() : memref<4xi32, 1>
    %v = memref.load %m[%c0] : memref<4xi32, 1>
    memref.store %c1, %m[%c0] : memref<4xi32, 1>
    return
}

// -----

// Expect no conversions, f32 is not an integer type.
// CHECK-LABEL: func @memref_f32
// CHECK:         [[M:%.+]] = memref.alloc() : memref<4xf32, 1>
// CHECK-NEXT:    [[V:%.+]] = memref.load [[M]][{{%.+}}] : memref<4xf32, 1>
// CHECK-NEXT:    memref.store {{%.+}}, [[M]][{{%.+}}] : memref<4xf32, 1>
// CHECK-NEXT:    return
func.func @memref_f32() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1.0 : f32
    %m = memref.alloc() : memref<4xf32, 1>
    %v = memref.load %m[%c0] : memref<4xf32, 1>
    memref.store %c1, %m[%c0] : memref<4xf32, 1>
    return
}

// -----

// CHECK-LABEL: func @memref_load_i4_zero_rank
// CHECK-NEXT:    %[[M:.*]]  = memref.alloc() : memref<i8>
// CHECK-NEXT:    %[[BASE:.*]], %[[OFFSET:.*]] = memref.extract_strided_metadata %[[M]] : memref<i8> -> memref<i8>, index
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load %[[M]][] : memref<i8>
// CHECK-NEXT:    %[[I:.*]] = arith.index_castui %[[OFFSET]] : index to i8
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : i8
// CHECK-NEXT:    %[[C4:.*]] = arith.constant 4 : i8
// CHECK-NEXT:    %[[REM:.*]] = arith.remui %[[I]], %[[C2]] : i8
// CHECK-NEXT:    %[[STEP:.*]] = arith.muli %[[REM]], %[[C4]] : i8
// CHECK-NEXT:    %[[SHIFT:.*]] = arith.shrsi %[[LOAD]], %[[STEP]] : i8
// CHECK-NEXT:    %[[RES:.*]] = arith.trunci %[[SHIFT]] : i8 to i4
// CHECK-NEXT:    return
func.func @memref_load_i4_zero_rank() {
    %0 = memref.alloc() : memref<i4>
    %1 = memref.load %0[] : memref<i4>
    return
}

// -----

// CHECK-LABEL: func @memref_load_i4
// CHECK-SAME:    (%[[ARG:.*]]: index)
// CHECK-NEXT:    %[[M:.*]]  = memref.alloc() : memref<4xi8>
// CHECK-NEXT:    %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]], %[[STRIDES:.*]] = memref.extract_strided_metadata %[[M]] : memref<4xi8> -> memref<i8>, index, index, index
// CHECK-NEXT:    %[[INDEX:.*]] = affine.apply #[[$MAP0]]()[%[[ARG]], %[[STRIDES]]]
// CHECK-NEXT:    %[[AOFF:.*]] = affine.apply #[[$MAP1]]()[%[[OFFSET]]]
// CHECK-NEXT:    %[[CAST:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[AOFF]]], sizes: [%[[SIZES]]], strides: [%[[STRIDES]]] : memref<i8> to memref<4xi8, strided<[1], offset: ?>>
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load %[[CAST]][%[[INDEX]]] : memref<4xi8, strided<[1], offset: ?>>
// CHECK-NEXT:    %[[I:.*]] = arith.index_castui %[[ARG]] : index to i8
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : i8
// CHECK-NEXT:    %[[C4:.*]] = arith.constant 4 : i8
// CHECK-NEXT:    %[[REM:.*]] = arith.remui %[[I]], %[[C2]] : i8
// CHECK-NEXT:    %[[STEP:.*]] = arith.muli %[[REM]], %[[C4]] : i8
// CHECK-NEXT:    %[[SHIFT:.*]] = arith.shrsi %[[LOAD]], %[[STEP]] : i8
// CHECK-NEXT:    %[[RES:.*]] = arith.trunci %[[SHIFT]] : i8 to i4
// CHECK-NEXT:    return
func.func @memref_load_i4(%arg0: index) {
    %0 = memref.alloc() : memref<4xi4>
    %1 = memref.load %0[%arg0] : memref<4xi4>
    return
}

// -----

// CHECK-LABEL: func @memref_load_i4_rank2
// CHECK-SAME:    (%[[ARG:.*]]: memref<4x128xi8>, %[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK-NEXT:    memref.assume_alignment %[[ARG]], 64 : memref<4x128xi8>
// CHECK-NEXT:    %[[BASE:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[ARG]] : memref<4x128xi8> -> memref<i8>, index, index, index, index, index
// CHECK-NEXT:    %[[INDEX:.*]] = affine.apply #[[$MAP2]]()[%[[ARG0]], %[[STRIDES]]#0, %[[ARG1]], %[[STRIDES]]#1]
// CHECK-NEXT:    %[[LSIZE:.*]] = affine.apply #[[$MAP3]]()[%[[SIZES]]#0, %[[SIZES]]#1]
// CHECK-NEXT:    %[[AOFF:.*]] = affine.apply #[[$MAP1]]()[%[[OFFSET]]]
// CHECK-NEXT:    %[[CAST:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [%[[AOFF]]], sizes: [%[[LSIZE]]], strides: [%[[STRIDES]]#1] : memref<i8> to memref<512xi8, strided<[1], offset: ?>>
// CHECK-NEXT:    %[[LOAD:.*]] = memref.load %[[CAST]][%[[INDEX]]] : memref<512xi8, strided<[1], offset: ?>>
// CHECK-NEXT:    %[[I:.*]] = arith.index_castui %[[ARG1]] : index to i8
// CHECK-NEXT:    %[[C2:.*]] = arith.constant 2 : i8
// CHECK-NEXT:    %[[C4:.*]] = arith.constant 4 : i8
// CHECK-NEXT:    %[[REM:.*]] = arith.remui %[[I]], %[[C2]] : i8
// CHECK-NEXT:    %[[STEP:.*]] = arith.muli %[[REM]], %[[C4]] : i8
// CHECK-NEXT:    %[[SHIFT:.*]] = arith.shrsi %[[LOAD]], %[[STEP]] : i8
// CHECK-NEXT:    %[[RES:.*]] = arith.trunci %[[SHIFT]] : i8 to i4
// CHECK-NEXT:    return
func.func @memref_load_i4_rank2(%0: memref<4x128xi4>, %arg0: index, %arg1: index) {
    memref.assume_alignment %0, 64 : memref<4x128xi4>
    %1 = memref.load %0[%arg0,%arg1] : memref<4x128xi4>
    return
}
