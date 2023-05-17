// RUN: mlir-opt %s -test-vector-transfer-flatten-patterns -split-input-file | FileCheck %s

func.func @transfer_read_flattenable_with_offset(
      %arg : memref<5x4x3x2xi8, strided<[24, 6, 2, 1], offset: ?>>) -> vector<5x4x3x2xi8> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0 : i8
    %v = vector.transfer_read %arg[%c0, %c0, %c0, %c0], %cst :
      memref<5x4x3x2xi8, strided<[24, 6, 2, 1], offset: ?>>, vector<5x4x3x2xi8>
    return %v : vector<5x4x3x2xi8>
}

// CHECK-LABEL: func @transfer_read_flattenable_with_offset
// CHECK-SAME:      %[[ARG:[0-9a-zA-Z]+]]: memref<5x4x3x2xi8
// CHECK:         %[[COLLAPSED:.+]] = memref.collapse_shape %[[ARG]] {{.}}[0, 1, 2, 3]
// CHECK:         %[[READ1D:.+]] = vector.transfer_read %[[COLLAPSED]]
// CHECK:         %[[VEC2D:.+]] = vector.shape_cast %[[READ1D]] : vector<120xi8> to vector<5x4x3x2xi8>
// CHECK:         return %[[VEC2D]]

// -----

func.func @transfer_write_flattenable_with_offset(
      %arg : memref<5x4x3x2xi8, strided<[24, 6, 2, 1], offset: ?>>, %vec : vector<5x4x3x2xi8>) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %vec, %arg [%c0, %c0, %c0, %c0] :
      vector<5x4x3x2xi8>, memref<5x4x3x2xi8, strided<[24, 6, 2, 1], offset: ?>>
    return
}

// CHECK-LABEL: func @transfer_write_flattenable_with_offset
// CHECK-SAME:      %[[ARG:[0-9a-zA-Z]+]]: memref<5x4x3x2xi8
// CHECK-SAME:      %[[VEC:[0-9a-zA-Z]+]]: vector<5x4x3x2xi8>
// CHECK-DAG:     %[[COLLAPSED:.+]] = memref.collapse_shape %[[ARG]] {{.}}[0, 1, 2, 3]{{.}} : memref<5x4x3x2xi8, {{.+}}> into memref<120xi8, {{.+}}>
// CHECK-DAG:     %[[VEC1D:.+]] = vector.shape_cast %[[VEC]] : vector<5x4x3x2xi8> to vector<120xi8>
// CHECK:         vector.transfer_write %[[VEC1D]], %[[COLLAPSED]]

// -----

func.func @transfer_write_0d(%arg : memref<i8>, %vec : vector<i8>) {
      vector.transfer_write %vec, %arg[] : vector<i8>, memref<i8>
      return
}

// CHECK-LABEL: func @transfer_write_0d
// CHECK-SAME:       %[[ARG:.+]]: memref<i8>
// CHECK-SAME:       %[[VEC:.+]]: vector<i8>
// CHECK:          vector.transfer_write %[[VEC]], %[[ARG]][] : vector<i8>, memref<i8>
// CHECK:          return

// -----

func.func @transfer_read_0d(%arg : memref<i8>) -> vector<i8> {
      %cst = arith.constant 0 : i8
      %0 = vector.transfer_read %arg[], %cst : memref<i8>, vector<i8>
      return %0 : vector<i8>
}

// CHECK-LABEL: func @transfer_read_0d
// CHECK-SAME:       %[[ARG:.+]]: memref<i8>
// CHECK:            %[[CST:.+]] = arith.constant 0 : i8
// CHECK:            %[[READ:.+]] = vector.transfer_read %[[ARG]][], %[[CST]] : memref<i8>
// CHECK:            return %[[READ]]

// -----

func.func @transfer_read_flattenable_with_dynamic_dims_and_indices(%arg0 : memref<?x?x8x4xi8, strided<[?, 32, 4, 1], offset: ?>>, %arg1 : index, %arg2 : index) -> vector<8x4xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %result = vector.transfer_read %arg0[%arg1, %arg2, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<?x?x8x4xi8, strided<[?, 32, 4, 1], offset: ?>>, vector<8x4xi8>
    return %result : vector<8x4xi8>
}

// CHECK-LABEL: func @transfer_read_flattenable_with_dynamic_dims_and_indices
// CHECK-SAME:    %[[ARG0:.+]]: memref<?x?x8x4xi8, {{.+}}>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index
// CHECK:       %[[C0_I8:.+]] = arith.constant 0 : i8
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[COLLAPSED:.+]] = memref.collapse_shape %[[ARG0]] {{\[}}[0], [1], [2, 3]{{\]}}
// CHECK-SAME:    : memref<?x?x8x4xi8, {{.+}}> into memref<?x?x32xi8, {{.+}}>
// CHECK:       %[[VEC1D:.+]] = vector.transfer_read %[[COLLAPSED]]
// CHECK-SAME:    [%[[ARG1]], %[[ARG2]], %[[C0]]], %[[C0_I8]]
// CHECK-SAME:    {in_bounds = [true]}
// CHECK-SAME:    : memref<?x?x32xi8, {{.+}}>, vector<32xi8>
// CHECK:       %[[VEC2D:.+]] = vector.shape_cast %[[VEC1D]] : vector<32xi8> to vector<8x4xi8>
// CHECK:       return %[[VEC2D]] : vector<8x4xi8>

// -----

func.func @transfer_write_flattenable_with_dynamic_dims_and_indices(%vec : vector<8x4xi8>, %dst : memref<?x?x8x4xi8, strided<[?, 32, 4, 1], offset: ?>>, %arg1 : index, %arg2 : index) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %vec, %dst[%arg1, %arg2, %c0, %c0] {in_bounds = [true, true]} : vector<8x4xi8>, memref<?x?x8x4xi8, strided<[?, 32, 4, 1], offset: ?>>
    return
}

// CHECK-LABEL: func @transfer_write_flattenable_with_dynamic_dims_and_indices
// CHECK-SAME:    %[[ARG0:.+]]: vector<8x4xi8>, %[[ARG1:.+]]: memref<?x?x8x4xi8, {{.+}}>, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[COLLAPSED:.+]] = memref.collapse_shape %[[ARG1]] {{\[}}[0], [1], [2, 3]{{\]}}
// CHECK-SAME:    : memref<?x?x8x4xi8, {{.+}}> into memref<?x?x32xi8, {{.+}}>
// CHECK:       %[[VEC1D:.+]] = vector.shape_cast %[[ARG0]] : vector<8x4xi8> to vector<32xi8>
// CHECK:       vector.transfer_write %[[VEC1D]], %[[COLLAPSED]]
// CHECK-SAME:    [%[[ARG2]], %[[ARG3]], %[[C0]]]
// CHECK-SAME:    {in_bounds = [true]}
// CHECK-SAME:    : vector<32xi8>, memref<?x?x32xi8, {{.+}}>

// -----

func.func @transfer_read_flattenable_negative(
      %arg : memref<5x4x3x2xi8, strided<[24, 6, 2, 1], offset: ?>>) -> vector<2x2x2x2xi8> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0 : i8
    %v = vector.transfer_read %arg[%c0, %c0, %c0, %c0], %cst :
      memref<5x4x3x2xi8, strided<[24, 6, 2, 1], offset: ?>>, vector<2x2x2x2xi8>
    return %v : vector<2x2x2x2xi8>
}

// CHECK-LABEL: func @transfer_read_flattenable_negative
//       CHECK:   vector.transfer_read {{.*}} vector<2x2x2x2xi8>

// -----

func.func @transfer_read_flattenable_negative2(
      %arg : memref<5x4x3x2xi8, strided<[24, 8, 2, 1], offset: ?>>) -> vector<5x4x3x2xi8> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0 : i8
    %v = vector.transfer_read %arg[%c0, %c0, %c0, %c0], %cst :
      memref<5x4x3x2xi8, strided<[24, 8, 2, 1], offset: ?>>, vector<5x4x3x2xi8>
    return %v : vector<5x4x3x2xi8>
}

// CHECK-LABEL: func @transfer_read_flattenable_negative2
//       CHECK:   vector.transfer_read {{.*}} vector<5x4x3x2xi8>
