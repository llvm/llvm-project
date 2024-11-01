// RUN: mlir-opt %s -test-vector-transfer-drop-unit-dims-patterns -split-input-file | FileCheck %s

func.func @transfer_read_rank_reducing(
      %arg : memref<1x1x3x2xi8, strided<[6, 6, 2, 1], offset: ?>>) -> vector<3x2xi8> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0 : i8
    %v = vector.transfer_read %arg[%c0, %c0, %c0, %c0], %cst :
      memref<1x1x3x2xi8, strided<[6, 6, 2, 1], offset: ?>>, vector<3x2xi8>
    return %v : vector<3x2xi8>
}

// CHECK-LABEL: func @transfer_read_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<1x1x3x2xi8
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0, 0, 0] [1, 1, 3, 2] [1, 1, 1, 1]
//  CHECK-SAME:     memref<1x1x3x2xi8, {{.*}}> to memref<3x2xi8, {{.*}}>
//       CHECK:   vector.transfer_read %[[SUBVIEW]]

// -----

func.func @transfer_write_rank_reducing(%arg : memref<1x1x3x2xi8, strided<[6, 6, 2, 1], offset: ?>>, %vec : vector<3x2xi8>) {
    %c0 = arith.constant 0 : index
    vector.transfer_write %vec, %arg [%c0, %c0, %c0, %c0] :
      vector<3x2xi8>, memref<1x1x3x2xi8, strided<[6, 6, 2, 1], offset: ?>>
    return
}

// CHECK-LABEL: func @transfer_write_rank_reducing
//  CHECK-SAME:     %[[ARG:.+]]: memref<1x1x3x2xi8
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG]][0, 0, 0, 0] [1, 1, 3, 2] [1, 1, 1, 1]
//  CHECK-SAME:     memref<1x1x3x2xi8, {{.*}}> to memref<3x2xi8, {{.*}}>
//       CHECK:   vector.transfer_write %{{.*}}, %[[SUBVIEW]]