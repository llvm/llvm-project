// RUN: mlir-opt %s -convert-openacc-to-scf -split-input-file | FileCheck %s

func.func @testenterdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  acc.enter_data if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}

// CHECK:      func @testenterdataop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK-NEXT:     acc.enter_data dataOperands(%{{.*}} : memref<f32>)
// CHECK-NEXT:   }

// -----

func.func @testexitdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.exit_data if(%ifCond) dataOperands(%0 : memref<f32>)
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK:      func @testexitdataop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK-NEXT:     acc.exit_data dataOperands(%{{.*}} : memref<f32>)
// CHECK-NEXT:   }

// -----

func.func @testupdateop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.update_device varPtr(%a : memref<f32>) -> memref<f32>
  acc.update if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}

// CHECK:      func @testupdateop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:        scf.if [[IFCOND]] {
// CHECK:          acc.update dataOperands(%{{.*}} : memref<f32>)
// CHECK-NEXT:   }

// -----

func.func @update_true(%arg0: memref<f32>) {
  %true = arith.constant true
  %0 = acc.update_device varPtr(%arg0 : memref<f32>) -> memref<f32>
  acc.update if(%true) dataOperands(%0 : memref<f32>)
  return
}

// CHECK-LABEL: func.func @update_true
// CHECK-NOT:     if
// CHECK:         acc.update dataOperands

// -----

func.func @update_false(%arg0: memref<f32>) {
  %false = arith.constant false
  %0 = acc.update_device varPtr(%arg0 : memref<f32>) -> memref<f32>
  acc.update if(%false) dataOperands(%0 : memref<f32>)
  return
}

// CHECK-LABEL: func.func @update_false
// CHECK-NOT:     acc.update dataOperands

// -----

func.func @enter_data_true(%d1 : memref<f32>) {
  %true = arith.constant true
  %0 = acc.create varPtr(%d1 : memref<f32>) -> memref<f32>
  acc.enter_data if(%true) dataOperands(%0 : memref<f32>) attributes {async}
  return
}

// CHECK-LABEL: func.func @enter_data_true
// CHECK-NOT:     if
// CHECK:           acc.enter_data dataOperands

// -----

func.func @enter_data_false(%d1 : memref<f32>) {
  %false = arith.constant false
  %0 = acc.create varPtr(%d1 : memref<f32>) -> memref<f32>
  acc.enter_data if(%false) dataOperands(%0 : memref<f32>) attributes {async}
  return
}

// CHECK-LABEL: func.func @enter_data_false
// CHECK-NOT:     acc.enter_data

// -----

func.func @exit_data_true(%d1 : memref<f32>) {
  %true = arith.constant true
  %0 = acc.getdeviceptr varPtr(%d1 : memref<f32>) -> memref<f32>
  acc.exit_data if(%true) dataOperands(%0 : memref<f32>) attributes {async}
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK-LABEL: func.func @exit_data_true
// CHECK-NOT:if
// CHECK:acc.exit_data dataOperands

// -----

func.func @exit_data_false(%d1 : memref<f32>) {
  %false = arith.constant false
  %0 = acc.getdeviceptr varPtr(%d1 : memref<f32>) -> memref<f32>
  acc.exit_data if(%false) dataOperands(%0 : memref<f32>) attributes {async}
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK-LABEL: func.func @exit_data_false
// CHECK-NOT:acc.exit_data

// -----
