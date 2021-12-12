// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

func @testenterdataop(%a: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.enter_data if(%ifCond) create(%a: memref<10xf32>)
  return
}

// CHECK: acc.enter_data create(%{{.*}} : memref<10xf32>)

// -----

func @testenterdataop(%a: memref<10xf32>) -> () {
  %ifCond = arith.constant false
  acc.enter_data if(%ifCond) create(%a: memref<10xf32>)
  return
}

// CHECK: func @testenterdataop
// CHECK-NOT: acc.enter_data

// -----

func @testexitdataop(%a: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.exit_data if(%ifCond) delete(%a: memref<10xf32>)
  return
}

// CHECK: acc.exit_data delete(%{{.*}} : memref<10xf32>)

// -----

func @testexitdataop(%a: memref<10xf32>) -> () {
  %ifCond = arith.constant false
  acc.exit_data if(%ifCond) delete(%a: memref<10xf32>)
  return
}

// CHECK: func @testexitdataop
// CHECK-NOT: acc.exit_data

// -----

func @testupdateop(%a: memref<10xf32>) -> () {
  %ifCond = arith.constant true
  acc.update if(%ifCond) host(%a: memref<10xf32>)
  return
}

// CHECK: acc.update host(%{{.*}} : memref<10xf32>)

// -----

func @testupdateop(%a: memref<10xf32>) -> () {
  %ifCond = arith.constant false
  acc.update if(%ifCond) host(%a: memref<10xf32>)
  return
}

// CHECK: func @testupdateop
// CHECK-NOT: acc.update

// -----

func @testenterdataop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.enter_data if(%ifCond) create(%a: memref<10xf32>)
  return
}

// CHECK:  func @testenterdataop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:    acc.enter_data if(%{{.*}}) create(%{{.*}} : memref<10xf32>)

// -----

func @testexitdataop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.exit_data if(%ifCond) delete(%a: memref<10xf32>)
  return
}

// CHECK: func @testexitdataop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:   acc.exit_data if(%{{.*}}) delete(%{{.*}} : memref<10xf32>)

// -----

func @testupdateop(%a: memref<10xf32>, %ifCond: i1) -> () {
  acc.update if(%ifCond) host(%a: memref<10xf32>)
  return
}

// CHECK:  func @testupdateop(%{{.*}}: memref<10xf32>, [[IFCOND:%.*]]: i1)
// CHECK:    acc.update if(%{{.*}}) host(%{{.*}} : memref<10xf32>)
