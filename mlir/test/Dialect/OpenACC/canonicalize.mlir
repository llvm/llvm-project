// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

func.func @testenterdataop(%a: memref<f32>) -> () {
  %ifCond = arith.constant true
  %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  acc.enter_data if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}

// CHECK: acc.enter_data dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testenterdataop(%a: memref<f32>) -> () {
  %ifCond = arith.constant false
  %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  acc.enter_data if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}

// CHECK: func @testenterdataop
// CHECK-NOT: acc.enter_data

// -----

func.func @testexitdataop(%a: memref<f32>) -> () {
  %ifCond = arith.constant true
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.exit_data if(%ifCond) dataOperands(%0 : memref<f32>)
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK: acc.exit_data dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testexitdataop(%a: memref<f32>) -> () {
  %ifCond = arith.constant false
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.exit_data if(%ifCond) dataOperands(%0 : memref<f32>)
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK: func @testexitdataop
// CHECK-NOT: acc.exit_data

// -----

func.func @testupdateop(%a: memref<f32>) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.update_host accPtr(%0 : memref<f32>) to varPtr(%a : memref<f32>)
  %ifCond = arith.constant true
  acc.update if(%ifCond) dataOperands(%0: memref<f32>)
  return
}

// CHECK: acc.update dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testupdateop(%a: memref<f32>) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.update_host accPtr(%0 : memref<f32>) to varPtr(%a : memref<f32>)
  %ifCond = arith.constant false
  acc.update if(%ifCond) dataOperands(%0: memref<f32>)
  return
}

// CHECK: func @testupdateop
// CHECK-NOT: acc.update{{.$}}

// -----

func.func @testenterdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.create varPtr(%a : memref<f32>) -> memref<f32>
  acc.enter_data if(%ifCond) dataOperands(%0 : memref<f32>)
  return
}

// CHECK:  func @testenterdataop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:    acc.enter_data if(%{{.*}}) dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testexitdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.exit_data if(%ifCond) dataOperands(%0 : memref<f32>)
  acc.delete accPtr(%0 : memref<f32>)
  return
}

// CHECK: func @testexitdataop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:   acc.exit_data if(%{{.*}}) dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testupdateop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.getdeviceptr varPtr(%a : memref<f32>) -> memref<f32>
  acc.update_host accPtr(%0 : memref<f32>) to varPtr(%a : memref<f32>)
  acc.update if(%ifCond) dataOperands(%0: memref<f32>)
  return
}

// CHECK:  func @testupdateop(%{{.*}}: memref<f32>, [[IFCOND:%.*]]: i1)
// CHECK:    acc.update if(%{{.*}}) dataOperands(%{{.*}} : memref<f32>)

// -----

func.func @testhostdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.use_device varPtr(%a : memref<f32>) -> memref<f32>
  %false = arith.constant false
  acc.host_data dataOperands(%0 : memref<f32>) if(%false) {
    acc.loop {
      acc.yield
    }
    acc.loop {
      acc.yield
    }
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @testhostdataop
// CHECK-NOT: acc.host_data
// CHECK: acc.loop
// CHECK: acc.yield
// CHECK: acc.loop
// CHECK: acc.yield

// -----

func.func @testhostdataop(%a: memref<f32>, %ifCond: i1) -> () {
  %0 = acc.use_device varPtr(%a : memref<f32>) -> memref<f32>
  %true = arith.constant true
  acc.host_data dataOperands(%0 : memref<f32>) if(%true) {
  }
  return
}

// CHECK-LABEL: func.func @testhostdataop
// CHECK: acc.host_data dataOperands(%{{.*}} : memref<f32>) {

// -----

func.func @update_no_op(%x : memref<i32>) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    acc.yield %xval : i32
  }
  return
}

// CHECK-LABEL: func.func @update_no_op
// CHECK-NOT: acc.atomic.update

// -----

func.func @update_write_op(%x : memref<i32>, %value: i32) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    acc.yield %value : i32
  }
  return
}

// CHECK-LABEL: func.func @update_write_op
// CHECK-SAME:            (%[[X:.+]]: memref<i32>, %[[VALUE:.+]]: i32)
// CHECK: acc.atomic.write %[[X]] = %[[VALUE]] : memref<i32>, i32
// CHECK-NOT: acc.atomic.update

// -----

func.func @update_normal(%x : memref<i32>, %value: i32) {
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval : i32):
    %newval = arith.addi %xval, %value : i32
    acc.yield %newval : i32
  }
  return
}

// CHECK-LABEL: func.func @update_normal
// CHECK: acc.atomic.update
// CHECK: arith.addi
// CHECK: acc.yield

// -----

func.func @update_unnecessary_computations(%x: memref<i32>) {
  %c0 = arith.constant 0 : i32
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = arith.addi %xval, %c0 : i32
    acc.yield %newval: i32
  }
  return
}

// CHECK-LABEL: func.func @update_unnecessary_computations
// CHECK-NOT: acc.atomic.update

// -----

func.func @update_unnecessary_computations(%x: memref<i32>) {
  %c0 = arith.constant 0 : i32
  acc.atomic.update %x : memref<i32> {
  ^bb0(%xval: i32):
    %newval = arith.muli %xval, %c0 : i32
    acc.yield %newval: i32
  }
  return
}

// CHECK-LABEL: func.func @update_unnecessary_computations
// CHECK-NOT: acc.atomic.update
// CHECK: acc.atomic.write
