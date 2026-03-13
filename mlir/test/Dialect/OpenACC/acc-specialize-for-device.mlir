// RUN: mlir-opt %s -acc-specialize-for-device | FileCheck %s

//===----------------------------------------------------------------------===//
// Data entry ops in specialized routines
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_0 func(@attach) seq
// CHECK-LABEL: func.func @attach
// CHECK-NOT:   acc.attach
func.func @attach(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_0, <seq>, "attach">} {
  %c0 = arith.constant 0 : i32
  %0 = acc.attach varPtr(%arg0 : memref<i32>) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

acc.routine @acc_routine_1 func(@copyin) seq
// CHECK-LABEL: func.func @copyin
// CHECK-NOT:   acc.copyin
func.func @copyin(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_1, <seq>, "copyin">} {
  %c0 = arith.constant 0 : i32
  %0 = acc.copyin varPtr(%arg0 : memref<i32>) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

acc.routine @acc_routine_2 func(@create) seq
// CHECK-LABEL: func.func @create
// CHECK-NOT:   acc.create
func.func @create(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_2, <seq>, "create">} {
  %c0 = arith.constant 0 : i32
  %0 = acc.create varPtr(%arg0 : memref<i32>) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

acc.routine @acc_routine_3 func(@present) seq
// CHECK-LABEL: func.func @present
// CHECK-NOT:   acc.present
func.func @present(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_3, <seq>, "present">} {
  %c0 = arith.constant 0 : i32
  %0 = acc.present varPtr(%arg0 : memref<i32>) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

//===----------------------------------------------------------------------===//
// Data entry ops INSIDE compute constructs (non-specialized functions)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @copyin_inside_parallel
// CHECK:       acc.parallel
// CHECK-NOT:   acc.copyin
// CHECK:       acc.yield
func.func @copyin_inside_parallel(%arg0 : memref<i32>) {
  %c0 = arith.constant 0 : i32
  acc.parallel {
    %0 = acc.copyin varPtr(%arg0 : memref<i32>) -> memref<i32>
    memref.store %c0, %0[] : memref<i32>
    acc.yield
  }
  return
}

//===----------------------------------------------------------------------===//
// Data entry ops OUTSIDE compute constructs should NOT be removed
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @copyin_outside_parallel
// CHECK:       acc.copyin
// CHECK:       acc.parallel
func.func @copyin_outside_parallel(%arg0 : memref<i32>) {
  %c0 = arith.constant 0 : i32
  %0 = acc.copyin varPtr(%arg0 : memref<i32>) -> memref<i32>
  acc.parallel dataOperands(%0 : memref<i32>) {
    memref.store %c0, %0[] : memref<i32>
    acc.yield
  }
  return
}

//===----------------------------------------------------------------------===//
// Data exit ops in specialized routines
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_copyout func(@copyout) worker
// CHECK-LABEL: func.func @copyout
// CHECK-NOT:   acc.copyout
func.func @copyout(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_copyout, <worker>, "copyout">} {
  %0 = acc.copyin varPtr(%arg0 : memref<i32>) -> memref<i32>
  acc.copyout accPtr(%0 : memref<i32>) to varPtr(%arg0 : memref<i32>)
  return
}

acc.routine @acc_routine_delete func(@delete) worker
// CHECK-LABEL: func.func @delete
// CHECK-NOT:   acc.delete
func.func @delete(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_delete, <worker>, "delete">} {
  %0 = acc.create varPtr(%arg0 : memref<i32>) -> memref<i32>
  acc.delete accPtr(%0 : memref<i32>)
  return
}

//===----------------------------------------------------------------------===//
// Erase ops (unstructured data and runtime ops)
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_enter_data func(@enter_data) worker
// CHECK-LABEL: func.func @enter_data
// CHECK-NOT:   acc.enter_data
func.func @enter_data(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_enter_data, <worker>, "enter_data">} {
  %0 = acc.create varPtr(%arg0 : memref<i32>) -> memref<i32>
  acc.enter_data dataOperands(%0 : memref<i32>)
  return
}

acc.routine @acc_routine_init func(@init_op) worker
// CHECK-LABEL: func.func @init_op
// CHECK-NOT:   acc.init
func.func @init_op() attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_init, <worker>, "init_op">} {
  acc.init
  return
}

acc.routine @acc_routine_wait func(@wait_op) worker
// CHECK-LABEL: func.func @wait_op
// CHECK-NOT:   acc.wait
func.func @wait_op() attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_wait, <worker>, "wait_op">} {
  acc.wait
  return
}

//===----------------------------------------------------------------------===//
// Region unwrap (structured data and compute constructs)
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_data func(@data_construct) worker
// CHECK-LABEL: func.func @data_construct
// CHECK-NOT:   acc.data
// CHECK:       arith.constant 42
func.func @data_construct(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_data, <worker>, "data_construct">} {
  %d = acc.create varPtr(%arg0 : memref<i32>) -> memref<i32>
  acc.data dataOperands(%d : memref<i32>) {
    %c42 = arith.constant 42 : i32
    memref.store %c42, %arg0[] : memref<i32>
    acc.terminator
  }
  return
}

acc.routine @acc_routine_parallel func(@parallel_construct) worker
// CHECK-LABEL: func.func @parallel_construct
// CHECK-NOT:   acc.parallel
// CHECK:       arith.constant 44
func.func @parallel_construct(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_parallel, <worker>, "parallel_construct">} {
  acc.parallel {
    %c44 = arith.constant 44 : i32
    memref.store %c44, %arg0[] : memref<i32>
    acc.yield
  }
  return
}

acc.routine @acc_routine_serial func(@serial_construct) worker
// CHECK-LABEL: func.func @serial_construct
// CHECK-NOT:   acc.serial
// CHECK:       arith.constant 45
func.func @serial_construct(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_serial, <worker>, "serial_construct">} {
  acc.serial {
    %c45 = arith.constant 45 : i32
    memref.store %c45, %arg0[] : memref<i32>
    acc.yield
  }
  return
}

acc.routine @acc_routine_kernels func(@kernels_construct) worker
// CHECK-LABEL: func.func @kernels_construct
// CHECK-NOT:   acc.kernels
// CHECK:       arith.constant 46
func.func @kernels_construct(%arg0 : memref<i32>) attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_kernels, <worker>, "kernels_construct">} {
  acc.kernels {
    %c46 = arith.constant 46 : i32
    memref.store %c46, %arg0[] : memref<i32>
    acc.terminator
  }
  return
}

//===----------------------------------------------------------------------===//
// Declare enter/exit strip in device routines
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_declare func(@dev_routine_declare) worker
// CHECK-LABEL: func.func @dev_routine_declare
// CHECK-NOT: acc.declare_enter
// CHECK-NOT: acc.declare_exit
func.func @dev_routine_declare() attributes {acc.specialized_routine = #acc.specialized_routine<@acc_routine_declare, <worker>, "dev_routine_declare">} {
  %var = memref.alloca() : memref<f32>
  %c = acc.create varPtr(%var : memref<f32>) -> memref<f32>
  %t = acc.declare_enter dataOperands(%c : memref<f32>)
  acc.declare_exit token(%t) dataOperands(%c : memref<f32>)
  return
}
