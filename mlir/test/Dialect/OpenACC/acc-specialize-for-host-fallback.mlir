// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(acc-specialize-for-host{enable-host-fallback=true}))' | FileCheck %s

//===----------------------------------------------------------------------===//
// Data entry ops - replaced with var (host fallback)
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_create func(@create) seq
// CHECK-LABEL: func.func @create
// CHECK-NOT:   acc.create
func.func @create(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_create]>} {
  %c0 = arith.constant 0 : i32
  %0 = acc.create varPtr(%arg0 : memref<i32>) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

acc.routine @acc_routine_copyin func(@copyin) seq
// CHECK-LABEL: func.func @copyin
// CHECK-NOT:   acc.copyin
func.func @copyin(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_copyin]>} {
  %c0 = arith.constant 0 : i32
  %0 = acc.copyin varPtr(%arg0 : memref<i32>) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

acc.routine @acc_routine_present func(@present) seq
// CHECK-LABEL: func.func @present
// CHECK-NOT:   acc.present
func.func @present(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_present]>} {
  %c0 = arith.constant 0 : i32
  %0 = acc.present varPtr(%arg0 : memref<i32>) -> memref<i32>
  memref.store %c0, %0[] : memref<i32>
  return
}

//===----------------------------------------------------------------------===//
// Data exit ops - erased (host fallback)
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_copyout func(@copyout) seq
// CHECK-LABEL: func.func @copyout
// CHECK-NOT:   acc.copyout
func.func @copyout(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_copyout]>} {
  %0 = acc.copyin varPtr(%arg0 : memref<i32>) -> memref<i32>
  acc.copyout accPtr(%0 : memref<i32>) to varPtr(%arg0 : memref<i32>)
  return
}

acc.routine @acc_routine_delete func(@delete) seq
// CHECK-LABEL: func.func @delete
// CHECK-NOT:   acc.delete
func.func @delete(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_delete]>} {
  %0 = acc.create varPtr(%arg0 : memref<i32>) -> memref<i32>
  acc.delete accPtr(%0 : memref<i32>)
  return
}

//===----------------------------------------------------------------------===//
// Runtime operations - erased (host fallback)
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_init func(@init_op) seq
// CHECK-LABEL: func.func @init_op
// CHECK-NOT:   acc.init
func.func @init_op() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_init]>} {
  acc.init
  return
}

acc.routine @acc_routine_shutdown func(@shutdown_op) seq
// CHECK-LABEL: func.func @shutdown_op
// CHECK-NOT:   acc.shutdown
func.func @shutdown_op() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_shutdown]>} {
  acc.shutdown
  return
}

acc.routine @acc_routine_wait func(@wait_op) seq
// CHECK-LABEL: func.func @wait_op
// CHECK-NOT:   acc.wait
func.func @wait_op() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_wait]>} {
  acc.wait
  return
}

//===----------------------------------------------------------------------===//
// Structured data and compute constructs - unwrap regions (host fallback)
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_data func(@data_construct) seq
// CHECK-LABEL: func.func @data_construct
// CHECK-NOT:   acc.data
// CHECK:       arith.constant 42
func.func @data_construct(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_data]>} {
  %0 = acc.create varPtr(%arg0 : memref<i32>) -> memref<i32>
  acc.data dataOperands(%0 : memref<i32>) {
    %c42 = arith.constant 42 : i32
    memref.store %c42, %arg0[] : memref<i32>
    acc.terminator
  }
  return
}

acc.routine @acc_routine_parallel func(@parallel_construct) seq
// CHECK-LABEL: func.func @parallel_construct
// CHECK-NOT:   acc.parallel
// CHECK:       arith.constant 44
func.func @parallel_construct(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_parallel]>} {
  acc.parallel {
    %c44 = arith.constant 44 : i32
    memref.store %c44, %arg0[] : memref<i32>
    acc.yield
  }
  return
}

acc.routine @acc_routine_serial func(@serial_construct) seq
// CHECK-LABEL: func.func @serial_construct
// CHECK-NOT:   acc.serial
// CHECK:       arith.constant 45
func.func @serial_construct(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_serial]>} {
  acc.serial {
    %c45 = arith.constant 45 : i32
    memref.store %c45, %arg0[] : memref<i32>
    acc.yield
  }
  return
}

acc.routine @acc_routine_kernels func(@kernels_construct) seq
// CHECK-LABEL: func.func @kernels_construct
// CHECK-NOT:   acc.kernels
// CHECK:       arith.constant 46
func.func @kernels_construct(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_kernels]>} {
  acc.kernels {
    %c46 = arith.constant 46 : i32
    memref.store %c46, %arg0[] : memref<i32>
    acc.terminator
  }
  return
}

//===----------------------------------------------------------------------===//
// Declare enter/exit - erased (host fallback)
//===----------------------------------------------------------------------===//

acc.routine @acc_routine_declare func(@declare_enter_exit) seq
// CHECK-LABEL: func.func @declare_enter_exit
// CHECK-NOT:   acc.declare_enter
// CHECK-NOT:   acc.declare_exit
func.func @declare_enter_exit(%arg0 : memref<i32>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_declare]>} {
  %0 = acc.create varPtr(%arg0 : memref<i32>) -> memref<i32>
  %token = acc.declare_enter dataOperands(%0 : memref<i32>)
  acc.declare_exit token(%token) dataOperands(%0 : memref<i32>)
  return
}
