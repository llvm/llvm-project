// RUN: mlir-opt %s -acc-to-llvm -split-input-file | FileCheck %s

// A use_device clause becomes the call giving the device address that the body
// of the construct works on, and the body takes the place of the construct.
// CHECK-LABEL: llvm.func @host_data
// CHECK-NOT: acc.host_data
// CHECK-NOT: acc.use_device
// CHECK: %[[NONE:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[DEV:.*]] = llvm.call @__tgt_acc_get_deviceptr({{.*}}, %[[VAR:.*]], %[[NONE]], %[[VAR]])
// CHECK: llvm.br ^bb[[BODY:[0-9]+]]
// CHECK: ^bb[[BODY]]:
// CHECK: llvm.store %{{.*}}, %[[DEV]]
func.func @host_data(%arg0: !llvm.ptr, %value: f32) {
  %devptr = acc.use_device varPtr(%arg0 : !llvm.ptr) varType(f32)
      name("a") -> !llvm.ptr
  acc.host_data dataOperands(%devptr : !llvm.ptr) {
    llvm.store %value, %devptr : f32, !llvm.ptr
    acc.terminator
  }
  return
}

// -----

// Each use_device clause of the construct is asked for on its own.
// CHECK-LABEL: llvm.func @host_data_two_clauses
// CHECK: llvm.call @__tgt_acc_get_deviceptr
// CHECK: llvm.call @__tgt_acc_get_deviceptr
func.func @host_data_two_clauses(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  %a = acc.use_device varPtr(%arg0 : !llvm.ptr) varType(f32)
      name("a") -> !llvm.ptr
  %b = acc.use_device varPtr(%arg1 : !llvm.ptr) varType(f32)
      name("b") -> !llvm.ptr
  acc.host_data dataOperands(%a, %b : !llvm.ptr, !llvm.ptr) {
    %value = llvm.load %a : !llvm.ptr -> f32
    llvm.store %value, %b : f32, !llvm.ptr
    acc.terminator
  }
  return
}

// -----

// An if_present clause is passed to the runtime, which then leaves an object
// that is not mapped on its host address.
// CHECK-LABEL: llvm.func @host_data_if_present
// CHECK: %[[IF_PRESENT:.*]] = llvm.mlir.constant(524288 : i64) : i64
// CHECK: llvm.call @__tgt_acc_get_deviceptr({{.*}}, %{{.*}}, %[[IF_PRESENT]], %{{.*}})
func.func @host_data_if_present(%arg0: !llvm.ptr, %value: f32) {
  %devptr = acc.use_device varPtr(%arg0 : !llvm.ptr) varType(f32)
      name("a") -> !llvm.ptr
  acc.host_data dataOperands(%devptr : !llvm.ptr) {
    llvm.store %value, %devptr : f32, !llvm.ptr
    acc.terminator
  } ifPresent
  return
}

// -----

// An if clause decides between the device address and the host one, and the
// body of the construct works on whichever the condition selected.
// CHECK-LABEL: llvm.func @host_data_if
// CHECK-SAME: (%[[VAR:.*]]: !llvm.ptr, %[[VALUE:.*]]: f32, %[[COND:.*]]: i1)
// CHECK: llvm.cond_br %[[COND]], ^bb[[THEN:[0-9]+]], ^bb[[ELSE:[0-9]+]]
// CHECK: ^bb[[THEN]]:
// CHECK: %[[DEV:.*]] = llvm.call @__tgt_acc_get_deviceptr
// CHECK: llvm.br ^bb[[SELECTED:[0-9]+]](%[[DEV]] : !llvm.ptr)
// CHECK: ^bb[[ELSE]]:
// CHECK: llvm.br ^bb[[SELECTED]](%[[VAR]] : !llvm.ptr)
// CHECK: ^bb[[SELECTED]](%[[PTR:.*]]: !llvm.ptr):
// CHECK: llvm.br ^bb[[BODY:[0-9]+]]
// CHECK: ^bb[[BODY]]:
// CHECK: llvm.store %[[VALUE]], %[[PTR]]
func.func @host_data_if(%arg0: !llvm.ptr, %value: f32, %cond: i1) {
  %devptr = acc.use_device varPtr(%arg0 : !llvm.ptr) varType(f32)
      name("a") -> !llvm.ptr
  acc.host_data if(%cond) dataOperands(%devptr : !llvm.ptr) {
    llvm.store %value, %devptr : f32, !llvm.ptr
    acc.terminator
  }
  return
}

// -----

// Bounds on the clause say which part of the object it names, which the
// address the runtime is asked about does not state: the body addresses that
// part of the object as it would on the host.
// CHECK-LABEL: llvm.func @host_data_section
// CHECK-SAME: (%[[VAR:.*]]: !llvm.ptr, %[[VALUE:.*]]: f32)
// CHECK: %[[DEV:.*]] = llvm.call @__tgt_acc_get_deviceptr({{.*}}, %[[VAR]], %{{.*}}, %[[VAR]])
// CHECK: %[[ELEM:.*]] = llvm.getelementptr %[[DEV]]
// CHECK: llvm.store %[[VALUE]], %[[ELEM]]
func.func @host_data_section(%arg0: !llvm.ptr, %value: f32) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c4 : index)
      extent(%c5 : index) stride(%c1 : index) startIdx(%c1 : index)
  %devptr = acc.use_device varPtr(%arg0 : !llvm.ptr) varType(f32)
      bounds(%bounds) name("a(1:5)") -> !llvm.ptr
  acc.host_data dataOperands(%devptr : !llvm.ptr) {
    %elem = llvm.getelementptr %devptr[2] : (!llvm.ptr) -> !llvm.ptr, f32
    llvm.store %value, %elem : f32, !llvm.ptr
    acc.terminator
  }
  return
}
