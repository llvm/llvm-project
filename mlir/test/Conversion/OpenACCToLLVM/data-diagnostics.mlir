// RUN: mlir-opt %s -acc-to-llvm -verify-diagnostics

func.func @wait_devnum(%arg0: !llvm.ptr, %devnum: i64, %queue: i64) {
  %size = arith.constant 4 : i64
  %map = acc.map_info varPtr(%arg0 : !llvm.ptr) varType(i32)
      size(%size : i64) elementSize(4) descKind(none)
      mapFlags(to) -> !llvm.ptr
  // expected-error @below {{not yet implemented: wait clause with a devnum modifier}}
  // expected-error @below {{failed to legalize operation 'acc.enter_data'}}
  acc.enter_data dataOperands(%map : !llvm.ptr)
      wait_devnum(%devnum : i64) wait(%queue : i64)
  return
}
