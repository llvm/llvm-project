// RUN: mlir-opt %s -acc-to-llvm -verify-diagnostics -split-input-file

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

// -----

// The clause below states the mapping that the exit data directive tears down,
// which is lowered on its own, and is at the same time read by the load. The
// exit call is given the host address of the object, so the clause cannot
// stand for its device address as well.
func.func @getdeviceptr_mapping_and_value(%arg0: !llvm.ptr) -> i32 {
  // expected-error @below {{not yet implemented: device address read from a clause that also states a mapping of the object}}
  // expected-error @below {{failed to legalize operation 'acc.getdeviceptr'}}
  %devptr = acc.getdeviceptr varPtr(%arg0 : !llvm.ptr) varType(i32)
      dataClause(acc_delete) structured(false) -> !llvm.ptr
  acc.exit_data dataOperands(%devptr : !llvm.ptr)
  acc.delete accPtr(%devptr : !llvm.ptr) dataClause(acc_delete)
      structured(false)
  %value = llvm.load %devptr : !llvm.ptr -> i32
  return %value : i32
}
