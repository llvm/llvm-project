// RUN: mlir-opt %s -split-input-file -verify-diagnostics

llvm.func @pad_waitcnt_rejects_numeric_counter() {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values for attribute 'counter' [load]}}
  rocdl.pad.waitcnt 1, 1
  llvm.return
}

// -----

llvm.func @pad_waitcnt_rejects_integer_counter_attr() {
  // expected-error@+1 {{attribute 'counter' failed to satisfy constraint}}
  "rocdl.pad.waitcnt"() {counter = 0 : i32, count = 1 : i32} : () -> ()
  llvm.return
}

// -----

llvm.func @pad_waitcnt_rejects_zero_count() {
  // expected-error@+1 {{attribute 'count' failed to satisfy constraint}}
  rocdl.pad.waitcnt load, 0
  llvm.return
}

// -----

llvm.func @pad_waitcnt_rejects_negative_count() {
  // expected-error@+1 {{attribute 'count' failed to satisfy constraint}}
  rocdl.pad.waitcnt load, -1
  llvm.return
}
