// RUN: mlir-opt %s -split-input-file -verify-diagnostics

llvm.func @pad_waitcnt_rejects_unsupported_counter() {
  // expected-error@+1 {{attribute 'counter' failed to satisfy constraint}}
  rocdl.pad.waitcnt 1, 1
  llvm.return
}

// -----

llvm.func @pad_waitcnt_rejects_negative_counter() {
  // expected-error@+1 {{attribute 'counter' failed to satisfy constraint}}
  rocdl.pad.waitcnt -1, 1
  llvm.return
}

// -----

llvm.func @pad_waitcnt_rejects_zero_count() {
  // expected-error@+1 {{attribute 'count' failed to satisfy constraint}}
  rocdl.pad.waitcnt 0, 0
  llvm.return
}

// -----

llvm.func @pad_waitcnt_rejects_negative_count() {
  // expected-error@+1 {{attribute 'count' failed to satisfy constraint}}
  rocdl.pad.waitcnt 0, -1
  llvm.return
}
