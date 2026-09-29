// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

llvm.func @pmevent_no_id() {
  // expected-error @below {{either `id` or `mask` must be set}}
  nvvm.pmevent 
}

// -----

llvm.func @pmevent_bigger15() {
  // expected-error @below {{`id` must be between 0 and 15}}
  nvvm.pmevent id  = 16
}

// -----

llvm.func @pmevent_many_ids() {
  // expected-error @below {{`id` and `mask` cannot be set at the same time}}
  nvvm.pmevent id = 1 mask = 1
}

