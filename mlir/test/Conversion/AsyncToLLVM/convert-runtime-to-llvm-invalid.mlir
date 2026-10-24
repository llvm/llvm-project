// RUN: mlir-opt %s -convert-async-to-llvm -split-input-file -verify-diagnostics

// Computing the storage size of an async value emits a GEP over the converted
// payload type. Payload types without an LLVM counterpart must be rejected
// instead of building the GEP with a null type.

func.func @create_value_unsupported_payload() {
  // expected-error @below {{failed to legalize operation 'async.runtime.create'}}
  %0 = async.runtime.create : !async.value<tensor<4xf32>>
  return
}

// -----

func.func @create_value_unsupported_payload_index() {
  // expected-error @below {{failed to legalize operation 'async.runtime.create'}}
  %0 = async.runtime.create : !async.value<tensor<?xindex>>
  return
}
