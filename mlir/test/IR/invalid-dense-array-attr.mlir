// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// Reject index element type in dense array attribute.
// expected-error@+1 {{expected integer or float type, got: 'index'}}
module attributes { a = array<index: 1, 2> } {}

// -----
// Reject tensor element type in dense array attribute.
// expected-error@+1 {{expected integer or float type, got: 'tensor<2xi32>'}}
module attributes { a = array<tensor<2xi32>: 1, 2> } {}

// -----
// Reject memref element type in dense array attribute.
// expected-error@+1 {{expected integer or float type, got: 'memref<2xi32>'}}
module attributes { a = array<memref<2xi32>: 1, 2> } {}

// -----
// Reject complex element type in dense array attribute.
// expected-error@+1 {{expected integer or float type, got: 'complex<f32>'}}
module attributes { a = array<complex<f32>: 1, 2> } {} 

