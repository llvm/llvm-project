// RUN: tr-opt %s -split-input-file -verify-diagnostics

// Parser: tiles cannot be dynamic.
func.func @dynamic_tile(%t: !tr.tile<?xf32>) {
  // expected-error@-1 {{tile dimensions must be static}}
  return
}

// -----

// Parser: named extents are dynamic, so they are illegal on tiles.
func.func @named_tile(%t: !tr.tile<Mxf32>) {
  // expected-error@-1 {{tile dimensions must be static}}
  return
}

// -----

// Verifier: zero extent. Spaces keep `0 x f32` from lexing as the hex
// integer `0xf32` (the same special case builtin tensor types have).
func.func @zero_tile(%t: !tr.tile<0 x f32>) {
  // expected-error@-1 {{tile dimensions must be positive}}
  return
}

// -----

// Verifier: element type.
func.func @bad_elem(%t: !tr.tile<128xindex>) {
  // expected-error@-1 {{element type must be an integer or float}}
  return
}

// -----

// Verifier: rank-0 buffer.
func.func @scalar_buffer(%b: !tr.buffer<f32>) {
  // expected-error@-1 {{buffer must have rank >= 1}}
  return
}
