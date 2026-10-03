// RUN: tr-opt %s -split-input-file -verify-diagnostics

// Verifier: reduce axis out of range.
func.func @bad_axis(%t: !tr.tile<128x128xf32>) -> !tr.tile<128xf32> {
  // expected-error@+1 {{axis 2 is out of range}}
  %r = tr.reduce_sum %t, axis = 2 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// -----

// Verifier: reduce result extent.
func.func @bad_reduce_shape(%t: !tr.tile<128x64xf32>) -> !tr.tile<32xf32> {
  // expected-error@+1 {{result extent must match the non-reduced dimensions}}
  %r = tr.reduce_sum %t, axis = 1 : !tr.tile<128x64xf32> -> !tr.tile<32xf32>
  return %r : !tr.tile<32xf32>
}

// -----

// Parser: SameOperandsAndResultType unifies the printed type onto both
// operands, so a shape mismatch is diagnosed as a prior-use type error.
func.func @bad_add(%a: !tr.tile<128xf32>,
                   %b: !tr.tile<64xf32>) -> !tr.tile<128xf32> {
  // expected-note@-1 {{prior use here}}
  // expected-error@+1 {{expects different type than prior uses}}
  %r = tr.add %a, %b : !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// -----

// Verifier: load index rank.
func.func @bad_load(%in: !tr.buffer<MxKxf32>, %i: index) -> !tr.tile<128x128xf32> {
  // expected-error@+1 {{expected 2 tile load indices, got 1}}
  %t = tr.load %in[%i] : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>
  return %t : !tr.tile<128x128xf32>
}

// -----

// Verifier: dim axis.
func.func @bad_dim(%in: !tr.buffer<Mxf32>) -> index {
  // expected-error@+1 {{axis 3 is out of range}}
  %d = tr.dim %in, 3 : !tr.buffer<Mxf32>, index
  return %d : index
}

// -----

// Parser: reduce assembly requires `axis =`.
func.func @parse_reduce(%t: !tr.tile<128x128xf32>) -> !tr.tile<128xf32> {
  // expected-error@+1 {{expected 'axis'}}
  %r = tr.reduce_sum %t, 1 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}
