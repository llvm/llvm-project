// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-cf -split-input-file -verify-diagnostics %s

// CHECK: scf.execute_region
func.func @execute_region_with_token(%cond: i1, %val: f32) -> f32 {
  // expected-error @below {{failed to legalize operation 'scf.execute_region'}}
  %res = scf.execute_region -> f32 {
  ^bb0(%t: token):
    scf.yield %val : f32
  }
  return %res : f32
}

// -----

// `scf.if` with a region-breaking terminator (`scf.break`) cannot be lowered
// by SCFToControlFlow yet.
func.func @if_with_break(%cond: i1, %val: f32) -> f32 {
  // expected-error @below {{failed to legalize operation 'scf.execute_region'}}
  %res = scf.execute_region -> f32 {
  ^bb0(%t: token):
    scf.if %cond {
      scf.break %t, %val : f32
    }
    scf.yield %val : f32
  }
  return %res : f32
}
