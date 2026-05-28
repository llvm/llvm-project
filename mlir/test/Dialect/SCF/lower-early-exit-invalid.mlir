// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(scf-lower-early-exit))' -split-input-file -verify-diagnostics

func.func @reject_cf_br(%v: f32) -> f32 {
  // expected-error @below {{'scf.execute_region' op contains unsupported terminator 'cf.br'; expected 'scf.yield' or 'scf.break'}}
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    cf.br ^bb1
  ^bb1:
    scf.yield %v : f32
  }
  return %0 : f32
}

// -----

func.func @reject_cf_cond_br(%c: i1, %v: f32) -> f32 {
  // expected-error @below {{'scf.execute_region' op contains unsupported terminator 'cf.cond_br'; expected 'scf.yield' or 'scf.break'}}
  %0 = scf.execute_region -> f32 {
  ^bb0(%tok: token):
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    scf.yield %v : f32
  ^bb2:
    scf.yield %v : f32
  }
  return %0 : f32
}
