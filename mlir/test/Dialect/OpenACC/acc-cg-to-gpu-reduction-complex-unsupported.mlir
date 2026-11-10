// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" -verify-diagnostics

// A complex accumulator is reduced one component at a time, which only gives
// the same answer when each component depends on itself alone. Multiplication
// mixes the two, so it is rejected rather than reduced componentwise.

func.func @complex_mul_thread_reduction(%arg_slot: memref<complex<f32>>) {
  %c1_pw = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
  // expected-error @below {{failed to legalize operation 'acc.compute_region'}}
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%a_slot = %arg_slot) : (memref<complex<f32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %one = arith.constant 1.0 : f32
    %init = complex.create %one, %one : complex<f32>
    %contrib = complex.create %one, %one : complex<f32>
    scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
      %inner_red = scf.parallel (%i) = (%tx_iv) to (%c8) step (%ktx) init (%init) -> complex<f32> {
        scf.reduce(%contrib : complex<f32>) {
        ^bb0(%lhs: complex<f32>, %rhs: complex<f32>):
          %prod = complex.mul %lhs, %rhs : complex<f32>
          scf.reduce.return %prod : complex<f32>
        }
      } {acc.par_dims = #acc<par_dims[sequential]>}
      // expected-error @below {{not yet implemented: reduction: mulf of complex<f32> across threads}}
      acc.reduction_accumulate %inner_red to %a_slot <mul> par_dims(#acc<par_dims[thread_x]>) : complex<f32> -> memref<complex<f32>>
      scf.reduce
    } {acc.par_dims = #acc<par_dims[thread_x]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}
