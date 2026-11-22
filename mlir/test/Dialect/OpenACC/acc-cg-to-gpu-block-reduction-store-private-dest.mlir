// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A decomposed complex reduction accumulates the real and imaginary parts per
// block, rebuilds the complex partial and stores it into a block-private
// accumulator, which is then combined into the result across blocks. The store
// into the block-private accumulator has a single writer per block, so it must
// stay a plain store: it must not be initialized from block 0 only nor become
// an atomic update. Only the combine into the result is atomic.

// CHECK-LABEL: func.func @complex_partial_store_private_dest
// CHECK:         gpu.launch
// CHECK:         %[[PART:.*]] = memref.memory_space_cast %{{.*}} : memref<complex<f32>, #gpu.address_space<workgroup>> to memref<complex<f32>>
// CHECK-NOT:     gpu.block_id
// CHECK:         gpu.all_reduce add
// CHECK-NOT:     gpu.block_id
// CHECK:         gpu.all_reduce add
// CHECK-NOT:     acc.atomic.update
// CHECK:         %[[CPLX:.*]] = complex.create
// CHECK-NOT:     acc.atomic.update
// CHECK:         memref.store %[[CPLX]], %[[PART]][] : memref<complex<f32>>
// CHECK-NOT:     acc.atomic.update %[[PART]]
// CHECK:         acc.atomic.update %arg{{[0-9]+}} : memref<complex<f32>>
// CHECK-NOT:     acc.atomic.update

func.func @complex_partial_store_private_dest(%a: memref<10xcomplex<f32>>, %res: memref<complex<f32>>) {
  %c10 = arith.constant 10 : index
  %bx = acc.par_width %c10 par_dim(#acc.par_dim<block_x>)
  %pv_part = acc.privatize par_dims(#acc<par_dims[block_x]>) {acc.active_par_dims = #acc<active_par_dims[block_x]>} : () -> !acc.private_type<memref<complex<f32>>>
  %pv_re = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>} : () -> !acc.private_type<memref<f32>>
  %pv_im = acc.privatize {acc.active_par_dims = #acc<active_par_dims[block_x]>} : () -> !acc.private_type<memref<f32>>
  acc.compute_region launch(%kbx = %bx) ins(%a_in = %a, %r_in = %res, %pp = %pv_part, %pre = %pv_re, %pim = %pv_im) : (memref<10xcomplex<f32>>, memref<complex<f32>>, !acc.private_type<memref<complex<f32>>>, !acc.private_type<memref<f32>>, !acc.private_type<memref<f32>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10_0 = arith.constant 10 : index
    %zero = complex.create %cst, %cst : complex<f32>
    %re = acc.private_local %pre {acc.active_par_dims = #acc<active_par_dims[block_x]>} : (!acc.private_type<memref<f32>>) -> memref<f32>
    %im = acc.private_local %pim {acc.active_par_dims = #acc<active_par_dims[block_x]>} : (!acc.private_type<memref<f32>>) -> memref<f32>
    %part = acc.private_local %pp {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x]>} : (!acc.private_type<memref<complex<f32>>>) -> memref<complex<f32>>
    scf.parallel (%b) = (%c0) to (%kbx) step (%c1) {
      %sum = scf.parallel (%i) = (%b) to (%c10_0) step (%kbx) init (%zero) -> complex<f32> {
        %e = memref.load %a_in[%i] : memref<10xcomplex<f32>>
        scf.reduce(%e : complex<f32>) {
        ^bb0(%lhs: complex<f32>, %rhs: complex<f32>):
          %s = complex.add %lhs, %rhs : complex<f32>
          scf.reduce.return %s : complex<f32>
        }
      } {acc.par_dims = #acc<par_dims[sequential]>}
      %sr = complex.re %sum : complex<f32>
      %si = complex.im %sum : complex<f32>
      acc.reduction_accumulate %sr to %re <add> par_dims(#acc<par_dims[block_x]>) : f32 -> memref<f32>
      acc.reduction_accumulate %si to %im <add> par_dims(#acc<par_dims[block_x]>) : f32 -> memref<f32>
      %lr = memref.load %re[] : memref<f32>
      %li = memref.load %im[] : memref<f32>
      %c = complex.create %lr, %li : complex<f32>
      acc.predicate_region {
        memref.store %c, %part[] : memref<complex<f32>>
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.predicate_region {
      acc.reduction_combine_region %part into %r_in : memref<complex<f32>> {
        %ra = memref.load %r_in[] : memref<complex<f32>>
        %pa = memref.load %part[] : memref<complex<f32>>
        %s = complex.add %ra, %pa : complex<f32>
        memref.store %s, %r_in[] : memref<complex<f32>>
        acc.yield
      } {acc.par_dims = #acc<par_dims[block_x]>}
    }
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}
