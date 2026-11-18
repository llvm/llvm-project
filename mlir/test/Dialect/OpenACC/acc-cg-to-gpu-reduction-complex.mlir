// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A cross-thread reduction takes integers and floats, so a complex accumulator
// is reduced one component at a time and rebuilt from the two results. Addition
// is componentwise, so reducing the parts separately gives the same answer.

// CHECK-LABEL: func.func @complex_thread_reduction
// CHECK:       gpu.launch
// CHECK:       %[[RE:.*]] = complex.re %[[VAL:.*]] : complex<f32>
// CHECK:       %[[RERED:.*]] = gpu.all_reduce add %[[RE]]
// CHECK:       %[[IM:.*]] = complex.im %[[VAL]] : complex<f32>
// CHECK:       %[[IMRED:.*]] = gpu.all_reduce add %[[IM]]
// CHECK:       %[[RES:.*]] = complex.create %[[RERED]], %[[IMRED]] : complex<f32>
// CHECK:       memref.store %[[RES]]

func.func @complex_thread_reduction(%arg_slot: memref<complex<f32>>) {
  %c1_pw = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%a_slot = %arg_slot) : (memref<complex<f32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %init = complex.create %zero, %zero : complex<f32>
    %contrib = complex.create %one, %one : complex<f32>
    scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
      %inner_red = scf.parallel (%i) = (%tx_iv) to (%c8) step (%ktx) init (%init) -> complex<f32> {
        scf.reduce(%contrib : complex<f32>) {
        ^bb0(%lhs: complex<f32>, %rhs: complex<f32>):
          %sum = complex.add %lhs, %rhs : complex<f32>
          scf.reduce.return %sum : complex<f32>
        }
      } {acc.par_dims = #acc<par_dims[sequential]>}
      acc.reduction_accumulate %inner_red to %a_slot <add> par_dims(#acc<par_dims[thread_x]>) : complex<f32> -> memref<complex<f32>>
      scf.reduce
    } {acc.par_dims = #acc<par_dims[thread_x]>}
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// An array accumulator reduces element by element, and each element goes
// through the same componentwise path, so a complex array needs no separate
// handling.

// CHECK-LABEL: func.func @complex_array_reduction
// CHECK:       gpu.launch
// CHECK-NOT:   acc.reduction_accumulate_array
// CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         %[[ELT:.*]] = memref.load %[[ALLOCA:.*]][%[[IV]]] : memref<2xcomplex<f32>>
// CHECK:         %[[RE:.*]] = complex.re %[[ELT]] : complex<f32>
// CHECK:         %[[RERED:.*]] = gpu.all_reduce add %[[RE]]
// CHECK:         %[[IM:.*]] = complex.im %[[ELT]] : complex<f32>
// CHECK:         %[[IMRED:.*]] = gpu.all_reduce add %[[IM]]
// CHECK:         %[[RES:.*]] = complex.create %[[RERED]], %[[IMRED]] : complex<f32>
// CHECK:         memref.store %[[RES]], %[[ALLOCA]][%[[IV]]] : memref<2xcomplex<f32>>
// CHECK:       }

func.func @complex_array_reduction(%arg0: memref<2xcomplex<f32>>) {
  %0 = acc.copyin varPtr(%arg0 : memref<2xcomplex<f32>>) dataClause(acc_reduction) implicit(true) name("r") -> memref<2xcomplex<f32>>
  acc.kernel_environment dataOperands(%0 : memref<2xcomplex<f32>>) {
    %c1_pw = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
    acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg2 = %0) : (memref<2xcomplex<f32>>) {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %zero = arith.constant 0.0 : f32
      %one = arith.constant 1.0 : f32
      %zc = complex.create %zero, %zero : complex<f32>
      %oc = complex.create %one, %one : complex<f32>
      %2 = acc.reduction_init %arg2 <add> : memref<2xcomplex<f32>> {
        %alloca = memref.alloca() : memref<2xcomplex<f32>>
        scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
          memref.store %zc, %alloca[%i] : memref<2xcomplex<f32>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloca : memref<2xcomplex<f32>>
      }
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
          %3 = memref.load %2[%c0] : memref<2xcomplex<f32>>
          %4 = complex.add %3, %oc : complex<f32>
          memref.store %4, %2[%c0] : memref<2xcomplex<f32>>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %b = acc.bounds extent(%c2 : index)
      acc.reduction_accumulate_array %2 bounds(%b) <add> par_dims(#acc<par_dims[block_x, thread_x]>) : memref<2xcomplex<f32>>
      acc.reduction_combine_region %2 into %arg2 : memref<2xcomplex<f32>> {
        scf.for %i = %c0 to %c2 step %c1 {
          %3 = memref.load %2[%i] : memref<2xcomplex<f32>>
          %4 = memref.load %arg2[%i] : memref<2xcomplex<f32>>
          %5 = complex.add %3, %4 : complex<f32>
          memref.store %5, %arg2[%i] : memref<2xcomplex<f32>>
        }
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<2xcomplex<f32>>) to varPtr(%arg0 : memref<2xcomplex<f32>>) dataClause(acc_reduction) implicit(true) name("r")
  return
}
