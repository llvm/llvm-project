// RUN: mlir-opt -pass-pipeline="func.func(convert-linalg-to-loops,lower-affine,convert-scf-to-cf,convert-arith-to-llvm),convert-vector-to-llvm,convert-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts" %s | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

func.func @main() {
  %A = memref.alloc() : memref<16x16xf32>
  %B = memref.alloc() : memref<16x16xf32>
  %C = memref.alloc() : memref<16x16xf32>

  %cf1 = arith.constant 1.00000e+00 : f32

  linalg.fill ins(%cf1 : f32) outs(%A : memref<16x16xf32>)
  linalg.fill ins(%cf1 : f32) outs(%B : memref<16x16xf32>)

  %reps = arith.constant 1 : index

  %t_start = call @rtclock() : () -> f64
  affine.for %arg0 = 0 to 5 {
    linalg.fill ins(%cf1 : f32) outs(%C : memref<16x16xf32>)
    func.call @sgemm_naive(%A, %B, %C) : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64
  %t = arith.subf %t_end, %t_start : f64

  %res = affine.load %C[0, 0]: memref<16x16xf32>
  vector.print %res: f32

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %M = memref.dim %C, %c0 : memref<16x16xf32>
  %N = memref.dim %C, %c1 : memref<16x16xf32>
  %K = memref.dim %A, %c1 : memref<16x16xf32>

  %f1 = arith.muli %M, %N : index
  %f2 = arith.muli %f1, %K : index

  // 2*M*N*K.
  %f3 = arith.muli %c2, %f2 : index
  %num_flops = arith.muli %reps, %f3 : index
  %num_flops_i = arith.index_cast %num_flops : index to i16
  %num_flops_f = arith.sitofp %num_flops_i : i16 to f64
  %flops = arith.divf %num_flops_f, %t : f64
  call @printFlops(%flops) : (f64) -> ()

  memref.dealloc %A : memref<16x16xf32>
  memref.dealloc %B : memref<16x16xf32>
  memref.dealloc %C : memref<16x16xf32>
  return
}
// CHECK: 17

func.func @sgemm_naive(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %arg3 = 0 to 16 {
    affine.for %arg4 = 0 to 16 {
      %m = memref.alloc() : memref<1xf32>
      %v = affine.load %arg2[%arg3, %arg4] : memref<16x16xf32>
      affine.store %v, %m[%c0] : memref<1xf32>
      affine.for %arg5 = 0 to 16 {
        %3 = affine.load %arg0[%arg3, %arg5] : memref<16x16xf32>
        %4 = affine.load %arg1[%arg5, %arg4] : memref<16x16xf32>
        %5 = affine.load %m[0] : memref<1xf32>
        %6 = arith.mulf %3, %4 : f32
        %7 = arith.addf %6, %5 : f32
        affine.store %7, %m[0] : memref<1xf32>
      }
      %s = affine.load %m[%c0] : memref<1xf32>
      affine.store %s, %arg2[%arg3, %arg4] : memref<16x16xf32>
      memref.dealloc %m : memref<1xf32>
    }
  }
  return
}

func.func private @printFlops(f64)
func.func private @rtclock() -> f64
