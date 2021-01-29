// RUN: mlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm %s | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | FileCheck %s

func @main() {
  %A = alloc() : memref<16x16xf32>
  %B = alloc() : memref<16x16xf32>
  %C = alloc() : memref<16x16xf32>

  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<16x16xf32>, f32
  linalg.fill(%B, %cf1) : memref<16x16xf32>, f32

  %reps = constant 1 : index

  %t_start = call @rtclock() : () -> f64
  affine.for %arg0 = 0 to 5 {
    linalg.fill(%C, %cf1) : memref<16x16xf32>, f32
    call @sgemm_naive(%A, %B, %C) : (memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64

  %res = affine.load %C[0, 0]: memref<16x16xf32>
  vector.print %res: f32

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index

  %M = dim %C, %c0 : memref<16x16xf32>
  %N = dim %C, %c1 : memref<16x16xf32>
  %K = dim %A, %c1 : memref<16x16xf32>

  %f1 = muli %M, %N : index
  %f2 = muli %f1, %K : index

  // 2*M*N*K.
  %f3 = muli %c2, %f2 : index
  %num_flops = muli %reps, %f3 : index
  %num_flops_i = index_cast %num_flops : index to i16
  %num_flops_f = sitofp %num_flops_i : i16 to f64
  %flops = divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()

  return
}
// CHECK: 17

func @sgemm_naive(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) {
  %c0 = constant 0 : index
  affine.for %arg3 = 0 to 16 {
    affine.for %arg4 = 0 to 16 {
      %m = alloc() : memref<1xf32>
      %v = affine.load %arg2[%arg3, %arg4] : memref<16x16xf32>
      affine.store %v, %m[%c0] : memref<1xf32>
      affine.for %arg5 = 0 to 16 {
        %3 = affine.load %arg0[%arg3, %arg5] : memref<16x16xf32>
        %4 = affine.load %arg1[%arg5, %arg4] : memref<16x16xf32>
        %5 = affine.load %m[0] : memref<1xf32>
        %6 = mulf %3, %4 : f32
        %7 = addf %6, %5 : f32
        affine.store %7, %m[0] : memref<1xf32>
      }
      %s = affine.load %m[%c0] : memref<1xf32>
      affine.store %s, %arg2[%arg3, %arg4] : memref<16x16xf32>
      dealloc %m : memref<1xf32>
    }
  }
  return
}

func private @print_flops(f64)
func private @rtclock() -> f64
