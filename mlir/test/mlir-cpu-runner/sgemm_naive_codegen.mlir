// RUN: mlir-opt -convert-linalg-to-loops -lower-affine -convert-loop-to-std -convert-std-to-llvm %s | mlir-cpu-runner -O3 -e main -entry-point-result=void -shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext | FileCheck %s

func @main() {
  %A = alloc() : memref<64x64xf32>
  %B = alloc() : memref<64x64xf32>
  %C = alloc() : memref<64x64xf32>

  %cf1 = constant 1.00000e+00 : f32

  linalg.fill(%A, %cf1) : memref<64x64xf32>, f32
  linalg.fill(%B, %cf1) : memref<64x64xf32>, f32

  %reps = constant 1 : index

  %t_start = call @rtclock() : () -> f64
  affine.for %arg0 = 0 to 5 {
    linalg.fill(%C, %cf1) : memref<64x64xf32>, f32
    call @sgemm_naive(%A, %B, %C) : (memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> f64
  %t = subf %t_end, %t_start : f64

  %pC = memref_cast %C : memref<64x64xf32> to memref<*xf32>
  call @print_memref_f32(%pC) : (memref<*xf32>) -> ()

  %M = dim %C, 0 : memref<64x64xf32>
  %N = dim %C, 1 : memref<64x64xf32>
  %K = dim %A, 1 : memref<64x64xf32>

  %f1 = muli %M, %N : index
  %f2 = muli %f1, %K : index

  // 2*M*N*K.
  %c2 = constant 2 : index
  %f3 = muli %c2, %f2 : index
  %num_flops = muli %reps, %f3 : index
  %num_flops_i = index_cast %num_flops : index to i64
  %num_flops_f = sitofp %num_flops_i : i64 to f64
  %flops = divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()

  return
}
// CHECK: 65,   65,   65,

func @sgemm_naive(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  %c0 = constant 0 : index
  affine.for %arg3 = 0 to 64 {
    affine.for %arg4 = 0 to 64 {
      %m = alloc() : memref<1xf32>
      %v = affine.load %arg2[%arg3, %arg4] : memref<64x64xf32>
      affine.store %v, %m[%c0] : memref<1xf32>
      affine.for %arg5 = 0 to 64 {
        %3 = affine.load %arg0[%arg3, %arg5] : memref<64x64xf32>
        %4 = affine.load %arg1[%arg5, %arg4] : memref<64x64xf32>
        %5 = affine.load %m[0] : memref<1xf32>
        %6 = mulf %3, %4 : f32
        %7 = addf %6, %5 : f32
        affine.store %7, %m[0] : memref<1xf32>
      }
      %s = affine.load %m[%c0] : memref<1xf32>
      affine.store %s, %arg2[%arg3, %arg4] : memref<64x64xf32>
      dealloc %m : memref<1xf32>
    }
  }
  return
}

func @print_flops(f64)
func @rtclock() -> f64
func @print_memref_f32(memref<*xf32>)
