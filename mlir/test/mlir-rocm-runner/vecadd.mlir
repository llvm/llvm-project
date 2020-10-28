// RUN: mlir-rocm-runner %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @vecadd(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %block_dim = dim %arg0, %c0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %block_dim, %block_y = %c1, %block_z = %c1) {
    %a = load %arg0[%tx] : memref<?xf32>
    %b = load %arg1[%tx] : memref<?xf32>
    %c = addf %a, %b : f32
    store %c, %arg2[%tx] : memref<?xf32>
    gpu.terminator
  }
  return
}

// CHECK: [2.46, 2.46, 2.46, 2.46, 2.46]
func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c5 = constant 5 : index
  %cf1dot23 = constant 1.23 : f32
  %0 = alloc() : memref<5xf32>
  %1 = alloc() : memref<5xf32>
  %2 = alloc() : memref<5xf32>
  %3 = memref_cast %0 : memref<5xf32> to memref<?xf32>
  %4 = memref_cast %1 : memref<5xf32> to memref<?xf32>
  %5 = memref_cast %2 : memref<5xf32> to memref<?xf32>
  scf.for %i = %c0 to %c5 step %c1 {
    store %cf1dot23, %3[%i] : memref<?xf32>
    store %cf1dot23, %4[%i] : memref<?xf32>
  }
  %6 = memref_cast %3 : memref<?xf32> to memref<*xf32>
  %7 = memref_cast %4 : memref<?xf32> to memref<*xf32>
  %8 = memref_cast %5 : memref<?xf32> to memref<*xf32>
  gpu.host_register %6 : memref<*xf32>
  gpu.host_register %7 : memref<*xf32>
  gpu.host_register %8 : memref<*xf32>
  %9 = call @mgpuMemGetDeviceMemRef1dFloat(%3) : (memref<?xf32>) -> (memref<?xf32>)
  %10 = call @mgpuMemGetDeviceMemRef1dFloat(%4) : (memref<?xf32>) -> (memref<?xf32>)
  %11 = call @mgpuMemGetDeviceMemRef1dFloat(%5) : (memref<?xf32>) -> (memref<?xf32>)

  call @vecadd(%9, %10, %11) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  call @print_memref_f32(%8) : (memref<*xf32>) -> ()
  return
}

func @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func @print_memref_f32(%ptr : memref<*xf32>)
