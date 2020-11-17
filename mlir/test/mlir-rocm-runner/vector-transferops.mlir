// RUN: mlir-rocm-runner %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @vectransferx2(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) {
  %cst = constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst, %block_z = %cst) {
    %f0 = constant 0.0: f32
    %base = constant 0 : index
    %f = vector.transfer_read %arg0[%base], %f0
        {permutation_map = affine_map<(d0) -> (d0)>} :
      memref<?xf32>, vector<2xf32>

    %c = addf %f, %f : vector<2xf32>

    %base1 = constant 1 : index
    vector.transfer_write %c, %arg1[%base1]
        {permutation_map = affine_map<(d0) -> (d0)>} :
      vector<2xf32>, memref<?xf32>

    gpu.terminator
  }
  return
}

func @vectransferx4(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) {
  %cst = constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst, %block_y = %cst, %block_z = %cst) {
    %f0 = constant 0.0: f32
    %base = constant 0 : index
    %f = vector.transfer_read %arg0[%base], %f0
        {permutation_map = affine_map<(d0) -> (d0)>} :
      memref<?xf32>, vector<4xf32>

    %c = addf %f, %f : vector<4xf32>

    vector.transfer_write %c, %arg1[%base]
        {permutation_map = affine_map<(d0) -> (d0)>} :
      vector<4xf32>, memref<?xf32>

    gpu.terminator
  }
  return
}

func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  %cf1 = constant 1.0 : f32
  %cf1dot23 = constant 1.23 : f32

  %arg0 = alloc() : memref<4xf32>
  %arg1 = alloc() : memref<4xf32>

  %22 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>
  %23 = memref_cast %arg1 : memref<4xf32> to memref<?xf32>

  scf.for %i = %c0 to %c4 step %c1 {
    store %cf1dot23, %22[%i] : memref<?xf32>
    store %cf1dot23, %23[%i] : memref<?xf32>
  }

  %cast0 = memref_cast %22 : memref<?xf32> to memref<*xf32>
  %cast1 = memref_cast %23 : memref<?xf32> to memref<*xf32>

  gpu.host_register %cast0 : memref<*xf32>
  gpu.host_register %cast1 : memref<*xf32>

  %24 = call @mgpuMemGetDeviceMemRef1dFloat(%22) : (memref<?xf32>) -> (memref<?xf32>)
  %26 = call @mgpuMemGetDeviceMemRef1dFloat(%23) : (memref<?xf32>) -> (memref<?xf32>)

  // CHECK: [1.23, 2.46, 2.46, 1.23]
  call @vectransferx2(%24, %26) : (memref<?xf32>,  memref<?xf32>) -> ()
  call @print_memref_f32(%cast1) : (memref<*xf32>) -> ()

  // CHECK: [2.46, 2.46, 2.46, 2.46]
  call @vectransferx4(%24, %26) : (memref<?xf32>,  memref<?xf32>) -> ()
  call @print_memref_f32(%cast1) : (memref<*xf32>) -> ()
  return
}

func private @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func private @print_memref_f32(%ptr : memref<*xf32>)
