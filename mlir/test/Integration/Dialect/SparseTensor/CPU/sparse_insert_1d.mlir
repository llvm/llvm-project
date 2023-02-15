// DEFINE: %{option} = enable-runtime-library=false
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

// Insertion example using pure codegen (no sparse runtime support lib).

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>

#trait_mul_s = {
  indexing_maps = [
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = x(i) * 2.0"
}

module {

  // Dumps pointers, indices, values for verification.
  func.func @dump(%argx: tensor<1024xf32, #SparseVector>) {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %p = sparse_tensor.pointers %argx { dimension = 0 : index }
       : tensor<1024xf32, #SparseVector> to memref<?xindex>
    %i = sparse_tensor.indices %argx { dimension = 0 : index }
       : tensor<1024xf32, #SparseVector> to memref<?xindex>
    %v = sparse_tensor.values %argx
       : tensor<1024xf32, #SparseVector> to memref<?xf32>
    %vp = vector.transfer_read %p[%c0], %c0: memref<?xindex>, vector<2xindex>
    %vi = vector.transfer_read %i[%c0], %c0: memref<?xindex>, vector<8xindex>
    %vv = vector.transfer_read %v[%c0], %f0: memref<?xf32>,   vector<8xf32>
    vector.print %vp : vector<2xindex>
    vector.print %vi : vector<8xindex>
    vector.print %vv : vector<8xf32>
    return
  }

  func.func @entry() {
    %f1    = arith.constant 1.0 : f32
    %f2    = arith.constant 2.0 : f32
    %f3    = arith.constant 3.0 : f32
    %f4    = arith.constant 4.0 : f32
    %c0    = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %c3    = arith.constant 3 : index
    %c8    = arith.constant 8 : index
    %c1023 = arith.constant 1023 : index

    // Build the sparse vector from straightline code.
    %0 = bufferization.alloc_tensor() : tensor<1024xf32, #SparseVector>
    %1 = sparse_tensor.insert %f1 into %0[%c0] : tensor<1024xf32, #SparseVector>
    %2 = sparse_tensor.insert %f2 into %1[%c1] : tensor<1024xf32, #SparseVector>
    %3 = sparse_tensor.insert %f3 into %2[%c3] : tensor<1024xf32, #SparseVector>
    %4 = sparse_tensor.insert %f4 into %3[%c1023] : tensor<1024xf32, #SparseVector>
    %5 = sparse_tensor.load %4 hasInserts : tensor<1024xf32, #SparseVector>

    // CHECK:      ( 0, 4 )
    // CHECK-NEXT: ( 0, 1, 3, 1023
    // CHECK-NEXT: ( 1, 2, 3, 4
    call @dump(%5) : (tensor<1024xf32, #SparseVector>) -> ()

    // Build another sparse vector in a loop.
    %6 = bufferization.alloc_tensor() : tensor<1024xf32, #SparseVector>
    %7 = scf.for %i = %c0 to %c8 step %c1 iter_args(%vin = %6) -> tensor<1024xf32, #SparseVector> {
      %ii = arith.muli %i, %c3 : index
      %vout = sparse_tensor.insert %f1 into %vin[%ii] : tensor<1024xf32, #SparseVector>
      scf.yield %vout : tensor<1024xf32, #SparseVector>
    }
    %8 = sparse_tensor.load %7 hasInserts : tensor<1024xf32, #SparseVector>

    // CHECK-NEXT: ( 0, 8 )
    // CHECK-NEXT: ( 0, 3, 6, 9, 12, 15, 18, 21 )
    // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1 )
    //
    call @dump(%8) : (tensor<1024xf32, #SparseVector>) -> ()

    // CHECK-NEXT: 4
    // CHECK-NEXT: 8
    %noe1 = sparse_tensor.number_of_entries %5 : tensor<1024xf32, #SparseVector>
    %noe2 = sparse_tensor.number_of_entries %8 : tensor<1024xf32, #SparseVector>
    vector.print %noe1 : index
    vector.print %noe2 : index

    // Free resources.
    bufferization.dealloc_tensor %5 : tensor<1024xf32, #SparseVector>
    bufferization.dealloc_tensor %8 : tensor<1024xf32, #SparseVector>
    return
  }
}
