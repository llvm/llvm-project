// RUN: mlir-opt %s --sparse-compiler=enable-runtime-library=false | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

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
    %cu = arith.constant 99 : index
    %fu = arith.constant 99.0 : f32
    %p = sparse_tensor.pointers %argx { dimension = 0 : index }
       : tensor<1024xf32, #SparseVector> to memref<?xindex>
    %i = sparse_tensor.indices %argx { dimension = 0 : index }
       : tensor<1024xf32, #SparseVector> to memref<?xindex>
    %v = sparse_tensor.values %argx
       : tensor<1024xf32, #SparseVector> to memref<?xf32>
    %vp = vector.transfer_read %p[%c0], %cu: memref<?xindex>, vector<8xindex>
    %vi = vector.transfer_read %i[%c0], %cu: memref<?xindex>, vector<8xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf32>,   vector<8xf32>
    vector.print %vp : vector<8xindex>
    vector.print %vi : vector<8xindex>
    vector.print %vv : vector<8xf32>
    return
  }

  func.func @entry() {
    %f1    = arith.constant 1.0 : f32
    %f2    = arith.constant 2.0 : f32
    %c0    = arith.constant 0 : index
    %c1    = arith.constant 1 : index
    %c3    = arith.constant 3 : index
    %c1023 = arith.constant 1023 : index

    // Build the sparse vector from code.
    %0 = bufferization.alloc_tensor() : tensor<1024xf32, #SparseVector>
    %1 = sparse_tensor.insert %f1 into %0[%c0] : tensor<1024xf32, #SparseVector>
    %2 = sparse_tensor.insert %f2 into %1[%c1] : tensor<1024xf32, #SparseVector>
    %3 = sparse_tensor.insert %f1 into %2[%c3] : tensor<1024xf32, #SparseVector>
    %4 = sparse_tensor.insert %f2 into %3[%c1023] : tensor<1024xf32, #SparseVector>
    %5 = sparse_tensor.load %4 hasInserts : tensor<1024xf32, #SparseVector>

    // CHECK:      ( 0, 4, 99, 99, 99, 99, 99, 99 )
    // CHECK-NEXT: ( 0, 1, 3, 1023, 99, 99, 99, 99 )
    // CHECK-NEXT: ( 1, 2, 1, 2, 99, 99, 99, 99 )
    call @dump(%5) : (tensor<1024xf32, #SparseVector>) -> ()

    bufferization.dealloc_tensor %5 : tensor<1024xf32, #SparseVector>
    return
  }
}
