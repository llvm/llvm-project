// RUN: mlir-opt %s \
// RUN:   --sparsification --sparse-tensor-conversion \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize  \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#Tensor1  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>
}>

#Tensor2  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (j,k,i)>
}>

#Tensor3  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (k,i,j)>
}>

//
// Integration test that tests conversions between sparse tensors.
//
module {
  func private @exit(index) -> ()

  //
  // Verify utilities.
  //
  func @checkf64(%arg0: memref<?xf64>, %arg1: memref<?xf64>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    // Same lengths?
    %0 = memref.dim %arg0, %c0 : memref<?xf64>
    %1 = memref.dim %arg1, %c0 : memref<?xf64>
    %2 = cmpi ne, %0, %1 : index
    scf.if %2 {
      call @exit(%c1) : (index) -> ()
    }
    // Same content?
    scf.for %i = %c0 to %0 step %c1 {
      %a = memref.load %arg0[%i] : memref<?xf64>
      %b = memref.load %arg1[%i] : memref<?xf64>
      %c = cmpf une, %a, %b : f64
      scf.if %c {
        call @exit(%c1) : (index) -> ()
      }
    }
    return
  }
  func @check(%arg0: memref<?xindex>, %arg1: memref<?xindex>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    // Same lengths?
    %0 = memref.dim %arg0, %c0 : memref<?xindex>
    %1 = memref.dim %arg1, %c0 : memref<?xindex>
    %2 = cmpi ne, %0, %1 : index
    scf.if %2 {
      call @exit(%c1) : (index) -> ()
    }
    // Same content?
    scf.for %i = %c0 to %0 step %c1 {
      %a = memref.load %arg0[%i] : memref<?xindex>
      %b = memref.load %arg1[%i] : memref<?xindex>
      %c = cmpi ne, %a, %b : index
      scf.if %c {
        call @exit(%c1) : (index) -> ()
      }
    }
    return
  }

  //
  // Output utility.
  //
  func @dumpf64(%arg0: memref<?xf64>) {
    %c0 = constant 0 : index
    %d0 = constant 0.0 : f64
    %0 = vector.transfer_read %arg0[%c0], %d0: memref<?xf64>, vector<24xf64>
    vector.print %0 : vector<24xf64>
    return
  }

  //
  // Main driver.
  //
  func @entry() {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    //
    // Initialize a 3-dim dense tensor.
    //
    %t = constant dense<[
       [  [  1.0,  2.0,  3.0,  4.0 ],
          [  5.0,  6.0,  7.0,  8.0 ],
          [  9.0, 10.0, 11.0, 12.0 ] ],
       [  [ 13.0, 14.0, 15.0, 16.0 ],
          [ 17.0, 18.0, 19.0, 20.0 ],
          [ 21.0, 22.0, 23.0, 24.0 ] ]
    ]> : tensor<2x3x4xf64>

    //
    // Convert dense tensor directly to various sparse tensors.
    //    tensor1: stored as 2x3x4
    //    tensor2: stored as 3x4x2
    //    tensor3: stored as 4x2x3
    //
    %1 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor1>
    %2 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor2>
    %3 = sparse_tensor.convert %t : tensor<2x3x4xf64> to tensor<2x3x4xf64, #Tensor3>

    //
    // Convert sparse tensor to various sparse tensors. Note that the result
    // should always correspond to the direct conversion, since the sparse
    // tensor formats have the ability to restore into the original ordering.
    //
    %a = sparse_tensor.convert %1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64, #Tensor1>
    %b = sparse_tensor.convert %2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64, #Tensor1>
    %c = sparse_tensor.convert %3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64, #Tensor1>
    %d = sparse_tensor.convert %1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64, #Tensor2>
    %e = sparse_tensor.convert %2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64, #Tensor2>
    %f = sparse_tensor.convert %3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64, #Tensor2>
    %g = sparse_tensor.convert %1 : tensor<2x3x4xf64, #Tensor1> to tensor<2x3x4xf64, #Tensor3>
    %h = sparse_tensor.convert %2 : tensor<2x3x4xf64, #Tensor2> to tensor<2x3x4xf64, #Tensor3>
    %i = sparse_tensor.convert %3 : tensor<2x3x4xf64, #Tensor3> to tensor<2x3x4xf64, #Tensor3>

    //
    // Check values equality.
    //

    %v1 = sparse_tensor.values %1 : tensor<2x3x4xf64, #Tensor1> to memref<?xf64>
    %v2 = sparse_tensor.values %2 : tensor<2x3x4xf64, #Tensor2> to memref<?xf64>
    %v3 = sparse_tensor.values %3 : tensor<2x3x4xf64, #Tensor3> to memref<?xf64>

    %av = sparse_tensor.values %a : tensor<2x3x4xf64, #Tensor1> to memref<?xf64>
    %bv = sparse_tensor.values %b : tensor<2x3x4xf64, #Tensor1> to memref<?xf64>
    %cv = sparse_tensor.values %c : tensor<2x3x4xf64, #Tensor1> to memref<?xf64>
    %dv = sparse_tensor.values %d : tensor<2x3x4xf64, #Tensor2> to memref<?xf64>
    %ev = sparse_tensor.values %e : tensor<2x3x4xf64, #Tensor2> to memref<?xf64>
    %fv = sparse_tensor.values %f : tensor<2x3x4xf64, #Tensor2> to memref<?xf64>
    %gv = sparse_tensor.values %g : tensor<2x3x4xf64, #Tensor3> to memref<?xf64>
    %hv = sparse_tensor.values %h : tensor<2x3x4xf64, #Tensor3> to memref<?xf64>
    %iv = sparse_tensor.values %i : tensor<2x3x4xf64, #Tensor3> to memref<?xf64>

    call @checkf64(%v1, %av) : (memref<?xf64>, memref<?xf64>) -> ()
    call @checkf64(%v1, %bv) : (memref<?xf64>, memref<?xf64>) -> ()
    call @checkf64(%v1, %cv) : (memref<?xf64>, memref<?xf64>) -> ()
    call @checkf64(%v2, %dv) : (memref<?xf64>, memref<?xf64>) -> ()
    call @checkf64(%v2, %ev) : (memref<?xf64>, memref<?xf64>) -> ()
    call @checkf64(%v2, %fv) : (memref<?xf64>, memref<?xf64>) -> ()
    call @checkf64(%v3, %gv) : (memref<?xf64>, memref<?xf64>) -> ()
    call @checkf64(%v3, %hv) : (memref<?xf64>, memref<?xf64>) -> ()
    call @checkf64(%v3, %iv) : (memref<?xf64>, memref<?xf64>) -> ()

    //
    // Check index equality.
    //

    %v10 = sparse_tensor.indices %1, %c0 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %v11 = sparse_tensor.indices %1, %c1 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %v12 = sparse_tensor.indices %1, %c2 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %v20 = sparse_tensor.indices %2, %c0 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %v21 = sparse_tensor.indices %2, %c1 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %v22 = sparse_tensor.indices %2, %c2 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %v30 = sparse_tensor.indices %3, %c0 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %v31 = sparse_tensor.indices %3, %c1 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %v32 = sparse_tensor.indices %3, %c2 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>

    %a10 = sparse_tensor.indices %a, %c0 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %a11 = sparse_tensor.indices %a, %c1 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %a12 = sparse_tensor.indices %a, %c2 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %b10 = sparse_tensor.indices %b, %c0 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %b11 = sparse_tensor.indices %b, %c1 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %b12 = sparse_tensor.indices %b, %c2 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %c10 = sparse_tensor.indices %c, %c0 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %c11 = sparse_tensor.indices %c, %c1 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %c12 = sparse_tensor.indices %c, %c2 : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>

    %d10 = sparse_tensor.indices %d, %c0 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %d11 = sparse_tensor.indices %d, %c1 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %d12 = sparse_tensor.indices %d, %c2 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %e10 = sparse_tensor.indices %e, %c0 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %e11 = sparse_tensor.indices %e, %c1 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %e12 = sparse_tensor.indices %e, %c2 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %f10 = sparse_tensor.indices %f, %c0 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %f11 = sparse_tensor.indices %f, %c1 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %f12 = sparse_tensor.indices %f, %c2 : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>

    %g10 = sparse_tensor.indices %g, %c0 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %g11 = sparse_tensor.indices %g, %c1 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %g12 = sparse_tensor.indices %g, %c2 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %h10 = sparse_tensor.indices %h, %c0 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %h11 = sparse_tensor.indices %h, %c1 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %h12 = sparse_tensor.indices %h, %c2 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %i10 = sparse_tensor.indices %i, %c0 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %i11 = sparse_tensor.indices %i, %c1 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %i12 = sparse_tensor.indices %i, %c2 : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>

    call @check(%v10, %a10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v11, %a11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v12, %a12) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v10, %b10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v11, %b11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v12, %b12) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v10, %c10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v11, %c11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v12, %c12) : (memref<?xindex>, memref<?xindex>) -> ()

    call @check(%v20, %d10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v21, %d11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v22, %d12) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v20, %e10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v21, %e11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v22, %e12) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v20, %f10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v21, %f11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v22, %f12) : (memref<?xindex>, memref<?xindex>) -> ()

    call @check(%v30, %g10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v31, %g11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v32, %g12) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v30, %h10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v31, %h11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v32, %h12) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v30, %i10) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v31, %i11) : (memref<?xindex>, memref<?xindex>) -> ()
    call @check(%v32, %i12) : (memref<?xindex>, memref<?xindex>) -> ()

    //
    // Sanity check direct results.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 )
    // CHECK-NEXT: ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24 )
    // CHECK-NEXT: ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24 )
    //
    call @dumpf64(%v1) : (memref<?xf64>) -> ()
    call @dumpf64(%v2) : (memref<?xf64>) -> ()
    call @dumpf64(%v3) : (memref<?xf64>) -> ()

    // Release the resources.
    sparse_tensor.release %1 : tensor<2x3x4xf64, #Tensor1>
    sparse_tensor.release %2 : tensor<2x3x4xf64, #Tensor2>
    sparse_tensor.release %3 : tensor<2x3x4xf64, #Tensor3>
    sparse_tensor.release %b : tensor<2x3x4xf64, #Tensor1>
    sparse_tensor.release %c : tensor<2x3x4xf64, #Tensor1>
    sparse_tensor.release %d : tensor<2x3x4xf64, #Tensor2>
    sparse_tensor.release %f : tensor<2x3x4xf64, #Tensor2>
    sparse_tensor.release %g : tensor<2x3x4xf64, #Tensor3>
    sparse_tensor.release %h : tensor<2x3x4xf64, #Tensor3>

    return
  }
}
