// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true"
// RUN: %{command}

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
  //
  // Output utilities.
  //
  func.func @dumpf64(%arg0: memref<?xf64>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c0], %d0: memref<?xf64>, vector<24xf64>
    vector.print %0 : vector<24xf64>
    return
  }
  func.func @dumpidx(%arg0: memref<?xindex>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %d0: memref<?xindex>, vector<25xindex>
    vector.print %0 : vector<25xindex>
    return
  }

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    //
    // Initialize a 3-dim dense tensor.
    //
    %t = arith.constant dense<[
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
    // Check number_of_entries.
    //
    // CHECK-COUNT-12: 24
    %nv1 = sparse_tensor.number_of_entries %1 : tensor<2x3x4xf64, #Tensor1>
    %nv2 = sparse_tensor.number_of_entries %2 : tensor<2x3x4xf64, #Tensor2>
    %nv3 = sparse_tensor.number_of_entries %3 : tensor<2x3x4xf64, #Tensor3>
    %nav = sparse_tensor.number_of_entries %a : tensor<2x3x4xf64, #Tensor1>
    %nbv = sparse_tensor.number_of_entries %b : tensor<2x3x4xf64, #Tensor1>
    %ncv = sparse_tensor.number_of_entries %c : tensor<2x3x4xf64, #Tensor1>
    %ndv = sparse_tensor.number_of_entries %d : tensor<2x3x4xf64, #Tensor2>
    %nev = sparse_tensor.number_of_entries %e : tensor<2x3x4xf64, #Tensor2>
    %nfv = sparse_tensor.number_of_entries %f : tensor<2x3x4xf64, #Tensor2>
    %ngv = sparse_tensor.number_of_entries %g : tensor<2x3x4xf64, #Tensor3>
    %nhv = sparse_tensor.number_of_entries %h : tensor<2x3x4xf64, #Tensor3>
    %niv = sparse_tensor.number_of_entries %i : tensor<2x3x4xf64, #Tensor3>
    vector.print %nv1 : index
    vector.print %nv2 : index
    vector.print %nv3 : index
    vector.print %nav : index
    vector.print %nbv : index
    vector.print %ncv : index
    vector.print %ndv : index
    vector.print %nev : index
    vector.print %nfv : index
    vector.print %ngv : index
    vector.print %nhv : index
    vector.print %niv : index

    //
    // Check values.
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 )
    // CHECK-NEXT: ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24 )
    // CHECK-NEXT: ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 )
    // CHECK-NEXT: ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24 )
    // CHECK-NEXT: ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24 )
    // CHECK-NEXT: ( 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24 )
    // CHECK-NEXT: ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24 )
    // CHECK-NEXT: ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24 )
    // CHECK-NEXT: ( 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24 )
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

    call @dumpf64(%v1) : (memref<?xf64>) -> ()
    call @dumpf64(%v2) : (memref<?xf64>) -> ()
    call @dumpf64(%v3) : (memref<?xf64>) -> ()
    call @dumpf64(%av) : (memref<?xf64>) -> ()
    call @dumpf64(%bv) : (memref<?xf64>) -> ()
    call @dumpf64(%cv) : (memref<?xf64>) -> ()
    call @dumpf64(%dv) : (memref<?xf64>) -> ()
    call @dumpf64(%ev) : (memref<?xf64>) -> ()
    call @dumpf64(%fv) : (memref<?xf64>) -> ()
    call @dumpf64(%gv) : (memref<?xf64>) -> ()
    call @dumpf64(%hv) : (memref<?xf64>) -> ()
    call @dumpf64(%iv) : (memref<?xf64>) -> ()

    //
    // Check indices.
    //
    // CHECK-NEXT: ( 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 )
    //
    %v10 = sparse_tensor.indices %1 { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %v11 = sparse_tensor.indices %1 { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %v12 = sparse_tensor.indices %1 { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %v20 = sparse_tensor.indices %2 { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %v21 = sparse_tensor.indices %2 { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %v22 = sparse_tensor.indices %2 { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %v30 = sparse_tensor.indices %3 { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %v31 = sparse_tensor.indices %3 { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %v32 = sparse_tensor.indices %3 { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>

    %a10 = sparse_tensor.indices %a { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %a11 = sparse_tensor.indices %a { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %a12 = sparse_tensor.indices %a { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %b10 = sparse_tensor.indices %b { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %b11 = sparse_tensor.indices %b { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %b12 = sparse_tensor.indices %b { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %c10 = sparse_tensor.indices %c { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %c11 = sparse_tensor.indices %c { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>
    %c12 = sparse_tensor.indices %c { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor1> to memref<?xindex>

    %d20 = sparse_tensor.indices %d { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %d21 = sparse_tensor.indices %d { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %d22 = sparse_tensor.indices %d { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %e20 = sparse_tensor.indices %e { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %e21 = sparse_tensor.indices %e { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %e22 = sparse_tensor.indices %e { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %f20 = sparse_tensor.indices %f { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %f21 = sparse_tensor.indices %f { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>
    %f22 = sparse_tensor.indices %f { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor2> to memref<?xindex>

    %g30 = sparse_tensor.indices %g { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %g31 = sparse_tensor.indices %g { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %g32 = sparse_tensor.indices %g { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %h30 = sparse_tensor.indices %h { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %h31 = sparse_tensor.indices %h { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %h32 = sparse_tensor.indices %h { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %i30 = sparse_tensor.indices %i { dimension = 0 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %i31 = sparse_tensor.indices %i { dimension = 1 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>
    %i32 = sparse_tensor.indices %i { dimension = 2 : index } : tensor<2x3x4xf64, #Tensor3> to memref<?xindex>

    call @dumpidx(%v10) : (memref<?xindex>) -> ()
    call @dumpidx(%v11) : (memref<?xindex>) -> ()
    call @dumpidx(%v12) : (memref<?xindex>) -> ()
    call @dumpidx(%v20) : (memref<?xindex>) -> ()
    call @dumpidx(%v21) : (memref<?xindex>) -> ()
    call @dumpidx(%v22) : (memref<?xindex>) -> ()
    call @dumpidx(%v30) : (memref<?xindex>) -> ()
    call @dumpidx(%v31) : (memref<?xindex>) -> ()
    call @dumpidx(%v32) : (memref<?xindex>) -> ()

    call @dumpidx(%a10) : (memref<?xindex>) -> ()
    call @dumpidx(%a11) : (memref<?xindex>) -> ()
    call @dumpidx(%a12) : (memref<?xindex>) -> ()
    call @dumpidx(%b10) : (memref<?xindex>) -> ()
    call @dumpidx(%b11) : (memref<?xindex>) -> ()
    call @dumpidx(%b12) : (memref<?xindex>) -> ()
    call @dumpidx(%c10) : (memref<?xindex>) -> ()
    call @dumpidx(%c11) : (memref<?xindex>) -> ()
    call @dumpidx(%c12) : (memref<?xindex>) -> ()

    call @dumpidx(%d20) : (memref<?xindex>) -> ()
    call @dumpidx(%d21) : (memref<?xindex>) -> ()
    call @dumpidx(%d22) : (memref<?xindex>) -> ()
    call @dumpidx(%e20) : (memref<?xindex>) -> ()
    call @dumpidx(%e21) : (memref<?xindex>) -> ()
    call @dumpidx(%e22) : (memref<?xindex>) -> ()
    call @dumpidx(%f20) : (memref<?xindex>) -> ()
    call @dumpidx(%f21) : (memref<?xindex>) -> ()
    call @dumpidx(%f22) : (memref<?xindex>) -> ()

    call @dumpidx(%g30) : (memref<?xindex>) -> ()
    call @dumpidx(%g31) : (memref<?xindex>) -> ()
    call @dumpidx(%g32) : (memref<?xindex>) -> ()
    call @dumpidx(%h30) : (memref<?xindex>) -> ()
    call @dumpidx(%h31) : (memref<?xindex>) -> ()
    call @dumpidx(%h32) : (memref<?xindex>) -> ()
    call @dumpidx(%i30) : (memref<?xindex>) -> ()
    call @dumpidx(%i31) : (memref<?xindex>) -> ()
    call @dumpidx(%i32) : (memref<?xindex>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %1 : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %2 : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %3 : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %b : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %c : tensor<2x3x4xf64, #Tensor1>
    bufferization.dealloc_tensor %d : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %f : tensor<2x3x4xf64, #Tensor2>
    bufferization.dealloc_tensor %g : tensor<2x3x4xf64, #Tensor3>
    bufferization.dealloc_tensor %h : tensor<2x3x4xf64, #Tensor3>

    return
  }
}
