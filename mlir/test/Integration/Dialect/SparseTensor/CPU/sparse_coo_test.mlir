// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ]
}>

#trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) + B(i,j)"
}

module {
  func.func @add_coo_csr(%arga: tensor<8x8xf32, #CSR>,
                         %argb: tensor<8x8xf32, #SortedCOO>)
		         -> tensor<8x8xf32> {
    %empty = tensor.empty() : tensor<8x8xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %init = linalg.fill
        ins(%zero : f32)
        outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
    %0 = linalg.generic #trait
      ins(%arga, %argb: tensor<8x8xf32, #CSR>,
                        tensor<8x8xf32, #SortedCOO>)
      outs(%init: tensor<8x8xf32>) {
        ^bb(%a: f32, %b: f32, %x: f32):
          %0 = arith.addf %a, %b : f32
          linalg.yield %0 : f32
        } -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  func.func @add_coo_coo(%arga: tensor<8x8xf32, #SortedCOO>,
                         %argb: tensor<8x8xf32, #SortedCOO>)
		         -> tensor<8x8xf32> {
    %empty = tensor.empty() : tensor<8x8xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %init = linalg.fill
        ins(%zero : f32)
        outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
    %0 = linalg.generic #trait
      ins(%arga, %argb: tensor<8x8xf32, #SortedCOO>,
                        tensor<8x8xf32, #SortedCOO>)
      outs(%init: tensor<8x8xf32>) {
        ^bb(%a: f32, %b: f32, %x: f32):
          %0 = arith.addf %a, %b : f32
          linalg.yield %0 : f32
        } -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  func.func @add_coo_dense(%arga: tensor<8x8xf32>,
                           %argb: tensor<8x8xf32, #SortedCOO>)
  	    	         -> tensor<8x8xf32> {
    %empty = tensor.empty() : tensor<8x8xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %init = linalg.fill
        ins(%zero : f32)
        outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
    %0 = linalg.generic #trait
      ins(%arga, %argb: tensor<8x8xf32>,
                        tensor<8x8xf32, #SortedCOO>)
      outs(%init: tensor<8x8xf32>) {
        ^bb(%a: f32, %b: f32, %x: f32):
          %0 = arith.addf %a, %b : f32
          linalg.yield %0 : f32
        } -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  func.func @entry() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c8  = arith.constant 8 : index

    %A = arith.constant dense<
        [ [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ],
          [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
          [ 2.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
          [ 3.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
          [ 4.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ],
          [ 5.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 ],
          [ 6.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6 ],
          [ 7.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7 ] ]
    > : tensor<8x8xf32>
    %B = arith.constant dense<
        [ [ 7.8, 2.8, 3.8, 0.8, 3.8, 0.1, 7.8, 8.8 ],
          [ 3.3, 2.3, 1.3, 4.3, 3.3, 6.3, 9.3, 8.3 ],
          [ 6.6, 2.6, 3.6, 4.6, 3.6, 6.6, 7.6, 7.6 ],
          [ 1.0, 3.0, 3.0, 4.0, 3.0, 6.0, 7.0, 8.0 ],
          [ 0.1, 2.1, 3.1, 4.1, 3.1, 6.1, 7.1, 8.1 ],
          [ 4.4, 2.4, 3.4, 4.4, 3.4, 6.4, 8.4, 8.4 ],
          [ 5.5, 3.5, 1.5, 4.5, 3.5, 6.5, 7.5, 8.5 ],
          [ 7.7, 2.7, 3.7, 0.7, 5.7, 3.7, 3.7, 0.7 ] ]
    > : tensor<8x8xf32>

    // Stress test with a "sparse" version of A and B.
    %CSR_A = sparse_tensor.convert %A
      : tensor<8x8xf32> to tensor<8x8xf32, #CSR>
    %COO_A = sparse_tensor.convert %A
      : tensor<8x8xf32> to tensor<8x8xf32, #SortedCOO>
    %COO_B = sparse_tensor.convert %B
      : tensor<8x8xf32> to tensor<8x8xf32, #SortedCOO>

    %C1 = call @add_coo_dense(%A, %COO_B) : (tensor<8x8xf32>,
                                             tensor<8x8xf32, #SortedCOO>)
                                          -> tensor<8x8xf32>
    %C2 = call @add_coo_csr(%CSR_A, %COO_B) : (tensor<8x8xf32, #CSR>,
                                               tensor<8x8xf32, #SortedCOO>)
                                            -> tensor<8x8xf32>
    %C3 = call @add_coo_coo(%COO_A, %COO_B) : (tensor<8x8xf32, #SortedCOO>,
                                               tensor<8x8xf32, #SortedCOO>)
                                            -> tensor<8x8xf32>
    //
    // Verify computed matrix C.
    //
    // CHECK-COUNT-3:      ( 8.8, 4.8, 6.8, 4.8, 8.8, 6.1, 14.8, 16.8 )
    // CHECK-NEXT-COUNT-3: ( 4.4, 4.4, 4.4, 8.4, 8.4, 12.4, 16.4, 16.4 )
    // CHECK-NEXT-COUNT-3: ( 8.8, 4.8, 6.8, 8.8, 8.8, 12.8, 14.8, 15.8 )
    // CHECK-NEXT-COUNT-3: ( 4.3, 5.3, 6.3, 8.3, 8.3, 12.3, 14.3, 16.3 )
    // CHECK-NEXT-COUNT-3: ( 4.5, 4.5, 6.5, 8.5, 8.5, 12.5, 14.5, 16.5 )
    // CHECK-NEXT-COUNT-3: ( 9.9, 4.9, 6.9, 8.9, 8.9, 12.9, 15.9, 16.9 )
    // CHECK-NEXT-COUNT-3: ( 12.1, 6.1, 5.1, 9.1, 9.1, 13.1, 15.1, 17.1 )
    // CHECK-NEXT-COUNT-3: ( 15.4, 5.4, 7.4, 5.4, 11.4, 10.4, 11.4, 9.4 )
    //
    %f0  = arith.constant 0.0 : f32
    scf.for %i = %c0 to %c8 step %c1 {
      %v1 = vector.transfer_read %C1[%i, %c0], %f0
        : tensor<8x8xf32>, vector<8xf32>
      %v2 = vector.transfer_read %C2[%i, %c0], %f0
        : tensor<8x8xf32>, vector<8xf32>
      %v3 = vector.transfer_read %C3[%i, %c0], %f0
        : tensor<8x8xf32>, vector<8xf32>
      vector.print %v1 : vector<8xf32>
      vector.print %v2 : vector<8xf32>
      vector.print %v3 : vector<8xf32>
    }

    // Release resources.
    bufferization.dealloc_tensor %C1 : tensor<8x8xf32>
    bufferization.dealloc_tensor %C2 : tensor<8x8xf32>
    bufferization.dealloc_tensor %C3 : tensor<8x8xf32>
    bufferization.dealloc_tensor %CSR_A : tensor<8x8xf32, #CSR>
    bufferization.dealloc_tensor %COO_A : tensor<8x8xf32, #SortedCOO>
    bufferization.dealloc_tensor %COO_B : tensor<8x8xf32, #SortedCOO>


    return
  }
}
