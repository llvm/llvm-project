// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = enable-runtime-library=false
// RUN: %{command}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"
// RUN: %{command}

#SV = #sparse_tensor.encoding<{ lvlTypes = [ "compressed" ] }>

#trait_reduction = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> ()>    // x (scalar out)
  ],
  iterator_types = ["reduction"],
  doc = "x += MIN_i a(i)"
}

// Examples of sparse vector MIN reductions.
module {

  // Custom MIN reduction: stored i32 elements only.
  func.func @min1(%arga: tensor<32xi32, #SV>, %argx: tensor<i32>) -> tensor<i32> {
    %c = tensor.extract %argx[] : tensor<i32>
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
          %1 = sparse_tensor.reduce %a, %b, %c : i32 {
            ^bb0(%x: i32, %y: i32):
	      %m = arith.minsi %x, %y : i32
              sparse_tensor.yield %m : i32
          }
          linalg.yield %1 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  // Regular MIN reduction: stored i32 elements AND implicit zeros.
  // Note that dealing with the implicit zeros is taken care of
  // by the sparse compiler to preserve semantics of the "original".
  func.func @min2(%arga: tensor<32xi32, #SV>, %argx: tensor<i32>) -> tensor<i32> {
    %c = tensor.extract %argx[] : tensor<i32>
    %0 = linalg.generic #trait_reduction
      ins(%arga: tensor<32xi32, #SV>)
      outs(%argx: tensor<i32>) {
        ^bb(%a: i32, %b: i32):
	  %m = arith.minsi %a, %b : i32
          linalg.yield %m : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }

  func.func @dump_i32(%arg0 : tensor<i32>) {
    %v = tensor.extract %arg0[] : tensor<i32>
    vector.print %v : i32
    return
  }

  func.func @entry() {
    %ri = arith.constant dense<999> : tensor<i32>

    // Vectors with a few zeros.
    %c_0_i32 = arith.constant dense<[
      2, 2, 7, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 3, 0, 9, 2, 2, 2, 2, 0, 5, 1, 7, 3
    ]> : tensor<32xi32>

    // Vectors with no zeros.
    %c_1_i32 = arith.constant dense<[
      2, 2, 7, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2,
      2, 2, 2, 2, 3, 2, 7, 2, 2, 2, 2, 2, 2, 1, 7, 3
    ]> : tensor<32xi32>

    // Convert constants to annotated tensors. Note that this
    // particular conversion only stores nonzero elements,
    // so we will have no explicit zeros, only implicit zeros.
    %sv0 = sparse_tensor.convert %c_0_i32
      : tensor<32xi32> to tensor<32xi32, #SV>
    %sv1 = sparse_tensor.convert %c_1_i32
      : tensor<32xi32> to tensor<32xi32, #SV>

    // Special case, construct a sparse vector with an explicit zero.
    %v = arith.constant sparse< [ [1], [7] ], [ 0, 22 ] > : tensor<32xi32>
    %sv2 = sparse_tensor.convert %v: tensor<32xi32> to tensor<32xi32, #SV>

    // Call the kernels.
    %0 = call @min1(%sv0, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %1 = call @min1(%sv1, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %2 = call @min1(%sv2, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %3 = call @min2(%sv0, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %4 = call @min2(%sv1, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>
    %5 = call @min2(%sv2, %ri) : (tensor<32xi32, #SV>, tensor<i32>) -> tensor<i32>

    // Verify results.
    //
    // CHECK: 1
    // CHECK: 1
    // CHECK: 0
    // CHECK: 0
    // CHECK: 1
    // CHECK: 0
    //
    call @dump_i32(%0) : (tensor<i32>) -> ()
    call @dump_i32(%1) : (tensor<i32>) -> ()
    call @dump_i32(%2) : (tensor<i32>) -> ()
    call @dump_i32(%3) : (tensor<i32>) -> ()
    call @dump_i32(%4) : (tensor<i32>) -> ()
    call @dump_i32(%5) : (tensor<i32>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv0 : tensor<32xi32, #SV>
    bufferization.dealloc_tensor %sv1 : tensor<32xi32, #SV>
    bufferization.dealloc_tensor %sv2 : tensor<32xi32, #SV>

    return
  }
}
