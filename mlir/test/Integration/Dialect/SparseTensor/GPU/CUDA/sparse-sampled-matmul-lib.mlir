// NOTE: this test requires gpu-sm80
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --sparsifier="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = \
// DEFINE:   env TENSOR0="%mlir_src_dir/test/Integration/data/test.mtx" \
// DEFINE:   mlir-cpu-runner \
// DEFINE:   --shared-libs=%mlir_cuda_runtime \
// DEFINE:   --shared-libs=%mlir_c_runner_utils \
// DEFINE:   --e main --entry-point-result=void \
// DEFINE: | FileCheck %s
//
// with RT lib:
//
// RUN: %{compile} enable-runtime-library=true"  | %{run}
//
// without RT lib:
//
// RUN: %{compile} enable-runtime-library=false" | %{run}

!Filename = !llvm.ptr

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#trait_sampled_dense_dense = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // S (in/out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "S(i,j) += spy[S(i,j)] x SUM_k A(i,k) B(k,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes sparse storage schemes, and
// runs the resulting code with the JIT compiler.
//
module {
  llvm.func @mgpuCreateSparseEnv()
  llvm.func @mgpuDestroySparseEnv()

  //
  // A kernel that computes a sampled dense matrix matrix multiplication
  // using a "spy" function and in-place update of the sampling sparse matrix.
  //
  func.func @sampled_dense_dense(%args: tensor<?x?xf32, #CSR>,
                                 %arga: tensor<?x?xf32>,
                                 %argb: tensor<?x?xf32>) -> tensor<?x?xf32, #CSR> {
    %result = linalg.generic #trait_sampled_dense_dense
      ins(%arga, %argb: tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%args: tensor<?x?xf32, #CSR>) {
        ^bb(%a: f32, %b: f32, %s: f32):
           %f0 = arith.constant 0.0 : f32
           %u = sparse_tensor.unary %s : f32 to f32
             present={
                ^bb0(%p: f32):
                  %mul = arith.mulf %a, %b : f32
                  sparse_tensor.yield %mul : f32
             }
             absent={}
           %r = sparse_tensor.reduce %s, %u, %f0 : f32 {
              ^bb0(%p: f32, %q: f32):
                %add = arith.addf %p, %q : f32
                sparse_tensor.yield %add : f32
            }
           linalg.yield %r : f32
      } -> tensor<?x?xf32, #CSR>
    return %result : tensor<?x?xf32, #CSR>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver.
  //
  func.func @main() {
    llvm.call @mgpuCreateSparseEnv() : () -> ()
    %d0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index

    // Initialize dense matrices.
    %a = tensor.generate %c5, %c10 {
    ^bb0(%i: index, %j: index):
      %p = arith.addi %i, %c1 : index
      %q = arith.index_cast %p : index to i32
      %d = arith.sitofp %q : i32 to f32
      tensor.yield %d : f32
    } : tensor<?x?xf32>
    %b = tensor.generate %c10, %c5 {
    ^bb0(%i: index, %j: index):
      %p = arith.addi %j, %c1 : index
      %q = arith.index_cast %p : index to i32
      %d = arith.sitofp %q : i32 to f32
      tensor.yield %d : f32
    } : tensor<?x?xf32>

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %s = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #CSR>

    // Call the kernel.
    %0 = call @sampled_dense_dense(%s, %a, %b)
       : (tensor<?x?xf32, #CSR>,
          tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32, #CSR>

    //
    // Print the result for verification.
    //
    // CHECK:   ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 9
    // CHECK-NEXT: dim = ( 5, 5 )
    // CHECK-NEXT: lvl = ( 5, 5 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 4, 5, 7, 9,
    // CHECK-NEXT: crd[1] : ( 0, 3, 1, 4, 2, 0, 3, 1, 4,
    // CHECK-NEXT: values : ( 11, 41.4, 42, 102.5, 93, 44.1, 164, 105.2, 255,
    // CHECK-NEXT: ----
    sparse_tensor.print %0 : tensor<?x?xf32, #CSR>

    // Create a much sparser sampling matrix.
    %t = arith.constant sparse<[[0,0], [0,1], [1,0], [3,4], [7,7]],
                               [1.0, 2.0, 3.0, 4.0, 5.0]
			      > : tensor<8x8xf32>
    %q = sparse_tensor.convert %t : tensor<8x8xf32> to tensor<?x?xf32, #CSR>
    %a2 = arith.constant dense<2.0> : tensor<8x8xf32>
    %b1 = arith.constant dense<1.0> : tensor<8x8xf32>
    %a2c = tensor.cast %a2 : tensor<8x8xf32> to tensor<?x?xf32>
    %b1c = tensor.cast %b1 : tensor<8x8xf32> to tensor<?x?xf32>

    // Call the kernel again.
    %1 = call @sampled_dense_dense(%q, %a2c, %b1c)
       : (tensor<?x?xf32, #CSR>,
          tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32, #CSR>

    //
    // Print the result for verification.
    //
    // CHECK:     ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 5
    // CHECK-NEXT: dim = ( 8, 8 )
    // CHECK-NEXT: lvl = ( 8, 8 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3, 3, 4, 4, 4, 4, 5,
    // CHECK-NEXT: crd[1] : ( 0, 1, 0, 4, 7,
    // CHECK-NEXT: values : ( 17, 18, 19, 20, 21,
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %1 : tensor<?x?xf32, #CSR>

    // Release the resources.
    bufferization.dealloc_tensor %0 : tensor<?x?xf32, #CSR>
    bufferization.dealloc_tensor %1 : tensor<?x?xf32, #CSR>

    llvm.call @mgpuDestroySparseEnv() : () -> ()
    return
  }
}
