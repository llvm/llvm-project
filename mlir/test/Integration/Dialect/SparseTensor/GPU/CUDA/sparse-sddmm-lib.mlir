// NOTE: this test requires gpu-sm80
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --sparsifier="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = \
// DEFINE:   env TENSOR0="%mlir_src_dir/test/Integration/data/block.mtx" \
// DEFINE:   mlir-runner \
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

#BSR = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i floordiv 2 : dense,
    j floordiv 2 : compressed,
    i mod 2 : dense,
    j mod 2 : dense)
}>

#trait_SDDMM = {
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
  // A kernel that computes a CSR sampled dense matrix matrix multiplication
  // using a "spy" function and in-place update of the sampling sparse matrix.
  //
  func.func @SDDMM(%args: tensor<?x?xf32, #CSR>,
                   %arga: tensor<?x?xf32>,
                   %argb: tensor<?x?xf32>) -> tensor<?x?xf32, #CSR> {
    %result = linalg.generic #trait_SDDMM
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

  //
  // A kernel that computes a BSR sampled dense matrix matrix multiplication
  // using a "spy" function and in-place update of the sampling sparse matrix.
  //
  func.func @SDDMM_block(%args: tensor<?x?xf32, #BSR>,
                         %arga: tensor<?x?xf32>,
                         %argb: tensor<?x?xf32>) -> tensor<?x?xf32, #BSR> {
    %result = linalg.generic #trait_SDDMM
      ins(%arga, %argb: tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%args: tensor<?x?xf32, #BSR>) {
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
      } -> tensor<?x?xf32, #BSR>
    return %result : tensor<?x?xf32, #BSR>
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
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index

    // Initialize dense matrices.
    %a = tensor.generate %c4, %c4 {
    ^bb0(%i: index, %j: index):
      %p = arith.addi %i, %c1 : index
      %q = arith.index_cast %p : index to i32
      %d = arith.sitofp %q : i32 to f32
      tensor.yield %d : f32
    } : tensor<?x?xf32>
    %b = tensor.generate %c4, %c6 {
    ^bb0(%i: index, %j: index):
      %p = arith.addi %j, %c1 : index
      %q = arith.index_cast %p : index to i32
      %d = arith.sitofp %q : i32 to f32
      tensor.yield %d : f32
    } : tensor<?x?xf32>

    // Read the sparse matrix from file, construct sparse storage.
    //
    //      +-----+-----+-----+
    //      | 1 2 | . . | 4 . |
    //      | . 3 | . . | . 5 |
    //      +-----+-----+-----+
    //      | . . | 6 7 | . . |
    //      | . . | 8 . | . . |
    //      +-----+-----+-----+
    //
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %m_csr = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #CSR>
    %m_bsr = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #BSR>

    // Call the kernel.
    %0 = call @SDDMM(%m_csr, %a, %b)
       : (tensor<?x?xf32, #CSR>,
          tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32, #CSR>
    %1 = call @SDDMM_block(%m_bsr, %a, %b)
       : (tensor<?x?xf32, #BSR>,
          tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32, #BSR>

    //
    // Print the result for verification. Note that the "spy" determines what
    // dot products are sampled, but the original contents are added back to
    // the result (which is why the block sparse version has actual results
    // in the original zero positions).
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 8
    // CHECK-NEXT: dim = ( 4, 6 )
    // CHECK-NEXT: lvl = ( 4, 6 )
    // CHECK-NEXT: pos[1] : ( 0, 3, 5, 7, 8 )
    // CHECK-NEXT: crd[1] : ( 0, 1, 4, 1, 5, 2, 3, 2 )
    // CHECK-NEXT: values : ( 5, 10, 24, 19, 53, 42, 55, 56 )
    // CHECK-NEXT: ----
    //
    // CHECK:      ---- Sparse Tensor ----
    // CHECK-NEXT: nse = 12
    // CHECK-NEXT: dim = ( 4, 6 )
    // CHECK-NEXT: lvl = ( 2, 3, 2, 2 )
    // CHECK-NEXT: pos[1] : ( 0, 2, 3 )
    // CHECK-NEXT: crd[1] : ( 0, 2, 1 )
    // CHECK-NEXT: values : ( 5, 10, 8, 19, 24, 24, 40, 53, 42, 55, 56, 64 )
    // CHECK-NEXT: ----
    //
    sparse_tensor.print %0 : tensor<?x?xf32, #CSR>
    sparse_tensor.print %1 : tensor<?x?xf32, #BSR>

    // Release the resources.
    bufferization.dealloc_tensor %0 : tensor<?x?xf32, #CSR>
    bufferization.dealloc_tensor %1 : tensor<?x?xf32, #BSR>

    llvm.call @mgpuDestroySparseEnv() : () -> ()
    return
  }
}
