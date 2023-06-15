//
// NOTE: this test requires gpu-sm80
//
// RUN: mlir-opt %s \
// RUN:   --sparse-compiler="enable-runtime-library=true enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71"  \
// RUN: | TENSOR0="%mlir_src_dir/test/Integration/data/test.mtx" \
// RUN:   mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --e entry --entry-point-result=void \
// RUN: | FileCheck %s
//

!Filename = !llvm.ptr<i8>

#CSR = #sparse_tensor.encoding<{
  lvlTypes = ["dense", "compressed"]
}>

#trait_sampled_dense_dense = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // S (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += S(i,j) SUM_k A(i,k) B(k,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // A kernel that computes a sampled matrix matrix multiplication.
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
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    %d0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index

    // Initialize dense matrices.
    %x = tensor.generate %c5, %c5 {
    ^bb0(%i : index, %j : index):
      tensor.yield %d0 : f32
    } : tensor<?x?xf32>

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

    // Print the result for verification.
    //
    // CHECK: ( 11, 41.4, 42, 102.5, 93, 44.1, 164, 105.2, 255 )
    %vm = sparse_tensor.values %0 : tensor<?x?xf32, #CSR> to memref<?xf32>
    %vv = vector.transfer_read %vm[%c0], %d0 : memref<?xf32>, vector<9xf32>
    vector.print %vv : vector<9xf32>

    // Release the resources.
    bufferization.dealloc_tensor %0 : tensor<?x?xf32, #CSR>

    return
  }
}
