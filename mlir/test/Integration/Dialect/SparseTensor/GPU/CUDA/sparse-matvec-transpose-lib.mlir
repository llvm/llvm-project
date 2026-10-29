// NOTE: this test requires gpu-sm80
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --sparsifier="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = mlir-runner \
// DEFINE:   --shared-libs=%mlir_cuda_runtime \
// DEFINE:   --shared-libs=%mlir_c_runner_utils \
// DEFINE:   --e main --entry-point-result=void \
// DEFINE: | FileCheck %s
//
// with RT lib (SoA COO):
//
// RUN: %{compile} enable-runtime-library=true"  | %{run}
//
// without RT lib (AoS COO): note, may fall back to CPU
//
// RUN: %{compile} enable-runtime-library=false" | %{run}

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed),
  posWidth = 64,
  crdWidth = 64
}>

#matvec_transpose = {
  indexing_maps = [
    affine_map<(i, j) -> (j, i)>,
    affine_map<(i, j) -> (j)>,
    affine_map<(i, j) -> (i)>
  ],
  iterator_types = ["parallel", "reduction"]
}

module {
  llvm.func @mgpuCreateSparseEnv()
  llvm.func @mgpuDestroySparseEnv()

  // Compute matrix vector y = A^T x on COO with default index coordinates.
  func.func @matvecCOO(%A: tensor<?x?xf64, #SortedCOO>, %x: tensor<?xf64>, %y_in: tensor<?xf64>) -> tensor<?xf64> {
    %y_out = linalg.generic #matvec_transpose
      ins(%A, %x: tensor<?x?xf64, #SortedCOO>, tensor<?xf64>)
      outs(%y_in: tensor<?xf64>) {
    ^bb0(%a: f64, %xval: f64, %yval: f64):
      %product = arith.mulf %a, %xval : f64
      %sum = arith.addf %yval, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?xf64>
    return %y_out : tensor<?xf64>
  }

  // Compute matrix vector y = A^T x on CSR with 32-bit positions and coordinates.
  func.func @matvecCSR(%A: tensor<?x?xf64, #CSR>, %x: tensor<?xf64>, %y_in: tensor<?xf64>) -> tensor<?xf64> {
    %y_out = linalg.generic #matvec_transpose
      ins(%A, %x: tensor<?x?xf64, #CSR>, tensor<?xf64>)
      outs(%y_in: tensor<?xf64>) {
    ^bb0(%a: f64, %xval: f64, %yval: f64):
      %product = arith.mulf %a, %xval : f64
      %sum = arith.addf %yval, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?xf64>
    return %y_out : tensor<?xf64>
  }

  // Compute matrix vector y = A^T x on CSC with 64-bit positions and coordinates.
  func.func @matvecCSC(%A: tensor<?x?xf64, #CSC>, %x: tensor<?xf64>, %y_in: tensor<?xf64>) -> tensor<?xf64> {
    %y_out = linalg.generic #matvec_transpose
      ins(%A, %x: tensor<?x?xf64, #CSC>, tensor<?xf64>)
      outs(%y_in: tensor<?xf64>) {
    ^bb0(%a: f64, %xval: f64, %yval: f64):
      %product = arith.mulf %a, %xval : f64
      %sum = arith.addf %yval, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?xf64>
    return %y_out : tensor<?xf64>
  }

  func.func @main() {
    llvm.call @mgpuCreateSparseEnv() : () -> ()
    %f0 = arith.constant 0.0 : f64
    %f1 = arith.constant 1.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Stress test with a rectangular dense matrix DA.
    %DA = tensor.generate {
    ^bb0(%i: index, %j: index):
      %k = arith.addi %i, %j : index
      %l = arith.index_cast %k : index to i64
      %f = arith.uitofp %l : i64 to f64
      tensor.yield %f : f64
    } : tensor<32x64xf64>

    // Convert to a "sparse" m x n matrix A.
    %Acoo = sparse_tensor.convert %DA : tensor<32x64xf64> to tensor<?x?xf64, #SortedCOO>
    %Acsr = sparse_tensor.convert %DA : tensor<32x64xf64> to tensor<?x?xf64, #CSR>
    %Acsc = sparse_tensor.convert %DA : tensor<32x64xf64> to tensor<?x?xf64, #CSC>

    // Initialize dense vector with m elements:
    //   (1, 2, 3, 4, ..., m)
    %d0 = tensor.dim %Acoo, %c0 : tensor<?x?xf64, #SortedCOO>
    %x = tensor.generate %d0 {
    ^bb0(%i : index):
      %k = arith.addi %i, %c1 : index
      %j = arith.index_cast %k : index to i64
      %f = arith.uitofp %j : i64 to f64
      tensor.yield %f : f64
    } : tensor<?xf64>

    // Initialize dense vectors to n zeros and n ones.
    %d1 = tensor.dim %Acoo, %c1 : tensor<?x?xf64, #SortedCOO>
    %y0 = tensor.generate %d1 {
    ^bb0(%i : index):
      tensor.yield %f0 : f64
    } : tensor<?xf64>
    %y1 = tensor.generate %d1 {
    ^bb0(%i : index):
      tensor.yield %f1 : f64
    } : tensor<?xf64>

    // Call the kernels.
    %0 = call @matvecCOO(%Acoo, %x, %y0) : (tensor<?x?xf64, #SortedCOO>,
                                            tensor<?xf64>,
                                            tensor<?xf64>) -> tensor<?xf64>
    %1 = call @matvecCSR(%Acsr, %x, %y0) : (tensor<?x?xf64, #CSR>,
                                            tensor<?xf64>,
                                            tensor<?xf64>) -> tensor<?xf64>
    %2 = call @matvecCSC(%Acsc, %x, %y0) : (tensor<?x?xf64, #CSC>,
                                            tensor<?xf64>,
                                            tensor<?xf64>) -> tensor<?xf64>
    %3 = call @matvecCOO(%Acoo, %x, %y1) : (tensor<?x?xf64, #SortedCOO>,
                                            tensor<?xf64>,
                                            tensor<?xf64>) -> tensor<?xf64>
    %4 = call @matvecCSR(%Acsr, %x, %y1) : (tensor<?x?xf64, #CSR>,
                                            tensor<?xf64>,
                                            tensor<?xf64>) -> tensor<?xf64>
    %5 = call @matvecCSC(%Acsc, %x, %y1) : (tensor<?x?xf64, #CSC>,
                                            tensor<?xf64>,
                                            tensor<?xf64>) -> tensor<?xf64>

    //
    // Sanity check on the results.
    //
    // CHECK-COUNT-3: ( 10912, 11440, 11968, 12496, 13024, 13552, 14080, 14608, 15136, 15664, 16192, 16720, 17248, 17776, 18304, 18832, 19360, 19888, 20416, 20944, 21472, 22000, 22528, 23056, 23584, 24112, 24640, 25168, 25696, 26224, 26752, 27280, 27808, 28336, 28864, 29392, 29920, 30448, 30976, 31504, 32032, 32560, 33088, 33616, 34144, 34672, 35200, 35728, 36256, 36784, 37312, 37840, 38368, 38896, 39424, 39952, 40480, 41008, 41536, 42064, 42592, 43120, 43648, 44176 )
    //
    // CHECK-COUNT-3: ( 10913, 11441, 11969, 12497, 13025, 13553, 14081, 14609, 15137, 15665, 16193, 16721, 17249, 17777, 18305, 18833, 19361, 19889, 20417, 20945, 21473, 22001, 22529, 23057, 23585, 24113, 24641, 25169, 25697, 26225, 26753, 27281, 27809, 28337, 28865, 29393, 29921, 30449, 30977, 31505, 32033, 32561, 33089, 33617, 34145, 34673, 35201, 35729, 36257, 36785, 37313, 37841, 38369, 38897, 39425, 39953, 40481, 41009, 41537, 42065, 42593, 43121, 43649, 44177 )
    //
    %pb0 = vector.transfer_read %0[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb0 : vector<64xf64>
    %pb1 = vector.transfer_read %1[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb1 : vector<64xf64>
    %pb2 = vector.transfer_read %2[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb2 : vector<64xf64>
    %pb3 = vector.transfer_read %3[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb3 : vector<64xf64>
    %pb4 = vector.transfer_read %4[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb4 : vector<64xf64>
    %pb5 = vector.transfer_read %5[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb5 : vector<64xf64>

    // Release the resources.
    bufferization.dealloc_tensor %Acoo : tensor<?x?xf64, #SortedCOO>
    bufferization.dealloc_tensor %Acsr : tensor<?x?xf64, #CSR>
    bufferization.dealloc_tensor %Acsc : tensor<?x?xf64, #CSC>

    llvm.call @mgpuDestroySparseEnv() : () -> ()
    return
  }
}
