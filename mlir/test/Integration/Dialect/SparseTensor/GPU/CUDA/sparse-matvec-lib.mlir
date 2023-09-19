//
// NOTE: this test requires gpu-sm80
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --sparse-compiler="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:   --shared-libs=%mlir_cuda_runtime \
// DEFINE:   --shared-libs=%mlir_c_runner_utils \
// DEFINE:   --e main --entry-point-result=void \
// DEFINE: | FileCheck %s
//
// with RT lib (SoA COO):
//
// RUN: %{compile} enable-runtime-library=true"  | %{run}
// RUN: %{compile} enable-runtime-library=true gpu-data-transfer-strategy=pinned-dma" | %{run}
// Tracker #64316
// RUNNOT: %{compile} enable-runtime-library=true gpu-data-transfer-strategy=zero-copy"  | %{run}
//
// without RT lib (AoS COO): note, may fall back to CPU
//
// RUN: %{compile} enable-runtime-library=false"  | %{run}
// RUN: %{compile} enable-runtime-library=false gpu-data-transfer-strategy=pinned-dma" | %{run}
// Tracker #64316
// RUNNOT: %{compile} enable-runtime-library=false gpu-data-transfer-strategy=zero-copy"  | %{run}
//

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

module {
  llvm.func @mgpuCreateSparseEnv()
  llvm.func @mgpuDestroySparseEnv()

  // Compute matrix vector y = Ax on COO with default index coordinates.
  func.func @matvecCOO(%A: tensor<?x?xf64, #SortedCOO>, %x: tensor<?xf64>, %y_in: tensor<?xf64>) -> tensor<?xf64> {
    %y_out = linalg.matvec
      ins(%A, %x: tensor<?x?xf64, #SortedCOO>, tensor<?xf64>)
      outs(%y_in: tensor<?xf64>) -> tensor<?xf64>
    return %y_out : tensor<?xf64>
  }

  // Compute matrix vector y = Ax on CSR with 32-bit positions and coordinates.
  func.func @matvecCSR(%A: tensor<?x?xf64, #CSR>, %x: tensor<?xf64>, %y_in: tensor<?xf64>) -> tensor<?xf64> {
    %y_out = linalg.matvec
      ins(%A, %x: tensor<?x?xf64, #CSR>, tensor<?xf64>)
      outs(%y_in: tensor<?xf64>) -> tensor<?xf64>
    return %y_out : tensor<?xf64>
  }

  func.func @main() {
    llvm.call @mgpuCreateSparseEnv() : () -> ()
    %f0 = arith.constant 0.0 : f64
    %f1 = arith.constant 1.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Stress test with a dense matrix DA.
    %DA = tensor.generate {
    ^bb0(%i: index, %j: index):
      %k = arith.addi %i, %j : index
      %l = arith.index_cast %k : index to i64
      %f = arith.uitofp %l : i64 to f64
      tensor.yield %f : f64
    } : tensor<64x64xf64>

    // Convert to a "sparse" m x n matrix A.
    %Acoo = sparse_tensor.convert %DA : tensor<64x64xf64> to tensor<?x?xf64, #SortedCOO>
    %Acsr = sparse_tensor.convert %DA : tensor<64x64xf64> to tensor<?x?xf64, #CSR>

    // Initialize dense vector with n elements:
    //   (1, 2, 3, 4, ..., n)
    %d1 = tensor.dim %Acoo, %c1 : tensor<?x?xf64, #SortedCOO>
    %x = tensor.generate %d1 {
    ^bb0(%i : index):
      %k = arith.addi %i, %c1 : index
      %j = arith.index_cast %k : index to i64
      %f = arith.uitofp %j : i64 to f64
      tensor.yield %f : f64
    } : tensor<?xf64>

    // Initialize dense vectors to m zeros and m ones.
    %d0 = tensor.dim %Acoo, %c0 : tensor<?x?xf64, #SortedCOO>
    %y0 = tensor.generate %d0 {
    ^bb0(%i : index):
      tensor.yield %f0 : f64
    } : tensor<?xf64>
    %y1 = tensor.generate %d0 {
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
    %2 = call @matvecCOO(%Acoo, %x, %y1) : (tensor<?x?xf64, #SortedCOO>,
                                            tensor<?xf64>,
					    tensor<?xf64>) -> tensor<?xf64>
    %3 = call @matvecCSR(%Acsr, %x, %y1) : (tensor<?x?xf64, #CSR>,
                                            tensor<?xf64>,
					    tensor<?xf64>) -> tensor<?xf64>

    //
    // Sanity check on the results.
    //
    // CHECK-COUNT-2: ( 87360, 89440, 91520, 93600, 95680, 97760, 99840, 101920, 104000, 106080, 108160, 110240, 112320, 114400, 116480, 118560, 120640, 122720, 124800, 126880, 128960, 131040, 133120, 135200, 137280, 139360, 141440, 143520, 145600, 147680, 149760, 151840, 153920, 156000, 158080, 160160, 162240, 164320, 166400, 168480, 170560, 172640, 174720, 176800, 178880, 180960, 183040, 185120, 187200, 189280, 191360, 193440, 195520, 197600, 199680, 201760, 203840, 205920, 208000, 210080, 212160, 214240, 216320, 218400 )
    //
    // CHECK-COUNT-2: ( 87361, 89441, 91521, 93601, 95681, 97761, 99841, 101921, 104001, 106081, 108161, 110241, 112321, 114401, 116481, 118561, 120641, 122721, 124801, 126881, 128961, 131041, 133121, 135201, 137281, 139361, 141441, 143521, 145601, 147681, 149761, 151841, 153921, 156001, 158081, 160161, 162241, 164321, 166401, 168481, 170561, 172641, 174721, 176801, 178881, 180961, 183041, 185121, 187201, 189281, 191361, 193441, 195521, 197601, 199681, 201761, 203841, 205921, 208001, 210081, 212161, 214241, 216321, 218401 )
    //
    %pb0 = vector.transfer_read %0[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb0 : vector<64xf64>
    %pb1 = vector.transfer_read %1[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb1 : vector<64xf64>
    %pb2 = vector.transfer_read %2[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb2 : vector<64xf64>
    %pb3 = vector.transfer_read %3[%c0], %f0 : tensor<?xf64>, vector<64xf64>
    vector.print %pb3 : vector<64xf64>

    // Release the resources.
    bufferization.dealloc_tensor %Acoo : tensor<?x?xf64, #SortedCOO>
    bufferization.dealloc_tensor %Acsr : tensor<?x?xf64, #CSR>

    llvm.call @mgpuDestroySparseEnv() : () -> ()
    return
  }
}
