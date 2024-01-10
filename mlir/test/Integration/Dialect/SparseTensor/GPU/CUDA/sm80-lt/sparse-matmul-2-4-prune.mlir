// NOTE: this test requires gpu-sm80 and cusparselt
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --sparsifier="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=%gpu_compilation_format
// DEFINE: %{run} = mlir-cpu-runner \
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

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i            : dense,
    j floordiv 4 : dense,
    j mod 4      : block2_4
  )
}>

module {

  llvm.func @mgpuCreateSparseLtEnv()
  llvm.func @mgpuDestroySparseLtEnv()

  func.func @matmul24(%Ad: tensor<16x16xf16>,
                      %B: tensor<16x16xf16>,
                      %Cin: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %A = sparse_tensor.convert %Ad : tensor<16x16xf16> to tensor<16x16xf16, #NV_24>
    %C = linalg.matmul
      ins(%A, %B: tensor<16x16xf16, #NV_24>, tensor<16x16xf16>)
      outs(%Cin: tensor<16x16xf16>) -> tensor<16x16xf16>
    return %C : tensor<16x16xf16>
  }

  func.func @main() {
    llvm.call @mgpuCreateSparseLtEnv() : () -> ()

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %f0 = arith.constant 0.0 : f16
    %f1 = arith.constant 1.0 : f16
    %f4 = arith.constant 4.0 : f16

    // Initial A, B, C matrices.
    %A = tensor.generate {
    ^bb0(%i: index, %j: index):
      %val = arith.andi %j, %c1 : index
      %cmp = arith.cmpi eq, %val, %c0 : index
      %res = arith.select %cmp, %f4, %f1 : f16
      tensor.yield %res : f16
    } : tensor<16x16xf16>
    %B = tensor.generate {
    ^bb0(%i: index, %j: index):
      %cmp = arith.cmpi eq, %i, %j : index
      %res = arith.select %cmp, %f1, %f0 : f16
      tensor.yield %res : f16
    } : tensor<16x16xf16>
    %C = tensor.generate {
    ^bb0(%i: index, %j: index):
      tensor.yield %f0 : f16
    } : tensor<16x16xf16>

    // Call the kernel.
    //
    // By effectively computing D = A B + C with id(B) and zero(C)
    // the resulting matrix returns the pruned A back to the caller.
    //
    %D = call @matmul24(%A, %B, %C): (tensor<16x16xf16>,
                                      tensor<16x16xf16>,
                                      tensor<16x16xf16>) -> (tensor<16x16xf16>)

    //
    // This was the original matrix.
    //
    // CHECK:      ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    // CHECK-NEXT: ( 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1 )
    //
    scf.for %i = %c0 to %c16 step %c1 {
      %va = vector.transfer_read %A[%i, %c0], %f0 : tensor<16x16xf16>, vector<16xf16>
      vector.print %va : vector<16xf16>
    }

    //
    // This is the STRIP-pruned matrix.
    //
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    // CHECK-NEXT: ( 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0 )
    //
    scf.for %i = %c0 to %c16 step %c1 {
      %vd = vector.transfer_read %D[%i, %c0], %f0 : tensor<16x16xf16>, vector<16xf16>
      vector.print %vd : vector<16xf16>
    }

    llvm.call @mgpuDestroySparseLtEnv() : () -> ()
    return
  }
}
