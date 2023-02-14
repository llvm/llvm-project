// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = TENSOR0="%mlir_src_dir/test/Integration/data/mttkrp_b.tns"  \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext | \
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

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4  enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = TENSOR0="%mlir_src_dir/test/Integration/data/mttkrp_b.tns" \
// REDEFINE: %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext -dlopen=%mlir_lib_dir/libmlir_runner_utils%shlibext| \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

!Filename = !llvm.ptr<i8>

#SparseTensor = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ]
}>

#mttkrp = {
  indexing_maps = [
    affine_map<(i,j,k,l) -> (i,k,l)>, // B
    affine_map<(i,j,k,l) -> (k,j)>,   // C
    affine_map<(i,j,k,l) -> (l,j)>,   // D
    affine_map<(i,j,k,l) -> (i,j)>    // A (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction", "reduction"],
  doc = "A(i,j) += B(i,k,l) * D(l,j) * C(k,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  func.func private @printMemrefF64(%ptr : tensor<*xf64>)

  //
  // Computes Matricized Tensor Times Khatri-Rao Product (MTTKRP) kernel. See
  // http://tensor-compiler.org/docs/data_analytics/index.html.
  //
  func.func @kernel_mttkrp(%argb: tensor<?x?x?xf64, #SparseTensor>,
                           %argc: tensor<?x?xf64>,
                           %argd: tensor<?x?xf64>,
                           %arga: tensor<?x?xf64>)
                               -> tensor<?x?xf64> {
    %0 = linalg.generic #mttkrp
      ins(%argb, %argc, %argd:
            tensor<?x?x?xf64, #SparseTensor>, tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%arga: tensor<?x?xf64>) {
      ^bb(%b: f64, %c: f64, %d: f64, %a: f64):
        %0 = arith.mulf %b, %c : f64
        %1 = arith.mulf %d, %0 : f64
        %2 = arith.addf %a, %1 : f64
        linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    %f0 = arith.constant 0.0 : f64
    %cst0 = arith.constant 0 : index
    %cst1 = arith.constant 1 : index
    %cst2 = arith.constant 2 : index

    // Read the sparse input tensor B from a file.
    %fileName = call @getTensorFilename(%cst0) : (index) -> (!Filename)
    %b = sparse_tensor.new %fileName
          : !Filename to tensor<?x?x?xf64, #SparseTensor>

    // Get sizes from B, pick a fixed size for dim-2 of A.
    %isz = tensor.dim %b, %cst0 : tensor<?x?x?xf64, #SparseTensor>
    %jsz = arith.constant 5 : index
    %ksz = tensor.dim %b, %cst1 : tensor<?x?x?xf64, #SparseTensor>
    %lsz = tensor.dim %b, %cst2 : tensor<?x?x?xf64, #SparseTensor>

    // Initialize dense input matrix C.
    %c = tensor.generate %ksz, %jsz {
    ^bb0(%k : index, %j : index):
      %k0 = arith.muli %k, %jsz : index
      %k1 = arith.addi %k0, %j : index
      %k2 = arith.index_cast %k1 : index to i32
      %kf = arith.sitofp %k2 : i32 to f64
      tensor.yield %kf : f64
    } : tensor<?x?xf64>

    // Initialize dense input matrix D.
    %d = tensor.generate %lsz, %jsz {
    ^bb0(%l : index, %j : index):
      %k0 = arith.muli %l, %jsz : index
      %k1 = arith.addi %k0, %j : index
      %k2 = arith.index_cast %k1 : index to i32
      %kf = arith.sitofp %k2 : i32 to f64
      tensor.yield %kf : f64
    } : tensor<?x?xf64>

    // Initialize dense output matrix A.
    %a = tensor.generate %isz, %jsz {
    ^bb0(%i : index, %j: index):
      tensor.yield %f0 : f64
    } : tensor<?x?xf64>

    // Call kernel.
    %0 = call @kernel_mttkrp(%b, %c, %d, %a)
      : (tensor<?x?x?xf64, #SparseTensor>,
        tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>

    // Print the result for verification.
    //
    // CHECK:      {{\[}}[16075,   21930,   28505,   35800,   43815],
    // CHECK-NEXT: [10000,   14225,   19180,   24865,   31280]]
    //
    %u = tensor.cast %0: tensor<?x?xf64> to tensor<*xf64>
    call @printMemrefF64(%u) : (tensor<*xf64>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %b : tensor<?x?x?xf64, #SparseTensor>

    return
  }
}
