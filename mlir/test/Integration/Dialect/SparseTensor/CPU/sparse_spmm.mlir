// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = TENSOR0="%mlir_src_dir/test/Integration/data/wide.mtx" \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
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
// REDEFINE: %{run} = TENSOR0="%mlir_src_dir/test/Integration/data/wide.mtx" \
// REDEFINE: lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

!Filename = !llvm.ptr<i8>

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

#spmm = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // A kernel that multiplies a sparse matrix A with a dense matrix B
  // into a dense matrix X.
  //
  func.func @kernel_spmm(%arga: tensor<?x?xf64, #SparseMatrix>,
                         %argb: tensor<?x?xf64>,
                         %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #spmm
      ins(%arga, %argb: tensor<?x?xf64, #SparseMatrix>, tensor<?x?xf64>)
      outs(%argx: tensor<?x?xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = arith.mulf %a, %b : f64
        %1 = arith.addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func.func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func.func @entry() {
    %i0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #SparseMatrix>

    // Initialize dense tensors.
    %b = tensor.generate %c256, %c4 {
    ^bb0(%i : index, %j : index):
      %k0 = arith.muli %i, %c4 : index
      %k1 = arith.addi %j, %k0 : index
      %k2 = arith.index_cast %k1 : index to i32
      %k = arith.sitofp %k2 : i32 to f64
      tensor.yield %k : f64
    } : tensor<?x?xf64>

    %x = tensor.generate %c4, %c4 {
    ^bb0(%i : index, %j : index):
      tensor.yield %i0 : f64
    } : tensor<?x?xf64>

    // Call kernel.
    %0 = call @kernel_spmm(%a, %b, %x)
      : (tensor<?x?xf64, #SparseMatrix>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>

    // Print the result for verification.
    //
    // CHECK: ( ( 3548, 3550, 3552, 3554 ), ( 6052, 6053, 6054, 6055 ), ( -56, -63, -70, -77 ), ( -13704, -13709, -13714, -13719 ) )
    //
    %v = vector.transfer_read %0[%c0, %c0], %i0: tensor<?x?xf64>, vector<4x4xf64>
    vector.print %v : vector<4x4xf64>

    // Release the resources.
    bufferization.dealloc_tensor %a : tensor<?x?xf64, #SparseMatrix>


    return
  }
}
