// RUN: mlir-opt %s \
// RUN:   --sparsification --sparse-tensor-conversion \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize  \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm --convert-std-to-llvm | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/wide.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = type !llvm.ptr<i8>

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
  func @kernel_spmm(%arga: tensor<?x?xf64, #SparseMatrix>,
                    %argb: tensor<?x?xf64>,
                    %argx: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = linalg.generic #spmm
      ins(%arga, %argb: tensor<?x?xf64, #SparseMatrix>, tensor<?x?xf64>)
      outs(%argx: tensor<?x?xf64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %0 = mulf %a, %b : f64
        %1 = addf %x, %0 : f64
        linalg.yield %1 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func @entry() {
    %i0 = constant 0.0 : f64
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c4 = constant 4 : index
    %c256 = constant 256 : index

    // Read the sparse matrix from file, construct sparse storage.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %a = sparse_tensor.new %fileName : !Filename to tensor<?x?xf64, #SparseMatrix>

    // Initialize dense vectors.
    %bdata = memref.alloc(%c256, %c4) : memref<?x?xf64>
    %xdata = memref.alloc(%c4, %c4) : memref<?x?xf64>
    scf.for %i = %c0 to %c256 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        %k0 = muli %i, %c4 : index
        %k1 = addi %j, %k0 : index
        %k2 = index_cast %k1 : index to i32
        %k = sitofp %k2 : i32 to f64
        memref.store %k, %bdata[%i, %j] : memref<?x?xf64>
      }
    }
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        memref.store %i0, %xdata[%i, %j] : memref<?x?xf64>
      }
    }
    %b = memref.tensor_load %bdata : memref<?x?xf64>
    %x = memref.tensor_load %xdata : memref<?x?xf64>

    // Call kernel.
    %0 = call @kernel_spmm(%a, %b, %x)
      : (tensor<?x?xf64, #SparseMatrix>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>

    // Print the result for verification.
    //
    // CHECK: ( ( 3548, 3550, 3552, 3554 ), ( 6052, 6053, 6054, 6055 ), ( -56, -63, -70, -77 ), ( -13704, -13709, -13714, -13719 ) )
    //
    %m = memref.buffer_cast %0 : memref<?x?xf64>
    %v = vector.transfer_read %m[%c0, %c0], %i0: memref<?x?xf64>, vector<4x4xf64>
    vector.print %v : vector<4x4xf64>

    // Release the resources.
    memref.dealloc %bdata : memref<?x?xf64>
    memref.dealloc %xdata : memref<?x?xf64>

    return
  }
}
