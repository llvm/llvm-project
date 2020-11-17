// RUN: mlir-opt %s \
// RUN:  -convert-scf-to-std -convert-vector-to-scf \
// RUN:  -convert-linalg-to-llvm -convert-vector-to-llvm | \
// RUN: MATRIX0="%mlir_integration_test_dir/data/test.mtx" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

module {
  func private @openMatrix(!llvm.ptr<i8>, memref<index>, memref<index>, memref<index>) -> ()
  func private @readMatrixItem(memref<index>, memref<index>, memref<f64>) -> ()
  func private @closeMatrix() -> ()
  func private @getMatrix(index) -> (!llvm.ptr<i8>)

  func @entry() {
    %d0  = constant 0.0 : f64
    %c0  = constant 0 : index
    %c1  = constant 1 : index
    %c5  = constant 5 : index
    %m   = alloc() : memref<index>
    %n   = alloc() : memref<index>
    %nnz = alloc() : memref<index>
    %i   = alloc() : memref<index>
    %j   = alloc() : memref<index>
    %d   = alloc() : memref<f64>

    //
    // Read the header of a sparse matrix. This yields the
    // size (m x n) and number of nonzero elements (nnz).
    //
    %file = call @getMatrix(%c0) : (index) -> (!llvm.ptr<i8>)
    call @openMatrix(%file, %m, %n, %nnz)
        : (!llvm.ptr<i8>, memref<index>,
	                  memref<index>, memref<index>) -> ()
    %M = load %m[]   : memref<index>
    %N = load %n[]   : memref<index>
    %Z = load %nnz[] : memref<index>

    //
    // At this point, code should prepare a proper sparse storage
    // scheme for an m x n matrix with nnz nonzero elements. For
    // simplicity, however, here we simply set up a dense matrix.
    //
    %a = alloc(%M, %N) : memref<?x?xf64>
    scf.for %ii = %c0 to %M step %c1 {
      scf.for %jj = %c0 to %N step %c1 {
        store %d0, %a[%ii, %jj] : memref<?x?xf64>
      }
    }

    //
    // Now we are ready to read in the nonzero elements of the
    // sparse matrix and insert these into a sparse storage
    // scheme. In this example, we simply insert them in the
    // dense matrix.
    //
    scf.for %k = %c0 to %Z step %c1 {
      call @readMatrixItem(%i, %j, %d)
          : (memref<index>, memref<index>, memref<f64>) -> ()
      %I = load %i[] : memref<index>
      %J = load %j[] : memref<index>
      %D = load %d[] : memref<f64>
      store %D, %a[%I, %J] : memref<?x?xf64>
    }
    call @closeMatrix() : () -> ()

    //
    // Verify that the results are as expected.
    //
    %A = vector.transfer_read %a[%c0, %c0], %d0 : memref<?x?xf64>, vector<5x5xf64>
    vector.print %M : index
    vector.print %N : index
    vector.print %Z : index
    vector.print %A : vector<5x5xf64>
    //
    // CHECK: 5
    // CHECK: 5
    // CHECK: 9
    //
    // CHECK: ( ( 1, 0, 0, 1.4, 0 ),
    // CHECK-SAME: ( 0, 2, 0, 0, 2.5 ),
    // CHECK-SAME: ( 0, 0, 3, 0, 0 ),
    // CHECK-SAME: ( 4.1, 0, 0, 4, 0 ),
    // CHECK-SAME: ( 0, 5.2, 0, 0, 5 ) )

    //
    // Free.
    //
    dealloc %m   : memref<index>
    dealloc %n   : memref<index>
    dealloc %nnz : memref<index>
    dealloc %i   : memref<index>
    dealloc %j   : memref<index>
    dealloc %d   : memref<f64>
    dealloc %a   : memref<?x?xf64>

    return
  }
}
