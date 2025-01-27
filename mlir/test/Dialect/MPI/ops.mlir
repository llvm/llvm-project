// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func.func @mpi_test(%ref : memref<100xf32>) -> () {
    // Note: the !mpi.retval result is optional on all operations except mpi.error_class

    // CHECK: %0 = mpi.init : !mpi.retval
    %err = mpi.init : !mpi.retval

    // CHECK-NEXT: %retval, %rank = mpi.comm_rank : !mpi.retval, i32
    %retval, %rank = mpi.comm_rank : !mpi.retval, i32

    // CHECK-NEXT: %retval2, %size = mpi.comm_size : !mpi.retval, i32
    %retval2, %size = mpi.comm_size : !mpi.retval, i32

    // CHECK-NEXT: mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32
    mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: %1 = mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval
    %err2 = mpi.send(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32
    mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: %2 = mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval
    %err3 = mpi.recv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: mpi.isend(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request
    %req1 = mpi.isend(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request

    // CHECK-NEXT: %1 = mpi.isend(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval, mpi.request
    %err2, %req2 = mpi.isend(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval, mpi.request

    // CHECK-NEXT: mpi.irecv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> mpi.request
    %req3 = mpi.irecv(%ref, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: %2 = mpi.irecv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval, mpi.request
    %err3, %req4 = mpi.irecv(%ref, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK-NEXT: mpi.wait(%req1) : mpi.request
    mpi.wait(%req1) : mpi.request

    // CHECK-NEXT: %3 = mpi.wait(%req1) : mpi.request -> !mpi.retval
    %err4 = mpi.wait(%req1) : mpi.request -> !mpi.retval

    // CHECK-NEXT: mpi.barrier : !mpi.retval
    mpi.barrier : !mpi.retval

    // CHECK-NEXT: %3 = mpi.finalize : !mpi.retval
    %rval = mpi.finalize : !mpi.retval

    // CHECK-NEXT: %4 = mpi.retval_check %retval = <MPI_SUCCESS> : i1
    %res = mpi.retval_check %retval = <MPI_SUCCESS> : i1

    // CHECK-NEXT: %5 = mpi.error_class %0 : !mpi.retval
    %errclass = mpi.error_class %err : !mpi.retval

    // CHECK-NEXT: return
    func.return
}
